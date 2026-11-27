"""A small value network: state -> expected result for a seat, in [-1, 1].

Inference is plain numpy, so game-playing machines don't need torch; training is in
`yomibench.train` (torch). A model is one .npz file: weights, the feature standardization and the
vocabulary, plus a small JSON header.

    from yomibench.valuenet import ValueNet
    net = ValueNet.load("checkpoints/v0.npz")
    MatrixBot(value=net)          # net(state, seat) -> float

Optional parts of a model, read from its params or meta (models without them load and score as
before):

- An auxiliary head (`w3_aux`, `b3_aux`; meta `aux_hp_weight` > 0), trained on the final hp
  margin to shape the shared layers. Play never reads it: `batch` and `forward` use only the
  value head; `hp_margin` exposes the aux head for diagnostics.
- `residual_hp: true`: the network's antisymmetric output is a correction to the hp-lead
  evaluator, value = clip(evaluate(state, seat) + net, -1, 1), with `search.evaluate`'s scaling
  (each fighter's hp fraction). The hp lead is read from the encoded features (`hp_lead`), which
  the Batch engine's previews carry too, so `batch_play.BatchValue` adds exactly the same term.
- Auxiliary search-value heads (`aux_search_w`, `aux_search_b`; meta `aux_search_heads`, the
  decision offsets they predict, see yomibench.train): training side losses. Play never reads them:
  `forward`'s value head reads only the layers below; `search_heads` exposes them for diagnostics.
- Candidate heads (keys starting `aux_cand_`; meta `aux_cand_value_weight` / `aux_mix_weight`, see
  yomibench.train): training side losses on full turns, from the trunk's last hidden layer and a
  candidate embedding (yomibench.policy's encoding): each candidate's search value, and a logit whose
  softmax over a seat's candidates is the solved mix. Play never reads them (`forward` reads only the
  keys above); `candidate_heads` exposes them for diagnostics.
- More hidden layers (meta `layers` > 2, train --layers): hidden layers w1 w2 wh3 wh4 ... and the
  output layer under w_out b_out instead of w3 b3 (layer_keys), so code from before this option
  fails loudly on such a model (no w3) instead of misreading a hidden layer as the output.

Where the network is evaluated: only here, in numpy (`forward`), for every engine. A batch engine plugin
engine computes the features (feature version, vocabulary, raw defaults: `batch_play.net_key`) and
never sees the weights; `batch_play.BatchValue` runs them through this `forward`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from yomibench import features, features_v2, features_v4
from yomibench.features import Vocab
from yomibench.search import terminal_value

EMBED = {"state": 8, "stance": 4, "char": 4, "kind": 4, "ostate": 4}
FEATURES = 3  # bump when features.encode changes; training writes 3 by default (4 with --inputs-v4)
# Feature versions this code can play: version -> its encoder module (encode, CATEGORY_OF_SLOT).
# Version 2 is checkpoint 1 and older, played with the frozen v2 encoder they were trained with.
# Version 4 (inputs v4.1): version 3's columns then the features_ext block, first layer split
# (w1 reads the v3 part, w1_ext the ext part). Registered by default so any worker/gate can load
# v4 models without extra env vars; zero effect on v2/v3 models.
ENCODERS = {2: features_v2, 3: features, 4: features_v4}
EXT_W1 = "w1_ext"  # version 4's first-layer rows for the ext inputs (numbers, then ext categorical embeddings)


def enable_inputs_v4() -> None:
    ENCODERS[4] = features_v4


V3_PLAYER_FIELDS = ("state_data", "combo_moves_used", "hitboxes")  # `di` stays: training filled it in


def strip_v3_state(state: dict[str, Any]) -> dict[str, Any]:
    """`state` as a pre-v3 engine would send it: no move fields on players, no object hitboxes.
    Shallow copies only; the caller's state is unchanged."""
    out = dict(state)
    out["players"] = [{k: v for k, v in p.items() if k not in V3_PLAYER_FIELDS} for p in state["players"]]
    if "objects" in state:
        out["objects"] = [{k: v for k, v in o.items() if k != "hitboxes"} for o in state["objects"]]
    return out


class ValueNet:
    def __init__(self, params: dict[str, np.ndarray], vocab: Vocab, meta: dict[str, Any]):
        self.p = params
        self.vocab = vocab
        self.vocab.frozen = True
        self.meta = meta
        self.version = meta.get("features", FEATURES)
        self.encoder = ENCODERS[4].encoder_for(meta) if self.version == 4 else ENCODERS[self.version]
        # A v3 model trained on states without the engine-v3 move fields (e.g. sim records from a
        # build that doesn't send them) has only ever seen them off. Real-engine states carry them,
        # which would put its inputs outside training; play it the states it was trained on.
        self.mask_v3_state = self.version in (3, 4) and meta.get("state_v3_fraction", 0.0) < 0.5
        self.residual_hp = bool(meta.get("residual_hp", False))
        self.__name__ = f"valuenet:{meta.get('name', '?')}"

    @staticmethod
    def load(path: str | Path) -> "ValueNet":
        with np.load(path, allow_pickle=False) as f:
            params = {k: f[k] for k in f.files if k != "meta"}
            meta = json.loads(str(f["meta"]))
        if meta.get("features") not in ENCODERS:
            # Inputs are positional, so another version's weights would read the wrong features.
            raise ValueError(f"{path}: model has feature version {meta.get('features')}; this code plays versions "
                             f"{sorted(ENCODERS)} and trains version {FEATURES}")
        vocab = Vocab(ids=meta.pop("vocab"), raw_defaults=meta.pop("raw_defaults", {}))
        return ValueNet(params, vocab, meta)

    def save(self, path: str | Path) -> None:
        meta = {**self.meta, "features": self.version, "vocab": self.vocab.ids, "raw_defaults": self.vocab.raw_defaults}
        np.savez(path, meta=np.array(json.dumps(meta)), **self.p)

    def batch(self, states: list[dict[str, Any]], seat: int) -> np.ndarray:
        """Values for `seat`: (f(seat's view) - f(opponent's view)) / 2, so value(s, 1) is exactly
        -value(s, 0), as the zero-sum search assumes."""
        values = self._both_views(states, seat, "value")
        for k, s in enumerate(states):  # a finished game has an exact value; the net never decides it
            t = terminal_value(s, seat)
            if t is not None:
                values[k] = t
        return values

    def _both_views(self, states: list[dict[str, Any]], seat: int, head: str) -> np.ndarray:
        """`head`'s antisymmetric output for `seat`; the value head with `residual_hp` adds the hp lead."""
        if self.mask_v3_state:
            states = [strip_v3_state(s) for s in states]
        out, mine = [], None
        for view in (seat, 1 - seat):
            encoded = [self.encoder.encode(s, view, self.vocab) for s in states]
            nums = np.array([e[0] for e in encoded], dtype=np.float32)
            cats = np.array([e[1] for e in encoded], dtype=np.int64)
            mine = nums if mine is None else mine
            out.append(forward(self.p, nums, cats, self.encoder.CATEGORY_OF_SLOT, head))
        values = (out[0] - out[1]) / 2
        if head == "value" and self.residual_hp:
            values = residual_value(values, hp_lead(mine, self.encoder))
        return values

    def hp_margin(self, states: list[dict[str, Any]], seat: int) -> np.ndarray:
        """The auxiliary head's predicted final hp margin for `seat` (hp fraction lead, antisymmetric
        like the value). Diagnostics only: play never reads it."""
        if "w3_aux" not in self.p:
            raise ValueError(f"{self.__name__} has no auxiliary hp-margin head")
        return self._both_views(states, seat, "aux")

    def search_heads(self, states: list[dict[str, Any]], seat: int) -> np.ndarray:
        """The auxiliary search-value heads' predictions for `seat` (rows x heads, at the decision
        offsets in meta `aux_search_heads`), antisymmetric like the value. Diagnostics only."""
        if SEARCH_W not in self.p:
            raise ValueError(f"{self.__name__} has no auxiliary search-value heads")
        if self.mask_v3_state:
            states = [strip_v3_state(s) for s in states]
        out = []
        for view in (seat, 1 - seat):
            encoded = [self.encoder.encode(s, view, self.vocab) for s in states]
            out.append(forward(self.p, np.array([e[0] for e in encoded], dtype=np.float32),
                               np.array([e[1] for e in encoded], dtype=np.int64), self.encoder.CATEGORY_OF_SLOT,
                               "search"))
        return (out[0] - out[1]) / 2

    def candidate_heads(self, state: dict[str, Any], seat: int, candidates: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """The candidate heads for `seat`'s `candidates` (records' dicts or engine Candidates) in
        `state`: (each one's predicted search value for `seat`, the predicted mix: a softmax over
        them). Diagnostics only: play never reads these heads."""
        from yomibench.policy import encode_candidate, flip_columns, view_flip

        if f"{CAND}value_w" not in self.p:
            raise ValueError(f"{self.__name__} has no candidate heads")
        if self.mask_v3_state:
            state = strip_v3_state(state)
        enc = [encode_candidate(c if isinstance(c, dict) else c.record()) for c in candidates]
        voc = self.meta.get("aux_cand_vocab", {})
        cact = np.array([[voc.get("action", {}).get(e[0], 0) for e in enc]], np.int64)
        cdkey = np.array([[voc.get("dkey", {}).get(e[1], 0) for e in enc]], np.int64)
        cnums = flip_columns(np.array([[e[2] for e in enc]], np.float32), view_flip(state, seat))
        nums, cats = self.encoder.encode(state, seat, self.vocab)
        h = trunk(self.p, np.array([nums], np.float32), np.array([cats], np.int64), self.encoder.CATEGORY_OF_SLOT)
        value, logit = candidate_heads(self.p, h, cnums, cact, cdkey)
        mix = np.exp(logit[0] - logit[0].max())
        return value[0], mix / mix.sum()

    def batch_encoded(self, nums: np.ndarray, cats: np.ndarray) -> np.ndarray:
        """The raw one-view output f (not antisymmetrized, no residual)."""
        return forward(self.p, nums, cats, self.encoder.CATEGORY_OF_SLOT)

    def value_encoded(self, nums0: np.ndarray, cats0: np.ndarray, nums1: np.ndarray, cats1: np.ndarray,
                      baseline: np.ndarray) -> np.ndarray:
        """p1's value from encoded views, `baseline` being `search.evaluate(state, 0)` (used only
        with `residual_hp`); no terminal override (training states are never terminal)."""
        values = (self.batch_encoded(nums0, cats0) - self.batch_encoded(nums1, cats1)) / 2
        return residual_value(values, baseline) if self.residual_hp else values

    def __call__(self, state: dict[str, Any], seat: int) -> float:
        return float(self.batch([state], seat)[0])


# Training's std is the feature's spread plus 1e-3, so a feature that never varied in training (a
# character the model never saw: every one of its raw fields stayed 0) has std 1e-3 exactly.
CONSTANT_STD = 1e-3 * (1 + 1e-4)


def hp_lead(nums: np.ndarray, encoder: Any = features) -> np.ndarray:
    """`search.evaluate`'s non-terminal score (my hp fraction minus the opponent's) from one view's
    encoded features (rows of `nums`), in float64. Inference features are float32, so this matches
    `evaluate` to about 1e-7."""
    at = encoder.PLAYER_NUMS.index("hp_frac")
    return nums[..., at].astype(np.float64) - nums[..., encoder.PLAYER_WIDTH + at].astype(np.float64)


def residual_value(correction: np.ndarray, lead: np.ndarray) -> np.ndarray:
    """A `residual_hp` model's value: the hp lead plus the net's correction, clipped to [-1, 1]
    (both terms antisymmetric, so the value is too)."""
    return np.clip(lead + correction, -1.0, 1.0)


SEARCH_W, SEARCH_B = "aux_search_w", "aux_search_b"  # auxiliary search-value heads (training side losses)
CAND = "aux_cand_"  # prefix of the candidate heads' keys (training side losses; see candidate_heads)
CAND_LAYERS = ("c1", "c2", "s", "j", "value", "logit")  # their linear layers: aux_cand_<name>_w / _b


def layer_keys(layers: int) -> tuple[list[tuple[str, str]], tuple[str, str]]:
    """([(weight, bias) key of each hidden layer], (weight, bias) key of the output layer) for a net
    with `layers` hidden layers: the 2-layer shape keeps w1 w2 w3; deeper nets write w1 w2 wh3 ... and
    w_out, never a w3 (see the module docstring)."""
    if layers == 2:
        return [("w1", "b1"), ("w2", "b2")], ("w3", "b3")
    return [("w1", "b1"), ("w2", "b2"), *((f"wh{i}", f"bh{i}") for i in range(3, layers + 1))], ("w_out", "b_out")


def n_layers(p: dict[str, np.ndarray]) -> int:
    """The number of hidden layers of a model's params."""
    if "w3" in p:
        return 2
    n = 2
    while f"wh{n + 1}" in p:
        n += 1
    return n


def forward(p: dict[str, np.ndarray], nums: np.ndarray, cats: np.ndarray,
            category_of_slot: list[str] = features.CATEGORY_OF_SLOT, head: str = "value",
            fast: "FastInput | None" = None) -> np.ndarray:
    """The same computation as yomibench.train.Net.forward, in numpy. `head`: "value" (the
    output play uses), "aux" (the auxiliary hp-margin head, `w3_aux`/`b3_aux`) or "search" (the
    auxiliary search-value heads: rows x heads).

    Deliberately, inputs the model never saw vary are fed as their training value: a feature that
    was constant in training enters as 0 after standardization, whatever its value now. Otherwise
    a Ninja-only model (ckpt1, ckpt2) shown a Cowboy state would get its Cowboy fields divided by
    1e-3 (bullets_left 6 -> 6000) and saturate. Names a model never saw (an unseen character, its
    states and objects) encode as the unknown id 0, whose embedding row is the trained "none"
    row, as Vocab documents. So an unseen character is played as an unknown fighter whose own
    fields are neutral; its shared fields (hp, position, state timing, meter, ...) still count.

    `fast`: a FastInput built from these params (selfplay --fast-net-input; default off) computes the
    first hidden layer from the raw inputs instead. Close to this computation, not bit for bit."""
    hidden, out = layer_keys(n_layers(p))
    if fast is not None:
        h = fast(nums, cats)
        for wk, bk in hidden[1:]:
            h = np.maximum(h @ p[wk] + p[bk], 0)
    else:
        h = trunk(p, nums, cats, category_of_slot)
    if head == "search":
        return np.tanh(h @ p[SEARCH_W] + p[SEARCH_B])
    w, b = (p[out[0]], p[out[1]]) if head == "value" else (p["w3_aux"], p["b3_aux"])
    return np.tanh(h @ w + b)[:, 0]


def trunk(p: dict[str, np.ndarray], nums: np.ndarray, cats: np.ndarray,
          category_of_slot: list[str] = features.CATEGORY_OF_SLOT) -> np.ndarray:
    """The last hidden layer's output, which every head reads (see forward for the inputs).

    A version-4 model (EXT_W1 in its params) splits the first layer: the v3 inputs (the first
    features.N_NUMERIC numbers and features.N_CATEGORICAL ids) through w1 exactly as a version-3 model
    computes them, plus the ext inputs through w1_ext. With w1_ext all zero (net surgery) the ext term is an
    exact 0, so the outputs equal the parent's bit for bit."""
    x = [np.where(p["std"] <= CONSTANT_STD, 0.0, (nums - p["mean"]) / p["std"]).astype(np.float32)]
    for slot, cat in enumerate(category_of_slot):
        table = p[f"emb_{cat}"]
        x.append(table[np.minimum(cats[:, slot], len(table) - 1)])
    hidden = layer_keys(n_layers(p))[0]
    if EXT_W1 in p:
        n3, c3 = features.N_NUMERIC, features.N_CATEGORICAL
        std = x[0]
        h3 = np.concatenate([std[:, :n3], *x[1:1 + c3]], axis=1)
        hx = np.concatenate([std[:, n3:], *x[1 + c3:]], axis=1)
        (wk, bk), hidden = hidden[0], hidden[1:]
        h = np.maximum(h3 @ p[wk] + hx @ p[EXT_W1] + p[bk], 0)
    else:
        h = np.concatenate(x, axis=1)
    for wk, bk in hidden:
        h = np.maximum(h @ p[wk] + p[bk], 0)
    return h


def candidate_heads(p: dict[str, np.ndarray], h: np.ndarray, cnums: np.ndarray, cact: np.ndarray,
                    cdkey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The candidate heads (the same computation as yomibench.train.CandHeads): (search value in
    [-1, 1], mix logit) of each candidate, rows x K. `h`: trunk outputs (rows x hidden) of the
    deciding seat's view; `cnums` (rows x K x policy.N_CAND): candidates' numbers mirrored into that
    view (policy.flip_columns); `cact` / `cdkey`: their ids in meta `aux_cand_vocab`. Diagnostics and
    tests only: play never calls this."""
    def lin(x, name):
        return x @ p[f"{CAND}{name}_w"] + p[f"{CAND}{name}_b"]
    mean, std = p[f"{CAND}cmean"], p[f"{CAND}cstd"]
    ta, td = p[f"{CAND}emb_action"], p[f"{CAND}emb_dkey"]
    x = np.concatenate([np.where(std <= CONSTANT_STD, 0.0, (cnums - mean) / std).astype(np.float32),
                        ta[np.minimum(cact, len(ta) - 1)], td[np.minimum(cdkey, len(td) - 1)]], axis=-1)
    c = np.maximum(lin(np.maximum(lin(x, "c1"), 0), "c2"), 0)
    s = np.broadcast_to(np.maximum(lin(h, "s"), 0)[:, None, :], c.shape)
    j = np.maximum(lin(np.concatenate([s, c, s * c], axis=-1), "j"), 0)
    return np.tanh(lin(j, "value"))[..., 0], lin(j, "logit")[..., 0]


EXT_W1 = "w1_ext"  # inputs v4 (features/inputs-v4): the first layer's rows for the ext inputs, see FastInput
FAST_TABLE_ROWS = 64  # FastInput's pre-multiplied tables up to this many rows a call, embeddings above it


class FastInput:
    """The first hidden layer of `p` with the input preparation folded into its weights, built once
    per model (selfplay --fast-net-input; default off). `fast(nums, cats)` is forward's
    relu(standardized nums @ W_num + embeddings @ W_emb + b1), computed as

        relu(nums @ (W_num / std) + (b1 - (mean / std) @ W_num) + <the embedding term>)

    so no per-call standardization and no concatenated input vector. The embedding term, by call size:
    up to FAST_TABLE_ROWS rows, the sum over slots of T_slot[id] with T_slot = emb_<category> @ (that
    slot's rows of W_emb) pre-multiplied, one gather for all slots (no matmul: fewest numpy calls); for
    more rows, one gather per category (all its slots at once) into a reused buffer, times those rows of
    W_emb (less memory traffic than hidden-width rows per slot). Inputs constant in training (std <=
    CONSTANT_STD) get a zero weight row, as forward feeds them 0; ids past a table are clamped as in
    forward. An inputs-v4 model (EXT_W1: the v3 numbers and slots through w1, the ext ones through
    w1_ext) folds the same way into one W_num and per-slot rows.

    NOT bit-identical to forward: the folding is done in float64 and rounded once, and the sums run
    in a different order, so values move by ~1e-6 and search can pick differently; bots record it
    (`net_input: "folded"` in their config). Inputs must be finite (the encoders' are): a non-finite
    number in a constant-in-training column would give 0 * inf here where forward feeds 0."""

    def __init__(self, p: dict[str, np.ndarray], category_of_slot: list[str] = features.CATEGORY_OF_SLOT):
        (wk, bk), = layer_keys(n_layers(p))[0][:1]
        mean, std = p["mean"].astype(np.float64), p["std"].astype(np.float64)
        n = len(mean)
        w1 = p[wk].astype(np.float64)
        widths = [p[f"emb_{cat}"].shape[1] for cat in category_of_slot]
        if EXT_W1 in p:
            n3, c3 = features.N_NUMERIC, features.N_CATEGORICAL
            wx = p[EXT_W1].astype(np.float64)
            w_num = np.vstack([w1[:n3], wx[:n - n3]])
            at = [n3, n - n3]  # next unread row of w1, of w1_ext
            rows = []
            for slot, width in enumerate(widths):
                src, k = (w1, 0) if slot < c3 else (wx, 1)
                rows.append(src[at[k]:at[k] + width])
                at[k] += width
            if at != [len(w1), len(wx)]:
                raise ValueError(f"first layer rows {len(w1)} + {len(wx)} don't match the inputs {at}")
        else:
            w_num, at, rows = w1[:n], n, []
            for width in widths:
                rows.append(w1[at:at + width])
                at += width
            if at != len(w1):
                raise ValueError(f"first layer has {len(w1)} rows, the inputs {at}")
        inv = np.where(std <= CONSTANT_STD, 0.0, 1.0 / std)
        self.w = np.ascontiguousarray(w_num * inv[:, None], dtype=np.float32)
        self.b = (p[bk].astype(np.float64) - (mean * inv) @ w_num).astype(np.float32)
        # small calls: pre-multiplied tables, stacked, with each slot's offset and last id
        tables = [p[f"emb_{cat}"].astype(np.float64) @ r for cat, r in zip(category_of_slot, rows)]
        self.table = np.ascontiguousarray(np.vstack(tables), dtype=np.float32)
        self.offset = np.cumsum([0] + [len(t) for t in tables[:-1]]).astype(np.int32)
        self.last = np.array([len(t) - 1 for t in tables], np.int32)
        # large calls: per category (its slots, its embedding table), W_emb's rows in that order
        self.groups, w_emb = [], []
        for cat in dict.fromkeys(category_of_slot):
            slots = [s for s, c in enumerate(category_of_slot) if c == cat]
            table = p[f"emb_{cat}"]
            self.groups.append((np.array(slots), table, len(table) - 1, len(slots) * table.shape[1]))
            w_emb += [rows[s] for s in slots]
        self.w_emb = np.ascontiguousarray(np.vstack(w_emb), dtype=np.float32)
        self._emb = np.empty((0, len(self.w_emb)), np.float32)  # the large calls' embedding buffer, grown

    def __call__(self, nums: np.ndarray, cats: np.ndarray) -> np.ndarray:
        n = len(nums)
        h = nums @ self.w
        h += self.b
        if n <= FAST_TABLE_ROWS:
            idx = np.minimum(cats, self.last)
            idx += self.offset
            h += self.table[idx].sum(axis=1)
        else:
            if len(self._emb) < n:
                self._emb = np.empty((n, len(self.w_emb)), np.float32)
            emb, at = self._emb[:n], 0
            for slots, table, last, width in self.groups:
                emb[:, at:at + width] = table[np.minimum(cats[:, slots], last)].reshape(n, width)
                at += width
            h += emb @ self.w_emb
        return np.maximum(h, 0, out=h)

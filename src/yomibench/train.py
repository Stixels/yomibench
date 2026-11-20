"""Train a value network on self-play records (needs the `train` extra: torch, numpy).

    python -m yomibench.train RECORD_DIR [RECORD_DIR ...] --out checkpoints/v0.npz [--eval-dir DIR]

Every recorded decision is one sample labelled with p1's final result (+1 win, -1 loss, 0 draw);
the model is antisymmetric in the seats (see load_samples), so it covers p2's view too. Truncated
games are skipped. Games, not positions, are split into train and validation sets, so positions
from one game never sit on both sides, and the split is frozen: each folder's held-out manifest
(`<dir>/.heldout.json`, or under `--heldout-root`; see yomibench.heldout) assigns every game file
once and forever, so a game validated in one run is never trained on in a later one, and games
added later are assigned once and appended. `--eval-dir` adds games used only for evaluation,
never trained on: made with exploration off, they score every run on the same unexplored play.
The report compares the network's validation (and evaluation) error with the hand-written
`evaluate`, by game phase: does the net predict the winner better than the hp lead?

Exploration (an ablation, not a settled rule; tools/explore_ablation.py trains every setting on
identical games and compares them on the same validation and evaluation positions): a decision
whose `explore` marks a seat true was not played from the strategy. A
uniform pick (opening noise from `--opening-epsilon`, or any mode but softmax) can take the game
anywhere, so the final result says little about the positions before it; a softmax pick
(`explore_mode: "softmax"`, set 4's decision-wide `--explore-epsilon`) samples a low-temperature
softmax of the candidates' values, close to the policy. `--explore-targets` sets which picks cut
outcome targets off: positions at or before a game's last such pick by either seat train on the
search value alone (value weight 1; a turn there with no search value is skipped), later ones on
the requested blend.
  uniform-only (default)  cut at the last uniform pick; softmax picks keep outcome targets
  all                     cut at the last exploratory pick of any mode (the rule before this option)
  none                    outcome targets everywhere
Validation always scores against the outcome and drops the positions at or before a game's last
uniform pick, whatever `--explore-targets` is, so runs that differ only in it score the same
positions. The share of training positions on value-only targets, and the share of training
decisions with outcome targets by phase (early, mid, late), are printed and recorded in the
model's meta and the ledger row. Notes in `oracle.explore` (factor exploration widening the
search) change no pick and don't count; records without these fields have no exploratory picks.
Report blended-target and outcome errors separately when comparing blends; they answer different
questions.

Playout cap randomization (selfplay --cheap-fraction): a decision marked `"search": "cheap"` was
searched with a capped candidate list and no refinement, so its search value is a weak target; it
counts as a decision without a search value: the value weight blends in nothing (outcome alone),
and in a value-only stretch before an exploratory pick it is skipped. Unmarked decisions are full.
For the training split only (validation and evaluation keep these targets, so their MSE compares
across settings): `--cheap-target next-full` blends a quick turn toward the next full turn's search
value in the same game instead (weight `--cheap-value-weight`, default `--value-weight`; outcome
alone when no full turn follows), and `--full-turns-only` drops quick turns from training.

Epoch selection (`--select`): `best` (the default here) keeps the epoch with the lowest validation
MSE; `final` keeps the last epoch, a fixed training length (the round-3 trainer trains 3 epochs and
keeps the last). Either way the meta and the ledger row record the choice (`select`), the epoch
kept (`kept_epoch`), its validation MSE (`val_mse`) and the lowest seen (`best_epoch`,
`best_val_mse`).

Auxiliary hp-margin head (`--aux-hp-weight W`, default 0: none): a second output trained on the
game's final hp margin (p1's hp fraction minus p2's, `search.evaluate`'s scale, from the record's
result), antisymmetric like the value, added to the loss with weight W. It only shapes the shared
layers: play reads the value head alone (see yomibench.valuenet).

Residual on the hp lead (`--residual-hp`): the net predicts a correction to the hand-written
evaluator, value = clip(evaluate(state) + net(state), -1, 1). Targets are unchanged; the loss is
taken on the unclipped sum (a clipped prediction would stop the gradient), validation on the
clipped value, which is what play uses. The model's meta says `residual_hp: true`.

Evaluation folders: a folder frozen by yomibench.evalset (with an `.evalset.json` manifest of
record sha256s, e.g. round 3's gate games) is verified against its manifest, and every
`--eval-dir` record is checked against every record in the training folders: a shared seed or
record sha256 is refused, so evaluation games are never trained on. The meta and the ledger row
report evaluation MSE by matchup and sign accuracy (`eval_mse_by_matchup`, `eval_sign_accuracy`).

Fine-tuning (`--init-from MODEL.npz`): training starts from a saved model instead of random
weights: its standardization, layers and (with `--aux-hp-weight`) aux head, and its embedding rows
matched by name to this run's vocabulary (a name it never saw starts as its unknown row 0, which is
how the model encoded that name, so the warm start computes what the model computes on it). The run's vocabulary
starts as the model's own (names and characters' start values), then adds the training folders'
new names: a name the model knows but these folders lack keeps its embedding instead of encoding as
unknown, so an eval set scores the same whatever the training folder (`--lr 0` rescoring on a
small folder reproduces the model's own eval MSE). Its feature version,
hidden size and `residual_hp` must match this run's. The meta records `init_from` (path, sha256, name).
A numeric input that was constant in the model's training data (std at CONSTANT_STD, so the model
always fed it 0: e.g. a new character's raw fields) but varies in this run's training data is
re-scaled: it takes this run's mean and std, and its first-layer weights (never trained) start at 0,
so the warm-started net computes exactly what the model computes until training grows them. Every
other input keeps the model's own mean and std. The log names them ("init-constant features
re-scaled") and the meta counts them (`rescaled_features`).

Multi-step targets (`--target-nstep N` or `--target-lambda L`; default off, training split only):
the search value a target blends in, (1 - value_weight) * outcome + value_weight * value, becomes a
mean of the full-turn search values from that turn on in the same game, with no discount: N-step
takes the plain mean of the first N (the turn's own first when it has one), lambda the mean
weighted L^i over all of them (normalized; L 0 is the turn's own value, L 1 the plain mean). Full
turns use it in place of their own value; quick turns use it with `--cheap-target next-full` (their
window starts at the next full turn), else stay on the outcome. The window never crosses a pick that
cuts outcome targets: it stops at the first uniform pick (least-tried included; any exploratory
pick with `--explore-targets all`) at or after the turn, that pick's own search value included
(it was searched before the pick). The value-only stretch before such a pick trains on the same
mean. Quick turns' search values and turns without one are never averaged in. The setting is in the
model's meta (`target_nstep` / `target_lambda`) and the ledger row's target blend.

Auxiliary search-value heads (`--aux-search-heads 1,4,16 --aux-search-weight 0.1`; default none):
one extra output per offset D, trained on the search value of the nearest full turn at or after
D decisions later in the same game (masked where the game has none, or past a pick that cuts the
window as above), antisymmetric like the value, a side loss (each head's masked MSE, averaged over
the heads, times the weight). They only shape the shared layers: the saved model keeps them under
`aux_search_w`/`aux_search_b`, keys play never reads (valuenet.forward's value head is unchanged).
Search values only: no resource counts are targets. The meta records `aux_search_heads`,
`aux_search_weight` and each head's validation MSE (`val_aux_search_mse`).

Depth (`--layers N`, default 2): N hidden layers of `--hidden` units, for capacity tests. The
network is evaluated in numpy only (valuenet.forward; a batch engine plugin computes features and
never reads the weights), so a deeper model plays with this code. It saves its output layer as
`w_out` instead of `w3` (valuenet.layer_keys), so older code refuses it (KeyError) rather than
misreading it; the meta says `layers`.

Candidate heads (`--aux-cand-value-weight W1`, `--aux-mix-weight W2`; default 0: off): side outputs
trained on what the search records at full turns (yomibench.search_record), for both seats: (a) each
candidate's search value (records' `candidate_values`, the seat's own side), regressed by a
candidate-conditioned head: the trunk's last hidden layer for that seat's view, combined with an
embedding of the candidate as yomibench.policy encodes it (action, data shape, aim/slider/parry and
factor values, mirrored into the view); (b) the solved mix (`strategies`): a softmax of a second
output of the same head over that seat's candidates, cross-entropy against the mix. Quick turns and
turns without those fields have no candidate targets. Each seat keeps at most `--aux-cand-k` K
candidates per turn (default 16): the mix's whole support (the K likeliest if it is larger), then a
seeded sample of the rest, so memory is about 12 bytes per candidate slot (2K per full turn) plus one
table of unique candidates; the mix is renormalized over the kept list. Losses: (a) the weighted MSE
over kept candidates with a search value, (b) the weighted mean cross-entropy per seat and turn,
times W1 and W2. Like the other auxiliary heads they only shape the shared layers: the saved model
keeps them under `aux_cand_*` keys, which play and older code never read (valuenet.forward's value
head is unchanged), and the meta records `aux_cand_value_weight`, `aux_mix_weight`, `aux_cand_k`,
the candidate vocabulary (`aux_cand_vocab`) and their validation scores (`val_aux_cand_value_mse`,
`val_aux_mix_ce`, `val_aux_mix_top1`). `--init-from` starts them fresh. The cached loader reads the
lists from a candidate side cache built next to each training folder's cache only when these flags
are on (yomibench.cache, `cands-k<K>/`): the main cache, its format and its encoder hash are
unchanged.

Both target switches are computed per game from the per-turn facts the record cache already holds
(search value, quick mark, exploration marks), by select_rows at load time: neither the cache
format nor its encoder hash changes.

Generations: give one folder per generation, oldest first. `--window N` trains on the last N of
them (say the last 3-5), `--games-per-gen K` takes at most K records from each (a seeded sample),
and `--gen-weights` scales each generation's samples in the loss (validation stays unweighted and
is also reported per generation).

Sample window (`--window-budget-games B`, off by default; `--window-recent-hours H`, `--window-seed S`):
a long window can outgrow the page cache, and the memory-mapped cache then pages on every epoch.
After the record filter, the training games are cut to at most B: every training game from the H
hours before the newest game, plus a uniform sample (seeded by S) of the older training games to
fill the budget; if the recent games alone exceed B, the most recent B (logged). A game's time is
its source file's mtime (the cache stores it and checks it against the file, so both loaders cut
the same games; the record header's `started` isn't in the cache). Validation is never sampled and
no game changes split: the frozen held-out set stays whole, so validation MSE compares across fits.
One line per fit logs the counts (recent kept, older sampled of the older, total, by candidate set
and by ceiling); the meta and the ledger row record them under `sample_window`. With the cache,
the dropped games' rows are never read: only the kept games' row ranges enter the splits.

Shared window (`--share-window DIR`, off by default; cached loader only): the selected window is
written under DIR, keyed by the data arguments, and a later run with the same ones reads it instead
of selecting again while the records are unchanged, training exactly the same model
(yomibench.cache, "Shared windows"); `--share-window-frozen` also reuses it after games arrived.
`shared_window` (meta and ledger) says whether the window was written or read; the ledger row's
`phase_seconds` then has `share_import` (and `share_export` when written), and a run that read the
window has no `window_select`, `row_select` or vocab pass of its own (its `vocab` is the lookup table).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# Record-parsing workers are spawned processes, and spawn re-runs the main module (this one, under
# `python -m yomibench.train`) as __mp_main__ in each. They only need read_slim: importing torch
# there costs each worker about 1.6 GB of committed memory.
if __name__ != "__mp_main__":
    import torch
    from torch import nn
else:
    nn = SimpleNamespace(Module=object)  # workers never build a Net

from yomibench import compat, evalset, heldout, ledger
from yomibench.compat import check_keys, record_keys  # noqa: F401  (record_keys: public here)
from yomibench.features import CATEGORY_OF_SLOT, CHARACTERS, N_CATEGORICAL, N_NUMERIC, NUMERIC_NAMES, Vocab, encode
from yomibench.policy import CAND_CATEGORIES, EMBED as CEMBED, N_CAND, X_COLUMNS, flip_columns
from yomibench.sampled_starts import is_sampled, start_offset
from yomibench.search import evaluate
from yomibench.search_record import DEFAULT_K, CandTable, SplitCands, finish_table, turn_candidates
from yomibench.selfplay import final_hp_margin, read_header, read_slim, uniform_explore
from yomibench.valuenet import (CAND, CAND_LAYERS, CONSTANT_STD, EMBED, FEATURES, SEARCH_B, SEARCH_W, ValueNet,
                                candidate_heads, forward, layer_keys, n_layers, residual_value, trunk)

PHASES = ((0.0, 1 / 3, "early"), (1 / 3, 2 / 3, "mid"), (2 / 3, 1.01, "late"))
EXPLORE_TARGETS = ("uniform-only", "all", "none")  # which exploratory picks cut outcome targets off
CHEAP_TARGETS = ("outcome", "next-full")  # a quick turn's training target (select_rows, --cheap-target)
SELECT = ("best", "final")  # which epoch's weights are kept: lowest validation MSE, or the last


class PhaseTimer:
    """Wall seconds per named phase of a fit (metrics.phase_seconds of its ledger row and the
    'phase seconds:' log line). `with timer("name"):` adds the block's seconds to that phase (a phase
    timed more than once adds up); phases keep the order they were first timed."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + max(0.0, float(seconds))

    def __call__(self, name: str):
        import contextlib

        @contextlib.contextmanager
        def timed():
            t = time.perf_counter()
            try:
                yield
            finally:
                self.add(name, time.perf_counter() - t)
        return timed()

    def rounded(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.seconds.items()}

    def line(self) -> str:
        return "phase seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in self.seconds.items())


def explore_cutoffs(turns: list[Any], explore_targets: str = "uniform-only") -> tuple[int, int]:
    """(last turn index on value-only targets, last turn index left out of validation); -1 for none.
    Validation's cutoff is the last uniform pick whatever `explore_targets` is (see the module docstring)."""
    if explore_targets not in EXPLORE_TARGETS:
        raise ValueError(f"explore_targets must be one of {EXPLORE_TARGETS}, not {explore_targets!r}")
    last_uniform = max((k for k, t in enumerate(turns) if uniform_explore(t)), default=-1)
    if explore_targets == "uniform-only":
        return last_uniform, last_uniform
    if explore_targets == "all":
        return max((k for k, t in enumerate(turns) if any(t.explore)), default=-1), last_uniform
    return -1, last_uniform


# float64 columns stay on the host (report columns); the rest go to the device for training.
DTYPES = {"nums0": np.float32, "cats0": np.int64, "nums1": np.float32, "cats1": np.int64, "y": np.float32,
          "w": np.float32, "hp_margin": np.float32, "outcome": np.float32, "phase": np.float64, "baseline": np.float64, "gen": np.float64,
          "matchup": np.float64}


def matchup_code(characters: list[str] | tuple[str, ...]) -> int:
    """A game's (p1, p2) characters as one number for the `matchup` column; -1 if unknown."""
    try:
        a, b = (CHARACTERS.index(str(c)) for c in characters)
    except ValueError:
        return -1
    return a * len(CHARACTERS) + b


def matchup_name(code: int) -> str:
    return "?" if code < 0 else f"{CHARACTERS[code // len(CHARACTERS)]}:{CHARACTERS[code % len(CHARACTERS)]}"
SHAPES = {"nums0": (N_NUMERIC,), "nums1": (N_NUMERIC,), "cats0": (N_CATEGORICAL,), "cats1": (N_CATEGORICAL,)}


def _bounded_map(pool: ProcessPoolExecutor, fn, items, window: int):
    """pool.map in order, but with at most `window` results in flight: Executor.map submits
    everything at once, and parsed games would pile up in memory while the encoder catches up."""
    items, pending = list(items), deque()
    for item in items[:window]:
        pending.append(pool.submit(fn, item))
    for item in items[window:]:
        yield pending.popleft().result()
        pending.append(pool.submit(fn, item))
    while pending:
        yield pending.popleft().result()


def check_compatible(headers: list[dict[str, Any]], allow_mixed_candidate_sets: bool = False,
                     allow_match_settings: list[dict[str, Any]] | None = None) -> None:
    """Refuse records of different games (record_keys), and of different candidate sets unless
    `allow_mixed_candidate_sets`; `allow_match_settings`: the match-settings variants that may mix
    (compat.check_keys)."""
    check_keys([record_keys(h) for h in headers], allow_mixed_candidate_sets,
               **({"allow_match_settings": allow_match_settings} if allow_match_settings else {}))


def variants_of(files: list[Any], keys_of: Any, allow_match_settings: list[dict[str, Any]] | None
                ) -> dict[Any, int] | None:
    """{file: index of the --allow-match-settings variant it matches (-1: none)}, None without a list."""
    if not allow_match_settings:
        return None
    out = {}
    for f in files:
        i = compat.match_variant(tuple(keys_of(f)), allow_match_settings)
        out[f] = -1 if i is None else i
    return out


def variant_mse(split: dict[str, Any], pred: np.ndarray | None,
                allow_match_settings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{variant label: {games, n, net, evaluate, sign_accuracy}}: a split's MSE against the outcome
    for each --allow-match-settings variant ("other": matching none, listed only if present)."""
    labels = [compat.variant_label(v) for v in allow_match_settings]
    codes = np.asarray(split.get("variant", np.zeros(0))).astype(int)
    games = list(split.get("variant_games") or [0] * (len(labels) + 1))
    out = {}
    for i, name in [*enumerate(labels), (-1, "other")]:
        m = codes == i
        n = int(m.sum())
        if i == -1 and not n and not games[-1]:
            continue
        row: dict[str, Any] = {"games": games[i], "n": n, "net": None, "evaluate": None, "sign_accuracy": None}
        if n and pred is not None:
            z = np.asarray(split["outcome"])[m]
            row.update(net=round(float(np.mean((pred[m] - z) ** 2)), 5),
                       evaluate=round(float(np.mean((np.asarray(split["baseline"])[m] - z) ** 2)), 5),
                       sign_accuracy=round(float(np.mean(np.sign(pred[m]) == np.sign(z))), 5))
        out[name] = row
    return out


def load_samples(dirs: list[Path], vocab: Vocab, val_fraction: float, seed: int,
                 value_weight: float = 0.0, workers: int | None = None, device: str | None = None,
                 allow_mixed_candidate_sets: bool = False, games_per_gen: int | None = None,
                 gen_weights: list[float] | None = None, explore_targets: str = "uniform-only",
                 eval_dirs: list[Path] | None = None, heldout_root: Path | None = None,
                 heldout_seed: int = 0, quick_turns: dict[str, Any] | None = None,
                 record_filter: dict[str, Any] | None = None,
                 allow_match_settings: list[dict[str, Any]] | None = None,
                 cand_k: int | None = None, window: dict[str, Any] | None = None,
                 exclude_sampled_starts: bool = False) -> dict[str, Any]:
    """One sample per recorded decision: the state encoded from both seats' views and p1's final
    result (+1, -1, 0). The model's value for p1 is (f(p1 view) - f(p2 view)) / 2, antisymmetric by
    construction, so p2's value is exactly its negative (the search relies on that).

    Splits: `train`, `val` and `eval`. Each folder's games split by its frozen held-out manifest
    (yomibench.heldout; `val_fraction` and `heldout_seed` only matter for a folder's first
    manifest, `heldout_root` keeps manifests outside the folders): held-out games are validation,
    never training. `eval_dirs` are games used only for evaluation (made with exploration off;
    a record with an exploratory pick is refused), never trained on. `heldout` maps each folder
    to its manifest (path, sha256, counts).

    The training target `y` blends in the search's own value for p1 at that decision:
    (1 - value_weight) * outcome + value_weight * value. `outcome` stays the final result, which is
    what validation and evaluation measure. `explore_targets` picks which exploratory picks put the
    positions before them on value-only targets (see the module docstring and explore_cutoffs);
    validation always drops the positions at or before a game's last uniform pick. Each split's
    `explored` counts the positions cut off (train: put on value-only targets or skipped; val:
    dropped) and `value_only` the samples on value-only targets; the training split's
    `outcome_coverage` is the share of its decisions with outcome targets, by phase and overall.

    With `device`, the training split goes there game by game (torch tensors), so host memory
    never holds more than a game of it; validation and evaluation stay numpy. Every record's
    header is checked first (check_compatible), before anything is parsed.

    Each folder in `dirs` is a generation (`gen` column: its index; -1 for evaluation games).
    `games_per_gen` caps each at that many records, a seeded sample of its training and held-out
    games in their proportions; `gen_weights` (one per folder, default 1) is each sample's loss
    weight, column `w`. Each split's `v3_fraction` is the share of its sample states that carry the
    engine-v3 move fields, over every included generation (the training split's is the model's
    `state_v3_fraction`); `v3_fraction_by_gen` breaks it down.

    `record_filter` (compat.apply_record_filter): only records matching it are kept, in every split;
    `record_filter` in the result says what was kept and excluded.

    `allow_match_settings` (compat.check_keys): match-settings variants that may share the window
    (and pass the record filter); each split then has a `variant` column (the variant index of the
    row's record, -1: none) and `variant_games` (games per variant, the last entry: none).

    `cand_k` K (the candidate heads): records are read with their candidate lists, and the training
    and validation splits get `cand_*` columns (search_record.split_arrays: a slot per row with a
    qualifying full turn, both seats' K-capped lists as rows of one candidate table) and the result
    a `cand_table` (search_record.finish_table). Without it nothing changes.

    `window` ({"recent_hours", "budget", "seed"}: train --window-budget-games): after the record
    filter, the training games are cut by sample_window (time: each file's mtime); `sample_window` in
    the result has its settings and counts. Without it nothing changes.

    `exclude_sampled_starts` (or the record filter's "exclude_sampled_starts": true): sampled-start
    records (yomibench.sampled_starts) are left out of every split, counted in `sampled_starts_excluded`.
    Without it they are read like any record, at weight 1, their rows' phases from their true decision
    index (select_rows' offset)."""
    explore_cutoffs([], explore_targets)  # validates the setting before any work
    if gen_weights is not None and len(gen_weights) != len(dirs):
        raise SystemExit(f"{len(gen_weights)} generation weights for {len(dirs)} folders")
    eval_dirs = list(eval_dirs or [])
    if {Path(d).resolve() for d in eval_dirs} & {Path(d).resolve() for d in dirs}:
        raise SystemExit("an --eval-dir folder is also a training folder; evaluation games must never be trained on")
    weights = gen_weights or [1.0] * len(dirs)
    chunks, gen_of, manifests = select_files(dirs, val_fraction, seed, games_per_gen, eval_dirs, heldout_root,
                                             heldout_seed)
    files = [p for c in chunks.values() for p in c]
    out: dict[str, Any] = {"heldout": manifests}
    # Training games first, so the vocabulary is built from them alone; names seen only in
    # validation or evaluation map to 0 (unknown) instead of getting embeddings training never updated.
    workers = workers or min(8, os.cpu_count() or 1)
    # Workers inherit the environment. Parsing does no linear algebra, and OpenBLAS otherwise
    # commits buffers for a thread per core on import (about 750 MB per worker on a 28-thread CPU).
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    pool = ProcessPoolExecutor(workers) if workers > 1 and len(files) > 1 else None
    load = (lambda chunk, fn=read_slim: _bounded_map(pool, fn, chunk, 2 * workers)) if pool else \
        (lambda chunk, fn=read_slim: map(fn, chunk))
    try:
        headers = list(load(files, read_header))
        if compat.excludes_sampled_starts(record_filter, exclude_sampled_starts):
            by_file = dict(zip(files, headers))
            out["sampled_starts_excluded"] = compat.drop_sampled_starts(chunks, lambda f: is_sampled(by_file[f]),
                                                                        partial(print, flush=True))
            files = [p for c in chunks.values() for p in c]
            headers = [by_file[p] for p in files]
        if record_filter:
            by_file = dict(zip(files, headers))
            out["record_filter"] = compat.apply_record_filter(chunks, lambda f: compat.record_keys(by_file[f]),
                                                              record_filter,
                                                              **({"allow_match_settings": allow_match_settings}
                                                                 if allow_match_settings else {}))
            files = [p for c in chunks.values() for p in c]
            headers = [by_file[p] for p in files]
        if window:
            by_file = dict(zip(files, headers))
            out["sample_window"] = sample_window(chunks, lambda f: Path(f).stat().st_mtime,
                                                 lambda f: compat.record_keys(by_file[f]), window["recent_hours"],
                                                 window["budget"], window["seed"], partial(print, flush=True))
            files = [p for c in chunks.values() for p in c]
            headers = [by_file[p] for p in files]
        compat.refuse_held_out(list(zip(files, headers)))
        check_compatible(headers, allow_mixed_candidate_sets, allow_match_settings)
        header_of = dict(zip(files, headers))
        variant_of = variants_of(files, lambda f: compat.record_keys(header_of[f]), allow_match_settings)
        table = CandTable() if cand_k else None
        for split in ("train", "val", "eval"):
            out[split] = _load_split(split, chunks[split], load, gen_of, len(dirs), weights, vocab, value_weight,
                                     explore_targets, device, quick_turns,
                                     **({"variant_of": variant_of, "n_variants": len(allow_match_settings)}
                                        if variant_of is not None else {}),
                                     **({"cands": (cand_k, table)} if cand_k and split != "eval" else {}))
        if cand_k:
            out["cand_table"] = finish_table(table.unums(), table.names, out["train"]["cand_uid"])
    finally:
        vocab.frozen = False
        if pool:
            pool.shutdown()
    return out


def select_files(dirs: list[Path], val_fraction: float, seed: int, games_per_gen: int | None,
                 eval_dirs: list[Path], heldout_root: Path | None, heldout_seed: int):
    """Which record files go to which split (load_samples; the training cache shares it): each
    folder's frozen held-out manifest, the `games_per_gen` sample, evaluation folders. Returns
    ({"train": [...], "val": [...], "eval": [...]}, {path: generation (-1: evaluation)}, manifests)."""
    gen_of: dict[Path, int] = {}
    chunks: dict[str, list[Path]] = {"train": [], "val": [], "eval": []}
    manifests: dict[str, Any] = {}
    for d in dirs:  # human-play folders are a held-out human test (compat.HELD_OUT_MARKER)
        marker = compat.held_out_folder(d)
        if marker:
            raise SystemExit(f"{d} is held out ({marker}): human-play games are never trained on")
    for gen, d in enumerate(dirs):
        tr_files, va_files, manifests[str(d)] = heldout.split(d, val_fraction, heldout_seed, heldout_root)
        total = len(tr_files) + len(va_files)
        if games_per_gen is not None and total > games_per_gen:
            n_va = min(len(va_files), round(games_per_gen * len(va_files) / total))
            if va_files and not n_va and games_per_gen >= 2:
                n_va = 1
            n_tr = min(len(tr_files), games_per_gen - n_va)
            tr_files = sorted(random.Random(f"{seed}:{gen}:train").sample(tr_files, n_tr))
            va_files = sorted(random.Random(f"{seed}:{gen}:val").sample(va_files, n_va))
        chunks["train"] += tr_files
        chunks["val"] += va_files
        gen_of.update((p, gen) for p in tr_files + va_files)
    for d in eval_dirs:
        found = sorted(Path(d).glob(heldout.PATTERN))
        chunks["eval"] += found
        gen_of.update((p, -1) for p in found)
    if chunks["eval"]:  # evaluation games are never trained on: frozen sets intact, no shared game
        try:
            for d in eval_dirs:
                if (Path(d) / evalset.MANIFEST).exists():
                    evalset.verify(Path(d))
            evalset.check_disjoint(chunks["eval"], [p for d in dirs for p in sorted(Path(d).glob(heldout.PATTERN))])
        except ValueError as e:
            raise SystemExit(str(e)) from e
    return chunks, gen_of, manifests


def _ceiling_label(keys: tuple[str, str]) -> str:
    """A record's ceiling for the sample window's counts: "800", "500", or "none" (no ceiling)."""
    have = json.loads(keys[0]).get("match_settings") or {}
    std = compat.SIM_SETTINGS_STANDARD
    on = have.get("has_ceiling", std["has_ceiling"])
    return str(have.get("ceiling_height", std["ceiling_height"])) if on not in (False, "false", "False") else "none"


def sample_window(chunks: dict[str, list[Any]], time_of: Any, keys_of: Any, recent_hours: float, budget: int,
                  seed: int, log: Any = print) -> dict[str, Any]:
    """--window-budget-games: cut the training split of `chunks` (in place, after the record filter)
    to at most `budget` games: every training game from the last `recent_hours` hours, plus a seeded
    uniform sample (`seed`) of the older training games to fill the budget; if the recent games
    alone exceed it, the most recent `budget` of them. A game's time is `time_of(path)` (the source
    file's mtime, which both loaders read the same: the cache checks it against the file); "now" is
    the newest training or validation game, so the cut depends only on the records, not the clock.
    Validation and evaluation are never sampled and no game changes split, so validation stays the
    same frozen held-out set across fits. Logs one line; returns the settings and counts."""
    train_files = list(chunks["train"])
    times = {p: float(time_of(p)) for p in [*train_files, *chunks["val"]]}
    newest = max(times.values(), default=0.0)
    cutoff = newest - recent_hours * 3600.0
    recent = [p for p in train_files if times[p] >= cutoff]
    older = [p for p in train_files if times[p] < cutoff]
    over = len(recent) > budget
    if over:  # newest first; a path breaks ties so both loaders agree
        recent = sorted(recent, key=lambda p: (-times[p], str(p)))[:budget]
        sampled: list[Any] = []
    else:
        rng = random.Random(f"{seed}:sample-window")
        sampled = rng.sample(sorted(older, key=str), min(len(older), budget - len(recent)))
    keep = set(recent) | set(sampled)
    chunks["train"][:] = [p for p in train_files if p in keep]  # the split's own order
    by_set: dict[str, int] = {}
    by_ceiling: dict[str, int] = {}
    for p in chunks["train"]:
        keys = tuple(keys_of(p))
        by_set[str(keys[1])] = by_set.get(str(keys[1]), 0) + 1
        by_ceiling[_ceiling_label(keys)] = by_ceiling.get(_ceiling_label(keys), 0) + 1
    summary = {"recent_hours": recent_hours, "budget_games": budget, "seed": seed, "time": "source file mtime",
               "newest": newest, "cutoff": cutoff, "candidates": len(train_files),
               "recent": len(recent) if not over else sum(times[p] >= cutoff for p in train_files),
               "recent_kept": len(recent), "older": len(older), "older_sampled": len(sampled),
               "kept": len(chunks["train"]), "recent_over_budget": over, "val_games": len(chunks["val"]),
               "val_sampled": False, "kept_by_candidate_set": dict(sorted(by_set.items())),
               "kept_by_ceiling": dict(sorted(by_ceiling.items()))}
    log(f"sample window (recent {recent_hours:g}h, budget {budget} training games, seed {seed}; game time: source "
        f"file mtime): kept {summary['recent_kept']} recent + {len(sampled)} older sampled (of {len(older)} older) = "
        f"{summary['kept']} of {len(train_files)} training games"
        + (f"; the {summary['recent']} recent games exceed the budget: kept the most recent {budget}" if over else "")
        + f"; validation kept complete ({len(chunks['val'])} games)"
        + "; kept by candidate set: " + ", ".join(f"{k} {n}" for k, n in summary["kept_by_candidate_set"].items())
        + "; kept by ceiling: " + ", ".join(f"{k} {n}" for k, n in summary["kept_by_ceiling"].items()))
    return summary


def window_stops(turns: list[Any], explore_targets: str = "uniform-only") -> list[int]:
    """For each turn k, the last turn index a multi-step target or an auxiliary search target may read
    from k: the first pick at or after k that cuts windows (a uniform pick, least-tried included, and
    with `explore_targets` "all" any exploratory pick), else the game's last turn. The search value at
    the pick itself was searched before the pick, so it is in the window; the turns after it are not."""
    stops, stop = [0] * len(turns), len(turns) - 1
    for k in range(len(turns) - 1, -1, -1):
        t = turns[k]
        if uniform_explore(t) or (explore_targets == "all" and any(t.explore)):
            stop = k
        stops[k] = stop
    return stops


def _full_values(turns: list[Any]) -> list[float | None]:
    """Each turn's search value if it is a full-search turn with one, else None (quick turns' values
    are no targets)."""
    return [None if t.cheap or t.value is None else float(t.value) for t in turns]


def multistep_values(turns: list[Any], explore_targets: str = "uniform-only", nstep: int | None = None,
                     lam: float | None = None) -> list[float | None]:
    """Per turn k, the multi-step search value (no discount): over the full-turn search values at
    turns k..window_stops[k] in order (v_0 is the first, k's own when it has one), with `nstep` N the
    plain mean of the first N, with `lam` L the mean weighted L^i (normalized over the values the
    window holds). None where the window holds no full-turn value."""
    full, stops = _full_values(turns), window_stops(turns, explore_targets)
    out: list[float | None] = [None] * len(turns)
    if nstep is not None:
        idx = [k for k, v in enumerate(full) if v is not None]
        j = len(idx)  # position in idx of the first full turn at or after k
        for k in range(len(turns) - 1, -1, -1):
            while j > 0 and idx[j - 1] >= k:
                j -= 1
            end = j  # one past the last full turn in the window, at most N of them
            while end < len(idx) and end - j < nstep and idx[end] <= stops[k]:
                end += 1
            if end > j:
                out[k] = sum(full[m] for m in idx[j:end]) / (end - j)
        return out
    num = den = 0.0
    for k in range(len(turns) - 1, -1, -1):
        if stops[k] == k:  # k cuts the window: nothing after it counts from here back
            num = den = 0.0
        if full[k] is not None:
            num, den = full[k] + lam * num, 1.0 + lam * den
        out[k] = num / den if den else None
    return out


def aux_search_targets(turns: list[Any], offsets: list[int] | tuple[int, ...],
                       explore_targets: str = "uniform-only") -> list[tuple[float, ...]]:
    """Per turn k and offset d, the search value of the nearest full turn at or after k + d; NaN (no
    target) where the game has none or it lies past window_stops[k] (a cutting pick in between)."""
    full, stops = _full_values(turns), window_stops(turns, explore_targets)
    nxt, at = [len(turns)] * (len(turns) + 1), len(turns)  # nxt[m]: first full turn index >= m
    for m in range(len(turns) - 1, -1, -1):
        if full[m] is not None:
            at = m
        nxt[m] = at
    out = []
    for k in range(len(turns)):
        row = []
        for d in offsets:
            m = nxt[min(k + d, len(turns))]
            row.append(full[m] if m < len(turns) and m <= stops[k] else float("nan"))
        out.append(tuple(row))
    return out


def select_rows(turns: list[Any], split: str, explore_targets: str, value_weight: float, z: float,
                cheap_target: str = "outcome", cheap_value_weight: float | None = None,
                full_turns_only: bool = False, target_nstep: int | None = None,
                target_lambda: float | None = None, aux_offsets: list[int] | tuple[int, ...] = (),
                offset: int = 0):
    """One game's samples for a split (load_samples; the training cache shares it): [(turn index,
    target y, phase)] plus the counts load_samples reports. Needs only each turn's `value`,
    `explore`, `explore_mode` and `cheap`. Validation and evaluation drop the positions at or before
    the last uniform pick; training puts the positions up to the target cutoff on value-only targets
    (skipping those with no value).

    Quick (playout-capped, `cheap`) turns, training split only (validation and evaluation keep the
    default targets, so their MSE stays comparable across these settings): `cheap_target` "outcome"
    (default) trains them like a turn without a search value; "next-full" blends them toward the
    next full turn's search value in the same game (weight `cheap_value_weight`, default
    `value_weight`; outcome alone when no full turn follows); `full_turns_only` drops them.

    Multi-step targets, training split only: `target_nstep` N or `target_lambda` L replaces the search
    value a target blends in (a full turn's own; with cheap_target "next-full", a quick turn's next
    full one) by multistep_values' mean over the window from that turn, which never crosses a cutting
    pick (window_stops). The value-only stretch before a pick trains on that mean too.

    `aux_offsets` (every split): the result's counts also hold "aux", one tuple per row of
    aux_search_targets for those offsets.

    `offset`: the true decision index of the game's first recorded turn (sampled_starts.start_offset:
    a sampled-start record begins at decision k, not at the opening). A row's phase is its true index
    over the game's true length, (offset + k) / (offset + len(turns)); 0 (games from the opening) is
    the row index over the record's turns, as before. Targets and cutoffs don't depend on it."""
    target_cut, val_cut = explore_cutoffs(turns, explore_targets)
    if cheap_target not in CHEAP_TARGETS:
        raise ValueError(f"cheap_target must be one of {CHEAP_TARGETS}, not {cheap_target!r}")
    if target_nstep is not None and target_lambda is not None:
        raise ValueError("target_nstep and target_lambda are two designs of one target: pick one")
    train = split == "train"
    multi = None  # the multi-step search value from each turn (training only)
    if train and (target_nstep is not None or target_lambda is not None):
        multi = multistep_values(turns, explore_targets, target_nstep, target_lambda)
    next_full: list[float | None] = [None] * len(turns)  # the next full turn's search value, after k
    if train and cheap_target == "next-full" and multi is None:
        nxt = None
        for k in range(len(turns) - 1, -1, -1):
            next_full[k] = nxt
            if not turns[k].cheap and turns[k].value is not None:
                nxt = turns[k].value
    # A cheap (playout-capped) search value is no target: such a turn trains like one without a value.
    rows: list[tuple[int, float, float]] = []
    explored = value_only = 0
    decisions, outcome_targets = [0] * len(PHASES), [0] * len(PHASES)
    for k, t in enumerate(turns):
        if t.cheap and train and full_turns_only:
            continue
        w = value_weight
        if not t.cheap:
            value = t.value if multi is None else multi[k]
        elif train and cheap_target == "next-full":
            value = next_full[k] if multi is None else multi[k]
            w = value_weight if cheap_value_weight is None else cheap_value_weight
        else:
            value = None
        phase = (offset + k) / max(1, offset + len(turns))
        blend = (1 - w) * z + w * (z if value is None else value)
        if split != "train":
            if k <= val_cut:
                explored += 1
                continue
            y = blend
        else:
            ph = next(i for i, (lo, hi, _) in enumerate(PHASES) if lo <= phase < hi)
            decisions[ph] += 1
            if k <= target_cut:
                explored += 1
                if value is None:
                    continue
                y = value
                value_only += 1
            else:
                y = blend
                outcome_targets[ph] += 1
        rows.append((k, y, phase))
    counts = {"explored": explored, "value_only": value_only, "decisions": decisions,
              "outcome_targets": outcome_targets}
    if aux_offsets:
        targets = aux_search_targets(turns, aux_offsets, explore_targets)
        counts["aux"] = [targets[k] for k, _, _ in rows]
    return rows, counts


def _load_split(split, chunk, load, gen_of, n_gens, weights, vocab, value_weight, explore_targets, device,
                quick_turns=None, variant_of=None, n_variants=0, cands=None):
    """One split's columns and counts (see load_samples). `cands`: (K, the run's CandTable) for the
    candidate heads' columns."""
    cand_rows = SplitCands(cands[0]) if cands else None
    variant_col, variant_games = [], [0] * (n_variants + 1)  # with variant_of (--allow-match-settings)
    vocab.frozen = split != "train"
    cols: dict[str, list] = {k: [] for k in DTYPES}
    aux_cols: list = []  # per game: its rows' auxiliary search targets (rows x offsets), with aux_offsets
    games = explored = value_only = 0
    v3_states, n_states = [0] * (n_gens + 1), [0] * (n_gens + 1)  # per generation; the last is evaluation
    decisions, outcome_targets = [0] * len(PHASES), [0] * len(PHASES)  # training coverage, by phase
    on_device = device is not None and split == "train"
    place = (lambda a: torch.from_numpy(a).to(device)) if on_device else (lambda a: a)
    games_in = load(chunk, partial(read_slim, policy=True)) if cands else load(chunk)
    for path, (header, result, turns) in zip(chunk, games_in):
        gen = gen_of[path]
        result = result["result"]
        if result["outcome"] == "truncated":
            continue
        if split == "eval" and any(any(t.explore) for t in turns):
            raise SystemExit(f"{path} has exploratory picks; --eval-dir games must be made with exploration off")
        games += 1
        if variant_of is not None:
            variant_games[variant_of[path]] += 1  # -1 (no variant): the last entry
        if turns and not vocab.frozen:
            vocab.learn_defaults(turns[0][0])  # characters' start values, for raw entries left out
        z = 0.0 if result["winner"] is None else (1.0 if result["winner"] == 0 else -1.0)
        margin = final_hp_margin(result, turns[0].state if turns else None)
        # Value-only targets up to the target cutoff; validation drops up to its own cutoff.
        rows, counts = select_rows(turns, split, explore_targets, value_weight, z, **(quick_turns or {}),
                                   offset=start_offset(header))
        mu = matchup_code(header.get("characters", ()))
        explored += counts["explored"]
        value_only += counts["value_only"]
        decisions = [a + b for a, b in zip(decisions, counts["decisions"])]
        outcome_targets = [a + b for a, b in zip(outcome_targets, counts["outcome_targets"])]
        g: dict[str, list] = {k: [] for k in cols}
        if cand_rows is not None:  # the rows' full-turn candidate lists, keyed as the side cache keys them
            for k, _, _ in rows:
                cand_rows.add_row(turn_candidates(turns[k], cands[0], f"{Path(path).name}:{k}"), cands[1])
        for k, y, phase in rows:
            state = turns[k].state
            n_states[gen] += 1  # over the states that become samples, in every generation
            v3_states[gen] += "hitboxes" in state["players"][0]  # engine-v3 move fields present
            for seat in (0, 1):
                n, c = encode(state, seat, vocab)
                g[f"nums{seat}"].append(n)
                g[f"cats{seat}"].append(c)
            g["y"].append(y)
            g["outcome"].append(z)
            g["hp_margin"].append(margin)
            g["phase"].append(phase)
            g["baseline"].append(evaluate(state, 0))
            g["w"].append(weights[gen] if gen >= 0 else 1.0)
            g["gen"].append(gen)
            g["matchup"].append(mu)
        if variant_of is not None:
            variant_col.append(np.full(len(g["y"]), variant_of[path], np.float64))
        if "aux" in counts:
            width = len(quick_turns["aux_offsets"])
            aux_cols.append(place(np.array(counts["aux"], np.float32).reshape(len(rows), width)))
        for k, dtype in DTYPES.items():  # compact per game: lists of Python floats cost ~8x more
            arr = np.array(g[k], dtype=dtype).reshape(len(g["y"]), *SHAPES.get(k, ()))
            cols[k].append(arr if dtype == np.float64 else place(arr))  # MPS has no float64
    out: dict[str, Any] = {}
    for k in DTYPES:  # one column at a time, dropping its pieces: never two copies of everything
        if not cols[k]:
            empty = np.zeros((0, *SHAPES.get(k, ())), DTYPES[k])
            out[k] = empty if DTYPES[k] == np.float64 else place(empty)
        else:
            out[k] = torch.cat(cols[k]) if on_device and DTYPES[k] != np.float64 else np.concatenate(cols[k])
        cols[k] = []
        if on_device and torch.cuda.is_available():
            # The caching allocator keeps the freed pieces, and on Windows (WDDM) device memory
            # counts against the system commit limit: hand them back.
            torch.cuda.empty_cache()
    if (quick_turns or {}).get("aux_offsets"):
        width = len(quick_turns["aux_offsets"])
        if not aux_cols:
            out["aux_search"] = place(np.zeros((0, width), np.float32))
        else:
            out["aux_search"] = torch.cat(aux_cols) if on_device else np.concatenate(aux_cols)
    out["games"] = games
    if cand_rows is not None:
        out.update(cand_rows.arrays())
    if variant_of is not None:
        out["variant"] = np.concatenate(variant_col) if variant_col else np.zeros(0, np.float64)
        out["variant_games"] = variant_games
    out["v3_fraction"] = sum(v3_states) / sum(n_states) if sum(n_states) else 0.0
    out["v3_fraction_by_gen"] = [v / n if n else None for v, n in zip(v3_states[:n_gens], n_states[:n_gens])]
    out["explored"] = explored
    out["value_only"] = value_only
    if split == "train":
        out["outcome_coverage"] = {name: round(o / d, 4) if d else None
                                   for (_, _, name), o, d in zip(PHASES, outcome_targets, decisions)}
        out["outcome_coverage"]["all"] = round(sum(outcome_targets) / sum(decisions), 4) if sum(decisions) else None
    return out


def _standardization(*arrays: Any, chunk: int = 1 << 14) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std (+1e-3) over the rows of all `arrays` (numpy or torch), in float64
    and in chunks: the same numbers as concatenating them and calling mean/std, without the copies."""
    def rows():
        for a in arrays:
            for i in range(0, len(a), chunk):
                c = a[i:i + chunk]
                yield c.cpu().double().numpy() if isinstance(c, torch.Tensor) else c.astype(np.float64)
    n = sum(len(a) for a in arrays)
    mean = sum(c.sum(axis=0) for c in rows()) / max(1, n)
    sq = sum(((c - mean) ** 2).sum(axis=0) for c in rows())
    return mean.astype(np.float32), (np.sqrt(sq / max(1, n)) + 1e-3).astype(np.float32)


CAND_HIDDEN = 64  # width of the candidate heads (CandHeads)


class CandHeads(nn.Module):
    """The candidate heads (--aux-cand-value-weight, --aux-mix-weight): each candidate embedded as
    yomibench.policy encodes it (standardized numbers, action and data-shape embeddings; two layers),
    combined with a projection of the trunk's last hidden layer for the deciding seat's view through
    [state, candidate, state * candidate] and a joint layer, read out as (search value in [-1, 1], mix
    logit). valuenet.candidate_heads is the same computation in numpy. Training only."""

    def __init__(self, hidden: int, sizes: dict[str, int], cmean: np.ndarray, cstd: np.ndarray):
        super().__init__()
        self.register_buffer("cmean", torch.tensor(cmean))
        self.register_buffer("cstd", torch.tensor(cstd))
        self.emb = nn.ModuleDict({cat: nn.Embedding(sizes[cat], CEMBED[cat]) for cat in CAND_CATEGORIES})
        w = CAND_HIDDEN
        self.c1, self.c2 = nn.Linear(N_CAND + sum(CEMBED[c] for c in CAND_CATEGORIES), w), nn.Linear(w, w)
        self.s, self.j = nn.Linear(hidden, w), nn.Linear(3 * w, w)
        self.value, self.logit = nn.Linear(w, 1), nn.Linear(w, 1)

    def forward(self, h: torch.Tensor, cnums: torch.Tensor, cact: torch.Tensor, cdkey: torch.Tensor):
        """`h` (n x hidden), `cnums` (n x K x N_CAND, mirrored into h's view), ids (n x K) ->
        (values, logits), n x K."""
        x = torch.where(self.cstd <= CONSTANT_STD, torch.zeros_like(cnums), (cnums - self.cmean) / self.cstd)
        c = torch.cat([x, self.emb["action"](cact), self.emb["dkey"](cdkey)], dim=-1)
        c = torch.relu(self.c2(torch.relu(self.c1(c))))
        st = torch.relu(self.s(h))[:, None, :].expand_as(c)
        j = torch.relu(self.j(torch.cat([st, c, st * c], dim=-1)))
        return torch.tanh(self.value(j))[..., 0], self.logit(j)[..., 0]

    def export(self) -> dict[str, np.ndarray]:
        """Under valuenet's `aux_cand_` keys, which play never reads."""
        g = lambda t: t.detach().cpu().numpy().copy()  # noqa: E731
        p = {f"{CAND}cmean": g(self.cmean), f"{CAND}cstd": g(self.cstd)}
        for cat in CAND_CATEGORIES:
            p[f"{CAND}emb_{cat}"] = g(self.emb[cat].weight)
        for name in CAND_LAYERS:
            layer = getattr(self, name)
            p[f"{CAND}{name}_w"], p[f"{CAND}{name}_b"] = g(layer.weight).T.copy(), g(layer.bias)
        return p


class Net(nn.Module):
    """`aux`: a second output layer on the shared trunk (the auxiliary hp-margin head). `layers`: hidden
    layers (2, the default, is the production shape). `aux_search`: that many auxiliary search-value
    heads (--aux-search-heads), one output layer on the trunk; training only, play never reads them.
    `cand`: (candidate vocabulary sizes, candidate mean, std) for the candidate heads (CandHeads),
    created last so the other layers draw the same initial weights as without them."""

    def __init__(self, sizes: dict[str, int], mean: np.ndarray, std: np.ndarray, hidden: int = 256,
                 aux: bool = False, layers: int = 2, aux_search: int = 0, cand: tuple | None = None):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.emb = nn.ModuleDict({cat: nn.Embedding(sizes[cat], EMBED[cat]) for cat in EMBED})
        width = N_NUMERIC + sum(EMBED[c] for c in CATEGORY_OF_SLOT)
        # Created in this order so that the default shape draws the same initial weights as before.
        self.l1, self.l2, self.l3 = nn.Linear(width, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)
        self.l3_aux = nn.Linear(hidden, 1) if aux else None
        self.deep = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 2)])  # hidden layers 3..
        self.l_search = nn.Linear(hidden, aux_search) if aux_search else None
        self.cand = CandHeads(hidden, *cand) if cand is not None else None

    def forward(self, nums: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        """The value head (what play uses)."""
        return torch.tanh(self.l3(self.trunk(nums, cats)))[:, 0]

    def heads(self, nums: torch.Tensor, cats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(value head, auxiliary head or None) from one pass through the trunk."""
        v, aux, _ = self.all_heads(nums, cats)
        return v, aux

    def all_heads(self, nums: torch.Tensor, cats: torch.Tensor):
        """(value head, auxiliary hp head or None, auxiliary search heads (rows x heads) or None)."""
        return self.outputs(nums, cats)[:3]

    def outputs(self, nums: torch.Tensor, cats: torch.Tensor):
        """all_heads' outputs and the trunk's (the candidate heads' input), from one pass."""
        h = self.trunk(nums, cats)
        aux = torch.tanh(self.l3_aux(h))[:, 0] if self.l3_aux is not None else None
        search = torch.tanh(self.l_search(h)) if self.l_search is not None else None
        return torch.tanh(self.l3(h))[:, 0], aux, search, h

    def trunk(self, nums: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        # Inputs constant in training enter as 0 (their training value), as valuenet.forward does.
        x = [torch.where(self.std <= CONSTANT_STD, torch.zeros_like(nums), (nums - self.mean) / self.std)]
        for slot, cat in enumerate(CATEGORY_OF_SLOT):
            x.append(self.emb[cat](cats[:, slot]))
        h = torch.relu(self.l1(torch.cat(x, dim=1)))
        h = torch.relu(self.l2(h))
        for layer in self.deep:
            h = torch.relu(layer(h))
        return h

    def hidden_layers(self) -> list:
        return [self.l1, self.l2, *self.deep]

    def export(self) -> dict[str, np.ndarray]:
        """The model's parameters under valuenet's keys (valuenet.layer_keys): the default shape
        writes w1 b1 w2 b2 w3 b3 as always; a deeper net writes its output layer as w_out b_out, so
        code from before --layers fails loudly on it (no w3) instead of misreading a hidden layer."""
        g = lambda t: t.detach().cpu().numpy().copy()  # noqa: E731  (a snapshot, not a view)
        p = {"mean": g(self.mean), "std": g(self.std)}
        for cat in EMBED:
            p[f"emb_{cat}"] = g(self.emb[cat].weight)
        hidden_keys, out_keys = layer_keys(len(self.hidden_layers()))
        for (wk, bk), layer in zip([*hidden_keys, out_keys], [*self.hidden_layers(), self.l3]):
            p[wk] = g(layer.weight).T.copy()
            p[bk] = g(layer.bias)
        if self.l3_aux is not None:
            p["w3_aux"], p["b3_aux"] = g(self.l3_aux.weight).T.copy(), g(self.l3_aux.bias)
        if self.l_search is not None:  # keys play never reads (valuenet.forward's value head)
            p[SEARCH_W], p[SEARCH_B] = g(self.l_search.weight).T.copy(), g(self.l_search.bias)
        if self.cand is not None:  # nor these
            p.update(self.cand.export())
        return p


def cand_batch(split: dict[str, Any], rows: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(positions in `rows` with a candidate slot, their slots' columns: flip n x 2, uid / val / prob
    n x 2 x K) for the candidate heads."""
    slot = split["cand_slot"][rows]
    sel = np.flatnonzero(slot >= 0)
    sl = slot[sel]
    return sel, {k: split[f"cand_{k}"][sl] for k in ("flip", "uid", "val", "prob")}


def cand_losses(heads: "CandHeads", h: list, c: dict[str, Any], table: dict[str, Any], w: Any):
    """(value MSE, mix cross-entropy) of the candidate heads on one batch's slots, torch: `h` both
    seats' trunk outputs at the slots' rows (n x hidden each), `c` cand_batch's columns as tensors,
    `table` the candidate table on the device, `w` each slot's row weight. The MSE is over kept
    candidates with a search value, weighted by row; the cross-entropy is per seat and slot, over the
    slot's kept candidates, weighted by row."""
    uid = c["uid"].long()
    valid = uid >= 0
    u = uid.clamp_min(0)
    flip = c["flip"][:, :, None, None]  # mirrors the x columns into each seat's view (policy.flip_columns)
    cnums = table["unums"][u] * torch.where(table["xmask"], flip, torch.ones_like(flip))
    n, k = uid.shape[0], uid.shape[2]
    hh = torch.stack(h, dim=1).reshape(n * 2, -1)
    v, logit = heads(hh, cnums.reshape(n * 2, k, -1), table["uact"][u].reshape(n * 2, k),
                     table["udkey"][u].reshape(n * 2, k))
    v, logit = v.reshape(n, 2, k), logit.reshape(n, 2, k)
    known = valid & torch.isfinite(c["val"])
    wk = torch.where(known, w[:, None, None].expand_as(c["val"]), torch.zeros_like(c["val"]))
    sq = torch.where(known, (v - torch.nan_to_num(c["val"])) ** 2, torch.zeros_like(v))
    mse = torch.sum(wk * sq) / torch.sum(wk).clamp_min(1e-12)
    logp = torch.log_softmax(logit.masked_fill(~valid, -1e9), dim=-1)
    prob = c["prob"]
    ce = -torch.sum(torch.where(prob > 0, prob * logp, torch.zeros_like(logp)), dim=-1)  # n x 2
    has = (prob.sum(dim=-1) > 0).float() * w[:, None]
    return mse, torch.sum(has * ce) / torch.sum(has).clamp_min(1e-12)


def cand_standardization(split: dict[str, Any], table: dict[str, Any], chunk: int = 4096):
    """Mean and std (+1e-3) of the kept candidates' numbers, mirrored into their seats' views, over a
    split's slots (as _standardization)."""
    total, s1 = 0, np.zeros(N_CAND)
    uid, flip = split["cand_uid"], split["cand_flip"]

    def blocks():
        for i in range(0, len(uid), chunk):
            u = uid[i:i + chunk]
            x = flip_columns(table["unums"][np.maximum(u, 0)], flip[i:i + chunk])
            yield x[u >= 0].astype(np.float64)
    for x in blocks():
        total += len(x)
        s1 += x.sum(axis=0)
    mean = s1 / max(1, total)
    sq = sum((((x - mean) ** 2).sum(axis=0) for x in blocks()), np.zeros(N_CAND))
    return mean.astype(np.float32), (np.sqrt(sq / max(1, total)) + 1e-3).astype(np.float32)


def cand_metrics(p: dict[str, np.ndarray], split: dict[str, Any], table: dict[str, Any],
                 chunk: int = 4096) -> dict[str, float | None]:
    """The saved candidate heads (numpy, valuenet.candidate_heads) on a split's slots: MSE against
    the kept candidates' search values, the mix cross-entropy (per seat and slot) and the share of
    seats and slots whose top logit is one of the mix's likeliest candidates."""
    slot = split["cand_slot"]
    rows = np.flatnonzero(slot >= 0)
    sq = nv = ce = nm = top = 0.0
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        _, c = cand_batch(split, r)
        u = np.maximum(c["uid"], 0)
        cnums = flip_columns(table["unums"][u], c["flip"])
        for s in (0, 1):
            h = trunk(p, np.asarray(split[f"nums{s}"][r], np.float32), np.asarray(split[f"cats{s}"][r], np.int64))
            v, logit = candidate_heads(p, h, cnums[:, s], table["uact"][u[:, s]], table["udkey"][u[:, s]])
            valid, val, prob = c["uid"][:, s] >= 0, c["val"][:, s], c["prob"][:, s]
            known = valid & np.isfinite(val)
            sq += float(np.sum(((v - np.nan_to_num(val)) ** 2)[known]))
            nv += int(known.sum())
            logit = np.where(valid, logit, -np.inf)
            logp = logit - logit.max(axis=1, keepdims=True)
            logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
            has = prob.sum(axis=1) > 0
            ce += float(-np.sum(np.where(prob > 0, prob * np.where(valid, logp, 0.0), 0.0)[has]))
            nm += int(has.sum())
            best = np.take_along_axis(prob, np.argmax(logit, axis=1)[:, None], axis=1)[:, 0]  # ties count as hits
            top += int(np.sum((best >= prob.max(axis=1) - 1e-6)[has]))
    return {"val_aux_cand_value_mse": round(sq / nv, 5) if nv else None,
            "val_aux_mix_ce": round(ce / nm, 5) if nm else None,
            "val_aux_mix_top1": round(top / nm, 5) if nm else None}


def report(split: dict[str, Any], pred: np.ndarray, gens: list[str] | None = None) -> list[str]:
    """Validation rows by game phase, then (with `gens`, the generation names) by generation, then
    by matchup when there is more than one."""
    rows = []
    groups = [(name, (split["phase"] >= lo) & (split["phase"] < hi)) for lo, hi, name in PHASES]
    if gens and len(gens) > 1:
        groups += [(f"gen {g}", split["gen"] == k) for k, g in enumerate(gens)]
    codes = sorted(set(np.asarray(split.get("matchup", np.zeros(0))).astype(int).tolist()))
    if len(codes) > 1:
        groups += [(matchup_name(c), split["matchup"] == c) for c in codes]
    for name, m in groups:
        if m.any():
            z = split["outcome"][m]
            net = float(np.mean((pred[m] - z) ** 2))
            base = float(np.mean((split["baseline"][m] - z) ** 2))
            acc = float(np.mean(np.sign(pred[m]) == np.sign(z)))
            rows.append(f"  {name:<6} n={int(m.sum()):>7}  mse net {net:.4f}  evaluate {base:.4f}  "
                        f"net sign accuracy {acc:.1%}")
    return rows


def matchup_mse(split: dict[str, Any], pred: np.ndarray) -> dict[str, dict[str, Any]]:
    """{"P1:P2": {n, net, evaluate}}: MSE against the outcome for each matchup in a split."""
    out = {}
    codes = np.asarray(split.get("matchup", np.zeros(0))).astype(int)
    for c in sorted(set(codes.tolist())):
        m = codes == c
        z = np.asarray(split["outcome"])[m]
        out[matchup_name(c)] = {"n": int(m.sum()), "net": round(float(np.mean((pred[m] - z) ** 2)), 5),
                                "evaluate": round(float(np.mean((np.asarray(split["baseline"])[m] - z) ** 2)), 5)}
    return out


def sign_accuracy(split: dict[str, Any], pred: np.ndarray) -> float | None:
    """The share of a split's positions where the prediction's sign is the outcome's."""
    z = np.asarray(split["outcome"])
    return round(float(np.mean(np.sign(pred) == np.sign(z))), 5) if len(z) else None


def init_vocab(init_from: Path | None) -> Vocab:
    """The run's starting vocabulary: empty, or with --init-from the model's own (names in its id
    order, characters' start values), open for the training folders' new names."""
    if init_from is None:
        return Vocab()
    src = ValueNet.load(init_from).vocab
    return Vocab(ids={cat: dict(t) for cat, t in src.ids.items()},
                 raw_defaults={c: dict(d) for c, d in src.raw_defaults.items()}, frozen=False)


def rescaled_inputs(init_std: np.ndarray, data_std: np.ndarray) -> np.ndarray:
    """Numeric inputs constant in the init model's training data (std at CONSTANT_STD: it always fed
    them 0) that vary in this run's training data: a boolean mask over the N_NUMERIC inputs."""
    return (np.asarray(init_std) <= CONSTANT_STD) & (np.asarray(data_std) > CONSTANT_STD)


def load_init(net: "Net", path: Path, vocab: Vocab,
              data_stats: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    """Start `net` from the saved model at `path` (see the module docstring, --init-from); returns
    what the meta records about it. `data_stats`: this run's training (mean, std); inputs constant
    for the model but varying here (rescaled_inputs) take them, with zero first-layer weights, so the
    started net computes what the model computes. None: the model's standardization throughout."""
    src = ValueNet.load(path)
    p = src.p
    mean, std = p["mean"].copy(), p["std"].copy()
    rescale = np.zeros(len(std), bool)
    if data_stats is not None:
        rescale = rescaled_inputs(std, data_stats[1])
        mean[rescale] = np.asarray(data_stats[0], np.float32)[rescale]
        std[rescale] = np.asarray(data_stats[1], np.float32)[rescale]
    with torch.no_grad():
        net.mean.copy_(torch.from_numpy(mean))
        net.std.copy_(torch.from_numpy(std))
        hidden_keys, out_keys = layer_keys(n_layers(p))
        for (wk, bk), layer in zip([*hidden_keys, out_keys], [*net.hidden_layers(), net.l3]):
            layer.weight.copy_(torch.from_numpy(p[wk].T.copy()))
            layer.bias.copy_(torch.from_numpy(p[bk]))
        # numeric inputs come first in l1's input (Net.trunk); a re-scaled one's weights were never trained
        net.l1.weight[:, torch.from_numpy(np.flatnonzero(rescale))] = 0.0
        if net.l3_aux is not None and "w3_aux" in p:
            net.l3_aux.weight.copy_(torch.from_numpy(p["w3_aux"].T.copy()))
            net.l3_aux.bias.copy_(torch.from_numpy(p["b3_aux"]))
        if net.l_search is not None and SEARCH_W in p and p[SEARCH_W].shape == tuple(net.l_search.weight.T.shape):
            net.l_search.weight.copy_(torch.from_numpy(p[SEARCH_W].T.copy()))
            net.l_search.bias.copy_(torch.from_numpy(p[SEARCH_B]))
        for cat in EMBED:  # rows by name: this run's ids may differ from the model's
            table, old = p[f"emb_{cat}"], src.vocab.ids.get(cat, {})
            emb = net.emb[cat].weight
            emb[0] = torch.from_numpy(table[0])
            for name, i in vocab.ids.get(cat, {}).items():
                if i >= len(emb):
                    continue
                # a name the model never saw starts as the model's unknown row (0): what it encoded it as, so
                # the warm start computes what the model computes on it until training moves the row
                j = old.get(name, 0)
                emb[i] = torch.from_numpy(table[min(j, len(table) - 1)])  # forward() clamps ids past the table
    return {"path": str(path), "sha256": evalset.file_sha256(Path(path)), "name": src.meta.get("name"),
            "rescaled": [NUMERIC_NAMES[k] for k in np.flatnonzero(rescale)]}


def check_init(path: Path, hidden: int, residual_hp: bool, layers: int = 2) -> str | None:
    """Why the model at `path` can't start this run (feature version, hidden size, depth, residual_hp), or None."""
    src = ValueNet.load(path)
    if src.version != FEATURES:
        return f"--init-from model has feature version {src.version}; training writes version {FEATURES}"
    if n_layers(src.p) != layers:
        return f"--init-from model has {n_layers(src.p)} hidden layers, not --layers {layers}"
    if src.p["w2"].shape[0] != hidden:
        return f"--init-from model has hidden size {src.p['w2'].shape[0]}, not --hidden {hidden}"
    if src.residual_hp != residual_hp:
        return f"--init-from model has residual_hp {src.residual_hp}; this run has --residual-hp {residual_hp}"
    return None


def _phase_mse(split: dict[str, Any], pred: np.ndarray) -> dict[str, float | None]:
    """{early, mid, late, all}: the prediction's MSE against the outcome."""
    out = {}
    for lo, hi, name in (*PHASES, (0.0, 1.01, "all")):
        m = (split["phase"] >= lo) & (split["phase"] < hi)
        out[name] = round(float(np.mean((pred[m] - split["outcome"][m]) ** 2)), 5) if m.any() else None
    return out


def parse_offsets(text: str | None) -> list[int]:
    """--aux-search-heads "1,4,16" -> [1, 4, 16]; None -> [] (no heads). ValueError when malformed."""
    if text is None:
        return []
    try:
        out = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        out = []
    if not out or min(out) < 1 or len(set(out)) != len(out):
        raise ValueError(f"--aux-search-heads needs distinct decision offsets of at least 1, e.g. 1,4,16 (not {text!r})")
    return out


def dry_run(args: argparse.Namespace) -> None:
    """--dry-run: the files a training would read (select_files, then --record-filter) and the
    compatibility checks it would make, without parsing a turn or training. SystemExit (nonzero)
    when the window would be refused, e.g. mixed candidate sets without --allow-mixed-candidate-sets."""
    chunks, _, _ = select_files(args.dirs, args.val_fraction, args.seed, args.games_per_gen, args.eval_dir,
                                args.heldout_root, args.heldout_seed)
    files = [p for c in chunks.values() for p in c]
    by_file = {p: read_header(p) for p in files}
    allow = args.allow_match_settings
    if compat.excludes_sampled_starts(args.record_filter, args.exclude_sampled_starts):
        compat.drop_sampled_starts(chunks, lambda f: is_sampled(by_file[f]), partial(print, flush=True))
    compat.apply_record_filter(chunks, lambda f: compat.record_keys(by_file[f]), args.record_filter,
                               **({"allow_match_settings": allow} if allow else {}))
    if args.window_budget_games is not None:
        sample_window(chunks, lambda f: Path(f).stat().st_mtime, lambda f: compat.record_keys(by_file[f]),
                      args.window_recent_hours, args.window_budget_games, args.window_seed, partial(print, flush=True))
    files = [p for c in chunks.values() for p in c]
    compat.refuse_held_out([(p, by_file[p]) for p in files])
    keys = [compat.record_keys(by_file[p]) for p in files]
    check_compatible([by_file[p] for p in files], args.allow_mixed_candidate_sets, allow)
    if allow:
        counts = [0] * (len(allow) + 1)
        for k in keys:
            i = compat.match_variant(k, allow)
            counts[-1 if i is None else i] += 1
        print("dry run: by match settings: " + ", ".join(
            f"{compat.variant_label(v)} {n}" for v, n in zip(allow, counts)) + (f", other {counts[-1]}" if counts[-1] else ""),
            flush=True)
    sets: dict[str, int] = {}
    for _, cset in keys:
        sets[cset] = sets.get(cset, 0) + 1
    print(f"dry run: {len(files)} records would train together (" + ", ".join(f"{k} {len(v)}" for k, v in chunks.items())
          + "; candidate sets " + ", ".join(f"{k} x{n}" for k, n in sorted(sets.items()))
          + f"; {len({g for g, _ in keys})} game key(s)); no training", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="yomibench.train")
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--early-stop-patience", type=int, default=0, metavar="N",
                    help="stop when validation MSE hasn't improved for N epochs after the best one (0: off, train all "
                         "--epochs); with --select best the best epoch is kept either way")
    ap.add_argument("--progress-seconds", type=float, default=60.0, metavar="S",
                    help="print a 'progress: epoch E batch B/N, samples/s' line every S seconds of training (0: off), "
                         "so a watcher can tell a slow fit from a hung one")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-fraction", type=float, default=0.15,
                    help="share of games a folder's held-out manifest holds out when first created (an existing "
                         "manifest keeps its own)")
    ap.add_argument("--heldout-seed", type=int, default=0,
                    help="seed of a new held-out manifest's assignment (an existing manifest keeps its own)")
    ap.add_argument("--heldout-root", type=Path, default=None,
                    help="keep held-out manifests here instead of in each record folder (for read-only folders)")
    ap.add_argument("--eval-dir", type=Path, action="append", default=[],
                    help="games for evaluation only, made with exploration off; never trained on (repeatable)")
    ap.add_argument("--seed", type=int, default=0, help="initialization, batch order and --games-per-gen sample")
    ap.add_argument("--name", default=None, help="model name (default: the output file's stem)")
    ap.add_argument("--value-weight", type=float, default=0.0,
                    help="blend the search's recorded value into the target: (1-w)*outcome + w*value")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=None, help="record-parsing processes (default: up to 8)")
    ap.add_argument("--window", type=int, default=None, metavar="N",
                    help="train on the last N of the given folders (one folder per generation, oldest first)")
    ap.add_argument("--games-per-gen", type=int, default=None, metavar="K",
                    help="at most K records from each folder (a seeded sample)")
    ap.add_argument("--gen-weights", default=None, metavar="W,W,...",
                    help="loss weight of each folder's samples, oldest first, after --window (default 1 each)")
    ap.add_argument("--explore-targets", choices=EXPLORE_TARGETS, default="uniform-only",
                    help="which exploratory picks put the positions before them on value-only targets: "
                         "uniform-only (opening noise; softmax picks keep outcome targets), all, or none "
                         "(default uniform-only; validation always drops positions before uniform picks)")
    ap.add_argument("--cheap-target", choices=CHEAP_TARGETS, default="outcome",
                    help="training target of quick (playout-capped) turns: outcome (default: as a turn without a "
                         "search value) or next-full (blend toward the next full turn's search value)")
    ap.add_argument("--cheap-value-weight", type=float, default=None, metavar="W",
                    help="with --cheap-target next-full: the blend weight of that value (default --value-weight)")
    ap.add_argument("--full-turns-only", action="store_true",
                    help="train on full-search turns only (quick turns dropped from the training split; validation "
                         "and evaluation unchanged)")
    ap.add_argument("--cache", type=Path, default=None, metavar="ROOT",
                    help="cache root for read-only record folders (default: each folder's .cache/; see yomibench.cache)")
    ap.add_argument("--in-memory", action="store_true",
                    help="the previous loader: parse and encode every record into memory each run (kept for one "
                         "release; the cache trains the same model up to float16 rounding of the inputs)")
    ap.add_argument("--allow-mixed-candidate-sets", action="store_true",
                    help="train on records from different candidate sets together (refused by default)")
    ap.add_argument("--record-filter", type=json.loads, default=None, metavar="JSON",
                    help='only the records matching this, in every split: {"candidate_set": "5", "match_settings": '
                         '{"stage_width": 600, ...}} (settings the filter doesn\'t name must be standard), optionally '
                         '"any_candidate_set_dirs": [folders whose records need only match the settings], and '
                         '"candidate_sets": ["5", "6"] (other sets kept too; needs --allow-mixed-candidate-sets); the rest are '
                         'excluded and counted (yomibench.compat.apply_record_filter). The round-3 trainer passes it '
                         'once the fleet manifest pins match settings or a candidate set')
    ap.add_argument("--exclude-sampled-starts", action="store_true",
                    help="leave sampled-start records (header start.kind sampled: self-play games begun mid-game "
                         "from a recent record's state, yomibench.sampled_starts) out of every split, e.g. when "
                         "building or scoring a holdout that must hold games from the opening only; the record "
                         "filter's \"exclude_sampled_starts\": true does the same. Default: they train at weight 1")
    ap.add_argument("--allow-match-settings", default=None, metavar="JSON-LIST",
                    help='match-settings variants that may share the window (and pass --record-filter), e.g. '
                         '\'[{"ceiling_height": 800}, {"ceiling_height": 500, "has_ceiling": true}]\': a setting any '
                         'variant names may vary between them (standard value where a variant doesn\'t name it); every '
                         'other setting must still be identical. Validation is reported per variant. Default: none '
                         '(records of different match settings are refused)')
    ap.add_argument("--dry-run", action="store_true",
                    help="pick the files (held-out manifests, --record-filter) and check the records can train "
                         "together (candidate sets, game keys), then exit without training")
    ap.add_argument("--select", choices=SELECT, default="best",
                    help="which epoch to keep: best (lowest validation MSE; the default) or final (the last epoch)")
    ap.add_argument("--aux-hp-weight", type=float, default=0.0, metavar="W",
                    help="add an auxiliary head trained on the final hp margin, weight W in the loss (default 0: none)")
    ap.add_argument("--residual-hp", action="store_true",
                    help="the net predicts a correction to the hp-lead evaluator: value = clip(evaluate + net)")
    ap.add_argument("--init-from", type=Path, default=None, metavar="MODEL.npz",
                    help="fine-tune: start from this saved model's weights (same features, hidden size, layers, "
                         "residual_hp)")
    ap.add_argument("--target-nstep", type=int, default=None, metavar="N",
                    help="multi-step targets: blend in the mean of the next N full-turn search values in the game "
                         "(the turn's own first, when it has one) instead of the turn's own; the window stops at a "
                         "uniform/least-tried pick (included). Training split only; default off")
    ap.add_argument("--target-lambda", type=float, default=None, metavar="L",
                    help="multi-step targets, the other design: the mean of the later full-turn search values "
                         "weighted L^i (L in [0, 1]; normalized over the window, which stops at a uniform/least-tried "
                         "pick); no discount on the outcome. Exclusive with --target-nstep; default off")
    ap.add_argument("--aux-search-heads", default=None, metavar="D,D,...",
                    help="auxiliary heads predicting the full-turn search value D decisions later (the nearest full "
                         "turn at or after k+D; masked past the game's end or a uniform/least-tried pick), e.g. "
                         "1,4,16; side losses only: play never reads them. Default: none")
    ap.add_argument("--aux-search-weight", type=float, default=0.1, metavar="W",
                    help="with --aux-search-heads: the heads' (mean) loss weight (default 0.1)")
    ap.add_argument("--aux-cand-value-weight", type=float, default=0.0, metavar="W",
                    help="candidate heads: regress each candidate's search value at full turns (both seats; a head "
                         "on the trunk and a candidate embedding), weight W in the loss (default 0: off). Side loss "
                         "only: play never reads it")
    ap.add_argument("--aux-mix-weight", type=float, default=0.0, metavar="W",
                    help="candidate heads: softmax over each seat's candidates at full turns, cross-entropy against "
                         "the search's solved mix, weight W (default 0: off). Side loss only")
    ap.add_argument("--aux-cand-k", type=int, default=DEFAULT_K, metavar="K",
                    help=f"with the candidate heads: candidates kept per seat per full turn (default {DEFAULT_K}): the "
                         "mix's whole support (its K likeliest if larger), then a seeded sample of the rest; about 12 "
                         "bytes per kept candidate (the mix is renormalized over the kept list)")
    ap.add_argument("--layers", type=int, default=2, metavar="N",
                    help="hidden layers of --hidden units (default 2, the production shape; more for capacity tests: "
                         "such a model saves w_out instead of w3, so only code with this option can play it)")
    ap.add_argument("--window-budget-games", type=int, default=None, metavar="B",
                    help="sample window (off by default): train on at most B games, every training game from the "
                         "last --window-recent-hours plus a seeded uniform sample of the older ones (validation "
                         "kept complete); see the module docstring")
    ap.add_argument("--window-recent-hours", type=float, default=0.0, metavar="H",
                    help="with --window-budget-games: keep every training game from the H hours before the newest "
                         "game (default 0: a uniform sample of the whole window)")
    ap.add_argument("--window-seed", type=int, default=0, metavar="S",
                    help="with --window-budget-games: the older games' sample (default 0)")
    ap.add_argument("--share-window", type=Path, default=None, metavar="DIR",
                    help="shared window (cached loader only; off by default): reuse the selected window another run "
                         "with the same data arguments wrote under DIR when the records are unchanged, else select as "
                         "usual and write it there; see 'Shared windows' in the module docstring")
    ap.add_argument("--share-window-frozen", action="store_true",
                    help="with --share-window: also reuse a window written before games arrived (its games' cache "
                         "entries unchanged); this run then trains on that window, not on the newer games")
    ledger.add_args(ap)
    args = ap.parse_args(argv)
    if args.allow_match_settings is not None:
        try:
            args.allow_match_settings = compat.parse_allow_match_settings(args.allow_match_settings)
        except ValueError as e:
            ap.error(str(e))
    given_dirs = list(args.dirs)  # before --window, for the ledger row
    if args.window is not None:
        if args.window < 1:
            ap.error("--window must be at least 1")
        args.dirs = args.dirs[-args.window:]
    if args.games_per_gen is not None and args.games_per_gen < 1:
        ap.error("--games-per-gen must be at least 1")
    window = None
    if args.window_budget_games is not None:
        if args.window_budget_games < 1 or args.window_recent_hours < 0:
            ap.error("--window-budget-games must be at least 1 and --window-recent-hours nonnegative")
        window = {"recent_hours": args.window_recent_hours, "budget": args.window_budget_games,
                  "seed": args.window_seed}
    elif args.window_recent_hours or args.window_seed:
        ap.error("--window-recent-hours and --window-seed need --window-budget-games")
    if args.share_window is not None and args.in_memory:
        ap.error("--share-window shares the cached loader's window; --in-memory has none")
    if args.share_window_frozen and args.share_window is None:
        ap.error("--share-window-frozen needs --share-window")
    if args.aux_hp_weight < 0:
        ap.error("--aux-hp-weight must be nonnegative")
    if args.full_turns_only and args.cheap_target != "outcome":
        ap.error("--full-turns-only drops quick turns; --cheap-target gives them targets: pick one")
    if args.cheap_value_weight is not None and (args.cheap_target != "next-full" or not 0 <= args.cheap_value_weight <= 1):
        ap.error("--cheap-value-weight is a weight in [0, 1] for --cheap-target next-full")
    cheap = {"cheap_target": args.cheap_target, "cheap_value_weight": args.cheap_value_weight,
             "full_turns_only": args.full_turns_only}
    if args.target_nstep is not None and args.target_lambda is not None:
        ap.error("--target-nstep and --target-lambda are two designs of the multi-step target: pick one")
    if args.target_nstep is not None and args.target_nstep < 1:
        ap.error("--target-nstep must be at least 1")
    if args.target_lambda is not None and not 0 <= args.target_lambda <= 1:
        ap.error("--target-lambda is a weight in [0, 1]")
    try:
        aux_offsets = parse_offsets(args.aux_search_heads)
    except ValueError as e:
        ap.error(str(e))
    if args.aux_search_weight < 0:
        ap.error("--aux-search-weight must be nonnegative")
    if args.layers < 2:
        ap.error("--layers must be at least 2")
    if args.aux_cand_value_weight < 0 or args.aux_mix_weight < 0:
        ap.error("--aux-cand-value-weight and --aux-mix-weight must be nonnegative")
    if args.aux_cand_k < 1:
        ap.error("--aux-cand-k must be at least 1")
    cand_k = args.aux_cand_k if args.aux_cand_value_weight > 0 or args.aux_mix_weight > 0 else None
    # Multi-step and auxiliary-search settings reach select_rows only when on: off, its calls are as before.
    multistep = {k: v for k, v in (("target_nstep", args.target_nstep), ("target_lambda", args.target_lambda))
                 if v is not None}
    targets = {**cheap, **multistep, **({"aux_offsets": aux_offsets} if aux_offsets else {})}
    if args.init_from is not None:
        try:
            why = check_init(args.init_from, args.hidden, args.residual_hp, args.layers)
        except (OSError, ValueError, KeyError) as e:
            why = f"--init-from {args.init_from}: {e}"
        if why:
            ap.error(why)
    gen_weights = [float(x) for x in args.gen_weights.split(",")] if args.gen_weights else None
    if gen_weights is not None and (len(gen_weights) != len(args.dirs) or min(gen_weights) < 0 or not sum(gen_weights)):
        ap.error(f"--gen-weights needs {len(args.dirs)} nonnegative weights, not all zero (one per folder trained on)")
    try:
        tags = ledger.parse_tags(args.tag)
    except ValueError as e:
        ap.error(str(e))
    ledger_path = ledger.path_from_args(args)
    if ledger_path:
        started, cpu, memory = ledger.utc_now(), ledger.cpu_load(), ledger.MemorySampler().start()

    if args.dry_run:
        dry_run(args)
        return
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    vocab = init_vocab(args.init_from)
    phases = PhaseTimer()  # metrics.phase_seconds: where the fit's time went (load_seconds/train_seconds kept)
    t_load = time.perf_counter()
    if args.in_memory:
        with phases("load_in_memory"):  # extract, encode and split in one pass: not split further
            data = load_samples(args.dirs, vocab, args.val_fraction, args.seed, args.value_weight, args.workers, device,
                                args.allow_mixed_candidate_sets, args.games_per_gen, gen_weights, args.explore_targets,
                                args.eval_dir, args.heldout_root, args.heldout_seed, quick_turns=targets,
                                record_filter=args.record_filter,
                                **({"allow_match_settings": args.allow_match_settings} if args.allow_match_settings else {}),
                                **({"cand_k": cand_k} if cand_k else {}), **({"window": window} if window else {}),
                                **({"exclude_sampled_starts": True} if args.exclude_sampled_starts else {}))
    else:  # the same splits and rows from yomibench.cache, input columns memory-mapped
        from yomibench.cache import open_cached

        data = open_cached(args.dirs, vocab, args.val_fraction, args.seed, args.value_weight, args.workers, device,
                           args.allow_mixed_candidate_sets, args.games_per_gen, gen_weights, args.explore_targets,
                           args.eval_dir, args.heldout_root, args.heldout_seed, root=args.cache,
                           hp_margin=args.aux_hp_weight > 0, quick_turns=targets, record_filter=args.record_filter,
                           **({"allow_match_settings": args.allow_match_settings} if args.allow_match_settings else {}),
                           **({"cand_k": cand_k} if cand_k else {}), **({"window": window} if window else {}),
                           **({"exclude_sampled_starts": True} if args.exclude_sampled_starts else {}),
                           phases=phases, share_dir=args.share_window, share_frozen=args.share_window_frozen)
    tr, va, ev = data["train"], data["val"], data["eval"]
    load_s = time.perf_counter() - t_load
    print(f"train: {tr['games']} games, {len(tr['y'])} samples; val (held out): {va['games']} games, "
          f"{len(va['y'])} samples" + (f"; eval: {ev['games']} games, {len(ev['y'])} samples" if args.eval_dir else "")
          + f" (loaded in {load_s:.1f}s)", flush=True)
    for d, m in data["heldout"].items():
        print(f"  {d}: held-out manifest {m['manifest']} ({m['heldout']} held out, {m['training']} training"
              + (f", {m['added']} newly assigned" if m["added"] else "") + f"; sha256 {m['sha256'][:12]})", flush=True)
    print(f"states with engine-v3 move fields: {tr['v3_fraction']:.1%} of training samples"
          + (f" (by generation: {', '.join('-' if f is None else f'{f:.0%}' for f in tr['v3_fraction_by_gen'])})"
             if len(args.dirs) > 1 else ""), flush=True)
    value_only_fraction = tr["value_only"] / len(tr["y"]) if len(tr["y"]) else 0.0
    print(f"value-only targets (--explore-targets {args.explore_targets}): {tr['value_only']} of {len(tr['y'])} "
          f"training positions ({value_only_fraction:.1%}); {va['explored']} validation positions dropped "
          f"(at or before a uniform pick)", flush=True)
    cov = tr["outcome_coverage"]
    print("training decisions with outcome targets: "
          + ", ".join(f"{k} {'-' if v is None else f'{v:.1%}'}" for k, v in cov.items()), flush=True)
    t_setup = time.perf_counter()  # standardization, the net, init and the training tensors
    # this run's standardization; with --init-from, load_init keeps the model's own except for inputs the
    # model saw as constant that vary here (re-scaled to these statistics, weights starting at 0)
    mean, std = _standardization(tr["nums0"], tr["nums1"])
    aux = args.aux_hp_weight > 0
    cand = None  # the candidate heads' settings and data (--aux-cand-value-weight / --aux-mix-weight)
    if cand_k:
        ctab = data["cand_table"]
        cmean, cstd = cand_standardization(tr, ctab)
        csizes = {c: len(ctab["vocab"][c]) + 1 for c in CAND_CATEGORIES}
        cand = {"table": ctab, "dev": {
            "unums": torch.from_numpy(ctab["unums"]).to(device), "uact": torch.from_numpy(ctab["uact"]).to(device),
            "udkey": torch.from_numpy(ctab["udkey"]).to(device),
            "xmask": torch.from_numpy(np.isin(np.arange(N_CAND), X_COLUMNS)).to(device)}}
        n_slots = int((tr["cand_slot"] >= 0).sum())
        print(f"candidate heads (value weight {args.aux_cand_value_weight}, mix weight {args.aux_mix_weight}, "
              f"K {cand_k}): {n_slots} of {len(tr['y'])} training positions are full turns with candidate lists; "
              f"{int((tr['cand_uid'] >= 0).sum())} kept candidates, {len(ctab['unums'])} unique", flush=True)
    net = Net(vocab.sizes(), mean, std, args.hidden, aux=aux, layers=args.layers,
              aux_search=len(aux_offsets), cand=(csizes, cmean, cstd) if cand else None).to(device)
    init, rescaled = None, []
    if args.init_from is not None:
        init = load_init(net, args.init_from, vocab, data_stats=(mean, std))
        rescaled = init.pop("rescaled")
        print(f"init-constant features re-scaled: {len(rescaled)}" + (f" ({', '.join(rescaled)})" if rescaled else ""),
              flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t = {k: tr.pop(k) for k in ("nums0", "cats0", "nums1", "cats1", "y", "w")}
    on_device = lambda a: torch.as_tensor(np.asarray(a, np.float32)).to(device)  # noqa: E731
    if args.residual_hp:  # evaluate(state, 0) per training row, the residual's base
        t["baseline"] = on_device(tr["baseline"])
    if aux:
        hp = tr.pop("hp_margin")
        t["hp_margin"] = hp if isinstance(hp, torch.Tensor) else on_device(hp)
        known = int(torch.isfinite(t["hp_margin"]).sum())
        print(f"auxiliary hp-margin head (weight {args.aux_hp_weight}): final hp margin known for {known} of "
              f"{len(t['y'])} training positions", flush=True)
    if aux_offsets:
        sa = tr.pop("aux_search")
        t["aux_search"] = sa if isinstance(sa, torch.Tensor) else on_device(sa)
        known_s = torch.isfinite(t["aux_search"]).sum(dim=0).tolist()
        print(f"auxiliary search heads (weight {args.aux_search_weight}): target known for "
              + ", ".join(f"+{d}: {int(n)}" for d, n in zip(aux_offsets, known_s))
              + f" of {len(t['y'])} training positions", flush=True)
    if multistep:
        print("multi-step search targets: " + ", ".join(f"{k} {v}" for k, v in multistep.items()), flush=True)
    from yomibench.cache import chunks_of

    def predict(split):  # p1's values over a split, in bounded chunks (one for an in-memory split)
        out = [value({k: torch.from_numpy(x).to(device) for k, x in c.items()}).cpu().numpy()
               for c in chunks_of(split)]
        corr = np.concatenate(out) if out else np.zeros(0)
        return residual_value(corr, np.asarray(split["baseline"])) if args.residual_hp else corr

    def value(d, idx=slice(None)):  # p1's value, antisymmetric (the correction, with --residual-hp)
        return (net(d["nums0"][idx], d["cats0"][idx]) - net(d["nums1"][idx], d["cats1"][idx])) / 2

    def heads(d, idx):  # p1's value (unclipped, with --residual-hp), aux and search-head outputs, antisymmetric;
        v0, a0, s0, h0 = net.outputs(d["nums0"][idx], d["cats0"][idx])  # and both views' trunk outputs
        v1, a1, s1, h1 = net.outputs(d["nums1"][idx], d["cats1"][idx])
        v = (v0 - v1) / 2
        if args.residual_hp:
            v = v + d["baseline"][idx]
        return v, (a0 - a1) / 2 if aux else None, (s0 - s1) / 2 if aux_offsets else None, (h0, h1)
    best = (float("inf"), None, -1, None)  # (val mse, weights, epoch, eval mse): lowest validation MSE
    final = best  # the same for the last epoch (weights only with --select final)
    phases.add("setup", time.perf_counter() - t_setup)
    t0 = time.perf_counter()
    epochs_run = 0
    n_batches = (len(t["y"]) + args.batch - 1) // args.batch
    next_note = time.monotonic() + args.progress_seconds
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(t["y"]), device=device)
        e0 = time.monotonic()
        for i in range(0, len(perm), args.batch):
            if args.progress_seconds > 0 and time.monotonic() >= next_note:  # a liveness line for watchers
                b = i // args.batch
                print(f"progress: epoch {epoch} batch {b}/{n_batches} ({b / max(n_batches, 1):.0%}), "
                      f"{i / max(time.monotonic() - e0, 1e-9):.0f} samples/s", flush=True)
                next_note = time.monotonic() + args.progress_seconds
            idx = perm[i:i + args.batch]
            w = t["w"][idx]
            v, a, sh, hs = heads(t, idx)
            loss = torch.sum(w * (v - t["y"][idx]) ** 2) / torch.sum(w).clamp_min(1e-12)
            if cand:  # the candidate heads on the batch's full turns
                sel, cb = cand_batch(tr, idx.cpu().numpy())
                if len(sel):
                    at = torch.from_numpy(sel).to(device)
                    cmse, cce = cand_losses(net.cand, [hs[0][at], hs[1][at]],
                                            {k: torch.from_numpy(x).to(device) for k, x in cb.items()}, cand["dev"],
                                            w[at])
                    loss = loss + args.aux_cand_value_weight * cmse + args.aux_mix_weight * cce
            if aux_offsets:  # each head's masked weighted MSE, averaged over the heads
                st = t["aux_search"][idx]
                known = torch.isfinite(st)
                ws = torch.where(known, w[:, None], torch.zeros_like(st))
                sq = torch.where(known, (sh - torch.nan_to_num(st)) ** 2, torch.zeros_like(sh))
                per_head = torch.sum(ws * sq, dim=0) / torch.sum(ws, dim=0).clamp_min(1e-12)
                loss = loss + args.aux_search_weight * per_head.mean()
            if aux:
                hp = t["hp_margin"][idx]
                wa = torch.where(torch.isfinite(hp), w, torch.zeros_like(w))
                sq = torch.where(torch.isfinite(hp), (a - torch.nan_to_num(hp)) ** 2, torch.zeros_like(a))
                loss = loss + args.aux_hp_weight * torch.sum(wa * sq) / torch.sum(wa).clamp_min(1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
        phases.add("train_epochs", time.monotonic() - e0)
        net.eval()
        with phases("epoch_eval"), torch.no_grad():  # each epoch's validation and evaluation predictions
            pred = predict(va)
            pred_e = predict(ev)
        val = float(np.mean((pred - va["outcome"]) ** 2)) if len(va["y"]) else loss.item()
        ev_mse = float(np.mean((pred_e - ev["outcome"]) ** 2)) if len(ev["y"]) else None
        print(f"epoch {epoch}: last batch loss {loss.item():.4f}, val mse {val:.4f}"
              + (f", eval mse {ev_mse:.4f}" if ev_mse is not None else ""), flush=True)
        improved = val < best[0]  # the best epoch is chosen on validation; evaluation only reports
        final = (val, net.export() if improved or args.select == "final" else None, epoch, ev_mse)
        best = final if improved else best
        epochs_run = epoch + 1
        if args.early_stop_patience and epoch - best[2] >= args.early_stop_patience:
            print(f"early stop: no validation improvement for {epoch - best[2]} epoch(s) after epoch {best[2]} "
                  f"(--early-stop-patience {args.early_stop_patience})", flush=True)
            break
    kept = best if args.select == "best" else final
    train_s = time.perf_counter() - t0
    print(f"trained {epochs_run} of {args.epochs} epochs in {train_s:.1f}s on {device}; "
          f"best val mse {best[0]:.4f} at epoch {best[2]}" + (f" (eval mse {best[3]:.4f})" if best[3] is not None else "")
          + f"; kept epoch {kept[2]} (--select {args.select})")
    has_val = bool(len(va["y"]))
    t_score = time.perf_counter()  # the kept model scored on validation and evaluation, per phase and matchup
    model = ValueNet(kept[1], vocab, {
        "name": args.name or args.out.stem, "hidden": args.hidden, "epochs": args.epochs, "best_epoch": best[2],
        "epochs_run": epochs_run, **({"early_stop_patience": args.early_stop_patience} if args.early_stop_patience else {}),
        "select": args.select, "kept_epoch": kept[2], "aux_hp_weight": args.aux_hp_weight,
        "residual_hp": args.residual_hp, "init_from": init,
        **({"layers": args.layers} if args.layers != 2 else {}),
        **multistep,
        **({"aux_search_heads": aux_offsets, "aux_search_weight": args.aux_search_weight} if aux_offsets else {}),
        **({"aux_cand_value_weight": args.aux_cand_value_weight, "aux_mix_weight": args.aux_mix_weight,
            "aux_cand_k": cand_k, "aux_cand_vocab": cand["table"]["vocab"]} if cand else {}),
        **({"rescaled_features": len(rescaled), "rescaled_feature_names": rescaled} if init is not None else {}),
        "train_games": tr["games"], "val_games": va["games"], "sources": [str(d) for d in args.dirs],
        "games_per_gen": args.games_per_gen, "gen_weights": gen_weights,
        "created": time.time(), "state_v3_fraction": tr.get("v3_fraction", 0.0),
        "explore_targets": args.explore_targets, "value_only_target_fraction": round(value_only_fraction, 4),
        **cheap,
        "outcome_target_coverage": cov, "best_val_mse": round(best[0], 5) if has_val else None,
        "val_mse": round(kept[0], 5) if has_val else None,
        "eval_mse": round(kept[3], 5) if kept[3] is not None else None, "eval_games": ev["games"],
        "eval_dirs": [str(d) for d in args.eval_dir],
        "heldout": {d: {"manifest": m["manifest"], "sha256": m["sha256"]} for d, m in data["heldout"].items()},
        **({"record_filter": data["record_filter"]} if data.get("record_filter") else {}),
        **({"sample_window": data["sample_window"]} if data.get("sample_window") else {}),
        **({"shared_window": data["shared_window"]} if data.get("shared_window") else {}),
    })

    def saved_values(split):  # p1's values from the saved model, as play computes them
        base, out, at = np.asarray(split["baseline"]), [], 0
        for c in chunks_of(split):  # chunks come in row order
            n = len(c["nums0"])
            out.append(model.value_encoded(c["nums0"], c["cats0"], c["nums1"], c["cats1"], base[at:at + n]))
            at += n
        return np.concatenate(out)
    if len(va["y"]):
        pred = saved_values(va)
        print("validation, by game phase (lower mse is better):")
        print("\n".join(report(va, pred, [d.name for d in args.dirs])))
        if args.allow_match_settings:
            print("validation, by match settings (--allow-match-settings):")
            for name, r in variant_mse(va, pred, args.allow_match_settings).items():
                print(f"  {name}: {r['games']} games, n={r['n']}"
                      + (f"  mse net {r['net']:.4f}  evaluate {r['evaluate']:.4f}  net sign accuracy "
                         f"{r['sign_accuracy']:.1%}" if r["n"] else ""))
    pred_e = None
    if len(ev["y"]):
        pred_e = saved_values(ev)
        print("evaluation (--eval-dir, never trained on), by game phase:")
        print("\n".join(report(ev, pred_e)))
    # Per-phase errors in the meta too, so runs can be compared from their files (explore_ablation.py).
    model.meta["val_positions"], model.meta["eval_positions"] = len(va["y"]), len(ev["y"])
    model.meta["val_mse_by_phase"] = _phase_mse(va, pred) if len(va["y"]) else None
    model.meta["val_mse_by_matchup"] = matchup_mse(va, pred) if len(va["y"]) else None
    model.meta["eval_mse_by_phase"] = _phase_mse(ev, pred_e) if pred_e is not None else None
    model.meta["eval_mse_by_matchup"] = matchup_mse(ev, pred_e) if pred_e is not None else None
    model.meta["eval_sign_accuracy"] = sign_accuracy(ev, pred_e) if pred_e is not None else None
    model.meta["val_sign_accuracy"] = sign_accuracy(va, pred) if len(va["y"]) else None
    if args.allow_match_settings:  # validation (and evaluation) per allowed variant, with counts
        model.meta["allow_match_settings"] = args.allow_match_settings
        model.meta["val_mse_by_match_settings"] = variant_mse(va, pred if len(va["y"]) else None,
                                                              args.allow_match_settings)
        model.meta["eval_mse_by_match_settings"] = variant_mse(ev, pred_e, args.allow_match_settings) \
            if args.eval_dir else None
        model.meta["train_games_by_match_settings"] = dict(zip(
            [*map(compat.variant_label, args.allow_match_settings), "other"], tr.get("variant_games") or []))
    if aux and len(va["y"]):  # the saved aux head against the final hp margin, where known
        hp_va = np.asarray(va["hp_margin"], np.float64)
        aux_va = np.concatenate([(forward(model.p, c["nums0"], c["cats0"], head="aux")
                                  - forward(model.p, c["nums1"], c["cats1"], head="aux")) / 2 for c in chunks_of(va)])
        m = np.isfinite(hp_va)
        model.meta["val_aux_hp_mse"] = round(float(np.mean((aux_va[m] - hp_va[m]) ** 2)), 5) if m.any() else None
    if aux_offsets and len(va["y"]):  # the saved search heads against their validation targets, where known
        sv = np.asarray(va["aux_search"], np.float64)
        pv = np.concatenate([(forward(model.p, c["nums0"], c["cats0"], head="search")
                              - forward(model.p, c["nums1"], c["cats1"], head="search")) / 2 for c in chunks_of(va)])
        model.meta["val_aux_search_mse"] = {
            str(d): (round(float(np.mean((pv[m, i] - sv[m, i]) ** 2)), 5) if m.any() else None)
            for i, d in enumerate(aux_offsets) for m in [np.isfinite(sv[:, i])]}
    if cand and len(va["y"]):  # the saved candidate heads against the validation turns' search records
        model.meta.update(cand_metrics(model.p, va, cand["table"]))
    phases.add("score", time.perf_counter() - t_score)
    with phases("export"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
    print(f"saved {args.out}")
    print(phases.line(), flush=True)
    print(json.dumps({k: v for k, v in model.meta.items() if k != "sources"}))
    if ledger_path:
        _append_ledger(args, ledger_path, tags, started, cpu, memory, tr, va, len(t["y"]), load_s, train_s, device,
                       best, kept, pred if len(va["y"]) else None, given_dirs, gen_weights, ev, pred_e,
                       data["heldout"], data.get("sample_window"), phases.rounded(), data.get("shared_window"))


def _append_ledger(args, path, tags, started, cpu, memory, tr, va, n_train, load_s, train_s, device, best, kept,
                   pred, given_dirs, gen_weights, ev, pred_e, manifests, window=None, phase_seconds=None,
                   shared_window=None) -> None:
    """One `train` row for the metrics ledger (yomibench.ledger); never fails the run. `shared_window`:
    what --share-window did (yomibench.cache open_cached), when given."""
    try:
        from yomibench.valuenet import FEATURES

        mem = memory.stop()
        manifest = ledger.dataset_manifest(args.dirs)
        per_dir = manifest["games_per_dir"]
        cap = args.games_per_gen
        selected = [manifests[str(d)]["heldout"] + manifests[str(d)]["training"] for d in args.dirs]
        selected = [min(n, cap) if cap else n for n in selected]  # --games-per-gen's sample
        n_gens = len(args.dirs)

        def by_gen(split):
            g = ledger._np(split.get("gen", np.zeros(0))).astype(np.int64)
            return np.bincount(g, minlength=n_gens).tolist() if g.size else [0] * n_gens
        tr_pos, va_pos = by_gen(tr), by_gen(va)
        generation_mix = [
            {"dir": str(d), "files": per_dir[str(d)], "selected": selected[i], "train_positions": tr_pos[i],
             "val_positions": va_pos[i], "weight": (gen_weights or [1.0] * n_gens)[i],
             "state_v3_fraction": (tr.get("v3_fraction_by_gen") or [None] * n_gens)[i],
             "val_state_v3_fraction": (va.get("v3_fraction_by_gen") or [None] * n_gens)[i]}
            for i, d in enumerate(args.dirs)]
        dropped = window["candidates"] - window["kept"] if window else 0  # games the sample window left out
        truncated = sum(selected) - dropped - tr["games"] - va["games"]  # training skips only truncated records
        constant = ledger.mean_of(tr["outcome"])  # the best constant guess, from the training split
        metrics = {
            "train_games": tr["games"], "val_games": va["games"], "train_positions": n_train,
            "val_positions": len(va["outcome"]), "truncated_games": truncated,
            "explored_positions": {"train": tr.get("explored"), "val": va.get("explored")},
            "explore_targets": args.explore_targets, "value_only_positions": tr.get("value_only"),
            "value_only_target_fraction": round(tr["value_only"] / n_train, 4) if n_train else None,
            # Training positions whose target includes the outcome. Per CPU hour is left to
            # aggregation: records carry wall seconds per game, not the CPU spent making them.
            "outcome_positions": n_train - (tr.get("value_only") or 0),
            "outcome_target_coverage": tr.get("outcome_coverage"),
            "eval_games": ev["games"], "eval_positions": len(ev["outcome"]),
            "eval_mse": round(kept[3], 5) if kept[3] is not None else None,
            "eval_phases": ledger.phase_mse(ev["phase"], ev["outcome"], ev["baseline"], pred_e, constant)
            if pred_e is not None else None,
            "load_seconds": round(load_s, 2), "train_seconds": round(train_s, 2), "epochs": args.epochs,
            **({"phase_seconds": phase_seconds} if phase_seconds is not None else {}),
            "best_epoch": best[2], "best_val_mse": round(best[0], 5) if va["games"] else None, "device": device,
            "select": args.select, "kept_epoch": kept[2], "val_mse": round(kept[0], 5) if va["games"] else None,
            "aux_hp_weight": args.aux_hp_weight, "residual_hp": args.residual_hp,
            "state_v3_fraction": round(tr.get("v3_fraction", 0.0), 4),
            "state_v3_fraction_by_gen": tr.get("v3_fraction_by_gen"), **mem,
            "phases": ledger.phase_mse(va["phase"], va["outcome"], va["baseline"], pred, constant)
            if pred is not None else None,
            "val_mse_by_matchup": matchup_mse(va, pred) if pred is not None else None,
            "eval_mse_by_matchup": matchup_mse(ev, pred_e) if pred_e is not None else None,
            "eval_sign_accuracy": sign_accuracy(ev, pred_e) if pred_e is not None else None,
            "val_sign_accuracy": sign_accuracy(va, pred) if pred is not None else None,
            "constant_prediction": round(constant, 5),
        }
        if window:  # the sample window's settings and counts
            metrics["sample_window"] = window
        if shared_window:  # --share-window: the window written or read (mode, folder)
            metrics["shared_window"] = shared_window
        if args.allow_match_settings:  # the allow-list used, and validation per variant
            metrics["allow_match_settings"] = args.allow_match_settings
            metrics["val_mse_by_match_settings"] = variant_mse(va, pred, args.allow_match_settings)
            metrics["eval_mse_by_match_settings"] = variant_mse(ev, pred_e, args.allow_match_settings) \
                if args.eval_dir else None
            metrics["train_games_by_match_settings"] = dict(zip(
                [*map(compat.variant_label, args.allow_match_settings), "other"], tr.get("variant_games") or []))
        skip = ("ledger", "no_ledger", "tag", "owner", "log") + (() if args.allow_match_settings else
                                                                ("allow_match_settings",)) \
            + (() if window else ("window_budget_games", "window_recent_hours", "window_seed")) \
            + (() if args.share_window else ("share_window", "share_window_frozen"))
        prov = ledger.provenance(
            settings={k: v for k, v in vars(args).items() if k not in skip}, started_utc=started, load=cpu,
            owner=args.owner, device=device, engine=ledger.header_engine(args.dirs),
            models={"out": ledger.model_info(args.out)}, dataset=manifest, features_version=FEATURES,
            split_seed=args.seed, generation_mix=generation_mix,
            heldout={d: {k: m[k] for k in ("manifest", "sha256", "seed", "fraction", "heldout", "training", "added")}
                     for d, m in manifests.items()},
            eval_set=ledger.dataset_manifest(args.eval_dir) if args.eval_dir else None,
            window={"n": args.window, "given": [str(d) for d in given_dirs], "trained_on": [str(d) for d in args.dirs],
                    "games_per_gen": cap, "gen_weights": gen_weights},
            target_blend={"value_weight": args.value_weight, "target": "(1 - w) * outcome + w * search value",
                          "cheap_target": args.cheap_target, "cheap_value_weight": args.cheap_value_weight,
                          "full_turns_only": args.full_turns_only,
                          **({"target_nstep": args.target_nstep} if args.target_nstep is not None else {}),
                          **({"target_lambda": args.target_lambda} if args.target_lambda is not None else {}),
                          **({"aux_search_heads": parse_offsets(args.aux_search_heads),
                              "aux_search_weight": args.aux_search_weight} if args.aux_search_heads else {}),
                          **({"aux_cand_value_weight": args.aux_cand_value_weight,
                              "aux_mix_weight": args.aux_mix_weight, "aux_cand_k": args.aux_cand_k}
                             if args.aux_cand_value_weight > 0 or args.aux_mix_weight > 0 else {}),
                          "explore_targets": args.explore_targets,
                          "explored": {"uniform-only": "positions at or before a game's last uniform (opening-noise) "
                                                       "pick train on the search value alone; softmax picks keep "
                                                       "outcome targets",
                                       "all": "positions at or before a game's last exploratory pick (any mode) "
                                              "train on the search value alone",
                                       "none": "outcome targets everywhere"}[args.explore_targets],
                          "validation": "outcome; positions at or before a game's last uniform pick dropped"},
            counts=ledger.counts(sum(selected), tr["games"] + va["games"],
                                 {"truncated": truncated, **({"sample_window": dropped} if window else {})}, 0),
            artifacts={"model": args.out, "inputs": args.dirs, "log": ledger.log_from_args(args)})
        ledger.append("train", metrics, prov, path=path, tags=tags)
    except Exception as e:  # noqa: BLE001  (the ledger never fails the run)
        import sys

        print(f"ledger: could not append a train row: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

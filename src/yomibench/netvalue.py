"""Scoring previewed positions with value nets, on features an engine computed for each preview (an engine that
returns features rather than states, such as a batch engine plugin). `BatchValue` gives the same numbers as
`ValueNet.batch` on the states, bit for bit."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from yomibench import features, phases
from yomibench.valuenet import ValueNet, forward, hp_lead, residual_value


def net_key(net: ValueNet) -> str:
    """What a net's features depend on: its feature version, vocabulary and raw defaults."""
    return json.dumps([net.version, net.vocab.ids, net.vocab.raw_defaults], sort_keys=True)


class Preview(dict):
    """One previewed pair. As a dict it is the next state when the engine was asked for states
    (else empty); `feats[net_key(net)]` is (nums (2, n_numeric), cats (2, n_categorical)), both
    seats' value-net features of that state for each net the engine serves; `winner` is the
    Batch's (0, 1, or -1 for none) and `done` whether the game is over there."""

    __slots__ = ("_feats", "_src", "_row", "done", "winner")

    def __init__(self, *args: Any, **kw: Any):
        super().__init__(*args, **kw)
        self._feats: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        # A batch engine's previews share their call's whole feature arrays (`_src`: net key -> (nums
        # (N, 2, n_numeric), cats (N, 2, n_categorical))) and point at their row of them (`_row`), so
        # BatchValue gathers a batch's inputs with one index instead of stacking per-preview rows.
        self._src: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._row = 0

    @property
    def feats(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._feats is None:
            self._feats = {} if self._src is None else \
                {key: (nums[self._row], cats[self._row]) for key, (nums, cats) in self._src.items()}
        return self._feats

    @feats.setter
    def feats(self, value: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._feats, self._src = value, None

    def terminal(self, seat: int) -> float | None:
        """`search.terminal_value(state, seat)`: the exact result if the game is over (KO or time:
        the Batch's `done` is the same test), else None."""
        if not self.done:
            return None
        return 0.0 if self.winner < 0 else (1.0 if self.winner == seat else -1.0)


def v3_strip_columns() -> tuple[np.ndarray, np.ndarray]:
    """The columns of a v3 encoding that `valuenet.strip_v3_state` changes, and the values they
    take there (features.py: a state without the move fields encodes `_move` as zeros but for
    DI; objects without hitboxes have no `_object_hit` and so no threats but their count)."""
    f = features
    cols: list[int] = []
    vals: list[float] = []
    move_at = len(f.PLAYER_NUMS) + len(f.TIMING_NUMS)
    for player_at in (0, f.PLAYER_WIDTH):
        for k, name in enumerate(f.MOVE_NUMS):
            if name not in ("di_x", "di_y"):
                cols.append(player_at + move_at + k)
                vals.append(0.0)
    objects_at = 2 * f.PLAYER_WIDTH
    for slot in range(f.OBJECT_SLOTS):
        for k, name in enumerate(f.OBJECT_NUMS):
            if name.startswith("hit_"):
                cols.append(objects_at + slot * len(f.OBJECT_NUMS) + k)
                vals.append(0.0)
    no_threat = {"active": 0.0, "active_damage": 0.0, "nearest": f.FAR, "damage_near": 0.0, "soonest": f.SOON}
    globals_at = objects_at + f.OBJECT_SLOTS * len(f.OBJECT_NUMS)
    for k, name in enumerate(f.GLOBAL_NUMS):
        side, _, what = name.partition("_")
        if side in ("mine", "theirs") and what in no_threat:
            cols.append(globals_at + k)
            vals.append(no_threat[what])
    return np.asarray(cols), np.asarray(vals, dtype=np.float32)


class BatchValue:
    """A `ValueNet` that scores `Preview`s: the same numbers as `ValueNet.batch` on their states
    (same encoder inputs, bit for bit, through the same `forward`, in the same batch shapes). A
    net that plays on states without the v3 move fields (`ValueNet.mask_v3_state`) gets the
    Batch's v3 features with those fields' columns set as a stripped state encodes them. A
    `residual_hp` net adds the hp lead read from the same features, as `ValueNet.batch` does."""

    def __init__(self, net: ValueNet, batched_forward: bool = False):
        """`batched_forward` (selfplay --batched-net-forward; default off): score both seats' views in one
        forward over 2N rows (a reused input buffer) instead of two N-row forwards. NOT bit-identical:
        BLAS kernels sum in an order that depends on the batch shape, so values move by up to ~1e-6 and
        search can pick differently; bots record it (`value_forward: "batched"` in their config)."""
        self.net = net
        self.key = net_key(net)
        self.__name__ = net.__name__  # bot configs (and records) name the value function by it
        self.strip = v3_strip_columns() if getattr(net, "mask_v3_state", False) else None
        self.batched_forward = batched_forward
        self._buf: tuple[np.ndarray, np.ndarray] | None = None  # batched_forward's (nums, cats) rows, grown

    def batch(self, previews: list[Preview], seat: int) -> np.ndarray:
        t = time.perf_counter()
        try:
            return self._batch(previews, seat)
        finally:
            phases.add("net", time.perf_counter() - t)

    def _batch(self, previews: list[Preview], seat: int) -> np.ndarray:
        nums, cats = self._inputs(previews)
        if self.strip is not None:
            nums[:, :, self.strip[0]] = self.strip[1]
        if self.batched_forward:
            n = len(previews)
            buf = self._buf
            if buf is None or len(buf[0]) < 2 * n or buf[0].shape[1:] != nums.shape[2:] or buf[0].dtype != nums.dtype:
                buf = self._buf = (np.empty((2 * n,) + nums.shape[2:], nums.dtype),
                                   np.empty((2 * n,) + cats.shape[2:], np.int64))
            bn, bc = buf[0][:2 * n], buf[1][:2 * n]
            bn[:n], bn[n:] = nums[:, seat], nums[:, 1 - seat]
            bc[:n], bc[n:] = cats[:, seat], cats[:, 1 - seat]
            both = forward(self.net.p, bn, bc, self.net.encoder.CATEGORY_OF_SLOT)
            out = [both[:n], both[n:]]
        else:
            cats = cats.astype(np.int64)
            out = [forward(self.net.p, nums[:, view], cats[:, view], self.net.encoder.CATEGORY_OF_SLOT)
                   for view in (seat, 1 - seat)]
        values = (out[0] - out[1]) / 2
        if getattr(self.net, "residual_hp", False):
            values = residual_value(values, hp_lead(nums[:, seat], self.net.encoder))
        for k, p in enumerate(previews):  # a finished game has an exact value; the net never decides it
            t = p.terminal(seat)
            if t is not None:
                values[k] = t
        return values

    def _inputs(self, previews: list[Preview]) -> tuple[np.ndarray, np.ndarray]:
        """The previews' features stacked ((N, 2, n) each, fresh arrays): what np.stack of their rows
        gives, value for value, gathered from their batch calls' arrays by row index when they carry
        them (one fancy index per call instead of a Python-level stack of N row views)."""
        key = self.key
        groups: dict[int, tuple[tuple[np.ndarray, np.ndarray], list[int], list[int]]] = {}
        for k, p in enumerate(previews):
            src = p._src
            if src is None or p._feats is not None:  # features set directly (tests, tools)
                return (np.stack([p.feats[key][0] for p in previews]), np.stack([p.feats[key][1] for p in previews]))
            g = groups.get(id(src))
            if g is None:
                g = groups[id(src)] = (src[key], [], [])
            g[1].append(k)
            g[2].append(p._row)
        if len(groups) == 1:
            (nums, cats), _, rows = next(iter(groups.values()))
            return nums[rows], cats[rows]  # a list index: a copy (BatchValue may overwrite stripped columns)
        first = next(iter(groups.values()), ((None, None),))[0]
        if first[0] is None or any(s[0].dtype != first[0].dtype or s[1].dtype != first[1].dtype
                                   or s[0].shape[1:] != first[0].shape[1:] or s[1].shape[1:] != first[1].shape[1:]
                                   for s, _, _ in groups.values()):
            return (np.stack([p.feats[key][0] for p in previews]), np.stack([p.feats[key][1] for p in previews]))
        n = len(previews)
        out_n = np.empty((n,) + first[0].shape[1:], first[0].dtype)
        out_c = np.empty((n,) + first[1].shape[1:], first[1].dtype)
        for (nums, cats), pos, rows in groups.values():
            out_n[pos] = nums[rows]
            out_c[pos] = cats[rows]
        return out_n, out_c

    def __call__(self, state: Preview, seat: int) -> float:
        return float(self.batch([state], seat)[0])

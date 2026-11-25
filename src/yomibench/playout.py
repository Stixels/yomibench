"""Playout cap randomization (KataGo's): which decisions are searched cheaply, and the capped
candidate lists a cheap search sees. Self-play uses it per game (`selfplay --cheap-fraction`), a
bot per its own decisions (`make_bot("matrix-do@cheap=P:N")`, for arenas and gates)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from yomibench.engine import Candidate


def cheap_turn(seed: int, turn: int, fraction: float) -> bool:
    """Whether decision `turn` of game `seed` is searched cheaply under playout cap randomization:
    a fixed hash of the seed and turn under `fraction`, like `selfplay.full_previews`, so the split depends on
    nothing else (not the process, the bot, or earlier picks' random draws)."""
    return fraction >= 1 or (fraction > 0 and int(hashlib.sha256(f"cheap:{seed}:{turn}".encode()).hexdigest()[:15], 16)
                             / 16 ** 15 < fraction)


CAP_POLICIES = ("list", "options-first", "rotate", "options-rotate", "options-shuffle")
# With `counts` (quick_policy's +parryxT only): a count widget with at most this many values is a mode widget.
COUNT_OPTION_MAX = 32


def _h(*parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, ("cap",) + parts)).encode()).hexdigest()[:15], 16)


def _leaves(value: Any, path: tuple = ()) -> list[tuple[tuple, Any]]:
    """A candidate's data (or extra) as (path, widget value) leaves: dicts by key, except the dicts a
    data scene's widgets return themselves (an XY plot or 8-way direction {x, y}, a slider {x} or
    {count}, a dropdown {id, name}), which are one leaf each."""
    if isinstance(value, dict) and value and set(value) not in ({"x", "y"}, {"x"}, {"count"}, {"id", "name"}):
        return [leaf for k in sorted(value) for leaf in _leaves(value[k], path + (k,))]
    return [(path, value)]


def _is_count(v: Any) -> bool:
    return isinstance(v, dict) and set(v) == {"count"}


def _is_mode(path: tuple, values: list[Any], counts: bool = False) -> bool:
    """Whether one widget leaf of a move is a mode widget (set 4's split: checkbox, dropdown, block
    height), from the values its variants show: a checkbox is a bool, a dropdown {id, name} or a
    string, a block height {x: 0, y: 0 | 1} under a key of a multi-widget dict. An XY plot, an 8-way
    direction or a slider is not (a top-level {x, y} is always an aim). With `counts` (quick_policy's
    +parryxT only; off everywhere else), a count widget {count} under a key of a multi-widget dict with at
    most COUNT_OPTION_MAX distinct values in the move (a parry timing, a charge level) is one too."""
    if all(isinstance(v, (bool, str)) or v is None for v in values):
        return any(v is not None for v in values)
    if all(isinstance(v, dict) and set(v) == {"id", "name"} for v in values):
        return True
    if counts and path and all(map(_is_count, values)) and len({v["count"] for v in values}) <= COUNT_OPTION_MAX:
        return True
    return bool(path) and all(isinstance(v, dict) and set(v) == {"x", "y"} and v["x"] == 0 and v["y"] in (0, 1)
                              for v in values)


def _frozen(value: Any) -> Any:
    """A hashable stand-in for a JSON value (string keys), equal to another's exactly when their
    `json.dumps(..., sort_keys=True)` strings are equal, without serialising: scalars carry their type
    (so True, 1 and 1.0 stay apart, as JSON writes them), floats compare by their repr (JSON's spelling:
    -0.0 and 0.0 differ, NaN equals NaN), lists become tuples and dicts their items sorted by key."""
    t = type(value)
    if t is dict:
        return (dict, tuple((k, _frozen(value[k])) for k in sorted(value)))
    if t is list or t is tuple:
        return (list, tuple(map(_frozen, value)))
    if t is float:
        return (float, repr(value))
    return (t, value)


def _option_tuples(cands: list[Candidate], counts: bool = False) -> tuple[list[Any], list[list[tuple]]]:
    """`option_keys` as hashable keys (each candidate's frozen mode-widget values, `_frozen`): equal
    exactly when the string keys are, without serialising anything. Also each candidate's mode
    paths and values ((path, value) pairs, the string key's content), so a key's string can be written
    for the few callers that need it (`_option_string`)."""
    by_move: dict[str, list[int]] = {}
    for i, c in enumerate(cands):
        by_move.setdefault(c.action, []).append(i)
    keys: list[Any] = [()] * len(cands)
    parts: list[list[tuple]] = [[] for _ in cands]
    for idx in by_move.values():
        leaves = [dict(_leaves(cands[i].data, ("data",)) + _leaves(cands[i].extra, ("extra",))) for i in idx]
        paths = sorted({p for lv in leaves for p in lv}, key=str)
        mode = [p for p in paths if _is_mode(p[1:], [lv.get(p) for lv in leaves], counts)
                and len({_frozen(lv.get(p)) for lv in leaves}) > 1]
        for i, lv in zip(idx, leaves):
            parts[i] = [(p, lv.get(p)) for p in mode]
            keys[i] = tuple((p, _frozen(v)) for p, v in parts[i])
    return keys, parts


def _option_string(part: list[tuple]) -> str:
    """One candidate's `option_keys` string from its `_option_tuples` parts."""
    return json.dumps([[list(p), v] for p, v in part], sort_keys=True)


def option_keys(cands: list[Candidate]) -> list[str]:
    """Each candidate's option within its move: its values of the mode widgets (`_is_mode`, judged over
    the move's variants in this list) that vary among those variants, in its data and its extra, as a
    string. Candidates of one move that differ only in aims, directions or sliders share an option.
    (`cap_candidates` groups by `_option_tuples`, the same equality without serialising.)"""
    return [_option_string(part) for part in _option_tuples(cands)[1]]


def cap_candidates(cands: Iterable[Candidate], cap: int, policy: str = "list", salt: str = "") -> list[Candidate]:
    """One seat's candidates cut to `cap`. Every Continue variant and one variant of each move are
    always kept (even over the cap); then further variants one per move per pass until the cap.
    Kept candidates stay in list order. A move is a run of candidates with one action. Policies:

    list           (default) the engine's per-seat budget (set 4's rules, `budget`): each move's
                   first variant, then its next variants in list order, passes over the moves in
                   list order, so a cheap decision sees about what `max_candidates` would give. The
                   engine round-robins over move categories (Attack, Special, ...) before moves;
                   candidates don't carry their category, so here every move is its own turn of the
                   round-robin: the same count and the same moves, possibly other data variants.
    options-first  as list, but first the passes add each move's other options (`option_keys`: a
                   distinct value of its mode widgets, e.g. a checkbox's other value), each option's
                   first variant in list order, one per move per pass; only then do list's passes add
                   further variants (other aims, directions, sliders) until the cap.
    rotate         each move's kept variant is `h(salt, move) mod n` of its n variants instead of
                   the first, its next ones follow cyclically, and each pass starts at move
                   `h(salt) mod moves` instead of the list start.
    options-rotate options-first with rotation: each move's always-kept variant is the first variant
                   of its option `h(salt, move) mod options` (not always the first option), its other
                   options follow cyclically, and every pass starts at move `h(salt) mod moves`
                   instead of the list start. A move's further variants still come in list order.
    options-shuffle options-first's order of moves, options and passes, but the variant that stands for
                   each option (and the order its other variants come in) is a seeded shuffle of that
                   option's variants (`h(salt, move, option, variant)`) instead of list order. The
                   kept options are exactly options-first's; only which aim, direction or slider value
                   represents each one changes, uniformly over the option's variants.

    `salt` (the game seed, turn and seat for a cheap decision) makes the rotation a fixed function
    of the decision, like `cheap_turn`; list and options-first ignore it."""
    if policy not in CAP_POLICIES:
        raise ValueError(f"cap policy must be one of {CAP_POLICIES}: {policy!r}")
    cands = list(cands)
    if cap <= 0 or len(cands) <= cap:
        return cands
    moves: list[list[int]] = []
    for i, c in enumerate(cands):
        if moves and cands[moves[-1][0]].action == c.action:
            moves[-1].append(i)
        else:
            moves.append([i])
    rotating = policy in ("rotate", "options-rotate")
    # Per move: the variants its option passes take (first: the one always kept), then the rest.
    phases: list[tuple[list[int], list[int]]] = []
    for m in moves:
        if policy == "options-shuffle":
            phases.append(_shuffled_options(cands, m, salt))
        elif policy in ("options-first", "options-rotate"):
            keys = _option_tuples([cands[i] for i in m])[0]
            heads = [i for k, i in enumerate(m) if keys[k] not in keys[:k]]
            if policy == "options-rotate":
                r = _h(salt, cands[m[0]].action) % len(heads)
                heads = heads[r:] + heads[:r]
            phases.append((heads, [i for i in m if i not in heads]))
        else:
            r = _h(salt, cands[m[0]].action) % len(m) if policy == "rotate" else 0
            order = m[r:] + m[:r]
            phases.append((order[:1], order[1:]))
    keep = {h[0] for h, _ in phases} | {i for m in moves if cands[m[0]].action == "Continue" for i in m}
    start = _h(salt) % len(moves) if rotating else 0
    phases = phases[start:] + phases[:start]
    for runs in ([h[1:] for h, _ in phases], [rest for _, rest in phases]):
        depth = 0
        while len(keep) < cap and any(len(r) > depth for r in runs):
            for r in runs:
                if len(keep) >= cap:
                    break
                if len(r) > depth:
                    keep.add(r[depth])
            depth += 1
    return [c for i, c in enumerate(cands) if i in keep]


def _spread_ranks(n: int) -> list[int]:
    """Each of n sorted values' place in a spread order of them: both ends first, then the midpoint of
    every gap, breadth first (n = 20: 1st, 20th, 10th, 5th, 15th, 3rd, 7th, ...), so any prefix of the
    order covers the whole range about evenly."""
    order, gaps = [0, n - 1][:n], [(0, n - 1)]
    while gaps:
        a, b = gaps.pop(0)
        if b - a > 1:
            order.append((a + b) // 2)
            gaps += [(a, (a + b) // 2), ((a + b) // 2, b)]
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r
    return rank


def _interleaved(parts: list[list[tuple]]) -> list[int]:
    """Option positions (`parts`: each option's mode (path, value) pairs, options in list order) reordered
    round-robin over the options' classes, a class being an option's values of its non-count mode widgets,
    in order of first appearance: the first option of each class, then the second of each, and so on.
    Within a class, options go by their count values' `_spread_ranks` (per count widget, among the
    move's values of it; widgets in path order), then list order."""
    counts: dict[tuple, list[int]] = {}
    for part in parts:
        for p, v in part:
            if _is_count(v):
                counts.setdefault(p, []).append(v["count"])
    rank = {}
    for p, vs in counts.items():
        vals = sorted(set(vs))
        rank[p] = dict(zip(vals, _spread_ranks(len(vals))))
    by: dict[Any, list[tuple]] = {}
    for pos, part in enumerate(parts):
        cls = tuple((p, _frozen(v)) for p, v in part if not _is_count(v))
        by.setdefault(cls, []).append((tuple(rank[p][v["count"]] for p, v in part if _is_count(v)), pos))
    runs = [[pos for _, pos in sorted(r)] for r in by.values()]
    return [r[d] for d in range(max(map(len, runs))) for r in runs if d < len(r)]


def _shuffled_options(cands: list[Candidate], m: list[int], salt: str,
                      counts: bool = False, per_class: int = 0) -> tuple[list[int], list[int]]:
    """options-shuffle's phases for one move (its candidate indices `m`): the options in list order of
    their first variant, each represented by the first of its variants in a seeded shuffle; then the
    rest, one per option per round in that order, each option's in its shuffled order. With `counts`
    (quick_policy's +parryxT only): count widgets are mode widgets too, and the options are `_interleaved`
    over their classes. With `per_class` T: of each class's options with a count value, only its first T
    in that order stay; the others, and every variant of them, are left out."""
    keys, parts = _option_tuples([cands[i] for i in m], counts)
    groups: dict[Any, list[int]] = {}
    first: dict[Any, int] = {}  # each option's first variant (position in m): its key's string is the hash's
    for pos, (k, i) in enumerate(zip(keys, m)):
        groups.setdefault(k, []).append(i)
        first.setdefault(k, pos)
    action = cands[m[0]].action
    order = [sorted(g, key=lambda i, k=_option_string(parts[first[k]]):
                    _h(salt, action, k, json.dumps([cands[i].data, cands[i].extra], sort_keys=True)))
             for k, g in groups.items()]
    if counts and len(order) > 2:
        perm = _interleaved([parts[first[k]] for k in groups])
        order = [order[j] for j in perm]
        if per_class:
            opts = [list(groups)[j] for j in perm]
            seen: dict[Any, int] = {}
            kept = []
            for k, g in zip(opts, order):
                if any(_is_count(v) for _, v in parts[first[k]]):
                    cls = tuple((p, _frozen(v)) for p, v in parts[first[k]] if not _is_count(v))
                    seen[cls] = seen.get(cls, 0) + 1
                    if seen[cls] > per_class:
                        continue
                kept.append(g)
            order = kept
    rest = [g[r] for r in range(1, max(map(len, order))) for g in order if r < len(g)]
    return [g[0] for g in order], rest


def add_extra_count_options(cands: list[Candidate], base: list[Candidate], salt: str, per_class: int) -> list[Candidate]:
    """`base` (a menu cut from `cands`) plus +parryxT's addition on top: for each move of `cands` with a count
    widget, its first `per_class` count options per value class in the spread, class-interleaved order
    (`_shuffled_options` with counts), each by its seeded variant, unless a candidate already in `base` shows
    that option (no option is added twice). In list order. Used by quick_policy's policy40+gumbel8+parryxT."""
    keep = {c.id for c in base}
    moves: list[list[int]] = []
    for i, c in enumerate(cands):
        if moves and cands[moves[-1][0]].action == c.action:
            moves[-1].append(i)
        else:
            moves.append([i])
    for m in moves:
        keys, parts = _option_tuples([cands[i] for i in m], counts=True)
        if not any(_is_count(v) for part in parts for _, v in part):
            continue
        shown = {keys[k] for k, i in enumerate(m) if cands[i].id in keep}
        key_of = {i: keys[k] for k, i in enumerate(m)}
        heads, _ = _shuffled_options(cands, m, salt, counts=True, per_class=per_class)
        for i in heads:
            if key_of[i] not in shown:
                keep.add(cands[i].id)
                shown.add(key_of[i])
    return [c for c in cands if c.id in keep]

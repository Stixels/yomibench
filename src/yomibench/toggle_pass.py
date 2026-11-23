"""Action space v7 L1: the toggle verification pass (off unless a bot is built with it: self-play's fleet flag, or a
human-play build's bot_factory).

Ported from the simulator plugin's toggle pass (its action space L1 spec). After a full decision's search converges (the double oracle plus factor refinement), each seat's
**toggle pool** is scanned once against the other seat's final support, the entries whose
best-response gain is above `margin` are admitted, and the game is solved again with them. Base lists
and the ordinary refinement are unchanged; quick (playout-capped) decisions never run it
(`ToggleMatrixBot.cheap()` is a plain MatrixBot), so cheap turns stay cheap.

The pool is data-driven: the engine's `toggle_pool` (the simulator plugin's, needs
`set6={"l1_pool": True}`) names, per move, its first listed variant
and the `extra_set` values that change a toggle, plus whether feint and reverse apply. No move, control
or style is named here, and nothing is weighted: entries are only added to the menu, and only when the
search itself finds they gain.

Per seat, the pool holds:
- for the `k` moves outside the support with the best reply value against the other seat's mix (all
  of them with `k=None`), their first variant with every combination of a toggle extra set (or none),
  feint and reverse, other than the variant itself;
- with `support_children`, for every support candidate, the same combinations on that candidate.

Self-play (selfplay --toggle-pass JSON, round-3 manifest flag `toggle_pass`, e.g.
{"k": 5, "margin": 0.02, "support_children": false}; absent or null: off, exactly as before): the bot
is a `ToggleMatrixBot`, the engine is built with `set6` plus "l1_pool": true (so the header's
candidate_rules name it), and the pass's candidates, strategies, candidate values and picks are
recorded like any other search's, with its summary in the decision's `oracle.toggle_pass`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from itertools import product
from typing import Any

from yomibench.bots import (FACTOR_DEFAULTS, MatrixBot, _ask, _content, _replies, _resolve, preview_states,
                            turn_candidates)
from yomibench.search import double_oracle

# The flag's defaults: production's variant (k=5, roots only, as the L1 spec recommends).
DEFAULTS = {"k": 5, "margin": 0.02, "support_children": False}


@dataclass(frozen=True)
class TogglePass:
    """The L1 pass's knobs. `k`: non-support moves scanned per seat (None: every one). `margin`:
    the best-response gain (p1 value units) an entry needs to be admitted (the self-play factor
    margin's default). `support_children`: also scan the support candidates' combined children
    (the L1 spec's default; False, production's default here, scans the k roots only)."""
    k: int | None = 5
    margin: float = 0.02
    support_children: bool = False


def parse_settings(spec: Any) -> dict[str, Any] | None:
    """The `toggle_pass` flag (a JSON object or its text) checked, defaults filled in:
    {"k", "margin", "support_children"}; None (off) for None, "", "null" or "none". ValueError otherwise."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "null", "none")):
        return None
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            raise ValueError(f"toggle_pass {spec!r} isn't JSON ({e})") from e
        if spec is None:
            return None
    if not isinstance(spec, dict):
        raise ValueError(f"toggle_pass {spec!r} isn't a JSON object")
    unknown = sorted(set(spec) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"toggle_pass: unknown keys {unknown} (known: {', '.join(DEFAULTS)})")
    out = {**DEFAULTS, **spec}
    k = out["k"]
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError(f"toggle_pass: k {k!r} must be a positive integer or null (every move)")
    m = out["margin"]
    if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or m < 0:
        raise ValueError(f"toggle_pass: margin {m!r} must be a finite number >= 0")
    out["margin"] = float(m)
    if not isinstance(out["support_children"], bool):
        raise ValueError(f"toggle_pass: support_children {out['support_children']!r} must be true or false")
    return out


def settings_tag(settings: dict[str, Any]) -> str:
    """The ledger tag value of parsed settings, e.g. "k:5,margin:0.02,support_children:0"."""
    return (f"k:{'all' if settings['k'] is None else settings['k']},margin:{settings['margin']},"
            f"support_children:{int(settings['support_children'])}")


def make(settings: dict[str, Any] | None) -> TogglePass | None:
    return None if settings is None else TogglePass(settings["k"], settings["margin"], settings["support_children"])


def parse_set6(spec: Any) -> dict[str, bool] | None:
    """The `set6` flag: the simulator's set-6 switches as {name: bool} (or its JSON text), passed to the
    engine as is (the engine refuses a name it doesn't know); None (the build's default) for None, "",
    "null" or "none". ValueError for anything that isn't such a mapping."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "null", "none")):
        return None
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            raise ValueError(f"set6 {spec!r} isn't JSON ({e})") from e
        if spec is None:
            return None
    if not isinstance(spec, dict) or not all(isinstance(k, str) and isinstance(v, bool) for k, v in spec.items()):
        raise ValueError(f"set6 {spec!r} isn't a JSON object of switch names to true/false")
    return dict(sorted(spec.items()))


def engine_set6(set6: dict[str, bool] | None, settings: dict[str, Any] | None) -> dict[str, bool] | None:
    """The set-6 switches the engine is built with: `set6`, plus "l1_pool": true when the pass is on
    (its pool needs it). Both off: None, so the engine is built exactly as before."""
    if settings is None:
        return set6
    return dict(sorted({**(set6 or {}), "l1_pool": True}.items()))


def _combos(feint: bool, reverse: bool, sets: list[int]) -> list[tuple[int, bool, bool]]:
    return list(product([0] + list(sets), [False, True] if feint else [False], [False, True] if reverse else [False]))


def pool_specs(obs, engine, view, strategies, replies, pools, k: int | None,
               support_children: bool = True) -> tuple[list, list[int], list[int]]:
    """The pass's expand specs [(key, spec)] for both seats (keys as `expand_near_support`'s), and
    each seat's full pool size (k = all) and how many of its specs are support children, counted before
    anything is asked."""
    cands = turn_candidates(obs, engine, view)
    out, sizes, sup = [], [0, 0], [0, 0]
    for seat in (0, 1):
        by_action = {p["action"]: p for p in pools[seat]}
        strat, u = strategies[seat], replies[seat]
        in_support = {cands[seat][i].action for i, p in enumerate(strat) if p > 0}
        best: dict[str, float] = {}
        for i, c in enumerate(cands[seat]):
            if c.action in by_action and c.action not in in_support:
                best[c.action] = max(best.get(c.action, float("-inf")), u[i])
        ranked = sorted(best, key=lambda a: (-best[a], a))
        roots: list[tuple[int, dict]] = []  # (candidate position in the view, pool entry)
        first = {c.id: i for i, c in enumerate(cands[seat])}
        for a in ranked:
            p = by_action[a]
            if p["candidate"] in first:
                roots.append((first[p["candidate"]], p))
        n_scan = len(roots) if k is None else min(k, len(roots))
        support = [(i, by_action[cands[seat][i].action]) for i, p in enumerate(strat)
                   if p > 0 and cands[seat][i].action in by_action]
        for n, (i, p) in enumerate(roots + support):
            c = cands[seat][i]
            combos = _combos(p["feint"], p["reverse"], p["extra_sets"])
            values = {f: (c.values or {}).get(f, FACTOR_DEFAULTS.get(f)) for f in (c.factors or {})}
            own = (values.get("extra_set", 0), values.get("feint", False), values.get("reverse", False))
            base = c.id if c.base is None else c.base
            entries = []
            for e, f, r in combos:
                if (e, f, r) == own:
                    continue
                nv = {**values, "extra_set": e}
                if p["feint"]:
                    nv["feint"] = f
                if p["reverse"]:
                    nv["reverse"] = r
                if e == 0 and "extra_set" not in (c.factors or {}):
                    nv.pop("extra_set")
                entries.append((seat, base, tuple(sorted(nv.items()))))
            sizes[seat] += len(entries)
            if n >= len(roots):
                sup[seat] += len(entries)
            if n < n_scan or (support_children and n >= len(roots)):  # scanned roots, then the support's children
                out.extend((key, {"seat": seat, "candidate": base, "extra_override": dict(key[2])}) for key in entries)
    return out, sizes, sup


def run_pass(bot: MatrixBot, obs, engine, result, cfg: TogglePass, seat: int | None = None):
    """The L1 pass on a converged oracle search's `result` (row, col, p1 value over the bot's view).
    Returns (result, info). Leaves the bot's cached result and replies updated in the turn cache."""
    t0 = time.perf_counter()
    info: dict[str, Any] = {"k": cfg.k}
    pool_fn = getattr(engine, "toggle_pool", None)
    oracle = engine.turn_cache.get("oracle") or {}
    if pool_fn is None or not oracle.get("converged") or oracle.get("fallback") or bot.policy is not None:
        info["skipped"] = "unsupported engine" if pool_fn is None else "not a converged unpruned oracle search"
        return result, info
    key = bot._key(seat)
    view = key
    cands = turn_candidates(obs, engine, view)
    row, col, value = result
    replies = engine.turn_cache.get((key, "replies")) or engine.turn_cache.get("replies")
    previewed0 = len(engine.turn_cache.get("previewed", {}))
    pools = [pool_fn(0), pool_fn(1)]
    specs, sizes, sup = pool_specs(obs, engine, view, (row, col), replies, pools, cfg.k,
                                   cfg.support_children)
    info["pool"] = sizes
    info["support_specs"] = sup
    info["specs"] = len(specs)
    asked: set = engine.turn_cache.setdefault("expanded", {}).setdefault(view, set())
    results: dict = engine.turn_cache.setdefault("expansions", {})
    keys = list(dict.fromkeys(k for k, _ in specs if k not in asked))
    asked.update(keys)
    fresh = set(keys)
    todo = list({k: s for k, s in specs if k in fresh and k not in results}.items())
    have: list[dict[str, list[int]]] = [{}, {}]
    for s in (0, 1):
        for x in cands[s]:
            have[s].setdefault(_content(x), []).append(x.id)
    if todo:
        _ask(obs, engine, todo)
    new: list[list[int]] = [[], []]
    for k in keys:
        s = k[0]
        c = _resolve(obs, engine, k, have[s])
        if c is None:
            continue
        cands[s].append(c)
        have[s].setdefault(_content(c), []).append(c.id)
        new[s].append(len(cands[s]) - 1)
    info["scanned"] = [len(new[0]), len(new[1])]
    info["ids"] = [[cands[s][i].id for i in new[s]] for s in (0, 1)]
    # Scan: each new entry against the other seat's final support.
    r_supp = [i for i, p in enumerate(row) if p > 0]
    c_supp = [j for j, p in enumerate(col) if p > 0]
    pairs = [(i, j) for i in new[0] for j in c_supp] + [(i, j) for j in new[1] for i in r_supp]
    vals = dict(zip(pairs, _values(bot, obs, engine, pairs, view))) if pairs else {}
    info["scan_pairs"] = len(pairs)
    gains: list[list[tuple[float, int]]] = [[], []]
    for i in new[0]:
        gains[0].append((sum(col[j] * vals[i, j] for j in c_supp) - value, i))
    for j in new[1]:
        gains[1].append((value - sum(row[i] * vals[i, j] for i in r_supp), j))
    admit = [[i for g, i in gains[s] if g > cfg.margin] for s in (0, 1)]
    info["admitted"] = [len(admit[0]), len(admit[1])]
    info["max_gain"] = [round(max((g for g, _ in gains[s]), default=0.0), 6) for s in (0, 1)]
    out = (row + [0.0] * len(new[0]), col + [0.0] * len(new[1]), value)
    resolved = False
    if admit[0] or admit[1]:
        # Re-solve on the searched candidates plus the admitted ones (entries not admitted stay listed
        # with probability 0: they were checked against the support the pass scanned).
        kept = [[i for i in range(len(row))] + admit[0], [j for j in range(len(col))] + admit[1]]

        def sub(ps):
            return _values(bot, obs, engine, [(kept[0][a], kept[1][b]) for a, b in ps], view)

        pos = [{x: n for n, x in enumerate(kept[s])} for s in (0, 1)]
        start = ([pos[0][i] for i in r_supp + admit[0]], [pos[1][j] for j in c_supp + admit[1]])
        r = double_oracle(len(kept[0]), len(kept[1]), sub, bot.iterations, bot.tolerance, start=start)
        info["rounds"] = r.rounds
        info["converged"] = r.converged
        if r.converged:
            full = [[0.0] * len(cands[0]), [0.0] * len(cands[1])]
            for s, st in ((0, r.row), (1, r.col)):
                for x, p in zip(kept[s], st):
                    full[s][x] = p
            out = (full[0], full[1], r.value)
            u = _replies(r)
            rep = ([float("-inf")] * len(cands[0]), [float("-inf")] * len(cands[1]))
            for s in (0, 1):
                for x, v in zip(kept[s], u[s]):
                    rep[s][x] = v
            engine.turn_cache["replies"] = rep
            resolved = True
        info["value_gain"] = round(out[2] - value, 6)
    rep = engine.turn_cache.get("replies")
    if not resolved and rep is not None:  # the new entries: their value against the support scanned
        r0 = list(rep[0]) + [sum(col[j] * vals[i, j] for j in c_supp) for i in new[0]]
        r1 = list(rep[1]) + [-sum(row[i] * vals[i, j] for i in r_supp) for j in new[1]]
        engine.turn_cache["replies"] = (r0, r1)
    engine.turn_cache[(key, "replies")] = engine.turn_cache.get("replies")
    engine.turn_cache[key] = out
    info["previews"] = len(engine.turn_cache.get("previewed", {})) - previewed0
    info["seconds"] = round(time.perf_counter() - t0, 4)
    oracle["toggle_pass"] = info
    return out, info


def _values(bot, obs, engine, pairs, view) -> list[float]:
    states = preview_states(obs, engine, pairs, view)
    batch = getattr(bot.value, "batch", None)
    return list(map(float, batch(states, 0))) if batch else [bot.value(st, 0) for st in states]


class ToggleMatrixBot(MatrixBot):
    """MatrixBot with the L1 pass after each full oracle decision (`toggle_pass`; None: exactly
    MatrixBot). Its search key carries the pass, so its cached results never mix with a plain bot's.
    A playout-capped (cheap) decision runs no pass: `cheap()` returns a bot without it."""

    def __init__(self, *args, toggle_pass: TogglePass | None = TogglePass(), **kw):
        super().__init__(*args, **kw)
        self.toggle_pass = toggle_pass
        if toggle_pass is not None:
            self.name += "-l1"

    @classmethod
    def wrap(cls, bot: MatrixBot, toggle_pass: TogglePass) -> "ToggleMatrixBot":
        """`bot` (a MatrixBot as make_bot built it: its value, search settings and exploration) with the pass."""
        out = cls.__new__(cls)
        out.__dict__.update(bot.__dict__)
        out.toggle_pass = toggle_pass
        out.name = bot.name + "-l1"
        return out

    def _key(self, seat: int | None = None) -> tuple:
        key = super()._key(seat)
        return key if self.toggle_pass is None else key + (("toggle_pass", self.toggle_pass),)

    def search_turn(self, obs, engine, seat: int | None = None):
        if self.toggle_pass is None or self.mode != "oracle" or not self.refine_rounds:
            return super().search_turn(obs, engine, seat)
        key = self._key(seat)
        cached = engine.turn_cache.get(key)
        if cached is not None:
            return cached
        result = super().search_turn(obs, engine, seat)
        out, _ = run_pass(self, obs, engine, result, self.toggle_pass, seat)
        return out

    def cheap(self) -> MatrixBot:
        bot = super().cheap()
        bot.toggle_pass = None
        return bot

    def config(self) -> dict[str, Any]:
        c = super().config()
        if self.toggle_pass is not None:
            c["toggle_pass"] = {"k": self.toggle_pass.k, "margin": self.toggle_pass.margin,
                                "support_children": self.toggle_pass.support_children}
        return c

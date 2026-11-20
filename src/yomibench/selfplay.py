"""Record self-play games for training: one gzipped JSONL file per game.

A record stores every decision's full state and candidate list (action, data, extra), so training
never needs to re-simulate, and ids stay meaningful even after the engine's candidate list
changes (`candidate_set` in the header says which list they index). Layout of a file:

    {"format": 3, "characters": [...], "seed": ..., "bots": [{config}, {config}],
     "engine": {hello}, "max_turns": ..., "opening": {"turns", "temperature", "epsilon"} | null,
     "solver": "native" | "python", "previews": "slim" | "full",
     "full_previews_fraction": F | null,
     "playout_cap": {"cheap_fraction", "cheap_cap", "cheap_cap_policy"?} | null,
     "full_previews_max_pairs": N?, "start": {"kind": "sampled", ...}?, "yomibench": git rev, ...}  header
    {"turn": ..., "state": ..., "actionable": ..., "candidates": [[...], [...]],
     "strategies": [[...], [...]] | null, "value": v_p1 | null,
     "candidate_values": [[...], [...]] | null, "previews": n,
     "previewed": [[i, j, state], ...]?, "previews_dropped": true?, "picks": [i, j], "opening": true?,
     "explore": [bool, bool]?, "explore_mode": "softmax"?, "explore_pick": {...}?, "oracle": {...}?,
     "search": "cheap" | "full"?, "ms": ...}                                          per decision
    {"result": {"winner": 0 | 1 | null, "outcome": "p1" | "p2" | "draw" | "truncated",
     "end": "ko" | "time" | "max_turns", "turns": ..., "hp": [...], ...}}            last line

`value` is p1's guaranteed value of the solved turn (p2's is its negative). `candidate_values` are
the search's value of every candidate against the other seat's solved strategy, each seat from its
own side (the bot's `replies`), rounded to 4 places: [0][i] is p1's value of playing its candidate
i against p2's mix, [1][j] p2's value of its candidate j against p1's mix (null for a value the
search never evaluated). They are written even without previews (they are the policy target, and
the search can't be rerun later); null when the turn had no search or the bot doesn't report
them, and absent in records from before them. `previewed` holds every
next state the search previewed (all n0 x n1 pairs for a full search, a subset for double oracle),
so a later value net can re-solve the turn offline. It is most of a record's size (set 4 previews
about three times as many pairs per turn), so records are slim by default: `previewed` is left out
and `previews` still counts the pairs. A seeded share of games (`full_previews_fraction`, default
0.01, chosen per game from its seed alone, so shards and reruns agree) keep it; the header's
`previews` says which a record is. A full game keeps `previewed` only on turns that previewed at most
`full_previews_max_pairs` pairs (header, when set; selfplay --full-previews-max-pairs, default 20000):
a turn over it (a rare expand-everything turn previews hundreds of thousands of pairs, one JSON line of
GBs) is recorded as a slim game records it, plus `"previews_dropped": true` (`previews` still counts
its pairs), so readers must not assume every turn of a full game has `previewed`. Everything
training reads (states, values, picks, results, exploration marks, candidate values) is the same in
both. A decision marked `"opening": true` was sampled
for variety (see `play_recorded`'s `opening`), not from `strategies`; `explore[seat]` true means
that seat's pick was exploration, not from the strategy: uniform over all its candidates (opening
noise) unless `explore_mode` says otherwise (`"softmax"`: sampled from a low-temperature softmax
of `replies`, near the policy). A `"least-tried"` pick (yomibench.least_tried) also carries
`explore_pick`: {"kind": "least-tried", "seat", "move_type", "options", "counts_at_pick", "matched", "of"}.

A header with `start` (yomibench.sampled_starts) is a sampled-start game: it began at decision `start.k`
of another record (`source_seed`, `source_sha256`, the picks to get there in `prefix_inputs`), so its
first row is decision k, not the opening (the true decision index of row r is k + r). Games from the
opening have no `start`.

Playout cap randomization (KataGo's; `play_recorded`'s `playout_cap=(P, N)`, header
`playout_cap`): each decision is cheap with probability P, drawn from the match seed and turn
alone (`cheap_turn`), otherwise full. A cheap decision searches only `cap_candidates(N)` of each
seat's candidates (the engine's per-seat budget rule, applied to this decision; `playout_cap=(P, N,
POLICY)` caps by another of its policies, recorded as `cheap_cap_policy`), with no factor
refinement or factor exploration (`MatrixBot.cheap`) and no opening or softmax exploration; its
`candidates`, `strategies`, `candidate_values` and `picks` are over that capped list. Both seats
share the decision's mode, recorded as `"search": "cheap"` or `"full"` on every decision of a game
played with a cap. Training keeps outcome targets on every decision but mixes in the search value
only on full ones (a cheap value is a weak target); decisions without the mark are full.

Files are written to a temp name and renamed, named by seed, so a rerun with the same seeds skips
finished games (any number of machines can write into one shared folder without coordinating as
long as their seed ranges don't overlap). A game whose engine crashes is dropped and logged to
drops.jsonl, never half-written.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

from yomibench import phases
from yomibench.bots import opening_weights, softmax_weights, turn_candidates
from yomibench.playout import cap_candidates, cheap_turn  # noqa: F401  (selfplay's names too)
from yomibench.engine import Engine, EngineCrashed, EngineError, Observation
from yomibench.ledger import add_selfplay_counts
from yomibench.least_tried import LeastTried
from yomibench.sampled_starts import SKIPS as SAMPLED_SKIPS
from yomibench.sampled_starts import begin as begin_sampled
from yomibench.search import SOLVER

# Category exploration (the draw arm): with probability E per eligible full-turn
# decision, one eligible seat plays a uniform pick among its candidates of a move category instead of
# its strategy. Marked explore + explore_mode "category:<name>", so (like any non-softmax exploration)
# positions before it train on value targets and positions after it keep outcome targets.
COWBOY_GUN = frozenset({"Brandish", "Guntrick", "Shoot2", "PointBlank", "ShootDodge2"})


def _cowboy_gun(c: Any) -> bool:
    """A Cowboy gun candidate: a gun move, or any attack with its Draw toggle on (extra gun_cancel)."""
    return c.action in COWBOY_GUN or (isinstance(c.extra, dict) and bool(c.extra.get("gun_cancel")))


CATEGORIES: dict[str, tuple[str, Callable[[Any], bool]]] = {"cowboy-gun": ("Cowboy", _cowboy_gun)}


def parse_category(spec: str) -> tuple[str, float]:
    """--explore-category NAME:E -> (NAME, E); NAME must be in CATEGORIES and 0 < E <= 1."""
    name, _, eps = spec.partition(":")
    if name not in CATEGORIES:
        raise ValueError(f"unknown exploration category {name!r}; known: {sorted(CATEGORIES)}")
    e = float(eps)
    if not 0 < e <= 1:
        raise ValueError(f"category epsilon must be in (0, 1], got {e}")
    return name, e

FORMAT = 3
# The candidate set self-play asks for when none is given (None: the engine's own default);
# tools/pool_eval.py uses the same default, so gates play what self-play recorded.
DEFAULT_CANDIDATE_SET: int | None = None


@functools.lru_cache(maxsize=1)
def git_revision() -> str:
    try:
        here = Path(__file__).resolve().parent
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _seat_state_names(obs: Observation) -> list:
    """Each seat's own engine state name (the state-rarity boost's state classes); None where absent."""
    players = (obs.state or {}).get("players") if isinstance(obs.state, dict) else None
    return [(players[s] or {}).get("state") if players and len(players) > s else None for s in (0, 1)]


def _least_expand(obs: Observation, engine: Engine, seat: int, idx: list[int]) -> int:
    """Least-tried "expand": every factor variant (one factor changed, DI included) of the listed
    candidates of `seat`, through the search's own expansion (expand_near_support with those
    candidates as off-support extras), added to this decision's full lists. Returns how many."""
    from yomibench.bots import expand_near_support

    full = turn_candidates(obs, engine)
    zero = ([0.0] * len(full[0]), [0.0] * len(full[1]))
    extra = (set(idx), set()) if seat == 0 else (set(), set(idx))
    return expand_near_support(obs, engine, zero, extra=extra, all_di=True)


def _cands(cands: list[list[Any]]) -> list[list[dict[str, Any]]]:
    return [[c.record() for c in seat] for seat in cands]


def _end_reason(obs: Observation) -> str:
    if not obs.done:
        return "max_turns"
    return "ko" if any(p["hp"] <= 0 for p in obs.state["players"]) else "time"


def play_recorded(engine: Engine, bot: Any, characters: tuple[str, str], seed: int, hello: dict[str, Any],
                  max_turns: int = 1000, max_candidates: int | None = None,
                  full_previews: bool = False, out: RecordWriter | None = None,
                  opening: tuple[int, float, float] | None = None,
                  explore_cfg: tuple[float, float] | None = None,
                  full_previews_fraction: float | None = None,
                  playout_cap: tuple[Any, ...] | None = None,
                  explore_category: tuple[str, float] | None = None,
                  explore_uniform_moves: float | None = None,
                  full_previews_max_pairs: int = 0,
                  explore_least_tried: LeastTried | None = None,
                  sampled_start: tuple[Any, Observation] | None = None) -> list[dict[str, Any]]:
    """One self-play game of `bot` against itself; returns the record's lines.

    `sampled_start=(sampled_starts.Start, observation)`: the game begins where `sampled_starts.begin` put the
    engine (`observation`, decision k of a source record) instead of a new match: the header carries
    `start` (Start.header), the rows begin at decision k, the first row's state carries the match's
    `raw_start`, and there is no opening window (decision k is past it). None: from the opening, as before.

    `bot.search_turn(obs, engine)` solves the turn once for both seats; each pick is sampled from
    that seat's certified strategy, the way the bot plays. A turn where neither seat
    has a choice is not searched. With `full_previews`, the turn's previewed next states are kept,
    so later value nets can re-solve every recorded turn without the engine; otherwise the record
    is slim (see the module docstring; `full_previews_fraction` only goes into the header). With
    `out`, lines go to that writer as they are made and only the header and result are returned.

    `full_previews_max_pairs=N` (0: no cap): with `full_previews`, a turn that previewed more than N
    pairs is written slim and marked `"previews_dropped": true`, as is any turn whose engine says it
    dropped them (`engine.previews_dropped`: `batch_play.BatchEngine` stops asking for state dicts
    once a turn passes N, so the turn's states were never all held); N goes into the header.

    `opening=(N, T, E)` diversifies starts: for each seat's first N decisions with a real choice,
    with probability E the pick is uniform over all candidates (marked in `explore`), otherwise
    it samples the solved strategy at temperature T (`bots.opening_weights`; T=1 is the strategy).

    `explore_cfg=(E, T)`: on each decision outside that opening noise, with probability E one seat
    (chosen at random) samples a softmax at temperature T over each of its candidates' values
    against the opponent's solved mix, instead of the strategy; recorded in `explore` with
    `explore_mode: "softmax"`.

    `explore_category=(NAME, E)` (CATEGORIES): on a full-turn decision outside the opening window
    with no other exploration, where a seat playing the category's character has category candidates,
    with probability E one such seat (at random when both are) picks uniformly among them; recorded in
    `explore` with `explore_mode: "category:NAME"`. Its own RNG stream: off, records are unchanged.

    `explore_uniform_moves=E` (the generic exploration arm: no move or character is named): on a
    full-turn decision outside the opening window with no other exploration, with probability E one seat
    chosen at random picks a **move** uniformly among its distinct actions, then a candidate of that move
    uniformly (so moves with many data variants aren't favoured); recorded in `explore` with
    `explore_mode: "uniform-moves"`. Its own RNG stream: off, records are unchanged.

    `explore_least_tried` (yomibench.least_tried.LeastTried; generic, no move or character is named):
    on a full-turn decision outside the opening window with no other exploration, with probability p
    one seat (at random among those under k such picks this game, with more than one move type and
    not tapered off for this game) plays its least-picked legal move type by the counts file, and
    within it the least-picked value of each option (ties at random; least_tried.LeastTried.choose);
    recorded in `explore` with `explore_mode: "least-tried"` and `explore_pick`, the header carrying
    the settings and the counts file's sha256. With a share of games below 1 (production's slice), only
    games whose seed draws under it (`LeastTried.game_selected`) explore; the header says which
    ("selected"), and the other games' decision and result lines are those of a game without it.
    Its own RNG stream: off, records are unchanged.

    `playout_cap=(P, N)` or `(P, N, POLICY)`: playout cap randomization (see the module docstring): a
    `cheap_turn` share P of decisions search `cap_candidates(N, POLICY)` per seat (salted with the
    seed, turn and seat; POLICY absent or "list": today's rule) with `bot.cheap()` and no opening or
    softmax exploration (a cheap decision doesn't count toward the opening window); every decision
    is marked `"search"`."""
    start = time.perf_counter()
    cap_policy = playout_cap[2] if playout_cap and len(playout_cap) > 2 else "list"
    rngs = (random.Random(f"{seed}:0"), random.Random(f"{seed}:1"))
    lines: list[dict[str, Any]] = [{
        "format": FORMAT, "characters": list(characters), "seed": seed,
        "bots": [bot.config(), bot.config()], "engine": hello, "max_candidates": max_candidates,
        "max_turns": max_turns,
        "opening": {"turns": opening[0], "temperature": opening[1], "epsilon": opening[2]} if opening else None,
        "solver": SOLVER, "previews": "full" if full_previews else "slim",
        "full_previews_fraction": full_previews_fraction,
        "explore": {"epsilon": explore_cfg[0], "temperature": explore_cfg[1]} if explore_cfg else None,
        "playout_cap": ({"cheap_fraction": playout_cap[0], "cheap_cap": playout_cap[1],
                         **({"cheap_cap_policy": playout_cap[2]} if cap_policy != "list" else {})}
                        if playout_cap else None),
        "yomibench": git_revision(), "platform": platform.system(),
        "started": time.time(),
    }]
    if explore_category:
        lines[0]["explore_category"] = {"name": explore_category[0], "epsilon": explore_category[1]}
    if explore_uniform_moves:
        lines[0]["explore_uniform_moves"] = {"epsilon": explore_uniform_moves}
    # A production slice (`games` below 1) explores in its selected games only: the others play exactly
    # as without it, with only this header entry ("selected": false) to tell them apart.
    least = explore_least_tried if explore_least_tried and explore_least_tried.game_selected(seed) else None
    if explore_least_tried:
        lines[0]["explore_least_tried"] = explore_least_tried.header(least is not None)
    if full_previews_max_pairs:
        lines[0]["full_previews_max_pairs"] = full_previews_max_pairs
    if sampled_start is not None:
        lines[0]["start"] = sampled_start[0].header()
    div0 = _divergence(engine)
    if out is not None:
        out.write(lines[0])
    engine_stats = getattr(engine, "stats", {})
    engine_stats["max_preview_pairs"] = 0  # this game's largest preview request (Engine.stats)
    if sampled_start is None:
        obs = engine.new_match(characters[0], characters[1], seed, max_candidates)
        decisions = [0, 0]  # each seat's decisions with a real choice so far (opening window)
    else:  # a sampled start: the engine already stands at decision k, past any opening window
        obs = sampled_start[1]
        decisions = [opening[0], opening[0]] if opening else [0, 0]
    raw_start = sampled_start[0].raw_start if sampled_start is not None else None
    # `fallback_turns`: turns where matrix-do's double oracle didn't converge and it solved the whole
    # (kept) table (the record's `oracle.fallback`); `max_preview_pairs`: the largest preview request.
    perf = {"decisions": 0, "searched": 0, "previews": 0, "preview_pairs": 0, "preview_ms": 0.0, "search_ms": 0.0,
            "fallback_turns": 0, "max_preview_pairs": 0}
    if playout_cap:
        perf.update(cheap_searched=0, cheap_previews=0)
        cheap_bot = bot.cheap()
    explore_rng = random.Random(f"{seed}:explore")
    category_rng = random.Random(f"{seed}:category")
    moves_rng = random.Random(f"{seed}:uniform-moves")
    if least:
        perf["least_tried"] = 0
        least_rng = random.Random(f"{seed}:least-tried")
        least_used = [0, 0]
        least_on = [least.seat_on(seed, seat, characters[seat]) for seat in (0, 1)]
    while not obs.done and obs.turn < max_turns:
        t = time.perf_counter()
        line: dict[str, Any] = {"turn": obs.turn, "state": obs.state, "actionable": list(obs.actionable)}
        if raw_start is not None:  # a sampled start's first row: the match's start values, as an opening row has
            if "raw_start" not in obs.state:
                line["state"] = {**obs.state, "raw_start": raw_start}
            raw_start = None
        cheap = bool(playout_cap) and cheap_turn(seed, obs.turn, playout_cap[0])
        if playout_cap:
            line["search"] = "cheap" if cheap else "full"
        if cheap:  # the decision's candidate lists, before any search reads them (turn_candidates)
            engine.turn_cache["base"] = [cap_candidates(cands, playout_cap[1], cap_policy, f"{seed}:{obs.turn}:{seat}")
                                         for seat, cands in enumerate(obs.candidates)]
        if all(len(seat) == 1 for seat in obs.candidates):
            picks = [0, 0]
            line.update(strategies=None, value=None, candidate_values=None, previews=0)
        else:
            pv0 = getattr(engine, "stats", {}).get("preview_seconds", 0.0)
            pp0 = getattr(engine, "stats", {}).get("preview_pairs", 0)
            ts = time.perf_counter()
            s0, s1, value = (cheap_bot if cheap else bot).search_turn(obs, engine)
            picks = []
            explore = [False, False]
            for seat, s in enumerate((s0, s1)):
                if opening and not cheap and len(s) > 1 and decisions[seat] < opening[0]:
                    # AlphaZero-style opening noise: with probability epsilon a uniform pick over
                    # every candidate (off the solved support), else the strategy at temperature T.
                    decisions[seat] += 1
                    line["opening"] = True
                    if rngs[seat].random() < opening[2]:
                        explore[seat] = True
                        picks.append(rngs[seat].randrange(len(s)))
                        continue
                    s = opening_weights(s, opening[1])
                picks.append(rngs[seat].choices(range(len(s)), weights=s)[0])
            replies = engine.turn_cache.get("replies")
            if explore_cfg and not cheap and not any(explore) and replies and explore_rng.random() < explore_cfg[0]:
                seat = explore_rng.randrange(2)
                w = softmax_weights(replies[seat], explore_cfg[1])
                if len(w) > 1 and sum(w) > 0:
                    picks[seat] = explore_rng.choices(range(len(w)), weights=w)[0]
                    explore[seat] = True
                    line["explore_mode"] = "softmax"
            if explore_category and not cheap and not any(explore) and not line.get("opening"):
                name, eps = explore_category
                char, is_member = CATEGORIES[name]
                now = turn_candidates(obs, engine)
                eligible = [(seat, idx) for seat in (0, 1) if characters[seat] == char and len(now[seat]) > 1
                            for idx in [[i for i, c in enumerate(now[seat]) if is_member(c)]] if idx]
                if eligible and category_rng.random() < eps:
                    seat, idx = eligible[category_rng.randrange(len(eligible))]
                    picks[seat] = category_rng.choice(idx)
                    explore[seat] = True
                    line["explore_mode"] = f"category:{name}"
            if explore_uniform_moves and not cheap and not any(explore) and not line.get("opening"):
                now = turn_candidates(obs, engine)
                seats = [seat for seat in (0, 1) if len({c.action for c in now[seat]}) > 1]
                if seats and moves_rng.random() < explore_uniform_moves:
                    seat = seats[moves_rng.randrange(len(seats))]
                    by_move: dict[str, list[int]] = {}
                    for i, c in enumerate(now[seat]):
                        by_move.setdefault(c.action, []).append(i)
                    move = sorted(by_move)[moves_rng.randrange(len(by_move))]
                    picks[seat] = moves_rng.choice(by_move[move])
                    explore[seat] = True
                    line["explore_mode"] = "uniform-moves"
            if least and not cheap and not any(explore) and not line.get("opening"):
                chosen = least.choose(turn_candidates(obs, engine), characters, least_used, least_on, least_rng,
                                      functools.partial(_least_expand, obs, engine) if least.expand else None,
                                      _seat_state_names(obs) if least.state_boost > 1 else None)
                if chosen:
                    seat, picks[seat], move, info = chosen
                    explore[seat] = True
                    least_used[seat] += 1
                    perf["least_tried"] += 1
                    line["explore_mode"] = "least-tried"
                    line["explore_pick"] = {"kind": "least-tried", "seat": seat, "move_type": move, **info}
            if any(explore):
                line["explore"] = explore
            previewed = engine.turn_cache.get("previewed", {})
            cv = engine.turn_cache.get("replies")
            line.update(strategies=[[round(p, 5) for p in s0], [round(p, 5) for p in s1]],
                        value=round(value, 5),
                        candidate_values=[[round(x, 4) if math.isfinite(x) else None for x in u] for u in cv]
                        if cv else None,
                        previews=len(previewed))
            perf["previews"] += len(previewed)
            if cheap:
                perf["cheap_searched"] += 1
                perf["cheap_previews"] += len(previewed)
            if full_previews and (getattr(engine, "previews_dropped", False)
                                  or 0 < full_previews_max_pairs < len(previewed)):
                line["previews_dropped"] = True  # over the cap: recorded as a slim game records it
            elif full_previews:
                line["previewed"] = [[i, j, st] for (i, j), st in sorted(previewed.items())]
        if "oracle" in engine.turn_cache:
            line["oracle"] = engine.turn_cache["oracle"]  # rounds, converged, fell back to full table
            perf["fallback_turns"] += bool(line["oracle"].get("fallback"))
        cands = turn_candidates(obs, engine)  # after the search: it may have added candidates (set 4)
        line["candidates"] = _cands(cands)
        line["picks"] = picks
        perf["decisions"] += 1
        if line["strategies"] is not None:  # searched: time the whole decision (search, exploration, refinement)
            perf["search_ms"] += (time.perf_counter() - ts) * 1000
            perf["preview_ms"] += (getattr(engine, "stats", {}).get("preview_seconds", 0.0) - pv0) * 1000
            perf["preview_pairs"] += getattr(engine, "stats", {}).get("preview_pairs", 0) - pp0
            perf["searched"] += 1
        line["ms"] = round((time.perf_counter() - t) * 1000)
        if out is not None:
            out.write(line)
        else:
            lines.append(line)
        t_step = time.perf_counter()
        obs = engine.step(cands[0][picks[0]].id, cands[1][picks[1]].id)
        phases.add("step", time.perf_counter() - t_step)  # timing only (the ledger's step_ms)
    perf["max_preview_pairs"] = engine_stats.get("max_preview_pairs", 0)
    # A game cut off by max_turns has no result; "truncated" keeps it out of outcome targets.
    if not obs.done:
        outcome = "truncated"
    else:
        outcome = "draw" if obs.winner is None else ("p1" if obs.winner == 0 else "p2")
    lines.append({"result": {
        "winner": obs.winner, "outcome": outcome, "end": _end_reason(obs), "turns": obs.turn, "done": obs.done,
        "hp": [p["hp"] for p in obs.state["players"]], "final_state": obs.state,
        "seconds": round(time.perf_counter() - start, 3),
        "perf": {k: round(v, 3) if isinstance(v, float) else v for k, v in perf.items()},
    }})
    div1 = _divergence(engine) if div0 is not None else None
    if div1 is not None:  # turns whose played pair previewed differently from the real step
        lines[-1]["result"]["divergence"] = {k: div1[k] - div0[k] for k in div0}
    if out is not None:
        out.write(lines[-1])
    return lines


class RecordWriter:
    """Writes one game's record line by line into a temporary gzip file, then renames it into place
    when the game finishes (a crash leaves no record). A game's previewed states run to tens of MB
    of JSON, so they are not kept in memory. The file is `<stem>-<sha1 prefix>.jsonl.gz`, the stem
    `g<seed>` by default (arena records add the seat order: both games of a pair share a seed)."""

    def __init__(self, out_dir: Path, seed: int, stem: str | None = None):
        self.out_dir, self.seed = out_dir, seed
        self.stem = stem or f"g{seed:012d}"
        self.tmp = out_dir / f".{self.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
        self.f = gzip.open(self.tmp, "wb")
        self.sha = hashlib.sha1()

    def write(self, line: dict[str, Any]) -> None:
        t = time.perf_counter()
        b = json.dumps(line, separators=(",", ":")).encode() + b"\n"
        self.sha.update(b)
        self.f.write(b)
        phases.add("record", time.perf_counter() - t)  # timing only (the ledger's record_ms)

    def finish(self) -> Path:
        t = time.perf_counter()
        self.f.close()
        phases.add("record", time.perf_counter() - t)
        path = self.out_dir / f"{self.stem}-{self.sha.hexdigest()[:12]}.jsonl.gz"
        os.replace(self.tmp, path)
        return path

    def abort(self) -> None:
        self.f.close()
        self.tmp.unlink(missing_ok=True)


def write_record(out_dir: Path, seed: int, lines: list[dict[str, Any]]) -> Path:
    body = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines).encode()
    digest = hashlib.sha1(body).hexdigest()[:12]
    path = out_dir / f"g{seed:012d}-{digest}.jsonl.gz"
    tmp = out_dir / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    with gzip.open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, path)
    return path


def read_record(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SlimTurn(NamedTuple):
    """One recorded decision, cut down to what training reads."""
    state: dict[str, Any]
    value: float | None               # p1's search value; None for a turn with no choice
    explore: tuple[bool, bool]        # each seat's pick was exploration, not from its strategy
    candidate_values: list[list[float]] | None = None  # see the module docstring; None if not recorded
    explore_mode: str | None = None   # how an exploratory pick was made: "uniform" or "softmax"; None if none
    cheap: bool = False               # searched cheaply (playout cap): its value is no training target
    strategies: list[list[float]] | None = None    # read_slim(policy=True) only: the search's mixed strategies
    candidates: list[list[dict[str, Any]]] | None = None  # read_slim(policy=True) only: both seats' lists
    turn: int | None = None           # read_slim(policy=True) only: the decision's turn number


def uniform_explore(turn: SlimTurn) -> bool:
    """An exploratory pick off the policy: opening noise, or any mode but near-policy softmax."""
    return any(turn.explore) and turn.explore_mode != "softmax"


def read_slim(path: Path, policy: bool = False) -> tuple[dict[str, Any], dict[str, Any], list[SlimTurn]]:
    """A record cut down to what training reads: header, result, and each turn's state, search
    value, exploration marks, candidate values and playout-cap mark. Parsing is most of training's loading time (the stored previews
    are large), so training runs this in worker processes and only the slim part crosses back. It
    lives here, not in train, so workers don't import torch. Fields a record doesn't have (older
    formats) come back as their defaults: a turn without a `search` mark is full. With `policy`, each
    turn also keeps its `strategies` and `candidates` (the policy net's targets and inputs;
    yomibench.policy); they are large, so the value net's loader leaves them out."""
    header: dict[str, Any] = {}
    last: dict[str, Any] = {}
    turns: list[SlimTurn] = []
    di: list[Any] = [{"x": 0, "y": 0}] * 2  # what each fighter's DI is: neutral at the start
    with gzip.open(path, "rt", encoding="utf-8") as f:  # line by line: a parsed turn with its
        for i, line in enumerate(f):                      # previews is large, so keep one at a time
            d = json.loads(line)
            if i == 0:
                header = d
            elif "result" in d:
                last = d
            else:
                _fill_di(d, di)
                explore = d.get("explore") or (False, False)
                explore = (bool(explore[0]), bool(explore[1]))
                # A mark without a mode is opening noise (uniform), as before set 4's softmax mode.
                mode = (d.get("explore_mode") or "uniform") if any(explore) else None
                turns.append(SlimTurn(d["state"], d.get("value"), explore, d.get("candidate_values"), mode,
                                      d.get("search") == "cheap",
                                      d.get("strategies") if policy else None,
                                      d.get("candidates") if policy else None,
                                      d.get("turn") if policy else None))
    return header, last, turns


def final_hp_margin(result: dict[str, Any], first_state: dict[str, Any] | None = None) -> float:
    """p1's hp fraction minus p2's at the end of a game (`search.evaluate`'s scale, in [-1, 1]), from
    a record's result line (`hp`; max hp from its `final_state`, else from `first_state`). NaN when
    the record doesn't say (the auxiliary loss skips those samples)."""
    res = result.get("result", result)
    players = (res.get("final_state") or first_state or {}).get("players")
    try:
        hp = res["hp"]
        lead = hp[0] / players[0]["max_hp"] - hp[1] / players[1]["max_hp"]
    except (TypeError, KeyError, IndexError, ZeroDivisionError):
        return float("nan")
    return max(-1.0, min(1.0, float(lead)))


def read_hp_margin(path: Path) -> float | None:
    """A record's `final_hp_margin` (None if unknown or truncated), reading only its first turn and
    result line (the training cache fills it in for games cached before it kept the margin)."""
    lines: list[str] = []  # the first turn's line and the last line, parsed only at the end
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 1:
                lines.append(line)
            elif i > 1:
                lines[1:] = [line]
    last = json.loads(lines[-1]) if lines else {}
    first = json.loads(lines[0]).get("state") if len(lines) > 1 else None
    res = last.get("result") or {}
    if not res or res.get("outcome") == "truncated":
        return None
    m = final_hp_margin(last, first)
    return None if m != m else m


def read_header(path: Path) -> dict[str, Any]:
    """A record's first line alone (cheap: the rest of the file is never decompressed)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.loads(f.readline())


def _fill_di(turn: dict[str, Any], di: list[Any]) -> None:
    """States from before engine state v3 have no `di`. A fighter's DI is exactly the DI of its
    last pick, so put that in (`di` carries it from turn to turn; None once unknown)."""
    for seat, p in enumerate(turn["state"]["players"]):
        if "di" not in p and di[seat] is not None:
            p["di"] = di[seat]
    try:
        di[:] = [turn["candidates"][s][turn["picks"][s]]["extra"]["DI"] for s in (0, 1)]
    except (KeyError, IndexError, TypeError):
        di[:] = [None, None]


def _remove_stale_temp_files(out_dir: Path, older_than: float = 1800.0) -> None:
    """Delete in-progress records left by a run that was killed. A live game writes every turn,
    so a temporary file untouched for half an hour belongs to no running game."""
    now = time.time()
    for tmp in out_dir.glob(".g*.tmp"):
        try:
            if now - tmp.stat().st_mtime > older_than:
                tmp.unlink()
        except OSError:
            pass


def done_seeds(out_dir: Path) -> set[int]:
    return {int(p.name[1:13]) for p in out_dir.glob("g*.jsonl.gz")}


def _reconnect(make: Callable[[], Engine], wait: float = 120.0,
               retry_busy: bool = False) -> tuple[Engine, dict[str, Any]]:
    """A crashed server is being restarted by its supervisor; keep trying until it answers.
    With `retry_busy`, a `busy` answer is retried too: after a network blip the server still
    holds our dead connection until its idle takeover (5 minutes)."""
    deadline = time.monotonic() + wait
    while True:
        try:
            engine = make()
            try:
                return engine, engine.hello()
            except BaseException:
                engine.close()
                raise
        except (OSError, EngineCrashed, EngineError) as e:
            if isinstance(e, EngineError) and not (retry_busy and "busy" in str(e)):
                raise
            if time.monotonic() > deadline:
                raise
            time.sleep(2.0)


# hello fields that must agree across a pool: a server running other code or rules would mix
# incompatible games into one dataset.
POOL_KEYS = ("engine", "version", "game_version", "candidate_set", "candidate_rules", "match_settings", "mod_md5")


def _divergence(engine: Engine) -> dict[str, int] | None:
    """The server's running preview-vs-step divergence counters, if it keeps them."""
    try:
        d = engine._call({"cmd": "stats"}).get("divergence")
    except (EngineError, KeyError, TypeError):
        return None
    return {"compared": d.get("compared", 0), "diverged": d.get("diverged", 0)} if isinstance(d, dict) else None


def full_previews(seed: int, fraction: float) -> bool:
    """Whether game `seed` keeps its previewed states: a fixed hash of the seed under `fraction`,
    so the choice depends on nothing else (not the process, shard or order of play)."""
    return fraction >= 1 or (fraction > 0 and int(hashlib.sha256(f"previews:{seed}".encode()).hexdigest()[:15], 16)
                             / 16 ** 15 < fraction)


def _add_game_stats(stats: dict[str, Any], result: dict[str, Any], characters: tuple[str, str]) -> None:
    stats["written"] = stats.get("written", 0) + 1
    stats["truncated"] = stats.get("truncated", 0) + (result["outcome"] == "truncated")
    stats["turns"] = stats.get("turns", 0) + result["turns"]
    stats["game_seconds"] = stats.get("game_seconds", 0.0) + result["seconds"]
    perf = result.get("perf", {})
    add_selfplay_counts(stats, perf)
    add_selfplay_counts(stats.setdefault("by_matchup", {}).setdefault(":".join(characters), {}), {
        "games": 1, **{k: perf.get(k, 0) for k in ("searched", "fallback_turns", "max_preview_pairs")}})


def fallback_counts(paths: Iterable[Path]) -> dict[str, dict[str, int]]:
    """Whole-table fallbacks counted from existing records (every format with `oracle` lines,
    which mark a turn whose double oracle didn't converge with `oracle.fallback`), by matchup
    ("P1:P2"): games, searched turns, fallback turns, and the most pairs a fallback turn previewed
    (its `previews`: the turn's previewed pairs, the whole kept table on a fallback turn). `paths`
    are record files or directories of them; e.g. `python -c "from pathlib import Path; from
    yomibench.selfplay import fallback_counts; print(fallback_counts([Path('DIR')]))"`."""
    out: dict[str, dict[str, int]] = {}
    for path in paths:
        for f in sorted(path.glob("g*.jsonl.gz")) if path.is_dir() else [path]:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                c: dict[str, int] | None = None
                for i, raw in enumerate(fh):
                    d = json.loads(raw)
                    if i == 0:
                        c = out.setdefault(":".join(d["characters"]), {
                            "games": 0, "searched_turns": 0, "fallback_turns": 0, "max_fallback_previews": 0})
                        c["games"] += 1
                    elif c is not None and d.get("strategies") is not None:
                        c["searched_turns"] += 1
                        if (d.get("oracle") or {}).get("fallback"):
                            c["fallback_turns"] += 1
                            c["max_fallback_previews"] = max(c["max_fallback_previews"], d.get("previews") or 0)
    return out


def _count_sampled(stats: dict[str, Any] | None, why: str | None) -> None:
    """One sampled slot in stats["sampled_starts"]: started from its sampled state (`why` None), or
    played from the opening for `why` (sampled_starts.SKIPS)."""
    if stats is None:
        return
    c = stats.setdefault("sampled_starts", {"slots": 0, "games": 0, **{k: 0 for k in SAMPLED_SKIPS}})
    c["slots"] += 1
    c["games" if why is None else why] = c.get("games" if why is None else why, 0) + 1


def run_selfplay(
    connects: list[Callable[[], Engine]],
    bot_factory: Callable[[], Any],
    characters: list[tuple[str, str]],
    seeds: Iterable[int],
    out_dir: Path,
    max_candidates: int | None = None,
    on_game: Callable[[int, list[dict[str, Any]] | None, str], None] | None = None,
    full_previews_fraction: float = 0.01,
    opening: tuple[int, float, float] | None = None,
    stats: dict[str, Any] | None = None,
    explore_cfg: tuple[float, float] | None = None,
    playout_cap: tuple[Any, ...] | None = None,
    explore_category: tuple[str, float] | None = None,
    explore_uniform_moves: float | None = None,
    full_previews_max_pairs: int = 0,
    explore_least_tried: LeastTried | None = None,
    claim: Callable[[int], bool] | None = None,
    owns: Callable[[int], bool] | None = None,
    playing: dict[int, Any] | None = None,
    guard: Any = None,
    sampled_starts: Any = None,
) -> tuple[int, int]:
    """Play every seed not already recorded in `out_dir`, one worker thread per engine. Game `seed`
    uses `characters[seed % len(characters)]`. Returns (games written, games dropped). Records are
    slim except for the games full_previews(seed, full_previews_fraction) picks, whose turns over
    `full_previews_max_pairs` previewed pairs are slim too (play_recorded). With `stats`
    (ledger.new_selfplay_stats()), adds this run's counts, timings and file sizes to it.
    With `claim`, a seed is played only if claim(seed) is True (processes sharing one seed list
    take seeds as they free up: see cli._selfplay_procs); `owns` then limits the skipped count to
    this process's share, so merged counters count each skipped seed once.
    `playing` (seed -> matchup of each game in play, kept here) and `guard` (the lock this run's
    bookkeeping takes) come from a memcap.MemoryCap, which takes `guard` for good before it ends
    the process: no game is then claimed, recorded or counted unseen.
    `sampled_starts` (a sampled_starts.StartPool; None: off, every game from the opening as before): the
    pool's seeded share of game pairs start from a state sampled from its records (sampled_starts module
    docstring), playing the source's characters; `stats["sampled_starts"]` counts the sampled slots
    ("slots"), the games started so ("games") and the slots that played from the opening instead, by
    reason (sampled_starts.SKIPS)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_temp_files(out_dir)
    have = done_seeds(out_dir)
    seeds = list(seeds)
    todo = iter([s for s in seeds if s not in have])
    if stats is not None:
        stats["skipped"] = stats.get("skipped", 0) + sum(s in have for s in seeds if owns is None or owns(s))
    lock = guard if guard is not None else threading.Lock()
    counts = [0, 0]
    errors: list[BaseException] = []
    stop = threading.Event()  # Ctrl-C: workers finish their current game's I/O and quit
    reference: list[dict[str, Any]] = []  # the first server's hello; every other must agree

    def worker(make: Callable[[], Engine]) -> None:
        engine: Engine | None = None
        bot = bot_factory()
        try:
            try:
                engine, hello = _reconnect(make)
            except EngineError as e:
                if "busy" not in str(e):
                    raise
                print(f"skipping a busy engine: {e}", flush=True)  # another client owns it
                return
            with lock:
                if not reference:
                    reference.append(hello)
                    if stats is not None:
                        stats["hello"] = hello
                mismatch = [k for k in POOL_KEYS if hello.get(k) != reference[0].get(k)]
            if mismatch:
                print(f"skipping an engine that differs from the pool in {mismatch}: "
                      f"{ {k: hello.get(k) for k in mismatch if k != 'match_settings'} }", flush=True)
                return
            while not stop.is_set():
                with lock:
                    seed = next(todo, None)
                    taken = seed is not None and (claim is None or claim(seed))
                    if taken and playing is not None:  # claimed and in play at once
                        playing[seed] = characters[seed % len(characters)]
                if seed is None:
                    return
                if not taken:
                    continue  # another process took it
                chars = characters[seed % len(characters)]
                phase0 = dict(phases.totals())  # this thread's phase times before the game (ledger timings)
                writer = RecordWriter(out_dir, seed)
                try:
                    start = None
                    if sampled_starts is not None:
                        sampled_starts.bind(hello)
                        source, why = sampled_starts.pick(seed)
                        if source is not None:
                            obs0, why = begin_sampled(engine, source, seed, max_candidates)
                            if obs0 is not None:
                                start, chars = (source, obs0), source.characters
                        if source is not None or why is not None:
                            with lock:
                                _count_sampled(stats, why)
                    full = full_previews(seed, full_previews_fraction)
                    lines = play_recorded(engine, bot, chars, seed, hello, max_candidates=max_candidates,
                                          full_previews=full, out=writer, opening=opening,
                                          explore_cfg=explore_cfg, full_previews_fraction=full_previews_fraction,
                                          playout_cap=playout_cap, explore_category=explore_category,
                                          explore_uniform_moves=explore_uniform_moves,
                                          full_previews_max_pairs=full_previews_max_pairs,
                                          explore_least_tried=explore_least_tried,
                                          **({"sampled_start": start} if start is not None else {}))
                except BaseException as e:
                    writer.abort()
                    if not isinstance(e, (EngineCrashed, EngineError)):
                        raise
                    with lock:
                        if playing is not None:
                            playing.pop(seed, None)
                        counts[1] += 1
                        if stats is not None:
                            stats["dropped"] = stats.get("dropped", 0) + 1
                        with open(out_dir / "drops.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps({"seed": seed, "characters": chars, "error": str(e),
                                                "time": time.time()}) + "\n")
                        if on_game:
                            on_game(seed, None, str(e))
                    engine.close()
                    engine = None
                    engine, hello = _reconnect(make, wait=360.0, retry_busy=True)
                    continue
                with lock:
                    path = writer.finish()
                    if playing is not None:
                        playing.pop(seed, None)
                    counts[0] += 1
                    if stats is not None:
                        _add_game_stats(stats, lines[-1]["result"], chars)
                        # Per-phase timings (net, solve, expand, step, record ms): stats only, never the record.
                        add_selfplay_counts(stats, {f"{k}_ms": (v - phase0[k]) * 1000
                                                    for k, v in phases.totals().items()})
                        kind = "full" if full else "slim"  # compressed bytes, by kind of record
                        stats[f"bytes_{kind}"] = stats.get(f"bytes_{kind}", 0) + path.stat().st_size
                        stats[f"games_{kind}"] = stats.get(f"games_{kind}", 0) + 1
                    if on_game:
                        on_game(seed, lines, "")
        except BaseException as e:  # surfaced after all workers stop
            print(f"a worker stopped: {type(e).__name__}: {e}", flush=True)
            with lock:
                errors.append(e)
        finally:
            if engine is not None:
                engine.close()

    # Daemon threads, joined with a timeout, so Ctrl-C reaches this thread: no new games start,
    # and a game still waiting on its engine is abandoned with the process (never half-written).
    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in connects]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:
        stop.set()
        raise
    if errors:
        raise errors[0]
    return counts[0], counts[1]

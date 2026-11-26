"""Bots pick one candidate id per turn. They may call `engine.preview` to look ahead."""

from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from yomibench.engine import Candidate, Engine, Observation
from yomibench.playout import CAP_POLICIES, cap_candidates, cheap_turn
from yomibench.search import SOLVER, double_oracle, evaluate, exploitability, solve_both


class Bot(Protocol):
    name: str

    def choose(self, obs: Observation, seat: int, engine: Engine, rng: random.Random) -> int: ...


class RandomBot:
    name = "random"

    def choose(self, obs: Observation, seat: int, engine: Engine, rng: random.Random) -> int:
        return rng.choice(turn_candidates(obs, engine)[seat]).id


class GreedyBot:
    """Assumes the opponent picks uniformly; maximizes the average previewed score."""

    name = "greedy"

    def choose(self, obs: Observation, seat: int, engine: Engine, rng: random.Random) -> int:
        payoff = _payoff_matrix(obs, seat, engine)
        means = [sum(row) / len(row) for row in payoff]
        best = max(means)
        return turn_candidates(obs, engine)[seat][rng.choice([i for i, m in enumerate(means) if m == best])].id


class MatrixBot:
    """Solves the turn's zero-sum (mine, theirs) table of previews and samples the mixed strategy.

    `search="full"` previews every pair. `search="oracle"` uses double oracle: it previews pairs
    only as needed, and stops only when neither side's strategy can be beaten by more than
    `tolerance` by ANY of the other side's moves (tools/do_check.py compares it with the full table
    on recorded turns). It starts from Continue plus `oracle_start` random moves per side: with many
    tied moves (the hand-written evaluation scores most neutral exchanges 0) a solver started from
    Continue alone would settle on Continue, and both bots would stand still all game. The random
    moves are drawn per match (seeded by the match seed, the turn and the seat, each seat from its
    own stream), so a position with several equilibria doesn't get the same one in every game, and
    neither seat's draw depends on the other's.

    Previewed states are shared through the engine's turn cache, so both seats' bots pay for a
    pair once; search results are cached per bot configuration, so two different bots (say a
    learned evaluator against the hand-written one) each run their own search.

    This is one-turn lookahead with a hand-written evaluation; swapping `evaluate` for a
    learned value network is the planned ML step.

    `policy` (prototype, oracle search only; default off, and off changes nothing): a
    `PolicyPrune` that restricts base candidates to a policy net's top `k` plus `random_k` seeded
    random others (Continue always stays), for the deciding seat only (`scope="own"`, gates and
    play against people: every opponent reply stays) or for both seats (`"both"`, self-play, where
    one search serves both). The search, factor refinement and exploration then run on that
    subgame, so its certificate covers the kept candidates only. On an `audit` share of decisions
    (seeded per decision) an independent unpruned search, with its own factor refinement, runs in a
    private turn cache, and each seat's best-response gain against the pruned answer over that
    search's candidates (and the pruned search's) is recorded in `oracle.policy`. The audit only
    measures: the pruned answer is played.
    """

    def __init__(self, iterations: int = 2000, value: Callable[[dict[str, Any], int], float] = evaluate,
                 search: str = "full", tolerance: float = 1e-3, oracle_start: int = 2, refine_rounds: int = 2,
                 di_refine_moves: int = 3,
                 candidate_set: int | None = None, explore: "FactorExplore | None" = None,
                 policy: "PolicyPrune | None" = None):
        if search not in ("full", "oracle"):
            raise ValueError(f"unknown search {search!r}")
        self.iterations = iterations
        self.value = value
        self.mode = search
        self.tolerance = tolerance
        self.oracle_start = oracle_start
        self.refine_rounds = refine_rounds
        self.di_refine_moves = di_refine_moves  # DI neighbours only for this many support moves per seat
        self.candidate_set = candidate_set  # asked of the engine for this bot's seat (arena A/B)
        self.explore = explore  # self-play only: off-support factor exploration (None: deterministic)
        # (P, N) or (P, N, POLICY): playout cap randomization on this bot's own decisions (`make_bot`'s
        # @cheap=P:N[:POLICY]; see _cheap_view); POLICY is `playout.cap_candidates`' (absent: "list").
        # None: every decision full. Self-play caps whole games itself instead.
        self.playout_cap: tuple[Any, ...] | None = None
        if policy is not None and search != "oracle":
            raise ValueError("policy pruning needs search='oracle'")
        self.policy = policy  # prototype: prune base candidates to a policy net's top K (None: off)
        # "batched" (`make_bot`'s @value_forward=batched): the Batch engine scores this bot's two views in one
        # value-net forward (`BatchValue(batched_forward=True)`; not bit-identical). None: two forwards.
        self.value_forward: str | None = None
        # Human play's turn timer only (yomibench.anytime): called with each double-oracle round's best-so-far
        # (row, col, p1 value, lists, converged). None everywhere else. Not part of the search key: it observes.
        self.progress: Callable[..., None] | None = None
        # Candidate set 7's di_fine_all children only for this many support candidates per seat, the most
        # likely first (`make_bot`'s @di_fine_moves=K); None: every support candidate (the default).
        self.di_fine_moves: int | None = None
        # Human play's quick policy only (yomibench.anytime with --quick-policy; default None, and None changes
        # nothing): an object with `scores(state, seat, candidates)` and `sha256`. The first double-oracle round
        # then starts from each seat's first candidate plus its `oracle_start` best-scored others instead of a
        # seeded random sample. Only the warm start changes: the search still certifies against every candidate.
        self.start_policy: Any = None
        self.name = "matrix" if search == "full" else "matrix-do"

    def _key(self, seat: int | None = None) -> tuple:
        """This bot's search key (and candidate view). An own-scope pruned search depends on the
        deciding seat, so it is keyed by it."""
        key = ("search", self.mode, self.iterations, self.tolerance, self.oracle_start, self.refine_rounds,
               self.di_refine_moves, self.explore, id(self.value))
        if getattr(self, "di_fine_moves", None) is not None:  # absent from the key at its default
            key += (("di_fine_moves", self.di_fine_moves),)
        if getattr(self, "start_policy", None) is not None:  # absent from the key at its default
            key += (("start_policy", self.start_policy.sha256),)
        if self.policy is None:
            return key
        return key + (self.policy,) + ((("own", seat),) if self.policy.scope == "own" else ())

    def search_turn(self, obs: Observation, engine: Engine, seat: int | None = None) -> tuple[list[float], list[float], float]:
        """Both seats' strategies and p1's guaranteed value for this turn. Previewed next states
        collect in engine.turn_cache["previewed"] as {(i, j): state}. `seat` is the deciding seat,
        which only an own-scope policy pruning reads (it needs one)."""
        if self.value_forward and not getattr(self.value, "batched_forward", False):
            raise ValueError(f"{self.name}: @value_forward={self.value_forward} needs --engine batch (the Batch "
                             f"engine's BatchValue does the batched forward), not {type(engine).__name__}")
        key = self._key(seat)
        cached = engine.turn_cache.get(key)
        if cached is not None:
            return cached
        view = key  # this bot's own candidate lists (see turn_candidates)
        cands = turn_candidates(obs, engine, view)
        n0, n1 = len(cands[0]), len(cands[1])

        def evaluate_pairs(pairs):
            states = preview_states(obs, engine, pairs, view)
            batch = getattr(self.value, "batch", None)  # a value net scores a list in one pass
            return list(map(float, batch(states, 0))) if batch else [self.value(st, 0) for st in states]

        if self.mode == "full":
            vals = evaluate_pairs([(i, j) for i in range(n0) for j in range(n1)])
            m = [vals[i * n1:(i + 1) * n1] for i in range(n0)]
            result = solve_both(m, self.iterations)
            engine.turn_cache["replies"] = _table_replies(m, result[0], result[1])
        else:
            match = getattr(engine, "match_seed", None)
            # The subgame the search plays: each seat's kept positions in this view (all, unless a
            # policy prunes). Strategies below are over kept positions; `full` maps them back.
            kept = [list(range(n0)), list(range(n1))]
            pinfo = None
            if self.policy:
                kept, pinfo = self.policy.keep(obs, cands, f"{match}:{obs.turn}:{obs.state.get('tick')}",
                                               self.policy.prune_seats(seat))

            base_known: dict[tuple[int, int], float] = {}

            def sub_pairs(pairs):
                v_pairs = [(kept[0][i], kept[1][j]) for i, j in pairs]
                vals = evaluate_pairs(v_pairs)
                base_known.update(zip(v_pairs, vals))
                return vals

            def full(strat, s):
                out = [0.0] * len(cands[s])
                for i, p in zip(kept[s], strat):
                    out[i] = p
                return out

            def full_replies(u):
                out = ([float("-inf")] * len(cands[0]), [float("-inf")] * len(cands[1]))
                for s in (0, 1):
                    for i, v in zip(kept[s], u[s]):
                        out[s][i] = v
                return out

            # `progress` (human play's turn timer, yomibench.anytime): sees every double-oracle round's strategies
            # over this view's lists as they stand then, so a search stopped part-way still has a best-so-far mix.
            # It only observes: the search is the same with or without it.
            progress = getattr(self, "progress", None)
            on_round = None
            if progress is not None:
                def on_round(res):
                    progress(full(res.row, 0), full(res.col, 1), res.value, [list(cands[0]), list(cands[1])],
                             res.converged)

            start_policy = getattr(self, "start_policy", None)
            if start_policy is None:
                start = tuple(
                    [0] + random.Random(f"{match}:{obs.turn}:{obs.state.get('tick')}:{s}")
                    .sample(range(1, n), min(self.oracle_start, n - 1))
                    for s, n in enumerate((len(kept[0]), len(kept[1]))))
            else:
                start = tuple(policy_start(start_policy.scores(obs.state, s, [cands[s][i] for i in kept[s]]),
                                           self.oracle_start) for s in (0, 1))
            r = double_oracle(len(kept[0]), len(kept[1]), sub_pairs, self.iterations, self.tolerance, start=start,
                              on_round=on_round)
            info = {"rounds": r.rounds, "converged": r.converged, "fallback": False}
            refined = 0
            ex = self.explore
            ex_rng = random.Random(f"{match}:{obs.turn}:{obs.state.get('tick')}:explore") if ex else None
            expand_all = bool(ex and ex_rng.random() < ex.all_prob)
            explored: dict[str, Any] = {}
            for rnd in range(self.refine_rounds):
                if not r.converged:
                    break
                extra = (set(), set())
                if ex:  # also near-optimal and random off-support candidates (all but DI factors)
                    u = full_replies(_replies(r))
                    for s in (0, 1):
                        v = r.value if s == 0 else -r.value
                        strat = full((r.row, r.col)[s], s)
                        off = [i for i in kept[s] if strat[i] <= 0 and cands[s][i].factors]
                        near = {i for i in off if ex.margin is not None and u[s][i] >= v - ex.margin}
                        rest = sorted(set(off) - near)
                        rand = set(ex_rng.sample(rest, min(ex.random_k, len(rest)))) if rnd == 0 else set()
                        extra[s].update(near | rand)
                        explored.setdefault("margin", [0, 0])[s] += len(near)
                        explored.setdefault("random", [0, 0])[s] += len(rand)
                    if expand_all and rnd == 0:
                        extra = tuple(set(k) for k in kept)
                        explored["all"] = True
                        explored["value_before"] = r.value
                lens = (len(cands[0]), len(cands[1]))
                added = expand_near_support(obs, engine, (full(r.row, 0), full(r.col, 1)), self.di_refine_moves,
                                            extra, all_di=expand_all and rnd == 0, view=view,
                                            fine_di_moves=getattr(self, "di_fine_moves", None))
                if not added:
                    break
                refined += added
                for s in (0, 1):  # a kept candidate's neighbours are kept too
                    kept[s] += range(lens[s], len(cands[s]))
                support = tuple([i for i, p in enumerate(st) if p > 0] for st in (r.row, r.col))
                r = double_oracle(len(kept[0]), len(kept[1]), sub_pairs, self.iterations, self.tolerance,
                                  start=support, on_round=on_round)
                info["rounds"] += r.rounds
                info["converged"] = r.converged
            if refined:
                info["refined"] = refined
            if explored:
                if explored.get("all"):
                    explored["gap"] = r.value - explored.pop("value_before")  # what expanding everything found
                info["explore"] = explored
            if r.converged:
                engine.turn_cache["replies"] = full_replies(_replies(r))
            if r.converged:
                result = (full(r.row, 0), full(r.col, 1), r.value)
            else:  # never play an uncertified answer: solve the (kept) full table instead
                k0, k1 = len(kept[0]), len(kept[1])
                vals = sub_pairs([(i, j) for i in range(k0) for j in range(k1)])
                m = [vals[i * k1:(i + 1) * k1] for i in range(k0)]
                row, col, value = _certified_full(m, self.iterations, self.tolerance)
                engine.turn_cache["replies"] = full_replies(_table_replies(m, row, col))
                result = (full(row, 0), full(col, 1), value)
                info["fallback"] = True
                info["fallback_gap"] = max(exploitability(m, row, col))
            if pinfo is not None:
                if pinfo.pop("audit"):
                    pinfo.update(self.audit(obs, engine, view, result))
                info["policy"] = pinfo
            engine.turn_cache["oracle"] = info
            engine.turn_cache[(key, "oracle")] = info
            engine.turn_cache[(key, "known")] = base_known
        # `replies` is shared, so the next bot's search overwrites it; keep this bot's (see `decision`).
        engine.turn_cache[(key, "replies")] = engine.turn_cache.get("replies")
        engine.turn_cache[key] = result
        return result

    def decision(self, obs: Observation, seat: int, engine: Engine) -> dict[str, Any]:
        """What this bot's search found for `seat`'s decision, for records and explanations: its own
        candidate lists (`lists`, its view, or its capped lists on a cheap decision), both seats'
        `strategies` over them, p1's `value`, and each seat's value of every candidate against the
        other seat's mix, from its own side (`replies`; -inf where never evaluated, None if the
        search kept none). Reads the search's cached results, so after `choose` it searches nothing."""
        view = self._cheap_view(obs, engine)
        if view:
            row, col, value = view["result"]
            return {"lists": view["candidates"], "strategies": (row, col), "value": value,
                    "replies": view.get("replies")}
        row, col, value = self.search_turn(obs, engine, seat)
        key = self._key(seat)
        return {"lists": turn_candidates(obs, engine, key), "strategies": (row, col), "value": value,
                "replies": engine.turn_cache.get((key, "replies"))}

    def audit(self, obs: Observation, engine: Engine, view: Any,
              result: tuple[list[float], list[float], float]) -> dict[str, Any]:
        """Check a pruned answer (`result`, over `view`'s lists) against an independent unpruned
        search: this bot without its policy, run in a private turn cache, so it refines factors
        around its own support and none of its candidates or previews reach the shared lists (or a
        self-play record). Returns each seat's best-response gain against the pruned answer over
        the candidates of both searches, the two values (p1's) and the candidate counts."""
        plain = copy.copy(self)
        plain.policy = None
        shared = engine.turn_cache
        base = shared.get("base", obs.candidates)
        nb = (len(base[0]), len(base[1]))
        # Base candidates come first in every cache's lists, so base x base previews carry over.
        private: dict[str, Any] = {"base": base, "previewed": {
            (i, j): st for (i, j), st in shared.get("previewed", {}).items() if i < nb[0] and j < nb[1]}}
        engine.turn_cache = private
        try:
            fr = plain.search_turn(obs, engine)
            audit_lists = turn_candidates(obs, engine, plain._key())
            memo = _states_by_id(private, turn_candidates(obs, engine))
        finally:
            engine.turn_cache = shared
        memo.update(_states_by_id(shared, turn_candidates(obs, engine)))
        gains = best_response_gains(engine, self.value, turn_candidates(obs, engine, view), audit_lists, result, memo)
        return {"audited": True, "audit_gain": [round(g, 6) for g in gains["gain"]],
                "audit_value": round(result[2], 6), "audit_full_value": round(fr[2], 6),
                "audit_candidates": gains["candidates"]}

    def cheap(self) -> "MatrixBot":
        """This bot for a playout-capped self-play decision (selfplay.play_recorded): the same
        search with no factor refinement and no factor exploration (no margin, random or
        expand-everything candidates), so it searches only the decision's given candidates."""
        bot = copy.copy(self)
        bot.refine_rounds, bot.explore, bot.policy = 0, None, None  # a capped list is already small
        return bot

    def _cheap_view(self, obs: Observation, engine: Engine) -> dict[str, Any] | None:
        """With `playout_cap` (P, N), on a cheap decision (`playout.cheap_turn` of the match seed and
        turn, as self-play draws it): this bot's own turn cache for it, holding its capped candidate
        lists (`cap_candidates(N)` of the engine's lists, both seats) and its `cheap()` search over
        them. The search runs once per decision on that private cache, so the opponent's bot never
        sees the capped lists or the cheap previews, and this bot never sees the opponent's
        expansions. None on a full decision: the bot is then exactly the plain bot."""
        if not self.playout_cap or not cheap_turn(engine.match_seed, obs.turn, self.playout_cap[0]):
            return None
        key = ("cheap view", self._key(), self.playout_cap)
        view = engine.turn_cache.get(key)
        if view is None:
            policy = self.playout_cap[2] if len(self.playout_cap) > 2 else "list"
            view = {"base": [cap_candidates(cands, self.playout_cap[1], policy, f"{engine.match_seed}:{obs.turn}:{seat}")
                             for seat, cands in enumerate(obs.candidates)]}
            shared, engine.turn_cache = engine.turn_cache, view
            try:
                view["result"] = self.cheap().search_turn(obs, engine)
            finally:
                engine.turn_cache = shared
            shared[key] = view
        return view

    def search(self, obs: Observation, seat: int, engine: Engine) -> tuple[list[float], float]:
        """This seat's equilibrium strategy over its candidates (unpruned) and its value."""
        view = self._cheap_view(obs, engine)
        row, col, value = view["result"] if view else self.search_turn(obs, engine, seat)
        return (row, value) if seat == 0 else (col, -value)

    def config(self) -> dict[str, Any]:
        return {"bot": self.name, "search": self.mode, "iterations": self.iterations, "prune": None,
                "tolerance": self.tolerance if self.mode == "oracle" else None,
                "oracle_start": self.oracle_start if self.mode == "oracle" else None,
                "refine_rounds": self.refine_rounds if self.mode == "oracle" else None,
                "di_refine_moves": self.di_refine_moves if self.mode == "oracle" else None,
                "explore": vars(self.explore) if self.explore else None,
                "solver": SOLVER,
                "candidate_set": self.candidate_set,
                "value": getattr(self.value, "__name__", repr(self.value)),
                # Only with selfplay --batched-net-forward or @value_forward=batched (not bit-identical to the default).
                **({"value_forward": "batched"} if getattr(self.value, "batched_forward", False)
                   or self.value_forward == "batched" else {}),
                **({"playout_cap": list(self.playout_cap)} if self.playout_cap else {}),
                **({"di_fine_moves": self.di_fine_moves} if getattr(self, "di_fine_moves", None) is not None else {}),
                **({"policy": self.policy.config()} if self.policy else {})}

    def choose(self, obs: Observation, seat: int, engine: Engine, rng: random.Random) -> int:
        strategy, _ = self.search(obs, seat, engine)  # played as solved: pruning would void the certificate
        (i,) = rng.choices(range(len(strategy)), weights=strategy)
        view = self._cheap_view(obs, engine)
        return (view["candidates"] if view else turn_candidates(obs, engine, self._key(seat)))[seat][i].id


class PolicyBot:
    """No search: samples its move from a policy net's softmax over the engine's base candidates,
    with temperature `temperature` (1: the net's own distribution, which imitates the search's
    mixed strategy; toward 0: its top move). No previews, no value net, no factor refinement, so
    it costs one net forward per decision and can be read only as far as its mix allows."""

    def __init__(self, net: Any, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError(f"policy temperature must be > 0: {temperature}")
        self.net, self.temperature = net, temperature
        self.name = f"policy:{getattr(net, 'meta', {}).get('name', '?')}"

    def strategy(self, obs: Observation, seat: int) -> list[float]:
        scores = [float(x) / self.temperature for x in self.net.scores(obs.state, seat, obs.candidates[seat])]
        top = max(scores)
        weights = [math.exp(x - top) for x in scores]
        total = sum(weights)
        return [w / total for w in weights]

    def config(self) -> dict[str, Any]:
        return {"bot": self.name, "search": None, "policy": getattr(self.net, "meta", {}).get("name", "?"),
                "temperature": self.temperature, "candidate_set": getattr(self, "candidate_set", None)}

    def choose(self, obs: Observation, seat: int, engine: Engine, rng: random.Random) -> int:
        (i,) = rng.choices(range(len(obs.candidates[seat])), weights=self.strategy(obs, seat))
        return obs.candidates[seat][i].id


def turn_candidates(obs: Observation, engine: Engine, view: Any = None) -> list[list[Candidate]]:
    """This decision's candidate lists.

    `view` None: every candidate any search added this decision with `expand`, in the order they
    were added (the record's lists; `previewed` and record indices refer to them). A bot's view
    (MatrixBot uses its search key): the engine's base lists plus only what that bot's own
    refinement asked for. Each bot searches and picks over its own view, so in a game between two
    bots neither sees what the other's search found, and the order the seats search in changes
    nothing (as in real play, where each player decides alone). Expansion results are shared as
    a cache (expand_near_support), so a candidate both bots ask for is built once. With one bot,
    as in self-play, its view and the full lists grow identically.

    The base lists are the engine's (`obs.candidates`) unless the turn cache sets `base`, as a
    playout-capped bot's private cache does (MatrixBot._cheap_view: its capped lists)."""
    base = engine.turn_cache.setdefault("base", obs.candidates)
    cands = engine.turn_cache.get("candidates")
    if cands is None:
        cands = engine.turn_cache["candidates"] = [list(base[0]), list(base[1])]
    if view is None:
        return cands
    views = engine.turn_cache.setdefault("views", {})
    if view not in views:
        views[view] = [list(base[0]), list(base[1])]
    return views[view]


def _union_index(obs: Observation, engine: Engine, view: Any) -> tuple[list[int], list[int]] | None:
    """For each seat, the position in the full lists of each candidate of `view` (None: no view)."""
    if view is None:
        return None
    vc, uc = turn_candidates(obs, engine, view), turn_candidates(obs, engine)
    out = []
    for seat in (0, 1):
        pos = {c.id: k for k, c in enumerate(uc[seat])}
        out.append([pos[c.id] for c in vc[seat]])
    return out[0], out[1]


FACTOR_DEFAULTS = {"di": 0, "feint": False, "reverse": False, "extra_set": 0, "aim": 0, "di_fine": 0,
                   "xy": 0, "di_fine_all": 0}
# Candidate set 7's fine path factors (the simulator's set-6 switches xy_fine, di_fine_all): asked only around support
# candidates, never for exploration extras (their cost model is children per support candidate). An engine
# without those switches on never offers them, so this changes nothing there.
SUPPORT_ONLY_FACTORS = frozenset({"xy", "di_fine_all"})


def _factor_values(factor: str, allowed: Any) -> list[Any]:
    """The values to ask for a factor. Booleans (feint, reverse) toggle; a count (di, extra_set)
    means indices 0..count-1; a list (candidate set 5's `aim` and `di_fine`) is the engine's own
    child values around this candidate, asked as given."""
    if factor in ("feint", "reverse"):
        return [False, True] if allowed else [False]
    if isinstance(allowed, (list, tuple)):
        return list(allowed)
    return list(range(int(allowed)))


def expand_near_support(obs: Observation, engine: Engine, strategies: tuple[list[float], list[float]],
                        di_moves: int | None = None, extra: tuple[set, set] = (set(), set()),
                        all_di: bool = False, view: Any = None, fine_di_moves: int | None = None) -> int:
    """Candidate set 4's factor refinement: for every candidate in either seat's support, ask the
    engine for its neighbours, the same move with one factor (DI, feint, reverse, extra set)
    changed, unless already asked this decision. Returns how many candidates were added.

    DI has the most values (17), so with `di_moves` only that many support candidates per seat,
    the most likely first, get DI neighbours (bot policy; the engine offers DI on every action).

    The search first solves the base candidates, then calls this and solves again with the
    neighbours available, so factors are only explored around moves worth playing; a neighbour
    that joins the support gets its own neighbours next round (combinations of factors).

    `extra` (self-play exploration) names off-support candidates per seat to expand too, with every
    factor except DI unless `all_di` (the occasional expand-everything decision), and never candidate set 7's
    support-only path factors (SUPPORT_ONLY_FACTORS).

    `strategies` and the additions are in `view`'s lists (see turn_candidates). What a view has
    asked for is its own; the engine's answers are shared (`expansions`: spec -> the candidate
    that spec builds, or None if it can't be built), so a spec another bot already asked for is
    taken from there, not asked again, and joins this view only because this bot asked for it.

    Candidate set 5's own expansions (an `aim` or `di_fine` path, or a Draw pair extra set) are
    refused by the engine when they repeat a candidate already listed this decision, by anyone
    (`same candidate as N`). Whether that happens depends on what was asked before, so it is not
    a property of the spec: each view applies it to its own lists instead (`_resolve`), and gets
    the candidate the spec builds when only another view holds the repeat."""
    cands = turn_candidates(obs, engine, view)
    full = turn_candidates(obs, engine)
    asked: set = engine.turn_cache.setdefault("expanded", {}).setdefault(view, set())
    results: dict = engine.turn_cache.setdefault("expansions", {})
    keys, specs = [], []
    for seat, strategy in enumerate(strategies):
        ranked = sorted((i for i, p in enumerate(strategy) if p > 0), key=lambda i: (-strategy[i], i))
        di_ok = set(ranked if di_moves is None else ranked[:di_moves])
        fine_di_ok = set(ranked if fine_di_moves is None else ranked[:fine_di_moves])
        for i, p in enumerate(strategy):
            c = cands[seat][i]
            if (p <= 0 and i not in extra[seat]) or not c.factors:
                continue
            base = c.id if c.base is None else c.base
            values = {f: (c.values or {}).get(f, FACTOR_DEFAULTS.get(f)) for f in c.factors}
            asked.add((seat, base, tuple(sorted(values.items()))))
            for f, allowed in c.factors.items():
                if f == "di" and i not in di_ok and not all_di:
                    continue
                if p <= 0 and f in SUPPORT_ONLY_FACTORS:
                    continue
                if f == "di_fine_all" and i not in fine_di_ok:
                    continue
                for v in _factor_values(f, allowed):
                    nv = {**values, f: v}
                    key = (seat, base, tuple(sorted(nv.items())))
                    if key not in asked:
                        asked.add(key)
                        keys.append(key)
                        if key not in results:
                            specs.append((key, {"seat": seat, "candidate": base, "extra_override": nv}))
    have: list[dict[str, list[int]]] = [{}, {}]  # per seat: this view's candidates by content, before asking
    for seat in {k[0] for k in keys}:
        for x in cands[seat]:
            have[seat].setdefault(_content(x), []).append(x.id)
    if specs:
        _ask(obs, engine, specs)
    added = 0
    for key in keys:  # in the order this view asked, so a lone bot's view matches the full lists
        seat = key[0]
        c = _resolve(obs, engine, key, have[seat])
        if c is None:
            continue
        if view is not None:  # the full lists got it when it was built
            cands[seat].append(c)
        have[seat].setdefault(_content(c), []).append(c.id)
        added += 1
    return added


# Set 5's path factors: digit base (a path's depth is its number of base-b digits).
PATH_FACTORS = {"aim": 3, "di_fine": 5}
# The engine's refusals of a path that doesn't exist (a step out of the aims, no DI step, too deep).
_PATH_ERROR = re.compile(r"^(aim|di_fine) \d+ (leaves |has no step |is not a path )")
_SAME_AS = re.compile(r"^same candidate as (\d+)$")


def _content(c: Candidate) -> str:
    """What the engine compares to call two candidates the same (action, data, extra)."""
    return json.dumps([c.action, c.data, c.extra])


def _ask(obs: Observation, engine: Engine, specs: list[tuple[tuple, dict[str, Any]]]) -> list[Any]:
    """Send `specs` ([(key, spec)]) to the engine and file its answers in the turn cache: a built
    candidate (also added to the full lists), or None with the reason in `refused` for a spec that
    can't be built, or, for a repeat (`same candidate as N`), None with N in `same_as` until a view
    needs the spec's own candidate. `dedupes` records the specs the engine refused as repeats (others
    are learned when a view needs to know: `_resolve`). Returns the raw answers."""
    full = turn_candidates(obs, engine)
    tc = engine.turn_cache
    results, same_as = tc.setdefault("expansions", {}), tc.setdefault("same_as", {})
    dedupes, refused = tc.setdefault("dedupes", {}), tc.setdefault("refused", {})
    answers = engine.expand([s for _, s in specs])
    for (key, spec), got in zip(specs, answers):
        if isinstance(got, Candidate):
            results[key] = got
            full[spec["seat"]].append(got)
            continue
        m = _SAME_AS.match(got)
        results[key] = None
        if m:
            same_as[key] = int(m.group(1))
            dedupes[key] = True
        else:
            refused[key] = got
    return answers


def _resolve(obs: Observation, engine: Engine, key: tuple, have: dict[str, list[int]]) -> Candidate | None:
    """What spec `key` adds to a view whose list for its seat holds `have` (content -> ids): as the
    engine would answer a bot alone with that list. That is the candidate the spec builds, unless
    it can't be built, or it is one the engine refuses as a repeat and the view already holds the
    same candidate.

    The engine's own answer differs only when another view (or an earlier search of this decision)
    built the repeat: then it refused a spec this view gets (`same_as`, made with `_built`), or it
    built one this view must refuse. A bot alone never meets either, so it asks nothing more."""
    seat = key[0]
    tc = engine.turn_cache
    results, same_as = tc.setdefault("expansions", {}), tc.setdefault("same_as", {})
    dedupes = tc.setdefault("dedupes", {})
    c = results.get(key)
    if c is None:
        n = same_as.get(key)
        if n is None:
            return None  # can't be built
        same = _listed(obs, engine, seat, n)
        if _content(same) in have:
            return None  # refused as a repeat, as it would be with this view's list alone
        c = results[key] = _built(obs, engine, key, same)
        del same_as[key]
        return c
    repeats = have.get(_content(c))
    if not repeats:
        return c
    if dedupes.get(key) is None:
        if min(repeats) < c.id:  # the engine built it while the repeat was listed: never refused
            dedupes[key] = False
        else:  # ask again: a spec that is refused as a repeat is now refused
            (again,) = engine.expand([_spec(key)])
            dedupes[key] = not isinstance(again, Candidate)
            if isinstance(again, Candidate):
                turn_candidates(obs, engine)[seat].append(again)
    return None if dedupes[key] else c


def _spec(key: tuple) -> dict[str, Any]:
    seat, base, values = key
    return {"seat": seat, "candidate": base, "extra_override": dict(values)}


def _listed(obs: Observation, engine: Engine, seat: int, n: int) -> Candidate:
    """The engine's candidate `n` of `seat` this decision, from the full lists, or else from the
    engine's own lists (one an earlier search of this decision built in another turn cache), then
    added to the full lists."""
    full = turn_candidates(obs, engine)
    for c in full[seat]:
        if c.id == n:
            return c
    (c,) = [c for c in engine.observe().candidates[seat] if c.id == n]
    full[seat].append(c)
    return c


def _built(obs: Observation, engine: Engine, key: tuple, same: Candidate) -> Candidate:
    """The candidate spec `key` builds, which the engine refused as a repeat of `same`: the same
    move (`same`'s id, action, data and extra) with the spec's own factor values and base. The
    engine makes a candidate's `values` from its root's and the override, its set-4 factors from
    the move, and each path factor's children from the move and the path's depth, so those are
    `same`'s when the depths agree; otherwise each child is asked for (a path refusal: none)."""
    seat, base, values = key
    nv = dict(values)
    root = next(c for c in turn_candidates(obs, engine)[seat] if c.id == base)
    rv = {**(root.values or {}), **nv}
    own = {f: rv.get(f, FACTOR_DEFAULTS.get(f)) for f in (same.values or {})}
    factors = dict(same.factors or {})
    probes = []
    for f, b in PATH_FACTORS.items():
        if f not in factors:
            continue
        depth, theirs = _depth(own[f], b), (same.values or {}).get(f, 0)
        if depth == _depth(theirs, b):
            factors[f] = [own[f] + (k - theirs) for k in factors[f]]
        else:
            step = b ** depth
            probes += [(f, own[f] + d * step) for d in range(1, b)]
    if probes:
        ov = {f: own.get(f, FACTOR_DEFAULTS.get(f)) for f in factors}
        kids = [(seat, base, tuple(sorted({**ov, f: v}.items()))) for f, v in probes]
        tc = engine.turn_cache
        known = set(tc.get("expansions", {})) | set(tc.get("same_as", {}))
        new = [k for k in dict.fromkeys(kids) if k not in known]
        if new:
            _ask(obs, engine, [(k, _spec(k)) for k in new])
        refused = tc.get("refused", {})
        for f in {f for f, _ in probes}:  # a child exists unless the engine refuses its path
            factors[f] = [v for (g, v), k in zip(probes, kids)
                          if g == f and not _PATH_ERROR.match(refused.get(k, ""))]
    return Candidate(same.id, same.action, same.data, same.extra, factors, own, base)


def _depth(path: int, b: int) -> int:
    n = 0
    while path:
        path //= b
        n += 1
    return n


def policy_start(scores: Any, k: int) -> list[int]:
    """A double-oracle warm start from policy scores (one per candidate): the first candidate, then the `k`
    best-scored others (ties by list order)."""
    n = len(scores)
    rest = sorted(range(1, n), key=lambda i: (-float(scores[i]), i))[:max(0, min(k, n - 1))]
    return [0] + rest


@dataclass(frozen=True)
class FactorExplore:
    """Self-play exploration of factors off the support (the arena never sets it). Besides the
    support, expand candidates whose value against the opponent's mix is within `margin` of the
    game value, plus `random_k` seeded random off-support candidates per seat per decision; on a
    `all_prob` share of decisions expand every candidate (all factors) and record the value gap
    that found in the decision's `oracle.explore`."""
    margin: float | None = 0.02  # None: no margin rule
    random_k: int = 2
    all_prob: float = 0.03


class PolicyPrune:
    """MatrixBot's policy pruning (prototype; see MatrixBot): keep Continue (position 0), the
    policy's top `k` base candidates and `random_k` seeded random others, for the seats `scope`
    names ("own": the deciding seat, every opponent reply kept; "both"); on an `audit` share of
    decisions, also run MatrixBot.audit. `net` scores candidates (yomibench.policy.PolicyNet)."""

    SCOPES = ("own", "both")

    def __init__(self, net: Any, k: int = 16, random_k: int = 4, audit: float = 0.05, scope: str = "own"):
        if scope not in self.SCOPES:
            raise ValueError(f"pruning scope must be one of {self.SCOPES}, not {scope!r}")
        if k < 1 or random_k < 0 or not 0 <= audit <= 1:
            raise ValueError(f"pruning needs k >= 1, random_k >= 0 and 0 <= audit <= 1: {k}, {random_k}, {audit}")
        self.net, self.k, self.random_k, self.audit, self.scope = net, k, random_k, audit, scope

    def config(self) -> dict[str, Any]:
        return {"net": getattr(self.net, "meta", {}).get("name", "?"), "k": self.k, "random_k": self.random_k,
                "audit": self.audit, "scope": self.scope}

    def prune_seats(self, seat: int | None) -> tuple[int, ...]:
        if self.scope == "both":
            return (0, 1)
        if seat is None:
            raise ValueError("own-scope pruning prunes the deciding seat's moves, so it needs search(obs, seat); "
                             "self-play, where one search serves both seats, uses scope 'both'")
        return (seat,)

    def keep(self, obs: Observation, cands: list[list[Candidate]], key: str,
             seats: tuple[int, ...]) -> tuple[list[list[int]], dict[str, Any]]:
        """Each seat's kept positions (pruned seats in policy order, Continue first) and notes for
        the record (`audit` is popped by the search)."""
        kept = []
        for seat in (0, 1):
            n = len(cands[seat])
            if seat not in seats or n <= self.k + self.random_k + 1:
                kept.append(list(range(n)))
                continue
            scores = self.net.scores(obs.state, seat, cands[seat])
            order = [i for i in sorted(range(n), key=lambda i: (-scores[i], i)) if i != 0]
            top, rest = order[:self.k], order[self.k:]
            rand = set(random.Random(f"{key}:policy:{seat}").sample(rest, min(self.random_k, len(rest))))
            kept.append([0] + top + [i for i in rest if i in rand])
        audit = random.Random(f"{key}:policy-audit").random() < self.audit
        return kept, {"kept": [len(k) for k in kept], "of": [len(c) for c in cands], "audit": audit}


def _states_by_id(cache: dict[str, Any], lists: list[list[Candidate]]) -> dict[tuple[int, int], dict[str, Any]]:
    """A turn cache's previewed states keyed by engine id pairs (its `previewed` is keyed by
    positions in its own full lists)."""
    return {(lists[0][i].id, lists[1][j].id): st for (i, j), st in cache.get("previewed", {}).items()}


def best_response_gains(engine: Engine, value: Callable[[dict[str, Any], int], float],
                        lists: list[list[Candidate]], other: list[list[Candidate]],
                        result: tuple[list[float], list[float], float],
                        memo: dict[tuple[int, int], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Each seat's best gain against `result` (strategies over `lists`, p1's value) by any
    candidate of `lists` or `other` (another search's lists for this decision; candidates are
    matched by engine id), and the engine id of each seat's best reply (`best`). Pairs not in
    `memo` (id pair -> previewed state) are previewed."""
    memo = {} if memo is None else memo
    row, col, v = result
    supp = ([(lists[0][i].id, p) for i, p in enumerate(row) if p > 0],
            [(lists[1][j].id, p) for j, p in enumerate(col) if p > 0])
    universe = [list(dict.fromkeys([c.id for c in lists[s]] + [c.id for c in other[s]])) for s in (0, 1)]
    pairs = [(u, j) for u in universe[0] for j, _ in supp[1]] + [(i, u) for u in universe[1] for i, _ in supp[0]]
    missing = [q for q in dict.fromkeys(pairs) if q not in memo]
    if missing:
        memo.update(zip(missing, (r.state for r in engine.preview(missing, candidates=False))))
    states = [memo[q] for q in pairs]
    batch = getattr(value, "batch", None)
    vals = dict(zip(pairs, map(float, batch(states, 0)) if batch else (value(st, 0) for st in states)))
    u0 = {u: sum(p * vals[(u, j)] for j, p in supp[1]) for u in universe[0]}
    u1 = {u: -sum(p * vals[(i, u)] for i, p in supp[0]) for u in universe[1]}
    best = [max(u0, key=u0.get), max(u1, key=u1.get)]  # first of the best, in universe order
    return {"gain": [u0[best[0]] - v, u1[best[1]] + v], "candidates": [len(universe[0]), len(universe[1])],
            "best": best}


def _replies(r) -> tuple[list[float], list[float]]:
    """Each seat's value (its own side) of every candidate against the other seat's solved mix,
    from the payoffs the double oracle evaluated (it has every row against the column support and
    every column against the row support)."""
    rows, cols = len(r.row), len(r.col)
    c_supp = [j for j in range(cols) if r.col[j]]
    r_supp = [i for i in range(rows) if r.row[i]]
    u0 = [sum(r.col[j] * r.known[i, j] for j in c_supp) if all((i, j) in r.known for j in c_supp) else float("-inf")
          for i in range(rows)]
    u1 = [-sum(r.row[i] * r.known[i, j] for i in r_supp) if all((i, j) in r.known for i in r_supp) else float("-inf")
          for j in range(cols)]
    return u0, u1


def softmax_weights(values: list[float], temperature: float) -> list[float]:
    """exp((v - best) / T) for every candidate (unknown values get 0)."""
    finite = [v for v in values if v != float("-inf")]
    best = max(finite) if finite else 0.0
    if temperature <= 0:
        return [1.0 if v == best else 0.0 for v in values]
    return [math.exp((v - best) / temperature) if v != float("-inf") else 0.0 for v in values]


def _table_replies(m: list[list[float]], row: list[float], col: list[float]) -> tuple[list[float], list[float]]:
    """_replies from a full payoff table (p1's view): each seat's own value of every candidate
    against the other seat's mix."""
    return ([sum(c * x for c, x in zip(col, r)) for r in m],
            [-sum(p * r[j] for p, r in zip(row, m)) for j in range(len(m[0]))])


def _certified_full(m: list[list[float]], iterations: int, tolerance: float):
    """The oracle fallback must meet the same certificate as an ordinary oracle answer."""
    its = max(1, iterations)
    while True:
        result = solve_both(m, its)
        if max(exploitability(m, result[0], result[1])) <= tolerance:
            return result
        if its >= 128000:
            raise RuntimeError("full-table fallback did not reach the exploitability tolerance")
        its = min(its * 2, 128000)


def opening_weights(strategy: list[float], temperature: float) -> list[float]:
    """Apply temperature within the solved support; zero-probability moves stay excluded.

    T=1 preserves the strategy, larger T flattens it toward uniform over its support, and
    T=0 keeps only its most likely moves. Scale by the largest entry to avoid overflow.
    Opening picks are exploration, so their distribution is not the equilibrium certificate.
    """
    best = max(strategy)
    if temperature == 0:
        return [float(p == best) for p in strategy]
    return [(p / best) ** (1 / temperature) if p else 0.0 for p in strategy]


def preview_states(obs: Observation, engine: Engine, pairs: list[tuple[int, int]],
                   view: Any = None) -> list[dict[str, Any]]:
    """Next states for (i, j) candidate-index pairs (in `view`'s lists, see turn_candidates),
    previewing only pairs no bot has previewed this turn (shared through engine.turn_cache, keyed
    by positions in the full lists, so both seats pay for a pair once)."""
    idx = _union_index(obs, engine, view)
    if idx is not None:
        pairs = [(idx[0][i], idx[1][j]) for i, j in pairs]
    previewed: dict[tuple[int, int], dict[str, Any]] = engine.turn_cache.setdefault("previewed", {})
    missing = [p for p in dict.fromkeys(pairs) if p not in previewed]
    if missing:
        cands = turn_candidates(obs, engine)
        ids = [(cands[0][i].id, cands[1][j].id) for i, j in missing]
        previewed.update(zip(missing, (r.state for r in engine.preview(ids, candidates=False))))
    return [previewed[p] for p in pairs]


def _payoff_matrix(
    obs: Observation,
    seat: int,
    engine: Engine,
    value: Callable[[dict[str, Any], int], float] = evaluate,
) -> list[list[float]]:
    """payoff[i][j]: value for `seat` when it plays its i-th candidate and the opponent its j-th."""
    n0, n1 = (len(c) for c in turn_candidates(obs, engine))
    states = preview_states(obs, engine, [(i, j) for i in range(n0) for j in range(n1)])
    vals = [value(st, seat) for st in states]  # p1-major: index i * n1 + j
    if seat == 0:
        return [vals[i * n1 : (i + 1) * n1] for i in range(n0)]
    return [[vals[i * n1 + j] for i in range(n0)] for j in range(n1)]


def make_bot(spec: str) -> Bot:
    """A builtin name, or `matrix:MODEL.npz` / `matrix-do:MODEL.npz` for a matrix bot that scores
    previews with a trained value net; it is named after the model so arenas tell them apart.
    A `@N` suffix (`matrix-do:MODEL.npz@2`) asks the engine for candidate set N on this bot's seat,
    for arenas between candidate sets on one engine. A `@cheap=P:N[:POLICY]` suffix (`matrix-do@cheap=0.75:32`,
    also after `@N`) turns on playout cap randomization for this bot's own decisions
    (`MatrixBot.playout_cap`), capping by `playout.cap_candidates`' POLICY (default `list`, today's
    rule; `matrix-do@cheap=0.75:32:options-first`); the bot is named with it. A `@prune=K:R:A[:SCOPE]:POLICY.npz` suffix
    (`matrix-do@prune=16:4:0.05:own:policy.npz`; prototype) prunes with a policy net (`PolicyPrune`:
    top K, R random, audit share A, scope `own` (default) or `both`); everything after the scope is
    the model path, so it may hold colons. A `@value_forward=batched` suffix (matrix bots with a model;
    `--engine batch` only) scores this bot's two views in one value-net forward, as selfplay
    --batched-net-forward does: not bit-identical to the default, so it can be gated on one seat.
    A `@di_fine_moves=K` suffix (matrix bots) asks candidate set 7's `di_fine_all` children only for each seat's K
    most likely support candidates (`MatrixBot.di_fine_moves`; default: every support candidate).
    `policy:POLICY.npz` is the no-search bot (`PolicyBot`), sampling the policy net's softmax;
    `@temp=T` sets its temperature, and `@N` works as above."""
    if spec.startswith("policy:"):
        return _policy_bot(spec[len("policy:"):])
    spec, sep, rest = spec.partition("@")
    cset, cheap, prune, forward, fine_di = "", None, None, None, None
    for opt in rest.split("@") if sep else []:
        if opt.startswith("di_fine_moves="):
            fine_di = int(opt[len("di_fine_moves="):])
            if fine_di < 1:
                raise ValueError(f"@di_fine_moves=K needs K >= 1: {opt!r}")
        elif opt.startswith("prune="):
            parts = opt[len("prune="):].split(":", 3)
            if len(parts) < 4 or not parts[3]:
                raise ValueError(f"@prune=K:R:A[:own|both]:POLICY.npz: {opt!r}")
            scope, path = "own", parts[3]
            head, colon, tail = path.partition(":")
            if colon and head in PolicyPrune.SCOPES:
                scope, path = head, tail
            if not path:
                raise ValueError(f"@prune needs a policy model path: {opt!r}")
            prune = (int(parts[0]), int(parts[1]), float(parts[2]), scope, path)
        elif opt.startswith("cheap="):
            p, _, rest_ = opt[len("cheap="):].partition(":")
            n, _, policy = rest_.partition(":")
            cheap = (float(p), int(n or 32))
            if not 0 <= cheap[0] <= 1 or cheap[1] < 1:
                raise ValueError(f"@cheap=P:N needs 0 <= P <= 1 and N >= 1: {opt!r}")
            if policy and policy not in CAP_POLICIES:
                raise ValueError(f"@cheap=P:N:POLICY takes a policy in {CAP_POLICIES}: {opt!r}")
            if policy and policy != "list":
                cheap += (policy,)
        elif opt.startswith("value_forward="):
            forward = opt[len("value_forward="):]
            if forward != "batched":
                raise ValueError(f"@value_forward takes only 'batched': {opt!r}")
        else:
            cset = opt
    name, _, model = spec.partition(":")
    if not model:
        bot = BUILTINS[name]()
    elif name not in ("matrix", "matrix-do"):
        raise ValueError(f"only matrix bots take a model: {spec!r}")
    else:
        from yomibench.valuenet import ValueNet

        net = ValueNet.load(model)
        bot = MatrixBot(value=net, search="full" if name == "matrix" else "oracle")
        bot.name = f"{name}:{net.meta.get('name', model)}"
    if cset:
        bot.candidate_set = int(cset)  # type: ignore[attr-defined]
        bot.name = f"{bot.name}@{cset}"
    if cheap:
        if not isinstance(bot, MatrixBot):
            raise ValueError(f"only matrix bots take @cheap: {spec!r}")
        bot.playout_cap = cheap
        bot.name = f"{bot.name}@cheap={cheap[0]:g}:{cheap[1]}" + "".join(f":{x}" for x in cheap[2:])
    if forward:
        if not model:
            raise ValueError(f"only matrix bots with a model take @value_forward: {spec!r}")
        bot.value_forward = forward  # type: ignore[union-attr]
        bot.name = f"{bot.name}@value_forward={forward}"
    if fine_di is not None:
        if not isinstance(bot, MatrixBot):
            raise ValueError(f"only matrix bots take @di_fine_moves: {spec!r}")
        bot.di_fine_moves = fine_di
        bot.name = f"{bot.name}@di_fine_moves={fine_di}"
    if prune:
        if not isinstance(bot, MatrixBot) or bot.mode != "oracle":
            raise ValueError(f"only matrix-do bots take @prune: {spec!r}")
        from yomibench.policy import PolicyNet

        k, r, a, scope, path = prune
        bot.policy = PolicyPrune(PolicyNet.load(path), k=k, random_k=r, audit=a, scope=scope)
        bot.name = f"{bot.name}@prune={k}:{r}:{a:g}:{scope}:{bot.policy.config()['net']}"
    return bot


def _policy_bot(spec: str) -> PolicyBot:
    """`policy:` specs: the model path, then optional `@temp=T` and `@N`. Options are read from the
    right, so the path may hold colons and `@`s of its own."""
    path, temp, cset = spec, 1.0, ""
    while True:
        head, sep, opt = path.rpartition("@")
        if not sep:
            break
        if opt.startswith("temp="):
            temp = float(opt[len("temp="):])
        elif opt.isdigit():
            cset = opt
        else:
            break
        path = head
    if not path:
        raise ValueError(f"policy bots need a model: policy:POLICY.npz[@temp=T][@N], not {spec!r}")
    from yomibench.policy import PolicyNet

    bot = PolicyBot(PolicyNet.load(path), temperature=temp)
    if temp != 1.0:
        bot.name = f"{bot.name}@temp={temp:g}"
    if cset:
        bot.candidate_set = int(cset)  # type: ignore[attr-defined]
        bot.name = f"{bot.name}@{cset}"
    return bot


BUILTINS: dict[str, Callable[[], Bot]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "matrix": MatrixBot,
    "matrix-do": lambda: MatrixBot(search="oracle"),
}

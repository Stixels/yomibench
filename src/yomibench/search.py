"""Evaluation and simultaneous-move solving."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from yomibench import phases


def _load_native():
    if os.environ.get("YOMIBENCH_SOLVER", "").lower() == "python":
        return None
    try:
        from yomisolve import alternating
    except ImportError:
        return None
    return alternating


_native = _load_native()
SOLVER = "native" if _native is not None else "python"  # which regret-matching loop runs (configs)


def terminal_value(state: dict[str, Any], seat: int) -> float | None:
    """The exact result for `seat` if the game is over in `state` (KO, or the time limit, where
    the higher hp wins as in game.gd end_game), else None. Evaluators must use it first."""
    me, opp = state["players"][seat], state["players"][1 - seat]
    over_on_time = state.get("time_left", 1) < 0
    if me["hp"] > 0 and opp["hp"] > 0 and not over_on_time:
        return None
    return 0.0 if me["hp"] == opp["hp"] else (1.0 if me["hp"] > opp["hp"] else -1.0)


def evaluate(state: dict[str, Any], seat: int) -> float:
    """Score in [-1, 1] for `seat`: health fraction lead, decisive on a KO or at the time limit."""
    done = terminal_value(state, seat)
    if done is not None:
        return done
    me, opp = state["players"][seat], state["players"][1 - seat]
    return me["hp"] / me["max_hp"] - opp["hp"] / opp["max_hp"]


def solve_zero_sum(payoff: list[list[float]], iterations: int) -> tuple[list[float], float]:
    """Approximate Nash strategy for the row player of a zero-sum matrix game.

    Regret matching+ self-play for both players; returns the row player's average
    strategy and its guaranteed value against the column player's best response.
    """
    row, _, value = solve_both(payoff, iterations)
    return row, value


def solve_both(payoff: list[list[float]], iterations: int) -> tuple[list[float], list[float], float]:
    """Both players' average strategies from regret matching+, and the row player's guaranteed
    value (the column player's is at most its negative).

    Updates alternate (the second player responds to the first player's new strategy), with linear
    averaging: on recorded 30x30 tables that is 10-40x closer to equilibrium than simultaneous
    updates for the same work. Alternation favors the side that moves second, so the solve runs
    twice, once with each side first, for half the iterations each, and averages the two: swapping
    the seats (solving -payoff transposed) returns exactly the swapped answer, bit for bit.
    Exploitability is convex, so the average is no more exploitable than the worse of the two runs
    (double_oracle checks the actual exploitability against its 1e-3 tolerance)."""
    t = time.perf_counter()
    rows, cols = len(payoff), len(payoff[0])
    half = (iterations + 1) // 2
    r1, c1 = _alternating(payoff, half)
    c2, r2 = _alternating([[-payoff[i][j] for i in range(rows)] for j in range(cols)], half)
    strategy = [(a + b) / 2 for a, b in zip(r1, r2)]
    col = [(a + b) / 2 for a, b in zip(c1, c2)]
    value = min(sum(strategy[i] * payoff[i][j] for i in range(rows)) for j in range(cols))
    phases.add("solve", time.perf_counter() - t)  # timing only (the ledger's solve_ms)
    return strategy, col, value


def _alternating(payoff: list[list[float]], iterations: int) -> tuple[list[float], list[float]]:
    """One alternating regret matching+ run, row player first; linear-averaged strategies. Uses the
    native `yomisolve` extension when it is installed (native/yomisolve: the same arithmetic in
    Rust, bit-identical to `_alternating_py` on Python 3.12+, 30-130x faster on the small tables
    the double oracle solves); set YOMIBENCH_SOLVER=python to force the pure-Python loop."""
    if _native is not None:
        return _native(payoff, iterations)
    return _alternating_py(payoff, iterations)


def _alternating_py(payoff: list[list[float]], iterations: int) -> tuple[list[float], list[float]]:
    """The pure-Python regret matching+ loop (the reference the native version reproduces)."""
    rows, cols = len(payoff), len(payoff[0])
    regret_r, regret_c = [0.0] * rows, [0.0] * cols
    avg_r, avg_c = [0.0] * rows, [0.0] * cols
    sc = _match(regret_c)
    for t in range(1, iterations + 1):
        row_vals = [sum(payoff[i][j] * sc[j] for j in range(cols)) for i in range(rows)]
        sr = _match(regret_r)
        ev = sum(sr[i] * row_vals[i] for i in range(rows))
        regret_r = [max(0.0, r + v - ev) for r, v in zip(regret_r, row_vals)]
        sr = _match(regret_r)
        col_vals = [-sum(payoff[i][j] * sr[i] for i in range(rows)) for j in range(cols)]
        ev = sum(sc[j] * col_vals[j] for j in range(cols))
        regret_c = [max(0.0, r + v - ev) for r, v in zip(regret_c, col_vals)]
        sc = _match(regret_c)
        for i in range(rows):
            avg_r[i] += t * sr[i]  # linear averaging
        for j in range(cols):
            avg_c[j] += t * sc[j]
    return [a / sum(avg_r) for a in avg_r], [a / sum(avg_c) for a in avg_c]


def prune(strategy: list[float], threshold: float = 0.02) -> list[float]:
    """Zero out probabilities below `threshold` and renormalize.

    Regret matching leaves small residual weight on poor moves; without this a bot plays a few
    junk moves per game. The largest entry always survives."""
    top = max(strategy)
    kept = [p if p >= threshold or p == top else 0.0 for p in strategy]
    total = sum(kept)
    return [p / total for p in kept]


def _match(regret: list[float]) -> list[float]:
    total = sum(regret)
    if total <= 0:
        return [1.0 / len(regret)] * len(regret)
    return [r / total for r in regret]


def exploitability(payoff: list[list[float]], row: list[float], col: list[float]) -> tuple[float, float]:
    """How much each side could gain by switching to its best pure reply in the full table:
    (row's best reply to `col` - value of the pair, value of the pair - column's best reply to `row`)."""
    rows, cols = len(payoff), len(payoff[0])
    v = sum(row[i] * col[j] * payoff[i][j] for i in range(rows) for j in range(cols) if row[i] and col[j])
    best_row = max(sum(col[j] * payoff[i][j] for j in range(cols) if col[j]) for i in range(rows))
    best_col = min(sum(row[i] * payoff[i][j] for i in range(rows) if row[i]) for j in range(cols))
    return best_row - v, v - best_col


@dataclass
class OracleResult:
    row: list[float]       # strategies over all moves
    col: list[float]
    value: float           # row's guaranteed value against any column reply (a lower bound)
    upper: float           # the most the row player could get against `col` (an upper bound)
    known: dict[tuple[int, int], float]  # every payoff evaluated
    rounds: int
    converged: bool        # both sides' best-reply gains <= tolerance against the returned strategies


def _purify(sub: list[float], moves: list[int], n: int, floor: float) -> list[float]:
    """Spread a restricted strategy over all n moves, dropping entries below `floor`."""
    full = [0.0] * n
    for i, p in zip(moves, sub):
        if p >= floor:
            full[i] = p
    total = sum(full)
    if total <= 0:  # every entry was tiny: keep the largest
        full[moves[max(range(len(sub)), key=sub.__getitem__)]] = total = 1.0
    return [p / total for p in full]


def double_oracle(
    n_rows: int,
    n_cols: int,
    evaluate_pairs,
    iterations: int = 2000,
    tolerance: float = 1e-3,
    start: tuple[list[int], list[int]] = ([0], [0]),
    add_per_round: int = 2,
    support_floor: float = 1e-3,
    max_rounds: int = 60,
    max_iterations: int = 32000,
    on_round: Callable[[OracleResult], None] | None = None,
) -> OracleResult:
    """Equilibrium of an n_rows x n_cols zero-sum game without evaluating the whole table.

    `evaluate_pairs(pairs)` returns the row player's payoff for each (i, j); every pair is asked
    for at most once. Each round solves the game restricted to the moves found so far, drops
    support entries below `support_floor`, and then checks the returned strategies against EVERY
    move of both sides (this needs only the payoffs of all moves against the other side's
    support). It stops when neither side can gain more than `tolerance` by any move, so the
    returned strategies are certified against the full table, not only the restricted one. Moves
    that gain more are added; when no new move is to blame, the restricted solve was too coarse,
    so the floor shrinks and iterations grow (capped at `max_iterations`). `converged` is False if
    `max_rounds` ran out; callers must not treat that result as certified.

    `value` is the row player's guaranteed value against every column reply (the minimum over
    columns), the same quantity `solve_both` returns for a full table.

    Seat-symmetric: solving the swapped game (-payoff transposed, with the starts swapped) returns
    exactly the swapped result, so neither seat is favored by the solver itself.

    `on_round` (optional) sees each round's result as soon as it is solved, converged or not: a
    caller that may be stopped mid-search keeps the latest one as its best-so-far answer (human
    play's turn timer, yomibench.anytime). It only observes; the search is the same with or without it."""
    known: dict[tuple[int, int], float] = {}

    def need(pairs):
        missing = [p for p in dict.fromkeys(pairs) if p not in known]
        if missing:
            for p, v in zip(missing, evaluate_pairs(missing)):
                known[p] = v

    rs, cs = list(dict.fromkeys(start[0])), list(dict.fromkeys(start[1]))
    its, floor = iterations, support_floor
    refinements = 0
    result = None
    for rounds in range(1, max_rounds + 1):
        need([(i, j) for i in rs for j in cs])
        r_sub, c_sub, _ = solve_both([[known[i, j] for j in cs] for i in rs], its)
        row, col = _purify(r_sub, rs, n_rows, floor), _purify(c_sub, cs, n_cols, floor)
        r_supp = [i for i in range(n_rows) if row[i]]
        c_supp = [j for j in range(n_cols) if col[j]]
        need([(i, j) for i in range(n_rows) for j in c_supp] + [(i, j) for i in r_supp for j in range(n_cols)])
        u_row = [sum(col[j] * known[i, j] for j in c_supp) for i in range(n_rows)]  # each row vs col
        u_col = [sum(row[i] * known[i, j] for i in r_supp) for j in range(n_cols)]  # row vs each column
        # The pair's value two ways, averaged, so swapping the seats gives exactly -v.
        v = (sum(row[i] * u_row[i] for i in r_supp) + sum(col[j] * u_col[j] for j in c_supp)) / 2
        lower, upper = min(u_col), max(u_row)
        result = OracleResult(row, col, lower, upper, known, rounds, False)
        result.converged = upper - v <= tolerance and v - lower <= tolerance
        if on_round is not None:
            on_round(result)
        if result.converged:
            return result
        add_r = [i for g, i in sorted(((u_row[i] - v, i) for i in range(n_rows) if i not in rs), reverse=True)
                 [:add_per_round] if g > tolerance]
        add_c = [j for g, j in sorted(((v - u_col[j], j) for j in range(n_cols) if j not in cs), reverse=True)
                 [:add_per_round] if g > tolerance]
        if add_r or add_c:
            rs += add_r
            cs += add_c
        else:  # the gain comes from moves already in the restricted game: solve it finer.
            # Dropping small entries is the cheap suspect, so the floor shrinks every time;
            # iterations double only every other time, up to max_iterations.
            floor /= 4
            refinements += 1
            if refinements % 2 == 0:
                its = min(its * 2, max_iterations)
    assert result is not None
    return result

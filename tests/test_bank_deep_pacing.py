"""Chess-clock pacing of the deeper stages (yomibench.anytime.bank_deep_after, Clock.deep_after): a contested bank turn's
allowance for widen / two-decision is counted from when the production search finishes, so a long stage 1 never eats
it; a routine (one-sided, DI-only, no Burst) turn answers as soon as stage 1 is done; the bank budget's hard stop,
the after-lock cap and the think cap still come first, and untimed turns are unchanged."""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from yomibench import anytime
from yomibench.anytime import BANK_RESERVE_S, Clock, bank_deep_after, bank_deep_s, contested
from yomibench.bots import MatrixBot

FLOOR = anytime.answer_within(15.0)  # 12.75 s


def _toy_position(moves=((2, 3), (4, 1), (0, 6))):
    from yomibench.engine import connect

    e = connect("toy")
    obs = e.new_match("A", "B", 1)
    for a, b in moves:
        obs = e.step(a, b)
    return e, obs


def _one_sided(obs, burst_seat: int | None = None):
    """The toy position as a one-sided turn (p1 only answers p2's turn), optionally with a Burst on a seat's menu."""
    cands = [list(obs.candidates[0]), list(obs.candidates[1])]
    if burst_seat is not None:
        cands[burst_seat][-1] = dataclasses.replace(cands[burst_seat][-1], action="Burst")
    return dataclasses.replace(obs, actionable=(False, True), candidates=(tuple(cands[0]), tuple(cands[1])))


class _StageOneSlowBot(MatrixBot):
    """The production bot, its evaluator slowed only until the clock's deeper-stage allowance starts: a long stage 1
    (5-9 s on a real neutral turn, scaled down) followed by quick deeper stages."""

    def __init__(self, clock: Clock, per_state: float = 0.0, after: float = 0.0):
        super().__init__(search="oracle")
        base = self.value

        def value(state, seat):
            time.sleep(per_state if clock.deep_from is None else after)
            return base(state, seat)

        self.value = value


# --- the allowance itself ------------------------------------------------------------------------------------------

def test_a_contested_turns_allowance_is_a_capped_share_of_the_room():
    assert anytime.BANK_DEEP_TURNS == 20.0 and anytime.BANK_DEEP_CAP_S == 25.0
    assert bank_deep_s(900.0) == pytest.approx(25.0)  # 840 s of room: capped at 25 s
    assert bank_deep_s(560.0) == pytest.approx(25.0)  # the cap holds down to 500 s of room
    assert bank_deep_s(300.0) == pytest.approx(12.0)  # 240 s of room: 12 s
    assert bank_deep_s(80.0) == pytest.approx(1.0)
    assert bank_deep_s(BANK_RESERVE_S) == 0.0 and bank_deep_s(40.0) == 0.0  # at or under the reserve: nothing
    assert bank_deep_s(900.0, turns=10.0, cap_s=60.0) == pytest.approx(60.0)
    assert bank_deep_s(900.0, cap_s=0.0) == 0.0


def test_the_allowance_tapers_smoothly_and_never_reaches_the_reserve_by_itself():
    banks = [BANK_RESERVE_S + 0.5 * k for k in range(0, 2000)]
    deeps = [bank_deep_s(b) for b in banks]
    assert all(b >= a for a, b in zip(deeps, deeps[1:]))  # monotone in the bank
    assert all(b - a <= 0.5 / anytime.BANK_DEEP_TURNS + 1e-9 for a, b in zip(deeps, deeps[1:]))  # no jumps
    assert all(bank - d >= BANK_RESERVE_S for bank, d in zip(banks, deeps))
    # Near the floor (just enough room for a bank turn) the allowance is under a second.
    assert bank_deep_s(BANK_RESERVE_S + FLOOR + 0.01) < 1.0
    # A bot that spends its whole allowance on every turn shrinks the room geometrically: never to the reserve.
    bank = 900.0
    for _ in range(300):
        bank -= bank_deep_s(bank)
    assert bank > BANK_RESERVE_S


def test_contested_turns_are_both_actionable_or_have_a_burst_in_play():
    _, obs = _toy_position()
    assert obs.actionable == (True, True) and contested(obs)
    assert bank_deep_after(obs, 300.0) == ("contested", pytest.approx(12.0))
    routine = _one_sided(obs)
    assert not contested(routine)
    assert bank_deep_after(routine, 900.0) == ("routine", 0.0)  # a routine turn gets nothing, however full the bank
    for seat in (0, 1):  # a Burst on either seat's menu makes it a real guess again
        assert contested(_one_sided(obs, burst_seat=seat))
    assert contested(dataclasses.replace(
        routine, candidates=(routine.candidates[0], (dataclasses.replace(routine.candidates[1][0],
                                                                         action="DefensiveBurst"),))))


# --- the Clock: counted from stage 1's end -------------------------------------------------------------------------

def test_the_allowance_starts_when_stage_one_ends_not_at_the_decisions_open():
    """Stage 1 took 9 s of a healthy 300 s bank: the old rule (1/100 of the room from the open, 2.4 s) had nothing left
    for the deeper stages; now they get 12 s from stage 1's end."""
    clock = Clock(0)
    clock.t0 -= 9.0  # the decision opened 9 s ago: stage 1 just finished
    clock.cap_at = clock.t0 + min(240.0, anytime.THINK_CAP_S)  # what a bank turn sets at its open
    clock.deep_after = bank_deep_s(300.0)
    assert clock.t0 + 240.0 / 100 < time.monotonic()  # the old allowance would be spent already
    clock.start_deep()
    assert clock.cap_at - clock.deep_from == pytest.approx(12.0)
    assert clock.deep_from - clock.t0 == pytest.approx(9.0, abs=0.05)
    assert clock.soft() is None  # the deeper stages may run
    at = clock.cap_at
    clock.start_deep()
    assert clock.cap_at == at  # once per decision


def test_the_allowance_never_moves_past_the_think_cap_or_the_budget():
    clock = Clock(0, hard_at=time.monotonic() + 0.5)
    clock.cap_at = clock.t0 + 0.4
    clock.deep_after = 100.0
    clock.start_deep()
    assert clock.cap_at == pytest.approx(clock.t0 + 0.4)  # never later than the cap already set
    assert clock.cap_at <= clock.hard_at


def test_an_untimed_turns_cap_is_unchanged():
    clock = Clock(0)
    clock.cap_at = clock.t0 + anytime.THINK_CAP_S
    clock.start_deep()  # no deep_after: not a bank turn
    assert clock.deep_from is None and clock.cap_at == pytest.approx(clock.t0 + anytime.THINK_CAP_S)


# --- think(): the deeper stages on a bank turn ---------------------------------------------------------------------

def _bank_clock(room: float, deep_after: float, think_cap: float = anytime.THINK_CAP_S) -> Clock:
    """A bank turn's clock as human_play sets it: the room as the hard stop, the after-lock cap, the think cap."""
    clock = Clock(0)
    clock.hard_at = clock.t0 + room
    clock.emergency_at = clock.t0 + anytime.emergency_within(room, room)
    clock.lock_cap = anytime.LOCK_CAP_S
    clock.cap_at = clock.t0 + min(room, think_cap)
    clock.deep_after = deep_after
    return clock


def test_a_contested_turn_runs_two_decision_after_a_long_stage_one():
    """A healthy bank (300 s: 12 s allowance) and a stage 1 longer than the old share from the open would be: the
    deeper stages still run, through two-decision."""
    e, obs = _toy_position()
    kind, deep = bank_deep_after(obs, 300.0)
    assert kind == "contested"
    clock = _bank_clock(240.0, deep)
    old_cap = clock.t0 + 0.15  # the old rule's cap, scaled down: stage 1 alone passes it
    out = anytime.think(_StageOneSlowBot(clock, per_state=0.01), obs, 1, e, clock, deep=True)
    stage1 = next(t for t in out.trace if t.get("name") == "production")
    assert out.production is not None and stage1["ms"] / 1000 > old_cap - clock.t0
    assert clock.deep_from is not None and clock.deep_from >= old_cap
    names = [t.get("name") for t in out.trace if t.get("kept")]
    assert "widen" in names and "two-decision" in names
    assert out.best.name == "two-decision" and out.path == "full"
    assert out.stopped == "done"  # every stage finished inside the allowance
    assert time.monotonic() <= clock.cap_at


def test_a_routine_turn_answers_as_soon_as_stage_one_is_done():
    """One-sided / DI-only, no Burst: no deeper stage runs, however much bank is left; the answer is the production
    search's, the moment it finishes."""
    e, obs = _toy_position()
    routine = _one_sided(obs)
    kind, deep = bank_deep_after(routine, 900.0)
    assert (kind, deep) == ("routine", 0.0)
    clock = _bank_clock(840.0, deep)
    out = anytime.think(MatrixBot(search="oracle"), routine, 1, e, clock, deep=True)
    assert out.production is not None and out.best is out.production and out.path == "full"
    assert out.stopped == "think-cap"
    assert [t["stage"] for t in out.trace] == [1]  # no deeper stage even started
    assert time.monotonic() - clock.deep_from < 0.5


def test_a_routine_turn_with_a_burst_in_play_still_thinks():
    e, obs = _toy_position()
    burst = _one_sided(obs, burst_seat=0)
    kind, deep = bank_deep_after(burst, 300.0)
    assert kind == "contested" and deep == pytest.approx(12.0)
    clock = _bank_clock(240.0, deep)
    out = anytime.think(MatrixBot(search="oracle"), burst, 1, e, clock, deep=True)
    assert any(t["stage"] >= 2 for t in out.trace)


def test_a_low_bank_tapers_the_allowance_and_stops_on_time():
    """Just above the floor (75 s: 15 s of room, 0.75 s allowance, scaled down here): slow deeper stages are cut at
    the allowance and the production answer stands, well inside the budget."""
    e, obs = _toy_position()
    deep = bank_deep_s(75.0)
    assert deep == pytest.approx(0.75)
    clock = _bank_clock(15.0, deep / 5)  # scaled: 0.15 s
    out = anytime.think(_StageOneSlowBot(clock, after=0.05), obs, 1, e, clock, deep=True)
    assert out.production is not None and out.path == "full"
    assert out.stopped == "think-cap"
    assert time.monotonic() - clock.deep_from < deep / 5 + 0.25  # one guarded step past the allowance at most
    assert time.monotonic() < clock.hard_at


def test_the_budgets_hard_stop_still_cuts_a_large_allowance():
    e, obs = _toy_position()
    clock = _bank_clock(0.6, 100.0)  # a tiny budget, a huge allowance: the budget wins
    t0 = time.monotonic()
    out = anytime.think(_StageOneSlowBot(clock, after=0.05), obs, 1, e, clock, deep=True)
    assert out.production is not None and out.best is not None
    assert out.stopped in ("think-cap", "deadline")  # cap_at was clamped to the budget's end
    assert time.monotonic() - t0 < 0.6 + 0.25


def test_the_after_lock_cap_still_cuts_the_deeper_stages():
    e, obs = _toy_position()
    clock = _bank_clock(240.0, 100.0)
    clock.lock_cap = 0.3
    clock.lock_converged = None
    threading.Timer(0.05, clock.signal, args=("locked_in",)).start()
    out = anytime.think(_StageOneSlowBot(clock, after=0.05), obs, 1, e, clock, deep=True)
    assert out.production is not None and out.best is not None
    assert out.stopped in ("lock-in", "lock-cap")  # the lock-in ends the deeper stages; the cap bounds the rest
    assert time.monotonic() <= clock.soft_at + 0.3 + 0.25


def test_an_untimed_turn_is_unchanged():
    """No deep_after: the think cap from the open, as before (and a lock-in still ends the deeper stages)."""
    e, obs = _toy_position()
    clock = Clock(0)
    clock.cap_at = time.monotonic()  # the cap has passed already
    out = anytime.think(MatrixBot(search="oracle"), obs, 1, e, clock, deep=True)
    assert out.stopped == "think-cap" and out.best is out.production and clock.deep_from is None

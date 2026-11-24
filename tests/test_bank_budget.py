"""Ranked (a chess clock): the bot spends its bank about as the human spends theirs, never plans past the reserve, and
plays on the 15 s floor once it has no more room (yomibench.anytime.bank_budget, Clock)."""
import time

import pytest

from yomibench import anytime
from yomibench.anytime import BANK_RESERVE_S, Clock, bank_budget

FLOOR = anytime.answer_within(15.0)  # 12.75 s: the per-turn stop every timed turn has


def test_a_bank_with_room_plans_up_to_the_reserve():
    assert BANK_RESERVE_S == 60.0
    assert bank_budget(900.0, 15.0) == ("bank", 840.0)
    assert bank_budget(300.0, 15.0) == ("bank", 240.0)


@pytest.mark.parametrize("bank", [900.0, 400.0, 120.0, 73.0, 60.0 + FLOOR + 0.01])
def test_a_bank_plan_never_reaches_below_the_reserve(bank):
    mode, budget = bank_budget(bank, 15.0)
    assert mode == "bank" and bank - budget == pytest.approx(BANK_RESERVE_S)


@pytest.mark.parametrize("bank", [60.0 + FLOOR, 70.0, 60.0, 40.0, 15.0])
def test_at_or_near_the_reserve_the_bot_plays_on_the_floor(bank):
    assert bank_budget(bank, 15.0) == ("floor", pytest.approx(FLOOR))


def test_the_floor_is_todays_per_turn_stop_and_respects_a_custom_reserve():
    assert bank_budget(15.0, 15.0) == ("floor", pytest.approx(12.75))
    assert bank_budget(100.0, 15.0, reserve_s=120.0)[0] == "floor"
    assert bank_budget(100.0, 15.0, reserve_s=0.0) == ("bank", 100.0)
    # a bank under the floor (the game tops it up first; if it didn't, the stop still fits what is left)
    assert bank_budget(8.0, 15.0) == ("floor", pytest.approx(anytime.answer_within(8.0)))


def test_a_bank_turn_keeps_the_after_lock_cap_beside_its_deadline():
    """Bank mode: the budget's end is the hard stop; once the human locks in, the after-lock cap applies too (v5's
    untimed rule), whichever comes first. A timed per-turn decision (lock_cap unset) is unchanged."""
    c = Clock(1, hard_at=time.monotonic() + 600)
    c.lock_cap = 0.05
    assert c.hard() is None and c.lock_hard_at() is None
    c.signal("locked_in")
    assert c.lock_hard_at() == pytest.approx(c.soft_at + 0.05)
    assert c.emergency() < c.lock_hard_at()  # the lock-cap's emergency point, not the budget's (600 s away)
    time.sleep(0.08)
    assert c.hard() == "lock-cap"
    timed = Clock(1, hard_at=time.monotonic() + 600, emergency_at=time.monotonic() + 500)
    timed.signal("locked_in")
    assert timed.lock_hard_at() is None and timed.hard() is None
    assert timed.emergency() == timed.emergency_at


def test_a_bank_turns_refinement_share_is_a_fortieth_of_its_room():
    """Refinement (after the first converged round) stops at (bank - reserve) / BANK_REFINE_TURNS; the base double
    oracle keeps the whole budget (test_anytime_paths covers the stop)."""
    assert anytime.BANK_REFINE_TURNS == 40.0
    assert anytime.bank_refine_s(900.0) == pytest.approx(21.0)
    assert anytime.bank_refine_s(300.0) == pytest.approx(6.0)
    assert anytime.bank_refine_s(50.0) == 0.0  # under the reserve: nothing (such a turn plays on the floor anyway)
    assert anytime.bank_refine_s(900.0, turns=20.0) == pytest.approx(42.0)


def test_the_refinement_share_shrinks_the_bank_geometrically():
    """A bot that spends its whole share every turn: (b - r) shrinks by (1 - 1/40) a turn, so 150 turns leave
    840 * 0.975^150 ~ 19 s of room, while a fixed 21 s a turn reaches the reserve by turn 40."""
    bank = 900.0
    for _ in range(150):
        bank -= anytime.bank_refine_s(bank)
    assert bank - BANK_RESERVE_S == pytest.approx(840.0 * 0.975 ** 150) and bank > BANK_RESERVE_S

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

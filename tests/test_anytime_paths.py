"""Which search produces a timed human-play answer (yomibench.anytime.think, on the toy game): with no clock the
full search always finishes; with a clock the full search's best-so-far mix is preferred whenever it has solved a
round; the quick search runs only as the emergency, when the full search solved nothing by the emergency point."""

from __future__ import annotations

import time

import pytest

from yomibench import anytime
from yomibench.anytime import Answer, Clock, emergency_within, turn_seconds
from yomibench.bots import MatrixBot


def _toy_position(moves=((2, 3), (4, 1), (0, 6))):
    from yomibench.engine import connect

    e = connect("toy")
    obs = e.new_match("A", "B", 1)
    for a, b in moves:
        obs = e.step(a, b)
    return e, obs


def _slow_bot(per_state: float) -> MatrixBot:
    bot = MatrixBot(search="oracle")
    base = bot.value

    def value(state, seat):
        time.sleep(per_state)
        return base(state, seat)

    bot.value = value
    return bot


@pytest.fixture()
def quick_calls(monkeypatch):
    """The emergency quick search, replaced by an instant answer that records it ran."""
    calls = []

    def quick(self):
        calls.append(time.monotonic())
        n = len(self.obs.candidates[1])
        return Answer(0, "quick", [list(self.obs.candidates[0]), list(self.obs.candidates[1])],
                      [1.0] + [0.0] * (len(self.obs.candidates[0]) - 1), [1.0] + [0.0] * (n - 1), 0.0)

    monkeypatch.setattr(anytime.Deepener, "quick", quick)
    return calls


def test_no_timer_means_no_deadline():
    assert turn_seconds({"timer_mode": "none", "turn_time": 30}) is None  # the box unticked / standard
    assert turn_seconds({"turn_time": 15, "turn_min_length": 15}) is None  # no timer_mode at all
    assert turn_seconds({"timer_mode": "chess", "turn_time": 15, "turn_min_length": 15}) == 15  # ranked, ticked
    assert turn_seconds({"timer_mode": "default", "turn_time": 30}) == 30


def test_no_clock_the_slow_full_search_finishes_and_no_quick_answer_runs(quick_calls):
    e, obs = _toy_position()
    clock = Clock(0)  # no hard_at, no emergency_at: what an untimed decision gets
    t0 = time.monotonic()
    out = anytime.think(_slow_bot(0.002), obs, 1, e, clock, deep=False)
    assert out.path == "full" and out.stopped == "done" and out.best is out.production
    assert not quick_calls and time.monotonic() - t0 > 0.05  # it took as long as it took
    e2, obs2 = _toy_position()
    row, col, value = MatrixBot(search="oracle").search_turn(obs2, e2, 1)
    assert (out.best.row, out.best.col, out.best.value) == (row, col, value)  # exactly the plain search


class _RoundsBot(MatrixBot):
    """A matrix bot whose search reports a solved round (its `progress` hook, as the double oracle's `on_round`
    does) every `every` seconds and never converges: it ends only when the clock stops it."""

    def __init__(self, every: float):
        super().__init__(search="oracle")
        self.every = every

    def search_turn(self, obs, engine, seat=None):
        lists = [list(obs.candidates[0]), list(obs.candidates[1])]
        k = 0
        while True:
            t = time.monotonic()
            while time.monotonic() - t < self.every:
                self.value(obs.state, 0)  # the guarded evaluator: the stop check
                time.sleep(0.002)
            k += 1
            row = [0.0] * len(lists[0])
            col = [0.0] * len(lists[1])
            row[k % len(row)], col[k % len(col)] = 1.0, 1.0
            if self.progress is not None:
                self.progress(row, col, 0.1 * k, lists, False)


def test_clock_on_a_cut_full_search_plays_its_best_so_far_not_the_quick_answer(quick_calls):
    e, obs = _toy_position()
    now = time.monotonic()
    clock = Clock(0, hard_at=now + 0.35, emergency_at=now + 0.2)
    out = anytime.think(_RoundsBot(every=0.05), obs, 1, e, clock, deep=False)
    assert out.production is None and out.stopped == "deadline"  # the deadline cut the full search ...
    assert out.path == "partial-full" and out.best.name == "partial-full"  # ... and its best so far is played
    assert not quick_calls  # past the emergency point too: a solved round means no quick answer
    k = out.best.info["rounds_seen"]
    assert k >= 4 and out.best.value == pytest.approx(0.1 * k)  # the LATEST round's mix
    for s, strat in ((0, out.best.row), (1, out.best.col)):  # a real mix over the search's own lists
        assert len(strat) == len(out.best.lists[s]) and sum(strat) == pytest.approx(1.0) and min(strat) >= 0
    assert out.best.label() == "full search, best so far"


def test_the_double_oracle_reports_every_round_and_the_search_is_unchanged():
    from yomibench.search import double_oracle

    table = [[(3 * i + 5 * j) % 7 - 3 for j in range(6)] for i in range(6)]
    seen = []
    plain = double_oracle(6, 6, lambda ps: [table[i][j] for i, j in ps])
    hooked = double_oracle(6, 6, lambda ps: [table[i][j] for i, j in ps], on_round=seen.append)
    assert (hooked.row, hooked.col, hooked.value, hooked.rounds) == (plain.row, plain.col, plain.value, plain.rounds)
    assert len(seen) == plain.rounds and seen[-1] is hooked


def test_clock_on_quick_only_when_the_full_search_solved_nothing_by_the_emergency_point(quick_calls):
    e, obs = _toy_position()
    now = time.monotonic()
    clock = Clock(0, hard_at=now + 0.4, emergency_at=now + 0.05)
    out = anytime.think(_slow_bot(0.2), obs, 1, e, clock, deep=False)  # its first round takes seconds
    assert out.path == "quick" and out.best.name == "quick" and len(quick_calls) == 1
    assert out.stopped == "emergency" and quick_calls[0] - now < 0.3  # the emergency point, well before the deadline
    assert [t.get("stopped") for t in out.trace][0] == "emergency"


def test_cut_before_the_emergency_point_with_nothing_solved_holds(quick_calls):
    e, obs = _toy_position()
    clock = Clock(0, hard_at=time.monotonic() + 0.03)  # no emergency point: e.g. superseded or a hard deadline
    out = anytime.think(_slow_bot(0.2), obs, 1, e, clock, deep=False)
    assert out.best is None and out.path == "continue" and not quick_calls


def test_the_emergency_point_leaves_the_full_search_most_of_the_turn():
    within = anytime.answer_within(15)  # ranked: answer by 12.75 s
    assert within == pytest.approx(12.75)
    assert emergency_within(15, within) == pytest.approx(10.5)  # quick only if nothing is solved by 10.5 s
    assert emergency_within(2.5, anytime.answer_within(2.5, 1.0, 0.0)) == pytest.approx(0.75)  # never before half
    assert emergency_within(60, anytime.answer_within(60)) == pytest.approx(42.0)


def test_thinking_until_lock_in_stops_at_its_cap_but_never_cuts_the_full_search():
    e, obs = _toy_position()
    clock = Clock(0)
    clock.cap_at = time.monotonic()  # the cap has passed already: no extra stage may run
    out = anytime.think(_slow_bot(0.001), obs, 1, e, clock, deep=True)
    assert out.production is not None and out.best is out.production and out.path == "full"
    assert out.stopped == "think-cap"


# --- the after-lock cap (no turn timer, private build v5) ------------------------------------------------------

def _locked_clock(cap: float, locked_ago: float = 0.0) -> Clock:
    clock = Clock(0)
    clock.lock_cap = cap
    clock.signal("locked_in")
    clock.soft_at -= locked_ago
    return clock


def test_no_lock_in_means_no_cap():
    clock = Clock(0)
    clock.lock_cap = 0.01
    assert clock.lock_hard_at() is None and clock.emergency() is None and clock.hard() is None


def test_the_cap_counts_from_the_lock_in_even_if_it_arrived_before_the_cap_was_set():
    clock = Clock(0)
    clock.signal("locked_in")  # the reader thread signalled before the decision set its cap
    clock.lock_cap = 10.0
    assert clock.lock_hard_at() == pytest.approx(clock.soft_at + 10.0)
    assert clock.emergency() == pytest.approx(clock.soft_at + 8.5)  # LOCK_CAP_S less the 1.5 s emergency margin
    assert anytime.LOCK_CAP_S == 10.0


def test_a_turn_deadline_wins_over_the_cap():
    clock = Clock(0, hard_at=time.monotonic() + 100)
    clock.lock_cap = 0.01
    clock.signal("locked_in")
    time.sleep(0.02)
    assert clock.lock_hard_at() is None and clock.hard() is None  # timed: only the deadline


def test_the_cap_cuts_the_full_search_and_plays_its_best_so_far(quick_calls):
    e, obs = _toy_position()
    clock = Clock(0)
    clock.lock_cap = 0.3
    t0 = time.monotonic()
    import threading

    threading.Timer(0.1, clock.signal, args=("locked_in",)).start()
    out = anytime.think(_RoundsBot(every=0.03), obs, 1, e, clock, deep=False)
    took = time.monotonic() - t0
    assert out.production is None and out.stopped == "lock-cap" and out.path == "partial-full"
    assert not quick_calls and 0.38 <= took < 0.7  # the lock-in at 0.1 s, then the 0.3 s cap


def test_the_cap_plays_the_quick_answer_only_if_nothing_was_solved(quick_calls):
    e, obs = _toy_position()
    clock = _locked_clock(0.4)
    out = anytime.think(_slow_bot(0.2), obs, 1, e, clock, deep=False)  # its first round takes seconds
    assert out.path == "quick" and out.stopped == "emergency" and len(quick_calls) == 1


def test_a_search_that_ends_before_the_cap_answers_at_once():
    e, obs = _toy_position()
    clock = _locked_clock(30.0)
    t0 = time.monotonic()
    out = anytime.think(MatrixBot(search="oracle"), obs, 1, e, clock, deep=True)
    assert out.path == "full" and out.production is not None and time.monotonic() - t0 < 5
    assert out.stopped == "lock-in"  # the extra stages stop at the lock-in; the cap is never waited out


def test_the_cap_stops_before_a_step_that_would_end_past_it():
    """Production's value batches are never split (its answers stay the plain search's), so the cap stops before a
    step as long as the longest one so far would cross it: the answer comes by the cap, not a batch later."""
    clock = _locked_clock(10.0, locked_ago=7.0)  # 3 s left
    assert clock.hard() is None
    clock.note_step(2.0)
    assert clock.hard() is None  # a 2 s step still ends inside the cap
    clock.note_step(3.5)
    assert clock.hard() == "lock-cap" and clock.step_s == 3.5
    clock.note_step(1.0)
    assert clock.step_s == 3.5  # the longest step is kept
    timed = Clock(0, hard_at=time.monotonic() + 3.0)
    timed.note_step(5.0)
    assert timed.hard() is None  # a turn deadline keeps its own margin (unchanged from v4)

"""The quick-menu floor (yomibench.anytime.think(floor=True), on the toy game): a fast search that finishes plays
exactly the plain search; a slow one whose full search is cut before it converges plays the floor (a certified small
menu), never its 1-2 round unconverged stub; only a converged round replaces the floor; and human play's auto switch
follows the measured preview rate."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from yomibench import anytime
from yomibench.anytime import Clock
from yomibench.bots import MatrixBot


def _toy_position(moves=((2, 3), (4, 1), (0, 6))):
    from yomibench.engine import connect

    e = connect("toy")
    obs = e.new_match("A", "B", 1)
    for a, b in moves:
        obs = e.step(a, b)
    return e, obs


class _RoundsBot(MatrixBot):
    """A matrix bot whose full search reports a round every `every` seconds (its `progress` hook, as the double
    oracle's `on_round` does), converged on the rounds in `converged`, and ends only when the clock stops it. Its
    quick turn (`cheap`) is the plain bot's, so the floor is a real certified search."""

    def __init__(self, every: float, converged=()):
        super().__init__(search="oracle")
        self.every, self.converged = every, set(converged)

    def cheap(self):
        b = MatrixBot(search="oracle").cheap()
        b.value = self.value
        return b

    def search_turn(self, obs, engine, seat=None):
        lists = [list(obs.candidates[0]), list(obs.candidates[1])]
        k = 0
        while True:
            t = time.monotonic()
            while time.monotonic() - t < self.every:
                self.value(obs.state, 0)  # the guarded evaluator: the stop check
                time.sleep(0.002)
            k += 1
            row, col = [0.0] * len(lists[0]), [0.0] * len(lists[1])
            row[k % len(row)], col[k % len(col)] = 1.0, 1.0  # a degenerate 100% pick: the stub
            if self.progress is not None:
                self.progress(row, col, 0.1 * k, lists, k in self.converged)


def _cut_clock(seconds: float) -> Clock:
    now = time.monotonic()
    return Clock(0, hard_at=now + seconds, emergency_at=now + seconds / 2)


def _plain(moves=((2, 3), (4, 1), (0, 6))):
    e, obs = _toy_position(moves)
    return MatrixBot(search="oracle").search_turn(obs, e, 1)


# --------------------------------------------------------------------------- (a) a fast PC: nothing changes


@pytest.mark.parametrize("moves", [((2, 3), (4, 1), (0, 6)), ((1, 1),), ((5, 0), (3, 3))])
def test_fast_the_full_search_finishes_and_plays_exactly_the_plain_search(moves):
    e, obs = _toy_position(moves)
    clock = Clock(0, hard_at=time.monotonic() + 60)  # a cuttable decision that the search beats
    out = anytime.think(MatrixBot(search="oracle"), obs, 1, e, clock, deep=False, floor=True)
    assert out.path == "full" and out.best is out.production and out.stopped == "done"
    assert out.trace[0]["name"] == "floor" and out.trace[1]["name"] == "production"
    assert (out.best.row, out.best.col, out.best.value) == _plain(moves)  # the floor's shared previews change nothing
    e2, o2 = _toy_position(moves)
    off = anytime.think(MatrixBot(search="oracle"), o2, 1, e2, Clock(0, hard_at=time.monotonic() + 60), deep=False)
    assert [[c.id for c in x] for x in out.best.lists] == [[c.id for c in x] for x in off.best.lists]
    assert out.best.info == off.best.info


def test_fast_deep_stages_build_on_production_not_the_floor():
    e, obs = _toy_position()
    clock = Clock(0, hard_at=time.monotonic() + 120)
    on = anytime.think(MatrixBot(search="oracle"), obs, 1, e, clock, deep=True, floor=True)
    e2, o2 = _toy_position()
    off = anytime.think(MatrixBot(search="oracle"), o2, 1, e2, Clock(0, hard_at=time.monotonic() + 120), deep=True)
    assert on.path == off.path == "full" and on.best.name == off.best.name == "two-decision"
    assert (on.best.row, on.best.col, on.best.value) == (off.best.row, off.best.col, off.best.value)


def test_the_floor_runs_only_on_a_decision_a_clock_can_cut():
    e, obs = _toy_position()
    out = anytime.think(MatrixBot(search="oracle"), obs, 1, e, Clock(0), deep=False, floor=True)  # untimed, no cap
    assert out.path == "full" and all(t.get("name") != "floor" for t in out.trace)
    e2, o2 = _toy_position()
    capped = Clock(0)
    capped.lock_cap = 10.0  # the after-lock cap: cuttable once the human locks in
    out2 = anytime.think(MatrixBot(search="oracle"), o2, 1, e2, capped, deep=False, floor=True)
    assert out2.path == "full" and out2.trace[0]["name"] == "floor"


def test_the_floor_is_certified_and_shares_its_base_previews():
    e, obs = _toy_position()
    d = anytime.Deepener(MatrixBot(search="oracle"), obs, 1, e, Clock(0))
    f = d.run_floor()
    assert f.name == "floor" and f.stage == 0 and f.info["converged"] and not f.info["fallback"]
    assert f.info["cap"] == anytime.QUICK_CAP and f.info["shared_previews"] > 0
    assert len(e.turn_cache["previewed"]) == f.info["shared_previews"]
    assert f.label() == "quick search (floor)"


# --------------------------------------------------------------------------- (b) a slow PC: the floor, not a stub


def test_slow_a_cut_unconverged_full_search_plays_the_floor_not_its_stub():
    e, obs = _toy_position()
    out = anytime.think(_RoundsBot(every=0.05), obs, 1, e, _cut_clock(0.4), deep=False, floor=True)
    assert out.production is None and out.stopped == "deadline"
    assert out.path == "floor" and out.best.name == "floor"
    assert out.best.info["converged"] and out.best.info["rounds_seen"] >= 1  # the stub rounds it refused
    e2, o2 = _toy_position()
    plain_floor = anytime.Deepener(MatrixBot(search="oracle"), o2, 1, e2, Clock(0))._quick_search("floor")
    assert (out.best.row, out.best.col, out.best.value) == (plain_floor.row, plain_floor.col, plain_floor.value)


def test_slow_without_the_floor_the_stub_is_what_would_play():
    """The regression the floor fixes: the same cut search without it plays its latest unconverged round."""
    e, obs = _toy_position()
    out = anytime.think(_RoundsBot(every=0.05), obs, 1, e, _cut_clock(0.4), deep=False)
    assert out.path == "partial-full" and out.best.info["converged"] is False
    assert max(out.best.row) == 1.0 and max(out.best.col) == 1.0  # a degenerate 100% pick


def test_slow_with_a_floor_there_is_no_emergency_quick_search(monkeypatch):
    calls = []
    monkeypatch.setattr(anytime.Deepener, "quick", lambda self: calls.append(1))
    e, obs = _toy_position()
    out = anytime.think(_RoundsBot(every=10.0), obs, 1, e, _cut_clock(0.4), deep=False, floor=True)  # no round at all
    assert out.path == "floor" and not calls and out.stopped == "deadline"


def test_a_hard_stop_during_the_floor_holds():
    e, obs = _toy_position()
    clock = Clock(0, hard_at=time.monotonic() - 1)  # already past the deadline
    out = anytime.think(_RoundsBot(every=0.05), obs, 1, e, clock, deep=False, floor=True)
    assert out.best is None and out.path == "continue" and out.stopped == "deadline"


# --------------------------------------------------------------------------- (c) no clobbers


def test_a_converged_round_replaces_the_floor():
    e, obs = _toy_position()
    out = anytime.think(_RoundsBot(every=0.04, converged={2}), obs, 1, e, _cut_clock(0.5), deep=False, floor=True)
    assert out.path == "partial-full" and out.best.info["converged"] is True
    assert out.best.value == pytest.approx(0.2)  # round 2, the converged one ...
    assert out.best.info["unconverged_rounds_after"] >= 1  # ... kept over every later unconverged round


@pytest.mark.parametrize("converged", [(), (1,), (3,)])
def test_never_an_unconverged_answer_over_a_certified_one(converged):
    e, obs = _toy_position()
    out = anytime.think(_RoundsBot(every=0.04, converged=converged), obs, 1, e, _cut_clock(0.5), deep=False,
                        floor=True)
    assert out.best is not None and out.best.info["converged"] is True
    assert out.path == ("partial-full" if converged else "floor")


# --------------------------------------------------------------------------- human play's auto switch


def _session(floor="auto", work=(0, 0.0)):
    from yomibench.human_play import Session

    s = SimpleNamespace(floor=floor, preview_work=list(work))
    for name in ("preview_rate", "use_floor", "note_preview_rate"):
        setattr(s, name, getattr(Session, name).__get__(s))
    return s


def test_auto_floor_follows_the_measured_preview_rate():
    assert _session().use_floor() is True  # nothing measured yet: the safe side
    slow = _session(work=(900, 20.0))  # 45 previews/s, the slow PC
    assert slow.preview_rate() == 45.0 and slow.use_floor() is True
    fast = _session(work=(6000, 10.0))  # 600/s
    assert fast.use_floor() is False
    assert _session("on", (6000, 10.0)).use_floor() is True and _session("off").use_floor() is False
    s = _session()
    s.note_preview_rate([{"stage": 1, "ms": 500.0, "work": {"previews": 400}},
                         {"stage": 1, "ms": 10.0, "work": {"previews": 0}}])
    assert s.preview_work == [400, 0.5] and s.use_floor() is False  # 800/s


def test_auto_floor_rate_ignores_the_deep_stages():
    """Thinking until lock-in: two-decision stages spend their time on branch continues, not previews. A fast PC's
    rate is its quick and full searches' alone, so the deep stages never pull it under the threshold."""
    s = _session()
    s.note_preview_rate([
        {"stage": 0, "ms": 200.0, "work": {"previews": 150, "continues": 0}},   # the floor
        {"stage": 1, "ms": 800.0, "work": {"previews": 450, "continues": 0}},   # the full search: 600/s together
        {"stage": 2, "ms": 3000.0, "work": {"previews": 120, "continues": 0}},  # widen
        {"stage": 3, "ms": 9000.0, "work": {"previews": 40, "continues": 300}},  # two-decision, cut by a lock-in
        {"stage": 4, "ms": 6000.0, "stopped": "lock-in", "work": {"previews": 10, "continues": 200}},
    ])
    assert s.preview_work == [600, 1.0] and s.preview_rate() == 600.0 and s.use_floor() is False
    blended = (600 + 120 + 40 + 10) / 19.0  # what counting every stage would say: ~40/s, a "slow" PC
    assert blended < anytime.FLOOR_RATE
    cut = _session()
    cut.note_preview_rate([{"stage": 1, "ms": 2000.0, "stopped": "deadline", "work": {"previews": 90}}])
    assert cut.preview_work == [90, 2.0]  # a cut full search still counts: it measured this PC


def test_session_refuses_an_unknown_floor():
    from yomibench.human_play import Session

    with pytest.raises(ValueError, match="floor"):
        Session("matrix-do", 6, "unused", floor="sometimes")

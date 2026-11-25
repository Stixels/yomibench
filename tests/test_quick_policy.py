"""Policy-picked quick menus (yomibench.quick_policy, cheap_cap_policy policy40+gumbel8[+parryx6]) and human play's
quick policy (yomibench.anytime with --quick-policy): menu rules, no double counting with +parryx6, determinism, the
pinned file, and that off changes nothing (the options-shuffle quick turn, the random warm start, full child lists)."""

import hashlib
import json
import zlib

import numpy as np
import pytest

from yomibench import anytime
from yomibench.anytime import Clock
from yomibench.bots import MatrixBot, policy_start
from yomibench.engine import Candidate


def parry_list(n_timings=20, others=8):
    """Two Continues, a ParryHigh (2 heights x n timings x 3 DI aims), then `others` moves of 4 aims each."""
    cands = [Candidate(0, "Continue", None, {"DI": {"x": 0, "y": 0}}),
             Candidate(1, "Continue", None, {"DI": {"x": 50, "y": 0}})]
    for y in (0, 1):
        for t in range(1, n_timings + 1):
            for dx in (-50, 0, 50):
                cands.append(Candidate(len(cands), "ParryHigh", {"Block Height": {"x": 0, "y": y},
                                                                 "Melee Parry Timing": {"count": t}},
                                       {"DI": {"x": dx, "y": 0}}))
    for k in range(others):
        for x in range(4):
            cands.append(Candidate(len(cands), f"Move{k}", {"x": x * 10, "y": 0}, None))
    return cands


def parry_key(c):
    return (c.data["Block Height"]["y"], c.data["Melee Parry Timing"]["count"])


class StubNet:
    """A deterministic stand-in for PolicyNet: a fixed pseudo-random score per candidate, plus `favour`."""

    def __init__(self, favour=None):
        self.favour = favour or (lambda c: 0.0)

    def scores(self, state, seat, cands):
        return np.array([(zlib.crc32(json.dumps([c.action, c.data, c.extra], sort_keys=True).encode()) % 1000) / 250.0
                         + self.favour(c) for c in cands])


def _moves(cands):
    return {c.action for c in cands}


# --------------------------------------------------------------------------- policy40+gumbel8


def test_policy_menu_keeps_continues_every_move_and_the_cap_in_list_order():
    from yomibench.quick_policy import policy_menu

    cands = parry_list()
    sc = StubNet().scores(None, 0, cands)
    for s in range(30):
        menu = policy_menu(cands, sc, 40, 8, f"5:{s}:0")
        assert len(menu) == 48 and len({c.id for c in menu}) == 48
        assert [c.id for c in menu] == sorted(c.id for c in menu)
        assert {c.id for c in cands if c.action == "Continue"} <= {c.id for c in menu}
        assert _moves(menu) == _moves(cands)
    small = cands[:30]
    assert policy_menu(small, sc[:30], 40, 8, "x") == small


def test_policy_part_follows_the_scores_and_random_slots_reach_low_scores():
    from yomibench.quick_policy import policy_menu

    cands = parry_list()
    favour = StubNet(lambda c: 100.0 if c.action.startswith("Move") else 0.0)
    sc = favour.scores(None, 0, cands)
    reached = set()
    for s in range(200):
        menu = policy_menu(cands, sc, 40, 8, f"9:{s}:1")
        assert all(c.id in {m.id for m in menu} for c in cands if c.action.startswith("Move"))
        reached |= {c.id for c in menu if c.action == "ParryHigh"}
    assert reached == {c.id for c in cands if c.action == "ParryHigh"}


def test_policy_menu_is_deterministic_per_salt():
    from yomibench.quick_policy import policy_menu

    cands = parry_list()
    sc = StubNet().scores(None, 0, cands)
    a = [c.id for c in policy_menu(cands, sc, 40, 8, "11:3:0")]
    assert a == [c.id for c in policy_menu(cands, sc.copy(), 40, 8, "11:3:0")]
    assert a != [c.id for c in policy_menu(cands, sc, 40, 8, "11:3:1")]


@pytest.mark.parametrize("favour_parry", [False, True])
def test_parryx6_adds_timings_on_top_without_duplicates(favour_parry):
    from yomibench.playout import _shuffled_options, add_extra_count_options
    from yomibench.quick_policy import QuickPolicy

    cands = parry_list()
    net = StubNet((lambda c: 50.0 if c.action == "ParryHigh" else 0.0) if favour_parry else None)
    qp = QuickPolicy(net, "0" * 64, "stub.npz")
    for s in range(25):
        salt = f"3:{s}"
        plain = qp.menus({}, [cands], 48, "policy40+gumbel8", salt)[0]
        got = qp.menus({}, [cands], 48, "policy40+gumbel8+parryx6", salt)[0]
        ids = [c.id for c in got]
        assert len(ids) == len(set(ids)) and ids == sorted(ids)
        assert {c.id for c in plain} <= set(ids)
        extra = [c for c in got if c.id not in {p.id for p in plain}]
        assert all(c.action == "ParryHigh" for c in extra)
        shown = {parry_key(c) for c in plain if c.action == "ParryHigh"}
        assert not {parry_key(c) for c in extra} & shown
        m = [c for c in cands if c.action == "ParryHigh"]
        heads, _ = _shuffled_options(cands, [c.id for c in m], f"{salt}:0", counts=True, per_class=6)
        assert {parry_key(c) for c in extra} == {parry_key(cands[i]) for i in heads} - shown
        assert len(extra) <= 12
        assert {(y, k) for y in (0, 1) for k in (1, 20, 10, 5, 15, 3)} <= {parry_key(c) for c in got
                                                                          if c.action == "ParryHigh"}
        assert add_extra_count_options(cands, plain, f"{salt}:0", 6) == got
        scored = qp.scored_menus({}, [cands], 48, "policy40+gumbel8+parryx6", salt)[0]
        assert scored[0] == got  # the same menu, with its candidates' scores
        full = net.scores({}, 0, cands)
        assert list(scored[1]) == [full[c.id] for c in got]


def test_menus_need_cap_k_plus_g_and_keep_short_lists_whole():
    from yomibench.quick_policy import QuickPolicy, check_cap, parse

    assert parse("policy40+gumbel8") == (40, 8, 0) and parse("policy40+gumbel8+parryx6") == (40, 8, 6)
    with pytest.raises(ValueError):
        check_cap("policy40+gumbel8", 32)
    with pytest.raises(ValueError):
        parse("policy41+gumbel8")
    qp = QuickPolicy(StubNet(), "0" * 64, "stub.npz")
    short = parry_list(n_timings=3, others=2)
    assert qp.menus({}, [short, short], 48, "policy40+gumbel8+parryx6", "1:1") == [short, short]
    (m, sc), _ = qp.scored_menus({}, [short, short], 48, "policy40+gumbel8+parryx6", "1:1", need_scores=True)
    assert m == short and len(sc) == len(short)


def test_the_count_options_are_off_for_every_cap_policy():
    """The ported count-widget code (`counts`) is used only by +parryxT: options-shuffle's lists are unchanged."""
    from yomibench.playout import CAP_POLICIES, _option_tuples, cap_candidates

    cands = parry_list()
    m = [c for c in cands if c.action == "ParryHigh"]
    assert len(set(_option_tuples(m)[0])) == 2  # the heights only: timings are not options without counts
    assert len(set(_option_tuples(m, counts=True)[0])) == 40
    assert CAP_POLICIES == ("list", "options-first", "rotate", "options-rotate", "options-shuffle")
    menu = cap_candidates(cands, 32, "options-shuffle", "7:1:0")
    assert len({parry_key(c) for c in menu if c.action == "ParryHigh"}) < 40


# --------------------------------------------------------------------------- the pinned file


def test_load_refuses_a_missing_file_or_a_sha_mismatch(tmp_path, monkeypatch):
    from yomibench import policy
    from yomibench.quick_policy import QuickPolicy

    with pytest.raises(SystemExit, match="no such file"):
        QuickPolicy.load(tmp_path / "nope.npz", "0" * 64)
    f = tmp_path / "q.npz"
    f.write_bytes(b"not a real net")
    with pytest.raises(SystemExit, match="sha256"):
        QuickPolicy.load(f, "0" * 64)
    monkeypatch.setattr(policy.PolicyNet, "load", staticmethod(lambda p: StubNet()))
    sha = hashlib.sha256(b"not a real net").hexdigest()
    qp = QuickPolicy.load(f, sha.upper())
    assert qp.header() == {"sha256": sha, "file": "q.npz"}


def test_human_play_flags_go_together_and_default_off(tmp_path, monkeypatch):
    import argparse

    from yomibench import policy
    from yomibench.human_play import add_play_args, load_quick_policy

    ap = argparse.ArgumentParser()
    add_play_args(ap)
    assert load_quick_policy(ap.parse_args([])) is None
    f = tmp_path / "q.npz"
    f.write_bytes(b"x")
    with pytest.raises(SystemExit, match="go together"):
        load_quick_policy(ap.parse_args(["--quick-policy", str(f)]))
    with pytest.raises(SystemExit, match="sha256"):
        load_quick_policy(ap.parse_args(["--quick-policy", str(f), "--quick-policy-sha256", "0" * 64]))
    monkeypatch.setattr(policy.PolicyNet, "load", staticmethod(lambda p: StubNet()))
    qp = load_quick_policy(ap.parse_args(["--quick-policy", str(f), "--quick-policy-sha256",
                                          hashlib.sha256(b"x").hexdigest()]))
    assert qp is not None and qp.file == "q.npz"


# --------------------------------------------------------------------------- human play's search (anytime)


def _toy_position(moves=((2, 3), (4, 1), (0, 6))):
    from yomibench.engine import connect

    e = connect("toy")
    obs = e.new_match("A", "B", 1)
    for a, b in moves:
        obs = e.step(a, b)
    return e, obs


class _Recording:
    """A QuickPolicy stand-in: StubNet scores; records each menus call; menus keep each list's first `keep`."""

    sha256, file = "f" * 64, "stub.npz"

    def __init__(self, keep=3):
        self.keep, self.calls, self.net = keep, [], StubNet()

    def header(self):
        return {"sha256": self.sha256, "file": self.file}

    def scores(self, state, seat, cands):
        self.calls.append(("scores", seat, len(cands)))
        return self.net.scores(state, seat, cands)

    def scored_menus(self, state, candidates, cap, policy, salt, need_scores=False):
        self.calls.append(("menus", cap, policy, salt))
        out = []
        for seat, c in enumerate(candidates):
            m = list(c)[:self.keep]
            out.append((m, self.net.scores(state, seat, m)))
        return out

    def menus(self, state, candidates, cap, policy, salt):
        return [m for m, _ in self.scored_menus(state, candidates, cap, policy, salt)]


def test_policy_start_is_the_first_candidate_and_the_best_scored_others():
    assert policy_start([0.0, 5.0, 1.0, 9.0], 2) == [0, 3, 1]
    assert policy_start([0.0, 1.0, 1.0], 2) == [0, 1, 2]
    assert policy_start([3.0], 2) == [0]


def _digest(out):
    rows = [(t.get("stage"), t.get("name"), t.get("kept"), t.get("value"), t.get("pairs")) for t in out.trace]
    b = out.best
    return (out.path, out.stopped, rows, b.name, b.row, b.col, b.value,
            [[c.id for c in lst] for lst in b.lists], b.info)


def test_off_every_stage_is_the_one_without_the_option():
    """quick_policy=None is today's search: the same answer and trace as before the option existed (think's
    default), every stage run to the end, and the quick turn's options-shuffle menu."""
    e1, o1 = _toy_position()
    a = anytime.think(MatrixBot(search="oracle"), o1, 1, e1, Clock(0), deep=True)
    e2, o2 = _toy_position()
    b = anytime.think(MatrixBot(search="oracle"), o2, 1, e2, Clock(0), deep=True, quick_policy=None)
    assert _digest(a) == _digest(b) and a.best.name == "two-decision"
    e3, o3 = _toy_position()
    q = anytime.Deepener(MatrixBot(search="oracle"), o3, 1, e3, Clock(0)).quick()
    assert q.info["cap"] == anytime.QUICK_CAP and q.info["cap_policy"] == "options-shuffle"
    assert "quick_policy" not in q.info
    assert MatrixBot(search="oracle")._key(1) == MatrixBot(search="oracle")._key(1)  # no start_policy in the key


def test_on_the_quick_turn_searches_the_policy_menus():
    e, obs = _toy_position()
    qp = _Recording(keep=3)
    q = anytime.Deepener(MatrixBot(search="oracle"), obs, 1, e, Clock(0), quick_policy=qp).quick()
    assert ("menus", anytime.QUICK_POLICY_CAP, anytime.QUICK_POLICY_MENU, f"{e.match_seed}:{obs.turn}") in qp.calls
    assert [len(x) for x in q.lists] == [3, 3]
    assert q.info["cap"] == 48 and q.info["cap_policy"] == "policy40+gumbel8+parryx6"
    assert q.info["quick_policy"] == qp.sha256


def test_on_the_full_search_warm_starts_from_the_policy_and_still_certifies():
    e, obs = _toy_position()
    qp = _Recording()
    out = anytime.think(MatrixBot(search="oracle"), obs, 1, e, Clock(0), deep=False, quick_policy=qp)
    prod = out.production
    assert prod is not None and prod.info.get("converged")
    assert ("scores", 0, len(obs.candidates[0])) in qp.calls and ("scores", 1, len(obs.candidates[1])) in qp.calls
    assert ("start_policy", qp.sha256) in prod.view
    # the same lists as the plain search, and an equilibrium of them (certified against every candidate)
    e2, o2 = _toy_position()
    plain = anytime.think(MatrixBot(search="oracle"), o2, 1, e2, Clock(0), deep=False)
    assert [[c.id for c in x] for x in prod.lists] == [[c.id for c in x] for x in plain.production.lists]
    assert abs(prod.value - plain.production.value) < 2e-3


def test_on_two_decision_children_search_the_policy_menus():
    e, obs = _toy_position()
    qp = _Recording(keep=2)
    out = anytime.think(MatrixBot(search="oracle"), obs, 1, e, Clock(0), deep=True, quick_policy=qp)
    child = [c for c in qp.calls if c[0] == "menus" and c[3].endswith(":depth2")]
    assert child and all(c[1] == 48 and c[2] == anytime.QUICK_POLICY_MENU for c in child)
    assert out.best is not None and out.best.name == "two-decision" and out.best.pairs > 0

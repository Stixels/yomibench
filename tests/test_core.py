import random
import sys

import pytest

from yomibench.arena import bradley_terry, play_game, run_round_robin
from yomibench.bots import GreedyBot, MatrixBot, RandomBot
from yomibench.engine import Engine, EngineError, Observation, Subprocess, connect
from yomibench.search import solve_zero_sum


def test_rock_paper_scissors_solves_to_uniform():
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    strategy, value = solve_zero_sum(rps, 5000)
    assert all(abs(p - 1 / 3) < 0.02 for p in strategy)
    assert abs(value) < 0.02


def test_dominated_row_gets_no_weight():
    strategy, value = solve_zero_sum([[1, 1], [0, 0]], 500)
    assert strategy[0] > 0.99
    assert value == pytest.approx(1, abs=0.01)


def test_preview_does_not_change_the_match():
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    engine.preview([(0, 0), (4, 5)])
    assert engine.observe() == obs


def test_preview_matches_step():
    engine = connect("toy")
    engine.new_match("Toy", "Toy", 0)
    for pair in [(4, 4), (4, 3), (5, 8), (7, 6)]:
        (predicted,) = engine.preview([pair])
        real = engine.step(*pair)
        assert real.state == predicted.state
        assert real.candidates == predicted.candidates


def test_illegal_candidate_is_an_error_not_a_crash():
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    with pytest.raises(EngineError):
        engine.step(len(obs.candidates[0]), 0)  # Super is not offered without meter


def test_games_are_reproducible():
    bots = (MatrixBot(iterations=200), RandomBot())
    a = play_game(connect("toy"), bots, ("Toy", "Toy"), seed=7)
    b = play_game(connect("toy"), bots, ("Toy", "Toy"), seed=7)
    assert (a.outcome, a.turns, a.hp) == (b.outcome, b.turns, b.hp)


def test_subprocess_engine_speaks_the_protocol():
    engine = Engine(Subprocess([sys.executable, "-m", "yomibench.toy"]))
    try:
        assert engine.hello()["engine"] == "toy"
        obs = engine.new_match("Toy", "Toy", 0)
        assert len(engine.preview([(0, 0)])) == 1
        assert engine.step(0, 0).turn == obs.turn + 1
    finally:
        engine.close()


def test_search_beats_random():
    results = run_round_robin(
        lambda: connect("toy"), [RandomBot(), MatrixBot(iterations=300)], ["Toy"], pairs_per_matchup=5, seed=3
    )
    ratings = bradley_terry(results, anchor="random")
    assert ratings["matrix"] > ratings["random"] + 200


def _pair(seed, a, b, first, second):
    """A seat-swapped pair: a (seat 0) v b, then b (seat 0) v a, with each game's outcome."""
    from yomibench.arena import GameResult

    return [
        GameResult((a, b), ("Toy", "Toy"), seed, first, 1, (1, 0), 0.0),
        GameResult((b, a), ("Toy", "Toy"), seed, second, 1, (1, 0), 0.0),
    ]


def test_bradley_terry_orders_by_results():
    results = [r for s in range(4) for r in _pair(s, "a", "b", "p1", "p2")] + _pair(9, "a", "b", "p2", "p1")
    r = bradley_terry(results, anchor="b")
    assert r["b"] == 1000 and r["a"] > 1100


def test_a_halt_drops_the_whole_seat_swapped_pair():
    from yomibench.arena import rated_pairs, win_matrix

    results = _pair(1, "a", "b", "p1", "p2") + _pair(2, "a", "b", "p1", "halt")
    assert len(rated_pairs(results)) == 1
    assert win_matrix(results)[("a", "b")] == (1.0, 2)


def test_bootstrap_interval_contains_the_rating():
    from yomibench.arena import bradley_terry_ci

    results = [r for s in range(20) for r in _pair(s, "a", "b", "p1" if s % 4 else "p2", "p2")]
    lo, hi = bradley_terry_ci(results, anchor="b")["a"]
    assert lo <= bradley_terry(results, anchor="b")["a"] <= hi and lo < hi


def test_both_seats_share_one_preview_table():
    from yomibench.engine import InProcess
    from yomibench.toy import ToyEngine

    class Counting(InProcess):
        previews = 0

        def request(self, msg):
            Counting.previews += msg.get("cmd") == "preview"
            return super().request(msg)

    engine = Engine(Counting(ToyEngine()))
    result = play_game(engine, (MatrixBot(iterations=100), GreedyBot()), ("Toy", "Toy"), seed=1, max_turns=5)
    assert result.turns == 5 and Counting.previews == 5


def test_prune_drops_residual_weight():
    from yomibench.search import prune

    assert prune([0.6, 0.39, 0.01]) == pytest.approx([0.6 / 0.99, 0.39 / 0.99, 0.0])
    assert prune([0.01, 0.01]) == pytest.approx([0.5, 0.5])  # ties at the top both survive


def test_garbled_engine_output_is_a_crash():
    from yomibench.engine import EngineCrashed

    engine = Engine(Subprocess([sys.executable, "-c", "import sys; sys.stdin.readline(); print('{trunc')"]))
    try:
        with pytest.raises(EngineCrashed):
            engine.hello()
    finally:
        engine.close()


def test_default_move_is_candidate_zero_and_lists_are_never_empty():
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    while not obs.done:
        assert all(len(seat) >= 1 and seat[0].action == "Wait" for seat in obs.candidates)
        obs = engine.step(1, 2)
    assert obs.candidates == ((), ())


def test_continue_from_a_preview_matches_stepping():
    engine = connect("toy")
    engine.new_match("Toy", "Toy", 0)
    (first,) = engine.preview([(4, 4)])
    second = engine.continue_from(first.handle, (5, 7))
    assert second.handle != first.handle and second.turn == first.turn + 1
    engine.step(4, 4)
    assert engine.step(5, 7).state == second.state


def test_continue_with_null_pair_plays_defaults_for_at_least_n_ticks():
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    (leaf,) = engine.preview([(0, 0)], candidates=False)
    assert leaf.candidates == ((), ())
    rolled = engine.continue_from(leaf.handle, None, until=5)
    assert rolled.state["tick"] >= leaf.state["tick"] + 5
    for _ in range(rolled.turn - obs.turn):
        obs = engine.step(0, 0)
    assert obs.state == rolled.state


def test_handles_go_stale_after_step_and_can_be_released():
    engine = connect("toy")
    engine.new_match("Toy", "Toy", 0)
    a, b = engine.preview([(0, 0), (1, 1)])
    assert engine.release([a.handle]) == 1
    with pytest.raises(EngineError, match="stale_handle"):
        engine.continue_from(a.handle)
    engine.step(0, 0)
    with pytest.raises(EngineError, match="stale_handle"):
        engine.continue_from(b.handle)


def test_continue_ids_index_the_handles_own_candidates():
    engine = connect("toy")
    engine.new_match("Toy", "Toy", 0)
    (leaf,) = engine.preview([(0, 0)])
    with pytest.raises(EngineError, match="bad_request"):
        engine.continue_from(leaf.handle, (len(leaf.candidates[0]), 0))


def test_observation_parses_optional_fields():
    raw = {"turn": 1, "done": False, "winner": None, "state": {"players": []}}
    obs = Observation.parse(raw)
    assert obs.candidates == ((), ()) and obs.actionable == (True, True) and obs.handle is None
    obs = Observation.parse({**raw, "actionable": [True, False], "handle": 9, "candidates": [[], []]})
    assert obs.actionable == (True, False) and obs.handle == 9


def test_parallel_round_robin_matches_sequential():
    bots = [RandomBot(), GreedyBot()]
    seq = run_round_robin(lambda: connect("toy"), bots, ["Toy"], pairs_per_matchup=2, seed=5)
    par = run_round_robin([lambda: connect("toy")] * 3, bots, ["Toy"], pairs_per_matchup=2, seed=5)
    key = lambda r: (r.bots, r.seed, r.outcome, r.turns, r.hp)  # noqa: E731
    assert [key(r) for r in seq] == [key(r) for r in par]


def test_run_dir_gets_override_cfg_and_our_scripts_only(tmp_path):
    from yomibench import game

    install = tmp_path / "install"
    install.mkdir()
    libs = ["libtbfg.so", "libsteam_api.so"] if game.LINUX else ["tbfg.dll", "steam_api64.dll"]
    for name in (game.EXE, game.PACK, *libs):
        (install / name).write_bytes(b"x")
    (install / "mods").mkdir()  # not copied
    run = game.build_run_dir(game.find_game_dir(install), tmp_path / "run")
    assert sorted(p.name for p in run.iterdir()) == sorted([game.EXE, game.PACK, *libs, "override.cfg", "yomibench"])
    cfg = (run / "override.cfg").read_text()
    assert 'SteamHustle="*' in cfg and "/yomibench/SteamStub.gd" in cfg and "/yomibench/YomiServer.gd" in cfg
    from yomibench import plugins

    want = set(game.MOD_SCRIPTS) | set(plugins.mod_overlays("YomiBench"))  # plus an installed plugin's files
    assert sorted(p.name for p in (run / "yomibench").iterdir()) == sorted(want)
    # idempotent
    assert game.build_run_dir(install, run) == run


def test_failed_reconnect_keeps_the_halt_and_raises():
    from yomibench.engine import EngineCrashed, InProcess
    from yomibench.toy import ToyEngine

    class Dies(InProcess):
        def request(self, msg):
            if msg.get("cmd") == "step":
                raise EngineCrashed("boom")
            return super().request(msg)

    made = []

    def make():
        if made:
            raise ConnectionRefusedError("server gone")
        made.append(1)
        return Engine(Dies(ToyEngine()))

    seen = []
    with pytest.raises(ConnectionRefusedError):
        run_round_robin([make, make], [RandomBot(), GreedyBot()], ["Toy"], 1, seed=0, on_result=seen.append)
    assert [r.outcome for r in seen] == ["halt"]


def test_one_solve_gives_both_seats_strategies():
    from yomibench.search import solve_both

    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    row, col, value = solve_both(rps, 5000)
    assert all(abs(p - 1 / 3) < 0.02 for p in row + col) and abs(value) < 0.02
    row, col, value = solve_both([[3, 1], [2, 0]], 2000)  # row 0 dominates; column 1 dominates
    assert row[0] > 0.99 and col[1] > 0.99 and value == pytest.approx(1, abs=0.01)


def test_double_oracle_matches_the_full_table_with_fewer_evaluations():
    from yomibench.search import double_oracle, exploitability, solve_both

    rng = random.Random(4)
    for _ in range(20):
        n0, n1 = rng.randint(5, 25), rng.randint(5, 25)
        m = [[rng.choice([0.0, 0.0, 0.0, rng.uniform(-1, 1)]) for _ in range(n1)] for _ in range(n0)]
        asked = []
        r = double_oracle(n0, n1, lambda ps: asked.extend(ps) or [m[i][j] for i, j in ps])
        assert len(asked) == len(set(asked)) == len(r.known)  # each pair evaluated once
        _, _, v_full = solve_both(m, 4000)
        assert r.converged and abs(r.value - v_full) < 0.01
        gains = exploitability(m, r.row, r.col)
        assert max(gains) <= 1e-3 + 1e-9  # certified against the whole table, both sides


def _oracle(m, **kw):
    from yomibench.search import double_oracle

    return double_oracle(len(m), len(m[0]), lambda ps: [m[i][j] for i, j in ps], **kw)


def test_oracle_certifies_moves_already_in_the_restricted_game():
    from yomibench.search import exploitability

    m = [[1, -1], [-1, 0]]  # restricted to everything after one round; the coarse solve must refine
    r = _oracle(m)
    assert r.converged and max(exploitability(m, r.row, r.col)) <= 1e-3 + 1e-9


def test_oracle_counts_dropped_support_mass():
    from yomibench.search import exploitability

    e = 0.0008
    m = [[1, -1, -1], [-e, e, -e]]
    r = _oracle(m, start=([0, 1], [0, 1]))
    assert r.converged and max(exploitability(m, r.row, r.col)) <= 1e-3 + 1e-9


def test_oracle_reports_running_out_of_rounds():
    r = _oracle([[0, 0], [1, 1]], max_rounds=1)
    assert not r.converged


def test_two_different_bots_each_run_their_own_search():
    calls = [0, 0]

    def mine(state, seat):
        calls[0] += 1
        return 0.0

    def theirs(state, seat):
        calls[1] += 1
        return 0.0

    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    rng = random.Random(0)
    MatrixBot(iterations=50, value=mine).choose(obs, 0, engine, rng)
    MatrixBot(iterations=50, value=theirs).choose(obs, 1, engine, rng)
    assert calls[0] > 0 and calls[1] > 0


def test_oracle_search_finds_the_full_tables_value_at_every_position():
    from yomibench.search import exploitability

    full, oracle = MatrixBot(iterations=1000), MatrixBot(iterations=1000, search="oracle")
    a, b = connect("toy"), connect("toy")
    oa, ob = a.new_match("Toy", "Toy", 3), b.new_match("Toy", "Toy", 3)
    rng = random.Random(3)
    fewer = 0
    while not oa.done:
        _, _, v_full = full.search_turn(oa, a)
        row, col, v_do = oracle.search_turn(ob, b)
        n0, n1 = (len(c) for c in oa.candidates)
        m = [[full.value(a.turn_cache["previewed"][i, j], 0)
              for j in range(n1)] for i in range(n0)]
        assert abs(v_full - v_do) < 0.01 and sum(exploitability(m, row, col)) < 0.01
        fewer += len(b.turn_cache["previewed"]) < n0 * n1
        pick = (rng.randrange(n0), rng.randrange(n1))
        oa, ob = a.step(*pick), b.step(*pick)
    assert fewer > 0


def test_refinement_work_is_capped():
    from yomibench.search import double_oracle

    m = [[1, -1], [-1e-6, 1e-6]]
    asked_iterations = []
    import yomibench.search as search

    real = search.solve_both
    try:
        search.solve_both = lambda payoff, its: asked_iterations.append(its) or real(payoff, min(its, 4000))
        double_oracle(2, 2, lambda ps: [m[i][j] for i, j in ps], tolerance=1e-7, start=([0, 1], [0, 1]))
    finally:
        search.solve_both = real
    assert max(asked_iterations) <= 32000


def test_an_unconverged_oracle_falls_back_to_the_full_table(monkeypatch):
    import yomibench.bots as bots
    from yomibench.search import double_oracle

    monkeypatch.setattr(bots, "double_oracle", lambda *a, **k: double_oracle(*a, **{**k, "max_rounds": 1}))
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    obs = engine.step(4, 1)
    bot = MatrixBot(iterations=200, search="oracle")
    bot.search_turn(obs, engine)
    info = engine.turn_cache["oracle"]
    n0, n1 = (len(c) for c in obs.candidates)
    assert info["converged"] or (info["fallback"] and len(engine.turn_cache["previewed"]) == n0 * n1)


def test_serve_retries_a_failed_server_without_stopping_the_others(tmp_path, monkeypatch):
    from yomibench import game

    class FakeProc:
        n = 0

        def __init__(self):
            FakeProc.n += 1
            self.pid, self.returncode, self.killed = FakeProc.n, None, False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed, self.returncode = True, -9

        def wait(self):
            return self.returncode

    started, attempts = [], {}

    def launch(rd, port, *a, **k):
        p = FakeProc()
        started.append((port, p))
        return p

    def wait_ready(port, p, host="127.0.0.1"):
        attempts[port] = attempts.get(port, 0) + 1
        if port == 7001 and attempts[port] == 1:
            raise TimeoutError("not ready")

    ticks = iter(range(10))

    def sleep(s):
        if next(ticks, None) is None:
            raise KeyboardInterrupt

    clock = iter(float(i * 100) for i in range(1000))
    monkeypatch.setattr(game, "build_run_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(game, "find_game_dir", lambda *a: tmp_path)
    monkeypatch.setattr(game, "launch", launch)
    monkeypatch.setattr(game, "wait_ready", wait_ready)
    monkeypatch.setattr(game.time, "sleep", sleep)
    monkeypatch.setattr(game.time, "monotonic", lambda: next(clock))
    game.serve(3, 7000, None, None)
    assert attempts[7001] >= 2  # retried
    assert started[1][1].killed  # the failed attempt was killed, not leaked
    assert all(p.killed for _, p in started)  # everything cleaned up on exit


def test_game_server_puts_a_below_normal_game_back_once_ready(monkeypatch, tmp_path):
    # The game raises itself to AboveNormal as it starts; GameServer(below_normal=True) lowers it after wait_ready.
    from yomibench import game

    order = []

    class FakeProc:
        pid, returncode = 4242, None

        def poll(self):
            return self.returncode

    monkeypatch.setattr(game, "build_run_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(game, "find_game_dir", lambda *a: tmp_path)
    monkeypatch.setattr(game, "launch", lambda *a, **k: FakeProc())
    monkeypatch.setattr(game, "wait_ready", lambda port, p: order.append("ready"))
    monkeypatch.setattr(game, "set_below_normal", lambda pid: order.append(("lower", pid)))
    game.GameServer(port=7000, below_normal=True)
    assert order == ["ready", ("lower", 4242)]
    order.clear()
    game.GameServer(port=7000)
    assert order == ["ready"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows priority classes")
def test_set_below_normal_lowers_a_running_process():
    import ctypes
    import subprocess
    from ctypes import wintypes

    from yomibench import game

    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                         creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
    try:
        assert game.set_below_normal(p.pid)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        assert k32.GetPriorityClass(int(p._handle)) == subprocess.BELOW_NORMAL_PRIORITY_CLASS
    finally:
        p.kill()
        p.wait()


def _swapped(m):
    return [[-m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def test_solve_is_seat_symmetric():
    from yomibench.search import exploitability, solve_both

    # A symmetric game (m == -m transposed) with a continuum of equilibria: rows/columns 0 and 1
    # are copies. One-sided alternation picked different equilibria for the two seats.
    m = [[0.0, 0.0, 0.3], [0.0, 0.0, 0.3], [-0.3, -0.3, 0.0]]
    assert m == _swapped(m)
    row, col, value = solve_both(m, 500)
    assert row == col and value == pytest.approx(0, abs=1e-3)
    assert max(exploitability(m, row, col)) < 1e-3
    rng = random.Random(9)
    for _ in range(10):  # any game: swapping the seats swaps the answer exactly
        a = [[rng.uniform(-1, 1) for _ in range(rng.randint(2, 6))]]
        a += [[rng.uniform(-1, 1) for _ in a[0]] for _ in range(rng.randint(1, 5))]
        r1, c1, v1 = solve_both(a, 300)
        r2, c2, _ = solve_both(_swapped(a), 300)
        assert (r1, c1) == (c2, r2)


def test_double_oracle_is_seat_symmetric():
    from yomibench.search import exploitability

    rng = random.Random(11)
    for _ in range(10):
        n = rng.randint(4, 12)
        half = [[rng.choice([0.0, 0.0, rng.uniform(-1, 1)]) for _ in range(n)] for _ in range(n)]
        m = [[half[i][j] - half[j][i] for j in range(n)] for i in range(n)]  # symmetric game
        r = _oracle(m, start=([0, 1], [0, 1]))
        assert r.converged and r.row == r.col and max(exploitability(m, r.row, r.col)) <= 1e-3 + 1e-9
        a = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n + 2)]
        r1 = _oracle(a, start=([0, 2], [1]))
        r2 = _oracle(_swapped(a), start=([1], [0, 2]))
        assert (r1.row, r1.col, r1.value, r1.upper) == (r2.col, r2.row, -r2.upper, -r2.value)


def test_oracle_start_varies_by_match_and_seat():
    bot = MatrixBot(iterations=200, search="oracle")
    starts = []
    import yomibench.bots as bots

    real = bots.double_oracle
    try:
        bots.double_oracle = lambda *a, **k: starts.append(k["start"]) or real(*a, **k)
        for seed in [*range(6), 0]:
            engine = connect("toy")
            bot.search_turn(engine.new_match("Toy", "Toy", seed), engine)
    finally:
        bots.double_oracle = real
    assert len({str(s) for s in starts}) > 1  # not one opening for every match
    assert starts[-1] == starts[0]  # but reproducible per match


def test_opening_weights_stay_within_the_support():
    from yomibench.bots import opening_weights

    assert opening_weights([0.75, 0.25, 0.0], 1) == [1.0, 1/3, 0.0]
    assert opening_weights([0.8, 0.2, 0.0], 2) == [1.0, 0.5, 0.0]
    assert opening_weights([0.4, 0.4, 0.2, 0.0], 0) == [1.0, 1.0, 0.0, 0.0]
    assert opening_weights([1.0, 0.0], 100) == [1.0, 0.0]


def test_opening_turns_are_sampled_and_marked(tmp_path):
    from yomibench.selfplay import play_recorded

    engine = connect("toy")
    hello = engine.hello()
    bot = MatrixBot(iterations=200, search="oracle")
    lines = play_recorded(engine, bot, ("Toy", "Toy"), 5, hello, max_turns=12, opening=(3, 0.05, 0.0))
    turns = lines[1:-1]
    assert lines[0]["opening"] == {"turns": 3, "temperature": 0.05, "epsilon": 0.0}
    assert [bool(ln.get("opening")) for ln in turns] == [ln["turn"] < 3 and ln["strategies"] is not None
                                                         for ln in turns]
    assert any(ln.get("opening") for ln in turns) and not any("explore" in ln for ln in turns)


def test_opening_epsilon_picks_off_the_support_and_records_it():
    from yomibench.selfplay import play_recorded

    engine = connect("toy")
    bot = MatrixBot(iterations=200, search="oracle")
    off_support = explored = 0
    for seed in range(6):
        lines = play_recorded(engine, bot, ("Toy", "Toy"), seed, engine.hello(), max_turns=20, opening=(4, 1.0, 1.0))
        searched = [ln for ln in lines[1:-1] if ln["strategies"] is not None]
        for k, ln in enumerate(searched):
            want = [k < 4 and len(c) > 1 for c in ln["candidates"]]  # epsilon 1: every opening pick explores
            assert ln.get("explore", [False, False]) == want
            for seat in (0, 1):
                explored += want[seat]
                off_support += want[seat] and ln["strategies"][seat][ln["picks"][seat]] == 0
            if k >= 4:
                assert all(ln["strategies"][s][ln["picks"][s]] > 0 for s in (0, 1))  # back on the strategy
    assert explored > 0 and off_support > 0
    plain = play_recorded(engine, bot, ("Toy", "Toy"), 0, engine.hello(), max_turns=20)
    assert plain[0]["opening"] is None and not any("explore" in ln or "opening" in ln for ln in plain[1:-1])


def test_oracle_full_fallback_rechecks_the_certificate():
    from yomibench.bots import _certified_full
    from yomibench.search import exploitability, solve_both

    m = [[0.0, 0.7, -0.2], [-0.6, 0.0, 0.9], [0.3, -0.8, 0.0]]
    assert max(exploitability(m, *solve_both(m, 1)[:2])) > 1e-3
    row, col, _ = _certified_full(m, 1, 1e-3)
    assert max(exploitability(m, row, col)) <= 1e-3


def test_a_game_cut_off_by_the_turn_limit_is_truncated_and_not_rated():
    from yomibench.arena import bradley_terry, rated_pairs, win_matrix

    r = play_game(connect("toy"), (RandomBot(), GreedyBot()), ("Toy", "Toy"), 3, max_turns=2)
    assert r.outcome == "truncated" and r.turns == 2
    assert r.score("random") is None and r.score("greedy") is None
    finished = play_game(connect("toy"), (RandomBot(), GreedyBot()), ("Toy", "Toy"), 3)
    assert finished.outcome in ("p1", "p2", "draw")

    # A truncated game drops its seat-swapped pair, like a halt; it is never a draw.
    results = _pair(1, "a", "b", "p1", "p2") + _pair(2, "a", "b", "truncated", "draw") + _pair(3, "a", "b", "p1", "p2")
    assert len(rated_pairs(results)) == 2
    assert win_matrix(results)[("a", "b")] == (1.0, 4)
    assert bradley_terry(results, anchor="b") == bradley_terry(_pair(1, "a", "b", "p1", "p2")
                                                               + _pair(3, "a", "b", "p1", "p2"), anchor="b")


def test_round_robin_reports_truncated_games(capsys):
    from yomibench.cli import main

    main(["arena", "--bots", "random,greedy", "--pairs", "1", "--max-turns", "2"])
    out = capsys.readouterr().out
    assert "'truncated': 2" in out and "excluded from scores: 2 truncated at 2 turns, 0 halted" in out
    assert "0 complete seat-swapped pairs rated" in out


@pytest.mark.parametrize("search", ["full", "oracle"])
def test_records_carry_each_candidates_value_even_without_previews(tmp_path, search):
    from yomibench.search import evaluate
    from yomibench.selfplay import play_recorded, read_slim, write_record

    engine = connect("toy")
    for record_table in (True, False):
        lines = play_recorded(engine, MatrixBot(iterations=300, search=search), ("Toy", "Toy"), 2, engine.hello(),
                              max_turns=15, full_previews=record_table)
        searched = [ln for ln in lines[1:-1] if ln["strategies"] is not None]
        assert searched and all(ln["candidate_values"] is None for ln in lines[1:-1] if ln["strategies"] is None)
        for ln in searched:
            cv, (s0, s1) = ln["candidate_values"], ln["strategies"]
            assert [len(u) for u in cv] == [len(c) for c in ln["candidates"]]
            assert all(x == round(x, 4) for u in cv for x in u)  # rounded: records stay small
            # Each seat's own view: p1's guaranteed value is its worst case over p2's candidates.
            assert -max(cv[1]) == pytest.approx(ln["value"], abs=2e-4)
            # The strategy pair's value from either side: p1's, and minus p2's.
            assert sum(p * x for p, x in zip(s0, cv[0])) == pytest.approx(-sum(p * x for p, x in zip(s1, cv[1])),
                                                                           abs=1e-3)
            if record_table and search == "full":  # the whole table is stored: check against it
                table = {(i, j): evaluate(st, 0) for i, j, st in ln["previewed"]}
                for i in range(len(s0)):
                    assert cv[0][i] == pytest.approx(sum(s1[j] * table[i, j] for j in range(len(s1))), abs=2e-4)
                for j in range(len(s1)):
                    assert cv[1][j] == pytest.approx(-sum(s0[i] * table[i, j] for i in range(len(s0))), abs=2e-4)
        if not record_table:
            assert all("previewed" not in ln for ln in searched)
            path = write_record(tmp_path, 2, lines)
            turns = read_slim(path)[2]
            assert [t.candidate_values for t in turns] == [ln["candidate_values"] for ln in lines[1:-1]]

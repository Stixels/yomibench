"""arena/pool_eval --record-dir: gate games as format-3 records, decisions unchanged."""

import json
import sys
from pathlib import Path

import pytest

from yomibench.arena import play_game, run_round_robin
from yomibench.bots import make_bot
from yomibench.engine import connect
from yomibench.selfplay import read_slim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from render_record import read_record as load, replay  # noqa: E402


def _games(record_dir=None, specs=("matrix-do", "matrix"), make=lambda: connect("toy"), chars=("Toy",), turns=30):
    steps = []

    def engine():
        e = make()
        step = e.step
        e.step = lambda a, b: (steps.append((a, b)), step(a, b))[1]
        return e
    results = run_round_robin(engine, [make_bot(s) for s in specs], list(chars), 1, 5, max_turns=turns,
                              record_dir=record_dir)
    return results, steps


def test_a_recorded_arena_game_reads_back_and_replays_to_its_result(tmp_path):
    results, _ = _games(tmp_path)
    paths = sorted(tmp_path.glob("g*.jsonl.gz"))
    assert len(paths) == len(results) == 2
    for r in results:  # named by seed and seat order
        (path,) = [p for p in paths if p.name.startswith(f"g{r.seed:012d}-{r.bots[0]}-vs-{r.bots[1]}-")]
        header, last, turns = read_slim(path)
        assert header["format"] == 3 and header["seed"] == r.seed and list(header["characters"]) == list(r.characters)
        assert [b["bot"] for b in header["bots"]] == list(r.bots)
        assert [b["search"] for b in header["bots"]] == [{"matrix-do": "oracle", "matrix": "full"}[n] for n in r.bots]
        assert len(turns) == r.turns and last["result"]["outcome"] == r.outcome
        assert tuple(last["result"]["hp"]) == r.hp
        head, decisions, result = load(path)
        obs = replay(connect("toy"), head, decisions, result)  # state, candidates and result checked throughout
        assert obs.turn == r.turns


def test_recording_does_not_change_decisions(tmp_path):
    plain, plain_steps = _games()
    recorded, recorded_steps = _games(tmp_path)
    assert plain_steps == recorded_steps and len(plain_steps) > 0
    strip = lambda rs: [(r.bots, r.characters, r.seed, r.outcome, r.turns, r.hp) for r in rs]  # noqa: E731
    assert strip(plain) == strip(recorded)


def test_a_halted_game_leaves_no_record(tmp_path):
    from yomibench.engine import EngineCrashed

    engine = connect("toy")
    calls = []
    step = engine.step

    def crash(a, b):
        calls.append(1)
        if len(calls) == 3:
            raise EngineCrashed("boom")
        return step(a, b)
    engine.step = crash
    with pytest.raises(EngineCrashed):
        play_game(engine, (make_bot("matrix-do"), make_bot("matrix")), ("Toy", "Toy"), 9, record_dir=tmp_path)
    assert not list(tmp_path.iterdir()) and "expand" not in vars(engine)


def test_set4_records_rebuild_expanded_picks_and_match_the_unrecorded_game(tmp_path):
    sim = pytest.importorskip("yomisim")
    if not hasattr(sim, "Batch"):
        pytest.skip("needs the simulator plugin's batch engine")
    from yomibench import plugins

    if plugins.hook("batch") is None:
        pytest.skip("no batch plugin installed")
    BatchEngine = plugins.hook("batch").BatchEngine

    try:
        BatchEngine(1, candidate_set=4, hp_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no set-4 Batch here: {e}")
    make = lambda: BatchEngine(1, candidate_set=4, hp_only=True)  # noqa: E731
    kw = dict(specs=("matrix-do", "matrix-do@cheap=0.5:32"), make=make, chars=("Ninja", "Cowboy"), turns=10)
    plain, plain_steps = _games(**kw)
    recorded, recorded_steps = _games(tmp_path, **kw)
    assert plain_steps == recorded_steps
    paths = sorted(tmp_path.glob("g*.jsonl.gz"))
    assert len(paths) == len(recorded) == 8
    expanded = 0
    for path in paths:
        head, decisions, result = load(path)
        expanded += sum("base" in t["candidates"][s][t["picks"][s]] for t in decisions for s in (0, 1))
        # Expanded picks rebuilt with expand, every state checked (the header's budget is the Batch's 0).
        replay(BatchEngine(1, candidate_set=4, hp_only=True, max_candidates=0), head, decisions, result)
    assert expanded > 0  # factor refinement's picks were recorded and replayed
    assert all(_check_searches(p) > 0 for p in paths)  # expansions and cheap decisions mapped onto the lists


def test_pool_eval_shard_children_write_their_games(tmp_path):
    import pool_eval

    out, rec = tmp_path / "s0.jsonl", tmp_path / "records"
    assert pool_eval.main(["--candidate", "matrix-do", "--previous-best", "matrix", "--quick", "--pairs", "3",
                           "--characters", "Toy", "--engine", "toy,toy", "--max-turns", "20", "--procs", "2",
                           "--shard", "0/2", "--shard-out", str(out), "--results", str(tmp_path / "r.jsonl"),
                           "--record-dir", str(rec), "--no-ledger"]) == 0
    games = [json.loads(x) for x in out.read_text().splitlines()[1:]]
    names = sorted(p.name for p in rec.glob("g*.jsonl.gz"))
    assert len(games) == len(names) > 0
    assert sorted((g["seed"], *g["bots"]) for g in games) == sorted(
        (h["seed"], h["bots"][0]["bot"], h["bots"][1]["bot"]) for h in (read_slim(rec / n)[0] for n in names))


def _check_searches(path):
    """Every searched decision of a recorded arena game carries each seat's own search: a mix over
    that seat's list holding its pick, the bot's mix for the other seat, and values whose best
    known entry is the seat's game value (a certified answer leaves no better reply)."""
    head, decisions, _ = load(path)
    searched = 0
    for t in decisions:
        assert t["value"] is None  # the trainer's gate eval reads `value` as a target: kept null
        if t["strategies"] is None:
            continue
        searched += 1
        for s in (0, 1):
            n, m = len(t["candidates"][s]), len(t["candidates"][1 - s])
            mix, opp, cv, v = (t["strategies"][s], t["opponent_strategies"][s], t["candidate_values"][s],
                               t["seat_values"][s])
            assert len(mix) == len(cv) == n and len(opp) == m
            assert abs(sum(mix) - 1) < 1e-3 and abs(sum(opp) - 1) < 1e-3
            assert mix[t["picks"][s]] > 0  # bots sample their own mix
            known = [x for x in cv if x is not None]
            assert known and max(known) <= v + 2e-3
            assert all(cv[i] is not None and abs(cv[i] - v) <= 2e-3 for i, p in enumerate(mix) if p > 0.01)
    return searched


def test_recorded_arena_decisions_carry_each_seats_search(tmp_path):
    _games(tmp_path)
    paths = sorted(tmp_path.glob("g*.jsonl.gz"))
    assert paths and all(_check_searches(p) > 0 for p in paths)


def test_a_seat_without_a_search_records_nulls(tmp_path):
    run_round_robin(lambda: connect("toy"), [make_bot("random"), make_bot("greedy")], ["Toy"], 1, 5,
                    max_turns=10, record_dir=tmp_path)
    for path in tmp_path.glob("g*.jsonl.gz"):
        _, decisions, _ = load(path)
        assert decisions and all(t["strategies"] is None and t["seat_values"] is None for t in decisions)

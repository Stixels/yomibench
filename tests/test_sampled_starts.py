"""Sampled-state starts (yomibench.sampled_starts): selection, the start pool, fast-forward and its check,
the records they write, and training's reading of them, on the toy engine (its `fast_forward` replays
recorded picks). The private batch engine's path is in test_batch_sampled_starts.py."""
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from yomibench import sampled_starts as ss
from yomibench.bots import MatrixBot
from yomibench.engine import Engine, InProcess, connect
from yomibench.selfplay import play_recorded, read_record, run_selfplay
from yomibench.toy import ToyEngine

RULES = "rules-a"
ON = {"share": 1.0, "window_hours": 12.0, "max_records": 1000}


def toy_engine():
    return connect("toy")


def play(out, seeds, pool=None, connects=None, stats=None, **kw):
    return run_selfplay(connects or [toy_engine], lambda: MatrixBot(iterations=20), [("Toy", "Toy")], seeds, out,
                        stats=stats, **({"sampled_starts": pool} if pool is not None else {}), **kw)


def by_seed(folder):
    return {read_record(p)[0]["seed"]: (p, read_record(p)) for p in sorted(Path(folder).glob("g*.jsonl.gz"))}


def rows(lines):
    return [x for x in lines[1:] if "result" not in x]


TIMING = ("started", "ms", "seconds", "perf")


def untimed(x):
    if isinstance(x, dict):
        return {k: untimed(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [untimed(v) for v in x]
    return x


# --- settings, selection, eligibility ------------------------------------------------------------

def test_settings_off_and_validation():
    for off in (None, "", {}, {"share": 0}, '{"share": 0, "window_hours": 5}'):
        assert ss.parse_settings(off) is None
    assert ss.parse_settings({"share": 0.1}) == {"share": 0.1, "window_hours": 12.0, "max_records": 1000}
    assert ss.parse_settings('{"share": 0.1, "window_hours": 6}')["window_hours"] == 6.0
    for bad in ({"share": 1.5}, {"share": -0.1}, {"share": 0.1, "window_hours": 0}, {"share": True},
                {"share": 0.1, "max_records": 0}, {"share": 0.1, "other": 1}, "[1]", "not json"):
        with pytest.raises(ValueError):
            ss.parse_settings(bad)
    assert ss.settings_tag(ss.parse_settings({"share": 0.1})) == "share:0.1,window_hours:12"


def test_selection_is_seeded_by_pair_and_near_the_share():
    picked = [ss.selected(s, 0.1) for s in range(20000)]
    assert all(picked[2 * m] == picked[2 * m + 1] for m in range(10000))  # both games of a pair
    assert 0.08 < sum(picked) / len(picked) < 0.12
    assert picked == [ss.selected(s, 0.1) for s in range(20000)]  # nothing but the seed
    assert not any(ss.selected(s, 0.0) for s in range(100)) and all(ss.selected(s, 1.0) for s in range(100))


def _write(path, header, rows_, result=None):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for line in [header, *rows_, {"result": result or {"outcome": "p1", "winner": 0}}]:
            f.write(json.dumps(line, separators=(",", ":")) + "\n")


def _row(i):
    return {"turn": i, "state": {"tick": i}, "candidates": [[{"action": "A", "data": i, "extra": None}],
                                                            [{"action": "B", "data": i, "extra": None}]],
            "picks": [0, 0]}


HELLO = {"engine": "x", "version": "2", "game_version": "1.0", "candidate_set": "6", "candidate_rules": RULES,
         "match_settings": {"stage_width": 600, "starting_meter": "1.5"}}


def test_only_this_engines_rules_and_settings_from_opening_games_in_the_window_are_eligible(tmp_path):
    hdr = {"seed": 5, "characters": ["Toy", "Toy"], "engine": HELLO}
    _write(tmp_path / "g000000000001-a.jsonl.gz", hdr, [_row(i) for i in range(8)])
    _write(tmp_path / "g000000000002-a.jsonl.gz", {**hdr, "engine": {**HELLO, "candidate_rules": "rules-b"}},
           [_row(i) for i in range(8)])
    _write(tmp_path / "g000000000003-a.jsonl.gz", {**hdr, "engine": {**HELLO, "match_settings": {"stage_width": 800}}},
           [_row(i) for i in range(8)])
    _write(tmp_path / "g000000000004-a.jsonl.gz", {**hdr, "start": {"kind": "sampled", "k": 2}},
           [_row(i) for i in range(8)])
    old = tmp_path / "g000000000005-a.jsonl.gz"
    _write(old, hdr, [_row(i) for i in range(8)])
    os.utime(old, (time.time() - 13 * 3600,) * 2)
    # settings compared by value: "1.5" == 1.5
    _write(tmp_path / "g000000000006-a.jsonl.gz",
           {**hdr, "engine": {**HELLO, "match_settings": {"stage_width": 600, "starting_meter": 1.5}}},
           [_row(i) for i in range(8)])
    pool = ss.StartPool(ON, [tmp_path])
    files = pool.bind(HELLO)
    assert [p.name[:13] for p in files] == ["g000000000001", "g000000000006"]
    assert pool.listing == {"files": 6, "old": 1, "unreadable": 0, "sampled": 1, "other_engine": 2, "eligible": 2}
    # an engine on other candidate rules finds nothing: every slot plays the opening, counted as no_source
    other = ss.StartPool(ON, [tmp_path])
    assert other.bind({**HELLO, "candidate_rules": "rules-c"}) == []
    assert other.pick(0) == (None, "no_source")


def test_k_is_uniform_in_1_to_n_minus_3_deterministic_and_short_sources_are_skipped(tmp_path):
    hdr = {"seed": 5, "characters": ["Toy", "Toy"], "engine": HELLO}
    src = tmp_path / "g000000000001-a.jsonl.gz"
    _write(src, hdr, [_row(i) for i in range(10)])
    pool = ss.StartPool(ON, [tmp_path])
    pool.bind(HELLO)
    ks = set()
    for seed in range(400):
        start, why = pool.pick(seed)
        assert why is None and 1 <= start.k <= 7
        ks.add(start.k)
        assert start.prefix == [[r["candidates"][0][0], r["candidates"][1][0]] for r in [_row(i) for i in range(start.k)]]
        assert start.state == {"tick": start.k} and start.source_seed == 5
        assert start.source_sha256 == hashlib.sha256(src.read_bytes()).hexdigest()
        again = ss.StartPool(ON, [tmp_path])
        again.bind(HELLO)
        assert again.pick(seed)[0] == start  # from the seed alone
        assert pool.pick(seed ^ 1)[0].k == start.k  # the pair's other game: the same start
    assert ks == set(range(1, 8))
    short = tmp_path / "short"
    short.mkdir()
    _write(short / "g000000000009-a.jsonl.gz", hdr, [_row(i) for i in range(4)])
    p = ss.StartPool(ON, [short])
    p.bind(HELLO)
    assert p.pick(0) == (None, "short_source")


def test_prefix_inputs_are_the_picked_candidates_as_inputs():
    row = {"candidates": [[{"action": "Wait", "data": None, "extra": None},
                           {"action": "Shoot", "data": {"x": 1}, "extra": {"DI": {"x": 3, "y": 0}}, "factors": {"di": 17},
                            "values": {"di": 3}, "base": 0}],
                          [{"action": "Block", "data": None, "extra": {"DI": {"x": 0, "y": 0}}}]],
           "picks": [1, 0]}
    assert ss.prefix_inputs([row, row], 1) == [[{"action": "Shoot", "data": {"x": 1}, "extra": {"DI": {"x": 3, "y": 0}}},
                                                {"action": "Block", "data": None, "extra": {"DI": {"x": 0, "y": 0}}}]]


# --- self-play from sampled starts -----------------------------------------------------------------

def _sources(tmp_path, seeds=range(100, 104)):
    src = tmp_path / "pool"
    play(src, seeds)
    return src


def test_off_plays_exactly_as_before(tmp_path):
    play(tmp_path / "a", range(4))
    stats = {}
    play(tmp_path / "b", range(4), pool=ss.StartPool({**ON, "share": 1e-12}, [tmp_path / "a"]), stats=stats)
    a, b = by_seed(tmp_path / "a"), by_seed(tmp_path / "b")
    assert sorted(a) == sorted(b) == [0, 1, 2, 3]
    for s in a:
        assert "start" not in a[s][1][0]
        assert untimed(a[s][1]) == untimed(b[s][1])
    assert "sampled_starts" not in stats


def test_a_sampled_start_record_is_self_contained_and_replays_to_its_first_state(tmp_path):
    src = _sources(tmp_path)
    sources = {lines[0]["seed"]: (p, lines) for p, lines in by_seed(src).values()}
    stats = {}
    play(tmp_path / "out", range(4), pool=ss.StartPool(ON, [src]), stats=stats)
    assert stats["sampled_starts"] == {"slots": 4, "games": 4, "no_source": 0, "short_source": 0, "mismatch": 0,
                                       "error": 0}
    games = by_seed(tmp_path / "out")
    for seed, (path, lines) in games.items():
        h, own = lines[0], rows(lines)
        st = h["start"]
        spath, slines = sources[st["source_seed"]]
        srows = rows(slines)
        assert set(st) == {"kind", "source_seed", "source_sha256", "k", "prefix_inputs"} and st["kind"] == "sampled"
        assert 1 <= st["k"] <= len(srows) - 3
        assert st["source_sha256"] == hashlib.sha256(spath.read_bytes()).hexdigest()
        assert st["prefix_inputs"] == [[r["candidates"][s][r["picks"][s]] for s in (0, 1)] for r in srows[:st["k"]]]
        assert h["characters"] == slines[0]["characters"] and h["seed"] == seed
        # the game's rows begin at decision k: its first state is the source's decision-k state
        assert ss.canon_json(own[0]["state"]) == ss.canon_json(srows[st["k"]]["state"])
        assert own[0]["turn"] == srows[st["k"]]["turn"] == st["k"]  # the toy's turn is its decision count
        # the record alone replays it: a new match on the source seed, then fast_forward(prefix_inputs)
        e = connect("toy")
        e.new_match(*h["characters"], st["source_seed"])
        assert ss.canon_json(e.fast_forward(st["prefix_inputs"]).state) == ss.canon_json(own[0]["state"])
    for a, b in ((0, 1), (2, 3)):  # pairs: one start, two games
        sa, sb = games[a][1][0]["start"], games[b][1][0]["start"]
        assert (sa["source_seed"], sa["k"]) == (sb["source_seed"], sb["k"])


def test_the_pairs_games_differ_in_their_exploration_seeds(tmp_path):
    src = _sources(tmp_path)
    pool = ss.StartPool(ON, [src])
    engine = toy_engine()
    pool.bind(engine.hello())
    starts = [pool.pick(s)[0] for s in (6, 7)]
    assert starts[0] == starts[1]
    seen = []
    for seed in (6, 7):
        obs, why = ss.begin(engine, starts[0], seed)
        assert why is None and engine.match_seed == seed  # the bot's per-match randomness keys on the game
        seen.append(ss.canon_json(obs.state))
    assert seen[0] == seen[1]  # the same state
    out = tmp_path / "explore"  # with exploration on every decision, the continuations differ
    play(out, range(6, 8), pool=ss.StartPool(ON, [src]), explore_cfg=(1.0, 5.0))
    g = by_seed(out)
    assert g[6][1][0]["start"]["k"] == g[7][1][0]["start"]["k"]
    assert [r["picks"] for r in rows(g[6][1])] != [r["picks"] for r in rows(g[7][1])]


class ShortReplay(InProcess):
    """A toy engine whose fast_forward plays one input short: the state reached isn't the record's."""
    def request(self, msg):
        if msg.get("cmd") == "fast_forward":
            msg = {**msg, "inputs": msg["inputs"][:-1]}
        return super().request(msg)


class NoFastForward(InProcess):
    def request(self, msg):
        if msg.get("cmd") == "fast_forward":
            return {"error": {"code": "unknown_cmd", "message": "'fast_forward'"}}
        return super().request(msg)


@pytest.mark.parametrize("transport, reason", [(ShortReplay, "mismatch"), (NoFastForward, "error")])
def test_a_failed_start_plays_the_slots_normal_game(tmp_path, transport, reason):
    src = _sources(tmp_path)
    stats = {}
    play(tmp_path / "out", range(2), pool=ss.StartPool(ON, [src]), stats=stats,
         connects=[lambda: Engine(transport(ToyEngine()))])
    assert stats["sampled_starts"][reason] == 2 and stats["sampled_starts"]["games"] == 0
    play(tmp_path / "plain", range(2))
    got, want = by_seed(tmp_path / "out"), by_seed(tmp_path / "plain")
    for s in (0, 1):  # exactly the game the slot plays without the flag
        assert "start" not in got[s][1][0]
        assert untimed(got[s][1]) == untimed(want[s][1])


def test_a_sampled_start_has_no_opening_window(tmp_path):
    src = _sources(tmp_path)
    pool = ss.StartPool(ON, [src])
    engine = toy_engine()
    pool.bind(engine.hello())
    start = pool.pick(0)[0]
    obs, _ = ss.begin(engine, start, 0)
    lines = play_recorded(engine, MatrixBot(iterations=20), start.characters, 0, engine.hello(),
                          opening=(3, 1.0, 1.0), sampled_start=(start, obs))
    assert lines[0]["opening"]["turns"] == 3 and lines[0]["start"]["k"] == start.k
    assert not any(r.get("opening") for r in rows(lines))


# --- training reads them -------------------------------------------------------------------------

def test_select_rows_phases_use_the_true_decision_index():
    from yomibench.selfplay import SlimTurn
    from yomibench.train import select_rows

    turns = [SlimTurn({}, 0.1, (False, False)) for _ in range(10)]
    plain, _ = select_rows(turns, "train", "uniform-only", 0.5, 1.0)
    assert [p for _, _, p in plain] == [k / 10 for k in range(10)]
    shifted, counts = select_rows(turns, "train", "uniform-only", 0.5, 1.0, offset=30)
    assert [p for _, _, p in shifted] == [(30 + k) / 40 for k in range(10)]  # all late in a 40-decision game
    assert counts["decisions"] == [0, 0, 10]
    assert [y for _, y, _ in shifted] == [y for _, y, _ in plain]  # targets don't depend on it


def test_training_reads_sampled_start_records_with_true_phases_and_can_exclude_them(tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    from yomibench import cache, train
    from yomibench.selfplay import read_header

    src = _sources(tmp_path)
    data = tmp_path / "data"
    play(data, range(4), pool=ss.StartPool(ON, [src]))
    play(data, range(10, 12))  # two games from the opening
    headers = {p.name: read_header(p) for p in data.glob("g*.jsonl.gz")}
    assert sum(ss.is_sampled(h) for h in headers.values()) == 4
    kw = dict(workers=1, heldout_root=tmp_path / "h")
    direct = train.load_samples([data], train.Vocab(), 0.3, 0, 0.5, **kw)
    cached = cache.open_cached([data], train.Vocab(), 0.3, 0, 0.5, root=tmp_path / "c", **kw)
    assert direct["train"]["games"] + direct["val"]["games"] == 6 == cached["train"]["games"] + cached["val"]["games"]
    for split in ("train", "val"):
        np.testing.assert_array_equal(np.asarray(cached[split]["phase"]), direct[split]["phase"])
    phases = np.concatenate([direct["train"]["phase"], direct["val"]["phase"]])
    want = set()
    for name, h in headers.items():
        n, k = len(rows(read_record(data / name))), ss.start_offset(h)
        want |= {round((k + r) / (k + n), 9) for r in range(n)}
    assert {round(float(p), 9) for p in phases} <= want
    assert int((phases == 0).sum()) == 2  # only the opening games have a row at phase 0
    # excluded (a holdout of games from the opening): only those remain, in every split
    ex = train.load_samples([data], train.Vocab(), 0.3, 0, 0.5, exclude_sampled_starts=True, **kw)
    assert ex["train"]["games"] + ex["val"]["games"] == 2 and ex["sampled_starts_excluded"]["excluded_total"] == 4
    exc = cache.open_cached([data], train.Vocab(), 0.3, 0, 0.5, root=tmp_path / "c", exclude_sampled_starts=True, **kw)
    assert exc["train"]["games"] + exc["val"]["games"] == 2
    rf = cache.open_cached([data], train.Vocab(), 0.3, 0, 0.5, root=tmp_path / "c",
                           record_filter={"exclude_sampled_starts": True, "match_settings": {}}, **kw)
    assert rf["sampled_starts_excluded"]["excluded_total"] == 4
    assert rf["train"]["games"] + rf["val"]["games"] == 2
    # shared windows (train --share-window): a window read back gives the same phases; excluding sampled
    # starts selects other rows, so it is another window (its own key), never the shared full one
    share = tmp_path / "share"
    wrote = cache.open_cached([data], train.Vocab(), 0.3, 0, 0.5, root=tmp_path / "c", share_dir=share, **kw)
    read = cache.open_cached([data], train.Vocab(), 0.3, 0, 0.5, root=tmp_path / "c", share_dir=share, **kw)
    assert (wrote["shared_window"]["mode"], read["shared_window"]["mode"]) == ("written", "exact")
    for split in ("train", "val"):
        np.testing.assert_array_equal(np.asarray(read[split]["phase"]), direct[split]["phase"])
    ex_w = cache.open_cached([data], train.Vocab(), 0.3, 0, 0.5, root=tmp_path / "c", share_dir=share,
                             exclude_sampled_starts=True, **kw)
    assert ex_w["shared_window"]["mode"] == "written" and ex_w["shared_window"]["dir"] != wrote["shared_window"]["dir"]
    assert ex_w["train"]["games"] + ex_w["val"]["games"] == 2
    ex_r = cache.open_cached([data], train.Vocab(), 0.3, 0, 0.5, root=tmp_path / "c", share_dir=share,
                             exclude_sampled_starts=True, **kw)
    assert ex_r["shared_window"]["mode"] == "exact" and ex_r["sampled_starts_excluded"]["excluded_total"] == 4

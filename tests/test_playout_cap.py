"""Playout cap randomization: seeded cheap/full split, capped cheap searches, outcome-only targets."""

import gzip
import json

import pytest

from yomibench.bots import FactorExplore, MatrixBot
from yomibench.engine import Candidate, connect
from yomibench.selfplay import SlimTurn, cap_candidates, cheap_turn, play_recorded, read_slim, write_record


def _game(seed, playout_cap=None, max_turns=40):
    engine = connect("toy")
    return play_recorded(engine, MatrixBot(iterations=50, search="oracle"), ("Toy", "Toy"), seed, engine.hello(),
                         max_turns=max_turns, playout_cap=playout_cap)


def _decisions(lines):
    return lines[1:-1]


def test_the_cheap_full_split_is_a_function_of_seed_and_turn_alone():
    assert [cheap_turn(7, t, 0.75) for t in range(200)] == [cheap_turn(7, t, 0.75) for t in range(200)]
    assert [cheap_turn(7, t, 0.75) for t in range(200)] != [cheap_turn(8, t, 0.75) for t in range(200)]
    share = sum(cheap_turn(s, t, 0.75) for s in range(20) for t in range(200)) / 4000
    assert share == pytest.approx(0.75, abs=0.03)
    assert not any(cheap_turn(3, t, 0.0) for t in range(200)) and all(cheap_turn(3, t, 1.0) for t in range(200))

    a, b = _game(5, (0.5, 32)), _game(5, (0.5, 32))
    assert [ln["search"] for ln in _decisions(a)] == [ln["search"] for ln in _decisions(b)]
    assert [ln["picks"] for ln in _decisions(a)] == [ln["picks"] for ln in _decisions(b)]
    marks = [ln["search"] for ln in _decisions(a)]
    assert marks == ["cheap" if cheap_turn(5, ln["turn"], 0.5) else "full" for ln in _decisions(a)]
    assert set(marks) == {"cheap", "full"}
    assert a[0]["playout_cap"] == {"cheap_fraction": 0.5, "cheap_cap": 32}
    perf = a[-1]["result"]["perf"]
    assert perf["cheap_searched"] == sum(ln["search"] == "cheap" and ln["strategies"] is not None
                                         for ln in _decisions(a))
    assert perf["cheap_previews"] == sum(ln["previews"] for ln in _decisions(a) if ln["search"] == "cheap")


def test_without_a_cap_records_are_as_before():
    plain, off = _game(5), _game(5, None)
    assert plain[0]["playout_cap"] is None
    assert all("search" not in ln for ln in _decisions(plain))
    strip = lambda lines: [{k: v for k, v in ln.items() if k not in ("ms", "started")} for ln in lines[:-1]]  # noqa: E731
    assert strip(plain) == strip(off)


def test_cheap_decisions_skip_softmax_and_opening_exploration():
    engine = connect("toy")
    lines = play_recorded(engine, MatrixBot(iterations=50, search="oracle"), ("Toy", "Toy"), 11, engine.hello(),
                          max_turns=60, playout_cap=(0.5, 32), opening=(50, 1.0, 1.0), explore_cfg=(1.0, 0.05))
    cheap = [ln for ln in _decisions(lines) if ln["search"] == "cheap" and ln["strategies"] is not None]
    full = [ln for ln in _decisions(lines) if ln["search"] == "full" and ln["strategies"] is not None]
    assert cheap and full
    assert not any(ln.get("explore") or ln.get("opening") for ln in cheap)
    assert all(any(ln.get("explore", [])) for ln in full)  # epsilon 1: every full decision explores


def test_the_cheap_bot_searches_without_refinement_or_factor_exploration():
    bot = MatrixBot(search="oracle", explore=FactorExplore(margin=0.02, random_k=2, all_prob=0.005))
    cheap = bot.cheap()
    assert (cheap.refine_rounds, cheap.explore) == (0, None)
    assert (bot.refine_rounds, bot.explore) == (2, FactorExplore(margin=0.02, random_k=2, all_prob=0.005))
    assert cheap.config()["refine_rounds"] == 0 and cheap._key() != bot._key()


def _c(i, action):
    return Candidate(i, action, None, None)


def test_cap_candidates_follows_the_engine_budget():
    # Continue with 2 variants, then moves A (4 variants), B (1), C (3): 10 candidates.
    cands = [_c(0, "Continue"), _c(1, "Continue"), *[_c(i, "A") for i in range(2, 6)], _c(6, "B"),
             *[_c(i, "C") for i in range(7, 10)]]
    ids = lambda cs: [c.id for c in cs]  # noqa: E731
    assert ids(cap_candidates(cands, 0)) == list(range(10))  # 0: no cap, as max_candidates
    assert ids(cap_candidates(cands, 10)) == list(range(10))
    # Every Continue variant and each move's first variant, even over the cap.
    assert ids(cap_candidates(cands, 2)) == [0, 1, 2, 6, 7]
    assert ids(cap_candidates(cands, 5)) == [0, 1, 2, 6, 7]
    # Then one more variant per move per pass, in list order; kept candidates stay in list order.
    assert ids(cap_candidates(cands, 6)) == [0, 1, 2, 3, 6, 7]
    assert ids(cap_candidates(cands, 7)) == [0, 1, 2, 3, 6, 7, 8]
    assert ids(cap_candidates(cands, 8)) == [0, 1, 2, 3, 4, 6, 7, 8]
    assert ids(cap_candidates(cands, 9)) == [0, 1, 2, 3, 4, 6, 7, 8, 9]


def test_a_cheap_decision_records_and_plays_the_capped_lists(monkeypatch):
    from yomibench import selfplay

    seen = []

    def cap(cands, n, *_):  # the toy's lists are small: cap them to their first two candidates
        out = list(cands)[:2]
        seen.append(out)
        return out

    monkeypatch.setattr(selfplay, "cap_candidates", cap)
    lines = _game(5, (0.5, 2))
    cheap = [ln for ln in _decisions(lines) if ln["search"] == "cheap"]
    assert cheap and all(len(seat) <= 2 for ln in cheap for seat in ln["candidates"])
    for ln in cheap:
        if ln["strategies"] is not None:
            assert [len(s) for s in ln["strategies"]] == [len(c) for c in ln["candidates"]]
            assert all(p < len(c) for p, c in zip(ln["picks"], ln["candidates"]))
    assert any(max(len(seat) for seat in ln["candidates"]) > 2 for ln in _decisions(lines) if ln["search"] == "full")


def _blend_targets(path, value_weight):
    from yomibench.train import select_rows

    _, result, turns = read_slim(path)
    w = result["result"]["winner"]
    z = 0.0 if w is None else (1.0 if w == 0 else -1.0)
    rows, _ = select_rows(turns, "train", "none", value_weight, z)
    return turns, z, {k: y for k, y, _ in rows}


def test_cheap_turns_train_on_the_outcome_alone_and_old_records_are_unchanged(tmp_path):
    capped = write_record(tmp_path, 5, _game(5, (0.5, 32)))
    turns, z, ys = _blend_targets(capped, 0.5)
    assert {t.cheap for t in turns} == {True, False}
    for k, t in enumerate(turns):
        expected = z if t.cheap or t.value is None else 0.5 * z + 0.5 * t.value
        assert ys[k] == pytest.approx(expected)

    # A record without marks (every record before playout caps) reads as all full: the value mixes
    # in everywhere, exactly as before; the same record with every turn marked full trains the same.
    old = write_record(tmp_path, 6, _game(6))
    turns, z, ys = _blend_targets(old, 0.5)
    assert not any(t.cheap for t in turns)
    assert all(ys[k] == pytest.approx(z if t.value is None else 0.5 * z + 0.5 * t.value) for k, t in enumerate(turns))
    marked = tmp_path / "marked" / old.name
    marked.parent.mkdir()
    with gzip.open(old, "rt") as f, gzip.open(marked, "wt") as g:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i and "result" not in d:
                d["search"] = "full"
            g.write(json.dumps(d) + "\n")
    assert _blend_targets(marked, 0.5)[2] == ys


def test_value_only_stretches_skip_cheap_turns():
    from yomibench.train import select_rows

    # Turns 0-2 sit at or before the last uniform pick (turn 2): value-only targets, and the cheap one
    # (turn 1) has none, so it is skipped. After the cut: blend on full turns, outcome on cheap ones.
    turns = [SlimTurn({}, 0.4, (False, False), None, None), SlimTurn({}, 0.3, (False, False), None, None, True),
             SlimTurn({}, 0.2, (True, False), None, "uniform"), SlimTurn({}, 0.1, (False, False), None, None),
             SlimTurn({}, 0.6, (False, False), None, None, True)]
    rows, counts = select_rows(turns, "train", "uniform-only", 0.5, 1.0)
    assert [(k, y) for k, y, _ in rows] == [(0, 0.4), (2, 0.2), (3, 0.55), (4, 1.0)]
    assert counts["explored"] == 3 and counts["value_only"] == 2


def test_the_cache_keeps_the_marks_and_trains_like_the_in_memory_path(tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    from yomibench import cache, train
    from yomibench.selfplay import run_selfplay

    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(8), tmp_path,
                 playout_cap=(0.5, 32))
    direct = train.load_samples([tmp_path], train.Vocab(), 0.25, 0, 0.5, workers=1)
    cached = cache.open_cached([tmp_path], train.Vocab(), 0.25, 0, 0.5, workers=1)
    manifest = json.loads((tmp_path / ".cache" / "manifest.json").read_text())
    marks = np.concatenate([np.load(tmp_path / ".cache" / s["name"] / "cheap.npy") for s in manifest["shards"]])
    assert set(marks.tolist()) == {0, 1}
    for name in ("train", "val"):
        for field in ("y", "outcome"):
            np.testing.assert_array_equal(np.asarray(cached[name][field]), direct[name][field])
    # Full turns still blend. Checked over both splits: record names hash wall-clock fields, so which
    # of the 8 games land in val varies run to run, and train alone can lack a blended turn.
    assert any((direct[n]["y"] != direct[n]["outcome"]).any() for n in ("train", "val"))


# --- the bot form (make_bot "@cheap=P:N"), for arenas and gates -----------------------------------

def test_the_cheap_bot_spec_parses():
    from yomibench.bots import make_bot

    bot = make_bot("matrix-do@cheap=0.75:32")
    assert isinstance(bot, MatrixBot) and bot.mode == "oracle" and bot.playout_cap == (0.75, 32)
    assert bot.name == "matrix-do@cheap=0.75:32" and bot.config()["playout_cap"] == [0.75, 32]
    both = make_bot("matrix@4@cheap=0.5")
    assert (both.candidate_set, both.playout_cap, both.name) == (4, (0.5, 32), "matrix@4@cheap=0.5:32")
    plain = make_bot("matrix-do")
    assert plain.playout_cap is None and "playout_cap" not in plain.config()  # existing configs unchanged
    assert plain._key() == bot._key()  # full decisions share the plain bot's search
    for bad in ("matrix-do@cheap=1.5:32", "matrix-do@cheap=0.5:0", "random@cheap=0.5:32"):
        with pytest.raises(ValueError):
            make_bot(bad)


def _arena_turns(seed, specs, monkeypatch, max_turns=40):
    """An arena game (arena.play_game's loop), returning per turn: the engine's lists, whether the
    turn is cheap for P=0.5, each seat's chosen id, the shared candidate lists, and the cheap view."""
    from yomibench import bots as bots_mod
    from yomibench.bots import make_bot

    monkeypatch.setattr(bots_mod, "cap_candidates", lambda cands, n, *_: list(cands)[:n])  # toy lists are short
    engine = connect("toy")
    bots = [make_bot(s) for s in specs]
    rngs = (__import__("random").Random(f"{seed}:0"), __import__("random").Random(f"{seed}:1"))
    obs = engine.new_match("Toy", "Toy", seed)
    out = []
    while not obs.done and obs.turn < max_turns:
        a, b = bots[0].choose(obs, 0, engine, rngs[0]), bots[1].choose(obs, 1, engine, rngs[1])
        views = [v for k, v in engine.turn_cache.items() if isinstance(k, tuple) and k[0] == "cheap view"]
        out.append(dict(engine=[[c.id for c in s] for s in obs.candidates], cheap=cheap_turn(seed, obs.turn, 0.5),
                        ids=(a, b), shared=[[c.id for c in s] for s in engine.turn_cache.get("candidates", obs.candidates)],
                        views=views))
        obs = engine.step(a, b)
    return out


def test_a_cheap_bot_in_an_arena_caps_only_its_own_decisions(monkeypatch):
    turns = _arena_turns(3, ("matrix-do@cheap=0.5:2", "matrix-do"), monkeypatch)
    assert {t["cheap"] for t in turns} == {True, False}
    for t in turns:
        assert t["shared"] == t["engine"]  # the opponent's lists are never capped
        if not t["cheap"]:
            assert t["views"] == []  # a full decision: exactly the plain bot
            continue
        (view,) = t["views"]  # the cheap decision is marked by this bot's own view, once per decision
        capped = [[c.id for c in s] for s in view["candidates"]]
        assert capped == [s[:2] for s in t["engine"]]
        assert t["ids"][0] in capped[0]  # the cheap bot plays from its capped list
        row, col, _ = view["result"]
        assert [len(row), len(col)] == [len(s) for s in capped]
    # The full bot still chooses over every candidate: somewhere it picks one the cap would drop.
    assert any(t["ids"][1] not in t["engine"][1][:2] for t in turns if t["cheap"])


def test_a_zero_fraction_cheap_bot_plays_the_plain_bots_games(monkeypatch):
    plain = _arena_turns(4, ("matrix-do", "matrix"), monkeypatch)
    off = _arena_turns(4, ("matrix-do@cheap=0:32", "matrix"), monkeypatch)
    assert [t["ids"] for t in plain] == [t["ids"] for t in off]


@pytest.mark.parametrize("specs, golden", [
    (("matrix-do", "matrix-do"), ["b58b9da4e490", "86226bf2ad4f", "8fc6271e72ee"]),
    (("matrix", "matrix-do"), ["a7f7751b7b0b", "b9392072824e", "0ee3b86a8dd7"]),
])
def test_existing_arena_games_are_unchanged(specs, golden):
    """Golden picks of toy arena games, recorded on main before the bot form existed."""
    import hashlib

    from yomibench.arena import play_game
    from yomibench.bots import make_bot

    got = []
    for seed in (1, 2, 3):
        engine, picks = connect("toy"), []
        step = engine.step
        engine.step = lambda a, b, step=step: (picks.append((a, b)), step(a, b))[1]
        r = play_game(engine, tuple(make_bot(s) for s in specs), ("Toy", "Toy"), seed, max_turns=60)
        assert (r.outcome, r.turns, r.hp) == ("truncated", 60, (100, 100))
        got.append(hashlib.sha1(json.dumps(picks).encode()).hexdigest()[:12])
    assert got == golden


# --- cap policies (cheap_cap_policy) ----------------------------------------------------------------

def _seat_list():
    """A realistic seat list: two Continue variants; aimed moves (top-level XY), a move with a checkbox
    and an aim (combos first, as the engine lists them), a parry (block height x timing slider), a
    dropdown, a checkbox-only move listed last (TimeBullet's shape), and single-variant moves."""
    out = [Candidate(0, "Continue", None, {"di": {"x": 0, "y": 0}}), Candidate(1, "Continue", None, {"di": {"x": 50, "y": 0}})]

    def add(action, data, extra=None):
        out.append(Candidate(len(out), action, data, extra))
    for k in range(6):
        add("Jump", {"x": 10 * k, "y": -50})
    for holster in (False, True):
        for x in (100, 70, 0):
            add("Shoot", {"x": x, "y": 0, "holster": holster})
    for low in (0, 1):
        for t in (1, 7, 13, 20):
            add("Parry", {"Block Height": {"x": 0, "y": low}, "Melee Parry Timing": {"count": t}})
    for i in range(3):
        add("Slash", {"Attack": {"id": i, "name": str(i)}, "Dir": {"x": 1, "y": 0}})
    for d in ((1, 0), (-1, 0), (0, 1)):
        add("Grab", {"Direction": {"x": d[0], "y": d[1]}, "Dash": False})
    for a in ("Wait", "Block", "Taunt"):
        add(a, None)
    add("Bullet", False)
    add("Bullet", True)
    return out


def test_option_keys_split_mode_widgets_from_aims():
    from yomibench.playout import option_keys

    cands = _seat_list()
    keys = option_keys(cands)
    by = lambda a: [keys[i] for i, c in enumerate(cands) if c.action == a]  # noqa: E731
    assert len(set(by("Jump"))) == 1 and len(set(by("Grab"))) == 1  # top-level XY, 8-way: aims
    assert len(set(by("Shoot"))) == 2 and len(set(by("Parry"))) == 2  # checkbox; block height (not the slider)
    assert len(set(by("Slash"))) == 3 and len(set(by("Bullet"))) == 2  # dropdown; a bare checkbox
    assert len(set(by("Continue"))) == 1  # DI extras are directions
    assert by("Shoot")[:3] == [by("Shoot")[0]] * 3 and by("Shoot")[3] != by("Shoot")[0]


def test_the_default_policy_is_todays_rule():
    from yomibench.playout import CAP_POLICIES

    assert CAP_POLICIES[0] == "list"
    cands = _seat_list()
    for cap in (0, 5, 15, 16, 20, 25, len(cands)):
        assert cap_candidates(cands, cap) == cap_candidates(cands, cap, "list", "7:3:1")
    # Pinned: Continue x2 and each move's first variant, then one per move per pass in list order.
    assert [c.id for c in cap_candidates(cands, 16)] == [0, 1, 2, 3, 8, 9, 14, 15, 22, 23, 25, 26, 28, 29, 30, 31]
    with pytest.raises(ValueError):
        cap_candidates(cands, 16, "nope")


@pytest.mark.parametrize("policy", ["options-first", "rotate", "options-rotate", "options-shuffle"])
def test_every_policy_keeps_the_same_count_all_continues_and_every_move(policy):
    cands = _seat_list()
    moves = {c.action for c in cands}
    for cap in (1, 5, 13, 14, 16, 20, 25, 30, len(cands) - 1):
        today = cap_candidates(cands, cap)
        for salt in ("1:0:0", "1:1:0", "9:4:1"):
            got = cap_candidates(cands, cap, policy, salt)
            assert len(got) == len(today)
            assert [c.id for c in got] == sorted(c.id for c in got)  # list order kept
            assert {c.id for c in got if c.action == "Continue"} == {0, 1}
            assert {c.action for c in got} == moves
            assert got == cap_candidates(cands, cap, policy, salt)  # deterministic
    assert cap_candidates(cands, len(cands), policy, "x") == cands


def test_options_first_offers_every_option_before_a_second_aim():
    from yomibench.playout import option_keys

    cands = _seat_list()
    keys = option_keys(cands)
    # 11 always kept (2 Continue + 9 moves); 5 more options exist (Shoot, Parry, Bullet 1 each, Slash 2).
    got = cap_candidates(cands, 15, "options-first")
    offered = {(c.action, keys[c.id]) for c in got}
    assert offered == {(c.action, k) for c, k in zip(cands, keys)} - {("Slash", keys[cands.index(
        next(c for c in cands if c.action == "Slash" and c.data["Attack"]["id"] == 2))])}  # 4 slots: one per move per pass
    assert sum(c.action == "Jump" for c in got) == 1  # no second aim while an option was waiting
    assert any(c.action == "Bullet" and c.data is True for c in got)  # the last-listed move's other option
    assert not any(c.action == "Bullet" and c.data is True for c in cap_candidates(cands, 15))  # today: cut
    more = cap_candidates(cands, 20, "options-first")
    assert {(c.action, keys[c.id]) for c in more} == {(c.action, k) for c, k in zip(cands, keys)}
    assert sum(c.action == "Jump" for c in more) == 2  # once every option is in, list's passes go on


@pytest.mark.parametrize("policy, unit", [("rotate", "variant"), ("options-rotate", "option")])
def test_rotations_cover_every_variant_or_option_over_decisions(policy, unit):
    from yomibench.playout import option_keys

    cands = _seat_list()
    keys = option_keys(cands)
    cap = 11  # no free slot: only the rotation of the kept variant can reach the others
    seen = set()
    for turn in range(200):
        got = cap_candidates(cands, cap, policy, f"5:{turn}:0")
        seen |= {c.id if unit == "variant" else (c.action, keys[c.id]) for c in got}
    want = {c.id for c in cands} if unit == "variant" else {(c.action, k) for c, k in zip(cands, keys)}
    assert seen == want
    assert len({tuple(c.id for c in cap_candidates(cands, 13, policy, f"5:{t}:0")) for t in range(20)}) > 1
    assert [c.id for c in cap_candidates(cands, cap, "list")] == [0, 1, 2, 8, 14, 22, 25, 28, 29, 30, 31]


def test_options_shuffle_keeps_options_firsts_options_and_varies_only_their_variants():
    from yomibench.playout import option_keys

    cands = _seat_list()
    keys = option_keys(cands)
    opts = lambda cs: {(c.action, keys[c.id]) for c in cs}  # noqa: E731
    for cap in (11, 13, 15, 16, 20, 25, 30):
        first = cap_candidates(cands, cap, "options-first")
        for t in range(30):
            got = cap_candidates(cands, cap, "options-shuffle", f"5:{t}:0")
            assert opts(got) == opts(first) and len(got) == len(first)  # the same options offered
    # With no free slot (one variant per move), the seeded shuffle reaches every variant of each move's
    # first option (every Jump aim, every high-Parry timing) over decisions; options-first always
    # shows the first variant.
    first_opt = {}
    for c in cands:
        first_opt.setdefault(c.action, keys[c.id])
    seen = set()
    for t in range(300):
        seen |= {c.id for c in cap_candidates(cands, 11, "options-shuffle", f"5:{t}:0")}
    assert seen == {c.id for c in cands if keys[c.id] == first_opt[c.action]}
    jumps = {c.id for t in range(300) for c in cap_candidates(cands, 11, "options-first", f"5:{t}:0") if c.action == "Jump"}
    assert len(jumps) == 1
    # A fixed function of the salt: the same decision gives the same list, another decision may not.
    assert cap_candidates(cands, 13, "options-shuffle", "5:3:1") == cap_candidates(cands, 13, "options-shuffle", "5:3:1")
    assert len({tuple(c.id for c in cap_candidates(cands, 13, "options-shuffle", f"5:{t}:0")) for t in range(20)}) > 1


def test_a_cap_policy_is_recorded_and_salted_by_seed_turn_and_seat(monkeypatch):
    from yomibench import selfplay

    calls = []

    def cap(cands, n, policy="list", salt=""):
        calls.append((policy, salt))
        return list(cands)[:n]

    monkeypatch.setattr(selfplay, "cap_candidates", cap)
    lines = _game(5, (0.5, 2, "options-first"))
    assert lines[0]["playout_cap"] == {"cheap_fraction": 0.5, "cheap_cap": 2, "cheap_cap_policy": "options-first"}
    cheap = [ln["turn"] for ln in _decisions(lines) if ln["search"] == "cheap"]
    assert calls == [("options-first", f"5:{t}:{s}") for t in cheap for s in (0, 1)]
    calls.clear()
    assert _game(5, (0.5, 2))[0]["playout_cap"] == {"cheap_fraction": 0.5, "cheap_cap": 2}
    assert {p for p, _ in calls} == {"list"}


def test_the_cheap_bot_spec_takes_a_cap_policy(monkeypatch):
    from yomibench import bots as bots_mod
    from yomibench.bots import make_bot

    bot = make_bot("matrix-do@cheap=0.75:32:options-first")
    assert bot.playout_cap == (0.75, 32, "options-first") and bot.name == "matrix-do@cheap=0.75:32:options-first"
    assert bot.config()["playout_cap"] == [0.75, 32, "options-first"]
    same = make_bot("matrix-do@cheap=0.75:32:list")  # the default policy is today's bot exactly
    assert same.playout_cap == (0.75, 32) and same.name == "matrix-do@cheap=0.75:32"
    with pytest.raises(ValueError):
        make_bot("matrix-do@cheap=0.75:32:nope")
    # The bot caps its own decisions with the policy, salted as self-play salts them.
    calls = []
    monkeypatch.setattr(bots_mod, "cap_candidates",
                        lambda cands, n, policy="list", salt="": (calls.append((policy, salt)), list(cands)[:n])[1])
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 3)
    turn = next(t for t in range(200) if cheap_turn(3, t, 0.75))
    while obs.turn < turn:
        obs = engine.step(obs.candidates[0][0].id, obs.candidates[1][0].id)
    bot.choose(obs, 0, engine, __import__("random").Random(0))
    assert calls == [("options-first", f"3:{turn}:0"), ("options-first", f"3:{turn}:1")]


def test_options_shuffle_games_are_deterministic_and_their_picks_step_as_previewed():
    from yomibench.playout import cap_candidates as cap

    strip = lambda lines: [{k: v for k, v in ln.items() if k not in ("ms", "started", "perf")} for ln in lines]  # noqa: E731
    a, b = _game(5, (0.5, 3, "options-shuffle")), _game(5, (0.5, 3, "options-shuffle"))
    assert strip(a[:-1]) == strip(b[:-1])  # a fixed function of the seed: reruns are identical
    # Replay the record: each cheap decision's lists are the policy's cap of the engine's lists (salted
    # seed:turn:seat), and the picked pair previews to exactly the state stepping it reaches.
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 5)
    cheap = 0
    for ln in _decisions(a):
        assert ln["turn"] == obs.turn
        lists = [cap(obs.candidates[s], 3, "options-shuffle", f"5:{obs.turn}:{s}") if ln["search"] == "cheap"
                 else obs.candidates[s] for s in (0, 1)]
        if ln["search"] == "cheap":
            cheap += 1
            assert [[(c.action, c.data, c.extra) for c in x] for x in lists] == \
                [[(c["action"], c.get("data"), c.get("extra")) for c in x] for x in ln["candidates"]]
        ids = [lists[s][ln["picks"][s]].id for s in (0, 1)] if ln["search"] == "cheap" else None
        if ids is None:  # a full decision: picks index the record's lists (base first, then expansions)
            ids = [next(c.id for c in obs.candidates[s] if (c.action, c.data, c.extra) == (
                ln["candidates"][s][ln["picks"][s]]["action"], ln["candidates"][s][ln["picks"][s]].get("data"),
                ln["candidates"][s][ln["picks"][s]].get("extra"))) for s in (0, 1)]
        (seen,) = engine.preview([tuple(ids)], candidates=False)
        obs = engine.step(*ids)
        assert obs.state == seen.state
    assert cheap > 3


def test_a_game_with_the_policy_off_is_bit_identical_to_todays():
    strip = lambda lines: [{k: v for k, v in ln.items() if k not in ("ms", "started", "perf")} for ln in lines]  # noqa: E731
    assert strip(_game(5, (0.5, 3))[:-1]) == strip(_game(5, (0.5, 3, "list"))[:-1])


def test_frozen_keys_equal_exactly_when_json_strings_do():
    """The cap policies' option keys (`_frozen`) group values exactly as their sorted-key JSON strings do:
    True / 1 / 1.0 and -0.0 / 0.0 stay apart, NaN equals NaN, dict order doesn't count."""
    import random as _random

    from yomibench.playout import _frozen

    rng = _random.Random(7)
    atoms = [None, True, False, 0, 1, -1, 0.0, -0.0, 1.0, 0.5, float("nan"), "", "a", "1", "true", "x\u00e9"]

    def value(depth=0):
        r = rng.random()
        if depth < 3 and r < 0.2:
            return [value(depth + 1) for _ in range(rng.randrange(3))]
        if depth < 3 and r < 0.4:
            ks = rng.sample(["x", "y", "id", "name", "count"], rng.randrange(4))
            return {k: value(depth + 1) for k in ks}
        return rng.choice(atoms)
    vals = [value() for _ in range(3000)]
    for _ in range(40000):
        a, b = rng.choice(vals), rng.choice(vals)
        assert (_frozen(a) == _frozen(b)) == (json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)), (a, b)
    assert _frozen({"x": 1, "y": 2}) == _frozen({"y": 2, "x": 1})


def test_option_tuples_group_as_option_keys_strings():
    from yomibench.playout import _option_tuples, option_keys

    cands = _seat_list() + [Candidate(40, "Flag", {"on": True}, None), Candidate(41, "Flag", {"on": "true"}, None),
                            Candidate(42, "Flag", {"on": None}, None), Candidate(43, "Flag", {"on": True}, None)]
    strings, (keys, _) = option_keys(cands), _option_tuples(cands)
    for i in range(len(cands)):
        for j in range(len(cands)):
            assert (strings[i] == strings[j]) == (keys[i] == keys[j])

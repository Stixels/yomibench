import json
import zipfile

import pytest

from yomibench import ladder
from yomibench.bots import MatrixBot
from yomibench.engine import connect
from yomibench.selfplay import run_selfplay


@pytest.fixture(scope="module")
def nets(tmp_path_factory):
    """A toy value net and a toy policy net, trained on a few toy self-play games."""
    pytest.importorskip("torch")
    from yomibench import train, train_policy

    root = tmp_path_factory.mktemp("nets")
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50), [("Toy", "Toy")], range(6), root / "g")
    value, policy = root / "toy-value.npz", root / "toy-policy.npz"
    train.main([str(root / "g"), "--out", str(value), "--epochs", "1", "--hidden", "8", "--workers", "1",
                "--val-fraction", "0.25", "--no-ledger"])
    train_policy.main([str(root / "g"), "--out", str(policy), "--heldout-root", str(root / "ho"), "--epochs", "1",
                       "--workers", "1", "--device", "cpu", "--dev-fraction", "0.2"])
    return value, policy


def _season(tmp_path, pairs=1):
    engine = connect("toy")
    hello = engine.hello()
    engine.close()
    root = tmp_path / "s1"
    ladder.init(root, hello, ["Toy"], pairs, 12345, None, None, None, light_k=2, max_turns=1000)
    return ladder.Season(root)


def test_bundles_pack_check_and_refuse_tampering(nets, tmp_path):
    value, policy = nets
    m = ladder.pack("ours-v1", "us", value, policy, tmp_path / "ours.zip")
    assert m["modes"] == ["reflex", "light", "full"] and m["digest"] == ladder.check(tmp_path / "ours.zip")["digest"]
    only_value = ladder.pack("v-only", "us", value, None, tmp_path / "v.zip")
    assert only_value["modes"] == ["full"]
    with pytest.raises(ladder.LadderError):
        ladder.pack("Bad Name", "us", value, None, tmp_path / "x.zip")
    with pytest.raises(ladder.LadderError):
        ladder.pack("empty", "us", None, None, tmp_path / "x.zip")

    # a changed file no longer matches its manifest
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(tmp_path / "ours.zip") as src, zipfile.ZipFile(bad, "w") as out:
        for n in src.namelist():
            out.writestr(n, src.read(n)[:-1] + b"x" if n.endswith("value.npz") else src.read(n))
    with pytest.raises(ladder.LadderError, match="sha256"):
        ladder.check(bad)
    # nothing outside NAME/{bundle.json,value.npz,policy.npz} is extracted
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("ours-v1/bundle.json", "{}")
        z.writestr("ours-v1/../../escape.txt", "x")
    with pytest.raises(ladder.LadderError, match="unexpected"):
        ladder.check(evil)


def test_a_season_plays_missing_games_only_and_rates_them(nets, tmp_path):
    value, policy = nets
    season = _season(tmp_path)
    ladder.pack("ours-v1", "us", value, policy, tmp_path / "a.zip")
    ladder.pack("theirs-v1", "them", value, None, tmp_path / "b.zip")
    season.add(tmp_path / "a.zip")
    assert season.add(tmp_path / "a.zip")["name"] == "ours-v1"  # the same bundle again changes nothing
    season.add(tmp_path / "b.zip")
    ladder.pack("ours-v1", "us", policy=None, value=value, out=tmp_path / "a2.zip")
    with pytest.raises(ladder.LadderError, match="new name"):
        season.add(tmp_path / "a2.zip")

    # Reflex: theirs-v1 has no policy, so it sits out; ours-v1 meets both anchors, the anchors each other.
    assert {(a, b) for a, b, _ in season.missing("reflex")} == {("matrix-do", "ours-v1"), ("matrix-do", "random"),
                                                                ("ours-v1", "random")}
    played = ladder.play(season, "reflex", ["toy"], progress=lambda s: None)
    assert played == 6 and season.missing("reflex") == []
    assert ladder.play(season, "reflex", ["toy"], progress=lambda s: None) == 0  # nothing left to play
    rows = season.rows("reflex")
    assert {tuple(sorted(r["bots"])) for r in rows} == {("matrix-do", "ours-v1"), ("matrix-do", "random"),
                                                        ("ours-v1", "random")}
    assert all(r["digests"][n] == season.entrants()[n]["digest"] for r in rows for n in r["bots"])

    # A new entrant leaves every other matchup's games (and seeds) as they were.
    seeds = {(tuple(r["bots"]), r["seed"]) for r in rows}
    ladder.pack("late-v1", "them", value, policy, tmp_path / "c.zip")
    season.add(tmp_path / "c.zip")
    assert all("late-v1" in (a, b) for a, b, _ in season.missing("reflex"))
    ladder.play(season, "reflex", ["toy"], progress=lambda s: None)
    assert seeds <= {(tuple(r["bots"]), r["seed"]) for r in season.rows("reflex")}

    board = ladder.rate(season)
    table = {r["entrant"]: r for r in board["modes"]["reflex"]["ratings"]}
    assert table["random"]["elo"] == 1000 and set(table) == {"random", "matrix-do", "ours-v1", "late-v1"}
    md = (season.root / "LEADERBOARD.md").read_text().splitlines()
    assert sum(" | ours-v1 | us | " in x for x in md) == 1 and sum(x.startswith("| ours-v1 |") for x in md) == 1
    assert json.loads((season.root / "leaderboard.json").read_text())["modes"]["reflex"]["games"] == 12


def test_two_copies_split_the_work_and_merge(nets, tmp_path):
    value, policy = nets
    ladder.pack("ours-v1", "us", value, policy, tmp_path / "a.zip")
    copies = []
    for name in ("mine", "friend"):
        season = _season(tmp_path / name)
        season.add(tmp_path / "a.zip")
        copies.append(season)
    mine, friend = copies
    ladder.play(mine, "full", ["toy"], only="random", progress=lambda s: None)
    ladder.play(friend, "full", ["toy"], progress=lambda s: None)
    assert ladder.merge(mine, "full", friend.results_path("full")) == 2  # all but the games it already had
    assert mine.missing("full") == []
    # the same seeds gave the same games on both copies
    key = lambda r: (tuple(r["bots"]), r["seed"], r["outcome"], r["turns"])  # noqa: E731
    assert sorted(map(key, mine.rows("full"))) == sorted(map(key, friend.rows("full")))

    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({**mine.rows("full")[0], "season": "elsewhere"}) + "\n")
    with pytest.raises(ladder.LadderError, match="season"):
        ladder.merge(mine, "full", other)


def test_a_season_refuses_another_engine(tmp_path):
    season = _season(tmp_path)
    season.config["engine"] = {**season.config["engine"], "version": "999"}
    with pytest.raises(ladder.LadderError, match="version"):
        ladder.play(season, "full", ["toy"], progress=lambda s: None)


def test_season_seed_from_both_secrets():
    assert ladder.season_seed("a", "b") == ladder.season_seed("b", "a") != ladder.season_seed("a", "c")
    assert len(ladder.commitment("a")) == 64


def test_a_net_from_another_trainer_is_refused_with_a_pointer_to_the_contract(tmp_path):
    np = pytest.importorskip("numpy")
    foreign = tmp_path / "foreign.npz"  # a net whose meta doesn't name one of our features versions
    np.savez(foreign, w1=np.zeros((3, 2), np.float32), meta=np.array(json.dumps({"vocab": {}})))
    with pytest.raises(ladder.LadderError, match=r"feature version None.*'What a compatible net is' in docs/SUBMIT\.md"):
        ladder.pack("foreign-v1", "them", foreign, None, tmp_path / "foreign.zip")

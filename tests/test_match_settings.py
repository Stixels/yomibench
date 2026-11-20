"""Match settings (e.g. the ranked preset) reach the engine, and records say which were played."""

import pytest

from yomibench import cli
from yomibench.engine import EngineError, connect
from yomibench.selfplay import read_record
from yomibench.toy import ToyEngine

RANKED = '{"stage_width": 600, "game_length": 3000, "starting_meter": 1.5}'
STANDARD = {"stage_width": 800, "game_length": 3600, "starting_meter": "0", "ceiling_height": 800}


class SettingsToy(ToyEngine):
    """The toy game with the settings protocol: hello lists the standard settings, and
    new_match takes overrides of them (refusing unknown keys and the unsupported ceiling)."""

    sent: list = []  # every new_match message, across instances

    def handle(self, msg):
        if msg.get("cmd") == "hello":
            return {**super().handle(msg), "match_settings": dict(STANDARD)}
        if msg.get("cmd") == "new_match":
            SettingsToy.sent.append(msg)
            for k, v in (msg.get("settings") or {}).items():
                if k not in STANDARD:
                    return {"error": {"code": "bad_request", "message": f"unknown setting {k}"}}
                if k == "ceiling_height" and v != STANDARD[k]:
                    return {"error": {"code": "unsupported", "message": "ceiling_height can't be changed"}}
        return super().handle(msg)


@pytest.fixture
def toy(monkeypatch):
    SettingsToy.sent = []
    monkeypatch.setattr("yomibench.toy.ToyEngine", SettingsToy)
    return SettingsToy


def _selfplay(out, *extra):
    return cli.main(["selfplay", "--engine", "toy", "--games", "3", "--out", str(out), "--no-ledger",
                     "--allow-sleep", *extra])


def test_selfplay_plays_every_game_at_the_given_settings_and_records_them(toy, tmp_path):
    assert _selfplay(tmp_path, "--match-settings", RANKED) == 0
    played = {"stage_width": 600, "game_length": 3000, "starting_meter": "1.5"}  # the sim's decimal string
    games = [m for m in toy.sent if m["seed"] in (0, 1, 2)]
    assert len(games) >= 3 and all(m["settings"] == played for m in games)
    files = sorted(tmp_path.glob("g*.jsonl.gz"))
    assert len(files) == 3
    for f in files:
        engine = read_record(f)[0]["engine"]
        assert engine["match_settings"] == {**STANDARD, **played}
        assert engine["default_match_settings"] == STANDARD


def test_without_settings_nothing_changes(toy, tmp_path):
    assert _selfplay(tmp_path) == 0
    assert toy.sent and not any("settings" in m for m in toy.sent)
    hello = SettingsToy().handle({"cmd": "hello"})
    for f in tmp_path.glob("g*.jsonl.gz"):
        assert read_record(f)[0]["engine"] == hello  # the engine's own hello, as before


def test_ranked_and_standard_records_never_train_together(toy, tmp_path):
    from yomibench.compat import check_keys, record_keys

    _selfplay(tmp_path / "ranked", "--match-settings", RANKED)
    _selfplay(tmp_path / "standard")
    headers = [read_record(f)[0] for f in sorted(tmp_path.glob("*/g*.jsonl.gz"))]
    with pytest.raises(SystemExit, match="match settings"):
        check_keys([record_keys(h) for h in headers])


def test_a_refused_setting_fails_at_connect_not_once_per_game(toy, tmp_path):
    with pytest.raises(EngineError, match="ceiling_height"):
        connect("toy", match_settings={"ceiling_height": 500})
    with pytest.raises(EngineError, match="unknown setting"):
        _selfplay(tmp_path, "--match-settings", '{"stage_wdith": 600}')
    assert not list(tmp_path.glob("g*.jsonl.gz")) and not (tmp_path / "drops.jsonl").exists()


@pytest.mark.parametrize("text", ["600", "{stage_width: 600}"])
def test_match_settings_must_be_a_json_object(text, capsys):
    with pytest.raises(SystemExit):
        cli.main(["selfplay", "--engine", "toy", "--out", "x", "--match-settings", text])
    assert "--match-settings" in capsys.readouterr().err


def test_arena_records_carry_the_settings(toy, tmp_path):
    assert cli.main(["arena", "--engine", "toy", "--bots", "random,greedy", "--pairs", "1", "--no-ledger",
                     "--record-dir", str(tmp_path), "--match-settings", '{"game_length": 3000}']) == 0
    assert all(m["settings"] == {"game_length": 3000} for m in toy.sent)
    files = list(tmp_path.glob("g*.jsonl.gz"))
    assert len(files) == 2
    assert all(read_record(f)[0]["engine"]["match_settings"]["game_length"] == 3000 for f in files)


def _sim_or_skip():
    sim = pytest.importorskip("yomisim")
    try:
        sim.Handler()
    except (RuntimeError, ValueError) as e:
        pytest.skip(f"the simulator cannot load its scene dumps: {e}")
    return sim


def test_the_sim_starts_a_ranked_match_on_the_ranked_preset():
    _sim_or_skip()
    import json

    engine = connect("sim", match_settings=json.loads(RANKED))
    obs = engine.new_match("Cowboy", "Cowboy", 1)
    assert obs.state["game_length"] == obs.state["time_left"] == 3000
    assert [(p["meter"], p["meter_partial"]) for p in obs.state["players"]] == [(1, 63), (1, 63)]  # 1.5 bars
    standard = connect("sim").new_match("Cowboy", "Cowboy", 1)
    assert standard.state["time_left"] == 3600 and standard.state["players"][0]["meter"] == 0


def test_the_batch_engine_plays_settings_or_refuses_them_on_an_older_build():
    """A build with Batch settings plays the ranked preset; an older one refuses settings clearly (and both play
    the standard settings when asked for none)."""
    sim = _sim_or_skip()
    if not hasattr(sim, "Batch"):
        pytest.skip("no batch engine in this simulator build")
    import json

    from yomibench import plugins

    if plugins.hook("batch") is None:
        pytest.skip("no batch plugin installed")
    BatchEngine = plugins.hook("batch").BatchEngine

    engine = BatchEngine(1)
    if hasattr(sim.Batch, "settings"):
        obs = engine.new_match("Cowboy", "Cowboy", 1, settings=json.loads(RANKED))
        assert obs.state["time_left"] == 3000
        assert [(p["meter"], p["meter_partial"]) for p in obs.state["players"]] == [(1, 63), (1, 63)]
    else:
        with pytest.raises(EngineError, match="standard match settings"):
            engine.new_match("Cowboy", "Cowboy", 1, settings={"stage_width": 600})
        with pytest.raises(EngineError, match="standard match settings"):
            BatchEngine(1, match_settings={"stage_width": 600})
    assert engine.new_match("Cowboy", "Cowboy", 1).state["time_left"] == 3600

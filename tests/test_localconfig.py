"""yomibench.localconfig: defaults without a file, the example file parses to the defaults, env beats file."""
import tomllib
from pathlib import Path

import pytest

from yomibench import localconfig
from yomibench.localconfig import LocalConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for var in ("YOMIBENCH_GAME_DIR", "YOMIBENCH_BEST", "YB_REPO", "YOMIBENCH_ENGINE_PLUGINS"):
        monkeypatch.delenv(var, raising=False)
    localconfig.load.cache_clear()
    yield
    localconfig.load.cache_clear()


def test_no_file_gives_the_public_defaults(monkeypatch):
    monkeypatch.setenv("YOMIBENCH_CONFIG", "-")
    assert localconfig.config_path() is None
    cfg = localconfig.load()
    assert cfg == LocalConfig()
    assert cfg.require_steam_install is True and cfg.hub_url is None and cfg.plugins == ()


def test_the_example_file_is_the_defaults():
    with open(ROOT / "yomibench.local.example.toml", "rb") as f:
        assert localconfig.parse(tomllib.load(f)) == LocalConfig()


def test_a_file_is_read_and_env_wins(tmp_path, monkeypatch):
    f = tmp_path / "c.toml"
    f.write_text('[machine]\nname = "rig"\n[paths]\ngame_dir = "G:/game"\n[hub]\nbest_url = "http://h/best.json"\n'
                 '[game]\nrequire_steam_install = false\n[plugins]\nmodules = ["a.b"]\n', encoding="utf-8")
    monkeypatch.setenv("YOMIBENCH_CONFIG", str(f))
    cfg = localconfig.load()
    assert (cfg.machine, cfg.game_dir, cfg.best_url, cfg.require_steam_install, cfg.plugins, cfg.source) == (
        "rig", Path("G:/game"), "http://h/best.json", False, ("a.b",), f)
    monkeypatch.setenv("YOMIBENCH_BEST", "/x/best.json")
    monkeypatch.setenv("YOMIBENCH_ENGINE_PLUGINS", "c.d, a.b")
    localconfig.load.cache_clear()
    cfg = localconfig.load()
    assert cfg.best_url == "/x/best.json" and cfg.plugins == ("a.b", "c.d")


def test_unknown_keys_warn_and_bad_types_fail():
    with pytest.warns(UserWarning, match="nope"):
        assert localconfig.parse({"machine": {"nope": 1}}) == LocalConfig()
    with pytest.raises(ValueError, match="require_steam_install"):
        localconfig.parse({"game": {"require_steam_install": "no"}})


def test_the_local_file_is_gitignored():
    assert "yomibench.local.toml" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

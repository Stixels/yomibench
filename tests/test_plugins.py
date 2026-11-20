"""yomibench.plugins: engines, the batch hook and mod overlays come from plugin modules, never by name."""
import sys
import types

import pytest

from yomibench import engine, game, localconfig, plugins


@pytest.fixture(autouse=True)
def _no_installed_plugins(monkeypatch):
    """Only the plugins each test names: none from installed packages' entry points."""
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda **kw: [])
    plugins.load.cache_clear()
    yield
    plugins.load.cache_clear()


@pytest.fixture
def fake_plugin(tmp_path, monkeypatch):
    overlay = tmp_path / "Extra.gd"
    overlay.write_text("extends Node\n", encoding="utf-8")
    batch = types.ModuleType("fakeplug_batch")
    batch.batch_arena_engines = lambda *a, **k: "arena"
    mod = types.ModuleType("fakeplug")
    mod.ENGINES = {"fake": lambda spec, cset: ("fake-engine", spec, cset)}
    mod.HOOKS = {"batch": "fakeplug_batch"}
    mod.MOD_OVERLAYS = {"YomiBench/Extra.gd": overlay, "YomiBenchVsBot/VsBot.gd": overlay}
    monkeypatch.setitem(sys.modules, "fakeplug", mod)
    monkeypatch.setitem(sys.modules, "fakeplug_batch", batch)
    monkeypatch.setenv("YOMIBENCH_ENGINE_PLUGINS", "fakeplug,not_installed_anywhere")
    localconfig.load.cache_clear()
    plugins.load.cache_clear()
    yield mod
    monkeypatch.delenv("YOMIBENCH_ENGINE_PLUGINS")
    localconfig.load.cache_clear()
    plugins.load.cache_clear()


def test_without_plugins_batch_is_refused_clearly():
    plugins.load.cache_clear()
    assert plugins.hook("batch") is None and plugins.engines() == {}
    with pytest.raises(SystemExit, match="engine 'batch' isn't available in this build"):
        plugins.batch()
    with pytest.raises(engine.EngineError, match="isn't available in this build"):
        engine.connect("sim")


def test_a_plugin_serves_engines_hooks_and_overlays(fake_plugin, tmp_path):
    assert engine.connect("fake:x") == ("fake-engine", "fake:x", None)
    assert plugins.batch().batch_arena_engines() == "arena"
    assert set(plugins.mod_overlays("YomiBench")) == {"Extra.gd"}
    assert set(plugins.mod_overlays("YomiBenchVsBot")) == {"VsBot.gd"}


def test_run_dirs_get_overlays_and_lose_stale_ones(fake_plugin, tmp_path):
    g = tmp_path / "game"
    g.mkdir()
    for name in (game.EXE, game.PACK):
        (g / name).write_bytes(b"x")
    run = game.build_run_dir(g, tmp_path / "run")
    assert (run / "yomibench" / "Extra.gd").exists()
    plugins.load.cache_clear()
    sys.modules["fakeplug"].MOD_OVERLAYS = {}
    game.build_run_dir(g, run)
    assert not (run / "yomibench" / "Extra.gd").exists()

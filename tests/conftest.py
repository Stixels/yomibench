import os

import pytest

# No developer's local config (yomibench.localconfig) reaches the tests: the public defaults, as in CI.
os.environ["YOMIBENCH_CONFIG"] = "-"
os.environ.pop("YOMIBENCH_ENGINE_PLUGINS", None)


@pytest.fixture(autouse=True)
def _fake_game_dirs_allowed(monkeypatch):
    """Tests build run dirs from fake game folders; the only-a-Steam-install rule has its own test."""
    from yomibench import game
    monkeypatch.setattr(game, "REQUIRE_STEAM_INSTALL", False)


@pytest.fixture(autouse=True)
def _private_ledger(tmp_path_factory, monkeypatch):
    """Runs under test append metrics rows; keep them out of the repo's metrics/ledger.jsonl."""
    monkeypatch.setenv("YOMIBENCH_LEDGER", str(tmp_path_factory.mktemp("ledger") / "ledger.jsonl"))


RUNAWAY_SITECUSTOMIZE = '''
import os, time
from yomibench import memcap, selfplay

_seed, _real, _on = int(os.environ["YOMIBENCH_TEST_RUNAWAY_SEED"]), selfplay.play_recorded, [False]


def play_recorded(engine, bot, chars, seed, *a, **kw):
    if seed == _seed:  # a runaway: it doesn't finish while its memory reads 64 GB
        _on[0] = True
        time.sleep(30)  # still here: the cap failed, and the game is then played and recorded
        _on[0] = False
    return _real(engine, bot, chars, seed, *a, **kw)


selfplay.play_recorded = play_recorded
memcap.process_memory_bytes = lambda: (64 << 30) if _on[0] else (1 << 20)
'''


@pytest.fixture
def runaway(tmp_path_factory, monkeypatch):
    """runaway(seed): in every Python process started from here on (self-play children are fresh
    interpreters, so a sitecustomize on their PYTHONPATH), game `seed` hangs with its process's
    memory reading 64 GB, as a search whose memory keeps growing; every other reading is 1 MB."""
    import os
    from pathlib import Path

    import yomibench

    def make(seed: int) -> None:
        site = tmp_path_factory.mktemp("runaway")
        (site / "sitecustomize.py").write_text(RUNAWAY_SITECUSTOMIZE, encoding="utf-8")
        src = str(Path(yomibench.__file__).parents[1])  # this checkout's code in the children too
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(site), src, os.environ.get("PYTHONPATH")])))
        monkeypatch.setenv("YOMIBENCH_TEST_RUNAWAY_SEED", str(seed))
    return make

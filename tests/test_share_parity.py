"""tools/share_parity.py: a tiny CPU run on synthetic records matches; a broken shared window doesn't."""

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import share_parity  # noqa: E402

from yomibench import cache, train  # noqa: E402
from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.selfplay import run_selfplay  # noqa: E402

AX = "--aux-search-heads 1,4 --aux-search-weight 0.1 --aux-hp-weight 0.1"  # Ax's heads: the same data key


def _records(path, seeds, explore=False):
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], seeds, path,
                 opening=(3, 1.0, 0.5) if explore else None)


@pytest.fixture
def live(tmp_path):
    """A 'live' data root (record folders, an eval folder, their caches and held-out manifests) and a
    saved fit command line over it, as the trainer writes them."""
    old, new, ev = tmp_path / "live" / "gen0", tmp_path / "live" / "gen1", tmp_path / "live" / "eval"
    _records(old, range(8), explore=True)
    _records(new, range(8, 16))
    _records(ev, range(16, 19))
    root, held = tmp_path / "live" / "cache", tmp_path / "live" / "heldout"
    for d in (old, new, ev):
        cache.build(d, root, workers=1, log=lambda *a: None)
    train.select_files([old, new], 0.25, 0, None, [ev], held, 0)  # the folders' held-out manifests
    cmd = (f"{sys.executable} -m yomibench.train {old} {new} --out {tmp_path / 'live' / 'm.npz'} --epochs 2 "
           f"--hidden 16 --value-weight 0.5 --workers 1 --seed 3 --eval-dir {ev} --window-budget-games 12 "
           f"--window-recent-hours 0 --window-seed 1 --heldout-root {held} --cache {root} --name m "
           f"--ledger {tmp_path / 'live' / 'ledger.jsonl'} --tag stage=train")
    (tmp_path / "cmdline.txt").write_text(cmd)
    snapshot = {p: p.stat().st_mtime_ns for p in (tmp_path / "live").rglob("*") if p.is_file()}
    return tmp_path, snapshot


def _run(tmp_path, work, *extra):
    return share_parity.main(["--fit-cmdline", str(tmp_path / "cmdline.txt"), "--work", str(work), "--arm-args", AX,
                              "--window-budget-games", "9", "--cpu", "--min-available-gb", "0",
                              "--min-commit-headroom-gb", "0", *extra])


def test_a_shared_window_matches_the_solo_fit_and_nothing_live_is_written(live, capsys):
    tmp_path, before = live
    work = tmp_path / "work"
    assert _run(tmp_path, work, "--control") == 0
    res = json.loads((work / "share-parity.json").read_text())
    assert res["match"] and res["sharing"] and res["weights_equal"] and res["data_equal"]
    assert res["verdict"] == "MATCH" and res["control_repeats"]  # on the CPU a fit repeats byte for byte
    assert [res["fits"][x]["shared_window"] for x in share_parity.FITS] == ["written", "exact", None]
    assert res["fits"]["B-shared"]["val_mse"] == res["fits"]["B-solo"]["val_mse"] is not None
    assert "row_select" in res["fits"]["B-solo"]["phase_seconds"]
    assert "row_select" not in res["fits"]["B-shared"]["phase_seconds"]  # read, not selected
    assert res["load_seconds"]["B-shared"] is not None and len(res["window_files"]) == 1
    assert res["fits"]["A"]["args"][res["fits"]["A"]["args"].index("--window-budget-games") + 1] == "9"
    assert not res["a_vs_b_data_differ"] or set(res["a_vs_b_data_differ"]) <= {"train.hp_margin", "val.hp_margin",
                                                                               "eval.hp_margin"}
    assert "share parity: MATCH" in capsys.readouterr().out
    after = {p: p.stat().st_mtime_ns for p in (tmp_path / "live").rglob("*") if p.is_file()}
    assert after == before  # the live folders, caches and manifests untouched


BREAK = '''
from yomibench import cache
_real = cache.load_window


def load_window(*a, **kw):  # a shared window whose targets are off by 1e-4
    win, shared = _real(*a, **kw)
    if win is not None:
        for s in win["splits"].values():
            s["y"] = s["y"] * 1.0001
    return win, shared


cache.load_window = load_window
'''


def test_a_broken_shared_window_is_a_mismatch(live, monkeypatch, capsys):
    tmp_path, _ = live
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(BREAK, encoding="utf-8")  # every fit process loads it
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), str(REPO / "src"), os.environ.get("PYTHONPATH", "")]))
    work = tmp_path / "work"
    assert _run(tmp_path, work) == 1
    res = json.loads((work / "share-parity.json").read_text())
    assert res["sharing"] and not res["match"] and res["verdict"] == "MISMATCH"  # it shared, and the shared fit differs
    assert not res["data_equal"] and "train.y" in res["data_differ"] and not res["weights_equal"]
    assert "share parity: MISMATCH" in capsys.readouterr().out


def test_a_different_data_key_is_a_mismatch_not_a_share(live):
    tmp_path, _ = live
    work = tmp_path / "work"
    assert share_parity.main(["--fit-cmdline", str(tmp_path / "cmdline.txt"), "--work", str(work), "--arm-args",
                              "--value-weight 0.3", "--cpu", "--min-available-gb", "0",
                              "--min-commit-headroom-gb", "0"]) == 1
    res = json.loads((work / "share-parity.json").read_text())
    assert not res["sharing"] and res["fits"]["B-shared"]["shared_window"] == "written"

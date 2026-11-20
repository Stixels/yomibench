"""train --share-window: arms with the same data arguments share one selected window and train
byte-identical models; arms with other data arguments, or records that changed, don't share."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yomibench import train  # noqa: E402
from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.selfplay import run_selfplay  # noqa: E402


def _records(path, seeds, explore=False):
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], seeds, path,
                 opening=(3, 1.0, 0.5) if explore else None)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Two generations (the first explored) and an eval folder; training on the CPU (exact replay)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    old, new, ev = tmp_path / "gen0", tmp_path / "gen1", tmp_path / "eval"
    _records(old, range(8), explore=True)
    _records(new, range(8, 16))
    _records(ev, range(16, 19))
    common = [str(old), str(new), "--epochs", "2", "--hidden", "16", "--value-weight", "0.5", "--workers", "1",
              "--seed", "3", "--heldout-root", str(tmp_path / "heldout"), "--cache", str(tmp_path / "cache"),
              "--eval-dir", str(ev), "--window-budget-games", "9", "--window-recent-hours", "0", "--window-seed", "1"]
    return tmp_path, common


ARMS = {  # as a round's arms: the same data arguments, different heads (Aall, Ax, Ar)
    "A": [],
    "X": ["--aux-search-heads", "1,4", "--aux-search-weight", "0.1", "--aux-hp-weight", "0.1"],
    "R": ["--aux-cand-value-weight", "0.05", "--aux-mix-weight", "0.02", "--aux-cand-k", "4"],
}


def _fit(common, out, extra=(), share=None, frozen=False):
    train.main([*common, *extra, "--out", str(out), "--name", out.stem]
               + (["--share-window", str(share)] if share else []) + (["--share-window-frozen"] if frozen else []))
    with np.load(out, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return arrays, meta


def _same_model(a, b):
    (wa, ma), (wb, mb) = a, b
    assert wa.keys() == wb.keys()
    for k in wa:
        assert wa[k].dtype == wb[k].dtype and wa[k].tobytes() == wb[k].tobytes(), k  # byte-equal weights
    skip = ("created", "shared_window")
    assert {k: v for k, v in ma.items() if k not in skip} == {k: v for k, v in mb.items() if k not in skip}
    assert ma["val_mse"] == mb["val_mse"] and ma["eval_mse"] == mb["eval_mse"]


def test_arms_sharing_a_window_train_the_models_they_train_alone(data):
    tmp, common = data
    alone = {name: _fit(common, tmp / "alone" / f"{name}.npz", extra) for name, extra in ARMS.items()}
    share = tmp / "share"
    shared = {name: _fit(common, tmp / "shared" / f"{name}.npz", extra, share) for name, extra in ARMS.items()}
    assert shared["A"][1]["shared_window"]["mode"] == "written"
    assert shared["X"][1]["shared_window"]["mode"] == shared["R"][1]["shared_window"]["mode"] == "exact"
    assert len({m["shared_window"]["dir"] for _, m in shared.values()}) == 1  # one window for the three arms
    assert len(list(share.iterdir())) == 1
    for name in ARMS:
        _same_model(alone[name], shared[name])
    assert "val_aux_search_mse" in shared["X"][1] and "val_aux_cand_value_mse" in shared["R"][1]
    rows = [json.loads(x) for x in open(os.environ["YOMIBENCH_LEDGER"], encoding="utf-8")]
    first, last = rows[-3]["metrics"], rows[-1]["metrics"]  # the shared A (wrote) and R (read) rows
    assert (first["shared_window"]["mode"], last["shared_window"]["mode"]) == ("written", "exact")
    assert "load_phase_seconds" not in last  # one schema: the fit's phase_seconds
    assert {"heldout_split", "cache_build", "cache_open", "eval_load", "candidate_cache", "share_import", "vocab",
            "assemble", "train_epochs"} <= set(last["phase_seconds"])
    assert not {"window_select", "row_select", "share_export"} & set(last["phase_seconds"])  # the window was read
    assert {"window_select", "row_select", "share_import", "share_export"} <= set(first["phase_seconds"])
    assert not {"share_import", "share_export"} & set(rows[0]["metrics"]["phase_seconds"])  # no sharing: as #220


def test_arms_with_other_data_arguments_dont_share(data):
    tmp, common = data
    share = tmp / "share"
    _, first = _fit(common, tmp / "a.npz", share=share)
    others = {"value weight": ["--value-weight", "0.3"], "sample window seed": ["--window-seed", "2"],
              "quick turns": ["--cheap-target", "next-full"], "explore targets": ["--explore-targets", "all"]}
    seen = {first["shared_window"]["dir"]}
    for why, extra in others.items():
        args = [*common, *extra]
        _, meta = _fit(args, tmp / f"{why.replace(' ', '-')}.npz", share=share)
        assert meta["shared_window"]["mode"] == "written", why  # its own window, never the first arm's
        seen.add(meta["shared_window"]["dir"])
    assert len(seen) == 1 + len(others) == len(list(share.iterdir()))
    alone = _fit([*common, "--value-weight", "0.3"], tmp / "alone.npz")
    shared = _fit([*common, "--value-weight", "0.3"], tmp / "again.npz", share=share)  # reads its own window
    assert shared[1]["shared_window"]["mode"] == "exact"
    assert alone[1]["val_mse"] == shared[1]["val_mse"]
    for k in alone[0]:
        assert np.array_equal(alone[0][k], shared[0][k]), k


def test_changed_records_select_again_unless_frozen(data):
    tmp, common = data
    share = tmp / "share"
    first = _fit(common, tmp / "first.npz", share=share)
    _records(tmp / "gen1", range(30, 32))  # games arrive between arms
    exact = _fit(common, tmp / "exact.npz", share=share)
    assert exact[1]["shared_window"]["mode"] == "written"  # selected again, the new games in
    assert exact[1]["val_games"] + exact[1]["train_games"] > first[1]["val_games"] + first[1]["train_games"] \
        or exact[1]["sample_window"]["candidates"] > first[1]["sample_window"]["candidates"]
    _records(tmp / "gen1", range(40, 42))
    frozen = _fit(common, tmp / "frozen.npz", share=share, frozen=True)
    assert frozen[1]["shared_window"]["mode"] == "frozen"  # the window written just before, without games 40-41
    assert frozen[1]["sample_window"] == exact[1]["sample_window"]
    assert (frozen[1]["train_games"], frozen[1]["val_games"]) == (exact[1]["train_games"], exact[1]["val_games"])
    assert frozen[1]["val_mse"] == exact[1]["val_mse"]  # same window, same seed: the same model
    for k in exact[0]:
        assert np.array_equal(exact[0][k], frozen[0][k]), k


def test_share_window_needs_the_cached_loader(data):
    tmp, common = data
    with pytest.raises(SystemExit):
        train.main([*common, "--out", str(tmp / "m.npz"), "--in-memory", "--share-window", str(tmp / "s")])
    with pytest.raises(SystemExit):
        train.main([*common, "--out", str(tmp / "m.npz"), "--share-window-frozen"])

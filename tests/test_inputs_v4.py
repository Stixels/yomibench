"""Inputs v4 (features_ext, feature version 4, net surgery): everything off is unchanged."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from yomibench import cache, features, features_ext, features_v4
from yomibench.features import Vocab
from yomibench.valuenet import EXT_W1, ValueNet, trunk

# cache.encoder_hash() of the trainer line before this branch (99e709c): the sidecar must not change it.
ENCODER_HASH_BEFORE = "853f7784eeb12ad1"


def test_encoder_hash_equals_the_trainer_lines():
    assert cache.encoder_hash() == ENCODER_HASH_BEFORE


def _state():
    hb = lambda st, act, dmg: {"active": act, "start_tick": st, "active_ticks": 3, "damage": dmg, "dir_x": "1",
                               "dir_y": "-0.5", "knockback": "4", "hit_height": 1, "facing": 1}
    p = lambda x, ch, data: {"character": ch, "hp": 800, "max_hp": 1000, "x": x, "y": 0, "vx": 1, "vy": 0, "facing": 1,
                             "state": "Chainsaw", "state_tick": 2, "hitlag": 0, "busy_interrupt": False,
                             "grounded": True, "stance": "Normal", "meter": 1, "meter_partial": 10, "burst": 1,
                             "burst_partial": 0, "combo": 0, "air_options": 0, "feints": 1, "state_length": 20,
                             "state_iasa": 15, "hitstun": None, "raw": {}, "state_data": data, "di": {"x": 0, "y": 0},
                             "combo_moves_used": {}, "hitboxes": [hb(5, False, 30), hb(1, True, 10)]}
    obj = lambda name, kind, owner, x, st, raw: {"name": name, "kind": kind, "owner": owner, "x": x, "y": -10,
                                                 "state": "Default", "state_tick": st, "raw": raw, "hitboxes": [],
                                                 "vx": 0.5, "vy": 0}
    objects = [obj(str(10 + k), "Caltrops", 2, 40 + k, 100 + k, {}) for k in range(9)]
    objects += [obj("3", "AfterImage", 1, -20, 50, {"detonating": True}),
                obj("4", "RobotGrenade", 2, 70, 300, {"ticks_left": 120, "active": True})]
    return {"tick": 400, "time_left": 2600, "game_length": 3000,
            "players": [p(-50, "Robot", {"Start": {"x": 10, "y": -5}, "End": {"x": -30, "y": 40}}),
                        p(60, "Cowboy", {"x": 5, "y": 5})],
            "objects": objects}


def test_encoder_hash_unchanged_by_sidecar():
    # features.py, valuenet.FEATURES and cache.FORMAT are untouched; the hash covers them, not valuenet.py.
    assert "features_ext" not in Path(features.__file__).read_text(encoding="utf-8")
    import yomibench.valuenet as vn
    assert vn.FEATURES == 3 and cache.FORMAT == 6


def test_v4_starts_with_v3_columns_bit_for_bit():
    s = _state()
    for seat in (0, 1):
        n3, c3 = features.encode(s, seat, Vocab())
        n4, c4 = features_v4.encode(s, seat, Vocab())
        assert n4[:len(n3)] == n3 and c4[:len(c3)] == c3
        assert len(n4) == features_v4.N_NUMERIC and len(c4) == features_v4.N_CATEGORICAL


def test_ext_reads_second_aim_reserved_slots_and_overflow():
    s = _state()
    v = Vocab()
    n, c = features_ext.encode_ext(s, 0, v)
    name = dict(zip(features_ext.EXT_NUMERIC_NAMES, n))
    assert name["ext3.me.n_xy"] == 2 and name["ext3.me.aim2_len"] == pytest.approx(50.0)
    assert name["ext3.me.aim2_x"] == pytest.approx(-0.6)  # me is left: no flip
    assert name["ext1.res0.present"] == 1 and name["ext1.res2.present"] == 1  # mine (AfterImage), theirs
    assert name["ext1.res0.detonating"] == 1
    theirs = [name[f"ext1.res{k}.same_kind"] for k in (2, 3)]
    assert sorted(theirs) == [1.0, 9.0]
    # 11 objects: the 3 farthest from either fighter (the AfterImage, 2 Caltrops) fall outside the 8 slots
    assert name["ext1.theirs.beyond8"] == 2 and name["ext1.mine.beyond8"] == 1
    assert name["ext.v4b"] == 0  # no engine ext fields in this state


def test_v41_orb_and_grenade_fixes():
    s = _state()
    s["objects"] = [
        {"name": "1", "kind": "Orb", "owner": 1, "x": 10, "y": -5, "state": "Default", "state_tick": 20,
         "vx": 0, "vy": 0, "hitboxes": [],
         "raw": {"locked": True, "push_ticks": 14, "strike_ticks_left": 5, "last_push_x": "-6.0", "last_push_y": "4.0"},
         "ext": {"orb_lifetime": 30000, "spawn_tick": 380}},
        {"name": "2", "kind": "RobotGrenade", "owner": 2, "x": 20, "y": -5, "state": "Default", "state_tick": 10,
         "vx": 0, "vy": 0, "hitboxes": [],
         "raw": {"hitbox_out": True, "active": False, "ticks_left": 80}},
    ]
    n0, _ = features_ext.encode_ext(s, 0, Vocab())
    name0 = dict(zip(features_ext.EXT_NUMERIC_NAMES, n0))
    # Orb: locked -> armed=1, orb_lifetime=30000 ignored (NO_LIFETIME=10000 -> ticks_left=0), two timers, push_x/y
    assert name0["ext1.res0.armed"] == 1.0
    assert name0["ext1.res0.ticks_left"] == 0.0
    assert name0["ext1.res0.timer"] == 14.0
    assert name0["ext1.res0.timer2"] == 5.0
    assert name0["ext1.res0.push_x"] == pytest.approx(-6.0)
    assert name0["ext1.res0.push_y"] == pytest.approx(4.0)
    # RobotGrenade: hitbox_out no longer masks active=False
    assert name0["ext1.res2.armed"] == 0.0
    assert name0["ext1.res2.timer"] == 80.0
    # Mirror symmetry for push_x from opponent's view (seat 1 flips x)
    n1, _ = features_ext.encode_ext(s, 1, Vocab())
    name1 = dict(zip(features_ext.EXT_NUMERIC_NAMES, n1))
    assert name1["ext1.res2.push_x"] == pytest.approx(6.0)
    assert name1["ext1.res2.push_y"] == pytest.approx(4.0)


def test_v41_extra_slots_pending_threat_and_stickybomb_countdown():
    s = _state()
    # 11 objects in _state() -> exobj0..2 filled (9th-11th nearest), exobj3 empty
    # Add a StickyBomb (countdown explode_ticks=7 at state_tick=40) with a pending hitbox near opp (x=55, y=0)
    s["objects"].append({
        "name": "99", "kind": "StickyBomb", "owner": 1, "x": 55, "y": 0, "state": "Default", "state_tick": 40,
        "vx": 0, "vy": 0, "raw": {"attached": True, "detonating": True},
        "ext": {"explode_ticks": 7, "lifetime": 999999, "spawn_tick": 360},
        "hitboxes": [{"active": False, "start_tick": 45, "active_ticks": 2, "damage": 90,
                      "dir_x": "1", "dir_y": "0", "knockback": "5", "hit_height": 1, "facing": 1}],
    })
    n, c = features_ext.encode_ext(s, 0, Vocab())
    name = dict(zip(features_ext.EXT_NUMERIC_NAMES, n))
    # StickyBomb countdown reads 7 directly (not max(0, 7 - 40) = 0)
    assert name["ext1.res0.ticks_left"] == 7.0 or name["ext1.res1.ticks_left"] == 7.0
    # Extra slots #9..#12 filled for all 4 overflow objects (12 total)
    for k in range(4):
        assert name[f"ext1.exobj{k}.present"] == 1.0
    # Pending threat summary for mine (targeting opp at x=60, y=0; StickyBomb at x=55, dist=5 < NEAR=80)
    assert name["ext1.mine.pending_count"] == 1.0
    assert name["ext1.mine.soonest_hit_in"] == 5.0
    assert name["ext1.mine.nearest_pending"] == pytest.approx(5.0)
    assert name["ext1.mine.pending_damage_near"] == 90.0


def test_ext_mirror_symmetry():
    s = _state()
    m = copy.deepcopy(s)
    for p in m["players"]:
        p["x"], p["vx"], p["facing"] = -p["x"], -p["vx"], -p["facing"]
    for o in m["objects"]:
        o["x"], o["vx"] = -o["x"], -o["vx"]
    def flipdata(d):
        if isinstance(d, dict):
            return {k: (-v if k == "x" else flipdata(v)) for k, v in d.items()}
        return d
    for p in m["players"]:
        p["state_data"] = flipdata(p["state_data"])
    for seat in (0, 1):
        a, _ = features_ext.encode_ext(s, seat, Vocab())
        b, _ = features_ext.encode_ext(m, seat, Vocab())
        assert np.allclose(a, b)


def test_parts_zero_their_columns_only():
    s = _state()
    full, cats = features_ext.encode_ext(s, 0, Vocab())
    only3, c3 = features_ext.encode_ext(s, 0, Vocab(), parts=(3,))
    sl = features_ext.PART_SLICES
    assert only3[sl[3]] == full[sl[3]]
    assert not any(only3[sl[1]]) and not any(only3[sl[2]]) and not any(c3)
    none, _ = features_ext.encode_ext(s, 0, Vocab(), parts=())
    assert not any(none[1:])


def _tiny_v3_model(tmp_path):
    rng = np.random.default_rng(0)
    vocab = Vocab()
    for seat in (0, 1):
        features.encode(_state(), seat, vocab)
    sizes = vocab.sizes()
    from yomibench.valuenet import EMBED
    width = features.N_NUMERIC + sum(EMBED[c] for c in features.CATEGORY_OF_SLOT)
    p = {"mean": rng.normal(size=features.N_NUMERIC).astype(np.float32),
         "std": (rng.random(features.N_NUMERIC) + 0.5).astype(np.float32),
         "w1": rng.normal(size=(width, 16)).astype(np.float32), "b1": rng.normal(size=16).astype(np.float32),
         "w2": rng.normal(size=(16, 16)).astype(np.float32), "b2": rng.normal(size=16).astype(np.float32),
         "w3": rng.normal(size=(16, 1)).astype(np.float32), "b3": rng.normal(size=1).astype(np.float32)}
    for cat in EMBED:
        p[f"emb_{cat}"] = rng.normal(size=(sizes[cat], EMBED[cat])).astype(np.float32)
    path = tmp_path / "v3.npz"
    ValueNet(p, vocab, {"features": 3, "name": "tiny", "state_v3_fraction": 1.0}).save(path)
    return path


@pytest.fixture(autouse=True)
def _v4_enabled():
    import yomibench.valuenet as vn
    had = dict(vn.ENCODERS)
    vn.enable_inputs_v4()
    yield
    vn.ENCODERS.clear()
    vn.ENCODERS.update(had)


def _make_v4_from_v3(parent: Path, out: Path, parts=(1, 2, 3)) -> None:
    """Build a zero-init v4 model from a v3 .npz (same layout as surgery, without training tools)."""
    with np.load(parent, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    meta["features"] = 4
    meta["ext_parts"] = list(parts)
    n_ext = features_ext.N_EXT_NUMERIC
    arrays["mean"] = np.concatenate([arrays["mean"], np.zeros(n_ext, np.float32)])
    arrays["std"] = np.concatenate([arrays["std"], np.ones(n_ext, np.float32)])
    ext_in = n_ext + sum(arrays[f"emb_{c}"].shape[1] for c in features_ext.EXT_CATEGORY_OF_SLOT)
    arrays[EXT_W1] = np.zeros((ext_in, arrays["w1"].shape[1]), np.float32)
    np.savez(out, meta=np.array(json.dumps(meta)), **arrays)


def test_v4_models_refused_unless_enabled(tmp_path, monkeypatch):
    import yomibench.valuenet as vn
    parent = _tiny_v3_model(tmp_path)
    out = tmp_path / "v4.npz"
    _make_v4_from_v3(parent, out, (1, 2, 3))
    monkeypatch.delitem(vn.ENCODERS, 4)
    with pytest.raises(ValueError, match="feature version 4"):
        ValueNet.load(out)


def test_surgery_is_bit_identical_and_loads_as_v4(tmp_path):
    parent = _tiny_v3_model(tmp_path)
    out = tmp_path / "v4.npz"
    _make_v4_from_v3(parent, out, (1, 2, 3))
    a, b = ValueNet.load(parent), ValueNet.load(out)
    assert b.version == 4 and EXT_W1 in b.p and not b.p[EXT_W1].any()
    s = _state()
    for seat in (0, 1):
        assert a.batch([s], seat).tolist() == b.batch([s], seat).tolist()


def test_ext_weights_matter_once_trained(tmp_path):
    parent = _tiny_v3_model(tmp_path)
    out = tmp_path / "v4.npz"
    _make_v4_from_v3(parent, out, (1, 2, 3))
    b = ValueNet.load(out)
    b.p["std"][features.N_NUMERIC:] = 1.0  # as if training found them varying
    b.p[EXT_W1] = np.full_like(b.p[EXT_W1], 0.01)
    nums, cats = b.encoder.encode(_state(), 0, b.vocab)
    h1 = trunk(b.p, np.array([nums], np.float32), np.array([cats]), b.encoder.CATEGORY_OF_SLOT)
    b.p[EXT_W1][:] = 0
    h0 = trunk(b.p, np.array([nums], np.float32), np.array([cats]), b.encoder.CATEGORY_OF_SLOT)
    assert not np.array_equal(h1, h0)


def test_v3_code_paths_ignore_nothing_new(tmp_path):
    # A version-3 model never sees ext columns: its encoder is still features.
    a = ValueNet.load(_tiny_v3_model(tmp_path))
    assert a.encoder is features

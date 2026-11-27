import pytest

torch = pytest.importorskip("torch")

from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.features import Vocab, encode  # noqa: E402
from yomibench.selfplay import run_selfplay  # noqa: E402


def test_mirrored_positions_encode_the_same():
    a = {"tick": 5, "players": [{"hp": 80, "max_hp": 100, "x": 10, "y": 0, "facing": 1},
                                {"hp": 60, "max_hp": 100, "x": 20, "y": 0, "facing": -1}], "objects": []}
    b = {"tick": 5, "players": [{"hp": 60, "max_hp": 100, "x": -20, "y": 0, "facing": 1},
                                {"hp": 80, "max_hp": 100, "x": -10, "y": 0, "facing": -1}], "objects": []}
    v = Vocab()
    assert encode(a, 0, v) == encode(b, 1, v)


# Horizontal raw fields, by what they mean in the game (world x positions, x directions and signed
# x speeds). Written out here rather than read from features.MIRRORED_RAW, so a field missing
# there fails the test instead of being skipped by it.
HORIZONTAL_RAW = {"Ninja": ["stored_momentum_x", "quick_slash_move_dir_x", "quick_slash_start_pos_x"],
                  "Cowboy": ["shot_dir_x", "stance_teleport_x", "fatal_cut_move_dir_x", "fatal_cut_start_pos_x"],
                  "Robot": ["loic_dir"], "Wizard": [], "Mutant": ["juke_dir_x"]}
# x quantities the game keeps relative to the fighter's facing: the same in the mirror image.
FACING_RELATIVE = {"lightning_slice_x"}


def _fighter(char, x, facing, sign=1):
    """A fighter with every directional field set; `sign=-1` gives its mirror image."""
    from yomibench.features import CHARACTER_RAW, SHARED_RAW

    raw = {name: 3 for name in SHARED_RAW}
    raw.update({name: k + 2 for k, name in enumerate(CHARACTER_RAW[char])})
    raw.update({name: sign * (k + 7) for k, name in enumerate(HORIZONTAL_RAW[char])})
    return {"character": char, "hp": 900, "max_hp": 1500, "x": sign * x, "y": -30, "vx": sign * 2.5, "vy": 1.0,
            "facing": sign * facing, "state": "QuickSlash", "state_tick": 3, "state_length": 20, "raw": raw,
            "state_data": {"Attack": {"id": 0}, "Direction": {"x": sign * 45, "y": -10}},
            "di": {"x": str(sign * 100), "y": "0"}, "combo_moves_used": {"QuickSlash": 2, "Uppercut": 1},
            "hitboxes": [{"active": False, "start_tick": 7, "active_ticks": 2, "damage": 80, "dir_x": "1.0",
                          "dir_y": "-0.5", "knockback": "9.0", "hit_height": 1, "facing": 1}]}


def _object(owner, x, vx, facing, sign=1):
    return {"kind": "Kunai", "state": "Default", "owner": owner, "x": sign * x, "y": -40, "vx": sign * vx, "vy": 0.5,
            "state_tick": 4, "hitboxes": [{"active": True, "start_tick": 1, "active_ticks": 5, "damage": 70,
                                           "dir_x": "1.0", "dir_y": "-1.0", "knockback": "5.0", "hit_height": 1,
                                           "facing": sign * facing}]}


@pytest.mark.parametrize("pair", [("Ninja", "Ninja"), ("Ninja", "Cowboy"), ("Robot", "Wizard"), ("Mutant", "Robot"),
                                  ("Cowboy", "Mutant")])
def test_mirrored_positions_encode_the_same_with_every_directional_field(pair):
    a, b = pair
    # p1 on the left facing right, p2 on the right; their objects flying at each other.
    s = {"tick": 50, "players": [_fighter(a, -100, 1), _fighter(b, 60, -1)],
         "objects": [_object(1, -20, 6.0, 1), _object(2, 30, -4.0, -1), _object(2, -90, 0.0, 1)]}
    # The mirror image with the seats swapped: same position seen from the other side.
    m = {"tick": 50, "players": [_fighter(b, 60, -1, -1), _fighter(a, -100, 1, -1)],
         "objects": [_object(2, -20, 6.0, 1, -1), _object(1, 30, -4.0, -1, -1), _object(1, -90, 0.0, 1, -1)]}
    for seat in (0, 1):
        v = Vocab()
        assert encode(s, seat, v) == encode(m, 1 - seat, v)
    # and the directional fields do reach the features (a mirror test can't see dropped inputs)
    v = Vocab()
    nums = encode(s, 0, v)[0]
    plain = encode({**s, "players": [{**s["players"][0], "raw": {}}, s["players"][1]]}, 0, v)[0]
    assert nums != plain


def test_every_horizontal_looking_raw_field_is_mirrored():
    import re

    from yomibench.features import CHARACTER_RAW, FACING_RELATIVE_RAW, MIRRORED_RAW, SHARED_RAW

    looks = re.compile(r"(^|_)(x|dir|facing|vel|force|pos)(_|$)")  # whole parts of the name
    names = set(SHARED_RAW) | {n for v in CHARACTER_RAW.values() for n in v}
    horizontal = {n for n in names if looks.search(n) and not n.endswith("_y")}  # y parts are vertical
    assert horizontal <= MIRRORED_RAW | FACING_RELATIVE_RAW
    assert {n for v in HORIZONTAL_RAW.values() for n in v} == MIRRORED_RAW
    assert FACING_RELATIVE_RAW == FACING_RELATIVE


def test_move_data_is_read_from_each_kind_of_submitted_data():
    from yomibench.features import _move_data

    assert _move_data({"x": -1, "y": 1}) == (-1, 1, 0, 0, 0)
    assert _move_data({"Attack": {"id": 0, "name": "No Attack"}, "Direction": {"x": 45, "y": 0}}) == (45, 0, 0, 0, 0)
    assert _move_data({"AutoCorrect": False, "Distance": {"x": 100}}) == (0, 0, 100, 0, 0)
    assert _move_data({"Block Height": {"x": 0, "y": 1}, "Melee Parry Timing": {"count": 20}}) == (0, 0, 0, 20, 1)
    for other in (None, True, "homing", 8, {"hitbox": None}):
        assert _move_data(other) == (0, 0, 0, 0, 0)


def test_a_state_from_before_v3_encodes_with_the_move_fields_off():
    from yomibench.features import MOVE_NUMS, N_NUMERIC, PLAYER_NUMS, TIMING_NUMS

    s = {"tick": 50, "players": [_fighter("Ninja", -100, 1), _fighter("Ninja", 60, -1)], "objects": []}
    old = {"tick": 50, "players": [{k: v for k, v in p.items() if k not in ("state_data", "combo_moves_used", "hitboxes")}
                                   for p in s["players"]], "objects": []}
    v = Vocab()
    new_nums, old_nums = encode(s, 0, v)[0], encode(old, 0, v)[0]
    assert len(new_nums) == len(old_nums) == N_NUMERIC
    at = len(PLAYER_NUMS) + len(TIMING_NUMS)
    assert new_nums[at] == 1 and old_nums[at] == 0  # the v3 flag
    di = at + MOVE_NUMS.index("di_x")
    assert old_nums[di] == new_nums[di] == 100  # DI still counts (records get it back from picks)


def test_old_records_get_di_back_from_the_picks(tmp_path):
    import gzip
    import json

    from yomibench.selfplay import read_slim

    def turn(n, dis):
        cands = [[{"action": "Continue", "data": None, "extra": {"DI": d}}] for d in dis]
        return {"turn": n, "state": {"players": [{"hp": 1}, {"hp": 1}]}, "candidates": cands, "picks": [0, 0]}
    path = tmp_path / "g000000000000-x.jsonl.gz"
    lines = [{"format": 3}, turn(0, [{"x": 100, "y": 0}, {"x": 0, "y": 0}]),
             turn(1, [{"x": 0, "y": -100}, {"x": 0, "y": 0}]), turn(2, [{"x": 0, "y": 0}] * 2), {"result": {}}]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(x) + "\n" for x in lines))
    _, _, turns = read_slim(path)
    assert all(t.candidate_values is None for t in turns)  # records from before candidate values
    assert [[p["di"] for p in s["players"]] for s, *_ in turns] == [
        [{"x": 0, "y": 0}] * 2, [{"x": 100, "y": 0}, {"x": 0, "y": 0}], [{"x": 0, "y": -100}, {"x": 0, "y": 0}]]


def test_directional_start_defaults_are_flipped_like_explicit_raw():
    import copy

    s = {"tick": 50, "players": [_fighter("Ninja", -100, 1), _fighter("Ninja", 60, -1)], "objects": []}
    for p in s["players"]:
        p["raw"] = {}
    v = Vocab(raw_defaults={"Ninja": {"stored_momentum_x": "12.5", "quick_slash_start_pos_x": -300}})
    explicit = copy.deepcopy(s)
    for p in explicit["players"]:
        p["raw"] = dict(v.raw_defaults["Ninja"])
    assert encode(s, 0, v) == encode(explicit, 0, v)


# Scores of the model built by _v2_model_and_states, computed by origin/main's code (feature version
# 2, before version 3 existed). A v2 model must keep scoring exactly this.
V2_GOLDEN = [0.05477985739707947, -0.05477985739707947, -0.08492863178253174, 0.08492863178253174]


def _v2_model_and_states():
    """A small version-2 model (checkpoint 1's input layout) and positions that exercise its inputs."""
    import numpy as np

    from yomibench import features_v2
    from yomibench.valuenet import EMBED, ValueNet

    rng = np.random.default_rng(7)
    n = features_v2.N_NUMERIC
    width = n + sum(EMBED[c] for c in features_v2.CATEGORY_OF_SLOT)
    p = {"mean": rng.normal(size=n).astype(np.float32), "std": (20 + 80 * rng.random(n)).astype(np.float32),
         "w1": (rng.normal(size=(width, 8)) * 0.5).astype(np.float32), "b1": 0.1 * rng.normal(size=8).astype(np.float32),
         "w2": (rng.normal(size=(8, 8)) * 0.3).astype(np.float32), "b2": 0.1 * rng.normal(size=8).astype(np.float32),
         "w3": (rng.normal(size=(8, 1)) * 0.1).astype(np.float32), "b3": np.zeros(1, np.float32)}
    for cat, dim in EMBED.items():
        p[f"emb_{cat}"] = rng.normal(size=(3, dim)).astype(np.float32)
    vocab = Vocab(ids={"state": {"Idle": 1, "QuickSlash": 2}, "stance": {"Normal": 1}, "char": {"Ninja": 1},
                       "kind": {"Kunai": 1}, "ostate": {"Default": 1}},
                  raw_defaults={"Ninja": {"sticky_bombs_left": 3, "stored_momentum_x": "0"}})

    def fighter(x, facing, state, raw):
        return {"character": "Ninja", "hp": 1100, "max_hp": 1500, "x": x, "y": -12, "vx": 2.5, "vy": -1,
                "facing": facing, "state": state, "stance": "Normal", "state_tick": 3, "state_length": 14,
                "state_iasa": 9, "hitstun": None, "meter": 1, "combo": 2, "raw": raw}
    states = [
        {"tick": 300, "time_left": 3300, "players": [fighter(-120, 1, "Idle", {"penalty": -12}),
                                                     fighter(80, -1, "QuickSlash", {"stored_momentum_x": "3.5"})],
         "objects": [{"kind": "Kunai", "state": "Default", "owner": 1, "x": -40, "y": -30, "vx": 6.0, "vy": 0,
                      "state_tick": 5, "raw": {}}]},
        {"tick": 900, "time_left": 2700, "players": [fighter(200, -1, "QuickSlash", {"stored_momentum_x": "-6.25"}),
                                                     fighter(-50, 1, "Idle", {"sticky_bombs_left": 1})],
         "objects": []},
    ]
    return ValueNet(p, vocab, {"name": "v2-golden", "features": 2}), states


def test_v2_models_still_load_and_score_exactly_as_before(tmp_path):
    from yomibench import features_v2
    from yomibench.valuenet import FEATURES, ValueNet

    assert FEATURES == 3
    net, states = _v2_model_and_states()
    net.save(tmp_path / "ckpt1-style.npz")
    old = ValueNet.load(tmp_path / "ckpt1-style.npz")
    assert old.version == 2 and old.encoder is features_v2
    assert [float(old(s, seat)) for s in states for seat in (0, 1)] == pytest.approx(V2_GOLDEN, abs=1e-7)
    old.save(tmp_path / "again.npz")  # saving keeps its version
    assert ValueNet.load(tmp_path / "again.npz").version == 2


def test_a_model_of_an_unknown_feature_version_is_refused(tmp_path):
    import json

    import numpy as np

    from yomibench.valuenet import ValueNet

    for version in (1, 5, None):
        path = tmp_path / f"v{version}.npz"
        np.savez(path, meta=np.array(json.dumps({"features": version, "vocab": {}})))
        with pytest.raises(ValueError, match=rf"feature version {version}; this code plays versions \[2, 3, 4\]"):
            ValueNet.load(path)


def test_train_then_numpy_inference_matches_torch(tmp_path):
    import numpy as np

    from yomibench import train
    from yomibench.valuenet import ValueNet

    run_selfplay([lambda: connect("toy")] * 2, lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(6), tmp_path)
    out = tmp_path / "v.npz"
    train.main([str(tmp_path), "--out", str(out), "--epochs", "2", "--hidden", "16"])
    net = ValueNet.load(out)
    vocab = Vocab()
    data = train.load_samples([tmp_path], vocab, 0.3, 0)
    t = train.Net(vocab.sizes(), np.zeros(train.N_NUMERIC, np.float32), np.ones(train.N_NUMERIC, np.float32), 16)
    with torch.no_grad():
        for k, v in net.p.items():  # load the saved weights into a torch net
            if k in ("mean", "std"):
                getattr(t, k).copy_(torch.tensor(v))
            elif k.startswith("emb_"):
                t.emb[k[4:]].weight.copy_(torch.tensor(v))
            else:
                layer = getattr(t, "l" + k[1])
                (layer.weight if k[0] == "w" else layer.bias).copy_(torch.tensor(v.T if k[0] == "w" else v))
        want = t(torch.tensor(data["train"]["nums0"]), torch.tensor(data["train"]["cats0"])).numpy()
    got = net.batch_encoded(data["train"]["nums0"], data["train"]["cats0"])
    assert np.allclose(want, got, atol=1e-5)
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    assert -1 <= net(obs.state, 0) <= 1
    assert net(obs.state, 1) == pytest.approx(-net(obs.state, 0))
    MatrixBot(iterations=30, value=net).choose(obs, 0, engine, __import__("random").Random(0))


def test_the_saved_model_is_the_best_epoch_not_a_mix(tmp_path):
    from yomibench import train

    t = train.Net({"state": 3, "stance": 2, "char": 2, "kind": 2, "ostate": 2}, __import__("numpy").zeros(train.N_NUMERIC, "float32"),
                  __import__("numpy").ones(train.N_NUMERIC, "float32"), 8)
    snap = t.export()
    before = {k: v.copy() for k, v in snap.items()}
    with torch.no_grad():
        for prm in t.parameters():
            prm.add_(1.0)
    assert all((snap[k] == before[k]).all() for k in snap)


def test_v3_models_trained_without_v3_state_play_states_without_it():
    import numpy as np

    from yomibench import train
    from yomibench.valuenet import ValueNet, strip_v3_state

    s = {"tick": 50, "players": [_fighter("Ninja", -100, 1), _fighter("Ninja", 60, -1)], "objects": []}
    v = Vocab()
    # Standardize on these very states: with raw inputs (hp 900, x -100, ...) every unit saturates,
    # tanh returns exactly +-1 for both views, and every value is 0 whatever the state.
    rows = np.array([encode(x, seat, v)[0] for x in (s, strip_v3_state(s)) for seat in (0, 1)], np.float32)
    torch.manual_seed(0)
    params = train.Net(v.sizes(), rows.mean(axis=0), rows.std(axis=0) + 1, 8).export()
    # trained on sim records that lacked the move fields (meta predates the fraction, like ckpt2)
    old_data = ValueNet(params, Vocab(ids=dict(v.ids)), {"features": 3})
    new_data = ValueNet(params, Vocab(ids=dict(v.ids)), {"features": 3, "state_v3_fraction": 1.0})
    assert old_data.mask_v3_state and not new_data.mask_v3_state
    assert old_data(s, 0) == old_data(strip_v3_state(s), 0) == new_data(strip_v3_state(s), 0)
    assert new_data(s, 0) != new_data(strip_v3_state(s), 0)
    assert "hitboxes" in s["players"][0]  # the caller's state is untouched

"""BatchValue gathers previews' features from their batch calls' arrays: the same inputs, bit for bit, as
stacking each preview's own rows."""

import numpy as np

from yomibench.netvalue import BatchValue, Preview


def _call(rng, n, key):
    nums = rng.standard_normal((n, 2, 7)).astype(np.float32)
    cats = rng.integers(0, 9, (n, 2, 3)).astype(np.int32)
    src = {key: (nums, cats)}
    out = []
    for k in range(n):
        p = Preview()
        p._src, p._row = src, k
        out.append(p)
    return out


def test_gathered_inputs_equal_stacked_rows():
    rng = np.random.default_rng(3)
    bv = BatchValue.__new__(BatchValue)
    bv.key = "k"
    a, b = _call(rng, 11, "k"), _call(rng, 5, "k")
    for previews in (a, a[3:9], [a[7], a[2], a[2]], a[::2] + b + a[1::3], [b[4], a[0], b[1]]):
        nums, cats = bv._inputs(previews)
        want_n = np.stack([p.feats["k"][0] for p in previews])
        want_c = np.stack([p.feats["k"][1] for p in previews])
        assert nums.dtype == want_n.dtype and cats.dtype == want_c.dtype
        assert nums.tobytes() == want_n.tobytes() and cats.tobytes() == want_c.tobytes()
        assert nums.flags.c_contiguous and nums.shape == want_n.shape
        nums[:] = 0  # a fresh copy: the call's arrays are untouched
        assert previews[0].feats["k"][0].any()


def test_features_set_directly_still_stack():
    bv = BatchValue.__new__(BatchValue)
    bv.key = "k"
    ps = []
    for k in range(3):
        p = Preview({"x": k})
        p.feats = {"k": (np.full((2, 4), k, np.float32), np.full((2, 2), k, np.int64))}
        ps.append(p)
    nums, cats = bv._inputs(ps)
    assert nums.shape == (3, 2, 4) and cats[2, 1, 1] == 2 and dict(ps[1]) == {"x": 1}


def test_batched_forward_is_close_but_flagged():
    """--batched-net-forward: one forward over both views (2N rows). Close to the default (BLAS sums in a
    batch-shape-dependent order, so not bit for bit), terminal values exact, and the bot config says so."""
    from types import SimpleNamespace

    from yomibench import features
    from yomibench.bots import MatrixBot

    rng = np.random.default_rng(5)
    cos = features.CATEGORY_OF_SLOT
    n_num, hid = 30, 16
    p = {"mean": np.zeros(n_num, np.float32), "std": np.ones(n_num, np.float32)}
    for cat in set(cos):
        p[f"emb_{cat}"] = rng.standard_normal((6, 3)).astype(np.float32)
    width = n_num + 3 * len(cos)
    p.update(w1=rng.standard_normal((width, hid)).astype(np.float32) * 0.1, b1=np.zeros(hid, np.float32),
             w2=rng.standard_normal((hid, hid)).astype(np.float32) * 0.1, b2=np.zeros(hid, np.float32),
             w3=rng.standard_normal((hid, 1)).astype(np.float32), b3=np.zeros(1, np.float32))
    net = SimpleNamespace(p=p, encoder=features, residual_hp=False, mask_v3_state=False, __name__="valuenet:t",
                          version=3, vocab=SimpleNamespace(ids={}, raw_defaults={}))
    plain, batched = BatchValue(net), BatchValue(net, batched_forward=True)
    for n in (1, 7, 64, 333):
        nums = rng.standard_normal((n, 2, n_num)).astype(np.float32)
        cats = rng.integers(0, 6, (n, 2, len(cos))).astype(np.int32)
        src = {plain.key: (nums, cats)}
        ps = []
        for k in range(n):
            q = Preview()
            q._src, q._row = src, k
            q.done, q.winner = k == 0, 1
            ps.append(q)
        for seat in (0, 1):
            a, b = plain.batch(ps, seat), batched.batch(ps, seat)
            assert np.allclose(a, b, atol=1e-5) and a[0] == b[0] == (1.0 if seat == 1 else -1.0)
    assert "value_forward" not in MatrixBot(value=plain).config()
    assert MatrixBot(value=batched).config()["value_forward"] == "batched"

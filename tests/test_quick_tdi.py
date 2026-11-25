"""quick-toggles-di's +tdiB quick menus (quick_policy.add_tdi): default off, size kept, one-factor variants only,
deterministic, refusals dropped."""
import json
import zlib

import numpy as np

from yomibench import quick_policy
from yomibench.engine import Candidate


class StubNet:
    def scores(self, state, seat, cands):
        return np.array([(zlib.crc32(json.dumps([c.action, c.data, c.extra], sort_keys=True).encode()) % 1000) / 250.0
                         for c in cands])


class StubQP:
    net = StubNet()


F = {"di": 5, "feint": True, "reverse": True, "extra_set": 3}
V = {"di": 0, "feint": False, "reverse": False, "extra_set": 0}


def cand(i, action, x):
    return Candidate(i, action, {"x": x, "y": 0}, {"DI": {"x": 0, "y": 0}}, dict(F), dict(V))


def menu():
    out = [Candidate(0, "Continue", None, {"DI": {"x": 0, "y": 0}}, {"di": 5}, {"di": 0})]
    for k in range(6):
        for x in range(4):  # 4 aim variants of each move (displace can make room)
            out.append(cand(len(out), f"Move{k}", x * 10))
    return out


class StubEngine:
    def __init__(self, refuse=()):
        self.next_id, self.asked, self.refuse = 1000, [], set(refuse)

    def expand(self, specs):
        out = []
        for s in specs:
            self.asked.append(s)
            ov = s["extra_override"]
            if ov.get("extra_set") in self.refuse:
                out.append("refused: no such extra set")
                continue
            self.next_id += 1
            out.append(Candidate(self.next_id, f"base{s['candidate']}", None, {"ov": ov}, None, dict(ov),
                                 base=s["candidate"]))
        return out


def test_names_parse_and_cap():
    for p in ("policy40+gumbel8+parryx6+tdi8", "policy40+gumbel8+tdi4"):
        assert quick_policy.is_quick_policy(p)
        assert quick_policy.parse(p)[:2] == (40, 8)
        quick_policy.check_cap(p, 48)
    assert quick_policy.tdi_budget("policy40+gumbel8+parryx6") == 0
    assert quick_policy.tdi_budget("policy40+gumbel8+parryx6+tdi8") == 8


def test_default_off_is_identity():
    m = [menu(), menu()]
    assert quick_policy.add_tdi(StubEngine(), {}, m, "policy40+gumbel8+parryx6", "1:2", StubQP()) is m


def test_size_kept_one_factor_variants_deterministic():
    e1, e2 = StubEngine(), StubEngine()
    a = quick_policy.add_tdi(e1, {}, [menu(), menu()], "policy40+gumbel8+parryx6+tdi8", "7:3", StubQP())
    b = quick_policy.add_tdi(e2, {}, [menu(), menu()], "policy40+gumbel8+parryx6+tdi8", "7:3", StubQP())
    for seat in (0, 1):
        assert len(a[seat]) == len(menu())
        added = [c for c in a[seat] if c.base is not None]
        assert len(added) == 8
        assert [c.extra for c in added] == [c.extra for c in b[seat] if c.base is not None]
        assert {c.action for c in a[seat] if c.base is None} == {c.action for c in menu()}  # no move lost
    for s in e1.asked:  # exactly one factor changed from the default values
        ov = s["extra_override"]
        changed = [f for f in ov if ov[f] != V.get(f, 0)]
        assert len(changed) == 1 and changed[0] in quick_policy.TDI_FACTORS


def test_refusals_dropped_and_salt_changes_pick():
    eng = StubEngine(refuse={1, 2})
    out = quick_policy.add_tdi(eng, {}, [menu(), menu()], "policy40+gumbel8+tdi8", "9:1", StubQP())
    for seat in (0, 1):
        assert all((c.values or {}).get("extra_set", 0) not in (1, 2) for c in out[seat] if c.base is not None)
    other = quick_policy.add_tdi(StubEngine(), {}, [menu(), menu()], "policy40+gumbel8+tdi8", "9:2", StubQP())
    assert [c.extra for c in other[0] if c.base is not None] != [c.extra for c in out[0] if c.base is not None]

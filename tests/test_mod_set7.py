"""Candidate set 7 in the mod (Candidates.gd, YomiServer.gd): its rules hashes are the simulator's."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mod" / "YomiBench"
spec = importlib.util.spec_from_file_location("rules_hash", ROOT / "tools/rules_hash.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

# The simulator plugin's hello()["candidate_rules"] for candidate set 6 with these switches.
SET6_DEFAULT = "3b8bb1b76cd8"
SET7 = "ff2f16e0ef3f"
SET7_SWITCH = "d1bd8211465b"


def test_set6_default_hash_is_unchanged():
    assert rh.rules_hash6(MOD) == SET6_DEFAULT


def test_set7_hashes_are_the_simulators():
    assert rh.rules_hash6(MOD, set6=rh.set7()) == SET7
    assert rh.rules_hash6(MOD, set6=rh.set7(switch_sets=True)) == SET7_SWITCH


def test_every_extra_switch_combination_has_its_own_aliased_hash():
    aliases = json.loads((MOD / "rules_aliases.json").read_text(encoding="utf-8"))["aliases"]
    rh.RAW = True
    try:
        raw = [rh.rules_hash6(MOD, set6={k: bool(b >> n & 1) for n, k in enumerate(rh.SET6_EXTRA_KEYS)})
               for b in range(1 << len(rh.SET6_EXTRA_KEYS))]
    finally:
        rh.RAW = False
    assert all(h in aliases for h in raw)
    assert len({aliases[h] for h in raw}) == len(raw)


def _gd_list(text, name):
    m = re.search(r"^const %s = \[(.*?)\]" % name, text, re.M | re.S)
    return re.findall(r'"([^"]+)"', m.group(1))


def test_server_switch_lists_match_the_hash_tool():
    server = (MOD / "YomiServer.gd").read_text(encoding="utf-8")
    assert tuple(_gd_list(server, "SET6_KEYS")) == rh.SET6_KEYS
    assert tuple(_gd_list(server, "SET6_EXTRA_KEYS")) == rh.SET6_EXTRA_KEYS
    assert tuple(_gd_list(server, "SET7_KEYS")) == rh.SET7_KEYS
    assert "7" in _gd_list(server, "CANDIDATE_SETS")


def _func(text, name):
    m = re.search(r"^func %s\(.*?(?=^func |\Z)" % name, text, re.M | re.S)
    return m.group(0)

"""Candidate set 7 in the mod (Candidates.gd, YomiServer.gd): its rules hashes are the simulator's, and `probe` starts
from fresh widgets and refuses DI while DI is disabled.

The real-game test at the end runs only with YOMIBENCH_REAL_GAME_TESTS=1 and a game install (yomibench.game); it
starts a muted, hidden server under set 7 + switch_sets."""

import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

import pytest

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


def test_probe_starts_from_fresh_widgets():
    body = _func((MOD / "Candidates.gd").read_text(encoding="utf-8"), "probe")
    code = [ln.strip() for ln in body.splitlines()[1:] if ln.strip() and not ln.strip().startswith("#")]
    assert code[0] == "_new_epoch()"


def test_probe_refuses_di_while_di_is_disabled():
    body = _func((MOD / "Candidates.gd").read_text(encoding="utf-8"), "_probe")
    assert re.search(r'if \(di\.x != 0 or di\.y != 0\) and not fighter\.di_enabled:\n\t+return \{"error": "DI is disabled"\}',
                     body)


# --- real game (opt-in) -------------------------------------------------------------------------------------------

def _real_game():
    if os.environ.get("YOMIBENCH_REAL_GAME_TESTS") != "1":
        pytest.skip("set YOMIBENCH_REAL_GAME_TESTS=1 to run against the game")
    from yomibench import game

    try:
        game.find_game_dir()
    except FileNotFoundError:
        pytest.skip("no game install")
    return game


STD = {"stage_width": 600, "game_length": 3000, "starting_meter": "1.5", "has_ceiling": True, "ceiling_height": 500}


def test_real_probe_is_fresh_and_refuses_disabled_di(tmp_path):
    game = _real_game()
    server = game.GameServer(run_dir=tmp_path / "run", below_normal=True,
                             extra_args=["--yomi-candidate-set=7", "--yomi-set6=switch_sets=1"])
    engine = game.connect_server(server.port, owned=server)
    if sys.platform == "win32":  # the game raises itself to AboveNormal as it starts; put it back
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x0600, False, server.proc.pid)
        if h:
            k32.SetPriorityClass(h, 0x4000)
            k32.CloseHandle(h)
    try:
        call = engine._call
        assert engine.hello()["candidate_rules"] == SET7_SWITCH
        # A probe of a switch or panel value's widget clicks gives the expanded candidate's extras, whatever the
        # expands, previews and probes before it left in the panel (the live compare's FlyDir memory, a juke left set).
        # A probe giving a listed or expanded candidate's exact input carries that candidate's metadata, and a probe
        # leaves expand's repeat answers ("same candidate as N") naming the candidate's own id.
        rng = random.Random(3)
        compared = with_meta = repeats = 0
        for seed, (p1, p2) in enumerate([("Robot", "Mutant"), ("Mutant", "Robot")]):
            obs = call({"cmd": "new_match", "p1": p1, "p2": p2, "seed": 500 + seed, "settings": STD})
            for _turn in range(25):
                if obs["done"]:
                    break
                for seat in range(2):
                    cands = obs["candidates"][seat]
                    seen = list(cands)  # this decision's list for the seat, as the engine holds it
                    for move in call({"cmd": "toggle_pool", "seat": seat})["moves"]:
                        # pool-only values (switch sets, L1 triples) are made on a fresh panel, as a probe makes them;
                        # a move's own sets are read off the working panel, as set 4 reads them
                        listed = cands[move["candidate"]]["factors"].get("extra_set", 1)
                        for k in [k for k in move["extra_sets"] if k >= listed][:3]:
                            w = call({"cmd": "extra_set_widgets", "seat": seat, "candidate": move["candidate"],
                                      "extra_set": k})["widgets"]
                            e = call({"cmd": "expand", "specs": [{"seat": seat, "candidate": move["candidate"],
                                                                  "extra_override": {"extra_set": k}}]})["results"][0]
                            if "candidate" not in e:
                                continue
                            assert e["id"] == len(seen)
                            seen.append(e["candidate"])
                            call({"cmd": "preview", "pairs": [[rng.randrange(len(obs["candidates"][0])),
                                                               rng.randrange(len(obs["candidates"][1]))]]})
                            p = call({"cmd": "probe", "probes": [{"seat": seat, "action": move["action"],
                                                                  "extra": {str(i): v for i, v in w}}]})["results"][0]
                            drop = ("DI", "reverse", "feint", "prediction")
                            got = {key: v for key, v in p["candidate"]["extra"].items() if key not in drop}
                            want = {key: v for key, v in e["candidate"]["extra"].items() if key not in drop}
                            assert got == want, (p1, p2, move["action"], k, w)
                            compared += 1
                            pc = p["candidate"]
                            assert p["id"] == len(seen)
                            same = [o for o in seen if "values" in o and json.dumps([o["action"], o["data"], o["extra"]],
                                    sort_keys=True) == json.dumps([pc["action"], pc["data"], pc["extra"]], sort_keys=True)]
                            if same:
                                o = same[0]
                                assert (pc["factors"], pc["values"]) == (o["factors"], o["values"])
                                assert pc["base"] == (o["id"] if o.get("base") is None else o["base"])
                                with_meta += 1
                            else:
                                assert "factors" not in pc and "values" not in pc
                            seen.append(pc)
                            again = call({"cmd": "expand", "specs": [{"seat": seat, "candidate": move["candidate"],
                                                                      "extra_override": {"extra_set": k}}]})["results"][0]
                            if "error" in again and again["error"].startswith("same candidate as "):
                                assert again["error"] == "same candidate as %d" % e["id"]
                                repeats += 1
                            elif "candidate" in again:
                                seen.append(again["candidate"])
                obs = call({"cmd": "step", "actions": [rng.randrange(len(obs["candidates"][0])),
                                                       rng.randrange(len(obs["candidates"][1]))]})
        assert compared > 50 and with_meta > 0 and repeats > 0, (compared, with_meta, repeats)
        # DI: a probe with DI is refused while DI is disabled, as step_raw refuses that input.
        call({"cmd": "new_match", "p1": "Cowboy", "p2": "Ninja", "seed": 1, "settings": {**STD, "di_enabled": False}})
        r = call({"cmd": "probe", "probes": [{"seat": 0, "action": "Continue", "di": {"x": 1.0, "y": 0.0}}]})
        assert r["results"][0] == {"error": "DI is disabled"}
        r = call({"cmd": "probe", "probes": [{"seat": 0, "action": "Continue", "di": {"x": 0, "y": 0}}]})
        assert "candidate" in r["results"][0]
    finally:
        engine.close()

"""Candidate rules hashes, as the server reports them in hello["candidate_rules"].

    python tools/rules_hash.py [MOD_DIR]          # set 4's hash
    python tools/rules_hash.py --all [MOD_DIR]    # set 4's, each of set 5's 16 switch combinations, then set 6's
                                                  # default, set 7's and set 7's with switch_sets
    python tools/rules_hash.py --raw ...          # the same before rules_aliases.json

md5 over a rules JSON followed by Candidates4.gd (set 4's frozen generator), each reduced to its
tokens: '#' comments (the .gd only) and every whitespace character outside string literals removed;
the first 12 hex digits. So comment and formatting edits keep the hash; any change to code, strings
or values changes it. A port computes the same function (YomiServer._rules_text, a simulator's
src/rules.rs).

Set 4: candidate_rules_v4.json. Set 5: candidate_rules_v5.json with the switches in its `set5`
object set to the combination played (`"draw_pairs": true` -> false, ...), so each combination has
its own hash; the generator text is still Candidates4.gd, as the simulator hashes it. Set 6:
candidate_rules_v6.json with set 5's and set 6's switch literals as played, and each switch outside the
16 that is on (SET6_EXTRA_KEYS, in order) appended to its `set6` object (YomiServer.set6_hash); set 7 is
set 6 with SET7_KEYS on.

Aliases: rules_aliases.json maps a computed hash to the frozen one it stands for, when the files
changed without changing a single list. Candidates4.gd lost its copy of the game's legality loop
(now the game's own ActionButtons panel), so its tokens and
the computed hashes changed while the rules did not; every set-4 and set-5 hash keeps its value.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SET5_KEYS = ("draw_pairs", "gun_aim_fine", "bullet_di", "draw_base")  # bit k of a combination
SET6_KEYS = ("pairs_ninja", "pairs_cowboy", "pairs_wizard", "pairs_mutant", "pairs_robot", "triples",
             "toggle_only_ninja", "toggle_only_cowboy", "toggle_only_wizard", "toggle_only_mutant", "toggle_only_robot",
             "clicks_ninja", "clicks_cowboy", "clicks_wizard", "clicks_mutant", "clicks_robot")
SET6_EXTRA_KEYS = ("juke_sections", "l1_pool", "xy_fine", "di_fine_all", "multi_xy_base", "multi_xy", "l1_panel_dirs",
                   "switch_sets")
SET7_KEYS = ("juke_sections", "l1_pool", "xy_fine", "di_fine_all", "multi_xy_base", "multi_xy")


def strip(text: str, comments: bool) -> str:
    out, quote, i = [], None, 0
    while i < len(text):
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            out.append(c)
        elif comments and c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif not c.isspace():
            out.append(c)
        i += 1
    return "".join(out)


RAW = False  # --raw: skip rules_aliases.json


def aliases(mod_dir: Path) -> dict[str, str]:
    f = mod_dir / "rules_aliases.json"
    return json.loads(f.read_text(encoding="utf-8"))["aliases"] if f.exists() else {}


def _hash(rules_tokens: str, mod_dir: Path) -> str:
    text = rules_tokens + strip((mod_dir / "Candidates4.gd").read_text(encoding="utf-8"), True)
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return h if RAW else aliases(mod_dir).get(h, h)


def rules_hash(mod_dir: Path) -> str:
    """Set 4's hash."""
    return _hash(strip((mod_dir / "candidate_rules_v4.json").read_text(encoding="utf-8"), False), mod_dir)


def set5_switches(bits: int) -> dict[str, bool]:
    return {k: bool(bits >> n & 1) for n, k in enumerate(SET5_KEYS)}


def rules_hash5(mod_dir: Path, switches: dict[str, bool]) -> str:
    """Set 5's hash with these switches (keys not given stay on)."""
    text = strip((mod_dir / "candidate_rules_v5.json").read_text(encoding="utf-8"), False)
    for k in SET5_KEYS:
        on = f'"{k}":true'
        assert text.count(on) == 1, f"candidate_rules_v5.json holds {on} once"
        text = text.replace(on, f'"{k}":{"true" if switches.get(k, True) else "false"}')
    return _hash(text, mod_dir)


def set6_default() -> dict[str, bool]:
    """Set 6 as released: pairs and clicks on, triples, toggle_only and the switches outside the 16 off."""
    return {k: k.startswith(("pairs_", "clicks_")) for k in SET6_KEYS} | {k: False for k in SET6_EXTRA_KEYS}


def set7(switch_sets: bool = False) -> dict[str, bool]:
    """Set 7's switches (set 6 with SET7_KEYS on), with or without switch_sets."""
    return set6_default() | {k: True for k in SET7_KEYS} | {"switch_sets": switch_sets}


def rules_hash6(mod_dir: Path, set5: dict[str, bool] | None = None, set6: dict[str, bool] | None = None) -> str:
    """Set 6's hash with these switches (set 5's not given stay on; set 6's not given keep set6_default)."""
    s5 = {k: True for k in SET5_KEYS} | (set5 or {})
    s6 = set6_default() | (set6 or {})
    text = strip((mod_dir / "candidate_rules_v6.json").read_text(encoding="utf-8"), False)
    for k in SET5_KEYS + SET6_KEYS:
        on = f'"{k}":true'
        assert text.count(on) == 1, f"candidate_rules_v6.json holds {on} once"
        text = text.replace(on, f'"{k}":{"true" if (s5 | s6)[k] else "false"}')
    tail = "".join(f',"{k}":true' for k in SET6_EXTRA_KEYS if s6.get(k))
    if tail:
        last = f'"clicks_robot":{"true" if s6["clicks_robot"] else "false"}}}'
        assert text.count(last) == 1, "candidate_rules_v6.json's set6 ends with clicks_robot"
        text = text.replace(last, last[:-1] + tail + "}")
    return _hash(text, mod_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    every = "--all" in args
    RAW = "--raw" in args
    args = [a for a in args if a not in ("--all", "--raw")]
    mod = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "mod" / "YomiBench"
    print(rules_hash(mod) if not every else f"4 {rules_hash(mod)}")
    if every:
        for b in range(16):
            on = [k for k, v in set5_switches(b).items() if v]
            print(f"5 {rules_hash5(mod, set5_switches(b))} {','.join(on) or 'none'}")
        print(f"6 {rules_hash6(mod)} default")
        print(f"7 {rules_hash6(mod, set6=set7())} set 7")
        print(f"7 {rules_hash6(mod, set6=set7(True))} set 7 + switch_sets")

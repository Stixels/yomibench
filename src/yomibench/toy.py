"""A tiny deterministic simultaneous-move fighting game that speaks the engine protocol.

It stands in for the real game so bots, search and the arena can be built and tested
without the game. It has the shape that matters: simultaneous picks, spacing, a
strike/block/grab triangle, meter, and parameterized moves.

Run as a stdio engine: `python -m yomibench.toy`.
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Any

STAGE = 30
MAX_HP = 100
TURN_CAP = 150
WALKS = (-3, -1, 1, 3)  # positive is toward the opponent

# name: (range, damage, meter gain, chip when blocked)
ATTACKS = {
    "Jab": (3, 6, 5, 0),
    "Heavy": (5, 14, 8, 1),
    "Super": (6, 28, 0, 5),
}
SUPER_COST = 50
GRAB = (2, 10)

# attacker beats defender's attack (defender's attack is interrupted)
BEATS = {("Jab", "Heavy"), ("Super", "Jab"), ("Super", "Heavy")}


def _new_state() -> dict[str, Any]:
    return {
        "tick": 0,
        "players": [
            {"hp": MAX_HP, "max_hp": MAX_HP, "x": 10, "meter": 0},
            {"hp": MAX_HP, "max_hp": MAX_HP, "x": 20, "meter": 0},
        ],
    }


def candidates(state: dict[str, Any], seat: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(action: str, data: Any = None) -> None:
        out.append({"id": len(out), "action": action, "data": data, "extra": None})

    add("Wait")  # the default move: candidate 0 (docs/PROTOCOL.md)
    for d in WALKS:
        add("Walk", {"Distance": d})
    add("Jab")
    add("Heavy")
    add("Grab")
    add("Block")
    if state["players"][seat]["meter"] >= SUPER_COST:
        add("Super")
    return out


def is_done(state: dict[str, Any]) -> bool:
    return state["tick"] >= TURN_CAP or any(p["hp"] <= 0 for p in state["players"])


def winner(state: dict[str, Any]) -> int | None:
    if not is_done(state):
        return None
    a, b = (p["hp"] for p in state["players"])
    return None if a == b else (0 if a > b else 1)


def _push(state: dict[str, Any], seat: int, amount: int) -> None:
    """Move `seat` `amount` away from the opponent, clamped to the stage."""
    me, opp = state["players"][seat], state["players"][1 - seat]
    away = -1 if me["x"] < opp["x"] else 1
    me["x"] = max(0, min(STAGE, me["x"] + away * amount))


def resolve(state: dict[str, Any], picks: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    s = copy.deepcopy(state)
    ps = s["players"]
    acts = [p["action"] for p in picks]

    # 1. movement; walking forward stops one unit short of the opponent
    for i, pick in enumerate(picks):
        if pick["action"] == "Walk":
            d = pick["data"]["Distance"]
            if d > 0:
                d = min(d, abs(ps[0]["x"] - ps[1]["x"]) - 1)
            _push(s, i, -d)
    dist = abs(ps[0]["x"] - ps[1]["x"])

    for i, a in enumerate(acts):
        if a == "Super":
            ps[i]["meter"] -= SUPER_COST

    # 2. who lands what
    hits: list[tuple[int, int]] = []  # (target, damage)
    pushes: list[tuple[int, int]] = []
    for i in (0, 1):
        a, d = acts[i], acts[1 - i]
        if a in ATTACKS:
            rng, dmg, gain, chip = ATTACKS[a]
            if dist > rng or (d, a) in BEATS:
                continue
            ps[i]["meter"] = min(100, ps[i]["meter"] + gain)
            if d == "Block":
                hits.append((1 - i, chip))
                pushes.append((1 - i, 2))
                ps[1 - i]["meter"] = min(100, ps[1 - i]["meter"] + 3)
            else:
                hits.append((1 - i, dmg))
                pushes.append((1 - i, 2))
        elif a == "Grab":
            rng, dmg = GRAB
            if dist <= rng and d not in ATTACKS and d != "Grab":
                hits.append((1 - i, dmg))
                pushes.append((1 - i, 3))

    for target, dmg in hits:
        ps[target]["hp"] = max(0, ps[target]["hp"] - dmg)
    for target, amount in pushes:
        _push(s, target, amount)

    s["tick"] += 1
    return s


class ToyEngine:
    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None
        self.handles: dict[int, dict[str, Any]] = {}  # handle -> state at a decision point
        self.next_handle = 1

    def _obs(self, state: dict[str, Any], with_candidates: bool = True) -> dict[str, Any]:
        done = is_done(state)
        obs = {
            "turn": state["tick"],
            "done": done,
            "winner": winner(state),
            "actionable": [not done, not done],
            "state": state,
        }
        if with_candidates:
            obs["candidates"] = [[] if done else candidates(state, i) for i in (0, 1)]
        return obs

    def _picks(self, state: dict[str, Any], ids: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(ids, list) or len(ids) != 2:
            raise ValueError("expected [id_p1, id_p2]")
        out = []
        for seat, cid in enumerate(ids):
            cands = candidates(state, seat)
            if not isinstance(cid, int) or not 0 <= cid < len(cands):
                raise ValueError(f"seat {seat}: no candidate {cid!r}")
            out.append(cands[cid])
        return out[0], out[1]

    def _handle(self, state: dict[str, Any], with_candidates: bool) -> dict[str, Any]:
        h = self.next_handle
        self.next_handle += 1
        self.handles[h] = state
        return {**self._obs(state, with_candidates), "handle": h}

    def _continue(self, state: dict[str, Any], pair: Any, until: Any) -> dict[str, Any]:
        if until == "decision":
            min_ticks = 0
        elif isinstance(until, dict) and isinstance(until.get("ticks"), int) and until["ticks"] >= 0:
            min_ticks = until["ticks"]
        else:
            raise ValueError('until must be "decision" or {"ticks": N}')
        start = state["tick"]
        state = resolve(state, self._picks(state, pair if pair is not None else [0, 0]))
        while state["tick"] - start < min_ticks and not is_done(state):
            state = resolve(state, self._picks(state, [0, 0]))
        return state

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            cmd = msg.get("cmd")
            if cmd == "hello":
                return {"engine": "toy", "version": "1", "characters": ["Toy"]}
            if cmd == "new_match":
                self.state = _new_state()
                self.handles.clear()
                return self._obs(self.state)
            if self.state is None:
                return _err("no_match", "send new_match first")
            want = msg.get("candidates", True) is not False
            if cmd == "observe":
                return self._obs(self.state)
            if cmd == "step":
                if is_done(self.state):
                    return _err("match_over", "the match has ended")
                self.state = resolve(self.state, self._picks(self.state, msg["actions"]))
                self.handles.clear()
                return self._obs(self.state)
            if cmd == "fast_forward":  # replay recorded picks (candidate dicts) from the current state
                for pair in msg["inputs"]:
                    picks = []
                    for seat, want in enumerate(pair):
                        key = (want["action"], want.get("data"), want.get("extra"))
                        found = [c for c in candidates(self.state, seat) if (c["action"], c["data"], c["extra"]) == key]
                        if not found:
                            return _err("bad_request", f"seat {seat}: no candidate {key!r}")
                        picks.append(found[0])
                    self.state = resolve(self.state, (picks[0], picks[1]))
                self.handles.clear()
                return self._obs(self.state)
            if cmd == "preview":
                if is_done(self.state):
                    return _err("match_over", "the match has ended")
                return {"results": [self._handle(resolve(self.state, self._picks(self.state, p)), want)
                                    for p in msg["pairs"]]}
            if cmd == "continue":
                start = self.handles.get(msg.get("handle"))
                if start is None:
                    return _err("stale_handle", f"unknown or stale handle {msg.get('handle')!r}")
                if is_done(start):
                    return _err("match_over", "that handle's match has ended")
                return self._handle(self._continue(start, msg.get("pair"), msg.get("until", "decision")), want)
            if cmd == "release":
                n = sum(self.handles.pop(h, None) is not None for h in msg.get("handles", []))
                return {"released": n}
            return _err("unknown_cmd", repr(cmd))
        except (KeyError, ValueError, TypeError) as e:
            return _err("bad_request", str(e))


def _err(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def main() -> None:
    engine = ToyEngine()
    for line in sys.stdin:
        if line.strip():
            sys.stdout.write(json.dumps(engine.handle(json.loads(line))) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

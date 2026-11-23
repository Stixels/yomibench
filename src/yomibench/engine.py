"""Engine connections: in-process, subprocess (stdio) or TCP, all speaking docs/PROTOCOL.md."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Protocol, Union


# The most candidate pairs one preview request to an engine carries. A bot can ask for far more in
# one call (matrix-do's whole-table fallback previews every kept pair when the double oracle doesn't
# converge: 40k+ pairs on big tables, far more on rare states), and an engine holds per-pair buffers
# for the whole call (older engine builds: ~15 KB a pair, ~8 GB for one 500k-pair call). Both
# engines' `preview` split a bigger call into consecutive requests of at most this many pairs and
# join the results in order. Read at call time, so tests can lower it.
MAX_PREVIEW_PAIRS = 20_000


def preview_chunks(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """`pairs` as consecutive slices of at most MAX_PREVIEW_PAIRS, in order. A call within the
    limit (an empty one too) is one slice: the same request as before chunking."""
    n = max(1, MAX_PREVIEW_PAIRS)
    return [pairs[k:k + n] for k in range(0, len(pairs), n)] or [pairs]


class EngineError(RuntimeError):
    """The engine answered with an error object."""


class EngineCrashed(RuntimeError):
    """The engine process died or the connection dropped."""


class Transport(Protocol):
    def request(self, msg: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


class InProcess:
    def __init__(self, handler: Any):
        self.handler = handler

    def request(self, msg: dict[str, Any]) -> dict[str, Any]:
        return self.handler.handle(msg)

    def close(self) -> None:
        pass


class _LineTransport:
    def _send_line(self, line: bytes) -> None: ...
    def _read_line(self) -> bytes: ...

    def request(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            self._send_line(json.dumps(msg, separators=(",", ":")).encode() + b"\n")
            line = self._read_line()
            if not line:
                raise EngineCrashed("engine closed the connection")
            return json.loads(line)
        except (OSError, ValueError) as e:  # ValueError covers a truncated or garbled line
            raise EngineCrashed(str(e)) from e


class Subprocess(_LineTransport):
    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _send_line(self, line: bytes) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def _read_line(self) -> bytes:
        assert self.proc.stdout
        return self.proc.stdout.readline()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class Tcp(_LineTransport):
    def __init__(self, host: str, port: int, timeout: float = 600.0, connect_wait: float = 30.0):
        # A supervised server may be restarting after a crash: retry refused connects for a while.
        deadline = time.monotonic() + connect_wait
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=timeout)
                break
            except ConnectionRefusedError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        self.file = self.sock.makefile("rb")

    def _send_line(self, line: bytes) -> None:
        self.sock.sendall(line)

    def _read_line(self) -> bytes:
        return self.file.readline()

    def close(self) -> None:
        self.file.close()
        self.sock.close()


@dataclass(frozen=True)
class Candidate:
    id: int
    action: str
    data: Any
    extra: Any
    # Candidate set 4: the factors `expand` may override on this candidate, e.g.
    # {"di": 9, "feint": true, "reverse": true, "extra_set": 3} (docs/PROTOCOL.md), and this
    # candidate's own factor values ({"di": 0, "feint": false, ...}); None on engines without them.
    factors: dict[str, Any] | None = None
    values: dict[str, Any] | None = None
    base: int | None = None  # for an expanded candidate: the candidate id it overrides

    @staticmethod
    def parse(c: dict[str, Any]) -> "Candidate":
        return Candidate(c["id"], c["action"], c.get("data"), c.get("extra"), c.get("factors"), c.get("values"),
                         c.get("base"))

    def record(self) -> dict[str, Any]:
        """What game records keep of a candidate (ids are only positions)."""
        out = {"action": self.action, "data": self.data, "extra": self.extra}
        if self.factors is not None:
            out["factors"] = self.factors
        if self.values is not None:
            out["values"] = self.values
        if self.base is not None:
            out["base"] = self.base
        return out


@dataclass(frozen=True)
class Observation:
    turn: int
    done: bool
    winner: int | None
    state: dict[str, Any]
    candidates: tuple[tuple[Candidate, ...], tuple[Candidate, ...]]
    actionable: tuple[bool, bool] = (True, True)
    handle: int | None = None  # set on preview / continue results

    @staticmethod
    def parse(raw: dict[str, Any]) -> "Observation":
        cands = tuple(
            tuple(Candidate.parse(c) for c in seat)
            for seat in raw.get("candidates", ([], []))
        )
        assert len(cands) == 2
        actionable = tuple(raw.get("actionable", (True, True)))
        return Observation(
            raw["turn"], raw["done"], raw.get("winner"), raw["state"], cands,  # type: ignore[arg-type]
            actionable, raw.get("handle"),  # type: ignore[arg-type]
        )


Until = Union[str, int]  # "decision", or a minimum number of game ticks


class Engine:
    """Typed client over a transport. Bots get this to call `preview` and `continue_from`."""

    def __init__(self, transport: Transport, owned: Any = None):
        self.transport = transport
        self._owned = owned  # e.g. a game server process this engine started
        # Per-turn results shared by both seats' bots (e.g. one search serving both); cleared on
        # every step / new_match.
        self.turn_cache: dict[str, Any] = {}
        self.match_seed: int | None = None
        # Running totals for the metrics ledger (cheap: one clock read per preview call). A call is
        # one bot request, however many engine requests it is split into; `max_preview_pairs` is
        # the largest request (pairs) since it was last zeroed (play_recorded zeroes it per game).
        self.stats = {"preview_calls": 0, "preview_pairs": 0, "preview_seconds": 0.0, "max_preview_pairs": 0}
        # The candidate set asked for at connect (see connect): sent with every new_match that
        # doesn't name its own, and reported by hello() as the set played.
        self.candidate_set: int | None = None
        # The match settings asked for at connect (see connect), the same way: sent with every
        # new_match that doesn't name its own, and reported by hello() as the settings played.
        self.match_settings: dict[str, Any] | None = None

    def _call(self, msg: dict[str, Any]) -> dict[str, Any]:
        resp = self.transport.request(msg)
        if "error" in resp:
            err = resp["error"]
            raise EngineError(f"{err.get('code')}: {err.get('message')}")
        return resp

    def hello(self) -> dict[str, Any]:
        resp = self._call({"cmd": "hello"})
        if self.candidate_set is not None and str(resp.get("candidate_set")) != str(self.candidate_set):
            # Records, pool checks and the ledger key on hello's candidate_set: report the set
            # every match is asked to play, and keep the engine's own default beside it.
            resp = {**resp, "candidate_set": str(self.candidate_set), "default_candidate_set": resp.get("candidate_set")}
        if self.match_settings:
            # The same for match settings (records only train together on equal ones: compat).
            standard = resp.get("match_settings") or {}
            resp = {**resp, "match_settings": {**standard, **self.match_settings}, "default_match_settings": standard}
        return resp

    def new_match(self, p1: str, p2: str, seed: int, max_candidates: int | None = None,
                  candidate_sets: tuple[int | None, int | None] | None = None,
                  settings: dict[str, Any] | None = None) -> Observation:
        """`max_candidates` overrides the engine's per-seat budget for this match (0: every variant).
        `candidate_sets` picks each seat's candidate rules (engines that support several, e.g. the
        real game's 4 and 2, for A/B arenas). `settings` overrides match settings (hello's
        `match_settings` keys, e.g. stage_width; default: the ones asked for at connect)."""
        self.turn_cache = {}
        self.match_seed = seed  # bots may key per-match randomness on it
        msg: dict[str, Any] = {"cmd": "new_match", "p1": p1, "p2": p2, "seed": seed}
        if max_candidates is not None:
            msg["max_candidates"] = max_candidates
        if candidate_sets is None and self.candidate_set is not None:
            candidate_sets = (self.candidate_set, self.candidate_set)
        if candidate_sets is not None:
            msg["candidate_sets"] = list(candidate_sets)
        if settings is None:
            settings = self.match_settings
        if settings:
            msg["settings"] = settings
        return Observation.parse(self._call(msg))

    def expand(self, specs: list[dict[str, Any]]) -> list[Candidate | str]:
        """Candidate set 4: add candidates to this decision that override factors of existing ones.
        Each spec is {"seat": 0|1, "candidate": id, "extra_override": {"di": k, "feint": bool,
        "reverse": bool, "extra_set": k}}; the engine builds each through the game's widgets and
        appends it to the seat's list (its id is the next position). Returns the new candidates, or
        the engine's reason for a spec it refused, in order."""
        results = self._call({"cmd": "expand", "specs": specs})["results"]
        return [Candidate.parse(r["candidate"]) if "candidate" in r else str(r.get("error")) for r in results]

    def observe(self) -> Observation:
        return Observation.parse(self._call({"cmd": "observe"}))

    def step(self, a1: int, a2: int) -> Observation:
        self.turn_cache = {}
        return Observation.parse(self._call({"cmd": "step", "actions": [a1, a2]}))

    def fast_forward(self, inputs: list[list[dict[str, Any]]], seed: int | None = None) -> Observation:
        """Sampled-state starts (yomibench.sampled_starts): from the current match's opening, play each
        decision's recorded P1 and P2 candidates (dicts as records keep them) in order, standing at the
        decision after the last; the observation there. `seed` (the game's own) then keys the match's
        per-match randomness on our side (`match_seed`), the restored state keeping the source's."""
        self.turn_cache = {}
        resp = self._call({"cmd": "fast_forward", "inputs": inputs})
        if seed is not None:
            self.match_seed = seed
        return Observation.parse(resp) if "state" in resp else self.observe()

    def step_raw(self, inputs: list[dict[str, Any]], raw_option_by_id: bool = False) -> Observation:
        """Engines that follow recorded games only: play P1's and P2's inputs ({action, data, extra} each) exactly as given,
        on or off the candidate grid, and answer like `step` (human play follows a real game). `raw_option_by_id`: the real
        game's mod checks an option widget's data against the panel's output for the recorded id, not its default item
        (YomiMatch.step_raw); off sends the request unchanged."""
        self.turn_cache = {}
        msg: dict[str, Any] = {"cmd": "step_raw", "inputs": inputs}
        if raw_option_by_id:
            msg["raw_option_by_id"] = True
        return Observation.parse(self._call(msg))

    def preview(self, pairs: list[tuple[int, int]], candidates: bool = True) -> list[Observation]:
        """One observation (with a `handle`) per pair. `candidates=False` skips the candidate
        lists, which is cheaper when only `state` is needed. Sent in requests of at most
        MAX_PREVIEW_PAIRS pairs (`preview_chunks`); each pair's result depends only on the current
        state and that pair, so the joined results are the one big request's."""
        t = time.perf_counter()
        results: list[dict[str, Any]] = []
        for chunk in preview_chunks(pairs):
            msg = {"cmd": "preview", "pairs": [list(p) for p in chunk], "candidates": candidates}
            results += self._call(msg)["results"]
        self.stats["preview_calls"] += 1
        self.stats["preview_pairs"] += len(pairs)
        self.stats["preview_seconds"] += time.perf_counter() - t
        self.stats["max_preview_pairs"] = max(self.stats.get("max_preview_pairs", 0), len(pairs))
        return [Observation.parse(r) for r in results]

    def continue_from(
        self, handle: int, pair: tuple[int, int] | None = None, until: Until = "decision", candidates: bool = True
    ) -> Observation:
        """Simulate on from a preview/continue handle. `pair` indexes that handle's candidates;
        None plays both seats' default (candidate 0). An int `until` keeps playing defaults until
        at least that many ticks have passed, then stops at the next decision point."""
        msg: dict[str, Any] = {
            "cmd": "continue", "handle": handle, "pair": list(pair) if pair is not None else None,
            "until": until if isinstance(until, str) else {"ticks": until}, "candidates": candidates,
        }
        return Observation.parse(self._call(msg))

    def save_replay(self, path: "str | os.PathLike[str]") -> dict[str, Any]:
        """Write the current match (real game engine only) as a game replay file at `path` on
        this machine; the game's replay browser plays it back. Returns {tick, hp, version}."""
        import base64
        from pathlib import Path

        resp = self._call({"cmd": "save_replay"})
        Path(path).write_bytes(base64.b64decode(resp.pop("replay_b64")))
        return resp

    def release(self, handles: list[int]) -> int:
        return self._call({"cmd": "release", "handles": handles})["released"]

    def close(self) -> None:
        self.transport.close()
        if self._owned is not None:
            self._owned.close()


def expand_specs(text: str) -> list[str]:
    """Engine specs from a comma list. `tcp:HOST:7777-7784` is one spec per port, and `@FILE` reads
    more specs from a file (one per line or comma-separated, `#` comments), e.g. a pool file listing
    every machine's servers."""
    out: list[str] = []
    for item in (s.strip() for s in text.split(",")):
        if not item:
            continue
        if item.startswith("@"):
            from pathlib import Path

            lines = Path(item[1:]).expanduser().read_text(encoding="utf-8").splitlines()
            out += expand_specs(",".join(line.split("#", 1)[0] for line in lines))
        elif item.startswith("tcp:") and "-" in item.rsplit(":", 1)[1]:
            head, ports = item.rsplit(":", 1)
            lo, hi = (int(p) for p in ports.split("-"))
            out += [f"{head}:{p}" for p in range(lo, hi + 1)]
        else:
            out.append(item)
    return out


SET5_KEYS = ("draw_pairs", "gun_aim_fine", "bullet_di", "draw_base")  # set 5's switches (A, B, C, R)


def set5_switches(spec: str | None = None) -> dict[str, bool] | None:
    """Candidate set 5's switches for an in-process engine plugin, from `spec`
    or else $YOMISIM_SET5: comma-separated `name=0|1` (e.g. `gun_aim_fine=0` plays set 5 without B,
    `draw_base=0` without R). Switches not named stay on. None when neither is set, so builds and
    runs that never use set 5 are untouched. The rules hash in hello() names what was played."""
    import os

    text = spec if spec is not None else os.environ.get("YOMISIM_SET5", "")
    if not text.strip():
        return None
    out: dict[str, bool] = {}
    for part in text.split(","):
        name, sep, val = part.strip().partition("=")
        if name not in SET5_KEYS or not sep or val not in ("0", "1"):
            raise ValueError(f"set-5 switches are name=0|1 with names {SET5_KEYS}: {part!r}")
        out[name] = val == "1"
    return out


def set5_spec(value: Any) -> str:
    """Candidate set 5's switches in $YOMISIM_SET5's form, every switch named in SET5_KEYS order
    (e.g. "draw_pairs=1,gun_aim_fine=0,bullet_di=1,draw_base=1"), from that form (unnamed switches
    on), "all" or "none", or a {name: bool} mapping (as a fleet manifest's `flags.set5` may hold it).
    Equal switches give the same text, so it can be compared and passed on. ValueError otherwise."""
    if isinstance(value, dict):
        bad = [k for k, v in value.items() if k not in SET5_KEYS or not isinstance(v, (bool, int))]
        if bad:
            raise ValueError(f"set-5 switches are {{name: bool}} with names {SET5_KEYS}: {bad}")
        on = {k: bool(value.get(k, True)) for k in SET5_KEYS}
    elif isinstance(value, str) and value.strip().lower() in ("all", "none"):
        on = {k: value.strip().lower() == "all" for k in SET5_KEYS}
    elif isinstance(value, str) and value.strip():
        on = {k: True for k in SET5_KEYS}
        on.update(set5_switches(value) or {})
    else:
        raise ValueError(f"not set-5 switches: {value!r}")
    return ",".join(f"{k}={int(on[k])}" for k in SET5_KEYS)


def connect(spec: str, candidate_set: int | None = None, match_settings: dict[str, Any] | None = None) -> Engine:
    """`toy`, `tcp:HOST:PORT`, `game` (start a real-game server for this engine; see
    yomibench.game), `game-pool[:N|auto]` (N headless copies of the game as one engine), `cmd:<shell-split
    command>`, or an engine a plugin registers (yomibench.plugins).

    `candidate_set` (None: the engine's default) is the candidate set every match plays: the sim
    plugin is built with it (a build without that set refuses it), and any other engine is asked for it
    in each new_match (`candidate_sets` for both seats). Either way hello() reports it.

    `match_settings` (None: the engine's standard ones) overrides match settings in every match,
    e.g. {"stage_width": 600, "game_length": 3000, "starting_meter": 1.5}; hello() reports the
    settings played. A number given for a setting the engine keeps as a decimal string (a simulator's
    fixed-point starting_meter) is sent as that string. The engine is asked for one match with
    them here, so a setting it refuses fails now, not as a dropped game per seed."""
    engine = _connect(spec, candidate_set)
    if candidate_set is not None and spec.split(":", 1)[0] not in plugin_engines():  # plugins are built with it
        engine.candidate_set = candidate_set
    if match_settings:
        try:
            hello = engine.hello()
            engine.match_settings = _settings_like(match_settings, hello.get("match_settings") or {})
            first = (hello.get("characters") or ["Toy"])[0]
            engine.new_match(first, first, 0)
        except EngineError as e:
            engine.close()
            raise EngineError(f"{spec} refused match settings {match_settings}: {e}") from e
    return engine


def _settings_like(settings: dict[str, Any], standard: dict[str, Any]) -> dict[str, Any]:
    """`settings` in the types of the engine's `standard` ones: a number for a setting kept as a
    string (a decimal such as starting_meter "0") is sent as its decimal string."""
    out = {}
    for k, v in settings.items():
        if isinstance(standard.get(k), str) and isinstance(v, (int, float)) and not isinstance(v, bool):
            v = str(v)
        out[k] = v
    return out


def _connect(spec: str, candidate_set: int | None) -> Engine:
    if spec == "toy":
        from yomibench.toy import ToyEngine

        return Engine(InProcess(ToyEngine()))
    if spec.startswith("tcp:"):
        _, host, port = spec.split(":")
        return Engine(Tcp(host, int(port)))
    if spec.startswith("game-pool"):
        from yomibench.pool_engine import connect_pool

        return connect_pool(spec)
    if spec == "game" or spec.startswith("game:"):
        from yomibench.game import GameServer

        server = GameServer(game_dir=spec[5:] or None)
        from yomibench.game import connect_server

        return connect_server(server.port, owned=server)
    if spec.startswith("cmd:"):
        import shlex

        return Engine(Subprocess(shlex.split(spec[4:])))
    name = spec.split(":", 1)[0]
    factory = plugin_engines().get(name)
    if factory is not None:
        return factory(spec, candidate_set)
    if name in KNOWN_PLUGIN_NAMES:
        raise EngineError(f"engine {name!r} isn't available in this build; use game-pool:auto (see docs/SETUP.md)")
    raise ValueError(f"unknown engine spec {spec!r}")


# Engines beyond the public ones (toy, tcp, game, game-pool, cmd) come from plugin modules (yomibench.plugins):
# each has ENGINES = {name: factory(spec, candidate_set) -> Engine}, serving `name` and `name:...`.
from yomibench.plugins import KNOWN_NAMES as KNOWN_PLUGIN_NAMES  # noqa: E402  (a clear error, not "unknown spec")


def plugin_engines() -> dict[str, Any]:
    from yomibench import plugins

    return plugins.engines()

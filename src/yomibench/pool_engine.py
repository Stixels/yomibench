"""One engine made of N real-game servers (the game as the engine, docs/PLAYING.md).

Human play can use the player's own installed game as the bot's engine instead of a simulator: N headless copies
(`yomibench.game.GameServer`, muted and windowless, each ~250 MB) all hold the same match, and each turn's
previews are split across them. Every call that changes the match state (`new_match`, `step`, `step_raw`,
`expand`) goes to all N, so their states and candidate lists stay identical; `observe` and `hello` read
server 0. A preview's handle is remapped to a pool-wide id, so `continue_from` and `release` reach the server
that holds it.

    connect("game-pool:4")      # 4 servers from the default game install
    connect("game-pool:auto")   # sized to this PC's cores and free memory (auto_size)
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from yomibench.engine import Candidate, Engine, EngineError, Observation, Until

SERVER_MB = 300          # a headless game server's memory, with headroom (measured ~250 MB)
MAX_SERVERS = 4


def free_memory_mb() -> int | None:
    """Available physical memory, or None if unknown."""
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            # Commit headroom too: a PC can show free RAM while its commit charge is near the limit.
            return int(min(m.ullAvailPhys, m.ullAvailPageFile) // (1 << 20))
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1 << 20))
    except (ValueError, OSError, AttributeError):
        return None


def auto_size(cores: int | None = None, free_mb: int | None = None, cap: int = MAX_SERVERS) -> int:
    """How many headless servers this PC can give the bot: half the logical cores beyond two (the game
    and the bot server keep theirs), and no more than free memory allows with 1 GB kept back; 1 to cap."""
    cores = cores if cores is not None else (os.cpu_count() or 2)
    free_mb = free_mb if free_mb is not None else free_memory_mb()
    by_cpu = max(1, (cores - 2) // 2)
    by_mem = cap if free_mb is None else max(1, (free_mb - 1024) // SERVER_MB)
    return max(1, min(cap, by_cpu, by_mem))


class PoolEngine(Engine):
    """An Engine over N engines holding the same match (see the module docstring)."""

    reusable = True  # human play keeps it across games (new_match resets every server)

    def __init__(self, engines: list[Engine]):
        if not engines:
            raise ValueError("a pool needs at least one engine")
        super().__init__(engines[0].transport)
        self.engines = engines
        self._ex = ThreadPoolExecutor(max_workers=len(engines), thread_name_prefix="pool")
        self._handles: dict[int, tuple[int, int]] = {}  # pool handle -> (server, its handle)
        self._next = 1

    # candidate_set / match_settings are set on the pool by connect(); every server must play them
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in ("candidate_set", "match_settings") and "engines" in self.__dict__:
            for e in self.engines:
                setattr(e, name, value)

    def _all(self, fn):
        """fn(engine) on every server at once; server 0's result (all must agree)."""
        results = list(self._ex.map(fn, self.engines))
        return results[0]

    def _wrap(self, k: int, obs: Observation) -> Observation:
        if obs.handle is None:
            return obs
        h = self._next
        self._next += 1
        self._handles[h] = (k, obs.handle)
        return Observation(obs.turn, obs.done, obs.winner, obs.state, obs.candidates, obs.actionable, h)

    def hello(self) -> dict[str, Any]:
        r = self.engines[0].hello()
        return {**r, "engine_pool": len(self.engines)}

    def new_match(self, p1: str, p2: str, seed: int, max_candidates: int | None = None,
                  candidate_sets: tuple[int | None, int | None] | None = None,
                  settings: dict[str, Any] | None = None) -> Observation:
        self.turn_cache = {}
        self.match_seed = seed
        self._handles.clear()
        return self._all(lambda e: e.new_match(p1, p2, seed, max_candidates, candidate_sets, settings))

    def expand(self, specs: list[dict[str, Any]]) -> list[Candidate | str]:
        results = list(self._ex.map(lambda e: e.expand(specs), self.engines))
        first = [c.record() if isinstance(c, Candidate) else c for c in results[0]]
        for k, r in enumerate(results[1:], 1):
            if [c.record() if isinstance(c, Candidate) else c for c in r] != first:
                raise EngineError(f"pool server {k} expanded differently from server 0")
        return results[0]

    def observe(self) -> Observation:
        return self.engines[0].observe()

    def step(self, a1: int, a2: int) -> Observation:
        self.turn_cache = {}
        self._release_all()
        return self._all(lambda e: e.step(a1, a2))

    def step_raw(self, inputs: list[dict[str, Any]], raw_option_by_id: bool = False) -> Observation:
        self.turn_cache = {}
        self._release_all()
        if raw_option_by_id:
            return self._all(lambda e: e.step_raw(inputs, raw_option_by_id=True))
        return self._all(lambda e: e.step_raw(inputs))

    def preview(self, pairs: list[tuple[int, int]], candidates: bool = True) -> list[Observation]:
        import time

        t = time.perf_counter()
        n = len(self.engines)
        size = -(-len(pairs) // n) if pairs else 0
        slices = [(k, pairs[k * size:(k + 1) * size]) for k in range(n)] if size else [(0, [])]
        parts = list(self._ex.map(lambda ks: (ks[0], self.engines[ks[0]].preview(ks[1], candidates)
                                              if ks[1] else []), slices))
        out = [self._wrap(k, o) for k, obs in parts for o in obs]
        self.stats["preview_calls"] += 1
        self.stats["preview_pairs"] += len(pairs)
        self.stats["preview_seconds"] += time.perf_counter() - t
        self.stats["max_preview_pairs"] = max(self.stats.get("max_preview_pairs", 0), len(pairs))
        return out

    def continue_from(self, handle: int, pair: tuple[int, int] | None = None, until: Until = "decision",
                      candidates: bool = True) -> Observation:
        k, h = self._handles[handle]
        return self._wrap(k, self.engines[k].continue_from(h, pair, until, candidates))

    def release(self, handles: list[int]) -> int:
        by: dict[int, list[int]] = {}
        for h in handles:
            if h in self._handles:
                k, own = self._handles.pop(h)
                by.setdefault(k, []).append(own)
        return sum(self.engines[k].release(hs) for k, hs in by.items())

    def _release_all(self) -> None:
        # The servers drop their own preview handles on step; forget ours.
        self._handles.clear()

    def save_replay(self, path):
        return self.engines[0].save_replay(path)

    def close(self) -> None:
        for e in self.engines:
            try:
                e.close()
            except Exception:  # noqa: BLE001  close the rest anyway
                pass
        self._ex.shutdown(wait=False)


def connect_pool(spec: str) -> PoolEngine:
    """`game-pool:N` or `game-pool:auto`, optionally `:GAME_DIR` after it (e.g. `game-pool:4:D:/Games/Yomi`).
    Servers start one after another (the game's native code dislikes simultaneous starts) at below-normal
    priority, muted and windowless, and die with this process (yomibench.game)."""
    from yomibench.game import GameServer, connect_server
    from yomibench.engine import Tcp

    rest = spec.split(":", 1)[1] if ":" in spec else "auto"
    count_s, _, game_dir = rest.partition(":")
    n = auto_size() if count_s in ("", "auto") else max(1, int(count_s))
    extra = [a for a in os.environ.get("YOMIBENCH_SERVER_ARGS", "").split() if a]
    # Its own scratch copy of the game: never share a run dir with `yomibench serve` or another tool (they
    # rewrite the scripts there).
    from pathlib import Path

    run_dir = Path(os.environ.get("YOMIBENCH_POOL_RUN_DIR") or Path.home() / ".yomibench" / "pool-run")
    engines: list[Engine] = []
    try:
        for _ in range(n):
            server = GameServer(game_dir=game_dir or None, run_dir=run_dir, extra_args=extra, below_normal=True)
            engines.append(connect_server(server.port, owned=server))
    except Exception:
        for e in engines:
            e.close()
        raise
    return PoolEngine(engines)

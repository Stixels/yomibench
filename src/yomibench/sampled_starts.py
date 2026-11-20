"""Sampled-state starts: a seeded share of self-play games start from a decision of a recent record.

A production worker plays most games from the opening. With `sampled_starts` on (round-3 manifest flag
`sampled_starts: {"share": 0.10, "window_hours": 12}`; selfplay `--sampled-starts JSON
--sampled-starts-pool DIR`), a share of game slots instead start from a state sampled from the worker's
own recent ranked records:

- Pairs. Game seeds 2m and 2m+1 are one pair (`pair_of`). A pair is sampled when a fixed hash of m falls
  under `share` (`selected`: nothing else, so shards, processes and reruns agree). Both games of a pair
  play the same start: no seat swap (the restored state fixes P1 and P2), and each game keeps its own
  seed, so its exploration draws (selfplay's per-seed RNGs, the bot's `engine.match_seed`) differ. Two
  games with the same bot and the same seeds would play the same continuation, since the restored
  state's RNG is the source game's.
- Source pool (`StartPool`): the record files in the pool folders (the round-3 worker keeps its finished
  chunks' records in DATA_ROOT/start-pool/, pruned to `window_hours` and `max_records`) written in the
  last `window_hours`, whose engine `hello` matches this engine's on the protocol and game versions,
  the candidate set, the candidate rules (`candidate_rules`) and the match settings. In production that
  means records of the ranked preset the worker plays; records of other rules or settings are not
  eligible. Records that are themselves sampled starts are never sources (their rows don't begin at the
  opening). No hub download: a worker samples its own games.
- The start. With a seeded RNG of the pair: a source uniformly over the eligible files (sorted by name),
  then k uniform in [1, n-3] for its n decisions (a source with n < 5 is skipped: `short_source`). The
  engine replays the source's opening and its first k decisions' picked candidates (`prefix_inputs`)
  with `fast_forward`; the state it then stands at must equal the source's decision k state
  (`canon_json` of both), otherwise the slot plays a normal game from the opening (`mismatch`). An
  engine without fast_forward, or one that refuses it, plays a normal game too (`error`).
- Records. A sampled-start game's header has `start` = {"kind": "sampled", "source_seed",
  "source_sha256" (of the source record file), "k", "prefix_inputs"}: the record replays alone (new
  match on source_seed with the header's characters, then fast_forward(prefix_inputs)). Its rows begin
  at decision k (the true decision index of row r is k + r: `start_offset`); its first row's state
  carries the match's `raw_start` (the training vocabulary's start values), as an opening game's first
  row does. Games from the opening have no `start` in their header.

Gates, screens and the control never use sampled starts: they play tools/pool_eval.py, which doesn't
read this flag (tools/round3_loop.py refuses a pool_eval command line naming it). Training reads
sampled-start records like any other, at weight 1; `train --exclude-sampled-starts` (or the record
filter's "exclude_sampled_starts": true) leaves them out of every split, e.g. for a holdout.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SALT = "sampled-start"
DEFAULT_WINDOW_HOURS = 12.0
DEFAULT_MAX_RECORDS = 1000
POOL_DIR = "start-pool"  # under a round-3 worker's data root
MIN_DECISIONS = 5  # a source needs at least this many decisions (k in [1, n-3])
# hello fields a source record's engine must share with this engine (match settings are compared by value).
ENGINE_KEYS = ("version", "game_version", "candidate_set", "candidate_rules")
# The skip reasons counted in selfplay stats["sampled_starts"] (besides "games" and "slots").
SKIPS = ("no_source", "short_source", "mismatch", "error")


def parse_settings(spec: Any) -> dict[str, Any] | None:
    """The `sampled_starts` flag checked, defaults filled in: {"share", "window_hours", "max_records"}, or
    None when off (absent, empty, or share 0). Accepts a dict or its JSON text. ValueError for a bad one."""
    if spec is None or spec == "" or spec == {}:
        return None
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError as e:
            raise ValueError(f"sampled_starts {spec!r} isn't JSON ({e})") from e
    if not isinstance(spec, dict):
        raise ValueError(f"sampled_starts {spec!r} isn't a JSON object")
    unknown = sorted(set(spec) - {"share", "window_hours", "max_records"})
    if unknown:
        raise ValueError(f"sampled_starts: unknown keys {unknown} (known: share, window_hours, max_records)")

    def number(key: str, default: float) -> float:
        v = spec.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"sampled_starts: {key} {v!r} isn't a finite number")
        return float(v)
    share = number("share", 0.0)
    if not 0 <= share <= 1:
        raise ValueError(f"sampled_starts: share {share} isn't in [0, 1]")
    if share == 0:
        return None
    window = number("window_hours", DEFAULT_WINDOW_HOURS)
    if window <= 0:
        raise ValueError(f"sampled_starts: window_hours {window} must be positive")
    most = spec.get("max_records", DEFAULT_MAX_RECORDS)
    if isinstance(most, bool) or not isinstance(most, int) or most < 1:
        raise ValueError(f"sampled_starts: max_records {most!r} must be a positive integer")
    return {"share": share, "window_hours": window, "max_records": most}


def settings_tag(settings: dict[str, Any]) -> str:
    """A ledger tag value for the settings, e.g. "share:0.1,window_hours:12"."""
    out = f"share:{settings['share']:g},window_hours:{settings['window_hours']:g}"
    if settings.get("max_records", DEFAULT_MAX_RECORDS) != DEFAULT_MAX_RECORDS:
        out += f",max_records:{settings['max_records']}"
    return out


def canon_json(value: Any) -> str:
    """Sorted-keys compact JSON: the comparison form of a state."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def pair_of(seed: int) -> int:
    return int(seed) // 2


def selected(seed: int, share: float) -> bool:
    """Whether game `seed`'s pair starts from a sampled state: a fixed hash of the pair under `share`."""
    if share >= 1:
        return True
    if share <= 0:
        return False
    h = int(hashlib.sha256(f"{SALT}:{pair_of(seed)}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return h < share


def start_offset(header: dict[str, Any]) -> int:
    """The true decision index of a record's first row: the header's start k for a sampled start, else 0."""
    start = header.get("start") if isinstance(header, dict) else None
    if isinstance(start, dict) and start.get("kind") == "sampled":
        return int(start.get("k") or 0)
    return 0


def is_sampled(header: dict[str, Any]) -> bool:
    start = header.get("start") if isinstance(header, dict) else None
    return isinstance(start, dict) and start.get("kind") == "sampled"


def _same_settings(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    from yomibench.compat import _canon

    a, b = a or {}, b or {}
    return all(_canon(a.get(k)) == _canon(b.get(k)) for k in set(a) | set(b))


def same_engine(record_engine: dict[str, Any] | None, hello: dict[str, Any]) -> bool:
    """A source record's engine (its header's `engine`, the hello it played under) plays what `hello` plays."""
    eng = record_engine or {}
    return (all(eng.get(k) == hello.get(k) for k in ENGINE_KEYS)
            and _same_settings(eng.get("match_settings"), hello.get("match_settings")))


INPUT_KEYS = ("action", "data", "extra")  # an input as the engine takes it (Engine.step_raw's form)


def prefix_inputs(rows: list[dict[str, Any]], k: int) -> list[list[dict[str, Any]]]:
    """The first k decisions' picked candidates, P1's and P2's: each the record's candidate dict cut to
    {action, data, extra} (a set-4+ record also keeps `factors`, `values` and an expansion's `base`, which
    describe the candidate, not the input; `extra` already holds an expansion's overridden factors)."""
    return [[{key: row["candidates"][s][row["picks"][s]].get(key) for key in INPUT_KEYS} for s in (0, 1)]
            for row in rows[:k]]


@dataclass(frozen=True)
class Start:
    """One sampled start: where the game begins and how to get there."""
    source: Path
    source_seed: int
    source_sha256: str
    k: int
    characters: tuple[str, str]
    prefix: list[list[dict[str, Any]]]
    state: dict[str, Any]          # the source's decision-k state: what fast_forward must reach
    raw_start: Any = None          # the match's raw start values (the source's first row), for the record

    def header(self) -> dict[str, Any]:
        return {"kind": "sampled", "source_seed": self.source_seed, "source_sha256": self.source_sha256,
                "k": self.k, "prefix_inputs": self.prefix}


def _read_header(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.loads(f.readline())
    except (OSError, ValueError, EOFError):
        return None


class StartPool:
    """The eligible source records of one self-play process, listed once per engine `hello` (bind)."""

    def __init__(self, settings: dict[str, Any], dirs: Iterable[Path], now: float | None = None):
        self.settings = settings
        self.share = float(settings["share"])
        self.dirs = [Path(d) for d in dirs]
        self.now = now
        self._lock = threading.Lock()
        self._bound: str | None = None
        self.files: list[Path] = []
        self.listing: dict[str, int] = {}
        self._cache: tuple[Path, str, dict[str, Any], list[dict[str, Any]]] | None = None

    def bind(self, hello: dict[str, Any]) -> list[Path]:
        """List the eligible records for an engine with this `hello` (once per distinct hello)."""
        key = canon_json({k: hello.get(k) for k in (*ENGINE_KEYS, "match_settings")})
        with self._lock:
            if self._bound != key:
                self.files, self.listing = self._eligible(hello)
                self._bound = key
            return self.files

    def _eligible(self, hello: dict[str, Any]) -> tuple[list[Path], dict[str, int]]:
        since = (self.now if self.now is not None else time.time()) - self.settings["window_hours"] * 3600
        counts = {"files": 0, "old": 0, "unreadable": 0, "sampled": 0, "other_engine": 0, "eligible": 0}
        out = []
        for d in self.dirs:
            for p in sorted(d.glob("g*.jsonl.gz")) if d.is_dir() else []:
                counts["files"] += 1
                try:
                    if p.stat().st_mtime < since:
                        counts["old"] += 1
                        continue
                except OSError:
                    counts["unreadable"] += 1
                    continue
                h = _read_header(p)
                if h is None:
                    counts["unreadable"] += 1
                elif h.get("start"):
                    counts["sampled"] += 1
                elif not same_engine(h.get("engine"), hello):
                    counts["other_engine"] += 1
                else:
                    out.append(p)
        out.sort(key=lambda p: p.name)
        counts["eligible"] = len(out)
        return out, counts

    def _load(self, path: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        with self._lock:
            if self._cache is not None and self._cache[0] == path:  # a pair's second game: the same source
                return self._cache[1:]  # type: ignore[return-value]
        data = path.read_bytes()
        lines = [json.loads(x) for x in gzip.decompress(data).decode("utf-8").splitlines() if x.strip()]
        header, rows = lines[0], [x for x in lines[1:] if "result" not in x]
        out = (hashlib.sha256(data).hexdigest(), header, rows)
        with self._lock:
            self._cache = (path, *out)
        return out

    def pick(self, seed: int) -> tuple[Start | None, str | None]:
        """(start, None) for a sampled slot with a usable source; (None, reason) for a sampled slot
        without one (`no_source`, `short_source`, `error`); (None, None) for a slot that plays the opening.
        Call bind first."""
        if not selected(seed, self.share):
            return None, None
        files = self.files
        if not files:
            return None, "no_source"
        rng = random.Random(f"{SALT}:{pair_of(seed)}")
        path = files[rng.randrange(len(files))]
        try:
            sha, header, rows = self._load(path)
        except (OSError, ValueError, EOFError) as e:
            print(f"sampled start: source {path.name} unreadable ({e}); playing from the opening", flush=True)
            return None, "error"
        n = len(rows)
        if n < MIN_DECISIONS:
            return None, "short_source"
        k = rng.randint(1, n - 3)
        return Start(path, int(header["seed"]), sha, k, tuple(header["characters"]),  # type: ignore[arg-type]
                     prefix_inputs(rows, k), rows[k]["state"], (rows[0].get("state") or {}).get("raw_start")), None


def begin(engine: Any, start: Start, seed: int, max_candidates: int | None = None) -> tuple[Any, str | None]:
    """Put `engine` at `start`: a new match on the source's seed and characters, then fast_forward through
    its prefix; the game's own `seed` then keys the engine's per-match randomness. Returns (observation,
    None), or (None, "mismatch") when the state reached isn't the source's decision-k state, or (None,
    "error") when the engine can't fast-forward (the caller then plays a normal game: its new_match
    resets the engine). EngineCrashed propagates (the game is dropped as any crashed game)."""
    from yomibench.engine import EngineError

    try:
        engine.new_match(start.characters[0], start.characters[1], start.source_seed, max_candidates)
        obs = engine.fast_forward(start.prefix, seed=seed)
    except (EngineError, AttributeError, NotImplementedError, TypeError, ValueError, KeyError) as e:
        print(f"sampled start: fast_forward failed ({type(e).__name__}: {e}); playing from the opening", flush=True)
        return None, "error"
    if canon_json(obs.state) != canon_json(start.state):
        print(f"sampled start: state after fast_forward to decision {start.k} of {start.source.name} differs from "
              "the record's; playing from the opening", flush=True)
        return None, "mismatch"
    return obs, None


def retain(records: Iterable[Path], pool: Path, settings: dict[str, Any] | None, now: float | None = None) -> dict[str, int]:
    """The round-3 worker's pool upkeep after a chunk: move the chunk's finished `records` (already tarred)
    into `pool` while `settings` is on, then prune the pool to the last `window_hours` and the newest
    `max_records` files (with `settings` off, the pool is emptied: its files are copies of tarred records).
    Returns {"moved", "pruned", "kept"}."""
    import os

    moved = pruned = 0
    if settings:
        pool.mkdir(parents=True, exist_ok=True)
        for f in records:
            try:
                os.replace(f, pool / f.name)
                moved += 1
            except OSError:
                pass
    if not pool.is_dir():
        return {"moved": 0, "pruned": 0, "kept": 0}
    since = (now if now is not None else time.time()) - (settings["window_hours"] * 3600 if settings else 0)
    files = []
    for p in pool.glob("g*.jsonl.gz"):
        try:
            files.append((p.stat().st_mtime, p.name, p))
        except OSError:
            pass
    files.sort(reverse=True)  # newest first
    most = settings["max_records"] if settings else 0
    kept = 0
    for i, (mtime, _, p) in enumerate(files):
        if not settings or mtime < since or i >= most:
            try:
                p.unlink()
                pruned += 1
            except OSError:
                pass
        else:
            kept += 1
    return {"moved": moved, "pruned": pruned, "kept": kept}

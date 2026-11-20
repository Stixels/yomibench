"""The metrics ledger: one JSON row per measured run, appended to metrics/ledger.jsonl.

Self-play, training and arena runs append a row when they finish (see below for the
schema), so game-generation speed, training cost and model strength can be compared across rounds
and across optimizations (tag rows `experiment=... variant=before|after` with `--tag`).

Appending never fails the run that measured it: errors are printed and swallowed. Everything here
is standard library; numpy and psutil are used only when the caller already has them.
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

SCHEMA = 2
KINDS = ("selfplay", "train", "arena")
SOURCES = ("measured", "backfill", "reported")
VALIDATION = ("unchecked", "passed", "failed", "inconclusive")

# Metric keys every row of a kind carries. A value may be null only if "metrics.<key>" is listed in
# the row's "unknown".
METRICS: dict[str, tuple[str, ...]] = {
    "selfplay": (
        "games_written", "games_valid", "games_dropped", "games_truncated", "games_skipped",
        "wall_seconds", "games_per_hour", "turns", "turns_per_sec", "searched_turns",
        "previews_per_turn", "preview_us_mean", "solver_ms_per_turn", "game_seconds_mean",
        "engines", "procs",
    ),
    "train": (
        "train_games", "val_games", "train_positions", "val_positions", "truncated_games",
        "load_seconds", "train_seconds", "epochs", "best_epoch", "best_val_mse", "device",
        "state_v3_fraction", "peak_rss_bytes", "peak_commit_bytes", "memory_method", "phases",
    ),
    "arena": ("games", "complete_pairs", "halted", "truncated", "matchups", "ratings"),
}
PROVENANCE_COMMON = ("git_sha", "git_dirty", "patch_hash", "machine", "owner", "cpu_count", "cpu_load", "runtime",
                     "started_utc", "ended_utc", "settings", "counts", "artifacts")
PROVENANCE: dict[str, tuple[str, ...]] = {
    "selfplay": PROVENANCE_COMMON + ("engine", "models", "dataset", "solver", "exploration"),
    "train": PROVENANCE_COMMON + ("engine", "models", "dataset", "features_version", "split_seed",
                                  "generation_mix", "target_blend", "window"),
    "arena": PROVENANCE_COMMON + ("engine", "models", "solver", "exploration"),
}
COUNTS = ("requested", "completed", "excluded", "errors")
TOP = ("schema", "id", "kind", "source", "tags", "metrics", "provenance", "unknown", "validation")

REPO = Path(__file__).resolve().parents[2]


def default_path() -> Path:
    """$YOMIBENCH_LEDGER, else metrics/ledger.jsonl in this checkout (else in the current folder)."""
    env = os.environ.get("YOMIBENCH_LEDGER")
    if env:
        return Path(env)
    root = REPO if (REPO / "pyproject.toml").exists() else Path.cwd()
    return root / "metrics" / "ledger.jsonl"


# --- rows ---------------------------------------------------------------------------------------

def make_row(kind: str, metrics: dict[str, Any], provenance: dict[str, Any], *, tags: dict[str, str] | None = None,
             source: str = "measured", row_id: str | None = None, unknown: Iterable[str] = (),
             notes: str | None = None) -> dict[str, Any]:
    """A ledger row. Required keys that are missing or null are set to null and listed in
    `unknown`, so a reader never mistakes "not measured" for zero. New rows start with validation
    "unchecked" (`yomibench ledger mark` appends a correction row that changes it)."""
    metrics, provenance = dict(metrics), dict(provenance)
    unk = list(dict.fromkeys(unknown))
    if isinstance(provenance.get("counts"), dict):
        provenance["counts"] = {**{k: None for k in COUNTS}, **provenance["counts"]}
        for k in COUNTS:
            if provenance["counts"][k] is None and f"provenance.counts.{k}" not in unk:
                unk.append(f"provenance.counts.{k}")
    for section, keys, d in (("metrics", METRICS.get(kind, ()), metrics),
                             ("provenance", PROVENANCE.get(kind, ()), provenance)):
        for k in keys:
            if section == "provenance" and k == "patch_hash" and provenance.get("git_dirty") is False:
                d.setdefault(k, None)  # clean checkout: there is no patch, which is known
                continue
            if d.get(k) is None:
                d[k] = None
                if f"{section}.{k}" not in unk:
                    unk.append(f"{section}.{k}")
    row: dict[str, Any] = {
        "schema": SCHEMA, "id": row_id or f"{kind}-{uuid.uuid4().hex[:12]}", "kind": kind, "source": source,
        "tags": {str(k): str(v) for k, v in (tags or {}).items()}, "metrics": metrics, "provenance": provenance,
        "unknown": unk, "validation": {"status": "unchecked"},
    }
    if notes:
        row["notes"] = notes
    return _jsonable(row)


def append(kind: str, metrics: dict[str, Any], provenance: dict[str, Any], *, path: Path | str | None = None,
           tags: dict[str, str] | None = None, source: str = "measured", row_id: str | None = None,
           unknown: Iterable[str] = (), notes: str | None = None, quiet: bool = False) -> dict[str, Any] | None:
    """Append one row to the ledger (default: default_path()). Never raises: a ledger problem must
    not cost a finished run, so errors are printed and None is returned."""
    try:
        row = make_row(kind, metrics, provenance, tags=tags, source=source, row_id=row_id, unknown=unknown,
                       notes=notes)
        errors = validate_row(row)
        if errors:
            raise ValueError("; ".join(errors))
        write_rows([row], path)
        if not quiet:
            print(f"ledger: appended {row['kind']} row {row['id']} to {path or default_path()}", flush=True)
        return row
    except Exception as e:  # noqa: BLE001  (the run's own result matters more than its ledger row)
        print(f"ledger: could not append a {kind} row: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None


def write_rows(rows: list[dict[str, Any]], path: Path | str | None = None) -> None:
    """Append rows, each one line, in a single write (concurrent appenders don't interleave lines)."""
    path = Path(path) if path else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, separators=(",", ":"), sort_keys=False, allow_nan=False) + "\n" for r in rows)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(data)


def read(path: Path | str | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else default_path()
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_row(row: Any) -> list[str]:
    """Schema errors for one row (empty if valid)."""
    if not isinstance(row, dict):
        return ["row is not an object"]
    errs = [f"missing {k}" for k in TOP if k not in row]
    if errs:
        return errs
    if row["schema"] != SCHEMA:
        errs.append(f"schema {row['schema']!r} (expected {SCHEMA})")
    kind = row["kind"]
    if kind not in KINDS:
        errs.append(f"unknown kind {kind!r}")
    if row["source"] not in SOURCES:
        errs.append(f"unknown source {row['source']!r}")
    if not isinstance(row["id"], str) or not row["id"]:
        errs.append("id must be a nonempty string")
    if not isinstance(row["tags"], dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                    for k, v in row["tags"].items()):
        errs.append("tags must map strings to strings")
    unk = row["unknown"]
    if not isinstance(unk, list) or not all(isinstance(u, str) for u in unk):
        errs.append("unknown must be a list of strings")
        unk = []
    for section, keys in (("metrics", METRICS.get(kind, ())), ("provenance", PROVENANCE.get(kind, ()))):
        d = row[section]
        if not isinstance(d, dict):
            errs.append(f"{section} must be an object")
            continue
        for k in keys:
            if k not in d:
                errs.append(f"{section}.{k} missing")
            elif d[k] is None and f"{section}.{k}" not in unk:
                if not (section == "provenance" and k == "patch_hash" and d.get("git_dirty") is False):
                    errs.append(f"{section}.{k} is null but not listed in unknown")
    counts = row["provenance"].get("counts") if isinstance(row["provenance"], dict) else None
    if isinstance(counts, dict):
        for k in COUNTS:
            if k not in counts:
                errs.append(f"provenance.counts.{k} missing")
            elif counts[k] is None and f"provenance.counts.{k}" not in unk:
                errs.append(f"provenance.counts.{k} is null but not listed in unknown")
    v = row["validation"]
    if not isinstance(v, dict) or v.get("status") not in VALIDATION:
        errs.append(f"validation.status must be one of {VALIDATION}")
    if "corrects" in row and (not isinstance(row["corrects"], str) or row["corrects"] == row["id"]):
        errs.append("corrects must name another row's id")
    if kind == "arena" and isinstance(row["metrics"].get("matchups"), list):
        for m in row["metrics"]["matchups"]:
            if not isinstance(m, dict) or not {"a", "b", "score"} <= set(m):
                errs.append("arena matchups need a, b and score")
                break
    try:
        json.dumps(row, allow_nan=False)
    except ValueError as e:
        errs.append(f"not strict JSON: {e}")
    return errs


def _jsonable(x: Any) -> Any:
    """NaN/inf become null (strict JSON); tuples become lists; Paths become strings."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "item") and callable(x.item):  # numpy / torch scalars
        try:
            return _jsonable(x.item())
        except (TypeError, ValueError):
            pass
    return x


# --- provenance ---------------------------------------------------------------------------------

def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_from_epoch(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@functools.lru_cache(maxsize=4)
def git_info(repo: Path = REPO) -> dict[str, Any]:
    """Source commit, whether tracked files differ from it, and when they do, `patch_hash`: sha256
    of `git diff HEAD` (16 hex digits), so two dirty runs can tell whether they ran the same code.
    metrics/ is left out of both: every append changes the ledger. The frozen Windows app
    (packaging/windows) has no checkout: it reports the commit its build baked in."""
    if getattr(sys, "frozen", False):
        try:
            return json.loads(Path(__file__).with_name("_build_info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"git_sha": None, "git_dirty": None, "patch_hash": None}

    def git(*a: str) -> bytes | None:
        try:
            r = subprocess.run(["git", *a], cwd=repo, capture_output=True, timeout=20)
            return r.stdout if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    sha = git("rev-parse", "HEAD")
    sha = sha.decode().strip() if sha else None
    status = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)metrics") if sha else None
    dirty = None if status is None else bool(status.strip())
    patch = None
    if dirty:
        diff = git("diff", "HEAD", "--binary", "--", ".", ":(exclude)metrics")
        patch = hashlib.sha256(diff).hexdigest()[:16] if diff is not None else None
    return {"git_sha": sha, "git_dirty": dirty, "patch_hash": patch}


def runtime(device: str | None = None) -> dict[str, Any]:
    """Interpreter and library versions. Versions come from package metadata, so nothing heavy is
    imported; torch's CUDA build and device are read only if torch is already loaded."""
    from importlib import metadata

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    out: dict[str, Any] = {"python": platform.python_version(), "implementation": platform.python_implementation(),
                           "numpy": version("numpy"), "torch": version("torch"), "yomisim": version("yomisim"),
                           "psutil": version("psutil")}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            out["torch_cuda"] = torch.version.cuda
            out["cuda_available"] = bool(torch.cuda.is_available())
            if device:
                out["device"] = device
                if device.startswith("cuda") and torch.cuda.is_available():
                    out["device_name"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
    elif device:
        out["device"] = device
    return out


def cpu_load(window: float = 0.25) -> dict[str, Any] | None:
    """System load at the start of a run: the load average where the OS has one, else CPU busy
    percent over `window` seconds (Windows, from GetSystemTimes)."""
    try:
        if hasattr(os, "getloadavg"):
            return {"loadavg": [round(x, 2) for x in os.getloadavg()]}
        try:
            import psutil  # type: ignore[import-not-found]

            return {"cpu_percent": psutil.cpu_percent(interval=window), "window_s": window}
        except ImportError:
            pass
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            def times() -> tuple[int, int]:
                idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
                v = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime  # noqa: E731
                return v(idle), v(kernel) + v(user)  # kernel time includes idle time
            i0, t0 = times()
            time.sleep(window)
            i1, t1 = times()
            busy = 100.0 * (1 - (i1 - i0) / max(1, t1 - t0))
            return {"cpu_percent": round(busy, 1), "window_s": window}
    except Exception:  # noqa: BLE001
        pass
    return None


def file_hash(path: Path | str) -> str | None:
    """sha256 of a file, first 16 hex digits (None if unreadable)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:16]
    except OSError:
        return None


def models_from_specs(specs: Iterable[str]) -> dict[str, Any]:
    """{bot spec: {path, sha256, name, features}} for specs that name a model file (`bot:MODEL`)."""
    out: dict[str, Any] = {}
    for spec in specs:
        _, _, model = spec.partition(":")
        if model:
            out[spec] = model_info(model)
    return out


def model_info(path: Path | str) -> dict[str, Any]:
    try:
        size: int | None = os.path.getsize(path)
    except OSError:
        size = None
    info: dict[str, Any] = {"path": str(path), "sha256": file_hash(path), "bytes": size}
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
        info.update(name=meta.get("name"), features=meta.get("features"))
    except Exception:  # noqa: BLE001
        pass
    return info


def dataset_manifest(dirs: Iterable[Path | str], pattern: str = "g*.jsonl.gz") -> dict[str, Any]:
    """Which games a run read or wrote: folders, game count, bytes, and a hash of the sorted
    (file name, size) list, so two rows can tell whether they saw the same dataset."""
    names: list[tuple[str, int]] = []
    per_dir: dict[str, int] = {}
    dirs = [Path(d) for d in dirs]
    for d in dirs:
        n0 = len(names)
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file() and Path(e.name).match(pattern):
                        names.append((e.name, e.stat().st_size))
        except OSError:
            pass
        per_dir[str(d)] = len(names) - n0
    names.sort()
    h = hashlib.sha256("".join(f"{n}\t{s}\n" for n, s in names).encode()).hexdigest()[:16]
    return {"dirs": [str(d) for d in dirs], "games": len(names), "bytes": sum(s for _, s in names),
            "files_hash": h, "games_per_dir": per_dir}


# hello keys that may carry an engine build stamp, first match wins. A simulator plugin stamps its build from
# main 16364c8 on; the names are matched loosely, and every stamp-like field is also kept verbatim
# in engine.build so nothing is lost if a name differs.
_COMMIT_KEYS = ("commit", "git_commit", "yomisim_commit", "build_commit", "git_sha")
_SCENE_KEYS = ("scene_digest", "scene_sha256", "scene_hash", "scene_md5")
_CONTRACT_KEYS = ("contract_hashes", "contracts", "contract_hash", "contract")
_CANDIDATE_RULES_KEYS = ("candidate_rules", "candidate_rules_hash", "candidate_set_hash", "candidates_hash", "candidate_hash")
_STAMP_WORDS = ("commit", "digest", "hash", "contract", "build", "sha")


def engine_info(hello: dict[str, Any] | None) -> dict[str, Any] | None:
    """The engine fields from a `hello`: name (yomi, a plugin's, toy), versions, rules, build stamps
    (commit, scene-dump digest, contract hashes, candidate rules hash; null when not sent)."""
    if not hello:
        return None
    rules = hello.get("match_settings")
    out = {k: hello.get(k) for k in ("engine", "version", "game_version", "mod_md5", "candidate_set", "copy_mode",
                                     "max_candidates")}
    build = {k: v for k, v in hello.items() if any(w in k.lower() for w in _STAMP_WORDS)}
    nested = build.get("build") if isinstance(build.get("build"), dict) else {}
    pick = lambda keys: next((src[k] for src in (hello, nested) for k in keys if src.get(k) is not None),  # noqa: E731
                             None)
    out["commit"] = pick(_COMMIT_KEYS)
    out["scene_digest"] = pick(_SCENE_KEYS)
    out["contract_hashes"] = pick(_CONTRACT_KEYS)
    out["candidate_rules_hash"] = pick(_CANDIDATE_RULES_KEYS)
    out["build"] = build or None
    out["rules_hash"] = (hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:16]
                         if rules is not None else None)
    return out


def header_engine(dirs: Iterable[Path | str]) -> dict[str, Any] | None:
    """The engine `hello` recorded in the first game file found in `dirs` (None if none)."""
    import gzip

    for d in dirs:
        for p in sorted(Path(d).glob("g*.jsonl.gz"))[:1]:
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    return engine_info(json.loads(f.readline()).get("engine"))
            except (OSError, ValueError):
                return None
    return None


def provenance(*, settings: dict[str, Any], started_utc: str, load: dict[str, Any] | None = None,
               ended_utc: str | None = None, owner: str | None = None, device: str | None = None,
               **extra: Any) -> dict[str, Any]:
    """The fields every row carries, plus `extra` (engine, models, dataset, counts, artifacts, ...)."""
    return {
        **git_info(),
        "machine": platform.node() or None,
        "owner": owner or default_owner(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "cpu_load": load,
        "runtime": runtime(device),
        "argv": sys.argv[1:],
        "settings": _jsonable({k: v for k, v in settings.items() if k not in ("func",)}),
        "started_utc": started_utc,
        "ended_utc": ended_utc or utc_now(),
        **extra,
    }


def default_owner() -> str | None:
    return os.environ.get("YOMIBENCH_OWNER") or platform.node() or None


def counts(requested: int | None, completed: int | None, excluded: dict[str, int | None] | None,
           errors: int | None) -> dict[str, Any]:
    """Requested / completed / excluded / errors, kept apart (excluded broken down by reason)."""
    reasons = dict(excluded or {})
    total = None if excluded is None or any(v is None for v in reasons.values()) else sum(reasons.values())
    return {"requested": requested, "completed": completed, "excluded": total, "excluded_by_reason": reasons,
            "errors": errors}


def solver_settings(bots: Iterable[Any]) -> dict[str, Any]:
    """{"implementation": "native" | "python" (the regret-matching loop that ran; native is the
    yomisolve extension), bot name: bot.config()}: search mode, iterations, tolerance, oracle
    settings, value function."""
    from yomibench.search import SOLVER

    out: dict[str, Any] = {"implementation": SOLVER}
    for b in bots:
        try:
            out[b.name] = b.config() if hasattr(b, "config") else {"bot": b.name}
        except Exception:  # noqa: BLE001
            out[getattr(b, "name", "?")] = None
    return out


_EXPLORATION_WORDS = ("opening", "epsilon", "temperature", "softmax", "explor", "noise", "factor")


def exploration_from(args: Any) -> dict[str, Any]:
    """Every exploration option the command has (opening noise, softmax / factor epsilons, ...),
    picked by name so options added on other branches are recorded too; {} when it has none."""
    return {k: v for k, v in sorted(vars(args).items()) if any(w in k for w in _EXPLORATION_WORDS)}


# --- command-line plumbing ----------------------------------------------------------------------

def add_args(parser: Any) -> None:
    parser.add_argument("--ledger", type=Path, default=None,
                        help="metrics ledger to append this run's row to (default: metrics/ledger.jsonl "
                             "in this checkout, or $YOMIBENCH_LEDGER)")
    parser.add_argument("--no-ledger", action="store_true", help="don't append a metrics row")
    parser.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                        help="label the ledger row, e.g. --tag experiment=batch-selfplay --tag variant=before "
                             "(repeatable)")
    parser.add_argument("--owner", default=None,
                        help="who ran this, for the ledger row (default: $YOMIBENCH_OWNER, else the machine name)")
    parser.add_argument("--log", type=Path, default=None,
                        help="where this run's console output is being saved, recorded in the ledger row "
                             "(default: $YOMIBENCH_LOG)")


def parse_tags(items: Iterable[str]) -> dict[str, str]:
    out = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"--tag wants KEY=VALUE, got {item!r}")
        out[k.strip()] = v.strip()
    return out


def log_from_args(args: Any) -> str | None:
    log = getattr(args, "log", None) or os.environ.get("YOMIBENCH_LOG")
    return str(log) if log else None


def path_from_args(args: Any) -> Path | None:
    """The ledger to append to, or None with --no-ledger."""
    if getattr(args, "no_ledger", False):
        return None
    return args.ledger or default_path()


# --- memory -------------------------------------------------------------------------------------

class MemorySampler:
    """Peak memory of this run, sampled every `interval` seconds on a daemon thread.

    With psutil: peak RSS and peak private (commit) bytes summed over this process and its
    children (training's record-parsing workers). Without psutil, on Windows: this process's own
    OS-tracked peaks (PeakWorkingSetSize, PeakPagefileUsage = peak commit), workers excluded, plus
    the lowest system commit headroom seen. Elsewhere: ru_maxrss of this process; commit unknown."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_rss: int | None = None
        self.peak_commit: int | None = None
        self.min_commit_headroom: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # type: ignore[import-not-found]

            self._psutil = psutil
            self._proc = psutil.Process()
        except ImportError:
            self._psutil = None
        self.method = ("psutil-process-tree" if self._psutil else
                       "os-peak-main-process" if sys.platform == "win32" or _has_rusage() else "unknown")

    def start(self) -> "MemorySampler":
        try:
            self._sample()
            self._thread = threading.Thread(target=self._run, daemon=True, name="ledger-memory")
            self._thread.start()
        except Exception:  # noqa: BLE001
            self.method = "unknown"
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:  # noqa: BLE001
                pass

    def _sample(self) -> None:
        if self._psutil:
            rss = commit = 0
            for p in [self._proc, *self._proc.children(recursive=True)]:
                try:
                    m = p.memory_info()
                    rss += m.rss
                    commit += getattr(m, "private", None) or getattr(m, "vms", 0)
                except (self._psutil.Error, OSError):
                    pass
            self.peak_rss = max(self.peak_rss or 0, rss)
            self.peak_commit = max(self.peak_commit or 0, commit)
        if sys.platform == "win32":
            headroom = _win_commit_headroom()
            if headroom is not None:
                self.min_commit_headroom = headroom if self.min_commit_headroom is None else \
                    min(self.min_commit_headroom, headroom)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(2 * self.interval + 1)
        try:
            self._sample()
        except Exception:  # noqa: BLE001
            pass
        if not self._psutil and self.method != "unknown":
            if sys.platform == "win32":
                self.peak_rss, self.peak_commit = _win_process_peaks()
            else:
                self.peak_rss = _rusage_peak()
        return {"peak_rss_bytes": self.peak_rss, "peak_commit_bytes": self.peak_commit,
                "min_commit_headroom_bytes": self.min_commit_headroom, "memory_method": self.method}


def _has_rusage() -> bool:
    try:
        import resource  # noqa: F401

        return True
    except ImportError:
        return False


def _rusage_peak() -> int | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except Exception:  # noqa: BLE001
        return None


def _win_process_peaks() -> tuple[int | None, int | None]:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        get = ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if not get(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None, None
        return int(c.PeakWorkingSetSize), int(c.PeakPagefileUsage)
    except Exception:  # noqa: BLE001
        return None, None


def _win_commit_headroom() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PI(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                        ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                        ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                        ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD),
                        ("ThreadCount", wintypes.DWORD)]
        pi = PI()
        pi.cb = ctypes.sizeof(PI)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
            return None
        return int((pi.CommitLimit - pi.CommitTotal) * pi.PageSize)
    except Exception:  # noqa: BLE001
        return None


# --- self-play ----------------------------------------------------------------------------------

def new_selfplay_stats() -> dict[str, Any]:
    """Counters run_selfplay fills in as games finish (sums, so shards merge by adding; the
    `MAX_STATS` keys merge by max). `by_matchup` holds per-matchup counters ("P1:P2" -> games,
    searched, fallback_turns, max_preview_pairs), merged the same way."""
    return {"written": 0, "dropped": 0, "truncated": 0, "skipped": 0, "turns": 0, "game_seconds": 0.0,
            "decisions": 0, "searched": 0, "previews": 0, "preview_pairs": 0, "preview_ms": 0.0,
            "search_ms": 0.0, "bytes_slim": 0, "bytes_full": 0, "games_slim": 0, "games_full": 0,
            "fallback_turns": 0, "max_preview_pairs": 0, "by_matchup": {}, "hello": None}


MAX_STATS = frozenset({"max_preview_pairs"})  # self-play counters that merge by max, not by sum


def add_selfplay_counts(out: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    """Add `part`'s numeric counters into `out` (summed; `MAX_STATS` keys by max), recursing into
    nested dicts (`by_matchup`). Other values are skipped. Returns `out`."""
    for k, v in part.items():
        if isinstance(v, dict):
            add_selfplay_counts(out.setdefault(k, {}), v)
        elif isinstance(v, (int, float)):
            out[k] = max(out.get(k, 0), v) if k in MAX_STATS else out.get(k, 0) + v
    return out


def merge_selfplay_stats(parts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    out = new_selfplay_stats()
    for p in parts:
        out["hello"] = out["hello"] or p.get("hello")
        add_selfplay_counts(out, {k: v for k, v in p.items() if k != "hello"})
    return out


def _phase_metrics(stats: dict[str, Any], div: Any) -> dict[str, Any]:
    """Per-phase timings of a search turn (docs: sim-time-budget's ledger proposal), each in ms per turn:
    `preview_sim_ms` (time in the engine's preview calls), `net_ms` (value-net scoring: input assembly and
    forward), `solve_ms` (the matrix-game solves, search.solve_both), `expand_ms` (candidate expansion),
    `search_other_ms` (the rest of the search turn: search bookkeeping and Python glue), all per searched
    turn; `step_ms` (engine step and observe) and `record_ms` (record encoding, compression and writing)
    per decision. `oracle_loop_ms_per_turn` is `solver_ms_per_turn` under the name that says what it
    times (the whole search turn minus the previews, not the solve). Stats without the phase counters
    (from before them, or engines that don't time them) leave these out."""
    if "net_ms" not in stats:
        return {}
    searched, decisions = stats["searched"], stats.get("decisions", 0)
    phase = {k: stats.get(k, 0.0) for k in ("preview_ms", "net_ms", "solve_ms", "expand_ms")}
    return {
        "preview_sim_ms": div(phase["preview_ms"], searched),
        "net_ms": div(phase["net_ms"], searched),
        "solve_ms": div(phase["solve_ms"], searched),
        "expand_ms": div(phase["expand_ms"], searched),
        "search_other_ms": div(stats["search_ms"] - sum(phase.values()), searched),
        "step_ms": div(stats.get("step_ms", 0.0), decisions),
        "record_ms": div(stats.get("record_ms", 0.0), decisions),
        "oracle_loop_ms_per_turn": div(stats["search_ms"] - stats["preview_ms"], searched),
    }


def selfplay_metrics(stats: dict[str, Any], wall_seconds: float, engines: int, procs: int,
                     interrupted: bool = False) -> dict[str, Any]:
    written, searched = stats["written"], stats["searched"]
    valid = written - stats["truncated"]
    div = lambda a, b, k=1.0: round(k * a / b, 3) if b else None  # noqa: E731
    # With selfplay --max-proc-mem-gb: the dropped games a process's memory cap ended (part of games_dropped).
    capped = {"games_dropped_memory": stats["dropped_memory"]} if "dropped_memory" in stats else {}
    return {
        "games_written": written, "games_valid": valid, "games_dropped": stats["dropped"],
        "games_truncated": stats["truncated"], "games_skipped": stats["skipped"],
        "wall_seconds": round(wall_seconds, 3), "games_per_hour": div(valid, wall_seconds, 3600.0),
        "turns": stats["turns"], "turns_per_sec": div(stats["turns"], wall_seconds),
        "decisions": stats["decisions"], "searched_turns": searched,
        "previews_per_turn": div(stats["previews"], searched),
        "preview_us_mean": div(stats["preview_ms"], stats["preview_pairs"], 1000.0),
        "solver_ms_per_turn": div(stats["search_ms"] - stats["preview_ms"], searched),
        "search_ms_per_turn": div(stats["search_ms"], searched),
        **_phase_metrics(stats, div),
        "game_seconds_mean": div(stats["game_seconds"], written),
        # Mean compressed size of the finished game files, overall and by kind (slim or full previews).
        "bytes_per_game": div(stats.get("bytes_slim", 0) + stats.get("bytes_full", 0),
                              stats.get("games_slim", 0) + stats.get("games_full", 0)),
        "bytes_per_game_slim": div(stats.get("bytes_slim", 0), stats.get("games_slim", 0)),
        "bytes_per_game_full": div(stats.get("bytes_full", 0), stats.get("games_full", 0)),
        "full_preview_games": stats.get("games_full", 0),
        # Playout cap randomization (selfplay --cheap-fraction): searched turns that were cheap.
        "cheap_searched_turns": stats.get("cheap_searched", 0),
        "cheap_turn_fraction": div(stats.get("cheap_searched", 0), searched),
        "previews_per_cheap_turn": div(stats.get("cheap_previews", 0), stats.get("cheap_searched", 0)),
        "previews_per_full_turn": div(stats["previews"] - stats.get("cheap_previews", 0),
                                      searched - stats.get("cheap_searched", 0)),
        # Matrix-do turns whose double oracle didn't converge, so the bot previewed and solved the
        # whole (kept) candidate table instead, overall and by matchup ("P1:P2"), and the largest
        # single preview request a bot made (pairs, before the engine splits it into
        # engine.MAX_PREVIEW_PAIRS chunks). Zero/empty for stats from before these counters.
        "fallback_turns": stats.get("fallback_turns", 0),
        "fallback_turn_fraction": div(stats.get("fallback_turns", 0), searched),
        "max_preview_pairs": stats.get("max_preview_pairs", 0),
        "fallback_by_matchup": {
            m: {"games": c.get("games", 0), "searched_turns": c.get("searched", 0),
                "fallback_turns": c.get("fallback_turns", 0), "max_preview_pairs": c.get("max_preview_pairs", 0)}
            for m, c in sorted(stats.get("by_matchup", {}).items())},
        "engines": engines, "procs": procs, "interrupted": interrupted, **capped,
        # Sampled-state starts (selfplay --sampled-starts): sampled slots, games started so, and the
        # slots played from the opening instead, by reason (yomibench.sampled_starts.SKIPS).
        **({"sampled_starts": dict(stats["sampled_starts"])} if stats.get("sampled_starts") else {}),
    }


# --- training -----------------------------------------------------------------------------------

def _np(x: Any) -> Any:
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def phase_mse(phase: Any, outcome: Any, baseline: Any, pred: Any, constant: float,
              phases: Iterable[tuple[float, float, str]] = ((0.0, 1 / 3, "early"), (1 / 3, 2 / 3, "mid"),
                                                            (2 / 3, 1.01, "late"))) -> dict[str, Any]:
    """Validation MSE by game phase for the net, the hand-written `evaluate` (hp) baseline, and a
    constant predictor (`constant`: the training set's mean outcome), plus "all"."""
    import numpy as np

    phase, z, base, pred = _np(phase), _np(outcome), _np(baseline), _np(pred)
    out: dict[str, Any] = {}
    for lo, hi, name in [*phases, (-1.0, 2.0, "all")]:
        m = (phase >= lo) & (phase < hi)
        if not m.any():
            out[name] = {"n": 0, "net": None, "evaluate": None, "constant": None}
            continue
        out[name] = {
            "n": int(m.sum()),
            "net": round(float(np.mean((pred[m] - z[m]) ** 2)), 5),
            "evaluate": round(float(np.mean((base[m] - z[m]) ** 2)), 5),
            "constant": round(float(np.mean((constant - z[m]) ** 2)), 5),
            "net_sign_accuracy": round(float(np.mean(np.sign(pred[m]) == np.sign(z[m]))), 4),
        }
    return out


def mean_of(x: Any) -> float:
    a = _np(x)
    return float(a.mean()) if a.size else 0.0


# --- arena --------------------------------------------------------------------------------------

def paired_score(pair_scores: list[float], samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Mean of per-pair scores (each pair: the two seat-swapped games, averaged) with a 95%
    paired-bootstrap interval: whole pairs are resampled, so seat advantage cancels inside each."""
    n = len(pair_scores)
    if not n:
        return {"score": None, "ci95": [None, None]}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(pair_scores, k=n)) / n for _ in range(samples))
    return {"score": round(sum(pair_scores) / n, 4),
            "ci95": [round(means[int(0.025 * (samples - 1))], 4), round(means[int(0.975 * (samples - 1))], 4)]}


def arena_metrics(results: list[Any], ratings: dict[str, float] | None = None,
                  ratings_ci: dict[str, tuple[float, float]] | None = None, anchor: str | None = None,
                  max_turns: int = 1000, samples: int = 2000, seed: int = 0,
                  order: list[str] | None = None) -> dict[str, Any]:
    """Per-matchup paired scores and pool ratings from arena GameResults (or dicts of them).

    As in yomibench.arena, halted games (engine error) and truncated games (cut off by the turn
    limit: no result) are unrated, and each drops its whole seat-swapped pair; both are counted
    separately. Each matchup is reported from the view of `a`, the bot earlier in `order` (e.g. the
    --bots list, challenger first), else the alphabetically first."""
    rs = [r if not isinstance(r, dict) else _Result(r) for r in results]

    def truncated(r: Any) -> bool:
        return r.outcome == "truncated"

    from yomibench.arena import pair_games  # seat-swapped pairs, candidate-character pairs too

    rank = {n: i for i, n in enumerate(order or [])}
    by_matchup: dict[tuple[str, str], list[list[Any]]] = {}
    for games in pair_games(rs):
        names = tuple(sorted(games[0].bots, key=lambda n: (rank.get(n, len(rank)), n)))
        by_matchup.setdefault(names, []).append(games)
    matchups = []
    for (a, b), groups in sorted(by_matchup.items(), key=lambda kv: [(rank.get(n, len(rank)), n) for n in kv[0]]):
        games = [r for g in groups for r in g]
        complete = [g for g in groups if len(g) == 2 and all(r.outcome not in UNRATED for r in g)]
        scores = [sum(_score(r, a) for r in g) / 2 for g in complete]
        rated = [r for g in complete for r in g]
        finished = [r for r in games if r.outcome not in UNRATED]  # with a result, pair complete or not
        m = {"a": a, "b": b, "games": len(games), "complete_pairs": len(complete),
             "halted": sum(r.outcome == "halt" for r in games), "truncated": sum(truncated(r) for r in games),
             "wins_a": sum(_score(r, a) == 1.0 for r in rated), "wins_b": sum(_score(r, a) == 0.0 for r in rated),
             "draws": sum(_score(r, a) == 0.5 for r in rated),
             "finished_games": len(finished), "wins_a_all": sum(_score(r, a) == 1.0 for r in finished),
             "wins_b_all": sum(_score(r, a) == 0.0 for r in finished),
             "pairs_won_a": sum(s > 0.5 for s in scores), "pairs_split": sum(s == 0.5 for s in scores),
             "pairs_lost_a": sum(s < 0.5 for s in scores)}
        m.update(paired_score(scores, samples, seed))
        for seat in (0, 1):  # a's record from each seat, over every finished game
            mine = [r for r in finished if r.bots[seat] == a]
            m[f"a_as_p{seat + 1}"] = {"wins": sum(_score(r, a) == 1.0 for r in mine),
                                      "losses": sum(_score(r, a) == 0.0 for r in mine), "games": len(mine)}
        matchups.append(m)
    rating_rows = None
    if ratings is not None:
        rating_rows = {n: {"elo": e, "ci95": list((ratings_ci or {}).get(n, (None, None)))} for n, e in ratings.items()}
    return {
        "games": len(rs), "complete_pairs": sum(m["complete_pairs"] for m in matchups),
        "halted": sum(r.outcome == "halt" for r in rs), "truncated": sum(truncated(r) for r in rs),
        "matchups": matchups, "ratings": rating_rows, "rating_anchor": anchor,
        "rating_method": "bradley-terry (yomibench.arena), paired bootstrap 95% CI" if ratings is not None else None,
    }


class _Result:
    def __init__(self, d: dict[str, Any]):
        self.bots, self.characters, self.seed = tuple(d["bots"]), tuple(d["characters"]), d["seed"]
        self.outcome, self.turns = d["outcome"], d.get("turns", 0)


UNRATED = ("halt", "truncated")  # as yomibench.arena.UNRATED


def arena_counts(metrics: dict[str, Any], scheduled: int | None) -> dict[str, Any]:
    """Requested / completed / excluded / errors for an arena run: completed games have a result;
    excluded are truncated games and finished games whose seat-swapped partner has none; errors
    are engine halts."""
    finished = metrics["games"] - metrics["halted"] - metrics["truncated"]
    return counts(scheduled, finished, {"truncated": metrics["truncated"],
                                        "partner_unrated": finished - 2 * metrics["complete_pairs"]},
                  metrics["halted"])


def append_arena(results: list[Any], bots: list[Any], *, path: Path, tags: dict[str, str], settings: dict[str, Any],
                 started_utc: str, load: dict[str, Any] | None, owner: str | None, hello: dict[str, Any] | None,
                 bot_specs: Iterable[str], scheduled: int | None, artifacts: dict[str, Any],
                 exploration: dict[str, Any] | None = None, anchor: str | None = None,
                 extra_metrics: dict[str, Any] | None = None, engines: int | None = None,
                 print_matchups: bool = True) -> dict[str, Any] | None:
    """Build and append an `arena` row (paired scores, pool Bradley-Terry ratings, counts). Never
    raises."""
    try:
        from yomibench.arena import bradley_terry, bradley_terry_ci

        anchor = anchor or bots[0].name
        ratings = bradley_terry(results, anchor=anchor)
        ci = bradley_terry_ci(results, anchor=anchor)
        metrics = arena_metrics(results, ratings, ci, anchor, order=[b.name for b in bots])
        metrics.update(extra_metrics or {})
        if print_matchups:
            for m in metrics["matchups"]:
                print(f"{m['a']} v {m['b']}: paired score {m['score']} 95% CI {m['ci95']} over "
                      f"{m['complete_pairs']} complete pairs")
        results_file = artifacts.get("results")
        if results_file and Path(results_file).exists():
            artifacts = {**artifacts, "results_sha256": file_hash(results_file)}
        prov = provenance(settings=settings, started_utc=started_utc, load=load, owner=owner,
                          engine=engine_info(hello), engines=engines, models=models_from_specs(bot_specs),
                          solver=solver_settings(bots), exploration=exploration or {},
                          counts=arena_counts(metrics, scheduled), artifacts=artifacts)
        return append("arena", metrics, prov, path=path, tags=tags)
    except Exception as e:  # noqa: BLE001  (the ledger never fails the run)
        print(f"ledger: could not append an arena row: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _score(r: Any, name: str) -> float:
    if r.outcome == "draw":
        return 0.5
    winner = r.bots[0] if r.outcome == "p1" else r.bots[1]
    return 1.0 if winner == name else 0.0


# --- `yomibench ledger` -------------------------------------------------------------------------

def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "?"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def describe(row: dict[str, Any]) -> str:
    m, kind = row.get("metrics", {}), row.get("kind")
    if kind == "selfplay":
        games = m.get('games_valid') if m.get('games_valid') is not None else m.get('games_written')
        return (f"{_fmt(games, 0)} games, {_fmt(m.get('games_per_hour'), 0)}/h, "
                f"{_fmt(m.get('turns_per_sec'), 2)} turns/s, {_fmt(m.get('previews_per_turn'), 1)} previews/turn, "
                f"preview {_fmt(m.get('preview_us_mean'), 0)}us, solver {_fmt(m.get('solver_ms_per_turn'), 1)}ms/turn, "
                f"{_fmt(m.get('engines'), 0)} engines")
    if kind == "train":
        ph = m.get("phases") or {}
        late = ph.get("late") or {}
        return (f"{_fmt(m.get('train_games'), 0)}+{_fmt(m.get('val_games'), 0)} games, "
                f"load {_fmt(m.get('load_seconds'))}s, train {_fmt(m.get('train_seconds'))}s on {_fmt(m.get('device'))}, "
                f"best val {_fmt(m.get('best_val_mse'), 4)}, late net/eval {_fmt(late.get('net'), 3)}/"
                f"{_fmt(late.get('evaluate'), 3)}")
    if kind == "arena":
        parts = [f"{x['a']} v {x['b']} {_fmt(x.get('score'), 3)} [{_fmt((x.get('ci95') or [None])[0], 3)}, "
                 f"{_fmt((x.get('ci95') or [None, None])[1], 3)}] ({_fmt(x.get('complete_pairs'), 0)} pairs)"
                 for x in m.get("matchups") or []]
        return "; ".join(parts) + f"; halted {_fmt(m.get('halted'), 0)}, truncated {_fmt(m.get('truncated'), 0)}"
    return ""


def effective(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows as currently corrected: a row with `corrects: ID` replaces row ID (and any earlier
    correction of it). The file itself is never edited."""
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for r in rows:
        root = r.get("corrects") or r.get("id")
        if root not in latest:
            order.append(root)
        latest[root] = r
    return [latest[k] for k in order]


def mark(path: Path | str | None, row_id: str, status: str, note: str | None = None,
         owner: str | None = None) -> dict[str, Any]:
    """Append a correction of row `row_id` whose validation is `status` (raises if no such row)."""
    if status not in VALIDATION:
        raise ValueError(f"validation must be one of {VALIDATION}")
    current = {(r.get("corrects") or r.get("id")): r for r in read(path)}
    root = next((k for k, r in current.items() if row_id in (k, r.get("id"))), None)
    if root is None:
        raise KeyError(f"no row {row_id!r} in {path or default_path()}")
    row = json.loads(json.dumps(current[root]))
    row.update(id=f"{root}~{uuid.uuid4().hex[:6]}", corrects=root,
               validation={"status": status, "note": note, "by": owner or default_owner(), "at": utc_now()})
    errors = validate_row(row)
    if errors:
        raise ValueError("; ".join(errors))
    write_rows([row], path)
    return row


def _load_fraction(row: dict[str, Any]) -> float | None:
    """Start-of-run CPU load as a fraction of the machine (0 idle, 1 saturated)."""
    prov = row.get("provenance") or {}
    load = prov.get("cpu_load") or {}
    if load.get("cpu_percent") is not None:
        return load["cpu_percent"] / 100
    if load.get("loadavg") and prov.get("cpu_count"):
        return load["loadavg"][0] / prov["cpu_count"]
    return None


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


WORKLOAD_IGNORED = {"out", "results", "log", "owner", "cmd"}  # settings that may differ between sides


def compare(rows: list[dict[str, Any]], experiment: str, metrics: list[str] | None = None, min_reps: int = 5,
            load_tolerance: float = 0.15) -> dict[str, Any]:
    """Group an experiment's rows by variant: median and range per numeric metric, and a verdict.

    The verdict is "inconclusive" unless every variant has at least `min_reps` rows, the variants'
    runs are interleaved in time, they ran the same workload (same kind and settings, output paths
    aside), and start-of-run CPU load was recorded and differs by less than `load_tolerance`
    (fraction of the machine) between variants' medians. Otherwise "comparable"."""
    sel = [r for r in effective(rows) if r.get("tags", {}).get("experiment") == experiment]
    variants: dict[str, list[dict[str, Any]]] = {}
    for r in sel:
        variants.setdefault(r.get("tags", {}).get("variant", "?"), []).append(r)
    names = sorted(variants, key=lambda v: (v != "before", v))
    reasons: list[str] = []
    if len(names) < 2:
        reasons.append(f"{len(names)} variant(s); need two to compare")
    for v in names:
        if len(variants[v]) < min_reps:
            reasons.append(f"{v}: {len(variants[v])} rows < {min_reps} repetitions")
    if len({r.get("kind") for r in sel}) > 1:
        reasons.append("rows of different kinds")
    workloads = {json.dumps({k: x for k, x in (r.get("provenance", {}).get("settings") or {}).items()
                             if k not in WORKLOAD_IGNORED}, sort_keys=True) for r in sel}
    if len(workloads) > 1:
        reasons.append("workloads differ (settings other than output paths)")
    machines = {r.get("provenance", {}).get("machine") for r in sel}
    if len(machines) > 1:
        reasons.append(f"rows from several machines: {sorted(map(str, machines))}")
    spans = {v: sorted(r.get("provenance", {}).get("started_utc") or "" for r in variants[v]) for v in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if spans[a] and spans[b] and (spans[a][-1] < spans[b][0] or spans[b][-1] < spans[a][0]):
                reasons.append(f"{a} and {b} not interleaved in time")
    loads = {v: [x for x in map(_load_fraction, variants[v]) if x is not None] for v in names}
    if any(len(loads[v]) < len(variants[v]) for v in names):
        reasons.append("CPU load not recorded for every row (contention uncontrolled)")
    elif names and max(_median(loads[v]) for v in names) - min(_median(loads[v]) for v in names) > load_tolerance:
        reasons.append("start-of-run CPU load differs between variants by more than "
                       f"{load_tolerance:.0%} of the machine")
    keys = metrics or sorted({k for r in sel for k, x in (r.get("metrics") or {}).items()
                              if isinstance(x, (int, float)) and not isinstance(x, bool)})
    table = {}
    for k in keys:
        row = {}
        for v in names:
            xs = [r["metrics"][k] for r in variants[v] if isinstance((r.get("metrics") or {}).get(k), (int, float))
                  and not isinstance(r["metrics"][k], bool)]
            row[v] = ({"n": len(xs), "median": _median(xs), "min": min(xs), "max": max(xs)} if xs else
                      {"n": 0, "median": None, "min": None, "max": None})
        base = row[names[0]]["median"] if names else None
        for v in names[1:]:
            m = row[v]["median"]
            row[v]["ratio"] = round(m / base, 4) if m is not None and base else None
        table[k] = row
    return {"experiment": experiment, "variants": {v: len(variants[v]) for v in names},
            "verdict": "inconclusive" if reasons else "comparable", "reasons": reasons, "metrics": table}


TALLY_KEYS = ("games", "complete_pairs", "wins_a", "wins_b", "draws", "halted", "truncated")


def _arena_tally(row: dict[str, Any]) -> tuple | None:
    """An arena row's per-matchup counts (bot names left out), or None if any are unknown."""
    ms = (row.get("metrics") or {}).get("matchups") or []
    tally = tuple(tuple(m.get(k) for k in TALLY_KEYS) for m in ms)
    return tally if tally and all(v is not None for t in tally for v in t) else None


def _arena_source(row: dict[str, Any]) -> dict[str, Any]:
    """What tells two arena runs apart: results file and its hash, seed, and the bots."""
    prov = row.get("provenance") or {}
    art, settings = prov.get("artifacts") or {}, prov.get("settings") or {}
    ms = (row.get("metrics") or {}).get("matchups") or []
    return {"results file": art.get("results"), "results sha256": art.get("results_sha256"),
            "seed": settings.get("seed"), "bots": sorted({n for m in ms for n in (m.get("a"), m.get("b"))})}


def tally_findings(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(warnings, errors) for arena rows with identical tallies, as currently corrected. Different
    results files, seeds or bots make it a coincidence worth stating (a warning); no recorded
    difference at all makes it a possible duplicate (an error)."""
    arena = [(r, t) for r in effective(rows) if r.get("kind") == "arena" and (t := _arena_tally(r)) is not None]
    warnings, errors = [], []
    for i, (a, ta) in enumerate(arena):
        for b, tb in arena[i + 1:]:
            if ta != tb:
                continue
            sa, sb = _arena_source(a), _arena_source(b)
            diff = [k for k in sa if sa[k] != sb[k] and (sa[k] is not None or sb[k] is not None)]
            counts = "; ".join(f"{g} games, {p} pairs, {wa}-{wb}-{d}" for g, p, wa, wb, d, *_ in ta)
            ida, idb = a.get("corrects") or a["id"], b.get("corrects") or b["id"]
            if diff:
                warnings.append(f"{ida} and {idb} have identical tallies ({counts}) but different "
                                f"{', '.join(diff)}: distinct runs whose counts coincide, not duplicates")
            else:
                errors.append(f"{ida} and {idb} have identical tallies ({counts}) and no recorded difference in "
                              "results file, seed or bots: possible duplicate")
    return warnings, errors


def cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="yomibench ledger")
    ap.add_argument("action", choices=("summary", "validate", "mark", "compare"))
    ap.add_argument("row_id", nargs="?", help="mark: the row to mark")
    ap.add_argument("--path", type=Path, default=None, help="ledger file (default: metrics/ledger.jsonl)")
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE", help="only rows with this tag")
    ap.add_argument("--json", action="store_true", help="summary / compare: print JSON")
    ap.add_argument("--validation", choices=VALIDATION, help="mark: the row's new validation status")
    ap.add_argument("--note", help="mark: why")
    ap.add_argument("--owner", help="mark: who (default: $YOMIBENCH_OWNER, else the machine name)")
    ap.add_argument("--experiment", help="compare: the experiment tag to compare")
    ap.add_argument("--metric", action="append", default=None, help="compare: only these metrics (repeatable)")
    ap.add_argument("--min-reps", type=int, default=5, help="compare: repetitions needed per variant (default 5)")
    ap.add_argument("--strict", action="store_true",
                    help="validate: also check arena rows with identical tallies (a warning when their results "
                         "files, seeds or bots differ; an error when nothing tells them apart)")
    args = ap.parse_args(argv)
    path = args.path or default_path()
    if args.action == "mark":
        if not args.row_id or not args.validation:
            ap.error("mark needs a row id and --validation")
        try:
            row = mark(path, args.row_id, args.validation, args.note, args.owner)
        except (KeyError, ValueError) as e:
            print(e.args[0] if e.args else e)
            return 1
        print(f"appended {row['id']}: {row['corrects']} validation {args.validation}")
        return 0
    if args.action == "compare":
        if not args.experiment:
            ap.error("compare needs --experiment")
        res = compare(read(path), args.experiment, args.metric, args.min_reps)
        if args.json:
            print(json.dumps(res, indent=1))
            return 0
        print(f"experiment {res['experiment']}: " + ", ".join(f"{v} x{n}" for v, n in res["variants"].items()))
        print(f"verdict: {res['verdict']}" + "".join(f"\n  - {x}" for x in res["reasons"]))
        names = list(res["variants"])
        for k, row in res["metrics"].items():
            cells = []
            for v in names:
                c = row[v]
                if c["n"]:
                    cells.append(f"{v} {_fmt(c['median'], 3)} [{_fmt(c['min'], 3)}-{_fmt(c['max'], 3)}] n={c['n']}"
                                 + (f" ({c['ratio']:.3f}x)" if c.get("ratio") is not None else ""))
                else:
                    cells.append(f"{v} ?")
            print(f"  {k:<22} " + " | ".join(cells))
        return 0
    if args.action == "validate":
        bad = 0
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            print(f"{path}: {e}")
            return 1
        ids: set[str] = set()
        for n, line in enumerate(lines, 1):  # noqa: B007
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                print(f"{path}:{n}: not JSON: {e}")
                bad += 1
                continue
            errs = validate_row(row)
            if isinstance(row, dict) and row.get("id") in ids:
                errs.append(f"duplicate id {row.get('id')}")
            if isinstance(row, dict) and row.get("corrects") and row["corrects"] not in ids:
                errs.append(f"corrects {row['corrects']!r}, which no earlier row has")
            if isinstance(row, dict):
                ids.add(row.get("id"))
            for e in errs:
                print(f"{path}:{n}: {e}")
            bad += bool(errs)
        print(f"{path}: {len(ids)} rows, {bad} invalid")
        if args.strict:
            try:
                warnings, errors = tally_findings(read(path))
            except ValueError:
                warnings, errors = [], []  # not JSON: already reported above
            for w in warnings:
                print(f"warning: {w}")
            for e in errors:
                print(f"error: {e}")
            bad += len(errors)
        return 1 if bad else 0
    want = parse_tags(args.tag)
    rows = [r for r in effective(read(path)) if (not args.kind or r.get("kind") == args.kind)
            and all(r.get("tags", {}).get(k) == v for k, v in want.items())]
    if args.json:
        for r in rows:
            print(json.dumps(r))
        return 0
    for r in rows:
        tags = ",".join(f"{k}={v}" for k, v in r.get("tags", {}).items())
        prov = r.get("provenance", {})
        print(f"{r.get('id'):<34} {r.get('kind'):<8} {r.get('source'):<8} {prov.get('machine') or '?':<14} "
              f"{(prov.get('started_utc') or '?')[:16]:<16} [{tags}]")
        status = (r.get("validation") or {}).get("status", "?")
        print(f"    {describe(r)}" + (f"  (unknown: {len(r.get('unknown', []))})" if r.get("unknown") else "")
              + (f"  [{status}]" if status != "unchecked" else ""))
    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r.get("kind")] = by_kind.get(r.get("kind"), 0) + 1
    print(f"{len(rows)} rows: " + ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items())))
    return 0

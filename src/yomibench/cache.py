"""A packed, incrementally built training cache of encoded self-play records.

    python -m yomibench.cache build RECORD_DIR [--cache ROOT] [--workers N] [--cands K] [--append-only]
    python -m yomibench.train RECORD_DIR --out model.npz [--cache ROOT]     # reads it memory-mapped

Encoding the records (parse the JSON, run the feature encoder) was most of training's time and
its memory grew with the number of games. Here each finished `g*.jsonl.gz` is encoded once into
append-only shards under `<RECORD_DIR>/.cache/` (or `ROOT/<name>-<hash>/` with `--cache ROOT`,
for read-only data folders); a later build encodes only games that appeared since. Training opens
the shards with `np.load(mmap_mode="r")`: file-backed pages, so memory stays flat as games grow.

The cache stores every decision of every game with the raw facts training selects on (search
value, each seat's exploration mark and mode, engine-v3 state, baseline), and no split decisions:
at open, `open_cached` picks files with train.select_files and rows with train.select_rows, the
same functions train.load_samples uses, so the cached path follows the in-memory loader's rules
(held-out manifests, generations and weights, exploration targets, evaluation folders) by
construction.

Layout of a cache directory:

    manifest.json   format, encoder hash, vocab (name -> cache id per category), raw_defaults,
                    exploration modes, shards, and per source file: size, mtime, shard, start,
                    rows, outcome, final hp margin (`hp_margin`: p1's hp fraction minus p2's at
                    the end, null if unknown; games cached before it have none until a build asks
                    for it with `hp_margin=True`, as training's --aux-hp-weight does), record keys,
                    characters, truncated
    s00000/         one shard per build: nums0.npy nums1.npy (float16, checked finite and in
                    range), cats0.npy cats1.npy (the smallest unsigned int that fits the vocab),
                    value.npy (float64, NaN = no search value), baseline.npy (float64), explore.npy
                    (uint8: bit 0 p1, bit 1 p2), mode.npy (uint8: 0 none, else 1 + index into the
                    manifest's explore_modes), v3.npy (uint8), cheap.npy (uint8: 1 for a
                    playout-capped decision)

Category ids are cache ids in order of first appearance at build time; training remaps them to the
order the in-memory path would assign from its training rows (names not in training become 0), so
for the same files and settings it trains the same model up to the float16 rounding of the inputs.

Any change to the encoder (features, read_slim, evaluate source or FEATURES), a changed or
removed source file, or a character whose start values differ from the cached `raw_defaults`
discards the cache and rebuilds it. Derived data: never commit it.

One writer at a time: build and build_cands hold the cache directory's `cache.lock` (O_EXCL, holding
{pid, host, at}; a holder on this host whose PID is gone is stale and removed) from reading the
manifest to writing it, and wait for a live holder. So a fit's own build and the trainer's
encode-at-extract child (`build --append-only`, round3_loop --encode-at-extract) never write one
cache at once, and whichever comes second finds the other's games already encoded. `--append-only`
only adds games: where a rebuild is due it changes nothing and leaves the rebuild to the next fit.

Candidate lists (`cands-k<K>/`, built only when training asks for them: train --aux-cand-value-weight
or --aux-mix-weight, `open_cached(cand_k=K)`): a side cache next to the shards, for the full turns'
capped candidate lists, search values and mixes (yomibench.search_record). It has its own manifest,
format and hash (cand_hash: the candidate encoder, the cap, K), and is keyed by source file like the
main cache, so building it never touches the main cache's manifest, shards or encoder hash, and a
run without those flags never reads or builds it. Layout:

    manifest.json   format, hash, k, names (action and data-shape names, by id), table_rows, shards,
                    per source file: source, shard, start, slots (qualifying turns), truncated
    s00000/         turn.npy (int32: the turn's index in its game, per slot), flip.npy (int8, slots x 2:
                    each seat's view mirror), uid.npy (int32, slots x 2 x K: rows of the candidate table,
                    -1 padding), val.npy and prob.npy (float32, slots x 2 x K: search value, NaN for none;
                    renormalized mix probability), and the candidate table rows this shard added:
                    unums.npy (float32, rows x policy.N_CAND), uact.npy / udkey.npy (int32 name ids)

Shared windows (`open_cached(share_dir=DIR)`, train --share-window; round3_loop --share-window): the arms
of a fit round select the same window from the same caches, and selecting it (select_rows over every
turn, then the vocab pass over the training rows' category ids) is most of a cached load. open_cached
splits the load into _select_window, whose result is small (per split: the games with selected rows,
and per row its turn index and target), and _materialize, which builds the splits from it and the
caches. With `share_dir`, the window is written to `DIR/<key>/` (window.json, window.npz), and a later
run with the same key reads it instead of selecting. The key hashes what decides the rows besides the
records: the folders, cache root, value weight, exploration and quick-turn targets, multi-step
targets, generation weights and cap, record filter, allowed match settings, sample window, and the
run's starting vocabulary (--init-from). The auxiliary heads don't decide rows, so arms that differ
only in --aux-search-heads, --aux-hp-weight or the candidate heads share a window: _materialize
computes their columns (aux_search, hp_margin, cand_*) from the caches at the window's rows, with the
same functions. The window is reused only when the records it was selected from are unchanged (a hash
of each split's files and their cache entries, `state`), so a shared run trains exactly the model
the same run would without sharing; otherwise the run selects as usual and replaces the window.
`share_frozen` (train --share-window-frozen) also reuses a window written before games arrived, if
its own games' cache entries are unchanged: that run trains on the earlier window, without the newer
games. The caches are brought up to date first either way. The load's phases (train's
metrics.phase_seconds) add `share_import` and, when a window is written, `share_export`; a run that
read one has no `window_select` or `row_select`, and its `vocab` is only the lookup tables.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import shutil
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = 6  # 5: raw per-turn facts (exploration marks and modes), selection at open; 6: playout-cap marks
COLUMNS = ("nums0", "nums1", "cats0", "cats1", "value", "baseline", "explore", "mode", "v3", "cheap")
INPUTS = ("nums0", "cats0", "nums1", "cats1")
SHARD_GAMES = 1000  # games per shard file set within one build (bounds the builder's memory)


# The encoder's identity, stored in every cache manifest: a cache built under another identity is rebuilt. It
# names what encoding a record gives, not the source text: comments and docstrings don't change it. Bump it only
# when encoded rows change (that rebuilds every cache); tests/test_encoder_identity.py fails when the code it
# covers (encoder_inputs) changes, and says which to do. 853f7784eeb12ad1 was the source hash of these inputs
# before a docstring-only edit; rows encoded since are byte-identical to it.
ENCODER = "853f7784eeb12ad1"


def encoder_hash() -> str:
    """The encoder identity caches are keyed by (ENCODER)."""
    return ENCODER


def encoder_inputs() -> list[str]:
    """The code and constants that decide encoded rows: FORMAT, the feature version, the features module, how
    record keys are made (compat.record_keys), the slim record reader, DI filling, and the baseline evaluator."""
    from yomibench import compat, features, search, selfplay, valuenet

    return [f"format {FORMAT}", f"features {valuenet.FEATURES}", inspect.getsource(features),
            inspect.getsource(compat.record_keys),
            inspect.getsource(selfplay.read_slim), inspect.getsource(selfplay._fill_di),
            inspect.getsource(search.evaluate), inspect.getsource(search.terminal_value)]


def cache_dir(record_dir: Path, root: Path | None = None) -> Path:
    if root is None:
        return record_dir / ".cache"
    tag = hashlib.sha1(str(record_dir.resolve()).encode()).hexdigest()[:8]
    return root / f"{record_dir.resolve().name}-{tag}"


def _numeric_half(values, path: str, column: str) -> np.ndarray:
    # Reject before conversion: float16 silently turns out-of-range finite inputs into inf.
    numbers = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numbers).all() or (np.abs(numbers) > np.finfo(np.float16).max).any():
        raise ValueError(f"{path}: {column} contains non-finite or out-of-range float16 inputs")
    return numbers.astype(np.float16)


def encode_record(path: str) -> dict[str, Any]:
    """Worker: parse one record and encode every decision from both seats, with a private
    vocabulary; the builder maps its ids onto the cache's. Raw defaults come from the game's own
    start values (the builder checks they agree with the cache's)."""
    from yomibench.compat import held_out_arm, record_keys
    from yomibench.features import CATEGORIES, Vocab, encode
    from yomibench.search import evaluate
    from yomibench.selfplay import final_hp_margin, read_slim

    header, result, turns = read_slim(Path(path))
    if held_out_arm(header):  # a human-play game: a held-out human test, never trained on
        raise ValueError(f"{path} is a held-out record (arm {held_out_arm(header)}); it can't be cached for training")
    res = result["result"]
    out: dict[str, Any] = {"seed": header.get("seed"), "keys": list(record_keys(header)),
                           "chars": [str(c) for c in header.get("characters", [])],
                           "truncated": res["outcome"] == "truncated", "rows": 0}
    from yomibench.sampled_starts import start_offset

    if start_offset(header):  # a sampled start: its rows begin at decision k (phases; --exclude-sampled-starts)
        out["start_k"] = start_offset(header)
    if out["truncated"]:
        return out
    vocab = Vocab()
    if turns:
        vocab.learn_defaults(turns[0][0])
    cols: dict[str, list] = {k: [] for k in COLUMNS}
    modes: list[str | None] = []
    for t in turns:
        for seat in (0, 1):
            n, c = encode(t.state, seat, vocab)
            cols[f"nums{seat}"].append(n)
            cols[f"cats{seat}"].append(c)
        cols["value"].append(np.nan if t.value is None else t.value)
        cols["baseline"].append(evaluate(t.state, 0))
        cols["explore"].append(int(bool(t.explore[0])) | (int(bool(t.explore[1])) << 1))
        modes.append(t.explore_mode)
        cols["v3"].append("hitboxes" in t.state["players"][0])  # engine-v3 move fields present
        cols["cheap"].append(t.cheap)
    out.update(rows=len(turns), raw_defaults=vocab.raw_defaults, modes=modes,
               outcome=0.0 if res["winner"] is None else (1.0 if res["winner"] == 0 else -1.0),
               hp_margin=final_hp_margin(result, turns[0].state if turns else None),
               names={cat: list(vocab.ids.get(cat, {})) for cat in CATEGORIES},  # local id i+1 -> name
               nums0=_numeric_half(cols["nums0"], path, "nums0"),
               nums1=_numeric_half(cols["nums1"], path, "nums1"),
               cats0=np.array(cols["cats0"], np.int32), cats1=np.array(cols["cats1"], np.int32),
               value=np.array(cols["value"], np.float64), baseline=np.array(cols["baseline"], np.float64),
               explore=np.array(cols["explore"], np.uint8), v3=np.array(cols["v3"], np.uint8),
               cheap=np.array(cols["cheap"], np.uint8))
    return out


def _sources(record_dir: Path) -> dict[str, list[float]]:
    return {p.name: [p.stat().st_size, p.stat().st_mtime] for p in sorted(record_dir.glob("g*.jsonl.gz"))}


def _empty_manifest() -> dict[str, Any]:
    from yomibench.features import CATEGORIES

    return {"format": FORMAT, "encoder": encoder_hash(), "vocab": {c: {} for c in CATEGORIES},
            "raw_defaults": {}, "explore_modes": [], "shards": [], "files": {}}


def _load_manifest(cdir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((cdir / "manifest.json").read_text())
    except (OSError, ValueError):
        return None


def _write_manifest(cdir: Path, manifest: dict[str, Any]) -> None:
    tmp = cdir / f"manifest.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, cdir / "manifest.json")


def _smallest_uint(max_id: int) -> type:
    return np.uint8 if max_id < 256 else (np.uint16 if max_id < 65536 else np.uint32)


LOCK = "cache.lock"


def _pid_alive(pid: int) -> bool:
    """Whether process `pid` runs on this machine (never signals it: os.kill(pid, 0) ends a process on Windows)."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5  # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_holder(path: Path) -> dict | None:
    """The lock's holder record; None when free, {} while unreadable (being written)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def writer_lock(cdir: Path, log=print, poll: float = 0.5):
    """Hold `cdir`'s writer lock (module docstring) for the body, waiting while another live process holds it."""
    path = Path(cdir) / LOCK
    start, said = time.monotonic(), 0
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            h = _lock_holder(path)
            if h is None:
                continue  # released meanwhile
            stale = (not _pid_alive(int(h.get("pid") or 0)) if h and h.get("host") == socket.gethostname() else False)
            if not h:  # unreadable: its holder is writing it, or died between creating and writing it
                try:
                    stale = time.time() - path.stat().st_mtime > 60
                except FileNotFoundError:
                    continue
            if stale:
                log(f"{path}: stale lock {h or '(unreadable)'} removed")
                if _lock_holder(path) == h:
                    path.unlink(missing_ok=True)
                continue
            if time.monotonic() - start > 60 * (said + 1):
                said += 1
                log(f"{path}: waiting for the writer pid {h.get('pid')} ({h.get('role', '?')}) to finish")
            time.sleep(poll)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "role": Path(sys.argv[0] or "python").name,
                       "at": time.time()}, f)
        break
    try:
        yield
    finally:
        if (_lock_holder(path) or {}).get("pid") == os.getpid():
            path.unlink(missing_ok=True)


def build(record_dir: Path, root: Path | None = None, workers: int | None = None, log=print,
          hp_margin: bool = False, append_only: bool = False) -> Path:
    """Bring `record_dir`'s cache up to date: encode the records it doesn't have yet, under the cache's
    writer lock. Returns the cache directory. `hp_margin`: also fill in the final hp margin of games
    cached without it. `append_only`: never discard a cache (a rebuild is left to the next build without it)."""
    record_dir = Path(record_dir)
    cdir = cache_dir(record_dir, root)
    cdir.mkdir(parents=True, exist_ok=True)
    with writer_lock(cdir, log):
        _build(record_dir, cdir, workers, log, hp_margin, append_only)
    return cdir


def _build(record_dir: Path, cdir: Path, workers: int | None, log, hp_margin: bool, append_only: bool) -> None:
    from yomibench.features import CATEGORIES, CATEGORY_OF_SLOT

    sources = _sources(record_dir)
    manifest = _load_manifest(cdir)
    reason = None
    if manifest is None:
        reason = "no cache yet"
    elif manifest.get("format") != FORMAT or manifest.get("encoder") != encoder_hash():
        reason = "the encoder changed"
    else:
        changed = [n for n, meta in manifest["files"].items() if sources.get(n) != meta["source"]]
        if changed:
            reason = f"{len(changed)} source files changed or disappeared (e.g. {changed[0]})"
    if reason is not None:
        if manifest is not None:
            if append_only:
                log(f"{cdir}: not changed ({reason}; --append-only leaves the rebuild to the next fit)")
                return
            log(f"rebuilding {cdir}: {reason}")
        for p in cdir.iterdir():  # only what a cache holds: shards and the manifest
            if p.is_dir() and p.name.startswith("s"):
                shutil.rmtree(p)
            elif p.name.startswith("manifest"):
                p.unlink()
        manifest = _empty_manifest()
    if hp_margin:
        _backfill_hp_margin(record_dir, cdir, manifest, workers, log)
    todo = [n for n in sources if n not in manifest["files"]]
    if not todo:
        return
    t0 = time.perf_counter()
    vocab = manifest["vocab"]
    modes = manifest["explore_modes"]
    workers = workers or min(8, os.cpu_count() or 1)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    pool = ProcessPoolExecutor(workers) if workers > 1 and len(todo) > 1 else None
    results = pool.map(encode_record, [str(record_dir / n) for n in todo], chunksize=4) if pool else \
        map(encode_record, [str(record_dir / n) for n in todo])
    pending: list[tuple[str, dict[str, Any]]] = []

    def flush() -> None:
        if not pending:
            return
        name = f"s{len(manifest['shards']):05d}"
        tmp = cdir / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        games = [(n, r) for n, r in pending if r["rows"]]
        width = max([max(len(t) for t in vocab.values())] + [0])
        for col in COLUMNS:
            parts = [r[col] for _, r in games]
            if col.startswith("cats"):
                arr = np.concatenate(parts).astype(_smallest_uint(width)) if parts else np.zeros((0, 0), np.uint8)
            else:
                arr = np.concatenate(parts) if parts else np.zeros(0)
            np.save(tmp / f"{col}.npy", arr)
        os.replace(tmp, cdir / name)
        start = 0
        for n, r in pending:
            manifest["files"][n] = {"source": sources[n], "shard": name, "start": start, "rows": r["rows"],
                                    "seed": r["seed"], "keys": r["keys"], "chars": r["chars"],
                                    "outcome": r.get("outcome"), "truncated": r["truncated"],
                                    "hp_margin": _json_margin(r.get("hp_margin")),
                                    **({"start_k": r["start_k"]} if r.get("start_k") else {})}
            start += r["rows"]
        manifest["shards"].append({"name": name, "rows": start})
        _write_manifest(cdir, manifest)  # the shard only counts once the manifest names it
        pending.clear()

    try:
        for done, (n, r) in enumerate(zip(todo, results), 1):
            if r["rows"]:
                for char, raw in r["raw_defaults"].items():
                    have = manifest["raw_defaults"].setdefault(char, raw)
                    if have != raw:
                        raise SystemExit(f"{n}: {char}'s start values differ from earlier games; one cache (and "
                                         "one training run) can't hold both")
                luts = {}  # the worker's ids -> cache ids, by category
                for cat in CATEGORIES:
                    table = vocab[cat]
                    for name in r["names"][cat]:
                        table.setdefault(name, len(table) + 1)
                    luts[cat] = np.array([0] + [table[x] for x in r["names"][cat]], np.int64)
                for col in ("cats0", "cats1"):
                    a = r[col]
                    for slot, cat in enumerate(CATEGORY_OF_SLOT):
                        a[:, slot] = luts[cat][a[:, slot]]
                for m in r["modes"]:
                    if m is not None and m not in modes:
                        modes.append(m)
                r["mode"] = np.array([0 if m is None else modes.index(m) + 1 for m in r["modes"]], np.uint8)
            pending.append((n, r))
            if len(pending) >= SHARD_GAMES:
                flush()
            if done % 500 == 0:
                log(f"  encoded {done}/{len(todo)} games ({time.perf_counter() - t0:.0f}s)")
        flush()
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    log(f"cache {cdir}: +{len(todo)} games in {time.perf_counter() - t0:.1f}s, "
        f"{sum(s['rows'] for s in manifest['shards'])} rows in {len(manifest['shards'])} shards")


def _json_margin(m: float | None) -> float | None:
    return None if m is None or m != m else m  # NaN (unknown) is stored as null


def _backfill_hp_margin(record_dir: Path, cdir: Path, manifest: dict[str, Any], workers: int | None, log) -> None:
    """Give every cached game without a `hp_margin` entry one, read from its record's first turn and
    result line (no re-encoding), and save the manifest."""
    from yomibench.selfplay import read_hp_margin

    missing = [n for n, m in manifest["files"].items() if "hp_margin" not in m]
    if not missing:
        return
    t0 = time.perf_counter()
    workers = workers or min(8, os.cpu_count() or 1)
    paths = [str(record_dir / n) for n in missing]
    if workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(workers) as pool:
            margins = list(pool.map(read_hp_margin, paths, chunksize=16))
    else:
        margins = [read_hp_margin(Path(x)) for x in paths]
    for n, m in zip(missing, margins):
        manifest["files"][n]["hp_margin"] = None if manifest["files"][n]["truncated"] else m
    _write_manifest(cdir, manifest)
    log(f"cache {cdir}: final hp margin filled in for {len(missing)} games in {time.perf_counter() - t0:.1f}s")


CAND_FORMAT = 1
CAND_COLUMNS = ("turn", "flip", "uid", "val", "prob")  # per slot (a qualifying full turn)
CAND_TABLE = ("unums", "uact", "udkey")  # per shard: the candidate-table rows it added


def cand_hash(k: int) -> str:
    """Changes whenever the candidate side cache could hold different lists (K, the encoder, the cap)."""
    from yomibench import features, policy, search_record, selfplay

    fns = (features._num, features._move_data, policy._bucket, policy._data_bag, policy.data_key,
           policy.encode_candidate, policy.view_flip, search_record.cap_positions, search_record.turn_candidates,
           search_record.table_key, selfplay.read_slim)
    parts = [f"cand format {CAND_FORMAT}", f"k {k}", f"policy features {policy.FEATURES}", repr(policy.CAND_NUMS),
             *(inspect.getsource(f) for f in fns)]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def cand_dir(cdir: Path, k: int) -> Path:
    return Path(cdir) / f"cands-k{k}"


def encode_record_cands(job: tuple[str, int]) -> dict[str, Any]:
    """Worker: one record's qualifying full turns (search_record.turn_candidates), with a private
    candidate table; the builder maps its rows onto the side cache's."""
    from yomibench.compat import held_out_arm
    from yomibench.search_record import CandTable, SplitCands, turn_candidates
    from yomibench.selfplay import read_slim

    path, k = job
    header, result, turns = read_slim(Path(path), policy=True)
    if held_out_arm(header):
        raise ValueError(f"{path} is a held-out record (arm {held_out_arm(header)}); it can't be cached for training")
    out: dict[str, Any] = {"truncated": result["result"]["outcome"] == "truncated", "slots": 0}
    if out["truncated"]:
        return out
    table, sc, at = CandTable(), SplitCands(k), []
    name = Path(path).name
    for i, t in enumerate(turns):
        seats = turn_candidates(t, k, f"{name}:{i}")
        if seats is not None:
            at.append(i)
            sc.add_row(seats, table)
    a = sc.arrays()
    out.update(slots=len(at), turn=np.array(at, np.int32), flip=a["cand_flip"].astype(np.int8), uid=a["cand_uid"],
               val=a["cand_val"], prob=a["cand_prob"], unums=table.unums(), names=table.names)
    return out


def build_cands(record_dir: Path, cdir: Path, k: int, workers: int | None = None, log=print,
                append_only: bool = False) -> Path:
    """Bring `record_dir`'s candidate side cache (under its cache directory `cdir`) up to date: add
    the records it doesn't have yet, under the cache directory's writer lock. Returns the side cache's
    directory. `append_only`: as build."""
    record_dir, cd = Path(record_dir), cand_dir(cdir, k)
    cd.mkdir(parents=True, exist_ok=True)
    with writer_lock(cdir, log):
        _build_cands(record_dir, cd, k, workers, log, append_only)
    return cd


def _build_cands(record_dir: Path, cd: Path, k: int, workers: int | None, log, append_only: bool) -> None:
    from yomibench.policy import CAND_CATEGORIES, N_CAND

    sources = _sources(record_dir)
    manifest = _load_manifest(cd)
    reason = None
    if manifest is None:
        reason = "no candidate cache yet"
    elif manifest.get("format") != CAND_FORMAT or manifest.get("encoder") != cand_hash(k):
        reason = "the candidate encoder changed"
    else:
        changed = [n for n, meta in manifest["files"].items() if sources.get(n) != meta["source"]]
        if changed:
            reason = f"{len(changed)} source files changed or disappeared (e.g. {changed[0]})"
    if reason is not None:
        if manifest is not None:
            if append_only:
                log(f"{cd}: not changed ({reason}; --append-only leaves the rebuild to the next fit)")
                return
            log(f"rebuilding {cd}: {reason}")
        for p in cd.iterdir():
            if p.is_dir() and p.name.startswith("s"):
                shutil.rmtree(p)
            elif p.name.startswith("manifest"):
                p.unlink()
        manifest = {"format": CAND_FORMAT, "encoder": cand_hash(k), "k": k,
                    "names": {c: [] for c in CAND_CATEGORIES}, "table_rows": 0, "shards": [], "files": {}}
    todo = [n for n in sources if n not in manifest["files"]]
    if not todo:
        return
    t0 = time.perf_counter()
    names = manifest["names"]
    name_id = {c: {n: i for i, n in enumerate(names[c])} for c in CAND_CATEGORIES}
    index: dict[tuple[str, str, bytes], int] = {}  # the table so far, rebuilt from the shards
    for s in manifest["shards"]:
        un, ua, ud = (np.load(cd / s["name"] / f"{c}.npy") for c in CAND_TABLE)
        for row, a, d in zip(un, ua, ud):
            index[names["action"][a], names["dkey"][d], np.ascontiguousarray(row).tobytes()] = len(index)
    workers = workers or min(8, os.cpu_count() or 1)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    jobs = [(str(record_dir / n), k) for n in todo]
    pool = ProcessPoolExecutor(workers) if workers > 1 and len(todo) > 1 else None
    results = pool.map(encode_record_cands, jobs, chunksize=4) if pool else map(encode_record_cands, jobs)
    pending: list[tuple[str, dict[str, Any]]] = []
    new_rows: dict[str, list] = {c: [] for c in CAND_TABLE}

    def name_of(cat: str, n: str) -> int:
        if n not in name_id[cat]:
            name_id[cat][n] = len(names[cat])
            names[cat].append(n)
        return name_id[cat][n]

    def flush() -> None:
        if not pending:
            return
        name = f"s{len(manifest['shards']):05d}"
        tmp = cd / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        games = [r for _, r in pending if r["slots"]]
        empty = {"turn": (0,), "flip": (0, 2), "uid": (0, 2, k), "val": (0, 2, k), "prob": (0, 2, k)}
        dtypes = {"turn": np.int32, "flip": np.int8, "uid": np.int32, "val": np.float32, "prob": np.float32}
        for col in CAND_COLUMNS:
            arr = np.concatenate([r[col] for r in games]) if games else np.zeros(empty[col])
            np.save(tmp / f"{col}.npy", arr.astype(dtypes[col]))
        np.save(tmp / "unums.npy", np.asarray(new_rows["unums"], np.float32).reshape(-1, N_CAND))
        np.save(tmp / "uact.npy", np.asarray(new_rows["uact"], np.int32))
        np.save(tmp / "udkey.npy", np.asarray(new_rows["udkey"], np.int32))
        os.replace(tmp, cd / name)
        start = 0
        for n, r in pending:
            manifest["files"][n] = {"source": sources[n], "shard": name, "start": start, "slots": r["slots"],
                                    "truncated": r["truncated"]}
            start += r["slots"]
        manifest["shards"].append({"name": name, "slots": start, "table_rows": len(new_rows["uact"])})
        manifest["table_rows"] += len(new_rows["uact"])
        _write_manifest(cd, manifest)
        pending.clear()
        for c in CAND_TABLE:
            new_rows[c] = []

    try:
        for done, (n, r) in enumerate(zip(todo, results), 1):
            if r["slots"]:
                lut = np.empty(len(r["unums"]), np.int32)  # the worker's table rows -> the side cache's
                for j, row in enumerate(r["unums"]):
                    key = (r["names"]["action"][j], r["names"]["dkey"][j], np.ascontiguousarray(row).tobytes())
                    if key not in index:
                        index[key] = len(index)
                        new_rows["unums"].append(row)
                        new_rows["uact"].append(name_of("action", key[0]))
                        new_rows["udkey"].append(name_of("dkey", key[1]))
                    lut[j] = index[key]
                r["uid"] = np.where(r["uid"] >= 0, lut[np.maximum(r["uid"], 0)], -1).astype(np.int32)
            pending.append((n, r))
            if len(pending) >= SHARD_GAMES:
                flush()
            if done % 500 == 0:
                log(f"  candidates: {done}/{len(todo)} games ({time.perf_counter() - t0:.0f}s)")
        flush()
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    log(f"candidate cache {cd}: +{len(todo)} games in {time.perf_counter() - t0:.1f}s, "
        f"{sum(s['slots'] for s in manifest['shards'])} full turns, {manifest['table_rows']} unique candidates")


class CandCache:
    """One record folder's candidate side cache: per-slot columns memory-mapped, the table in memory."""

    def __init__(self, cd: Path, k: int):
        from yomibench.policy import N_CAND

        self.manifest = _load_manifest(cd)
        if self.manifest is None or self.manifest.get("encoder") != cand_hash(k):
            raise SystemExit(f"{cd}: missing or stale candidate cache")
        self.cols = {s["name"]: {c: np.load(cd / s["name"] / f"{c}.npy", mmap_mode="r") for c in CAND_COLUMNS}
                     for s in self.manifest["shards"]}
        tables = [[np.load(cd / s["name"] / f"{c}.npy") for c in CAND_TABLE] for s in self.manifest["shards"]]
        self.unums = np.concatenate([t[0] for t in tables]) if tables else np.zeros((0, N_CAND), np.float32)
        names = self.manifest["names"]
        self.names = {"action": [names["action"][i] for t in tables for i in t[1]],
                      "dkey": [names["dkey"][i] for t in tables for i in t[2]]}

    def slots_of(self, name: str) -> tuple[str | None, dict[int, int]]:
        """(shard, {turn index: row in the shard}) of one source file's qualifying turns."""
        g = self.manifest["files"].get(name)
        if not g or not g["slots"]:
            return None, {}
        turns = self.cols[g["shard"]]["turn"][g["start"]:g["start"] + g["slots"]]
        return g["shard"], {int(t): g["start"] + j for j, t in enumerate(turns)}


class Cache:
    """One record folder's cache, memory-mapped."""

    def __init__(self, cdir: Path, record_dir: Path):
        self.dir, self.record_dir = cdir, record_dir
        self.manifest = _load_manifest(cdir)
        if self.manifest is None or self.manifest.get("encoder") != encoder_hash():
            raise SystemExit(f"{cdir}: missing or stale cache; run python -m yomibench.cache build {record_dir}")
        self.cols = {s["name"]: {c: np.load(cdir / s["name"] / f"{c}.npy", mmap_mode="r") for c in COLUMNS}
                     for s in self.manifest["shards"]}


class _Rows:
    """A split's rows as (shard, row in shard), gathering input columns from the memory-mapped
    shards with one fancy index per shard, category ids remapped to the training vocab."""

    def __init__(self, shards: list[tuple[dict[str, np.ndarray], np.ndarray]], which: np.ndarray, local: np.ndarray):
        self.shards, self.which, self.local = shards, which, local

    def gather(self, rows: np.ndarray, col: str) -> np.ndarray:
        rows = np.asarray(rows, np.int64)
        which, local = self.which[rows], self.local[rows]
        first = self.shards[0][0][col] if self.shards else None
        shape = first.shape[1:] if first is not None else ()
        out = np.empty((len(rows), *shape), np.float32 if col.startswith("nums") else
                       (np.int64 if col.startswith("cats") else np.float64))
        for p in np.unique(which):
            sel = np.nonzero(which == p)[0]
            arrays, lut = self.shards[p]
            loc = local[sel]
            order = np.argsort(loc, kind="stable")  # page-friendly reads, then put back in order
            sel, loc = sel[order], loc[order]
            block = arrays[col][loc]
            if col.startswith("cats"):
                block = lut[np.arange(lut.shape[0])[None, :], block.astype(np.int64)]
            out[sel] = block
        return out


class LazyColumn:
    """One input column of a cached split (nums as float32, cats as int64), read on demand: slicing
    gives numpy (train._standardization streams it), indexing with a torch index tensor gives a
    tensor on that index's device (main's minibatches)."""

    def __init__(self, rows: _Rows, col: str, n: int):
        self.rows, self.col, self.n = rows, col, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.rows.gather(np.arange(*idx.indices(self.n)), self.col)
        # a torch tensor: a minibatch, on the training device (numpy 2 arrays have a .device too)
        if hasattr(idx, "device") and not isinstance(idx, np.ndarray):
            import torch

            return torch.from_numpy(self.rows.gather(idx.cpu().numpy(), self.col)).to(idx.device)
        return self.rows.gather(np.asarray(idx), self.col)


def open_cached(dirs: list[Path], vocab, val_fraction: float, seed: int, value_weight: float = 0.0,
                workers: int | None = None, device: str | None = None, allow_mixed_candidate_sets: bool = False,
                games_per_gen: int | None = None, gen_weights: list[float] | None = None,
                explore_targets: str = "uniform-only", eval_dirs: list[Path] | None = None,
                heldout_root: Path | None = None, heldout_seed: int = 0, root: Path | None = None,
                log=print, hp_margin: bool = False, quick_turns: dict[str, Any] | None = None,
                record_filter: dict[str, Any] | None = None,
                allow_match_settings: list[dict[str, Any]] | None = None, cand_k: int | None = None,
                window: dict[str, Any] | None = None, phases=None, share_dir: Path | None = None,
                share_frozen: bool = False, exclude_sampled_starts: bool = False) -> dict[str, Any]:
    """train.load_samples' result from the folders' caches (brought up to date first), with the
    same splits, rows, targets, weights and counts, and `vocab` filled the same way; the input
    columns (nums0/1, cats0/1) are LazyColumns, everything else plain arrays (y and w on `device`
    for training, as load_samples puts them). `hp_margin`: fill in the final hp margin of games
    cached without it first (training's --aux-hp-weight; else those games' `hp_margin` is NaN).
    `quick_turns`: select_rows' target options (quick turns, multi-step targets; with `aux_offsets`
    each split also has an `aux_search` column, rows x offsets, NaN where masked).
    `allow_match_settings`: as load_samples (the `variant` column and `variant_games`).
    `cand_k`: as load_samples (the training and validation splits' `cand_*` columns and `cand_table`),
    from each training folder's candidate side cache (build_cands; built or brought up to date
    first). Without it no side cache is read or built.
    `window`: as load_samples (train.sample_window after the record filter, each game's time the
    source mtime the manifest checked against the file, `sample_window` in the result). Only the
    kept games' row ranges are read afterwards: the dropped games' shard pages are never touched.
    `phases`: a train.PhaseTimer to add this load's phases to (heldout_split, cache_build, cache_open,
    eval_load, candidate_cache, share_import, window_select, row_select, vocab, share_export,
    assemble); None: not timed.

    `share_dir` (train --share-window): the window (see "Shared windows" in the module docstring) is
    read from, or written to, `share_dir/<key>/`; `share_frozen` also reuses one written before games
    arrived. `shared_window` in the result says what sharing did.

    `exclude_sampled_starts` (or the record filter's "exclude_sampled_starts": true): as load_samples
    (sampled-start records, the manifest entries with a `start_k`, left out of every split before the
    record filter; `sampled_starts_excluded` in the result). It changes the selected rows, so a shared
    window's key includes it (only when on: keys without it are main's). Kept, a sampled start's rows
    take their phase from their true decision index, (start_k + k) / (start_k + rows), both in
    select_rows' counts and in the `phase` column; start_k comes from the record itself (its header),
    so a shared window's record state already covers it."""
    from yomibench.compat import excludes_sampled_starts
    from yomibench.train import PhaseTimer, explore_cutoffs, select_files

    timer = phases if phases is not None else PhaseTimer()
    explore_cutoffs([], explore_targets)  # validates the setting before any work
    if gen_weights is not None and len(gen_weights) != len(dirs):
        raise SystemExit(f"{len(gen_weights)} generation weights for {len(dirs)} folders")
    eval_dirs = list(eval_dirs or [])
    if {Path(d).resolve() for d in eval_dirs} & {Path(d).resolve() for d in dirs}:
        raise SystemExit("an --eval-dir folder is also a training folder; evaluation games must never be trained on")
    weights = gen_weights or [1.0] * len(dirs)
    # Files first: select_files checks evaluation folders (frozen manifests, no game shared with
    # training) before any of their records is read.
    with timer("heldout_split"):
        chunks, gen_of, manifests = select_files(dirs, val_fraction, seed, games_per_gen, eval_dirs, heldout_root,
                                                 heldout_seed)
    caches = {}
    for d in dirs:
        with timer("cache_build"):  # brought up to date: new games encoded
            cdir = build(Path(d), root, workers, log, hp_margin)
        with timer("cache_open"):
            caches[Path(d).resolve()] = Cache(cdir, Path(d))
    for d in eval_dirs:
        with timer("eval_load"):  # the eval folders' caches, built and opened
            caches[Path(d).resolve()] = Cache(build(Path(d), root, workers, log, hp_margin), Path(d))
    with timer("candidate_cache"):
        cand_caches = {Path(d).resolve(): CandCache(build_cands(Path(d), caches[Path(d).resolve()].dir, cand_k, workers,
                                                                log), cand_k) for d in dirs} if cand_k else {}
    order = list(caches.values())  # a game's cache is its index here (the window's game_cache)
    no_sampled = excludes_sampled_starts(record_filter, exclude_sampled_starts)
    shared = None
    win = None
    if share_dir is not None:
        # What decides the window's rows and targets besides the records. The auxiliary heads
        # (aux_offsets, hp_margin, cand_k) don't: _materialize computes their columns per run.
        settings = {"value_weight": value_weight, "allow_mixed_candidate_sets": allow_mixed_candidate_sets,
                    "games_per_gen": games_per_gen, "gen_weights": gen_weights, "explore_targets": explore_targets,
                    "quick_turns": {k: v for k, v in (quick_turns or {}).items() if k != "aux_offsets"},
                    "record_filter": record_filter, "allow_match_settings": allow_match_settings, "window": window,
                    "dirs": [str(Path(d).resolve()) for d in dirs],
                    "eval_dirs": [str(Path(d).resolve()) for d in eval_dirs],
                    "root": str(Path(root).resolve()) if root is not None else None,
                    **({"exclude_sampled_starts": True} if no_sampled else {})}
        with timer("share_import"):  # the key, the records' state, and the window if one fits
            path = Path(share_dir) / window_key(settings, vocab)
            state = window_state(chunks, caches)
            win, shared = load_window(path, state, order, share_frozen, log)
    if win is None:
        win = _select_window(chunks, gen_of, caches, order, vocab, value_weight, explore_targets, quick_turns,
                             record_filter, allow_match_settings, window, allow_mixed_candidate_sets, log, timer,
                             no_sampled)
        if share_dir is not None:
            with timer("share_export"):
                save_window(path, win, state, manifests, order, log)
            shared = {"mode": "written", "dir": str(path)}
    else:
        vocab.ids, vocab.raw_defaults = win["vocab"]
        if shared["mode"] == "frozen":  # the window's own held-out counts, so the run's counts agree with it
            manifests = win["heldout"]
    out = _materialize(win, order, cand_caches, cand_k, vocab, len(dirs), weights, device, explore_targets,
                       quick_turns, allow_match_settings, timer)
    out["heldout"] = manifests
    for k in ("record_filter", "sample_window", "sampled_starts_excluded"):
        if win.get(k):
            out[k] = win[k]
    if shared:
        out["shared_window"] = shared
    return out


def _cache_of(caches: dict[Path, Cache]):
    """record path -> its folder's Cache (`caches` keyed by resolved folder), resolving each folder
    once: Path.resolve is a system call (about 0.2 ms on Windows), and a window has ~10^5 files."""
    by_parent: dict[Path, Cache] = {}

    def get(p) -> Cache:
        parent = Path(p).parent
        c = by_parent.get(parent)
        if c is None:
            c = by_parent[parent] = caches[parent.resolve()]
        return c
    return get


def _turns(c: Cache, g: dict[str, Any]) -> list[Any]:
    """One cached game's per-turn facts as the SlimTurns select_rows reads (no state)."""
    from yomibench.selfplay import SlimTurn

    a, lo, hi = c.cols[g["shard"]], g["start"], g["start"] + g["rows"]
    modes = c.manifest["explore_modes"]
    values, marks, mode, cheap = a["value"][lo:hi], a["explore"][lo:hi], a["mode"][lo:hi], a["cheap"][lo:hi]
    return [SlimTurn(None, None if np.isnan(v) else float(v), (bool(m & 1), bool(m & 2)), None,
                     None if md == 0 else modes[md - 1], bool(ch))
            for v, m, md, ch in zip(values, marks, mode, cheap)]


def _select_window(chunks, gen_of, caches, order, vocab, value_weight, explore_targets, quick_turns, record_filter,
                   allow_match_settings, window, allow_mixed_candidate_sets, log, timer,
                   exclude_sampled_starts: bool = False) -> dict[str, Any]:
    """The window: the record filter, the sample window and train.select_rows over the cached per-turn
    facts, then the training vocab (filled into `vocab`). Per split: the games with selected rows
    (cache index, file name, generation, variant, rows selected), per row its turn index `k` and
    target `y` (with aux_offsets also `aux`), and the counts load_samples reports. Everything else a
    split holds follows from these and the caches (_materialize)."""
    from yomibench.compat import apply_record_filter, check_keys, drop_sampled_starts
    from yomibench.features import CATEGORIES, CATEGORY_OF_SLOT
    from yomibench.train import PHASES, sample_window, select_rows, variants_of

    index = {id(c): i for i, c in enumerate(order)}
    cache_of = _cache_of(caches)

    def entry(p: Path):
        c = cache_of(p)
        return c, c.manifest["files"][Path(p).name]
    allow = {"allow_match_settings": allow_match_settings} if allow_match_settings else {}
    t0 = time.perf_counter()  # the record filter, the sample window and the compatibility checks
    no_sampled = (drop_sampled_starts(chunks, lambda p: bool(entry(p)[1].get("start_k")), log)
                  if exclude_sampled_starts else None)
    filtered = apply_record_filter(chunks, lambda p: entry(p)[1]["keys"], record_filter, log, **allow)
    windowed = sample_window(chunks, lambda p: entry(p)[1]["source"][1], lambda p: entry(p)[1]["keys"],
                             window["recent_hours"], window["budget"], window["seed"], log) if window else None
    files = [p for c in chunks.values() for p in c]
    check_keys([tuple(entry(p)[1]["keys"]) for p in files], allow_mixed_candidate_sets, **allow)
    variant_of = variants_of(files, lambda p: entry(p)[1]["keys"], allow_match_settings) or {}
    timer.add("window_select", time.perf_counter() - t0)
    t0 = time.perf_counter()

    # Rows per split, chosen by train.select_rows from the cached per-turn facts.
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "eval"):
        g_cache, g_name, g_gen, g_variant, g_rows, ks, ys, aux = [], [], [], [], [], [], [], []
        games = explored = value_only = 0
        decisions, outcome_targets = [0] * len(PHASES), [0] * len(PHASES)
        variant_games = [0] * (len(allow_match_settings or []) + 1)
        for p in chunks[split]:
            c, g = entry(p)
            if g["truncated"]:
                continue
            if split == "eval" and g["rows"] and c.cols[g["shard"]]["explore"][g["start"]:g["start"] + g["rows"]].any():
                raise SystemExit(f"{p} has exploratory picks; --eval-dir games must be made with exploration off")
            games += 1
            variant_games[variant_of.get(p, -1)] += 1
            if not g["rows"]:
                continue
            rows, counts = select_rows(_turns(c, g), split, explore_targets, value_weight, g["outcome"],
                                       **(quick_turns or {}), offset=g.get("start_k", 0))
            explored += counts["explored"]
            value_only += counts["value_only"]
            decisions = [x + y for x, y in zip(decisions, counts["decisions"])]
            outcome_targets = [x + y for x, y in zip(outcome_targets, counts["outcome_targets"])]
            if "aux" in counts:  # one small array per game: a list of row tuples would cost ~10x the memory
                aux.append(np.asarray(counts["aux"], np.float32).reshape(len(rows), len(quick_turns["aux_offsets"])))
            if not rows:
                continue
            g_cache.append(index[id(c)])
            g_name.append(Path(p).name)
            g_gen.append(gen_of[p])
            g_variant.append(variant_of.get(p, -1))
            g_rows.append(len(rows))
            ks.append(np.array([k for k, _, _ in rows], np.int32))
            ys.append(np.array([y for _, y, _ in rows], np.float32))
        s = {"game_cache": np.asarray(g_cache, np.int32), "game_name": g_name, "game_gen": np.asarray(g_gen, np.int32),
             "game_variant": np.asarray(g_variant, np.int32), "game_rows": np.asarray(g_rows, np.int64),
             "k": np.concatenate(ks) if ks else np.zeros(0, np.int32),
             "y": np.concatenate(ys) if ys else np.zeros(0, np.float32),
             "counts": {"games": games, "explored": explored, "value_only": value_only, "decisions": decisions,
                        "outcome_targets": outcome_targets, "variant_games": variant_games}}
        if (quick_turns or {}).get("aux_offsets"):
            s["aux"] = np.concatenate(aux) if aux else np.zeros((0, len(quick_turns["aux_offsets"])), np.float32)
        splits[split] = s
    timer.add("row_select", time.perf_counter() - t0)
    t0 = time.perf_counter()

    # The training vocab: names in order of first appearance over the training rows, as the
    # in-memory loader assigns them (per row: p1's view slots, then p2's); start values from the
    # training games' characters.
    for p in chunks["train"]:
        c, g = entry(p)
        if not g["truncated"] and g["rows"]:
            for char in g["chars"]:
                if char in c.manifest["raw_defaults"]:
                    vocab.raw_defaults.setdefault(char, dict(c.manifest["raw_defaults"][char]))
    names = _names_by_id(order)
    slots = {cat: [s for s, x in enumerate(CATEGORY_OF_SLOT) if x == cat] for cat in CATEGORIES}
    tr = splits["train"]
    starts = np.concatenate([[0], np.cumsum(tr["game_rows"])])
    entries = [order[ci].manifest["files"][name] for ci, name in zip(tr["game_cache"].tolist(), tr["game_name"])]
    i = 0
    while i < len(entries):  # consecutive games of one shard at a time
        ci, shard = int(tr["game_cache"][i]), entries[i]["shard"]
        j = i
        while j < len(entries) and int(tr["game_cache"][j]) == ci and entries[j]["shard"] == shard:
            j += 1
        rows = np.concatenate([entries[m]["start"] + tr["k"][starts[m]:starts[m + 1]].astype(np.int64)
                               for m in range(i, j)])
        a = order[ci].cols[shard]
        both = np.stack([a["cats0"][rows], a["cats1"][rows]], axis=1)
        for cat in CATEGORIES:
            seq = both[:, :, slots[cat]].reshape(-1)
            seq = seq[seq > 0]
            uniq, first = np.unique(seq, return_index=True)
            for cache_id in uniq[np.argsort(first)]:
                vocab.get(cat, names[ci, cat][cache_id])
        i = j
    timer.add("vocab", time.perf_counter() - t0)
    return {"splits": splits, "record_filter": filtered, "sample_window": windowed, "sampled_starts_excluded": no_sampled,
            "vocab": (vocab.ids, vocab.raw_defaults)}


def _names_by_id(order: list[Cache]) -> dict[tuple[int, str], list[str]]:
    """(cache index, category) -> list where [cache id] = name."""
    from yomibench.features import CATEGORIES

    out = {}
    for ci, c in enumerate(order):
        for cat in CATEGORIES:
            table = c.manifest["vocab"].get(cat, {})
            inv = [""] * (len(table) + 1)
            for name, i in table.items():
                inv[i] = name
            out[ci, cat] = inv
    return out


def _materialize(win, order, cand_caches, cand_k, vocab, n_gens, weights, device, explore_targets, quick_turns,
                 allow_match_settings, timer) -> dict[str, Any]:
    """open_cached's splits from a window (_select_window's, or a shared one): input columns gathered
    lazily from the shards with category ids remapped to `vocab`, the per-row columns from the games'
    cache entries, and the auxiliary heads' columns (aux_search, hp_margin, cand_*) from the caches."""
    from yomibench.features import CATEGORY_OF_SLOT
    from yomibench.train import PHASES, aux_search_targets, matchup_code

    t0 = time.perf_counter()  # the category lookup tables (the vocab pass's last step)
    vocab.frozen = True
    names = _names_by_id(order)
    lut2d = {}
    for ci in range(len(order)):
        tables = [np.array([0] + [vocab.get(cat, n) for n in names[ci, cat][1:]], np.int64) for cat in CATEGORY_OF_SLOT]
        width = max(len(t) for t in tables)
        lut2d[ci] = np.stack([np.pad(t, (0, width - len(t))) for t in tables])
    vocab.frozen = False
    timer.add("vocab", time.perf_counter() - t0)
    t0 = time.perf_counter()  # each split's columns: the row gathers read the shards' pages
    offsets = (quick_turns or {}).get("aux_offsets")
    out: dict[str, Any] = {}
    for split, s in win["splits"].items():
        counts, per = s["counts"], s["game_rows"]
        g = [order[ci].manifest["files"][name] for ci, name in zip(s["game_cache"].tolist(), s["game_name"])]
        shards, shard_no, game_shard = [], {}, []
        for ci, e in zip(s["game_cache"].tolist(), g):
            key = (ci, e["shard"])
            if key not in shard_no:
                shard_no[key] = len(shards)
                shards.append((order[ci].cols[e["shard"]], lut2d[ci]))
            game_shard.append(shard_no[key])
        k = s["k"].astype(np.int64)
        rows = _Rows(shards, np.repeat(np.asarray(game_shard, np.int32), per),
                     np.repeat(np.asarray([e["start"] for e in g], np.int64), per) + k)
        n = len(k)
        gen_i = np.repeat(s["game_gen"].astype(np.int64), per)
        v3 = rows.gather(np.arange(n), "v3") if n else np.zeros(0)
        slot = np.where(gen_i < 0, n_gens, gen_i)  # evaluation games (-1) count in the last entry
        n_states = np.bincount(slot, minlength=n_gens + 1).tolist()
        v3_states = np.bincount(slot[v3 != 0], minlength=n_gens + 1).tolist()
        y = s["y"]
        w = np.repeat(np.asarray([float(weights[x]) if x >= 0 else 1.0 for x in s["game_gen"].tolist()], np.float64),
                      per).astype(np.float32)
        hp = np.repeat(np.asarray([np.nan if e.get("hp_margin") is None else e["hp_margin"] for e in g], np.float64),
                       per).astype(np.float32)
        on_device = device is not None and split == "train"
        if on_device:
            import torch

            y, w, hp = (torch.from_numpy(a).to(device) for a in (y, w, hp))
        d: dict[str, Any] = {col: LazyColumn(rows, col, n) for col in INPUTS}
        if offsets:  # auxiliary search targets (train.aux_search_targets)
            a_ = s.get("aux")
            if a_ is None:  # a shared window: from the cached per-turn facts at the window's rows
                at, parts = 0, []
                for ci, e, m in zip(s["game_cache"].tolist(), g, per.tolist()):
                    targets = aux_search_targets(_turns(order[ci], e), offsets, explore_targets)
                    parts.append(np.asarray([targets[x] for x in s["k"][at:at + m].tolist()], np.float32)
                                 .reshape(m, len(offsets)))
                    at += m
                a_ = np.concatenate(parts) if parts else np.zeros((0, len(offsets)), np.float32)
            if on_device:
                import torch

                a_ = torch.from_numpy(a_).to(device)
            d["aux_search"] = a_
        d.update(y=y, w=w, hp_margin=hp, gen=gen_i.astype(np.float64),
                 # a sampled start's rows begin at decision start_k: phase is the true index over the true length
                 phase=(k + np.repeat(np.asarray([e.get("start_k", 0) for e in g], np.int64), per)).astype(np.float64)
                 / np.repeat(np.asarray([max(1, e.get("start_k", 0) + e["rows"]) for e in g], np.float64), per),
                 matchup=np.repeat(np.asarray([matchup_code(e["chars"]) for e in g], np.float64), per),
                 outcome=np.repeat(np.asarray([e["outcome"] for e in g], np.float64), per).astype(np.float32),
                 baseline=rows.gather(np.arange(n), "baseline") if n else np.zeros(0),
                 games=counts["games"], explored=counts["explored"], value_only=counts["value_only"],
                 v3_fraction=sum(v3_states) / sum(n_states) if sum(n_states) else 0.0,
                 v3_fraction_by_gen=[v / m if m else None for v, m in zip(v3_states[:n_gens], n_states[:n_gens])])
        if allow_match_settings:
            d.update(variant=np.repeat(s["game_variant"].astype(np.float64), per), variant_games=counts["variant_games"])
        if cand_caches and split != "eval":  # each row's qualifying full turn in the side cache, if any
            cref, at = [], 0
            for ci, name, m in zip(s["game_cache"].tolist(), s["game_name"], per.tolist()):
                cc = cand_caches[order[ci].record_dir.resolve()]
                shard_, slot_of = cc.slots_of(name)
                cref += [(id(cc), shard_, slot_of.get(x, -1)) for x in s["k"][at:at + m].tolist()]
                at += m
            d.update(_cand_columns(cref, cand_caches, cand_k))
        if split == "train":
            d["outcome_coverage"] = {name: round(o / dd, 4) if dd else None
                                     for (_, _, name), o, dd in zip(PHASES, counts["outcome_targets"], counts["decisions"])}
            d["outcome_coverage"]["all"] = (round(sum(counts["outcome_targets"]) / sum(counts["decisions"]), 4)
                                            if sum(counts["decisions"]) else None)
        out[split] = d
    if cand_caches:
        from yomibench.search_record import finish_table

        # one table for the run: the folders' side-cache tables in turn (_cand_columns offsets the uids)
        ccs = list(cand_caches.values())
        out["cand_table"] = finish_table(np.concatenate([c.unums for c in ccs]),
                                         {cat: [n for c in ccs for n in c.names[cat]] for cat in ("action", "dkey")},
                                         out["train"]["cand_uid"])
    timer.add("assemble", time.perf_counter() - t0)
    return out


# Shared windows (train --share-window): see the module docstring.
SHARE_FORMAT = 1
_WINDOW_ARRAYS = ("game_cache", "game_gen", "game_variant", "game_rows", "k", "y")


def _entry_json(e: dict[str, Any] | None) -> str:
    """A cache entry as hashed for sharing: without `hp_margin`, which --aux-hp-weight builds fill in
    later and which never decides a row (_materialize reads it per run)."""
    return json.dumps({k: v for k, v in (e or {}).items() if k != "hp_margin"}, sort_keys=True)


def window_key(settings: dict[str, Any], vocab) -> str:
    """A shared window's name: a hash of what decides its rows besides the records (open_cached's data
    arguments, the encoder, and the run's starting vocabulary in its order)."""
    h = hashlib.sha256(json.dumps({"format": SHARE_FORMAT, "encoder": encoder_hash(), "settings": settings},
                                  sort_keys=True, default=str).encode())
    h.update(json.dumps([vocab.ids, vocab.raw_defaults], default=str).encode())  # insertion order matters
    return h.hexdigest()[:24]


def window_state(chunks: dict[str, list[Path]], caches: dict[Path, Cache]) -> str:
    """A hash of the records a window is selected from: each split's files (held-out manifests,
    --games-per-gen sample, evaluation folders), their cache entries, and each cache's vocab, start
    values and exploration modes. Equal states (and keys) select equal windows."""
    h, cache_of = hashlib.sha256(), _cache_of(caches)
    for c in caches.values():
        h.update(json.dumps([str(c.dir.resolve()), {k: c.manifest.get(k) for k in
                                                    ("format", "encoder", "vocab", "raw_defaults", "explore_modes")}],
                            sort_keys=True).encode())
    for split, files in chunks.items():
        for p in files:
            h.update(f"{split}\0{p}\0".encode())
            h.update(_entry_json(cache_of(p).manifest["files"].get(Path(p).name)).encode())
    return h.hexdigest()


def _games_digest(win: dict[str, Any], order: list[Cache]) -> str:
    """A hash of the cache entries of a window's games (the rows it reads) and the caches' encoders."""
    h = hashlib.sha256(json.dumps([c.manifest.get("encoder") for c in order]).encode())
    for split, s in win["splits"].items():
        for ci, name in zip(s["game_cache"].tolist(), s["game_name"]):
            h.update(f"{split}\0{ci}\0{name}\0".encode())
            h.update(_entry_json(order[ci].manifest["files"].get(name)).encode())
    return h.hexdigest()


def save_window(path: Path, win: dict[str, Any], state: str, manifests: dict[str, Any], order: list[Cache],
                log=print) -> None:
    """Write a window to `path` (window.json, window.npz), replacing an older one there."""
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.savez(tmp / "window.npz", **{f"{split}.{k}": s[k] for split, s in win["splits"].items() for k in _WINDOW_ARRAYS})
    meta = {"format": SHARE_FORMAT, "state": state, "games_digest": _games_digest(win, order),
            "caches": [str(c.dir.resolve()) for c in order], "vocab": list(win["vocab"]), "heldout": manifests,
            "record_filter": win["record_filter"], "sample_window": win["sample_window"],
            **({"sampled_starts_excluded": win["sampled_starts_excluded"]} if win.get("sampled_starts_excluded") else {}),
            "splits": {split: {"game_name": s["game_name"], "counts": s["counts"]} for split, s in win["splits"].items()},
            "created": time.time(), "pid": os.getpid()}
    (tmp / "window.json").write_text(json.dumps(meta), encoding="utf-8")
    old = path.parent / f".{path.name}.{os.getpid()}.old"
    if path.exists():
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)
    log(f"shared window written: {path} ({sum(len(s['k']) for s in win['splits'].values())} rows)")


def load_window(path: Path, state: str, order: list[Cache], frozen: bool = False, log=print):
    """(window, {"mode", "dir"}) from `path` when it holds a window of these records (`state`; mode
    "exact"), or with `frozen` one whose games' cache entries are unchanged though games arrived since
    (mode "frozen"); else (None, None)."""
    try:
        meta = json.loads((path / "window.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as e:
        log(f"shared window {path}: not read ({type(e).__name__}: {e}); selecting as usual")
        return None, None
    if meta.get("format") != SHARE_FORMAT or meta.get("caches") != [str(c.dir.resolve()) for c in order]:
        log(f"shared window {path}: another format or other caches; selecting as usual")
        return None, None
    exact = meta["state"] == state
    if not exact and not frozen:
        log(f"shared window {path}: the records changed since it was written; selecting as usual")
        return None, None
    with np.load(path / "window.npz", allow_pickle=False) as z:
        splits = {split: {**{k: z[f"{split}.{k}"] for k in _WINDOW_ARRAYS}, "game_name": m["game_name"],
                          "counts": m["counts"]} for split, m in meta["splits"].items()}
    win = {"splits": splits, "record_filter": meta["record_filter"], "sample_window": meta["sample_window"],
           "vocab": tuple(meta["vocab"]), "heldout": meta["heldout"],
           "sampled_starts_excluded": meta.get("sampled_starts_excluded")}
    if not exact and _games_digest(win, order) != meta["games_digest"]:
        log(f"shared window {path}: its games' cache entries changed; selecting as usual")
        return None, None
    log(f"shared window read: {path} ({sum(len(s['k']) for s in splits.values())} rows; "
        + ("the same records" if exact else "frozen: games that arrived since it was written are left out")
        + "); selection skipped")
    return win, {"mode": "exact" if exact else "frozen", "dir": str(path)}


def _cand_columns(cref: list[tuple[int, str | None, int]], cand_caches: dict[Path, CandCache], k: int) -> dict[str, Any]:
    """A split's `cand_*` columns (search_record.split_arrays) from its rows' side-cache slots
    ((cache id, shard, row in shard), row -1: none); uids index the run's table, the side caches'
    tables concatenated in `cand_caches` order."""
    from yomibench.search_record import split_arrays

    by_id, offset, at = {}, {}, 0
    for c in cand_caches.values():
        by_id[id(c)], offset[id(c)] = c, at
        at += len(c.unums)
    have = np.array([i for i, r in enumerate(cref) if r[2] >= 0], np.int64)
    slot = np.full(len(cref), -1, np.int64)
    slot[have] = np.arange(len(have))
    cols = {"flip": np.zeros((len(have), 2), np.float32), "uid": np.full((len(have), 2, k), -1, np.int32),
            "val": np.full((len(have), 2, k), np.nan, np.float32), "prob": np.zeros((len(have), 2, k), np.float32)}
    groups: dict[tuple[int, str], list[int]] = {}
    for j, i in enumerate(have):
        groups.setdefault(cref[i][:2], []).append(j)
    for (cid, shard), js in groups.items():
        js = np.asarray(js, np.int64)
        loc = np.asarray([cref[have[j]][2] for j in js], np.int64)
        a = by_id[cid].cols[shard]
        for c in cols:
            block = np.asarray(a[c][loc])
            if c == "uid":
                block = np.where(block >= 0, block + offset[cid], -1)
            cols[c][js] = block
    return split_arrays(slot, cols["flip"], cols["uid"], cols["val"], cols["prob"], k)


def chunks_of(split: dict[str, Any], size: int = 1 << 16):
    """A split's inputs in row chunks as numpy dicts: whole arrays for an in-memory split, gathered
    blocks for a cached one (bounded memory for validation and evaluation)."""
    n = len(split["outcome"])
    if not isinstance(split["nums0"], LazyColumn):
        if n:
            yield {k: split[k] for k in INPUTS}
        return
    for i in range(0, n, size):
        yield {k: split[k][i:min(i + size, n)] for k in INPUTS}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="yomibench.cache")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="encode a record folder's new games into its cache")
    b.add_argument("dirs", type=Path, nargs="+")
    b.add_argument("--cache", type=Path, help="cache root for read-only folders (default: <dir>/.cache)")
    b.add_argument("--workers", type=int, default=None, help="encoding processes (default: up to 8)")
    b.add_argument("--cands", type=int, action="append", default=[], metavar="K",
                   help="also bring each folder's candidate side cache cands-k<K> up to date (train --aux-cand-k K)")
    b.add_argument("--append-only", action="store_true",
                   help="only add games: where a rebuild is due, change nothing (the next fit rebuilds)")
    args = ap.parse_args(argv)
    for d in args.dirs:
        cdir = build(d, args.cache, args.workers, append_only=args.append_only)
        for k in args.cands:
            build_cands(d, cdir, k, args.workers, append_only=args.append_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

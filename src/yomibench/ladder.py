"""Model bundles and a persistent Elo ladder (docs/MATCHES.md).

A bundle is one entrant's frozen nets: a zip holding `NAME/bundle.json` and `value.npz` and/or
`policy.npz`. `bundle.json` names the entrant and owner and holds each file's sha256; the bundle's
digest is the sha256 of its name and file hashes, so a bundle can't change without becoming a
different entrant. The nets load with `allow_pickle=False`: a bundle carries arrays, never code.

    yomibench bundle pack --name my-bot-1 --owner your-github-name --value v.npz --policy p.npz [--out my-bot-1.zip]
    yomibench bundle check my-bot-1.zip

A ladder season is a folder: `season.json` pins the engine (its hello: build, candidate set and
rules, match settings), the characters, the seat-swapped pairs per matchup and the season seed.
Entrants live in `entrants/NAME/`; every game is a row in `results/MODE.jsonl`. Each mode is its
own division, every entrant plays every other one (and the anchors `random` and the hand-written
`matrix-do`) with the same bot settings:

    reflex  policy:POLICY                       (no search)
    light   matrix-do:VALUE, policy prunes both seats to the top `light_k` (no audit)
    full    matrix-do:VALUE

A matchup's seeds come from the season seed and the two entrants' digests only, so adding an
entrant never changes another matchup's games, a rerun replays the same games, and two copies of a
season can split the work and `merge` their results. `play` plays only missing games; `rate` fits
Bradley-Terry (Elo scale, `random` = 1000, 95% paired-bootstrap intervals; yomibench.arena) and
writes LEADERBOARD.md and leaderboard.json.

    yomibench ladder seed --commit SECRET          # share this hash first
    yomibench ladder seed SECRET_A SECRET_B        # after both reveal: the season seed
    yomibench ladder init SEASON --engine game-pool:auto --characters Ninja,Cowboy --pairs 25 --seed N
    yomibench ladder add SEASON my-bot-1.zip
    yomibench ladder play SEASON --mode light [--engine game-pool:auto]
    yomibench ladder rate SEASON
    yomibench ladder merge SEASON other/results/light.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

FORMAT = 1
FILES = ("value.npz", "policy.npz")
MODES = ("reflex", "light", "full")
NEEDS = {"reflex": ("policy.npz",), "light": ("value.npz", "policy.npz"), "full": ("value.npz",)}
ANCHORS = ("random", "matrix-do")  # builtin entrants in every mode; `random` is the rating anchor (1000)
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,39}$")
MAX_FILE = 512 * 1024 * 1024
# hello fields a season pins: games on any other build, rules or settings are refused
PINNED = ("engine", "version", "game_version", "candidate_set", "candidate_rules", "match_settings")
PINNED_BUILD = ("yomisim_sha", "yomisim_dirty", "scene_sha256")


class LadderError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- bundles


def digest(manifest: dict[str, Any]) -> str:
    """The entrant's identity: its name and its files' hashes."""
    files = {k: v["sha256"] for k, v in sorted(manifest["files"].items())}
    return hashlib.sha256(json.dumps({"name": manifest["name"], "files": files}, sort_keys=True).encode()).hexdigest()


def _net_info(path: Path, kind: str) -> dict[str, Any]:
    """Loads the net (refusing other feature versions) and returns what the manifest records."""
    try:
        if kind == "value.npz":
            from yomibench.valuenet import ValueNet

            meta = ValueNet.load(path).meta
            return {"net": meta.get("name"), "features": meta.get("features")}
        from yomibench.policy import PolicyNet

        meta = PolicyNet.load(path).meta
        return {"net": meta.get("name"), "features": meta.get("features"), "policy_features": meta.get("policy_features")}
    except LadderError:
        raise
    except Exception as e:
        hint = ""
        if "feature version" in str(e):  # a net from another trainer, or an old one: point at the contract
            hint = ("; nets must be trained on yomibench's own features, see 'What a compatible net is' in "
                    "docs/SUBMIT.md")
        raise LadderError(f"{kind} doesn't load: {e}{hint}") from e


def pack(name: str, owner: str, value: Path | None, policy: Path | None, out: Path, notes: str = "") -> dict[str, Any]:
    if not NAME.match(name):
        raise LadderError(f"bundle names are 1-40 of a-z 0-9 . _ - (starting with a letter or digit): {name!r}")
    sources = {k: p for k, p in (("value.npz", value), ("policy.npz", policy)) if p is not None}
    if not sources:
        raise LadderError("a bundle needs --value, --policy or both")
    files = {k: {"sha256": _sha256(p), "bytes": p.stat().st_size, **_net_info(p, k)} for k, p in sources.items()}
    manifest = {"format": FORMAT, "name": name, "owner": owner, "created": _now(), "files": files,
                "modes": [m for m in MODES if all(f in files for f in NEEDS[m])], "notes": notes}
    manifest["digest"] = digest(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{name}/bundle.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for k, p in sources.items():
            z.write(p, f"{name}/{k}")
    return manifest


def _verify_dir(d: Path) -> dict[str, Any]:
    """A bundle folder's manifest, after checking every file against it and loading each net."""
    try:
        manifest = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise LadderError(f"{d}: no readable bundle.json ({e})") from e
    if manifest.get("format") != FORMAT:
        raise LadderError(f"{d}: bundle format {manifest.get('format')!r}; this code reads {FORMAT}")
    name, files = manifest.get("name", ""), manifest.get("files") or {}
    if not NAME.match(str(name)) or not files or set(files) - set(FILES):
        raise LadderError(f"{d}: bad name or files in bundle.json")
    for k, info in files.items():
        if _sha256(d / k) != info.get("sha256"):
            raise LadderError(f"{d}: {k} doesn't match its sha256 in bundle.json")
        _net_info(d / k, k)
    if manifest.get("digest") != digest(manifest):
        raise LadderError(f"{d}: digest doesn't match the name and file hashes")
    manifest["modes"] = [m for m in MODES if all(f in files for f in NEEDS[m])]
    return manifest


def _extract(bundle: Path, into: Path) -> Path:
    """Extracts a bundle zip into `into` (only NAME/bundle.json and NAME/{value,policy}.npz, no
    other paths) and returns the bundle folder."""
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1:
            raise LadderError(f"{bundle}: a bundle zip holds one NAME/ folder")
        (top,) = tops
        allowed = {f"{top}/{f}" for f in ("bundle.json", *FILES)} | {f"{top}/"}
        if not NAME.match(top) or set(names) - allowed or f"{top}/bundle.json" not in names:
            raise LadderError(f"{bundle}: unexpected entries {sorted(set(names) - allowed)}")
        for info in z.infolist():
            if info.file_size > MAX_FILE:
                raise LadderError(f"{bundle}: {info.filename} is larger than {MAX_FILE >> 20} MB")
        d = into / top
        d.mkdir(parents=True)
        for n in names:
            if not n.endswith("/"):
                (d / n.split("/", 1)[1]).write_bytes(z.read(n))
    return d


def check(bundle: Path) -> dict[str, Any]:
    """Verifies a bundle zip or folder and returns its manifest."""
    if bundle.is_dir():
        return _verify_dir(bundle)
    with tempfile.TemporaryDirectory() as tmp:
        return _verify_dir(_extract(bundle, Path(tmp)))


# ---------------------------------------------------------------- seasons


def season_seed(a: str, b: str) -> int:
    """The season seed from both sides' revealed secrets (order doesn't matter)."""
    x, y = sorted((a, b))
    return int(hashlib.sha256(f"{x}:{y}".encode()).hexdigest()[:8], 16)


def commitment(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def engine_pin(hello: dict[str, Any]) -> dict[str, Any]:
    pin = {k: hello.get(k) for k in PINNED if k in hello}
    build = hello.get("build") or {}
    if build:
        pin["build"] = {k: build.get(k) for k in PINNED_BUILD if k in build}
    return pin


class Season:
    def __init__(self, root: Path):
        self.root = root
        try:
            self.config = json.loads((root / "season.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise LadderError(f"{root}: not a ladder season (no readable season.json: {e})") from e
        self.lock = threading.Lock()

    # entrants -------------------------------------------------------
    def entrants(self) -> dict[str, dict[str, Any]]:
        """name -> manifest (with `dir`), anchors included."""
        out: dict[str, dict[str, Any]] = {a: {"name": a, "owner": "builtin", "digest": "builtin", "modes": list(MODES),
                                              "files": {}} for a in ANCHORS}
        edir = self.root / "entrants"
        for d in sorted(edir.iterdir()) if edir.is_dir() else []:
            if (d / "bundle.json").is_file():
                m = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
                m["modes"] = [x for x in MODES if all(f in m["files"] for f in NEEDS[x])]
                out[m["name"]] = {**m, "dir": d}
        return out

    def add(self, bundle: Path) -> dict[str, Any]:
        edir = self.root / "entrants"
        edir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            src = bundle if bundle.is_dir() else _extract(bundle, Path(tmp))
            manifest = _verify_dir(src)
            name = manifest["name"]
            if name in ANCHORS:
                raise LadderError(f"{name!r} is a builtin anchor's name")
            have = self.entrants().get(name)
            if have:
                if have["digest"] == manifest["digest"]:
                    return have
                raise LadderError(f"an entrant named {name!r} with other files is already in this season; "
                                  f"a new version needs a new name (e.g. {name}-2)")
            dest = edir / name
            stage = Path(tmp) / "stage"
            shutil.copytree(src, stage)
            os.replace(stage, dest)
        return self.entrants()[name]

    def bot(self, name: str, mode: str):
        """The bot entrant `name` plays in `mode`, named after the entrant."""
        from yomibench.bots import BUILTINS, MatrixBot, PolicyBot, PolicyPrune

        e = self.entrants()[name]
        if name in ANCHORS:
            bot = BUILTINS[name]()
        else:
            d: Path = e["dir"]
            value = policy = None
            if "value.npz" in NEEDS[mode]:
                from yomibench.valuenet import ValueNet

                value = ValueNet.load(d / "value.npz")
            if "policy.npz" in NEEDS[mode]:
                from yomibench.policy import PolicyNet

                policy = PolicyNet.load(d / "policy.npz")
            if mode == "reflex":
                bot = PolicyBot(policy)
            elif mode == "light":
                prune = PolicyPrune(policy, k=self.config["light_k"], random_k=0, audit=0.0, scope="both")
                bot = MatrixBot(value=value, search="oracle", policy=prune)
            else:
                bot = MatrixBot(value=value, search="oracle")
        bot.name = name
        return bot

    # results ---------------------------------------------------------
    def results_path(self, mode: str) -> Path:
        return self.root / "results" / f"{mode}.jsonl"

    def rows(self, mode: str) -> list[dict[str, Any]]:
        """This mode's games, one per key (a rated game wins over a halted or cut-off one), from
        entrants still in the season with the same files."""
        path = self.results_path(mode)
        if not path.exists():
            return []
        current = {n: e["digest"] for n, e in self.entrants().items()}
        best: dict[tuple, dict[str, Any]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if any(current.get(n) != r["digests"].get(n) for n in r["bots"]):
                    continue
                k = game_key(r)
                if k not in best or (best[k]["outcome"] in ("halt", "truncated") and r["outcome"] not in ("halt", "truncated")):
                    best[k] = r
        return list(best.values())

    def append(self, mode: str, rows: list[dict[str, Any]]) -> None:
        path = self.results_path(mode)
        path.parent.mkdir(exist_ok=True)
        with self.lock, open(path, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)

    # schedule --------------------------------------------------------
    def matchup_seed(self, a: dict[str, Any], b: dict[str, Any]) -> int:
        x, y = sorted((f"{a['name']}@{a['digest']}", f"{b['name']}@{b['digest']}"))
        h = hashlib.sha256(f"{self.config['seed']}:{x}:{y}".encode()).hexdigest()
        return int(h[:7], 16)  # game seeds are seed * 1_000_003 + game number (arena.run_round_robin)

    def missing(self, mode: str, only: str | None = None) -> list[tuple[str, str, list[tuple[tuple[str, str], int, bool]]]]:
        """(a, b, schedule of missing seat-swapped pairs) per matchup of `mode` with games to play."""
        from yomibench.arena import pair_schedule

        ents = {n: e for n, e in self.entrants().items() if mode in e["modes"]}
        names = sorted(ents)
        done: set[tuple] = set()
        for r in self.rows(mode):
            if r["outcome"] != "halt":
                done.add(game_key(r))
        out = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if only and only not in (a, b):
                    continue
                seed = self.matchup_seed(ents[a], ents[b])
                todo = []
                for chars, game_no in pair_schedule(self.config["characters"], self.config["pairs"]):
                    gs = seed * 1_000_003 + game_no
                    keys = [_key(mode, (a, b), chars, gs), _key(mode, (b, a), chars, gs)]
                    if not all(k in done for k in keys):
                        todo.append((chars, game_no, False))
                if todo:
                    out.append((a, b, todo))
        return out


def _key(mode: str, bots: tuple[str, str], chars: tuple[str, str], seed: int) -> tuple:
    return (mode, tuple(bots), tuple(chars), int(seed))


def game_key(r: dict[str, Any]) -> tuple:
    return _key(r["mode"], tuple(r["bots"]), tuple(r["characters"]), r["seed"])


def init(root: Path, engine_hello: dict[str, Any], characters: list[str], pairs: int, seed: int,
         candidate_set: int | None, match_settings: dict[str, Any] | None, set5: str | None,
         light_k: int = 6, max_turns: int = 1000, name: str | None = None) -> dict[str, Any]:
    if (root / "season.json").exists():
        raise LadderError(f"{root} already holds a season")
    if pairs < 1 or light_k < 1:
        raise LadderError("--pairs and --light-k must be positive")
    root.mkdir(parents=True, exist_ok=True)
    config = {"format": FORMAT, "name": name or root.name, "created": _now(), "engine": engine_pin(engine_hello),
              "candidate_set": candidate_set, "match_settings": match_settings, "set5": set5,
              "characters": characters, "pairs": pairs, "seed": seed, "light_k": light_k, "max_turns": max_turns,
              "modes": list(MODES), "anchors": list(ANCHORS), "rating_anchor": "random"}
    _write_json(root / "season.json", config)
    return config


def _connect_checked(season: Season, spec: str):
    from yomibench.engine import connect

    c = season.config
    engine = connect(spec, c["candidate_set"], match_settings=c["match_settings"])
    got = engine_pin(engine.hello())
    if got != c["engine"]:
        diff = sorted(k for k in set(got) | set(c["engine"]) if got.get(k) != c["engine"].get(k))
        engine.close()
        raise LadderError(f"engine {spec} isn't this season's: {', '.join(diff)} differ "
                          f"(season {[c['engine'].get(k) for k in diff]}, engine {[got.get(k) for k in diff]})")
    return engine


def play(season: Season, mode: str, specs: list[str], only: str | None = None, record_dir: Path | None = None,
         progress=print) -> int:
    """Plays every missing game of `mode` and returns how many it played."""
    from yomibench.arena import run_round_robin

    if mode not in MODES:
        raise LadderError(f"modes are {MODES}")
    if season.config.get("set5"):
        os.environ["YOMISIM_SET5"] = season.config["set5"]
    _connect_checked(season, specs[0]).close()  # refuse a wrong engine before any game
    ents = season.entrants()
    played = 0
    for a, b, todo in season.missing(mode, only):
        progress(f"{mode}: {a} v {b}: {len(todo)} pairs to play")
        seed = season.matchup_seed(ents[a], ents[b])
        digests = {a: ents[a]["digest"], b: ents[b]["digest"]}

        def on_result(r, digests=digests):
            row = {**asdict(r), "mode": mode, "digests": digests, "season": season.config["name"], "played": _now()}
            season.append(mode, [row])
            progress(f"  {r.bots[0]} v {r.bots[1]} {r.characters[0]}/{r.characters[1]} seed {r.seed}: "
                     f"{r.outcome} in {r.turns} turns, {r.seconds:.0f}s")

        results = run_round_robin([lambda s=s: _connect_checked(season, s) for s in specs],
                                  [season.bot(a, mode), season.bot(b, mode)], season.config["characters"],
                                  season.config["pairs"], seed, on_result=on_result,
                                  max_turns=season.config["max_turns"], schedule=todo, record_dir=record_dir)
        played += len(results)
    return played


def merge(season: Season, mode: str, path: Path) -> int:
    """Appends another copy's results of `mode` that this season doesn't have (same season seed and
    entrants: anything else would be different games). Returns how many were new."""
    have = {game_key(r) for r in season.rows(mode)}
    ents = season.entrants()
    game_nos = {g for _, g in _schedule(season)}
    new = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode") != mode or r.get("season") != season.config["name"]:
                raise LadderError(f"{path}: a row from season {r.get('season')!r}, mode {r.get('mode')!r}")
            for n in r["bots"]:
                if n not in ents or ents[n]["digest"] != r["digests"].get(n):
                    raise LadderError(f"{path}: entrant {n!r} isn't in this season with the same files; add its bundle first")
            a, b = sorted(r["bots"])
            seed = season.matchup_seed(ents[a], ents[b])
            if (r["seed"] - seed * 1_000_003) not in game_nos:
                raise LadderError(f"{path}: game seed {r['seed']} isn't on this season's schedule for {a} v {b}")
            k = game_key(r)
            if k not in have:
                have.add(k)
                new.append(r)
    season.append(mode, new)
    return len(new)


def _schedule(season: Season) -> list[tuple[tuple[str, str], int]]:
    from yomibench.arena import pair_schedule

    return pair_schedule(season.config["characters"], season.config["pairs"])


def rate(season: Season) -> dict[str, Any]:
    """Every mode's ratings; writes leaderboard.json and LEADERBOARD.md."""
    from yomibench.arena import (
        GameResult,
        bradley_terry,
        bradley_terry_ci,
        rated_pairs,
        win_matrix,
    )

    ents = season.entrants()
    out: dict[str, Any] = {"season": season.config["name"], "rated": _now(), "anchor": "random (1000)", "modes": {}}
    for mode in MODES:
        rows = season.rows(mode)
        results = [GameResult(tuple(r["bots"]), tuple(r["characters"]), r["seed"], r["outcome"], r["turns"],
                              tuple(r["hp"]) if r.get("hp") else None, r["seconds"], r.get("detail", ""))
                   for r in rows]
        pairs = rated_pairs(results)
        if not pairs:
            continue
        rated = [g for p in pairs for g in p]
        ratings = bradley_terry(results, anchor="random")
        ci = bradley_terry_ci(results, anchor="random", samples=500)
        wm = win_matrix(results)
        table = []
        for n, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
            mine = [g for g in rated if n in g.bots]
            score = sum(g.score(n) or 0 for g in mine)
            table.append({"entrant": n, "owner": ents.get(n, {}).get("owner", "?"), "elo": elo,
                          "ci95": list(ci.get(n, (None, None))), "games": len(mine),
                          "score": round(score / len(mine), 3) if mine else None,
                          "digest": ents.get(n, {}).get("digest", "?")[:12]})
        out["modes"][mode] = {"ratings": table, "games": len(rated),
                              "head_to_head": {f"{a} v {b}": {"score": round(s, 3), "games": g} for (a, b), (s, g) in sorted(wm.items())}}
    _write_json(season.root / "leaderboard.json", out)
    (season.root / "LEADERBOARD.md").write_text(_markdown(season, out), encoding="utf-8")
    return out


def _markdown(season: Season, board: dict[str, Any]) -> str:
    c = season.config
    eng = c["engine"]
    build = (eng.get("build") or {}).get("yomisim_sha", "")
    lines = [f"# Ladder: {c['name']}", "",
             (f"Engine {' '.join(str(x) for x in (eng.get('engine'), eng.get('game_version'), build[:12]) if x)}, candidate set "
             f"{eng.get('candidate_set')}, characters {', '.join(c['characters'])}, {c['pairs']} seat-swapped pairs "
             f"per character pairing per matchup. Elo scale, `random` = 1000, 95% paired-bootstrap intervals."), ""]
    for mode, m in board["modes"].items():
        lines += [f"## {mode} ({m['games']} games)", "", "| # | Entrant | Owner | Elo | 95% | Games | Score |",
                  "| ---: | --- | --- | ---: | --- | ---: | ---: |"]
        for i, r in enumerate(m["ratings"], 1):
            lo, hi = r["ci95"]
            lines.append(f"| {i} | {r['entrant']} | {r['owner']} | {r['elo']:.0f} | {lo:.0f} to {hi:.0f} | "
                         f"{r['games']} | {r['score']:.0%} |")
        names = [r["entrant"] for r in m["ratings"]]
        lines += ["", "Head to head (row's score against column):", "",
                  "| | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
        for a in names:
            cells = []
            for b in names:
                h = m["head_to_head"].get(f"{a} v {b}")
                cells.append("" if a == b or not h else f"{h['score']:.0%} ({h['games']})")
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def bundle_cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="yomibench bundle", description="pack or check a model bundle (docs/MATCHES.md)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack", help="zip a value and/or policy net with a manifest")
    pk.add_argument("--name", required=True, help="entrant name: a-z 0-9 . _ -, at most 40")
    pk.add_argument("--owner", required=True)
    pk.add_argument("--value", type=Path, help="value net (.npz): Light and Full modes")
    pk.add_argument("--policy", type=Path, help="policy net (.npz): Reflex and Light modes")
    pk.add_argument("--notes", default="")
    pk.add_argument("--out", type=Path, help="default: NAME.zip")
    ck = sub.add_parser("check", help="verify a bundle's hashes and load its nets")
    ck.add_argument("bundle", type=Path)
    args = p.parse_args(argv)
    try:
        if args.cmd == "pack":
            m = pack(args.name, args.owner, args.value, args.policy, args.out or Path(f"{args.name}.zip"), args.notes)
            print(f"{args.out or args.name + '.zip'}: {m['name']} ({m['owner']}), digest {m['digest'][:12]}, "
                  f"modes {', '.join(m['modes'])}")
        else:
            m = check(args.bundle)
            files = ", ".join(f"{k} ({v.get('net')}, {v['sha256'][:12]})" for k, v in sorted(m["files"].items()))
            print(f"ok: {m['name']} ({m['owner']}), digest {m['digest'][:12]}, modes {', '.join(m['modes'])}; {files}")
    except LadderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def ladder_cli(argv: list[str]) -> int:
    from yomibench.cli import _match_settings  # one parser for --match-settings

    p = argparse.ArgumentParser(prog="yomibench ladder", description="a persistent Elo ladder (docs/MATCHES.md)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sd = sub.add_parser("seed", help="season seed from both secrets, or --commit SECRET for the hash to share first")
    sd.add_argument("secrets", nargs="*")
    sd.add_argument("--commit", metavar="SECRET")
    it = sub.add_parser("init", help="start a season pinned to an engine")
    it.add_argument("season", type=Path)
    it.add_argument("--engine", default="sim", help="engine to pin (its hello: build, rules, settings)")
    it.add_argument("--characters", required=True, help="comma-separated; every pairing is played")
    it.add_argument("--pairs", type=int, required=True, help="seat-swapped pairs per character pairing per matchup")
    it.add_argument("--seed", type=int, required=True, help="season seed (ladder seed A B)")
    it.add_argument("--candidate-set", type=int)
    it.add_argument("--match-settings", type=_match_settings)
    it.add_argument("--set5", help="candidate set 5 switches, as arena --set5")
    it.add_argument("--light-k", type=int, default=6, help="Light mode's policy shortlist per seat (plus Continue)")
    it.add_argument("--max-turns", type=int, default=1000)
    ad = sub.add_parser("add", help="add a bundle (zip or folder) as an entrant")
    ad.add_argument("season", type=Path)
    ad.add_argument("bundles", type=Path, nargs="+")
    pl = sub.add_parser("play", help="play every missing game of a mode")
    pl.add_argument("season", type=Path)
    pl.add_argument("--mode", choices=MODES, required=True)
    pl.add_argument("--engine", default="sim", help="engine specs, one game at a time per spec (e.g. sim,sim,sim)")
    pl.add_argument("--only", help="only this entrant's matchups")
    pl.add_argument("--record-dir", type=Path, help="also write every game as a record (for showcases)")
    pl.add_argument("--dry-run", action="store_true", help="list the missing games, play nothing")
    rt = sub.add_parser("rate", help="fit ratings; write LEADERBOARD.md and leaderboard.json")
    rt.add_argument("season", type=Path)
    mg = sub.add_parser("merge", help="add another copy's results of a mode")
    mg.add_argument("season", type=Path)
    mg.add_argument("--mode", choices=MODES, required=True)
    mg.add_argument("results", type=Path, nargs="+")
    args = p.parse_args(argv)
    try:
        if args.cmd == "seed":
            if args.commit is not None:
                print(commitment(args.commit))
            elif len(args.secrets) == 2:
                print(season_seed(*args.secrets))
            else:
                p.error("seed takes --commit SECRET or two secrets")
        elif args.cmd == "init":
            from yomibench.engine import connect, set5_spec

            set5 = set5_spec(args.set5) if args.set5 else None
            if set5:
                os.environ["YOMISIM_SET5"] = set5
            engine = connect(args.engine, args.candidate_set, match_settings=args.match_settings)
            hello = engine.hello()
            engine.close()
            c = init(args.season, hello, args.characters.split(","), args.pairs, args.seed, args.candidate_set,
                     args.match_settings, set5, args.light_k, args.max_turns)
            print(f"season {c['name']} at {args.season}: engine {json.dumps(c['engine'], sort_keys=True)}")
        elif args.cmd == "add":
            season = Season(args.season)
            for b in args.bundles:
                e = season.add(b)
                print(f"{e['name']} ({e['owner']}): digest {e['digest'][:12]}, modes {', '.join(e['modes'])}")
        elif args.cmd == "play":
            season = Season(args.season)
            if args.dry_run:
                for a, b, todo in season.missing(args.mode, args.only):
                    print(f"{args.mode}: {a} v {b}: {len(todo)} pairs ({2 * len(todo)} games)")
                return 0
            from yomibench.engine import expand_specs

            n = play(season, args.mode, expand_specs(args.engine), args.only, args.record_dir,
                     progress=lambda s: print(s, flush=True))
            print(f"played {n} games; `yomibench ladder rate {args.season}` for the ratings")
        elif args.cmd == "rate":
            season = Season(args.season)
            rate(season)
            print((season.root / "LEADERBOARD.md").read_text(encoding="utf-8"))
        elif args.cmd == "merge":
            season = Season(args.season)
            for r in args.results:
                print(f"{r}: {merge(season, args.mode, r)} new games")
    except LadderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0

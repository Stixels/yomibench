"""One generation's public release files from its Showcase card: gen-N.json plus its videos, for a GitHub Release.

    python tools/site_release.py CARD.json --out DIR [--videos-root DIR]

gen-N.json holds only what the site shows: the generation, when it was promoted, its settings era (settings preset,
candidate set), the model's sha256, the promotion test against the previous generation (pooled and per matchup,
with 95% intervals and game counts), any anchor results against older generations, and the featured and cast videos
(file, slot, matchup, result, turns, which side the new generation played). No paths, machine names or seeds. Videos are
copied as gen-N-<name>.mp4, each with a poster still (gen-N-<name>.jpg). The output is checked with tools/privacy_scan.py's rules before it's written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window when run from a GUI process on Windows


def gen_of(name: str | None) -> int | None:
    m = re.search(r"(?:generation|gen)\s*(\d+)", name or "")
    return int(m.group(1)) if m else None


def block(b: dict) -> dict:
    out = {"score": b.get("score"), "ci95": b.get("ci95"), "games": b.get("games")}
    if b.get("games") is None and b.get("pairs") is not None:
        out["games"] = 2 * int(b["pairs"])
    return out


def video(v: dict, n: int, prev: int | None, folder: str) -> dict:
    seat = v.get("candidate_seat")
    return {"file": f"gen-{n}-{Path(v['web']).name}", "source": v["web"], "slot": v.get("slot"),
            "why": v.get("why"), "matchup": (v.get("pairing") or "").replace("-", " vs "), "result": v.get("result"),
            "turns": v.get("turns"),
            "new_side": {"p1": "left, P1", "p2": "right, P2"}.get(seat), "opponent_generation": prev, "group": folder}


def release(card: dict) -> tuple[dict, list[tuple[str, str]]]:
    n = int(str(card["checkpoint"]).split("-g")[-1])
    prev = gen_of(card.get("opponent"))
    vo = card.get("vs_opponent") or {}
    anchors = {}
    for name, r in ((card.get("anchor_gate") or {}).get("results") or {}).items():
        g = gen_of(name)
        anchors[f"gen-{g}" if g is not None else name] = block(r)
    vids = [video(v, n, prev, "featured") for v in card.get("featured") or []]
    if (card.get("cowboy_cast_source") or {}).get("source") == "gate":  # older cards' casts were self-play
        vids += [video(v, n, prev, "cast") for v in card.get("cowboy_cast") or []]
    rel = {
        "generation": n, "promoted_utc": card.get("promoted"), "opponent_generation": prev,
        "era": {"settings": card.get("settings"), "candidate_set": card.get("candidate_set")},
        "model_sha256": Path(str(card.get("model", ""))).stem or None,
        "games_trained": (card.get("games_trained") or {}).get("total"),
        "gate": {"pooled": block(vo.get("pooled") or {}),
                 "matchups": {k.replace("-", " vs "): block(b) for k, b in (vo.get("pairings") or {}).items()}},
        "anchors": anchors,
        "videos": [{k: v for k, v in x.items() if k != "source"} for x in vids],
    }
    return rel, [(x["source"], x["file"]) for x in vids]


def poster(video: Path) -> Path | None:
    """A still from the middle of the game, beside the video as NAME.jpg (the site shows it before playing).
    Needs ffmpeg and ffprobe; without them the site shows the video's first frame instead."""
    out = video.with_suffix(".jpg")
    try:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(video)], capture_output=True, text=True, check=True,
                                   creationflags=NOWIN).stdout.strip() or 0)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{dur * 0.45:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=960:-2", "-q:v", "5", str(out)], check=True, creationflags=NOWIN)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return out


def scrub_check(text: str) -> list[str]:
    """Why `text` may not be published: tools/privacy_scan.py's rules (plus the local deny file, if configured)."""
    spec = importlib.util.spec_from_file_location("privacy_scan", ROOT / "tools" / "privacy_scan.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    return [why for rx, why in ps.rules(ps.default_deny_file()) if rx.search(text)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("card", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videos-root", type=Path, help="a folder searched by file name for videos the card names elsewhere")
    a = ap.parse_args(argv)
    card = json.loads(a.card.read_text(encoding="utf-8"))
    rel, copies = release(card)
    text = json.dumps(rel, indent=1)
    bad = scrub_check(text)
    if bad:
        raise SystemExit(f"gen-{rel['generation']}.json refused by the scrub: {sorted(set(bad))}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"gen-{rel['generation']}.json").write_text(text, encoding="utf-8")
    found = {}
    if a.videos_root:  # older cards name web copies on another machine: find them by file name
        found = {p.name: p for p in a.videos_root.rglob("*.mp4")}
    missing = []
    for src, dst in copies:
        p = Path(src) if Path(src).exists() else found.get(Path(src.replace("\\", "/")).name)
        if p is None:
            missing.append(Path(src.replace("\\", "/")).name)
            continue
        shutil.copyfile(p, a.out / dst)
        poster(a.out / dst)
    if missing:
        raise SystemExit(f"gen-{rel['generation']}: videos not found: {missing} (pass --videos-root)")
    print(f"gen-{rel['generation']}: {len(copies)} videos -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

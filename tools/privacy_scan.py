"""Privacy scan: refuse addresses, personal emails, local paths, oversized or data files and internal notes.

    python tools/privacy_scan.py [REPO] [--deny FILE] [--commits RANGE] [--report out.json]

Scans every tracked file of REPO (default: this checkout) with the generic rules below, plus one private regex per
line from a deny file (names, machine names) when one is given or the local config names it (`[privacy]
deny_file`, yomibench.localconfig). The deny file lives outside every repository: its patterns are private.
`--commits A..B` also checks those commits' authors, committers and messages. Exit 1 on any finding. CI runs it
without a deny file; site releases (tools/site_release.py) use the same rules.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GENERIC = [
    (r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b", "tailnet IP"),
    (r"\b192\.168\.\d{1,3}\.\d{1,3}\b", "LAN IP"),
    (r"[\w.+-]+@(?!users\.noreply\.github\.com\b|anthropic\.com\b|example\.(?:com|org)\b)[\w-]+\.(?:com|net|org|io|dev|me)\b",
     "email"),
    (r"(?i)\b[A-Z]:[\\/]+(?:Users|yomibench[\w-]*|yomisim[\w-]*|SteamLibrary|experiments)\b", "local path"),
    (r"(?<![\w.])/(?:Users|home)/[A-Za-z][\w.-]*", "home path"),
    (r"~/Code\b", "home path"),
    (r"(?i)\.ts\.net\b|\btailscale\b|\btailnet\b", "private network"),
    (r"(?i)human-replays|ranked replays", "human-replay corpus"),
    (r"(?i)yomisim", "private simulator"),
    (r"\bA\d{2,3}\b", "action number"),
    (r"(?i)coordination thread|\bthe coordinator\b|user (?:said|decided|asked)", "internal process"),
    (r"(?i)the user'?s (?:decision|own play|monitors)|\bagent (?:starts|launched)", "internal process"),
    (r"\(#\d{2,4}\)|\bPR ?#\d{2,4}\b|(?<![\w&])#1\d\d:", "private PR reference"),
    (r"claude:viewport|Staged Site", "preview tool"),
]
# Identifiers of the optional private engine plugin that data and interop carry (env vars, hello/record keys, the
# engine's name in records, import guards). Reviewed: they name it, they don't describe it; prose may not.
ALLOW_TOKENS = [
    r"YOMISIM_(?:SET5|SCENE)", r"yomisim_(?:sha|dirty|commit)", r"_have_yomisim", r'(?:find_spec|importorskip|version)\("yomisim"\)',
    r'"yomisim"', r"\byomisim=\{", r"\byomisim\.(?:Handler|Batch|CANDIDATE_SETS)\b", r"^\s*yomisim = ",
]
# Whole lines allowed by path: text whose exact bytes are hashed (the rules files' comment strings feed
# hello.candidate_rules), and the credits (the README's Acknowledgements, the site's credits block and footer).
ALLOW_LINES = [
    (r"mod/YomiBench/candidate_rules_v[56]\.json", r'^\s*"(?:comment|set6_spec)":'),
    (r"README\.md", r"^- \*\*TRUEcoiness, MoreBlasta, glopsy\*\*, YOMI Hustle players, for direction and documentation\.$"),
    (r"site/build\.py", r"^\s*'<p><b>TRUEcoiness, MoreBlasta, glopsy</b>, YOMI Hustle players, for direction and documentation\.</p></section>'\)$"),
    (r"site/build\.py", r"^FOOT_CREDITS = 'Thanks to Ivy Sly, developer of Your Only Move Is HUSTLE, and to TRUEcoiness, MoreBlasta and glopsy\.'$"),
    (r"tools/privacy_scan\.py", r"."),   # this file states the rules
    (r"tests/test_privacy_scan\.py", r"."),
]
BAD_EXT = {".pck", ".tscn", ".tres", ".import", ".ogg", ".wav", ".mp3", ".exe", ".dll", ".so", ".dylib",
           ".x86_64", ".npz", ".gz", ".jsonl", ".tar", ".zip", ".whl", ".bundle", ".pyc", ".replay"}
SKIP_TEXT = (".lock",)
MAX_BYTES = 5 * 1024 * 1024


def default_deny_file() -> Path | None:
    from yomibench import localconfig  # noqa: PLC0415  (tools put src on the path through the installed package)

    return localconfig.load().privacy_deny_file


def rules(deny: Path | None = None) -> list[tuple[re.Pattern, str]]:
    out = [(re.compile(p), why) for p, why in GENERIC]
    if deny is not None:
        for line in Path(deny).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                out.append((re.compile(line.strip()), "deny-list term"))
    return out


def scan_text(rs, path: str, text: str) -> list[dict]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        if any(re.fullmatch(pp, path) and re.search(lp, line) for pp, lp in ALLOW_LINES):
            continue
        checked = line
        for tok in ALLOW_TOKENS:
            checked = re.sub(tok, "", checked)
        for rx, why in rs:
            if rx.search(checked):
                hits.append({"path": path, "line": n, "why": why, "text": line.strip()[:160]})
    return hits


def scan_repo(repo: Path, rs) -> list[dict]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, check=True).stdout
    hits = []
    for rel in filter(None, out.decode().split("\0")):
        f = repo / rel
        if not f.is_file():
            continue
        if f.suffix.lower() in BAD_EXT:
            hits.append({"path": rel, "line": 0, "why": "asset, data or binary file", "text": ""})
        if f.stat().st_size > MAX_BYTES:
            hits.append({"path": rel, "line": 0, "why": "file over 5 MB", "text": ""})
        if rel.endswith(SKIP_TEXT):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += scan_text(rs, rel, text)
    return hits


def scan_commits(repo: Path, rs, rng: str) -> list[dict]:
    log = subprocess.run(["git", "-C", str(repo), "log", "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x1e", rng],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    hits = []
    for rec in filter(str.strip, log.split("\x1e")):
        sha, author, committer, msg = (rec.strip("\n").split("\0") + ["", "", ""])[:4]
        for h in scan_text(rs, "(commit)", f"{author}\n{committer}\n{msg}"):
            hits.append({**h, "path": f"commit {sha[:10]}"})
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", type=Path, nargs="?", default=ROOT)
    ap.add_argument("--deny", type=Path, help="private terms, one regex per line (default: the local config's)")
    ap.add_argument("--commits", help="also check these commits' people and messages (a git range, e.g. main..HEAD)")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    deny = a.deny or default_deny_file()
    rs = rules(deny)
    hits = scan_repo(a.repo, rs) + (scan_commits(a.repo, rs, a.commits) if a.commits else [])
    if a.report:
        a.report.write_text(json.dumps(hits, indent=1), encoding="utf-8")
    for h in hits[:60]:
        print(f"  {h['path']}:{h['line']}: {h['why']}: {h['text']}")
    print(f"{len(hits)} finding(s){' (with a deny file)' if deny else ' (generic rules only)'}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

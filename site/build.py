"""Build the static site: the Arena (home), the Lab, Get started and Submit, every page at its own URL.

    python site/build.py --releases DIR [--docs docs] [--arena arena] [--out _site] [--site-url URL]
    python site/build.py --releases DIR --spa FILE      # one-file preview with hash routes (#lab-showcase-g13)

DIR holds each release's files (gen-N.json, its videos and their poster stills), as the Pages workflow downloads
them; `arena` holds the Arena's data (settings.json, bots.json, results/*.jsonl; see arena/README.md).

Routes (pretty URLs: each is a folder with an index.html; links between pages are relative, so the site works under
any base path; --site-url adds canonical links and the 404 page's base):

    /                       the Arena: leaderboard and the Showcase-match hero (/arena/ redirects here)
    /bots/g13/              a bot's page (house generations are gN; Arena entries use their name)
    /lab/                   the Lab overview: what this is, current-best stats, the training story
    /lab/showcase/          the latest generation's Showcase, and every earlier one at /lab/showcase/gN/
    /lab/generations/       the rating history by generation
    /lab/how-it-works/  /lab/training/  /lab/mod/  /lab/writeups/NAME/
    /get-started/  /get-started/setup/  /submit/  /submit/matches/  /play/

The one-file preview (--spa) holds every route and shows the one named by the URL's hash, the path with slashes as
hyphens (/lab/showcase/g13/ is #lab-showcase-g13; / is #arena); the back and forward buttons work.

Ratings: Bradley-Terry over every head-to-head (the house generations' promotion tests and anchor results from the
releases, and the Arena's rated matches), Elo-like scale with the oldest generation at 0, 95% intervals from a
parametric bootstrap. The hero is the longest game between the two best bots (by turns when every candidate game
records them, else by video length). A change of settings preset or candidate set starts a new era.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import math
import random
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
esc = html.escape
# Public docs and where they live on the site.
DOC_ROUTES = {"SETUP.md": "/get-started/setup/", "TRAINING.md": "/lab/training/", "METHODOLOGY.md": "/lab/how-it-works/",
              "PLAYING.md": "/play/", "MOD.md": "/lab/mod/", "SUBMIT.md": "/submit/"}
REPO = "https://github.com/Stixels/yomibench"
FORM = f"{REPO}/issues/new?template=submit-bot.yml"
NAV = [("/", "Arena", "arena"), ("/lab/", "Lab", "lab"), ("/get-started/", "Get started|Start", "start"),
       ("/submit/", "Submit", "submit"), ("/play/", "Play", "play")]
LAB_NAV = [("/lab/", "Overview"), ("/lab/showcase/", "Showcase"), ("/lab/generations/", "Generations"),
           ("/lab/how-it-works/", "How it works"), ("/lab/training/", "Training"), ("/lab/mod/", "The game mod")]
FONTS = ("https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=Inter:wght@400;500;600;700"
         "&family=JetBrains+Mono:wght@500&display=swap")
ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' "
        "height='32' rx='7' fill='%23c2410c'/%3E%3Cpath d='M9 9l7 8 7-8M16 17v7' stroke='white' stroke-width='3.2' "
        "fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E")
# Credits, in the user's own words and names (README.public.md and README.md carry the same text).
CREDITS = ('<section class="credits"><h2>Credits</h2>'
           '<p><b>Ivy Sly</b>, developer of <i>Your Only Move Is HUSTLE</i>, for permission to build this project, including a Rust rewrite of the engine.</p>'
           '<p><b>jackmaiorino</b>, for <a href="https://github.com/jackmaiorino/spellbench">spellbench</a>, the Magic: The Gathering bot arena yomibench is a sibling of.</p>'
           '<p><b>TRUEcoiness, MoreBlasta, glopsy</b>, YOMI Hustle players, for direction and documentation.</p></section>')
FOOT_CREDITS = 'Thanks to Ivy Sly, developer of Your Only Move Is HUSTLE, and to TRUEcoiness, MoreBlasta and glopsy.'


# --- links: bodies carry route and media tokens, resolved per output mode ----------------------------------------
def L(path: str) -> str:
    return f"@@L:{path}@@"


def M(file: str) -> str:
    return f"@@M:{file}@@"


def hash_of(path: str) -> str:
    return path.strip("/").replace("/", "-") or "arena"


def rel_to(here: str, path: str) -> str:
    """The relative link from the page at route `here` to route `path` (or a root file)."""
    depth = len([p for p in here.strip("/").split("/") if p])
    target = path.lstrip("/")
    return ("../" * depth + target) or "./"


def resolve(text: str, here: str, spa: bool) -> str:
    def link(m):
        path, anchor = m.group(1).partition("#")[::2]
        if spa:
            return "#" + hash_of(path)
        return rel_to(here, path) + (f"#{anchor}" if anchor else "")
    text = re.sub(r"@@L:([^@]*)@@", link, text)
    return re.sub(r"@@M:([^@]*)@@", lambda m: ("media/" if spa else rel_to(here, "/media/")) + m.group(1), text)


# --- data ------------------------------------------------------------------------------------------------------
def house(n) -> str:
    return f"gen-{n}"


def is_house(bot: str) -> bool:
    return re.fullmatch(r"gen-\d+", bot) is not None


def label(bot: str) -> str:
    m = re.fullmatch(r"gen-(\d+)", bot)
    return f"HustleBot g{m.group(1)}" if m else bot


def short(bot: str) -> str:
    m = re.fullmatch(r"gen-(\d+)", bot)
    return f"g{m.group(1)}" if m else bot


def bot_route(bot: str) -> str:
    m = re.fullmatch(r"gen-(\d+)", bot)
    return f"/bots/g{m.group(1)}/" if m else f"/bots/{re.sub(r'[^a-z0-9._-]', '-', bot.lower())}/"


def spec_hash(spec: dict) -> str:
    """The Arena spec's identity (as tools/arena_run.py computes it): sha256 of its canonical JSON without `notes`."""
    body = {k: v for k, v in spec.items() if k != "notes"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def load_arena(arena: Path | None) -> tuple[list[dict], list[dict], dict | None]:
    """(entered bots, rated match rows of the current spec, the spec) from the Arena's data folder. Rows rated under
    an earlier spec belong to an earlier era and aren't rated."""
    if arena is None or not arena.is_dir():
        return [], [], None
    bots = json.loads((arena / "bots.json").read_text(encoding="utf-8")) if (arena / "bots.json").exists() else []
    spec = json.loads((arena / "spec.json").read_text(encoding="utf-8")) if (arena / "spec.json").exists() else None
    rows = []
    for f in sorted((arena / "results").glob("*.jsonl")) if (arena / "results").is_dir() else []:
        rows += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    if spec is not None:
        rows = [r for r in rows if r.get("spec", spec_hash(spec)) == spec_hash(spec)]
    return bots, rows, spec


def results(rels: list[dict], rows: list[dict] = ()) -> list[tuple[str, str, float, int]]:
    """(bot, other bot, the first's score, games) for every head-to-head: the releases' promotion tests and anchor
    results against older generations, and the Arena's rated matches."""
    out = []
    for r in rels:
        n, g = r["generation"], r["gate"]["pooled"]
        if r.get("opponent_generation") is not None and g.get("score") is not None and g.get("games"):
            out.append((house(n), house(r["opponent_generation"]), g["score"], int(g["games"])))
        for name, a in (r.get("anchors") or {}).items():
            m = re.fullmatch(r"gen-(\d+)", name)
            if m and a.get("score") is not None and a.get("games"):
                out.append((house(n), house(int(m.group(1))), a["score"], int(a["games"])))
    for row in rows:
        if row.get("score") is not None and row.get("games"):
            out.append((row["a"], row["b"], float(row["score"]), int(row["games"])))
    return out


# --- ratings ---------------------------------------------------------------------------------------------------
WEIGHTS_LAG = 2  # HustleBot's weights are published this many generations behind the current best
ARENA_ONLY_GAMES = 100  # per bot, in the current spec's era, before the Arena rates on its own games alone
PRIOR_GAMES = 2  # each player also drew this many games against a fixed reference: keeps a perfect score finite


def bradley_terry(res, players, base, iters=500):
    """Minorization-maximization (Hunter 2004); draws count half (the score). Scale: `base` at 0."""
    th = {p: 1.0 for p in players}
    wins = {p: PRIOR_GAMES / 2 for p in players}
    for i, j, s, n in res:
        wins[i] += s * n
        wins[j] += (1 - s) * n
    for _ in range(iters):
        new = {}
        for p in players:
            den = PRIOR_GAMES / (th[p] + 1.0) + sum(n / (th[i] + th[j]) for i, j, s, n in res if p in (i, j))
            new[p] = wins[p] / den
        th = new
    return {p: 400 * math.log10(th[p] / th[base]) for p in players}


def ratings_rows(rows, rels):
    """Ratings from the Arena's rows alone, the oldest generation at 0."""
    res = results([], rows)
    players = sorted({p for i, j, _, _ in res for p in (i, j)})
    base = house(min(r["generation"] for r in rels)) if rels and house(min(r["generation"] for r in rels)) in players         else players[0]
    point = bradley_terry(res, players, base)
    rng = random.Random(0)
    boots = {p: [] for p in players}
    for _ in range(200):
        sample = [(i, j, sum(rng.random() < s for _ in range(n)) / n, n) for i, j, s, n in res]
        b = bradley_terry(sample, players, base, iters=200)
        for p in players:
            boots[p].append(b[p])
    return point, {p: (sorted(v)[5], sorted(v)[194]) for p, v in boots.items()}


def ratings(rels, rows=()):
    res = results(rels, rows)
    players = sorted({p for i, j, _, _ in res for p in (i, j)})
    if not players:
        return {}, {}
    base = house(min(r["generation"] for r in rels)) if rels else players[0]  # the oldest generation is 0
    point = bradley_terry(res, players, base)
    rng = random.Random(0)
    boots = {p: [] for p in players}
    for _ in range(200):
        sample = [(i, j, sum(rng.random() < s for _ in range(n)) / n, n) for i, j, s, n in res]
        b = bradley_terry(sample, players, base, iters=200)
        for p in players:
            boots[p].append(b[p])
    ci = {p: (sorted(v)[5], sorted(v)[194]) for p, v in boots.items()}
    return point, ci


# --- videos ----------------------------------------------------------------------------------------------------
def mp4_seconds(path: Path) -> float | None:
    """A video's length from its mp4 header (the mvhd box), or None."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    i = data.find(b"mvhd")
    if i < 4 or len(data) < i + 36:
        return None
    if data[i + 4] == 1:
        scale, dur = int.from_bytes(data[i + 24:i + 28], "big"), int.from_bytes(data[i + 28:i + 36], "big")
    else:
        scale, dur = int.from_bytes(data[i + 16:i + 20], "big"), int.from_bytes(data[i + 20:i + 24], "big")
    return dur / scale if scale else None


def games(rels: list[dict], media: Path) -> list[dict]:
    """Every published game video, with its two bots, its length and its poster (when the release has one)."""
    out = []
    for r in rels:
        for v in r["videos"]:
            poster = Path(v["file"]).with_suffix(".jpg").name
            out.append({**v, "gen": r["generation"], "bots": (house(r["generation"]), house(v["opponent_generation"])),
                        "seconds": mp4_seconds(media / v["file"]),
                        "poster": poster if (media / poster).exists() else None})
    return out


def showcase_match(board: list[str], vids: list[dict]) -> tuple[dict | None, str]:
    """The hero game: the longest game between the two best bots (or, with no game between them, the best bot's
    longest), and a line saying why it was picked."""
    if not board or not vids:
        return None, ""
    top = set(board[:2])
    pool = [v for v in vids if set(v["bots"]) == top] if len(top) == 2 else []
    why = "The longest published replay between the two top-rated bots" if pool else ""
    if not pool:
        pool = [v for v in vids if board[0] in v["bots"]]
        why = "The longest published replay of the top-rated bot"
    if not pool:
        return None, ""
    if all(v.get("turns") for v in pool):
        return max(pool, key=lambda v: v["turns"]), why
    return max(pool, key=lambda v: v.get("seconds") or 0), why


def turns_of(v) -> int | None:
    if v.get("turns"):
        return int(v["turns"])
    m = re.search(r"(\d+) turns", v.get("why") or "")
    return int(m.group(1)) if m else None


def clock(s):
    return "" if not s else f"{int(s) // 60}:{int(s) % 60:02d}"


ICONS = {k: f'<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">{v}</svg>' for k, v in {
    "play": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9l5 3-5 3z" class="fill"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1'
            'M4.9 19.1L7 17M17 7l2.1-2.1"/>',
    "loop": '<path d="M4 12a8 8 0 0 1 14-5.3M20 12a8 8 0 0 1-14 5.3"/><path d="M18 3v4h-4M6 21v-4h4"/>',
    "chart": '<path d="M3 20h18"/><path d="M5 16l4-5 4 3 6-8"/>',
    "doc": '<path d="M6 3h9l4 4v14H6z"/><path d="M9 12h7M9 16h7"/>',
}.items()}


# --- small pieces ----------------------------------------------------------------------------------------------
def pct(x):
    return "—" if x is None else f"{round(100 * x)}%"


def day(utc) -> str:
    try:
        d = datetime.datetime.fromisoformat(str(utc).replace("Z", "+00:00"))
    except ValueError:
        return ""
    return f"{d:%b} {d.day}, {d.year}"


def era_label(e):
    return f"{e.get('settings', '?')} settings · set\u00a0{e.get('candidate_set', '?')}"


def kcount(n) -> str:
    return "—" if not n else (f"{n / 1000:.0f}k" if n >= 10000 else f"{n:,}")


def video_card(v) -> str:
    res = (v.get("result") or "").lower()
    a, b = v["bots"]
    seat = f'{short(a)} <span class="dim">({esc(v["new_side"])})</span> vs {short(b)}' if v.get("new_side") else \
        f"{short(a)} vs {short(b)}"
    why = re.sub(r"^Cowboy (mirror|vs \w+),\s*", "", v.get("why") or "")
    poster = f' poster="{M(v["poster"])}"' if v.get("poster") else ""
    length = " · ".join(x for x in (f"{turns_of(v)} turns" if turns_of(v) else "", clock(v.get("seconds"))) if x)
    return (f'<figure class="game"><video controls preload="none" playsinline{poster} src="{M(v["file"])}"></video>'
            f'<figcaption><div class="row"><span class="why">{esc(why[:1].upper() + why[1:])}</span>'
            f'<span class="res {esc(res)}">{esc(res.title())}</span></div><div class="seats">{seat}</div>'
            f'<div class="meta">{esc(v.get("matchup") or "")}{" · " + esc(length) if length else ""}</div>'
            f'</figcaption></figure>')


def tile(value, name, sub="", big=False) -> str:
    return (f'<div class="tile{" big" if big else ""}"><span class="tname">{esc(name)}</span>'
            f'<span class="tval">{value}</span>{f"<span class=tsub>{sub}</span>" if sub else ""}</div>')


COPY_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><g class="cp"><rect x="8" y="8" width="14" height="14" rx="2"/>'
             '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></g><path class="ok" d="M5 12l5 5L20 7"/></svg>')


def copy_button(target: str, label: str, text: str | None = None, small: bool = False,
                native_title: bool | None = None) -> str:
    """An icon copy button (COPY wires it). It copies `text` when given, else the text of the element `target`.
    A button inside a box that clips overflow (the Get started command boxes) can't show the CSS label, so it gets a
    native title instead; that's the default for small buttons."""
    extra = f' data-copy-text="{esc(text)}"' if text is not None else ""
    if small if native_title is None else native_title:
        extra += f' title="{esc(label)}"'
    return (f'<button type="button" class="iconbtn{" sm" if small else ""}" data-copy="{target}"{extra} '
            f'aria-label="{esc(label)}" data-tip="{esc(label)}">{COPY_ICON}</button>')


def one_line(cmd: str) -> str:
    """Commands with their `\\` line continuations joined, so they paste into PowerShell as well as a POSIX shell."""
    return re.sub(r"[ \t]*\\\n[ \t]*", " ", cmd)


def meter(x) -> str:
    w = 0 if x is None else max(0, min(100, round(100 * x)))
    return f'<span class="meter" aria-hidden="true"><i style="width:{w}%"></i></span>'


def cibar(lo, hi, p, scale) -> str:
    a, b = scale
    f = lambda v: 100 * (v - a) / (b - a) if b > a else 50  # noqa: E731
    return (f'<span class="cibar" aria-hidden="true"><i style="left:{f(lo):.1f}%;width:{max(1.5, f(hi) - f(lo)):.1f}%">'
            f'</i><b style="left:{f(p):.1f}%"></b></span>')


def ci_text(bot, ci, base_bot) -> str:
    if bot == base_bot:
        return "rating anchor"
    lo, hi = ci[bot]
    return f"{lo:.0f} to {hi:.0f}"


def arena_matchups(rows: list[dict], bot: str) -> dict[str, tuple[float, int]]:
    """A bot's record by character matchup in the Arena's rows, its own character first: {"X vs Y": (score, games)}."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        if bot not in (r["a"], r["b"]):
            continue
        for k, m in (r.get("matchups") or {}).items():
            x, _, y = k.partition(" vs ")
            key, s = (k, m["score"]) if r["a"] == bot else (f"{y} vs {x}", 1 - m["score"])
            a = acc.setdefault(key, [0.0, 0])
            a[0] += s * m["games"]
            a[1] += m["games"]
    return {k: (w / n, n) for k, (w, n) in sorted(acc.items()) if n}


CHARACTERS = ("Cowboy", "Ninja", "Wizard", "Mutant", "Robot")
ELO = math.log(10) / 400


def cowboy_gate(rel: dict) -> bool:
    """A Cowboy-focused promotion test: the new generation played Cowboy in every game ("Cowboy vs X" cells)."""
    keys = list((rel["gate"].get("matchups") or {}))
    return bool(keys) and all(k.startswith("Cowboy vs ") for k in keys)


def character_results(rows: list[dict], rels: list[dict]) -> dict[str, list[tuple]]:
    """{character: [(bot, opponent, score, games, source)]}: each game counted for the side that played that
    character. The Arena's games give every character; promotion tests give only Cowboy, and only the Cowboy-focused
    ones (the new generation is Cowboy in every game; older tests seat-swapped characters, so they say nothing about
    one character)."""
    out: dict[str, list[tuple]] = {c: [] for c in CHARACTERS}
    for r in rows:
        for ca, cb, _seat, s, _turns in r.get("games_detail") or []:
            out.setdefault(ca, []).append((r["a"], r["b"], float(s), 1, "arena"))
            out.setdefault(cb, []).append((r["b"], r["a"], 1 - float(s), 1, "arena"))
    for rel in rels:
        if rel.get("opponent_generation") is None or not cowboy_gate(rel):
            continue
        for m in (rel["gate"]["matchups"] or {}).values():
            if m.get("score") is not None and m.get("games"):
                out["Cowboy"].append((house(rel["generation"]), house(rel["opponent_generation"]), m["score"],
                                      int(m["games"]), "gate"))
    return out


def character_ratings(res: list[tuple], point: dict[str, float]) -> dict[str, tuple]:
    """{bot: (rating, low, high, games, provisional)} for one character: the rating that best explains the bot's
    results when it plays that character, against opponents at their overall ratings, shrunk toward its overall
    rating by PRIOR_GAMES draws; an approximate 95% interval. Provisional when every game is a promotion test's."""
    by: dict[str, list[tuple]] = {}
    for bot, opp, s, n, src in res:
        if bot in point and opp in point:
            by.setdefault(bot, []).append((point[opp], s, n, src))
    out = {}
    for bot, xs in by.items():
        data = [(r, s, n) for r, s, n, _ in xs] + [(point[bot], 0.5, PRIOR_GAMES)]
        th = point[bot]
        for _ in range(100):
            ps = [(n, s, 1 / (1 + 10 ** ((r - th) / 400))) for r, s, n in data]
            g = sum(n * (s - q) for n, s, q in ps) * ELO
            h = sum(n * q * (1 - q) for n, _, q in ps) * ELO * ELO
            step = max(-200.0, min(200.0, g / h)) if h else 0.0
            th += step
            if abs(step) < 1e-6:
                break
        info = sum(n * q * (1 - q) for n, _, q in ((n, s, 1 / (1 + 10 ** ((r - th) / 400))) for r, s, n in data))
        se = 1 / (ELO * math.sqrt(info)) if info else 400.0
        out[bot] = (th, th - 1.96 * se, th + 1.96 * se, sum(x[2] for x in xs), all(x[3] == "gate" for x in xs))
    return out


def bot_link(bot):
    return f'<a href="{L(bot_route(bot))}">{esc(label(bot))}</a>'


def subnav(active: str) -> str:
    return ('<nav class="subnav" aria-label="Lab">' + "".join(
        f'<a href="{L(p)}"{" class=on aria-current=page" if p == active else ""}>{esc(t)}</a>' for p, t in LAB_NAV)
        + "</nav>")


def chart(rels, point, ci) -> str:
    """The rating history: a wide chart, and a compact one for phones (CSS shows one)."""
    wide, small = _chart(rels, point, ci, 820, 340, False), _chart(rels, point, ci, 420, 300, True)
    return f'<div class="chartwrap"><div class="chart-wide">{wide}</div><div class="chart-small">{small}</div></div>'         if wide else ""


def _chart(rels, point, ci, w, h, compact) -> str:
    gens = [r["generation"] for r in rels if house(r["generation"]) in point]
    if not gens:
        return ""
    p = {g: point[house(g)] for g in gens}
    c = {g: ci[house(g)] for g in gens}
    pl, pr, pt, pb = (40, 10, 30, 34) if compact else (52, 18, 34, 40)
    lo = math.floor((min(c[g][0] for g in gens) - 40) / 200) * 200
    hi = math.ceil((max(c[g][1] for g in gens) + 40) / 200) * 200
    x = lambda g: pl + (g - gens[0]) / max(1, gens[-1] - gens[0]) * (w - pl - pr)  # noqa: E731
    y = lambda v: pt + (hi - v) / (hi - lo) * (h - pt - pb)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Rating by generation, with 95% '
             f'intervals and eras">']
    eras, start = [], gens[0]
    era_of = {r["generation"]: r["era"] for r in rels}
    for a, b in zip(gens, gens[1:] + [None]):
        if b is None or era_of[a] != era_of[b]:
            eras.append((start, a, era_of[a]))
            start = b
    for k, (a, b, e) in enumerate(eras):
        x0 = pl if k == 0 else (x(a) + x(eras[k - 1][1])) / 2
        x1 = w - pr if k == len(eras) - 1 else (x(b) + x(eras[k + 1][0])) / 2
        name = (f'set {e.get("candidate_set", "?")}' if compact else
                f'{e.get("settings", "?")} · set {e.get("candidate_set", "?")}')
        last_narrow = k == len(eras) - 1 and x1 - x0 < (60 if compact else 120)  # short last era: label from its right
        tx, anchor = (x1 - 6, "end") if last_narrow else (x0 + 8, "start")
        parts.append(f'<rect x="{x0:.1f}" y="{pt}" width="{x1 - x0:.1f}" height="{h - pt - pb}" class="band{k % 2}"/>'
                     f'<text x="{tx:.1f}" y="{pt - 10}" class="eralbl" text-anchor="{anchor}">{esc(name)}</text>')
    for v in range(lo, hi + 1, 200):
        parts.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{pl - 8}" y="{y(v) + 4:.1f}" class="yl">{v}</text>')
    for g in gens:
        parts.append(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{y(c[g][0]):.1f}" y2="{y(c[g][1]):.1f}" class="whisk"/>')
    parts.append(f'<polyline points="{" ".join(f"{x(g):.1f},{y(p[g]):.1f}" for g in gens)}" class="line"/>')
    for g in gens:
        parts.append(f'<circle cx="{x(g):.1f}" cy="{y(p[g]):.1f}" r="4.5" class="dot"><title>HustleBot g{g}: '
                     f'{p[g]:.0f} ({c[g][0]:.0f} to {c[g][1]:.0f})</title></circle>'
                     f'<text x="{x(g):.1f}" y="{h - pb + 20}" class="xl">{g}</text>')
    parts.append(f'<text x="{w - pr}" y="{h - 6}" class="xt">generation</text></svg>')
    return "".join(parts)


def what_changed(r, prev) -> list[str]:
    if prev is None:
        return ["The first published generation"]
    out = []
    if r["era"].get("settings") != prev["era"].get("settings"):
        out.append(f'Switched to {r["era"].get("settings")} match settings')
    if r["era"].get("candidate_set") != prev["era"].get("candidate_set"):
        out.append(f'Candidate set {r["era"].get("candidate_set")}: more moves to choose from')
    chars = lambda x: {c for k in (x["gate"]["matchups"] or {}) for c in k.split(" vs ")}  # noqa: E731
    if len(chars(r)) > len(chars(prev)):
        out.append(f"Promotion test widened to Cowboy against all {len(chars(r))} characters")
    return out or ["Same settings, more training"]


# --- the build -------------------------------------------------------------------------------------------------
def build(releases: Path, docs: Path, out: Path | None, arena: Path | None = None, site_url: str = "",
          spa: Path | None = None, spa_title: str = "yomibench", spa_head: str = "") -> dict[str, tuple[str, str, str]]:
    """Writes the site to `out` (pretty URLs) and/or the one-file preview to `spa`; returns the routes."""
    rels = sorted((json.loads(p.read_text(encoding="utf-8")) for p in releases.glob("gen-*.json")),
                  key=lambda r: r["generation"])
    entered, rows, aspec = load_arena(arena)
    if out is not None and out.exists():
        shutil.rmtree(out)
    dirs = [d / "media" for d in (out, spa.parent if spa else None) if d is not None]
    for media in dirs:
        media.mkdir(parents=True, exist_ok=True)
        for r in rels:
            for v in r["videos"]:
                for name in (v["file"], Path(v["file"]).with_suffix(".jpg").name):
                    if (releases / name).exists() and not (media / name).exists():
                        shutil.copyfile(releases / name, media / name)
    media = dirs[0]
    # Until every bot has ARENA_ONLY_GAMES games in this spec's era, the house generations' promotion tests fill in.
    arena_bots = {x for r in rows for x in (r["a"], r["b"])}
    pool = set(json.loads((arena / "house.json").read_text(encoding="utf-8"))) \
        if arena is not None and (arena / "house.json").exists() else {house(r["generation"]) for r in rels}
    arena_only = bool(rows) and all(sum(r["games"] for r in rows if b in (r["a"], r["b"])) >= ARENA_ONLY_GAMES
                                    for b in arena_bots | pool | {e["name"] for e in entered})
    rated_rels = [] if arena_only else rels
    point, ci = ratings(rated_rels, rows) if rated_rels else ratings_rows(rows, rels)
    res = results(rated_rels, rows)
    vids = games(rels, media)
    by_gen = {house(r["generation"]): r for r in rels}
    owners = {b["name"]: b.get("owner", "") for b in entered}
    board = sorted(point, key=lambda b: -point[b])
    played = {b: sum(n for i, j, _, n in res if b in (i, j)) for b in board}
    scored = {b: sum((s if i == b else 1 - s) * n for i, j, s, n in res if b in (i, j)) for b in board}
    scale = (min(ci[b][0] for b in board), max(ci[b][1] for b in board)) if board else (0, 1)
    cur = rels[-1] if rels else None
    base_bot = house(rels[0]["generation"]) if rels else None
    # HustleBot's weights are published two generations behind the current best: the newest two have none public.
    weights_lag = {house(r["generation"]) for r in rels[-WEIGHTS_LAG:]}
    routes: dict[str, tuple[str, str, str]] = {}  # path -> (title, section, body)

    # Arena (home) ------------------------------------------------------------------------------------------------
    hero = ""
    best, why = showcase_match(board, vids)
    if best:
        a, b = best["bots"]
        facts = [x for x in (best.get("matchup"), f"{turns_of(best)} turns" if turns_of(best) else "",
                             clock(best.get("seconds")), f'{(best.get("result") or "").title()} for {short(a)}') if x]
        stand = "".join(f'<a class="stand" href="{L(bot_route(x))}"><span class="rk">#{board.index(x) + 1}</span>'
                        f'<span class="nm">{esc(label(x))}</span><span class="rt">{point[x]:.0f}</span></a>'
                        for x in (a, b) if x in point)
        poster = f' poster="{M(best["poster"])}"' if best.get("poster") else ""
        hero = (f'<section class="hero"><div class="herotext"><p class="eyebrow">Showcase match</p>'
                f'<h2 class="herotitle">{esc(short(a))} <span class="vs">vs</span> {esc(short(b))}</h2>'
                f'<p class="lead">{esc(why)}.'
                f'{" " + esc(short(a)) + " played " + esc(best["new_side"]) + "." if best.get("new_side") else ""}</p>'
                f'<div class="chips">{"".join(f"<span class=chip>{esc(x)}</span>" for x in facts)}</div>'
                f'<div class="stands">{stand}</div></div>'
                f'<div class="herovid"><video data-autoplay muted loop playsinline controls preload="metadata"{poster} '
                f'src="{M(best["file"])}"></video></div></section>')
    if arena_only:
        groups = [(None, board)]
    else:  # promotion history: one block per settings era, newest era first, ranked within each
        key = lambda b: (json.dumps(by_gen[b]["era"], sort_keys=True) if b in by_gen  # noqa: E731
                         else "before" if is_house(b) else "entries")
        order = ["entries"] + [json.dumps(r["era"], sort_keys=True) for r in reversed(rels)] + ["before"]
        eras = sorted({key(b) for b in board}, key=lambda e: order.index(e) if e in order else len(order))
        groups = [(e, [b for b in board if key(b) == e]) for e in eras]
    trs = []
    for era_key, members in groups:
        if era_key is not None:
            name = ("Before the published releases" if era_key == "before" else "Arena entries" if era_key == "entries" else
                    ("Current era: " if cur and era_key == json.dumps(cur["era"], sort_keys=True) else "Earlier era: ")
                    + era_label(json.loads(era_key)))
            trs.append(f'<tr class="erarow"><td colspan="5">{esc(name)}</td></tr>')
        for k, b in enumerate(members, 1):
            owner = ('<span class="era">house bot, weights not public</span>' if b in weights_lag else "")                 if is_house(b) else f'<span class="era">by {esc(owners.get(b) or "—")}</span>'
            trs.append(f'<tr{" class=top1" if k == 1 else ""}><td class="rank">{k}</td>'
                       f'<th scope="row"><a class="botname" href="{L(bot_route(b))}"><span class="full">{esc(label(b))}'
                       f'</span><span class="shortn">{esc(short(b))}</span></a>{owner}</th>'
                       f'<td class="num rating"><b>{point[b]:.0f}</b><span class="ci">{ci_text(b, ci, base_bot)}</span>'
                       f'</td><td class="cicol">{cibar(ci[b][0], ci[b][1], point[b], scale)}</td>'
                       f'<td class="num hide-s">{played[b]}</td></tr>')
    spec_line = ""
    if aspec:
        arena_games = sum(r.get("games") or 0 for r in rows)
        spec_line = (f'<p class="specline"><span>Played on our simulator, as game '
                     f'{esc(aspec["game"]["version"])} (Steam build {esc(aspec["game"]["steam_build_id"])})</span>'
                     f'<span>harness {esc(aspec["harness"]["version"])}'
                     f'</span><span>candidate set {aspec["candidate_set"]}</span><span>spec <code>'
                     f'{spec_hash(aspec)}</code></span><span>{arena_games} Arena games</span></p>')
    chars = character_results(rows, rels)
    char_rt = {c: character_ratings(chars.get(c, []), point) for c in CHARACTERS}
    tabs, panels = [], []
    for i, c in enumerate(CHARACTERS):
        rt = char_rt[c]
        tabs.append(f'<input type="radio" name="chartab" id="ct-{c}"{" checked" if i == 0 else ""}>'
                    f'<label for="ct-{c}">{c}</label>')
        if rt:
            body_rows = "".join(
                f'<tr><td class="rank">{k}</td><th scope="row"><a class="botname" href="{L(bot_route(b))}">'
                f'<span class="full">{esc(label(b))}</span><span class="shortn">{esc(short(b))}</span></a>'
                + ('<span class="era">provisional: promotion tests only</span>' if prov else "")
                + f'</th><td class="num rating"><b>{th:.0f}</b><span class="ci">{lo:.0f} to {hi:.0f}</span></td>'
                f'<td class="num">{n}</td></tr>'
                for k, (b, (th, lo, hi, n, prov)) in enumerate(sorted(rt.items(), key=lambda kv: -kv[1][0]), 1))
            inner = (f'<div class="tablewrap"><table class="board"><thead><tr><th class="rank">#</th><th>Bot</th>'
                     f'<th class="num">Rating as {c}</th><th class="num">Games as {c}</th></tr></thead>'
                     f'<tbody>{body_rows}</tbody></table></div>')
        else:
            inner = f'<p class="small empty">No games as {c} yet. They come from the Arena\'s round-robin.</p>'
        panels.append(f'<div class="tp tp-{c}">{inner}</div>')
    by_character = (f'<section class="bychar"><div class="sechead"><h2>Rating by character</h2><span class="small">'
                    f'approximate 95% interval</span></div><p class="small note">How strong each bot is when it plays '
                    f'that character: the rating that best explains its results as that character, against opponents '
                    f'at their overall ratings. The Arena\'s games count for every character. Promotion tests count only '
                    f'for Cowboy, and only the Cowboy-focused ones, so Cowboy numbers from them alone are provisional. The intervals '
                    f'take the opponents\' ratings as given, so they are narrower than the overall ones.'
                    f'</p><div class="tabs">{"".join(tabs)}<div class="tabpanels">{"".join(panels)}</div></div>'
                    f'</section>')
    pooled = ("The overall rating counts all 25 character pairings equally, each played from both seats; ratings "
              "by character are below.")
    if arena_only:
        board_title = "Leaderboard"
        note = f"Rated on the Arena's own ranked games, all under the spec below. {pooled}"
    else:
        board_title = "House promotion history (provisional)"
        note = ("Until the Arena's own ranked matches are in, this table is HustleBot's history, and provisional: "
                "each generation rated from the promotion tests that published it, in the settings of its era. Those "
                "tests are Cowboy-focused (in the current era the new generation plays Cowboy in every game: 60 mirror "
                "pairs and 16 pairs against each other character), so compare ratings within an era. The Arena's "
                f"round-robin of the current era is under way ({sum(r['games'] for r in rows)} games so far) and takes "
                f"over once every bot has {ARENA_ONLY_GAMES}. {pooled}")
    body = (f'<header class="pagehead"><p class="eyebrow">Your Only Move Is HUSTLE · bot arena</p>'
            f'<h1>The Arena</h1><p class="lead">Phase 1 is a value-net division: every entry\'s nets play inside the '
            f'same search, ours, under one fixed set of ranked settings and the same thinking budget per move, so a '
            f'rating measures the nets, not a whole bot. HustleBot, our own bot, is the house entry: each of its '
            f'generations is listed as HustleBot gN.</p><p class="small">Rated games run on our private simulator of '
            f'the game, checked against the real game; every video is the game replayed in the real '
            f'game.</p></header>'
            f'{hero}<section><div class="sechead"><h2>{board_title}</h2><span class="small">Bradley–Terry rating with '
            f'an approximate 95% interval · g{rels[0]["generation"] if rels else "?"} = 0</span></div>'
            f'<p class="small note">{esc(note)}</p>'
            f'<div class="tablewrap"><table class="board"><thead><tr><th class="rank">#</th>'
            f'<th>Bot</th><th class="num">Rating</th><th class="cicol"><span class="sr">Interval</span></th>'
            f'<th class="num hide-s">Games</th></tr></thead><tbody>{"".join(trs)}</tbody>'
            f'</table></div>{spec_line}</section>{by_character}'
            f'<section class="cards three"><a class="card" href="{L("/submit/")}"><b>Submit a bot</b><span>The checks '
            f'are live; rated matches open soon.</span></a><a class="card" href="{L("/get-started/")}">'
            f'<b>Train your own</b><span>One PC and your copy of the game.</span></a><a class="card" '
            f'href="{L("/lab/")}"><b>The Lab</b><span>How HustleBot is made, generation by generation.</span></a>'
            f'</section>')
    routes["/"] = ("Arena", "arena", body)

    # a page per bot ---------------------------------------------------------------------------------------------
    for b in board:
        opp = {}
        for i, j, s, n in res:
            if b in (i, j):
                o, sc = (j, s) if i == b else (i, 1 - s)
                t = opp.setdefault(o, [0.0, 0])
                t[0] += sc * n
                t[1] += n
        opp_rows = "".join(f'<tr><th scope="row">{bot_link(o)}</th><td class="num">{pct(w / n)}</td>'
                           f'<td class="metercol">{meter(w / n)}</td><td class="num">{n}</td></tr>'
                           for o, (w, n) in sorted(opp.items(), key=lambda kv: -point.get(kv[0], 0)))
        r = by_gen.get(b)
        mu = ""
        if r:
            mu = "".join(f'<div class="mu"><span>{esc(k)}</span>{meter(m["score"])}<b>{pct(m["score"])}</b>'
                         f'<span class="dim">{m.get("games")}</span></div>'
                         for k, m in (r["gate"]["matchups"] or {}).items())
        own = [v for v in vids if v["bots"][0] == b]
        seen = [v for v in vids if v["bots"][1] == b]
        entry = next((e for e in entered if e["name"] == b), None)
        facts = []
        if r:
            facts = [f'Promoted {day(r.get("promoted_utc"))}', era_label(r["era"]),
                     f'model {(r.get("model_sha256") or "")[:12]}']
        elif entry:
            facts = [f'by {entry.get("owner", "")}', f'entered {day(entry.get("entered_utc"))}']
        elif is_house(b):
            facts = ["An early generation without a release of its own, rated from its results against later ones"]
        arena_mu = arena_matchups(rows, b)
        mine = [(c, char_rt[c][b]) for c in CHARACTERS if b in char_rt[c]]
        by_char = ("".join(tile(f"{th:.0f}", f"as {c}", f"{lo:.0f} to {hi:.0f} · {n} games"
                                + (" · provisional" if prov else "")) for c, (th, lo, hi, n, prov) in mine))
        notes = f'<p class="lead">{esc(entry["notes"])}</p>' if entry and entry.get("notes") else ""
        links = ((f' · <a href="{REPO}/releases/tag/gen-{r["generation"]}">release</a> · house bot, weights not public'
                  if b in weights_lag else
                  f' · <a href="{REPO}/releases/tag/gen-{r["generation"]}">release and weights</a>') if r else "")
        tiles = (tile(f"{point[b]:.0f}", "Rating", ci_text(b, ci, base_bot), True)
                 + tile(f"#{board.index(b) + 1}", "Rank", f"of {len(board)}") + tile(str(played[b]), "Games")
                 + tile(pct(scored[b] / played[b] if played[b] else None), "Score", "share of points"))
        body = (f'<nav class="crumbs"><a href="{L("/")}">← Arena</a></nav>'
                f'<header class="pagehead"><p class="eyebrow">{"The house bot" if is_house(b) else "Arena entry"}</p>'
                f'<h1>{esc(label(b))}</h1><p class="small">{esc(" · ".join(facts))}{links}</p>{notes}</header>'
                f'<div class="tiles">{tiles}</div>'
                + (f'<section><h2>Rating by character</h2><div class="tiles">{by_char}</div></section>' if by_char else "")
                + f'<section><h2>Results by opponent</h2><div class="tablewrap"><table><thead><tr><th>Opponent</th>'
                f'<th class="num">Score</th><th class="metercol"><span class="sr">Score bar</span></th>'
                f'<th class="num">Games</th></tr></thead><tbody>{opp_rows}</tbody></table></div></section>'
                + (f'<section><h2>Record by matchup in the Arena</h2><p class="small">Its character first, from its '
                   f'own games in the Arena; each pairing is played from both seats.</p><div class="mus">'
                   + "".join(f'<div class="mu"><span>{esc(k)}</span>{meter(s)}<b>{pct(s)}</b><span class="dim">{n}'
                             f'</span></div>' for k, (s, n) in arena_mu.items()) + '</div></section>' if arena_mu else "")
                + (f'<section><h2>Record by matchup in its promotion test</h2><p class="small">'
                   + ("Its character first (it played Cowboy in every game)" if cowboy_gate(r) else
                      "P1's character first; it played each pairing from both seats")
                   + f', against {esc(label(house(r["opponent_generation"])))}.</p>'
                   f'<div class="mus">{mu}</div></section>' if mu else "")
                + (f'<section><h2>Featured replays</h2><div class="grid">{"".join(video_card(v) for v in own)}</div>'
                   f'</section>' if own else "")
                + (f'<details class="panel"><summary>As the opponent · {len(seen)} games</summary><div class="grid">'
                   f'{"".join(video_card(v) for v in seen)}</div></details>' if seen else ""))
        routes[bot_route(b)] = (label(b), "arena", body)

    # Lab overview -----------------------------------------------------------------------------------------------
    writeups = sorted((docs / "lab").glob("*.md")) if (docs / "lab").is_dir() else []
    tiles = ""
    if cur:
        g, cb = cur["gate"]["pooled"], house(cur["generation"])
        tiles = (tile(f'g{cur["generation"]}', "Current best HustleBot", f'promoted {day(cur.get("promoted_utc"))}', True)
                 + tile(pct(g["score"]), "Promotion test",
                        f'vs g{cur["opponent_generation"]}, {pct(g["ci95"][0])}–{pct(g["ci95"][1])}, {g["games"]} games')
                 + (tile(f"{point[cb]:.0f}", "Rating", f"{ci[cb][0]:.0f} to {ci[cb][1]:.0f}") if cb in point else "")
                 + tile(str(len(rels)), "Generations",
                        f'published, {rels[0]["generation"]} to {rels[-1]["generation"]}'))
    line = []
    prev = None
    for r in rels:
        g = r["gate"]["pooled"]
        changed = what_changed(r, prev)
        new_era = prev is not None and r["era"] != prev["era"]
        line.append(f'<li class="{"newera" if new_era else ""}"><div class="when">{esc(day(r.get("promoted_utc")))}</div>'
                    f'<div class="node"></div><div class="what"><a class="gen" href="{L(bot_route(house(r["generation"])))}">'
                    f'HustleBot g{r["generation"]}</a><span class="result">scored <b>{pct(g.get("score"))}</b> vs g'
                    f'{r.get("opponent_generation")} · {g.get("games")} games</span>'
                    f'<span class="changed">{esc("; ".join(changed))}</span></div></li>')
        prev = r
    cards = [("/lab/showcase/", "Showcase", "Each generation's best games from the test that promoted it.", "play"),
             ("/lab/how-it-works/", "How it works", "Search over the game's real moves, scored by a value net.", "gear"),
             ("/lab/training/", "Training", "Self-play, training and promotion, on one PC or many.", "loop"),
             ("/lab/generations/", "Generations", "The rating history, era by era.", "chart")]
    cards += [(f"/lab/writeups/{w.stem.lower()}/", w.stem.replace("-", " ").title(), "Write-up", "doc") for w in writeups]
    card_html = "".join(f'<a class="card" href="{L(p)}">{ICONS[i]}<b>{esc(t)}</b><span>{esc(d)}</span></a>'
                        for p, t, d, i in cards)
    body = (subnav("/lab/") + f'<header class="pagehead"><p class="eyebrow">The Lab</p><h1>HustleBot: a bot that learns '
            f'by playing itself</h1><p class="lead">yomibench is an open-source fan project for <i>Your Only Move Is '
            f'HUSTLE</i>, made with the developer Ivy Sly\'s permission and credited below; it isn\'t an official '
            f'product of the game. HustleBot is its bot: each generation trains on its own games and is promoted only '
            f'if it beats the best generation before it. '
            f'This page updates itself on every promotion.</p><p class="small">HustleBot trains in a private '
            f'simulator, a Rust rewrite of the game\'s engine made with Ivy Sly\'s permission and checked against the '
            f'real game; every video on this site is replayed in the real game. Everything you can download runs on '
            f'the regular game: headless copies of your own Steam install.</p></header><div class="tiles">{tiles}</div>'
            f'<section class="cards">{card_html}</section>'
            f'<section><div class="sechead"><h2>The training story</h2><span class="small">newest first</span></div>'
            f'<ol class="timeline">{"".join(reversed(line))}</ol></section>{CREDITS}')
    routes["/lab/"] = ("Lab", "lab", body)

    # Lab: generations -------------------------------------------------------------------------------------------
    rows_ = [f'<tr class="erarow"><td colspan="5">Current era: {esc(era_label(cur["era"]))}</td></tr>'] if cur else []
    prev_era = None
    for r in reversed(rels):
        n, g = r["generation"], r["gate"]["pooled"]
        if prev_era is not None and r["era"] != prev_era:
            rows_.append(f'<tr class="erarow"><td colspan="5">Earlier era: {esc(era_label(r["era"]))}'
                         f'</td></tr>')
        prev_era = r["era"]
        hb = house(n)
        rows_.append(f'<tr><th scope="row">{bot_link(hb)}</th><td class="hide-s">{esc(day(r.get("promoted_utc")))}</td>'
                     + (f'<td class="num rating"><b>{point[hb]:.0f}</b><span class="ci">{ci_text(hb, ci, base_bot)}'
                        f'</span></td>' if hb in point else '<td class="num">—</td>')
                     + f'<td class="num">{pct(g.get("score"))}<span class="ci">vs g{r.get("opponent_generation")} · '
                     f'{g.get("games")}</span></td><td class="num hide-s">{kcount(r.get("games_trained"))}</td></tr>')
    body = (subnav("/lab/generations/") + f'<header class="pagehead"><h1>Generations</h1><p class="lead">Each '
            f'generation is promoted by beating the one before it. Ratings are a Bradley–Terry fit over the promotion '
            f'tests, with g{rels[0]["generation"] if rels else "?"} (the first published) at 0; bars show an '
            f'approximate 95% interval.</p></header>{chart(rels, point, ci)}'
            f'<p class="small">A chain of promotion tests tends to overstate gains: a promoted net is the one that won '
            f'its test. Settings changed between eras (shaded), so ratings compare cleanly only within one.</p>'
            f'<div class="tablewrap"><table><thead><tr><th>Generation</th><th class="hide-s">Promoted</th><th class="num">Rating</th>'
            f'<th class="num">Promotion test</th><th class="num hide-s">Games trained</th></tr></thead>'
            f'<tbody>{"".join(rows_)}</tbody></table></div>')
    routes["/lab/generations/"] = ("Generations", "lab", body)

    # Lab: showcase (format unchanged), the latest and one page per generation ----------------------------------
    def showcase_block(r):
        n = r["generation"]
        mine = [v for v in vids if v["gen"] == n]
        feat = [v for v in mine if v.get("group") == "featured"]
        cast = [v for v in mine if v.get("group") == "cast"]
        mu = "".join(f'<span class="chip">{esc(k)} <b>{pct(m["score"])}</b></span>'
                     for k, m in (r["gate"]["matchups"] or {}).items())
        g = r["gate"]["pooled"]
        inner = (f'<div class="sechead"><h2>HustleBot g{n} <span class="dim">vs g'
                 f'{r.get("opponent_generation")}</span></h2><span class="small">scored {pct(g.get("score"))} over '
                 f'{g.get("games")} games · {esc(day(r.get("promoted_utc")))}</span></div>'
                 f'<div class="chips">{mu}</div><div class="grid">{"".join(video_card(v) for v in feat)}</div>')
        if cast:
            inner += (f'<details class="panel"><summary>Cowboy vs the cast · {len(cast)} games</summary>'
                      f'<div class="grid">{"".join(video_card(v) for v in cast)}</div></details>')
        return inner

    intro = ('<p class="lead">For each generation: four Cowboy-mirror games from the test that promoted it (its longest '
             'win, biggest win, best comeback and worst loss), and for recent generations Cowboy against each other '
             'character. Each video says which side the new generation played.</p>')
    gen_cards = []
    for r in reversed(rels[:-1]):
        n = r["generation"]
        pv = next((v["poster"] for v in vids if v["gen"] == n and v.get("poster")), None)
        gen_cards.append(f'<a class="card{" withimg" if pv else ""}" href="{L(f"/lab/showcase/g{n}/")}">'
                         + (f'<img src="{M(pv)}" alt="" loading="lazy">' if pv else "")
                         + f'<b>HustleBot g{n}</b><span>scored {pct(r["gate"]["pooled"].get("score"))} vs g'
                         f'{r.get("opponent_generation")}</span></a>')
    if cur:
        body = (subnav("/lab/showcase/") + f'<header class="pagehead"><h1>Showcase</h1>{intro}</header>'
                f'<section class="gen">{showcase_block(cur)}</section>'
                + (f'<section><h2>Earlier generations</h2><div class="cards four">{"".join(gen_cards)}</div></section>'
                   if gen_cards else ""))
        routes["/lab/showcase/"] = ("Showcase", "lab", body)
    for r in rels:
        n = r["generation"]
        nav_ = "".join(f'<a href="{L(f"/lab/showcase/g{x}/")}"{" class=on aria-current=page" if x == n else ""}>{x}</a>'
                       for x in (q["generation"] for q in rels))
        body = (subnav("/lab/showcase/") + f'<header class="pagehead"><p class="eyebrow">Showcase</p>'
                f'<h1>HustleBot g{n}</h1></header><nav class="gennav" aria-label="Generations">{nav_}</nav>'
                f'<section class="gen">{showcase_block(r)}</section>')
        routes[f"/lab/showcase/g{n}/"] = (f"Showcase: generation {n}", "lab", body)

    # Get started ------------------------------------------------------------------------------------------------
    steps = [("Download the harness", "You need your own copy of the game on Steam, Python 3.11 or newer and uv. "
              "Clone the repository, then follow the setup guide.", "/get-started/setup/", "Setup guide",
              f"git clone {REPO}\ncd yomibench\nuv sync --extra train"),
             ("Train your own on one PC", "One command runs the whole loop on one machine: self-play in headless "
              "copies of your game, training, and a promotion test. Start from scratch or from a published generation "
              "(weights are published two generations behind the current best).",
              "/lab/training/", "Training guide", "uv run python tools/train_loop.py \\\n  --out ~/.yomibench/train"),
             ("Submit it", "Pack your best net (named in OUT/best.json) into a bundle and enter it in the Arena. The "
              "checks are live; rated matches open soon.", "/submit/", "How to submit",
              "uv run yomibench bundle pack \\\n  --name my-bot-1 \\\n  --owner YOUR-GITHUB-USERNAME \\\n  --value BEST_MODEL.npz")]
    body = ('<header class="pagehead"><p class="eyebrow">Get started</p><h1>Train a bot and enter it</h1><p class="lead">'
            'Three steps from your copy of the game to a bot in the Arena.</p></header><ol class="steps">'
            + "".join(f'<li><h2>{esc(t)}</h2><p>{esc(d)}</p><pre class="cmd" id="step-{i}">'
                      f'{copy_button(f"step-{i}", "Copy commands", one_line(c), small=True)}<code>{esc(c)}</code></pre>'
                      f'<a class="more" href="{L(h)}">{esc(lk)} →</a></li>'
                      for i, (t, d, h, lk, c) in enumerate(steps, 1)) + "</ol>")
    routes["/get-started/"] = ("Get started", "start", body)

    # docs and write-ups -----------------------------------------------------------------------------------------
    try:
        import markdown
        md = lambda t: markdown.markdown(t, extensions=["tables", "fenced_code"])  # noqa: E731
    except ImportError:
        md = lambda t: "<pre>" + esc(t) + "</pre>"  # noqa: E731

    def fix(t):
        def link(m):
            if m.group(1) in DOC_ROUTES:
                return f"]({L(DOC_ROUTES[m.group(1)] + (m.group(2) or ''))})"
            return f"]({REPO}/blob/main/docs/{m.group(1)}{m.group(2) or ''})"  # a doc that isn't a site page
        return re.sub(r"\]\((?:\.\./)?(?:docs/)?([A-Z]+\.md)(#[\w-]+)?\)", link, t)

    def with_copy(page: str, key: str) -> str:
        """Every code block gets a copy icon, in a wrapper outside the block's own scroll box, so the icon stays put
        when a long line scrolls and its label isn't clipped. It copies the block's text, with `\\` continuations
        joined (one_line)."""
        n = 0

        def box(m):
            nonlocal n
            n += 1
            pid = f"code-{key}-{n}"
            text = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).rstrip("\n")
            return (f'<div class="codebox"><pre id="{pid}">{m.group(1)}{m.group(2)}</code></pre>'
                    f'{copy_button(pid, "Copy", one_line(text), small=True, native_title=False)}</div>')
        return re.sub(r"<pre>(<code[^>]*>)(.*?)</code></pre>", box, page, flags=re.S)

    def doc_html(text, copy_key=""):  # tables scroll inside their own box
        page = md(fix(text)).replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
        return with_copy(page, copy_key) if copy_key else page

    sections = {"/get-started/setup/": "start", "/lab/training/": "lab", "/lab/how-it-works/": "lab", "/play/": "play",
                "/lab/mod/": "lab", "/submit/": "submit"}
    for name, path in DOC_ROUTES.items():
        src = docs / name
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        title = (re.search(r"^#\s+(.+)$", text, re.M) or [None, name[:-3].title()])[1]
        top = ""
        if path == "/submit/":  # the callout carries the status and the form link the doc repeats for GitHub
            text = re.sub(r"\A(# .+\n)\n> .+?\n\n", r"\1\n", text, flags=re.S)
            text = re.sub(r"\n\*\*\[Open the submission form\]\([^)]+\)\*\*\n", "\n", text)
            top = (f'<div class="callout"><div><b>The checks are live; rated matches open soon.</b><span>Submit now and '
                   f'the checks tell you whether your bot is ready. Valid bots are played once rated matches start.'
                   f'</span></div><a class="button" href="{FORM}">Open the submission form</a></div>'
                   '<div class="tiles">' + tile("Data only", "Phase 1", "value nets in our search, never code")
                   + tile("Same budget", "Every bot", "fixed ranked settings")
                   + tile("Automatic", "Checks", "size, contents, a test position") + "</div>")
        if path == "/get-started/setup/":
            top = f'<nav class="crumbs"><a href="{L("/get-started/")}">← Get started</a></nav>'
        sub = subnav(path) if sections[path] == "lab" else ""
        copy_key = {"/submit/": "submit", "/play/": "play"}.get(path, "")  # the pages people copy commands from
        routes[path] = (title, sections[path], f'{sub}{top}<article class="doc">{doc_html(text, copy_key)}</article>')
    for w in writeups:
        routes[f"/lab/writeups/{w.stem.lower()}/"] = (
            w.stem.replace("-", " ").title(), "lab",
            subnav("") + f'<article class="doc">{doc_html(w.read_text(encoding="utf-8"))}</article>')
    notfound = ('<header class="pagehead"><p class="eyebrow">404</p><h1>Page not found</h1><p class="lead">That page '
                f'isn\'t here. Try the <a href="{L("/")}">Arena</a>, the <a href="{L("/lab/")}">Lab</a> or '
                f'<a href="{L("/get-started/")}">Get started</a>.</p></header>')

    targets = set(re.findall(r"@@L:([^@#]*)", "".join(b for _, _, b in routes.values()) + notfound + nav_html("")))
    missing = sorted(targets - set(routes))
    if missing:  # a link to a page that isn't built: usually a doc missing from --docs
        raise SystemExit(f"links to routes that weren't built: {missing}")
    if out is not None:
        write_pages(out, routes, notfound, site_url)
    if spa is not None:
        write_spa(spa, routes, notfound, spa_title, spa_head)
    print(f"{len(rels)} generations, {len(entered)} entries, {len(board)} rated bots, {len(routes)} routes, "
          f"{sum(len(r['videos']) for r in rels)} videos -> {out or spa}")
    return routes


# --- output ----------------------------------------------------------------------------------------------------
def nav_html(section: str) -> str:
    def name(t):
        full, _, small = t.partition("|")
        return esc(full) if not small else f'<span class="full">{esc(full)}</span><span class="shortn">{esc(small)}</span>'
    return "".join(f'<a href="{L(p)}" data-section="{s}"{" class=on aria-current=page" if s == section else ""}>'
                   f'{name(t)}</a>' for p, t, s in NAV)


GITHUB = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0a8 8 0 0 0-2.5 15.6c.4 0 '
          '.5-.2.5-.4v-1.5c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.3 1.9.9 2.3.7.1-.5'
          '.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-4 0-.9.3-1.6.8-2.1-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.5 7.5 0 0 1 4 0c1.5-1 '
          '2.2-.8 2.2-.8.4 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.1 0 3.1-1.9 3.7-3.6 3.9.3.3.6.8.6 1.5v2.2c0 .2.1.5.6.4A8 8 0 0 0 '
          '8 0z"/></svg>')
HEADER = ('<a class="skip" href="#main">Skip to content</a><header class="top"><div class="bar"><a class="brand" '
          'href="{home}"><span class="mark" aria-hidden="true"></span><span class="brandtext">yomibench</span></a>'
          '<nav class="mainnav" aria-label="Site">{nav}</nav>'
          f'<a class="gh" href="{REPO}" aria-label="yomibench on GitHub">{GITHUB}</a></div></header>')
FOOTER = ('<footer class="foot"><div class="footin"><p><b>yomibench</b> is open source (MIT), '
          f'<a href="{REPO}">on GitHub</a>: a fan project, made with the developer\'s permission, not an official product. The videos show bot matches played in <i>Your Only Move Is HUSTLE</i>, Ivy '
          'Sly\'s game; no game files are distributed here, and the harness runs on your own copy from Steam.</p>'
          f'<p class="dim">{FOOT_CREDITS}</p></div></footer>')
# Starts the hero's video only when the viewer hasn't asked for reduced motion; otherwise it waits on its poster.
MOTION = ('<script>function yomiPlay(){if(!matchMedia("(prefers-reduced-motion: reduce)").matches)'
          'document.querySelectorAll("video[data-autoplay]").forEach(function(v){v.muted=true;v.play().catch(function()'
          '{})})}yomiPlay()</script>')
# Copy buttons: button[data-copy] names the element whose text it copies, unless data-copy-text gives the exact text
# (a command box's one-line form). Delegated, so it also works on the one-file preview's routes; where the clipboard
# API isn't available (a page opened from a file), it copies through a hidden text box.
COPY = ('<script>document.addEventListener("click",function(e){var b=e.target.closest("button[data-copy]");if(!b)return;'
        'var t=document.getElementById(b.dataset.copy);if(!t)return;var x=b.dataset.copyText||t.textContent;'
        'function done(ok){var o=b.dataset.label||b.getAttribute("aria-label"),m=ok?"Copied":"Copy failed";'
        'b.dataset.label=o;b.classList.remove("done","fail");b.classList.add(ok?"done":"fail");b.dataset.tip=m;'
        'b.setAttribute("aria-label",m);clearTimeout(b._t);b._t=setTimeout(function(){b.classList.remove("done","fail");'
        'b.dataset.tip=o;b.setAttribute("aria-label",o)},1600)}function pick(){var a=document.createElement("textarea");'
        'a.value=x;a.setAttribute("readonly","");a.style.position="fixed";a.style.opacity="0";document.body.appendChild(a);'
        'a.select();var ok=false;try{ok=document.execCommand("copy")}catch(_){}a.remove();done(ok)}'
        'if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(x).then(function(){done(true)},'
        'pick)}else{pick()}})</script>')


def write_pages(out: Path, routes, notfound: str, site_url: str) -> None:
    shutil.copyfile(HERE / "style.css", out / "style.css")
    base = site_url.rstrip("/")

    def doc(path, title, section, body, canonical=True, base_href=""):
        css = rel_to(path, "/style.css")
        head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                f'content="width=device-width,initial-scale=1,viewport-fit=cover">{base_href}'
                f'<title>{esc(title)} · yomibench</title><meta name="description" content="A bot arena and lab for '
                f'Your Only Move Is HUSTLE: bots that learn by playing themselves.">'
                + (f'<link rel="canonical" href="{esc(base + path)}">' if base and canonical else "")
                + f'<link rel="icon" href="{ICON}"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                f'<link rel="stylesheet" href="{FONTS}"><link rel="stylesheet" href="{css}"></head><body>')
        page = (head + HEADER.format(home=L("/"), nav=nav_html(section)) + f'<main id="main" class="wrap">{body}</main>'
                + FOOTER + MOTION + COPY + "</body></html>")
        return page

    for path, (title, section, body) in routes.items():
        d = out / path.strip("/")
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(resolve(doc(path, title, section, body), path, False), encoding="utf-8")
    # /arena/ is the home page
    (out / "arena").mkdir(exist_ok=True)
    (out / "arena" / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Arena · yomibench</title>'
        + (f'<link rel="canonical" href="{esc(base)}/">' if base else "")
        + '<meta http-equiv="refresh" content="0; url=../"></head><body><a href="../">The Arena</a></body></html>',
        encoding="utf-8")
    # 404: served at any depth, so its links are rooted at the site's base
    base_path = ("/" + base.split("://", 1)[-1].split("/", 1)[1].strip("/") + "/").replace("//", "/") \
        if base.count("/") > 2 else "/"
    base_href = f'<base href="{esc(base_path)}">' if base else ""  # without --site-url, links stay relative
    page = doc("/", "Page not found", "", notfound, canonical=False, base_href=base_href)
    (out / "404.html").write_text(resolve(page, "/", False), encoding="utf-8")


SPA_SCRIPT = """<script>
(function () {
  var view = document.getElementById("main"), routes = {};
  document.querySelectorAll("template[data-route]").forEach(function (t) { routes[t.dataset.route] = t; });
  function show() {
    var name = decodeURIComponent(location.hash.slice(1)) || "arena";
    var t = routes[name] || routes["not-found"];
    view.replaceChildren(t.content.cloneNode(true));
    if (window.yomiPlay) { yomiPlay(); }
    document.title = t.dataset.title + " · yomibench";
    document.querySelectorAll(".mainnav a").forEach(function (a) {
      var on = a.dataset.section === t.dataset.section;
      a.classList.toggle("on", on);
      if (on) { a.setAttribute("aria-current", "page"); } else { a.removeAttribute("aria-current"); }
    });
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", show);
  show();
})();
</script>"""


def write_spa(spa: Path, routes, notfound: str, title: str = "yomibench", head: str = "") -> None:
    """One file for a preview that can't serve folders: every route in a template, shown by the URL's hash. `title`
    and `head` (extra head markup) are the preview host's."""
    spa.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "style.css", spa.parent / "style.css")
    tpl = "".join(f'<template data-route="{hash_of(p)}" data-title="{esc(t)}" data-section="{s}">{b}</template>'
                  for p, (t, s, b) in routes.items())
    tpl += f'<template data-route="not-found" data-title="Page not found" data-section="">{notfound}</template>'
    page = (f'<title>{esc(title)}</title>\n{head}\n'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="{FONTS}">'
            '<link rel="stylesheet" href="style.css">\n'
            + HEADER.format(home=L("/"), nav=nav_html("arena")) + '<main id="main" class="wrap"></main>' + FOOTER
            + tpl + MOTION + COPY + SPA_SCRIPT)
    spa.write_text(resolve(page, "/", True), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--releases", type=Path, required=True)
    ap.add_argument("--docs", type=Path, default=HERE.parent / "docs")
    ap.add_argument("--arena", type=Path, default=HERE.parent / "arena")
    ap.add_argument("--out", type=Path, help="the site folder (pretty URLs); default _site unless --spa is given")
    ap.add_argument("--site-url", default="", help="the site's public URL, for canonical links and the 404 page")
    ap.add_argument("--spa", type=Path, help="also write the one-file preview with hash routes")
    ap.add_argument("--spa-title", default="yomibench", help="the preview's title")
    ap.add_argument("--spa-head", default="", help="extra head markup the preview host needs")
    a = ap.parse_args(argv)
    out = a.out if a.out or a.spa else HERE.parent / "_site"
    build(a.releases, a.docs, out, a.arena, a.site_url, a.spa, a.spa_title, a.spa_head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

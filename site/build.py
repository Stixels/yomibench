"""Build the static site: the Arena (home), History, the Record book, Replays, the Lab and Get started, every page at
its own URL.

    python site/build.py --releases DIR [--docs docs] [--arena arena] [--out _site] [--site-url URL]
    python site/build.py --releases DIR --spa FILE      # one-file preview with hash routes (#history-records)

DIR holds each release's files (gen-N.json, its videos and their poster stills), as the Pages workflow downloads
them; `arena` holds the Arena's data (settings.json, bots.json, results/*.jsonl; see arena/README.md).

Routes (pretty URLs: each is a folder with an index.html; links between pages are relative, so the site works under
any base path; --site-url adds canonical links and the 404 page's base):

    /                       the Arena: the reigning champion, the ladder, the champion's matchup chart, recent results
    /history/               the climb: rating and promotion margin by generation, and every generation in order
    /history/records/       the record book: longest game, quickest knockout, biggest comeback, closest promotion...
    /replays/               every published replay, filterable by generation, matchup, moment and result
    /bots/g13/              a bot's page; for a house generation it is also its showcase (Arena entries use their name)
    /lab/  /lab/how-it-works/  /lab/training/  /lab/mod/  /lab/writeups/NAME/
    /get-started/  /get-started/setup/  /submit/ (submissions paused: a short note)  /play/

Pages that moved keep their URL as a redirect: /arena/ (the home page), /lab/generations/ (History), /lab/showcase/
(the newest generation's page) and /lab/showcase/gN/ (/bots/gN/).

The one-file preview (--spa) holds every route and shows the one named by the URL's hash, the path with slashes as
hyphens (/history/records/ is #history-records; / is #arena); the back and forward buttons work.

Ratings: Bradley-Terry over every head-to-head (the house generations' promotion tests and anchor results from the
releases, and the Arena's rated matches), Elo-like scale with the oldest generation at 0, 95% intervals from a
parametric bootstrap (each resample refit from the point estimate until it converges). The hero's video is the longest
game between the two best bots (by turns when every candidate game records them, else by video length). A change of
settings preset or candidate set starts a new era.
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
              "PLAYING.md": "/play/", "MOD.md": "/lab/mod/"}
REPO = "https://github.com/Stixels/yomibench"
NAV = [("/", "Arena", "arena"), ("/history/", "History", "history"), ("/replays/", "Replays", "replays"),
       ("/lab/", "Lab", "lab"), ("/play/", "Play", "play")]
LAB_NAV = [("/lab/", "Overview"), ("/lab/how-it-works/", "How it works"), ("/lab/training/", "Training"),
           ("/lab/mod/", "The game mod")]
HISTORY_NAV = [("/history/", "The climb"), ("/history/records/", "Record book"), ("/replays/", "All replays")]
FONTS = ("https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Inter:wght@400;500;600;700"
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


def gen_of(bot: str) -> int | None:
    m = re.fullmatch(r"gen-(\d+)", bot)
    return int(m.group(1)) if m else None


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


# Reference bots from the releases' anchor results that aren't generations, with the name the site shows.
BASELINES = {"basic bot (plain search, no net)": "Plain search, no net"}


def baselines(rels: list[dict]) -> dict[str, list[tuple[int, float, int]]]:
    """{site name: [(generation, its score against the baseline, games)]}."""
    out: dict[str, list] = {}
    for r in rels:
        for name, a in (r.get("anchors") or {}).items():
            if name in BASELINES and a.get("score") is not None and a.get("games"):
                out.setdefault(BASELINES[name], []).append((r["generation"], a["score"], int(a["games"])))
    return out


# --- ratings ---------------------------------------------------------------------------------------------------
WEIGHTS_LAG = 2  # HustleBot's weights are published this many generations behind the current best
ARENA_ONLY_GAMES = 100  # per bot, in the current spec's era, before the Arena rates on its own games alone
# When each Arena spec's rating era started, and the spec it replaced: {spec hash: (date, previous spec hash)}.
# b33e39c692b8 and ea7c0f29f0cf (both Nov 20) played no rated games, so the era after f6910abccc01 starts here.
ERA_STARTS = {"d9feb02ae9da": ("Nov 20, 2026", "f6910abccc01")}
PRIOR_GAMES = 2  # each player also drew this many games against a fixed reference: keeps a perfect score finite
BOOTSTRAPS = 200


def _sig(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _solve(h: list[list[float]], g: list[float]) -> list[float]:
    """h x = g for a symmetric positive definite h (Cholesky)."""
    n = len(g)
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            v = h[i][j] - sum(c[i][k] * c[j][k] for k in range(j))
            c[i][j] = math.sqrt(v) if i == j else v / c[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (g[i] - sum(c[i][k] * y[k] for k in range(i))) / c[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(c[k][i] * x[k] for k in range(i + 1, n))) / c[i][i]
    return x


def _strengths(res, players, init=None, iters=100, tol=1e-10) -> dict[str, float]:
    """The Bradley-Terry maximum-likelihood strengths; draws count half (the score). Each player also drew
    PRIOR_GAMES games against a fixed reference of strength 1, which keeps a perfect score finite and pins the scale.
    Newton's method on the log-strengths: the likelihood is concave, so it converges in a few steps even along a long
    chain of head-to-heads (each generation against the one before), where minorization-maximization needs thousands
    of iterations. Warm-started from `init` when given."""
    idx = {p: k for k, p in enumerate(players)}
    games = [(idx[i], idx[j], s, n) for i, j, s, n in res]
    x = [math.log(init[p]) if init else 0.0 for p in players]
    for _ in range(iters):
        g = [PRIOR_GAMES * (0.5 - _sig(v)) for v in x]
        h = [[0.0] * len(x) for _ in x]
        for k, v in enumerate(x):
            q = _sig(v)
            h[k][k] = PRIOR_GAMES * q * (1 - q)
        for i, j, s, n in games:
            q = _sig(x[i] - x[j])
            g[i] += n * (s - q)
            g[j] -= n * (s - q)
            w = n * q * (1 - q)
            h[i][i] += w
            h[j][j] += w
            h[i][j] -= w
            h[j][i] -= w
        step = _solve(h, g)
        big = max(abs(d) for d in step)
        damp = 2.0 / big if big > 2.0 else 1.0  # far from the optimum: a bounded step
        x = [v + damp * d for v, d in zip(x, step)]
        if big < tol:
            break
    return {p: math.exp(x[k]) for p, k in idx.items()}


def bradley_terry(res, players, base, init=None, tol=1e-10):
    """Ratings on the Elo-like scale with `base` at 0, and the strengths (to warm-start a refit)."""
    th = _strengths(res, players, init, tol=tol)
    return {p: 400 * math.log10(th[p] / th[base]) for p in players}, th


def _fit(res, base):
    players = sorted({p for i, j, _, _ in res for p in (i, j)})
    if not players:
        return {}, {}
    base = base if base in players else players[0]
    point, th = bradley_terry(res, players, base)
    rng = random.Random(0)
    boots = {p: [] for p in players}
    for _ in range(BOOTSTRAPS):
        sample = [(i, j, sum(rng.random() < s for _ in range(n)) / n, n) for i, j, s, n in res]
        b, _ = bradley_terry(sample, players, base, init=th, tol=1e-7)
        for p in players:
            boots[p].append(b[p])
    lo, hi = round(0.025 * BOOTSTRAPS), round(0.975 * BOOTSTRAPS) - 1
    return point, {p: (sorted(v)[lo], sorted(v)[hi]) for p, v in boots.items()}


def ratings_rows(rows, rels):
    """Ratings from the Arena's rows alone, the oldest generation at 0."""
    return _fit(results([], rows), house(min(r["generation"] for r in rels)) if rels else "")


def ratings(rels, rows=()):
    res = results(rels, rows)
    return _fit(res, house(min(r["generation"] for r in rels)) if rels else (res[0][0] if res else ""))


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
        for k, v in enumerate(r["videos"]):
            poster = Path(v["file"]).with_suffix(".jpg").name
            out.append({**v, "gen": r["generation"], "bots": (house(r["generation"]), house(v["opponent_generation"])),
                        "seconds": mp4_seconds(media / v["file"]), "id": f"g{r['generation']}-{k + 1}",
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


SLOTS = {"longest win": "Longest win", "biggest win": "Biggest win", "comeback": "Best comeback", "worst loss": "Worst loss"}


def sides(v) -> tuple[str, str]:
    """(the new generation's character, its opponent's), from the matchup (P1 first) and the side it played."""
    a, _, b = (v.get("matchup") or "").partition(" vs ")
    return (b, a) if (v.get("new_side") or "").startswith("right") else (a, b)


def moment(v) -> str:
    """The game's headline from its `why` ("Cowboy mirror, best comeback: from -1230 hp" -> "From −1,230 hp")."""
    text = (v.get("why") or "").partition(": ")[2] or (v.get("why") or "")
    text = re.sub(r"(\d{4,})", lambda m: f"{int(m.group(1)):,}", text).replace("-", "−")
    return text[:1].upper() + text[1:]


# --- small pieces ----------------------------------------------------------------------------------------------
def pct(x):
    return "—" if x is None else f"{round(100 * x)}%"


def when(utc):
    try:
        return datetime.datetime.fromisoformat(str(utc).replace("Z", "+00:00"))
    except ValueError:
        return None


def day(utc) -> str:
    d = when(utc)
    return f"{d:%b} {d.day}, {d.year}" if d else ""


def mday(utc) -> str:
    d = when(utc)
    return f"{d:%b} {d.day}" if d else ""


def span_text(seconds: float) -> str:
    h = seconds / 3600
    if h < 1:
        return f"{round(seconds / 60)} min"
    if h < 36:
        return f"{h:.1f} hours".replace(".0 ", " ")
    return f"{h / 24:.1f} days".replace(".0 ", " ")


def era_label(e):
    return f"{e.get('settings', '?')} settings · set {e.get('candidate_set', '?')}"


def era_short(e):
    return f"{str(e.get('settings', '?')).title()} · set {e.get('candidate_set', '?')}"


def kcount(n) -> str:
    return "—" if not n else (f"{n / 1000:.0f}k" if n >= 10000 else f"{n:,}")


def big_count(n) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else kcount(n)


def signed(x) -> str:
    return f"+{x:.0f}" if x >= 0.5 else (f"−{-x:.0f}" if x <= -0.5 else "±0")


ICONS = {k: f'<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">{v}</svg>' for k, v in {
    "play": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9l5 3-5 3z" class="fill"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1'
            'M4.9 19.1L7 17M17 7l2.1-2.1"/>',
    "loop": '<path d="M4 12a8 8 0 0 1 14-5.3M20 12a8 8 0 0 1-14 5.3"/><path d="M18 3v4h-4M6 21v-4h4"/>',
    "chart": '<path d="M3 20h18"/><path d="M5 16l4-5 4 3 6-8"/>',
    "doc": '<path d="M6 3h9l4 4v14H6z"/><path d="M9 12h7M9 16h7"/>',
    "trophy": '<path d="M8 4h8v5a4 4 0 0 1-8 0z"/><path d="M8 6H5a3 3 0 0 0 3 4M16 6h3a3 3 0 0 1-3 4M12 13v4M8 20h8"/>',
    "mod": '<rect x="3" y="7" width="18" height="11" rx="3"/><path d="M7 12.5h4M9 10.5v4M15 12h.01M17.5 13.5h.01"/>',
    "bolt": '<path d="M13 3L5 13h6l-1 8 8-10h-6z"/>',
}.items()}
CROWN = ('<svg class="crown" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 8l4.5 4L12 5l4.5 7L21 8l-2 11H5z"/>'
         '</svg>')
PLAY_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l11-6.5z"/></svg>'


def tile(value, name, sub="", big=False) -> str:
    return (f'<div class="tile{" big" if big else ""}"><span class="tname">{esc(name)}</span>'
            f'<span class="tval">{value}</span>{f"<span class=tsub>{sub}</span>" if sub else ""}</div>')


def badge(bot: str, cur_era: bool = False, champ: bool = False) -> str:
    g = gen_of(bot)
    text = f"g{g}" if g is not None else (re.sub(r"[^A-Za-z0-9]", "", bot)[:2] or "?").upper()
    cls = "gb" + (" cur" if cur_era else "") + (" champ" if champ else "") + ("" if g is not None else " ent")
    return f'<span class="{cls}" aria-hidden="true">{esc(text)}</span>'


# Prompts a visitor pastes into a coding agent (Claude Code, Codex, Cursor...). They point the agent at the repo's
# own guides rather than restating them, so they stay right when a command changes.
PROMPT_START = f"""Help me train a bot for yomibench. yomibench is an open-source bot arena \
for the game Your Only Move Is HUSTLE: {REPO}

Work from the project's own guides; don't guess commands.

1. Check what I need: git, Python 3.11 or newer, uv, and the game installed through Steam (the repository's \
docs/SETUP.md says which versions and systems work). Tell me what's missing and how to install it.
2. Clone the repository, then change into the clone as a separate command (Windows PowerShell may not run two \
commands joined with &&). Read README.md, docs/SETUP.md and docs/TRAINING.md.
3. Install with `uv sync --extra train --extra test` and run the tests (`uv run pytest -q`). Tell me if anything \
fails.
4. Before you start training, tell me how much of my PC it will use (cores and memory) and roughly how long until \
it promotes a first model of my own (the loop keeps going until I stop it), and wait for my go-ahead. Training runs \
copies of the game in the background.

Explain each step in plain words. Never ask me for passwords or tokens; I'll sign in to GitHub myself."""


COPY_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><g class="cp"><rect x="8" y="8" width="14" height="14" rx="2"/>'
             '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></g><path class="ok" d="M5 12l5 5L20 7"/></svg>')
BOT_ICON = ('<svg class="boticon" viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="8" width="16" height="12" rx="3"/>'
            '<path d="M12 4v4M9 13h.01M15 13h.01M9 17h6"/></svg>')


def copy_button(target: str, label: str, text: str | None = None, small: bool = False,
                native_title: bool | None = None) -> str:
    """An icon copy button (COPY wires it). It copies `text` when given, else the text of the element `target`.
    A button inside a box that clips overflow (the Get started command boxes) can't show the CSS label, so it gets a
    native title instead; that's the default for small buttons."""
    extra = f' data-copy-text="{esc(text)}"' if text is not None else ""
    if small if native_title is None else native_title:
        extra += f' title="{esc(label)}"'
    return (f'<button type="button" class="iconbtn{" sm" if small else ""}" data-copy="{target}"{extra} '
            f'aria-label="{esc(label)}" data-tip-css="{esc(label)}">{COPY_ICON}</button>')


def one_line(cmd: str) -> str:
    """Commands with their `\\` line continuations joined, so they paste into PowerShell as well as a POSIX shell."""
    return re.sub(r"[ \t]*\\\n[ \t]*", " ", cmd)


def agent_prompt(key: str, title: str, blurb: str, prompt: str) -> str:
    """The agent prompt as one slim bar, closed by default. The copy button sits outside the <summary> (a button
    inside one is a control within a control), positioned over the bar, so it copies without opening it."""
    pid = f"prompt-{key}"
    return (f'<div class="agent"><details><summary><span class="agentt">{BOT_ICON}<b>{esc(title)}</b></span>'
            f'<span class="agentd">{esc(blurb)}</span><span class="tog"><span class="s">Show<span class="wide"> prompt'
            f'</span></span><span class="h">Hide</span></span></summary><div class="agentbody">'
            f'<pre class="cmd prompt" id="{pid}" tabindex="0"><code>'
            f'{esc(prompt)}</code></pre></div></details>{copy_button(pid, "Copy prompt")}</div>')


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


def cast_gate(rel: dict) -> bool:
    """The sequential promotion test (generation 16 on): both bots play every character from both seats, so the
    matchup cells cover all 25 pairings (weighted toward Cowboy, as training is)."""
    keys = list((rel["gate"].get("matchups") or {}))
    return len(keys) == len(CHARACTERS) ** 2 and any(not k.startswith("Cowboy vs ") for k in keys)


def gate_history(rels: list[dict]) -> str:
    """The kinds of promotion test the published generations had, oldest first."""
    def span(gs):
        return f"generation {gs[0]}" if len(gs) == 1 else f"generations {gs[0]} to {gs[-1]}"
    cow = [r["generation"] for r in rels if cowboy_gate(r)]
    cast = [r for r in rels if cast_gate(r)]
    early = [r["generation"] for r in rels if not cowboy_gate(r) and not cast_gate(r)
             and (not cow or r["generation"] < cow[0])]
    out = []
    if early:
        out.append(f"For {span(early)} the test played Cowboy and Ninja from both seats.")
    if cow:
        out.append(f"For {span(cow)} it was Cowboy-focused: the new generation played Cowboy in every game, mostly "
                   f"the mirror.")
    if cast:
        last = cast[-1]
        pairs = int(last["gate"]["pooled"]["games"]) // 2
        out.append(f"From generation {cast[0]['generation']} both bots play every character from both seats, "
                   f"weighted toward Cowboy as in training (generation {last['generation']}: {pairs} pairs, "
                   f"{2 * pairs} games).")
    return " ".join(out)


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


def subnav(items, active: str, name: str) -> str:
    return (f'<nav class="subnav" aria-label="{esc(name)}">' + "".join(
        f'<a href="{L(p)}"{" class=on aria-current=page" if p == active else ""}>{esc(t)}</a>' for p, t in items)
        + "</nav>")


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
    if cast_gate(r) and not cast_gate(prev):
        out.append("Promotion test with every character on both sides")
    return out or ["Same settings, more training"]


# --- replays ---------------------------------------------------------------------------------------------------
def replay_card(v, *, show_gen: bool = True, feature: bool = False) -> str:
    """A replay as a card: its poster with a play button (a link to the video, which the page's script swaps for an
    inline player), the moment it shows, who played which side, and filter data for the Replays page."""
    res = (v.get("result") or "").lower()
    mine, opp = sides(v)
    a, b = v["bots"]
    seat = (v.get("new_side") or "").replace("left, P1", "P1, left").replace("right, P2", "P2, right")
    turns = turns_of(v)
    facts = " · ".join(x for x in (f"{turns} turns" if turns else "", clock(v.get("seconds"))) if x)
    slot = SLOTS.get(v.get("slot") or "", (v.get("slot") or "").capitalize())
    img = (f'<img src="{M(v["poster"])}" alt="" loading="lazy" decoding="async">' if v.get("poster")
           else '<span class="noposter"></span>')
    mu = "mirror" if mine == opp else f"vs {opp}"
    title = f"{short(a)} {slot.lower()}: {moment(v)}"
    return (f'<article class="rcard{" feat" if feature else ""}" id="{v["id"]}" data-gen="{v["gen"]}" '
            f'data-mu="{esc(mine)} {esc(mu)}" data-slot="{esc(v.get("slot") or "")}" data-result="{esc(res)}">'
            f'<a class="lite" href="{M(v["file"])}" data-poster="{M(v["poster"]) if v.get("poster") else ""}" '
            f'aria-label="Play: {esc(title)}">{img}<span class="playbtn">{PLAY_ICON}</span>'
            + (f'<span class="dur">{esc(clock(v.get("seconds")))}</span>' if v.get("seconds") else "")
            + f'</a><div class="rbody"><div class="rtop"><span class="slot">{esc(slot)}</span>'
            f'<span class="res {esc(res)}">{esc(res.title())}</span></div>'
            f'<h3 class="moment">{esc(moment(v))}</h3>'
            f'<p class="fight"><span class="me"><b>{esc(short(a))}</b> {esc(mine)}</span><span class="vs">vs</span>'
            f'<span class="them">{esc(short(b))} {esc(opp)}</span></p>'
            f'<p class="meta">{esc(short(a))} played {esc(seat)}{" · " + esc(facts) if facts else ""}'
            + (f' · <a href="{L(bot_route(a))}">{esc(label(a))}</a>' if show_gen else "")
            + '</p></div></article>')


def replay_grid(vs, **kw) -> str:
    return f'<div class="rgrid">{"".join(replay_card(v, **kw) for v in vs)}</div>'


# --- the matchup heatmap ---------------------------------------------------------------------------------------
CHAR_TILE_GAMES = 10  # a bot page shows its rating as a character from this many games on
THIN = 20  # cells with fewer games than this are hatched: too few to read much into


def heatmap(rel: dict, name: str) -> str:
    """A promotion test's score by character pairing, as a table: rows are the first character, columns the second.
    Diverging around 50%: the new generation's colour when it scored more, its opponent's when less."""
    mus = {k: m for k, m in (rel["gate"].get("matchups") or {}).items() if m.get("score") is not None}
    if not mus:
        return ""
    cow = cowboy_gate(rel)
    firsts = [c for c in CHARACTERS if any(k.partition(" vs ")[0] == c for k in mus)]
    seconds = [c for c in CHARACTERS if any(k.partition(" vs ")[2] == c for k in mus)]
    me, opp = short(house(rel["generation"])), short(house(rel["opponent_generation"]))
    head = "".join(f'<th scope="col">{"vs " if cow else ""}{esc(c)}</th>' for c in seconds)
    body = []
    for a in firsts:
        cells = []
        for b in seconds:
            m = mus.get(f"{a} vs {b}")
            if not m:
                cells.append('<td class="hmc none"><span class="sr">not played</span></td>')
                continue
            s, n = float(m["score"]), int(m.get("games") or 0)
            k = min(100 if n >= THIN else 40, round(100 * abs(s - 0.5) / 0.3))  # few games: a paler cell
            lo, hi = (m.get("ci95") or [None, None])[:2]
            where = f"{me} as {a} vs {opp} as {b}" if cow else f"P1 {a} vs P2 {b}, both bots on both sides"
            tip = (f"{me} scored {pct(s)}\n{where}\n{n} games"
                   + (f" · 95% {pct(lo)} to {pct(hi)}" if lo is not None else ""))
            cls = f'hmc {"pos" if s >= 0.5 else "neg"}{" thin" if n < THIN else ""}{" strong" if k >= 55 else ""}'
            cells.append(f'<td class="{cls}" style="--k:{k}%" tabindex="0" data-tip="{esc(tip)}">'
                         f'<b>{round(100 * s)}</b><small>{n}</small></td>')
        body.append(f'<tr><th scope="row">{"as " if cow else ""}{esc(a)}</th>{"".join(cells)}</tr>')
    corner = "" if cow else '<span class="axes">P1 ↓ · P2 →</span>'
    legend = (f'<div class="hmlegend" aria-hidden="true"><span>{esc(opp)} better</span><i class="ramp"></i>'
              f'<span>{esc(me)} better</span><span class="thinkey"><i></i>under {THIN} games</span></div>')
    return (f'<div class="hm-wrap"><table class="hm" aria-label="{esc(name)}"><thead><tr><th>{corner}</th>{head}</tr>'
            f'</thead><tbody>{"".join(body)}</tbody></table>{legend}</div>')


# --- charts ----------------------------------------------------------------------------------------------------
def _ticks(lo, hi, step):
    return [v for v in range(int(math.floor(lo / step) * step), int(math.ceil(hi / step) * step) + 1, step)]


def rating_chart(gens, point, ci, era_of, champ, w, h, compact) -> str:
    """Rating by generation: the line, its 95% band, the eras as shaded columns. Each point links to its page."""
    if not gens:
        return ""
    p = {g: point[house(g)] for g in gens}
    c = {g: ci[house(g)] for g in gens}
    pl, pr, pt, pb = (40, 14, 30, 30) if compact else (52, 64, 34, 36)
    lo = min(min(c[g][0] for g in gens), min(p.values())) - 40
    hi = max(max(c[g][1] for g in gens), max(p.values())) + 40
    step = 200 if hi - lo > 600 else 100
    ticks = _ticks(lo, hi, step)
    lo, hi = ticks[0], ticks[-1]
    xs = {g: pl + k * (w - pl - pr) / max(1, len(gens) - 1) for k, g in enumerate(gens)}
    y = lambda v: pt + (hi - v) / (hi - lo) * (h - pt - pb)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="HustleBot rating by generation, with 95% '
           f'intervals and eras">']
    eras, start = [], 0
    for k in range(1, len(gens) + 1):
        if k == len(gens) or era_of(gens[k]) != era_of(gens[start]):
            eras.append((gens[start], gens[k - 1], era_of(gens[start])))
            start = k
    half = (w - pl - pr) / max(1, len(gens) - 1) / 2
    for k, (a, b, e) in enumerate(eras):
        x0, x1 = max(pl - 6, xs[a] - half), min(w - pr + 6, xs[b] + half)
        name = "early" if e is None else (f'set {e.get("candidate_set")}' if compact else era_short(e))
        fits = len(name) * (6.4 if compact else 6.8) < x1 - x0 - 8
        out.append(f'<rect x="{x0:.1f}" y="{pt - 22}" width="{x1 - x0:.1f}" height="{h - pt - pb + 22}" '
                   f'class="band{k % 2}{" now" if k == len(eras) - 1 else ""}"/>'
                   + (f'<text x="{x0 + 7:.1f}" y="{pt - 8}" class="eralbl">{esc(name)}</text>' if fits else ""))
    for v in ticks:
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"zero" if v == 0 else "grid"}"/>'
                   f'<text x="{pl - 8}" y="{y(v) + 4:.1f}" class="yl">{v}</text>')
    band = " ".join(f"{xs[g]:.1f},{y(c[g][1]):.1f}" for g in gens) + " " + \
        " ".join(f"{xs[g]:.1f},{y(c[g][0]):.1f}" for g in reversed(gens))
    out.append(f'<polygon points="{band}" class="ciband"/>')
    out.append(f'<polyline points="{" ".join(f"{xs[g]:.1f},{y(p[g]):.1f}" for g in gens)}" class="line"/>')
    every = 1 if not compact or len(gens) <= 10 else 2
    for k, g in enumerate(gens):
        is_champ = house(g) == champ
        tip = f"HustleBot g{g}: {p[g]:.0f}\n95% {c[g][0]:.0f} to {c[g][1]:.0f}" + \
            (f"\n{era_short(era_of(g))}" if era_of(g) else "\nbefore the published releases")
        out.append(f'<circle cx="{xs[g]:.1f}" cy="{y(p[g]):.1f}" r="{6 if is_champ else 4}" '
                   f'class="dot{" champ" if is_champ else ""}"/>')
        if k % every == 0 or k == len(gens) - 1:
            out.append(f'<text x="{xs[g]:.1f}" y="{h - pb + 20}" class="xl{" on" if is_champ else ""}">{g}</text>')
        out.append(f'<a href="{L(f"/bots/g{g}/")}" aria-label="HustleBot g{g}, rating {p[g]:.0f}">'
                   f'<rect x="{xs[g] - half:.1f}" y="{pt}" width="{2 * half:.1f}" height="{h - pt - pb}" class="hit" '
                   f'data-tip="{esc(tip)}"/></a>')
    last = gens[-1]
    if not compact:
        out.append(f'<text x="{xs[last] + 12:.1f}" y="{y(p[last]) + 4:.1f}" class="endlbl">g{last} · {p[last]:.0f}</text>')
    out.append("</svg>")
    return "".join(out)


def margin_chart(rels, w, h, compact) -> str:
    """Each promotion test's score against the generation before, as a bar up from 50% with its 95% interval."""
    rs = [r for r in rels if r["gate"]["pooled"].get("score") is not None]
    if not rs:
        return ""
    pl, pr, pt, pb = (40, 14, 14, 30) if compact else (52, 64, 16, 36)
    top = max(max((r["gate"]["pooled"].get("ci95") or [0, r["gate"]["pooled"]["score"]])[1] for r in rs),
              max(r["gate"]["pooled"]["score"] for r in rs))
    hi = math.ceil(top * 10 + 0.2) / 10
    lo = 0.45
    y = lambda v: pt + (hi - v) / (hi - lo) * (h - pt - pb)  # noqa: E731
    step = (w - pl - pr) / max(1, len(rs) - 1)
    bw = min(26, step * 0.56)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Promotion test score by generation, '
           f'against the generation before">']
    v = 0.5
    while v <= hi + 1e-9:
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"zero" if abs(v - .5) < 1e-9 else "grid"}"/>'
                   f'<text x="{pl - 8}" y="{y(v) + 4:.1f}" class="yl">{round(v * 100)}%</text>')
        v += 0.1
    every = 1 if not compact or len(rs) <= 10 else 2
    for k, r in enumerate(rs):
        g = r["gate"]["pooled"]
        x = pl + k * step
        s = g["score"]
        y0, y1 = y(0.5), y(max(s, 0.5))
        rad = min(4, (y0 - y1) / 2)
        out.append(f'<path d="M{x - bw / 2:.1f},{y0:.1f} V{y1 + rad:.1f} Q{x - bw / 2:.1f},{y1:.1f} {x - bw / 2 + rad:.1f},'
                   f'{y1:.1f} H{x + bw / 2 - rad:.1f} Q{x + bw / 2:.1f},{y1:.1f} {x + bw / 2:.1f},{y1 + rad:.1f} V{y0:.1f} Z" '
                   f'class="bar{" last" if k == len(rs) - 1 else ""}"/>')
        ci = g.get("ci95")
        if ci:
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(ci[0]):.1f}" y2="{y(ci[1]):.1f}" class="whisk"/>')
        if k % every == 0 or k == len(rs) - 1:
            out.append(f'<text x="{x:.1f}" y="{h - pb + 20}" class="xl">{r["generation"]}</text>')
        tip = (f"g{r['generation']} vs g{r.get('opponent_generation')}: {pct(s)}"
               + (f"\n95% {pct(ci[0])} to {pct(ci[1])}" if ci else "") + f"\n{g.get('games')} games · {mday(r.get('promoted_utc'))}")
        gpath = f"/bots/g{r['generation']}/"
        out.append(f'<a href="{L(gpath)}" aria-label="g{r["generation"]}: {pct(s)}">'
                   f'<rect x="{x - step / 2:.1f}" y="{pt}" width="{step:.1f}" height="{h - pt - pb}" class="hit" '
                   f'data-tip="{esc(tip)}"/></a>')
    out.append("</svg>")
    return "".join(out)


def dual(fn, *args, wide=(960, 340), small=(420, 300)) -> str:
    """A chart drawn twice: wide, and compact for phones (CSS shows one)."""
    a, b = fn(*args, *wide, False), fn(*args, *small, True)
    return f'<div class="chartwrap viz"><div class="chart-wide">{a}</div><div class="chart-small">{b}</div></div>' if a else ""


# --- the record book -------------------------------------------------------------------------------------------
def record_book(rels, vids, point) -> dict[str, list[dict]]:
    """{group: [{title, value, holder, detail, video}]}: superlatives across the published generations and replays."""
    fights, promos, training = [], [], []
    timed = [v for v in vids if turns_of(v)]
    if timed:
        v = max(timed, key=lambda v: (turns_of(v), -v["gen"]))
        fights.append({"title": "Longest game", "value": f"{turns_of(v)} turns", "holder": v["bots"][0], "video": v,
                       "detail": f"{SLOTS.get(v.get('slot'), '')} against {short(v['bots'][1])}"})
        quick = [v for v in timed if v.get("result") == "win" and v.get("turns")]
        if quick:
            v = min(quick, key=lambda v: (v["turns"], v["gen"]))
            mine, opp = sides(v)
            fights.append({"title": "Quickest knockout", "value": f"{v['turns']} turns", "holder": v["bots"][0],
                           "video": v, "detail": f"as {mine} vs {opp}, {moment(v).lower()}"})
    backs = [(int(m.group(1)), v) for v in vids for m in [re.search(r"from -(\d+) hp", v.get("why") or "")] if m]
    if backs:
        d, v = max(backs, key=lambda x: (x[0], -x[1]["gen"]))
        fights.append({"title": "Biggest comeback", "value": f"−{d:,} hp", "holder": v["bots"][0], "video": v,
                       "detail": f"down {d:,} of 1,500 hp and won, against {short(v['bots'][1])}"})
    flawless = [v for v in vids if v.get("result") == "win" and re.search(r"\+1500 hp", v.get("why") or "")]
    if flawless:
        v = flawless[-1]
        fights.append({"title": "Flawless wins on camera", "value": str(len(flawless)), "holder": v["bots"][0],
                       "video": v, "detail": f"won without losing a single hp; the latest is {short(v['bots'][0])}'s"})
    gated = [r for r in rels if r["gate"]["pooled"].get("score") is not None]
    if gated:
        r = max(gated, key=lambda r: r["gate"]["pooled"]["score"])
        promos.append({"title": "Most decisive promotion", "value": pct(r["gate"]["pooled"]["score"]),
                       "holder": house(r["generation"]),
                       "detail": f"against g{r['opponent_generation']} over {r['gate']['pooled']['games']} games"})
        r = min(gated, key=lambda r: r["gate"]["pooled"]["score"])
        promos.append({"title": "Closest promotion", "value": f'{100 * r["gate"]["pooled"]["score"]:.1f}%',
                       "holder": house(r["generation"]),
                       "detail": f"against g{r['opponent_generation']} over {r['gate']['pooled']['games']} games"})
        r = max(gated, key=lambda r: (r["gate"]["pooled"].get("games") or 0, r["generation"]))
        promos.append({"title": "Biggest promotion test", "value": f'{r["gate"]["pooled"]["games"]:,} games',
                       "holder": house(r["generation"]), "detail": f"against g{r['opponent_generation']}"})
    jumps = [(point[house(r["generation"])] - point[house(r["opponent_generation"])], r) for r in rels
             if r.get("opponent_generation") is not None and house(r["generation"]) in point
             and house(r["opponent_generation"]) in point]
    if jumps:
        d, r = max(jumps, key=lambda x: x[0])
        promos.append({"title": "Biggest rating jump", "value": signed(d), "holder": house(r["generation"]),
                       "detail": f"over g{r['opponent_generation']}"})
    if rels:
        r = max(rels, key=lambda r: r.get("games_trained") or 0)
        training.append({"title": "Most self-play behind one generation", "value": kcount(r.get("games_trained")),
                         "holder": house(r["generation"]), "detail": "games in its training window"})
    stamps = [(when(r.get("promoted_utc")), r) for r in rels if when(r.get("promoted_utc"))]
    gaps = [((b[0] - a[0]).total_seconds(), a[1], b[1]) for a, b in zip(stamps, stamps[1:])]
    if gaps:
        s, a, b = min(gaps, key=lambda x: x[0])
        training.append({"title": "Fastest promotion", "value": span_text(s), "holder": house(b["generation"]),
                         "detail": f"after g{a['generation']} took the title"})
        s, a, b = max(gaps, key=lambda x: x[0])
        training.append({"title": "Longest reign", "value": span_text(s), "holder": house(a["generation"]),
                         "detail": f"champion until g{b['generation']} beat it"})
    return {"On camera": fights, "Promotions": promos, "Training": training}


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
    cur_era = json.dumps(cur["era"], sort_keys=True) if cur else None
    base_bot = house(rels[0]["generation"]) if rels else None
    champ = board[0] if board else None
    # HustleBot's weights are published two generations behind the current best: the newest two have none public.
    weights_lag = {house(r["generation"]) for r in rels[-WEIGHTS_LAG:]}
    has_play = (docs / "PLAYING.md").exists()
    routes: dict[str, tuple[str, str, str]] = {}  # path -> (title, section, body)
    redirects: dict[str, str] = {"/arena/": "/"}  # path -> the route it moved to

    def in_cur_era(b):
        return not is_house(b) or (b in by_gen and json.dumps(by_gen[b]["era"], sort_keys=True) == cur_era)

    def delta(b):
        r = by_gen.get(b)
        if not r or r.get("opponent_generation") is None:
            return None
        o = house(r["opponent_generation"])
        return point[b] - point[o] if b in point and o in point else None

    chars = character_results(rows, rels)
    char_rt = {c: character_ratings(chars.get(c, []), point) for c in CHARACTERS}

    # Arena (home) ------------------------------------------------------------------------------------------------
    best, why = showcase_match(board, vids)
    hero = ""
    if champ:
        cr = by_gen.get(champ)
        if cr:
            g = cr["gate"]["pooled"]
            facts = [tile(f"{point[champ]:.0f}", "Rating", ci_text(champ, ci, base_bot)),
                     tile(pct(g.get("score")), "Promotion test",
                          f'vs g{cr.get("opponent_generation")} · {g.get("games")} games'),
                     tile(kcount(cr.get("games_trained")), "Self-play games", "in its training window"),
                     tile(esc(mday(cr.get("promoted_utc"))), "Champion since", esc(str(when(cr.get("promoted_utc")).year))
                          if when(cr.get("promoted_utc")) else "")]
            lead = (f'Promoted {esc(day(cr.get("promoted_utc")))} after scoring {pct(g.get("score"))} against '
                    f'{esc(label(house(cr["opponent_generation"])))} over {g.get("games")} games. It learned by '
                    f'playing itself.')
        else:
            facts = [tile(f"{point[champ]:.0f}", "Rating", ci_text(champ, ci, base_bot)),
                     tile(str(played[champ]), "Games"),
                     tile(pct(scored[champ] / played[champ] if played[champ] else None), "Score", "share of points")]
            lead = f'An Arena entry by {esc(owners.get(champ) or "—")}.'
        btns = [f'<a class="btn primary" href="{L(bot_route(champ) + "#in-action")}">{PLAY_ICON}Watch '
                f'{esc(short(champ))} play</a>']
        if has_play and is_house(champ):
            btns.append(f'<a class="btn ghost" href="{L("/play/")}">Play against HustleBot</a>')
        vid = ""
        if best:
            a, b = best["bots"]
            chips = [x for x in (best.get("matchup"), f"{turns_of(best)} turns" if turns_of(best) else "",
                                 clock(best.get("seconds")), f'{(best.get("result") or "").title()} for {short(a)}') if x]
            poster = f' poster="{M(best["poster"])}"' if best.get("poster") else ""
            mine, opp = sides(best)
            vid = (f'<figure class="herovid"><div class="screen"><video data-autoplay muted loop playsinline controls '
                   f'preload="metadata"{poster} src="{M(best["file"])}"></video></div><figcaption>'
                   f'<p class="eyebrow">Showcase match</p><p class="vsline"><span class="me">{esc(short(a))} '
                   f'<small>{esc(mine)}</small></span><span class="vs">vs</span><span class="them">{esc(short(b))} '
                   f'<small>{esc(opp)}</small></span></p><p class="small">{esc(why)}.'
                   f'{" " + esc(short(a)) + " played " + esc(best["new_side"]) + "." if best.get("new_side") else ""}'
                   f'</p><div class="chips">{"".join(f"<span class=chip>{esc(x)}</span>" for x in chips)}</div>'
                   f'</figcaption></figure>')
        hero = (f'<section class="hero{"" if vid else " novid"}"><div class="herotext"><p class="kicker">{CROWN}'
                f'Reigning champion</p><h2 class="champname"><a href="{L(bot_route(champ))}">'
                + (f'HustleBot <span>g{gen_of(champ)}</span>' if is_house(champ) else esc(champ))
                + f'</a></h2><p class="lead">{lead}</p><div class="tiles hero-tiles">{"".join(facts)}</div>'
                f'<div class="btns">{"".join(btns)}</div></div>{vid}</section>')

    # the ladder: one table; "This era" hides the earlier eras' rows (CSS), "All time" shows them
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
        now = era_key in (None, "entries", cur_era)
        if era_key is not None:
            name = ("Before the published releases" if era_key == "before" else "Arena entries" if era_key == "entries"
                    else ("Current era: " if era_key == cur_era else "Earlier era: ") + era_label(json.loads(era_key)))
            trs.append(f'<tr class="erarow{"" if now else " old"}"><td colspan="6">{esc(name)}</td></tr>')
        for k, b in enumerate(members, 1):
            tag = ('<span class="era">house bot, weights not public</span>' if b in weights_lag else
                   '<span class="era">house bot</span>') if is_house(b) else \
                f'<span class="era">by {esc(owners.get(b) or "—")}</span>'
            d = delta(b)
            cls = " ".join(x for x in ("top1" if k == 1 else "", "old" if not now else "", "champ" if b == champ else "") if x)
            cls_attr = f' class="{cls}"' if cls else ""
            trs.append(f'<tr{cls_attr}><td class="rank">{k}</td>'
                       f'<th scope="row"><a class="botname" href="{L(bot_route(b))}">'
                       f'{badge(b, in_cur_era(b), b == champ)}<span class="bn"><span class="full">{esc(label(b))}</span>'
                       f'<span class="shortn">{esc(short(b))}</span>{tag}</span></a></th>'
                       f'<td class="num rating"><b>{point[b]:.0f}</b><span class="ci">{ci_text(b, ci, base_bot)}</span>'
                       f'</td><td class="cicol">{cibar(ci[b][0], ci[b][1], point[b], scale)}</td>'
                       f'<td class="num hide-s delta">{"" if d is None else signed(d)}</td>'
                       f'<td class="num hide-s">{played[b]:,}</td></tr>')
    multi_era = not arena_only and len(groups) > 1
    ladder_tabs = ('<div class="seg" role="group" aria-label="Which bots"><input type="radio" name="lad" id="lad-now" '
                   'checked><label for="lad-now">This era</label><input type="radio" name="lad" id="lad-all">'
                   '<label for="lad-all">All time</label></div>') if multi_era else ""
    seat = (f'<a class="openseat" href="{L("/get-started/")}"><span class="gb ghost" aria-hidden="true">+</span>'
            f'<span><b>Open seat: your bot here</b><span class="small">Every contender so far is a HustleBot generation. '
            f'The Arena opens to outside bots with two divisions, model alone and full bot. Train one now on your own '
            f'PC.</span></span><span class="go">Train a bot →</span></a>')
    base_line = ""
    bl = baselines(rels)
    if bl:
        base_line = '<p class="baselines"><span class="tname">Baselines</span>' + "".join(
            f'<span><b>{esc(name)}</b>: scored {pct(1 - sum(s * n for _, s, n in xs) / sum(n for _, _, n in xs))} '
            f'in {sum(n for _, _, n in xs)} games against '
            + ", ".join(f"g{g}" for g, _, _ in xs) + "</span>" for name, xs in bl.items()) + "</p>"
    pooled = ("The overall rating counts all 25 character pairings equally, each played from both seats; ratings "
              "by character are below.")
    if arena_only:
        board_title = "The ladder"
        note = f"Rated on the Arena's own ranked games, all under the spec below. {pooled}"
    else:
        board_title = "The ladder"
        note = ("Until the Arena's own ranked matches are in, the ladder is HustleBot's history, and provisional: "
                "each generation rated from the promotion test that published it, in the settings of its era. "
                f"{gate_history(rels)} Compare ratings within an era. {pooled}")
    era_start = ERA_STARTS.get(spec_hash(aspec)) if aspec else None
    if era_start:
        note += (f" A new rating era started with spec {spec_hash(aspec)} on {era_start[0]}; earlier provisional "
                 f"results stay under the old spec {era_start[1]}.")
    spec_line = ""
    if aspec:
        arena_games = sum(r.get("games") or 0 for r in rows)
        spec_line = (f'<p class="specline"><span>game {esc(aspec["game"]["version"])} (Steam build '
                     f'{esc(aspec["game"]["steam_build_id"])})</span><span>harness {esc(aspec["harness"]["version"])}'
                     f'</span><span>candidate set {aspec["candidate_set"]}</span><span>spec <code>'
                     f'{spec_hash(aspec)}</code></span><span>{arena_games} Arena games</span></p>')
    ladder = (f'<section class="ladder" id="ladder"><div class="sechead"><div><p class="eyebrow">Leaderboard</p><h2>'
              f'{board_title}</h2></div>{ladder_tabs}</div>'
              f'<div class="tablewrap"><table class="board"><thead><tr><th class="rank">#</th>'
              f'<th>Bot</th><th class="num">Rating</th><th class="cicol"><span class="sr">95% interval</span></th>'
              f'<th class="num hide-s" title="Rating gained over the generation it beat">Gain</th>'
              f'<th class="num hide-s">Games</th></tr></thead><tbody>{"".join(trs)}</tbody></table></div>{seat}'
              f'{base_line}<details class="explain"><summary>How the ratings work</summary><div><p>Bradley–Terry '
              f'ratings on an Elo-like scale, with g{rels[0]["generation"] if rels else "?"} (the first published '
              f'generation) at 0 and a 95% interval from a bootstrap. Gain is the rating a generation added over the one '
              f'it beat. {esc(note)}</p><p>Rated games run on our private simulator of the game, checked against the '
              f'real game; every video on this site is the game replayed in the real game.</p>{spec_line}</div>'
              f'</details></section>')

    # the champion's matchup chart, and ratings by character
    matchup_sec = ""
    if champ in by_gen and by_gen[champ].get("opponent_generation") is not None:
        cr = by_gen[champ]
        hm = heatmap(cr, f"{short(champ)} against g{cr['opponent_generation']} by character pairing")
        if hm:
            how = ("Its promotion test against " + esc(short(house(cr["opponent_generation"]))) +
                   (", with " + esc(short(champ)) + " as Cowboy every game." if cowboy_gate(cr) else
                    ": every character pairing, P1's character down the side, P2's across the top, each played from "
                    "both seats. Hover a cell for its games and interval."))
            matchup_sec = (f'<div class="panelbox"><div class="sechead"><div><p class="eyebrow">Matchups</p><h2>How '
                           f'{esc(short(champ))} fares by character</h2></div></div><p class="small">{how}</p>{hm}</div>')
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
                for k, (b, (th, lo, hi, n, prov)) in enumerate(sorted(rt.items(), key=lambda kv: -kv[1][0])[:8], 1))
            inner = (f'<div class="tablewrap flat"><table class="board compact"><thead><tr><th class="rank">#</th>'
                     f'<th>Bot</th><th class="num">As {c}</th><th class="num">Games</th></tr></thead>'
                     f'<tbody>{body_rows}</tbody></table></div>')
        else:
            inner = f'<p class="small empty">No games as {c} yet. They come with the Arena\'s own ranked matches.</p>'
        panels.append(f'<div class="tp tp-{c}">{inner}</div>')
    by_character = (f'<div class="panelbox bychar"><div class="sechead"><div><p class="eyebrow">Per character</p><h2>'
                    f'Rating by character</h2></div></div><p class="small">How strong each bot is playing that '
                    f'character, against opponents at their overall ratings. Promotion tests count only for Cowboy, and '
                    f'only the Cowboy-focused ones, so those numbers are provisional.</p><div class="tabs">'
                    f'{"".join(tabs)}<div class="tabpanels">{"".join(panels)}</div></div></div>')

    # recent results: promotion tests and the Arena's matches, newest first
    feed = []
    for r in rels:
        g = r["gate"]["pooled"]
        if r.get("opponent_generation") is not None and g.get("score") is not None:
            feed.append((r.get("promoted_utc") or "", "Promotion test", house(r["generation"]),
                         house(r["opponent_generation"]), g["score"], int(g.get("games") or 0)))
    pairs: dict[tuple, list] = {}
    for row in rows:
        if row.get("score") is None or not row.get("games"):
            continue
        a, b, s = row["a"], row["b"], float(row["score"])
        if (b, a) in pairs:
            a, b, s = b, a, 1 - s
        t = pairs.setdefault((a, b), [0.0, 0, ""])
        t[0] += s * row["games"]
        t[1] += int(row["games"])
        t[2] = max(t[2], row.get("utc") or "")
    feed += [(u, "Arena", a, b, w / n, n) for (a, b), (w, n, u) in pairs.items() if n]
    feed = [(u, k, a, b, s, n) if s >= 0.5 else (u, k, b, a, 1 - s, n) for u, k, a, b, s, n in feed]  # winner first
    feed.sort(key=lambda x: x[0], reverse=True)
    feed_html = "".join(
        f'<li><span class="when">{esc(mday(u))}</span><span class="kind{" arena" if k == "Arena" else ""}">{esc(k)}'
        f'</span><span class="pair"><a href="{L(bot_route(a))}" class="{"w" if s >= 0.5 else ""}">{esc(short(a))}</a>'
        f'<span class="score"><b>{round(100 * s)}</b>–<b>{100 - round(100 * s)}</b></span>'
        f'<a href="{L(bot_route(b))}" class="{"w" if s < 0.5 else ""}">{esc(short(b))}</a></span>'
        f'<span class="split" aria-hidden="true"><i style="width:{100 * s:.0f}%"></i></span>'
        f'<span class="n">{n:,} games</span></li>' for u, k, a, b, s, n in feed[:6])
    recent = (f'<div class="panelbox"><div class="sechead"><div><p class="eyebrow">Results</p><h2>Recent matches</h2>'
              f'</div><a class="more" href="{L("/history/")}">Full history →</a></div><ol class="feed">{feed_html}</ol>'
              f'</div>') if feed_html else ""
    champ_vids = [v for v in vids if v["bots"][0] == champ and v.get("group") == "featured"]
    latest = (f'<section><div class="sechead"><div><p class="eyebrow">Replays</p><h2>{esc(short(champ))} in action</h2>'
              f'</div><a class="more" href="{L("/replays/")}">All {len(vids)} replays →</a></div>'
              f'{replay_grid(champ_vids, show_gen=False)}</section>') if champ_vids else ""
    total_trained = sum(r.get("games_trained") or 0 for r in rels)
    total_test = sum(int(r["gate"]["pooled"].get("games") or 0) for r in rels)
    strip = ""
    if rels:
        strip = ('<div class="strip">' + "".join(
            f'<div><b>{v}</b><span>{esc(t)}</span></div>' for v, t in (
                (str(len(rels)), "generations promoted"), (big_count(total_trained), "self-play games trained on"),
                (f"{total_test:,}", "promotion-test games"), (str(len(vids)), "replays in the real game"),
                (str(len(CHARACTERS)), "characters"))) + "</div>")
    body = (f'<header class="pagehead arenahead"><p class="eyebrow">Your Only Move Is HUSTLE · bot arena</p>'
            f'<h1>The Arena</h1><p class="lead">Bots play <i>Your Only Move Is HUSTLE</i> under one fixed rule set, with '
            f'the same thinking budget for every move. Right now every contender is a generation of HustleBot, the '
            f'house bot, each one promoted after a head-to-head test against the champion before it.</p></header>'
            f'{hero}{strip}{ladder}<div class="twocol">{matchup_sec}<div class="stack">{recent}{by_character}</div></div>'
            f'{latest}'
            f'<section class="cards three cta"><a class="card" href="{L("/get-started/")}">{ICONS["bolt"]}'
            f'<b>Train your own</b><span>One PC and your copy of the game: self-play, training and promotion in one '
            f'command.</span></a>'
            + (f'<a class="card" href="{L("/play/")}">{ICONS["mod"]}<b>Play HustleBot</b><span>Take on the house bot in '
               f'the real game with the vs-bot mod.</span></a>' if has_play else "")
            + f'<a class="card" href="{L("/history/")}">{ICONS["chart"]}<b>The climb</b><span>Every generation, its '
            f'rating and the test that promoted it.</span></a></section>')
    routes["/"] = ("Arena", "arena", body)

    # a page per bot: for a house generation, its showcase too ---------------------------------------------------
    gens_sorted = [r["generation"] for r in rels]
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
        own = [v for v in vids if v["bots"][0] == b]
        feat = [v for v in own if v.get("group") == "featured"]
        cast = [v for v in own if v.get("group") == "cast"]
        rest = [v for v in own if v.get("group") not in ("featured", "cast")]
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
        mine = [(c, char_rt[c][b]) for c in CHARACTERS if b in char_rt[c] and char_rt[c][b][3] >= CHAR_TILE_GAMES]
        by_char = ("".join(tile(f"{th:.0f}", f"as {c}", f"{lo:.0f} to {hi:.0f} · {n} games"
                                + (" · provisional" if prov else "")) for c, (th, lo, hi, n, prov) in mine))
        notes = f'<p class="lead">{esc(entry["notes"])}</p>' if entry and entry.get("notes") else ""
        links = ((f' · <a href="{REPO}/releases/tag/gen-{r["generation"]}">release</a> · house bot, weights not public'
                  if b in weights_lag else
                  f' · <a href="{REPO}/releases/tag/gen-{r["generation"]}">release and weights</a>') if r else "")
        tiles = tile(f"{point[b]:.0f}", "Rating", ci_text(b, ci, base_bot), True) + \
            tile(f"#{board.index(b) + 1}", "Rank", f"of {len(board)}")
        if r:
            g = r["gate"]["pooled"]
            tiles += tile(pct(g.get("score")), "Promotion test", f'vs g{r.get("opponent_generation")} · '
                          f'{g.get("games")} games') + tile(kcount(r.get("games_trained")), "Self-play games",
                                                            "in its training window")
        else:
            tiles += tile(f"{played[b]:,}", "Games") + \
                tile(pct(scored[b] / played[b] if played[b] else None), "Score", "share of points")
        g_n = gen_of(b)
        step = ""
        if r and g_n in gens_sorted:
            k = gens_sorted.index(g_n)
            prev_g = gens_sorted[k - 1] if k else None
            next_g = gens_sorted[k + 1] if k + 1 < len(gens_sorted) else None
            step = ('<nav class="genstep" aria-label="Generations">'
                    + (f'<a href="{L(f"/bots/g{prev_g}/")}" rel="prev">← g{prev_g}</a>' if prev_g else "<span></span>")
                    + (f'<a href="{L(f"/bots/g{next_g}/")}" rel="next">g{next_g} →</a>' if next_g else "<span></span>")
                    + "</nav>")
        tags = []
        if b == champ:
            tags.append(f'<span class="tag champ">{CROWN}Champion</span>')
        if r:
            tags.append(f'<span class="tag">{esc(era_short(r["era"]))}</span>')
        hm = heatmap(r, f"{short(b)} against g{r['opponent_generation']} by character pairing") \
            if r and r.get("opponent_generation") is not None else ""
        crumb = (f'<a href="{L("/history/")}">← History</a>' if is_house(b) else f'<a href="{L("/")}">← Arena</a>')
        body = (f'<nav class="crumbs">{crumb}{step}</nav>'
                f'<header class="pagehead bothead">{badge(b, in_cur_era(b), b == champ)}<div><p class="eyebrow">'
                f'{"The house bot" if is_house(b) else "Arena entry"}</p><h1>{esc(label(b))}</h1>'
                f'<div class="tagrow">{"".join(tags)}</div><p class="small">{esc(" · ".join(facts))}{links}</p>{notes}'
                f'</div></header><div class="tiles">{tiles}</div>'
                + (f'<section id="in-action"><div class="sechead"><h2>{esc(short(b))} in action</h2><span class="small">'
                   f'Cowboy mirror games from the test that promoted it: its longest win, biggest win, best comeback '
                   f'and worst loss</span></div>{replay_grid(feat, show_gen=False, feature=True)}</section>'
                   if feat else "")
                + (f'<details class="panel"><summary>Cowboy vs the cast · {len(cast)} games</summary>'
                   f'{replay_grid(cast, show_gen=False)}</details>' if cast else "")
                + (replay_grid(rest, show_gen=False) if rest else "")
                + (f'<section><div class="sechead"><h2>Record by matchup in its promotion test</h2></div>'
                   f'<p class="small">' + (f"It played Cowboy in every game, against {esc(label(house(r['opponent_generation'])))}."
                                           if cowboy_gate(r) else
                                           f"Against {esc(label(house(r['opponent_generation'])))}: P1's character down "
                                           f"the side, P2's across the top; it played each pairing from both seats.")
                   + f'</p>{hm}</section>' if hm else "")
                + (f'<section><h2>Rating by character</h2><div class="tiles">{by_char}</div></section>' if by_char else "")
                + f'<section><h2>Results by opponent</h2><div class="tablewrap"><table><thead><tr><th>Opponent</th>'
                f'<th class="num">Score</th><th class="metercol"><span class="sr">Score bar</span></th>'
                f'<th class="num">Games</th></tr></thead><tbody>{opp_rows}</tbody></table></div></section>'
                + (f'<section><h2>Record by matchup in the Arena</h2><p class="small">Its character first, from its '
                   f'own games in the Arena; each pairing is played from both seats.</p><div class="mus">'
                   + "".join(f'<div class="mu"><span>{esc(k)}</span>{meter(s)}<b>{pct(s)}</b><span class="dim">{n}'
                             f'</span></div>' for k, (s, n) in arena_mu.items()) + '</div></section>' if arena_mu else "")
                + (f'<details class="panel"><summary>As the opponent · {len(seen)} games</summary>'
                   f'{replay_grid(seen)}</details>' if seen else ""))
        routes[bot_route(b)] = (label(b), "history" if is_house(b) else "arena", body)

    # History: the climb ------------------------------------------------------------------------------------------
    house_gens = sorted(gen_of(b) for b in point if is_house(b))
    era_of = lambda g: by_gen[house(g)]["era"] if house(g) in by_gen else None  # noqa: E731
    stats = ""
    if rels and house(rels[-1]["generation"]) in point and house(rels[0]["generation"]) in point:
        first, last = rels[0], rels[-1]
        gain = point[house(last["generation"])] - point[house(first["generation"])]
        d0, d1 = when(first.get("promoted_utc")), when(last.get("promoted_utc"))
        days = (d1 - d0).total_seconds() / 86400 if d0 and d1 else None
        stats = ('<div class="strip">' + "".join(f'<div><b>{v}</b><span>{esc(t)}</span></div>' for v, t in (
            (signed(gain), f"rating, g{first['generation']} to g{last['generation']}"),
            (f"{days:.0f} days" if days is not None else "—", f"{mday(first.get('promoted_utc'))} to {mday(last.get('promoted_utc'))}"),
            (str(len(rels)), "promotions"), (big_count(total_trained), "self-play games trained on"),
            (f"{total_test:,}", "promotion-test games"))) + "</div>")
    tl = []
    prev = None
    for r in rels:
        tl.append((r, what_changed(r, prev)))
        prev = r
    tl_html, last_era = [], None
    for r, changed in reversed(tl):
        e = json.dumps(r["era"], sort_keys=True)
        if e != last_era:
            members = [x["generation"] for x, _ in tl if json.dumps(x["era"], sort_keys=True) == e]
            tl_html.append(f'<li class="tl-era"><span>{"Current era" if e == cur_era else "Era"}</span>'
                           f'<b>{esc(era_label(r["era"]))}</b><span class="dim">g{members[0]}'
                           f'{"–g" + str(members[-1]) if len(members) > 1 else ""}</span></li>')
            last_era = e
        hb = house(r["generation"])
        g = r["gate"]["pooled"]
        pv = next((v["poster"] for v in vids if v["gen"] == r["generation"] and v.get("poster")
                   and v.get("group") == "featured"), None)
        d = delta(hb)
        tl_html.append(
            f'<li class="tl-item{" champ" if hb == champ else ""}"><a class="tl-card" href="{L(bot_route(hb))}">'
            + (f'<img src="{M(pv)}" alt="" loading="lazy">' if pv else '<span class="noposter"></span>')
            + f'<div class="tl-body"><div class="tl-top">{badge(hb, e == cur_era, hb == champ)}<b>HustleBot '
            f'g{r["generation"]}</b>{f"<span class=tag>{CROWN}Champion</span>" if hb == champ else ""}'
            f'<span class="when">{esc(day(r.get("promoted_utc")))}</span></div>'
            f'<p class="tl-res">{"Beat" if (g.get("score") or 0) > 0.5 else "Promoted against"} g{r.get("opponent_generation")} '
            f'<b>{pct(g.get("score"))}</b> over '
            f'{g.get("games")} games{" · " + signed(d) + " rating" if d is not None else ""}'
            f'{" · " + kcount(r.get("games_trained")) + " self-play games" if r.get("games_trained") else ""}</p>'
            f'<p class="changed">{esc("; ".join(changed))}</p></div></a></li>')
    table_rows = []
    for r in reversed(rels):
        n, g = r["generation"], r["gate"]["pooled"]
        hb = house(n)
        d = delta(hb)
        table_rows.append(f'<tr><th scope="row">{bot_link(hb)}</th><td class="hide-s">{esc(day(r.get("promoted_utc")))}</td>'
                          f'<td class="hide-s">{esc(era_label(r["era"]))}</td>'
                          + (f'<td class="num rating"><b>{point[hb]:.0f}</b><span class="ci">{ci_text(hb, ci, base_bot)}'
                             f'</span></td>' if hb in point else '<td class="num">—</td>')
                          + f'<td class="num">{"" if d is None else signed(d)}</td>'
                          f'<td class="num">{pct(g.get("score"))}<span class="ci">vs g{r.get("opponent_generation")} · '
                          f'{g.get("games")}</span></td><td class="num hide-s">{kcount(r.get("games_trained"))}</td></tr>')
    body = (subnav(HISTORY_NAV, "/history/", "History")
            + f'<header class="pagehead"><p class="eyebrow">History</p><h1>The climb</h1><p class="lead">Every '
            f'HustleBot generation learned from its own games and had to pass a head-to-head test against the champion '
            f'before it. This is that ladder, one promotion at a time.</p></header>{stats}'
            f'<section><div class="sechead"><div><h2>Rating by generation</h2></div><span class="small">by generation · 95% band · '
            f'g{rels[0]["generation"] if rels else "?"} = 0 · eras shaded</span></div>'
            f'{dual(rating_chart, house_gens, point, ci, era_of, champ)}'
            f'<p class="small note">A chain of promotion tests tends to overstate gains: a promoted net is the one that '
            f'won its test. Match settings and the move set changed between eras, so ratings compare cleanly only '
            f'within one.</p></section>'
            f'<section><div class="sechead"><div><h2>How each promotion was won</h2></div><span class="small">score '
            f'against the generation before · 95% interval</span></div>'
            f'{dual(margin_chart, rels, wide=(960, 240), small=(420, 220))}'
            f'<details class="panel"><summary>As a table</summary><div class="tablewrap flat"><table><thead><tr>'
            f'<th>Generation</th><th class="hide-s">Promoted</th><th class="hide-s">Era</th><th class="num">Rating</th>'
            f'<th class="num">Gain</th><th class="num">Promotion test</th><th class="num hide-s">Self-play games</th>'
            f'</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div></details></section>'
            f'<section><div class="sechead"><div><p class="eyebrow">Timeline</p><h2>Every generation</h2></div>'
            f'<a class="more" href="{L("/history/records/")}">The record book →</a></div>'
            f'<ol class="tl">{"".join(tl_html)}</ol></section>')
    routes["/history/"] = ("History", "history", body)
    redirects["/lab/generations/"] = "/history/"

    # History: the record book ------------------------------------------------------------------------------------
    book = record_book(rels, vids, point)
    groups_html = []
    for gname, recs in book.items():
        if not recs:
            continue
        cards = []
        for rec in recs:
            v = rec.get("video")
            media_html = ""
            if v:
                img = (f'<img src="{M(v["poster"])}" alt="" loading="lazy">' if v.get("poster")
                       else '<span class="noposter"></span>')
                media_html = (f'<a class="lite" href="{M(v["file"])}" data-poster="{M(v["poster"]) if v.get("poster") else ""}" '
                              f'aria-label="Play: {esc(rec["title"])}">{img}<span class="playbtn">{PLAY_ICON}</span></a>')
            cards.append(f'<article class="rec{" withvid" if v else ""}">{media_html}<div class="recbody">'
                         f'<span class="tname">{esc(rec["title"])}</span><span class="recval">{esc(rec["value"])}</span>'
                         f'<span class="recwho"><a href="{L(bot_route(rec["holder"]))}">{esc(label(rec["holder"]))}</a>'
                         f'</span><span class="small">{esc(rec["detail"])}</span></div></article>')
        groups_html.append(f'<section><div class="sechead"><h2>{esc(gname)}</h2>'
                           + ('<span class="small">among the published replays: each generation\'s featured games and, '
                              'from g10, Cowboy against each other character</span>' if gname == "On camera" else "")
                           + f'</div><div class="recs{" big" if gname == "On camera" else ""}">{"".join(cards)}</div>'
                           f'</section>')
    era_rows = []
    for e in dict.fromkeys(json.dumps(r["era"], sort_keys=True) for r in rels):
        members = [r for r in rels if json.dumps(r["era"], sort_keys=True) == e]
        top = max(members, key=lambda r: point.get(house(r["generation"]), -1e9))
        era_rows.append(f'<tr><th scope="row">{esc(era_label(json.loads(e)))}</th><td>g{members[0]["generation"]}'
                        f'{"–g" + str(members[-1]["generation"]) if len(members) > 1 else ""}</td>'
                        f'<td>{bot_link(house(top["generation"]))}</td>'
                        f'<td class="num">{point.get(house(top["generation"]), 0):.0f}</td>'
                        f'<td class="num hide-s">{kcount(sum(r.get("games_trained") or 0 for r in members))}</td></tr>')
    body = (subnav(HISTORY_NAV, "/history/records/", "History")
            + f'<header class="pagehead"><p class="eyebrow">History</p><h1>The record book</h1><p class="lead">'
            f'The longest games, the quickest knockouts, the narrowest promotions: the extremes of HustleBot\'s '
            f'history so far. Every record updates itself when a new generation is promoted.</p></header>'
            + "".join(groups_html)
            + f'<section><h2>Era by era</h2><div class="tablewrap"><table><thead><tr><th>Era</th><th>Generations</th>'
            f'<th>Best</th><th class="num">Rating</th><th class="num hide-s">Self-play games</th></tr></thead><tbody>'
            f'{"".join(reversed(era_rows))}</tbody></table></div></section>')
    routes["/history/records/"] = ("Record book", "history", body)

    # Replays ----------------------------------------------------------------------------------------------------
    ordered = sorted(vids, key=lambda v: (-v["gen"], v.get("group") != "featured"))
    mus = sorted({f"{sides(v)[0]} {'mirror' if sides(v)[0] == sides(v)[1] else 'vs ' + sides(v)[1]}" for v in vids},
                 key=lambda m: (m.split()[0] != "Cowboy", "mirror" not in m, m))
    opt = lambda val, text: f'<option value="{esc(val)}">{esc(text)}</option>'  # noqa: E731
    filters = (f'<form class="filters" data-filter="rgrid" aria-label="Filter replays">'
               f'<label><span>Generation</span><select name="gen">{opt("", "All")}'
               + "".join(opt(str(r["generation"]), f"g{r['generation']}") for r in reversed(rels)) + '</select></label>'
               f'<label><span>Matchup</span><select name="mu">{opt("", "All")}'
               + "".join(opt(m, m) for m in mus) + '</select></label>'
               f'<fieldset class="chipset"><legend>Moment</legend>'
               + "".join(f'<label><input type="radio" name="slot" value="{k}"{" checked" if not k else ""}><span>{t}</span>'
                         f'</label>' for k, t in [("", "All")] + list(SLOTS.items())) + '</fieldset>'
               f'<fieldset class="chipset"><legend>Result</legend>'
               + "".join(f'<label><input type="radio" name="result" value="{k}"{" checked" if not k else ""}><span>{t}'
                         f'</span></label>' for k, t in (("", "All"), ("win", "Wins"), ("loss", "Losses"))) + '</fieldset>'
               f'<output class="count" aria-live="polite">{len(vids)} replays</output></form>')
    body = (f'<header class="pagehead"><p class="eyebrow">Replays</p><h1>Every published game</h1><p class="lead">'
            f'{len(vids)} games from the promotion tests, each replayed in the real game: for every generation its '
            f'longest win, biggest win, best comeback and worst loss in the Cowboy mirror, and from g10 on, Cowboy '
            f'against the rest of the cast. Each card says which side the new generation played.</p></header>'
            f'{filters}{replay_grid(ordered)}<p class="small empty nores" hidden>No replays match those filters.</p>')
    routes["/replays/"] = ("Replays", "replays", body)
    if cur:
        redirects["/lab/showcase/"] = f"/bots/g{cur['generation']}/"
    for r in rels:
        redirects[f"/lab/showcase/g{r['generation']}/"] = f"/bots/g{r['generation']}/"

    # Lab overview -----------------------------------------------------------------------------------------------
    writeups = sorted((docs / "lab").glob("*.md")) if (docs / "lab").is_dir() else []
    tiles = ""
    if cur:
        g, cb = cur["gate"]["pooled"], house(cur["generation"])
        tiles = (tile(f'g{cur["generation"]}', "Current best HustleBot", f'promoted {day(cur.get("promoted_utc"))}', True)
                 + tile(pct(g["score"]), "Promotion test",
                        f'vs g{cur["opponent_generation"]}, {pct((g.get("ci95") or [None])[0])}–'
                        f'{pct((g.get("ci95") or [None, None])[1])}, {g["games"]} games')
                 + (tile(f"{point[cb]:.0f}", "Rating", f"{ci[cb][0]:.0f} to {ci[cb][1]:.0f}") if cb in point else "")
                 + tile(str(len(rels)), "Generations",
                        f'published, {rels[0]["generation"]} to {rels[-1]["generation"]}'))
    cards = [("/lab/how-it-works/", "How it works", "Search over the game's real moves, scored by a value net.", "gear"),
             ("/lab/training/", "Training", "Self-play, training and promotion, on one PC or many.", "loop"),
             ("/history/", "History", "The rating climb, generation by generation, and the record book.", "chart"),
             ("/replays/", "Replays", "Every published game, replayed in the real game.", "play"),
             ("/lab/mod/", "The game mod", "The in-game engine server the harness talks to.", "mod")]
    cards = [c for c in cards if c[0] not in ("/lab/how-it-works/", "/lab/training/", "/lab/mod/")
             or (docs / next(k for k, v in DOC_ROUTES.items() if v == c[0])).exists()]
    cards += [(f"/lab/writeups/{w.stem.lower()}/", w.stem.replace("-", " ").title(), "Write-up", "doc") for w in writeups]
    card_html = "".join(f'<a class="card" href="{L(p)}">{ICONS[i]}<b>{esc(t)}</b><span>{esc(d)}</span></a>'
                        for p, t, d, i in cards)
    body = (subnav(LAB_NAV, "/lab/", "Lab") + f'<header class="pagehead"><p class="eyebrow">The Lab</p><h1>HustleBot: a '
            f'bot that learns by playing itself</h1><p class="lead">yomibench is an open-source fan project for <i>Your '
            f'Only Move Is HUSTLE</i>, made with the developer Ivy Sly\'s permission and credited below; it isn\'t an '
            f'official product of the game. HustleBot is its bot: each generation trains on its own games and is '
            f'promoted only if it beats the best generation before it. This page updates itself on every promotion.'
            f'</p><p class="small">HustleBot trains in a private simulator, a Rust rewrite of the game\'s engine made '
            f'with Ivy Sly\'s permission and checked against the real game; every video on this site is replayed in '
            f'the real game. Everything you can download runs on the regular game: headless copies of your own Steam '
            f'install.</p></header><div class="tiles">{tiles}</div>'
            f'<section class="cards three">{card_html}</section>{CREDITS}')
    routes["/lab/"] = ("Lab", "lab", body)

    # Get started ------------------------------------------------------------------------------------------------
    steps = [("Download the harness", "You need your own copy of the game on Steam, Python 3.11 or newer and uv. "
              "Clone the repository, then follow the setup guide.", "/get-started/setup/", "Setup guide",
              f"git clone {REPO}\ncd yomibench\nuv sync --extra train"),
             ("Train your own on one PC", "One command runs the whole loop on one machine: self-play in headless "
              "copies of your game, training, and a promotion test. Start from scratch or from a published generation "
              "(weights are published two generations behind the current best).",
              "/lab/training/", "Training guide", "uv run python tools/train_loop.py \\\n  --out ~/.yomibench/train")]
    body = ('<header class="pagehead"><p class="eyebrow">Get started</p><h1>Train a bot</h1><p class="lead">'
            'Two steps from your copy of the game to a bot of your own.</p></header>'
            + agent_prompt("start", "Prompt to get started",
                           "A prompt that walks you through setup and training", PROMPT_START)
            + '<ol class="steps">'
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
                "/lab/mod/": "lab"}
    for name, path in DOC_ROUTES.items():
        src = docs / name
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        title = (re.search(r"^#\s+(.+)$", text, re.M) or [None, name[:-3].title()])[1]
        top = ""
        if path == "/get-started/setup/":
            top = f'<nav class="crumbs"><a href="{L("/get-started/")}">← Get started</a></nav>'
        sub = subnav(LAB_NAV, path, "Lab") if sections[path] == "lab" else ""
        copy_key = {"/play/": "play"}.get(path, "")  # the pages people copy commands from
        routes[path] = (title, sections[path], f'{sub}{top}<article class="doc">{doc_html(text, copy_key)}</article>')
    # Submissions are paused while the Arena is redesigned: /submit/ is only this note (docs/SUBMIT.md stays in the
    # repository, marked paused, and isn't built), so old links land somewhere; nothing on the site links here.
    routes["/submit/"] = ("Submissions paused", "arena",
                          '<header class="pagehead"><p class="eyebrow">Submit</p><h1>Submissions are paused</h1>'
                          '</header><div class="callout"><div><b>The Arena is being redesigned.</b><span>Two divisions '
                          'are coming: Model alone (submit a model file, rated with no search) and Full bot (your own '
                          'bot and search, run in a sealed container with the same time per move for everyone). '
                          'Submissions are paused until then.</span></div></div>')
    for w in writeups:
        routes[f"/lab/writeups/{w.stem.lower()}/"] = (
            w.stem.replace("-", " ").title(), "lab",
            subnav(LAB_NAV, "", "Lab") + f'<article class="doc">{doc_html(w.read_text(encoding="utf-8"))}</article>')
    notfound = ('<header class="pagehead"><p class="eyebrow">404</p><h1>Page not found</h1><p class="lead">That page '
                f'isn\'t here. Try the <a href="{L("/")}">Arena</a>, <a href="{L("/history/")}">History</a> or '
                f'<a href="{L("/replays/")}">Replays</a>.</p></header>')
    redirects = {p: t for p, t in redirects.items() if p not in routes}

    targets = set(re.findall(r"@@L:([^@#]*)", "".join(b for _, _, b in routes.values()) + notfound + nav_html("")
                             + FOOTER))
    targets |= set(redirects.values())
    missing = sorted(targets - set(routes))
    if missing:  # a link to a page that isn't built: usually a doc missing from --docs
        raise SystemExit(f"links to routes that weren't built: {missing}")
    if out is not None:
        write_pages(out, routes, notfound, site_url, redirects)
    if spa is not None:
        write_spa(spa, routes, notfound, spa_title, spa_head, redirects)
    print(f"{len(rels)} generations, {len(entered)} entries, {len(board)} rated bots, {len(routes)} routes, "
          f"{len(redirects)} redirects, {sum(len(r['videos']) for r in rels)} videos -> {out or spa}")
    return routes


# --- output ----------------------------------------------------------------------------------------------------
def nav_html(section: str) -> str:
    return "".join(f'<a href="{L(p)}" data-section="{s}"{" class=on aria-current=page" if s == section else ""}>'
                   f'{esc(t)}</a>' for p, t, s in NAV)


GITHUB = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0a8 8 0 0 0-2.5 15.6c.4 0 '
          '.5-.2.5-.4v-1.5c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.3 1.9.9 2.3.7.1-.5'
          '.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-4 0-.9.3-1.6.8-2.1-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.5 7.5 0 0 1 4 0c1.5-1 '
          '2.2-.8 2.2-.8.4 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.1 0 3.1-1.9 3.7-3.6 3.9.3.3.6.8.6 1.5v2.2c0 .2.1.5.6.4A8 8 0 0 0 '
          '8 0z"/></svg>')
THEME_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" '
              'fill="currentColor" stroke="none"/></svg>')
HEADER = ('<a class="skip" href="#main">Skip to content</a><header class="top"><div class="bar"><a class="brand" '
          'href="{home}"><span class="mark" aria-hidden="true"></span><span class="brandtext">yomibench</span></a>'
          '<nav class="mainnav" aria-label="Site">{nav}</nav><div class="actions">'
          '<a class="btn primary sm cta{start_on}" href="{start}" data-section="start">Train a bot</a>'
          f'<button type="button" class="themebtn" aria-label="Switch light or dark theme" title="Light or dark">'
          f'{THEME_ICON}</button><a class="gh" href="{REPO}" aria-label="yomibench on GitHub">{GITHUB}</a></div></div>'
          '</header>')
FOOTER = ('<footer class="foot"><div class="footin"><div class="footgrid"><div class="footabout"><a class="brand" '
          f'href="{L("/")}"><span class="mark" aria-hidden="true"></span><span>yomibench</span></a>'
          '<p><b>yomibench</b> is open source (MIT), '
          f'<a href="{REPO}">on GitHub</a>: a fan project, made with the developer\'s permission, not an official product. The videos show bot matches played in <i>Your Only Move Is HUSTLE</i>, Ivy '
          'Sly\'s game; no game files are distributed here, and the harness runs on your own copy from Steam.</p></div>'
          f'<nav aria-label="Explore"><b>Explore</b><a href="{L("/")}">Arena</a><a href="{L("/history/")}">History</a>'
          f'<a href="{L("/history/records/")}">Record book</a><a href="{L("/replays/")}">Replays</a></nav>'
          f'<nav aria-label="Build"><b>Build</b><a href="{L("/get-started/")}">Get started</a><a href="{L("/lab/")}">'
          f'The Lab</a><a href="{REPO}/releases">Releases</a><a href="{REPO}">Source</a></nav></div>'
          f'<p class="dim">{FOOT_CREDITS}</p></div></footer>')
# Applies a theme the viewer picked before (the toggle stores it) before the first paint.
THEME_EARLY = ('<script>try{var t=localStorage.getItem("yomi-theme");if(t==="light"||t==="dark")'
               'document.documentElement.dataset.theme=t}catch(e){}</script>')
# The page's behaviour, all delegated so it also works on the one-file preview's routes:
# - the hero's video starts only when the viewer hasn't asked for reduced motion; otherwise it waits on its poster;
# - a replay card's poster link becomes an inline player on click (without the script, it opens the video);
# - chart and heatmap tooltips: one tooltip for every [data-tip] mark, on hover and on keyboard focus;
# - the Replays filters; the theme toggle.
SCRIPT = r"""<script>
(function () {
  function play() {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    document.querySelectorAll("video[data-autoplay]").forEach(function (v) { v.muted = true; v.play().catch(function () {}); });
  }
  window.yomiInit = function () { play(); filter(); };
  document.addEventListener("click", function (e) {
    var a = e.target.closest("a.lite");
    if (a && !e.metaKey && !e.ctrlKey && !e.shiftKey) {
      e.preventDefault();
      var v = document.createElement("video");
      v.controls = true; v.autoplay = true; v.playsInline = true; v.preload = "auto"; v.src = a.getAttribute("href");
      if (a.dataset.poster) v.poster = a.dataset.poster;
      v.className = "liveplayer";
      a.replaceWith(v); v.focus();
      document.querySelectorAll("video.liveplayer").forEach(function (o) { if (o !== v) o.pause(); });
      return;
    }
    if (e.target.closest(".themebtn")) {
      var root = document.documentElement, dark = root.dataset.theme ? root.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("yomi-theme", root.dataset.theme); } catch (_) {}
    }
  });
  var tip = document.createElement("div");
  tip.className = "tip"; tip.setAttribute("role", "tooltip"); tip.hidden = true;
  document.addEventListener("DOMContentLoaded", function () { document.body.appendChild(tip); });
  function show(el, x, y) {
    var lines = (el.getAttribute("data-tip") || "").split("\n");
    tip.replaceChildren();
    lines.forEach(function (t, i) { var d = document.createElement(i ? "span" : "b"); d.textContent = t; tip.appendChild(d); });
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(Math.max(8, x - w / 2), innerWidth - w - 8), top = y - h - 14;
    if (top < 8) top = y + 18;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  }
  document.addEventListener("pointermove", function (e) {
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (el) show(el, e.clientX, e.clientY); else tip.hidden = true;
  });
  document.addEventListener("focusin", function (e) {
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) { tip.hidden = true; return; }
    var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top);
  });
  document.addEventListener("focusout", function () { tip.hidden = true; });
  document.addEventListener("scroll", function () { tip.hidden = true; }, { passive: true });
  function filter() {
    var f = document.querySelector("form.filters");
    if (!f) return;
    var q = new FormData(f), n = 0;
    document.querySelectorAll(".rgrid .rcard").forEach(function (c) {
      var ok = (!q.get("gen") || c.dataset.gen === q.get("gen")) && (!q.get("mu") || c.dataset.mu === q.get("mu"))
        && (!q.get("slot") || c.dataset.slot === q.get("slot")) && (!q.get("result") || c.dataset.result === q.get("result"));
      c.hidden = !ok; if (ok) n++;
    });
    var o = f.querySelector(".count"); if (o) o.textContent = n + (n === 1 ? " replay" : " replays");
    var none = document.querySelector(".nores"); if (none) none.hidden = n > 0;
  }
  document.addEventListener("change", function (e) { if (e.target.closest("form.filters")) filter(); });
  document.addEventListener("submit", function (e) { if (e.target.closest("form.filters")) e.preventDefault(); });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", window.yomiInit);
  else window.yomiInit();
})();
</script>"""
# Copy buttons: button[data-copy] names the element whose text it copies, unless data-copy-text gives the exact text
# (a command box's one-line form). Delegated, so it also works on the one-file preview's routes; where the clipboard
# API isn't available (a page opened from a file), it copies through a hidden text box.
COPY = ('<script>document.addEventListener("click",function(e){var b=e.target.closest("button[data-copy]");if(!b)return;'
        'var t=document.getElementById(b.dataset.copy);if(!t)return;var x=b.dataset.copyText||t.textContent;'
        'function done(ok){var o=b.dataset.label||b.getAttribute("aria-label"),m=ok?"Copied":"Copy failed";'
        'b.dataset.label=o;b.classList.remove("done","fail");b.classList.add(ok?"done":"fail");b.dataset.tipCss=m;'
        'b.setAttribute("aria-label",m);clearTimeout(b._t);b._t=setTimeout(function(){b.classList.remove("done","fail");'
        'b.dataset.tipCss=o;b.setAttribute("aria-label",o)},1600)}function pick(){var a=document.createElement("textarea");'
        'a.value=x;a.setAttribute("readonly","");a.style.position="fixed";a.style.opacity="0";document.body.appendChild(a);'
        'a.select();var ok=false;try{ok=document.execCommand("copy")}catch(_){}a.remove();done(ok)}'
        'if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(x).then(function(){done(true)},'
        'pick)}else{pick()}})</script>')


def header_html(section: str) -> str:
    return HEADER.format(home=L("/"), nav=nav_html(section), start=L("/get-started/"),
                         start_on=" on" if section == "start" else "")


def write_pages(out: Path, routes, notfound: str, site_url: str, redirects: dict[str, str] | None = None) -> None:
    shutil.copyfile(HERE / "style.css", out / "style.css")
    base = site_url.rstrip("/")

    def doc(path, title, section, body, canonical=True, base_href="", moved_to=None):
        css = rel_to(path, "/style.css")
        canon = moved_to if moved_to is not None else path
        head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                f'content="width=device-width,initial-scale=1,viewport-fit=cover">{base_href}'
                + (f'<meta http-equiv="refresh" content="0; url={L(moved_to)}">' if moved_to is not None else "")
                + f'<title>{esc(title)} · yomibench</title><meta name="description" content="A bot arena for Your Only '
                f'Move Is HUSTLE: bots that learn by playing themselves, rated head to head.">'
                f'<meta name="color-scheme" content="light dark">'
                + (f'<link rel="canonical" href="{esc(base + canon)}">' if base and canonical else "")
                + f'<link rel="icon" href="{ICON}"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                f'<link rel="stylesheet" href="{FONTS}"><link rel="stylesheet" href="{css}">{THEME_EARLY}</head><body>')
        return (head + header_html(section) + f'<main id="main" class="wrap">{body}</main>' + FOOTER + SCRIPT + COPY
                + "</body></html>")

    for path, (title, section, body) in routes.items():
        d = out / path.strip("/")
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(resolve(doc(path, title, section, body), path, False), encoding="utf-8")
    for path, target in (redirects or {}).items():
        d = out / path.strip("/")
        d.mkdir(parents=True, exist_ok=True)
        title = routes[target][0]
        body = (f'<header class="pagehead"><p class="eyebrow">Moved</p><h1>{esc(title)}</h1><p class="lead">This page '
                f'moved to <a href="{L(target)}">{esc(title)}</a>.</p></header>')
        (d / "index.html").write_text(resolve(doc(path, title, routes[target][1], body, moved_to=target), path, False),
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
    if (t.dataset.redirect) { location.replace("#" + t.dataset.redirect); return; }
    view.replaceChildren(t.content.cloneNode(true));
    if (window.yomiInit) { yomiInit(); }
    document.title = t.dataset.title + " · yomibench";
    document.querySelectorAll(".mainnav a, .actions .cta").forEach(function (a) {
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


def write_spa(spa: Path, routes, notfound: str, title: str = "yomibench", head: str = "",
              redirects: dict[str, str] | None = None) -> None:
    """One file for a preview that can't serve folders: every route in a template, shown by the URL's hash. `title`
    and `head` (extra head markup) are the preview host's."""
    spa.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "style.css", spa.parent / "style.css")
    tpl = "".join(f'<template data-route="{hash_of(p)}" data-title="{esc(t)}" data-section="{s}">{b}</template>'
                  for p, (t, s, b) in routes.items())
    tpl += "".join(f'<template data-route="{hash_of(p)}" data-redirect="{hash_of(t)}" data-title="" data-section="">'
                   f'</template>' for p, t in (redirects or {}).items())
    tpl += f'<template data-route="not-found" data-title="Page not found" data-section="">{notfound}</template>'
    page = (f'<title>{esc(title)}</title>\n{head}\n<meta name="color-scheme" content="light dark">'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="{FONTS}">'
            f'<link rel="stylesheet" href="style.css">{THEME_EARLY}\n'
            + header_html("arena") + '<main id="main" class="wrap"></main>' + FOOTER
            + tpl + SCRIPT + COPY + SPA_SCRIPT)
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

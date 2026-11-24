"""site/build.py: every route is built at its own pretty URL with working relative links (moved pages redirect), the
one-file preview holds every route under its hash, the ratings put the stronger side ahead with each rating inside its
interval, the hero is the longest game between the top two, and History, the record book and Replays are built from
the releases."""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
sb = importlib.util.module_from_spec(spec)
sys.modules["site_build"] = sb
spec.loader.exec_module(sb)


def vid(n, prev, name, turns=None):
    return {"file": f"gen-{n}-{name}.mp4", "slot": "longest win", "why": "Longest win", "matchup": "Cowboy vs Cowboy",
            "result": "win", "new_side": "left, P1", "opponent_generation": prev, "group": "featured", "turns": turns}


def rel(n, prev, score, games, era=("ranked", 6), videos=None):
    return {"generation": n, "promoted_utc": f"2026-01-0{n}T00:00:00Z", "opponent_generation": prev,
            "era": {"settings": era[0], "candidate_set": era[1]}, "games_trained": 1000 * n,
            "gate": {"pooled": {"score": score, "ci95": [score - 0.05, score + 0.05], "games": games},
                     "matchups": {"Cowboy vs Cowboy": {"score": score, "ci95": [0, 1], "games": games}}},
            "anchors": {}, "videos": videos if videos is not None else [vid(n, prev, "a")]}


DOCS = {"SETUP.md": "# Setup\nSee [training](TRAINING.md).\n",
        **{f"{n}.md": f"# {n.title()}\n" for n in ("TRAINING", "METHODOLOGY", "PLAYING", "MOD", "SUBMIT", "MATCHES")}}


def site(tmp_path, rels, arena_rows=None, bots=None, docs=DOCS, spa=False):
    src = tmp_path / "releases"
    src.mkdir()
    for r in rels:
        (src / f"gen-{r['generation']}.json").write_text(json.dumps(r), encoding="utf-8")
        for v in r["videos"]:
            (src / v["file"]).write_bytes(b"\0")
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in docs.items():
        (d / name).write_text(text, encoding="utf-8")
    arena = tmp_path / "arena"
    (arena / "results").mkdir(parents=True, exist_ok=True)
    (arena / "bots.json").write_text(json.dumps(bots or []), encoding="utf-8")
    if arena_rows:
        (arena / "results" / "ranked.jsonl").write_text("\n".join(json.dumps(r) for r in arena_rows), encoding="utf-8")
    out = tmp_path / "_site"
    sb.build(src, d, out, arena, "https://example.org/yomibench", tmp_path / "spa" / "index.html" if spa else None)
    return out


def read(out, route):
    return (out / route.strip("/") / "index.html").read_text(encoding="utf-8")


def test_every_route_has_its_own_url_and_links_resolve(tmp_path):
    rels = [rel(2, 1, 0.7, 200, ("standard", 4)), rel(3, 2, 0.6, 200), rel(4, 3, 0.55, 240)]
    out = site(tmp_path, rels)
    moved = {"/lab/showcase/": "/bots/g4/", "/lab/showcase/g2/": "/bots/g2/", "/lab/generations/": "/history/",
             "/arena/": "/"}
    for route in ("/", "/bots/g4/", "/bots/g1/", "/history/", "/history/records/", "/replays/", "/lab/",
                  "/lab/training/", "/get-started/", "/get-started/setup/", "/submit/", *moved):
        page = read(out, route)
        canonical = moved.get(route, route)
        assert f'<link rel="canonical" href="https://example.org/yomibench{canonical}">' in page, route
        for href in re.findall(r'(?:href|src)="([^"#:]+)"', page):  # every relative link points at a real file
            target = (out / route.strip("/") / href).resolve()
            assert target.exists() or (target / "index.html").exists(), (route, href)
    assert (out / "404.html").exists() and '<base href="/yomibench/">' in (out / "404.html").read_text(encoding="utf-8")
    assert 'url=../' in (out / "arena" / "index.html").read_text(encoding="utf-8")


def test_the_arena_lab_and_bot_pages(tmp_path):
    rels = [rel(2, 1, 0.7, 200, ("standard", 4)), rel(3, 2, 0.6, 200), rel(4, 3, 0.55, 240)]
    out = site(tmp_path, rels)
    home = read(out, "/")
    assert "The Arena" in home and 'class="board"' in home and 'aria-current=page' in home
    assert home.index("bots/g4/") < home.index("bots/g1/")  # strongest first
    assert "Earlier era: standard settings · set 4" in home
    history = read(out, "/history/")
    assert "rating anchor" in history and "The climb" in history and "Candidate set 6" in history
    bot = read(out, "/bots/g4/")
    assert "Results by opponent" in bot and "Record by matchup" in bot and "gen-4-a.mp4" in bot
    assert "As the opponent" in read(out, "/bots/g3/")
    assert sb.CREDITS in read(out, "/lab/")
    assert "g4 in action" in bot and "url=../../bots/g4/" in read(out, "/lab/showcase/")
    assert "train_loop.py" in read(out, "/get-started/") and "lab/training/" in read(out, "/get-started/setup/")
    assert all(sb.FOOT_CREDITS in p.read_text(encoding="utf-8") for p in out.rglob("*.html") if p.parent.name != "arena")
    point, ci = sb.ratings(rels)
    assert point["gen-2"] == 0 and point["gen-2"] < point["gen-3"] < point["gen-4"]
    assert ci["gen-4"][0] < point["gen-4"] < ci["gen-4"][1]


def test_the_one_file_preview_holds_every_route_under_its_hash(tmp_path):
    out = site(tmp_path, [rel(2, 1, 0.7, 200), rel(3, 2, 0.6, 200)], spa=True)
    page = (tmp_path / "spa" / "index.html").read_text(encoding="utf-8")
    names = set(re.findall(r'<template data-route="([^"]+)"', page))
    assert {"arena", "history", "history-records", "replays", "bots-g3", "lab", "lab-showcase", "lab-showcase-g2",
            "get-started", "submit", "not-found"} <= names
    assert 'data-route="lab-showcase-g2" data-redirect="bots-g2"' in page
    assert 'href="#bots-g2"' in page and "@@" not in page and 'href="media/gen-3-a.mp4"' in page
    assert "hashchange" in page and (tmp_path / "spa" / "style.css").exists() and out.exists()


def test_the_hero_is_the_longest_game_between_the_top_two(tmp_path):
    rels = [rel(2, 1, 0.7, 200), rel(3, 2, 0.8, 200, videos=[vid(3, 2, "short", 120), vid(3, 2, "long", 300)])]
    home = read(site(tmp_path, rels), "/")
    hero = home[home.index('class="hero"'):home.index("</section>")]
    assert "gen-3-long.mp4" in hero and "300 turns" in hero and "two top-rated bots" in hero


def test_arena_entries_join_the_leaderboard_with_their_own_page(tmp_path):
    rels = [rel(2, 1, 0.6, 200), rel(3, 2, 0.6, 200)]
    rows = [{"a": "gen-3", "b": "someone-v1", "score": 0.2, "games": 100, "utc": "2026-02-01T00:00:00Z"}]
    bots = [{"name": "someone-v1", "owner": "Someone", "digest": "d" * 64, "settings": {}, "entered_utc": "2026-02-01"}]
    out = site(tmp_path, rels, rows, bots)
    home = read(out, "/")
    assert home.index("bots/someone-v1/") < home.index("bots/g3/") and "Someone" in home and "At launch" not in home
    page = read(out, "/bots/someone-v1/")
    assert "Arena entry" in page and "by Someone" in page and "bots/g3/" in page


def test_mp4_length_is_read_from_the_header(tmp_path):
    box = b"\0\0\0\x6cmvhd" + b"\0" * 4 + b"\0" * 8 + (1000).to_bytes(4, "big") + (95_000).to_bytes(4, "big")
    f = tmp_path / "x.mp4"
    f.write_bytes(b"\0\0\0\x08ftyp" + box + b"\0" * 80)
    assert sb.mp4_seconds(f) == 95.0 and sb.clock(95.0) == "1:35"


def test_ratings_by_character_use_arena_games_and_only_cowboy_gates(tmp_path):
    cow = {"Cowboy vs Cowboy": {"score": 0.6, "games": 120}, "Cowboy vs Ninja": {"score": 0.55, "games": 32}}
    r3 = rel(3, 2, 0.6, 152)
    r3["gate"]["matchups"] = cow
    r2 = rel(2, 1, 0.6, 200)
    r2["gate"]["matchups"] = {"Ninja vs Cowboy": {"score": 0.6, "games": 100}, "Cowboy vs Ninja": {"score": 0.6, "games": 100}}
    rels = [r2, r3]
    detail = [["Ninja", "Robot", "p1", 1.0, 50], ["Robot", "Ninja", "p2", 0.0, 60]]
    rows = [{"a": "gen-3", "b": "gen-2", "score": 0.5, "games": 2, "games_detail": detail, "spec": "x",
             "matchups": {"Ninja vs Robot": {"score": 1.0, "games": 1}, "Robot vs Ninja": {"score": 0.0, "games": 1}}}]
    res = sb.character_results(rows, rels)
    assert [x[4] for x in res["Cowboy"]] == ["gate", "gate"]  # Cowboy-focused gates count for Cowboy only
    assert res["Ninja"] == [("gen-3", "gen-2", 1.0, 1, "arena"), ("gen-2", "gen-3", 1.0, 1, "arena")]
    assert sb.character_results([], [r2])["Cowboy"] == []  # a seat-swapped gate says nothing about one character
    point = {"gen-2": 0.0, "gen-3": 100.0}
    rt = sb.character_ratings(res["Cowboy"], point)
    assert rt["gen-3"][4] is True and rt["gen-3"][1] < rt["gen-3"][0] < rt["gen-3"][2]  # provisional, with interval
    home = read(site(tmp_path, rels), "/")
    assert "Rating by character" in home and 'id="ct-Robot"' in home and "provisional" in home
    assert "all 25 character pairings equally" in home


def test_the_newest_two_generations_have_no_public_weights(tmp_path):
    rels = [rel(2, 1, 0.7, 200), rel(3, 2, 0.6, 200), rel(4, 3, 0.55, 240)]
    out = site(tmp_path, rels)
    assert "house bot, weights not public" in read(out, "/bots/g4/") and "house bot, weights not public" in read(out, "/bots/g3/")
    assert "release and weights" in read(out, "/bots/g2/") and "weights not public" not in read(out, "/bots/g2/")
    assert read(out, "/").count("house bot, weights not public") == 2


def test_a_new_spec_says_its_rating_era_started(tmp_path):
    spec = json.loads((ROOT / "arena" / "spec.json").read_text(encoding="utf-8"))
    assert sb.spec_hash(spec) in sb.ERA_STARTS  # repinning arena/spec.json means saying when its era started
    (tmp_path / "arena").mkdir()
    (tmp_path / "arena" / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    home = read(site(tmp_path, [rel(2, 1, 0.7, 200), rel(3, 2, 0.6, 200)]), "/")
    date, old = sb.ERA_STARTS[sb.spec_hash(spec)]
    assert f"A new rating era started with spec {sb.spec_hash(spec)} on {date}" in home and old in home


def test_submissions_are_paused(tmp_path):
    """While the Arena is redesigned, /submit/ is only the paused note: no page links to it, the nav has no Submit
    item, and nothing points at the submission issue form."""
    out = site(tmp_path, [rel(2, 1, 0.7, 200), rel(3, 2, 0.6, 200)])
    note = read(out, "/submit/")
    assert "Submissions are paused until then." in note and "Full bot" in note and "Model alone" in note
    for page in out.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert "issues/new" not in text and "submit-bot" not in text, page
        assert not re.search(r'<a [^>]*href="[^"]*submit/"', text), page  # its own canonical link isn't a link to it


def test_every_rating_sits_inside_its_interval():
    """A long chain of promotion tests converges slowly: the point fit and every bootstrap refit run to convergence,
    so no rating lands outside its own 95% interval (the old fixed-iteration fit put the newest generations above it)."""
    rels = [rel(n, n - 1, 0.6, 400) for n in range(2, 18)]
    point, ci = sb.ratings(rels)
    assert all(ci[b][0] <= point[b] <= ci[b][1] for b in point)
    assert point["gen-17"] > point["gen-10"] > point["gen-2"] == 0


def test_history_records_and_replays(tmp_path):
    v = [vid(3, 2, "long", 300), vid(3, 2, "quick", 40)]
    v[1].update(slot="biggest win", why="Cowboy mirror, biggest win: +1500 hp")
    v.append({**vid(3, 2, "back", 150), "slot": "comeback", "why": "Cowboy mirror, best comeback: from -1230 hp"})
    v.append({**vid(3, 2, "cast", 80), "group": "cast", "matchup": "Wizard vs Cowboy", "new_side": "right, P2"})
    rels = [rel(2, 1, 0.7, 200), rel(3, 2, 0.55, 800, videos=v)]
    out = site(tmp_path, rels)
    records = read(out, "/history/records/")
    for text in ("Longest game", "300 turns", "Quickest knockout", "40 turns", "Biggest comeback", "−1,230 hp",
                 "Closest promotion", "55.0%", "Most decisive promotion", "70%", "Biggest promotion test", "800 games"):
        assert text in records, text
    replays = read(out, "/replays/")
    assert replays.count('class="rcard') == 5 and 'data-mu="Cowboy vs Wizard"' in replays  # its own character first
    assert 'data-slot="comeback"' in replays and 'name="gen"' in replays and "From −1,230 hp" in replays
    history = read(out, "/history/")
    assert history.count('class="tl-item') == 2 and 'class="chart"' in history and "bots/g3/" in history
    bot = read(out, "/bots/g3/")
    assert 'class="hm"' in bot and "Cowboy vs the cast" in bot and "played P2, right" in bot

# AGENTS.md

A guide for coding agents (and people) working in this repository. Read it before you change anything. The
human-facing docs are [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/](docs/); this file
points at them rather than repeating them.

## What this is

yomibench is a harness for search-plus-learning bots for the game *Your Only Move Is HUSTLE*, an Arena that rates
bots people submit, and **HustleBot**, the project's own bot. The site, https://stixels.github.io/yomibench/, is
built from this repository and its releases.

- The harness runs the **player's own Steam install** of the game as an engine (headless copies, made on their PC).
  Nothing of the game is in this repository, and nothing of it may ever be added (see Conventions).
- A built-in **toy engine** (`--engine toy`) runs everything without the game. Use it for quick checks.
- HustleBot's nets were trained with a fast simulator that is private and not part of this repository. Code paths
  and tests that need it skip; you never need it.

## Setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Run each command on its own line (Windows PowerShell 5.1
can't join commands with `&&`):

```
git clone https://github.com/Stixels/yomibench
cd yomibench
uv sync --extra train --extra test
```

Extras: `net` (numpy) is enough to pack and check a submission; `train` (numpy and torch) is needed to train and for
the full test suite; `test` adds pytest. In a clone synced with only some extras, `uv run --extra net ...` adds numpy
for that command. `uv.lock` is committed: when you change dependencies in `pyproject.toml`, run `uv lock` and
commit the updated lock with the change.

The native solver (`native/yomisolve`, needs a Rust toolchain) is optional and only a speedup; see its README.

## Tests

```
uv run pytest -q
```

Run the whole suite before you open a PR. While you iterate, run the files you touched, for example:

```
uv run pytest -q tests/test_site_build.py tests/test_validate_submission.py
```

Some tests skip, by design, when what they need is absent: the private simulator (always absent here), the game
(a Steam install of *Your Only Move Is HUSTLE*), torch (without the `train` extra), the native solver, ffmpeg, or
Windows. A skip is not a failure. Without the `train` extra, test files that import numpy fail to collect, so sync
`--extra train --extra test` before you judge a run.

## Layout

| path | what |
| --- | --- |
| `src/yomibench/` | the harness: engine protocol (`engine.py`), the game as an engine (`game.py`, `pool_engine.py`), search and bots (`search.py`, `bots/`), self-play, training, gates, ratings, the `yomibench` CLI (`cli.py`) |
| `tools/` | scripts: `train_loop.py` (one-PC training), `pool_eval.py` (gates), `validate_submission.py` (the Arena's submission check), diagnostics |
| `tests/` | pytest suite |
| `mod/YomiBench/` | the in-game engine server (GDScript); `candidate_rules_v*.json` are hashed into records and the Arena spec |
| `mod/YomiBenchVsBot/` | the "Multiplayer (vs bot)" mod for playing HustleBot |
| `native/yomisolve/` | optional Rust solver, bit-identical to the Python one |
| `docker/` | the game's Linux build in Docker (mount your own Steam library) |
| `site/` | `build.py` and `style.css`: the static site |
| `docs/` | the guides; most are also site pages (Setup, Training, Submit, Play, ...) |
| `arena/` | Arena data: spec, house bots, entered bots, rated results (see `arena/README.md`) |
| `.github/` | the Pages workflow, the submission check workflow and the submission issue form |

## Running things

The harness, on the toy engine (`--no-ledger` keeps the run from appending a row to `metrics/ledger.jsonl` in your
clone):

```
uv run yomibench arena --engine toy --bots random,greedy,matrix --pairs 4 --no-ledger
uv run yomibench --help
```

With the game installed through Steam, use `--engine game-pool:auto` (docs/SETUP.md).

Training on one PC (docs/TRAINING.md). It runs headless copies of the game until you stop it; rerunning the same
command resumes. The second command is a check on the toy engine: one cycle of self-play, training and gate in a
few minutes (with the gate's default of all five characters it takes much longer):

```
uv run python tools/train_loop.py --out ~/.yomibench/train
uv run python tools/train_loop.py --out ~/.yomibench/toy-train --engine toy --characters Cowboy,Ninja --cycles 1 --games-per-cycle 20 --pairs 2
```

`OUT/best.json` names the promoted model (its `model` field); on the toy check the gate usually ends "too close to
call" and nothing is promoted. Keep run output (records, models, caches) outside
the repository; never commit it.

The site, the way the Pages workflow builds it (it needs the releases' `gen-N.json` files; videos are optional).
In bash:

```
mkdir -p releases
for tag in $(gh release list --limit 200 --json tagName -q '.[].tagName'); do gh release download "$tag" --dir releases --skip-existing --pattern '*.json'; done
uv run --with markdown==3.7 python site/build.py --releases releases --docs docs --arena arena --out _site --site-url https://stixels.github.io/yomibench
```

In PowerShell, the download loop is
`gh release list --limit 200 --json tagName -q '.[].tagName' | ForEach-Object { gh release download $_ --dir releases --skip-existing --pattern '*.json' }`.
Open `_site/index.html`, or add `--spa preview.html` for a one-file preview. Delete `releases/` and `_site/` after.

## Submitting a bot (the Arena)

Phase 1 is **data only**: a bundle of nets (arrays, no code) plus a JSON entry, sent through the issue form. The
`submission` workflow runs `tools/validate_submission.py` from `main` on it and comments the result. Nothing
submitted is ever run, and there is no pull-request route for entries. Rated matches are not open yet. The full
guide is [docs/SUBMIT.md](docs/SUBMIT.md) (the site's Submit page):

```
uv run --extra net yomibench bundle pack --name my-bot-1 --owner your-github-username --value PATH/TO/BEST_MODEL.npz
uv run --extra net python tools/validate_submission.py entry.json my-bot-1.zip
```

- `owner` is the GitHub username (the name in `github.com/USERNAME`), not the display name, and must be the
  account that opens the issue.
- Save `entry.json` as UTF-8 without a BOM.
- HustleBot's own published weights are refused; a bundle must hold nets the submitter trained.
- If you help someone submit, give them the form and what to paste. Never open the issue for them.

## Conventions

- **No game content, ever.** Don't commit the game's files or anything extracted, decompiled or copied from it:
  packs, binaries, scripts, scenes, sprites, data tables. Don't paste or transcribe its code; write code that calls
  the game. Players use their own Steam install.
- **Privacy.** No personal data: no real names, handles, emails, IP addresses, hostnames or tokens, in code, tests,
  fixtures, docs or commit messages. Fixtures use made-up names (`my-bot-1`, `your-github-username`). No paths from
  your machine: use placeholders such as `~/.yomibench/train`, `PATH/TO/MODEL.npz` or `%USERPROFILE%`. No human
  players' replays or stats.
- **Small PRs with tests.** One change per PR, with a test that checks behaviour a user can see. Say what changed,
  why, and how you checked it (commands and results). For a speedup or a strength claim, include the numbers and
  how you measured them. Match the surrounding code; prefer deleting code to adding layers.
- **Through a PR only.** Branch, push, open a PR against `main`; a maintainer reviews and merges. Never push to
  `main`.

## What not to touch

Leave these alone unless a maintainer asks for the change:

- **Arena data:** `arena/spec.json` (its pins name the rating era; any change starts a new one), `arena/house.json`,
  `arena/bots.json` and `arena/results/` (written by the Arena's own runs and accepted submissions).
- **Code the Arena pins:** the spec fingerprints the harness's play code, the mod and the candidate rules
  (`mod/YomiBench/candidate_rules_v*.json`, hashed into every record). A change there needs a maintainer's go and a
  note in the PR.
- **Releases:** the `gen-N` releases (results, videos, weights) and the `house-hashes` pre-release (array hashes of
  every house net, which the submission check uses) are published by maintainers. Don't create, edit or delete
  them, and don't commit weights (`*.npz`) or videos.
- **Generated output:** `_site/`, `releases/`, `uv.lock`, `metrics/ledger.jsonl` rows from your own runs, and run
  folders are never committed.
- **Workflows:** changes to `.github/workflows/` (Pages deploy, submission check) need a maintainer's review;
  the submission workflow must never run code from a submission.

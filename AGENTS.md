# AGENTS.md

A guide for coding agents working in `yomibench`. Read this before changing anything. User-facing guides live in
[README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [docs/](docs/); this file points to them rather than
repeating them.

## What this is

`yomibench` is a public harness for search-plus-learning bots for *Your Only Move Is HUSTLE*, an Arena for rating
submitted nets, and **HustleBot**. The site (`https://stixels.github.io/yomibench/`) is built from this repo and its
releases.

- Runs the **player's own Steam install** of the game as headless engine copies. Nothing from the game is in this repo.
- Built-in **toy engine** (`--engine toy`) runs everything without the game.
- HustleBot's nets were trained on a private fast simulator not in this repo; code paths and tests needing it skip
  automatically.

## Ways to hurt yourself

1. **Committing game content.** Never commit game files, packs, binaries, scripts, scenes, sprites, data tables, or
   decompiled/transcribed code. Write code that calls the game.
2. **Leaking personal data.** No real names, handles, emails, IP addresses, hostnames, tokens, machine-specific paths,
   or human replays/stats in code, tests, fixtures, docs, or commits. Use placeholders (`my-bot-1`,
   `your-github-username`, `~/.yomibench/train`, `PATH/TO/MODEL.npz`). Run `uv run python tools/privacy_scan.py .`.
3. **Touching pinned Arena files unasked.** Leave `arena/spec.json`, `arena/house.json`, `arena/bots.json`,
   `arena/results/`, `mod/YomiBench/candidate_rules_v*.json`, and `.github/workflows/` alone unless asked (changing
   fingerprinted play code or candidate rules invalidates the rating era).
4. **Committing run output or pushing to `main`.** Never commit `_site/`, `releases/`, `*.npz`, videos, or
   `metrics/ledger.jsonl` rows from your own runs (use `--no-ledger`). Branch and open a PR; never push to `main`.

## Setup & tests

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Run commands on separate lines (Windows PowerShell 5.1 does not
support `&&`).

```sh
uv sync --extra train --extra test
uv run pytest -q
```

- Iterate with targeted test files: `uv run pytest -q tests/test_site_build.py tests/test_validate_submission.py`.
- Skips for absent optional dependencies (private sim, Steam install, torch, native solver, ffmpeg, Windows) are
  expected. Sync `--extra train --extra test` before judging collection.
- When changing `pyproject.toml`, run `uv lock` and commit `uv.lock`.
- Optional Rust speedup: `native/yomisolve/` (see [native/yomisolve/README.md](native/yomisolve/README.md)).

## Layout

| Path | What |
|---|---|
| `src/yomibench/` | Harness: engine protocol (`engine.py`), real-game engine (`game.py`, `pool_engine.py`), search & bots (`search.py`, `bots/`), self-play, training, gates, ratings, CLI (`cli.py`) |
| `tools/` | `train_loop.py`, `pool_eval.py`, `validate_submission.py`, `privacy_scan.py`, diagnostics |
| `tests/` | pytest suite |
| `mod/YomiBench/` | In-game GDScript engine server (`candidate_rules_v*.json` are hashed into records & Arena spec) |
| `mod/YomiBenchVsBot/` | In-game "Multiplayer (vs bot)" mod |
| `native/yomisolve/` | Optional bit-identical Rust regret-matching solver |
| `docker/` | Linux game build in Docker (mounts user's own Steam library) |
| `site/` | Static site generator (`build.py`, `style.css`) |
| `docs/` | User-facing guides rendered to the site (`DOC_ROUTES` in `site/build.py`) + `PROTOCOL.md` + `SUBMIT.md` |
| `arena/` | Arena spec, house/entered bots, rated results (see [arena/README.md](arena/README.md)) |
| `.github/` | Pages deploy workflow, submission validation workflow (must never execute submitted code) |

## Quick checks

```sh
uv run yomibench arena --engine toy --bots random,greedy,matrix --pairs 4 --no-ledger
uv run python tools/train_loop.py --out ~/.yomibench/toy-train --engine toy --characters Cowboy,Ninja --cycles 1 --games-per-cycle 20 --pairs 2
```

Site build (download release `gen-*.json` into `releases/`, then delete `releases/` and `_site/` after):

```sh
uv run --with markdown==3.7 python site/build.py --releases releases --docs docs --arena arena --out _site --site-url https://stixels.github.io/yomibench
```

## Documentation

Most code changes do not need a documentation change. Agents can read the code.

- `docs/` holds only **user-facing site guides** (`DOC_ROUTES` in `site/build.py`: `SETUP.md`, `TRAINING.md`,
  `METHODOLOGY.md`, `PLAYING.md`, `MOD.md`), the external wire contract (`PROTOCOL.md`), and the Arena submission
  guide (`SUBMIT.md`).
- Do not add internal "how it works" docs, config/plugin catalogs, field or flag enumerations, control-flow narratives,
  or PR summaries. Code, `--help`, `yomibench.local.example.toml`, and tests are the source of truth.
- Put a non-obvious local constraint in a short code comment (1–4 lines explaining *why*, not *what*).
- When documented behavior changes, edit or remove the stale text rather than appending a new account. Avoid hardcoding
  generation numbers or mutable status in prose.

## Plans and work artifacts

- Do not commit implementation plans, research notes, or agent scratch files. Keep temporary material outside the
  worktree; the merged PR is the record.

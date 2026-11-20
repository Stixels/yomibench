# Contributing

Thanks for helping. Bug fixes, faster code, better bots and clearer docs are all welcome.

## How changes land

- **Every change goes through a pull request.** Make a branch (on this repo if you have access,
  or on a fork), push it, and open a PR against `main`.
- **A maintainer reviews and merges.** Even if you have write access, don't merge your own PR,
  and never push to `main` directly: releases are built from it.
- Keep each PR to one change. Say what changed, why, and how you checked it. For a speedup or
  a strength claim, include the numbers and how you measured them (the run's ledger rows, `src/yomibench/ledger.py`).

## Before you open a PR

```
uv sync --extra train --extra test
uv run pytest
```

The tests run without the game. If your change touches the in-game server (`mod/YomiBench/`)
or the real-game engine path, also run it against the game and say so in the PR.

## What never goes in the repo

- **Game content.** *Your Only Move Is HUSTLE* is closed source and isn't ours to redistribute.
  Don't commit its files, or code or data extracted, decompiled or copied from it: packs,
  binaries, scripts, scenes, sprites, data tables. `.gitignore` blocks the common file types,
  but it can't catch pasted code. Write your own code that calls the game. Don't transcribe its
  code.
- **Other people's data.** No replays of human players, player names, or stats taken from other
  players or sites. Summaries we computed ourselves (for example, blended move weights) are fine
  if they can't be traced back to one person.
- **Run output.** Self-play records, checkpoints, caches and logs stay out (`out/`, `runs/`,
  `checkpoints/`, `.cache/`). Share them some other way if a PR needs them.
- **Your machines' details.** No IP addresses, hostnames, home-folder paths, tokens or keys.
  Use placeholders such as `100.x.y.z` or `%USERPROFILE%`.
- **Rows in `metrics/ledger.jsonl` from your own runs.** `selfplay`, `train` and `arena` append to
  it by default. Pass `--ledger PATH` (a file outside the repo) or `--no-ledger`, unless a
  maintainer asked for the row.

## Style

Match the surrounding code. Keep changes small and readable, and prefer deleting code to adding
layers. Tests should check behaviour a user can see, not internals.

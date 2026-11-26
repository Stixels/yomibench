# yomibench

Train and play against search-plus-learning bots for *Your Only Move Is HUSTLE*, and meet **HustleBot**, ours.

> **A fan project.** yomibench is made with the permission of Ivy Sly, the developer of *Your Only Move Is HUSTLE*,
> and credited below. It isn't an official product of the game, and nothing of the game is in this repository: the
> harness runs on your own copy from Steam.

HustleBot looks ahead by previewing what every pair of moves would do, solves the resulting simultaneous-move
payoff table, and learns to judge positions from games it plays against itself. Each new generation has to beat
the previous best in a test whose rule is fixed before any game, before it's promoted.

- **Leaderboard and videos:** [stixels.github.io/yomibench](https://stixels.github.io/yomibench/): the Arena, every
  HustleBot generation, its rating, and replays of its games.
- **Play against it:** an in-game mod that adds **Multiplayer (vs bot)** to the game's menu, on Windows. Step by step:
  [stixels.github.io/yomibench/play](https://stixels.github.io/yomibench/play/).

## What's in the box

- **A harness:** an engine protocol, the real game as an engine (headless copies of your own install),
  search bots, self-play, a value-net trainer, staged promotion tests and ratings.
- **The game mod:** the scripts that turn the game into an engine server, and the vs-bot mod.
- **A toy engine** to try everything without the game.

## Quick start

```
uv sync --extra train --extra test          # Python 3.11+
uv run yomibench arena --engine toy --bots random,greedy,matrix --pairs 4
```

With the game installed through Steam (version 1.10.0):

```
uv run yomibench arena --engine game-pool:auto --bots matrix,matrix-do --characters Cowboy,Ninja --pairs 2
uv run python tools/train_loop.py --out ~/.yomibench/train          # train your own bot (docs/TRAINING.md)
```

## How training works

1. **Preview:** for each pair of candidate moves, the engine plays the turn out.
2. **Solve:** the bot treats the turn as a payoff table and finds a mixed strategy (it can't see your move).
3. **Play itself:** many games against itself, recording every decision.
4. **Learn:** a small network learns to predict who wins from a position.
5. **Gate:** the new net plays the current best in seat-swapped pairs; only a clear win promotes it.

More: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Limits and honesty

- **Your own copy of the game is required.** The harness only runs your own Steam install (version 1.10.0).
  Nothing from the game is included here; headless copies are made from your install, on your PC.
- **Speed:** the real game is slow to simulate. One PC plays tens to low hundreds of games an hour.
- **HustleBot's nets** were trained with a private fast simulator that isn't part of this repository. They're
  offered as starter weights, so you can continue from them instead of starting from scratch. Weights are
  published two generations behind the current best: the newest two generations' nets stay private.
- **Never use bots online.** The mod refuses online sessions and lobbies; it only plays locally against you.

## Docs

[Setup](docs/SETUP.md) · [Training](docs/TRAINING.md) · [Methodology](docs/METHODOLOGY.md) ·
[Playing against the bot](docs/PLAYING.md) · [The game mod](docs/MOD.md) · [Protocol](docs/PROTOCOL.md)

## Acknowledgements

- **Ivy Sly**, developer of *Your Only Move Is HUSTLE*, for permission to build this project, including a Rust rewrite of the engine.
- **TRUEcoiness, MoreBlasta, glopsy**, YOMI Hustle players, for direction and documentation.
- **jackmaiorino**, for [spellbench](https://github.com/jackmaiorino/spellbench), the Magic: The Gathering bot arena yomibench is a sibling of.

## Credits and license

*Your Only Move Is HUSTLE* is made by Ivy Sly; all of the game and its files belong to them. This project
borrows ideas from AlphaZero and KataGo (self-play, playout caps, weight averaging).

MIT, for this code only. See [LICENSE](LICENSE).

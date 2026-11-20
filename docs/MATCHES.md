# Head-to-head matches and the ladder

> A local tool for running your own seasons between bundles on your PC. The public Arena uses the same
> bundles, under its own pinned settings (`arena/spec.json`, [Submit](public/SUBMIT.md)).

How people's trained models play each other fairly. Each side hands over a **bundle** (frozen nets,
no code), and a **ladder season** plays every entrant against every other one in each mode and
keeps an Elo leaderboard. Code: `src/yomibench/ladder.py`.

## Modes

Each mode is a separate division with its own ratings. Inside a mode every entrant runs the same
bot with the same settings; only the weights differ.

| Mode | What picks the move | Bundle needs |
| --- | --- | --- |
| **reflex** (no search) | The policy net, sampled from its softmax. No previews. | policy |
| **light** | The policy shortlists each side's top `light_k` moves (default 6, plus Continue); the value net scores those previews and the solver mixes over them. | policy + value |
| **full** | Double-oracle search over every candidate with factor refinement, scored by the value net. | value |

Budgets are fixed in candidates and solver settings, not seconds, so machine speed never decides a
game. A bundle enters every mode its files allow.

## Bundles

```
yomibench bundle pack --name my-bot-1 --owner your-github-name --value value.npz --policy policy.npz
yomibench bundle check my-bot-1.zip
```

A bundle is a zip of `NAME/bundle.json` plus `value.npz` and/or `policy.npz`. The manifest holds
each file's sha256 and the nets' feature versions. The bundle's digest covers its name and file
hashes. Nets load with `allow_pickle=False`, so a bundle can hold arrays only, and `check` refuses
any changed file, any net of a feature version this code can't play, and any other zip entry. Send
bundles over any channel; the digest proves what arrived.

## A season

A season is a folder. `season.json` pins the engine: the engine build the hello reports, candidate set and
rules, and match settings. It also pins the characters, the seat-swapped pairs per character
pairing per matchup, and the season seed. `play` refuses an engine that differs in any pinned
field, so results are only ever compared on one build. A new build means a new season, and old
bundles can be added to it again.

1. **Seed.** Each side picks a secret and shares `yomibench ladder seed --commit SECRET` first.
   After both reveal, `yomibench ladder seed SECRET_A SECRET_B` gives the season seed. Check each
   reveal against its hash. Neither side could have tuned on these seeds.
2. **Start.** `yomibench ladder init SEASON --engine game-pool:auto --candidate-set 6 --characters Ninja,Cowboy
   --pairs 25 --seed N [--match-settings JSON]`.
3. **Enter.** `yomibench ladder add SEASON ours.zip my-bot-1.zip`. A name is bound to its files; a
   new version enters under a new name (`my-bot-2`). `random` and the hand-written `matrix-do` are
   built-in entrants in every mode. `random` anchors the scale at Elo 1000.
4. **Play.** `yomibench ladder play SEASON --mode light --engine game-pool:auto` plays only the games
   that are missing, one game at a time per engine spec. Use `--dry-run` to list them first and
   `--only NAME` to play one entrant's matchups. A matchup's seeds come from the season seed and
   the two entrants' digests, so a new entrant never changes anyone else's games, and a rerun
   replays the same games. Full games in the real game are slow; leave a PC running.
5. **Rate.** `yomibench ladder rate SEASON` writes `LEADERBOARD.md` and `leaderboard.json`. For each
   mode they show Bradley-Terry ratings on the Elo scale with 95% paired-bootstrap intervals, games
   played, score, and the head-to-head table.

## Splitting the work between two copies

Two people can each keep a copy of the same season: the same `season.json`, the same bundles and
the same engine build. Each plays some matchups (`--only`), then they swap `results/MODE.jsonl`
files and run `yomibench ladder merge SEASON --mode MODE their-results.jsonl`. Merging takes only
rows from this season, from entrants with the same digests and on this season's schedule.
Duplicates drop out. Any game can be replayed to check it, and any game played with `--record-dir`
can be rendered as a video (`tools/render_record.py`).

# How the bot works

*Your Only Move Is HUSTLE* is a turn-based fighting game with **simultaneous** turns: both players pick a move
(with its parameters: aim, timing, direction) without seeing the other's, then the turn plays out. That makes every
turn a small game of guessing, where the best play is usually a mix of moves rather than one fixed choice.

## 1. Candidates

The game's moves have continuous parameters, so the bot first turns the menu into a finite list per player: each
legal move with a set of parameter values (a few aims, directions and timings), plus the extra-panel options
(DI, feints, reverses and each character's own toggles). Legality always comes from the game's own action panel.
The list is the *candidate set*; set 6 is the finest so far. More options mean a stronger but slower bot.

## 2. Preview and solve

For a turn with m candidates for one player and n for the other, the bot previews the pairs: the engine plays the
turn out for each pair and reports the position after it. Each previewed position is scored, which gives an m×n
payoff table. The bot solves it with regret matching for a mixed strategy, the equilibrium of that turn,
then samples its move from that mix. On big turns it refines the table around the best answers instead of
previewing every pair (double oracle).

## 3. Judging positions: the value net

Scoring a previewed position needs a sense of who's winning. That's a small neural network that reads a position
(both fighters' health, meter, positions, states, the clock) and predicts the result. The first generation has
none: it scores positions by health alone.

## 4. Self-play

The current best plays itself. Every decision is recorded: the position, the candidates, the solved strategies.
Some turns use a cheap, capped search (playout cap randomization), so more games fit in the same time, and a
little exploration keeps rare moves in the data.

## 5. Training

The value net trains on those records to predict the final result from each position. A held-out slice of games,
fixed once, measures it honestly. Averaging the weights of the last few checkpoints gives a steadier net.

## 6. Gating

A new net is promoted only if it beats the current best in a test whose rule is fixed before any game: seat-swapped
pairs (each seed played from both sides), staged so that a clear result stops early, with 95% confidence
intervals. Promotion is decided on the result over all matchups, and a floor per matchup blocks a large
regression in one; a small matchup can still come out below 50% (each generation's page shows them).
From generation 16 the test is sequential: a GSPRT over up to 400 pairs, looked at every 20 pairs, with both bots
playing every character (weighted toward Cowboy, as in training), a guard per character and a floor on the
Cowboy mirror.

## 7. Ratings

The leaderboard fits a Bradley–Terry model over every published head-to-head result (the promotion tests, and
the Arena's own ranked games as they come in) and shows it on an Elo-like scale. Two caveats go with it:

- A chain of promotion tests alone overstates gains, since a promoted net is the one that happened to win its test.
  The Arena's own ranked games will correct for that once the redesigned Arena runs.
- Settings changed between eras (characters, candidate sets, the ranked preset). Ratings compare cleanly only
  within an era, and the chart marks the breaks.

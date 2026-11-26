# Engine protocol (v1)

The harness talks to an engine with one JSON object per line, request then response.
Transports: in-process (the toy engine), a subprocess on stdin/stdout, or TCP on
localhost (the in-game YOMI server, `mod/YomiBench`).

Turns are simultaneous: the harness collects one candidate id from each seat, then
sends both in a single `step`. A bot never sees the other seat's pick for the current turn.

## Candidates

The engine owns legality and parameter discretization. Each observation lists, per
seat, the moves that seat may pick right now:

```json
{"id": 3, "action": "Dash", "data": {"Distance": 50, "AutoCorrect": true}, "extra": {}}
```

Bots answer with an `id`. The engine only ever applies data it generated itself, so a bot
cannot crash it with malformed parameters (the real game's native math panics on bad input).

Rules every engine keeps:

- At a decision point both seats' lists are non-empty.
- `candidates[seat][0]` is that seat's **default move**: hold / carry on with what it is doing
  (`"Continue"` with neutral DI in YOMI, `"Wait"` in the toy). Default continuations use it.
- A seat with no real choice this decision gets exactly one candidate, the default.
  `actionable[seat]` is false when the seat only gets to answer the other seat's turn (YOMI's
  `busy_interrupt`). Such a seat still gets whatever the game lets a busy player do (in YOMI a
  burst, or DI while being hit), so it can have more than one candidate; with none of those it has
  exactly one.

## Requests

| `cmd` | fields | response |
|---|---|---|
| `hello` | | `{"engine", "version", "characters", ...}` |
| `new_match` | `p1`, `p2` (character names), `seed` | observation |
| `observe` | | observation |
| `step` | `actions: [id_p1, id_p2]` | observation after the turn resolves |
| `preview` | `pairs: [[id_p1, id_p2], ...]`, `candidates` (bool, default true) | `{"results": [observation, ...]}`, one per pair, each with a `handle`; the real match is unchanged |
| `continue` | `handle`, `pair` (`[id_p1, id_p2]` or `null`), `until`, `candidates` | observation with a new `handle` |
| `release` | `handles: [h, ...]` | `{"released": n}` |
| `expand` (optional) | `specs: [{"seat", "candidate", "extra_override"}, ...]` | `{"results": [{"id", "candidate"} or {"error"}, ...]}` |

`preview` is batched so a bot can fill a whole payoff matrix in one round trip. Each result is a
full observation of the next decision point (or the match end) after that pair, with an integer
`handle` naming it. With `"candidates": false` the result has no `candidates` field. That is
cheaper, and enough for the matrix and greedy bots, which only read `state`. The handle still
works with `continue`.

`continue` simulates on from a handle without touching the real match, and returns a new
observation and handle:

- `pair` ids index **that handle's** candidate list. `null` means the default continuation:
  both seats play candidate 0.
- `until: "decision"` (default) stops at the next decision point.
  `until: {"ticks": N}` keeps going, with both seats playing candidate 0 at every decision point
  in between, until at least `N` game ticks have passed, then stops at the next decision point.
  So every handle is a decision point (or the match end), never mid-turn.
- Continuing from a handle whose match is over is an error (`match_over`).

Handles are valid until the next `step` or `new_match`; using one after that is an error
(`stale_handle`). `release` frees handles early (optional; engines may keep them cheap).
`turn` in a handle's observation counts decisions since the start of the match, real and
simulated.

`expand` (engines with factors, e.g. YOMI candidate set 4): a candidate may carry `factors`, the
things the engine lets a player change on it independently of the move (YOMI: `di`, `feint`,
`reverse`, `extra_set`), and `values`, its own setting of each. `expand` asks for copies of listed
candidates with some factors overridden, e.g. `{"seat": 0, "candidate": 7, "extra_override":
{"di": 3, "feint": true}}`; the engine validates each against what a player could select, appends
the accepted ones to that seat's candidate list for this decision (ids continue the list; each
has `base`, the candidate it derives from) and says why it refused the others. They are ordinary
candidates for `preview` and `step` until the next `step` or `new_match`.
`expand` applies to the real decision only; `continue` pairs still index the chosen handle's own
candidate list. A preview of an expanded pick can be continued through its returned handle. Records keep
`base` and `values` with an expanded pick. YOMI's factor values are in `mod/YomiBench/Candidates.gd`.
`new_match` may take `candidate_sets: [p1, p2]` on engines that offer several (YOMI: 5, 4 and 2;
a seat on 5 plays the server's set-5 switches, `hello.set5`).
`hello.candidate_rules` names the generation rules (a hash), so records say which rules produced
their lists; `tools/candidates_replay.py` checks that an engine reproduces a record's lists, using
only `new_match`, `expand` and `step` (one command per decision).

Engine-specific extras (not required): `step_raw` (engines that can follow a recorded game: `inputs: [{action, data, extra}, {...}]`, each seat's exact input played as given, on or off the candidate grid, answered like `step`; YOMI refuses inputs a player could not have sent), `save_replay` (YOMI: the match so far as a game replay
file; with no `path`, its bytes come back as `replay_b64`). `detail: true` on `observe`/`step`/`preview`/`continue`
adds a fuller `detail` snapshot; YOMI also has `stats` (timing and preview-vs-step divergence
counters) and test/debug commands (`moves`, `probe`, `deep_diff`, `trace: true`; see
`mod/YomiBench/YomiServer.gd`). A `probe` appends a candidate built from chosen widget values to the
current decision's list; bots never need it.

## Observation

```json
{
  "turn": 12,
  "done": false,
  "winner": null,
  "actionable": [true, false],
  "state": {"tick": 140, "players": [{"hp": 1200, "max_hp": 1500, "x": -40, "y": 0, "meter": 2, "burst": 1, "state": "Idle"}, {...}]},
  "candidates": [[...p1 candidates...], [...p2 candidates...]]
}
```

`winner` is `0`, `1`, `null` (draw or unfinished). `state.players[i]` always has `hp` and
`max_hp`; everything else is engine-specific and passed through to bots unchanged. `actionable`
is optional (missing means both).

Candidate ids are positions in this decision's list, nothing more. Anything stored for later
(game records, training data) keeps each pick's `action`, `data` and `extra`, plus the engine's
`candidate_set` from `hello` (bumped whenever the same position can produce a different list). Preview and `continue` results add `"handle"`. When the match
is over both candidate lists are empty.

Errors are `{"error": {"code": "...", "message": "..."}}`. Codes: `bad_request`, `no_match`,
`match_over`, `stale_handle`, `bad_character`, `unknown_cmd`. A TCP engine that runs one match per
process serves one client at a time: it answers a second connection with a `busy` error and
closes it.

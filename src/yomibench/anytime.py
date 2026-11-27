"""Anytime search for human play: answer before the game's turn timer, and (opt-in) think until
the human locks in.

**No turn timer** (timer off, or the standard settings): there is no deadline. The production search
always finishes, and nothing here cuts it (`turn_seconds` is None without a running timer).

**The turn deadline** (a turn timer is running, e.g. ranked with "Turn timer" ticked): a decision has
one turn of the match's turn timer (`turn_seconds`), or less if `GameTimer`, which follows the game's
own clock for the bot's seat (ui/UILayer.gd: per-turn, chess bank or increment), has less left, and
must be answered `margin` before that runs out: max(DEADLINE_MARGIN_S, DEADLINE_FRACTION x that time).
The budget goes to the production search first. Every double-oracle round it solves is kept as its
best-so-far mix (`MatrixBot.progress`), so the answer is, by preference:

- `full`: the production search finished (or a deeper stage after it).
- `partial-full`: the deadline cut the production search; its latest round's mix is played.
- `quick`: the emergency only. If the production search has solved no round by the emergency point
  (`Clock.emergency_at`: max(EMERGENCY_MARGIN_S, EMERGENCY_FRACTION x the time) before the deadline, but
  never in the first half of the time to the deadline), it
  stops, and production's quick turn (`cap_candidates` 32 per seat, options-shuffle, no refinement) gets
  the rest of the time.
- `continue`: not even the quick search finished by the deadline; P2 holds (Continue).

**The quick-menu floor** (`think(floor=True)`; human play turns it on for a PC whose search is slow, `FLOOR_RATE`):
on a decision a clock can cut (a turn deadline or the after-lock cap), the quick search (as the emergency's, above)
runs first, to the end, before the production search. It is certified (no factor refinement, so its double oracle or
full-table fallback always certifies), and it is played (`floor`) unless the production search finishes or its
best-so-far round is converged: an unconverged best-so-far never replaces it, so a slow PC plays a solved small menu
instead of a 1-2 round stub of the full lists (on a slow PC the full search's first rounds can take the whole turn,
and an unconverged round is often a single move at 100%). With a floor there is no emergency point. The floor's
previews of base pairs are handed to the production search's cache (the same states the sim would return), so its
answer is the plain search's.

When the deadline doesn't hit, the answer is exactly the plain search's.

**The after-lock cap** (no turn timer, since private build v5): the bot thinks while the human decides, and once
the human locks in it has at most LOCK_CAP_S (10 s) more (`Clock.lock_cap`). A search that finishes earlier answers
at once (a cap, not a minimum). If the cap cuts the production search, its best-so-far mix is played
(`partial-full`, as for a deadline), the quick search only if it solved no round by LOCK_CAP_S less the emergency
margin after the lock-in, and Continue if not even that finished. Timed decisions never get it: their turn deadline
(above) is unchanged. Once the production search has a converged answer (its base double oracle's equilibrium, kept as
the best-so-far), the cap shrinks to LOCK_CONVERGED_S after the later of the lock-in and that answer ("lock-converged"):
only factor refinement is cut, and the turn plays its latest converged answer. A chess bank turn also stops refinement
at its share of the bank (`bank_refine_s`, "refine-share"); neither rule ever cuts a search that has no converged round.

**Think until I lock in** (on by default since private build v4: the bot uses the human's thinking time): the
bot answers a decision with the deepest search it finished before it had to stop, at the latest THINK_CAP_S
after the decision opened (configurable; a cap on the extra stages only, so an untimed game never waits on
the bot indefinitely; the production search always finishes first). Stages run
in order; each one's answer replaces the stored one only when it completes and is certified
(converged), so a stop mid-stage keeps the last finished answer:

1. `production`: the bot's own search, exactly as human play runs it without this option (the
   same bot, settings and candidate view). Only a hard stop (the turn timer's, or the decision
   being superseded) can cut it short; a lock-in never does, so the answer is never weaker than
   the plain bot's.
2. `widen`: the same search with more factor refinement: DI neighbours for every support move (no
   3-move cap), near-optimal off-support moves expanded too (within 0.02 of the value), 4
   refinement rounds instead of 2. Its own candidate view; previews are shared with stage 1.
3. `two-decision` on the top K support pairs of the stored answer by
   r[i]*c[j] (K = 4, then 8): each pair's previewed state is a live branch (the preview's handle),
   its next decision is solved by double oracle with the evaluator at the leaves (base candidate
   lists, no expand), and that subgame's value replaces the pair's payoff; then the first decision
   is re-solved, and pairs that newly join the top K are branched (at most 2 rounds). Exact: a
   child is the handle's `continue` with that pair, the same clone-and-step a preview is.
4. `two-decision` with K = 16 and up to 3 re-solve rounds.

The plan proposed the Batch's spare slots for the branches. Human play's sim is a Handler (it must
follow the human's off-grid inputs with `step_raw`, which the Batch doesn't have), and the
Handler's preview handle plus `continue` is the same exact branch with no engine change.

Every engine and evaluator call goes through a stop check (`Clock`): previews in chunks of
`PREVIEW_CHUNK` pairs, value batches in chunks of `VALUE_CHUNK` states (stage 1 checks before each
batch but never splits one, so its numbers are the production search's bit for bit), and one check
per `continue`. A lock-in therefore lands within one chunk, about 50 ms.

Nothing here is used by self-play, gates or the arena; with the option off, human play only uses
its `Clock` (a few attributes per decision) and never runs a stage.

**Quick policy** (human_play --quick-policy PATH --quick-policy-sha256 SHA; default off, and off changes
nothing: every answer is bit for bit the one without it). A pinned policy net (yomibench.quick_policy, the
same file and menu rule self-play's quick turns use) helps three places:

- the emergency quick turn searches each seat's policy menu (QUICK_POLICY_MENU at QUICK_POLICY_CAP, as
  self-play's cheap_cap_policy) instead of `cap_candidates` QUICK_CAP options-shuffle;
- stages 1 and 2 warm-start their first double-oracle round from each seat's first candidate plus its
  `oracle_start` best-scored others (`MatrixBot.start_policy`) instead of a seeded random sample. Only the
  start changes: the search still certifies its answer against every candidate, so the production answer
  is an equilibrium of the same lists (it can be another one when the game has several);
- stages 3 and 4 solve each branched pair's next decision over both seats' policy menus (the same rule,
  salted by the pair) instead of every candidate, warm-started the same way. A child's value is the
  evaluator's anyway; the menus let more pairs branch before a stop.
"""

from __future__ import annotations

import copy
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from yomibench.bots import FactorExplore, MatrixBot, policy_start, preview_states, turn_candidates
from yomibench.engine import Candidate, Engine, Observation
from yomibench.playout import cap_candidates
from yomibench.quick_policy import add_tdi
from yomibench.search import double_oracle

PREVIEW_CHUNK = 128  # pairs per preview request in a stoppable stage (~350 us a pair on a loaded Mac)
VALUE_CHUNK = 256    # states per value-net batch in stages 2-4
# A decision is answered max(DEADLINE_MARGIN_S, DEADLINE_FRACTION x its time) before the turn timer
# runs out: a 15 s turn is answered by 12.75 s. Both are configurable (human_play --deadline-margin,
# --deadline-fraction).
DEADLINE_MARGIN_S = 2.0
DEADLINE_FRACTION = 0.15
# With a turn timer and no production round solved this long before the deadline, the emergency quick search
# gets the rest: max(EMERGENCY_MARGIN_S, EMERGENCY_FRACTION x the decision's time). A ranked 15 s turn: by
# 10.5 s the production search must have solved one round, or the quick search runs until 12.75 s.
EMERGENCY_MARGIN_S = 1.5
EMERGENCY_FRACTION = 0.15
# Thinking until lock-in stops its extra stages this long after the decision opened even if the human hasn't locked
# in (human_play --think-cap). A soft stop: the production search before them always finishes.
THINK_CAP_S = 60.0
# No turn timer: once the human locks in, the bot has at most this much longer to answer (human_play --lock-cap).
LOCK_CAP_S = 10.0
# Once the human has locked in AND the production search has a converged (certified) answer, the bot answers within
# this much longer instead of the full after-lock cap (human_play --lock-converged; 0 turns it off): the base double
# oracle's equilibrium is already playable, so only refinement's last rounds are lost, and only on turns where the
# human was fast. Counted from the later of the two. The full LOCK_CAP_S stays while nothing has converged.
LOCK_CONVERGED_S = 1.5
# A chess clock (ranked: a 15-minute bank, a 15 s floor per turn): the bot spends its bank about as the human spends
# theirs, but never plans a decision past (bank left - BANK_RESERVE_S) (human_play --bank-reserve). With no more room
# than the floor's own stop it plays on the floor, the per-turn deadline every timed turn has (bank_budget).
BANK_RESERVE_S = 60.0
# On a chess bank turn the production search may use all of the room above the reserve, but the optional deeper stages
# after it (widen, two-decision) stop at 1/BANK_DEEP_TURNS of that room (bank_deep_s; human_play --bank-deep-turns): the
# game charges the bot's clock while it waits on the human, and thinking until every lock-in drained a 900 s bank to
# the floor in every g16 audit game (by turn 29-74). A share of what is left shrinks as the bank does, so it never
# reaches the floor by itself.
BANK_DEEP_TURNS = 100.0
# On a chess bank turn, the production search's factor refinement (everything after its first converged round) stops at
# 1/BANK_REFINE_TURNS of the room above the reserve (bank_refine_s; human_play --bank-refine-turns; 840 s of room: 21 s),
# and the turn plays the latest converged answer. The base double oracle keeps the whole budget, so the share never
# makes a turn answer unsolved. On a slow PC refinement alone took most of each turn and long games bled the bank to the
# floor (g17 audit: 9-17 floor deadlines in each 100+ turn game); a share of what is left shrinks as the bank does.
BANK_REFINE_TURNS = 40.0
PATHS = ("full", "partial-full", "floor", "quick", "continue")  # what produced a timed answer (logged per turn)
# Human play runs the quick-menu floor (module doc) while the bot's measured preview rate (previews per second of
# search, value net included) is below this, and until it has a measurement: tens per second on a slow PC whose full
# search stops after 1-2 rounds, hundreds on one where it converges.
FLOOR_RATE = 150.0
QUICK_CAP, QUICK_POLICY = 32, "options-shuffle"  # production's quick turn (self-play's cheap_cap / cheap_cap_policy)
# With a quick policy (human_play --quick-policy): self-play's policy-picked quick menus (yomibench.quick_policy).
QUICK_POLICY_MENU, QUICK_POLICY_CAP = "policy40+gumbel8+parryx6", 48
# Emergency quick fallback keeps cap 32 (matching options-shuffle's 32x32 budget so it finishes before the hard stop)
# while still adding up to 8 policy-picked DI/toggle variants per seat via +tdi8.
QUICK_POLICY_EMERGENCY_MENU, QUICK_POLICY_EMERGENCY_CAP = "policy24+gumbel8+tdi8", 32
# The match settings that are the game's turn clock, not the simulation (ui/UILayer.gd and main.gd
# read them; game.gd never does): kept for the deadline, never sent to the sim.
TIMER_KEYS = ("timer_mode", "turn_time", "turn_min_length", "increment_starting_time", "increment_per_turn",
              "increment_max_time")


class Stopped(Exception):
    """A stop check fired; `why` is "lock-in", "deadline", "lock-cap", "lock-converged", "refine-share", "superseded"
    or "emergency"."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


def new_work() -> dict[str, int]:
    """A stage's work tally: pairs previewed in the sim, positions the evaluator scored, value
    batches (one per oracle round or refinement step in matrix-do), expands and continues."""
    return {"previews": 0, "positions": 0, "value_batches": 0, "expands": 0, "continues": 0}


class Clock:
    """The stop signals of one decision. A soft stop (the human locked in) ends the extra stages;
    a hard stop (the turn timer's deadline, a `deadline` message, or the decision superseded by
    the next request) ends any stage, the production search too. Thread-safe: the transport's
    reader thread signals, the search thread checks."""

    def __init__(self, turn: int | None = None, hard_at: float | None = None,
                 superseded: Callable[[], bool] | None = None, emergency_at: float | None = None):
        self.turn, self.hard_at, self.superseded = turn, hard_at, superseded
        # With a turn timer: when a production search that has solved nothing yet gives way to the quick search.
        self.emergency_at = emergency_at
        # Thinking until lock-in: when its extra stages stop even without a lock-in (a soft stop: THINK_CAP_S).
        self.cap_at: float | None = None
        # The after-lock cap (LOCK_CAP_S): no turn timer, or a chess clock's bank turn (beside its `hard_at`, the
        # budget's end). Once the human locks in, a hard stop `lock_cap` seconds later ("lock-cap"), with an
        # emergency point before it. Read lazily from `soft_at`, so a lock-in that arrives before the decision set
        # this still counts. Unset (None) on a floor or per-turn timed decision: its deadline only.
        self.lock_cap: float | None = None
        # LOCK_CONVERGED_S: with the after-lock cap set, a hard stop ("lock-converged") this long after the later of
        # the lock-in and `converged_at`. None: the full cap only.
        self.lock_converged: float | None = None
        # When the production search first had a converged answer (monotonic; set by its progress hook), else None.
        self.converged_at: float | None = None
        # A chess bank turn's refinement share (bank_refine_s): past this (monotonic), a production search that has a
        # converged answer stops ("refine-share") and plays it. Unset (None) elsewhere; never cuts an unsolved search.
        self.refine_at: float | None = None
        self.t0 = time.monotonic()
        self._soft, self._hard = threading.Event(), threading.Event()
        self._soft_why = self._hard_why = ""
        self.soft_at: float | None = None  # when the lock-in arrived (monotonic)
        # The running stage's work so far (reset by `run` per stage; Guarded/GuardedValue add to it after
        # each call returns): logged per stage, finished or cut. Counting only: no extra compute.
        self.work: dict[str, int] = new_work()
        # The longest single guarded step (a preview chunk, a value batch, an expand, a continue) this decision: the
        # after-lock cap stops before a step that would end past it, so the answer comes by the cap, not one step
        # later. (A turn deadline keeps its margin instead: unchanged.)
        self.step_s = 0.0

    def note_step(self, seconds: float) -> None:
        if seconds > self.step_s:
            self.step_s = seconds

    def signal(self, kind: str) -> None:
        """A notification for this decision: `locked_in` (soft) or `deadline` (hard)."""
        if kind == "deadline":
            self.stop("deadline")
        else:
            if not self._soft.is_set():
                self._soft_why, self.soft_at = "lock-in", time.monotonic()
            self._soft.set()

    def stop(self, why: str) -> None:
        if not self._hard.is_set():
            self._hard_why = why
        self._hard.set()

    def hard(self) -> str | None:
        if self._hard.is_set():
            return self._hard_why
        if self.hard_at is not None and time.monotonic() >= self.hard_at:
            return "deadline"
        at = self.lock_hard_at()
        if at is not None and time.monotonic() + self.step_s >= at:
            return "lock-cap"
        at = self.lock_converged_at()
        if at is not None and time.monotonic() + self.step_s >= at:
            return "lock-converged"
        if self.superseded is not None and self.superseded():
            return "superseded"
        return None

    def lock_hard_at(self) -> float | None:
        """The after-lock cap's hard stop (monotonic), once the human has locked in; None without one."""
        if self.lock_cap is None or self.soft_at is None:
            return None
        return self.soft_at + self.lock_cap

    def lock_converged_at(self) -> float | None:
        """The early after-lock stop (monotonic): `lock_converged` after the later of the lock-in and the production
        search's first converged answer. None until both have happened, or without the after-lock cap or this."""
        if self.lock_cap is None or not self.lock_converged or self.soft_at is None or self.converged_at is None:
            return None
        return max(self.soft_at, self.converged_at) + self.lock_converged

    def refine_over(self) -> bool:
        """A bank turn's refinement share is spent and the production search has a converged answer to play."""
        return (self.refine_at is not None and self.converged_at is not None
                and time.monotonic() + self.step_s >= self.refine_at)

    def emergency(self) -> float | None:
        """When a production search that has solved nothing yet gives way to the quick search: the turn
        deadline's emergency point, else the after-lock cap's (its LOCK_CAP_S less the emergency margin)."""
        at = self.lock_hard_at()
        lock = None
        if at is not None:
            assert self.lock_cap is not None
            lock = at - self.lock_cap + emergency_within(self.lock_cap, self.lock_cap)
        both = [t for t in (self.emergency_at, lock) if t is not None]
        return min(both) if both else None

    def soft(self) -> str | None:
        why = self.hard() or (self._soft_why if self._soft.is_set() else None)
        if why is None and self.cap_at is not None and time.monotonic() >= self.cap_at:
            why = "think-cap"
        return why

    def check_hard(self) -> None:
        why = self.hard()
        if why:
            raise Stopped(why)

    def check_soft(self) -> None:
        why = self.soft()
        if why:
            raise Stopped(why)


def turn_seconds(timer_settings: dict[str, Any]) -> float | None:
    """The seconds one turn has on the match's turn timer, which every answer must beat:
    `chess` its per-turn minimum `turn_min_length` (ranked: 15 s), `increment` its
    `increment_per_turn`, and otherwise (`default`) `turn_time` seconds. None when no timer runs
    (`timer_mode` "none" or absent: the "Turn timer" box unticked, or the standard settings), so an
    untimed decision has no deadline and its full search always finishes; also None when the
    settings name no positive per-turn time at all."""
    mode = timer_settings.get("timer_mode")
    if not isinstance(mode, str) or mode == "none":
        return None
    key = {"chess": "turn_min_length", "increment": "increment_per_turn"}.get(mode, "turn_time")
    v = _num(timer_settings.get(key))
    if (v is None or v <= 0) and key != "turn_time":
        v = _num(timer_settings.get("turn_time")) if mode != "chess" else None
    return v if v is not None and v > 0 else None


def answer_within(available: float, margin_s: float = DEADLINE_MARGIN_S,
                  fraction: float = DEADLINE_FRACTION) -> float:
    """How long a decision with `available` seconds on the clock may take: the time less
    max(margin_s, fraction x the time), never below 0."""
    return max(0.0, available - max(margin_s, fraction * available))


def emergency_within(available: float, answer_s: float) -> float:
    """When (seconds into a decision that must be answered by `answer_s`, with `available` on the clock) a
    production search that has solved nothing yet gives way to the quick search: `answer_s` less
    max(EMERGENCY_MARGIN_S, EMERGENCY_FRACTION x `available`), but never before half of `answer_s` (the full
    search always gets most of a short turn)."""
    return max(answer_s / 2, answer_s - max(EMERGENCY_MARGIN_S, EMERGENCY_FRACTION * available))


def bank_budget(bank_left: float, turn_s: float, reserve_s: float = BANK_RESERVE_S,
                margin_s: float = DEADLINE_MARGIN_S, fraction: float = DEADLINE_FRACTION) -> tuple[str, float]:
    """A chess-clock decision's plan, as (mode, budget_s), with `bank_left` seconds on the bot's clock (after the
    turn's top-up to the floor) and `turn_s` the floor (ranked: 15 s):

    - "bank": the room above the reserve, `bank_left - reserve_s`, when that is more than the floor's own stop.
      The bot thinks while the human does and answers once the human has locked in (plus at most the after-lock
      cap), when its searches end, or at the budget's end, whichever is first; never past the reserve.
    - "floor": otherwise the per-turn stop every timed turn has, answer_within(min(bank_left, turn_s)): a ranked
      turn is answered by 12.75 s, inside the 15 s the game tops a low bank up to."""
    floor = answer_within(min(bank_left, turn_s), margin_s, fraction)
    room = bank_left - reserve_s
    if room > floor:
        return "bank", room
    return "floor", floor


def bank_deep_s(bank_left: float, reserve_s: float = BANK_RESERVE_S, turns: float = BANK_DEEP_TURNS) -> float:
    """How long into a chess-clock bank turn the optional deeper stages may run: this turn's share of the room above
    the reserve, (bank_left - reserve_s) / turns (900 s and the 60 s reserve: 8.4 s). The production search is not
    bound by it (only the bank budget's hard stop and the after-lock cap cut that)."""
    return max(0.0, bank_left - reserve_s) / turns


def bank_refine_s(bank_left: float, reserve_s: float = BANK_RESERVE_S, turns: float = BANK_REFINE_TURNS) -> float:
    """How long into a chess-clock bank turn the production search's factor refinement may run: this turn's share of
    the room above the reserve, (bank_left - reserve_s) / turns (900 s and the 60 s reserve: 21 s). Past it the turn
    plays the latest converged answer; a search with no converged round yet keeps the whole budget (`Clock.refine_at`)."""
    return max(0.0, bank_left - reserve_s) / turns


def _num(v: Any) -> float | None:
    return None if isinstance(v, bool) or not isinstance(v, (int, float)) else float(v)


class GameTimer:
    """The game's turn timer for the bot's seat, as the real game runs it (ui/UILayer.gd,
    on_player_actionable and _apply_increment; main.gd sets it up):

    - `default`: `turn_time` seconds every turn.
    - `chess`: a bank of `turn_time` minutes; each turn after the first, a bank under
      `turn_min_length` seconds is topped up to it. Ranked: a 15-minute bank, 15 s minimum.
    - `increment`: a bank of max(`increment_starting_time`, `increment_per_turn`) seconds, plus
      `increment_per_turn` each turn after the first, capped at `increment_max_time` minutes (0: none).

    `available()` is the time this decision has; `spend(s)` charges what answering took (the bot's
    clock runs from the decision to its answer). `of(settings)` is None for no timer
    (`timer_mode` "none" or absent: the singleplayer default)."""

    def __init__(self, mode: str, turn_time: float, turn_min: float, inc: float, inc_start: float, inc_max_min: float):
        self.mode, self.turn_time, self.turn_min = mode, turn_time, turn_min
        self.inc, self.inc_max = inc, inc_max_min * 60
        self.first = True
        if mode == "chess":
            self.bank = turn_time * 60
        elif mode == "increment":
            self.bank = self._cap(max(inc_start, inc))
        else:
            self.bank = turn_time

    @staticmethod
    def of(settings: dict[str, Any]) -> "GameTimer | None":
        mode = settings.get("timer_mode")
        if not isinstance(mode, str) or mode == "none":
            return None
        tt, tmin = _num(settings.get("turn_time")), _num(settings.get("turn_min_length"))
        inc, inc0, incmax = (_num(settings.get(k)) for k in ("increment_per_turn", "increment_starting_time",
                                                               "increment_max_time"))
        if mode == "increment":
            inc = 10.0 if inc is None else inc  # main.gd's defaults
            return GameTimer(mode, 0.0, 0.0, inc, 60.0 if inc0 is None else inc0, incmax or 0.0)
        if tt is None or tt <= 0:
            return None  # main.gd starts no timer without turn_time
        # default, chess, and anything else the game would run like default (seconds per turn)
        return GameTimer(mode if mode == "chess" else "default", tt, tmin or 0.0, 0.0, 0.0, 0.0)

    def _cap(self, bank: float) -> float:
        return min(bank, self.inc_max) if self.inc_max > 0 else bank

    def available(self) -> float:
        if self.mode == "default":
            return self.turn_time
        if not self.first:
            if self.mode == "chess" and self.bank < self.turn_min:
                self.bank = self.turn_min
            elif self.mode == "increment":
                self.bank = self._cap(self.bank + self.inc)
        self.first = False
        return self.bank

    def spend(self, seconds: float) -> None:
        if self.mode != "default":
            self.bank = max(0.0, self.bank - seconds)

    def state(self) -> dict[str, Any]:
        return {"mode": self.mode, "bank_s": round(self.bank, 3)}


class Guarded:
    """An Engine with a stop check before every preview chunk, expand and continue. Attribute
    reads and writes (turn_cache, stats, match_seed) go to the engine itself."""

    def __init__(self, engine: Engine, check: Callable[[], None], chunk: int = PREVIEW_CHUNK,
                 work: dict[str, int] | None = None, step: Callable[[float], None] | None = None):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_check", check)
        object.__setattr__(self, "_chunk", max(1, chunk))
        object.__setattr__(self, "_work", work if work is not None else new_work())
        object.__setattr__(self, "_step", step or (lambda _s: None))  # each step's duration (Clock.note_step)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._engine, name, value)

    def preview(self, pairs: list[tuple[int, int]], candidates: bool = True) -> list[Observation]:
        out: list[Observation] = []
        for k in range(0, len(pairs), self._chunk):
            self._check()
            chunk = pairs[k:k + self._chunk]
            t = time.monotonic()
            out += self._engine.preview(chunk, candidates)
            self._step(time.monotonic() - t)
            self._work["previews"] += len(chunk)
        return out

    def expand(self, specs: list[dict[str, Any]]) -> list[Candidate | str]:
        self._check()
        t = time.monotonic()
        out = self._engine.expand(specs)
        self._step(time.monotonic() - t)
        self._work["expands"] += len(specs)
        return out

    def continue_from(self, handle: int, pair: tuple[int, int] | None = None, until: Any = "decision",
                      candidates: bool = True) -> Observation:
        self._check()
        t = time.monotonic()
        out = self._engine.continue_from(handle, pair, until, candidates)
        self._step(time.monotonic() - t)
        self._work["continues"] += 1
        return out


class GuardedValue:
    """An evaluator (a function, or a value net with `batch`) with a stop check before each call.
    `chunk` splits batches (stages 2-4); None keeps each batch whole (stage 1: bit-identical)."""

    def __init__(self, value: Callable[[dict[str, Any], int], float], check: Callable[[], None],
                 chunk: int | None = None, work: dict[str, int] | None = None,
                 step: Callable[[float], None] | None = None):
        self.value, self.check, self.chunk = value, check, chunk
        self.work = work if work is not None else new_work()
        self.step = step or (lambda _s: None)  # each batch's duration (Clock.note_step)
        self.__name__ = getattr(value, "__name__", repr(value))
        if getattr(value, "batch", None) is not None:
            self.batch = self._batch

    def __call__(self, state: dict[str, Any], seat: int) -> float:
        self.check()
        out = self.value(state, seat)
        self.work["positions"] += 1
        return out

    def _batch(self, states: list[dict[str, Any]], seat: int) -> Any:
        if not self.chunk:
            self.check()
            t = time.monotonic()
            res = self.value.batch(states, seat)
            self.step(time.monotonic() - t)
            self.work["positions"] += len(states)
            self.work["value_batches"] += 1
            return res
        out: list[float] = []
        for k in range(0, len(states), self.chunk):
            self.check()
            part = states[k:k + self.chunk]
            t = time.monotonic()
            out += map(float, self.value.batch(part, seat))
            self.step(time.monotonic() - t)
            self.work["positions"] += len(part)
        self.work["value_batches"] += 1
        return out

    def many(self, states: list[dict[str, Any]], seat: int = 0) -> list[float]:
        if not states:
            return []
        if getattr(self, "batch", None) is not None:
            return list(map(float, self.batch(states, seat)))
        out = [float(self(st, seat)) for st in states]
        self.work["value_batches"] += 1
        return out


@dataclass
class Answer:
    """One finished stage: both seats' strategies over `lists` (a bot's candidate view), p1's value."""
    stage: int
    name: str
    lists: list[list[Candidate]]
    row: list[float]
    col: list[float]
    value: float
    view: Any = None
    pairs: int = 0          # two-decision: first-decision pairs whose payoff is a solved subgame
    info: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0

    def strategy(self, seat: int) -> tuple[list[float], float]:
        return (self.row, self.value) if seat == 0 else (self.col, -self.value)

    def label(self) -> str:
        return {"quick": "quick search", "floor": "quick search (floor)", "production": "full search",
                "partial-full": "full search, best so far", "widen": "widened"}.get(self.name, f"2-decision x{self.pairs}")


def widened(bot: MatrixBot) -> MatrixBot:
    """Stage 2's bot: the production bot with more factor refinement (see the module doc)."""
    b = copy.copy(bot)
    b.refine_rounds = max(4, bot.refine_rounds)
    b.di_refine_moves = None
    b.explore = FactorExplore(margin=0.02, random_k=0, all_prob=0.0)
    return b


class Deepener:
    """The stages for one decision of `bot` (the production MatrixBot) as `seat` on `engine`."""

    TWO_DECISION = ((3, 4, 2), (3, 8, 2), (4, 16, 3))  # (stage, K, re-solve rounds)
    TWO_DECISION_POLICY = ((5, 32, 3),)  # wider depth-2 when quick_policy is on
    THREE_DECISION_POLICY = ((6, 4, 4, 2), (6, 8, 4, 2))  # (stage, root_k, child_k, rounds) when quick_policy is on

    def __init__(self, bot: MatrixBot, obs: Observation, seat: int, engine: Engine, clock: Clock,
                 quick_policy: Any = None):
        if not isinstance(bot, MatrixBot):
            raise ValueError("thinking until lock-in needs a matrix bot")
        self.bot, self.obs, self.seat, self.engine, self.clock = bot, obs, seat, engine, clock
        # A pinned yomibench.quick_policy.QuickPolicy (human_play --quick-policy), or None: off (see the module doc).
        self.quick_policy = quick_policy
        self.depth2: dict[tuple[int, int], float] = {}  # (id p1, id p2) -> solved subgame value (p1's)
        self.depth3: dict[tuple[int, int], float] = {}  # (id p1, id p2) -> depth-3 solved subgame value (p1's)
        self.leaf: dict[tuple[Any, int, int], float] = {}  # (view, i, j) -> evaluator value (p1's)
        self.unbranchable: set[tuple[int, int]] = set()
        self.unbranchable3: set[tuple[int, int]] = set()
        self.child_previews = 0
        # The production search's best-so-far: its latest solved double-oracle round (MatrixBot.progress).
        self.partial: Answer | None = None
        self.partial_rounds = 0
        # The quick-menu floor (module doc), once it has finished; None: off or not finished.
        self.floor: Answer | None = None
        self._quick_cache: dict[str, Any] = {}  # the last quick search's private turn cache
        # A bot's search key holds id(value): keep every stage's evaluator alive for the decision,
        # so no later stage's evaluator can reuse an id and meet an earlier stage's cached search.
        self.keep: list[Any] = []

    def stages(self, deep: bool = True) -> list[Stage]:
        """The production search, then with `deep` the extra stages (think until lock-in). The production
        stage stops only on a hard stop, or (a turn timer) at the emergency point while it has solved
        nothing (see `think`)."""
        out: list[Stage] = [(1, lambda _b: self.production(), False)]
        if deep:
            out.append((2, self.widen, True))
            for stage, k, rounds in self.TWO_DECISION:
                out.append((stage, (lambda best, s=stage, k=k, r=rounds: self.two_decision(best, s, k, r)), True))
            if self.quick_policy is not None:
                for stage, k, rounds in self.TWO_DECISION_POLICY:
                    out.append((stage, (lambda best, s=stage, k=k, r=rounds: self.two_decision(best, s, k, r)), True))
                for stage, k, ck, rounds in self.THREE_DECISION_POLICY:
                    out.append((stage, (lambda best, s=stage, k=k, ck=ck, r=rounds:
                                        self.three_decision(best, s, k, ck, r)), True))
        return out

    # --- stage 0: the emergency quick search -----------------------------------------------------

    def quick(self) -> Answer:
        """The emergency answer (a turn timer, and the production search solved nothing by the emergency
        point): production's quick turn: each seat's candidates capped at QUICK_CAP (QUICK_POLICY),
        no factor refinement, searched in a private turn cache so nothing of it reaches the
        production search's lists or previews."""
        return self._quick_search("quick")

    def _quick_search(self, name: str) -> Answer:
        """The quick turn's search (`quick`), named `name`: the emergency's or the floor's (`run_floor`)."""
        check = self.clock.check_hard
        b = self.bot.cheap()
        b.value = GuardedValue(self.bot.value, check, VALUE_CHUNK, self.clock.work, step=self.clock.note_step)
        self.keep.append(b.value)
        eng = Guarded(self.engine, check, work=self.clock.work, step=self.clock.note_step)
        qp = self.quick_policy
        if qp is None:
            cap, cap_policy = QUICK_CAP, QUICK_POLICY
            base = [cap_candidates(c, QUICK_CAP, QUICK_POLICY, f"{self.engine.match_seed}:{self.obs.turn}:{s}")
                    for s, c in enumerate(self.obs.candidates)]
        else:  # emergency quick fallback: cap-32 policy menu + up to 8 DI/toggle variants per seat
            cap, cap_policy = QUICK_POLICY_EMERGENCY_CAP, QUICK_POLICY_EMERGENCY_MENU
            salt = f"{self.engine.match_seed}:{self.obs.turn}"
            check()
            base = qp.menus(self.obs.state, self.obs.candidates, cap, cap_policy, salt)
            check()
            base = add_tdi(eng, self.obs.state, base, cap_policy, salt, qp)
        shared, private = self.engine.turn_cache, {"base": base}
        self.engine.turn_cache = private
        try:
            row, col, value = b.search_turn(self.obs, eng, self.seat)  # type: ignore[arg-type]
            lists = turn_candidates(self.obs, eng, b._key(self.seat))  # type: ignore[arg-type]
        finally:
            self.engine.turn_cache = shared
        info = {**dict(private.get("oracle") or {}), "cap": cap, "cap_policy": cap_policy}
        if qp is not None:
            info["quick_policy"] = qp.sha256
        self._quick_cache = private
        return Answer(0, name, lists, row, col, value, info=info)

    def run_floor(self) -> Answer:
        """The quick-menu floor (module doc): the quick search, then its previews of base x base pairs copied into
        the shared turn cache, keyed as the production search looks them up (positions in the full lists). A
        preview is the sim's next state for a pair from this position, so the production search finds the same
        states it would have previewed itself: only fewer sim calls."""
        ans = self._quick_search("floor")
        private = self._quick_cache
        pc, done = private.get("candidates"), private.get("previewed") or {}
        if pc and done:
            full = turn_candidates(self.obs, self.engine)
            pos = [{c.id: k for k, c in enumerate(full[s])} for s in (0, 1)]
            previewed = self.engine.turn_cache.setdefault("previewed", {})
            shared = 0
            for (i, j), st in done.items():
                a, b = pos[0].get(pc[0][i].id), pos[1].get(pc[1][j].id)
                if a is not None and b is not None and (a, b) not in previewed:
                    previewed[(a, b)] = st
                    shared += 1
            ans.info["shared_previews"] = shared
        self.floor = ans
        return ans

    # --- stages 1 and 2: the matrix bot's own search -----------------------------------------

    def _search(self, bot: MatrixBot, stage: int, name: str, check: Callable[[], None],
                chunk: int | None, progress: Callable[..., None] | None = None) -> Answer:
        b = copy.copy(bot)
        b.progress = progress
        if self.quick_policy is not None:
            b.start_policy = self.quick_policy  # the warm start (module doc); its own search key and view
        b.value = GuardedValue(bot.value, check, chunk, self.clock.work, step=self.clock.note_step)
        self.keep.append(b.value)
        eng = Guarded(self.engine, check, work=self.clock.work, step=self.clock.note_step)
        row, col, value = b.search_turn(self.obs, eng, self.seat)  # type: ignore[arg-type]
        key = b._key(self.seat)
        lists = turn_candidates(self.obs, eng, key)  # type: ignore[arg-type]
        info = dict(self.engine.turn_cache.get("oracle") or {})
        return Answer(stage, name, lists, row, col, value, view=key, info=info)

    def production(self) -> Answer:
        def check() -> None:
            self.clock.check_hard()
            at = self.clock.emergency()
            if at is not None and self.partial is None and self.floor is None and time.monotonic() >= at:
                raise Stopped("emergency")
            if self.clock.refine_over():
                raise Stopped("refine-share")

        def progress(row: list[float], col: list[float], value: float, lists: list[list[Candidate]],
                     converged: bool) -> None:
            # A certified (converged) best-so-far is replaced only by another converged one. The base double oracle
            # converges, then factor refinement runs further double-oracle searches whose early rounds are unconverged
            # mixes: letting those replace it played unsolved strategies whenever the clock cut refinement (g17 human
            # audit, 11/23: 106 of 116 unconverged partial-full turns, e.g. Taunt at 100% at 144 HP).
            self.partial_rounds += 1
            if self.partial is not None and self.partial.info.get("converged") and not converged:
                self.partial.info["unconverged_rounds_after"] = self.partial.info.get("unconverged_rounds_after", 0) + 1
                return
            self.partial = Answer(1, "partial-full", lists, row, col, value, info={"converged": converged})
            if converged and self.clock.converged_at is None:  # the base double oracle's equilibrium: playable now
                self.clock.converged_at = time.monotonic()

        return self._search(self.bot, 1, "production", check, None, progress)

    def widen(self, _best: Answer | None) -> Answer | None:
        if self.bot.mode != "oracle":
            return None  # a full-table bot has nothing to refine
        return self._search(widened(self.bot), 2, "widen", self.clock.check_soft, VALUE_CHUNK)

    # --- stages 3 and 4: two-decision search on the stored answer ----------------------------

    def _leaf_values(self, base: Answer, value: GuardedValue, eng: Guarded):
        lists = base.lists

        def evaluate(pairs: list[tuple[int, int]]) -> list[float]:
            missing = [p for p in dict.fromkeys(pairs) if (base.view, *p) not in self.leaf]
            if missing:
                states = preview_states(self.obs, eng, missing, base.view)  # type: ignore[arg-type]
                for p, v in zip(missing, value.many(states)):
                    self.leaf[(base.view, *p)] = v
            out = []
            for i, j in pairs:
                ids = (lists[0][i].id, lists[1][j].id)
                out.append(self.depth3.get(ids, self.depth2.get(ids, self.leaf[(base.view, i, j)])))
            return out
        return evaluate

    def branch(self, ids: tuple[int, int], value: GuardedValue, eng: Guarded) -> float | None:
        """The solved next decision after the pair `ids` (p1's value), or None (match over, no
        choice to solve, or the child search didn't converge: the pair keeps its one-decision
        payoff)."""
        (h,) = eng.preview([ids], candidates=True)
        handles = [h.handle] if h.handle is not None else []
        try:
            if h.done or h.handle is None:
                return None
            c0, c1 = h.candidates
            if not c0 or not c1 or len(c0) * len(c1) == 1:
                return None
            seed = f"{self.engine.match_seed}:{self.obs.turn}:{ids[0]}:{ids[1]}:depth2"
            qp = self.quick_policy
            if qp is not None:  # both seats' policy menus, warm-started from their best scores (module doc)
                self.clock.check_soft()
                (c0, s0), (c1, s1) = qp.scored_menus(h.state, [c0, c1], QUICK_POLICY_CAP, QUICK_POLICY_MENU, seed,
                                                     need_scores=True)
                start = (policy_start(s0, self.bot.oracle_start), policy_start(s1, self.bot.oracle_start))
            else:
                start = tuple([0] + random.Random(f"{seed}:{s}").sample(range(1, n), min(self.bot.oracle_start, n - 1))
                              for s, n in enumerate((len(c0), len(c1))))

            def child(pairs: list[tuple[int, int]]) -> list[float]:
                out: list[float] = []
                for k in range(0, len(pairs), PREVIEW_CHUNK):
                    obs = [eng.continue_from(h.handle, (c0[a].id, c1[b].id), candidates=False)
                           for a, b in pairs[k:k + PREVIEW_CHUNK]]
                    self.child_previews += len(obs)
                    self.engine.release([o.handle for o in obs if o.handle is not None])
                    out += value.many([o.state for o in obs])
                return out

            r = double_oracle(len(c0), len(c1), child, self.bot.iterations, self.bot.tolerance, start=start)
            return r.value if r.converged else None
        finally:
            if handles:
                self.engine.release(handles)

    def two_decision(self, base: Answer | None, stage: int, k: int, rounds: int) -> Answer | None:
        if base is None:
            return None
        check = self.clock.check_soft
        value = GuardedValue(self.bot.value, check, VALUE_CHUNK, self.clock.work, step=self.clock.note_step)
        eng = Guarded(self.engine, check, work=self.clock.work, step=self.clock.note_step)
        lists, row, col = base.lists, base.row, base.col
        evaluate = self._leaf_values(base, value, eng)
        n0, n1 = len(lists[0]), len(lists[1])
        result, refused = None, 0
        for rnd in range(rounds):
            ranked = sorted(((row[i] * col[j], i, j) for i in range(n0) if row[i] > 0 for j in range(n1) if col[j] > 0),
                            key=lambda t: (-t[0], t[1], t[2]))[:k]
            todo = [(lists[0][i].id, lists[1][j].id) for _, i, j in ranked]
            todo = [ids for ids in todo if ids not in self.depth2 and ids not in self.unbranchable]
            if not todo and result is not None:
                break
            for ids in todo:
                v = self.branch(ids, value, eng)
                if v is None:
                    self.unbranchable.add(ids)
                    refused += 1
                else:
                    self.depth2[ids] = v
            support = ([i for i in range(n0) if row[i] > 0], [j for j in range(n1) if col[j] > 0])
            result = double_oracle(n0, n1, evaluate, self.bot.iterations, self.bot.tolerance, start=support)
            if not result.converged:
                return None  # never store an uncertified answer
            row, col = result.row, result.col
        assert result is not None
        pairs = len(self.depth2)
        info = {"rounds": result.rounds, "converged": True, "depth2_pairs": pairs, "unbranchable": refused,
                "child_previews": self.child_previews, "k": k}
        return Answer(stage, "two-decision", lists, row, col, result.value, view=base.view, pairs=pairs, info=info)

    # --- stage 6: three-decision (depth-3) search when quick_policy is on --------------------

    def _solve_policy_handle(self, h: Observation, seed: str, value: GuardedValue, eng: Guarded):
        """Solve one preview handle `h` over both seats' policy menus; returns `(result, c0, c1, cache)`
        or None if terminal/trivial/unconverged."""
        if h.done or h.handle is None:
            return None
        c0, c1 = h.candidates
        if not c0 or not c1 or len(c0) * len(c1) == 1:
            return None
        qp = self.quick_policy
        assert qp is not None
        self.clock.check_soft()
        (c0, s0), (c1, s1) = qp.scored_menus(h.state, [c0, c1], QUICK_POLICY_CAP, QUICK_POLICY_MENU, seed,
                                             need_scores=True)
        start = (policy_start(s0, self.bot.oracle_start), policy_start(s1, self.bot.oracle_start))
        cache: dict[tuple[int, int], float] = {}

        def evaluate(pairs: list[tuple[int, int]]) -> list[float]:
            missing = [p for p in dict.fromkeys(pairs) if p not in cache]
            for k in range(0, len(missing), PREVIEW_CHUNK):
                batch = missing[k:k + PREVIEW_CHUNK]
                obs = [eng.continue_from(h.handle, (c0[a].id, c1[b].id), candidates=False) for a, b in batch]
                self.child_previews += len(obs)
                self.engine.release([o.handle for o in obs if o.handle is not None])
                for p, v in zip(batch, value.many([o.state for o in obs])):
                    cache[p] = v
            return [cache[p] for p in pairs]

        r = double_oracle(len(c0), len(c1), evaluate, self.bot.iterations, self.bot.tolerance, start=start)
        return (r, c0, c1, cache) if r.converged else None

    def branch_depth3(self, ids: tuple[int, int], child_k: int, value: GuardedValue, eng: Guarded) -> float | None:
        """Depth-3 value for root pair `ids`: solve its depth-2 subgame, expand its top `child_k` support
        pairs one more turn over policy menus, and re-solve the depth-2 subgame."""
        (h,) = eng.preview([ids], candidates=True)
        handles = [h.handle] if h.handle is not None else []
        try:
            seed2 = f"{self.engine.match_seed}:{self.obs.turn}:{ids[0]}:{ids[1]}:depth2"
            solved = self._solve_policy_handle(h, seed2, value, eng)
            if solved is None:
                return None
            r, c0, c1, cache = solved
            ranked = sorted(((r.row[a] * r.col[b], a, b)
                             for a in range(len(c0)) if r.row[a] > 0
                             for b in range(len(c1)) if r.col[b] > 0),
                            key=lambda t: (-t[0], t[1], t[2]))[:child_k]
            d3: dict[tuple[int, int], float] = {}
            for _, a, b in ranked:
                h2 = eng.continue_from(h.handle, (c0[a].id, c1[b].id), candidates=True)
                self.child_previews += 1
                try:
                    seed3 = f"{seed2}:{c0[a].id}:{c1[b].id}:depth3"
                    s3 = self._solve_policy_handle(h2, seed3, value, eng)
                    if s3 is not None:
                        d3[(a, b)] = s3[0].value
                finally:
                    if h2.handle is not None:
                        self.engine.release([h2.handle])
            if not d3:
                return r.value

            def evaluate_up(pairs: list[tuple[int, int]]) -> list[float]:
                missing = [p for p in dict.fromkeys(pairs) if p not in d3 and p not in cache]
                for k in range(0, len(missing), PREVIEW_CHUNK):
                    batch = missing[k:k + PREVIEW_CHUNK]
                    obs = [eng.continue_from(h.handle, (c0[a].id, c1[b].id), candidates=False) for a, b in batch]
                    self.child_previews += len(obs)
                    self.engine.release([o.handle for o in obs if o.handle is not None])
                    for p, v in zip(batch, value.many([o.state for o in obs])):
                        cache[p] = v
                return [d3.get(p, cache[p]) for p in pairs]

            support = ([a for a in range(len(c0)) if r.row[a] > 0], [b for b in range(len(c1)) if r.col[b] > 0])
            r_up = double_oracle(len(c0), len(c1), evaluate_up, self.bot.iterations, self.bot.tolerance, start=support)
            return r_up.value if r_up.converged else r.value
        finally:
            if handles:
                self.engine.release(handles)

    def three_decision(self, base: Answer | None, stage: int, k: int, child_k: int, rounds: int) -> Answer | None:
        if base is None or self.quick_policy is None:
            return None
        check = self.clock.check_soft
        value = GuardedValue(self.bot.value, check, VALUE_CHUNK, self.clock.work, step=self.clock.note_step)
        eng = Guarded(self.engine, check, work=self.clock.work, step=self.clock.note_step)
        lists, row, col = base.lists, base.row, base.col
        evaluate = self._leaf_values(base, value, eng)
        n0, n1 = len(lists[0]), len(lists[1])
        result, refused = None, 0
        for _ in range(rounds):
            ranked = sorted(((row[i] * col[j], i, j) for i in range(n0) if row[i] > 0 for j in range(n1) if col[j] > 0),
                            key=lambda t: (-t[0], t[1], t[2]))[:k]
            todo = [(lists[0][i].id, lists[1][j].id) for _, i, j in ranked]
            todo = [ids for ids in todo if ids not in self.depth3 and ids not in self.unbranchable3]
            if not todo and result is not None:
                break
            for ids in todo:
                v = self.branch_depth3(ids, child_k, value, eng)
                if v is None:
                    self.unbranchable3.add(ids)
                    refused += 1
                else:
                    self.depth3[ids] = v
            support = ([i for i in range(n0) if row[i] > 0], [j for j in range(n1) if col[j] > 0])
            result = double_oracle(n0, n1, evaluate, self.bot.iterations, self.bot.tolerance, start=support)
            if not result.converged:
                return None  # never store an uncertified answer
            row, col = result.row, result.col
        assert result is not None
        pairs = len(self.depth3)
        info = {"rounds": result.rounds, "converged": True, "depth3_pairs": pairs, "depth2_pairs": len(self.depth2),
                "unbranchable": refused, "child_previews": self.child_previews, "k": k, "child_k": child_k}
        return Answer(stage, "three-decision", lists, row, col, result.value, view=base.view, pairs=pairs, info=info)


Stage = tuple[int, Callable[[Answer | None], "Answer | None"], bool]  # (number, stage, does a lock-in stop it)


@dataclass
class Outcome:
    best: Answer | None              # the answer to play (None: nothing finished: P2 holds with Continue)
    stopped: str                     # a Stopped `why`, "think-cap" or "done"
    trace: list[dict[str, Any]]      # per stage tried: stage, name, ms, and how it ended
    production: Answer | None        # the production search's answer, if it finished
    path: str = "full"               # one of PATHS: what produced `best`


def run(stages: list[Stage], clock: Clock) -> Outcome:
    """Run `stages` in order, keeping the last one that finished with an answer. A stop ends the
    run; a stage stopped part-way changes nothing. A stage whose flag is False (quick, production)
    runs whatever the soft stop says: only a hard stop ends it, or keeps it from starting."""
    best: Answer | None = None
    production: Answer | None = None
    trace: list[dict[str, Any]] = []
    stopped = "done"
    for n, stage, soft in stages:
        why = clock.soft() if soft else clock.hard()
        if why:
            stopped = why
            break
        t = time.perf_counter()
        clock.work.update(new_work())  # in place: the stage's guards hold this dict
        try:
            ans = stage(best)
        except Stopped as e:  # how far the stage got before the stop: its work so far
            trace.append({"stage": n, "ms": round((time.perf_counter() - t) * 1000, 1), "stopped": e.why,
                          "work": dict(clock.work)})
            stopped = e.why
            break
        ms = round((time.perf_counter() - t) * 1000, 1)
        if ans is None:
            trace.append({"stage": n, "ms": ms, "kept": False, "work": dict(clock.work)})
            continue
        ans.ms = ms
        trace.append({"stage": ans.stage, "name": ans.name, "ms": ms, "kept": True, "pairs": ans.pairs,
                      "value": round(ans.value, 6), "n_candidates": [len(ans.lists[0]), len(ans.lists[1])],
                      "work": dict(clock.work)})
        best = ans
        if ans.name == "production":
            production = ans
    return Outcome(best, stopped, trace, production)


def think(bot: MatrixBot, obs: Observation, seat: int, engine: Engine, clock: Clock, *, deep: bool = True,
          quick_policy: Any = None, floor: bool = False) -> Outcome:
    """`bot`'s answer for `seat`: production's search, then, with `deep`, deeper stages until the clock
    stops it or every stage is done. If a hard stop cuts the production search (a turn timer's deadline),
    its best-so-far round is played (`partial-full`). With nothing solved by the clock's `emergency_at`, the
    quick search gets the time left until the deadline (`quick`); if even that is cut, nothing (`continue`).
    `quick_policy` (a pinned yomibench.quick_policy.QuickPolicy, or None: off) as the module doc says. With
    `floor`, on a decision the clock can cut, the quick-menu floor first (module doc): played (`floor`) when the
    production search neither finished nor has a converged best-so-far."""
    d = Deepener(bot, obs, seat, engine, clock, quick_policy=quick_policy)
    trace: list[dict[str, Any]] = []
    if floor and (clock.hard_at is not None or clock.lock_cap is not None):
        f = run([(0, lambda _b: d.run_floor(), False)], clock)
        trace = f.trace
        if f.best is None:  # a hard stop cut the floor: the production search would not start either
            out = Outcome(None, f.stopped, trace, None, "continue")
            return out
    out = run(d.stages(deep=deep), clock)
    out.trace = trace + out.trace
    if out.best is not None:
        return out  # full: the production search finished (and maybe deeper stages after it)
    if d.partial is not None and (d.floor is None or d.partial.info.get("converged")):
        d.partial.info["rounds_seen"] = d.partial_rounds
        out.best, out.path = d.partial, "partial-full"
        return out
    if d.floor is not None:  # only a converged round replaces the floor: never a 1-2 round stub
        d.floor.info["rounds_seen"] = d.partial_rounds
        out.best, out.path = d.floor, "floor"
        return out
    if out.stopped != "emergency":  # cut (deadline, superseded) before the emergency point: nothing to play
        out.path = "continue"
        return out
    q = run([(0, lambda _b: d.quick(), False)], clock)
    out.trace += q.trace
    out.best, out.path = q.best, ("quick" if q.best is not None else "continue")
    if q.best is None:
        out.stopped = q.stopped
    return out

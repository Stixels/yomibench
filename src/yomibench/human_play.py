"""Human-vs-bot play: the bot server the game's YomiHuman mod (mod/YomiBench/YomiHuman.gd) talks to.

A person plays P1 in the real game's UI; a yomibench bot plays P2. The bot never sees the real
game's state. The server holds an engine match that follows the real one:

- `match_start`: the mod reports the match seed, both characters and the settings; the server
  starts the same match on the sim (`new_match`), with the bot's candidate set.
- `decision`: at every decision the mod sends the inputs both players actually gave last turn
  (P1's are often off the candidate grid) and a compact snapshot of the real state. The server
  plays the inputs on the sim (`step_raw`), compares the sim's state with the snapshot (any
  difference marks the game desynced: logged, play goes on, the game is flagged invalid), runs the
  bot on the sim's position and answers with its pick's `{action, data, extra}`, which the mod
  submits for P2 through the game's own action path.
- `end` (or `abort`): last turn's inputs, the final snapshot and the real game's winner. The server
  checks the end against the sim, writes the game's turn log and appends one row to
  `ledger-human-play.jsonl`.

Messages are JSON lines over TCP on 127.0.0.1, one reply per request, in order (see
docs/PLAYING.md). Nothing here records a player name: the human seat is "human".

No turn timer (timer off, or the standard settings): the bot's full search always finishes; nothing
here cuts it short.

Turn timer: when the match has one (`timer_mode` not "none"; never in singleplayer's default
settings), every decision is answered before it runs out, in every search mode: by the decision's
time less max(2 s, 15%) (configurable). The time goes to the full search first; if the deadline cuts
it, its best-so-far mix is played, and production's quick turn only if the full search solved nothing
by an emergency point (yomibench.anytime). The timer settings are the game's clock, not the
simulation: they are never sent to the sim.

Each decision's log line says what produced the bot's move (`bot.path`: full, partial-full, quick or
continue), with its think time and the clock it left (`bot.clock`); the result line counts the paths.

Think until lock-in (option `think_until_lockin`, default on; the panel's "Bot uses your thinking time"): the bot
finishes its full search, then keeps deepening it (yomibench.anytime) until the human locks in, or `think_cap`
seconds (default 60) after the decision opened, whichever is first. The mod tells the server only
THAT the human locked in, with a one-way `{"type": "locked_in", "turn": N}` (no reply); a timer's
`deadline` likewise. Those carry only type and turn, and anything more is refused whole.

The after-lock cap (no turn timer, since private build v5; `lock_cap`, default 10 s): once the human locks in, the
bot has at most `lock_cap` seconds more. A search that finishes sooner answers at once; one the cap cuts plays its
best so far (`partial-full`). The mod sends `locked_in` in every untimed match for this, whatever the option says.
Timed decisions keep their turn deadline only, except on a chess clock (ranked, since g15's build): there the bot
spends its bank about as the human spends theirs. While its bank has room above the reserve (`bank_reserve`, 60 s)
it thinks until the human locks in, then at most `lock_cap` more (or less if its searches end), never past (bank
left - reserve); with less room it plays on the floor, the per-turn deadline above (anytime.bank_budget). The
production search may use that whole room; the optional deeper stages after it stop at the turn's share of the room,
(bank left - reserve) / `bank_deep_turns` (anytime.bank_deep_s; 840 s of room: 8.4 s), since the game charges the
bot's clock while it waits on the human. The bank is the game's own P2 clock as the mod reads it at the turn's open
(`bot_clock_s` in the decision), else the bot's model of it. Each decision logs `bank_left_s`, `budget_s`, `deep_s` and
`mode` (bank | floor) in its `clock`.

A forfeit (the game's pause menu): the mod's `end` carries `forfeit: 0` (P1, the human, forfeited) and the winner;
the game is recorded as finished (`reason` "forfeit"), the sim's position as it was (its match never ended).

Per game the server writes `<out>/games/<game id>.jsonl`: a header, one line per decision (the sim
state, the sync check, the bot's strategy over its candidates and its value, its pick, both actual
inputs), and the result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import queue
import random
import re
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from yomibench import anytime, compat, ledger
from yomibench.engine import Engine, EngineError, Observation, connect

PROTOCOL = 1
CHARACTERS = ("Ninja", "Cowboy", "Wizard", "Mutant", "Robot")  # the five the sim plays
BOT_SEAT = 1  # the bot plays P2; the human P1
DEFAULT_PORT = 7790
# Human-play games are a held-out human test: never trained on. Every game
# log's header and every ledger row carries this arm, and the out folder carries compat's marker.
ARM = compat.HUMAN_PLAY_ARM

# The fixed bots human play offers (tools/human_play.py --bot). `{ckpt}` is the checkpoint path.
# Besides these: `current-best` (the round's promoted net, from the hub's r3-best.json) and `gen-N`
# (promoted generation N, r3-g<N>.npz), resolved by resolve_bot.
BOTS: dict[str, tuple[str, int]] = {
    "matrix-do-s4": ("matrix-do", 4),
    "ckpt2b-s2": ("matrix-do:{ckpt}@2", 2),
    "ckpt2b-s4": ("matrix-do:{ckpt}@4", 4),
}
# Where current-best / gen-N look up the best net (a best.json URL or path): $YOMIBENCH_BEST, else the local
# config's `[hub] best_url` (yomibench.localconfig), else a best.json in ~/.yomibench (a one-PC training loop
# writes it).
from yomibench import localconfig as _localconfig  # noqa: E402

DEFAULT_BEST = _localconfig.load().best_url or str(Path.home() / ".yomibench" / "best.json")
PRODUCTION_SET = 6  # the candidate set production and gates play: set 6 (set 5 with all four switches, plus the
# expand-only toggle pairs and clicks), rules 3b8bb1b76cd8; the simulator must be a set-6 build (f9cd2f1 or later)
GEN = re.compile(r"^gen-(\d+)$")
# How the bot turns its solved mixed strategy into one move: "mixed" (the default, and the panel's recommended style)
# samples it (the equilibrium play: unpredictable, so it can't be read and punished), "strongest" plays its most
# likely move, except Burst against Continue/DI (BURST_ACTIONS; pick_index).
PICK_STYLES = ("mixed", "strongest")
# The defensive bursts (the game's Burst out of hitstun, as candidates name it). In a simultaneous game Burst against
# Continue/DI is a mixed choice at equilibrium (a predictable Burst gets parried), so "strongest" never takes the
# argmax between them: with any weight on Burst it plays Burst with exactly its equilibrium probability. Argmax
# starved Burst to 0% whenever one DI direction outweighed it (a lost game in the g16 human audit: 0 bursts in 166
# turns, holding one, with Burst at 15-39% on eight turns of the final combo).
BURST_ACTIONS = frozenset({"Burst", "DefensiveBurst"})
# Match settings presets (the panel's "Settings"): overrides of the standard online lobby's settings
# (the sim's hello `match_settings`). The mod applies the chosen one to the real match and reports the
# match's settings; the server starts the sim with whatever differs from standard (Engine.new_match's
# `settings`), and a setting the sim refuses is refused clearly.
SETTINGS_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {},
    # The game's ranked preset (stage_width is the half-width: walls at x = +-600), with ranked
    # lobbies' ceiling at 500 (an engine that can't simulate it refuses the preset, and the
    # launcher then offers standard only). Its timer is every ranked replay's too: a chess clock, a
    # 15-minute bank, at least 15 s a turn. Timer settings are the game's clock, not the simulation: the sim
    # never gets them (TIMER_KEYS).
    "ranked": {"stage_width": 600, "game_length": 3000, "starting_meter": "1.5",
               "ceiling_height": 500, "has_ceiling": True,
               "timer_mode": "chess", "turn_time": 15, "turn_min_length": 15},
}
TIMER_KEYS = anytime.TIMER_KEYS
# What a decision message may carry. Moves are simultaneous: the human's input for this turn only
# ever arrives with the next decision (`inputs`, last turn's), after the bot has answered.
# bot_clock_s: the bot's own clock (the game's P2 turn timer) as the turn opened; never anything of the human's.
DECISION_KEYS = frozenset({"type", "turn", "inputs", "snapshot", "p2_fallback", "bot_clock_s"})
# One-way messages (no reply, so the mod's reply order is untouched): the human locked in this
# turn's move, or the turn timer is about to run out. They carry only `type` and `turn` — never
# what the human picked — and anything more is refused whole (think until lock-in: yomibench.anytime).
NOTIFY_TYPES = ("locked_in", "deadline")
NOTIFY_KEYS = frozenset({"type", "turn"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# What the sync check compares (the mod's snapshot has exactly these). Velocities are the exact
# fixed-point strings (`detail` states), so equal states compare equal after JSON.
PLAYER_FIELDS = ("hp", "x", "y", "vx", "vy", "facing", "state", "state_tick", "hitlag", "busy_interrupt",
                 "grounded", "meter", "meter_partial", "burst", "burst_partial", "combo", "feints", "air_options")
OBJECT_FIELDS = ("kind", "owner", "x", "y", "state")


@dataclass
class BotChoice:
    """What `--bot` resolved to: the make_bot spec, its candidate set, the label logs and the
    ledger use, and for a hub net its generation, sha256 and where it came from."""
    spec: str
    candidate_set: int
    label: str
    model: dict[str, Any] | None = field(default=None)


def resolve_bot(label: str, ckpt: str | None = None, *, best: str = DEFAULT_BEST, models: Path | str | None = None,
                sha256: str | None = None, candidate_set: int | None = None,
                fetch: Callable[[str], bytes] | None = None, log: Callable[[str], None] = print) -> BotChoice:
    """A BOTS label, `current-best`, `gen-N`, or a raw bot spec with `@N` naming its set (default 4).

    `current-best` reads `best` (the hub's r3-best.json: a URL or a local path), downloads its model
    into `models` (default ~/.yomibench/models, as <sha256>.npz) unless already there, and checks the
    sha256 against r3-best.json. `gen-N` takes r3-g<N>.npz from the same place; its sha256 comes from
    r3-best.json when N is the current generation, else from the trainer ledger's promotion row
    (r3-ledger-trainer.jsonl beside r3-best.json), or `sha256` given explicitly. A file whose sha256
    differs is refused, never loaded. Hub nets play the production search (`matrix-do`, its default
    settings: double oracle, factor refinement) on candidate set `candidate_set` (default PRODUCTION_SET)."""
    if label in BOTS:
        spec, cset = BOTS[label]
        if "{ckpt}" in spec:
            if not ckpt:
                raise ValueError(f"bot {label} needs a checkpoint (--ckpt PATH/ckpt2b.npz)")
            spec = spec.format(ckpt=ckpt)
        return BotChoice(spec, cset, label)
    m = GEN.match(label)
    if label == "current-best" or m:
        model = hub_model(best, int(m.group(1)) if m else None, models, sha256, fetch=fetch, log=log)
        cset = candidate_set or PRODUCTION_SET
        return BotChoice(f"matrix-do:{model['path']}", cset, f"r3-g{model['generation']}", {**model, "asked": label})
    tail = label.rsplit("@", 1)[-1] if "@" in label else ""
    return BotChoice(label, int(tail) if tail.isdigit() else 4, label)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (a URL the player configured)
        return r.read()


def _read(source: str, name: str | None, fetch: Callable[[str], bytes]) -> bytes:
    """`source` itself (name None) or the file `name` beside it: a URL on the hub or a local path."""
    if re.match(r"^https?://", source):
        return fetch(source if name is None else urllib.parse.urljoin(source, name))
    path = Path(source) if name is None else Path(source).parent / name
    return path.read_bytes()


def hub_model(best: str, generation: int | None, models: Path | str | None = None, sha256: str | None = None, *,
              fetch: Callable[[str], bytes] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """A promoted net, sha256-checked, as {path, generation, sha256, source}: the current best
    (`generation` None) or generation N (see resolve_bot). If the hub can't be reached, the last
    r3-best.json seen (kept in `models`) stands in for it, with a note."""
    fetch = fetch or _fetch
    folder = Path(models) if models else Path.home() / ".yomibench" / "models"
    folder.mkdir(parents=True, exist_ok=True)
    last = folder / "last-r3-best.json"
    try:
        cur = json.loads(_read(best, None, fetch))
        last.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    except (OSError, ValueError) as e:
        if not last.exists():
            raise ValueError(f"can't read {best} ({e}) and no earlier copy in {folder}") from e
        cur = json.loads(last.read_text(encoding="utf-8"))
        log(f"note: {best} unreachable ({e}); using the last copy seen (generation {cur.get('generation')})")
    gen = int(cur["generation"]) if generation is None else generation
    name = cur["model"] if generation is None else f"r3-g{gen}.npz"
    want = sha256 or (cur["sha256"] if gen == int(cur["generation"]) else None)
    if want is None:
        want = promoted_sha256(best, gen, fetch)
    if not want or not SHA256.match(want):
        raise ValueError(f"no sha256 for generation {gen}: not the current best, and no promotion row for it in "
                         f"the trainer ledger beside {best} (pass --sha256)")
    path = folder / f"{want}.npz"
    if path.exists() and _sha256_file(path) == want:
        return {"path": str(path), "generation": gen, "sha256": want, "source": "cache", "model": name}
    try:
        data = _read(best, name, fetch)
    except OSError as e:
        raise ValueError(f"{name} not found beside {best}: {e}") from e
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ValueError(f"{name}: sha256 {got[:12]} differs from the promoted {want[:12]}; refusing to load it")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    log(f"downloaded {name} (generation {gen}, sha256 {want[:12]}) to {path}")
    return {"path": str(path), "generation": gen, "sha256": want, "source": best, "model": name}


def promoted_sha256(best: str, generation: int, fetch: Callable[[str], bytes]) -> str | None:
    """Generation N's sha256 from the trainer ledger (r3-ledger-trainer.jsonl beside r3-best.json):
    the last gate row that promoted it, whose candidate model is stored as <sha256>.npz."""
    try:
        text = _read(best, "r3-ledger-trainer.jsonl", fetch).decode("utf-8")
    except OSError:
        return None
    found = None
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        tags, gate = row.get("tags") or {}, (row.get("metrics") or {}).get("gate") or {}
        if gate.get("decision") != "promote" or str(tags.get("generation")) != str(generation):
            continue
        entries = list(((row.get("provenance") or {}).get("models") or {}).values())
        cand = str(tags.get("candidate") or "")
        named = [e for e in entries if cand and e.get("name") and (cand.endswith(e["name"]) or e["name"].endswith(cand))]
        for e in named[:1] or entries[:1]:  # the candidate (pool_eval lists it first)
            stem = Path(str(e.get("path", "")).replace("\\", "/")).stem
            if SHA256.match(stem):
                found = stem
    return found


def local_model(path: Path | str, generation: int, sha256: str, *, candidate_set: int | None = None,
                source: str = "bundled") -> BotChoice:
    """A promoted net from a local file whose sha256 is already known (the Windows app's bundled net):
    no hub, no network. Played like `gen-N`: the production search on candidate set `candidate_set`
    (default PRODUCTION_SET). A file whose sha256 differs is refused, never loaded."""
    path = Path(path)
    if not SHA256.match(sha256):
        raise ValueError(f"not a sha256: {sha256!r}")
    got = _sha256_file(path)
    if got != sha256:
        raise ValueError(f"{path}: sha256 {got[:12]} differs from the expected {sha256[:12]}; refusing to load it")
    model = {"path": str(path), "generation": int(generation), "sha256": sha256, "source": source,
             "model": path.name, "asked": source}
    return BotChoice(f"matrix-do:{path}", candidate_set or PRODUCTION_SET, f"r3-g{int(generation)}", model)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_of(state: dict[str, Any]) -> dict[str, Any]:
    """The compact snapshot of a protocol `state` (a `detail` state for exact velocities): what the
    mod sends of the real game. Used for the sim's side of the sync check, and by tests."""
    return {
        "tick": state["tick"],
        "players": [{k: p.get(k) for k in PLAYER_FIELDS} for p in state["players"]],
        "objects": [{k: o.get(k) for k in OBJECT_FIELDS} for o in state.get("objects", [])],
    }


def sync_diff(sim: dict[str, Any], real: dict[str, Any]) -> list[str]:
    """Differences between two snapshots, as `path: sim X real Y` (empty when in sync). Numbers
    compare by value (JSON may turn 3 into 3.0); a field the real snapshot lacks is not compared."""
    out: list[str] = []

    def same(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b and type(a) is type(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
        return a == b

    if not same(sim.get("tick"), real.get("tick")):
        out.append(f"tick: sim {sim.get('tick')} real {real.get('tick')}")
    for i, (sp, rp) in enumerate(zip(sim.get("players", []), real.get("players", []))):
        for k in PLAYER_FIELDS:
            if k in rp and not same(sp.get(k), rp[k]):
                out.append(f"p{i + 1}.{k}: sim {sp.get(k)!r} real {rp[k]!r}")
    so, ro = sim.get("objects", []), real.get("objects")
    if ro is not None:
        if len(so) != len(ro):
            out.append(f"objects: sim {len(so)} real {len(ro)}")
        for i, (a, b) in enumerate(zip(so, ro)):
            for k in OBJECT_FIELDS:
                if k in b and not same(a.get(k), b[k]):
                    out.append(f"objects[{i}].{k}: sim {a.get(k)!r} real {b[k]!r}")
    return out


def _same_input(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Equal inputs, up to JSON's int/float (the mod's JSON numbers come back as floats) and the
    text "[Object:null]", JSON's print of a null Object, which the mod submits as null."""
    def norm(v: Any) -> Any:
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if v == "[Object:null]":
            return None
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v
    if a is None or b is None:
        return a is b
    keys = ("action", "data", "extra")
    return norm({k: a.get(k) for k in keys}) == norm({k: b.get(k) for k in keys})


def _round(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


def _argmax(strategy: list[float], among: list[int]) -> int:
    return max(among, key=lambda k: (strategy[k], -k))  # the first of equals


def burst_mass(strategy: list[float], actions: list[str] | None) -> float:
    """The strategy's total weight on a defensive Burst (BURST_ACTIONS); 0 without actions."""
    if not actions:
        return 0.0
    return sum(p for p, a in zip(strategy, actions) if p > 0 and a in BURST_ACTIONS)


def pick_index(strategy: list[float], style: str, rng: random.Random, actions: list[str] | None = None) -> int:
    """The move a strategy plays: sampled ("mixed") or its most likely one ("strongest"; the first of equals).

    "strongest" with weight on a defensive Burst (`actions`, the candidates' action names): Burst against the rest is
    drawn from the equilibrium, Burst with probability = its total weight, then the most likely move within the side
    drawn. So "strongest" bursts as often as the equilibrium does, never 0% (BURST_ACTIONS)."""
    if style == "strongest":
        q = burst_mass(strategy, actions)
        if q > 0:
            assert actions is not None
            bursts = [k for k, a in enumerate(actions) if a in BURST_ACTIONS and strategy[k] > 0]
            rest = [k for k in range(len(strategy)) if k not in bursts and strategy[k] > 0]
            if not rest or rng.random() < q:
                return _argmax(strategy, bursts)
            return _argmax(strategy, rest)
        return _argmax(strategy, list(range(len(strategy))))
    (i,) = rng.choices(range(len(strategy)), weights=strategy)
    return i


def _sim_part(settings: dict[str, Any]) -> dict[str, Any]:
    """Match settings without the turn timer's (the game's clock, which the sim doesn't simulate)."""
    return {k: v for k, v in settings.items() if k not in TIMER_KEYS}


def _matrix_info(strategy: list[float], value: float, lst: list[Any], n_base: int, style: str,
                 rng: random.Random, oracle: dict[str, Any] | None) -> dict[str, Any]:
    """A matrix bot's answer as the turn log keeps it: its support over its candidate list `lst`,
    its value, and the pick drawn from `rng` in `style`."""
    actions = [c.action for c in lst]
    i = pick_index(strategy, style, rng, actions)
    support = [{"i": k, "p": _round(p), **lst[k].record()} for k, p in enumerate(strategy) if p > 0]
    info: dict[str, Any] = {"n_candidates": len(lst), "n_base": n_base, "strategy": support,
                            "value": _round(value), "pick": lst[i].record(), "pick_index": i,
                            "pick_style": style}
    if style == "strongest" and burst_mass(strategy, actions) > 0:
        info["pick_rule"] = "burst-mixed"  # Burst against the rest drawn from the equilibrium (pick_index)
    if oracle is not None:
        info["oracle"] = oracle
    return info


def valid_notification(msg: Any) -> bool:
    """A `locked_in` / `deadline` message carries exactly its type and the decision's turn (a
    whole number): anything more could carry the human's pick."""
    if not isinstance(msg, dict) or set(msg) != NOTIFY_KEYS or msg.get("type") not in NOTIFY_TYPES:
        return False
    t = msg["turn"]
    return not isinstance(t, bool) and isinstance(t, (int, float)) and float(t).is_integer() and t >= 0


class Game:
    """One human-vs-bot game on the server: the sim match, the sync state and the turn log."""

    def __init__(self, gid: str, p1: str, p2: str, seed: int, path: Path):
        self.id, self.characters, self.seed, self.path = gid, (p1, p2), seed, path
        self.started_utc = ledger.utc_now()
        self.t0 = time.time()
        self.obs: Observation | None = None
        self.turns = 0          # decisions played (the mod's turn count)
        self.desync_turns: list[int] = []
        self.first_desync: dict[str, Any] | None = None
        self.sim_error: str | None = None  # the sim could not follow (it stopped at the last good turn)
        self.p2_overrides = 0   # P2's actual input was not the bot's pick
        self.p2_fallbacks = 0   # the mod played Continue instead of an unusable pick (desynced games)
        self.pending: dict[str, Any] | None = None  # this decision's log line, written once its inputs are known
        self.pick: dict[str, Any] | None = None
        self.think_ms: list[float] = []
        self.options: dict[str, Any] = {}
        self.match_settings: dict[str, Any] = {}  # what differs from the standard settings
        self.preset: str | None = None
        self.settings: dict[str, Any] = {}  # the real match's settings, as the mod sent them
        self.timer: anytime.GameTimer | None = None  # the bot's turn clock (None: no turn timer)
        self.turn_s: float | None = None  # one turn of the match's turn timer: every answer's ceiling
        self.deadline_hits = 0  # decisions whose deadline cut the production search
        self.lock_cap_hits = 0  # untimed decisions whose after-lock cap cut the production search
        self.untimed_locked = 0  # untimed stoppable decisions the human locked in during (the cap could apply)
        self.paths: dict[str, int] = {}  # what produced each searched move (anytime.PATHS): count per path
        self.file = open(path, "w", encoding="utf-8", newline="\n")

    @property
    def desynced(self) -> bool:
        return bool(self.desync_turns) or self.sim_error is not None

    def write(self, line: dict[str, Any]) -> None:
        self.file.write(json.dumps(line, separators=(",", ":"), allow_nan=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


class Session:
    """Answers the mod's messages for one bot. `handle(msg) -> reply`; transport-free, so tests feed
    it fake mod messages. `engine_factory()` returns a fresh sim Engine for each match."""

    def __init__(self, bot_spec: str, candidate_set: int, out_dir: Path | str, *, label: str | None = None,
                 engine_factory: Callable[[], Engine] | None = None, ledger_path: Path | str | None = None,
                 schedule: list[tuple[str, str, int]] | None = None, mode: str = "human", scene: str | None = None,
                 model: dict[str, Any] | None = None, pick: str = "mixed", show_estimate: bool = False,
                 settings: str = "standard", think_until_lockin: bool = True, turn_timer: bool | None = None,
                 think_cap: float = anytime.THINK_CAP_S, lock_cap: float = anytime.LOCK_CAP_S,
                 bank_reserve: float = anytime.BANK_RESERVE_S,
                 bank_deep_turns: float = anytime.BANK_DEEP_TURNS,
                 deadline_margin: float = anytime.DEADLINE_MARGIN_S,
                 deadline_fraction: float = anytime.DEADLINE_FRACTION, log: Callable[[str], None] = print,
                 engine_spec: str | None = None, bot_factory: Callable[[str], Any] | None = None):
        """`bot_factory(bot_spec)` builds the bot (default yomibench.bots.make_bot): a build that plays a search
        variant of its own (e.g. a candidate set's extra pass) passes its own."""
        from yomibench.bots import make_bot

        self.bot_spec, self.candidate_set = bot_spec, candidate_set
        if deadline_margin < 0 or not 0 <= deadline_fraction < 1:
            raise ValueError(f"deadline margin must be >= 0 s and its fraction in [0, 1): {deadline_margin}, "
                             f"{deadline_fraction}")
        self.deadline_margin, self.deadline_fraction = deadline_margin, deadline_fraction
        if not think_cap > 0:
            raise ValueError(f"the think cap must be > 0 s: {think_cap}")
        self.think_cap = float(think_cap)
        if not lock_cap > 0:
            raise ValueError(f"the after-lock cap must be > 0 s: {lock_cap}")
        self.lock_cap = float(lock_cap)
        if not bank_reserve >= 0:
            raise ValueError(f"the bank reserve must be >= 0 s: {bank_reserve}")
        self.bank_reserve = float(bank_reserve)
        if not bank_deep_turns > 0:
            raise ValueError(f"bank deep turns must be > 0: {bank_deep_turns}")
        self.bank_deep_turns = float(bank_deep_turns)
        self.bot = (bot_factory or make_bot)(bot_spec)
        if getattr(self.bot, "playout_cap", None):  # human play always searches full turns
            raise ValueError(f"{bot_spec}: human play searches every turn in full; drop the @cheap option")
        self.label = label or self.bot.name
        self.model = model  # a hub net's generation, sha256 and source (resolve_bot), for logs and rows
        self.out = Path(out_dir)
        (self.out / "games").mkdir(parents=True, exist_ok=True)
        compat.mark_held_out(self.out, ARM)  # training refuses this folder (and any folder under it)
        self.ledger_path = Path(ledger_path) if ledger_path else self.out / "ledger-human-play.jsonl"
        sim = "sim" if scene is None else f"sim:{scene}"
        # The bot's engine: a simulator plugin (sim) or the player's own game, N headless copies (game-pool:N|auto,
        # option D). Default: the simulator when it's installed, else the game.
        self.engine_spec = engine_spec or (sim if _have_yomisim() else "game-pool:auto")
        spec = sim if self.engine_spec == "sim" else self.engine_spec
        self.engine_factory = engine_factory or (lambda: connect(spec, candidate_set))
        self.engine = self.engine_factory()
        self.hello = self.engine.hello()
        self.settings = self.hello.get("match_settings") or {}
        offered = self.hello.get("characters")  # the sim build's (all five on current builds)
        self.characters = [c for c in CHARACTERS if offered is None or c in offered]
        # Each preset, tried on the sim once: the panel offers only those it plays.
        self.presets, self.presets_refused = {}, {}
        for name, over in SETTINGS_PRESETS.items():
            err = self._try_settings(self.settings_like(_sim_part(over)))
            if err is None:
                self.presets[name] = dict(over)
            else:
                self.presets_refused[name] = err
        if settings not in SETTINGS_PRESETS:
            raise ValueError(f"settings preset must be one of {', '.join(SETTINGS_PRESETS)}, not {settings!r}")
        if settings in self.presets_refused:
            raise ValueError(f"this sim can't play the {settings} settings: {self.presets_refused[settings]}")
        self.options = self._options({"pick": pick, "show_estimate": show_estimate, "settings": settings,
                                      "think_until_lockin": think_until_lockin,
                                      # the panel's "Turn timer" default; unset: on for a preset with a timer (ranked)
                                      **({} if turn_timer is None else {"turn_timer": bool(turn_timer)})})
        self.schedule = list(schedule or [])  # autoplay: (p1, p2, seed) games the mod starts in order
        self.mode = mode
        self.log = log
        self.game: Game | None = None
        self.results: list[dict[str, Any]] = []  # one summary per finished or abandoned game
        from yomibench.search import SOLVER

        # Which regret-matching loop the search runs (native yomisolve, or the ~30-130x slower pure-Python loop);
        # every game header's bot_config names it too.
        self.solver = SOLVER
        log(f"bot search solver: {SOLVER}" + ("" if SOLVER == "native" else
                                               " (the native yomisolve extension isn't installed: slower searches)"))

    # --- protocol ---------------------------------------------------------------------------

    def handle(self, msg: Any, clock: anytime.Clock | None = None) -> dict[str, Any]:
        """One request's reply. `clock`: a decision's stop signals (the transport makes one per
        decision as it reads it, and signals it on `locked_in` / `deadline`); only a decision that
        thinks until lock-in reads it."""
        try:
            if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
                raise ValueError("a message is a JSON object with a string 'type'")
            if msg["type"] == "decision":
                return self._on_decision(msg, clock)
            fn = getattr(self, f"_on_{msg['type']}", None)
            if fn is None:
                raise ValueError(f"unknown message type {msg['type']!r}")
            return fn(msg)
        except Exception as e:  # noqa: BLE001  (a bad message must not kill the server; the mod shows it)
            self.log(f"error: {type(e).__name__}: {e}")
            return {"type": "error", "message": f"{type(e).__name__}: {e}"}

    def _on_hello(self, msg: dict[str, Any]) -> dict[str, Any]:
        if msg.get("protocol") != PROTOCOL:
            raise ValueError(f"the mod speaks protocol {msg.get('protocol')!r}, this server {PROTOCOL}")
        return {"type": "hello", "protocol": PROTOCOL, "bot": self.label, "candidate_set": self.candidate_set,
                "characters": list(self.characters), "mode": self.mode, "options": dict(self.options),
                "pick_styles": list(PICK_STYLES), "settings_presets": self.presets, "lock_cap_s": self.lock_cap,
                "settings_refused": self.presets_refused, "model": self.hello_model(), "next": self._next()}

    def hello_model(self) -> dict[str, Any] | None:
        """The net the bot plays, for the mod's status bar: its generation, a short sha256 (the hub's record of the
        net, `hub_sha256`, when the file is a trimmed copy of it) and where it came from (`source`). None for a bot
        that isn't a promoted net."""
        m = self.model or {}
        if not m.get("sha256"):
            return None
        return {"generation": m.get("generation"), "sha": str(m.get("hub_sha256") or m["sha256"])[:8],
                "source": m.get("source")}

    def settings_like(self, settings: dict[str, Any]) -> dict[str, Any]:
        """`settings` in the types of the sim's standard ones: JSON's 600.0 as 600 for an integer
        setting (the sim rejects a float there), a number as its decimal string for a setting the
        sim keeps as a string (starting_meter "1.5")."""
        out = {}
        for k, v in settings.items():
            std = self.settings.get(k)
            if isinstance(std, bool) or isinstance(v, bool):
                pass
            elif isinstance(std, int) and isinstance(v, float) and v.is_integer():
                v = int(v)
            elif isinstance(std, str) and isinstance(v, (int, float)):
                v = str(int(v)) if float(v).is_integer() else str(v)
            out[k] = v
        return out

    def _overrides(self, settings: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """The settings that differ from the sim's standard ones (the sim's types), and the keys
        the sim doesn't have (not simulated: logged only)."""
        typed = self.settings_like({k: v for k, v in _sim_part(settings).items() if k in self.settings})
        return ({k: v for k, v in typed.items() if v != self.settings[k]},
                sorted(k for k in settings if k not in self.settings and k not in TIMER_KEYS))

    def _timer_of(self, settings: dict[str, Any]) -> dict[str, Any]:
        """The timer settings, standard values filled in (the game's clock, not simulated)."""
        std = {k: self.settings[k] for k in TIMER_KEYS if k in self.settings}
        return self.settings_like({**std, **{k: v for k, v in settings.items() if k in TIMER_KEYS}})

    def _try_settings(self, overrides: dict[str, Any]) -> str | None:
        """None if the sim starts a match with `overrides`, else its refusal."""
        if not overrides:
            return None
        first = (self.characters or list(CHARACTERS))[0]
        try:
            self.engine.new_match(first, first, 0, settings=overrides)
            return None
        except EngineError as e:
            return str(e)

    def _options(self, given: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
        """Per-game options (the launcher's defaults, or the panel's in match_start), checked."""
        out = dict(base or {})
        if given is None:
            return out
        if not isinstance(given, dict):
            raise ValueError("options must be an object")
        for k, v in given.items():
            if k == "pick":
                if v not in PICK_STYLES:
                    raise ValueError(f"pick style must be one of {', '.join(PICK_STYLES)}, not {v!r}")
            elif k in ("show_estimate", "think_until_lockin", "turn_timer"):
                v = bool(v)
            elif k == "settings":
                if v not in self.presets:
                    raise ValueError(f"settings preset must be one of {', '.join(self.presets)}, not {v!r}")
            else:
                raise ValueError(f"unknown option {k!r}")
            out[k] = v
        return out

    def _next(self) -> dict[str, Any] | None:
        """Autoplay: the next game the mod should start (None: no more; the mod then quits)."""
        if not self.schedule:
            return None
        p1, p2, seed = self.schedule.pop(0)
        return {"p1": p1, "p2": p2, "seed": seed}

    def _on_match_start(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self.game is not None:
            self._finish(None, "abandoned", "a new match started")
        p1, p2, seed = msg["p1"], msg["p2"], int(msg["seed"])
        for c in (p1, p2):
            if c not in self.characters:
                raise ValueError(f"the bots play {', '.join(self.characters)}, not {c!r}")
        options = self._options(msg.get("options"), self.options)
        settings = msg.get("settings") or {}
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")
        # The real match's settings decide; the preset name is only the panel's label for them.
        overrides, unknown = self._overrides(settings)
        timer = self._timer_of(settings)
        # compared as overrides of standard, so a preset naming a standard value (has_ceiling true) still matches.
        # The label names the simulated settings: the turn timer is the panel's own switch ("timer" in the header).
        preset = next((n for n, o in self.presets.items() if self._overrides(o)[0] == overrides), None)
        gid = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{self.label}-{p1}-{p2}-{seed}"
        gid = "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in gid)
        if not getattr(self.engine, "reusable", False):
            self.engine.close()
            self.engine = self.engine_factory()  # a fresh sim per game: nothing carries over
        # (a game-server pool is kept: new_match resets every server, and restarting N games costs seconds)
        self.game = g = Game(gid, p1, p2, seed, self.out / "games" / f"{gid}.jsonl")
        g.options = options
        g.match_settings, g.preset = overrides, preset
        g.settings = dict(settings)
        g.timer = anytime.GameTimer.of(timer)
        # With a running timer, every answer beats one turn of it; without one there is no deadline at all.
        g.turn_s = anytime.turn_seconds(timer) if g.timer is not None else None
        refused = None
        try:
            g.obs = self.engine.new_match(p1, p2, seed, settings=overrides or None)
        except EngineError as e:  # the sim can't play this match: say so; the game can't count
            refused = f"the bot's simulator can't play these match settings {overrides}: {e}"
        g.write({"type": "header", "game": gid, "arm": ARM, "train": False, "bot": self.label,
                 "bot_spec": self.bot_spec, "model": self.model,
                 "bot_seat": "p2", "human_seat": "p1", "value_of": "the bot (P2)",
                 "candidate_set": self.candidate_set, "search": "full turns", "options": options,
                 "characters": [p1, p2], "seed": seed, "mode": self.mode, "started_utc": g.started_utc,
                 "match_settings": overrides, "settings_preset": preset, "settings_unknown": unknown,
                 "timer": timer if g.timer is not None else None,
                 "deadline": None if g.turn_s is None else {"turn_s": g.turn_s, "margin_s": self.deadline_margin,
                                                            "fraction": self.deadline_fraction},
                 "settings_refused": refused, "mod": msg.get("mod"),
                 "yomibench": ledger.git_info(),
                 "engine": ledger.engine_info({**self.hello, "match_settings": {**self.settings, **overrides}}),
                 "bot_config": self.bot.config() if hasattr(self.bot, "config") else None})
        if refused:
            g.sim_error = refused
            self.log(f"game {gid}: {refused}")
            return {"type": "match_started", "game": gid, "ok": False, "refused": True, "message": refused}
        self.log(f"game {gid}: started ({p1} vs bot {p2}, seed {seed}, {preset or overrides or 'standard'} settings)")
        return {"type": "match_started", "game": gid, "ok": True, "settings_preset": preset}

    def _apply_inputs(self, g: Game, msg: dict[str, Any]) -> None:
        """Play last turn's actual inputs on the sim and complete its log line."""
        inputs = msg.get("inputs")
        if g.pending is None:
            return
        line, g.pending = g.pending, None
        line["inputs"] = inputs
        if msg.get("p2_fallback"):
            g.p2_fallbacks += 1
            line["p2_fallback"] = True
        if inputs is None or len(inputs) != 2 or any(i is None for i in inputs):
            line["p2_matches_pick"] = None
            if g.sim_error is None:
                g.sim_error = f"turn {line['turn']}: inputs missing"
        else:
            line["p2_matches_pick"] = _same_input(inputs[1], g.pick)
            if not line["p2_matches_pick"]:
                g.p2_overrides += 1
            if g.sim_error is None:
                if g.obs is not None and g.obs.done:
                    g.sim_error = f"turn {line['turn']}: the sim's match is over but the real one goes on"
                else:
                    try:
                        g.obs = self.engine.step_raw(inputs)
                    except EngineError as e:
                        g.sim_error = f"turn {line['turn']}: step_raw refused the inputs: {e}"
        if g.sim_error and "sim_error" not in line:
            line["sim_error"] = g.sim_error
        g.write(line)

    def _check(self, g: Game, snap: dict[str, Any] | None, turn: int) -> dict[str, Any]:
        if g.sim_error is not None:
            return {"ok": False, "sim_error": g.sim_error}
        if not isinstance(snap, dict):
            return {"ok": False, "error": "no snapshot"}
        detail = self.engine._call({"cmd": "observe", "detail": True}).get("detail")
        diffs = sync_diff(snapshot_of(detail), snap)
        if diffs:
            g.desync_turns.append(turn)
            if g.first_desync is None:
                g.first_desync = {"turn": turn, "tick": snap.get("tick"), "diffs": diffs[:20]}
                self.log(f"game {g.id}: DESYNC at turn {turn} tick {snap.get('tick')}: {'; '.join(diffs[:5])}")
        return {"ok": not diffs, "diffs": diffs[:40]}

    def _on_decision(self, msg: dict[str, Any], clock: anytime.Clock | None = None) -> dict[str, Any]:
        extra = sorted(set(msg) - DECISION_KEYS)
        if extra:  # anything else could carry the human's pick for this turn: the bot must never see it
            raise ValueError(f"a decision carries only last turn's inputs and the snapshot, not {extra}")
        g = self._need_game()
        turn = int(msg.get("turn", g.turns))
        clock = clock or anytime.Clock(turn)
        bank = g.timer.available() if g.timer is not None else None  # the game's clock runs every turn
        reported = msg.get("bot_clock_s")
        if bank is not None and g.timer.mode != "default" and isinstance(reported, (int, float))                 and not isinstance(reported, bool) and reported >= 0:
            # The game's own P2 clock wins over the model (it also counts what the transport took); a chess turn
            # that opens under the floor is topped up to it.
            bank = float(reported)
            if g.timer.mode == "chess":
                bank = max(bank, g.timer.turn_min)
            g.timer.bank = bank
        plan = budget = deep = None
        if bank is not None and g.timer.mode == "chess" and g.turn_s is not None:
            plan, budget = anytime.bank_budget(bank, g.turn_s, self.bank_reserve, self.deadline_margin,
                                               self.deadline_fraction)
        if plan == "bank" and clock.hard_at is None:
            # The bank's room: the production search may use all of it (the hard stop), then the after-lock cap; never
            # past the reserve. The optional deeper stages stop at this turn's share of the room (bank_deep_s): the
            # game charges the bot's clock while it waits on the human, so thinking until every lock-in drained the
            # bank to the floor in every g16 audit game.
            available = budget
            deep = anytime.bank_deep_s(bank, self.bank_reserve, self.bank_deep_turns)
            clock.hard_at = clock.t0 + budget
            clock.emergency_at = clock.t0 + anytime.emergency_within(budget, budget)
            clock.lock_cap = self.lock_cap
            clock.cap_at = clock.t0 + min(budget, deep)
        else:
            # The time this answer has: one turn of the match's turn timer, or less if a bank has less left.
            available = min((t for t in (bank, g.turn_s) if t is not None), default=None)
            if available is not None and clock.hard_at is None:
                within = anytime.answer_within(available, self.deadline_margin, self.deadline_fraction)
                clock.hard_at = clock.t0 + within
                clock.emergency_at = clock.t0 + anytime.emergency_within(available, within)
                budget = within if plan == "floor" else budget
        if clock.cap_at is None:  # thinking until lock-in never runs past this, timer or not
            clock.cap_at = clock.t0 + self.think_cap
        if available is None:  # no turn timer: at most lock_cap seconds after the human locks in
            clock.lock_cap = self.lock_cap
        pending = g.pending
        try:
            return self._decide(g, msg, turn, clock, available)
        finally:
            took = time.monotonic() - clock.t0
            if g.timer is not None:
                g.timer.spend(took)
            bot = (g.pending or {}).get("bot") if g.pending is not pending else None
            if bot is not None:  # this decision's line: its think time and the clock it left
                bot["clock"] = {"available_s": None if available is None else round(available, 3),
                                "think_s": round(took, 3),
                                "left_s": None if available is None else round(available - took, 3),
                                "bank_s": None if g.timer is None else round(g.timer.bank, 3),
                                "bank_left_s": None if plan is None else round(bank, 3),
                                "budget_s": None if budget is None else round(budget, 3), "mode": plan,
                                "deep_s": None if deep is None else round(deep, 3)}

    def _decide(self, g: Game, msg: dict[str, Any], turn: int, clock: anytime.Clock,
                available: float | None) -> dict[str, Any]:
        self._apply_inputs(g, msg)
        sync = self._check(g, msg.get("snapshot"), turn)
        g.turns = turn + 1
        obs = g.obs
        line: dict[str, Any] = {"type": "decision", "turn": turn, "tick": (msg.get("snapshot") or {}).get("tick"),
                                "sync": sync, "real": msg.get("snapshot")}
        if g.sim_error is not None or obs is None or obs.done or not obs.candidates[BOT_SEAT]:
            # The sim can't say what P2 may do here: the mod plays the game's default for P2.
            g.pick = {"action": "Continue", "data": None, "extra": None}
            line.update({"state": obs.state if obs else None, "bot": None, "pick": g.pick})
            g.pending = line
            return {"type": "pick", "turn": turn, **g.pick, "fallback": True, "desynced": True}
        t = time.perf_counter()
        depth = None
        think = bool(g.options.get("think_until_lockin"))
        if ((think or available is not None or clock.lock_cap is not None) and len(obs.candidates[BOT_SEAT]) > 1
                and isinstance(self.bot, anytime.MatrixBot)):
            # A turn timer, the after-lock cap or thinking until lock-in: the stoppable search (with no deadline
            # or cap hit and no deeper stage finished, its answer is the plain search's bit for bit).
            bot_info, depth = self._anytime(g, obs, clock, deep=think, available=available)
        else:
            bot_info = self._think(g, obs)
            if bot_info.get("strategy") is not None:
                bot_info["path"] = "full"  # the plain search: no deadline, it always finishes
        g.think_ms.append((time.perf_counter() - t) * 1000)
        bot_info["think_ms"] = round(g.think_ms[-1], 1)
        if bot_info.get("path"):
            g.paths[bot_info["path"]] = g.paths.get(bot_info["path"], 0) + 1
        g.pick = {k: bot_info["pick"][k] for k in ("action", "data", "extra")}
        line.update({"sim_turn": obs.turn, "state": obs.state, "actionable": list(obs.actionable), "bot": bot_info})
        g.pending = line
        reply = {"type": "pick", "turn": turn, **g.pick, "desynced": g.desynced}
        if bot_info.get("fallback"):
            reply["fallback"] = True
        if depth is not None:
            reply["depth"] = depth  # how deep the bot got (the mod's bottom bar); nothing of its pick
        if g.options.get("show_estimate") and bot_info.get("value") is not None:
            # The bot's value for this position (its expected result, -1..1) as an expected score in
            # percent; the mod shows it once this turn has been played.
            reply["estimate"] = round(50 * (1 + bot_info["value"]))
        return reply

    def _think(self, g: Game, obs: Observation) -> dict[str, Any]:
        """The bot's strategy, value and sampled pick for P2 on the sim's position."""
        from yomibench.bots import MatrixBot, turn_candidates

        rng = random.Random(f"{g.seed}:{BOT_SEAT}:{obs.turn}")
        style = g.options.get("pick", "mixed")
        cands = obs.candidates[BOT_SEAT]
        if len(cands) == 1:  # no choice: nothing to search
            return {"n_candidates": 1, "strategy": [{"i": 0, "p": 1.0, **cands[0].record()}], "value": None,
                    "pick": cands[0].record(), "pick_index": 0}
        if isinstance(self.bot, MatrixBot):
            strategy, value = self.bot.search(obs, BOT_SEAT, self.engine)
            view = self.bot._cheap_view(obs, self.engine)
            lst = (view["candidates"] if view else turn_candidates(obs, self.engine, self.bot._key(BOT_SEAT)))[BOT_SEAT]
            return _matrix_info(strategy, value, lst, len(cands), style, rng, self.engine.turn_cache.get("oracle"))
        cid = self.bot.choose(obs, BOT_SEAT, self.engine, rng)
        lst = turn_candidates(obs, self.engine)[BOT_SEAT]
        i = next(k for k, c in enumerate(lst) if c.id == cid)
        return {"n_candidates": len(lst), "strategy": None, "value": None, "pick": lst[i].record(), "pick_index": i}

    # --- think until lock-in (yomibench.anytime) ---------------------------------------------

    def _anytime(self, g: Game, obs: Observation, clock: anytime.Clock, *, deep: bool,
                 available: float | None) -> tuple[dict[str, Any], dict[str, Any]]:
        """The bot's answer from yomibench.anytime. With a turn timer (`available` seconds on the
        bot's clock), the production search runs until the clock's hard stop at the deadline: its
        best-so-far mix if the deadline cuts it, the quick search only if it solved nothing by the
        emergency point. Without one, nothing cuts the production search. With `deep` (think until
        lock-in), deeper stages follow until the human locks in. Returns the log's bot info and the
        reply's `depth`."""
        out = anytime.think(self.bot, obs, BOT_SEAT, self.engine, clock, deep=deep)  # type: ignore[arg-type]
        style = g.options.get("pick", "mixed")
        cands = obs.candidates[BOT_SEAT]

        def rng() -> random.Random:  # the production pick's stream, for every answer
            return random.Random(f"{g.seed}:{BOT_SEAT}:{obs.turn}")

        lock_ms = None if clock.soft_at is None else round((clock.soft_at - clock.t0) * 1000, 1)
        best = out.best
        fallback = "continue" if best is None else ("quick" if best.name == "quick" else None)
        work = {k: sum(t.get("work", {}).get(k, 0) for t in out.trace) for k in anytime.new_work()}
        notes: dict[str, Any] = {"path": out.path, "stopped": out.stopped, "fallback": fallback, "trace": out.trace,
                                 "work": work,
                                 "lock_in_ms": lock_ms, "deep": deep,
                                 "deadline_s": None if clock.hard_at is None else round(clock.hard_at - clock.t0, 3),
                                 "lock_cap_s": clock.lock_cap,
                                 "timer_available_s": None if available is None else round(available, 3),
                                 "answered_s": round(time.monotonic() - clock.t0, 3)}
        played = {"partial-full": "the full search's best so far", "quick": "the emergency quick search"}
        if available is None and clock.soft_at is not None:
            g.untimed_locked += 1
        if out.production is None and clock.lock_cap is not None and out.stopped in ("lock-cap", "emergency"):
            g.lock_cap_hits += 1
            notes["lock_cut"] = True
            self.log(f"game {g.id}: turn {obs.turn}: the {clock.lock_cap:g} s after-lock cap cut the full search; "
                     f"played {played.get(out.path, 'Continue')}")
        elif out.production is None and out.stopped in ("deadline", "emergency"):
            g.deadline_hits += 1
            self.log(f"game {g.id}: turn {obs.turn}: the deadline ({notes['deadline_s']} s) cut the full search; "
                     f"played {played.get(out.path, 'Continue')}")
        if best is None:  # nothing finished by the deadline: P2 holds (Continue)
            info: dict[str, Any] = {"n_candidates": len(cands), "n_base": len(cands), "strategy": None, "value": None,
                                    "pick": cands[0].record(), "pick_index": 0, "pick_style": style, "fallback": True,
                                    "path": out.path}
            info["anytime"] = {"stage": None, **notes}
            return info, {"stage": None, "label": "out of time", "pairs": 0, "stopped": out.stopped}
        strategy, value = best.strategy(BOT_SEAT)
        info = _matrix_info(strategy, value, best.lists[BOT_SEAT], len(cands), style, rng(), best.info or None)
        info["path"] = out.path
        notes.update(stage=best.stage, name=best.name, pairs=best.pairs)
        prod = out.production
        if prod is not None and prod is not best:  # what the plain bot would have played, for comparison
            ps, pv = prod.strategy(BOT_SEAT)
            pi = _matrix_info(ps, pv, prod.lists[BOT_SEAT], len(cands), style, rng(), prod.info or None)
            notes["production"] = {k: pi[k] for k in ("n_candidates", "strategy", "value", "pick", "pick_index")}
        info["anytime"] = notes
        return info, {"stage": best.stage, "label": best.label(), "pairs": best.pairs, "stopped": out.stopped}

    def _on_end(self, msg: dict[str, Any]) -> dict[str, Any]:
        g = self._need_game()
        turn = int(msg.get("turn", g.turns))
        forfeit = msg.get("forfeit")
        if forfeit not in (None, 0, 1) or isinstance(forfeit, bool):
            raise ValueError(f"forfeit must be 0 (P1), 1 (P2) or absent, not {forfeit!r}")
        if forfeit is not None:
            # The game ended on a forfeit (its pause menu): the forfeiter's last input is the game's own Forfeit,
            # which the sim doesn't play. The sim's match stays where the last turn left it.
            winner = msg.get("winner", 1 - forfeit)
            if winner != 1 - forfeit:
                raise ValueError(f"a forfeit by {forfeit} is won by {1 - forfeit}, not {winner!r}")
            if g.pending is not None:
                g.pending["inputs"] = msg.get("inputs")
                g.pending["forfeit"] = forfeit
                g.write(g.pending)
                g.pending = None
            summary = self._finish(winner, "finished", "forfeit", final=msg.get("snapshot"), forfeit=forfeit)
            return {"type": "result", **{k: summary[k] for k in ("game", "result", "valid", "turns", "desync")},
                    "forfeit": forfeit, "log": str(g.path), "ledger": str(self.ledger_path), "next": self._next()}
        self._apply_inputs(g, msg)
        sync = self._check(g, msg.get("snapshot"), turn)
        if g.sim_error is None and g.obs is not None and not g.obs.done:
            g.desync_turns.append(turn)
            sync = {**sync, "ok": False, "end": "the real match ended, the sim's didn't"}
            if g.first_desync is None:
                g.first_desync = {"turn": turn, "tick": (msg.get("snapshot") or {}).get("tick"), "diffs": [sync["end"]]}
        winner = msg.get("winner")
        if winner not in (0, 1, None):
            raise ValueError(f"winner must be 0, 1 or null, not {winner!r}")
        summary = self._finish(winner, "finished", None, sync=sync, final=msg.get("snapshot"))
        return {"type": "result", **{k: summary[k] for k in ("game", "result", "valid", "turns", "desync")},
                "log": str(g.path), "ledger": str(self.ledger_path), "next": self._next()}

    def _on_abort(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self.game is None:
            return {"type": "aborted", "game": None, "next": self._next()}
        gid = self.game.id
        self._finish(None, "abandoned", str(msg.get("reason") or "aborted"))
        return {"type": "aborted", "game": gid, "next": self._next()}

    def _need_game(self) -> Game:
        if self.game is None:
            raise ValueError("no match in progress (send match_start first)")
        return self.game

    # --- results ----------------------------------------------------------------------------

    def connection_lost(self) -> None:
        """The mod went away (game closed or crashed): an unfinished game is abandoned."""
        if self.game is not None:
            self._finish(None, "abandoned", "the game disconnected")

    def _finish(self, winner: int | None, how: str, reason: str | None, sync: dict[str, Any] | None = None,
                final: dict[str, Any] | None = None, forfeit: int | None = None) -> dict[str, Any]:
        g = self.game
        assert g is not None
        self.game = None
        if g.pending is not None:  # a decision whose inputs never came
            g.pending["inputs"] = None
            g.write(g.pending)
            g.pending = None
        if how == "abandoned":
            result = "abandoned"
        else:
            result = "draw" if winner is None else ("human" if winner == 0 else "bot")
        sim_winner = g.obs.winner if g.obs is not None and g.obs.done else None
        valid = how == "finished" and not g.desynced
        summary = {
            "game": g.id, "result": result, "valid": valid, "turns": g.turns, "desync": g.desynced,
            "desync_turns": len(g.desync_turns), "first_desync": g.first_desync, "sim_error": g.sim_error,
            "characters": list(g.characters), "seed": g.seed, "winner": winner, "sim_winner": sim_winner,
            "final_tick": (final or {}).get("tick"),
            "final_hp": [p.get("hp") for p in (final or {}).get("players", [])] or None,
            "p2_overrides": g.p2_overrides, "p2_fallbacks": g.p2_fallbacks, "reason": reason,
            "deadline_hits": g.deadline_hits, "paths": dict(g.paths), "forfeit": forfeit,
            "lock_cap_hits": g.lock_cap_hits, "untimed_locked": g.untimed_locked,
            "think_ms_mean": _round(sum(g.think_ms) / len(g.think_ms), 1) if g.think_ms else None,
            "think_ms_max": _round(max(g.think_ms), 1) if g.think_ms else None,
            "seconds": round(time.time() - g.t0, 1),
        }
        g.write({"type": "result", **summary, "final_sync": sync})
        g.close()
        self.results.append(summary)
        self._ledger_row(g, summary)
        first = g.sim_error or f"first at turn {(g.first_desync or {}).get('turn')}"
        state = "in sync every turn" if not summary["desync"] else f"DESYNCED ({summary['desync_turns']} turns; {first})"
        self.log(f"game {g.id}: {result} after {g.turns} turns, {state}; log {g.path}")
        return summary

    def _ledger_row(self, g: Game, s: dict[str, Any]) -> None:
        """One `arena` row per game (a bot against an opponent, scored): the bot is `a`, the human
        seat `b`; tagged experiment=human-play. Not rated (one game, no pairs)."""
        score = {"bot": 1.0, "human": 0.0, "draw": 0.5}.get(s["result"])
        finished = s["result"] != "abandoned"
        matchup = {"a": self.label, "b": "human", "games": 1, "complete_pairs": 0, "score": score,
                   "wins_a": int(score == 1.0), "wins_b": int(score == 0.0), "draws": int(score == 0.5),
                   "finished_games": int(finished), "valid_games": int(s["valid"])}
        metrics = {"games": 1, "complete_pairs": 0, "halted": 0, "truncated": int(not finished),
                   "matchups": [matchup], "ratings": None,
                   **{k: s[k] for k in ("result", "valid", "turns", "desync", "desync_turns", "characters", "seed",
                                        "final_tick", "final_hp", "p2_overrides", "p2_fallbacks", "deadline_hits",
                                        "lock_cap_hits", "forfeit",
                                        "think_ms_mean", "think_ms_max", "seconds")},
                   "bot_seat": "p2", "candidate_set": self.candidate_set,
                   "candidate_rules": self.hello.get("candidate_rules"), "sim_winner": s["sim_winner"]}
        artifacts: dict[str, Any] = {"turn_log": str(g.path), "turn_log_sha256": ledger.file_hash(g.path)}
        build = self.hello.get("build") or {}
        prov = ledger.provenance(
            settings={"bot": self.label, "bot_spec": self.bot_spec, "candidate_set": self.candidate_set,
                      "mode": self.mode, "search": "full turns", **g.options, "characters": list(g.characters), "seed": g.seed,
                      "match_settings": g.match_settings},
            started_utc=g.started_utc, engines=1,
            engine=ledger.engine_info({**self.hello, "match_settings": {**self.settings, **g.match_settings}}),
            models=ledger.models_from_specs([self.bot_spec]), solver=ledger.solver_settings([self.bot]),
            exploration={}, artifacts=artifacts,
            counts=ledger.counts(1, int(s["valid"]), {"desync": int(finished and s["desync"]),
                                                      "abandoned": int(not finished)}, 0),
            yomisim={"sha": build.get("yomisim_sha"), "dirty": build.get("yomisim_dirty")})
        tags = {"experiment": "human-play", "arm": ARM, "bot": self.label, "mode": self.mode,
                "pick": g.options.get("pick", "mixed"), "settings": g.preset or "custom",
                "valid": "true" if s["valid"] else "false"}
        if self.model:
            tags.update(generation=str(self.model["generation"]), model_sha256=self.model["sha256"][:16])
        ledger.append("arena", metrics, prov, path=self.ledger_path, tags=tags,
                      unknown=["metrics.ratings"], quiet=True)


@contextlib.contextmanager
def worker_paused(pause: Path, log: Callable[[str], None] = print) -> Iterator[bool]:
    """Hold the round-3 worker's PAUSE file for the duration: create it, and delete it on the way
    out however the block exits. A file that already existed (someone else paused the worker) is
    left in place, with a note. Yields whether this call created it."""
    existed = pause.exists()
    if existed:
        log(f"note: {pause} already exists; it stays when this exits (someone else paused the worker)")
    else:
        pause.parent.mkdir(parents=True, exist_ok=True)
        pause.write_text("human play (tools/human_play.py)\n", encoding="utf-8")
        log(f"paused the round-3 worker: {pause}")
    try:
        yield not existed
    finally:
        if existed:
            log(f"note: left {pause} in place (it existed before this run)")
        else:
            try:
                pause.unlink()
                log(f"unpaused the round-3 worker (removed {pause})")
            except FileNotFoundError:
                pass
            except OSError as e:
                log(f"warning: could not remove {pause}: {e}; delete it by hand")


# --- TCP --------------------------------------------------------------------------------------

class Server:
    """Serves one mod connection at a time on 127.0.0.1 (a later one replaces it: the game was
    restarted). `serve_forever` blocks; `start()` runs it in a daemon thread."""

    def __init__(self, session: Session, port: int = DEFAULT_PORT, host: str = "127.0.0.1"):
        self.session, self.host = session, host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(2)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> "Server":
        self.thread = threading.Thread(target=self.serve_forever, name="human-play-server", daemon=True)
        self.thread.start()
        return self

    def serve_forever(self) -> None:
        self.sock.settimeout(0.5)
        while not self.stopped.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._serve(conn)

    def _serve(self, conn: socket.socket) -> None:
        """This thread reads; a worker thread answers requests in order, one reply each, and owns
        the sim while it thinks. One-way notifications (`locked_in`, `deadline`) never queue or
        get a reply: they signal the stop clock of the last request read, if it is their decision,
        so a decision still thinking hears them and a stale one is dropped. Any request that
        arrives while a decision thinks supersedes it (the mod moved on)."""
        conn.settimeout(None)
        f = conn.makefile("rb")
        requests: queue.Queue = queue.Queue()

        def superseded() -> bool:
            return self.stopped.is_set() or not requests.empty()

        worker = threading.Thread(target=self._answer, args=(conn, requests), name="human-play-worker", daemon=True)
        worker.start()
        last: anytime.Clock | None = None  # the last request read, if a decision: notifications go to it
        try:
            for raw in f:
                if self.stopped.is_set():
                    break
                try:
                    msg = json.loads(raw)
                except ValueError:
                    msg = None
                if isinstance(msg, dict) and msg.get("type") in NOTIFY_TYPES:
                    if not valid_notification(msg):  # logged by key names only, never acted on
                        keys = sorted(map(str, msg))
                        self.session.log(f"refused a notification (only type and turn allowed): keys {keys}")
                    elif last is not None and last.turn == int(msg["turn"]):
                        last.signal(msg["type"])
                    continue
                last = None
                turn = msg.get("turn") if isinstance(msg, dict) and msg.get("type") == "decision" else None
                if isinstance(turn, (int, float)):
                    last = anytime.Clock(int(turn), superseded=superseded)  # t0: when it was read
                requests.put((msg, last))
        except OSError:
            pass
        finally:
            requests.put(None)  # also supersedes a decision still thinking
            worker.join()
            with self.lock:
                self.session.connection_lost()
            f.close()
            conn.close()

    def _answer(self, conn: socket.socket, requests: queue.Queue) -> None:
        alive = True
        while True:
            item = requests.get()
            if item is None:
                return
            msg, clock = item
            with self.lock:
                reply = self.session.handle(msg, clock)
            if alive:
                try:
                    conn.sendall(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
                except OSError:
                    alive = False  # keep draining until the reader stops

    def close(self) -> None:
        self.stopped.set()
        try:
            self.sock.close()
        except OSError:
            pass
        if self.thread is not None:
            self.thread.join(timeout=5)
        with self.lock:
            self.session.connection_lost()


def add_bot_args(ap: argparse.ArgumentParser, required: bool = True) -> None:
    """The bot flags every human-play entry point takes (see bot_from_args)."""
    ap.add_argument("--bot", required=required, default=None if required else "matrix-do-s4",
                    help=f"{', '.join(BOTS)}, current-best, gen-N (a promoted net, e.g. gen-8), "
                         "or a bot spec (make_bot) with @N")
    ap.add_argument("--ckpt", help="checkpoint for the ckpt2b bots")
    ap.add_argument("--best", default=os.environ.get("YOMIBENCH_BEST", DEFAULT_BEST),
                    help=f"r3-best.json for current-best / gen-N: a URL or a local path (default {DEFAULT_BEST}, "
                         "or $YOMIBENCH_BEST); its model files are read from beside it")
    ap.add_argument("--models", type=Path, default=None,
                    help="where hub nets are kept, as <sha256>.npz (default ~/.yomibench/models)")
    ap.add_argument("--sha256", help="gen-N: the net's sha256, if the trainer ledger can't give it")
    ap.add_argument("--candidate-set", type=int, default=None,
                    help=f"current-best / gen-N: candidate set (default {PRODUCTION_SET}, production's)")
    add_play_args(ap)


def _have_yomisim() -> bool:
    import importlib.util

    return importlib.util.find_spec("yomisim") is not None


def add_play_args(ap: argparse.ArgumentParser) -> None:
    """The play options (the panel's starting choices and the deadline), without the bot choice:
    the Windows app (packaging/windows) plays only its bundled net."""
    ap.add_argument("--pick", choices=PICK_STYLES, default="mixed",
                    help="how the bot picks from its solved strategy: mixed samples it (default: the equilibrium "
                         "play), strongest plays its most likely move (changeable on the panel)")
    ap.add_argument("--settings", choices=tuple(SETTINGS_PRESETS), default="ranked",
                    help="match settings preset: ranked (default: stage 600, length 3000, 1.5 bars of meter, ceiling 500, "
                         "the settings production trains on) or standard (the online lobby default) (changeable on the panel)")
    ap.add_argument("--show-estimate", action="store_true",
                    help="show the bot's win estimate in the bottom bar after each turn (changeable on the panel)")
    ap.add_argument("--deadline-margin", type=float, default=anytime.DEADLINE_MARGIN_S,
                    help="with a turn timer, answer at least this many seconds before it runs out "
                         f"(default {anytime.DEADLINE_MARGIN_S:g})")
    ap.add_argument("--deadline-fraction", type=float, default=anytime.DEADLINE_FRACTION,
                    help="... or this fraction of the turn's time, whichever is more "
                         f"(default {anytime.DEADLINE_FRACTION:g}: a 15 s turn is answered by 12.75 s)")
    ap.add_argument("--turn-timer", choices=("on", "off"), default=None,
                    help="the game's turn timer: on (ranked's chess clock; a preset without a timer gets the game's lobby timer) "
                         "or off; default: on for ranked, off for standard, as the game (changeable on the panel)")
    ap.add_argument("--think-until-lockin", action=argparse.BooleanOptionalAction, default=True,
                    help="the bot keeps deepening its search until you lock in your move, after its full search "
                         "(which always finishes first), at most --think-cap seconds a turn; default on "
                         "(--no-think-until-lockin: answer as soon as the full search ends; changeable on the panel)")
    ap.add_argument("--think-cap", type=float, default=anytime.THINK_CAP_S, metavar="SECONDS",
                    help="thinking until lock-in stops this long after a turn opens even if you haven't locked in "
                         f"(default {anytime.THINK_CAP_S:g}; a timed turn's deadline comes first)")
    ap.add_argument("--lock-cap", type=float, default=anytime.LOCK_CAP_S, metavar="SECONDS",
                    help="no turn timer: once you lock in, the bot answers within this many seconds (its best so far "
                         f"if its search hasn't finished; default {anytime.LOCK_CAP_S:g}; timed turns keep their deadline)")
    ap.add_argument("--bank-reserve", type=float, default=anytime.BANK_RESERVE_S, metavar="SECONDS",
                    help="ranked (a chess clock): the bot never plans its bank below this; with no more room it plays "
                         f"on the 15 s floor (default {anytime.BANK_RESERVE_S:g})")
    ap.add_argument("--bank-deep-turns", type=float, default=anytime.BANK_DEEP_TURNS, metavar="N",
                    help="ranked (a chess clock): after its full search, the bot's deeper stages use at most 1/N of its "
                         f"bank's room above the reserve per turn (default {anytime.BANK_DEEP_TURNS:g})")
    ap.add_argument("--engine", default=None, metavar="sim|game-pool:N|game-pool:auto",
                    help="the bot's engine: a simulator plugin (sim) or N headless copies of your own game (game-pool, sized to "
                         "this PC with auto). Default: sim when it is installed, else game-pool:auto")


def check_deadline_args(args: argparse.Namespace) -> None:
    """The launcher's deadline flags can only tighten the deadline, never loosen it (user decision,
    the bot always answers before the game's turn timer)."""
    m, f = getattr(args, "deadline_margin", None), getattr(args, "deadline_fraction", None)
    if m is not None and m < anytime.DEADLINE_MARGIN_S:
        raise SystemExit(f"--deadline-margin can only tighten the deadline: at least {anytime.DEADLINE_MARGIN_S} s, not {m}")
    if f is not None and not anytime.DEADLINE_FRACTION <= f < 1:
        raise SystemExit(f"--deadline-fraction can only tighten the deadline: in [{anytime.DEADLINE_FRACTION}, 1), not {f}")


def bot_from_args(args: argparse.Namespace, log: Callable[[str], None] = print) -> BotChoice:
    check_deadline_args(args)
    try:
        return resolve_bot(args.bot, args.ckpt, best=args.best, models=args.models, sha256=args.sha256,
                           candidate_set=args.candidate_set, log=log)
    except ValueError as e:
        raise SystemExit(f"--bot {args.bot}: {e}") from e


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--launch-hidden" in argv:  # the mod's bot client, after the configured server args (hidden_desktop)
        from yomibench.hidden_desktop import main as launch_hidden

        return launch_hidden(argv[argv.index("--launch-hidden"):])
    ap = argparse.ArgumentParser(prog="python -m yomibench.human_play", description=__doc__.split("\n\n")[0])
    add_bot_args(ap)
    ap.add_argument("--out", required=True, help="folder for games/*.jsonl and ledger-human-play.jsonl")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (tests: the Docker bridge)")
    ap.add_argument("--scene", help="the simulator plugin's scene dump directory (else $YOMISIM_SCENE / the build's default)")
    ap.add_argument("--autoplay", metavar="P1-P2:SEED,...",
                    help="headless test: the games the mod starts in order (e.g. Ninja-Cowboy:5,Cowboy-Cowboy:6)")
    args = ap.parse_args(argv)
    bot = bot_from_args(args)
    schedule = parse_schedule(args.autoplay) if args.autoplay else None
    session = Session(bot.spec, bot.candidate_set, args.out, label=bot.label, scene=args.scene, schedule=schedule,
                      mode="autoplay" if schedule else "human", model=bot.model,
                      pick=args.pick, show_estimate=args.show_estimate,
                      settings=args.settings, think_until_lockin=args.think_until_lockin, think_cap=args.think_cap,
                      lock_cap=args.lock_cap, turn_timer=None if args.turn_timer is None else args.turn_timer == "on",
                      deadline_margin=args.deadline_margin, deadline_fraction=args.deadline_fraction,
                      bank_reserve=args.bank_reserve, bank_deep_turns=args.bank_deep_turns,
                      engine_spec=args.engine)
    server = Server(session, args.port, args.host)
    print(f"human-play bot server: {session.label} (candidate set {bot.candidate_set}) on {args.host}:{server.port}; "
          f"engine build {session.hello.get('build', {}).get('yomisim_sha', '?')[:7]}; logs in {session.out}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


def parse_schedule(text: str) -> list[tuple[str, str, int]]:
    out = []
    for item in (s.strip() for s in text.split(",")):
        if item:
            chars, _, seed = item.partition(":")
            p1, _, p2 = chars.partition("-")
            out.append((p1, p2, int(seed)))
    return out


if __name__ == "__main__":
    sys.exit(main())

"""Inputs v4: a sidecar block of value-net inputs appended after features.encode's columns.

ACTION_SPACE_V7 §6. `features.py` is not edited, so `cache.encoder_hash()` and every existing cache stay valid;
this module has its own version (EXT_FEATURES) and hash (ext_hash). A model trained with the block has meta
`features = 4`: features.encode's columns, then encode_ext's (valuenet.ENCODERS[4] = features_v4).

Same conventions as features.py: egocentric and mirrored (x flipped so the opponent is on the positive side;
every horizontal quantity flips), raw numbers (the model standardizes them), names as vocabulary ids (the
shared "kind" / "ostate" tables). Generic throughout: object kinds and states are ids, timers are found by
field-name patterns, nothing names a move or a character.

Three parts, each selectable (`parts`, default all; columns of a part left out are zeros, so the layout never
changes with the selection):

- **1. Object identity and lifetime.** RESERVED_PER_OWNER slots per owner (mine first) for persistent objects,
  chosen from every live object (not only the 8 nearest, so they never fall off): one object per (owner, kind),
  the oldest of its kind, the kinds ranked by that age. Per slot: position and velocity, state tick, age, ticks
  left, an object-level timer, armed / detonating flags, the next hitbox's timing, how many of that kind the
  owner has. The 8 nearest slots of features.encode get ticks left, timer and armed flag. Per side: objects
  beyond the 8 slots and their distinct kinds.
- **2. Readiness and follow-up windows.** Per fighter: ticks to iasa, and (engine `ext` fields) the next
  interrupt frame, hold / land-cancel windows, cancel flags, and whether a hitbox offers a follow-up.
- **3. Second aim and hitbox geometry.** Per fighter: the second `xy` field of the submitted state data
  (Chainsaw's End), how many there are, the HIT_SLOTS next hitboxes by start tick (timing, damage, knockback,
  direction, and engine geometry), and the hurtbox (engine).

- **4. Committed extra** (engine `ext.extra`): the extra the fighter's current state was entered with, as
  one fixed vector (EXTRA_BOOL, EXTRA_XY), for controls whose effect no reported variable keeps.
- **5. Raw gaps** (RAWGAP): reported fighter variables the v3 raw block doesn't read, where a toggle's effect
  persists or g15's residual is high (set-7 re-audit).

Tier A columns come from records as they are. Tier B columns need the engine's `ext` dicts (players[i].ext,
objects[i].ext, a hitbox's ext), which the default state doesn't carry; they read 0 when absent and the
`v4b` column says whether the state carried them (as features' `v3` flag did).
"""

from __future__ import annotations

import hashlib
import inspect
import math
from typing import Any

from yomibench import features
from yomibench.features import FAR, NEAR, NO_HIT_YET, OBJECT_SLOTS, SOON, Vocab, _num

EXT_FEATURES = 3  # bump when encode_ext changes (3: lean v4.1 — Orb/grenade/StickyBomb fixes, 12 obj slots, pending threat)
PARTS = (1, 2, 3, 4, 5)
RESERVED_PER_OWNER = 2
EXTRA_OBJ_SLOTS = 4

# Object-level raw fields read as a timer / flag (generic name patterns; first two matches in sorted field order).
TIMER_SUFFIXES = ("ticks_left", "_ticks", "cooldown", "_frames", "_tick")
ARMED_FLAGS = ("active", "armed", "attached", "is_locked", "locked", "launched")
DETONATING_FLAGS = ("detonating", "exploded")
# Engine ext fields (tier B), read from a dict's "ext" (or the dict itself).
# Scripts use huge lifetimes (30000, 1e6, 1e7) to mean "no lifetime"; a timer at or above this is ignored (game length 3000).
NO_LIFETIME = 10000
# Every ext number is clipped to +-CLIP (float16, the training cache's storage, tops out at 65504).
CLIP = 60000.0
OBJ_EXT_TIMERS = ("lifetime", "explode_ticks", "attach_frames", "explode_tick", "orb_lifetime", "arming_time",
                  "floor_slide_lifetime")
OBJ_EXT_COUNTDOWNS = frozenset({"explode_ticks", "attach_frames"})

RES_NUMS = ("present", "dx", "dy", "vx", "vy", "state_tick", "age", "ticks_left", "timer", "timer2", "armed",
            "detonating", "hit_in", "hit_active", "hit_damage", "same_kind", "push_x", "push_y")
SLOT_NUMS = ("ticks_left", "timer", "armed")
EXTRA_OBJ_NUMS = ("present", "mine", "dx", "dy", "vx", "vy", "state_tick", "ticks_left", "armed",
                  "hit_in", "hit_active", "hit_damage")
SIDE_NUMS = ("beyond8", "beyond8_kinds", "pending_count", "soonest_hit_in", "nearest_pending", "pending_damage_near")
READY_NUMS = ("ticks_to_iasa", "next_interrupt_in", "interruptible_opp_turn", "followup")
AIM_NUMS = ("aim2_x", "aim2_y", "aim2_len", "n_xy")
FIGHTER3_NUMS = AIM_NUMS

N_RESERVED = 2 * RESERVED_PER_OWNER
# Column ranges per part, in order: global, part 1, part 2, part 3, part 4, part 5.
PART1_NAMES = ([f"res{k}.{n}" for k in range(N_RESERVED) for n in RES_NUMS]
               + [f"obj{k}.{n}" for k in range(OBJECT_SLOTS) for n in SLOT_NUMS]
               + [f"exobj{k}.{n}" for k in range(EXTRA_OBJ_SLOTS) for n in EXTRA_OBJ_NUMS]
               + [f"{side}.{n}" for side in ("mine", "theirs") for n in SIDE_NUMS])
PART2_NAMES = [f"{side}.{n}" for side in ("me", "opp") for n in READY_NUMS]
PART3_NAMES = [f"{side}.{n}" for side in ("me", "opp") for n in FIGHTER3_NUMS]
# Part 4: the committed extra (engine ext.extra) as one fixed vector shared by every character: each boolean
# control 0/1, each xy control (x mirrored, y). Lean v4.1 drops dead keys and keys already in v3/Part 5.
EXTRA_BOOL = ("armor_enabled", "back_juke", "bloom", "detach", "drive_cancel", "explode", "fly_enabled", "hindsight",
              "honk", "lock_orb", "nade_activated", "pull", "release", "spike_enabled")
EXTRA_XY = ("fly_dir", "orb_push", "release_dir")
EXTRA_NUMS = ("has_extra",) + EXTRA_BOOL + tuple(f"{k}_{c}" for k in EXTRA_XY for c in ("x", "y"))
# Part 5: fighter variables `raw` reports that features.encode doesn't read, chosen by the set-7 re-audit
# and 11/24 v4.1 audit (§2.3). Same convention as features.CHARACTER_RAW (absent -> character's start value),
# shared by name across characters (a variable a character lacks reads 0). None is a world-x quantity.
# `last_parry_tick` enters as ticks since (game tick minus it).
RAWGAP = ("reverse_state", "bullet_cancelling", "detonating", "detonating_bombs", "shifted_this_turn",
          "juked_this_turn", "up_juke_ticks", "detach_delay", "bloomed_last_turn", "feinted_last",
          "blocked_last_turn", "got_blocked", "can_nudge", "block_used_air_movement", "last_parry_tick",
          "quick_slash_move_dir_y", "lightning_slice_y", "orb_accumulator", "fast_fall_landing",
          "blocked_hitbox_plus_frames", "penalty_ticks", "spark_speed_frames", "tether_ticks",
          "propel_friction_ticks", "magnetize_bomb_strength", "drift_jump_timer",
          "cancel_opponent_hitstun_countdown")
PART4_NAMES = [f"{side}.{n}" for side in ("me", "opp") for n in EXTRA_NUMS]
PART5_NAMES = [f"{side}.{n}" for side in ("me", "opp") for n in RAWGAP]
_PART_NAMES = {1: PART1_NAMES, 2: PART2_NAMES, 3: PART3_NAMES, 4: PART4_NAMES, 5: PART5_NAMES}
EXT_NUMERIC_NAMES = tuple(["ext.v4b"] + [f"ext{k}.{n}" for k, names in _PART_NAMES.items() for n in names])
N_EXT_NUMERIC = len(EXT_NUMERIC_NAMES)
PART_SLICES = {}
_at = 1
for _k, _names in _PART_NAMES.items():
    PART_SLICES[_k] = slice(_at, _at + len(_names))
    _at += len(_names)
assert _at == N_EXT_NUMERIC
EXT_CATEGORY_OF_SLOT = ["kind", "ostate"] * (N_RESERVED + EXTRA_OBJ_SLOTS)
N_EXT_CATEGORICAL = len(EXT_CATEGORY_OF_SLOT)
EXT_CAT_PART = [1] * N_EXT_CATEGORICAL  # every ext categorical belongs to part 1


def ext_hash() -> str:
    """Changes whenever encode_ext could give different columns (and with features.py's encoder, which it
    reuses for the object order)."""
    from yomibench import cache
    parts = [f"ext {EXT_FEATURES}", inspect.getsource(inspect.getmodule(ext_hash)), cache.encoder_hash()]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def _ext(d: dict[str, Any], key: str) -> Any:
    e = d.get("ext")
    if isinstance(e, dict) and key in e:
        return e[key]
    return None


def _has_ext(state: dict[str, Any]) -> bool:
    return any(isinstance(p.get("ext"), dict) for p in state.get("players", ()))


def _raw_timers(raw: dict[str, Any]) -> tuple[float, float]:
    t1 = 0.0
    for k in sorted(raw):
        if k.endswith(TIMER_SUFFIXES) and not isinstance(raw[k], bool):
            v = _num(raw[k])
            if v:
                if not t1:
                    t1 = v
                else:
                    return t1, v
    return t1, 0.0


def _raw_timer(raw: dict[str, Any]) -> float:
    return _raw_timers(raw)[0]


def _flag(raw: dict[str, Any], names: tuple[str, ...]) -> float:
    return float(any(raw.get(n) is True for n in names))


def _obj_age(o: dict[str, Any], tick: float) -> float:
    alive = _ext(o, "ticks_alive")
    if alive is not None:
        return _num(alive)
    spawn = _ext(o, "spawn_tick")
    if spawn is not None:
        return tick - _num(spawn)
    return _num(o.get("state_tick", 0))


def _obj_ticks_left(o: dict[str, Any]) -> float:
    """Ticks until the object's state ends by its own timer (engine ext), else 0 (unknown)."""
    st = _num(o.get("state_tick", 0))
    for k in OBJ_EXT_TIMERS:
        v = _ext(o, k)
        if v is not None and 0 < _num(v) < NO_LIFETIME:
            return _num(v) if k in OBJ_EXT_COUNTDOWNS else max(0.0, _num(v) - st)
    length = _ext(o, "anim_length")
    if length is not None and not _ext(o, "endless") and 0 < _num(length) < NO_LIFETIME:
        return max(0.0, _num(length) - st)
    return 0.0


def _obj_hit(o: dict[str, Any]) -> tuple[float, float, float]:
    """(ticks to the next pending hitbox, any active, damage of the hardest active or pending)."""
    st = _num(o.get("state_tick", 0))
    hbs = [h for h in o.get("hitboxes", ()) if h.get("active") or st < _num(h.get("start_tick")) + _num(h.get("active_ticks"))]
    if not hbs:
        return NO_HIT_YET, 0.0, 0.0
    return (min(max(0.0, _num(h.get("start_tick")) - st) for h in hbs), float(any(h.get("active") for h in hbs)),
            max(_num(h.get("damage")) for h in hbs))


def _closeness(o, mx, my, ox, oy):  # features.encode's sort key
    return min(abs(o["x"] - mx) + abs(o["y"] - my), abs(o["x"] - ox) + abs(o["y"] - oy))


def _part1(state, seat, vocab, flip, me, opp, tick, nums, cats):
    my_id = seat + 1
    objects = state.get("objects", [])
    mx, my = me["x"], me.get("y", 0.0)
    ox, oy = opp["x"], opp.get("y", 0.0)
    for owner_mine in (True, False):
        own = [o for o in objects if (o.get("owner") == my_id) == owner_mine]
        groups: dict[str, list[dict[str, Any]]] = {}
        for o in own:
            groups.setdefault(str(o.get("kind", o.get("name", ""))), []).append(o)
        # per kind its oldest object (ties: lowest name); kinds by that age, then kind name: deterministic
        best = []
        for kind, objs in groups.items():
            o = max(objs, key=lambda o: (_obj_age(o, tick), -_safe_name(o)))
            best.append((-_obj_age(o, tick), kind, o, len(objs)))
        best.sort(key=lambda t: (t[0], t[1]))
        for k in range(RESERVED_PER_OWNER):
            if k < len(best):
                _, kind, o, n = best[k]
                raw = o.get("raw") or {}
                hit_in, hit_active, hit_dmg = _obj_hit(o)
                t1, t2 = _raw_timers(raw)
                nums += [1.0, (o["x"] - mx) * flip, o["y"] - my, _num(o.get("vx", 0.0)) * flip, _num(o.get("vy", 0.0)),
                         _num(o.get("state_tick", 0)), _obj_age(o, tick), _obj_ticks_left(o), t1, t2,
                         _flag(raw, ARMED_FLAGS), _flag(raw, DETONATING_FLAGS), hit_in, hit_active, hit_dmg, float(n),
                         _num(raw.get("last_push_x")) * flip, _num(raw.get("last_push_y"))]
                cats += [vocab.get("kind", kind), vocab.get("ostate", str(o.get("state", "")))]
            else:
                nums += [0.0] * len(RES_NUMS)
                cats += [0, 0]
    objs = sorted(objects, key=lambda o: _closeness(o, mx, my, ox, oy))
    for k in range(OBJECT_SLOTS):
        if k < len(objs):
            raw = objs[k].get("raw") or {}
            nums += [_obj_ticks_left(objs[k]), _raw_timer(raw), _flag(raw, ARMED_FLAGS)]
        else:
            nums += [0.0] * len(SLOT_NUMS)
    for k in range(EXTRA_OBJ_SLOTS):
        idx = OBJECT_SLOTS + k
        if idx < len(objs):
            o = objs[idx]
            raw = o.get("raw") or {}
            hit_in, hit_active, hit_dmg = _obj_hit(o)
            nums += [1.0, float(o.get("owner") == my_id), (o["x"] - mx) * flip, o["y"] - my,
                     _num(o.get("vx", 0.0)) * flip, _num(o.get("vy", 0.0)), _num(o.get("state_tick", 0)),
                     _obj_ticks_left(o), _flag(raw, ARMED_FLAGS), hit_in, hit_active, hit_dmg]
            cats += [vocab.get("kind", str(o.get("kind", o.get("name", "")))), vocab.get("ostate", str(o.get("state", "")))]
        else:
            nums += [0.0] * len(EXTRA_OBJ_NUMS)
            cats += [0, 0]
    rest = objs[OBJECT_SLOTS:]
    for owner_mine in (True, False):
        side_rest = [o for o in rest if (o.get("owner") == my_id) == owner_mine]
        tx, ty = (ox, oy) if owner_mine else (mx, my)
        p_count = p_near_dmg = 0.0
        p_soon, p_near = NO_HIT_YET, FAR
        for o in objs:
            if (o.get("owner") == my_id) != owner_mine:
                continue
            hit_in, hit_active, hit_dmg = _obj_hit(o)
            if hit_in < NO_HIT_YET and not hit_active:
                p_count += 1.0
                if hit_in < p_soon:
                    p_soon = hit_in
                dist = math.hypot(tx - o["x"], ty - o["y"])
                if dist < p_near:
                    p_near = dist
                if dist < NEAR:
                    p_near_dmg += hit_dmg
        nums += [float(len(side_rest)), float(len({str(o.get("kind", "")) for o in side_rest})),
                 p_count, p_soon, p_near, p_near_dmg]


def _safe_name(o: dict[str, Any]) -> float:
    try:
        return float(o.get("name"))
    except (TypeError, ValueError):
        return 0.0


def _ready(p: dict[str, Any]) -> list[float]:
    tick = _num(p.get("state_tick", 0))
    iasa = p.get("state_iasa")
    has_iasa = iasa is not None and _num(iasa) > 0
    to_iasa = max(0.0, _num(iasa) - tick) if has_iasa else 0.0
    frames = _ext(p, "interrupt_frames")
    nxt = SOON
    if isinstance(frames, list):
        later = [_num(f) - tick for f in frames if _num(f) > tick]
        nxt = min(later, default=SOON)
    followup = 0.0
    for h in p.get("hitboxes") or ():
        if _ext(h, "cancellable") or _ext(h, "followup_state"):
            followup = 1.0
    return [to_iasa, nxt, float(bool(_ext(p, "interruptible_on_opponent_turn"))), followup]


def xy_fields(data: Any) -> list[tuple[float, float]]:
    """Every {"x", "y"} dict in submitted state data, in features._move_data's walk order."""
    out: list[tuple[float, float]] = []

    def walk(d: Any, key: str) -> None:
        if not isinstance(d, dict) or key == "Block Height":
            return
        if "x" in d and "y" in d:
            out.append((_num(d["x"]), _num(d["y"])))
        elif "x" in d or "count" in d:
            return
        else:
            for k, v in d.items():
                walk(v, k)

    walk(data, "")
    return out


def _fighter3(p: dict[str, Any], flip: float) -> list[float]:
    xys = xy_fields(p.get("state_data")) if "hitboxes" in p else []
    if len(xys) >= 2:
        ax, ay = xys[1]
        n = math.hypot(ax, ay)
        return [ax / n * flip if n else 0.0, ay / n if n else 0.0, n, float(len(xys))]
    return [0.0, 0.0, 0.0, float(len(xys))]


def _extra(p: dict[str, Any], flip: float) -> list[float]:
    ex = _ext(p, "extra")
    if not isinstance(ex, dict):
        return [0.0] * len(EXTRA_NUMS)
    out = [1.0] + [float(ex.get(k) is True) for k in EXTRA_BOOL]
    for k in EXTRA_XY:
        v = ex.get(k)
        out += [_num(v.get("x")) * flip, _num(v.get("y"))] if isinstance(v, dict) else [0.0, 0.0]
    return out


def _rawgap(p: dict[str, Any], vocab: Vocab, tick: float) -> list[float]:
    raw = p.get("raw") or {}
    defaults = vocab.raw_defaults.get(str(p.get("character", "")), {})
    out = []
    for name in RAWGAP:
        v = _num(raw[name] if name in raw else defaults.get(name, 0))
        if name == "last_parry_tick":
            v = min(SOON, max(0.0, tick - v)) if v > 0 else SOON
        out.append(v)
    return out


def merge_ext(state: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """`state` with the engine ext fields of a backfill row (engine backfill: per player its ext
    fields and `hitboxes` (each hitbox's ext), per object, in the state's object order, its ext fields and
    `hitboxes`) attached as the live engine's `ext: true` state carries them: players[i].ext,
    players[i].hitboxes[j].ext, objects[k].ext, objects[k].hitboxes[j].ext. Shallow copies; the input is
    unchanged. Raises ValueError when the row doesn't line up with the state (object count or names)."""
    out = dict(state)
    players = []
    for p, e in zip(state["players"], row["players"]):
        q = dict(p)
        q["ext"] = {k: v for k, v in e.items() if k != "hitboxes"}
        q["hitboxes"] = [{**h, "ext": he} for h, he in zip(p.get("hitboxes") or (), e.get("hitboxes") or ())]
        players.append(q)
    out["players"] = players
    objs = state.get("objects", [])
    eobjs = row.get("objects", [])
    if len(objs) != len(eobjs):
        raise ValueError(f"ext row has {len(eobjs)} objects, state {len(objs)}")
    merged = []
    for o, e in zip(objs, eobjs):
        if str(e.get("name")) != str(o.get("name")) or str(e.get("kind")) != str(o.get("kind")):
            raise ValueError(f"ext object {e.get('name')}/{e.get('kind')} != state {o.get('name')}/{o.get('kind')}")
        q = dict(o)
        q["ext"] = {k: v for k, v in e.items() if k not in ("hitboxes", "name", "kind")}
        q["hitboxes"] = [{**h, "ext": he} for h, he in zip(o.get("hitboxes") or (), e.get("hitboxes") or ())]
        merged.append(q)
    out["objects"] = merged
    return out


def read_ext_rows(path: str) -> list[dict[str, Any]]:
    """A backfill file's rows in record order (header and final line dropped). Rows whose replay didn't equal the
    record (`equal` false) come back as None, so the caller leaves those decisions without ext."""
    import gzip
    import json
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0 or d.get("final"):
                continue
            rows.append(d if d.get("equal", True) else None)
    return rows


def encode_ext(state: dict[str, Any], seat: int, vocab: Vocab, parts: tuple[int, ...] = PARTS) -> tuple[list[float], list[int]]:
    """(ext numeric features, ext categorical ids) for `seat`; columns of parts not in `parts` are 0."""
    me, opp = state["players"][seat], state["players"][1 - seat]
    flip = 1.0 if me["x"] <= opp["x"] else -1.0
    tick = _num(state.get("tick", 0))
    nums: list[float] = [float(_has_ext(state))]
    cats: list[int] = []
    _part1(state, seat, vocab, flip, me, opp, tick, nums, cats)
    nums += _ready(me) + _ready(opp)
    nums += _fighter3(me, flip) + _fighter3(opp, flip)
    nums += _extra(me, flip) + _extra(opp, flip)
    nums += _rawgap(me, vocab, tick) + _rawgap(opp, vocab, tick)
    assert len(nums) == N_EXT_NUMERIC and len(cats) == N_EXT_CATEGORICAL
    nums = [CLIP if v > CLIP else (-CLIP if v < -CLIP else v) for v in nums]
    if tuple(parts) != PARTS:
        for part, sl in PART_SLICES.items():
            if part not in parts:
                nums[sl] = [0.0] * (sl.stop - sl.start)
        if 1 not in parts:
            cats = [0] * N_EXT_CATEGORICAL
    return nums, cats


# Unused-import guard for names re-exported for callers (FAR used by tests of the overflow summary).
__all__ = ["encode_ext", "ext_hash", "xy_fields", "merge_ext", "read_ext_rows", "EXT_FEATURES", "N_EXT_NUMERIC", "N_EXT_CATEGORICAL",
           "EXT_NUMERIC_NAMES", "EXT_CATEGORY_OF_SLOT", "PART_SLICES", "PARTS", "FAR"]

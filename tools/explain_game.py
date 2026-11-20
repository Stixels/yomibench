"""Explain a recorded game's decisions: why each seat's bot picked what it picked.

    python tools/explain_game.py RECORD.jsonl.gz [...] --out DIR [--procs 2] [--net NAME=PATH]

For every decision where a seat is actionable and has a choice, the output (DIR/<stem>.explain.json)
holds both fighters' state (hp, meter bars, burst, combo, the Cowboy's gun and bullets, distance)
and, per acting seat:

- its top candidates by value against the opponent's mix, as HP-equivalents relative to the seat's
  game value (value 1.0 = `hp_per_value` HP, priced per net from resource_pricing.json's `hp+100`:
  value 0.01 is about 12 HP for gen 6), plus every move of its mix and its pick;
- the mix it played, the opponent mix its search solved, and the pick;
- flags: `in_mix`; `flat_tie` (the best move outside the mix is within 0.005 of the game value: the
  net can't tell them apart; moves inside a mix are equal by construction, so they are never
  compared with each other, and a "move" is an action: factor variants of a mixed move, such as
  another DI, are reported apart as `variant_gap`); `confident` (that gap is 0.02 or more); `search` (`full` or `quick`,
  a playout-capped self-play decision); `explored` with its mode (`uniform` opening noise or
  `softmax`); `opening`;
- a timeline `category` (explored > quick > flat_tie > confident > mixed > moderate), a `waste` kind
  for the picks people ask about (Wait, Taunt, Continue from an idle Wait/Fall state) and a
  one-line reading.

Where the values come from:

- self-play records store each decision's search (`strategies`, `value`, `candidate_values`): used as is;
- gate records from arena.py since it records searches (`seat_values`): used as is, each seat by its own bot;
- older gate and showcase records store none: the game is replayed in the simulator plugin and each seat's bot
  (rebuilt from the header's config and net, the full search gates use) searches every recorded
  state again, exactly as in the game (same order, same seeded draws), so the re-search must
  reproduce every recorded pick; the output says `values: "recomputed"` with the replay's checks.

Nets are found by the header's value names (`valuenet:t15-T2`) in --net-dir folders (default:
$YOMIBENCH_DATA, else ~/yomibench-data: round3/models, gather, checkpoints), by their meta name;
`--net NAME=PATH` overrides. Runs at low priority (nice 19), one math thread per process, at most
2 processes.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"  # before numpy loads: one math thread per process

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FLAT_TIE = 0.005    # best move outside the mix within this of the game value: a flat tie
CONFIDENT = 0.02    # ... this far or more below it: a confident choice
TOP = 8             # candidates listed per seat (plus the mix and the pick)
DEFAULT_HP_PER_VALUE = 1200.0  # value 0.01 ~ 12 HP, when no pricing covers the net
IDLE_STATES = ("Wait", "Fall")  # a fighter's idle states: Continue there keeps it idle
MAX_PROCS = 2


def data_root() -> Path:
    env = os.environ.get("YOMIBENCH_DATA")
    return Path(env) if env else Path.home() / "yomibench-data"


def default_net_dirs() -> list[Path]:
    root = data_root()
    return [root / "round3" / "models", root / "gather", root / "checkpoints"]


def read_record(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    lines = [json.loads(x) for x in gzip.open(path, "rt", encoding="utf-8")]
    decisions = [x for x in lines[1:] if "picks" in x]
    result = next((x["result"] for x in lines[1:] if "result" in x), None)
    return lines[0], decisions, result


# --- nets and pricing --------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def net_index(dirs: list[Path]) -> dict[str, list[Path]]:
    """Meta name -> model files with it, in folder order (a file's meta is read, not its weights)."""
    import numpy as np

    out: dict[str, list[Path]] = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.npz")) if Path(d).is_dir() else []:
            try:
                with np.load(p, allow_pickle=False) as f:
                    name = json.loads(str(f["meta"])).get("name")
            except Exception:  # noqa: BLE001  (not a value net)
                continue
            if name:
                out.setdefault(name, []).append(p)
    return out


def resolve_net(value_name: str | None, overrides: dict[str, Path], index: Callable[[], dict[str, list[Path]]]
                ) -> dict[str, Any] | None:
    """{name, path, sha256} of the net a header names (`valuenet:NAME`), None for a bot without one.
    Several files with the name must be the same net (one sha256)."""
    if not value_name or not value_name.startswith("valuenet:"):
        return None
    name = value_name.split(":", 1)[1]
    if name in overrides:
        path = overrides[name]
        return {"name": name, "path": str(path), "sha256": file_sha256(path)}
    paths = index().get(name, [])
    if not paths:
        raise LookupError(f"no model named {name!r} in the net folders; pass --net {name}=PATH")
    shas = {p: file_sha256(p) for p in paths}
    if len(set(shas.values())) > 1:
        raise LookupError(f"several different models are named {name!r}: "
                          + ", ".join(f"{p} ({s[:12]})" for p, s in shas.items()) + f"; pass --net {name}=PATH")
    return {"name": name, "path": str(paths[0]), "sha256": shas[paths[0]]}


def hp_per_value(sha256: str | None, pricing: dict[str, Any] | None) -> tuple[float, str]:
    """HP per unit of value for a net (100 / its mean value of +100 hp) and where that came from."""
    if pricing and sha256:
        for gen, sha in pricing.get("nets", {}).items():
            if sha == sha256:
                mean = pricing.get("results", {}).get(gen, {}).get("hp+100", {}).get("mean")
                if mean:
                    return round(100 / mean, 1), f"resource_pricing {gen} hp+100"
    return DEFAULT_HP_PER_VALUE, "default (net not priced): value 0.01 ~ 12 HP"


# --- searches per decision ---------------------------------------------------------------------

def _seat_search(strategy, opponent, values, value) -> dict[str, Any]:
    return {"strategy": list(strategy), "opponent": list(opponent), "values": list(values), "value": value}


def recorded_searches(header: dict[str, Any], decisions: list[dict[str, Any]]) -> list[list[Any]] | None:
    """Each decision's [seat 0, seat 1] search from the record itself (each seat from its own side,
    over the record's lists), or None if the record doesn't store searches (older gate records)."""
    if header.get("arena"):
        if not any(d.get("seat_values") is not None for d in decisions):
            return None
        out = []
        for d in decisions:
            row = []
            for s in (0, 1):
                if d.get("seat_values") is None or d["seat_values"][s] is None or d["candidate_values"][s] is None:
                    row.append(None)
                else:
                    row.append(_seat_search(d["strategies"][s], d["opponent_strategies"][s], d["candidate_values"][s],
                                            d["seat_values"][s]))
            out.append(row)
        return out
    out = []
    for d in decisions:
        if not d.get("strategies") or d.get("value") is None or not d.get("candidate_values"):
            out.append([None, None])
            continue
        st, v = d["strategies"], d["value"]
        out.append([_seat_search(st[s], st[1 - s], d["candidate_values"][s], v if s == 0 else -v) for s in (0, 1)])
    return out


def _key(c: dict[str, Any]) -> str:
    return json.dumps(c, sort_keys=True)


def bots_for(header: dict[str, Any], nets: list[dict[str, Any] | None]) -> list[Any]:
    """Each seat's bot as the game ran it: the header's search config and net. Raises if the rebuilt
    bot's config differs from the recorded one (another search would explain another game)."""
    from yomibench.bots import make_bot

    bots = []
    for s, cfg in enumerate(header["bots"]):
        kind = {"oracle": "matrix-do", "full": "matrix"}[cfg["search"]]
        bot = make_bot(f"{kind}:{nets[s]['path']}" if nets[s] else kind)
        bot.candidate_set = cfg.get("candidate_set")
        if cfg.get("playout_cap"):
            bot.playout_cap = tuple(cfg["playout_cap"])
        have = bot.config()
        keys = ("search", "iterations", "tolerance", "oracle_start", "refine_rounds", "di_refine_moves", "explore",
                "candidate_set", "value", "playout_cap", "policy")
        diff = {k: (cfg.get(k), have.get(k)) for k in keys if cfg.get(k) != have.get(k)}
        if diff:
            raise ValueError(f"seat {s}: the rebuilt bot's search differs from the record's (record, rebuilt): {diff}")
        bot.name = cfg.get("bot", bot.name)
        bots.append(bot)
    return bots


def research(header: dict[str, Any], decisions: list[dict[str, Any]], result: dict[str, Any] | None,
             nets: list[dict[str, Any] | None], make_engine: Callable[[list[Any]], Any] | None = None
             ) -> tuple[list[list[Any]], dict[str, Any]]:
    """Replay the game and search every recorded decision again with each seat's own bot, as
    arena._play did: seat 0 then seat 1 choose on the shared engine with their seeded draws, so the
    picks must come out as recorded (`reproduced`). Every state is checked against the record.
    `make_engine(bots)` builds the engine (default: a batch engine serving both nets, as gates play)."""
    from render_record import Divergence, pick_ids, state_diff

    bots = bots_for(header, nets)
    if make_engine is None:
        from yomibench.plugins import batch

        batch_arena_engines = batch().batch_arena_engines

        cset = header.get("engine", {}).get("candidate_set")
        engine = batch_arena_engines(bots, int(cset) if cset not in (None, "") else None)("batch:1")
    else:
        engine = make_engine(bots)
    want, have = header.get("engine", {}).get("candidate_rules"), engine.hello().get("candidate_rules")
    if want != have:
        raise Divergence(f"candidate_rules differ: record {want}, engine {have}")
    seed = header["seed"]
    rngs = (random.Random(f"{seed}:0"), random.Random(f"{seed}:1"))
    sets = tuple(b.candidate_set for b in bots)
    chars = header["characters"]
    obs = engine.new_match(chars[0], chars[1], seed, candidate_sets=sets) if any(sets) else \
        engine.new_match(chars[0], chars[1], seed)
    out, reproduced, seat_decisions, t0 = [], 0, 0, time.perf_counter()
    for k, d in enumerate(decisions):
        where = f"decision {k} (turn {d['turn']})"
        if obs.done:
            raise Divergence(f"{where}: the replay already ended")
        diff = state_diff(d["state"], obs.state)
        if diff:
            raise Divergence(f"{where}: state differs:\n  " + "\n  ".join(diff))
        ids, same = [], True
        for s in (0, 1):
            ids.append(bots[s].choose(obs, s, engine, rngs[s]))
        infos = [bots[s].decision(obs, s, engine) for s in (0, 1)]  # after both choices, as the recorder reads them
        # The record's lists, plus any candidate a search found that the game's searches didn't (a
        # re-search that left the game's path, say on float differences between machines): appended,
        # so the record's indices stay valid and no probability is dropped.
        lists = [list(d["candidates"][side]) for side in (0, 1)]
        pos = [{_key(c): i for i, c in enumerate(lists[side])} for side in (0, 1)]
        for info in infos:
            for side in (0, 1):
                for c in info["lists"][side]:
                    k = _key(c.record())
                    if k not in pos[side]:
                        pos[side][k] = len(lists[side])
                        lists[side].append(c.record())
        hits = []
        for s in (0, 1):
            picked = next(c for c in infos[s]["lists"][s] if c.id == ids[s])
            hit = _key(picked.record()) == _key(d["candidates"][s][d["picks"][s]])
            hits.append(hit)
            same &= hit
            if len(d["candidates"][s]) > 1:
                seat_decisions += 1
                reproduced += hit
        row = []
        for s, info in enumerate(infos):
            def onto(side: int, xs: Any, fill: Any) -> list[Any]:
                o = [fill] * len(lists[side])
                for c, x in zip(info["lists"][side], xs):
                    o[pos[side][_key(c.record())]] = x
                return o
            replies = info["replies"]
            if replies is None:
                row.append(None)
                continue
            v = info["value"] if s == 0 else -info["value"]
            row.append({**_seat_search(
                onto(s, [round(p, 5) for p in info["strategies"][s]], 0.0),
                onto(1 - s, [round(p, 5) for p in info["strategies"][1 - s]], 0.0),
                onto(s, [round(x, 4) if math.isfinite(x) else None for x in replies[s]], None), round(v, 5)),
                "reproduced": hits[s]})
        if any(len(lists[side]) > len(d["candidates"][side]) for side in (0, 1)):
            row = [None if r is None else {**r, "candidates": lists} for r in row]
        out.append(row)
        if not same:  # play the recorded picks (expanded ones rebuilt), so the replay stays on the record
            try:
                ids = pick_ids(engine, d)
            except Divergence as e:
                raise Divergence(f"{where} {e}") from None
        obs = engine.step(*ids)
    if result is not None:
        have_r = {"done": obs.done, "winner": obs.winner, "turns": obs.turn}
        want_r = {k: result.get(k) for k in have_r}
        if want_r != have_r:
            raise Divergence(f"result differs: record {want_r}, replay {have_r}")
    return out, {"engine": engine.hello().get("engine"), "yomisim": engine.hello().get("build", {}).get("yomisim_sha"),
                 "states_checked": len(decisions), "picks_reproduced": reproduced, "seat_decisions": seat_decisions,
                 "seconds": round(time.perf_counter() - t0, 1)}


# --- explanation -------------------------------------------------------------------------------

def label(c: dict[str, Any]) -> str:
    """A short readable name for a candidate: action, its data, and factor overrides."""
    out = c["action"]
    data = c.get("data")
    if isinstance(data, dict):
        def show(k: str, v: Any) -> str | None:
            if isinstance(v, bool):
                return k if v else None
            if isinstance(v, dict) and "name" in v:
                return str(v["name"])
            if isinstance(v, dict) and set(v) <= {"x", "y"}:
                return f"{k} ({','.join(str(v[a]) for a in ('x', 'y') if a in v)})"
            if isinstance(v, dict) and set(v) == {"count"}:
                return f"{k} {v['count']}"
            return f"{k} {v}"
        xy = "x" in data and "y" in data and not isinstance(data["x"], dict)
        parts = ([f"({data['x']},{data['y']})"] if xy else []) + [
            t for k, v in data.items() if not (xy and k in ("x", "y")) for t in [show(k, v)] if t]
        if parts:
            out += " " + "; ".join(parts)
    elif data is not None:
        out += f" {data}"
    extra = c.get("extra") or {}
    di = extra.get("DI") if isinstance(extra, dict) else None
    if isinstance(di, dict) and (di.get("x") or di.get("y")):
        out += f" DI({di.get('x')},{di.get('y')})"
    if isinstance(extra, dict):
        out += "".join(f" {k}" for k in ("feint", "reverse") if extra.get(k) is True)
    vals = c.get("values") or {}
    if vals.get("extra_set"):
        out += f" set{vals['extra_set']}"
    return out if len(out) <= 64 else out[:63] + "…"


def seat_state(p: dict[str, Any], actionable: bool) -> dict[str, Any]:
    raw = p.get("raw") or {}
    out = {"character": p.get("character"), "hp": p.get("hp"), "max_hp": p.get("max_hp"),
           "meter": p.get("meter"), "meter_partial": p.get("meter_partial"), "burst": p.get("burst"),
           "combo": p.get("combo"), "state": p.get("state"), "grounded": p.get("grounded"),
           "hitstun": p.get("hitstun") is not None, "actionable": bool(actionable), "x": p.get("x"), "y": p.get("y")}
    if p.get("character") == "Cowboy":
        out["has_gun"] = raw.get("has_gun")
        out["bullets"] = raw.get("bullets_left")
    return out


def waste_kind(c: dict[str, Any], player: dict[str, Any]) -> str | None:
    """The picks people read as wasted: Wait, Taunt, and Continue from an idle state."""
    if c["action"] == "Wait":
        return "wait"
    if c["action"] == "Taunt":
        return "taunt"
    if c["action"] == "Continue" and player.get("state") in IDLE_STATES:
        return "idle_continue"
    return None


def _hp(x: float, factor: float) -> float:
    return round(x * factor, 1)


def explain_seat(d: dict[str, Any], s: int, search: dict[str, Any], factor: float) -> dict[str, Any]:
    lists = search.get("candidates") or d["candidates"]  # a re-search may add candidates (see research)
    cands, strategy, values, v = lists[s], search["strategy"], search["values"], search["value"]
    pick = d["picks"][s]
    support = [i for i, p in enumerate(strategy) if p > 0]
    known = [i for i, x in enumerate(values) if x is not None]
    # The comparison is with other moves: a factor variant of a move in the mix (another DI, a feint)
    # is usually the same move to the net (DI matters only when hit), so it is reported apart.
    mixed = {cands[i]["action"] for i in support}
    outside = [i for i in known if strategy[i] <= 0 and cands[i]["action"] not in mixed]
    variants = [i for i in known if strategy[i] <= 0 and cands[i]["action"] in mixed]
    best_out = max(outside, key=lambda i: values[i]) if outside else None
    gap = round(v - values[best_out], 5) if best_out is not None else None
    variant_gap = round(v - max(values[i] for i in variants), 5) if variants else None
    flat = gap is not None and gap <= FLAT_TIE
    confident = gap is not None and gap >= CONFIDENT
    quick = d.get("search") == "cheap"
    explored = bool((d.get("explore") or [False, False])[s])
    mode = (d.get("explore_mode") or "uniform") if explored else None
    in_mix = strategy[pick] > 0
    pv = values[pick]
    ranked = sorted(known, key=lambda i: (-values[i], -strategy[i], i))
    shown = list(dict.fromkeys(ranked[:TOP] + sorted(support, key=lambda i: -strategy[i]) + [pick]))
    shown.sort(key=lambda i: (values[i] is None, -(values[i] if values[i] is not None else 0), -strategy[i], i))
    top = [{"i": i, "label": label(cands[i]), "value": values[i],
            "hp": _hp(values[i] - v, factor) if values[i] is not None else None,
            "p": strategy[i], "picked": i == pick} for i in shown]
    opp = search["opponent"]
    opp_cands = lists[1 - s]
    opp_mix = [{"i": j, "label": label(opp_cands[j]), "p": opp[j]}
               for j in sorted((j for j, p in enumerate(opp) if p > 0), key=lambda j: -opp[j])[:6]]
    category = ("explored" if explored else "quick" if quick else "flat_tie" if flat else "confident" if confident
                else "mixed" if len(support) > 1 else "moderate")
    player = d["state"]["players"][s]
    entry = {
        "n_candidates": len(cands), "n_known": len(known), "support": len(support), "value": v,
        "value_hp": _hp(v, factor),
        "pick": {"i": pick, "label": label(cands[pick]), "action": cands[pick]["action"], "p": strategy[pick],
                 "value": pv, "hp": _hp(pv - v, factor) if pv is not None else None},
        "best_outside": None if best_out is None else {
            "i": best_out, "label": label(cands[best_out]), "value": values[best_out], "gap": gap,
            "gap_hp": _hp(gap, factor)},
        "variant_gap": variant_gap,
        "flags": {"in_mix": in_mix, "flat_tie": flat, "confident": confident, "search": "quick" if quick else "full",
                  "explored": explored, "explore_mode": mode, "opening": bool(d.get("opening"))},
        "category": category, "waste": waste_kind(cands[pick], player),
        "top": top, "opponent_mix": opp_mix,
    }
    if "reproduced" in search:
        entry["reproduced"] = search["reproduced"]
    entry["reading"] = reading(entry)
    return entry


def reading(e: dict[str, Any]) -> str:
    """One plain-English line for a seat's decision."""
    line = _reading(e)
    if e.get("reproduced") is False:
        line = "(the re-search left the game's path here, so this mix isn't the one played) " + line
    return line


def _reading(e: dict[str, Any]) -> str:
    f, pick, bo = e["flags"], e["pick"]["label"], e["best_outside"]
    gap_hp = None if bo is None else abs(bo["gap_hp"])
    if f["explored"]:
        loss = e["pick"]["hp"]
        how = "uniformly at random" if f["explore_mode"] == "uniform" else "from a softmax of the values"
        return (f"exploration: {pick} was drawn {how}, not from the mix"
                + (f"; the net rated it {abs(loss):.0f} HP below the game value" if loss is not None and loss < -0.5 else ""))
    pre = "quick search (capped list, no refinement): " if f["search"] == "quick" else ""
    if not f["in_mix"]:
        return pre + f"{pick} is outside the solved mix"
    if bo is None:
        return pre + "only the mix's moves were evaluated, so nothing outside it can be compared"
    if f["flat_tie"]:
        return pre + (f"near-tie: the net can't tell these apart (the best move outside the mix, {bo['label']}, "
                      f"is {gap_hp:.0f} HP behind)")
    if e["support"] > 1:
        lead = f"{e['support']} moves mixed so the opponent can't exploit a habit"
        if f["confident"]:
            return pre + f"confident mix: {lead}; every other move is at least {gap_hp:.0f} HP worse"
        return pre + f"mixed: {lead}; the best other move is {gap_hp:.0f} HP behind"
    if f["confident"]:
        return pre + f"confident: the net prefers {pick} by {gap_hp:.0f} HP over any other move"
    return pre + f"moderate: {pick} leads the best other move ({bo['label']}) by {gap_hp:.0f} HP"


def explain_decision(d: dict[str, Any], searches: list[Any], factors: list[float]) -> dict[str, Any] | None:
    acting = [s for s in (0, 1) if d["actionable"][s] and len(d["candidates"][s]) > 1 and searches[s] is not None]
    if not acting:
        return None
    players = d["state"]["players"]
    return {"turn": d["turn"], "tick": d["state"].get("tick"),
            "search": "quick" if d.get("search") == "cheap" else "full",
            "distance": abs(players[0]["x"] - players[1]["x"]),
            "state": [seat_state(players[s], d["actionable"][s]) for s in (0, 1)],
            "seats": [explain_seat(d, s, searches[s], factors[s]) if s in acting else None for s in (0, 1)]}


def summarize(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in (0, 1):
        seats = [d["seats"][s] for d in decisions if d["seats"][s]]
        cats: dict[str, int] = {}
        waste: dict[str, dict[str, int]] = {}
        for e in seats:
            cats[e["category"]] = cats.get(e["category"], 0) + 1
            if e["waste"]:
                w = waste.setdefault(e["waste"], {"n": 0})
                w["n"] += 1
                for flag in ("flat_tie", "confident", "explored"):
                    w[flag] = w.get(flag, 0) + bool(e["flags"][flag])
                w["quick"] = w.get("quick", 0) + (e["flags"]["search"] == "quick")
                by = w.setdefault("by_category", {})
                by[e["category"]] = by.get(e["category"], 0) + 1
        out.append({"decisions": len(seats), "categories": cats,
                    "flat_tie": sum(e["flags"]["flat_tie"] for e in seats),
                    "confident": sum(e["flags"]["confident"] for e in seats), "waste": waste})
    return out


def explain(path: Path, overrides: dict[str, Path] | None = None, net_dirs: list[Path] | None = None,
            pricing: dict[str, Any] | None = None, make_engine: Callable[[list[Any]], Any] | None = None
            ) -> dict[str, Any]:
    t0 = time.perf_counter()
    header, decisions, result = read_record(path)
    cache: dict[str, Any] = {}

    def index() -> dict[str, list[Path]]:
        if "i" not in cache:
            cache["i"] = net_index(net_dirs if net_dirs is not None else default_net_dirs())
        return cache["i"]
    searches = recorded_searches(header, decisions)
    needs_nets = searches is None
    nets: list[dict[str, Any] | None] = []
    for cfg in header["bots"]:
        try:
            nets.append(resolve_net(cfg.get("value"), overrides or {}, index))
        except LookupError:
            if needs_nets:
                raise
            nets.append(None)  # only pricing needs it: the default factor
    recompute = None
    if searches is None:
        searches, recompute = research(header, decisions, result, nets, make_engine)
    factors, bots = [], []
    for s, cfg in enumerate(header["bots"]):
        f, source = hp_per_value(nets[s]["sha256"] if nets[s] else None, pricing)
        factors.append(f)
        bots.append({"seat": s, "bot": cfg.get("bot"), "value": cfg.get("value"), "search": cfg.get("search"),
                     "net": nets[s], "hp_per_value": f, "hp_source": source})
    explained = [e for e in (explain_decision(d, searches[k], factors) for k, d in enumerate(decisions)) if e]
    return {
        "tool": "explain_game", "version": 1, "record": path.name, "kind": "gate" if header.get("arena") else "selfplay",
        "characters": header["characters"], "seed": header["seed"], "bots": bots,
        "playout_cap": header.get("playout_cap"), "explore": header.get("explore"),
        "values": "recomputed" if recompute else "recorded", "recompute": recompute,
        "thresholds": {"flat_tie": FLAT_TIE, "confident": CONFIDENT, "top": TOP, "idle_states": list(IDLE_STATES),
                       "gap": "game value minus the best known candidate whose action is not in the mix"},
        "result": None if result is None else {k: result.get(k) for k in ("winner", "outcome", "end", "turns", "hp")},
        "decisions_total": len(decisions), "summary": summarize(explained), "decisions": explained,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _job(args: tuple) -> tuple[str, str]:
    path, out, overrides, net_dirs, pricing, force = args
    target = out / (path.name.split(".")[0] + ".explain.json")
    if target.exists() and not force:
        return path.name, f"exists: {target}"
    try:
        e = explain(path, overrides, net_dirs, pricing)
    except Exception as ex:  # noqa: BLE001  (reported per record)
        return path.name, f"FAILED: {type(ex).__name__}: {ex}"
    tmp = target.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(e, separators=(",", ":")))
    os.replace(tmp, target)
    rc = e["recompute"] or {}
    return path.name, (f"{len(e['decisions'])} decisions, values {e['values']}"
                       + (f" (picks reproduced {rc['picks_reproduced']}/{rc['seat_decisions']})" if rc else "")
                       + f", {e['seconds']}s -> {target}")


def _low_priority() -> None:
    try:
        if hasattr(os, "nice"):  # POSIX; Windows has none
            os.nice(19 - os.nice(0))
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True, help="folder for the <stem>.explain.json files")
    ap.add_argument("--net", action="append", default=[], metavar="NAME=PATH", help="the model for a header value name")
    ap.add_argument("--net-dir", action="append", type=Path, help="folders searched for nets by meta name "
                    "(default: round3/models, gather and checkpoints under $YOMIBENCH_DATA or ~/yomibench-data)")
    ap.add_argument("--pricing", type=Path, help="resource_pricing.json (default: its place under the data folder)")
    ap.add_argument("--procs", type=int, default=1, help=f"processes (at most {MAX_PROCS})")
    ap.add_argument("--force", action="store_true", help="redo records whose output exists")
    args = ap.parse_args(argv)
    _low_priority()
    overrides = {}
    for item in args.net:
        name, sep, p = item.partition("=")
        if not sep:
            ap.error(f"--net wants NAME=PATH: {item!r}")
        overrides[name.removeprefix("valuenet:")] = Path(p)
    pricing_path = args.pricing or data_root() / "coordination" / "results" / "resource_pricing.json"
    pricing = json.loads(pricing_path.read_text()) if pricing_path.is_file() else None
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, args.out, overrides, args.net_dir, pricing, args.force) for p in args.records]
    procs = max(1, min(args.procs, MAX_PROCS, len(jobs)))
    failed = 0
    if procs == 1:
        results = map(_job, jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(procs, initializer=_low_priority)
        results = pool.map(_job, jobs)
    for name, msg in results:
        failed += msg.startswith("FAILED")
        print(f"{name}: {msg}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

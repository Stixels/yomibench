"""Check double oracle against the full table on recorded turns (full-search records: format 2, or format 3 from --bot matrix).

For every searched turn: the full-table solve and double oracle on the same payoffs (evaluate on
the stored previewed states). Reports how far apart their values are, how exploitable each one's
strategies are in the FULL table, and how many previews double oracle would have needed.

    python tools/do_check.py RECORD_DIR [--tolerance 0.001]

With --net, compares solver settings on the recorded states of slim self-play records (round 3's
set-4 net self-play) instead: each sampled decision's state is reached on a batch engine by
replaying the record's picks (states checked against the record, as tools/render_record.py does),
then searched by the record's own bot (its header config: refinement, factor exploration, which
is seeded by match and turn) scoring with the given net, once per --settings TOL:FLOOR (the
solver's `tolerance` and `support_floor`), each on a fresh turn cache from the same snapshot. Per
character pairing it reports previews per turn, the |value difference| to the first setting
(median, p90, p99, max), support sizes, and the total-variation distance between strategies
(per seat over candidate identities, the larger seat). The first setting should be the
self-play one: it is checked to reproduce the recorded value and preview count.

    python tools/do_check.py RECORD_DIR --net MODEL.npz --settings 0.001:0.001,0.01:0.01 --decisions 300
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

from yomibench.search import double_oracle, evaluate, exploitability, solve_both
from yomibench.selfplay import read_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--start", type=int, default=2, help="random moves per side in the starting set")
    ap.add_argument("--net", type=Path, help="replay mode (see the module docstring): the value net to search with")
    ap.add_argument("--settings", default="0.001:0.001,0.01:0.01",
                    help="replay mode: comma-separated TOLERANCE:SUPPORT_FLOOR solver settings, the first the baseline")
    ap.add_argument("--decisions", type=int, default=300, help="replay mode: decisions to sample, split over pairings")
    ap.add_argument("--per-game", type=int, default=4, help="replay mode: at most this many decisions per game")
    ap.add_argument("--seed", type=int, default=0, help="replay mode: sampling seed")
    ap.add_argument("--json", type=Path, help="replay mode: write every decision's measurements here (JSONL)")
    args = ap.parse_args()
    if args.net:
        return replay_check(args)
    rows = []
    unconverged = 0
    for path in sorted(args.dir.glob("g*.jsonl.gz")):
        for t in read_record(path)[1:-1]:
            n0, n1 = (len(c) for c in t["candidates"])
            if t.get("table"):  # format 2
                states = {(i, j): t["table"][i * n1 + j] for i in range(n0) for j in range(n1)}
            else:  # format 3; needs a full search's record
                states = {(i, j): st for i, j, st in t.get("previewed") or []}
                if len(states) != n0 * n1:
                    continue
            m = [[evaluate(states[i, j], 0) for j in range(n1)] for i in range(n0)]
            r, c, v = solve_both(m, args.iterations)
            t0 = time.perf_counter()
            rng = random.Random(f"{t['turn']}:{t['state'].get('tick')}")
            start = tuple([0] + rng.sample(range(1, n), min(args.start, n - 1)) for n in (n0, n1))
            res = double_oracle(n0, n1, lambda pairs: [m[i][j] for i, j in pairs], args.iterations,
                                args.tolerance, start=start)
            dr, dc, dv, known, rounds = res.row, res.col, res.value, res.known, res.rounds
            unconverged += not res.converged
            rows.append({
                "n": n0 * n1, "previews": len(known), "rounds": rounds, "dv": abs(dv - v),
                "full_gap": max(exploitability(m, r, c)), "do_gap": max(exploitability(m, dr, dc)),
                "ms": (time.perf_counter() - t0) * 1000,
            })
    if not rows:
        sys.exit("no turns with tables")
    big = [x for x in rows if x["n"] >= 400]
    tot_full = sum(x["n"] for x in rows)
    tot_do = sum(x["previews"] for x in rows)
    print(f"{len(rows)} turns ({len(big)} with >= 400 pairs)")
    print(f"previews: full {tot_full}, double oracle {tot_do} ({tot_do / tot_full:.0%}); "
          f"on big turns {sum(x['previews'] for x in big)}/{sum(x['n'] for x in big)}")
    print(f"value difference: median {statistics.median(x['dv'] for x in rows):.5f}, max {max(x['dv'] for x in rows):.5f}")
    print(f"exploitability in the full table (the larger of the two sides' best-reply gains):")
    print(f"  full-table solve: median {statistics.median(x['full_gap'] for x in rows):.5f}, max {max(x['full_gap'] for x in rows):.5f}")
    print(f"  double oracle:    median {statistics.median(x['do_gap'] for x in rows):.5f}, max {max(x['do_gap'] for x in rows):.5f}")
    print(f"turns where double oracle is worse than the full solve by > tolerance: "
          f"{sum(x['do_gap'] > x['full_gap'] + args.tolerance for x in rows)}")
    print(f"not converged within max_rounds: {unconverged}")
    print(f"rounds: median {statistics.median(x['rounds'] for x in rows)}, max {max(x['rounds'] for x in rows)}; "
          f"solver ms median {statistics.median(x['ms'] for x in rows):.0f}")


def _sample(paths: list[Path], total: int, per_game: int, seed: int) -> dict[Path, set[int]]:
    """{record: decision indices} for about `total` full searched decisions, split evenly over the
    character pairings, at most `per_game` from one game (seeded)."""
    import gzip
    import json

    rng = random.Random(seed)
    games: dict[tuple, list[tuple[Path, list[int]]]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f]
        full = [k for k, t in enumerate(lines[1:-1]) if t.get("strategies") and t.get("search", "full") == "full"]
        games.setdefault(tuple(lines[0]["characters"]), []).append((path, full))
    quota = -(-total // max(1, len(games)))
    chosen: dict[Path, set[int]] = {}
    for _, gs in sorted(games.items()):
        rng.shuffle(gs)
        left = quota
        for path, full in gs:
            if left <= 0:
                break
            take = rng.sample(full, min(per_game, len(full), left))
            if take:
                chosen[path] = set(take)
                left -= len(take)
    return chosen


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _pick_ids(engine, obs, t: dict) -> list[int]:
    """Engine ids of a decision's recorded picks. Base candidates are found by identity in the
    engine's lists (a playout-capped decision records only its capped list, so a pick's position
    isn't its id); expanded ones (with `base`) are rebuilt with `expand`."""
    from yomibench.engine import Candidate

    ids = []
    for seat in (0, 1):
        c = t["candidates"][seat][t["picks"][seat]]
        if "base" in c:
            (got,) = engine.expand([{"seat": seat, "candidate": c["base"], "extra_override": c["values"]}])
            if not isinstance(got, Candidate) or got.record() != c:
                raise ValueError(f"seat {seat}: the recorded expanded pick didn't rebuild ({got})")
            ids.append(got.id)
        else:
            ids.append(next(e.id for e in obs.candidates[seat]
                            if (e.action, e.data, e.extra) == (c["action"], c["data"], c["extra"])))
    return ids


def replay_check(args) -> None:
    import functools
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_record import read_record as load_record, state_diff

    from yomibench import bots as bots_mod, search
    from yomibench.plugins import batch
    from yomibench.bots import FactorExplore, MatrixBot, turn_candidates
    from yomibench.valuenet import ValueNet

    BatchEngine, BatchValue = batch().BatchEngine, batch().BatchValue
    settings = [tuple(float(x) for x in s.split(":")) for s in args.settings.split(",")]
    net = BatchValue(ValueNet.load(args.net))
    chosen = _sample(sorted(args.dir.glob("g*.jsonl.gz")), args.decisions, args.per_game, args.seed)
    engine = BatchEngine(1, net=net.net, candidate_set=4)
    out = args.json.open("w") if args.json else None
    rows, skipped = [], []
    plain_floor = search.double_oracle
    t_start = time.perf_counter()
    try:
        for path, want in sorted(chosen.items()):
            head, turns, _ = load_record(path)
            if head["engine"].get("candidate_rules") != engine.hello().get("candidate_rules"):
                skipped.append((path.name, "candidate rules differ"))
                continue
            cfg = head["bots"][0]
            obs = engine.new_match(*head["characters"], head["seed"])
            for k, t in enumerate(turns):
                if k > max(want):
                    break
                if k in want:
                    diff = state_diff(t["state"], obs.state)
                    if diff:
                        skipped.append((path.name, f"decision {k}: state differs ({diff[0]})"))
                        break
                    snap = engine.batch.snapshot([0])
                    res = []
                    for tol, floor in settings:
                        engine.batch.restore(snap, [0])
                        engine.turn_cache, engine._previewed = {}, set()
                        bot = MatrixBot(iterations=cfg["iterations"], value=net, search="oracle", tolerance=tol,
                                        oracle_start=cfg["oracle_start"], refine_rounds=cfg["refine_rounds"],
                                        di_refine_moves=cfg["di_refine_moves"],
                                        explore=FactorExplore(**cfg["explore"]) if cfg.get("explore") else None)
                        bots_mod.double_oracle = functools.partial(plain_floor, support_floor=floor)
                        t0 = time.perf_counter()
                        try:
                            row, col, value = bot.search_turn(obs, engine)
                        finally:
                            bots_mod.double_oracle = plain_floor
                        cands = turn_candidates(obs, engine, bot._key())
                        strat = [{json.dumps(c.record(), sort_keys=True): p for c, p in zip(cands[s], st) if p > 0}
                                 for s, st in enumerate((row, col))]
                        res.append({"value": value, "previews": len(engine.turn_cache.get("previewed", {})),
                                    "support": [len(x) for x in strat], "strat": strat,
                                    "ms": (time.perf_counter() - t0) * 1000})
                    engine.batch.restore(snap, [0])
                    engine.turn_cache, engine._previewed = {}, set()
                    base = res[0]
                    tv = [max(0.5 * sum(abs(r["strat"][s].get(c, 0.0) - base["strat"][s].get(c, 0.0))
                                        for c in set(r["strat"][s]) | set(base["strat"][s])) for s in (0, 1))
                          for r in res]
                    row_out = {"record": path.name, "decision": k, "turn": t["turn"],
                               "pairing": ":".join(head["characters"]),
                               "recorded": {"value": t.get("value"), "previews": t.get("previews")},
                               "settings": [{"tolerance": tol, "support_floor": floor, "value": r["value"],
                                             "dv": abs(r["value"] - base["value"]), "previews": r["previews"],
                                             "support": r["support"], "tv": d, "ms": round(r["ms"])}
                                            for (tol, floor), r, d in zip(settings, res, tv)]}
                    rows.append(row_out)
                    if out:
                        out.write(json.dumps(row_out) + "\n")
                        out.flush()
                engine.turn_cache = {}
                obs = engine.step(*_pick_ids(engine, obs, t))
    finally:
        if out:
            out.close()
    if not rows:
        sys.exit(f"no decisions measured; skipped: {skipped}")
    print(f"{len(rows)} decisions from {len(chosen)} games in {time.perf_counter() - t_start:.0f}s; "
          f"skipped {len(skipped)} games" + (f" (e.g. {skipped[0]})" if skipped else ""))
    same = sum(round(r["settings"][0]["value"], 5) == r["recorded"]["value"]
               and r["settings"][0]["previews"] == r["recorded"]["previews"] for r in rows)
    print(f"baseline setting reproduces the recorded value and preview count on {same}/{len(rows)} decisions")
    pairings = sorted({r["pairing"] for r in rows}) + ["all"]
    for si, (tol, floor) in enumerate(settings):
        print(f"\nsetting tolerance {tol:g}, support_floor {floor:g}" + (" (baseline)" if si == 0 else ""))
        print(f"  {'pairing':<14}{'n':>4}{'previews/turn':>15}{'vs base':>9}{'|dv| med':>10}{'p90':>9}{'p99':>9}"
              f"{'max':>9}{'support/seat':>14}{'TV med':>8}{'p90':>7}{'max':>7}{'ms/turn':>9}")
        for pg in pairings:
            sel = [r for r in rows if pg in ("all", r["pairing"])]
            st = [r["settings"][si] for r in sel]
            bp = sum(r["settings"][0]["previews"] for r in sel)
            pv = sum(x["previews"] for x in st)
            dv, tv = [x["dv"] for x in st], [x["tv"] for x in st]
            sup = sum(sum(x["support"]) for x in st) / (2 * len(st))
            print(f"  {pg:<14}{len(sel):>4}{pv / len(sel):>15.0f}{pv / bp:>9.2f}{statistics.median(dv):>10.5f}"
                  f"{_pct(dv, 0.9):>9.5f}{_pct(dv, 0.99):>9.5f}{max(dv):>9.5f}{sup:>14.2f}"
                  f"{statistics.median(tv):>8.3f}{_pct(tv, 0.9):>7.3f}{max(tv):>7.3f}"
                  f"{sum(x['ms'] for x in st) / len(st):>9.0f}")


if __name__ == "__main__":
    main()

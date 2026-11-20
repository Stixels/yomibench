from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from yomibench import ledger
from yomibench.arena import bradley_terry, bradley_terry_ci, rated_pairs, run_round_robin, win_matrix
from yomibench.bots import make_bot
from yomibench.engine import connect, expand_specs, set5_spec
from yomibench.playout import CAP_POLICIES
from yomibench.selfplay import DEFAULT_CANDIDATE_SET, parse_category


def shard_seeds(seeds, k: int, n: int, n_matchups: int) -> list[int]:
    """The seeds `--shard K/N` child k of n plays when it doesn't claim seeds (--procs children
    claim them: see _selfplay_procs). Seed s plays matchups[s % n_matchups], whoever plays it.
    Seeds are dealt round-robin, so a run of seeds shorter than a long weighted cycle still keeps
    every child busy, and the deal shifts by one child every lcm(n, n_matchups) seeds, so each
    child plays every matchup within about n cycles (plain `s % n` gives each child a single
    matchup when n shares a factor with the matchup count: 8 children, 4 matchups). Over any run
    of consecutive seeds the children's loads differ by at most 2 (1 can't be had with that)."""
    m = max(1, n_matchups)
    period = n * m // math.gcd(n, m)
    return [s for s in seeds if (s + s // period) % n == k]


MATCH_SETTINGS_HELP = (
    "match settings for every game, as a JSON object of hello's match_settings keys, e.g. the ranked preset "
    "'{\"stage_width\":600,\"game_length\":3000,\"starting_meter\":1.5}'. Records carry the settings played "
    "(header engine.match_settings), so they never train with standard-settings records. The batch engine "
    "needs an engine plugin with batch settings. Default: the engine's standard settings")

SET5_HELP = ("candidate set 5's switches for the sim and batch engines, as $YOMISIM_SET5 takes them (comma-separated "
             "name=0|1 of draw_pairs, gun_aim_fine, bullet_di, draw_base; unnamed ones stay on), or 'all'; sets "
             "$YOMISIM_SET5 for this run. The record header's candidate_rules names what was played. Default: "
             "$YOMISIM_SET5 as it is")


def _match_settings(text: str) -> dict:
    """--match-settings: a JSON object of match settings."""
    try:
        settings = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not JSON: {e}") from e
    if not isinstance(settings, dict):
        raise argparse.ArgumentTypeError("must be a JSON object, e.g. '{\"stage_width\":600}'")
    return settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="yomibench")
    sub = p.add_subparsers(dest="cmd", required=True)

    arena = sub.add_parser("arena", help="round-robin the given bots and print ratings")
    arena.add_argument("--engine", default="toy",
                       help="toy | tcp:HOST:PORT[-PORT] | game | cmd:COMMAND | @POOLFILE; comma-separate "
                            "several to play games in parallel")
    arena.add_argument("--bots", default="random,greedy,matrix")
    arena.add_argument("--characters", default="Toy", help="comma-separated character names")
    arena.add_argument("--pairs", type=int, default=10, help="seat-swapped pairs per matchup")
    arena.add_argument("--seed", type=int, default=1)
    arena.add_argument("--results", type=Path, help="append each game's result as JSONL (was --ledger)")
    arena.add_argument("--record-dir", type=Path, metavar="DIR",
                       help="also write every game as a format-3 record in DIR (per-turn states and picks, as "
                            "self-play records; tools/render_record.py replays them); decisions are unchanged")
    arena.add_argument("--match-settings", type=_match_settings, metavar="JSON", help=MATCH_SETTINGS_HELP)
    arena.add_argument("--set5", metavar="SWITCHES", help=SET5_HELP)
    ledger.add_args(arena)
    arena.add_argument("--max-turns", type=int, default=1000,
                       help="a game still running after this many turns is truncated (not rated)")

    serve = sub.add_parser("serve", help="run real-game engine servers (needs the game installed)")
    serve.add_argument("--count", type=int, default=1, help="number of server processes")
    serve.add_argument("--port", type=int, default=7777, help="first port; servers use consecutive ports")
    serve.add_argument("--game-dir", help="game install (default: $YOMIBENCH_GAME_DIR or the Steam libraries)")
    serve.add_argument("--max-candidates", type=int, help="per-seat candidate budget (server default 0: every variant)")
    serve.add_argument("--run-dir", type=Path, help="scratch dir for the game copy (default ~/.yomibench/run)")
    serve.add_argument("--host", help='bind address (default 127.0.0.1; "*" for all interfaces, e.g. in Docker)')
    serve.add_argument("--divergence-log", action="store_true",
                       help="log turns where the preview of the played pair differed from the step")
    serve.add_argument("--candidate-rules",
                       help="serve the candidate set a candidate_rules hash names (set 4's or a set-5 switch "
                            "combination's, e.g. a record header's); default set 4")
    serve.add_argument("--set5", help='set 5\'s switches for seats on set "5": all (default), none, or the ones on, '
                                      "comma-separated (draw_pairs,gun_aim_fine,bullet_di,draw_base)")

    rd = sub.add_parser("render", help="render replay files to MP4 with the real game (needs a display and ffmpeg)")
    rd.add_argument("replays", type=Path, nargs="+")
    rd.add_argument("--out", type=Path, required=True, help="folder for the videos")
    rd.add_argument("--game-dir", help="game install (default: $YOMIBENCH_GAME_DIR or the Steam libraries)")
    rd.add_argument("--tail", type=int, default=150, help="frames to keep after the match ends (60 per second)")
    rd.add_argument("--keep-frames", action="store_true")
    rd.add_argument("--states", action="store_true",
                    help="no video: log every tick's fighters and objects to OUT/<stem>.states.jsonl")

    sp = sub.add_parser("selfplay", help="record self-play games for training (one .jsonl.gz per game)")
    sp.add_argument("--engine", default="toy",
                    help="same forms as arena --engine, or batch[:THREADS] (the sim's native Batch API; see "
                         "yomibench.batch_play; it plays --candidate-set like sim); one game at a time per spec")
    sp.add_argument("--bot", default="matrix-do",
                    help="matrix-do (double oracle), matrix (full table), or either with :MODEL.npz for a value net")
    sp.add_argument("--matchups", default="Toy:Toy",
                    help="comma-separated P1:P2 pairs; game SEED plays matchups[SEED %% count]")
    sp.add_argument("--games", type=int, default=10)
    sp.add_argument("--first-seed", type=int, default=0,
                    help="seeds FIRST..FIRST+GAMES-1; give each machine its own range when sharing a folder")
    sp.add_argument("--out", type=Path, required=True, help="folder for game files (reruns skip finished seeds)")
    sp.add_argument("--max-candidates", type=int, help="per-seat budget for these matches (0: every variant)")
    sp.add_argument("--candidate-set", type=int, default=DEFAULT_CANDIDATE_SET, metavar="N",
                    help="candidate set every match plays (e.g. 4; default: the engine's own). The sim is built "
                         "with it; other engines are asked for it per match. Records carry it in the header")
    sp.add_argument("--match-settings", type=_match_settings, metavar="JSON", help=MATCH_SETTINGS_HELP)
    sp.add_argument("--set5", metavar="SWITCHES", help=SET5_HELP)
    sp.add_argument("--ecoqos", action="store_true",
                    help="Windows: keep the default power throttling (EcoQoS) for these processes; by default "
                         "they opt out, so Windows 11 runs them on P-cores as well as E-cores (yomibench.winsched)")
    sp.add_argument("--procs", type=int, default=1,
                    help="worker processes, each playing its own share of the seeds; for the in-process sim "
                         "use about one per core (threads can't share the sim: it holds Python's GIL)")
    sp.add_argument("--full-previews-fraction", type=float, default=0.01, metavar="F",
                    help="share of games that keep each turn's previewed states (most of a record's size), chosen "
                         "per game from its seed; the rest are slim, with everything training reads (default 0.01)")
    sp.add_argument("--batched-net-forward", action="store_true",
                    help="batch engine: score both seats' views of a preview batch in one value-net forward "
                         "(faster; NOT bit-identical to the default two forwards, so search can pick differently; "
                         "bots record value_forward: batched). Default off")
    sp.add_argument("--no-previews", action="store_true",
                    help="no game keeps previewed states: the same as --full-previews-fraction 0")
    sp.add_argument("--full-previews-max-pairs", type=int, default=20000, metavar="N",
                    help="in a game that keeps previewed states, a turn that previews more than N pairs is "
                         "recorded slim instead, marked \"previews_dropped\" (a rare expand-everything turn "
                         "previews hundreds of thousands: one record line of GBs); the batch engine then asks "
                         "for no state dicts. 0: no cap (default 20000)")
    sp.add_argument("--opening-random-turns", type=int, default=0, metavar="N",
                    help="opening noise for each seat's first N decisions with a choice, marked \"opening\" "
                         "in the record (default 0: off); see --opening-epsilon and --opening-temperature")
    sp.add_argument("--opening-epsilon", type=float, default=0.0, metavar="E",
                    help="with probability E an opening pick is uniform over all candidates, off the solved "
                         "support (recorded in \"explore\"); default 0")
    sp.add_argument("--explore-epsilon", type=float, default=0.05, metavar="E",
                    help="per decision, with probability E one seat samples a softmax over its candidates' values "
                         "against the opponent's mix (recorded in \"explore\", explore_mode softmax); 0 disables")
    sp.add_argument("--explore-temperature", type=float, default=0.05, metavar="T",
                    help="softmax temperature for --explore-epsilon, in value units (default 0.05)")
    sp.add_argument("--explore-category", default=None, metavar="NAME:E",
                    help="category exploration (the draw arm): with probability E per eligible full-turn decision, "
                         "a seat plays a uniform pick among its NAME candidates (selfplay.CATEGORIES, e.g. "
                         "cowboy-gun:0.15); recorded as explore_mode category:NAME. Default off")
    sp.add_argument("--explore-uniform-moves", type=float, default=0.0, metavar="E",
                    help="generic exploration: with probability E per full-turn decision, one seat at random "
                         "picks a move uniformly among its distinct actions, then a candidate of it; recorded as "
                         "explore_mode uniform-moves. No move or character is named. Default 0 (off)")
    sp.add_argument("--explore-least-tried", default="", metavar="JSON",
                    help="least-tried move-type exploration (yomibench.least_tried; generic, no move or character is "
                         "named), e.g. '{\"p\": 0.03, \"k\": 2, \"window_games\": 20000}': with probability p per "
                         "full-turn decision outside the opening window, at most k times per game per seat, a seat "
                         "plays its least-picked legal move type, then within it the least-picked value of each option, "
                         "by --least-tried-counts; recorded as "
                         "explore_mode least-tried with explore_pick. \"games\": G (0 < G <= 1, default 1) limits "
                         "it to a seeded share G of games (by seed; the header's selected says which). Default off")
    sp.add_argument("--least-tried-counts", type=Path, metavar="PATH",
                    help="the counts file --explore-least-tried reads (least_tried.refresh_counts / write_counts; "
                         "the round-3 worker writes one per chunk); its sha256 goes into every record's header")
    sp.add_argument("--sampled-starts", default="", metavar="JSON",
                    help="sampled-state starts (yomibench.sampled_starts), e.g. '{\"share\": 0.1, \"window_hours\": 12}': "
                         "a seeded share of game pairs (seeds 2m, 2m+1) start from a state sampled from the "
                         "--sampled-starts-pool records written in the last window_hours whose engine rules and match "
                         "settings are this engine's (fast_forward, checked against the record's state; on a mismatch "
                         "the game plays from the opening). Header start.kind sampled. Default off")
    sp.add_argument("--sampled-starts-pool", type=Path, action="append", default=[], metavar="DIR",
                    help="a folder of source records for --sampled-starts (repeatable; the round-3 worker passes its "
                         "DATA_ROOT/start-pool)")
    sp.add_argument("--factor-margin", type=float, default=0.02, metavar="M",
                    help="set 4: also refine factors of off-support candidates within M of the game value")
    sp.add_argument("--factor-random", type=int, default=2, metavar="K",
                    help="set 4: also refine factors of K seeded random off-support candidates per seat")
    sp.add_argument("--factor-all", type=float, default=0.03, metavar="P",
                    help="set 4: on a share P of decisions refine every candidate's factors, logging the value "
                         "gap found (oracle.explore.gap); --factor-margin -1 --factor-random 0 --factor-all 0 "
                         "turns factor exploration off")
    sp.add_argument("--cheap-fraction", type=float, default=0.0, metavar="P",
                    help="playout cap randomization: per decision, with probability P (seeded by the match seed "
                         "and turn) a cheap search: --cheap-cap candidates per seat, no factor refinement or "
                         "exploration; marked \"search\" in the record, and training uses its search value "
                         "for nothing (default 0: every decision full)")
    sp.add_argument("--cheap-cap", type=int, default=32, metavar="N",
                    help="per-seat candidate budget of a cheap decision, by the engine's budget rule (default 32)")
    sp.add_argument("--cheap-cap-policy", default="list", choices=CAP_POLICIES,
                    help="which candidates a cheap decision keeps (playout.cap_candidates): list (default; the "
                         "engine's budget rule), options-first (every mode-widget option of a move before a "
                         "second aim), rotate or options-rotate (seeded rotations), options-shuffle (options-first with a "
                         "seeded choice of each option's aim/direction/slider variant); recorded in the header's "
                         "playout_cap as cheap_cap_policy unless list")
    sp.add_argument("--opening-temperature", type=float, default=1.0, metavar="T",
                    help="other opening picks sample the solved strategy with weights probability ** (1/T) "
                         "(default 1: the strategy; higher is flatter within the support, 0 keeps maxima)")
    sp.add_argument("--max-proc-mem-gb", type=float, default=0.0, metavar="G",
                    help="per-process memory cap: a self-play process whose memory (private/commit bytes on "
                         "Windows, resident elsewhere) passes G GB drops the game(s) it is playing (logged as "
                         "dropped, never replayed in the run) and is replaced, instead of exhausting the "
                         "machine; plays in a child process even with --procs 1 (yomibench.memcap). "
                         "Default 0 (off)")
    sp.add_argument("--shard", help=argparse.SUPPRESS)  # K/N: this process plays shard_seeds(..., K, N, ...)
    sp.add_argument("--stats-out", type=Path, help=argparse.SUPPRESS)  # shard: write ledger counters here
    sp.add_argument("--claim-dir", type=Path, help=argparse.SUPPRESS)  # shard: take seeds by claiming them here
    ledger.add_args(sp)
    sp.add_argument("--allow-sleep", action="store_true",
                    help="let the machine sleep during the run (by default it is kept awake until the run ends)")

    lg = sub.add_parser("ledger", help="summarize or validate the metrics ledger (see src/yomibench/ledger.py)",
                        add_help=False)
    lg.add_argument("rest", nargs=argparse.REMAINDER)
    for name, text in (("bundle", "pack or check a model bundle for the ladder (docs/MATCHES.md)"),
                       ("ladder", "a persistent Elo ladder of model bundles (docs/MATCHES.md)")):
        sp_ = sub.add_parser(name, help=text, add_help=False)
        sp_.add_argument("rest", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)
    if args.cmd == "ledger":
        return ledger.cli(args.rest)
    if args.cmd in ("bundle", "ladder"):
        from yomibench import ladder as ladder_mod

        return (ladder_mod.bundle_cli if args.cmd == "bundle" else ladder_mod.ladder_cli)(args.rest)
    if args.cmd in ("selfplay", "arena"):
        try:
            tags = ledger.parse_tags(args.tag)
        except ValueError as e:
            p.error(str(e))
        ledger_path = ledger.path_from_args(args)
        started, load = ledger.utc_now(), (ledger.cpu_load() if ledger_path else None)
    if args.cmd == "arena" and ledger_path and ledger_path.exists():
        with open(ledger_path, encoding="utf-8") as f:
            first = f.readline()
        if first.strip() and '"schema"' not in first:
            p.error(f"{ledger_path} holds per-game arena results, not metrics rows: arena --ledger is now the "
                    "metrics ledger; use --results for per-game results")
    if args.cmd in ("selfplay", "arena") and args.set5:
        try:
            os.environ["YOMISIM_SET5"] = set5_spec(args.set5)  # the engines read it; --procs children inherit it
        except ValueError as e:
            p.error(f"--set5: {e}")
    if args.cmd == "selfplay":
        if args.opening_random_turns < 0:
            p.error("--opening-random-turns must be nonnegative")
        if not math.isfinite(args.opening_temperature) or args.opening_temperature < 0:
            p.error("--opening-temperature must be finite and nonnegative")
        if not 0 <= args.opening_epsilon <= 1:
            p.error("--opening-epsilon must be between 0 and 1")
        if not 0 <= args.cheap_fraction <= 1:
            p.error("--cheap-fraction must be between 0 and 1")
        if args.cheap_cap < 1:
            p.error("--cheap-cap must be positive")
        if not math.isfinite(args.max_proc_mem_gb) or args.max_proc_mem_gb < 0:
            p.error("--max-proc-mem-gb must be finite and nonnegative")
        if args.full_previews_max_pairs < 0:
            p.error("--full-previews-max-pairs must be nonnegative")
        if args.explore_least_tried:
            _least_tried(args)  # checks the settings and the counts file before any child starts
        try:
            _sampled_starts(args)  # checks the settings before any child starts
        except ValueError as e:
            p.error(f"--sampled-starts: {e}")
    if args.cmd == "selfplay":
        import time

        from yomibench.selfplay import run_selfplay

        # Self-play runs one search per process; a math library starting a thread per core in each
        # of many processes left them fighting over the cores (25x slower, ~850 MB each on Windows).
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "RAYON_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        if not args.ecoqos:  # every process: --procs children run this same branch
            from yomibench.winsched import opt_out_of_ecoqos

            opt_out_of_ecoqos()
        specs = expand_specs(args.engine)
        t0 = time.perf_counter()
        if not args.shard:  # once per run (shards inherit the same environment)
            from yomibench.search import SOLVER

            print(f"solver: {SOLVER}" + ("" if SOLVER == "native" else
                  " (pure Python; install native/yomisolve for a 30-130x faster solver, same results)"), flush=True)
        if (args.procs > 1 or args.max_proc_mem_gb > 0) and not args.shard:
            from yomibench.selfplay import done_seeds

            # Seeds recorded before the children start: counted here once. Claiming children can't
            # tell them from seeds a sibling finished a moment ago, so they count no skipped seeds.
            run = range(args.first_seed, args.first_seed + args.games)
            before = len(done_seeds(args.out) & set(run)) if args.out.is_dir() else 0
            code, stats = _selfplay_procs(argv if argv is not None else sys.argv[1:], args.procs, specs,
                                          collect=ledger_path is not None,
                                          mem_cap=(args.max_proc_mem_gb, args.out, run)
                                          if args.max_proc_mem_gb > 0 else None)
            if stats:
                stats["skipped"] = before
            if ledger_path:
                _selfplay_row(args, ledger_path, tags, stats, time.perf_counter() - t0, len(specs), started, load,
                              interrupted=code == 130)
            return code
        if not args.allow_sleep:
            _keep_awake()
        matchups = [tuple(m.split(":")) for m in args.matchups.split(",")]
        stats = ledger.new_selfplay_stats()

        def progress(seed, lines, error):
            if lines is None:
                print(f"seed {seed}: dropped ({error})", flush=True)
                return
            r = lines[-1]["result"]
            print(f"seed {seed}: {lines[0]['characters']} {r['outcome']} in {r['turns']} turns, "
                  f"{r['seconds']}s", flush=True)

        seeds = range(args.first_seed, args.first_seed + args.games)
        claim = owns = None
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            mine = shard_seeds(seeds, k, n, len(matchups))
            if args.claim_dir:  # every child walks all seeds and plays those it claims first
                claim = _seed_claimer(args.claim_dir)
                owns = lambda s: False  # noqa: E731  (the parent counts skipped seeds)
            else:
                seeds = mine
        cap = None
        if args.max_proc_mem_gb > 0 and args.claim_dir:  # a child of _selfplay_procs
            from yomibench.memcap import MemoryCap

            cap = MemoryCap(args.max_proc_mem_gb, args.claim_dir,
                            before_exit=lambda: _selfplay_finish(args, None, tags, stats, time.perf_counter() - t0,
                                                                 len(specs), started, load)).start()
        connects = [lambda spec=spec: connect(spec, args.candidate_set, args.match_settings) for spec in specs]
        bot_factory = lambda: _selfplay_bot(args)  # noqa: E731
        if any(s.split(":", 1)[0] == "batch" for s in specs):
            try:
                from yomibench.batch_play import batch_selfplay_parts
            except ModuleNotFoundError as e:
                if e.name != "yomibench.batch_play":
                    raise
                raise SystemExit("engine 'batch' isn't available in this build; use game-pool:auto (see docs/SETUP.md)")

            connects, bot_factory = batch_selfplay_parts(specs, bot_factory, args.max_candidates,
                                                         _previews_fraction(args), args.candidate_set,
                                                         args.full_previews_max_pairs, args.match_settings,
                                                         batched_net_forward=args.batched_net_forward)
        try:
            written, dropped = run_selfplay(
                connects, bot_factory, matchups,
                seeds, args.out, args.max_candidates, progress, full_previews_fraction=_previews_fraction(args),
                opening=((args.opening_random_turns, args.opening_temperature, args.opening_epsilon)
                         if args.opening_random_turns else None), stats=stats,
                explore_cfg=(args.explore_epsilon, args.explore_temperature) if args.explore_epsilon > 0 else None,
                playout_cap=(((args.cheap_fraction, args.cheap_cap)
                              + ((args.cheap_cap_policy,) if args.cheap_cap_policy != "list" else ()))
                             if args.cheap_fraction > 0 else None),
                explore_category=parse_category(args.explore_category) if args.explore_category else None,
                explore_uniform_moves=args.explore_uniform_moves if args.explore_uniform_moves > 0 else None,
                full_previews_max_pairs=args.full_previews_max_pairs,
                explore_least_tried=_least_tried(args),
                claim=claim, owns=owns, playing=cap.playing if cap else None, guard=cap.lock if cap else None,
                **({"sampled_starts": pool} if (pool := _sampled_starts(args)) is not None else {}),
            )
        except KeyboardInterrupt:
            print("interrupted; finished games are kept, rerun the same command to resume", flush=True)
            _selfplay_finish(args, ledger_path, tags, stats, time.perf_counter() - t0, len(specs), started, load,
                             interrupted=True)
            return 130
        _selfplay_finish(args, ledger_path, tags, stats, time.perf_counter() - t0, len(specs), started, load)
        mins = (time.perf_counter() - t0) / 60
        print(f"{written} games written, {dropped} dropped, {mins:.1f} min "
              f"({written / max(mins, 1e-9) * 60:.0f} games/hour on {len(specs)} engines)")
        if stats.get("sampled_starts"):
            s = stats["sampled_starts"]
            print("sampled starts: " + ", ".join(f"{k} {v}" for k, v in s.items()), flush=True)
        return 0
    if args.cmd == "render":
        from yomibench.render import render

        render(args.replays, args.out, args.game_dir, tail=args.tail, keep_frames=args.keep_frames, states=args.states)
        return 0
    if args.cmd == "serve":
        from yomibench.game import serve as run_servers

        # The servers read these options from their environment (YomiServer._option).
        if args.candidate_rules:
            os.environ["YOMI_CANDIDATE_RULES"] = args.candidate_rules
        if args.set5:
            os.environ["YOMI_SET5"] = args.set5
        run_servers(args.count, args.port, args.game_dir, args.max_candidates, run_dir=args.run_dir,
                    divergence_logs=args.divergence_log, host=args.host)
        return 0
    if args.cmd == "arena":
        bots = [make_bot(spec) for spec in args.bots.split(",")]
        specs = expand_specs(args.engine)
        hellos: list[dict] = []  # the first engine's hello, for the ledger row
        batch = None
        if any(s.split(":", 1)[0] == "batch" for s in specs):
            try:
                from yomibench.batch_play import batch_arena_engines
            except ModuleNotFoundError as e:
                if e.name != "yomibench.batch_play":
                    raise
                raise SystemExit("engine 'batch' isn't available in this build; use game-pool:auto (see docs/SETUP.md)")

            # every bot's net, each seat scored on its own features
            batch = batch_arena_engines(bots, match_settings=args.match_settings)

        def make(spec: str):
            engine = (batch(spec) if batch and spec.split(":", 1)[0] == "batch"
                      else connect(spec, match_settings=args.match_settings))
            if not hellos:
                try:
                    hellos.append(engine.hello())
                except Exception:  # noqa: BLE001  (only provenance)
                    hellos.append({})
            return engine
        n_chars = len(args.characters.split(","))
        scheduled = len(bots) * (len(bots) - 1) // 2 * n_chars * n_chars * args.pairs * 2
        results = run_round_robin(
            [lambda spec=spec: make(spec) for spec in specs],
            bots,
            args.characters.split(","),
            args.pairs,
            args.seed,
            args.results,
            on_result=lambda r: print(f"{r.bots[0]:>8} v {r.bots[1]:<8} {r.outcome:>4}  turns={r.turns} {r.seconds}s"),
            max_turns=args.max_turns,
            record_dir=args.record_dir,
        )
        print()
        outcomes = Counter(r.outcome for r in results)
        print(f"{len(results)} games: {dict(outcomes)}")
        if outcomes["truncated"] or outcomes["halt"]:
            print(f"excluded from scores: {outcomes['truncated']} truncated at {args.max_turns} turns, "
                  f"{outcomes['halt']} halted (with their seat-swapped partners)")
        print(f"{len(rated_pairs(results))} complete seat-swapped pairs rated")
        anchor = bots[0].name
        ratings = bradley_terry(results, anchor=anchor)
        ci = bradley_terry_ci(results, anchor=anchor)
        for name, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
            lo, hi = ci[name]
            print(f"  {name:<10} {elo:7.1f}  95% [{lo:7.1f}, {hi:7.1f}]")
        names = [b.name for b in bots]
        matrix = win_matrix(results)
        print()
        print("score rate (row vs column):")
        print(" " * 10 + "".join(f"{n:>10}" for n in names))
        for a in names:
            cells = "".join(f"{'-':>10}" if a == b else f"{matrix.get((a, b), (float('nan'), 0))[0]:>10.2f}" for b in names)
            print(f"{a:<10}{cells}")
        if ledger_path:
            print()
            ledger.append_arena(
                results, bots, path=ledger_path, tags=tags, settings=_settings(args), started_utc=started, load=load,
                owner=args.owner, hello=hellos[0] if hellos else None, bot_specs=args.bots.split(","),
                scheduled=scheduled, artifacts={"results": args.results, "log": ledger.log_from_args(args)},
                exploration=ledger.exploration_from(args), anchor=anchor, engines=len(specs),
                extra_metrics={"max_turns": args.max_turns})
    return 0


def _settings(args: argparse.Namespace) -> dict:
    skip = {"shard", "stats_out", "claim_dir", "ledger", "no_ledger", "tag", "rest", "owner", "log"}
    return {k: v for k, v in vars(args).items() if k not in skip}


def _selfplay_finish(args, ledger_path, tags, stats, wall, engines, started, load, interrupted=False) -> None:
    """A shard hands its counters to the parent; a lone process appends its own row."""
    if args.stats_out:
        try:
            args.stats_out.write_text(json.dumps(stats), encoding="utf-8")
        except (OSError, TypeError, ValueError) as e:
            print(f"ledger: could not write shard stats: {e}", file=sys.stderr)
    elif ledger_path:
        _selfplay_row(args, ledger_path, tags, stats, wall, engines, started, load, interrupted)


def _selfplay_row(args, ledger_path, tags, stats, wall, engines, started, load, interrupted=False) -> None:
    try:
        metrics = ledger.selfplay_metrics(stats, wall, engines, args.procs, interrupted)
        metrics["full_previews_fraction"] = _previews_fraction(args)
        models = ledger.models_from_specs([args.bot])
        features = next((m.get("features") for m in models.values() if m.get("features") is not None), None)
        try:
            solver = ledger.solver_settings([_selfplay_bot(args)])
        except Exception:  # noqa: BLE001
            solver = None
        prov = ledger.provenance(
            settings=_settings(args), started_utc=started, load=load, owner=args.owner,
            engine=ledger.engine_info(stats.get("hello")), models=models, features_version=features,
            dataset=ledger.dataset_manifest([args.out]), solver=solver, exploration=ledger.exploration_from(args),
            counts=ledger.counts(args.games, metrics["games_valid"],
                                 {"already_recorded": stats["skipped"], "truncated": stats["truncated"]},
                                 stats["dropped"]),
            artifacts={"out_dir": args.out, "drops": args.out / "drops.jsonl", "log": ledger.log_from_args(args)})
        ledger.append("selfplay", metrics, prov, path=ledger_path, tags=tags)
    except Exception as e:  # noqa: BLE001  (the ledger never fails the run)
        print(f"ledger: could not append a selfplay row: {type(e).__name__}: {e}", file=sys.stderr)


def _previews_fraction(args) -> float:
    return 0.0 if args.no_previews else args.full_previews_fraction


def _selfplay_bot(args):
    """The self-play bot, with set-4 factor exploration (self-play only; arena bots stay deterministic)."""
    bot = make_bot(args.bot)
    if hasattr(bot, "explore") and (args.factor_margin >= 0 or args.factor_random > 0 or args.factor_all > 0):
        from yomibench.bots import FactorExplore

        bot.explore = FactorExplore(margin=args.factor_margin if args.factor_margin >= 0 else None,
                                    random_k=args.factor_random, all_prob=args.factor_all)
    return bot


def _keep_awake() -> None:
    """Keep this machine from sleeping until this process exits (no settings are changed)."""
    import os
    import subprocess
    import sys

    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    elif sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])

IN_PROCESS = ("sim", "toy", "batch")  # engines each process makes for itself; any other spec is one server


def _seed_claimer(claim_dir: Path):
    """claim(seed) -> True for the first process to ask for `seed` in `claim_dir` (an exclusive
    file create, atomic on every OS), False for every later one."""
    def claim(seed: int) -> bool:
        try:
            os.close(os.open(claim_dir / str(seed), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False
    return claim


def _sampled_starts(args):
    """--sampled-starts as a sampled_starts.StartPool over --sampled-starts-pool; None when off.
    ValueError for invalid settings, or settings without a pool folder."""
    from yomibench import sampled_starts

    settings = sampled_starts.parse_settings(args.sampled_starts)
    if settings is None:
        return None
    if not args.sampled_starts_pool:
        raise ValueError("needs --sampled-starts-pool DIR")
    return sampled_starts.StartPool(settings, args.sampled_starts_pool)


def _least_tried(args):
    """--explore-least-tried with --least-tried-counts as a least_tried.LeastTried; None when off."""
    from yomibench import least_tried

    try:
        settings = least_tried.parse_settings(args.explore_least_tried)
    except ValueError as e:
        raise SystemExit(f"--explore-least-tried: {e}") from e
    if settings is None:
        return None
    if args.least_tried_counts is None:
        raise SystemExit("--explore-least-tried needs --least-tried-counts PATH")
    return least_tried.LeastTried.load(settings, args.least_tried_counts)


def _selfplay_procs(argv: list[str], procs: int, specs: list[str], collect: bool = False,
                    mem_cap: tuple[float, Path, range] | None = None) -> tuple[int, dict]:
    """Run `procs` copies of this selfplay command. Children share the run's seeds through a claim
    folder: each walks the seed list and plays the seeds it claims first, so a process that
    finishes early takes the next seed instead of idling while one child works through a fixed
    share of long games, and a rerun after a child died spreads the leftovers over every child.
    In-process engines (sim, toy) are made in every child; servers are split between children, so a
    server is never shared. Returns the worst exit code and, with `collect`, the children's merged
    ledger counters (the parent appends the one row).

    With `mem_cap` (--max-proc-mem-gb G, the out folder, the run's seeds), children run a
    memcap.MemoryCap: a child that passes G drops the game(s) it was playing and exits with
    memcap.EXIT_CODE. Those seeds are logged and counted as dropped (`dropped_memory`) and keep
    their claims, so nothing replays them (nor a rerun: seeds drops.jsonl marks as memory drops
    are claimed before the children start); a replacement child takes the dead one's place and
    claims on. A child that exits so with no game dropped isn't replaced (a failure, as before)."""
    import shutil
    import tempfile

    servers = [s for s in specs if s.split(":", 1)[0] not in IN_PROCESS]
    if servers and len(servers) < procs:
        raise SystemExit(f"--procs {procs} needs at least as many engine servers ({len(servers)} given)")
    stats_dir = Path(tempfile.mkdtemp(prefix="yomibench-stats-")) if collect else None
    claim_dir = Path(tempfile.mkdtemp(prefix="yomibench-claims-"))
    earlier = _memory_dropped_before(claim_dir, *mem_cap[1:]) if mem_cap else []

    def spawn(k: int, gen: int = 0) -> subprocess.Popen:
        mine = specs if not servers else servers[k::procs]
        cmd = [sys.executable, "-m", "yomibench.cli", *argv, "--shard", f"{k}/{procs}", "--engine", ",".join(mine),
               "--no-ledger", "--claim-dir", str(claim_dir)]
        if stats_dir:
            cmd += ["--stats-out", str(stats_dir / (f"{k}.json" if gen == 0 else f"{k}r{gen}.json"))]
        return subprocess.Popen(cmd)

    children = [spawn(k) for k in range(procs)]
    drops: list[dict] = []
    try:
        if mem_cap:
            code = _wait_replacing(children, spawn, claim_dir, mem_cap[1], drops)
        else:
            code = max(c.wait() for c in children)
    except KeyboardInterrupt:
        for c in children:
            c.terminate()
        for c in children:
            c.wait()
        code = 130
    if mem_cap:
        from yomibench.memcap import collect_drops

        drops += _record_drops(collect_drops(claim_dir), mem_cap[1])  # any left by children ended by Ctrl-C
    shutil.rmtree(claim_dir, ignore_errors=True)
    if not stats_dir:
        return code, {}
    parts = []
    for f in sorted(stats_dir.glob("*.json")):
        try:
            parts.append(json.loads(f.read_text(encoding="utf-8")))
            f.unlink()
        except (OSError, ValueError):
            pass
    try:
        stats_dir.rmdir()
    except OSError:
        pass
    stats = ledger.merge_selfplay_stats(parts)
    if mem_cap:
        n = len(drops) + len(earlier)
        stats["dropped"] += n
        stats["dropped_memory"] = n
    return code, stats


def _memory_dropped_before(claim_dir: Path, out: Path, run: range) -> list[int]:
    """Seeds of this run an earlier run dropped at its memory cap (drops.jsonl in `out`) and nothing
    has recorded since: claimed here, so a resumed chunk doesn't replay a runaway game."""
    from yomibench.selfplay import done_seeds

    try:
        with open(out / "drops.jsonl", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
    except (OSError, ValueError):
        return []
    have = done_seeds(out)
    seeds = sorted({d["seed"] for d in lines if isinstance(d, dict) and d.get("memory_cap")
                    and d.get("seed") in run and d["seed"] not in have})
    for s in seeds:
        (claim_dir / str(s)).touch()
        print(f"seed {s}: dropped (memory cap exceeded in an earlier run; not replayed)", flush=True)
    return seeds


def _record_drops(drops: list[dict], out: Path) -> list[dict]:
    """Log games children dropped at their memory cap (memcap.collect_drops), as run_selfplay logs
    a crashed game: a progress line and a drops.jsonl entry, here marked memory_cap."""
    from yomibench.memcap import drop_reason
    from yomibench.selfplay import done_seeds

    have = done_seeds(out) if drops else set()
    drops = [d for d in drops if d["seed"] not in have]  # finished as the cap tripped: recorded, not dropped
    for d in drops:
        reason = drop_reason(d)
        print(f"seed {d['seed']}: dropped ({reason})", flush=True)
        for tmp in out.glob(f".g{d['seed']:012d}.{d.get('pid')}.*.tmp"):  # its unfinished record (RecordWriter)
            tmp.unlink(missing_ok=True)
        try:
            with open(out / "drops.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"seed": d["seed"], "characters": d["characters"], "error": reason,
                                    "memory_cap": True, "memory_bytes": d["memory_bytes"],
                                    "time": d["time"]}) + "\n")
        except OSError as e:
            print(f"could not log a dropped seed in {out}: {e}", file=sys.stderr)
    return drops


def _wait_replacing(children: list, spawn, claim_dir: Path, out: Path, drops: list[dict]) -> int:
    """Wait for every child; a child ended by its memory cap is replaced (`children` is updated in
    place, for Ctrl-C) once its dropped games are logged into `drops`. Returns the worst exit code
    of the others (0 when the cap was the only trouble)."""
    import time

    from yomibench.memcap import EXIT_CODE, collect_drops

    codes, gens = [], [0] * len(children)
    # pids of children that dropped a game at the cap. A drop file names its process and that
    # process's parent: on Windows a venv's python.exe is a launcher that runs the interpreter as a
    # child, so the pid the parent spawned (children[k].pid) is the dropping process's ppid.
    tripped: set[int] = set()
    live = set(range(len(children)))
    while live:
        for k in sorted(live):
            rc = children[k].poll()
            if rc is None:
                continue
            if rc == EXIT_CODE:
                found = collect_drops(claim_dir)  # a sibling's too, if it tripped at the same time
                tripped |= {d.get(key) for d in found for key in ("pid", "ppid")} - {None}
                drops += _record_drops(found, out)
            # Replaced only after dropping a game (whose claim it used up), so at most once per seed.
            if rc == EXIT_CODE and children[k].pid in tripped:
                gens[k] += 1
                children[k] = spawn(k, gens[k])
            else:
                codes.append(rc)
                live.discard(k)
        if live:
            time.sleep(0.2)
    return max(codes, default=0)


if __name__ == "__main__":
    raise SystemExit(main())

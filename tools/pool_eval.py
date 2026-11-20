"""Gate a candidate checkpoint for promotion: seat-swapped games against a fixed anchor pool.
It fails closed: a candidate is promoted only on predeclared evidence, never for lack of it.

python tools/pool_eval.py --candidate matrix-do:CANDIDATE.npz --previous-best matrix-do:BEST.npz \\
    --ckpt1 matrix-do:ckpt1.npz --ckpt2 matrix-do:ckpt2.npz \\
    --engine game-pool:auto --characters Ninja,Cowboy --candidate-set 6 --results results/pool.jsonl

The anchor pool is fixed so gates stay comparable: the previous best, ckpt0 (plain matrix-do),
ckpt1 and ckpt2. --history adds opponents that are reported but not part of the decision. An
opponent whose spec repeats an earlier one (say the previous best is ckpt2) is played once,
under the first label, and its criteria use that matchup (`aliases` records it). Each opponent
plays --pairs seat-swapped pairs (default 200) per character pairing: `--characters Ninja,Cowboy`
is all four pairings (Ninja:Ninja, Ninja:Cowboy, Cowboy:Ninja, Cowboy:Cowboy; in a seat-swapped
pair the candidate plays each side once). --candidate-set is the set every game plays (default
the same as self-play's, yomibench.selfplay.DEFAULT_CANDIDATE_SET), so a gate plays the rules
self-play recorded.

--pairs-by-pairing "Cowboy:Cowboy=60,Cowboy:Ninja=60,Ninja:Cowboy=60,Ninja:Ninja=32" (instead of
--pairs) schedules that many pairs in each pairing (every pairing of --characters): a focus-weighted
gate. Pair i of a pairing keeps the seed it has with --pairs M (M the smallest count) for i < M, so
the first 32 pairs of each pairing above are the --pairs 32 games; shards interleave the pairings
in proportion (yomibench.arena.pair_schedule). Each pairing is judged on its own scheduled pairs
(--min-pairs defaults to 75% of each); pooled scores are over all complete pairs, so pairings weigh
by their counts. A --stage with pairs_by_pairing (yomibench.gate_seq) needs the same counts.

The Cowboy gate (--quick --cowboy-mirror-pairs 60 --cowboy-vs "Ninja=16,Wizard=16,Mutant=16,Robot=16",
--characters Cowboy then each X in that order, instead of --pairs; yomibench.gate_seq.cowboy): the
candidate plays Cowboy in every game. 60 seat-swapped Cowboy:Cowboy pairs, and per X 16
candidate-character pairs: the candidate is Cowboy and the previous best X in both games, the
candidate P1 in the first (Cowboy:X) and P2 in the second (X:Cowboy), on the same seed
(yomibench.arena.cell_schedule; they pair up by yomibench.arena.pair_games). No Ninja mirror. It is
judged per cell (the Cowboy mirror, and "Cowboy v X" for each X) and pooled by the Cowboy rule
(gate_seq's module docstring); without --stage, as the rule's final look. Each cell also reports
sufficient data (>= 75% of its scheduled pairs complete, <= --max-failure-rate halted). Before it
plays, every engine's hello must list every character the gate plays; if one doesn't, it plays
nothing and exits 6. --rule and --min-pairs don't apply.

A Cowboy gate's cell extension (the per-character floor; gate_seq.cowboy_extension): the gate's flags
plus --cowboy-extension "Ninja=32" play only those cells' candidate-character pairs, on the gate's
--seed, numbered after the gate's own (gate_seq.extension_offsets), so no game repeats a gate game.
It plays shards (--shard) or decides from them at its stage (--from-shards --stage, the stage holding
the gate's numbers of each extended cell): promote (exit 0) unless an extended cell's point score over
all its complete pairs, the gate's and the extension's, is below the floor: reject (exit 1, blocked).

Every matchup is reported as the candidate's paired score (each seat-swapped pair's two games
averaged, so seat advantage cancels) with a 95% paired-bootstrap interval (whole pairs
resampled). Halted and truncated games drop their whole pair. Printed summaries and reasons say it
as a win percentage: "won 64% of games (58–71%)" is the candidate's points in those games, a draw
counting half a win, with the 95% interval; the JSON keeps the fractions (score, ci95).

Predeclared criteria (printed before any game and stored in the summary), applied to every
character pairing separately:
  sufficient data    every anchor matchup has at least --min-pairs complete pairs (default 75% of
                     those scheduled: 150 of 200), and at most --max-failure-rate (5%) of its
                     scheduled pairs hit an engine halt
  non-inferiority    every anchor matchup: score >= 0.45 and upper 95% bound >= 0.50
  improvement        against the previous best: lower 95% bound > 0.50
Decision for each pairing, in this order:
  reject        an anchor matchup with sufficient data fails non-inferiority (a regression)
  inconclusive  any anchor matchup lacks sufficient data, or the candidate is non-inferior
                without a significant improvement: it may keep training, but is not promoted
  promote       sufficient data, non-inferiority and improvement
Overall decision, --rule pooled (the default since 2026-11-15, user-approved):
  reject        any pairing with sufficient data has a matchup whose upper 95% bound is below 0.50
                (the per-pairing regression guard: a clear loss there), or the pooled games fail
                non-inferiority against an anchor
  inconclusive  any pairing lacks sufficient data, or the pooled games show no significant
                improvement (pooled lower 95% bound vs the previous best <= 0.50)
  promote       otherwise: pooled improvement, every pairing guarded, every anchor non-inferior
--rule pairing is the original rule: reject if any pairing rejects, promote only if every pairing
promotes (each by the criteria above), else inconclusive. Pooled scores are reported either way.
The exit status is 0 for promote, 1 for reject and 2 for inconclusive. Game rows and the summary
are appended to the --results JSONL file, and one `arena` row with the gate to the metrics
ledger (--ledger, default metrics/ledger.jsonl; see yomibench.ledger). Decisions use only this
invocation's games, never older rows.

Seat preflight: every run that plays games first checks `--seat-preflight N` decisions (default 20)
per pairing between two copies of the candidate: at each, both decide twice on the same state, seat
1 searching first and then seat 2, and each seat's pick and strategy must agree both ways. If not, it
plays nothing and exits 4: a bot's decisions would depend on the seat order, as set 4's shared
expansions once made them. Seat-swapping two copies of one net couldn't show that.

--procs N plays the schedule in N processes (the in-process sim holds Python's GIL, so threads
can't share it): each plays every N-th seat-swapped pair with the same seeds, an in-process engine
spec (sim, toy) is made in every child and servers are split between them; the parent merges the
games and decides.

Sharding across machines: `--shard K/N --shard-out FILE` plays only pairs K mod N of the schedule
(with --procs P, in P processes) and writes a shard file: a header line naming the shard and the
schedule (seed, pairs, characters, candidate set, max turns, mode, and the candidate and every
opponent by model sha256, so paths may differ between machines), then the games.
`--from-shards F1,F2,...` decides from shard files instead of playing, with the same arguments
otherwise. It fails closed: unless the files are shards 0..N-1 of one N, each exactly once, of the
schedule these arguments describe, each holding exactly that shard's games, it decides nothing
and exits 3. The criteria and the decision rule are the same as for a gate played here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

# A gate plays one search per process, like self-play (yomibench.cli selfplay): a math library
# starting a thread per core in each --procs child fights the other processes, and self-play
# next to it, for the cores. Set before numpy/torch load; children inherit it. A batch engine's
# sizes its own pool explicitly, so RAYON_NUM_THREADS doesn't touch --engine batch:N.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from yomibench import ledger  # noqa: E402
from yomibench import gate_seq  # noqa: E402
from yomibench.arena import cell_schedule, pair_games, pair_schedule, rated_pairs, run_round_robin, win_matrix
from yomibench.bots import make_bot
from yomibench.cli import _match_settings
from yomibench.engine import connect, expand_specs, set5_spec
from yomibench.selfplay import DEFAULT_CANDIDATE_SET

IN_PROCESS = ('sim', 'toy', 'batch')  # engine specs every child process makes for itself

ANCHORS = ('previous-best', 'ckpt0', 'ckpt1', 'ckpt2')
NONINFERIOR_SCORE, NONINFERIOR_UPPER, IMPROVEMENT_LOWER = 0.45, 0.50, 0.50
GUARD_UPPER = 0.50  # --rule pooled: a pairing regresses when a matchup's upper 95% bound is below this
# (a score floor such as 0.45 at 50 pairs rejects an equal net about 1 time in 10, per review)
RULES = ('pooled', 'pairing')
EXIT = {'promote': 0, 'reject': 1, 'inconclusive': 2, 'pass': 0}
SHARD_MISMATCH = 3  # --from-shards: the files aren't one schedule's shards; no decision
CONTINUE = 5  # --stage: a look before the last that neither rejects nor ends the stage; play on
SEAT_PREFLIGHT = 4  # the seat preflight failed: a bot decides differently when the other seat searches first
ENGINE_CHARACTERS = 6  # an engine doesn't play a character the gate needs; nothing played


QUICK_ANCHORS = ('previous-best',)  # --quick: the candidate against the previous best alone


def criteria(min_pairs, max_failure_rate, scheduled_pairs, anchors=ANCHORS, rule='pooled'):
    if rule == 'pooled':
        return {
            'rule': 'pooled', 'anchors': list(anchors),
            'per': 'pooled over every character pairing, with a per-pairing regression guard',
            'sufficient_data': f'every anchor matchup in every pairing: >= {min_pairs} of {scheduled_pairs} scheduled '
                               f'pairs complete, <= {max_failure_rate:.0%} of scheduled pairs halted',
            'pairing_guard': f'every pairing with sufficient data: every anchor matchup has upper 95% bound >= '
                             f'{GUARD_UPPER}',
            'non_inferiority': f'pooled, every anchor matchup: score >= {NONINFERIOR_SCORE} and upper 95% bound >= '
                               f'{NONINFERIOR_UPPER}',
            'improvement': f'pooled, vs the previous best: lower 95% bound > {IMPROVEMENT_LOWER}',
            'decision': 'reject if a guarded pairing or a pooled anchor matchup regresses; else inconclusive if any '
                        'pairing lacks data or there is no pooled improvement; else promote',
            'ci': 'paired bootstrap, 2000 resamples of whole seat-swapped pairs',
            'min_pairs': min_pairs, 'max_failure_rate': max_failure_rate, 'scheduled_pairs': scheduled_pairs,
        }
    return {'rule': 'pairing',
        'anchors': list(anchors), 'per': 'character pairing (every criterion, in every pairing)',
        'sufficient_data': f'every anchor matchup: >= {min_pairs} of {scheduled_pairs} scheduled pairs complete, '
                           f'<= {max_failure_rate:.0%} of scheduled pairs halted',
        'non_inferiority': f'every anchor matchup: score >= {NONINFERIOR_SCORE} and upper 95% bound >= '
                           f'{NONINFERIOR_UPPER}',
        'improvement': f'vs the previous best: lower 95% bound > {IMPROVEMENT_LOWER}',
        'decision': 'reject if a matchup with sufficient data fails non-inferiority; else inconclusive if data is '
                    'insufficient or there is no improvement; else promote',
        'ci': 'paired bootstrap, 2000 resamples of whole seat-swapped pairs',
        'min_pairs': min_pairs, 'max_failure_rate': max_failure_rate, 'scheduled_pairs': scheduled_pairs,
    }


def matchup_scores(results, candidate='candidate', samples=2000, seed=0):
    """{opponent: {score, ci95, complete_pairs, halted_pairs, games, truncated, halted}} from the
    candidate's view: the paired score over complete seat-swapped pairs, paired-bootstrap 95% CI."""
    games: dict[str, list] = {}
    for r in results:
        if candidate in r.bots:
            games.setdefault(next(b for b in r.bots if b != candidate), []).append(r)
    out = {}
    for opp, rs in games.items():
        pairs = rated_pairs(rs)
        scores = [sum(r.score(candidate) for r in p) / 2 for p in pairs]
        out[opp] = {**ledger.paired_score(scores, samples, seed), 'complete_pairs': len(pairs),
                    'halted_pairs': sum(any(r.outcome == 'halt' for r in g) for g in pair_games(rs)),
                    'games': len(rs), 'truncated': sum(r.outcome == 'truncated' for r in rs),
                    'halted': sum(r.outcome == 'halt' for r in rs)}
    return out


def gate_summary(results, scheduled_pairs, min_pairs=None, max_failure_rate=0.05, aliases=None,
                 candidate='candidate', samples=2000, pairings=None, anchors=ANCHORS, rule='pooled'):
    """The decision on `results` (see the module docstring), judged per character pairing.
    `scheduled_pairs` is per matchup and pairing, or {"P1:P2": pairs} (--pairs-by-pairing; then
    `min_pairs` is an int for every pairing or such a dict, default 75% of each); `pairings` lists
    the pairings that were scheduled (default: those in `results`), so a pairing with no games is
    inconclusive, not skipped."""
    pairings = [tuple(p) for p in pairings] if pairings else sorted({tuple(r.characters) for r in results})
    if isinstance(scheduled_pairs, dict):
        if not isinstance(min_pairs, dict):
            min_pairs = {k: math.ceil(0.75 * n) if min_pairs is None else min_pairs for k, n in scheduled_pairs.items()}
        sched = {pr: scheduled_pairs[':'.join(pr)] for pr in pairings}
        need = {pr: min_pairs[':'.join(pr)] for pr in pairings}
        total_sched, total_need = sum(sched.values()), sum(need.values())
    else:
        min_pairs = math.ceil(0.75 * scheduled_pairs) if min_pairs is None else min_pairs
        sched, need = dict.fromkeys(pairings, scheduled_pairs), dict.fromkeys(pairings, min_pairs)
        total_sched, total_need = scheduled_pairs * max(1, len(pairings)), min_pairs * max(1, len(pairings))
    per = []
    for pr in pairings:
        j = _judge([r for r in results if tuple(r.characters) == pr], sched[pr], need[pr], max_failure_rate,
                   aliases, candidate, samples, anchors)
        per.append({'characters': list(pr), **{k: j[k] for k in ('decision', 'reason', 'improvement', 'checks',
                                                                  'previous_best_score', 'ci95', 'matchups')}})
    pooled = _judge(results, total_sched, total_need, max_failure_rate, aliases, candidate, samples, anchors)
    decisions = [p['decision'] for p in per]
    if not per:
        decision, reason = 'inconclusive', 'no games'
    elif rule == 'pooled':
        decision, reason = _pooled_decision(per, pooled)
    else:
        decision = 'reject' if 'reject' in decisions else ('promote' if set(decisions) == {'promote'} else 'inconclusive')
        reason = per[0]['reason'] if len(per) == 1 else '; '.join(
            f"{':'.join(p['characters'])} {p['decision']}: {p['reason']}" for p in per if p['decision'] == decision)
    return {'type': 'pool_summary', 'decision': decision, 'reason': reason,
            'criteria': criteria(min_pairs if isinstance(min_pairs, dict) else max(1, min_pairs), max_failure_rate,
                                 scheduled_pairs, anchors, rule),
            'improvement': bool(per) and all(p['improvement'] for p in per), 'pairings': per,
            'checks': pooled['checks'], 'previous_best_score': pooled['previous_best_score'],
            'ci95': pooled['ci95'], 'completed_pairs_vs_previous': pooled['completed_pairs_vs_previous'],
            'matchups': pooled['matchups'], 'outcomes': dict(Counter(r.outcome for r in results)),
            'matrix': [{'row': a, 'column': b, 'score': rate, 'games': n}
                       for (a, b), (rate, n) in sorted(win_matrix(results).items())]}


def _pooled_decision(per, pooled):
    """--rule pooled on the per-pairing and pooled judgements (see the module docstring)."""
    guard = [(p, a, c) for p in per for a, c in p['checks'].items()
             if c['sufficient_data'] and c['ci95'][1] is not None and c['ci95'][1] < GUARD_UPPER]
    pooled_regress = [a for a, c in pooled['checks'].items() if c['sufficient_data'] and not c['non_inferior']]
    short = [(p, a) for p in per for a, c in p['checks'].items() if not c['sufficient_data']]
    prev = pooled['checks']['previous-best']
    if guard or pooled_regress:
        return 'reject', '; '.join(
            [f"{':'.join(p['characters'])} regression vs {a}: {won(c['score'], c['ci95'])} (upper < {GUARD_UPPER})"
             for p, a, c in guard]
            + [f"pooled regression vs {a}: {won(pooled['checks'][a]['score'], pooled['checks'][a]['ci95'])}"
               for a in pooled_regress])
    if short:
        return 'inconclusive', '; '.join(f"{':'.join(p['characters'])} insufficient data vs {a}" for p, a in short)
    if not pooled['improvement']:
        return 'inconclusive', f"no significant improvement (pooled): {won(prev['score'], prev['ci95'])}"
    return 'promote', (f"pooled improvement: {won(prev['score'], prev['ci95'])}; no pairing clearly worse; "
                       f"non-inferior to every anchor")


def _judge(results, scheduled_pairs, min_pairs, max_failure_rate, aliases, candidate, samples, anchors=ANCHORS):
    """The criteria on one set of games (one pairing, or all pairings pooled)."""
    aliases = aliases or {}
    matchups = matchup_scores(results, candidate, samples)
    empty = {'score': None, 'ci95': [None, None], 'complete_pairs': 0, 'halted_pairs': 0, 'games': 0,
             'truncated': 0, 'halted': 0}
    checks = {}
    for anchor in anchors:
        m = matchups.get(aliases.get(anchor, anchor), empty)
        failure = m['halted_pairs'] / scheduled_pairs if scheduled_pairs else 1.0
        sufficient = m['complete_pairs'] >= max(1, min_pairs) and failure <= max_failure_rate
        lo, hi = m['ci95']
        noninferior = (m['score'] is not None and m['score'] >= NONINFERIOR_SCORE and hi >= NONINFERIOR_UPPER)
        checks[anchor] = {'played_as': aliases.get(anchor, anchor), 'sufficient_data': sufficient,
                          'failure_rate': round(failure, 4), 'non_inferior': noninferior,
                          'score': m['score'], 'ci95': m['ci95'], 'complete_pairs': m['complete_pairs']}
    prev = checks['previous-best']
    improvement = prev['ci95'][0] is not None and prev['ci95'][0] > IMPROVEMENT_LOWER
    regressions = [a for a, c in checks.items() if c['sufficient_data'] and not c['non_inferior']]
    short = [a for a, c in checks.items() if not c['sufficient_data']]
    if regressions:
        decision = 'reject'
        reason = '; '.join(f"regression vs {a}: {won(checks[a]['score'], checks[a]['ci95'])}" for a in regressions)
    elif short:
        decision = 'inconclusive'
        reason = '; '.join(f"insufficient data vs {a}: {checks[a]['complete_pairs']} complete pairs (need "
                           f"{max(1, min_pairs)}), {checks[a]['failure_rate']:.1%} halted pairs" for a in short)
    elif not improvement:
        decision = 'inconclusive'
        reason = f"non-inferior but no significant improvement: {won(prev['score'], prev['ci95'])}"
    else:
        decision = 'promote'
        reason = f"improvement: {won(prev['score'], prev['ci95'])}; non-inferior to every anchor"
    return {'decision': decision, 'reason': reason, 'improvement': improvement, 'checks': checks,
            'previous_best_score': prev['score'], 'ci95': prev['ci95'],
            'completed_pairs_vs_previous': prev['complete_pairs'],
            'matchups': [{'opponent': opp, **m} for opp, m in matchups.items()]}


def format_table(summary):
    """The gate as text: each pairing's (or Cowboy gate cell's) decision and every matchup's paired
    score and interval, as win percentages."""
    lines = []
    for c in summary.get('cells') or []:
        if 'all_pairs' in c:  # a cell extension's cell: its own pairs, then the cell over all its pairs
            flag = (f"; with the gate's, {c['all_pairs']} pairs " + (f"{c['all_score']:.1%}" if c['all_score'] is not None
                                                                     else 'none') + (' BELOW THE FLOOR' if c['below_floor'] else ''))
        else:
            flag = ' BROKEN' if c.get('broken') else '' if c.get('guarded') or c['cell'] == gate_seq.MIRROR else ' (unguarded)'
        lines.append(f"{c['cell']:<16} {won(c['score'], c['ci95']):<32} {c['complete_pairs']} of {c['scheduled_pairs']} "
                     f"pairs complete, {c['halted_pairs']} halted{flag}")
    for p in [] if summary.get('cells') else summary['pairings']:
        lines.append(f"{':'.join(p['characters'])}: {p['decision'].upper()} ({p['reason']})")
        for m in p['matchups']:
            lines.append(f"  candidate v {m['opponent']:<14} {won(m['score'], m['ci95']):<32} {m['complete_pairs']} "
                         f"complete pairs ({m['truncated']} truncated, {m['halted']} halted games)")
    lines.append('pooled over cells:' if summary.get('cells') else 'pooled over pairings:')
    for m in summary['matchups']:
        lines.append(f"  candidate v {m['opponent']:<14} {won(m['score'], m['ci95']):<32} {m['complete_pairs']} "
                     f"complete pairs")
    return '\n'.join(lines)


def won(score, ci95):
    return gate_seq.won(score, ci95)


def _cell_rows(results, spec, max_failure_rate, candidate, samples):
    """Per Cowboy gate (or extension) cell in `spec` ([(cell, matchups, scheduled pairs)]): its score,
    95% CI, complete and halted pairs, sufficient data, and its complete pairs' scores (`scores`)."""
    cells = []
    for name, matchups, sched in spec:
        rs = [r for r in results if tuple(r.characters) in matchups and candidate in r.bots]
        j = _judge(rs, sched, math.ceil(0.75 * sched), max_failure_rate, {}, candidate, samples, QUICK_ANCHORS)
        c = j['checks']['previous-best']
        m = next((x for x in j['matchups'] if x['opponent'] == 'previous-best'), {'halted_pairs': 0})
        cells.append({'cell': name, 'characters': [gate_seq.COWBOY, matchups[0][1]],
                      'games': ['Cowboy:Cowboy, candidate P1', 'Cowboy:Cowboy, candidate P2'] if name == gate_seq.MIRROR
                      else [f"Cowboy:{matchups[0][1]}, candidate P1 as Cowboy", f"{matchups[0][1]}:Cowboy, candidate P2 as Cowboy"],
                      'scheduled_pairs': sched, 'complete_pairs': c['complete_pairs'], 'halted_pairs': m['halted_pairs'],
                      'failure_rate': c['failure_rate'], 'sufficient_data': c['sufficient_data'],
                      'score': c['score'], 'ci95': c['ci95'],
                      'scores': [sum(r.score(candidate) for r in p) / 2 for p in rated_pairs(
                          [r for r in rs if 'previous-best' in r.bots])]})
    return cells


def cowboy_summary(results, st, look, max_failure_rate=0.05, candidate='candidate', samples=2000):
    """The Cowboy gate (yomibench.gate_seq.cowboy) on `results` at look `look` of its stage `st`:
    per cell (the Cowboy mirror, each Cowboy v X) its score, 95% CI, complete and halted pairs and
    sufficient data (>= 75% of its scheduled pairs complete, <= `max_failure_rate` halted), the
    pooled score over every gate pair, and gate_seq.cowboy_verdict's decision (pass promotes)."""
    cells = _cell_rows(results, gate_seq.cowboy_cells(st), max_failure_rate, candidate, samples)
    total = sum(c['scheduled_pairs'] for c in cells)
    pooled = _judge(results, total, math.ceil(0.75 * total), max_failure_rate, {}, candidate, samples, QUICK_ANCHORS)
    prev = pooled['checks']['previous-best']
    v = gate_seq.cowboy_verdict(st, look, cells, {'score': prev['score'], 'ci95': prev['ci95']})
    flags = {r['cell']: r for r in v['cowboy']['cells']}
    for c in cells:
        c.update(label=won(c['score'], c['ci95']), mean=flags[c['cell']]['mean'], se=flags[c['cell']]['se'],
                 guarded=flags[c['cell']]['guarded'], broken=flags[c['cell']]['broken'])
        del c['scores']
    decision = {'pass': 'promote' if st.get('final') else 'pass'}.get(v['verdict'], v['verdict'])
    return {'type': 'pool_summary', 'decision': decision,
            'reason': f"{st['name']} look {look + 1}/{len(st['looks'])}: {v['reason']}",
            'criteria': {'rule': 'cowboy', 'anchors': list(QUICK_ANCHORS),
                         'per': 'Cowboy gate cell (the Cowboy mirror, each Cowboy v X), and pooled over every gate pair',
                         **{k: st[k] for k in ('cells', 'looks', 'reject_z', 'accept_z', 'final_z', 'guard_z',
                                               'cell_block_upper', 'not_worse', 'min_pairs_per_pairing', 'pooled')},
                         **{k: st[k] for k in ('mirror_floor', 'floor_futility') if k in st},
                         'sufficient_data': f'reported per cell: >= 75% of its scheduled pairs complete, '
                                            f'<= {max_failure_rate:.0%} of them halted',
                         'decision': st['rule'],
                         'ci': 'paired bootstrap, 2000 resamples of whole pairs (reported); the rule uses mean +- z x SE',
                         'scores': 'win percentage: the candidate\'s points in the games, a draw half a win',
                         'max_failure_rate': max_failure_rate, 'scheduled_pairs': total},
            'improvement': v['verdict'] == 'pass', 'pairings': cells, 'cells': cells,
            'pooled': {'score': prev['score'], 'ci95': prev['ci95'], 'label': won(prev['score'], prev['ci95']),
                       'complete_pairs': prev['complete_pairs'], 'mean': v['cowboy']['pooled']['mean'],
                       'se': v['cowboy']['pooled']['se']},
            'rule_path': v['cowboy']['path'], 'blocked_by': v['cowboy']['blocked_by'],
            'unguarded': v['cowboy']['unguarded'],
            'floor_cp': v['cowboy'].get('floor_cp'), 'floor_futile': v['cowboy'].get('floor_futile'),
            'checks': pooled['checks'], 'previous_best_score': prev['score'], 'ci95': prev['ci95'],
            'completed_pairs_vs_previous': prev['complete_pairs'], 'matchups': pooled['matchups'],
            'outcomes': dict(Counter(r.outcome for r in results)),
            'matrix': [{'row': a, 'column': b, 'score': rate, 'games': n}
                       for (a, b), (rate, n) in sorted(win_matrix(results).items())], 'stage': v}


def extension_summary(results, st, max_failure_rate=0.05, candidate='candidate', samples=2000):
    """A Cowboy gate's cell extension (yomibench.gate_seq.cowboy_extension) on `results`: per extended
    cell the extension's own score, 95% CI, complete and halted pairs, and the cell over all its complete
    pairs, the gate's and the extension's (all_pairs, all_score), against the per-character floor;
    gate_seq.extension_verdict decides: promote, or reject when a cell is below the floor (blocked)."""
    cells = _cell_rows(results, gate_seq.extension_cells(st), max_failure_rate, candidate, samples)
    v = gate_seq.extension_verdict(st, cells)
    rows = {r['cell']: r for r in v['cells']}
    for c in cells:
        r = rows[c['cell']]
        c.update(label=won(c['score'], c['ci95']), main=r['main'], all_pairs=r['complete_pairs'], all_score=r['score'],
                 below_floor=r['below_floor'])
        del c['scores']
    total = int(st['pairs'])
    pooled = _judge(results, total, math.ceil(0.75 * total), max_failure_rate, {}, candidate, samples, QUICK_ANCHORS)
    prev = pooled['checks']['previous-best']
    return {'type': 'pool_summary', 'decision': 'promote' if v['verdict'] == 'pass' else 'reject',
            'reason': f"{st['name']}: {v['reason']}",
            'criteria': {'rule': 'cowboy_extension', 'anchors': list(QUICK_ANCHORS),
                         'per': "extended Cowboy-vs-X cell, over the gate's and the extension's complete pairs",
                         **{k: st[k] for k in ('cells', 'extension_vs', 'extension_pairs', 'cell_floor', 'main_cells')},
                         'decision': st['rule'],
                         'scores': "win percentage: the candidate's points in the games, a draw half a win",
                         'max_failure_rate': max_failure_rate, 'scheduled_pairs': total},
            'improvement': v['verdict'] == 'pass', 'pairings': cells, 'cells': cells,
            'pooled': {'score': prev['score'], 'ci95': prev['ci95'], 'label': won(prev['score'], prev['ci95']),
                       'complete_pairs': prev['complete_pairs']},
            'blocked_by': v['blocked_by'], 'cell_floor': st['cell_floor'],
            'checks': pooled['checks'], 'previous_best_score': prev['score'], 'ci95': prev['ci95'],
            'completed_pairs_vs_previous': prev['complete_pairs'], 'matchups': pooled['matchups'],
            'outcomes': dict(Counter(r.outcome for r in results)),
            'matrix': [{'row': a, 'column': b, 'score': rate, 'games': n}
                       for (a, b), (rate, n) in sorted(win_matrix(results).items())], 'stage': v}


def _connect(spec, candidate_set, match_settings=None):
    """engine.connect, with match settings only when given (a standard-settings gate connects as it always has)."""
    return connect(spec, candidate_set, match_settings) if match_settings else connect(spec, candidate_set)


def engine_characters(spec, candidate_set, candidate, match_settings=None):
    """The characters engine `spec` plays, from its hello (None if it doesn't say)."""
    if spec.split(':', 1)[0] == 'batch':
        from yomibench.batch_play import batch_arena_engines

        engine = batch_arena_engines([make_bot(candidate)], candidate_set, match_settings)(spec)
    else:
        engine = _connect(spec, candidate_set, match_settings)
    try:
        return engine.hello().get('characters')
    finally:
        close = getattr(engine, 'close', None)
        if close:
            close()


def _play(args, pool, specs, characters, shard, results_path, hellos):
    """Play the candidate against every pool opponent (this shard's pairs only, with `shard`),
    appending each game to `results_path`. Returns the GameResults. A `batch` engine serves the
    candidate's and the current opponent's nets, each seat scored on its own net's features."""
    batch = {}

    def make(spec):
        if spec.split(':', 1)[0] == 'batch':
            engine = batch['make'](spec)
        else:
            engine = _connect(spec, args.candidate_set, args.match_settings)
        if not hellos:
            try:
                hellos.append(engine.hello())
            except Exception:  # noqa: BLE001  (only provenance)
                hellos.append({})
        return engine
    candidate = make_bot(args.candidate)
    candidate.name = 'candidate'
    results = []
    # Load one opponent at a time, so the pool need not keep every network in memory.
    for label, spec in pool:
        opponent = make_bot(spec)
        opponent.name = label
        if any(s.split(':', 1)[0] == 'batch' for s in specs):
            from yomibench.batch_play import batch_arena_engines

            batch['make'] = batch_arena_engines([candidate, opponent], args.candidate_set, args.match_settings)
        by_pairing = {'pairs_by_pairing': _by_pairing(args.pairs_by_pairing)} if args.pairs_by_pairing else {}
        if args.cowboy:
            by_pairing = {'schedule': _cowboy_schedule(args.cowboy)}
        results.extend(run_round_robin([lambda s=s: make(s) for s in specs], [candidate, opponent], characters,
                                       args.pairs, args.seed, results_path, max_turns=args.max_turns, shard=shard,
                                       record_dir=args.record_dir, **by_pairing))
        del opponent
    return results


def _config(spec):
    bot = make_bot(spec)
    return bot.config() if hasattr(bot, 'config') else {'bot': spec}


class ShardMismatch(Exception):
    pass


def bot_identity(spec):
    """A bot spec with its model file named by sha256, so the same bot matches across machines. A
    suffix (`@cheap=0.75:32`, `@4`) stays: it changes how the bot plays."""
    name, _, rest = spec.partition(':')
    model, at, suffix = rest.partition('@')
    if model and Path(model).is_file():
        return f'{name}:sha256={hashlib.sha256(Path(model).read_bytes()).hexdigest()}' + (at + suffix)
    return spec


def _by_pairing(counts):
    """{"P1:P2": pairs} -> {(P1, P2): pairs}, as yomibench.arena schedules them."""
    return {tuple(k.split(':')): v for k, v in counts.items()}


def _cowboy_cell_spec(cowboy):
    """(cell, matchups, scheduled pairs) of what a Cowboy gate ({mirror_pairs, cowboy_vs}) or its cell
    extension (with `extension`) plays."""
    if cowboy.get('extension'):
        return gate_seq.extension_cells({'extension_vs': cowboy['extension']})
    return gate_seq.cowboy_cells(cowboy)


def _cowboy_schedule(cowboy):
    """The Cowboy gate's cell_schedule from {mirror_pairs, cowboy_vs}: the mirror's ordinary pairs,
    and each X's candidate-character pairs (the candidate, bot a, on Cowboy). With `extension`
    ({X: pairs}), only those cells' extension pairs, numbered after the gate's (gate_seq.extension_offsets)."""
    if cowboy.get('extension'):
        first = gate_seq.extension_offsets(cowboy['cowboy_vs'], cowboy['extension'])
        return cell_schedule({}, {(gate_seq.COWBOY, x): n for x, n in cowboy['extension'].items()},
                             {(gate_seq.COWBOY, x): first[x] for x in cowboy['extension']})
    return cell_schedule({(gate_seq.COWBOY, gate_seq.COWBOY): cowboy['mirror_pairs']},
                         {(gate_seq.COWBOY, x): n for x, n in cowboy['cowboy_vs'].items()})


def schedule(args, pool, characters):
    """What a gate plays, independent of where its files are: shards must agree on all of it."""
    out = {'seed': args.seed, 'pairs': args.pairs, 'characters': list(characters),
           'candidate_set': args.candidate_set, 'max_turns': args.max_turns,
           'mode': 'quick' if args.quick else 'pool', 'candidate': bot_identity(args.candidate),
           'pool': [[label, bot_identity(spec)] for label, spec in pool]}
    if args.pairs_by_pairing:
        out['pairs_by_pairing'] = args.pairs_by_pairing
    if getattr(args, 'match_settings', None):  # only when given: standard-settings schedules stay as they were
        out['match_settings'] = args.match_settings
    if getattr(args, 'set5', None):
        out['set5'] = args.set5
    if getattr(args, 'cowboy', None):  # the X order fixes their seeds, so it's kept as a list
        out['cowboy'] = {'mirror_pairs': args.cowboy['mirror_pairs'], 'vs': [list(x) for x in args.cowboy['cowboy_vs'].items()]}
        if args.cowboy.get('extension'):  # only then: a gate's schedule stays as it was
            out['cowboy']['extension'] = [list(x) for x in args.cowboy['extension'].items()]
    return out


def shard_games(sched, k, n):
    """(bots, characters, seed) of every game shard k of n plays, as run_round_robin schedules them
    (one round robin per opponent, game numbers from 0 in each; shards split the schedule order)."""
    by_pairing, cowboy = sched.get('pairs_by_pairing'), sched.get('cowboy')
    if cowboy:
        pairs = _cowboy_schedule({'mirror_pairs': cowboy['mirror_pairs'], 'cowboy_vs': dict(cowboy['vs']),
                                  'extension': dict(cowboy.get('extension') or [])})
    else:
        pairs = [(chars, g, False) for chars, g in
                 pair_schedule(sched['characters'], sched['pairs'], _by_pairing(by_pairing) if by_pairing else None)]
    out = []
    for label, _ in sched['pool']:
        for place, (chars, g, held) in enumerate(pairs):
            if place % n == k:
                seed = sched['seed'] * 1_000_003 + g
                out += [(('candidate', label), tuple(chars), seed),
                        ((label, 'candidate'), (chars[1], chars[0]) if held else tuple(chars), seed)]
    return out


def write_shard_header(path, sched, k, n, engine):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {'type': 'shard_header', 'shard': [k, n], 'schedule': sched, 'engine': engine,
            'host': socket.gethostname(), 'code': ledger.git_info()}  # the commit that played these games
    path.write_text(json.dumps(head) + '\n', encoding='utf-8')


def read_shard(path, sched, k=None, n=None):
    """(header, game dicts) of a shard file, checked against `sched` (and k/n if given): its games
    must be exactly its shard's. ShardMismatch otherwise."""
    try:
        lines = [x for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
        head = json.loads(lines[0]) if lines else {}
        games = [json.loads(x) for x in lines[1:]]
    except (OSError, ValueError) as e:
        raise ShardMismatch(f'{path}: unreadable ({e})') from e
    if head.get('type') != 'shard_header':
        raise ShardMismatch(f'{path}: no shard header')
    if head.get('schedule') != sched:
        theirs = head.get('schedule') or {}
        diff = sorted(key for key in set(sched) | set(theirs) if sched.get(key) != theirs.get(key))
        raise ShardMismatch(f'{path}: a different schedule (differs in {", ".join(diff)})')
    try:
        hk, hn = (int(x) for x in head['shard'])
        got = sorted((tuple(g['bots']), tuple(g['characters']), int(g['seed'])) for g in games)
    except (KeyError, TypeError, ValueError) as e:
        raise ShardMismatch(f'{path}: malformed ({e!r})') from e
    if not 0 <= hk < hn or (k is not None and (hk, hn) != (k, n)):
        raise ShardMismatch(f'{path}: shard {hk}/{hn}, expected {k}/{n}')
    want = sorted(shard_games(sched, hk, hn))
    if got != want:
        raise ShardMismatch(f'{path}: its {len(got)} games are not shard {hk}/{hn} of this schedule '
                            f'({len(want)} games)')
    return head, games


def load_shards(paths, sched, partial=False):
    """Every game of shard files that together are one schedule: shards 0..N-1 of one N, each
    exactly once (with `partial`, any of them, each at most once). Returns (game dicts, a source row
    per file). ShardMismatch otherwise."""
    games, sources = [], []
    for path in paths:
        head, gs = read_shard(path, sched)
        games += gs
        sources.append({'file': str(path), 'shard': head['shard'], 'host': head.get('host'),
                        'engine': head.get('engine'), 'games': len(gs),
                        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    ns = {s['shard'][1] for s in sources}
    if len(ns) != 1:
        raise ShardMismatch(f'shards of different splits: {sorted(ns)}' if ns else 'no shard files')
    n = ns.pop()
    ks = sorted(s['shard'][0] for s in sources)
    if partial and len(set(ks)) != len(ks):
        raise ShardMismatch(f'shards {ks}: a shard given twice')
    if not partial and ks != list(range(n)):
        raise ShardMismatch(f'shards {ks} are not 0..{n - 1} each exactly once')
    return games, sources


def stage_decision(st, look, summary, results, candidate='candidate'):
    """The preregistered stage's verdict (yomibench.gate_seq) at `look` on this gate's core (the
    pooled score against the previous best) and its per-pairing matchups. Returns (decision, reason,
    verdict): continue, reject, pass (stage passed) or promote (a final stage passed), inconclusive."""
    from yomibench import gate_seq

    rs = [r for r in results if candidate in r.bots and 'previous-best' in r.bots]
    scores = [sum(r.score(candidate) for r in p) / 2 for p in rated_pairs(rs)]
    core = {**gate_seq.pair_stats(scores), 'score': summary['previous_best_score'], 'ci95': summary['ci95']}
    matchups = [{'characters': p['characters'], 'score': p['previous_best_score'], 'ci95': p['ci95']}
                for p in summary['pairings']]
    v = gate_seq.verdict(st, look, core, matchups)
    decision = {'pass': 'promote' if st.get('final') else 'pass'}.get(v['verdict'], v['verdict'])
    return decision, f"{st['name']} look {look + 1}/{len(st['looks'])}: {v['reason']}", v

def _game(d):
    from yomibench.arena import GameResult

    d = dict(d, bots=tuple(d['bots']), characters=tuple(d['characters']))
    d['hp'] = tuple(d['hp']) if d.get('hp') is not None else None
    return GameResult(**d)


def _play_procs(argv, args, specs, sched, outer=(0, 1)):
    """Run --procs children (see the module docstring); returns their games (dicts). With `outer`
    (k, n), the children split shard k of n: child j plays shard k + n*j of n*procs."""
    import subprocess
    import tempfile

    servers = [s for s in specs if s.split(':', 1)[0] not in IN_PROCESS]
    if servers and len(servers) < args.procs:
        raise SystemExit(f'--procs {args.procs} needs at least as many engine servers ({len(servers)} given)')
    tmp = Path(tempfile.mkdtemp(prefix='yomibench-gate-'))
    drop = {'--procs', '--engine', '--ledger', '--results', '--tag', '--owner', '--log', '--shard', '--shard-out',
            '--from-shards', '--seat-preflight'}
    base, skip = [], False
    for a in argv:  # the parent's options, minus those set per child
        if skip:
            skip = False
            continue
        if a.split('=', 1)[0] in drop:
            skip = '=' not in a
            continue
        if a in ('--below-normal', '--no-ledger'):
            continue
        base.append(a)
    children = []
    k0, n0 = outer
    n = n0 * args.procs
    for j in range(args.procs):
        mine = specs if not servers else servers[j::args.procs]
        cmd = [sys.executable, str(Path(__file__).resolve()), *base, '--engine', ','.join(mine),
               '--results', str(tmp / f'unused-{j}.jsonl'), '--no-ledger',
               '--shard', f'{k0 + n0 * j}/{n}', '--shard-out', str(tmp / f'games-{j}.jsonl'),
               '--seat-preflight', '0']  # the parent ran it
        flags = 0x00004000 if args.below_normal and sys.platform == 'win32' else 0  # BELOW_NORMAL_PRIORITY_CLASS
        children.append(subprocess.Popen(cmd, creationflags=flags))
    codes = [c.wait() for c in children]
    if any(codes):
        raise SystemExit(f'gate children failed with exit codes {codes}')
    games = []
    for j in range(args.procs):
        try:
            games += read_shard(tmp / f'games-{j}.jsonl', sched, k0 + n0 * j, n)[1]
        except ShardMismatch as e:
            raise SystemExit(f'gate child {j}: {e}') from e
    return games


def _append_games(path, games):
    with path.open('a', encoding='utf-8') as out:
        for d in games:
            out.write(json.dumps(d) + '\n')


class SeatPreflightFailed(Exception):
    pass


def _decide(bots, obs, engine, first):
    """Both bots' decisions on `obs` from a fresh turn cache, seat `first` searching first. Per seat:
    the pick's candidate record (expand ids aren't stable) and the strategy (bots that have one)."""
    from yomibench.bots import turn_candidates

    engine.turn_cache = {}
    ids = [None, None]
    for seat in (first, 1 - first):
        ids[seat] = bots[seat].choose(obs, seat, engine, random.Random(f'{obs.turn}:{seat}'))
    full = turn_candidates(obs, engine)
    out = []
    for seat in (0, 1):  # ids are per seat
        pick = next(c for c in full[seat] if c.id == ids[seat])
        search = getattr(bots[seat], 'search', None)
        strategy = search(obs, seat, engine)[0] if search else []
        out.append((ids[seat], pick.record(), list(strategy)))
    return out


def seat_preflight(spec, engine_spec, candidate_set, characters, seed, decisions, max_turns=1000, pairings=None,
                   match_settings=None):
    """The gate's seat check, before it plays: `spec` loaded twice (two bots, as in a gate between two
    nets) on each character pairing, for the first `decisions` decisions with a real choice. At each,
    both bots decide twice on the same state from fresh turn caches: seat 2 searching first, then
    seat 1 first (as arena.play_game asks). Each seat's pick and strategy must be the same both ways,
    or SeatPreflightFailed: a gate between two nets would then score seats unequally (as when
    seat 2's search once saw seat 1's set-4 expansions). Swapping two copies of one net between seats can't
    show that: the copies play the same game either way, as they did on the shared-expansion code.
    `pairings` ((P1, P2) matchups) replaces every pairing of `characters`. `match_settings`: the gate's
    (None: standard). Returns the decisions checked."""
    bots = [make_bot(spec), make_bot(spec)]
    if engine_spec.split(':', 1)[0] == 'batch':
        from yomibench.batch_play import batch_arena_engines

        engine = batch_arena_engines(bots, candidate_set, match_settings)(engine_spec)
    else:
        engine = _connect(engine_spec, candidate_set, match_settings)
    sets = tuple(getattr(b, 'candidate_set', None) for b in bots)
    checked = 0
    try:
        for a, b in pairings or [(a, b) for a in characters for b in characters]:
            obs = (engine.new_match(a, b, seed, candidate_sets=sets) if any(sets)
                   else engine.new_match(a, b, seed))
            n = 0
            while not obs.done and obs.turn < max_turns and n < decisions:
                if all(len(c) == 1 for c in obs.candidates):
                    obs = engine.step(obs.candidates[0][0].id, obs.candidates[1][0].id)
                    continue
                late = _decide(bots, obs, engine, first=1)
                usual = _decide(bots, obs, engine, first=0)  # last, so its ids are the engine's current ones
                for seat in (0, 1):
                    (_, rec, s), (_, rec2, s2) = usual[seat], late[seat]
                    if rec != rec2 or len(s) != len(s2) or any(abs(x - y) > 1e-9 for x, y in zip(s, s2)):
                        raise SeatPreflightFailed(
                            f'{a}:{b} turn {obs.turn}: the seat {seat + 1} bot decides differently when '
                            f'seat {2 - seat} searches first ({len(s)} vs {len(s2)} strategy entries, '
                            f'{"same" if rec == rec2 else "different"} pick)')
                n += 1
                obs = engine.step(usual[0][0], usual[1][0])
            checked += n
    finally:
        close = getattr(engine, 'close', None)
        if close:
            close()
    return checked


def anchor_pool(previous_best, ckpt1, ckpt2, history=()):
    """[(label, spec)] with repeated specs dropped (first label kept), and {dropped label: kept label}."""
    pool, aliases, seen = [], {}, {}
    entries = [('previous-best', previous_best), ('ckpt0', 'matrix-do'), ('ckpt1', ckpt1), ('ckpt2', ckpt2)]
    entries += [(f'history-{i}', spec) for i, spec in enumerate(history)]
    for label, spec in entries:
        key = _spec_key(spec)
        if key in seen:
            aliases[label] = seen[key]
            continue
        seen[key] = label
        pool.append((label, spec))
    return pool, aliases


def _spec_key(spec):
    name, _, rest = spec.partition(':')
    model, at, suffix = rest.partition('@')
    return (name, str(Path(model).resolve()).lower() + at + suffix) if model else (name, at + suffix)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidate', required=True, help='bot spec, e.g. matrix-do:CHECKPOINT.npz')
    ap.add_argument('--previous-best', required=True, help='previous best bot spec')
    ap.add_argument('--ckpt1', help='anchor: checkpoint 1 bot spec (required unless --quick)')
    ap.add_argument('--ckpt2', help='anchor: checkpoint 2 bot spec (required unless --quick)')
    ap.add_argument('--rule', choices=RULES, default='pooled',
                    help='overall promotion rule (see the module docstring); pairing is the original rule')
    ap.add_argument('--quick', action='store_true',
                    help='a quick gate: the candidate against the previous best only, with the same criteria and '
                         'promote rule on that one matchup (for engines too slow for the whole pool); a promotion '
                         'from it still owes a full pool gate')
    ap.add_argument('--history', action='append', default=[],
                    help='extra opponent bot spec, reported but not part of the decision; repeatable')
    ap.add_argument('--engine', default='toy', help='arena engine specs, comma separated')
    ap.add_argument('--characters', default='Toy', help='comma-separated; every ordered pairing is played')
    ap.add_argument('--candidate-set', type=int, default=DEFAULT_CANDIDATE_SET, metavar='N',
                    help="candidate set every game plays (default: self-play's default, the engine's own)")
    ap.add_argument('--match-settings', type=_match_settings, metavar='JSON',
                    help='match settings for every game (e.g. the ranked preset \'{"stage_width":600,"game_length":3000,'
                         '"starting_meter":"1.5"}\'), as selfplay --match-settings; part of the schedule, so shards '
                         'agree on it. Default: the engine\'s standard settings')
    ap.add_argument('--set5', metavar='SWITCHES',
                    help="candidate set 5's switches (selfplay --set5's forms; sets $YOMISIM_SET5 for this run and its "
                         "children); part of the schedule. Default: $YOMISIM_SET5 as it is")
    ap.add_argument('--pairs', type=int, default=None,
                    help='seat-swapped pairs per opponent and character pairing (default 200)')
    ap.add_argument('--pairs-by-pairing', metavar='P1:P2=N,...',
                    help='instead of --pairs: pairs per opponent in each character pairing, every pairing of '
                         '--characters (see above)')
    ap.add_argument('--cowboy-mirror-pairs', type=int, metavar='N',
                    help='--quick, with --cowboy-vs, instead of --pairs: the Cowboy gate\'s seat-swapped Cowboy:Cowboy '
                         'pairs (see above)')
    ap.add_argument('--cowboy-vs', metavar='X=N,...',
                    help='the Cowboy gate\'s candidate-character cells: per X, N pairs with the candidate on Cowboy '
                         'in both games (see above)')
    ap.add_argument('--cowboy-extension', metavar='X=N,...',
                    help='with the Cowboy gate\'s flags: play only its cell extension, N more pairs in each cell X '
                         '(see above)')
    ap.add_argument('--min-pairs', type=int, default=None,
                    help='complete pairs each anchor matchup needs (default 75%% of those scheduled)')
    ap.add_argument('--max-failure-rate', type=float, default=0.05,
                    help='most scheduled pairs per anchor matchup that may hit an engine halt (default 0.05)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-turns', type=int, default=1000)
    ap.add_argument('--results', type=Path, required=True,
                    help='JSONL file for game rows and the summary (was --ledger)')
    ap.add_argument('--procs', type=int, default=1, help='processes to play the schedule in (see above)')
    ap.add_argument('--ecoqos', action='store_true',
                    help='Windows: keep the default power throttling (EcoQoS); by default gate processes opt out, so '
                         'Windows 11 runs them on P-cores as well as E-cores (yomibench.winsched)')
    ap.add_argument('--below-normal', action='store_true', help='run --procs children at BelowNormal priority (Windows)')
    ap.add_argument('--shard', metavar='K/N', help='play only pairs K mod N of the schedule into --shard-out')
    ap.add_argument('--shard-out', type=Path, help='with --shard K/N: the shard file to write (see above)')
    ap.add_argument('--seat-preflight', type=int, default=20, metavar='N',
                    help='before playing: check N decisions per pairing between two copies of the candidate, '
                         'each seat deciding the same whichever seat searches first, or exit 4 (0: skip)')
    ap.add_argument('--partial', action='store_true',
                    help='with --from-shards: any of one split\'s shards, each at most once (a stage look)')
    ap.add_argument('--stage', type=Path, metavar='STAGE.json',
                    help='with --from-shards --quick: the gate job\'s preregistered stage (yomibench.gate_seq); '
                         'decide at --look K: exit 0 pass (or promote at a final stage), 1 reject, 5 play on')
    ap.add_argument('--look', type=int, default=None, help='with --stage: the look index (0-based)')
    ap.add_argument('--from-shards', help='comma-separated shard files: decide from them instead of playing')
    ap.add_argument('--record-dir', type=Path, metavar='DIR',
                    help='also write every game played here as a format-3 record in DIR (arena.play_game; '
                         '--procs children and --shard runs write theirs too); decisions are unchanged')
    ledger.add_args(ap)
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(str(a).startswith('--sampled-starts') for a in (sys.argv[1:] if argv is None else argv)):
        # Gates, screens and the control always play from the opening (yomibench.sampled_starts is self-play only).
        ap.error('--sampled-starts is self-play only: gates, screens and the control never use sampled starts')
    args = ap.parse_args(argv)
    if args.set5:  # canonical text (the schedule keeps it); the engines read $YOMISIM_SET5
        try:
            args.set5 = set5_spec(args.set5)
        except ValueError as e:
            ap.error(f'--set5: {e}')
        os.environ['YOMISIM_SET5'] = args.set5
    if not args.ecoqos:  # every process: --procs children and --shard runs come through here too
        from yomibench.winsched import opt_out_of_ecoqos

        opt_out_of_ecoqos()
    characters = args.characters.split(',')
    if args.pairs_by_pairing is not None:  # parsed here: from now on {"P1:P2": pairs}, and --pairs their total
        if args.pairs is not None:
            ap.error('--pairs and --pairs-by-pairing are exclusive')
        try:
            args.pairs_by_pairing = gate_seq.check_pairs_by_pairing(
                gate_seq.parse_pairs_by_pairing(args.pairs_by_pairing), characters)
        except ValueError as e:
            ap.error(f'--pairs-by-pairing: {e}')
        args.pairs = sum(args.pairs_by_pairing.values())
    elif args.pairs is None and args.cowboy_mirror_pairs is None:
        args.pairs = 200
    args.cowboy = None
    if args.cowboy_mirror_pairs is not None or args.cowboy_vs is not None:  # parsed here: {mirror_pairs, cowboy_vs}
        if args.pairs is not None or args.pairs_by_pairing or args.min_pairs is not None:
            ap.error('the Cowboy gate (--cowboy-mirror-pairs, --cowboy-vs) replaces --pairs, --pairs-by-pairing and '
                     '--min-pairs')
        if not args.quick or args.cowboy_mirror_pairs is None or args.cowboy_vs is None:
            ap.error('the Cowboy gate needs --quick, --cowboy-mirror-pairs and --cowboy-vs')
        try:
            vs = gate_seq.parse_cowboy_vs(args.cowboy_vs)
            gate_seq.cowboy(1, args.cowboy_mirror_pairs, vs, strict=False)
        except ValueError as e:
            ap.error(f'--cowboy-mirror-pairs / --cowboy-vs: {e}')
        args.cowboy = {'mirror_pairs': args.cowboy_mirror_pairs, 'cowboy_vs': vs}
        if characters != gate_seq.cowboy_characters(args.cowboy):
            ap.error(f"--characters must be {','.join(gate_seq.cowboy_characters(args.cowboy))} for this Cowboy gate "
                     f"(Cowboy, then each X in --cowboy-vs order)")
        args.pairs = args.cowboy_mirror_pairs + sum(vs.values())
    if args.cowboy_extension is not None:
        if args.cowboy is None:
            ap.error('--cowboy-extension extends a Cowboy gate: it needs --cowboy-mirror-pairs and --cowboy-vs')
        try:
            args.cowboy['extension'] = gate_seq.check_extension(args.cowboy['cowboy_vs'],
                                                                gate_seq.parse_cowboy_vs(args.cowboy_extension))
        except ValueError as e:
            ap.error(f'--cowboy-extension: {e}')
        args.pairs = sum(args.cowboy['extension'].values())
    extension = bool(args.cowboy and args.cowboy.get('extension'))
    if args.pairs < 1 or args.max_turns < 1 or (args.min_pairs is not None and args.min_pairs < 1):
        ap.error('--pairs, --min-pairs and --max-turns must be positive')
    if not 0 <= args.max_failure_rate < 1:
        ap.error('--max-failure-rate must be in [0, 1)')
    try:
        tags = ledger.parse_tags(args.tag)
    except ValueError as e:
        ap.error(str(e))
    ledger_path = ledger.path_from_args(args)
    started, load = ledger.utc_now(), (ledger.cpu_load() if ledger_path else None)
    specs = expand_specs(args.engine)
    pairings = ([pr for _, prs, _ in _cowboy_cell_spec(args.cowboy) for pr in prs] if args.cowboy
                else [(a, b) for a in characters for b in characters])
    # Criteria apply per pairing: each anchor matchup schedules --pairs pairs in every pairing (or
    # its --pairs-by-pairing count, the criteria's pairs then {"P1:P2": pairs}).
    scheduled = args.pairs_by_pairing or args.pairs
    if args.pairs_by_pairing:
        min_pairs = {k: args.min_pairs if args.min_pairs is not None else math.ceil(0.75 * n)
                     for k, n in args.pairs_by_pairing.items()}
    else:
        min_pairs = args.min_pairs if args.min_pairs is not None else math.ceil(0.75 * args.pairs)
    total = args.pairs if args.pairs_by_pairing or args.cowboy else args.pairs * len(pairings)  # per opponent
    if args.quick:
        pool, aliases, anchors = [('previous-best', args.previous_best)], {}, QUICK_ANCHORS
    else:
        if not (args.ckpt1 and args.ckpt2):
            ap.error('--ckpt1 and --ckpt2 are required (or --quick)')
        pool, aliases = anchor_pool(args.previous_best, args.ckpt1, args.ckpt2, args.history)
        anchors = ANCHORS
    if args.from_shards and (args.shard or args.procs > 1):
        ap.error('--from-shards plays nothing: no --shard or --procs')
    if (args.partial or args.stage) and not args.from_shards:
        ap.error('--partial and --stage decide from --from-shards')
    if args.stage and (not args.quick or args.look is None):
        ap.error('--stage needs --quick (the previous best is the core matchup) and --look K')
    stage = json.loads(args.stage.read_text(encoding='utf-8')) if args.stage else None
    if stage is not None and not 0 <= args.look < len(stage['looks']):
        ap.error(f"--look must be 0..{len(stage['looks']) - 1}")
    if stage is not None and stage.get('pairs_by_pairing') != args.pairs_by_pairing:
        ap.error(f"--pairs-by-pairing {args.pairs_by_pairing} is not the stage's {stage.get('pairs_by_pairing')}")
    if stage is not None and args.pairs != stage['pairs']:
        ap.error(f"--pairs {args.pairs} is not the stage's {stage['pairs']}")
    if stage is not None and (args.cowboy is not None) != (gate_seq.is_cowboy(stage) or gate_seq.is_extension(stage)):
        ap.error('a Cowboy gate stage needs the Cowboy gate flags, and they need its stage')
    if stage is not None and extension != gate_seq.is_extension(stage):
        ap.error('a cell extension stage needs --cowboy-extension, and it needs its stage')
    if extension and stage is None and not args.shard:
        ap.error('--cowboy-extension plays shards (--shard), or decides from them at its --stage')
    if stage is not None and extension:
        if (stage.get('mirror_pairs') != args.cowboy['mirror_pairs']
                or list((stage.get('cowboy_vs') or {}).items()) != list(args.cowboy['cowboy_vs'].items())
                or list((stage.get('extension_vs') or {}).items()) != list(args.cowboy['extension'].items())):
            ap.error(f"--cowboy-mirror-pairs / --cowboy-vs / --cowboy-extension are not the stage's extension "
                     f"({stage.get('mirror_pairs')}, {stage.get('cowboy_vs')}, {stage.get('extension_vs')})")
    elif stage is not None and args.cowboy is not None:
        try:
            want = gate_seq.cowboy(stage['seed'], args.cowboy['mirror_pairs'], args.cowboy['cowboy_vs'],
                                   guard_z=stage['guard_z'], min_pairs=stage['min_pairs_per_pairing'])[0]
        except (KeyError, ValueError) as e:
            ap.error(f'--stage: not a valid Cowboy gate stage ({e})')
        if json.loads(json.dumps(want)) != stage or list(stage['cowboy_vs']) != list(want['cowboy_vs']):
            ap.error(f"--cowboy-mirror-pairs {args.cowboy['mirror_pairs']} --cowboy-vs {args.cowboy_vs} is not the "
                     f"stage's Cowboy gate ({stage.get('mirror_pairs')}, {stage.get('cowboy_vs')}), or the stage's "
                     f"bounds aren't the plan's")
    if args.cowboy and not args.from_shards:  # every run that plays a Cowboy gate: before any game
        want = gate_seq.cowboy_characters(args.cowboy)
        for spec in dict.fromkeys(specs):
            have = engine_characters(spec, args.candidate_set, args.candidate, args.match_settings)
            missing = [c for c in want if c not in (have or [])]
            if missing:
                print(f'engine {spec} does not play {", ".join(missing)} (its hello lists '
                      f'{", ".join(have) if have else "no characters"}); this gate plays {", ".join(want)}. '
                      f'No games played, no decision', file=sys.stderr)
                return ENGINE_CHARACTERS
    if args.seat_preflight > 0 and not args.from_shards:  # every run that plays games
        t0 = time.perf_counter()
        try:
            n = seat_preflight(args.candidate, specs[0], args.candidate_set, characters, args.seed,
                               args.seat_preflight, args.max_turns, pairings if args.cowboy else None,
                               args.match_settings)
        except SeatPreflightFailed as e:
            print(f'seat preflight FAILED: {e}; no games played, no decision', file=sys.stderr)
            return SEAT_PREFLIGHT
        print(f'seat preflight: {n} decisions over {len(pairings)} pairings, each seat the same whichever seat '
              f'searches first ({time.perf_counter() - t0:.0f}s)', flush=True)
    if args.shard:  # play this share into a shard file, leave deciding to whoever merges the shards
        try:
            k, n = (int(x) for x in args.shard.split('/'))
        except ValueError:
            ap.error('--shard wants K/N')
        if not 0 <= k < n or args.shard_out is None:
            ap.error('--shard K/N needs 0 <= K < N and --shard-out')
        sched = schedule(args, pool, characters)
        if args.procs > 1:
            games = _play_procs(argv, args, specs, sched, (k, n))
            write_shard_header(args.shard_out, sched, k, n, args.engine)
            _append_games(args.shard_out, games)
        else:
            write_shard_header(args.shard_out, sched, k, n, args.engine)
            _play(args, pool, specs, characters, (k, n), args.shard_out, [])
        return 0
    sources = None
    if args.from_shards:
        try:
            games, sources = load_shards([Path(f) for f in args.from_shards.split(',') if f],
                                         schedule(args, pool, characters), partial=args.partial)
        except ShardMismatch as e:
            print(f'--from-shards: {e}; no decision', file=sys.stderr)
            return SHARD_MISMATCH
    rule = None
    if extension:  # a cell extension: its stage's rule
        rule = stage
        print('predeclared criteria (Cowboy gate cell extension):\n' + '\n'.join(
            f'  {k}: {rule[k]}' for k in ('cells', 'rule', 'cell_floor', 'main_cells')), flush=True)
    elif args.cowboy:  # the stage's rule, or the plan's at its final look
        rule = stage or gate_seq.cowboy(1, args.cowboy['mirror_pairs'], args.cowboy['cowboy_vs'], strict=False)[0]
        print('predeclared criteria (Cowboy gate):\n' + '\n'.join(
            f'  {k}: {rule[k]}' for k in ('cells', 'rule', 'reject_z', 'accept_z', 'final_z', 'guard_z',
                                          'cell_block_upper', 'not_worse', 'min_pairs_per_pairing')), flush=True)
    else:
        declared = criteria(min_pairs, args.max_failure_rate, scheduled, anchors)
        print('predeclared criteria:\n' + '\n'.join(f'  {k}: {declared[k]}' for k in
                                                     ('sufficient_data', 'non_inferiority', 'improvement', 'decision')),
              flush=True)
    hellos = []
    if sources is not None:
        _append_games(args.results, games)
        results = [_game(d) for d in games]
        print('decided from shards: ' + ', '.join(f"{s['shard'][0]}/{s['shard'][1]} ({s['host']}, {s['games']} games)"
                                                  for s in sources))
    elif args.procs > 1:
        games = _play_procs(argv, args, specs, schedule(args, pool, characters))
        _append_games(args.results, games)
        results = [_game(d) for d in games]
        try:  # the engine's hello, for the ledger row
            if specs[0].split(':', 1)[0] == 'batch':
                from yomibench.batch_play import batch_arena_engines

                e = batch_arena_engines([make_bot(args.candidate)], args.candidate_set, args.match_settings)(specs[0])
            else:
                e = _connect(specs[0], args.candidate_set, args.match_settings)
            hellos.append(e.hello())
            e.close()
        except Exception:  # noqa: BLE001  (only provenance)
            pass
    else:
        results = _play(args, pool, specs, characters, None, args.results, hellos)
    # Bot configs for the ledger row (the bots themselves were freed after their games).
    opponents = [SimpleNamespace(name=label, config=lambda c=_config(spec): c) for label, spec in pool]
    candidate = SimpleNamespace(name='candidate', config=lambda c=_config(args.candidate): c)
    if extension:
        summary = extension_summary(results, rule, args.max_failure_rate)
    elif rule is not None:
        summary = cowboy_summary(results, rule, args.look if stage is not None else len(rule['looks']) - 1,
                                 args.max_failure_rate)
    else:
        summary = gate_summary(results, scheduled, min_pairs, args.max_failure_rate, aliases, pairings=pairings,
                               anchors=anchors, rule=args.rule)
    summary['mode'] = 'quick' if args.quick else 'pool'
    summary.update(candidate=args.candidate, pool=dict(pool), aliases=aliases, engine=args.engine, seed=args.seed,
                   pairs=args.pairs, characters=characters, max_turns=args.max_turns,
                   candidate_set=args.candidate_set)
    if args.match_settings:
        summary['match_settings'] = args.match_settings
    if args.set5:
        summary['set5'] = args.set5
    if args.pairs_by_pairing:
        summary['pairs_by_pairing'] = args.pairs_by_pairing
    if args.cowboy:
        summary['cowboy'] = args.cowboy
        if stage is None:
            del summary['stage']  # the rule decided as at its final look; there is no stage
    if sources is not None:
        summary['shards'] = sources
    if stage is not None and args.cowboy:
        summary['stage'].update(plan=stage, scheduled_pairs=total)
    elif stage is not None:
        gate_decision, gate_reason = summary['decision'], summary['reason']
        summary['decision'], summary['reason'], summary['stage'] = stage_decision(stage, args.look, summary, results)
        summary['stage'].update(gate_rule={'decision': gate_decision, 'reason': gate_reason}, plan=stage,
                                scheduled_pairs=total)
    with args.results.open('a', encoding='utf-8') as out:
        out.write(json.dumps(summary, allow_nan=False) + '\n')
    print(format_table(summary))
    print(f"gate: {summary['decision'].upper()} ({summary['reason']})")
    if ledger_path:  # a look that plays on is logged too, tagged stage=gate-look so it never counts as a gate result
        interim = summary['decision'] == 'continue'
        gate = {k: summary[k] for k in ('decision', 'reason', 'criteria', 'improvement', 'pairings', 'checks',
                                        'previous_best_score', 'ci95', 'completed_pairs_vs_previous', 'matchups',
                                        'candidate_set', 'mode')}
        for k in ('stage', 'cells', 'pooled', 'rule_path', 'blocked_by', 'unguarded', 'cowboy', 'cell_floor'):
            if k in summary:
                gate[k] = summary[k]
        settings = {k: v for k, v in vars(args).items() if k not in ('ledger', 'no_ledger', 'tag', 'owner', 'log')}
        ledger.append_arena(
            results, [candidate, *opponents], path=ledger_path,
            tags={'stage': 'gate', **tags, **({'stage': 'gate-look', 'interim': 'true'} if interim else {})},
            settings=settings,
            started_utc=started, load=load, owner=args.owner, hello=hellos[0] if hellos else None,
            bot_specs=[args.candidate, *(spec for _, spec in pool)], anchor='ckpt0', engines=len(specs),
            scheduled=len(pool) * total * 2, print_matchups=False,
            artifacts={'results': args.results, 'log': ledger.log_from_args(args)},
            extra_metrics={'gate': gate, 'pool': dict(pool), 'aliases': aliases, 'max_turns': args.max_turns,
                           **({'shards': sources} if sources is not None else {})})
    return CONTINUE if summary['decision'] == 'continue' else EXIT[summary['decision']]


if __name__ == '__main__':
    raise SystemExit(main())

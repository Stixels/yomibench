"""Promotion gate: predeclared criteria, fail closed, paired scores over complete pairs only."""
import importlib.util
import json
from pathlib import Path

import pytest

from yomibench import ledger
from yomibench.arena import GameResult

spec = importlib.util.spec_from_file_location('pool_eval', Path(__file__).parents[1] / 'tools/pool_eval.py')
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def pair(seed, outcomes, opponent):
    return [GameResult(seats, ('Toy', 'Toy'), seed, outcome, 1, (1, 1), 0.0)
            for seats, outcome in zip([('candidate', opponent), (opponent, 'candidate')], outcomes)]


WIN, LOSS, SPLIT, HALT = ('p1', 'p2'), ('p2', 'p1'), ('p1', 'p1'), ('halt', 'p1')  # pair scores 1, 0, 0.5, -


def games(opponent, wins=0, losses=0, splits=0, halts=0, truncs=0):
    kinds = [WIN] * wins + [LOSS] * losses + [SPLIT] * splits + [HALT] * halts + [('truncated', 'p1')] * truncs
    return [r for seed, k in enumerate(kinds) for r in pair(seed, k, opponent)]


def anchors(prev=(120, 80), others=(100, 100), **override):
    """200 pairs against each anchor: (wins, losses) for the previous best and for the rest."""
    out = games('previous-best', *prev)
    for a in ('ckpt0', 'ckpt1', 'ckpt2'):
        out += override.get(a, games(a, *others))
    return out


def test_promote_needs_improvement_and_non_inferiority_everywhere():
    s = pool.gate_summary(anchors(), 200)
    assert s['decision'] == 'promote' and s['improvement'] and s['ci95'][0] > 0.5
    assert all(c['sufficient_data'] and c['non_inferior'] for c in s['checks'].values())
    assert pool.EXIT[s['decision']] == 0


def test_non_inferior_without_improvement_is_inconclusive_not_promote():
    s = pool.gate_summary(anchors(prev=(100, 100)), 200)
    assert s['decision'] == 'inconclusive' and not s['improvement'] and 'no significant improvement' in s['reason']


def test_a_regression_against_any_anchor_rejects():
    s = pool.gate_summary(anchors(ckpt1=games('ckpt1', 80, 120)), 200)  # 0.40 vs ckpt1
    assert s['decision'] == 'reject' and 'regression vs ckpt1' in s['reason'] and s['improvement']
    s = pool.gate_summary(anchors(ckpt0=games('ckpt0', 460, 540)), 1000)  # 0.46, but upper bound < 0.50
    assert s['decision'] == 'reject' and s['checks']['ckpt0']['ci95'][1] < 0.5
    s = pool.gate_summary(anchors(prev=(88, 112)), 200)  # 0.44 vs the previous best
    assert s['decision'] == 'reject' and 'regression vs previous-best' in s['reason']
    assert pool.EXIT['reject'] == 1


def test_too_few_pairs_or_too_many_halts_is_inconclusive_never_promote():
    assert pool.gate_summary([], 200)['decision'] == 'inconclusive'  # zero pairs
    s = pool.gate_summary(anchors(ckpt2=games('ckpt2', 70, 70)), 200)  # 140 < 150 complete pairs
    assert s['decision'] == 'inconclusive' and 'insufficient data vs ckpt2' in s['reason']
    s = pool.gate_summary(anchors(ckpt2=games('ckpt2', 95, 94, halts=11)), 200)  # 5.5% halted pairs
    assert s['decision'] == 'inconclusive' and s['checks']['ckpt2']['failure_rate'] == 0.055
    s = pool.gate_summary(anchors(ckpt2=games('ckpt2', 95, 95, halts=10)), 200)  # 5% is allowed
    assert s['decision'] == 'promote'
    # Truncations aren't failures, but their pairs don't count as complete either.
    s = pool.gate_summary(anchors(ckpt2=games('ckpt2', 70, 70, truncs=60)), 200)
    assert s['decision'] == 'inconclusive' and s['checks']['ckpt2']['failure_rate'] == 0.0
    # A clear regression still rejects even if another matchup is short of data (fail closed).
    s = pool.gate_summary(anchors(ckpt1=games('ckpt1', 40, 160), ckpt2=games('ckpt2', 5, 5)), 200)
    assert s['decision'] == 'reject'


def test_an_anchor_played_once_under_an_alias_is_judged_on_that_matchup():
    results = games('previous-best', 120, 80) + games('ckpt0', 100, 100) + games('ckpt1', 100, 100)
    s = pool.gate_summary(results, 200, aliases={'ckpt2': 'previous-best'})
    assert s['decision'] == 'promote' and s['checks']['ckpt2']['played_as'] == 'previous-best'
    assert pool.gate_summary(results, 200)['decision'] == 'inconclusive'  # ckpt2 never played


def test_history_opponents_are_reported_but_never_decide():
    s = pool.gate_summary(anchors() + games('history-0', 0, 200), 200)
    assert s['decision'] == 'promote' and {m['opponent'] for m in s['matchups']} >= {'history-0'}


def test_every_matchup_gets_a_paired_score_and_interval():
    s = pool.gate_summary(anchors(ckpt0=games('ckpt0', 50, 150)), 200)
    by = {m['opponent']: m for m in s['matchups']}
    assert by['previous-best']['score'] == 0.6 and by['ckpt0']['score'] == 0.25 and by['ckpt1']['score'] == 0.5
    assert all(m['ci95'][0] <= m['score'] <= m['ci95'][1] for m in by.values())


def test_the_anchor_pool_is_fixed_and_plays_a_repeated_spec_once(tmp_path):
    ckpt2 = tmp_path / 'ckpt2.npz'
    p, aliases = pool.anchor_pool(f'matrix-do:{ckpt2}', 'matrix-do:ckpt1.npz', f'matrix-do:{tmp_path}/./ckpt2.npz')
    assert [label for label, _ in p] == ['previous-best', 'ckpt0', 'ckpt1'] and aliases == {'ckpt2': 'previous-best'}


def test_pool_cli_prints_criteria_and_records_games_summary_and_ledger(tmp_path, capsys):
    results, metrics = tmp_path / 'pool.jsonl', tmp_path / 'metrics.jsonl'
    argv = ['--candidate', 'random', '--previous-best', 'greedy', '--ckpt1', 'random', '--ckpt2', 'greedy',
            '--history', 'matrix', '--pairs', '1', '--max-turns', '2', '--results', str(results),
            '--ledger', str(metrics), '--tag', 'gate=toy']
    assert pool.main(argv) == 2  # every game truncated: inconclusive
    out = capsys.readouterr().out
    assert out.index('predeclared criteria') < out.index('gate: INCONCLUSIVE')
    rows = [json.loads(x) for x in results.read_text().splitlines()]
    s = rows[-1]
    assert len(rows) == 9  # previous-best, ckpt0, ckpt1, history-0 (ckpt2 repeats the previous best): 2 games each
    assert s['outcomes'] == {'truncated': 8} and s['aliases'] == {'ckpt2': 'previous-best'}
    assert s['criteria']['min_pairs'] == 1 and s['criteria']['scheduled_pairs'] == 1
    from yomibench import ledger as metrics_ledger

    (row,) = [json.loads(x) for x in metrics.read_text().splitlines()]
    assert metrics_ledger.validate_row(row) == [] and row['tags'] == {'stage': 'gate', 'gate': 'toy'}
    assert row['metrics']['gate']['decision'] == 'inconclusive' and 'non_inferiority' in row['metrics']['gate']['criteria']
    assert row['provenance']['counts']['requested'] == 8
    with pytest.raises(SystemExit):  # the anchors are required
        pool.main(['--candidate', 'random', '--previous-best', 'greedy', '--results', str(results)])


def test_defaults_are_200_pairs_and_150_to_decide(monkeypatch, tmp_path):
    seen = set()

    def fake_round_robin(connects, bots, characters, pairs, seed, results, max_turns, shard=None, record_dir=None):
        seen.add(pairs)
        return []
    monkeypatch.setattr(pool, 'run_round_robin', fake_round_robin)
    monkeypatch.setattr(pool, 'make_bot', lambda spec: type('B', (), {'name': spec})())
    code = pool.main(['--candidate', 'c', '--previous-best', 'p', '--ckpt1', 'a', '--ckpt2', 'b',
                      '--results', str(tmp_path / 'r.jsonl'), '--no-ledger', '--seat-preflight', '0'])
    summary = json.loads((tmp_path / 'r.jsonl').read_text().splitlines()[-1])
    assert seen == {200} and summary['criteria']['min_pairs'] == 150 and summary['criteria']['max_failure_rate'] == 0.05
    assert code == 2 and summary['decision'] == 'inconclusive'


SHARD_BASE = ['--candidate', 'random', '--previous-best', 'greedy', '--quick', '--pairs', '3', '--max-turns', '30',
              '--characters', 'Toy', '--no-ledger']


def _games(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    return sorted((tuple(r['bots']), r['seed'], r['outcome'], r['turns']) for r in rows if 'outcome' in r)


def _summary(path):
    s = json.loads(Path(path).read_text().splitlines()[-1])
    return {k: s[k] for k in ('decision', 'reason', 'pairings', 'matchups', 'checks', 'ci95')}


def test_shards_from_several_runs_decide_like_one_run(tmp_path):
    one = tmp_path / 'one.jsonl'
    code = pool.main(SHARD_BASE + ['--results', str(one)])
    shards = []
    for k, procs in ((0, '1'), (1, '2')):  # shard 1 split again over two processes
        out = tmp_path / f's{k}.jsonl'
        assert pool.main(SHARD_BASE + ['--shard', f'{k}/2', '--shard-out', str(out), '--procs', procs,
                                       '--results', str(tmp_path / 'unused')]) == 0
        head = json.loads(out.read_text().splitlines()[0])
        assert head['type'] == 'shard_header' and head['shard'] == [k, 2]
        assert head['code'] == ledger.git_info()  # the commit that played the shard's games
        shards.append(str(out))
    merged = tmp_path / 'merged.jsonl'
    assert pool.main(SHARD_BASE + ['--from-shards', ','.join(shards[::-1]), '--results', str(merged)]) == code
    assert _games(merged) == _games(one) and len(_games(one)) == 6
    assert _summary(merged) == _summary(one)
    s = json.loads(merged.read_text().splitlines()[-1])
    assert sorted(x['shard'][0] for x in s['shards']) == [0, 1]


def test_from_shards_fails_closed_on_any_mismatch(tmp_path, capsys):
    def shard(k, n, *extra):
        out = tmp_path / f's{k}-{n}-{len(extra)}.jsonl'
        pool.main(SHARD_BASE + list(extra) + ['--shard', f'{k}/{n}', '--shard-out', str(out),
                                               '--results', str(tmp_path / 'unused')])
        return str(out)

    s0, s1 = shard(0, 2), shard(1, 2)
    other_seed = shard(1, 2, '--seed', '7')
    third = shard(2, 3)
    tampered = tmp_path / 'tampered.jsonl'
    lines = Path(s1).read_text().splitlines()
    tampered.write_text('\n'.join(lines[:-1]) + '\n')  # a game missing
    for files in ([s0, other_seed], [s0, s0], [s0], [s0, s1, third], [s0, str(tampered)],
                  [s0, str(tmp_path / 'absent.jsonl')]):
        results = tmp_path / 'r.jsonl'
        results.unlink(missing_ok=True)
        assert pool.main(SHARD_BASE + ['--from-shards', ','.join(files), '--results', str(results)]) == pool.SHARD_MISMATCH
        assert not results.exists()  # no games, no summary: nothing decided
    assert 'differs in seed' in capsys.readouterr().err
    results = tmp_path / 'r.jsonl'  # a different candidate than the shards played
    other = SHARD_BASE[:1] + ['greedy'] + SHARD_BASE[2:]
    assert pool.main(other + ['--from-shards', f'{s0},{s1}', '--results', str(results)]) == pool.SHARD_MISMATCH
    assert pool.main(SHARD_BASE + ['--from-shards', f'{s0},{s1}', '--results', str(results)]) in pool.EXIT.values()


def test_bot_identity_names_models_by_content(tmp_path):
    a, b = tmp_path / 'a.npz', tmp_path / 'sub' / 'b.npz'
    b.parent.mkdir()
    a.write_bytes(b'model')
    b.write_bytes(b'model')
    assert pool.bot_identity(f'matrix-do:{a}') == pool.bot_identity(f'matrix-do:{b}') != f'matrix-do:{a}'
    assert pool.bot_identity('matrix-do') == 'matrix-do'
    other = tmp_path / 'elsewhere' / 'same.npz'  # a quick-turn spec of the same file at another path
    other.parent.mkdir()
    other.write_bytes(a.read_bytes())
    cheap = pool.bot_identity(f'matrix-do:{a}@cheap=0.75:32')
    assert cheap == pool.bot_identity(f'matrix-do:{other}@cheap=0.75:32') and cheap.endswith('@cheap=0.75:32')
    assert cheap != pool.bot_identity(f'matrix-do:{a}')


def test_pool_eval_pins_math_threads_before_numpy():
    """Gate children run one search each, like self-play: no math library may start a thread per core."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS") and k != "VECLIB_MAXIMUM_THREADS"}
    env["PYTHONPATH"] = str(repo / "src")
    code = ("import sys; sys.path.insert(0, 'tools'); import os, pool_eval; "
            "assert all(os.environ[v] == '1' for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', "
            "'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))")
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_a_stage_look_decides_from_some_microshards_and_the_last_look_from_all(tmp_path):
    from yomibench import gate_seq

    def shards(cand, prev):
        base = ['--candidate', cand, '--previous-best', prev, '--quick', '--pairs', '16', '--characters', 'Toy',
                '--no-ledger']
        out = []
        for k in range(4):
            f = tmp_path / f'{cand}-s{k}.jsonl'
            assert pool.main(base + ['--shard', f'{k}/4', '--shard-out', str(f), '--results', str(tmp_path / 'u')]) == 0
            out.append(str(f))
        return base, out

    def look(base, files, st, k, name):
        plan = tmp_path / f'{name}.stage.json'
        plan.write_text(json.dumps(st))
        res = tmp_path / f'{name}.jsonl'
        code = pool.main(base + ['--stage', str(plan), '--look', str(k), '--partial', '--from-shards', ','.join(files),
                                 '--results', str(res)])
        return code, json.loads(res.read_text().splitlines()[-1])

    screen = gate_seq.stage('screen', 'cheap', 11, 16, gate_seq.LOOKS, reject_z=gate_seq.REJECT_Z, guard_upper=0.5)
    final = gate_seq.stage('confirm', 'full', 12, 16, (0.25, 0.5, 0.75, 1.0), final=True)
    base, weak = shards('random', 'greedy')  # 4 of 16 pairs per shard: look 0 (25%) is due with one shard
    code, s = look(base, weak[:1], screen, 0, 'weak-early')
    assert code == pool.EXIT['reject'] and s['decision'] == 'reject' and s['stage']['verdict'] == 'reject'
    assert '4.048' in s['reason'] and s['stage']['scheduled_pairs'] == 16  # mean 0 + 4.048 x SE < 0.5
    base, strong = shards('greedy', 'random')
    code, s = look(base, strong[:1], final, 0, 'strong-early')
    assert code == pool.CONTINUE and s['decision'] == 'continue'  # no early accept
    code, s = look(base, strong, final, 3, 'strong-last')
    assert code == 0 and s['decision'] == 'promote' and s['stage']['gate_rule']['decision'] == 'promote'
    code, s = look(base, strong, screen, 3, 'strong-screen')
    assert code == 0 and s['decision'] == 'pass'  # a screen's pass is not yet a promotion
    plan = tmp_path / 'p.json'
    plan.write_text(json.dumps(screen))
    assert pool.main(base + ['--stage', str(plan), '--look', '0', '--partial', '--from-shards',
                             f'{strong[0]},{strong[0]}', '--results', str(tmp_path / 'x')]) == pool.SHARD_MISMATCH


def test_an_interim_look_writes_a_gate_look_row_not_a_gate_result(tmp_path):
    """A look that plays on (pool_eval --stage, decision continue) still writes a ledger row, tagged
    stage=gate-look and interim=true, so the board sees every look; only a deciding look is stage=gate."""
    from yomibench import gate_seq
    from yomibench import ledger as metrics_ledger

    base = ['--candidate', 'greedy', '--previous-best', 'random', '--quick', '--characters', 'Toy',
            '--pairs', '16', '--min-pairs', '1', '--seed', '5', '--engine', 'toy', '--max-turns', '60']
    files = []
    for k in range(4):
        f = tmp_path / f's{k}.jsonl'
        assert pool.main(base + ['--shard', f'{k}/4', '--shard-out', str(f), '--results', str(tmp_path / 'u'),
                                 '--no-ledger']) == 0
        files.append(str(f))
    st = gate_seq.stage('confirm', 'full', 12, 16, (0.25, 0.5, 0.75, 1.0), final=True)
    plan = tmp_path / 'stage.json'
    plan.write_text(json.dumps(st))
    led = tmp_path / 'ledger.jsonl'
    code = pool.main(base + ['--stage', str(plan), '--look', '0', '--partial', '--from-shards', files[0],
                             '--results', str(tmp_path / 'r0.jsonl'), '--ledger', str(led), '--tag', 'stage=gate'])
    assert code == pool.CONTINUE
    rows = metrics_ledger.read(led)
    assert len(rows) == 1 and rows[0]['tags']['stage'] == 'gate-look' and rows[0]['tags']['interim'] == 'true'
    assert metrics_ledger.validate_row(rows[0]) == []
    code = pool.main(base + ['--stage', str(plan), '--look', '3', '--partial', '--from-shards', ','.join(files),
                             '--results', str(tmp_path / 'r3.jsonl'), '--ledger', str(led), '--tag', 'stage=gate'])
    assert code in (0, pool.EXIT['reject'])
    rows = metrics_ledger.read(led)
    assert len(rows) == 2 and rows[1]['tags']['stage'] == 'gate' and 'interim' not in rows[1]['tags']


FOCUS = {'Cowboy:Cowboy': 60, 'Cowboy:Ninja': 60, 'Ninja:Cowboy': 60, 'Ninja:Ninja': 32}
FOCUS_FLAG = 'Cowboy:Cowboy=60,Cowboy:Ninja=60,Ninja:Cowboy=60,Ninja:Ninja=32'
CN = ['Cowboy', 'Ninja']


def _sched(pairs=32, by_pairing=None):
    s = {'seed': 9, 'pairs': pairs, 'characters': CN, 'pool': [['previous-best', 'p']]}
    return {**s, 'pairs_by_pairing': by_pairing} if by_pairing else s


def _old_shard_games(sched, k, n):
    """shard_games as it was before per-pairing counts: the uniform schedule must not change."""
    out = []
    for label, _ in sched['pool']:
        g = 0
        for c1 in sched['characters']:
            for c2 in sched['characters']:
                for _ in range(sched['pairs']):
                    if g % n == k:
                        seed = sched['seed'] * 1_000_003 + g
                        out += [(('candidate', label), (c1, c2), seed), ((label, 'candidate'), (c1, c2), seed)]
                    g += 1
    return out


def test_without_per_pairing_counts_the_schedule_and_its_shards_are_unchanged():
    from types import SimpleNamespace

    for pairs, n in ((50, 40), (32, 1), (3, 7)):
        for k in range(n):
            assert pool.shard_games(_sched(pairs), k, n) == _old_shard_games(_sched(pairs), k, n)
    args = SimpleNamespace(seed=1, pairs=50, candidate_set=4, max_turns=1000, quick=True, candidate='c',
                           pairs_by_pairing=None)
    assert 'pairs_by_pairing' not in pool.schedule(args, [('previous-best', 'p')], CN)


def test_per_pairing_counts_schedule_that_many_pairs_and_keep_the_first_32_of_each_pairing():
    focus, plain = _sched(212, FOCUS), _sched(32)
    games = pool.shard_games(focus, 0, 1)[::2]  # one row per pair (the candidate's seat 0 game)
    by = {p: [seed for _, chars, seed in games if ':'.join(chars) == p] for p in FOCUS}
    assert {p: len(s) for p, s in by.items()} == FOCUS
    assert len({seed for _, _, seed in games}) == 212  # every pair its own seed
    first = {p: [seed for _, chars, seed in pool.shard_games(plain, 0, 1)[::2] if ':'.join(chars) == p] for p in FOCUS}
    assert all(by[p][:32] == first[p] for p in FOCUS)  # the same games whether a pairing has 32 or 60


def test_microshards_cover_a_per_pairing_schedule_once_and_an_early_look_sees_every_pairing():
    import math
    from collections import Counter

    focus = _sched(212, FOCUS)
    n = math.ceil(212 / 5)  # round3_loop: 5 pairs per microshard
    shards = [pool.shard_games(focus, k, n) for k in range(n)]
    every = Counter(g for s in shards for g in s)
    assert every == Counter(pool.shard_games(focus, 0, 1)) and set(every.values()) == {1}
    early = Counter(':'.join(chars) for s in shards[:11] for _, chars, _ in s[::2])  # the first 11 claimed
    assert sum(early.values()) >= 53 and min(early.values()) >= 8  # 25% of 212, and look 1's minimum
    assert all(abs(early[p] - FOCUS[p] * sum(early.values()) / 212) <= 1 for p in FOCUS)  # in proportion


def _cn_pairs(chars, wins=0, losses=0, splits=0, seed0=0):
    kinds = [WIN] * wins + [LOSS] * losses + [SPLIT] * splits
    return [GameResult(seats, chars, seed0 + i, outcome, 1, (1, 1), 0.0)
            for i, k in enumerate(kinds)
            for seats, outcome in zip([('candidate', 'previous-best'), ('previous-best', 'candidate')], k)]


def _focus_look(look, cc, cn, nc, nn):
    from yomibench import gate_seq

    (st,) = gate_seq.full_core(42, pairs_by_pairing=FOCUS, characters=CN)
    results = (_cn_pairs(('Cowboy', 'Cowboy'), *cc) + _cn_pairs(('Cowboy', 'Ninja'), *cn, seed0=100)
               + _cn_pairs(('Ninja', 'Cowboy'), *nc, seed0=200) + _cn_pairs(('Ninja', 'Ninja'), *nn, seed0=300))
    summary = pool.gate_summary(results, FOCUS, None, pairings=[tuple(p.split(':')) for p in FOCUS],
                                anchors=pool.QUICK_ANCHORS)
    return summary, pool.stage_decision(st, look, summary, results)


def test_a_look_scores_the_mean_over_all_complete_pairs_with_that_means_se():
    """The focus-weighted mean: every complete pair counts once, so a pairing weighs by its pairs."""
    import math

    # (wins, losses, splits): CC 20 won + 10 split, CN 15/15, NC 30 split, NN 4 won 12 lost (106 pairs)
    summary, (decision, _, v) = _focus_look(1, (20, 0, 10), (15, 15, 0), (0, 0, 30), (4, 12, 0))
    n1, nh, n0 = 20 + 15 + 4, 10 + 30, 15 + 12  # pairs scoring 1, 0.5 and 0
    n = n1 + nh + n0
    mean = (n1 + 0.5 * nh) / n
    se = math.sqrt((n1 * (1 - mean) ** 2 + nh * (0.5 - mean) ** 2 + n0 * mean ** 2) / (n - 1) / n)
    assert v['core']['n'] == 106 and v['core']['mean'] == pytest.approx(mean) and v['core']['se'] == pytest.approx(se)
    assert mean != pytest.approx((25 / 30 + 15 / 30 + 15 / 30 + 4 / 16) / 4)  # not the mean of pairing means
    assert decision == 'continue'  # 0.557 - 2.862 x 0.038 < 0.5 < 0.557 + 2.862 x 0.038
    per = {':'.join(p['characters']): (p['checks']['previous-best']['complete_pairs'], p['previous_best_score'])
           for p in summary['pairings']}
    assert per == {'Cowboy:Cowboy': (30, pytest.approx(25 / 30, abs=1e-4)), 'Cowboy:Ninja': (30, 0.5),
                   'Ninja:Cowboy': (30, 0.5), 'Ninja:Ninja': (16, 0.25)}


def test_the_guard_holds_on_the_down_weighted_ninja_mirror_too():
    strong = (50, 10, 0)  # 0.83 in each Cowboy pairing
    summary, (decision, reason, _) = _focus_look(3, strong, strong, strong, (8, 24, 0))  # the mirror at 0.25
    nn = next(p for p in summary['pairings'] if p['characters'] == ['Ninja', 'Ninja'])
    assert summary['previous_best_score'] > 0.7 and nn['ci95'][1] < 0.5  # pooled far above, only the mirror low
    assert all(p['ci95'][1] >= 0.5 for p in summary['pairings'] if p is not nn)
    assert decision == 'reject' and 'guard' in reason and 'Ninja:Ninja' in reason
    _, (decision, reason, _) = _focus_look(1, (28, 2, 0), (28, 2, 0), (28, 2, 0), (2, 14, 0))
    assert decision == 'continue' and 'guard fails' in reason and 'Ninja:Ninja' in reason  # no early accept either
    _, (decision, _, _) = _focus_look(3, strong, strong, strong, (16, 16, 0))
    assert decision == 'promote'


def test_per_pairing_shards_played_here_decide_like_one_run_and_must_match_the_stage(tmp_path, capsys):
    """End to end on the toy engine: --pairs-by-pairing shards hold exactly their share of the
    per-pairing schedule (read_shard checks every game), merge to the one-run games, and a --stage
    look needs the stage's counts."""
    from yomibench import gate_seq

    counts = 'Cowboy:Cowboy=4,Cowboy:Ninja=4,Ninja:Cowboy=4,Ninja:Ninja=2'
    base = ['--candidate', 'greedy', '--previous-best', 'random', '--quick', '--characters', 'Cowboy,Ninja',
            '--pairs-by-pairing', counts, '--max-turns', '60', '--no-ledger', '--seat-preflight', '0']
    one = tmp_path / 'one.jsonl'
    pool.main(base + ['--results', str(one)])
    assert len(_games(one)) == 2 * 14
    files = []
    for k in range(3):
        f = tmp_path / f's{k}.jsonl'
        assert pool.main(base + ['--shard', f'{k}/3', '--shard-out', str(f), '--results', str(tmp_path / 'u')]) == 0
        files.append(str(f))
    merged = tmp_path / 'merged.jsonl'
    pool.main(base + ['--from-shards', ','.join(files), '--results', str(merged)])
    assert _games(merged) == _games(one)
    s = json.loads(merged.read_text().splitlines()[-1])
    assert s['pairs'] == 14 and s['pairs_by_pairing'] == gate_seq.parse_pairs_by_pairing(counts)
    st = gate_seq.stage('c', 'full', 12, None, (1.0,), min_pairs_per_pairing=2, final=True,
                        pairs_by_pairing=gate_seq.parse_pairs_by_pairing(counts), characters=CN)
    plan = tmp_path / 'stage.json'
    plan.write_text(json.dumps(st))
    res = tmp_path / 'look.jsonl'
    code = pool.main(base + ['--stage', str(plan), '--look', '0', '--partial', '--from-shards', ','.join(files),
                             '--results', str(res)])
    look = json.loads(res.read_text().splitlines()[-1])
    assert code in pool.EXIT.values() and look['stage']['scheduled_pairs'] == 14 and look['stage']['core']['n'] <= 14
    uniform = [a for a in base if a not in ('--pairs-by-pairing', counts)] + ['--pairs', '14']
    with pytest.raises(SystemExit):  # a per-pairing stage looked at on a uniform schedule
        pool.main(uniform + ['--stage', str(plan), '--look', '0', '--partial', '--from-shards', files[0],
                             '--results', str(res)])
    assert pool.main(uniform + ['--from-shards', ','.join(files), '--results', str(res)]) == pool.SHARD_MISMATCH


def test_pairs_by_pairing_flag_errors(tmp_path):
    base = ['--candidate', 'greedy', '--previous-best', 'random', '--quick', '--characters', 'Cowboy,Ninja',
            '--results', str(tmp_path / 'r.jsonl'), '--no-ledger', '--seat-preflight', '0']
    for extra in (['--pairs-by-pairing', FOCUS_FLAG, '--pairs', '50'],
                  ['--pairs-by-pairing', 'Cowboy:Cowboy=60,Cowboy:Ninja=60,Ninja:Cowboy=60'],
                  ['--pairs-by-pairing', FOCUS_FLAG + ',Toy:Toy=5'],
                  ['--pairs-by-pairing', 'Cowboy:Cowboy=60,Cowboy:Ninja=60,Ninja:Cowboy=60,Ninja:Ninja=0']):
        with pytest.raises(SystemExit):
            pool.main(base + extra)


def test_play_with_a_batch_engine_builds_it_from_the_plugin_per_opponent(monkeypatch, tmp_path):
    """A batch:N spec (the Arena's) plays through the batch plugin's engine factory: the plugin import must not
    clobber _play's own engine table (it once did, and every batch game crashed with a TypeError)."""
    from types import SimpleNamespace

    from yomibench import plugins

    built, served = [], []

    class Engine:
        def hello(self):
            return {'engine': 'fake-batch'}

    def batch_arena_engines(bots, candidate_set, match_settings):
        built.append([b.name for b in bots])
        return lambda spec: served.append(spec) or Engine()

    monkeypatch.setattr(plugins, 'batch', lambda: SimpleNamespace(batch_arena_engines=batch_arena_engines))
    monkeypatch.setattr(pool, 'make_bot', lambda spec: SimpleNamespace(name=spec))

    def fake_round_robin(connects, bots, characters, pairs, seed, results, **kw):
        for c in connects:
            c()
        return []
    monkeypatch.setattr(pool, 'run_round_robin', fake_round_robin)
    args = SimpleNamespace(candidate='cand', candidate_set=6, match_settings=None, pairs_by_pairing=None, cowboy=None,
                           pairs=1, seed=1, max_turns=30, record_dir=None)
    hellos = []
    pool._play(args, [('previous-best', 'p'), ('ckpt1', 'a')], ['batch:4'], ['Ninja'], None, tmp_path / 'r.jsonl',
               hellos)
    assert built == [['candidate', 'previous-best'], ['candidate', 'ckpt1']]
    assert served == ['batch:4', 'batch:4'] and hellos == [{'engine': 'fake-batch'}]

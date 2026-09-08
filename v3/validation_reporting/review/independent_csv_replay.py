"""Replay archived CSV arithmetic with Fraction/Decimal, without core/run imports.

This is the persisted form of the post-result review originally run through
stdin. It checks downstream selection/aggregation, not raw logits or inference.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from fractions import Fraction as F
import hashlib
import io
import json
from pathlib import Path
import platform


def replay(repo, archive, output):
    getcontext().prec = 60
    folder = archive / 'analysis'
    if output.exists():
        raise RuntimeError('Refuse to replace previous review')
    if output.is_relative_to(archive) or archive.is_relative_to(output):
        raise RuntimeError('Review output must be separate from original archive')
    names = ['metrics.csv', 'episodes.csv', 'contrasts.csv', 'controls.csv',
             'draws.csv', 'results.json', 'FILE_SHA256.json']
    blobs = {name: (folder / name).read_bytes() for name in names}
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = json.loads(blobs['FILE_SHA256.json'])
    for name, expected in manifest.items():
        if sha((folder / name).read_bytes()) != expected:
            raise AssertionError(('output hash mismatch', name))
    plan_bytes = (archive / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    canonical = json.dumps({k: v for k, v in plan.items() if k != 'plan_sha256'},
                           sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    assert sha(canonical) == plan['plan_sha256']
    result = json.loads(blobs['results.json'])
    assert result['plan_sha256'] == plan['plan_sha256']
    assert result['status'] == 'complete' and result['draw_count'] == 24
    source_hashes = {}
    for rel, expected in plan['sources'].items():
        actual = sha((repo / rel).read_bytes())
        assert actual == expected, ('source changed', rel)
        source_hashes[rel] = actual

    def read_rows(name):
        return list(csv.DictReader(io.StringIO(blobs[name].decode('utf-8'))))

    errors, counts = Counter(), Counter()

    def check(group, actual, expected):
        error = abs(float(actual) - float(expected))
        if error > 1e-9:
            raise AssertionError((group, actual, expected, error))
        errors[group] = max(errors[group], error)
        counts[group] += 1

    metrics = read_rows('metrics.csv')
    assert len(metrics) == 2400
    lookup = {}
    for row in metrics:
        d, f, c = [int(row[k]) for k in ('draw', 'fold', 'config_index')]
        role, half, n = row['role'], row['half'], int(row['n'])
        key = d, f, c, role, half
        assert key not in lookup
        if role != 'test':
            assert n == 144
            # Reconstruct the integer count from the rounded equal-support
            # percentage. This deliberately does not reread original logits.
            implied = Decimal(row['uar']) * Decimal(144) / Decimal(100)
            correct = int(implied.to_integral_value())
            assert abs(implied - Decimal(correct)) < Decimal('1e-10')
            assert 0 <= correct <= 144
            value = F(100 * correct, 144)
        else:
            assert n > 0
            # Unequal test class counts are possible: retain the stored UAR,
            # not accuracy, as an exact decimal rational.
            value = F(Decimal(row['uar']))
        assert 0 <= value <= 100
        lookup[key] = value
        check('metric_decimal_roundtrip', row['uar'], value)
    expected_keys = {
        (d, f, c, r, h) for d in range(24) for f in range(5) for c in range(4)
        for r, hs in [('seen', ('A', 'B')), ('unseen', ('A', 'B')), ('test', ('all',))]
        for h in hs
    }
    assert set(lookup) == expected_keys

    eps, contrasts, controls = [], [], []
    for d in range(24):
        for f in range(5):
            for direction, selection, report in [('A_to_B', 'A', 'B'), ('B_to_A', 'B', 'A')]:
                chosen = {}
                for rule in ('seen', 'unseen'):
                    config = max(range(4), key=lambda c: (lookup[d, f, c, rule, selection], -c))
                    v = lookup[d, f, config, rule, selection]
                    hs = lookup[d, f, config, 'seen', report]
                    hu = lookup[d, f, config, 'unseen', report]
                    own = hs if rule == 'seen' else hu
                    test = lookup[d, f, config, 'test', 'all']
                    row = dict(draw=d, fold=f, direction=direction, rule=rule, config_index=config,
                               selection_uar=v, report_seen_uar=hs, report_unseen_uar=hu, test_uar=test,
                               reuse_pp=v-own, report_gap_pp=own-test, apparent_gap_pp=v-test,
                               report_exposure_pp=hs-hu)
                    assert row['apparent_gap_pp'] == row['reuse_pp'] + row['report_gap_pp']
                    eps.append(row)
                    chosen[rule] = row
                s, u = chosen['seen'], chosen['unseen']
                row = dict(draw=d, fold=f, direction=direction,
                           apparent_contrast_pp=s['apparent_gap_pp']-u['apparent_gap_pp'],
                           reuse_contrast_pp=s['reuse_pp']-u['reuse_pp'],
                           report_gap_contrast_pp=s['report_gap_pp']-u['report_gap_pp'],
                           test_contrast_pp=s['test_uar']-u['test_uar'],
                           config_agreement=int(s['config_index'] == u['config_index']))
                assert row['apparent_contrast_pp'] == row['reuse_contrast_pp'] + row['report_gap_contrast_pp']
                contrasts.append(row)
                controls.append(dict(draw=d, fold=f, direction=direction,
                    fixed_seen_reuse_pp=lookup[d, f, 3, 'seen', selection]-lookup[d, f, 3, 'seen', report],
                    fixed_unseen_reuse_pp=lookup[d, f, 3, 'unseen', selection]-lookup[d, f, 3, 'unseen', report],
                    fixed_report_exposure_pp=lookup[d, f, 3, 'seen', report]-lookup[d, f, 3, 'unseen', report],
                    fixed_test_difference_pp=F(0)))
            for rule in ('seen', 'unseen'):
                assert sum(x['reuse_pp'] for x in eps if x['draw'] == d and x['fold'] == f and x['rule'] == rule) >= 0
                assert sum(x['fixed_'+rule+'_reuse_pp'] for x in controls if x['draw'] == d and x['fold'] == f) == 0

    def check_rows(name, expected, identifiers):
        actual = read_rows(name)
        assert len(actual) == len(expected)
        index = {}
        for row in actual:
            key = tuple(row[k] for k in identifiers)
            assert key not in index
            index[key] = row
        for row in expected:
            key = tuple(str(row[k]) for k in identifiers)
            saved = index.pop(key)
            assert set(saved) == set(row)
            for field in set(row)-set(identifiers):
                check(name, saved[field], row[field])
        assert not index

    check_rows('episodes.csv', eps, ['draw', 'fold', 'direction', 'rule'])
    check_rows('contrasts.csv', contrasts, ['draw', 'fold', 'direction'])
    check_rows('controls.csv', controls, ['draw', 'fold', 'direction'])

    def mean(values):
        return sum(values, F(0)) / len(values)

    draws = []
    for d in range(24):
        row = {'draw': d}
        for rule in ('seen', 'unseen'):
            keys = set(eps[0])-{'draw', 'fold', 'direction', 'rule', 'config_index'}
            for key in keys:
                row[rule+'__'+key] = mean([mean([r[key] for r in eps if r['draw'] == d and r['fold'] == f and r['rule'] == rule]) for f in range(5)])
        for group in (contrasts, controls):
            keys = set(group[0])-{'draw', 'fold', 'direction'}
            for key in keys:
                row[key] = mean([mean([r[key] for r in group if r['draw'] == d and r['fold'] == f]) for f in range(5)])
        draws.append(row)
    check_rows('draws.csv', draws, ['draw'])
    means = {k: mean([r[k] for r in draws]) for k in draws[0] if k != 'draw'}
    assert set(means) == set(result['means'])
    for k, v in means.items():
        check('result_means', result['means'][k], v)
    config_counts = {rule: {str(k): v for k, v in sorted(Counter(r['config_index'] for r in eps if r['rule'] == rule).items())} for rule in ('seen', 'unseen')}
    assert config_counts == result['config_counts']
    assert means['fixed_seen_reuse_pp'] == means['fixed_unseen_reuse_pp'] == means['fixed_test_difference_pp'] == 0
    assert means['apparent_contrast_pp'] == means['reuse_contrast_pp'] + means['report_gap_contrast_pp']
    check('historical_fixed_exposure', means['fixed_report_exposure_pp'], '2.8385416666666674')
    for name, content in blobs.items():
        assert (folder/name).read_bytes() == content
    assert (archive/'plan.json').read_bytes() == plan_bytes
    for rel, expected in source_hashes.items():
        assert sha((repo/rel).read_bytes()) == expected
    review = {
        'review_type': 'post-result independent CSV arithmetic replay',
        'pass': True, 'finished_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'input_directory': str(folder),
        'input_sha256': {**{name: sha(b) for name, b in blobs.items()}, '../plan.json': sha(plan_bytes)},
        'manifest_files_verified': len(manifest), 'analysis_plan_sha256': plan['plan_sha256'],
        'frozen_sources_unchanged': source_hashes,
        'method': 'Independent standard-library Fraction/Decimal arithmetic; no import of core.py/run.py. Reconstruct 1920 equal-support half-score correct counts from CSV with strict integer tolerance; preserve 480 test UAR decimal strings, choose argmax with lowest-index ties, rebuild all episodes/contrasts/controls, then directions->5 folds->24 draws.',
        'scope_limits': ['Checks downstream CSV selection and aggregation, not raw-logit UAR or model inference.',
                         'Uses the same fixed post-result split; not an independent scientific replication.',
                         'No new significance tests, intervals, model fits, or endpoint selection.'],
        'rows': {'metrics': len(metrics), 'episodes': len(eps), 'contrasts': len(contrasts), 'controls': len(controls), 'draws': len(draws)},
        'comparisons': dict(counts), 'max_absolute_pp_error': dict(errors),
        'exact_integer_and_fraction_identities_pass': True,
        'means_independently_replayed': {k: float(v) for k, v in sorted(means.items())},
        'exact_fraction_means': {k: f'{v.numerator}/{v.denominator}' for k, v in sorted(means.items())},
        'config_counts': config_counts,
        'same_configuration_episodes': sum(r['config_agreement'] for r in contrasts),
        'new_statistical_tests': 0, 'new_confidence_intervals': 0,
        'replay_source_sha256': sha(Path(__file__).read_bytes()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(review, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({'pass': True, 'output': str(output), 'max_error_pp': max(errors.values()),
                      'comparisons': sum(counts.values()), 'same_config': review['same_configuration_episodes']}))


def main():
    if not __debug__:
        raise RuntimeError('Do not disable review assertions with python -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New JSON path outside the input archive')
    args = parser.parse_args()
    replay(args.repo.resolve(), args.archive.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()

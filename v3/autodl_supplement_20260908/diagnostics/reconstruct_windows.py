"""Post-hoc, descriptive window reconstruction from already accepted CSVs.

No torch, GPU, new logits, confidence intervals, tests, or p values. The four
requested windows are all retained. W15 must reproduce the archived selections,
contexts, draws, absolute means, and primary means before any output is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

CORPORA = ('cremad', 'subesco', 'ravdess')
WINDOWS = (8, 10, 12, 15)
RULES = ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar')
EFFECTS = ('D_CE_pp', 'D_UAR_pp', 'J_pp')
BALANCED_QUERY_N = {'cremad': 288, 'subesco': 112, 'ravdess': 64}
CURVE_FIELDS = ('unit_id', 'corpus', 'draw', 'fold', 'epoch',
                'seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar', 'outer_uar')
PIN_CURVES = 'f210fa9208d087c72aeb4156d6ea8523b7601a4cd6c4d16b6355462bab6541c4'
PIN_RESULTS = '13b07f8c825a8d43f1e62db1aa400b0a5c939c6ac193b2780960446e8dc55088'
TOLERANCE_PP = 1e-11


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mean(values):
    values = list(values)
    require(values and all(math.isfinite(float(v)) for v in values), 'nonfinite/empty mean')
    return math.fsum(values) / len(values)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validation_count(uar, corpus):
    """Recover correct counts only for the original complete, balanced queries.

    Whole-role native UAR equals total accuracy here because all classes have
    equal support. This integer lattice preserves exact UAR ties, not tolerance-
    based tie merging. Outer UAR is NOT rounded to this lattice.
    """
    n = BALANCED_QUERY_N[corpus]
    count = round(uar * n / 100)
    require(0 <= count <= n and abs(uar - 100 * count / n) <= 1e-11,
            'validation UAR is off original balanced-query lattice')
    return count


def parse_curves(raw):
    require(raw and set(raw[0]) == set(CURVE_FIELDS), 'unexpected curves CSV schema')
    groups = defaultdict(dict)
    for entry in raw:
        row = dict(entry)
        for key in ('draw', 'fold', 'epoch'):
            row[key] = int(row[key])
        c, d, f, epoch = (row[k] for k in ('corpus', 'draw', 'fold', 'epoch'))
        require(c in CORPORA and d in range(24) and f in range(5) and epoch in range(1, 16),
                'unexpected corpus/context/epoch')
        uid = f'fpc_formal_{c}_d{d:02d}_f{f}_A'
        require(row['unit_id'] == uid, 'non-main/incorrect UID')
        require(epoch not in groups[uid], 'duplicate curve epoch')
        for key in ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar', 'outer_uar'):
            row[key] = float(row[key])
            require(math.isfinite(row[key]) and row[key] >= 0, 'invalid curve metric')
            if key.endswith('uar'):
                require(row[key] <= 100, 'UAR outside percent range')
        for rule in ('seen_uar', 'unseen_uar'):
            row[rule + '_count'] = validation_count(row[rule], c)
        groups[uid][epoch] = row
    wanted = {f'fpc_formal_{c}_d{d:02d}_f{f}_A'
              for c in CORPORA for d in range(24) for f in range(5)}
    require(set(groups) == wanted, 'not the complete 360-main grid')
    require(all(set(rows) == set(range(1, 16)) for rows in groups.values()),
            'incomplete 15-epoch trajectory')
    return {uid: [rows[e] for e in range(1, 16)] for uid, rows in groups.items()}


def select_window(rows, window):
    require(window in WINDOWS, 'window outside fixed diagnostic set')
    prefix = [r for r in rows if 1 <= r['epoch'] <= window]
    require({r['epoch'] for r in prefix} == set(range(1, window + 1))
            and len(prefix) == window, 'incomplete window')
    prefix = sorted(prefix, key=lambda r: r['epoch'])
    first = prefix[0]
    value = {k: first[k] for k in ('unit_id', 'corpus', 'draw', 'fold')}
    value['window'] = window
    for rule in RULES:
        if rule.endswith('ce'):
            winner = min(prefix, key=lambda r: (r[rule], r['epoch']))
        else:
            winner = min(prefix, key=lambda r: (-r[rule + '_count'], r['epoch']))
        value[rule + '_epoch'] = winner['epoch']
        value['T_' + rule + '_uar'] = winner['outer_uar']
    value['D_CE_pp'] = value['T_seen_ce_uar'] - value['T_unseen_ce_uar']
    value['D_UAR_pp'] = value['T_seen_uar_uar'] - value['T_unseen_uar_uar']
    value['J_pp'] = value['D_CE_pp'] - value['D_UAR_pp']
    value['ce_disagreement'] = int(value['seen_ce_epoch'] != value['unseen_ce_epoch'])
    value['uar_disagreement'] = int(value['seen_uar_epoch'] != value['unseen_uar_epoch'])
    value['last_window_outer_uar'] = prefix[-1]['outer_uar']
    return value


def describe(values):
    values = list(values)
    return dict(n=len(values), mean=mean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), median=statistics.median(values), maximum=max(values),
                positive_count=sum(v > 0 for v in values),
                negative_count=sum(v < 0 for v in values), zero_count=sum(v == 0 for v in values))


def reconstruct(groups):
    contexts = [select_window(groups[uid], w)
                for c in CORPORA for w in WINDOWS for d in range(24) for f in range(5)
                for uid in [f'fpc_formal_{c}_d{d:02d}_f{f}_A']]
    averages = (*EFFECTS, *(f'T_{r}_uar' for r in RULES),
                'last_window_outer_uar', 'ce_disagreement', 'uar_disagreement')
    draws, summaries = [], []
    for c in CORPORA:
        for w in WINDOWS:
            part = [x for x in contexts if (x['corpus'], x['window']) == (c, w)]
            for d in range(24):
                folds = [x for x in part if x['draw'] == d]
                require(len(folds) == 5 and {x['fold'] for x in folds} == set(range(5)),
                        'draw does not contain five folds')
                draws.append(dict(corpus=c, window=w, draw=d,
                                  **{k: mean(x[k] for x in folds) for k in averages}))
            dd = [x for x in draws if (x['corpus'], x['window']) == (c, w)]
            summary = dict(corpus=c, window=w, contexts=120, draws=24, folds_per_draw=5,
                           **{k: mean(x[k] for x in dd) for k in averages},
                           effect_draw_descriptions={k: describe(x[k] for x in dd) for k in EFFECTS},
                           selected_epoch_counts={r: dict(sorted(Counter(x[r + '_epoch'] for x in part).items()))
                                                  for r in RULES},
                           selected_at_boundary_fraction={r: mean(x[r + '_epoch'] == w for x in part)
                                                          for r in RULES})
            for criterion in ('ce', 'uar'):
                field = 'D_CE_pp' if criterion == 'ce' else 'D_UAR_pp'
                dis = [x[field] for x in part if x[criterion + '_disagreement']]
                same = [x[field] for x in part if not x[criterion + '_disagreement']]
                require(all(v == 0 for v in same), 'same epoch has nonzero effect')
                q = len(dis) / len(part)
                e = mean(dis) if dis else None
                product = q * e if e is not None else 0.0
                summary[criterion + '_disagreement_decomposition'] = dict(
                    q=q, n_disagree=len(dis), conditional_mean_effect_pp=e,
                    q_times_e_pp=product, observed_effect_pp=summary[field],
                    identity_abs_error_pp=abs(product - summary[field]),
                    same_epoch_effect_max_abs_pp=max(map(abs, same), default=0.0))
            summaries.append(summary)
    return contexts, draws, summaries


def verify_w15(contexts, draws, summaries, scores, results):
    errors = defaultdict(list)
    mapped = {'D_CE_pp': 'delta_CE_pp', 'D_UAR_pp': 'delta_UAR_pp', 'J_pp': 'J_pp'}
    cc = {x['unit_id']: x for x in read_csv(scores / 'contexts.csv')}
    dd = {(x['corpus'], int(x['draw'])): x for x in read_csv(scores / 'draws.csv')}
    ss = {(x['unit_id'], x['rule']): x for x in read_csv(scores / 'selected_epochs.csv')}
    require(len(cc) == 360 and len(dd) == 72 and len(ss) == 1800, 'reference grids differ')
    selected_checked = 0
    for x in (r for r in contexts if r['window'] == 15):
        for ours, old in mapped.items():
            errors['context_effects'].append(abs(x[ours] - float(cc[x['unit_id']][old])))
        for rule in RULES:
            reference = ss[x['unit_id'], rule]
            require(x[rule + '_epoch'] == int(reference['selected_epoch']), 'W15 selected epoch mismatch')
            errors['selected_outer'].append(abs(x['T_' + rule + '_uar'] - float(reference['outer_uar'])))
            selected_checked += 1
    for x in (r for r in draws if r['window'] == 15):
        for ours, old in mapped.items():
            errors['draw_effects'].append(abs(x[ours] - float(dd[x['corpus'], x['draw']][old])))
    for x in (r for r in summaries if r['window'] == 15):
        c = x['corpus']
        for ours, old in (('D_CE_pp', 'delta_CE'), ('J_pp', 'J')):
            target = next(t for t in results['tests'] if (t['corpus'], t['estimand']) == (c, old))
            errors['main_table_means'].append(abs(x[ours] - target['mean_pp']))
        errors['main_table_means'].append(abs(x['D_UAR_pp'] - results['corpora'][c]['delta_UAR_descriptive_mean_pp']))
        for rule in RULES:
            errors['absolute_means'].append(abs(x['T_' + rule + '_uar'] - results['corpora'][c]['rules'][rule]['mean_outer_uar']))
    maxima = {k: max(v, default=0.0) for k, v in errors.items()}
    require(max(maxima.values()) <= TOLERANCE_PP, 'W15 numerical reconstruction mismatch')
    return dict(pass_=True, selected_epochs_checked=selected_checked,
                checked_values={k: len(v) for k, v in errors.items()},
                max_abs_error_pp=maxima, tolerance_pp=TOLERANCE_PP,
                note='CSV outer UAR is already rounded float64; original contrasts used rational subtraction. '
                     'Agreement is numerical within tolerance, not bitwise identity of rational arithmetic.')


def checked_inputs(reports):
    manifest = document(reports / 'ARTIFACT_MANIFEST.json')
    names = ('scores/curves.csv', 'scores/contexts.csv', 'scores/draws.csv',
             'scores/selected_epochs.csv', 'scores/results.json', 'scores/complete_gate.json',
             'audits/numeric_original.json')
    pins = {}
    for name in names:
        path = reports / name
        pin = manifest['files'][name]
        require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                'archival manifest mismatch: ' + name)
        pins[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    require(pins['scores/curves.csv']['sha256'] == PIN_CURVES, 'unexpected original curves')
    require(pins['scores/results.json']['sha256'] == PIN_RESULTS, 'unexpected original result')
    result, gate, audit = (document(reports / n) for n in
                           ('scores/results.json', 'scores/complete_gate.json', 'audits/numeric_original.json'))
    require(result['counts']['formal_fits'] == 384 and result['counts']['main_A'] == 360
            and result['counts']['control_B'] == 24 and result['counts']['pilot_scored'] == 0,
            'unexpected original score count')
    require(gate['pass'] is True and gate['phase'] == 'formal' and gate['verified_units'] == 384,
            'original complete gate missing')
    require(audit['pass'] is True and audit['formal_units'] == 384
            and audit['main_result_sha256'] == PIN_RESULTS, 'original numeric audit identity mismatch')
    require(manifest['plan_sha256'] == result['plan_sha256'] == gate['plan_sha256'] == audit['plan_sha256'],
            'source plan identity mismatch')
    pins['ARTIFACT_MANIFEST.json'] = dict(sha256=sha(reports / 'ARTIFACT_MANIFEST.json'),
                                        bytes=(reports / 'ARTIFACT_MANIFEST.json').stat().st_size)
    return result, pins


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    reports, out = args.reports.resolve(), args.out.resolve()
    require(not out.exists(), 'output directory already exists; refusing overwrite')
    script_hash = sha(__file__)
    result, pins = checked_inputs(reports)
    groups = parse_curves(read_csv(reports / 'scores/curves.csv'))
    contexts, draws, summaries = reconstruct(groups)
    comparison = verify_w15(contexts, draws, summaries, reports / 'scores', result)
    provenance = dict(created_at=datetime.now(timezone.utc).isoformat(),
                      script_path=str(Path(__file__).resolve()), script_sha256=script_hash,
                      python=sys.version, reports_path=str(reports), input_files=pins,
                      original_plan_sha256=result['plan_sha256'], original_scored_formal_units=384,
                      current_trajectory_count=360, control_curves_used=0, pilot_used=0,
                      original_weights_rehashed=False, raw_logits_recomputed=False)
    payload = dict(schema='ser-old-window-descriptive-1', provenance=provenance,
                   scope='Post-hoc old-data description. No new fits, no p values/CI, no prospective claim.',
                   windows=list(WINDOWS), selection='validation only; CE min and exact balanced UAR max; earliest ties',
                   aggregation='five folds equally averaged per draw, then 24 draws equally averaged; no corpus pooling',
                   validation_query_count=BALANCED_QUERY_N, w15_verification=comparison,
                   contexts=contexts, draws=draws, summaries=summaries)
    for name, pin in pins.items():
        require(sha(reports / name) == pin['sha256'], 'input changed during computation')
    require(sha(__file__) == script_hash, 'script changed during computation')
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'window_contexts.csv', contexts)
    write_csv(out / 'window_draws.csv', draws)
    simple = []
    for s in summaries:
        flat = {k: v for k, v in s.items() if not isinstance(v, dict)}
        for criterion in ('ce', 'uar'):
            dec = s[criterion + '_disagreement_decomposition']
            flat[criterion + '_conditional_disagreement_effect_pp'] = dec['conditional_mean_effect_pp']
            flat[criterion + '_disagreement_count'] = dec['n_disagree']
        for rule in RULES:
            flat[rule + '_boundary_fraction'] = s['selected_at_boundary_fraction'][rule]
        simple.append(flat)
    write_csv(out / 'window_summary.csv', simple)
    with (out / 'window_diagnostics.json').open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    closure = dict(pass_=True, script_sha256=script_hash,
                   files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())})
    with (out / 'OUTPUT_MANIFEST.json').open('x', encoding='utf-8') as stream:
        json.dump(closure, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=str(out), counts=dict(contexts=len(contexts), draws=len(draws), summaries=len(summaries)),
                          w15=comparison), ensure_ascii=False))


if __name__ == '__main__':
    main()

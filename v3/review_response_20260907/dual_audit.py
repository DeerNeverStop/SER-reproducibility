"""Post-review exploratory audit of archived DUAL metadata and predictions.

No GPU, checkpoint weights, new fits, endpoint replacement, or publishing.
The fixed-effects intervals below are newly specified audit calculations,
not a replication of an undocumented reviewer standard-error procedure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from scipy.stats import t as student_t


PINS = {'inputs/plan.json': 'a765bad72a0c6be88040ca0dadac6848d139988714af26e738a06403ad0cb1e1',
        'analysis/formal_gate.json': '5cafbc5d32080f09b2bf9cb4488c5fb6b9455395b6a9ad887bd195a57bf41b68',
        'analysis/scores/results.json': '95b0cf199c9c77361b2956b12eb275e0d45c5f6227867324086dd995bc3028a9'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def summary(values, inference=True):
    x = np.asarray(values, dtype=float)
    assert x.shape == (24,) and np.isfinite(x).all()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    out = dict(mean_pp=mean, draw_sd_pp=sd, positive_draws=int((x > 0).sum()), draws=24,
               values_pp=x.tolist(), analysis_status='post-review exploratory')
    if inference:
        se = sd / np.sqrt(24)
        assert se > 0
        half = float(student_t.ppf(.975, 23)) * se
        out.update(standard_error_pp=se, t_ci95_pp=[mean-half, mean+half],
                   t_statistic=mean/se, p_two_sided=float(2*student_t.sf(abs(mean/se), 23)),
                   method='one-sample t, df=23, unadjusted, conditional fixed-corpus draw distribution; not a new confirmatory test')
    return out


def fixed_effects(person_rows):
    people = sorted({r['speaker'] for r in person_rows})
    contexts = sorted({(r['draw'], r['fold']) for r in person_rows})
    y = np.asarray([r['uar_percent'] for r in person_rows])
    exposure = np.asarray([r['seen'] for r in person_rows], dtype=float)
    pi = np.asarray([people.index(r['speaker']) for r in person_rows])
    ci = np.asarray([contexts.index((r['draw'], r['fold'])) for r in person_rows])
    di = np.asarray([r['draw'] for r in person_rows])
    fi = np.asarray([r['fold'] for r in person_rows])
    p_dummy = np.eye(len(people))[pi, 1:]
    designs = {
        'speaker_plus_joint_draw_fold_context': np.column_stack((np.ones(len(y)), exposure, p_dummy, np.eye(120)[ci, 1:])),
        'speaker_plus_additive_draw_and_fold': np.column_stack((np.ones(len(y)), exposure, p_dummy, np.eye(24)[di, 1:], np.eye(5)[fi, 1:])),
        'speaker_only': np.column_stack((np.ones(len(y)), exposure, p_dummy)),
    }
    outputs = {}
    for name, X in designs.items():
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        assert rank == X.shape[1]
        out = dict(coefficient_pp=float(beta[1]), rows=len(y), columns=X.shape[1],
                   model='OLS of per-speaker per-context six-class UAR on seen indicator plus named fixed effects; equal row weights')
        if name == 'speaker_plus_joint_draw_fold_context':
            # Explicit CR1 sandwich across 24 complete randomization draws.
            residual = y-X@beta
            bread = np.linalg.inv(X.T@X)
            meat = np.zeros((X.shape[1], X.shape[1]))
            for draw in range(24):
                score = X[di == draw].T @ residual[di == draw]
                meat += np.outer(score, score)
            correction = 24/23 * (len(y)-1)/(len(y)-rank)
            covariance = correction * bread @ meat @ bread
            se = float(np.sqrt(covariance[1, 1]))
            half = float(student_t.ppf(.975, 23)) * se
            out.update(cluster='complete draw, G=24', standard_error_pp=se,
                       cr1_correction=correction, reference_df=23,
                       ci95_pp=[float(beta[1]-half), float(beta[1]+half)],
                       interval_identity='new post-review exploratory CR1 cluster-t interval; Claude CI method not supplied, so not its verified replication')
        outputs[name] = out
    return outputs


def run(repo, archive, out):
    if out.exists() or out.is_relative_to(archive):
        raise RuntimeError('Output must be new and outside the archived experiment')
    input_hashes = {}
    for rel, expected in PINS.items():
        assert sha(archive/rel) == expected, rel
        input_hashes[str(archive/rel)] = expected
    plan, gate, results = [read(archive/rel) for rel in PINS]
    assert plan['plan_sha256'] == digest({k:v for k,v in plan.items() if k != 'plan_sha256'})
    assert gate['gate_sha256'] == digest({k:v for k,v in gate.items() if k != 'gate_sha256'})
    assert gate['pass'] is True and gate['phase'] == 'formal' and gate['units'] == 480
    assert gate['plan_sha256'] == results['plan_sha256'] == plan['plan_sha256']
    assert set(gate['done_sha256']) == {u['unit_id'] for u in plan['units']}
    for rel, expected in plan['sources'].items():
        assert sha(repo/rel) == expected, rel
        input_hashes[str(repo/rel)] = expected
    for key in ('manifest', 'demographics'):
        path = repo/plan['input'][key+'_path']
        assert sha(path) == plan['input'][key+'_sha256']
        input_hashes[str(path)] = sha(path)
    metadata = {r['relative_path']:r for r in csv_rows(repo/plan['input']['manifest_path'])}
    for rel, expected in results['table_sha256'].items():
        path = archive/'analysis/scores'/rel
        assert sha(path) == expected
        input_hashes[str(path)] = expected
    draws = csv_rows(archive/'analysis/scores/draws.csv')
    assert len(draws) == 24 and {int(r['draw']) for r in draws} == set(range(24))
    draws.sort(key=lambda r:int(r['draw']))
    metrics = csv_rows(archive/'analysis/scores/pred_metrics.csv')
    assert len(metrics) == 4320
    mi = {(int(r['draw']), int(r['fold']), int(r['config_index']), r['role'], r['checkpoint']):r for r in metrics}
    assert len(mi) == 4320

    # Metadata exposure counts refer to validation roles, not raw recording counts.
    units = sorted([u for u in plan['units'] if u['config_index'] == 3], key=lambda u:(u['draw'],u['fold']))
    assert len(units) == 120
    role_counts = defaultdict(Counter)
    person_rows = []
    for unit in units:
        uid = unit['unit_id']
        folder = archive/'runs/formal/units'/uid
        done_path = folder/'DONE'
        assert sha(done_path) == gate['done_sha256'][uid]
        input_hashes[str(done_path)] = sha(done_path)
        done = read(done_path)
        assert done['unit_sha256'] == digest(unit) and done['unit_id'] == uid
        assert done['phase'] == 'formal' and done['plan_sha256'] == plan['plan_sha256']
        files = {}
        for name in ('predictions.npz', 'receipt.json'):
            rels = [p for p in done['artifacts'] if Path(p).name == name]
            assert len(rels) == 1
            path = (folder/rels[0]).resolve()
            assert path.is_relative_to(folder.resolve()) and sha(path) == done['artifacts'][rels[0]]
            input_hashes[str(path)] = sha(path)
            files[name] = path
        receipt = read(files['receipt.json'])
        assert receipt['epochs_run'] == 15 and receipt['unit_sha256'] == digest(unit)
        with np.load(files['predictions.npz'], allow_pickle=False) as arrays:
            for role, people, seen in [('val_seen', unit['fit_speakers'], 1), ('val_unseen', unit['unseen_speakers'], 0)]:
                paths = arrays[role+'__paths'].tolist()
                labels = arrays[role+'__labels']
                logits = arrays[role+'__last__logits']
                assert paths == unit[role] and logits.shape == (288, 6) and np.isfinite(logits).all()
                assert labels.tolist() == [int(metadata[p]['label_index']) for p in paths]
                speakers = np.asarray([metadata[p]['speaker'] for p in paths])
                assert set(speakers) == set(people)
                predictions = logits.argmax(axis=1)
                per_uar = []
                for person in sorted(people):
                    mask = speakers == person
                    y, correct = labels[mask], predictions[mask] == labels[mask]
                    assert len(y) == 12 and np.array_equal(np.bincount(y,minlength=6), np.full(6,2))
                    value = 100.0*int(correct.sum())/12
                    role_counts[person]['seen' if seen else 'unseen'] += 1
                    person_rows.append(dict(draw=unit['draw'], fold=unit['fold'], unit_id=uid,
                                            speaker=person, seen=seen, records=12, correct=int(correct.sum()), uar_percent=value))
                    per_uar.append(value)
                reference = float(mi[unit['draw'],unit['fold'],3,role,'last']['uar_percent'])
                assert abs(np.mean(per_uar)-reference) < 1e-9
    assert len(person_rows) == 5760 and len(role_counts) == 91
    shares = [v['seen']/sum(v.values()) for v in role_counts.values()]
    overlaps = []
    for fold in range(5):
        fits = [set(u['fit_speakers']) for u in units if u['fold'] == fold]
        for i in range(24):
            for j in range(i+1,24):
                overlaps.append(len(fits[i]&fits[j]))
    unadjusted_pooled = float(np.mean([r['uar_percent'] for r in person_rows if r['seen']])-
                              np.mean([r['uar_percent'] for r in person_rows if not r['seen']]))
    person_differences = []
    for person in sorted(role_counts):
        values = [r for r in person_rows if r['speaker'] == person]
        person_differences.append(np.mean([r['uar_percent'] for r in values if r['seen']])-
                                  np.mean([r['uar_percent'] for r in values if not r['seen']]))
    fe = fixed_effects(person_rows)

    wanted = ('primary_delta_gap_pp', 'fixed_config_last_delta_gap_pp', 'selection_increment_pp',
              'primary_delta_test_pp', 'fixed_config_best_delta_test_pp',
              'fixed_config_last_seen_gap_pp', 'fixed_config_last_unseen_gap_pp')
    contrasts = {k: summary([float(r[k]) for r in draws]) for k in wanted}
    epoch_only_by_config = {}
    for c in range(4):
        per_draw = [float(np.mean([float(mi[d,f,c,'test','best_seen']['uar_percent'])-
                                  float(mi[d,f,c,'test','best_unseen']['uar_percent']) for f in range(5)])) for d in range(24)]
        epoch_only_by_config[str(c)] = summary(per_draw)
    allcfg = np.mean([epoch_only_by_config[str(c)]['values_pp'] for c in range(4)],axis=0)
    epoch_summary = summary(allcfg)
    selection_epochs = {
        cp: [int(mi[d,f,c,'test',cp]['checkpoint_epoch']) for d in range(24) for f in range(5) for c in range(4)]
        for cp in ('best_seen','best_unseen')}
    epoch_agree = sum(a == b for a,b in zip(selection_epochs['best_seen'],selection_epochs['best_unseen']))
    # Demonstrate that available epoch bins are selected checkpoints across
    # different trajectories. No unrecorded per-epoch test curve is invented.
    test_rows = [r for r in metrics if r['role'] == 'test']
    unique_test = {}
    for r in test_rows:
        key = (r['unit_id'], int(r['checkpoint_epoch']))
        if key in unique_test:
            assert float(unique_test[key]['uar_percent']) == float(r['uar_percent'])
        unique_test[key] = r
    bins = {}
    for label, lo, hi in [('epochs_8_to_10',8,10),('epochs_14_to_15',14,15)]:
        bins[label] = {}
        for mode, source in [('three_checkpoint_rows_including_aliases',test_rows),('unique_unit_epoch',list(unique_test.values()))]:
            chosen = [r for r in source if lo <= int(r['checkpoint_epoch']) <= hi]
            bins[label][mode] = dict(rows=len(chosen), distinct_trajectories=len({r['unit_id'] for r in chosen}),
                                    mean_uar_percent=float(np.mean([float(r['uar_percent']) for r in chosen])))
    output = dict(status='complete', analysis_identity='post-Claude-review exploratory audit; original prespecified endpoint unchanged',
        generated_at=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)),
        archive=str(archive), plan_sha256=plan['plan_sha256'], input_sha256=input_hashes,
        no_gpu=True, no_checkpoint_weights_read=True, original_source_unchanged=True,
        interval_scope='All newly computed t/CR1 intervals and p-values are post-review exploratory, unadjusted and conditional; no equivalence or mechanism claim.',
        role_coverage=dict(people=91, all_people_both_roles=all(v['seen']>0 and v['unseen']>0 for v in role_counts.values()),
            observations=5760, seen_share_denominator='seen+unseen validation-role observations; excludes outer and unselected eligible contexts',
            seen_share_equal_person_mean=float(np.mean(shares)), seen_share_min=float(min(shares)), seen_share_max=float(max(shares)),
            pooled_seen_fraction=.5, same_fold_pairwise_fit_overlap_mean=float(np.mean(overlaps)), overlap_comparisons=len(overlaps),
            per_person={p:dict(v,seen_share=v['seen']/sum(v.values())) for p,v in sorted(role_counts.items())}),
        raw_effects=dict(pooled_person_context_gap_pp=unadjusted_pooled,
                        equal_person_difference_of_role_means_pp=float(np.mean(person_differences)),
                        weighting_note='These differ because per-person role frequencies differ; neither is silently substituted for the frozen whole-fold estimator.'),
        fixed_effect_models=fe,
        claude_fixed_effect_claim=dict(point_pp=2.967,ci95_pp=[2.395,3.581],
            missing_specification=['regression code/design matrix','joint-context versus additive draw/fold effects','row weights','cluster/resampling unit','CI formula or bootstrap seed/replications'],
            interval_reproducibility='Not established; own explicit audit specification supplied instead.'),
        draw_contrasts=contrasts, epoch_only_delta_test_all_four_configs=epoch_summary,
        epoch_only_delta_test_by_config=epoch_only_by_config,
        selected_epochs_all_four_configs={k:dict(mean=float(np.mean(v)),counts=dict(Counter(v))) for k,v in selection_epochs.items()},
        selected_epoch_agreement=dict(numerator=epoch_agree,denominator=480,fraction=epoch_agree/480),
        selected_checkpoint_epoch_bins=bins,
        epoch_bin_limit='Selected checkpoint cross-sections; only best_seen/best_unseen/last test predictions exist. No complete within-trajectory learning curve or causal peak epoch is identified.')
    for path, expected in input_hashes.items():
        assert sha(path) == expected, ('input changed during audit',path)
    out.mkdir(parents=True)
    with (out/'dual_person_rows.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(person_rows[0]));writer.writeheader();writer.writerows(person_rows)
    (out/'dual_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','role_coverage':{k:v for k,v in output['role_coverage'].items() if k != 'per_person'},
        'raw':output['raw_effects'],'fe':fe,'epoch_only_allcfg':epoch_summary['mean_pp'],'bins':bins},ensure_ascii=False))


def main():
    if not __debug__:
        raise RuntimeError('Do not disable audit assertions')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.repo.resolve(),args.archive.resolve(),args.out.resolve())


if __name__ == '__main__':
    main()

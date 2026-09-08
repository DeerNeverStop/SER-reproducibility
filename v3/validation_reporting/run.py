"""Prepare then run a post-result diagnostic from immutable last-epoch logits."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from v3.data_design.core_plan import digest, file_sha, read_metadata, require, write_json
from v3.inner_validation.plan import load_plan
from .core import analyze_context, uar

PROGRAM = 'SER26-VALIDATION-REPORT-EXPLORATORY-1'
PINNED = {
    'inputs/plan.json': 'a765bad72a0c6be88040ca0dadac6848d139988714af26e738a06403ad0cb1e1',
    'analysis/formal_gate.json': '5cafbc5d32080f09b2bf9cb4488c5fb6b9455395b6a9ad887bd195a57bf41b68',
    'analysis/scores/results.json': '95b0cf199c9c77361b2956b12eb275e0d45c5f6227867324086dd995bc3028a9',
}
FIXED_EXPOSURE = 2.8385416666666674


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources(repo):
    folder = repo / 'v3/validation_reporting'
    return {p.relative_to(repo).as_posix(): file_sha(p)
            for p in sorted([*folder.glob('*.py'), folder / 'SPEC.md'])}


def originals(repo, archive):
    for rel, sha in PINNED.items():
        require(file_sha(archive / rel) == sha, 'pinned original changed: ' + rel)
    old = load_plan(archive / 'inputs/plan.json', repo)
    gate = read_json(archive / 'analysis/formal_gate.json')
    require(gate['pass'] and gate['units'] == 480 and gate['phase'] == 'formal', 'old full gate failed')
    require(gate['plan_sha256'] == old['plan_sha256'], 'old gate/plan mismatch')
    require(gate['gate_sha256'] == digest({k: v for k, v in gate.items() if k != 'gate_sha256'}),
            'old gate semantic hash differs')
    pipeline = read_json(archive / 'analysis/pipeline.json')
    require(pipeline['pass_all_steps'] and pipeline['state'] == 'completed'
            and len(pipeline['steps']) == 6 and all(s['exit_code'] == 0 for s in pipeline['steps']),
            'old completed six-stage pipeline absent')
    require(file_sha(archive / 'runs/formal/ledger.jsonl') == gate['ledger_sha256'], 'old ledger differs')
    return old, gate


def contexts(old, meta):
    rows = {r['relative_path']: r for r in meta['clean']}
    result = []
    for unit in old['units']:
        if unit['config_index'] != 0:
            continue
        ctx = {'draw': unit['draw'], 'fold': unit['fold'], 'halves': {}}
        for role, field in [('seen', 'fit_speakers'), ('unseen', 'unseen_speakers')]:
            halves = {'A': [], 'B': []}
            for sex in ('Female', 'Male'):
                people = [s for s in unit[field] if meta['sex'][s] == sex]
                require(len(people) == 12, 'old sex count changed')
                ranked = sorted(people, key=lambda s: (digest([PROGRAM, unit['draw'], unit['fold'], role, sex, s]), s))
                halves['A'].extend(ranked[:6])
                halves['B'].extend(ranked[6:])
            ctx['halves'][role] = {}
            for half, people in halves.items():
                paths = [p for p in unit['val_' + role] if rows[p]['speaker'] in people]
                require(len(paths) == 144 and len(people) == 12, 'half budget changed')
                require(Counter(int(rows[p]['label_index']) for p in paths) == Counter({c: 24 for c in range(6)}),
                        'half class counts changed')
                ctx['halves'][role][half] = {'people': sorted(people), 'paths': paths}
            require(not set(halves['A']) & set(halves['B']), 'half people overlap')
        result.append(ctx)
    require(len(result) == 120, 'wrong context count')
    return result


def prepare(repo, archive, out):
    require(not out.exists(), 'prepare refuses to overwrite an existing directory')
    old, gate = originals(repo, archive)
    meta = read_metadata(repo)
    plan = {'program': PROGRAM, 'created_at': now(), 'analysis_status': 'post-result exploratory',
            'prior_results_already_seen': True, 'new_predictions_scored': False,
            'archive': str(archive), 'source_commit_parent': '1da2b6074e5409178606573435c186df5978d0e8',
            'sources': sources(repo), 'original_files': {**PINNED,
                'analysis/pipeline.json': file_sha(archive / 'analysis/pipeline.json'),
                'runs/formal/ledger.jsonl': gate['ledger_sha256']},
            'metadata': meta['input'], 'contexts': contexts(old, meta)}
    plan['plan_sha256'] = digest(plan)
    write_json(out / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'contexts': 120, 'plan_sha256': plan['plan_sha256'],
                      'predictions_scored': False}))


def check_plan(repo, archive, out):
    plan = read_json(out / 'plan.json')
    require(plan['program'] == PROGRAM and plan['prior_results_already_seen'] is True, 'wrong new plan')
    require(plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}), 'new plan hash differs')
    require(plan['sources'] == sources(repo), 'new execution source/spec changed after preparation')
    require(Path(plan['archive']).resolve() == archive, 'archive path changed')
    for rel, sha in plan['original_files'].items():
        require(file_sha(archive / rel) == sha, 'original changed: ' + rel)
    old, gate = originals(repo, archive)
    meta = read_metadata(repo)
    require(plan['metadata'] == meta['input'] and plan['contexts'] == contexts(old, meta), 'new split replay differs')
    return plan, old, gate, meta


def prediction_gate(old, gate, archive):
    """Hash every relevant artifact before allowing any diagnostic scores."""
    hashes, paths = {}, {}
    require(set(gate['done_sha256']) == {u['unit_id'] for u in old['units']}, 'old gate inventory differs')
    for unit in old['units']:
        uid = unit['unit_id']
        folder = archive / 'runs/formal/units' / uid
        done_path = folder / 'DONE'
        require(file_sha(done_path) == gate['done_sha256'][uid], 'DONE changed: ' + uid)
        done = read_json(done_path)
        require(done['unit_id'] == uid and done['phase'] == 'formal'
                and done['plan_sha256'] == old['plan_sha256'] and done['unit_sha256'] == digest(unit),
                'DONE identity differs: ' + uid)
        hashes[done_path.relative_to(archive).as_posix()] = file_sha(done_path)
        paths[uid] = {}
        for name in ('predictions.npz', 'receipt.json'):
            matches = [rel for rel in done['artifacts'] if Path(rel).name == name]
            require(len(matches) == 1, 'ambiguous artifact: ' + uid)
            rel = matches[0]
            path = (folder / rel).resolve()
            require(path.is_relative_to(folder.resolve()), 'artifact outside unit directory')
            actual = file_sha(path)
            require(actual == done['artifacts'][rel], 'artifact changed: ' + uid + '/' + name)
            hashes[path.relative_to(archive).as_posix()] = actual
            paths[uid][name] = path
        receipt = read_json(paths[uid]['receipt.json'])
        require(receipt['epochs_run'] == 15, 'trajectory is not fixed last15')
    return hashes, paths


def average_rows(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def aggregate(episodes, contrasts, controls):
    ep_keys = [k for k in episodes[0] if k not in ('draw', 'fold', 'direction', 'rule', 'config_index')]
    co_keys = [k for k in contrasts[0] if k not in ('draw', 'fold', 'direction')]
    ct_keys = [k for k in controls[0] if k not in ('draw', 'fold', 'direction')]
    draws = []
    for draw in range(24):
        row = {'draw': draw}
        for rule in ('seen', 'unseen'):
            folds = [average_rows([r for r in episodes if r['draw'] == draw and r['fold'] == fold and r['rule'] == rule], ep_keys)
                     for fold in range(5)]
            row.update({rule + '__' + k: v for k, v in average_rows(folds, ep_keys).items()})
        for collection, keys in ((contrasts, co_keys), (controls, ct_keys)):
            folds = [average_rows([r for r in collection if r['draw'] == draw and r['fold'] == fold], keys)
                     for fold in range(5)]
            row.update(average_rows(folds, keys))
        draws.append(row)
    return draws, average_rows(draws, [k for k in draws[0] if k != 'draw'])


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(repo, archive, out):
    require(not (out / 'analysis').exists(), 'analysis refuses to overwrite previous output')
    started, start_clock = now(), time.perf_counter()
    plan, old, gate, meta = check_plan(repo, archive, out)
    hashes, paths = prediction_gate(old, gate, archive)
    dest = out / 'analysis'
    write_json(dest / 'prediction_gate.json', {'pass': True, 'verified_at': now(),
        'units': 480, 'plan_sha256': plan['plan_sha256'], 'archive_file_sha256': hashes,
        'scores_computed': False, 'weights_rescanned': False, 'prior_weight_gate': PINNED['analysis/formal_gate.json']})
    indexed = {(u['draw'], u['fold'], u['config_index']): u for u in old['units']}
    rows = {r['relative_path']: r for r in meta['clean']}
    episodes, contrasts, controls, metric_rows = [], [], [], []
    maximum_error = 0.0
    for ctx in plan['contexts']:
        draw, fold = ctx['draw'], ctx['fold']
        scores = {}
        for cfg in range(4):
            unit = indexed[draw, fold, cfg]
            uid = unit['unit_id']
            scores[cfg] = {}
            with np.load(paths[uid]['predictions.npz'], allow_pickle=False) as arrays:
                for role in ('seen', 'unseen', 'test'):
                    original_role = 'test' if role == 'test' else 'val_' + role
                    all_paths = arrays[original_role + '__paths'].tolist()
                    labels = arrays[original_role + '__labels']
                    logits = arrays[original_role + '__last__logits']
                    require(all_paths == unit[original_role], 'NPZ path/order mismatch')
                    require(labels.tolist() == [int(rows[p]['label_index']) for p in all_paths], 'NPZ labels differ from metadata')
                    require(logits.shape == (len(all_paths), 6) and np.isfinite(logits).all(), 'invalid last logits')
                    halves = {'all': {'paths': all_paths}} if role == 'test' else ctx['halves'][role]
                    scores[cfg][role] = {}
                    for half, info in halves.items():
                        allowed = set(info['paths'])
                        mask = np.array([p in allowed for p in all_paths])
                        require(int(mask.sum()) == len(allowed), 'slice missing paths')
                        y, x = labels[mask], logits[mask]
                        value = uar(y, x)
                        # Separate direct per-class mean implementation; does not call core.uar.
                        predictions = np.argmax(x, axis=1)
                        alternate = sum(float(np.mean(predictions[y == c] == c)) for c in range(6)) * (100.0 / 6.0)
                        maximum_error = max(maximum_error, abs(value - alternate))
                        require(abs(value - alternate) <= 1e-9, 'numerical replay failed')
                        scores[cfg][role][half] = value
                        metric_rows.append({'draw': draw, 'fold': fold, 'config_index': cfg,
                                            'role': role, 'half': half, 'n': int(mask.sum()), 'uar': value})
        ep, co, ct = analyze_context(draw, fold, scores)
        episodes.extend(ep)
        contrasts.extend(co)
        controls.extend(ct)
    require((len(metric_rows), len(episodes), len(contrasts), len(controls)) == (2400, 480, 240, 240), 'output counts differ')
    draws, means = aggregate(episodes, contrasts, controls)
    require(abs(means['fixed_report_exposure_pp'] - FIXED_EXPOSURE) < 1e-9, 'fixed historical exposure replay differs')
    for key in ('fixed_seen_reuse_pp', 'fixed_unseen_reuse_pp', 'fixed_test_difference_pp'):
        require(abs(means[key]) < 1e-9, 'fixed control identity failed')
    # Sources, original plan, and all prediction/receipt/DONE bytes must still match.
    check_plan(repo, archive, out)
    for rel, sha in hashes.items():
        require(file_sha(archive / rel) == sha, 'archive changed during analysis: ' + rel)
    for name, values in [('metrics', metric_rows), ('episodes', episodes), ('contrasts', contrasts),
                         ('controls', controls), ('draws', draws)]:
        csv_write(dest / (name + '.csv'), values)
    result = {'program': PROGRAM, 'status': 'complete', 'scope': 'post-result exploratory; fixed archived last15 predictions; no new inference',
        'plan_sha256': plan['plan_sha256'], 'started_at': started, 'finished_at': now(),
        'wall_seconds': time.perf_counter() - start_clock, 'python': platform.python_version(), 'numpy': np.__version__,
        'new_gpu_training_runs': 0, 'new_model_inference_runs': 0, 'original_trajectories': 480,
        'means': means, 'draw_count': 24, 'draw_aggregation': 'directions then five folds then 24 draws, equal weights',
        'config_counts': {rule: dict(sorted(Counter(r['config_index'] for r in episodes if r['rule'] == rule).items())) for rule in ('seen', 'unseen')},
        'checks': {'source_and_archive_unchanged': True, 'uar_replay_values': 2400, 'uar_replay_max_abs_pp': maximum_error,
                   'fixed_historical_exposure_pp': FIXED_EXPOSURE, 'fixed_identities_pass': True},
        'statistical_tests_performed': False, 'confidence_intervals_computed': False}
    write_json(dest / 'results.json', result)
    output_hashes = {p.name: file_sha(p) for p in sorted(dest.iterdir()) if p.is_file()}
    write_json(dest / 'FILE_SHA256.json', output_hashes)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'analyze'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo, archive, out = args.repo.resolve(), args.archive.resolve(), args.out.resolve()
    require(Path(__file__).resolve().parents[2] == repo, 'runner outside intended repo')
    require(not out.is_relative_to(archive) and not archive.is_relative_to(out), 'new output must be separate from original archive')
    (prepare if args.action == 'prepare' else analyze)(repo, archive, out)


if __name__ == '__main__':
    main()

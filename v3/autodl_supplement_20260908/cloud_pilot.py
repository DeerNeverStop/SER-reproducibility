"""Read-only technical admission of all eight AutoDL pilot trajectories.

No training, model loading or scientific outer metric is performed here.
Large checkpoints are stat-checked; their earlier byte/restore verification is
bound through the runner's immutable receipt and COMPLETE marker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .engine import GROUPS, RULES, digest, file_sha, require, unit_identity
from .plan import CORPORA, MODELS, read_json, validate_plan

TOLERANCE = 1e-5
HISTORY_INTEGER = ('epoch', 'optimizer_steps', 'scaler_skipped_steps')
HISTORY_FLOAT = ('train_loss', *RULES)
FILES = ('checkpoint.pt', 'predictions.npz', 'history.json')


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _read(path, tracked, root):
    require(path.is_file() and not path.is_symlink(), f'missing/unsafe file: {path.name}')
    sha = file_sha(path)
    value = json.loads(path.read_bytes())
    require(file_sha(path) == sha, 'JSON changed during read')
    tracked[path.relative_to(root).as_posix()] = sha
    return value


def _inventory(plan):
    pilots = plan['pilots']
    require(len(pilots) == 8 and len({u['unit_id'] for u in pilots}) == 8, 'need exact eight pilots')
    expected = {(c, m, 15): 1 for c in CORPORA for m in MODELS}
    expected.update({(c, 'wavlm_base_plus', 45): 1 for c in ('subesco', 'ravdess')})
    require(Counter((u['corpus'], u['model'], u['config']['epochs']) for u in pilots) == expected,
            'pilot factor/window inventory differs')
    require(Counter((u['corpus'], u['model']) for u in plan['units'])
            == {(c, m): 120 for c in CORPORA for m in MODELS}, 'formal extrapolation inventory differs')
    for unit in pilots:
        unit_identity(unit)
        require(unit['phase'] == 'technical_pilot' and unit['arm'] == 'A'
                and unit['backend_required'] == 'autodl', 'pilot identity differs')
        uid = unit['unit_id']
        require(isinstance(uid, str) and uid and all(c.isalnum() or c in '_-' for c in uid), 'unsafe UID')
    by_group = {(u['corpus'], u['model'], u['config']['epochs']): u for u in pilots}
    # Same context and streams across the actual pilot backbones and prefix pair.
    matched = ('draw', 'fold', 'n_classes', 'fit', 'report', 'report_batches', 'seeds',
               'reference_panel_sha256')
    for corpus in CORPORA:
        anchor = by_group[corpus, 'hubert_base', 15]
        for unit in (u for u in pilots if u['corpus'] == corpus):
            require(all(unit[k] == anchor[k] for k in matched), 'pilot matched contexts/streams differ')
    for corpus in ('subesco', 'ravdess'):
        short, long = (by_group[corpus, 'wavlm_base_plus', e] for e in (15, 45))
        require({k: v for k, v in short['config'].items() if k != 'epochs'}
                == {k: v for k, v in long['config'].items() if k != 'epochs'}, 'prefix optimizer config differs')
    return by_group


def _load_pilot(unit, plan, root, lock_sha, env_sha, tracked):
    directory = root / 'pilots' / unit['unit_id']
    require(directory.is_dir() and not directory.is_symlink()
            and directory.resolve().is_relative_to(root), 'pilot directory missing/unsafe')
    complete = _read(directory / 'COMPLETE.json', tracked, root)
    receipt = _read(directory / 'receipt.json', tracked, root)
    receipt_sha = tracked[(directory / 'receipt.json').relative_to(root).as_posix()]
    require(set(complete) == {'receipt_sha256', 'unit_id', 'unit_sha256', 'source_lock_sha256', 'plan_sha256'},
            'COMPLETE schema differs')
    require(complete['receipt_sha256'] == receipt_sha and complete['unit_id'] == unit['unit_id']
            and complete['unit_sha256'] == unit['unit_sha256']
            and complete['plan_sha256'] == plan['plan_sha256']
            and complete['source_lock_sha256'] == lock_sha, 'COMPLETE binding differs')
    require(receipt['unit_id'] == unit['unit_id'] and receipt['phase'] == unit['phase']
            and receipt['unit_sha256'] == unit['unit_sha256']
            and receipt['plan_sha256'] == plan['plan_sha256']
            and receipt['source_lock_sha256'] == lock_sha
            and receipt['environment_sha256'] == env_sha, 'receipt provenance differs')
    require(receipt.get('schema') == 'ser-autodl-receipt-1', 'receipt schema differs')
    expected_files = set(FILES) | ({'fresh_restore.json'} if unit['permanent_checkpoint_sample'] else set())
    require(set(receipt['files']) == expected_files, 'receipt artifact inventory differs')
    for name in expected_files:
        path, claim = directory / name, receipt['files'][name]
        require(set(claim) == {'sha256', 'bytes'} and _sha(claim['sha256'])
                and type(claim['bytes']) is int and claim['bytes'] > 0, 'invalid file claim')
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == claim['bytes'],
                'artifact size/presence differs')
        if name != 'checkpoint.pt':
            require(file_sha(path) == claim['sha256'], 'small artifact SHA differs')
            tracked[path.relative_to(root).as_posix()] = claim['sha256']
    info = receipt['info']
    require(info['epochs'] == unit['config']['epochs'] and info['windows'] == unit['windows']
            and info['unit_sha256'] == unit['unit_sha256']
            and info['model_identity']['name'] == unit['model'], 'engine identity differs')
    require(info['checkpoint_sha256'] == receipt['files']['checkpoint.pt']['sha256']
            and info['checkpoint_reload_verified'] is True and info['checkpoint_retained'] is True
            and info['outer_scores_computed'] is False, 'weight verification not complete')
    require(_sha(info['initial_head_sha256']), 'invalid initial head identity')
    selected = info['selected_epochs']
    require(set(selected) == {str(w) for w in unit['windows']}, 'selected windows differ')
    for window, rules in selected.items():
        require(set(rules) == set(RULES) | {'last'} and rules['last'] == int(window)
                and all(type(e) is int and 1 <= e <= int(window) for e in rules.values()),
                'selected epoch bounds differ')
    epochs = sorted({e for rules in selected.values() for e in rules.values()})
    require(info['checkpoint_unique_epochs'] == epochs and info['checkpoint_unique_epoch_count'] == len(epochs),
            'saved state inventory differs')
    differences = info['reload_by_epoch_group']
    require(set(differences) == {f'{e}/{g}' for e in epochs for g in GROUPS}
            and all(_number(v) and 0 <= v <= TOLERANCE for v in differences.values()),
            'checkpoint replay coverage or error differs')
    require(info['reload_checks'] == len(differences)
            and info['reload_checked_values'] == len(epochs) * sum(len(unit['report'][g]) * unit['n_classes'] for g in GROUPS)
            and info['reload_max_abs_diff'] == max(differences.values()), 'checkpoint replay counts differ')
    if unit['permanent_checkpoint_sample']:
        fresh = _read(directory / 'fresh_restore.json', tracked, root)
        require(fresh.get('schema') == 'ser-autodl-fresh-restore-1' and fresh.get('passed') is True
                and info.get('fresh_process_restore_verified') is True, 'fresh-process restoration missing')
        for key in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256', 'environment_sha256'):
            require(fresh[key] == receipt[key], 'fresh-process provenance differs')
        require(fresh['checkpoint_sha256'] == receipt['files']['checkpoint.pt']['sha256']
                and fresh['predictions_sha256'] == receipt['files']['predictions.npz']['sha256'],
                'fresh-process artifact identity differs')
        require(set(fresh['comparisons']) == set(differences)
                and all(_number(v) and 0 <= v <= TOLERANCE for v in fresh['comparisons'].values())
                and fresh['max_abs_diff'] == max(fresh['comparisons'].values()), 'fresh-process replay differs')
    require(_number(info['wall_seconds']) and info['wall_seconds'] > 0, 'invalid pilot wall time')
    allocated, reserved = (info[k] for k in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'))
    require(type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved, 'invalid CUDA memory receipt')
    history = _read(directory / 'history.json', tracked, root)
    require(len(history) == unit['config']['epochs'], 'history length differs')
    expected_steps = math.ceil(len(unit['fit']) / unit['config']['batch_size'])
    for epoch, row in enumerate(history, 1):
        require(set(row) == set(HISTORY_INTEGER) | set(HISTORY_FLOAT), 'unexpected history fields')
        require(all(type(row[k]) is int for k in HISTORY_INTEGER) and row['epoch'] == epoch
                and row['optimizer_steps'] == expected_steps and 0 <= row['scaler_skipped_steps'] <= expected_steps,
                'history execution extent differs')
        require(all(_number(row[k]) for k in HISTORY_FLOAT), 'nonfinite history value')
    require(info['optimizer_steps'] == sum(r['optimizer_steps'] for r in history)
            and info['scaler_skipped_steps'] == sum(r['scaler_skipped_steps'] for r in history),
            'optimizer summary differs')
    arrays = {}
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'epochs'} | {g + '__' + k for g in GROUPS for k in ('paths', 'labels', 'all_epoch_logits')},
                'prediction inventory differs')
        require(saved['epochs'].dtype == np.int64 and saved['epochs'].tolist() == list(range(1, unit['config']['epochs'] + 1)),
                'prediction epoch coverage differs')
        for group in GROUPS:
            paths, labels, logits = (saved[group + '__' + k] for k in ('paths', 'labels', 'all_epoch_logits'))
            expected_labels = [int(plan['rows'][unit['corpus']][p]['label_index']) for p in unit['report'][group]]
            require(paths.dtype.kind == 'U' and paths.tolist() == unit['report'][group]
                    and labels.dtype == np.int64 and labels.tolist() == expected_labels, 'prediction paths/labels differ')
            require(logits.dtype == np.float64 and logits.shape == (unit['config']['epochs'], len(paths), unit['n_classes'])
                    and np.isfinite(logits).all(), 'prediction shape/values differ')
            arrays[group] = logits.copy()
    return dict(receipt=receipt, info=info, history=history, arrays=arrays, receipt_sha256=receipt_sha)


def evaluate(plan, run_dir, source_lock_sha, vram_bytes):
    """Return a sealed technical gate; integrity errors raise, failed checks block.

    ``run_dir`` contains ENVIRONMENT.json and pilots/<uid>/ five files.
    Prefix tolerance is absolute 1e-5, with no relative-error escape hatch.
    """
    validate_plan(plan)
    require(_sha(source_lock_sha), 'invalid source lock identity')
    require(type(vram_bytes) is int and vram_bytes > 0, 'invalid physical VRAM bytes')
    root = Path(run_dir).resolve()
    indexed = _inventory(plan)
    require({p.name for p in (root / 'pilots').iterdir() if p.is_dir()}
            == {u['unit_id'] for u in plan['pilots']}, 'pilot directory set differs')
    tracked = {}
    environment = _read(root / 'ENVIRONMENT.json', tracked, root)
    env_sha = environment.get('environment_sha256')
    require(_sha(env_sha) and digest({k: v for k, v in environment.items() if k != 'environment_sha256'}) == env_sha,
            'environment seal differs')
    require(environment.get('schema') == 'ser-autodl-environment-1' and environment.get('backend') == 'autodl'
            and environment.get('vram_bytes') == vram_bytes, 'AutoDL environment/VRAM identity differs')
    data = {u['unit_id']: _load_pilot(u, plan, root, source_lock_sha, env_sha, tracked) for u in plan['pilots']}
    prefix_checks = []
    for corpus in ('subesco', 'ravdess'):
        short_u, long_u = (indexed[corpus, 'wavlm_base_plus', e] for e in (15, 45))
        short, long = data[short_u['unit_id']], data[long_u['unit_id']]
        logit_diff = {g: float(np.max(np.abs(short['arrays'][g] - long['arrays'][g][:15]))) for g in GROUPS}
        history_diff = {k: max(abs(s[k] - l[k]) for s, l in zip(short['history'], long['history'][:15]))
                        for k in HISTORY_FLOAT}
        integer_equal = all(s[k] == l[k] for s, l in zip(short['history'], long['history'][:15]) for k in HISTORY_INTEGER)
        selected_equal = short['info']['selected_epochs']['15'] == long['info']['selected_epochs']['15']
        passed = (max(logit_diff.values()) <= TOLERANCE and max(history_diff.values()) <= TOLERANCE
                  and integer_equal and selected_equal)
        prefix_checks.append(dict(corpus=corpus, short_unit_id=short_u['unit_id'], long_unit_id=long_u['unit_id'],
                                  pass_check=passed, logits_max_abs_by_role=logit_diff,
                                  history_max_abs_by_field=history_diff, history_integer_fields_equal=integer_equal,
                                  selected_15_equal=selected_equal))
    head_checks = []
    for corpus in CORPORA:
        units = [u for u in plan['pilots'] if u['corpus'] == corpus]
        hashes = {u['unit_id']: data[u['unit_id']]['info']['initial_head_sha256'] for u in units}
        head_checks.append(dict(corpus=corpus, head_seed=units[0]['seeds']['head'],
                                hashes=hashes, pass_check=len(set(hashes.values())) == 1))
    peaks = {uid: {k: record['info'][k] for k in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')}
             for uid, record in data.items()}
    memory_pass = all(v['peak_cuda_reserved_bytes'] <= .9 * vram_bytes for v in peaks.values())
    estimates = []
    for corpus in CORPORA:
        for model in MODELS:
            epochs = 45 if model == 'wavlm_base_plus' and corpus != 'cremad' else 15
            unit = indexed[corpus, model, epochs]
            seconds = data[unit['unit_id']]['info']['wall_seconds']
            estimates.append(dict(corpus=corpus, model=model, epochs=epochs, pilot_unit_id=unit['unit_id'],
                                  pilot_wall_seconds=seconds, formal_units=120, estimated_fit_wall_seconds=120 * seconds))
    total = sum(row['estimated_fit_wall_seconds'] for row in estimates)
    for relative, sha in tracked.items():
        require(file_sha(root / relative) == sha, 'pilot small artifact changed during evaluation')
    # Large files remain present; do not represent this stat recheck as a byte rehash.
    for uid, record in data.items():
        require((root / 'pilots' / uid / 'checkpoint.pt').stat().st_size
                == record['receipt']['files']['checkpoint.pt']['bytes'], 'checkpoint changed size')
    result = dict(schema='ser-autodl-supplement-pilot-gate-1',
                  **{'pass': memory_pass and all(c['pass_check'] for c in prefix_checks + head_checks)},
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan['plan_sha256'],
                  source_lock_sha256=source_lock_sha, environment_sha256=env_sha,
                  pilots=8, formal_units=720, files_sha256=tracked,
                  receipts_sha256={uid: record['receipt_sha256'] for uid, record in data.items()},
                  prefix_absolute_tolerance=TOLERANCE, prefix_checks=prefix_checks, head_checks=head_checks,
                  checkpoint_reload_checks=sum(record['info']['reload_checks'] for record in data.values()),
                  checkpoint_reload_max_abs_diff=max(record['info']['reload_max_abs_diff'] for record in data.values()),
                  vram_bytes=vram_bytes, max_reserved_fraction=.9, memory_pass=memory_pass, peaks_by_unit=peaks,
                  timing=dict(groups=estimates, estimated_formal_fit_wall_seconds=total,
                              estimated_formal_fit_wall_hours=total / 3600,
                              with_25pct_margin_seconds=total * 1.25, with_25pct_margin_hours=total * 1.25 / 3600,
                              scope='six individual pilot wall times times 120; includes in-engine reload; excludes independent15 companions, setup, transfer, later gates and billing; extrapolation, not a confidence interval'),
                  outer_scores_computed=False,
                  checkpoint_scope='current stat/presence and bound earlier runner byte/reload evidence; this gate does not rehash large checkpoints or independently restore them')
    result['formal_allowed'] = result['pass']
    result['pilot_gate_sha256'] = digest(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--source-lock-sha', required=True)
    p.add_argument('--vram-bytes', required=True, type=int)
    p.add_argument('--out')
    args = p.parse_args()
    root = Path(args.run_dir).resolve()
    out = Path(args.out).resolve() if args.out else root / 'pilot_gate.json'
    require(not out.exists() and not out.is_relative_to(root / 'pilots'), 'unsafe/existing gate output')
    result = evaluate(read_json(args.plan), root, args.source_lock_sha, args.vram_bytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'pass': result['pass'], 'pilot_gate_sha256': result['pilot_gate_sha256'], 'out': str(out)}))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

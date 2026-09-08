"""Independent numerical replay of the complete dual-validation experiment.

No scoring module is imported. The frozen plan loader, metadata reader and
runner's artifact gate are shared integrity dependencies; confusion matrices,
configuration decisions, fold/draw aggregation and inference are rebuilt here.
The CLI never fits a model or uses a network, and rejects incomplete formal data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.special import betainc, betaincinv

FROZEN_PLAN_SHA = '570cae10577e13bc17a5838d872251beeb52d8b216886d19a8dcb6418a7f5bbd'
SHAPE = (24, 5, 4, 3, 3)  # draw, fold, configuration, role, checkpoint
ROLE_NAMES = ('val_seen', 'val_unseen', 'test')
STATE_NAMES = ('best_seen', 'best_unseen', 'last')
FAMILY_NAMES = ('primary', 'fixed_config_best', 'fixed_config_last')
TABLES = ('pred_metrics.csv', 'selected_episodes.csv', 'draws.csv')
BOOTSTRAP_N, BOOTSTRAP_SEED, TOLERANCE = 50_000, 2026090701, 1e-9
INFERENCE_SCOPE = (
    'Conditional on this fixed CREMA-D corpus, encoder, training procedure and '
    'prespecified randomization. The 24 draws are the inferential units; five '
    'folds are averaged within each draw. Neither recordings nor people are '
    'resampled. No population, cross-corpus or causal-mechanism claim is certified.'
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_path(path):
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        try:
            status = part.lstat()
        except FileNotFoundError:
            continue
        require(not part.is_symlink() and not (getattr(status, 'st_file_attributes', 0) & 0x400),
                'linked/reparse input or output path: ' + str(part))
    return path


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding='utf-8'))


def confusion_uar(labels, logits):
    """Build the 6 by 6 matrix via joint-label counts, not Boolean class slicing."""
    y, z = np.asarray(labels), np.asarray(logits)
    require(y.ndim == 1 and y.dtype.kind in 'iu' and y.size > 0, 'invalid integer truth vector')
    require(y.min() >= 0 and y.max() < 6, 'truth is outside the fixed six classes')
    require(z.shape == (len(y), 6) and z.dtype.kind in 'fiu' and np.isfinite(z).all(), 'invalid finite logits')
    guessed = np.argmax(z, axis=1)
    matrix = np.bincount(6*y.astype(np.int64)+guessed, minlength=36).reshape(6, 6)
    support = matrix.sum(axis=1)
    require((support > 0).all(), 'whole-fold truth lacks one of the six classes')
    recalls = np.diag(matrix).astype(np.float64)/support
    # Same mathematical arithmetic order prevents floating-point tie changes;
    # the counts and decisions feeding it are independently reconstructed.
    return 100.0*float(np.add.reduce(recalls)/6.0), matrix


def linear_quantile(values, probability):
    ordered = sorted(float(x) for x in values)
    require(ordered and 0 <= probability <= 1, 'invalid quantile input')
    location = (len(ordered)-1)*probability
    low = math.floor(location)
    high = math.ceil(location)
    return ordered[low]+(ordered[high]-ordered[low])*(location-low)


def bootstrap_by_weights(effects):
    """Same PCG64 resamples, represented as counts; no indexed-value means."""
    effects = np.asarray(effects, dtype=np.float64)
    require(effects.shape == (24,) and np.isfinite(effects).all(), 'bootstrap needs 24 finite effects')
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    averages = []
    remaining = BOOTSTRAP_N
    while remaining:
        batch = min(2048, remaining)
        draws = rng.integers(0, 24, size=(batch, 24))
        weights = np.zeros((batch, 24), dtype=np.int16)
        np.add.at(weights, (np.repeat(np.arange(batch), 24), draws.ravel()), 1)
        require(np.all(weights.sum(axis=1) == 24), 'bootstrap multiplicities are invalid')
        averages.extend((weights @ effects/24.0).tolist())
        remaining -= batch
    return [linear_quantile(averages, .025), linear_quantile(averages, .975)]


def recompute_inference(effects):
    values = [float(x) for x in effects]
    require(len(values) == 24 and all(math.isfinite(x) for x in values), 't test needs exactly 24 finite draws')
    mean = math.fsum(values)/24
    identical = all(x == values[0] for x in values)
    sd = 0.0 if identical else math.sqrt(math.fsum((x-mean)**2 for x in values)/23)
    se = sd/math.sqrt(24)
    # Two-sided Student tail from incomplete beta; critical value via beta inverse.
    beta_cut = float(betaincinv(23/2, .5, .05))
    critical = math.sqrt(23*(1-beta_cut)/beta_cut)
    t_value = None if identical else mean/se
    p_value = None if identical else float(betainc(23/2, .5, 23/(23+t_value*t_value)))
    return {'estimate_pp': mean, 'draws': 24, 'df': 23, 'sample_sd_pp': sd, 'standard_error_pp': se,
        'alternative': 'two-sided', 't_statistic': t_value, 'p_value_two_sided': p_value,
        't_ci95_pp': [mean-critical*se, mean+critical*se], 't_test_defined': not identical,
        'degenerate_variance_note': ('All 24 effects are identical; t and p are undefined. '
                                    'The zero-width interval is only the zero-SE formula.' if identical else None),
        'bootstrap_sensitivity': {
            'method': 'percentile bootstrap of 24 draw effects with replacement; linear quantiles',
            'repetitions': BOOTSTRAP_N, 'seed': BOOTSTRAP_SEED, 'bit_generator': 'PCG64',
            'ci95_pp': bootstrap_by_weights(values), 'additional_hypothesis_test': False},
        'scope': INFERENCE_SCOPE,
        'contrast': 'mean_draw(mean_fold[(V_seen-T_seen)-(V_unseen-T_unseen)])'}


def grid(plan):
    units = plan['units']
    require(len(units) == 480 and len({u['unit_id'] for u in units}) == 480, 'exactly 480 unique formal units required')
    lookup = {}
    for unit in units:
        key = unit['draw'], unit['fold'], unit['config_index']
        require(all(type(x) is int for x in key) and key not in lookup, 'duplicate or invalid factorial cell')
        lookup[key] = unit
    require(set(lookup) == {(d, f, c) for d in range(24) for f in range(5) for c in range(4)}, 'formal factorial grid differs')
    settings = plan['analysis']
    require(settings['draws'] == 24 and settings['fixed_config_index'] == 3
            and settings['bootstrap_repetitions'] == BOOTSTRAP_N and settings['bootstrap_seed'] == BOOTSTRAP_SEED,
            'frozen analysis constants differ')
    for d in range(24):
        for f in range(5):
            first = lookup[d, f, 0]
            require(len(first['fit']) == 576 and len(first['val_seen']) == len(first['val_unseen']) == 288,
                    'development sample budgets differ')
            for c in range(4):
                other = lookup[d, f, c]
                require(all(first[k] == other[k] for k in ('fit', *ROLE_NAMES, 'train_seed')),
                        'candidate configuration changes the common data/seed')
            require(lookup[d, f, 3]['config']['lr_encoder'] == 5e-5
                    and lookup[d, f, 3]['config']['lr_head'] == 1e-3, 'fixed configuration 3 differs')
    return lookup


def reconstruct(plan, loader):
    """Pure numerical reconstruction. A caller supplies already sealed batches."""
    units = grid(plan)
    cube = np.empty(SHAPE, dtype=np.float64)
    epochs = np.empty(SHAPE[:3]+(3,), dtype=np.int64)
    prediction_rows, labels_by_cell = [], {}
    wanted_keys = {role+'__'+suffix for role in ROLE_NAMES
                   for suffix in ('paths', 'labels', 'best_seen__logits', 'best_unseen__logits', 'last__logits')}
    for (d, f, c), unit in units.items():
        batch = loader(unit)
        arrays, receipt = batch['predictions'], batch['receipt']
        require(set(arrays) == wanted_keys, 'prediction array inventory differs')
        chosen = [receipt['best_seen_epoch'], receipt['best_unseen_epoch'], receipt['epochs_run']]
        require(chosen[2] == 15 and all(type(e) is int and 1 <= e <= 15 for e in chosen), 'invalid checkpoint epochs')
        epochs[d, f, c] = chosen
        for ri, role in enumerate(ROLE_NAMES):
            paths, truth = np.asarray(arrays[role+'__paths']), np.asarray(arrays[role+'__labels'])
            require(paths.ndim == 1 and paths.dtype.kind in 'US' and paths.tolist() == unit[role], 'path alignment differs')
            require(truth.shape == (len(paths),), 'truth length differs')
            key = d, f, role
            if key in labels_by_cell:
                require(np.array_equal(truth, labels_by_cell[key]), 'configurations disagree on acted labels')
            else:
                labels_by_cell[key] = truth.copy()
            for ci, checkpoint in enumerate(STATE_NAMES):
                value, _ = confusion_uar(truth, arrays[f'{role}__{checkpoint}__logits'])
                cube[d, f, c, ri, ci] = value
                prediction_rows.append({'unit_id': unit['unit_id'], 'draw': d, 'fold': f, 'config_index': c,
                    'train_seed': unit['train_seed'], 'role': role, 'checkpoint': checkpoint,
                    'checkpoint_epoch': chosen[ci], 'records': len(truth), 'uar_percent': value})
            for i in range(3):
                for j in range(i+1, 3):
                    if chosen[i] == chosen[j]:
                        require(np.array_equal(arrays[f'{role}__{STATE_NAMES[i]}__logits'],
                                               arrays[f'{role}__{STATE_NAMES[j]}__logits']), 'same epoch logits differ')
    # Argmax along configuration returns the smallest index on an exact tie.
    winners = np.stack((cube[:, :, :, 0, 0].argmax(axis=2), cube[:, :, :, 1, 1].argmax(axis=2)), axis=-1)
    # Numerical layout: draw, fold, family, rule, validation/test.
    panels = np.empty((24, 5, 3, 2, 2), dtype=np.float64)
    episodes = []
    for d in range(24):
        for f in range(5):
            for rule in range(2):
                selected = int(winners[d, f, rule])
                record = {'draw': d, 'fold': f, 'rule': ('seen', 'unseen')[rule],
                    'selection_role': ROLE_NAMES[rule], 'selected_config_index': selected,
                    'selected_unit_id': units[d, f, selected]['unit_id'], 'train_seed': units[d, f, selected]['train_seed']}
                for family in range(3):
                    config = selected if family == 0 else 3
                    checkpoint = 2 if family == 2 else rule
                    v = float(cube[d, f, config, rule, checkpoint])
                    t = float(cube[d, f, config, 2, checkpoint])
                    panels[d, f, family, rule] = (v, t)
                    prefix = FAMILY_NAMES[family]
                    record.update({prefix+'_config_index': config, prefix+'_checkpoint': STATE_NAMES[checkpoint],
                        prefix+'_epoch': int(epochs[d, f, config, checkpoint]), prefix+'_validation_uar_percent': v,
                        prefix+'_test_uar_percent': t, prefix+'_gap_pp': v-t})
                episodes.append(record)
    gaps = panels[..., 0]-panels[..., 1]
    delta_v = panels[:, :, :, 0, 0]-panels[:, :, :, 1, 0]
    delta_t = panels[:, :, :, 0, 1]-panels[:, :, :, 1, 1]
    delta_gap = gaps[..., 0]-gaps[..., 1]
    require(np.array_equal(panels[:, :, 2, 0, 1], panels[:, :, 2, 1, 1])
            and np.all(delta_t[:, :, 2] == 0), 'fixed-last common-test cancellation failed')
    require(np.max(np.abs(delta_gap-(delta_v-delta_t))) <= TOLERANCE, 'difference identity failed')
    delta_gap[:, :, 2] = delta_v[:, :, 2]
    residual = delta_gap-(delta_v-delta_t)
    measures = {
        'seen_validation_uar_percent': panels[:, :, :, 0, 0], 'unseen_validation_uar_percent': panels[:, :, :, 1, 0],
        'seen_test_uar_percent': panels[:, :, :, 0, 1], 'unseen_test_uar_percent': panels[:, :, :, 1, 1],
        'seen_gap_pp': gaps[..., 0], 'unseen_gap_pp': gaps[..., 1], 'delta_validation_pp': delta_v,
        'delta_test_pp': delta_t, 'delta_gap_pp': delta_gap, 'identity_residual_pp': residual}
    draw_rows = []
    for d in range(24):
        row = {'draw': d, 'folds': 5}
        for fi, family in enumerate(FAMILY_NAMES):
            for name, matrix in measures.items():
                row[family+'_'+name] = math.fsum(float(x) for x in matrix[d, :, fi])/5
        row['selection_increment_pp'] = row['primary_delta_gap_pp']-row['fixed_config_last_delta_gap_pp']
        require(row['fixed_config_last_delta_test_pp'] == 0
                and row['fixed_config_last_delta_gap_pp'] == row['fixed_config_last_delta_validation_pp'], 'draw cancellation failed')
        draw_rows.append(row)
    summaries = {}
    for field in draw_rows[0]:
        if field not in ('draw', 'folds'):
            numbers = [r[field] for r in draw_rows]
            summaries[field] = {'mean': math.fsum(numbers)/24, 'draw_sd': statistics.stdev(numbers),
                                'minimum': min(numbers), 'maximum': max(numbers)}
    results = {
        'schema': 'ser-dual-validation-analysis-1', 'phase': 'formal', 'plan_sha256': plan['plan_sha256'],
        'analyzed_formal_units': 480, 'excluded_pilot_runs': len(plan['pilot_units']), 'draws': 24, 'folds_per_draw': 5,
        'configurations': 4, 'aggregation': 'whole-fold six-class macro recall; equal five-fold mean within draw; equal mean of 24 draws',
        'selection': 'minimum own-validation CE epoch supplied by sealed runner; maximum own-validation UAR configuration; smallest config_index on exact ties',
        'primary': recompute_inference([r['primary_delta_gap_pp'] for r in draw_rows]),
        'primary_hypothesis_tests': 1, 'descriptive_draw_summaries': summaries, 'fixed_config_index': 3,
        'fixed_config': units[0, 0, 3]['config'],
        'control_scope': {'fixed_config_best': 'fixed hyperparameters; respective own-validation checkpoint selection remains',
            'fixed_config_last': 'fixed hyperparameters and last epoch; no validation selection; identical test model',
            'selection_increment_pp': 'primary gap difference minus fixed-config last gap difference; descriptive, not an identified mediation effect'},
        'test_difference': 'primary_delta_test_pp is the paired T_seen minus T_unseen difference after each rule selects its configuration/checkpoint',
        'identity': 'delta_gap = delta_validation - delta_test; fixed-config last delta_test is exactly zero',
        'inference_scope': INFERENCE_SCOPE, 'additional_control_hypothesis_tests': 0,
        'no_outcome_based_stopping': True, 'all_prespecified_controls_reported': True}
    return {'pred_metrics.csv': prediction_rows, 'selected_episodes.csv': episodes, 'draws.csv': draw_rows, 'results.json': results}


class Comparison:
    def __init__(self):
        self.numeric_values_checked = 0
        self.maximum_absolute_error = 0.0

    def check(self, expected, actual, location):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), 'JSON field inventory differs: '+location)
            for key in expected:
                self.check(expected[key], actual[key], location+'.'+key)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), 'JSON list length differs: '+location)
            for i, value in enumerate(expected):
                self.check(value, actual[i], location+f'[{i}]')
        elif expected is None or type(expected) in (str, bool):
            require(type(actual) is type(expected) and actual == expected, 'non-numeric field differs: '+location)
        else:
            require(type(expected) in (int, float) and type(actual) in (int, float), 'numeric type differs: '+location)
            require(math.isfinite(expected) and math.isfinite(actual), 'nonfinite compared value: '+location)
            error = abs(expected-actual)
            self.numeric_values_checked += 1
            self.maximum_absolute_error = max(self.maximum_absolute_error, error)
            require(error <= TOLERANCE and (type(expected) is not int or type(actual) is int),
                    f'numeric value differs: {location}; absolute error {error}')


def compare_csv(path, expected, keys, comparator):
    with safe_path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        header, observed = reader.fieldnames, list(reader)
    columns = list(expected[0])
    require(header == columns, 'CSV column inventory/order differs: '+Path(path).name)
    require(len(observed) == len(expected), 'CSV row count differs: '+Path(path).name)
    wanted = {tuple(str(row[k]) for k in keys): row for row in expected}
    require(len(wanted) == len(expected), 'internal duplicate expected CSV key')
    actual_by_key = {}
    for row in observed:
        require(set(row) == set(columns) and all(value is not None for value in row.values()), 'malformed CSV row')
        key = tuple(row[k] for k in keys)
        require(key not in actual_by_key and key in wanted, 'duplicate/unexpected CSV key')
        actual_by_key[key] = row
    before, maximum = comparator.numeric_values_checked, 0.0
    per_column = Counter()
    for key, expected_row in wanted.items():
        actual_row = actual_by_key[key]
        for field, expected_value in expected_row.items():
            raw = actual_row[field]
            if type(expected_value) is int:
                require(raw == str(int(raw)), 'noncanonical integer CSV value')
                actual_value = int(raw)
            elif type(expected_value) is float:
                actual_value = float(raw)
            else:
                actual_value = raw
            comparator.check(expected_value, actual_value, f'{Path(path).name}:{key}:{field}')
            if type(expected_value) in (int, float):
                per_column[field] += 1
                maximum = max(maximum, abs(expected_value-actual_value))
    return {'rows': len(observed), 'columns': columns, 'sha256': sha_file(path),
            'numeric_values_checked': comparator.numeric_values_checked-before,
            'numeric_values_checked_by_column': dict(per_column), 'maximum_absolute_error': maximum}


def accepted_gate(plan, gate):
    require(gate.get('schema') == 'ser-dual-validation-result-gate-1' and gate.get('pass') is True
            and gate.get('phase') == 'formal' and type(gate.get('units')) is int and gate['units'] == 480,
            'only a complete 480-unit formal gate is accepted')
    require(gate.get('plan_sha256') == plan['plan_sha256'] and gate.get('scores_computed') is False, 'gate identity differs')
    require(content_hash({k:v for k,v in gate.items() if k!='gate_sha256'}) == gate.get('gate_sha256'), 'gate content hash differs')
    require(set(gate['done_sha256']) == {u['unit_id'] for u in plan['units']}, 'gate DONE set differs')


def audit(repo, plan_path, outroot, gate_path, results_path, out):
    replay_source_sha = sha_file(__file__)
    repo, plan_path, outroot, gate_path, results_path, out = map(safe_path, (repo, plan_path, outroot, gate_path, results_path, out))
    require(not out.exists() and not out.is_relative_to(outroot) and not out.is_relative_to(results_path)
            and out not in (plan_path, gate_path, Path(__file__).resolve()), 'audit output overlaps immutable inputs or exists')
    require({p.name for p in results_path.iterdir()} == set(TABLES)|{'results.json'}, 'main report must contain exactly three CSVs and results.json')
    sys.path.insert(0, str(repo))
    planner = importlib.import_module('v3.inner_validation.plan')
    runner = importlib.import_module('v3.inner_validation.run')
    metadata = importlib.import_module('v3.data_design.core_plan')
    for module, name in ((planner, 'v3/inner_validation/plan.py'), (runner, 'v3/inner_validation/run.py'),
                         (metadata, 'v3/data_design/core_plan.py')):
        require(Path(module.__file__).resolve() == repo/name, 'shared verifier imported from another checkout')
    plan = planner.load_plan(plan_path, repo)
    require(plan['plan_sha256'] == FROZEN_PLAN_SHA, 'not the designated frozen dual-validation plan')
    grid(plan)
    gate = read_json(gate_path)
    accepted_gate(plan, gate)
    gate_file_sha = sha_file(gate_path)
    rows = {r['relative_path']:r for r in metadata.read_metadata(repo)['clean']}
    fresh_gate = runner.result_gate(plan, outroot, 'formal', rows)
    require(fresh_gate['done_sha256'] == gate['done_sha256'] and fresh_gate['ledger_sha256'] == gate['ledger_sha256'],
            'current full artifact/ledger closure differs from accepted gate')
    report_hashes = {name: sha_file(results_path/name) for name in (*TABLES, 'results.json')}
    actual_result = read_json(results_path/'results.json')
    require(actual_result['table_sha256'] == {name:report_hashes[name] for name in TABLES}, 'main CSV byte identities differ')
    closure = {}
    for unit in plan['units']:
        folder = safe_path(outroot/'formal/units'/unit['unit_id'])
        done_path = safe_path(folder/'DONE')
        require(sha_file(done_path) == gate['done_sha256'][unit['unit_id']], 'DONE changed after integrity verification')
        done = read_json(done_path)
        pred = safe_path(folder/'attempts/0001/predictions.npz')
        receipt = safe_path(folder/'attempts/0001/receipt.json')
        closure[unit['unit_id']] = (pred, done['artifacts']['attempts/0001/predictions.npz'],
                                   receipt, done['artifacts']['attempts/0001/receipt.json'], done_path)
    def loader(unit):
        pred, pred_sha, receipt, receipt_sha, done_path = closure[unit['unit_id']]
        require(sha_file(pred) == pred_sha and sha_file(receipt) == receipt_sha, 'committed prediction/receipt bytes changed')
        with np.load(pred, allow_pickle=False) as arrays:
            values = {key: arrays[key].copy() for key in arrays.files}
        for role in ROLE_NAMES:
            require(values[role+'__labels'].tolist() == [int(rows[p]['label_index']) for p in unit[role]],
                    'truth differs from byte-verified manifest metadata')
        require(sha_file(pred) == pred_sha and sha_file(done_path) == gate['done_sha256'][unit['unit_id']], 'closure changed while reading')
        return {'predictions': values, 'receipt': read_json(receipt)}
    rebuilt = reconstruct(plan, loader)
    rebuilt['results.json'].update(source_sha256=sha_file(repo/'v3/inner_validation/score.py'),
        plan_file_sha256=sha_file(plan_path), accepted_gate_file_sha256=gate_file_sha,
        accepted_gate_sha256=gate['gate_sha256'], done_mapping_sha256=content_hash(gate['done_sha256']),
        table_sha256={name: report_hashes[name] for name in TABLES})
    comparison = Comparison()
    key_columns = {'pred_metrics.csv': ('unit_id', 'role', 'checkpoint'),
                   'selected_episodes.csv': ('draw', 'fold', 'rule'), 'draws.csv': ('draw',)}
    table_checks = {name: compare_csv(results_path/name, rebuilt[name], key_columns[name], comparison) for name in TABLES}
    before_json = comparison.numeric_values_checked
    comparison.check(rebuilt['results.json'], actual_result, 'results.json')
    planner.load_plan(plan_path, repo)
    require(sha_file(gate_path) == gate_file_sha and sha_file(outroot/'formal/ledger.jsonl') == gate['ledger_sha256'], 'gate/ledger changed during replay')
    for uid, (pred, pred_sha, receipt, receipt_sha, done_path) in closure.items():
        require(sha_file(pred) == pred_sha and sha_file(receipt) == receipt_sha
                and sha_file(done_path) == gate['done_sha256'][uid], 'unit closure changed during replay')
    require(all(sha_file(results_path/name) == wanted for name,wanted in report_hashes.items()), 'main report changed during replay')
    require(sha_file(__file__) == replay_source_sha, 'replay source changed during execution')
    result = {'schema': 'ser-dual-validation-independent-numerical-replay-1', 'pass': True,
        'verified_at': datetime.now(timezone.utc).isoformat(), 'phase': 'formal', 'formal_units': 480,
        'excluded_pilot_runs': 4, 'plan_sha256': plan['plan_sha256'], 'plan_file_sha256': sha_file(plan_path),
        'accepted_gate_file_sha256': gate_file_sha, 'accepted_gate_sha256': gate['gate_sha256'],
        'fresh_integrity_gate_sha256': fresh_gate['gate_sha256'], 'done_mapping_sha256': content_hash(gate['done_sha256']),
        'replay_source_sha256': replay_source_sha, 'score_source_sha256': sha_file(repo/'v3/inner_validation/score.py'),
        'main_results_file_sha256': report_hashes['results.json'], 'table_checks': table_checks,
        'results_json_numeric_values_checked': comparison.numeric_values_checked-before_json,
        'numeric_values_checked': comparison.numeric_values_checked,
        'maximum_absolute_error': comparison.maximum_absolute_error, 'absolute_tolerance': TOLERANCE,
        'bootstrap_repetitions': BOOTSTRAP_N, 'bootstrap_seed': BOOTSTRAP_SEED,
        'independent_aggregation': 'six-by-six confusion matrices; own-validation argmax configuration; equal-fold then equal-draw averaging',
        'independent_inference': 'Student tail/critical value from incomplete beta; PCG64 resamples accumulated as multiplicity weights; explicit interpolated quantiles',
        'shared_integrity_dependencies': ['frozen plan.load_plan', 'frozen run.result_gate', 'frozen metadata reader'],
        'primary_score_functions_imported': False, 'new_model_fits': 0, 'network_used': False,
        'scope': 'Numerical and artifact consistency only; shared integrity code is not an independent implementation of training or experimental design.'}
    result['audit_sha256'] = content_hash(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'plan', 'outroot', 'gate', 'results', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo, args.plan, args.outroot, args.gate, args.results, args.out)
    print(json.dumps({k:result[k] for k in ('pass', 'formal_units', 'numeric_values_checked', 'maximum_absolute_error')}, indent=2))


if __name__ == '__main__':
    main()

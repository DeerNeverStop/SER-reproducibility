"""Independent numeric replay after all 384 formal units and main scoring.

Never imports score.py or engine.metrics. The frozen plan/source checker is
shared; native-class recall, CE, epoch selection, aggregation, t inference and
Holm adjustment are implemented here. No fitting, GPU, audio inference or
checkpoint deserialization. The prior complete gate binds checkpoint identities;
this audit freshly hashes all predictions, receipts, histories and DONE files,
but does not rehash huge checkpoint blobs. Never run against partial results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import time

import numpy as np
from scipy.stats import t as t_distribution


PROGRAM = 'SER26-FINAL-CHECKPOINT-PROGRAM-1'
CORPORA = ('cremad', 'subesco', 'ravdess')
CLASSES = dict(cremad=6, subesco=7, ravdess=8)
ROLES = ('A', 'B', 'outer')
RULES = ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar', 'last')
TABLE_COUNTS = dict(contexts=360, draws=72, selected_epochs=1800, curves=5400, controls=24)
TOLERANCE = 1e-9
INTERPRETATION = {
    'positive_delta': 'seen-based checkpoint rule has higher outer UAR; not an estimate of V-T optimism',
    'J': 'difference between CE-based and UAR-based checkpoint-rule effects',
    'scope': 'conditional fixed-corpus randomization and training program; native tasks differ across corpora',
    'prior_data_exposure': True, 'bootstrap': False, 'equivalence_test': False,
    'curves_and_oracle': 'descriptive; outer never selects a retained checkpoint',
    'control': 'whole training-group replacement; not an individual speaker effect; query also used by main A selection',
    'zero_variance': 'undefined t remains untested; reserved Holm family slot treated as p=1 internally',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def average(values):
    items = list(values)
    require(items and all(math.isfinite(float(value)) for value in items), 'invalid mean')
    return math.fsum(items) / len(items)


def formal_units(plan):
    require(plan['program'] == PROGRAM, 'wrong study program')
    units = [u for u in plan['units'] if u['phase'] == 'formal']
    desired = {(c, d, f, 'A') for c in CORPORA for d in range(24) for f in range(5)}
    desired.update(('cremad', d, 0, 'B') for d in range(24))
    coordinates = [(u['corpus'], u['draw'], u['fold'], u['arm']) for u in units]
    require(len(units) == len(set(coordinates)) == len({u['unit_id'] for u in units}) == 384
            and set(coordinates) == desired, 'all 360 main and 24 control units required')
    for unit in units:
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', unit['unit_id']) is not None
                and unit['unit_id'] not in ('.', '..'), 'unsafe unit ID')
        require(unit['n_classes'] == CLASSES[unit['corpus']], 'native label count differs')
        require(unit['selection_enabled'] is (unit['arm'] == 'A')
                and unit['prediction_epochs'] == (list(range(1, 16)) if unit['arm'] == 'A' else [15]),
                'main/control epoch contract differs')
    return sorted(units, key=lambda u: (CORPORA.index(u['corpus']), u['draw'], u['fold'], u['arm']))


def verify_arrays(unit, arrays, rows):
    expected_keys = {'epochs'} | {role + '__' + field for role in ROLES
                                  for field in ('paths', 'labels', 'all_epoch_logits')}
    require(set(arrays) == expected_keys, 'prediction array inventory differs')
    epochs = arrays['epochs']
    require(epochs.ndim == 1 and epochs.dtype.kind in 'iu'
            and epochs.tolist() == unit['prediction_epochs'], 'prediction epoch index differs')
    for role in ROLES:
        paths, labels, logits = (arrays[role + '__' + field] for field in
                                 ('paths', 'labels', 'all_epoch_logits'))
        require(paths.ndim == 1 and paths.tolist() == unit['report'][role], 'prediction paths differ')
        expected = np.asarray([int(rows[path]['label_index']) for path in unit['report'][role]], dtype=np.int64)
        require(labels.dtype.kind in 'iu' and labels.shape == expected.shape and np.array_equal(labels, expected),
                'prediction labels differ from frozen rows')
        require(set(labels.tolist()) == set(range(unit['n_classes'])), 'native class support differs')
        require(logits.dtype == np.float32 and logits.shape == (len(epochs), len(paths), unit['n_classes'])
                and np.isfinite(logits).all(), 'prediction tensor support differs')


def native_metric(labels, logits):
    """Boolean class recalls, independently of the main scorer's matrix code."""
    labels, logits = np.asarray(labels), np.asarray(logits)
    require(labels.ndim == 1 and labels.dtype.kind in 'iu' and len(labels) > 0
            and logits.ndim == 2 and len(logits) == len(labels)
            and np.isfinite(logits).all(), 'invalid metric input')
    classes = logits.shape[1]
    require(set(labels.tolist()) == set(range(classes)), 'metric label support differs')
    winners = np.argmax(logits, axis=1)
    recalls = []
    for label in range(classes):
        relevant = labels == label
        recalls.append(Fraction(int(np.count_nonzero(winners[relevant] == label)), int(np.count_nonzero(relevant))))
    macro = sum(recalls, Fraction(0)) / classes
    # Exact CE ties follow the frozen float64 operation order. This standalone
    # implementation does not call either production metric implementation.
    matrix = np.asarray(logits, dtype=np.float64)
    peak = np.amax(matrix, axis=1)
    exponentials = np.exp(matrix - peak[:, np.newaxis])
    log_partition = peak + np.log(np.sum(exponentials, axis=1))
    cross_entropy = float(np.mean(log_partition - matrix[np.arange(len(labels)), labels]))
    require(math.isfinite(cross_entropy) and cross_entropy >= 0, 'invalid CE')
    return cross_entropy, macro


def read_trajectory(unit, arrays, rows):
    verify_arrays(unit, arrays, rows)
    epochs = unit['prediction_epochs']
    measures = {role: {epoch: native_metric(arrays[role + '__labels'], logits)
                       for epoch, logits in zip(epochs, arrays[role + '__all_epoch_logits'])}
                for role in ROLES}
    selected = {'last': 15}
    if unit['arm'] == 'A':
        for role, prefix in (('A', 'seen'), ('B', 'unseen')):
            selected[prefix + '_ce'] = min(epochs, key=lambda epoch: (measures[role][epoch][0], epoch))
            selected[prefix + '_uar'] = min(epochs, key=lambda epoch: (-measures[role][epoch][1], epoch))
    return {'unit': unit, 'values': measures, 'selected': selected}


def independent_t(values):
    values = [float(v) for v in values]
    require(len(values) == 24 and all(math.isfinite(v) for v in values), 't requires 24 complete draws')
    center = average(values)
    sd = 0. if all(v == values[0] for v in values) else math.sqrt(math.fsum((v-center)**2 for v in values)/23)
    result = {'mean_pp': center, 'n_draws': 24, 'df': 23, 'sd_draw_pp': sd,
              'interval_scope': 'pointwise, not simultaneous; conditional on fixed corpus and program'}
    if sd == 0:
        result.update(status='undefined_t_zero_sample_variance', se_pp=0., t_statistic=None,
                      p_two_sided=None, pointwise_95_ci_pp=None)
    else:
        error = sd / math.sqrt(24)
        statistic = center / error
        half_width = float(t_distribution.ppf(.975, df=23)) * error
        result.update(status='estimated', se_pp=error, t_statistic=statistic,
                      p_two_sided=float(2*t_distribution.sf(abs(statistic), df=23)),
                      pointwise_95_ci_pp=[center-half_width, center+half_width])
    return result


def adjust_six(tests):
    require(len(tests) == 6, 'fixed six-test family required')
    ordered = sorted(range(6), key=lambda index: (tests[index]['p_two_sided']
                     if tests[index]['p_two_sided'] is not None else 1., index))
    products = [(6-rank)*(tests[index]['p_two_sided'] if tests[index]['p_two_sided'] is not None else 1.)
                for rank, index in enumerate(ordered)]
    for rank, index in enumerate(ordered):
        p = tests[index]['p_two_sided']
        adjusted = min(1., max(products[:rank+1])) if p is not None else None
        tests[index].update(holm_p_six=adjusted, reject_familywise_05=bool(adjusted is not None and adjusted <= .05))


def recompute(plan, loader):
    """Pure reference calculation; I/O caller must establish full gate first."""
    ordered = formal_units(plan)
    trajectories = {}
    tables = {key: [] for key in TABLE_COUNTS}
    exact, by_draw = {}, defaultdict(list)
    for unit in ordered:
        value = read_trajectory(unit, loader(unit), plan['rows'][unit['corpus']])
        identity = (unit['corpus'], unit['draw'], unit['fold'], unit['arm'])
        trajectories[identity] = value
        if unit['arm'] == 'B':
            continue
        values, picked = value['values'], value['selected']
        base = dict(unit_id=unit['unit_id'], corpus=unit['corpus'], draw=unit['draw'], fold=unit['fold'])
        percentages = {epoch: 100*float(values['outer'][epoch][1]) for epoch in range(1, 16)}
        oracle = max(percentages.values())
        for epoch in range(1, 16):
            tables['curves'].append({**base, 'epoch': epoch, 'seen_ce': values['A'][epoch][0],
                'unseen_ce': values['B'][epoch][0], 'seen_uar': 100*float(values['A'][epoch][1]),
                'unseen_uar': 100*float(values['B'][epoch][1]), 'outer_uar': percentages[epoch]})
        for rule in RULES:
            epoch = picked[rule]
            tables['selected_epochs'].append({**base, 'rule': rule, 'selected_epoch': epoch,
                'seen_uar': 100*float(values['A'][epoch][1]), 'unseen_uar': 100*float(values['B'][epoch][1]),
                'outer_uar': percentages[epoch], 'oracle_outer_uar': oracle,
                'oracle_shortfall_pp': oracle-percentages[epoch], 'oracle_used_for_selection': False})
        outer_recall = lambda rule: values['outer'][picked[rule]][1]
        ce_difference = (outer_recall('seen_ce')-outer_recall('unseen_ce'))*100
        uar_difference = (outer_recall('seen_uar')-outer_recall('unseen_uar'))*100
        key = identity[:3]
        exact[key] = (ce_difference, uar_difference, ce_difference-uar_difference)
        row = {**base, 'delta_CE_pp': float(ce_difference), 'delta_UAR_pp': float(uar_difference),
               'J_pp': float(ce_difference-uar_difference), 'last_outer_uar': percentages[15],
               'oracle_outer_uar': oracle,
               'middle_8_10_minus_late_14_15_pp': average(percentages[e] for e in (8, 9, 10))
                                                - average(percentages[e] for e in (14, 15)),
               'ce_epoch_agreement': int(picked['seen_ce'] == picked['unseen_ce']),
               'uar_epoch_agreement': int(picked['seen_uar'] == picked['unseen_uar'])}
        tables['contexts'].append(row)
        by_draw[identity[:2]].append(row)
    for corpus in CORPORA:
        for draw in range(24):
            entries = by_draw[corpus, draw]
            require(len(entries) == 5 and {r['fold'] for r in entries} == set(range(5)), 'missing fold')
            effects = [float(sum((exact[corpus, draw, fold][index] for fold in range(5)), Fraction())/5)
                       for index in range(3)]
            tables['draws'].append(dict(corpus=corpus, draw=draw,
                delta_CE_pp=effects[0], delta_UAR_pp=effects[1], J_pp=effects[2],
                last_outer_uar=average(r['last_outer_uar'] for r in entries),
                middle_8_10_minus_late_14_15_pp=average(r['middle_8_10_minus_late_14_15_pp'] for r in entries)))
    for draw in range(24):
        left = trajectories['cremad', draw, 0, 'A']
        right = trajectories['cremad', draw, 0, 'B']
        require(all(left['unit'][key] == right['unit'][key] for key in
                    ('pair_id', 'report', 'report_batches', 'seeds')), 'control pair frame differs')
        a, b, c, d = (left['values']['A'][15][1], right['values']['A'][15][1],
                       left['values']['B'][15][1], right['values']['B'][15][1])
        tables['controls'].append(dict(corpus='cremad', draw=draw, fold=0, pair_id=left['unit']['pair_id'],
            A_unit_id=left['unit']['unit_id'], B_unit_id=right['unit']['unit_id'],
            QA_MA15=100*float(a), QA_MB15=100*float(b), QB_MA15=100*float(c), QB_MB15=100*float(d),
            E_pp=float((a-b+d-c)*50)))
    tests = []
    corpora = {}
    for corpus in CORPORA:
        draw_rows = [r for r in tables['draws'] if r['corpus'] == corpus]
        context_rows = [r for r in tables['contexts'] if r['corpus'] == corpus]
        for effect in ('delta_CE', 'J'):
            tests.append(dict(corpus=corpus, estimand=effect,
                              **independent_t([r[effect+'_pp'] for r in draw_rows])))
        rule_summary = {}
        for rule in RULES:
            chosen = [r for r in tables['selected_epochs'] if r['corpus'] == corpus and r['rule'] == rule]
            rule_summary[rule] = dict(mean_outer_uar=average(r['outer_uar'] for r in chosen),
                mean_selected_epoch=average(r['selected_epoch'] for r in chosen),
                selected_epoch_counts={str(epoch): count for epoch, count in sorted(Counter(r['selected_epoch'] for r in chosen).items())},
                mean_oracle_shortfall_pp=average(r['oracle_shortfall_pp'] for r in chosen))
        corpora[corpus] = dict(main_trajectories=120, n_draws=24, folds_per_draw=5,
            delta_UAR_descriptive_mean_pp=average(r['delta_UAR_pp'] for r in draw_rows),
            last_outer_descriptive_mean_uar=average(r['last_outer_uar'] for r in draw_rows),
            middle_8_10_minus_late_14_15_descriptive_mean_pp=average(r['middle_8_10_minus_late_14_15_pp'] for r in draw_rows),
            ce_epoch_agreement_fraction=average(r['ce_epoch_agreement'] for r in context_rows),
            uar_epoch_agreement_fraction=average(r['uar_epoch_agreement'] for r in context_rows), rules=rule_summary)
    adjust_six(tests)
    require({name: len(rows) for name, rows in tables.items()} == TABLE_COUNTS, 'table extent differs')
    result = dict(schema='ser-final-program-score-1', program=PROGRAM, plan_sha256=plan['plan_sha256'],
        analysis_contract=plan['analysis'],
        counts=dict(formal_fits=384, main_A=360, control_B=24, pilot_scored=0,
                    main_contexts=360, corpus_draws=72, family_tests=6),
        tests=tests, corpora=corpora,
        control=dict(estimand='0.5*((QA(MA15)-QA(MB15))+(QB(MB15)-QB(MA15)))', n_pairs=24, n_added_B_fits=24,
            scope='CREMA-D prespecified fold0; descriptive only', E_mean_pp=average(r['E_pp'] for r in tables['controls']), inference=False),
        interpretation=INTERPRETATION.copy())
    return result, tables, {value['unit']['unit_id']: value['selected'] for value in trajectories.values()}


class Comparator:
    def __init__(self):
        self.numeric_values_checked = 0
        self.max_abs_error = 0.
        self.mismatch_count = 0
        self.examples = []
        self.column_errors = {}

    def bad(self, path, expected, observed):
        self.mismatch_count += 1
        if len(self.examples) < 40:
            self.examples.append(dict(path=path, expected=str(expected), observed=str(observed)))

    def check(self, expected, observed, path, column=None):
        if isinstance(expected, dict):
            if not isinstance(observed, dict) or set(expected) != set(observed):
                self.bad(path, sorted(expected), sorted(observed) if isinstance(observed, dict) else type(observed).__name__)
                return
            for key in expected:
                self.check(expected[key], observed[key], path+'.'+key, column)
        elif isinstance(expected, list):
            if not isinstance(observed, list) or len(expected) != len(observed):
                self.bad(path, len(expected), len(observed) if isinstance(observed, list) else type(observed).__name__)
                return
            for i, (left, right) in enumerate(zip(expected, observed)):
                self.check(left, right, f'{path}[{i}]', column)
        elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
            if type(observed) is not type(expected) or observed != expected:
                self.bad(path, expected, observed)
        elif isinstance(expected, (int, float)):
            self.numeric_values_checked += 1
            if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
                self.bad(path, expected, observed)
                return
            error = abs(float(expected)-float(observed))
            self.max_abs_error = max(self.max_abs_error, error)
            if column is not None:
                self.column_errors[column] = max(self.column_errors.get(column, 0.), error)
            if (isinstance(expected, int) and observed != expected) or error > TOLERANCE:
                self.bad(path, expected, observed)
        else:
            raise TypeError('unsupported comparison type '+type(expected).__name__)

    def csv(self, path, reference):
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        require(columns and len(columns) == len(set(columns)) and set(columns) == set(reference[0]), 'CSV columns differ: '+path.name)
        require(len(rows) == len(reference), 'CSV row count differs: '+path.name)
        before, errors = self.numeric_values_checked, self.mismatch_count
        for i, (expected, observed) in enumerate(zip(reference, rows)):
            for key, wanted in expected.items():
                text = observed[key]
                try:
                    if isinstance(wanted, bool):
                        require(text in ('True', 'False'), 'invalid CSV boolean')
                        actual = text == 'True'
                    elif isinstance(wanted, int):
                        require(re.fullmatch(r'-?\d+', text) is not None, 'invalid CSV integer')
                        actual = int(text)
                    elif isinstance(wanted, float):
                        actual = float(text)
                    else:
                        actual = text
                except (ValueError, TypeError):
                    self.bad(f'{path.name}[{i}].{key}', wanted, text)
                    continue
                self.check(wanted, actual, f'{path.name}[{i}].{key}', path.name+'.'+key)
        return dict(rows=len(rows), columns=columns, sha256=sha(path),
                    numeric_values_checked=self.numeric_values_checked-before, mismatches=self.mismatch_count-errors)


def checked_runner(repo):
    sys.path.insert(0, str(repo))
    module = importlib.import_module('v3.final_program_20260907.run')
    require(Path(module.__file__).resolve() == (repo/'v3/final_program_20260907/run.py').resolve(), 'wrong cached runner checkout')
    module.check_import_locations(repo)
    return module


def gate_identity(gate, plan, lock):
    ids = {unit['unit_id'] for unit in formal_units(plan)}
    require(gate.get('schema') == 'ser-final-program-complete-gate-1' and gate.get('pass') is True
            and gate.get('phase') == lock['phase'] == 'formal'
            and gate.get('expected_units') == gate.get('verified_units') == 384,
            'complete formal 384 gate is required')
    require(gate['plan_sha256'] == plan['plan_sha256'] == lock['plan_sha256']
            and gate['lock_sha256'] == lock['lock_sha256'], 'gate phase identity differs')
    require(set(gate['done_sha256']) == set(gate['artifacts_sha256']) == ids, 'gate does not bind exact 384 IDs')


def execute(args):
    began = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    repo, phase, results, out = (Path(value).resolve() for value in
                                 (args.repo, args.run_dir, args.results, args.out_json))
    require(out.suffix.lower() == '.json' and not out.exists() and not out.is_relative_to(phase)
            and not out.is_relative_to(results), 'audit output must be fresh and outside inputs/results')
    operator_sha = sha(__file__)
    runner = checked_runner(repo)
    plan, lock = runner.checked_plan(repo, phase)
    require(lock['phase'] == 'formal', 'pilot numerical scoring is prohibited')
    units = formal_units(plan)
    observed = json_read(results/'results.json')
    require(observed['result_sha256'] == semantic_sha({key: value for key, value in observed.items() if key != 'result_sha256'}),
            'main result semantic hash differs')
    gate_path = results/'complete_gate.json'
    gate = json_read(gate_path)
    gate_identity(gate, plan, lock)
    gate_sha = sha(gate_path)
    require(observed['complete_gate_snapshot'] == dict(path='complete_gate.json', sha256=gate_sha)
            and observed['inputs']['complete_gate_sha256'] == gate_sha, 'score gate snapshot differs')
    pinned = {str(results/'results.json'): sha(results/'results.json'), str(gate_path): gate_sha}
    score_inputs = {'COMPLETE_GATE.json': gate_sha}
    for name in ('SOURCE_LOCK.json', 'plan_snapshot.json', 'ledger.jsonl'):
        path = phase/name
        pinned[str(path)] = score_inputs[name] = sha(path)
    require(score_inputs['ledger.jsonl'] == gate['ledger_sha256'], 'ledger is not the complete-gate snapshot')
    # A later harmless re-audit may refresh the gate timestamp. Compare the
    # full closure identity and retain the original scoring gate snapshot.
    live_gate = json_read(phase/'COMPLETE_GATE.json')
    gate_identity(live_gate, plan, lock)
    for field in ('done_sha256', 'artifacts_sha256', 'ledger_sha256'):
        require(live_gate[field] == gate[field], 'current phase closure differs from scoring closure')
    pinned[str(phase/'COMPLETE_GATE.json')] = sha(phase/'COMPLETE_GATE.json')
    require({p.name for p in (phase/'units').iterdir() if p.is_dir()} == {u['unit_id'] for u in units},
            'phase unit directories do not close exactly')
    paths, receipt_selected, checkpoint_pins = {}, {}, {}
    # Establish every commitment and prediction/label inventory BEFORE any
    # outer argmax, CE, selection, effect or scientific statistic is computed.
    for unit in units:
        folder = phase/'units'/unit['unit_id']
        done_bytes = (folder/'DONE').read_bytes()
        done_sha = hashlib.sha256(done_bytes).hexdigest()
        require(done_sha == gate['done_sha256'][unit['unit_id']], 'DONE differs from complete gate')
        done = json.loads(done_bytes)
        require(done['schema'] == 'ser-final-program-done-1' and done['unit_id'] == unit['unit_id']
                and done['unit_sha256'] == semantic_sha(unit) and done['lock_sha256'] == lock['lock_sha256'], 'DONE identity differs')
        require(re.fullmatch(r'attempts/\d{4}', done['attempt']) is not None, 'unsafe attempt path')
        expected = {done['attempt']+'/'+name for name in ('predictions.npz', 'checkpoint.pt', 'history.json', 'receipt.json')}
        require(set(done['artifacts']) == expected and done['artifacts'] == gate['artifacts_sha256'][unit['unit_id']],
                'artifact inventory differs from gate')
        pinned[str(folder/'DONE')] = score_inputs[(folder/'DONE').relative_to(phase).as_posix()] = done_sha
        for relative, expected_sha in done['artifacts'].items():
            path = (folder/relative).resolve()
            require(path.is_relative_to(folder.resolve()) and re.fullmatch('[0-9a-f]{64}', expected_sha) is not None,
                    'unsafe artifact identity')
            if path.name == 'checkpoint.pt':
                checkpoint_pins[unit['unit_id']] = {'path': str(path), 'gate_sha256': expected_sha,
                    'exists_at_audit': path.is_file(), 'bytes_at_audit': path.stat().st_size if path.is_file() else None,
                    'rehash_performed': False}
                continue
            require(sha(path) == expected_sha, 'small artifact differs from gate: '+relative)
            pinned[str(path)] = expected_sha
            if path.name in ('predictions.npz', 'receipt.json'):
                score_inputs[path.relative_to(phase).as_posix()] = expected_sha
        attempt = folder/done['attempt']
        receipt = json_read(attempt/'receipt.json')
        require(receipt['unit_id'] == unit['unit_id'] and receipt['unit_sha256'] == semantic_sha(unit)
                and receipt['lock_sha256'] == lock['lock_sha256'] and receipt['phase'] == 'formal', 'receipt identity differs')
        receipt_selected[unit['unit_id']] = receipt['info']['selected_epochs']
        paths[unit['unit_id']] = attempt/'predictions.npz'
        with np.load(paths[unit['unit_id']], allow_pickle=False) as arrays:
            verify_arrays(unit, arrays, plan['rows'][unit['corpus']])
    score_rel = 'v3/final_program_20260907/score.py'
    require(score_rel in lock['sources'] and observed['inputs']['score_source_sha256'] == lock['sources'][score_rel],
            'score implementation is not the frozen source')
    recorded_phase = observed['inputs']['run_dir']
    relocated = recorded_phase != str(phase)
    allow_relocated = bool(getattr(args, 'allow_relocated_inputs', False))
    require(isinstance(recorded_phase, str) and '\x00' not in recorded_phase
            and (PurePosixPath(recorded_phase).is_absolute() or PureWindowsPath(recorded_phase).is_absolute()),
            'recorded score run directory must be absolute metadata')
    require(not relocated or allow_relocated, 'score run directory differs; explicit --allow-relocated-inputs required')
    # Only the sealed score's original run_dir string is retained as metadata.
    # Every actual file is read below the caller's current phase and all
    # relative inventories, original byte commitments and values stay exact.
    expected_inputs = dict(run_dir=recorded_phase, lock_sha256=lock['lock_sha256'], complete_gate_sha256=gate_sha,
                           score_source_sha256=lock['sources'][score_rel], byte_sha256=score_inputs, complete_gate_scope=gate['scope'])
    require(observed['inputs'] == expected_inputs, 'main score input closure differs')
    require(set(observed['tables']) == {name+'.csv' for name in TABLE_COUNTS}, 'main table inventory differs')
    for name, count in TABLE_COUNTS.items():
        path = results/(name+'.csv')
        require(observed['tables'][path.name] == dict(rows=count, sha256=sha(path)), 'table hash/count declaration differs')
        pinned[str(path)] = sha(path)
    def loader(unit):
        with np.load(paths[unit['unit_id']], allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    reference, tables, selected = recompute(plan, loader)
    comparator = Comparator()
    comparator.check(selected, receipt_selected, 'receipt.selected_epochs')
    table_audits = {name+'.csv': comparator.csv(results/(name+'.csv'), rows) for name, rows in tables.items()}
    reference.update(inputs=expected_inputs, complete_gate_snapshot=dict(path='complete_gate.json', sha256=gate_sha),
                     tables=observed['tables'])
    # Hash text is validated above, rather than expecting independently rounded
    # t arithmetic to produce byte-identical JSON floating point encodings.
    comparator.check(reference, {key: value for key, value in observed.items() if key != 'result_sha256'}, 'results.json')
    for path, expected_sha in pinned.items():
        require(sha(path) == expected_sha, 'audit input changed during computation: '+path)
    again, again_lock = runner.checked_plan(repo, phase)
    require(again['plan_sha256'] == plan['plan_sha256'] and again_lock['lock_sha256'] == lock['lock_sha256']
            and sha(__file__) == operator_sha, 'frozen or audit source changed during computation')
    result = dict(schema='ser-final-program-independent-numeric-audit-1',
        passed=comparator.mismatch_count == 0, created_at=created, completed_at=datetime.now(timezone.utc).isoformat(),
        phase='formal', formal_units=384, plan_sha256=plan['plan_sha256'], lock_sha256=lock['lock_sha256'],
        source_commit=lock['source_commit'], frozen_sources=lock['sources'], audit_source_sha256=operator_sha,
        source_and_input_sha256=pinned, score_gate_sha256=gate_sha,
        main_result_sha256=pinned[str(results/'results.json')], main_result_semantic_sha256=observed['result_sha256'],
        relocation=dict(explicitly_allowed=allow_relocated, applied=relocated,
                        recorded_run_dir=recorded_phase, actual_run_dir=str(phase),
                        scope='Only the original absolute inputs.run_dir is metadata; no sealed byte or SHA is rewritten or ignored'),
        numeric_values_checked=comparator.numeric_values_checked, max_abs_error=comparator.max_abs_error,
        tolerance=TOLERANCE, mismatches=comparator.mismatch_count, mismatch_examples=comparator.examples,
        csv=table_audits, max_abs_error_by_csv_column=comparator.column_errors,
        tests_checked=[dict(corpus=row['corpus'], estimand=row['estimand']) for row in reference['tests']],
        checkpoint_gate_pins_not_rehashed=checkpoint_pins,
        wall_seconds=time.perf_counter()-began, new_fits=0, model_inference_performed=False,
        scope='All 384 formal prediction commitments/paths/labels checked before independent numerical scoring; '
              'five CSVs and all result JSON values compared. Source/plan checking shares the frozen runner; '
              'statistics do not import the main scorer or engine metrics. Complete gate binds historical '
              'checkpoint identities, but current checkpoint bytes are not rehashed or deserialized; '
              'this is numerical replay, not a new full-weight gate or independent training reproduction.')
    result['pass'] = result.pop('passed')
    result['audit_sha256'] = semantic_sha(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'run-dir', 'results', 'out-json'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--allow-relocated-inputs', action='store_true',
                        help='Treat only the sealed inputs.run_dir absolute path as original-location metadata; all byte/SHA checks remain mandatory')
    args = parser.parse_args()
    try:
        result = execute(args)
        print(json.dumps({key: result[key] for key in ('pass', 'formal_units', 'numeric_values_checked', 'max_abs_error', 'mismatches')}))
        return 0 if result['pass'] else 1
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, error=str(error))))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Synthetic-only numeric and archive-contract tests; never import score.py.

Archive fixtures use an explicitly synthetic source/full-gate stub and tiny
non-model checkpoint bytes. They test closure/replay mechanics, not real gates.
Known recall/effect/t/Holm answers below separately test numerical correctness.
"""
from copy import deepcopy
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import ttest_1samp

from . import independent_numeric_audit as audit


def synthetic_plan():
    plan = dict(program=audit.PROGRAM, analysis={'fixture': 'SYNTHETIC_NO_EXECUTION_GATE'}, rows={}, units=[])
    for corpus, classes in audit.CLASSES.items():
        report, rows = {}, {}
        for role in audit.ROLES:
            counts = [1+c%3 for c in range(classes)] if role == 'outer' else [2]*classes
            labels = np.repeat(np.arange(classes), counts)
            report[role] = [f'SYNTHETIC/{corpus}/{role}/{i}' for i in range(len(labels))]
            rows.update({p: {'label_index': int(y)} for p, y in zip(report[role], labels)})
        plan['rows'][corpus] = rows
        for draw in range(24):
            for fold in range(5):
                arms = ('A', 'B') if corpus == 'cremad' and fold == 0 else ('A',)
                for arm in arms:
                    plan['units'].append(dict(unit_id=f'{corpus}_{draw}_{fold}_{arm}', corpus=corpus,
                        phase='formal', draw=draw, fold=fold, arm=arm, n_classes=classes,
                        pair_id=f'{corpus}_{draw}_{fold}', seeds={'synthetic': draw*5+fold},
                        report=deepcopy(report), report_batches={role: [paths] for role, paths in report.items()},
                        prediction_epochs=list(range(1, 16)) if arm == 'A' else [15], selection_enabled=arm == 'A'))
    pilot = deepcopy(plan['units'][0]); pilot.update(phase='pilot', unit_id='FORBIDDEN_PILOT')
    plan['units'].append(pilot)
    plan['plan_sha256'] = audit.semantic_sha(plan)
    return plan


def make_logits(y, correct, classes, margin):
    pred = (y+1) % classes
    pred[:correct] = y[:correct]
    z = np.zeros((len(y), classes), np.float32)
    z[np.arange(len(y)), pred] = margin
    return z


def arrays_for(unit, plan):
    assert unit['phase'] == 'formal', 'pilot must never be loaded'
    arrays = {'epochs': np.asarray(unit['prediction_epochs'], dtype=np.int64)}
    for role, paths in unit['report'].items():
        labels = np.asarray([plan['rows'][unit['corpus']][p]['label_index'] for p in paths], dtype=np.int64)
        values = []
        for epoch in unit['prediction_epochs']:
            if unit['arm'] == 'B':
                count = len(labels)//3 if role == 'A' else 2*len(labels)//3
                margin = 2.
            elif role == 'outer':
                count = (epoch+2*unit['draw']+unit['fold']) % (len(labels)+1)
                margin = 2.
            else:
                high = (1, 6) if role == 'A' else (3, 7)
                ce_best = (2, 9) if role == 'A' else (4, 10)
                count, margin = (len(labels), .02) if epoch in high else ((len(labels)-1, 4.) if epoch in ce_best else (0, 8.))
            values.append(make_logits(labels, count, unit['n_classes'], margin))
        arrays[role+'__paths'] = np.asarray(paths)
        arrays[role+'__labels'] = labels
        arrays[role+'__all_epoch_logits'] = np.stack(values)
    return arrays


def test_native_macro_recall_and_ce_known_answers():
    labels = np.asarray([0, 1, 1, 1], dtype=np.int64)
    logits = np.asarray([[3, 0]]*4, dtype=np.float32)
    ce, recall = audit.native_metric(labels, logits)
    assert recall == Fraction(1, 2) and recall != Fraction(1, 4)
    assert abs(ce - (math.log1p(math.exp(-3))+2.25)) < 1e-12
    for classes in (6, 7, 8):
        y = np.arange(classes, dtype=np.int64)
        ce, recall = audit.native_metric(y, np.zeros((classes, classes), np.float32))
        assert recall == Fraction(1, classes)
        assert abs(ce-math.log(classes)) < 1e-12


def test_four_independent_winners_and_earliest_ties():
    plan = synthetic_plan(); unit = plan['units'][0]
    value = audit.read_trajectory(unit, arrays_for(unit, plan), plan['rows'][unit['corpus']])
    assert value['selected'] == dict(last=15, seen_ce=2, seen_uar=1, unseen_ce=4, unseen_uar=3)
    b = next(unit for unit in plan['units'] if unit['arm'] == 'B')
    assert audit.read_trajectory(b, arrays_for(b, plan), plan['rows']['cremad'])['selected'] == {'last': 15}


def test_counts_known_context_effect_and_control():
    plan = synthetic_plan()
    result, tables, selected = audit.recompute(plan, lambda unit: arrays_for(unit, plan))
    assert {name: len(rows) for name, rows in tables.items()} == audit.TABLE_COUNTS
    first = tables['contexts'][0]
    assert first['delta_CE_pp'] == float(Fraction(-125, 9))
    assert first['delta_UAR_pp'] == float(Fraction(-50, 3))
    assert first['J_pp'] == float(Fraction(25, 9))
    assert all(row['E_pp'] == float(Fraction(50, 3)) for row in tables['controls'])
    assert result['counts']['pilot_scored'] == 0 and len(selected) == 384
    assert len(result['tests']) == 6 and result['control']['inference'] is False


def test_t_and_holm_independent_known_references():
    values = [float(i*i % 17)-5 for i in range(24)]
    result = audit.independent_t(values)
    reference = ttest_1samp(values, 0.)
    assert abs(result['t_statistic']-float(reference.statistic)) < 1e-12
    assert abs(result['p_two_sided']-float(reference.pvalue)) < 1e-12
    assert result['df'] == 23
    tests = [{'p_two_sided': p} for p in (.03, .001, .9, .015, .01, .04)]
    audit.adjust_six(tests)
    np.testing.assert_allclose([t['holm_p_six'] for t in tests], [.09, .006, .9, .06, .05, .09], atol=1e-15)
    for value in (0., 2.1):
        assert audit.independent_t([value]*24)['p_two_sided'] is None


def test_exact_zero_interaction_is_not_float_residual(monkeypatch):
    plan = synthetic_plan()
    quartets = [(Fraction(1, 3), Fraction(2, 3), Fraction(1, 6), Fraction(1, 2)),
                (Fraction(5, 6), Fraction(1, 2), Fraction(2, 3), Fraction(1, 3))]
    def synthetic_trajectory(unit, unused_arrays, unused_rows):
        a, b, c, d = quartets[unit['draw'] % 2]
        outer = [a, b, c, d]+[Fraction(1, 2)]*11
        values = {'outer': {e: (1., outer[e-1]) for e in unit['prediction_epochs']},
                  'A': {e: (1., a if unit['arm'] == 'A' else b) for e in unit['prediction_epochs']},
                  'B': {e: (1., c if unit['arm'] == 'A' else d) for e in unit['prediction_epochs']}}
        selected = {'last': 15}
        if unit['arm'] == 'A':
            selected.update(seen_ce=1, unseen_ce=2, seen_uar=3, unseen_uar=4)
        return dict(unit=unit, values=values, selected=selected)
    monkeypatch.setattr(audit, 'read_trajectory', synthetic_trajectory)
    result, tables, _ = audit.recompute(plan, lambda unit: None)
    assert all(row['J_pp'] == 0. for row in tables['contexts']+tables['draws'])
    assert all(row['E_pp'] == 0. for row in tables['controls'])
    assert all(row['p_two_sided'] is None and not row['reject_familywise_05'] for row in result['tests'] if row['estimand'] == 'J')


def test_invalid_plan_array_and_comparison_contracts():
    plan = synthetic_plan(); unit = plan['units'][0]
    bad_plan = deepcopy(plan); bad_plan['units'].pop(0)
    with pytest.raises(ValueError, match='384|360 main'): audit.formal_units(bad_plan)
    for field in ('paths', 'labels', 'epoch'):
        data = arrays_for(unit, plan)
        if field == 'paths': data['A__paths'][0] = 'BAD'
        elif field == 'labels': data['A__labels'][0] = 1
        else: data['epochs'][0] = 0
        with pytest.raises(ValueError): audit.verify_arrays(unit, data, plan['rows']['cremad'])
    comparator = audit.Comparator()
    comparator.check({'x': 1.}, {'x': 1.+2e-9}, 'fixture')
    comparator.check(False, True, 'boolean')
    assert comparator.mismatch_count == 2


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def seal_result(path, value):
    value.pop('result_sha256', None)
    value['result_sha256'] = audit.semantic_sha(value)
    write_json(path, value)


@pytest.fixture(scope='module')
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_numeric_closure')
    phase, results = root/'formal', root/'scores'
    phase.mkdir(); results.mkdir()
    plan = synthetic_plan()
    lock = dict(phase='formal', plan_sha256=plan['plan_sha256'], source_commit='SYNTHETIC_NO_REAL_FREEZE',
                sources={'v3/final_program_20260907/score.py': '0'*64})
    lock['lock_sha256'] = audit.semantic_sha(lock)
    write_json(phase/'SOURCE_LOCK.json', lock); write_json(phase/'plan_snapshot.json', plan)
    (phase/'ledger.jsonl').write_bytes(b'{"synthetic_gate_stub":true}\n')
    gate = dict(schema='ser-final-program-complete-gate-1', phase='formal', expected_units=384, verified_units=384,
                plan_sha256=plan['plan_sha256'], lock_sha256=lock['lock_sha256'], done_sha256={}, artifacts_sha256={},
                ledger_sha256=audit.sha(phase/'ledger.jsonl'), scope='SYNTHETIC GATE STUB; NO REAL TRAINING')
    gate['pass'] = True
    reference, tables, selected = audit.recompute(plan, lambda unit: arrays_for(unit, plan))
    input_hashes = {name: audit.sha(phase/name) for name in ('SOURCE_LOCK.json', 'plan_snapshot.json', 'ledger.jsonl')}
    for unit in audit.formal_units(plan):
        folder = phase/'units'/unit['unit_id']; attempt = folder/'attempts/0001'; attempt.mkdir(parents=True)
        np.savez_compressed(attempt/'predictions.npz', **arrays_for(unit, plan))
        (attempt/'checkpoint.pt').write_bytes(b'NOT_A_MODEL_TESTS_MUST_NOT_DESERIALIZE')
        write_json(attempt/'history.json', [])
        receipt = dict(unit_id=unit['unit_id'], unit_sha256=audit.semantic_sha(unit), phase='formal',
                       lock_sha256=lock['lock_sha256'], info={'selected_epochs': selected[unit['unit_id']]})
        write_json(attempt/'receipt.json', receipt)
        artifacts = {'attempts/0001/'+path.name: audit.sha(path) for path in attempt.iterdir()}
        done = dict(schema='ser-final-program-done-1', unit_id=unit['unit_id'], unit_sha256=audit.semantic_sha(unit),
                    lock_sha256=lock['lock_sha256'], attempt='attempts/0001', artifacts=artifacts)
        write_json(folder/'DONE', done)
        gate['done_sha256'][unit['unit_id']] = audit.sha(folder/'DONE')
        gate['artifacts_sha256'][unit['unit_id']] = artifacts
        for path in (folder/'DONE', attempt/'predictions.npz', attempt/'receipt.json'):
            input_hashes[path.relative_to(phase).as_posix()] = audit.sha(path)
    write_json(phase/'COMPLETE_GATE.json', gate)
    shutil.copyfile(phase/'COMPLETE_GATE.json', results/'complete_gate.json')
    input_hashes['COMPLETE_GATE.json'] = audit.sha(results/'complete_gate.json')
    reference['inputs'] = dict(run_dir=str(phase.resolve()), lock_sha256=lock['lock_sha256'],
        complete_gate_sha256=input_hashes['COMPLETE_GATE.json'], score_source_sha256='0'*64,
        byte_sha256=input_hashes, complete_gate_scope=gate['scope'])
    reference['complete_gate_snapshot'] = dict(path='complete_gate.json', sha256=input_hashes['COMPLETE_GATE.json'])
    reference['tables'] = {}
    for name, rows in tables.items():
        path = results/(name+'.csv')
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        reference['tables'][path.name] = dict(rows=len(rows), sha256=audit.sha(path))
    seal_result(results/'results.json', reference)
    return root


@pytest.fixture
def packet(template, tmp_path, monkeypatch):
    clone = tmp_path/'packet'; shutil.copytree(template, clone)
    phase, results = clone/'formal', clone/'scores'
    result = audit.json_read(results/'results.json'); result['inputs']['run_dir'] = str(phase.resolve())
    seal_result(results/'results.json', result)
    plan, lock = audit.json_read(phase/'plan_snapshot.json'), audit.json_read(phase/'SOURCE_LOCK.json')
    def checked(repo, requested_phase):
        assert requested_phase == phase
        assert audit.json_read(phase/'SOURCE_LOCK.json') == lock
        assert audit.json_read(phase/'plan_snapshot.json') == plan
        return plan, lock
    monkeypatch.setattr(audit, 'checked_runner', lambda repo: SimpleNamespace(checked_plan=checked))
    return SimpleNamespace(repo=Path.cwd(), run_dir=phase, results=results, out_json=clone/'audit.json')


def test_full_synthetic_closure_roundtrip_and_protected_output(packet):
    result = audit.execute(packet)
    assert result['pass'] and result['formal_units'] == 384 and result['mismatches'] == 0
    assert result['max_abs_error'] == 0 and result['numeric_values_checked'] > 50000
    assert len(result['csv']) == 5 and len(result['tests_checked']) == 6
    assert all(not value['rehash_performed'] for value in result['checkpoint_gate_pins_not_rehashed'].values())
    old_bytes = packet.out_json.read_bytes()
    with pytest.raises(ValueError, match='fresh'): audit.execute(packet)
    assert packet.out_json.read_bytes() == old_bytes


def test_partial_gate_rejected_before_prediction_loading(packet, monkeypatch):
    gate = audit.json_read(packet.results/'complete_gate.json'); gate['verified_units'] = 383
    write_json(packet.results/'complete_gate.json', gate)
    def forbidden(*args, **kwargs): raise AssertionError('predictions must not be loaded')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='complete formal 384'): audit.execute(packet)
    assert not packet.out_json.exists()


def test_corrupt_last_prediction_rejected_before_any_numerical_scoring(packet, monkeypatch):
    target = sorted((packet.run_dir/'units').iterdir())[-1]/'attempts/0001/predictions.npz'
    target.write_bytes(target.read_bytes()+b'corrupt')
    def forbidden(*args, **kwargs): raise AssertionError('numeric recomputation must not start')
    monkeypatch.setattr(audit, 'recompute', forbidden)
    with pytest.raises(ValueError, match='small artifact differs'): audit.execute(packet)
    assert not packet.out_json.exists()


def test_rehashed_wrong_csv_number_is_detected(packet):
    path = packet.results/'contexts.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream); fields, rows = reader.fieldnames, list(reader)
    rows[0]['J_pp'] = str(float(rows[0]['J_pp'])+.1)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    doc = audit.json_read(packet.results/'results.json'); doc['tables']['contexts.csv']['sha256'] = audit.sha(path)
    seal_result(packet.results/'results.json', doc)
    result = audit.execute(packet)
    assert not result['pass'] and result['mismatches'] == 1
    assert result['max_abs_error'] > .09


def test_rehashed_wrong_json_effect_is_detected(packet):
    doc = audit.json_read(packet.results/'results.json'); doc['tests'][0]['mean_pp'] += .25
    seal_result(packet.results/'results.json', doc)
    result = audit.execute(packet)
    assert not result['pass'] and result['mismatches'] == 1 and result['max_abs_error'] == .25


def test_output_cannot_enter_phase_or_score_inputs(packet):
    for output in (packet.run_dir/'new.json', packet.run_dir/'units/x/a.json', packet.results/'new.json'):
        packet.out_json = output
        with pytest.raises(ValueError, match='outside inputs/results'): audit.execute(packet)
        assert not output.exists()

"""Known-answer synthetic arrays and report-tampering cases; never imports score."""
import ast
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from v3.inner_validation.audit_tools import replay


def synthetic():
    units = []
    for d in range(24):
        for f in range(5):
            panel = {role: [f'SYNTHETIC/{d}/{f}/{role}/{i}' for i in range(n)]
                     for role, n in (('fit', 576), ('val_seen', 288), ('val_unseen', 288), ('test', 36*(f+1)))}
            for c in range(4):
                units.append({'unit_id': f'dual_d{d:02d}_f{f}_c{c}', 'draw': d, 'fold': f,
                    'config_index': c, 'train_seed': 200+d*5+f,
                    'config': {'lr_encoder': (1e-5, 5e-5)[c//2], 'lr_head': (3e-4, 1e-3)[c%2]}, **panel})
    plan = {'plan_sha256': 'a'*64, 'units': units, 'pilot_units': [u['unit_id'] for u in units[:4]],
            'analysis': {'draws': 24, 'fixed_config_index': 3, 'bootstrap_repetitions': 50000, 'bootstrap_seed': 2026090701}}
    def loader(unit):
        d, f, c = unit['draw'], unit['fold'], unit['config_index']
        arrays = {}
        for role in replay.ROLE_NAMES:
            support = 6*(f+1) if role == 'test' else 48
            labels = np.repeat(np.arange(6, dtype=np.int64), support)
            arrays[role+'__paths'] = np.asarray(unit[role])
            arrays[role+'__labels'] = labels
            if role == 'val_seen':
                counts = ([36, 20, 36, 32][c]+d%3, [24, 26, 28, 30][c], 20+d%4)
            elif role == 'val_unseen':
                counts = ([22, 24, 26, 28][c], [30, 42, 28, 34][c]+f%2, 18+f%3)
            else:
                seen = (support if f == 0 else 0) if c == 0 else support
                unseen = support//2 if c == 1 else support//3
                counts = (seen, unseen, support//3)
            for checkpoint, correct in zip(replay.STATE_NAMES, counts):
                guessed = (labels+1)%6
                for emotion in range(6):
                    guessed[emotion*support:emotion*support+correct] = emotion
                logits = np.zeros((len(labels), 6), dtype=np.float64)
                logits[np.arange(len(labels)), guessed] = 1
                arrays[f'{role}__{checkpoint}__logits'] = logits
        return {'predictions': arrays, 'receipt': {'best_seen_epoch': 3, 'best_unseen_epoch': 7, 'epochs_run': 15}}
    return plan, loader


@pytest.fixture(scope='module')
def rebuilt():
    plan, loader = synthetic()
    return replay.reconstruct(plan, loader)


def test_full_grid_has_analytically_known_primary_and_selection_increment(rebuilt):
    assert [len(rebuilt[name]) for name in replay.TABLES] == [4320, 240, 24]
    result = rebuilt['results.json']
    assert result['primary']['estimate_pp'] == pytest.approx(18.75)
    assert result['descriptive_draw_summaries']['fixed_config_last_delta_gap_pp']['mean'] == pytest.approx(5.625)
    assert result['descriptive_draw_summaries']['selection_increment_pp']['mean'] == pytest.approx(13.125)
    assert result['primary']['df'] == 23 and result['primary_hypothesis_tests'] == 1


def test_own_validation_winners_ties_and_equal_fold_weights(rebuilt):
    for episode in rebuilt['selected_episodes.csv']:
        expected = 0 if episode['rule'] == 'seen' else 1
        assert episode['selected_config_index'] == expected
        assert episode['primary_epoch'] == (3 if expected == 0 else 7)
    for draw in rebuilt['draws.csv']:
        # Fold 0 is perfect and has the fewest clips; other four folds are wrong.
        assert draw['primary_seen_test_uar_percent'] == 20
        assert draw['primary_seen_test_uar_percent'] != pytest.approx(100/15)
        assert draw['primary_unseen_test_uar_percent'] == 50
        assert draw['fixed_config_last_delta_test_pp'] == 0
        assert draw['fixed_config_last_delta_gap_pp'] == draw['fixed_config_last_delta_validation_pp']
        assert draw['primary_delta_gap_pp'] == pytest.approx(draw['primary_delta_validation_pp']-draw['primary_delta_test_pp'])


def test_confusion_matrix_counts_and_macro_not_micro():
    truth = np.asarray([0]*9+[1, 2, 3, 4, 5], dtype=np.int64)
    logits = np.zeros((14, 6)); logits[:, 0] = 1
    value, matrix = replay.confusion_uar(truth, logits)
    assert matrix.shape == (6, 6) and matrix[0, 0] == 9 and matrix[:, 0].sum() == 14
    assert value == pytest.approx(100/6) and value != pytest.approx(100*9/14)
    with pytest.raises(ValueError, match='six classes'):
        replay.confusion_uar(truth[:-1], logits[:-1])
    logits[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite logits'):
        replay.confusion_uar(truth, logits)


def test_beta_student_inference_matches_independent_scipy_reference():
    values = np.linspace(-2, 3, 24)
    result = replay.recompute_inference(values)
    reference = stats.ttest_1samp(values, 0)
    assert result['t_statistic'] == pytest.approx(reference.statistic, abs=1e-12)
    assert result['p_value_two_sided'] == pytest.approx(reference.pvalue, abs=1e-12)
    assert result['t_ci95_pp'] == pytest.approx(reference.confidence_interval(.95), abs=1e-10)


def test_weight_bootstrap_uses_the_same_50000_resamples_without_indexed_value_aggregation():
    effects = np.linspace(-17, 29, 24)**3/1000
    rng = np.random.Generator(np.random.PCG64(2026090701))
    samples = effects[rng.integers(0, 24, size=(50000, 24))].mean(axis=1)
    reference = np.quantile(samples, [.025, .975], method='linear')
    assert replay.bootstrap_by_weights(effects) == pytest.approx(reference, abs=1e-12)


def test_degenerate_primary_is_explicit_and_json_finite():
    for value in (0.0, 1.0):
        result = replay.recompute_inference([value]*24)
        assert result['t_test_defined'] is False and result['t_statistic'] is None and result['p_value_two_sided'] is None
        assert result['t_ci95_pp'] == [value, value]
        assert result['bootstrap_sensitivity']['ci95_pp'] == [value, value]
        json.dumps(result, allow_nan=False)


def write_table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def test_three_complete_csv_and_json_contracts_are_all_compared(tmp_path, rebuilt):
    comparison = replay.Comparison()
    key_map = {'pred_metrics.csv': ('unit_id', 'role', 'checkpoint'),
               'selected_episodes.csv': ('draw', 'fold', 'rule'), 'draws.csv': ('draw',)}
    for name in replay.TABLES:
        write_table(tmp_path/name, rebuilt[name])
        audit = replay.compare_csv(tmp_path/name, rebuilt[name], key_map[name], comparison)
        assert audit['rows'] == len(rebuilt[name]) and audit['columns'] == list(rebuilt[name][0])
        assert sum(audit['numeric_values_checked_by_column'].values()) == audit['numeric_values_checked']
    result_file = tmp_path/'results.json'
    result_file.write_text(json.dumps(rebuilt['results.json'], allow_nan=False), encoding='utf-8')
    comparison.check(rebuilt['results.json'], replay.read_json(result_file), 'results.json')
    assert comparison.numeric_values_checked > 30000 and comparison.maximum_absolute_error == 0


def test_csv_changed_metric_and_duplicate_row_are_rejected(tmp_path, rebuilt):
    wanted = rebuilt['pred_metrics.csv']
    changed = copy.deepcopy(wanted)
    changed[0]['uar_percent'] += 1
    path = tmp_path/'pred_metrics.csv'; write_table(path, changed)
    with pytest.raises(ValueError, match='numeric value differs'):
        replay.compare_csv(path, wanted, ('unit_id', 'role', 'checkpoint'), replay.Comparison())
    changed[0] = changed[1]
    write_table(path, changed)
    with pytest.raises(ValueError, match='duplicate/unexpected CSV key'):
        replay.compare_csv(path, wanted, ('unit_id', 'role', 'checkpoint'), replay.Comparison())


def test_json_numeric_and_unexpected_fields_are_rejected(rebuilt):
    expected = rebuilt['results.json']
    changed = copy.deepcopy(expected); changed['primary']['estimate_pp'] += .01
    with pytest.raises(ValueError, match='numeric value differs'):
        replay.Comparison().check(expected, changed, 'results.json')
    changed = copy.deepcopy(expected); changed['new_hypothesis'] = 1
    with pytest.raises(ValueError, match='field inventory'):
        replay.Comparison().check(expected, changed, 'results.json')


def test_partial_grid_and_misaligned_predictions_are_rejected():
    plan, loader = synthetic()
    plan['units'].pop()
    with pytest.raises(ValueError, match='480'):
        replay.reconstruct(plan, lambda u: pytest.fail('partial data read'))
    plan, loader = synthetic()
    def broken(unit):
        batch = loader(unit); batch['predictions']['test__paths'] = batch['predictions']['test__paths'][::-1]
        return batch
    with pytest.raises(ValueError, match='path alignment'):
        replay.reconstruct(plan, broken)


def test_pilot_or_incomplete_gate_is_not_accepted():
    plan, _ = synthetic()
    gate = {'schema': 'ser-dual-validation-result-gate-1', 'pass': True, 'phase': 'formal', 'units': 480,
            'plan_sha256': plan['plan_sha256'], 'scores_computed': False,
            'done_sha256': {u['unit_id']: 'b'*64 for u in plan['units']}}
    gate['gate_sha256'] = replay.content_hash(gate)
    replay.accepted_gate(plan, gate)
    pilot = copy.deepcopy(gate); pilot['phase'] = 'pilot'; pilot['units'] = 4
    with pytest.raises(ValueError, match='480-unit formal gate'):
        replay.accepted_gate(plan, pilot)
    partial = copy.deepcopy(gate); partial['done_sha256'].pop(plan['units'][0]['unit_id'])
    partial['gate_sha256'] = replay.content_hash({k:v for k,v in partial.items() if k!='gate_sha256'})
    with pytest.raises(ValueError, match='DONE set'):
        replay.accepted_gate(plan, partial)


def test_output_cannot_be_inside_main_results_before_any_integrity_import(tmp_path):
    results = tmp_path/'main'; results.mkdir()
    with pytest.raises(ValueError, match='overlaps immutable inputs'):
        replay.audit(tmp_path/'repo', tmp_path/'plan.json', tmp_path/'runs', tmp_path/'gate.json', results, results/'audit.json')


def test_replay_has_no_main_scoring_module_import():
    tree = ast.parse(Path(replay.__file__).read_text(encoding='utf-8'))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom): imported.append(node.module or '')
        if isinstance(node, ast.Import): imported.extend(x.name for x in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
            imported.append(ast.literal_eval(node.args[0]))
    assert not any(name.endswith('.score') for name in imported)

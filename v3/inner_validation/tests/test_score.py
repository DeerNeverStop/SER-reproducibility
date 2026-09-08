"""Synthetic predictions only; no models, real corpus, or GPU are loaded."""
import copy

import numpy as np
import pytest
from scipy import stats

from v3.inner_validation import score


def fixture_inputs():
    units = []
    for draw in range(24):
        for fold in range(5):
            paths = {role: [f'SYNTHETIC/d{draw}/f{fold}/{role}/{i:04d}' for i in range(n)]
                     for role, n in (('fit', 576), ('val_seen', 288), ('val_unseen', 288), ('test', 216))}
            for config in range(4):
                units.append(dict(unit_id=f'dual_d{draw:02d}_f{fold}_c{config}', draw=draw, fold=fold,
                    config_index=config, train_seed=1000+draw*5+fold,
                    config={'lr_encoder': (1e-5, 5e-5)[config//2], 'lr_head': (3e-4, 1e-3)[config%2]}, **paths))
    plan = {'plan_sha256': 'a'*64, 'units': units,
            'pilot_units': [u['unit_id'] for u in units[:4]],
            'analysis': {'draws': 24, 'fixed_config_index': 3,
                         'bootstrap_repetitions': 50000, 'bootstrap_seed': 2026090701}}

    def loader(unit):
        draw, fold, config = unit['draw'], unit['fold'], unit['config_index']
        arrays = {}
        for role in score.ROLES:
            support = 36 if role == 'test' else 48
            labels = np.repeat(np.arange(6, dtype=np.int64), support)
            arrays[role+'__paths'] = np.asarray(unit[role], dtype='U')
            arrays[role+'__labels'] = labels
            if role == 'val_seen':
                counts = {'best_seen': [24, 36, 36, 30][config]+draw%3,
                          'best_unseen': [20, 25, 30, 28][config]+fold%2,
                          'last': [16, 20, 22, 24][config]+draw%3}
            elif role == 'val_unseen':
                counts = {'best_seen': [16, 20, 30, 24][config],
                          'best_unseen': [20, 24, 40, 30][config]+fold%2,
                          'last': [14, 18, 22, 20][config]+fold%3}
            else:
                # Config 3 has the best seen-rule test score, but must not be selected by it.
                counts = {'best_seen': [10, 18+draw%5, 22, 36][config],
                          'best_unseen': [12, 17, 24+fold%3, 30][config],
                          'last': [12, 14, 16, 20][config]+draw%4}
            for checkpoint, count in counts.items():
                prediction = (labels+1)%6
                for label in range(6):
                    prediction[label*support:label*support+count] = label
                logits = np.zeros((len(labels), 6), dtype=np.float64)
                logits[np.arange(len(labels)), prediction] = 1.0
                arrays[f'{role}__{checkpoint}__logits'] = logits
        return {'predictions': arrays,
                'receipt': {'best_seen_epoch': 5, 'best_unseen_epoch': 7, 'epochs_run': 15}}
    return plan, loader


@pytest.fixture(scope='module')
def analyzed():
    plan, loader = fixture_inputs()
    return score.analyze(plan, loader)


def test_complete_factorial_selection_uses_own_validation_not_test(analyzed):
    assert len(analyzed['pred_metrics']) == 480*3*3
    assert len(analyzed['selected_episodes']) == 24*5*2
    assert len(analyzed['draws']) == 24
    for row in analyzed['selected_episodes']:
        if row['rule'] == 'seen':
            assert row['selected_config_index'] == 1  # Exact UAR tie with config 2 selects the first index.
            assert row['primary_checkpoint'] == 'best_seen'
            assert row['primary_epoch'] == 5
        else:
            assert row['selected_config_index'] == 2
            assert row['primary_checkpoint'] == 'best_unseen'
            assert row['primary_epoch'] == 7
    assert analyzed['results']['excluded_pilot_runs'] == 4
    first = analyzed['selected_episodes'][0]
    assert first['primary_validation_uar_percent'] == 75
    assert first['primary_test_uar_percent'] == 50
    assert first['fixed_config_best_test_uar_percent'] == 100


def test_fold_averaging_and_fixed_last_cancellation(analyzed):
    first = analyzed['draws'][0]
    expected = 75-50-((40+41+40+41+40)/5/48*100-(24+25+26+24+25)/5/36*100)
    assert first['primary_delta_gap_pp'] == pytest.approx(expected)
    for row in analyzed['draws']:
        assert row['fixed_config_last_delta_test_pp'] == 0.0
        assert row['fixed_config_last_seen_test_uar_percent'] == row['fixed_config_last_unseen_test_uar_percent']
        assert row['fixed_config_last_delta_gap_pp'] == row['fixed_config_last_delta_validation_pp']
        assert row['selection_increment_pp'] == row['primary_delta_gap_pp']-row['fixed_config_last_delta_gap_pp']
        assert row['primary_delta_gap_pp'] == pytest.approx(row['primary_delta_validation_pp']-row['primary_delta_test_pp'])


def test_t_test_is_over_24_draws_and_bootstrap_is_fixed(analyzed):
    values = np.asarray([r['primary_delta_gap_pp'] for r in analyzed['draws']])
    reference = stats.ttest_1samp(values, 0)
    result = analyzed['results']['primary']
    assert result['df'] == 23 and result['draws'] == 24
    assert result['t_statistic'] == pytest.approx(reference.statistic)
    assert result['p_value_two_sided'] == pytest.approx(reference.pvalue)
    assert result['t_ci95_pp'] == pytest.approx(reference.confidence_interval(.95))
    bootstrap = result['bootstrap_sensitivity']
    assert bootstrap['seed'] == 2026090701 and bootstrap['repetitions'] == 50000
    rerun = score.primary_inference(values)
    assert rerun['bootstrap_sensitivity'] == bootstrap
    assert analyzed['results']['primary_hypothesis_tests'] == 1
    assert analyzed['results']['additional_control_hypothesis_tests'] == 0


def test_whole_fold_uar_is_macro_recall_not_micro_accuracy():
    labels = np.array([0]*9+[1, 2, 3, 4, 5], dtype=np.int64)
    logits = np.zeros((14, 6)); logits[:, 0] = 1
    assert score.whole_fold_uar(labels, logits) == pytest.approx(100/6)
    assert score.whole_fold_uar(labels, logits) != pytest.approx(9/14*100)
    with pytest.raises(ValueError, match='six acted classes'):
        score.whole_fold_uar(labels[:-1], logits[:-1])
    logits[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        score.whole_fold_uar(labels, logits)


def test_unequal_fold_support_does_not_turn_into_clip_pooling():
    plan, original_loader = fixture_inputs()
    for unit in plan['units']:
        support = 6*(unit['fold']+1)
        unit['test'] = [f"SYNTHETIC/d{unit['draw']}/f{unit['fold']}/test/{i:04d}" for i in range(6*support)]
    def loader(unit):
        value = original_loader(unit)
        labels = np.repeat(np.arange(6, dtype=np.int64), 6*(unit['fold']+1))
        arrays = value['predictions']
        arrays['test__paths'], arrays['test__labels'] = np.asarray(unit['test']), labels
        for checkpoint in score.CHECKPOINTS:
            prediction = labels if checkpoint == 'best_seen' and unit['fold'] == 0 else (labels+1)%6
            logits = np.zeros((len(labels), 6))
            logits[np.arange(len(labels)), prediction] = 1
            arrays['test__'+checkpoint+'__logits'] = logits
        return value
    report = score.analyze(plan, loader)
    # First fold is perfect, other four are wrong: equal-fold mean is 20%.
    # Pooling clips would incorrectly yield 100*(6/(6+12+18+24+30)) = 6.67%.
    assert report['draws'][0]['primary_seen_test_uar_percent'] == 20.0
    assert report['draws'][0]['primary_seen_test_uar_percent'] != pytest.approx(100/15)


def test_partial_or_changed_plan_is_rejected_before_prediction_loading():
    plan, _ = fixture_inputs()
    plan['units'].pop()
    with pytest.raises(ValueError, match='480'):
        score.analyze(plan, lambda unit: pytest.fail('partial plan read predictions'))
    plan, _ = fixture_inputs()
    plan['units'][1]['train_seed'] += 1
    with pytest.raises(ValueError, match='common data or training seed'):
        score.validate_analysis_plan(plan)
    plan, _ = fixture_inputs()
    plan['analysis']['bootstrap_seed'] += 1
    with pytest.raises(ValueError, match='settings changed'):
        score.validate_analysis_plan(plan)


def test_prediction_order_and_same_epoch_aliases_are_checked():
    plan, loader = fixture_inputs()
    def wrong_order(unit):
        value = loader(unit)
        value['predictions']['test__paths'] = value['predictions']['test__paths'][::-1]
        return value
    with pytest.raises(ValueError, match='ordering differ'):
        score.analyze(plan, wrong_order)
    def impossible_same_epoch(unit):
        value = loader(unit); value['receipt']['best_seen_epoch'] = 15
        return value
    with pytest.raises(ValueError, match='same saved epoch'):
        score.analyze(plan, impossible_same_epoch)


def test_gate_is_exact_formal_grid_and_cannot_be_a_pilot_gate():
    plan, _ = fixture_inputs()
    gate = dict(schema='ser-dual-validation-result-gate-1', phase='formal', units=480,
                plan_sha256=plan['plan_sha256'], scores_computed=False,
                done_sha256={u['unit_id']: 'b'*64 for u in plan['units']})
    gate['pass'] = True; gate['gate_sha256'] = score.digest(gate)
    score.validate_complete_gate(plan, gate)
    broken = copy.deepcopy(gate); broken['phase'] = 'pilot'
    with pytest.raises(ValueError, match='formal gate'):
        score.validate_complete_gate(plan, broken)
    broken = copy.deepcopy(gate); broken['done_sha256'].pop(plan['units'][0]['unit_id'])
    broken['gate_sha256'] = score.digest({k:v for k,v in broken.items() if k!='gate_sha256'})
    with pytest.raises(ValueError, match='exact formal DONE'):
        score.validate_complete_gate(plan, broken)


def test_degenerate_draws_do_not_emit_invented_p_values_or_nan():
    for value in (0.0, 1.0):
        result = score.primary_inference(np.full(24, value))
        assert result['t_test_defined'] is False
        assert result['t_statistic'] is result['p_value_two_sided'] is None
        assert result['t_ci95_pp'] == [value, value]
        assert result['bootstrap_sensitivity']['ci95_pp'] == [value, value]

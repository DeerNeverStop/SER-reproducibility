"""One prespecified draw-level test after all 480 formal trajectories are sealed.

The pure analyze function accepts a loader returning {'predictions': arrays,
'receipt': receipt}. Epochs are chosen by the runner's own-validation loss;
configuration selection here uses own-validation whole-fold six-class UAR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import t as student_t

DRAWS, FOLDS, CONFIGS, FORMAL_UNITS = 24, 5, 4, 480
ROLES = ('val_seen', 'val_unseen', 'test')
CHECKPOINTS = ('best_seen', 'best_unseen', 'last')
RULES = ('seen', 'unseen')
FAMILIES = ('primary', 'fixed_config_best', 'fixed_config_last')
FIXED_CONFIG_INDEX = 3
BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED = 50_000, 2026090701
IDENTITY_TOLERANCE = 1e-10
INFERENCE_SCOPE = (
    'Conditional on this fixed CREMA-D corpus, encoder, training procedure and '
    'prespecified randomization. The 24 draws are the inferential units; five '
    'folds are averaged within each draw. Neither recordings nor people are '
    'resampled. No population, cross-corpus or causal-mechanism claim is certified.'
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def whole_fold_uar(labels, logits):
    """Percent macro recall over six acted labels, pooling all clips in the fold."""
    labels, logits = np.asarray(labels), np.asarray(logits)
    require(labels.ndim == 1 and labels.dtype.kind in 'iu' and labels.size > 0,
            'labels must be a nonempty integer vector')
    require(logits.shape == (labels.size, 6) and logits.dtype.kind in 'fiu'
            and np.isfinite(logits).all(), 'logits must be finite N by 6 real values')
    require(set(labels.tolist()) == set(range(6)), 'every role must contain all six acted classes')
    prediction = np.argmax(logits, axis=1)  # NumPy's first-class tie rule is fixed.
    recalls = [float(np.mean(prediction[labels == c] == c)) for c in range(6)]
    return 100.0 * float(np.mean(recalls))


def validate_analysis_plan(plan):
    units = plan['units']
    require(len(units) == FORMAL_UNITS, 'exactly 480 formal units are required, not a pilot or partial grid')
    require(len({u['unit_id'] for u in units}) == FORMAL_UNITS, 'duplicate formal unit ID')
    expected = {(d, f, c) for d in range(DRAWS) for f in range(FOLDS) for c in range(CONFIGS)}
    require({(u['draw'], u['fold'], u['config_index']) for u in units} == expected,
            '24 draw by 5 fold by 4 config grid is incomplete')
    analysis = plan['analysis']
    require(analysis['draws'] == DRAWS and analysis['fixed_config_index'] == FIXED_CONFIG_INDEX
            and analysis['bootstrap_repetitions'] == BOOTSTRAP_REPETITIONS
            and analysis['bootstrap_seed'] == BOOTSTRAP_SEED, 'prespecified analysis settings changed')
    indexed = {(u['draw'], u['fold'], u['config_index']): u for u in units}
    for draw in range(DRAWS):
        for fold in range(FOLDS):
            first = indexed[draw, fold, 0]
            require(len(first['fit']) == 576 and len(first['val_seen']) == len(first['val_unseen']) == 288,
                    'fixed fit or validation recording budget changed')
            for role in ('fit', *ROLES):
                require(first[role] and len(first[role]) == len(set(first[role])), 'duplicate or empty role paths')
            for config in range(CONFIGS):
                unit = indexed[draw, fold, config]
                require(all(type(unit[k]) is int for k in ('draw', 'fold', 'config_index', 'train_seed')),
                        'grid and seed values must be integers')
                require(all(unit[k] == first[k] for k in ('fit', *ROLES, 'train_seed')),
                        'configuration changes the common data or training seed')
            fixed = indexed[draw, fold, FIXED_CONFIG_INDEX]['config']
            require(fixed['lr_encoder'] == 5e-5 and fixed['lr_head'] == 1e-3,
                    'fixed configuration 3 learning rates changed')
    return indexed


def epoch_map(receipt):
    require(type(receipt.get('epochs_run')) is int and receipt['epochs_run'] == 15,
            'all 15 training epochs are required')
    result = {'best_seen': receipt['best_seen_epoch'], 'best_unseen': receipt['best_unseen_epoch'],
              'last': receipt['epochs_run']}
    require(all(type(e) is int and 1 <= e <= 15 for e in result.values()), 'checkpoint epoch is invalid')
    return result


def primary_inference(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (DRAWS,) and np.isfinite(values).all(), 'primary test requires exactly 24 finite draw effects')
    mean = float(values.mean())
    degenerate = bool(np.all(values == values[0]))
    sd = 0.0 if degenerate else float(values.std(ddof=1))
    se = sd / np.sqrt(DRAWS)
    critical = float(student_t.ppf(.975, DRAWS - 1))
    statistic = None if degenerate else float(mean / se)
    p = None if degenerate else float(2 * student_t.sf(abs(statistic), DRAWS - 1))
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, DRAWS, size=(BOOTSTRAP_REPETITIONS, DRAWS))
    boot_means = values[indices].mean(axis=1)
    boot_low, boot_high = np.quantile(boot_means, [.025, .975], method='linear')
    return {
        'estimate_pp': mean, 'draws': DRAWS, 'df': DRAWS - 1,
        'sample_sd_pp': sd, 'standard_error_pp': float(se),
        'alternative': 'two-sided', 't_statistic': statistic, 'p_value_two_sided': p,
        't_ci95_pp': [float(mean - critical * se), float(mean + critical * se)],
        't_test_defined': not degenerate,
        'degenerate_variance_note': ('All 24 effects are identical; t and p are undefined. '
                                     'The zero-width interval is only the zero-SE formula.' if degenerate else None),
        'bootstrap_sensitivity': {
            'method': 'percentile bootstrap of 24 draw effects with replacement; linear quantiles',
            'repetitions': BOOTSTRAP_REPETITIONS, 'seed': BOOTSTRAP_SEED,
            'bit_generator': 'PCG64', 'ci95_pp': [float(boot_low), float(boot_high)],
            'additional_hypothesis_test': False,
        },
        'scope': INFERENCE_SCOPE,
    }


def analyze(plan, predloader: Callable):
    """Pure analysis; the caller must establish a complete formal integrity gate."""
    indexed = validate_analysis_plan(plan)
    metrics, epochs, pred_rows, reference_labels = {}, {}, [], {}
    required_keys = {f'{role}__{suffix}' for role in ROLES
                     for suffix in ('paths', 'labels', *(c + '__logits' for c in CHECKPOINTS))}
    for (draw, fold, config), unit in sorted(indexed.items()):
        loaded = predloader(unit)
        arrays, receipt = loaded['predictions'], loaded['receipt']
        require(set(arrays) == required_keys, 'prediction key inventory differs')
        chosen_epochs = epoch_map(receipt)
        epochs[draw, fold, config] = chosen_epochs
        for role in ROLES:
            paths, labels = np.asarray(arrays[f'{role}__paths']), np.asarray(arrays[f'{role}__labels'])
            require(paths.ndim == 1 and paths.dtype.kind in 'US' and paths.tolist() == unit[role],
                    'prediction paths or ordering differ from the formal plan')
            require(labels.shape == (len(paths),), 'prediction label count differs')
            cell_role = draw, fold, role
            if cell_role in reference_labels:
                require(np.array_equal(labels, reference_labels[cell_role]), 'configurations disagree on query labels')
            else:
                reference_labels[cell_role] = labels.copy()
            for checkpoint in CHECKPOINTS:
                logits = np.asarray(arrays[f'{role}__{checkpoint}__logits'])
                value = whole_fold_uar(labels, logits)
                metrics[draw, fold, config, role, checkpoint] = value
                pred_rows.append(dict(unit_id=unit['unit_id'], draw=draw, fold=fold, config_index=config,
                    train_seed=unit['train_seed'], role=role, checkpoint=checkpoint,
                    checkpoint_epoch=chosen_epochs[checkpoint], records=len(labels), uar_percent=value))
            # The engine evaluates each unique epoch once and aliases its states.
            for i, first in enumerate(CHECKPOINTS):
                for second in CHECKPOINTS[i + 1:]:
                    if chosen_epochs[first] == chosen_epochs[second]:
                        require(np.array_equal(arrays[f'{role}__{first}__logits'], arrays[f'{role}__{second}__logits']),
                                'the same saved epoch produced different predictions')
    episodes, cells = [], {}
    for draw in range(DRAWS):
        for fold in range(FOLDS):
            sides = {}
            for rule in RULES:
                role, checkpoint = 'val_' + rule, 'best_' + rule
                selected = min(range(CONFIGS), key=lambda c: (-metrics[draw, fold, c, role, checkpoint], c))
                values = {}
                for family, config, ck in (
                    ('primary', selected, checkpoint),
                    ('fixed_config_best', FIXED_CONFIG_INDEX, checkpoint),
                    ('fixed_config_last', FIXED_CONFIG_INDEX, 'last')):
                    v, t = metrics[draw, fold, config, role, ck], metrics[draw, fold, config, 'test', ck]
                    values[family] = dict(v=v, t=t, gap=v-t, config=config, checkpoint=ck,
                                          epoch=epochs[draw, fold, config][ck])
                sides[rule] = values
                episode = dict(draw=draw, fold=fold, rule=rule, selection_role=role,
                    selected_config_index=selected, selected_unit_id=indexed[draw, fold, selected]['unit_id'],
                    train_seed=indexed[draw, fold, selected]['train_seed'])
                for family, value in values.items():
                    episode.update({family + '_config_index': value['config'], family + '_checkpoint': value['checkpoint'],
                        family + '_epoch': value['epoch'], family + '_validation_uar_percent': value['v'],
                        family + '_test_uar_percent': value['t'], family + '_gap_pp': value['gap']})
                episodes.append(episode)
            cell = {}
            for family in FAMILIES:
                seen, unseen = sides['seen'][family], sides['unseen'][family]
                delta_v, delta_t = seen['v'] - unseen['v'], seen['t'] - unseen['t']
                direct_gap = seen['gap'] - unseen['gap']
                require(abs(direct_gap - (delta_v - delta_t)) <= IDENTITY_TOLERANCE, 'gap difference identity failed')
                if family == 'fixed_config_last':
                    require(seen['t'] == unseen['t'] and delta_t == 0.0, 'fixed-config last must share the exact same test model')
                    delta_gap = delta_v  # Canonical exact cancellation of the common T.
                else:
                    delta_gap = direct_gap
                cell[family] = dict(seen_validation_uar_percent=seen['v'], unseen_validation_uar_percent=unseen['v'],
                    seen_test_uar_percent=seen['t'], unseen_test_uar_percent=unseen['t'], seen_gap_pp=seen['gap'],
                    unseen_gap_pp=unseen['gap'], delta_validation_pp=delta_v, delta_test_pp=delta_t,
                    delta_gap_pp=delta_gap, identity_residual_pp=delta_gap-(delta_v-delta_t))
            cells[draw, fold] = cell
    draw_rows = []
    for draw in range(DRAWS):
        row = dict(draw=draw, folds=FOLDS)
        for family in FAMILIES:
            for key in cells[draw, 0][family]:
                row[family + '_' + key] = float(np.mean([cells[draw, fold][family][key] for fold in range(FOLDS)]))
        row['selection_increment_pp'] = row['primary_delta_gap_pp'] - row['fixed_config_last_delta_gap_pp']
        require(row['fixed_config_last_delta_test_pp'] == 0.0
                and row['fixed_config_last_delta_gap_pp'] == row['fixed_config_last_delta_validation_pp'],
                'fixed-last cancellation changed during fold averaging')
        draw_rows.append(row)
    descriptive = {}
    for key in draw_rows[0]:
        if key not in ('draw', 'folds'):
            values = np.asarray([row[key] for row in draw_rows], dtype=np.float64)
            descriptive[key] = {'mean': float(values.mean()), 'draw_sd': float(values.std(ddof=1)),
                                'minimum': float(values.min()), 'maximum': float(values.max())}
    primary = primary_inference([row['primary_delta_gap_pp'] for row in draw_rows])
    primary['contrast'] = 'mean_draw(mean_fold[(V_seen-T_seen)-(V_unseen-T_unseen)])'
    result = dict(schema='ser-dual-validation-analysis-1', phase='formal', plan_sha256=plan['plan_sha256'],
        analyzed_formal_units=FORMAL_UNITS, excluded_pilot_runs=len(plan.get('pilot_units', [])),
        draws=DRAWS, folds_per_draw=FOLDS, configurations=CONFIGS,
        aggregation='whole-fold six-class macro recall; equal five-fold mean within draw; equal mean of 24 draws',
        selection='minimum own-validation CE epoch supplied by sealed runner; maximum own-validation UAR configuration; smallest config_index on exact ties',
        primary=primary, primary_hypothesis_tests=1, descriptive_draw_summaries=descriptive,
        fixed_config_index=FIXED_CONFIG_INDEX, fixed_config=indexed[0, 0, FIXED_CONFIG_INDEX]['config'],
        control_scope={'fixed_config_best': 'fixed hyperparameters; respective own-validation checkpoint selection remains',
                       'fixed_config_last': 'fixed hyperparameters and last epoch; no validation selection; identical test model',
                       'selection_increment_pp': 'primary gap difference minus fixed-config last gap difference; descriptive, not an identified mediation effect'},
        test_difference='primary_delta_test_pp is the paired T_seen minus T_unseen difference after each rule selects its configuration/checkpoint',
        identity='delta_gap = delta_validation - delta_test; fixed-config last delta_test is exactly zero',
        inference_scope=INFERENCE_SCOPE, additional_control_hypothesis_tests=0,
        no_outcome_based_stopping=True, all_prespecified_controls_reported=True)
    require(len(pred_rows) == 4320 and len(episodes) == 240 and len(draw_rows) == 24, 'analysis output grid incomplete')
    return {'pred_metrics': pred_rows, 'selected_episodes': episodes, 'draws': draw_rows, 'results': result}


def validate_complete_gate(plan, gate):
    require(gate.get('schema') == 'ser-dual-validation-result-gate-1' and gate.get('pass') is True
            and gate.get('phase') == 'formal' and gate.get('units') == FORMAL_UNITS,
            'a complete 480-unit formal gate is required before scoring')
    require(gate.get('plan_sha256') == plan['plan_sha256'] and gate.get('scores_computed') is False,
            'formal gate identity/scope differs')
    require(gate.get('gate_sha256') == digest({k: v for k, v in gate.items() if k != 'gate_sha256'}),
            'formal gate semantic hash differs')
    require(set(gate.get('done_sha256', {})) == {u['unit_id'] for u in plan['units']},
            'gate does not bind the exact formal DONE set')


def execute(repo, plan_path, outroot, gate_path, out):
    from v3.inner_validation import plan as planner, run as runner
    from v3.data_design.core_plan import read_metadata
    repo, plan_path, outroot, gate_path, out = [Path(p).resolve() for p in (repo, plan_path, outroot, gate_path, out)]
    require(Path(__file__).resolve() == repo/'v3/inner_validation/score.py'
            and Path(planner.__file__).resolve() == repo/'v3/inner_validation/plan.py'
            and Path(runner.__file__).resolve() == repo/'v3/inner_validation/run.py', 'imported code is from another checkout')
    require(not out.exists() and not out.is_relative_to(outroot) and not outroot.is_relative_to(out)
            and not plan_path.is_relative_to(out) and not gate_path.is_relative_to(out), 'output is not a new directory outside verified inputs')
    plan = planner.load_plan(plan_path, repo)
    validate_analysis_plan(plan)
    gate_raw_sha = file_sha(gate_path)
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    validate_complete_gate(plan, gate)
    rows = {r['relative_path']: r for r in read_metadata(repo)['clean']}
    phase_root, verified = outroot/'formal', {}
    ledger_path = phase_root/'ledger.jsonl'
    require(file_sha(ledger_path) == gate['ledger_sha256'], 'formal ledger changed since complete gate')
    # Finish every unit's artifact verification before opening any arrays for scoring.
    for unit in plan['units']:
        receipt = runner.verify_unit(unit, phase_root, plan['plan_sha256'], rows=rows)
        require(receipt is not None, 'formal unit not complete: ' + unit['unit_id'])
        directory = phase_root/'units'/unit['unit_id']
        done_path = directory/'DONE'
        require(file_sha(done_path) == gate['done_sha256'][unit['unit_id']], 'DONE changed since complete gate')
        done = json.loads(done_path.read_text(encoding='utf-8'))
        name = 'attempts/0001/predictions.npz'
        require(name in done['artifacts'], 'committed predictions absent')
        verified[unit['unit_id']] = (receipt, directory/name, done['artifacts'][name], done_path)
    def loader(unit):
        receipt, path, wanted, done_path = verified[unit['unit_id']]
        require(file_sha(path) == wanted, 'predictions changed after gate')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        require(file_sha(path) == wanted and file_sha(done_path) == gate['done_sha256'][unit['unit_id']],
                'prediction closure changed while loading')
        return {'predictions': arrays, 'receipt': receipt}
    report = analyze(plan, loader)
    require(file_sha(gate_path) == gate_raw_sha and file_sha(ledger_path) == gate['ledger_sha256'],
            'accepted gate or ledger changed during analysis')
    planner.load_plan(plan_path, repo)
    for uid, (_, path, wanted, done_path) in verified.items():
        require(file_sha(path) == wanted and file_sha(done_path) == gate['done_sha256'][uid],
                'prediction closure changed during analysis')
    out.mkdir(parents=True)
    for name in ('pred_metrics', 'selected_episodes', 'draws'):
        with (out/(name+'.csv')).open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[name][0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(report[name])
    results = report['results']
    results.update(source_sha256=file_sha(__file__), plan_file_sha256=file_sha(plan_path),
                   accepted_gate_file_sha256=gate_raw_sha, accepted_gate_sha256=gate['gate_sha256'],
                   done_mapping_sha256=digest(gate['done_sha256']),
                   table_sha256={name+'.csv': file_sha(out/(name+'.csv')) for name in ('pred_metrics', 'selected_episodes', 'draws')})
    with (out/'results.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'plan', 'outroot', 'gate', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.repo, args.plan, args.outroot, args.gate, args.out)
    print(json.dumps({'complete': True, 'formal_units': result['analyzed_formal_units'],
                      'primary': result['primary'], 'output': str(args.out)}, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()

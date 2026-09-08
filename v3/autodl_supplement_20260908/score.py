"""Independent, full-matrix scoring of the AutoDL supplement.

No engine metric or selection function is imported. The pure ``analyze`` API
requires all 720 formal units; its loader returns (NPZ arrays, selected_epochs).
File/source admission is a separate gate used by the command-line entry point.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

CORPORA = ('cremad', 'subesco', 'ravdess')
MODELS = ('wavlm_base_plus', 'hubert_base')
CLASSES = dict(cremad=6, subesco=7, ravdess=8)
GROUPS = ('A', 'B', 'outer')
RULES = ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar')
METRICS = ('D_CE', 'D_UAR', 'J', 'T_seen_ce', 'T_unseen_ce',
           'T_seen_uar', 'T_unseen_uar', 'T_last')
PANEL_FIELDS = ('fit', 'report', 'report_batches', 'seeds', 'n_classes',
                'group_speakers', 'test_speakers', 'fit_prompts',
                'query_prompts', 'reference_panel_sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    blob = Path(path).read_bytes()
    return json.loads(gzip.decompress(blob) if blob[:2] == b'\x1f\x8b' else blob)


def check_seal(value, key):
    require(isinstance(value, dict) and key in value
            and digest({k: v for k, v in value.items() if k != key}) == value[key],
            f'{key} differs')


def expected_family():
    items = [dict(id=f'hubert_{c}_{e}', corpus=c, endpoint=e, model='hubert_base',
                  window=15, comparison='absolute_seen_minus_unseen')
             for c in CORPORA for e in ('D_CE', 'J')]
    items += [dict(id=f'window_{c}_{e}', corpus=c, endpoint=e,
                   model='wavlm_base_plus', comparison='paired_window45_minus_window15')
              for c in ('subesco', 'ravdess') for e in ('L_CE', 'L_J')]
    return items


def validate_grid(plan):
    units = plan['units']
    expected = {(c, d, f, m) for c in CORPORA for d in range(24)
                for f in range(5) for m in MODELS}
    require(plan['formal_units'] == len(units) == 720, 'need exactly 720 formal units')
    require(plan['stats']['family'] == expected_family()
            and plan['stats']['family_size'] == 10
            and plan['stats']['test'] == 'two-sided one-sample t'
            and plan['stats']['multiplicity'] == 'Holm'
            and plan['stats']['alpha'] == .05 and plan['stats']['df'] == 23,
            'frozen ten-test contract differs')
    actual = []
    ids = []
    for unit in units:
        c, d, f, m = (unit[k] for k in ('corpus', 'draw', 'fold', 'model'))
        require(type(d) is int and type(f) is int, 'draw/fold must be integers')
        require(unit['phase'] == 'formal' and unit['arm'] == 'A'
                and unit['seen_group'] == 'A' and unit['unseen_group'] == 'B',
                'pilot or non-A unit entered formal analysis')
        require((c, d, f, m) in expected and unit['n_classes'] == CLASSES[c],
                'unexpected corpus/model/native classes/context')
        total = 45 if m == 'wavlm_base_plus' and c != 'cremad' else 15
        require(unit['config']['epochs'] == total
                and unit['windows'] == ([15] if total == 15 else [15, 45]),
                'model/window matrix differs')
        require(isinstance(unit['unit_id'], str)
                and re.fullmatch(r'[A-Za-z0-9_-]+', unit['unit_id']), 'unsafe unit ID')
        require(set(unit['report']) == set(GROUPS), 'report groups differ')
        occupied = set(unit['fit'])
        for group in GROUPS:
            paths = unit['report'][group]
            require(paths and len(paths) == len(set(paths))
                    and not occupied.intersection(paths), 'recording role overlap')
            occupied.update(paths)
            require([p for batch in unit['report_batches'][group] for p in batch] == paths,
                    'fixed report batching differs')
        actual.append((c, d, f, m))
        ids.append(unit['unit_id'])
    require(len(set(ids)) == 720 and len(set(actual)) == 720 and set(actual) == expected,
            'missing or duplicate formal context')
    index = dict(zip(actual, units))
    for c, d, f, _ in actual:
        left, right = (index[c, d, f, m] for m in MODELS)
        require(all(left[k] == right[k] for k in PANEL_FIELDS),
                'backbones do not share their fixed panel/seeds')
    return units


def _validated_arrays(logits, labels, classes):
    y = np.asarray(labels)
    x = np.asarray(logits)
    require(y.ndim == 1 and y.dtype.kind in 'iu' and len(y) > 0,
            'labels must be nonempty integer vector')
    require(type(classes) is int and classes > 1 and np.all((y >= 0) & (y < classes)),
            'label outside native classes')
    require(x.shape == (len(y), classes) and x.dtype.kind == 'f'
            and np.isfinite(x).all(), 'invalid logits shape/dtype/values')
    support = np.bincount(y.astype(np.int64), minlength=classes)
    require((support > 0).all(), 'missing native class')
    return x.astype(np.float64, copy=False), y.astype(np.int64, copy=False), support


def uar(logits, labels, classes):
    """Percentage UAR as a Fraction; unequal class support stays macro recall."""
    x, y, support = _validated_arrays(logits, labels, classes)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (y, x.argmax(axis=1)), 1)
    if np.all(support == support[0]):
        return Fraction(100 * int(np.trace(confusion)), int(confusion.sum()))
    return sum((Fraction(100 * int(confusion[c, c]), int(support[c]))
                for c in range(classes)), Fraction(0)) / classes


def ce_and_uar(logits, labels, classes):
    x, y, _ = _validated_arrays(logits, labels, classes)
    maximum = x.max(axis=1)
    lse = maximum + np.log(np.exp(x - maximum[:, None]).sum(axis=1))
    ce = float(np.mean(lse - x[np.arange(len(y)), y]))
    require(math.isfinite(ce), 'nonfinite cross entropy')
    return ce, uar(x, y, classes)


def measure_unit(unit, predictions, rows):
    expected_keys = {'epochs'} | {f'{g}__{suffix}' for g in GROUPS
                                 for suffix in ('paths', 'labels', 'all_epoch_logits')}
    require(set(predictions) == expected_keys, 'NPZ key contract differs')
    total, classes = unit['config']['epochs'], unit['n_classes']
    epochs = np.asarray(predictions['epochs'])
    require(epochs.dtype.kind in 'iu' and np.array_equal(epochs, np.arange(1, total + 1)),
            'incomplete or misordered epoch vector')
    measured = {}
    for group in GROUPS:
        paths = np.asarray(predictions[f'{group}__paths'])
        labels = np.asarray(predictions[f'{group}__labels'])
        logits = np.asarray(predictions[f'{group}__all_epoch_logits'])
        wanted = unit['report'][group]
        require(paths.ndim == 1 and paths.dtype.kind == 'U' and paths.tolist() == wanted,
                'prediction path ordering differs')
        expected_labels = [rows[p]['label_index'] for p in wanted]
        require(labels.dtype.kind in 'iu' and labels.shape == (len(wanted),)
                and labels.tolist() == expected_labels, 'prediction label identity differs')
        require(logits.shape == (total, len(wanted), classes)
                and logits.dtype.kind == 'f' and np.isfinite(logits).all(),
                'incomplete or invalid all-epoch logits')
        if group == 'outer':
            measured[group] = [(None, uar(x, labels, classes)) for x in logits]
        else:
            measured[group] = [ce_and_uar(x, labels, classes) for x in logits]
    selected = {}
    for window in unit['windows']:
        choice = {'last': window}
        for rule in RULES:
            group = 'A' if rule.startswith('seen_') else 'B'
            if rule.endswith('_ce'):
                offset = min(range(window), key=lambda e: (measured[group][e][0], e))
            else:
                offset = min(range(window), key=lambda e: (-measured[group][e][1], e))
            choice[rule] = offset + 1
        selected[str(window)] = choice
    return measured, selected


def mean_exact(values):
    require(bool(values), 'empty exact mean')
    return sum(values, Fraction(0)) / len(values)


def t_estimate(values, *, inference):
    require(len(values) == 24, 'inference requires 24 complete draws')
    x = np.asarray([float(v) for v in values], dtype=np.float64)
    require(np.isfinite(x).all(), 'nonfinite draw values')
    mean = float(mean_exact(values))
    # Constant rational values must not become tiny numerical sample variance.
    sd = 0.0 if all(v == values[0] for v in values) else float(np.std(x, ddof=1))
    se = sd / math.sqrt(24)
    result = dict(mean_pp=mean, n_draws=24, df=23, sd_draw_pp=sd, se_pp=se,
                  interval_scope='pointwise95%; conditional on fixed corpus/panels/program',
                  p_two_sided=None, holm_p_ten=None)
    if sd == 0:
        return dict(result, status='undefined_t_zero_sample_variance',
                    pointwise_95_ci_pp=None, t_statistic=None)
    from scipy.stats import t
    half = float(t.ppf(.975, 23)) * se
    result.update(pointwise_95_ci_pp=[mean - half, mean + half],
                  status='inferential_estimate' if inference else 'descriptive_estimate',
                  t_statistic=mean / se if inference else None)
    if inference:
        result['p_two_sided'] = float(2 * t.sf(abs(mean / se), 23))
    return result


def holm_ten(p_values):
    require(len(p_values) == 10, 'Holm must retain exactly ten slots')
    require(all(p is None or (math.isfinite(p) and 0 <= p <= 1) for p in p_values),
            'invalid p values')
    raw = [1.0 if p is None else p for p in p_values]
    order = sorted(range(10), key=lambda i: raw[i])
    adjusted, previous = [None] * 10, 0.0
    for rank, index in enumerate(order):
        previous = min(1.0, max(previous, (10 - rank) * raw[index]))
        if p_values[index] is not None:
            adjusted[index] = previous
    return adjusted


def analyze(plan, prediction_loader):
    """Return results and CSV row tables, using all 720 formal units.

    ``prediction_loader(unit) -> (dict_of_npz_arrays, selected_epochs_mapping)``.
    A caller using this pure API owns artifact/source admission. The CLI adds it.
    No pilot data is ever requested from the loader.
    """
    units = validate_grid(plan)
    exact, unit_rows, selected_rows, curves = {}, [], [], []
    for unit in sorted(units, key=lambda u: (CORPORA.index(u['corpus']), u['draw'],
                                            u['fold'], MODELS.index(u['model']))):
        arrays, receipt_selection = prediction_loader(unit)
        measured, selected = measure_unit(unit, arrays, plan['rows'][unit['corpus']])
        require(receipt_selection == selected, 'receipt checkpoint selection differs from logits')
        require(all(type(epoch) is int for rules in receipt_selection.values()
                    for epoch in rules.values()), 'receipt epochs must be integers')
        c, d, f, m = (unit[k] for k in ('corpus', 'draw', 'fold', 'model'))
        common = dict(unit_id=unit['unit_id'], corpus=c, draw=d, fold=f, model=m)
        for epoch in range(1, unit['config']['epochs'] + 1):
            aa, bb, tt = (measured[g][epoch - 1] for g in GROUPS)
            curves.append(dict(common, epoch=epoch, seen_ce=aa[0], unseen_ce=bb[0],
                               seen_uar=float(aa[1]), unseen_uar=float(bb[1]), outer_uar=float(tt[1]),
                               uar_unit='UAR_percent', ce_unit='natural_log_loss'))
        for window in unit['windows']:
            rules = selected[str(window)]
            target = {r: measured['outer'][e - 1][1] for r, e in rules.items()}
            dc, du = target['seen_ce'] - target['unseen_ce'], target['seen_uar'] - target['unseen_uar']
            values = dict(D_CE=dc, D_UAR=du, J=dc - du,
                          **{f'T_{r}': target[r] for r in (*RULES, 'last')})
            exact[c, d, f, m, window] = values
            unit_rows.append(dict(common, window=window,
                                  units={k: 'UAR_percent' if k.startswith('T_') else 'percentage_points'
                                         for k in METRICS},
                                  **{k + '_pp': float(v) for k, v in values.items()},
                                  **{'epoch_' + k: v for k, v in rules.items()},
                                  ce_epoch_disagreement=int(rules['seen_ce'] != rules['unseen_ce']),
                                  uar_epoch_disagreement=int(rules['seen_uar'] != rules['unseen_uar'])))
            oracle = max(measured['outer'][e][1] for e in range(window))
            for rule, epoch in rules.items():
                selected_rows.append(dict(common, window=window, rule=rule, selected_epoch=epoch,
                                          unit='UAR_percent', oracle_shortfall_unit='percentage_points',
                                          outer_uar=float(target[rule]), oracle_outer_uar=float(oracle),
                                          oracle_shortfall_pp=float(oracle - target[rule]),
                                          oracle_used_for_selection=False))
    draw_exact, draw_rows = {}, []
    for c in CORPORA:
        settings = [('model_window', m, w) for m in MODELS
                    for w in ([15, 45] if m == 'wavlm_base_plus' and c != 'cremad' else [15])]
        settings.append(('hubert_minus_wavlm15', 'hubert_minus_wavlm', 15))
        if c != 'cremad':
            settings.append(('window45_minus15', 'wavlm_base_plus', '45_minus15'))
        for kind, m, window in settings:
            for d in range(24):
                fold_values = []
                for f in range(5):
                    if kind == 'model_window':
                        value = exact[c, d, f, m, window]
                    else:
                        if kind == 'hubert_minus_wavlm15':
                            left, right = exact[c, d, f, 'hubert_base', 15], exact[c, d, f, 'wavlm_base_plus', 15]
                        else:
                            left, right = exact[c, d, f, m, 45], exact[c, d, f, m, 15]
                        value = {k: left[k] - right[k] for k in METRICS}
                    fold_values.append(value)
                values = {k: mean_exact([v[k] for v in fold_values]) for k in METRICS}
                draw_exact[c, kind, m, window, d] = values
                draw_rows.append(dict(corpus=c, comparison=kind, model=m, window=window, draw=d,
                                      units={k: 'UAR_percent' if kind == 'model_window' and k.startswith('T_')
                                             else 'percentage_points' for k in METRICS},
                                      **{k + '_pp': float(v) for k, v in values.items()}))
    tests = []
    for item in expected_family():
        c, m, endpoint = item['corpus'], item['model'], item['endpoint']
        if endpoint.startswith('L_'):
            kind, w, key = 'window45_minus15', '45_minus15', {'L_CE': 'D_CE', 'L_J': 'J'}[endpoint]
        else:
            kind, w, key = 'model_window', 15, endpoint
        values = [draw_exact[c, kind, m, w, d][key] for d in range(24)]
        tests.append(dict(item, unit='percentage_points', **t_estimate(values, inference=True)))
    for row, p in zip(tests, holm_ten([r['p_two_sided'] for r in tests])):
        row.update(holm_p_ten=p, reject_familywise_05=bool(p is not None and p <= .05))
    descriptions = []
    settings = sorted({k[:4] for k in draw_exact}, key=str)
    for c, kind, m, w in settings:
        for metric in METRICS:
            values = [draw_exact[c, kind, m, w, d][metric] for d in range(24)]
            descriptions.append(dict(corpus=c, comparison=kind, model=m, window=w, endpoint=metric,
                                     unit='UAR_percent' if kind == 'model_window' and metric.startswith('T_')
                                          else 'percentage_points',
                                     hypothesis_test=False, **t_estimate(values, inference=False)))
    results = dict(schema='ser-autodl-supplement-results-1', program=plan.get('program'),
                   plan_sha256=plan.get('plan_sha256'),
                   counts=dict(formal_units=720, unit_windows=len(unit_rows),
                               full_epoch_curve_rows=len(curves), main_tests=10, draws_per_test=24,
                               folds_per_draw=5, pilots_included=0),
                   tests=tests, descriptive=descriptions,
                   interpretation=dict(uar='whole-role native-class macro recall; percent/percentage points',
                                       primary_family='ten two-sided tests, Holm alpha0.05',
                                       independent_unit='draw mean after five equal-weight folds',
                                       intervals='pointwise95%, not simultaneous or equivalence tests',
                                       historical_results='excluded from new pairwise estimates',
                                       zero_variance='t and CI undefined; slot retained conservatively',
                                       inclusion='all prespecified outcomes, including opposite and imprecise effects'))
    results['interpretation']['numeric_field_units'] = (
        'The legacy *_pp numeric keys are retained; explicit unit/units metadata governs: '
        'absolute model-window T values are UAR_percent, all contrasts are percentage_points.')
    return results, dict(units=unit_rows, draws=draw_rows, primary_tests=tests,
                         descriptive=descriptions, selected_epochs=selected_rows, curves=curves)


def safe_member(root, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative
            and ':' not in relative and not PurePosixPath(relative).is_absolute()
            and all(part not in ('', '.', '..') for part in relative.split('/')),
            'unsafe relative member')
    root = Path(root).resolve()
    cursor = root
    for part in relative.split('/'):
        cursor = cursor / part
        require(not cursor.is_symlink(), 'symlink input member')
        require(not (hasattr(cursor, 'is_junction') and cursor.is_junction()), 'junction input member')
    full = cursor.resolve()
    require(full.is_relative_to(root), 'member escapes root')
    return full


def checked_file(path, descriptor):
    require(isinstance(descriptor, dict) and type(descriptor.get('bytes')) is int
            and descriptor['bytes'] >= 0 and isinstance(descriptor.get('sha256'), str),
            'invalid file descriptor')
    require(path.is_file() and path.stat().st_size == descriptor['bytes']
            and file_sha(path) == descriptor['sha256'], f'file identity differs: {path.name}')
    return descriptor['sha256']


def verify_unit_files(unit, unit_dir, plan_sha, lock_sha):
    """Check committed bytes, including a sealed release/backup when pruned.

    This admits archived predictions, not an independent re-execution of the
    training or an attestation of an off-instance filesystem we cannot access.
    """
    unit_dir = Path(unit_dir)
    pinned = {}

    def load(name):
        path = safe_member(unit_dir, name)
        require(path.is_file(), f'missing unit artifact: {name}')
        pinned[str(path)] = file_sha(path)
        return read_json(path)

    check_seal(unit, 'unit_sha256')
    receipt = load('receipt.json')
    complete = load('COMPLETE.json')
    identity = dict(unit_id=unit['unit_id'], unit_sha256=unit['unit_sha256'],
                    plan_sha256=plan_sha, source_lock_sha256=lock_sha)
    for item in (receipt, complete):
        require(all(item.get(k) == v for k, v in identity.items()), 'unit completion identity differs')
    receipt_sha = pinned[str(safe_member(unit_dir, 'receipt.json'))]
    require(set(complete) == set(identity) | {'receipt_sha256'}
            and complete.get('receipt_sha256') == receipt_sha, 'COMPLETE does not bind receipt bytes')
    sample = unit['permanent_checkpoint_sample']
    require(type(sample) is bool, 'invalid checkpoint retention flag')
    files = receipt.get('files')
    require(isinstance(files, dict)
            and set(files) == {'predictions.npz', 'history.json', 'checkpoint.pt'}
                | ({'fresh_restore.json'} if sample else set()),
            'receipt artifact inventory differs')
    for name in sorted(set(files) - {'checkpoint.pt'}):
        path = safe_member(unit_dir, name)
        pinned[str(path)] = checked_file(path, files[name])
    info = receipt['info']
    require(receipt.get('phase') == 'formal' and info.get('unit_sha256') == unit['unit_sha256']
            and info.get('epochs') == unit['config']['epochs'] and info.get('windows') == unit['windows']
            and info.get('checkpoint_reload_verified') is True
            and type(info.get('reload_max_abs_diff')) in (float, int)
            and math.isfinite(info['reload_max_abs_diff'])
            and 0 <= info['reload_max_abs_diff'] <= 1e-5
            and info.get('outer_scores_computed') is False,
            'unit did not pass its committed checkpoint-reload gate')
    choices = info.get('selected_epochs')
    require(isinstance(choices, dict) and set(choices) == {str(w) for w in unit['windows']},
            'receipt window coverage differs')
    for window, rules in choices.items():
        require(isinstance(rules, dict) and set(rules) == set(RULES) | {'last'}
                and rules['last'] == int(window)
                and all(type(e) is int and 1 <= e <= int(window) for e in rules.values()),
                'receipt epoch choice is invalid')
    selected_union = sorted({e for rules in choices.values() for e in rules.values()})
    replay_keys = {f'{e}/{g}' for e in selected_union for g in GROUPS}
    replay = info.get('reload_by_epoch_group')
    require(isinstance(replay, dict) and set(replay) == replay_keys
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1e-5
                    for v in replay.values())
            and info.get('reload_checks') == len(replay_keys)
            and info.get('checkpoint_unique_epochs') == selected_union
            and info.get('checkpoint_unique_epoch_count') == len(selected_union)
            and info.get('checkpoint_sha256') == files['checkpoint.pt']['sha256']
            and info['reload_max_abs_diff'] == max(replay.values())
            and info.get('reload_checked_values') == len(selected_union) * unit['n_classes']
                * sum(len(unit['report'][g]) for g in GROUPS),
            'in-unit replay does not cover every retained state and report group')
    if sample:
        fresh = read_json(safe_member(unit_dir, 'fresh_restore.json'))
        comparisons = fresh.get('comparisons')
        require(info.get('fresh_process_restore_verified') is True
                and fresh.get('schema') == 'ser-autodl-fresh-restore-1'
                and fresh.get('passed') is True
                and all(fresh.get(k) == v for k, v in identity.items())
                and fresh.get('environment_sha256') == receipt.get('environment_sha256')
                and fresh.get('checkpoint_sha256') == files['checkpoint.pt']['sha256']
                and fresh.get('predictions_sha256') == files['predictions.npz']['sha256']
                and isinstance(comparisons, dict) and set(comparisons) == replay_keys
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1e-5
                        for v in comparisons.values())
                and fresh.get('max_abs_diff') == max(comparisons.values()),
                'permanent sample lacks complete bound fresh-process restoration')
    history = read_json(safe_member(unit_dir, 'history.json'))
    total = unit['config']['epochs']
    require(isinstance(history, list) and len(history) == total
            and [r.get('epoch') for r in history] == list(range(1, total + 1)),
            'incomplete epoch history')
    for row in history:
        require(type(row.get('epoch')) is int
                and all(type(row.get(k)) in (int, float) and math.isfinite(row[k])
                        for k in ('train_loss', *RULES)), 'invalid history metric')
        require(type(row.get('optimizer_steps')) is int and row['optimizer_steps'] > 0
                and type(row.get('scaler_skipped_steps')) is int
                and 0 <= row['scaler_skipped_steps'] <= row['optimizer_steps'],
                'invalid optimizer/AMP history')
    checkpoint = safe_member(unit_dir, 'checkpoint.pt')
    release_path = safe_member(unit_dir, 'checkpoint_release.json')
    if sample:
        require(checkpoint.is_file() and not release_path.exists(), 'permanent sample checkpoint released')
    if checkpoint.is_file():
        pinned[str(checkpoint)] = checked_file(checkpoint, files['checkpoint.pt'])
        retention = 'checkpoint_present_and_hashed'
    else:
        require(not sample, 'missing permanent checkpoint sample')
        release, ack = load('checkpoint_release.json'), load('backup_ack.json')
        check_seal(release, 'release_sha256')
        check_seal(ack, 'backup_ack_sha256')
        require(release.get('schema') == 'ser-checkpoint-release-1'
                and all(release.get(k) == v and ack.get(k) == v for k, v in identity.items())
                and release.get('receipt_sha256') == ack.get('receipt_sha256') == receipt_sha,
                'release/backup identity differs')
        require(release.get('checkpoint_sha256') == files['checkpoint.pt']['sha256']
                and release.get('backup_ack_sha256') == ack['backup_ack_sha256']
                and release.get('checkpoint_reload_verified') is True
                and release.get('checkpoint_retained') is False
                and ack.get('backup_location') == 'off-instance-local-verified'
                and type(ack.get('checkpoint_in_backup')) is bool,
                'missing checkpoint lacks bound backup/reload evidence')
        needed = {'predictions.npz', 'history.json', 'receipt.json', 'COMPLETE.json'}
        require(isinstance(ack.get('files'), dict) and needed.issubset(ack['files']),
                'backup acknowledgement lacks required permanent artifacts')
        for name, desc in ack['files'].items():
            path = safe_member(unit_dir, name)
            if name == 'checkpoint.pt' and not path.exists():
                require(ack['checkpoint_in_backup'] is True and desc == files[name],
                        'checkpoint backup descriptor differs')
            else:
                pinned[str(path)] = checked_file(path, desc)
        require(('checkpoint.pt' in ack['files']) is ack['checkpoint_in_backup'],
                'checkpoint backup flag contradicts inventory')
        retention = 'released_after_bound_backup_ack_and_in_unit_reload'
    return receipt, dict(unit_id=unit['unit_id'], receipt_sha256=receipt_sha,
                         complete_sha256=pinned[str(safe_member(unit_dir, 'COMPLETE.json'))],
                         retention=retention), pinned


def verify_runtime_records(run_dir, plan, lock):
    """Bind archived runtime bytes without resolving or using cloud-local paths."""
    pinned = {}
    records = {}
    for name in ('ENVIRONMENT.json', 'RUNTIME_PATHS.local.json', 'PLAN.json.gz'):
        path = safe_member(run_dir, name)
        require(path.is_file(), f'missing frozen runtime record: {name}')
        pinned[str(path)] = file_sha(path)
        records[name] = read_json(path)
    environment, runtime, snapshot = (records[n] for n in
                                      ('ENVIRONMENT.json', 'RUNTIME_PATHS.local.json', 'PLAN.json.gz'))
    check_seal(environment, 'environment_sha256')
    check_seal(snapshot, 'plan_sha256')
    require(environment['environment_sha256'] == lock['environment_sha256']
            and pinned[str(safe_member(run_dir, 'RUNTIME_PATHS.local.json'))] == lock['runtime_paths_sha256']
            and snapshot == plan, 'frozen environment/runtime/plan snapshot differs')
    require(set(runtime.get('models', {})) == set(lock['models']) == set(MODELS),
            'runtime model inventory differs')
    for name, expected in lock['models'].items():
        require(all(runtime['models'][name].get(k) == v for k, v in expected.items()),
                'runtime model identity differs')
    return pinned


def audit_inputs(plan, plan_path, run_dir):
    """Require a complete formal inventory before any scientific aggregation."""
    from .plan import validate_plan
    validate_plan(plan)
    units = validate_grid(plan)
    run_dir = Path(run_dir).resolve()
    source_path = safe_member(run_dir, 'SOURCE_LOCK.json')
    lock = read_json(source_path)
    check_seal(lock, 'source_lock_sha256')
    require(lock['plan_sha256'] == plan['plan_sha256'], 'source lock plan differs')
    repo = Path(__file__).resolve().parents[2]
    require(isinstance(lock.get('sources'), dict)
            and 'v3/autodl_supplement_20260908/score.py' in lock['sources'],
            'score source not frozen')
    require(set(lock['sources']) == {p.relative_to(repo).as_posix()
                                    for p in Path(__file__).resolve().parent.glob('*.py')},
            'frozen source inventory differs from current package')
    pinned = {str(Path(plan_path).resolve()): file_sha(plan_path), str(source_path): file_sha(source_path)}
    pinned.update(verify_runtime_records(run_dir, plan, lock))
    for relative, sha in lock['sources'].items():
        path = safe_member(repo, relative)
        require(path.is_file() and file_sha(path) == sha, 'running source differs from frozen lock')
        pinned[str(path)] = sha
    unit_root = safe_member(run_dir, 'units')
    require(unit_root.is_dir(), 'missing formal units directory')
    require({p.name for p in unit_root.iterdir()} == {u['unit_id'] for u in units},
            'formal directory contains missing or unexpected units/pilots')
    receipts, audits = {}, []
    for unit in units:
        path = safe_member(unit_root, unit['unit_id'])
        require(path.is_dir(), 'unit directory absent')
        receipt, audit, hashes = verify_unit_files(unit, path, plan['plan_sha256'], lock['source_lock_sha256'])
        wanted = lock['models'][unit['model']]
        actual = receipt['info'].get('model_identity', {})
        require(actual == wanted and receipt.get('environment_sha256') == lock['environment_sha256'],
                'trained model/runtime identity differs')
        receipts[unit['unit_id']] = receipt
        audits.append(audit)
        pinned.update(hashes)
    audit = dict(schema='ser-autodl-score-input-audit-1', phase='formal', pass_complete=True,
                 expected_units=720, verified_units=720, plan_sha256=plan['plan_sha256'],
                 source_lock_sha256=lock['source_lock_sha256'], unit_audits=audits,
                 file_sha256=pinned,
                 scope='source/plan seals and complete committed bytes; released states require bound '
                       'backup acknowledgements and in-unit replay; not a fresh training replication')
    return receipts, audit


def output_path(out, plan_path, run_dir):
    out = Path(out).resolve()
    run_dir, plan_path = Path(run_dir).resolve(), Path(plan_path).resolve()
    require(not out.exists(), 'output directory must be new')
    require(not out.is_relative_to(run_dir) and not run_dir.is_relative_to(out)
            and not out.is_relative_to(plan_path.parent), 'output overlaps frozen inputs')
    return out


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            value = {k: ('NA' if row.get(k) is None else
                         json.dumps(row[k], ensure_ascii=False, allow_nan=False)
                         if isinstance(row[k], (list, dict)) else row[k]) for k in keys}
            writer.writerow(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = output_path(args.out, args.plan, args.run_dir)
    plan = read_json(args.plan)
    receipts, audit = audit_inputs(plan, args.plan, args.run_dir)
    run_dir = Path(args.run_dir).resolve()

    def loader(unit):
        path = safe_member(run_dir, f"units/{unit['unit_id']}/predictions.npz")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return arrays, receipts[unit['unit_id']]['info']['selected_epochs']

    result, tables = analyze(plan, loader)
    for path, sha in audit['file_sha256'].items():
        require(Path(path).is_file() and file_sha(path) == sha, 'input changed during scoring')
    out.mkdir(parents=True)
    input_file = out / 'input_audit.json'
    input_file.write_bytes(canonical(audit) + b'\n')
    table_files = {}
    for key, rows in tables.items():
        path = out / (key + '.csv')
        _write_csv(path, rows)
        table_files[path.name] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
    result.update(inputs=dict(plan_file=str(Path(args.plan).resolve()), plan_file_sha256=file_sha(args.plan),
                              run_dir=str(run_dir), input_audit_file='input_audit.json',
                              input_audit_sha256=file_sha(input_file)), tables=table_files)
    result['result_sha256'] = digest(result)
    results_path = out / 'results.json'
    results_path.write_bytes(canonical(result) + b'\n')
    inventory = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in out.iterdir()}
    manifest = dict(schema='ser-autodl-score-output-1', files=inventory, plan_sha256=plan['plan_sha256'])
    manifest['manifest_sha256'] = digest(manifest)
    (out / 'OUTPUT_MANIFEST.json').write_bytes(canonical(manifest) + b'\n')
    print(json.dumps(dict(out=str(out), formal_units=720, main_tests=10,
                          result_sha256=result['result_sha256'])))


if __name__ == '__main__':
    main()

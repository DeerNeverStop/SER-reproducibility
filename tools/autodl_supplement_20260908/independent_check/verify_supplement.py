"""Independent numeric replay after a complete AutoDL 720-unit archive.

No frozen score/engine function is imported. This process never trains a model,
loads a checkpoint with torch, or reports partial-matrix scientific effects.
"""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction as F
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.stats import t as student_t

CORPORA = {'cremad': 6, 'subesco': 7, 'ravdess': 8}
MODELS = ('hubert_base', 'wavlm_base_plus')
RULES = ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar', 'last')
METRICS = ('D_CE', 'D_UAR', 'J', *(f'T_{r}' for r in RULES))
TABLES = ('units', 'draws', 'primary_tests', 'descriptive', 'selected_epochs', 'curves')


def demand(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw)


def read_bound(path, pins):
    """Parse exactly the bytes whose SHA is recorded, including prior pin checks."""
    raw = Path(path).read_bytes()
    key = str(Path(path).resolve()); current = hashlib.sha256(raw).hexdigest()
    demand(key not in pins or pins[key] == current, 'JSON changed between binding and parsing')
    pins[key] = current
    return json.loads(gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while data := handle.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def seal_ok(value, field):
    demand(value.get(field) == digest({k: v for k, v in value.items() if k != field}),
           f'invalid self seal: {field}')


def member(root, relative):
    demand(isinstance(relative, str) and relative and ':' not in relative and '\\' not in relative
           and not PurePosixPath(relative).is_absolute()
           and all(p not in ('', '.', '..') for p in relative.split('/')), 'unsafe relative path')
    current = Path(root).resolve()
    for part in relative.split('/'):
        current /= part
        demand(not current.is_symlink() and not (hasattr(current, 'is_junction') and current.is_junction()),
               'symlink or junction input')
    demand(current.resolve().is_relative_to(Path(root).resolve()), 'path escapes archive')
    return current


def pin(path, expected, pins):
    demand(set(expected) == {'bytes', 'sha256'} and type(expected['bytes']) is int,
           'invalid file descriptor')
    demand(str(path.resolve()) not in pins or pins[str(path.resolve())] == expected['sha256'],
           'earlier parsed bytes differ from manifest claim')
    demand(path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
           f'file bytes differ: {path.name}')
    pins[str(path.resolve())] = expected['sha256']


def family():
    rows = [(f'hubert_{c}_{e}', c, 'hubert_base', 'model_window', 15, e, e)
            for c in CORPORA for e in ('D_CE', 'J')]
    rows.extend((f'window_{c}_{e}', c, 'wavlm_base_plus', 'window45_minus15',
                 '45_minus15', e, source)
                for c in ('subesco', 'ravdess') for e, source in (('L_CE', 'D_CE'), ('L_J', 'J')))
    return rows


def full_grid(plan):
    units = plan['units']
    demand(len(units) == plan['formal_units'] == 720, 'complete 720-unit grid is required before scoring')
    wanted = {(c, d, f, m) for c in CORPORA for d in range(24) for f in range(5) for m in MODELS}
    actual, ids = {}, set()
    for u in units:
        key = u['corpus'], u['draw'], u['fold'], u['model']
        demand(key in wanted and key not in actual and u['unit_id'] not in ids, 'duplicate/unknown grid cell')
        demand(type(u['draw']) is type(u['fold']) is int and u['phase'] == 'formal'
               and u['arm'] == u['seen_group'] == 'A' and u['unseen_group'] == 'B', 'nonformal role entered grid')
        total = 45 if u['model'] == 'wavlm_base_plus' and u['corpus'] != 'cremad' else 15
        demand(u['windows'] == ([15, 45] if total == 45 else [15])
               and u['config']['epochs'] == total and u['n_classes'] == CORPORA[u['corpus']], 'model/window/task mismatch')
        demand(len(u['fit']) == {'cremad': 576, 'subesco': 224, 'ravdess': 64}[u['corpus']], 'fit budget differs')
        occupied = set(u['fit'])
        demand(len(occupied) == len(u['fit']), 'duplicate fit recording')
        for g in ('A', 'B', 'outer'):
            paths = u['report'][g]
            demand(paths and len(paths) == len(set(paths)) and not occupied.intersection(paths), 'recording roles overlap')
            demand(sum(u['report_batches'][g], []) == paths, 'fixed batch layout differs')
            occupied.update(paths)
        actual[key] = u; ids.add(u['unit_id'])
    demand(set(actual) == wanted, 'missing full-grid context')
    for c, d, f, _ in actual:
        h, w = (actual[c, d, f, m] for m in MODELS)
        demand(all(h[k] == w[k] for k in ('fit', 'report', 'report_batches', 'seeds',
                                         'group_speakers', 'test_speakers', 'fit_prompts', 'query_prompts')),
               'paired backbones use different panels')
    demand(plan['stats']['family_size'] == 10 and plan['stats']['df'] == 23
           and plan['stats']['multiplicity'] == 'Holm'
           and plan['stats']['test'] == 'two-sided one-sample t'
           and [r['id'] for r in plan['stats']['family']] == [r[0] for r in family()], 'primary family differs')
    return actual


def audit_backup(root, repo):
    """No prediction array is opened until all archive receipts/bytes pass."""
    root = Path(root).resolve(); pins = {}
    plan = read_bound(member(root, 'PLAN.json.gz'), pins)
    seal_ok(plan, 'plan_sha256'); full_grid(plan)
    completed = read_bound(member(root, 'BACKUP_COMPLETE.json'), pins)
    seal_ok(completed, 'backup_complete_sha256')
    ids = {u['unit_id'] for u in plan['units']}
    demand(completed['formal_units'] == 720 and set(completed['units']) == ids
           and {p.name for p in member(root, 'units').iterdir()} == ids,
           'complete archived 720 UID set required before prediction access')
    lock = read_bound(member(root, 'SOURCE_LOCK.json'), pins); seal_ok(lock, 'source_lock_sha256')
    demand(completed['plan_sha256'] == lock['plan_sha256'] == plan['plan_sha256']
           and completed['source_lock_sha256'] == lock['source_lock_sha256'], 'archive/plan/source identity mismatch')
    for name, expected in completed['top_files'].items():
        pin(member(root, name), expected, pins)
    for name in ('PLAN.json.gz', 'SOURCE_LOCK.json', 'ENVIRONMENT.json', 'RUNTIME_PATHS.local.json',
                 'ADMISSION.json', 'pilot_gate.json'):
        demand(str(member(root, name)) in pins, 'missing frozen top-level record')
    environment = read_bound(member(root, 'ENVIRONMENT.json'), pins); seal_ok(environment, 'environment_sha256')
    demand(environment['environment_sha256'] == lock['environment_sha256']
           and sha(member(root, 'RUNTIME_PATHS.local.json')) == lock['runtime_paths_sha256'], 'runtime identity differs')
    for name, expected_sha in lock['sources'].items():
        path = member(repo, name)
        demand(path.is_file() and sha(path) == expected_sha, 'current frozen source differs')
        pins[str(path.resolve())] = expected_sha
    demand('v3/autodl_supplement_20260908/score.py' in lock['sources'], 'frozen scorer absent')
    receipts = {}; samples = 0
    for unit in plan['units']:
        seal_ok(unit, 'unit_sha256')
        uid = unit['unit_id']; folder = member(root, 'units/' + uid)
        saved = completed['units'][uid]
        demand(saved['unit_id'] == uid, 'archived unit entry differs')
        for name, expected in saved['files'].items():
            pin(member(folder, name), expected, pins)
        for name in ('receipt.json', 'COMPLETE.json', 'predictions.npz', 'history.json', 'backup_ack.json'):
            demand(name in saved['files'], 'required committed file absent')
        complete, receipt = read_bound(folder / 'COMPLETE.json', pins), read_bound(folder / 'receipt.json', pins)
        identity = dict(unit_id=uid, unit_sha256=unit['unit_sha256'], plan_sha256=plan['plan_sha256'],
                        source_lock_sha256=lock['source_lock_sha256'])
        demand(all(complete.get(k) == receipt.get(k) == v for k, v in identity.items()), 'committed identity differs')
        demand(sha(folder / 'receipt.json') == complete['receipt_sha256'] == saved['receipt_sha256'], 'receipt SHA differs')
        pin(folder / 'COMPLETE.json', saved['complete_record'], pins)
        start_path = member(root, 'attempts/' + uid + '.started.json')
        pin(start_path, saved['attempt_record'], pins)
        start = read_bound(start_path, pins)
        demand(all(start.get(k) == identity[k] for k in ('unit_id', 'unit_sha256', 'source_lock_sha256')), 'attempt identity differs')
        demand(receipt['phase'] == 'formal' and receipt['environment_sha256'] == lock['environment_sha256']
               and receipt['info']['model_identity'] == lock['models'][unit['model']]
               and receipt['info']['checkpoint_reload_verified'] is True
               and receipt['info']['outer_scores_computed'] is False, 'committed runtime/reload receipt differs')
        for name in ('predictions.npz', 'history.json'):
            demand(receipt['files'][name] == saved['files'][name], 'committed payload descriptor differs')
        ack = read_bound(folder / 'backup_ack.json', pins); seal_ok(ack, 'backup_ack_sha256')
        demand(all(ack.get(k) == v for k, v in identity.items())
               and ack['receipt_sha256'] == complete['receipt_sha256']
               and ack['backup_location'] == 'off-instance-local-verified'
               and saved['backup_ack_sha256'] == ack['backup_ack_sha256'], 'ACK identity differs')
        demand({'predictions.npz', 'history.json', 'receipt.json', 'COMPLETE.json'} <= set(ack['files'])
               and ('checkpoint.pt' in ack['files']) is ack['checkpoint_in_backup'], 'ACK inventory/retention mismatch')
        for name, expected in ack['files'].items():
            demand(saved['files'].get(name) == expected, 'ACK does not match actual retained bytes')
        if unit['permanent_checkpoint_sample']:
            samples += 1
            demand({'checkpoint.pt', 'fresh_restore.json'} <= set(saved['files'])
                   and receipt['files']['checkpoint.pt'] == saved['files']['checkpoint.pt']
                   and receipt['files']['fresh_restore.json'] == saved['files']['fresh_restore.json'], 'sample checkpoint not archived')
            fresh = read_bound(folder / 'fresh_restore.json', pins)
            demand(fresh['passed'] is True and all(fresh[k] == v for k, v in identity.items())
                   and fresh['checkpoint_sha256'] == saved['files']['checkpoint.pt']['sha256']
                   and fresh['predictions_sha256'] == saved['files']['predictions.npz']['sha256'], 'fresh restore identity differs')
        else:
            demand({'checkpoint_release.json', 'checkpoint_release_intent.json'} <= set(saved['files']), 'release receipts absent')
            release = read_bound(folder / 'checkpoint_release.json', pins); seal_ok(release, 'release_sha256')
            demand(release == read_bound(folder / 'checkpoint_release_intent.json', pins)
                   and all(release[k] == v for k, v in identity.items())
                   and release['backup_ack_sha256'] == ack['backup_ack_sha256']
                   and release['receipt_sha256'] == complete['receipt_sha256']
                   and release['checkpoint_sha256'] == receipt['files']['checkpoint.pt']['sha256']
                   and release['checkpoint_reload_verified'] is True and release['checkpoint_retained'] is False,
                   'released checkpoint lacks complete bound evidence')
        receipts[uid] = receipt
    demand(samples == 12, 'twelve fixed checkpoint samples required')
    return plan, receipts, pins


def class_recall(logits, labels, classes):
    x, y = np.asarray(logits), np.asarray(labels)
    demand(y.ndim == 1 and y.dtype.kind in 'iu' and len(y) and np.all((y >= 0) & (y < classes)), 'invalid labels')
    demand(x.shape == (len(y), classes) and x.dtype.kind == 'f' and np.isfinite(x).all(), 'invalid logits')
    guessed = x.argmax(axis=1)
    fractions = []
    for c in range(classes):
        indices = np.flatnonzero(y == c)
        demand(len(indices) > 0, 'native class absent')
        correct = sum(int(guessed[i] == c) for i in indices)
        fractions.append(F(correct, len(indices)))
    return sum(fractions, F(0)) * F(100, classes)


def cross_entropy(logits, labels):
    # Independently expressed float64 log-softmax; no engine/scorer import.
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    maxima = np.amax(x, axis=1)
    log_partition = np.log(np.sum(np.exp(x - maxima[:, None]), axis=1)) + maxima
    return float(np.mean(log_partition - x[np.arange(y.size), y]))


def measure(unit, arrays, rows):
    expected = {'epochs'} | {g + '__' + field for g in ('A', 'B', 'outer')
                            for field in ('paths', 'labels', 'all_epoch_logits')}
    demand(set(arrays) == expected, 'unexpected NPZ fields')
    total, k = unit['config']['epochs'], unit['n_classes']
    epochs = np.asarray(arrays['epochs'])
    demand(epochs.dtype.kind in 'iu' and epochs.tolist() == list(range(1, total + 1)), 'missing epoch prefix')
    measured = {}
    for group in ('A', 'B', 'outer'):
        p, y, x = (np.asarray(arrays[group + '__' + name]) for name in ('paths', 'labels', 'all_epoch_logits'))
        demand(p.dtype.kind == 'U' and p.tolist() == unit['report'][group], 'path identity/order differs')
        demand(y.dtype.kind in 'iu' and y.tolist() == [rows[v]['label_index'] for v in unit['report'][group]], 'label identity differs')
        demand(x.shape == (total, len(y), k) and x.dtype.kind == 'f' and np.isfinite(x).all(), 'incomplete prediction cube')
        measured[group] = [(cross_entropy(frame, y) if group != 'outer' else None, class_recall(frame, y, k))
                           for frame in x]
    selected = {}
    for window in unit['windows']:
        chosen = {'last': window}
        for group, name in (('A', 'seen'), ('B', 'unseen')):
            ce = [v[0] for v in measured[group][:window]]
            uar = [v[1] for v in measured[group][:window]]
            chosen[name + '_ce'] = ce.index(min(ce)) + 1
            chosen[name + '_uar'] = uar.index(max(uar)) + 1
        selected[str(window)] = chosen
    return measured, selected


def exact_mean(values):
    return sum(values, F(0)) / len(values)


def estimate(values, inferential):
    demand(len(values) == 24, 'twenty-four draw means required')
    mean = float(exact_mean(values))
    # Variance uses exact rational centering and math.fsum, not numpy.std.
    centered = [float(v - exact_mean(values)) for v in values]
    sd = math.sqrt(math.fsum(v*v for v in centered) / 23)
    se = sd / math.sqrt(24)
    result = dict(mean_pp=mean, sd_draw_pp=sd, se_pp=se, df=23, n_draws=24,
                  pointwise_95_ci_pp=None, p_two_sided=None, holm_p_ten=None, t_statistic=None)
    if sd:
        half = float(student_t.ppf(.975, 23)) * se
        result['pointwise_95_ci_pp'] = [mean-half, mean+half]
        if inferential:
            result['t_statistic'] = mean / se
            result['p_two_sided'] = float(2 * student_t.sf(abs(mean / se), 23))
    return result


def adjusted_p_ten(values):
    demand(len(values) == 10, 'exactly ten Holm slots required')
    normalized = [1.0 if p is None else p for p in values]
    demand(all(math.isfinite(p) and 0 <= p <= 1 for p in normalized), 'invalid raw p')
    ordered = sorted(enumerate(normalized), key=lambda pair: pair[1])
    out = [None] * 10
    for rank, (original, _) in enumerate(ordered):
        if values[original] is not None:
            out[original] = min(1.0, max((10-j)*ordered[j][1] for j in range(rank+1)))
    return out


def recompute(plan, loader):
    indexed = full_grid(plan)  # Must precede even the first loader call.
    by_context, unit_rows, selected_rows, curves = {}, [], [], []
    disagreements = {}
    for key in sorted(indexed):
        unit = indexed[key]; c, d, fold, model = key
        arrays, receipt_selection = loader(unit)
        series, selected = measure(unit, arrays, plan['rows'][c])
        demand(selected == receipt_selection, 'checkpoint choices differ from committed receipt')
        common = dict(unit_id=unit['unit_id'], corpus=c, draw=d, fold=fold, model=model)
        for e in range(1, unit['config']['epochs']+1):
            curves.append(dict(common, epoch=e, seen_ce=series['A'][e-1][0], unseen_ce=series['B'][e-1][0],
                               seen_uar=float(series['A'][e-1][1]), unseen_uar=float(series['B'][e-1][1]),
                               outer_uar=float(series['outer'][e-1][1])))
        for window in unit['windows']:
            choice = selected[str(window)]
            target = {r: series['outer'][choice[r]-1][1] for r in RULES}
            ce = target['seen_ce'] - target['unseen_ce']
            ua = target['seen_uar'] - target['unseen_uar']
            metrics = dict(D_CE=ce, D_UAR=ua, J=ce-ua, **{'T_'+r: target[r] for r in RULES})
            by_context[c, d, fold, model, window] = metrics
            unit_rows.append(dict(common, window=window,
                                  **{m+'_pp': float(v) for m, v in metrics.items()},
                                  **{'epoch_'+r: choice[r] for r in RULES},
                                  ce_epoch_disagreement=int(choice['seen_ce'] != choice['unseen_ce']),
                                  uar_epoch_disagreement=int(choice['seen_uar'] != choice['unseen_uar'])))
            oracle = max(p[1] for p in series['outer'][:window])
            for r in RULES:
                selected_rows.append(dict(common, window=window, rule=r, selected_epoch=choice[r],
                                          outer_uar=float(target[r]), oracle_outer_uar=float(oracle),
                                          oracle_shortfall_pp=float(oracle-target[r]), oracle_used_for_selection=False))
            for label, effect in (('CE', ce), ('UAR', ua)):
                differing = choice['seen_'+label.lower()] != choice['unseen_'+label.lower()]
                demand(differing or effect == 0, 'same checkpoint has different common-test scores')
                disagreements.setdefault((c, model, window, label), []).append((differing, effect))
    draw_values, draws = {}, []
    for c in CORPORA:
        settings = [('model_window', 'hubert_base', 15), ('model_window', 'wavlm_base_plus', 15),
                    ('hubert_minus_wavlm15', 'hubert_minus_wavlm', 15)]
        if c != 'cremad':
            settings += [('model_window', 'wavlm_base_plus', 45),
                         ('window45_minus15', 'wavlm_base_plus', '45_minus15')]
        for kind, model, window in settings:
            for d in range(24):
                values = {}
                for metric in METRICS:
                    pieces = []
                    for f in range(5):
                        if kind == 'model_window':
                            piece = by_context[c, d, f, model, window][metric]
                        elif kind == 'hubert_minus_wavlm15':
                            piece = by_context[c, d, f, 'hubert_base', 15][metric] - by_context[c, d, f, 'wavlm_base_plus', 15][metric]
                        else:
                            piece = by_context[c, d, f, model, 45][metric] - by_context[c, d, f, model, 15][metric]
                        pieces.append(piece)
                    values[metric] = sum(pieces, F(0)) / 5
                draw_values[c, kind, model, window, d] = values
                draws.append(dict(corpus=c, comparison=kind, model=model, window=window, draw=d,
                                  **{m+'_pp': float(v) for m, v in values.items()}))
    main = []
    for ident, c, m, kind, w, endpoint, metric in family():
        row = estimate([draw_values[c, kind, m, w, d][metric] for d in range(24)], True)
        main.append(dict(id=ident, corpus=c, model=m, endpoint=endpoint, unit='percentage_points', **row))
    for row, corrected in zip(main, adjusted_p_ten([r['p_two_sided'] for r in main])):
        row['holm_p_ten'] = corrected
        row['reject_familywise_05'] = corrected is not None and corrected <= .05
    descriptive = []
    for c, kind, model, window in sorted({key[:4] for key in draw_values}, key=str):
        for metric in METRICS:
            values = [draw_values[c, kind, model, window, d][metric] for d in range(24)]
            descriptive.append(dict(corpus=c, comparison=kind, model=model, window=window, endpoint=metric,
                                    unit='UAR_percent' if kind == 'model_window' and metric.startswith('T_') else 'percentage_points',
                                    hypothesis_test=False, **estimate(values, False)))
    identities = []
    for (c, model, window, rule), entries in sorted(disagreements.items()):
        demand(len(entries) == 120, 'q identity requires full 120-context panel')
        different = [effect for flag, effect in entries if flag]
        q = F(len(different), 120); overall = exact_mean([effect for _, effect in entries])
        conditional = exact_mean(different) if different else None
        product = q * conditional if different else F(0)
        demand(product == overall, 'exact q times conditional-effect identity failed')
        identities.append(dict(corpus=c, model=model, window=window, rule=rule,
                               n_contexts=120, n_disagree=len(different), q=float(q),
                               conditional_effect_pp=None if conditional is None else float(conditional),
                               overall_effect_pp=float(overall), product_pp=float(product), exact_identity=True,
                               inference='descriptive algebra only; not a causal mechanism or independent Bernoulli sample'))
    return dict(primary_tests=main, descriptive=descriptive, units=unit_rows, draws=draws,
                selected_epochs=selected_rows, curves=curves, q_times_e=identities)


def compare_value(actual, expected, location, errors):
    if isinstance(expected, bool) or expected is None:
        demand(actual is expected, 'logical/NA mismatch at ' + location)
    elif isinstance(expected, (int, float)):
        demand(type(actual) in (int, float) and math.isfinite(actual), 'invalid numeric at '+location)
        error = abs(float(actual) - expected); errors.append(error)
        demand(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11), 'numeric mismatch at '+location)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), 'list shape mismatch at '+location)
        for i, value in enumerate(expected): compare_value(actual[i], value, f'{location}/{i}', errors)
    else:
        demand(actual == expected, 'identity mismatch at '+location)


def parse_csv_cell(raw, expected):
    if raw == 'NA': return None
    if isinstance(expected, bool):
        return {'True': True, 'False': False}.get(raw, raw)
    if isinstance(expected, (int, float)): return float(raw)
    if isinstance(expected, list): return json.loads(raw)
    return raw


def compare_scores(replay, scores, plan, backup, pins):
    scores = Path(scores).resolve()
    manifest = read_bound(member(scores, 'OUTPUT_MANIFEST.json'), pins); seal_ok(manifest, 'manifest_sha256')
    demand(manifest['plan_sha256'] == plan['plan_sha256'], 'score manifest plan differs')
    for name, expected in manifest['files'].items(): pin(member(scores, name), expected, pins)
    result = read_bound(member(scores, 'results.json'), pins); seal_ok(result, 'result_sha256')
    demand(result['plan_sha256'] == plan['plan_sha256'], 'score plan differs')
    demand(result['counts'] == dict(formal_units=720, unit_windows=960, full_epoch_curve_rows=18000,
                                   main_tests=10, draws_per_test=24, folds_per_draw=5, pilots_included=0),
           'frozen score count metadata differs')
    input_audit = read_bound(member(scores, 'input_audit.json'), pins)
    demand(sha(member(scores, 'input_audit.json')) == result['inputs']['input_audit_sha256']
           and input_audit['plan_sha256'] == plan['plan_sha256']
           and input_audit['verified_units'] == input_audit['expected_units'] == 720,
           'frozen scorer input audit differs')
    # The same bytes must underpin the two pipelines, not another run with the same grid.
    frozen_pins = input_audit['file_sha256']
    lock = read_bound(member(backup, 'SOURCE_LOCK.json'), pins)
    demand(input_audit['source_lock_sha256'] == lock['source_lock_sha256'], 'scorer source-lock binding differs')
    for relative in ['units/' + u['unit_id'] + '/' + n for u in plan['units']
                     for n in ('predictions.npz', 'history.json', 'receipt.json', 'COMPLETE.json')]:
        path = str(member(backup, relative).resolve())
        demand(frozen_pins.get(path) == pins.get(path), 'scorer/replay artifact identity differs')
    errors = []
    for key, identity_fields in (
        ('primary_tests', ('id',)), ('descriptive', ('corpus', 'comparison', 'model', 'window', 'endpoint'))):
        theirs = result['tests' if key == 'primary_tests' else key]
        index = {tuple(row[k] for k in identity_fields): row for row in theirs}
        demand(len(index) == len(theirs) == len(replay[key]), 'result row count differs')
        for expected in replay[key]:
            actual = index[tuple(expected[k] for k in identity_fields)]
            for field, value in expected.items(): compare_value(actual[field], value, key+'/'+str(field), errors)
    csv_keys = dict(units=('unit_id', 'window'), draws=('corpus', 'comparison', 'model', 'window', 'draw'),
                    primary_tests=('id',), descriptive=('corpus', 'comparison', 'model', 'window', 'endpoint'),
                    selected_epochs=('unit_id', 'window', 'rule'), curves=('unit_id', 'epoch'))
    counts = {}
    for table, keys in csv_keys.items():
        filename = table + '.csv'
        demand(result['tables'][filename] == manifest['files'][filename], 'table identity differs')
        with member(scores, filename).open(encoding='utf-8', newline='') as handle:
            raw = list(csv.DictReader(handle))
        indexed = {tuple(row[k] for k in keys): row for row in raw}
        demand(len(indexed) == len(raw) == len(replay[table]), 'CSV row count/duplicate mismatch')
        for row in replay[table]:
            other = indexed[tuple(str(row[k]) for k in keys)]
            for field, value in row.items():
                compare_value(parse_csv_cell(other[field], value), value, filename+'/'+field, errors)
        counts[table] = len(raw)
    return dict(compared_scalar_numbers=len(errors), max_abs_difference=max(errors, default=0.),
                rows=counts, numerical_tolerance=dict(relative=1e-9, absolute=1e-11))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('backup', 'scores', 'repo', 'out'): parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    demand(not out.exists(), 'output must be a new directory')
    for source in (args.backup.resolve(), args.scores.resolve(), args.repo.resolve()):
        demand(not out.is_relative_to(source) and not source.is_relative_to(out), 'output overlaps inputs')
    plan, receipts, pins = audit_backup(args.backup, args.repo)
    pins[str(Path(__file__).resolve())] = sha(__file__)
    def loader(unit):
        path = member(args.backup, 'units/' + unit['unit_id'] + '/predictions.npz')
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        return arrays, receipts[unit['unit_id']]['info']['selected_epochs']
    replay = recompute(plan, loader)
    comparison = compare_scores(replay, args.scores, plan, args.backup, pins)
    for path, expected in pins.items(): demand(sha(path) == expected, 'input changed during independent replay')
    result = dict(schema='ser-autodl-independent-replay-1', passed=True, formal_units=720,
                  plan_sha256=plan['plan_sha256'], source_sha256=sha(__file__), comparison=comparison,
                  primary_tests=replay['primary_tests'], descriptive=replay['descriptive'],
                  q_times_e=replay['q_times_e'],
                  scope='independent numerical implementation on complete archived logits; no new training, '
                        'no new hypotheses, all negative/imprecise outcomes retained; q times e is descriptive algebra')
    result['audit_sha256'] = digest(result)
    out.mkdir(parents=True)
    (out / 'independent_replay.json').write_bytes(canonical(result)+b'\n')
    (out / 'input_sha256.json').write_bytes(canonical(pins)+b'\n')
    print(json.dumps(dict(passed=True, formal_units=720, main_tests=10, **comparison)))


if __name__ == '__main__': main()

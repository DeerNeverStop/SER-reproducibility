"""Descriptive old N14R2 decomposition; no new tests, training or inference.

Reads a pre-existing release staging archive, verifies its byte commitments,
reselects all 240 HPO episodes from 1,920 histories, and recomputes selected/c4
outer UAR from the archived CSVs. Only small source copies and a new report are
written. This is not a new full raw-audio/cache/weight gate.
"""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import tarfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def mean(values):
    values = list(values)
    require(values and all(math.isfinite(x) for x in values), 'invalid mean')
    return math.fsum(values) / len(values)


def close(left, right):
    require(math.isclose(left, right, rel_tol=0, abs_tol=1e-10), f'numeric mismatch: {left} != {right}')


def run(source_repo, evidence, output):
    base = source_repo/'v2_1/n14r2'
    stage = base/'release_staging/SER26-N14R2-complete-20260906'
    require(not output.exists(), 'output must be new')
    manifest = {}
    for line in (stage/'RELEASE_SHA256SUMS').read_text(encoding='utf-8').splitlines():
        h, name = line.split('  ', 1)
        require(name not in manifest and Path(name).name == name, 'unsafe release inventory')
        manifest[name] = h
    # These identities were already retained in the prior completed delivery.
    anchors = {
        'n14r2_results.metadata-corrected.json': '92146e7dcf01cc8fd923f78d08017e638d501732ff23110008caa64b5f6ef6c6',
        'n14r2_verification.metadata-corrected.json': 'e82f088d88770e6a201ad93b581a5460aa9fba27658a9810ba63a9ed115cf843',
        'analysis_lock.json': 'e471cb3c2a3b26e799d8f85cbab86f9818b656db356b86eb3524a0e0ec0c3b82',
        'n14r2-closed-run.tar': '162e209301252397c6d7bf79ae8c3a781fca93f8ef7dbe2babace94d4f44f9e9',
    }
    require(all(manifest[k] == v for k, v in anchors.items()), 'prior release anchor differs')
    for name, h in manifest.items():
        require(sha(stage/name) == h, 'release hash mismatch: '+name)
    evidence.mkdir(parents=True, exist_ok=True)
    copy_names = ['RELEASE_SHA256SUMS', 'analysis_lock.json', 'completion.json',
                  'source-bindings.json', 'analysis-verification-receipt.json',
                  'n14r2_results.metadata-corrected.json', 'n14r2_results.original.json',
                  'n14r2_verification.metadata-corrected.json', 'n14r2_metadata-erratum.json']
    for name in copy_names:
        target = evidence/name
        if target.exists():
            require(sha(target) == sha(stage/name), 'existing evidence differs; never overwrite: '+name)
        else:
            with target.open('xb') as f:
                f.write((stage/name).read_bytes())
    result = read(stage/'n14r2_results.metadata-corrected.json')
    original = read(stage/'n14r2_results.original.json')
    require(original.pop('n_planned_draws') == 24 and result['n_planned_draws'] == 28
            and original == {k: v for k, v in result.items() if k != 'n_planned_draws'}, 'erratum is not metadata-only')
    lock, completed = read(stage/'analysis_lock.json'), read(stage/'completion.json')
    check = dict(lock); payload = check.pop('lock_payload_sha256')
    require(digest(check) == payload == completed['analysis_lock_payload_sha256'], 'lock payload differs')
    require(completed['analysis_lock_sha256'] == manifest['analysis_lock.json'], 'completion lock differs')
    require(result['analysis_draw_ids'] == lock['complete_draw_ids'] == completed['complete_draw_ids'] == list(range(24)), 'wrong draws')
    require(lock['n_locked_units'] == 1920 and completed['status'] == 'complete'
            and not completed['blocked_draw_ids'] and not completed['void_draws'], 'incomplete old study')
    verification = read(stage/'n14r2_verification.metadata-corrected.json')
    require(verification['pass'] is True and verification['differences'] == [], 'prior corrected verifier did not pass')
    require(verification['reconstructed'] == result, 'prior reconstruction differs from result')
    bindings = read(stage/'source-bindings.json')
    require(bindings['frozen_commit'] == '4fe0bb52f15b56a6626363be31a378b0f9661293', 'wrong historical source')
    for name in ('score.py', 'verify.py', 'PINS.json'):
        require(sha(base/name) == bindings['sha256'][name], 'current historical source differs: '+name)
        committed = subprocess.check_output(['git', 'show', bindings['frozen_commit']+':v2_1/n14r2/'+name], cwd=source_repo)
        require(hashlib.sha256(committed).hexdigest() == bindings['sha256'][name], 'Git source identity differs: '+name)
    require(sha(base/'spec.json') == lock['preregistration_hashes']['spec.json'], 'spec differs')
    require(sha(base/'plan/run_plan.csv') == lock['plan_hashes']['run_plan.csv'], 'plan differs')
    spec = read(base/'spec.json')
    require(spec['descriptors']['D09R2']['fixed_config_index'] == 4, 'wrong fixed comparator')
    with (base/'plan/run_plan.csv').open(encoding='utf-8', newline='') as f:
        all_rows = list(csv.DictReader(f))
    rows = [row for row in all_rows if int(row['draw_id']) in range(24)]
    require(len(all_rows) == 2240 and len(rows) == len({row['unit_id'] for row in rows}) == 1920, 'plan extent differs')
    require(set(lock['unit_artifact_hashes']) == {row['unit_id'] for row in rows}, 'locked unit set differs')
    expected = {(d, f, cell, c) for d in range(24) for f in range(5)
                for cell in ('GR_hpo', 'GG_hpo') for c in range(8)}
    require({(int(r['draw_id']), int(r['fold']), r['cell'], int(r['config_index'])) for r in rows} == expected
            and all(r['train_rep'] == '0' for r in rows), 'episode grid differs')
    checked_members, candidates, cached_predictions = {}, defaultdict(list), {}
    with tarfile.open(stage/'n14r2-closed-run.tar', 'r:') as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), 'duplicate archive member')
        def member(uid, name):
            binding = lock['unit_artifact_hashes'][uid]
            path = binding['attempt_path']+'/'+name
            entry = archive.getmember(path)
            require(entry.isfile(), 'committed payload is not regular file')
            raw = archive.extractfile(entry).read()
            require(hashlib.sha256(raw).hexdigest() == binding['artifact_hashes'][name], 'unit artifact differs: '+path)
            checked_members[path] = binding['artifact_hashes'][name]
            return raw
        for row in rows:
            uid = row['unit_id']
            history, unit = json.loads(member(uid, 'history.json')), json.loads(member(uid, 'unit.json'))
            require(unit['unit_id'] == uid and unit['config_sha256'] == row['config_sha256']
                    and unit['cell'] == row['cell'] and unit['fold'] == int(row['fold'])
                    and unit['r'] == int(row['r']) and unit['status'] == 'done', 'unit identity differs')
            require(history and [h['epoch'] for h in history] == list(range(1, len(history)+1))
                    and unit['epochs_run'] == len(history), 'history extent differs')
            require(all(math.isfinite(h['val_loss']) and 0 <= h['val_uar'] <= 100 for h in history), 'bad val history')
            best = min(history, key=lambda h: (h['val_loss'], h['epoch']))
            require(unit['best_epoch'] == best['epoch'], 'checkpoint differs')
            close(unit['val_loss_best'], best['val_loss']); close(unit['val_uar_best'], best['val_uar'])
            key = (int(row['draw_id']), int(row['fold']), row['cell'])
            candidates[key].append(dict(unit_id=uid, config_index=int(row['config_index']),
                                        V=best['val_uar'], best_epoch=best['epoch'], n_test=int(row['n_test'])))
        def test_metric(candidate):
            uid = candidate['unit_id']
            if uid in cached_predictions:
                return cached_predictions[uid]
            records = list(csv.DictReader(io.StringIO(member(uid, 'predictions.csv').decode('utf-8'))))
            require(len(records) == candidate['n_test'], 'outer size differs')
            truth, counts, correct = {}, Counter(), Counter()
            for row in records:
                y, pred = int(row['y_true']), int(row['y_pred'])
                require(row['relative_path'] not in truth and y in range(6) and pred in range(6), 'invalid prediction identity')
                logits = [float(row['logit_'+str(c)]) for c in range(6)]
                require(all(math.isfinite(x) for x in logits) and max(range(6), key=lambda c: logits[c]) == pred, 'argmax differs')
                truth[row['relative_path']] = y
                counts[y] += 1; correct[y] += int(y == pred)
            require(set(counts) == set(range(6)), 'missing native class')
            uar = float(sum((Fraction(correct[c], counts[c]) for c in range(6)), Fraction())*100/6)
            cached_predictions[uid] = (uar, truth)
            return uar, truth
        fold_records = []
        for draw in range(24):
            for fold in range(5):
                record = dict(draw=draw, fold=fold, selected={}, fixed_c4={})
                common_truth = None
                stored = result['draws'][str(draw)]['folds'][str(fold)]['train_reps']['0']
                for cell in ('GR_hpo', 'GG_hpo'):
                    pool = candidates[draw, fold, cell]
                    require(sorted(c['config_index'] for c in pool) == list(range(8)), 'candidate set differs')
                    selected = min(pool, key=lambda c: (-c['V'], c['config_index']))
                    fixed = next(c for c in pool if c['config_index'] == 4)
                    require(selected['unit_id'] == stored[cell]['selected_unit_id']
                            and selected['config_index'] == stored[cell]['selected_config_index'], 'HPO winner differs')
                    for rule, candidate in [('selected', selected), ('fixed_c4', fixed)]:
                        T, truth = test_metric(candidate)
                        require(common_truth is None or common_truth == truth, 'outer members/labels are not identical')
                        common_truth = truth
                        record[rule][cell] = {**candidate, 'T': T, 'V_minus_T': candidate['V']-T}
                    close(selected['V'], stored[cell]['val_uar'])
                    close(record['selected'][cell]['T'], stored[cell]['test_uar'])
                    close(record['fixed_c4'][cell]['T'], stored[cell]['fixed_index4_test_uar'])
                for rule in ('selected', 'fixed_c4'):
                    gr, gg = record[rule]['GR_hpo'], record[rule]['GG_hpo']
                    record[rule]['delta_V'] = gr['V']-gg['V']
                    record[rule]['delta_T'] = gr['T']-gg['T']
                    record[rule]['delta_gap'] = gr['V_minus_T']-gg['V_minus_T']
                record['D09R2'] = record['selected']['delta_T']-record['fixed_c4']['delta_T']
                close(record['selected']['delta_gap'], stored['z']); close(record['D09R2'], stored['d09r'])
                fold_records.append(record)
    draw_records = []
    summary = {}
    for draw in range(24):
        part = [r for r in fold_records if r['draw'] == draw]
        dd = dict(draw=draw)
        for rule in ('selected', 'fixed_c4'):
            dd[rule] = {cell: {k: mean(r[rule][cell][k] for r in part) for k in ('V', 'T', 'V_minus_T')}
                        for cell in ('GR_hpo', 'GG_hpo')}
            dd[rule].update({k: mean(r[rule][k] for r in part) for k in ('delta_V', 'delta_T', 'delta_gap')})
        dd['D09R2'] = mean(r['D09R2'] for r in part)
        close(dd['selected']['delta_gap'], result['draws'][str(draw)]['z'])
        close(dd['D09R2'], result['draws'][str(draw)]['d09r'])
        draw_records.append(dd)
    for rule in ('selected', 'fixed_c4'):
        summary[rule] = {cell: {k: mean(d[rule][cell][k] for d in draw_records) for k in ('V', 'T', 'V_minus_T')}
                         for cell in ('GR_hpo', 'GG_hpo')}
        summary[rule].update({k: mean(d[rule][k] for d in draw_records) for k in ('delta_V', 'delta_T', 'delta_gap')})
    summary['D09R2'] = mean(d['D09R2'] for d in draw_records)
    summary['selected_minus_fixed_delta_gap'] = summary['selected']['delta_gap']-summary['fixed_c4']['delta_gap']
    close(summary['selected']['delta_gap'], result['primary']['mean'])
    close(summary['D09R2'], result['D09R2']['mean'])
    audit = dict(schema='n14r2-descriptive-decomposition-audit-1', passed=True,
                 analysis_status='post-hoc descriptive decomposition; no new confidence intervals or hypothesis tests',
                 source_repo=str(source_repo), release_staging=str(stage), release_tag='SER26-N14R2-complete-20260906',
                 frozen_commit=bindings['frozen_commit'], release_manifest_sha256=sha(stage/'RELEASE_SHA256SUMS'),
                 release_files_verified=manifest, source_sha256=bindings['sha256'],
                 audit_script_sha256=sha(__file__), n_draws=24, folds_per_draw=5, n_histories=1920,
                 n_selected_episodes=240, n_fixed_episodes=240, n_unique_outer_csvs=len(cached_predictions),
                 matched_outer_sets=120, checked_tar_member_sha256=checked_members,
                 selected_same_config_between_cells=sum(r['selected']['GR_hpo']['config_index']==r['selected']['GG_hpo']['config_index'] for r in fold_records),
                 selected_is_c4_counts={cell:sum(r['selected'][cell]['config_index']==4 for r in fold_records) for cell in ('GR_hpo','GG_hpo')},
                 summary=summary, draws=draw_records, folds=fold_records,
                 scope='Reconstructs all validation HPO winners from pinned histories and selected/c4 T from native six-class saved predictions; '
                       'prior complete verifier is checked as evidence, not rerun; no raw audio, feature-cache validation, model inference, weights or new significance tests.')
    with output.open('x', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_repo.resolve(), args.evidence.resolve(), args.out.resolve())
    print(json.dumps({k: result[k] for k in ('passed', 'n_histories', 'n_unique_outer_csvs', 'summary')}, indent=2))

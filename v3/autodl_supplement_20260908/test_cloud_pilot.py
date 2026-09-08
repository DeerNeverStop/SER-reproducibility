"""Tiny synthetic admission fixtures; no actual models, audio or cloud reads."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from . import cloud_pilot as gate
from .engine import digest, file_sha

LOCK = 'a' * 64


def write_json(path, value):
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')


def reseal_receipt(directory, receipt):
    write_json(directory / 'receipt.json', receipt)
    write_json(directory / 'COMPLETE.json', dict(receipt_sha256=file_sha(directory / 'receipt.json'),
               unit_id=receipt['unit_id'], unit_sha256=receipt['unit_sha256'], source_lock_sha256=LOCK,
               plan_sha256=receipt['plan_sha256']))


def fixture(tmp_path, monkeypatch):
    # The production validate_plan checks the full real 720 metadata manifests.
    # This test replaces ONLY that prerequisite with a synthetic semantic seal;
    # all eight-pilot inventory, artifact and technical checks run unchanged.
    monkeypatch.setattr(gate, 'validate_plan', lambda p: gate.require(
        p['plan_sha256'] == digest({k: v for k, v in p.items() if k != 'plan_sha256'}), 'bad fixture seal'))
    root = tmp_path / 'runs'
    root.mkdir()
    env = dict(schema='ser-autodl-environment-1', backend='autodl', synthetic_fixture=True, vram_bytes=1000)
    env['environment_sha256'] = digest(env)
    write_json(root / 'ENVIRONMENT.json', env)
    pilots, rows = [], {}
    for ci, corpus in enumerate(gate.CORPORA):
        classes = 6 + ci
        corpus_rows = {}
        report = {g: [f'{g}/{c}.wav' for c in range(classes)] for g in gate.GROUPS}
        for paths in report.values():
            for c, path in enumerate(paths):
                corpus_rows[path] = dict(label_index=c)
        rows[corpus] = corpus_rows
        for model in gate.MODELS:
            budgets = [15, 45] if model == 'wavlm_base_plus' and corpus != 'cremad' else [15]
            for epochs in budgets:
                unit = dict(unit_id=f'SYNTHETIC_{corpus}_{model}_{epochs}', phase='technical_pilot',
                            corpus=corpus, model=model, arm='A', backend_required='autodl', draw=23, fold=4,
                            n_classes=classes, fit=['fit/0.wav'], report=report,
                            report_batches={g: [p] for g, p in report.items()},
                            seeds=dict(head=123 + ci, crop=23, order=4, torch_training=6),
                            reference_panel_sha256=digest(corpus),
                            permanent_checkpoint_sample=True,
                            config=dict(epochs=epochs, batch_size=16), windows=[15] if epochs == 15 else [15, 45])
                unit['unit_sha256'] = digest(unit)
                pilots.append(unit)
    formal = [dict(corpus=c, model=m, unit_id=f'formal_{c}_{m}_{i}')
              for c in gate.CORPORA for m in gate.MODELS for i in range(120)]
    plan = dict(rows=rows, pilots=pilots, units=formal)
    plan['plan_sha256'] = digest(plan)
    for unit in pilots:
        directory = root / 'pilots' / unit['unit_id']
        directory.mkdir(parents=True)
        epochs, classes = unit['config']['epochs'], unit['n_classes']
        arrays = dict(epochs=np.arange(1, epochs + 1, dtype=np.int64))
        for group in gate.GROUPS:
            arrays[group + '__paths'] = np.asarray(unit['report'][group], dtype=str)
            arrays[group + '__labels'] = np.arange(classes, dtype=np.int64)
            arrays[group + '__all_epoch_logits'] = np.stack([
                np.eye(classes, dtype=np.float64) + e / 100 for e in range(1, epochs + 1)])
        np.savez_compressed(directory / 'predictions.npz', **arrays)
        history = [dict(epoch=e, optimizer_steps=1, scaler_skipped_steps=0,
                        train_loss=1 / e, **{r: .1 for r in gate.RULES}) for e in range(1, epochs + 1)]
        write_json(directory / 'history.json', history)
        # Deliberately not a torch file: this gate stat-checks checkpoint blobs,
        # while its prerequisite runner byte/restore gate owns their contents.
        (directory / 'checkpoint.pt').write_bytes(b'SYNTHETIC checkpoint already checked by prerequisite')
        files = {name: dict(sha256=file_sha(directory / name), bytes=(directory / name).stat().st_size)
                 for name in gate.FILES}
        selected = {str(w): dict(**{r: 1 for r in gate.RULES}, last=w) for w in unit['windows']}
        stored_epochs = sorted({1, *unit['windows']})
        differences = {f'{e}/{g}': 0. for e in stored_epochs for g in gate.GROUPS}
        info = dict(epochs=epochs, windows=unit['windows'], unit_sha256=unit['unit_sha256'],
                    model_identity=dict(name=unit['model']), initial_head_sha256=digest(['head', unit['corpus']]),
                    checkpoint_sha256=files['checkpoint.pt']['sha256'], checkpoint_reload_verified=True,
                    checkpoint_retained=True, outer_scores_computed=False, selected_epochs=selected,
                    fresh_process_restore_verified=True,
                    checkpoint_unique_epochs=stored_epochs, checkpoint_unique_epoch_count=len(stored_epochs),
                    reload_by_epoch_group=differences, reload_checks=len(differences),
                    reload_checked_values=len(stored_epochs) * 3 * classes * classes,
                    reload_max_abs_diff=0., wall_seconds=10. * epochs,
                    peak_cuda_allocated_bytes=300, peak_cuda_reserved_bytes=500,
                    optimizer_steps=epochs, scaler_skipped_steps=0)
        receipt = dict(schema='ser-autodl-receipt-1', unit_id=unit['unit_id'], phase=unit['phase'],
                       unit_sha256=unit['unit_sha256'], plan_sha256=plan['plan_sha256'],
                       source_lock_sha256=LOCK, environment_sha256=env['environment_sha256'], files=files, info=info)
        fresh = dict(schema='ser-autodl-fresh-restore-1', passed=True,
                     **{k: receipt[k] for k in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256', 'environment_sha256')},
                     checkpoint_sha256=files['checkpoint.pt']['sha256'], predictions_sha256=files['predictions.npz']['sha256'],
                     comparisons=differences, max_abs_diff=0.)
        write_json(directory / 'fresh_restore.json', fresh)
        files['fresh_restore.json'] = dict(sha256=file_sha(directory / 'fresh_restore.json'),
                                           bytes=(directory / 'fresh_restore.json').stat().st_size)
        reseal_receipt(directory, receipt)
    return plan, root


def update_prediction_binding(directory, receipt):
    fresh = json.loads((directory / 'fresh_restore.json').read_text())
    fresh['predictions_sha256'] = receipt['files']['predictions.npz']['sha256']
    write_json(directory / 'fresh_restore.json', fresh)
    receipt['files']['fresh_restore.json'] = dict(sha256=file_sha(directory / 'fresh_restore.json'),
                                                 bytes=(directory / 'fresh_restore.json').stat().st_size)


def target(plan, root, corpus='subesco', model='wavlm_base_plus', epochs=15):
    unit = next(u for u in plan['pilots'] if (u['corpus'], u['model'], u['config']['epochs']) == (corpus, model, epochs))
    directory = root / 'pilots' / unit['unit_id']
    return unit, directory, json.loads((directory / 'receipt.json').read_text())


def test_all_eight_gate_and_six_group_time_excluding_extra15(tmp_path, monkeypatch):
    plan, root = fixture(tmp_path, monkeypatch)
    result = gate.evaluate(plan, root, LOCK, 1000)
    assert result['pass'] and result['formal_allowed']
    assert len(result['receipts_sha256']) == 8 and len(result['prefix_checks']) == 2
    assert len(result['head_checks']) == 3 and all(c['pass_check'] for c in result['head_checks'])
    # Four short primary groups, two long primary groups; exclude two short companions.
    expected = (4 * 150 + 2 * 450) * 120
    assert result['timing']['estimated_formal_fit_wall_seconds'] == expected
    assert result['timing']['with_25pct_margin_seconds'] == expected * 1.25
    assert result['outer_scores_computed'] is False
    assert result['pilot_gate_sha256'] == digest({k: v for k, v in result.items() if k != 'pilot_gate_sha256'})
    assert 'rehash large' in result['checkpoint_scope']


@pytest.mark.parametrize('kind', ['logits', 'history', 'selection', 'head', 'memory'])
def test_complete_but_technically_bad_blocks_formal(tmp_path, monkeypatch, kind):
    plan, root = fixture(tmp_path, monkeypatch)
    unit, directory, receipt = target(plan, root)
    if kind == 'logits':
        with np.load(directory / 'predictions.npz') as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        arrays['outer__all_epoch_logits'][0, 0, 0] += 1.01e-5
        np.savez_compressed(directory / 'predictions.npz', **arrays)
        path = directory / 'predictions.npz'
        receipt['files'][path.name] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
        update_prediction_binding(directory, receipt)
    elif kind == 'history':
        path = directory / 'history.json'
        history = json.loads(path.read_text())
        history[0]['train_loss'] += 1.01e-5
        write_json(path, history)
        receipt['files'][path.name] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
        update_prediction_binding(directory, receipt)
    elif kind == 'selection':
        receipt['info']['selected_epochs']['15']['seen_ce'] = 15  # still in saved union
    elif kind == 'head':
        receipt['info']['initial_head_sha256'] = 'c' * 64
    else:
        receipt['info']['peak_cuda_reserved_bytes'] = 901
    reseal_receipt(directory, receipt)
    result = gate.evaluate(plan, root, LOCK, 1000)
    assert not result['pass'] and not result['formal_allowed']


@pytest.mark.parametrize('kind', ['missing', 'extra', 'bytes', 'complete', 'source', 'environment', 'reload', 'labels'])
def test_corrupt_or_incomplete_evidence_rejected(tmp_path, monkeypatch, kind):
    plan, root = fixture(tmp_path, monkeypatch)
    unit, directory, receipt = target(plan, root)
    if kind == 'missing':
        (directory / 'predictions.npz').unlink()
    elif kind == 'extra':
        (root / 'pilots/EXTRA').mkdir()
    elif kind == 'bytes':
        with (directory / 'history.json').open('ab') as handle:
            handle.write(b' ')
    elif kind == 'complete':
        complete = json.loads((directory / 'COMPLETE.json').read_text())
        complete['receipt_sha256'] = 'f' * 64
        write_json(directory / 'COMPLETE.json', complete)
    elif kind == 'source':
        receipt['source_lock_sha256'] = 'f' * 64
        reseal_receipt(directory, receipt)
    elif kind == 'environment':
        receipt['environment_sha256'] = 'f' * 64
        reseal_receipt(directory, receipt)
    elif kind == 'reload':
        receipt['info']['reload_checks'] -= 1
        reseal_receipt(directory, receipt)
    else:
        with np.load(directory / 'predictions.npz') as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        arrays['outer__labels'] = np.roll(arrays['outer__labels'], 1)
        np.savez_compressed(directory / 'predictions.npz', **arrays)
        path = directory / 'predictions.npz'
        receipt['files'][path.name] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
        reseal_receipt(directory, receipt)
    with pytest.raises(ValueError):
        gate.evaluate(plan, root, LOCK, 1000)


def test_subthreshold_prefix_numeric_difference_admitted_and_accounted(tmp_path, monkeypatch):
    plan, root = fixture(tmp_path, monkeypatch)
    _, directory, receipt = target(plan, root)
    with np.load(directory / 'predictions.npz') as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays['A__all_epoch_logits'][0, 0, 0] += .5e-5
    np.savez_compressed(directory / 'predictions.npz', **arrays)
    path = directory / 'predictions.npz'
    receipt['files'][path.name] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
    update_prediction_binding(directory, receipt)
    reseal_receipt(directory, receipt)
    result = gate.evaluate(plan, root, LOCK, 1000)
    assert result['pass']
    assert result['prefix_checks'][0]['logits_max_abs_by_role']['A'] == pytest.approx(.5e-5)

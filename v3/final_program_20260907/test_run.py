"""CPU-only artifact/recovery regressions; no corpus scores or large WavLM."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import sys

import numpy as np
import pytest
import torch

from v3.final_program_20260907 import run


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True), encoding='utf8')


@pytest.fixture
def committed(tmp_path, monkeypatch):
    unit = dict(unit_id='SYNTHETIC_ONLY', phase='pilot', corpus='toy', n_classes=6,
                arm='A', selection_enabled=True, prediction_epochs=list(range(1, 16)),
                fit=[], report={}, report_batches={}, config=dict(epochs=15, batch_size=16, fp16=True))
    rows = {}
    for group in ('fit', 'A', 'B', 'outer'):
        paths = [f'{group}/{c}.wav' for c in range(6)]
        if group == 'fit':
            unit['fit'] = paths
        else:
            unit['report'][group] = paths
            unit['report_batches'][group] = [paths]
        for c, p in enumerate(paths):
            rows[p] = dict(label_index=c, speaker='A' if group == 'fit' else group,
                           sentence='fit-text' if group == 'fit' else 'query-text')
    lock = {'lock_sha256': 'c' * 64, 'phase': 'pilot'}
    folder = tmp_path/'units'/unit['unit_id']
    attempt = folder/'attempts'/'0001'
    attempt.mkdir(parents=True)
    arrays = {'epochs': np.arange(1, 16, dtype=np.int64)}
    measured = {}
    for group, peak in (('A', 4), ('B', 9), ('outer', 7)):
        arrays[group+'__paths'] = np.asarray(unit['report'][group], dtype=str)
        arrays[group+'__labels'] = np.arange(6, dtype=np.int64)
        arrays[group+'__all_epoch_logits'] = np.stack([
            np.eye(6, dtype=np.float32) * (20 - abs(epoch-peak)) for epoch in range(1, 16)])
        if group != 'outer':
            measured[group] = [run.metrics(x, np.arange(6), 6)
                               for x in arrays[group+'__all_epoch_logits']]
    selected = {'last': 15, 'seen_ce': 4, 'unseen_ce': 9, 'seen_uar': 1, 'unseen_uar': 1}
    history = [dict(epoch=i+1, train_loss=.2, optimizer_steps=1, scaler_skipped_steps=0,
                    seen_ce=measured['A'][i][0], unseen_ce=measured['B'][i][0],
                    seen_uar=float(measured['A'][i][1]), unseen_uar=float(measured['B'][i][1]))
               for i in range(15)]
    diffs = {f'{e}/{g}': 0. for e in set(selected.values()) for g in ('A', 'B', 'outer')}
    info = dict(epochs=15, wall_seconds=.1, reload_max_abs_diff=0.,
                outer_scores_computed=False, initial_state_sha256='a'*64,
                frozen_parameter_sha256='b'*64, selected_epochs=selected,
                reload_by_epoch_group=diffs)
    receipt = dict(schema='ser-final-program-receipt-1', unit_id=unit['unit_id'], phase='pilot',
                   unit_sha256=run.digest(unit), lock_sha256=lock['lock_sha256'], info=info,
                   peak_allocated_bytes=0, peak_reserved_bytes=0,
                   environment={'device': 'SYNTHETIC_CPU_ONLY'})
    spec = {'fc.weight': ((6, 2), torch.float32), 'fc.bias': ((6,), torch.float32),
            'enc.toy_buffer': ((), torch.int64)}
    monkeypatch.setattr(run, 'expected_delta_spec', lambda _: spec)
    checkpoint = dict(schema='ser-final-program-delta-1', unit_id=unit['unit_id'], n_classes=6,
                      selected_epochs=selected, initial_state_sha256='a'*64,
                      frozen_parameter_sha256='b'*64,
                      base_state_sha256='d8e745fb761c56d209431413bcc373f92784dc2c8d850199b6caa5cf05ca2c36',
                      trainable_parameter_names=['fc.weight', 'fc.bias'], buffer_names=['enc.toy_buffer'],
                      epoch_states={str(e): {k: torch.zeros(shape, dtype=dtype)
                                           for k, (shape, dtype) in spec.items()}
                                    for e in set(selected.values())})
    np.savez_compressed(attempt/'predictions.npz', **arrays)
    dump(attempt/'history.json', history)
    dump(attempt/'receipt.json', receipt)
    torch.save(checkpoint, attempt/'checkpoint.pt')
    done = dict(schema='ser-final-program-done-1', unit_id=unit['unit_id'],
                unit_sha256=run.digest(unit), lock_sha256=lock['lock_sha256'],
                attempt='attempts/0001', artifacts={})
    fixture = SimpleNamespace(unit=unit, lock=lock, rows=rows, out=tmp_path, folder=folder,
                              attempt=attempt, arrays=arrays, history=history,
                              receipt=receipt, checkpoint=checkpoint, done=done)
    repin(fixture)
    dump(folder/'reservations'/'0001.json', dict(unit_id=unit['unit_id'],
         unit_sha256=run.digest(unit), lock_sha256=lock['lock_sha256'], time=run.now()))
    return fixture


def repin(f):
    f.done['artifacts'] = {p.relative_to(f.folder).as_posix(): run.file_sha(p)
                           for p in f.attempt.iterdir()}
    dump(f.folder/'DONE', f.done)


def verify(f):
    return run.verify_done(f.unit, f.out, f.lock, f.rows)


def test_valid_candidate_is_verified_before_done_commit(committed):
    f = committed
    (f.folder/'DONE').unlink()
    done, receipt = run.verify_done(f.unit, f.out, f.lock, f.rows, done_doc=f.done)
    assert done == f.done and receipt['info']['epochs'] == 15
    assert not (f.folder/'DONE').exists()


def test_cross_attempt_unpinned_receipt_is_rejected(committed):
    f = committed
    dump(f.folder/'attempts'/'0002'/'receipt.json', f.receipt)
    f.done['attempt'] = 'attempts/0002'
    dump(f.folder/'DONE', f.done)
    with pytest.raises(ValueError, match='committed attempt'):
        verify(f)


@pytest.mark.parametrize('artifact', ['predictions.npz', 'checkpoint.pt'])
def test_hash_consistent_invalid_payload_is_rejected(committed, artifact):
    f = committed
    (f.attempt/artifact).write_bytes(b'not a valid serialized payload')
    repin(f)
    with pytest.raises(Exception):
        verify(f)


@pytest.mark.parametrize('fault', ['empty', 'missing_epoch', 'wrong_steps', 'too_many_skips'])
def test_hash_consistent_incomplete_history_is_rejected(committed, fault):
    f = committed
    if fault == 'empty':
        f.history.clear()
    elif fault == 'missing_epoch':
        f.history[-1]['epoch'] = 14
    elif fault == 'wrong_steps':
        f.history[0]['optimizer_steps'] = 0
    else:
        f.history[0]['scaler_skipped_steps'] = 2
    dump(f.attempt/'history.json', f.history)
    repin(f)
    with pytest.raises(ValueError):
        verify(f)


@pytest.mark.parametrize('field,value', [('epochs', 0), ('wall_seconds', -1),
    ('wall_seconds', float('nan')), ('reload_max_abs_diff', -1),
    ('reload_max_abs_diff', float('inf'))])
def test_invalid_extent_and_reload_self_reports_rejected(committed, field, value):
    f = committed
    f.receipt['info'][field] = value
    dump(f.attempt/'receipt.json', f.receipt)
    repin(f)
    with pytest.raises(ValueError):
        verify(f)


@pytest.mark.parametrize('fault', ['labels', 'paths', 'dtype', 'shape', 'epoch'])
def test_prediction_identity_and_support_rejected_even_after_rehash(committed, fault):
    f = committed
    if fault == 'labels':
        f.arrays['A__labels'][0] = 1
    elif fault == 'paths':
        f.arrays['A__paths'] = f.arrays['A__paths'][::-1]
    elif fault == 'dtype':
        f.arrays['A__all_epoch_logits'] = f.arrays['A__all_epoch_logits'].astype(np.float64)
    elif fault == 'shape':
        f.arrays['A__all_epoch_logits'] = f.arrays['A__all_epoch_logits'][:, :-1]
    else:
        f.arrays['epochs'][-1] = 14
    np.savez_compressed(f.attempt/'predictions.npz', **f.arrays)
    repin(f)
    with pytest.raises(ValueError):
        verify(f)


def test_resealed_selection_cannot_override_validation_winner(committed):
    f = committed
    f.receipt['info']['selected_epochs']['seen_ce'] = 7
    dump(f.attempt/'receipt.json', f.receipt)
    repin(f)
    with pytest.raises(ValueError, match='selected epochs'):
        verify(f)


@pytest.mark.parametrize('fault', ['missing', 'shape', 'dtype', 'nan'])
def test_actual_small_checkpoint_deltas_checked(committed, fault):
    f = committed
    delta = f.checkpoint['epoch_states']['15']
    if fault == 'missing':
        del delta['fc.bias']
    elif fault == 'shape':
        delta['fc.bias'] = torch.zeros(7)
    elif fault == 'dtype':
        delta['fc.bias'] = delta['fc.bias'].half()
    else:
        delta['fc.bias'][0] = float('nan')
    torch.save(f.checkpoint, f.attempt/'checkpoint.pt')
    repin(f)
    with pytest.raises(ValueError):
        verify(f)


def test_runtime_audio_same_name_wrong_bytes_rejected(tmp_path):
    audio = tmp_path/'audio'; audio.mkdir()
    p = audio/'example.wav'; p.write_bytes(b'original')
    unit = dict(corpus='toy', fit=['example.wav'], report={'A': [], 'B': [], 'outer': []})
    plan = {'rows': {'toy': {'example.wav': {'bytes': 8, 'sha256': run.file_sha(p)}}}}
    run.verify_audio_inputs(plan, [unit], {'toy': str(audio)}, tmp_path)
    p.write_bytes(b'altered!')  # Same path and length.
    with pytest.raises(ValueError, match='runtime audio identity'):
        run.verify_audio_inputs(plan, [unit], {'toy': str(audio)}, tmp_path)


def test_local_namespace_package_passes_and_foreign_module_fails(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[2]
    run.check_import_locations(repo)
    monkeypatch.setitem(sys.modules, 'v3._synthetic_foreign_module',
                        SimpleNamespace(__file__=str(tmp_path/'elsewhere.py')))
    with pytest.raises(ValueError, match='outside selected repository'):
        run.check_import_locations(repo)


def test_checked_plan_replays_not_only_self_hash(tmp_path, monkeypatch):
    plan = {'units': [{'unit_id': 'toy', 'phase': 'pilot'}]}
    plan['plan_sha256'] = run.plan_digest(plan)
    dump(tmp_path/'plan_snapshot.json', plan)
    lock = dict(phase='pilot', phase_units=['toy'], sources={},
                plan_sha256=plan['plan_sha256'], plan_byte_sha256=run.file_sha(tmp_path/'plan_snapshot.json'))
    lock['lock_sha256'] = run.digest(lock)
    dump(tmp_path/'SOURCE_LOCK.json', lock)
    monkeypatch.setattr(run, 'check_import_locations', lambda _: None)
    monkeypatch.setattr(run, 'sources', lambda *a: {})
    def reject(actual, repo):
        assert actual == plan
        raise ValueError('synthetic semantic replay rejects malformed plan')
    monkeypatch.setattr(run, 'validate_plan', reject)
    with pytest.raises(ValueError, match='semantic replay'):
        run.checked_plan(tmp_path, tmp_path)


def start(f):
    run.event(f.out, dict(event='unit_start', unit_id=f.unit['unit_id'], attempt='attempts/0001'))


def test_done_crash_window_recovers_only_once(committed):
    f = committed
    verify(f)
    start(f)
    run.reconcile_done(f.unit, f.out)
    first = (f.out/'ledger.jsonl').read_bytes()
    run.reconcile_done(f.unit, f.out)
    assert (f.out/'ledger.jsonl').read_bytes() == first
    records = run.ledger_rows(f.out)
    successes = [r for r in records if r['event'] == 'unit_done']
    assert len(successes) == 1 and successes[0]['done_sha256'] == run.file_sha(f.folder/'DONE')
    assert successes[0]['recovered_after_commit'] is True


@pytest.mark.parametrize('fault', ['no_start', 'failed', 'duplicate_start'])
def test_done_recovery_requires_unique_nonfailed_attempt(committed, fault):
    f = committed
    verify(f)
    if fault != 'no_start':
        start(f)
    if fault == 'failed':
        run.event(f.out, dict(event='unit_failed', unit_id=f.unit['unit_id'], attempt='attempts/0001'))
    elif fault == 'duplicate_start':
        start(f)
    with pytest.raises(ValueError):
        run.reconcile_done(f.unit, f.out)


def test_partial_ledger_is_preserved_and_rejected(committed):
    f = committed
    verify(f)
    raw = b'{"event":"unit_start"'
    (f.out/'ledger.jsonl').write_bytes(raw)
    with pytest.raises(ValueError, match='incomplete ledger'):
        run.reconcile_done(f.unit, f.out)
    assert (f.out/'ledger.jsonl').read_bytes() == raw

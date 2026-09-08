"""CPU fixtures for explicit interruption recovery; CUDA calls are mocked."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from v3.final_program_20260907 import run


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def interrupted(tmp_path):
    unit = dict(unit_id='SYNTHETIC_RETRY', corpus='toy', phase='formal')
    lock = dict(phase='formal', lock_sha256='a'*64)
    folder = tmp_path/'units'/unit['unit_id']
    reservation = dict(time=run.now(), unit_id=unit['unit_id'], unit_sha256=run.digest(unit),
                       lock_sha256=lock['lock_sha256'])
    dump(folder/'reservations'/'0001.json', reservation)
    (tmp_path/'ledger.jsonl').write_bytes(b'')
    return SimpleNamespace(out=tmp_path, unit=unit, lock=lock, folder=folder, reservation=reservation)


def add(f, event, **fields):
    run.event(f.out, dict(event=event, unit_id=f.unit['unit_id'], attempt='attempts/0001', **fields))


def records(f):
    return run.ledger_rows(f.out)


@pytest.mark.parametrize('had_start', [False, True])
def test_orphan_reservation_or_start_gets_one_explicit_terminal(interrupted, had_start):
    f = interrupted
    if had_start:
        add(f, 'unit_start')
    else:
        # A process record makes this a real nonempty ledger, without a unit start.
        run.event(f.out, dict(event='process_start', lock_sha256=f.lock['lock_sha256']))
    run.close_interrupted_attempts(f.unit, f.out, f.lock)
    original = (f.out/'ledger.jsonl').read_bytes()
    run.close_interrupted_attempts(f.unit, f.out, f.lock)
    assert (f.out/'ledger.jsonl').read_bytes() == original
    terminal = [r for r in records(f) if r['event'] == 'unit_attempt_abandoned']
    assert len(terminal) == 1 and terminal[0]['had_start'] is had_start
    assert terminal[0]['reason'] == 'phase_lock_reacquired_after_interruption'
    assert run.read_json(f.folder/'reservations'/'0001.json') == f.reservation


def test_recorded_failure_is_not_rewritten_as_abandonment(interrupted):
    f = interrupted
    add(f, 'unit_start'); add(f, 'unit_failed')
    before = (f.out/'ledger.jsonl').read_bytes()
    run.close_interrupted_attempts(f.unit, f.out, f.lock)
    assert (f.out/'ledger.jsonl').read_bytes() == before
    assert [r['event'] for r in records(f)] == ['unit_start', 'unit_failed']


@pytest.mark.parametrize('fault', ['duplicate_start', 'failure_without_start', 'unknown_event',
                                  'wrong_abandonment', 'wrong_reservation', 'partial_line'])
def test_invalid_history_is_preserved_without_append(interrupted, fault):
    f = interrupted
    add(f, 'unit_start')
    if fault == 'duplicate_start':
        add(f, 'unit_start')
    elif fault == 'failure_without_start':
        (f.out/'ledger.jsonl').write_bytes(b'')
        add(f, 'unit_failed')
    elif fault == 'unknown_event':
        add(f, 'silently_ignore_previous_attempt')
    elif fault == 'wrong_abandonment':
        add(f, 'unit_attempt_abandoned', had_start=False,
            reason='phase_lock_reacquired_after_interruption')
    elif fault == 'wrong_reservation':
        dump(f.folder/'reservations'/'0001.json', dict(f.reservation, unit_sha256='bad'))
    else:
        with (f.out/'ledger.jsonl').open('ab') as stream:
            stream.write(b'{"event":')
    before = (f.out/'ledger.jsonl').read_bytes()
    with pytest.raises(ValueError):
        run.close_interrupted_attempts(f.unit, f.out, f.lock)
    assert (f.out/'ledger.jsonl').read_bytes() == before


def test_invalid_later_reservation_cannot_partially_close_first(interrupted):
    f = interrupted
    add(f, 'unit_start')
    dump(f.folder/'reservations'/'0002.json', dict(f.reservation, lock_sha256='wrong'))
    before = (f.out/'ledger.jsonl').read_bytes()
    with pytest.raises(ValueError, match='identity'):
        run.close_interrupted_attempts(f.unit, f.out, f.lock)
    assert (f.out/'ledger.jsonl').read_bytes() == before


def test_a_committed_DONE_cannot_be_abandoned(interrupted):
    f = interrupted
    add(f, 'unit_start')
    dump(f.folder/'DONE', {'unit_id': f.unit['unit_id']})
    before = (f.out/'ledger.jsonl').read_bytes()
    with pytest.raises(ValueError, match='committed'):
        run.close_interrupted_attempts(f.unit, f.out, f.lock)
    assert (f.out/'ledger.jsonl').read_bytes() == before


@pytest.mark.parametrize('retry_flag', [False, True])
def test_execute_only_recovers_with_explicit_retry_inside_phase_lock(interrupted, monkeypatch, retry_flag):
    import torch
    from v3.deploy import engines_deploy
    f = interrupted
    add(f, 'unit_start')
    model_path = f.out/'synthetic_model_bytes'; model_path.write_bytes(b'no model is loaded')
    f.lock['model_sha256'] = run.file_sha(model_path)
    roots = f.out/'roots.json'; dump(roots, {'toy': str(f.out/'audio')})
    plan = {'units': [f.unit], 'rows': {'toy': {}}}
    monkeypatch.setattr(run, 'checked_plan', lambda *a: (plan, f.lock))
    monkeypatch.setattr(run, 'verify_audio_inputs', lambda *a: None)
    monkeypatch.setattr(run, 'validate_unit', lambda *a: None)
    monkeypatch.setattr(engines_deploy, 'WaveCache', lambda *a: {})
    monkeypatch.setattr(torch, 'set_num_threads', lambda *a: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'SYNTHETIC_NO_CUDA_CALL')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    held, calls = {'value': False}, []
    original_lock, original_close = run.execution_lock, run.close_interrupted_attempts
    @contextmanager
    def tracked_lock(path):
        with original_lock(path):
            held['value'] = True
            try:
                yield
            finally:
                held['value'] = False
    def close(*args):
        assert held['value'] and retry_flag
        calls.append('recovered')
        return original_close(*args)
    def stop_before_fit(*args, **kwargs):
        raise RuntimeError('SYNTHETIC_STOP_BEFORE_ANY_TRAINING')
    monkeypatch.setattr(run, 'execution_lock', tracked_lock)
    monkeypatch.setattr(run, 'close_interrupted_attempts', close)
    monkeypatch.setattr(run, 'fit_unit', stop_before_fit)
    args = SimpleNamespace(repo=Path.cwd(), out=f.out, model_path=model_path, roots=roots,
                           device='cuda:0', unit_id=None, max_units=1, max_hours=1,
                           retry_failed=retry_flag)
    if retry_flag:
        with pytest.raises(RuntimeError, match='STOP_BEFORE_ANY_TRAINING'):
            run.execute(args)
        assert calls == ['recovered']
        assert len([r for r in records(f) if r['event'] == 'unit_attempt_abandoned']) == 1
        assert (f.folder/'reservations'/'0002.json').is_file()
    else:
        with pytest.raises(ValueError, match='explicit retry flag'):
            run.execute(args)
        assert calls == []
        assert not any(r['event'] == 'unit_attempt_abandoned' for r in records(f))
        assert not (f.folder/'reservations'/'0002.json').exists()

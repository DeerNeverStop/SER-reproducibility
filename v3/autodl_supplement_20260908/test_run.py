"""Mock-only runner guards: no actual runtime probe, audio, GPU or cloud calls."""
import builtins
import copy
import json

import pytest

from . import cloud_pilot, run
from .engine import digest


def stub_context(monkeypatch, tmp_path, *, passed=True):
    plan = dict(plan_sha256='a' * 64, units=[], pilots=[])
    lock = dict(source_lock_sha256='b' * 64, environment_sha256='c' * 64)
    env = dict(environment_sha256='c' * 64, vram_bytes=1000)
    monkeypatch.setattr(run, 'checked_run', lambda *a, **kw: (plan, lock, env, {}))
    gate = dict(schema='ser-autodl-supplement-pilot-gate-1', **{'pass': passed}, formal_allowed=passed,
                plan_sha256=plan['plan_sha256'], source_lock_sha256=lock['source_lock_sha256'],
                environment_sha256=env['environment_sha256'], timing={'with_25pct_margin_hours': 2.})
    gate['pilot_gate_sha256'] = digest(gate)
    monkeypatch.setattr(cloud_pilot, 'evaluate', lambda *args: copy.deepcopy(gate))
    monkeypatch.setattr(run.engine, 'train', lambda *a, **kw: pytest.fail('mock admission must not train'))
    return plan, lock, env, gate


def test_rejected_technical_gate_cannot_create_admission(tmp_path, monkeypatch):
    stub_context(monkeypatch, tmp_path, passed=False)
    with pytest.raises(ValueError, match='pilot technical gate failed'):
        run.admission(tmp_path, 100, 2, 1, 3)
    assert (tmp_path / 'pilot_gate.json').is_file()
    assert not (tmp_path / 'ADMISSION.json').exists()


def test_passed_gate_binds_admission_and_timing_without_training(tmp_path, monkeypatch):
    plan, lock, env, gate = stub_context(monkeypatch, tmp_path)
    run.admission(tmp_path, 100, 2, 1, 3)
    admitted = json.loads((tmp_path / 'ADMISSION.json').read_text())
    assert admitted['passed'] and admitted['projected_total_yuan'] == 8
    assert admitted['projected_formal_hours'] == 2
    assert admitted['pilot_gate_sha256'] == gate['pilot_gate_sha256']
    assert admitted['source_lock_sha256'] == lock['source_lock_sha256']
    assert admitted['environment_sha256'] == env['environment_sha256']
    assert admitted['admission_sha256'] == digest({k: v for k, v in admitted.items() if k != 'admission_sha256'})


@pytest.mark.parametrize('budget,price,reserve,spent', [(5, 2, 1, 3), (101, 2, 1, 3), (100, -2, 1, 3)])
def test_out_of_budget_cannot_admit(tmp_path, monkeypatch, budget, price, reserve, spent):
    stub_context(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        run.admission(tmp_path, budget, price, reserve, spent)
    assert not (tmp_path / 'ADMISSION.json').exists()


def forbid_framework_import(monkeypatch):
    native = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'torchaudio'):
            pytest.fail('runtime host guard must fail before any torch/torchaudio import')
        return native(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked)


def test_windows_runtime_rejected_before_framework_or_cuda(monkeypatch):
    monkeypatch.setattr(run.platform, 'system', lambda: 'Windows')
    forbid_framework_import(monkeypatch)
    with pytest.raises(ValueError, match='AutoDL Linux'):
        run.runtime({})


@pytest.mark.parametrize('fault', ['backend', 'instance', 'hostname'])
def test_wrong_linux_host_rejected_before_framework(monkeypatch, fault):
    monkeypatch.setattr(run.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(run.socket, 'gethostname', lambda: 'SYNTHETIC-host')
    provision = dict(backend='autodl', instance_id='pro-SYNTHETIC', hostname='SYNTHETIC-host')
    provision[{'backend': 'backend', 'instance': 'instance_id', 'hostname': 'hostname'}[fault]] = 'wrong'
    forbid_framework_import(monkeypatch)
    with pytest.raises(ValueError, match='attestation'):
        run.runtime(provision)


def test_formal_requires_actual_admission_even_with_no_units(tmp_path, monkeypatch):
    stub_context(monkeypatch, tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        run.train_batch(tmp_path, 'formal', 1)


def test_existing_invalid_complete_cannot_be_silently_skipped(tmp_path, monkeypatch):
    plan, _, _, _ = stub_context(monkeypatch, tmp_path)
    unit = dict(unit_id='SYNTHETIC-bad-complete', phase='technical_pilot', unit_sha256='f' * 64)
    plan['pilots'] = [unit]
    directory = tmp_path / 'pilots' / unit['unit_id']
    directory.mkdir(parents=True)
    (directory / 'COMPLETE.json').write_text('{}', encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        run.train_batch(tmp_path, 'technical_pilot', 1)


@pytest.mark.parametrize('target', ['gate_pass', 'gate_reference', 'admission_environment'])
def test_modified_admission_closure_blocks_formal(tmp_path, monkeypatch, target):
    stub_context(monkeypatch, tmp_path)
    run.admission(tmp_path, 100, 2, 1, 3)
    name = 'ADMISSION.json' if target == 'admission_environment' else 'pilot_gate.json'
    key = 'admission_sha256' if target == 'admission_environment' else 'pilot_gate_sha256'
    path = tmp_path / name
    value = json.loads(path.read_text())
    value.pop(key)
    if target == 'gate_pass':
        value['pass'] = False
    elif target == 'gate_reference':
        value['unexpected_new_field'] = 'SYNTHETIC modification after admission'
    else:
        value['environment_sha256'] = 'f' * 64
    value[key] = digest(value)
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        run.train_batch(tmp_path, 'formal', 1)
    assert not (tmp_path / 'TRAINING.lock').exists()


def test_exclusive_runner_preserves_foreign_lock_and_cleans_own_failure(tmp_path, monkeypatch):
    stub_context(monkeypatch, tmp_path)
    lock = tmp_path / 'TRAINING.lock'
    lock.write_text('SYNTHETIC owner', encoding='utf-8')
    with pytest.raises(FileExistsError):
        run.train_batch(tmp_path, 'technical_pilot', 1)
    assert lock.read_text() == 'SYNTHETIC owner'
    lock.unlink()
    with pytest.raises(FileNotFoundError):
        run.train_batch(tmp_path, 'formal', 1)
    assert not lock.exists()


def release_fixture(tmp_path, monkeypatch):
    unit = dict(unit_id='SYNTHETIC-release', phase='formal', permanent_checkpoint_sample=False,
                model='wavlm_base_plus', config={'epochs': 15}, windows=[15])
    unit['unit_sha256'] = digest(unit)
    plan = dict(plan_sha256='a' * 64, units=[unit], pilots=[])
    identity = dict(name='wavlm_base_plus', file_sha256='1' * 64, state_sha256='2' * 64)
    lock = dict(source_lock_sha256='b' * 64, environment_sha256='c' * 64,
                models={'wavlm_base_plus': identity})
    monkeypatch.setattr(run, 'checked_run', lambda *a, **kw: (plan, lock, {}, {}))
    directory = tmp_path / 'units' / unit['unit_id']
    directory.mkdir(parents=True)
    for name in ('checkpoint.pt', 'predictions.npz', 'history.json'):
        (directory / name).write_bytes(b'SYNTHETIC receipt fixture; not actual model/prediction data')
    receipt = dict(schema='ser-autodl-receipt-1', unit_id=unit['unit_id'], phase='formal',
                   unit_sha256=unit['unit_sha256'], plan_sha256=plan['plan_sha256'],
                   source_lock_sha256=lock['source_lock_sha256'], environment_sha256=lock['environment_sha256'],
                   files={n: run.file_record(directory / n) for n in ('checkpoint.pt', 'predictions.npz', 'history.json')},
                   info=dict(checkpoint_reload_verified=True, unit_sha256=unit['unit_sha256'],
                             model_identity=identity, epochs=15, windows=[15], outer_scores_computed=False))
    run.write_new(directory / 'receipt.json', receipt)
    complete = {k: receipt[k] for k in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256')}
    complete['receipt_sha256'] = run.file_sha(directory / 'receipt.json')
    run.write_new(directory / 'COMPLETE.json', complete)
    ack = dict(**{k: receipt[k] for k in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256')},
               receipt_sha256=complete['receipt_sha256'], backup_location='off-instance-local-verified',
               files={n: run.file_record(directory / n) for n in ('predictions.npz', 'history.json', 'receipt.json', 'COMPLETE.json')})
    run.write_new(directory / 'backup_ack.json', run.seal(ack, 'backup_ack_sha256'))
    return unit, plan, lock, directory


def test_release_recovers_interrupted_final_record_and_retains_small_artifacts(tmp_path, monkeypatch):
    unit, plan, lock, directory = release_fixture(tmp_path, monkeypatch)
    native = run.write_new
    def interrupt_final(path, value, *args, **kwargs):
        if path.name == 'checkpoint_release.json':
            raise OSError('SYNTHETIC interruption after unlink')
        return native(path, value, *args, **kwargs)
    monkeypatch.setattr(run, 'write_new', interrupt_final)
    with pytest.raises(OSError, match='SYNTHETIC'):
        run.release_backed_up_weights(tmp_path, unit['unit_id'])
    assert not (directory / 'checkpoint.pt').exists()
    assert (directory / 'checkpoint_release_intent.json').is_file()
    assert not (directory / 'checkpoint_release.json').exists()
    monkeypatch.setattr(run, 'write_new', native)
    run.release_backed_up_weights(tmp_path, unit['unit_id'])
    run.release_backed_up_weights(tmp_path, unit['unit_id'])  # idempotent completed cleanup
    assert (directory / 'checkpoint_release.json').read_bytes() == (directory / 'checkpoint_release_intent.json').read_bytes()
    assert all((directory / n).is_file() for n in ('receipt.json', 'predictions.npz', 'history.json', 'COMPLETE.json'))
    run.checked_existing(directory, unit, plan, lock)


def test_missing_checkpoint_without_authorized_release_rejected(tmp_path, monkeypatch):
    unit, plan, lock, directory = release_fixture(tmp_path, monkeypatch)
    (directory / 'checkpoint.pt').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        run.checked_existing(directory, unit, plan, lock)
    with pytest.raises(ValueError, match='without release intent'):
        run.release_backed_up_weights(tmp_path, unit['unit_id'])


def test_bad_backup_ack_prevents_weight_deletion(tmp_path, monkeypatch):
    unit, _, _, directory = release_fixture(tmp_path, monkeypatch)
    path = directory / 'backup_ack.json'
    ack = json.loads(path.read_text())
    ack.pop('backup_ack_sha256')
    ack['files']['predictions.npz']['sha256'] = 'f' * 64
    path.write_text(json.dumps(run.seal(ack, 'backup_ack_sha256')), encoding='utf-8')
    with pytest.raises(ValueError):
        run.release_backed_up_weights(tmp_path, unit['unit_id'])
    assert (directory / 'checkpoint.pt').is_file()
    assert not (directory / 'checkpoint_release_intent.json').exists()

"""AutoDL-only execution, prospective admission and bounded checkpoint storage."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

from . import engine
from .engine import digest, file_sha, require
from .plan import read_json, write_new, seal, check_seal, validate_plan, verify_audio

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path):
    return dict(bytes=Path(path).stat().st_size, sha256=file_sha(path))


def source_files():
    return {p.relative_to(ROOT).as_posix(): file_sha(p)
            for p in sorted(PACKAGE.glob('*.py'))}


def runtime(provision):
    # This guard deliberately precedes CUDA initialization on an unsupported host.
    require(platform.system() == 'Linux', 'all real GPU work must run on AutoDL Linux')
    require(provision['backend'] == 'autodl' and provision['instance_id'].startswith('pro-')
            and socket.gethostname() == provision['hostname'], 'AutoDL host attestation differs')
    import torch
    import torchaudio
    require(sys.version_info[:2] == (3, 11), 'Python major/minor differs')
    require(torch.__version__ == torchaudio.__version__ == '2.11.0+cu128', 'framework versions differ')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected one AutoDL GPU')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    require(provision['gpu_uuid'] in smi, 'provisioned GPU identity differs')
    props = torch.cuda.get_device_properties(0)
    env = dict(schema='ser-autodl-environment-1', backend='autodl',
               instance_id=provision['instance_id'], hostname=socket.gethostname(),
               python=platform.python_version(), platform=platform.platform(),
               packages={name: importlib.metadata.version(name) for name in
                         ('torch', 'torchaudio', 'numpy', 'scipy', 'librosa', 'soundfile')},
               installed_packages=sorted([d.metadata['Name'], d.version] for d in importlib.metadata.distributions()),
               cuda=torch.version.cuda, nvidia_smi=smi, gpu_name=props.name,
               vram_bytes=props.total_memory, num_threads=4, num_interop_threads=2)
    return seal(env, 'environment_sha256')


def prepare(args):
    run = Path(args.run_dir).resolve()
    require(not (run / 'SOURCE_LOCK.json').exists(), 'run is already frozen')
    plan = read_json(args.plan)
    validate_plan(plan)
    provision = read_json(args.provision)
    env = runtime(provision)
    paths = dict(models=read_json(args.models)['models'], audio_roots=read_json(args.audio_roots))
    audio_check = verify_audio(plan, paths['audio_roots'])
    for spec in paths['models'].values():
        require(file_sha(spec['path']) == spec['file_sha256'], 'base model file differs')
    require(set(paths['models']) == set(engine.MODEL_NAMES), 'model inventory differs')
    run.mkdir(parents=True, exist_ok=True)
    write_new(run / 'PLAN.json.gz', plan, compressed=True)
    write_new(run / 'PROVISION.json', provision)
    write_new(run / 'ENVIRONMENT.json', env)
    write_new(run / 'RUNTIME_PATHS.local.json', paths)
    write_new(run / 'INPUT_VERIFICATION.json', audio_check)
    lock = seal(dict(schema='ser-autodl-source-lock-1', created_at_utc=now(),
                     source_commit=args.source_commit, sources=source_files(),
                     models={k: engine.model_identity(v) for k, v in paths['models'].items()},
                     runtime_paths_sha256=file_sha(run / 'RUNTIME_PATHS.local.json'),
                     plan_sha256=plan['plan_sha256'], environment_sha256=env['environment_sha256'],
                     private_preexecution_freeze_not_public_preregistration=True), 'source_lock_sha256')
    write_new(run / 'SOURCE_LOCK.json', lock)
    print(json.dumps(dict(prepared=True, source_lock_sha256=lock['source_lock_sha256'],
                          plan_sha256=plan['plan_sha256'], gpu=env['gpu_name'])), flush=True)


def checked_run(run, require_gpu=True):
    run = Path(run).resolve()
    plan, lock = read_json(run / 'PLAN.json.gz'), read_json(run / 'SOURCE_LOCK.json')
    validate_plan(plan)
    check_seal(lock, 'source_lock_sha256')
    env = read_json(run / 'ENVIRONMENT.json')
    check_seal(env, 'environment_sha256')
    require(lock['plan_sha256'] == plan['plan_sha256'] and lock['sources'] == source_files(), 'frozen source/plan differs')
    require(lock['environment_sha256'] == env['environment_sha256'], 'environment binding differs')
    if require_gpu:
        current = runtime(read_json(run / 'PROVISION.json'))
        require(current == env, 'runtime changed after freeze')
    paths = read_json(run / 'RUNTIME_PATHS.local.json')
    require(file_sha(run / 'RUNTIME_PATHS.local.json') == lock['runtime_paths_sha256'], 'runtime input paths differ')
    require({k: engine.model_identity(v) for k, v in paths['models'].items()} == lock['models'], 'model identity differs')
    return plan, lock, env, paths


def unit_directory(run, unit):
    return Path(run) / ('units' if unit['phase'] == 'formal' else 'pilots') / unit['unit_id']


def verify_complete(folder, unit, plan, lock, *, checkpoint=True):
    complete, receipt = read_json(folder / 'COMPLETE.json'), read_json(folder / 'receipt.json')
    for key, expected in (('unit_id', unit['unit_id']), ('unit_sha256', unit['unit_sha256']),
                          ('plan_sha256', plan['plan_sha256']), ('source_lock_sha256', lock['source_lock_sha256'])):
        require(complete[key] == receipt[key] == expected, 'completion binding differs')
    require(complete['receipt_sha256'] == file_sha(folder / 'receipt.json'), 'receipt bytes differ')
    require(set(receipt['files']) == {'checkpoint.pt', 'predictions.npz', 'history.json'}
            | ({'fresh_restore.json'} if unit['permanent_checkpoint_sample'] else set()),
            'receipt artifact inventory differs')
    require(receipt['phase'] == unit['phase'] and receipt['environment_sha256'] == lock['environment_sha256']
            and receipt['info']['unit_sha256'] == unit['unit_sha256']
            and receipt['info']['model_identity'] == lock['models'][unit['model']], 'receipt runtime/model identity differs')
    for name, record in receipt['files'].items():
        if checkpoint or name != 'checkpoint.pt':
            require(file_record(folder / name) == record, 'artifact bytes differ')
    require(receipt['info']['checkpoint_reload_verified'] is True, 'reload verification missing')
    return receipt


def verify_ack(folder, receipt):
    ack = read_json(folder / 'backup_ack.json')
    check_seal(ack, 'backup_ack_sha256')
    for key in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256'):
        require(ack[key] == receipt[key], 'backup acknowledgment binding differs')
    require(ack['backup_location'] == 'off-instance-local-verified'
            and ack['receipt_sha256'] == file_sha(folder / 'receipt.json'), 'off-instance receipt backup missing')
    for name in ('predictions.npz', 'history.json', 'receipt.json', 'COMPLETE.json'):
        require(ack['files'][name] == file_record(folder / name), 'off-instance artifact backup differs')
    return ack


def checked_existing(folder, unit, plan, lock):
    present = (folder / 'checkpoint.pt').exists()
    require(present or not unit['permanent_checkpoint_sample'], 'fixed recovery checkpoint is missing')
    receipt = verify_complete(folder, unit, plan, lock, checkpoint=present)
    if not present:
        release = read_json(folder / 'checkpoint_release.json')
        check_seal(release, 'release_sha256')
        ack = verify_ack(folder, receipt)
        for key in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256'):
            require(release[key] == receipt[key], 'checkpoint release identity differs')
        require(release['checkpoint_sha256'] == receipt['files']['checkpoint.pt']['sha256']
                and release['receipt_sha256'] == file_sha(folder / 'receipt.json')
                and release['backup_ack_sha256'] == ack['backup_ack_sha256']
                and release['checkpoint_retained'] is False and release['checkpoint_reload_verified'] is True,
                'checkpoint release evidence differs')
    elif (folder / 'backup_ack.json').exists():
        verify_ack(folder, receipt)
    return receipt


@contextmanager
def exclusive_training(run):
    path = Path(run) / 'TRAINING.lock'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, engine.canonical(dict(pid=os.getpid(), hostname=socket.gethostname(), at_utc=now())))
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        path.unlink()


def restore(run, unit_id):
    import numpy as np
    import torch
    plan, lock, env, paths = checked_run(run)
    unit = next(u for u in plan['units'] + plan['pilots'] if u['unit_id'] == unit_id)
    folder = unit_directory(run, unit)
    spec = paths['models'][unit['model']]
    model, _ = engine.build_model(spec, unit['seeds']['head'], unit['n_classes'])
    saved, checkpoint_sha = engine.validate_checkpoint(unit, spec, folder / 'checkpoint.pt', model)
    cache = engine.WaveCache(paths['audio_roots'][unit['corpus']])
    arrays = np.load(folder / 'predictions.npz', allow_pickle=False)
    model.to('cuda').eval()
    comparisons = {}
    for epoch, delta in saved['epoch_states'].items():
        engine.load_delta(model, delta, saved['frozen_parameter_sha256'])
        for group in engine.GROUPS:
            observed = engine.predict_group(model, unit['report_batches'][group], cache, 160000,
                                            unit['n_classes'], torch.device('cuda'))
            expected = arrays[f'{group}__all_epoch_logits'][int(epoch) - 1]
            error = float(np.max(np.abs(observed - expected)))
            require(error <= 1e-5, 'fresh-process checkpoint restoration differs')
            comparisons[f'{epoch}/{group}'] = error
    report = dict(schema='ser-autodl-fresh-restore-1', unit_id=unit_id,
                  unit_sha256=unit['unit_sha256'], plan_sha256=plan['plan_sha256'],
                  source_lock_sha256=lock['source_lock_sha256'], environment_sha256=env['environment_sha256'],
                  checkpoint_sha256=checkpoint_sha, predictions_sha256=file_sha(folder / 'predictions.npz'),
                  comparisons=comparisons, max_abs_diff=max(comparisons.values()), passed=True,
                  scope='fresh-process inference of saved deltas; not independent retraining')
    write_new(folder / 'fresh_restore.json', report)


def admission(run, budget_yuan, gpu_price, storage_reserve, already_spent):
    from .cloud_pilot import evaluate
    plan, lock, env, _ = checked_run(run)
    gate = evaluate(plan, Path(run), lock['source_lock_sha256'], env['vram_bytes'])
    write_new(Path(run) / 'pilot_gate.json', gate)
    require(gate['pass'] and gate['formal_allowed'], 'pilot technical gate failed')
    # Filled from timing only; no scored outer effects enter admission.
    hours = gate['timing']['with_25pct_margin_hours']
    estimated = already_spent + hours * gpu_price + storage_reserve
    require(all(math.isfinite(x) for x in (budget_yuan, gpu_price, storage_reserve, already_spent))
            and storage_reserve >= 0 and already_spent >= 0
            and 0 < gpu_price and 0 < budget_yuan <= 100 and estimated <= budget_yuan,
            'resource forecast exceeds declared budget')
    value = seal(dict(schema='ser-autodl-admission-1', plan_sha256=plan['plan_sha256'],
                      source_lock_sha256=lock['source_lock_sha256'], environment_sha256=env['environment_sha256'],
                      pilot_gate_sha256=gate['pilot_gate_sha256'], admitted_at_utc=now(),
                      budget_yuan=budget_yuan, gpu_hourly_yuan=gpu_price,
                      already_spent_yuan=already_spent, storage_reserve_yuan=storage_reserve,
                      projected_total_yuan=estimated, projected_formal_hours=hours,
                      passed=True, outer_outcomes_used=False), 'admission_sha256')
    write_new(Path(run) / 'ADMISSION.json', value)
    print(json.dumps(value), flush=True)


def train_batch(run, phase, limit):
    with exclusive_training(run):
        _train_batch(run, phase, limit)


def _train_batch(run, phase, limit):
    plan, lock, env, paths = checked_run(run)
    run = Path(run).resolve()
    require(1 <= limit <= 8, 'bounded batch limit is 1..8')
    if phase == 'formal':
        admitted = read_json(run / 'ADMISSION.json')
        check_seal(admitted, 'admission_sha256')
        require(admitted['passed'] and admitted['source_lock_sha256'] == lock['source_lock_sha256']
                and admitted['plan_sha256'] == plan['plan_sha256']
                and admitted['environment_sha256'] == env['environment_sha256'], 'formal admission differs')
        gate = read_json(run / 'pilot_gate.json')
        check_seal(gate, 'pilot_gate_sha256')
        require(gate['pass'] and gate['formal_allowed'] and gate['pilot_gate_sha256'] == admitted['pilot_gate_sha256'],
                'pilot admission gate differs')
        for key in ('plan_sha256', 'source_lock_sha256', 'environment_sha256'):
            require(gate[key] == admitted[key], 'pilot gate context differs')
    units = plan['units'] if phase == 'formal' else plan['pilots']
    completed = 0
    for unit in units:
        folder = unit_directory(run, unit)
        if (folder / 'COMPLETE.json').exists():
            checked_existing(folder, unit, plan, lock)
            continue
        if phase == 'formal':
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(admitted['admitted_at_utc'])).total_seconds() / 3600
            next_hours = max(g['pilot_wall_seconds'] for g in gate['timing']['groups']) * 1.25 / 3600
            require(admitted['already_spent_yuan'] + (elapsed + next_hours) * admitted['gpu_hourly_yuan']
                    + admitted['storage_reserve_yuan'] < admitted['budget_yuan'], 'declared elapsed-time budget reached')
            waiting = [u for u in plan['units'] if (unit_directory(run, u) / 'COMPLETE.json').exists()
                       and not (unit_directory(run, u) / 'backup_ack.json').exists()]
            require(len(waiting) < 8, 'off-instance backup acknowledgments required before more training')
        require(shutil.disk_usage(run).free >= 20 * 1024**3, 'free disk below 20 GiB')
        raw_paths = set(unit['fit']).union(*unit['report'].values())
        audio_root = Path(paths['audio_roots'][unit['corpus']]).resolve()
        for relative in sorted(raw_paths):
            audio_file = (audio_root / relative).resolve()
            expected = plan['rows'][unit['corpus']][relative]
            require(audio_file.is_relative_to(audio_root) and file_record(audio_file)
                    == dict(bytes=expected['bytes'], sha256=expected['sha256']), 'unit audio input differs')
        started = run / 'attempts' / f"{unit['unit_id']}.started.json"
        require(not started.exists(), 'unfinished attempt exists; no silent training retry')
        write_new(started, dict(unit_id=unit['unit_id'], unit_sha256=unit['unit_sha256'],
                               source_lock_sha256=lock['source_lock_sha256'], at_utc=now()))
        began = time.monotonic()
        try:
            info = engine.train(unit, plan['rows'][unit['corpus']], paths['audio_roots'],
                                paths['models'][unit['model']], folder, 'cuda')
            filenames = ['checkpoint.pt', 'predictions.npz', 'history.json']
            if unit['permanent_checkpoint_sample']:
                subprocess.run([sys.executable, '-B', '-m', 'v3.autodl_supplement_20260908.run',
                                'restore', '--run-dir', str(run), '--unit-id', unit['unit_id']],
                               cwd=ROOT, check=True)
                filenames.append('fresh_restore.json')
                info['fresh_process_restore_verified'] = True
            info['total_unit_wall_seconds'] = time.monotonic() - began
            receipt = dict(schema='ser-autodl-receipt-1', unit_id=unit['unit_id'], phase=unit['phase'],
                           unit_sha256=unit['unit_sha256'], plan_sha256=plan['plan_sha256'],
                           source_lock_sha256=lock['source_lock_sha256'], environment_sha256=env['environment_sha256'],
                           files={name: file_record(folder / name) for name in filenames}, info=info,
                           completed_at_utc=now())
            write_new(folder / 'receipt.json', receipt)
            complete = {k: receipt[k] for k in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256')}
            complete['receipt_sha256'] = file_sha(folder / 'receipt.json')
            write_new(folder / 'COMPLETE.json', complete)
            verify_complete(folder, unit, plan, lock)
            completed += 1
            print(json.dumps(dict(completed=unit['unit_id'], phase=phase,
                                  wall_seconds=info['total_unit_wall_seconds'],
                                  peak_vram_gib=info['peak_cuda_reserved_bytes']/1024**3)), flush=True)
        except BaseException as exc:
            write_new(run / 'attempts' / f"{unit['unit_id']}.failed.json",
                      dict(at_utc=now(), exception_type=type(exc).__name__, detail=str(exc)))
            raise
        if completed >= limit:
            break


def release_backed_up_weights(run, unit_id):
    # Explicit per-file retention release; there is no recursive filesystem removal.
    plan, lock, _, _ = checked_run(run, require_gpu=False)
    unit = next(u for u in plan['units'] if u['unit_id'] == unit_id)
    require(not unit['permanent_checkpoint_sample'], 'fixed recovery weights must remain present')
    folder = unit_directory(run, unit).resolve()
    require(folder.is_relative_to(Path(run).resolve() / 'units'), 'release path escapes run')
    present = (folder / 'checkpoint.pt').exists()
    receipt = verify_complete(folder, unit, plan, lock, checkpoint=present)
    ack = verify_ack(folder, receipt)
    release = seal(dict(schema='ser-checkpoint-release-1',
                        **{k: receipt[k] for k in ('unit_id', 'unit_sha256', 'plan_sha256', 'source_lock_sha256')},
                        receipt_sha256=file_sha(folder / 'receipt.json'),
                        checkpoint_sha256=receipt['files']['checkpoint.pt']['sha256'],
                        backup_ack_sha256=ack['backup_ack_sha256'], checkpoint_reload_verified=True,
                        checkpoint_retained=False, at_utc=now()), 'release_sha256')
    # Intent is separately retained if interruption occurs between deletion and final record.
    intent = folder / 'checkpoint_release_intent.json'
    final = folder / 'checkpoint_release.json'
    if intent.exists():
        existing = read_json(intent)
        check_seal(existing, 'release_sha256')
        require(all(existing[k] == v for k, v in release.items() if k not in ('at_utc', 'release_sha256')),
                'checkpoint release intent differs')
        release = existing
    else:
        require(present, 'checkpoint is missing without release intent')
        write_new(intent, release)
    if present:
        (folder / 'checkpoint.pt').unlink()
    if final.exists():
        require(read_json(final) == release, 'final release record differs')
    else:
        write_new(final, release)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for arg in ('plan', 'run-dir', 'models', 'audio-roots', 'provision', 'source-commit'):
        prep.add_argument('--' + arg, required=True)
    for command in ('pilot', 'formal', 'restore', 'admit', 'release'):
        s = sub.add_parser(command)
        s.add_argument('--run-dir', required=True)
        if command in ('pilot', 'formal'):
            s.add_argument('--limit', type=int, default=8)
        elif command in ('restore', 'release'):
            s.add_argument('--unit-id', required=True)
        else:
            for arg in ('budget-yuan', 'gpu-hourly-yuan', 'storage-reserve-yuan', 'already-spent-yuan'):
                s.add_argument('--' + arg, type=float, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command in ('pilot', 'formal'):
        train_batch(args.run_dir, 'technical_pilot' if args.command == 'pilot' else 'formal', args.limit)
    elif args.command == 'restore':
        restore(args.run_dir, args.unit_id)
    elif args.command == 'release':
        release_backed_up_weights(args.run_dir, args.unit_id)
    else:
        admission(args.run_dir, args.budget_yuan, args.gpu_hourly_yuan,
                  args.storage_reserve_yuan, args.already_spent_yuan)


if __name__ == '__main__':
    main()

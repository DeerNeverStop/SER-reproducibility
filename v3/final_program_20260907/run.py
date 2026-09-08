"""Frozen-source execution, recoverable attempts and explicit artifact closure."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

from v3.inner_validation.plan import source_files as original_sources
from .engine import fit_unit, validate_unit, metrics
from .plan import load_plan, validate as validate_plan
from .ledger_contract import audit_events, ABANDON_REASON


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def sources(repo, include_analysis=True):
    result = original_sources(repo)
    folder = repo / 'v3/final_program_20260907'
    names = ['__init__.py', 'plan.py', 'engine.py', 'run.py', 'ledger_contract.py', 'SCIENCE_DESIGN.md']
    if include_analysis:
        names.append('score.py')
    for f in [folder / n for n in names]:
        require(f.is_file(), 'required source missing: ' + str(f))
        result[f.relative_to(repo).as_posix()] = file_sha(f)
    return dict(sorted(result.items()))


def check_import_locations(repo):
    for name, module in list(sys.modules.items()):
        if name == 'v2' or name == 'v3' or name.startswith(('v2.', 'v3.')):
            path = getattr(module, '__file__', None)
            expected=(repo/Path(*name.split('.'))).resolve()
            if path is None:
                locations=list(getattr(module,'__path__',[]))
                require(bool(locations) and all(Path(p).resolve()==expected for p in locations),
                        'loaded namespace outside selected repository: '+name)
            else:
                require(Path(path).resolve() in (expected.with_suffix('.py'), expected/'__init__.py'),
                        'loaded project module outside selected repository: '+name)


def plan_digest(plan):
    return digest({k: v for k, v in plan.items() if k != 'plan_sha256'})


def freeze(repo, plan_path, out, phase, model_path):
    require(not (out / 'SOURCE_LOCK.json').exists(), 'source lock already exists')
    check_import_locations(repo)
    plan = load_plan(plan_path, repo)
    require(plan.get('plan_sha256') == plan_digest(plan), 'plan semantic hash differs')
    model_sha = file_sha(model_path)
    require(model_sha == '136a3e720c04f2c77bf7a4dc6a3868b14d5a2c145a988114b733cb1a8428be98', 'base bytes differ')
    source_map=sources(repo,phase=='formal')
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    for rel,expected in source_map.items():
        committed=subprocess.check_output(['git','show',source_commit+':'+rel],cwd=repo)
        require(hashlib.sha256(committed).hexdigest()==expected,'commit does not contain exact frozen source: '+rel)
    lock = dict(schema='ser-final-program-source-lock-1', created_at=now(), phase=phase,
                plan_sha256=plan['plan_sha256'], plan_byte_sha256=file_sha(plan_path),
                source_commit=source_commit,
                private_preexecution_freeze_not_public_preregistration=True,
                sources=source_map, model_sha256=model_sha,
                phase_units=[u['unit_id'] for u in plan['units'] if u['phase'] == phase])
    lock['lock_sha256'] = digest(lock)
    require(lock['phase_units'], 'empty phase')
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'plan_snapshot.json').exists(), 'snapshot already exists')
    shutil.copyfile(plan_path, out / 'plan_snapshot.json')
    write_json(out / 'SOURCE_LOCK.json', lock)
    print(json.dumps({'frozen': phase, 'units': len(lock['phase_units']), 'lock_sha256': lock['lock_sha256']}))


def checked_plan(repo, out):
    check_import_locations(repo)
    lock, plan = read_json(out / 'SOURCE_LOCK.json'), read_json(out / 'plan_snapshot.json')
    require(lock['lock_sha256'] == digest({k: v for k, v in lock.items() if k != 'lock_sha256'}), 'lock altered')
    require(lock['sources'] == sources(repo,lock['phase']=='formal'), 'frozen source changed')
    require(file_sha(out / 'plan_snapshot.json') == lock['plan_byte_sha256']
            and plan_digest(plan) == plan['plan_sha256'] == lock['plan_sha256'], 'frozen plan changed')
    expected = [u['unit_id'] for u in plan['units'] if u['phase'] == lock['phase']]
    require(expected == lock['phase_units'], 'phase inventory differs')
    validate_plan(plan, repo)
    return plan, lock


@contextmanager
def execution_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            if not f.read(1):
                f.write(b'0')
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def event(out, value):
    with (out / 'ledger.jsonl').open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(dict(time=now(), **value), ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def verify_done(unit, out, lock, rows, weights=True, done_doc=None):
    import numpy as np
    folder = out / 'units' / unit['unit_id']
    done = read_json(folder / 'DONE') if done_doc is None else done_doc
    require(done['unit_sha256'] == digest(unit) and done['unit_id'] == unit['unit_id']
            and done['lock_sha256'] == lock['lock_sha256'], 'DONE identity mismatch')
    expected_names = {'checkpoint.pt', 'predictions.npz', 'history.json', 'receipt.json'}
    require(re.fullmatch(r'attempts/\d{4}', done['attempt']) is not None, 'invalid attempt path')
    require(set(done['artifacts']) == {done['attempt'] + '/' + n for n in expected_names},
            'artifact paths do not close over the committed attempt')
    for rel, expected in done['artifacts'].items():
        path = (folder / rel).resolve()
        require(path.is_relative_to(folder.resolve()), 'artifact path traversal')
        if weights or path.name != 'checkpoint.pt':
            require(path.is_file() and file_sha(path) == expected, 'artifact differs: ' + rel)
    receipt = read_json(folder / done['attempt'] / 'receipt.json')
    require(receipt['unit_sha256'] == digest(unit) and receipt['lock_sha256'] == lock['lock_sha256']
            and receipt['unit_id'] == unit['unit_id'] and receipt['phase'] == unit['phase'], 'receipt identity')
    info = receipt['info']
    require(info['epochs'] == 15 and math.isfinite(info['wall_seconds']) and info['wall_seconds'] > 0, 'invalid execution extent')
    require(math.isfinite(info['reload_max_abs_diff']) and 0 <= info['reload_max_abs_diff'] <= 1e-5
            and info['outer_scores_computed'] is False,
            'replay / outer-score contract differs')
    for key in ('initial_state_sha256', 'frozen_parameter_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', info[key]) is not None, 'invalid parameter identity')
    expected_labels = validate_unit(unit, rows)
    root = folder / done['attempt']
    history = read_json(root / 'history.json')
    require(len(history) == 15 and [h['epoch'] for h in history] == list(range(1, 16)), 'incomplete trajectory history')
    for h in history:
        require(h['optimizer_steps'] == math.ceil(len(unit['fit'])/16)
                and type(h['scaler_skipped_steps']) is int and 0 <= h['scaler_skipped_steps'] <= h['optimizer_steps']
                and math.isfinite(h['train_loss']) and h['train_loss'] >= 0, 'invalid training history')
    expected_keys = {'epochs'} | {g+'__'+k for g in ('A','B','outer')
                                 for k in ('paths','labels','all_epoch_logits')}
    with np.load(root / 'predictions.npz', allow_pickle=False) as npz:
        require(set(npz.files) == expected_keys and npz['epochs'].tolist() == unit['prediction_epochs'], 'NPZ schema/epochs')
        for g in ('A','B','outer'):
            require(npz[g+'__paths'].tolist() == unit['report'][g]
                    and np.array_equal(npz[g+'__labels'], expected_labels[g]), 'NPZ paths/labels differ')
            values = npz[g+'__all_epoch_logits']
            require(values.shape == (len(unit['prediction_epochs']),len(unit['report'][g]),unit['n_classes'])
                    and values.dtype == np.float32 and np.isfinite(values).all(), 'NPZ prediction support differs')
        expected_selected = {'last':15}
        if unit['selection_enabled']:
            for g, name in ((unit['arm'],'seen'), ('B' if unit['arm']=='A' else 'A','unseen')):
                values = [metrics(x,expected_labels[g],unit['n_classes']) for x in npz[g+'__all_epoch_logits']]
                expected_selected[name+'_ce'] = min(range(15),key=lambda i:values[i][0])+1
                expected_selected[name+'_uar'] = max(range(15),key=lambda i:values[i][1])+1
                for i,v in enumerate(values):
                    require(abs(history[i][name+'_ce']-v[0]) <= 1e-10
                            and abs(history[i][name+'_uar']-float(v[1])) <= 1e-12, 'stored validation metric differs')
        require(info['selected_epochs'] == expected_selected, 'selected epochs differ from archived validation predictions')
    expected_diffs = {f'{e}/{g}' for e in set(expected_selected.values()) for g in ('A','B','outer')}
    require(set(info['reload_by_epoch_group']) == expected_diffs
            and all(math.isfinite(v) and 0 <= v <= 1e-5 for v in info['reload_by_epoch_group'].values())
            and max(info['reload_by_epoch_group'].values()) == info['reload_max_abs_diff'], 'incomplete reload coverage')
    if weights:
        import torch
        saved = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=True)
        require(saved['schema']=='ser-final-program-delta-1' and saved['unit_id']==unit['unit_id']
                and saved['n_classes']==unit['n_classes'] and saved['selected_epochs']==expected_selected
                and saved['base_state_sha256']=='d8e745fb761c56d209431413bcc373f92784dc2c8d850199b6caa5cf05ca2c36'
                and saved['initial_state_sha256']==info['initial_state_sha256']
                and saved['frozen_parameter_sha256']==info['frozen_parameter_sha256'], 'checkpoint identity differs')
        require(set(saved['epoch_states'])=={str(e) for e in expected_selected.values()},'checkpoint epoch inventory differs')
        expected_spec = expected_delta_spec(unit['n_classes'])
        require(set(saved['trainable_parameter_names'])|set(saved['buffer_names'])==set(expected_spec), 'checkpoint key inventory differs')
        for delta in saved['epoch_states'].values():
            require(set(delta)==set(expected_spec), 'incomplete state delta')
            for name,tensor in delta.items():
                shape,dtype=expected_spec[name]
                require(isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==shape
                        and tensor.dtype==dtype and bool(torch.isfinite(tensor).all()), 'invalid state tensor '+name)
    return done, receipt


_DELTA_SPECS = {}


def expected_delta_spec(classes):
    import torch
    import torchaudio
    if classes not in _DELTA_SPECS:
        before = torch.get_rng_state().clone()
        enc = torchaudio.models.wavlm_model(**torchaudio.pipelines.WAVLM_BASE_PLUS._params)
        spec = {'enc.'+name:(tuple(p.shape),p.dtype) for name,p in enc.named_parameters()
                if any(name.startswith(f'encoder.transformer.layers.{i}.') for i in (8,9,10,11))}
        spec.update({'enc.'+name:(tuple(p.shape),p.dtype) for name,p in enc.named_buffers()})
        spec.update({'fc.weight':((classes,768),torch.float32),'fc.bias':((classes,),torch.float32)})
        _DELTA_SPECS[classes] = spec
        del enc
        torch.set_rng_state(before)
    return _DELTA_SPECS[classes]


def ledger_rows(out):
    path=out/'ledger.jsonl'
    if not path.exists():
        return []
    raw=path.read_bytes()
    require(raw.endswith(b'\n'), 'incomplete ledger suffix requires explicit repair')
    return [json.loads(line) for line in raw.decode('utf-8').splitlines()]


def reconcile_done(unit,out):
    done=read_json(out/'units'/unit['unit_id']/'DONE')
    done_sha=file_sha(out/'units'/unit['unit_id']/'DONE')
    rows=[r for r in ledger_rows(out) if r.get('unit_id')==unit['unit_id']]
    starts=[r for r in rows if r['event']=='unit_start' and r.get('attempt')==done['attempt']]
    require(len(starts)==1,'committed attempt must have exactly one ledger start')
    require(not any(r['event']=='unit_failed' and r.get('attempt')==done['attempt'] for r in rows),
            'committed attempt has a terminal failure')
    reservation=read_json(out/'units'/unit['unit_id']/'reservations'/(Path(done['attempt']).name+'.json'))
    require(reservation['unit_sha256']==done['unit_sha256'] and reservation['lock_sha256']==done['lock_sha256']
            and reservation['unit_id']==unit['unit_id'],'attempt reservation identity mismatch')
    completed=[r for r in rows if r['event']=='unit_done']
    require(len(completed)<=1,'duplicate ledger success')
    if completed:
        require(completed[0]['done_sha256']==done_sha,'ledger DONE SHA differs')
    else:
        event(out,dict(event='unit_done',unit_id=unit['unit_id'],done_sha256=done_sha,recovered_after_commit=True))


def verify_audio_inputs(plan,units,roots,out):
    summary={}
    for corpus in sorted({u['corpus'] for u in units}):
        root=Path(roots[corpus]).resolve()
        needed=sorted({p for u in units if u['corpus']==corpus for p in
                       [*u['fit'], *[p for group in u['report'].values() for p in group]]})
        for rel in needed:
            path=(root/rel).resolve(); row=plan['rows'][corpus][rel]
            require(path.is_relative_to(root) and path.is_file() and path.stat().st_size==int(row['bytes'])
                    and file_sha(path)==row['sha256'],'runtime audio identity differs: '+corpus+'/'+rel)
        summary[corpus]=dict(root=str(root),verified_files=len(needed),
                             byte_inventory_sha256=digest({p:plan['rows'][corpus][p]['sha256'] for p in needed}))
    write_json(out/'audio_gate.json',dict(time=now(),passed=True,corpora=summary))


def close_interrupted_attempts(unit, out, lock):
    """Called only after explicit retry and successful OS-lock reacquisition."""
    folder = out/'units'/unit['unit_id']
    require(not (folder/'DONE').exists(), 'committed unit cannot be abandoned')
    reservations = sorted((folder/'reservations').iterdir())
    require(all(p.is_file() and re.fullmatch(r'000[12]\.json', p.name) for p in reservations),
            'invalid or partial prior reservation inventory')
    require([int(p.stem) for p in reservations] == list(range(1, len(reservations)+1))
            and len(reservations) <= 2, 'prior reservation sequence differs')
    rows = [r for r in ledger_rows(out) if r.get('unit_id') == unit['unit_id']]
    known = {'attempts/'+p.stem for p in reservations}
    require(all(r.get('attempt') in known for r in rows), 'prior event has no matching reservation')
    pending = []
    for path in reservations:
        identity = read_json(path)
        require(identity.get('unit_id') == unit['unit_id'] and identity.get('unit_sha256') == digest(unit)
                and identity.get('lock_sha256') == lock['lock_sha256'], 'prior reservation identity differs')
        attempt = 'attempts/'+path.stem
        state, had_start = 'reserved', False
        for row in [r for r in rows if r['attempt'] == attempt]:
            kind = row['event']
            if kind == 'unit_start':
                require(state == 'reserved', 'duplicate or post-terminal prior start')
                state, had_start = 'started', True
            elif kind == 'unit_failed':
                require(state == 'started', 'prior failure lacks a unique open start')
                state = 'failed'
            elif kind == 'unit_attempt_abandoned':
                require(state in ('reserved', 'started') and row.get('had_start') is had_start
                        and row.get('reason') == ABANDON_REASON, 'invalid prior abandonment')
                state = 'abandoned'
            else:
                raise ValueError('unexpected prior event without DONE: '+str(kind))
        if state in ('reserved', 'started'):
            pending.append(dict(event='unit_attempt_abandoned', unit_id=unit['unit_id'],
                                attempt=attempt, had_start=had_start, reason=ABANDON_REASON))
    # Validate everything before appending any terminal event. Failed attempts stay failed.
    for row in pending:
        event(out, row)


def execute(args):
    import torch
    import torchaudio
    from v3.deploy.engines_deploy import WaveCache
    plan, lock = checked_plan(args.repo, args.out)
    require(file_sha(args.model_path) == lock['model_sha256'], 'model file changed')
    roots = read_json(args.roots)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    device = torch.device(args.device)
    require(device.type == 'cuda' and torch.cuda.is_available(), 'formal/pilot requires CUDA')
    env = dict(torch=torch.__version__, torchaudio=torchaudio.__version__, cuda=torch.version.cuda,
               device=torch.cuda.get_device_name(device), threads=torch.get_num_threads(),
               pid=os.getpid(), started_at=now())
    units = [u for u in plan['units'] if u['phase'] == lock['phase']]
    if args.unit_id:
        wanted = set(args.unit_id)
        require(wanted <= {u['unit_id'] for u in units}, 'unknown requested unit')
        units = [u for u in units if u['unit_id'] in wanted]
    caches, completed_here, began = {}, 0, time.perf_counter()
    with execution_lock(args.out / 'execution.lock'):
        verify_audio_inputs(plan,units,roots,args.out)
        event(args.out, dict(event='process_start', environment=env, lock_sha256=lock['lock_sha256']))
        for unit in units:
            folder = args.out / 'units' / unit['unit_id']
            if (folder / 'DONE').exists():
                verify_done(unit, args.out, lock,plan['rows'][unit['corpus']])
                reconcile_done(unit,args.out)
                continue
            if completed_here >= args.max_units or time.perf_counter() - began >= args.max_hours * 3600:
                event(args.out, dict(event='operational_yield', completed_here=completed_here))
                return
            attempts = sorted(folder.glob('reservations/[0-9][0-9][0-9][0-9].json')) if folder.exists() else []
            require(not attempts or args.retry_failed, 'prior interrupted/failed attempt; explicit retry flag required')
            require(len(attempts) < 2, 'at most two attempts per unit')
            if attempts:
                close_interrupted_attempts(unit, args.out, lock)
            attempt = folder / 'attempts' / f'{len(attempts)+1:04d}'
            write_json(folder/'reservations'/f'{len(attempts)+1:04d}.json',
                       dict(time=now(),unit_id=unit['unit_id'],unit_sha256=digest(unit),lock_sha256=lock['lock_sha256']))
            event(args.out, dict(event='unit_start', unit_id=unit['unit_id'], attempt=attempt.relative_to(folder).as_posix()))
            torch.cuda.reset_peak_memory_stats(device)
            try:
                corpus = unit['corpus']
                if corpus not in caches:
                    caches[corpus] = WaveCache(Path(roots[corpus]), 16000)
                validate_unit(unit, plan['rows'][corpus])
                info = fit_unit(unit, plan['rows'][corpus], Path(roots[corpus]), args.model_path,
                                attempt, device, cache=caches[corpus])
                receipt = dict(schema='ser-final-program-receipt-1', unit_id=unit['unit_id'],
                               phase=lock['phase'], unit_sha256=digest(unit), lock_sha256=lock['lock_sha256'],
                               environment=env, completed_at=now(), info=info,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                write_json(attempt / 'receipt.json', receipt)
                artifacts = {p.relative_to(folder).as_posix(): file_sha(p) for p in attempt.iterdir()}
                done = dict(schema='ser-final-program-done-1', unit_id=unit['unit_id'],
                            unit_sha256=digest(unit), lock_sha256=lock['lock_sha256'],
                            attempt=attempt.relative_to(folder).as_posix(), artifacts=artifacts)
                verify_done(unit,args.out,lock,plan['rows'][unit['corpus']],done_doc=done)
                write_json(folder / 'DONE', done)
                event(args.out, dict(event='unit_done', unit_id=unit['unit_id'], done_sha256=file_sha(folder / 'DONE')))
                completed_here += 1
                print(json.dumps({'unit': unit['unit_id'], 'seconds': round(info['wall_seconds'], 3),
                                  'peak_GiB': round(receipt['peak_allocated_bytes']/2**30, 3),
                                  'complete_this_process': completed_here}), flush=True)
            except BaseException:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'last_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                event(args.out, dict(event='commit_requires_recovery' if (folder/'DONE').exists() else 'unit_failed',
                                    unit_id=unit['unit_id'], attempt=attempt.relative_to(folder).as_posix()))
                raise
        checked_plan(args.repo, args.out)
        event(args.out, dict(event='process_complete', completed_here=completed_here))


def audit_phase(repo,out):
    """Full declared phase, payload semantics and ledger; never score outer labels."""
    plan,lock=checked_plan(repo,out)
    expected=[u for u in plan['units'] if u['phase']==lock['phase']]
    require(len(expected)==(384 if lock['phase']=='formal' else 4),'phase extent differs')
    actual={p.name for p in (out/'units').iterdir() if p.is_dir()}
    require(actual=={u['unit_id'] for u in expected},'unit directory inventory incomplete or unexpected')
    dones,artifacts,receipts={},{},{}
    with execution_lock(out/'execution.lock'):
        for i,unit in enumerate(expected,1):
            require((out/'units'/unit['unit_id']/'DONE').is_file(),'phase incomplete: '+unit['unit_id'])
            done,receipt=verify_done(unit,out,lock,plan['rows'][unit['corpus']],weights=True)
            reconcile_done(unit,out)
            dones[unit['unit_id']]=file_sha(out/'units'/unit['unit_id']/'DONE')
            artifacts[unit['unit_id']]=done['artifacts']
            receipts[unit['unit_id']]=receipt
            if i%20==0:
                print(json.dumps({'audit_verified':i,'expected':len(expected)}),flush=True)
        pairs={}
        for u in expected:
            pairs.setdefault(u['pair_id'],{})[u['arm']]=u
        for pair in pairs.values():
            if 'B' in pair:
                a,b=pair['A'],pair['B']
                require(a['report_batches']==b['report_batches'] and a['seeds']==b['seeds'], 'paired frame differs')
                require(receipts[a['unit_id']]['info']['initial_state_sha256']==
                        receipts[b['unit_id']]['info']['initial_state_sha256'],'paired initial weights differ')
        journal = audit_events(expected, out, lock, ledger_rows(out))
        checked_plan(repo,out)
        gate=dict(schema='ser-final-program-complete-gate-1',pass_=True,phase=lock['phase'],
                  expected_units=len(expected),verified_units=len(dones),plan_sha256=plan['plan_sha256'],
                  lock_sha256=lock['lock_sha256'],done_sha256=dones,artifacts_sha256=artifacts,
                  ledger_sha256=file_sha(out/'ledger.jsonl'),ledger_audit=journal,
                  reservation_sha256=journal['reservation_sha256'],completed_at=now(),
                  scope='Full declared phase including checkpoint hashes/structure, predictions, history and ledger; no outer scores')
        gate['pass']=gate.pop('pass_')
        write_json(out/'COMPLETE_GATE.json',gate)
        print(json.dumps({'phase':lock['phase'],'full_phase_pass':True,'units':len(dones)}),flush=True)
        return gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['freeze', 'run', 'audit'])
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--phase', choices=['pilot', 'formal'], default='pilot')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--roots', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--unit-id', action='append')
    p.add_argument('--max-units', type=int, default=10000)
    p.add_argument('--max-hours', type=float, default=12)
    p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    a.repo, a.out = a.repo.resolve(), a.out.resolve()
    if a.action == 'freeze':
        require(a.plan is not None and a.model_path is not None, 'plan and model paths required')
        freeze(a.repo, a.plan, a.out, a.phase, a.model_path)
    elif a.action == 'run':
        require(a.roots is not None and a.model_path is not None, 'audio roots and model path required')
        execute(a)
    else:
        audit_phase(a.repo,a.out)


if __name__ == '__main__':
    main()

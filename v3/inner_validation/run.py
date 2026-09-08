"""Sealed, append-only fits and integrity gate; no scientific scoring."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import platform
import time
from datetime import datetime,timezone
import numpy as np
from v3.inner_validation.plan import load_plan,input_gate,read_metadata,digest,file_sha,require

ROLES=('val_seen','val_unseen','test')
CHECKPOINTS=('best_seen','best_unseen','last')
ARTIFACTS=('predictions.npz','checkpoint.pt','history.json','receipt.json')

def now():return datetime.now(timezone.utc).isoformat()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    require(not path.exists(),'refusing to replace existing record')
    os.replace(temporary,path)

def event(root,kind,**values):
    with (Path(root)/'ledger.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(event=kind,time=now(),**values),allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())

def verify_arrays(path,unit,rows=None):
    keys={f'{r}__{k}' for r in ROLES for k in ('paths','labels')}
    keys|={f'{r}__{c}__logits' for r in ROLES for c in CHECKPOINTS}
    with np.load(path,allow_pickle=False) as a:
        require(set(a.files)==keys,'prediction key mismatch')
        for role in ROLES:
            paths,labels=a[f'{role}__paths'],a[f'{role}__labels']
            require(paths.dtype.kind=='U' and paths.tolist()==unit[role],'prediction order differs')
            require(labels.dtype==np.int64 and labels.shape==(len(paths),) and set(labels.tolist())==set(range(6)), 'invalid labels')
            if rows is not None:
                require(labels.tolist()==[int(rows[p]['label_index']) for p in paths],'truth mismatch')
            for ck in CHECKPOINTS:
                scores=a[f'{role}__{ck}__logits']
                require(scores.dtype==np.float64 and scores.shape==(len(paths),6) and np.isfinite(scores).all(),'invalid logits')

def verify_unit(unit,phase_root,plan_sha,rows=None):
    folder=Path(phase_root)/'units'/unit['unit_id'];done=folder/'DONE'
    if not done.exists():return None
    d=read(done);phase=Path(phase_root).name
    require(d['schema']=='ser-dual-validation-done-1' and d['unit_id']==unit['unit_id']
            and d['phase']==phase and d['plan_sha256']==plan_sha and d['unit_sha256']==digest(unit),'DONE identity mismatch')
    expected={f'attempts/0001/{n}' for n in ARTIFACTS}
    require(set(d['artifacts'])==expected,'unexpected artifact list')
    for name,sha in d['artifacts'].items():
        require((folder/name).is_file() and file_sha(folder/name)==sha,'artifact missing/corrupt: '+name)
    receipt=read(folder/'attempts/0001/receipt.json')
    require(receipt['schema']=='ser-dual-validation-receipt-1' and receipt['plan_sha256']==plan_sha and receipt['unit_sha256']==digest(unit)
            and receipt['unit_id']==unit['unit_id'] and receipt['phase']==phase,'receipt identity mismatch')
    require(receipt['epochs_run']==15 and receipt['checkpoint_reload_verified'] is True,'epoch/reload gate failed')
    for key in ('fit_seconds','peak_cuda_bytes','checkpoint_reload_max_abs_diff'):
        value=receipt[key];require(type(value) in (int,float) and math.isfinite(value) and value>=0,'invalid resource/reload statistic')
    history=read(folder/'attempts/0001/history.json')
    require([h['epoch'] for h in history]==list(range(1,16)),'epoch history incomplete')
    require(all(math.isfinite(h[k]) for h in history for k in ('train_loss','val_seen_loss','val_unseen_loss')),'nonfinite history')
    require(all(h['optimizer_steps']==36 and type(h['scaler_skipped_steps']) is int and
                0<=h['scaler_skipped_steps']<=36 for h in history),'optimizer step budget changed')
    for cell in ('seen','unseen'):
        require(receipt[f'best_{cell}_epoch']==min(history,key=lambda h:h[f'val_{cell}_loss'])['epoch'],'checkpoint selection changed')
    unique=sorted({receipt['best_seen_epoch'],receipt['best_unseen_epoch'],15})
    require(receipt['checkpoint_unique_epochs']==unique and receipt['checkpoint_unique_epoch_count']==len(unique),'epoch storage inventory differs')
    verify_arrays(folder/'attempts/0001/predictions.npz',unit,rows)
    return receipt

def phase_units(plan,phase):
    require(phase in ('pilot','formal'),'unknown phase')
    return plan['units'] if phase=='formal' else [u for u in plan['units'] if u['unit_id'] in plan['pilot_units']]

def reconcile_done_event(root,unit,plan_sha,rows):
    require(verify_unit(unit,root,plan_sha,rows) is not None,'cannot recover an unverified completion')
    payload=(Path(root)/'ledger.jsonl').read_bytes()
    require(payload.endswith(b'\n'),'truncated ledger needs explicit repair')
    events=[json.loads(line) for line in payload.decode().splitlines()]
    own=[e for e in events if e.get('unit_id')==unit['unit_id']]
    require(sum(e['event']=='unit_start' for e in own)==1 and
            not any(e['event']=='unit_failed' for e in own),'completion lacks a unique nonfailed start')
    completed=[e for e in own if e['event']=='unit_done']
    require(len(completed)<=1,'duplicate completion event')
    if not completed:
        event(root,'unit_done',unit_id=unit['unit_id'],recovered_from_verified_done=True,
              done_sha256=file_sha(Path(root)/'units'/unit['unit_id']/'DONE'))

def result_gate(plan,outroot,phase,rows):
    root=Path(outroot)/phase;units=phase_units(plan,phase);mapping={}
    for u in units:
        require(verify_unit(u,root,plan['plan_sha256'],rows) is not None,'missing unit '+u['unit_id'])
        mapping[u['unit_id']]=file_sha(root/'units'/u['unit_id']/'DONE')
    ledger=root/'ledger.jsonl';payload=ledger.read_bytes()
    require(payload.endswith(b'\n'),'truncated ledger')
    events=[json.loads(line) for line in payload.decode().splitlines()]
    starts=Counter(e['unit_id'] for e in events if e['event']=='unit_start')
    stops=Counter(e['unit_id'] for e in events if e['event']=='unit_done')
    require(starts==stops==Counter(mapping.keys()),'start/completion ledger does not match full plan')
    require(not any(e['event']=='unit_failed' for e in events),'failed unit needs reviewed repair')
    gate=dict(schema='ser-dual-validation-result-gate-1',**{'pass':True},phase=phase,
              plan_sha256=plan['plan_sha256'],units=len(units),done_sha256=mapping,
              ledger_sha256=file_sha(ledger),verified_at=now(),scores_computed=False)
    gate['gate_sha256']=digest(gate);return gate

def run(args,plan,rows):
    import torch
    from v3.inner_validation.engine import fit_dual
    from v3.deploy.engines_deploy import WaveCache
    require(Path(__file__).resolve().parents[2]==args.repo.resolve() and
            Path(__import__('inspect').getfile(fit_dual)).resolve()==
            (args.repo/'v3/inner_validation/engine.py').resolve(),'loaded execution source is outside verified repository')
    require(torch.cuda.is_available(),'real execution requires CUDA')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda:0')
    selected=phase_units(plan,args.phase)
    if args.unit_ids:
        ids=read(args.unit_ids)
        require(isinstance(ids,list) and len(ids)==len(set(ids)) and ids,'invalid requested IDs')
        require(set(ids)<={u['unit_id'] for u in selected},'unknown/out-of-phase IDs')
        selected=[u for u in selected if u['unit_id'] in ids]
    root=args.outroot/args.phase;root.mkdir(parents=True,exist_ok=True)
    lock=root/'RUNNING.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),started=now())).encode());os.close(descriptor)
    try:
        env=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
            threads=torch.get_num_threads(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32)
        wave=WaveCache(args.audio_root,16000)
        event(root,'batch_start',unit_ids=[u['unit_id'] for u in selected],environment=env)
        for unit in selected:
            if verify_unit(unit,root,plan['plan_sha256'],rows):
                reconcile_done_event(root,unit,plan['plan_sha256'],rows);continue
            free=__import__('shutil').disk_usage(root).free
            require(free>1_500_000_000,'insufficient disk for another lossless checkpoint')
            folder=root/'units'/unit['unit_id'];attempt=folder/'attempts/0001'
            attempt.mkdir(parents=True,exist_ok=False)
            event(root,'unit_start',unit_id=unit['unit_id'])
            started=time.perf_counter()
            try:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
                arrays,history,info=fit_dual(unit,rows,args.model,args.audio_root,device,attempt/'checkpoint.pt',wave)
                elapsed=time.perf_counter()-started
                with (attempt/'predictions.npz').open('xb') as f:
                    np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
                atomic(attempt/'history.json',history)
                receipt=dict(schema='ser-dual-validation-receipt-1',unit_id=unit['unit_id'],
                    unit_sha256=digest(unit),plan_sha256=plan['plan_sha256'],phase=args.phase,
                    fit_seconds=elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                    environment=env,finished_at=now(),**info)
                atomic(attempt/'receipt.json',receipt)
                verify_arrays(attempt/'predictions.npz',unit,rows)
                atomic(folder/'DONE',dict(schema='ser-dual-validation-done-1',unit_id=unit['unit_id'],
                    unit_sha256=digest(unit),plan_sha256=plan['plan_sha256'],phase=args.phase,
                    artifacts={f'attempts/0001/{name}':file_sha(attempt/name) for name in ARTIFACTS}))
                verify_unit(unit,root,plan['plan_sha256'],rows)
                event(root,'unit_done',unit_id=unit['unit_id'],fit_seconds=elapsed)
                print(json.dumps(dict(unit=unit['unit_id'],phase=args.phase,done=True,
                    seconds=round(elapsed,3),peak_cuda_bytes=receipt['peak_cuda_bytes'])),flush=True)
                del arrays,history;gc.collect()
            except BaseException as exc:
                committed=False
                if (folder/'DONE').exists():
                    try:
                        reconcile_done_event(root,unit,plan['plan_sha256'],rows);committed=True
                    except Exception:pass
                if not committed:
                    event(root,'unit_failed',unit_id=unit['unit_id'],error_type=type(exc).__name__,error=str(exc))
                raise
        event(root,'batch_done',units=len(selected))
    finally:
        lock.unlink()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('run','input-gate','result-gate'))
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--outroot',type=Path);ap.add_argument('--phase',choices=('pilot','formal'),default='formal')
    ap.add_argument('--audio-root',type=Path);ap.add_argument('--model',type=Path)
    ap.add_argument('--unit-ids',type=Path);ap.add_argument('--out',type=Path)
    args=ap.parse_args();plan=load_plan(args.plan,args.repo)
    rows={r['relative_path']:r for r in read_metadata(args.repo)['clean']}
    if args.action=='result-gate': result=result_gate(plan,args.outroot,args.phase,rows)
    else:
        result=input_gate(plan,args.repo,args.audio_root,args.model)
        if args.action=='run':run(args,plan,rows);return
    if args.out:atomic(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='done_sha256'}))

if __name__=='__main__':main()

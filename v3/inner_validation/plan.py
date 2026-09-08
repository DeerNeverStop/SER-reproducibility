"""Outcome-blind plan and invariant checks for the paired validation experiment."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from v3.data_design.core_plan import read_metadata, ranked, digest, file_sha, write_json, require
from v3.speaker_coverage.plan import BASE_SOURCE_PATHS

PROGRAM = 'SER26-DUAL-VALIDATION-1'
DRAWS, FOLDS = 24, 5
BASE_SHA = '136a3e720c04f2c77bf7a4dc6a3868b14d5a2c145a988114b733cb1a8428be98'
ROLES = ('fit', 'val_seen', 'val_unseen', 'test')
ANALYSIS = dict(primary='draw_mean[(V_seen-T_seen)-(V_unseen-T_unseen)]',draws=24,
    selection='minimum loss checkpoint; maximum own-validation UAR config; first index ties',
    fixed_config_index=3,bootstrap_repetitions=50000,bootstrap_seed=2026090701)
CONFIGS = [dict(model='wavlm_base_plus_ft', epochs=15, batch_size=16,
    fp16=True, early_stopping=False, crop_seconds=3., eval_cap_seconds=10.,
    trainable_layers='top4+head', lr_encoder=enc, lr_head=head, weight_decay=.01)
    for enc in (1e-5,5e-5) for head in (3e-4,1e-3)]

def seed(*parts):
    return int(digest([PROGRAM,*parts])[:16],16) % (2**31-1)

def choose(values,*parts):
    return ranked(values, ':'.join(map(str,(PROGRAM,*parts))))

def source_files(repo):
    names=set(BASE_SOURCE_PATHS)
    for package in ('v3','v3/data_design','v3/deploy','v3/speaker_coverage','v2/ser_v2'):
        init=repo/package/'__init__.py'
        if init.exists():names.add(init.relative_to(repo).as_posix())
    names.update(p.relative_to(repo).as_posix() for p in (repo/'v3/speaker_coverage').glob('*.py'))
    names.update(p.relative_to(repo).as_posix() for p in (repo/'v3/inner_validation').glob('*.py'))
    names.add('v3/inner_validation/PROTOCOL.md')
    return {name:file_sha(repo/name) for name in sorted(names)}

def generate(meta):
    units=[]
    for draw in range(DRAWS):
        outer={sex:choose([s for s in meta['speakers'] if meta['sex'][s]==sex],draw,sex,'outer')
               for sex in ('Female','Male')}
        for fold in range(FOLDS):
            test_people=sorted(s for group in outer.values() for s in group[fold::FOLDS])
            prompts=choose(meta['prompts'],draw,fold,'texts')
            fit_texts,query_texts=prompts[:4],prompts[4:6]
            eligible=[s for s in meta['speakers'] if s not in test_people and
                all((s,p,c) in meta['cells'] for p in fit_texts+query_texts for c in range(6))]
            fit_people=[];unseen=[]
            for sex in ('Female','Male'):
                group=choose([s for s in eligible if meta['sex'][s]==sex],draw,fold,sex,'development')
                require(len(group)>=24,'fewer than 24 complete candidates in sex stratum')
                fit_people.extend(group[:12]);unseen.extend(group[12:24])
            def paths(people,texts,complete=True):
                out=[]
                for s in sorted(people):
                    for p in texts:
                        for c in range(6):
                            row=meta['cells'].get((s,p,c))
                            require(row is not None or not complete,'missing development cell')
                            if row: out.append(row['relative_path'])
                return sorted(out)
            panel=dict(fit=paths(fit_people,fit_texts),val_seen=paths(fit_people,query_texts),
                val_unseen=paths(unseen,query_texts),test=paths(test_people,query_texts,False),
                fit_speakers=sorted(fit_people),unseen_speakers=sorted(unseen),test_speakers=test_people,
                fit_prompts=fit_texts,query_prompts=query_texts,eligible_speakers=sorted(eligible))
            for index,config in enumerate(CONFIGS):
                units.append(dict(unit_id=f'dual_d{draw:02d}_f{fold}_c{index}',draw=draw,fold=fold,
                    config_index=index,train_seed=seed(draw,fold,'train'),config=config.copy(),**panel))
    return units

def validate(plan,meta):
    require(plan['schema']=='ser-dual-validation-plan-1' and plan['program']==PROGRAM,'unknown plan schema/program')
    require(plan['analysis']==ANALYSIS,'analysis contract changed')
    units=plan['units']; require(len(units)==480,'formal grid must contain exactly 480 units')
    rows={r['relative_path']:r for r in meta['clean']}
    require(len({u['unit_id'] for u in units})==480,'duplicate unit')
    require({(u['draw'],u['fold'],u['config_index']) for u in units}==
        {(d,f,c) for d in range(24) for f in range(5) for c in range(4)},'grid incomplete')
    require(plan['pilot_units']==[u['unit_id'] for u in units if u['draw']==u['fold']==0], 'pilot must be fixed four units')
    for u in units:
        require(u['config']==CONFIGS[u['config_index']],'unknown configuration')
        require(u['train_seed']==seed(u['draw'],u['fold'],'train'),'wrong seed')
        require(u['unit_id']==f"dual_d{u['draw']:02d}_f{u['fold']}_c{u['config_index']}",'bad ID')
        require(len(u['fit'])==576 and len(u['val_seen'])==len(u['val_unseen'])==288,'budget changed')
        for role in ROLES:
            values=u[role]
            require(values==sorted(set(values)) and set(values)<=set(rows),'unsafe/duplicate/nonmanifest path')
            require(set(int(rows[p]['label_index']) for p in values)==set(range(6)),'role missing class')
        for i,a in enumerate(ROLES):
            for b in ROLES[i+1:]: require(not set(u[a])&set(u[b]),'overlapping recordings')
        people={role:{rows[p]['speaker'] for p in u[role]} for role in ROLES}
        require(people['fit']==people['val_seen']==set(u['fit_speakers']) and len(people['fit'])==24,'seen identity mismatch')
        require(people['val_unseen']==set(u['unseen_speakers']) and len(people['val_unseen'])==24,'unseen count mismatch')
        require(people['test']==set(u['test_speakers']),'test people mismatch')
        require(not people['fit']&people['val_unseen'] and not (people['fit']|people['val_unseen'])&people['test'],'identity leakage')
        require(len(u['fit_prompts'])==4 and len(u['query_prompts'])==2 and
                not set(u['fit_prompts'])&set(u['query_prompts']),'text sets not disjoint')
        for role in ROLES:
            texts=u['fit_prompts'] if role=='fit' else u['query_prompts']
            require({rows[p]['sentence'] for p in u[role]}==set(texts),'wrong prompt support')
            require(all(meta['cells'][(rows[p]['speaker'],rows[p]['sentence'],int(rows[p]['label_index']))]['relative_path']==p
                        for p in u[role]),'not prespecified representative')
            if role!='test':
                counts=Counter((meta['sex'][rows[p]['speaker']],rows[p]['sentence'],int(rows[p]['label_index'])) for p in u[role])
                require(len(counts)==2*len(texts)*6 and set(counts.values())=={12},'sex/text/class budgets unequal')
            else:
                require(all(any(rows[p]['speaker']==s and int(rows[p]['label_index'])==c for p in u[role])
                            for s in people['test'] for c in range(6)),'test person missing a class')
        panel_keys=[*ROLES,'fit_speakers','unseen_speakers','test_speakers','fit_prompts','query_prompts','eligible_speakers']
        if u['config_index']:
            first=next(x for x in units if x['draw']==u['draw'] and x['fold']==u['fold'] and x['config_index']==0)
            require(all(u[k]==first[k] for k in panel_keys),'configuration changes data')
    for draw in range(24):
        groups=[u['test_speakers'] for u in units if u['draw']==draw and u['config_index']==0]
        require(Counter(s for group in groups for s in group)==Counter(meta['speakers']),'outer people not partitioned')
    require(units==generate(meta),'plan does not replay the frozen randomization or complete recording panels')
    return rows

def load_plan(path,repo):
    require(Path(__file__).resolve().parents[2]==Path(repo).resolve(),'loaded planner is outside the verified repository')
    plan=json.loads(Path(path).read_text(encoding='utf-8'))
    require(plan['program']==PROGRAM and plan['draft'] is False,'not a frozen formal plan')
    require(digest({k:v for k,v in plan.items() if k!='plan_sha256'})==plan['plan_sha256'],'plan hash changed')
    require(plan['sources']==source_files(Path(repo)),'source freeze mismatch')
    meta=read_metadata(Path(repo))
    require(plan['input']==meta['input'],'metadata changed')
    validate(plan,meta)
    return plan

def input_gate(plan,repo,audio_root,model):
    require(file_sha(model)==plan['base_file_sha256']==BASE_SHA,'base weights changed')
    meta=read_metadata(Path(repo));validate(plan,meta)
    for r in meta['clean']:
        require(file_sha(Path(audio_root)/r['relative_path'])==r['sha256'],'raw recording changed: '+r['relative_path'])
    return dict(pass_gate=True,plan_sha256=plan['plan_sha256'],raw_files=7435,
                base_file_sha256=BASE_SHA,source_count=len(plan['sources']),scores_read=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--draft',action='store_true');args=ap.parse_args()
    require(not args.out.exists(),'refusing to overwrite a plan')
    meta=read_metadata(args.repo);units=generate(meta)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip()
    plan=dict(schema='ser-dual-validation-plan-1',program=PROGRAM,draft=args.draft,
        source_commit=commit,sources=source_files(args.repo),input=meta['input'],base_file_sha256=BASE_SHA,
        units=units,pilot_units=[u['unit_id'] for u in units if u['draw']==u['fold']==0],
        analysis=ANALYSIS.copy())
    validate(plan,meta);plan['plan_sha256']=digest(plan);write_json(args.out,plan)
    print(json.dumps(dict(plan_sha256=plan['plan_sha256'],formal_units=len(units),pilot_units=4,draft=args.draft)))

if __name__=='__main__':main()

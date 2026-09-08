"""Outcome-free native-label panels for 360 main + 24 control + 4 pilot fits.

Only manifests/demographics are read. Audio, model weights, predictions and
scientific results are not read. Execution/source freezing is a separate gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from v2.ser_v2.corpora import CORPORA, apply_hygiene, load_manifest, parse_filename
from v3.data_design.core_plan import digest, file_sha, read_metadata, require, write_json


PROGRAM = 'SER26-FINAL-CHECKPOINT-PROGRAM-1'
DRAWS, FOLDS = 24, 5
CORPUS_ORDER = ('cremad', 'subesco', 'ravdess')
CONFIG = dict(model='wavlm_base_plus_ft', trainable_layers='top4+head', epochs=15,
              batch_size=16, fp16=True, lr_encoder=5e-5, lr_head=1e-3,
              weight_decay=.01, crop_seconds=3.0, eval_cap_seconds=10.0,
              early_stopping=False, hyperparameter_search=False)
DESIGNS = {
    'cremad': dict(n_classes=6, people_per_group=24, fit_texts=4, query_texts=2,
                   raw_count=7442, clean_count=7435, speakers=91,
                   representative='MD preferred, otherwise XX; same legacy core representative'),
    'subesco': dict(n_classes=7, people_per_group=8, fit_texts=4, query_texts=2,
                   raw_count=7000, clean_count=6998, speakers=20,
                   representative='T1 only; no fallback or score-based selection'),
    'ravdess': dict(n_classes=8, people_per_group=8, fit_texts=1, query_texts=1,
                   raw_count=1440, clean_count=1439, speakers=24,
                   representative='normal intensity and R1 only; no fallback'),
}
ANALYSIS = dict(
    main_trajectories=360, swap_control_trajectories=24, formal_fits=384, pilot_fits=4,
    main_arm='A', swap_control=dict(corpus='cremad', fold=0, arm='B', epoch=15,
                                  paired_A='same formal corpus/draw/fold A', inference=False),
    tests=[dict(corpus=c, estimand=e) for c in CORPUS_ORDER for e in ('delta_CE', 'J')],
    estimands=dict(delta_CE='T(seen,CE)-T(unseen,CE)',
                   delta_UAR='T(seen,UAR)-T(unseen,UAR)', J='delta_CE-delta_UAR'),
    family_size=6, multiplicity='Holm, familywise alpha .05', alternative='two-sided',
    aggregation='whole-role native-class UAR; five-fold mean within draw; equal 24-draw mean',
    inference='one-sample t on 24 complete draw effects, df23; conditional fixed-corpus scope',
    pointwise_interval=.95, bootstrap=False, practical_reference_pp=1.0,
    equivalence_test=False, additional_control_hypothesis_tests=0,
    tie='earliest epoch for exact CE/UAR ties; UAR equal supports uses integer correct count',
    scientific_scoring_requires_all_formal_units=True,
)
EXECUTION = dict(
    native_labels=True, dynamic_head=True, full_epochs=15,
    report_batches='separate A/B/outer batches; reset batch boundary per group; never mix outer with validation',
    train_mode='model.train every epoch; eval only for prediction; validation must not consume training RNG',
    randomization='separate initialization/order/crop/torch_training seeds; arm excluded from pair seed',
    crop_key='[PROGRAM,crop_seed,epoch_1based,fit_slot_index_0based]',
    crop_integer_mapping='(first_64_hash_bits*(max(0,n_samples-48000)+1))//2**64',
    paired_initial_full_state_hash=True, report_labels_do_not_change_training=True,
    outer_inputs_do_not_change_validation_predictions_or_selection=True,
    A_predictions='all 15 epochs; four validation winners plus last deduplicated lossless deltas',
    B_predictions='last epoch only; no checkpoint selection; one lossless last delta',
    restore_every_saved_state=True, no_outer_score_during_training=True,
)


def ranked(values, *parts):
    return sorted(values, key=lambda value: (digest([PROGRAM, *parts, value]), value))


def seed(*parts):
    return int(digest([PROGRAM, *parts])[:16], 16) % (2**31-1)


def crop_start(crop_seed, epoch, slot, n_samples):
    """Shared uniform location, exact integer arithmetic including short waves."""
    require(all(type(x) is int for x in (crop_seed, epoch, slot, n_samples)), 'integer crop inputs required')
    require(crop_seed >= 0 and 1 <= epoch <= 15 and slot >= 0 and n_samples >= 0, 'invalid crop coordinates')
    h64 = int(digest([PROGRAM, crop_seed, epoch, slot])[:16], 16)
    return (h64 * (max(0, n_samples-48000)+1)) // 2**64


def _load_metadata(repo):
    registry_path = repo/'v2/plan_rc2/hygiene_log.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    crema = read_metadata(repo)
    metas, inputs = {}, {'hygiene_registry': dict(path=registry_path.relative_to(repo).as_posix(), sha256=file_sha(registry_path))}
    for corpus in CORPUS_ORDER:
        spec, design = CORPORA[corpus], DESIGNS[corpus]
        path = repo/f'v2/manifests/{corpus}_manifest.csv'
        raw = load_manifest(path)
        require(len(raw) == design['raw_count'], 'raw corpus size changed: '+corpus)
        require(len({r['relative_path'] for r in raw}) == len(raw), 'duplicate manifest paths')
        for row in raw:
            rel = row['relative_path']
            require(not PurePosixPath(rel).is_absolute() and '..' not in PurePosixPath(rel).parts
                    and ':' not in rel and '\\' not in rel, 'unsafe relative path')
            parsed = parse_filename(corpus, Path(rel).name)
            require(parsed is not None and all(row[k] == v for k,v in parsed.items()), 'filename/manifest semantics differ')
            require(row['corpus'] == corpus and 0 <= row['label_index'] < len(spec.labels)
                    and spec.labels[row['label_index']] == row['label'], 'native label mapping changed')
        clean, hygiene = apply_hygiene(corpus, raw)
        require(hygiene == registry[corpus], 'hygiene does not match recorded registry: '+corpus)
        require(len(clean) == design['clean_count'], 'clean corpus size changed')
        speakers = sorted({r['speaker'] for r in clean})
        require(len(speakers) == design['speakers'], 'speaker count changed')
        sex = ({s: {'Female':'F','Male':'M'}[crema['sex'][s]] for s in speakers} if corpus == 'cremad'
               else {r['speaker']:r['sex'] for r in clean})
        require(set(sex.values()) == {'F','M'}, 'missing source sex stratum')
        cells, candidates = {}, defaultdict(list)
        for row in clean:
            require(corpus == 'cremad' or row['sex'] == sex[row['speaker']], 'inconsistent source sex')
            row['sex_stratum'] = sex[row['speaker']]
            candidates[row['speaker'],row['sentence'],row['label_index']].append(row)
        for key, options in candidates.items():
            if corpus == 'cremad':
                options = [r for r in options if r['intensity'] in ('MD','XX')]
                if options:
                    cells[key] = min(options,key=lambda r:(0 if r['intensity']=='MD' else 1,r['relative_path']))
            elif corpus == 'subesco':
                options = [r for r in options if r['take'] == 'T1']
                require(len(options) == 1, 'SUBESCO T1 cell absent/ambiguous')
                cells[key] = options[0]
            else:
                options = [r for r in options if r['take'] == 'R1' and r['intensity'] == 'normal']
                require(len(options) == 1, 'RAVDESS normal/R1 cell absent/ambiguous')
                cells[key] = options[0]
        if corpus == 'cremad':
            require({r['relative_path'] for r in clean} == {r['relative_path'] for r in crema['clean']}, 'CREMA cleanup differs from legacy')
            require({k:v['relative_path'] for k,v in cells.items()} == {k:v['relative_path'] for k,v in crema['cells'].items()}, 'CREMA representatives differ')
        inputs[corpus] = dict(path=path.relative_to(repo).as_posix(),sha256=file_sha(path))
        metas[corpus] = dict(rows={r['relative_path']:r for r in clean},cells=cells,sex=sex,
                             speakers=speakers,prompts=sorted({r['sentence'] for r in clean}),hygiene=hygiene)
    inputs['cremad_demographics'] = dict(path=crema['input']['demographics_path'], sha256=crema['input']['demographics_sha256'])
    return metas, inputs


def _panel(meta, corpus, phase, draw, fold):
    design = DESIGNS[corpus]
    n = design['n_classes']; per_sex = design['people_per_group']//2
    parts = phase, corpus, draw, fold
    outer = {sex: ranked([s for s in meta['speakers'] if meta['sex'][s] == sex], phase,corpus,draw,sex,'outer') for sex in ('F','M')}
    test = sorted(s for people in outer.values() for s in people[fold::FOLDS])
    prompts = ranked(meta['prompts'], *parts, 'texts')
    fit_prompts = prompts[:design['fit_texts']]
    query_prompts = prompts[design['fit_texts']:design['fit_texts']+design['query_texts']]
    eligible = [s for s in meta['speakers'] if s not in test and
                all((s,p,c) in meta['cells'] for p in fit_prompts+query_prompts for c in range(n))]
    groups = {'A':[], 'B':[]}
    for sex in ('F','M'):
        people = ranked([s for s in eligible if meta['sex'][s] == sex],*parts,sex,'groups')
        require(len(people) >= 2*per_sex, 'insufficient complete development people; no re-draw: '+str(parts))
        groups['A'].extend(people[:per_sex]); groups['B'].extend(people[per_sex:2*per_sex])
    fit = {'A':[], 'B':[]}
    for sex in ('F','M'):
        ordered = {a:[s for s in groups[a] if meta['sex'][s] == sex] for a in ('A','B')}
        for rank in range(per_sex):
            for prompt in fit_prompts:
                for label in range(n):
                    cells = {a:meta['cells'][ordered[a][rank],prompt,label] for a in ('A','B')}
                    require(cells['A']['take'] == cells['B']['take'] and cells['A']['intensity'] == cells['B']['intensity'], 'paired take/intensity differs')
                    for arm in ('A','B'): fit[arm].append(cells[arm]['relative_path'])
    def paths(people, complete):
        output = []
        for s in people:
            for prompt in query_prompts:
                for label in range(n):
                    cell = meta['cells'].get((s,prompt,label))
                    require(cell is not None or not complete, 'missing required query representative')
                    if cell is not None: output.append(cell['relative_path'])
        return sorted(output)
    report = {a:paths(groups[a],True) for a in ('A','B')};report['outer'] = paths(test,False)
    return dict(pair_id=f'fpc_{phase}_{corpus}_d{draw:02d}_f{fold}',corpus=corpus,phase=phase,draw=draw,fold=fold,
                n_classes=n,group_speakers=groups,test_speakers=test,eligible_speakers=sorted(eligible),
                fit_prompts=fit_prompts,query_prompts=query_prompts,report=report,
                report_batches={r:[paths[i:i+16] for i in range(0,len(paths),16)] for r,paths in report.items()},
                seeds={kind:seed(*parts,kind) for kind in ('initialization','order','crop','torch_training')},
                arm_execution_order=['A','B'] if seed(*parts,'execution_order')%2 == 0 else ['B','A']), fit


def _units(metas):
    units = []
    for phase in ('formal','pilot'):
        for corpus in CORPUS_ORDER:
            for draw in (range(DRAWS) if phase == 'formal' else (0,)):
                for fold in (range(FOLDS) if phase == 'formal' else (0,)):
                    panel, fit = _panel(metas[corpus],corpus,phase,draw,fold)
                    use_B = corpus == 'cremad' and fold == 0
                    for arm in panel['arm_execution_order']:
                        if arm == 'B' and not use_B: continue
                        units.append(dict(**deepcopy(panel), unit_id=panel['pair_id']+'_'+arm, arm=arm,
                            seen_group=arm,unseen_group='B' if arm=='A' else 'A',fit=fit[arm],config=CONFIG.copy(),
                            selection_enabled=arm=='A',prediction_epochs=list(range(1,16)) if arm=='A' else [15],
                            analysis_role='checkpoint_main' if arm=='A' else 'role_swap_control'))
    return units


def _semantic(metas, inputs):
    return dict(program=PROGRAM,status='metadata_validated_not_execution_frozen',execution_ready=False,
                config=CONFIG.copy(),analysis=deepcopy(ANALYSIS),execution=deepcopy(EXECUTION),input=inputs,
                corpora={c:dict(**DESIGNS[c],labels=list(CORPORA[c].labels),
                               sex_source='CREMA demographics Female/Male mapping' if c=='cremad' else 'source filename F/M code; not independent self-reported gender') for c in CORPUS_ORDER},
                rows={c:metas[c]['rows'] for c in CORPUS_ORDER},units=_units(metas))


def _validate_structure(plan, metas):
    units=plan['units'];require(len(units)==388 and len({u['unit_id'] for u in units})==388,'unit count/uniqueness')
    expected={(c,d,f,'A') for c in CORPUS_ORDER for d in range(24) for f in range(5)} | {('cremad',d,0,'B') for d in range(24)}
    require({(u['corpus'],u['draw'],u['fold'],u['arm']) for u in units if u['phase']=='formal'}==expected,'formal grid differs')
    require(Counter(u['phase'] for u in units)==Counter(formal=384,pilot=4),'phase counts differ')
    for u in units:
        meta,design=metas[u['corpus']],DESIGNS[u['corpus']]
        rows=meta['rows']; n=design['n_classes']; people=design['people_per_group']
        require(u['config']==CONFIG and u['n_classes']==n,'configuration/native classes differ')
        require(u['selection_enabled'] is (u['arm']=='A') and u['prediction_epochs']==(list(range(1,16)) if u['arm']=='A' else [15]),'main/control prediction contract differs')
        require(len(u['fit'])==len(set(u['fit']))==people*design['fit_texts']*n,'fit budget/duplicates')
        sets=[set(u['group_speakers'][a]) for a in ('A','B')]+[set(u['test_speakers'])]
        require(len(sets[0])==len(sets[1])==people and all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3)),'speaker leakage')
        require(len(set(u['fit_prompts']))==design['fit_texts'] and len(set(u['query_prompts']))==design['query_texts']
                and not set(u['fit_prompts'])&set(u['query_prompts']),'text budget/leakage')
        require({rows[p]['speaker'] for p in u['fit']}==set(u['group_speakers'][u['arm']]),'wrong fit people')
        require(Counter((rows[p]['sex_stratum'],rows[p]['sentence'],rows[p]['label_index']) for p in u['fit'])
                ==Counter({(s,t,c):people//2 for s in ('F','M') for t in u['fit_prompts'] for c in range(n)}),'fit sex/text/class quota differs')
        occupied=set(u['fit'])
        for role in ('A','B','outer'):
            paths=u['report'][role]; require(paths==sorted(set(paths)) and paths and not occupied&set(paths),'report duplicate/order/leakage')
            occupied.update(paths)
            expected_people=set(u['test_speakers']) if role=='outer' else set(u['group_speakers'][role])
            require({rows[p]['speaker'] for p in paths}==expected_people,'report people differ')
            require(all(rows[p]['sentence'] in u['query_prompts'] for p in paths),'report texts differ')
            for s in expected_people:
                require({rows[p]['label_index'] for p in paths if rows[p]['speaker']==s}==set(range(n)),'report person lacks native class')
            if role!='outer':
                require(len(paths)==people*design['query_texts']*n,'validation budget differs')
                require(Counter(meta['sex'][s] for s in expected_people)==Counter(F=people//2,M=people//2),'validation sex quota differs')
            require(u['report_batches'][role]==[paths[i:i+16] for i in range(0,len(paths),16)],'role-specific batch manifest differs')
    for c in CORPUS_ORDER:
        for d in range(DRAWS):
            outer=[s for u in units if u['phase']=='formal' and u['corpus']==c and u['draw']==d and u['arm']=='A' for s in u['test_speakers']]
            require(sorted(outer)==metas[c]['speakers'],'outer does not partition corpus in draw')
    pairs=defaultdict(list)
    for u in units:pairs[u['pair_id']].append(u)
    for pair in pairs.values():
        if len(pair)==2:
            require({u['arm'] for u in pair}=={'A','B'},'paired arms differ')
            a,b=pair
            require(all(a[k]==b[k] for k in ('seeds','group_speakers','report','report_batches','fit_prompts','query_prompts','n_classes')),'pair framework differs')
            for pa,pb in zip(a['fit'],b['fit']):
                ra,rb=metas[a['corpus']]['rows'][pa],metas[b['corpus']]['rows'][pb]
                require(all(ra[k]==rb[k] for k in ('sex_stratum','sentence','label_index','take','intensity')),'paired slot metadata differ')


def generate(repo):
    metas,inputs=_load_metadata(Path(repo).resolve())
    plan=_semantic(metas,inputs);_validate_structure(plan,metas)
    plan['plan_sha256']=digest(plan)
    return plan


def validate(plan, repo):
    """Metadata structural gates plus exact semantic replay; not source freeze."""
    metas,inputs=_load_metadata(Path(repo).resolve())
    _validate_structure(plan,metas)
    expected=_semantic(metas,inputs)
    require(set(plan)==set(expected)|{'plan_sha256'},'unknown/missing plan fields')
    require(all(plan[k]==v for k,v in expected.items()),'outcome-free semantic replay differs')
    require(plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),'plan semantic hash differs')
    return True


def load_plan(path, repo):
    plan=json.loads(Path(path).read_text(encoding='utf-8'));validate(plan,repo);return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'refuse to replace existing plan')
    plan=generate(args.repo);write_json(args.out,plan)
    print(json.dumps(dict(program=PROGRAM,plan_sha256=plan['plan_sha256'],formal=384,pilot=4,
                         main=360,control=24,execution_ready=False)))


if __name__=='__main__':main()

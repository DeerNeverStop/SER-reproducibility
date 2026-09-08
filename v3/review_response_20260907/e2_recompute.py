"""Post-hoc E2 geometry and review checks; no selection changes or model execution."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, csv, hashlib, json, math
import numpy as np
from scipy.stats import t


def norm(x):
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all() and np.linalg.norm(x) > 1e-12
    return x / np.linalg.norm(x)


def csv_write(path, rows):
    with path.open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    a = ap.parse_args()
    names = ['e2_geometry_rows.csv', 'e2_geometry_contexts.csv', 'e2_review_numbers.json']
    assert all(not (a.outdir/n).exists() for n in names), 'Use new output names/directory'
    pins = {}

    def read(path):
        assert path.suffix != '.pt'
        b = path.read_bytes(); pins[str(path.resolve())] = hashlib.sha256(b).hexdigest(); return b

    def js(path): return json.loads(read(path))
    def rows(path): return list(csv.DictReader(read(path).decode('utf-8-sig').splitlines()))
    plan_files = list((a.archive/'inputs').glob('plan_*.json')); assert len(plan_files) == 1
    plan = js(plan_files[0]); content = {k:v for k,v in plan.items() if k != 'plan_sha256'}
    semantic = hashlib.sha256(json.dumps(content, ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert semantic == plan['plan_sha256'] and len(plan['units']) == 720
    for path, sha in plan['source_sha256'].items():
        assert hashlib.sha256(read(a.repo/path)).hexdigest() == sha
    raw = rows(a.repo/plan['input']['manifest_path'])
    assert pins[str((a.repo/plan['input']['manifest_path']).resolve())] == plan['input']['manifest_sha256']
    byhash = defaultdict(list)
    for r in raw: byhash[r['sha256']].append(r)
    removed = {'1076_MTI_SAD_XX.wav'}
    for group in byhash.values():
        remove = group if len({r['label'] for r in group}) > 1 else sorted(group,key=lambda r:r['relative_path'])[1:]
        removed.update(r['relative_path'] for r in remove)
    clean = [r for r in raw if r['relative_path'] not in removed]; assert len(clean) == 7435
    bypath = {r['relative_path']:r for r in clean}
    grouped = defaultdict(list)
    for r in clean: grouped[r['speaker'],r['sentence'],int(r['label_index'])].append(r)
    cells = {}
    for key, group in grouped.items():
        opts = [r for r in group if r['intensity'] in ('MD','XX')]
        if opts: cells[key] = min(opts,key=lambda r:(r['intensity']!='MD',r['relative_path']))
    speakers = sorted({r['speaker'] for r in clean}); assert len(speakers) == 91
    neutral = {int(r['label_index']) for r in clean if r['label']=='neutral'}; assert len(neutral)==1
    neutral = neutral.pop()
    asvpath = a.archive/'inputs/features'/plan['input']['asv_path']
    assert hashlib.sha256(read(asvpath)).hexdigest() == plan['input']['asv_sha256']
    with np.load(asvpath, allow_pickle=False) as z:
        asv = {str(p): np.array(x,dtype=np.float64) for p,x in zip(z['paths'],z['embeddings'])}
    assert len(asv)==7435 and all(v.shape==(192,) for v in asv.values())

    def centroid(paths): return norm(np.mean([norm(asv[p]) for p in paths],axis=0))
    e0path=a.repo/'v3/speaker_coverage/reports/data/diagnostics/e0_policy_geometry.csv'
    old = {(int(r['fold']),int(r['rotation']),int(r['draw']),r['policy']):r for r in rows(e0path)
           if r['encoder']=='ecapa' and r['pool_scope']=='including_selected'}
    index = {(u['fold'],u['rotation'],u['draw'],u['policy']):u for u in plan['units'] if u['model']=='ridge_wavlm'}
    assert len(index)==270
    perperson, contexts, pool, omissions, replay_errors = [], [], [], [], []
    for (fold,rotation,draw,policy),unit in sorted(index.items()):
        fixed=index[fold,rotation,draw,'U']
        assert unit['test']==fixed['test'] and unit['val']==fixed['val']
        prompts=sorted({bypath[p]['sentence'] for p in unit['fit']}); assert len(prompts)==8
        selected=sorted({bypath[p]['speaker'] for p in unit['fit']}); assert len(selected)==24
        test=sorted({bypath[p]['speaker'] for p in unit['test']})
        excluded={bypath[p]['speaker'] for role in ('test','val') for p in unit[role]}
        candidates=sorted(s for s in speakers if s not in excluded and all((s,p,l) in cells for p in prompts for l in range(6)))
        assert set(selected)<=set(candidates) and not set(selected)&set(test)
        centers={s:centroid([cells[s,p,neutral]['relative_path'] for p in prompts]) for s in candidates}
        full=np.stack([norm(centers[s]) for s in candidates])
        distance=np.clip(1-full@full.T,0,2); np.fill_diagonal(distance,0)
        chosen=[candidates.index(s) for s in selected]
        closest=np.sort(distance[:,chosen],axis=1)
        metrics=dict(nn1_mean=float(closest[:,0].mean()),nn1_max=float(closest[:,0].max()),nn3_mean=float(closest[:,:3].mean()))
        for m,v in metrics.items(): replay_errors.append(abs(v-float(old[fold,rotation,draw,policy][m])))
        pool.append(dict(fold=fold,rotation=rotation,draw=draw,policy=policy,**metrics))
        train=np.stack([norm(centers[s]) for s in selected])
        for method in ('matched_eight_train_prompts_complete','actual_test_query_neutral_available'):
            current=[]
            for speaker in test:
                if method=='matched_eight_train_prompts_complete':
                    missing=[p for p in prompts if (speaker,p,neutral) not in cells]
                    if missing:
                        omissions.append(dict(method=method,fold=fold,rotation=rotation,draw=draw,policy=policy,
                                              speaker=speaker,missing_prompts=missing))
                        continue
                    paths=[cells[speaker,p,neutral]['relative_path'] for p in prompts]
                else:
                    paths=sorted(p for p in unit['test'] if bypath[p]['speaker']==speaker and int(bypath[p]['label_index'])==neutral)
                    assert len(paths) in (1,2)
                vector=centroid(paths)
                ds=np.clip(1-train@vector,0,2)
                order=sorted(range(len(selected)),key=lambda j:(float(ds[j]),selected[j]))
                record=dict(method=method,fold=fold,rotation=rotation,draw=draw,policy=policy,
                    test_speaker=speaker,test_reference_count=len(paths),train_reference_count_per_speaker=8,
                    nn1=float(ds[order[0]]),nn3_mean=float(ds[order[:3]].mean()),third_nearest=float(ds[order[2]]),
                    nearest_speaker=selected[order[0]],nearest3_speakers='|'.join(selected[j] for j in order[:3]),
                    test_reference_paths='|'.join(paths))
                perperson.append(record);current.append(record)
            assert current
            contexts.append(dict(method=method,fold=fold,rotation=rotation,draw=draw,policy=policy,
                n_test_people_expected=len(test),n_test_people_used=len(current),
                nn1_mean=float(np.mean([x['nn1'] for x in current])),
                nn1_max=float(max(x['nn1'] for x in current)),
                nn3_mean=float(np.mean([x['nn3_mean'] for x in current]))))
    assert max(replay_errors)<1e-10

    def summarize_contexts(items):
        table={(r['fold'],r['rotation'],r['draw'],r['policy']):r for r in items}
        assert len(table)==270
        means={p:{m:float(np.mean([r[m] for r in items if r['policy']==p])) for m in ('nn1_mean','nn1_max','nn3_mean')} for p in ('U','R','C')}
        wins={}
        for metric in ('nn1_mean','nn1_max','nn3_mean'):
            wins[metric]={}
            for left,right in [('C','U'),('R','U'),('C','R')]:
                ds=[table[f,r,d,left][metric]-table[f,r,d,right][metric] for f in range(5) for r in range(6) for d in range(3)]
                wins[metric][left+'-'+right]=dict(mean_difference=float(np.mean(ds)),lower_count=sum(x < -1e-12 for x in ds),tie_count=sum(abs(x)<=1e-12 for x in ds),n_views=90)
        return dict(equal_context_means=means,paired_differences_and_counts=wins,
                    scope='90 fold/rotation/draw views; R/C share membership and geometry over draws, so only 30 distinct R/C contexts; folds/rotations reuse people and are not independent experiments')
    geo={}
    for method in sorted({r['method'] for r in contexts}):
        subset=[r for r in perperson if r['method']==method]
        g=summarize_contexts([r for r in contexts if r['method']==method])
        personmeans={}
        for policy in ('U','R','C'):
            grp=defaultdict(list)
            for r in subset:
                if r['policy']==policy:grp[r['test_speaker']].append(r)
            assert len(grp)==91
            vals={s:{m:float(np.mean([r[m] for r in rr])) for m in ('nn1','nn3_mean')} for s,rr in grp.items()}
            personmeans[policy]=dict(n_people=len(vals),mean_nn1=float(np.mean([x['nn1'] for x in vals.values()])),
                mean_nn3=float(np.mean([x['nn3_mean'] for x in vals.values()])),
                reference_count_histogram=dict(Counter(r['test_reference_count'] for r in subset if r['policy']==policy)),
                min_views_per_person=min(map(len,grp.values())),max_views_per_person=max(map(len,grp.values())))
        g['equal_person_after_rotation_draw_average']=personmeans;geo[method]=g

    resultdir=a.repo/'v3/speaker_coverage/reports/data/e2'
    results=js(resultdir/'results.json')
    tables={n:rows(resultdir/n) for n in ('per_speaker.csv','per_repeat_speaker.csv','contrasts.csv','fold_contrasts.csv')}
    for n in tables:assert pins[str((resultdir/n).resolve())]==results['table_sha256'][n]
    person=tables['per_speaker.csv']; assert len(person)==3*3*2*91
    lookup={(r['model'],r['policy'],r['checkpoint'],r['speaker']):float(r['uar_percent']) for r in person}
    rng=np.random.default_rng(20260906); boot=rng.integers(0,91,size=(10000,91))
    lastbest={};vectors=[]
    for policy in ('U','R','C'):
        v=np.array([lookup['cnn',policy,'last',s]-lookup['cnn',policy,'best',s] for s in speakers]);vectors.append(v)
        low,high=np.quantile(v[boot].mean(1),[.025,.975],method='linear')
        lastbest[policy]=dict(mean_last_minus_best_pp=float(v.mean()),ci95_low=float(low),ci95_high=float(high),
                              n_people_last_better=int((v>0).sum()),n_people_tied=int((v==0).sum()),n_people=91)
    lastbest['equal_three_policy_mean_pp']=float(np.mean(vectors))
    # Demonstrate the review's normal approximation; do not present it as valid p-values.
    approx=[]
    for r in tables['contrasts.csv']:
        delta=float(r['difference_pp']); low=float(r['ci95_low_pp']); high=float(r['ci95_high_pp'])
        se=(high-low)/(2*1.959963984540054)
        p=math.erfc(abs(delta)/(se*math.sqrt(2)))
        approx.append(dict(model=r['model'],checkpoint=r['checkpoint'],contrast=r['contrast'],
                           point=delta,ci_low=low,ci_high=high,normal_se_proxy=se,normal_p_proxy=p))
    running=0.
    for rank,i in enumerate(sorted(range(len(approx)),key=lambda i:approx[i]['normal_p_proxy'])):
        running=max(running,min(1.,(len(approx)-rank)*approx[i]['normal_p_proxy']))
        approx[i]['holm_12_proxy']=running
    primary=next(r for r in tables['contrasts.csv'] if r['role']=='primary')
    foldvals=[float(r['difference_pp_descriptive']) for r in tables['fold_contrasts.csv'] if (r['model'],r['checkpoint'],r['contrast'])==('cnn','best','C-U')]
    drawvals=json.loads(primary['repeat_differences_pp'])
    def nominal_t(values):
        x=np.array(values);half=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
        return dict(n=len(x),mean=float(x.mean()),ci95=[float(x.mean()-half),float(x.mean()+half)],
                    valid_independence_not_established=True)
    report=dict(schema='ser-e2-posthoc-review-1',plan_sha256=plan['plan_sha256'],source_sha256=pins,
        old_selection_or_results_modified=False,models_fitted=False,test_SER_predictions_used_for_geometry=False,
        test_neutral_annotations_used=True,ecapa_scope='frozen 192D normalized neutral references; known neutral labels used, no SER outcomes or heldout distances used to reselect training people',
        pool_old_csv_max_absolute_replay_error=max(replay_errors),pool=summarize_contexts(pool),
        outer_geometry=geo,reference_omissions=omissions,
        geometry_scope='Post-hoc descriptive geometry, no inferential CI or new selection gate. Primary reference-matched geometry requires all eight neutral references; secondary actual-query geometry uses the available one/two neutral query recordings and differs in reference text/count.',
        cnn_last_minus_best=lastbest,
        cnn_last_best_interval_scope='Exploratory paired-person percentile bootstrap after averaging the three fixed draws, conditional on already fitted models; not a causal early-stopping mechanism or independent confirmation',
        normal_holm_approximation_only=approx,
        multiplicity_warning='Percentile bootstrap endpoints do not identify a normal standard error or an exact null p-value. Holm requires valid marginal p-values; these proxies cannot establish an exact corrected claim. The 12 rows include duplicate Ridge best/last endpoints and mix one frozen primary with sensitivity contrasts.',
        alternative_primary_intervals_descriptive=dict(fold_equal_weight=nominal_t(foldvals),draw_equal_weight=nominal_t(drawvals),
            caveat='Folds share training people; draws reuse all test people, and R/C membership is fixed across draws. Wider t intervals are not automatically more honest. Equal-fold mean is not identical to the frozen equal-person estimand.'),
        numpy_version=np.__version__)
    read(Path(__file__))
    a.outdir.mkdir(parents=True,exist_ok=True)
    csv_write(a.outdir/names[0],perperson);csv_write(a.outdir/names[1],contexts)
    report['output_csv_sha256']={n:hashlib.sha256((a.outdir/n).read_bytes()).hexdigest() for n in names[:2]}
    with (a.outdir/names[2]).open('x',encoding='utf8',newline='\n') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'pool':report['pool'],'outer':geo,'omissions':len(omissions),'cnn':lastbest},ensure_ascii=True))


if __name__=='__main__':main()

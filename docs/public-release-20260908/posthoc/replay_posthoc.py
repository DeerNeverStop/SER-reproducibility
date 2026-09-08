"""Post hoc audit of PR22 diagnostics; immutable archived CSV inputs, no GPU.

This is not a new primary analysis, a patience experiment, or a changed family.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import math
import numpy as np
from scipy import stats

FAMILY = [
    'hubert_cremad_D_CE', 'hubert_cremad_J',
    'hubert_subesco_D_CE', 'hubert_subesco_J',
    'hubert_ravdess_D_CE', 'hubert_ravdess_J',
    'window_subesco_L_CE', 'window_subesco_L_J',
    'window_ravdess_L_CE', 'window_ravdess_L_J',
]
MANIFEST_SHA = 'e1042806268c32acc01316ca52dc15ef1edefbcda11cc1ac60a56f7679caa521'
REVIEW_SHA = 'e614743e60bcbead708c46a07129912cc2e1e440e8bd7bc2550c9757d0b87e13'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def holm_ten(p):
    assert len(p) == 10
    clean = np.array([1.0 if v is None else v for v in p])
    order = np.argsort(clean, kind='stable')
    adjusted = np.minimum(1.0, np.maximum.accumulate(clean[order] * np.arange(10, 0, -1)))
    out = np.empty(10); out[order] = adjusted
    return [None if p[i] is None else float(out[i]) for i in range(10)]


def estimate(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and np.all(np.isfinite(x))
    mean = math.fsum(map(float, x)) / len(x)
    sd = math.sqrt(math.fsum((float(v)-mean)**2 for v in x)/(len(x)-1))
    p = None if sd == 0 else float(2 * stats.t.sf(abs(mean/(sd/math.sqrt(len(x)))), len(x)-1))
    return dict(n=len(x), df=len(x)-1, mean_pp=mean, sd_pp=sd, p_two_sided=p)


def read_inputs(repo, review):
    folder = repo / 'docs/autodl-supplement-20260908/results'
    manifest_raw = (folder / 'OUTPUT_MANIFEST.json').read_bytes()
    assert sha(manifest_raw) == MANIFEST_SHA
    manifest = json.loads(manifest_raw)
    pins = {str(folder / 'OUTPUT_MANIFEST.json'): sha(manifest_raw)}
    tables = {}
    for name in ('primary_tests', 'draws', 'selected_epochs', 'units'):
        path = folder / (name + '.csv'); raw = path.read_bytes()
        assert dict(bytes=len(raw), sha256=sha(raw)) == manifest['files'][path.name]
        pins[str(path)] = sha(raw)
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    # Public projection: review text is private and is not authenticated here.
    # Every scientific CSV input is still checked against its original manifest.
    assert [r['id'] for r in tables['primary_tests']] == FAMILY
    assert [len(tables[k]) for k in ('primary_tests','draws','selected_epochs','units')] == [10,312,4800,960]
    return tables, pins, manifest['plan_sha256']


def main_vectors(tables):
    vectors = []
    for test in tables['primary_tests']:
        long = test['endpoint'].startswith('L_')
        kind, window = ('window45_minus15', '45_minus15') if long else ('model_window','15')
        field = {'L_CE':'D_CE', 'L_J':'J'}.get(test['endpoint'], test['endpoint']) + '_pp'
        rows = [r for r in tables['draws'] if (r['corpus'],r['model'],r['comparison'],r['window']) ==
                (test['corpus'],test['model'],kind,window)]
        assert len(rows) == 24 and {int(r['draw']) for r in rows} == set(range(24))
        rows.sort(key=lambda r:int(r['draw']))
        vectors.append([float(r[field]) for r in rows])
    matrix = np.asarray(vectors)
    baseline = [estimate(x) for x in matrix]
    adjusted = holm_ten([r['p_two_sided'] for r in baseline])
    for actual, derived, p in zip(tables['primary_tests'],baseline,adjusted):
        assert abs(float(actual['mean_pp']) - derived['mean_pp']) < 1e-12
        assert abs(float(actual['p_two_sided']) - derived['p_two_sided']) < 1e-12
        assert abs(float(actual['holm_p_ten']) - p) < 1e-12
    return matrix, baseline, adjusted


def leave_one_draw_out(matrix, baseline_p):
    all_runs = []
    for excluded in range(24):
        est = [estimate(np.delete(x, excluded)) for x in matrix]
        adjusted = holm_ten([r['p_two_sided'] for r in est])
        all_runs.append(dict(excluded_draw=excluded, tests=[dict(id=key, **row, holm_p_ten=p,
                             reject=(p is not None and p<=.05)) for key,row,p in zip(FAMILY,est,adjusted)]))
    summary = []
    for i, key in enumerate(FAMILY):
        rows = [run['tests'][i] for run in all_runs]
        original = baseline_p[i] <= .05
        summary.append(dict(id=key, full_sample_reject=original,
            mean_pp_range=[min(r['mean_pp'] for r in rows),max(r['mean_pp'] for r in rows)],
            raw_p_range=[min(r['p_two_sided'] for r in rows),max(r['p_two_sided'] for r in rows)],
            holm_p_range=[min(r['holm_p_ten'] for r in rows),max(r['holm_p_ten'] for r in rows)],
            n_reject=sum(r['reject'] for r in rows),
            changed_decision_excluded_draws=[d for d,r in enumerate(rows) if r['reject'] != original],
            nonpositive_mean_excluded_draws=[d for d,r in enumerate(rows) if r['mean_pp'] <= 0]))
    return dict(scope='post hoc deletion sensitivity; ten fixed two-sided tests recomputed jointly each time; n23 df22; no changed primary conclusions',
                draw_indices='zero-based 0..23', summary=summary, runs=all_runs)


def epoch_descriptions(tables):
    grouped = {}
    for row in tables['selected_epochs']:
        key=(row['corpus'],row['model'],int(row['window']),row['rule'])
        grouped.setdefault(key,[]).append(row)
    assert len(grouped)==40
    output=[]
    for (c,m,w,r),rows in sorted(grouped.items()):
        assert len(rows)==120 and {(int(x['draw']),int(x['fold'])) for x in rows}=={(d,f) for d in range(24) for f in range(5)}
        epochs=[int(x['selected_epoch']) for x in rows]
        assert min(epochs)>=1 and max(epochs)<=w
        output.append(dict(corpus=c,model=m,window=w,rule=r,n_contexts=120,
            n_at_upper_boundary=sum(e==w for e in epochs),upper_boundary_percent=sum(e==w for e in epochs)/1.2,
            n_above15=sum(e>15 for e in epochs),above15_percent=sum(e>15 for e in epochs)/1.2,
            mean_epoch=math.fsum(epochs)/120,min_epoch=min(epochs),max_epoch=max(epochs),
            last_is_fixed_not_selection=(r=='last')))
    return output


def window_zeros(tables):
    out=[]
    for c in ('subesco','ravdess'):
        draws=[r for r in tables['draws'] if (r['corpus'],r['model'],r['comparison'])==(c,'wavlm_base_plus','window45_minus15')]
        assert len(draws)==24
        indices=[int(r['draw']) for r in draws if float(r['D_CE_pp'])==0.0]
        by_window={w:{(int(r['draw']),int(r['fold'])):r for r in tables['units'] if
                      (r['corpus'],r['model'],int(r['window']))==(c,'wavlm_base_plus',w)} for w in (15,45)}
        assert len(by_window[15])==len(by_window[45])==120
        same_epoch={}; same_outer={}
        for rule in ('seen_ce','unseen_ce','seen_uar','unseen_uar'):
            same_epoch[rule]=sum(by_window[15][key]['epoch_'+rule]==by_window[45][key]['epoch_'+rule] for key in by_window[15])
            same_outer[rule]=sum(by_window[15][key]['T_'+rule+'_pp']==by_window[45][key]['T_'+rule+'_pp'] for key in by_window[15])
        out.append(dict(corpus=c,n_draws=24,L_CE_zero_draws=indices,n_L_CE_zero=len(indices),
                        unseen_ce_zero_draws=[int(r['draw']) for r in draws if float(r['T_unseen_ce_pp'])==0.0],
                        unchanged_epoch_contexts=same_epoch,unchanged_outer_uar_contexts=same_outer))
    return out


def correlation_descriptions(matrix, baseline_p):
    chosen=[i for i,p in enumerate(baseline_p) if p<=.05]
    def details(x, ids):
        corr=np.corrcoef(x)
        eig=np.linalg.eigvalsh(corr)
        return dict(ids=ids,correlation=corr.tolist(),eigenvalues=eig.tolist(),
                    participation_ratio=float(np.trace(corr)**2 / np.sum(corr*corr)))
    selected=details(matrix[chosen],[FAMILY[i] for i in chosen])
    full=details(matrix,FAMILY)
    ideal=np.eye(len(chosen))
    for corpus in ('cremad','subesco'):
        i=selected['ids'].index('hubert_'+corpus+'_D_CE'); j=selected['ids'].index('hubert_'+corpus+'_J')
        ideal[i,j]=ideal[j,i]=selected['correlation'][i][j]
    return dict(scope='post hoc covariance dimension among selected endpoints, not independent discoveries; 24 aligned draws conditional on shared data',
                six_rejections=selected,ten_endpoints=full,
                ideal_two_blocks_participation_ratio=float(np.trace(ideal)**2/np.sum(ideal*ideal)),
                hubert_D_CE_J_correlations={c:float(np.corrcoef(matrix[FAMILY.index('hubert_'+c+'_D_CE')],matrix[FAMILY.index('hubert_'+c+'_J')])[0,1]) for c in ('cremad','subesco','ravdess')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    # Review text is not needed for any numerical calculation.
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); assert not args.out.exists()
    tables,pins,plan_sha=read_inputs(args.repo,None)
    matrix,baseline,p=main_vectors(tables)
    output=dict(schema='ser-public-posthoc-diagnostic-audit-1',post_hoc=True,
        scope='Public CSV-only post hoc diagnostic replay; private review text not authenticated; no new patience simulations, no GPU, no altered test family, no modification of original results',
        plan_sha256=plan_sha,review_commit='d047ee7925c8e57ea7809882e19e40a0e7820951',
        input_sha256=pins,source_sha256=sha(Path(__file__).read_bytes()),
        original_tests=[dict(id=k,**v,holm_p_ten=q) for k,v,q in zip(FAMILY,baseline,p)],
        leave_one_draw_out=leave_one_draw_out(matrix,p),
        selected_epoch_descriptions=epoch_descriptions(tables),
        window_zeros=window_zeros(tables),correlation=correlation_descriptions(matrix,p))
    for path,expected in pins.items(): assert sha(Path(path).read_bytes())==expected
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as handle:
        json.dump(output,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write('\n')
    print(json.dumps(dict(passed=True,original_tests=10,leave_one_runs=24,epoch_groups=40,
                          output=str(args.out),sha256=sha(args.out.read_bytes())),ensure_ascii=False))


if __name__=='__main__':
    main()

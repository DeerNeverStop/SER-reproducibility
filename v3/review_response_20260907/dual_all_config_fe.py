"""One prespecified review reconstruction: average all four last-epoch configs.

Motivation: Claude's unadjusted 2.801 pp matches the mean of the four fixed-last
configuration gaps, not configuration 3 alone. No alternative fits or CIs are
searched to match an undocumented reviewer result.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from v3.review_response_20260907.dual_audit import PINS, csv_rows, digest, read, sha
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    repo,archive,out=args.repo.resolve(),args.archive.resolve(),args.out.resolve()
    assert not out.exists() and not out.is_relative_to(archive)
    pins={}
    for rel,expected in PINS.items():
        assert sha(archive/rel)==expected
        pins[str(archive/rel)]=expected
    plan=read(archive/'inputs/plan.json');gate=read(archive/'analysis/formal_gate.json')
    assert plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    assert gate['plan_sha256']==plan['plan_sha256'] and gate['pass'] and gate['units']==480
    for rel,expected in plan['sources'].items():
        assert sha(repo/rel)==expected
        pins[str(repo/rel)]=expected
    manifest=repo/plan['input']['manifest_path']
    assert sha(manifest)==plan['input']['manifest_sha256']
    pins[str(manifest)]=sha(manifest)
    meta={r['relative_path']:r for r in csv_rows(manifest)}
    observations={}
    for unit in plan['units']:
        uid=unit['unit_id'];folder=archive/'runs/formal/units'/uid
        path=folder/'DONE'
        assert sha(path)==gate['done_sha256'][uid]
        pins[str(path)]=sha(path);done=read(path)
        assert done['unit_sha256']==digest(unit)
        assert done['plan_sha256']==plan['plan_sha256'] and done['phase']=='formal'
        rel=[r for r in done['artifacts'] if Path(r).name=='predictions.npz']
        assert len(rel)==1
        path=(folder/rel[0]).resolve()
        assert path.is_relative_to(folder.resolve()) and sha(path)==done['artifacts'][rel[0]]
        pins[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as arrays:
            for role,people,seen in [('val_seen',unit['fit_speakers'],1),('val_unseen',unit['unseen_speakers'],0)]:
                paths=arrays[role+'__paths'].tolist();y=arrays[role+'__labels'];z=arrays[role+'__last__logits']
                assert paths==unit[role] and z.shape==(288,6) and np.isfinite(z).all()
                assert y.tolist()==[int(meta[p]['label_index']) for p in paths]
                ids=np.asarray([meta[p]['speaker'] for p in paths]);pred=z.argmax(1)
                for person in people:
                    mask=ids==person
                    assert mask.sum()==12 and np.array_equal(np.bincount(y[mask],minlength=6),np.full(6,2))
                    key=unit['draw'],unit['fold'],person,seen
                    row=observations.setdefault(key,{})
                    assert unit['config_index'] not in row
                    row[unit['config_index']]=100.0*int((pred[mask]==y[mask]).sum())/12
    assert len(observations)==5760 and all(set(row)==set(range(4)) for row in observations.values())
    keys=sorted(observations)
    people=sorted({k[2] for k in keys});contexts=sorted({k[:2] for k in keys})
    assert len(people)==91 and len(contexts)==120
    y=np.asarray([np.mean(list(observations[k].values())) for k in keys])
    exposure=np.asarray([k[3] for k in keys])
    pi=np.asarray([people.index(k[2]) for k in keys]);ci=np.asarray([contexts.index(k[:2]) for k in keys])
    X=np.column_stack((np.ones(len(keys)),exposure,np.eye(91)[pi,1:],np.eye(120)[ci,1:]))
    beta,_,rank,_=np.linalg.lstsq(X,y,rcond=None)
    assert rank==211
    raw=float(y[exposure==1].mean()-y[exposure==0].mean())
    result=dict(status='complete',identity='post-review exploratory reconstruction of all-four-config last-epoch FE point only',
        rationale='Reviewer unadjusted 2.801 equals average of four fixed-last configuration gaps; this is the sole additional model reconstruction.',
        generated_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(__file__).with_name('dual_audit.py')),plan_sha256=plan['plan_sha256'],input_sha256=pins,
        data='Each of 5760 speaker-by-draw/fold observations is the equal mean of four fixed-last configuration UARs; no fourfold increase of people or inferential n.',
        rows=5760,people=91,contexts=120,configurations_averaged=4,
        unadjusted_pooled_gap_pp=raw,adjusted_coefficient_pp=float(beta[1]),
        regression='Equal-row OLS: UAR ~ seen + speaker fixed effects + joint draw/fold fixed effects; intercept plus 90 speaker and 119 context indicators, 211 columns.',
        reviewer_point_pp=2.967,reviewer_unadjusted_pp=2.801,
        reviewer_point_matches_rounding=round(float(beta[1]),3)==2.967,
        reviewer_unadjusted_matches_rounding=round(raw,3)==2.801,
        interval='No interval computed here. Reviewer CI construction/resampling/weights/seed unspecified; point agreement does not reproduce CI.',
        new_tests=0,new_intervals=0,no_gpu=True,no_checkpoint_weights_read=True)
    for path,expected in pins.items():
        assert sha(path)==expected
    out.mkdir(parents=True)
    (out/'dual_all_config_fe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},ensure_ascii=False))


if __name__=='__main__':
    if not __debug__:raise RuntimeError('Do not disable assertions')
    main()

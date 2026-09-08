"""Publication figures from the completed, independently checked final program.

No partial-batch plotting, model inference, new tests, smoothing or data-driven
epoch selection. Native tasks remain separate. Curves and role controls are
descriptive; hypothesis intervals are the already frozen pointwise t intervals.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CORPORA = ('cremad', 'subesco', 'ravdess')
NAMES = dict(cremad='CREMA-D (6 classes)', subesco='SUBESCO (7 classes)', ravdess='RAVDESS (8 classes)')
COUNTS = dict(contexts=360, draws=72, selected_epochs=1800, curves=5400, controls=24)
INK, SEEN, UNSEEN = '#222222', '#0072B2', '#D55E00'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load(results, audit_path):
    result = read_json(results/'results.json')
    require(result['result_sha256'] == digest({k:v for k,v in result.items() if k!='result_sha256'}),
            'result semantic hash differs')
    require(result['counts'] == dict(formal_fits=384, main_A=360, control_B=24, pilot_scored=0,
                                    main_contexts=360, corpus_draws=72, family_tests=6), 'incomplete program')
    gate_path=results/'complete_gate.json'
    gate=read_json(gate_path)
    require(sha(gate_path)==result['complete_gate_snapshot']['sha256']==result['inputs']['complete_gate_sha256']
            and gate['pass'] is True and gate['phase']=='formal' and gate['verified_units']==384
            and gate['expected_units']==384, 'complete formal gate required')
    audit=read_json(audit_path)
    require(audit['audit_sha256']==digest({k:v for k,v in audit.items() if k!='audit_sha256'}), 'audit hash differs')
    require(audit['pass'] is True and audit['formal_units']==384 and audit['mismatches']==0
            and audit['lock_sha256']==gate['lock_sha256'] and audit['plan_sha256']==result['plan_sha256']
            and audit['score_gate_sha256']==sha(gate_path), 'independent numerical pass required')
    source_pin=audit['source_and_input_sha256'].get(str((results/'results.json').resolve()))
    require(source_pin==sha(results/'results.json'), 'audit did not check these exact scores')
    pins={str(results/'results.json'):sha(results/'results.json'), str(gate_path):sha(gate_path),
          str(audit_path):sha(audit_path), str(Path(__file__).resolve()):sha(__file__)}
    tables={}
    for name,count in COUNTS.items():
        path=results/(name+'.csv')
        require(result['tables'][path.name]==dict(rows=count,sha256=sha(path)), 'table commitment differs')
        with path.open(encoding='utf8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        require(len(rows)==count, 'table extent differs: '+name)
        pins[str(path)]=sha(path)
        tables[name]=rows
    require({(t['corpus'],t['estimand']) for t in result['tests']}==
            {(c,e) for c in CORPORA for e in ('delta_CE','J')} and len(result['tests'])==6,
            'test family differs')
    return result,tables,pins


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':11,
                         'axes.labelsize':10, 'xtick.labelsize':9, 'ytick.labelsize':10,
                         'legend.fontsize':9, 'pdf.fonttype':42, 'ps.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.edgecolor':'#777777', 'savefig.facecolor':'white'})


def save(fig, out, name, files):
    for suffix in ('png','pdf','svg'):
        path=out/(name+'.'+suffix)
        fig.savefig(path,dpi=200,bbox_inches='tight',pad_inches=.10)
        files[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    plt.close(fig)


def hypothesis_plot(result,out,files):
    fig,axes=plt.subplots(1,2,figsize=(10,3.2),constrained_layout=True)
    lookup={(v['corpus'],v['estimand']):v for v in result['tests']}
    for axis, estimand, label in zip(axes,('delta_CE','J'),
            ('CE checkpoint rule: seen − unseen','Criterion interaction: CE effect − UAR effect')):
        axis.axvspan(-1,1,color='#F1F1F1',zorder=0)
        axis.axvline(0,color='#777777',lw=.8,zorder=1)
        for index,corpus in enumerate(CORPORA):
            value=lookup[corpus,estimand]
            point=value['mean_pp']; interval=value['pointwise_95_ci_pp']
            require(math.isfinite(point),'nonfinite estimate')
            if interval is not None:
                axis.plot(interval,[index,index],color=INK,lw=1.7,zorder=2)
                axis.plot(interval,[index,index],'|',color=INK,ms=7,zorder=2)
            axis.plot(point,index,'o',color=SEEN,ms=6,zorder=3)
            p=value['holm_p_six']
            ptext='undefined' if p is None else ('<0.001' if p<.001 else f'{p:.3f}')
            axis.annotate('Holm p '+ptext,(point,index),xytext=(0,10),textcoords='offset points',
                          ha='center',va='bottom',fontsize=9)
        axis.set_yticks(range(3),[NAMES[c] for c in CORPORA])
        axis.set_ylim(2.5,-.6)
        axis.set_xlabel('UAR difference (percentage points)')
        axis.set_title(label,pad=13)
        axis.grid(axis='x',alpha=.18)
    fig.suptitle('Pointwise 95% t intervals; six prespecified tests',fontsize=12)
    save(fig,out,'checkpoint_rule_effects',files)


def curves_plot(rows,out,files):
    expected={(c,d,f,e) for c in CORPORA for d in range(24) for f in range(5) for e in range(1,16)}
    lookup={(r['corpus'],int(r['draw']),int(r['fold']),int(r['epoch'])):r for r in rows}
    require(set(lookup)==expected and len(lookup)==len(rows),'complete 120×15 curves per corpus required')
    # Render at a width that can be placed in the paper without shrinking
    # 9-pt labels below the template minimum. Verify the final embedded PDF.
    fig,axes=plt.subplots(1,3,figsize=(6.5,2.75),sharey=True,constrained_layout=True)
    x=np.arange(1,16)
    for axis,corpus in zip(axes,CORPORA):
        for draw in range(24):
            for fold in range(5):
                line=[float(lookup[corpus,draw,fold,e]['outer_uar']) for e in x]
                axis.plot(x,line,color='#999999',alpha=.12,lw=.45,zorder=0)
        # Draw the outer mean below the almost coincident unseen mean. Hollow
        # markers identify every unseen epoch in colour and grayscale, while
        # the gaps in its dotted line leave the outer solid line visible.
        policies = [
            ('outer_uar', INK, '-', 'Outer test', dict(lw=1.5, zorder=2)),
            ('seen_uar', SEEN, '--', 'Seen validation', dict(lw=1.8, zorder=3)),
            ('unseen_uar', UNSEEN, (0, (1.2, 2.4)), 'Unseen validation',
             dict(lw=1.4, zorder=4, marker='o', markersize=3.2,
                  markerfacecolor='white', markeredgewidth=0.9)),
        ]
        for column,color,dash,label,appearance in policies:
            line=[math.fsum(float(lookup[corpus,d,f,e][column]) for d in range(24) for f in range(5))/120
                  for e in x]
            axis.plot(x,line,color=color,ls=dash,label=label,**appearance)
        axis.set(title=NAMES[corpus],xlabel='Epoch',xticks=[1,5,10,15],ylim=(0,100))
        axis.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Native-task UAR (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    legend=dict(zip(labels,handles))
    labels=['Seen validation','Unseen validation','Outer test']
    fig.legend([legend[label] for label in labels],labels,
               loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('All 15 epochs\nFaint lines: individual outer-test trajectories',fontsize=11)
    save(fig,out,'complete_epoch_curves',files)


def controls_plot(rows,out,files):
    ordered=sorted(rows,key=lambda r:int(r['draw']))
    require([int(r['draw']) for r in ordered]==list(range(24)),'24 complete control pairs required')
    values=np.array([float(r['E_pp']) for r in ordered])
    require(np.isfinite(values).all(),'nonfinite control')
    fig,axis=plt.subplots(figsize=(8.6,2.7),constrained_layout=True)
    axis.axhline(0,color='#888888',lw=.8)
    axis.axhline(math.fsum(values)/24,color=UNSEEN,lw=1.5,label='Mean of 24 pairs')
    axis.plot(np.arange(1,25),values,'o',color=SEEN,ms=4.5,label='Prespecified fold 0 in each draw')
    axis.set(xlabel='Draw (1–24)',ylabel='Role-swap E (UAR pp)',xticks=[1,4,8,12,16,20,24],
             title='CREMA-D paired last-epoch control — descriptive only')
    axis.grid(axis='y',alpha=.15)
    axis.legend(frameon=False,loc='best',ncol=2)
    save(fig,out,'paired_role_controls',files)


def execute(args):
    results,audit,out=args.results.resolve(),args.audit.resolve(),args.out.resolve()
    require(not out.exists() and not out.is_relative_to(results) and not results.is_relative_to(out)
            and not out.is_relative_to(audit.parent),'fresh figure directory outside score/audit directories required')
    result,tables,pins=load(results,audit)
    phase=Path(result['inputs']['run_dir']).resolve()
    require(not out.is_relative_to(phase) and not phase.is_relative_to(out),'figures must not modify frozen phase')
    out.mkdir(parents=True,exist_ok=False)
    style(); files={}
    hypothesis_plot(result,out,files)
    curves_plot(tables['curves'],out,files)
    controls_plot(tables['controls'],out,files)
    for path,expected in pins.items():
        require(sha(path)==expected,'figure input changed: '+path)
    record=dict(created_at=datetime.now(timezone.utc).isoformat(),program=result['program'],
                plan_sha256=result['plan_sha256'],inputs=pins,files=files,new_fits=0,new_tests=0,
                figure_notes={
                    'checkpoint_rule_effects':'Pointwise, not simultaneous, 95% t intervals; Holm across the fixed six tests. Grey ±1 pp is a prespecified practical reference, not an equivalence region or test.',
                    'complete_epoch_curves':'120 equally weighted main trajectories per native corpus; all 15 epochs, without smoothing or selected-epoch bins. No curve confidence intervals or additional hypothesis tests. Compact native canvas with an external legend; final embedded size and labels still require PDF and visual checks.',
                    'paired_role_controls':'24 CREMA-D fold-0 fixed-last matched frame pairs; whole training-group replacement, descriptive only. No confidence intervals or p values.'},
                visual_review_required=True,visual_review_performed_by_script=False)
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(figures=3,files=len(files),out=str(out),new_tests=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','audit','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    execute(parser.parse_args())

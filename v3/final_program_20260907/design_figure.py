"""Draw only the frozen study design; never read any fitted scientific result."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCIENCE_SHA = '62d7e74e1fabdca3ccfd05e14512abb1cbeef8d5857873fb8c969af31f84e8f1'
PLAN_SHA = '393109434af0bfb6d18205f8f3713aa5e08d08f0f0ac3ffc4a63f8e635422c25'
INK, SEEN, UNSEEN = '#222222', '#0072B2', '#D55E00'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(out: Path):
    out = out.resolve()
    specification = Path(__file__).with_name('SCIENCE_DESIGN.md')
    if sha(specification) != SCIENCE_SHA:
        raise ValueError('diagram must be reviewed when the frozen specification changes')
    if out.exists():
        raise ValueError('choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'pdf.fonttype':42, 'ps.fonttype':42,
                         'savefig.facecolor':'white'})
    # The PDF is intended for full-width placement at its native 7.15 inches.
    # Every text object is >=9 pt; do not shrink it in the manuscript.
    fig = plt.figure(figsize=(7.15,3.7))
    axis = fig.add_axes([.015,.035,.97,.94])
    axis.set(xlim=(0,1), ylim=(0,1)); axis.axis('off')

    def box(x,y,w,h,text,color=INK,face='#F5F5F5',weight='normal'):
        axis.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
                                     linewidth=.9,edgecolor=color,facecolor=face,zorder=2))
        axis.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,
                  linespacing=1.25,color=INK,fontweight=weight,zorder=3)

    def arrow(start,end,color=INK,dashed=False,connection='arc3,rad=0'):
        axis.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,
            linewidth=1.0,linestyle='--' if dashed else '-',color=color,
            connectionstyle=connection,zorder=1))

    axis.text(.5,.985,'Shared training trajectory, four checkpoint rules',
              ha='center',va='top',fontsize=10,fontweight='bold',color=INK)
    axis.text(.5,.918,'3 native tasks: CREMA-D (6 classes), SUBESCO (7), RAVDESS (8)',
              ha='center',va='top',fontsize=9,color=INK)

    box(.018,.585,.185,.195,'Fit on\nA speakers\nFit prompts')
    box(.267,.56,.205,.245,'One WavLM\ntraining trajectory\n15 complete epochs',weight='bold')
    box(.553,.735,.20,.135,'A query: seen\nmin CE / max UAR',SEEN,'#EEF6FA')
    box(.553,.49,.20,.135,'B query: unseen\nmin CE / max UAR',UNSEEN,'#FCF2EB')
    box(.809,.55,.177,.27,'Same outer test\n4 rule outputs\n+ all 15 epochs')
    arrow((.21,.682),(.259,.682))
    arrow((.48,.75),(.545,.802),SEEN)
    arrow((.48,.615),(.545,.557),UNSEEN)
    arrow((.761,.802),(.801,.744),SEEN)
    arrow((.761,.557),(.801,.619),UNSEEN)
    axis.text(.651,.673,'4 rules, no extra fits',ha='center',va='center',fontsize=9)

    box(.018,.287,.455,.19,
        'A/B query use the same query prompts,\ndisjoint from fit prompts.\nOuter identities are held out from fitting.')
    axis.text(.75,.394,'Outer scores never select a checkpoint.',
              ha='center',va='center',fontsize=9,fontweight='bold')
    axis.text(.75,.321,'Per task: 24 draws × 5 folds = 120 fits\nAverage 5 folds before draw-level inference',
              ha='center',va='center',fontsize=9,linespacing=1.3)

    axis.plot([.018,.986],[.238,.238],color='#BBBBBB',linewidth=.7)
    axis.text(.018,.194,'Additional CREMA-D control',ha='left',va='center',fontsize=9,fontweight='bold')
    axis.text(.018,.125,'24 B-fit models: prespecified fold 0 in each draw; same report panels and matched fit slots.',
              ha='left',va='center',fontsize=9)
    axis.text(.018,.060,'Compare paired A/B models only at epoch 15. Role-swap E is descriptive; no added tests.',
              ha='left',va='center',fontsize=9)

    files = {}
    for extension in ('pdf','png','svg'):
        path = out/('shared_trajectory_design.'+extension)
        fig.savefig(path,dpi=220)
        files[path.name] = {'sha256':sha(path), 'bytes':path.stat().st_size}
    plt.close(fig)
    if sha(specification) != SCIENCE_SHA:
        raise ValueError('specification changed during figure creation')
    record = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Design schematic only; no fitted results read or scientific statistics computed',
        science_sha256=SCIENCE_SHA,plan_sha256=PLAN_SHA,
        script_sha256=sha(__file__),files=files,
        native_width_inches=7.15,native_height_inches=3.7,minimum_text_points=9,
        main_A_fits=360,additional_B_fits=24,pilots_in_scientific_counts=0,
        main_hypotheses='Three corpora × (delta_CE, J), fixed six-test Holm family',
        formal_results_read=False,visual_review_required=True)
    (out/'DESIGN_FIGURE_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(out=str(out),files=len(files),formal_results_read=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    draw(parser.parse_args().out)

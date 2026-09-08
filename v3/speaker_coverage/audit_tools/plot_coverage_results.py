"""Render prespecified E2 contrasts after the complete result gate; no new inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLAN_SHA = '8ea69b57b7030e4b431f503d8562de230cefed2cfa7d4d092f703e8b164cae4f'
MODELS = ('cnn', 'ridge_wavlm', 'wavlm_ft')
NAMES = {'cnn': 'CNN', 'ridge_wavlm': 'Frozen WavLM + Ridge', 'wavlm_ft': 'WavLM fine-tuning'}


def render(result, output, *, preview=False):
    if not preview:
        assert result['plan_sha256'] == PLAN_SHA
        assert result['verified_formal_units'] == 720
        assert result['all_comparisons_reported'] is True
        assert result['bootstrap_repetitions'] == 10000
    contrasts = {(r['model'], r['checkpoint'], r['contrast']): r for r in result['contrasts']}
    means = {(r['model'], r['checkpoint'], r['policy']): r for r in result['policies']}
    assert len(contrasts) == 12 and len(means) == 18
    rows = [(m, c) for m in MODELS for c in ('C-U', 'R-U')]
    edge = max(abs(r[k]) for r in contrasts.values() for k in ('ci95_low_pp', 'ci95_high_pp'))
    half = max(1., np.ceil((edge + .3) * 2) / 2)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    y = np.array([5.6, 4.6, 3.1, 2.1, .6, -.4])
    for ax, checkpoint in zip(axes, ('best', 'last')):
        ax.axvline(0, color='#5b6575', linewidth=1)
        for idx, (model, contrast) in enumerate(rows):
            r = contrasts[model, checkpoint, contrast]
            lo, mid, hi = (r[k] for k in ('ci95_low_pp', 'difference_pp', 'ci95_high_pp'))
            primary = (model, checkpoint, contrast) == ('cnn', 'best', 'C-U')
            color = '#0868ac' if primary else ('#457b65' if contrast == 'C-U' else '#8d6097')
            ax.plot([lo, hi], [y[idx], y[idx]], color=color, lw=2.4 if primary else 1.8)
            ax.plot(mid, y[idx], 'D' if primary else 'o', color=color, ms=7 if primary else 5.5)
            ax.text(.99, y[idx] + .27, f'{mid:+.2f} [{lo:+.2f}, {hi:+.2f}]',
                    transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=9, color=color)
        ax.set_xlim(-half, half)
        ax.set_ylim(-1, 6.4)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.set_xlabel('Difference in mean speaker UAR (percentage points)')
        ax.set_title('Best validation-loss checkpoint' if checkpoint == 'best' else 'Last epoch: sensitivity analysis',
                     loc='left', fontsize=12, weight='bold', pad=13)
    axes[0].set_yticks(y, [f'{NAMES[m]}\n{c}' + ('  (primary)' if m == 'cnn' and c == 'C-U' else '')
                         for m, c in rows])
    title = 'Speaker selection with 576 SER training recordings per fit'
    if preview:
        title = 'SYNTHETIC LAYOUT PREVIEW — no experimental results'
    fig.suptitle(title, x=.03, ha='left', fontsize=16, weight='bold', y=.99)
    fig.text(.03, .04, 'U: uniform  |  R: representative  |  C: farthest-first coverage\n'
             '95% paired-speaker bootstrap intervals; 91 people. Conditional on fitted models and this corpus.\n'
             '720 formal fits; separate 19-unit technical pilot excluded. No equivalence or causal-mediation claim.',
             fontsize=9, color='#445064', va='bottom')
    fig.subplots_adjust(left=.21, right=.985, top=.83, bottom=.26, wspace=.12)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(output/f'e2_contrasts.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        for checkpoint, color, style in [('best', '#0868ac', '-o'), ('last', '#c16c27', '--s')]:
            vals = [means[model, checkpoint, p]['mean_speaker_uar_percent'] for p in 'URC']
            ax.plot(range(3), vals, style, color=color, label='Best' if checkpoint == 'best' else 'Last', lw=1.7)
            other = 'last' if checkpoint == 'best' else 'best'
            for i, v in enumerate(vals):
                other_v = means[model, other, 'URC'[i]]['mean_speaker_uar_percent']
                above = v > other_v or (v == other_v and checkpoint == 'best')
                ax.annotate(f'{v:.2f}', (i,v), xytext=(0,7 if above else -13),
                            textcoords='offset points', ha='center', fontsize=9, color=color)
        ax.set_title(NAMES[model], fontsize=11, weight='bold')
        ax.set_xticks(range(3), ['U: uniform', 'R: representative', 'C: coverage'])
        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-.25, 2.25)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Mean speaker UAR (%)')
    axes[-1].legend(loc='lower right', frameon=False)
    fig.suptitle('Policy means, with equal weight for each of 91 people' if not preview else title,
                 x=.045, ha='left', fontsize=14, weight='bold')
    fig.text(.045,.035,'Descriptive means; three panel draws for CNN/Ridge, two training seeds on draw 0 for WavLM fine-tuning.',
             fontsize=9,color='#445064')
    fig.tight_layout(rect=(0,.08,1,.91))
    for ext in ('png','svg'):
        fig.savefig(output/f'e2_policy_means.{ext}',dpi=180)
    plt.close(fig)


def fixture():
    rng = np.random.default_rng(41)
    contrasts, policies = [], []
    for model in MODELS:
        for checkpoint in ('best','last'):
            for contrast in ('C-U','R-U'):
                mid = float(rng.uniform(-2,2))
                contrasts.append(dict(model=model,checkpoint=checkpoint,contrast=contrast,
                    difference_pp=mid,ci95_low_pp=mid-1.2,ci95_high_pp=mid+1.2))
            for policy in 'URC':
                policies.append(dict(model=model,checkpoint=checkpoint,policy=policy,
                                     mean_speaker_uar_percent=float(rng.uniform(38,58))))
    return dict(contrasts=contrasts,policies=policies)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--synthetic-preview',action='store_true')
    args=parser.parse_args()
    if args.synthetic_preview:
        assert args.results is None
        render(fixture(),args.out,preview=True)
    else:
        assert args.results is not None
        payload=args.results.read_bytes()
        render(json.loads(payload),args.out)
        receipt={'schema':'ser-coverage-figure-receipt-1','results_sha256':hashlib.sha256(payload).hexdigest(),
                 'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('e2_*')},
                 'new_inference_performed':False}
        (args.out/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()

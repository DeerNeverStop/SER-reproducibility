"""Publication figures from sealed 480-unit score tables; no new inference.

Normal mode requires the exact frozen plan, accepted complete gate and score
folder. --demo creates explicitly labelled synthetic illustrations only.
This nested tool is outside the frozen training-source glob.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

FAMILIES = ('primary', 'fixed_config_best', 'fixed_config_last')
TABLES = ('pred_metrics.csv', 'selected_episodes.csv', 'draws.csv')
FIELDS = ('seen_validation_uar_percent', 'unseen_validation_uar_percent',
          'seen_test_uar_percent', 'unseen_test_uar_percent',
          'seen_gap_pp', 'unseen_gap_pp', 'delta_validation_pp',
          'delta_test_pp', 'delta_gap_pp')
SCOPE = ('Whole-fold six-class UAR; five folds averaged within each of 24 draws. '
         'Inference is conditional on the fixed corpus and prespecified procedure.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value, name):
    number = float(value)
    require(math.isfinite(number), 'nonfinite field: ' + name)
    return number


def validate_contract(plan, gate, result):
    """Structural checks only; these are not a replacement for the frozen gate."""
    require(plan.get('schema') == 'ser-dual-validation-plan-1' and
            plan.get('program') == 'SER26-DUAL-VALIDATION-1' and plan.get('draft') is False,
            'a frozen formal dual-validation plan is required')
    require(plan.get('plan_sha256') == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
            'plan semantic hash differs')
    units = plan['units']
    require(len(units) == 480 and len({u['unit_id'] for u in units}) == 480 and
            {(u['draw'], u['fold'], u['config_index']) for u in units} ==
            {(d, f, c) for d in range(24) for f in range(5) for c in range(4)},
            'the complete 480-unit grid is required')
    require(gate.get('schema') == 'ser-dual-validation-result-gate-1' and
            gate.get('pass') is True and gate.get('phase') == 'formal' and
            gate.get('units') == 480 and gate.get('scores_computed') is False,
            'a passed complete formal gate must precede plotting')
    require(gate.get('gate_sha256') == digest({k: v for k, v in gate.items() if k != 'gate_sha256'}),
            'gate semantic hash differs')
    require(set(gate.get('done_sha256', {})) == {u['unit_id'] for u in units},
            'gate must bind every formal DONE')
    require(result.get('schema') == 'ser-dual-validation-analysis-1' and
            result.get('phase') == 'formal' and result.get('analyzed_formal_units') == 480 and
            result.get('draws') == 24 and result.get('folds_per_draw') == 5 and
            result.get('configurations') == 4 and result.get('fixed_config_index') == 3,
            'score output does not describe the complete prespecified analysis')
    require(plan['plan_sha256'] == gate.get('plan_sha256') == result.get('plan_sha256') and
            result.get('accepted_gate_sha256') == gate['gate_sha256'] and
            result.get('done_mapping_sha256') == digest(gate['done_sha256']),
            'plan / accepted gate / score identity differs')
    require(result.get('source_sha256') == plan.get('sources', {}).get('v3/inner_validation/score.py') and
            isinstance(result.get('source_sha256'), str), 'score source is not the frozen scorer')
    require(result.get('fixed_config', {}).get('lr_encoder') == 5e-5 and
            result.get('fixed_config', {}).get('lr_head') == 1e-3, 'labelled fixed configuration differs')
    require(result.get('primary_hypothesis_tests') == 1 and
            result.get('additional_control_hypothesis_tests') == 0 and
            result.get('all_prespecified_controls_reported') is True,
            'prespecified inference or controls differ')
    primary = result['primary']
    require(primary.get('draws') == 24 and primary.get('df') == 23 and
            primary.get('alternative') == 'two-sided' and type(primary.get('t_test_defined')) is bool,
            'primary interval contract differs')
    require(set(result.get('table_sha256', {})) == set(TABLES), 'score table inventory differs')


def plot_data(result, raw_draws):
    require(len(raw_draws) == 24 and sorted(int(row['draw']) for row in raw_draws) == list(range(24)),
            'exactly 24 distinct complete draw rows are required')
    rows = sorted(raw_draws, key=lambda row: int(row['draw']))
    require(all(int(row['folds']) == 5 for row in rows), 'a draw does not contain all five folds')
    arrays = {}
    descriptions = result['descriptive_draw_summaries']
    for family in FAMILIES:
        for field in FIELDS:
            key = family + '_' + field
            values = np.asarray([finite(row[key], key) for row in rows])
            require(abs(float(values.mean()) - finite(descriptions[key]['mean'], key)) < 1e-9,
                    'draw table disagrees with recorded descriptive mean: ' + key)
            if field.endswith('uar_percent'):
                require(bool(((values >= 0) & (values <= 100)).all()), 'UAR outside 0 to 100')
            arrays[key] = values
        require(np.allclose(arrays[family+'_delta_gap_pp'],
                            arrays[family+'_delta_validation_pp']-arrays[family+'_delta_test_pp'],
                            atol=1e-9, rtol=0), 'draw gap identity differs')
        require(np.allclose(arrays[family+'_delta_validation_pp'],
                            arrays[family+'_seen_validation_uar_percent']-arrays[family+'_unseen_validation_uar_percent'],
                            atol=1e-9, rtol=0) and
                np.allclose(arrays[family+'_delta_test_pp'],
                            arrays[family+'_seen_test_uar_percent']-arrays[family+'_unseen_test_uar_percent'],
                            atol=1e-9, rtol=0), 'draw validation/test difference identity differs')
    require(bool((arrays['fixed_config_last_delta_test_pp'] == 0).all()) and
            np.array_equal(arrays['fixed_config_last_delta_gap_pp'],
                           arrays['fixed_config_last_delta_validation_pp']),
            'fixed-config last common-test cancellation differs')
    primary = result['primary']
    estimate = finite(primary['estimate_pp'], 'primary estimate')
    require(abs(estimate - arrays['primary_delta_gap_pp'].mean()) < 1e-9,
            'primary estimate disagrees with its 24 draw values')
    ci = [finite(x, 'recorded t CI') for x in primary['t_ci95_pp']]
    require(len(ci) == 2 and ci[0] <= estimate <= ci[1], 'invalid recorded primary t interval')
    return dict(arrays=arrays, primary_mean=estimate, primary_ci=ci,
                t_test_defined=primary['t_test_defined'])


def load_inputs(plan_path, gate_path, score_dir):
    paths = [plan_path, gate_path, score_dir/'results.json', *(score_dir/name for name in TABLES)]
    require(all(path.is_file() for path in paths), 'a plan, complete gate and all sealed score tables are required')
    pins = {str(path): file_sha(path) for path in paths}
    plan, gate, result = (read_json(p) for p in paths[:3])
    validate_contract(plan, gate, result)
    require(result['plan_file_sha256'] == pins[str(plan_path)] and
            result['accepted_gate_file_sha256'] == pins[str(gate_path)],
            'score output accepted different plan or gate bytes')
    require(all(result['table_sha256'][name] == pins[str(score_dir/name)] for name in TABLES),
            'a score table changed since scoring')
    with (score_dir/'draws.csv').open(encoding='utf-8', newline='') as stream:
        raw_draws = list(csv.DictReader(stream))
    data = plot_data(result, raw_draws)
    return plan, result, data, pins


def make_figures(data, synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    ink, blue, orange, gray = '#172B3A', '#2166AC', '#B56218', '#76838B'
    arrays = data['arrays']
    heading = 'SYNTHETIC DEMO — NOT EXPERIMENT RESULTS' if synthetic else 'Speaker overlap in validation'
    fig1, (left, right) = plt.subplots(1, 2, figsize=(10.7, 4.9), gridspec_kw={'width_ratios': [1.04, 1]})
    fig1.subplots_adjust(left=.075, right=.97, top=.78, bottom=.21, wspace=.37)
    fig1.suptitle(heading, x=.075, y=.98, ha='left', fontsize=14, weight='bold',
                  color='#9C2E2E' if synthetic else ink)
    fig1.text(.075, .91, 'Both selection rules share every training trajectory and use their own validation sets.',
              fontsize=10, color=gray)
    x = np.arange(2)
    means = {role: [float(arrays[f'primary_{rule}_{role}_uar_percent'].mean()) for rule in ('seen', 'unseen')]
             for role in ('validation', 'test')}
    for role, shift, color, label in (('validation', -.19, blue, 'Own validation'),
                                      ('test', .19, orange, 'Common SI test')):
        bars = left.bar(x + shift, means[role], width=.34, color=color, label=label, zorder=3)
        left.bar_label(bars, labels=[f'{v:.2f}' for v in means[role]], padding=4, fontsize=10, color=ink)
    left.set(xticks=x, xticklabels=['Seen-speaker\nvalidation rule', 'Unseen-speaker\nvalidation rule'],
             ylabel='UAR (%)', ylim=(0, 106), title='A   Selected-model validation and test means')
    left.set_yticks([0, 20, 40, 60, 80, 100])
    left.set_axisbelow(True)
    left.grid(axis='y', color='#E4E8EB', linewidth=.7)
    left.legend(loc='upper center', bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, fontsize=9)
    effects = arrays['primary_delta_gap_pp']
    y = np.arange(1, 25)
    right.axvline(0, color='#A4AFB5', lw=1, ls='--', zorder=1)
    right.scatter(effects, y, s=22, color=gray, alpha=.85, edgecolors='white', linewidths=.35, zorder=3)
    mean, ci = data['primary_mean'], data['primary_ci']
    if data['t_test_defined']:
        right.errorbar(mean, 27.5, xerr=np.asarray([[mean-ci[0]], [ci[1]-mean]]),
                       fmt='D', color=blue, capsize=4, markersize=6, elinewidth=2, zorder=4)
        annotation = f'Mean {mean:+.2f} pp; 95% t CI [{ci[0]:+.2f}, {ci[1]:+.2f}]'
    else:
        right.scatter([mean], [27.5], marker='D', s=40, color=blue, zorder=4)
        annotation = f'Mean {mean:+.2f} pp; t interval undefined (zero draw variance)'
    right.axhline(25.5, color='#E4E8EB', linewidth=.8)
    right.set(yticks=[1, 6, 12, 18, 24, 27.5], yticklabels=['1', '6', '12', '18', '24', 'Mean'],
              ylim=(0, 30), ylabel='Draw (five-fold mean)', xlabel=r'$\Delta(V-T)$ (percentage points)',
              title='B   Prespecified primary gap difference')
    right.xaxis.set_major_locator(MaxNLocator(nbins=5))
    right.text(.5, 1.13, annotation, transform=right.transAxes, ha='center', fontsize=9, color=ink)
    fig1.text(.075, .055, r'$\Delta(V-T) = (V_{seen}-T_{seen})-(V_{unseen}-T_{unseen})$. '
              'Positive values indicate greater relative optimism for seen-speaker validation.', fontsize=9, color=gray)
    fig1.text(.075, .017, '24 complete draws, 5 folds per draw. Only the prespecified primary 95% t interval is shown.',
              fontsize=9, color=gray)

    fig2, ax = plt.subplots(figsize=(9.6, 4.6))
    fig2.subplots_adjust(left=.26, right=.96, top=.77, bottom=.23)
    fig2.suptitle('SYNTHETIC DEMO — NOT EXPERIMENT RESULTS' if synthetic else 'Prespecified analysis and controls',
                  x=.055, y=.97, ha='left', fontsize=14, weight='bold', color='#9C2E2E' if synthetic else ink)
    fig2.text(.055, .885, 'Each point is one complete draw; diamonds are means. No inferential intervals in this panel.',
              fontsize=10, color=gray)
    offsets = np.linspace(-.19, .19, 24)  # Same draw offset in each row; never alter effect coordinates.
    colors = [blue, orange, '#47796D']
    for index, (family, color) in enumerate(zip(FAMILIES, colors)):
        values = arrays[family+'_delta_gap_pp']
        center = 2-index
        ax.scatter(values, center+offsets, s=24, color=color, alpha=.65, linewidths=.35,
                   edgecolors='white', zorder=3)
        ax.scatter([float(values.mean())], [center], s=70, marker='D', color=ink,
                   edgecolors='white', linewidths=.8, zorder=5)
    ax.axvline(0, color='#A4AFB5', lw=1, ls='--', zorder=1)
    ax.set(yticks=[2, 1, 0], yticklabels=['Selected config + own best\nPrimary procedure',
                                       'Fixed config + own best\nCheckpoint selection remains',
                                       'Fixed config + last\nNo validation selection'],
           ylim=(-.5, 2.6), xlabel=r'$\Delta(V-T)$ (percentage points)')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.grid(axis='x', color='#E4E8EB', linewidth=.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.legend(handles=[Line2D([], [], color=gray, marker='o', linestyle='', markersize=5, label='One draw'),
                       Line2D([], [], color=ink, marker='D', linestyle='', markersize=6, label='Mean (no interval)')],
              loc='upper center', bbox_to_anchor=(.5, 1.16), ncol=2, frameon=False, fontsize=9)
    fig2.text(.055, .095, 'Controls are descriptive: no additional tests or confidence intervals. '
              'Their differences do not identify causal mediation.', fontsize=9, color=gray)
    fig2.text(.055, .048, 'Fixed configuration: encoder LR 5e-5, head LR 1e-3. '
              'Fixed-config last uses the identical test model for both rules.', fontsize=9, color=gray)
    return {'figure1_validation_optimism': fig1, 'figure2_prespecified_controls': fig2}


def chinese_notes(data, synthetic):
    flag = ('**合成数据演练，不能用作实验结果或论文证据；演练区间为排版用的人工设定值，'
            '不是统计推断。**\n\n' if synthetic else '')
    value, ci = data['primary_mean'], data['primary_ci']
    interval = (f'预定主效应均值为 {value:+.3f} pp，已记录的唯一主 95% t 区间为 '
                f'[{ci[0]:+.3f}, {ci[1]:+.3f}] pp。' if data['t_test_defined'] else
                f'主效应均值为 {value:+.3f} pp；所有 draw 相同，t 推断未定义，图中不画零宽区间。')
    return (flag+'图 1 左侧为两套规则各自选模后的 validation 和共同 SI test UAR，'
            '先每个 draw 内五折等权平均，再对 24 个 draw 等权平均；柱形无置信区间。'
            '右侧每个灰点是一整个 draw 的 Δ(V−T)，蓝色菱形为主均值。'+interval+'\n\n'
            '图 2 同时保留 primary、固定配置各自 best、固定配置 last 的全部 24 个 draw 和均值；'
            '控制项只作描述，不增加检验或区间。固定配置 best 仍用验证集选择 checkpoint；'
            '固定配置 last 没有验证选择，两规则外测 T 精确相同。\n\n'
            '正 Δ 表示 seen 验证相对于共同陌生说话人外测更乐观，并不等于外测性能下降。'
            '区间是固定语料及预定随机化流程条件下的 draw 级推断，不能当作独立新说话人的人群区间；'
            '也不证明声纹机制、因果中介或 SD–SI 差距消除。'
            '本工具只呈现已封存评分表与已记录主区间，不读取 logits、不训练、不新增统计检验。\n')


def synthetic_data():
    """Deterministic fake values for layout inspection, never a study estimate."""
    index = np.arange(24, dtype=float)
    arrays = {}
    for family, scale in zip(FAMILIES, (1., .72, .43)):
        seen_v = 61 + 2*np.sin(index*.43) + scale
        unseen_v = 57 + 1.6*np.cos(index*.37)
        unseen_t = 54 + 1.1*np.sin(index*.23)
        seen_t = unseen_t if family == 'fixed_config_last' else unseen_t+.5*np.cos(index*.41)*scale
        dv, dt = seen_v-unseen_v, seen_t-unseen_t
        values = (seen_v, unseen_v, seen_t, unseen_t, seen_v-seen_t,
                  unseen_v-unseen_t, dv, dt, dv-dt)
        arrays.update({family+'_'+field: value for field, value in zip(FIELDS, values)})
    # The demo interval is a chosen layout fixture, not an inferential calculation.
    mean = float(arrays['primary_delta_gap_pp'].mean())
    return dict(arrays=arrays, primary_mean=mean, primary_ci=[mean-.62, mean+.62], t_test_defined=True)


def execute(args):
    out = Path(args.out).resolve()
    require(not out.exists(), 'figure output directory must be new; nothing is overwritten')
    synthetic = bool(args.demo)
    pins, plan_sha = {}, None
    if synthetic:
        require(not any((args.plan, args.gate, args.scores)), '--demo must not consume real experiment inputs')
        data = synthetic_data()
    else:
        require(all((args.plan, args.gate, args.scores)), '--plan, --gate and --scores are all required')
        plan_path, gate_path, score_dir = [Path(p).resolve() for p in (args.plan, args.gate, args.scores)]
        require(not out.is_relative_to(score_dir) and not score_dir.is_relative_to(out) and
                not plan_path.is_relative_to(out) and not gate_path.is_relative_to(out),
                'output must be outside the score inputs and cannot contain plan/gate inputs')
        plan, result, data, pins = load_inputs(plan_path, gate_path, score_dir)
        plan_sha = plan['plan_sha256']
    operator_sha = file_sha(__file__)
    figures = make_figures(data, synthetic)
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib.pyplot as plt
    for name, figure in figures.items():
        figure.savefig(out/(name+'.png'), dpi=300)
        figure.savefig(out/(name+'.svg'))
        plt.close(figure)
    with (out/'figure_notes_zh.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(chinese_notes(data, synthetic))
    require(all(file_sha(path) == wanted for path, wanted in pins.items()), 'a plot input changed during rendering')
    require(file_sha(__file__) == operator_sha, 'plotting operator changed during rendering')
    names = [name+suffix for name in figures for suffix in ('.png', '.svg')]+['figure_notes_zh.md']
    receipt = dict(schema='ser-dual-validation-figures-1', synthetic_demo=synthetic,
                   complete_formal_gate_required=not synthetic, plan_sha256=plan_sha,
                   source_sha256=operator_sha, input_sha256=pins,
                   output_sha256={name: file_sha(out/name) for name in names},
                   new_statistical_tests=0, new_inferential_intervals=0,
                   primary_interval='Recorded prespecified 95% t CI only; none for descriptive controls',
                   scope=SCOPE, figure_roles={'figure1': 'Selected-rule V/T means and 24-draw primary effect',
                                             'figure2': 'Three prespecified analyses; draw values and means only'},
                   created_at_unix=time.time())
    with (out/'figure_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'figures_written': 2, 'formats': ['png', 'svg'], 'synthetic_demo': synthetic,
                      'new_tests': 0, 'output': str(out)}, ensure_ascii=False))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'gate', 'scores'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--demo', action='store_true', help='Only synthetic data, explicitly watermarked')
    execute(parser.parse_args())


if __name__ == '__main__':
    main()

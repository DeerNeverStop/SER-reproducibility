"""Prespecified, post-completion analysis of the 384-trajectory final program.

No engine metric or stored epoch choice is used to select a checkpoint here.
This is independent numerical analysis of saved logits, not independent training.
The CLI first invokes the complete source/weights/ledger gate. Pilot predictions
are never loaded. Only delta_CE and J receive inference (six tests in total).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .plan import PROGRAM, ANALYSIS, CORPUS_ORDER

RULES = ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar')
GROUPS = ('A', 'B', 'outer')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def classification(labels, logits):
    """Return float64 mean CE and exact rational macro recall in [0,1]."""
    y, z = np.asarray(labels), np.asarray(logits)
    require(y.ndim == 1 and y.dtype.kind in 'iu' and y.size > 0, 'invalid integer labels')
    require(z.ndim == 2 and z.shape[0] == y.size and z.shape[1] >= 2
            and z.dtype.kind in 'fiu' and np.isfinite(z).all(), 'invalid finite logits')
    classes = z.shape[1]
    require(bool(((y >= 0) & (y < classes)).all()), 'label outside native head')
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (y, z.argmax(axis=1)), 1)
    supports = confusion.sum(axis=1)
    require(bool((supports > 0).all()), 'missing native class support')
    if bool((supports == supports[0]).all()):
        recall = Fraction(int(np.trace(confusion)), int(confusion.sum()))
    else:
        recall = sum((Fraction(int(confusion[c, c]), int(supports[c]))
                      for c in range(classes)), Fraction(0)) / classes
    values = z.astype(np.float64)
    maximum = values.max(axis=1)
    shifted = values - maximum[:, None]
    # Reimplement the frozen float64 operation order to preserve exact CE ties;
    # independent code/metrics are used, not a different floating-point convention.
    normalizer = maximum + np.log(np.exp(shifted).sum(axis=1))
    losses = normalizer - values[np.arange(y.size), y]
    ce = float(np.mean(losses))
    require(math.isfinite(ce) and ce >= 0, 'invalid mean CE')
    return ce, recall


def check_grid(plan):
    require(plan['program'] == PROGRAM and plan['analysis'] == ANALYSIS, 'analysis contract differs')
    units = [u for u in plan['units'] if u['phase'] == 'formal']
    wanted = {(c, d, f, 'A') for c in CORPUS_ORDER for d in range(24) for f in range(5)}
    wanted |= {('cremad', d, 0, 'B') for d in range(24)}
    actual = [(u['corpus'], u['draw'], u['fold'], u['arm']) for u in units]
    require(len(units) == 384 and set(actual) == wanted and len(set(actual)) == 384,
            'complete 360 main plus 24 control grid required')
    require(len({u['unit_id'] for u in units}) == 384, 'duplicate formal UID')
    for u in units:
        require(u['n_classes'] == {'cremad': 6, 'subesco': 7, 'ravdess': 8}[u['corpus']], 'native classes differ')
        require(u['selection_enabled'] is (u['arm'] == 'A')
                and u['prediction_epochs'] == (list(range(1, 16)) if u['arm'] == 'A' else [15]),
                'selection/epoch contract differs')
    return units


def trajectory(unit, arrays, rows):
    """Independently select epochs after exact path/label/epoch validation."""
    epochs = np.asarray(arrays['epochs'])
    require(epochs.dtype.kind in 'iu' and epochs.ndim == 1
            and epochs.tolist() == unit['prediction_epochs'], 'prediction epoch support differs')
    scores = {}
    for group in GROUPS:
        paths = np.asarray(arrays[group+'__paths'])
        y = np.asarray(arrays[group+'__labels'])
        z = np.asarray(arrays[group+'__all_epoch_logits'])
        require(paths.tolist() == unit['report'][group], 'prediction path order differs')
        expected = np.asarray([rows[p]['label_index'] for p in unit['report'][group]])
        require(y.dtype.kind in 'iu' and y.shape == expected.shape and np.array_equal(y, expected),
                'prediction labels differ from plan')
        require(z.dtype == np.float32 and z.shape == (len(epochs), len(paths), unit['n_classes'])
                and np.isfinite(z).all(), 'prediction shape/dtype/finiteness differs')
        scores[group] = [classification(y, values) for values in z]
    selected = {'last': 15}
    if unit['arm'] == 'A':
        for name, group in (('seen', 'A'), ('unseen', 'B')):
            selected[name+'_ce'] = min(range(15), key=lambda i: scores[group][i][0])+1
            selected[name+'_uar'] = max(range(15), key=lambda i: scores[group][i][1])+1
    if 'stored_selected_epochs' in arrays:
        require(arrays['stored_selected_epochs'] == selected,
                'independent selection differs from committed checkpoint choice')
    return dict(unit=unit, epochs=epochs.tolist(), scores=scores, selected=selected)


def mean(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(float(v)) for v in values), 'invalid mean support')
    return math.fsum(values) / len(values)


def t_estimate(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (24,) and np.isfinite(x).all(), 'inference requires 24 complete draw effects')
    average = mean(x)
    sd = 0.0 if bool((x == x[0]).all()) else float(np.std(x, ddof=1))
    result = dict(mean_pp=average, n_draws=24, df=23, sd_draw_pp=sd,
                  interval_scope='pointwise, not simultaneous; conditional on fixed corpus and program')
    if sd == 0:
        return dict(result, status='undefined_t_zero_sample_variance', se_pp=0.0,
                    t_statistic=None, p_two_sided=None, pointwise_95_ci_pp=None)
    se = sd / math.sqrt(24)
    statistic = average / se
    half = float(student_t.ppf(.975, 23))*se
    return dict(result, status='estimated', se_pp=se, t_statistic=statistic,
                p_two_sided=float(2*student_t.sf(abs(statistic), 23)),
                pointwise_95_ci_pp=[average-half, average+half])


def holm_six(p_values):
    """Undefined t tests occupy their fixed family slot conservatively as p=1."""
    require(len(p_values) == 6, 'Holm family must remain six')
    require(all(p is None or (math.isfinite(p) and 0 <= p <= 1) for p in p_values), 'invalid p value')
    order = sorted(range(6), key=lambda i: (1.0 if p_values[i] is None else p_values[i], i))
    adjusted, running = [None]*6, 0.0
    for rank, i in enumerate(order):
        p = 1.0 if p_values[i] is None else p_values[i]
        running = max(running, min(1.0, (6-rank)*p))
        if p_values[i] is not None:
            adjusted[i] = running
    return adjusted


def analyze(plan, prediction_loader):
    """Pure saved-logit analysis; caller must provide the full archival gate."""
    units = check_grid(plan)
    contexts, curves, selections, controls = [], [], [], []
    measured, exact_effects = {}, {}
    for u in sorted(units, key=lambda x: (CORPUS_ORDER.index(x['corpus']), x['draw'], x['fold'], x['arm'])):
        value = trajectory(u, prediction_loader(u), plan['rows'][u['corpus']])
        key = (u['corpus'], u['draw'], u['fold'], u['arm'])
        measured[key] = value
        if u['arm'] != 'A':
            continue
        base = dict(unit_id=u['unit_id'], corpus=u['corpus'], draw=u['draw'], fold=u['fold'])
        sc = value['scores']
        outer = [100*float(pair[1]) for pair in sc['outer']]
        oracle = max(outer)
        for i, epoch in enumerate(value['epochs']):
            curves.append(dict(base, epoch=epoch, seen_ce=sc['A'][i][0], unseen_ce=sc['B'][i][0],
                               seen_uar=100*float(sc['A'][i][1]), unseen_uar=100*float(sc['B'][i][1]),
                               outer_uar=outer[i]))
        target = {}
        for rule in (*RULES, 'last'):
            epoch = value['selected'][rule]
            i = value['epochs'].index(epoch)
            # Keep the exact native-class recall until paired subtraction.
            # Separate percent conversions can make an algebraic zero into a
            # draw-varying ~1e-14 residual, which a t statistic may magnify.
            target[rule] = sc['outer'][i][1]
            selections.append(dict(base, rule=rule, selected_epoch=epoch,
                                   seen_uar=100*float(sc['A'][i][1]), unseen_uar=100*float(sc['B'][i][1]),
                                   outer_uar=outer[i], oracle_outer_uar=oracle,
                                   oracle_shortfall_pp=oracle-outer[i], oracle_used_for_selection=False))
        delta_ce = 100*(target['seen_ce']-target['unseen_ce'])
        delta_uar = 100*(target['seen_uar']-target['unseen_uar'])
        exact_effects[u['corpus'],u['draw'],u['fold']] = {
            'delta_CE_pp':delta_ce, 'delta_UAR_pp':delta_uar, 'J_pp':delta_ce-delta_uar}
        contexts.append(dict(base, delta_CE_pp=float(delta_ce), delta_UAR_pp=float(delta_uar),
                             J_pp=float(delta_ce-delta_uar),
                             last_outer_uar=outer[-1], oracle_outer_uar=oracle,
                             middle_8_10_minus_late_14_15_pp=mean(outer[7:10])-mean(outer[13:15]),
                             ce_epoch_agreement=int(value['selected']['seen_ce']==value['selected']['unseen_ce']),
                             uar_epoch_agreement=int(value['selected']['seen_uar']==value['selected']['unseen_uar'])))
    for draw in range(24):
        a, b = measured['cremad', draw, 0, 'A'], measured['cremad', draw, 0, 'B']
        require(a['unit']['pair_id'] == b['unit']['pair_id']
                and a['unit']['report'] == b['unit']['report']
                and a['unit']['report_batches'] == b['unit']['report_batches']
                and a['unit']['seeds'] == b['unit']['seeds'], 'control pair frame differs')
        ai, bi = a['epochs'].index(15), b['epochs'].index(15)
        qa_ma = a['scores']['A'][ai][1]; qa_mb = b['scores']['A'][bi][1]
        qb_ma = a['scores']['B'][ai][1]; qb_mb = b['scores']['B'][bi][1]
        controls.append(dict(corpus='cremad', draw=draw, fold=0, pair_id=a['unit']['pair_id'],
                             A_unit_id=a['unit']['unit_id'], B_unit_id=b['unit']['unit_id'],
                             QA_MA15=100*float(qa_ma), QA_MB15=100*float(qa_mb),
                             QB_MA15=100*float(qb_ma), QB_MB15=100*float(qb_mb),
                             E_pp=float(50*((qa_ma-qa_mb)+(qb_mb-qb_ma)))))
    draws = []
    for corpus in CORPUS_ORDER:
        for draw in range(24):
            part = [r for r in contexts if r['corpus']==corpus and r['draw']==draw]
            require({r['fold'] for r in part} == set(range(5)) and len(part)==5, 'incomplete draw')
            draws.append(dict(corpus=corpus, draw=draw,
                              **{k:float(sum((exact_effects[corpus,draw,fold][k] for fold in range(5)),
                                              Fraction(0))/5)
                                 for k in ('delta_CE_pp','delta_UAR_pp','J_pp')},
                              **{k:mean(r[k] for r in part) for k in (
                                  'last_outer_uar','middle_8_10_minus_late_14_15_pp')}))
    tests = []
    for corpus in CORPUS_ORDER:
        for estimand in ('delta_CE', 'J'):
            test = t_estimate([r[estimand+'_pp'] for r in draws if r['corpus']==corpus])
            tests.append(dict(corpus=corpus, estimand=estimand, **test))
    adjusted = holm_six([r['p_two_sided'] for r in tests])
    for test, p in zip(tests, adjusted):
        test['holm_p_six'] = p
        test['reject_familywise_05'] = bool(p is not None and p <= .05)
    summaries = {}
    for corpus in CORPUS_ORDER:
        dd = [r for r in draws if r['corpus']==corpus]
        ss = [r for r in selections if r['corpus']==corpus]
        cc = [r for r in contexts if r['corpus']==corpus]
        summaries[corpus] = dict(
            main_trajectories=120, n_draws=24, folds_per_draw=5,
            delta_UAR_descriptive_mean_pp=mean(r['delta_UAR_pp'] for r in dd),
            last_outer_descriptive_mean_uar=mean(r['last_outer_uar'] for r in dd),
            middle_8_10_minus_late_14_15_descriptive_mean_pp=mean(r['middle_8_10_minus_late_14_15_pp'] for r in dd),
            ce_epoch_agreement_fraction=mean(r['ce_epoch_agreement'] for r in cc),
            uar_epoch_agreement_fraction=mean(r['uar_epoch_agreement'] for r in cc),
            rules={rule:dict(mean_outer_uar=mean(r['outer_uar'] for r in ss if r['rule']==rule),
                            mean_selected_epoch=mean(r['selected_epoch'] for r in ss if r['rule']==rule),
                            selected_epoch_counts={str(epoch):count for epoch,count in sorted(
                                Counter(r['selected_epoch'] for r in ss if r['rule']==rule).items())},
                            mean_oracle_shortfall_pp=mean(r['oracle_shortfall_pp'] for r in ss if r['rule']==rule))
                   for rule in (*RULES,'last')})
    result = dict(schema='ser-final-program-score-1', program=PROGRAM,
                  plan_sha256=plan['plan_sha256'], analysis_contract=ANALYSIS,
                  counts=dict(formal_fits=384, main_A=360, control_B=24, pilot_scored=0,
                              main_contexts=360, corpus_draws=72, family_tests=6),
                  tests=tests, corpora=summaries,
                  control=dict(estimand='0.5*((QA(MA15)-QA(MB15))+(QB(MB15)-QB(MA15)))',
                               n_pairs=24, n_added_B_fits=24, scope='CREMA-D prespecified fold0; descriptive only',
                               E_mean_pp=mean(r['E_pp'] for r in controls), inference=False),
                  interpretation=dict(
                      positive_delta='seen-based checkpoint rule has higher outer UAR; not an estimate of V-T optimism',
                      J='difference between CE-based and UAR-based checkpoint-rule effects',
                      scope='conditional fixed-corpus randomization and training program; native tasks differ across corpora',
                      prior_data_exposure=True, bootstrap=False, equivalence_test=False,
                      curves_and_oracle='descriptive; outer never selects a retained checkpoint',
                      control='whole training-group replacement; not an individual speaker effect; query also used by main A selection',
                      zero_variance='undefined t remains untested; reserved Holm family slot treated as p=1 internally'))
    return result, dict(contexts=contexts, draws=draws, selected_epochs=selections, curves=curves, controls=controls)


def check_complete_gate(gate, plan, lock):
    ids = {u['unit_id'] for u in check_grid(plan)}
    require(gate.get('pass') is True and gate.get('phase')=='formal'
            and gate.get('expected_units')==384 and gate.get('verified_units')==384,
            'complete formal 384 gate required before scoring')
    require(gate['plan_sha256']==plan['plan_sha256'] and gate['lock_sha256']==lock['lock_sha256'], 'gate identity differs')
    require(set(gate['done_sha256'])==ids and set(gate['artifacts_sha256'])==ids, 'gate must bind every formal unit')


def score_archive(repo, run_dir, out):
    from . import run
    repo, run_dir, out = Path(repo).resolve(), Path(run_dir).resolve(), Path(out).resolve()
    require(not out.exists() and not out.is_relative_to(run_dir) and not run_dir.is_relative_to(out),
            'output must be fresh, outside the frozen phase, and not contain inputs')
    require(read_json(run_dir/'SOURCE_LOCK.json')['phase']=='formal', 'pilot scientific scoring is forbidden')
    # This full gate includes all weights and all formal units, not only prediction bytes.
    gate = run.audit_phase(repo, run_dir)
    plan, lock = run.checked_plan(repo, run_dir)
    check_complete_gate(gate, plan, lock)
    top = {name:file_sha(run_dir/name) for name in ('SOURCE_LOCK.json','plan_snapshot.json','COMPLETE_GATE.json','ledger.jsonl')}
    require(read_json(run_dir/'COMPLETE_GATE.json') == gate, 'persisted complete gate differs')
    require(top['ledger.jsonl']==gate['ledger_sha256'], 'ledger differs from complete gate')
    input_hashes = dict(top)
    def loader(unit):
        folder = run_dir/'units'/unit['unit_id']
        require(file_sha(folder/'DONE')==gate['done_sha256'][unit['unit_id']], 'DONE changed after full gate')
        done = read_json(folder/'DONE')
        require(done['artifacts']==gate['artifacts_sha256'][unit['unit_id']], 'artifact inventory differs from gate')
        root = folder/done['attempt']
        for name in ('predictions.npz','receipt.json'):
            relative = done['attempt']+'/'+name
            require(file_sha(root/name)==done['artifacts'][relative], 'analysis artifact changed after full gate')
            input_hashes[(root/name).relative_to(run_dir).as_posix()] = done['artifacts'][relative]
        input_hashes[(folder/'DONE').relative_to(run_dir).as_posix()] = gate['done_sha256'][unit['unit_id']]
        with np.load(root/'predictions.npz', allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in z.files}
        arrays['stored_selected_epochs'] = read_json(root/'receipt.json')['info']['selected_epochs']
        return arrays
    result, tables = analyze(plan, loader)
    for rel, expected in input_hashes.items():
        require(file_sha(run_dir/rel)==expected, 'analysis input changed during scoring: '+rel)
    require(lock['sources']==run.sources(repo, True), 'source changed during analysis')
    result['inputs'] = dict(run_dir=str(run_dir), lock_sha256=lock['lock_sha256'],
                           complete_gate_sha256=top['COMPLETE_GATE.json'],
                           score_source_sha256=file_sha(Path(__file__)), byte_sha256=input_hashes,
                           complete_gate_scope=gate['scope'])
    out.mkdir(parents=True, exist_ok=False)
    # Preserve the exact gate that released this analysis; a later re-audit can
    # legitimately write a new timestamped gate in the run directory.
    with (out/'complete_gate.json').open('xb') as f:
        f.write((run_dir/'COMPLETE_GATE.json').read_bytes())
    require(file_sha(out/'complete_gate.json')==top['COMPLETE_GATE.json'], 'gate snapshot drift')
    result['complete_gate_snapshot'] = dict(path='complete_gate.json', sha256=top['COMPLETE_GATE.json'])
    result['tables'] = {}
    for name, values in tables.items():
        path = out/(name+'.csv')
        with path.open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
        result['tables'][path.name] = dict(rows=len(values), sha256=file_sha(path))
    result['result_sha256'] = digest(result)
    with (out/'results.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = score_archive(args.repo, args.run_dir, args.out)
    print(json.dumps(dict(complete=True, counts=result['counts'], result_sha256=result['result_sha256'])))


if __name__ == '__main__':
    main()

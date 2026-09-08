"""Synthetic tests only; no real scientific predictions or GPU are accessed."""
from copy import deepcopy
from fractions import Fraction
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t as student_t

from . import score


def logits_for(labels, correct, classes, margin=2):
    y = np.asarray(labels, dtype=np.int64)
    predicted = (y+1) % classes
    predicted[:correct] = y[:correct]
    z = np.zeros((len(y), classes), np.float32)
    z[np.arange(len(y)), predicted] = margin
    return z


def synthetic_plan():
    """Minimal numerical plan, explicitly not an execution/metadata fixture."""
    p = dict(program=score.PROGRAM, analysis=deepcopy(score.ANALYSIS),
             plan_sha256='SYNTHETIC_ONLY', rows={}, units=[])
    for corpus, classes in (('cremad',6),('subesco',7),('ravdess',8)):
        rows, report = {}, {}
        for group in score.GROUPS:
            report[group] = []
            # Uneven outer class counts exercise macro recall, not accuracy.
            labels = np.repeat(np.arange(classes), [1+c%3 for c in range(classes)] if group=='outer' else 2)
            for i, label in enumerate(labels):
                path = f'SYNTHETIC/{corpus}/{group}/{i}.wav'
                rows[path] = dict(label_index=int(label)); report[group].append(path)
        p['rows'][corpus] = rows
        for draw in range(24):
            for fold in range(5):
                for arm in ('A','B') if corpus=='cremad' and fold==0 else ('A',):
                    p['units'].append(dict(unit_id=f'{corpus}-{draw}-{fold}-{arm}', phase='formal',
                        corpus=corpus, draw=draw, fold=fold, arm=arm, n_classes=classes,
                        pair_id=f'{corpus}-{draw}-{fold}', seeds={'SYNTHETIC':draw*5+fold},
                        report=deepcopy(report), report_batches={g:[report[g]] for g in report},
                        prediction_epochs=list(range(1,16)) if arm=='A' else [15], selection_enabled=arm=='A'))
    # The analysis must never ask a loader for a pilot, even when it is present.
    pilot = deepcopy(p['units'][0]); pilot['phase']='pilot'; pilot['unit_id']='DO_NOT_READ_PILOT'
    p['units'].append(pilot)
    return p


def synthetic_arrays(unit, plan):
    arrays = {'epochs':np.asarray(unit['prediction_epochs'], dtype=np.int64)}
    classes = unit['n_classes']
    for group in score.GROUPS:
        paths = unit['report'][group]
        labels = np.asarray([plan['rows'][unit['corpus']][p]['label_index'] for p in paths], dtype=np.int64)
        values = []
        for epoch in unit['prediction_epochs']:
            if group in ('A','B'):
                peak = 2 if group=='A' else 7
                correct = len(labels) if epoch==peak else len(labels)//2
            else:
                correct = (epoch + 2*unit['draw'] + unit['fold']) % (len(labels)+1)
            if unit['arm']=='B':
                correct = len(labels)//3 if group=='A' else 2*len(labels)//3
            values.append(logits_for(labels, correct, classes))
        arrays[group+'__paths'] = np.asarray(paths)
        arrays[group+'__labels'] = labels
        arrays[group+'__all_epoch_logits'] = np.stack(values)
    return arrays


class MetricTests(unittest.TestCase):
    def test_unequal_support_macro_is_not_accuracy(self):
        y = np.array([0,1,1,1], dtype=np.int64)
        z = np.array([[3,0],[3,0],[3,0],[3,0]], dtype=np.float32)
        ce, recall = score.classification(y,z)
        self.assertEqual(recall,Fraction(1,2))
        self.assertNotEqual(float(recall),.25)
        self.assertAlmostEqual(ce,math.log1p(math.exp(-3))+2.25)

    def test_equal_support_different_correct_distribution_exact_tie(self):
        y = np.array([0,0,1,1,2,2], dtype=np.int64)
        z1 = logits_for(y,3,3)
        z2 = np.zeros_like(z1); predictions=np.array([0,1,1,2,2,0])
        z2[np.arange(6),predictions]=2
        self.assertEqual(score.classification(y,z1)[1],score.classification(y,z2)[1])
        self.assertEqual(score.classification(y,z1)[1],Fraction(1,2))

    def test_invalid_label_logits_and_absent_class_rejected(self):
        valid_y=np.array([0,1]); valid_z=np.eye(2,dtype=np.float32)
        bad=[(valid_y.astype(float),valid_z), (np.array([0,2]),valid_z),
             (np.array([0,0]),valid_z), (valid_y,np.array([[1,np.nan],[0,1]])),
             (valid_y,valid_z[0]), (np.array([],dtype=int),np.empty((0,2)))]
        for y,z in bad:
            with self.subTest(y=y,z=z),self.assertRaises(ValueError):score.classification(y,z)

    def test_extreme_logits_stable(self):
        ce,recall=score.classification(np.array([0,1]),np.array([[10000,-10000],[-10000,10000]],np.float32))
        self.assertEqual(ce,0); self.assertEqual(recall,1)


class InferenceTests(unittest.TestCase):
    def test_t_uses_24_draws_df23_and_two_sided(self):
        x=np.arange(24,dtype=float)/7-1
        r=score.t_estimate(x)
        se=float(np.std(x,ddof=1))/math.sqrt(24)
        self.assertEqual(r['n_draws'],24);self.assertEqual(r['df'],23)
        self.assertAlmostEqual(r['se_pp'],se)
        self.assertAlmostEqual(r['p_two_sided'],2*student_t.sf(abs(np.mean(x)/se),23))
        self.assertAlmostEqual(r['pointwise_95_ci_pp'][1]-r['mean_pp'],student_t.ppf(.975,23)*se)
        self.assertAlmostEqual(score.t_estimate(-x)['p_two_sided'],r['p_two_sided'])
        with self.assertRaises(ValueError):score.t_estimate(x[:23])

    def test_holm_known_unsorted_example_and_missing_test_slot(self):
        actual=score.holm_six([.03,.001,.9,.015,.01,.04])
        np.testing.assert_allclose(actual,[.09,.006,.9,.06,.05,.09])
        self.assertEqual(score.holm_six([.01,None,1,1,1,1])[0],.06)
        self.assertIsNone(score.holm_six([.01,None,1,1,1,1])[1])
        with self.assertRaises(ValueError):score.holm_six([.01]*5)

    def test_zero_variance_does_not_fabricate_significance(self):
        for value in (0,2,2.1):
            r=score.t_estimate([value]*24)
            self.assertEqual(r['mean_pp'],value)
            self.assertIsNone(r['p_two_sided']);self.assertIsNone(r['pointwise_95_ci_pp'])


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=synthetic_plan();cls.unit=cls.plan['units'][0]

    def measured(self,arrays,unit=None):
        u=unit or self.unit
        return score.trajectory(u,arrays,self.plan['rows'][u['corpus']])

    def test_selection_and_earliest_exact_tie(self):
        arrays=synthetic_arrays(self.unit,self.plan)
        for g in ('A','B'):
            arrays[g+'__all_epoch_logits'][9]=arrays[g+'__all_epoch_logits'][1 if g=='A' else 6]
        selected=self.measured(arrays)['selected']
        self.assertEqual(selected,dict(last=15,seen_ce=2,unseen_ce=7,seen_uar=2,unseen_uar=7))

    def test_ce_and_uar_can_choose_different_checkpoints(self):
        arrays=synthetic_arrays(self.unit,self.plan)
        y=arrays['A__labels'];n=len(y);c=self.unit['n_classes']
        for i in range(15):arrays['A__all_epoch_logits'][i]=logits_for(y,0,c,10)
        arrays['A__all_epoch_logits'][0]=logits_for(y,n,c,.01)
        arrays['A__all_epoch_logits'][1]=logits_for(y,n-1,c,4)
        selection=self.measured(arrays)['selected']
        self.assertEqual(selection['seen_uar'],1)
        self.assertEqual(selection['seen_ce'],2)

    def test_outer_scores_cannot_change_selection(self):
        arrays=synthetic_arrays(self.unit,self.plan)
        selected=self.measured(arrays)['selected']
        arrays['outer__all_epoch_logits']=arrays['outer__all_epoch_logits'][::-1].copy()
        self.assertEqual(self.measured(arrays)['selected'],selected)

    def test_b_single_last_epoch_does_not_index_fifteen(self):
        b=next(u for u in self.plan['units'] if u['arm']=='B')
        got=self.measured(synthetic_arrays(b,self.plan),b)
        self.assertEqual(got['epochs'],[15]);self.assertEqual(got['selected'],{'last':15})

    def test_path_label_epoch_and_committed_selection_mutation_rejected(self):
        mutations=[lambda a:a['epochs'].__setitem__(0,0),
                   lambda a:a['A__paths'].__setitem__(0,'BAD'),
                   lambda a:a['A__labels'].__setitem__(0,1),
                   lambda a:a.__setitem__('stored_selected_epochs',{'last':15}),
                   lambda a:a.__setitem__('outer__all_epoch_logits',a['outer__all_epoch_logits'].astype(np.float64))]
        for mutation in mutations:
            arrays=synthetic_arrays(self.unit,self.plan);mutation(arrays)
            with self.assertRaises(ValueError):self.measured(arrays)


class AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=synthetic_plan();cls.loaded=[]
        def loader(u):
            cls.loaded.append(u['phase']);return synthetic_arrays(u,cls.plan)
        cls.result,cls.tables=score.analyze(cls.plan,loader)

    def test_full_counts_and_pilot_never_loaded(self):
        self.assertEqual(self.loaded,['formal']*384)
        self.assertEqual({k:len(v) for k,v in self.tables.items()},
                         dict(contexts=360,draws=72,selected_epochs=1800,curves=5400,controls=24))
        self.assertEqual(len(self.result['tests']),6)
        self.assertEqual({r['estimand'] for r in self.result['tests']},{'delta_CE','J'})
        self.assertNotIn('p_two_sided',self.result['control'])

    def test_equal_fold_then_equal_draw_not_pooled_recordings(self):
        for r in self.tables['draws']:
            subset=[c for c in self.tables['contexts'] if c['corpus']==r['corpus'] and c['draw']==r['draw']]
            # Output cells have been rounded to float; the implementation keeps
            # exact recall differences through the five-fold average.
            self.assertAlmostEqual(r['delta_CE_pp'],math.fsum(c['delta_CE_pp'] for c in subset)/5,places=12)
            self.assertAlmostEqual(r['J_pp'],r['delta_CE_pp']-r['delta_UAR_pp'])
        for test in self.result['tests']:
            dd=[r[test['estimand']+'_pp'] for r in self.tables['draws'] if r['corpus']==test['corpus']]
            self.assertEqual(test['mean_pp'],math.fsum(dd)/24)

    def test_control_same_query_signed_cross_model_formula(self):
        for r in self.tables['controls']:
            self.assertAlmostEqual(r['E_pp'],.5*((r['QA_MA15']-r['QA_MB15'])+(r['QB_MB15']-r['QB_MA15'])),places=12)
        self.assertEqual(self.result['control']['E_mean_pp'],math.fsum(r['E_pp'] for r in self.tables['controls'])/24)

    def test_oracle_shortfall_nonnegative_and_not_epoch_selection(self):
        for r in self.tables['selected_epochs']:
            self.assertGreaterEqual(r['oracle_shortfall_pp'],0)
            self.assertEqual(r['oracle_shortfall_pp'],r['oracle_outer_uar']-r['outer_uar'])
            self.assertFalse(r['oracle_used_for_selection'])

    def test_exact_zero_interactions_and_fold_cancellation_stay_zero(self):
        quartets=[(Fraction(1,3),Fraction(2,3),Fraction(1,6),Fraction(1,2)),
                  (Fraction(5,6),Fraction(1,2),Fraction(2,3),Fraction(1,3))]
        old_residuals=[]
        for draw in range(24):
            a,b,c,d=[100*float(v) for v in quartets[draw%2]]
            old_residuals.append((a-b)-(c-d))
        # This fixture would produce a false strong t result if percentages
        # were separately rounded before the two paired differences.
        self.assertGreater(min(old_residuals),0)
        self.assertLess(score.t_estimate(old_residuals)['p_two_sided'],1e-6)
        def measured(unit, unused_arrays, unused_rows, cancel_folds=False):
            # Test aggregation after rational classification/epoch selection;
            # the separate metric and trajectory tests cover those stages.
            if cancel_folds:
                delta=[Fraction(1,3),Fraction(1,6),Fraction(-1,2),Fraction(1,10),Fraction(-1,10)][unit['fold']]
                a,c=Fraction(1,2)+delta/2,Fraction(1,2)+delta/2
                b,d=Fraction(1,2)-delta/2,Fraction(1,2)-delta/2
            else:
                a,b,c,d=quartets[unit['draw']%2]
            epochs=unit['prediction_epochs']
            recalls=[a,b,c,d]+[Fraction(1,2)]*11
            scores={'outer':[(1.,recalls[e-1]) for e in epochs]}
            scores['A']=[(1.,a if unit['arm']=='A' else b) for e in epochs]
            scores['B']=[(1.,c if unit['arm']=='A' else d) for e in epochs]
            selected={'last':15}
            if unit['arm']=='A':
                selected.update(seen_ce=1,unseen_ce=2,seen_uar=3,unseen_uar=4)
            return dict(unit=unit,epochs=epochs,scores=scores,selected=selected)
        with patch.object(score,'trajectory',side_effect=measured):
            result,tables=score.analyze(self.plan,lambda unit:None)
        self.assertTrue(all(r['J_pp']==0. for r in tables['contexts']))
        self.assertTrue(all(r['J_pp']==0. for r in tables['draws']))
        self.assertTrue(all(r['E_pp']==0. for r in tables['controls']))
        for test in result['tests']:
            if test['estimand']=='J':
                self.assertEqual(test['mean_pp'],0.)
                self.assertIsNone(test['p_two_sided'])
                self.assertFalse(test['reject_familywise_05'])
        with patch.object(score,'trajectory',side_effect=lambda *args:measured(*args,cancel_folds=True)):
            cancelled,table=score.analyze(self.plan,lambda unit:None)
        self.assertTrue(all(r['delta_CE_pp']==r['delta_UAR_pp']==r['J_pp']==0. for r in table['draws']))
        self.assertTrue(all(t['p_two_sided'] is None and not t['reject_familywise_05'] for t in cancelled['tests']))

    def test_grid_incomplete_duplicate_or_native_head_rejected(self):
        for mutate in (lambda p:p['units'].pop(0),
                       lambda p:p['units'].__setitem__(1,deepcopy(p['units'][0])),
                       lambda p:p['units'][0].__setitem__('n_classes',8)):
            p=deepcopy(self.plan);mutate(p)
            with self.assertRaises(ValueError):score.check_grid(p)

    def test_pair_frame_mismatch_rejected(self):
        p=deepcopy(self.plan);b=next(u for u in p['units'] if u['phase']=='formal' and u['arm']=='B')
        b['seeds']={'DIFFERENT':1}
        with self.assertRaisesRegex(ValueError,'control pair frame'):
            score.analyze(p,lambda u:synthetic_arrays(u,p))

    def test_complete_gate_cannot_be_count_only_or_pilot(self):
        ids={u['unit_id'] for u in self.plan['units'] if u['phase']=='formal'}
        lock={'lock_sha256':'L'}
        gate={'pass':True,'phase':'formal','expected_units':384,'verified_units':384,
              'plan_sha256':self.plan['plan_sha256'],'lock_sha256':'L',
              'done_sha256':{u:'D' for u in ids},'artifacts_sha256':{u:{} for u in ids}}
        score.check_complete_gate(gate,self.plan,lock)
        for mutate in (lambda g:g.__setitem__('phase','pilot'),
                       lambda g:g['done_sha256'].pop(next(iter(ids))),
                       lambda g:g.__setitem__('verified_units',383)):
            bad=deepcopy(gate);mutate(bad)
            with self.assertRaises(ValueError):score.check_complete_gate(bad,self.plan,lock)

    def test_cli_gate_failure_prevents_analysis_and_output(self):
        from . import run
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);out=folder/'scores'
            (folder/'formal').mkdir()
            (folder/'formal'/'SOURCE_LOCK.json').write_text('{"phase":"formal"}',encoding='utf-8')
            with patch.object(run,'audit_phase',side_effect=ValueError('not complete'),create=True), \
                 patch.object(score,'analyze') as analyze:
                with self.assertRaisesRegex(ValueError,'not complete'):
                    score.score_archive(folder,folder/'formal',out)
                analyze.assert_not_called();self.assertFalse(out.exists())

    def test_output_cannot_be_any_frozen_phase_descendant(self):
        from . import run
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);phase=repo/'formal';phase.mkdir()
            forbidden=[phase,phase/'new_scores',phase/'units'/'unit0'/'analysis',
                       phase/'unused'/'..'/'nested'/'scores',repo]
            with patch.object(score,'read_json') as read,patch.object(run,'audit_phase') as audit:
                for out in forbidden:
                    with self.subTest(out=out),self.assertRaisesRegex(ValueError,'outside the frozen phase'):
                        score.score_archive(repo,phase,out)
                read.assert_not_called();audit.assert_not_called()
            self.assertEqual(list(phase.iterdir()),[])
            # A sibling is allowed to proceed to input validation, not scoring.
            with patch.object(score,'read_json',side_effect=ValueError('input sentinel')) as read:
                with self.assertRaisesRegex(ValueError,'input sentinel'):
                    score.score_archive(repo,phase,repo/'fresh_scores')
                self.assertEqual(read.call_count,1)

    def test_pilot_is_rejected_before_its_predictions_are_audited(self):
        from . import run
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);phase=folder/'pilot';phase.mkdir()
            (phase/'SOURCE_LOCK.json').write_text('{"phase":"pilot"}',encoding='utf-8')
            with patch.object(run,'audit_phase',create=True) as audit:
                with self.assertRaisesRegex(ValueError,'pilot scientific'):
                    score.score_archive(folder,phase,folder/'scores')
                audit.assert_not_called()

    def test_full_synthetic_archive_produces_hash_closed_outputs(self):
        from . import run
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);phase=repo/'formal';phase.mkdir();out=repo/'scores'
            lock={'phase':'formal','lock_sha256':'L','sources':{'SYNTHETIC_ONLY':'S'}}
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value),encoding='utf-8')
            write(phase/'SOURCE_LOCK.json',lock);write(phase/'plan_snapshot.json',self.plan)
            (phase/'ledger.jsonl').write_text('{}\n',encoding='utf-8')
            gate={'pass':True,'phase':'formal','expected_units':384,'verified_units':384,
                  'plan_sha256':self.plan['plan_sha256'],'lock_sha256':'L',
                  'done_sha256':{},'artifacts_sha256':{},'scope':'SYNTHETIC MOCK GATE, NO REAL WEIGHTS',
                  'ledger_sha256':score.file_sha(phase/'ledger.jsonl')}
            for u in self.plan['units']:
                if u['phase']!='formal':continue
                folder=phase/'units'/u['unit_id'];attempt=folder/'attempts/0001';attempt.mkdir(parents=True)
                arrays=synthetic_arrays(u,self.plan)
                np.savez_compressed(attempt/'predictions.npz',**arrays)
                selected=score.trajectory(u,arrays,self.plan['rows'][u['corpus']])['selected']
                write(attempt/'receipt.json',{'info':{'selected_epochs':selected}})
                artifacts={'attempts/0001/'+name:score.file_sha(attempt/name) for name in ('predictions.npz','receipt.json')}
                write(folder/'DONE',{'attempt':'attempts/0001','artifacts':artifacts})
                gate['done_sha256'][u['unit_id']]=score.file_sha(folder/'DONE')
                gate['artifacts_sha256'][u['unit_id']]=artifacts
            write(phase/'COMPLETE_GATE.json',gate)
            with patch.object(run,'audit_phase',return_value=gate,create=True), \
                 patch.object(run,'checked_plan',return_value=(self.plan,lock)), \
                 patch.object(run,'sources',return_value=lock['sources']):
                result=score.score_archive(repo,phase,out)
                with self.assertRaisesRegex(ValueError,'fresh'):
                    score.score_archive(repo,phase,out)
            saved=score.read_json(out/'results.json')
            self.assertEqual(saved['result_sha256'],score.digest({k:v for k,v in saved.items() if k!='result_sha256'}))
            self.assertEqual(saved['complete_gate_snapshot']['sha256'],score.file_sha(out/'complete_gate.json'))
            for name,pin in result['tables'].items():self.assertEqual(pin['sha256'],score.file_sha(out/name))
            self.assertEqual(saved['counts']['pilot_scored'],0)


if __name__=='__main__':
    unittest.main()

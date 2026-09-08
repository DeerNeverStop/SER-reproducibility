"""SYNTHETIC numerical and admission checks; no real experimental inputs."""
import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction as F

import numpy as np
import verify_supplement as audit


def sealed(value, field):
    return {**value, field: audit.digest(value)}


def plan_fixture():
    rows, units = {}, []
    for corpus, classes in audit.CORPORA.items():
        paths = {g: [f'SYNTHETIC/{corpus}/{g}/{i}.wav' for i in range(2*classes)] for g in ('A', 'B', 'outer')}
        rows[corpus] = {p: {'label_index': i % classes} for role in paths.values() for i, p in enumerate(role)}
        for draw in range(24):
            for fold in range(5):
                for model in audit.MODELS:
                    n = 45 if model == 'wavlm_base_plus' and corpus != 'cremad' else 15
                    unit = dict(corpus=corpus, draw=draw, fold=fold, model=model,
                                unit_id=f'SYNTHETIC_{corpus}_{draw}_{fold}_{model}', phase='formal', arm='A',
                                seen_group='A', unseen_group='B', windows=[15, 45] if n == 45 else [15],
                                config={'epochs': n}, n_classes=classes,
                                fit=[f'SYNTHETIC/{corpus}/fit/{i}.wav' for i in range({'cremad':576,'subesco':224,'ravdess':64}[corpus])],
                                report=paths, report_batches={g:[v] for g,v in paths.items()},
                                seeds={'head':draw*5+fold,'order':1,'crop':2,'torch_training':3},
                                group_speakers={'A':['a'],'B':['b']}, test_speakers=['outer'],
                                fit_prompts=['train'], query_prompts=['query'], permanent_checkpoint_sample=False)
                    units.append(sealed(unit, 'unit_sha256'))
    p = dict(program='SYNTHETIC_ONLY', formal_units=720, units=units, rows=rows,
             stats=dict(family_size=10, df=23, multiplicity='Holm', test='two-sided one-sample t',
                        family=[{'id':r[0]} for r in audit.family()]))
    return sealed(p, 'plan_sha256')


def classified(y, k, correct, confidence=1.):
    logits = np.zeros((len(y), k), dtype=np.float64)
    guessed = y.copy(); guessed[correct:] = (guessed[correct:] + 1) % k
    logits[np.arange(len(y)), guessed] = confidence
    return logits


def arrays_fixture(unit):
    k, n = unit['n_classes'], unit['config']['epochs']
    y = np.arange(2*k, dtype=np.int64) % k
    data = {'epochs':np.arange(1,n+1,dtype=np.int64)}
    chosen = {'15':dict(seen_ce=2, unseen_ce=3, seen_uar=1, unseen_uar=1, last=15)}
    if n == 45: chosen['45'] = dict(seen_ce=16, unseen_ce=17, seen_uar=1, unseen_uar=1, last=45)
    low = 2 + unit['draw'] % 3 + unit['fold'] % 2
    for g in ('A','B','outer'):
        if g == 'outer':
            cube = np.stack([classified(y,k,6) for _ in range(n)])
            cube[1] = classified(y,k,low)
            cube[2] = classified(y,k,4)
            if n == 45:
                cube[15] = classified(y,k,low+1)
                cube[16] = classified(y,k,4)
        else:
            cube = np.stack([classified(y,k,len(y),.1) for _ in range(n)])
            cube[1 if g == 'A' else 2] = classified(y,k,len(y),2.)
            if n == 45: cube[15 if g == 'A' else 16] = classified(y,k,len(y),4.)
        data[g+'__paths'] = np.asarray(unit['report'][g],dtype=str)
        data[g+'__labels'] = y.copy()
        data[g+'__all_epoch_logits'] = cube
    return data, chosen


class IndependentNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan_fixture(); cls.loaded = []
        def loader(unit):
            cls.loaded.append(unit['unit_id'])
            return arrays_fixture(unit)
        cls.result = audit.recompute(cls.plan, loader)

    def test_720_full_grid_and_all_six_output_tables(self):
        self.assertEqual(len(set(self.loaded)),720)
        expected = dict(units=960,draws=312,primary_tests=10,descriptive=104,
                        selected_epochs=4800,curves=18000,q_times_e=16)
        self.assertEqual({k:len(v) for k,v in self.result.items()},expected)

    def test_known_negative_primary_and_positive_window_effect(self):
        # Across five folds mean correct-count contrast is draw%3 - 1.6.
        # Across 24 draws mean is -0.6, yielding -5 pp for the 12-item CREMA role.
        rows = {r['id']:r for r in self.result['primary_tests']}
        self.assertAlmostEqual(rows['hubert_cremad_D_CE']['mean_pp'],-5.)
        self.assertAlmostEqual(rows['hubert_cremad_J']['mean_pp'],-5.)
        self.assertAlmostEqual(rows['window_subesco_L_CE']['mean_pp'],100/14)
        self.assertAlmostEqual(rows['window_ravdess_L_J']['mean_pp'],100/16)
        self.assertLess(rows['hubert_cremad_D_CE']['t_statistic'],0)
        self.assertIsNotNone(rows['hubert_cremad_D_CE']['p_two_sided'])

    def test_exact_q_identity_includes_zero_disagreement_without_fake_conditional(self):
        for row in self.result['q_times_e']:
            self.assertEqual(row['product_pp'],row['overall_effect_pp'])
            if row['rule']=='UAR':
                self.assertEqual(row['n_disagree'],0)
                self.assertIsNone(row['conditional_effect_pp'])
            else:
                self.assertEqual(row['n_disagree'],120)

    def test_missing_or_duplicate_or_pilot_rejected_before_loader(self):
        for fault in ('missing','duplicate','pilot'):
            p = dict(self.plan,units=list(self.plan['units']))
            if fault=='missing': p['units'].pop()
            elif fault=='duplicate': p['units'][-1]=p['units'][0]
            else: p['units'][0]=dict(p['units'][0],phase='technical_pilot')
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                audit.recompute(p,lambda _: self.fail('incomplete grid reached prediction loader'))

    def test_native_macro_recall_not_accuracy(self):
        y=np.asarray([0,0,1],dtype=np.int64)
        x=np.asarray([[1.,0.],[0.,1.],[0.,1.]])
        self.assertEqual(audit.class_recall(x,y,2),F(75))
        for wrong in (y.astype(float),np.asarray([0,0,0]),np.asarray([0,0,-1])):
            with self.assertRaises(ValueError): audit.class_recall(x,wrong,2)

    def test_equal_support_ties_earliest_and_outer_never_selects(self):
        unit=self.plan['units'][0]; data, expected=arrays_fixture(unit)
        observed, selected=audit.measure(unit,data,self.plan['rows'][unit['corpus']])
        self.assertEqual(selected,expected)
        rng=np.random.RandomState(991)
        data['outer__all_epoch_logits']=rng.normal(size=data['outer__all_epoch_logits'].shape)
        self.assertEqual(audit.measure(unit,data,self.plan['rows'][unit['corpus']])[1],selected)
        self.assertEqual(selected['15']['seen_uar'],1)

    def test_paths_labels_shape_nan_and_receipt_mismatch(self):
        unit=self.plan['units'][0]
        for fault in ('path','label','shape','nan'):
            data,_=arrays_fixture(unit)
            if fault=='path':data['A__paths']=data['A__paths'][::-1]
            elif fault=='label':data['B__labels'][0]=2
            elif fault=='shape':data['outer__all_epoch_logits']=data['outer__all_epoch_logits'][:-1]
            else:data['A__all_epoch_logits'][0,0,0]=np.nan
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                audit.measure(unit,data,self.plan['rows'][unit['corpus']])
        def wrong_loader(u):
            data,choices=arrays_fixture(u);choices['15']['seen_ce']=1
            return data,choices
        with self.assertRaisesRegex(ValueError,'choices'):audit.recompute(self.plan,wrong_loader)

    def test_fraction_cancellation_zero_variance_and_two_sided_symmetry(self):
        self.assertEqual(audit.exact_mean([F(1,3)]*3+[F(-1),F(0)]),0)
        for constant in (F(0),F(7,3)):
            out=audit.estimate([constant]*24,True)
            self.assertEqual(out['sd_draw_pp'],0)
            self.assertIsNone(out['p_two_sided']);self.assertIsNone(out['pointwise_95_ci_pp'])
        values=[F(i-13,7) for i in range(24)]
        positive=audit.estimate(values,True);negative=audit.estimate([-x for x in values],True)
        self.assertEqual(positive['p_two_sided'],negative['p_two_sided'])
        self.assertEqual(positive['t_statistic'],-negative['t_statistic'])

    def test_holm_retains_all_ten_and_descriptive_has_no_extra_tests(self):
        self.assertEqual(audit.adjusted_p_ten([.001,.01]+[None]*8)[:2],[.01,.09])
        with self.assertRaises(ValueError):audit.adjusted_p_ten([.01]*6)
        self.assertTrue(all(r['p_two_sided'] is None and r['t_statistic'] is None
                            and r['holm_p_ten'] is None for r in self.result['descriptive']))

    def test_paired_window_before_fold_aggregation_and_units(self):
        rows=[r for r in self.result['draws'] if r['corpus']=='subesco' and r['comparison']=='window45_minus15']
        self.assertEqual(len(rows),24)
        self.assertTrue(all(abs(r['D_CE_pp']-100/14)<1e-12 for r in rows))
        for row in self.result['descriptive']:
            expected='UAR_percent' if row['comparison']=='model_window' and row['endpoint'].startswith('T_') else 'percentage_points'
            self.assertEqual(row['unit'],expected)

    def test_incomplete_archive_refuses_before_any_npz_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'units').mkdir()
            for unit in self.plan['units']:(root/'units'/unit['unit_id']).mkdir()
            (root/'PLAN.json.gz').write_bytes(gzip.compress(audit.canonical(self.plan),mtime=0))
            value=dict(formal_units=720,units={u['unit_id']:{} for u in self.plan['units'][:-1]})
            (root/'BACKUP_COMPLETE.json').write_bytes(audit.canonical(sealed(value,'backup_complete_sha256')))
            with self.assertRaisesRegex(ValueError,'720 UID'):
                audit.audit_backup(root,root/'nonexistent-repo')

    def test_numeric_comparison_rejects_sign_ci_or_na_changes(self):
        with self.assertRaises(ValueError):audit.compare_value(-2.,2.,'signed effect',[])
        with self.assertRaises(ValueError):audit.compare_value([1.,3.],[1.,2.],'CI',[])
        with self.assertRaises(ValueError):audit.compare_value(0.,None,'undefined p',[])
        for unsafe in ('../outside','/absolute','C:/bad','a\\b'):
            with self.assertRaises(ValueError):audit.member(Path.cwd(),unsafe)

    def test_parsed_json_bytes_cannot_be_replaced_before_manifest_pin(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_') as tmp:
            path=Path(tmp)/'receipt.json';path.write_text('{"value":1}',encoding='utf-8')
            pins={};self.assertEqual(audit.read_bound(path,pins),{'value':1})
            path.write_text('{"value":2}',encoding='utf-8')
            altered={'bytes':path.stat().st_size,'sha256':audit.sha(path)}
            with self.assertRaisesRegex(ValueError,'earlier parsed'):
                audit.pin(path,altered,pins)
            with self.assertRaisesRegex(ValueError,'between binding'):
                audit.read_bound(path,pins)

    def test_six_csv_and_json_contract_roundtrip_then_reject_changed_signed_effect(self):
        # Interface test only: these fake score files are generated from the
        # SYNTHETIC oracle above, not presented as a second scientific pipeline.
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_') as tmp:
            root=Path(tmp);scores=root/'scores';backup=root/'backup'
            scores.mkdir();backup.mkdir();pins={}
            for unit in self.plan['units']:
                for name in ('predictions.npz','history.json','receipt.json','COMPLETE.json'):
                    pins[str(audit.member(backup,'units/'+unit['unit_id']+'/'+name))]='1'*64
            (backup/'SOURCE_LOCK.json').write_bytes(audit.canonical({'source_lock_sha256':'2'*64}))
            input_audit=dict(plan_sha256=self.plan['plan_sha256'],source_lock_sha256='2'*64,
                             expected_units=720,verified_units=720,file_sha256=pins.copy())
            (scores/'input_audit.json').write_bytes(audit.canonical(input_audit))
            table_records={}
            for table in audit.TABLES:
                path=scores/(table+'.csv'); rows=self.result[table]
                with path.open('w',encoding='utf-8',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({k:'NA' if v is None else json.dumps(v) if isinstance(v,list) else v
                                         for k,v in row.items()})
                table_records[path.name]={'bytes':path.stat().st_size,'sha256':audit.sha(path)}
            result=dict(plan_sha256=self.plan['plan_sha256'],tests=self.result['primary_tests'],
                        descriptive=self.result['descriptive'],tables=table_records,
                        counts=dict(formal_units=720,unit_windows=960,full_epoch_curve_rows=18000,
                                    main_tests=10,draws_per_test=24,folds_per_draw=5,pilots_included=0),
                        inputs={'input_audit_sha256':audit.sha(scores/'input_audit.json')})
            def commit_fake_result():
                (scores/'results.json').write_bytes(audit.canonical(sealed(result,'result_sha256')))
                records={p.name:{'bytes':p.stat().st_size,'sha256':audit.sha(p)}
                         for p in scores.iterdir() if p.name!='OUTPUT_MANIFEST.json'}
                manifest=sealed(dict(plan_sha256=self.plan['plan_sha256'],files=records),'manifest_sha256')
                (scores/'OUTPUT_MANIFEST.json').write_bytes(audit.canonical(manifest))
            commit_fake_result()
            compared=audit.compare_scores(self.result,scores,self.plan,backup,pins.copy())
            self.assertGreater(compared['compared_scalar_numbers'],100000)
            self.assertEqual(compared['max_abs_difference'],0.)
            result['tests']=copy.deepcopy(result['tests'])
            result['tests'][0]['mean_pp'] *= -1
            commit_fake_result()
            with self.assertRaisesRegex(ValueError,'numeric mismatch'):
                audit.compare_scores(self.result,scores,self.plan,backup,pins.copy())


if __name__=='__main__':unittest.main()

"""SYNTHETIC-only score and archive-admission tests; no model training/data."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import score


def synthetic_plan():
    rows, panels, units = {}, {}, []
    for corpus, classes in score.CLASSES.items():
        report = {g: [f'{corpus}/{g}/{i}.wav' for i in range(2 * classes)] for g in score.GROUPS}
        rows[corpus] = {p: {'label_index': i % classes}
                        for paths in report.values() for i, p in enumerate(paths)}
        panels[corpus] = dict(fit=[f'{corpus}/fit.wav'], report=report,
                              report_batches={g: [paths] for g, paths in report.items()},
                              seeds=dict(head=1, order=2, crop=3, torch_training=4),
                              n_classes=classes, group_speakers={'A': ['a'], 'B': ['b']},
                              test_speakers=['outer'], fit_prompts=['fit'], query_prompts=['query'])
    for corpus in score.CORPORA:
        for draw in range(24):
            for fold in range(5):
                for model in score.MODELS:
                    total = 45 if corpus != 'cremad' and model == 'wavlm_base_plus' else 15
                    unit = dict(panels[corpus], corpus=corpus, draw=draw, fold=fold, model=model,
                                unit_id=f'SYNTHETIC_{corpus}_{draw}_{fold}_{model}',
                                phase='formal', arm='A', seen_group='A', unseen_group='B',
                                config={'epochs': total}, windows=[15] if total == 15 else [15, 45],
                                reference_panel_sha256=score.digest([corpus, draw, fold]),
                                permanent_checkpoint_sample=False)
                    unit['unit_sha256'] = score.digest(unit)
                    units.append(unit)
    return dict(program='SYNTHETIC_TEST_ONLY', units=units, rows=rows, formal_units=720,
                pilots=[{'unit_id': 'SYNTHETIC_PILOT_MUST_NOT_BE_LOADED'}],
                stats=dict(family=score.expected_family(), family_size=10, test='two-sided one-sample t',
                           multiplicity='Holm', alpha=.05, df=23))


def logits(labels, classes, correct, confidence=1.):
    pred = labels.copy()
    pred[correct:] = (pred[correct:] + 1) % classes
    result = np.zeros((len(labels), classes), dtype=np.float64)
    result[np.arange(len(labels)), pred] = confidence
    return result


def expected_selection(unit):
    result = {'15': dict(seen_ce=2, unseen_ce=3, seen_uar=1, unseen_uar=1, last=15)}
    if 45 in unit['windows']:
        result['45'] = dict(seen_ce=16, unseen_ce=17, seen_uar=1, unseen_uar=1, last=45)
    return result


def synthetic_predictions(unit):
    k, total = unit['n_classes'], unit['config']['epochs']
    y = np.arange(2 * k, dtype=np.int64) % k
    n = len(y)
    result = {'epochs': np.arange(1, total + 1, dtype=np.int64)}
    for group in score.GROUPS:
        if group != 'outer':
            a = np.stack([logits(y, k, n, .2) for _ in range(total)])
            a[1 if group == 'A' else 2] = logits(y, k, n, 3.)
            if total == 45:
                a[15 if group == 'A' else 16] = logits(y, k, n, 5.)
        else:
            low = n // 3
            high = low + unit['draw'] % 3 + 1 + unit['fold'] % 2
            high += int(unit['model'] == 'hubert_base')
            a = np.stack([logits(y, k, n // 2) for _ in range(total)])
            a[1], a[2] = logits(y, k, high), logits(y, k, low)
            if total == 45:
                a[15] = logits(y, k, high + unit['draw'] % 2 + 1)
                a[16] = logits(y, k, low)
        result[f'{group}__paths'] = np.asarray(unit['report'][group], dtype=str)
        result[f'{group}__labels'] = y.copy()
        result[f'{group}__all_epoch_logits'] = a
    return result


class PureScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = synthetic_plan()
        cls.loaded = []

        def loader(unit):
            cls.loaded.append(unit['unit_id'])
            return synthetic_predictions(unit), expected_selection(unit)

        cls.result, cls.tables = score.analyze(cls.plan, loader)

    def test_full_grid_and_no_pilot_reads(self):
        self.assertEqual(len(set(self.loaded)), 720)
        self.assertNotIn('SYNTHETIC_PILOT_MUST_NOT_BE_LOADED', self.loaded)
        self.assertEqual(self.result['counts']['main_tests'], 10)
        self.assertTrue(all(r['unit'] == 'percentage_points' for r in self.result['tests']))
        for row in self.result['descriptive']:
            expected = ('UAR_percent' if row['comparison'] == 'model_window'
                        and row['endpoint'].startswith('T_') else 'percentage_points')
            self.assertEqual(row['unit'], expected)
        self.assertEqual(len(self.tables['units']), 960)
        self.assertEqual(len(self.tables['curves']), 18000)
        self.assertEqual(len(self.tables['selected_epochs']), 4800)
        self.assertEqual(len(self.tables['draws']), 312)

    def test_known_whole_role_and_equal_fold_means(self):
        main = {r['id']: r for r in self.result['tests']}
        # 24 draws have mean draw%3 = 1; five folds have mean fold%2 = .4.
        # HuBERT adds one correct item: mean difference is 3.4 of 12/14/16.
        self.assertAlmostEqual(main['hubert_cremad_D_CE']['mean_pp'], float(Fraction(340, 12)))
        self.assertAlmostEqual(main['hubert_subesco_J']['mean_pp'], float(Fraction(340, 14)))
        # The long window adds 1 + draw%2 = 1.5 correct predictions on average.
        self.assertAlmostEqual(main['window_subesco_L_CE']['mean_pp'], float(Fraction(150, 14)))
        self.assertAlmostEqual(main['window_ravdess_L_J']['mean_pp'], float(Fraction(150, 16)))
        self.assertTrue(all(r['n_draws'] == 24 and r['df'] == 23 for r in main.values()))

    def test_paired_backbone_differences_have_no_p_values(self):
        rows = [r for r in self.result['descriptive']
                if r['comparison'] == 'hubert_minus_wavlm15' and r['endpoint'] == 'D_CE']
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertAlmostEqual(row['mean_pp'], 100 / (2 * score.CLASSES[row['corpus']]))
            self.assertIsNone(row['p_two_sided'])
            self.assertIsNone(row['holm_p_ten'])
            self.assertFalse(row['hypothesis_test'])
            self.assertEqual(row['status'], 'undefined_t_zero_sample_variance')

    def test_missing_duplicate_or_pilot_units_fail_before_loading(self):
        for mutation in ('missing', 'duplicate_context', 'pilot'):
            with self.subTest(mutation=mutation):
                plan = dict(self.plan, units=list(self.plan['units']))
                if mutation == 'missing':
                    plan['units'].pop()
                else:
                    u = copy.deepcopy(plan['units'][0])
                    if mutation == 'duplicate_context':
                        u['unit_id'] += '_duplicate'
                        plan['units'][2] = u
                    else:
                        u['phase'] = 'technical_pilot'
                        plan['units'][0] = u
                with self.assertRaises(ValueError):
                    score.analyze(plan, lambda _: self.fail('loader called before grid admission'))

    def test_changed_test_panel_or_family_rejected(self):
        plan = dict(self.plan, units=list(self.plan['units']))
        u = copy.deepcopy(plan['units'][1])
        u['report']['outer'][0] = 'SYNTHETIC_different.wav'
        u['report_batches']['outer'] = [u['report']['outer']]
        plan['units'][1] = u
        with self.assertRaisesRegex(ValueError, 'panel'):
            score.validate_grid(plan)
        plan = dict(self.plan, stats=dict(self.plan['stats'], family_size=6))
        with self.assertRaisesRegex(ValueError, 'ten-test'):
            score.validate_grid(plan)

    def test_full_window_selection_checks_receipt(self):
        unit = next(u for u in self.plan['units'] if u['config']['epochs'] == 45)
        data = synthetic_predictions(unit)
        _, choice = score.measure_unit(unit, data, self.plan['rows'][unit['corpus']])
        self.assertEqual(choice, expected_selection(unit))
        self.assertEqual(choice['15']['seen_ce'], 2)
        self.assertEqual(choice['45']['seen_ce'], 16)
        first = self.plan['units'][0]
        wrong = expected_selection(first)
        wrong['15']['seen_ce'] = 1
        with self.assertRaisesRegex(ValueError, 'receipt checkpoint'):
            score.analyze(self.plan, lambda u: (synthetic_predictions(u), wrong))

    def test_outer_outcomes_cannot_change_checkpoint_selection(self):
        unit = self.plan['units'][0]
        data = synthetic_predictions(unit)
        _, before = score.measure_unit(unit, data, self.plan['rows'][unit['corpus']])
        rng = np.random.RandomState(123)
        data['outer__all_epoch_logits'] = rng.normal(size=data['outer__all_epoch_logits'].shape)
        _, after = score.measure_unit(unit, data, self.plan['rows'][unit['corpus']])
        self.assertEqual(before, after)

    def test_path_labels_epoch_shape_and_nonfinite_rejected(self):
        unit = self.plan['units'][0]
        for kind in ('paths', 'labels', 'epochs', 'shape', 'nan'):
            with self.subTest(kind=kind):
                data = synthetic_predictions(unit)
                if kind == 'paths': data['outer__paths'] = data['outer__paths'][::-1]
                if kind == 'labels': data['A__labels'][0] = 1
                if kind == 'epochs': data['epochs'][-1] = 14
                if kind == 'shape': data['B__all_epoch_logits'] = data['B__all_epoch_logits'][:-1]
                if kind == 'nan': data['A__all_epoch_logits'][0, 0, 0] = np.nan
                with self.assertRaises(ValueError):
                    score.measure_unit(unit, data, self.plan['rows'][unit['corpus']])

    def test_macro_recall_not_accuracy_and_bad_labels(self):
        y = np.asarray([0, 0, 1], dtype=np.int64)
        x = np.asarray([[2., 0.], [0., 2.], [0., 2.]])
        self.assertEqual(score.uar(x, y, 2), Fraction(75))
        for bad in (y.astype(float), np.asarray([0, 0, -1]), np.asarray([0, 0, 0])):
            with self.assertRaises(ValueError): score.uar(x, bad, 2)

    def test_exact_equal_support_tie_and_earliest_epoch(self):
        unit = self.plan['units'][0]
        data = synthetic_predictions(unit)
        y, k = data['A__labels'], unit['n_classes']
        one = logits(y, k, len(y) - 1)
        two = logits(y, k, len(y) - 1)
        # Same number correct but a different class contains the mistake.
        two[0], two[-1] = two[-1].copy(), two[0].copy()
        # Explicitly make exactly one wrong prediction at index 0 instead.
        two = logits(y, k, len(y))
        two[0] = 0.; two[0, (int(y[0]) + 1) % k] = 1.
        self.assertEqual(score.uar(one, y, k), score.uar(two, y, k))
        data['A__all_epoch_logits'][:] = logits(y, k, len(y) - 2)
        data['A__all_epoch_logits'][0] = one
        data['A__all_epoch_logits'][1] = two
        _, choice = score.measure_unit(unit, data, self.plan['rows'][unit['corpus']])
        self.assertEqual(choice['15']['seen_uar'], 1)

    def test_fraction_cancellation_and_constant_t_are_undefined(self):
        exact = score.mean_exact([Fraction(1, 3)] * 3 + [Fraction(-1), Fraction(0)])
        self.assertEqual(exact, 0)
        for value in (exact, Fraction(7, 3)):
            row = score.t_estimate([value] * 24, inference=True)
            self.assertEqual(row['sd_draw_pp'], 0)
            self.assertIsNone(row['p_two_sided'])
            self.assertIsNone(row['pointwise_95_ci_pp'])

    def test_holm_fixed_ten_and_signed_two_sided(self):
        values = [Fraction(i - 4, 7) for i in range(24)]
        pos, neg = (score.t_estimate(v, inference=True)
                    for v in (values, [-x for x in values]))
        self.assertEqual(pos['p_two_sided'], neg['p_two_sided'])
        self.assertEqual(score.holm_ten([.001, .01] + [None] * 8)[:2], [.01, .09])
        self.assertTrue(all(v is None for v in score.holm_ten([None] * 10)))
        with self.assertRaises(ValueError): score.holm_ten([.01] * 6)


def write_json(path, obj):
    path.write_bytes(score.canonical(obj) + b'\n')


def descriptor(path):
    return dict(bytes=path.stat().st_size, sha256=score.file_sha(path))


class FileGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.unit = synthetic_plan()['units'][0]
        self.plan_sha, self.lock_sha = 'a' * 64, 'b' * 64
        self.unit_dir = self.root / 'unit'
        self.unit_dir.mkdir()
        (self.unit_dir / 'predictions.npz').write_bytes(b'SYNTHETIC byte-gate fixture, not real predictions')
        (self.unit_dir / 'checkpoint.pt').write_bytes(b'SYNTHETIC checkpoint byte fixture, never torch-loaded')
        history = [dict(epoch=e, train_loss=1., optimizer_steps=1, scaler_skipped_steps=0,
                        seen_ce=1., unseen_ce=1., seen_uar=.5, unseen_uar=.5) for e in range(1, 16)]
        write_json(self.unit_dir / 'history.json', history)
        self.identity = dict(unit_id=self.unit['unit_id'], unit_sha256=self.unit['unit_sha256'],
                             plan_sha256=self.plan_sha, source_lock_sha256=self.lock_sha)
        files = {n: descriptor(self.unit_dir / n) for n in ('predictions.npz', 'history.json', 'checkpoint.pt')}
        selections = expected_selection(self.unit)
        union = sorted({e for r in selections.values() for e in r.values()})
        checks = {f'{e}/{g}': 0. for e in union for g in score.GROUPS}
        info = dict(epochs=15, windows=[15], selected_epochs=selections,
                    unit_sha256=self.unit['unit_sha256'],
                    checkpoint_reload_verified=True, reload_max_abs_diff=0., outer_scores_computed=False,
                    reload_by_epoch_group=checks, reload_checks=len(checks),
                    checkpoint_unique_epochs=union, checkpoint_unique_epoch_count=len(union),
                    checkpoint_sha256=files['checkpoint.pt']['sha256'],
                    reload_checked_values=len(union) * self.unit['n_classes']
                        * sum(len(v) for v in self.unit['report'].values()))
        write_json(self.unit_dir / 'receipt.json', dict(self.identity, phase='formal',
                                                       environment_sha256='e' * 64, files=files, info=info))
        write_json(self.unit_dir / 'COMPLETE.json',
                   dict(self.identity, receipt_sha256=score.file_sha(self.unit_dir / 'receipt.json')))

    def verify(self):
        return score.verify_unit_files(self.unit, self.unit_dir, self.plan_sha, self.lock_sha)

    def make_sample(self):
        self.unit.pop('unit_sha256')
        self.unit['permanent_checkpoint_sample'] = True
        self.unit['unit_sha256'] = score.digest(self.unit)
        self.identity['unit_sha256'] = self.unit['unit_sha256']
        receipt = score.read_json(self.unit_dir / 'receipt.json')
        receipt.update(self.identity)
        receipt['info']['unit_sha256'] = self.unit['unit_sha256']
        receipt['info']['fresh_process_restore_verified'] = True
        fresh = dict(self.identity, schema='ser-autodl-fresh-restore-1', passed=True,
                     environment_sha256=receipt['environment_sha256'],
                     checkpoint_sha256=receipt['files']['checkpoint.pt']['sha256'],
                     predictions_sha256=receipt['files']['predictions.npz']['sha256'],
                     comparisons=receipt['info']['reload_by_epoch_group'], max_abs_diff=0.)
        write_json(self.unit_dir / 'fresh_restore.json', fresh)
        receipt['files']['fresh_restore.json'] = descriptor(self.unit_dir / 'fresh_restore.json')
        write_json(self.unit_dir / 'receipt.json', receipt)
        write_json(self.unit_dir / 'COMPLETE.json',
                   dict(self.identity, receipt_sha256=score.file_sha(self.unit_dir / 'receipt.json')))

    def release(self):
        receipt_sha = score.file_sha(self.unit_dir / 'receipt.json')
        ack = dict(self.identity, receipt_sha256=receipt_sha,
                   files={n: descriptor(self.unit_dir / n)
                          for n in ('predictions.npz', 'history.json', 'receipt.json', 'COMPLETE.json')},
                   backup_location='off-instance-local-verified', checkpoint_in_backup=False)
        ack['backup_ack_sha256'] = score.digest(ack)
        write_json(self.unit_dir / 'backup_ack.json', ack)
        release = dict(self.identity, schema='ser-checkpoint-release-1', receipt_sha256=receipt_sha,
                       checkpoint_sha256=score.file_sha(self.unit_dir / 'checkpoint.pt'),
                       backup_ack_sha256=ack['backup_ack_sha256'],
                       checkpoint_reload_verified=True, checkpoint_retained=False)
        release['release_sha256'] = score.digest(release)
        write_json(self.unit_dir / 'checkpoint_release.json', release)
        (self.unit_dir / 'checkpoint.pt').unlink()

    def test_present_and_correctly_released_checkpoint_pass_byte_gate(self):
        _, audit, _ = self.verify()
        self.assertEqual(audit['retention'], 'checkpoint_present_and_hashed')
        self.release()
        _, audit, _ = self.verify()
        self.assertIn('bound_backup_ack', audit['retention'])

    def test_missing_checkpoint_without_release_and_mutated_payload_fail(self):
        (self.unit_dir / 'checkpoint.pt').unlink()
        with self.assertRaises(ValueError): self.verify()
        (self.unit_dir / 'predictions.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.verify()

    def test_resealed_release_cannot_substitute_another_checkpoint(self):
        self.release()
        path = self.unit_dir / 'checkpoint_release.json'
        release = score.read_json(path)
        release.pop('release_sha256'); release['checkpoint_sha256'] = 'f' * 64
        release['release_sha256'] = score.digest(release)
        write_json(path, release)
        with self.assertRaises(ValueError): self.verify()

    def test_permanent_sample_cannot_be_pruned(self):
        self.make_sample()
        self.verify()
        self.release()
        with self.assertRaisesRegex(ValueError, 'permanent'):
            self.verify()

    def test_sample_fresh_restore_must_cover_all_states_and_inputs(self):
        self.make_sample()
        original = score.read_json(self.unit_dir / 'fresh_restore.json')
        for key in ('comparisons', 'predictions_sha256', 'environment_sha256', 'passed'):
            with self.subTest(key=key):
                fresh = copy.deepcopy(original)
                if key == 'comparisons': fresh[key].pop(next(iter(fresh[key])))
                elif key == 'passed': fresh[key] = False
                else: fresh[key] = '0' * 64
                write_json(self.unit_dir / 'fresh_restore.json', fresh)
                receipt = score.read_json(self.unit_dir / 'receipt.json')
                receipt['files']['fresh_restore.json'] = descriptor(self.unit_dir / 'fresh_restore.json')
                write_json(self.unit_dir / 'receipt.json', receipt)
                write_json(self.unit_dir / 'COMPLETE.json',
                           dict(self.identity, receipt_sha256=score.file_sha(self.unit_dir / 'receipt.json')))
                with self.assertRaisesRegex(ValueError, 'fresh-process'):
                    self.verify()

    def test_archived_cloud_runtime_paths_are_hashed_not_resolved(self):
        environment = dict(schema='SYNTHETIC', backend='autodl')
        environment['environment_sha256'] = score.digest(environment)
        plan = dict(program='SYNTHETIC')
        plan['plan_sha256'] = score.digest(plan)
        models = {m: dict(name=m, file_sha256='1' * 64, state_sha256='2' * 64) for m in score.MODELS}
        runtime = dict(models={m: dict(spec, path='/nonexistent/cloud/' + m + '.pth')
                               for m, spec in models.items()}, audio_roots={'SYNTHETIC': '/no/local/audio'})
        write_json(self.root / 'ENVIRONMENT.json', environment)
        write_json(self.root / 'PLAN.json.gz', plan)
        write_json(self.root / 'RUNTIME_PATHS.local.json', runtime)
        lock = dict(models=models, environment_sha256=environment['environment_sha256'],
                    runtime_paths_sha256=score.file_sha(self.root / 'RUNTIME_PATHS.local.json'))
        self.assertEqual(len(score.verify_runtime_records(self.root, plan, lock)), 3)
        runtime['audio_roots']['SYNTHETIC'] = '/changed'
        write_json(self.root / 'RUNTIME_PATHS.local.json', runtime)
        with self.assertRaisesRegex(ValueError, 'runtime'):
            score.verify_runtime_records(self.root, plan, lock)

    def test_false_reload_and_changed_complete_fail(self):
        receipt = score.read_json(self.unit_dir / 'receipt.json')
        receipt['info']['checkpoint_reload_verified'] = False
        write_json(self.unit_dir / 'receipt.json', receipt)
        write_json(self.unit_dir / 'COMPLETE.json',
                   dict(self.identity, receipt_sha256=score.file_sha(self.unit_dir / 'receipt.json')))
        with self.assertRaisesRegex(ValueError, 'reload'):
            self.verify()

    def test_output_and_member_boundaries(self):
        for path in ('../outside', '/absolute', 'a\\b', 'C:/a', 'a/./b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                score.safe_member(self.root, path)
        phase = self.root / 'formal'; phase.mkdir()
        plan = self.root / 'plans' / 'plan.json'
        with self.assertRaises(ValueError): score.output_path(phase / 'scores', plan, phase)
        with self.assertRaises(ValueError): score.output_path(self.root, plan, phase)
        allowed = score.output_path(self.root / 'scores', plan, phase)
        self.assertEqual(allowed, (self.root / 'scores').resolve())


if __name__ == '__main__':
    unittest.main()

"""Small synthetic checks of selection and completeness; no real model use."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('windows', Path(__file__).with_name('reconstruct_windows.py'))
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


def rows():
    return [dict(unit_id='fpc_formal_ravdess_d00_f0_A', corpus='ravdess', draw=0, fold=0,
                 epoch=e, seen_ce=1 + e / 100, unseen_ce=1 + e / 100,
                 seen_uar=50.0, unseen_uar=50.0, seen_uar_count=32, unseen_uar_count=32,
                 outer_uar=float(e)) for e in range(1, 16)]


class SelectionTests(unittest.TestCase):
    def test_outer_is_never_used_for_selection_and_ties_are_earliest(self):
        data = rows()
        data[0]['outer_uar'] = 0.0
        data[7]['outer_uar'] = 100.0
        chosen = W.select_window(data, 8)
        self.assertTrue(all(chosen[r + '_epoch'] == 1 for r in W.RULES))
        self.assertEqual(chosen['D_CE_pp'], 0)
        self.assertEqual(chosen['D_UAR_pp'], 0)
        self.assertEqual(chosen['J_pp'], 0)

    def test_small_ce_difference_is_not_merged_into_a_tie(self):
        data = rows()
        data[0]['seen_ce'] = 1.0
        data[1]['seen_ce'] = 1.0 - 1e-14
        self.assertEqual(W.select_window(data, 8)['seen_ce_epoch'], 2)

    def test_window_excludes_better_later_validation(self):
        data = rows()
        data[14]['seen_ce'] = 0.01
        self.assertEqual(W.select_window(data, 12)['seen_ce_epoch'], 1)
        self.assertEqual(W.select_window(data, 15)['seen_ce_epoch'], 15)

    def test_disagreement_can_decrease_when_window_grows(self):
        data = rows()
        data[1]['unseen_ce'] = 0.8
        data[9]['seen_ce'] = data[9]['unseen_ce'] = 0.1
        self.assertEqual(W.select_window(data, 8)['ce_disagreement'], 1)
        self.assertEqual(W.select_window(data, 10)['ce_disagreement'], 0)

    def test_validation_lattice_does_not_round_arbitrary_scores(self):
        self.assertEqual(W.validation_count(100 * 37 / 64, 'ravdess'), 37)
        with self.assertRaisesRegex(ValueError, 'lattice'):
            W.validation_count(57.0, 'ravdess')

    def test_missing_epoch_and_unplanned_window_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            W.select_window(rows()[1:], 8)
        with self.assertRaisesRegex(ValueError, 'fixed diagnostic'):
            W.select_window(rows(), 14)

    def test_fold_first_complete_reconstruction(self):
        groups = {}
        for c in W.CORPORA:
            for d in range(24):
                for f in range(5):
                    uid = f'fpc_formal_{c}_d{d:02d}_f{f}_A'
                    data = copy.deepcopy(rows())
                    for r in data:
                        r.update(unit_id=uid, corpus=c, draw=d, fold=f)
                    data[1]['unseen_ce'] = 0.2
                    data[0]['outer_uar'] = float(f + d)
                    data[1]['outer_uar'] = 0.0
                    groups[uid] = data
        contexts, draws, summary = W.reconstruct(groups)
        self.assertEqual((len(contexts), len(draws), len(summary)), (1440, 288, 12))
        self.assertEqual(draws[0]['D_CE_pp'], 2.0)
        self.assertEqual(summary[0]['D_CE_pp'], 13.5)
        self.assertAlmostEqual(summary[0]['ce_disagreement_decomposition']['identity_abs_error_pp'], 0)

    def test_partial_or_pilot_curve_grid_rejected(self):
        raw = [{k: str(v) for k, v in r.items() if k in W.CURVE_FIELDS} for r in rows()]
        with self.assertRaisesRegex(ValueError, '360-main'):
            W.parse_curves(raw)
        raw[0]['unit_id'] = raw[0]['unit_id'].replace('formal', 'pilot')
        with self.assertRaisesRegex(ValueError, 'UID'):
            W.parse_curves(raw)


if __name__ == '__main__':
    unittest.main()

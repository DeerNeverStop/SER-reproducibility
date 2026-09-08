"""Check paired metadata design and reject scientific-role corruption."""
from copy import deepcopy
from pathlib import Path
import unittest

from v3.data_design.core_plan import read_metadata
from .role_swap_plan import build, validate


class RoleSwapPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = read_metadata(Path(__file__).resolve().parents[3])
        cls.plan = build(cls.meta)

    def test_full_grid_replays_without_predictions(self):
        self.assertEqual(self.plan, build(self.meta))
        self.assertEqual(len(self.plan['pairs']) * 2, 240)
        self.assertEqual(self.plan['execution_requirements']['formal_fits'], 240)

    def test_swapping_arm_names_keeps_batch_content_but_violates_frozen_assignment(self):
        plan = deepcopy(self.plan)
        p = plan['pairs'][0]
        for slot in p['fit_slots']:
            slot['A'], slot['B'] = slot['B'], slot['A']
        for field in ('group_speakers', 'report'):
            p[field]['A'], p[field]['B'] = p[field]['B'], p[field]['A']
        with self.assertRaisesRegex(ValueError, 'design replay differs'):
            validate(plan, self.meta)
        self.assertEqual(p['shared_report_batches'], self.plan['pairs'][0]['shared_report_batches'])

    def test_training_report_overlap_is_rejected(self):
        plan = deepcopy(self.plan)
        p = plan['pairs'][0]
        p['report']['A'][0] = p['fit_slots'][0]['A']
        with self.assertRaises(ValueError):
            validate(plan, self.meta)

    def test_wrong_arm_or_slot_label_is_rejected(self):
        plan = deepcopy(self.plan)
        p = plan['pairs'][0]
        p['fit_slots'][0]['A'] = p['fit_slots'][0]['B']
        with self.assertRaises(ValueError):
            validate(plan, self.meta)
        plan = deepcopy(self.plan)
        plan['pairs'][0]['fit_slots'][0]['label'] = 99
        with self.assertRaises(ValueError):
            validate(plan, self.meta)

    def test_batch_order_or_configuration_drift_is_rejected(self):
        plan = deepcopy(self.plan)
        batch = plan['pairs'][0]['shared_report_batches'][0]
        batch[0], batch[1] = batch[1], batch[0]
        with self.assertRaises(ValueError):
            validate(plan, self.meta)

    def test_seed_rank_endpoint_and_order_mutations_are_rejected(self):
        for mutate in (
            lambda p: p['pairs'][0]['seeds'].__setitem__('order', 0),
            lambda p: p['pairs'][0]['fit_slots'][0].__setitem__('person_rank', 99),
            lambda p: p['inference'].__setitem__('test', 'choose best p'),
            lambda p: p['execution_requirements'].__setitem__('pilot_in_primary', True),
            lambda p: p['pairs'][0]['arm_execution_order'].reverse(),
        ):
            plan = deepcopy(self.plan)
            mutate(plan)
            with self.assertRaisesRegex(ValueError, 'design replay differs'):
                validate(plan, self.meta)
        plan = deepcopy(self.plan)
        plan['config']['epochs'] = 16
        with self.assertRaises(ValueError):
            validate(plan, self.meta)


if __name__ == '__main__':
    unittest.main()

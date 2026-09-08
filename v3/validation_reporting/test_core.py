"""Synthetic numerical checks; these fixtures are not experimental evidence."""

import copy
import unittest

import numpy as np

from v3.validation_reporting.core import analyze_context, uar


def fixture():
    # Four intentionally crossing ranking profiles: A and B select differently.
    rows = [
        ((90, 40), (50, 40), 20),
        ((80, 95), (70, 30), 30),
        ((70, 60), (90, 55), 40),
        ((60, 70), (60, 95), 50),
    ]
    return {
        c: {"seen": {"A": s[0], "B": s[1]},
            "unseen": {"A": u[0], "B": u[1]}, "test": {"all": t}}
        for c, (s, u, t) in enumerate(rows)
    }


def row_for(episodes, direction, rule):
    return next(r for r in episodes if r["direction"] == direction and r["rule"] == rule)


class UARTests(unittest.TestCase):
    def test_macro_recall_not_accuracy(self):
        # Class recalls 1, 0, 1 -> 66.67%; sample accuracy is 4/5 = 80%.
        labels = np.array([0, 0, 0, 1, 2])
        logits = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0], [0, 0, 2]])
        self.assertAlmostEqual(uar(labels, logits), 200 / 3)

    def test_perfect_six_class_and_argmax_tie(self):
        self.assertEqual(uar(np.arange(6), np.eye(6)), 100.0)
        self.assertEqual(uar([0, 1], [[1, 1], [1, 1]]), 50.0)

    def test_fractional_recall_and_scale_invariance(self):
        labels = [0, 0, 1, 1, 1, 2]
        logits = np.eye(3)[[0, 1, 1, 2, 0, 2]]
        self.assertAlmostEqual(uar(labels, logits), 100 * (0.5 + 1 / 3 + 1) / 3)
        self.assertAlmostEqual(uar(labels, logits), uar(labels, 4 * logits - 7))

    def test_equal_support_preserves_mathematical_configuration_ties(self):
        labels = np.repeat(np.arange(6), 24)

        def logits_for(correct_counts):
            pred = (labels + 1) % 6
            for label, correct in enumerate(correct_counts):
                indices = np.flatnonzero(labels == label)
                pred[indices[:correct]] = label
            return np.eye(6)[pred]

        # Both have exactly 50/144 correct. Direct floating per-class means
        # formerly differed by ~1e-14 and incorrectly favored config 1.
        a = uar(labels, logits_for([11, 11, 15, 2, 10, 1]))
        b = uar(labels, logits_for([16, 21, 0, 10, 1, 2]))
        self.assertEqual(a, b)
        self.assertEqual(a, 100.0 * 50 / 144)
        scores = fixture()
        for c in scores:
            scores[c]["seen"]["A"] = 0.0
        scores[0]["seen"]["A"] = a
        scores[1]["seen"]["A"] = b
        self.assertEqual(analyze_context(0, 0, scores)[0][0]["config_index"], 0)

    def test_reject_bad_logits_and_labels(self):
        invalid = [
            ([], np.empty((0, 2))),
            ([0, 1], [0, 1]),
            ([0, 1], [[1, 0]]),
            ([0, 1], [[1], [1]]),
            ([0, 1], [[float("nan"), 0], [0, 1]]),
            ([0, 1], [[float("inf"), 0], [0, 1]]),
            ([0, 1], [[1j, 0], [0, 1]]),
            ([0, 1], [["1", "0"], ["0", "1"]]),
            ([[0], [1]], np.eye(2)),
            ([0.0, 1.0], np.eye(2)),
            ([False, True], np.eye(2)),
            ([-1, 1], np.eye(2)),
            ([0, 2], np.eye(2)),
            ([0, 0], np.eye(2)),
        ]
        for labels, logits in invalid:
            with self.subTest(labels=labels, logits=logits):
                with self.assertRaises(ValueError):
                    uar(labels, logits)


class ContextTests(unittest.TestCase):
    def test_selection_reverse_and_hand_calculated_contrasts(self):
        episodes, contrasts, controls = analyze_context(2, 4, fixture())
        self.assertEqual((len(episodes), len(contrasts), len(controls)), (4, 2, 2))
        self.assertEqual([r["config_index"] for r in episodes], [0, 2, 1, 3])
        self.assertEqual(row_for(episodes, "A_to_B", "seen"), {
            "draw": 2, "fold": 4, "direction": "A_to_B", "rule": "seen", "config_index": 0,
            "selection_uar": 90.0, "report_seen_uar": 40.0, "report_unseen_uar": 40.0,
            "test_uar": 20.0, "reuse_pp": 50.0, "report_gap_pp": 20.0,
            "apparent_gap_pp": 70.0, "report_exposure_pp": 0.0,
        })
        self.assertEqual(contrasts[0], {
            "draw": 2, "fold": 4, "direction": "A_to_B", "apparent_contrast_pp": 20.0,
            "reuse_contrast_pp": 15.0, "report_gap_contrast_pp": 5.0,
            "test_contrast_pp": -20.0, "config_agreement": 0,
        })
        self.assertEqual(contrasts[1]["apparent_contrast_pp"], 20.0)
        self.assertEqual(contrasts[1]["reuse_contrast_pp"], -20.0)
        self.assertEqual(contrasts[1]["report_gap_contrast_pp"], 40.0)

    def test_ties_choose_lowest_config_independent_of_dict_order(self):
        scores = fixture()
        for c in scores:
            for role in ("seen", "unseen"):
                scores[c][role] = {"A": 65.0, "B": 65.0}
        scores = dict(reversed(list(scores.items())))
        episodes, contrasts, _ = analyze_context(0, 0, scores)
        self.assertEqual([r["config_index"] for r in episodes], [0, 0, 0, 0])
        self.assertTrue(all(r["config_agreement"] == 1 for r in contrasts))
        scores[0]["seen"]["A"] = 64.0
        self.assertEqual(analyze_context(0, 0, scores)[0][0]["config_index"], 1)

    def test_report_and_test_values_never_change_that_direction_selection(self):
        before = analyze_context(0, 0, fixture())[0]
        for direction, selection_half, report_half in (("A_to_B", "A", "B"), ("B_to_A", "B", "A")):
            for rule in ("seen", "unseen"):
                changed = fixture()
                other_role = "unseen" if rule == "seen" else "seen"
                for c in changed:
                    changed[c][rule][report_half] = float(100 if c == 0 else 0)
                    changed[c][other_role] = {"A": float(c), "B": float(100 - c)}
                    changed[c]["test"]["all"] = float(100 - 10 * c)
                after = analyze_context(0, 0, changed)[0]
                self.assertEqual(row_for(before, direction, rule)["config_index"],
                                 row_for(after, direction, rule)["config_index"])
                self.assertEqual(row_for(before, direction, rule)["selection_uar"],
                                 row_for(after, direction, rule)["selection_uar"])

    def test_half_exchange_swaps_directions(self):
        scores = fixture()
        swapped = copy.deepcopy(scores)
        for config in swapped:
            for role in ("seen", "unseen"):
                swapped[config][role] = {"A": scores[config][role]["B"], "B": scores[config][role]["A"]}
        original = analyze_context(0, 0, scores)
        reverse = analyze_context(0, 0, swapped)
        for old_group, new_group in zip(original, reverse):
            for old in old_group:
                new_direction = "B_to_A" if old["direction"] == "A_to_B" else "A_to_B"
                new = next(r for r in new_group if r["direction"] == new_direction and r.get("rule") == old.get("rule"))
                self.assertEqual({k: v for k, v in old.items() if k != "direction"},
                                 {k: v for k, v in new.items() if k != "direction"})

    def test_positive_two_way_reuse_is_constructed_not_each_direction(self):
        scores = fixture()
        for c in scores:
            scores[c]["seen"] = {"A": 5.0, "B": 15.0}
        episodes, _, _ = analyze_context(0, 0, scores)
        self.assertEqual(row_for(episodes, "A_to_B", "seen")["reuse_pp"], -10.0)
        self.assertEqual(row_for(episodes, "B_to_A", "seen")["reuse_pp"], 10.0)
        self.assertEqual(sum(r["reuse_pp"] for r in episodes if r["rule"] == "seen"), 0.0)

    def test_fixed_identities_and_decompositions_randomized(self):
        rng = np.random.default_rng(913)
        for _ in range(200):
            scores = {c: {"seen": {"A": rng.uniform(0, 100), "B": rng.uniform(0, 100)},
                          "unseen": {"A": rng.uniform(0, 100), "B": rng.uniform(0, 100)},
                          "test": {"all": rng.uniform(0, 100)}} for c in range(4)}
            episodes, contrasts, controls = analyze_context(0, 0, scores)
            for rule in ("seen", "unseen"):
                self.assertGreaterEqual(sum(r["reuse_pp"] for r in episodes if r["rule"] == rule), -1e-10)
                self.assertAlmostEqual(sum(r[f"fixed_{rule}_reuse_pp"] for r in controls), 0.0)
            for row in contrasts:
                self.assertAlmostEqual(row["apparent_contrast_pp"], row["reuse_contrast_pp"] + row["report_gap_contrast_pp"])
            self.assertTrue(all(r["fixed_test_difference_pp"] == 0 for r in controls))
            expected = np.mean(list(scores[3]["seen"].values())) - np.mean(list(scores[3]["unseen"].values()))
            self.assertAlmostEqual(np.mean([r["fixed_report_exposure_pp"] for r in controls]), expected)

    def test_input_is_not_mutated(self):
        scores = fixture()
        before = copy.deepcopy(scores)
        analyze_context(0, 0, scores)
        self.assertEqual(scores, before)

    def test_strict_score_contract(self):
        mutations = [
            lambda s: s.pop(3),
            lambda s: s.__setitem__(4, copy.deepcopy(s[3])),
            lambda s: s[0].pop("test"),
            lambda s: s[0]["seen"].pop("B"),
            lambda s: s[0]["test"].__setitem__("A", 30),
            lambda s: s[0]["seen"].__setitem__("A", float("nan")),
            lambda s: s[0]["seen"].__setitem__("A", float("inf")),
            lambda s: s[0]["seen"].__setitem__("A", 100.1),
            lambda s: s[0]["seen"].__setitem__("A", -0.1),
            lambda s: s[0]["seen"].__setitem__("A", True),
            lambda s: s[0]["seen"].__setitem__("A", "90"),
        ]
        for mutation in mutations:
            scores = fixture()
            mutation(scores)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    analyze_context(0, 0, scores)
        bool_keys = fixture()
        value = bool_keys.pop(0)
        bool_keys[False] = value
        with self.assertRaises(ValueError):
            analyze_context(0, 0, bool_keys)
        for draw, fold in ((-1, 0), (0, -1), (True, 0), (0, 0.5)):
            with self.assertRaises(ValueError):
                analyze_context(draw, fold, fixture())


if __name__ == "__main__":
    unittest.main()

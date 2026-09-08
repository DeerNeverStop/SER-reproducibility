"""Plan replay and artifact corruption checks, without reading real predictions."""
import json
from pathlib import Path
import tempfile
import unittest

from v3.data_design.core_plan import digest, file_sha, read_metadata, write_json
from v3.inner_validation.plan import generate
from .core import analyze_context
from .run import aggregate, contexts, prediction_gate
from .test_core import fixture


class SplitTests(unittest.TestCase):
    def test_metadata_only_split_replays_and_is_balanced(self):
        repo = Path(__file__).resolve().parents[2]
        meta = read_metadata(repo)
        old = {'units': generate(meta)}
        first = contexts(old, meta)
        self.assertEqual(first, contexts(old, meta))
        self.assertEqual(len(first), 120)
        for ctx in first:
            for role in ('seen', 'unseen'):
                a, b = (ctx['halves'][role][half] for half in ('A', 'B'))
                self.assertFalse(set(a['people']) & set(b['people']))
                self.assertEqual(len(set(a['paths']) | set(b['paths'])), 288)
                for half in (a, b):
                    self.assertEqual(sum(meta['sex'][s] == 'Female' for s in half['people']), 6)
                    self.assertEqual(sum(meta['sex'][s] == 'Male' for s in half['people']), 6)

    def test_draws_equal_weight_directions_then_folds(self):
        episodes, contrasts, controls = [], [], []
        for draw in range(24):
            for fold in range(5):
                scores = fixture()
                for c in scores:
                    scores[c]['test']['all'] = float(draw + fold + c)
                ep, co, ct = analyze_context(draw, fold, scores)
                episodes.extend(ep)
                contrasts.extend(co)
                controls.extend(ct)
        draws, means = aggregate(episodes, contrasts, controls)
        self.assertEqual(len(draws), 24)
        # Seen choices are c0 in A, c1 in B; test = draw + fold + config.
        self.assertAlmostEqual(draws[7]['seen__test_uar'], 7 + 2 + 0.5)
        self.assertAlmostEqual(means['seen__test_uar'], 11.5 + 2 + 0.5)
        self.assertAlmostEqual(means['fixed_seen_reuse_pp'], 0)


class ArtifactTests(unittest.TestCase):
    def test_predgate_rejects_changed_prediction_and_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            unit = {'unit_id': 'fixture'}
            folder = archive / 'runs/formal/units/fixture'
            attempt = folder / 'attempts/0001'
            attempt.mkdir(parents=True)
            prediction = attempt / 'predictions.npz'
            prediction.write_bytes(b'synthetic prediction bytes, never scored')
            write_json(attempt / 'receipt.json', {'epochs_run': 15})
            done = {'unit_id': 'fixture', 'phase': 'formal', 'plan_sha256': 'fixture-plan',
                    'unit_sha256': digest(unit), 'artifacts': {
                        'attempts/0001/' + name: file_sha(attempt / name)
                        for name in ('predictions.npz', 'receipt.json')}}
            write_json(folder / 'DONE', done)
            old = {'plan_sha256': 'fixture-plan', 'units': [unit]}
            gate = {'done_sha256': {'fixture': file_sha(folder / 'DONE')}}
            hashes, paths = prediction_gate(old, gate, archive)
            self.assertEqual(len(hashes), 3)
            self.assertEqual(paths['fixture']['predictions.npz'], prediction.resolve())
            prediction.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                prediction_gate(old, gate, archive)
            (folder / 'DONE').write_text(json.dumps({}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'DONE changed'):
                prediction_gate(old, gate, archive)


if __name__ == '__main__':
    unittest.main()

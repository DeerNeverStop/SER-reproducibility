"""Independent synthetic CPU red-team checks; no study data or real WavLM.

Run with CUDA_VISIBLE_DEVICES empty. All waveform caches and checkpoints below
are temporary toys; running this file does not compute scientific study scores.
"""
import copy
from fractions import Fraction
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from v3.final_program_20260907 import engine, plan


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize('classes', [6, 7, 8])
def test_metrics_match_independent_ce_and_exact_macro_recall(classes):
    labels = np.repeat(np.arange(classes), np.arange(1, classes + 1))
    logits = np.arange(len(labels) * classes, dtype=np.float64).reshape(-1, classes) / 17
    logits[np.arange(len(labels)), (labels + (np.arange(len(labels)) % 2)) % classes] += 10
    logits += 1000  # Stable CE despite a large shared offset.
    ce, uar = engine.metrics(logits, labels, classes)
    reference_ce = float(torch.nn.functional.cross_entropy(
        torch.from_numpy(logits), torch.from_numpy(labels), reduction='mean'))
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, np.argmax(logits, axis=1)), 1)
    reference_uar = sum((Fraction(int(confusion[c, c]), int(confusion[c].sum()))
                         for c in range(classes)), Fraction()) / classes
    assert abs(ce - reference_ce) < 2e-12
    assert isinstance(uar, Fraction) and uar == reference_uar
    assert uar != Fraction(int(np.trace(confusion)), len(labels))


@pytest.mark.parametrize('classes', [6, 7, 8])
def test_equal_logits_choose_first_class_and_equal_uar_is_rational(classes):
    labels = np.arange(classes)
    ce, score = engine.metrics(np.zeros((classes, classes)), labels, classes)
    assert ce == pytest.approx(np.log(classes), abs=1e-15)
    assert score == Fraction(1, classes)
    _, repeated = engine.metrics(np.zeros((classes * 3, classes)), np.tile(labels, 3), classes)
    assert score == repeated  # No floating macro-recall tie noise.


@pytest.mark.parametrize('fault', ['missing', 'nan', 'outside'])
def test_invalid_validation_metrics_fail_closed(fault):
    labels = np.arange(6)
    logits = np.zeros((6, 6))
    if fault == 'missing':
        labels[-1] = 0
    elif fault == 'nan':
        logits[0, 0] = np.nan
    else:
        labels[-1] = 6
    with pytest.raises(ValueError):
        engine.metrics(logits, labels, 6)


def test_paired_crop_start_uses_one_exact_plan_hash_independent_of_length():
    assert engine.crop_start is plan.crop_start
    values = {}
    for epoch in range(1, 16):
        for slot in range(100):
            # Independent canonical-byte hash; do not invoke plan.digest.
            payload = json.dumps(['SER26-FINAL-CHECKPOINT-PROGRAM-1', 77, epoch, slot],
                                 ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            h64 = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')
            shared_uniform = Fraction(h64, 2**64)
            assert 0 <= shared_uniform < 1
            for length in (0, 9, 47999, 48000, 48001, 48011, 61003):
                span = max(0, length - 48000) + 1
                expected = (shared_uniform * span).__floor__()
                start = engine.crop_start(77, epoch, slot, length)
                assert type(start) is int and start == expected
                assert 0 <= start <= max(0, length - 48000)
                values[epoch, slot, length] = start
    assert {key: engine.crop_start(77, *key) for key in reversed(values)} == values
    crop, lengths = 48000, (48011, 61003)
    cache = {str(length): np.arange(length, dtype=np.float32) for length in lengths}
    for length in lengths:
        offset = values[1, 0, length]
        actual = engine.waveform_batch([str(length)], cache.__getitem__, 160000,
                                       crop=crop, starts=[offset])
        np.testing.assert_array_equal(actual.numpy()[0], np.arange(offset, offset + crop))


def test_short_training_waves_always_pad_to_exact_three_seconds():
    cache = {'tiny': np.asarray([1., 2.]), 'short': np.arange(17, dtype=np.float32)}
    original = {key: value.copy() for key, value in cache.items()}
    actual = engine.waveform_batch(['tiny', 'short'], cache.__getitem__, 160000,
                                   crop=48000, starts=[0, 0])
    assert actual.shape == (2, 48000)
    np.testing.assert_array_equal(actual.numpy()[0, :2], [1, 2])
    assert np.count_nonzero(actual.numpy()[0, 2:]) == 0
    for key in cache:
        np.testing.assert_array_equal(cache[key], original[key])
    # Evaluation retains variable length and caps its own batch only.
    evaluated = engine.waveform_batch(['tiny', 'short'], cache.__getitem__, 10)
    assert evaluated.shape == (2, 10)


def synthetic_unit(classes=6, arm='A'):
    unit = dict(unit_id='SYNTHETIC_ONLY', n_classes=classes, arm=arm,
                selection_enabled=arm == 'A',
                prediction_epochs=list(range(1, 16)) if arm == 'A' else [15],
                seeds=dict(initialization=31, torch_training=47, order=59, crop=71),
                config=dict(epochs=15, batch_size=16, fp16=True, lr_encoder=5e-5,
                            lr_head=1e-3, weight_decay=.01,
                            crop_seconds=3., eval_cap_seconds=10.),
                report={}, report_batches={})
    rows, cache = {}, {}
    for group in ('fit', *engine.GROUPS):
        count = classes * (2 if group == 'fit' else 3)
        paths = [f'{group}/{i:03}.wav' for i in range(count)]
        if group == 'fit':
            unit['fit'] = paths
        else:
            unit['report'][group] = paths
            unit['report_batches'][group] = [paths[i:i + 16] for i in range(0, count, 16)]
        speaker = arm if group == 'fit' else group
        for i, path in enumerate(paths):
            rows[path] = dict(label_index=str(i % classes), speaker=speaker,
                              sentence='train-text' if group == 'fit' else 'query-text')
            cache[path] = np.linspace(.05, .2 + i / 25, 9 + i % 7, dtype=np.float32)
    return unit, rows, cache


class TinyTrajectory(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.3),
                                       torch.nn.Linear(5, 5))
        for parameter in self.enc[0].parameters():
            parameter.requires_grad = False
        self.fc = torch.nn.Linear(5, classes)
        self.register_buffer('fixed_buffer', torch.tensor(2))
        self.forward_modes = []

    def forward(self, waves):
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        x = torch.stack((waves.mean(dim=1), waves.square().mean(dim=1), waves[:, 0]), dim=1)
        return self.fc(self.enc(x))


def toy_builder(monkeypatch):
    built = []
    def build(path, seed, classes):
        torch.manual_seed(seed)
        model = TinyTrajectory(classes)
        built.append(model)
        return model, engine.tensor_mapping_sha256(model.state_dict())
    monkeypatch.setattr(engine, 'build_model', build)
    return built


def test_four_rules_choose_own_best_and_earliest_exact_tie(tmp_path, monkeypatch):
    arm = 'A'
    unit, rows, cache = synthetic_unit(arm=arm)
    built = toy_builder(monkeypatch)
    calls = []
    def controlled_metrics(logits, labels, classes):
        ordinal = len(calls)
        epoch, group = ordinal // 2 + 1, ('A', 'B')[ordinal % 2]
        role = 'seen' if group == arm else 'unseen'
        calls.append((epoch, group))
        ce_low = (2, 3) if role == 'seen' else (4, 5)
        uar_high = (5, 6) if role == 'seen' else (3, 4)
        return (.1 if epoch in ce_low else 2.,
                Fraction(2, 3) if epoch in uar_high else Fraction(1, 3))
    monkeypatch.setattr(engine, 'metrics', controlled_metrics)
    out = tmp_path / arm
    info = engine.fit_unit(unit, rows, tmp_path, tmp_path / 'fake.pt', out,
                           torch.device('cpu'), cache.__getitem__)
    wanted = dict(seen_ce=2, unseen_ce=4, seen_uar=5, unseen_uar=3, last=15)
    assert info['selected_epochs'] == wanted
    assert calls == [(epoch, group) for epoch in range(1, 16) for group in ('A', 'B')]
    checkpoint = torch.load(out / 'checkpoint.pt', weights_only=True)
    assert checkpoint['selected_epochs'] == wanted
    assert set(checkpoint['epoch_states']) == {'2', '3', '4', '5', '15'}
    assert all('fixed_buffer' in delta for delta in checkpoint['epoch_states'].values())
    assert info['reload_max_abs_diff'] == 0
    history = json.loads((out / 'history.json').read_text())
    assert len(history) == 15
    assert sum(row['optimizer_steps'] for row in history) == 15
    training_calls = [entry for entry in built[0].forward_modes if entry[1]]
    assert training_calls == [(True, True)] * 15  # Each epoch re-enters train after eval.
    assert all(not mode for mode, grad in built[0].forward_modes if not grad)
    assert all(not mode and not grad for mode, grad in built[1].forward_modes)


def test_b_arm_saves_only_last_and_never_selects_or_computes_metrics(tmp_path, monkeypatch):
    unit, rows, cache = synthetic_unit(arm='B')
    built = toy_builder(monkeypatch)
    def forbidden_metrics(*args):
        raise AssertionError('B arm must not compute validation or outer scores')
    monkeypatch.setattr(engine, 'metrics', forbidden_metrics)
    out = tmp_path / 'last_only_B'
    info = engine.fit_unit(unit, rows, tmp_path, tmp_path / 'fake.pt', out,
                           torch.device('cpu'), cache.__getitem__)
    assert info['selected_epochs'] == {'last': 15}
    assert set(info['reload_by_epoch_group']) == {'15/A', '15/B', '15/outer'}
    history = json.loads((out / 'history.json').read_text())
    assert len(history) == 15
    assert all(set(item) == {'epoch', 'train_loss', 'optimizer_steps', 'scaler_skipped_steps'}
               for item in history)
    assert [entry for entry in built[0].forward_modes if entry[1]] == [(True, True)] * 15
    assert sum(not grad for mode, grad in built[0].forward_modes) == 6  # Three groups, two batches.
    with np.load(out / 'predictions.npz') as saved:
        assert saved['epochs'].tolist() == [15]
        assert saved['outer__all_epoch_logits'].shape == (1, 18, 6)
    checkpoint = torch.load(out / 'checkpoint.pt', weights_only=True)
    assert checkpoint['selected_epochs'] == {'last': 15}
    assert set(checkpoint['epoch_states']) == {'15'}


def test_outer_waveforms_and_labels_cannot_change_validation_or_training(tmp_path, monkeypatch):
    unit, rows, cache = synthetic_unit()
    toy_builder(monkeypatch)
    changes = copy.deepcopy(cache)
    changed_rows = copy.deepcopy(rows)
    for i, path in enumerate(unit['report']['outer']):
        changes[path] = np.full(800 + i * 137, .9 - i / 100, dtype=np.float32)
        changed_rows[path]['label_index'] = str((int(rows[path]['label_index']) + 1) % 6)
    prediction_calls, metrics_count = [], []
    native_predict, native_metrics = engine.predict_group, engine.metrics
    def checked_predict(model, batches, wavcache, cap, classes, device):
        assert not model.training
        paths = [p for batch in batches for p in batch]
        group = paths[0].split('/')[0]
        assert batches == unit['report_batches'][group]
        assert paths == unit['report'][group]
        prediction_calls.append(group)
        return native_predict(model, batches, wavcache, cap, classes, device)
    def validation_only(logits, labels, classes):
        metrics_count.append(1)
        return native_metrics(logits, labels, classes)
    monkeypatch.setattr(engine, 'predict_group', checked_predict)
    monkeypatch.setattr(engine, 'metrics', validation_only)
    outputs, receipts = [], []
    for index, (metadata, wavecache) in enumerate(((rows, cache), (changed_rows, changes))):
        out = tmp_path / str(index)
        receipts.append(engine.fit_unit(unit, metadata, tmp_path, tmp_path / 'fake.pt', out,
                                        torch.device('cpu'), wavecache.__getitem__))
        outputs.append(out)
    assert len(metrics_count) == 15 * 2 * 2  # A/B only, never an outer metric call.
    assert receipts[0]['selected_epochs'] == receipts[1]['selected_epochs']
    assert receipts[0]['initial_state_sha256'] == receipts[1]['initial_state_sha256']
    assert all(r['outer_scores_computed'] is False for r in receipts)
    with np.load(outputs[0] / 'predictions.npz') as left, np.load(outputs[1] / 'predictions.npz') as right:
        expected_keys = {f'{g}__{field}' for g in engine.GROUPS
                         for field in ('paths', 'labels', 'all_epoch_logits')}
        expected_keys.add('epochs')
        assert set(left.files) == set(right.files) == expected_keys
        assert left['epochs'].tolist() == right['epochs'].tolist() == list(range(1, 16))
        for group in ('A', 'B'):
            np.testing.assert_array_equal(left[f'{group}__all_epoch_logits'], right[f'{group}__all_epoch_logits'])
        assert not np.array_equal(left['outer__all_epoch_logits'], right['outer__all_epoch_logits'])
        assert not np.array_equal(left['outer__labels'], right['outer__labels'])
        assert left['outer__all_epoch_logits'].shape == (15, 18, 6)
    histories = [json.loads((out / 'history.json').read_text()) for out in outputs]
    assert histories[0] == histories[1]
    allowed_history = {'epoch', 'train_loss', 'optimizer_steps', 'scaler_skipped_steps', *engine.RULES}
    assert all(set(row) == allowed_history for row in histories[0])
    checkpoints = [torch.load(out / 'checkpoint.pt', weights_only=True) for out in outputs]
    assert checkpoints[0]['selected_epochs'] == checkpoints[1]['selected_epochs']
    for epoch in checkpoints[0]['epoch_states']:
        for key, value in checkpoints[0]['epoch_states'][epoch].items():
            assert torch.equal(value, checkpoints[1]['epoch_states'][epoch][key])


@pytest.mark.parametrize('source', ['python', 'torch'])
def test_fit_rejects_evaluation_randomness_before_sealing(tmp_path, monkeypatch, source):
    unit, rows, cache = synthetic_unit()
    toy_builder(monkeypatch)
    native_predict = engine.predict_group
    def consumes_rng(*args, **kwargs):
        if source == 'python':
            random.random()
        else:
            torch.rand(1)
        return native_predict(*args, **kwargs)
    monkeypatch.setattr(engine, 'predict_group', consumes_rng)
    out = tmp_path / 'must_not_seal'
    with pytest.raises(ValueError, match='evaluation changed training RNG'):
        engine.fit_unit(unit, rows, tmp_path, tmp_path / 'fake.pt', out,
                        torch.device('cpu'), cache.__getitem__)
    assert not (out / 'checkpoint.pt').exists()
    assert not (out / 'predictions.npz').exists()


def test_prediction_preserves_python_numpy_and_torch_rng():
    unit, rows, cache = synthetic_unit()
    model = TinyTrajectory(6).eval()
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    engine.predict_group(model, unit['report_batches']['A'], cache.__getitem__,
                         160000, 6, torch.device('cpu'))
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


@pytest.mark.parametrize('fault', ['order', 'mixed_group', 'speaker', 'text', 'selection', 'epochs'])
def test_role_contract_mutations_are_rejected(fault):
    unit, rows, _ = synthetic_unit()
    if fault == 'order':
        unit['report_batches']['A'][0].reverse()
    elif fault == 'mixed_group':
        unit['report_batches']['A'][0][0] = unit['report']['outer'][0]
    elif fault == 'speaker':
        rows[unit['report']['outer'][0]]['speaker'] = 'A'
    elif fault == 'text':
        rows[unit['report']['B'][0]]['sentence'] = 'train-text'
    elif fault == 'selection':
        unit['selection_enabled'] = False
    else:
        unit['prediction_epochs'] = [15]
    with pytest.raises(ValueError):
        engine.validate_unit(unit, rows)


@pytest.mark.parametrize('classes', [6, 7, 8])
def test_native_head_size_top_four_and_reconstructible_initialization(tmp_path, monkeypatch, classes):
    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.transformer = torch.nn.Module()
            self.encoder.transformer.layers = torch.nn.ModuleList(
                [torch.nn.Linear(1, 1) for _ in range(6)])
        def extract_features(self, waves):
            features = waves.mean(dim=1)[:, None, None].expand(-1, 2, 768)
            return [features], None
    encoder = FakeEncoder()
    base = tmp_path / 'synthetic_encoder.pt'
    torch.save(encoder.state_dict(), base)
    fake_audio = SimpleNamespace(
        pipelines=SimpleNamespace(WAVLM_BASE_PLUS=SimpleNamespace(
            _model_type='WavLM', _normalize_waveform=False, sample_rate=16000, _params={})),
        models=SimpleNamespace(wavlm_model=lambda **kwargs: FakeEncoder()))
    monkeypatch.setitem(sys.modules, 'torchaudio', fake_audio)
    monkeypatch.setattr(engine, 'state_dict_sha256', lambda model: engine.WEIGHTS_SHA)
    model, identity = engine.build_model(base, 17, classes)
    second, second_identity = engine.build_model(base, 17, classes)
    assert model.fc.out_features == classes
    assert model(torch.ones((2, 5))).shape == (2, classes)
    assert identity == second_identity
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in model.state_dict().items())
    for index, layer in enumerate(model.enc.encoder.transformer.layers):
        assert all(p.requires_grad == (index >= 2) for p in layer.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
    with pytest.raises(ValueError, match='unexpected native label count'):
        engine.build_model(base, 17, 5)

"""Synthetic CPU checks only: tiny models/caches, no study data or GPU work."""
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

from . import engine


def seal(unit):
    unit['unit_sha256'] = engine.digest({k: v for k, v in unit.items() if k != 'unit_sha256'})
    return unit


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def synthetic_unit(classes=6, arm='A', epochs=15):
    rows, cache, report = {}, {}, {}
    for group in ('fit', *engine.GROUPS):
        paths = []
        for c in range(classes):
            p = f'{group}/{c}.wav'
            rows[p] = dict(label_index=c, speaker=arm if group == 'fit' else group,
                           sentence='fit_text' if group == 'fit' else 'query_text')
            cache[p] = np.linspace(-.3, .4 + c / 20, 19 + c * 7, dtype=np.float32)
            paths.append(p)
        report[group] = paths
    unit = dict(unit_id=f'SYNTHETIC-{classes}-{arm}-{epochs}', corpus='synthetic',
                arm=arm, n_classes=classes, fit=report.pop('fit'), report=report,
                report_batches={g: [p[:3], p[3:]] for g, p in report.items()},
                windows=[15] if epochs == 15 else [15, 45],
                seeds=dict(head=123, torch_training=456, order=789, crop=912),
                config=dict(epochs=epochs, batch_size=16, fp16=True, lr_encoder=.001,
                            lr_head=.01, weight_decay=.01, crop_seconds=3, eval_cap_seconds=10))
    return seal(unit), rows, cache


SPEC = dict(name='wavlm_base_plus', path='SYNTHETIC-not-read.pt',
            file_sha256='a' * 64, state_sha256='b' * 64)


class TinyTrajectory(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.25),
                                       torch.nn.Linear(5, 5))
        for p in self.enc[0].parameters():
            p.requires_grad = False
        self.fc = torch.nn.Linear(5, classes)
        self.register_buffer('fixed_buffer', torch.tensor(2))
        self.calls = []

    def forward(self, waves):
        self.calls.append((self.training, torch.is_grad_enabled()))
        x = torch.stack((waves.mean(1), waves.square().mean(1), waves[:, 0]), 1)
        return self.fc(self.enc(x))


def toy_builder(monkeypatch):
    built = []
    def build(spec, head_seed, classes):
        engine.set_seed(head_seed)
        model = TinyTrajectory(classes)
        built.append(model)
        return model, engine.tensor_mapping_sha256(model.state_dict())
    monkeypatch.setattr(engine, 'build_model', build)
    return built


@pytest.mark.parametrize('classes', [6, 7, 8])
def test_native_metrics_match_independent_confusion_and_ce(classes):
    labels = np.repeat(np.arange(classes), np.arange(1, classes + 1))
    logits = np.random.default_rng(33).normal(size=(len(labels), classes)) + 1000
    ce, uar = engine.metrics(logits, labels, classes)
    expected_ce = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    confusion = np.zeros((classes, classes), np.int64)
    np.add.at(confusion, (labels, logits.argmax(1)), 1)
    expected_uar = sum((Fraction(int(confusion[c, c]), int(confusion[c].sum()))
                        for c in range(classes)), Fraction()) / classes
    assert abs(ce - expected_ce) < 2e-12
    assert uar == expected_uar and isinstance(uar, Fraction)


def test_selection_exact_ties_earliest_and_no_outer_rule():
    best, selected = {}, {}
    values = dict(seen_ce=.5, unseen_ce=.3, seen_uar=Fraction(1, 3), unseen_uar=Fraction(2, 3))
    assert engine.update_selection(best, selected, 1, values) == list(engine.RULES)
    assert engine.update_selection(best, selected, 2, values) == []
    assert set(selected.values()) == {1}
    changed = dict(values, seen_ce=.4, unseen_uar=Fraction(3, 4))
    assert engine.update_selection(best, selected, 3, changed) == ['seen_ce', 'unseen_uar']
    with pytest.raises(ValueError):
        engine.update_selection(best, selected, 4, dict(changed, outer_uar=1))


def test_shared_crop_integer_mapping_and_short_padding():
    for epoch in (1, 15, 16, 45):
        for slot in range(12):
            blob = json.dumps([engine.CROP_DOMAIN, 19, epoch, slot],
                              separators=(',', ':')).encode()
            integer = int.from_bytes(hashlib.sha256(blob).digest()[:8], 'big')
            assert 0 <= Fraction(integer, 2**64) < 1
            for length in (0, 8, 47999, 48000, 48001, 89117):
                expected = (Fraction(integer, 2**64) * (max(0, length - 48000) + 1)).__floor__()
                assert engine.crop_start(19, epoch, slot, length) == expected
    cache = {'short': np.arange(13, dtype=np.float32), 'long': np.arange(62003, dtype=np.float32)}
    start = engine.crop_start(19, 45, 0, 62003)
    batch = engine.waveform_batch(['short', 'long'], cache.__getitem__, 160000,
                                  crop=48000, starts=[0, start]).numpy()
    assert batch.shape == (2, 48000)
    np.testing.assert_array_equal(batch[0, :13], cache['short'])
    assert not batch[0, 13:].any()
    np.testing.assert_array_equal(batch[1], cache['long'][start:start + 48000])
    with pytest.raises(ValueError):
        engine.crop_start(19, 46, 0, 12)


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_long_trajectory_dual_windows_dedup_own_roles_and_all_epoch_replay(tmp_path, monkeypatch, arm):
    unit, rows, cache = synthetic_unit(arm=arm, epochs=45)
    built = toy_builder(monkeypatch)
    calls = []
    def metric(logits, labels, classes):
        epoch, group = len(calls) // 2 + 1, ('A', 'B')[len(calls) % 2]
        calls.append((epoch, group))
        seen = group == arm
        ce = .1 if epoch in ((20, 21) if seen else (23, 24)) else (
            .5 if epoch in ((2, 3) if seen else (4, 5)) else 2.)
        uar = Fraction(9, 10) if epoch in ((27, 28) if seen else (30, 31)) else (
            Fraction(2, 3) if epoch in ((5, 6) if seen else (3, 4)) else Fraction(1, 3))
        return ce, uar
    monkeypatch.setattr(engine, 'metrics', metric)
    out = tmp_path / arm
    info = engine.train(unit, rows, {}, SPEC, out, 'cpu', cache=cache.__getitem__)
    wanted = {'15': dict(seen_ce=2, unseen_ce=4, seen_uar=5, unseen_uar=3, last=15),
              '45': dict(seen_ce=20, unseen_ce=23, seen_uar=27, unseen_uar=30, last=45)}
    assert info['selected_epochs'] == wanted
    assert calls == [(e, g) for e in range(1, 46) for g in ('A', 'B')]
    checkpoint = torch.load(out / 'checkpoint.pt', weights_only=True)
    assert checkpoint['selected_epochs'] == wanted
    assert len(checkpoint['epoch_states']) == 10
    assert checkpoint['training_resume_supported'] is False
    assert all('fixed_buffer' in delta for delta in checkpoint['epoch_states'].values())
    assert info['reload_max_abs_diff'] == 0 and len(info['reload_by_epoch_group']) == 30
    assert info['reload_checks'] == 30 and info['reload_checked_values'] == 30 * 6 * 6
    assert info['outer_scores_computed'] is False
    assert [call for call in built[0].calls if call[1]] == [(True, True)] * 45
    assert all(not mode for mode, grad in built[0].calls if not grad)
    assert all(not mode and not grad for mode, grad in built[1].calls)
    with np.load(out / 'predictions.npz') as saved:
        assert len(saved.files) == 10 and saved['epochs'].tolist() == list(range(1, 46))
        assert saved['outer__all_epoch_logits'].shape == (45, 6, 6)
        assert saved['outer__all_epoch_logits'].dtype == np.float64
    assert info['optimizer_steps'] == 45 and info['scaler_skipped_steps'] == 0
    loaded, sha = engine.validate_checkpoint(unit, SPEC, out / 'checkpoint.pt')
    assert sha == engine.file_sha(out / 'checkpoint.pt') and loaded['selected_epochs'] == wanted
    wrong_unit = copy.deepcopy(unit)
    wrong_unit['seeds']['crop'] += 1
    with pytest.raises(ValueError, match='unit SHA differs'):
        engine.validate_checkpoint(wrong_unit, SPEC, out / 'checkpoint.pt')
    loaded['epoch_states']['45'].pop('fixed_buffer')
    torch.save(loaded, tmp_path / 'corrupt_checkpoint.pt')
    with pytest.raises(ValueError, match='coverage'):
        engine.validate_checkpoint(unit, SPEC, tmp_path / 'corrupt_checkpoint.pt')
    with pytest.raises(ValueError, match='fresh'):
        engine.train(unit, rows, {}, SPEC, out, 'cpu', cache=cache.__getitem__)


def test_15_epoch_prefix_equals_45_despite_total_budget_and_outer_mutation(tmp_path, monkeypatch):
    unit15, rows, cache = synthetic_unit()
    unit45 = copy.deepcopy(unit15)
    unit45['config']['epochs'], unit45['windows'] = 45, [15, 45]
    seal(unit45)
    changed_rows, changed_cache = copy.deepcopy(rows), copy.deepcopy(cache)
    for index, path in enumerate(unit45['report']['outer']):
        changed_cache[path] = np.full(2000 + index * 131, .8, np.float32)
        changed_rows[path]['label_index'] = (index + 1) % 6
    toy_builder(monkeypatch)
    info15 = engine.train(unit15, rows, {}, SPEC, tmp_path / 'short', 'cpu', cache=cache.__getitem__)
    info45 = engine.train(unit45, changed_rows, {}, SPEC, tmp_path / 'long', 'cpu',
                          cache=changed_cache.__getitem__)
    assert info15['selected_epochs']['15'] == info45['selected_epochs']['15']
    assert info15['initial_state_sha256'] == info45['initial_state_sha256']
    h15 = json.loads((tmp_path / 'short/history.json').read_text())
    h45 = json.loads((tmp_path / 'long/history.json').read_text())
    assert h15 == h45[:15]
    with np.load(tmp_path / 'short/predictions.npz') as short, np.load(tmp_path / 'long/predictions.npz') as long:
        for role in ('A', 'B'):
            np.testing.assert_array_equal(short[f'{role}__all_epoch_logits'], long[f'{role}__all_epoch_logits'][:15])
        assert not np.array_equal(short['outer__all_epoch_logits'], long['outer__all_epoch_logits'][:15])
    short = torch.load(tmp_path / 'short/checkpoint.pt', weights_only=True)
    long = torch.load(tmp_path / 'long/checkpoint.pt', weights_only=True)
    for epoch, state in short['epoch_states'].items():
        for key, value in state.items():
            assert torch.equal(value, long['epoch_states'][epoch][key])


@pytest.mark.parametrize('source', ['python', 'numpy', 'torch'])
def test_eval_randomness_rejected_before_seal(tmp_path, monkeypatch, source):
    unit, rows, cache = synthetic_unit()
    toy_builder(monkeypatch)
    native = engine.predict_group
    def consume(*args):
        {'python': random.random, 'numpy': np.random.random, 'torch': lambda: torch.rand(1)}[source]()
        return native(*args)
    monkeypatch.setattr(engine, 'predict_group', consume)
    with pytest.raises(ValueError, match='evaluation changed'):
        engine.train(unit, rows, {}, SPEC, tmp_path / 'bad', 'cpu', cache=cache.__getitem__)
    assert not (tmp_path / 'bad/checkpoint.pt').exists()


@pytest.mark.parametrize('fault', ['group', 'order', 'speaker', 'text', 'windows', 'epoch', 'seed'])
def test_invalid_contracts_fail_closed(fault):
    unit, rows, _ = synthetic_unit()
    if fault == 'group':
        unit['report_batches']['A'][0][0] = unit['report']['outer'][0]
    elif fault == 'order':
        unit['report_batches']['A'][0].reverse()
    elif fault == 'speaker':
        rows[unit['report']['outer'][0]]['speaker'] = 'A'
    elif fault == 'text':
        rows[unit['report']['B'][0]]['sentence'] = 'fit_text'
    elif fault == 'windows':
        unit['windows'] = [14]
    elif fault == 'epoch':
        unit['config']['epochs'] = 16
    else:
        unit['seeds']['order'] = -1
    seal(unit)
    with pytest.raises(ValueError):
        engine.validate_unit(unit, rows)


class FakeEncoder(torch.nn.Module):
    def __init__(self, extra_rng=0):
        super().__init__()
        torch.rand(extra_rng)
        self.encoder = torch.nn.Module()
        self.encoder.transformer = torch.nn.Module()
        self.encoder.transformer.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(12)])

    def extract_features(self, waves):
        return [waves.mean(1)[:, None, None].expand(-1, 2, 768)], None


def fake_torchaudio(monkeypatch):
    params = dict(encoder_num_layers=12, encoder_embed_dim=768)
    bundle = lambda kind: SimpleNamespace(_model_type=kind, _normalize_waveform=False,
                                          sample_rate=16000, _params=params)
    monkeypatch.setitem(sys.modules, 'torchaudio', SimpleNamespace(
        pipelines=SimpleNamespace(WAVLM_BASE_PLUS=bundle('WavLM'), HUBERT_BASE=bundle('Wav2Vec2')),
        models=SimpleNamespace(wavlm_model=lambda **kw: FakeEncoder(0),
                               wav2vec2_model=lambda **kw: FakeEncoder(47))))


@pytest.mark.parametrize('classes', [6, 7, 8])
def test_strict_bases_native_heads_top4_and_constructor_rng_independence(tmp_path, monkeypatch, classes):
    fake_torchaudio(monkeypatch)
    base = FakeEncoder()
    path = tmp_path / 'base.pt'
    torch.save(base.state_dict(), path)
    common = dict(path=str(path), file_sha256=engine.file_sha(path), state_sha256=engine.state_dict_sha256(base))
    wavlm, identity = engine.build_model(dict(common, name='wavlm_base_plus'), 123, classes)
    hubert, second_identity = engine.build_model(dict(common, name='hubert_base'), 123, classes)
    assert identity == second_identity  # Same fake base, independently seeded identical heads.
    assert torch.equal(wavlm.fc.weight, hubert.fc.weight)
    assert wavlm.fc.out_features == classes
    for index, layer in enumerate(hubert.enc.encoder.transformer.layers):
        assert all(p.requires_grad == (index >= 8) for p in layer.parameters())
    for fault in ('file_sha256', 'state_sha256'):
        with pytest.raises(ValueError, match='SHA differs'):
            engine.build_model(dict(common, name='hubert_base', **{fault: 'f' * 64}), 123, classes)
    state = base.state_dict()
    state.pop(next(iter(state)))
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        engine.build_model(dict(common, name='hubert_base', file_sha256=engine.file_sha(path)), 123, classes)


def test_delta_missing_buffer_or_frozen_mutation_rejected():
    model = TinyTrajectory(6)
    expected = engine.frozen_hash(model)
    names, buffers = engine.delta_names(model)
    delta = engine.cpu_state(model, set(names) | set(buffers))
    engine.load_delta(model, delta, expected)
    broken = dict(delta)
    broken.pop('fixed_buffer')
    with pytest.raises(ValueError):
        engine.load_delta(model, broken, expected)
    with torch.no_grad():
        model.enc[0].weight.add_(1)
    with pytest.raises(ValueError, match='frozen encoder'):
        engine.load_delta(model, delta, expected)

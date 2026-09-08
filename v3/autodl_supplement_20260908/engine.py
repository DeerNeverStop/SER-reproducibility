"""Offline two-backbone training with shared validation rules on one trajectory.

The caller owns source/input locks, attempts, receipts, ledgers and retention.
All saved deltas remain present after this function: no cleanup is performed.
Outer labels are archived and validated structurally, never scored or selected.
"""
from __future__ import annotations

from fractions import Fraction
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import time

import numpy as np

GROUPS = ('A', 'B', 'outer')
RULES = ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar')
CROP_DOMAIN = 'SER26-AUTODL-SUPPLEMENT-CROP-1'
CHECKPOINT_SCHEMA = 'ser-autodl-supplement-delta-1'
MODEL_NAMES = ('wavlm_base_plus', 'hubert_base')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unit_identity(unit):
    actual = digest({k: v for k, v in unit.items() if k != 'unit_sha256'})
    require(unit.get('unit_sha256') == actual, 'unit SHA differs')
    return actual


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_mapping_sha256(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        h.update(canonical(dict(name=name, shape=list(value.shape), dtype=str(value.dtype))))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_dict_sha256(module):
    """Legacy-compatible semantic identity: sorted keys, raw tensor bytes."""
    h = hashlib.sha256()
    for _, value in sorted(module.state_dict().items()):
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_identity(spec):
    require(spec.get('name') in MODEL_NAMES, 'unknown backbone')
    for key in ('file_sha256', 'state_sha256'):
        value = spec.get(key)
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'invalid model SHA')
    return {key: spec[key] for key in ('name', 'file_sha256', 'state_sha256')}


def build_model(spec, head_seed, classes):
    """CPU construction from pinned local bytes; never get_model/download.

    Head initialization has its own seed *after* encoder construction/loading,
    so different backbone constructor RNG use cannot change the initial head.
    """
    import torch
    import torchaudio
    identity = model_identity(spec)
    require(type(head_seed) is int and 0 <= head_seed < 2**63, 'invalid head seed')
    require(type(classes) is int and classes in (6, 7, 8), 'unexpected native label count')
    blob = Path(spec['path']).read_bytes()
    require(hashlib.sha256(blob).hexdigest() == identity['file_sha256'], 'base file SHA differs')
    bundle = (torchaudio.pipelines.WAVLM_BASE_PLUS if spec['name'] == 'wavlm_base_plus'
              else torchaudio.pipelines.HUBERT_BASE)
    wanted_type = 'WavLM' if spec['name'] == 'wavlm_base_plus' else 'Wav2Vec2'
    require(bundle._model_type == wanted_type and bundle._normalize_waveform is False
            and bundle.sample_rate == 16000, 'bundle semantics differ')
    require(bundle._params['encoder_num_layers'] == 12
            and bundle._params['encoder_embed_dim'] == 768, 'backbone dimensions differ')
    constructor = (torchaudio.models.wavlm_model if spec['name'] == 'wavlm_base_plus'
                   else torchaudio.models.wav2vec2_model)
    encoder = constructor(**bundle._params)
    encoder.load_state_dict(torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True), strict=True)
    del blob
    require(state_dict_sha256(encoder) == identity['state_sha256'], 'base semantic SHA differs')
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    layers = encoder.encoder.transformer.layers
    require(len(layers) == 12, 'encoder layer count differs')
    for layer in layers[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    set_seed(head_seed)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = encoder
            self.fc = torch.nn.Linear(768, classes)

        def forward(self, waves):
            values, _ = self.enc.extract_features(waves)
            return self.fc(values[-1].mean(dim=1))

    model = Model()
    return model, tensor_mapping_sha256(model.state_dict())


def frozen_hash(model):
    return tensor_mapping_sha256({name: p for name, p in model.named_parameters() if not p.requires_grad})


def delta_names(model):
    return (sorted(n for n, p in model.named_parameters() if p.requires_grad),
            sorted(n for n, _ in model.named_buffers()))


def cpu_state(model, keys):
    return {n: t.detach().cpu().clone() for n, t in model.state_dict().items() if n in keys}


def load_delta(model, delta, expected_frozen_hash):
    names, buffers = delta_names(model)
    require(set(delta) == set(names) | set(buffers), 'delta parameter/buffer coverage differs')
    result = model.load_state_dict(delta, strict=False)
    missing = {n for n, p in model.named_parameters() if not p.requires_grad}
    require(set(result.missing_keys) == missing and not result.unexpected_keys, 'invalid delta keys')
    require(frozen_hash(model) == expected_frozen_hash, 'frozen encoder changed')


def crop_start(seed, epoch, slot, n_samples):
    """Exact shared uniform-to-offset mapping, independent of model/total epochs."""
    require(all(type(x) is int for x in (seed, epoch, slot, n_samples)), 'crop inputs must be integers')
    require(seed >= 0 and 1 <= epoch <= 45 and slot >= 0 and n_samples >= 0, 'invalid crop coordinates')
    integer = int(digest([CROP_DOMAIN, seed, epoch, slot])[:16], 16)
    return integer * (max(0, n_samples - 48000) + 1) // 2**64


class WaveCache:
    """Same raw audio preprocessing as the earlier frozen engine, offline."""
    def __init__(self, audio_root):
        self.audio_root = Path(audio_root).resolve()
        self.cache = {}

    def __call__(self, path):
        import librosa
        if path not in self.cache:
            p = PurePosixPath(path)
            require(isinstance(path, str) and '\\' not in path and ':' not in path
                    and not p.is_absolute() and all(x not in ('', '.', '..') for x in path.split('/')),
                    'unsafe relative audio path')
            full = (self.audio_root / path).resolve()
            require(full.is_relative_to(self.audio_root), 'audio escapes root')
            wave, _ = librosa.load(str(full), sr=16000, mono=True)
            wave, _ = librosa.effects.trim(wave, top_db=30.0)
            self.cache[path] = wave.astype(np.float16)
        return self.cache[path]


def waveform_batch(paths, cache, cap, *, crop=None, starts=None):
    import torch
    require(paths and (crop is None) == (starts is None), 'invalid waveform batch')
    require(starts is None or len(starts) == len(paths), 'crop count differs')
    waves = []
    for index, path in enumerate(paths):
        wav = cache(path)
        require(len(wav) > 0 and np.isfinite(wav).all(), 'invalid processed audio')
        if crop is not None:
            require(type(starts[index]) is int and 0 <= starts[index] <= max(0, len(wav) - crop),
                    'invalid crop start')
            offset = starts[index]
            wav = wav[offset:offset + crop]
            wav = np.pad(wav, (0, crop - len(wav)))
        else:
            wav = wav[:cap]
        waves.append(torch.as_tensor(wav, dtype=torch.float32))
    return torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)


def metrics(logits, labels, classes):
    """Float64 CE, exact rational native-class UAR; A/B validation only."""
    values, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    require(values.shape == (len(labels), classes) and np.isfinite(values).all(), 'invalid logits')
    require(np.all((labels >= 0) & (labels < classes)), 'invalid labels')
    support = np.bincount(labels, minlength=classes)
    require((support > 0).all(), 'missing validation class')
    maximum = values.max(axis=1)
    lse = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    ce = float(np.mean(lse - values[np.arange(len(labels)), labels]))
    pred = values.argmax(axis=1)
    uar = sum((Fraction(int(((pred == c) & (labels == c)).sum()), int(support[c]))
               for c in range(classes)), Fraction()) / classes
    require(math.isfinite(ce), 'nonfinite CE')
    return ce, uar


def predict_group(model, batches, cache, cap, classes, device):
    import torch
    require(not model.training, 'prediction requires eval mode')
    values = []
    with torch.no_grad():
        for paths in batches:
            logits = model(waveform_batch(paths, cache, cap).to(device)).detach().cpu().numpy()
            require(logits.shape == (len(paths), classes) and np.isfinite(logits).all(), 'prediction differs')
            values.append(logits.astype(np.float64))
    return np.concatenate(values)


def rng_state(device):
    import torch
    numpy = np.random.get_state()
    return (random.getstate(), (numpy[0], numpy[1].copy(), *numpy[2:]), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def check_rng(before, device):
    import torch
    after = rng_state(device)
    require(before[0] == after[0] and before[1][0] == after[1][0]
            and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
            and torch.equal(before[2], after[2])
            and (before[3] is None or torch.equal(before[3], after[3])), 'evaluation changed training RNG')


def validate_unit(unit, rows):
    unit_identity(unit)
    cfg, classes = unit['config'], unit['n_classes']
    require(type(classes) is int and classes in (6, 7, 8) and unit['arm'] in ('A', 'B'), 'invalid unit')
    require(type(cfg['epochs']) is int and cfg['epochs'] in (15, 45), 'invalid epoch budget')
    require(unit['windows'] == ([15] if cfg['epochs'] == 15 else [15, 45]), 'invalid selection windows')
    require(cfg['batch_size'] == 16 and cfg['fp16'] is True and cfg['crop_seconds'] == 3
            and cfg['eval_cap_seconds'] == 10, 'training configuration differs')
    for key in ('lr_encoder', 'lr_head', 'weight_decay'):
        require(type(cfg[key]) in (int, float) and math.isfinite(cfg[key])
                and (cfg[key] >= 0 if key == 'weight_decay' else cfg[key] > 0), 'invalid optimizer settings')
    for key in ('head', 'torch_training', 'order', 'crop'):
        require(type(unit['seeds'][key]) is int and 0 <= unit['seeds'][key] < 2**32, 'invalid seed')
    require(set(unit['report']) == set(GROUPS) and set(unit['report_batches']) == set(GROUPS),
            'report groups differ')
    occupied, labels = set(), {}
    for group, paths in [('fit', unit['fit']), *((g, unit['report'][g]) for g in GROUPS)]:
        require(paths and len(paths) == len(set(paths)), 'empty or duplicate role')
        require(not occupied.intersection(paths), 'audio role overlap')
        occupied.update(paths)
        require(all(p in rows for p in paths), 'unknown audio path')
        y = np.asarray([int(rows[p]['label_index']) for p in paths], dtype=np.int64)
        require(np.all((y >= 0) & (y < classes)) and (np.bincount(y, minlength=classes) > 0).all(),
                'native label support differs')
        labels[group] = y
    for group in GROUPS:
        batches = unit['report_batches'][group]
        require([p for batch in batches for p in batch] == unit['report'][group]
                and all(0 < len(batch) <= 16 for batch in batches), 'report batch ordering/size differs')
    fit_people = {rows[p]['speaker'] for p in unit['fit']}
    people = {g: {rows[p]['speaker'] for p in unit['report'][g]} for g in GROUPS}
    other = 'B' if unit['arm'] == 'A' else 'A'
    require(fit_people == people[unit['arm']], 'seen speakers differ from fit')
    require(not fit_people.intersection(people[other] | people['outer'])
            and not people[other].intersection(people['outer']), 'speaker leakage')
    require(not {rows[p]['sentence'] for p in unit['fit']}.intersection(
        {rows[p]['sentence'] for g in GROUPS for p in unit['report'][g]}), 'fit/query text overlap')
    return labels


def update_selection(best, selected, epoch, values):
    """Strict improvement preserves the earliest exact tie; never accepts outer."""
    require(set(values) == set(RULES), 'selection values differ')
    improved = [r for r in RULES if r not in best
                or (values[r] < best[r] if r.endswith('_ce') else values[r] > best[r])]
    for rule in improved:
        best[rule], selected[rule] = values[rule], epoch
    return improved


def _write_json(path, value):
    with path.open('xb') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False).encode('utf-8') + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def validate_checkpoint(unit, model_spec, checkpoint_path, model=None):
    """Return (CPU payload, byte SHA), validating saved states without inference.

    A provided model must be a fresh initial model; otherwise one is built on
    CPU. Selection against archived validation logits remains the runner's job.
    This function intentionally fails when the actual checkpoint is absent.
    """
    import torch
    identity = model_identity(model_spec)
    path = Path(checkpoint_path)
    sha = file_sha(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require(file_sha(path) == sha, 'checkpoint changed while loading')
    expected_keys = {'schema', 'unit_id', 'unit_sha256', 'n_classes', 'model_identity', 'head_seed',
                     'windows', 'epochs', 'selected_epochs', 'epoch_states', 'initial_state_sha256',
                     'initial_head_sha256',
                     'frozen_parameter_sha256', 'trainable_parameter_names', 'buffer_names',
                     'config_sha256', 'training_resume_supported'}
    require(set(saved) == expected_keys and saved['schema'] == CHECKPOINT_SCHEMA, 'checkpoint schema differs')
    require(saved['unit_id'] == unit['unit_id'] and saved['unit_sha256'] == unit_identity(unit)
            and saved['config_sha256'] == digest(unit['config']) and saved['model_identity'] == identity
            and saved['head_seed'] == unit['seeds']['head'] and saved['n_classes'] == unit['n_classes']
            and saved['windows'] == unit['windows'] and saved['epochs'] == unit['config']['epochs']
            and saved['training_resume_supported'] is False, 'checkpoint identity differs')
    selected = saved['selected_epochs']
    require(set(selected) == {str(w) for w in unit['windows']}, 'checkpoint windows differ')
    for window, rules in selected.items():
        require(set(rules) == set(RULES) | {'last'} and rules['last'] == int(window)
                and all(type(e) is int and 1 <= e <= int(window) for e in rules.values()),
                'invalid checkpoint selection epochs')
    epochs = {str(e) for rules in selected.values() for e in rules.values()}
    require(set(saved['epoch_states']) == epochs, 'checkpoint retained epoch coverage differs')
    if model is None:
        model, _ = build_model(model_spec, unit['seeds']['head'], unit['n_classes'])
    require(tensor_mapping_sha256(model.state_dict()) == saved['initial_state_sha256']
            and tensor_mapping_sha256(model.fc.state_dict()) == saved['initial_head_sha256']
            and frozen_hash(model) == saved['frozen_parameter_sha256'], 'checkpoint base initialization differs')
    names, buffers = delta_names(model)
    require(names == saved['trainable_parameter_names'] and buffers == saved['buffer_names'],
            'checkpoint trainable names differ')
    reference = model.state_dict()
    for delta in saved['epoch_states'].values():
        require(set(delta) == set(names) | set(buffers), 'delta coverage differs')
        for name, tensor in delta.items():
            require(isinstance(tensor, torch.Tensor) and tensor.shape == reference[name].shape
                    and tensor.dtype == reference[name].dtype and bool(torch.isfinite(tensor).all()),
                    'invalid delta tensor shape/dtype/values')
    return saved, sha


def train(unit, rows, audio_roots, model_spec, output_dir, device, *, cache=None):
    """Write checkpoint.pt, predictions.npz, history.json; return receipt fields.

    ``selected_epochs`` is {window-string: {four rules and last: epoch}}.
    The NPZ always has epochs=1..total and the same ten keys as the old engine.
    This saves evaluation deltas, NOT an optimizer/RNG training-resume snapshot.
    """
    import torch
    device, out = torch.device(device), Path(output_dir)
    require(device.type in ('cpu', 'cuda'), 'unsupported device')
    labels = validate_unit(unit, rows)
    identity = model_identity(model_spec)
    require(unit.get('model', identity['name']) == identity['name']
            and unit['config'].get('model', identity['name']) == identity['name'], 'unit backbone differs')
    require(not out.exists(), 'attempt output must be fresh')
    out.mkdir(parents=True)
    cfg, classes = unit['config'], unit['n_classes']
    began = time.perf_counter()
    cache = cache if cache is not None else WaveCache(audio_roots[unit['corpus']])
    model, initial_sha = build_model(model_spec, unit['seeds']['head'], classes)
    initial_head_sha = tensor_mapping_sha256(model.fc.state_dict())
    original_frozen = frozen_hash(model)
    names, buffers = delta_names(model)
    keys = set(names) | set(buffers)
    model.to(device)
    set_seed(unit['seeds']['torch_training'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.enc.parameters() if p.requires_grad], 'lr': cfg['lr_encoder']},
        {'params': model.fc.parameters(), 'lr': cfg['lr_head']}], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    order_rng = np.random.RandomState(unit['seeds']['order'])
    selected = {str(w): {} for w in unit['windows']}
    best = {str(w): {} for w in unit['windows']}
    selected_states = {str(w): {} for w in unit['windows']}
    history, all_logits = [], {g: [] for g in GROUPS}
    seen, unseen = unit['arm'], ('B' if unit['arm'] == 'A' else 'A')
    cap, crop = 160000, 48000
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order = order_rng.permutation(len(unit['fit']))
        total_loss, steps, skips = 0.0, 0, 0
        for start in range(0, len(order), cfg['batch_size']):
            indexes = order[start:start + cfg['batch_size']]
            paths = [unit['fit'][i] for i in indexes]
            starts = [crop_start(unit['seeds']['crop'], epoch, int(i), len(cache(path)))
                      for i, path in zip(indexes, paths)]
            waves = waveform_batch(paths, cache, cap, crop=crop, starts=starts).to(device)
            targets = torch.tensor(labels['fit'][indexes], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                loss = torch.nn.functional.cross_entropy(model(waves), targets)
            require(bool(torch.isfinite(loss)), 'nonfinite training loss')
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            skips += int(scaler.get_scale() < old_scale)
            steps += 1
            total_loss += float(loss.detach().cpu()) * len(paths)
        model.eval()
        before = rng_state(device)
        predictions = {g: predict_group(model, unit['report_batches'][g], cache, cap, classes, device)
                       for g in GROUPS}
        check_rng(before, device)
        measured = {g: metrics(predictions[g], labels[g], classes) for g in ('A', 'B')}
        values = {'seen_ce': measured[seen][0], 'unseen_ce': measured[unseen][0],
                  'seen_uar': measured[seen][1], 'unseen_uar': measured[unseen][1]}
        for group in GROUPS:
            all_logits[group].append(predictions[group])
        changed = {}
        for window in unit['windows']:
            if epoch <= window:
                key = str(window)
                changed[key] = update_selection(best[key], selected[key], epoch, values)
                if epoch == window:
                    selected[key]['last'] = epoch
                    changed[key].append('last')
        state = cpu_state(model, keys) if any(changed.values()) else None
        for window, rules in changed.items():
            for rule in rules:
                selected_states[window][rule] = state
        history.append(dict(epoch=epoch, train_loss=total_loss / len(order), optimizer_steps=steps,
                            scaler_skipped_steps=skips, **{k: float(v) for k, v in values.items()}))
    require(frozen_hash(model) == original_frozen, 'frozen encoder changed')
    states = {str(selected[w][r]): selected_states[w][r] for w in selected for r in selected[w]}
    checkpoint = dict(schema=CHECKPOINT_SCHEMA, unit_id=unit['unit_id'], unit_sha256=unit_identity(unit), n_classes=classes,
                      model_identity=identity, head_seed=unit['seeds']['head'], windows=unit['windows'],
                      epochs=cfg['epochs'], selected_epochs=selected, epoch_states=states,
                      initial_state_sha256=initial_sha, initial_head_sha256=initial_head_sha,
                      frozen_parameter_sha256=original_frozen,
                      trainable_parameter_names=names, buffer_names=buffers,
                      config_sha256=digest(cfg), training_resume_supported=False)
    with (out / 'checkpoint.pt').open('xb') as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    checkpoint_sha = file_sha(out / 'checkpoint.pt')
    del model, optimizer, checkpoint, states, selected_states, state
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    replay, replay_sha = build_model(model_spec, unit['seeds']['head'], classes)
    require(replay_sha == initial_sha, 'reconstructed initialization differs')
    saved, verified_sha = validate_checkpoint(unit, model_spec, out / 'checkpoint.pt', model=replay)
    require(verified_sha == checkpoint_sha and saved['selected_epochs'] == selected,
            'checkpoint identity changed on disk')
    replay.to(device).eval()
    differences, checked_values = {}, 0
    for epoch, delta in saved['epoch_states'].items():
        load_delta(replay, delta, original_frozen)
        for group in GROUPS:
            actual = predict_group(replay, unit['report_batches'][group], cache, cap, classes, device)
            expected = all_logits[group][int(epoch) - 1]
            diff = float(np.max(np.abs(actual - expected)))
            require(diff <= 1e-5, 'checkpoint replay logits differ')
            differences[f'{epoch}/{group}'] = diff
            checked_values += int(expected.size)
    require(file_sha(out / 'checkpoint.pt') == checkpoint_sha, 'checkpoint bytes changed during replay')
    arrays = {'epochs': np.arange(1, cfg['epochs'] + 1, dtype=np.int64)}
    for group in GROUPS:
        arrays[f'{group}__paths'] = np.asarray(unit['report'][group], dtype=str)
        arrays[f'{group}__labels'] = labels[group]
        arrays[f'{group}__all_epoch_logits'] = np.stack(all_logits[group])
    with (out / 'predictions.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    _write_json(out / 'history.json', history)
    info = dict(epochs=cfg['epochs'], windows=unit['windows'], selected_epochs=selected,
                unit_sha256=unit_identity(unit), initial_head_sha256=initial_head_sha,
                model_identity=identity, initial_state_sha256=initial_sha,
                frozen_parameter_sha256=original_frozen, checkpoint_sha256=checkpoint_sha,
                checkpoint_unique_epoch_count=len(saved['epoch_states']),
                checkpoint_unique_epochs=sorted(map(int, saved['epoch_states'])),
                reload_max_abs_diff=max(differences.values()), reload_by_epoch_group=differences,
                reload_checks=len(differences), reload_checked_values=checked_values,
                checkpoint_reload_verified=True, outer_scores_computed=False,
                optimizer_steps=sum(r['optimizer_steps'] for r in history),
                scaler_skipped_steps=sum(r['scaler_skipped_steps'] for r in history),
                optimizer_steps_scope='batch update attempts; skips inferred from GradScaler scale decrease',
                training_resume_supported=False, checkpoint_retained=True,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                report_batching='separate fixed A/B/outer groups',
                wall_seconds=time.perf_counter() - began)
    del replay, saved
    gc.collect()
    return info

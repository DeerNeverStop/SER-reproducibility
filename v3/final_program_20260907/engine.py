"""Native-label WavLM trajectory with four validation checkpoint rules.

Outer labels are stored, never scored here. Three fixed report groups have
separate batches, so outer waveform lengths cannot affect validation logits.
The caller owns frozen plans, source checks, attempts, receipts and ledgers.
"""
from __future__ import annotations

from fractions import Fraction
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

from v3.speaker_coverage.run import (
    WEIGHTS_SHA, cpu_state, delta_names, frozen_hash, load_delta,
    tensor_mapping_sha256,
)
from v3.deploy.engines_deploy import WaveCache, set_seed, state_dict_sha256
from .plan import crop_start

GROUPS = ('A', 'B', 'outer')
RULES = ('seen_ce', 'unseen_ce', 'seen_uar', 'unseen_uar')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_model(path, seed, classes):
    import torch
    import torchaudio
    require(classes in (6, 7, 8), 'unexpected native label count')
    set_seed(seed)
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    require(bundle._model_type == 'WavLM' and not bundle._normalize_waveform
            and bundle.sample_rate == 16000, 'bundle semantics differ')
    encoder = torchaudio.models.wavlm_model(**bundle._params)
    encoder.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    require(state_dict_sha256(encoder) == WEIGHTS_SHA, 'pretrained encoder identity differs')
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for layer in encoder.encoder.transformer.layers[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

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


def waveform_batch(paths, cache, cap, *, crop=None, starts=None):
    import torch
    waves = []
    require((crop is None) == (starts is None), 'crop contract differs')
    require(starts is None or len(starts) == len(paths), 'crop coordinate count differs')
    for i, path in enumerate(paths):
        wav = cache(path)
        require(len(wav) > 0 and np.isfinite(wav).all(), 'invalid processed audio')
        if crop is not None:
            require(type(starts[i]) is int and 0 <= starts[i] <= max(0, len(wav) - crop), 'invalid crop start')
        if crop is not None and len(wav) > crop:
            offset = starts[i]
            wav = wav[offset:offset + crop]
        elif crop is not None:
            wav = np.pad(wav, (0, crop - len(wav)))
        elif crop is None:
            wav = wav[:cap]
        waves.append(torch.as_tensor(wav, dtype=torch.float32))
    return torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)


def metrics(logits, labels, classes):
    """Float64 CE and exact rational UAR tie ordering, validation only."""
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    require(values.shape == (len(labels), classes) and np.isfinite(values).all(), 'invalid logits')
    require(np.all((labels >= 0) & (labels < classes)), 'invalid labels')
    support = np.bincount(labels, minlength=classes)
    require((support > 0).all(), 'missing validation class')
    maximum = values.max(axis=1)
    lse = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    ce = float(np.mean(lse - values[np.arange(len(labels)), labels]))
    pred = values.argmax(axis=1)
    uar = sum((Fraction(int(((pred == c) & (labels == c)).sum()), int(support[c]))
               for c in range(classes)), Fraction(0)) / classes
    require(math.isfinite(ce), 'nonfinite CE')
    return ce, uar


def predict_group(model, batches, cache, cap, classes, device):
    import torch
    values = []
    with torch.no_grad():
        for paths in batches:
            waves = waveform_batch(paths, cache, cap).to(device)
            logits = model(waves).detach().cpu().numpy()
            require(logits.shape == (len(paths), classes) and np.isfinite(logits).all(), 'prediction differs')
            values.append(logits)
    return np.concatenate(values)


def rng_state(device):
    import torch
    return (random.getstate(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def check_rng(before, device):
    import torch
    after = rng_state(device)
    require(before[0] == after[0] and torch.equal(before[1], after[1])
            and (before[2] is None or torch.equal(before[2], after[2])), 'evaluation changed training RNG')


def validate_unit(unit, rows):
    classes = unit['n_classes']
    require(classes in (6, 7, 8) and unit['arm'] in ('A', 'B'), 'invalid unit')
    require(unit['selection_enabled'] is (unit['arm'] == 'A'), 'selection role contract differs')
    require(unit['prediction_epochs'] == (list(range(1, 16)) if unit['selection_enabled'] else [15]),
            'prediction epoch contract differs')
    occupied = set()
    labels = {}
    for group, paths in [('fit', unit['fit']), *unit['report'].items()]:
        require(paths and len(paths) == len(set(paths)), 'empty or duplicate role')
        require(not occupied.intersection(paths), 'audio role overlap')
        occupied.update(paths)
        require(all(path in rows for path in paths), 'unknown audio path')
        y = np.asarray([int(rows[p]['label_index']) for p in paths], dtype=np.int64)
        require(np.all((y >= 0) & (y < classes)) and (np.bincount(y, minlength=classes) > 0).all(), 'label support')
        labels[group] = y
    require(set(unit['report']) == set(GROUPS), 'report groups differ')
    for group in GROUPS:
        batches = unit['report_batches'][group]
        require([p for batch in batches for p in batch] == unit['report'][group], 'report batch ordering differs')
        require(all(0 < len(b) <= 16 for b in batches), 'report batch size differs')
    fit_people = {rows[p]['speaker'] for p in unit['fit']}
    people = {g: {rows[p]['speaker'] for p in unit['report'][g]} for g in GROUPS}
    other = 'B' if unit['arm'] == 'A' else 'A'
    require(fit_people == people[unit['arm']], 'seen speakers differ from fit')
    require(not fit_people.intersection(people[other] | people['outer'])
            and not people[other].intersection(people['outer']), 'speaker leakage')
    require(not {rows[p]['sentence'] for p in unit['fit']}.intersection(
            {rows[p]['sentence'] for g in GROUPS for p in unit['report'][g]}), 'fit/query text overlap')
    return labels


def fit_unit(unit, rows, audio_root, model_path, out, device, cache=None):
    import torch
    labels = validate_unit(unit, rows)
    cfg, classes = unit['config'], unit['n_classes']
    require(cfg['epochs'] == 15 and cfg['batch_size'] == 16 and cfg['fp16'] is True,
            'frozen training settings differ')
    require(not out.exists(), 'attempt output must be fresh')
    out.mkdir(parents=True)
    began = time.perf_counter()
    cache = cache if cache is not None else WaveCache(audio_root, 16000)
    model, initial_sha = build_model(model_path, unit['seeds']['initialization'], classes)
    original_frozen = frozen_hash(model)
    names, buffers = delta_names(model)
    keep = set(names) | set(buffers)
    model.to(device)
    set_seed(unit['seeds']['torch_training'])
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.enc.parameters() if p.requires_grad], 'lr': cfg['lr_encoder']},
        {'params': model.fc.parameters(), 'lr': cfg['lr_head']}], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    order_rng = np.random.RandomState(unit['seeds']['order'])
    rules = RULES if unit['selection_enabled'] else ()
    best = {r: None for r in rules}
    selected, selected_states, history = {}, {}, []
    all_logits = {g: [] for g in GROUPS}
    seen, unseen = unit['arm'], ('B' if unit['arm'] == 'A' else 'A')
    cap, crop = int(cfg['eval_cap_seconds'] * 16000), int(cfg['crop_seconds'] * 16000)
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
        values = {}
        if epoch in unit['prediction_epochs']:
            model.eval()
            before = rng_state(device)
            predictions = {g: predict_group(model, unit['report_batches'][g], cache, cap, classes, device)
                           for g in GROUPS}
            check_rng(before, device)
            if unit['selection_enabled']:
                measured = {g: metrics(predictions[g], labels[g], classes) for g in ('A', 'B')}
                values = {'seen_ce': measured[seen][0], 'unseen_ce': measured[unseen][0],
                          'seen_uar': measured[seen][1], 'unseen_uar': measured[unseen][1]}
            for group in GROUPS:
                all_logits[group].append(predictions[group])
        improved = [r for r in rules if best[r] is None
                    or (values[r] < best[r] if r.endswith('_ce') else values[r] > best[r])]
        state = cpu_state(model, keep) if improved or epoch == cfg['epochs'] else None
        for rule in improved:
            best[rule], selected[rule], selected_states[rule] = values[rule], epoch, state
        if epoch == cfg['epochs']:
            selected['last'], selected_states['last'] = epoch, state
        history.append(dict(epoch=epoch, train_loss=total_loss / len(order), optimizer_steps=steps,
                            scaler_skipped_steps=skips, **{k: float(v) for k, v in values.items()}))
    require(frozen_hash(model) == original_frozen, 'frozen encoder changed')
    states = {str(selected[r]): selected_states[r] for r in selected}
    checkpoint = dict(schema='ser-final-program-delta-1', unit_id=unit['unit_id'], n_classes=classes,
                      initial_state_sha256=initial_sha, frozen_parameter_sha256=original_frozen,
                      base_state_sha256=WEIGHTS_SHA, selected_epochs=selected,
                      trainable_parameter_names=names, buffer_names=buffers, epoch_states=states)
    with (out / 'checkpoint.pt').open('xb') as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    del model, optimizer, checkpoint, states, selected_states, state
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    saved = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=True)
    replay, replay_sha = build_model(model_path, unit['seeds']['initialization'], classes)
    require(replay_sha == initial_sha, 'reconstructed initialization differs')
    replay.to(device).eval()
    differences = {}
    for epoch, delta in saved['epoch_states'].items():
        load_delta(replay, delta, original_frozen)
        for group in GROUPS:
            actual = predict_group(replay, unit['report_batches'][group], cache, cap, classes, device)
            expected = all_logits[group][unit['prediction_epochs'].index(int(epoch))]
            diff = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            require(diff <= 1e-5, 'checkpoint replay logits differ')
            differences[f'{epoch}/{group}'] = diff
    arrays = {'epochs': np.asarray(unit['prediction_epochs'], dtype=np.int64)}
    for group in GROUPS:
        arrays[f'{group}__paths'] = np.asarray(unit['report'][group], dtype=str)
        arrays[f'{group}__labels'] = labels[group]
        arrays[f'{group}__all_epoch_logits'] = np.stack(all_logits[group])
    np.savez_compressed(out / 'predictions.npz', **arrays)
    (out / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    info = dict(selected_epochs=selected, epochs=cfg['epochs'], initial_state_sha256=initial_sha,
                frozen_parameter_sha256=original_frozen, reload_max_abs_diff=max(differences.values()),
                reload_by_epoch_group=differences, outer_scores_computed=False,
                report_batching='separate fixed A/B/outer groups; outer never pads a validation batch',
                wall_seconds=time.perf_counter() - began)
    del replay, saved
    gc.collect()
    return info

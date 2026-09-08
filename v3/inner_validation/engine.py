"""One FT trajectory, two validation selections, lossless epoch-delta replay.

No accuracy, UAR, HPO selection, test ranking, or network calls are performed.
The caller owns plan/source verification, attempts, timing, memory and DONE.
"""
from __future__ import annotations

import gc
import math
import os
import random
from pathlib import Path

import numpy as np

from v3.speaker_coverage import run as legacy

SCHEMA = "ser-inner-validation-ft-delta-1"
ROLES = ("val_seen", "val_unseen", "test")
CHECKPOINTS = ("best_seen", "best_unseen", "last")
PREDICTION_KEYS = ({f"{r}__{c}__logits" for r in ROLES for c in CHECKPOINTS}
                   | {f"{r}__{k}" for r in ROLES for k in ("paths", "labels")})
require = legacy.require
build_ft_model = legacy.build_ft_model
delta_names = legacy.delta_names
cpu_state = legacy.cpu_state
load_delta = legacy.load_delta
predict_wave = legacy.predict_wave
frozen_hash = legacy.frozen_hash


def validate_unit(unit, rows):
    cfg = unit["config"]
    fixed = {"epochs": 15, "batch_size": 16, "fp16": True,
             "early_stopping": False, "weight_decay": 0.01,
             "crop_seconds": 3.0, "eval_cap_seconds": 10.0}
    require(all(cfg.get(k) == v for k, v in fixed.items()), "fixed FT configuration differs")
    require(cfg.get("fp16") is True and cfg.get("early_stopping") is False,
            "invalid FT boolean configuration")
    require(cfg.get("lr_encoder") in (1e-5, 5e-5) and cfg.get("lr_head") in (3e-4, 1e-3),
            "learning rate outside frozen candidate grid")
    require(cfg.get("trainable_layers") == "top4+head", "FT must use top4+head")
    require(type(unit.get("train_seed")) is int and isinstance(unit.get("unit_id"), str)
            and unit["unit_id"], "unit seed/identity missing")
    labels, occupied = {}, set()
    for role in ("fit", *ROLES):
        paths = unit[role]
        require(isinstance(paths, list) and paths and all(isinstance(p, str) for p in paths)
                and len(paths) == len(set(paths)), "empty, duplicated or invalid role paths")
        require(not occupied.intersection(paths), "audio paths cross roles")
        occupied.update(paths)
        require(all(p in rows for p in paths), "unknown audio metadata path")
        y = np.asarray([int(rows[p]["label_index"]) for p in paths], dtype=np.int64)
        require(np.all((y >= 0) & (y < 6)), "invalid six-class labels")
        if role != "test":
            counts = np.bincount(y, minlength=6)
            require((counts > 0).all() and len(set(counts.tolist())) == 1,
                    "fit and both validation roles must be class-balanced")
        labels[role] = y
    return labels


def _rng_snapshot(batch, device):
    import torch
    return (random.getstate(), np.random.get_state(), batch.rng.get_state(),
            torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None)


def _same_numpy_state(left, right):
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


def _require_rng_unchanged(before, batch, device):
    import torch
    after = _rng_snapshot(batch, device)
    require(before[0] == after[0] and _same_numpy_state(before[1], after[1])
            and _same_numpy_state(before[2], after[2]) and torch.equal(before[3], after[3])
            and (before[4] is None or torch.equal(before[4], after[4])),
            "validation consumed training random state")


def _validation_loss(model, paths, labels, batch, device):
    import torch
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(paths), 16):
            logits = model(batch(paths[start:start + 16], False).to(device))
            target = torch.tensor(labels[start:start + 16], dtype=torch.long, device=device)
            total += float(torch.nn.functional.cross_entropy(logits, target, reduction="sum").cpu())
    value = total / len(paths)
    require(math.isfinite(value), "non-finite validation loss")
    return value


def _optimizer_step(loss, optimizer, scaler):
    """Return whether GradScaler skipped this attempted update (scale decreased)."""
    previous = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return int(scaler.get_scale() < previous)


def full_dual_loop(model, unit, rows, batch, device):
    """Return epoch->delta, checkpoint->epoch, 15-row history, frozen hash.

    optimizer_steps counts attempted minibatch updates; scaler_skipped_steps
    separately records overflows inferred from a decrease of GradScaler scale.
    """
    import torch
    from v3.deploy.engines_deploy import set_seed
    labels = validate_unit(unit, rows)
    cfg = unit["config"]
    set_seed(unit["train_seed"])
    model.to(device)
    require(all(p.dtype == torch.float32 for p in model.parameters()), "FT parameters must remain FP32")
    initial_frozen = frozen_hash(model)
    trainable, buffers = delta_names(model)
    keys = set(trainable) | set(buffers)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": cfg["lr_encoder"]},
        {"params": model.fc.parameters(), "lr": cfg["lr_head"]}], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss()
    order = np.arange(len(unit["fit"]))
    best_loss = {"val_seen": math.inf, "val_unseen": math.inf}
    best_epochs, best_states, history = {}, {}, []
    last_state = None
    for epoch in range(1, 16):
        model.train()
        batch.rng.shuffle(order)
        total, steps, skipped = 0.0, 0, 0
        for start in range(0, len(order), 16):
            indexes = order[start:start + 16]
            paths = [unit["fit"][i] for i in indexes]
            xb = batch(paths, True).to(device)
            yb = torch.tensor(labels["fit"][indexes], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            require(bool(torch.isfinite(loss)), "non-finite training loss")
            skipped += _optimizer_step(loss, optimizer, scaler)
            total += float(loss.detach().cpu()) * len(paths)
            steps += 1
        model.eval()
        rng_before = _rng_snapshot(batch, device)
        losses = {r: _validation_loss(model, unit[r], labels[r], batch, device)
                  for r in ("val_seen", "val_unseen")}
        _require_rng_unchanged(rng_before, batch, device)
        history.append({"epoch": epoch, "train_loss": total / len(order),
                        "val_seen_loss": losses["val_seen"], "val_unseen_loss": losses["val_unseen"],
                        "optimizer_steps": steps, "scaler_skipped_steps": skipped})
        improved = [r for r in losses if losses[r] < best_loss[r]]
        snapshot = cpu_state(model, keys) if improved or epoch == 15 else None
        for role in improved:
            best_loss[role] = losses[role]
            best_epochs[role], best_states[role] = epoch, snapshot
        if epoch == 15:
            last_state = snapshot
    require(frozen_hash(model) == initial_frozen, "FT changed frozen parameters")
    require(set(best_epochs) == {"val_seen", "val_unseen"} and last_state is not None,
            "missing validation selection or last checkpoint")
    selected = {"best_seen": best_epochs["val_seen"], "best_unseen": best_epochs["val_unseen"], "last": 15}
    states = {str(best_epochs[r]): best_states[r] for r in ("val_seen", "val_unseen")}
    states["15"] = last_state
    return states, selected, history, initial_frozen


def _make_batch(unit, audio_root, wav_cache):
    from v3.deploy.engines_deploy import WaveCache, wavlm_batch
    wav = wav_cache if wav_cache is not None else WaveCache(audio_root, 16000)
    rng = np.random.RandomState(unit["train_seed"])
    def batch(paths, train):
        return wavlm_batch(paths, train, wav, 48000, 160000, rng)
    batch.rng = rng
    return batch


def _epoch_predictions(model, states, unit, batch, device, frozen):
    predictions = {}
    for epoch in sorted(states, key=int):
        load_delta(model, states[epoch], frozen)
        for role in ROLES:
            values = np.asarray(predict_wave(model, unit[role], batch, device), dtype=np.float64)
            require(values.shape == (len(unit[role]), 6) and np.isfinite(values).all(),
                    "invalid role prediction shape or nonfinite logits")
            predictions[epoch, role] = values
    return predictions


def _validate_saved(saved, model, unit, selected, params_sha, base_sha, frozen):
    import torch
    require(saved.get("schema") == SCHEMA and saved.get("unit_id") == unit["unit_id"]
            and saved.get("config") == unit["config"] and saved.get("train_seed") == unit["train_seed"]
            and saved.get("epochs_run") == 15 and saved.get("selected_epochs") == selected,
            "checkpoint identity or selected epoch mismatch")
    require(saved.get("base_file_sha256") == base_sha and saved.get("base_state_sha256") == legacy.WEIGHTS_SHA
            and saved.get("bundle_params_sha256") == params_sha
            and saved.get("frozen_parameter_sha256") == frozen and frozen_hash(model) == frozen,
            "checkpoint base/bundle/frozen identity mismatch")
    trainable, buffers = delta_names(model)
    require(saved.get("trainable_parameter_names") == trainable and saved.get("buffer_names") == buffers,
            "checkpoint parameter/buffer inventory mismatch")
    states = saved.get("epoch_states", {})
    require(set(states) == {str(e) for e in selected.values()}, "checkpoint unique epoch inventory mismatch")
    template = model.state_dict()
    keys = set(trainable) | set(buffers)
    for state in states.values():
        require(set(state) == keys, "checkpoint delta coverage mismatch")
        for name, value in state.items():
            target = template[name]
            require(isinstance(value, torch.Tensor) and value.dtype == target.dtype and value.shape == target.shape,
                    "checkpoint tensor shape or lossless dtype mismatch")
            require(bool(torch.isfinite(value).all()), "checkpoint tensor is nonfinite")


def fit_dual(unit, rows, model_path, audio_root, device, checkpoint_path, wav_cache=None):
    """Fit once; seal three selected checkpoints and all nine role predictions."""
    import torch
    labels = validate_unit(unit, rows)
    model_path, checkpoint_path = Path(model_path), Path(checkpoint_path)
    require(not checkpoint_path.exists(), "checkpoint already exists; use a fresh attempt")
    base_sha = legacy.file_sha(model_path)
    batch = _make_batch(unit, audio_root, wav_cache)
    model, params_sha = build_ft_model(model_path, unit["train_seed"])
    states, selected, history, frozen = full_dual_loop(model, unit, rows, batch, device)
    original = _epoch_predictions(model, states, unit, batch, device, frozen)
    trainable, buffers = delta_names(model)
    checkpoint = {"schema": SCHEMA, "unit_id": unit["unit_id"], "config": unit["config"],
                  "train_seed": unit["train_seed"], "epochs_run": len(history), "selected_epochs": selected,
                  "base_file_sha256": base_sha, "base_state_sha256": legacy.WEIGHTS_SHA,
                  "bundle_params_sha256": params_sha, "frozen_parameter_sha256": frozen,
                  "trainable_parameter_names": trainable, "buffer_names": buffers, "epoch_states": states}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    del model, states, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(legacy.file_sha(model_path) == base_sha, "base file changed during fit")
    replay, replay_params_sha = build_ft_model(model_path, unit["train_seed"])
    require(replay_params_sha == params_sha, "model constructor bundle changed")
    _validate_saved(saved, replay, unit, selected, params_sha, base_sha, frozen)
    replay.to(device)
    restored = _epoch_predictions(replay, saved["epoch_states"], unit, batch, device, frozen)
    differences = {key: legacy.reload_difference(original[key], restored[key]) for key in original}
    arrays, named_differences = {}, {}
    for role in ROLES:
        arrays[f"{role}__paths"] = np.asarray(unit[role], dtype=str)
        arrays[f"{role}__labels"] = labels[role]
        for name in CHECKPOINTS:
            key = f"{role}__{name}__logits"
            epoch = str(selected[name])
            arrays[key] = original[epoch, role].copy()
            named_differences[key] = differences[epoch, role]
    require(set(arrays) == PREDICTION_KEYS, "prediction schema mismatch")
    unique_epochs = sorted(set(selected.values()))
    info = {"epochs_run": len(history), "best_seen_epoch": selected["best_seen"],
            "best_unseen_epoch": selected["best_unseen"], "checkpoint_unique_epochs": unique_epochs,
            "checkpoint_unique_epoch_count": len(unique_epochs), "frozen_parameter_sha256": frozen,
            "checkpoint_format": "wavlm_dual_validation_epoch_deltas", "checkpoint_reload_verified": True,
            "checkpoint_reload_max_abs_diff": max(differences.values()),
            "checkpoint_reload_max_abs_diff_by_prediction": named_differences}
    del replay, saved, restored
    return arrays, history, info

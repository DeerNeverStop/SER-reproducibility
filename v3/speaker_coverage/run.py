"""Pinned coverage-study runner. Predictions are sealed; no test scores are computed.

Output: OUT/PHASE/units/UID/attempts/NNNN/{predictions,checkpoint,history,receipt}.
Only a hash-verified DONE commits an attempt. Pilot and formal never share output.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from v3.data_design import core_run as core

PROGRAM = "SER26-SPEAKER-COVERAGE-1"
SCHEMA = "ser-speaker-coverage-1"
IntegrityError = core.IntegrityError
require, digest, file_sha = core.require, core.digest, core.file_sha
read_json, atomic_json, relative_file = core.read_json, core.atomic_json, core.relative_file
WEIGHTS_SHA = "d8e745fb761c56d209431413bcc373f92784dc2c8d850199b6caa5cf05ca2c36"
MODELS = ("cnn", "ridge_wavlm", "wavlm_ft")
PREDICTION_KEYS = {"paths", "labels", "logits", "pred", "proba",
                   "last_logits", "last_pred", "last_proba"}


def phase_units(plan, phase):
    require(phase in {"pilot", "formal"}, "invalid phase")
    if phase == "formal":
        return plan["units"]
    selected = [u for u in plan["units"] if u["fold"] == u["rotation"] == 0 and
                (u["block"] == "core" or
                 (u["draw"] == 0 and u["policy"] == "U" and u.get("seed_index", 0) == 0))]
    require(Counter(u["block"] for u in selected) == {"core": 18, "ft": 1},
            "pilot must be the fixed 18 core units and one full FT probe")
    return selected


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.ndim == 2 and scores.shape[1] == 6 and np.isfinite(scores).all(),
            "nonfinite or invalid classification scores")
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def prediction_arrays(unit, rows, logits, last_logits):
    arrays = {"paths": np.asarray(unit["test"], dtype=str),
              "labels": np.asarray([int(rows[p]["label_index"]) for p in unit["test"]], dtype=np.int64)}
    for prefix, value in (("", logits), ("last_", last_logits)):
        value = np.asarray(value, dtype=np.float64)
        require(value.shape == (len(unit["test"]), 6), "prediction count mismatch")
        arrays[prefix + "logits"] = value
        arrays[prefix + "proba"] = softmax(value)
        arrays[prefix + "pred"] = value.argmax(axis=1).astype(np.int64)
    return arrays


def validate_predictions(path, unit, rows=None):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == PREDICTION_KEYS, "unexpected prediction fields")
        require(data["paths"].dtype.kind == "U" and data["paths"].tolist() == unit["test"],
                "prediction path order differs from plan")
        labels = data["labels"]
        require(labels.shape == (len(unit["test"]),) and labels.dtype == np.int64
                and np.isin(labels, np.arange(6)).all(), "invalid sealed labels")
        if rows is not None:
            require(labels.tolist() == [int(rows[p]["label_index"]) for p in unit["test"]],
                    "sealed labels differ from pinned manifest")
        for prefix in ("", "last_"):
            logits, pred, proba = (data[prefix + name] for name in ("logits", "pred", "proba"))
            require(logits.shape == (len(labels), 6) and logits.dtype == np.float64,
                    "invalid score shape/dtype")
            require(pred.dtype == np.int64 and pred.shape == labels.shape and
                    np.array_equal(pred, logits.argmax(1)), "prediction disagrees with scores")
            require(proba.dtype == np.float64 and proba.shape == logits.shape and
                    np.isfinite(proba).all() and np.allclose(proba, softmax(logits), rtol=1e-12, atol=1e-12),
                    "probabilities disagree with scores")


def verify_completed(unit, phase_out, plan_sha, phase, rows=None):
    directory = Path(phase_out) / "units" / unit["unit_id"]
    done_path = directory / "DONE"
    if not done_path.exists():
        return False
    done = read_json(done_path)
    expected = {"schema": "ser-speaker-coverage-done-1", "unit_id": unit["unit_id"],
                "plan_sha256": plan_sha, "phase": phase, "model": unit["model"], "block": unit["block"]}
    require(all(done.get(k) == v for k, v in expected.items()), "DONE identity mismatch")
    require(type(done.get("attempt")) is int and done["attempt"] > 0, "invalid DONE attempt")
    prefix = f"attempts/{done['attempt']:04d}/"
    artifacts = done.get("artifacts", {})
    checkpoint = "checkpoint.npz" if unit["model"] == "ridge_wavlm" else "checkpoint.pt"
    require(set(artifacts) == {prefix + n for n in ("predictions.npz", "receipt.json", "history.json", checkpoint)},
            "incomplete committed artifact set")
    for name, sha in artifacts.items():
        p = relative_file(directory, name)
        require(p.is_file() and file_sha(p) == sha, f"completed artifact corrupt: {name}")
    receipt = read_json(directory / prefix / "receipt.json")
    require(receipt.get("schema") == "ser-speaker-coverage-result-1" and
            all(receipt.get(k) == v for k, v in expected.items() if k != "schema") and
            receipt.get("attempt") == done["attempt"] and
            receipt.get("unit_config_sha256") == digest(unit["config"]), "receipt identity mismatch")
    require(receipt.get("checkpoint_reload_verified") is True, "checkpoint was not reloaded")
    epochs = 0 if unit["model"] == "ridge_wavlm" else unit["config"]["epochs"]
    require(receipt.get("epochs_run") == epochs, "incomplete fixed epoch budget")
    history = read_json(directory / prefix / "history.json")
    require([r.get("epoch") for r in history] == list(range(1, epochs + 1)), "history epochs missing")
    require(all(not any("uar" in key or "accuracy" in key for key in row) for row in history),
            "scientific scores must not be exported by runner")
    if epochs:
        best_epoch = min(history, key=lambda r: r["val_loss"])["epoch"]
        require(receipt.get("best_epoch") == best_epoch, "checkpoint selection differs from minimum validation loss")
    for name in ("fit_seconds", "wall_seconds", "peak_cuda_bytes"):
        require(type(receipt.get(name)) in (int, float) and math.isfinite(receipt[name]) and receipt[name] >= 0,
                "invalid resource receipt")
    validate_predictions(directory / prefix / "predictions.npz", unit, rows)
    return True


def write_completed(unit, phase_out, plan, phase, attempt, arrays, history, info, environment):
    directory = Path(phase_out) / "units" / unit["unit_id"]
    attempt_dir = directory / "attempts" / f"{attempt:04d}"
    require(not (directory / "DONE").exists(), "refusing to overwrite committed unit")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    pred_path = attempt_dir / "predictions.npz"
    with pred_path.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    validate_predictions(pred_path, unit)
    atomic_json(attempt_dir / "history.json", history)
    receipt = {"schema": "ser-speaker-coverage-result-1", "unit_id": unit["unit_id"],
               "plan_sha256": plan["plan_sha256"], "phase": phase, "model": unit["model"],
               "block": unit["block"], "attempt": attempt, "unit_config_sha256": digest(unit["config"]),
               "input_sha256": digest(plan["input"]), "environment": environment,
               "finished_at": core.now(), "proba_semantics": "softmax_scores_not_calibrated", **info}
    atomic_json(attempt_dir / "receipt.json", receipt)
    checkpoint = "checkpoint.npz" if unit["model"] == "ridge_wavlm" else "checkpoint.pt"
    names = ("predictions.npz", "receipt.json", "history.json", checkpoint)
    require(all((attempt_dir / name).is_file() for name in names), "missing checkpoint or result artifact")
    artifacts = {p.relative_to(directory).as_posix(): file_sha(p) for p in (attempt_dir / n for n in names)}
    atomic_json(directory / "DONE", {"schema": "ser-speaker-coverage-done-1", "unit_id": unit["unit_id"],
                "plan_sha256": plan["plan_sha256"], "phase": phase, "model": unit["model"],
                "block": unit["block"], "attempt": attempt, "artifacts": artifacts})
    require(verify_completed(unit, phase_out, plan["plan_sha256"], phase), "commit verification failed")


def finish_blocks(plan, phase_out, phase, rows=None):
    units = phase_units(plan, phase)
    counts = {}
    for block in ("core", "ft"):
        block_units = [u for u in units if u["block"] == block]
        done = {u["unit_id"]: file_sha(Path(phase_out) / "units" / u["unit_id"] / "DONE")
                for u in block_units if verify_completed(u, phase_out, plan["plan_sha256"], phase, rows)}
        counts[block] = {"complete": len(done), "expected": len(block_units)}
        target = Path(phase_out) / f"completion_{block}.json"
        if len(done) != len(block_units):
            require(not target.exists(), "completion marker exists for incomplete/corrupt block")
            continue
        marker = {"schema": "ser-speaker-coverage-block-complete-1", "plan_sha256": plan["plan_sha256"],
                  "phase": phase, "block": block, "count": len(done), "done_sha256": done,
                  "scores_computed": False, "independent_verification_required": True}
        if target.exists():
            require(read_json(target) == marker, "completion identity changed")
        else:
            atomic_json(target, marker)
    return counts


def cpu_state(model, keys=None):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if keys is None or k in keys}


def reload_difference(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    require(expected.shape == actual.shape and np.isfinite(actual).all(), "invalid checkpoint replay")
    require(np.allclose(expected, actual, rtol=1e-5, atol=1e-5), "checkpoint replay differs from original logits")
    return float(np.max(np.abs(expected - actual), initial=0.0))


def reload_info(best_diff, last_diff, checkpoint_format):
    return {"checkpoint_format": checkpoint_format, "checkpoint_reload_verified": True,
            "checkpoint_reload_best_max_abs_diff": best_diff,
            "checkpoint_reload_last_max_abs_diff": last_diff,
            "checkpoint_reload_max_abs_diff": max(best_diff, last_diff)}


def full_cnn_loop(model, x_fit, y_fit, x_val, y_val, cfg, seed, device, augment=None):
    """Frozen P1 optimizer/augmentation semantics, with an unconditional epoch budget."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from v3.deploy.engines_deploy import set_seed
    set_seed(seed)
    model.to(device)
    x_fit, x_val = (torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in (x_fit, x_val))
    y_fit_tensor = torch.from_numpy(np.asarray(y_fit, dtype=np.int64))
    y_val_tensor = torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_fit, y_fit_tensor), batch_size=cfg["batch_size"],
                        shuffle=True, generator=generator, num_workers=0)
    counts = np.bincount(y_fit, minlength=6).astype(np.float64)
    require((counts > 0).all(), "training set lacks an emotion")
    weight = torch.tensor(counts.sum() / counts / 6, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    best_loss, best_state, best_epoch, history = math.inf, None, None, []
    rng = np.random.RandomState(seed)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total_weight, steps = 0.0, 0.0, 0
        for xb, yb in loader:
            if augment is not None:
                xb = augment(xb, rng)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            require(bool(torch.isfinite(loss)), "non-finite training loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += float(F.cross_entropy(logits.detach(), yb, weight=weight, reduction="sum").cpu())
                total_weight += float(weight[yb].sum().cpu())
            steps += 1
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(x_val.to(device)), y_val_tensor, weight=weight).cpu())
        require(math.isfinite(val_loss), "non-finite validation loss")
        history.append({"epoch": epoch, "train_loss": total_loss / total_weight,
                        "val_loss": val_loss, "optimizer_steps": steps})
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, cpu_state(model)
    require(best_state is not None, "no validation checkpoint")
    return best_state, cpu_state(model), history, best_epoch


def fit_cnn(unit, rows, features, device, checkpoint_path):
    import torch
    import advanced_models
    from v3.deploy.engines_deploy import p1_arch, predict_torch, set_seed, spec_augment
    cfg, seed = unit["config"], unit["train_seed"]
    require(cfg["epochs"] == 100 and cfg["early_stopping"] is False, "CNN must execute all 100 epochs")
    labels = core.StrictLabels(rows)
    x_fit, x_val, x_test = (features.get("cremad", "logmel", unit[role]) for role in ("fit", "val", "test"))
    arch, arch_source = p1_arch(cfg)
    def build():
        set_seed(seed)
        return advanced_models.build_neural_model("cnn", n_mels=64, n_classes=6, config=arch)
    model = build()
    augment = lambda xb, rng: spec_augment(xb, rng, arch["time_mask"], arch["freq_mask"], arch["noise_std"])
    best, last, history, best_epoch = full_cnn_loop(
        model, x_fit, labels.labels("cremad", unit["fit"]), x_val,
        labels.labels("cremad", unit["val"]), cfg, seed, device, augment)
    last_logits = predict_torch(model, x_test, device)
    model.load_state_dict(best, strict=True)
    best_logits = predict_torch(model, x_test, device)
    checkpoint = {"schema": "ser-speaker-coverage-cnn-checkpoint-1", "config": cfg,
                  "train_seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
                  "best_state": best, "last_state": last}
    torch.save(checkpoint, checkpoint_path)
    del model
    gc.collect()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    replay = build().to(device)
    replay.load_state_dict(saved["best_state"], strict=True)
    best_diff = reload_difference(best_logits, predict_torch(replay, x_test, device))
    replay.load_state_dict(saved["last_state"], strict=True)
    last_diff = reload_difference(last_logits, predict_torch(replay, x_test, device))
    del replay
    return best_logits, last_logits, history, {"epochs_run": len(history), "best_epoch": best_epoch,
            "arch": arch, "arch_source": arch_source, **reload_info(best_diff, last_diff, "cnn_full_best_last")}


def fit_ridge(unit, rows, features, checkpoint_path):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    cfg = unit["config"]
    x_fit = features.get("cremad", "wavlm_base_plus", unit["fit"], 12)
    x_test = features.get("cremad", "wavlm_base_plus", unit["test"], 12)
    y_fit = core.StrictLabels(rows).labels("cremad", unit["fit"])
    scaler = StandardScaler().fit(x_fit)
    model = RidgeClassifier(alpha=cfg["alpha"], class_weight=cfg["class_weight"],
                            solver=cfg["solver"], tol=cfg["tol"], random_state=unit["train_seed"])
    model.fit(scaler.transform(x_fit), y_fit)
    require(model.classes_.tolist() == list(range(6)), "Ridge class order differs")
    logits = np.asarray(model.decision_function(scaler.transform(x_test)), dtype=np.float64)
    np.savez(checkpoint_path, coef=model.coef_, intercept=model.intercept_, classes=model.classes_,
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_,
             n_features_in=np.asarray(scaler.n_features_in_, dtype=np.int64),
             n_samples_seen=np.asarray(scaler.n_samples_seen_, dtype=np.int64))
    with np.load(checkpoint_path, allow_pickle=False) as saved:
        replay_scaler = StandardScaler()
        replay_scaler.mean_, replay_scaler.scale_, replay_scaler.var_ = (saved[k] for k in ("scaler_mean", "scaler_scale", "scaler_var"))
        replay_scaler.n_features_in_, replay_scaler.n_samples_seen_ = int(saved["n_features_in"]), int(saved["n_samples_seen"])
        transformed = replay_scaler.transform(x_test)
        replay_logits = transformed @ saved["coef"].T + saved["intercept"]
        require(saved["classes"].tolist() == list(range(6)), "saved Ridge class order differs")
        diff = reload_difference(logits, replay_logits)
    return logits, logits.copy(), [], {"epochs_run": 0, "best_epoch": None, **reload_info(diff, diff, "ridge_npz")}


def tensor_mapping_sha256(values):
    import hashlib
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        result.update(core.canonical({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}))
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def frozen_hash(model):
    return tensor_mapping_sha256({name: p for name, p in model.named_parameters() if not p.requires_grad})


def build_ft_model(model_path, seed):
    """Construct locally: this function must never call bundle.get_model/download."""
    import torch
    import torchaudio
    from v3.deploy.engines_deploy import set_seed, state_dict_sha256
    set_seed(seed)
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    require(bundle._model_type == "WavLM" and bundle._normalize_waveform is False
            and bundle.sample_rate == 16000, "offline WavLM bundle semantics changed")
    encoder = torchaudio.models.wavlm_model(**bundle._params)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    require(state_dict_sha256(encoder) == WEIGHTS_SHA, "offline WavLM state differs from frozen base")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for layer in encoder.encoder.transformer.layers[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    class Head(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
            self.fc = torch.nn.Linear(768, 6)
        def forward(self, wav):
            features, _ = self.enc.extract_features(wav)
            return self.fc(features[-1].mean(dim=1))
    return Head(encoder), digest(bundle._params)


def delta_names(model):
    trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    buffers = sorted(name for name, _ in model.named_buffers())
    return trainable, buffers


def load_delta(model, delta, expected_frozen_hash):
    trainable, buffers = delta_names(model)
    require(set(delta) == set(trainable) | set(buffers), "delta omits trainable parameters or buffers")
    result = model.load_state_dict(delta, strict=False)
    missing = {name for name, p in model.named_parameters() if not p.requires_grad}
    require(set(result.missing_keys) == missing and not result.unexpected_keys, "invalid delta coverage")
    require(frozen_hash(model) == expected_frozen_hash, "frozen base parameters changed")


def full_ft_loop(model, unit, rows, batch, device):
    """Legacy top-four-layer FT, full budget, without any accuracy calculation."""
    import torch
    from v3.deploy.engines_deploy import set_seed
    cfg, seed = unit["config"], unit["train_seed"]
    require(cfg["epochs"] == 15 and cfg["batch_size"] == 16 and cfg["early_stopping"] is False,
            "FT must execute 15 epochs with batch size 16")
    set_seed(seed)
    model.to(device)
    initial_frozen = frozen_hash(model)
    trainable, buffers = delta_names(model)
    keys = set(trainable) | set(buffers)
    require(all(p.dtype == torch.float32 for p in model.parameters()), "FT parameter storage must remain fp32")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": cfg["lr_encoder"]},
        {"params": model.fc.parameters(), "lr": cfg["lr_head"]}], weight_decay=cfg["weight_decay"])
    labels = core.StrictLabels(rows)
    y_fit, y_val = (labels.labels("cremad", unit[k]) for k in ("fit", "val"))
    counts = np.bincount(y_fit, minlength=6).astype(np.float64)
    require((counts > 0).all(), "FT training set lacks an emotion")
    weight = torch.tensor(counts.sum() / counts / 6, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    use_amp = device.type == "cuda" and cfg["fp16"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = batch.rng
    best_loss, best_epoch, best_state, history = math.inf, None, None, []
    order = np.arange(len(unit["fit"]))
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        rng.shuffle(order)
        total_loss, steps = 0.0, 0
        for start in range(0, len(order), 16):
            indexes = order[start:start + 16]
            paths = [unit["fit"][i] for i in indexes]
            xb = batch(paths, True).to(device)
            yb = torch.tensor(y_fit[indexes], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(xb), yb)
            require(bool(torch.isfinite(loss)), "non-finite FT training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(paths)
            steps += 1
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(unit["val"]), 16):
                paths = unit["val"][start:start + 16]
                logits = model(batch(paths, False).to(device))
                yb = torch.tensor(y_val[start:start + 16], dtype=torch.long, device=device)
                val_loss += float(criterion(logits, yb).cpu()) * len(paths)
        val_loss /= len(unit["val"])
        require(math.isfinite(val_loss), "non-finite FT validation loss")
        history.append({"epoch": epoch, "train_loss": total_loss / len(order),
                        "val_loss": val_loss, "optimizer_steps": steps})
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, cpu_state(model, keys)
    require(frozen_hash(model) == initial_frozen, "FT changed frozen parameters")
    require(best_state is not None, "no FT validation checkpoint")
    return best_state, cpu_state(model, keys), history, best_epoch, initial_frozen


def predict_wave(model, paths, batch, device):
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), 16):
            outputs.append(model(batch(paths[start:start + 16], False).to(device)).cpu().numpy())
    return np.asarray(np.concatenate(outputs), dtype=np.float64)


def fit_ft(unit, rows, model_path, audio_root, device, checkpoint_path, wav_cache=None):
    import torch
    from v3.deploy.engines_deploy import WaveCache, wavlm_batch
    cfg, seed = unit["config"], unit["train_seed"]
    wav = wav_cache if wav_cache is not None else WaveCache(audio_root, 16000)
    rng = np.random.RandomState(seed)
    def batch(paths, train):
        return wavlm_batch(paths, train, wav, int(cfg["crop_seconds"] * 16000),
                           int(cfg["eval_cap_seconds"] * 16000), rng)
    batch.rng = rng
    model, params_sha = build_ft_model(model_path, seed)
    best, last, history, best_epoch, frozen = full_ft_loop(model, unit, rows, batch, device)
    last_logits = predict_wave(model, unit["test"], batch, device)
    load_delta(model, best, frozen)
    logits = predict_wave(model, unit["test"], batch, device)
    trainable, buffers = delta_names(model)
    checkpoint = {"schema": "ser-speaker-coverage-ft-delta-1", "config": cfg,
                  "train_seed": seed, "epochs_run": len(history), "best_epoch": best_epoch,
                  "base_file_sha256": file_sha(model_path), "base_state_sha256": WEIGHTS_SHA,
                  "bundle_params_sha256": params_sha, "frozen_parameter_sha256": frozen,
                  "trainable_parameter_names": trainable, "buffer_names": buffers,
                  "best_state": best, "last_state": last}
    torch.save(checkpoint, checkpoint_path)
    del model, checkpoint, best, last
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(file_sha(model_path) == saved["base_file_sha256"], "base file changed during FT")
    replay, replay_params_sha = build_ft_model(model_path, seed)
    require(replay_params_sha == saved["bundle_params_sha256"], "bundle parameters changed")
    replay.to(device)
    load_delta(replay, saved["best_state"], frozen)
    best_diff = reload_difference(logits, predict_wave(replay, unit["test"], batch, device))
    load_delta(replay, saved["last_state"], frozen)
    last_diff = reload_difference(last_logits, predict_wave(replay, unit["test"], batch, device))
    del replay, saved
    return logits, last_logits, history, {"epochs_run": len(history), "best_epoch": best_epoch,
            "frozen_parameter_sha256": frozen, **reload_info(best_diff, last_diff, "wavlm_delta_best_last")}


def select_units(plan, phase, model, unit_ids=None):
    allowed = phase_units(plan, phase)
    if unit_ids is not None:
        require(isinstance(unit_ids, list) and unit_ids and all(isinstance(x, str) for x in unit_ids)
                and len(unit_ids) == len(set(unit_ids)), "unit-ids must be a nonempty unique JSON list")
        lookup = {u["unit_id"]: u for u in allowed}
        require(set(unit_ids) <= set(lookup), "unit-ids contains an unknown or out-of-phase unit")
        require(all(lookup[uid]["model"] == model for uid in unit_ids), "unit-ids contains a different model")
        return [u for u in allowed if u["unit_id"] in set(unit_ids)]
    return [u for u in allowed if u["model"] == model]


def next_attempt(directory, retry_failed, max_attempts):
    attempts = Path(directory) / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempts.iterdir())
    require(all(p.is_dir() and len(p.name) == 4 and p.name.isdecimal() for p in previous),
            "unexpected attempt artifact; refusing partial reuse")
    numbers = [int(p.name) for p in previous]
    require(numbers == list(range(1, len(numbers) + 1)), "non-contiguous attempt history")
    require(not numbers or retry_failed, "incomplete attempt exists; explicit --retry-failed required")
    attempt = len(numbers) + 1
    require(attempt <= max_attempts, "unit retry budget exhausted")
    target = attempts / f"{attempt:04d}"
    target.mkdir(exist_ok=False)
    return attempt, target


def ledger_events(phase_out):
    path = Path(phase_out) / "ledger.jsonl"
    if not path.exists():
        return []
    data = path.read_bytes()
    require(data.endswith(b"\n"), "truncated ledger requires explicit repair; refusing to invent events")
    return [json.loads(line) for line in data.decode("utf-8").splitlines()]


def record_verified_done(unit, phase_out, recovered=False):
    """Close the DONE-to-ledger crash window, using the original recorded attempt only."""
    directory = Path(phase_out) / "units" / unit["unit_id"]
    done = read_json(directory / "DONE")
    receipt = read_json(directory / "attempts" / f"{done['attempt']:04d}" / "receipt.json")
    events = [e for e in ledger_events(phase_out) if e.get("unit_id") == unit["unit_id"]]
    successful = [e for e in events if e.get("event") == "unit_done"]
    if successful:
        require(len(successful) == 1 and successful[0].get("attempt") == done["attempt"]
                and successful[0].get("fit_seconds") == receipt["fit_seconds"], "DONE disagrees with prior completion ledger")
        return
    attempt_events = [e for e in events if e.get("attempt") == done["attempt"]]
    require(sum(e.get("event") == "unit_start" for e in attempt_events) == 1
            and not any(e.get("event") == "unit_failed" for e in attempt_events),
            "committed result lacks a unique nonfailed ledger start")
    extra = {"recovered_from_done": True, "done_sha256": file_sha(directory / "DONE")} if recovered else {}
    core.append_event(phase_out, "unit_done", unit_id=unit["unit_id"], attempt=done["attempt"],
                      fit_seconds=receipt["fit_seconds"], peak_cuda_bytes=receipt["peak_cuda_bytes"], **extra)


def reject_offloaded_repeat(unit, phase_out):
    require(not any(e.get("event") == "unit_done" and e.get("unit_id") == unit["unit_id"]
                    for e in ledger_events(phase_out)),
            "previously completed unit is absent (possibly offloaded); restore its verified backup or select unrun unit-ids")


def guard_paths(repo, plan_path, features, audio_root, model_path, plan, rows):
    inp = plan["input"]
    paths = [plan_path, model_path, plan_path.parent / "capacity.json",
             plan_path.parent / "selection_diagnostics.json"]
    paths += [relative_file(repo, p) for p in plan["source_sha256"]]
    paths += [relative_file(repo, inp[k + "_path"]) for k in ("manifest", "demographics")]
    paths += [relative_file(features, p) for p in inp["feature_files"].values()]
    paths += [relative_file(features, inp[k]) for k in ("asv_path", "asv_receipt_file")]
    paths += [relative_file(audio_root, p) for p in rows]
    return paths


def execute(args):
    repo, plan_path, features = (Path(p).resolve() for p in (args.repo, args.plan, args.features))
    audio_root, model_path = Path(args.audio_root).resolve(), Path(args.wavlm_model).resolve()
    require(repo == REPO, "runner import checkout differs from --repo")
    require(args.threads >= 1 and args.max_attempts >= 1, "threads and max-attempts must be positive")
    require(args.limit is None or args.limit >= 1, "limit must be positive")
    plan = read_json(plan_path)
    require(plan.get("schema") == SCHEMA and plan.get("program") == PROGRAM, "wrong study plan")
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "plan identity mismatch")
    requested = read_json(args.unit_ids) if args.unit_ids else None
    units = select_units(plan, args.phase, args.model, requested)
    require(units, "no units selected")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    import torch
    from threadpoolctl import threadpool_limits
    from v3.speaker_coverage.verify import verify_plan
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device_name = args.device or ("cpu" if args.model == "ridge_wavlm" else "cuda")
    require(args.model != "ridge_wavlm" or device_name == "cpu", "Ridge uses CPU only")
    device = torch.device(device_name)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable; no automatic CPU fallback")
    # Both phases use the independent CPU verifier, including raw audio/base/cache/source identity.
    gate = verify_plan(repo, plan_path, features, relative_file(features, plan["input"]["asv_path"]),
                       model_path, audio_root=audio_root)
    require(gate.get("pass") is True and gate.get("formal_allowed") is True,
            "independent plan verification did not authorize execution")
    rows = core.load_manifest(relative_file(repo, plan["input"]["manifest_path"]))
    feats = core.StrictFeatures(features, plan["input"], rows)
    guard = core.FileGuard(guard_paths(repo, plan_path, features, audio_root, model_path, plan, rows))
    runtime = core.environment(device_name, args.threads)
    import torchaudio
    runtime.update(torch=torch.__version__, torchaudio=torchaudio.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                   fit_timing="model build + full training + predictions + checkpoint disk save/reload/replay",
                   wall_timing="attempt setup through pre-receipt artifact preparation; excludes initial shared gate")
    phase_out = Path(args.out).resolve() / args.phase
    require(phase_out not in (repo, plan_path.parent, features, audio_root), "unsafe output location")
    phase_out.mkdir(parents=True, exist_ok=True)
    wav_cache = None
    if args.model == "wavlm_ft":
        from v3.deploy.engines_deploy import WaveCache
        wav_cache = WaveCache(audio_root, 16000)
    completed, skipped = 0, 0
    with core.runner_lock(phase_out), threadpool_limits(limits=args.threads):
        identity = {"schema": "ser-speaker-coverage-output-1", "plan_sha256": plan["plan_sha256"],
                    "phase": args.phase, "program": PROGRAM}
        identity_path = phase_out / "identity.json"
        if identity_path.exists():
            require(read_json(identity_path) == identity, "output belongs to a different frozen plan/phase")
        else:
            require(not (phase_out / "units").exists(), "units exist without output identity")
            atomic_json(identity_path, identity)
        atomic_json(phase_out / "plan_snapshot.json", plan)
        atomic_json(phase_out / "verification.json", gate)
        core.append_event(phase_out, "invocation_start", plan_sha256=plan["plan_sha256"], phase=args.phase,
                          model=args.model, selected_unit_ids=[u["unit_id"] for u in units],
                          selection_sha256=digest([u["unit_id"] for u in units]), environment=runtime)
        for unit in units:
            guard.check()
            if verify_completed(unit, phase_out, plan["plan_sha256"], args.phase, rows):
                record_verified_done(unit, phase_out, recovered=True)
                skipped += 1
                continue
            if args.limit is not None and completed >= args.limit:
                break
            directory = phase_out / "units" / unit["unit_id"]
            reject_offloaded_repeat(unit, phase_out)
            attempt, attempt_dir = next_attempt(directory, args.retry_failed, args.max_attempts)
            core.append_event(phase_out, "unit_start", unit_id=unit["unit_id"], attempt=attempt)
            wall_start = time.perf_counter()
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                fit_start = time.perf_counter()
                if unit["model"] == "ridge_wavlm":
                    result = fit_ridge(unit, rows, feats, attempt_dir / "checkpoint.npz")
                elif unit["model"] == "cnn":
                    result = fit_cnn(unit, rows, feats, device, attempt_dir / "checkpoint.pt")
                else:
                    result = fit_ft(unit, rows, model_path, audio_root, device, attempt_dir / "checkpoint.pt", wav_cache)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                logits, last_logits, history, info = result
                info.update(fit_seconds=time.perf_counter() - fit_start,
                            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
                arrays = prediction_arrays(unit, rows, logits, last_logits)
                guard.check()
                feats.unchanged()
                info["wall_seconds"] = time.perf_counter() - wall_start
                write_completed(unit, phase_out, plan, args.phase, attempt, arrays, history, info, runtime)
                record_verified_done(unit, phase_out)
                completed += 1
                print(json.dumps({"event": "unit_done", "phase": args.phase, "model": unit["model"],
                                  "unit_id": unit["unit_id"], "completed_this_invocation": completed,
                                  "epochs_run": info["epochs_run"], "fit_seconds": info["fit_seconds"]}), flush=True)
                del result, logits, last_logits, arrays
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except BaseException as exc:
                if not (directory / "DONE").exists():
                    core.append_event(phase_out, "unit_failed", unit_id=unit["unit_id"], attempt=attempt,
                                      error_type=type(exc).__name__, error=str(exc))
                raise
        guard.check()
        blocks = finish_blocks(plan, phase_out, args.phase, rows)
        summary = {"phase": args.phase, "model": args.model, "newly_completed": completed,
                   "skipped_verified": skipped, "blocks": blocks, "scores_computed": False}
        core.append_event(phase_out, "invocation_end", **summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "features", "audio-root", "wavlm-model", "out"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--model", choices=MODELS, required=True)
    result.add_argument("--phase", choices=("pilot", "formal"), required=True)
    result.add_argument("--device", choices=("cpu", "cuda"))
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--limit", "--max-units", type=int)
    result.add_argument("--unit-ids", help="JSON list of fixed plan IDs; preserves plan order")
    result.add_argument("--retry-failed", action="store_true", help="start a fresh bounded attempt; never reuse partial weights")
    result.add_argument("--max-attempts", type=int, default=2)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(parser().parse_args()), sort_keys=True))

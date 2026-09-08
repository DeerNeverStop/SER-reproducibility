"""Deployment-study adaptations of the v2 GPU training engines (SER26-DEPLOY-2, block B2).

Source: v2/ser_v2/train.py (ENGINE_VERSION "ser_v2-train-1") -- torch_train_loop, predict_torch, spec_augment,
engine_p1_frozen and engine_wavlm are copied verbatim. The ONE behavioural addition: after the best checkpoint
is restored, logits are also computed on fold["val"] through the same evaluation path as the per-epoch
validation pass, and returned next to the test logits:

    test_logits, val_logits, history, info = engine(row, cfg, fold, data, device)

Seeding, augmentation, class-balanced loss, optimiser, schedule, stopping rule and checkpoint restore are
unchanged. Unchanged helpers (set_seed, stats.uar) are imported read-only from v2/ser_v2; the model builder is
imported exactly as v2 does (advanced_models + tuned_standard_experiment.candidate_configs on the repo root).
DEPLOY-2 additionally seeds before encoder/head creation, guards the zero-trainable-layer case,
and returns the restored checkpoint state for persistent integrity-checked storage.
Nothing under v2/ is modified. torch is imported lazily so a CPU-only caller can install common.cpu_guard first.

`data` is any object exposing
    n_classes: int
    labels(paths) -> np.ndarray[int64]
    logmel(paths) -> np.ndarray[float32, (n, n_mels, frames)]     (cnn / p1_frozen)
    audio_root: Path                                                 (wavlm_ft)
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "v2"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from ser_v2.stats import uar  # noqa: E402  (v2 helper, read-only, unchanged)
from ser_v2.train import ENGINE_VERSION as V2_ENGINE_VERSION  # noqa: E402
from ser_v2.train import set_seed  # noqa: E402  (v2 helper, read-only, unchanged)

ENGINE_VERSION = "deploy-engines-2-seeded-head-checkpoints"
V2_SOURCE = "v2/ser_v2/train.py"

# tuned_standard_experiment.candidate_configs("cnn")[2] as frozen by v2 (config_source in unit_configs.json).
# Used only when the repo-root module chain cannot be imported; when it can, the two MUST agree (checked).
P1_CNN_ARCH_FROZEN = {"width": 48, "n_blocks": 3, "kernel_size": 5, "dilation_base": 1, "dropout": 0.1,
                      "lr": 1e-3, "weight_decay": 1e-4, "augment": True, "balanced_loss": True,
                      "time_mask": 16, "freq_mask": 8, "noise_std": 0.1}

# sha256 over the sorted state_dict of torchaudio WAVLM_BASE_PLUS.get_model() as recorded by v2's SSL caches
# (features/<base>__wavlm_base_plus__*.json: weights_sha256). The fine-tune engine refuses other weights.
WAVLM_BASE_PLUS_WEIGHTS_SHA256_V2 = "d8e745fb761c56d209431413bcc373f92784dc2c8d850199b6caa5cf05ca2c36"


# ----------------------------------------------------------------------------- v2 copies (unchanged)

def torch_train_loop(model, X_fit, y_fit, X_val, y_val, cfg: dict, seed: int, n_classes: int, device, augment_fn=None):
    """P1 semantics: AdamW, cosine T_max=epochs, class-balanced CE (fit subset), strict-min validation
    loss checkpoint, patience on validation loss, restore best. Returns (model, history, info)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    set_seed(seed)
    model.to(device)
    Xf, yf = torch.from_numpy(np.asarray(X_fit, dtype=np.float32)), torch.from_numpy(np.asarray(y_fit))
    Xv, yv = torch.from_numpy(np.asarray(X_val, dtype=np.float32)), torch.from_numpy(np.asarray(y_val))
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(Xf, yf), batch_size=int(cfg.get("batch_size", 64)), shuffle=True, generator=g)
    counts = np.bincount(y_fit, minlength=n_classes).astype(np.float64)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / n_classes, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    epochs, patience = int(cfg.get("epochs", 100)), int(cfg.get("patience", 15))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss, best_state, best_epoch, stale, history = math.inf, None, None, 0, []
    t0 = time.perf_counter()
    aug_rng = np.random.RandomState(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        tl, tw = 0.0, 0.0
        for xb, yb in train_loader:
            if augment_fn is not None:
                xb = augment_fn(xb, aug_rng)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tl += float(F.cross_entropy(logits.detach(), yb, weight=weight, reduction="sum").cpu())
                tw += float(weight[yb].sum().cpu())
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = model(Xv.to(device))
            vloss = float(F.cross_entropy(vl, yv.to(device), weight=weight).cpu())
            vpred = vl.argmax(dim=1).cpu().numpy()
        history.append({"epoch": epoch, "train_loss": tl / max(tw, 1e-9), "val_loss": vloss, "val_uar": uar(np.asarray(y_val), vpred, n_classes)})
        if vloss < best_loss:
            best_loss, best_epoch, stale = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint")
    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "val_loss_best": best_loss,
            "val_uar_best": history[best_epoch - 1]["val_uar"], "train_seconds": time.perf_counter() - t0}
    return model, history, info


def predict_torch(model, X, device) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)).cpu().numpy()
    return np.asarray(out, dtype=np.float64)


def spec_augment(xb, rng: np.random.RandomState, time_mask=16, freq_mask=8, noise_std=0.1):
    """P1 augmentation contract: 2 time masks, 2 frequency masks, Gaussian noise, training only."""
    import torch
    x = xb.clone()
    n, n_mels, frames = x.shape
    for i in range(n):
        for _ in range(2):
            t = int(rng.randint(0, time_mask + 1)); t0 = int(rng.randint(0, max(1, frames - t)))
            x[i, :, t0:t0 + t] = 0.0
            f = int(rng.randint(0, freq_mask + 1)); f0 = int(rng.randint(0, max(1, n_mels - f)))
            x[i, f0:f0 + f, :] = 0.0
    if noise_std > 0:
        x = x + torch.from_numpy(rng.normal(0, noise_std, size=x.shape).astype(np.float32))
    return x


# ----------------------------------------------------------------------------- P1 scratch CNN (p1_frozen)

def p1_arch(cfg: dict) -> tuple[dict, str]:
    """v2: arch = tuned_standard_experiment.candidate_configs(cfg["model"])[2] with lr/weight_decay/dropout
    overridden from the unit config. Falls back to the frozen constant for "cnn" only if the repo-root import
    chain is unavailable; when both are available they must agree."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    source = "tuned_standard_experiment.candidate_configs(model)[2]"
    try:
        import tuned_standard_experiment as tse
        arch = dict(tse.candidate_configs(cfg["model"])[2])
        if cfg["model"] == "cnn" and arch != P1_CNN_ARCH_FROZEN:
            raise RuntimeError(f"candidate_configs('cnn')[2] drifted from the frozen v2 architecture: {arch}")
    except ImportError as exc:
        if cfg["model"] != "cnn":
            raise
        arch, source = dict(P1_CNN_ARCH_FROZEN), f"frozen_constant (tuned_standard_experiment unavailable: {exc!r})"
    for k in ("lr", "weight_decay", "dropout"):
        if k in cfg:
            arch[k] = cfg[k]
    return arch, source


def engine_p1_frozen(row, cfg, fold, data, device):
    """v2 engine_p1_frozen + val-logit export. Returns (test_logits, val_logits, history, info)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import advanced_models
    arch, arch_source = p1_arch(cfg)
    n_classes = data.n_classes
    X_fit = data.logmel(fold["fit"]); X_val = data.logmel(fold["val"]); X_te = data.logmel(fold["test"])
    y_fit, y_val = data.labels(fold["fit"]), data.labels(fold["val"])
    set_seed(int(row["train_seed"]))
    model = advanced_models.build_neural_model(cfg["model"], n_mels=X_fit.shape[1], n_classes=n_classes, config=arch)
    aug = (lambda xb, rng: spec_augment(xb, rng, arch.get("time_mask", 16), arch.get("freq_mask", 8), arch.get("noise_std", 0.1))) if arch.get("augment", True) else None
    model, history, info = torch_train_loop(model, X_fit, y_fit, X_val, y_val, {**cfg, "lr": arch["lr"], "weight_decay": arch["weight_decay"]},
                                            int(row["train_seed"]), n_classes, device, aug)
    test_logits = predict_torch(model, X_te, device)
    val_logits = predict_torch(model, X_val, device)          # ADDITION: val logits of the restored checkpoint
    info = {**info, "arch": arch, "arch_source": arch_source, "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "_checkpoint_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return test_logits, val_logits, history, info


# ----------------------------------------------------------------------------- WavLM partial fine-tune

def load_wav_trimmed(path: Path, sr: int = 16000) -> np.ndarray:
    """v2 waveform cache entry: librosa.load(sr, mono) -> trim top_db 30 -> float16."""
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30.0)
    return y.astype(np.float16)


class WaveCache:
    """In-RAM float16 waveform cache keyed by manifest relative_path (v2 engine_wavlm.wav)."""

    def __init__(self, audio_root: Path, sr: int = 16000):
        self.audio_root, self.sr, self.cache = Path(audio_root), sr, {}

    def __call__(self, path: str) -> np.ndarray:
        if path not in self.cache:
            self.cache[path] = load_wav_trimmed(self.audio_root / path, self.sr)
        return self.cache[path]

    def __len__(self):
        return len(self.cache)


def wavlm_batch(paths, train: bool, wav, crop: int, cap: int, rng: np.random.RandomState):
    """v2 engine_wavlm.batch, verbatim: training -> one random crop of `crop` samples (zero-padded when
    shorter); evaluation -> first `cap` samples; batch zero-padded to its longest member. Float32 tensor."""
    import torch
    arrs = []
    for p in paths:
        y = wav(p).astype(np.float32)
        if train:
            if len(y) > crop:
                s = rng.randint(0, len(y) - crop + 1); y = y[s:s + crop]
            y = np.pad(y, (0, crop - len(y)))
        else:
            y = y[:cap]
        arrs.append(y)
    L = max(len(a) for a in arrs)
    return torch.from_numpy(np.stack([np.pad(a, (0, L - len(a))) for a in arrs]))


def state_dict_sha256(module) -> str:
    """Same digest as v2 ser_v2.features.build_ssl_cache (sorted state_dict, raw tensor bytes)."""
    h = hashlib.sha256()
    for _, t in sorted(module.state_dict().items()):
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def engine_wavlm(row, cfg, fold, data, device, check_weights: bool = True):
    """v2 engine_wavlm (partial fine-tune of the top 4 transformer layers + mean-pool + linear head, or the
    frozen same-regime comparator) + val-logit export. Returns (test_logits, val_logits, history, info)."""
    import torch
    import torchaudio
    set_seed(int(row["train_seed"]))  # DEPLOY-2: include the randomly initialised head.
    n_classes = data.n_classes
    audio_root = Path(data.audio_root)
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    enc = bundle.get_model()
    weights_sha = state_dict_sha256(enc)
    if check_weights and weights_sha != WAVLM_BASE_PLUS_WEIGHTS_SHA256_V2:
        raise RuntimeError(f"WAVLM_BASE_PLUS weights differ from v2 ({weights_sha[:12]} != {WAVLM_BASE_PLUS_WEIGHTS_SHA256_V2[:12]})")
    trainable_top = 4 if cfg["engine"] == "wavlm_partial_ft" else 0
    for p in enc.parameters():
        p.requires_grad = False
    layers = enc.encoder.transformer.layers
    for layer in (layers[len(layers) - trainable_top:] if trainable_top else []):
        for p in layer.parameters():
            p.requires_grad = True

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.fc = torch.nn.Linear(768, n_classes)

        def forward(self, wav):
            feats_, _ = self.enc.extract_features(wav)
            return self.fc(feats_[-1].mean(dim=1))

    model = Head()
    sr = 16000
    wav = WaveCache(audio_root, sr)
    crop = int(cfg["crop_seconds"] * sr); cap = int(cfg["eval_cap_seconds"] * sr)
    rng = np.random.RandomState(int(row["train_seed"]))

    def batch(paths, train):
        return wavlm_batch(paths, train, wav, crop, cap, rng)

    y_fit, y_val = data.labels(fold["fit"]), data.labels(fold["val"])
    set_seed(int(row["train_seed"]))
    model.to(device)
    params = [{"params": [p for n, p in model.enc.named_parameters() if p.requires_grad], "lr": float(cfg["lr_encoder"])},
              {"params": model.fc.parameters(), "lr": float(cfg["lr_head"])}]
    opt = torch.optim.AdamW(params, weight_decay=float(cfg["weight_decay"]))
    counts = np.bincount(y_fit, minlength=n_classes).astype(np.float64)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / n_classes, dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    # torch.amp.GradScaler("cuda", ...) is the class v2 reached through the torch.cuda.amp.GradScaler alias.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.get("fp16", True)))
    bs = int(cfg["batch_size"]); best, best_state, best_epoch, stale, history = math.inf, None, None, 0, []
    order = np.arange(len(fold["fit"]))
    t0 = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train(); rng.shuffle(order)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            xb = batch([fold["fit"][j] for j in idx], True).to(device); yb = torch.from_numpy(y_fit[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); vl, vp = 0.0, []
        with torch.no_grad():
            for i in range(0, len(fold["val"]), bs):
                paths = fold["val"][i:i + bs]
                xb = batch(paths, False).to(device); yb = torch.from_numpy(y_val[i:i + bs]).to(device)
                out = model(xb); vl += float(crit(out, yb).cpu()) * len(paths); vp.extend(out.argmax(1).cpu().numpy().tolist())
        vl /= len(fold["val"])
        history.append({"epoch": epoch, "train_loss": None, "val_loss": vl, "val_uar": uar(y_val, np.asarray(vp), n_classes)})
        if vl < best:
            best, best_epoch, stale = vl, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
    model.load_state_dict(best_state); model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(fold["test"]), bs):
            logits.append(model(batch(fold["test"][i:i + bs], False).to(device)).float().cpu().numpy())
    val_logits = []                                            # ADDITION: val logits of the restored checkpoint,
    with torch.no_grad():                                      # same batching/cap as the per-epoch validation pass
        for i in range(0, len(fold["val"]), bs):
            val_logits.append(model(batch(fold["val"][i:i + bs], False).to(device)).float().cpu().numpy())
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "val_loss_best": best, "val_uar_best": history[best_epoch - 1]["val_uar"],
            "train_seconds": time.perf_counter() - t0, "fp16": scaler.is_enabled(), "encoder_weights_sha256": weights_sha,
            "trainable_top_layers": trainable_top, "n_waveforms_cached": len(wav),
            "n_parameters_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "_checkpoint_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return np.concatenate(logits).astype(np.float64), np.concatenate(val_logits).astype(np.float64), history, info


ENGINES = {"p1_frozen": engine_p1_frozen, "wavlm_partial_ft": engine_wavlm, "wavlm_frozen_same_regime": engine_wavlm}
PLAN_MODEL_ENGINE = {"cnn": "p1_frozen", "wavlm_ft": "wavlm_partial_ft"}

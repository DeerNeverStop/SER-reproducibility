"""Unit runner. Executes pending units of one arm in the fixed plan order, writes the
contract outputs (predictions.csv, unit.json, history.json, DONE), appends the attempt
ledger, enforces per-arm caps, and never touches an outer test set before training ends.

Engines
  p1_frozen        the P1 scratch models (advanced_models.build_neural_model, config index 2)
  linear_probe     early-stopped linear head on cached SSL final-state features
  ridge_a1         StandardScaler(train fold) + RidgeClassifier(alpha=1, balanced), CPU
  wavlm_partial_ft / wavlm_frozen_same_regime   (waveform engines; PREP dry run required)
  synthetic        tiny logistic model on synthetic features (CPU dry run only)

    python -m ser_v2.train --plan PLAN --manifests MANIFESTS --features FEATS --run RUN --arm CTRL [--engine-override synthetic] [--max-units N] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import corpora
from .common import atomic_write_json, atomic_write_text, canonical_json, read_csv, read_json, sha256_text
from .features import FeatureStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_VERSION = "ser_v2-train-1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- data access

class PlanIO:
    def __init__(self, plan_dir: Path, manifests_dir: Path, run_dir: Path):
        self.plan_dir, self.run_dir = plan_dir, run_dir
        self.plan = read_csv(plan_dir / "run_plan.csv")
        self.index = read_json(plan_dir / "split_index.json")
        self.sha_to_key = {v["sha256"]: k for k, v in self.index.items()}
        self.configs = read_json(plan_dir / "unit_configs.json")
        self.arms = read_json(plan_dir.parent / "registry" / "arms.json") if (plan_dir.parent / "registry" / "arms.json").exists() else None
        self.manifest: dict[str, dict[str, dict]] = {}
        self.n_classes: dict[str, int] = {}
        for base in ("ravdess", "cremad", "subesco", "subesco_980"):
            p = manifests_dir / f"{base}_manifest.csv"
            if p.exists():
                rows, _ = corpora.apply_hygiene("subesco" if base == "subesco_980" else base, corpora.load_manifest(p))
            else:
                rows = corpora.synthetic_manifest(base, 20, 10, 5)
            self.manifest[base] = {r["relative_path"]: r for r in rows}
            self.n_classes[base] = len(corpora.CORPORA["subesco" if base == "subesco_980" else base].labels)
        self._tables: dict[str, dict] = {}

    def fold(self, row: dict) -> dict:
        key = self.sha_to_key[row["split_sha256"]]
        if key not in self._tables:
            self._tables[key] = read_json(self.plan_dir / "splits" / f"{key}.json")
        table = self._tables[key]
        f = next(f for f in table["folds"] if f["fold"] == int(row["fold"]))
        if "fit" not in f:
            test, val = set(f["test"]), set(f["val"])
            f = {**f, "fit": [p for p in table["population"] if p not in test and p not in val]}
        return f

    def labels(self, base: str, paths: list[str]) -> np.ndarray:
        m = self.manifest[base]
        return np.asarray([int(m[p]["label_index"]) for p in paths], dtype=np.int64)

    def speakers(self, base: str, paths: list[str]) -> list[str]:
        m = self.manifest[base]
        return [m[p]["speaker"] for p in paths]


# ----------------------------------------------------------------------------- engines

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


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
        from .stats import uar
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


def engine_p1_frozen(row, cfg, fold, io: PlanIO, feats: FeatureStore, device):
    sys.path.insert(0, str(REPO_ROOT))
    import advanced_models
    import tuned_standard_experiment as tse
    base = row["base_corpus"]
    arch = dict(tse.candidate_configs(cfg["model"])[2])
    for k in ("lr", "weight_decay", "dropout"):
        if k in cfg:
            arch[k] = cfg[k]
    n_classes = io.n_classes[base]
    X_fit = feats.get(base, "logmel", fold["fit"]); X_val = feats.get(base, "logmel", fold["val"]); X_te = feats.get(base, "logmel", fold["test"])
    y_fit, y_val = io.labels(base, fold["fit"]), io.labels(base, fold["val"])
    set_seed(int(row["train_seed"]))
    model = advanced_models.build_neural_model(cfg["model"], n_mels=X_fit.shape[1], n_classes=n_classes, config=arch)
    aug = (lambda xb, rng: spec_augment(xb, rng, arch.get("time_mask", 16), arch.get("freq_mask", 8), arch.get("noise_std", 0.1))) if arch.get("augment", True) else None
    model, history, info = torch_train_loop(model, X_fit, y_fit, X_val, y_val, {**cfg, "lr": arch["lr"], "weight_decay": arch["weight_decay"]},
                                            int(row["train_seed"]), n_classes, device, aug)
    return predict_torch(model, X_te, device), history, info, model


def engine_linear_probe(row, cfg, fold, io: PlanIO, feats: FeatureStore, device):
    import torch
    base = row["base_corpus"]
    n_classes = io.n_classes[base]
    state = int(cfg.get("feature_state", 12))
    X_fit = feats.get(base, cfg["model"], fold["fit"], state); X_val = feats.get(base, cfg["model"], fold["val"], state); X_te = feats.get(base, cfg["model"], fold["test"], state)
    mu, sd = X_fit.mean(axis=0), X_fit.std(axis=0) + 1e-6
    X_fit, X_val, X_te = (X_fit - mu) / sd, (X_val - mu) / sd, (X_te - mu) / sd
    y_fit, y_val = io.labels(base, fold["fit"]), io.labels(base, fold["val"])
    set_seed(int(row["train_seed"]))
    model = torch.nn.Linear(X_fit.shape[1], n_classes)
    model, history, info = torch_train_loop(model, X_fit, y_fit, X_val, y_val, cfg, int(row["train_seed"]), n_classes, device)
    return predict_torch(model, X_te, device), history, info, model


def engine_ridge(row, cfg, fold, io: PlanIO, feats: FeatureStore, device):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    base = row["base_corpus"]
    enc = cfg["model"].replace("ridge_a1_", "") if "model" in cfg else row["model"].replace("ridge_a1_", "")
    state = int(cfg.get("feature_state", 12))
    X_fit = feats.get(base, enc, fold["fit"], state); X_te = feats.get(base, enc, fold["test"], state)
    y_fit = io.labels(base, fold["fit"])
    scaler = StandardScaler().fit(X_fit) if cfg.get("scaler", "train_fold_standard") == "train_fold_standard" else None
    if scaler is not None:
        X_fit, X_te = scaler.transform(X_fit), scaler.transform(X_te)
    clf = RidgeClassifier(alpha=float(cfg.get("alpha", 1.0)), class_weight=cfg.get("class_weight", "balanced")).fit(X_fit, y_fit)
    scores = clf.decision_function(X_te)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    n_classes = io.n_classes[base]
    logits = np.full((len(X_te), n_classes), -1e9); logits[:, clf.classes_] = scores
    return logits, [], {"best_epoch": 0, "epochs_run": 0, "val_loss_best": None, "val_uar_best": None, "train_seconds": 0.0}, None


def engine_synthetic(row, cfg, fold, io: PlanIO, feats: FeatureStore, device):
    """CPU dry run: logistic regression on flattened synthetic features (logmel mean over time)."""
    from sklearn.linear_model import LogisticRegression
    base = row["base_corpus"]
    kind = "logmel" if row["model"] in ("cnn", "resnet_se", "transformer") or row["arm"] in ("MECH2X2", "HPO") else \
        (row["model"].replace("ridge_a1_", "") if row["model"] not in ("wavlm_base_plus_ft", "wavlm_base_plus_frozen_sr") else "wavlm_base_plus")
    def load(paths):
        X = feats.get(base, kind, paths, None if kind == "logmel" else 12)
        return X.mean(axis=2) if X.ndim == 3 else X
    X_fit, X_val, X_te = load(fold["fit"]), load(fold["val"] or fold["fit"][:8]), load(fold["test"])
    y_fit, y_val = io.labels(base, fold["fit"]), io.labels(base, fold["val"] or fold["fit"][:8])
    clf = LogisticRegression(max_iter=200, C=0.5, random_state=int(row["train_seed"]) % (2 ** 31)).fit(X_fit, y_fit)
    n_classes = io.n_classes[base]
    from .stats import uar
    val_uar = uar(y_val, clf.predict(X_val), n_classes)
    logits = np.full((len(X_te), n_classes), -1e9); logits[:, clf.classes_] = clf.decision_function(X_te) if n_classes > 2 else np.stack([-clf.decision_function(X_te), clf.decision_function(X_te)], 1)
    return logits, [{"epoch": 1, "train_loss": 0.0, "val_loss": 0.0, "val_uar": val_uar}], \
        {"best_epoch": 1, "epochs_run": 1, "val_loss_best": 0.0, "val_uar_best": val_uar, "train_seconds": 0.0}, None


def engine_wavlm(row, cfg, fold, io: PlanIO, feats: FeatureStore, device, audio_root: Path | None = None):
    """Partial fine-tune (top 4 transformer layers + mean-pool + linear head) or frozen same-regime
    comparator (pool + head only), from an in-RAM float16 waveform cache. Requires torchaudio and
    the corpus audio; exercised by the PREP dry run on the training machine."""
    import torch
    import torchaudio
    import librosa
    base = row["base_corpus"]
    n_classes = io.n_classes[base]
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    enc = bundle.get_model()
    trainable_top = 4 if cfg["engine"] == "wavlm_partial_ft" else 0
    for p in enc.parameters():
        p.requires_grad = False
    layers = enc.encoder.transformer.layers
    for layer in layers[len(layers) - trainable_top:]:
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
    cache: dict[str, np.ndarray] = {}

    def wav(path):
        if path not in cache:
            y, _ = librosa.load(str(audio_root / path), sr=sr, mono=True)
            y, _ = librosa.effects.trim(y, top_db=30.0)
            cache[path] = y.astype(np.float16)
        return cache[path]

    crop = int(cfg["crop_seconds"] * sr); cap = int(cfg["eval_cap_seconds"] * sr)
    rng = np.random.RandomState(int(row["train_seed"]))

    def batch(paths, train):
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

    y_fit, y_val = io.labels(base, fold["fit"]), io.labels(base, fold["val"])
    set_seed(int(row["train_seed"]))
    model.to(device)
    params = [{"params": [p for n, p in model.enc.named_parameters() if p.requires_grad], "lr": float(cfg["lr_encoder"])},
              {"params": model.fc.parameters(), "lr": float(cfg["lr_head"])}]
    opt = torch.optim.AdamW(params, weight_decay=float(cfg["weight_decay"]))
    counts = np.bincount(y_fit, minlength=n_classes).astype(np.float64)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / n_classes, dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.get("fp16", True)))
    bs = int(cfg["batch_size"]); best, best_state, best_epoch, stale, history = math.inf, None, None, 0, []
    order = np.arange(len(fold["fit"]))
    t0 = time.perf_counter()
    from .stats import uar
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
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "val_loss_best": best, "val_uar_best": history[best_epoch - 1]["val_uar"],
            "train_seconds": time.perf_counter() - t0, "fp16": scaler.is_enabled()}
    return np.concatenate(logits).astype(np.float64), history, info, model


ENGINES = {"p1_frozen": engine_p1_frozen, "linear_probe": engine_linear_probe, "ridge_a1": engine_ridge,
           "synthetic": engine_synthetic, "wavlm_partial_ft": engine_wavlm, "wavlm_frozen_same_regime": engine_wavlm}


# ----------------------------------------------------------------------------- unit execution

def write_unit(row: dict, cfg: dict, fold: dict, io: PlanIO, logits: np.ndarray, history, info, run_dir: Path,
               started: str, model=None, persist_checkpoint: bool = False) -> str:
    base = row["base_corpus"]
    paths = fold["test"]
    y_true = io.labels(base, paths)
    y_pred = np.argmax(logits, axis=1)
    spk = io.speakers(base, paths)
    m = io.manifest[base]
    order = sorted(range(len(paths)), key=lambda i: int(m[paths[i]]["sample_index"]))
    K = logits.shape[1]
    lines = ["sample_index,relative_path,speaker,y_true,y_pred," + ",".join(f"logit_{i}" for i in range(K))]
    for i in order:
        lines.append(f"{int(m[paths[i]]['sample_index'])},{paths[i]},{spk[i]},{int(y_true[i])},{int(y_pred[i])}," + ",".join(repr(float(v)) for v in logits[i]))
    text = "\n".join(lines) + "\n"
    sha = sha256_text(text)
    udir = run_dir / "units" / row["unit_id"]
    udir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(udir / "predictions.csv", text)
    atomic_write_json(udir / "history.json", history)
    unit = {"unit_id": row["unit_id"], "arm": row["arm"], "corpus_level": row["corpus_level"], "model": row["model"], "cell": row["cell"],
            "fold": int(row["fold"]), "r": int(row["r"]), "seed_index": int(row["seed_index"]), "train_seed": int(row["train_seed"]),
            "config_sha256": row["config_sha256"], "split_sha256": row["split_sha256"], "predictions_sha256": sha,
            "n_fit": len(fold["fit"]), "n_val": len(fold["val"]), "n_test": len(paths), "best_epoch": info.get("best_epoch"),
            "epochs_run": info.get("epochs_run"), "val_uar_best": info.get("val_uar_best"), "val_loss_best": info.get("val_loss_best"),
            "gpu_seconds": info.get("train_seconds", 0.0), "started_at": started, "finished_at": now(), "resumed": False,
            "engine_version": ENGINE_VERSION, "status": "done", "config": cfg}
    atomic_write_json(udir / "unit.json", unit)
    if persist_checkpoint and model is not None:
        try:
            import torch
            torch.save(model.state_dict(), udir / "checkpoint.pt")
        except Exception as exc:  # noqa: BLE001
            unit["checkpoint_error"] = repr(exc)
    atomic_write_text(udir / "DONE", sha + "\n")
    return sha


def run_arm(plan_dir: Path, manifests_dir: Path, features_dir: Path, run_dir: Path, arm: str, engine_override: str | None,
            device_name: str, max_units: int | None, audio_root: Path | None, unit_filter: dict) -> dict:
    io = PlanIO(plan_dir, manifests_dir, run_dir)
    feats = FeatureStore(features_dir)
    try:
        import torch
        device = torch.device(device_name if (device_name != "cuda" or torch.cuda.is_available()) else "cpu")
    except ImportError:
        device = None
    rows = [r for r in io.plan if r["arm"] == arm and all(str(r.get(k)) == str(v) for k, v in unit_filter.items())]
    rows.sort(key=lambda r: (r["corpus_level"], r["model"], int(r["r"]), int(r["fold"]), r["cell"], int(r["seed_index"]), r["config_sha256"]))
    cap_h = None
    if io.arms:
        cap_h = io.arms["arms"].get(arm, {}).get("cap")
    spent = 0.0
    for r in rows:
        uj = run_dir / "units" / r["unit_id"] / "unit.json"
        if (run_dir / "units" / r["unit_id"] / "DONE").exists() and uj.exists():
            spent += float(read_json(uj).get("gpu_seconds", 0.0))
    ledger = run_dir / "ledger.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    n_done, n_skipped, n_failed = 0, 0, 0
    for r in rows:
        if (run_dir / "units" / r["unit_id"] / "DONE").exists():
            n_skipped += 1
            continue
        if max_units is not None and n_done >= max_units:
            break
        if cap_h is not None and cap_h > 0 and spent / 3600 + float(r["est_gpu_sec"]) / 3600 > cap_h and engine_override is None:  # cap 0 = CPU arm, uncapped
            with open(ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"unit_id": r["unit_id"], "event": "cap_reached", "arm": arm, "spent_hours": spent / 3600, "cap": cap_h, "at": now()}) + "\n")
            print(f"cap reached for {arm}: {spent/3600:.2f} h of {cap_h}; stop and record a deviation before continuing", file=sys.stderr)
            break
        cfg = io.configs[r["config_sha256"]]
        engine_name = engine_override or cfg["engine"]
        fold = io.fold(r)
        started = now()
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"unit_id": r["unit_id"], "event": "start", "engine": engine_name, "at": started}) + "\n")
        try:
            fn = ENGINES[engine_name]
            if engine_name.startswith("wavlm"):
                logits, history, info, model = fn(r, cfg, fold, io, feats, device, audio_root)
            else:
                logits, history, info, model = fn(r, cfg, fold, io, feats, device)
            persist = (r["arm"] == "CTRL" and int(r["r"]) == 0 and int(r["seed_index"]) == 0 and r["model"] in ("cnn", "resnet_se")
                       and r["cell"] in ("RR", "GG")) or r["arm"] == "FT"
            sha = write_unit(r, cfg, fold, io, logits, history, info, run_dir, started, model, persist)
            spent += float(info.get("train_seconds", 0.0))
            n_done += 1
            with open(ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"unit_id": r["unit_id"], "event": "done", "predictions_sha256": sha, "seconds": info.get("train_seconds"), "at": now()}) + "\n")
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            with open(ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"unit_id": r["unit_id"], "event": "failed", "error": repr(exc)[:500], "at": now()}) + "\n")
            print(f"unit {r['unit_id'][:12]} failed: {exc!r}", file=sys.stderr)
    return {"arm": arm, "done": n_done, "skipped_done": n_skipped, "failed": n_failed, "spent_hours": spent / 3600, "cap_hours": cap_h}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--engine-override", default=None, choices=[None, "synthetic"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--audio-root", type=Path, default=None)
    ap.add_argument("--filter", default="", help="k=v,k=v restriction on plan columns (e.g. corpus_level=ravdess,model=cnn)")
    a = ap.parse_args(argv)
    flt = dict(kv.split("=", 1) for kv in a.filter.split(",") if kv)
    print(json.dumps(run_arm(a.plan, a.manifests, a.features, a.run, a.arm, a.engine_override, a.device, a.max_units, a.audio_root, flt)))


if __name__ == "__main__":
    main()

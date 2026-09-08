"""Training, metrics, and statistics for the tuned SER experiments."""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fno_data import spec_augment


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SpectrogramDataset(Dataset):
    """In-memory spectrogram dataset with reproducible train-only augmentation."""

    def __init__(
        self,
        X,
        y,
        augment: bool = False,
        seed: int = 0,
        time_mask: int = 16,
        freq_mask: int = 8,
        noise_std: float = 0.1,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self.aug_cfg = {
            "time_mask_width": int(time_mask),
            "freq_mask_width": int(freq_mask),
            "noise_std": float(noise_std),
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.augment:
            x = spec_augment(x, rng=self.rng, **self.aug_cfg)
        return torch.tensor(x, dtype=torch.float32), int(self.y[index])


def _loader(
    X,
    y=None,
    batch_size: int = 64,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = 0,
    augment_config: dict | None = None,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if y is None:
        dataset = TensorDataset(torch.tensor(np.asarray(X), dtype=torch.float32))
    else:
        augment_config = augment_config or {}
        dataset = SpectrogramDataset(
            X,
            y,
            augment=augment,
            seed=seed + 7919,
            time_mask=int(augment_config.get("time_mask", 16)),
            freq_mask=int(augment_config.get("freq_mask", 8)),
            noise_std=float(augment_config.get("noise_std", 0.1)),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_neural(model, X, y, device, batch_size: int = 64) -> dict:
    model.eval()
    logits = []
    with torch.no_grad():
        for (xb,) in _loader(X, batch_size=batch_size):
            logits.append(model(xb.to(device)).cpu())
    logits = torch.cat(logits).numpy()
    pred = logits.argmax(axis=1)
    result = classification_metrics(y, pred)
    result["predictions"] = pred
    result["logits"] = logits
    return result


def balanced_class_weights(y, n_classes: int):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def _criterion(y_train, n_classes: int, balanced_loss: bool, device):
    if not balanced_loss:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(
        weight=balanced_class_weights(y_train, n_classes).to(device)
    )


def _validation_pass(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb_device = yb.to(device)
            logits = model(xb)
            total_loss += float(criterion(logits, yb_device).cpu()) * len(yb)
            total_count += len(yb)
            preds.append(logits.argmax(1).cpu())
            targets.append(yb)
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def train_with_validation(
    model,
    X_fit,
    y_fit,
    X_val,
    y_val,
    max_epochs: int,
    patience: int,
    device,
    config: dict,
    batch_size: int = 64,
    seed: int = 0,
):
    """Select a checkpoint by validation loss and restore it."""
    set_reproducible_seed(seed)
    model.to(device)
    augment_config = {
        "time_mask": config.get("time_mask", 16),
        "freq_mask": config.get("freq_mask", 8),
        "noise_std": config.get("noise_std", 0.1),
    }
    train_loader = _loader(
        X_fit,
        y_fit,
        batch_size=batch_size,
        shuffle=True,
        augment=bool(config.get("augment", True)),
        seed=seed,
        augment_config=augment_config,
    )
    val_loader = _loader(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    n_classes = int(max(np.max(y_fit), np.max(y_val)) + 1)
    criterion = _criterion(
        y_fit,
        n_classes,
        bool(config.get("balanced_loss", True)),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_epochs)
    )

    best_state = None
    best_epoch = 0
    best_loss = float("inf")
    best_balanced_accuracy = -1.0
    best_metrics = None
    stale = 0
    started = time.perf_counter()
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_metrics = _validation_pass(model, val_loader, criterion, device)
        history.append({"epoch": epoch, **val_metrics})
        improved = (
            val_metrics["loss"] < best_loss - 1e-7
            or (
                math.isclose(val_metrics["loss"], best_loss, abs_tol=1e-7)
                and val_metrics["balanced_accuracy"] > best_balanced_accuracy
            )
        )
        if improved:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_loss = float(val_metrics["loss"])
            best_balanced_accuracy = float(val_metrics["balanced_accuracy"])
            best_metrics = dict(val_metrics)
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("training ended without a validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "best_val_loss": float(best_loss),
        "best_val_balanced_accuracy": float(best_balanced_accuracy),
        "best_val_macro_f1": float(best_metrics["macro_f1"]),
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
    }


def train_fixed_epochs(
    model,
    X,
    y,
    epochs: int,
    device,
    config: dict,
    batch_size: int = 64,
    seed: int = 0,
    scheduler_t_max: int | None = None,
):
    """Retrain on all outer-training speakers using the validated LR horizon."""
    set_reproducible_seed(seed)
    model.to(device)
    augment_config = {
        "time_mask": config.get("time_mask", 16),
        "freq_mask": config.get("freq_mask", 8),
        "noise_std": config.get("noise_std", 0.1),
    }
    loader = _loader(
        X,
        y,
        batch_size=batch_size,
        shuffle=True,
        augment=bool(config.get("augment", True)),
        seed=seed,
        augment_config=augment_config,
    )
    n_classes = int(np.max(y) + 1)
    criterion = _criterion(
        y,
        n_classes,
        bool(config.get("balanced_loss", True)),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler_t_max = epochs if scheduler_t_max is None else scheduler_t_max
    if scheduler_t_max < epochs:
        raise ValueError("scheduler_t_max cannot be smaller than training epochs")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, scheduler_t_max)
    )
    started = time.perf_counter()
    last_loss = float("nan")
    for _ in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        last_loss = float(np.mean(losses))
    return model, {
        "epochs": int(epochs),
        "scheduler_t_max": int(scheduler_t_max),
        "last_train_loss": last_loss,
        "train_seconds": float(time.perf_counter() - started),
    }


def config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def sample_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float("nan")
    return float(values.mean()), float(values.std(ddof=1))


def paired_comparison(a, b) -> dict:
    """Paired comparison for vectors ``b - a`` with small-sample safeguards."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired vectors must be one-dimensional with equal shape")
    if len(a) < 2:
        raise ValueError("paired comparison needs at least two pairs")
    difference = b - a
    mean = float(difference.mean())
    sample_sd = float(difference.std(ddof=1))
    sem = sample_sd / math.sqrt(len(difference))
    critical = float(stats.t.ppf(0.975, len(difference) - 1))
    ci = [mean - critical * sem, mean + critical * sem]
    if np.allclose(difference, 0.0):
        t_statistic = 0.0
        t_pvalue = 1.0
        wilcoxon_p = 1.0
    else:
        t_result = stats.ttest_1samp(difference, 0.0)
        t_statistic = float(t_result.statistic)
        t_pvalue = float(t_result.pvalue)
        try:
            wilcoxon_p = float(
                stats.wilcoxon(
                    difference,
                    zero_method="wilcox",
                    alternative="two-sided",
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0

    if len(difference) <= 20:
        observed = abs(mean)
        total = 2 ** len(difference)
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(difference)):
            permuted = float(np.mean(difference * np.asarray(signs)))
            if abs(permuted) >= observed - 1e-12:
                extreme += 1
        sign_flip_p = extreme / total
    else:
        sign_flip_p = float("nan")

    return {
        "n_pairs": int(len(difference)),
        "mean_difference": mean,
        "sample_sd_difference": sample_sd,
        "standard_error": sem,
        "ci95": [float(ci[0]), float(ci[1])],
        "paired_t": t_statistic,
        "paired_t_p": t_pvalue,
        "wilcoxon_p": wilcoxon_p,
        "exact_sign_flip_p": float(sign_flip_p),
        "cohens_dz": mean / sample_sd if sample_sd > 0 else float("inf"),
        "positive_pairs": int(np.sum(difference > 0)),
        "negative_pairs": int(np.sum(difference < 0)),
        "differences": difference.tolist(),
    }


def aggregate_by_fold(rows, metrics=("accuracy", "balanced_accuracy", "macro_f1")):
    """Average initialization seeds within fold, then summarize across folds."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        model = str(row["model"])
        fold = int(row["fold"])
        for metric in metrics:
            grouped[model][fold][metric].append(float(row[metric]))

    summary = {}
    fold_values = {}
    for model, folds in grouped.items():
        fold_values[model] = {}
        summary[model] = {}
        for fold, metric_values in folds.items():
            fold_values[model][str(fold)] = {
                metric: float(np.mean(values))
                for metric, values in metric_values.items()
            }
        for metric in metrics:
            values = [
                fold_values[model][str(fold)][metric]
                for fold in sorted(folds)
            ]
            mean, sample_sd = sample_mean_std(values)
            summary[model][metric] = {
                "mean": mean,
                "sample_sd": sample_sd,
                "n_folds": len(values),
            }
    return summary, fold_values

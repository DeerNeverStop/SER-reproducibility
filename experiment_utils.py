"""
experiment_utils.py
===================
Shared helpers for the follow-up SER/FNO experiments.

These helpers keep the newer experiments honest without changing the original
scripts that produced the report numbers:

* split validation speakers only from the training speakers;
* choose neural-network checkpoints by validation accuracy, not test accuracy;
* evaluate the held-out test speakers once per trained checkpoint;
* write compact CSV/JSON artifacts for later reporting.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for repeatable splits and initializations."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested="auto"):
    """Resolve a user device request into 'cpu' or 'cuda'."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def split_train_val_indices(indices, y, groups, val_size=0.2, seed=0):
    """Split existing training indices into fit/validation indices by speaker.

    The validation set is drawn only from speakers already assigned to the
    training side of an outer split. This keeps held-out test speakers untouched.
    If a tiny synthetic dataset has too few groups, fall back to a stratified
    random split so smoke tests can still run.
    """
    indices = np.asarray(indices)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(indices) < 2:
        raise ValueError("need at least two samples to create a validation split")

    local_y = y[indices]
    local_groups = groups[indices]
    unique_groups = np.unique(local_groups)
    if len(unique_groups) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        fit_local, val_local = next(
            gss.split(np.zeros(len(indices)), local_y, local_groups)
        )
        return indices[fit_local], indices[val_local]

    counts = np.bincount(local_y.astype(int))
    stratify = local_y if np.all(counts[counts > 0] >= 2) else None
    fit_local, val_local = train_test_split(
        np.arange(len(indices)),
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    return indices[fit_local], indices[val_local]


def _loader(X, y=None, batch_size=16, shuffle=False):
    x_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    if y is None:
        ds = TensorDataset(x_tensor)
    else:
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.long)
        ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model, X, y, device, batch_size=16):
    """Return accuracy for a neural model on a fixed dataset."""
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in _loader(X, batch_size=batch_size):
            preds.append(model(xb.to(device)).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    return float(accuracy_score(y, pred))


def train_model_with_validation(
    model,
    X_fit,
    y_fit,
    X_val,
    y_val,
    epochs,
    device,
    lr=1e-3,
    batch_size=16,
    patience=12,
    weight_decay=1e-4,
):
    """Train a neural model and restore the best validation checkpoint.

    Selection uses validation accuracy, with validation loss as a tie-breaker.
    The test set is intentionally not visible to this function.
    """
    model.to(device)
    train_loader = _loader(X_fit, y_fit, batch_size=batch_size, shuffle=True)
    val_loader = _loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    best_state = None
    best_epoch = 0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses, preds, gts = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb_dev = yb.to(device)
                logits = model(xb)
                val_losses.append(float(crit(logits, yb_dev).cpu()))
                preds.append(logits.argmax(1).cpu())
                gts.append(yb)
        val_loss = float(np.mean(val_losses))
        val_acc = float(accuracy_score(torch.cat(gts).numpy(), torch.cat(preds).numpy()))

        improved = (
            val_acc > best_val_acc
            or (val_acc == best_val_acc and val_loss < best_val_loss)
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1

        if patience > 0 and stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("training finished without a validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "epochs_run": int(epoch),
        "train_seconds": float(time.perf_counter() - started),
    }


def matrix_gap(matrix, rates):
    """Compute diagonal, off-diagonal and transfer gap for an accuracy matrix."""
    diag = float(np.mean([matrix[(r, r)] for r in rates]))
    off = float(np.mean([matrix[(a, b)] for a in rates for b in rates if a != b]))
    return diag, off, diag - off


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def paired_noise(a, b):
    """Rough seed-to-seed noise estimate used by the original verdict logic."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(a.var() + b.var()) + 1e-9)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

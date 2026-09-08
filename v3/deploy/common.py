"""Shared helpers for SER26-DEPLOY-1 (v3/deploy). Read-only access to v2 data; LF files; no v2/ edits.

Layout assumptions
- REPO_ROOT: this worktree (tracked code, v2/manifests, v2/plan_rc2/split_index.json, v2/registry).
- V2_DATA_ROOT (env SER_V2_DATA_ROOT, default E:/科研/essay/SER-v2/v2): untracked v2 artifacts that stay local:
  features/*.npz, runs/main/units/*, plan_rc2/splits/*.json. Split files are verified against split_index.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROGRAM = "SER26-DEPLOY-2"
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPS = 10000
REPO_ROOT = Path(__file__).resolve().parents[2]
V2_DATA_ROOT = Path(os.environ.get("SER_V2_DATA_ROOT", r"E:\科研\essay\SER-v2\v2"))
DEPLOY_ROOT = REPO_ROOT / "v3" / "deploy"
WORK_ROOT = Path(os.environ.get("SER_DEPLOY_WORK", str(DEPLOY_ROOT / "work")))
LEVELS = {"ravdess": "ravdess", "cremad": "cremad", "subesco_980": "subesco"}   # level -> base corpus
ENCODERS = ("wavlm_base_plus", "hubert_base", "wav2vec2_base")
STATE = 12


class IntegrityError(ValueError):
    pass


def require(cond, msg):
    if not cond:
        raise IntegrityError(msg)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ranked(items, salt: str):
    return sorted(items, key=lambda it: (digest([salt, it]), it))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- CPU guard

def cpu_guard(threads: int = 2) -> dict:
    """Make this process invisible to CUDA, limit BLAS/torch threads, lower priority on Windows."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    priority = "unchanged"
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        if kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            priority = "below_normal"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=threads)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
        require(not torch.cuda.is_initialized(), "CUDA was initialised in a CPU-only process")
    except ImportError:
        pass
    return {"cuda_visible_devices": "-1", "threads": threads, "priority": priority}


# ----------------------------------------------------------------------------- manifests / labels

def manifest_path(base: str) -> Path:
    return REPO_ROOT / "v2" / "manifests" / f"{base}_manifest.csv"


def load_manifest(base: str) -> dict:
    """relative_path -> row (speaker, sex, label, label_index, sentence, take, intensity, sha256)."""
    rows = {}
    with open(manifest_path(base), encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            r["label_index"] = int(r["label_index"])
            rows[r["relative_path"]] = r
    require(rows, f"empty manifest for {base}")
    return rows


def class_table(manifest: dict) -> list:
    """Fixed class order: labels sorted by label_index (as v2)."""
    pairs = sorted({(r["label_index"], r["label"]) for r in manifest.values()})
    require([p[0] for p in pairs] == list(range(len(pairs))), "label_index must be contiguous from 0")
    return [p[1] for p in pairs]


def labels_for(manifest: dict, paths) -> np.ndarray:
    return np.asarray([manifest[p]["label_index"] for p in paths], dtype=np.int64)


def speakers_for(manifest: dict, paths) -> list:
    return [manifest[p]["speaker"] for p in paths]


# ----------------------------------------------------------------------------- split tables

def split_index() -> dict:
    return read_json(REPO_ROOT / "v2" / "plan_rc2" / "split_index.json")


def load_split(key: str, verify: bool = True) -> dict:
    """v2 split table {key, population, folds:[{fold, test, val, meta}]}; fit = population - test - val."""
    idx = split_index()
    require(key in idx, f"unknown split key {key}")
    entry = idx[key]
    path = V2_DATA_ROOT / "plan_rc2" / entry["path"]
    require(path.is_file(), f"split file missing locally: {path}")
    if verify:
        require(sha256_file(path) == entry["sha256"], f"split file sha mismatch for {key}")
    table = read_json(path)
    pop = set(table["population"])
    for f in table["folds"]:
        test, val = set(f["test"]), set(f["val"])
        require(test <= pop and val <= pop and not (test & val), f"malformed fold in {key}")
        f["fit"] = sorted(pop - test - val)
    return table


def outer_fold_speakers(manifest: dict, fold: dict) -> dict:
    return {role: sorted({manifest[p]["speaker"] for p in fold[role]}) for role in ("fit", "val", "test")}


# ----------------------------------------------------------------------------- features

class FeatureCache:
    """Loads one v2 npz (X, paths) lazily and serves rows by relative_path."""

    def __init__(self, base: str, kind: str):
        self.base, self.kind = base, kind
        cands = sorted((V2_DATA_ROOT / "features").glob(f"{base}__{kind}__*.npz"))
        require(len(cands) == 1, f"expected exactly one cache for {base}/{kind}, found {len(cands)}")
        self.path = cands[0]
        self.sha256 = sha256_file(self.path)
        z = np.load(self.path, allow_pickle=False)
        self.X = z["X"]
        self.index = {p: i for i, p in enumerate(z["paths"].tolist())}
        require(len(self.index) == self.X.shape[0], "duplicate paths in feature cache")

    def get(self, paths, state=None) -> np.ndarray:
        idx = np.fromiter((self.index[p] for p in paths), dtype=np.int64, count=len(paths))
        X = self.X[idx]
        if state is not None:
            require(X.ndim == 3, "state selection needs a 3-d SSL cache")
            X = X[:, state, :]
        return np.ascontiguousarray(X, dtype=np.float32)


# ----------------------------------------------------------------------------- metrics

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def uar(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted average recall over the FIXED class set; classes absent from y_true are skipped
    (v2 convention) - callers that need strict fixed-class support should check support first."""
    recalls = []
    for c in range(n_classes):
        m = y_true == c
        if m.any():
            recalls.append(float((y_pred[m] == c).mean()))
    require(recalls, "no classes present")
    return 100.0 * float(np.mean(recalls))


def per_speaker_uar(y_true, y_pred, speakers, n_classes: int) -> dict:
    y_true, y_pred, speakers = np.asarray(y_true), np.asarray(y_pred), np.asarray(speakers)
    return {s: uar(y_true[speakers == s], y_pred[speakers == s], n_classes) for s in sorted(set(speakers.tolist()))}


def class_recall(y_true, y_pred, c: int) -> float:
    m = np.asarray(y_true) == c
    return float("nan") if not m.any() else 100.0 * float((np.asarray(y_pred)[m] == c).mean())


def bootstrap_indices(n_speakers: int, seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_speakers, size=(reps, n_speakers))


def summarize_paired(values: np.ndarray, indices: np.ndarray) -> dict:
    """values: per-speaker vector (speaker order fixed & sorted); percentile 95% CI of the mean."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and indices.shape[1] == values.shape[0], "bootstrap shape mismatch")
    sampled = values[indices].mean(axis=1)
    lo, hi = np.percentile(sampled, [2.5, 97.5], method="linear")
    return {"estimate": float(values.mean()), "ci95": [float(lo), float(hi)], "n": int(values.shape[0])}


# ----------------------------------------------------------------------------- prediction files

def write_predictions_csv(path: Path, paths, speakers, y_true, y_pred, scores: np.ndarray, score_prefix: str) -> str:
    """Write a prediction table; returns its sha256. Columns: relative_path,speaker,y_true,y_pred,<prefix>_0.."""
    scores = np.asarray(scores, dtype=np.float64)
    header = ["relative_path", "speaker", "y_true", "y_pred"] + [f"{score_prefix}_{i}" for i in range(scores.shape[1])]
    lines = [",".join(header)]
    for p, s, yt, yp, row in zip(paths, speakers, y_true, y_pred, scores):
        lines.append(",".join([p, s, str(int(yt)), str(int(yp))] + [repr(float(v)) for v in row]))
    atomic_write_text(path, "\n".join(lines) + "\n")
    return sha256_file(path)


def read_predictions_csv(path: Path, score_prefix: str):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in rows[0].keys() if c.startswith(score_prefix + "_")]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    scores = np.asarray([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)
    return {
        "paths": [r["relative_path"] for r in rows],
        "speakers": [r["speaker"] for r in rows],
        "y_true": np.asarray([int(r["y_true"]) for r in rows]),
        "y_pred": np.asarray([int(r["y_pred"]) for r in rows]),
        "scores": scores,
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0

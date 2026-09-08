"""Frozen engineering core for project-07 P1 protocol-premium measurement.

This module contains only deterministic input, preprocessing, split, and
artifact helpers.  It never trains a model and never reads an outer-test
prediction.  The formal runner imports the four model implementations from the
read-only SER repository after verifying the preregistered commit and hashes.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import platform
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import scipy
import sklearn
import soundfile as sf
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SER_ROOT = Path(r"E:\科研\SER")
RESULT_ROOT = PROJECT_ROOT / "results" / "protocol_premium"
PREREG_PATH = PROJECT_ROOT / "P1_PREREGISTRATION.md"

PREREG_SHA256 = "91db3edc544f76fc412f4dfc5f35adef23934635ca3ede36c8089c38bd14b782"
PREREG_FROZEN_PREFIX_BYTES = 12_426
SER_COMMIT = "d11768ed31a91f611472b89e52f3f6bb97030c40"
SER_FILE_SHA256 = {
    "advanced_models.py": "c1e18011dbb6aeb1717a4ee5f88afb5e8352ec61044837aeb5b7d70f6fe26ad5",
    "advanced_experiment_utils.py": "ea175e3f7366feeee505d2441796268f58e6daae7aa6b625e6d0cb78a4b0e04e",
    "fno_data.py": "9762eeb91268258443dd8a1908336d2cffeed9097f6e2c66461f4419ce0f3c88",
    "tuned_standard_experiment.py": "865f60a6d27f28514b57ac174c9a3432382718f300ebe4d75d1f280d81402bff",
    "fno_model.py": "c24555bf21f7bd0cb09fa7861450b579818422671939b08b51b01c3d915242f9",
}

MODELS = ("cnn", "resnet_se", "transformer", "fno")
PROTOCOLS = ("random", "groupkfold", "loso")
SEEDS = (0, 1, 2)
BOOTSTRAP_SEED = 202608101744
BOOTSTRAP_REPLICATES = 10_000

PREPROCESS_CONFIG = {
    "version": "p1_logmel_v1",
    "sr": 22050,
    "mono": True,
    "trim_top_db": 30,
    "minimum_seconds": 0.1,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
    "fmax": 11025,
    "frames": 128,
    "power_to_db_ref": "np.max",
    "time_policy": "right-pad-with-spectrogram-min-or-left-truncate",
    "normalization": "per-sample-global-zscore",
    "eps": 1e-6,
}

CORPORA = {
    "ravdess": {
        "root": SER_ROOT / "data",
        "expected_n": 1440,
        "expected_speakers": 24,
        "labels": ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"),
        "license": "CC BY-NC-SA 4.0",
        "source": "https://doi.org/10.5281/zenodo.1188976",
        "outer_folds": {"random": 5, "groupkfold": 5, "loso": 24},
    },
    "cremad": {
        "root": SER_ROOT / "AudioWAV",
        "expected_n": 7442,
        "expected_speakers": 91,
        "labels": ("angry", "disgust", "fearful", "happy", "neutral", "sad"),
        "license": "ODbL 1.0 / DbCL 1.0",
        "source": "https://github.com/CheyneyComputerScience/CREMA-D",
        "outer_folds": {"random": 5, "groupkfold": 5, "loso": 91},
    },
}

EXPECTED_MODEL_CONFIGS = {
    "cnn": {
        "width": 48, "n_blocks": 3, "kernel_size": 5, "dilation_base": 1,
        "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
        "augment": True, "balanced_loss": True, "time_mask": 16,
        "freq_mask": 8, "noise_std": 0.1,
    },
    "resnet_se": {
        "width": 48, "n_blocks": 4, "kernel_size": 5, "max_dilation": 4,
        "dropout": 0.1, "se_reduction": 4, "lr": 1e-3,
        "weight_decay": 1e-4, "augment": True, "balanced_loss": True,
        "time_mask": 16, "freq_mask": 8, "noise_std": 0.1,
    },
    "transformer": {
        "d_model": 96, "n_heads": 4, "n_layers": 2, "ff_mult": 2.0,
        "dropout": 0.1, "position_mode": "normalized", "pooling": "attentive",
        "lr": 3e-4, "weight_decay": 1e-4, "augment": True,
        "balanced_loss": True, "time_mask": 16, "freq_mask": 8,
        "noise_std": 0.1,
    },
    "fno": {
        "width": 48, "modes": 16, "n_layers": 4, "dropout": 0.1,
        "lr": 1e-3, "weight_decay": 1e-4, "augment": True,
        "balanced_loss": True, "time_mask": 16, "freq_mask": 8,
        "noise_std": 0.1,
    },
}

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
CREMAD_EMOTIONS = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u32(text: str) -> int:
    """First 32 bits of SHA256(UTF-8 exact key), interpreted big-endian."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp.npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git_head(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def assert_frozen_inputs() -> dict:
    prereg_bytes = PREREG_PATH.read_bytes()
    observed = {
        "preregistration_frozen_prefix_bytes": PREREG_FROZEN_PREFIX_BYTES,
        "preregistration_frozen_prefix_sha256": sha256_bytes(prereg_bytes[:PREREG_FROZEN_PREFIX_BYTES]),
        "preregistration_full_sha256": sha256_bytes(prereg_bytes),
        "ser_commit": git_head(SER_ROOT),
        "ser_file_sha256": {name: sha256_file(SER_ROOT / name) for name in SER_FILE_SHA256},
    }
    errors = []
    if len(prereg_bytes) < PREREG_FROZEN_PREFIX_BYTES:
        errors.append("P1_PREREGISTRATION.md was truncated")
    elif observed["preregistration_frozen_prefix_sha256"] != PREREG_SHA256:
        errors.append("P1_PREREGISTRATION.md frozen prefix drift")
    if observed["ser_commit"] != SER_COMMIT:
        errors.append("SER commit drift")
    for name, expected in SER_FILE_SHA256.items():
        if observed["ser_file_sha256"][name] != expected:
            errors.append(f"SER file hash drift: {name}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return observed


def verify_model_configs() -> dict:
    assert_frozen_inputs()
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SER_ROOT))
    try:
        from tuned_standard_experiment import candidate_configs
        observed = {model: candidate_configs(model)[2] for model in MODELS}
    finally:
        if sys.path and sys.path[0] == str(SER_ROOT):
            sys.path.pop(0)
        sys.dont_write_bytecode = old_dont_write
    if canonical_json(observed) != canonical_json(EXPECTED_MODEL_CONFIGS):
        raise RuntimeError("candidate_configs(model)[2] drifted from preregistration")
    return observed


def parse_label(corpus: str, filename: str) -> tuple[str | None, str | None, str | None]:
    stem = Path(filename).stem
    if corpus == "ravdess":
        parts = stem.split("-")
        if len(parts) != 7 or parts[2] not in RAVDESS_EMOTIONS or not parts[6].isdigit():
            return None, None, "invalid_ravdess_filename"
        return RAVDESS_EMOTIONS[parts[2]], f"{int(parts[6]):02d}", None
    if corpus == "cremad":
        parts = stem.split("_")
        if len(parts) != 4 or not parts[0].isdigit() or parts[2] not in CREMAD_EMOTIONS:
            return None, None, "invalid_cremad_filename"
        return CREMAD_EMOTIONS[parts[2]], str(int(parts[0])), None
    raise ValueError(corpus)


MANIFEST_FIELDS = [
    "sample_index", "corpus", "relative_path", "file_size_bytes", "sha256",
    "label_name", "label_index", "speaker_id", "wav_samplerate", "wav_frames",
    "wav_channels", "wav_subtype", "parse_status", "read_status", "error",
]


def prepare_manifest(corpus: str) -> tuple[Path, list[dict[str, str]]]:
    cfg = CORPORA[corpus]
    root = Path(cfg["root"])
    out = RESULT_ROOT / "manifests" / f"{corpus}_manifest.csv"
    if out.exists():
        rows = read_csv(out)
        if len(rows) != int(cfg["expected_n"]):
            raise RuntimeError(f"existing {corpus} manifest row count drift")
        return out, rows

    paths = sorted(root.rglob("*.wav"), key=lambda p: p.relative_to(root).as_posix())
    rows: list[dict[str, str]] = []
    labels = tuple(cfg["labels"])
    for index, path in enumerate(paths):
        label, speaker, parse_error = parse_label(corpus, path.name)
        read_status = "ok"
        error = parse_error or ""
        samplerate = frames = channels = subtype = ""
        try:
            info = sf.info(str(path))
            samplerate = str(info.samplerate)
            frames = str(info.frames)
            channels = str(info.channels)
            subtype = str(info.subtype)
        except Exception as exc:  # recorded and then rejected by the fixed-population gate
            read_status = "error"
            error = (error + "; " if error else "") + f"{type(exc).__name__}: {exc}"
        rows.append({
            "sample_index": str(index),
            "corpus": corpus,
            "relative_path": path.relative_to(root).as_posix(),
            "file_size_bytes": str(path.stat().st_size),
            "sha256": sha256_file(path),
            "label_name": label or "",
            "label_index": str(labels.index(label)) if label in labels else "",
            "speaker_id": speaker or "",
            "wav_samplerate": samplerate,
            "wav_frames": frames,
            "wav_channels": channels,
            "wav_subtype": subtype,
            "parse_status": "ok" if parse_error is None else "error",
            "read_status": read_status,
            "error": error,
        })
    atomic_write_csv(out, MANIFEST_FIELDS, rows)
    return out, rows


def validate_manifest(corpus: str, path: Path, rows: list[dict[str, str]]) -> dict:
    cfg = CORPORA[corpus]
    errors = []
    if len(rows) != int(cfg["expected_n"]):
        errors.append(f"n={len(rows)} expected={cfg['expected_n']}")
    indices = [int(row["sample_index"]) for row in rows]
    if indices != list(range(len(rows))):
        errors.append("sample_index is not contiguous canonical order")
    bad = [row for row in rows if row["parse_status"] != "ok" or row["read_status"] != "ok"]
    if bad:
        errors.append(f"parse/read failures={len(bad)}")
    speakers = sorted({row["speaker_id"] for row in rows})
    if len(speakers) != int(cfg["expected_speakers"]):
        errors.append(f"speakers={len(speakers)} expected={cfg['expected_speakers']}")
    labels = Counter(row["label_name"] for row in rows)
    if set(labels) != set(cfg["labels"]):
        errors.append(f"label set drift={sorted(labels)}")
    relpaths = [row["relative_path"] for row in rows]
    if relpaths != sorted(relpaths) or len(set(relpaths)) != len(relpaths):
        errors.append("relative paths are not sorted unique")
    if errors:
        raise RuntimeError(f"{corpus} fixed-population gate failed: " + "; ".join(errors))
    return {
        "corpus": corpus,
        "manifest_path": path.relative_to(RESULT_ROOT).as_posix(),
        "manifest_sha256": sha256_file(path),
        "n_samples": len(rows),
        "n_speakers": len(speakers),
        "n_labels": len(labels),
        "label_counts": dict(sorted(labels.items())),
        "license": cfg["license"],
        "source": cfg["source"],
        "parse_failures": 0,
        "read_failures": 0,
    }


def compute_logmel(path: Path) -> np.ndarray:
    cfg = PREPROCESS_CONFIG
    y, _ = librosa.load(str(path), sr=int(cfg["sr"]), mono=True)
    y, _ = librosa.effects.trim(y, top_db=float(cfg["trim_top_db"]))
    minimum = int(round(float(cfg["minimum_seconds"]) * int(cfg["sr"])))
    if y.size < minimum:
        y = np.pad(y, (0, minimum - y.size))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=int(cfg["sr"]),
        n_mels=int(cfg["n_mels"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        fmax=float(cfg["fmax"]),
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    frames = int(cfg["frames"])
    if logmel.shape[1] < frames:
        logmel = np.pad(
            logmel,
            ((0, 0), (0, frames - logmel.shape[1])),
            mode="constant",
            constant_values=float(logmel.min()),
        )
    else:
        logmel = logmel[:, :frames]
    logmel = logmel.astype(np.float32, copy=False)
    logmel = (logmel - logmel.mean(dtype=np.float64)) / (logmel.std(dtype=np.float64) + float(cfg["eps"]))
    return np.asarray(logmel, dtype=np.float32)


def preprocessing_code_hash() -> str:
    payload = canonical_json(PREPROCESS_CONFIG) + "\n" + inspect.getsource(compute_logmel)
    return sha256_bytes(payload.encode("utf-8"))


def cache_path(corpus: str) -> Path:
    code = preprocessing_code_hash()[:16]
    name = (
        f"mel_{corpus}_v1_sr22050_mel64_nfft1024_hop512_fmax11025_"
        f"frames128_trim30_sample-zscore_code-{code}.npz"
    )
    return RESULT_ROOT / "cache" / name


def prepare_cache(corpus: str, manifest_path: Path, rows: list[dict[str, str]]) -> dict:
    out = cache_path(corpus)
    manifest_sha = sha256_file(manifest_path)
    signature = {
        "corpus": corpus,
        "manifest_sha256": manifest_sha,
        "preprocess": PREPROCESS_CONFIG,
        "preprocessing_code_hash": preprocessing_code_hash(),
    }
    if out.exists():
        with np.load(out, allow_pickle=False) as data:
            if tuple(data["X"].shape) != (len(rows), 64, 128):
                raise RuntimeError(f"existing cache shape drift: {out}")
            observed_signature = json.loads(str(data["signature_json"].item()))
            if canonical_json(observed_signature) != canonical_json(signature):
                raise RuntimeError(f"existing cache signature drift: {out}")
            if not np.isfinite(data["X"]).all():
                raise RuntimeError(f"non-finite values in existing cache: {out}")
        return {
            "corpus": corpus,
            "cache_path": out.relative_to(RESULT_ROOT).as_posix(),
            "cache_sha256": sha256_file(out),
            "cache_shape": [len(rows), 64, 128],
            "signature": signature,
        }

    root = Path(CORPORA[corpus]["root"])
    X = np.empty((len(rows), 64, 128), dtype=np.float32)
    errors = []
    for index, row in enumerate(rows):
        try:
            X[index] = compute_logmel(root / row["relative_path"])
        except Exception as exc:
            errors.append({
                "sample_index": row["sample_index"],
                "relative_path": row["relative_path"],
                "error": f"{type(exc).__name__}: {exc}",
            })
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"[{corpus}] preprocessed {index + 1}/{len(rows)}", flush=True)
    error_path = RESULT_ROOT / "cache" / f"{corpus}_preprocessing_errors.csv"
    atomic_write_csv(error_path, ["sample_index", "relative_path", "error"], errors)
    if errors:
        raise RuntimeError(f"{corpus} preprocessing failed for {len(errors)} fixed-population files")
    if not np.isfinite(X).all():
        raise RuntimeError(f"{corpus} preprocessing produced non-finite values")
    atomic_save_npz(out, X=X, signature_json=np.asarray(canonical_json(signature)))
    return {
        "corpus": corpus,
        "cache_path": out.relative_to(RESULT_ROOT).as_posix(),
        "cache_sha256": sha256_file(out),
        "cache_shape": list(X.shape),
        "signature": signature,
    }


def load_cache(corpus: str, expected_sha256: str | None = None) -> np.ndarray:
    path = cache_path(corpus)
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"cache hash drift: {path}")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["X"], dtype=np.float32)


def manifest_arrays(corpus: str, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([int(row["label_index"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["speaker_id"] for row in rows])
    return y, groups


def generate_outer_splits(corpus: str, y: np.ndarray, groups: np.ndarray) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    indices = np.arange(len(y), dtype=np.int64)
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    splits["random"] = [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(indices, y)
    ]
    splits["groupkfold"] = [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in GroupKFold(n_splits=5).split(indices, y, groups)
    ]
    loso = []
    for speaker in sorted(np.unique(groups).tolist()):
        test = indices[groups == speaker]
        train = indices[groups != speaker]
        loso.append((train, test))
    splits["loso"] = loso
    for protocol, folds in splits.items():
        expected = int(CORPORA[corpus]["outer_folds"][protocol])
        if len(folds) != expected:
            raise RuntimeError(f"{corpus}/{protocol} folds={len(folds)} expected={expected}")
        tests = np.concatenate([test for _, test in folds])
        if not np.array_equal(np.sort(tests), indices) or len(np.unique(tests)) != len(indices):
            raise RuntimeError(f"{corpus}/{protocol} outer test is not exactly-once OOF")
        for train, test in folds:
            if np.intersect1d(train, test).size:
                raise RuntimeError(f"{corpus}/{protocol} train/test index overlap")
            if protocol != "random" and set(groups[train]) & set(groups[test]):
                raise RuntimeError(f"{corpus}/{protocol} speaker overlap")
    return splits


def fold_hash(train: np.ndarray, test: np.ndarray) -> str:
    payload = canonical_json({"train": train.tolist(), "test": test.tolist()})
    return sha256_bytes(payload.encode("utf-8"))


def inner_split(
    corpus: str,
    protocol: str,
    outer_fold: int,
    seed: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    key = f"P1-inner|{corpus}|{protocol}|{outer_fold}|{seed}"
    base = stable_u32(key)
    all_labels = set(range(len(CORPORA[corpus]["labels"])))
    if protocol == "random":
        fit, val = train_test_split(
            outer_train,
            test_size=0.25,
            stratify=y[outer_train],
            random_state=base,
        )
        fit = np.sort(np.asarray(fit, dtype=np.int64))
        val = np.sort(np.asarray(val, dtype=np.int64))
        attempt = 0
    else:
        fit = val = None
        attempt = -1
        for offset in range(100):
            state = (base + offset) & 0xFFFFFFFF
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=state)
            local_fit, local_val = next(splitter.split(outer_train, y[outer_train], groups[outer_train]))
            candidate_fit = np.sort(outer_train[local_fit].astype(np.int64))
            candidate_val = np.sort(outer_train[local_val].astype(np.int64))
            if set(y[candidate_fit].tolist()) == all_labels and set(y[candidate_val].tolist()) == all_labels:
                fit, val, attempt = candidate_fit, candidate_val, offset
                break
        if fit is None or val is None:
            raise RuntimeError(f"inner_split_infeasible:{corpus}/{protocol}/fold={outer_fold}/seed={seed}")
    if np.intersect1d(fit, val).size or np.intersect1d(fit, outer_test).size or np.intersect1d(val, outer_test).size:
        raise RuntimeError("inner fit/validation/outer-test index overlap")
    if not np.array_equal(np.sort(np.concatenate([fit, val])), np.sort(outer_train)):
        raise RuntimeError("inner fit+validation does not partition outer train")
    if set(y[fit].tolist()) != all_labels or set(y[val].tolist()) != all_labels:
        raise RuntimeError("inner split missing a class")
    if protocol != "random" and set(groups[fit]) & set(groups[val]):
        raise RuntimeError("strict inner split has speaker overlap")
    return fit, val, base, attempt


def prepare_splits(corpus: str, rows: list[dict[str, str]]) -> dict:
    split_dir = RESULT_ROOT / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    y, groups = manifest_arrays(corpus, rows)
    splits = generate_outer_splits(corpus, y, groups)
    assignment_rows = []
    fold_rows = []
    inner_rows = []
    for protocol in PROTOCOLS:
        for fold_index, (train, test) in enumerate(splits[protocol]):
            fhash = fold_hash(train, test)
            train_speakers = set(groups[train].tolist())
            test_speakers = set(groups[test].tolist())
            overlap = train_speakers & test_speakers
            for sample_index in test.tolist():
                assignment_rows.append({
                    "corpus": corpus,
                    "protocol": protocol,
                    "outer_fold": str(fold_index),
                    "sample_index": str(sample_index),
                    "relative_path": rows[sample_index]["relative_path"],
                    "speaker_id": rows[sample_index]["speaker_id"],
                    "label_index": rows[sample_index]["label_index"],
                    "outer_fold_sha256": fhash,
                })
            fold_rows.append({
                "corpus": corpus,
                "protocol": protocol,
                "outer_fold": str(fold_index),
                "n_train": str(len(train)),
                "n_test": str(len(test)),
                "n_train_speakers": str(len(train_speakers)),
                "n_test_speakers": str(len(test_speakers)),
                "n_overlap": str(len(overlap)),
                "overlap_fraction": format(len(overlap) / max(len(test_speakers), 1), ".17g"),
                "test_speaker_ids": "|".join(sorted(test_speakers)),
                "outer_fold_sha256": fhash,
            })
            for seed in SEEDS:
                fit, val, base, attempt = inner_split(
                    corpus, protocol, fold_index, seed, train, test, y, groups
                )
                rel = Path("splits") / "inner" / corpus / protocol / f"fold_{fold_index:03d}_seed_{seed}.npz"
                path = RESULT_ROOT / rel
                expected_key = f"P1-inner|{corpus}|{protocol}|{fold_index}|{seed}"
                if not path.exists():
                    atomic_save_npz(
                        path,
                        fit_idx=fit,
                        val_idx=val,
                        outer_test_idx=test,
                        key=np.asarray(expected_key),
                        base_u32=np.asarray(base, dtype=np.uint32),
                        accepted_offset=np.asarray(attempt, dtype=np.int64),
                    )
                else:
                    with np.load(path, allow_pickle=False) as existing:
                        comparisons = {
                            "fit_idx": np.array_equal(existing["fit_idx"], fit),
                            "val_idx": np.array_equal(existing["val_idx"], val),
                            "outer_test_idx": np.array_equal(existing["outer_test_idx"], test),
                            "key": str(existing["key"].item()) == expected_key,
                            "base_u32": int(existing["base_u32"].item()) == base,
                            "accepted_offset": int(existing["accepted_offset"].item()) == attempt,
                        }
                    if not all(comparisons.values()):
                        failed = sorted(name for name, ok in comparisons.items() if not ok)
                        raise RuntimeError(f"existing inner split drift {path}: {failed}")
                inner_rows.append({
                    "corpus": corpus,
                    "protocol": protocol,
                    "outer_fold": str(fold_index),
                    "seed": str(seed),
                    "inner_key": expected_key,
                    "base_u32": str(base),
                    "accepted_offset": str(attempt),
                    "n_fit": str(len(fit)),
                    "n_validation": str(len(val)),
                    "n_outer_test": str(len(test)),
                    "inner_split_path": rel.as_posix(),
                    "inner_split_sha256": sha256_file(path),
                    "outer_fold_sha256": fhash,
                })
    assignment_path = split_dir / f"{corpus}_outer_test_assignments.csv"
    fold_path = split_dir / f"{corpus}_outer_fold_summary.csv"
    inner_path = split_dir / f"{corpus}_inner_split_inventory.csv"
    atomic_write_csv(assignment_path, [
        "corpus", "protocol", "outer_fold", "sample_index", "relative_path",
        "speaker_id", "label_index", "outer_fold_sha256",
    ], assignment_rows)
    atomic_write_csv(fold_path, [
        "corpus", "protocol", "outer_fold", "n_train", "n_test",
        "n_train_speakers", "n_test_speakers", "n_overlap", "overlap_fraction",
        "test_speaker_ids", "outer_fold_sha256",
    ], fold_rows)
    atomic_write_csv(inner_path, [
        "corpus", "protocol", "outer_fold", "seed", "inner_key", "base_u32",
        "accepted_offset", "n_fit", "n_validation", "n_outer_test",
        "inner_split_path", "inner_split_sha256", "outer_fold_sha256",
    ], inner_rows)
    return {
        "corpus": corpus,
        "outer_assignments_path": assignment_path.relative_to(RESULT_ROOT).as_posix(),
        "outer_assignments_sha256": sha256_file(assignment_path),
        "outer_fold_summary_path": fold_path.relative_to(RESULT_ROOT).as_posix(),
        "outer_fold_summary_sha256": sha256_file(fold_path),
        "inner_inventory_path": inner_path.relative_to(RESULT_ROOT).as_posix(),
        "inner_inventory_sha256": sha256_file(inner_path),
        "n_outer_folds": sum(len(splits[p]) for p in PROTOCOLS),
        "n_inner_splits": len(inner_rows),
    }


def load_inner_inventory(corpus: str) -> dict[tuple[str, int, int], dict[str, str]]:
    path = RESULT_ROOT / "splits" / f"{corpus}_inner_split_inventory.csv"
    return {
        (row["protocol"], int(row["outer_fold"]), int(row["seed"])): row
        for row in read_csv(path)
    }


def prepare_run_plan(prepared: dict) -> dict:
    plan_path = RESULT_ROOT / "run_plan.csv"
    rows = []
    sequence = 0
    for corpus in ("ravdess", "cremad"):
        inventory = load_inner_inventory(corpus)
        manifest_sha = prepared["corpora"][corpus]["manifest"]["manifest_sha256"]
        cache_sha = prepared["corpora"][corpus]["cache"]["cache_sha256"]
        for model in MODELS:
            for protocol in PROTOCOLS:
                n_folds = int(CORPORA[corpus]["outer_folds"][protocol])
                for seed in SEEDS:
                    for fold in range(n_folds):
                        sequence += 1
                        inv = inventory[(protocol, fold, seed)]
                        unit_id = f"{corpus}__{model}__{protocol}__s{seed}__f{fold:03d}"
                        rows.append({
                            "sequence": str(sequence),
                            "unit_id": unit_id,
                            "corpus": corpus,
                            "model": model,
                            "protocol": protocol,
                            "seed": str(seed),
                            "outer_fold": str(fold),
                            "n_outer_folds": str(n_folds),
                            "n_fit": inv["n_fit"],
                            "n_validation": inv["n_validation"],
                            "n_outer_test": inv["n_outer_test"],
                            "manifest_sha256": manifest_sha,
                            "cache_sha256": cache_sha,
                            "outer_fold_sha256": inv["outer_fold_sha256"],
                            "inner_split_path": inv["inner_split_path"],
                            "inner_split_sha256": inv["inner_split_sha256"],
                        })
    fields = [
        "sequence", "unit_id", "corpus", "model", "protocol", "seed",
        "outer_fold", "n_outer_folds", "n_fit", "n_validation", "n_outer_test",
        "manifest_sha256", "cache_sha256", "outer_fold_sha256",
        "inner_split_path", "inner_split_sha256",
    ]
    atomic_write_csv(plan_path, fields, rows)
    if len(rows) != 1620:
        raise RuntimeError(f"run plan has {len(rows)} units, expected 1620")
    return {
        "path": plan_path.relative_to(RESULT_ROOT).as_posix(),
        "sha256": sha256_file(plan_path),
        "n_units": len(rows),
        "ravdess_units": sum(row["corpus"] == "ravdess" for row in rows),
        "cremad_units": sum(row["corpus"] == "cremad" for row in rows),
    }


def environment_record() -> dict:
    return {
        "created_at": now_iso(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_arch_list": torch.cuda.get_arch_list(),
    }


def output_size_bytes(root: Path = RESULT_ROOT) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0

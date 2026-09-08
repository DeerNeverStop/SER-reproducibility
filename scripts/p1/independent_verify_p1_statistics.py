"""Independent P1 statistical verification from frozen prediction artifacts.

This CPU-only verifier intentionally does not import the primary summarizer,
the existing verifier, the training runner, or SER model code.  It rebuilds the
complete OOF cells from outer-fold prediction CSVs, recomputes every frozen
descriptive and paired statistic, and compares them with the published summary
tables.  The default command is for the final, complete 72-cell P1 matrix; any
missing or invalid prediction unit is a hard verification failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "protocol_premium"

MODELS = ("cnn", "resnet_se", "transformer", "fno")
PROTOCOLS = ("random", "groupkfold", "loso")
SEEDS = (0, 1, 2)
METRICS = ("uar", "accuracy", "macro_f1")
CONTRASTS = {
    "RG": ("random", "groupkfold"),
    "RL": ("random", "loso"),
    "GL": ("groupkfold", "loso"),
}
BOOTSTRAP_SEED = 202608101744
BOOTSTRAP_REPLICATES = 10_000
HOLM_ALPHA = 0.05
NUMERIC_TOLERANCE = 5e-12

CORPORA: dict[str, dict[str, Any]] = {
    "ravdess": {
        "expected_n": 1440,
        "expected_speakers": 24,
        "labels": ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"),
        "folds": {"random": 5, "groupkfold": 5, "loso": 24},
    },
    "cremad": {
        "expected_n": 7442,
        "expected_speakers": 91,
        "labels": ("angry", "disgust", "fearful", "happy", "neutral", "sad"),
        "folds": {"random": 5, "groupkfold": 5, "loso": 91},
    },
}

FROZEN_INPUT_SHA256 = {
    "run_plan.csv": "ffac41581a09a769d7f299c2e27a94f75d7ca6b594196b3c81843cbe4d01296a",
    "manifests/ravdess_manifest.csv": "259b453ce6135242b03a34d113209a4b8a0234687c9717e56fbe2bd75acb7ab2",
    "manifests/cremad_manifest.csv": "8b628265bcea9ed36ac96a0fffe753423a8c4d8c87befb5775a2ca36c673c9b8",
    "splits/ravdess_outer_test_assignments.csv": "6cfa5aea57eb03b4fa1f1e8596bd081ee83866436775948d0632defaa91b953b",
    "splits/cremad_outer_test_assignments.csv": "81cf1f28685582b92a0ad9c58e7a743dbc992416fa9d1576a11bd388a0595f93",
}

SUMMARY_SCHEMAS: dict[str, list[str]] = {
    "cell_completeness.csv": [
        "corpus", "model", "protocol", "seed", "status", "reason",
        "n_expected_folds", "n_success_folds", "n_missing_units", "n_failed_units",
        "n_invalid_units", "n_expected_utterances", "n_oof_utterances",
    ],
    "oof_seed_metrics.csv": [
        "corpus", "model", "protocol", "seed", "status", "reason", "n_utterances",
        "n_speakers", "uar", "accuracy", "macro_f1",
    ],
    "protocol_metric_summary.csv": [
        "corpus", "model", "protocol", "metric", "status", "reason",
        "n_required_seeds", "n_complete_seeds", "mean", "sample_sd",
    ],
    "speaker_seed_metrics.csv": [
        "corpus", "model", "protocol", "seed", "speaker_id", "status", "reason",
        "n_utterances", "uar", "accuracy", "macro_f1",
    ],
    "speaker_protocol_metrics.csv": [
        "corpus", "model", "protocol", "speaker_id", "status", "reason",
        "n_required_seeds", "n_complete_seeds", "n_utterances_per_seed",
        "uar_seed_mean", "accuracy_seed_mean", "macro_f1_seed_mean",
    ],
    "paired_speaker_differences.csv": [
        "corpus", "model", "contrast", "protocol_a", "protocol_b", "speaker_id",
        "status", "reason", "n_required_seeds_per_protocol", "uar_difference",
        "accuracy_difference", "macro_f1_difference",
    ],
    "protocol_premium.csv": [
        "corpus", "model", "contrast", "protocol_a", "protocol_b", "metric",
        "status", "reason", "point_mean_difference", "speaker_difference_sample_sd",
        "ci95_low", "ci95_high", "bootstrap_seed", "bootstrap_replicates",
        "n_speakers", "n_speakers_expected", "n_required_seed_cells",
        "n_missing_seed_cells", "n_missing_units", "n_failed_units", "n_invalid_units",
        "wilcoxon_statistic", "wilcoxon_p_raw", "wilcoxon_error",
        "sign_test_p_sensitivity", "n_zero_differences", "holm_family_size",
        "holm_adjusted_p", "holm_reject_0_05",
    ],
    "outer_fold_speaker_overlap.csv": [
        "corpus", "protocol", "outer_fold", "n_train_speakers", "n_test_speakers",
        "n_overlap", "overlap_fraction", "speaker_disjoint_required", "assertion_pass",
    ],
}

TABLE_KEYS = {
    "cell_completeness.csv": ("corpus", "model", "protocol", "seed"),
    "oof_seed_metrics.csv": ("corpus", "model", "protocol", "seed"),
    "protocol_metric_summary.csv": ("corpus", "model", "protocol", "metric"),
    "speaker_seed_metrics.csv": ("corpus", "model", "protocol", "seed", "speaker_id"),
    "speaker_protocol_metrics.csv": ("corpus", "model", "protocol", "speaker_id"),
    "paired_speaker_differences.csv": ("corpus", "model", "contrast", "speaker_id"),
    "protocol_premium.csv": ("corpus", "model", "contrast", "metric"),
    "outer_fold_speaker_overlap.csv": ("corpus", "protocol", "outer_fold"),
}

NUMERIC_FIELDS = {
    "oof_seed_metrics.csv": {"uar", "accuracy", "macro_f1"},
    "protocol_metric_summary.csv": {"mean", "sample_sd"},
    "speaker_seed_metrics.csv": {"uar", "accuracy", "macro_f1"},
    "speaker_protocol_metrics.csv": {"uar_seed_mean", "accuracy_seed_mean", "macro_f1_seed_mean"},
    "paired_speaker_differences.csv": {"uar_difference", "accuracy_difference", "macro_f1_difference"},
    "protocol_premium.csv": {
        "point_mean_difference", "speaker_difference_sample_sd", "ci95_low", "ci95_high",
        "wilcoxon_statistic", "wilcoxon_p_raw", "sign_test_p_sensitivity", "holm_adjusted_p",
    },
    "outer_fold_speaker_overlap.csv": {"overlap_fraction"},
}


class IndependentVerificationError(RuntimeError):
    """The raw prediction matrix or a published statistic failed verification."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float_text(value: float | None) -> str:
    return "" if value is None else format(float(value), ".17g")


def as_int(value: Any, context: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise IndependentVerificationError(f"invalid integer for {context}: {value!r}") from exc


def as_float(value: Any, context: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise IndependentVerificationError(f"invalid float for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise IndependentVerificationError(f"non-finite float for {context}: {value!r}")
    return result


def speaker_sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def read_csv(
    path: Path,
    *,
    required_fields: Iterable[str] | None = None,
    exact_fields: Sequence[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise IndependentVerificationError(f"missing CSV: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise IndependentVerificationError(f"CSV has no header: {path}")
            fields = list(reader.fieldnames)
            if exact_fields is not None and fields != list(exact_fields):
                raise IndependentVerificationError(
                    f"schema mismatch for {path}: observed={fields}, expected={list(exact_fields)}"
                )
            if required_fields is not None:
                missing = sorted(set(required_fields) - set(fields))
                if missing:
                    raise IndependentVerificationError(f"{path} missing columns: {missing}")
            return fields, list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise IndependentVerificationError(f"cannot read {path}: {exc}") from exc


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def check_frozen_hash(result_root: Path, relative: str, enforce: bool) -> str:
    path = result_root / relative
    if not path.is_file():
        raise IndependentVerificationError(f"missing frozen input: {path}")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise IndependentVerificationError(f"cannot hash frozen input {path}: {exc}") from exc
    if enforce:
        expected = FROZEN_INPUT_SHA256.get(relative)
        if expected is None or actual != expected:
            raise IndependentVerificationError(
                f"frozen input hash mismatch for {relative}: observed={actual}, expected={expected}"
            )
    return actual


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int, context: str) -> dict[str, float]:
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred) or len(y_true) == 0:
        raise IndependentVerificationError(f"invalid vectors for {context}")
    if np.any(y_true < 0) or np.any(y_true >= n_labels) or np.any(y_pred < 0) or np.any(y_pred >= n_labels):
        raise IndependentVerificationError(f"label out of range for {context}")
    confusion = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    support = confusion.sum(axis=1)
    if np.any(support == 0):
        raise IndependentVerificationError(
            f"fixed-class metric undefined for {context}; absent true classes={np.flatnonzero(support == 0).tolist()}"
        )
    predicted = confusion.sum(axis=0)
    recall = np.diag(confusion) / support
    f1_denom = support + predicted
    f1 = np.divide(
        2.0 * np.diag(confusion), f1_denom,
        out=np.zeros(n_labels, dtype=float), where=f1_denom != 0,
    )
    return {
        "uar": float(np.mean(recall)),
        "accuracy": float(np.trace(confusion) / np.sum(confusion)),
        "macro_f1": float(np.mean(f1)),
    }


def bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    if differences.ndim != 1 or len(differences) == 0 or not np.all(np.isfinite(differences)):
        raise IndependentVerificationError("invalid speaker difference vector for bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_speakers = len(differences)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    done = 0
    while done < BOOTSTRAP_REPLICATES:
        take = min(1000, BOOTSTRAP_REPLICATES - done)
        indices = rng.integers(0, n_speakers, size=(take, n_speakers))
        means[done:done + take] = differences[indices].mean(axis=1)
        done += take
    low, high = np.percentile(means, [2.5, 97.5], method="linear")
    return float(low), float(high)


def wilcoxon_values(differences: np.ndarray) -> dict[str, str]:
    zeros = int(np.count_nonzero(differences == 0.0))
    if zeros == len(differences):
        return {
            "wilcoxon_statistic": "0",
            "wilcoxon_p_raw": "1",
            "wilcoxon_error": "",
            "sign_test_p_sensitivity": "",
            "n_zero_differences": str(zeros),
        }
    try:
        result = wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(statistic) or not math.isfinite(p_value):
            raise ValueError("scipy.stats.wilcoxon returned a non-finite result")
        return {
            "wilcoxon_statistic": float_text(statistic),
            "wilcoxon_p_raw": float_text(p_value),
            "wilcoxon_error": "",
            "sign_test_p_sensitivity": "",
            "n_zero_differences": str(zeros),
        }
    except (ValueError, FloatingPointError) as exc:
        nonzero = differences[differences != 0.0]
        sign_p = 1.0 if len(nonzero) == 0 else float(
            binomtest(
                int(np.count_nonzero(nonzero > 0)), len(nonzero), 0.5,
                alternative="two-sided",
            ).pvalue
        )
        return {
            "wilcoxon_statistic": "",
            "wilcoxon_p_raw": "",
            "wilcoxon_error": f"{type(exc).__name__}: {exc}",
            "sign_test_p_sensitivity": float_text(sign_p),
            "n_zero_differences": str(zeros),
        }


def apply_holm(premium_rows: list[dict[str, str]]) -> None:
    uar_rows = [row for row in premium_rows if row["metric"] == "uar"]
    if len(uar_rows) != 24:
        raise IndependentVerificationError(f"fixed Holm family must contain 24 UAR rows, found {len(uar_rows)}")
    ordered: list[tuple[float, tuple[str, str, str], dict[str, str]]] = []
    for row in uar_rows:
        if row["wilcoxon_p_raw"] == "":
            p_value = 1.0
        else:
            p_value = as_float(row["wilcoxon_p_raw"], "Wilcoxon p")
        ordered.append((p_value, (row["corpus"], row["model"], row["contrast"]), row))
    ordered.sort(key=lambda item: (item[0], item[1]))
    running = 0.0
    for rank, (p_value, _, row) in enumerate(ordered):
        running = max(running, min(1.0, (24 - rank) * p_value))
        row["holm_family_size"] = "24"
        if row["wilcoxon_p_raw"] != "":
            row["holm_adjusted_p"] = float_text(running)
            row["holm_reject_0_05"] = str(running <= HOLM_ALPHA).lower()


def load_manifests(
    result_root: Path,
    corpora: Mapping[str, Mapping[str, Any]],
    enforce_hashes: bool,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[int, dict[str, str]]], dict[str, list[str]], dict[str, str]]:
    rows_by_corpus: dict[str, list[dict[str, str]]] = {}
    maps: dict[str, dict[int, dict[str, str]]] = {}
    speakers: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    required = ("sample_index", "relative_path", "label_name", "label_index", "speaker_id", "parse_status", "read_status")
    for corpus, cfg in corpora.items():
        relative = f"manifests/{corpus}_manifest.csv"
        path = result_root / relative
        hashes[relative] = check_frozen_hash(result_root, relative, enforce_hashes)
        _, rows = read_csv(path, required_fields=required)
        if len(rows) != int(cfg["expected_n"]):
            raise IndependentVerificationError(f"{corpus} manifest n={len(rows)} expected={cfg['expected_n']}")
        index_map: dict[int, dict[str, str]] = {}
        for expected_index, row in enumerate(rows):
            index = as_int(row["sample_index"], f"{corpus} manifest index")
            if index != expected_index or index in index_map:
                raise IndependentVerificationError(f"{corpus} manifest is not contiguous canonical order")
            if row["parse_status"] != "ok" or row["read_status"] != "ok":
                raise IndependentVerificationError(f"{corpus} manifest row {index} is not readable/parsed")
            label_index = as_int(row["label_index"], f"{corpus} manifest label")
            labels = tuple(cfg["labels"])
            if label_index not in range(len(labels)) or row["label_name"] != labels[label_index]:
                raise IndependentVerificationError(f"{corpus} manifest label mismatch at {index}")
            if not row["speaker_id"]:
                raise IndependentVerificationError(f"{corpus} empty speaker at {index}")
            index_map[index] = row
        corpus_speakers = sorted({row["speaker_id"] for row in rows}, key=speaker_sort_key)
        if len(corpus_speakers) != int(cfg["expected_speakers"]):
            raise IndependentVerificationError(
                f"{corpus} speakers={len(corpus_speakers)} expected={cfg['expected_speakers']}"
            )
        all_labels = set(range(len(tuple(cfg["labels"]))))
        for speaker in corpus_speakers:
            observed = {int(row["label_index"]) for row in rows if row["speaker_id"] == speaker}
            if observed != all_labels:
                raise IndependentVerificationError(
                    f"{corpus} speaker {speaker} lacks fixed classes: {sorted(all_labels - observed)}"
                )
        rows_by_corpus[corpus] = rows
        maps[corpus] = index_map
        speakers[corpus] = corpus_speakers
    return rows_by_corpus, maps, speakers, hashes


def load_assignments(
    result_root: Path,
    corpora: Mapping[str, Mapping[str, Any]],
    manifest_maps: Mapping[str, Mapping[int, Mapping[str, str]]],
    speakers: Mapping[str, Sequence[str]],
    enforce_hashes: bool,
) -> tuple[dict[tuple[str, str, int], dict[int, dict[str, str]]], list[dict[str, str]], dict[str, str]]:
    assignments: dict[tuple[str, str, int], dict[int, dict[str, str]]] = {}
    overlap_rows: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    required = ("corpus", "protocol", "outer_fold", "sample_index", "relative_path", "speaker_id", "label_index", "outer_fold_sha256")
    for corpus, cfg in corpora.items():
        relative = f"splits/{corpus}_outer_test_assignments.csv"
        path = result_root / relative
        hashes[relative] = check_frozen_hash(result_root, relative, enforce_hashes)
        _, rows = read_csv(path, required_fields=required)
        seen_by_protocol: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            if row["corpus"] != corpus or row["protocol"] not in PROTOCOLS:
                raise IndependentVerificationError(f"invalid assignment identity in {path}: {row}")
            protocol = row["protocol"]
            fold = as_int(row["outer_fold"], "assignment fold")
            index = as_int(row["sample_index"], "assignment sample")
            if fold not in range(int(cfg["folds"][protocol])) or index not in manifest_maps[corpus]:
                raise IndependentVerificationError(f"invalid assignment coordinate: {corpus}/{protocol}/{fold}/{index}")
            manifest = manifest_maps[corpus][index]
            for field in ("relative_path", "speaker_id", "label_index"):
                if row[field] != manifest[field]:
                    raise IndependentVerificationError(
                        f"assignment/manifest {field} mismatch: {corpus}/{protocol}/{fold}/{index}"
                    )
            key = (corpus, protocol, fold)
            fold_map = assignments.setdefault(key, {})
            if index in fold_map:
                raise IndependentVerificationError(f"duplicate assignment: {key}/{index}")
            fold_map[index] = row
            seen_by_protocol[protocol].append(index)
        expected_indices = list(range(int(cfg["expected_n"])))
        all_indices = set(expected_indices)
        for protocol in PROTOCOLS:
            if sorted(seen_by_protocol[protocol]) != expected_indices:
                raise IndependentVerificationError(f"{corpus}/{protocol} is not exactly-once OOF")
            for fold in range(int(cfg["folds"][protocol])):
                key = (corpus, protocol, fold)
                if key not in assignments or not assignments[key]:
                    raise IndependentVerificationError(f"missing/empty assignment fold: {key}")
                test_indices = set(assignments[key])
                train_indices = all_indices - test_indices
                test_speakers = {manifest_maps[corpus][i]["speaker_id"] for i in test_indices}
                train_speakers = {manifest_maps[corpus][i]["speaker_id"] for i in train_indices}
                overlap = train_speakers & test_speakers
                disjoint_required = protocol != "random"
                if disjoint_required and overlap:
                    raise IndependentVerificationError(f"strict protocol speaker overlap: {key}: {sorted(overlap)}")
                if protocol == "loso":
                    expected_speaker = str(speakers[corpus][fold])
                    if test_speakers != {expected_speaker}:
                        raise IndependentVerificationError(
                            f"LOSO canonical speaker mismatch: {key}: observed={sorted(test_speakers)}, expected={expected_speaker}"
                        )
                fraction = len(overlap) / max(len(test_speakers), 1)
                overlap_rows.append({
                    "corpus": corpus,
                    "protocol": protocol,
                    "outer_fold": str(fold),
                    "n_train_speakers": str(len(train_speakers)),
                    "n_test_speakers": str(len(test_speakers)),
                    "n_overlap": str(len(overlap)),
                    "overlap_fraction": float_text(fraction),
                    "speaker_disjoint_required": str(disjoint_required).lower(),
                    "assertion_pass": "true",
                })
    return assignments, overlap_rows, hashes


def load_plan(
    result_root: Path,
    corpora: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[tuple[str, str, int], Mapping[int, Mapping[str, str]]],
    enforce_hashes: bool,
) -> tuple[list[dict[str, str]], str]:
    relative = "run_plan.csv"
    digest = check_frozen_hash(result_root, relative, enforce_hashes)
    required = (
        "sequence", "unit_id", "corpus", "model", "protocol", "seed", "outer_fold",
        "n_outer_folds", "n_outer_test",
    )
    _, rows = read_csv(result_root / relative, required_fields=required)
    expected_keys = {
        (corpus, model, protocol, seed, fold)
        for corpus, cfg in corpora.items()
        for model in MODELS
        for protocol in PROTOCOLS
        for seed in SEEDS
        for fold in range(int(cfg["folds"][protocol]))
    }
    seen: set[tuple[str, str, str, int, int]] = set()
    unit_ids: set[str] = set()
    sequences: set[int] = set()
    for row in rows:
        key = (
            row["corpus"], row["model"], row["protocol"],
            as_int(row["seed"], "plan seed"), as_int(row["outer_fold"], "plan fold"),
        )
        if key not in expected_keys or key in seen:
            raise IndependentVerificationError(f"invalid/duplicate run-plan key: {key}")
        seen.add(key)
        if row["unit_id"] in unit_ids:
            raise IndependentVerificationError(f"duplicate run-plan unit_id: {row['unit_id']}")
        unit_ids.add(row["unit_id"])
        sequence = as_int(row["sequence"], "plan sequence")
        if sequence in sequences:
            raise IndependentVerificationError(f"duplicate run-plan sequence: {sequence}")
        sequences.add(sequence)
        corpus, _, protocol, _, fold = key
        if as_int(row["n_outer_folds"], "plan n_outer_folds") != int(corpora[corpus]["folds"][protocol]):
            raise IndependentVerificationError(f"plan n_outer_folds mismatch: {key}")
        if as_int(row["n_outer_test"], "plan n_outer_test") != len(assignments[(corpus, protocol, fold)]):
            raise IndependentVerificationError(f"plan n_outer_test mismatch: {key}")
    if seen != expected_keys:
        raise IndependentVerificationError(
            f"run-plan matrix mismatch; missing={sorted(expected_keys - seen)[:5]}, extra={sorted(seen - expected_keys)[:5]}"
        )
    if sequences != set(range(1, len(rows) + 1)):
        raise IndependentVerificationError("run-plan sequence is not contiguous from 1")
    return sorted(rows, key=lambda row: int(row["sequence"])), digest


def prediction_path(result_root: Path, row: Mapping[str, str]) -> Path:
    return (
        result_root / "units" / row["corpus"] / row["model"] / row["protocol"]
        / f"seed_{int(row['seed'])}" / f"fold_{int(row['outer_fold']):03d}" / "predictions.csv"
    )


def load_unit_predictions(
    result_root: Path,
    row: Mapping[str, str],
    assignment: Mapping[int, Mapping[str, str]],
    manifest_map: Mapping[int, Mapping[str, str]],
    labels: Sequence[str],
) -> tuple[list[dict[str, Any]], str, str]:
    path = prediction_path(result_root, row)
    required = [
        "unit_id", "corpus", "model", "protocol", "seed", "outer_fold", "sample_index",
        "relative_path", "speaker_id", "true_label_index", "true_label_name",
        "predicted_label_index", "predicted_label_name",
    ] + [f"logit_{label}" for label in labels]
    _, rows = read_csv(path, exact_fields=required)
    if len(rows) != len(assignment):
        raise IndependentVerificationError(
            f"prediction count mismatch for {row['unit_id']}: observed={len(rows)}, expected={len(assignment)}"
        )
    parsed: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, pred in enumerate(rows, start=2):
        context = f"{row['unit_id']} predictions row {position}"
        index = as_int(pred["sample_index"], f"{context} sample_index")
        if index in seen or index not in assignment or index not in manifest_map:
            raise IndependentVerificationError(f"duplicate/out-of-fold sample {index} in {context}")
        seen.add(index)
        manifest = manifest_map[index]
        expected_text = {
            "unit_id": row["unit_id"],
            "corpus": row["corpus"],
            "model": row["model"],
            "protocol": row["protocol"],
            "seed": row["seed"],
            "outer_fold": row["outer_fold"],
            "relative_path": manifest["relative_path"],
            "speaker_id": manifest["speaker_id"],
            "true_label_index": manifest["label_index"],
            "true_label_name": manifest["label_name"],
        }
        for field, expected in expected_text.items():
            if pred[field] != expected:
                raise IndependentVerificationError(f"{context} {field} mismatch")
        logits = np.asarray(
            [as_float(pred[f"logit_{label}"], f"{context} logit_{label}") for label in labels],
            dtype=float,
        )
        derived = int(np.argmax(logits))
        if as_int(pred["predicted_label_index"], f"{context} predicted label") != derived:
            raise IndependentVerificationError(f"{context} predicted index is not argmax(logits)")
        if pred["predicted_label_name"] != labels[derived]:
            raise IndependentVerificationError(f"{context} predicted name mismatch")
        parsed.append({
            "sample_index": index,
            "speaker_id": manifest["speaker_id"],
            "y_true": int(manifest["label_index"]),
            "y_pred": derived,
        })
    if seen != set(assignment):
        raise IndependentVerificationError(f"prediction sample set mismatch for {row['unit_id']}")
    relative = path.resolve().relative_to(result_root.resolve()).as_posix()
    return sorted(parsed, key=lambda item: item["sample_index"]), relative, sha256_file(path)


def recalculate_statistics(
    result_root: Path,
    *,
    corpora: Mapping[str, Mapping[str, Any]] = CORPORA,
    enforce_hashes: bool = True,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    result_root = result_root.resolve()
    manifests, manifest_maps, speakers, manifest_hashes = load_manifests(result_root, corpora, enforce_hashes)
    assignments, overlap_rows, assignment_hashes = load_assignments(
        result_root, corpora, manifest_maps, speakers, enforce_hashes,
    )
    plan, plan_hash = load_plan(result_root, corpora, assignments, enforce_hashes)
    plan_by_cell: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in plan:
        plan_by_cell[(row["corpus"], row["model"], row["protocol"], int(row["seed"]))].append(row)

    tables: dict[str, list[dict[str, str]]] = {name: [] for name in SUMMARY_SCHEMAS}
    tables["outer_fold_speaker_overlap.csv"] = overlap_rows
    cell_metrics: dict[tuple[str, str, str, int], dict[str, float]] = {}
    speaker_seed: dict[tuple[str, str, str, int, str], dict[str, float]] = {}
    prediction_hash_rows: list[tuple[str, str]] = []

    for corpus, cfg in corpora.items():
        labels = tuple(cfg["labels"])
        for model in MODELS:
            for protocol in PROTOCOLS:
                for seed in SEEDS:
                    key = (corpus, model, protocol, seed)
                    planned = sorted(plan_by_cell[key], key=lambda item: int(item["outer_fold"]))
                    if len(planned) != int(cfg["folds"][protocol]):
                        raise IndependentVerificationError(f"full-cell fold count mismatch: {key}")
                    combined: list[dict[str, Any]] = []
                    for plan_row in planned:
                        fold = int(plan_row["outer_fold"])
                        parsed, relative, digest = load_unit_predictions(
                            result_root,
                            plan_row,
                            assignments[(corpus, protocol, fold)],
                            manifest_maps[corpus],
                            labels,
                        )
                        combined.extend(parsed)
                        prediction_hash_rows.append((relative, digest))
                    observed_indices = [item["sample_index"] for item in combined]
                    expected_indices = list(range(int(cfg["expected_n"])))
                    if sorted(observed_indices) != expected_indices or len(set(observed_indices)) != len(observed_indices):
                        raise IndependentVerificationError(f"cell is not exactly-once OOF: {key}")
                    ordered = sorted(combined, key=lambda item: item["sample_index"])
                    y_true = np.asarray([item["y_true"] for item in ordered], dtype=np.int64)
                    y_pred = np.asarray([item["y_pred"] for item in ordered], dtype=np.int64)
                    metrics = metric_values(y_true, y_pred, len(labels), f"OOF {key}")
                    cell_metrics[key] = metrics
                    tables["cell_completeness.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "seed": str(seed),
                        "status": "complete", "reason": "",
                        "n_expected_folds": str(len(planned)), "n_success_folds": str(len(planned)),
                        "n_missing_units": "0", "n_failed_units": "0", "n_invalid_units": "0",
                        "n_expected_utterances": str(cfg["expected_n"]), "n_oof_utterances": str(len(ordered)),
                    })
                    tables["oof_seed_metrics.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "seed": str(seed),
                        "status": "complete", "reason": "", "n_utterances": str(len(ordered)),
                        "n_speakers": str(len({item["speaker_id"] for item in ordered})),
                        **{metric: float_text(metrics[metric]) for metric in METRICS},
                    })
                    for speaker in speakers[corpus]:
                        selected = [item for item in ordered if item["speaker_id"] == speaker]
                        values = metric_values(
                            np.asarray([item["y_true"] for item in selected], dtype=np.int64),
                            np.asarray([item["y_pred"] for item in selected], dtype=np.int64),
                            len(labels),
                            f"speaker {key}/{speaker}",
                        )
                        speaker_seed[(corpus, model, protocol, seed, speaker)] = values
                        tables["speaker_seed_metrics.csv"].append({
                            "corpus": corpus, "model": model, "protocol": protocol, "seed": str(seed),
                            "speaker_id": speaker, "status": "complete", "reason": "",
                            "n_utterances": str(len(selected)),
                            **{metric: float_text(values[metric]) for metric in METRICS},
                        })

    protocol_speaker: dict[tuple[str, str, str, str], dict[str, float]] = {}
    speaker_counts = {
        (corpus, speaker): sum(row["speaker_id"] == speaker for row in manifests[corpus])
        for corpus in corpora
        for speaker in speakers[corpus]
    }
    for corpus in corpora:
        for model in MODELS:
            for protocol in PROTOCOLS:
                for metric in METRICS:
                    values = np.asarray(
                        [cell_metrics[(corpus, model, protocol, seed)][metric] for seed in SEEDS],
                        dtype=float,
                    )
                    tables["protocol_metric_summary.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "metric": metric,
                        "status": "complete", "reason": "", "n_required_seeds": "3", "n_complete_seeds": "3",
                        "mean": float_text(float(np.mean(values))),
                        "sample_sd": float_text(float(np.std(values, ddof=1))),
                    })
                for speaker in speakers[corpus]:
                    means = {
                        metric: float(np.mean([
                            speaker_seed[(corpus, model, protocol, seed, speaker)][metric]
                            for seed in SEEDS
                        ]))
                        for metric in METRICS
                    }
                    protocol_speaker[(corpus, model, protocol, speaker)] = means
                    tables["speaker_protocol_metrics.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "speaker_id": speaker,
                        "status": "complete", "reason": "", "n_required_seeds": "3", "n_complete_seeds": "3",
                        "n_utterances_per_seed": str(speaker_counts[(corpus, speaker)]),
                        "uar_seed_mean": float_text(means["uar"]),
                        "accuracy_seed_mean": float_text(means["accuracy"]),
                        "macro_f1_seed_mean": float_text(means["macro_f1"]),
                    })

    premium_rows: list[dict[str, str]] = []
    for corpus, cfg in corpora.items():
        for model in MODELS:
            for contrast, (protocol_a, protocol_b) in CONTRASTS.items():
                speaker_differences: dict[str, dict[str, float]] = {}
                for speaker in speakers[corpus]:
                    left = protocol_speaker[(corpus, model, protocol_a, speaker)]
                    right = protocol_speaker[(corpus, model, protocol_b, speaker)]
                    differences = {metric: left[metric] - right[metric] for metric in METRICS}
                    speaker_differences[speaker] = differences
                    tables["paired_speaker_differences.csv"].append({
                        "corpus": corpus, "model": model, "contrast": contrast,
                        "protocol_a": protocol_a, "protocol_b": protocol_b, "speaker_id": speaker,
                        "status": "complete", "reason": "", "n_required_seeds_per_protocol": "3",
                        "uar_difference": float_text(differences["uar"]),
                        "accuracy_difference": float_text(differences["accuracy"]),
                        "macro_f1_difference": float_text(differences["macro_f1"]),
                    })
                for metric in METRICS:
                    values = np.asarray(
                        [speaker_differences[speaker][metric] for speaker in speakers[corpus]],
                        dtype=float,
                    )
                    low, high = bootstrap_ci(values)
                    premium: dict[str, str] = {
                        "corpus": corpus, "model": model, "contrast": contrast,
                        "protocol_a": protocol_a, "protocol_b": protocol_b, "metric": metric,
                        "status": "complete", "reason": "",
                        "point_mean_difference": float_text(float(np.mean(values))),
                        "speaker_difference_sample_sd": float_text(float(np.std(values, ddof=1))),
                        "ci95_low": float_text(low), "ci95_high": float_text(high),
                        "bootstrap_seed": str(BOOTSTRAP_SEED),
                        "bootstrap_replicates": str(BOOTSTRAP_REPLICATES),
                        "n_speakers": str(len(values)),
                        "n_speakers_expected": str(cfg["expected_speakers"]),
                        "n_required_seed_cells": "6", "n_missing_seed_cells": "0",
                        "n_missing_units": "0", "n_failed_units": "0", "n_invalid_units": "0",
                        "wilcoxon_statistic": "", "wilcoxon_p_raw": "", "wilcoxon_error": "",
                        "sign_test_p_sensitivity": "", "n_zero_differences": "",
                        "holm_family_size": "", "holm_adjusted_p": "", "holm_reject_0_05": "",
                    }
                    if metric == "uar":
                        premium.update(wilcoxon_values(values))
                    premium_rows.append(premium)
    apply_holm(premium_rows)
    tables["protocol_premium.csv"] = premium_rows

    aggregate = hashlib.sha256()
    for relative, digest in sorted(prediction_hash_rows):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    provenance = {
        "run_plan_sha256": plan_hash,
        "manifest_sha256": manifest_hashes,
        "assignment_sha256": assignment_hashes,
        "prediction_files": len(prediction_hash_rows),
        "prediction_inventory_sha256": aggregate.hexdigest(),
    }
    return tables, provenance


def table_index(name: str, rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, ...], Mapping[str, str]]:
    keys = TABLE_KEYS[name]
    result: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key in result:
            raise IndependentVerificationError(f"duplicate summary key in {name}: {key}")
        result[key] = row
    return result


def compare_published_tables(
    expected_tables: Mapping[str, Sequence[Mapping[str, str]]],
    summary_dir: Path,
) -> tuple[dict[str, int], dict[str, float]]:
    counts: dict[str, int] = {}
    max_abs_error: dict[str, float] = {}
    for name, fields in SUMMARY_SCHEMAS.items():
        _, observed_rows = read_csv(summary_dir / name, exact_fields=fields)
        expected_rows = list(expected_tables[name])
        observed = table_index(name, observed_rows)
        expected = table_index(name, expected_rows)
        if set(observed) != set(expected):
            raise IndependentVerificationError(
                f"summary key mismatch in {name}; missing={sorted(set(expected) - set(observed))[:5]}, "
                f"extra={sorted(set(observed) - set(expected))[:5]}"
            )
        table_max = 0.0
        numeric = NUMERIC_FIELDS.get(name, set())
        for key in sorted(expected):
            expected_row = expected[key]
            observed_row = observed[key]
            for field in fields:
                expected_value = str(expected_row.get(field, ""))
                observed_value = str(observed_row.get(field, ""))
                if field in numeric:
                    if expected_value == "" or observed_value == "":
                        if expected_value != observed_value:
                            raise IndependentVerificationError(
                                f"blank/numeric mismatch in {name} {key} field={field}: "
                                f"observed={observed_value!r}, expected={expected_value!r}"
                            )
                        continue
                    left = as_float(observed_value, f"published {name}/{key}/{field}")
                    right = as_float(expected_value, f"independent {name}/{key}/{field}")
                    difference = abs(left - right)
                    table_max = max(table_max, difference)
                    if not math.isclose(left, right, rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE):
                        raise IndependentVerificationError(
                            f"numeric mismatch in {name} {key} field={field}: observed={left}, expected={right}"
                        )
                elif field == "wilcoxon_error" and expected_value:
                    if not observed_value:
                        raise IndependentVerificationError(f"missing Wilcoxon error in {name} {key}")
                elif observed_value != expected_value:
                    raise IndependentVerificationError(
                        f"value mismatch in {name} {key} field={field}: "
                        f"observed={observed_value!r}, expected={expected_value!r}"
                    )
        counts[name] = len(observed_rows)
        max_abs_error[name] = table_max
    return counts, max_abs_error


def verify_statistics(
    result_root: Path,
    summary_dir: Path,
    *,
    corpora: Mapping[str, Mapping[str, Any]] = CORPORA,
    enforce_hashes: bool = True,
) -> dict[str, Any]:
    expected_tables, provenance = recalculate_statistics(
        result_root, corpora=corpora, enforce_hashes=enforce_hashes,
    )
    counts, max_abs_error = compare_published_tables(expected_tables, summary_dir)
    return {
        "schema_version": "p1_independent_statistics_verification_v1",
        "verified_at": now_iso(),
        "status": "pass",
        "independence": {
            "imports_primary_summarizer": False,
            "imports_existing_verifier": False,
            "imports_training_or_ser_code": False,
            "raw_sources": ["run_plan.csv", "manifests/*.csv", "splits/*_outer_test_assignments.csv", "units/**/predictions.csv"],
        },
        "frozen_statistics": {
            "seeds": list(SEEDS),
            "speaker_paired_contrasts": {name: f"{left} - {right}" for name, (left, right) in CONTRASTS.items()},
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_percentiles": [2.5, 97.5],
            "bootstrap_percentile_method": "linear",
            "wilcoxon": {"zero_method": "wilcox", "correction": False, "alternative": "two-sided", "method": "auto"},
            "holm_family_size": 24,
            "holm_alpha": HOLM_ALPHA,
            "numeric_tolerance": NUMERIC_TOLERANCE,
        },
        "checks": {
            "full_72_cell_oof_exactly_once": True,
            "strict_protocol_speaker_disjointness": True,
            "utterance_metrics_recomputed": True,
            "three_seed_mean_and_sample_sd_recomputed": True,
            "speaker_paired_differences_recomputed": True,
            "cluster_bootstrap_10000_recomputed": True,
            "wilcoxon_and_sign_sensitivity_recomputed": True,
            "fixed_24_test_holm_recomputed": True,
            "published_statistical_summary_tables_match": True,
        },
        "provenance": provenance,
        "table_rows": counts,
        "max_absolute_numeric_error": max_abs_error,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }


def build_synthetic_fixture(root: Path) -> dict[str, dict[str, Any]]:
    corpora = {
        "toy_a": {"expected_n": 8, "expected_speakers": 2, "labels": ("left", "right"), "folds": {"random": 2, "groupkfold": 2, "loso": 2}},
        "toy_b": {"expected_n": 8, "expected_speakers": 2, "labels": ("left", "right"), "folds": {"random": 2, "groupkfold": 2, "loso": 2}},
    }
    assignment_lookup: dict[tuple[str, str, int], list[int]] = {}
    manifest_rows_by_corpus: dict[str, list[dict[str, str]]] = {}
    for corpus_index, (corpus, cfg) in enumerate(corpora.items()):
        manifest_rows = []
        for index in range(int(cfg["expected_n"])):
            speaker = str(index // 4 + 1)
            label_index = index % 2
            manifest_rows.append({
                "sample_index": str(index),
                "relative_path": f"{corpus}/speaker-{speaker}/sample-{index}.wav",
                "label_name": cfg["labels"][label_index],
                "label_index": str(label_index),
                "speaker_id": speaker,
                "parse_status": "ok",
                "read_status": "ok",
            })
        manifest_rows_by_corpus[corpus] = manifest_rows
        write_csv(
            root / "manifests" / f"{corpus}_manifest.csv",
            ["sample_index", "relative_path", "label_name", "label_index", "speaker_id", "parse_status", "read_status"],
            manifest_rows,
        )
        assignment_rows = []
        for protocol in PROTOCOLS:
            for index, manifest in enumerate(manifest_rows):
                if protocol == "random":
                    fold = index % 2
                else:
                    fold = int(manifest["speaker_id"]) - 1
                assignment_lookup.setdefault((corpus, protocol, fold), []).append(index)
                assignment_rows.append({
                    "corpus": corpus, "protocol": protocol, "outer_fold": str(fold),
                    "sample_index": str(index), "relative_path": manifest["relative_path"],
                    "speaker_id": manifest["speaker_id"], "label_index": manifest["label_index"],
                    "outer_fold_sha256": f"toy-{corpus_index}-{protocol}-{fold}",
                })
        write_csv(
            root / "splits" / f"{corpus}_outer_test_assignments.csv",
            ["corpus", "protocol", "outer_fold", "sample_index", "relative_path", "speaker_id", "label_index", "outer_fold_sha256"],
            assignment_rows,
        )

    plan_fields = [
        "sequence", "unit_id", "corpus", "model", "protocol", "seed", "outer_fold",
        "n_outer_folds", "n_fit", "n_validation", "n_outer_test", "manifest_sha256",
        "cache_sha256", "outer_fold_sha256", "inner_split_path", "inner_split_sha256",
    ]
    plan_rows = []
    sequence = 0
    for corpus_index, (corpus, cfg) in enumerate(corpora.items()):
        labels = tuple(cfg["labels"])
        manifests = manifest_rows_by_corpus[corpus]
        for model_index, model in enumerate(MODELS):
            for protocol_index, protocol in enumerate(PROTOCOLS):
                for seed in SEEDS:
                    for fold in range(int(cfg["folds"][protocol])):
                        sequence += 1
                        unit_id = f"{corpus}__{model}__{protocol}__seed-{seed}__fold-{fold:03d}"
                        test_indices = assignment_lookup[(corpus, protocol, fold)]
                        plan_row = {
                            "sequence": str(sequence), "unit_id": unit_id, "corpus": corpus,
                            "model": model, "protocol": protocol, "seed": str(seed), "outer_fold": str(fold),
                            "n_outer_folds": str(cfg["folds"][protocol]), "n_fit": "2", "n_validation": "2",
                            "n_outer_test": str(len(test_indices)), "manifest_sha256": "toy-manifest",
                            "cache_sha256": "toy-cache", "outer_fold_sha256": f"toy-{corpus_index}-{protocol}-{fold}",
                            "inner_split_path": f"toy/{unit_id}.npz", "inner_split_sha256": "toy-inner",
                        }
                        plan_rows.append(plan_row)
                        prediction_fields = [
                            "unit_id", "corpus", "model", "protocol", "seed", "outer_fold", "sample_index",
                            "relative_path", "speaker_id", "true_label_index", "true_label_name",
                            "predicted_label_index", "predicted_label_name",
                        ] + [f"logit_{label}" for label in labels]
                        prediction_rows = []
                        wrong_cutoff = {"random": 1, "groupkfold": 2, "loso": 3}[protocol]
                        for sample_index in test_indices:
                            manifest = manifests[sample_index]
                            true_index = int(manifest["label_index"])
                            difficulty = (sample_index + model_index + seed + corpus_index) % 8
                            pred_index = 1 - true_index if difficulty < wrong_cutoff else true_index
                            item = {
                                "unit_id": unit_id, "corpus": corpus, "model": model,
                                "protocol": protocol, "seed": str(seed), "outer_fold": str(fold),
                                "sample_index": str(sample_index), "relative_path": manifest["relative_path"],
                                "speaker_id": manifest["speaker_id"], "true_label_index": str(true_index),
                                "true_label_name": labels[true_index], "predicted_label_index": str(pred_index),
                                "predicted_label_name": labels[pred_index],
                            }
                            for label_index, label in enumerate(labels):
                                item[f"logit_{label}"] = "2" if label_index == pred_index else "-1"
                            prediction_rows.append(item)
                        write_csv(prediction_path(root, plan_row), prediction_fields, prediction_rows)
    write_csv(root / "run_plan.csv", plan_fields, plan_rows)
    return corpora


def run_self_test() -> dict[str, Any]:
    known = metric_values(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0, 1, 1, 1], dtype=np.int64),
        2,
        "analytic self-test",
    )
    analytic_expected = {"uar": 0.75, "accuracy": 0.75, "macro_f1": 11.0 / 15.0}
    for metric, expected in analytic_expected.items():
        if not math.isclose(known[metric], expected, rel_tol=0.0, abs_tol=1e-15):
            raise IndependentVerificationError(
                f"analytic metric self-test failed for {metric}: observed={known[metric]}, expected={expected}"
            )
    bootstrap_known = bootstrap_ci(np.linspace(-0.3, 0.6, 24, dtype=float))
    bootstrap_expected = (0.042391304347826064, 0.25923913043478258)
    if not all(
        math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)
        for observed, expected in zip(bootstrap_known, bootstrap_expected)
    ):
        raise IndependentVerificationError(
            f"bootstrap RNG/percentile oracle failed: observed={bootstrap_known}, expected={bootstrap_expected}"
        )
    wilcoxon_known = wilcoxon_values(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float))
    if wilcoxon_known["wilcoxon_statistic"] != "0" or not math.isclose(
        float(wilcoxon_known["wilcoxon_p_raw"]), 0.125, rel_tol=0.0, abs_tol=1e-15
    ):
        raise IndependentVerificationError(f"Wilcoxon oracle failed: {wilcoxon_known}")
    wilcoxon_all_zero = wilcoxon_values(np.zeros(4, dtype=float))
    if wilcoxon_all_zero["wilcoxon_p_raw"] != "1" or wilcoxon_all_zero["n_zero_differences"] != "4":
        raise IndependentVerificationError(f"all-zero Wilcoxon oracle failed: {wilcoxon_all_zero}")
    original_wilcoxon = globals()["wilcoxon"]
    try:
        def forced_wilcoxon_failure(*_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("synthetic Wilcoxon domain failure")

        globals()["wilcoxon"] = forced_wilcoxon_failure
        sign_fallback = wilcoxon_values(np.asarray([1.0, 2.0, 3.0, -4.0], dtype=float))
    finally:
        globals()["wilcoxon"] = original_wilcoxon
    if (
        sign_fallback["wilcoxon_p_raw"] != ""
        or not sign_fallback["wilcoxon_error"]
        or not math.isclose(float(sign_fallback["sign_test_p_sensitivity"]), 0.625, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise IndependentVerificationError(f"Wilcoxon sign-test fallback oracle failed: {sign_fallback}")
    holm_rows = []
    for corpus in ("h0", "h1"):
        for model in MODELS:
            for contrast in CONTRASTS:
                holm_rows.append({
                    "corpus": corpus, "model": model, "contrast": contrast, "metric": "uar",
                    "wilcoxon_p_raw": "1", "holm_family_size": "",
                    "holm_adjusted_p": "", "holm_reject_0_05": "",
                })
    holm_rows[0]["wilcoxon_p_raw"] = "0.001"
    holm_rows[1]["wilcoxon_p_raw"] = "0.01"
    holm_rows[2]["wilcoxon_p_raw"] = "0.02"
    holm_rows[-1]["wilcoxon_p_raw"] = ""
    apply_holm(holm_rows)
    holm_expected = (0.024, 0.23, 0.44)
    if any(
        not math.isclose(float(holm_rows[index]["holm_adjusted_p"]), expected, rel_tol=0.0, abs_tol=1e-15)
        for index, expected in enumerate(holm_expected)
    ):
        raise IndependentVerificationError(
            f"Holm oracle failed: observed={[holm_rows[i]['holm_adjusted_p'] for i in range(3)]}, expected={holm_expected}"
        )
    if (
        holm_rows[0]["holm_reject_0_05"] != "true"
        or holm_rows[1]["holm_reject_0_05"] != "false"
        or any(row["holm_family_size"] != "24" for row in holm_rows)
        or holm_rows[-1]["holm_adjusted_p"] != ""
        or holm_rows[-1]["holm_reject_0_05"] != ""
    ):
        raise IndependentVerificationError("Holm rejection/family/missing-placeholder oracle failed")
    with tempfile.TemporaryDirectory(prefix="p1-independent-statistics-") as temp_name:
        root = Path(temp_name) / "protocol_premium"
        corpora = build_synthetic_fixture(root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables, provenance = recalculate_statistics(root, corpora=corpora, enforce_hashes=False)
        if len(tables["cell_completeness.csv"]) != 72:
            raise IndependentVerificationError("self-test did not exercise all 72 cells")
        if len([row for row in tables["protocol_premium.csv"] if row["metric"] == "uar"]) != 24:
            raise IndependentVerificationError("self-test did not exercise fixed 24-test Holm family")
        random_overlap = [
            row for row in tables["outer_fold_speaker_overlap.csv"]
            if row["protocol"] == "random"
        ]
        strict_overlap = [
            row for row in tables["outer_fold_speaker_overlap.csv"]
            if row["protocol"] != "random"
        ]
        if not random_overlap or not all(int(row["n_overlap"]) > 0 for row in random_overlap):
            raise IndependentVerificationError("self-test random arm lacks intended speaker overlap")
        if not all(row["n_overlap"] == "0" for row in strict_overlap):
            raise IndependentVerificationError("self-test strict arm has speaker overlap")
        rg_uar = [
            float(row["uar_difference"])
            for row in tables["paired_speaker_differences.csv"]
            if row["contrast"] == "RG"
        ]
        if not any(value > 0.0 for value in rg_uar) or any(value < 0.0 for value in rg_uar):
            raise IndependentVerificationError("self-test failed the frozen random-minus-group direction check")

        summary_dir = root / "summary"
        for name, fields in SUMMARY_SCHEMAS.items():
            write_csv(summary_dir / name, fields, tables[name])
        counts, errors = compare_published_tables(tables, summary_dir)

        _, rows = read_csv(summary_dir / "oof_seed_metrics.csv", exact_fields=SUMMARY_SCHEMAS["oof_seed_metrics.csv"])
        original = rows[0]["uar"]
        rows[0]["uar"] = float_text(float(original) + 0.125)
        write_csv(summary_dir / "oof_seed_metrics.csv", SUMMARY_SCHEMAS["oof_seed_metrics.csv"], rows)
        mismatch_detected = False
        try:
            compare_published_tables(tables, summary_dir)
        except IndependentVerificationError:
            mismatch_detected = True
        if not mismatch_detected:
            raise IndependentVerificationError("self-test failed to detect a deliberately altered published metric")
        return {
            "status": "pass",
            "synthetic_only": True,
            "no_formal_predictions_read": True,
            "cells_exercised": 72,
            "holm_tests_exercised": 24,
            "prediction_files_exercised": provenance["prediction_files"],
            "published_tables_compared": len(counts),
            "analytic_metric_formula_checked": True,
            "bootstrap_rng_percentile_oracle_checked": True,
            "wilcoxon_oracle_checked": True,
            "wilcoxon_all_zero_and_sign_fallback_checked": True,
            "holm_oracle_checked": True,
            "holm_missing_placeholder_checked": True,
            "random_minus_group_direction_checked": True,
            "max_abs_error_before_tamper": max(errors.values(), default=0.0),
            "deliberate_metric_mismatch_detected": mismatch_detected,
            "temporary_directory_removed_on_exit": True,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true", help="run a temporary synthetic 72-cell test; never reads formal predictions")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        try:
            report = run_self_test()
        except IndependentVerificationError as exc:
            print(json.dumps({"status": "self_test_failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0

    result_root = args.result_root.resolve()
    summary_dir = (args.summary_dir or result_root / "summary").resolve()
    report_path = (args.report or summary_dir / "independent_statistics_verification.json").resolve()
    try:
        report = verify_statistics(result_root, summary_dir, corpora=CORPORA, enforce_hashes=True)
        atomic_write_json(report_path, report)
    except IndependentVerificationError as exc:
        print(json.dumps({"status": "verification_failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "pass", "report": str(report_path)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

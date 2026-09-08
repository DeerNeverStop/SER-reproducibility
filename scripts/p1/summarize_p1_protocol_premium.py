"""Rebuild project-07 P1 protocol-premium results from utterance CSVs.

This is the preregistered *primary* analysis implementation.  It deliberately
does not trust metrics printed by the training runner.  A
``corpus x model x protocol x seed`` cell is usable only when every planned
outer fold has a successful, internally consistent prediction artifact and the
folds form exactly-once OOF coverage.  Otherwise the entire cell is missing;
that missingness propagates to the three-seed summaries and paired contrasts.

The script is CPU-only.  It imports neither torch nor the SER model code.
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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "protocol_premium"
PREREG_PATH = PROJECT_ROOT / "P1_PREREGISTRATION.md"
PREREG_SHA256 = "91db3edc544f76fc412f4dfc5f35adef23934635ca3ede36c8089c38bd14b782"
PREREG_FROZEN_PREFIX_BYTES = 12_426

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
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
GPU_SECONDS_LIMIT = 200 * 3600
WALL_SECONDS_LIMIT = 14 * 86400
OUTPUT_BYTES_LIMIT = 20 * 1024**3

EXPECTED_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
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

TABLE_FIELDS: dict[str, list[str]] = {
    "unit_completeness.csv": [
        "sequence", "unit_id", "corpus", "model", "protocol", "seed", "outer_fold",
        "status", "reason", "n_expected_predictions", "n_observed_predictions",
    ],
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
    "budget_audit.csv": [
        "status", "formal_started_at", "last_recorded_activity_at", "ledger_records",
        "unresolved_ledger_records", "gpu_seconds", "gpu_seconds_limit", "gpu_within_limit",
        "wall_seconds", "wall_seconds_limit", "wall_within_limit", "output_bytes",
        "output_bytes_limit", "output_within_limit", "budget_stop_present", "budget_stop_reasons",
    ],
    "source_files.csv": ["scope", "relative_path", "size_bytes", "sha256"],
}


class AnalysisError(RuntimeError):
    """Input or integrity failure that prevents a faithful analysis."""


@dataclass
class AnalysisBundle:
    tables: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]
    source_files: list[dict[str, str]]


class SourceRegistry:
    def __init__(self, result_root: Path, project_root: Path) -> None:
        self.result_root = result_root.resolve()
        self.project_root = project_root.resolve()
        self._items: dict[tuple[str, str], dict[str, str]] = {}

    def add(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise AnalysisError(f"required source file is missing: {path}")
        try:
            relative = resolved.relative_to(self.result_root).as_posix()
            scope = "results"
        except ValueError:
            try:
                relative = resolved.relative_to(self.project_root).as_posix()
                scope = "project"
            except ValueError:
                relative = str(resolved)
                scope = "absolute"
        item = {
            "scope": scope,
            "relative_path": relative,
            "size_bytes": str(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
        }
        self._items[(scope, relative)] = item

    def rows(self) -> list[dict[str, str]]:
        return [self._items[key] for key in sorted(self._items)]

    def assert_unchanged(self) -> None:
        for item in self.rows():
            if item["scope"] == "results":
                path = self.result_root / item["relative_path"]
            elif item["scope"] == "project":
                path = self.project_root / item["relative_path"]
            else:
                path = Path(item["relative_path"])
            if not path.is_file() or str(path.stat().st_size) != item["size_bytes"] or sha256_file(path) != item["sha256"]:
                raise AnalysisError(f"source changed during analysis: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_prefix(path: Path, n_bytes: int) -> str:
    with path.open("rb") as handle:
        payload = handle.read(n_bytes)
    if len(payload) != n_bytes:
        raise AnalysisError(f"{path} has fewer than {n_bytes} frozen bytes")
    return hashlib.sha256(payload).hexdigest()


def stable_u32(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def speaker_sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def read_csv_file(path: Path, registry: SourceRegistry | None = None) -> tuple[list[str], list[dict[str, str]]]:
    if registry is not None:
        registry.add(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AnalysisError(f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except UnicodeDecodeError as exc:
        raise AnalysisError(f"CSV is not UTF-8: {path}: {exc}") from exc


def read_json_file(path: Path, registry: SourceRegistry | None = None) -> dict[str, Any]:
    if registry is not None:
        registry.add(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON root must be an object: {path}")
    return value


def read_json_value(path: Path, registry: SourceRegistry | None = None) -> Any:
    if registry is not None:
        registry.add(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid JSON {path}: {exc}") from exc


def require_fields(path: Path, fields: Sequence[str], required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise AnalysisError(f"{path} missing columns: {missing}")


def as_int(value: Any, context: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid integer for {context}: {value!r}") from exc


def as_finite_float(value: Any, context: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid float for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"non-finite float for {context}: {value!r}")
    return result


def float_text(value: float | None) -> str:
    return "" if value is None else format(float(value), ".17g")


def join_reasons(reasons: Iterable[str]) -> str:
    return "|".join(sorted({reason for reason in reasons if reason}))


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int, context: str) -> dict[str, float]:
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred) or len(y_true) == 0:
        raise AnalysisError(f"invalid prediction vectors for {context}")
    if np.any(y_true < 0) or np.any(y_true >= n_labels) or np.any(y_pred < 0) or np.any(y_pred >= n_labels):
        raise AnalysisError(f"label index out of range for {context}")
    confusion = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    support = confusion.sum(axis=1)
    if np.any(support == 0):
        missing = np.flatnonzero(support == 0).tolist()
        raise AnalysisError(f"UAR undefined because true classes are absent for {context}: {missing}")
    recall = np.diag(confusion) / support
    predicted = confusion.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2.0 * np.diag(confusion), denom, out=np.zeros(n_labels, dtype=float), where=denom != 0)
    return {
        "uar": float(np.mean(recall)),
        "accuracy": float(np.trace(confusion) / np.sum(confusion)),
        "macro_f1": float(np.mean(f1)),
    }


def validate_failure_records(value: Any, expected_attempts: Sequence[int], context: str) -> tuple[list[str], float]:
    reasons: list[str] = []
    if not isinstance(value, list):
        return [f"{context}_not_list"], 0.0
    attempts: list[int] = []
    elapsed_total = 0.0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            reasons.append(f"{context}_item_{index}_not_object")
            continue
        try:
            attempts.append(as_int(item.get("attempt"), f"{context} attempt"))
            elapsed = as_finite_float(item.get("elapsed_seconds"), f"{context} elapsed_seconds")
            if elapsed < 0:
                reasons.append(f"{context}_negative_elapsed")
            elapsed_total += max(elapsed, 0.0)
        except AnalysisError as exc:
            reasons.append(f"{context}_numeric_invalid:{exc}")
        if not str(item.get("exception_type", "")).strip():
            reasons.append(f"{context}_exception_type_missing")
        if not str(item.get("exception", "")).strip():
            reasons.append(f"{context}_exception_missing")
        if not str(item.get("traceback", "")).strip():
            reasons.append(f"{context}_traceback_missing")
    if attempts != list(expected_attempts):
        reasons.append(f"{context}_attempt_sequence_{attempts}_expected_{list(expected_attempts)}")
    return reasons, elapsed_total


def validate_history(
    history_rows: Sequence[Mapping[str, str]], run_record: Mapping[str, Any], context: str
) -> list[str]:
    reasons: list[str] = []
    epochs: list[int] = []
    validation_losses: list[float] = []
    for index, item in enumerate(history_rows):
        try:
            epoch = as_int(item["epoch"], f"{context} epoch")
            epochs.append(epoch)
            train_loss = as_finite_float(item["train_loss"], f"{context} train_loss")
            validation_loss = as_finite_float(item["validation_loss"], f"{context} validation_loss")
            validation_losses.append(validation_loss)
            lr = as_finite_float(item["learning_rate"], f"{context} learning_rate")
            if train_loss < 0 or validation_loss < 0 or lr < 0:
                reasons.append(f"{context}_negative_loss_or_lr_at_{index + 1}")
            for metric in ("validation_uar", "validation_accuracy", "validation_macro_f1"):
                value = as_finite_float(item[metric], f"{context} {metric}")
                if not 0.0 <= value <= 1.0:
                    reasons.append(f"{context}_{metric}_out_of_range_at_{index + 1}")
        except AnalysisError as exc:
            reasons.append(f"{context}_row_{index + 1}_invalid:{exc}")
    if epochs != list(range(1, len(history_rows) + 1)):
        reasons.append(f"{context}_epochs_not_contiguous")
    try:
        epochs_run = as_int(run_record.get("epochs_run"), f"{context} epochs_run")
        best_epoch = as_int(run_record.get("best_epoch"), f"{context} best_epoch")
    except AnalysisError as exc:
        reasons.append(f"{context}_checkpoint_fields_invalid:{exc}")
        return reasons
    if epochs_run != len(history_rows) or not 1 <= epochs_run <= MAX_EPOCHS:
        reasons.append(f"{context}_epochs_run_mismatch")
    if validation_losses:
        expected_best_epoch = int(np.argmin(np.asarray(validation_losses, dtype=float))) + 1
        if best_epoch != expected_best_epoch:
            reasons.append(f"{context}_best_epoch_not_first_strict_validation_loss_minimum")
        try:
            recorded_best = as_finite_float(run_record.get("best_validation_loss"), f"{context} best_validation_loss")
            if not math.isclose(recorded_best, validation_losses[expected_best_epoch - 1], rel_tol=1e-12, abs_tol=1e-12):
                reasons.append(f"{context}_best_validation_loss_mismatch")
        except AnalysisError as exc:
            reasons.append(f"{context}_best_validation_loss_invalid:{exc}")
        if epochs_run < MAX_EPOCHS and epochs_run - best_epoch != PATIENCE:
            reasons.append(f"{context}_early_stop_patience_mismatch")
    return reasons


def validate_prereg(project_root: Path, registry: SourceRegistry) -> None:
    path = project_root / "P1_PREREGISTRATION.md"
    registry.add(path)
    actual = sha256_prefix(path, PREREG_FROZEN_PREFIX_BYTES)
    if actual != PREREG_SHA256:
        raise AnalysisError(f"frozen preregistration prefix hash drift: {actual}")


def load_plan(result_root: Path, registry: SourceRegistry) -> list[dict[str, str]]:
    path = result_root / "run_plan.csv"
    fields, rows = read_csv_file(path, registry)
    require_fields(path, fields, [
        "sequence", "unit_id", "corpus", "model", "protocol", "seed", "outer_fold",
        "n_outer_folds", "n_fit", "n_validation", "n_outer_test", "manifest_sha256", "cache_sha256",
        "outer_fold_sha256", "inner_split_path", "inner_split_sha256",
    ])
    expected_keys = {
        (corpus, model, protocol, seed, fold)
        for corpus in CORPORA
        for model in MODELS
        for protocol in PROTOCOLS
        for seed in SEEDS
        for fold in range(CORPORA[corpus]["folds"][protocol])
    }
    seen_keys: set[tuple[str, str, str, int, int]] = set()
    unit_ids: set[str] = set()
    sequences: set[int] = set()
    for row in rows:
        key = (row["corpus"], row["model"], row["protocol"], as_int(row["seed"], "plan seed"), as_int(row["outer_fold"], "plan fold"))
        if key in seen_keys:
            raise AnalysisError(f"duplicate run-plan cell/fold key: {key}")
        seen_keys.add(key)
        if row["unit_id"] in unit_ids:
            raise AnalysisError(f"duplicate unit_id: {row['unit_id']}")
        unit_ids.add(row["unit_id"])
        sequence = as_int(row["sequence"], "plan sequence")
        if sequence in sequences:
            raise AnalysisError(f"duplicate plan sequence: {sequence}")
        sequences.add(sequence)
        if key in expected_keys and as_int(row["n_outer_folds"], "n_outer_folds") != CORPORA[key[0]]["folds"][key[2]]:
            raise AnalysisError(f"n_outer_folds drift for {key}")
    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)[:10]
        extra = sorted(seen_keys - expected_keys)[:10]
        raise AnalysisError(f"run-plan matrix mismatch; missing={missing}, extra={extra}")
    if sequences != set(range(1, len(rows) + 1)):
        raise AnalysisError("run-plan sequence is not contiguous from 1")
    split_hashes: dict[str, str] = {}
    resolved_root = result_root.resolve()
    for row in rows:
        relative = row["inner_split_path"]
        prior = split_hashes.setdefault(relative, row["inner_split_sha256"])
        if prior != row["inner_split_sha256"]:
            raise AnalysisError(f"conflicting planned hashes for inner split {relative}")
    for relative, expected_hash in sorted(split_hashes.items()):
        path = (result_root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise AnalysisError(f"inner split escapes result root: {relative}") from exc
        registry.add(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise AnalysisError(f"inner split hash drift: {relative}: {actual_hash} != {expected_hash}")
    return sorted(rows, key=lambda row: int(row["sequence"]))


def load_manifests(result_root: Path, registry: SourceRegistry) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[int, dict[str, str]]], dict[str, list[str]]]:
    manifests: dict[str, list[dict[str, str]]] = {}
    by_index: dict[str, dict[int, dict[str, str]]] = {}
    speakers: dict[str, list[str]] = {}
    required = ["sample_index", "relative_path", "label_name", "label_index", "speaker_id", "parse_status", "read_status"]
    for corpus, cfg in CORPORA.items():
        path = result_root / "manifests" / f"{corpus}_manifest.csv"
        fields, rows = read_csv_file(path, registry)
        require_fields(path, fields, required)
        if len(rows) != cfg["expected_n"]:
            raise AnalysisError(f"{corpus} manifest n={len(rows)} expected={cfg['expected_n']}")
        index_map: dict[int, dict[str, str]] = {}
        for expected_index, row in enumerate(rows):
            index = as_int(row["sample_index"], f"{corpus} manifest sample_index")
            if index != expected_index or index in index_map:
                raise AnalysisError(f"{corpus} manifest is not contiguous canonical order")
            if row["parse_status"] != "ok" or row["read_status"] != "ok":
                raise AnalysisError(f"{corpus} manifest includes unreadable/unparsed row {index}")
            label_index = as_int(row["label_index"], f"{corpus} label_index")
            if label_index not in range(len(cfg["labels"])) or row["label_name"] != cfg["labels"][label_index]:
                raise AnalysisError(f"{corpus} label mismatch at sample {index}")
            if not row["speaker_id"]:
                raise AnalysisError(f"{corpus} empty speaker at sample {index}")
            index_map[index] = row
        corpus_speakers = sorted({row["speaker_id"] for row in rows}, key=speaker_sort_key)
        if len(corpus_speakers) != cfg["expected_speakers"]:
            raise AnalysisError(f"{corpus} speakers={len(corpus_speakers)} expected={cfg['expected_speakers']}")
        all_labels = set(range(len(cfg["labels"])))
        for speaker in corpus_speakers:
            observed = {int(row["label_index"]) for row in rows if row["speaker_id"] == speaker}
            if observed != all_labels:
                raise AnalysisError(f"{corpus} speaker {speaker} lacks fixed emotion classes: {sorted(all_labels - observed)}")
        manifests[corpus], by_index[corpus], speakers[corpus] = rows, index_map, corpus_speakers
    return manifests, by_index, speakers


def load_assignments_and_overlap(
    result_root: Path,
    registry: SourceRegistry,
    manifest_maps: Mapping[str, Mapping[int, Mapping[str, str]]],
) -> tuple[dict[tuple[str, str, int], dict[int, dict[str, str]]], list[dict[str, Any]]]:
    assignments: dict[tuple[str, str, int], dict[int, dict[str, str]]] = {}
    overlap_rows: list[dict[str, Any]] = []
    for corpus, cfg in CORPORA.items():
        path = result_root / "splits" / f"{corpus}_outer_test_assignments.csv"
        fields, rows = read_csv_file(path, registry)
        require_fields(path, fields, ["corpus", "protocol", "outer_fold", "sample_index", "relative_path", "speaker_id", "label_index", "outer_fold_sha256"])
        seen_per_protocol: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            protocol = row["protocol"]
            fold = as_int(row["outer_fold"], "assignment outer_fold")
            index = as_int(row["sample_index"], "assignment sample_index")
            if row["corpus"] != corpus or protocol not in PROTOCOLS or fold not in range(cfg["folds"][protocol]) or index not in manifest_maps[corpus]:
                raise AnalysisError(f"invalid assignment row in {path}: {row}")
            manifest = manifest_maps[corpus][index]
            for field in ("relative_path", "speaker_id", "label_index"):
                if row[field] != manifest[field]:
                    raise AnalysisError(f"assignment/manifest {field} mismatch: {corpus}/{protocol}/fold={fold}/sample={index}")
            key = (corpus, protocol, fold)
            assignments.setdefault(key, {})
            if index in assignments[key]:
                raise AnalysisError(f"duplicate assignment: {key}/sample={index}")
            assignments[key][index] = row
            seen_per_protocol[protocol].append(index)
        expected_indices = list(range(cfg["expected_n"]))
        for protocol in PROTOCOLS:
            if sorted(seen_per_protocol[protocol]) != expected_indices:
                raise AnalysisError(f"{corpus}/{protocol} assignments are not exactly-once OOF")
            for fold in range(cfg["folds"][protocol]):
                if (corpus, protocol, fold) not in assignments or not assignments[(corpus, protocol, fold)]:
                    raise AnalysisError(f"missing/empty assignment fold: {corpus}/{protocol}/{fold}")

        summary_path = result_root / "splits" / f"{corpus}_outer_fold_summary.csv"
        summary_fields, summary_rows = read_csv_file(summary_path, registry)
        require_fields(summary_path, summary_fields, [
            "corpus", "protocol", "outer_fold", "n_train_speakers", "n_test_speakers",
            "n_overlap", "overlap_fraction", "outer_fold_sha256",
        ])
        source_summary: dict[tuple[str, int], dict[str, str]] = {}
        for row in summary_rows:
            protocol = row["protocol"]
            fold = as_int(row["outer_fold"], f"{corpus} fold-summary outer_fold")
            if row["corpus"] != corpus or protocol not in PROTOCOLS or fold not in range(cfg["folds"][protocol]):
                raise AnalysisError(f"invalid fold-summary row in {summary_path}: {row}")
            if (protocol, fold) in source_summary:
                raise AnalysisError(f"duplicate fold-summary row: {corpus}/{protocol}/{fold}")
            source_summary[(protocol, fold)] = row
        if len(source_summary) != sum(cfg["folds"].values()):
            raise AnalysisError(f"{corpus} outer fold summary has duplicate/missing rows")
        all_indices = set(range(cfg["expected_n"]))
        for protocol in PROTOCOLS:
            for fold in range(cfg["folds"][protocol]):
                key = (corpus, protocol, fold)
                test_indices = set(assignments[key])
                train_indices = all_indices - test_indices
                train_speakers = {manifest_maps[corpus][i]["speaker_id"] for i in train_indices}
                test_speakers = {manifest_maps[corpus][i]["speaker_id"] for i in test_indices}
                overlap = train_speakers & test_speakers
                fraction = len(overlap) / max(len(test_speakers), 1)
                source = source_summary.get((protocol, fold))
                if source is None:
                    raise AnalysisError(f"missing fold summary row: {key}")
                expected_values = {
                    "n_train_speakers": len(train_speakers),
                    "n_test_speakers": len(test_speakers),
                    "n_overlap": len(overlap),
                }
                for field, expected in expected_values.items():
                    if as_int(source[field], f"{key} {field}") != expected:
                        raise AnalysisError(f"fold-summary {field} mismatch for {key}")
                if not math.isclose(as_finite_float(source["overlap_fraction"], f"{key} overlap_fraction"), fraction, rel_tol=0.0, abs_tol=1e-15):
                    raise AnalysisError(f"fold-summary overlap_fraction mismatch for {key}")
                hashes = {row["outer_fold_sha256"] for row in assignments[key].values()}
                if hashes != {source["outer_fold_sha256"]}:
                    raise AnalysisError(f"fold hash mismatch in assignment/summary for {key}")
                required_disjoint = protocol != "random"
                assertion_pass = not required_disjoint or len(overlap) == 0
                if not assertion_pass:
                    raise AnalysisError(f"strict protocol has speaker overlap: {key}")
                overlap_rows.append({
                    "corpus": corpus,
                    "protocol": protocol,
                    "outer_fold": str(fold),
                    "n_train_speakers": str(len(train_speakers)),
                    "n_test_speakers": str(len(test_speakers)),
                    "n_overlap": str(len(overlap)),
                    "overlap_fraction": float_text(fraction),
                    "speaker_disjoint_required": str(required_disjoint).lower(),
                    "assertion_pass": str(assertion_pass).lower(),
                })
    return assignments, overlap_rows


def unit_path(result_root: Path, row: Mapping[str, str]) -> Path:
    return result_root / "units" / row["corpus"] / row["model"] / row["protocol"] / f"seed_{int(row['seed'])}" / f"fold_{int(row['outer_fold']):03d}"


def inspect_unit(
    result_root: Path,
    row: Mapping[str, str],
    registry: SourceRegistry,
    manifest_map: Mapping[int, Mapping[str, str]],
    assignment: Mapping[int, Mapping[str, str]],
    labels: Sequence[str],
    expected_tool_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    base = unit_path(result_root, row)
    status_row: dict[str, Any] = {
        "sequence": row["sequence"], "unit_id": row["unit_id"], "corpus": row["corpus"],
        "model": row["model"], "protocol": row["protocol"], "seed": row["seed"],
        "outer_fold": row["outer_fold"], "status": "missing", "reason": "run_json_missing",
        "n_expected_predictions": str(len(assignment)), "n_observed_predictions": "0",
    }
    run_path = base / "run.json"
    if not run_path.is_file():
        return status_row, None
    try:
        run_record = read_json_file(run_path, registry)
    except AnalysisError as exc:
        status_row.update(status="invalid", reason=f"run_json_invalid:{exc}")
        return status_row, None
    run_status = str(run_record.get("status", ""))
    if run_status == "failed_after_retry":
        failure_reasons: list[str] = []
        if run_record.get("unit_id") != row["unit_id"]:
            failure_reasons.append("failed_run_unit_id_mismatch")
        for field, expected in (("attempts", 2), ("failed_attempts", 2), ("outer_test_evaluation_passes", 0)):
            try:
                if as_int(run_record.get(field), f"failed run {field}") != expected:
                    failure_reasons.append(f"failed_run_{field}_mismatch")
            except AnalysisError:
                failure_reasons.append(f"failed_run_{field}_invalid")
        failure_path = base / "failures.json"
        if not failure_path.is_file():
            failure_reasons.append("failures_json_missing")
        else:
            try:
                failure_value = read_json_value(failure_path, registry)
                record_reasons, elapsed = validate_failure_records(failure_value, (1, 2), "failed_attempts")
                failure_reasons.extend(record_reasons)
                recorded_gpu = as_finite_float(run_record.get("gpu_seconds_including_failed_attempts"), "failed run gpu seconds")
                if recorded_gpu < 0 or not math.isclose(recorded_gpu, elapsed, rel_tol=1e-9, abs_tol=1e-6):
                    failure_reasons.append("failed_run_gpu_seconds_mismatch")
            except AnalysisError as exc:
                failure_reasons.append(f"failures_json_invalid:{exc}")
        status_row.update(
            status="invalid" if failure_reasons else "failed",
            reason=join_reasons(failure_reasons) if failure_reasons else "failed_after_retry",
        )
        return status_row, None
    if run_status != "success":
        status_row.update(status="invalid", reason=f"unexpected_run_status:{run_status or 'blank'}")
        return status_row, None

    reasons: list[str] = []
    if run_record.get("unit_id") != row["unit_id"]:
        reasons.append("run_unit_id_mismatch")
    try:
        if as_int(run_record.get("outer_test_evaluation_passes", -1), "outer_test_evaluation_passes") != 1:
            reasons.append("outer_test_evaluation_passes_not_one")
    except AnalysisError:
        reasons.append("outer_test_evaluation_passes_invalid")
    config_path = base / "config.json"
    history_path = base / "history.csv"
    predictions_path = base / "predictions.csv"
    if not config_path.is_file():
        reasons.append("config_missing")
    else:
        try:
            config = read_json_file(config_path, registry)
            exact = {
                "unit_id": row["unit_id"], "corpus": row["corpus"], "model": row["model"],
                "protocol": row["protocol"], "seed": int(row["seed"]), "outer_fold": int(row["outer_fold"]),
                "sequence": int(row["sequence"]), "n_fit": int(row["n_fit"]),
                "n_validation": int(row["n_validation"]), "n_outer_test": int(row["n_outer_test"]),
                "manifest_sha256": row["manifest_sha256"],
                "cache_sha256": row.get("cache_sha256"), "outer_fold_sha256": row["outer_fold_sha256"],
                "inner_split_path": row.get("inner_split_path"),
                "inner_split_sha256": row.get("inner_split_sha256"),
                "runner_sha256": expected_tool_hashes["runner_sha256"],
                "core_sha256": expected_tool_hashes["core_sha256"],
                "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "checkpoint_rule": "strict minimum validation loss",
                "outer_test_evaluation_policy": "exactly once after checkpoint restoration",
                "mixed_precision": False,
            }
            for field, expected in exact.items():
                if config.get(field) != expected:
                    reasons.append(f"config_{field}_mismatch")
            if config.get("model_config") != EXPECTED_MODEL_CONFIGS[row["model"]]:
                reasons.append("config_model_config_mismatch")
            expected_seed_key = f"P1-train|{row['corpus']}|{row['protocol']}|{row['outer_fold']}|{row['seed']}"
            if config.get("train_seed_key") != expected_seed_key or config.get("train_seed_u32") != stable_u32(expected_seed_key):
                reasons.append("config_train_seed_mismatch")
            if run_record.get("config_sha256") != sha256_file(config_path):
                reasons.append("config_sha256_mismatch")
        except AnalysisError as exc:
            reasons.append(f"config_invalid:{exc}")
    if not history_path.is_file():
        reasons.append("history_missing")
    else:
        try:
            history_fields, history_rows = read_csv_file(history_path, registry)
            require_fields(history_path, history_fields, [
                "epoch", "train_loss", "validation_loss", "validation_uar",
                "validation_accuracy", "validation_macro_f1", "learning_rate",
            ])
            if not history_rows:
                reasons.append("history_empty")
            else:
                reasons.extend(validate_history(history_rows, run_record, row["unit_id"]))
            if run_record.get("history_sha256") != sha256_file(history_path):
                reasons.append("history_sha256_mismatch")
        except AnalysisError as exc:
            reasons.append(f"history_invalid:{exc}")
    try:
        attempts = as_int(run_record.get("attempts"), "success attempts")
        failed_attempts = as_int(run_record.get("failed_attempts"), "success failed_attempts")
        if attempts not in (1, 2) or failed_attempts != attempts - 1:
            reasons.append("success_attempt_counts_invalid")
        active_seconds = as_finite_float(run_record.get("gpu_active_seconds"), "success gpu_active_seconds")
        train_seconds = as_finite_float(run_record.get("train_seconds"), "success train_seconds")
        total_gpu_seconds = as_finite_float(run_record.get("gpu_seconds_including_failed_attempts"), "success total gpu seconds")
        if min(active_seconds, train_seconds, total_gpu_seconds) < 0 or active_seconds + 1e-6 < train_seconds:
            reasons.append("success_timing_invalid")
        previous_path = base / "previous_attempt_failures.json"
        if failed_attempts:
            if not previous_path.is_file():
                reasons.append("previous_attempt_failures_missing")
                previous_elapsed = 0.0
            else:
                previous_value = read_json_value(previous_path, registry)
                previous_reasons, previous_elapsed = validate_failure_records(previous_value, (1,), "previous_attempt")
                reasons.extend(previous_reasons)
            if not math.isclose(total_gpu_seconds, active_seconds + previous_elapsed, rel_tol=1e-9, abs_tol=1e-6):
                reasons.append("success_total_gpu_seconds_mismatch")
        elif previous_path.exists():
            reasons.append("unexpected_previous_attempt_failures")
        elif not math.isclose(total_gpu_seconds, active_seconds, rel_tol=1e-9, abs_tol=1e-6):
            reasons.append("success_total_gpu_seconds_mismatch")
    except AnalysisError as exc:
        reasons.append(f"success_run_accounting_invalid:{exc}")
    if not predictions_path.is_file():
        reasons.append("predictions_missing")
        status_row.update(status="invalid", reason=join_reasons(reasons))
        return status_row, None
    actual_prediction_sha = sha256_file(predictions_path)
    if run_record.get("predictions_sha256") != actual_prediction_sha:
        reasons.append("predictions_sha256_mismatch")
    try:
        fields, prediction_rows = read_csv_file(predictions_path, registry)
    except AnalysisError as exc:
        reasons.append(f"predictions_invalid:{exc}")
        status_row.update(status="invalid", reason=join_reasons(reasons))
        return status_row, None
    required_fields = [
        "unit_id", "corpus", "model", "protocol", "seed", "outer_fold", "sample_index",
        "relative_path", "speaker_id", "true_label_index", "true_label_name",
        "predicted_label_index", "predicted_label_name",
    ] + [f"logit_{label}" for label in labels]
    try:
        require_fields(predictions_path, fields, required_fields)
    except AnalysisError as exc:
        reasons.append(f"prediction_schema_invalid:{exc}")
        status_row.update(status="invalid", reason=join_reasons(reasons), n_observed_predictions=str(len(prediction_rows)))
        return status_row, None
    try:
        recorded_n_predictions = as_int(run_record.get("n_predictions", -1), "run n_predictions")
    except AnalysisError:
        recorded_n_predictions = -1
        reasons.append("run_n_predictions_invalid")
    if len(prediction_rows) != len(assignment) or recorded_n_predictions != len(prediction_rows):
        reasons.append("prediction_count_mismatch")
    parsed: list[dict[str, Any]] = []
    seen: set[int] = set()
    for local, pred in enumerate(prediction_rows):
        context = f"{row['unit_id']} prediction row {local + 2}"
        try:
            index = as_int(pred["sample_index"], f"{context} sample_index")
            if index in seen:
                raise AnalysisError(f"duplicate sample_index {index}")
            seen.add(index)
            if index not in assignment or index not in manifest_map:
                raise AnalysisError(f"sample {index} is outside planned outer test")
            manifest = manifest_map[index]
            exact_text = {
                "unit_id": row["unit_id"], "corpus": row["corpus"], "model": row["model"],
                "protocol": row["protocol"], "seed": row["seed"], "outer_fold": row["outer_fold"],
                "relative_path": manifest["relative_path"], "speaker_id": manifest["speaker_id"],
                "true_label_index": manifest["label_index"], "true_label_name": manifest["label_name"],
            }
            for field, expected in exact_text.items():
                if pred[field] != expected:
                    raise AnalysisError(f"{field} mismatch")
            logits = np.asarray([as_finite_float(pred[f"logit_{label}"], f"{context} logit_{label}") for label in labels], dtype=float)
            derived_pred = int(np.argmax(logits))
            if as_int(pred["predicted_label_index"], f"{context} predicted_label_index") != derived_pred:
                raise AnalysisError("stored prediction is not argmax(logits)")
            if pred["predicted_label_name"] != labels[derived_pred]:
                raise AnalysisError("predicted_label_name mismatch")
            parsed.append({
                "sample_index": index,
                "speaker_id": manifest["speaker_id"],
                "y_true": int(manifest["label_index"]),
                "y_pred": derived_pred,
            })
        except AnalysisError as exc:
            reasons.append(f"row_invalid:{context}:{exc}")
    if seen != set(assignment):
        reasons.append("prediction_sample_set_mismatch")
    status_row["n_observed_predictions"] = str(len(prediction_rows))
    if reasons:
        status_row.update(status="invalid", reason=join_reasons(reasons))
        return status_row, None
    status_row.update(status="success", reason="")
    return status_row, sorted(parsed, key=lambda item: item["sample_index"])


def cell_reason_from_units(unit_rows: Sequence[Mapping[str, Any]]) -> str:
    counts = defaultdict(int)
    for row in unit_rows:
        counts[str(row["status"])] += 1
    parts = [f"{status}_units={counts[status]}" for status in ("missing", "failed", "invalid") if counts[status]]
    return ";".join(parts)


def bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    if differences.ndim != 1 or len(differences) == 0 or not np.all(np.isfinite(differences)):
        raise AnalysisError("invalid speaker differences for bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(differences)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    # Chunking fixes peak memory while retaining exactly 10,000 whole-speaker draws.
    done = 0
    while done < BOOTSTRAP_REPLICATES:
        take = min(1000, BOOTSTRAP_REPLICATES - done)
        indices = rng.integers(0, n, size=(take, n))
        means[done:done + take] = differences[indices].mean(axis=1)
        done += take
    low, high = np.percentile(means, [2.5, 97.5], method="linear")
    return float(low), float(high)


def wilcoxon_result(differences: np.ndarray) -> dict[str, str]:
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
        if not math.isfinite(float(result.statistic)) or not math.isfinite(float(result.pvalue)):
            raise ValueError("scipy.stats.wilcoxon returned a non-finite result")
        return {
            "wilcoxon_statistic": float_text(float(result.statistic)),
            "wilcoxon_p_raw": float_text(float(result.pvalue)),
            "wilcoxon_error": "",
            "sign_test_p_sensitivity": "",
            "n_zero_differences": str(zeros),
        }
    except (ValueError, FloatingPointError) as exc:
        nonzero = differences[differences != 0.0]
        sign_p = 1.0 if len(nonzero) == 0 else float(binomtest(int(np.count_nonzero(nonzero > 0)), len(nonzero), 0.5, alternative="two-sided").pvalue)
        return {
            "wilcoxon_statistic": "",
            "wilcoxon_p_raw": "",
            "wilcoxon_error": f"{type(exc).__name__}: {exc}",
            "sign_test_p_sensitivity": float_text(sign_p),
            "n_zero_differences": str(zeros),
        }


def apply_holm(rows: list[dict[str, Any]]) -> None:
    uar_rows = [row for row in rows if row["metric"] == "uar"]
    if len(uar_rows) != 24:
        raise AnalysisError(f"Holm family must have 24 UAR rows, found {len(uar_rows)}")
    indexed: list[tuple[float, tuple[str, str, str], dict[str, Any], bool]] = []
    for row in uar_rows:
        observed = row["status"] == "complete" and row["wilcoxon_p_raw"] != ""
        p = float(row["wilcoxon_p_raw"]) if observed else 1.0
        indexed.append((p, (row["corpus"], row["model"], row["contrast"]), row, observed))
    indexed.sort(key=lambda item: (item[0], item[1]))
    running = 0.0
    m = len(indexed)
    for rank, (p, _, row, observed) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        row["holm_family_size"] = str(m)
        if observed:
            row["holm_adjusted_p"] = float_text(running)
            row["holm_reject_0_05"] = str(running <= HOLM_ALPHA).lower()
        else:
            row["holm_adjusted_p"] = ""
            row["holm_reject_0_05"] = ""


def parse_iso(value: Any, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid ISO timestamp for {context}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise AnalysisError(f"timezone-naive timestamp for {context}: {value!r}")
    return parsed


def formal_artifact_size_bytes(result_root: Path) -> int:
    """Count runner-managed P1 artifacts, excluding derived summary outputs."""
    derived_summary_root = (result_root / "summary").resolve()
    total = 0
    for path in result_root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(derived_summary_root)
        except ValueError:
            total += path.stat().st_size
    return total


def audit_budget(
    result_root: Path,
    registry: SourceRegistry,
    unit_rows: Sequence[Mapping[str, Any]],
    expected_tool_hashes: Mapping[str, str],
) -> dict[str, Any]:
    finalized = [row for row in unit_rows if row["status"] in {"success", "failed", "invalid"}]
    formal_path = result_root / "formal_start.json"
    formal_started: datetime | None = None
    if formal_path.is_file():
        formal = read_json_file(formal_path, registry)
        formal_started = parse_iso(formal.get("started_at"), "formal start")
        for field, expected in expected_tool_hashes.items():
            if formal.get(field) != expected:
                raise AnalysisError(f"formal_start {field} mismatch")
    elif finalized:
        raise AnalysisError("finalized units exist without formal_start.json")

    plan_unit_ids = {str(row["unit_id"]) for row in unit_rows}
    ledger_paths = sorted((result_root / "attempt_ledger").glob("*.json")) if (result_root / "attempt_ledger").is_dir() else []
    gpu_seconds = 0.0
    unresolved = 0
    activity_times: list[datetime] = []
    status_by_unit: dict[str, list[str]] = defaultdict(list)
    unresolved_statuses = {"running", "compute_success", "success_uncommitted", "interrupted_after_outer_test_access"}
    for path in ledger_paths:
        record = read_json_file(path, registry)
        unit_id = str(record.get("unit_id", ""))
        if unit_id not in plan_unit_ids:
            raise AnalysisError(f"ledger references unknown unit: {path}")
        if record.get("runner_sha256") != expected_tool_hashes["runner_sha256"] or record.get("core_sha256") != expected_tool_hashes["core_sha256"]:
            raise AnalysisError(f"ledger tool hash mismatch: {path}")
        attempt = as_int(record.get("attempt"), f"ledger attempt {path}")
        if attempt not in (1, 2):
            raise AnalysisError(f"ledger attempt outside 1/2: {path}")
        charged = as_finite_float(record.get("charged_seconds", 0), f"ledger charged_seconds {path}")
        if charged < 0:
            raise AnalysisError(f"negative ledger charge: {path}")
        gpu_seconds += charged
        status = str(record.get("status", ""))
        status_by_unit[unit_id].append(status)
        if status in unresolved_statuses:
            unresolved += 1
        for field in ("committed_at", "ended_at", "last_heartbeat_at", "started_at"):
            if record.get(field):
                activity_times.append(parse_iso(record[field], f"ledger {field} {path}"))
                break
    successful_unit_ids = {str(row["unit_id"]) for row in unit_rows if row["status"] == "success"}
    failed_unit_ids = {str(row["unit_id"]) for row in unit_rows if row["status"] == "failed"}
    if successful_unit_ids and not ledger_paths:
        raise AnalysisError("successful units exist without attempt ledger")
    for unit_id in successful_unit_ids:
        if "success_committed" not in status_by_unit[unit_id]:
            raise AnalysisError(f"successful unit lacks success_committed ledger: {unit_id}")
    for unit_id in failed_unit_ids:
        if status_by_unit[unit_id].count("failed_model_attempt") != 2:
            raise AnalysisError(f"failed unit lacks exactly two failed attempt ledgers: {unit_id}")

    last_activity = max(activity_times) if activity_times else formal_started
    wall_seconds = 0.0
    if formal_started is not None and last_activity is not None:
        wall_seconds = (last_activity - formal_started).total_seconds()
        if wall_seconds < 0:
            raise AnalysisError("last P1 activity predates formal start")
    output_bytes = formal_artifact_size_bytes(result_root) if result_root.exists() else 0
    budget_stop_path = result_root / "budget_stop.json"
    budget_stop_reasons = ""
    if budget_stop_path.is_file():
        budget_stop = read_json_file(budget_stop_path, registry)
        reasons = budget_stop.get("reasons", [])
        budget_stop_reasons = "|".join(str(item) for item in reasons) if isinstance(reasons, list) else str(reasons)
    gpu_ok = gpu_seconds <= GPU_SECONDS_LIMIT
    wall_ok = wall_seconds <= WALL_SECONDS_LIMIT
    output_ok = output_bytes <= OUTPUT_BYTES_LIMIT
    status = "pass" if gpu_ok and wall_ok and output_ok and unresolved == 0 else "fail"
    return {
        "status": status,
        "formal_started_at": formal_started.isoformat() if formal_started else "",
        "last_recorded_activity_at": last_activity.isoformat() if last_activity else "",
        "ledger_records": str(len(ledger_paths)),
        "unresolved_ledger_records": str(unresolved),
        "gpu_seconds": float_text(gpu_seconds),
        "gpu_seconds_limit": str(GPU_SECONDS_LIMIT),
        "gpu_within_limit": str(gpu_ok).lower(),
        "wall_seconds": float_text(wall_seconds),
        "wall_seconds_limit": str(WALL_SECONDS_LIMIT),
        "wall_within_limit": str(wall_ok).lower(),
        "output_bytes": str(output_bytes),
        "output_bytes_limit": str(OUTPUT_BYTES_LIMIT),
        "output_within_limit": str(output_ok).lower(),
        "budget_stop_present": str(budget_stop_path.is_file()).lower(),
        "budget_stop_reasons": budget_stop_reasons,
    }
    for row in rows:
        if row["metric"] != "uar":
            row["holm_family_size"] = ""
            row["holm_adjusted_p"] = ""
            row["holm_reject_0_05"] = ""


def analyze(result_root: Path = DEFAULT_RESULT_ROOT, project_root: Path = PROJECT_ROOT) -> AnalysisBundle:
    result_root = result_root.resolve()
    project_root = project_root.resolve()
    registry = SourceRegistry(result_root, project_root)
    validate_prereg(project_root, registry)
    runner_path = project_root / "tools" / "run_p1_protocol_premium.py"
    core_path = project_root / "tools" / "p1_protocol_core.py"
    registry.add(runner_path)
    registry.add(core_path)
    expected_tool_hashes = {
        "runner_sha256": sha256_file(runner_path),
        "core_sha256": sha256_file(core_path),
    }
    plan = load_plan(result_root, registry)
    manifests, manifest_maps, speakers = load_manifests(result_root, registry)
    for corpus in CORPORA:
        manifest_path = result_root / "manifests" / f"{corpus}_manifest.csv"
        planned_hashes = {row["manifest_sha256"] for row in plan if row["corpus"] == corpus}
        actual_hash = sha256_file(manifest_path)
        if planned_hashes != {actual_hash}:
            raise AnalysisError(
                f"run-plan manifest hash mismatch for {corpus}: planned={sorted(planned_hashes)}, actual={actual_hash}"
            )
    assignments, overlap_rows = load_assignments_and_overlap(result_root, registry, manifest_maps)

    plan_by_cell: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in plan:
        plan_by_cell[(row["corpus"], row["model"], row["protocol"], int(row["seed"]))].append(row)

    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_FIELDS}
    tables["outer_fold_speaker_overlap.csv"] = overlap_rows
    cell_metrics: dict[tuple[str, str, str, int], dict[str, float]] = {}
    speaker_seed: dict[tuple[str, str, str, int, str], dict[str, float]] = {}
    cell_state: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    for corpus in CORPORA:
        labels = CORPORA[corpus]["labels"]
        for model in MODELS:
            for protocol in PROTOCOLS:
                for seed in SEEDS:
                    key = (corpus, model, protocol, seed)
                    planned = sorted(plan_by_cell[key], key=lambda row: int(row["outer_fold"]))
                    unit_rows: list[dict[str, Any]] = []
                    predictions: list[dict[str, Any]] = []
                    for row in planned:
                        fold = int(row["outer_fold"])
                        assignment = assignments[(corpus, protocol, fold)]
                        if row["outer_fold_sha256"] != next(iter(assignment.values()))["outer_fold_sha256"]:
                            raise AnalysisError(f"run-plan/assignment fold hash mismatch: {key}/fold={fold}")
                        if as_int(row["n_outer_test"], "plan n_outer_test") != len(assignment):
                            raise AnalysisError(f"run-plan n_outer_test mismatch: {key}/fold={fold}")
                        unit_status, unit_predictions = inspect_unit(
                            result_root, row, registry, manifest_maps[corpus], assignment, labels,
                            expected_tool_hashes,
                        )
                        unit_rows.append(unit_status)
                        tables["unit_completeness.csv"].append(unit_status)
                        if unit_predictions is not None:
                            predictions.extend(unit_predictions)
                    status_counts = defaultdict(int)
                    for item in unit_rows:
                        status_counts[item["status"]] += 1
                    complete = status_counts["success"] == len(planned)
                    reason = "" if complete else cell_reason_from_units(unit_rows)
                    if complete:
                        observed_indices = [item["sample_index"] for item in predictions]
                        if sorted(observed_indices) != list(range(CORPORA[corpus]["expected_n"])) or len(set(observed_indices)) != len(observed_indices):
                            complete = False
                            reason = "oof_not_exactly_once"
                    state = {
                        "status": "complete" if complete else "missing_full_cell",
                        "reason": reason,
                        "n_expected_folds": len(planned),
                        "n_success_folds": status_counts["success"],
                        "n_missing_units": status_counts["missing"],
                        "n_failed_units": status_counts["failed"],
                        "n_invalid_units": status_counts["invalid"],
                    }
                    cell_state[key] = state
                    tables["cell_completeness.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "seed": str(seed),
                        "status": state["status"], "reason": reason,
                        "n_expected_folds": str(len(planned)), "n_success_folds": str(status_counts["success"]),
                        "n_missing_units": str(status_counts["missing"]), "n_failed_units": str(status_counts["failed"]),
                        "n_invalid_units": str(status_counts["invalid"]),
                        "n_expected_utterances": str(CORPORA[corpus]["expected_n"]),
                        "n_oof_utterances": str(len(predictions)) if complete else "",
                    })
                    if complete:
                        ordered = sorted(predictions, key=lambda item: item["sample_index"])
                        y_true = np.asarray([item["y_true"] for item in ordered], dtype=np.int64)
                        y_pred = np.asarray([item["y_pred"] for item in ordered], dtype=np.int64)
                        metrics = metric_values(y_true, y_pred, len(labels), f"OOF {key}")
                        cell_metrics[key] = metrics
                        n_utterances = str(len(ordered))
                        n_speakers = str(len({item["speaker_id"] for item in ordered}))
                        for speaker in speakers[corpus]:
                            selected = [item for item in ordered if item["speaker_id"] == speaker]
                            speaker_metrics = metric_values(
                                np.asarray([item["y_true"] for item in selected], dtype=np.int64),
                                np.asarray([item["y_pred"] for item in selected], dtype=np.int64),
                                len(labels), f"speaker {key}/{speaker}",
                            )
                            speaker_seed[(corpus, model, protocol, seed, speaker)] = speaker_metrics
                            tables["speaker_seed_metrics.csv"].append({
                                "corpus": corpus, "model": model, "protocol": protocol, "seed": str(seed),
                                "speaker_id": speaker, "status": "complete", "reason": "",
                                "n_utterances": str(len(selected)),
                                **{metric: float_text(speaker_metrics[metric]) for metric in METRICS},
                            })
                    else:
                        metrics = {metric: None for metric in METRICS}
                        n_utterances = n_speakers = ""
                        for speaker in speakers[corpus]:
                            tables["speaker_seed_metrics.csv"].append({
                                "corpus": corpus, "model": model, "protocol": protocol, "seed": str(seed),
                                "speaker_id": speaker, "status": "missing_full_cell", "reason": reason,
                                "n_utterances": "", "uar": "", "accuracy": "", "macro_f1": "",
                            })
                    tables["oof_seed_metrics.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "seed": str(seed),
                        "status": state["status"], "reason": reason, "n_utterances": n_utterances,
                        "n_speakers": n_speakers,
                        **{metric: float_text(metrics[metric]) for metric in METRICS},
                    })

    protocol_speaker: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for corpus in CORPORA:
        for model in MODELS:
            for protocol in PROTOCOLS:
                complete_seeds = [seed for seed in SEEDS if (corpus, model, protocol, seed) in cell_metrics]
                missing_reason = "" if len(complete_seeds) == len(SEEDS) else join_reasons(
                    cell_state[(corpus, model, protocol, seed)]["reason"] for seed in SEEDS if seed not in complete_seeds
                )
                for metric in METRICS:
                    values = [cell_metrics[(corpus, model, protocol, seed)][metric] for seed in SEEDS if seed in complete_seeds]
                    full = len(values) == len(SEEDS)
                    tables["protocol_metric_summary.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "metric": metric,
                        "status": "complete" if full else "missing_full_cell", "reason": missing_reason,
                        "n_required_seeds": str(len(SEEDS)), "n_complete_seeds": str(len(values)),
                        "mean": float_text(float(np.mean(values))) if full else "",
                        "sample_sd": float_text(float(np.std(values, ddof=1))) if full else "",
                    })
                for speaker in speakers[corpus]:
                    source = [speaker_seed.get((corpus, model, protocol, seed, speaker)) for seed in SEEDS]
                    full = all(item is not None for item in source)
                    if full:
                        means = {metric: float(np.mean([item[metric] for item in source if item is not None])) for metric in METRICS}
                        protocol_speaker[(corpus, model, protocol, speaker)] = means
                        n_utterances = next(
                            row["n_utterances"] for row in tables["speaker_seed_metrics.csv"]
                            if row["corpus"] == corpus and row["model"] == model and row["protocol"] == protocol
                            and row["seed"] == "0" and row["speaker_id"] == speaker
                        )
                    else:
                        means = {metric: None for metric in METRICS}
                        n_utterances = ""
                    tables["speaker_protocol_metrics.csv"].append({
                        "corpus": corpus, "model": model, "protocol": protocol, "speaker_id": speaker,
                        "status": "complete" if full else "missing_full_cell", "reason": "" if full else missing_reason,
                        "n_required_seeds": str(len(SEEDS)), "n_complete_seeds": str(sum(item is not None for item in source)),
                        "n_utterances_per_seed": n_utterances,
                        "uar_seed_mean": float_text(means["uar"]),
                        "accuracy_seed_mean": float_text(means["accuracy"]),
                        "macro_f1_seed_mean": float_text(means["macro_f1"]),
                    })

    premium_rows: list[dict[str, Any]] = []
    for corpus in CORPORA:
        for model in MODELS:
            for contrast, (protocol_a, protocol_b) in CONTRASTS.items():
                missing_cells = [
                    (protocol, seed)
                    for protocol in (protocol_a, protocol_b)
                    for seed in SEEDS
                    if (corpus, model, protocol, seed) not in cell_metrics
                ]
                full = not missing_cells
                contrast_reason = "" if full else "missing_seed_cells=" + ",".join(f"{p}:s{s}" for p, s in missing_cells)
                speaker_differences: dict[str, dict[str, float]] = {}
                for speaker in speakers[corpus]:
                    left = protocol_speaker.get((corpus, model, protocol_a, speaker))
                    right = protocol_speaker.get((corpus, model, protocol_b, speaker))
                    if full and left is not None and right is not None:
                        differences = {metric: left[metric] - right[metric] for metric in METRICS}
                        speaker_differences[speaker] = differences
                        status = "complete"
                    else:
                        differences = {metric: None for metric in METRICS}
                        status = "missing_full_cell"
                    tables["paired_speaker_differences.csv"].append({
                        "corpus": corpus, "model": model, "contrast": contrast,
                        "protocol_a": protocol_a, "protocol_b": protocol_b, "speaker_id": speaker,
                        "status": status, "reason": "" if status == "complete" else contrast_reason,
                        "n_required_seeds_per_protocol": str(len(SEEDS)),
                        "uar_difference": float_text(differences["uar"]),
                        "accuracy_difference": float_text(differences["accuracy"]),
                        "macro_f1_difference": float_text(differences["macro_f1"]),
                    })
                relevant_states = [cell_state[(corpus, model, protocol, seed)] for protocol in (protocol_a, protocol_b) for seed in SEEDS]
                missing_units = sum(state["n_missing_units"] for state in relevant_states)
                failed_units = sum(state["n_failed_units"] for state in relevant_states)
                invalid_units = sum(state["n_invalid_units"] for state in relevant_states)
                for metric in METRICS:
                    row: dict[str, Any] = {
                        "corpus": corpus, "model": model, "contrast": contrast,
                        "protocol_a": protocol_a, "protocol_b": protocol_b, "metric": metric,
                        "status": "complete" if full else "missing_full_cell", "reason": contrast_reason,
                        "point_mean_difference": "", "speaker_difference_sample_sd": "",
                        "ci95_low": "", "ci95_high": "", "bootstrap_seed": str(BOOTSTRAP_SEED),
                        "bootstrap_replicates": str(BOOTSTRAP_REPLICATES),
                        "n_speakers": str(len(speaker_differences)) if full else "0",
                        "n_speakers_expected": str(CORPORA[corpus]["expected_speakers"]),
                        "n_required_seed_cells": str(2 * len(SEEDS)),
                        "n_missing_seed_cells": str(len(missing_cells)),
                        "n_missing_units": str(missing_units), "n_failed_units": str(failed_units),
                        "n_invalid_units": str(invalid_units),
                        "wilcoxon_statistic": "", "wilcoxon_p_raw": "", "wilcoxon_error": "",
                        "sign_test_p_sensitivity": "", "n_zero_differences": "",
                        "holm_family_size": "", "holm_adjusted_p": "", "holm_reject_0_05": "",
                    }
                    if full:
                        values = np.asarray([speaker_differences[speaker][metric] for speaker in speakers[corpus]], dtype=float)
                        if len(values) != CORPORA[corpus]["expected_speakers"]:
                            raise AnalysisError(f"paired speaker set mismatch for {corpus}/{model}/{contrast}/{metric}")
                        low, high = bootstrap_ci(values)
                        row.update({
                            "point_mean_difference": float_text(float(np.mean(values))),
                            "speaker_difference_sample_sd": float_text(float(np.std(values, ddof=1))),
                            "ci95_low": float_text(low), "ci95_high": float_text(high),
                        })
                        if metric == "uar":
                            row.update(wilcoxon_result(values))
                    premium_rows.append(row)
    apply_holm(premium_rows)
    tables["protocol_premium.csv"] = premium_rows

    budget_row = audit_budget(
        result_root, registry, tables["unit_completeness.csv"], expected_tool_hashes
    )
    tables["budget_audit.csv"] = [budget_row]

    registry.assert_unchanged()
    source_rows = registry.rows()
    tables["source_files.csv"] = source_rows
    complete_cells = sum(row["status"] == "complete" for row in tables["cell_completeness.csv"])
    metadata = {
        "schema_version": "p1_protocol_premium_summary_v1",
        "created_at": now_iso(),
        "status": (
            "complete" if complete_cells == 72 and budget_row["status"] == "pass"
            else "invalid_budget" if budget_row["status"] != "pass"
            else "incomplete"
        ),
        "complete_cells": complete_cells,
        "required_cells": 72,
        "planned_training_units": len(plan),
        "complete_training_units": sum(row["status"] == "success" for row in tables["unit_completeness.csv"]),
        "budget_audit": budget_row,
        "frozen_rules": {
            "seeds": list(SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_percentile_method": "numpy.percentile linear 2.5/97.5",
            "holm_family": "2 corpora x 4 models x 3 UAR contrasts",
            "holm_family_size": 24,
            "holm_alpha": HOLM_ALPHA,
            "missing_confirmatory_p_policy": "retain the fixed 24-test family; a missing test is a non-rejected p=1 placeholder for Holm ordering, while its published p and adjusted p remain blank",
            "missingness": "any missing/failed/invalid outer fold invalidates the full corpus-model-protocol-seed OOF cell; any missing seed cell invalidates its three-seed protocol summary and every contrast requiring that protocol",
            "contrast_directions": {name: f"{a} - {b}" for name, (a, b) in CONTRASTS.items()},
        },
        "preregistration_frozen_prefix_bytes": PREREG_FROZEN_PREFIX_BYTES,
        "preregistration_frozen_prefix_sha256": PREREG_SHA256,
    }
    return AnalysisBundle(tables=tables, metadata=metadata, source_files=source_rows)


def atomic_write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


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


def write_bundle(bundle: AnalysisBundle, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for name, fields in TABLE_FIELDS.items():
        path = output_dir / name
        atomic_write_csv(path, fields, bundle.tables[name])
        output_hashes[name] = sha256_file(path)
    script_hash = sha256_file(Path(__file__).resolve())
    manifest = dict(bundle.metadata)
    manifest["summarizer_sha256"] = script_hash
    manifest["outputs"] = output_hashes
    manifest["source_files_sha256"] = output_hashes["source_files.csv"]
    manifest_path = output_dir / "analysis_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true", help="return exit 3 after writing honest missingness tables unless all 72 seed cells are complete")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.result_root / "summary"
    try:
        bundle = analyze(args.result_root, args.project_root)
        manifest_path = write_bundle(bundle, output_dir)
    except AnalysisError as exc:
        print(json.dumps({"status": "input_error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": bundle.metadata["status"],
        "complete_cells": bundle.metadata["complete_cells"],
        "required_cells": bundle.metadata["required_cells"],
        "analysis_manifest": str(manifest_path),
    }, ensure_ascii=False, sort_keys=True))
    if args.require_complete and bundle.metadata["status"] != "complete":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

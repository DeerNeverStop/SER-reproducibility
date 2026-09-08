"""Fail-closed, outcome-blind execution wrapper for the frozen N14R plan.

The wrapper reuses the frozen v2 neural-network engine but does not reuse its
budget stopping logic.  It executes complete draws in their preregistered
order, permits at most one retry after a recorded unit failure, verifies every
completed unit before resume, and performs only within-unit structural and
checkpoint-consistency checks.  It never selects configurations or computes a
fold, cell, draw, or cross-draw outcome.  Scoring is a separate post-completion
command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .spec import ALL_DRAW_IDS, N_PRIMARY_DRAWS, SPEC, STUDY_ID, canonical_json


class IntegrityError(RuntimeError):
    """Raised when a frozen or completed artifact does not match its contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: Any) -> None:
    from v2.ser_v2.common import atomic_write_json

    atomic_write_json(path, value)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_plan(plan_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    with (plan_dir / "run_plan.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    configs = _read_json(plan_dir / "unit_configs.json")
    required = {
        "unit_id", "arm", "draw_id", "draw_role", "fold", "train_rep",
        "cell", "config_index", "r", "seed_index", "train_seed",
        "config_sha256", "split_sha256",
    }
    if not rows or not required.issubset(rows[0]):
        raise IntegrityError(f"run_plan.csv lacks fields: {sorted(required - set(rows[0] if rows else []))}")
    if len(rows) != int(SPEC["design"]["maximum_units"]):
        raise IntegrityError("run_plan.csv does not contain the frozen maximum unit count")
    if len({row["unit_id"] for row in rows}) != len(rows):
        raise IntegrityError("duplicate unit_id in run_plan.csv")
    for row in rows:
        if row["arm"] != STUDY_ID:
            raise IntegrityError(f"unexpected arm for {row['unit_id']}: {row['arm']}")
        if int(row["draw_id"]) != int(row["r"]) or int(row["train_rep"]) != int(row["seed_index"]):
            raise IntegrityError(f"legacy compatibility mirrors disagree for {row['unit_id']}")
        cfg = configs.get(row["config_sha256"])
        if cfg is None or _sha256_bytes(canonical_json(cfg).encode("utf-8")) != row["config_sha256"]:
            raise IntegrityError(f"missing or mis-hashed config for {row['unit_id']}")
        if int(cfg.get("hpo_config_index", -1)) != int(row["config_index"]):
            raise IntegrityError(f"config index mismatch for {row['unit_id']}")
    return rows, configs


IDENTITY_FIELDS = (
    "arm", "corpus_level", "model", "cell", "fold", "r", "seed_index",
    "train_seed", "config_sha256", "split_sha256", "n_fit", "n_val", "n_test",
)

INTEGER_IDENTITY_FIELDS = {"fold", "r", "seed_index", "train_seed", "n_fit", "n_val", "n_test"}
PREDICTION_FIELDS = [
    "sample_index", "relative_path", "speaker", "y_true", "y_pred",
    *(f"logit_{label}" for label in range(6)),
]
FLOAT_ATOL = 1e-10


def preflight_completed_unit(
    row: dict[str, str],
    run_dir: Path,
    configs: dict[str, dict[str, Any]],
    expected_fold: dict[str, Any] | None = None,
    manifest: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Return ``done`` or ``pending``; raise on a corrupt completed unit.

    This gate parses each unit only for local integrity and exact frozen-split
    coverage.  It never selects a configuration, aggregates folds/cells, or
    exposes an effect statistic, preserving the no-interim-scoring rule.
    """
    unit_dir = run_dir / "units" / row["unit_id"]
    done_path = unit_dir / "DONE"
    if not done_path.exists():
        return "pending"
    needed = [unit_dir / name for name in ("unit.json", "history.json", "predictions.csv")]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise IntegrityError(f"DONE unit {row['unit_id']} is missing files: {missing}")
    try:
        unit = _read_json(unit_dir / "unit.json")
        history = _read_json(unit_dir / "history.json")
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"unreadable metadata for completed unit {row['unit_id']}: {exc}") from exc
    if not isinstance(history, list) or not history:
        raise IntegrityError(f"history is not a non-empty list for completed unit {row['unit_id']}")
    for field in IDENTITY_FIELDS:
        expected: Any = row[field]
        actual = unit.get(field)
        if field in INTEGER_IDENTITY_FIELDS:
            expected = int(expected)
        if actual != expected:
            raise IntegrityError(
                f"completed unit {row['unit_id']} has {field}={actual!r}, expected {expected!r}"
            )
    if unit.get("unit_id") != row["unit_id"] or unit.get("status") != "done":
        raise IntegrityError(f"unit identity/status mismatch for {row['unit_id']}")
    expected_cfg = configs[row["config_sha256"]]
    if canonical_json(unit.get("config")) != canonical_json(expected_cfg):
        raise IntegrityError(f"unit config payload mismatch for {row['unit_id']}")

    try:
        epochs = [int(item["epoch"]) for item in history]
        losses = [float(item["val_loss"]) for item in history]
        validation_uars = [float(item["val_uar"]) for item in history]
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"malformed validation history for {row['unit_id']}") from exc
    if epochs != list(range(1, len(history) + 1)) or int(unit.get("epochs_run", -1)) != len(history):
        raise IntegrityError(f"history epoch sequence/epochs_run mismatch for {row['unit_id']}")
    if not all(math.isfinite(value) for value in (*losses, *validation_uars)):
        raise IntegrityError(f"non-finite validation history for {row['unit_id']}")
    if not all(0.0 <= value <= 100.0 for value in validation_uars):
        raise IntegrityError(f"validation UAR outside 0..100 for {row['unit_id']}")
    best_index = min(range(len(losses)), key=losses.__getitem__)
    try:
        reported_epoch = int(unit["best_epoch"])
        reported_loss = float(unit["val_loss_best"])
        reported_uar = float(unit["val_uar_best"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"invalid best-checkpoint metadata for {row['unit_id']}") from exc
    if reported_epoch != epochs[best_index]:
        raise IntegrityError(f"best_epoch is not the first minimum-loss epoch for {row['unit_id']}")
    if not math.isclose(reported_loss, losses[best_index], rel_tol=0.0, abs_tol=FLOAT_ATOL):
        raise IntegrityError(f"val_loss_best differs from history for {row['unit_id']}")
    if not math.isclose(reported_uar, validation_uars[best_index], rel_tol=0.0, abs_tol=FLOAT_ATOL):
        raise IntegrityError(f"val_uar_best differs from history for {row['unit_id']}")

    prediction_path = unit_dir / "predictions.csv"
    try:
        with prediction_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PREDICTION_FIELDS:
                raise IntegrityError(f"prediction columns changed for {row['unit_id']}")
            predictions = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise IntegrityError(f"unreadable prediction table for {row['unit_id']}: {exc}") from exc
    if len(predictions) != int(row["n_test"]):
        raise IntegrityError(f"prediction row count differs from plan for {row['unit_id']}")
    seen_paths: set[str] = set()
    seen_indices: set[int] = set()
    parsed_predictions: dict[str, tuple[int, str, int]] = {}
    for prediction in predictions:
        path = prediction.get("relative_path", "")
        try:
            sample_index = int(prediction["sample_index"])
            y_true = int(prediction["y_true"])
            y_pred = int(prediction["y_pred"])
            logits = [float(prediction[f"logit_{label}"]) for label in range(6)]
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError(f"malformed prediction row for {row['unit_id']}") from exc
        if not path or path in seen_paths or sample_index in seen_indices:
            raise IntegrityError(f"duplicate or missing prediction identity for {row['unit_id']}")
        if not (0 <= y_true < 6 and 0 <= y_pred < 6) or not all(math.isfinite(value) for value in logits):
            raise IntegrityError(f"invalid prediction label/logit for {row['unit_id']}")
        if max(range(6), key=logits.__getitem__) != y_pred:
            raise IntegrityError(f"prediction argmax mismatch for {row['unit_id']}")
        seen_paths.add(path)
        seen_indices.add(sample_index)
        parsed_predictions[path] = (sample_index, str(prediction.get("speaker", "")), y_true)

    if expected_fold is not None:
        for key, plan_count in (("fit", "n_fit"), ("val", "n_val"), ("test", "n_test")):
            if len(expected_fold.get(key, [])) != int(row[plan_count]):
                raise IntegrityError(f"frozen {key} count differs from plan for {row['unit_id']}")
        expected_paths = list(expected_fold["test"])
        if set(expected_paths) != seen_paths or len(expected_paths) != len(predictions):
            raise IntegrityError(f"predictions do not exactly cover frozen test paths for {row['unit_id']}")
        if manifest is None:
            raise IntegrityError("manifest is required with expected_fold")
        for path in expected_paths:
            reference = manifest.get(path)
            if reference is None:
                raise IntegrityError(f"frozen test path is absent from manifest: {path}")
            observed_index, observed_speaker, observed_label = parsed_predictions[path]
            if (
                observed_index != int(reference["sample_index"])
                or observed_speaker != str(reference["speaker"])
                or observed_label != int(reference["label_index"])
            ):
                raise IntegrityError(f"prediction/manifest identity mismatch for {row['unit_id']}: {path}")

    prediction_sha = _sha256_file(unit_dir / "predictions.csv")
    expected_receipt = (prediction_sha + "\n").encode("ascii")
    if done_path.read_bytes() != expected_receipt or unit.get("predictions_sha256") != prediction_sha:
        raise IntegrityError(f"prediction/DONE/unit hash mismatch for {row['unit_id']}")
    return "done"


def _failure_counts(run_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    path = run_dir / "ledger.jsonl"
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid ledger line {line_no}: {exc}") from exc
            if event.get("event") == "failed" and isinstance(event.get("unit_id"), str):
                uid = event["unit_id"]
                counts[uid] = counts.get(uid, 0) + 1
    return counts


def _reconcile_orphan_attempts(
    rows: list[dict[str, str]],
    run_dir: Path,
    configs: dict[str, dict[str, Any]],
    plan_io: Any,
) -> None:
    """Close starts left without a terminal event by an interrupted process.

    The exclusive wrapper lock guarantees there is no live trainer at this
    point.  A fully valid atomically written unit is recovered as a success;
    otherwise the interrupted start consumes one failed attempt.
    """
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        if any((run_dir / "units" / row["unit_id"] / "DONE").exists() for row in rows):
            raise IntegrityError("completed artifacts exist without an execution ledger")
        return
    plan_by_id = {row["unit_id"]: row for row in rows}
    starts: dict[str, int] = {}
    state: dict[str, str] = {}
    done_receipts: dict[str, str] = {}
    with ledger.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid ledger line {line_number}: {exc}") from exc
            uid = event.get("unit_id")
            kind = event.get("event")
            if uid not in plan_by_id or kind not in {"start", "done", "failed"}:
                raise IntegrityError(f"unknown unit/event at ledger line {line_number}")
            if kind == "start":
                if state.get(uid, "idle") != "idle":
                    raise IntegrityError(f"start event is out of sequence for {uid}")
                starts[uid] = starts.get(uid, 0) + 1
                if starts[uid] > int(SPEC["completion"]["maximum_attempts_per_unit"]):
                    raise IntegrityError(f"unit exceeded the frozen two-attempt ceiling: {uid}")
                state[uid] = "open"
            elif kind in {"done", "failed"}:
                if state.get(uid, "idle") != "open":
                    raise IntegrityError(f"terminal event without a matching start for {uid}")
                state[uid] = "done" if kind == "done" else "idle"
                if kind == "done":
                    done_receipts[uid] = str(event.get("predictions_sha256", ""))
    for uid, current in state.items():
        row = plan_by_id[uid]
        expected_fold = plan_io.fold(row)
        manifest = plan_io.manifest[row["base_corpus"]]
        if current == "done":
            if preflight_completed_unit(row, run_dir, configs, expected_fold, manifest) != "done":
                raise IntegrityError(f"ledger-complete unit lacks a valid DONE artifact: {uid}")
            if done_receipts.get(uid) != _sha256_file(run_dir / "units" / uid / "predictions.csv"):
                raise IntegrityError(f"ledger prediction receipt mismatch for {uid}")
            continue
        if current != "open":
            continue
        if preflight_completed_unit(row, run_dir, configs, expected_fold, manifest) == "done":
            prediction_sha = _sha256_file(run_dir / "units" / uid / "predictions.csv")
            _append_event(ledger, {
                "unit_id": uid, "event": "done", "predictions_sha256": prediction_sha,
                "seconds": None, "at": _now(), "recovered_after_interruption": True,
            })
            state[uid] = "done"
        else:
            _append_event(ledger, {
                "unit_id": uid, "event": "failed",
                "error": "interrupted attempt recovered without a complete DONE artifact",
                "at": _now(), "recovered_after_interruption": True,
            })
            state[uid] = "idle"
    for row in rows:
        uid = row["unit_id"]
        if (run_dir / "units" / uid / "DONE").exists() and state.get(uid) != "done":
            # Two closed failures are handled by the permanent draw-void gate;
            # any other unreceipted success is outside the execution contract.
            if starts.get(uid, 0) >= int(SPEC["completion"]["maximum_attempts_per_unit"]):
                continue
            raise IntegrityError(f"completed artifact has no matching ledger success: {uid}")


def _verify_pins(plan_dir: Path, features_dir: Path) -> dict[str, Any]:
    study_root = Path(__file__).resolve().parent
    repo = study_root.parents[1]
    if plan_dir.resolve() != (study_root / "plan").resolve():
        raise IntegrityError("N14R must execute the canonical pinned plan directory")
    pins_path = study_root / "PINS.json"
    if not pins_path.is_file():
        raise IntegrityError("missing frozen preregistration file: PINS.json")
    try:
        pins = _read_json(pins_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"PINS.json is unreadable: {exc}") from exc
    from .make_pins import verify_record

    failures = verify_record(pins, repo, features_dir)
    if failures:
        raise IntegrityError(f"PINS.json verification failed: {failures[:8]}")
    return pins


def _locked_package_versions(study_root: Path) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    lock_path = study_root / "requirements-lock.txt"
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(lock_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise IntegrityError(f"unsupported requirements lock line {line_number}: {line!r}")
        name, wanted = line.split("==", 1)
        try:
            actual = version(name)
        except PackageNotFoundError as exc:
            raise IntegrityError(f"frozen package is not installed: {name}") from exc
        if actual != wanted:
            raise IntegrityError(f"frozen package version mismatch for {name}: {actual} != {wanted}")
        expected[name] = actual
    if not expected:
        raise IntegrityError("requirements-lock.txt is empty")
    return dict(sorted(expected.items(), key=lambda item: item[0].lower()))


def _validate_feature_cache(
    plan_dir: Path,
    manifests_dir: Path,
    features_dir: Path,
    pins: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    from v2.ser_v2 import corpora, features as feature_code

    entries = pins.get("external_feature_cache")
    if not isinstance(entries, list) or len(entries) != 2 or not all(isinstance(item, dict) for item in entries):
        raise IntegrityError("PINS.json must identify exactly one CREMA-D NPZ and its JSON metadata")
    npz_entries = [item for item in entries if str(item.get("name", "")).endswith(".npz")]
    json_entries = [item for item in entries if str(item.get("name", "")).endswith(".json")]
    if len(npz_entries) != 1 or len(json_entries) != 1:
        raise IntegrityError("PINS external cache inventory must contain one NPZ and one JSON")
    npz_path = features_dir / str(npz_entries[0]["name"])
    metadata_path = features_dir / str(json_entries[0]["name"])
    if npz_path.stem != metadata_path.stem:
        raise IntegrityError("pinned feature NPZ and metadata do not share a stem")
    actual_candidates = sorted(features_dir.glob("cremad__logmel__*.npz"))
    if len(actual_candidates) != 1 or actual_candidates[0].resolve() != npz_path.resolve():
        raise IntegrityError("feature directory must contain exactly the single pinned CREMA-D log-mel NPZ")

    manifest_path = manifests_dir / "cremad_manifest.csv"
    with (plan_dir / "run_plan.csv").open("r", encoding="utf-8", newline="") as handle:
        plan_rows = list(csv.DictReader(handle))
    manifest_hashes = {row.get("manifest_sha256") for row in plan_rows}
    if len(manifest_hashes) != 1 or not manifest_path.is_file() or _sha256_file(manifest_path) != next(iter(manifest_hashes)):
        raise IntegrityError("execution manifest is absent or differs from the frozen plan")
    raw_manifest = corpora.load_manifest(manifest_path)
    clean_manifest, hygiene = corpora.apply_hygiene("cremad", raw_manifest)
    if (
        hygiene.get("n_input") != int(SPEC["design"]["manifest_input_rows"])
        or hygiene.get("n_kept") != int(SPEC["design"]["analysis_population_rows_after_hygiene"])
    ):
        raise IntegrityError("CREMA-D hygiene counts differ from the frozen design")
    expected_paths = [str(row["relative_path"]) for row in clean_manifest]

    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            if set(payload.files) != {"X", "paths"}:
                raise IntegrityError("feature NPZ members must be exactly X and paths")
            matrix = payload["X"]
            raw_paths = payload["paths"]
            if matrix.dtype != np.dtype("float32") or matrix.shape != (len(expected_paths), 64, 128):
                raise IntegrityError("feature matrix shape/dtype differs from the frozen contract")
            if not matrix.flags.c_contiguous or not bool(np.isfinite(matrix).all()):
                raise IntegrityError("feature matrix must be C-contiguous and entirely finite")
            if raw_paths.ndim != 1 or raw_paths.dtype.kind != "U":
                raise IntegrityError("feature paths must be a one-dimensional Unicode array")
            cache_paths = [str(value) for value in raw_paths.tolist()]
    except (OSError, ValueError, KeyError) as exc:
        raise IntegrityError(f"feature NPZ is unreadable: {exc}") from exc
    if cache_paths != expected_paths or len(set(cache_paths)) != len(cache_paths):
        raise IntegrityError("feature paths do not exactly equal the hygienic manifest in order")
    for value in cache_paths:
        parsed = PurePosixPath(value)
        if not value or parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value:
            raise IntegrityError(f"unsafe or non-canonical feature path: {value!r}")

    metadata = _read_json(metadata_path)
    if metadata != {
        "corpus": "cremad",
        "n": len(expected_paths),
        "config": feature_code.LOGMEL,
        "sha256": _sha256_file(npz_path),
    }:
        raise IntegrityError("feature metadata differs from the frozen log-mel contract")
    path_token = _sha256_bytes("\n".join(expected_paths).encode("utf-8"))[:8]
    code_token = _sha256_file(Path(feature_code.__file__).resolve())[:8]
    if not npz_path.name.endswith(f"__m{path_token}__c{code_token}.npz"):
        raise IntegrityError("feature-cache filename content-address tokens are invalid")

    split_index = _read_json(plan_dir / "split_index.json")
    if not isinstance(split_index, dict) or len(split_index) != 56:
        raise IntegrityError("split index must contain exactly 56 cell tables")
    expected_set = set(expected_paths)
    for key, entry in split_index.items():
        split_path = plan_dir / str(entry.get("path", ""))
        split = _read_json(split_path)
        if split.get("population") != expected_paths:
            raise IntegrityError(f"split population differs from feature cache: {key}")
        folds = split.get("folds")
        if not isinstance(folds, list) or len(folds) != 5:
            raise IntegrityError(f"split does not contain five folds: {key}")
        for fold in folds:
            parts = [list(fold.get(name, [])) for name in ("fit", "val", "test")]
            sets = [set(part) for part in parts]
            if any(len(part) != len(group) for part, group in zip(parts, sets)):
                raise IntegrityError(f"split contains duplicate paths: {key}")
            if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2] or set.union(*sets) != expected_set:
                raise IntegrityError(f"split does not partition the hygienic manifest: {key}")

    return [
        {
            "basename": path.name,
            "path": str(path.resolve()),
            "sha256": entry["sha256"],
            "size_bytes": path.stat().st_size,
        }
        for path, entry in ((npz_path, npz_entries[0]), (metadata_path, json_entries[0]))
    ]


def _environment_record(plan_dir: Path, manifests_dir: Path, features_dir: Path) -> dict[str, Any]:
    import numpy
    import scipy
    import sklearn
    import torch
    import librosa

    files = ["run_plan.csv", "split_index.json", "unit_configs.json", "plan_summary.json", "hygiene.json"]
    study_root = plan_dir.parent
    repo = Path(__file__).resolve().parents[2]
    if manifests_dir.resolve() != (repo / "v2" / "manifests").resolve():
        raise IntegrityError("N14R must execute against the canonical pinned manifest directory")
    preregistration_files = ["spec.json", "PINS.json"]
    missing_prereg = [name for name in preregistration_files if not (study_root / name).exists()]
    if missing_prereg:
        raise IntegrityError(f"missing frozen preregistration files: {missing_prereg}")
    pins = _verify_pins(plan_dir, features_dir)
    external_feature_cache = _validate_feature_cache(plan_dir, manifests_dir, features_dir, pins)
    package_versions = _locked_package_versions(study_root)
    runtime = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "librosa": librosa.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if runtime != SPEC["execution_environment"] or not torch.cuda.is_available():
        raise IntegrityError(
            f"runtime differs from frozen execution environment: observed={runtime!r}"
        )
    record: dict[str, Any] = {
        "schema": "ser26-n14r-execution-environment-1",
        **runtime,
        "python_full": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "package_versions": package_versions,
        "plan_files": {name: _sha256_file(plan_dir / name) for name in files},
        "preregistration_files": {
            name: _sha256_file(study_root / name) for name in preregistration_files
        },
        "external_feature_cache": external_feature_cache,
    }
    fingerprint_payload = {k: v for k, v in record.items() if k != "schema"}
    record["fingerprint_sha256"] = _sha256_bytes(canonical_json(fingerprint_payload).encode("utf-8"))
    return record


def freeze_execution_environment(
    plan_dir: Path, manifests_dir: Path, run_dir: Path, features_dir: Path,
) -> dict[str, Any]:
    current = _environment_record(plan_dir, manifests_dir, features_dir)
    path = run_dir / "execution_environment.json"
    if path.exists():
        prior = _read_json(path)
        if prior.get("fingerprint_sha256") != current["fingerprint_sha256"]:
            raise IntegrityError("execution environment changed since the first N14R fit")
    else:
        _write_json_atomic(path, current)
    return current


@contextmanager
def exclusive_runner_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".n14r-run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise IntegrityError(
            f"runner lock exists at {lock}; verify no process is active before removing a stale lock"
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": _now()}) + "\n")
        yield
    finally:
        if lock.exists():
            lock.unlink()


def _draw_rows(rows: list[dict[str, str]], draw_id: int) -> list[dict[str, str]]:
    selected = [row for row in rows if int(row["draw_id"]) == draw_id]
    if len(selected) != int(SPEC["completion"]["units_per_draw"]):
        raise IntegrityError(f"draw {draw_id} does not contain exactly 80 planned units")
    return selected


def _draw_state(
    rows: list[dict[str, str]],
    run_dir: Path,
    configs: dict[str, dict[str, Any]],
    plan_io: Any | None = None,
) -> tuple[str, list[str]]:
    pending: list[str] = []
    for row in rows:
        expected_fold = plan_io.fold(row) if plan_io is not None else None
        manifest = plan_io.manifest[row["base_corpus"]] if plan_io is not None else None
        if preflight_completed_unit(row, run_dir, configs, expected_fold, manifest) != "done":
            pending.append(row["unit_id"])
    return ("complete" if not pending else "pending"), pending


def _exhausted_unit_ids(rows: list[dict[str, str]], failures: dict[str, int]) -> list[str]:
    """A second recorded failure permanently voids a draw, even after a later DONE."""
    return sorted(row["unit_id"] for row in rows if failures.get(row["unit_id"], 0) >= 2)


def create_analysis_lock(
    plan_dir: Path,
    run_dir: Path,
    rows: list[dict[str, str]],
    complete_draw_ids: list[int],
    environment_fingerprint: str,
    audit_ledger: Path,
) -> dict[str, Any]:
    """Hash the locked analysis set without parsing any outcome value."""
    path = run_dir / "analysis_lock.json"
    selected = [row for row in rows if int(row["draw_id"]) in set(complete_draw_ids)]
    if len(selected) != N_PRIMARY_DRAWS * int(SPEC["completion"]["units_per_draw"]):
        raise IntegrityError("analysis lock requires exactly 24 complete 80-unit draws")
    artifacts: dict[str, dict[str, str]] = {}
    for row in sorted(selected, key=lambda item: item["unit_id"]):
        unit_dir = run_dir / "units" / row["unit_id"]
        artifacts[row["unit_id"]] = {
            name: _sha256_file(unit_dir / name)
            for name in ("unit.json", "history.json", "predictions.csv", "DONE")
        }
    payload: dict[str, Any] = {
        "schema": "ser26-n14r-analysis-lock-1",
        "study_id": STUDY_ID,
        "created_at": _now(),
        "contains_outcome_statistics": False,
        "complete_draw_ids": list(complete_draw_ids),
        "n_complete_draws": len(complete_draw_ids),
        "n_locked_units": len(selected),
        "environment_fingerprint": environment_fingerprint,
        "plan_hashes": {
            name: _sha256_file(plan_dir / name)
            for name in ("run_plan.csv", "split_index.json", "unit_configs.json", "plan_summary.json", "hygiene.json")
        },
        "preregistration_hashes": {
            name: _sha256_file(plan_dir.parent / name)
            for name in ("spec.json", "PINS.json")
        },
        "external_feature_cache": _read_json(run_dir / "execution_environment.json").get(
            "external_feature_cache"
        ),
        "execution_ledger_sha256": _sha256_file(run_dir / "ledger.jsonl"),
        "audit_ledger_prefix_sha256": _sha256_file(audit_ledger),
        "unit_artifact_hashes": artifacts,
    }
    payload_without_self = canonical_json(payload).encode("utf-8")
    payload["lock_payload_sha256"] = _sha256_bytes(payload_without_self)
    if path.exists():
        prior = _read_json(path)
        # Timestamps are part of the original lock.  Never silently rewrite it.
        for key in (
            "complete_draw_ids", "n_locked_units", "environment_fingerprint",
            "plan_hashes", "preregistration_hashes", "execution_ledger_sha256",
            "external_feature_cache", "unit_artifact_hashes",
        ):
            if prior.get(key) != payload.get(key):
                raise IntegrityError(f"existing analysis lock disagrees on {key}")
        return prior
    _write_json_atomic(path, payload)
    _append_event(audit_ledger, {
        "event": "analysis_lock", "at": _now(),
        "complete_draw_ids": list(complete_draw_ids),
        "analysis_lock_sha256": _sha256_file(path),
        "contains_outcome_statistics": False,
    })
    return payload


def execute(
    plan_dir: Path,
    manifests_dir: Path,
    features_dir: Path,
    run_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute until 24 complete draws exist or all four reserves are exhausted."""
    from v2.ser_v2 import train

    if device != "cuda":
        raise IntegrityError("the frozen N14R execution device is exactly 'cuda'")

    rows, configs = load_plan(plan_dir)
    audit_ledger = run_dir / "n14r_execution_ledger.jsonl"
    complete: list[int] = []
    void: dict[int, str] = {}
    activated: list[int] = []
    with exclusive_runner_lock(run_dir):
        environment = freeze_execution_environment(plan_dir, manifests_dir, run_dir, features_dir)
        plan_io = train.PlanIO(plan_dir, manifests_dir, run_dir)
        _reconcile_orphan_attempts(rows, run_dir, configs, plan_io)
        _append_event(audit_ledger, {
            "event": "runner_start", "at": _now(),
            "environment_fingerprint": environment["fingerprint_sha256"],
        })
        for draw_id in ALL_DRAW_IDS:
            if len(complete) >= N_PRIMARY_DRAWS:
                break
            draw = _draw_rows(rows, draw_id)
            failures = _failure_counts(run_dir)
            exhausted = _exhausted_unit_ids(draw, failures)
            if exhausted:
                void[draw_id] = f"{len(exhausted)} unit(s) exhausted the fixed two-attempt allowance"
                _append_event(audit_ledger, {
                    "event": "draw_void", "draw_id": draw_id, "at": _now(),
                    "reason": void[draw_id], "unit_ids": exhausted,
                })
                continue
            state, pending = _draw_state(draw, run_dir, configs, plan_io)
            if state == "complete":
                complete.append(draw_id)
                continue
            activated.append(draw_id)
            _append_event(audit_ledger, {
                "event": "draw_activated", "draw_id": draw_id,
                "draw_role": draw[0]["draw_role"], "at": _now(),
                "n_pending": len(pending),
            })
            # Re-enter only while every missing unit remains below the frozen
            # two-attempt ceiling.  run_arm sees only this draw and preserves
            # its fixed plan order.
            while True:
                failures = _failure_counts(run_dir)
                exhausted = _exhausted_unit_ids(draw, failures)
                if exhausted:
                    void[draw_id] = f"{len(exhausted)} unit(s) exhausted the fixed two-attempt allowance"
                    _append_event(audit_ledger, {
                        "event": "draw_void", "draw_id": draw_id, "at": _now(),
                        "reason": void[draw_id], "unit_ids": exhausted,
                    })
                    break
                state, pending = _draw_state(draw, run_dir, configs, plan_io)
                if state == "complete":
                    complete.append(draw_id)
                    _append_event(audit_ledger, {"event": "draw_complete", "draw_id": draw_id, "at": _now()})
                    break
                # Rebuild and compare the full PINS inventory immediately before
                # every training pass.  This catches source/cache additions or
                # mutations during a long resumable execution.
                _verify_pins(plan_dir, features_dir)
                before_done = len(draw) - len(pending)
                result = train.run_arm(
                    plan_dir, manifests_dir, features_dir, run_dir, STUDY_ID,
                    None, device, None, None, {"draw_id": str(draw_id)},
                )
                _verify_pins(plan_dir, features_dir)
                state_after, pending_after = _draw_state(draw, run_dir, configs, plan_io)
                after_done = len(draw) - len(pending_after)
                _append_event(audit_ledger, {
                    "event": "draw_pass_finished", "draw_id": draw_id, "at": _now(),
                    "done_before": before_done, "done_after": after_done,
                    "runner_result": result,
                })
                if state_after != "complete" and after_done == before_done and int(result.get("failed", 0)) == 0:
                    raise IntegrityError(f"draw {draw_id} made no progress and recorded no failure")
        status = "complete" if len(complete) == N_PRIMARY_DRAWS else "not_tested_incomplete"
        analysis_lock = None
        if status == "complete":
            _verify_pins(plan_dir, features_dir)
            analysis_lock = create_analysis_lock(
                plan_dir, run_dir, rows, complete,
                environment["fingerprint_sha256"], audit_ledger,
            )
        completion = {
            "schema": "ser26-n14r-completion-1",
            "study_id": STUDY_ID,
            "status": status,
            "required_complete_draws": N_PRIMARY_DRAWS,
            "complete_draw_ids": complete,
            "n_complete_draws": len(complete),
            "void_draws": {str(k): v for k, v in sorted(void.items())},
            "activated_draw_ids_this_invocation": activated,
            "environment_fingerprint": environment["fingerprint_sha256"],
            "analysis_lock_payload_sha256": (
                analysis_lock["lock_payload_sha256"] if analysis_lock is not None else None
            ),
            "written_at": _now(),
            "contains_outcome_statistics": False,
        }
        _write_json_atomic(run_dir / "completion.json", completion)
        _append_event(audit_ledger, {"event": "runner_stop", "at": _now(), "status": status})
        return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = execute(args.plan, args.manifests, args.features, args.run, args.device)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

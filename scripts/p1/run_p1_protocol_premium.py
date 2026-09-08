"""Prepare and execute the frozen P1 protocol-premium experiment.

Formal usage (SER environment, project directory):

    python tools/run_p1_protocol_premium.py prepare
    python tools/run_p1_protocol_premium.py smoke
    python tools/run_p1_protocol_premium.py run --corpus ravdess
    python tools/run_p1_protocol_premium.py run --corpus cremad
    python tools/run_p1_protocol_premium.py status

The runner is resumable.  A completed unit directory is immutable and skipped.
No outer-test aggregate is printed or calculated here; all final metrics are
rebuilt later from the per-utterance logits CSV files.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from p1_protocol_core import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CORPORA,
    EXPECTED_MODEL_CONFIGS,
    MODELS,
    PREPROCESS_CONFIG,
    PROJECT_ROOT,
    PROTOCOLS,
    RESULT_ROOT,
    SEEDS,
    SER_COMMIT,
    SER_FILE_SHA256,
    SER_ROOT,
    atomic_write_csv,
    atomic_write_json,
    assert_frozen_inputs,
    canonical_json,
    environment_record,
    load_cache,
    now_iso,
    output_size_bytes,
    prepare_cache,
    prepare_manifest,
    prepare_run_plan,
    prepare_splits,
    read_csv,
    sha256_file,
    stable_u32,
    validate_manifest,
    verify_model_configs,
)


MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
GPU_SECONDS_LIMIT = 200 * 3600
WALL_DAYS_LIMIT = 14
OUTPUT_BYTES_LIMIT = 20 * 1024**3
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = RUNNER_PATH.with_name("p1_protocol_core.py")


def import_ser_symbols_after_hash_gate():
    assert_frozen_inputs()
    sys.path.insert(0, str(SER_ROOT))
    try:
        import advanced_models
        import fno_data
    finally:
        if sys.path and sys.path[0] == str(SER_ROOT):
            sys.path.pop(0)
    expected_modules = {
        "advanced_models": SER_ROOT / "advanced_models.py",
        "fno_data": SER_ROOT / "fno_data.py",
    }
    observed_modules = {
        "advanced_models": Path(advanced_models.__file__).resolve(),
        "fno_data": Path(fno_data.__file__).resolve(),
    }
    for name, expected in expected_modules.items():
        if observed_modules[name] != expected.resolve():
            raise RuntimeError(f"unexpected module source for {name}: {observed_modules[name]}")
    return advanced_models.build_neural_model, advanced_models.parameter_count, fno_data.spec_augment


BUILD_MODEL = None
PARAMETER_COUNT = None
SPEC_AUGMENT = None


def ensure_ser_symbols_loaded() -> None:
    global BUILD_MODEL, PARAMETER_COUNT, SPEC_AUGMENT
    if BUILD_MODEL is None:
        BUILD_MODEL, PARAMETER_COUNT, SPEC_AUGMENT = import_ser_symbols_after_hash_gate()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class TrainDataset(Dataset):
    """In-memory data with deterministic, fresh train-only augmentation."""

    def __init__(self, X, y, seed: int, config: dict):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.rng = np.random.default_rng(seed + 7919)
        self.config = config

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        spec = SPEC_AUGMENT(
            self.X[index],
            n_time_masks=2,
            time_mask_width=int(self.config["time_mask"]),
            n_freq_masks=2,
            freq_mask_width=int(self.config["freq_mask"]),
            noise_std=float(self.config["noise_std"]),
            rng=self.rng,
        )
        return torch.tensor(spec, dtype=torch.float32), int(self.y[index])


def make_train_loader(X, y, seed: int, config: dict) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TrainDataset(X, y, seed, config),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


def make_eval_loader(X, y) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(np.asarray(X), dtype=torch.float32),
        torch.tensor(np.asarray(y), dtype=torch.int64),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def class_balanced_criterion(y_fit: np.ndarray, n_classes: int, device: torch.device) -> nn.Module:
    counts = np.bincount(y_fit.astype(np.int64), minlength=n_classes)
    if np.any(counts == 0):
        raise RuntimeError(f"fit subset is missing classes: counts={counts.tolist()}")
    weights = len(y_fit) / (n_classes * counts.astype(np.float64))
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def validation_pass(model, loader, criterion, device) -> dict:
    model.eval()
    total_weighted_loss = 0.0
    total_target_weight = 0.0
    predictions = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb_device = yb.to(device, non_blocking=True)
            logits = model(xb)
            weighted_loss = F.cross_entropy(
                logits, yb_device, weight=criterion.weight, reduction="sum"
            )
            target_weight = criterion.weight[yb_device].sum()
            total_weighted_loss += float(weighted_loss.detach().cpu())
            total_target_weight += float(target_weight.detach().cpu())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(yb.numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {"loss": total_weighted_loss / total_target_weight, **metrics(y_true, y_pred)}


def evaluate_outer_test_once(model, X, y, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_parts = []
    with torch.no_grad():
        for xb, _ in make_eval_loader(X, y):
            logits_parts.append(model(xb.to(device, non_blocking=True)).cpu().numpy())
    logits = np.concatenate(logits_parts).astype(np.float32, copy=False)
    return logits, logits.argmax(axis=1).astype(np.int64)


def train_one_fit(
    model, X_fit, y_fit, X_val, y_val, config, train_seed, n_classes, device,
    epoch_callback=None,
):
    set_reproducible_seed(train_seed)
    model.to(device)
    train_loader = make_train_loader(X_fit, y_fit, train_seed, config)
    val_loader = make_eval_loader(X_val, y_val)
    criterion = class_balanced_criterion(y_fit, n_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_state = None
    best_epoch = None
    best_loss = math.inf
    stale = 0
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += float(F.cross_entropy(
                    logits.detach(), yb, weight=criterion.weight, reduction="sum"
                ).cpu())
                train_weight_sum += float(criterion.weight[yb].sum().detach().cpu())
        scheduler.step()
        val = validation_pass(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / train_weight_sum,
            "validation_loss": val["loss"],
            "validation_uar": val["uar"],
            "validation_accuracy": val["accuracy"],
            "validation_macro_f1": val["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if epoch_callback is not None:
            epoch_callback(epoch, time.perf_counter() - started)
        # Preregistered checkpoint rule: strict minimum validation loss only.
        if val["loss"] < best_loss:
            best_loss = float(val["loss"])
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if best_state is None or best_epoch is None:
        raise RuntimeError("training ended without a validation-loss checkpoint")
    model.load_state_dict(best_state)
    return model, history, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss": best_loss,
        "train_seconds": elapsed,
    }


def prepared_record_path() -> Path:
    return RESULT_ROOT / "prepared.json"


def load_prepared() -> dict:
    path = prepared_record_path()
    if not path.exists():
        raise RuntimeError("prepare has not completed")
    prepared = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(RESULT_ROOT / prepared["run_plan"]["path"]) != prepared["run_plan"]["sha256"]:
        raise RuntimeError("run_plan.csv hash drift")
    return prepared


def prepare() -> None:
    if prepared_record_path().exists():
        prepared = load_prepared()
        print(canonical_json({"status": "already_prepared", "run_plan": prepared["run_plan"]}))
        return
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = assert_frozen_inputs()
    models = verify_model_configs()
    prepared = {
        "prepared_at": now_iso(),
        "status": "prepared_before_formal_training",
        "frozen_inputs": frozen,
        "model_configs": models,
        "preprocess_config": PREPROCESS_CONFIG,
        "training": {
            "optimizer": "AdamW",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "scheduler": "CosineAnnealingLR(T_max=100)",
            "early_stopping": "strict minimum validation loss; patience=15; restore best",
            "class_weights": "fit-subset only",
            "mixed_precision": False,
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "statistics": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confirmatory_family": 24,
        },
        "limits": {
            "gpu_hours": 200,
            "wall_days": WALL_DAYS_LIMIT,
            "output_gib": 20,
            "failed_unit_retries": 1,
        },
        "corpora": {},
        "tool_sha256": {
            RUNNER_PATH.name: sha256_file(RUNNER_PATH),
            CORE_PATH.name: sha256_file(CORE_PATH),
        },
    }
    manifest_summaries = []
    for corpus in ("ravdess", "cremad"):
        manifest_path, rows = prepare_manifest(corpus)
        manifest = validate_manifest(corpus, manifest_path, rows)
        manifest_summaries.append(manifest)
        cache = prepare_cache(corpus, manifest_path, rows)
        splits = prepare_splits(corpus, rows)
        prepared["corpora"][corpus] = {"manifest": manifest, "cache": cache, "splits": splits}
    atomic_write_csv(
        RESULT_ROOT / "manifests" / "manifest_summary.csv",
        [
            "corpus", "manifest_path", "manifest_sha256", "n_samples", "n_speakers",
            "n_labels", "label_counts", "license", "source", "parse_failures", "read_failures",
        ],
        [{**row, "label_counts": canonical_json(row["label_counts"])} for row in manifest_summaries],
    )
    prepared["run_plan"] = prepare_run_plan(prepared)
    prepared["environment"] = environment_record()
    atomic_write_json(RESULT_ROOT / "environment.json", prepared["environment"])
    atomic_write_json(prepared_record_path(), prepared)
    print(canonical_json({"status": "prepared", "run_plan": prepared["run_plan"]}))


def refreeze_engineering() -> None:
    if (RESULT_ROOT / "formal_start.json").exists() or (RESULT_ROOT / "units").exists() or ledger_dir().exists():
        raise RuntimeError("cannot re-freeze engineering after formal training has started")
    prepared = load_prepared()
    frozen = assert_frozen_inputs()
    models = verify_model_configs()
    for corpus in ("ravdess", "cremad"):
        manifest_path = RESULT_ROOT / "manifests" / f"{corpus}_manifest.csv"
        rows = read_csv(manifest_path)
        manifest = validate_manifest(corpus, manifest_path, rows)
        if manifest["manifest_sha256"] != prepared["corpora"][corpus]["manifest"]["manifest_sha256"]:
            raise RuntimeError(f"manifest changed before engineering re-freeze: {corpus}")
        cache_path = RESULT_ROOT / prepared["corpora"][corpus]["cache"]["cache_path"]
        if sha256_file(cache_path) != prepared["corpora"][corpus]["cache"]["cache_sha256"]:
            raise RuntimeError(f"cache changed before engineering re-freeze: {corpus}")
        split_record = prepare_splits(corpus, rows)
        if canonical_json(split_record) != canonical_json(prepared["corpora"][corpus]["splits"]):
            raise RuntimeError(f"split artifacts changed before engineering re-freeze: {corpus}")
    plan_record = prepare_run_plan(prepared)
    if plan_record["sha256"] != prepared["run_plan"]["sha256"]:
        raise RuntimeError("run plan changed before engineering re-freeze")
    prepared["engineering_refrozen_at"] = now_iso()
    prepared["engineering_refreeze_reason"] = "preflight review fixes before first real P1 training result"
    prepared["frozen_inputs"] = frozen
    prepared["model_configs"] = models
    prepared["training"].update({
        "early_stopping": "strict global minimum of exact validation-set weighted CE; patience=15; restore best",
        "validation_loss_aggregation": "sum(weighted per-sample CE) / sum(target class weights) over full validation set",
        "outer_test_retry_policy": "no retry after any outer-test access; artifact commit is retried in-memory only",
        "gpu_budget_accounting": "atomic per-attempt ledger with epoch heartbeat and train+outer-test wall charge",
    })
    prepared["tool_sha256"] = {
        RUNNER_PATH.name: sha256_file(RUNNER_PATH),
        CORE_PATH.name: sha256_file(CORE_PATH),
    }
    atomic_write_json(prepared_record_path(), prepared)
    print(canonical_json({
        "status": "engineering_refrozen",
        "runner_sha256": prepared["tool_sha256"][RUNNER_PATH.name],
        "core_sha256": prepared["tool_sha256"][CORE_PATH.name],
        "run_plan_sha256": prepared["run_plan"]["sha256"],
    }))


def smoke() -> None:
    frozen = assert_frozen_inputs()
    observed_configs = verify_model_configs()
    ensure_ser_symbols_loaded()
    synthetic_y = np.tile(np.arange(8, dtype=np.int64), 24)
    synthetic_groups = np.repeat(np.asarray([f"{i + 1:02d}" for i in range(24)]), 8)
    from p1_protocol_core import generate_outer_splits, inner_split
    checks = []
    splits = generate_outer_splits("ravdess", synthetic_y, synthetic_groups)
    for protocol, folds in splits.items():
        tests = np.concatenate([test for _, test in folds])
        checks.append({"check": f"{protocol}_test_once", "pass": len(tests) == len(set(tests.tolist())) == len(synthetic_y)})
        for fold, (train, test) in enumerate(folds):
            checks.append({
                "check": f"{protocol}_fold_{fold}_index_disjoint",
                "pass": np.intersect1d(train, test).size == 0,
            })
            if protocol != "random":
                checks.append({
                    "check": f"{protocol}_fold_{fold}_speaker_disjoint",
                    "pass": not (set(synthetic_groups[train]) & set(synthetic_groups[test])),
                })
            fit, val, _, _ = inner_split(
                "ravdess", protocol, fold, 0, train, test, synthetic_y, synthetic_groups
            )
            checks.append({
                "check": f"{protocol}_fold_{fold}_inner_partitions_train",
                "pass": np.array_equal(np.sort(np.concatenate([fit, val])), np.sort(train)),
            })
            checks.append({
                "check": f"{protocol}_fold_{fold}_test_unseen_by_inner",
                "pass": np.intersect1d(np.concatenate([fit, val]), test).size == 0,
            })

    X = np.random.default_rng(123).normal(size=(4, 64, 128)).astype(np.float32)
    for model_name in MODELS:
        set_reproducible_seed(123)
        model = BUILD_MODEL(model_name, 64, 8, observed_configs[model_name])
        with torch.no_grad():
            output = model(torch.tensor(X))
        checks.append({
            "check": f"{model_name}_shape_4x8",
            "pass": tuple(output.shape) == (4, 8),
        })
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, xb):
            self.forward_calls += 1
            return torch.zeros((len(xb), 8), dtype=xb.dtype, device=xb.device)

    counting_model = CountingModel()
    test_X = np.zeros((BATCH_SIZE + 3, 64, 128), dtype=np.float32)
    test_y = np.zeros(BATCH_SIZE + 3, dtype=np.int64)
    logits, predictions = evaluate_outer_test_once(counting_model, test_X, test_y, torch.device("cpu"))
    checks.extend([
        {
            "check": "outer_test_helper_visits_every_utterance_once",
            "pass": len(logits) == len(predictions) == len(test_X),
        },
        {
            "check": "outer_test_helper_is_one_loader_pass",
            "pass": counting_model.forward_calls == math.ceil(len(test_X) / BATCH_SIZE),
        },
        {
            "check": "attempt_unit_contains_one_outer_test_call_site",
            "pass": inspect.getsource(attempt_unit).count("evaluate_outer_test_once(") == 1,
        },
    ])
    failed = [item["check"] for item in checks if not item["pass"]]
    result = {
        "created_at": now_iso(),
        "status": "pass" if not failed else "fail",
        "synthetic_only": True,
        "no_p1_metrics": True,
        "n_checks": len(checks),
        "failed": failed,
        "checks": checks,
        "frozen_inputs": frozen,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "core_sha256": sha256_file(CORE_PATH),
    }
    atomic_write_json(RESULT_ROOT / "smoke_test.json", result)
    if failed:
        raise RuntimeError(f"synthetic smoke failed: {failed}")
    print(canonical_json({"status": "pass", "checks": len(checks), "synthetic_only": True}))


def unit_dir(row: dict[str, str]) -> Path:
    return (
        RESULT_ROOT / "units" / row["corpus"] / row["model"] / row["protocol"]
        / f"seed_{int(row['seed'])}" / f"fold_{int(row['outer_fold']):03d}"
    )


def final_unit_status(path: Path) -> str | None:
    run_path = path / "run.json"
    if not run_path.exists():
        return None
    try:
        return str(json.loads(run_path.read_text(encoding="utf-8"))["status"])
    except Exception:
        return None


def load_manifest_rows(corpus: str) -> list[dict[str, str]]:
    return read_csv(RESULT_ROOT / "manifests" / f"{corpus}_manifest.csv")


def attempt_unit(
    row: dict[str, str], X: np.ndarray, manifest_rows: list[dict[str, str]],
    device: torch.device, attempt: int, epoch_callback=None, phase_callback=None,
) -> dict:
    ensure_ser_symbols_loaded()
    corpus = row["corpus"]
    model_name = row["model"]
    protocol = row["protocol"]
    seed = int(row["seed"])
    fold = int(row["outer_fold"])
    split_path = RESULT_ROOT / row["inner_split_path"]
    if sha256_file(split_path) != row["inner_split_sha256"]:
        raise RuntimeError(f"inner split hash drift: {split_path}")
    with np.load(split_path, allow_pickle=False) as split:
        fit_idx = np.asarray(split["fit_idx"], dtype=np.int64)
        val_idx = np.asarray(split["val_idx"], dtype=np.int64)
        test_idx = np.asarray(split["outer_test_idx"], dtype=np.int64)
    y_all = np.asarray([int(item["label_index"]) for item in manifest_rows], dtype=np.int64)
    train_seed = stable_u32(f"P1-train|{corpus}|{protocol}|{fold}|{seed}")
    set_reproducible_seed(train_seed)
    config = dict(EXPECTED_MODEL_CONFIGS[model_name])
    model = BUILD_MODEL(model_name, 64, len(CORPORA[corpus]["labels"]), config)
    parameter_count = PARAMETER_COUNT(model)
    unit_config = {
        "unit_id": row["unit_id"],
        "sequence": int(row["sequence"]),
        "corpus": corpus,
        "model": model_name,
        "protocol": protocol,
        "seed": seed,
        "outer_fold": fold,
        "attempt": attempt,
        "train_seed_key": f"P1-train|{corpus}|{protocol}|{fold}|{seed}",
        "train_seed_u32": train_seed,
        "model_config": config,
        "parameter_count": parameter_count,
        "n_classes": len(CORPORA[corpus]["labels"]),
        "label_order": list(CORPORA[corpus]["labels"]),
        "n_fit": len(fit_idx),
        "n_validation": len(val_idx),
        "n_outer_test": len(test_idx),
        "manifest_sha256": row["manifest_sha256"],
        "cache_sha256": row["cache_sha256"],
        "outer_fold_sha256": row["outer_fold_sha256"],
        "inner_split_path": row["inner_split_path"],
        "inner_split_sha256": row["inner_split_sha256"],
        "runner_sha256": sha256_file(RUNNER_PATH),
        "core_sha256": sha256_file(CORE_PATH),
        "run_plan_sha256": sha256_file(RESULT_ROOT / "run_plan.csv"),
        "environment_sha256": sha256_file(RESULT_ROOT / "environment.json"),
        "ser_commit": SER_COMMIT,
        "ser_file_sha256": SER_FILE_SHA256,
        "preprocess_config": PREPROCESS_CONFIG,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": MAX_EPOCHS,
        "class_balanced_cross_entropy": True,
        "class_weights_source": "fit subset only",
        "validation_loss_aggregation": "sum(weighted per-sample CE) / sum(target class weights)",
        "augmentation_scope": "fit subset only; online each epoch",
        "validation_augmentation": False,
        "outer_test_augmentation": False,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
        "checkpoint_rule": "strict minimum validation loss",
        "outer_test_evaluation_policy": "exactly once after checkpoint restoration",
        "mixed_precision": False,
    }
    model, history, fit_summary = train_one_fit(
        model,
        X[fit_idx],
        y_all[fit_idx],
        X[val_idx],
        y_all[val_idx],
        config,
        train_seed,
        len(CORPORA[corpus]["labels"]),
        device,
        epoch_callback,
    )
    if phase_callback is not None:
        phase_callback("outer_test_running")
    logits, predictions = evaluate_outer_test_once(model, X[test_idx], y_all[test_idx], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if phase_callback is not None:
        phase_callback("outer_test_complete")
    return {
        "config": unit_config,
        "history": history,
        "test_idx": test_idx,
        "y_true": y_all[test_idx],
        "predictions": predictions,
        "logits": logits,
        "fit_summary": fit_summary,
    }


def write_success_unit(row, result, previous_failures, manifest_rows) -> None:
    final = unit_dir(row)
    if final.exists():
        raise RuntimeError(f"refusing to overwrite existing unit directory: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.parent / f".{final.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        atomic_write_json(stage / "config.json", result["config"])
        history_fields = [
            "epoch", "train_loss", "validation_loss", "validation_uar",
            "validation_accuracy", "validation_macro_f1", "learning_rate",
        ]
        history_rows = [
            {key: (str(item[key]) if key == "epoch" else format(float(item[key]), ".17g")) for key in history_fields}
            for item in result["history"]
        ]
        atomic_write_csv(stage / "history.csv", history_fields, history_rows)
        labels = CORPORA[row["corpus"]]["labels"]
        prediction_fields = [
            "unit_id", "corpus", "model", "protocol", "seed", "outer_fold",
            "sample_index", "relative_path", "speaker_id", "true_label_index",
            "true_label_name", "predicted_label_index", "predicted_label_name",
        ] + [f"logit_{label}" for label in labels]
        prediction_rows = []
        for local, sample_index in enumerate(result["test_idx"].tolist()):
            true_index = int(result["y_true"][local])
            pred_index = int(result["predictions"][local])
            item = {
                "unit_id": row["unit_id"], "corpus": row["corpus"],
                "model": row["model"], "protocol": row["protocol"],
                "seed": row["seed"], "outer_fold": row["outer_fold"],
                "sample_index": str(sample_index),
                "relative_path": manifest_rows[sample_index]["relative_path"],
                "speaker_id": manifest_rows[sample_index]["speaker_id"],
                "true_label_index": str(true_index), "true_label_name": labels[true_index],
                "predicted_label_index": str(pred_index), "predicted_label_name": labels[pred_index],
            }
            for label_index, label in enumerate(labels):
                item[f"logit_{label}"] = format(float(result["logits"][local, label_index]), ".17g")
            prediction_rows.append(item)
        atomic_write_csv(stage / "predictions.csv", prediction_fields, prediction_rows)
        if previous_failures:
            atomic_write_json(stage / "previous_attempt_failures.json", previous_failures)
        run_record = {
            "unit_id": row["unit_id"],
            "status": "success",
            "completed_at": now_iso(),
            **result["fit_summary"],
            "attempts": 1 + len(previous_failures),
            "failed_attempts": len(previous_failures),
            "failed_attempt_seconds": sum(float(item["elapsed_seconds"]) for item in previous_failures),
            "gpu_seconds_including_failed_attempts": float(result["fit_summary"]["gpu_active_seconds"]) + sum(float(item["elapsed_seconds"]) for item in previous_failures),
            "outer_test_evaluation_passes": 1,
            "n_predictions": len(prediction_rows),
            "config_sha256": sha256_file(stage / "config.json"),
            "history_sha256": sha256_file(stage / "history.csv"),
            "predictions_sha256": sha256_file(stage / "predictions.csv"),
        }
        atomic_write_json(stage / "run.json", run_record)
        os.replace(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def write_failed_unit(row, failures) -> None:
    final = unit_dir(row)
    if final.exists():
        raise RuntimeError(f"refusing to overwrite existing unit directory: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.parent / f".{final.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        atomic_write_json(stage / "failures.json", failures)
        atomic_write_json(stage / "config.json", {
            "unit_id": row["unit_id"], "plan_row": row,
            "runner_sha256": sha256_file(RUNNER_PATH), "core_sha256": sha256_file(CORE_PATH),
        })
        atomic_write_json(stage / "run.json", {
            "unit_id": row["unit_id"], "status": "failed_after_retry",
            "completed_at": now_iso(), "attempts": len(failures),
            "failed_attempts": len(failures),
            "gpu_seconds_including_failed_attempts": sum(float(item["elapsed_seconds"]) for item in failures),
            "outer_test_evaluation_passes": 0,
        })
        os.replace(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class BudgetLimitReached(RuntimeError):
    def __init__(self, state: dict):
        super().__init__("P1 compute budget reached")
        self.state = state


def ledger_dir() -> Path:
    return RESULT_ROOT / "attempt_ledger"


def read_ledger_records(unit_id: str | None = None) -> list[tuple[Path, dict]]:
    root = ledger_dir()
    if not root.exists():
        return []
    records = []
    pattern = f"{unit_id}__*.json" if unit_id else "*.json"
    for path in sorted(root.glob(pattern)):
        records.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return records


def begin_attempt_ledger(row: dict[str, str], attempt: int) -> tuple[Path, float]:
    path = ledger_dir() / f"{row['unit_id']}__attempt-{attempt}__{uuid.uuid4().hex}.json"
    record = {
        "unit_id": row["unit_id"],
        "sequence": int(row["sequence"]),
        "attempt": attempt,
        "status": "running",
        "phase": "training",
        "started_at": now_iso(),
        "last_heartbeat_at": now_iso(),
        "charged_seconds": 0.0,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "core_sha256": sha256_file(CORE_PATH),
    }
    atomic_write_json(path, record)
    return path, time.perf_counter()


def update_attempt_ledger(path: Path, **updates) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(updates)
    record["last_heartbeat_at"] = now_iso()
    atomic_write_json(path, record)
    return record


def reconcile_stale_ledgers() -> None:
    now = datetime.now().astimezone()
    for path, record in read_ledger_records():
        if record.get("status") != "running":
            continue
        started = datetime.fromisoformat(record["started_at"])
        charged = max(float(record.get("charged_seconds", 0.0)), (now - started).total_seconds())
        phase = str(record.get("phase", "unknown"))
        status = "interrupted_before_outer_test" if phase == "training" else "interrupted_after_outer_test_access"
        update_attempt_ledger(
            path,
            status=status,
            ended_at=now_iso(),
            charged_seconds=charged,
            note="stale running ledger reconciled conservatively at next runner start",
        )


def completed_gpu_seconds() -> float:
    return sum(float(record.get("charged_seconds", 0.0)) for _, record in read_ledger_records())


def formal_start() -> datetime:
    path = RESULT_ROOT / "formal_start.json"
    if not path.exists():
        atomic_write_json(path, {
            "started_at": now_iso(),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "core_sha256": sha256_file(CORE_PATH),
            "no_metrics_seen_before_lock": True,
        })
    return datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["started_at"])


def budget_gate(
    started_at: datetime,
    scan_output: bool = True,
    gpu_seconds_override: float | None = None,
) -> tuple[bool, dict]:
    state = {
        "checked_at": now_iso(),
        "gpu_seconds": completed_gpu_seconds() if gpu_seconds_override is None else float(gpu_seconds_override),
        "wall_seconds": (datetime.now().astimezone() - started_at).total_seconds(),
        "output_bytes": output_size_bytes() if scan_output else None,
        "gpu_seconds_limit": GPU_SECONDS_LIMIT,
        "wall_seconds_limit": WALL_DAYS_LIMIT * 86400,
        "output_bytes_limit": OUTPUT_BYTES_LIMIT,
    }
    reasons = []
    if state["gpu_seconds"] >= GPU_SECONDS_LIMIT:
        reasons.append("gpu_hours_limit")
    if state["wall_seconds"] >= WALL_DAYS_LIMIT * 86400:
        reasons.append("wall_days_limit")
    if scan_output and state["output_bytes"] >= OUTPUT_BYTES_LIMIT:
        reasons.append("output_size_limit")
    state["reasons"] = reasons
    return not reasons, state


@contextlib.contextmanager
def exclusive_run_lock(corpus: str):
    path = RESULT_ROOT / ".formal_run.lock"
    token = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another formal runner may be active; lock exists: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"token": token, "pid": os.getpid(), "corpus": corpus, "created_at": now_iso()}, handle)
            handle.write("\n")
        yield
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                path.unlink()
        except FileNotFoundError:
            pass


def run(corpus: str) -> None:
    with exclusive_run_lock(corpus):
        run_locked(corpus)


def run_locked(corpus: str) -> None:
    assert_frozen_inputs()
    verify_model_configs()
    ensure_ser_symbols_loaded()
    prepared = load_prepared()
    smoke_path = RESULT_ROOT / "smoke_test.json"
    smoke_record = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else {}
    if (
        smoke_record.get("status") != "pass"
        or smoke_record.get("runner_sha256") != sha256_file(RUNNER_PATH)
        or smoke_record.get("core_sha256") != sha256_file(CORE_PATH)
    ):
        raise RuntimeError("synthetic smoke gate has not passed")
    if not torch.cuda.is_available() or "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError("CUDA/sm_120 gate failed")
    if sha256_file(RUNNER_PATH) != prepared["tool_sha256"][RUNNER_PATH.name] or sha256_file(CORE_PATH) != prepared["tool_sha256"][CORE_PATH.name]:
        raise RuntimeError("formal runner/core changed after prepare; re-freeze before training")
    device = torch.device("cuda:0")
    rows = [row for row in read_csv(RESULT_ROOT / "run_plan.csv") if row["corpus"] == corpus]
    X = load_cache(corpus, prepared["corpora"][corpus]["cache"]["cache_sha256"])
    manifest_path = RESULT_ROOT / "manifests" / f"{corpus}_manifest.csv"
    expected_manifest_sha = prepared["corpora"][corpus]["manifest"]["manifest_sha256"]
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise RuntimeError(f"manifest hash drift: {manifest_path}")
    manifest_rows = load_manifest_rows(corpus)
    validate_manifest(corpus, manifest_path, manifest_rows)
    started_at = formal_start()
    reconcile_stale_ledgers()
    processed = 0
    for row in rows:
        existing = final_unit_status(unit_dir(row))
        if existing in {"success", "failed_after_retry"}:
            continue
        prior_ledgers = [record for _, record in read_ledger_records(row["unit_id"])]
        if any(record.get("status") in {
            "compute_success", "success_uncommitted", "interrupted_after_outer_test_access"
        } for record in prior_ledgers):
            raise RuntimeError(
                f"test-once gate: {row['unit_id']} has prior outer-test access without committed artifacts"
            )
        allowed, budget = budget_gate(started_at)
        if not allowed:
            atomic_write_json(RESULT_ROOT / "budget_stop.json", budget)
            print(canonical_json({"status": "budget_stop", **budget}), flush=True)
            break
        failures = [
            {
                "attempt": int(record["attempt"]),
                "failed_at": record.get("ended_at", record.get("last_heartbeat_at", "")),
                "elapsed_seconds": float(record.get("charged_seconds", 0.0)),
                "exception_type": record.get("exception_type", ""),
                "exception": record.get("exception", ""),
                "traceback": record.get("traceback", ""),
                "ledger_path": record.get("ledger_path", ""),
            }
            for record in prior_ledgers if record.get("status") == "failed_model_attempt"
        ]
        if len(failures) >= 2:
            write_failed_unit(row, failures)
            processed += 1
            continue
        success = False
        for attempt in range(len(failures) + 1, 3):
            prior_gpu_seconds = completed_gpu_seconds()
            ledger_path, began = begin_attempt_ledger(row, attempt)

            def heartbeat(epoch, _train_elapsed):
                charged = time.perf_counter() - began
                update_attempt_ledger(
                    ledger_path, phase="training", epoch=int(epoch), charged_seconds=charged
                )
                heartbeat_allowed, heartbeat_budget = budget_gate(
                    started_at,
                    scan_output=False,
                    gpu_seconds_override=prior_gpu_seconds + charged,
                )
                if not heartbeat_allowed:
                    raise BudgetLimitReached(heartbeat_budget)

            def phase_update(phase):
                update_attempt_ledger(
                    ledger_path, phase=phase, charged_seconds=time.perf_counter() - began
                )

            try:
                result = attempt_unit(
                    row, X, manifest_rows, device, attempt,
                    epoch_callback=heartbeat, phase_callback=phase_update,
                )
                charged = time.perf_counter() - began
                result["fit_summary"]["gpu_active_seconds"] = charged
                update_attempt_ledger(
                    ledger_path,
                    status="compute_success",
                    phase="commit",
                    ended_at=now_iso(),
                    charged_seconds=charged,
                )
                commit_errors = []
                for commit_attempt in (1, 2):
                    try:
                        write_success_unit(row, result, failures, manifest_rows)
                        commit_errors = []
                        break
                    except BaseException as commit_exc:
                        commit_errors.append({
                            "commit_attempt": commit_attempt,
                            "exception_type": type(commit_exc).__name__,
                            "exception": str(commit_exc),
                            "traceback": traceback.format_exc(),
                        })
                if commit_errors:
                    update_attempt_ledger(
                        ledger_path,
                        status="success_uncommitted",
                        commit_errors=commit_errors,
                    )
                    raise RuntimeError(
                        f"artifact commit failed after outer-test access for {row['unit_id']}; refusing retrain"
                    )
                update_attempt_ledger(
                    ledger_path,
                    status="success_committed",
                    phase="complete",
                    committed_at=now_iso(),
                    charged_seconds=charged,
                )
                success = True
                processed += 1
                print(canonical_json({
                    "status": "success", "sequence": int(row["sequence"]),
                    "unit_id": row["unit_id"], "attempt": attempt,
                    "best_epoch": result["fit_summary"]["best_epoch"],
                    "epochs_run": result["fit_summary"]["epochs_run"],
                    "train_seconds": round(result["fit_summary"]["train_seconds"], 3),
                    "completed_in_this_process": processed,
                }), flush=True)
                break
            except BudgetLimitReached as exc:
                charged = time.perf_counter() - began
                update_attempt_ledger(
                    ledger_path,
                    status="budget_stopped_during_training",
                    phase="training",
                    ended_at=now_iso(),
                    charged_seconds=charged,
                )
                atomic_write_json(RESULT_ROOT / "budget_stop.json", exc.state)
                print(canonical_json({"status": "budget_stop", **exc.state}), flush=True)
                write_status()
                return
            except KeyboardInterrupt:
                charged = time.perf_counter() - began
                current = json.loads(ledger_path.read_text(encoding="utf-8"))
                phase = str(current.get("phase", "training"))
                update_attempt_ledger(
                    ledger_path,
                    status=("interrupted_before_outer_test" if phase == "training" else "interrupted_after_outer_test_access"),
                    ended_at=now_iso(),
                    charged_seconds=charged,
                )
                raise
            except BaseException as exc:
                current = json.loads(ledger_path.read_text(encoding="utf-8"))
                if current.get("status") in {"compute_success", "success_uncommitted"}:
                    raise
                elapsed = time.perf_counter() - began
                current_phase = str(current.get("phase", "training"))
                if current_phase != "training":
                    update_attempt_ledger(
                        ledger_path,
                        status="interrupted_after_outer_test_access",
                        phase=current_phase,
                        ended_at=now_iso(),
                        charged_seconds=elapsed,
                        exception_type=type(exc).__name__,
                        exception=str(exc),
                        traceback=traceback.format_exc(),
                    )
                    raise RuntimeError(
                        f"outer-test access failed for {row['unit_id']}; test-once gate forbids retry"
                    ) from exc
                failure = {
                    "attempt": attempt,
                    "failed_at": now_iso(),
                    "elapsed_seconds": elapsed,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "ledger_path": ledger_path.relative_to(RESULT_ROOT).as_posix(),
                }
                failures.append(failure)
                update_attempt_ledger(
                    ledger_path,
                    status="failed_model_attempt",
                    phase=current_phase,
                    ended_at=now_iso(),
                    charged_seconds=elapsed,
                    exception_type=failure["exception_type"],
                    exception=failure["exception"],
                    traceback=failure["traceback"],
                    ledger_path=failure["ledger_path"],
                )
                print(canonical_json({
                    "status": "attempt_failed", "unit_id": row["unit_id"],
                    "attempt": attempt, "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if not success:
            write_failed_unit(row, failures)
            processed += 1
            print(canonical_json({"status": "failed_after_retry", "unit_id": row["unit_id"]}), flush=True)
        if processed % 10 == 0:
            write_status()
    write_status()


def write_status() -> dict:
    plan = read_csv(RESULT_ROOT / "run_plan.csv") if (RESULT_ROOT / "run_plan.csv").exists() else []
    rows = []
    counts = {"success": 0, "failed_after_retry": 0, "pending": 0}
    for row in plan:
        status = final_unit_status(unit_dir(row)) or "pending"
        counts[status] = counts.get(status, 0) + 1
        rows.append({
            "sequence": row["sequence"], "unit_id": row["unit_id"],
            "corpus": row["corpus"], "model": row["model"],
            "protocol": row["protocol"], "seed": row["seed"],
            "outer_fold": row["outer_fold"], "status": status,
        })
    if plan:
        atomic_write_csv(RESULT_ROOT / "run_status.csv", [
            "sequence", "unit_id", "corpus", "model", "protocol", "seed", "outer_fold", "status",
        ], rows)
    summary = {
        "created_at": now_iso(),
        "planned": len(plan),
        **counts,
        "gpu_seconds_recorded": completed_gpu_seconds(),
        "output_bytes": output_size_bytes(),
    }
    atomic_write_json(RESULT_ROOT / "run_status_summary.json", summary)
    print(canonical_json(summary), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("refreeze-engineering")
    sub.add_parser("smoke")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--corpus", required=True, choices=("ravdess", "cremad"))
    sub.add_parser("status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "refreeze-engineering":
        refreeze_engineering()
    elif args.command == "smoke":
        smoke()
    elif args.command == "run":
        run(args.corpus)
    elif args.command == "status":
        write_status()


if __name__ == "__main__":
    main()

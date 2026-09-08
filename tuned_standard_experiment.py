"""Nested speaker-independent tuning for traditional and neural SER models.

This experiment compares tuned SVM, CNN, FNO, residual SE attention, and a
small Transformer under the same outer GroupKFold speaker splits.  All model
selection happens inside each outer training fold.  The outer test fold is
evaluated once per final fit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from advanced_experiment_utils import (
    aggregate_by_fold,
    classification_metrics,
    config_key,
    evaluate_neural,
    paired_comparison,
    set_reproducible_seed,
    train_fixed_epochs,
    train_with_validation,
)
from advanced_models import build_neural_model, parameter_count
from dataset import build_dataset, list_audio
from evaluate import per_speaker_normalize
from experiment_utils import resolve_device, write_csv, write_json
from fno_data import N_MELS, build_mel_dataset, normalize_spectrograms


MODEL_LABELS = {
    "fno": "FNO_tuned",
    "cnn": "CNN_tuned",
    "resnet_se": "ResNet1D_SE",
    "transformer": "Transformer_tuned",
}


def parse_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_models(text):
    if text == "all":
        return ["svm", "fno", "cnn", "resnet_se", "transformer"]
    models = [item.strip() for item in text.split(",") if item.strip()]
    allowed = set(MODEL_LABELS) | {"svm"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    return models


def _train_options(
    *,
    lr=1e-3,
    weight_decay=1e-4,
    augment=True,
    balanced_loss=True,
    time_mask=16,
    freq_mask=8,
    noise_std=0.1,
):
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "augment": augment,
        "balanced_loss": balanced_loss,
        "time_mask": time_mask,
        "freq_mask": freq_mask,
        "noise_std": noise_std,
    }


def candidate_configs(model_name):
    common = _train_options()
    no_aug = _train_options(augment=False, noise_std=0.0)
    if model_name == "fno":
        return [
            {"width": 32, "modes": 16, "n_layers": 4, "dropout": 0.1, **no_aug},
            {"width": 32, "modes": 16, "n_layers": 4, "dropout": 0.1, **common},
            {"width": 48, "modes": 16, "n_layers": 4, "dropout": 0.1, **common},
            {"width": 32, "modes": 8, "n_layers": 4, "dropout": 0.1, **common},
            {"width": 32, "modes": 24, "n_layers": 4, "dropout": 0.2, **common},
            {
                "width": 48,
                "modes": 16,
                "n_layers": 6,
                "dropout": 0.2,
                **_train_options(lr=5e-4),
            },
        ]
    if model_name == "cnn":
        return [
            {
                "width": 32,
                "n_blocks": 3,
                "kernel_size": 5,
                "dilation_base": 1,
                "dropout": 0.1,
                **no_aug,
            },
            {
                "width": 32,
                "n_blocks": 3,
                "kernel_size": 5,
                "dilation_base": 1,
                "dropout": 0.1,
                **common,
            },
            {
                "width": 48,
                "n_blocks": 3,
                "kernel_size": 5,
                "dilation_base": 1,
                "dropout": 0.1,
                **common,
            },
            {
                "width": 64,
                "n_blocks": 4,
                "kernel_size": 3,
                "dilation_base": 1,
                "dropout": 0.2,
                **common,
            },
            {
                "width": 48,
                "n_blocks": 4,
                "kernel_size": 7,
                "dilation_base": 1,
                "dropout": 0.2,
                **_train_options(lr=5e-4),
            },
            {
                "width": 48,
                "n_blocks": 3,
                "kernel_size": 5,
                "dilation_base": 2,
                "dropout": 0.2,
                **_train_options(lr=5e-4),
            },
        ]
    if model_name == "resnet_se":
        return [
            {
                "width": 32,
                "n_blocks": 4,
                "kernel_size": 5,
                "max_dilation": 4,
                "dropout": 0.1,
                "se_reduction": 4,
                **no_aug,
            },
            {
                "width": 32,
                "n_blocks": 4,
                "kernel_size": 5,
                "max_dilation": 4,
                "dropout": 0.1,
                "se_reduction": 4,
                **common,
            },
            {
                "width": 48,
                "n_blocks": 4,
                "kernel_size": 5,
                "max_dilation": 4,
                "dropout": 0.1,
                "se_reduction": 4,
                **common,
            },
            {
                "width": 32,
                "n_blocks": 6,
                "kernel_size": 5,
                "max_dilation": 4,
                "dropout": 0.2,
                "se_reduction": 4,
                **_train_options(lr=5e-4),
            },
            {
                "width": 48,
                "n_blocks": 4,
                "kernel_size": 3,
                "max_dilation": 4,
                "dropout": 0.2,
                "se_reduction": 8,
                **common,
            },
            {
                "width": 48,
                "n_blocks": 4,
                "kernel_size": 7,
                "max_dilation": 2,
                "dropout": 0.2,
                "se_reduction": 4,
                **_train_options(lr=5e-4),
            },
        ]
    if model_name == "transformer":
        return [
            {
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "ff_mult": 2.0,
                "dropout": 0.1,
                "position_mode": "normalized",
                "pooling": "attentive",
                **_train_options(lr=5e-4, augment=False, noise_std=0.0),
            },
            {
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "ff_mult": 2.0,
                "dropout": 0.1,
                "position_mode": "normalized",
                "pooling": "attentive",
                **_train_options(lr=5e-4),
            },
            {
                "d_model": 96,
                "n_heads": 4,
                "n_layers": 2,
                "ff_mult": 2.0,
                "dropout": 0.1,
                "position_mode": "normalized",
                "pooling": "attentive",
                **_train_options(lr=3e-4),
            },
            {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 3,
                "ff_mult": 2.0,
                "dropout": 0.2,
                "position_mode": "normalized",
                "pooling": "attentive",
                **_train_options(lr=3e-4),
            },
            {
                "d_model": 96,
                "n_heads": 4,
                "n_layers": 3,
                "ff_mult": 4.0,
                "dropout": 0.2,
                "position_mode": "normalized",
                "pooling": "mean",
                **_train_options(lr=3e-4),
            },
            {
                "d_model": 96,
                "n_heads": 4,
                "n_layers": 2,
                "ff_mult": 2.0,
                "dropout": 0.1,
                "position_mode": "index",
                "pooling": "attentive",
                **_train_options(lr=3e-4),
            },
        ]
    raise ValueError(f"unknown model: {model_name}")


def validate_alignment(y_feat, groups_feat, y_mel, groups_mel):
    if not np.array_equal(y_feat, y_mel):
        raise RuntimeError("feature and mel label orders differ; rebuild caches")
    if not np.array_equal(groups_feat, groups_mel):
        raise RuntimeError("feature and mel speaker orders differ; rebuild caches")


def sample_ids_for_dataset(data_dir, scheme, expected_count):
    items = list_audio(data_dir, scheme)
    if len(items) != expected_count:
        return [f"sample_{index:05d}" for index in range(expected_count)]
    return [str(path) for path, _, _ in items]


def svm_estimator():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf")),
        ]
    )


def tune_svm(
    X,
    y,
    groups,
    train_idx,
    inner_folds,
    n_jobs,
):
    n_splits = min(inner_folds, len(np.unique(groups[train_idx])))
    cv = GroupKFold(n_splits=n_splits)
    grid = {
        "clf__C": [0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", 1e-3, 1e-2, 1e-1],
        "clf__class_weight": [None, "balanced"],
    }
    search = GridSearchCV(
        svm_estimator(),
        grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(
        X[train_idx],
        y[train_idx],
        groups=groups[train_idx],
    )
    return search


def tune_neural_config(
    model_name,
    configs,
    X,
    y,
    groups,
    train_idx,
    n_classes,
    outer_fold,
    args,
    device,
):
    local_groups = groups[train_idx]
    n_splits = min(args.inner_folds, len(np.unique(local_groups)))
    inner = GroupKFold(n_splits=n_splits)
    tuning_rows = []
    candidates = []
    for config_id, config in enumerate(configs[: args.search_budget]):
        fold_metrics = []
        print(
            f"    {model_name} config {config_id + 1}/"
            f"{min(len(configs), args.search_budget)}"
        )
        for inner_fold, (fit_local, val_local) in enumerate(
            inner.split(X[train_idx], y[train_idx], local_groups),
            start=1,
        ):
            fit_idx = train_idx[fit_local]
            val_idx = train_idx[val_local]
            run_seed = (
                args.tune_seed
                + outer_fold * 100000
                + config_id * 1000
                + inner_fold
            )
            set_reproducible_seed(run_seed)
            model = build_neural_model(model_name, N_MELS, n_classes, config)
            model, info = train_with_validation(
                model,
                X[fit_idx],
                y[fit_idx],
                X[val_idx],
                y[val_idx],
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=device,
                config=config,
                batch_size=args.batch_size,
                seed=run_seed,
            )
            record = {
                "model": model_name,
                "outer_fold": outer_fold,
                "config_id": config_id,
                "inner_fold": inner_fold,
                "config_json": config_key(config),
                "parameter_count": parameter_count(model),
                "best_epoch": info["best_epoch"],
                "epochs_run": info["epochs_run"],
                "val_loss": info["best_val_loss"],
                "val_balanced_accuracy": info["best_val_balanced_accuracy"],
                "val_macro_f1": info["best_val_macro_f1"],
                "train_seconds": info["train_seconds"],
                "capped": int(info["best_epoch"] == args.max_epochs),
            }
            tuning_rows.append(record)
            fold_metrics.append(record)

        candidate = {
            "config_id": config_id,
            "config": config,
            "mean_val_balanced_accuracy": float(
                np.mean([row["val_balanced_accuracy"] for row in fold_metrics])
            ),
            "mean_val_macro_f1": float(
                np.mean([row["val_macro_f1"] for row in fold_metrics])
            ),
            "mean_val_loss": float(np.mean([row["val_loss"] for row in fold_metrics])),
            "selected_epochs": max(
                1,
                int(round(np.median([row["best_epoch"] for row in fold_metrics]))),
            ),
            "parameter_count": int(fold_metrics[0]["parameter_count"]),
        }
        candidates.append(candidate)

    selected = max(
        candidates,
        key=lambda item: (
            item["mean_val_balanced_accuracy"],
            item["mean_val_macro_f1"],
            -item["mean_val_loss"],
        ),
    )
    for row in tuning_rows:
        row["selected"] = int(row["config_id"] == selected["config_id"])
    return selected, tuning_rows, candidates


def append_predictions(
    rows,
    model,
    fold,
    seed,
    indices,
    y_true,
    y_pred,
    groups,
    sample_ids,
    label_encoder,
):
    for position, sample_index in enumerate(indices):
        rows.append(
            {
                "model": model,
                "fold": fold,
                "seed": seed,
                "sample_index": int(sample_index),
                "sample_id": sample_ids[sample_index],
                "speaker": str(groups[sample_index]),
                "true_index": int(y_true[position]),
                "pred_index": int(y_pred[position]),
                "true_label": str(label_encoder.inverse_transform([y_true[position]])[0]),
                "pred_label": str(label_encoder.inverse_transform([y_pred[position]])[0]),
                "correct": int(y_true[position] == y_pred[position]),
            }
        )


def add_pairwise_comparisons(summary, fold_values):
    reference = MODEL_LABELS["fno"]
    if reference not in fold_values:
        return {}
    comparisons = {}
    reference_folds = sorted(int(fold) for fold in fold_values[reference])
    for model, values in fold_values.items():
        if model == reference:
            continue
        model_folds = sorted(int(fold) for fold in values)
        shared = sorted(set(reference_folds) & set(model_folds))
        comparisons[f"{model}_minus_{reference}"] = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            a = [fold_values[reference][str(fold)][metric] for fold in shared]
            b = [fold_values[model][str(fold)][metric] for fold in shared]
            if len(shared) >= 2:
                comparisons[f"{model}_minus_{reference}"][metric] = paired_comparison(a, b)
    summary["pairwise_against_fno"] = comparisons
    return comparisons


def ensure_outputs_available(out_dir: Path, overwrite: bool):
    expected = [
        out_dir / "tuned_standard_scores.csv",
        out_dir / "tuned_standard_tuning.csv",
        out_dir / "tuned_standard_predictions.csv",
        out_dir / "tuned_standard_summary.json",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing outputs: {names}; use --overwrite"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Nested speaker-independent tuning for SVM and neural SER models."
    )
    parser.add_argument("data_dir")
    parser.add_argument("--scheme", default="ravdess", choices=["ravdess", "folder"])
    parser.add_argument("--feature-cache", default="features_cache.npz")
    parser.add_argument("--mel-cache", default="mel_cache.npz")
    parser.add_argument("--models", default="all")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--search-budget", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--norm", default="speaker", choices=["sample", "speaker"])
    parser.add_argument("--tune-seed", type=int, default=20260727)
    parser.add_argument("--svm-jobs", type=int, default=-1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out-dir", default="results/tuned_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    models = parse_models(args.models)
    seeds = parse_ints(args.seeds)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_outputs_available(out_dir, args.overwrite)

    feature_cache = args.feature_cache or None
    mel_cache = args.mel_cache or None
    X_feat, y_feat, groups_feat, _ = build_dataset(
        args.data_dir, args.scheme, feature_cache
    )
    X_mel, y_mel, groups_mel = build_mel_dataset(
        args.data_dir, args.scheme, mel_cache
    )
    validate_alignment(y_feat, groups_feat, y_mel, groups_mel)
    X_feat_norm = per_speaker_normalize(X_feat, groups_feat)
    X_mel = normalize_spectrograms(X_mel, groups_mel, mode=args.norm)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_feat)
    groups = np.asarray(groups_feat)
    sample_ids = sample_ids_for_dataset(
        args.data_dir, args.scheme, len(y_encoded)
    )
    n_classes = len(encoder.classes_)
    n_outer = min(args.outer_folds, len(np.unique(groups)))
    outer = GroupKFold(n_splits=n_outer)

    print(f"device: {device}")
    print(f"models: {models}")
    print(
        f"{len(y_encoded)} samples | {len(np.unique(groups))} speakers | "
        f"{n_classes} classes"
    )
    print(
        f"outer folds={n_outer} | inner folds={args.inner_folds} | "
        f"neural seeds={seeds} | search budget={args.search_budget}"
    )

    score_rows = []
    tuning_rows = []
    prediction_rows = []
    selection = {}

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer.split(X_feat, y_encoded, groups),
        start=1,
    ):
        print(
            f"\n{'=' * 72}\nouter fold {outer_fold}/{n_outer}: "
            f"train={len(train_idx)} test={len(test_idx)}\n{'=' * 72}"
        )
        selection[str(outer_fold)] = {}

        if "svm" in models:
            for model_label, X_variant in (
                ("SVM_raw_tuned", X_feat),
                ("SVM_speaker_norm_tuned", X_feat_norm),
            ):
                print(f"  tuning {model_label}")
                search = tune_svm(
                    X_variant,
                    y_encoded,
                    groups,
                    train_idx,
                    args.inner_folds,
                    args.svm_jobs,
                )
                estimator = clone(search.best_estimator_).fit(
                    X_variant[train_idx], y_encoded[train_idx]
                )
                pred = estimator.predict(X_variant[test_idx])
                metrics = classification_metrics(y_encoded[test_idx], pred)
                score_rows.append(
                    {
                        "model": model_label,
                        "fold": outer_fold,
                        "seed": "",
                        **metrics,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "train_speakers": len(np.unique(groups[train_idx])),
                        "test_speakers": len(np.unique(groups[test_idx])),
                        "selected_epochs": "",
                        "parameter_count": "",
                        "config_json": json.dumps(
                            search.best_params_, sort_keys=True
                        ),
                    }
                )
                selection[str(outer_fold)][model_label] = {
                    "best_params": search.best_params_,
                    "best_inner_balanced_accuracy": float(search.best_score_),
                }
                append_predictions(
                    prediction_rows,
                    model_label,
                    outer_fold,
                    "",
                    test_idx,
                    y_encoded[test_idx],
                    pred,
                    groups,
                    sample_ids,
                    encoder,
                )
                print(
                    f"    UAR={metrics['balanced_accuracy']:.3f} "
                    f"acc={metrics['accuracy']:.3f} "
                    f"params={search.best_params_}"
                )

        for model_name in [model for model in models if model in MODEL_LABELS]:
            print(f"  tuning {MODEL_LABELS[model_name]}")
            selected, model_tuning_rows, candidates = tune_neural_config(
                model_name,
                candidate_configs(model_name),
                X_mel,
                y_encoded,
                groups,
                train_idx,
                n_classes,
                outer_fold,
                args,
                device,
            )
            tuning_rows.extend(model_tuning_rows)
            model_label = MODEL_LABELS[model_name]
            selection[str(outer_fold)][model_label] = {
                **selected,
                "candidates": candidates,
            }
            print(
                f"    selected config {selected['config_id']} | "
                f"inner UAR={selected['mean_val_balanced_accuracy']:.3f} | "
                f"epochs={selected['selected_epochs']} | "
                f"params={selected['parameter_count']:,}"
            )

            for seed in seeds:
                run_seed = (
                    args.tune_seed
                    + outer_fold * 1000000
                    + seed * 10000
                    + selected["config_id"]
                )
                set_reproducible_seed(run_seed)
                model = build_neural_model(
                    model_name, N_MELS, n_classes, selected["config"]
                )
                model, train_info = train_fixed_epochs(
                    model,
                    X_mel[train_idx],
                    y_encoded[train_idx],
                    selected["selected_epochs"],
                    device,
                    selected["config"],
                    batch_size=args.batch_size,
                    seed=run_seed,
                    scheduler_t_max=args.max_epochs,
                )
                evaluation = evaluate_neural(
                    model,
                    X_mel[test_idx],
                    y_encoded[test_idx],
                    device,
                    batch_size=args.batch_size,
                )
                metrics = {
                    key: evaluation[key]
                    for key in ("accuracy", "balanced_accuracy", "macro_f1")
                }
                score_rows.append(
                    {
                        "model": model_label,
                        "fold": outer_fold,
                        "seed": seed,
                        **metrics,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "train_speakers": len(np.unique(groups[train_idx])),
                        "test_speakers": len(np.unique(groups[test_idx])),
                        "selected_epochs": selected["selected_epochs"],
                        "scheduler_t_max": train_info["scheduler_t_max"],
                        "parameter_count": parameter_count(model),
                        "train_seconds": train_info["train_seconds"],
                        "config_json": config_key(selected["config"]),
                    }
                )
                append_predictions(
                    prediction_rows,
                    model_label,
                    outer_fold,
                    seed,
                    test_idx,
                    y_encoded[test_idx],
                    evaluation["predictions"],
                    groups,
                    sample_ids,
                    encoder,
                )
                print(
                    f"    seed {seed}: UAR={metrics['balanced_accuracy']:.3f} "
                    f"acc={metrics['accuracy']:.3f}"
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    aggregate, fold_values = aggregate_by_fold(score_rows)
    summary = {
        "args": vars(args),
        "models": models,
        "classes": [str(label) for label in encoder.classes_],
        "class_counts": {
            str(label): int(count)
            for label, count in Counter(y_feat).items()
        },
        "aggregate_by_fold": aggregate,
        "fold_values": fold_values,
        "selection": selection,
    }
    add_pairwise_comparisons(summary, fold_values)

    write_csv(out_dir / "tuned_standard_scores.csv", score_rows)
    write_csv(out_dir / "tuned_standard_tuning.csv", tuning_rows)
    write_csv(out_dir / "tuned_standard_predictions.csv", prediction_rows)
    write_json(out_dir / "tuned_standard_summary.json", summary)

    print(f"\n{'=' * 72}\nFINAL FOLD-LEVEL SUMMARY\n{'=' * 72}")
    for model, metrics in aggregate.items():
        uar = metrics["balanced_accuracy"]
        accuracy = metrics["accuracy"]
        print(
            f"{model:<26s} UAR {uar['mean']:.3f} +- {uar['sample_sd']:.3f} | "
            f"acc {accuracy['mean']:.3f} +- {accuracy['sample_sd']:.3f}"
        )
    print(f"\nWrote tuned standard results -> {out_dir}")


if __name__ == "__main__":
    main()

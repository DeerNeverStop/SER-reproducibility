"""Machine-readable constants and invariants for N14R.

This module deliberately contains no result-loading or training code.  It is the
single Python authority used by planning, execution, and scoring code.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

STUDY_ID = "N14R"
SPEC_VERSION = "1.0.0"
SEED_NAMESPACE = "SER26-N14R"
CORPUS_LEVEL = "cremad"
MODEL = "resnet_se"
CELLS = ("GR_hpo", "GG_hpo")
N_PRIMARY_DRAWS = 24
N_RESERVE_DRAWS = 4
N_FOLDS = 5
N_TRAIN_REPS = 1
N_CLASSES = 6
PRIMARY_DRAW_IDS = tuple(range(N_PRIMARY_DRAWS))
RESERVE_DRAW_IDS = tuple(range(N_PRIMARY_DRAWS, N_PRIMARY_DRAWS + N_RESERVE_DRAWS))
ALL_DRAW_IDS = PRIMARY_DRAW_IDS + RESERVE_DRAW_IDS
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 202609040001
FIXED_CONFIG_INDEX = 4

# Preserve the exact v2 enumeration order.  In particular, index 4 is the
# fixed comparator (lr=1e-3, weight_decay=1e-4, dropout=0.1).
HPO_GRID = tuple(
    {
        "engine": "p1_frozen",
        "model": MODEL,
        "lr": lr,
        "weight_decay": wd,
        "dropout": dropout,
        "batch_size": 64,
        "epochs": 100,
        "patience": 15,
        "hpo_config_index": i,
    }
    for i, (lr, wd, dropout) in enumerate(
        (lr, wd, dropout)
        for lr in (3e-4, 1e-3)
        for wd in (1e-4, 1e-3)
        for dropout in (0.1, 0.3)
    )
)

PLAN_FIELDS = (
    "study_id", "spec_version", "unit_id", "arm", "manifest_sha256", "draw_id",
    "draw_role", "analysis_order", "fold", "train_rep", "cell",
    "config_index", "model", "corpus_level", "base_corpus", "panel_draw", "r",
    "seed_index", "split_seed", "inner_seed", "train_seed", "outer_sha256",
    "inner_sha256", "split_sha256", "config_sha256", "n_fit", "n_val",
    "n_test", "est_gpu_sec", "cap_group", "conditional", "truncation_rank", "status",
)

SPEC: dict[str, Any] = {
    "study_id": STUDY_ID,
    "spec_version": SPEC_VERSION,
    "seed_namespace": SEED_NAMESPACE,
    "legacy_pilot": {
        "source_commit": "c0c0beb",
        "draws": [0],
        "use": "design_only",
        "excluded_from_confirmatory_analysis": True,
    },
    "execution_environment": {
        "python": "3.13.9",
        "numpy": "2.4.6",
        "scipy": "1.18.0",
        "scikit_learn": "1.9.0",
        "torch": "2.11.0+cu128",
        "torch_cuda": "12.8",
        "cudnn": 91900,
        "librosa": "0.11.0",
        "gpu": "NVIDIA GeForce RTX 5070",
    },
    "design": {
        "corpus_level": CORPUS_LEVEL,
        "model": MODEL,
        "manifest_input_rows": 7442,
        "analysis_population_rows_after_hygiene": 7435,
        "hygiene_rule": "v2.ser_v2.corpora.apply_hygiene('cremad'): H1/H2/H3",
        "cells": list(CELLS),
        "primary_draw_ids": list(PRIMARY_DRAW_IDS),
        "reserve_draw_ids": list(RESERVE_DRAW_IDS),
        "folds_per_draw": N_FOLDS,
        "train_reps": N_TRAIN_REPS,
        "configs": list(HPO_GRID),
        "expected_primary_units": N_PRIMARY_DRAWS * N_FOLDS * len(CELLS) * len(HPO_GRID),
        "maximum_units": len(ALL_DRAW_IDS) * N_FOLDS * len(CELLS) * len(HPO_GRID),
    },
    "estimand": {
        "population": "fixed CREMA-D corpus conditional on the frozen algorithmic randomization distribution",
        "independent_unit": "complete five-fold outer split draw",
        "fold_metric": "whole-fold six-class macro-UAR in percentage points",
        "delta_draw": "mean_fold[((V-T)_GR_hpo)-((V-T)_GG_hpo)]",
        "theta": "mean_delta_over_24_complete_primary_or_replacement_draws",
    },
    "inference": {
        "hypothesis": "theta=0 versus theta!=0",
        "directional_claim_requires": "mean_delta>0 and holm_adjusted_two_sided_p<=0.05",
        "primary_test": "two-sided one-sample Student t test over draw-level deltas",
        "primary_ci": "two-sided 95% Student t confidence interval over draw-level deltas",
        "multiplicity": "Holm family N14R of size 1",
        "bootstrap_sensitivity": {"method": "percentile", "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "sign_flip_sensitivity": "full enumeration of 2^24 sign assignments; symmetry assumption required",
    },
    "power": {
        "model": "noncentral t, two-sided alpha 0.05",
        "alternative_mean_pp": 1.5,
        "sd_pp": 2.5,
        "n_draws": N_PRIMARY_DRAWS,
        "power": 0.804,
    },
    "budget": {
        "primary_first_attempt_projected_gpu_hours": 92.9,
        "all_28_draws_first_attempt_projected_gpu_hours": 108.4,
        "all_28_draws_all_retries_projected_gpu_hours": 216.8,
        "resource_allocation_gpu_hours": 125.0,
        "resource_allocation_is_stopping_rule": False,
        "hard_runtime_cap": None,
    },
    "completion": {
        "required_complete_draws": N_PRIMARY_DRAWS,
        "maximum_activated_draws": len(ALL_DRAW_IDS),
        "units_per_draw": N_FOLDS * len(CELLS) * len(HPO_GRID),
        "retry_per_failed_unit": 1,
        "maximum_attempts_per_unit": 2,
        "maximum_unique_units": len(ALL_DRAW_IDS) * N_FOLDS * len(CELLS) * len(HPO_GRID),
        "theoretical_maximum_training_attempts": 2 * len(ALL_DRAW_IDS) * N_FOLDS * len(CELLS) * len(HPO_GRID),
        "reserve_activation_order": list(RESERVE_DRAW_IDS),
        "interim_effect_scoring_permitted": False,
    },
    "descriptors": {
        "D09R": {
            "formula": "mean_draw mean_fold{[T_GR(sel)-T_GG(sel)]-[T_GR(idx4)-T_GG(idx4)]}",
            "fixed_config_index": FIXED_CONFIG_INDEX,
            "inference": "mean and 95% Student-t CI only; no p-value or verdict",
        }
    },
    "artifacts": {
        "plan": "plan/run_plan.csv",
        "split_pattern": "plan/splits/n14r__d{draw_id:02d}__{cell}.json",
        "unit_pattern": "runs/n14r/units/{unit_id}/unit.json",
        "history_pattern": "runs/n14r/units/{unit_id}/history.json",
        "prediction_pattern": "runs/n14r/units/{unit_id}/predictions.csv",
        "done_pattern": "runs/n14r/units/{unit_id}/DONE",
        "ledger": "runs/n14r/ledger.jsonl",
        "result": "results/n14r_results.json",
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_spec(spec: dict[str, Any] = SPEC) -> None:
    """Fail closed if a purported N14R spec changes a frozen invariant."""
    design = spec["design"]
    expected_environment = {
        "python": "3.13.9", "numpy": "2.4.6", "scipy": "1.18.0",
        "scikit_learn": "1.9.0", "torch": "2.11.0+cu128",
        "torch_cuda": "12.8", "cudnn": 91900, "librosa": "0.11.0",
        "gpu": "NVIDIA GeForce RTX 5070",
    }
    checks = (
        (spec["study_id"] == STUDY_ID, "study_id"),
        (spec["seed_namespace"] == SEED_NAMESPACE, "seed_namespace"),
        (design["primary_draw_ids"] == list(range(24)), "primary draw IDs"),
        (design["reserve_draw_ids"] == list(range(24, 28)), "reserve draw IDs"),
        (design["folds_per_draw"] == 5 and design["train_reps"] == 1, "fold/rep counts"),
        (design["cells"] == ["GR_hpo", "GG_hpo"], "cells"),
        (design["manifest_input_rows"] == 7442, "raw manifest size"),
        (design["analysis_population_rows_after_hygiene"] == 7435, "hygienic population size"),
        (len(design["configs"]) == 8, "HPO grid"),
        (design["expected_primary_units"] == 1920, "primary units"),
        (design["maximum_units"] == 2240, "maximum units"),
        (spec["completion"]["units_per_draw"] == 80, "units per draw"),
        (spec["completion"]["maximum_attempts_per_unit"] == 2, "attempts per unit"),
        (spec["completion"]["theoretical_maximum_training_attempts"] == 4480, "maximum attempts"),
        (spec["completion"]["interim_effect_scoring_permitted"] is False, "interim scoring"),
        (spec["legacy_pilot"]["excluded_from_confirmatory_analysis"] is True, "pilot exclusion"),
        (spec["execution_environment"] == expected_environment, "execution environment"),
        (spec["inference"]["multiplicity"] == "Holm family N14R of size 1", "multiplicity"),
        (spec["inference"]["directional_claim_requires"] ==
         "mean_delta>0 and holm_adjusted_two_sided_p<=0.05", "directional boundary"),
        (spec["budget"]["resource_allocation_is_stopping_rule"] is False, "budget stopping rule"),
        (spec["budget"]["hard_runtime_cap"] is None, "hard runtime cap"),
    )
    for ok, name in checks:
        if not ok:
            raise ValueError(f"invalid frozen N14R invariant: {name}")


validate_spec()

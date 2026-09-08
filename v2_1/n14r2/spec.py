"""N14R2 constants: scientific design and prospective execution contract.

No outcomes or trainer state are loaded here.  ``spec.json`` is an exact JSON
serialization of SPEC; both are pinned before the first formal fit dispatch.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

STUDY_ID = "N14R2"
SPEC_VERSION = "2.0.0"
SEED_NAMESPACE = "SER26-N14R2"
CORPUS_LEVEL = "cremad"
MODEL = "resnet_se"
CELLS = ("GR_hpo", "GG_hpo")
N_PRIMARY_DRAWS = 24
N_RESERVE_DRAWS = 4
N_FOLDS = 5
N_TRAIN_REPS = 1
N_CLASSES = 6
N_PODS = 4
FIT_WORKERS_PER_POD = 8
PRIMARY_DRAW_IDS = tuple(range(N_PRIMARY_DRAWS))
RESERVE_DRAW_IDS = tuple(range(N_PRIMARY_DRAWS, N_PRIMARY_DRAWS + N_RESERVE_DRAWS))
ALL_DRAW_IDS = PRIMARY_DRAW_IDS + RESERVE_DRAW_IDS
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 202609050001
FIXED_CONFIG_INDEX = 4

# Preserve the original v2 enumeration, including fixed comparator index 4.
HPO_GRID = tuple(
    {"engine": "p1_frozen", "model": MODEL, "lr": lr,
     "weight_decay": wd, "dropout": dropout, "batch_size": 64,
     "epochs": 100, "patience": 15, "hpo_config_index": i}
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


def pod_index_for_draw(draw_id: int) -> int:
    """A whole draw is permanently assigned to one of four nodes."""
    if isinstance(draw_id, bool) or not isinstance(draw_id, int) or draw_id not in ALL_DRAW_IDS:
        raise ValueError(f"invalid draw_id: {draw_id!r}")
    return draw_id % N_PODS


SPEC: dict[str, Any] = {
    "study_id": STUDY_ID,
    "spec_version": SPEC_VERSION,
    "seed_namespace": SEED_NAMESPACE,
    "registration": {
        "execution_requires_pinned_commit_tag_and_public_receipt": True,
        "machine_spec_json_is_exact_serialization_of_python_spec": True,
        "prospective_status_is_established_by_external_timestamp_not_this_file": True,
    },
    "legacy_pilot": {
        "source_commit": "c0c0beb", "draws": [0], "use": "design_only",
        "excluded_from_confirmatory_analysis": True,
    },
    "previous_n14r": {
        "status": "not_tested_incomplete",
        "source_preregistration_commit": "fb28c28004fa4f33322736995a63651775dceecb",
        "completed_draws": 16,
        "successful_fits_including_partial_draw": 1298,
        "reused_draws": 0, "reused_fits": 0,
        "excluded_from_all_n14r2_estimates_and_decisions": True,
        "effects_inspected_for_this_redesign": False,
    },
    "execution_environment": {
        "python": "3.12.3", "numpy": "2.4.6", "scipy": "1.18.0",
        "scikit_learn": "1.9.0", "torch": "2.11.0+cu128",
        "torch_cuda": "12.8", "cudnn": 91900, "librosa": "0.11.0",
        "gpu": "NVIDIA GeForce RTX 4090",
    },
    "execution": {
        "operating_system": "Linux",
        "nodes": N_PODS, "gpus_per_node": 1,
        "fit_workers_per_node": FIT_WORKERS_PER_POD,
        "torch_cpu_threads_per_fit_worker": 1,
        "process_start_method": "spawn",
        "draw_assignment": "draw_id % 4",
        "whole_draw_remains_on_assigned_node": True,
        "paired_cells_configs_share_node_gpu_runtime": True,
        "unit_seeds_reset_before_every_fit": True,
        "distributed_gradient_training": False,
        "mixed_precision_training": False,
        "precision_interpretation": "FP32 tensors; no autocast or GradScaler; original backend TF32 flags retained",
        "epoch_checkpoint_resume": False,
        "immutable_attempt_directories": True,
        "single_writer_per_node_ledger": True,
        "global_reserve_authorization_required": True,
        "all_primary_draws_resolved_before_any_reserve": True,
    },
    "design": {
        "corpus_level": CORPUS_LEVEL, "model": MODEL,
        "manifest_input_rows": 7442,
        "analysis_population_rows_after_hygiene": 7435,
        "hygiene_rule": "v2.ser_v2.corpora.apply_hygiene('cremad'): H1/H2/H3",
        "cells": list(CELLS), "primary_draw_ids": list(PRIMARY_DRAW_IDS),
        "reserve_draw_ids": list(RESERVE_DRAW_IDS),
        "folds_per_draw": N_FOLDS, "train_reps": N_TRAIN_REPS,
        "configs": list(HPO_GRID), "expected_primary_units": 1920,
        "maximum_units": 2240,
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
        "multiplicity": "Holm family N14R2 of size 1",
        "bootstrap_sensitivity": {"method": "percentile", "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "sign_flip_sensitivity": "full enumeration of 2^24 sign assignments; symmetry assumption required",
    },
    "power": {
        "model": "noncentral t, two-sided alpha 0.05",
        "alternative_mean_pp": 1.5, "sd_pp": 2.5,
        "n_draws": N_PRIMARY_DRAWS, "power": 0.804,
    },
    "budget": {
        "agent_guard_usd": 25.0,
        "agent_guard_action": "pause_before_further_dispatch_and_request_direction",
        "resource_allocation_is_stopping_rule": False,
        "hard_runtime_cap": None,
        "plan_est_gpu_sec_zero_means_unestimated_not_zero_cost": True,
        "measured_benchmark_is_not_formal_completion_time": True,
    },
    "completion": {
        "required_complete_draws": N_PRIMARY_DRAWS,
        "maximum_activated_draws": len(ALL_DRAW_IDS),
        "units_per_draw": 80, "retry_per_failed_unit": 1,
        "maximum_attempts_per_unit": 2, "maximum_unique_units": 2240,
        "theoretical_maximum_training_attempts": 4480,
        "attempt_boundary": "health_checked_process_then_immutable_attempt_reservation_and_dispatch",
        "dispatch_counts_even_if_start_is_uncertain": True,
        "initialization_or_health_check_without_dispatch_counts_as_attempt": False,
        "infrastructure_failure_action": "pause_and_circuit_break",
        "infrastructure_exhaustion_action": "blocked_not_void_no_reserve",
        "infrastructure_retry_requires_fresh_healthy_process_and_global_authorization": True,
        "failure_classification": "outcome_blind_allowlisted_training_failure_otherwise_infrastructure_or_unknown",
        "training_failure_allowlist": ["FloatingPointError after infrastructure/CUDA exclusions", "exact lowercased exception message: no validation checkpoint"],
        "two_allowlisted_training_failures_action": "void_whole_draw",
        "mixed_or_unknown_exhausted_failures_action": "blocked_not_void_no_reserve",
        "exhausted_infrastructure_change_requires_prospective_public_amendment": True,
        "primary_resolved_states_before_reserves": ["complete", "void"],
        "void_status_requires": "two_allowlisted_training_failures_in_one_unit; void_training_failure_is_a_semantic_alias",
        "reserve_activation_order": list(RESERVE_DRAW_IDS),
        "reserve_activation": "one_at_a_time_global_authorization_only_as_needed_after_all_primaries_resolved",
        "analysis_draw_selection": "first_24_complete_in_frozen_analysis_order",
        "interim_effect_scoring_permitted": False,
        "fewer_than_24_after_all_eligible_reserves": "not_tested_incomplete_no_primary_p_value",
        "paused_or_blocked_is_terminal_incomplete": False,
    },
    "descriptors": {
        "D09R2": {
            "formula": "mean_draw mean_fold{[T_GR(sel)-T_GG(sel)]-[T_GR(idx4)-T_GG(idx4)]}",
            "fixed_config_index": FIXED_CONFIG_INDEX,
            "inference": "mean and 95% Student-t CI only; no p-value or verdict",
        }
    },
    "artifacts": {
        "plan": "plan/run_plan.csv",
        "split_pattern": "plan/splits/n14r2__d{draw_id:02d}__{cell}.json",
        "attempt_pattern": "runs/n14r2/nodes/node{node_index}/units/{unit_id}/attempt_{attempt:02d}",
        "unit_pattern": "runs/n14r2/nodes/node{node_index}/units/{unit_id}/attempt_{attempt:02d}/unit.json",
        "history_pattern": "runs/n14r2/nodes/node{node_index}/units/{unit_id}/attempt_{attempt:02d}/history.json",
        "prediction_pattern": "runs/n14r2/nodes/node{node_index}/units/{unit_id}/attempt_{attempt:02d}/predictions.csv",
        "receipt_pattern": "runs/n14r2/nodes/node{node_index}/units/{unit_id}/attempt_{attempt:02d}/artifact_receipt.json",
        "done_pattern": "runs/n14r2/nodes/node{node_index}/units/{unit_id}/DONE.json",
        "node_ledger_pattern": "runs/n14r2/nodes/node{node_index}/ledger.jsonl",
        "environment_pattern": "runs/n14r2/nodes/node{node_index}/execution_environment.json",
        "global_ledger": "runs/n14r2/global_ledger.jsonl",
        "analysis_lock": "runs/n14r2/analysis_lock.json",
        "result": "results/n14r2_results.json",
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_FROZEN_SPEC_JSON = canonical_json(SPEC)


def validate_spec(spec: dict[str, Any] | None = None) -> None:
    """Fail closed for any contract mutation, including nested execution rules."""
    candidate = SPEC if spec is None else spec
    if canonical_json(candidate) != _FROZEN_SPEC_JSON:
        raise ValueError("invalid frozen N14R2 contract: candidate differs from spec.py")
    if (len(HPO_GRID) != 8 or HPO_GRID != tuple(SPEC["design"]["configs"])
            or len(ALL_DRAW_IDS) * N_FOLDS * len(CELLS) * len(HPO_GRID) != 2240):
        raise ValueError("invalid frozen N14R2 constants")


validate_spec()

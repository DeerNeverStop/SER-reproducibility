"""Frozen, outcome-independent planning calculation for N14R.

This module is deliberately separate from the scorer.  It records how the
number of independent outer-split draws was chosen before any N14R model fit.
The historical v2 fold values are planning inputs only and are never included
in the confirmatory analysis.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.stats import nct, t


PILOT_FOLD_EFFECTS_PP = np.asarray(
    [
        2.2958991558819832,
        2.353445865810116,
        -1.3674419847574342,
        0.31437250183857657,
        4.84435082804999,
    ],
    dtype=float,
)
PILOT_RESULTS_SHA256 = "3280f9da0033b43c8726d2fa96ec266aa16be2fdd3553c26cbf288822f756aae"
PILOT_COMMIT = "c0c0bebfb1509b9606e73b806f5b303fe4c26546"
DESIGN_MEAN_PP = 1.5
PLANNING_SD_PP = 2.5
ALPHA = 0.05
TARGET_POWER = 0.80
N_PRIMARY_DRAWS = 24
N_RESERVE_DRAWS = 4
PLANNED_UNITS_PER_DRAW = 5 * 2 * 8


def two_sided_one_sample_t_power(n: int, mean: float, sd: float, alpha: float) -> float:
    """Power of a two-sided one-sample t test under a normal planning model."""
    df = n - 1
    noncentrality = mean / (sd / math.sqrt(n))
    critical = t.ppf(1.0 - alpha / 2.0, df)
    return float(nct.cdf(-critical, df, noncentrality) + 1.0 - nct.cdf(critical, df, noncentrality))


def planning_record() -> dict:
    powers = {
        str(n): two_sided_one_sample_t_power(n, DESIGN_MEAN_PP, PLANNING_SD_PP, ALPHA)
        for n in range(8, N_PRIMARY_DRAWS + 1)
    }
    return {
        "schema": "ser26-n14r-power-1",
        "computed_before_n14r_fits": True,
        "pilot": {
            "source_commit": PILOT_COMMIT,
            "source_verifier_results_sha256": PILOT_RESULTS_SHA256,
            "fold_effects_pp": PILOT_FOLD_EFFECTS_PP.tolist(),
            "mean_pp": float(PILOT_FOLD_EFFECTS_PP.mean()),
            "sample_sd_pp": float(PILOT_FOLD_EFFECTS_PP.std(ddof=1)),
            "confirmatory_inclusion": False,
        },
        "planning_model": {
            "distribution": "Normal draw-level effects",
            "test": "two-sided one-sample Student t",
            "alpha": ALPHA,
            "design_mean_pp": DESIGN_MEAN_PP,
            "planning_sd_pp": PLANNING_SD_PP,
            "target_power": TARGET_POWER,
            "rationale": "The planning SD exceeds the observed five-fold pilot SD and is used directly as a draw-level SD.",
        },
        "design": {
            "primary_draws": N_PRIMARY_DRAWS,
            "reserve_draws": N_RESERVE_DRAWS,
            "planned_units_per_draw_before_retries": PLANNED_UNITS_PER_DRAW,
            "primary_planned_units_before_retries": N_PRIMARY_DRAWS * PLANNED_UNITS_PER_DRAW,
            "maximum_unique_planned_units_before_retries": (N_PRIMARY_DRAWS + N_RESERVE_DRAWS) * PLANNED_UNITS_PER_DRAW,
            "maximum_attempts_per_unit": 2,
            "theoretical_maximum_training_attempts": 2 * (N_PRIMARY_DRAWS + N_RESERVE_DRAWS) * PLANNED_UNITS_PER_DRAW,
            "projected_primary_first_attempt_gpu_hours": 92.9,
            "projected_all_28_draws_first_attempt_gpu_hours": 108.4,
            "projected_all_28_draws_all_retries_gpu_hours": 216.8,
            "soft_resource_allocation_gpu_hours": 125.0,
            "projected_primary_power": powers[str(N_PRIMARY_DRAWS)],
            "first_n_reaching_target": next(int(n) for n, p in powers.items() if p >= TARGET_POWER),
            "adaptive_reestimation": False,
        },
        "resource_limit_is_scientific_stopping_rule": False,
    }


def main() -> None:
    print(json.dumps(planning_record(), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

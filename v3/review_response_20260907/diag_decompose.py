"""Independent CSV-only DIAG claim audit; no original scoring imports or fits.

Usage: python v3/review_response_20260907/diag_decompose.py --repo .
Outputs are new diag_* files in this script's directory. No empirical tests/CI.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np

ROLES = ("seen", "unseen")
HALVES = ("A", "B")
CONTEXTS = tuple((d, f) for d in range(24) for f in range(5))
DIRECTIONS = (("A_to_B", "A", "B"), ("B_to_A", "B", "A"))
TOL = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    require(bool(values), "empty mean")
    return sum(values, F(0)) / len(values)


def count_score(value, n):
    """Exact UAR from known equal-class-support n=144/288 correct counts."""
    v = float(value)
    k = round(v * n / 100)
    require(0 <= k <= n and abs(100 * k / n - v) < TOL, "not a count-grid score")
    return F(100 * k, n)


def packet(value):
    return {"pp": float(value), "exact_fraction": str(value)}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def components(indicators, differences):
    """E[I D] = E[I] E[D] + covariance, finite population denominator n."""
    probability = mean(indicators)
    actual = mean(i * d for i, d in zip(indicators, differences))
    product = probability * mean(differences)
    cov = mean((i - probability) * (d - mean(differences))
               for i, d in zip(indicators, differences))
    require(actual == product + cov, "exact covariance identity")
    return probability, actual, product, cov


def self_checks():
    # Global frequency weighting fails even though all choices are valid.
    p, a, m, cov = components([F(1), F(0)], [F(10), F(0)])
    require((p, a, m, cov) == (F(1, 2), F(5), F(5, 2), F(5, 2)), "synthetic covariance")
    p, a, m, cov = components([F(1), F(1)], [F(10), F(0)])
    require(a == m and cov == 0, "constant selection")
    require(count_score("50.69444444444444", 144) == F(7300, 144), "score grid")
    require(min(range(4), key=lambda c: (-[F(2), F(2), F(1), F(0)][c], c)) == 0,
            "tie rule")
    return 4


def expected_max4():
    # Numerical integration of 4*z*phi(z)*Phi(z)^3, not a copied score routine.
    z = np.linspace(-10.0, 10.0, 40001)
    cdf = np.fromiter((statistics.NormalDist().cdf(float(t)) for t in z), float)
    y = 4 * z * np.exp(-z * z / 2) / math.sqrt(2 * math.pi) * cdf ** 3
    return float(np.trapezoid(y, z))


def toy_simulation(sigma, mean_offsets, tau_ratio, seed, index, repetitions):
    """A=mu+tau*Zshared+sigma*Za; B=mu+tau*Zshared+sigma*Zb.

    Each replicate has 120 independent toy contexts. Both directions are averaged.
    tau is a shared context-specific configuration quality component, not fitted.
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index])))
    shared = rng.standard_normal((repetitions, 120, 4)) * sigma * tau_ratio
    errors = rng.standard_normal((repetitions, 120, 2, 4)) * sigma
    scores = np.asarray(mean_offsets)[None, None, None, :] + shared[:, :, None, :] + errors
    a, b = scores[:, :, 0, :], scores[:, :, 1, :]
    ia, ib = np.argmax(a, axis=-1), np.argmax(b, axis=-1)
    own_a = np.take_along_axis(a, ia[..., None], axis=-1)[..., 0]
    report_b = np.take_along_axis(b, ia[..., None], axis=-1)[..., 0]
    own_b = np.take_along_axis(b, ib[..., None], axis=-1)[..., 0]
    report_a = np.take_along_axis(a, ib[..., None], axis=-1)[..., 0]
    reuse = ((own_a - report_b + own_b - report_a) / 2).mean(axis=1)
    require(reuse.min() >= -1e-10, "toy argmax nonnegativity")
    return {
        "replicate_mean_reuse_pp": float(reuse.mean()),
        "replicate_sd_pp": float(reuse.std(ddof=1)),
        "replicate_quantiles_025_975_pp": np.quantile(reuse, [.025, .975]).tolist(),
        "quantile_scope": "simulation outcome distribution under stated toy assumptions; NOT an empirical CI",
        "mean_offsets_pp": list(map(float, mean_offsets)),
        "shared_context_quality_tau_over_sigma": tau_ratio,
        "seed_sequence_entropy": [seed, index],
    }


def audit(repo, seed, simulations):
    names = {
        "metrics": "v3/validation_reporting/evidence/analysis/metrics.csv",
        "episodes": "v3/validation_reporting/evidence/analysis/episodes.csv",
        "results": "v3/validation_reporting/evidence/analysis/results.json",
        "old_full_metrics": "v3/inner_validation/reports/results/pred_metrics.csv",
        "claude_review": "v3/CLAUDE_EXPERIMENT_REVIEW_20260907.md",
    }
    paths = {key: repo / value for key, value in names.items()}
    before = {key: sha(path) for key, path in paths.items()}
    rows = load_csv(paths["metrics"])
    require(len(rows) == 2400, "expected 2400 metrics")
    scores = {}
    all_keys = set()
    for row in rows:
        d, f, c = (int(row[k]) for k in ("draw", "fold", "config_index"))
        key = (d, f, c, row["role"], row["half"])
        require(key not in all_keys, "duplicate metric")
        all_keys.add(key)
        if row["role"] in ROLES:
            require(int(row["n"]) == 144, "half must contain 144 recordings")
            scores[key] = count_score(row["uar"], 144)
    expected = {(d, f, c, role, half) for d, f in CONTEXTS for c in range(4)
                for role in ROLES for half in HALVES}
    require(set(scores) == expected, "exact half metric coverage")
    require(all_keys == expected | {(d, f, c, "test", "all") for d, f in CONTEXTS
                                   for c in range(4)}, "test metric coverage")
    old = {}
    for row in load_csv(paths["old_full_metrics"]):
        if row["checkpoint"] == "last" and row["role"] in ("val_seen", "val_unseen"):
            key = (int(row["draw"]), int(row["fold"]), int(row["config_index"]),
                   row["role"].removeprefix("val_"))
            require(key not in old and int(row["records"]) == 288
                    and int(row["checkpoint_epoch"]) == 15, "old full identity")
            old[key] = count_score(row["uar_percent"], 288)
    require(set(old) == {key[:4] for key in scores}, "old full coverage")
    for key, v in old.items():
        require(v == (scores[(*key, "A")] + scores[(*key, "B")]) / 2,
                "old full must exactly equal the two halves")
    episodes = load_csv(paths["episodes"])
    require(len(episodes) == 480, "expected 480 episodes")
    eidx = {}
    for row in episodes:
        key = (int(row["draw"]), int(row["fold"]), row["direction"], row["rule"])
        require(key not in eidx, "duplicate episode")
        eidx[key] = row
    require(set(eidx) == {(d, f, direction, role) for d, f in CONTEXTS
                         for direction, _, _ in DIRECTIONS for role in ROLES},
            "exact episode coverage")
    cells = []
    for d, f in CONTEXTS:
        for direction, selhalf, rephalf in DIRECTIONS:
            for role in ROLES:
                choice = min(range(4), key=lambda c: (-scores[d, f, c, role, selhalf], c))
                ep = eidx[d, f, direction, role]
                require(int(ep["config_index"]) == choice, "episode choice mismatch")
                exp = scores[d, f, choice, "seen", rephalf] - scores[d, f, choice, "unseen", rephalf]
                reuse = scores[d, f, choice, role, selhalf] - scores[d, f, choice, role, rephalf]
                require(abs(float(exp) - float(ep["report_exposure_pp"])) < TOL, "exposure mismatch")
                require(abs(float(reuse) - float(ep["reuse_pp"])) < TOL, "reuse mismatch")
                cells.append({"draw": d, "fold": f, "direction": direction, "rule": role,
                              "report_half": rephalf, "config_index": choice,
                              "exposure": exp, "reuse": reuse})
    fixed = {c: mean(old[d, f, c, "seen"] - old[d, f, c, "unseen"]
                     for d, f in CONTEXTS) for c in range(4)}
    summary = json.loads(paths["results"].read_text(encoding="utf-8"))
    decomposition = {}
    noise = {}
    for role in ROLES:
        subset = [r for r in cells if r["rule"] == role]
        rows_c = []
        for c in range(4):
            indicators = [F(int(r["config_index"] == c)) for r in subset]
            differences = [scores[r["draw"], r["fold"], c, "seen", r["report_half"]]
                           - scores[r["draw"], r["fold"], c, "unseen", r["report_half"]]
                           for r in subset]
            probability, actual, product, cov = components(indicators, differences)
            require(mean(differences) == fixed[c], "marginal full/half equality")
            selected_differences = [d for i, d in zip(indicators, differences) if i]
            rows_c.append({
                "config_index": c, "selected_count": int(sum(indicators)),
                "selection_probability": float(probability),
                "global_fixed_config_exposure": packet(fixed[c]),
                "conditional_selected_cell_exposure": packet(mean(selected_differences)),
                "actual_joint_contribution": packet(actual),
                "product_of_marginals_contribution": packet(product),
                "covariance_contribution": packet(cov),
            })
        actual = mean(r["exposure"] for r in subset)
        product = sum((F(r["product_of_marginals_contribution"]["exact_fraction"]) for r in rows_c), F(0))
        covariance = sum((F(r["covariance_contribution"]["exact_fraction"]) for r in rows_c), F(0))
        require(actual == product + covariance, "total decomposition")
        require(abs(float(actual) - summary["means"][role + "__report_exposure_pp"]) < TOL,
                "published result mismatch")
        claimed_prediction = {"seen": 2.935, "unseen": 2.942}[role]
        decomposition[role] = {
            "episodes": len(subset), "actual_cell_weighted_exposure": packet(actual),
            "product_global_frequency_times_global_config_mean": packet(product),
            "sum_covariance": packet(covariance), "per_config": rows_c,
            "claude_reported_prediction_pp": claimed_prediction,
            "actual_minus_claude_reported_prediction_pp": float(actual) - claimed_prediction,
            "exact_product_minus_claude_reported_prediction_pp": float(product) - claimed_prediction,
            "actual_reuse": packet(mean(r["reuse"] for r in subset)),
        }
        delta = np.array([[float(scores[d, f, c, role, "A"] - scores[d, f, c, role, "B"])
                           for c in range(4)] for d, f in CONTEXTS])
        sigma = math.sqrt(float(np.sum((delta - delta.mean(axis=1, keepdims=True)) ** 2))
                          / (2 * 120 * 3))
        means = np.array([float(mean(old[d, f, c, role] for d, f in CONTEXTS)) for c in range(4)])
        noise[role] = {
            "sigma_e_under_additive_independent_error_assumption_pp": sigma,
            "sigma_formula": "sqrt(sum_p,c (d_pc-mean_c d_pc)^2 / (2*120*(4-1))), d=A-B",
            "raw_config_half_difference_correlation": np.corrcoef(delta.T).tolist(),
            "raw_correlation_scope": "descriptive; common half-difficulty can induce correlation without changing argmax",
            "observed_fixed_config_role_means_percent": means.tolist(),
        }
    pooled_sigma = math.sqrt(sum(noise[r]["sigma_e_under_additive_independent_error_assumption_pp"] ** 2
                                 for r in ROLES) / 2)
    multiplier = expected_max4()
    scenarios = []
    # Analyst-chosen illustrative grid; not calibrated to match observed reuse.
    for i, tau in enumerate((0.0, 0.5, 1.0, 2.0)):
        simulation = toy_simulation(pooled_sigma, [0] * 4, tau, seed, i, simulations)
        simulation.update({
            "name": "equal_global_means_shared_context_signal_tau_" + str(tau),
            "analytic_expected_reuse_pp": pooled_sigma * multiplier / math.sqrt(1 + tau * tau),
            "sigma_e_pp": pooled_sigma,
        })
        scenarios.append(simulation)
    for i, role in enumerate(ROLES, start=4):
        means = np.array(noise[role]["observed_fixed_config_role_means_percent"])
        offsets = means - means.mean()
        simulation = toy_simulation(pooled_sigma, offsets, 0.0, seed, i, simulations)
        simulation.update({"name": "observed_global_mean_offsets_" + role,
                           "sigma_e_pp": pooled_sigma,
                           "mean_source": "same data; illustrative plug-in, NOT a fitted/validated null"})
        scenarios.append(simulation)
    after = {key: sha(path) for key, path in paths.items()}
    require(before == after, "inputs changed during audit")
    output = {
        "schema": "ser-diag-claim-decomposition-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "scope": "CSV-only descriptive algebra and explicitly hypothetical sensitivity; no new formal inference",
        "input_files": {key: {"path": names[key], "sha256": before[key],
                             "bytes": paths[key].stat().st_size} for key in names},
        "script_sha256": sha(Path(__file__)),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "coverage": {"metrics": 2400, "half_metrics": 1920, "old_last_full_metrics": 960,
                     "episodes": 480, "contexts": 120, "directions": 2,
                     "exact_full_half_equalities": 960, "self_checks": self_checks()},
        "formula": "E[sum_c I_c D_c] = sum_c E[I_c] E[D_c] + sum_c Cov(I_c,D_c)",
        "covariance_denominator": "finite empirical population n=240, not sample n-1",
        "decomposition": decomposition,
        "noise_scale_descriptives": noise,
        "pooled_sigma_e_pp": pooled_sigma,
        "claude_claimed_sigma_e_pp": 2.233,
        "Emax_four_iid_standard_normals": multiplier,
        "iid_equal_config_expected_reuse_using_claimed_sigma_pp": 2.233 * multiplier,
        "simulation": {"seed": seed, "rng": "PCG64 with per-scenario SeedSequence",
                       "replicates": simulations, "independent_toy_contexts_per_replicate": 120,
                       "scenarios": scenarios},
        "empirical_p_values_computed": False,
        "empirical_confidence_intervals_computed": False,
        "training_or_inference_performed": False,
        "limitations": [
            "Reuses committed CSV, not logits or raw audio, and does not independently validate labels/predictions.",
            "Reconstructed sigma matches a plausible ANOVA-style estimator; Claude supplied no script, so its exact procedure is unknown.",
            "The toy independent-context outcome quantiles are not calibrated for overlapping real speakers, folds or draws.",
            "Normal homoscedastic additive errors, equal configuration truths, correlations and latent quality are assumptions.",
            "A lower observed reuse than the equal-true-configuration noise case does not imply zero information.",
            "A nonzero covariance term refutes the claimed global identity, but does not establish causal exposure or independent new evidence."
        ],
    }
    return output, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-prefix", default="diag_decomposition")
    parser.add_argument("--seed", type=int, default=2026090716)
    parser.add_argument("--simulations", type=int, default=1000)
    args = parser.parse_args()
    require(100 <= args.simulations <= 10000, "bounded simulation size")
    require(args.out_prefix.startswith("diag_") and Path(args.out_prefix).name == args.out_prefix,
            "output basename must begin diag_")
    output_dir = Path(__file__).resolve().parent
    out_json = output_dir / (args.out_prefix + ".json")
    out_csv = output_dir / (args.out_prefix + "_cells.csv")
    require(not out_json.exists() and not out_csv.exists(), "refuse overwrite")
    self_checks()
    result, cells = audit(args.repo.resolve(), args.seed, args.simulations)
    with out_csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["draw", "fold", "direction", "rule",
                               "report_half", "config_index", "exposure_pp", "exposure_fraction",
                               "reuse_pp", "reuse_fraction"], lineterminator="\n")
        writer.writeheader()
        for row in cells:
            writer.writerow({**{k: row[k] for k in writer.fieldnames if k in row},
                             "exposure_pp": float(row["exposure"]),
                             "exposure_fraction": str(row["exposure"]),
                             "reuse_pp": float(row["reuse"]),
                             "reuse_fraction": str(row["reuse"])})
    result["cell_output"] = {"path": out_csv.name, "sha256": sha(out_csv), "rows": len(cells)}
    with out_json.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"pass": True, "outputs": [out_json.name, out_csv.name],
                     "decomposition": result["decomposition"],
                     "pooled_sigma": result["pooled_sigma_e_pp"],
                     "scenarios": result["simulation"]["scenarios"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


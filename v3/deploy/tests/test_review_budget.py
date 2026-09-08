"""Synthetic tests for v3.deploy.review_budget (B1): tau rule, accepted error, capture rate, quota variant, AURC, CIs."""
from __future__ import annotations

import numpy as np
import pytest

from v3.deploy import common, review_budget as rb
from v3.deploy.common import IntegrityError

CONF = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99])
ERR = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
SPK = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
Y_TRUE = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
Y_PRED = np.where(ERR, (Y_TRUE + 1) % 3, Y_TRUE)


def test_quantile_tau_and_accepted_error_and_capture():
    m = rb.budget_metrics(CONF, ERR, Y_TRUE, Y_PRED, SPK, 0.20, 3)["metrics"]
    tau = np.quantile(CONF, 0.2, method="linear")          # 0.48
    assert m["tau"] == pytest.approx(tau)
    acc = CONF >= tau                                       # 8 accepted; errors accepted at 0.60, 0.80 -> 2/8
    assert m["coverage"] == pytest.approx(0.8)
    assert m["accepted_error"] == pytest.approx(2 / 8)
    assert m["error_capture_rate"] == pytest.approx(2 / 4)  # 0.30, 0.40 rejected errors of 4
    assert m["overall_error_rate"] == pytest.approx(0.4)
    assert m["accepted_uar"] == pytest.approx(common.uar(Y_TRUE[acc], Y_PRED[acc], 3))


def test_per_speaker_coverage_profile():
    out = rb.budget_metrics(CONF, ERR, Y_TRUE, Y_PRED, SPK, 0.20, 3)
    sp = out["speakers"]
    assert sp["A"]["coverage"] == pytest.approx(3 / 5) and sp["B"]["coverage"] == pytest.approx(1.0)
    assert sp["A"]["accepted_error"] == pytest.approx(1 / 3) and sp["B"]["accepted_error"] == pytest.approx(1 / 5)
    assert sp["A"]["n_err_rej"] == 2 and sp["B"]["n_err_rej"] == 0
    m = out["metrics"]
    assert m["speaker_coverage_min"] == pytest.approx(0.6)
    assert m["speakers_below_0.70"] == 1 and m["speakers_zero"] == 0
    assert m["speaker_coverage_p10"] == pytest.approx(np.percentile([0.6, 1.0], 10, method="linear"))


def test_rejected_class_mix_sums_to_one():
    mix = rb.budget_metrics(CONF, ERR, Y_TRUE, Y_PRED, SPK, 0.20, 3)["rejected_mix"]
    assert sum(r["rejected_count"] for r in mix) == 2
    assert sum(r["rejected_share"] for r in mix) == pytest.approx(1.0)
    assert sum(r["class_share_overall"] for r in mix) == pytest.approx(1.0)


def test_quota_variant_rejects_bottom_b_per_speaker():
    q = rb.quota_variant(CONF, ERR, SPK, 0.20)
    tau_a, tau_b = np.quantile(CONF[:5], 0.2, method="linear"), np.quantile(CONF[5:], 0.2, method="linear")
    acc = np.concatenate([CONF[:5] >= tau_a, CONF[5:] >= tau_b])   # A rejects 0.30; B rejects 0.80 (an error)
    assert acc.sum() == 8
    assert q["quota_coverage"] == pytest.approx(0.8)
    assert q["quota_accepted_error"] == pytest.approx((ERR & acc).sum() / 8)
    assert q["quota_error_capture_rate"] == pytest.approx(2 / 4)


def test_aurc_and_oracle():
    a = rb.aurc(CONF, ERR)
    # descending confidence: errors at ranks 5 (0.80), 7 (0.60), 9 (0.40), 10 (0.30)
    risk = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4]) / np.arange(1, 11)
    assert a["aurc"] == pytest.approx(risk.mean())
    oracle = np.maximum(0, np.arange(1, 11) - 6) / np.arange(1, 11)
    assert a["oracle_aurc"] == pytest.approx(oracle.mean())
    assert a["e_aurc"] >= 0
    perfect = rb.aurc(np.where(ERR, 0.1, 0.9), ERR)
    assert perfect["aurc"] == pytest.approx(perfect["oracle_aurc"])


def test_bootstrap_ratio_estimate_is_mean_of_pooled_ratios():
    nums = np.array([[1, 0, 2], [0, 1, 1]], dtype=float)
    dens = np.array([[4, 4, 4], [4, 4, 4]], dtype=float)
    idx = common.bootstrap_indices(3, reps=200)
    r = rb.bootstrap_ratio(nums, dens, idx)
    assert r["estimate"] == pytest.approx(np.mean([3 / 12, 2 / 12]))
    assert r["ci95"][0] <= r["estimate"] <= r["ci95"][1] and r["n"] == 3
    ind = rb.bootstrap_indicator(np.array([[1, 0, 0], [1, 1, 0]]), idx)
    assert ind["estimate"] == pytest.approx(0.5)


def _pooled(seed):
    rng = np.random.default_rng(seed)
    n = 40
    speakers = np.repeat(np.array(["A", "B", "C", "D"]), 10)
    y = rng.integers(0, 3, size=n)
    logits = rng.normal(size=(n, 3)) + 2.0 * np.eye(3)[y]
    return {"speakers": speakers, "y_true": y, "y_pred": logits.argmax(axis=1), "scores": logits}


def test_aggregate_group_shapes_and_endpoints():
    reps = {"r0s0": rb.score_replicate(_pooled(0), 3), "r1s1": rb.score_replicate(_pooled(1), 3)}
    agg = rb.aggregate_group("lv", "m", reps, 3, ["a", "b", "c"])
    assert len(agg["summary"]) == len(rb.BUDGETS)
    assert len(agg["per_speaker"]) == len(rb.BUDGETS) * 4
    assert len(agg["mix"]) == len(rb.BUDGETS) * 3
    q = [e["quantity"] for e in agg["endpoints"]]
    assert q == ["accepted_error", "error_capture_rate", "share_speakers_coverage_below_0.70", "count_speakers_coverage_below_0.70"]
    ep = {e["quantity"]: e for e in agg["endpoints"]}
    row = next(r for r in agg["summary"] if r["b"] == 0.20)
    assert ep["accepted_error"]["estimate"] == pytest.approx(row["accepted_error"])
    assert ep["count_speakers_coverage_below_0.70"]["estimate"] == pytest.approx(row["speakers_below_0.70"])
    assert ep["count_speakers_coverage_below_0.70"]["estimate"] == pytest.approx(4 * ep["share_speakers_coverage_below_0.70"]["estimate"])
    assert all(e["comparison"] == "absolute" for e in agg["endpoints"])
    cov = [r["coverage"] for r in agg["per_speaker"] if r["b"] == 0.20]
    assert np.mean(cov) == pytest.approx(row["coverage"])   # equal speaker sizes -> speaker mean == pooled coverage


def test_pool_replicates_requires_five_disjoint_folds():
    def item(fold, speakers):
        return {"row": {"corpus_level": "lv", "model": "m", "r": "0", "seed_index": "0", "fold": str(fold), "unit_id": f"u{fold}"},
                "pred": {"paths": [f"p{fold}{s}" for s in speakers], "speakers": speakers, "y_true": np.zeros(len(speakers), int),
                         "y_pred": np.zeros(len(speakers), int), "scores": np.zeros((len(speakers), 2))}}
    ok = [item(k, [f"s{k}"]) for k in range(5)]
    pooled = rb.pool_replicates(ok)
    assert list(pooled) == [("lv", "m", 0, 0)] and pooled[("lv", "m", 0, 0)]["speakers"].shape == (5,)
    with pytest.raises(IntegrityError, match="folds"):
        rb.pool_replicates(ok[:4])
    dup = ok[:4] + [item(4, ["s0"])]
    with pytest.raises(IntegrityError, match="two test folds"):
        rb.pool_replicates(dup)


def test_select_v2_units_filters():
    rows = [
        {"arm": "CTRL", "cell": "GG", "corpus_level": "ravdess", "model": "cnn", "r": "0", "seed_index": "0", "fold": "0"},
        {"arm": "CTRL", "cell": "GR", "corpus_level": "ravdess", "model": "cnn", "r": "0", "seed_index": "0", "fold": "0"},
        {"arm": "FT", "cell": "GG", "corpus_level": "cremad", "model": "wavlm_base_plus_ft", "r": "0", "seed_index": "1", "fold": "0"},
        {"arm": "FT", "cell": "GG", "corpus_level": "cremad", "model": "wavlm_base_plus_frozen_sr", "r": "0", "seed_index": "0", "fold": "0"},
        {"arm": "CTRL", "cell": "GG", "corpus_level": "subesco_full", "model": "cnn", "r": "0", "seed_index": "0", "fold": "0"},
    ]
    sel = rb.select_v2_units(rows)
    assert [(r["corpus_level"], r["model"]) for r in sel] == [("cremad", "wavlm_base_plus_ft"), ("ravdess", "cnn")]
    assert [r["model"] for r in rb.select_v2_units(rows, models=["cnn"])] == ["cnn"]


def test_quota_floor_and_path_ties_are_order_invariant():
    paths = [f"p{i}" for i in range(7)]
    a = rb.quota_accept(np.ones(7), .2, paths)
    assert (~a).sum() == 1
    perm = np.array([6, 4, 2, 0, 5, 3, 1])
    b = rb.quota_accept(np.ones(7), .2, np.array(paths)[perm])
    assert set(np.array(paths)[~a]) == set(np.array(paths)[perm][~b])
    assert rb.quota_accept(np.ones(3), .2, ["a", "b", "c"]).all()


def test_missing_accepted_class_is_not_fixed_class_uar():
    m = rb.budget_metrics(np.array([.1,.5,.6,.7,.8]), np.zeros(5, bool), np.array([2,0,1,0,1]),
                          np.array([2,0,1,0,1]), np.array(["a","a","a","b","b"]), .2, 3)["metrics"]
    assert np.isnan(m["accepted_uar"])
    assert m["accepted_present_class_uar"] == 100
    assert m["accepted_support_2"] == 0


def test_fold_sensitivity_is_predefined_and_can_differ():
    p = _pooled(2)
    p["paths"] = [f"p{i}" for i in range(40)]
    p["fold"] = np.repeat(np.arange(5), 8)
    score = rb.score_replicate(p, 3)
    # Global floor(.2*40)=8; separate folds 5*floor(.2*8)=5.
    assert score["budgets"][.2]["metrics"]["n_rejected"] == 8
    assert score["fold_budgets"][.2]["metrics"]["n_rejected"] == 5
    agg = rb.aggregate_group("lv", "m", {"r0": score}, 3, ["a","b","c"])
    assert len(agg["fold_sensitivity"]["endpoints"]) == 4


def test_zero_denominator_bootstraps_are_reported():
    out = rb.bootstrap_ratio(np.zeros((1,2)), np.zeros((1,2)), common.bootstrap_indices(2, reps=30))
    assert np.isnan(out["estimate"]) and out["n_undefined_bootstrap"] == 30

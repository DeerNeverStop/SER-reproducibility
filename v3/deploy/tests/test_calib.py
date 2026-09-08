"""Synthetic tests for v3.deploy.calib (B2): threshold rules, formulas, pairing order, contract, plan determinism."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from v3.deploy import calib, common
from v3.deploy.common import IntegrityError


# ----------------------------------------------------------------------------- helpers

def _pred(paths, speakers, y_true, logits):
    logits = np.asarray(logits, dtype=np.float64)
    return {"paths": list(paths), "speakers": list(speakers), "y_true": np.asarray(y_true), "y_pred": logits.argmax(axis=1),
            "scores": logits}


def _logits_with_conf(conf, correct, y_true, n_classes=3):
    """3-class logits whose max softmax equals conf and whose argmax is y_true (correct) or another class."""
    out = []
    for c, ok, y in zip(conf, correct, y_true):
        p = np.full(n_classes, (1.0 - c) / (n_classes - 1))
        pred = y if ok else (y + 1) % n_classes
        p[pred] = c
        out.append(np.log(p))
    return np.asarray(out)


# ----------------------------------------------------------------------------- threshold rules

def test_primary_tau_is_linear_quantile():
    conf = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert calib.primary_tau(conf, 0.20) == pytest.approx(0.28)
    assert calib.primary_tau(conf, 0.20) == pytest.approx(np.quantile(conf, 0.2, method="linear"))


def test_score_unit_coverage_and_error_formulas():
    conf_v = np.array([0.36, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99])
    ok_v = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=bool)          # errors concentrated at low confidence
    y_v = np.arange(10) % 3
    val = _pred([f"v{i}" for i in range(10)], ["s"] * 10, y_v, _logits_with_conf(conf_v, ok_v, y_v))
    conf_t = np.array([0.36, 0.45, 0.50, 0.55, 0.65, 0.85, 0.90, 0.99])
    ok_t = np.array([0, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
    y_t = np.arange(8) % 3
    spk_t = ["A", "A", "A", "A", "B", "B", "B", "B"]
    test = _pred([f"t{i}" for i in range(8)], spk_t, y_t, _logits_with_conf(conf_t, ok_t, y_t))
    res = calib.score_unit(val, test)
    tau = np.quantile(conf_v, 0.2, method="linear")                      # 0.48
    p = res["primary"]
    assert p["tau"] == pytest.approx(tau)
    acc_v = conf_v >= tau                                                  # 8 accepted, 1 error (0.60)
    assert p["val_coverage"] == pytest.approx(0.8)
    assert p["val_accepted_error"] == pytest.approx(1 / 8)
    acc_t = conf_t >= tau                                                  # 6 accepted: 0.50(err) .55 .65 .85(err) .90 .99
    assert p["test_coverage"] == pytest.approx(acc_t.mean())
    assert p["test_accepted_error"] == pytest.approx(2 / 6)
    assert p["gap_coverage"] == pytest.approx(acc_t.mean() - 0.8)
    assert p["gap_accepted_error"] == pytest.approx(2 / 6 - 1 / 8)
    sp = res["speakers"]
    assert sp["A"]["coverage"] == pytest.approx(2 / 4) and sp["A"]["accepted_error"] == pytest.approx(1 / 2)
    assert sp["B"]["coverage"] == pytest.approx(1.0) and sp["B"]["accepted_error"] == pytest.approx(1 / 4)
    assert sp["A"]["gap_coverage"] == pytest.approx(0.5 - 0.8)
    assert sp["B"]["gap_accepted_error"] == pytest.approx(1 / 4 - 1 / 8)


def test_secondary_rule_smallest_rejection_meeting_half_risk():
    conf = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    err = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)                  # val error 3/8 -> r* = 3/16
    s = calib.secondary_tau(conf, err)
    assert s["r_star"] == pytest.approx(3 / 16)
    # reject 0 -> 3/8; reject 1 -> 2/7; reject 2 -> 1/6 <= 3/16  => tau = 0.5, rejection 2/8
    assert s["tau"] == pytest.approx(0.5)
    assert s["val_rejection"] == pytest.approx(2 / 8)
    assert s["val_accepted_error"] == pytest.approx(1 / 6)


def test_secondary_rule_ties_use_threshold_semantics():
    conf = np.array([0.5, 0.5, 0.5, 0.9])
    err = np.array([1, 1, 0, 0], dtype=bool)                              # r* = 1/4; only tau > 0.5 works
    s = calib.secondary_tau(conf, err)
    assert s["tau"] == pytest.approx(0.9) and s["val_rejection"] == pytest.approx(0.75)


def test_secondary_rule_zero_error_keeps_everything():
    s = calib.secondary_tau(np.array([0.2, 0.9]), np.array([0, 0], dtype=bool))
    assert s["val_rejection"] == 0.0 and s["tau"] == pytest.approx(0.2)


# ----------------------------------------------------------------------------- pairing / aggregation

def _scored(cell, r, fold, cov, err):
    speakers = {s: {"coverage": cov[s], "accepted_error": err[s], "gap_coverage": cov[s] - 0.8, "gap_accepted_error": err[s] - 0.1,
                    "coverage_secondary": cov[s], "accepted_error_secondary": err[s], "gap_accepted_error_secondary": err[s] - 0.1,
                    "n": 4, "n_acc": 2, "n_err_acc": 1} for s in cov}
    return {"unit_id": f"{cell}{r}{fold}", "level": "lv", "model": "m", "cell": cell, "r": r, "fold": fold,
            "primary": {"tau": 0.5, "val_accepted_error": 0.1, "test_coverage": 0.8, "gap_coverage": 0.0},
            "secondary": {"tau": 0.5, "val_accepted_error": 0.1, "test_coverage": 0.8}, "speakers": speakers}


def test_gr_minus_gg_pairing_and_r_average():
    scored = [
        _scored("GG", 0, 0, {"A": 0.80, "B": 0.60}, {"A": 0.10, "B": 0.20}),
        _scored("GR", 0, 0, {"A": 0.90, "B": 0.50}, {"A": 0.30, "B": 0.25}),
        _scored("GG", 1, 0, {"A": 0.70}, {"A": 0.10}), _scored("GG", 1, 1, {"B": 0.60}, {"B": 0.20}),
        _scored("GR", 1, 0, {"A": 0.90}, {"A": 0.20}), _scored("GR", 1, 1, {"B": 0.70}, {"B": 0.15}),
    ]
    agg = calib.aggregate({}, scored)
    ep = {e["quantity"]: e for e in agg["endpoints"]}
    # per speaker averaged over r, then GR - GG: A: (0.9+0.9)/2-(0.8+0.7)/2 = 0.15; B: (0.5+0.7)/2-(0.6+0.6)/2 = 0.0
    assert ep["coverage"]["estimate"] == pytest.approx(np.mean([0.15, 0.0]))
    assert ep["coverage"]["n"] == 2 and ep["coverage"]["comparison"] == "GR-GG"
    # A: (0.3+0.2)/2-(0.1+0.1)/2 = 0.15; B: (0.25+0.15)/2-(0.2+0.2)/2 = 0.0
    assert ep["accepted_error"]["estimate"] == pytest.approx(0.075)
    assert ep["coverage"]["ci95"][0] <= ep["coverage"]["estimate"] <= ep["coverage"]["ci95"][1]
    ps = {(r["cell"], r["speaker"]): r for r in agg["per_speaker_rows"]}
    assert ps[("GR", "A")]["coverage"] == pytest.approx(0.9) and ps[("GG", "B")]["coverage"] == pytest.approx(0.6)
    absolute = {a["cell"]: a for a in agg["absolute"]}
    assert absolute["GR"]["speaker.coverage_profile"]["count_below_0.70"] == 1   # B at 0.6


def test_speaker_matrix_rejects_missing_speaker():
    scored = [_scored("GG", 0, 0, {"A": 0.8}, {"A": 0.1}), _scored("GR", 0, 0, {"A": 0.8, "B": 0.5}, {"A": 0.1, "B": 0.1})]
    with pytest.raises(IntegrityError):
        calib.aggregate({}, scored)


# ----------------------------------------------------------------------------- contract / refusal

def _row(unit_id, cell):
    return {"unit_id": unit_id, "level": "lv", "base": "b", "model": "ridge_x", "cell": cell, "r": 0, "fold": 0, "seed_index": 0,
            "train_seed": 0, "split_key": f"k_{cell}", "split_sha256": "s", "config_sha256": "c", "engine": "ridge",
            "n_classes": 3, "n_fit": 4, "n_val": 4, "n_test": 4}


def _write_unit(run: Path, row):
    rng = np.random.default_rng(int(row["cell"] == "GR"))
    y = np.array([0, 1, 2, 0])
    val = _pred(["a", "b", "c", "d"], ["A", "A", "B", "B"], y, rng.normal(size=(4, 3)))
    test = _pred(["e", "f", "g", "h"], ["A", "A", "B", "B"], y, rng.normal(size=(4, 3)))
    return calib.write_unit_outputs(calib.unit_dir(run, row["unit_id"]), row, {"timing": {}, "environment": {}}, val, test)


def test_contract_files_and_columns(tmp_path):
    row = _row("u1", "GG")
    unit = _write_unit(tmp_path, row)
    udir = calib.unit_dir(tmp_path, "u1")
    assert sorted(p.name for p in udir.iterdir()) == ["DONE", "test_predictions.csv", "unit.json", "val_predictions.csv"]
    header = (udir / "val_predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "relative_path,speaker,y_true,y_pred,logit_0,logit_1,logit_2"
    done = (udir / "DONE").read_bytes()
    assert done == f"{unit['val_sha256']} {unit['test_sha256']}\n".encode()
    assert unit["val_sha256"] == common.sha256_file(udir / "val_predictions.csv")
    assert calib.done_status(udir, 3, 4, 4) == "ok"
    assert calib.done_status(udir, 4, 4, 4).startswith("bad header")
    assert calib.done_status(udir, 3, 5, 4).startswith("row count")


def test_score_refuses_missing_done_unless_filtered(tmp_path):
    plan = {"units": [_row("u1", "GG"), _row("u2", "GR"), dict(_row("u3", "GG"), model="cnn", engine="p1_frozen"),
                      dict(_row("u4", "GR"), model="cnn", engine="p1_frozen")]}
    _write_unit(tmp_path, plan["units"][0])
    _write_unit(tmp_path, plan["units"][1])
    with pytest.raises(IntegrityError, match="not complete"):
        calib.score_plan(plan, tmp_path, None)
    res = calib.score_plan(plan, tmp_path, ["ridge_x"])
    assert len(res["scored"]) == 2
    (calib.unit_dir(tmp_path, "u2") / "DONE").unlink()
    with pytest.raises(IntegrityError, match="missing DONE"):
        calib.score_plan(plan, tmp_path, ["ridge_x"])
    with pytest.raises(IntegrityError, match="unknown models"):
        calib.score_plan(plan, tmp_path, ["nope"])


def test_tampered_csv_is_detected(tmp_path):
    row = _row("u1", "GG")
    _write_unit(tmp_path, row)
    p = calib.unit_dir(tmp_path, "u1") / "test_predictions.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("e,A", "e,Z"), encoding="utf-8")
    assert calib.done_status(calib.unit_dir(tmp_path, "u1")) == "DONE sha mismatch"


# ----------------------------------------------------------------------------- plan determinism

def _synthetic_inputs():
    level, base = "ravdess", "ravdess"
    cfg_cnn = {"engine": "p1_frozen", "model": "cnn", "lr": 0.001}
    cfg_ft = {"engine": "wavlm_partial_ft", "model": "wavlm_base_plus_ft", "seed": 0}
    sha_cnn, sha_ft = common.digest(cfg_cnn), common.digest(cfg_ft)
    splits = {}
    for cell in calib.B2_CELLS:
        for r in calib.B2_RS:
            folds = [{"fold": k, "fit": [f"f{k}"] * 3, "val": [f"v{k}"], "test": [f"t{k}"]} for k in calib.B2_FOLDS]
            splits[f"ctrl__{level}__{cell}__r{r}"] = {"sha256": f"sha_{cell}_{r}", "table": {"folds": folds, "population": []}}
    v2_rows = []
    for cell in calib.B2_CELLS:
        for r in calib.B2_RS:
            for fold in calib.B2_FOLDS:
                if r == 0 or cell == "GG":       # leave some cnn references absent to exercise the derived rule
                    v2_rows.append({"arm": "CTRL", "model": "cnn", "corpus_level": level, "cell": cell, "r": str(r), "fold": str(fold),
                                    "seed_index": "0", "train_seed": str(calib.v2_train_seed(level, fold, 0)), "config_sha256": sha_cnn,
                                    "split_sha256": f"sha_{cell}_{r}", "unit_id": f"v2cnn{cell}{r}{fold}", "base_corpus": base})
    for fold in calib.B2_FOLDS:
        v2_rows.append({"arm": "FT", "model": "wavlm_base_plus_ft", "corpus_level": level, "cell": "GG", "r": "0", "fold": str(fold),
                        "seed_index": "0", "train_seed": str(calib.v2_ft_train_seed(level, fold, 0)), "config_sha256": sha_ft,
                        "split_sha256": "sha_GG_0", "unit_id": f"v2ft{fold}", "base_corpus": base})
    features = {k: {"file": f"{k}.npz", "sha256": f"fsha_{k}"} for k in list(common.ENCODERS) + ["logmel"]}
    levels = {level: {"base": base, "feature_corpus": base, "manifest_sha256": "msha", "classes": ["a", "b"], "n_speakers": 2,
                      "splits": splits, "features": features}}
    return {"v2_rows": v2_rows, "configs": {sha_cnn: cfg_cnn, sha_ft: cfg_ft}, "levels": levels}


def test_plan_units_deterministic_and_counts(monkeypatch):
    monkeypatch.setattr(calib, "B2_LEVELS", ("ravdess",))
    inputs = _synthetic_inputs()
    a = calib.build_units(copy.deepcopy(inputs))
    b = calib.build_units(copy.deepcopy(inputs))
    assert [u["unit_id"] for u in a] == [u["unit_id"] for u in b]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    counts = {}
    for u in a:
        counts[u["model"]] = counts.get(u["model"], 0) + 1
    assert counts == {"ridge_wavlm_base_plus": 30, "ridge_hubert_base": 30, "ridge_wav2vec2_base": 30, "cnn": 30, "wavlm_ft": 10}
    assert all(u["r"] == 0 for u in a if u["model"] == "wavlm_ft")
    cnn = [u for u in a if u["model"] == "cnn"]
    assert sum(u["train_seed_source"] == "v2_unit" for u in cnn) == 20
    assert sum(u["train_seed_source"].startswith("derived") for u in cnn) == 10
    assert all(u["train_seed"] == calib.v2_train_seed("ravdess", u["fold"], 0) for u in cnn)
    assert all(u["unit_id"] == calib.make_unit_id(u) for u in a)
    # identity is exactly the 13 documented fields
    assert set(calib.unit_identity(a[0])) == {"program", "module", *calib.IDENTITY_FIELDS}
    # a config change changes ids; nothing else about the row identity leaks in
    changed = copy.deepcopy(inputs)
    monkeypatch.setattr(calib, "RIDGE_CONFIG", dict(calib.RIDGE_CONFIG, alpha=2.0))
    c = calib.build_units(changed)
    ridge_a = {u["unit_id"] for u in a if u["engine"] == "ridge"}
    ridge_c = {u["unit_id"] for u in c if u["engine"] == "ridge"}
    assert not (ridge_a & ridge_c)
    assert {u["unit_id"] for u in a if u["engine"] != "ridge"} == {u["unit_id"] for u in c if u["engine"] != "ridge"}


def test_seed_rule_matches_v2_primitive():
    assert calib.stable_u32("SER26|train|cremad|0|0") == calib.v2_train_seed("cremad", 0, 0)
    assert calib.v2_train_seed("ravdess", 0, 0) == 3020244400          # v2 run_plan.csv, CTRL cnn ravdess fold 0
    assert calib.v2_train_seed("cremad", 0, 0) == 1367736462           # v2 CTRL cnn cremad r0 fold 0
    assert calib.v2_ft_train_seed("subesco_980", 0, 0) == 360048798    # v2 FT GG subesco_980 fold 0 seed 0


def test_secondary_no_feasible_threshold_risk_is_na():
    s = calib.secondary_tau(np.array([.5,.5,.9]), np.ones(3, bool))
    assert not s["feasible"] and np.isinf(s["tau"])
    assert s["val_coverage"] == 0 and np.isnan(s["val_accepted_error"])


def test_primary_ties_gap_uses_actual_val_coverage():
    val = _pred(["v1","v2","v3"], ["v"]*3, [0,1,2], [[1,0,0]]*3)
    test = _pred(["t1","t2","t3"], ["t"]*3, [0,1,2], [[1,0,0]]*3)
    scored = calib.score_unit(val, test)
    assert scored["primary"]["val_coverage"] == 1
    assert scored["primary"]["gap_coverage"] == 0
    assert scored["primary"]["test_review_deviation"] == pytest.approx(-.2)


def test_repeat_risk_na_does_not_silently_select_other_repeats():
    a = calib.reduce_over_r(np.array([[np.nan, .1], [.5, .3]]))
    assert np.isnan(a[0]) and a[1] == pytest.approx(.2)
    ep = calib.paired_endpoint(a)
    assert ep["n"] == 1 and ep["n_population"] == 2 and ep["n_dropped"] == 1
    assert ep["n_undefined_bootstrap"] > 0

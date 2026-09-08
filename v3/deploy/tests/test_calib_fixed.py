"""Metadata isolation, deterministic matching, model identity and threshold tests."""
import copy
from collections import Counter

import numpy as np
import pytest

from v3.deploy import calib_fixed as fx, common


def fixture():
    manifest, fold = {}, {r: [] for r in ("fit", "val", "test")}
    for role, speakers in (("fit", ["a", "b", "c", "d"]), ("val", ["e", "f"]), ("test", ["g", "h"])):
        for speaker in speakers:
            for sentence in ("s1", "s2"):
                for label in range(3):
                    for take in range(4):
                        p = f"{speaker}/{sentence}/{label}/{take}"
                        manifest[p] = {"speaker": speaker, "sentence": sentence, "label_index": label, "label": str(label)}
                        fold[role].append(p)
    return manifest, fold


def test_split_matches_every_donor_cell_and_preserves_training():
    manifest, fold = fixture()
    s = fx.build_split(manifest, fold, "predeclared")
    assert s == fx.build_split(manifest, fold, "predeclared")
    assert len(s["seen"]) == len(s["new"]) == 24
    assert s["test"] == fold["test"]
    assert set(s["seen"]) | set(s["fit"]) == set(fold["fit"])
    for pair in s["donor_pairs"]:
        assert sum(manifest[p]["speaker"] == pair["seen"] for p in s["fit"]) == 12
        for role in ("seen", "new"):
            counts = Counter((manifest[p]["sentence"], manifest[p]["label_index"]) for p in s[role] if manifest[p]["speaker"] == pair[role])
            assert sum(counts.values()) == 12
    assert all(c["n_each_selected"] == min(c["n_seen_eligible"], c["n_new_available"]) for c in s["cell_feasibility"])


def test_split_does_not_depend_on_source_recording_order():
    manifest, fold = fixture()
    changed = copy.deepcopy(fold)
    changed["fit"].reverse()
    changed["val"].reverse()
    assert fx.build_split(manifest, fold, "locked") == fx.build_split(manifest, changed, "locked")


def test_absent_cell_dropped_symmetrically():
    manifest, fold = fixture()
    fold["fit"] = [p for p in fold["fit"] if not (manifest[p]["sentence"] == "s1" and manifest[p]["label_index"] == 0)]
    s = fx.build_split(manifest, fold, "locked")
    dropped = [c for c in s["cell_feasibility"] if c["status"] == "dropped_whole_cell"]
    forced = [c for c in dropped if c["sentence"] == "s1" and c["class_index"] == 0]
    assert len(forced) == 2
    assert all(c["n_seen_available"] == 0 and c["n_each_selected"] == 0 for c in forced)
    assert not any(manifest[p]["sentence"] == "s1" and manifest[p]["label_index"] == 0 for p in s["new"])


def test_impossible_calibration_class_is_early_failure():
    manifest, fold = fixture()
    fold["fit"] = [p for p in fold["fit"] if manifest[p]["label_index"] != 2]
    with pytest.raises(common.IntegrityError, match="lacks a class"):
        fx.build_split(manifest, fold, "locked")


@pytest.mark.parametrize("tamper", ["overlap", "swap", "lose_donor"])
def test_split_tampering_rejected(tamper):
    manifest, fold = fixture()
    s = fx.build_split(manifest, fold, "locked")
    if tamper == "overlap":
        s["fit"].append(s["seen"][0])
    elif tamper == "swap":
        s["seen"], s["new"] = s["new"], s["seen"]
    else:
        donor = s["donor_pairs"][0]["seen"]
        s["fit"] = [p for p in s["fit"] if manifest[p]["speaker"] != donor]
    with pytest.raises(common.IntegrityError):
        fx.validate_split(s, manifest)


def test_saved_model_roundtrip_exact_scores(tmp_path):
    common.cpu_guard(2)
    rng = np.random.default_rng(41)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = np.arange(60) % 3
    model = fx.fit_model(X, y, 3, fx.RIDGE_CONFIG)
    p = tmp_path / "model.npz"
    np.savez_compressed(p, **model)
    restored = dict(np.load(p))
    assert np.array_equal(fx.predict_model(model, X), fx.predict_model(restored, X))
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    classifier = RidgeClassifier(alpha=1, class_weight="balanced", solver="lsqr", tol=1e-4).fit(scaler.transform(X), y)
    assert np.array_equal(fx.predict_model(restored, X), classifier.decision_function(scaler.transform(X)).astype(np.float64))


def pred(conf, prefix, labels=None):
    conf = np.asarray(conf)
    p = np.column_stack([conf, (1-conf)/2, (1-conf)/2])
    y = np.arange(len(conf)) % 3 if labels is None else np.asarray(labels)
    return {"paths": [f"{prefix}{i}" for i in range(len(conf))], "scores": np.log(p), "y_pred": p.argmax(axis=1),
            "y_true": y, "speakers": [f"{prefix}{i//3}" for i in range(len(conf))]}


def test_thresholds_are_label_blind_and_test_shared():
    ps = {"seen": pred([.4,.5,.6,.7,.8,.9], "s"), "new": pred([.5,.6,.7,.8,.9,.95], "n"),
          "test": pred([.4,.5,.6,.7,.8,.9], "t")}
    res = fx.score_unit(ps)
    shuffled = copy.deepcopy(ps)
    for p in shuffled.values():
        p["y_true"] = (p["y_true"] + 1) % 3
    res2 = fx.score_unit(shuffled)
    for role in ("seen", "new"):
        assert res[role]["tau"] == res2[role]["tau"]
        assert res[role]["test_coverage"] == res2[role]["test_coverage"]
        assert res[role]["gap_coverage"] == pytest.approx(res[role]["test_coverage"] - res[role]["val_coverage"])
    assert res["seen"]["tau"] != res["new"]["tau"]


def test_quota_ties_retain_actual_calibration_coverage():
    p = {"seen": pred([.5]*6, "s"), "new": pred([.6]*6, "n"), "test": pred([.5]*6, "t")}
    res = fx.score_unit(p)
    assert res["seen"]["val_coverage"] == 1
    assert res["seen"]["gap_coverage"] == 0
    assert res["new"]["test_coverage"] == 0
    assert np.isnan(res["new"]["test_accepted_error"])


def test_aggregate_pairs_before_repeat_reduction():
    scored = []
    for r in (0, 1):
        res = {"r": r, "fold": 0, "level": "lv", "model": "m"}
        for role, shift in (("seen", .1), ("new", 0)):
            vals = {q: shift for q in ("review_deviation", "gap_coverage", "accepted_error", "gap_accepted_error")}
            res[role] = {"speakers": {"a": vals.copy(), "b": vals.copy()}}
        if r == 0:
            res["seen"]["speakers"]["a"]["accepted_error"] = np.nan
        scored.append(res)
    endpoints = {e["quantity"]: e for e in fx.aggregate(scored)}
    assert endpoints["accepted_error"]["n_dropped"] == 1
    assert endpoints["accepted_error"]["estimate"] == pytest.approx(.1)


def test_one_unit_raw_contract_and_resealed_label_tamper(tmp_path):
    common.cpu_guard(2)
    manifest, fold = fixture()
    split = fx.build_split(manifest, fold, "unit-test")
    rng = np.random.default_rng(82)
    vectors = {p: rng.normal(size=6).astype(np.float32) for p in manifest}
    class Cache:
        sha256 = "synthetic-feature-sha"
        def get(self, paths, state=None):
            return np.stack([vectors[p] for p in paths])
    row = {"program": fx.PROGRAM, "module": fx.MODULE, "level": "synthetic", "model": "ridge_test",
           "r": 0, "fold": 0, "split_key": "split", "split_sha256": common.digest(split),
           "feature_sha256": Cache.sha256, "config": fx.RIDGE_CONFIG, "config_sha256": common.digest(fx.RIDGE_CONFIG),
           "n_classes": 3, **{f"n_{r}": len(split[r]) for r in fx.ROLES}}
    row["unit_id"] = fx.make_unit_id(row)
    meta = fx.run_one_unit({"splits": {"split": split}}, row, tmp_path, manifest, Cache(), {"device": "cpu"}, "synthetic")
    udir = tmp_path / "units" / row["unit_id"]
    assert fx.done_status(udir, row, split, manifest) == "ok"
    assert len((udir / "DONE").read_text().split()) == 4
    with pytest.raises(common.IntegrityError, match="overwrite"):
        fx.run_one_unit({"splits": {"split": split}}, row, tmp_path, manifest, Cache(), {}, "synthetic")
    p = udir / "new_predictions.csv"
    lines = p.read_text().splitlines()
    fields = lines[1].split(",")
    fields[2] = str((int(fields[2]) + 1) % 3)
    lines[1] = ",".join(fields)
    common.atomic_write_text(p, "\n".join(lines) + "\n")
    assert fx.done_status(udir, row, split, manifest) == "DONE sha mismatch"
    # Even recomputed bookkeeping must not conceal manifest-label corruption.
    meta["new_sha256"] = common.sha256_file(p)
    common.atomic_write_json(udir / "unit.json", meta)
    common.atomic_write_text(udir / "DONE", " ".join([meta[f"{r}_sha256"] for r in fx.PRED_ROLES] + [meta["model_sha256"]]) + "\n")
    assert fx.done_status(udir, row, split, manifest) == "prediction labels/speakers mismatch"

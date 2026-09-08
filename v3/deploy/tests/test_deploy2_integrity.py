"""Adversarial DEPLOY-2 tests: literal expected masks, leakage mutations, independent math.

Only synthetic inputs; no real outcomes or pretrained model downloads.
"""
from __future__ import annotations
import copy
import csv
import hashlib
from pathlib import Path
import numpy as np
import pytest
from v3.deploy import verify as V
from v3.deploy import verify_deploy2 as D
from v3.deploy import common

common.cpu_guard(2)


def test_quota_all_ties_exact_count_and_permutation_invariance():
    paths = [f"speaker/{i}.wav" for i in range(13)]
    got = D.quota(np.ones(13), paths, .2)
    assert (~got).sum() == 2
    excluded = {paths[i] for i in np.flatnonzero(~got)}
    expected = set(sorted(paths, key=lambda p: hashlib.sha256(p.encode()).hexdigest())[:2])
    assert excluded == expected
    permutation = np.arange(13)[::-1]
    got2 = D.quota(np.ones(13), [paths[i] for i in permutation], .2)
    assert {paths[permutation[i]] for i in np.flatnonzero(~got2)} == excluded


def test_quota_rejects_duplicate_identity():
    with pytest.raises(V.VerificationError):
        D.quota([.5, .9], ["a", "a"], .2)


def test_risk_threshold_respects_tied_block_and_no_off_by_one():
    threshold = D.risk_threshold([.6, .6, .8, .9], [True, True, False, False])
    assert threshold["tau"] == .8
    assert threshold["val_rejection"] == .5
    assert threshold["val_accepted_error"] == 0
    assert D.risk_threshold([.6, .7, .8, .9], [True, False, False, False])["tau"] == .7


def test_risk_threshold_no_feasible_returns_nan_risk():
    result = D.risk_threshold([.5, .5, .9], [True, True, True])
    assert result["tau"] == float("inf") and result["val_coverage"] == 0
    assert np.isnan(result["val_accepted_error"])


def pred(probs, truth, speakers=None):
    probabilities = np.column_stack([probs, 1-np.asarray(probs)])
    return {"logits": np.log(probabilities), "y_true": np.asarray(truth), "y_pred": probabilities.argmax(1),
            "speakers": speakers or ["s"]*len(truth), "paths": [f"p{i}" for i in range(len(truth))]}


def test_primary_calibration_label_blind_and_actual_coverage_gap():
    val = pred([.6]*5, [0, 1, 0, 1, 0])
    test = pred([.55, .65, .75], [0, 0, 1])
    out = D.calibration_unit(val, test)
    assert out["primary"]["val_coverage"] == 1
    assert out["primary"]["test_coverage"] == pytest.approx(2/3)
    assert out["primary"]["gap_coverage"] == pytest.approx(-1/3)
    altval = copy.deepcopy(val); altval["y_true"] = 1-altval["y_true"]
    alttest = copy.deepcopy(test); alttest["y_true"] = 1-alttest["y_true"]
    alt = D.calibration_unit(altval, alttest)
    assert out["primary"]["tau"] == alt["primary"]["tau"]
    assert out["primary"]["test_coverage"] == alt["primary"]["test_coverage"]


def test_independent_formula_matches_calibration_and_catches_test_threshold():
    from v3.deploy import calib
    val = pred([.6, .6, .8, .9], [1, 1, 0, 0])
    test = pred([.51, .61, .95], [0, 0, 1])
    convert = lambda p: dict(p, scores=p["logits"])
    expected = D.calibration_unit(val, test)
    actual = calib.score_unit(convert(val), convert(test))
    assert D.assert_numeric(expected, actual) > 20
    bad = copy.deepcopy(actual)
    bad["primary"]["tau"] = float(np.quantile(V.softmax_max(test["logits"]), .2))
    with pytest.raises(V.VerificationError, match="tau"):
        D.assert_numeric(expected, bad)


def metadata_fixture(tmp_path):
    rows = []
    for s in ("f", "seen", "new", "test"):
        for sentence in ("a", "b"):
            for label in range(2):
                for take in range(4):
                    rows.append(dict(relative_path=f"{s}/{sentence}/{label}/{take}.wav", speaker=s,
                                     sentence=sentence, label=str(label), label_index=label))
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    manifest = V.Manifest(path)
    split = {r: [] for r in ("fit", "seen", "new", "test")}
    for row in rows:
        s = row["speaker"]; p = row["relative_path"]
        if s == "f" or (s == "seen" and not p.endswith("/0.wav")):
            split["fit"].append(p)
        elif s == "seen":
            split["seen"].append(p)
        elif s == "new" and p.endswith("/0.wav"):
            split["new"].append(p)
        elif s == "test":
            split["test"].append(p)
    return manifest, split


def test_fixed_partition_metadata_has_matched_cells(tmp_path):
    man, split = metadata_fixture(tmp_path)
    report = V.verify_fixed_partition(split, man)
    assert report["matched_cells"] == 4
    assert report["rows"]["seen"] == report["rows"]["new"] == 4


@pytest.mark.parametrize("mutation", ["same_recording", "swap_seen_new", "lost_donor", "cell_mismatch"])
def test_fixed_partition_rejects_scientific_mutations(tmp_path, mutation):
    man, split = metadata_fixture(tmp_path)
    if mutation == "same_recording":
        split["fit"].append(split["seen"][0])
    elif mutation == "swap_seen_new":
        split["seen"], split["new"] = split["new"], split["seen"]
    elif mutation == "lost_donor":
        split["fit"] = [p for p in split["fit"] if man.speaker(p) != "seen"]
    else:
        split["new"].pop()
    with pytest.raises(V.VerificationError):
        V.verify_fixed_partition(split, man)


def test_speaker_bootstrap_uses_full_identity_frame_with_nan():
    values = np.asarray([1., np.nan, 3., 10.])
    result = D.interval(values)
    idx = np.random.default_rng(20260905).integers(0, 4, size=(10000, 4))
    sampled = np.asarray([draw[np.isfinite(draw)].mean() if np.isfinite(draw).any() else np.nan for draw in values[idx]])
    assert result["n"] == 3 and result["n_dropped"] == 1
    assert result["ci95"] == np.nanpercentile(sampled, [2.5, 97.5]).tolist()


def test_fixed_identical_calibrators_share_test_and_yield_zero_difference():
    p = pred([.6, .7, .8, .9], [0, 0, 1, 0], ["a", "a", "b", "b"])
    r = D.fixed_unit({"seen": p, "new": copy.deepcopy(p), "test": p})
    units = [dict(r, level="synthetic", model="ridge", r=0, fold=0, unit_id="one")]
    ep = D.fixed_endpoints(units)
    assert len(ep) == 4
    assert all(e["estimate"] == 0 and e["ci95"] == [0, 0] for e in ep)


def test_cnn_engine_never_requests_test_truth():
    import torch
    from v3.deploy import engines_deploy
    from v3.deploy.tests.test_engines_deploy import SyntheticData, CNN_CFG
    data = SyntheticData(); fold = data.fold(); old_labels = data.labels
    def guarded_labels(paths):
        assert not (set(paths) & set(fold["test"])), "test labels entered training engine"
        return old_labels(paths)
    data.labels = guarded_labels
    test, val, history, info = engines_deploy.engine_p1_frozen({"train_seed": 20260905}, CNN_CFG, fold, data, torch.device("cpu"))
    assert test.shape[0] == len(fold["test"]) and val.shape[0] == len(fold["val"])
    assert "_checkpoint_state" in info


def test_returned_checkpoint_replays_exported_cnn_logits():
    import torch
    import advanced_models
    from v3.deploy import engines_deploy
    from v3.deploy.tests.test_engines_deploy import SyntheticData, CNN_CFG
    data = SyntheticData(); fold = data.fold()
    test, val, _, info = engines_deploy.engine_p1_frozen({"train_seed": 95}, CNN_CFG, fold, data, torch.device("cpu"))
    fresh = advanced_models.build_neural_model("cnn", n_mels=64, n_classes=data.n_classes, config=info["arch"])
    fresh.load_state_dict(info["_checkpoint_state"])
    replay = engines_deploy.predict_torch(fresh, data.logmel(fold["test"]), torch.device("cpu"))
    np.testing.assert_array_equal(replay, test)


def test_b2_independent_full_nested_aggregate_and_all_ci_fields():
    from v3.deploy import calib
    expected_units, actual_units = [], []
    for repeat in range(2):
        for cell in ("GG", "GR"):
            val = pred([.6, .7, .8, .9], [0, 0, 1, 0] if cell == "GG" else [1, 0, 0, 0])
            test = pred([.55, .65+repeat*.03, .8, .95], [0, 1, 0, 0], ["a", "a", "b", "b"])
            meta = dict(unit_id=f"{repeat}{cell}", level="toy", model="ridge", cell=cell, r=repeat, fold=0)
            expected_units.append(dict(D.calibration_unit(val, test), **meta))
            actual_units.append(dict(calib.score_unit(dict(val, scores=val["logits"]), dict(test, scores=test["logits"])), **meta))
    expected = D.b2_aggregate(expected_units); actual = calib.aggregate({}, actual_units)
    for key in ("absolute", "per_speaker_rows"):
        actual[key].sort(key=lambda r: (r["level"], r["model"], r["cell"], r.get("speaker", "")))
    assert D.assert_numeric(expected, actual) > 200
    broken = copy.deepcopy(actual); broken["endpoints"][0]["ci95"][0] += 1e-5
    with pytest.raises(V.VerificationError, match="ci95"):
        D.assert_numeric(expected, broken)


def test_b1_independent_full_summary_quota_fold_sensitivity_and_ci():
    from v3.deploy import review_budget as B
    groups, reps = {}, {}
    for repeat in range(2):
        p = pred(np.asarray([.6, .6, .65, .7, .75, .8, .85, .9, .95, .99]),
                 [0, 1, 0, 0, 1, 0, 1, 0, 0, repeat], ["a"]*5+["b"]*5)
        p["fold"] = np.asarray([0]*5+[1]*5)
        p["speakers"] = np.asarray(p["speakers"])
        groups[("toy", "ridge", repeat, 0)] = dict(p, conf=V.softmax_max(p["logits"]), C=2)
        reps[str(repeat)] = B.score_replicate(dict(p, scores=p["logits"]), 2)
    class Ctx:
        def manifest(self, level):
            return type("Man", (), {"classes": ["zero", "one"]})()
    expected = D.b1_aggregate(groups, Ctx()); actual = B.aggregate_group("toy", "ridge", reps, 2, ["zero", "one"])
    assert D.assert_numeric(expected, actual) > 170
    assert D.assert_numeric(D.b1_aggregate(groups, Ctx(), True), actual["fold_sensitivity"]) > 170
    broken = copy.deepcopy(actual); broken["summary"][0]["n_rejected"] += 1
    with pytest.raises(V.VerificationError, match="n_rejected"):
        D.assert_numeric(expected, broken)


def test_a_full_raw_prediction_roundtrip_and_reference_mutations(tmp_path):
    import json
    from v3.deploy import enroll as A
    from v3.deploy.tests import test_enroll as F
    man = F.make_manifest(); tables = F.make_tables(man); plan = F.make_plan(man, tables)
    features = F.make_features(man)
    manifests = tmp_path / "manifests"; manifests.mkdir()
    manifest_path = manifests / "cremad_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(man.values())))); w.writeheader(); w.writerows(man.values())
    v2root = tmp_path / "v2"; (v2root / "plan_rc2" / "splits").mkdir(parents=True)
    index = {}
    for r, table in tables.items():
        file = v2root / "plan_rc2" / "splits" / f"toy_r{r}.json"
        file.write_text(json.dumps(table), encoding="utf-8")
        index[f"toy_r{r}"] = {"path": f"splits/toy_r{r}.json", "sha256": common.sha256_file(file)}
    index_path = tmp_path / "index.json"; index_path.write_text(json.dumps(index), encoding="utf-8")
    for unit in plan["units"]:
        unit["split_sha256"] = index[unit["split_key"]]["sha256"]
        unit["unit_id"] = common.digest({k.split("{")[0]: unit[k.split("{")[0]] for k in plan["config"]["unit_id_fields"]})
    plan = A.seal_plan(plan)
    out = tmp_path / "work" / "A"
    A.run_units(plan, plan["units"], out, {"toy": man}, {("toy", "encA"): lambda ps: np.asarray([features[p] for p in ps])}, log=lambda *_: None)
    result = A.score_run(plan, out, tmp_path / "results")
    ctx = V.Context(V.Sources(manifests, index_path, v2root, tmp_path / "unused.csv", {"toy": "cremad"}))
    report = V.Report(); acc, seen = V.verify_a_integrity(V.normalise_plan_a(plan), tmp_path / "work", ctx, report)
    assert not report.failed(), report.failures
    count = D.verify_a_results(plan, acc, seen, ctx, result, V.read_csv_rows(tmp_path / "results" / "per_speaker.csv"))
    assert count > 500
    unit = V.normalise_plan_a(plan)["units"][0]
    records = A.reference_records(plan, plan["units"][0])
    bad = copy.deepcopy(records); bad[0]["source_speaker"] = "s999"
    report = V.Report(); V.verify_reference_records(plan, unit, bad, ctx.manifest("toy"), report)
    assert report.failed() and any(r["check"] == "reference:draw_replay" for r in report.failures)
    broken = copy.deepcopy(result); broken["endpoints"] = broken["endpoints"][1:]
    with pytest.raises(V.VerificationError, match="missing endpoint"):
        D.verify_a_results(plan, acc, seen, ctx, broken, V.read_csv_rows(tmp_path / "results" / "per_speaker.csv"))


def test_wavlm_seed_precedes_head_and_test_truth_never_read(monkeypatch):
    """Tiny fake pretrained encoder exercises the real FT loop without downloads."""
    import sys
    import types
    import torch
    from v3.deploy import engines_deploy as E
    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.scale = torch.nn.Parameter(torch.ones(()))
        def forward(self, x): return x*self.scale
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = torch.nn.Module(); self.encoder.transformer = torch.nn.Module()
            self.encoder.transformer.layers = torch.nn.ModuleList([Scale() for _ in range(6)])
        def extract_features(self, x):
            values = x[:, :2, None].expand(-1, -1, 768)
            for layer in self.encoder.transformer.layers: values = layer(values)
            return [values], None
    fake = types.SimpleNamespace(pipelines=types.SimpleNamespace(WAVLM_BASE_PLUS=types.SimpleNamespace(get_model=Encoder)))
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    class Waves:
        def __init__(self, *a, **k): pass
        def __call__(self, path): return np.linspace(-.2, .2, 32, dtype=np.float16)+int(path[-1])*.01
        def __len__(self): return 10
    monkeypatch.setattr(E, "WaveCache", Waves)
    fold = {"fit": ["fit0", "fit1", "fit2", "fit3"], "val": ["val0", "val1"], "test": ["test0", "test1"]}
    class Data:
        n_classes = 2; audio_root = Path("unused")
        def labels(self, paths):
            assert not any(p.startswith("test") for p in paths)
            return np.asarray([int(p[-1]) % 2 for p in paths])
    cfg = dict(engine="wavlm_partial_ft", crop_seconds=.001, eval_cap_seconds=.002, lr_encoder=.0001,
               lr_head=.001, weight_decay=.0001, batch_size=2, epochs=2, patience=2, fp16=False)
    torch.manual_seed(3)
    a = E.engine_wavlm({"train_seed": 97}, cfg, fold, Data(), torch.device("cpu"), check_weights=False)
    torch.manual_seed(999); torch.rand(17)
    b = E.engine_wavlm({"train_seed": 97}, cfg, fold, Data(), torch.device("cpu"), check_weights=False)
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
    assert a[3]["trainable_top_layers"] == 4
    assert "_checkpoint_state" in a[3]


def test_fixed_aggregate_independent_roundtrip_and_missing_repeat_detection():
    from v3.deploy import calib_fixed as F
    scored = []
    for repeat in (0, 1):
        seen = pred([.6, .7, .8, .9], [0, 0, 1, 0])
        novel = pred([.7, .8, .9, .95], [1, 0, 0, 0])
        test = pred([.55, .66, .88, .99], [0, 1, repeat, 0], ["a", "a", "b", "b"])
        preds = {k: dict(p, scores=p["logits"]) for k, p in {"seen": seen, "new": novel, "test": test}.items()}
        actual = F.score_unit(preds); expected = D.fixed_unit(preds)
        assert D.assert_numeric(expected, actual) > 30
        scored.append(dict(actual, level="toy", model="ridge", r=repeat, fold=0, unit_id=str(repeat)))
    assert D.assert_numeric(D.fixed_endpoints(scored), F.aggregate(scored)) > 24
    broken = copy.deepcopy(scored); del broken[1]["seen"]["speakers"]["a"]
    with pytest.raises(V.VerificationError, match="incomplete"):
        D.fixed_endpoints(broken)


def test_managed_b2_location_and_duplicate_copy_are_unambiguous(tmp_path):
    managed = tmp_path / "B2_gpu" / "units" / "u"; managed.mkdir(parents=True)
    assert V.b2_unit_path(tmp_path, "u", "cnn") == managed
    (tmp_path / "B2" / "units" / "u").mkdir(parents=True)
    with pytest.raises(V.VerificationError, match="duplicate"):
        V.b2_unit_path(tmp_path, "u", "cnn")

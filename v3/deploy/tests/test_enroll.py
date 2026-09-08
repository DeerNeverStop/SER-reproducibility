"""Block A (enrollment composition) unit tests on a tiny synthetic corpus; CPU only, no real caches."""
from __future__ import annotations

import copy
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from v3.deploy import common as C  # noqa: E402
from v3.deploy import enroll as A  # noqa: E402

CLASSES = ["angry", "happy", "neutral", "sad"]
SPEAKERS = [f"s{i}" for i in range(8)]
SENTENCES = list("ABCDEF")
LEVEL, BASE = "toy", "cremad"
D = 12


def make_manifest():
    man = {}
    for s in SPEAKERS:
        for sent in SENTENCES:
            for ci, cls in enumerate(CLASSES):
                for take in range(1 if cls == "neutral" else 2):
                    p = f"{s}_{sent}_{cls}_{take}.wav"
                    man[p] = {"relative_path": p, "speaker": s, "label": cls, "label_index": ci, "sentence": sent, "sha256": ""}
    return man


def make_features(man):
    rng = np.random.default_rng(7)
    delta = rng.normal(0, 1.5, size=(len(CLASSES), D))
    mu = {s: rng.normal(0, 2.0, size=D) for s in SPEAKERS}
    sc = {s: rng.uniform(0.5, 1.5, size=D) for s in SPEAKERS}
    feats = {}
    for p, r in man.items():
        feats[p] = mu[r["speaker"]] + sc[r["speaker"]] * (delta[r["label_index"]] + rng.normal(0, 0.7, size=D))
    return feats


def make_tables(man):
    pop = sorted(man)
    by_spk = defaultdict(list)
    for p in pop:
        by_spk[man[p]["speaker"]].append(p)
    layout = {0: [["s0", "s1", "s2", "s3"], ["s4", "s5", "s6", "s7"]],
              1: [["s0", "s2", "s4", "s6"], ["s1", "s3", "s5", "s7"]]}
    tables = {}
    for r, folds in layout.items():
        out = []
        for k, test in enumerate(folds):
            rest = [s for s in SPEAKERS if s not in test]
            val, fit = rest[:1], rest[1:]
            out.append({"fold": k, "meta": {}, "test": [p for s in test for p in by_spk[s]],
                        "val": [p for s in val for p in by_spk[s]], "fit": [p for s in fit for p in by_spk[s]]})
        tables[r] = {"key": f"toy_r{r}", "population": pop, "folds": out}
    return tables


def make_plan(man, tables):
    lp = A.build_level_plan(LEVEL, BASE, man, tables, CLASSES, rule=("hash", 3))
    encs = ["encA"]
    units = A.build_units({LEVEL: lp}, encs, {LEVEL: {0: "toy_r0", 1: "toy_r1"}},
                          {LEVEL: {0: "0" * 64, 1: "1" * 64}}, {LEVEL: {"encA": "f" * 64}})
    plan = {"program": C.PROGRAM, "module": "A", "config": copy.deepcopy(A.CONFIG), "encoders": encs,
            "levels": {LEVEL: lp}, "units": units}
    return A.seal_plan(plan)


@pytest.fixture(scope="module")
def toy():
    man = make_manifest()
    feats = make_features(man)
    plan = make_plan(man, make_tables(man))
    x_of = lambda paths: np.asarray([feats[p] for p in paths], dtype=np.float64)  # noqa: E731
    return {"man": man, "feats": feats, "plan": plan, "x_of": x_of, "lp": plan["levels"][LEVEL]}


@pytest.fixture(scope="module")
def full_run(toy, tmp_path_factory):
    out = tmp_path_factory.mktemp("runA")
    counts = A.run_units(toy["plan"], toy["plan"]["units"], out, {LEVEL: toy["man"]},
                         {(LEVEL, "encA"): toy["x_of"]}, log=lambda *_: None)
    assert counts == {"done": len(toy["plan"]["units"])}
    return out


def cond(lp, cid):
    return next(c for c in lp["conditions"] if c["id"] == cid)


# ------------------------------------------------------------------ sentence split & feasibility

def test_sentence_split_disjoint_and_frozen():
    cremad = ["DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI"]
    s = A.sentence_split("cremad", cremad)
    assert not set(s["Q"]) & set(s["E"]) and set(s["Q"]) | set(s["E"]) == set(cremad)
    assert len(s["Q"]) == 5 and s["Q"] == ["IWW", "WSI", "ITS", "TIE", "TSI"]
    s = A.sentence_split("subesco", ["S1", "S10", "S2", "S3", "S6", "S8", "S9"])
    assert s["Q"] == ["S9", "S1", "S8"] and len(s["E"]) == 4
    s = A.sentence_split("ravdess", ["S1", "S2"])
    assert s["Q"] == ["S1"] and s["E"] == ["S2"]
    with pytest.raises(C.IntegrityError):
        A.sentence_split("ravdess", ["S1", "S2", "S3"])


def test_plan_q_e_disjoint_and_counts(toy):
    lp = toy["lp"]
    q_sents = set(lp["sentences"]["Q"])
    for s, entry in lp["per_speaker"].items():
        assert {toy["man"][p]["sentence"] for p in entry["Q"]} <= q_sents
        e_paths = [p for c in CLASSES for p in entry["E_by_class"][c]]
        assert not ({toy["man"][p]["sentence"] for p in e_paths} & q_sents)
        assert not set(entry["Q"]) & set(e_paths)
        assert entry["E_class_counts"] == {"angry": 6, "happy": 6, "neutral": 3, "sad": 6}
        assert entry["n_Q"] == 21
    assert lp["q_all_classes_every_speaker"] is True
    assert lp["neutral_index"] == 2


def test_feasibility_refuses_insufficient_pools(toy):
    lp = toy["lp"]
    assert cond(lp, "balanced@5")["feasible"] and cond(lp, "balanced@10")["feasible"]
    c20 = cond(lp, "balanced@20")
    assert not c20["feasible"] and "neutral" in c20["reason"] and "needs 5" in c20["reason"]
    assert cond(lp, "single:neutral@3")["feasible"]
    assert not cond(lp, "single:neutral@5")["feasible"]
    assert cond(lp, "single:angry@5")["feasible"]
    assert cond(lp, "other_balanced@5")["feasible"]
    assert not cond(lp, "other_single:neutral@5")["feasible"]
    assert cond(lp, "oracle_all")["feasible"] and cond(lp, "oracle_all")["estimators"] == ["naive"]
    assert cond(lp, "none")["estimators"] == ["none"]
    # a level whose folds hold a single test speaker gets no other_* conditions
    conds = A.build_conditions(CLASSES, lp["class_order"], {s: lp["per_speaker"][s]["E_class_counts"] for s in SPEAKERS}, 1)
    assert all(not c["feasible"] for c in conds if c["composition"].startswith("other_"))
    assert A.feasibility_reason({"a": {"x": 2}}, {"x": 3}) == "speaker a: class x has 2 in E, needs 3"
    assert A.feasibility_reason({"a": {"x": 3}}, {"x": 3}) is None
    with pytest.raises(C.IntegrityError):
        A.draw_references(lp, LEVEL, 0, 0, "s0", c20, 0)


def test_unit_ids_and_pairs(toy):
    plan = toy["plan"]
    assert len(plan["units"]) == 4 and len({u["unit_id"] for u in plan["units"]}) == 4
    u = plan["units"][0]
    assert u["unit_id"] == C.digest({k: v for k, v in u.items() if k not in ("unit_id", "test_speakers", "n_test_speakers")})
    for r_key, folds in toy["lp"]["pairs"].items():
        for f_key, pairs in folds.items():
            spk = toy["lp"]["fold_test_speakers"][r_key][f_key]
            assert set(pairs) == set(spk) and set(pairs.values()) == set(spk)
            assert all(a != b for a, b in pairs.items())
    assert A.pair_speakers(["x"], "salt") == {}


# ------------------------------------------------------------------ draws

def test_draws_reproducible(toy):
    lp = toy["lp"]
    c = cond(lp, "balanced@10")
    a = A.draw_references(lp, LEVEL, 0, 0, "s0", c, 0)
    b = A.draw_references(lp, LEVEL, 0, 0, "s0", c, 0)
    assert a == b and len(a) == 10 and len(set(a)) == 10
    assert A.draw_references(lp, LEVEL, 0, 0, "s0", c, 1) != a
    assert A.draw_references(lp, LEVEL, 1, 0, "s0", c, 0) != a
    assert A.draw_seed(LEVEL, 0, "s0", "balanced", 10, 0) == int(C.digest([C.PROGRAM, LEVEL, 0, "s0", "balanced", 10, 0])[:16], 16)
    e_paths = {p for cls in CLASSES for p in lp["per_speaker"]["s0"]["E_by_class"][cls]}
    assert set(a) <= e_paths
    s = cond(lp, "single:sad@5")
    d = A.draw_references(lp, LEVEL, 0, 0, "s0", s, 3)
    assert len(d) == 5 and all(toy["man"][p]["label"] == "sad" and toy["man"][p]["speaker"] == "s0" for p in d)
    o = cond(lp, "other_single:sad@5")
    partner = lp["pairs"]["r0"]["f0"]["s0"]
    d = A.draw_references(lp, LEVEL, 0, 0, "s0", o, 0)
    assert all(toy["man"][p]["speaker"] == partner for p in d) and partner != "s0"
    allp = A.draw_references(lp, LEVEL, 0, 0, "s0", cond(lp, "oracle_all"), 0)
    assert len(allp) == 42 and set(lp["per_speaker"]["s0"]["Q"]) <= set(allp)


def test_balanced_draw_class_coverage(toy):
    lp = toy["lp"]
    order = lp["class_order"]
    for n in (5, 10):
        for draw in range(5):
            refs = A.draw_references(lp, LEVEL, 0, 1, "s5", cond(lp, f"balanced@{n}"), draw)
            got = Counter(toy["man"][p]["label"] for p in refs)
            assert got == Counter(A.balanced_need(order, n))
    assert A.balanced_need(order, 5) == {order[0]: 2, order[1]: 1, order[2]: 1, order[3]: 1}


# ------------------------------------------------------------------ estimators

def test_shrink_formula():
    m_ref, v_ref = np.array([1.0, 2.0]), np.array([4.0, 9.0])
    m_g, v_g = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    m, v = A.shrink_stats(m_ref, v_ref, 4, m_g, v_g, 8)
    assert np.allclose(m, [4.0 / 12, 8.0 / 12]) and np.allclose(v, [(16 + 8) / 12, (36 + 8) / 12])
    m, v = A.shrink_stats(m_ref, v_ref, 4, m_g, v_g, 0)
    assert np.allclose(m, m_ref) and np.allclose(v, v_ref)


def fit_toy_model(toy, unit_index=0):
    plan, lp = toy["plan"], toy["lp"]
    u = plan["units"][unit_index]
    paths = A.unit_train_paths(lp, u["test_speakers"])
    return A.fit_unit_model(toy["x_of"](paths), C.labels_for(toy["man"], paths), C.speakers_for(toy["man"], paths), len(CLASSES)), u


def test_em_prior_and_prior_corrected(toy):
    model, u = fit_toy_model(toy)
    prior0 = model.prior
    assert abs(prior0.sum() - 1) < 1e-12
    # near one-hot posteriors for class 0 -> EM prior concentrates on class 0
    P = np.full((30, 4), 0.02)
    P[:, 0] = 0.94
    pi, it = A.em_prior(P, prior0, 50, 1e-8)
    assert abs(pi.sum() - 1) < 1e-12 and pi[0] > 0.9 and 1 <= it <= 50
    # posteriors equal to the training prior keep the prior unchanged
    pi, _ = A.em_prior(np.tile(prior0, (7, 1)), prior0)
    assert np.allclose(pi, prior0)
    # deltas zero -> E4 mean equals E1 mean; with k=0 the predictions coincide exactly
    lp = toy["lp"]
    spk = u["test_speakers"][0]
    refs = A.draw_references(lp, LEVEL, u["r"], u["fold"], spk, cond(lp, "balanced@10"), 0)
    x_ref, x_q = toy["x_of"](refs), toy["x_of"](lp["per_speaker"][spk]["Q"])
    zero = copy.copy(model)
    zero.delta = np.zeros_like(model.delta)
    cfg0 = copy.deepcopy(A.CONFIG)
    cfg0["shrink_k"] = 0
    m1, v1, _ = A.estimator_stats(zero, "naive", x_ref, lp["neutral_index"], cfg0)
    m4, v4, info = A.estimator_stats(zero, "prior_corrected", x_ref, lp["neutral_index"], cfg0)
    assert np.allclose(m1, m4) and np.allclose(v1, v4) and abs(info["pi"].sum() - 1) < 1e-12
    p1 = zero.predict_norm(A.zscore(x_q, m1, v1, 1e-3))
    p4 = zero.predict_norm(A.zscore(x_q, m4, v4, 1e-3))
    assert np.array_equal(p1, p4)
    # with real deltas and balanced references the correction is small relative to a single-class shift
    m4b, _, _ = A.estimator_stats(model, "prior_corrected", x_ref, lp["neutral_index"], cfg0)
    single = A.draw_references(lp, LEVEL, u["r"], u["fold"], spk, cond(lp, "single:angry@5"), 0)
    m1s, _, _ = A.estimator_stats(model, "naive", toy["x_of"](single), lp["neutral_index"], cfg0)
    m4s, _, _ = A.estimator_stats(model, "prior_corrected", toy["x_of"](single), lp["neutral_index"], cfg0)
    assert np.linalg.norm(m4b - m1) < np.linalg.norm(m4s - m1s)


def test_neutral_filter_selection(toy):
    P = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.2], [0.2, 0.2, 0.5, 0.1], [0.1, 0.1, 0.4, 0.4]])
    keep, fb = A.select_neutral(P, 2, 2)
    assert keep.tolist() == [1, 2, 3] and not fb          # row 3: argmax tie -> first max (index 2)
    P[3, 3] = 0.5
    P[2, 0] = 0.6
    keep, fb = A.select_neutral(P, 2, 2)
    assert keep.tolist() == [1, 2] and fb                 # only one predicted neutral -> top-2 by neutral posterior
    model, u = fit_toy_model(toy)
    lp = toy["lp"]
    spk = u["test_speakers"][1]
    x_ref = toy["x_of"](A.draw_references(lp, LEVEL, u["r"], u["fold"], spk, cond(lp, "single:happy@5"), 2))
    m, v, info = A.neutral_filter_stats(model, x_ref, lp["neutral_index"])
    assert len(info["kept"]) == 2 and all(k >= 2 for k in info["kept"]) and m.shape == (D,) and (v > 0).all()
    m2, v2 = A.shrink_stats(*A.naive_stats(x_ref), 5, model.m_glob, model.v_glob, 8)
    assert np.allclose(v, v2)
    cfg = copy.deepcopy(A.CONFIG)
    cfg["neutral_filter"]["round2_normalisation"] = "naive"
    m_alt, v_alt, info_alt = A.neutral_filter_stats(model, x_ref, lp["neutral_index"], cfg)
    assert np.allclose(v_alt, v) and info_alt["kept"][0] == info["kept"][0] and info_alt["fallback"][0] == info["fallback"][0]
    cfg["neutral_filter"]["round2_normalisation"] = "bogus"
    with pytest.raises(C.IntegrityError):
        A.neutral_filter_stats(model, x_ref, lp["neutral_index"], cfg)


# ------------------------------------------------------------------ run outputs

def test_run_rows_q_identical_across_conditions(toy, full_run):
    plan, lp = toy["plan"], toy["lp"]
    u = plan["units"][0]
    unit_dir = full_run / "units" / u["unit_id"]
    assert A.done_verified(unit_dir)
    groups = A.read_unit_predictions(unit_dir)
    for (cid, est, draw, spk), g in groups.items():
        assert sorted(g["paths"]) == sorted(lp["per_speaker"][spk]["Q"])
        assert est in cond(lp, cid)["estimators"] and draw < cond(lp, cid)["draws"]
    n_sets = sum(len(c["estimators"]) * c["draws"] for c in lp["conditions"] if c["feasible"])
    assert len(groups) == n_sets * len(u["test_speakers"])
    model = C.read_json(unit_dir / "model.json")
    assert model["rows"] == n_sets * len(u["test_speakers"]) * 21 and model["predictions_sha256"] == (unit_dir / "DONE").read_text().split()[0]
    with gzip.open(unit_dir / "predictions.csv.gz", "rt") as fh:
        assert fh.readline().strip() == ",".join(A.PRED_COLUMNS)
    # deterministic bytes: recomputing the unit reproduces the same sha
    rows, _ = A.compute_unit(plan, u, toy["man"], toy["x_of"])
    import hashlib
    assert hashlib.sha256(A.predictions_bytes(rows)).hexdigest() == model["predictions_sha256"]
    # rerun skips verified units and logs it
    counts = A.run_units(plan, plan["units"], full_run, {LEVEL: toy["man"]}, {(LEVEL, "encA"): toy["x_of"]}, log=lambda *_: None)
    assert counts == {"skipped": len(plan["units"])}
    events = [C.canonical(e) for e in map(__import__("json").loads, (full_run / "ledger.jsonl").read_text().splitlines())]
    assert any('"event":"skip"' in e for e in events) and any('"event":"done"' in e for e in events)


# ------------------------------------------------------------------ score

def test_nest_mean_order():
    assert A.nest_mean({0: [1.0, 1.0, 4.0], 1: [10.0]}) == pytest.approx(6.0)     # not the flat mean 4.0
    assert A.nest_mean({1: [10.0], 0: [2.0]}) == pytest.approx(6.0)
    v = A.nest_mean_vec({0: [[0.0, 2.0], [2.0, 2.0]], 1: [[4.0, 0.0]]})
    assert np.allclose(v, [2.5, 1.0])
    with pytest.raises(C.IntegrityError):
        A.nest_mean({0: []})


def test_score_refuses_incomplete_then_scores(toy, full_run, tmp_path):
    plan = toy["plan"]
    u = plan["units"][-1]
    done = full_run / "units" / u["unit_id"] / "DONE"
    original = done.read_text()
    done.write_text("0" * 64 + "\n")
    with pytest.raises(C.IntegrityError, match="verified DONE"):
        A.score_run(plan, full_run, tmp_path / "bad")
    done.unlink()
    with pytest.raises(C.IntegrityError, match="verified DONE"):
        A.score_run(plan, full_run, tmp_path / "bad")
    done.write_text(original)
    res = A.score_run(plan, full_run, tmp_path / "res")
    for name in ("per_speaker.csv", "endpoints.json", "harm_share.csv", "recall.csv"):
        assert (tmp_path / "res" / name).is_file()
    lp = toy["lp"]
    feasible_pairs = sum(len(c["estimators"]) for c in lp["conditions"] if c["feasible"])
    lines = (tmp_path / "res" / "per_speaker.csv").read_text().splitlines()
    assert lines[0] == "level,enc,condition,estimator,speaker,uar" and len(lines) - 1 == feasible_pairs * len(SPEAKERS)
    eps = {e["endpoint"] for e in res["endpoints"]}
    assert eps == {1, 2, 3, 4, 5, 6, 7}
    ep3 = [e for e in res["endpoints"] if e["endpoint"] == 3]
    ep3_5 = [e for e in ep3 if e["N"] == 5]
    assert ep3_5 and ep3_5[0]["classes"] == ["angry", "happy", "sad"] and ep3_5[0].get("primary") is False
    assert any(e["N"] == 3 for e in ep3)
    assert all(e["N"] == 3 for e in res["endpoints"] if e.get("primary"))
    identities = {e["id"]: e for e in res["endpoints"]}
    assert len(identities) == len(plan["endpoint_manifest"])
    for identity in plan["endpoint_manifest"]:
        actual = identities[identity["id"]]
        for key, value in identity.items():
            assert actual.get(key, {"available": True, "primary": False}.get(key)) == value
    matched = next(e for e in ep3 if e["condition"] == "single:balanced_classes@3")
    assert matched["classes"] == [c for c in CLASSES if c in cond(lp, "balanced@3")["need"]]
    assert matched["primary"] is False
    for e in res["endpoints"]:
        if e.get("available", True):
            assert e["n"] == len(SPEAKERS) and e["ci95"][0] <= e["estimate"] <= e["ci95"][1]
    assert len(res["absolute"]) == feasible_pairs
    assert res["feasibility"][LEVEL]["balanced@20"]["feasible"] is False
    assert res["inputs"]["n_units"] == 4
    # per-speaker values agree with an independent nested mean over the raw rows
    ps = {tuple(l.split(",")[:5]): float(l.split(",")[5]) for l in lines[1:]}
    raw = defaultdict(lambda: defaultdict(list))
    for unit in plan["units"]:
        for (cid, est, draw, spk), g in A.read_unit_predictions(full_run / "units" / unit["unit_id"]).items():
            raw[(cid, est, spk)][unit["r"]].append(C.uar(g["y_true"], g["y_pred"], len(CLASSES)))
    for (cid, est, spk), by_r in raw.items():
        assert ps[(LEVEL, "encA", cid, est, spk)] == pytest.approx(np.mean([np.mean(v) for v in by_r.values()]))


def test_shared_donor_samples_and_replay_receipt(toy, full_run):
    plan, lp = toy["plan"], toy["lp"]
    for unit in plan["units"]:
        unit_dir = full_run / "units" / unit["unit_id"]
        with gzip.open(unit_dir / "references.json.gz", "rt", encoding="utf-8") as fh:
            receipt = __import__("json").load(fh)
        assert receipt["records"] == A.reference_records(plan, unit)
        assert receipt["plan_sha256"] == plan["plan_sha256"]
        records = {(r["speaker"], r["condition"], r["draw"]): r for r in receipt["records"]}
        for rec in records.values():
            paths, source = rec["paths"], rec["source_speaker"]
            assert len(paths) == len(set(paths))
            assert {toy["man"][p]["speaker"] for p in paths} == {source}
            if rec["condition"] == "oracle_all":
                assert set(lp["per_speaker"][source]["Q"]) <= set(paths)
            else:
                assert not set(paths) & set(lp["per_speaker"][source]["Q"])
                assert len(paths) == cond(lp, rec["condition"])["N"]
            if rec["condition"].startswith("other_"):
                assert source != rec["speaker"]
                own = records[(source, rec["condition"].removeprefix("other_"), rec["draw"])]
                assert paths == own["paths"]


def test_reference_labels_do_not_enter_estimation(toy):
    plan, unit = toy["plan"], toy["plan"]["units"][0]
    clean, _ = A.compute_unit(plan, unit, toy["man"], toy["x_of"])
    polluted = copy.deepcopy(toy["man"])
    for speaker in unit["test_speakers"]:
        for pool in toy["lp"]["per_speaker"][speaker]["E_by_class"].values():
            for path in pool:
                polluted[path]["label"] = "must_not_be_read"
                polluted[path]["label_index"] = 999
    altered, _ = A.compute_unit(plan, unit, polluted, toy["x_of"])
    assert clean == altered


def test_receipt_tampering_rejected_even_after_rehash(toy, full_run, tmp_path):
    import shutil
    run = tmp_path / "copy"
    shutil.copytree(full_run, run)
    udir = run / "units" / toy["plan"]["units"][0]["unit_id"]
    ref_path = udir / "references.json.gz"
    with gzip.open(ref_path, "rt", encoding="utf-8") as fh:
        receipt = __import__("json").load(fh)
    rec = next(r for r in receipt["records"] if r["condition"].startswith("other_"))
    rec["source_speaker"] = rec["speaker"]
    ref_path.write_bytes(gzip.compress(C.canonical(receipt).encode(), mtime=0))
    assert A.done_verified(udir) is False
    model = C.read_json(udir / "model.json")
    model["references_sha256"] = C.sha256_file(ref_path)
    C.atomic_write_json(udir / "model.json", model)
    C.atomic_write_text(udir / "DONE", model["predictions_sha256"] + " " + C.sha256_file(udir / "model.json") + "\n")
    assert A.done_verified(udir)
    with pytest.raises(C.IntegrityError, match="reference receipt"):
        A.verify_run_complete(toy["plan"], run)


def test_main_harm_thresholds_after_emotion_average(toy, monkeypatch, tmp_path):
    plan, lp = toy["plan"], toy["lp"]
    uar, recalls = {}, {}
    for c in lp["conditions"]:
        if not c["feasible"]:
            continue
        for est in c["estimators"]:
            # Mean over four single emotions: (70+40+40+40)/4 = 47.5.
            # All speakers are harmed by >2 points vs 50, not 75% of speakers.
            value = (70 if c["composition"] == "single:angry" else 40) if c["composition"].startswith("single:") else 50
            key = (LEVEL, "encA", c["id"], est)
            uar[key] = {s: float(value) for s in lp["speakers"]}
            recalls[key] = {s: np.full(len(CLASSES), value, dtype=float) for s in lp["speakers"]}
    monkeypatch.setattr(A, "verify_run_complete", lambda *_: {})
    monkeypatch.setattr(A, "collect_per_speaker", lambda *_: (uar, recalls))
    result = A.score_run(plan, tmp_path / "run", tmp_path / "res")
    main = next(e for e in result["endpoints"] if e["endpoint"] == 7 and e.get("primary"))
    assert main["estimate"] == 1.0
    assert main["ci95"] == [1.0, 1.0]
    recall = [e for e in result["endpoints"] if e["endpoint"] == 4]
    assert any(e.get("N") == 5 and e.get("target_class") for e in recall)
    assert next(e for e in recall if e.get("primary"))["estimate"] == -2.5


def test_missing_query_class_rejected_before_fitting():
    man = make_manifest()
    # A fixed partition keeps the missing class specifically in Q.
    for path in [p for p, r in man.items() if r["speaker"] == "s0" and r["sentence"] in "ABC" and r["label"] == "neutral"]:
        del man[path]
    with pytest.raises(C.IntegrityError, match="every fixed evaluation class"):
        A.build_level_plan(LEVEL, BASE, man, make_tables(man), CLASSES, rule=("fixed", list("ABC"), list("DEF")))


def test_em_numerical_extremes_and_invalid_input():
    pi, iterations = A.em_prior(np.asarray([[1.0, 0.0], [1.0, 0.0]]), np.asarray([0.5, 0.5]))
    assert np.isfinite(pi).all() and pi.sum() == pytest.approx(1.0) and iterations <= 50
    for bad in (np.asarray([[0.0, 0.0]]), np.asarray([[float("nan"), 1.0]]), np.asarray([[-0.1, 1.1]])):
        with pytest.raises(C.IntegrityError, match="pseudo-posteriors"):
            A.em_prior(bad, np.asarray([0.5, 0.5]))


def test_plan_mutation_refused_before_run(toy, tmp_path):
    changed = copy.deepcopy(toy["plan"])
    changed["config"]["primary_N"] = 5
    with pytest.raises(C.IntegrityError, match="plan digest"):
        A.run_units(changed, changed["units"], tmp_path, {}, {}, log=lambda *_: None)

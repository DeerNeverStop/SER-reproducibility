"""Adversarial coverage-plan tests; synthetic fixtures are never formal gates."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from v3.speaker_coverage import plan as compiler
from v3.speaker_coverage import verify as v


@pytest.fixture(scope="module")
def fixture():
    speakers = [str(1001 + i) for i in range(60)]
    prompts = ["DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI"]
    names = ("angry", "disgust", "fearful", "happy", "neutral", "sad")
    codes = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
    sex = {s: "Female" if i < 30 else "Male" for i, s in enumerate(speakers)}
    rng = np.random.default_rng(39072)
    rows, cells, embeddings = [], {}, {}
    for person in speakers:
        center = rng.normal(size=8)
        for prompt in prompts:
            for label, (name, code) in enumerate(zip(names, codes)):
                path = f"{person}_{prompt}_{code}_MD.wav"
                row = {"relative_path": path, "corpus": "cremad", "speaker": person,
                       "sentence": prompt, "label": name, "label_index": str(label),
                       "intensity": "MD", "sha256": v.digest(path), "bytes": "10"}
                rows.append(row)
                cells[person, prompt, label] = row
                if label == 4:
                    embeddings[path] = center + rng.normal(scale=0.05, size=8)
    meta = {"speakers": speakers, "prompts": prompts, "sex": sex, "cells": cells,
            "clean": rows, "raw": rows, "input": {"fixture": True}}
    plan, capacity, diagnostics = compiler.build_from_metadata(meta, embeddings, {"fixture": True})
    return plan, rows, sex, embeddings, capacity, diagnostics


def run(fixture):
    return v.verify_document(*fixture[:4], fixture[4], fixture[5], complete=False)


def reseal(plan, capacity, diagnostics):
    for unit in plan["units"]:
        unit["fit_manifest_sha256"] = v.digest(unit["fit"])
        unit["unit_id"] = v.digest({k: x for k, x in unit.items() if k != "unit_id"})
    plan["plan_sha256"] = v.digest({k: x for k, x in plan.items() if k != "plan_sha256"})
    capacity["plan_sha256"] = diagnostics["plan_sha256"] = plan["plan_sha256"]


def test_independent_complete_grid_fixture_never_allows_formal(fixture):
    result = run(fixture)
    assert result["pass"] and result["n_units"] == 720
    assert result["formal_allowed"] is False and result["scope"] == "test_fixture_only"
    assert result["policy_trajectories_checked"] == 150
    assert result["scientific_outcomes_read"] is False
    assert result["units_by_model"] == {"ridge_wavlm": 270, "cnn": 270, "wavlm_ft": 180}


@pytest.mark.parametrize("mutation,match", [
    ("count", "720"), ("duplicate", "duplicate"), ("config", "config"),
    ("seed", "seed"), ("labels", "label"), ("val", "fixed outer"),
    ("draw", "fixed shared slot graph|repeat identical|policy selection"), ("cell", "emotion cell|prompt|recordings"),
    ("reference", "reference metadata"), ("selection", "selected policy"),
    ("objective", "objective"), ("layout", "layout"), ("targetgate", "selection_queries"),
])
def test_rehashed_tampering_is_rejected(fixture, mutation, match):
    plan, rows, sex, embeddings, capacity, diag = fixture
    plan, capacity, diag = deepcopy(plan), deepcopy(capacity), deepcopy(diag)
    rows = deepcopy(rows)
    unit = plan["units"][0]
    if mutation == "count":
        plan["units"].pop()
    elif mutation == "duplicate":
        plan["units"][-1] = deepcopy(plan["units"][0])
    elif mutation == "config":
        unit["config"]["alpha"] = 9
    elif mutation == "seed":
        unit["train_seed"] += 1
    elif mutation == "labels":
        rows[0]["label"] = "sad"
    elif mutation == "val":
        unit["val"] = unit["val"][1:]
    elif mutation == "draw":
        partner = next(u for u in plan["units"] if u["model"] == unit["model"] and u["fold"] == 0
                       and u["rotation"] == 0 and u["draw"] == 1 and u["policy"] == unit["policy"])
        partner["fit"] = unit["fit"][:]
    elif mutation == "cell":
        path = unit["fit"][0]
        replacement = next(r["relative_path"] for r in rows if r["speaker"] == path.split("_")[0]
                           and r["relative_path"] not in unit["fit"])
        unit["fit"] = sorted([replacement] + unit["fit"][1:])
    elif mutation == "reference":
        c = capacity["conditions"][0]
        speaker = c["eligible_speakers"][0]
        c["reference_paths"][speaker][0] = unit["test"][0]
        c["reference_sha256"] = v.digest(c["reference_paths"])
    elif mutation == "selection":
        diag["conditions"][0]["selections"][0]["selected_speakers"].pop()
    elif mutation == "objective":
        diag["conditions"][0]["selections"][0]["mean_nn1"] += 0.1
    elif mutation == "layout":
        diag["conditions"][0]["layouts"][0]["abstract_slots"]["Female"][0][0] = 99
    elif mutation == "targetgate":
        plan["design"]["selection_queries"] = "test speakers"
    reseal(plan, capacity, diag)
    with pytest.raises(ValueError, match=match):
        v.verify_document(plan, rows, sex, embeddings, capacity, diag, complete=False)


def test_plan_and_unit_hashes_rejected(fixture):
    changed = deepcopy(fixture[0])
    changed["units"][0]["train_seed"] += 1
    with pytest.raises(ValueError, match="plan hash"):
        v.verify_document(changed, *fixture[1:4], fixture[4], fixture[5], complete=False)
    changed["plan_sha256"] = v.digest({k: x for k, x in changed.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="unit hash"):
        v.verify_document(changed, *fixture[1:4], fixture[4], fixture[5], complete=False)


def test_fixture_cannot_enter_complete_gate(fixture):
    with pytest.raises(ValueError, match="fixture"):
        v.verify_document(*fixture[:4], fixture[4], fixture[5], complete=True)


def test_independent_audio_checks_actual_bytes_and_size(tmp_path):
    path = tmp_path / "recording.wav"
    path.write_bytes(b"original")
    rows = [{"relative_path": path.name, "sha256": v.byte_hash(path), "bytes": str(path.stat().st_size)}]
    assert v.verify_audio(rows, tmp_path)["audio_files_verified"] == 1
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="byte hash"):
        v.verify_audio(rows, tmp_path)
    with pytest.raises(ValueError, match="audio-root"):
        v.verify_audio(rows, None)


def test_paths_duplicate_json_and_nonfinite_rejected(tmp_path):
    for name in ("../escape", "/absolute", "C:/absolute", "a/../b", "a\\b", "./x"):
        with pytest.raises(ValueError):
            v.local(tmp_path, name)
    for content in ('{"x":1,"x":2}', '{"x":NaN}'):
        path = tmp_path / "bad.json"
        path.write_text(content)
        with pytest.raises(ValueError):
            v.read_json(path)


@pytest.mark.parametrize("forbidden_operation", ["heldout", "iterate"])
def test_instrumented_compiler_rejects_target_reference_reads(fixture, monkeypatch, forbidden_operation):
    _, rows, sex, embeddings, capacity, _ = fixture
    forbidden_path = next(r["relative_path"] for r in rows
                          if r["speaker"] in capacity["conditions"][0]["test_speakers"]
                          and r["label_index"] == "4")

    def leaking_compiler(meta, mapping, fold, rotation):
        if forbidden_operation == "heldout":
            return mapping[forbidden_path]
        return list(mapping)

    monkeypatch.setattr(compiler, "plan_context", leaking_compiler)
    with pytest.raises(ValueError, match="nontraining reference|global embedding"):
        v.audit_reference_access(rows, sex, embeddings, capacity, complete=False)


def test_semantic_and_byte_hash_are_distinct(tmp_path):
    model = tmp_path / "weights.pth"
    model.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ValueError, match="byte hash"):
        v.verify_wavlm(model, {"wavlm_model_file": model.name, "wavlm_model_sha256": "a" * 64})


@pytest.mark.parametrize("corruption", [None, "nonfinite", "paths", "shape"])
def test_feature_gate_checks_payload_not_just_a_self_consistent_hash(tmp_path, corruption):
    import json
    inp = {"feature_files": {}, "feature_sha256": {}}
    for kind, shape, dtype in (("logmel", (1, 64, 128), "float32"),
                               ("wavlm_base_plus", (1, 13, 768), "float16")):
        values = np.zeros(shape, dtype=dtype)
        paths = np.array(["recording.wav"])
        if kind == "logmel":
            if corruption == "nonfinite":
                values.flat[0] = np.nan
            elif corruption == "paths":
                paths = np.array(["wrong.wav"])
            elif corruption == "shape":
                values = values[:, :, :1]
        path = tmp_path / (kind + ".npz")
        np.savez_compressed(path, X=values, paths=paths)
        inp["feature_files"][kind] = path.name
        inp["feature_sha256"][kind] = v.byte_hash(path)
        path.with_suffix(".json").write_text(json.dumps({
            "corpus": "cremad", "encoder": kind, "sha256": v.byte_hash(path), "weights_sha256": v.WEIGHTS_SHA}))
    if corruption:
        with pytest.raises(ValueError, match="nonfinite|population|shape"):
            v.verify_features(tmp_path, inp, ["recording.wav"])
    else:
        v.verify_features(tmp_path, inp, ["recording.wav"])


def test_cli_fails_closed_without_input_files(tmp_path, capsys):
    args = [part for key in ("repo", "plan", "features", "asv", "wavlm-model", "audio-root")
            for part in ("--" + key, str(tmp_path / "missing"))]
    assert v.main(args) == 1
    report = __import__("json").loads(capsys.readouterr().out)
    assert not report["pass"] and not report["formal_allowed"]

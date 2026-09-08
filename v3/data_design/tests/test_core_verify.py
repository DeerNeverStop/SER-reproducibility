"""Metadata-only fault injections; no fitted model or scientific result."""
import copy
import hashlib
import json

import pytest

from v3.data_design import core_verify as verify


EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
PROMPTS = tuple("AA" + chr(ord("A") + i) for i in range(12))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def reseal(plan):
    for unit in plan["units"]:
        payload = {k: v for k, v in unit.items() if k != "unit_id"}
        unit["unit_id"] = sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    plan["plan_sha256"] = sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return plan


@pytest.fixture(scope="module")
def source_fixture():
    female = [str(1000 + i) for i in range(30)]
    male = [str(1030 + i) for i in range(30)]
    sex = {s: "Female" for s in female} | {s: "Male" for s in male}
    rows = []

    def add_row(speaker, prompt, label, intensity):
        path = f"{speaker}_{prompt}_{EMOTIONS[label]}_{intensity}.wav"
        rows.append({"relative_path": path, "sha256": sha(path), "corpus": "cremad",
                     "speaker": speaker, "sentence": prompt, "label_index": str(label), "intensity": intensity})
        return path

    paths = {}
    for speaker in sex:
        for prompt in PROMPTS:
            for label in range(6):
                intensity = "MD" if prompt == PROMPTS[0] else "XX"
                paths[(speaker, prompt, label)] = add_row(speaker, prompt, label, intensity)
    # A genuine preference decision: MD, XX and HI coexist for the same cell.
    add_row("1000", PROMPTS[0], 0, "XX")
    add_row("1000", PROMPTS[0], 0, "HI")
    val_people = female[24:28] + male[24:28]
    test_people = female[28:30] + male[28:30]
    common = PROMPTS[:6]
    test_prompts = PROMPTS[6:8]
    replacement = PROMPTS[8:10]
    val = sorted(paths[(s, p, c)] for s in val_people for p in common for c in range(6))
    test = sorted(paths[(s, p, c)] for s in test_people for p in test_prompts for c in range(6))
    units = []
    for B in (288, 576):
        for S in (12, 48):
            k = B // (6 * S)
            for scenario, prompts in (("prompt_seen", common + test_prompts), ("prompt_new", common + replacement)):
                fit = []
                for people in (female[:S // 2], male[:S // 2]):
                    for j, speaker in enumerate(people):
                        for t in range(k):
                            p = prompts[(j * k + t) % 8]
                            fit.extend(paths[(speaker, p, c)] for c in range(6))
                for model, config in (
                    ("ridge_wavlm", {"model": "ridge_a1_wavlm_base_plus", "feature_state": 12, "alpha": 1.0,
                                     "class_weight": "balanced", "solver": "lsqr", "tol": 0.0001}),
                    ("cnn", {"model": "cnn", "batch_size": 32, "epochs": 100, "patience": 15,
                             "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1}),
                ):
                    units.append({"model": model, "fold": 0, "rotation": 0, "draw": 0, "scenario": scenario,
                                  "B": B, "S": S, "P_global": 8, "P_per_speaker": k, "R": 1,
                                  "train_seed": 20260905, "fit": sorted(fit), "val": list(val), "test": list(test),
                                  "config": config})
    plan = reseal({"schema": "ser-study2-core-1", "program": "SER26-STUDY2-CORE-1", "units": units})
    return plan, rows, sex


@pytest.fixture
def fixture(source_fixture):
    return copy.deepcopy(source_fixture)


def check(fixture):
    plan, rows, sex = fixture
    return verify.verify_document(plan, rows, sex, complete=False)


def test_fixture_checks_pass_without_claiming_real_core(fixture):
    report = check(fixture)
    assert report["pass"] and report["n_units"] == 16
    assert report["scope"] == "test_fixture_only"
    assert report["scientific_outcomes_read"] is False


def test_partial_fixture_cannot_pass_full_cli_scope(fixture):
    plan, rows, sex = fixture
    with pytest.raises(verify.VerificationError, match="1440"):
        verify.verify_document(plan, rows, sex)


def test_hash_change_detected_before_semantics(fixture):
    fixture[0]["units"][0]["fit"].pop()
    with pytest.raises(verify.VerificationError, match="plan hash"):
        check(fixture)


def test_resealed_recording_drop_rejected(fixture):
    plan = fixture[0]
    plan["units"][0]["fit"].pop()
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="budget"):
        check(fixture)


@pytest.mark.parametrize("intensity", ["XX", "HI"])
def test_resealed_non_md_choice_rejected_when_md_exists(fixture, intensity):
    plan = fixture[0]
    fit = plan["units"][0]["fit"]
    chosen = "1000_AAA_ANG_MD.wav"
    fit[fit.index(chosen)] = f"1000_AAA_ANG_{intensity}.wav"
    fit.sort()
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="nonrepresentative"):
        check(fixture)


def test_unrecognized_intensity_can_exist_raw_but_is_not_selected(fixture):
    plan, rows, sex = fixture
    base = next(r for r in rows if r["relative_path"] == "1000_AAL_SAD_XX.wav")
    base["relative_path"] = "1000_AAL_SAD_X.wav"
    base["sha256"] = sha(base["relative_path"])
    base["intensity"] = "X"
    reps, report = verify.representatives(rows, complete=False)
    assert base["relative_path"] not in reps
    assert ["1000", "AAL", 5] in report["unavailable_cells"]
    assert verify.verify_document(plan, rows, sex, complete=False)["pass"]


def test_missing_stop_recording_is_allowed_only_when_source_is_missing(fixture):
    plan, rows, sex = fixture
    path = plan["units"][0]["val"][0]
    rows[:] = [r for r in rows if r["relative_path"] != path]
    for unit in plan["units"]:
        unit["val"].remove(path)
    reseal(plan)
    report = verify.verify_document(plan, rows, sex, complete=False)
    assert report["early_stop_rows_range"] == [287, 287]


def test_omitting_available_stop_item_rejected(fixture):
    plan = fixture[0]
    for unit in plan["units"]:
        unit["val"].pop()
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="exact available"):
        check(fixture)


def test_test_speaker_leakage_rejected_after_reseal(fixture):
    plan = fixture[0]
    fit = plan["units"][0]["fit"]
    fit[:] = sorted(p.replace("1000_", "1028_") for p in fit)
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="speaker overlap"):
        check(fixture)


def test_unseen_prompt_leakage_rejected(fixture):
    plan = fixture[0]
    unit = next(u for u in plan["units"] if u["scenario"] == "prompt_new")
    unit["fit"] = sorted(p.replace("_AAI_", "_AAG_") for p in unit["fit"])
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="unseen test prompt"):
        check(fixture)


def test_adjacency_change_rejected_even_with_balanced_degrees(fixture):
    plan = fixture[0]
    # S48, B288 assigns one prompt per speaker. Swap two same-sex identities only
    # in the new condition: counts, class support, sex balance and degrees remain.
    for unit in plan["units"]:
        if unit["B"] == 288 and unit["S"] == 48 and unit["scenario"] == "prompt_new":
            swapped = []
            for p in unit["fit"]:
                if p.startswith("1000_"):
                    p = "1001_" + p[5:]
                elif p.startswith("1001_"):
                    p = "1000_" + p[5:]
                swapped.append(p)
            unit["fit"] = sorted(swapped)
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="common-prompt adjacency"):
        check(fixture)


def test_resealed_missing_model_not_independent_replication(fixture):
    plan = fixture[0]
    plan["units"] = [u for u in plan["units"] if u["model"] != "cnn"]
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="paired model"):
        check(fixture)


def test_seed_change_in_one_condition_rejected(fixture):
    plan = fixture[0]
    for unit in plan["units"]:
        if unit["scenario"] == "prompt_new":
            unit["train_seed"] += 1
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="scenario pairing changes training seed"):
        check(fixture)


def test_unregistered_model_hyperparameter_rejected(fixture):
    plan = fixture[0]
    plan["units"][0]["config"]["alpha"] = 100.0
    reseal(plan)
    with pytest.raises(verify.VerificationError, match="model configuration"):
        check(fixture)


def test_manifest_label_tampering_rejected(fixture):
    fixture[1][0]["label_index"] = "1"
    with pytest.raises(verify.VerificationError, match="label/corpus"):
        check(fixture)


def test_metadata_only_cli_still_checks_source_hashes(tmp_path):
    manifest, demographics, source = (tmp_path / n for n in ("manifest.csv", "sex.csv", "source.py"))
    manifest.write_text("relative_path\n", encoding="utf-8")
    demographics.write_text("ActorID,Sex\n", encoding="utf-8")
    source.write_text("version = 2\n", encoding="utf-8")
    plan = {"input": {"manifest_path": manifest.name, "manifest_sha256": verify.byte_hash(manifest),
                      "demographics_path": demographics.name, "demographics_sha256": verify.byte_hash(demographics)},
            "source_sha256": {source.name: sha("version = 1\n")}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="source byte hash"):
        verify.verify_plan_file(tmp_path, path, metadata_only=True)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"units": [], "units": [1]}', encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="duplicate JSON"):
        verify.read_json(path)

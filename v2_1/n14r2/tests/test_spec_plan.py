from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from v2_1.n14r2 import plan, seeds
from v2_1.n14r2.spec import (
    ALL_DRAW_IDS, BOOTSTRAP_SEED, FIT_WORKERS_PER_POD, HPO_GRID, N_PODS,
    PLAN_FIELDS, SEED_NAMESPACE, SPEC, SPEC_VERSION, STUDY_ID,
    pod_index_for_draw, sha256_json, validate_spec,
)


def synthetic_rows(n_speakers: int = 91) -> list[dict[str, str]]:
    rows = []
    for speaker in range(n_speakers):
        for label in range(6):
            for take in range(1 + ((speaker * speaker + label * 7 + speaker * label * 3) % 5)):
                rows.append({
                    "sample_index": str(len(rows)),
                    "relative_path": f"s{speaker:02d}_y{label}_t{take}.wav",
                    "speaker": f"s{speaker:02d}", "label": f"y{label}",
                    "label_index": str(label),
                    "sha256": hashlib.sha256(f"{speaker}|{label}|{take}".encode()).hexdigest(),
                })
    return rows


@pytest.fixture(scope="module")
def synthetic_plan():
    return plan.make_plan(synthetic_rows(), "a" * 64)


def test_spec_json_is_exact_single_authority():
    validate_spec()
    frozen = json.loads((Path(__file__).parents[1] / "spec.json").read_text(encoding="utf-8"))
    assert frozen == SPEC
    validate_spec(frozen)
    assert (STUDY_ID, SPEC_VERSION, SEED_NAMESPACE) == ("N14R2", "2.0.0", "SER26-N14R2")
    assert BOOTSTRAP_SEED == 202609050001
    assert len(HPO_GRID) == 8
    assert (HPO_GRID[4]["lr"], HPO_GRID[4]["weight_decay"], HPO_GRID[4]["dropout"]) == (1e-3, 1e-4, 0.1)
    assert SPEC["inference"]["multiplicity"] == "Holm family N14R2 of size 1"
    assert SPEC["previous_n14r"]["reused_draws"] == SPEC["previous_n14r"]["reused_fits"] == 0
    assert SPEC["previous_n14r"]["excluded_from_all_n14r2_estimates_and_decisions"] is True


@pytest.mark.parametrize("section,key,value", [
    ("completion", "maximum_attempts_per_unit", 3),
    ("completion", "infrastructure_exhaustion_action", "void"),
    ("completion", "interim_effect_scoring_permitted", True),
    ("execution", "fit_workers_per_node", 4),
    ("execution", "all_primary_draws_resolved_before_any_reserve", False),
    ("execution_environment", "gpu", "NVIDIA GeForce RTX 5070"),
    ("budget", "resource_allocation_is_stopping_rule", True),
    ("previous_n14r", "reused_fits", 1280),
])
def test_spec_rejects_mutations(section, key, value):
    bad = copy.deepcopy(SPEC)
    bad[section][key] = value
    with pytest.raises(ValueError, match="frozen N14R2"):
        validate_spec(bad)


def test_outcome_blind_failure_budget_and_reserve_contract():
    c = SPEC["completion"]
    assert c["maximum_attempts_per_unit"] == 2
    assert c["theoretical_maximum_training_attempts"] == 4480
    assert c["dispatch_counts_even_if_start_is_uncertain"] is True
    assert c["initialization_or_health_check_without_dispatch_counts_as_attempt"] is False
    assert c["infrastructure_failure_action"] == "pause_and_circuit_break"
    assert c["infrastructure_exhaustion_action"] == "blocked_not_void_no_reserve"
    assert c["mixed_or_unknown_exhausted_failures_action"] == "blocked_not_void_no_reserve"
    assert c["two_allowlisted_training_failures_action"] == "void_whole_draw"
    assert c["primary_resolved_states_before_reserves"] == ["complete", "void"]
    assert c["reserve_activation_order"] == [24, 25, 26, 27]
    assert c["paused_or_blocked_is_terminal_incomplete"] is False
    assert SPEC["budget"]["agent_guard_usd"] == 25.0
    assert SPEC["budget"]["resource_allocation_is_stopping_rule"] is False


def test_fixed_balanced_draw_assignment():
    assert N_PODS == 4 and FIT_WORKERS_PER_POD == 8
    assignments = {n: [d for d in ALL_DRAW_IDS if pod_index_for_draw(d) == n] for n in range(4)}
    assert assignments == {n: list(range(n, 28, 4)) for n in range(4)}
    assert all(sum(d < 24 for d in ds) == 6 for ds in assignments.values())
    for bad in (-1, 28, True, 2.0, "2"):
        with pytest.raises(ValueError):
            pod_index_for_draw(bad)


def test_new_seed_domains_do_not_reuse_old_study_seeds():
    # Read-only public source imports; no N14R run, result, history, or prediction.
    from v2_1.n14r import seeds as old_seeds

    def table(module):
        return (
            [module.split_seed(d) for d in ALL_DRAW_IDS]
            + [module.inner_seed(d, f, c) for d in ALL_DRAW_IDS for f in range(5) for c in ("GR_hpo", "GG_hpo")]
            + [module.train_seed(d, f) for d in ALL_DRAW_IDS for f in range(5)]
        )

    new, old = table(seeds), table(old_seeds)
    assert len(new) == 448 and len(set(new)) == 448
    assert not (set(new) & set(old))
    assert seeds.inner_seed(3, 2, "GR_hpo") != seeds.inner_seed(3, 2, "GG_hpo")
    assert seeds.split_seed(0) != seeds.stable_u32("SER26|split|cremad|0")
    assert seeds.train_seed(3, 2) == seeds.train_seed(3, 2)


@pytest.mark.parametrize("bad", [-1, 28, True, 2.0, "2"])
def test_seed_rejects_invalid_draw(bad):
    with pytest.raises(ValueError):
        seeds.split_seed(bad)


def test_plan_grid_pairing_identity_and_partition(synthetic_plan):
    units, split_docs = synthetic_plan
    assert len(units) == len({u["unit_id"] for u in units}) == 2240
    assert sum(u["draw_role"] == "primary" for u in units) == 1920
    assert sum(u["draw_role"] == "reserve" for u in units) == 320
    assert all(tuple(u) == PLAN_FIELDS for u in units)
    assert {u["arm"] for u in units} == {STUDY_ID}
    assert len(split_docs) == 56
    for d in ALL_DRAW_IDS:
        draw = [u for u in units if u["draw_id"] == d]
        assert len(draw) == 80
        assert {pod_index_for_draw(u["draw_id"]) for u in draw} == {d % 4}
        for f in range(5):
            fr = [u for u in draw if u["fold"] == f]
            assert len({u["outer_sha256"] for u in fr}) == 1
            assert len({u["train_seed"] for u in fr}) == 1
            assert {u["cell"] for u in fr} == {"GR_hpo", "GG_hpo"}
    for u in units:
        payload = {k: v for k, v in u.items() if k not in {"unit_id", "est_gpu_sec", "status"}}
        payload.update(seed_namespace=SEED_NAMESPACE, config=HPO_GRID[u["config_index"]])
        assert u["unit_id"] == sha256_json(payload)
    for key, doc in split_docs.items():
        assert key.startswith("n14r2__") and len(doc["folds"]) == 5
        assert all((f["meta"]["inner_overlap_speaker_count"] > 0) == key.endswith("GR_hpo") for f in doc["folds"])
        tests = [p for f in doc["folds"] for p in f["test"]]
        assert len(tests) == len(set(tests)) == len(doc["population"])
        assert set(tests) == set(doc["population"])


def test_generation_is_deterministic(synthetic_plan):
    assert plan.make_plan(synthetic_rows(), "a" * 64) == synthetic_plan


@pytest.mark.parametrize("field,value", [
    ("arm", "N14R"), ("train_seed", 1), ("draw_role", "reserve"),
    ("config_sha256", "b" * 64), ("unit_id", "c" * 64),
])
def test_validator_rejects_plan_mutations(synthetic_plan, field, value):
    units, docs = synthetic_plan
    changed = [dict(u) for u in units]
    changed[0][field] = value
    with pytest.raises(RuntimeError):
        plan.validate_plan(changed, docs)


def test_cost_and_progress_do_not_change_scientific_identity(synthetic_plan):
    units, docs = synthetic_plan
    changed = [dict(u) for u in units]
    changed[0].update(est_gpu_sec=999.0, status="done")
    plan.validate_plan(changed, docs)


def test_build_hashes_and_csv_roundtrip(tmp_path):
    manifest = tmp_path / "cremad_manifest.csv"
    rows = synthetic_rows()
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "plan"
    summary = plan.build(manifest, out)
    assert summary["study_id"] == STUDY_ID
    assert summary["primary_units"] == 1920 and summary["maximum_units"] == 2240
    assert summary["spec_sha256"] == sha256_json(SPEC)
    assert summary["draw_to_node"] == {str(d): d % 4 for d in ALL_DRAW_IDS}
    for field, filename in (("run_plan_sha256", "run_plan.csv"), ("hygiene_sha256", "hygiene.json"), ("split_index_sha256", "split_index.json")):
        assert summary[field] == hashlib.sha256((out / filename).read_bytes()).hexdigest()
    index = json.loads((out / "split_index.json").read_text(encoding="utf-8"))
    docs = {}
    for key, entry in index.items():
        path = out / entry["path"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        docs[key] = json.loads(path.read_text(encoding="utf-8"))
    with (out / "run_plan.csv").open(encoding="utf-8", newline="") as handle:
        roundtrip = list(csv.DictReader(handle))
    plan.validate_plan(roundtrip, docs)
    hygiene = json.loads((out / "hygiene.json").read_text(encoding="utf-8"))
    assert hygiene["schema"] == "ser26-n14r2-hygiene-1"
    assert hygiene["n_input"] == hygiene["n_kept"] == len(rows)


def test_missing_class_is_rejected():
    with pytest.raises(ValueError):
        plan.make_plan([r for r in synthetic_rows() if r["label_index"] != "5"], "b" * 64)


def test_real_manifest_hygiene_and_existing_trainer_planio(tmp_path):
    from v2.ser_v2.train import PlanIO

    repo = Path(__file__).resolve().parents[3]
    manifests = repo / "v2" / "manifests"
    manifest = manifests / "cremad_manifest.csv"
    rows, raw_sha, hygiene = plan.load_manifest(manifest)
    assert (hygiene["n_input"], hygiene["n_kept"], hygiene["n_dropped"]) == (7442, 7435, 7)
    assert [int(r["sample_index"]) for r in rows] == list(range(7435))
    assert raw_sha == hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected_drops = {
        "1006_TIE_HAP_XX.wav", "1006_TIE_NEU_XX.wav", "1013_WSI_DIS_XX.wav",
        "1013_WSI_SAD_XX.wav", "1017_IWW_ANG_XX.wav", "1017_IWW_FEA_XX.wav",
        "1076_MTI_SAD_XX.wav",
    }
    assert {x["relative_path"] for x in hygiene["dropped"]} == expected_drops
    out = tmp_path / "plan"
    summary = plan.build(manifest, out)
    assert (summary["manifest_input_rows"], summary["analysis_population_rows"]) == (7442, 7435)
    io = PlanIO(out, manifests, tmp_path / "unused_run")
    assert len(io.plan) == 2240 and io.plan[0]["arm"] == STUDY_ID
    for row in io.plan[::8]:
        fold = io.fold(row)
        assert set(fold["fit"] + fold["val"] + fold["test"]) == {r["relative_path"] for r in rows}
        assert not (expected_drops & set(fold["fit"] + fold["val"] + fold["test"]))
        assert io.configs[row["config_sha256"]]["engine"] == "p1_frozen"

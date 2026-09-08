"""Metadata-only Study II checks; no real cache, model, prediction, or score access."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

import pytest

from v3.data_design import core_plan as core

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def metadata():
    return core.read_metadata(REPO)


@pytest.fixture(scope="module")
def constructed():
    # Only cache metadata is stubbed. The complete real manifest and actual
    # builder construct all 1,440 units, without opening either feature array.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core, "feature_identity", lambda _: {
            "feature_sha256": {"wavlm_base_plus": "1" * 64, "logmel": "2" * 64},
            "feature_files": {"wavlm_base_plus": "TEST_METADATA_ONLY.npz", "logmel": "TEST_METADATA_ONLY.npz"}})
        return core.build(REPO, Path("unused_TEST_metadata"), require_runtime=False)


def by_path(meta):
    return {r["relative_path"]: r for r in meta["clean"]}


def test_complete_unit_grid_and_identity(constructed):
    plan, inventory, _ = constructed
    assert len(plan["units"]) == len({u["unit_id"] for u in plan["units"]}) == 1440
    assert Counter(u["model"] for u in plan["units"]) == {"ridge_wavlm": 720, "cnn": 720}
    assert inventory["core_units"] == 1440 and len(inventory["conditions"]) == 90
    assert plan["plan_sha256"] == core.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    for u in plan["units"]:
        assert u["unit_id"] == core.digest({k: v for k, v in u.items() if k != "unit_id"})


def test_representatives_prefer_md_then_xx_and_exclude_other_intensities(metadata):
    groups = defaultdict(list)
    for row in metadata["clean"]:
        groups[(row["speaker"], row["sentence"], int(row["label_index"]))].append(row)
    expected = {}
    for key, rows in groups.items():
        medium = sorted((r for r in rows if r["intensity"] == "MD"), key=lambda r: r["relative_path"])
        unspecified = sorted((r for r in rows if r["intensity"] == "XX"), key=lambda r: r["relative_path"])
        if medium or unspecified:
            expected[key] = (medium or unspecified)[0]
    assert metadata["cells"] == expected
    assert len(expected) == 6524
    assert Counter(r["intensity"] for r in expected.values()) == {"MD": 455, "XX": 6069}
    assert len(groups) > len(expected)  # At least one cell cannot use HI/LO as a fallback.


def test_all_roles_exact_quotas_and_sex_balance_per_prompt(constructed):
    plan, _, meta = constructed
    index = by_path(meta)
    for u in plan["units"]:
        panels = {role: [index[p] for p in u[role]] for role in ("fit", "val", "test")}
        speakers = {role: {r["speaker"] for r in rows} for role, rows in panels.items()}
        assert not (speakers["fit"] & speakers["val"] or speakers["fit"] & speakers["test"]
                    or speakers["val"] & speakers["test"])
        paths = u["fit"] + u["val"] + u["test"]
        assert len(paths) == len(set(paths)) == len({index[p]["sha256"] for p in paths})
        assert len(u["fit"]) == u["B"] and len(speakers["fit"]) == u["S"]
        assert len(speakers["val"]) == 8
        assert Counter(meta["sex"][s] for s in speakers["fit"]) == {"Female": u["S"] // 2, "Male": u["S"] // 2}
        assert {r["intensity"] for rows in panels.values() for r in rows} <= {"MD", "XX"}
        fit = panels["fit"]
        assert Counter(r["label_index"] for r in fit) == {str(c): u["B"] // 6 for c in range(6)}
        for speaker in speakers["fit"]:
            utterances = [r for r in fit if r["speaker"] == speaker]
            assert len({r["sentence"] for r in utterances}) == u["P_per_speaker"]
            assert Counter(r["label_index"] for r in utterances) == {str(c): u["P_per_speaker"] for c in range(6)}
        prompts = {r["sentence"] for r in fit}
        assert len(prompts) == 8
        for prompt in prompts:
            utterances = [r for r in fit if r["sentence"] == prompt]
            people = {r["speaker"] for r in utterances}
            assert len(people) == u["B"] // 48
            assert Counter(meta["sex"][s] for s in people) == {"Female": u["B"] // 96, "Male": u["B"] // 96}
        for role in ("val", "test"):
            for speaker in speakers[role]:
                assert {int(r["label_index"]) for r in panels[role] if r["speaker"] == speaker} == set(range(6))


def test_seen_new_are_identical_graphs_under_prompt_renaming(constructed):
    plan, _, meta = constructed
    index = by_path(meta)
    grouped = defaultdict(dict)
    for u in plan["units"]:
        key = tuple(u[k] for k in ("model", "draw", "fold", "rotation", "B", "S"))
        grouped[key][u["scenario"]] = u
    assert len(grouped) == 720
    for pair in grouped.values():
        seen, new = pair["prompt_seen"], pair["prompt_new"]
        test_prompts, common, slots = core.prompt_slots(meta, seen["rotation"])
        assert seen["val"] == new["val"] and seen["test"] == new["test"]
        assert seen["train_seed"] == new["train_seed"] and seen["config"] == new["config"]
        graphs = []
        for scenario, unit in (("prompt_seen", seen), ("prompt_new", new)):
            positions = {p: i for i, p in enumerate(slots[scenario])}
            graphs.append({(index[p]["speaker"], positions[index[p]["sentence"]], index[p]["label_index"])
                           for p in unit["fit"]})
        assert graphs[0] == graphs[1]
        assert set(test_prompts) <= {index[p]["sentence"] for p in seen["fit"]}
        assert not set(test_prompts) & {index[p]["sentence"] for p in new["fit"]}
        assert {index[p]["sentence"] for p in seen["val"]} == set(common)


def test_six_rotations_and_five_folds_cover_all_prompts_and_people(metadata):
    coverage = Counter(p for rotation in range(6) for p in core.prompt_slots(metadata, rotation)[0])
    assert coverage == {p: 1 for p in metadata["prompts"]}
    for draw in range(3):
        tested = Counter()
        for fold in range(5):
            test, stop, pool = core.roles(metadata, draw, fold)
            assert not (test & stop or test & pool or stop & pool)
            assert test | stop | pool == set(metadata["speakers"])
            assert Counter(metadata["sex"][s] for s in stop) == {"Female": 4, "Male": 4}
            tested.update(test)
        assert tested == {s: 1 for s in metadata["speakers"]}


def test_manifest_and_demographic_input_order_does_not_change_design(metadata, tmp_path):
    for relative in ("v2/manifests/cremad_manifest.csv", "v3/data_design/metadata/VideoDemographics.csv"):
        with (REPO / relative).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames, list(reader)
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(reversed(rows))
    permuted = core.read_metadata(tmp_path)
    assert permuted["cells"] == metadata["cells"] and permuted["sex"] == metadata["sex"]
    assert permuted["input"]["manifest_sha256"] != metadata["input"]["manifest_sha256"]
    for draw in range(3):
        for fold in range(5):
            assert core.roles(permuted, draw, fold) == core.roles(metadata, draw, fold)
            pool = core.roles(metadata, draw, fold)[2]
            for rotation in range(6):
                assert core.prompt_slots(permuted, rotation) == core.prompt_slots(metadata, rotation)
                slots = core.prompt_slots(metadata, rotation)[2]
                prompts = set().union(*map(set, slots.values()))
                original, _ = core.select_speakers(metadata, pool, prompts, "order-test")
                reordered, _ = core.select_speakers(permuted, list(reversed(sorted(pool))), prompts, "order-test")
                assert reordered == original
                assert core.panel(metadata, original[:12], slots["prompt_new"], 288, "order-test", True) == core.panel(
                    permuted, reordered[:12], slots["prompt_new"], 288, "order-test", True)


@pytest.mark.parametrize("budget", [0, 287, 648, -72])
def test_impossible_budget_is_rejected(metadata, budget):
    pool = core.roles(metadata, 0, 0)[2]
    slots = core.prompt_slots(metadata, 0)[2]
    people, _ = core.select_speakers(metadata, pool, set().union(*map(set, slots.values())), "infeasible")
    with pytest.raises(ValueError):
        core.panel(metadata, people[:12], slots["prompt_new"], budget, "infeasible", True)


def test_insufficient_sex_stratum_is_rejected(metadata):
    pool = [s for s in metadata["speakers"] if metadata["sex"][s] == "Male"]
    with pytest.raises(ValueError, match="insufficient complete eligible"):
        core.select_speakers(metadata, pool, metadata["prompts"], "no-female")

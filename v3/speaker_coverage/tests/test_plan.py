"""Synthetic fixtures test planning invariants, never corpus feasibility/results."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from v3.speaker_coverage import plan as p


@pytest.fixture(scope="module")
def synthetic():
    people = [f"person_{i:03d}" for i in range(60)]
    prompts = [f"text_{i:02d}" for i in range(12)]
    labels = ("angry", "disgust", "fear", "happy", "neutral", "sad")
    cells, embeddings = {}, {}
    rng = np.random.default_rng(4307)
    sex = {s: "Female" if i < 30 else "Male" for i, s in enumerate(people)}
    for speaker in people:
        center = rng.normal(size=8)
        for prompt in prompts:
            for label, name in enumerate(labels):
                path = f"{speaker}_{prompt}_{name}.wav"
                row = {"speaker": speaker, "sentence": prompt, "label_index": str(label),
                       "label": name, "relative_path": path, "sha256": p.digest(path), "intensity": "MD"}
                cells[speaker, prompt, label] = row
                if name == "neutral":
                    embeddings[path] = center + rng.normal(scale=0.04, size=8)
    meta = {"speakers": people, "prompts": prompts, "sex": sex, "cells": cells,
            "clean": list(cells.values()), "raw": list(cells.values()), "input": {"fixture": True}}
    return meta, embeddings


@pytest.fixture(scope="module")
def full_plan(synthetic):
    meta, embeddings = synthetic
    return p.build_from_metadata(meta, embeddings,
                                 {"fixture": True, "manifest_sha256": "synthetic-manifest",
                                  "asv_sha256": "synthetic-embeddings"})


def test_complete_grid_budgets_fixed_roles_and_distinct_draws(synthetic, full_plan):
    meta, _ = synthetic
    plan, capacity, diagnostics = full_plan
    assert plan["schema"] == "ser-speaker-coverage-1"
    assert plan["design"]["fixture"] and not plan["design"]["runtime_ready"]
    assert Counter(u["block"] for u in plan["units"]) == {"core": 540, "ft": 180}
    assert Counter(u["model"] for u in plan["units"]) == {"cnn": 270, "ridge_wavlm": 270, "wavlm_ft": 180}
    assert len(capacity["conditions"]) == len(diagnostics["conditions"]) == 30
    assert all(s["evaluation_population"].startswith("same_complete_training")
               for c in diagnostics["conditions"] for s in c["selections"])
    assert diagnostics["no_heldout_distance_gate"]
    by_context = {}
    fits = {}
    roles_by_fold = {}
    for unit in plan["units"]:
        assert unit["unit_id"] == p.digest({k: v for k, v in unit.items() if k != "unit_id"})
        key = unit["fold"], unit["rotation"]
        expected = by_context.setdefault(key, (unit["val"], unit["test"]))
        assert expected == (unit["val"], unit["test"])
        if unit["model"] == "ridge_wavlm":
            p.validate_panel(meta, unit["fit"])
            p._check_boundaries(meta, unit["fit"], unit["val"], unit["test"])
            fits.setdefault((*key, unit["policy"]), set()).add(tuple(unit["fit"]))
        if unit["model"] != "ridge_wavlm":
            assert unit["config"]["early_stopping"] is False
            assert unit["config"]["patience"] > unit["config"]["epochs"]
    assert all(len(value) == 3 for value in fits.values())
    for condition in capacity["conditions"]:
        pair = condition["test_speakers"], condition["stop_speakers"]
        assert roles_by_fold.setdefault(condition["fold"], pair) == pair
        assert set(condition["train_prompts"]).isdisjoint(condition["query_prompts"])
        assert set(condition["stop_prompts"]) <= set(condition["train_prompts"])
    assert plan["plan_sha256"] == p.digest({k: v for k, v in plan.items() if k != "plan_sha256"})


def test_no_stop_query_or_emotional_embedding_access(synthetic):
    meta, embeddings = synthetic
    test, stop, pool = p.roles(meta, 0, 0)
    _, _, slots = p.prompt_slots(meta, 0)
    prompts = slots["prompt_new"]
    expected = {meta["cells"][s, text, 4]["relative_path"] for s in pool for text in prompts}

    class AuditedMapping(dict):
        def __init__(self):
            super().__init__({key: embeddings[key] for key in expected})
            self.reads = []

        def __getitem__(self, key):
            assert key in expected, f"forbidden reference read {key}"
            self.reads.append(key)
            return super().__getitem__(key)

    permitted_only = AuditedMapping()
    panels, capacity, _ = p.plan_context(meta, permitted_only, 0, 0)
    assert set(permitted_only.reads) == expected
    assert set(capacity["eligible_speakers"]).isdisjoint(test | stop)
    assert len(panels) == 9
    # Supplying arbitrary held-out vectors must not alter this context at all.
    perturbed = {key: np.full(8, np.nan) for key in embeddings if key not in expected}
    perturbed.update({key: embeddings[key] for key in expected})
    assert p.plan_context(meta, perturbed, 0, 0) == p.plan_context(meta, permitted_only, 0, 0)


def test_reference_normalization_and_training_value_failures(synthetic):
    meta, embeddings = synthetic
    _, _, pool = p.roles(meta, 0, 0)
    _, _, slots = p.prompt_slots(meta, 0)
    people = sorted(pool)
    centers, refs = p.candidate_centroids(meta, people, slots["prompt_new"], embeddings)
    assert all(np.isclose(np.linalg.norm(v), 1.0) for v in centers.values())
    speaker = people[0]
    expected = np.mean([embeddings[path] / np.linalg.norm(embeddings[path]) for path in refs[speaker]], axis=0)
    np.testing.assert_allclose(centers[speaker], expected / np.linalg.norm(expected))
    broken = dict(embeddings)
    broken[refs[speaker][0]] = np.zeros(8)
    with pytest.raises(ValueError, match="zero ASV norm"):
        p.candidate_centroids(meta, people, slots["prompt_new"], broken)
    del broken[refs[speaker][0]]
    with pytest.raises(ValueError, match="missing training reference"):
        p.candidate_centroids(meta, people, slots["prompt_new"], broken)


def test_incomplete_candidates_excluded_before_selection(synthetic):
    original, embeddings = synthetic
    meta = deepcopy(original)
    _, _, pool = p.roles(meta, 0, 0)
    _, _, slots = p.prompt_slots(meta, 0)
    excluded = sorted(pool)[0]
    del meta["cells"][excluded, slots["prompt_new"][0], 1]
    panels, capacity, _ = p.plan_context(meta, embeddings, 0, 0)
    assert excluded not in capacity["eligible_speakers"]
    assert excluded not in capacity["reference_paths"]
    assert all(excluded not in panel["speaker_prompts"] for panel in panels)


def test_no_capacity_repair_by_using_stop_people(synthetic):
    original, embeddings = synthetic
    meta = deepcopy(original)
    _, _, pool = p.roles(meta, 0, 0)
    _, _, slots = p.prompt_slots(meta, 0)
    females = sorted(s for s in pool if meta["sex"][s] == "Female")
    for speaker in females[11:]:
        del meta["cells"][speaker, slots["prompt_new"][0], 1]
    with pytest.raises(ValueError, match="candidate capacity"):
        p.plan_context(meta, embeddings, 0, 0)


def test_policy_determinism_trace_and_distinct_objectives(synthetic):
    meta, embeddings = synthetic
    _, _, pool = p.roles(meta, 0, 0)
    _, _, slots = p.prompt_slots(meta, 0)
    centers, _ = p.candidate_centroids(meta, pool, slots["prompt_new"], embeddings)
    for policy in p.POLICIES:
        people, trace = p.select_policy(pool, centers, meta["sex"], policy, 0, 0)
        assert (people, trace) == p.select_policy(reversed(sorted(pool)), centers, meta["sex"], policy, 0, 0)
        if policy != "U":
            assert (people, trace) == p.select_policy(pool, centers, meta["sex"], policy, 0, 0, draw=2)
            means = [item["mean_nn1"] for item in trace["selection_trace"]]
            assert all(b <= a + 1e-12 for a, b in zip(means, means[1:]))
            if policy == "R":
                swaps = [item for item in trace["selection_trace"] if item["step"] == "same_sex_swap"]
                assert len(swaps) <= 100
                assert all(meta["sex"][s["out"]] == meta["sex"][s["in"]] for s in swaps)
        assert Counter(meta["sex"][s] for s in people) == {"Female": 12, "Male": 12}


def test_common_graph_retry_is_bounded_and_cannot_silently_duplicate(synthetic, monkeypatch):
    meta, embeddings = synthetic
    frozen_graph = p.abstract_layout(0, 0, 0)
    monkeypatch.setattr(p, "abstract_layout", lambda *args, **kwargs: frozen_graph)
    with pytest.raises(ValueError, match="100 predefined attempts"):
        p.plan_context(meta, embeddings, 0, 0)


def test_byte_duplicate_cross_boundary_is_rejected(synthetic):
    original, embeddings = synthetic
    meta = deepcopy(original)
    panels, _, _ = p.plan_context(meta, embeddings, 0, 0)
    panel = panels[0]
    by_path = {r["relative_path"]: r for r in meta["cells"].values()}
    by_path[panel["test"][0]]["sha256"] = by_path[panel["fit"][0]]["sha256"]
    with pytest.raises(ValueError, match="byte overlap"):
        p._check_boundaries(meta, panel["fit"], panel["val"], panel["test"])


def test_asv_input_identity_tracks_change_and_rejects_ambiguous_paths(tmp_path):
    cache = tmp_path / "asv.npz"
    np.savez(cache, paths=np.array(["one.wav", "two.wav"]), embeddings=np.eye(2, dtype=np.float32))
    first, first_id = p.load_asv(cache)
    np.savez(cache, paths=np.array(["one.wav", "two.wav"]), embeddings=np.array([[1., .1], [0., 1.]]))
    second, second_id = p.load_asv(cache)
    assert first_id["asv_sha256"] != second_id["asv_sha256"]
    assert not np.array_equal(first["one.wav"], second["one.wav"])
    for names, message in [(["one.wav", "one.wav"], "duplicate ASV"),
                           (["../one.wav", "two.wav"], "unsafe")]:
        np.savez(cache, paths=np.array(names), embeddings=np.eye(2))
        with pytest.raises(ValueError, match=message):
            p.load_asv(cache)


def test_input_and_source_change_invalidate_plan_identity(full_plan):
    plan, _, _ = full_plan
    for field, value in [("asv_sha256", "changed-asv"), ("manifest_sha256", "changed-label-source")]:
        changed = deepcopy(plan)
        changed["input"][field] = value
        del changed["plan_sha256"]
        assert p.digest(changed) != plan["plan_sha256"]
    changed = deepcopy(plan)
    changed["source_sha256"]["planner.py"] = "different-implementation"
    del changed["plan_sha256"]
    assert p.digest(changed) != plan["plan_sha256"]

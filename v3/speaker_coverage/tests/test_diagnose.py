"""Diagnostics tests use fabricated vectors/UARs only, never experiment evidence."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from v3.data_design import core_plan as legacy
from v3.speaker_coverage import diagnose as d
from v3.speaker_coverage import plan as p
from v3.speaker_coverage.tests.test_plan import full_plan, synthetic  # shared synthetic fixtures


def test_tied_ranks_constant_data_and_conditional_bootstrap():
    np.testing.assert_allclose(d.rank_average([1, 2, 2, 4]), [1, 2.5, 2.5, 4])
    assert np.isclose(d.spearman([1, 2, 2, 4], [4, 2, 2, 1]), -1)
    assert d.spearman([1, 1, 1, 1], [2, 3, 4, 5]) is None
    first = d.bootstrap_spearman(np.arange(20), np.arange(20), repetitions=100)
    assert first == d.bootstrap_spearman(np.arange(20), np.arange(20), repetitions=100)
    assert first["rho"] == pytest.approx(1)
    assert first["n_people"] == 20 and first["requested_bootstraps"] == 100
    assert "not new-training-population" in first["uncertainty"]


def test_nn1_and_nn3_do_not_share_zero_self_distance_interpretation():
    matrix = np.array([[0, .2, .4, .6], [.2, 0, .3, .5], [.4, .3, 0, .4], [.6, .5, .4, 0]])
    rows = d.geometry_rows(matrix, ["a", "b", "c", "d"], ["a", "b", "c"])
    inclusive, exclusive = rows
    assert inclusive["pool_scope"] == "including_selected"
    assert inclusive["nn1_mean"] == pytest.approx(.1)
    assert exclusive["nn1_mean"] == pytest.approx(.4)
    assert inclusive["nn3_mean"] > inclusive["nn1_mean"]
    assert exclusive["n_evaluation_people"] == 1


def test_e0_complete_descriptions_and_cross_encoder_fixed_selections(synthetic, full_plan):
    meta, vectors = synthetic
    plan, _, _ = full_plan
    original_units = deepcopy(plan["units"])
    report, tables = d.e0_diagnostics(meta, plan, vectors, vectors)
    assert report["n_fold_rotation_contexts"] == 30
    assert report["xvector_status"] == "completed"
    assert report["thresholds"] is None and report["policy_or_budget_changes"] == []
    assert report["used_new_ser_scores"] is False
    assert plan["units"] == original_units
    geometry = tables["e0_policy_geometry.csv"]
    assert len(geometry) == 30 * 3 * 3 * 2 * 2
    assert len(tables["e0_policy_overlap.csv"]) == 30 * 3 * 3
    cross = [row for row in report["space_summaries"] if row["comparison"] == "ecapa_vs_xvector"]
    assert len(cross) == 30
    assert all(row["pairwise_distance_spearman_descriptive"] == pytest.approx(1) for row in cross)
    assert all(row["mean_nn3_intersection_fraction"] == 1 for row in cross)
    assert all(row["pairs_are_not_independent_samples"] for row in cross)


def test_missing_xvector_support_is_reported_without_changing_pool(synthetic, full_plan):
    meta, vectors = synthetic
    plan, _, _ = full_plan
    xvector = dict(vectors)
    del xvector[next(iter(xvector))]
    report, tables = d.e0_diagnostics(meta, plan, vectors, xvector)
    assert report["xvector_status"] == "incomplete"
    assert report["reference_support_failures"]
    assert report["policy_or_budget_changes"] == []
    primary = [row for row in tables["e0_policy_geometry.csv"] if row["encoder"] == "ecapa"]
    assert len(primary) == 30 * 3 * 3 * 2


def legacy_fixture(missing_neutral=False):
    """80 fabricated people; old S48 and role invariants remain feasible."""
    people = [f"p{i:03d}" for i in range(80)]
    prompts = [f"t{i:02d}" for i in range(12)]
    meta = {"speakers": people, "prompts": prompts,
            "sex": {s: "Female" if i < 40 else "Male" for i, s in enumerate(people)}, "cells": {}}
    embeddings = {}
    rng = np.random.default_rng(902)
    for speaker in people:
        center = rng.normal(size=6)
        for prompt in prompts:
            for label in range(6):
                if missing_neutral and (speaker, prompt, label) == (people[-1], prompts[0], 4):
                    continue
                path = f"{speaker}_{prompt}_{label}.wav"
                meta["cells"][speaker, prompt, label] = {
                    "speaker": speaker, "sentence": prompt, "label_index": str(label),
                    "label": "neutral" if label == 4 else f"emotion{label}",
                    "relative_path": path, "sha256": p.digest(path)}
                if label == 4:
                    embeddings[path] = center + rng.normal(scale=.03, size=6)
    units = []
    for draw in range(3):
        for fold in range(5):
            test, stop, pool = legacy.roles(meta, draw, fold)
            for rotation in range(6):
                query, common, slots = legacy.prompt_slots(meta, rotation)
                union = set().union(*map(set, slots.values()))
                ordered, _ = legacy.select_speakers(meta, pool, union, f"synthetic-legacy:{draw}:{fold}:{rotation}", 48)
                fit = legacy.panel(meta, ordered, slots["prompt_new"], 576,
                                   f"synthetic-legacy:{draw}:{fold}:{rotation}", stratified=True)
                units.append({"model": "cnn", "B": 576, "S": 48, "scenario": "prompt_new",
                              "fold": fold, "rotation": rotation, "draw": draw, "fit": fit,
                              "val": [r["relative_path"] for r in legacy.rows_for(meta, stop, common)],
                              "test": [r["relative_path"] for r in legacy.rows_for(meta, test, query)]})
    scores = {speaker: [40 + i / 3, 41 + i / 3, 42 + i / 3] for i, speaker in enumerate(people)}
    return meta, {"units": units, "fixture": True}, embeddings, scores


def test_e1_uses_actual_legacy_fit_and_person_not_unit_sample_size():
    meta, old_plan, embeddings, scores = legacy_fixture()
    report, people = d.e1_diagnostics(meta, old_plan, scores, embeddings, repetitions=50)
    assert report["status"] == "completed_historical_csv_association"
    assert report["n_legacy_fit_instances"] == 90
    assert report["n_complete_reference_people"] == report["primary_spearman"]["n_people"] == 80
    assert len(people) == 80 and all(row["n_contexts"] == 18 for row in people)
    assert report["voice_only_covariate"]["included_in_model"] is False
    assert report["used_new_ser_scores"] is False and not report["causal_interpretation"]
    assert "raw logits not replayed" in report["uar_aggregation"]
    assert report["global_loso_spearman"]["n_people"] == 80


def test_e1_missing_references_excludes_person_without_imputation():
    meta, old_plan, embeddings, scores = legacy_fixture(missing_neutral=True)
    report, people = d.e1_diagnostics(meta, old_plan, scores, embeddings, repetitions=20)
    assert report["n_legacy_people"] == 80
    assert report["n_complete_reference_people"] == 79
    assert report["excluded_people"][0]["speaker"] == "p079"
    assert report["reference_support_failures"]
    assert all(row["speaker"] != "p079" for row in people)


def test_e1_rejects_a_partial_legacy_grid_and_wrong_csv_mean(tmp_path):
    meta, old_plan, embeddings, scores = legacy_fixture()
    old_plan["units"].pop()
    with pytest.raises(ValueError, match="incomplete legacy"):
        d.e1_diagnostics(meta, old_plan, scores, embeddings, repetitions=2)
    path = tmp_path / "synthetic_per_speaker.csv"
    path.write_text("model,B,S,scenario,speaker,draw0_uar_pp,draw1_uar_pp,draw2_uar_pp,mean_uar_pp\n"
                    "cnn,576,48,prompt_new,p001,40,41,42,99\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy mean"):
        d.legacy_scores(path)

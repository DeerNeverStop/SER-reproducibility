from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from v2_1.n14r.score import (
    BOOTSTRAP_SEED,
    Candidate,
    ScoreError,
    SplitStore,
    _canonical_sha256,
    _expected_analysis_draw_ids,
    _load_hygienic_manifest_file,
    _load_candidate,
    bootstrap_mean_summary,
    choose_config,
    exact_sign_flip_summary,
    macro_uar_6,
    one_sample_t_summary,
    score_run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reserve_void_uses_next_nonvoid_reserve() -> None:
    assert _expected_analysis_draw_ids({0, 24}) == [*range(1, 24), 25]


def test_macro_uar_is_whole_fold_and_requires_all_six_classes() -> None:
    truth = np.repeat(np.arange(6), 2)
    pred = truth.copy()
    pred[::2] = (pred[::2] + 1) % 6
    assert macro_uar_6(truth, pred) == pytest.approx(50.0)
    with pytest.raises(ScoreError, match="class 5 is absent"):
        macro_uar_6(range(5), range(5))
    with pytest.raises(ScoreError, match="0..5"):
        macro_uar_6(range(6), [0, 1, 2, 3, 4, 6])


def test_selection_requires_strict_grid_and_uses_lowest_index_tie() -> None:
    candidates = [Candidate(i, 80.0 if i in (2, 5) else float(i), 50.0) for i in range(8)]
    assert choose_config(reversed(candidates)).config_index == 2
    with pytest.raises(ScoreError, match="exactly once"):
        choose_config(candidates[:-1])
    duplicated = candidates[:-1] + [Candidate(6, 99.0, 50.0)]
    with pytest.raises(ScoreError, match="exactly once"):
        choose_config(duplicated)


def test_primary_t_direction_bootstrap_and_exact_sign_flip() -> None:
    values = np.linspace(0.5, 2.0, 24)
    summary = one_sample_t_summary(values)
    assert summary["n"] == 24 and summary["df"] == 23
    assert summary["mean"] > 0 and summary["p_two_sided"] <= 0.05
    assert summary["supported"] and summary["verdict"] == "supported"
    negative = one_sample_t_summary(-values)
    assert negative["p_two_sided"] <= 0.05 and not negative["supported"]

    first = bootstrap_mean_summary(values, reps=2_000, seed=BOOTSTRAP_SEED)
    second = bootstrap_mean_summary(values, reps=2_000, seed=BOOTSTRAP_SEED)
    assert first == second and first["reps"] == 2_000

    small = exact_sign_flip_summary([1.0, 2.0, 3.0, 4.0])
    assert small["n_permutations"] == 16
    assert small["mean"]["extreme_count"] == 2
    assert small["mean"]["p_two_sided"] == pytest.approx(0.125)
    # This is a real complete enumeration, not a Monte-Carlo placeholder.
    full = exact_sign_flip_summary([(-1.0) ** i * (i + 1) / 10.0 for i in range(24)])
    assert full["n_permutations"] == 2**24
    assert full["mean"]["extreme_count"] > 0
    assert full["studentized"]["extreme_count"] > 0


def _candidate_fixture(tmp_path: Path) -> tuple[dict[str, str], Path, SplitStore, dict, dict]:
    plan_dir = tmp_path / "plan"
    units_dir = tmp_path / "run" / "units"
    split_dir = plan_dir / "splits"
    split_dir.mkdir(parents=True)

    manifest: dict[str, dict] = {}
    refs = []
    for index in range(18):
        path = f"audio/{index:02d}.wav"
        item = {"sample_index": index, "relative_path": path, "label_index": index % 6, "speaker": f"s{index // 6}"}
        refs.append(item)
        manifest[path] = {"sample_index": index, "y_true": index % 6, "speaker": item["speaker"]}
    fit, val, test = refs[:6], refs[6:12], refs[12:18]
    outer_payload = {"outer_train": sorted(fit + val, key=lambda item: item["sample_index"]), "test": test}
    inner_payload = {"fit": fit, "val": val}
    outer_sha, inner_sha = _canonical_sha256(outer_payload), _canonical_sha256(inner_payload)
    split_doc = {
        "folds": [{
            "fold": 0,
            "fit": [item["relative_path"] for item in fit],
            "val": [item["relative_path"] for item in val],
            "test": [item["relative_path"] for item in test],
            "meta": {"study_id": "N14R", "draw_id": 0, "cell": "GR_hpo", "split_seed": 10,
                     "inner_seed": 20, "outer_sha256": outer_sha, "inner_sha256": inner_sha},
        }],
        "meta": {"study_id": "N14R", "spec_version": "1.0.0", "manifest_sha256": "m" * 64,
                 "draw_id": 0, "draw_role": "primary", "analysis_order": 0, "split_seed": 10},
    }
    split_text = json.dumps(split_doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    split_path = split_dir / "n14r__d00__GR_hpo.json"
    split_path.write_text(split_text, encoding="utf-8")
    split_sha = _sha(split_path)
    (plan_dir / "split_index.json").write_text(json.dumps({"k": {"sha256": split_sha, "path": "splits/n14r__d00__GR_hpo.json"}}), encoding="utf-8")
    config = {"engine": "p1_frozen", "model": "resnet_se", "lr": 3e-4, "weight_decay": 1e-4,
              "dropout": 0.1, "batch_size": 64, "epochs": 100, "patience": 15, "hpo_config_index": 0}
    config_sha = _canonical_sha256(config)
    row = {
        "study_id": "N14R", "spec_version": "1.0.0", "unit_id": "u0", "arm": "N14R",
        "manifest_sha256": "m" * 64, "draw_id": "0", "draw_role": "primary", "analysis_order": "0",
        "fold": "0", "train_rep": "0", "cell": "GR_hpo", "config_index": "0", "model": "resnet_se",
        "corpus_level": "cremad", "base_corpus": "cremad", "panel_draw": "", "r": "0", "seed_index": "0",
        "split_seed": "10", "inner_seed": "20", "train_seed": "30", "outer_sha256": outer_sha,
        "inner_sha256": inner_sha, "split_sha256": split_sha, "config_sha256": config_sha,
        "n_fit": "6", "n_val": "6", "n_test": "6", "est_gpu_sec": "1", "cap_group": "N14R",
        "conditional": "0", "truncation_rank": "99", "status": "pending",
    }
    unit_dir = units_dir / row["unit_id"]
    unit_dir.mkdir(parents=True)
    history = [
        {"epoch": 1, "val_loss": 0.7, "val_uar": 60.0},
        {"epoch": 2, "val_loss": 0.5, "val_uar": 75.0},
        {"epoch": 3, "val_loss": 0.5, "val_uar": 99.0},
    ]
    (unit_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    unit = {key: (int(row[key]) if key in {"fold", "r", "seed_index", "train_seed", "n_fit", "n_val", "n_test"} else row[key])
            for key in ("unit_id", "arm", "corpus_level", "model", "cell", "fold", "r", "seed_index", "train_seed",
                        "config_sha256", "split_sha256", "n_fit", "n_val", "n_test")}
    unit.update({"status": "done", "config": config, "best_epoch": 2, "epochs_run": 3,
                 "val_loss_best": 0.5, "val_uar_best": 75.0})
    pred_path = unit_dir / "predictions.csv"
    fields = ["sample_index", "relative_path", "speaker", "y_true", "y_pred", *[f"logit_{i}" for i in range(6)]]
    with pred_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in test:
            label = item["label_index"]
            logits = [0.0] * 6
            logits[label] = 1.0
            writer.writerow({"sample_index": item["sample_index"], "relative_path": item["relative_path"],
                             "speaker": item["speaker"], "y_true": label, "y_pred": label,
                             **{f"logit_{i}": logits[i] for i in range(6)}})
    unit["predictions_sha256"] = _sha(pred_path)
    (unit_dir / "unit.json").write_text(json.dumps(unit), encoding="utf-8")
    (unit_dir / "DONE").write_text(unit["predictions_sha256"] + "\n", encoding="utf-8")
    return row, units_dir, SplitStore(plan_dir, manifest), manifest, {config_sha: config}


def test_candidate_uses_first_minimum_loss_history_epoch_and_raw_test_fold(tmp_path: Path) -> None:
    row, units_dir, splits, manifest, configs = _candidate_fixture(tmp_path)
    candidate = _load_candidate(row, units_dir, splits, manifest, configs)
    assert candidate.val_uar == 75.0  # epoch 2, not the later tied epoch 3
    assert candidate.test_uar == 100.0

    unit_path = units_dir / row["unit_id"] / "unit.json"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    unit["val_uar_best"] = 99.0
    unit_path.write_text(json.dumps(unit), encoding="utf-8")
    with pytest.raises(ScoreError, match="val_uar_best differs"):
        _load_candidate(row, units_dir, splits, manifest, configs)


def test_prediction_tampering_voids_candidate(tmp_path: Path) -> None:
    row, units_dir, splits, manifest, configs = _candidate_fixture(tmp_path)
    pred_path = units_dir / row["unit_id"] / "predictions.csv"
    text = pred_path.read_text(encoding="utf-8").replace("audio/12.wav", "audio/99.wav")
    pred_path.write_text(text, encoding="utf-8")
    with pytest.raises(ScoreError, match="predictions hash mismatch"):
        _load_candidate(row, units_dir, splits, manifest, configs)


def test_fewer_than_24_complete_draws_is_not_tested_without_outcome_parsing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    completion = {
        "schema": "ser26-n14r-completion-1", "study_id": "N14R", "status": "not_tested_incomplete",
        "n_complete_draws": 23, "complete_draw_ids": list(range(23)),
    }
    (run_dir / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    result = score_run(
        root / "v2_1" / "n14r" / "plan",
        run_dir,
        manifest_path=root / "v2" / "manifests" / "cremad_manifest.csv",
    )
    assert not result["tested"]
    assert result["status"] == "not tested"
    assert result["n_complete_draws"] == 23
    assert result["primary"]["verdict"] == "not tested"
    assert result["primary"]["p_two_sided"] is None


def test_real_cremad_manifest_uses_frozen_hygiene_and_renumbering() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "v2" / "manifests" / "cremad_manifest.csv"
    manifest = _load_hygienic_manifest_file(path, _sha(path))
    assert len(manifest) == 7435
    assert sorted(row["sample_index"] for row in manifest.values()) == list(range(7435))
    assert {row["y_true"] for row in manifest.values()} == set(range(6))
    # H1 removes both members of each conflicting-byte duplicate group; H3
    # removes the preregistered unusable zero-variance file.
    for dropped in (
        "1006_TIE_HAP_XX.wav",
        "1006_TIE_NEU_XX.wav",
        "1013_WSI_DIS_XX.wav",
        "1013_WSI_SAD_XX.wav",
        "1017_IWW_ANG_XX.wav",
        "1017_IWW_FEA_XX.wav",
        "1076_MTI_SAD_XX.wav",
    ):
        assert dropped not in manifest

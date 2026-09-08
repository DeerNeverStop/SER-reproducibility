from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from v2_1.n14r2 import score


def test_six_class_uar_and_config_tie_break() -> None:
    assert score.macro_uar_6(range(6), range(6)) == 100.0
    with pytest.raises(score.ScoreError, match="absent"):
        score.macro_uar_6([0, 1], [0, 1])
    candidates = [score.Candidate(i, 70.0, float(i)) for i in reversed(range(8))]
    assert score.choose_config(candidates).config_index == 0


def test_t_and_bootstrap_are_deterministic() -> None:
    values = np.linspace(-1.0, 2.0, 24)
    summary = score.one_sample_t_summary(values)
    assert summary["n"] == 24 and summary["mean"] == pytest.approx(0.5)
    left = score.bootstrap_mean_summary(values, reps=200, seed=123)
    right = score.bootstrap_mean_summary(values, reps=200, seed=123)
    assert left == right


def test_incomplete_completion_is_fail_closed_without_outcome_read(tmp_path: Path) -> None:
    plan, run = tmp_path / "plan", tmp_path / "run"
    plan.mkdir(); run.mkdir()
    (run / "completion.json").write_text(json.dumps({
        "schema": "ser26-n14r2-completion-1", "study_id": "N14R2",
        "status": "not_tested_incomplete", "contains_outcome_statistics": False,
        "analysis_lock_payload_sha256": None,
    }), encoding="utf-8")
    poison = run / "nodes" / "node0" / "units" / "poison" / "attempt_01"
    poison.mkdir(parents=True)
    (poison / "predictions.csv").write_text("must-not-be-read", encoding="utf-8")
    ids, source, lock = score._validate_completion_lock(plan, run, [])
    assert ids == [] and lock is None and source["completion_status"] == "not_tested_incomplete"


def test_locked_attempt_path_rejects_escape_and_detects_mutation(tmp_path: Path) -> None:
    uid = "u0"
    unit_dir = tmp_path / "nodes" / "node0" / "units" / uid / "attempt_01"
    unit_dir.mkdir(parents=True)
    for name in ("unit.json", "history.json", "predictions.csv", "DONE"):
        (unit_dir / name).write_text(name, encoding="utf-8")
    receipt = unit_dir / "artifact_receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    entry = {
        "status": "done", "attempt": 1,
        "attempt_path": f"nodes/node0/units/{uid}/attempt_01",
        "artifact_receipt_sha256": score._sha256(receipt),
        "artifact_hashes": score._artifact_hashes(unit_dir),
    }
    row = {"unit_id": uid, "draw_id": "0"}
    assert score._safe_locked_unit_dir(tmp_path, row, entry)[0] == unit_dir
    entry["attempt_path"] = "../escape"
    with pytest.raises(score.ScoreError, match="unsafe"):
        score._safe_locked_unit_dir(tmp_path, row, entry)

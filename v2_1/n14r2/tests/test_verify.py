from __future__ import annotations

import ast
from pathlib import Path

import pytest

from v2_1.n14r2 import verify


def test_verifier_has_no_forbidden_runtime_imports() -> None:
    source = Path(verify.__file__).read_text(encoding="utf-8")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.endswith((".score", ".plan", ".stats", ".run", ".close")) for name in imported)


def test_machine_spec_2_is_accepted() -> None:
    spec_path = Path(verify.__file__).with_name("spec.json")
    assert verify._validate_spec(spec_path)["spec_version"] == "2.0.0"


def test_frozen_plan_splits_and_hygiene_replay_independently() -> None:
    study = Path(verify.__file__).parent
    plan = study / "plan"
    rows = verify._load_plan(plan)
    assert len(rows) == 2240
    manifest_sha = next(iter({row["manifest_sha256"] for row in rows}))
    manifest, _, hygiene = verify._load_manifest(plan, manifest_sha)
    verify._validate_hygiene(plan, manifest_sha, hygiene)
    assert len(manifest) == 7435
    assert len(verify._validate_plan_and_splits(plan, rows, manifest)) == 280


def test_verifier_selection_and_result_mutation_detection() -> None:
    candidates = [
        {"config_index": i, "val_uar": 80.0, "test_uar": float(i), "unit_id": f"u{i}"}
        for i in reversed(range(8))
    ]
    assert verify._select_episode(candidates)["config_index"] == 0
    differences = verify.compare_results(
        {"primary": {"p_two_sided": 0.01, "verdict": "supported"}},
        {"primary": {"p_two_sided": 0.02, "verdict": "not supported"}},
    )
    assert {item["path"] for item in differences} == {
        "$.primary.p_two_sided", "$.primary.verdict",
    }


def test_locked_attempt_path_is_exact_and_byte_bound(tmp_path: Path) -> None:
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
        "artifact_receipt_sha256": verify.sha256_file(receipt),
        "artifact_hashes": {
            name: verify.sha256_file(unit_dir / name)
            for name in ("unit.json", "history.json", "predictions.csv", "DONE")
        },
    }
    row = {"unit_id": uid, "draw_id": 0}
    assert verify._locked_attempt_dir(tmp_path, row, entry)[0] == unit_dir
    (unit_dir / "history.json").write_text("changed", encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="changed"):
        verify._locked_attempt_dir(tmp_path, row, entry)


def test_missing_lock_stops_before_outcome_replay(tmp_path: Path) -> None:
    with pytest.raises(verify.VerificationError, match="completion.json"):
        verify._validate_lock(tmp_path / "plan", tmp_path / "run", [], tmp_path / "spec.json")

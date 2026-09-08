"""Synthetic-only tests for the ignored metadata erratum helper."""
import csv
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "metadata_erratum_local", Path(__file__).with_name("metadata_erratum.py")
)
erratum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(erratum)


def encoded(value):
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


def materials():
    result = {
        "schema": erratum.RESULT_SCHEMA,
        "study_id": "N14R2",
        "spec_version": "2.0.0",
        "expected_complete_draws": 24,
        "n_planned_draws": 24,
        "n_complete_draws": 24,
        "analysis_draw_ids": list(range(24)),
        "tested": True,
        "status": "tested",
        # Invented opaque values ensure the helper preserves unrelated content.
        "primary": {"invented": 1.23456789012345},
        "draws": {"0": {"invented": -9.0}},
    }
    failure = {
        "schema": erratum.VERIFY_SCHEMA,
        "study_id": "N14R2",
        "pass": False,
        "differences": erratum.DIFFERENCE,
        "reconstructed": {"invented": "NEVER INSPECTED"},
    }
    lines = ["draw_id,opaque"]
    for draw in range(28):
        lines.extend(f"{draw},x" for _ in range(80))
    return encoded(result), encoded(failure), ("\n".join(lines) + "\n").encode()


def bind_hashes(monkeypatch, original, failure, plan):
    monkeypatch.setattr(erratum, "ORIGINAL_SHA", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(erratum, "FAILURE_SHA", hashlib.sha256(failure).hexdigest())
    monkeypatch.setattr(erratum, "PLAN_SHA", hashlib.sha256(plan).hexdigest())


def write_inputs(tmp_path, monkeypatch, *, original=None, failure=None, plan=None):
    defaults = materials()
    original = defaults[0] if original is None else original
    failure = defaults[1] if failure is None else failure
    plan = defaults[2] if plan is None else plan
    bind_hashes(monkeypatch, original, failure, plan)
    paths = [tmp_path / name for name in ("original.json", "failed.json", "plan.csv")]
    for path, data in zip(paths, (original, failure, plan)):
        path.write_bytes(data)
    return paths


def test_exact_one_byte_metadata_correction_preserves_everything_else(monkeypatch):
    original, failure, plan = materials()
    bind_hashes(monkeypatch, original, failure, plan)
    corrected, offset = erratum.correct_bytes(original, failure, plan)
    assert len(corrected) == len(original)
    assert [(i, a, b) for i, (a, b) in enumerate(zip(original, corrected)) if a != b] == [
        (offset, ord("4"), ord("8"))
    ]
    before, after = json.loads(original), json.loads(corrected)
    after["n_planned_draws"] = 24
    assert after == before


def test_cli_exclusive_outputs_receipt_and_input_preservation(tmp_path, monkeypatch, capsys):
    inputs = write_inputs(tmp_path, monkeypatch)
    output, receipt = tmp_path / "corrected.json", tmp_path / "receipt.json"
    before = [p.read_bytes() for p in inputs]
    erratum.main(["--original", str(inputs[0]), "--failure", str(inputs[1]),
                  "--plan", str(inputs[2]), "--output", str(output), "--receipt", str(receipt)])
    record = json.loads(receipt.read_text())
    assert record["contains_outcome_statistics"] is False
    assert record["objects_equal_except_declared_field"] is True
    assert record["corrected_result_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert [p.read_bytes() for p in inputs] == before
    stdout = capsys.readouterr().out
    assert "1.23456789012345" not in stdout and '"draws"' not in stdout
    with pytest.raises(ValueError, match="absent"):
        erratum.main(["--original", str(inputs[0]), "--failure", str(inputs[1]),
                      "--plan", str(inputs[2]), "--output", str(output), "--receipt", str(tmp_path / "r2")])


@pytest.mark.parametrize("which", ["original", "failure", "plan"])
def test_every_input_hash_is_mandatory(monkeypatch, which):
    original, failure, plan = materials()
    bind_hashes(monkeypatch, original, failure, plan)
    values = {"original": original, "failure": failure, "plan": plan}
    values[which] += b"x"
    with pytest.raises(ValueError, match="SHA-256"):
        erratum.correct_bytes(values["original"], values["failure"], values["plan"])


@pytest.mark.parametrize("mutation", ["schema", "pass", "extra_difference", "float_actual"])
def test_failed_verification_contract_is_exact(monkeypatch, mutation):
    original, failure, plan = materials()
    report = json.loads(failure)
    if mutation == "schema":
        report["schema"] = "wrong"
    elif mutation == "pass":
        report["pass"] = True
    elif mutation == "extra_difference":
        report["differences"].append({"path": "$.x", "expected": 1, "actual": 2})
    else:
        report["differences"][0]["actual"] = 24.0
    failure = encoded(report)
    bind_hashes(monkeypatch, original, failure, plan)
    with pytest.raises(ValueError, match="single metadata"):
        erratum.correct_bytes(original, failure, plan)


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("spec_version", "1.0.0"), ("tested", 1),
    ("n_planned_draws", 24.0), ("n_complete_draws", 24.0),
    ("analysis_draw_ids", [float(i) for i in range(24)]),
])
def test_result_metadata_types_and_schema_are_strict(monkeypatch, field, value):
    original, failure, plan = materials()
    result = json.loads(original)
    result[field] = value
    original = encoded(result)
    bind_hashes(monkeypatch, original, failure, plan)
    with pytest.raises(ValueError, match="metadata"):
        erratum.correct_bytes(original, failure, plan)


def test_plan_requires_exactly_eighty_rows_per_each_draw(monkeypatch):
    original, failure, plan = materials()
    rows = plan.decode().splitlines()
    # Preserve the total and set of IDs while making draw 0/1 counts 79/81.
    rows[1] = "1,x"
    plan = ("\n".join(rows) + "\n").encode()
    bind_hashes(monkeypatch, original, failure, plan)
    with pytest.raises(ValueError, match="28-draw inventory"):
        erratum.correct_bytes(original, failure, plan)


def test_inputs_and_target_ancestors_reject_reparse_points(tmp_path, monkeypatch):
    inputs = write_inputs(tmp_path, monkeypatch)
    marked = inputs[0]
    original_lstat = Path.lstat

    def lstat(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == marked.absolute():
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        erratum.main(["--original", str(inputs[0]), "--failure", str(inputs[1]),
                      "--plan", str(inputs[2]), "--output", str(tmp_path / "out"),
                      "--receipt", str(tmp_path / "receipt")])


def test_real_symlink_input_rejected_when_supported(tmp_path, monkeypatch):
    inputs = write_inputs(tmp_path, monkeypatch)
    link = tmp_path / "original-link.json"
    try:
        link.symlink_to(inputs[0])
    except OSError:
        pytest.skip("Native symlink privilege unavailable; reparse simulation covers the gate")
    with pytest.raises(ValueError, match="Symlink"):
        erratum.main(["--original", str(link), "--failure", str(inputs[1]),
                      "--plan", str(inputs[2]), "--output", str(tmp_path / "out"),
                      "--receipt", str(tmp_path / "receipt")])

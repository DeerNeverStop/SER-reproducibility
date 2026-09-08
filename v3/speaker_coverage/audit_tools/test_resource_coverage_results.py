"""Small synthetic receipt fixtures; no real predictions or trained tensors."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

import resource_coverage_results as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def seal(value, field="seal_sha256"):
    value[field] = r.digest({k: v for k, v in value.items() if k != field})
    return value


@pytest.fixture
def resource_phase(tmp_path):
    phase, root = "formal", tmp_path / "runs/formal"
    root.mkdir(parents=True)
    plan = {"plan_sha256": "a" * 64, "input": {"fixture_only": True}, "units": []}
    events, entries = [], []
    start = datetime(2026, 9, 6, tzinfo=timezone.utc)
    at = lambda second: (start + timedelta(seconds=second)).isoformat()
    for i, model in enumerate(r.MODELS):
        uid = r.digest(["resource fixture only", model])
        epochs = r.EPOCHS[model]
        config = {"epochs": epochs, "batch_size": 16 if model == "wavlm_ft" else 32}
        unit = {"unit_id": uid, "model": model, "block": "ft" if model == "wavlm_ft" else "core", "config": config,
                "fit": [f"synthetic{j}" for j in range(576)]}
        plan["units"].append(unit)
        # CNN: failed attempt then successful retry. FT: unclosed crash attempt
        # then successful retry. Ridge: one successful attempt.
        retry = model != "ridge_wavlm"
        attempt = 2 if retry else 1
        origin = i * 1000
        events.append({"event": "invocation_start", "at": at(origin)})
        if retry:
            events.append({"event": "unit_start", "at": at(origin + 1), "unit_id": uid, "attempt": 1})
            if model == "cnn":
                events.append({"event": "unit_failed", "at": at(origin + 21), "unit_id": uid, "attempt": 1, "error_type": "SyntheticFailure"})
        events.append({"event": "unit_start", "at": at(origin + 31), "unit_id": uid, "attempt": attempt})
        fit = {"cnn": 100.0, "ridge_wavlm": 2.0, "wavlm_ft": 150.0}[model]
        events.append({"event": "unit_done", "at": at(origin + 31 + fit + 1), "unit_id": uid, "attempt": attempt, "fit_seconds": fit})
        directory = root / "units" / uid
        prefix = f"attempts/{attempt:04d}/"
        artifact_root = directory / prefix
        artifact_root.mkdir(parents=True)
        checkpoint = "checkpoint.npz" if model == "ridge_wavlm" else "checkpoint.pt"
        for name in (checkpoint, "predictions.npz"):
            (artifact_root / name).write_bytes(b"SYNTHETIC: never deserialize")
        history = [{"epoch": epoch, "train_loss": 1.0, "val_loss": 1.0 / epoch,
                    "optimizer_steps": (576 + config["batch_size"] - 1) // config["batch_size"]} for epoch in range(1, epochs + 1)]
        write(artifact_root / "history.json", history)
        receipt = {"schema": "ser-speaker-coverage-result-1", "unit_id": uid, "plan_sha256": plan["plan_sha256"],
                   "phase": phase, "model": model, "block": unit["block"], "attempt": attempt,
                   "input_sha256": r.digest(plan["input"]), "unit_config_sha256": r.digest(config),
                   "epochs_run": epochs, "best_epoch": epochs if epochs else None, "checkpoint_reload_verified": True,
                   "checkpoint_format": r.FORMATS[model], "checkpoint_reload_best_max_abs_diff": 0.0,
                   "checkpoint_reload_last_max_abs_diff": 1e-7, "checkpoint_reload_max_abs_diff": 1e-7,
                   "fit_seconds": fit, "wall_seconds": fit + 0.5, "peak_cuda_bytes": 0 if model == "ridge_wavlm" else 1024 * (i + 1),
                   "finished_at": at(origin + 31 + fit + 1), "environment": {"device": "cpu" if model == "ridge_wavlm" else "cuda",
                       "gpu": None if model == "ridge_wavlm" else "Synthetic GPU",
                       "fit_timing": "model build + training + prediction + checkpoint replay", "wall_timing": "attempt setup to pre-receipt"}}
        write(artifact_root / "receipt.json", receipt)
        done = {k: receipt[k] for k in ("unit_id", "plan_sha256", "phase", "model", "block", "attempt")}
        done.update(schema="ser-speaker-coverage-done-1", artifacts={prefix + p.name: r.file_sha(p) for p in artifact_root.iterdir()})
        write(directory / "DONE", done)
        entries.append({"unit": unit, "directory": directory, "receipt": artifact_root / "receipt.json", "history": artifact_root / "history.json",
                        "done_sha": r.file_sha(directory / "DONE"), "fit": fit, "wall": fit + 0.5,
                        "artifact_bytes": sum(p.stat().st_size for p in artifact_root.iterdir())})
    for name in ("identity.json", "plan_snapshot.json", "verification.json"):
        write(root / name, {"synthetic": name})
    (root / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    block_audits = {}
    for block in ("core", "ft"):
        selected = [e for e in entries if e["unit"]["block"] == block]
        value = {"schema": "ser-speaker-coverage-independent-result-seal-1", "pass": True, "block_complete": True,
                 "phase": phase, "block": block, "plan_sha256": plan["plan_sha256"], "count": len(selected),
                 "done_sha256": {e["unit"]["unit_id"]: e["done_sha"] for e in selected},
                 "artifact_bytes_checked": sum(e["artifact_bytes"] for e in selected),
                 "summed_successful_fit_seconds": sum(e["fit"] for e in selected),
                 "summed_successful_unit_wall_seconds": sum(e["wall"] for e in selected)}
        for name, key in (("ledger.jsonl", "ledger_sha256"), ("identity.json", "output_identity_sha256"),
                          ("plan_snapshot.json", "executed_plan_snapshot_sha256"), ("verification.json", "independent_plan_gate_receipt_sha256")):
            value[key] = r.file_sha(root / name)
        block_audits[block] = seal(value)
    gate = seal({"schema": "ser-speaker-coverage-all-result-seal-1", "pass": True, "all_blocks_complete": True,
                 "phase": phase, "plan_sha256": plan["plan_sha256"], "count": 3, "blocks": block_audits})
    return plan, root, gate, entries


def check(fixture):
    plan, root, gate, _ = fixture
    return r.audit_phase(plan, "formal", plan["units"], root.parent, gate)


def test_resource_summary_separates_success_failure_and_unknown_cost(resource_phase):
    report, rows, attempts = check(resource_phase)
    summary = report["summary"]
    assert summary["successful_units"] == 3 and summary["started_attempts"] == 5
    assert summary["failed_attempts"] == 1 and summary["attempts_without_terminal_event"] == 1
    assert summary["successful_units_requiring_retry"] == 2
    assert summary["fit_wall_seconds"] == {"count": 3, "minimum": 2.0, "median": 100.0, "maximum": 150.0, "sum": 252.0}
    assert summary["summed_cuda_successful_fit_wall_seconds"] == 250 and summary["summed_cpu_successful_fit_wall_seconds"] == 2
    assert summary["epochs_run_total"] == 115 and summary["failed_attempt_observed_event_span_seconds"]["sum"] == 20
    assert summary["failed_attempt_fit_seconds"] is None and summary["unclosed_attempt_cost"] is None
    assert summary["peak_cuda_allocated_bytes"]["maximum"] == 3072 and "sum" not in summary["peak_cuda_allocated_bytes"]
    assert report["ledger_envelope"]["event_envelope_seconds"] > summary["fit_wall_seconds"]["sum"]
    assert len(rows) == 3 and len(attempts) == 5


def test_resource_audit_does_not_open_prediction_or_checkpoint_payloads(resource_phase, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name in ("predictions.npz", "checkpoint.pt", "checkpoint.npz"):
            raise AssertionError("resource audit must not load large prediction/checkpoint payloads")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    assert check(resource_phase)[0]["summary"]["successful_units"] == 3


@pytest.mark.parametrize("field,value,match", [
    ("epochs_run", 99, "epoch budget"), ("best_epoch", 99, "checkpoint receipt"),
    ("checkpoint_reload_verified", False, "checkpoint receipt"),
    ("checkpoint_reload_last_max_abs_diff", float("nan"), "nonfinite JSON"),
    ("wall_seconds", 1.0, "shorter than fit"), ("peak_cuda_bytes", 1.5, "must be an integer"),
])
def test_resealed_resource_receipt_still_has_to_match_runtime_contract(resource_phase, field, value, match):
    plan, root, _, entries = resource_phase
    entry = entries[0]
    receipt = r.read_json(entry["receipt"])
    receipt[field] = value
    write(entry["receipt"], receipt)
    done_path = entry["directory"] / "DONE"
    done = r.read_json(done_path)
    prefix = f"attempts/{done['attempt']:04d}/"
    done["artifacts"][prefix + "receipt.json"] = r.file_sha(entry["receipt"])
    write(done_path, done)
    with pytest.raises(ValueError, match=match):
        r.inspect_success(entry["unit"], "formal", root, plan, r.file_sha(done_path))


def test_gate_with_different_phase_or_tampered_seal_is_rejected(resource_phase):
    plan, root, gate, _ = resource_phase
    gate["phase"] = "pilot"
    with pytest.raises(ValueError, match="identity mismatch"):
        r.validate_gate(gate, plan, "formal", plan["units"], root)
    gate["phase"] = "formal"
    gate["seal_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content seal"):
        r.validate_gate(gate, plan, "formal", plan["units"], root)


def test_current_artifact_size_cannot_silently_differ_from_accepted_gate(resource_phase):
    path = resource_phase[-1][0]["directory"] / "attempts/0002/checkpoint.pt"
    path.write_bytes(path.read_bytes() + b"extra bytes")
    with pytest.raises(ValueError, match="byte count changed"):
        check(resource_phase)


def test_changed_receipt_or_ledger_needs_a_new_accepted_gate(resource_phase):
    path = resource_phase[-1][0]["receipt"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="current bytes differ"):
        check(resource_phase)


def test_changed_ledger_cannot_be_treated_as_original_gate(resource_phase):
    path = resource_phase[1] / "ledger.jsonl"
    path.write_text(path.read_text() + json.dumps({"event": "extra"}) + "\n")
    with pytest.raises(ValueError, match="current bytes differ"):
        check(resource_phase)


def test_missing_inputs_fail_without_guessing_rental_duration_or_charges(tmp_path, capsys):
    args = [part for name in ("repo", "plan", "outroot", "pilot-gate", "formal-gate") for part in ("--" + name, str(tmp_path / name))]
    output = tmp_path / "audit.json"
    assert r.main(args + ["--out", str(output)]) == 1
    report = r.read_json(output)
    assert report["pass"] is False and report["prediction_arrays_loaded"] is False
    assert report["scientific_metrics_read_or_computed"] is False
    assert json.loads(capsys.readouterr().out)["pass"] is False


def test_small_fixture_cannot_be_loaded_as_a_complete_frozen_plan(tmp_path, resource_phase):
    plan = deepcopy(resource_phase[0])
    plan.update(schema="ser-speaker-coverage-1", program="SER26-SPEAKER-COVERAGE-1")
    seal(plan, "plan_sha256")
    path = tmp_path / "synthetic-plan.json"
    write(path, plan)
    with pytest.raises(ValueError, match="all 720 formal units"):
        r.load_frozen_plan(tmp_path, path)

"""Tamper-and-reseal tests for result artifact integrity, without training."""
from copy import deepcopy
import json

import numpy as np
import pytest

from v3.speaker_coverage import verify as v


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def completed_ridge(tmp_path):
    unit = {"unit_id": "f" * 64, "model": "ridge_wavlm", "block": "core", "config": deepcopy(v.CONFIGS["ridge_wavlm"]),
            "train_seed": 234, "fit": [f"train{i}" for i in range(576)], "test": ["test0.wav", "test1.wav"]}
    plan = {"plan_sha256": "a" * 64, "input": {"manifest_sha256": "b" * 64}}
    root = tmp_path / unit["unit_id"]
    attempt = root / "attempts/0001"
    attempt.mkdir(parents=True)
    metadata = {"test0.wav": {"label_index": "0"}, "test1.wav": {"label_index": "1"}}
    logits = np.arange(12, dtype=np.float64).reshape(2, 6)
    ex = np.exp(logits - logits.max(1, keepdims=True))
    arrays = {"paths": np.array(unit["test"]), "labels": np.array([0, 1], dtype=np.int64)}
    for prefix in ("", "last_"):
        arrays.update({prefix + "logits": logits, prefix + "proba": ex / ex.sum(1, keepdims=True),
                       prefix + "pred": logits.argmax(1).astype(np.int64)})
    np.savez(attempt / "predictions.npz", **arrays)
    np.savez(attempt / "checkpoint.npz", coef=np.zeros((6, 768)), intercept=np.zeros(6), classes=np.arange(6),
             scaler_mean=np.zeros(768), scaler_scale=np.ones(768), scaler_var=np.ones(768),
             n_features_in=np.array(768, dtype=np.int64), n_samples_seen=np.array(576, dtype=np.int64))
    identity = {"unit_id": unit["unit_id"], "plan_sha256": plan["plan_sha256"], "phase": "formal", "model": unit["model"], "block": unit["block"], "attempt": 1}
    receipt = {"schema": "ser-speaker-coverage-result-1", **identity, "unit_config_sha256": v.digest(unit["config"]),
               "input_sha256": v.digest(plan["input"]), "environment": {"fixture": True}, "fit_seconds": 1.5,
               "wall_seconds": 2.0, "peak_cuda_bytes": 0, "checkpoint_reload_verified": True, "epochs_run": 0,
               "best_epoch": None, "checkpoint_format": "ridge_npz", "checkpoint_reload_best_max_abs_diff": 0,
               "checkpoint_reload_last_max_abs_diff": 0, "checkpoint_reload_max_abs_diff": 0}
    write_json(attempt / "receipt.json", receipt)
    write_json(attempt / "history.json", [])
    done = {"schema": "ser-speaker-coverage-done-1", **identity, "artifacts": {}}
    write_json(root / "DONE", done)
    fixture = {"unit": unit, "root": root, "plan": plan, "phase": "formal", "metadata": metadata, "attempt": attempt}
    reseal(fixture)
    return fixture


def reseal(fixture):
    root, attempt = fixture["root"], fixture["attempt"]
    done = json.loads((root / "DONE").read_text())
    done["artifacts"] = {p.relative_to(root).as_posix(): v.byte_hash(p) for p in attempt.iterdir()}
    write_json(root / "DONE", done)


def verify(fixture):
    return v.verify_result_unit(fixture["unit"], fixture["root"], fixture["plan"], fixture["phase"], fixture["metadata"])


def test_complete_unit_is_integrity_only_and_does_not_count_old_attempts(completed_ridge):
    report = verify(completed_ridge)
    assert report["ignored_uncommitted_attempts"] == 0 and report["artifact_bytes"] > 0
    assert "scores" not in report and "accuracy" not in report


@pytest.mark.parametrize("field,value,match", [
    ("input_sha256", "d" * 64, "input/config"), ("unit_config_sha256", "e" * 64, "input/config"),
    ("epochs_run", 1, "epoch/history"), ("best_epoch", 1, "minimum validation"),
    ("checkpoint_reload_verified", False, "replay not confirmed"),
    ("checkpoint_reload_last_max_abs_diff", float("nan"), "replay receipt"),
    ("wall_seconds", 0.1, "fit duration"), ("phase", "pilot", "identity"),
])
def test_resealed_receipt_tampering_fails(completed_ridge, field, value, match):
    path = completed_ridge["attempt"] / "receipt.json"
    document = json.loads(path.read_text())
    document[field] = value
    write_json(path, document)
    reseal(completed_ridge)
    with pytest.raises(ValueError, match=match + "|nonfinite"):
        verify(completed_ridge)


@pytest.mark.parametrize("field,value,match", [
    ("labels", np.array([1, 1], dtype=np.int64), "labels"),
    ("paths", np.array(["test1.wav", "test0.wav"]), "path order"),
    ("last_pred", np.array([0, 0], dtype=np.int64), "argmax"),
    ("last_proba", np.zeros((2, 6), dtype=np.float64), "probability"),
    ("last_logits", np.full((2, 6), np.nan), "logits"),
])
def test_resealed_prediction_tampering_fails(completed_ridge, field, value, match):
    path = completed_ridge["attempt"] / "predictions.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays[field] = value
    np.savez(path, **arrays)
    reseal(completed_ridge)
    with pytest.raises(ValueError, match=match):
        verify(completed_ridge)


def test_artifact_bytes_and_exact_set_required(completed_ridge):
    path = completed_ridge["attempt"] / "checkpoint.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="byte hash"):
        verify(completed_ridge)
    (completed_ridge["attempt"] / "surprise.json").write_text("{}")
    reseal(completed_ridge)
    with pytest.raises(ValueError, match="artifact set"):
        verify(completed_ridge)


def test_later_uncommitted_attempt_prevents_ambiguous_seal(completed_ridge):
    (completed_ridge["root"] / "attempts/0002").mkdir()
    with pytest.raises(ValueError, match="unsealed later"):
        verify(completed_ridge)


def test_resealed_truncated_scaler_rejected(completed_ridge):
    path = completed_ridge["attempt"] / "checkpoint.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["scaler_mean"] = np.zeros(767)
    np.savez(path, **arrays)
    reseal(completed_ridge)
    with pytest.raises(ValueError, match="scaler shape"):
        verify(completed_ridge)


def test_all_blocks_gate_never_returns_partial_success(monkeypatch):
    seen = []
    def block(*args, **kwargs):
        seen.append(args[4])
        if args[4] == "ft":
            raise v.VerificationError("missing FT unit")
        return {"count": 540, "plan_sha256": "a" * 64}
    monkeypatch.setattr(v, "verify_result", block)
    with pytest.raises(ValueError, match="missing FT"):
        v.verify_results("repo", "plan", "out", consolidate=True)
    assert seen == ["core", "ft"]


@pytest.fixture
def phase_receipts(completed_ridge):
    fixture = completed_ridge
    phase_root, unit, plan = fixture["root"].parent, fixture["unit"], fixture["plan"]
    write_json(phase_root / "identity.json", {"schema": "ser-speaker-coverage-output-1", "plan_sha256": plan["plan_sha256"],
                                             "phase": "formal", "program": "SER26-SPEAKER-COVERAGE-1"})
    write_json(phase_root / "plan_snapshot.json", plan)
    write_json(phase_root / "verification.json", {"pass": True, "formal_allowed": True, "plan_sha256": plan["plan_sha256"],
                                                 "scope": "complete_real_plan_and_actual_input_bytes"})
    at = "2026-09-06T00:00:00+00:00"
    events = [
        {"event": "invocation_start", "at": at, "plan_sha256": plan["plan_sha256"], "phase": "formal", "model": unit["model"],
         "selected_unit_ids": [unit["unit_id"]], "selection_sha256": v.digest([unit["unit_id"]])},
        {"event": "unit_start", "at": at, "unit_id": unit["unit_id"], "attempt": 1},
        {"event": "unit_done", "at": at, "unit_id": unit["unit_id"], "attempt": 1, "fit_seconds": 1.5},
        {"event": "invocation_end", "at": at, "phase": "formal", "scores_computed": False},
    ]
    entries = {unit["unit_id"]: verify(fixture)}
    return phase_root, plan, [unit], entries, events


def phase_check(fixture):
    root, plan, units, entries, events = fixture
    (root / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return v.verify_phase_receipts(root, plan, "formal", units, entries)


def test_ledger_matches_unique_done_and_original_plan(phase_receipts):
    report = phase_check(phase_receipts)
    assert report["ledger_event_count"] == 4 and report["ledger_failed_attempts"] == 0
    assert len(report["independent_plan_gate_receipt_sha256"]) == 64


@pytest.mark.parametrize("mutation,match", [
    (lambda e: e.insert(3, deepcopy(e[2])), "terminal event"),
    (lambda e: e[2].update(attempt=2), "terminal event"),
    (lambda e: e[2].update(fit_seconds=99), "missing/different"),
    (lambda e: e[0].update(selection_sha256="0" * 64), "invocation identity"),
    (lambda e: e[0].update(phase="pilot"), "invocation identity"),
    (lambda e: e.pop(2), "missing/different"),
])
def test_ledger_tampering_prevents_seal(phase_receipts, mutation, match):
    mutation(phase_receipts[-1])
    with pytest.raises(ValueError, match=match):
        phase_check(phase_receipts)


def test_fixture_gate_receipt_does_not_qualify_as_real_plan_gate(phase_receipts):
    root = phase_receipts[0]
    receipt = json.loads((root / "verification.json").read_text())
    receipt.update(formal_allowed=False, scope="test_fixture_only")
    write_json(root / "verification.json", receipt)
    with pytest.raises(ValueError, match="complete independent plan gate"):
        phase_check(phase_receipts)


def test_crash_window_recovery_requires_original_start_and_current_done_hash(phase_receipts):
    events, entries = phase_receipts[-1], phase_receipts[3]
    uid = phase_receipts[2][0]["unit_id"]
    events[2].update(recovered_from_done=True, done_sha256=entries[uid]["done_sha256"])
    events.insert(2, deepcopy(events[0]))
    events.insert(2, {"event": "stale_lock_recovered", "at": events[0]["at"], "previous_pid": 1234})
    assert phase_check(phase_receipts)["ledger_event_count"] == 6
    events[4]["done_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="recovered ledger DONE hash"):
        phase_check(phase_receipts)


def test_recovery_cannot_relabel_a_failed_attempt_as_success(phase_receipts):
    events, entries = phase_receipts[-1], phase_receipts[3]
    uid = phase_receipts[2][0]["unit_id"]
    events[2].update(recovered_from_done=True, done_sha256=entries[uid]["done_sha256"])
    events.insert(2, {"event": "unit_failed", "at": events[0]["at"], "unit_id": uid, "attempt": 1})
    with pytest.raises(ValueError, match="terminal event"):
        phase_check(phase_receipts)

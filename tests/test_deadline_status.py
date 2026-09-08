from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from v3.data_design import deadline_status as ds


NOW = datetime(2026, 9, 5, 7, tzinfo=timezone.utc)


def completed(n=100):
    return [{"event": "done", "unit_id": str(i),
             "at": (NOW - timedelta(minutes=n - i)).isoformat()} for i in range(n)]


def ridge_complete():
    return [{"event": "unit_done", "unit_id": str(i), "model": "ridge_wavlm", "wall_seconds": .1} for i in range(720)]


def test_duplicate_done_does_not_inflate_progress_or_rate():
    events = completed()
    result = ds.old_status(events + [events[-1]], NOW)
    assert result["unique_done"] == 100
    assert result["remaining_primary_estimate"] == 1820
    assert all(w["units_per_hour"] == 60 for w in result["throughput"])


def test_second_failure_withholds_naive_draw_eta():
    result = ds.old_status(completed() + [{"event": "failed", "unit_id": "x"}] * 2, NOW)
    assert result["remaining_primary_estimate"] is None
    assert result["training_eta_utc"] is None


def test_recovered_single_failure_keeps_eta():
    result = ds.old_status([{"event": "failed", "unit_id": "0"}] + completed(), NOW)
    assert result["unrecovered_failed_units"] == []
    assert result["training_eta_utc"] is not None


def test_stale_progress_is_not_presented_as_live_throughput():
    result = ds.old_status(completed(), NOW + timedelta(hours=2))
    assert result["stale_over_one_hour"]
    assert result["training_eta_utc"] is None


def test_cloud_alert_fires_before_compute_gate():
    old = ds.old_status(completed(), NOW)
    old["training_eta_utc"] = "2026-09-10T12:00:00+00:00"
    result = ds.forecast(NOW, old, ds.study_status(ridge_complete()))
    assert result["needs_attention"]
    assert "exceeds September 10" in result["reasons"][0]


def test_gpu_calendar_gate_uses_toronto_offset():
    old = ds.old_status(completed(), NOW)
    study = ds.study_status(ridge_complete())
    before = ds.forecast(ds.GPU_GATE - timedelta(seconds=1), old, study)
    after = ds.forecast(ds.GPU_GATE, old, study)
    assert not before["needs_attention"]
    assert "GPU unavailable" in after["reasons"][0]
    assert not ds.forecast(ds.GPU_GATE, old, study, gpu_ready=True)["needs_attention"]


def test_partial_cnn_timing_retains_budget_allowance_and_increases_if_slow():
    events = [{"event": "unit_done", "unit_id": str(i), "model": "cnn", "wall_seconds": 10} for i in range(8)]
    result = ds.study_status(events + [events[-1]])
    assert result["done_by_model"]["cnn"] == 8
    assert result["cnn_seconds_per_unit_allowance"] == 120
    events.append({"event": "unit_done", "unit_id": "slow", "model": "cnn", "wall_seconds": 200})
    assert ds.study_status(events)["cnn_seconds_per_unit_allowance"] == 300


def test_invalid_metadata_fails_explicitly():
    with pytest.raises(ValueError, match="timezone"):
        ds.moment("2026-09-05T07:00:00")
    with pytest.raises(ValueError, match="future"):
        ds.old_status(completed(), NOW - timedelta(days=1))
    with pytest.raises(ValueError, match="duration"):
        ds.study_status([{"event": "unit_done", "unit_id": "x", "model": "cnn", "wall_seconds": float("nan")}])


def test_recovered_done_uses_verified_metadata(tmp_path):
    uid = "a" * 64
    folder = tmp_path / "units" / uid
    folder.mkdir(parents=True)
    receipt = json.dumps({"unit_id": uid, "model": "cnn", "wall_seconds": 5}).encode()
    (folder / "unit.json").write_bytes(receipt)
    (folder / "DONE").write_text(json.dumps({"unit_json_sha256": hashlib.sha256(receipt).hexdigest()}))
    events = [{"event": "unit_done", "unit_id": uid, "attempt": 1, "recovered_from_done": True}]
    metadata = ds.recovery_metadata(events, tmp_path)
    assert ds.study_status(events, metadata)["done_by_model"]["cnn"] == 1
    (folder / "unit.json").write_bytes(receipt + b" ")
    with pytest.raises(ValueError, match="hash/identity"):
        ds.recovery_metadata(events, tmp_path)


def test_remaining_cpu_or_exhausted_fit_prevents_full_completion_eta():
    old = ds.old_status(completed(), NOW)
    cnn = [{"event": "unit_done", "unit_id": str(i), "model": "cnn", "wall_seconds": 5} for i in range(720)]
    result = ds.forecast(NOW, old, ds.study_status(cnn), gpu_ready=True)
    assert result["needs_attention"] and result["projected_compute_and_verification_end_utc"] is None
    failed = ridge_complete() + [{"event": "unit_failed", "unit_id": "failed-cnn", "attempt": 2}]
    result = ds.forecast(NOW, old, ds.study_status(failed), gpu_ready=True)
    assert result["needs_attention"] and result["projected_compute_and_verification_end_utc"] is None


def test_partial_snapshot_produces_attention_report(tmp_path, capsys):
    evidence = tmp_path / "v3/data_design/evidence"
    evidence.mkdir(parents=True)
    (evidence / "PLAN_LOCK.json").write_text('{"source_sha256": {}}')
    ledger = tmp_path / "v2_1/n14r/runs/n14r/ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b'{"event":')
    ds.main(["--repo", str(tmp_path), "--n14r-root", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["forecast"]["needs_attention"]
    assert "unfinished line" in result["input_error"]

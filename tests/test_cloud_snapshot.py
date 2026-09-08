"""Small TEST archives only: no real prediction files or NPZ decoding."""
import io
import json
from pathlib import Path
import tarfile

import pytest

from v3.data_design import cloud_snapshot as snapshot


@pytest.fixture
def active_snapshot(tmp_path):
    run, ops = tmp_path / "TEST-run", tmp_path / "TEST-ops"
    run.mkdir()
    ops.mkdir()
    uid, pending_uid = "a" * 64, "b" * 64
    folder = run / "units" / uid
    folder.mkdir(parents=True)
    environment = {"host": "TEST-host", "fixture": True}
    (run / "run_identity.json").write_text(json.dumps({"plan_sha256": snapshot.PLAN_SHA}), encoding="utf-8")
    (run / "environment_cnn.json").write_text(json.dumps(environment), encoding="utf-8")
    (run / ".core-run.lock").write_text('{"pid": 1, "host": "TEST-host"}', encoding="utf-8")
    done_event = {"event": "unit_done", "unit_id": uid, "attempt": 1}
    events = [done_event, done_event, {"event": "unit_start", "unit_id": pending_uid, "attempt": 1}]
    (run / "ledger.jsonl").write_bytes(b"".join(json.dumps(event).encode() + b"\n" for event in events))
    prediction_bytes = b"TEST fixture bytes; intentionally not a valid NPZ"
    receipt = json.dumps({"unit_id": uid, "plan_sha256": snapshot.PLAN_SHA,
                          "model": "cnn", "environment": environment, "attempt": 1,
                          "predictions_sha256": snapshot.digest(prediction_bytes)}).encode()
    (folder / "predictions.npz").write_bytes(prediction_bytes)
    (folder / "unit.json").write_bytes(receipt)
    (folder / "DONE").write_text(json.dumps({
        "unit_id": uid, "plan_sha256": snapshot.PLAN_SHA,
        "unit_json_sha256": snapshot.digest(receipt),
        "predictions_sha256": snapshot.digest(prediction_bytes),
    }), encoding="utf-8")
    pending = run / "units" / pending_uid
    pending.mkdir()
    (pending / "predictions.npz").write_bytes(b"TEST incomplete unit, never include")
    # A previous successful process status cannot make this active block complete.
    (ops / "full.status").write_text("0", encoding="utf-8")
    captured = snapshot.capture(run, ops)
    return {"run": run, "uid": uid, "pending_uid": pending_uid,
            "prediction_bytes": prediction_bytes, "captured": captured}


def test_active_snapshot_remains_partial_and_excludes_unfinished_units(active_snapshot):
    case = active_snapshot
    captured = case["captured"]
    assert captured["cnn_done"] == 1
    assert captured["cnn_block_stopped_complete"] is False
    assert captured["scientific_metrics_read"] is False
    with tarfile.open(captured["archive"], "r:gz") as archive:
        names = archive.getnames()
        report = json.load(archive.extractfile("snapshot.json"))
    assert "core/.core-run.lock" in names
    assert not any(case["pending_uid"] in name for name in names)
    assert report["cnn_done"] == 1 and report["cnn_block_stopped_complete"] is False


def test_valid_roundtrip_preserves_all_hashed_bytes(active_snapshot, tmp_path):
    case = active_snapshot
    captured = case["captured"]
    destination = tmp_path / "TEST-restored"
    result = snapshot.restore(captured["archive"], captured["sha256"], destination)
    assert result["pass"] is True
    assert result["cnn_done"] == 1 and result["cnn_block_stopped_complete"] is False
    assert result["scientific_metrics_read"] is False
    report = json.loads((destination / "snapshot.json").read_text(encoding="utf-8"))
    assert result["files_verified"] == len(report["files_sha256"])
    for relative, expected in report["files_sha256"].items():
        assert snapshot.digest((destination / relative).read_bytes()) == expected
    assert (destination / "core/units" / case["uid"] / "predictions.npz").read_bytes() == case["prediction_bytes"]
    assert (destination / "core/ledger.jsonl").read_bytes() == (case["run"] / "ledger.jsonl").read_bytes()


def test_existing_destination_is_never_overwritten(active_snapshot, tmp_path):
    captured = active_snapshot["captured"]
    destination = tmp_path / "TEST-existing"
    destination.mkdir()
    sentinel = destination / "existing.txt"
    sentinel.write_bytes(b"TEST existing data")
    with pytest.raises(ValueError, match="already exists; never overwrite"):
        snapshot.restore(captured["archive"], captured["sha256"], destination)
    assert sentinel.read_bytes() == b"TEST existing data"
    assert list(destination.iterdir()) == [sentinel]


def test_transfer_tampering_is_rejected_before_destination_creation(active_snapshot, tmp_path):
    captured = active_snapshot["captured"]
    archive = Path(captured["archive"])
    with archive.open("ab") as stream:
        stream.write(b"TEST transfer corruption")
    destination = tmp_path / "TEST-tampered-output"
    with pytest.raises(ValueError, match="Transfer SHA mismatch"):
        snapshot.restore(archive, captured["sha256"], destination)
    assert not destination.exists()


def test_traversal_is_rejected_before_any_extraction(tmp_path):
    archive = tmp_path / "TEST-traversal.tar.gz"
    unsafe_name = "../TEST-escaped"
    unsafe_bytes = b"TEST must not escape"
    report = {"schema": "ser-study2-cloud-snapshot-1", "plan_sha256": snapshot.PLAN_SHA,
              "cnn_done": 0, "cnn_block_stopped_complete": False,
              "files_sha256": {unsafe_name: snapshot.digest(unsafe_bytes)}}
    with tarfile.open(archive, "w:gz") as target:
        for name, payload in (("snapshot.json", json.dumps(report).encode()), (unsafe_name, unsafe_bytes)):
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            target.addfile(item, io.BytesIO(payload))
    destination = tmp_path / "TEST-traversal-output"
    with pytest.raises(ValueError, match="Unsafe archive member"):
        snapshot.restore(archive, snapshot.digest(archive.read_bytes()), destination)
    assert not destination.exists()
    assert not (tmp_path / "TEST-escaped").exists()

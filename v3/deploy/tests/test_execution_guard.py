"""Adversarial tests for orchestration; synthetic files only, no training."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from v3.deploy import execution as e
from v3.deploy import common as c


def guard(tmp_path, deadline=None):
    return e.RunGuard(tmp_path, {"lock_sha256": "a" * 64}, deadline or datetime.now(timezone.utc) + timedelta(hours=1))


def test_restart_cannot_reset_two_attempt_limit(tmp_path):
    u = {"unit_id": "a" * 64}
    guard(tmp_path).before(u)
    guard(tmp_path).before(u)
    with pytest.raises(c.IntegrityError, match="two attempts"):
        guard(tmp_path).before(u)


def test_deadline_refuses_start_without_increment(tmp_path):
    g = guard(tmp_path, datetime.now(timezone.utc) - timedelta(seconds=1))
    with pytest.raises(c.IntegrityError, match="deadline"):
        g.before({"unit_id": "x"})
    assert not g.journal.exists()


def test_done_without_seal_is_not_silently_reused(tmp_path):
    (tmp_path / "DONE").write_text("unknown")
    with pytest.raises(c.IntegrityError, match="without managed seal"):
        guard(tmp_path).sealed({"unit_id": "x"}, tmp_path)


def test_sealed_checkpoint_and_plan_metadata_are_bound(tmp_path):
    unit = {"unit_id": "x", "train_seed": 7, "config": {"epochs": 2}}
    for name in ["checkpoint.pt", "val_predictions.csv", "test_predictions.csv", "DONE", "history.json"]:
        (tmp_path / name).write_text(name)
    meta = {**unit, "engine_info": {"checkpoint_sha256": c.sha256_file(tmp_path / "checkpoint.pt")}}
    c.atomic_write_json(tmp_path / "unit.json", meta)
    g = guard(tmp_path)
    g.finish(unit, tmp_path)
    assert g.sealed(unit, tmp_path)
    with pytest.raises(c.IntegrityError, match="metadata differs"):
        g.sealed({**unit, "train_seed": 8}, tmp_path)
    (tmp_path / "checkpoint.pt").write_text("corrupt")
    with pytest.raises(c.IntegrityError, match="corrupted"):
        g.sealed(unit, tmp_path)


def test_missing_required_lock_file_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path)
    with pytest.raises(c.IntegrityError, match="required lock input missing"):
        e.source_files()


def test_concurrent_runner_rejected_and_existing_lock_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "verify_lock", lambda *a: {"lock_sha256": "x"})
    monkeypatch.setattr(e, "environment", lambda p: {"profile": p})
    (tmp_path / ".managed.lock").write_text("someone else")
    with pytest.raises(FileExistsError):
        with e.run_context(tmp_path, tmp_path / "lock.json", datetime.now(timezone.utc)):
            pass
    assert (tmp_path / ".managed.lock").read_text() == "someone else"


def test_environment_drift_refuses_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "verify_lock", lambda *a: {"lock_sha256": "x"})
    monkeypatch.setattr(e, "environment", lambda p: {"profile": p, "host": "new"})
    c.atomic_write_json(tmp_path / "execution_environment.json", {"lock_sha256": "x", "environment": {"host": "old"}})
    with pytest.raises(c.IntegrityError, match="environment/lock changed"):
        with e.run_context(tmp_path, tmp_path / "lock.json", datetime.now(timezone.utc)):
            pass
    assert not (tmp_path / ".managed.lock").exists()

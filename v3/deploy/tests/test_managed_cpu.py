"""Synthetic operational checks; no corpus caches, fitting, or scientific scores."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from v3.deploy import common as C
from v3.deploy import managed_cpu as M


def make_jobs(n=1):
    return [{"module": "B2", "unit": {"unit_id": f"u{i}", "model": "ridge_toy"}, "plan_file_sha256": "b" * 64}
            for i in range(n)]


LOCK = {"lock_sha256": "a" * 64}


class Worker:
    def __init__(self, failures=0):
        self.failures, self.calls = failures, Counter()

    def run(self, job, udir):
        self.calls[M.job_key(job)] += 1
        if self.calls[M.job_key(job)] <= self.failures:
            raise RuntimeError("synthetic transient fitting failure")
        udir.mkdir(parents=True, exist_ok=True)
        for filename in M.REQUIRED_FILES[job["module"]]:
            C.atomic_write_text(udir / filename, json.dumps(job["unit"]) + "\n")

    def validate(self, job, udir):
        for filename in M.REQUIRED_FILES[job["module"]]:
            C.require(json.loads((udir / filename).read_text()) == job["unit"], "synthetic identity mismatch")


def test_exclusive_queue_blocks_concurrent_and_releases(tmp_path):
    with M.exclusive_queue(tmp_path):
        assert (tmp_path / ".managed_cpu.lock").is_file()
        with pytest.raises(C.IntegrityError, match="second queue"):
            with M.exclusive_queue(tmp_path):
                pass
    assert not (tmp_path / ".managed_cpu.lock").exists()


def test_one_failure_then_retry_is_durable_and_resume_skips(tmp_path):
    jobs = make_jobs(2)
    guard = M.QueueGuard(tmp_path, LOCK, jobs)
    worker = Worker(failures=1)
    result = M.run_queue(guard, worker)
    assert result["complete"] and result["planned_units"] == 2
    assert sum(guard.attempts.values()) == 4
    rows = M.read_journal(guard.journal)
    assert Counter(r["event"] for r in rows) == {"start": 4, "failed": 2, "done": 2}
    assert result["scientific_scores_computed"] is False
    resumed = M.QueueGuard(tmp_path, LOCK, jobs)
    second_worker = Worker()
    assert M.run_queue(resumed, second_worker)["complete"]
    assert not second_worker.calls
    assert M.read_journal(guard.journal) == rows


def test_two_attempt_limit_persists_across_restarts(tmp_path):
    jobs = make_jobs()
    first = M.QueueGuard(tmp_path, LOCK, jobs)
    # An interrupted started attempt is already charged, even without failure.
    first.before(jobs[0])
    resumed = M.QueueGuard(tmp_path, LOCK, jobs)
    worker = Worker(failures=9)
    result = M.run_queue(resumed, worker)
    assert result["complete"] is False and result["failed_units"] == 1
    assert sum(worker.calls.values()) == 1
    third = M.QueueGuard(tmp_path, LOCK, jobs)
    final_worker = Worker()
    assert not M.run_queue(third, final_worker)["complete"]
    assert not final_worker.calls
    assert not (third.control / "COMPLETE.json").exists()


def test_truncated_journal_refused_without_resetting_attempts(tmp_path):
    jobs = make_jobs()
    guard = M.QueueGuard(tmp_path, LOCK, jobs)
    guard.before(jobs[0])
    with guard.journal.open("ab") as fh:
        fh.write(b'{"event":"start"')
    with pytest.raises(C.IntegrityError, match="truncated"):
        M.QueueGuard(tmp_path, LOCK, jobs)


def test_changed_lock_or_plan_receipt_refused(tmp_path):
    jobs = make_jobs()
    guard = M.QueueGuard(tmp_path, LOCK, jobs)
    assert M.run_queue(guard, Worker())["complete"]
    with pytest.raises(C.IntegrityError, match="identity drift"):
        M.QueueGuard(tmp_path, {"lock_sha256": "d" * 64}, jobs)
    changed = make_jobs()
    changed[0]["plan_file_sha256"] = "f" * 64
    other = M.QueueGuard(tmp_path, LOCK, changed)
    with pytest.raises(C.IntegrityError, match="identity mismatch"):
        M.run_queue(other, Worker())


def test_raw_done_recovered_only_after_managed_start(tmp_path):
    jobs = make_jobs()
    guard = M.QueueGuard(tmp_path, LOCK, jobs)
    worker = Worker()
    worker.run(jobs[0], guard.unit_dir(jobs[0]))
    with pytest.raises(C.IntegrityError, match="unmanaged"):
        guard.sealed(jobs[0], worker.validate)
    guard.before(jobs[0])
    assert guard.sealed(jobs[0], worker.validate)
    assert guard.attempts[M.job_key(jobs[0])] == 1


def test_corrupted_or_truncated_seal_refused(tmp_path):
    jobs = make_jobs()
    guard = M.QueueGuard(tmp_path, LOCK, jobs)
    worker = Worker()
    assert M.run_queue(guard, worker)["complete"]
    udir = guard.unit_dir(jobs[0])
    seal_path = udir / "CPU_SEALED.json"
    saved = seal_path.read_bytes()
    seal_path.write_bytes(b'{"schema":')
    with pytest.raises(C.IntegrityError, match="truncated or malformed"):
        guard.sealed(jobs[0], worker.validate)
    seal_path.write_bytes(saved)
    (udir / "test_predictions.csv").write_text("tampered")
    with pytest.raises(C.IntegrityError, match="output drift"):
        guard.sealed(jobs[0], worker.validate)


def test_missing_sealed_file_cannot_be_called_complete(tmp_path):
    jobs = make_jobs(2)
    guard = M.QueueGuard(tmp_path, LOCK, jobs)
    worker = Worker()
    guard.before(jobs[0])
    worker.run(jobs[0], guard.unit_dir(jobs[0]))
    guard.seal(jobs[0], worker.validate)
    with pytest.raises(C.IntegrityError, match="incomplete"):
        guard.finish(worker.validate)
    assert not (guard.control / "COMPLETE.json").exists()


def test_source_drift_checked_before_any_attempt(tmp_path):
    guard = M.QueueGuard(tmp_path, LOCK, make_jobs())
    worker = Worker()
    def fail_source():
        raise C.IntegrityError("source drift")
    with pytest.raises(C.IntegrityError, match="source drift"):
        M.run_queue(guard, worker, fail_source)
    assert not guard.attempts and not worker.calls


def test_cpu_worker_A_raw_roundtrip_uses_no_scientific_scorer(tmp_path, monkeypatch):
    from .test_enroll import make_manifest, make_features, make_tables, make_plan, LEVEL
    import numpy as np
    manifest = make_manifest()
    features = make_features(manifest)
    plan = make_plan(manifest, make_tables(manifest))
    class Cache:
        def get(self, paths, state=None):
            return np.asarray([features[p] for p in paths], dtype=np.float64)
    monkeypatch.setattr(M.F, "validate_plan", lambda *_: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("scientific scorer called from managed queue")
    monkeypatch.setattr(M.A, "score_run", forbidden)
    monkeypatch.setattr(M.B, "score_plan", forbidden)
    monkeypatch.setattr(M.F, "score_plan", forbidden)
    worker = M.CpuWorker({"A": plan, "B2-FIX": {}}, tmp_path, {"threads": 2})
    monkeypatch.setattr(worker, "manifest", lambda *_: manifest)
    monkeypatch.setattr(worker, "get_cache", lambda *_: Cache())
    jobs = [{"module": "A", "unit": plan["units"][0], "plan_file_sha256": "b" * 64}]
    guard = M.QueueGuard(tmp_path, LOCK, jobs)
    result = M.run_queue(guard, worker)
    assert result["complete"] and result["counts"] == {"A": 1}
    assert (guard.unit_dir(jobs[0]) / "references.json.gz").is_file()

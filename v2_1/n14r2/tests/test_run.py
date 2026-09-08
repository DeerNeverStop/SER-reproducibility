"""Deterministic scheduler fault tests: no CUDA, engine, corpus or real outcomes."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2_1.n14r2 import run, worker


def rows_for(draw=0, count=3):
    return [{"unit_id": worker.hash_json([draw, i]), "study_id": "N14R2", "arm": "N14R2",
             "draw_id": str(draw), "fold": "0", "cell": "GR_hpo", "config_index": str(i)} for i in range(count)]


def fake_artifacts(task, failure=None):
    directory = Path(task["attempt_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if failure:
        receipt = worker.receipt_for(task, status="failed", failure_class=failure)
    else:
        # Intentionally NOT JSON/CSV: coordinator must only hash, never parse
        # model outputs. Worker validation is tested separately in frozen tests.
        for name in worker.ARTIFACT_NAMES:
            (directory / name).write_bytes(b"opaque-test-artifact\n")
        receipt = worker.receipt_for(task, status="done", artifact_hashes={n: worker.hash_file(directory / n) for n in worker.ARTIFACT_NAMES})
    worker.publish_json(directory / "artifact_receipt.json", receipt)
    return {"event": "result", "unit_id": task["row"]["unit_id"], "attempt": task["attempt"],
            "status": receipt["status"], "failure_class": receipt["failure_class"],
            "receipt_sha256": worker.hash_file(directory / "artifact_receipt.json")}


class FakePool:
    def __init__(self, count, settings):
        self.messages = [{"event": "ready", "slot": i, "pid": i + 100, "cuda_healthy": True, "health_checksum": 4096.0} for i in range(count)]
        self.sent, self.open, self.closed = [], False, False
        self.failure = settings.get("failure")
        self.health_failure = settings.get("health_failure", False)
        self.crash = settings.get("crash", False)
        self.duplicate = settings.get("duplicate", False)
        self.pause_on_send = settings.get("pause_on_send", False)
        self.pause_file = settings.get("pause_file")

    def receive(self, timeout=0.25):
        if self.health_failure:
            self.health_failure = False
            self.open = True
            return {"event": "worker_error", "slot": 0}
        return self.messages.pop(0) if self.messages else None

    def send(self, slot, task):
        self.sent.append(task)
        if self.pause_on_send:
            Path(self.pause_file).write_text("operator budget pause\n", encoding="utf-8")
        if self.crash:
            self.open = True
            return
        result = fake_artifacts(task, self.failure)
        result["slot"] = slot
        self.messages.append(result)
        if self.duplicate:
            self.messages.append(dict(result))
        if self.failure:
            self.open = True

    def stopped(self):
        return self.open

    def dead(self):
        return []

    def close(self, abort=False):
        self.closed = True


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.node = self.root / "nodes" / "node0"
        self.node.mkdir(parents=True)
        self.rows = rows_for()
        self.pools = []

    def factory(self, count, settings):
        pool = FakePool(count, settings)
        self.pools.append(pool)
        return pool

    def coordinate(self, **settings):
        ledger = run.Ledger(self.node, 0)
        result = run.coordinate(self.node, self.rows, {}, None, 0, ledger,
                                {0: {"authorization_id": "primary0"}}, settings,
                                workers=2, pool_factory=self.factory, health_timeout=1, unit_timeout=1)
        return result, ledger

    def orphan(self, *, receipt=False, raw_done=False):
        ledger = run.Ledger(self.node, 0)
        ledger.invocation_id = "old-process"
        ledger.append("runner_start")
        row = self.rows[0]
        start = ledger.append("start", unit_id=row["unit_id"], draw_id=0, attempt=1)
        task = run.task_for(row, start, self.node, 0)
        Path(task["attempt_dir"]).mkdir(parents=True)
        if receipt:
            fake_artifacts(task)
        elif raw_done:
            for name in worker.ARTIFACT_NAMES:
                (Path(task["attempt_dir"]) / name).write_bytes(b"unsealed")
        return ledger, task

    def test_success_resume_never_repeats_completed_units(self):
        result, ledger = self.coordinate()
        self.assertEqual(result["status"], "authorized_draws_terminal")
        self.assertEqual(sum(e["event"] == "start" for e in ledger.records), 3)
        result, ledger = self.coordinate()
        self.assertEqual(len(self.pools), 1)
        self.assertEqual(sum(e["event"] == "start" for e in ledger.records), 3)
        for row in self.rows:
            pointer = worker.read_json(self.node / "units" / row["unit_id"] / "DONE.json")
            self.assertEqual(set(pointer), {"attempt", "artifact_receipt_sha256"})

    def test_no_coordinator_parses_raw_artifact_json(self):
        # Fake artifacts are invalid JSON by construction; byte hashes suffice.
        result, _ = self.coordinate()
        self.assertEqual(result["draws"][0]["status"], "complete")

    def test_health_failure_allocates_zero_attempts(self):
        result, ledger = self.coordinate(health_failure=True)
        self.assertEqual(result["status"], "circuit_open_resume_required")
        self.assertFalse(any(e["event"] == "start" for e in ledger.records))
        self.assertEqual(ledger.records[-1]["event"], "circuit_break")

    def test_cuda_failure_stops_dispatch_and_requires_fresh_pool(self):
        result, ledger = self.coordinate(failure="cuda")
        self.assertEqual(len(self.pools[0].sent), 1)
        self.assertEqual(ledger.records[-1]["event"], "circuit_break")
        self.assertEqual(result["status"], "circuit_open_resume_required")
        result, ledger = self.coordinate()
        self.assertEqual(len(self.pools), 2)
        attempts = run.parse_attempts(ledger.records, self.rows, 0)
        self.assertEqual(len(attempts[self.rows[0]["unit_id"]]), 2)
        self.assertEqual(result["draws"][0]["status"], "complete")

    def test_two_cuda_failures_block_not_void_no_third_attempt(self):
        self.coordinate(failure="cuda")
        result, _ = self.coordinate(failure="cuda")
        self.assertEqual(result["draws"][0]["status"], "blocked")
        result, ledger = self.coordinate()
        self.assertEqual(len(self.pools), 2)
        self.assertEqual(result["status"], "blocked_not_void")
        self.assertEqual(sum(e["event"] == "start" for e in ledger.records), 2)
        self.assertFalse((self.node / "draws" / "draw_00.json").exists())

    def test_two_explicit_training_failures_void(self):
        self.coordinate(failure="training")
        result, _ = self.coordinate(failure="training")
        self.assertEqual(result["draws"][0]["status"], "void")
        self.assertEqual(worker.read_json(self.node / "draws" / "draw_00.json")["status"], "void")

    def test_mixed_exhaustion_is_not_void(self):
        self.coordinate(failure="cuda")
        result, _ = self.coordinate(failure="training")
        self.assertEqual(result["draws"][0]["status"], "blocked")

    def test_worker_exit_conservatively_consumes_attempt(self):
        result, ledger = self.coordinate(crash=True)
        failures = [e for e in ledger.records if e["event"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["failure_class"], "circuit_cancelled")
        self.assertEqual(result["status"], "circuit_open_resume_required")

    def test_orphan_receipt_recovers_done_without_retry(self):
        ledger, task = self.orphan(receipt=True)
        attempts = run.reconcile(self.node, self.rows, 0, ledger, {}, None)
        self.assertEqual(attempts[task["row"]["unit_id"]][0]["terminal"]["event"], "done")
        self.assertTrue((Path(task["attempt_dir"]).parent / "DONE.json").exists())

    def test_orphan_raw_done_without_receipt_is_interrupted(self):
        ledger, task = self.orphan(raw_done=True)
        attempts = run.reconcile(self.node, self.rows, 0, ledger, {}, None)
        terminal = attempts[task["row"]["unit_id"]][0]["terminal"]
        self.assertEqual((terminal["event"], terminal["failure_class"]), ("failed", "interrupted"))
        self.assertEqual((Path(task["attempt_dir"]) / "DONE").read_bytes(), b"unsealed")
        self.assertFalse((Path(task["attempt_dir"]).parent / "DONE.json").exists())

    def test_missing_attempt_directory_after_start_is_reconciled(self):
        ledger = run.Ledger(self.node, 0)
        ledger.invocation_id = "crash"
        ledger.append("runner_start")
        ledger.append("start", unit_id=self.rows[0]["unit_id"], draw_id=0, attempt=1)
        attempts = run.reconcile(self.node, self.rows, 0, ledger, {}, None)
        self.assertEqual(attempts[self.rows[0]["unit_id"]][0]["terminal"]["failure_class"], "interrupted")

    def test_missing_receipt_after_terminal_is_fatal(self):
        self.coordinate()
        uid = self.rows[0]["unit_id"]
        (self.node / "units" / uid / "attempt_01" / "artifact_receipt.json").unlink()
        with self.assertRaisesRegex(worker.IntegrityError, "missing or changed receipt"):
            self.coordinate()

    def test_changed_artifact_is_fatal(self):
        self.coordinate()
        uid = self.rows[0]["unit_id"]
        (self.node / "units" / uid / "attempt_01" / "history.json").write_bytes(b"changed")
        with self.assertRaisesRegex(worker.IntegrityError, "missing/changed"):
            self.coordinate()

    def test_extra_or_foreign_attempt_directory_is_fatal(self):
        self.coordinate()
        (self.node / "units" / self.rows[0]["unit_id"] / "attempt_03").mkdir()
        with self.assertRaisesRegex(worker.IntegrityError, "orphan/extra"):
            self.coordinate()

    def test_duplicate_done_pointer_is_not_overwritten(self):
        self.coordinate()
        with self.assertRaises(worker.IntegrityError):
            run.publish_pointer(self.node, self.rows[0]["unit_id"], 2, "0" * 64)

    def test_atomic_publication_never_replaces(self):
        path = self.root / "immutable.json"
        worker.publish_json(path, {"original": True})
        with self.assertRaises(FileExistsError):
            worker.publish_json(path, {"original": False})
        self.assertEqual(worker.read_json(path), {"original": True})

    def test_duplicate_start_terminal_or_sequence_is_rejected(self):
        _, ledger = self.coordinate()
        records = ledger.records
        start = next(e for e in records if e["event"] == "start")
        with self.assertRaises(worker.IntegrityError):
            run.parse_attempts([*records, start], self.rows, 0)
        with (self.node / "ledger.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(worker.canonical(records[-1]) + "\n")
        with self.assertRaisesRegex(worker.IntegrityError, "sequence"):
            run.Ledger(self.node, 0)

    def test_torn_ledger_fails_closed_without_repair(self):
        path = self.node / "ledger.jsonl"
        path.write_bytes(b'{"seq":1')
        before = path.read_bytes()
        with self.assertRaisesRegex(worker.IntegrityError, "torn"):
            run.read_jsonl(path)
        self.assertEqual(path.read_bytes(), before)

    def test_post_failure_start_is_rejected(self):
        _, ledger = self.coordinate(failure="cuda")
        row = self.rows[1]
        ledger.append("start", unit_id=row["unit_id"], draw_id=0, attempt=1)
        with self.assertRaisesRegex(worker.IntegrityError, "post-failure"):
            run.parse_attempts(ledger.records, self.rows, 0)

    def test_preflight_never_dispatches_fits(self):
        result = run.preflight_pool(self.node, {}, workers=2, health_timeout=1, pool_factory=self.factory)
        self.assertEqual(result["fit_attempts_dispatched"], 0)
        self.assertFalse(self.pools[0].sent)
        self.assertTrue(self.pools[0].closed)
        self.assertFalse((self.node / "units").exists())
        self.assertFalse((self.node / "ledger.jsonl").exists())

    def test_pause_drains_active_fit_without_failure_or_attempt_loss(self):
        pause = self.root / "PAUSE_REQUEST"
        result, ledger = self.coordinate(pause_file=str(pause), pause_on_send=True)
        self.assertEqual(result["status"], "paused_operator")
        self.assertEqual(len(self.pools[0].sent), 1)
        self.assertEqual(sum(e["event"] == "done" for e in ledger.records), 1)
        self.assertFalse(any(e["event"] in {"failed", "circuit_break"} for e in ledger.records))
        self.assertEqual(ledger.records[-1]["pause_request_sha256"], worker.hash_file(pause))
        before = sum(e["event"] == "start" for e in ledger.records)
        result, ledger = self.coordinate(pause_file=str(pause))
        self.assertEqual(result["status"], "paused_operator")
        self.assertEqual(sum(e["event"] == "start" for e in ledger.records), before)
        self.assertTrue(pause.exists())

    def test_lock_blocks_second_owner_and_survives_stale_filename(self):
        path = self.node / ".coordinator.lock"
        with worker.file_lock(path):
            with self.assertRaises(worker.IntegrityError):
                with worker.file_lock(path):
                    self.fail("second owner acquired lock")
        with worker.file_lock(path):
            self.assertTrue(path.exists())

    def test_failure_classifier_conservative(self):
        cases = [(RuntimeError("CUDA error: unknown error"), "cuda"),
                 (MemoryError(), "infrastructure"), (ValueError("unexpected"), "infrastructure"),
                 (FloatingPointError(), "training"), (RuntimeError("no validation checkpoint"), "training")]
        for exc, expected in cases:
            self.assertEqual(worker.classify_failure(exc), expected)


class GlobalAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "global_ledger.jsonl"
        self.events = []
        self.append("run_start")

    def append(self, event, **fields):
        payload = {"seq": len(self.events) + 1, "event": event, "study_id": "N14R2", "at": "test",
                   "prev_event_sha256": self.events[-1]["event_sha256"] if self.events else None, **fields}
        payload["event_sha256"] = worker.hash_json(payload)
        self.events.append(payload)
        self.path.write_text("".join(worker.canonical(e) + "\n" for e in self.events), encoding="utf-8")

    def authorize(self, draw):
        self.append("draw_authorized", draw_id=draw, draw_role="primary" if draw < 24 else "reserve", node_id=f"node{draw % 4}",
                    shard_remainder=draw % 4, authorization_id=f"draw{draw}", plan_run_sha256="a" * 64, pins_sha256="b" * 64)

    def terminal(self, draw, status="complete"):
        self.append("draw_terminal", draw_id=draw, node_id=f"node{draw % 4}", status=status,
                    node_draw_receipt_sha256="c" * 64, exhausted_unit_ids=["u"] if status == "void" else [], reason="training" if status == "void" else None)

    def selected(self, **kwargs):
        return run.authorized_draws(self.path, shard=0, plan_run_sha256="a" * 64, pins_sha256="b" * 64, **kwargs)

    def primaries(self):
        for draw in range(24):
            self.authorize(draw)

    def test_shards_whole_draws_primary_only(self):
        self.primaries()
        self.assertEqual(set(self.selected()), {0, 4, 8, 12, 16, 20})

    def test_reserve_requires_every_primary_terminal(self):
        self.primaries()
        self.terminal(0, "void")
        self.authorize(24)
        with self.assertRaisesRegex(worker.IntegrityError, "all primary"):
            self.selected(reserve_draw=24)

    def test_reserve_is_explicit_and_requires_void(self):
        self.primaries()
        for draw in range(24):
            self.terminal(draw, "void" if draw == 0 else "complete")
        self.authorize(24)
        self.assertNotIn(24, self.selected())
        self.assertIn(24, self.selected(reserve_draw=24))

    def test_reserve_not_allowed_for_infrastructure_pending(self):
        self.primaries()
        for draw in range(24):
            self.terminal(draw)
        self.authorize(24)
        with self.assertRaisesRegex(worker.IntegrityError, "not justified"):
            self.selected(reserve_draw=24)

    def test_corrupt_chain_and_duplicate_authorization_rejected(self):
        self.authorize(0)
        self.authorize(0)
        with self.assertRaisesRegex(worker.IntegrityError, "duplicate"):
            self.selected()
        self.events[-1]["pins_sha256"] = "d" * 64
        self.path.write_text("".join(worker.canonical(e) + "\n" for e in self.events), encoding="utf-8")
        with self.assertRaisesRegex(worker.IntegrityError, "hash chain"):
            self.selected()

    def test_recovery_requires_exact_previous_node_ledger_hash(self):
        self.authorize(0)
        with self.assertRaisesRegex(worker.IntegrityError, "recovery authorization"):
            self.selected(recovery_ledger_sha256="e" * 64)
        self.append("node_recovery_authorized", node_id="node0", shard_remainder=0, authorization_id="recovery1",
                    prior_node_ledger_sha256="e" * 64, plan_run_sha256="a" * 64, pins_sha256="b" * 64)
        self.assertIn(0, self.selected(recovery_ledger_sha256="e" * 64))
        with self.assertRaises(worker.IntegrityError):
            self.selected(recovery_ledger_sha256="f" * 64)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the record-44 and record-46 BLOCK items in
run_d1_training.py.

Record-44 series (fail-open in runner 013ec674...):
  A. code drift after prepare -> run refused (verify_frozen_at_run);
  B. prepared freeze -> prepare refuses overwrite; refreeze refused once
     training artifacts exist;
  C. failed / crashed(stale) attempts are charged; charged_seconds sums ALL
     attempt outcomes;
  D. third attempt refused (MAX_ATTEMPTS = 2);
  E. epoch-boundary budget/deadline gate raises BudgetStop mid-fit;
  F. test-once: a non-committed attempt that reached the outer-test phase
     permanently blocks retraining of that unit.

Record-46 series (fail-open in runner 637010dc...):
  G1. stale reconciliation charges max(heartbeat, wall since started_at) --
      "600 s wall / 1 s heartbeat must charge >= 600 s";
  G2. whole-run exclusive lock: second acquirer refused; stale lock refused
      (never auto-broken);
  G3. attempt-file O_EXCL: concurrent creators of the same attempt number --
      exactly one wins (threaded race, Codex's own reproduction);
  G4. loss ledger persists across "restarts"; final_status returns
      all_units_complete ONLY on full success and no losses;
  G5. outer-test phase marking re-gates cap/deadline BEFORE test access
      (source-order check on the wiring);
  G6. run() smoke gate: absent/stale smoke_test.json refused.

All tests run on a temporary TRAIN_ROOT with fabricated ledger entries --
no GPU, no real training tree, no D1 metrics.
"""
import inspect
import json
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, r"E:\科研\claudework\07-ser-repro-protocol-audit\tools")
import run_d1_training as rd  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(("PASS  " if ok else "FAIL  ") + name)
    passed, failed = passed + (1 if ok else 0), failed + (0 if ok else 1)


def expect_raises(fn, exc_type, needle=""):
    try:
        fn()
        return False
    except exc_type as exc:
        return needle in str(exc)
    except Exception:
        return False


def main():
    tmp = Path(tempfile.mkdtemp(prefix="d1_gate_tests_"))
    real_train_root = rd.TRAIN_ROOT
    rd.TRAIN_ROOT = tmp  # every path helper resolves through this module global
    try:
        # ---- A. code-drift gate ----
        prepared_ok = {"tool_sha256": {
            rd.RUNNER_PATH.name: rd.core.sha256_file(rd.RUNNER_PATH),
            "run_p1_protocol_premium.py": rd.core.sha256_file(Path(rd.p1.__file__)),
            "p1_protocol_core.py": rd.core.sha256_file(Path(rd.core.__file__)),
        }}
        try:
            rd.verify_frozen_at_run(prepared_ok)
            check("A1: verify_frozen_at_run accepts matching hashes", True)
        except Exception as exc:
            check(f"A1: verify_frozen_at_run accepts matching hashes ({exc})", False)
        tampered = {"tool_sha256": dict(prepared_ok["tool_sha256"])}
        tampered["tool_sha256"][rd.RUNNER_PATH.name] = "0" * 64
        check("A2: FAIL-CLOSED -- D1 runner hash drift refused",
              expect_raises(lambda: rd.verify_frozen_at_run(tampered), RuntimeError, "drift"))
        tampered2 = {"tool_sha256": dict(prepared_ok["tool_sha256"])}
        tampered2["p1"] = None
        tampered2["tool_sha256"]["p1_protocol_core.py"] = "f" * 64
        check("A3: FAIL-CLOSED -- core hash drift refused",
              expect_raises(lambda: rd.verify_frozen_at_run(tampered2), RuntimeError, "drift"))

        # ---- B. prepared freeze semantics ----
        rd.TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        rd.core.atomic_write_json(rd.prepared_path(), {"frozen": True})
        check("B1: FAIL-CLOSED -- prepare refuses overwriting existing freeze",
              expect_raises(lambda: rd.prepare(refreeze=False), RuntimeError, "refused"))
        (rd.TRAIN_ROOT / "units").mkdir()
        check("B2: FAIL-CLOSED -- refreeze refused once training artifacts exist",
              expect_raises(lambda: rd.prepare(refreeze=True), RuntimeError, "after training"))
        shutil.rmtree(rd.TRAIN_ROOT / "units")

        # ---- C. charging of failures and stale attempts ----
        a1 = rd.open_attempt("unitX")
        rd.update_attempt(a1, status="failed", elapsed_seconds=100.0)
        check("C1: failed attempt time IS charged", abs(rd.charged_seconds() - 100.0) < 1e-9)
        a2 = rd.open_attempt("unitX")  # second (last allowed) attempt
        rd.update_attempt(a2, elapsed_seconds=50.0)  # still "running" -> crash simulation
        check("C2: running(stale) attempt time IS charged at last heartbeat",
              abs(rd.charged_seconds() - 150.0) < 1e-9)
        notes = rd.reconcile_stale_attempts()
        check("C3: startup reconciliation closes stale attempt, keeps charge",
              len(notes) == 1 and abs(rd.charged_seconds() - 150.0) < 1e-9
              and rd.read_attempt(a2)["status"] == "stale_crashed")

        # ---- D. attempt cap ----
        check("D1: FAIL-CLOSED -- third attempt refused",
              expect_raises(lambda: rd.open_attempt("unitX"), RuntimeError, "attempt cap"))

        # ---- E. epoch-boundary gates ----
        check("E1: FAIL-CLOSED -- epoch gate stops when base+elapsed >= 4h cap",
              expect_raises(lambda: rd.epoch_budget_check(rd.GPU_SECONDS_CAP - 10.0, 10.0),
                             rd.BudgetStop, "cap"))
        try:
            rd.epoch_budget_check(0.0, 60.0)
            check("E2: epoch gate passes far below cap and before deadline", True)
        except Exception as exc:
            check(f"E2: epoch gate passes far below cap ({exc})", False)
        after_deadline = rd.HARD_STOP + timedelta(seconds=1)
        check("E3: FAIL-CLOSED -- epoch gate stops past the 8/22 hard stop",
              expect_raises(lambda: rd.epoch_budget_check(0.0, 1.0, now=after_deadline),
                             rd.BudgetStop, "hard stop"))
        check("E4: pair-level budget_gate also stops at cap (ledger holds 150s < cap, "
              "so it passes now)", not expect_raises(rd.budget_gate, rd.BudgetStop))

        # ---- F. test-once permanence ----
        b1 = rd.open_attempt("unitY")
        rd.update_attempt(b1, phase="outer_test", status="stale_crashed", elapsed_seconds=30.0)
        check("F1: FAIL-CLOSED -- outer-test-phase attempt without commit blocks retraining",
              expect_raises(lambda: rd.open_attempt("unitY"), RuntimeError, "test-once"))
        c1 = rd.open_attempt("unitZ")
        rd.update_attempt(c1, phase="commit", status="failed_commit", elapsed_seconds=30.0)
        check("F2: FAIL-CLOSED -- failed-commit attempt (test accessed) blocks retraining",
              expect_raises(lambda: rd.open_attempt("unitZ"), RuntimeError, "test-once"))
        d1 = rd.open_attempt("unitW")
        rd.update_attempt(d1, phase="commit", status="success", elapsed_seconds=30.0)
        check("F3: committed success does NOT block (skip happens at unit_status level)",
              not expect_raises(lambda: rd.open_attempt("unitW"), RuntimeError, "test-once"))

        # ---- G1. conservative stale charging: 600s wall / 1s heartbeat ----
        base_before = rd.charged_seconds()
        g1 = rd.open_attempt("unitG1")
        rec = rd.read_attempt(g1)
        started_600s_ago = (datetime.now(timezone(timedelta(hours=-4)))
                             - timedelta(seconds=600)).isoformat(timespec="seconds")
        rec.update({"started_at": started_600s_ago, "elapsed_seconds": 1.0,
                     "status": "running"})
        rd.core.atomic_write_json(g1, rec)
        rd.reconcile_stale_attempts()
        g1_charged = rd.charged_seconds() - base_before
        check(f"G1: FAIL-CLOSED -- stale 600s-wall/1s-heartbeat charged >= 600s "
              f"(got {g1_charged:.1f}s)", g1_charged >= 600.0)
        check("G1b: reconciled attempt closed as stale_crashed",
              rd.read_attempt(g1)["status"] == "stale_crashed")

        # ---- G2. whole-run exclusive lock ----
        lock = rd.acquire_run_lock()
        check("G2: FAIL-CLOSED -- second acquirer refused while lock is live",
              expect_raises(rd.acquire_run_lock, RuntimeError, "lock"))
        rd.release_run_lock(lock)
        lock2 = rd.acquire_run_lock()
        check("G2b: lock reacquirable after explicit release", lock2.exists())
        rd.release_run_lock(lock2)

        # ---- G3. attempt O_EXCL race: exactly one winner ----
        results = []
        barrier = threading.Barrier(2)

        def racer():
            barrier.wait()
            try:
                results.append(("ok", rd.open_attempt("unitG3")))
            except (RuntimeError, FileExistsError) as exc:
                results.append(("refused", str(exc)))

        t1, t2 = threading.Thread(target=racer), threading.Thread(target=racer)
        t1.start(); t2.start(); t1.join(); t2.join()
        winners = [r for r in results if r[0] == "ok"]
        check(f"G3: FAIL-CLOSED -- concurrent attempt race has exactly one winner "
              f"(winners={len(winners)}, refusals={len(results) - len(winners)})",
              len(winners) == 1 and len(results) == 2)
        check("G3b: exactly one attempt file exists after the race",
              len(rd.attempt_files_for("unitG3")) == 1)

        # ---- G4. loss ledger + terminal-state semantics ----
        rows2 = [
            {"unit_id": "u1", "channel": "speech", "model": "cnn", "protocol": "random",
             "seed": "0", "outer_fold": "0"},
            {"unit_id": "u2", "channel": "song", "model": "cnn", "protocol": "random",
             "seed": "0", "outer_fold": "0"},
        ]
        for row in rows2:
            d = rd.unit_dir(row)
            d.mkdir(parents=True, exist_ok=True)
            rd.core.atomic_write_json(d / "run.json", {"status": "success"})
        check("G4a: full success + empty loss ledger -> all_units_complete",
              rd.final_status(rows2)["status"] == "all_units_complete")
        rd.record_unit_loss("u2", "test-once guard tripped")
        check("G4b: FAIL-CLOSED -- any persisted loss forbids all_units_complete",
              rd.final_status(rows2)["status"] == "incomplete/not_conclusive")
        check("G4c: loss survives 'restart' (fresh read from disk)",
              "u2" in rd.permanently_lost_units())
        check("G4d: loss ledger lists the lost unit in the terminal record",
              "u2" in rd.final_status(rows2)["lost_units"])

        # ---- G5. pre-test budget re-gate wiring (source order) ----
        src = inspect.getsource(rd.run_one_unit)
        gate_pos = src.find("epoch_budget_check(base_charged, time.perf_counter() - started)")
        mark_pos = src.find("update_attempt(attempt_path, phase=phase")
        check("G5: outer-test phase marker re-gates budget BEFORE writing the phase "
              "(and before any test access)", 0 < gate_pos < mark_pos)

        # ---- G6. smoke gate ----
        check("G6a: FAIL-CLOSED -- absent smoke_test.json refuses run",
              expect_raises(rd.verify_smoke_gate, RuntimeError, "smoke"))
        rd.core.atomic_write_json(rd.TRAIN_ROOT / "smoke_test.json",
                                    {"status": "pass", "runner_sha256": "0" * 64})
        check("G6b: FAIL-CLOSED -- smoke from a different runner revision refused",
              expect_raises(rd.verify_smoke_gate, RuntimeError, "different runner"))
        rd.core.atomic_write_json(rd.TRAIN_ROOT / "smoke_test.json",
                                    {"status": "pass",
                                     "runner_sha256": rd.core.sha256_file(rd.RUNNER_PATH)})
        check("G6c: matching pass smoke accepted",
              not expect_raises(rd.verify_smoke_gate, RuntimeError))
    finally:
        rd.TRAIN_ROOT = real_train_root
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{passed} PASS, {failed} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

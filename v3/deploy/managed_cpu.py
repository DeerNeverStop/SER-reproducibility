"""One outcome-blind DEPLOY-2 CPU queue: A, B2 Ridge, then B2-FIX.

The execution lock must already exist. The queue hides CUDA, uses two threads
and BelowNormal priority, fsyncs every attempt start, and seals raw artifacts.
It never calls a scientific scorer. Interrupted starts consume an attempt.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import contextlib
import gzip
import json
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import calib as B, calib_fixed as F, common as C, enroll as A, execution as E

MODULE_DIRS = {"A": "A", "B2": "B2_cpu", "B2-FIX": "B2_FIXED"}
REQUIRED_FILES = {
    "A": ("predictions.csv.gz", "references.json.gz", "model.json", "DONE"),
    "B2": ("val_predictions.csv", "test_predictions.csv", "unit.json", "DONE"),
    "B2-FIX": ("seen_predictions.csv", "new_predictions.csv", "test_predictions.csv", "model.npz", "unit.json", "DONE"),
}


def durable_append(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(C.canonical(value) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_journal(path: Path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    C.require(not raw or raw.endswith(b"\n"), "CPU attempt journal has a truncated final record; audit required")
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise C.IntegrityError("CPU attempt journal is malformed; audit required") from exc
    C.require(all(isinstance(row, dict) for row in rows), "invalid CPU attempt journal records")
    return rows


@contextlib.contextmanager
def exclusive_queue(work_root: Path):
    work_root.mkdir(parents=True, exist_ok=True)
    path = work_root / ".managed_cpu.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise C.IntegrityError(f"CPU runner lock exists: {path}; do not launch a second queue") from exc
    token = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": C.now()}
    try:
        os.write(fd, C.canonical(token).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        yield
    finally:
        # Never remove a lock that another process has replaced.
        if path.is_file() and C.read_json(path) == token:
            path.unlink()


def job_key(job):
    return f"{job['module']}:{job['unit']['unit_id']}"


def load_jobs(plan_dir: Path):
    jobs, plans = [], {}
    for module, filename in (("A", "A.json"), ("B2", "B2.json"), ("B2-FIX", "B2_FIXED.json")):
        path = plan_dir / filename
        plan = C.read_json(path)
        C.require(plan.get("program") == C.PROGRAM and plan.get("module") == module, f"invalid CPU plan {filename}")
        units = [u for u in plan["units"] if module != "B2" or u["engine"] == "ridge"]
        C.require(len({u["unit_id"] for u in units}) == len(units), f"duplicate units in {filename}")
        plans[module] = plan
        for unit in units:
            jobs.append({"module": module, "unit": unit, "plan_file_sha256": C.sha256_file(path)})
    C.require(len({job_key(j) for j in jobs}) == len(jobs), "duplicate CPU queue jobs")
    return jobs, plans


class QueueGuard:
    """Durable attempt accounting, strict unit seals, and operational receipts."""

    def __init__(self, root: Path, lock: dict, jobs: list, deadline=None):
        self.root, self.lock, self.jobs, self.deadline = root, lock, jobs, deadline
        self.control = root / "cpu_control"
        self.control.mkdir(parents=True, exist_ok=True)
        self.journal = self.control / "attempts.jsonl"
        self.by_key = {job_key(j): j for j in jobs}
        self.attempts = Counter()
        self.done_events = set()
        for row in read_journal(self.journal):
            key = row.get("job_key")
            C.require(key in self.by_key and row.get("lock_sha256") == lock["lock_sha256"], "CPU journal identity drift")
            C.require(row.get("event") in ("start", "done", "failed"), "unknown CPU journal event")
            if row["event"] == "start":
                self.attempts[key] += 1
                C.require(row.get("attempt") == self.attempts[key] <= 2, "invalid CPU attempt sequence")
            else:
                C.require(self.attempts[key] >= 1 and row.get("attempt") == self.attempts[key], "CPU event has no matching attempt")
                if row["event"] == "done":
                    self.done_events.add(key)

    def unit_dir(self, job):
        return self.root / MODULE_DIRS[job["module"]] / "units" / job["unit"]["unit_id"]

    def event(self, job, event, **extra):
        key = job_key(job)
        durable_append(self.journal, {"job_key": key, "unit_id": job["unit"]["unit_id"], "module": job["module"],
                       "event": event, "attempt": self.attempts[key], "lock_sha256": self.lock["lock_sha256"],
                       "at": C.now(), **extra})

    def before(self, job):
        key = job_key(job)
        if self.deadline is not None:
            C.require(datetime.now(timezone.utc) < self.deadline, "CPU deadline reached")
        C.require(self.attempts[key] < 2, f"two CPU attempts exhausted: {key}")
        self.attempts[key] += 1
        self.event(job, "start")

    def seal(self, job, validate):
        udir = self.unit_dir(job)
        validate(job, udir)
        files = REQUIRED_FILES[job["module"]]
        C.require(all((udir / n).is_file() for n in files), "required CPU unit artifact missing")
        seal = {"schema": "ser26-deploy-cpu-unit-seal-2", "program": C.PROGRAM,
                "module": job["module"], "unit_id": job["unit"]["unit_id"], "unit": job["unit"],
                "lock_sha256": self.lock["lock_sha256"], "plan_file_sha256": job["plan_file_sha256"],
                "files": {name: C.sha256_file(udir / name) for name in files}}
        C.atomic_write_json(udir / "CPU_SEALED.json", seal)
        if job_key(job) not in self.done_events:
            self.event(job, "done")
            self.done_events.add(job_key(job))

    def sealed(self, job, validate):
        udir = self.unit_dir(job)
        path = udir / "CPU_SEALED.json"
        if not path.exists():
            if (udir / "DONE").exists():
                # Crash after raw DONE but before managed seal: recover only if
                # the durable journal proves a managed attempt existed.
                C.require(self.attempts[job_key(job)] > 0, "unmanaged CPU DONE requires identity audit")
                self.seal(job, validate)
                return True
            return False
        try:
            seal = C.read_json(path)
        except (ValueError, OSError) as exc:
            raise C.IntegrityError("CPU seal is truncated or malformed") from exc
        expected = {"schema": "ser26-deploy-cpu-unit-seal-2", "program": C.PROGRAM,
                    "module": job["module"], "unit_id": job["unit"]["unit_id"], "unit": job["unit"],
                    "lock_sha256": self.lock["lock_sha256"], "plan_file_sha256": job["plan_file_sha256"]}
        C.require(all(seal.get(k) == v for k, v in expected.items()), "CPU sealed identity mismatch")
        C.require(self.attempts[job_key(job)] > 0, "CPU seal has no durable managed attempt")
        C.require(set(seal.get("files", {})) == set(REQUIRED_FILES[job["module"]]), "CPU seal has incomplete artifact list")
        for name, sha in seal["files"].items():
            C.require((udir / name).is_file() and C.sha256_file(udir / name) == sha, f"CPU sealed output drift: {name}")
        validate(job, udir)
        if job_key(job) not in self.done_events:
            self.event(job, "done")
            self.done_events.add(job_key(job))
        return True

    def status(self, complete, failed, active=None):
        result = {"program": C.PROGRAM, "updated_at": C.now(), "lock_sha256": self.lock["lock_sha256"],
                  "planned_units": len(self.jobs), "complete_units": complete, "failed_units": failed,
                  "attempts_started": sum(self.attempts.values()), "active_job": active,
                  "scientific_scores_computed": False}
        C.atomic_write_json(self.control / "status.json", result)
        return result

    def finish(self, validate):
        C.require(all(self.sealed(job, validate) for job in self.jobs), "CPU queue is incomplete")
        result = {"schema": "ser26-deploy-cpu-completion-2", "program": C.PROGRAM,
                  "lock_sha256": self.lock["lock_sha256"], "completed_at": C.now(),
                  "planned_units": len(self.jobs), "counts": dict(Counter(j["module"] for j in self.jobs)),
                  "unit_seal_sha256": {job_key(j): C.sha256_file(self.unit_dir(j) / "CPU_SEALED.json") for j in self.jobs},
                  "journal_sha256": C.sha256_file(self.journal), "scientific_scores_computed": False}
        C.atomic_write_json(self.control / "COMPLETE.json", result)
        return result


class CpuWorker:
    def __init__(self, plans, root, guard_info):
        self.plans, self.root, self.guard_info = plans, root, guard_info
        self.manifests, self.splits = {}, {}
        self.cache, self.cache_key = None, None
        A.verify_plan_digest(plans["A"])
        F.validate_plan(plans["B2-FIX"])

    def manifest(self, base):
        if base not in self.manifests:
            self.manifests[base] = C.load_manifest(base)
        return self.manifests[base]

    def get_cache(self, base, kind, sha):
        key = (base, kind)
        if self.cache_key != key:
            self.cache = None  # release the previous full-layer cache first
            self.cache = C.FeatureCache(base, kind)
            self.cache_key = key
        C.require(self.cache.sha256 == sha, "CPU feature cache hash mismatch")
        return self.cache

    def split(self, unit):
        key = unit["split_key"]
        if key not in self.splits:
            self.splits[key] = C.load_split(key, verify=True)
        return next(f for f in self.splits[key]["folds"] if int(f["fold"]) == int(unit["fold"]))

    def run(self, job, udir):
        module, unit = job["module"], job["unit"]
        plan = self.plans[module]
        started = C.now()
        if module == "A":
            lp = plan["levels"][unit["level"]]
            manifest = self.manifest(lp["base"])
            cache = self.get_cache(lp["base"], unit["enc"], unit["feature_sha256"])
            state = plan["config"]["state"]
            rows, info = A.compute_unit(plan, unit, manifest, lambda paths: cache.get(paths, state).astype(np.float64))
            A.write_unit(udir, plan, unit, rows, info, self.guard_info)
        elif module == "B2":
            manifest, fold = self.manifest(unit["base"]), self.split(unit)
            cache = self.get_cache(unit["feature_corpus"], unit["feature_kind"], unit["feature_sha256"])
            C.require(tuple(len(fold[r]) for r in ("fit", "val", "test")) == tuple(unit[f"n_{r}"] for r in ("fit", "val", "test")),
                      "CPU B2 fold count mismatch")
            with C.Timer() as timer:
                X = {r: cache.get(fold[r], state=unit["config"]["state"]) for r in ("fit", "val", "test")}
                scores = B.fit_ridge_unit(X["fit"], C.labels_for(manifest, fold["fit"]), X["val"], X["test"],
                                          unit["n_classes"], unit["config"])
                preds = {r: {"paths": fold[r], "speakers": C.speakers_for(manifest, fold[r]),
                             "y_true": C.labels_for(manifest, fold[r]), "y_pred": s.argmax(axis=1), "scores": s}
                         for r, s in zip(("val", "test"), scores)}
            B.write_unit_outputs(udir, unit, {"environment": B.environment_info(2), "best_epoch": None, "epochs_run": None,
                "timing": {"started_at": started, "finished_at": C.now(), "seconds": timer.seconds}}, preds["val"], preds["test"])
        else:
            manifest = self.manifest(unit["base"])
            cache = self.get_cache(unit["feature_corpus"], unit["feature_kind"], unit["feature_sha256"])
            F.run_one_unit(plan, unit, self.root / MODULE_DIRS[module], manifest, cache, B.environment_info(2), started)

    def validate(self, job, udir):
        module, unit, plan = job["module"], job["unit"], self.plans[job["module"]]
        if module == "A":
            C.require(A.done_verified(udir), "A raw receipt invalid")
            meta = C.read_json(udir / "model.json")
            C.require(meta.get("unit") == unit and meta.get("plan_sha256") == plan["plan_sha256"], "A raw identity mismatch")
            with gzip.open(udir / "references.json.gz", "rt", encoding="utf-8") as fh:
                receipt = json.load(fh)
            C.require(receipt.get("records") == A.reference_records(plan, unit), "A raw references mismatch")
            # Labels and all prediction groups are checked without computing UAR.
            groups = A.read_unit_predictions(udir)
            lp = plan["levels"][unit["level"]]
            expected = {(c["id"], est, d, s) for s in unit["test_speakers"] for c in lp["conditions"] if c["feasible"]
                        for est in c["estimators"] for d in range(c["draws"])}
            C.require(set(groups) == expected, "A raw prediction groups differ from plan")
            for (_, _, _, speaker), g in groups.items():
                entry = lp["per_speaker"][speaker]
                C.require(g["paths"] == entry["Q"] and np.array_equal(g["y_true"], entry["Q_labels"]), "A raw Q/label drift")
                C.require(((0 <= g["y_pred"]) & (g["y_pred"] < len(lp["classes"]))).all(), "A prediction outside class set")
        elif module == "B2":
            C.require(B.done_status(udir, unit["n_classes"], unit["n_val"], unit["n_test"]) == "ok", "B2 raw receipt invalid")
            meta = C.read_json(udir / "unit.json")
            C.require(all(meta.get(k) == v for k, v in unit.items()), "B2 raw unit identity mismatch")
            manifest, fold = self.manifest(unit["base"]), self.split(unit)
            for role in ("val", "test"):
                p = C.read_predictions_csv(udir / f"{role}_predictions.csv", "logit")
                C.require(p["paths"] == fold[role] and p["speakers"] == C.speakers_for(manifest, fold[role])
                          and np.array_equal(p["y_true"], C.labels_for(manifest, fold[role])), "B2 path/speaker/label drift")
                C.require(np.isfinite(p["scores"]).all() and np.array_equal(p["scores"].argmax(axis=1), p["y_pred"]), "B2 invalid predictions")
        else:
            C.require(F.done_status(udir, unit, plan["splits"][unit["split_key"]], self.manifest(unit["base"])) == "ok",
                      "B2-FIX raw receipt invalid")


def run_queue(guard: QueueGuard, worker, before_unit=None):
    complete, failed = 0, 0
    for job in guard.jobs:
        if before_unit is not None:
            before_unit()
        if guard.sealed(job, worker.validate):
            complete += 1
            guard.status(complete, failed)
            continue
        key = job_key(job)
        while guard.attempts[key] < 2:
            guard.before(job)
            guard.status(complete, failed, key)
            start = time.perf_counter()
            try:
                worker.run(job, guard.unit_dir(job))
                guard.seal(job, worker.validate)
            except Exception as exc:
                guard.event(job, "failed", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - start)
                if isinstance(exc, C.IntegrityError) or (guard.unit_dir(job) / "DONE").exists():
                    raise
                print(f"CPU technical attempt {guard.attempts[key]}/2 failed: {key} ({type(exc).__name__})", flush=True)
                continue
            complete += 1
            print(f"CPU completed {complete}/{len(guard.jobs)}: {key} wall={time.perf_counter() - start:.1f}s", flush=True)
            break
        else:
            failed += 1
        guard.status(complete, failed)
    if failed:
        return {"complete": False, **guard.status(complete, failed)}
    result = guard.finish(worker.validate)
    return {"complete": True, **result}


def verify_runtime_sources(lock):
    # Full cache hashes are checked once by execution.verify_lock and again when
    # a cache is loaded. Source/plan bytes and environment are checked per unit.
    for rel, sha in lock["files"].items():
        path = C.REPO_ROOT / rel
        C.require(path.is_file() and C.sha256_file(path) == sha, f"CPU runtime source/plan drift: {rel}")
    C.require(E.environment("cpu") == lock["cpu_environment"], "CPU runtime environment drift")


def main(argv=None):
    guard_info = C.cpu_guard(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("run", "status"))
    parser.add_argument("--work-root", type=Path, default=C.DEPLOY_ROOT / "work")
    parser.add_argument("--plan-dir", type=Path, default=C.DEPLOY_ROOT / "work/plan")
    parser.add_argument("--lock", type=Path, default=E.LOCK_PATH)
    parser.add_argument("--deadline", help="timezone-aware ISO timestamp; no new unit starts after it")
    args = parser.parse_args(argv)
    if args.cmd == "status":
        path = args.work_root / "cpu_control/status.json"
        state = C.read_json(path) if path.is_file() else {"started": False, "scientific_scores_computed": False}
        state["runner_lock_present"] = (args.work_root / ".managed_cpu.lock").is_file()
        interrupted = args.work_root / "cpu_control/INTERRUPTED.json"
        if interrupted.is_file():
            state["last_interruption"] = C.read_json(interrupted)
        print(C.canonical(state))
        return 0
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00")) if args.deadline else None
    C.require(deadline is None or deadline.tzinfo is not None, "CPU deadline must specify timezone")
    with exclusive_queue(args.work_root):
        lock = E.verify_lock(args.lock, profile="cpu")
        jobs, plans = load_jobs(args.plan_dir)
        for job in jobs:
            # Plan location must refer to the exact locked plan file, never an
            # independently regenerated sibling that happens to use the same ID.
            filename = {"A": "A.json", "B2": "B2.json", "B2-FIX": "B2_FIXED.json"}[job["module"]]
            rel = (args.plan_dir / filename).resolve().relative_to(C.REPO_ROOT.resolve()).as_posix()
            C.require(lock["files"].get(rel) == job["plan_file_sha256"], "CPU plan file not in execution lock")
        guard = QueueGuard(args.work_root, lock, jobs, deadline)
        worker = CpuWorker(plans, args.work_root, guard_info)
        try:
            result = run_queue(guard, worker, lambda: verify_runtime_sources(lock))
        except Exception as exc:
            C.atomic_write_json(guard.control / "INTERRUPTED.json", {"program": C.PROGRAM, "at": C.now(),
                                "lock_sha256": lock["lock_sha256"], "error": f"{type(exc).__name__}: {exc}",
                                "scientific_scores_computed": False})
            raise
    print(C.canonical({k: v for k, v in result.items() if k != "unit_seal_sha256"}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Process-isolated adapter around the unchanged P1 engine and artifact writer.

No training runs on import. Each spawned process owns its RNGs, CUDA context,
PlanIO and feature cache. The coordinator is the sole attempt-ledger writer.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import queue
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_NAMES = ("unit.json", "history.json", "predictions.csv", "DONE")
RECEIPT_SCHEMA = "ser26-n14r2-artifact-receipt-1"
INFRA_FAILURES = frozenset({"cuda", "infrastructure", "worker_exit", "timeout", "interrupted", "circuit_cancelled", "integrity"})


class IntegrityError(RuntimeError):
    """An identity, authorization or immutable artifact failed verification."""


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_json(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sync_dir(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def publish_json(path, value):
    """Durably publish a new file atomically; never replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic no-replace publication on both NTFS and local Linux filesystems.
        os.link(temporary, path)
        sync_dir(path.parent)
    finally:
        os.unlink(temporary)


@contextmanager
def file_lock(path):
    """OS-owned lock: a stale filename is not itself a held lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise IntegrityError(f"lock held: {path.name}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class N14R2PlanIO:
    """Strict wrapper: portable N14R2 keys, no missing-manifest fallback."""

    def __init__(self, plan_dir, manifest_dir, run_dir):
        from v2.ser_v2.train import PlanIO
        plan_dir, manifest_dir = Path(plan_dir), Path(manifest_dir)
        if not (manifest_dir / "cremad_manifest.csv").is_file():
            raise IntegrityError("CREMA-D manifest is required; synthetic fallback forbidden")
        self._io = PlanIO(plan_dir, manifest_dir, Path(run_dir))
        self.manifest, self.n_classes = self._io.manifest, self._io.n_classes
        self.configs, self.plan = self._io.configs, self._io.plan

    def labels(self, *args):
        return self._io.labels(*args)

    def speakers(self, *args):
        return self._io.speakers(*args)

    def fold(self, row):
        if row["arm"] != "N14R2" or row["study_id"] != "N14R2":
            raise IntegrityError("old-study unit forbidden")
        key = self._io.sha_to_key.get(row["split_sha256"])
        expected = f"n14r2__d{int(row['draw_id']):02d}__{row['cell']}"
        if key != expected:
            raise IntegrityError("N14R2 split key mismatch")
        path = self._io.plan_dir / "splits" / f"{key}.json"
        if hash_file(path) != row["split_sha256"]:
            raise IntegrityError("split bytes changed")
        fold = self._io.fold(row)
        for part in ("fit", "val", "test"):
            if len(fold[part]) != int(row[f"n_{part}"]):
                raise IntegrityError("split population count mismatch")
        return fold


class _UnitDirectory:
    def __init__(self, path, unit_id):
        self.path, self.unit_id = path, unit_id

    def __truediv__(self, unit_id):
        if unit_id != self.unit_id:
            raise IntegrityError("unit directory adapter mismatch")
        return self.path


class _RunDirectory:
    def __init__(self, path, unit_id):
        self.path, self.unit_id = path, unit_id

    def __truediv__(self, component):
        if component != "units":
            raise IntegrityError("unit directory adapter mismatch")
        return _UnitDirectory(self.path, self.unit_id)


def validate_artifacts(attempt_dir, row, configs, io):
    """Reuse the frozen within-unit integrity gate; never aggregate outcomes."""
    from v2_1.n14r.run import preflight_completed_unit
    try:
        state = preflight_completed_unit(
            row, _RunDirectory(Path(attempt_dir), row["unit_id"]), configs,
            io.fold(row), io.manifest[row["base_corpus"]])
    except Exception as exc:
        # Do not propagate any data-bearing message into outcome-blind logs.
        raise IntegrityError("within-unit structural/checkpoint integrity failed") from exc
    if state != "done":
        raise IntegrityError("complete artifact set is missing DONE")


def classify_failure(exc):
    message = str(exc).lower()
    if any(key in message for key in ("cuda", "cudnn", "cublas", "device-side", "nvidia")):
        return "cuda"
    if isinstance(exc, IntegrityError):
        return "integrity"
    if isinstance(exc, (OSError, MemoryError, SystemError)):
        return "infrastructure"
    # Only an explicit algorithmic failure can eventually justify a void draw.
    if isinstance(exc, FloatingPointError) or message == "no validation checkpoint":
        return "training"
    return "infrastructure"


def receipt_for(task, *, status, failure_class=None, artifact_hashes=None):
    return {
        "schema": RECEIPT_SCHEMA, "study_id": "N14R2",
        "node_id": task["node_id"], "shard_remainder": task["shard_remainder"],
        "unit_id": task["row"]["unit_id"], "draw_id": int(task["row"]["draw_id"]),
        "attempt": task["attempt"], "status": status, "failure_class": failure_class,
        "plan_row_sha256": hash_json(task["row"]),
        "artifact_hashes": artifact_hashes or {},
        "started_at": task["started_at"], "finished_at": now(),
    }


def healthcheck():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA requires exactly one visible device")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("CUDA device must be the frozen RTX 4090")
    with torch.no_grad():
        value = torch.ones((16, 16), device="cuda")
        result = value @ value
        torch.cuda.synchronize()
        checksum = float(result.sum().cpu())
        if not bool(torch.isfinite(result).all().item()) or checksum != 4096.0:
            raise RuntimeError("CUDA healthcheck failed")
    del value, result
    torch.cuda.empty_cache()
    return {"pid": os.getpid(), "cuda_healthy": True,
            "device_name": torch.cuda.get_device_name(0), "threads": torch.get_num_threads(),
            "health_operation": "ones(16,16) @ ones(16,16), synchronize, sum", "health_device": "cuda:0",
            "health_checksum": checksum,
            "backend_precision": {
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "autocast_enabled": bool(torch.is_autocast_enabled()),
            }}


def fit_task(task, io, feats):
    """One fit, unchanged engine + writer, staged then atomically receipted."""
    import torch
    from v2.ser_v2.train import engine_p1_frozen, write_unit
    attempt_dir = Path(task["attempt_dir"])
    receipt_path = attempt_dir / "artifact_receipt.json"
    if receipt_path.exists() or any((attempt_dir / n).exists() for n in ARTIFACT_NAMES):
        raise IntegrityError("attempt artifacts already exist")
    try:
        row, cfg = task["row"], io.configs[task["row"]["config_sha256"]]
        if cfg.get("engine") != "p1_frozen":
            raise IntegrityError("only the frozen P1 engine is allowed")
        fold = io.fold(row)
        logits, history, info, model = engine_p1_frozen(row, cfg, fold, io, feats, torch.device("cuda"))
        staging = attempt_dir / ".staging"
        write_unit(row, cfg, fold, io, logits, history, info, staging,
                   task["started_at"], model, persist_checkpoint=False)
        source = staging / "units" / row["unit_id"]
        validate_artifacts(source, row, io.configs, io)
        for name in ARTIFACT_NAMES:
            with (source / name).open("rb") as handle:
                os.fsync(handle.fileno())
            os.rename(source / name, attempt_dir / name)
        sync_dir(attempt_dir)
        receipt = receipt_for(task, status="done", artifact_hashes={n: hash_file(attempt_dir / n) for n in ARTIFACT_NAMES})
        publish_json(receipt_path, receipt)
        del logits, history, info, model
        gc.collect()
        torch.cuda.empty_cache()
        return {"status": "done", "receipt_sha256": hash_file(receipt_path), "failure_class": None}
    except Exception as exc:
        # If successful evidence has already been sealed, cleanup failures must
        # not turn a completed fit into a second contradictory receipt.
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            return {"status": receipt["status"], "receipt_sha256": hash_file(receipt_path),
                    "failure_class": receipt["failure_class"], "postfit_failure": classify_failure(exc)}
        receipt = receipt_for(task, status="failed", failure_class=classify_failure(exc))
        publish_json(receipt_path, receipt)
        return {"status": "failed", "receipt_sha256": hash_file(receipt_path), "failure_class": receipt["failure_class"]}


def worker_main(slot, incoming, outgoing, settings, stop_event):
    """Persistent worker; receives at most one task at a time (no prefetch)."""
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    try:
        with file_lock(Path(settings["node_dir"]) / ".worker-locks" / f"slot{slot}.lock"):
            import torch
            from v2.ser_v2.features import FeatureStore
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            io = N14R2PlanIO(settings["plan_dir"], settings["manifest_dir"], settings["node_dir"])
            feats = FeatureStore(Path(settings["feature_dir"]))
            # Force cache loading before any attempts are allocated.
            feats.load("cremad", "logmel")
            health = healthcheck()
            if health["backend_precision"] != settings["backend_precision"]:
                raise IntegrityError("worker backend precision differs from frozen environment")
            outgoing.put({"event": "ready", "slot": slot, **health})
            while not stop_event.is_set():
                try:
                    task = incoming.get(timeout=1)
                except queue.Empty:
                    if os.getppid() != settings["coordinator_pid"]:
                        return
                    continue
                if task is None or stop_event.is_set():
                    return
                result = fit_task(task, io, feats)
                if result["status"] != "done" or result.get("postfit_failure"):
                    stop_event.set()
                outgoing.put({"event": "result", "slot": slot, "unit_id": task["row"]["unit_id"],
                              "attempt": task["attempt"], **result})
                if stop_event.is_set():
                    return
    except BaseException as exc:
        stop_event.set()
        outgoing.put({"event": "worker_error", "slot": slot, "failure_class": classify_failure(exc),
                      "exception_type": type(exc).__name__})

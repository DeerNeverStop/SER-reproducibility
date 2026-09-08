"""DEPLOY-2 fail-closed execution identity, per-host environment and GPU receipts.

Only operational metadata are inspected. No scientific scores are computed.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

from . import common as c

LOCK_PATH = c.DEPLOY_ROOT / "EXECUTION_LOCK.json"
PLAN_RELS = ["v3/deploy/work/plan/A.json", "v3/deploy/work/plan/B2.json", "v3/deploy/work/plan/B2_FIXED.json"]


def environment(profile):
    names = ["numpy", "scipy", "scikit-learn", "torch", "torchaudio", "librosa", "soundfile", "threadpoolctl"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    out = {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
           "hostname": socket.gethostname(), "packages": versions, "profile": profile}
    if profile == "gpu":
        import torch
        c.require(torch.cuda.is_available(), "GPU execution requires CUDA")
        out.update(gpu=torch.cuda.get_device_name(0), vram=torch.cuda.get_device_properties(0).total_memory,
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return out


def source_files():
    required = ["v3/deploy/SPEC_DEPLOY2_ZH.md", "v3/deploy/PRIOR_EXPOSURE.md",
                "v3/deploy/calib_fixed.py", "v3/deploy/enroll.py", "v3/deploy/calib.py", "v3/deploy/verify.py",
                "v3/deploy/calib_gpu.py", "v3/deploy/managed_gpu.py", "v3/deploy/execution.py",
                "v2/plan_rc2/split_index.json", "v2/plan_rc2/unit_configs.json", "v2/plan_rc2/run_plan.csv"] + PLAN_RELS
    for rel in required:
        c.require((c.REPO_ROOT / rel).is_file(), f"required lock input missing: {rel}")
    paths = {c.REPO_ROOT / rel for rel in required}
    for pat in ["*.py", "v3/deploy/*.py", "v3/deploy/pod/*.py", "v3/deploy/pod/*.sh", "v3/deploy/pod/*.json",
                "v2/ser_v2/*.py", "v2/manifests/*.csv"]:
        paths.update(p for p in c.REPO_ROOT.glob(pat) if p.is_file())
    return {p.relative_to(c.REPO_ROOT).as_posix(): c.sha256_file(p) for p in sorted(paths)}


def build_lock():
    plans = [c.read_json(c.REPO_ROOT / rel) for rel in PLAN_RELS]
    for rel, plan in zip(PLAN_RELS, plans):
        c.require(plan.get("program") == c.PROGRAM, f"program mismatch: {rel}")
        ids = [u["unit_id"] for u in plan["units"]]
        c.require(len(ids) == len(set(ids)), f"duplicate plan units: {rel}")
    c.require([len(p["units"]) for p in plans] == [135, 390, 135], "unexpected formal design counts")
    b2 = plans[1]
    gpu_files = {u["feature_file"] for u in b2["units"] if u["model"] == "cnn"}
    cpu_files = {u["feature_file"] for u in b2["units"] if u["model"].startswith("ridge")}
    for level in plans[0]["levels"].values():
        cpu_files.update(level["feature_files"].values())
    caches = {}
    for name in sorted(cpu_files | gpu_files):
        p = c.V2_DATA_ROOT / "features" / name
        c.require(p.is_file(), f"cache missing: {name}")
        caches[name] = {"sha256": c.sha256_file(p), "profiles": [k for k, names in [("cpu", cpu_files), ("gpu", gpu_files)] if name in names]}
    splits = {}
    idx = c.split_index()
    for level in c.LEVELS:
        for cell in ("GG", "GR"):
            for r in range(3):
                key = f"ctrl__{level}__{cell}__r{r}"
                rel = "plan_rc2/" + idx[key]["path"]
                c.require(c.sha256_file(c.V2_DATA_ROOT / rel) == idx[key]["sha256"], f"bad split {key}")
                splits[rel] = idx[key]["sha256"]
    lock = {"schema": "ser26-deploy-execution-lock-2", "program": c.PROGRAM, "created_at": c.now(),
            "prior_outcomes_seen": True, "current_formal_outcomes_seen_before_lock": False,
            "files": source_files(), "caches": caches, "splits": splits,
            "cpu_environment": environment("cpu"), "counts": {"A_units": 135, "B2_units": 390, "B2_FIXED_units": 135}}
    lock["lock_sha256"] = c.digest(lock)
    return lock


def verify_lock(lock_path=LOCK_PATH, profile="cpu", check_environment=True):
    lock = c.read_json(lock_path)
    c.require(lock.get("schema") == "ser26-deploy-execution-lock-2" and lock.get("program") == c.PROGRAM, "wrong execution identity")
    c.require(lock["lock_sha256"] == c.digest({k: v for k, v in lock.items() if k != "lock_sha256"}), "lock digest mismatch")
    c.require(lock["files"] and all(isinstance(v, str) and len(v) == 64 for v in lock["files"].values()), "incomplete source lock")
    c.require(lock["files"] == source_files(), "mandatory source/plan lock entries absent or changed")
    for rel, expected in lock["files"].items():
        p = c.REPO_ROOT / rel
        c.require(p.is_file() and c.sha256_file(p) == expected, f"source/plan drift: {rel}")
    for name, entry in lock["caches"].items():
        if profile in entry["profiles"]:
            p = c.V2_DATA_ROOT / "features" / name
            c.require(p.is_file() and c.sha256_file(p) == entry["sha256"], f"cache drift: {name}")
    for rel, expected in lock["splits"].items():
        p = c.V2_DATA_ROOT / rel
        c.require(p.is_file() and c.sha256_file(p) == expected, f"split drift: {rel}")
    if profile == "cpu" and check_environment:
        c.require(environment("cpu") == lock["cpu_environment"], "CPU environment changed")
    return lock


def append(path, item):
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


class RunGuard:
    def __init__(self, out, lock, deadline):
        self.out, self.lock, self.deadline = Path(out), lock, deadline
        self.journal = self.out / "managed_attempts.jsonl"
        self.attempts = {}
        if self.journal.exists():
            c.require(self.journal.read_bytes().endswith(b"\n"), "truncated managed journal; audit required")
            for line in self.journal.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["event"] == "start":
                    self.attempts[row["unit_id"]] = self.attempts.get(row["unit_id"], 0) + 1

    def before(self, unit):
        uid = unit["unit_id"]
        c.require(datetime.now(timezone.utc) < self.deadline, "execution deadline reached")
        c.require(self.attempts.get(uid, 0) < 2, f"two attempts exhausted: {uid}")
        self.attempts[uid] = self.attempts.get(uid, 0) + 1
        append(self.journal, {"event": "start", "unit_id": uid, "attempt": self.attempts[uid], "at": c.now()})

    def sealed(self, unit, udir):
        seal_path = udir / "SEALED.json"
        if not seal_path.exists():
            c.require(not (udir / "DONE").exists(), f"DONE without managed seal needs audit: {unit['unit_id']}")
            return False
        seal = c.read_json(seal_path)
        c.require(seal["unit_id"] == unit["unit_id"] and seal["lock_sha256"] == self.lock["lock_sha256"], "sealed identity mismatch")
        c.require(set(seal["files"]) == {"val_predictions.csv", "test_predictions.csv", "unit.json", "DONE", "history.json", "checkpoint.pt"}, "incomplete sealed artifact inventory")
        for name, expected in seal["files"].items():
            p = udir / name
            c.require(p.is_file() and c.sha256_file(p) == expected, f"sealed output corrupted: {name}")
        meta = c.read_json(udir / "unit.json")
        c.require(all(meta.get(k) == v for k, v in unit.items()), "unit metadata differs from plan")
        return True

    def finish(self, unit, udir):
        names = ["val_predictions.csv", "test_predictions.csv", "unit.json", "DONE", "history.json", "checkpoint.pt"]
        for name in names:
            c.require((udir / name).is_file(), f"required unit artifact missing: {name}")
        meta = c.read_json(udir / "unit.json")
        c.require(all(meta.get(k) == v for k, v in unit.items()), "wrong completed unit identity")
        c.require(meta["engine_info"]["checkpoint_sha256"] == c.sha256_file(udir / "checkpoint.pt"), "bad checkpoint")
        c.atomic_write_json(udir / "SEALED.json", {"program": c.PROGRAM, "unit_id": unit["unit_id"],
                            "lock_sha256": self.lock["lock_sha256"], "files": {n: c.sha256_file(udir / n) for n in names}})
        append(self.journal, {"event": "done", "unit_id": unit["unit_id"], "at": c.now()})


@contextlib.contextmanager
def run_context(out, lock_path, deadline):
    lock = verify_lock(lock_path, "gpu")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lockfile = out / ".managed.lock"
    fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "at": c.now()}).encode())
    os.close(fd)
    try:
        env = environment("gpu")
        ep = out / "execution_environment.json"
        identity = {"lock_sha256": lock["lock_sha256"], "environment": env}
        if ep.exists():
            c.require(c.read_json(ep) == identity, "GPU environment/lock changed; new run identity required")
        else:
            c.atomic_write_json(ep, identity)
        yield RunGuard(out, lock, deadline)
    finally:
        lockfile.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["write", "verify"])
    p.add_argument("--lock", type=Path, default=LOCK_PATH)
    p.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")
    a = p.parse_args()
    if a.cmd == "write":
        c.require(not a.lock.exists(), "execution lock already exists; never overwrite a formal lock")
        lock = build_lock()
        c.atomic_write_json(a.lock, lock)
    else:
        lock = verify_lock(a.lock, a.profile)
    print(json.dumps({"pass": True, "program": c.PROGRAM, "lock_sha256": lock["lock_sha256"], "counts": lock["counts"]}))


if __name__ == "__main__":
    main()

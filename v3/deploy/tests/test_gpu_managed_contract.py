"""Synthetic managed-GPU execution boundary tests; no CUDA, models or real data."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import numpy as np
import pytest

from v3.deploy import calib, calib_gpu as gpu, common as c, execution as ex, managed_gpu as managed


@pytest.fixture(autouse=True)
def cpu_only():
    c.cpu_guard(2)


def deadline():
    return datetime.now(timezone.utc) + timedelta(hours=1)


def unit_fixture(tmp_path, monkeypatch):
    manifest = {}
    roles = {}
    for role, n in (("fit", 6), ("val", 3), ("test", 3)):
        roles[role] = [f"{role}/{i}.wav" for i in range(n)]
        for i, p in enumerate(roles[role]):
            manifest[p] = {"speaker": role, "label_index": i % 3, "label": str(i % 3)}
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("synthetic manifest, no audio\n", encoding="utf-8")
    cache_path = tmp_path / "fake.npz"
    cache_path.write_bytes(b"synthetic cache; fixture bypasses parsing")
    cfg = {"model": "cnn", "engine": "p1_frozen", "epochs": 1}
    row = {"program": c.PROGRAM, "module": "B2", "spec": "DEPLOY-2", "level": "fake", "base": "fake",
           "feature_corpus": "fake", "model": "cnn", "engine": "p1_frozen", "cell": "GG", "r": 0,
           "fold": 0, "seed_index": 0, "train_seed": 17, "split_key": "fake_split", "split_sha256": "f"*64,
           "config": cfg, "config_sha256": c.digest(cfg), "feature_kind": "logmel", "feature_file": cache_path.name,
           "feature_sha256": c.sha256_file(cache_path), "manifest_sha256": c.sha256_file(manifest_path),
           "n_classes": 3, **{f"n_{r}": len(paths) for r, paths in roles.items()}}
    row["unit_id"] = calib.make_unit_id(row)
    plan = {"program": c.PROGRAM, "module": "B2", "units": [row]}
    planpath = tmp_path / "B2.json"
    c.atomic_write_json(planpath, plan)
    monkeypatch.setattr(gpu, "load_manifest", lambda base: manifest)
    monkeypatch.setattr(c, "manifest_path", lambda base: manifest_path)
    monkeypatch.setattr(gpu, "load_split", lambda *a, **k: {"folds": [dict(roles, fold=0)], "population": sum(roles.values(), [])})
    monkeypatch.setattr(c, "split_index", lambda: {"fake_split": {"sha256": row["split_sha256"]}})
    class FakeCache:
        def __init__(self, *args):
            self.path, self.sha256 = cache_path, c.sha256_file(cache_path)
        def get(self, paths):
            return np.zeros((len(paths), 4, 5), np.float32)
    monkeypatch.setattr(c, "FeatureCache", FakeCache)
    monkeypatch.setattr(gpu, "make_device", lambda *args: (SimpleNamespace(type="cuda"), None))
    monkeypatch.setattr(gpu, "environment_info", lambda *args: {"hostname": "synthetic", "gpu_name": "fake-gpu", "torch": "fake"})
    from v3.deploy import engines_deploy
    import torch
    calls = []
    def engine(u, cfg, fold, data, device):
        calls.append(u["unit_id"])
        scores = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
        return scores.copy(), scores.copy(), [{"epoch": 1}], {"best_epoch": 1, "epochs_run": 1,
                    "_checkpoint_state": {"synthetic.weight": torch.ones(3)}, "train_seconds": .01}
    monkeypatch.setitem(engines_deploy.ENGINES, "p1_frozen", engine)
    out = tmp_path / "run"
    out.mkdir()
    lock = {"lock_sha256": "a"*64, "files": {"B2.json": c.sha256_file(planpath)}}
    guard = ex.RunGuard(out, lock, deadline())
    args = SimpleNamespace(models="cnn", plan=str(planpath), units=None, out=str(out), device="cuda", threads=8,
                           audio_root=None, audio_profile="pod", verify_audio=True, fail_fast=True, limit=None, run_guard=guard)
    return args, row, plan, guard, calls


def test_guard_is_integrated_with_engine_and_seals_checkpoint(tmp_path, monkeypatch):
    args, row, plan, guard, calls = unit_fixture(tmp_path, monkeypatch)
    assert gpu.cmd_run(args) == 0
    assert calls == [row["unit_id"]]
    udir = Path(args.out) / "units" / row["unit_id"]
    seal = c.read_json(udir / "SEALED.json")
    assert set(seal["files"]) == {"val_predictions.csv", "test_predictions.csv", "unit.json", "DONE", "history.json", "checkpoint.pt"}
    assert seal["lock_sha256"] == guard.lock["lock_sha256"]
    assert guard.sealed(row, udir)
    assert [c.read_json(udir / "unit.json")["engine_info"]["checkpoint_sha256"], c.sha256_file(udir / "checkpoint.pt")].count(seal["files"]["checkpoint.pt"]) == 2
    assert gpu.cmd_run(args) == 0
    assert len(calls) == 1
    assert guard.attempts[row["unit_id"]] == 1
    (udir / "checkpoint.pt").write_bytes(b"tampered")
    with pytest.raises(c.IntegrityError, match="corrupted"):
        gpu.cmd_run(args)
    assert len(calls) == 1


def test_seal_may_not_omit_checkpoint_entry(tmp_path, monkeypatch):
    args, row, _, guard, _ = unit_fixture(tmp_path, monkeypatch)
    assert gpu.cmd_run(args) == 0
    udir = Path(args.out) / "units" / row["unit_id"]
    seal = c.read_json(udir / "SEALED.json")
    del seal["files"]["checkpoint.pt"]
    c.atomic_write_json(udir / "SEALED.json", seal)
    (udir / "checkpoint.pt").unlink()
    with pytest.raises(c.IntegrityError):
        guard.sealed(row, udir)


def test_attempt_limit_is_durable_across_process_guards(tmp_path):
    row = {"unit_id": "one"}
    lock = {"lock_sha256": "a"*64}
    ex.RunGuard(tmp_path, lock, deadline()).before(row)
    ex.RunGuard(tmp_path, lock, deadline()).before(row)
    with pytest.raises(c.IntegrityError, match="two attempts"):
        ex.RunGuard(tmp_path, lock, deadline()).before(row)
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(c.IntegrityError, match="deadline"):
        ex.RunGuard(tmp_path, lock, past).before({"unit_id": "second"})


def test_exclusive_runner_and_environment_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "verify_lock", lambda *a, **k: {"lock_sha256": "a"*64})
    monkeypatch.setattr(ex, "environment", lambda *a: {"host": "synthetic"})
    with ex.run_context(tmp_path, tmp_path / "unused.json", deadline()):
        with pytest.raises(FileExistsError):
            with ex.run_context(tmp_path, tmp_path / "unused.json", deadline()):
                pass
    assert not (tmp_path / ".managed.lock").exists()
    monkeypatch.setattr(ex, "environment", lambda *a: {"host": "changed"})
    with pytest.raises(c.IntegrityError, match="environment"):
        with ex.run_context(tmp_path, tmp_path / "unused.json", deadline()):
            pass


def test_cuda_device_enforces_requested_eight_threads_without_real_cuda(monkeypatch):
    import torch
    import threadpoolctl
    threads, pools = [], []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: threads.append(n))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "device", lambda name: SimpleNamespace(type=name))
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda limits: pools.append(limits))
    device, guard = gpu.make_device("cuda", 8)
    assert device.type == "cuda" and guard is None
    assert threads == [8] and pools == [8]


def test_formal_entrypoint_passes_guard_and_forces_cuda_defaults(tmp_path, monkeypatch):
    deploy = tmp_path / "v3" / "deploy"
    planpath = deploy / "work" / "plan" / "B2.json"
    c.atomic_write_json(planpath, {"program": c.PROGRAM, "module": "B2", "units": []})
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(c, "DEPLOY_ROOT", deploy)
    guard = SimpleNamespace(lock={"lock_sha256": "a"*64, "files": {"v3/deploy/work/plan/B2.json": c.sha256_file(planpath)}})
    @contextmanager
    def context(*args):
        yield guard
    monkeypatch.setattr(managed, "run_context", context)
    captured = []
    monkeypatch.setattr(gpu, "cmd_run", lambda a: captured.append(a) or 0)
    monkeypatch.setattr(sys, "argv", ["managed_gpu", "--plan", str(planpath), "--out", str(tmp_path / "run"), "--limit", "1", "--deadline", deadline().isoformat()])
    assert managed.main() == 0
    args = captured[0]
    assert args.run_guard is guard and args.device == "cuda" and args.threads == 8
    assert args.verify_audio is True and args.fail_fast is True


def test_formal_entrypoint_rejects_same_count_modified_plan(tmp_path, monkeypatch):
    original = {"program": c.PROGRAM, "module": "B2", "units": [{"unit_id": "same", "train_seed": 1}]}
    deploy = tmp_path / "v3" / "deploy"
    official, alternate = deploy / "work/plan/B2.json", tmp_path / "alternate.json"
    c.atomic_write_json(official, original)
    changed = copy.deepcopy(original)
    changed["units"][0]["train_seed"] = 999
    c.atomic_write_json(alternate, changed)
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(c, "DEPLOY_ROOT", deploy)
    guard = SimpleNamespace(lock={"lock_sha256": "a"*64, "files": {"v3/deploy/work/plan/B2.json": c.sha256_file(official)}})
    @contextmanager
    def context(*args):
        yield guard
    monkeypatch.setattr(managed, "run_context", context)
    invoked = []
    monkeypatch.setattr(gpu, "cmd_run", lambda a: invoked.append(True) or 0)
    monkeypatch.setattr(sys, "argv", ["managed_gpu", "--plan", str(alternate), "--out", str(tmp_path / "run"), "--deadline", deadline().isoformat()])
    with pytest.raises((c.IntegrityError, ValueError)):
        managed.main()
    assert not invoked


def test_formal_entrypoint_rejects_thread_drift(tmp_path, monkeypatch):
    deploy = tmp_path / "v3" / "deploy"
    planpath = deploy / "work/plan/B2.json"
    c.atomic_write_json(planpath, {"program": c.PROGRAM, "module": "B2", "units": []})
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(c, "DEPLOY_ROOT", deploy)
    guard = SimpleNamespace(lock={"lock_sha256": "a"*64, "files": {"v3/deploy/work/plan/B2.json": c.sha256_file(planpath)}})
    @contextmanager
    def context(*args):
        yield guard
    monkeypatch.setattr(managed, "run_context", context)
    invoked = []
    monkeypatch.setattr(gpu, "cmd_run", lambda a: invoked.append(True) or 0)
    monkeypatch.setattr(sys, "argv", ["managed_gpu", "--plan", str(planpath), "--out", str(tmp_path / "run"),
                                    "--deadline", deadline().isoformat(), "--threads", "3"])
    with pytest.raises((c.IntegrityError, SystemExit)):
        managed.main()
    assert not invoked


def test_lock_verifier_rejects_missing_required_plan_entries(tmp_path, monkeypatch):
    one = tmp_path / "random.py"
    one.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path)
    lock = {"schema": "ser26-deploy-execution-lock-2", "program": c.PROGRAM,
            "files": {"random.py": c.sha256_file(one)}, "caches": {}, "splits": {}}
    lock["lock_sha256"] = c.digest(lock)
    p = tmp_path / "lock.json"
    c.atomic_write_json(p, lock)
    with pytest.raises(c.IntegrityError):
        ex.verify_lock(p, profile="gpu", check_environment=False)


def test_same_locked_path_with_modified_bytes_is_refused(tmp_path, monkeypatch):
    deploy = tmp_path / "v3/deploy"
    p = deploy / "work/plan/B2.json"
    plan = {"program": c.PROGRAM, "module": "B2", "units": [{"train_seed": 1}]}
    c.atomic_write_json(p, plan)
    sha = c.sha256_file(p)
    plan["units"][0]["train_seed"] = 2
    c.atomic_write_json(p, plan)
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(c, "DEPLOY_ROOT", deploy)
    guard = SimpleNamespace(lock={"lock_sha256": "a"*64, "files": {"v3/deploy/work/plan/B2.json": sha}})
    @contextmanager
    def context(*args):
        yield guard
    monkeypatch.setattr(managed, "run_context", context)
    invoked = []
    monkeypatch.setattr(gpu, "cmd_run", lambda a: invoked.append(True) or 0)
    monkeypatch.setattr(sys, "argv", ["managed_gpu", "--plan", str(p), "--out", str(tmp_path / "run"), "--deadline", deadline().isoformat()])
    with pytest.raises(c.IntegrityError, match="bound"):
        managed.main()
    assert not invoked


def test_truncated_attempt_journal_is_not_silently_ignored(tmp_path):
    (tmp_path / "managed_attempts.jsonl").write_bytes(b'{"event":"start","unit_id":"x"}')
    with pytest.raises(c.IntegrityError, match="truncated"):
        ex.RunGuard(tmp_path, {"lock_sha256": "a"*64}, deadline())

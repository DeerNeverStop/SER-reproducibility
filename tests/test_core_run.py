"""Failure-oriented runner checks. All predictions and fit calls here are fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import sys
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from v3.data_design import core_run as runner

REPO = Path(runner.__file__).resolve().parents[2]


def bind(unit):
    unit["unit_id"] = runner.digest({k: v for k, v in unit.items() if k != "unit_id"})
    return unit


@pytest.fixture
def panel():
    rows = {}
    people = [f"fit{i:02}" for i in range(12)] + [f"val{i:02}" for i in range(8)] + ["test00", "test01"]
    for speaker in people:
        for prompt in range(12):
            for label, index in runner.LABELS.items():
                path = f"{speaker}_P{prompt:02}_{label}_XX.wav"
                rows[path] = {"speaker": speaker, "sentence": f"P{prompt:02}",
                              "label": label, "label_index": str(index), "intensity": "XX",
                              "relative_path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}
    def paths(speakers, prompts):
        return sorted(p for p, r in rows.items() if r["speaker"] in speakers
                      and int(r["sentence"][1:]) in prompts)
    fit = []
    for i in range(12):
        fit += paths({f"fit{i:02}"}, range(4 * (i % 2), 4 * (i % 2) + 4))
    unit = bind({"model": "ridge_wavlm", "fold": 0, "rotation": 0, "draw": 0,
                 "scenario": "prompt_seen", "B": 288, "S": 12, "P_global": 8,
                 "P_per_speaker": 4, "R": 1, "train_seed": 20260905,
                 "fit": sorted(fit), "val": paths({f"val{i:02}" for i in range(8)}, range(2, 8)),
                 "test": paths({"test00", "test01"}, range(2)),
                 "config": {"model": "ridge_a1_wavlm_base_plus", "feature_state": 12,
                            "alpha": 1.0, "class_weight": "balanced", "solver": "lsqr", "tol": 0.0001}})
    return rows, unit


def predictions(unit):
    return np.zeros((len(unit["test"]), 6), dtype=np.float64)


def completed(out, unit, plan_sha="plan"):
    runner.write_completed(unit, out, plan_sha, predictions(unit),
                           {"fit_seconds": 0.01, "wall_seconds": 0.02, "attempt": 1},
                           {"fixture": True})


def test_budget_speaker_leak_and_prompt_leak_are_rejected(panel):
    rows, unit = panel
    runner.validate_unit(unit, rows, set(rows))
    short = copy.deepcopy(unit)
    short["fit"].pop()
    with pytest.raises(runner.IntegrityError, match="budget|coverage|class"):
        runner.validate_unit(bind(short), rows, set(rows))
    leak = copy.deepcopy(unit)
    leak["val"] = sorted(leak["val"] + leak["fit"][:6])
    with pytest.raises(runner.IntegrityError, match="speaker leakage"):
        runner.validate_unit(bind(leak), rows, set(rows))
    unseen = copy.deepcopy(unit)
    unseen["scenario"] = "prompt_new"
    with pytest.raises(runner.IntegrityError, match="prompt leakage"):
        runner.validate_unit(bind(unseen), rows, set(rows))


def test_byte_corruption_and_semantic_prediction_order_are_rejected(tmp_path, panel):
    _, unit = panel
    completed(tmp_path, unit)
    assert runner.verify_completed(unit, tmp_path, "plan")
    directory = tmp_path / "units" / unit["unit_id"]
    with (directory / "predictions.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(runner.IntegrityError, match="corrupt"):
        runner.verify_completed(unit, tmp_path, "plan")
    # Even internally consistent byte receipts cannot excuse incorrect path order.
    np.savez(directory / "predictions.npz", paths=np.asarray(unit["test"][::-1]), logits=predictions(unit))
    receipt = runner.read_json(directory / "unit.json")
    receipt["predictions_sha256"] = runner.file_sha(directory / "predictions.npz")
    runner.atomic_json(directory / "unit.json", receipt)
    done = runner.read_json(directory / "DONE")
    done.update(predictions_sha256=receipt["predictions_sha256"],
                unit_json_sha256=runner.file_sha(directory / "unit.json"))
    runner.atomic_json(directory / "DONE", done)
    with pytest.raises(runner.IntegrityError, match="path order"):
        runner.verify_completed(unit, tmp_path, "plan")


def test_nonfinite_predictions_never_receive_done(tmp_path, panel):
    _, unit = panel
    logits = predictions(unit)
    logits[0, 0] = np.nan
    with pytest.raises(runner.IntegrityError, match="values"):
        runner.write_completed(unit, tmp_path, "plan", logits, {}, {})
    assert not (tmp_path / "units" / unit["unit_id"] / "DONE").exists()


def test_partial_closure_and_corrupt_closed_result(tmp_path, panel, monkeypatch):
    _, first = panel
    second = bind({**first, "train_seed": first["train_seed"] + 1})
    # A two-unit fixture exercises closure, never a real reduced core.
    monkeypatch.setattr(runner, "EXPECTED_UNITS", 2)
    plan = {"plan_sha256": "plan", "units": [first, second]}
    completed(tmp_path, first)
    assert runner.close_if_complete(plan, tmp_path) is False
    assert not (tmp_path / "analysis_lock.json").exists()
    completed(tmp_path, second)
    assert runner.close_if_complete(plan, tmp_path) is True
    assert runner.close_if_complete(plan, tmp_path) is True
    (tmp_path / "units" / first["unit_id"] / "unit.json").write_text("{}")
    with pytest.raises(runner.IntegrityError, match="corrupt"):
        runner.close_if_complete(plan, tmp_path)


def test_active_lock_rejected_and_stale_lock_audited(tmp_path, monkeypatch):
    with runner.runner_lock(tmp_path):
        lock = runner.read_json(tmp_path / ".core-run.lock")
        assert lock["pid"] == os.getpid() and "argv" in lock
        with pytest.raises(runner.IntegrityError, match="active"):
            with runner.runner_lock(tmp_path):
                pytest.fail("concurrent runner entered")
    runner.atomic_json(tmp_path / ".core-run.lock", {"pid": 9876543, "host": socket.gethostname()})
    monkeypatch.setattr(runner, "pid_alive", lambda _pid: False)
    with runner.runner_lock(tmp_path):
        pass
    assert "stale_lock_recovered" in (tmp_path / "ledger.jsonl").read_text()


def test_feature_paths_must_match_hygienic_order(tmp_path):
    rows = {"a.wav": {}, "b.wav": {}}
    specification = {"feature_files": {}, "feature_sha256": {}}
    for kind, shape, dtype in (("logmel", (2, 64, 128), np.float32),
                               ("wavlm_base_plus", (2, 13, 768), np.float16)):
        path = tmp_path / f"{kind}.npz"
        np.savez(path, X=np.zeros(shape, dtype=dtype), paths=np.asarray(["b.wav", "a.wav"]))
        specification["feature_files"][kind] = path.name
        specification["feature_sha256"][kind] = runner.file_sha(path)
    with pytest.raises(runner.IntegrityError, match="cache path population"):
        runner.StrictFeatures(tmp_path, specification, rows)


@pytest.fixture
def orchestration(tmp_path, panel, monkeypatch):
    rows, unit = panel
    plan = {"plan_sha256": "fixture-plan", "design": {"runtime_complete": True},
            "units": [unit], "source_sha256": {},
            "input": {"manifest_path": "v3/data_design/core_run.py",
                      "demographics_path": "v3/data_design/core_run.py"}}
    plan_path = tmp_path / "fixture-plan.json"
    runner.atomic_json(plan_path, plan)
    class NoFeatures:
        def unchanged(self):
            pass
    monkeypatch.setattr(runner, "load_inputs", lambda *_args: (rows, NoFeatures()))
    monkeypatch.setattr(runner, "environment", lambda *_args: {"fixture": "environment-a"})
    calls = []
    def fixture_fit(current, *_args):
        calls.append(current["unit_id"])
        return predictions(current), {"fit_seconds": 0.001}
    monkeypatch.setattr(runner, "fit_unit", fixture_fit)
    return plan, plan_path, tmp_path / "run", calls


def test_resume_skips_valid_and_rejects_environment_change(orchestration, monkeypatch):
    plan, path, out, calls = orchestration
    first = runner.run(REPO, path, out.parent, out, threads=1)
    assert first["newly_done"] == 1 and first["core_complete"] is False
    second = runner.run(REPO, path, out.parent, out, threads=1)
    assert second["newly_done"] == 0 and second["skipped_valid_done"] == 1 and len(calls) == 1
    monkeypatch.setattr(runner, "environment", lambda *_args: {"fixture": "environment-b"})
    with pytest.raises(runner.IntegrityError, match="environment differs"):
        runner.run(REPO, path, out.parent, out, threads=1)
    assert not (out / "analysis_lock.json").exists()


def test_two_failed_fits_exhaust_unit(orchestration, monkeypatch):
    _, path, out, calls = orchestration
    def failure(*_args):
        calls.append("fit")
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(runner, "fit_unit", failure)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="fixture failure"):
            runner.run(REPO, path, out.parent, out, threads=1)
    with pytest.raises(runner.IntegrityError, match="exhausted two attempts"):
        runner.run(REPO, path, out.parent, out, threads=1)
    assert len(calls) == 2 and not (out / "analysis_lock.json").exists()


def test_orphan_counts_as_first_attempt(orchestration):
    plan, path, out, calls = orchestration
    out.mkdir()
    runner.append_event(out, "unit_start", unit_id=plan["units"][0]["unit_id"], attempt=1)
    result = runner.run(REPO, path, out.parent, out, threads=1)
    assert result["newly_done"] == 1 and len(calls) == 1
    events = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert any(e.get("orphan") is True and e["attempt"] == 1 for e in events)
    assert [e["attempt"] for e in events if e["event"] == "unit_start"] == [1, 2]


def test_draft_and_gpu_gate_block_before_fit(orchestration):
    plan, path, out, calls = orchestration
    plan["design"]["runtime_complete"] = False
    runner.atomic_json(path, plan)
    with pytest.raises(runner.IntegrityError, match="draft plan"):
        runner.run(REPO, path, out.parent, out, threads=1)
    with pytest.raises(runner.IntegrityError, match="GPU release"):
        runner.run(REPO, path, out.parent, out, model="cnn", device="cuda", threads=1)
    assert calls == []


def test_gpu_handoff_requires_verified_byte_bound_evidence(tmp_path):
    objects = {"completion": {"status": "complete", "n_complete_draws": 24},
               "analysis_lock": {"n_complete_draws": 24, "n_locked_units": 1920},
               "verification": {"pass": False}}
    receipt = {"schema": "ser-study2-gpu-release-1", "released": True, "n14r_verification_pass": True}
    for name, obj in objects.items():
        path = tmp_path / f"{name}.json"
        runner.atomic_json(path, obj)
        receipt[name + "_path"] = path.name
        receipt[name + "_sha256"] = runner.file_sha(path)
    path = tmp_path / "release.json"
    runner.atomic_json(path, receipt)
    with pytest.raises(runner.IntegrityError, match="verification failed"):
        runner.validate_gpu_release(path)
    runner.atomic_json(tmp_path / "verification.json", {"pass": True})
    with pytest.raises(runner.IntegrityError, match="identity mismatch"):
        runner.validate_gpu_release(path)


def test_verified_source_mutation_is_rejected(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    guard = runner.FileGuard([path])
    path.write_text("modified source")
    with pytest.raises(runner.IntegrityError, match="changed"):
        guard.check()


def test_independent_host_permission_is_host_plan_and_time_scoped(tmp_path):
    receipt = {"schema": "ser-study2-independent-host-1", "authorized": True,
               "study2_only": True, "host": socket.gethostname(),
               "local_n14r_host": "different-original-host", "plan_sha256": "locked-plan",
               "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
    path = tmp_path / "independent-host.json"
    runner.atomic_json(path, receipt)
    assert runner.validate_gpu_release(path, plan_sha256="locked-plan")["scope"] == "authorized-independent-host"
    with pytest.raises(runner.IntegrityError, match="plan identity"):
        runner.validate_gpu_release(path, plan_sha256="different-plan")
    for field, replacement, match in (
        ("authorized", False, "not authorized"),
        ("host", "wrong-host", "authorized host"),
        ("local_n14r_host", socket.gethostname(), "authorized host"),
        ("expires_at", (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), "expired"),
        ("expires_at", "2099-01-01T00:00:00", "not UTC"),
    ):
        runner.atomic_json(path, {**receipt, field: replacement})
        with pytest.raises(runner.IntegrityError, match=match):
            runner.validate_gpu_release(path, plan_sha256="locked-plan")


def test_cpu_restrictions_precede_data_loading_and_reject_initialized_cuda(orchestration, monkeypatch):
    _, path, out, calls = orchestration
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    with pytest.raises(runner.IntegrityError, match="initialized CUDA"):
        runner.run(REPO, path, out.parent, out, threads=1)
    assert calls == []


def test_cpu_guard_detects_cuda_initialization_at_exit(monkeypatch):
    state = {"initialized": False}
    fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: state["initialized"]))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(runner.IntegrityError, match="initialized CUDA"):
        with runner.cpu_execution_guard("cpu"):
            state["initialized"] = True

"""Generated TEST artifacts only. Opaque NPZ bytes must never be interpreted."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import pytest

from v3.data_design import core_merge as merge


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(merge.canonical(value) + b"\n")


@pytest.fixture(scope="module")
def TEST_sources(tmp_path_factory):
    repo = tmp_path_factory.mktemp("TEST_independent_model_merge")
    sources = {model: repo / ("TEST_source_" + model) for model in merge.MODELS}
    units = []
    for model in merge.MODELS:
        for index in range(720):
            unit = {"model": model, "TEST_index": index}
            unit["unit_id"] = merge.sha_bytes(merge.canonical(unit))
            units.append(unit)
    metadata_path = repo / "TEST_manifest.csv"
    metadata_path.write_bytes(b"TEST metadata only\n")
    code = repo / "TEST_frozen_source.py"
    code.write_bytes(b"# TEST pinned source\n")
    plan = {"program": merge.PROGRAM, "design": {"runtime_complete": True}, "units": units,
            "source_sha256": {code.name: merge.file_sha(code)},
            "input": {"manifest_path": metadata_path.name, "manifest_sha256": merge.file_sha(metadata_path),
                      "demographics_path": metadata_path.name, "demographics_sha256": merge.file_sha(metadata_path)}}
    plan["plan_sha256"] = merge.sha_bytes(merge.canonical(plan))
    plan_path = repo / "TEST_plan.json"
    put(plan_path, plan)
    identity = {"schema": "ser-study2-run-1", "program": merge.PROGRAM, "plan_sha256": plan["plan_sha256"]}
    for model, root in sources.items():
        root.mkdir()
        env = {"model": model, "host": "TEST_windows" if model == "ridge_wavlm" else "TEST_linux",
               "device": "cpu" if model == "ridge_wavlm" else "cuda:0"}
        put(root / "run_identity.json", identity)
        put(root / f"environment_{model}.json", env)
        events = [{"event": "runner_start", "model": model, "plan_sha256": plan["plan_sha256"]}]
        for unit in [u for u in units if u["model"] == model]:
            uid = unit["unit_id"]; folder = root / "units" / uid
            folder.mkdir(parents=True)
            # Not a valid NPZ: an accidental np.load must cause the test to fail.
            prediction = folder / "predictions.npz"
            prediction.write_bytes(b"TEST opaque prediction payload: " + uid.encode())
            attempt = 2 if unit["TEST_index"] == 0 else 1
            for n in range(1, attempt + 1):
                events.append({"event": "unit_start", "unit_id": uid, "model": model, "attempt": n})
                events.append({"event": "unit_done" if n == attempt else "unit_failed", "unit_id": uid,
                               "model": model, "attempt": n})
            receipt = {"schema": "ser-study2-unit-1", "unit_id": uid, "plan_sha256": plan["plan_sha256"],
                       "model": model, "environment": env, "attempt": attempt, "fit_seconds": 1.0,
                       "wall_seconds": 2.0, "predictions_sha256": merge.file_sha(prediction)}
            put(folder / "unit.json", receipt)
            put(folder / "DONE", {"schema": "ser-study2-unit-done-1", "unit_id": uid,
                                   "plan_sha256": plan["plan_sha256"], "unit_json_sha256": merge.file_sha(folder / "unit.json"),
                                   "predictions_sha256": merge.file_sha(prediction)})
        events.append({"event": "runner_stop", "model": model, "plan_sha256": plan["plan_sha256"]})
        # Exercise the join boundary without rewriting the archived original.
        payload = b"\n".join(merge.canonical(e) for e in events)
        if model == "cnn": payload += b"\n"
        (root / "ledger.jsonl").write_bytes(payload)
    return repo, plan, plan_path, sources


@pytest.fixture(autouse=True)
def forbid_arrays(monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("merge must not parse prediction arrays"))


def stub_TEST_metadata(monkeypatch, plan):
    monkeypatch.setattr(merge.core_verify, "verify_plan_file", lambda *a, **k: {
        "pass": True, "plan_sha256": plan["plan_sha256"], "scope": "generated_TEST_fixture"})


def test_full_merge_atomic_publication_and_exact_reuse(TEST_sources, monkeypatch):
    repo, plan, plan_path, sources = TEST_sources
    stub_TEST_metadata(monkeypatch, plan)
    out = repo / "v3/data_design/work/TEST_merged_core"
    result = merge.merge(repo, plan_path, sources["ridge_wavlm"], sources["cnn"], out)
    assert result["pass"] and result["n_units"] == 1440 and not result["reused_existing"]
    assert not result["scientific_results_verified"] and not result["scientific_predictions_parsed"]
    ledger = (sources["ridge_wavlm"] / "ledger.jsonl").read_bytes() + b"\n" + (sources["cnn"] / "ledger.jsonl").read_bytes()
    assert (out / "ledger.jsonl").read_bytes() == ledger
    for model in merge.MODELS:
        assert (out / f"merge_sources/{model}/ledger.jsonl").read_bytes() == (sources[model] / "ledger.jsonl").read_bytes()
        assert (out / f"environment_{model}.json").read_bytes() == (sources[model] / f"environment_{model}.json").read_bytes()
    before = merge.file_sha(out / "MERGE_DONE")
    again = merge.merge(repo, plan_path, sources["ridge_wavlm"], sources["cnn"], out)
    assert again["reused_existing"] and merge.file_sha(out / "MERGE_DONE") == before
    assert not list(out.parent.glob(".TEST_merged_core.merge*"))
    # Corrupted existing output must remain untouched rather than overwritten.
    victim = out / "units" / plan["units"][0]["unit_id"] / "predictions.npz"
    original = victim.read_bytes()
    try:
        victim.write_bytes(b"TEST deliberately damaged output")
        with pytest.raises(ValueError, match="seal mismatch"):
            merge.merge(repo, plan_path, sources["ridge_wavlm"], sources["cnn"], out)
        assert victim.read_bytes() == b"TEST deliberately damaged output"
    finally:
        victim.write_bytes(original)


@pytest.mark.parametrize("kind", ["active_lock", "extra_unit", "incomplete_unit", "partial_closure", "other_environment"])
def test_source_population_and_lock_refusals(TEST_sources, kind):
    _, plan, _, sources = TEST_sources
    root = sources["ridge_wavlm"]
    restore = None
    if kind == "active_lock": extra = root / ".core-run.lock"; extra.write_bytes(b"TEST lock")
    elif kind == "extra_unit": extra = root / "units" / ("f" * 64); extra.mkdir()
    elif kind == "partial_closure": extra = root / "completion.json"; extra.write_bytes(b"{}")
    elif kind == "other_environment": extra = root / "environment_cnn.json"; extra.write_bytes(b"{}")
    else:
        extra = root / "units" / plan["units"][0]["unit_id"] / "DONE"
        restore = extra.read_bytes(); extra.unlink()
    try:
        with pytest.raises((ValueError, OSError)):
            merge.source_inventory(plan, root, "ridge_wavlm")
    finally:
        if restore is not None: extra.write_bytes(restore)
        elif extra.is_dir(): extra.rmdir()
        else: extra.unlink()


@pytest.mark.parametrize("field,value", [("attempt", 1), ("environment", {"host": "TEST_wrong"}),
                                         ("plan_sha256", "e" * 64), ("model", "cnn")])
def test_resealed_wrong_receipts_are_rejected(TEST_sources, field, value):
    _, plan, _, sources = TEST_sources
    root = sources["ridge_wavlm"]; folder = root / "units" / plan["units"][0]["unit_id"]
    paths = [folder / "unit.json", folder / "DONE"]
    originals = {p: p.read_bytes() for p in paths}
    try:
        receipt = json.loads(originals[paths[0]]); receipt[field] = value; put(paths[0], receipt)
        done = json.loads(originals[paths[1]]); done["unit_json_sha256"] = merge.file_sha(paths[0]); put(paths[1], done)
        with pytest.raises(ValueError): merge.audit_source(plan, root, "ridge_wavlm", {})
    finally:
        for path, payload in originals.items(): path.write_bytes(payload)


@pytest.mark.parametrize("kind", ["orphan_start", "third_attempt", "cross_model", "wrong_plan"])
def test_ledger_cannot_be_repaired_or_relabelled(TEST_sources, kind):
    _, plan, _, sources = TEST_sources
    root = sources["ridge_wavlm"]; path = root / "ledger.jsonl"
    original = path.read_bytes(); events = [json.loads(line) for line in original.splitlines()]
    uid = plan["units"][0]["unit_id"]
    if kind == "orphan_start": events = [e for e in events if not (e.get("event") == "unit_done" and e.get("unit_id") == uid)]
    elif kind == "third_attempt": events.append({"event": "unit_start", "unit_id": uid, "attempt": 3})
    elif kind == "cross_model": events.append({"event": "runner_stop", "model": "cnn"})
    else: events[0]["plan_sha256"] = "e" * 64
    try:
        path.write_bytes(b"\n".join(merge.canonical(e) for e in events))
        with pytest.raises(ValueError): merge.audit_source(plan, root, "ridge_wavlm", {})
    finally: path.write_bytes(original)


@pytest.mark.parametrize("target", ["outside", "source", "source_child", "source_parent", "old_v2"])
def test_dangerous_output_paths_are_rejected(TEST_sources, target):
    repo, _, plan_path, sources = TEST_sources
    ridge, cnn = sources["ridge_wavlm"], sources["cnn"]
    choices = {"outside": repo.parent / "TEST_forbidden", "source": ridge, "source_child": ridge / "merged",
               "source_parent": repo, "old_v2": repo / "v2" / "merged"}
    with pytest.raises(ValueError): merge.checked_paths(repo, plan_path, ridge, cnn, choices[target])


def test_injected_copy_failure_keeps_sources_and_no_published_output(TEST_sources, monkeypatch):
    repo, plan, plan_path, sources = TEST_sources
    stub_TEST_metadata(monkeypatch, plan)
    out = repo / "v3/data_design/work/TEST_copy_failure"
    source_before = merge.file_sha(sources["ridge_wavlm"] / "ledger.jsonl")
    def fail(*args): raise OSError("TEST injected interruption")
    monkeypatch.setattr(merge, "copy_verified", fail)
    with pytest.raises(OSError, match="TEST injected"):
        merge.merge(repo, plan_path, sources["ridge_wavlm"], sources["cnn"], out)
    assert not out.exists()
    assert merge.file_sha(sources["ridge_wavlm"] / "ledger.jsonl") == source_before
    assert list(out.parent.glob(".TEST_copy_failure.merge-*"))
    assert not (out.parent / ".TEST_copy_failure.merge.lock").exists()


def test_source_mutation_after_audit_is_detected(TEST_sources):
    _, plan, _, sources = TEST_sources
    path = sources["cnn"] / "ledger.jsonl"; original = path.read_bytes()
    snapshot = {str(path): merge.file_sha(path)}
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="input changed"):
            merge.check_unchanged(plan, sources, snapshot)
    finally: path.write_bytes(original)


def test_existing_merge_lock_is_never_removed(tmp_path):
    out = tmp_path / "TEST_output"
    lock = tmp_path / ".TEST_output.merge.lock"; lock.write_bytes(b"TEST other process")
    with pytest.raises(FileExistsError):
        with merge.exclusive_merge(out): pytest.fail("must not enter locked merge")
    assert lock.read_bytes() == b"TEST other process"


def test_byte_copy_rejects_source_hash_mismatch(tmp_path):
    src, dst = tmp_path / "TEST_src", tmp_path / "TEST_dst"
    src.write_bytes(b"TEST changed")
    with pytest.raises(ValueError, match="copy does not match"):
        merge.copy_verified(src, dst, "0" * 64)
    assert src.read_bytes() == b"TEST changed"


def test_runner_start_during_final_hash_scan_is_rejected(TEST_sources, monkeypatch):
    _, plan, _, sources = TEST_sources
    root = sources["cnn"]; path = root / "ledger.jsonl"; lock = root / ".core-run.lock"
    original_sha = merge.file_sha
    snapshot = {str(path.resolve()): original_sha(path)}
    def start_after_hash(target):
        value = original_sha(target)
        if Path(target) == path: lock.write_bytes(b"TEST runner started during scan")
        return value
    monkeypatch.setattr(merge, "file_sha", start_after_hash)
    try:
        with pytest.raises(ValueError, match="runner lock"):
            merge.check_unchanged(plan, sources, snapshot)
    finally: lock.unlink(missing_ok=True)


@pytest.mark.skipif(os.name != "nt", reason="requires the real Windows process API")
def test_windows_real_cli_help_starts_without_handle_truncation():
    result = subprocess.run([sys.executable, "-m", "v3.data_design.core_merge", "--help"],
                            cwd=Path(merge.__file__).resolve().parents[2], capture_output=True,
                            text=True, timeout=30, creationflags=subprocess.NORMAL_PRIORITY_CLASS)
    assert result.returncode == 0, result.stderr
    assert "--ridge-run" in result.stdout and "--cnn-run" in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="requires the real Windows process API")
def test_windows_real_cli_changes_normal_child_to_below_normal():
    # Execute __main__, then inspect the same child's actual OS priority class.
    # --help exits before any plan/run files can be accessed.
    script = '''
import ctypes
from ctypes import wintypes
import runpy
import sys
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetCurrentProcess.argtypes = ()
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.GetPriorityClass.argtypes = (wintypes.HANDLE,)
kernel.GetPriorityClass.restype = wintypes.DWORD
assert kernel.GetPriorityClass(kernel.GetCurrentProcess()) == 0x20
sys.argv = ["core_merge", "--help"]
try:
    runpy.run_module("v3.data_design.core_merge", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("CLI help must exit before accessing any artifacts")
assert kernel.GetPriorityClass(kernel.GetCurrentProcess()) == 0x4000
print("TEST_PRIORITY_BELOW_NORMAL_CONFIRMED")
'''
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(merge.__file__).resolve().parents[2], capture_output=True,
                            text=True, timeout=30, creationflags=subprocess.NORMAL_PRIORITY_CLASS)
    assert result.returncode == 0, result.stderr
    assert "TEST_PRIORITY_BELOW_NORMAL_CONFIRMED" in result.stdout

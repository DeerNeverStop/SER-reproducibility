"""Merge two closed model blocks without interpreting predictions or fitting.

Only the plan's 720 Ridge and 720 CNN units are accepted. Sources must be
offline snapshots with no runner lock. NPZ files are opaque bytes throughout.
An output is published by a same-filesystem directory rename only after all
checks pass. Interrupted staging directories/locks are retained for inspection;
they are never silently resumed, removed, or used to repair a source ledger.
An existing output is reusable only when its complete provenance verifies.
Run from the repository root with ``python -m v3.data_design.core_merge``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.data_design import core_score, core_verify

PROGRAM = "SER26-STUDY2-CORE-1"
MODELS = ("ridge_wavlm", "cnn")
UNIT_FILES = ("DONE", "unit.json", "predictions.npz")
require = core_verify.require
read_json = core_verify.read_json
file_sha = core_verify.byte_hash
canonical = core_score.canonical


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def ordinary(path, *, directory=False):
    """Reject links/junctions at every traversed component, including Windows."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        status = component.lstat()
        require(not stat.S_ISLNK(status.st_mode)
                and not (getattr(status, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                f"linked path is not an offline ordinary artifact: {component}")
    require(path.is_dir() if directory else path.is_file(), f"required {'directory' if directory else 'file'} missing: {path}")
    return path


def snapshot_file(path, snapshot):
    path = ordinary(path)
    value = file_sha(path)
    snapshot[str(path.resolve())] = value
    return value


def source_inventory(plan, root, model):
    root = ordinary(root, directory=True).resolve()
    require(not (root / ".core-run.lock").exists(), "source has active or unresolved runner lock")
    require(not any((root / name).exists() for name in ("completion.json", "analysis_lock.json")),
            "a 720-unit source must not claim full core closure")
    other = "cnn" if model == "ridge_wavlm" else "ridge_wavlm"
    require(not (root / f"environment_{other}.json").exists(), "source contains the other model environment")
    units = [unit for unit in plan["units"] if unit["model"] == model]
    known = {unit["unit_id"] for unit in units}
    require(len(units) == len(known) == 720, "exactly 720 unique planned units per source required")
    directory = ordinary(root / "units", directory=True)
    require({p.name for p in directory.iterdir()} == known, "source has missing or extra model units")
    names = ["run_identity.json", "ledger.jsonl", f"environment_{model}.json"]
    for uid in sorted(known):
        unit_directory = ordinary(directory / uid, directory=True)
        require({p.name for p in unit_directory.iterdir()} == set(UNIT_FILES), "source has incomplete or extra unit artifacts")
        names.extend(f"units/{uid}/{name}" for name in UNIT_FILES)
    return units, names


def audit_source(plan, root, model, snapshot):
    units, names = source_inventory(plan, root, model)
    hashes = {name: snapshot_file(root / name, snapshot) for name in names}
    identity = {"schema": "ser-study2-run-1", "program": PROGRAM, "plan_sha256": plan["plan_sha256"]}
    require(read_json(root / "run_identity.json") == identity, "source run identity mismatch")
    environment = read_json(root / f"environment_{model}.json")
    require(isinstance(environment, dict) and environment, "missing model environment object")
    known = {unit["unit_id"] for unit in units}
    ledger_payload = (root / "ledger.jsonl").read_bytes()
    require(sha_bytes(ledger_payload) == hashes["ledger.jsonl"], "source ledger changed during audit")
    # Operational events remain untouched, but cannot smuggle a second model/ID.
    for line in ledger_payload.decode("utf-8").splitlines():
        event = json.loads(line, object_pairs_hook=core_verify._object,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite ledger {token}")))
        require(isinstance(event, dict), "invalid source ledger event")
        require("model" not in event or event["model"] == model, "cross-model event in source ledger")
        require("unit_id" not in event or event["unit_id"] in known, "unplanned event in source ledger")
    attempts, ledger_sha = core_score.replay_ledger(root / "ledger.jsonl", known, plan["plan_sha256"])
    require(ledger_sha == hashes["ledger.jsonl"], "source ledger changed during replay")
    for unit in units:
        uid = unit["unit_id"]
        core_score.verify_unit_artifacts(plan["plan_sha256"], unit, root / "units" / uid,
                                         hashes[f"units/{uid}/DONE"], environment, attempts[uid])
    return {"model": model, "path": str(root), "n_units": 720,
            "ledger_sha256": ledger_sha, "artifact_sha256": hashes}, ledger_payload


def check_unchanged(plan, sources, snapshot):
    for model, root in sources.items():
        source_inventory(plan, root, model)
    for name, expected in snapshot.items():
        ordinary(name)
        require(file_sha(name) == expected, f"input changed during merge: {name}")
    # The bulk scan can take time: recheck the runner boundary after it, too.
    for model, root in sources.items():
        source_inventory(plan, root, model)
        ledger = root / "ledger.jsonl"
        expected = snapshot.get(str(ledger.resolve()))
        if expected is not None:
            require(file_sha(ledger) == expected, "source ledger changed at publication boundary")


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, value):
    write_new(path, canonical(value) + b"\n")


def copy_verified(source, target, expected):
    ordinary(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with source.open("rb") as src, target.open("xb") as dst:
        for payload in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(payload)
            hasher.update(payload)
        dst.flush()
        os.fsync(dst.fileno())
    require(hasher.hexdigest() == expected and file_sha(target) == expected, "copy does not match audited source")


def all_files(root):
    ordinary(root, directory=True)
    result = {}
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        for name in subdirectories:
            ordinary(Path(directory) / name, directory=True)
        for name in files:
            path = ordinary(Path(directory) / name)
            result[path.relative_to(root).as_posix()] = path
    return result


def expected_files(provenance):
    expected = {"run_identity.json", "ledger.jsonl", "completion.json", "analysis_lock.json",
                "merge_provenance.json", "MERGE_DONE"}
    for model in MODELS:
        expected.add(f"environment_{model}.json")
        expected.update(f"merge_sources/{model}/{name}" for name in ("ledger.jsonl", "run_identity.json"))
        expected.update(name for name in provenance["sources"][model]["artifact_sha256"] if name.startswith("units/"))
    return expected


def verify_output(plan, root, provenance, merged_ledger):
    """Verify only bytes/receipts. This does not approve scientific results."""
    root = Path(root)
    require(not (root / ".core-run.lock").exists(), "merged output has a runner lock")
    files = all_files(root)
    require(set(files) == expected_files(provenance), "existing merge has missing or extra artifacts")
    require(read_json(root / "merge_provenance.json") == provenance, "merge provenance does not match exact inputs/code")
    seal = read_json(root / "MERGE_DONE")
    hashes = {name: file_sha(path) for name, path in files.items() if name != "MERGE_DONE"}
    require(seal == {"schema": "ser-study2-merge-done-1", "program": PROGRAM,
                     "plan_sha256": plan["plan_sha256"], "artifact_sha256": hashes}, "merge output seal mismatch")
    require((root / "ledger.jsonl").read_bytes() == merged_ledger, "merged ledger is not exact ordered source concatenation")
    for model in MODELS:
        source_hashes = provenance["sources"][model]["artifact_sha256"]
        for name, expected in source_hashes.items():
            target = f"merge_sources/{model}/{name}" if name in ("ledger.jsonl", "run_identity.json") else name
            require(hashes[target] == expected, "merged artifact differs from preserved source")
    # No core_run import: its completion routine would interpret NPZ arrays.
    identity, _ = core_score.verify_closed(plan, root)
    require(identity["ledger_sha256"] == sha_bytes(merged_ledger), "merged ledger identity mismatch")
    return {"schema": "ser-study2-merge-verification-1", "pass": True, "n_units": 1440,
            "plan_sha256": plan["plan_sha256"], "merge_done_sha256": file_sha(root / "MERGE_DONE"),
            "scientific_predictions_parsed": False, "scientific_results_verified": False}


@contextmanager
def exclusive_merge(out):
    lock = out.parent / ("." + out.name + ".merge.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical({"pid": os.getpid(), "out": str(out)}) + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        yield
    finally:
        # Only our lock is removed; staging data is retained on any failure.
        lock.unlink()


def checked_paths(repo, plan_path, ridge_run, cnn_run, out):
    raw = [Path(p).absolute() for p in (repo, plan_path, ridge_run, cnn_run, out)]
    for path, directory in zip(raw[:4], (True, False, True, True)):
        ordinary(path, directory=directory)
    repo, plan_path, ridge, cnn, out = [p.resolve() for p in raw]
    allowed = repo / "v3/data_design/work"
    require(out.is_relative_to(allowed) and out != allowed, "merge output must be a new child of repo/v3/data_design/work")
    paths = (ridge, cnn, out)
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a)
                for i, a in enumerate(paths) for b in paths[i + 1:]), "source/output paths overlap")
    require(not plan_path.is_relative_to(out), "output contains plan")
    for path in (ridge, cnn, out):
        require(not any(part.lower() in {"v2", "v2_1", "n14r"} for part in path.parts), "old v2/N14R path forbidden")
    # For a not-yet-created destination, inspect every existing ancestor.
    for path in (raw[4], *raw[4].parents):
        if path.exists() or path.is_symlink():
            ordinary(path, directory=True)
    return repo, plan_path, {"ridge_wavlm": ridge, "cnn": cnn}, out


def merge(repo, plan_path, ridge_run, cnn_run, out):
    repo, plan_path, sources, out = checked_paths(repo, plan_path, ridge_run, cnn_run, out)
    snapshot = {}
    merger_sha = snapshot_file(Path(__file__).resolve(), snapshot)
    snapshot_file(Path(core_score.__file__).resolve(), snapshot)
    snapshot_file(Path(core_verify.__file__).resolve(), snapshot)
    metadata = core_verify.verify_plan_file(repo, plan_path, metadata_only=True)
    plan_hash = snapshot_file(plan_path, snapshot)
    plan = read_json(plan_path)
    require(plan.get("plan_sha256") == metadata["plan_sha256"] == sha_bytes(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})),
            "plan changed after metadata verification")
    require(plan.get("design", {}).get("runtime_complete") is True, "draft plan cannot be merged")
    for relative, expected in plan["source_sha256"].items():
        require(snapshot_file(repo / relative, snapshot) == expected, "pinned source changed after metadata verification")
    for field in ("manifest", "demographics"):
        require(snapshot_file(repo / plan["input"][field + "_path"], snapshot) == plan["input"][field + "_sha256"],
                "pinned metadata changed after verification")
    blocks, raw_ledgers = {}, {}
    for model in MODELS:
        blocks[model], raw_ledgers[model] = audit_source(plan, sources[model], model, snapshot)
    merged_ledger = raw_ledgers["ridge_wavlm"]
    if not merged_ledger.endswith(b"\n"):
        merged_ledger += b"\n"
    merged_ledger += raw_ledgers["cnn"]
    provenance = {"schema": "ser-study2-merge-1", "program": PROGRAM, "plan_sha256": plan["plan_sha256"],
                  "plan_file_sha256": plan_hash, "merger_sha256": merger_sha,
                  "closure_checker_sha256": snapshot[str(Path(core_score.__file__).resolve())],
                  "n_units": 1440, "sources": blocks, "ledger_source_order": list(MODELS),
                  "ledger_join": "ridge original bytes; append LF only if absent; cnn original bytes",
                  "merged_ledger_sha256": sha_bytes(merged_ledger), "new_fit_events": 0,
                  "prediction_arrays_parsed": False, "metadata_verification": metadata}
    out.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_merge(out):
        if out.exists():
            result = verify_output(plan, out, provenance, merged_ledger)
            check_unchanged(plan, sources, snapshot)
            return {**result, "reused_existing": True, "out": str(out)}
        staging = out.parent / ("." + out.name + ".merge-" + uuid.uuid4().hex)
        staging.mkdir()
        for model in MODELS:
            for name, expected in blocks[model]["artifact_sha256"].items():
                target = f"merge_sources/{model}/{name}" if name in ("ledger.jsonl", "run_identity.json") else name
                copy_verified(sources[model] / name, staging / target, expected)
        copy_verified(sources["ridge_wavlm"] / "run_identity.json", staging / "run_identity.json",
                      blocks["ridge_wavlm"]["artifact_sha256"]["run_identity.json"])
        write_new(staging / "ledger.jsonl", merged_ledger)
        core_score.replay_ledger(staging / "ledger.jsonl", {u["unit_id"] for u in plan["units"]}, plan["plan_sha256"])
        mapping = {u["unit_id"]: blocks[u["model"]]["artifact_sha256"][f"units/{u['unit_id']}/DONE"] for u in plan["units"]}
        write_json(staging / "completion.json", {"schema": "ser-study2-completion-1", "program": PROGRAM,
                   "plan_sha256": plan["plan_sha256"], "n_units": 1440, "units_done": 1440, "done_sha256": mapping})
        write_json(staging / "analysis_lock.json", {"schema": "ser-study2-analysis-lock-1", "program": PROGRAM,
                   "plan_sha256": plan["plan_sha256"], "n_units": 1440, "done_sha256": mapping,
                   "completion_sha256": file_sha(staging / "completion.json")})
        write_json(staging / "merge_provenance.json", provenance)
        output_hashes = {name: file_sha(path) for name, path in all_files(staging).items()}
        write_json(staging / "MERGE_DONE", {"schema": "ser-study2-merge-done-1", "program": PROGRAM,
                   "plan_sha256": plan["plan_sha256"], "artifact_sha256": output_hashes})
        result = verify_output(plan, staging, provenance, merged_ledger)
        check_unchanged(plan, sources, snapshot)
        require(not out.exists(), "output appeared during merge; refusing overwrite")
        # On the target Windows host rename refuses an existing destination.
        os.rename(staging, out)
        return {**result, "reused_existing": False, "out": str(out)}


def main(argv=None):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = ()
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.SetPriorityClass.restype = wintypes.BOOL
        require(bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)),
                f"cannot set BelowNormal process priority (Windows error {ctypes.get_last_error()})")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "ridge-run", "cnn-run", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = merge(args.repo, args.plan, args.ridge_run, args.cnn_run, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"pass": False, "error": str(exc), "scientific_predictions_parsed": False}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

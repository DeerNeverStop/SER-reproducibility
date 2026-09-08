"""Separate-process disk replay for four preselected formal dual-validation units.

Frozen architecture and preprocessing are reused; this is not an independent
model implementation. No fitting, HPO, scientific scoring or downloads occur.
This nested audit tool is intentionally outside the training source glob.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

ROLES = ("val_seen", "val_unseen", "test")
CHECKPOINTS = ("best_seen", "best_unseen", "last")
FIXED_IDS = [f"dual_d00_f0_c{i}" for i in range(4)]
ATOL = RTOL = 1e-5


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "file changed while hashing")
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def select_units(plan):
    units = [u for u in plan["units"] if u["draw"] == u["fold"] == 0]
    require(len(units) == 4 and {u["config_index"] for u in units} == set(range(4)),
            "fixed four-configuration panel missing or duplicated")
    units.sort(key=lambda u: u["config_index"])
    require([u["unit_id"] for u in units] == FIXED_IDS, "fixed formal unit identities differ")
    require(all(u[r] == units[0][r] for u in units for r in ROLES),
            "configuration changes evaluation population/order")
    return units


def load_expected(path, unit):
    import numpy as np
    expected = {}
    with np.load(path, allow_pickle=False) as saved:
        keys = ({f"{r}__{c}__logits" for r in ROLES for c in CHECKPOINTS}
                | {f"{r}__{k}" for r in ROLES for k in ("paths", "labels")})
        require(set(saved.files) == keys, "prediction archive schema differs")
        for role in ROLES:
            paths = saved[role + "__paths"]
            require(paths.dtype.kind == "U" and paths.tolist() == unit[role], "prediction path order differs")
            for ck in CHECKPOINTS:
                key = f"{role}__{ck}__logits"
                value = saved[key]
                require(value.dtype == np.float64 and value.shape == (len(unit[role]), 6)
                        and np.isfinite(value).all(), "invalid stored logits")
                expected[key] = value.copy()
    return expected


def compare(expected, actual):
    import numpy as np
    actual = np.asarray(actual, dtype=np.float64)
    require(actual.shape == expected.shape and np.isfinite(actual).all(), "invalid restored logits")
    return {"pass": bool(np.allclose(expected, actual, rtol=RTOL, atol=ATOL)),
            "max_abs_diff": float(np.max(np.abs(expected - actual), initial=0.)),
            "rows": len(expected), "classes": 6, "rtol": RTOL, "atol": ATOL}


def check_module(module, repo, relative):
    require(Path(module.__file__).resolve() == (repo / relative).resolve(),
            "loaded module is outside the pinned checkout: " + relative)


def current_environment(torch, np):
    extra = {}
    for package in ("torchaudio", "librosa", "soundfile"):
        try:
            extra[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            extra[package] = None
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "threads": torch.get_num_threads(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "audit_extra_package_versions": extra}


def check_environment(saved, actual):
    for key in ("python", "numpy", "torch", "cuda", "cudnn", "gpu", "threads", "matmul_tf32", "cudnn_tf32"):
        require(saved.get(key) == actual.get(key), "original execution environment differs: " + key)


def execute(args):
    started = time.perf_counter()
    repo, outroot = Path(args.repo).resolve(), Path(args.outroot).resolve()
    out, plan_path = Path(args.out).resolve(), Path(args.plan).resolve()
    model_path, audio_root = Path(args.model).resolve(), Path(args.audio_root).resolve()
    require(not out.exists(), "refusing repeated/existing audit output")
    require(out != repo and repo not in out.parents and out != outroot and outroot not in out.parents,
            "audit output must be outside frozen source and result directories")
    operator_sha, plan_file_sha = file_sha(__file__), file_sha(plan_path)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(repo))
    planner = importlib.import_module("v3.inner_validation.plan")
    engine = importlib.import_module("v3.inner_validation.engine")
    runner = importlib.import_module("v3.inner_validation.run")
    deployment = importlib.import_module("v3.deploy.engines_deploy")
    for module, relative in ((planner, "v3/inner_validation/plan.py"),
                             (engine, "v3/inner_validation/engine.py"),
                             (runner, "v3/inner_validation/run.py"),
                             (deployment, "v3/deploy/engines_deploy.py"),
                             (engine.legacy, "v3/speaker_coverage/run.py")):
        check_module(module, repo, relative)
    plan = planner.load_plan(plan_path, repo)
    units = select_units(plan)
    metadata = planner.read_metadata(repo)
    require(metadata["input"] == plan["input"], "metadata changed after plan loading")
    rows = {r["relative_path"]: r for r in metadata["clean"]}
    base_sha = file_sha(model_path)
    require(base_sha == plan["base_file_sha256"], "offline base bytes differ from plan")
    pinned = {str(plan_path): plan_file_sha, str(model_path): base_sha}
    for kind in ("manifest", "demographics"):
        path = (repo / plan["input"][kind + "_path"]).resolve()
        sha = plan["input"][kind + "_sha256"]
        require(file_sha(path) == sha, "metadata bytes differ from plan")
        pinned[str(path)] = sha
    committed = {}
    for unit in units:
        receipt = runner.verify_unit(unit, outroot / "formal", plan["plan_sha256"], rows)
        require(receipt is not None, "fixed formal unit is not committed: " + unit["unit_id"])
        directory = outroot / "formal/units" / unit["unit_id"]
        done = read_json(directory / "DONE")
        pins = {str(directory / "DONE"): file_sha(directory / "DONE")}
        pins.update({str(directory / name): sha for name, sha in done["artifacts"].items()})
        committed[unit["unit_id"]] = (directory / "attempts/0001", receipt, pins)
        pinned.update(pins)
    eval_paths = sorted({p for u in units for role in ROLES for p in u[role]})
    for relative in eval_paths:
        path = (audio_root / relative).resolve()
        require(audio_root in path.parents, "audio path escapes its root")
        require(file_sha(path) == rows[relative]["sha256"], "evaluation waveform bytes changed")
        pinned[str(path)] = rows[relative]["sha256"]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    require(torch.cuda.is_available(), "original CUDA environment required; no CPU fallback")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    environment = current_environment(torch, np)
    for _, receipt, _ in committed.values():
        check_environment(receipt["environment"], environment)
    def forbidden(*unused, **kwargs):
        raise RuntimeError("fitting/download is prohibited in checkpoint audit")
    engine.fit_dual = engine.full_dual_loop = forbidden
    engine.legacy.fit_ft = engine.legacy.full_ft_loop = forbidden
    torch.hub.download_url_to_file = forbidden
    out.mkdir(parents=True, exist_ok=False)
    request = {"schema": "ser-dual-validation-restore-request-1", "plan_sha256": plan["plan_sha256"],
               "plan_file_sha256": plan_file_sha, "operator_source_sha256": operator_sha,
               "unit_ids": FIXED_IDS, "phase": "formal", "roles": list(ROLES),
               "checkpoint_names": list(CHECKPOINTS), "environment": environment,
               "base_file_sha256": base_sha, "evaluation_audio_count": len(eval_paths),
               "input_scope": "selected evaluation waveforms; no fit waveforms required for replay",
               "source_sha256": plan["sources"], "scientific_scores_computed": False,
               "architecture_scope": "shared frozen engine architecture/preprocessing; separate-process disk restoration"}
    write_once(out / "request.json", request)
    prepared = time.perf_counter()
    wav = deployment.WaveCache(audio_root, 16000)
    for relative in eval_paths:
        wav(relative)
    waveform_seconds = time.perf_counter() - prepared
    reports, inference_seconds = [], 0.
    with threadpool_limits(limits=1):
        for unit in units:
            attempt, receipt, pins = committed[unit["unit_id"]]
            expected = load_expected(attempt / "predictions.npz", unit)
            begun = time.perf_counter()
            saved = torch.load(attempt / "checkpoint.pt", map_location="cpu", weights_only=True)
            model, params_sha = engine.build_ft_model(model_path, unit["train_seed"])
            selected = {"best_seen": receipt["best_seen_epoch"], "best_unseen": receipt["best_unseen_epoch"], "last": 15}
            frozen = engine.frozen_hash(model)
            require(frozen == receipt["frozen_parameter_sha256"], "receipt frozen base hash differs")
            engine._validate_saved(saved, model, unit, selected, params_sha, base_sha, frozen)
            model.to(device)
            torch.cuda.synchronize(device)
            constructor_seconds = time.perf_counter() - begun
            batch = engine._make_batch(unit, audio_root, wav)
            comparisons, state_seconds, forward_seconds = {}, {}, {}
            for epoch in sorted(set(selected.values())):
                begun = time.perf_counter()
                engine.load_delta(model, saved["epoch_states"][str(epoch)], frozen)
                torch.cuda.synchronize(device)
                state_seconds[str(epoch)] = time.perf_counter() - begun
                for role in ROLES:
                    begun = time.perf_counter()
                    actual = engine.predict_wave(model, unit[role], batch, device)
                    torch.cuda.synchronize(device)
                    duration = time.perf_counter() - begun
                    forward_seconds[f"epoch{epoch}__{role}"] = duration
                    inference_seconds += duration
                    for ck in CHECKPOINTS:
                        if selected[ck] == epoch:
                            key = f"{role}__{ck}__logits"
                            comparisons[key] = compare(expected[key], actual)
            require(len(comparisons) == 9, "incomplete checkpoint/role replay")
            for path, sha in pins.items():
                require(file_sha(path) == sha, "committed result changed during restoration")
            result = {"schema": "ser-dual-validation-restore-unit-1", "unit_id": unit["unit_id"],
                      "pass": all(c["pass"] for c in comparisons.values()), "comparisons": comparisons,
                      "max_abs_diff": max(c["max_abs_diff"] for c in comparisons.values()),
                      "selected_epochs": selected, "unique_epochs": sorted(set(selected.values())),
                      "base_file_sha256": base_sha, "base_state_sha256": saved["base_state_sha256"],
                      "bundle_params_sha256": params_sha, "frozen_parameter_sha256": frozen,
                      "environment": environment, "model_constructor_seconds": constructor_seconds,
                      "epoch_state_load_seconds": state_seconds, "forward_seconds": forward_seconds,
                      "committed_file_sha256": pins, "scientific_scores_computed": False}
            write_once(out / (unit["unit_id"] + ".json"), result)
            reports.append(result)
            print(json.dumps({"unit_id": unit["unit_id"], "pass": result["pass"],
                              "unique_epochs": result["unique_epochs"], "max_abs_diff": result["max_abs_diff"]}), flush=True)
            del model, saved, expected, actual
            gc.collect()
            torch.cuda.empty_cache()
    for path, sha in pinned.items():
        require(file_sha(path) == sha, "pinned input/result changed during audit")
    require(planner.source_files(repo) == plan["sources"], "frozen source changed during audit")
    require(file_sha(__file__) == operator_sha, "audit operator changed during process")
    summary = {"schema": "ser-dual-validation-restore-summary-1", "pass": all(r["pass"] for r in reports),
               "plan_sha256": plan["plan_sha256"], "units_checked": len(reports), "checkpoint_role_comparisons": 36,
               "reports": {r["unit_id"]: file_sha(out / (r["unit_id"] + ".json")) for r in reports},
               "request_sha256": file_sha(out / "request.json"), "base_file_sha256": base_sha,
               "environment": environment, "input_and_artifact_sha256": pinned,
               "waveform_prepare_seconds": waveform_seconds, "forward_seconds_total": inference_seconds,
               "total_wall_seconds": time.perf_counter() - started,
               "verified_at": datetime.now(timezone.utc).isoformat(), "scientific_scores_computed": False,
               "limitation": "Four prespecified formal units only; shared architecture and preprocessing, separate-process disk replay. No model quality claim."}
    write_once(out / "summary.json", summary)
    print(json.dumps({"pass": summary["pass"], "units_checked": 4, "summary": str(out / "summary.json")}), flush=True)
    return 0 if summary["pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "outroot", "audio-root", "model", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        return execute(args)
    except Exception as error:
        print(json.dumps({"pass": False, "error_type": type(error).__name__, "error": str(error)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

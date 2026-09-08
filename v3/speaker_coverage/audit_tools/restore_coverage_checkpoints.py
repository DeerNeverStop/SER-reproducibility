#!/usr/bin/env python3
"""Separate-process checkpoint restoration for a fixed nine-unit technical audit.

Only formal fold=0/rotation=0/draw=0, U/R/C x Ridge/CNN/FT (FT seed_index=0).
No fitting, accuracy/UAR calculation, policy ranking, model download, or edits
to frozen research sources. Original test logits are read only for numerical
replay comparison. This reuses frozen architectures and inference functions;
it is a separate-process disk restoration check, not an independent algorithm.

Example on the original RTX 5090 environment:
 python restore_coverage_checkpoints.py --repo /workspace/ser-coverage \
   --plan /workspace/coverage-plan/plan.json --results-root /workspace/coverage-runs \
   --features /workspace/coverage-features --audio-root /path/to/cremad \
   --wavlm-model /path/to/wavlm_base_plus.pth --out /workspace/ops-coverage/restore9 \
   --model all --threads 1

Use --describe for a metadata-only fixed selection inventory. Use --self-test
for synthetic CPU tests, never real predictions. A model filter selects the
three predetermined policies for that model; arbitrary unit IDs are forbidden.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import tempfile
import time

# Import frozen modules without writing bytecode into their checkout.
sys.dont_write_bytecode = True

COMMIT = "db7836c0ac3564ed44849aa6f8d553984b846064"
PLAN_SHA = "8ea69b57b7030e4b431f503d8562de230cefed2cfa7d4d092f703e8b164cae4f"
PLAN_FILE_SHA = "2c201b05bde2d4b07216ace064a542c0a32f0ed469a119ec8dc034acbe289774"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = ROOT / "SER-speaker-execution-20260906"
FIXED = {
    "ridge_wavlm_U": "7ed442bd04c4c3a11f77bb6ea917e193372eaef60d7346be0ff253dbe507b19a",
    "ridge_wavlm_R": "5cd83051abe4b26db4b932fd6bbcd68639b25fb4c48d94113e42dd82ebcdf706",
    "ridge_wavlm_C": "3ef963d8d7f711f4267bfa123d6ad7d9f8940d76866d6b6af1352f109934100c",
    "cnn_U": "e20ec904e9a522fbc684aa425fb720858dc55f5894757c13964a6cf993ecbee4",
    "cnn_R": "8005aa9d68d864735207b475ff9ced7ef1656686a4ef3565384b9ac675402b8b",
    "cnn_C": "e89582d649eddd4aa62117cd9114bc9b6dd687e04cf2868c3b5d9e5b728d23a3",
    "wavlm_ft_U": "c5b67673cbd1f8e475cdd70fedece2f90a4d8d956a26b31c0fd3637862a76d09",
    "wavlm_ft_R": "c791d766b9a4ed59f02f06e723952f5ee89e252045715d6671c571780d52c202",
    "wavlm_ft_C": "1aed66ca0826c77fc958d07b116b442658072a2ead98aead10cd04d3a1e1b78f",
}
MODELS = ("ridge_wavlm", "cnn", "wavlm_ft")
RTOL = ATOL = 1e-5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def no_links(path):
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            value = item.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(value.st_mode) and not
                (getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)),
                "symlink/reparse path refused")
    return path


def relative(root, name):
    parsed = PurePosixPath(name)
    require(isinstance(name, str) and name and str(parsed) == name and not parsed.is_absolute()
            and ".." not in parsed.parts and "\\" not in name and ":" not in name, "unsafe relative path")
    boundary = no_links(root)
    path = no_links(boundary.joinpath(*parsed.parts))
    require(os.path.commonpath([str(path), str(boundary)]) == str(boundary), "path outside root")
    return path


def sha_file(path):
    path = no_links(path)
    require(path.is_file(), "missing regular file: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "file changed while hashing")
    return h.hexdigest()


def read_json(path):
    return json.loads(no_links(path).read_text(encoding="utf-8"))


def write_once(path, value):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def select(plan, model="all"):
    require(plan.get("schema") == "ser-speaker-coverage-1" and plan.get("plan_sha256") == PLAN_SHA,
            "only the fixed coverage plan is supported")
    require(plan["plan_sha256"] == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "plan content hash differs")
    chosen = [u for u in plan["units"] if u["fold"] == u["rotation"] == u["draw"] == 0
              and (u["model"] != "wavlm_ft" or u["seed_index"] == 0)]
    require(len(chosen) == 9 and {u["model"] + "_" + u["policy"]: u["unit_id"] for u in chosen} == FIXED,
            "fixed nine-unit selection differs")
    for unit in chosen:
        require(unit["unit_id"] == digest({k: v for k, v in unit.items() if k != "unit_id"})
                and len(unit["test"]) == len(set(unit["test"])) == 228, "selected unit hash/test support differs")
    require(len({tuple(u["test"]) for u in chosen}) == 1, "selected test path order differs across models/policies")
    require(model == "all" or model in MODELS, "unsupported model filter")
    return sorted((u for u in chosen if model == "all" or u["model"] == model),
                  key=lambda u: (MODELS.index(u["model"]), ("U", "R", "C").index(u["policy"])))


def check_sources(repo, plan):
    required = {"v3/speaker_coverage/run.py", "v3/deploy/engines_deploy.py", "v3/data_design/core_run.py",
                "advanced_models.py", "tuned_standard_experiment.py", "v2/ser_v2/train.py"}
    require(required <= plan["source_sha256"].keys(), "inference source pins missing")
    paths = []
    for name, expected in plan["source_sha256"].items():
        path = relative(repo, name)
        require(sha_file(path) == expected, "frozen source changed: " + name)
        paths.append(path)
    return paths


def inspect_commit(unit, results_root, plan):
    directory = relative(results_root, "formal/units/" + unit["unit_id"])
    done_path = directory / "DONE"
    done_hash = sha_file(done_path)
    done = read_json(done_path)
    identity = {"unit_id": unit["unit_id"], "plan_sha256": PLAN_SHA, "phase": "formal",
                "model": unit["model"], "block": unit["block"]}
    require(done.get("schema") == "ser-speaker-coverage-done-1" and
            all(done.get(k) == v for k, v in identity.items()), "formal DONE identity differs")
    number = done.get("attempt")
    require(type(number) is int and number > 0, "invalid committed attempt")
    prefix = f"attempts/{number:04d}/"
    checkpoint = "checkpoint.npz" if unit["model"] == "ridge_wavlm" else "checkpoint.pt"
    names = {prefix + n for n in (checkpoint, "predictions.npz", "receipt.json", "history.json")}
    require(set(done.get("artifacts", {})) == names, "committed artifact set differs")
    pinned = {str(done_path): done_hash}
    for name, expected in done["artifacts"].items():
        path = relative(directory, name)
        require(sha_file(path) == expected, "committed artifact bytes changed")
        pinned[str(path)] = expected
    receipt = read_json(directory / prefix / "receipt.json")
    require(receipt.get("schema") == "ser-speaker-coverage-result-1" and
            all(receipt.get(k) == v for k, v in identity.items()) and receipt.get("attempt") == number
            and receipt.get("unit_config_sha256") == digest(unit["config"])
            and receipt.get("input_sha256") == digest(plan["input"]), "receipt identity differs")
    require(receipt.get("checkpoint_reload_verified") is True, "runner did not complete checkpoint reload check")
    return {"directory": directory, "checkpoint": directory / prefix / checkpoint,
            "predictions": directory / prefix / "predictions.npz", "receipt": receipt, "pins": pinned}


def expected_logits(path, test_paths):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {"paths", "labels", "logits", "pred", "proba", "last_logits", "last_pred", "last_proba"},
                "prediction archive fields differ")
        paths = saved["paths"]
        require(paths.dtype.kind == "U" and paths.tolist() == test_paths, "prediction path order differs")
        arrays = {name: saved[name].copy() for name in ("logits", "last_logits")}
    for values in arrays.values():
        require(values.shape == (len(test_paths), 6) and values.dtype == np.float64
                and np.isfinite(values).all(), "stored logits invalid")
    # Labels, predicted classes, and probabilities are deliberately not loaded.
    return arrays


def compare(expected, restored):
    import numpy as np
    restored = np.asarray(restored, dtype=np.float64)
    require(restored.shape == expected.shape and np.isfinite(restored).all(), "restored logits invalid")
    difference = np.abs(restored - expected)
    return {"pass": bool(np.allclose(restored, expected, rtol=RTOL, atol=ATOL)),
            "max_abs_diff": float(difference.max()) if difference.size else 0.0,
            "rtol": RTOL, "atol": ATOL, "rows": int(expected.shape[0]), "classes": 6}


def replay_ridge(checkpoint, x_test):
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import LabelBinarizer, StandardScaler
    with np.load(checkpoint, allow_pickle=False) as saved:
        require(set(saved.files) == {"coef", "intercept", "classes", "scaler_mean", "scaler_scale", "scaler_var", "n_features_in", "n_samples_seen"},
                "Ridge checkpoint fields differ")
        values = {key: saved[key].copy() for key in saved.files}
    require(values["classes"].tolist() == list(range(6)), "Ridge class order differs")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = (values[k] for k in ("scaler_mean", "scaler_scale", "scaler_var"))
    scaler.n_features_in_, scaler.n_samples_seen_ = int(values["n_features_in"]), int(values["n_samples_seen"])
    model = RidgeClassifier()  # restore fitted state; no fit/fit_transform call
    model.coef_, model.intercept_ = values["coef"], values["intercept"]
    # RidgeClassifier.classes_ is a read-only property backed by this fitted
    # label transformer. Restore its known six-class metadata without fitting.
    model._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
    model._label_binarizer.classes_ = values["classes"]
    model._label_binarizer.y_type_ = "multiclass"
    model._label_binarizer.sparse_input_ = False
    model.n_features_in_ = scaler.n_features_in_
    require(np.asarray(x_test).dtype == np.float32 and x_test.shape[1] == scaler.n_features_in_, "Ridge test cache dtype/dimension differs")
    return model, scaler


def replay_cnn(checkpoint, unit, device, engine, advanced_models):
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved.get("schema") == "ser-speaker-coverage-cnn-checkpoint-1" and saved.get("config") == unit["config"]
            and saved.get("train_seed") == unit["train_seed"] and saved.get("epochs_run") == 100,
            "CNN checkpoint identity/config differs")
    arch, _ = engine.p1_arch(unit["config"])
    engine.set_seed(unit["train_seed"])
    model = advanced_models.build_neural_model("cnn", n_mels=64, n_classes=6, config=arch).to(device)
    return model, saved


def validate_ft(saved, unit, model, bundle_sha, base_sha, runner):
    require(saved.get("schema") == "ser-speaker-coverage-ft-delta-1" and saved.get("config") == unit["config"]
            and saved.get("train_seed") == unit["train_seed"] and saved.get("epochs_run") == 15,
            "FT checkpoint identity/config differs")
    require(saved.get("base_file_sha256") == base_sha and saved.get("base_state_sha256") == runner.WEIGHTS_SHA
            and saved.get("bundle_params_sha256") == bundle_sha, "FT base/bundle identity differs")
    trainable, buffers = runner.delta_names(model)
    require(saved.get("trainable_parameter_names") == trainable and saved.get("buffer_names") == buffers,
            "FT trainable/buffer inventory differs")
    require(runner.frozen_hash(model) == saved.get("frozen_parameter_sha256"), "FT frozen base parameters differ")


def environment_matches(saved, current):
    for key in ("python", "executable", "threads", "versions", "torch", "torchaudio", "cuda", "device"):
        require(saved.get(key) == current.get(key), "restoration environment differs: " + key)
    if current["device"] == "cuda":
        require(saved.get("gpu") == current.get("gpu") and "5090" in current["gpu"], "original RTX 5090 environment required")


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def execute(args):
    started = time.perf_counter()
    repo, out = no_links(args.repo), no_links(args.out)
    results_root = no_links(args.results_root)
    require(repo != out and repo not in out.parents and results_root != out and results_root not in out.parents,
            "technical report output must be outside frozen repo and scientific results")
    require(not out.exists(), "use a new technical report directory")
    require(sha_file(args.plan) == PLAN_FILE_SHA, "frozen plan file bytes differ")
    plan = read_json(args.plan)
    units = select(plan, args.model)
    source_paths = check_sources(repo, plan)
    commits = {u["unit_id"]: inspect_commit(u, results_root, plan) for u in units}
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    import torchaudio
    from threadpoolctl import threadpool_limits
    from v3.speaker_coverage import run as runner
    from v3.deploy import engines_deploy as engine
    import advanced_models
    for module, expected in ((runner, "v3/speaker_coverage/run.py"), (engine, "v3/deploy/engines_deploy.py"),
                             (runner.core, "v3/data_design/core_run.py"), (advanced_models, "advanced_models.py")):
        require(Path(module.__file__).resolve() == relative(repo, expected), "imported inference module came from another checkout")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Explicit fail-closed tripwires; none of these functions is used below.
    def forbidden(*unused, **kwargs):
        raise RuntimeError("training/download is prohibited in checkpoint restoration")
    for name in ("fit_cnn", "fit_ridge", "fit_ft", "full_cnn_loop", "full_ft_loop"):
        setattr(runner, name, forbidden)
    torch.hub.download_url_to_file = forbidden
    inp = plan["input"]
    manifest = relative(repo, inp["manifest_path"])
    require(sha_file(manifest) == inp["manifest_sha256"], "audio metadata bytes differ")
    rows = runner.core.load_manifest(manifest)
    features = None
    used = set(u["model"] for u in units)
    input_pins = {str(no_links(args.plan)): PLAN_FILE_SHA, str(manifest): inp["manifest_sha256"]}
    if used.intersection(("cnn", "ridge_wavlm")):
        require(args.features is not None, "CNN/Ridge require the frozen feature directory")
        features = runner.core.StrictFeatures(no_links(args.features), inp, rows)
        input_pins.update({str(path): inp["feature_sha256"][kind] for kind, path in features.files.items()})
    wave, base_sha, waveform_seconds = None, None, 0.0
    if "wavlm_ft" in used:
        require(args.audio_root is not None and args.wavlm_model is not None, "FT requires existing audio and offline WavLM base")
        base_sha = sha_file(args.wavlm_model)
        require(base_sha == inp["wavlm_model_sha256"], "offline WavLM file differs")
        input_pins[str(no_links(args.wavlm_model))] = base_sha
        for path in units[0]["test"]:
            actual = relative(args.audio_root, path)
            require(sha_file(actual) == rows[path]["sha256"], "test waveform bytes differ")
            input_pins[str(actual)] = rows[path]["sha256"]
        wave = engine.WaveCache(no_links(args.audio_root), 16000)
        begun = time.perf_counter()
        for path in units[0]["test"]:
            wave(path)
        waveform_seconds = time.perf_counter() - begun
    # Both caches retain exact frozen semantics, but materialization is timed
    # outside the forward-pass budget. Only test rows are requested.
    test_arrays = {}
    for model, kind, state in (("cnn", "logmel", None), ("ridge_wavlm", "wavlm_base_plus", 12)):
        if model in used:
            test_arrays[model] = features.get("cremad", kind, units[0]["test"], state)
    out.mkdir(parents=True)
    request = {"schema": "ser-coverage-restore9-request-1", "plan_sha256": PLAN_SHA, "plan_file_sha256": PLAN_FILE_SHA,
        "architecture_reference_commit": COMMIT, "operator_source_sha256": sha_file(__file__), "model_filter": args.model,
        "unit_ids": [u["unit_id"] for u in units], "threads": args.threads, "rtol": RTOL, "atol": ATOL,
        "inference_precision": "FP32_no_autocast_for_CNN_and_FT;Ridge_saved_scaler_and_estimator_dtypes",
        "scope": "separate_process_disk_restoration_reusing_frozen_architecture_preprocessing_and_inference",
        "scientific_metrics_computed": False, "fit_functions_called": False, "download_allowed": False}
    write_once(out / "request.json", request)
    reports, forward_seconds = [], 0.0
    with threadpool_limits(limits=args.threads):
        for unit in units:
            committed = commits[unit["unit_id"]]
            device = torch.device("cpu" if unit["model"] == "ridge_wavlm" else "cuda")
            if device.type == "cuda":
                require(torch.cuda.is_available(), "original CUDA environment unavailable; no CPU substitution")
            current = runner.core.environment(device.type, args.threads)
            current.update(torch=torch.__version__, torchaudio=torchaudio.__version__, cuda=torch.version.cuda,
                           gpu=torch.cuda.get_device_name() if device.type == "cuda" else None)
            environment_matches(committed["receipt"]["environment"], current)
            expected = expected_logits(committed["predictions"], unit["test"])
            load_start = time.perf_counter()
            saved = None
            if unit["model"] == "ridge_wavlm":
                model, scaler = replay_ridge(committed["checkpoint"], test_arrays["ridge_wavlm"])
            elif unit["model"] == "cnn":
                model, saved = replay_cnn(committed["checkpoint"], unit, device, engine, advanced_models)
            else:
                saved = torch.load(committed["checkpoint"], map_location="cpu", weights_only=True)
                model, bundle_sha = runner.build_ft_model(no_links(args.wavlm_model), unit["train_seed"])
                validate_ft(saved, unit, model, bundle_sha, base_sha, runner)
                model.to(device)
            if saved is not None:
                require(saved.get("best_epoch") == committed["receipt"]["best_epoch"], "saved checkpoint epoch differs from DONE receipt")
            restore_seconds = time.perf_counter() - load_start
            comparisons, times, state_load_times = {}, {}, {}
            for which, key in (("best", "logits"), ("last", "last_logits")):
                sync(torch, device)
                state_started = time.perf_counter()
                if unit["model"] == "cnn":
                    model.load_state_dict(saved[which + "_state"], strict=True)
                elif unit["model"] == "wavlm_ft":
                    runner.load_delta(model, saved[which + "_state"], saved["frozen_parameter_sha256"])
                sync(torch, device)
                state_load_times[which] = time.perf_counter() - state_started
                begun = time.perf_counter()
                if unit["model"] == "ridge_wavlm":
                    restored = np.asarray(model.decision_function(scaler.transform(test_arrays["ridge_wavlm"])), dtype=np.float64)
                elif unit["model"] == "cnn":
                    restored = engine.predict_torch(model, test_arrays["cnn"], device)
                else:
                    rng = np.random.RandomState(unit["train_seed"])
                    def batch(paths, train):
                        require(train is False, "random training crop is prohibited")
                        return engine.wavlm_batch(paths, False, wave, int(unit["config"]["crop_seconds"] * 16000),
                                                  int(unit["config"]["eval_cap_seconds"] * 16000), rng)
                    restored = runner.predict_wave(model, unit["test"], batch, device)
                sync(torch, device)
                times[which] = time.perf_counter() - begun
                forward_seconds += times[which]
                comparisons[which] = compare(expected[key], restored)
            # Byte checks are repeated after restoration, not inferred from
            # unchanged filenames or the runner's earlier reload receipt.
            for path, expected_sha in committed["pins"].items():
                require(sha_file(path) == expected_sha, "committed bytes changed during restoration")
            result = {"unit_id": unit["unit_id"], "model": unit["model"], "policy": unit["policy"],
                "pass": all(v["pass"] for v in comparisons.values()), "comparisons": comparisons,
                "model_build_and_checkpoint_read_seconds": restore_seconds,
                "state_load_and_validation_seconds": state_load_times,
                "inference_seconds": times, "committed_file_sha256": committed["pins"],
                "environment": current, "scientific_metrics_computed": False}
            write_once(out / (unit["unit_id"] + ".json"), result)
            reports.append(result)
            print(canonical({"unit_id": unit["unit_id"], "pass": result["pass"], "best_max_abs_diff": comparisons["best"]["max_abs_diff"],
                             "last_max_abs_diff": comparisons["last"]["max_abs_diff"], "inference_seconds": sum(times.values())}), flush=True)
            del model, saved, expected, restored
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for path, expected_sha in input_pins.items():
        require(sha_file(path) == expected_sha, "inference input changed during process")
    check_sources(repo, plan)
    require(sha_file(__file__) == request["operator_source_sha256"], "restoration script changed during process")
    summary = {"schema": "ser-coverage-restore9-summary-1", "finished_at": datetime.now(timezone.utc).isoformat(),
        "pass": all(r["pass"] for r in reports), "plan_sha256": PLAN_SHA, "request_sha256": digest(request),
        "units_requested": len(units), "units_checked": len(reports), "full_nine_checked": len(reports) == 9,
        "reports": {r["unit_id"]: sha_file(out / (r["unit_id"] + ".json")) for r in reports},
        "input_sha256": input_pins, "source_sha256": plan["source_sha256"],
        "waveform_prepare_seconds": waveform_seconds, "inference_seconds_total": forward_seconds,
        "inference_under_one_minute": forward_seconds <= 60, "total_wall_seconds": time.perf_counter() - started,
        "scientific_metrics_computed": False, "fit_functions_called": False,
        "limitation": "Nine predetermined units only. Separate process and disk restore; frozen inference algorithm reused, not independently reimplemented. This does not establish model quality or policy superiority."}
    write_once(out / "summary.json", summary)
    print(canonical({"pass": summary["pass"], "units_checked": len(reports), "inference_seconds": forward_seconds,
                     "total_wall_seconds": summary["total_wall_seconds"], "summary": str(out / "summary.json")}))
    return 0 if summary["pass"] else 1


def self_test(repo):
    """Synthetic CPU data only; no saved study predictions, GPU, or training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(no_links(repo)))
    from v3.speaker_coverage import run as runner
    from v3.deploy import engines_deploy as engine
    import advanced_models
    checks = 0
    a = np.arange(18, dtype=np.float64).reshape(3, 6)
    require(compare(a, a.copy())["pass"] and not compare(a, a + 1)["pass"], "comparison tolerance failure")
    checks += 1
    with tempfile.TemporaryDirectory(prefix="coverage_restore_SYNTHETIC_") as name:
        root = Path(name)
        paths = ["SYNTHETIC_1", "SYNTHETIC_2", "SYNTHETIC_3"]
        archive = {"paths": np.asarray(paths), "logits": a, "last_logits": a + 1,
                   "labels": np.zeros(3, dtype=np.int64), "pred": np.zeros(3, dtype=np.int64),
                   "last_pred": np.zeros(3, dtype=np.int64), "proba": a, "last_proba": a}
        np.savez(root / "fake_predictions.npz", **archive)
        require(expected_logits(root / "fake_predictions.npz", paths)["last_logits"][0, 0] == 1, "best/last key confusion")
        checks += 1
        try:
            expected_logits(root / "fake_predictions.npz", paths[::-1])
        except ValueError:
            checks += 1
        else:
            raise AssertionError("test path permutation accepted")
        # Fitted-state fixture is created algebraically, never with model.fit.
        x = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        coef = np.arange(12, dtype=np.float64).reshape(6, 2) / 10
        mean, scale = np.array([.25, -.5]), np.array([2., 3.])
        np.savez(root / "fake_ridge.npz", coef=coef, intercept=np.arange(6, dtype=np.float64), classes=np.arange(6),
                 scaler_mean=mean, scaler_scale=scale, scaler_var=scale ** 2, n_features_in=np.array(2), n_samples_seen=np.array(576))
        model, scaler = replay_ridge(root / "fake_ridge.npz", x)
        transformed = x.copy()
        transformed -= mean
        transformed /= scale
        wanted = transformed @ coef.T + np.arange(6)
        require(compare(wanted, model.decision_function(scaler.transform(x)))["pass"], "Ridge float32 scaler restoration differs")
        checks += 1
        cfg = {"model": "cnn", "lr": .001, "weight_decay": .0001, "dropout": .1}
        unit = {"config": cfg, "train_seed": 123}
        arch, _ = engine.p1_arch(cfg)
        engine.set_seed(123)
        original = advanced_models.build_neural_model("cnn", n_mels=64, n_classes=6, config=arch)
        best = runner.cpu_state(original)
        last = {k: v.clone() for k, v in best.items()}
        bias = next(k for k, v in last.items() if k.endswith("bias") and v.ndim == 1 and len(v) == 6)
        last[bias] = last[bias] + 0.1
        torch.save({"schema": "ser-speaker-coverage-cnn-checkpoint-1", "config": cfg, "train_seed": 123,
                    "epochs_run": 100, "best_epoch": 1, "best_state": best, "last_state": last}, root / "fake_cnn.pt")
        samples = np.random.default_rng(22).normal(size=(3, 64, 128)).astype(np.float32)
        expected = engine.predict_torch(original, samples, torch.device("cpu"))
        rebuilt, saved = replay_cnn(root / "fake_cnn.pt", unit, torch.device("cpu"), engine, advanced_models)
        rebuilt.load_state_dict(saved["best_state"], strict=True)
        require(compare(expected, engine.predict_torch(rebuilt, samples, torch.device("cpu")))["pass"], "synthetic CNN disk rebuild differs")
        rebuilt.load_state_dict(saved["last_state"], strict=True)
        require(not compare(expected, engine.predict_torch(rebuilt, samples, torch.device("cpu")))["pass"], "best/last CNN difference lost")
        checks += 2
        class TinyDelta(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)
                self.head = torch.nn.Parameter(torch.tensor([3.]))
                self.register_buffer("offset", torch.tensor([.5]))
        tiny = TinyDelta()
        frozen = runner.frozen_hash(tiny)
        torch.save({"best_state": {"head": torch.tensor([5.]), "offset": torch.tensor([.5])},
                    "last_state": {"head": torch.tensor([7.]), "offset": torch.tensor([.6])}}, root / "fake_delta.pt")
        saved = torch.load(root / "fake_delta.pt", weights_only=True)
        runner.load_delta(tiny, saved["best_state"], frozen)
        require(tiny.head.item() == 5 and tiny.frozen.item() == 2, "delta failed to preserve frozen base")
        runner.load_delta(tiny, saved["last_state"], frozen)
        require(tiny.head.item() == 7 and runner.frozen_hash(tiny) == frozen, "last delta/frozen identity mismatch")
        checks += 1
        try:
            runner.load_delta(tiny, {"head": torch.tensor([1.])}, frozen)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("incomplete delta buffer inventory accepted")
    print(canonical({"pass": True, "checks": checks, "real_study_predictions_read": False,
                     "training_performed": False, "gpu_used": False, "synthetic_fixtures_only": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--wavlm-model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.repo)
        return 0
    args.plan = args.plan or args.repo / "v3/speaker_coverage/work/plan/plan.json"
    if args.describe:
        require(sha_file(args.plan) == PLAN_FILE_SHA, "frozen plan bytes differ")
        units = select(read_json(args.plan), args.model)
        print(json.dumps({"units": [{k: u[k] for k in ("unit_id", "model", "policy", "fold", "rotation", "draw")} for u in units],
            "test_rows_each": 228, "FT_forward_batches_both_states": sum(u["model"] == "wavlm_ft" for u in units) * 30,
            "neural_inference": "FP32_no_autocast_original_batching", "real_predictions_read": False}, indent=2))
        return 0
    require(args.results_root is not None and args.out is not None and args.threads > 0, "results-root, new out directory, and positive threads required")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())

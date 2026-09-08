"""Run the locked Study II core without calculating or displaying test scores.

The full 1,440-unit plan is checked independently of its compiler.  Ridge is a
CPU-only entry point; CNN needs an explicit, verified N14R handoff receipt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import socket
import sys
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

PROGRAM = "SER26-STUDY2-CORE-1"
PLAN_SCHEMA = "ser-study2-core-1"
EXPECTED_UNITS = 1440
LABELS = {"angry": 0, "disgust": 1, "fearful": 2, "happy": 3, "neutral": 4, "sad": 5}
KINDS = {"logmel", "wavlm_base_plus"}


class IntegrityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            IntegrityError(f"nonfinite JSON constant: {value}")))


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def append_event(out, event, **fields):
    with (Path(out) / "ledger.jsonl").open("ab") as stream:
        stream.write(canonical({"event": event, "at": now(), **fields}) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def relative_file(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "expected a normalized relative path")
    p = PurePosixPath(relative)
    require(not p.is_absolute() and ".." not in p.parts and ":" not in relative,
            "absolute or escaping relative path")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    require(result.is_relative_to(root), "resolved path escapes its root")
    return result


def load_manifest(path):
    """Independent raw-to-hygienic CREMA-D mapping; never synthesize input."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    require(len(raw) == 7442, "expected the pinned 7,442-row raw CREMA-D manifest")
    by_hash, by_path = defaultdict(list), {}
    for row in raw:
        p = row["relative_path"]
        relative_file(Path(path).parent, p)
        require(p not in by_path, "duplicate manifest path")
        require(row["label"] in LABELS and int(row["label_index"]) == LABELS[row["label"]],
                "manifest class mapping differs from the frozen six classes")
        sha = row["sha256"]
        require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                "invalid byte identity")
        by_hash[sha].append(row)
        by_path[p] = row
    drop = set()
    for group in by_hash.values():
        if len(group) > 1:
            excluded = group if len({r["label"] for r in group}) > 1 else sorted(
                group, key=lambda r: r["relative_path"])[1:]
            drop.update(r["relative_path"] for r in excluded)
    drop.add("1076_MTI_SAD_XX.wav")
    rows = {p: by_path[p] for p in sorted(by_path) if p not in drop}
    require(len(rows) == 7435, "hygienic CREMA-D population must contain 7,435 paths")
    return rows


def validate_unit(unit, rows, representatives):
    require(unit.get("unit_id") == digest({k: v for k, v in unit.items() if k != "unit_id"}),
            "unit identity mismatch")
    model = unit.get("model")
    require(model in {"ridge_wavlm", "cnn"}, "unknown model")
    require(unit.get("fold") in range(5) and unit.get("rotation") in range(6)
            and unit.get("draw") in range(3), "out-of-design fold, rotation or draw")
    require(unit.get("scenario") in {"prompt_seen", "prompt_new"}, "unknown scenario")
    require(unit.get("B") in {288, 576} and unit.get("S") in {12, 48}, "unknown policy")
    require(unit.get("P_global") == 8 and unit.get("R") == 1
            and unit.get("P_per_speaker") == unit["B"] // (6 * unit["S"]),
            "invalid prompt or repeat quota")
    require(type(unit.get("train_seed")) is int and 0 <= unit["train_seed"] < 2**32,
            "invalid training seed")
    partitions = {}
    for role in ("fit", "val", "test"):
        paths = unit.get(role)
        require(isinstance(paths, list) and paths and paths == sorted(set(paths)),
                f"{role} paths must be nonempty, sorted and unique")
        require(all(p in rows and p in representatives for p in paths),
                f"{role} contains an unknown, excluded or nonrepresentative recording")
        partitions[role] = {
            "speakers": {rows[p]["speaker"] for p in paths},
            "prompts": {rows[p]["sentence"] for p in paths},
            "bytes": {rows[p]["sha256"] for p in paths},
        }
        require(len(partitions[role]["bytes"]) == len(paths), f"{role} byte duplicates")
        for speaker in partitions[role]["speakers"]:
            require({int(rows[p]["label_index"]) for p in paths if rows[p]["speaker"] == speaker}
                    == set(range(6)), f"{role} speaker lacks a class")
    for left, right in itertools.combinations(partitions, 2):
        require(partitions[left]["speakers"].isdisjoint(partitions[right]["speakers"]),
                f"{left}/{right} speaker leakage")
        require(partitions[left]["bytes"].isdisjoint(partitions[right]["bytes"]),
                f"{left}/{right} byte leakage")
    fit, val, test = (partitions[r] for r in ("fit", "val", "test"))
    require(len(unit["fit"]) == unit["B"] and len(fit["speakers"]) == unit["S"]
            and len(fit["prompts"]) == 8, "fit budget or coverage mismatch")
    require(len(val["speakers"]) == 8 and len(val["prompts"]) == 6
            and len(test["prompts"]) == 2, "evaluation role sizes mismatch")
    require(val["prompts"].isdisjoint(test["prompts"])
            and val["prompts"] <= fit["prompts"], "early-stop content is not the common pool")
    if unit["scenario"] == "prompt_new":
        require(fit["prompts"].isdisjoint(test["prompts"]), "test prompt leakage")
    else:
        require(test["prompts"] <= fit["prompts"], "seen test prompts were not covered")
    cells = Counter((rows[p]["speaker"], rows[p]["sentence"], rows[p]["label"])
                    for p in unit["fit"])
    require(set(cells.values()) == {1}, "intensity variants counted as repeats")
    for speaker in fit["speakers"]:
        prompts = {rows[p]["sentence"] for p in unit["fit"] if rows[p]["speaker"] == speaker}
        require(len(prompts) == unit["P_per_speaker"], "per-speaker prompt quota mismatch")
        require(all(cells[(speaker, prompt, label)] == 1 for prompt in prompts for label in LABELS),
                "fit speaker/prompt cell lacks a class")
    cfg = unit.get("config", {})
    if model == "ridge_wavlm":
        required = {"model": "ridge_a1_wavlm_base_plus", "feature_state": 12,
                    "alpha": 1.0, "class_weight": "balanced", "solver": "lsqr", "tol": 0.0001}
        require(all(cfg.get(k) == v for k, v in required.items()), "Ridge config mismatch")
    else:
        require(all(cfg.get(k) == v for k, v in {"model": "cnn", "batch_size": 32,
                    "epochs": 100, "patience": 15}.items()), "CNN config mismatch")
        require(all(type(cfg.get(k)) in (int, float) and math.isfinite(cfg[k])
                    for k in ("lr", "weight_decay", "dropout")), "invalid CNN optimizer config")
        require(cfg["lr"] > 0 and cfg["weight_decay"] >= 0 and 0 <= cfg["dropout"] < 1,
                "invalid CNN optimizer domain")


def validate_plan(plan, rows):
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("program") == PROGRAM,
            "wrong study or plan schema")
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "plan hash mismatch")
    units = plan.get("units", [])
    require(len(units) == EXPECTED_UNITS and len({u["unit_id"] for u in units}) == EXPECTED_UNITS,
            "core requires exactly 1,440 unique planned units")
    cells = defaultdict(list)
    for p, row in rows.items():
        cells[(row["speaker"], row["sentence"], row["label"])].append(p)
    representatives = set()
    for paths in cells.values():
        options = [p for p in paths if rows[p]["intensity"] in {"MD", "XX"}]
        if options:
            representatives.add(min(options, key=lambda p: (
                0 if rows[p]["intensity"] == "MD" else 1, p)))
    combinations = set()
    evaluations, model_panels = {}, {}
    for unit in units:
        validate_unit(unit, rows, representatives)
        key = tuple(unit[k] for k in ("model", "fold", "rotation", "draw", "scenario", "B", "S"))
        require(key not in combinations, "duplicate scientific design cell")
        combinations.add(key)
        pairing = tuple(unit[k] for k in ("fold", "rotation", "draw"))
        actual = (tuple(unit["val"]), tuple(unit["test"]))
        require(pairing not in evaluations or evaluations[pairing] == actual,
                "test or early-stop panel differs across paired conditions")
        evaluations[pairing] = actual
        model_key = key[1:]
        require(model_key not in model_panels or model_panels[model_key] == unit["fit"],
                "models do not share the same training panel")
        model_panels[model_key] = unit["fit"]
    expected = set(itertools.product(("ridge_wavlm", "cnn"), range(5), range(6), range(3),
                                     ("prompt_seen", "prompt_new"), (288, 576), (12, 48)))
    require(combinations == expected, "incomplete core design grid")


def validate_sources(repo, plan):
    sources = plan.get("source_sha256", {})
    require("v3/data_design/core_run.py" in sources, "runner source must be pinned")
    for relative, expected in sources.items():
        path = relative_file(repo, relative)
        require(path.is_file() and file_sha(path) == expected, f"source identity mismatch: {relative}")


class StrictFeatures:
    """Only explicitly pinned files; path identity is checked before any fit."""
    def __init__(self, root, specification, rows):
        self.files, self.arrays, self.index, self.stats = {}, {}, {}, {}
        require(set(specification["feature_files"]) == KINDS
                and set(specification["feature_sha256"]) == KINDS, "incomplete feature identity")
        expected_paths = sorted(rows)
        for kind in sorted(KINDS):
            name = specification["feature_files"][kind]
            require(PurePosixPath(name).name == name, "feature name must be an explicit basename")
            path = relative_file(root, name)
            require(path.is_file() and file_sha(path) == specification["feature_sha256"][kind],
                    f"feature byte identity mismatch: {kind}")
            with np.load(path, allow_pickle=False) as data:
                require(set(data.files) == {"X", "paths"}, "unexpected cache arrays")
                paths = data["paths"]
                require(paths.ndim == 1 and paths.dtype.kind == "U"
                        and paths.tolist() == expected_paths, "cache path population or order mismatch")
            with zipfile.ZipFile(path) as archive, archive.open("X.npy") as stream:
                version = np.lib.format.read_magic(stream)
                require(version in {(1, 0), (2, 0)}, "unsupported cache array header")
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(stream)
            suffix = (64, 128) if kind == "logmel" else (13, 768)
            dtype_expected = np.dtype("float32" if kind == "logmel" else "float16")
            require(shape == (len(rows), *suffix) and dtype == dtype_expected,
                    "cache dimensions or dtype mismatch")
            self.files[kind] = path
            self.index[kind] = {p: i for i, p in enumerate(expected_paths)}
            self.stats[kind] = (path.stat().st_size, path.stat().st_mtime_ns)

    def unchanged(self):
        for kind, path in self.files.items():
            require((path.stat().st_size, path.stat().st_mtime_ns) == self.stats[kind],
                    "feature file changed during this invocation")

    def get(self, corpus, kind, paths, state=None):
        require(corpus == "cremad" and kind in self.files, "unregistered corpus or cache")
        self.unchanged()
        if kind not in self.arrays:
            with np.load(self.files[kind], allow_pickle=False) as data:
                array = data["X"]
            require(np.isfinite(array).all(), "nonfinite feature values")
            self.arrays[kind] = array
        require(all(p in self.index[kind] for p in paths), "unknown requested feature path")
        indices = np.asarray([self.index[kind][p] for p in paths], dtype=np.int64)
        array = self.arrays[kind][indices]
        if state is not None:
            require(kind == "wavlm_base_plus" and state == 12, "unregistered feature layer")
            array = array[:, state, :]
        return np.asarray(array, dtype=np.float32)


class StrictLabels:
    def __init__(self, rows):
        self.rows = rows
        self.n_classes = {"cremad": 6}

    def labels(self, base, paths):
        require(base == "cremad", "unregistered corpus")
        return np.asarray([int(self.rows[p]["label_index"]) for p in paths], dtype=np.int64)


def load_inputs(repo, plan, feature_root):
    specification = plan["input"]
    path = relative_file(repo, specification["manifest_path"])
    require(path.is_file() and file_sha(path) == specification["manifest_sha256"],
            "raw manifest identity mismatch")
    demographics = relative_file(repo, specification["demographics_path"])
    require(demographics.is_file() and file_sha(demographics) == specification["demographics_sha256"],
            "demographics identity mismatch")
    rows = load_manifest(path)
    validate_plan(plan, rows)
    validate_sources(repo, plan)
    return rows, StrictFeatures(feature_root, specification, rows)


class FileGuard:
    """Observe mutation after the initial byte verification, without repeated I/O."""
    def __init__(self, paths):
        self.stats = {Path(p): (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths}

    def check(self):
        for path, expected in self.stats.items():
            require(path.is_file() and (path.stat().st_size, path.stat().st_mtime_ns) == expected,
                    f"verified input or source changed during invocation: {path.name}")


def validate_gpu_release(path, *, plan_sha256=None):
    require(path is not None, "CNN requires an explicit verified N14R GPU release")
    path = Path(path).resolve()
    receipt = read_json(path)
    if receipt.get("schema") == "ser-study2-independent-host-1":
        require(receipt.get("authorized") is True and receipt.get("study2_only") is True,
                "independent-host use was not authorized for Study II")
        require(receipt.get("host") == socket.gethostname()
                and isinstance(receipt.get("local_n14r_host"), str)
                and receipt["local_n14r_host"]
                and receipt["local_n14r_host"] != receipt["host"],
                "independent-host receipt does not identify a different authorized host")
        require(isinstance(plan_sha256, str) and plan_sha256
                and receipt.get("plan_sha256") == plan_sha256,
                "independent-host receipt plan identity mismatch")
        try:
            expiry = datetime.fromisoformat(receipt["expires_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError("invalid independent-host expiry") from exc
        require(expiry.tzinfo is not None and expiry.utcoffset().total_seconds() == 0
                and expiry > datetime.now(timezone.utc), "independent-host receipt is expired or not UTC")
        return {"path": str(path), "sha256": file_sha(path), "scope": "authorized-independent-host",
                "expires_at": receipt["expires_at"]}
    require(receipt.get("schema") == "ser-study2-gpu-release-1"
            and receipt.get("released") is True
            and receipt.get("n14r_verification_pass") is True, "invalid GPU release")
    evidence = {}
    for name in ("completion", "analysis_lock", "verification"):
        raw_path = Path(receipt[name + "_path"])
        target = raw_path.resolve() if raw_path.is_absolute() else (path.parent / raw_path).resolve()
        require(target.is_file() and file_sha(target) == receipt.get(name + "_sha256"),
                f"GPU release {name} identity mismatch")
        evidence[name] = read_json(target)
    require(evidence["completion"].get("status") == "complete"
            and evidence["completion"].get("n_complete_draws") == 24,
            "N14R completion is not complete")
    require(evidence["analysis_lock"].get("n_locked_units") == 1920
            and evidence["analysis_lock"].get("n_complete_draws") == 24,
            "N14R analysis lock is incomplete")
    require(evidence["verification"].get("pass") is True, "N14R independent verification failed")
    return {"path": str(path), "sha256": file_sha(path)}


def pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return True  # malformed or unknown locks are not safe to take over
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # invalid PID vs access denied
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def runner_lock(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".core-run.lock"
    value = {"pid": os.getpid(), "host": socket.gethostname(), "created_at": now(),
             "nonce": uuid.uuid4().hex, "executable": sys.executable, "argv": list(sys.argv)}
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        old = read_json(lock)
        require(old.get("host") == socket.gethostname() and not pid_alive(old.get("pid")),
                "output directory has an active or unverifiable runner lock")
        os.replace(lock, out / (".core-run.lock.stale-" + uuid.uuid4().hex))
        append_event(out, "stale_lock_recovered", previous_pid=old.get("pid"))
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise IntegrityError("another runner acquired the output lock") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield
    finally:
        if lock.exists() and read_json(lock).get("nonce") == value["nonce"]:
            lock.unlink()


def environment(device, threads):
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    value = {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
             "platform": platform.platform(), "host": socket.gethostname(),
             "device": device, "threads": threads, "versions": versions}
    if device == "cuda":
        import torch
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        value.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version())
    return value


def assert_cpu_has_no_cuda():
    torch_module = sys.modules.get("torch")
    require(torch_module is None or not torch_module.cuda.is_initialized(),
            "CPU-only execution cannot use an already initialized CUDA runtime")


def restrict_cpu_process():
    """Restrict only this new process; never change the running N14R worker."""
    assert_cpu_has_no_cuda()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    priority = "unchanged"
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel.GetPriorityClass.restype = wintypes.DWORD
        handle = kernel.GetCurrentProcess()
        require(bool(kernel.SetPriorityClass(handle, 0x4000)) and kernel.GetPriorityClass(handle) == 0x4000,
                "cannot set and verify BelowNormal process priority")
        priority = "below_normal"
    return {"cuda_visible_devices": "-1", "cuda_initialized": False, "priority": priority}


@contextmanager
def cpu_execution_guard(device):
    if device == "cpu":
        assert_cpu_has_no_cuda()
    try:
        yield
    finally:
        if device == "cpu":
            assert_cpu_has_no_cuda()


def fit_unit(unit, rows, feats, device):
    labels = StrictLabels(rows)
    if unit["model"] == "ridge_wavlm":
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
        cfg = unit["config"]
        x_fit = feats.get("cremad", "wavlm_base_plus", unit["fit"], 12)
        x_test = feats.get("cremad", "wavlm_base_plus", unit["test"], 12)
        y_fit = labels.labels("cremad", unit["fit"])
        started = time.perf_counter()
        scaler = StandardScaler().fit(x_fit)
        model = RidgeClassifier(alpha=cfg["alpha"], class_weight=cfg["class_weight"],
                                solver=cfg["solver"], tol=cfg["tol"], random_state=unit["train_seed"])
        model.fit(scaler.transform(x_fit), y_fit)
        fit_seconds = time.perf_counter() - started
        require(model.classes_.tolist() == list(range(6)), "fitted Ridge class order mismatch")
        logits = model.decision_function(scaler.transform(x_test))
        return logits, {"fit_seconds": fit_seconds, "epochs_run": 0, "best_epoch": None}
    from v2.ser_v2 import train as legacy_train
    require(Path(legacy_train.__file__).resolve() == Path(__file__).resolve().parents[2] / "v2/ser_v2/train.py",
            "CNN import resolved outside this pinned checkout")
    row = {"base_corpus": "cremad", "train_seed": unit["train_seed"]}
    fold = {role: unit[role] for role in ("fit", "val", "test")}
    logits, _history, info, _model = legacy_train.engine_p1_frozen(row, unit["config"], fold, labels, feats, device)
    # Never publish history/validation accuracy or derive any test metric here.
    return logits, {"fit_seconds": info["train_seconds"], "epochs_run": info["epochs_run"],
                    "best_epoch": info["best_epoch"]}


def validate_predictions(path, unit):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"paths", "logits"}, "unexpected prediction arrays")
        paths, logits = data["paths"], data["logits"]
        require(paths.ndim == 1 and paths.dtype.kind == "U" and paths.tolist() == unit["test"],
                "prediction path order or coverage differs from test")
        require(logits.shape == (len(unit["test"]), 6) and logits.dtype.kind == "f"
                and np.isfinite(logits).all(), "invalid prediction dimensions or values")


def verify_completed(unit, out, plan_sha256):
    directory = Path(out) / "units" / unit["unit_id"]
    done_path = directory / "DONE"
    if not done_path.exists():
        return False
    done = read_json(done_path)
    require(done.get("schema") == "ser-study2-unit-done-1"
            and done.get("unit_id") == unit["unit_id"]
            and done.get("plan_sha256") == plan_sha256, "DONE identity mismatch")
    for filename, field in (("unit.json", "unit_json_sha256"), ("predictions.npz", "predictions_sha256")):
        path = directory / filename
        require(path.is_file() and file_sha(path) == done.get(field), f"completed {filename} is corrupt")
    receipt = read_json(directory / "unit.json")
    require(receipt.get("schema") == "ser-study2-unit-1"
            and receipt.get("unit_id") == unit["unit_id"]
            and receipt.get("plan_sha256") == plan_sha256
            and receipt.get("model") == unit["model"]
            and isinstance(receipt.get("environment"), dict) and receipt["environment"]
            and receipt.get("predictions_sha256") == done["predictions_sha256"],
            "unit receipt identity mismatch")
    require(all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k])
                and receipt[k] >= 0 for k in ("fit_seconds", "wall_seconds")), "invalid timing receipt")
    validate_predictions(directory / "predictions.npz", unit)
    return True


def write_completed(unit, out, plan_sha256, logits, timing, runtime):
    directory = Path(out) / "units" / unit["unit_id"]
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / "DONE").exists(), "refusing to overwrite a completed unit")
    temp = directory / ("predictions.npz.tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("wb") as stream:
            np.savez(stream, paths=np.asarray(unit["test"], dtype=str),
                     logits=np.asarray(logits, dtype=np.float64))
            stream.flush()
            os.fsync(stream.fileno())
        validate_predictions(temp, unit)
        os.replace(temp, directory / "predictions.npz")
    finally:
        temp.unlink(missing_ok=True)
    prediction_sha = file_sha(directory / "predictions.npz")
    receipt = {"schema": "ser-study2-unit-1", "unit_id": unit["unit_id"],
               "plan_sha256": plan_sha256, "model": unit["model"],
               "environment": runtime, "predictions_sha256": prediction_sha,
               "finished_at": now(), **timing}
    atomic_json(directory / "unit.json", receipt)
    atomic_json(directory / "DONE", {"schema": "ser-study2-unit-done-1",
                "unit_id": unit["unit_id"], "plan_sha256": plan_sha256,
                "unit_json_sha256": file_sha(directory / "unit.json"),
                "predictions_sha256": prediction_sha})
    verify_completed(unit, out, plan_sha256)


def close_if_complete(plan, out):
    out = Path(out)
    units = plan["units"]
    known = {u["unit_id"] for u in units}
    present = {p.name for p in (out / "units").iterdir()} if (out / "units").exists() else set()
    require(present <= known, "output contains an unplanned unit directory")
    done, environments = {}, {}
    for unit in units:
        if verify_completed(unit, out, plan["plan_sha256"]):
            recorded_environment = read_json(out / "units" / unit["unit_id"] / "unit.json")["environment"]
            require(unit["model"] not in environments or environments[unit["model"]] == recorded_environment,
                    "completed units mix runtime environments within a model")
            environments[unit["model"]] = recorded_environment
            done[unit["unit_id"]] = file_sha(out / "units" / unit["unit_id"] / "DONE")
    complete = len(units) == EXPECTED_UNITS and len(done) == EXPECTED_UNITS
    if not complete:
        require(not (out / "analysis_lock.json").exists() and not (out / "completion.json").exists(),
                "closure exists for an incomplete or corrupted core")
        return False
    completion = {"schema": "ser-study2-completion-1", "program": PROGRAM,
                  "plan_sha256": plan["plan_sha256"], "n_units": EXPECTED_UNITS,
                  "units_done": EXPECTED_UNITS, "done_sha256": done}
    completion_path = out / "completion.json"
    if completion_path.exists():
        require(read_json(completion_path) == completion, "existing completion differs")
    else:
        atomic_json(completion_path, completion)
    lock = {"schema": "ser-study2-analysis-lock-1", "program": PROGRAM,
            "plan_sha256": plan["plan_sha256"], "n_units": EXPECTED_UNITS,
            "done_sha256": done, "completion_sha256": file_sha(completion_path)}
    lock_path = out / "analysis_lock.json"
    if lock_path.exists():
        existing = read_json(lock_path)
        require({k: v for k, v in existing.items() if k != "created_at"} == lock,
                "existing analysis lock differs")
    else:
        atomic_json(lock_path, {**lock, "created_at": now()})
        append_event(out, "analysis_locked", n_units=EXPECTED_UNITS, plan_sha256=plan["plan_sha256"])
    return True


def attempt_state(out, known_units):
    """Count real starts, including interrupted fits; terminal events do not add attempts."""
    path = Path(out) / "ledger.jsonl"
    counts, last = Counter(), {}
    if not path.exists():
        return counts, last
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("event")
            if kind not in {"unit_start", "unit_failed", "unit_done"}:
                continue
            uid = event.get("unit_id")
            require(uid in known_units, "attempt ledger contains an unplanned unit")
            if kind == "unit_start":
                require(last.get(uid) != "unit_start", "unreconciled overlapping unit attempts")
                counts[uid] += 1
                require(counts[uid] <= 2, "attempt ledger exceeds the two-attempt ceiling")
            require(event.get("attempt") == counts[uid] and counts[uid] > 0,
                    "attempt ledger numbering mismatch")
            last[uid] = kind
    return counts, last


def reconcile_attempts(plan, out):
    by_id = {u["unit_id"]: u for u in plan["units"]}
    counts, last = attempt_state(out, by_id)
    for uid, event in last.items():
        if event == "unit_start":
            if verify_completed(by_id[uid], out, plan["plan_sha256"]):
                append_event(out, "unit_done", unit_id=uid, attempt=counts[uid],
                             recovered_from_done=True)
            else:
                append_event(out, "unit_failed", unit_id=uid, attempt=counts[uid],
                             error_type="InterruptedAttempt", orphan=True)
    return counts


def run(repo, plan_path, feature_root, out, model="ridge_wavlm", device="cpu",
        max_units=None, threads=2, gpu_release=None):
    repo, out, plan_path = Path(repo).resolve(), Path(out).resolve(), Path(plan_path).resolve()
    require(repo == Path(__file__).resolve().parents[2], "--repo must be this runner's checkout")
    require(model in {"ridge_wavlm", "cnn"}, "unknown requested model")
    require((model, device) in {("ridge_wavlm", "cpu"), ("cnn", "cuda")},
            "Ridge runs on CPU; CNN requires explicit CUDA and a handoff")
    require(type(threads) is int and 1 <= threads <= 16, "threads must be between 1 and 16")
    require(max_units is None or type(max_units) is int and max_units > 0,
            "max-units must be a positive integer")
    require(out != repo and not any(out.is_relative_to(repo / p) for p in ("v2", "v2_1")),
            "outputs cannot overwrite source or legacy experiment directories")
    require(not (out / ".n14r-run.lock").exists(), "refusing the N14R output directory")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    cpu_restrictions = restrict_cpu_process() if device == "cpu" else None
    from threadpoolctl import threadpool_limits
    plan = read_json(plan_path)
    release = validate_gpu_release(gpu_release, plan_sha256=plan.get("plan_sha256")) if model == "cnn" else None
    require(plan.get("design", {}).get("runtime_complete") is True,
            "draft plan: runtime_complete must be true before training")
    with threadpool_limits(limits=threads), cpu_execution_guard(device):
        rows, features = load_inputs(repo, plan, Path(feature_root).resolve())
    guarded = [plan_path, relative_file(repo, plan["input"]["manifest_path"]),
               relative_file(repo, plan["input"]["demographics_path"])]
    guarded.extend(relative_file(repo, name) for name in plan["source_sha256"])
    if release:
        guarded.append(Path(release["path"]))
    guard = FileGuard(guarded)
    if model == "cnn":
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    runtime = environment(device, threads)
    if cpu_restrictions:
        runtime["cpu_restrictions"] = cpu_restrictions
    if release:
        runtime["gpu_release"] = release
    n_done, n_skipped = 0, 0
    with runner_lock(out), threadpool_limits(limits=threads), cpu_execution_guard(device):
        identity_path = out / "run_identity.json"
        identity = {"schema": "ser-study2-run-1", "program": PROGRAM,
                    "plan_sha256": plan["plan_sha256"]}
        if identity_path.exists():
            require(read_json(identity_path) == identity, "output belongs to a different plan")
        else:
            require(not (out / "units").exists(), "unidentified existing output units")
            atomic_json(identity_path, identity)
        environment_path = out / f"environment_{model}.json"
        if environment_path.exists():
            require(read_json(environment_path) == runtime,
                    "runtime environment differs for this model; use a separate run")
        else:
            atomic_json(environment_path, runtime)
        attempts = reconcile_attempts(plan, out)
        append_event(out, "runner_start", model=model, device=device, threads=threads,
                     plan_sha256=plan["plan_sha256"])
        for unit in plan["units"]:
            if unit["model"] != model:
                continue
            if verify_completed(unit, out, plan["plan_sha256"]):
                receipt = read_json(out / "units" / unit["unit_id"] / "unit.json")
                require(receipt["environment"] == runtime,
                        "completed unit runtime differs from this model's environment")
                n_skipped += 1
                continue
            if max_units is not None and n_done >= max_units:
                break
            require(attempts[unit["unit_id"]] < 2,
                    "unit exhausted two attempts; core remains incomplete, no third fit is allowed")
            if release and release.get("scope") == "authorized-independent-host":
                require(datetime.fromisoformat(release["expires_at"]) > datetime.now(timezone.utc),
                        "independent-host authorization expired before the next fit")
            guard.check()
            features.unchanged()
            directory = out / "units" / unit["unit_id"]
            if directory.exists():
                if (directory / "unit.json").exists():
                    old = read_json(directory / "unit.json")
                    require(old.get("unit_id") == unit["unit_id"]
                            and old.get("plan_sha256") == plan["plan_sha256"],
                            "partial unit belongs to another identity")
                append_event(out, "partial_unit_retry", unit_id=unit["unit_id"])
            attempts[unit["unit_id"]] += 1
            attempt = attempts[unit["unit_id"]]
            append_event(out, "unit_start", unit_id=unit["unit_id"], model=model, attempt=attempt)
            started = time.perf_counter()
            try:
                logits, timing = fit_unit(unit, rows, features, device)
                timing["wall_seconds"] = time.perf_counter() - started
                timing["attempt"] = attempt
                guard.check()
                features.unchanged()
                if device == "cpu":
                    assert_cpu_has_no_cuda()
                write_completed(unit, out, plan["plan_sha256"], logits, timing, runtime)
            except BaseException as exc:
                append_event(out, "unit_failed", unit_id=unit["unit_id"], attempt=attempt,
                             error_type=type(exc).__name__)
                raise
            n_done += 1
            append_event(out, "unit_done", unit_id=unit["unit_id"], model=model,
                         wall_seconds=timing["wall_seconds"], attempt=attempt)
            print(json.dumps({"event": "unit_done", "model": model, "newly_done": n_done,
                              "unit_id": unit["unit_id"], "wall_seconds": timing["wall_seconds"]}), flush=True)
        guard.check()
        features.unchanged()
        if device == "cpu":
            assert_cpu_has_no_cuda()
        complete = close_if_complete(plan, out)
        result = {"model": model, "newly_done": n_done, "skipped_valid_done": n_skipped,
                  "core_complete": complete, "plan_sha256": plan["plan_sha256"]}
        append_event(out, "runner_stop", **result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("ridge_wavlm", "cnn"), default="ridge_wavlm")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--gpu-release", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.repo, args.plan, args.features, args.out, args.model,
                         args.device, args.max_units, args.threads, args.gpu_release),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

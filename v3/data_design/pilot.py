"""Blind, CPU-only Ridge pilot: integrity and cost receipts, never efficacy scores."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import csv
import ctypes
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
import uuid

SCHEMA = "ser-study2-blind-pilot-1"
MODEL = {"name": "RidgeClassifier", "alpha": 1.0, "class_weight": "balanced",
         "solver": "lsqr", "tol": 1e-4, "class_order": list(range(6)), "threads": 2,
         "device": "cpu", "scaler": "StandardScaler_fit_only"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as f:
            if callable(data):
                data(f)
            else:
                f.write(canonical(data) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def relative_path(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value, "invalid relative path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == value, "noncanonical relative path")
    return value


def reject_test_keys(value):
    if isinstance(value, dict):
        require(not any(re.search(r"(^|_)test($|_)", k) for k in value), "test inputs forbidden in blind pilot")
        for item in value.values():
            reject_test_keys(item)
    elif isinstance(value, list):
        for item in value:
            reject_test_keys(item)


def restrict_process():
    torch = sys.modules.get("torch")
    require(torch is None or not torch.cuda.is_initialized(), "CUDA already initialized")
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        require(bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)), "cannot set BelowNormal priority")


def load_inputs(repo, plan_path, features):
    import numpy as np
    plan = read_json(plan_path)
    require(plan.get("schema") == SCHEMA, "wrong plan schema")
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "plan hash mismatch")
    reject_test_keys(plan)
    spec = plan["input"]
    require(spec["feature_kind"] == "wavlm_base_plus" and spec["feature_state"] == 12, "wrong feature specification")
    manifest = (Path(repo).resolve() / relative_path(spec["manifest_path"])).resolve()
    require(manifest.is_relative_to(Path(repo).resolve()), "manifest escapes repository")
    require(file_sha(manifest) == spec["manifest_sha256"], "manifest hash mismatch")
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(len(reader.fieldnames or []) == len(set(reader.fieldnames or [])), "duplicate manifest columns")
        require({"relative_path", "speaker", "sha256", "label_index", "corpus"} <= set(reader.fieldnames or []), "missing manifest columns")
        rows = list(reader)
    index = {}
    for row in rows:
        path = relative_path(row["relative_path"])
        require(path not in index and row["speaker"] and row["corpus"] == "cremad", "invalid manifest identity")
        require(re.fullmatch("[0-9a-f]{64}", row["sha256"]) and row["label_index"] in {str(i) for i in range(6)}, "invalid manifest class/hash")
        index[path] = row
    features = Path(features).resolve()
    require("SYNTHETIC" not in str(features).upper(), "synthetic cache forbidden")
    require(file_sha(features) == spec["feature_sha256"], "feature hash mismatch")
    meta_path = features.with_suffix(".json")
    metadata = read_json(meta_path)
    require("SYNTHETIC" not in canonical(metadata).decode().upper(), "synthetic metadata forbidden")
    require(metadata.get("encoder") == spec["feature_kind"] and metadata.get("corpus") == "cremad"
            and metadata.get("sha256") == spec["feature_sha256"], "feature metadata identity mismatch")
    with np.load(features, allow_pickle=False) as z:
        require(set(z.files) == {"X", "paths"}, "unexpected feature columns")
        X, paths = z["X"], z["paths"]
    require(paths.ndim == 1 and paths.dtype.kind == "U", "feature paths must be Unicode vector")
    names = paths.tolist()
    require(names and len(names) == len(set(names)) and set(names) <= set(index), "duplicate/unknown cache paths")
    require(X.shape == (len(names), 13, 768) and X.dtype.kind == "f" and np.isfinite(X).all(), "invalid feature shape/type/values")
    require(metadata.get("n") == len(names), "feature metadata row count mismatch")
    units = plan["units"]
    require(isinstance(units, list) and units, "empty units")
    unit_ids = set()
    for unit in units:
        uid = unit["unit_id"]
        require(isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,120}", uid) and uid not in unit_ids, "invalid/duplicate unit id")
        unit_ids.add(uid)
        require(isinstance(unit["train_seed"], int) and not isinstance(unit["train_seed"], bool) and 0 <= unit["train_seed"] < 2**32, "invalid training seed")
        fit, select = unit["fit"], unit["select"]
        require(isinstance(fit, list) and isinstance(select, list) and fit and select, "empty fit/select")
        for panel in (fit, select):
            require(len(panel) == len(set(panel)) and set(panel) <= set(names), "missing/duplicate panel paths")
        require(not set(fit) & set(select), "fit/select path leakage")
        require(not {index[p]["speaker"] for p in fit} & {index[p]["speaker"] for p in select}, "speaker leakage")
        hashes = [index[p]["sha256"] for p in fit + select]
        require(len(hashes) == len(set(hashes)), "byte duplicate within/across panels")
        require({index[p]["label_index"] for p in fit} == {str(i) for i in range(6)}, "fit missing fixed class")
        require(unit["config"]["B"] == len(fit) and unit["config"]["S"] == len({index[p]["speaker"] for p in fit}), "configuration quota mismatch")
    identity = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"], "input": spec,
                "feature_metadata_sha256": file_sha(meta_path), "runner_sha256": file_sha(__file__), "model": MODEL}
    return plan, index, np.asarray(X[:, 12, :], dtype=np.float32), {p: i for i, p in enumerate(names)}, identity


def verify_unit(folder, expected, select):
    import numpy as np
    receipt, done = read_json(folder / "unit.json"), read_json(folder / "DONE")
    require(receipt.get("identity") == expected, "unit identity mismatch")
    require(done == {"unit_sha256": expected["unit_sha256"], "receipt_sha256": file_sha(folder / "unit.json"),
                     "predictions_sha256": file_sha(folder / "predictions.npz")}, "DONE hash mismatch")
    require(receipt["predictions_sha256"] == done["predictions_sha256"], "prediction receipt mismatch")
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
        require(set(z.files) == {"paths", "logits"}, "unexpected prediction columns")
        require(z["paths"].tolist() == select and z["logits"].shape == (len(select), 6)
                and z["logits"].dtype.kind == "f" and np.isfinite(z["logits"]).all(), "invalid prediction identities/values")
    for key in ("unit_wall_seconds", "scaler_seconds", "fit_seconds", "predict_seconds"):
        require(isinstance(receipt[key], (float, int)) and 0 <= receipt[key] < float("inf"), "invalid timing receipt")
    return receipt


def run(repo, plan_path, features, out):
    started = time.perf_counter()
    restrict_process()
    import numpy as np
    import scipy
    import sklearn
    import threadpoolctl
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    from threadpoolctl import threadpool_limits
    plan, manifest, X, cache_index, identity = load_inputs(repo, plan_path, features)
    identity["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                           "sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / "pilot_identity.json"
    if identity_path.exists():
        require(read_json(identity_path) == identity, "output belongs to different plan/inputs/runner/runtime")
    else:
        require(not any(out.iterdir()), "unidentified nonempty output")
        atomic(identity_path, identity)
    prep_seconds = time.perf_counter() - started
    receipts, fitted = [], 0
    with threadpool_limits(limits=2):
        for unit in plan["units"]:
            expected = {"run_sha256": digest(identity), "unit_sha256": digest(unit)}
            folder = out / "units" / unit["unit_id"]
            if (folder / "DONE").exists():
                receipts.append(verify_unit(folder, expected, unit["select"]))
                continue
            begin = time.perf_counter()
            A = X[[cache_index[p] for p in unit["fit"]]]
            B = X[[cache_index[p] for p in unit["select"]]]
            y = np.asarray([int(manifest[p]["label_index"]) for p in unit["fit"]])
            tick = time.perf_counter()
            scaler = StandardScaler()
            A, B = scaler.fit_transform(A), scaler.transform(B)
            scaler_seconds = time.perf_counter() - tick
            model = RidgeClassifier(alpha=1.0, class_weight="balanced", solver="lsqr", tol=1e-4,
                                    random_state=unit["train_seed"])
            tick = time.perf_counter()
            model.fit(A, y)
            fit_seconds = time.perf_counter() - tick
            require(model.classes_.tolist() == MODEL["class_order"], "unexpected classifier column identity")
            tick = time.perf_counter()
            logits = model.decision_function(B)
            predict_seconds = time.perf_counter() - tick
            require(logits.shape == (len(B), 6) and np.isfinite(logits).all(), "invalid model output")
            atomic(folder / "predictions.npz", lambda f: np.savez(f, paths=np.asarray(unit["select"]), logits=logits))
            receipt = {"identity": expected, "predictions_sha256": file_sha(folder / "predictions.npz"),
                       "unit_wall_seconds": time.perf_counter() - begin, "scaler_seconds": scaler_seconds,
                       "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                       "fit_count": len(A), "select_count": len(B)}
            atomic(folder / "unit.json", receipt)
            atomic(folder / "DONE", {"unit_sha256": expected["unit_sha256"], "receipt_sha256": file_sha(folder / "unit.json"),
                                     "predictions_sha256": receipt["predictions_sha256"]})
            receipts.append(verify_unit(folder, expected, unit["select"]))
            fitted += 1
    completion = {"identity": identity, "completed_units": len(receipts), "fitted_this_invocation": fitted,
                  "reused_units": len(receipts) - fitted, "invocation_wall_seconds": time.perf_counter() - started,
                  "preparation_seconds": prep_seconds, "total_recorded_unit_wall_seconds": sum(r["unit_wall_seconds"] for r in receipts),
                  "total_recorded_fit_seconds": sum(r["fit_seconds"] for r in receipts)}
    atomic(out / "pilot_completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("repo", "plan", "features", "out"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    completion = run(args.repo, args.plan, args.features, args.out)
    print(json.dumps({k: v for k, v in completion.items() if k != "identity"}))


if __name__ == "__main__":
    main()

"""Create and verify the prospective N14R2 source/input/environment manifest.

The four environment receipts are outcome-free and must exist before the
preregistration commit.  Their exact bytes, GPU UUIDs, driver versions, and
draw-shard assignments are bound here.  Run, result, and evidence directories
are deliberately excluded so this module never inventories fitted outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "ser26-n14r2-pins-1"
BINDINGS_SCHEMA = "ser26-n14r2-environment-bindings-1"
ENVIRONMENT_SCHEMA = "ser26-n14r2-environment-1"
STUDY_ID = "N14R2"
BASE_COMMIT = "186b40d4a9db720d246f92dfef5ba73020903b9c"
BASE_TAG_COMMITS = {
    "SER26-prereg-1": "a9b33dbe11d222d82398579e584a239d6908466b",
    "SER26-prereg-2": "f473cddd6777e5641c77ca2e8268e77b243c5fc2",
}
REUSED_FILES = (
    "v2_1/__init__.py",
    "advanced_experiment_utils.py",
    "advanced_models.py",
    "dataset.py",
    "evaluate.py",
    "experiment_utils.py",
    "features.py",
    "fno_data.py",
    "fno_model.py",
    "tuned_standard_experiment.py",
    "v2/manifests/cremad_manifest.csv",
    "v2_1/n14r/__init__.py",
    "v2_1/n14r/run.py",
    "v2_1/n14r/spec.py",
    "v2/ser_v2/__init__.py",
    "v2/ser_v2/common.py",
    "v2/ser_v2/corpora.py",
    "v2/ser_v2/features.py",
    "v2/ser_v2/stats.py",
    "v2/ser_v2/train.py",
)
CORE_PACKAGES = {
    "torch": "2.11.0+cu128",
    "numpy": "2.4.6",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "librosa": "0.11.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to replace immutable receipt: {path}")


def _validate_environment_receipt(record: Any, node_id: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{node_id}: environment receipt is not an object")
    if (record.get("schema") != ENVIRONMENT_SCHEMA or record.get("study_id") != STUDY_ID
            or record.get("node_id") != node_id or record.get("shard_modulus") != 4
            or record.get("shard_remainder") != int(node_id[-1])):
        raise ValueError(f"{node_id}: environment identity/shard mismatch")
    contract = record.get("contract")
    if not isinstance(contract, dict) or record.get("contract_sha256") != sha256_json(contract):
        raise ValueError(f"{node_id}: environment contract hash mismatch")
    packages = contract.get("packages")
    if not isinstance(packages, dict):
        raise ValueError(f"{node_id}: missing complete package inventory")
    for name, version in CORE_PACKAGES.items():
        if packages.get(name) != version:
            raise ValueError(f"{node_id}: package {name} is {packages.get(name)!r}, expected {version!r}")
    gpu = contract.get("gpu")
    health = contract.get("cuda_health")
    if (contract.get("python") != "3.12.3" or contract.get("system") != "Linux"
            or contract.get("torch_cuda") != "12.8" or contract.get("cudnn") != 91900
            or contract.get("cpu_threads_per_worker") != 1
            or not isinstance(gpu, dict) or gpu.get("name") != "NVIDIA GeForce RTX 4090"
            or not isinstance(gpu.get("uuid"), str) or not gpu["uuid"]
            or not isinstance(gpu.get("driver_version"), str) or not gpu["driver_version"]
            or int(gpu.get("memory_total_mib", 0)) <= 0
            or not isinstance(health, dict) or health.get("device") != "cuda:0"
            or float(health.get("checksum", -1)) != 4.0
            or float(contract.get("cuda_health_checksum", -1)) != 4.0):
        raise ValueError(f"{node_id}: frozen runtime/GPU/CUDA-health contract mismatch")
    if not isinstance(record.get("recorded_at"), str) or not record["recorded_at"]:
        raise ValueError(f"{node_id}: missing prospective receipt timestamp")
    return contract


def build_environment_bindings(repo: Path, environment_dir: Path) -> dict[str, Any]:
    repo, environment_dir = repo.resolve(), environment_dir.resolve()
    nodes: dict[str, Any] = {}
    uuids: set[str] = set()
    for remainder in range(4):
        node_id = f"node{remainder}"
        receipt_path = environment_dir / f"{node_id}.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing prospective environment receipt: {receipt_path}")
        record = _read_json(receipt_path)
        contract = _validate_environment_receipt(record, node_id)
        uuid = contract["gpu"]["uuid"]
        if uuid in uuids:
            raise ValueError(f"GPU UUID is reused across nodes: {uuid}")
        uuids.add(uuid)
        try:
            receipt_path.relative_to(repo)
        except ValueError as exc:
            raise ValueError("environment receipts must live inside the frozen repository") from exc
        nodes[node_id] = {
            "shard_remainder": remainder,
            "environment_sha256": sha256_file(receipt_path),
            "gpu_uuid": uuid,
            "driver_version": contract["gpu"]["driver_version"],
        }
    payload: dict[str, Any] = {
        "schema": BINDINGS_SCHEMA,
        "study_id": STUDY_ID,
        "nodes": nodes,
    }
    return payload


def _tracked_study_files(repo: Path) -> list[Path]:
    study = repo / "v2_1" / "n14r2"
    excluded_parts = {
        "__pycache__", ".pytest_cache", "runs", "results", "evidence", "release_staging", "local",
    }
    out = []
    for path in study.rglob("*"):
        if not path.is_file() or path.name == "PINS.json":
            continue
        if any(part in excluded_parts for part in path.relative_to(study).parts):
            continue
        out.append(path)
    return sorted(out, key=lambda item: item.relative_to(repo).as_posix())


def build_record(repo: Path, features: Path, environment_dir: Path | None = None) -> dict[str, Any]:
    repo, features = repo.resolve(), features.resolve()
    study = repo / "v2_1" / "n14r2"
    environment_dir = (environment_dir or study / "environment").resolve()
    bindings_path = study / "ENVIRONMENT_BINDINGS.json"
    expected_bindings = build_environment_bindings(repo, environment_dir)
    if not bindings_path.is_file() or _read_json(bindings_path) != expected_bindings:
        raise ValueError("ENVIRONMENT_BINDINGS.json is missing or differs from the four receipts")
    paths = _tracked_study_files(repo) + [repo / name for name in REUSED_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot pin missing files: {missing}")
    files = {
        path.relative_to(repo).as_posix(): sha256_file(path)
        for path in sorted(set(paths), key=lambda item: item.relative_to(repo).as_posix())
    }
    feature_paths = sorted(features.glob("cremad__logmel__*.npz"))
    if len(feature_paths) != 1:
        raise RuntimeError(f"expected exactly one CREMA-D log-mel cache, found {len(feature_paths)}")
    feature_paths += [feature_paths[0].with_suffix(".json")]
    if not all(path.is_file() for path in feature_paths):
        raise FileNotFoundError("CREMA-D log-mel metadata is missing")
    external = [
        {"name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in feature_paths
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "study_id": STUDY_ID,
        "base_commit": BASE_COMMIT,
        "base_tag_commits": BASE_TAG_COMMITS,
        "files": files,
        "external_feature_cache": external,
        "environment_bindings": expected_bindings,
        "environment_bindings_sha256": sha256_file(bindings_path),
        "exclusions": [
            "PINS.json (self)", "runs/**", "results/**", "evidence/**",
            "release_staging/**", "local/**", "audio", "old N14/N14R fitted outputs", "__pycache__",
        ],
    }
    payload["content_manifest_sha256"] = sha256_json(payload)
    return payload


def verify_record(
    record: Any, repo: Path, features: Path, environment_dir: Path | None = None,
) -> list[str]:
    if not isinstance(record, dict):
        return ["record_not_object"]
    try:
        expected = build_record(repo.resolve(), features.resolve(), environment_dir)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return [f"inventory_rebuild:{type(exc).__name__}:{exc}"]
    return [key for key in sorted(set(record) | set(expected)) if record.get(key) != expected.get(key)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--bindings-out", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("PINS.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    study = args.repo.resolve() / "v2_1" / "n14r2"
    environment_dir = (args.environment_dir or study / "environment").resolve()
    bindings_out = (args.bindings_out or study / "ENVIRONMENT_BINDINGS.json").resolve()
    if args.verify:
        failures = verify_record(_read_json(args.out), args.repo, args.features, environment_dir)
        print(json.dumps({"pass": not failures, "failures": failures}, indent=1))
        return 0 if not failures else 1
    bindings = build_environment_bindings(args.repo, environment_dir)
    _write_json_exclusive(bindings_out, bindings)
    record = build_record(args.repo, args.features, environment_dir)
    _write_json_exclusive(args.out, record)
    print(json.dumps({
        "out": str(args.out), "n_files": len(record["files"]),
        "n_external": len(record["external_feature_cache"]),
        "n_environments": len(record["environment_bindings"]["nodes"]),
        "content_manifest_sha256": record["content_manifest_sha256"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

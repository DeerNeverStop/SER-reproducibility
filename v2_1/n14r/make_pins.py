"""Create or verify the N14R pre-execution file manifest.

The manifest intentionally excludes itself and post-freeze run, result,
evidence, and release-staging outputs. It includes every N14R source/contract
file, every generated split, all local Python modules imported by the reused v2
training engine, and the external CREMA-D log-mel cache hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "ser26-n14r-pins-1"
BASE_COMMIT = "c0c0bebfb1509b9606e73b806f5b303fe4c26546"
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
    "v2/ser_v2/__init__.py",
    "v2/ser_v2/common.py",
    "v2/ser_v2/corpora.py",
    "v2/ser_v2/features.py",
    "v2/ser_v2/stats.py",
    "v2/ser_v2/train.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _tracked_study_files(repo: Path) -> list[Path]:
    study = repo / "v2_1" / "n14r"
    excluded_parts = {
        "__pycache__", ".pytest_cache", "runs", "results", "evidence", "release_staging",
    }
    out = []
    for path in study.rglob("*"):
        if not path.is_file() or path.name == "PINS.json":
            continue
        if any(part in excluded_parts for part in path.relative_to(study).parts):
            continue
        out.append(path)
    return sorted(out, key=lambda path: path.relative_to(repo).as_posix())


def build_record(repo: Path, features: Path) -> dict[str, Any]:
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
    payload = {
        "schema": SCHEMA,
        "base_commit": BASE_COMMIT,
        "base_tag_commits": BASE_TAG_COMMITS,
        "files": files,
        "external_feature_cache": external,
        "exclusions": [
            "PINS.json (self)", "runs/**", "results/**", "evidence/**",
            "release_staging/**", "audio", "__pycache__",
        ],
    }
    payload["content_manifest_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def verify_record(record: dict[str, Any], repo: Path, features: Path) -> list[str]:
    """Compare a receipt with a fresh complete inventory, not just listed items.

    Rebuilding is intentional: otherwise deleting a file entry and recomputing
    the self-hash, or adding a second cache that the legacy loader prefers,
    could turn an incomplete receipt into an apparently valid one.
    """
    failures: list[str] = []
    if not isinstance(record, dict):
        return ["record_not_object"]
    try:
        expected = build_record(repo.resolve(), features.resolve())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return [f"inventory_rebuild:{type(exc).__name__}:{exc}"]
    for key in sorted(set(record) | set(expected)):
        if record.get(key) != expected.get(key):
            failures.append(key)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("PINS.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        failures = verify_record(json.loads(args.out.read_text(encoding="utf-8")), args.repo, args.features)
        print(json.dumps({"pass": not failures, "failures": failures}, indent=1))
        return 0 if not failures else 1
    record = build_record(args.repo, args.features)
    # PINS is generated evidence, not hand-authored source.  Atomic replacement
    # avoids leaving a plausible partial manifest after interruption.
    from v2.ser_v2.common import atomic_write_json

    atomic_write_json(args.out, record)
    print(json.dumps({
        "out": str(args.out), "n_files": len(record["files"]),
        "n_external": len(record["external_feature_cache"]),
        "content_manifest_sha256": record["content_manifest_sha256"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

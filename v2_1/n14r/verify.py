"""Fail-closed, independent verifier for the preregistered N14R rerun.

This file deliberately does not import :mod:`v2_1.n14r.score`, ``plan`` or
``stats``.  The verifier treats the byte-frozen spec, plan, split tables,
analysis lock, execution ledger and raw trainer outputs as its only evidence.
It reconstructs every outcome used by the scorer and compares the scorer JSON
recursively (including draw values, p values and the verdict).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats as scipy_stats


RESULT_SCHEMA = "ser-v2.1-n14r-results-1"
VERIFY_SCHEMA = "ser-v2.1-n14r-verification-1"
STUDY_ID = "N14R"
SPEC_VERSION = "1.0.0"
SEED_NAMESPACE = "SER26-N14R"
CELLS = ("GR_hpo", "GG_hpo")
N_CLASSES = 6
N_FOLDS = 5
N_DRAWS = 28
N_ANALYSIS_DRAWS = 24
CONFIGS = tuple(range(8))
FIXED_CONFIG = 4
ALPHA = 0.05
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 202_609_040_001
FLOAT_ATOL = 1e-10
H3_UNUSABLE_BASENAMES = frozenset({"1076_MTI_SAD_XX.wav"})
BASE_COMMIT = "c0c0bebfb1509b9606e73b806f5b303fe4c26546"
BASE_TAG_COMMITS = {
    "SER26-prereg-1": "a9b33dbe11d222d82398579e584a239d6908466b",
    "SER26-prereg-2": "f473cddd6777e5641c77ca2e8268e77b243c5fc2",
}
REUSED_PINNED_FILES = (
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
PIN_EXCLUDED_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", "runs", "results", "evidence", "release_staging"}
)
PIN_EXCLUSIONS = [
    "PINS.json (self)", "runs/**", "results/**", "evidence/**",
    "release_staging/**", "audio", "__pycache__",
]

PLAN_FIELDS = (
    "study_id", "spec_version", "unit_id", "arm", "manifest_sha256", "draw_id",
    "draw_role", "analysis_order", "fold", "train_rep", "cell", "config_index",
    "model", "corpus_level", "base_corpus", "panel_draw", "r", "seed_index",
    "split_seed", "inner_seed", "train_seed", "outer_sha256", "inner_sha256",
    "split_sha256", "config_sha256", "n_fit", "n_val", "n_test", "est_gpu_sec",
    "cap_group", "conditional", "truncation_rank", "status",
)
INT_FIELDS = {
    "draw_id", "analysis_order", "fold", "train_rep", "config_index", "r",
    "seed_index", "split_seed", "inner_seed", "train_seed", "n_fit", "n_val",
    "n_test", "conditional", "truncation_rank",
}
UNIT_ID_EXCLUDED = {"unit_id", "est_gpu_sec", "status"}
HPO_GRID = tuple(
    {
        "engine": "p1_frozen", "model": "resnet_se", "lr": lr,
        "weight_decay": wd, "dropout": dropout, "batch_size": 64,
        "epochs": 100, "patience": 15, "hpo_config_index": i,
    }
    for i, (lr, wd, dropout) in enumerate(
        (lr, wd, dropout)
        for lr in (3e-4, 1e-3)
        for wd in (1e-4, 1e-3)
        for dropout in (0.1, 0.3)
    )
)


class VerificationError(ValueError):
    """Evidence is incomplete, inconsistent, or not independently replayable."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _u32_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def expected_seed(kind: str, draw: int, fold: int | None = None, cell: str | None = None) -> int:
    if kind == "outer":
        key = f"{SEED_NAMESPACE}|outer|cremad|draw={draw}"
    elif kind == "inner":
        if fold is None or cell not in CELLS:
            raise VerificationError("inner seed requires a fold and frozen cell")
        rule = "random" if cell == "GR_hpo" else "grouped"
        key = f"{SEED_NAMESPACE}|inner|cremad|draw={draw}|fold={fold}|rule={rule}"
    elif kind == "train":
        if fold is None:
            raise VerificationError("train seed requires a fold")
        key = f"{SEED_NAMESPACE}|train|cremad|draw={draw}|fold={fold}|rep=0"
    else:
        raise VerificationError(f"unknown seed kind {kind!r}")
    return _u32_seed(key)


def _typed(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in PLAN_FIELDS:
        value: Any = row[key]
        if key in INT_FIELDS:
            value = int(value)
        elif key == "est_gpu_sec":
            value = float(value)
        out[key] = value
    return out


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _expected_analysis_draw_ids(void_ids: set[int]) -> list[int]:
    return [draw for draw in range(N_DRAWS) if draw not in void_ids][:N_ANALYSIS_DRAWS]


def _validate_spec(spec_path: Path) -> dict[str, Any]:
    spec = _json(spec_path)
    _require(isinstance(spec, dict), "spec.json is not an object")
    design = spec.get("design", {})
    inference = spec.get("primary_inference", {})
    completion = spec.get("completion_and_stopping", {})
    checks = {
        "study_id": spec.get("study_id") == STUDY_ID,
        "spec_version": spec.get("spec_version") == SPEC_VERSION,
        "seed_namespace": spec.get("seed_namespace") == SEED_NAMESPACE,
        "primary IDs": design.get("primary_draw_ids") == list(range(24)),
        "reserve IDs": design.get("reserve_draw_ids") == list(range(24, 28)),
        "folds": design.get("folds_per_draw") == 5,
        "reps": design.get("train_reps") == 1,
        "cells": design.get("cells") == list(CELLS),
        "grid": design.get("config_indices") == list(range(8)),
        "units": design.get("units_per_draw") == 80,
        "manifest input": design.get("manifest_input_rows") == 7442,
        "hygienic population": design.get("analysis_population_rows_after_hygiene") == 7435,
        "draws": completion.get("required_complete_draws") == 24,
        "activated draws": completion.get("maximum_activated_draws") == 28,
        "attempts per unit": completion.get("maximum_attempts_per_unit") == 2,
        "maximum attempts": completion.get("theoretical_maximum_training_attempts") == 4480,
        "bootstrap": inference.get("bootstrap") == {
            "role": "sensitivity_only", "method": "percentile",
            "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
        },
        "directional boundary": inference.get("directional_success") ==
        "mean > 0 and Holm-adjusted two-sided p <= 0.05",
        "pilot excluded": spec.get("legacy_pilot", {}).get("excluded_from_confirmatory_analysis") is True,
        "execution environment": spec.get("execution_environment") == {
            "python": "3.13.9", "numpy": "2.4.6", "scipy": "1.18.0",
            "scikit_learn": "1.9.0", "torch": "2.11.0+cu128",
            "torch_cuda": "12.8", "cudnn": 91900, "librosa": "0.11.0",
            "gpu": "NVIDIA GeForce RTX 5070",
        },
        "no hard cap": spec.get("budget", {}).get("hard_runtime_cap") is None,
    }
    bad = [name for name, ok in checks.items() if not ok]
    _require(not bad, "frozen spec invariant changed: " + ", ".join(bad))
    return spec


def _apply_cremad_hygiene(source: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently reproduce frozen H1/H2/H3, then sort and re-index.

    H1 drops every member of an exact-byte duplicate group when its frozen
    labels conflict.  H2 keeps only the lexically first path when all labels
    agree.  H3 drops the preregistered unusable CREMA-D recording.
    """
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        digest = str(row.get("sha256", ""))
        _require(len(digest) == 64, "manifest contains an invalid audio SHA-256")
        by_sha[digest].append(row)
    drop: dict[str, str] = {}
    for digest, group in by_sha.items():
        if len(group) < 2:
            continue
        if len({int(row["label_index"]) for row in group}) > 1:
            for row in group:
                drop.setdefault(row["relative_path"],
                                f"H1 label-conflicting duplicate group {digest[:12]}")
        else:
            keep = min(row["relative_path"] for row in group)
            for row in group:
                if row["relative_path"] != keep:
                    drop.setdefault(row["relative_path"],
                                    f"H2 exact duplicate of lexically first path ({digest[:12]})")
    for row in source:
        if Path(row["relative_path"]).name in H3_UNUSABLE_BASENAMES:
            drop.setdefault(row["relative_path"], "H3 registered unusable file")
    kept = [dict(row) for row in source if row["relative_path"] not in drop]
    kept.sort(key=lambda row: row["relative_path"])
    for index, row in enumerate(kept):
        row["sample_index"] = index
    hygiene = {
        "corpus": "cremad", "n_input": len(source), "n_kept": len(kept),
        "n_dropped": len(drop),
        "dropped": [{"relative_path": path, "reason": reason}
                    for path, reason in sorted(drop.items())],
    }
    return kept, hygiene


def _load_manifest(
    plan_dir: Path, expected_sha: str,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    candidates: list[Path] = []
    for base in (plan_dir, *plan_dir.parents):
        candidates += [base / "manifest.csv", base / "manifests" / "cremad_manifest.csv",
                       base / "v2" / "manifests" / "cremad_manifest.csv"]
    for path in dict.fromkeys(candidates):
        if path.is_file() and sha256_file(path) == expected_sha:
            with path.open("r", encoding="utf-8", newline="") as handle:
                source = list(csv.DictReader(handle))
            required = {"sample_index", "relative_path", "speaker", "label_index", "sha256"}
            _require(bool(source) and required.issubset(source[0]), "manifest columns are incomplete")
            raw_rows = [{**r, "sample_index": int(r["sample_index"]),
                         "label_index": int(r["label_index"])} for r in source]
            _require(len({r["relative_path"] for r in raw_rows}) == len(raw_rows),
                     "manifest relative paths are not unique")
            rows, hygiene = _apply_cremad_hygiene(raw_rows)
            _require([r["sample_index"] for r in rows] == list(range(len(rows))),
                     "hygiene-filtered indices are not contiguous and zero based")
            _require({r["label_index"] for r in rows} == set(range(6)),
                     "hygiene-filtered manifest does not contain exactly labels 0..5")
            return rows, path, hygiene
    raise VerificationError("cannot locate the byte-identical frozen CREMA-D manifest")


def _validate_hygiene(
    plan_dir: Path, manifest_sha256: str, computed: dict[str, Any],
) -> str:
    path = plan_dir / "hygiene.json"
    _require(path.is_file(), "plan/hygiene.json is missing")
    digest = sha256_file(path)
    summary = _json(plan_dir / "plan_summary.json")
    _require(summary.get("hygiene_sha256") == digest,
             "plan_summary hygiene exact-byte hash mismatch")
    _require(summary.get("manifest_sha256") == manifest_sha256,
             "plan_summary raw-manifest hash mismatch")
    _require(summary.get("manifest_input_rows") == computed["n_input"] and
             summary.get("analysis_population_rows") == computed["n_kept"],
             "plan_summary hygiene population counts mismatch")
    expected = {"schema": "ser26-n14r-hygiene-1",
                "manifest_sha256": manifest_sha256, **computed}
    actual = _json(path)
    _require(actual == expected, "plan/hygiene.json differs from independent H1/H2/H3 replay")
    _require(actual["n_input"] == 7442 and actual["n_kept"] == 7435 and
             actual["n_dropped"] == 7 and len(actual["dropped"]) == 7,
             "CREMA-D hygiene must remove exactly the seven frozen items")
    return digest


def _load_plan(plan_dir: Path) -> list[dict[str, Any]]:
    path = plan_dir / "run_plan.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == PLAN_FIELDS, "run_plan.csv header changed")
        rows = [_typed(r) for r in reader]
    _require(len(rows) == 2240, "plan must have exactly 2240 rows")
    _require(len({r["unit_id"] for r in rows}) == 2240, "plan unit_id values are not unique")
    return rows


def _ref(path: str, by_path: dict[str, dict[str, Any]]) -> dict[str, Any]:
    _require(path in by_path, f"split path absent from manifest: {path}")
    row = by_path[path]
    return {"sample_index": row["sample_index"], "relative_path": path,
            "label_index": row["label_index"], "speaker": row["speaker"]}


def _refs(paths: list[str], by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    _require(isinstance(paths, list) and paths and len(paths) == len(set(paths)),
             "split partition is empty or contains duplicate paths")
    return [_ref(str(p), by_path) for p in paths]


def _validate_plan_and_splits(
    plan_dir: Path, rows: list[dict[str, Any]], manifest: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    by_path = {r["relative_path"]: r for r in manifest}
    population = set(by_path)
    index = _json(plan_dir / "split_index.json")
    configs = _json(plan_dir / "unit_configs.json")
    summary = _json(plan_dir / "plan_summary.json")
    _require(summary.get("study_id") == STUDY_ID and summary.get("spec_version") == SPEC_VERSION and
             summary.get("maximum_units") == 2240 and summary.get("primary_units") == 1920,
             "plan_summary frozen counts/identity mismatch")
    _require(summary.get("run_plan_sha256") == sha256_file(plan_dir / "run_plan.csv") and
             summary.get("split_index_sha256") == sha256_file(plan_dir / "split_index.json"),
             "plan_summary byte hashes mismatch")
    _require(isinstance(index, dict) and len(index) == 56, "split index must name 56 cell tables")
    expected_configs = {sha256_json(cfg): cfg for cfg in HPO_GRID}
    _require(configs == expected_configs, "unit_configs.json differs from the frozen 8-config grid")
    scopes: dict[tuple[int, int, str], dict[str, Any]] = {}
    seen_draw_outer_hashes: set[str] = set()
    for draw in range(N_DRAWS):
        for cell in CELLS:
            key = f"n14r__d{draw:02d}__{cell}"
            entry = index.get(key)
            _require(isinstance(entry, dict), f"missing split index entry {key}")
            path = plan_dir / str(entry.get("path", ""))
            _require(path.is_file(), f"missing split file {path}")
            raw_sha = sha256_file(path)
            _require(raw_sha == entry.get("sha256"), f"split index hash mismatch: {key}")
            doc = _json(path)
            _require(sha256_json(doc) == raw_sha, f"split is not canonical JSON: {key}")
            meta = doc.get("meta", {})
            _require(meta.get("study_id") == STUDY_ID and int(meta.get("draw_id", -1)) == draw,
                     f"split metadata identity mismatch: {key}")
            _require(meta.get("manifest_sha256") == rows[0]["manifest_sha256"],
                     f"split manifest metadata mismatch: {key}")
            _require(int(meta.get("split_seed", -1)) == expected_seed("outer", draw),
                     f"split seed mismatch: {key}")
            _require(set(doc.get("population", [])) == population and
                     len(doc.get("population", [])) == len(population),
                     f"split population mismatch: {key}")
            folds = doc.get("folds")
            _require(isinstance(folds, list) and len(folds) == 5, f"{key} lacks five folds")
            draw_outer_pairs: list[dict[str, Any]] = []
            all_test_paths: list[str] = []
            for fold, fd in enumerate(folds):
                _require(int(fd.get("fold", -1)) == fold, f"fold order mismatch: {key}")
                fm = fd.get("meta", {})
                fit, val, test = (_refs(fd[name], by_path) for name in ("fit", "val", "test"))
                fit_paths = {x["relative_path"] for x in fit}
                val_paths = {x["relative_path"] for x in val}
                test_paths = {x["relative_path"] for x in test}
                all_test_paths.extend(x["relative_path"] for x in test)
                _require(not (fit_paths & val_paths or fit_paths & test_paths or val_paths & test_paths),
                         f"overlapping partitions: {key} fold {fold}")
                _require(fit_paths | val_paths | test_paths == population,
                         f"partitions do not cover manifest: {key} fold {fold}")
                for name, refs in (("fit", fit), ("val", val), ("test", test)):
                    _require({x["label_index"] for x in refs} == set(range(6)),
                             f"{name} lacks six-class support: {key} fold {fold}")
                outer_train = sorted([*fit, *val], key=lambda x: x["sample_index"])
                test_sorted = sorted(test, key=lambda x: x["sample_index"])
                draw_outer_pairs.append({"train": [x["sample_index"] for x in outer_train],
                                         "test": [x["sample_index"] for x in test_sorted]})
                _require(not ({x["speaker"] for x in outer_train} & {x["speaker"] for x in test}),
                         f"outer speaker leakage: {key} fold {fold}")
                if cell == "GG_hpo":
                    _require(not ({x["speaker"] for x in fit} & {x["speaker"] for x in val}),
                             f"GG inner speaker leakage: {key} fold {fold}")
                outer_sha = sha256_json({"outer_train": outer_train, "test": test_sorted})
                inner_sha = sha256_json({"fit": sorted(fit, key=lambda x: x["sample_index"]),
                                         "val": sorted(val, key=lambda x: x["sample_index"])})
                _require(fm.get("outer_sha256") == outer_sha and fm.get("inner_sha256") == inner_sha,
                         f"partition hash mismatch: {key} fold {fold}")
                _require(int(fm.get("inner_seed", -1)) == expected_seed("inner", draw, fold, cell),
                         f"inner seed mismatch: {key} fold {fold}")
                scopes[(draw, fold, cell)] = {
                    "fit": fit, "val": val, "test": test, "outer_sha256": outer_sha,
                    "inner_sha256": inner_sha, "split_sha256": raw_sha,
                }
            _require(meta.get("draw_outer_sha256") == sha256_json(draw_outer_pairs),
                     f"draw-level outer partition hash mismatch: {key}")
            _require(len(all_test_paths) == len(population) and set(all_test_paths) == population,
                     f"outer tests do not partition the population exactly once: {key}")
            if cell == CELLS[0]:
                _require(meta["draw_outer_sha256"] not in seen_draw_outer_hashes,
                         f"duplicate outer draw detected: {draw}")
                seen_draw_outer_hashes.add(meta["draw_outer_sha256"])
    for row in rows:
        d, f, c, k = row["draw_id"], row["fold"], row["cell"], row["config_index"]
        _require(d in range(28) and f in range(5) and c in CELLS and k in CONFIGS,
                 f"invalid plan coordinate for {row['unit_id']}")
        _require(row["study_id"] == STUDY_ID and row["spec_version"] == SPEC_VERSION and
                 row["arm"] == STUDY_ID and row["model"] == "resnet_se" and
                 row["corpus_level"] == "cremad" and row["train_rep"] == 0,
                 f"frozen plan identity mismatch for {row['unit_id']}")
        _require(row["draw_role"] == ("primary" if d < 24 else "reserve") and
                 row["analysis_order"] == d, f"draw role/order mismatch for {row['unit_id']}")
        _require(row["split_seed"] == expected_seed("outer", d) and
                 row["inner_seed"] == expected_seed("inner", d, f, c) and
                 row["train_seed"] == expected_seed("train", d, f),
                 f"seed mismatch for {row['unit_id']}")
        scope = scopes[(d, f, c)]
        for key in ("outer_sha256", "inner_sha256", "split_sha256"):
            _require(row[key] == scope[key], f"{key} mismatch for {row['unit_id']}")
        _require((row["n_fit"], row["n_val"], row["n_test"]) ==
                 (len(scope["fit"]), len(scope["val"]), len(scope["test"])),
                 f"partition count mismatch for {row['unit_id']}")
        cfg = HPO_GRID[k]
        _require(row["config_sha256"] == sha256_json(cfg), f"config hash mismatch for {row['unit_id']}")
        scientific = {name: row[name] for name in PLAN_FIELDS if name not in UNIT_ID_EXCLUDED}
        scientific.update(seed_namespace=SEED_NAMESPACE, config=cfg)
        _require(row["unit_id"] == sha256_json(scientific), f"unit_id hash mismatch for {row['unit_id']}")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["draw_id"], row["fold"])].append(row)
    for key, items in grouped.items():
        _require(len(items) == 16, f"fold {key} is not a two-cell eight-config grid")
        _require({r["config_index"] for r in items if r["cell"] == "GR_hpo"} == set(CONFIGS) and
                 {r["config_index"] for r in items if r["cell"] == "GG_hpo"} == set(CONFIGS),
                 f"missing or extra config in fold {key}")
        _require(len({r["outer_sha256"] for r in items}) == 1 and
                 len({r["train_seed"] for r in items}) == 1,
                 f"outer pairing/train seed differs in fold {key}")
        _require([x["relative_path"] for x in scopes[(*key, "GR_hpo")]["test"]] ==
                 [x["relative_path"] for x in scopes[(*key, "GG_hpo")]["test"]],
                 f"GR/GG outer-test rows differ in fold {key}")
    return scopes


def _repo_root(start: Path) -> Path:
    # GitHub release/source archives intentionally omit .git.  PINS lives at
    # <root>/v2_1/n14r/PINS.json, so retain offline verification when that
    # structural root is unambiguous; the complete pinned inventory below is
    # still verified byte for byte.
    resolved = start.resolve()
    if len(resolved.parents) >= 3:
        candidate = resolved.parents[2]
        if resolved.parent == (candidate / "v2_1" / "n14r").resolve():
            return candidate
    for path in (start, *start.parents):
        if (path / ".git").exists():
            return path
    raise VerificationError("cannot locate the N14R release root for PINS verification")


def _portable_basename(value: str) -> str:
    """Extract a receipt basename independent of the verifier host OS."""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _validate_pins(pins_path: Path, lock: dict[str, Any], environment: dict[str, Any]) -> bool:
    pins = _json(pins_path)
    payload = {key: value for key, value in pins.items() if key != "content_manifest_sha256"}
    _require(pins.get("schema") == "ser26-n14r-pins-1" and
             pins.get("content_manifest_sha256") == sha256_json(payload),
             "PINS.json content-manifest hash mismatch")
    files = pins.get("files")
    _require(isinstance(files, dict) and files, "PINS.json lacks a non-empty files mapping")
    root = _repo_root(pins_path)
    _require(pins.get("base_commit") == BASE_COMMIT and
             pins.get("base_tag_commits") == BASE_TAG_COMMITS,
             "PINS.json base commit/tag ancestry changed")
    _require(pins.get("exclusions") == PIN_EXCLUSIONS,
             "PINS.json exclusions changed")
    study = root / "v2_1" / "n14r"
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in study.rglob("*")
        if path.is_file() and path.name != "PINS.json" and
        not any(part in PIN_EXCLUDED_PARTS for part in path.relative_to(study).parts)
    }
    expected_paths.update(REUSED_PINNED_FILES)
    _require(set(files) == expected_paths,
             "PINS.json source inventory is incomplete or contains unexpected paths")
    for rel, digest in files.items():
        path = root / str(rel)
        _require(path.is_file(), f"pinned source is missing: {rel}")
        _require(sha256_file(path) == digest, f"pinned source hash mismatch: {rel}")
    # Feature cache paths are host-local.  Validate them only when the frozen
    # execution record or lock exposes a concrete path, never by basename guess.
    external = pins.get("external_feature_cache", {})
    runtime = lock.get("external_feature_cache", environment.get("external_feature_cache"))
    _require(isinstance(external, list) and len(external) == 2,
             "PINS.json must identify exactly two external feature-cache files")
    _require(runtime is not None, "execution lock lacks external feature-cache receipts")
    if runtime is not None:
        entries = runtime if isinstance(runtime, list) else [runtime]
        _require(len(entries) == 2, "execution record must lock exactly two feature-cache files")
        expected = external if isinstance(external, list) else [external]
        expected_by_name = {
            str(x.get("basename", x.get("name"))): x
            for x in expected if isinstance(x, dict) and x.get("basename", x.get("name"))
        }
        _require(len(expected_by_name) == len(expected) == 2,
                 "PINS.json external feature-cache basenames are not unique")
        runtime_items: list[tuple[dict[str, Any], Path, str]] = []
        for item in entries:
            _require(isinstance(item, dict) and item.get("path"), "invalid runtime feature-cache pin")
            path_text = str(item["path"])
            recorded_name = _portable_basename(path_text)
            _require(bool(recorded_name) and item.get("basename") == recorded_name,
                     "runtime feature-cache path/basename mismatch")
            runtime_items.append((item, Path(path_text), recorded_name))
        runtime_names = [name for _, _, name in runtime_items]
        _require(len(set(runtime_names)) == len(runtime_names) == 2 and
                 set(runtime_names) == set(expected_by_name),
                 "runtime feature-cache receipts are duplicated or incomplete")
        for item, path, recorded_name in runtime_items:
            pin = expected_by_name[recorded_name]
            _require(item.get("sha256") == pin.get("sha256") and
                     int(item.get("size_bytes", -1)) == int(pin.get("size_bytes", -2)),
                     f"external feature-cache receipt mismatch: {recorded_name}")
            if path.is_file():
                _require(sha256_file(path) == pin.get("sha256") and
                         path.stat().st_size == int(pin.get("size_bytes", -1)),
                         f"external feature-cache bytes mismatch: {path}")
        return all(path.is_file() for _, path, _ in runtime_items)
    return False


def _replay_attempt_ledger(
    events: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Independently replay every unit attempt as a strict two-state machine."""
    plan_ids = {row["unit_id"] for row in rows}
    replay: dict[str, dict[str, Any]] = {}
    for ordinal, event in enumerate(events, 1):
        uid = event.get("unit_id")
        kind = event.get("event")
        _require(uid in plan_ids, f"attempt event {ordinal} names an unknown unit")
        _require(kind in {"start", "done", "failed"},
                 f"attempt event {ordinal} has forbidden type {kind!r}")
        item = replay.setdefault(str(uid), {
            "state": "idle", "starts": 0, "done": 0, "failed": 0,
            "done_receipt": None,
        })
        if kind == "start":
            _require(item["state"] == "idle", f"out-of-sequence start for {uid}")
            item["starts"] += 1
            _require(item["starts"] <= 2, f"third training attempt for {uid}")
            item["state"] = "open"
        else:
            _require(item["state"] == "open", f"terminal event without open attempt for {uid}")
            item[kind] += 1
            if kind == "done":
                _require(item["done"] == 1, f"multiple successful attempts for {uid}")
                receipt = event.get("predictions_sha256")
                _require(isinstance(receipt, str) and len(receipt) == 64 and
                         all(char in "0123456789abcdef" for char in receipt),
                         f"invalid success receipt for {uid}")
                item["done_receipt"] = receipt
                item["state"] = "done"
            else:
                item["state"] = "idle"
    for uid, item in replay.items():
        _require(item["state"] != "open", f"unterminated training attempt for {uid}")
        _require(item["starts"] == item["done"] + item["failed"],
                 f"attempt/terminal counts disagree for {uid}")
        _require(item["failed"] <= 2, f"too many recorded failures for {uid}")
    return replay


def _validate_lock(plan_dir: Path, run_dir: Path, rows: list[dict[str, Any]], spec_path: Path) -> tuple[dict[str, Any], list[int], bool]:
    completion_path = run_dir / "completion.json"
    _require(completion_path.is_file(), "completion.json is required before outcomes are read")
    completion = _json(completion_path)
    _require(completion.get("schema") == "ser26-n14r-completion-1" and
             completion.get("study_id") == STUDY_ID and completion.get("status") == "complete",
             "completion record does not close a complete N14R run")
    lock_path = run_dir / "analysis_lock.json"
    _require(lock_path.is_file(), "analysis_lock.json is required before outcomes are read")
    lock = _json(lock_path)
    _require(lock.get("schema") == "ser26-n14r-analysis-lock-1" and lock.get("study_id") == STUDY_ID,
             "analysis lock schema/identity mismatch")
    _require(lock.get("contains_outcome_statistics") is False, "analysis lock contains outcome statistics")
    clone = dict(lock)
    self_hash = clone.pop("lock_payload_sha256", None)
    _require(self_hash == sha256_json(clone), "analysis lock payload hash mismatch")
    _require(completion.get("analysis_lock_payload_sha256") == self_hash,
             "completion record does not commit the analysis-lock payload")
    ids = lock.get("complete_draw_ids", lock.get("analysis_draw_ids"))
    _require(isinstance(ids, list) and len(ids) == 24 and len(set(ids)) == 24,
             "analysis lock must contain exactly 24 distinct draws")
    ids = [int(x) for x in ids]
    _require(completion.get("complete_draw_ids") == ids and completion.get("n_complete_draws") == 24,
             "completion and analysis lock disagree on complete draws")
    _require(ids == sorted(ids) and all(0 <= x < 28 for x in ids), "analysis draw order/IDs invalid")
    hashes = lock.get("plan_hashes")
    _require(isinstance(hashes, dict), "analysis lock lacks plan_hashes")
    for name in ("run_plan.csv", "split_index.json", "unit_configs.json", "hygiene.json", "plan_summary.json"):
        _require(hashes.get(name) == sha256_file(plan_dir / name), f"locked plan hash mismatch: {name}")
    prereg = lock.get("preregistration_hashes")
    pins_path = plan_dir.parent / "PINS.json"
    _require(isinstance(prereg, dict), "analysis lock lacks preregistration_hashes")
    _require(prereg.get("spec.json") == sha256_file(spec_path), "locked spec.json hash is mismatched")
    _require(pins_path.is_file() and prereg.get("PINS.json") == sha256_file(pins_path),
             "locked PINS.json hash is missing or mismatched")
    environment = run_dir / "execution_environment.json"
    _require(environment.is_file(), "execution_environment.json is missing")
    env = _json(environment)
    _require(lock.get("environment_fingerprint") == env.get("fingerprint_sha256"),
             "execution environment fingerprint mismatch")
    _require(completion.get("environment_fingerprint") == lock.get("environment_fingerprint"),
             "completion environment fingerprint mismatch")
    env_payload = {k: v for k, v in env.items() if k not in {"schema", "fingerprint_sha256"}}
    _require(env.get("fingerprint_sha256") == sha256_json(env_payload), "environment record hash mismatch")
    for name, digest in hashes.items():
        _require(env.get("plan_files", {}).get(name) == digest, f"environment plan hash mismatch: {name}")
    _require(env.get("preregistration_files") == prereg,
             "environment preregistration hashes differ from analysis lock")
    _require(lock.get("external_feature_cache") == env.get("external_feature_cache"),
             "locked external feature cache differs from execution environment")
    external_cache_bytes_verified = _validate_pins(pins_path, lock, env)
    ledger = run_dir / "ledger.jsonl"
    audit = run_dir / "n14r_execution_ledger.jsonl"
    _require(ledger.is_file() and sha256_file(ledger) == lock.get("execution_ledger_sha256"),
             "execution ledger hash mismatch")
    plan_unit_ids = {r["unit_id"] for r in rows}
    ledger_events: list[dict[str, Any]] = []
    with ledger.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"malformed execution ledger line {line_no}") from exc
            _require(isinstance(event, dict) and event.get("unit_id") in plan_unit_ids,
                     f"execution ledger line {line_no} has an unknown unit")
            _require(event.get("event") in {"start", "done", "failed"},
                     f"execution ledger line {line_no} has an unknown event")
            ledger_events.append(event)
    attempts = _replay_attempt_ledger(ledger_events, rows)
    audit_hash = lock.get("audit_ledger_prefix_sha256")
    _require(isinstance(audit_hash, str) and len(audit_hash) == 64,
             "analysis lock lacks the audit-ledger prefix hash")
    _require(audit.is_file(), "audit ledger is missing")
    raw = audit.read_bytes()
    lines = raw.splitlines(keepends=True)
    prefix = b""
    found = False
    before: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"malformed audit ledger line {line_no}") from exc
        _require(isinstance(event, dict) and isinstance(event.get("event"), str),
                 f"invalid audit ledger line {line_no}")
        if event.get("event") == "analysis_lock":
            _require(event.get("analysis_lock_sha256") == sha256_file(lock_path),
                     "audit ledger analysis-lock hash mismatch")
            found = True
            break
        before.append(event)
        prefix += line
    _require(found and sha256_bytes(prefix) == audit_hash,
             "audit ledger prefix hash mismatch")
    outcome_words = {"z", "d09r", "uar", "p", "p_value", "effect", "verdict", "mean", "ci"}
    _require(all(not (outcome_words & set(event)) for event in before),
             "audit ledger contains outcome statistics before analysis lock")
    raw_voids = completion.get("void_draws")
    _require(isinstance(raw_voids, dict), "completion void_draws must be an object")
    try:
        void_ids = {int(x) for x in raw_voids}
    except (TypeError, ValueError) as exc:
        raise VerificationError("completion contains an invalid void draw ID") from exc
    _require(all(0 <= draw < N_DRAWS for draw in void_ids),
             "completion contains an out-of-range void draw")
    audit_void_events = [event for event in before if event.get("event") == "draw_void"]
    try:
        audit_void = {int(event["draw_id"]) for event in audit_void_events}
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("audit ledger contains an invalid void draw ID") from exc
    _require(void_ids == audit_void, "completion void draws disagree with audit ledger")
    expected_ids = _expected_analysis_draw_ids(void_ids)
    _require(ids == expected_ids, "analysis lock is not the first 24 non-void draws")
    attempted_reserves = sorted({x for x in (*ids, *void_ids) if x >= 24})
    if attempted_reserves:
        _require(attempted_reserves == list(range(24, max(attempted_reserves) + 1)),
                 "reserve attempts are not ascending and contiguous")
    for reserve in attempted_reserves:
        activation = next((i for i, e in enumerate(before) if e.get("event") == "draw_activated" and
                           int(e.get("draw_id", -1)) == reserve), None)
        _require(activation is not None, f"reserve draw {reserve} lacks activation event")
        prior_void = {int(e["draw_id"]) for e in before[:activation] if e.get("event") == "draw_void"}
        _require(len(prior_void) >= reserve - 23, f"reserve draw {reserve} activated too early")

    row_by_id = {row["unit_id"]: row for row in rows}
    attempted_draws = set(ids) | void_ids
    _require(all(int(row_by_id[uid]["draw_id"]) in attempted_draws for uid in attempts),
             "execution ledger contains attempts from an unused draw")
    for draw_id in void_ids:
        exhausted_ids = sorted(uid for uid, item in attempts.items()
                               if int(row_by_id[uid]["draw_id"]) == draw_id and
                               item["failed"] == 2 and item["done"] == 0)
        _require(bool(exhausted_ids),
                 f"void draw {draw_id} lacks a unit with two closed failures")
        matching_events = [event for event in audit_void_events
                           if int(event["draw_id"]) == draw_id]
        _require(all(event.get("unit_ids") == exhausted_ids and
                     event.get("reason") == raw_voids[str(draw_id)]
                     for event in matching_events),
                 f"void draw {draw_id} audit receipt disagrees with exhausted units")
    expected_units = {r["unit_id"] for r in rows if r["draw_id"] in ids}
    artifacts = lock.get("unit_artifact_hashes")
    _require(isinstance(artifacts, dict) and set(artifacts) == expected_units,
             "locked unit artifact set is not exactly the 24-draw analysis set")
    _require(lock.get("n_locked_units") == 1920, "analysis lock does not contain 1920 units")
    for unit_id in expected_units:
        item = attempts.get(unit_id)
        _require(item is not None and item["state"] == "done" and
                 item["done"] == 1 and item["failed"] <= 1,
                 f"locked unit lacks one success or exceeded its retry allowance: {unit_id}")
        _require(item["starts"] == item["done"] + item["failed"] and item["starts"] <= 2,
                 f"locked unit attempt ledger is incomplete or exceeds two attempts: {unit_id}")
        _require(item["done_receipt"] == artifacts[unit_id]["predictions.csv"],
                 f"execution-ledger prediction receipt mismatch: {unit_id}")
    return lock, ids, external_cache_bytes_verified


def macro_uar_6(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    truth = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    _require(truth.ndim == pred.ndim == 1 and truth.size == pred.size and truth.size > 0,
             "invalid prediction vectors")
    _require(bool(np.all((truth >= 0) & (truth < 6) & (pred >= 0) & (pred < 6))),
             "prediction labels are outside 0..5")
    recalls = []
    for label in range(6):
        mask = truth == label
        _require(bool(mask.any()), f"prediction table lacks true class {label}")
        recalls.append(float(np.mean(pred[mask] == label)))
    return 100.0 * float(np.mean(recalls))


def _select_episode(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen eight-config selection rule (also useful to mutation tests)."""
    _require(len(candidates) == 8 and
             sorted(int(x["config_index"]) for x in candidates) == list(CONFIGS),
             "HPO episode must contain config indices 0..7 exactly once")
    _require(all(math.isfinite(float(x["val_uar"])) for x in candidates),
             "HPO validation UAR is non-finite")
    return min(candidates, key=lambda x: (-float(x["val_uar"]), int(x["config_index"])))


def _history_choice(path: Path) -> tuple[int, float]:
    rows = _json(path)
    _require(isinstance(rows, list) and rows, f"history is empty or not a list: {path}")
    best: tuple[float, int, float] | None = None
    epochs: set[int] = set()
    for ordinal, row in enumerate(rows):
        _require(isinstance(row, dict), f"history row is not an object: {path}")
        try:
            epoch = int(row.get("epoch", ordinal))
            loss, uar = float(row["val_loss"]), float(row["val_uar"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(f"history row lacks epoch/val_loss/val_uar: {path}") from exc
        _require(math.isfinite(loss) and math.isfinite(uar), f"non-finite history value: {path}")
        _require(epoch not in epochs, f"duplicate history epoch: {path}")
        epochs.add(epoch)
        if best is None or loss < best[0]:
            best = (loss, epoch, uar)
    assert best is not None
    return best[1], best[2]


def _predictions(path: Path, expected: list[dict[str, Any]]) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ({"sample_index", "relative_path", "speaker", "y_true", "y_pred"} |
                    {f"logit_{i}" for i in range(6)})
        _require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                 f"prediction header is incomplete: {path}")
        rows = list(reader)
    _require(len(rows) == len(expected), f"outer-test prediction row count mismatch: {path}")
    actual = {r["relative_path"]: r for r in rows}
    _require(len(actual) == len(rows) and set(actual) == {r["relative_path"] for r in expected},
             f"predictions are not an exact outer-test cover: {path}")
    truth: list[int] = []
    pred: list[int] = []
    for ref in expected:
        row = actual[ref["relative_path"]]
        try:
            _require(int(row["sample_index"]) == ref["sample_index"], f"sample index mismatch: {path}")
            _require(row["speaker"] == ref["speaker"], f"speaker mismatch: {path}")
            _require(int(row["y_true"]) == ref["label_index"], f"label mismatch: {path}")
            y_pred = int(row["y_pred"])
            logits = [float(row[f"logit_{i}"]) for i in range(6)]
        except (TypeError, ValueError) as exc:
            raise VerificationError(f"invalid prediction value: {path}") from exc
        _require(all(math.isfinite(x) for x in logits), f"non-finite logit: {path}")
        _require(int(np.argmax(np.asarray(logits))) == y_pred,
                 f"y_pred is not the lowest-index logit argmax: {path}")
        truth.append(ref["label_index"])
        pred.append(y_pred)
    return macro_uar_6(truth, pred)


def _unit_candidate(
    row: dict[str, Any], run_dir: Path, scope: dict[str, Any], locked: dict[str, Any],
) -> dict[str, Any]:
    unit_dir = run_dir / "units" / row["unit_id"]
    paths = {name: unit_dir / name for name in ("unit.json", "history.json", "predictions.csv", "DONE")}
    _require(all(p.is_file() for p in paths.values()), f"unit artifact missing: {row['unit_id']}")
    observed = {name: sha256_file(path) for name, path in paths.items()}
    _require(locked == observed, f"analysis-lock artifact hash mismatch: {row['unit_id']}")
    unit = _json(paths["unit.json"])
    _require(paths["DONE"].read_text(encoding="utf-8") == observed["predictions.csv"] + "\n",
             f"DONE receipt mismatch: {row['unit_id']}")
    for key in ("unit_id", "arm", "corpus_level", "model", "cell", "fold", "r",
                "seed_index", "train_seed", "config_sha256", "split_sha256", "n_fit", "n_val", "n_test"):
        expected = row[key]
        actual = unit.get(key)
        if isinstance(expected, int):
            try:
                actual = int(actual)
            except (TypeError, ValueError):
                pass
        _require(actual == expected, f"unit/plan mismatch for {key}: {row['unit_id']}")
    _require(unit.get("status") == "done" and unit.get("config") == HPO_GRID[row["config_index"]],
             f"unit status/config mismatch: {row['unit_id']}")
    _require(unit.get("predictions_sha256") == observed["predictions.csv"],
             f"unit prediction hash mismatch: {row['unit_id']}")
    if "history_sha256" in unit:
        _require(unit["history_sha256"] == observed["history.json"],
                 f"unit history hash mismatch: {row['unit_id']}")
    history_rows = _json(paths["history.json"])
    epoch, val_uar = _history_choice(paths["history.json"])
    best_row = next(x for x in history_rows if int(x.get("epoch", -1)) == epoch)
    _require(int(unit.get("best_epoch", -1)) == epoch, f"best_epoch mismatch: {row['unit_id']}")
    _require(math.isclose(float(unit.get("val_loss_best", math.nan)), float(best_row["val_loss"]),
                          rel_tol=0.0, abs_tol=FLOAT_ATOL),
             f"val_loss_best does not match minimum-loss history: {row['unit_id']}")
    try:
        reported_val = float(unit["val_uar_best"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"invalid val_uar_best: {row['unit_id']}") from exc
    _require(math.isclose(reported_val, val_uar, rel_tol=0.0, abs_tol=FLOAT_ATOL),
             f"val_uar_best does not match minimum-loss history: {row['unit_id']}")
    _require(0.0 <= val_uar <= 100.0, f"val_uar_best is outside 0..100: {row['unit_id']}")
    test_uar = _predictions(paths["predictions.csv"], scope["test"])
    return {"config_index": row["config_index"], "unit_id": row["unit_id"],
            "val_uar": val_uar, "test_uar": test_uar}


def one_sample_t(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    n, mean, sd = len(x), float(x.mean()), float(x.std(ddof=1))
    if sd == 0:
        se, t, p, ci = 0.0, (0.0 if mean == 0 else math.copysign(math.inf, mean)), (1.0 if mean == 0 else 0.0), [mean, mean]
    else:
        se = sd / math.sqrt(n)
        t = mean / se
        p = float(2 * scipy_stats.t.sf(abs(t), n - 1))
        q = float(scipy_stats.t.ppf(0.975, n - 1))
        ci = [mean - q * se, mean + q * se]
    supported = bool(mean > 0 and p <= ALPHA)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "df": n - 1, "t": float(t),
            "p_two_sided": p, "p_holm": p, "ci_level": .95, "ci": [float(x) for x in ci],
            "direction": ">0", "alpha": ALPHA, "supported": supported,
            "verdict": "supported" if supported else "not supported"}


def bootstrap(values: list[float], reps: int, seed: int) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], method="linear")
    return {"method": "draw bootstrap percentile", "seed": seed, "reps": reps,
            "mean": float(x.mean()), "ci_level": .95, "ci": [float(lo), float(hi)]}


def _sign_sums(x: np.ndarray) -> np.ndarray:
    sums = np.zeros(1)
    for value in x:
        sums = np.concatenate((sums + value, sums - value))
    return sums


def _studentized(sums: np.ndarray, square_sum: float, n: int) -> np.ndarray:
    v = (n * square_sum - sums * sums) / (n - 1)
    tol = np.finfo(float).eps * max(1.0, n * square_sum) * 32
    v = np.where((v < 0) & (v > -tol), 0, v)
    out = np.empty_like(sums)
    good = v > 0
    out[good] = np.abs(sums[good]) / np.sqrt(v[good])
    out[~good] = np.where(np.abs(sums[~good]) <= tol, 0, np.inf)
    return out


def exact_sign_flip(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    n, split = len(x), len(x) // 2
    left, right = _sign_sums(x[:split]), _sign_sums(x[split:])
    observed_sum, square_sum = float(x.sum()), float(x @ x)
    observed_t = float(_studentized(np.asarray([observed_sum]), square_sum, n)[0])
    mtol = np.finfo(float).eps * max(1.0, abs(observed_sum)) * 32
    ttol = np.finfo(float).eps * max(1.0, observed_t if math.isfinite(observed_t) else 1.0) * 64
    mc = tc = 0
    chunk = max(1, min(len(left), (1 << 20) // max(1, len(right))))
    for start in range(0, len(left), chunk):
        sums = left[start:start + chunk, None] + right[None, :]
        mc += int(np.count_nonzero(np.abs(sums) + mtol >= abs(observed_sum)))
        perm_t = _studentized(sums, square_sum, n)
        tc += int(np.count_nonzero(np.isinf(perm_t))) if math.isinf(observed_t) else int(np.count_nonzero(perm_t + ttol >= observed_t))
    total = 1 << n
    return {"method": "complete exact sign-flip enumeration", "n": n, "n_permutations": total,
            "mean": {"statistic_abs": abs(observed_sum / n), "extreme_count": mc, "p_two_sided": mc / total},
            "studentized": {"statistic_abs": observed_t, "extreme_count": tc, "p_two_sided": tc / total}}


def reconstruct(
    plan_dir: Path, run_dir: Path, *, spec_path: Path | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS, bootstrap_seed: int = BOOTSTRAP_SEED,
    compute_exact_sign_flip: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    spec_path = spec_path or Path(__file__).with_name("spec.json")
    _validate_spec(spec_path)
    rows = _load_plan(plan_dir)
    manifest_shas = {r["manifest_sha256"] for r in rows}
    _require(len(manifest_shas) == 1, "plan contains multiple manifest hashes")
    manifest_sha = next(iter(manifest_shas))
    manifest, manifest_path, hygiene = _load_manifest(plan_dir, manifest_sha)
    hygiene_sha = _validate_hygiene(plan_dir, manifest_sha, hygiene)
    scopes = _validate_plan_and_splits(plan_dir, rows, manifest)
    lock, analysis_ids, external_cache_bytes_verified = _validate_lock(plan_dir, run_dir, rows, spec_path)
    row_map: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["draw_id"] in analysis_ids:
            row_map[(row["draw_id"], row["fold"], row["cell"])].append(row)
    candidates: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for key, group in row_map.items():
        _require(len(group) == 8 and {r["config_index"] for r in group} == set(CONFIGS),
                 f"analysis episode lacks the exact eight-config grid: {key}")
        candidates[key] = [_unit_candidate(r, run_dir, scopes[key],
                          lock["unit_artifact_hashes"][r["unit_id"]]) for r in group]
    draws: dict[str, Any] = {}
    z_values, d09_values = [], []
    for draw in analysis_ids:
        folds_out, fz, fd09 = {}, [], []
        for fold in range(5):
            picked, fixed = {}, {}
            for cell in CELLS:
                group = candidates[(draw, fold, cell)]
                picked[cell] = _select_episode(group)
                fixed[cell] = next(x for x in group if x["config_index"] == FIXED_CONFIG)
            z = (picked["GR_hpo"]["val_uar"] - picked["GR_hpo"]["test_uar"]) - (picked["GG_hpo"]["val_uar"] - picked["GG_hpo"]["test_uar"])
            d09 = picked["GR_hpo"]["test_uar"] - picked["GG_hpo"]["test_uar"] - fixed["GR_hpo"]["test_uar"] + fixed["GG_hpo"]["test_uar"]
            fz.append(z); fd09.append(d09)
            rep = {"z": z, "d09r": d09}
            for cell in CELLS:
                rep[cell] = {"selected_config_index": picked[cell]["config_index"],
                             "selected_unit_id": picked[cell]["unit_id"], "val_uar": picked[cell]["val_uar"],
                             "test_uar": picked[cell]["test_uar"], "fixed_index4_test_uar": fixed[cell]["test_uar"]}
            folds_out[str(fold)] = {"z": z, "d09r": d09, "train_reps": {"0": rep}}
        draw_z, draw_d09 = float(np.mean(fz)), float(np.mean(fd09))
        z_values.append(draw_z); d09_values.append(draw_d09)
        draws[str(draw)] = {"complete": True, "draw_role": "primary" if draw < 24 else "reserve",
                            "analysis_order": draw, "void_reasons": [], "z": draw_z,
                            "d09r": draw_d09, "folds": folds_out}
    primary = {"tested": True, "n_draws": 24, "draw_ids": analysis_ids,
               "values": z_values, **one_sample_t(z_values)}
    d09t = one_sample_t(d09_values)
    d09r = {"descriptive_only": True, "n": 24, "values": d09_values, "mean": d09t["mean"],
            "sd": d09t["sd"], "ci_level": .95, "ci": d09t["ci"],
            "formula": "[T_GR(selected)-T_GG(selected)]-[T_GR(config4)-T_GG(config4)]"}
    source_hashes = {"completion_status": "complete", "analysis_lock_sha256": sha256_file(run_dir / "analysis_lock.json"),
                     "run_plan_sha256": sha256_file(plan_dir / "run_plan.csv"),
                     "split_index_sha256": sha256_file(plan_dir / "split_index.json"),
                     "unit_configs_sha256": sha256_file(plan_dir / "unit_configs.json"),
                     "manifest_sha256": sha256_file(manifest_path)}
    result = {"schema": RESULT_SCHEMA, "study_id": STUDY_ID, "spec_version": SPEC_VERSION,
              "expected_complete_draws": 24, "n_planned_draws": 28, "n_complete_draws": 24,
              "analysis_draw_ids": analysis_ids, "tested": True, "status": "tested",
              "estimand": "mean_draw((V-T)_GR-(V-T)_GG), train reps first then five folds equally weighted",
              "primary": primary, "bootstrap_sensitivity": bootstrap(z_values, bootstrap_reps, bootstrap_seed),
              "exact_sign_flip_sensitivity": exact_sign_flip(z_values) if compute_exact_sign_flip else None,
              "D09R": d09r, "draws": draws,
              "integrity": {"pass": True, "global_failures": [], "void_draw_ids": []},
              "source_hashes": source_hashes}
    evidence = {"spec_json_sha256": sha256_file(spec_path), "run_plan_sha256": sha256_file(plan_dir / "run_plan.csv"),
                "split_index_sha256": sha256_file(plan_dir / "split_index.json"),
                "manifest_sha256": sha256_file(manifest_path), "analysis_lock_sha256": sha256_file(run_dir / "analysis_lock.json"),
                "hygiene_sha256": hygiene_sha,
                "external_cache_bytes_verified": external_cache_bytes_verified}
    return result, evidence


def compare_results(expected: Any, actual: Any, path: str = "$") -> list[dict[str, Any]]:
    """Strict recursive comparison; no scorer field is silently ignored."""
    differences: list[dict[str, Any]] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                differences.append({"path": child, "expected": "<absent>", "actual": actual[key]})
            elif key not in actual:
                differences.append({"path": child, "expected": expected[key], "actual": "<absent>"})
            else:
                differences.extend(compare_results(expected[key], actual[key], child))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append({"path": path, "expected_length": len(expected), "actual_length": len(actual)})
        for i, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(compare_results(left, right, f"{path}[{i}]"))
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            differences.append({"path": path, "expected": expected, "actual": actual})
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isinf(float(expected)) and math.isinf(float(actual)) and
                math.copysign(1, float(expected)) == math.copysign(1, float(actual))) and not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=FLOAT_ATOL):
            differences.append({"path": path, "expected": expected, "actual": actual})
    elif expected != actual:
        differences.append({"path": path, "expected": expected, "actual": actual})
    return differences


def verify_run(
    plan_dir: str | Path, run_dir: str | Path, result_path: str | Path, *,
    spec_path: str | Path | None = None, output_path: str | Path | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS, bootstrap_seed: int = BOOTSTRAP_SEED,
    compute_exact_sign_flip: bool = True,
) -> dict[str, Any]:
    expected, evidence = reconstruct(Path(plan_dir), Path(run_dir),
        spec_path=None if spec_path is None else Path(spec_path), bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed, compute_exact_sign_flip=compute_exact_sign_flip)
    actual = _json(Path(result_path))
    differences = compare_results(expected, actual)
    report = {"schema": VERIFY_SCHEMA, "study_id": STUDY_ID, "pass": not differences,
              "differences": differences, "evidence": evidence, "reconstructed": expected}
    if output_path is not None:
        Path(output_path).write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if differences:
        raise VerificationError(f"scorer result differs at {len(differences)} field(s); first: {differences[0]['path']}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_run(args.plan, args.run, args.result, spec_path=args.spec, output_path=args.output)
    except VerificationError as exc:
        raise SystemExit(f"N14R verification FAILED: {exc}") from exc
    print(json.dumps({"pass": report["pass"], "output": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()

"""Fail-closed, independent verifier for the preregistered N14R2 rerun.

This file deliberately does not import :mod:`v2_1.n14r2.score`, ``plan`` or
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
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from scipy import stats as scipy_stats


RESULT_SCHEMA = "ser-v2.1-n14r2-results-1"
VERIFY_SCHEMA = "ser-v2.1-n14r2-verification-1"
STUDY_ID = "N14R2"
SPEC_VERSION = "2.0.0"
SEED_NAMESPACE = "SER26-N14R2"
CELLS = ("GR_hpo", "GG_hpo")
N_CLASSES = 6
N_FOLDS = 5
N_DRAWS = 28
N_ANALYSIS_DRAWS = 24
CONFIGS = tuple(range(8))
FIXED_CONFIG = 4
ALPHA = 0.05
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 202_609_050_001
FLOAT_ATOL = 1e-10
H3_UNUSABLE_BASENAMES = frozenset({"1076_MTI_SAD_XX.wav"})
BASE_COMMIT = "186b40d4a9db720d246f92dfef5ba73020903b9c"
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
PIN_EXCLUDED_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", "runs", "results", "evidence", "release_staging", "local"}
)
PIN_EXCLUSIONS = [
    "PINS.json (self)", "runs/**", "results/**", "evidence/**",
    "release_staging/**", "local/**", "audio", "old N14/N14R fitted outputs", "__pycache__",
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
    inference = spec.get("inference", {})
    completion = spec.get("completion", {})
    checks = {
        "study_id": spec.get("study_id") == STUDY_ID,
        "spec_version": spec.get("spec_version") == SPEC_VERSION,
        "seed_namespace": spec.get("seed_namespace") == SEED_NAMESPACE,
        "primary IDs": design.get("primary_draw_ids") == list(range(24)),
        "reserve IDs": design.get("reserve_draw_ids") == list(range(24, 28)),
        "folds": design.get("folds_per_draw") == 5,
        "reps": design.get("train_reps") == 1,
        "cells": design.get("cells") == list(CELLS),
        "grid": design.get("configs") == list(HPO_GRID),
        "units": completion.get("units_per_draw") == 80,
        "manifest input": design.get("manifest_input_rows") == 7442,
        "hygienic population": design.get("analysis_population_rows_after_hygiene") == 7435,
        "draws": completion.get("required_complete_draws") == 24,
        "activated draws": completion.get("maximum_activated_draws") == 28,
        "attempts per unit": completion.get("maximum_attempts_per_unit") == 2,
        "maximum attempts": completion.get("theoretical_maximum_training_attempts") == 4480,
        "bootstrap": inference.get("bootstrap_sensitivity") == {
            "method": "percentile", "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
        },
        "directional boundary": inference.get("directional_claim_requires") ==
        "mean_delta>0 and holm_adjusted_two_sided_p<=0.05",
        "pilot excluded": spec.get("legacy_pilot", {}).get("excluded_from_confirmatory_analysis") is True,
        "execution environment": spec.get("execution_environment") == {
            "python": "3.12.3", "numpy": "2.4.6", "scipy": "1.18.0",
            "scikit_learn": "1.9.0", "torch": "2.11.0+cu128",
            "torch_cuda": "12.8", "cudnn": 91900, "librosa": "0.11.0",
            "gpu": "NVIDIA GeForce RTX 4090",
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
    expected = {"schema": "ser26-n14r2-hygiene-1",
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
            key = f"n14r2__d{draw:02d}__{cell}"
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
    # <root>/v2_1/n14r2/PINS.json, so retain offline verification when that
    # structural root is unambiguous; the complete pinned inventory below is
    # still verified byte for byte.
    resolved = start.resolve()
    if len(resolved.parents) >= 3:
        candidate = resolved.parents[2]
        if resolved.parent == (candidate / "v2_1" / "n14r2").resolve():
            return candidate
    for path in (start, *start.parents):
        if (path / ".git").exists():
            return path
    raise VerificationError("cannot locate the N14R2 release root for PINS verification")


def _portable_basename(value: str) -> str:
    """Extract a receipt basename independent of the verifier host OS."""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _validate_pins(pins_path: Path, lock: dict[str, Any], environment: dict[str, Any]) -> bool:
    pins = _json(pins_path)
    payload = {key: value for key, value in pins.items() if key != "content_manifest_sha256"}
    _require(pins.get("schema") == "ser26-n14r2-pins-1" and
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
    study = root / "v2_1" / "n14r2"
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


def _read_strict_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    raw = path.read_bytes()
    _require(not raw or raw.endswith(b"\n"), f"torn JSONL ledger: {path}")
    events, prefixes = [], []
    prefix = b""
    for line_no, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            event = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VerificationError(f"malformed ledger line {line_no}: {path}") from exc
        _require(isinstance(event, dict) and event.get("seq") == line_no,
                 f"ledger sequence mismatch at line {line_no}: {path}")
        prefixes.append(prefix)
        prefix += line
        events.append(event)
    return events, prefixes


def _locked_attempt_dir(run_dir: Path, row: dict[str, Any], entry: Any) -> tuple[Path, dict[str, str]]:
    _require(isinstance(entry, dict) and entry.get("status") == "done",
             f"locked successful attempt is malformed: {row['unit_id']}")
    try:
        attempt, raw = int(entry["attempt"]), str(entry["attempt_path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"locked attempt identity is malformed: {row['unit_id']}") from exc
    expected = (Path("nodes") / f"node{row['draw_id'] % 4}" / "units" / row["unit_id"]
                / f"attempt_{attempt:02d}")
    pure = PurePosixPath(raw)
    _require(not pure.is_absolute() and ".." not in pure.parts and Path(*pure.parts) == expected,
             f"locked attempt path is unsafe or misrouted: {row['unit_id']}")
    attempt_dir = run_dir.joinpath(*pure.parts)
    artifacts = entry.get("artifact_hashes")
    _require(isinstance(artifacts, dict) and
             set(artifacts) == {"unit.json", "history.json", "predictions.csv", "DONE"},
             f"locked artifact map is malformed: {row['unit_id']}")
    receipt = attempt_dir / "artifact_receipt.json"
    _require(receipt.is_file() and sha256_file(receipt) == entry.get("artifact_receipt_sha256"),
             f"artifact receipt changed after closure: {row['unit_id']}")
    _require(all((attempt_dir / name).is_file() and
                 sha256_file(attempt_dir / name) == digest for name, digest in artifacts.items()),
             f"outcome artifact changed after closure: {row['unit_id']}")
    return attempt_dir, artifacts


def _validate_lock(
    plan_dir: Path, run_dir: Path, rows: list[dict[str, Any]], spec_path: Path,
) -> tuple[dict[str, Any], list[int], bool]:
    completion_path = run_dir / "completion.json"
    _require(completion_path.is_file(), "completion.json is required before outcomes are read")
    completion = _json(completion_path)
    _require(completion.get("schema") == "ser26-n14r2-completion-1" and
             completion.get("study_id") == STUDY_ID and completion.get("status") == "complete" and
             completion.get("contains_outcome_statistics") is False,
             "completion record does not close a complete outcome-blind N14R2 run")
    lock_path = run_dir / "analysis_lock.json"
    _require(lock_path.is_file(), "analysis_lock.json is required before outcomes are read")
    lock = _json(lock_path)
    _require(lock.get("schema") == "ser26-n14r2-analysis-lock-1" and
             lock.get("study_id") == STUDY_ID and lock.get("spec_version") == SPEC_VERSION,
             "analysis lock schema/identity mismatch")
    _require(lock.get("contains_outcome_statistics") is False, "analysis lock contains outcome statistics")
    clone = dict(lock)
    self_hash = clone.pop("lock_payload_sha256", None)
    _require(self_hash == sha256_json(clone), "analysis lock payload hash mismatch")
    _require(completion.get("analysis_lock_payload_sha256") == self_hash and
             completion.get("analysis_lock_sha256") == sha256_file(lock_path),
             "completion record does not commit the exact analysis lock")
    ids = lock.get("complete_draw_ids")
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
    bindings_path = plan_dir.parent / "ENVIRONMENT_BINDINGS.json"
    _require(isinstance(prereg, dict), "analysis lock lacks preregistration_hashes")
    for name, path in (("spec.json", spec_path), ("PINS.json", pins_path),
                       ("ENVIRONMENT_BINDINGS.json", bindings_path)):
        _require(path.is_file() and prereg.get(name) == sha256_file(path),
                 f"locked preregistration hash is missing or mismatched: {name}")
    bindings = _json(bindings_path)
    _require(bindings.get("schema") == "ser26-n14r2-environment-bindings-1" and
             bindings.get("study_id") == STUDY_ID and lock.get("environment_bindings") == bindings,
             "environment bindings changed")
    # Source inventory and optional external-cache bytes are checked without
    # importing the PINS builder or any runner code.
    external_cache_bytes_verified = _validate_pins(pins_path, lock, {})
    node_ids = {f"node{i}" for i in range(4)}
    environment_hashes = lock.get("environment_receipt_hashes")
    node_ledger_hashes = lock.get("node_ledger_hashes")
    _require(isinstance(environment_hashes, dict) and set(environment_hashes) == node_ids and
             isinstance(node_ledger_hashes, dict) and set(node_ledger_hashes) == node_ids,
             "analysis lock lacks four node environment/ledger hashes")
    environment_contracts: dict[str, dict[str, Any]] = {}
    for node_id in sorted(node_ids):
        node_dir = run_dir / "nodes" / node_id
        env_path, ledger_path = node_dir / "execution_environment.json", node_dir / "ledger.jsonl"
        binding = bindings.get("nodes", {}).get(node_id, {})
        _require(env_path.is_file() and sha256_file(env_path) == environment_hashes[node_id] ==
                 binding.get("environment_sha256"), f"environment receipt changed: {node_id}")
        env = _json(env_path)
        contract = env.get("contract")
        _require(env.get("schema") == "ser26-n14r2-environment-1" and
                 env.get("study_id") == STUDY_ID and env.get("node_id") == node_id and
                 env.get("shard_modulus") == 4 and env.get("shard_remainder") == int(node_id[-1]) and
                 isinstance(contract, dict) and env.get("contract_sha256") == sha256_json(contract),
                  f"environment receipt identity/hash mismatch: {node_id}")
        environment_contracts[node_id] = contract
        gpu = contract.get("gpu", {})
        _require(gpu.get("uuid") == binding.get("gpu_uuid") and
                 gpu.get("driver_version") == binding.get("driver_version") and
                 gpu.get("name") == "NVIDIA GeForce RTX 4090",
                 f"bound GPU identity changed: {node_id}")
        _require(ledger_path.is_file() and sha256_file(ledger_path) == node_ledger_hashes[node_id],
                 f"node ledger changed: {node_id}")
    global_path = run_dir / "global_ledger.jsonl"
    _require(global_path.is_file() and sha256_file(global_path) == lock.get("global_ledger_sha256") ==
             completion.get("global_ledger_sha256"), "global ledger changed")
    global_events, _ = _read_strict_jsonl(global_path)
    _require(global_events and global_events[-1].get("event") == "run_stop",
              "global ledger lacks terminal run_stop")
    _require(global_events[-1].get("status") in {"ready_for_closure", "complete"},
             "global ledger was not stopped for complete closure")
    _require(global_events[0].get("event") == "run_start" and
             sum(e.get("event") == "run_start" for e in global_events) == 1 and
             sum(e.get("event") == "run_stop" for e in global_events) == 1 and
             global_events[0].get("plan_run_sha256") == hashes["run_plan.csv"] and
             global_events[0].get("pins_sha256") == prereg["PINS.json"],
             "global run boundary/frozen-input binding is invalid")
    global_start = global_events[0]
    _require(re.fullmatch(r"[0-9a-f]{40}", str(global_start.get("preregistration_commit", ""))) is not None and
             str(global_start.get("preregistration_tag", "")).startswith("SER26-N14R2-prereg-") and
             str(global_start.get("public_receipt_url", "")).startswith(
                 "https://github.com/DeerNeverStop/SER/pull/6#issuecomment-") and
             isinstance(global_start.get("public_receipt_created_at"), str) and
             global_start.get("contains_outcome_statistics") is False,
             "global run_start lacks the prospective public registration receipt")
    global_prefix_hashes: set[str] = set()
    authorizations_by_prefix: dict[str, dict[str, str]] = {}
    seen_authorizations: dict[int, str] = {}
    global_prefix = b""
    for line, global_event in zip(global_path.read_bytes().splitlines(keepends=True), global_events):
        global_prefix += line
        if global_event.get("event") == "draw_authorized":
            seen_authorizations[int(global_event["draw_id"])] = global_event["authorization_id"]
        digest = sha256_bytes(global_prefix)
        global_prefix_hashes.add(digest)
        authorizations_by_prefix[digest] = {
            str(draw): auth_id for draw, auth_id in seen_authorizations.items()
        }
    previous = None
    authorizations: dict[int, dict[str, Any]] = {}
    terminals: dict[int, dict[str, Any]] = {}
    recoveries: dict[str, dict[str, Any]] = {}
    registered: dict[str, dict[str, Any]] = {}
    reserves: list[int] = []
    for event in global_events:
        payload = {key: value for key, value in event.items() if key != "event_sha256"}
        _require(event.get("study_id") == STUDY_ID and
                 event.get("prev_event_sha256") == previous and
                 event.get("event_sha256") == sha256_json(payload),
                 "global authorization ledger hash chain is invalid")
        previous = event["event_sha256"]
        kind = event.get("event")
        if kind == "node_recovery_authorized":
            auth_id = event.get("authorization_id")
            node_id = event.get("node_id")
            _require(isinstance(auth_id, str) and auth_id and auth_id not in recoveries and node_id in node_ids and
                     event.get("shard_remainder") == int(node_id[-1]) and
                     event.get("plan_run_sha256") == hashes["run_plan.csv"] and
                     event.get("pins_sha256") == prereg["PINS.json"] and
                     isinstance(event.get("prior_node_ledger_sha256"), str),
                     "invalid global node recovery authorization")
            recoveries[auth_id] = event
        elif kind == "node_registered":
            node_id = event.get("node_id")
            binding = bindings.get("nodes", {}).get(node_id, {})
            _require(node_id in node_ids and node_id not in registered and
                     all(event.get(key) == binding.get(key) for key in
                         ("shard_remainder", "environment_sha256", "gpu_uuid", "driver_version")),
                     "global node registration differs from prospective binding")
            registered[node_id] = event
        elif kind == "draw_authorized":
            draw = event.get("draw_id")
            _require(type(draw) is int and draw in range(28) and draw not in authorizations and
                     event.get("node_id") == f"node{draw % 4}" and event.get("shard_remainder") == draw % 4 and
                     event.get("draw_role") == ("primary" if draw < 24 else "reserve") and
                     event.get("plan_run_sha256") == hashes["run_plan.csv"] and
                     event.get("pins_sha256") == prereg["PINS.json"] and
                     isinstance(event.get("authorization_id"), str) and event["authorization_id"],
                     "invalid global draw authorization")
            if draw >= 24:
                _require(draw == 24 + len(reserves) and set(range(24)).issubset(terminals) and
                         all(terminals[d]["status"] in {"complete", "void"} for d in range(24)),
                         "reserve authorized before all primary draws resolved or out of order")
                _require(sum(x["status"] == "void" for x in terminals.values()) >= draw - 23,
                         "reserve authorization lacks enough prior voids")
                if draw > 24:
                    _require(draw - 1 in terminals, "reserve authorized before prior reserve was terminal")
                _require(sum(x["status"] == "complete" for x in terminals.values()) < 24,
                         "reserve authorized after 24 complete draws")
                reserves.append(draw)
            else:
                _require(not reserves, "primary draw authorized after reserve activation")
            authorizations[draw] = event
        elif kind == "draw_terminal":
            draw = event.get("draw_id")
            _require(type(draw) is int and draw in authorizations and draw not in terminals and
                     event.get("node_id") == f"node{draw % 4}" and event.get("status") in {"complete", "void"} and
                     isinstance(event.get("node_draw_receipt_sha256"), str),
                     "invalid global draw terminal event")
            terminals[draw] = event
        else:
            _require(kind in {"run_start", "circuit_break", "run_stop"},
                     f"unknown global event {kind!r}")
    _require(set(range(24)).issubset(authorizations), "global ledger did not authorize all primary draws")
    _require(set(registered) == node_ids, "global ledger lacks exactly four node registrations")
    row_by_id = {row["unit_id"]: row for row in rows}
    all_attempts: dict[str, list[dict[str, Any]]] = {}
    node_prefix_hashes: dict[str, dict[int, str]] = {}
    for node_id in sorted(node_ids):
        ledger_path = run_dir / "nodes" / node_id / "ledger.jsonl"
        events, prefixes = _read_strict_jsonl(ledger_path)
        states: dict[str, str] = {}
        open_attempts: dict[str, int] = {}
        attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        invocation = None
        previous_event = None
        used_recoveries: set[str] = set()
        circuit_invocations: set[str] = set()
        failed_invocations: set[str] = set()
        ready_slots: dict[str, set[int]] = defaultdict(set)
        infra_invocations: set[str] = set()
        prefix_by_line = {i + 1: sha256_bytes(prefixes[i]) for i in range(len(events))}
        node_prefix_hashes[node_id] = prefix_by_line
        for ordinal, event in enumerate(events, 1):
            _require(event.get("study_id") == STUDY_ID and event.get("node_id") == node_id and
                     event.get("shard_remainder") == int(node_id[-1]),
                     f"node ledger identity mismatch: {node_id} line {ordinal}")
            kind, event_invocation = event.get("event"), event.get("invocation_id")
            _require(isinstance(event_invocation, str) and event_invocation,
                     f"node invocation missing: {node_id} line {ordinal}")
            if kind == "runner_start":
                _require(invocation is None or event_invocation != invocation,
                         f"node invocation ID was reused: {node_id}")
                if invocation is not None:
                    recovery_id = event.get("recovery_authorization_id")
                    prior_hash = event.get("prior_node_ledger_sha256")
                    clean = (previous_event is not None and previous_event.get("event") == "runner_stop" and
                             previous_event.get("status") in {"authorized_draws_terminal", "paused_operator"})
                    if clean:
                        _require(recovery_id is None and prior_hash is None,
                                 f"clean node restart claimed recovery: {node_id}")
                    else:
                        recovery = recoveries.get(recovery_id)
                        source_hash = recovery.get("prior_node_ledger_sha256") if recovery else None
                        boundaries = [i for i in range(1, ordinal + 1)
                                      if prefix_by_line[i] == source_hash]
                        _require(recovery is not None and recovery_id not in used_recoveries and
                                 recovery.get("node_id") == node_id and
                                 recovery.get("shard_remainder") == int(node_id[-1]) and prior_hash == source_hash and
                                 len(boundaries) == 1 and
                                 all(item.get("event") in {"done", "failed", "circuit_break"}
                                     for item in events[boundaries[0] - 1:ordinal - 1]),
                                 f"node restart lacks exact global recovery authorization: {node_id}")
                        used_recoveries.add(recovery_id)
                else:
                    _require(event.get("recovery_authorization_id") is None and
                             event.get("prior_node_ledger_sha256") is None,
                             f"initial node invocation claimed recovery: {node_id}")
                invocation = event_invocation
                global_sha = event.get("global_ledger_sha256")
                expected_authorizations = {
                    str(draw): auth_id
                    for draw, auth_id in authorizations_by_prefix.get(global_sha, {}).items()
                    if int(draw) % 4 == int(node_id[-1])
                }
                _require(event.get("plan_run_sha256") == hashes["run_plan.csv"] and
                         event.get("pins_sha256") == prereg["PINS.json"] and
                         event.get("execution_environment_sha256") == environment_hashes[node_id] and
                         event.get("global_ledger_sha256") in global_prefix_hashes and
                         event.get("authorization_ids") == expected_authorizations and
                         event.get("workers") == 8,
                         f"runner_start provenance binding mismatch: {node_id}")
                previous_event = event
                continue
            _require(invocation is not None and event_invocation == invocation,
                     f"node invocation changed without runner_start: {node_id}")
            if kind == "circuit_break":
                circuit_invocations.add(invocation)
                previous_event = event
                continue
            if kind == "worker_ready":
                slot = event.get("slot")
                _require(type(slot) is int and slot in range(8) and slot not in ready_slots[invocation] and
                         event.get("cuda_healthy") is True and event.get("health_checksum") == 4096.0 and
                         event.get("backend_precision") == environment_contracts[node_id].get("backend_precision"),
                         f"invalid or duplicate worker health receipt: {node_id}")
                ready_slots[invocation].add(slot)
                previous_event = event
                continue
            if kind == "runner_stop":
                _require(event.get("status") in {"authorized_draws_terminal", "paused_operator"},
                         f"invalid clean runner_stop status: {node_id}")
                previous_event = event
                continue
            _require(kind in {"start", "done", "failed"}, f"unknown node ledger event: {kind!r}")
            uid, draw, attempt = event.get("unit_id"), event.get("draw_id"), event.get("attempt")
            row = row_by_id.get(uid)
            _require(row is not None and draw == row["draw_id"] and draw % 4 == int(node_id[-1]) and
                     draw in authorizations and type(attempt) is int and attempt in (1, 2),
                     f"unplanned/misrouted attempt: {node_id} line {ordinal}")
            if kind == "start":
                _require(event.get("authorization_id") == authorizations[draw]["authorization_id"] and
                         event.get("plan_row_sha256") == sha256_json({k: str(v) for k, v in row.items()}) and
                         len(ready_slots[invocation]) == 8 and invocation not in circuit_invocations and
                         invocation not in failed_invocations and uid not in open_attempts and
                         states.get(uid) != "done" and attempt == len(attempts[uid]) + 1,
                         f"invalid attempt start: {uid}")
                open_attempts[uid] = attempt
                previous_event = event
                continue
            _require(open_attempts.get(uid) == attempt, f"terminal without matching start: {uid}")
            attempt_dir = run_dir / "nodes" / node_id / "units" / uid / f"attempt_{attempt:02d}"
            receipt_path = attempt_dir / "artifact_receipt.json"
            _require(receipt_path.is_file() and sha256_file(receipt_path) == event.get("receipt_sha256"),
                     f"terminal receipt hash mismatch: {uid}")
            receipt = _json(receipt_path)
            _require(set(receipt) == {"schema", "study_id", "node_id", "shard_remainder", "unit_id",
                                      "draw_id", "attempt", "status", "failure_class", "plan_row_sha256",
                                      "artifact_hashes", "started_at", "finished_at"} and
                     receipt.get("schema") == "ser26-n14r2-artifact-receipt-1" and
                     receipt.get("study_id") == STUDY_ID and receipt.get("node_id") == node_id and
                     receipt.get("shard_remainder") == int(node_id[-1]) and
                     receipt.get("unit_id") == uid and receipt.get("draw_id") == draw and
                     receipt.get("attempt") == attempt and receipt.get("status") == kind and
                     isinstance(receipt.get("started_at"), str) and isinstance(receipt.get("finished_at"), str) and
                     receipt.get("plan_row_sha256") == sha256_json({k: str(v) for k, v in row.items()}),
                     f"attempt receipt identity/status mismatch: {uid}")
            artifact_hashes = receipt.get("artifact_hashes")
            if kind == "done":
                _require(receipt.get("failure_class") is None and isinstance(artifact_hashes, dict) and
                         set(artifact_hashes) == {"unit.json", "history.json", "predictions.csv", "DONE"} and
                         all((attempt_dir / name).is_file() and sha256_file(attempt_dir / name) == digest
                             for name, digest in artifact_hashes.items()),
                         f"completed receipt/artifact mismatch: {uid}")
                pointer_path = attempt_dir.parent / "DONE.json"
                pointer = _json(pointer_path)
                _require(set(pointer) == {"attempt", "artifact_receipt_sha256"} and
                         pointer["attempt"] == attempt and pointer["artifact_receipt_sha256"] == event["receipt_sha256"],
                         f"DONE pointer mismatch: {uid}")
                states[uid] = "done"
            else:
                failure = receipt.get("failure_class")
                _require(failure in {"training", "cuda", "infrastructure", "worker_exit", "timeout",
                                     "interrupted", "circuit_cancelled", "integrity"} and not artifact_hashes and
                         event.get("failure_class") == failure,
                         f"failed receipt classification mismatch: {uid}")
                states[uid] = "failed"
                failed_invocations.add(invocation)
                if failure != "training":
                    infra_invocations.add(invocation)
            record = {"attempt": attempt, "status": kind, "failure_class": receipt.get("failure_class"),
                      "artifact_receipt_sha256": event["receipt_sha256"],
                      "attempt_path": attempt_dir.relative_to(run_dir).as_posix(),
                      "artifact_hashes": artifact_hashes or {}}
            if kind == "done":
                record["done_pointer_sha256"] = sha256_file(attempt_dir.parent / "DONE.json")
            attempts[uid].append(record)
            del open_attempts[uid]
            previous_event = event
        _require(not open_attempts, f"node ledger contains orphan starts: {node_id}")
        _require(events[-1].get("event") == "runner_stop" and
                 events[-1].get("status") in {"authorized_draws_terminal", "paused_operator"},
                 f"complete closure requires a clean final node stop: {node_id}")
        _require(infra_invocations.issubset(circuit_invocations),
                 f"node infrastructure failure lacks circuit break: {node_id}")
        _require(not (set(all_attempts) & set(attempts)), "unit appeared in multiple node ledgers")
        all_attempts.update(attempts)
    used_recovery_ids = {
        event.get("recovery_authorization_id")
        for node_id in sorted(node_ids)
        for event in _read_strict_jsonl(run_dir / "nodes" / node_id / "ledger.jsonl")[0]
        if event.get("event") == "runner_start" and event.get("recovery_authorization_id") is not None
    }
    _require(used_recovery_ids == set(recoveries), "global recovery authorization inventory mismatch")
    _require(set(terminals) == set(authorizations), "complete closure contains unresolved authorized draws")
    _require(lock.get("all_attempt_receipts") == all_attempts,
             "analysis lock does not exactly bind independently replayed attempts")
    draw_hashes = lock.get("draw_receipt_hashes")
    _require(isinstance(draw_hashes, dict) and set(map(int, draw_hashes)) == set(terminals),
             "locked draw-receipt inventory differs from global terminals")
    complete, voids = [], set()
    for draw, terminal in terminals.items():
        path = run_dir / "nodes" / f"node{draw % 4}" / "draws" / f"draw_{draw:02d}.json"
        _require(path.is_file() and sha256_file(path) == terminal["node_draw_receipt_sha256"] ==
                 draw_hashes[str(draw)], f"draw receipt hash mismatch: {draw}")
        receipt = _json(path)
        planned = {row["unit_id"] for row in rows if row["draw_id"] == draw}
        selected, selected_hashes = receipt.get("selected_attempts"), receipt.get("artifact_receipt_sha256")
        _require(receipt.get("schema") == "ser26-n14r2-node-draw-1" and
                 receipt.get("study_id") == STUDY_ID and receipt.get("node_id") == f"node{draw % 4}" and
                 receipt.get("draw_id") == draw and set(receipt.get("unit_ids", [])) == planned and
                 receipt.get("status") == terminal["status"] and isinstance(selected, dict) and
                 isinstance(selected_hashes, dict), f"draw receipt identity mismatch: {draw}")
        done = {uid for uid in planned if any(x["status"] == "done" for x in all_attempts.get(uid, []))}
        _require(set(selected) == set(selected_hashes) == done, f"draw selected-attempt map mismatch: {draw}")
        exhausted = {uid for uid in planned if len(all_attempts.get(uid, [])) == 2 and
                     all(x["status"] == "failed" and x["failure_class"] == "training"
                         for x in all_attempts[uid])}
        blocked = {uid for uid in planned if len(all_attempts.get(uid, [])) == 2 and uid not in done | exhausted}
        _require(not blocked, f"mixed/infrastructure exhaustion was treated as terminal: {draw}")
        if terminal["status"] == "complete":
            _require(done == planned and not receipt.get("exhausted_unit_ids"), f"incomplete draw declared complete: {draw}")
            complete.append(draw)
        else:
            _require(bool(exhausted) and set(receipt.get("exhausted_unit_ids", [])) == exhausted and
                     set(terminal.get("exhausted_unit_ids", [])) == exhausted and terminal.get("reason"),
                     f"void draw lacks exact two-training-failure evidence: {draw}")
            voids.add(draw)
    expected_ids = sorted(complete)[:24]
    _require(ids == expected_ids and len(complete) == 24,
             "analysis lock is not exactly the first 24 complete draws")
    expected_units = {row["unit_id"] for row in rows if row["draw_id"] in ids}
    artifacts = lock.get("unit_artifact_hashes")
    _require(isinstance(artifacts, dict) and set(artifacts) == expected_units and
             lock.get("n_locked_units") == 1920,
             "locked unit artifact set is not exactly the 24-draw analysis set")
    for row in rows:
        if row["unit_id"] in expected_units:
            _require(artifacts[row["unit_id"]] in all_attempts[row["unit_id"]],
                     f"selected artifact is not a ledgered immutable attempt: {row['unit_id']}")
            _locked_attempt_dir(run_dir, row, artifacts[row["unit_id"]])
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
    unit_dir, locked_artifacts = _locked_attempt_dir(run_dir, row, locked)
    paths = {name: unit_dir / name for name in ("unit.json", "history.json", "predictions.csv", "DONE")}
    _require(all(p.is_file() for p in paths.values()), f"unit artifact missing: {row['unit_id']}")
    observed = {name: sha256_file(path) for name, path in paths.items()}
    _require(locked_artifacts == observed,
             f"analysis-lock artifact hash mismatch: {row['unit_id']}")
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
              "D09R2": d09r, "draws": draws,
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
        raise SystemExit(f"N14R2 verification FAILED: {exc}") from exc
    print(json.dumps({"pass": report["pass"], "output": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()

"""Generate and validate the frozen N14R split and unit plan.

Usage:
  python -m v2_1.n14r.plan --manifest v2/manifests/cremad_manifest.csv --out <directory>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from v2.ser_v2 import corpora as v2_corpora

from . import seeds
from .spec import (ALL_DRAW_IDS, CELLS, CORPUS_LEVEL, HPO_GRID, MODEL, N_CLASSES,
                   N_FOLDS, PLAN_FIELDS, PRIMARY_DRAW_IDS, SPEC, SPEC_VERSION,
                   STUDY_ID, canonical_json, sha256_json, validate_spec)

EST_GPU_SEC_PER_UNIT = 92.9 * 3600.0 / (24 * 5 * 2 * 8)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _csv_text(fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Load CREMA-D exactly as the reused trainer does, then renumber kept rows.

    The raw file hash remains the manifest identity committed into every unit.
    ``hygiene`` separately commits the deterministic derived population.
    """
    raw = path.read_bytes()
    raw_sha256 = sha256_bytes(raw)
    rows = v2_corpora.load_manifest(path)
    required = {"sample_index", "relative_path", "speaker", "label", "label_index"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest lacks required fields: {sorted(required)}")
    rows.sort(key=lambda r: int(r["sample_index"]))
    indices = [int(r["sample_index"]) for r in rows]
    if indices != list(range(len(rows))):
        raise ValueError("sample_index must be contiguous and zero-based")
    if len({r["relative_path"] for r in rows}) != len(rows):
        raise ValueError("relative_path must be unique")
    kept, hygiene = v2_corpora.apply_hygiene("cremad", rows)
    kept_indices = [int(r["sample_index"]) for r in kept]
    if kept_indices != list(range(len(kept))):
        raise RuntimeError("hygiene output sample_index must be contiguous and zero-based")
    if len({int(r["label_index"]) for r in kept}) != N_CLASSES:
        raise ValueError("N14R requires exactly six label_index levels")
    hygiene = {
        "schema": "ser26-n14r-hygiene-1",
        "manifest_sha256": raw_sha256,
        "corpus": hygiene["corpus"],
        "n_input": int(hygiene["n_input"]),
        "n_kept": int(hygiene["n_kept"]),
        "n_dropped": int(hygiene["n_dropped"]),
        "dropped": hygiene["dropped"],
    }
    return kept, raw_sha256, hygiene


def _all_classes(values: np.ndarray) -> bool:
    return set(values.tolist()) == set(range(N_CLASSES))


def _inner_random(train: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    fit, val = train_test_split(train, test_size=0.25, stratify=y[train], random_state=seed)
    return np.sort(fit), np.sort(val), 0


def _inner_grouped(train: np.ndarray, y: np.ndarray, speaker: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=(seed + offset) & 0xFFFFFFFF)
        local_fit, local_val = next(splitter.split(train, y[train], speaker[train]))
        fit, val = np.sort(train[local_fit]), np.sort(train[local_val])
        if _all_classes(y[fit]) and _all_classes(y[val]):
            return fit, val, offset
    raise RuntimeError("grouped inner split did not contain all six classes after 100 frozen attempts")


def _refs(idx: np.ndarray, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {"sample_index": int(i), "relative_path": rows[int(i)]["relative_path"],
         "label_index": int(rows[int(i)]["label_index"]), "speaker": rows[int(i)]["speaker"]}
        for i in idx
    ]


def _partition_hash(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def _draw_role(draw_id: int) -> str:
    return "primary" if draw_id in PRIMARY_DRAW_IDS else "reserve"


def make_plan(rows: list[dict[str, str]], manifest_sha256: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    validate_spec()
    y = np.asarray([int(r["label_index"]) for r in rows], dtype=np.int64)
    speaker = np.asarray([r["speaker"] for r in rows])
    if set(y.tolist()) != set(range(N_CLASSES)):
        raise ValueError("N14R requires all six frozen label indices 0..5")
    idx = np.arange(len(rows), dtype=np.int64)
    plan: list[dict[str, Any]] = []
    split_docs: dict[str, dict[str, Any]] = {}
    seen_outer_draw_hashes: set[str] = set()

    for analysis_order, draw_id in enumerate(ALL_DRAW_IDS):
        sseed = seeds.split_seed(draw_id)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=sseed)
        outer = [(np.sort(tr), np.sort(te)) for tr, te in sgkf.split(idx, y, speaker)]
        tests = np.concatenate([test for _, test in outer])
        if sorted(tests.tolist()) != idx.tolist() or len(np.unique(tests)) != len(idx):
            raise RuntimeError("outer tests must partition the manifest exactly once")
        draw_outer_hash = sha256_json([{"train": tr.tolist(), "test": te.tolist()} for tr, te in outer])
        if draw_outer_hash in seen_outer_draw_hashes:
            raise RuntimeError("outer draws must be unique")
        seen_outer_draw_hashes.add(draw_outer_hash)
        cell_docs = {
            cell: {
                "key": f"n14r__d{draw_id:02d}__{cell}",
                "study_id": STUDY_ID, "spec_version": SPEC_VERSION, "manifest_sha256": manifest_sha256,
                "draw_id": draw_id, "draw_role": _draw_role(draw_id), "analysis_order": analysis_order,
                "split_seed": sseed, "draw_outer_sha256": draw_outer_hash,
                "population": [r["relative_path"] for r in rows], "folds": [],
            }
            for cell in CELLS
        }
        for fold, (train, test) in enumerate(outer):
            if set(speaker[train]) & set(speaker[test]):
                raise RuntimeError("outer train and test speakers overlap")
            if not (_all_classes(y[train]) and _all_classes(y[test])):
                raise RuntimeError("every outer partition must contain all six classes")
            outer_payload = {"outer_train": _refs(train, rows), "test": _refs(test, rows)}
            outer_sha = _partition_hash(outer_payload)
            for cell in CELLS:
                iseed = seeds.inner_seed(draw_id, fold, cell)
                if cell == "GR_hpo":
                    fit, val, attempt = _inner_random(train, y, iseed)
                else:
                    fit, val, attempt = _inner_grouped(train, y, speaker, iseed)
                if not (_all_classes(y[fit]) and _all_classes(y[val])):
                    raise RuntimeError("every inner partition must contain all six classes")
                if cell == "GG_hpo" and (set(speaker[fit]) & set(speaker[val])):
                    raise RuntimeError("GG_hpo inner fit and validation speakers overlap")
                overlap_speakers = set(speaker[fit]) & set(speaker[val])
                if cell == "GR_hpo" and not overlap_speakers:
                    raise RuntimeError("GR_hpo must have speaker overlap between inner fit and validation")
                if sorted(np.concatenate([fit, val]).tolist()) != train.tolist():
                    raise RuntimeError("inner fit and validation must partition outer train")
                inner_payload = {"fit": _refs(fit, rows), "val": _refs(val, rows)}
                inner_sha = _partition_hash(inner_payload)
                legacy_fold = {
                    "fold": fold,
                    "test": [rows[int(i)]["relative_path"] for i in test],
                    "val": [rows[int(i)]["relative_path"] for i in val],
                    "fit": [rows[int(i)]["relative_path"] for i in fit],
                    "meta": {
                        "study_id": STUDY_ID, "draw_id": draw_id, "cell": cell,
                        "split_seed": sseed, "inner_seed": iseed, "inner_attempt": attempt,
                        "outer_sha256": outer_sha, "inner_sha256": inner_sha,
                        "inner_overlap_speaker_count": len(overlap_speakers),
                        "inner_overlap_speaker_fraction": len(overlap_speakers) / len(set(speaker[val])),
                    },
                }
                cell_docs[cell]["folds"].append(legacy_fold)
                # The full-file split hash is filled after all five folds exist.
                for config_index, config in enumerate(HPO_GRID):
                    cfg_sha = sha256_json(config)
                    row = {
                        "study_id": STUDY_ID, "spec_version": SPEC_VERSION,
                        "arm": "N14R", "manifest_sha256": manifest_sha256, "draw_id": draw_id,
                        "draw_role": _draw_role(draw_id), "analysis_order": analysis_order,
                        "fold": fold, "train_rep": 0, "cell": cell,
                        "config_index": config_index, "model": MODEL,
                        "corpus_level": CORPUS_LEVEL, "base_corpus": "cremad", "panel_draw": "",
                        "r": draw_id, "seed_index": 0, "split_seed": sseed,
                        "inner_seed": iseed, "train_seed": seeds.train_seed(draw_id, fold),
                        "outer_sha256": outer_sha, "inner_sha256": inner_sha,
                        "split_sha256": "__FILL_AFTER_CELL_TABLE__", "config_sha256": cfg_sha,
                        "n_fit": len(fit), "n_val": len(val),
                        "n_test": len(test), "est_gpu_sec": round(EST_GPU_SEC_PER_UNIT, 3),
                        "cap_group": "N14R", "conditional": 0, "truncation_rank": 99, "status": "pending",
                    }
                    plan.append(row)
        for cell, doc in cell_docs.items():
            key = doc["key"]
            table = {"population": doc["population"], "folds": doc["folds"],
                     "meta": {k: v for k, v in doc.items() if k not in {"key", "population", "folds"}}}
            split_sha = sha256_json(table)
            split_docs[key] = table
            for row in plan:
                if row["draw_id"] == draw_id and row["cell"] == cell:
                    row["split_sha256"] = split_sha
                    scientific = {k: row[k] for k in row if k not in {"unit_id", "est_gpu_sec", "status"}}
                    scientific["seed_namespace"] = SPEC["seed_namespace"]
                    scientific["config"] = HPO_GRID[row["config_index"]]
                    row["unit_id"] = sha256_json(scientific)
    plan = [{field: row[field] for field in PLAN_FIELDS} for row in plan]
    validate_plan(plan, split_docs)
    return plan, split_docs


def validate_plan(plan: list[dict[str, Any]], split_docs: dict[str, dict[str, Any]]) -> None:
    if len(plan) != 2240 or len({r["unit_id"] for r in plan}) != len(plan):
        raise RuntimeError("plan must contain 2240 unique units")
    for draw_id in ALL_DRAW_IDS:
        rows = [r for r in plan if int(r["draw_id"]) == draw_id]
        if len(rows) != 80:
            raise RuntimeError(f"draw {draw_id} does not contain exactly 80 units")
        for fold in range(N_FOLDS):
            fold_rows = [r for r in rows if int(r["fold"]) == fold]
            if len(fold_rows) != 16:
                raise RuntimeError("each fold requires two cells x eight configs")
            if len({r["outer_sha256"] for r in fold_rows}) != 1 or len({r["train_seed"] for r in fold_rows}) != 1:
                raise RuntimeError("cells/configs must share outer test and training seed")
            for cell in CELLS:
                cr = [r for r in fold_rows if r["cell"] == cell]
                if sorted(int(r["config_index"]) for r in cr) != list(range(8)):
                    raise RuntimeError("each cell requires the complete eight-config grid")
        gr = split_docs[f"n14r__d{draw_id:02d}__GR_hpo"]
        gg = split_docs[f"n14r__d{draw_id:02d}__GG_hpo"]
        if [f["test"] for f in gr["folds"]] != [f["test"] for f in gg["folds"]]:
            raise RuntimeError("GR/GG test paths must match exactly by fold")
        for table, expected_overlap in ((gr, True), (gg, False)):
            for fold in table["folds"]:
                overlap = fold["meta"]["inner_overlap_speaker_count"] > 0
                if overlap != expected_overlap:
                    raise RuntimeError("inner speaker-overlap cell invariant failed")


def build(manifest: Path, out: Path) -> dict[str, Any]:
    rows, manifest_sha, hygiene = load_manifest(manifest)
    plan, split_docs = make_plan(rows, manifest_sha)
    hygiene_text = json.dumps(hygiene, indent=1, sort_keys=True) + "\n"
    atomic_text(out / "hygiene.json", hygiene_text)
    atomic_text(out / "run_plan.csv", _csv_text(PLAN_FIELDS, plan))
    split_index = {}
    for key, doc in split_docs.items():
        text = canonical_json(doc)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        atomic_text(out / "splits" / f"{key}.json", text)
        split_index[key] = {"sha256": sha, "path": f"splits/{key}.json", "n_folds": 5}
    split_index_text = json.dumps(split_index, indent=1, sort_keys=True) + "\n"
    atomic_text(out / "split_index.json", split_index_text)
    configs = {sha256_json(cfg): cfg for cfg in HPO_GRID}
    atomic_text(out / "unit_configs.json", json.dumps(configs, indent=1, sort_keys=True) + "\n")
    plan_text = _csv_text(PLAN_FIELDS, plan)
    # Hashes in the summary refer to exact file bytes, including their final LF.
    summary = {
        "study_id": STUDY_ID, "spec_version": SPEC_VERSION, "manifest_sha256": manifest_sha,
        "primary_draws": 24, "reserve_draws": 4, "units_per_draw": 80,
        "primary_units": 1920, "maximum_units": 2240,
        "primary_first_attempt_projected_gpu_hours": 92.9,
        "all_28_draws_first_attempt_projected_gpu_hours": 108.4,
        "all_28_draws_all_retries_projected_gpu_hours": 216.8,
        "soft_resource_allocation_gpu_hours": 125.0,
        "maximum_attempts_per_unit": 2,
        "theoretical_maximum_training_attempts": 4480,
        "hygiene_sha256": sha256_bytes(hygiene_text.encode("utf-8")),
        "manifest_input_rows": int(hygiene["n_input"]),
        "analysis_population_rows": int(hygiene["n_kept"]),
        "run_plan_sha256": sha256_bytes(plan_text.encode("utf-8")),
        "split_index_sha256": sha256_bytes(split_index_text.encode("utf-8")),
    }
    atomic_text(out / "plan_summary.json", json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.out), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

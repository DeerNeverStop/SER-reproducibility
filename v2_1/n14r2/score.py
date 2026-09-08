"""Independent scorer for the N14R2 outer-draw rerun.

This module intentionally does not import the v2 scorer or statistics module.  It
uses only frozen plan files and atomic raw unit outputs.  The independent verifier
is expected to reimplement the same contract without importing this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from scipy import stats as scipy_stats

from v2.ser_v2 import corpora as v2_corpora


SCHEMA = "ser-v2.1-n14r2-results-1"
EXPECTED_DRAWS = 24
N_FOLDS = 5
N_CLASSES = 6
GRID_INDICES = tuple(range(8))
CELLS = ("GR_hpo", "GG_hpo")
FIXED_CONFIG_INDEX = 4
ALPHA = 0.05
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 202_609_050_001
FLOAT_ATOL = 1e-10

# Duplicated here deliberately so the scorer is independent of executable plan
# code.  A changed plan/config hash cannot silently redefine the frozen grid.
HPO_GRID = tuple(
    {
        "engine": "p1_frozen",
        "model": "resnet_se",
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "batch_size": 64,
        "epochs": 100,
        "patience": 15,
        "hpo_config_index": i,
    }
    for i, (lr, weight_decay, dropout) in enumerate(
        (lr, weight_decay, dropout)
        for lr in (3e-4, 1e-3)
        for weight_decay in (1e-4, 1e-3)
        for dropout in (0.1, 0.3)
    )
)


class ScoreError(ValueError):
    """A unit or draw violates the frozen scoring contract."""


def _expected_analysis_draw_ids(void_ids: set[int]) -> list[int]:
    return [draw for draw in range(28) if draw not in void_ids][:EXPECTED_DRAWS]


@dataclass(frozen=True)
class Candidate:
    config_index: int
    val_uar: float
    test_uar: float
    unit_id: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def macro_uar_6(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Six-class macro recall in percentage points, with no absent-class skipping."""
    truth = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if truth.ndim != 1 or pred.ndim != 1 or truth.size != pred.size or truth.size == 0:
        raise ScoreError("y_true/y_pred must be equally sized, non-empty vectors")
    if np.any((truth < 0) | (truth >= N_CLASSES)) or np.any((pred < 0) | (pred >= N_CLASSES)):
        raise ScoreError("labels and predictions must be integers in 0..5")
    recalls: list[float] = []
    for label in range(N_CLASSES):
        mask = truth == label
        if not bool(mask.any()):
            raise ScoreError(f"class {label} is absent; six-class macro-UAR is undefined")
        recalls.append(float(np.mean(pred[mask] == label)))
    return 100.0 * float(np.mean(recalls))


def choose_config(candidates: Iterable[Candidate]) -> Candidate:
    """Select largest validation UAR, resolving exact ties by the lowest index."""
    values = list(candidates)
    indices = [c.config_index for c in values]
    if len(values) != 8 or sorted(indices) != list(GRID_INDICES) or len(set(indices)) != 8:
        raise ScoreError("an HPO episode must contain each config index 0..7 exactly once")
    if any(not math.isfinite(c.val_uar) for c in values):
        raise ScoreError("validation UAR must be finite")
    return min(values, key=lambda c: (-c.val_uar, c.config_index))


def one_sample_t_summary(values: Iterable[float], alpha: float = ALPHA) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ScoreError("one-sample t inference requires at least two finite draw values")
    n = int(x.size)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        se = 0.0
        if mean == 0.0:
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = math.copysign(math.inf, mean), 0.0
        ci = [mean, mean]
    else:
        se = sd / math.sqrt(n)
        t_stat = mean / se
        p_value = float(2.0 * scipy_stats.t.sf(abs(t_stat), n - 1))
        critical = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, n - 1))
        ci = [mean - critical * se, mean + critical * se]
    supported = bool(mean > 0.0 and p_value <= alpha)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "df": n - 1,
        "t": float(t_stat),
        "p_two_sided": p_value,
        "ci_level": 1.0 - alpha,
        "ci": [float(ci[0]), float(ci[1])],
        "direction": ">0",
        "alpha": alpha,
        "supported": supported,
        "verdict": "supported" if supported else "not supported",
    }


def bootstrap_mean_summary(
    values: Iterable[float], reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or reps <= 0:
        raise ScoreError("bootstrap requires finite draw values and a positive replication count")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(int(reps), x.size))
    means = x[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5], method="linear")
    return {
        "method": "draw bootstrap percentile",
        "seed": int(seed),
        "reps": int(reps),
        "mean": float(x.mean()),
        "ci_level": 0.95,
        "ci": [float(low), float(high)],
    }


def _all_sign_sums(x: np.ndarray) -> np.ndarray:
    sums = np.zeros(1, dtype=float)
    for value in x:
        sums = np.concatenate((sums + value, sums - value))
    return sums


def _abs_studentized_from_sums(sums: np.ndarray, square_sum: float, n: int) -> np.ndarray:
    # t = S / sqrt((n*sum(x^2)-S^2)/(n-1)); sum(x^2) is sign-invariant.
    variance_term = (n * square_sum - sums * sums) / (n - 1)
    tolerance = np.finfo(float).eps * max(1.0, n * square_sum) * 32.0
    variance_term = np.where((variance_term < 0.0) & (variance_term > -tolerance), 0.0, variance_term)
    out = np.empty_like(sums, dtype=float)
    regular = variance_term > 0.0
    out[regular] = np.abs(sums[regular]) / np.sqrt(variance_term[regular])
    degenerate = ~regular
    out[degenerate] = np.where(np.abs(sums[degenerate]) <= tolerance, 0.0, np.inf)
    return out


def exact_sign_flip_summary(values: Iterable[float]) -> dict[str, Any]:
    """Enumerate all 2^n sign flips for mean and studentized sensitivities.

    Meet-in-the-middle keeps peak memory small for the preregistered n=24 while
    still visiting every one of the 16,777,216 assignments exactly once.
    """
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or x.size == 0 or x.size > 24 or not np.all(np.isfinite(x)):
        raise ScoreError("exact sign flipping supports 1..24 finite draw values")
    n = int(x.size)
    split = n // 2
    left = _all_sign_sums(x[:split])
    right = _all_sign_sums(x[split:])
    observed_sum = float(x.sum())
    observed_mean_abs = abs(observed_sum / n)
    observed_t_abs = float(_abs_studentized_from_sums(np.asarray([observed_sum]), float(x @ x), n)[0]) if n > 1 else math.inf
    mean_count = 0
    student_count = 0
    square_sum = float(x @ x)
    mean_tol = np.finfo(float).eps * max(1.0, abs(observed_sum)) * 32.0
    t_tol = np.finfo(float).eps * max(1.0, observed_t_abs if math.isfinite(observed_t_abs) else 1.0) * 64.0
    # At n=24 this forms at most a 256 x 4096 matrix (~8 MiB) per chunk.
    rows_per_chunk = max(1, min(left.size, (1 << 20) // max(1, right.size)))
    for start in range(0, left.size, rows_per_chunk):
        sums = left[start : start + rows_per_chunk, None] + right[None, :]
        mean_count += int(np.count_nonzero(np.abs(sums) + mean_tol >= abs(observed_sum)))
        if n == 1:
            student_count += int(sums.size)
        else:
            permuted_t = _abs_studentized_from_sums(sums, square_sum, n)
            if math.isinf(observed_t_abs):
                student_count += int(np.count_nonzero(np.isinf(permuted_t)))
            else:
                student_count += int(np.count_nonzero(permuted_t + t_tol >= observed_t_abs))
    total = 1 << n
    return {
        "method": "complete exact sign-flip enumeration",
        "n": n,
        "n_permutations": total,
        "mean": {
            "statistic_abs": observed_mean_abs,
            "extreme_count": mean_count,
            "p_two_sided": float(mean_count / total),
        },
        "studentized": {
            "statistic_abs": observed_t_abs,
            "extreme_count": student_count,
            "p_two_sided": float(student_count / total),
        },
    }


def _normal_cell(value: str) -> str:
    aliases = {"GR": "GR_hpo", "GG": "GG_hpo", "GR_hpo": "GR_hpo", "GG_hpo": "GG_hpo"}
    try:
        return aliases[value]
    except KeyError as exc:
        raise ScoreError(f"unknown N14R2 cell {value!r}") from exc


def _int(row: dict[str, Any], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreError(f"invalid or missing integer field {key}") from exc


def _load_plan(plan_dir: Path) -> list[dict[str, str]]:
    path = plan_dir / "run_plan.csv"
    if not path.is_file():
        raise ScoreError(f"missing plan file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ScoreError("run_plan.csv is empty")
    required = {
        "study_id", "unit_id", "draw_id", "draw_role", "analysis_order", "fold", "train_rep",
        "cell", "config_index", "model", "corpus_level", "split_seed", "inner_seed", "train_seed",
        "outer_sha256", "inner_sha256", "config_sha256", "manifest_sha256", "spec_version", "n_fit",
        "n_val", "n_test", "status", "arm", "base_corpus", "panel_draw", "r", "seed_index",
        "split_sha256", "cap_group", "conditional", "truncation_rank",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ScoreError(f"run_plan.csv lacks required columns: {', '.join(missing)}")
    return rows


def _iter_sha_entries(value: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        sha = value.get("sha256")
        if isinstance(sha, str):
            yield sha, value
        for child in value.values():
            yield from _iter_sha_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_sha_entries(child)


class SplitStore:
    def __init__(self, plan_dir: Path, manifest: dict[str, dict[str, Any]]):
        self.plan_dir = plan_dir
        self.by_sha: dict[str, Any] = {}
        self.metadata_by_sha: dict[str, dict[str, Any]] = {}
        index_path = plan_dir / "split_index.json"
        self.index = _read_json(index_path) if index_path.is_file() else None
        indexed_file_hashes: dict[Path, str] = {}
        if self.index is not None:
            for sha, entry in _iter_sha_entries(self.index):
                raw_path = entry.get("path")
                if raw_path:
                    path = Path(str(raw_path))
                    if not path.is_absolute():
                        path = plan_dir / path
                    indexed_file_hashes[path.resolve()] = sha
        splits_dir = plan_dir / "splits"
        if not splits_dir.is_dir():
            raise ScoreError("missing splits directory")
        for path in sorted(splits_dir.glob("*.json")):
            resolved = path.resolve()
            if resolved in indexed_file_hashes and _sha256(path) != indexed_file_hashes[resolved]:
                raise ScoreError(f"split index file hash mismatch for {path.name}")
            doc = _read_json(path)
            if not isinstance(doc, dict) or not isinstance(doc.get("folds"), list):
                raise ScoreError(f"invalid split document {path.name}")
            doc_meta = doc.get("meta", doc)
            metadata = {key: doc_meta.get(key) for key in (
                "study_id", "spec_version", "manifest_sha256", "draw_id", "draw_role", "analysis_order", "split_seed"
            )}
            doc_sha = _canonical_sha256(doc)
            if resolved in indexed_file_hashes and doc_sha != indexed_file_hashes[resolved]:
                raise ScoreError(f"non-canonical or mismatched split bytes in {path.name}")
            self.by_sha[doc_sha] = doc
            self.metadata_by_sha[doc_sha] = metadata
            document_test_paths: list[str] = []
            for fold_doc in doc["folds"]:
                if not isinstance(fold_doc, dict):
                    raise ScoreError(f"invalid fold in {path.name}")
                fold_meta = fold_doc.get("meta", {})
                # New split documents carry full references.  Compatibility split
                # documents carry paths; reconstruct the committed references from
                # the manifest before validating the scientific partition hashes.
                def refs(items: Any) -> Any:
                    if not isinstance(items, list) or not items or isinstance(items[0], dict):
                        return items
                    rebuilt = []
                    for item in items:
                        if item not in manifest:
                            raise ScoreError(f"split path {item} is absent from the frozen manifest")
                        m = manifest[item]
                        rebuilt.append({
                            "sample_index": m["sample_index"], "relative_path": item,
                            "label_index": m["y_true"], "speaker": m["speaker"],
                        })
                    return rebuilt
                test_refs = refs(fold_doc.get("test"))
                fit_refs = refs(fold_doc.get("fit"))
                val_refs = refs(fold_doc.get("val"))
                outer_train = fold_doc.get("outer_train")
                if outer_train is None and fit_refs is not None and val_refs is not None:
                    outer_train = sorted([*fit_refs, *val_refs], key=lambda item: int(item["sample_index"]))
                outer_payload = {"outer_train": refs(outer_train), "test": test_refs}
                outer_train_paths = {item["relative_path"] for item in outer_payload["outer_train"]}
                test_paths = {item["relative_path"] for item in outer_payload["test"]}
                if outer_train_paths & test_paths or outer_train_paths | test_paths != set(manifest):
                    raise ScoreError(f"outer train/test is not a disjoint manifest partition in {path.name}")
                if ({item["label_index"] for item in outer_payload["outer_train"]} != set(range(N_CLASSES))
                        or {item["label_index"] for item in outer_payload["test"]} != set(range(N_CLASSES))):
                    raise ScoreError(f"outer split lacks a frozen class in {path.name}")
                document_test_paths.extend(item["relative_path"] for item in outer_payload["test"])
                outer_sha = _canonical_sha256(outer_payload)
                recorded_outer_sha = fold_doc.get("outer_sha256", fold_meta.get("outer_sha256"))
                if recorded_outer_sha != outer_sha:
                    raise ScoreError(f"outer partition hash mismatch in {path.name}")
                if outer_sha in self.by_sha and self.by_sha[outer_sha] != outer_payload:
                    raise ScoreError("two different outer partitions share a hash")
                self.by_sha[outer_sha] = outer_payload
                self.metadata_by_sha[outer_sha] = {**metadata, "fold": fold_doc.get("fold")}
                cells = fold_doc.get("cells")
                if not isinstance(cells, dict):
                    cell_name = fold_meta.get("cell", doc_meta.get("cell"))
                    cells = {cell_name: {"fit": fit_refs, "val": val_refs, **fold_meta}}
                for cell, cell_doc in cells.items():
                    inner_payload = {"fit": refs(cell_doc.get("fit")), "val": refs(cell_doc.get("val"))}
                    fit_paths = {item["relative_path"] for item in inner_payload["fit"]}
                    val_paths = {item["relative_path"] for item in inner_payload["val"]}
                    if fit_paths & val_paths or fit_paths | val_paths != outer_train_paths:
                        raise ScoreError(f"inner fit/val is not a disjoint outer-train partition in {path.name}")
                    if ({item["label_index"] for item in inner_payload["fit"]} != set(range(N_CLASSES))
                            or {item["label_index"] for item in inner_payload["val"]} != set(range(N_CLASSES))):
                        raise ScoreError(f"inner split lacks a frozen class in {path.name}")
                    if _normal_cell(str(cell)) == "GG_hpo":
                        fit_speakers = {item["speaker"] for item in inner_payload["fit"]}
                        val_speakers = {item["speaker"] for item in inner_payload["val"]}
                        if fit_speakers & val_speakers:
                            raise ScoreError(f"GG_hpo inner speakers overlap in {path.name}")
                    inner_sha = _canonical_sha256(inner_payload)
                    if cell_doc.get("inner_sha256", fold_meta.get("inner_sha256")) != inner_sha:
                        raise ScoreError(f"inner partition hash mismatch in {path.name}")
                    if inner_sha in self.by_sha and self.by_sha[inner_sha] != inner_payload:
                        raise ScoreError("two different inner partitions share a hash")
                    self.by_sha[inner_sha] = inner_payload
                    self.metadata_by_sha[inner_sha] = {
                        **metadata,
                        "fold": fold_doc.get("fold"),
                        "cell": cell,
                        "inner_seed": cell_doc.get("inner_seed", fold_meta.get("inner_seed")),
                    }
            population = doc.get("population")
            if isinstance(population, list) and (
                len(document_test_paths) != len(population) or set(document_test_paths) != set(population)
            ):
                raise ScoreError(f"outer test folds do not cover the population exactly once in {path.name}")

    def get(self, sha: str) -> Any:
        if sha not in self.by_sha:
            raise ScoreError(f"partition hash {sha} is absent from the frozen split documents")
        return self.by_sha[sha]

    def validate_metadata(self, sha: str, row: dict[str, str], *, inner: bool) -> None:
        meta = self.metadata_by_sha.get(sha, {})
        keys = ("study_id", "spec_version", "manifest_sha256", "draw_id", "draw_role", "analysis_order", "split_seed", "fold")
        for key in keys:
            if key not in meta or not _same_field(row[key], meta[key], key):
                raise ScoreError(f"{row['unit_id']}: split/plan metadata mismatch for {key}")
        if inner:
            if _normal_cell(str(meta.get("cell"))) != _normal_cell(row["cell"]):
                raise ScoreError(f"{row['unit_id']}: inner split cell mismatch")
            if int(meta.get("inner_seed", -1)) != int(row["inner_seed"]):
                raise ScoreError(f"{row['unit_id']}: inner split seed mismatch")


def _fold_scope(value: Any, fold: int) -> Any:
    if isinstance(value, dict) and "folds" in value:
        folds = value["folds"]
        if isinstance(folds, list):
            for item in folds:
                if isinstance(item, dict) and int(item.get("fold", item.get("fold_id", -1))) == fold:
                    return item
        if isinstance(folds, dict):
            for key in (str(fold), f"fold_{fold}", f"f{fold}"):
                if key in folds:
                    return folds[key]
    if isinstance(value, dict):
        for key in (str(fold), f"fold_{fold}", f"f{fold}"):
            if key in value and isinstance(value[key], (dict, list)):
                return value[key]
    return value


def _cell_scope(value: Any, cell: str) -> Any:
    if not isinstance(value, dict):
        return value
    aliases = (cell, cell.removesuffix("_hpo"))
    for container in (value.get("cells"), value.get("inner"), value):
        if isinstance(container, dict):
            for key in aliases:
                if key in container:
                    return container[key]
    return value


def _partition_payload(value: Any, partition: str, fold: int, cell: str) -> Any:
    scoped = _cell_scope(_fold_scope(value, fold), cell)
    scoped = _fold_scope(_cell_scope(scoped, cell), fold)
    if not isinstance(scoped, dict):
        raise ScoreError(f"cannot locate {partition} partition in split")
    aliases = ("test", "outer_test") if partition == "test" else ("val", "validation", "inner_val")
    for key in aliases:
        if key in scoped:
            return scoped[key]
    # Parallel arrays directly on the fold/cell object.
    prefix = "test" if partition == "test" else "val"
    paths = scoped.get(f"{prefix}_relative_paths", scoped.get(f"{prefix}_paths"))
    indices = scoped.get(f"{prefix}_indices", scoped.get(f"{prefix}_sample_indices"))
    labels = scoped.get(f"{prefix}_labels", scoped.get(f"{prefix}_y_true"))
    if paths is not None:
        return {"relative_paths": paths, "sample_indices": indices, "y_true": labels}
    raise ScoreError(f"cannot locate {partition} partition in split")


def _expected_rows(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict) and any(k in payload for k in ("relative_paths", "paths")):
        paths = payload.get("relative_paths", payload.get("paths"))
        indices = payload.get("sample_indices", payload.get("indices"))
        labels = payload.get("y_true", payload.get("labels", payload.get("label_index")))
        if not isinstance(paths, list):
            raise ScoreError("split relative_paths must be a list")
        if indices is not None and len(indices) != len(paths):
            raise ScoreError("split path/index arrays differ in length")
        if labels is not None and isinstance(labels, list) and len(labels) != len(paths):
            raise ScoreError("split path/label arrays differ in length")
        payload = [
            {
                "relative_path": path,
                "sample_index": None if indices is None else indices[i],
                "y_true": None if labels is None or not isinstance(labels, list) else labels[i],
            }
            for i, path in enumerate(paths)
        ]
    if not isinstance(payload, list):
        raise ScoreError("split partition must be a list or parallel-array object")
    out: dict[str, dict[str, Any]] = {}
    for item in payload:
        if isinstance(item, str):
            path, index, label = item, None, None
        elif isinstance(item, dict):
            path = item.get("relative_path", item.get("path"))
            index = item.get("sample_index", item.get("index"))
            label = item.get("y_true", item.get("label_index", item.get("label")))
        else:
            raise ScoreError("split rows must include relative paths")
        if path is None or str(path) in out:
            raise ScoreError("split partition has a missing or duplicate relative path")
        out[str(path)] = {"sample_index": None if index is None else int(index), "y_true": None if label is None else int(label)}
    if not out:
        raise ScoreError("split partition is empty")
    return out


def _compatible_expected_rows(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> bool:
    if set(left) != set(right):
        return False
    for path in left:
        for field in ("sample_index", "y_true"):
            a, b = left[path].get(field), right[path].get(field)
            if a is not None and b is not None and a != b:
                return False
    return True


def _load_hygienic_manifest_file(path: Path, expected_sha: str) -> dict[str, dict[str, Any]]:
    """Load and clean CREMA-D exactly as the reused trainer/plan do.

    ``expected_sha`` commits the original 7,442-row manifest bytes.  H1/H2/H3
    hygiene then deterministically derives and renumbers the 7,435-row scoring
    population; split sample indices refer to this derived population.
    """
    if not path.is_file() or _sha256(path) != expected_sha:
        raise ScoreError("manifest is missing or differs from the frozen raw-manifest hash")
    try:
        raw_rows = v2_corpora.load_manifest(path)
        kept, hygiene = v2_corpora.apply_hygiene("cremad", raw_rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreError(f"failed to apply frozen CREMA-D hygiene: {exc}") from exc
    if (int(hygiene.get("n_input", -1)) != 7442 or int(hygiene.get("n_kept", -1)) != 7435
            or int(hygiene.get("n_dropped", -1)) != 7):
        raise ScoreError("CREMA-D hygiene counts differ from the frozen 7442 -> 7435 contract")
    indices = [int(row["sample_index"]) for row in kept]
    if indices != list(range(7435)) or {int(row["label_index"]) for row in kept} != set(range(N_CLASSES)):
        raise ScoreError("hygienic manifest indices/classes differ from the frozen contract")
    paths = [str(row["relative_path"]) for row in kept]
    if len(set(paths)) != len(paths):
        raise ScoreError("hygienic manifest contains duplicate relative paths")
    return {
        str(row["relative_path"]): {
            "sample_index": int(row["sample_index"]),
            "y_true": int(row["label_index"]),
            "speaker": str(row["speaker"]),
        }
        for row in kept
    }


def _load_manifest(plan_dir: Path, expected_sha: str) -> dict[str, dict[str, Any]]:
    candidates: list[Path] = []
    for ancestor in (plan_dir, *plan_dir.parents):
        candidates.extend(
            [
                ancestor / "manifest.csv",
                ancestor / "manifests" / "cremad_manifest.csv",
                ancestor / "v2" / "manifests" / "cremad_manifest.csv",
            ]
        )
    for path in candidates:
        if path.is_file() and _sha256(path) == expected_sha:
            return _load_hygienic_manifest_file(path, expected_sha)
    return {}


def _validate_plan_hygiene(
    plan_dir: Path, manifest: dict[str, dict[str, Any]], expected_manifest_sha: str,
) -> None:
    hygiene_path = plan_dir / "hygiene.json"
    summary_path = plan_dir / "plan_summary.json"
    if not hygiene_path.is_file() or not summary_path.is_file():
        raise ScoreError("plan hygiene.json/plan_summary.json is missing")
    hygiene = _read_json(hygiene_path)
    summary = _read_json(summary_path)
    if (summary.get("hygiene_sha256") != _sha256(hygiene_path)
            or int(summary.get("manifest_input_rows", -1)) != 7442
            or int(summary.get("analysis_population_rows", -1)) != 7435
            or summary.get("manifest_sha256") != expected_manifest_sha):
        raise ScoreError("plan_summary hygiene hash/count/manifest commitment is invalid")
    ordered_paths = [
        path for path, _ in sorted(manifest.items(), key=lambda item: int(item[1]["sample_index"]))
    ]
    expected_drops = {
        "1006_TIE_HAP_XX.wav", "1006_TIE_NEU_XX.wav",
        "1013_WSI_DIS_XX.wav", "1013_WSI_SAD_XX.wav",
        "1017_IWW_ANG_XX.wav", "1017_IWW_FEA_XX.wav",
        "1076_MTI_SAD_XX.wav",
    }
    dropped = hygiene.get("dropped")
    if (hygiene.get("schema") != "ser26-n14r2-hygiene-1"
            or hygiene.get("manifest_sha256") != expected_manifest_sha
            or int(hygiene.get("n_input", -1)) != 7442
            or int(hygiene.get("n_kept", -1)) != 7435
            or int(hygiene.get("n_dropped", -1)) != 7
            or not isinstance(dropped, list)
            or {item.get("relative_path") for item in dropped if isinstance(item, dict)} != expected_drops
            or len(manifest) != 7435):
        raise ScoreError("plan hygiene record differs from the independently replayed population")
    if len(ordered_paths) != 7435:
        raise ScoreError("hygienic manifest ordering is incomplete")


def _unit_paths(units_dir: Path, unit_id: str) -> dict[str, Path]:
    nested = units_dir / unit_id
    if nested.is_dir():
        return {name: nested / name for name in ("unit.json", "history.json", "predictions.csv", "DONE")}
    # Read-only compatibility for early flat fixtures; execution uses nested units.
    return {
        "unit.json": units_dir / f"{unit_id}.json",
        "history.json": units_dir / f"{unit_id}.history.json",
        "predictions.csv": units_dir / f"{unit_id}.predictions.csv",
        "DONE": units_dir / f"{unit_id}.DONE",
    }


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = ["sample_index", "relative_path", "speaker", "y_true", "y_pred", *[f"logit_{i}" for i in range(6)]]
        if reader.fieldnames != expected_fields:
            raise ScoreError(f"{path.name} does not have the exact frozen prediction columns")
        rows = list(reader)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row["relative_path"]
        if key in seen:
            raise ScoreError(f"{path.name} has duplicate prediction row {key}")
        seen.add(key)
        record = (
            {
                "sample_index": int(row["sample_index"]),
                "relative_path": row["relative_path"],
                "speaker": row["speaker"],
                "y_true": int(row["y_true"]),
                "y_pred": int(row["y_pred"]),
            }
        )
        logits = []
        for label in range(N_CLASSES):
            key_name = f"logit_{label}"
            if key_name not in row:
                raise ScoreError(f"{path.name} lacks {key_name}")
            value = float(row[key_name])
            if not math.isfinite(value):
                raise ScoreError(f"{path.name} contains non-finite logits")
            logits.append(value)
        if int(np.argmax(np.asarray(logits, dtype=float))) != record["y_pred"]:
            raise ScoreError(f"{path.name}: y_pred does not equal lowest-index argmax(logits)")
        out.append(record)
    return out


def _validate_partition(
    actual: list[dict[str, Any]], expected: dict[str, dict[str, Any]], manifest: dict[str, dict[str, int]], name: str
) -> tuple[np.ndarray, np.ndarray]:
    by_path = {row["relative_path"]: row for row in actual}
    if set(by_path) != set(expected) or len(actual) != len(expected):
        raise ScoreError(f"{name} predictions are not an exact one-to-one cover of the frozen split")
    truth: list[int] = []
    pred: list[int] = []
    for path in sorted(expected):
        row, frozen = by_path[path], expected[path]
        if frozen["sample_index"] is not None and row["sample_index"] != frozen["sample_index"]:
            raise ScoreError(f"{name} sample_index mismatch for {path}")
        reference = frozen["y_true"]
        if path in manifest:
            if frozen["sample_index"] is not None and manifest[path]["sample_index"] != frozen["sample_index"]:
                raise ScoreError(f"manifest/split sample_index mismatch for {path}")
            if reference is not None and manifest[path]["y_true"] != reference:
                raise ScoreError(f"manifest/split label mismatch for {path}")
            reference = manifest[path]["y_true"]
        if reference is None:
            raise ScoreError(f"no frozen label source is available for {path}")
        if row["y_true"] != reference:
            raise ScoreError(f"{name} y_true mismatch for {path}")
        if path in manifest and row.get("speaker") != manifest[path]["speaker"]:
            raise ScoreError(f"{name} speaker mismatch for {path}")
        truth.append(row["y_true"])
        pred.append(row["y_pred"])
    return np.asarray(truth, dtype=int), np.asarray(pred, dtype=int)


IDENTITY_FIELDS = (
    "unit_id", "arm", "corpus_level", "model", "cell", "fold", "r", "seed_index",
    "train_seed", "config_sha256", "split_sha256", "n_fit", "n_val", "n_test",
)
INTEGER_FIELDS = {
    "draw_id", "fold", "train_rep", "config_index", "split_seed", "inner_seed", "train_seed",
    "n_fit", "n_val", "n_test", "analysis_order", "r", "seed_index", "conditional", "truncation_rank",
}


def _same_field(plan_value: Any, unit_value: Any, key: str) -> bool:
    if key in INTEGER_FIELDS:
        try:
            return int(plan_value) == int(unit_value)
        except (TypeError, ValueError):
            return False
    if key == "cell":
        try:
            return _normal_cell(str(plan_value)) == _normal_cell(str(unit_value))
        except ScoreError:
            return False
    return str(plan_value) == str(unit_value)


def _load_candidate(
    row: dict[str, str], unit_dir: Path, splits: SplitStore, manifest: dict[str, dict[str, Any]],
    configs: dict[str, dict[str, Any]],
) -> Candidate:
    unit_id = row["unit_id"]
    paths = {name: unit_dir / name for name in ("unit.json", "history.json", "predictions.csv", "DONE")}
    unit_path, history_path, pred_path, done_path = (
        paths["unit.json"], paths["history.json"], paths["predictions.csv"], paths["DONE"]
    )
    if not all(path.is_file() for path in (unit_path, history_path, pred_path, done_path)):
        raise ScoreError(f"{unit_id}: incomplete unit/history/prediction/DONE artifact set")
    unit = _read_json(unit_path)
    for key in IDENTITY_FIELDS:
        if key not in unit or not _same_field(row[key], unit[key], key):
            raise ScoreError(f"{unit_id}: unit/plan mismatch for {key}")
    if unit.get("status") != "done":
        raise ScoreError(f"{unit_id}: unit status is not done")
    prediction_sha = _sha256(pred_path)
    if done_path.read_text(encoding="utf-8").strip() != prediction_sha or unit.get("predictions_sha256") != prediction_sha:
        raise ScoreError(f"{unit_id}: predictions hash mismatch")
    expected_config = configs.get(row["config_sha256"])
    if expected_config is None or _canonical_sha256(expected_config) != row["config_sha256"]:
        raise ScoreError(f"{unit_id}: missing or invalid frozen config")
    if unit.get("config") != expected_config:
        raise ScoreError(f"{unit_id}: embedded config differs from the frozen plan config")
    if int(expected_config.get("hpo_config_index", -1)) != _int(row, "config_index"):
        raise ScoreError(f"{unit_id}: config index differs from frozen config payload")
    outer = splits.get(row["outer_sha256"])
    inner = splits.get(row["inner_sha256"])
    splits.validate_metadata(row["outer_sha256"], row, inner=False)
    splits.validate_metadata(row["inner_sha256"], row, inner=True)
    split_doc = splits.get(row["split_sha256"])
    fold = _int(row, "fold")
    cell = _normal_cell(row["cell"])
    expected_test = _expected_rows(_partition_payload(outer, "test", fold, cell))
    expected_val = _expected_rows(_partition_payload(inner, "val", fold, cell))
    expected_fit = _expected_rows(inner["fit"])
    # The compatibility split table is itself committed by split_sha256; ensure
    # the row points to the same fold and cell payload.
    split_test = _expected_rows(_partition_payload(split_doc, "test", fold, cell))
    split_val = _expected_rows(_partition_payload(split_doc, "val", fold, cell))
    if (not _compatible_expected_rows(split_test, expected_test)
            or not _compatible_expected_rows(split_val, expected_val)):
        raise ScoreError(f"{unit_id}: compatibility/full partition commitments disagree")
    predictions = _read_predictions(pred_path)
    test_true, test_pred = _validate_partition(predictions, expected_test, manifest, f"{unit_id} test")
    if (len(expected_fit) != _int(row, "n_fit") or len(expected_val) != _int(row, "n_val")
            or len(test_true) != _int(row, "n_test")):
        raise ScoreError(f"{unit_id}: prediction counts differ from plan")
    test_uar = macro_uar_6(test_true, test_pred)
    history = _read_json(history_path)
    if not isinstance(history, list) or not history:
        raise ScoreError(f"{unit_id}: history must be a non-empty epoch list")
    try:
        losses = np.asarray([float(item["val_loss"]) for item in history], dtype=float)
        val_uars = np.asarray([float(item["val_uar"]) for item in history], dtype=float)
        epochs = [int(item["epoch"]) for item in history]
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreError(f"{unit_id}: malformed validation history") from exc
    if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(val_uars)):
        raise ScoreError(f"{unit_id}: non-finite validation history")
    if epochs != list(range(1, len(history) + 1)) or int(unit.get("epochs_run", -1)) != len(history):
        raise ScoreError(f"{unit_id}: history epoch sequence/epochs_run mismatch")
    best_index = int(np.argmin(losses))  # np.argmin gives the frozen earliest-tie rule.
    if int(unit.get("best_epoch", -1)) != epochs[best_index]:
        raise ScoreError(f"{unit_id}: best_epoch is not the first minimum-val-loss epoch")
    if not math.isclose(float(unit.get("val_loss_best")), float(losses[best_index]), rel_tol=0.0, abs_tol=FLOAT_ATOL):
        raise ScoreError(f"{unit_id}: val_loss_best differs from history")
    val_uar = float(val_uars[best_index])
    if not math.isclose(float(unit.get("val_uar_best")), val_uar, rel_tol=0.0, abs_tol=FLOAT_ATOL):
        raise ScoreError(f"{unit_id}: val_uar_best differs from the selected history epoch")
    if not (0.0 <= val_uar <= 100.0):
        raise ScoreError(f"{unit_id}: val_uar_best is outside 0..100")
    return Candidate(_int(row, "config_index"), val_uar, test_uar, unit_id)


def _not_tested(reason: str, n: int = 0, values: list[float] | None = None) -> dict[str, Any]:
    return {
        "tested": False,
        "n": int(n),
        "values": [] if values is None else values,
        "mean": None,
        "sd": None,
        "se": None,
        "df": None,
        "t": None,
        "p_two_sided": None,
        "ci_level": 0.95,
        "ci": [None, None],
        "direction": ">0",
        "alpha": ALPHA,
        "supported": False,
        "verdict": "not tested",
        "reason": reason,
    }


def _artifact_hashes(unit_dir: Path) -> dict[str, str]:
    return {name: _sha256(unit_dir / name) for name in ("unit.json", "history.json", "predictions.csv", "DONE")}


def _safe_locked_unit_dir(
    run_dir: Path, row: dict[str, str], entry: Any,
) -> tuple[Path, dict[str, str]]:
    if not isinstance(entry, dict) or entry.get("status") != "done":
        raise ScoreError(f"{row['unit_id']}: locked successful-attempt entry is malformed")
    try:
        attempt = int(entry["attempt"])
        raw_path = str(entry["attempt_path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreError(f"{row['unit_id']}: locked attempt identity is malformed") from exc
    expected = (
        Path("nodes") / f"node{int(row['draw_id']) % 4}" / "units" / row["unit_id"]
        / f"attempt_{attempt:02d}"
    )
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or ".." in pure.parts or Path(*pure.parts) != expected:
        raise ScoreError(f"{row['unit_id']}: locked attempt path is unsafe or misrouted")
    unit_dir = run_dir.joinpath(*pure.parts)
    artifacts = entry.get("artifact_hashes")
    if not isinstance(artifacts, dict) or set(artifacts) != {"unit.json", "history.json", "predictions.csv", "DONE"}:
        raise ScoreError(f"{row['unit_id']}: locked artifact mapping is malformed")
    receipt = unit_dir / "artifact_receipt.json"
    if not receipt.is_file() or _sha256(receipt) != entry.get("artifact_receipt_sha256"):
        raise ScoreError(f"{row['unit_id']}: artifact receipt changed after closure")
    observed = {name: _sha256(unit_dir / name) for name in artifacts}
    if observed != artifacts:
        raise ScoreError(f"{row['unit_id']}: outcome artifact changed after closure")
    return unit_dir, artifacts


def _validate_completion_lock(
    plan_dir: Path, run_dir: Path, rows: list[dict[str, str]],
) -> tuple[list[int], dict[str, Any], dict[str, Any] | None]:
    completion_path = run_dir / "completion.json"
    if not completion_path.is_file():
        raise ScoreError("completion.json is missing; outcomes must not be parsed before the run is closed")
    completion = _read_json(completion_path)
    if completion.get("schema") != "ser26-n14r2-completion-1" or completion.get("study_id") != "N14R2":
        raise ScoreError("completion.json study identity mismatch")
    if completion.get("contains_outcome_statistics") is not False:
        raise ScoreError("completion receipt must explicitly exclude outcome statistics")
    if completion.get("status") != "complete":
        if (run_dir / "analysis_lock.json").exists() or completion.get("analysis_lock_payload_sha256") is not None:
            raise ScoreError("incomplete N14R2 execution must not have an analysis lock")
        return [], {"completion_status": completion.get("status"), "completion": completion}, None
    lock_path = run_dir / "analysis_lock.json"
    if not lock_path.is_file():
        raise ScoreError("complete run lacks analysis_lock.json")
    lock = _read_json(lock_path)
    if lock.get("schema") != "ser26-n14r2-analysis-lock-1" or lock.get("study_id") != "N14R2":
        raise ScoreError("analysis lock schema/study mismatch")
    if lock.get("contains_outcome_statistics") is not False:
        raise ScoreError("analysis lock must explicitly exclude outcome statistics")
    check_payload = dict(lock)
    recorded_payload_sha = check_payload.pop("lock_payload_sha256", None)
    if recorded_payload_sha != _canonical_sha256(check_payload):
        raise ScoreError("analysis lock self-hash mismatch")
    if (completion.get("analysis_lock_payload_sha256") != recorded_payload_sha
            or completion.get("analysis_lock_sha256") != _sha256(lock_path)):
        raise ScoreError("completion does not commit the exact analysis lock")
    draw_ids = [int(value) for value in lock.get("complete_draw_ids", [])]
    completion_ids = [int(value) for value in completion.get("complete_draw_ids", [])]
    if draw_ids != completion_ids or len(draw_ids) != EXPECTED_DRAWS or len(set(draw_ids)) != EXPECTED_DRAWS:
        raise ScoreError("completion/analysis lock must agree on 24 unique draw IDs")
    order = {int(row["draw_id"]): int(row["analysis_order"]) for row in rows}
    if draw_ids != sorted(draw_ids, key=lambda draw: order.get(draw, 10**9)):
        raise ScoreError("locked draw IDs are not in frozen analysis order")
    selected = [row for row in rows if int(row["draw_id"]) in set(draw_ids)]
    if len(selected) != EXPECTED_DRAWS * N_FOLDS * len(CELLS) * len(GRID_INDICES):
        raise ScoreError("analysis lock does not select exactly 1,920 planned units")
    if int(lock.get("n_complete_draws", -1)) != EXPECTED_DRAWS or int(lock.get("n_locked_units", -1)) != len(selected):
        raise ScoreError("analysis lock completion/unit counts are invalid")
    expected_plan_hashes = lock.get("plan_hashes")
    if not isinstance(expected_plan_hashes, dict):
        raise ScoreError("analysis lock lacks plan hashes")
    for name in ("run_plan.csv", "split_index.json", "unit_configs.json", "hygiene.json", "plan_summary.json"):
        path = plan_dir / name
        if not path.is_file() or expected_plan_hashes.get(name) != _sha256(path):
            raise ScoreError(f"analysis lock plan hash mismatch for {name}")
    preregistration_hashes = lock.get("preregistration_hashes")
    if not isinstance(preregistration_hashes, dict):
        raise ScoreError("analysis lock lacks preregistration hashes")
    for name in ("spec.json", "PINS.json", "ENVIRONMENT_BINDINGS.json"):
        path = plan_dir.parent / name
        if not path.is_file() or preregistration_hashes.get(name) != _sha256(path):
            raise ScoreError(f"analysis lock preregistration hash mismatch for {name}")
    pins = _read_json(plan_dir.parent / "PINS.json")
    pins_payload = {key: value for key, value in pins.items() if key != "content_manifest_sha256"}
    if pins.get("schema") != "ser26-n14r2-pins-1" or pins.get("content_manifest_sha256") != _canonical_sha256(pins_payload):
        raise ScoreError("PINS.json content manifest is invalid")
    repo = plan_dir.parents[2]
    pinned_files = pins.get("files")
    if not isinstance(pinned_files, dict) or not pinned_files:
        raise ScoreError("PINS.json has no source files")
    for relative, expected_sha in pinned_files.items():
        path = repo / relative
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ScoreError(f"pinned source changed after freeze: {relative}")
    bindings_path = plan_dir.parent / "ENVIRONMENT_BINDINGS.json"
    bindings = _read_json(bindings_path)
    if (pins.get("environment_bindings") != bindings
            or pins.get("environment_bindings_sha256") != _sha256(bindings_path)
            or lock.get("environment_bindings") != bindings):
        raise ScoreError("prospective environment bindings changed")
    environment_hashes = lock.get("environment_receipt_hashes")
    node_ledgers = lock.get("node_ledger_hashes")
    node_ids = {f"node{i}" for i in range(4)}
    if (not isinstance(environment_hashes, dict) or set(environment_hashes) != node_ids
            or not isinstance(node_ledgers, dict) or set(node_ledgers) != node_ids):
        raise ScoreError("analysis lock lacks the four node environment/ledger hashes")
    for node_id in sorted(node_ids):
        node_dir = run_dir / "nodes" / node_id
        env = node_dir / "execution_environment.json"
        ledger = node_dir / "ledger.jsonl"
        binding = bindings.get("nodes", {}).get(node_id, {})
        if (not env.is_file() or _sha256(env) != environment_hashes[node_id]
                or environment_hashes[node_id] != binding.get("environment_sha256")
                or not ledger.is_file() or _sha256(ledger) != node_ledgers[node_id]):
            raise ScoreError(f"locked node environment/ledger changed: {node_id}")
    global_ledger = run_dir / "global_ledger.jsonl"
    if (not global_ledger.is_file()
            or _sha256(global_ledger) != lock.get("global_ledger_sha256")
            or completion.get("global_ledger_sha256") != lock.get("global_ledger_sha256")):
        raise ScoreError("locked global authorization ledger changed")
    artifacts = lock.get("unit_artifact_hashes")
    if not isinstance(artifacts, dict) or set(artifacts) != {row["unit_id"] for row in selected}:
        raise ScoreError("analysis lock unit artifact set mismatch")
    for row in selected:
        _safe_locked_unit_dir(run_dir, row, artifacts[row["unit_id"]])
    return draw_ids, {
        "completion_status": "complete",
        "analysis_lock_sha256": _sha256(lock_path),
        "run_plan_sha256": _sha256(plan_dir / "run_plan.csv"),
        "split_index_sha256": _sha256(plan_dir / "split_index.json"),
        "unit_configs_sha256": _sha256(plan_dir / "unit_configs.json"),
        "manifest_sha256": rows[0]["manifest_sha256"],
    }, lock


def score_run(
    plan_dir: str | Path,
    run_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    compute_exact_sign_flip: bool = True,
) -> dict[str, Any]:
    plan_dir, run_dir = Path(plan_dir), Path(run_dir)
    rows = _load_plan(plan_dir)
    plan_ids = [row["unit_id"] for row in rows]
    counts: dict[str, int] = defaultdict(int)
    for unit_id in plan_ids:
        counts[unit_id] += 1
    duplicate_ids = {unit_id for unit_id, count in counts.items() if count > 1}
    expected_sha_values = {row["manifest_sha256"] for row in rows}
    configs_path = plan_dir / "unit_configs.json"
    if not configs_path.is_file():
        raise ScoreError("missing unit_configs.json")
    configs = _read_json(configs_path)
    if not isinstance(configs, dict):
        raise ScoreError("unit_configs.json must be an object keyed by hash")
    by_draw: dict[int, list[dict[str, str]]] = defaultdict(list)
    draw_order: dict[int, int] = {}
    draw_role: dict[int, str] = {}
    global_failures: list[str] = []
    if duplicate_ids:
        global_failures.append(f"duplicate unit_id values: {sorted(duplicate_ids)}")
    if len(expected_sha_values) != 1:
        global_failures.append("plan contains more than one manifest_sha256")
    if len(rows) != 28 * N_FOLDS * len(CELLS) * len(GRID_INDICES):
        global_failures.append("plan must contain exactly 2,240 rows")
    for row in rows:
        try:
            draw = _int(row, "draw_id")
            by_draw[draw].append(row)
            order = _int(row, "analysis_order")
            expected_role = "primary" if draw < 24 else "reserve"
            if (row["study_id"] != "N14R2" or row["spec_version"] != "2.0.0" or row["arm"] != "N14R2"
                    or row["model"] != "resnet_se" or row["corpus_level"] != "cremad"
                    or row["base_corpus"] != "cremad" or row["status"] != "pending"
                    or order != draw or row["draw_role"] != expected_role or _int(row, "train_rep") != 0):
                global_failures.append(f"{row['unit_id']}: frozen plan identity/design invariant changed")
            if draw in draw_order and draw_order[draw] != order:
                global_failures.append(f"draw {draw}: inconsistent analysis_order")
            draw_order[draw] = order
            if draw in draw_role and draw_role[draw] != row["draw_role"]:
                global_failures.append(f"draw {draw}: inconsistent draw_role")
            draw_role[draw] = row["draw_role"]
            if _int(row, "r") != draw or _int(row, "seed_index") != _int(row, "train_rep"):
                global_failures.append(f"{row['unit_id']}: legacy r/seed_index mirrors disagree")
            config_index = _int(row, "config_index")
            if config_index not in GRID_INDICES:
                raise ScoreError(f"{row['unit_id']}: config_index is outside 0..7")
            expected_config = configs.get(row["config_sha256"])
            if expected_config is None or _canonical_sha256(expected_config) != row["config_sha256"]:
                global_failures.append(f"{row['unit_id']}: plan config hash is invalid")
            elif expected_config != HPO_GRID[config_index]:
                global_failures.append(f"{row['unit_id']}: plan config differs from frozen grid index {config_index}")
            scientific: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"unit_id", "est_gpu_sec", "status"}:
                    continue
                scientific[key] = int(value) if key in INTEGER_FIELDS else value
            scientific["seed_namespace"] = "SER26-N14R2"
            scientific["config"] = expected_config
            if expected_config is not None and _canonical_sha256(scientific) != row["unit_id"]:
                global_failures.append(f"{row['unit_id']}: unit_id does not commit the scientific plan row")
        except ScoreError as exc:
            global_failures.append(str(exc))

    if set(by_draw) != set(range(28)):
        global_failures.append("plan draw IDs must be exactly 0..27")
    if global_failures:
        raise ScoreError("; ".join(dict.fromkeys(global_failures)))

    locked_draw_ids, source_hashes, analysis_lock = _validate_completion_lock(plan_dir, run_dir, rows)
    if not locked_draw_ids:
        reason = f"run completion status is {source_hashes.get('completion_status')!r}; N14R2 is not tested"
        completion_meta = source_hashes.get("completion") or {}
        incomplete_n = int(completion_meta.get("n_complete_draws", 0))
        result = {
            "schema": SCHEMA, "study_id": rows[0]["study_id"], "spec_version": rows[0]["spec_version"],
            "expected_complete_draws": EXPECTED_DRAWS, "n_planned_draws": 28, "n_complete_draws": incomplete_n,
            "analysis_draw_ids": [], "tested": False, "status": "not tested", "estimand": None,
            "primary": _not_tested(reason, incomplete_n), "bootstrap_sensitivity": None,
            "exact_sign_flip_sensitivity": None,
            "D09R2": {"descriptive_only": True, "n": 0, "values": [], "mean": None, "sd": None,
                     "ci_level": 0.95, "ci": [None, None], "reason": reason},
            "draws": {}, "integrity": {"pass": True, "global_failures": [], "void_draw_ids": []},
            "source_hashes": source_hashes,
        }
        if output_path is not None:
            _atomic_result(Path(output_path), result)
        return result

    rows = [row for row in rows if int(row["draw_id"]) in set(locked_draw_ids)]
    by_draw = defaultdict(list)
    draw_order = {}
    draw_role = {}
    for row in rows:
        draw = int(row["draw_id"])
        by_draw[draw].append(row)
        draw_order[draw] = int(row["analysis_order"])
        draw_role[draw] = row["draw_role"]
    expected_manifest_sha = next(iter(expected_sha_values))
    if manifest_path is None:
        manifest = _load_manifest(plan_dir, expected_manifest_sha)
    else:
        manifest_file = Path(manifest_path)
        if manifest_file.is_dir():
            manifest_file = manifest_file / "cremad_manifest.csv"
        manifest = _load_hygienic_manifest_file(manifest_file, expected_manifest_sha)
    if not manifest:
        raise ScoreError("the frozen CREMA-D manifest could not be located; pass --manifest explicitly")
    _validate_plan_hygiene(plan_dir, manifest, expected_manifest_sha)
    splits = SplitStore(plan_dir, manifest)

    draw_results: dict[str, dict[str, Any]] = {}
    complete: list[tuple[int, int, float, float]] = []
    for draw in sorted(by_draw, key=lambda d: (draw_order.get(d, 10**9), d)):
        failures = list(global_failures) if global_failures else []
        draw_rows = by_draw[draw]
        episodes: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
        reps_by_fold: dict[int, set[int]] = defaultdict(set)
        for row in draw_rows:
            try:
                fold, rep, cell = _int(row, "fold"), _int(row, "train_rep"), _normal_cell(row["cell"])
                episodes[(fold, rep, cell)].append(row)
                reps_by_fold[fold].add(rep)
                if row["unit_id"] in duplicate_ids:
                    failures.append(f"{row['unit_id']}: duplicated in plan")
            except ScoreError as exc:
                failures.append(str(exc))
        if set(reps_by_fold) != set(range(N_FOLDS)):
            failures.append(f"draw {draw}: folds must be exactly 0..4")
        rep_sets = list(reps_by_fold.values())
        if not rep_sets or any(not reps for reps in rep_sets) or any(reps != rep_sets[0] for reps in rep_sets[1:]):
            failures.append(f"draw {draw}: train_rep set differs across folds")
        expected_episode_keys = {
            (fold, rep, cell) for fold in range(N_FOLDS) for rep in (rep_sets[0] if rep_sets else set()) for cell in CELLS
        }
        if set(episodes) != expected_episode_keys:
            failures.append(f"draw {draw}: episode grid is incomplete or contains extras")
        for fold in range(N_FOLDS):
            for rep in (rep_sets[0] if rep_sets else set()):
                shared_rows = [row for cell in CELLS for row in episodes.get((fold, rep, cell), [])]
                if shared_rows and len({row["outer_sha256"] for row in shared_rows}) != 1:
                    failures.append(f"draw {draw} fold {fold} rep {rep}: GR/GG outer hashes differ")
                if shared_rows and len({row["train_seed"] for row in shared_rows}) != 1:
                    failures.append(f"draw {draw} fold {fold} rep {rep}: GR/GG/config train seeds differ")
        candidate_groups: dict[tuple[int, int, str], list[Candidate]] = {}
        for key in sorted(episodes):
            episode_rows = episodes[key]
            indices: list[int] = []
            for row in episode_rows:
                try:
                    indices.append(_int(row, "config_index"))
                except ScoreError as exc:
                    failures.append(f"draw {draw}: {exc}")
            if len(episode_rows) != 8 or sorted(indices) != list(GRID_INDICES):
                failures.append(f"draw {draw} episode {key}: requires exactly config indices 0..7")
                continue
            candidates: list[Candidate] = []
            for row in episode_rows:
                try:
                    if analysis_lock is None:
                        raise ScoreError("complete scoring requires a validated analysis lock")
                    unit_dir, _ = _safe_locked_unit_dir(
                        run_dir, row, analysis_lock["unit_artifact_hashes"][row["unit_id"]],
                    )
                    candidates.append(_load_candidate(row, unit_dir, splits, manifest, configs))
                except (ScoreError, OSError, json.JSONDecodeError, csv.Error) as exc:
                    failures.append(f"draw {draw}: {exc}")
            if len(candidates) == 8:
                candidate_groups[key] = candidates
        failures = list(dict.fromkeys(failures))
        if failures:
            draw_results[str(draw)] = {
                "complete": False,
                "draw_role": draw_role.get(draw),
                "analysis_order": draw_order.get(draw),
                "void_reasons": failures,
                "z": None,
                "d09r": None,
                "folds": {},
            }
            continue
        folds_out: dict[str, Any] = {}
        fold_zs: list[float] = []
        fold_d09s: list[float] = []
        for fold in range(N_FOLDS):
            rep_zs: list[float] = []
            rep_d09s: list[float] = []
            rep_out: dict[str, Any] = {}
            for rep in sorted(reps_by_fold[fold]):
                selected: dict[str, Candidate] = {}
                fixed: dict[str, Candidate] = {}
                for cell in CELLS:
                    candidates = candidate_groups[(fold, rep, cell)]
                    selected[cell] = choose_config(candidates)
                    fixed[cell] = next(c for c in candidates if c.config_index == FIXED_CONFIG_INDEX)
                gr_gap = selected["GR_hpo"].val_uar - selected["GR_hpo"].test_uar
                gg_gap = selected["GG_hpo"].val_uar - selected["GG_hpo"].test_uar
                z = gr_gap - gg_gap
                d09r = (
                    selected["GR_hpo"].test_uar - selected["GG_hpo"].test_uar
                    - fixed["GR_hpo"].test_uar + fixed["GG_hpo"].test_uar
                )
                rep_zs.append(z)
                rep_d09s.append(d09r)
                rep_out[str(rep)] = {
                    "z": z,
                    "d09r": d09r,
                    "GR_hpo": {
                        "selected_config_index": selected["GR_hpo"].config_index,
                        "selected_unit_id": selected["GR_hpo"].unit_id,
                        "val_uar": selected["GR_hpo"].val_uar,
                        "test_uar": selected["GR_hpo"].test_uar,
                        "fixed_index4_test_uar": fixed["GR_hpo"].test_uar,
                    },
                    "GG_hpo": {
                        "selected_config_index": selected["GG_hpo"].config_index,
                        "selected_unit_id": selected["GG_hpo"].unit_id,
                        "val_uar": selected["GG_hpo"].val_uar,
                        "test_uar": selected["GG_hpo"].test_uar,
                        "fixed_index4_test_uar": fixed["GG_hpo"].test_uar,
                    },
                }
            fold_z = float(np.mean(rep_zs))
            fold_d09 = float(np.mean(rep_d09s))
            fold_zs.append(fold_z)
            fold_d09s.append(fold_d09)
            folds_out[str(fold)] = {"z": fold_z, "d09r": fold_d09, "train_reps": rep_out}
        draw_z = float(np.mean(fold_zs))
        draw_d09 = float(np.mean(fold_d09s))
        draw_results[str(draw)] = {
            "complete": True,
            "draw_role": draw_role[draw],
            "analysis_order": draw_order[draw],
            "void_reasons": [],
            "z": draw_z,
            "d09r": draw_d09,
            "folds": folds_out,
        }
        complete.append((draw_order[draw], draw, draw_z, draw_d09))

    complete.sort()
    selected_complete = complete[:EXPECTED_DRAWS]
    selected_draw_ids = [draw for _, draw, _, _ in selected_complete]
    z_values = [z for _, _, z, _ in selected_complete]
    d09_values = [value for _, _, _, value in selected_complete]
    enough = len(selected_complete) == EXPECTED_DRAWS
    if enough:
        primary = {"tested": True, "n_draws": EXPECTED_DRAWS, "draw_ids": selected_draw_ids,
                   "values": z_values, **one_sample_t_summary(z_values)}
        primary["p_holm"] = primary["p_two_sided"]
        bootstrap: dict[str, Any] | None = bootstrap_mean_summary(z_values, bootstrap_reps, bootstrap_seed)
        sign_flip: dict[str, Any] | None = exact_sign_flip_summary(z_values) if compute_exact_sign_flip else None
        d09_t = one_sample_t_summary(d09_values)
        d09r = {
            "descriptive_only": True,
            "n": EXPECTED_DRAWS,
            "values": d09_values,
            "mean": d09_t["mean"],
            "sd": d09_t["sd"],
            "ci_level": d09_t["ci_level"],
            "ci": d09_t["ci"],
            "formula": "[T_GR(selected)-T_GG(selected)]-[T_GR(config4)-T_GG(config4)]",
        }
    else:
        reason = f"only {len(selected_complete)}/{EXPECTED_DRAWS} complete draws; N14R2 is not tested"
        primary = _not_tested(reason, len(selected_complete), z_values)
        bootstrap = None
        sign_flip = None
        d09r = {
            "descriptive_only": True,
            "n": len(selected_complete),
            "values": d09_values,
            "mean": None,
            "sd": None,
            "ci_level": 0.95,
            "ci": [None, None],
            "reason": reason,
            "formula": "[T_GR(selected)-T_GG(selected)]-[T_GR(config4)-T_GG(config4)]",
        }
    result = {
        "schema": SCHEMA,
        "study_id": rows[0]["study_id"],
        "spec_version": rows[0]["spec_version"],
        "expected_complete_draws": EXPECTED_DRAWS,
        "n_planned_draws": len(by_draw),
        "n_complete_draws": len(complete),
        "analysis_draw_ids": selected_draw_ids,
        "tested": enough,
        "status": "tested" if enough else "not tested",
        "estimand": "mean_draw((V-T)_GR-(V-T)_GG), train reps first then five folds equally weighted",
        "primary": primary,
        "bootstrap_sensitivity": bootstrap,
        "exact_sign_flip_sensitivity": sign_flip,
        "D09R2": d09r,
        "draws": draw_results,
        "integrity": {
            "pass": enough and all(draw_results[str(draw)]["complete"] for draw in selected_draw_ids),
            "global_failures": global_failures,
            "void_draw_ids": [int(draw) for draw, item in draw_results.items() if not item["complete"]],
        },
        "source_hashes": source_hashes,
    }
    if output_path is not None:
        _atomic_result(Path(output_path), result)
    return result


def _atomic_result(destination: Path, result: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _default_output(run_dir: Path) -> Path:
    if run_dir.name == "n14r2" and run_dir.parent.name == "runs":
        return run_dir.parent.parent / "results" / "n14r2_results.json"
    return run_dir / "n14r2_results.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Pinned CREMA-D manifest CSV or its directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or _default_output(args.run)
    result = score_run(args.plan, args.run, manifest_path=args.manifest, output_path=output)
    print(json.dumps({"output": str(output), "status": result["status"], "n_complete_draws": result["n_complete_draws"]}, indent=1))


if __name__ == "__main__":
    main()

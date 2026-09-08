#!/usr/bin/env python3
"""Close the P0.5 evidence-tier timing gap under AGENTS.md section 12.

This script never imports a candidate repository as a Python module and never
executes a whole repository.  For repositories that remain in tier 1, it
compiles and executes only the frozen, decisive AST statement(s) that construct
the train/test split.  Candidate inputs are replaced by a synthetic manifest
with explicit speaker IDs.  A separately written adapter is compared on a
finite, predeclared set of row-order cases.  Such agreement is empirical
equivalence evidence only for the stated input domain, seeds and cases; it is
not a general mathematical proof.

No network, GPU, model fit, installer, downloader or candidate binary is used.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle as sklearn_shuffle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "sources_p0_5"
RESULT_ROOT = ROOT / "results" / "execution_validation"
EQUIV_ROOT = RESULT_ROOT / "equivalence"
BASELINE_SCRIPT = ROOT / "tools" / "run_p0_5_execution_validation.py"
CASE_SEEDS = [0, 1, 2, 3, 7, 11, 17, 23, 42, 101, 313, 997, 20260810, 20260811, 20260812, 20260813]

EXPECTED_BASELINE = {
    "repository_evidence.csv": "c55360b0f4141d8f74af7543d25afc6d4b478cb1d90c9cf4a7bce35f2389ec1c",
    "tier_summary.csv": "367ac9acd4a40a5af3b251a68ad1ad5cd7baf53efbaa55a12adedb81851ba438",
    "change_registry.csv": "44b71a2da513549ceae7ce80864670ddbe2a7d441294db9e418dd52d53c8e298",
    "synthetic_manifest.csv": "4a22b612a2a964002cc71246b082acf5d3a10ae5c5710d79c783418638c7debb",
    "source_inventory.csv": "abe7225e4809507a6baf2cc9a2b2f333b1728cacc5c7321a0507459072412426",
}
BASELINE_SNAPSHOT = RESULT_ROOT / "baseline_repository_evidence.csv"


@dataclass(frozen=True)
class SourceLocator:
    relative_path: str
    cells: tuple[int, ...] = ()


SIMPLE_LOCATORS = {
    "P05R001": SourceLocator("notebooks/emotion_classifier_final.ipynb", (15,)),
    "P05R003": SourceLocator("train.py"),
    "P05R005": SourceLocator("emotion speech recognition code (2).ipynb", (24,)),
    "P05R006": SourceLocator("Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb", (15,)),
    "P05R007": SourceLocator("hsl621_emotion_recognition.ipynb", (39,)),
    "P05R009": SourceLocator("emotionravdess2.ipynb", (12,)),
    "P05R011": SourceLocator("utils.py"),
    "P05R013": SourceLocator("train_model.py"),
    "P05R014": SourceLocator("scripts/train_emotion_classifier.py"),
    "P05R017": SourceLocator("Speech_Emotion_Recognition.ipynb", (51, 52)),
    "P05R019": SourceLocator("Model.py"),
    "P05R021": SourceLocator("SER_v2_kaggle.ipynb", (8,)),
    "P05R023": SourceLocator("Inception_framing.py"),
    "P05R024": SourceLocator("train.py"),
    "P05R026": SourceLocator("main.ipynb", (3,)),
    "P05R028": SourceLocator("*.ipynb", (48,)),
    "P05R029": SourceLocator("DCNN_model.ipynb", (9,)),
}

CUSTOM_IDS = {"P05R010", "P05R015", "P05R016", "P05R018", "P05R020", "P05R022", "P05R027"}
TIER1_IDS = set(SIMPLE_LOCATORS) | CUSTOM_IDS
if len(TIER1_IDS) != 24:
    raise RuntimeError(f"Tier-1 roster must contain 24 repositories, got {len(TIER1_IDS)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("p05_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen baseline analysis module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_source(repo: Any, locator: SourceLocator) -> Path:
    base = SOURCE_ROOT / repo.source_dir
    if "*" in locator.relative_path:
        matches = sorted(base.glob(locator.relative_path))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one source for {repo.repository_id}: {matches}")
        return matches[0]
    return base / locator.relative_path


def source_code(repo: Any, locator: SourceLocator) -> tuple[Path, str]:
    path = resolve_source(repo, locator)
    if locator.cells:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        code = "\n".join("".join(cells[index].get("source", [])) for index in locator.cells)
    else:
        code = path.read_text(encoding="utf-8", errors="strict")
    return path, code


def split_assignments(code: str) -> tuple[list[ast.Assign], list[ast.Return]]:
    tree = ast.parse(code)
    assignments: list[ast.Assign] = []
    returns: list[ast.Return] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "train_test_split":
                assignments.append(node)
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "train_test_split":
                returns.append(node)
    assignments.sort(key=lambda item: (item.lineno, item.col_offset))
    returns.sort(key=lambda item: (item.lineno, item.col_offset))
    return assignments, returns


def compile_nodes(nodes: list[ast.stmt], label: str) -> Any:
    module = ast.Module(body=[copy.deepcopy(node) for node in nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, f"<frozen-candidate-fragment:{label}>", "exec")


def row_subset(rows: list[dict[str, str]], corpora: Iterable[str]) -> list[dict[str, str]]:
    wanted = set(corpora)
    return [row for row in rows if row["corpus"] in wanted]


def rows_for_repo(repo: Any, manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    p = repo.params
    if repo.method == "random_rows":
        return row_subset(manifest, p["corpora"])
    if repo.method == "augmented_random_rows":
        rows: list[dict[str, str]] = []
        for row in row_subset(manifest, [p["corpus"]]):
            for variant in range(p["variants"]):
                clone = dict(row)
                clone["row_id"] = f"{row['row_id']}:aug{variant}"
                rows.append(clone)
        return rows
    if repo.method == "windowed_random_rows":
        rows = []
        for row in row_subset(manifest, [p["corpus"]]):
            for window in range(p["windows"]):
                clone = dict(row)
                clone["row_id"] = f"{row['row_id']}:window{window}"
                rows.append(clone)
        return rows
    corpus = p.get("corpus")
    corpora = p.get("corpora")
    return row_subset(manifest, [corpus] if corpus else corpora)


def permute_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    order = np.random.RandomState(seed).permutation(len(rows))
    return [rows[int(index)] for index in order]


def index_namespace(rows: list[dict[str, str]]) -> dict[str, Any]:
    indices = np.arange(len(rows), dtype=int)
    emotions = np.array([row["emotion"] for row in rows], dtype=object)
    namespace: dict[str, Any] = {
        "np": np,
        "train_test_split": train_test_split,
        "X_base": indices,
        "Y_base": emotions,
        "X": indices,
        "Y": emotions,
        "train_scaled": indices,
        "train": indices,
        "target": emotions,
        "y": emotions,
        "y_encoded": emotions,
        "y_cat": emotions,
        "labels": emotions,
        "all_labels": emotions,
        "all_indices": indices,
        "file_path_list": indices,
        "dataset": indices.tolist(),
        "idx": indices,
        "y_int": emotions,
        "y_base": emotions,
        "image_paths": indices,
        "filtered": [None] * len(rows),
        "config": SimpleNamespace(TEST_SIZE=0.2, RANDOM_STATE=42),
        "TEST_SIZE": 0.1,
        "RANDOM_SEED": 42,
        "x": indices.tolist(),
        "test_size": 0.25,
    }
    return namespace


SIMPLE_OUTPUTS = {
    "P05R001": ("x_train", "x_test"),
    "P05R003": ("X_train", "X_test"),
    "P05R005": ("x_train", "x_test"),
    "P05R006": ("X_train", "X_test"),
    "P05R007": ("x_train", "x_test"),
    "P05R009": ("X_train", "X_val"),
    "P05R011": ("train_fps", "val_fps"),
    "P05R013": ("X_train", "X_test"),
    "P05R014": ("train_data", "test_data"),
    "P05R017": ("X_train", "X_test"),
    "P05R019": ("train_idx", "val_idx"),
    "P05R021": ("idx_tr", "idx_te"),
    "P05R023": ("train_paths", "test_paths"),
    "P05R024": ("X_train", "X_test"),
    "P05R028": ("X_train", "X_test"),
    "P05R029": ("X_train", "X_test"),
}


def as_indices(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim > 1:
        array = array[:, 0]
    return array.astype(int)


def run_simple_candidate(repo: Any, rows: list[dict[str, str]], case_seed: int) -> tuple[np.ndarray, np.ndarray, str, str]:
    path, code = source_code(repo, SIMPLE_LOCATORS[repo.repository_id])
    assignments, returns = split_assignments(code)
    expected_calls = 2 if repo.repository_id in {"P05R014", "P05R017", "P05R021", "P05R029"} else 1
    if len(assignments) + len(returns) != expected_calls:
        raise RuntimeError(f"Unexpected split-call count for {repo.repository_id}: {len(assignments)} assignments, {len(returns)} returns")
    namespace = index_namespace(rows)
    np.random.seed(case_seed)
    if assignments:
        exec(compile_nodes(assignments, repo.repository_id), {"__builtins__": {}}, namespace)
    if returns:
        result_assign = ast.Assign(targets=[ast.Name(id="__result__", ctx=ast.Store())], value=copy.deepcopy(returns[0].value))
        exec(compile_nodes([result_assign], repo.repository_id), {"__builtins__": {}}, namespace)
        train_values, test_values = namespace["__result__"][0], namespace["__result__"][1]
    else:
        train_name, test_name = SIMPLE_OUTPUTS[repo.repository_id]
        train_values, test_values = namespace[train_name], namespace[test_name]
    fragment_dump = "\n".join(ast.dump(node, include_attributes=False) for node in assignments + returns)
    return as_indices(train_values), as_indices(test_values), path.relative_to(SOURCE_ROOT / repo.source_dir).as_posix(), text_sha256(fragment_dump)


def run_simple_adapter(repo_id: str, rows: list[dict[str, str]], case_seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(rows), dtype=int)
    labels = np.array([row["emotion"] for row in rows], dtype=object)
    np.random.seed(case_seed)
    if repo_id == "P05R001":
        return train_test_split(idx, test_size=0.2, random_state=0, shuffle=True)
    if repo_id == "P05R003":
        return train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    if repo_id == "P05R005":
        return train_test_split(idx, test_size=0.2)
    if repo_id == "P05R006":
        return train_test_split(idx, test_size=0.2, stratify=labels)
    if repo_id == "P05R007":
        return train_test_split(idx, random_state=0, shuffle=True)
    if repo_id == "P05R009":
        return train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    if repo_id == "P05R011":
        return train_test_split(idx, labels, test_size=0.1)[0:2]
    if repo_id == "P05R013":
        return train_test_split(idx, test_size=0.2, random_state=42)
    if repo_id == "P05R014":
        train, test = train_test_split(idx, test_size=0.2, random_state=42)
        train, _val = train_test_split(train, test_size=0.1, random_state=42)
        return np.asarray(train), np.asarray(test)
    if repo_id == "P05R017":
        train, test = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
        train, _val = train_test_split(train, test_size=0.1, random_state=42, shuffle=True)
        return np.asarray(train), np.asarray(test)
    if repo_id == "P05R019":
        return train_test_split(idx, test_size=0.1, random_state=42, stratify=labels)
    if repo_id == "P05R021":
        train, temp = train_test_split(idx, test_size=0.15, stratify=labels, random_state=42)
        _val, test = train_test_split(temp, test_size=0.5, stratify=labels[temp], random_state=42)
        return np.asarray(train), np.asarray(test)
    if repo_id == "P05R023":
        return train_test_split(idx, test_size=0.3, random_state=42, stratify=labels)
    if repo_id == "P05R024":
        return train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    if repo_id == "P05R026":
        return train_test_split(idx, test_size=0.25, random_state=9)
    if repo_id == "P05R028":
        return train_test_split(idx, test_size=0.3, random_state=1)
    if repo_id == "P05R029":
        train, test = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
        train, _val = train_test_split(train, test_size=0.1, stratify=labels[train], random_state=42)
        return np.asarray(train), np.asarray(test)
    raise KeyError(repo_id)


class MiniFrame:
    """Minimal in-memory frame for the exact P05R015 boolean-index fragment."""

    def __init__(self, row_ids: Iterable[int], speakers: Iterable[int]):
        self.row_ids = np.asarray(list(row_ids), dtype=int)
        self.speakers = np.asarray(list(speakers), dtype=int)

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            if key != "speaker":
                raise KeyError(key)
            return self.speakers
        mask = np.asarray(key, dtype=bool)
        return MiniFrame(self.row_ids[mask], self.speakers[mask])


def pair_from_indices(rows: list[dict[str, str]], train_indices: Iterable[int], test_indices: Iterable[int], label: str = "final_evaluation") -> dict[str, Any]:
    train_i = [int(value) for value in np.asarray(list(train_indices)).reshape(-1)]
    test_i = [int(value) for value in np.asarray(list(test_indices)).reshape(-1)]
    train_speakers = sorted({rows[index]["speaker_id"] for index in train_i})
    test_speakers = sorted({rows[index]["speaker_id"] for index in test_i})
    intersection = sorted(set(train_speakers) & set(test_speakers))
    return {
        "pair": label,
        "train_rows": len(train_i),
        "test_rows": len(test_i),
        "train_speakers": train_speakers,
        "test_speakers": test_speakers,
        "speaker_intersection": intersection,
        "speaker_intersection_count": len(intersection),
    }


def custom_candidate_and_adapter(repo: Any, rows: list[dict[str, str]], case_seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]], str, str, str]:
    rid = repo.repository_id
    base = SOURCE_ROOT / repo.source_dir
    if rid == "P05R010":
        path = base / "train.py"
        code = path.read_text(encoding="utf-8")
        tree = ast.parse(code)
        selected = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and (31 <= node.lineno <= 39 or 130 <= node.lineno <= 139)]
        selected.sort(key=lambda node: (node.lineno, node.col_offset))
        data0 = [[index, index % 7] for index in range(len(rows))]
        labels0 = np.array([index % 8 for index in range(len(rows))], dtype=int)
        namespace = {"np": np, "shuffle": sklearn_shuffle, "data": copy.deepcopy(data0), "labels": labels0.copy()}
        np.random.seed(case_seed)
        exec(compile_nodes(selected, rid), {"__builtins__": {"len": len, "int": int}}, namespace)
        cand = [(as_indices(namespace["train_data"]), as_indices(namespace["test_data"]))]
        np.random.seed(case_seed)
        matrix = np.hstack((np.asarray(data0), labels0.reshape(-1, 1)))
        shuffled = sklearn_shuffle(matrix)
        adapter = [(as_indices(shuffled[:-180][:1020, :-1]), as_indices(shuffled[-181:-1, :-1]))]
        fragment = "\n".join(ast.dump(node, include_attributes=False) for node in selected)
        return cand, adapter, "train.py:L31-L39|L130-L139", text_sha256(fragment), "exact shuffle-plus-fixed-slice statements"

    if rid == "P05R015":
        path = base / "modules" / "formatter.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = [node for node in ast.walk(tree) if isinstance(node, ast.stmt) and 57 <= getattr(node, "lineno", -1) <= 61 and isinstance(node, (ast.Assign, ast.Expr))]
        selected = [node for node in selected if not any(isinstance(parent, ast.Assign) and parent is not node for parent in [])]
        selected.sort(key=lambda node: (node.lineno, node.col_offset))
        # Keep only the five top-level statements from format_dataset.
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "format_dataset")
        selected = [node for node in function.body if 57 <= getattr(node, "lineno", -1) <= 61]
        speaker_values = [int(row["speaker_id"].split(":")[-1]) for row in rows]
        frame = MiniFrame(range(len(rows)), speaker_values)
        namespace = {"np": np, "df": frame, "seed": 42}
        exec(compile_nodes(selected, rid), {"__builtins__": {}}, namespace)
        cand = [(namespace["df_train"].row_ids, namespace["df_test"].row_ids)]
        np.random.seed(42)
        test_speakers = np.random.choice(np.unique(speaker_values), size=2, replace=False)
        mask = np.isin(speaker_values, test_speakers)
        adapter = [(np.arange(len(rows))[~mask], np.arange(len(rows))[mask])]
        fragment = "\n".join(ast.dump(node, include_attributes=False) for node in selected)
        return cand, adapter, "modules/formatter.py:L57-L61", text_sha256(fragment), "exact NumPy speaker-choice and dataframe-mask statements on MiniFrame"

    if rid == "P05R016":
        path = base / "train_val_set.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and node.lineno == 51)
        actor_codes = [row["speaker_id"].split(":")[-1] for row in rows]
        files = ["x" * 18 + actor + f"_{index:05d}" for index, actor in enumerate(actor_codes)]
        namespace = {"range": range, "len": len, "aud_files": files, "aud_data": list(range(len(rows))), "aud_train": [], "aud_test": [], "aud_val": [], "aud_train_files": [], "aud_test_files": [], "aud_val_files": []}
        exec(compile_nodes([loop], rid), {"__builtins__": {}}, namespace)
        cand = [(np.asarray(namespace["aud_train"], dtype=int), np.asarray(namespace["aud_test"], dtype=int))]
        train = [index for index, actor in enumerate(actor_codes) if actor not in {"21", "22", "23", "24"}]
        test = [index for index, actor in enumerate(actor_codes) if actor in {"23", "24"}]
        adapter = [(np.asarray(train), np.asarray(test))]
        return cand, adapter, "train_val_set.py:L51-L60", text_sha256(ast.dump(loop, include_attributes=False)), "exact filename-slice branch"

    if rid == "P05R018":
        path = base / "iemocap_emotion_recognition_csv.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "".join(notebook["cells"][9].get("source", []))
        tree = ast.parse(code)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_speaker_independent_cv")
        unique_assign = next(node for node in function.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "unique_speakers" for target in node.targets))
        loop = next(node for node in function.body if isinstance(node, ast.For))
        decisive_lines = {43, 47, 48, 49, 51, 52, 53}
        decisive = [node for node in loop.body if getattr(node, "lineno", -1) in decisive_lines]
        slim_loop = ast.For(target=copy.deepcopy(loop.target), iter=copy.deepcopy(loop.iter), body=[copy.deepcopy(node) for node in decisive], orelse=[])
        slim_loop.body.append(ast.parse("observed.append((X_train.copy(), X_test.copy()))").body[0])
        namespace = {
            "np": np,
            "sorted": sorted,
            "set": set,
            "enumerate": enumerate,
            "len": len,
            "speakers_array": np.array([row["speaker_id"].split(":")[-1] for row in rows], dtype=object),
            "file_paths": np.arange(len(rows), dtype=int),
            "y": np.array([row["emotion"] for row in rows], dtype=object),
            "observed": [],
        }
        exec(compile_nodes([unique_assign, slim_loop], rid), {"__builtins__": {}}, namespace)
        cand = [(as_indices(train), as_indices(test)) for train, test in namespace["observed"]]
        speakers = sorted(set(namespace["speakers_array"]))
        adapter = []
        for fold, test_speaker in enumerate(speakers, 1):
            val_speaker = speakers[fold % len(speakers)]
            test_mask = namespace["speakers_array"] == test_speaker
            val_mask = namespace["speakers_array"] == val_speaker
            adapter.append((np.arange(len(rows))[~(test_mask | val_mask)], np.arange(len(rows))[test_mask]))
        fragment = ast.dump(unique_assign, include_attributes=False) + ast.dump(slim_loop, include_attributes=False)
        return cand, adapter, "iemocap_emotion_recognition_csv.ipynb:cell9 split-mask fragment", text_sha256(fragment), "exact LOSO loop header and mask/index assignments"

    if rid == "P05R020":
        path = base / "notebooks" / "01_Data_Preparation" / "04_dataset_split.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "".join(notebook["cells"][6].get("source", []))
        speakers = np.array(sorted({row["speaker_id"] for row in rows}), dtype=object)
        namespace = {"speakers": speakers}
        exec(compile(code, "<frozen-candidate-fragment:P05R020>", "exec"), {"__builtins__": {"__import__": __import__}}, namespace)
        train_s, test_s = set(namespace["train_speakers"]), set(namespace["test_speakers"])
        cand = [(np.array([i for i, row in enumerate(rows) if row["speaker_id"] in train_s]), np.array([i for i, row in enumerate(rows) if row["speaker_id"] in test_s]))]
        train_s2, temp_s2 = train_test_split(speakers, test_size=0.30, random_state=42)
        _val_s2, test_s2 = train_test_split(temp_s2, test_size=0.50, random_state=42)
        train_set2, test_set2 = set(train_s2), set(test_s2)
        adapter = [(np.array([i for i, row in enumerate(rows) if row["speaker_id"] in train_set2]), np.array([i for i, row in enumerate(rows) if row["speaker_id"] in test_set2]))]
        return cand, adapter, "04_dataset_split.ipynb:cell6", text_sha256(code), "entire safe speaker-split cell"

    if rid == "P05R022":
        path = base / "crnn" / "train.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "".join(notebook["cells"][15].get("source", []))
        tree = ast.parse(code)
        skf_assign = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "skf" for target in node.targets))
        loop = next(node for node in tree.body if isinstance(node, ast.For))
        labels = np.array([row["emotion"] for row in rows], dtype=object)
        namespace = {"StratifiedKFold": StratifiedKFold, "X_raw": np.arange(len(rows)), "y_raw": labels}
        np.random.seed(case_seed)
        exec(compile_nodes([skf_assign], rid), {"__builtins__": {}}, namespace)
        iterator = eval(compile(ast.Expression(copy.deepcopy(loop.iter)), "<frozen-candidate-fragment:P05R022:split>", "eval"), {"__builtins__": {}}, namespace)
        cand = [(np.asarray(train), np.asarray(test)) for train, test in iterator]
        np.random.seed(case_seed)
        adapter = [(np.asarray(train), np.asarray(test)) for train, test in StratifiedKFold(n_splits=4, shuffle=True).split(np.arange(len(rows)), labels, labels)]
        fragment = ast.dump(skf_assign, include_attributes=False) + ast.dump(loop.iter, include_attributes=False)
        return cand, adapter, "crnn/train.ipynb:cell15 constructor and split iterator", text_sha256(fragment), "exact StratifiedKFold constructor and split call"

    if rid == "P05R027":
        path = base / "shared-dataset-loader" / "nbdev-upc-aidl-iemocap-datasets" / "src" / "nbdev_upc_aidl_iemocap_datasets" / "core.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DatasetsFactory")
        partition_assign = next(node for node in class_node.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "IEMOCAP_SPEAKERS_PARTITIONS" for target in node.targets))
        method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "build_dataset")
        branch = next(node for node in method.body if isinstance(node, ast.If) and getattr(node, "lineno", 0) == 776)
        namespace: dict[str, Any] = {}
        exec(compile_nodes([partition_assign], rid), {"__builtins__": {}}, namespace)
        partitions = namespace["IEMOCAP_SPEAKERS_PARTITIONS"]
        dummy = SimpleNamespace(IEMOCAP_SPEAKERS_PARTITIONS=partitions, EXTERNAL_AUDIO_SPEAKERS_PARTITIONS={"train": [], "validation": [], "test": []}, DATASET_AUDIO_CHUNK_GROUP_ID_INWORLD_CUTOFF=8)
        selected: dict[str, list[str]] = {}
        for partition in ("train", "test"):
            local = {"partition_type": partition, "speaker_ids_filter": None, "id": 0, "self": dummy, "ValueError": ValueError}
            exec(compile_nodes([branch], rid), {"__builtins__": {}}, local)
            selected[partition] = list(local["speaker_ids_filter"])
        normalized = [row["speaker_id"].split(":")[-1].replace("Ses01F", "Ses01__F").replace("Ses01M", "Ses01__M").replace("Ses02F", "Ses02__F").replace("Ses02M", "Ses02__M").replace("Ses03F", "Ses03__F").replace("Ses03M", "Ses03__M").replace("Ses04F", "Ses04__F").replace("Ses04M", "Ses04__M").replace("Ses05F", "Ses05__F").replace("Ses05M", "Ses05__M") for row in rows]
        cand = [(np.array([i for i, speaker in enumerate(normalized) if speaker in selected["train"]]), np.array([i for i, speaker in enumerate(normalized) if speaker in selected["test"]]))]
        manual_train = {"Ses03__F", "Ses03__M", "Ses04__F", "Ses04__M", "Ses05__F", "Ses05__M"}
        manual_test = {"Ses01__F", "Ses01__M"}
        adapter = [(np.array([i for i, speaker in enumerate(normalized) if speaker in manual_train]), np.array([i for i, speaker in enumerate(normalized) if speaker in manual_test]))]
        fragment = ast.dump(partition_assign, include_attributes=False) + ast.dump(branch, include_attributes=False)
        return cand, adapter, "core.py:L531-L548|L775-L785", text_sha256(fragment), "exact partition literal and partition_type branch"

    raise KeyError(rid)


def compare_pairs(candidate: list[tuple[np.ndarray, np.ndarray]], adapter: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    if len(candidate) != len(adapter):
        return False
    for (candidate_train, candidate_test), (adapter_train, adapter_test) in zip(candidate, adapter):
        if not np.array_equal(np.asarray(candidate_train), np.asarray(adapter_train)):
            return False
        if not np.array_equal(np.asarray(candidate_test), np.asarray(adapter_test)):
            return False
    return True


def first_tier_evidence(repo: Any, manifest: list[dict[str, str]]) -> dict[str, Any]:
    base_rows = rows_for_repo(repo, manifest)
    case_records: list[dict[str, Any]] = []
    primary_pairs: list[dict[str, Any]] = []
    source_ref = ""
    fragment_hash = ""
    execution_scope = ""
    for case_index, seed in enumerate(CASE_SEEDS):
        rows = base_rows if case_index == 0 else permute_rows(base_rows, seed)
        if repo.repository_id in SIMPLE_LOCATORS:
            cand_train, cand_test, source_ref, fragment_hash = run_simple_candidate(repo, rows, seed)
            adapter_train, adapter_test = run_simple_adapter(repo.repository_id, rows, seed)
            candidate_pairs = [(cand_train, cand_test)]
            adapter_pairs = [(adapter_train, adapter_test)]
            execution_scope = "exact train_test_split assignment/return AST statement(s)"
        else:
            candidate_pairs, adapter_pairs, source_ref, fragment_hash, execution_scope = custom_candidate_and_adapter(repo, rows, seed)
        passed = compare_pairs(candidate_pairs, adapter_pairs)
        case_records.append({
            "case_index": case_index,
            "seed": seed,
            "input_rows": len(rows),
            "input_speakers": len({row["speaker_id"] for row in rows}),
            "pair_count": len(candidate_pairs),
            "exact_index_agreement": passed,
        })
        if case_index == 0:
            primary_pairs = [pair_from_indices(rows, train, test, f"pair_{pair_index:02d}") for pair_index, (train, test) in enumerate(candidate_pairs)]
    passed_count = sum(bool(row["exact_index_agreement"]) for row in case_records)
    if passed_count != len(CASE_SEEDS):
        raise RuntimeError(f"Property comparison failed for {repo.repository_id}: {passed_count}/{len(CASE_SEEDS)}")
    max_intersection = max(pair["speaker_intersection_count"] for pair in primary_pairs)
    observed = "yes" if max_intersection > 0 else "no"
    return {
        "repository_id": repo.repository_id,
        "evidence_tier_before": "tier1_execution_confirmed",
        "evidence_tier_after": "tier1_execution_confirmed",
        "candidate_module_imported": False,
        "whole_repository_executed": False,
        "original_source_fragment_executed": True,
        "execution_scope": execution_scope,
        "source_reference": source_ref,
        "source_file_sha256": sha256((SOURCE_ROOT / repo.source_dir / source_ref.split(":", 1)[0].split("|", 1)[0])) if (SOURCE_ROOT / repo.source_dir / source_ref.split(":", 1)[0].split("|", 1)[0]).is_file() else "see source_inventory.csv",
        "source_fragment_ast_sha256": fragment_hash,
        "adapter_equivalence": "pass_finite_cases",
        "property_case_count": len(CASE_SEEDS),
        "property_cases_passed": passed_count,
        "property_case_seeds": CASE_SEEDS,
        "input_domain": "case 0 is the complete frozen synthetic target-corpus manifest for this repository; cases 1-15 are deterministic row-order permutations of the same manifest, including any frozen augmentation/window expansion",
        "output_contract": "ordered train and final-evaluation row-index arrays; multi-stage paths include their exact inner validation removal before the final train array",
        "coverage_boundary": "Only the decisive frozen split fragment and its declared arrays/lists are exercised. Upstream audio/feature extraction, external manifests, model code and performance are outside scope.",
        "general_mathematical_proof": False,
        "primary_pairs": primary_pairs,
        "max_speaker_intersection": max_intersection,
        "split_risk_observed": observed,
        "cases": case_records,
        "safety": {"network": False, "gpu": False, "candidate_module_import": False, "whole_repository": False, "training": False},
    }


def boundary_evidence(row: dict[str, str], before: str, after: str, decision: str, reason: str) -> dict[str, Any]:
    return {
        "repository_id": row["repository_id"],
        "evidence_tier_before": before,
        "evidence_tier_after": after,
        "candidate_module_imported": False,
        "whole_repository_executed": False,
        "original_source_fragment_executed": False,
        "execution_scope": "none",
        "adapter_equivalence": decision,
        "property_case_count": 0,
        "property_cases_passed": 0,
        "property_case_seeds": [],
        "input_domain": "not applicable",
        "output_contract": "not claimed",
        "coverage_boundary": reason,
        "general_mathematical_proof": False,
        "primary_pairs": [],
        "max_speaker_intersection": None,
        "split_risk_observed": "unknown" if after == "tier3_unknown" else row["split_risk_final"],
        "cases": [],
        "safety": {"network": False, "gpu": False, "candidate_module_import": False, "whole_repository": False, "training": False},
    }


def main() -> int:
    for relative in ("synthetic_manifest.csv", "source_inventory.csv"):
        expected = EXPECTED_BASELINE[relative]
        actual = sha256(RESULT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Baseline artifact drift before section-12 closure: {relative}: {actual} != {expected}")

    if BASELINE_SNAPSHOT.exists():
        actual = sha256(BASELINE_SNAPSHOT)
        expected = EXPECTED_BASELINE["repository_evidence.csv"]
        if actual != expected:
            raise RuntimeError(f"Frozen baseline snapshot drift: {actual} != {expected}")
        old_rows = read_csv(BASELINE_SNAPSHOT)
    else:
        for relative in ("repository_evidence.csv", "tier_summary.csv", "change_registry.csv"):
            expected = EXPECTED_BASELINE[relative]
            actual = sha256(RESULT_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"Baseline artifact drift before first section-12 closure: {relative}: {actual} != {expected}")
        old_rows = read_csv(RESULT_ROOT / "repository_evidence.csv")
        write_csv(BASELINE_SNAPSHOT, list(old_rows[0].keys()), old_rows)
        if sha256(BASELINE_SNAPSHOT) != EXPECTED_BASELINE["repository_evidence.csv"]:
            raise RuntimeError("Could not freeze the exact baseline repository evidence snapshot")

    baseline = load_baseline_module()
    specs = {repo.repository_id: repo for repo in baseline.SPECS}
    old_by_id = {row["repository_id"]: row for row in old_rows}
    manifest = read_csv(RESULT_ROOT / "synthetic_manifest.csv")
    if len(old_rows) != 30 or len(specs) != 30 or set(old_by_id) != set(specs):
        raise RuntimeError("Frozen 30-repository roster mismatch")

    survey = {row["repository_id"]: row for row in read_csv(ROOT / "p0_5_random_survey.csv")}
    new_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    new_outputs: dict[str, dict[str, Any]] = {}

    for old in old_rows:
        rid = old["repository_id"]
        repo = specs[rid]
        before_tier = old["evidence_tier"]
        if rid in TIER1_IDS:
            evidence = first_tier_evidence(repo, manifest)
            after_tier = "tier1_execution_confirmed"
            final_risk = evidence["split_risk_observed"]
            if final_risk != old["split_risk_final"]:
                raise RuntimeError(f"Unexpected exact-fragment outcome change for {rid}: {old['split_risk_final']} -> {final_risk}")
            validation_method = "direct_frozen_ast_fragment_plus_finite_property_test"
            candidate_executed = "true"
            synthetic_rows = max(pair["train_rows"] + pair["test_rows"] for pair in evidence["primary_pairs"])
            synthetic_speakers = len(set().union(*(set(pair["train_speakers"]) | set(pair["test_speakers"]) for pair in evidence["primary_pairs"])))
            evaluation_pairs = len(evidence["primary_pairs"])
            max_intersection: Any = evidence["max_speaker_intersection"]
            decision = "retained_tier1_exact_fragment_and_16_of_16_cases"
            inevitability = old["inevitability_or_limit"] + " Section-12 closure executed only the exact frozen split AST fragment; 16/16 finite manifest/order cases matched a separate adapter. This is bounded empirical equivalence evidence, not a general proof."
        elif rid == "P05R008":
            after_tier = "tier3_unknown"
            final_risk = "unknown"
            reason = "The frozen notebooks point training and evaluation to external Kaggle Train/Test directories. Comments and saved outputs describe five test actor folders (Actor 20-24), but the actual Train/Test manifests and construction rule are absent; source lines cannot prove that the two external directories are speaker-disjoint. Executing a locally invented folder tree would only test that invention."
            evidence = boundary_evidence(old, before_tier, after_tier, "downgraded_external_manifest_absent", reason)
            validation_method = "static_boundary_review_external_partition_manifest_absent"
            candidate_executed = "false"
            synthetic_rows = 0
            synthetic_speakers = 0
            evaluation_pairs = 0
            max_intersection = ""
            decision = "downgraded_tier1_to_unknown"
            inevitability = reason
        else:
            after_tier = before_tier
            final_risk = old["split_risk_final"]
            reason = old["inevitability_or_limit"]
            evidence = boundary_evidence(old, before_tier, after_tier, "not_applicable_no_safe_decisive_original_path", reason)
            validation_method = old["validation_method"]
            candidate_executed = "false"
            synthetic_rows = old["synthetic_rows_evaluated"]
            synthetic_speakers = old["synthetic_speakers_evaluated"]
            evaluation_pairs = old["evaluation_pairs"]
            max_intersection = old["max_speaker_intersection"]
            decision = "tier_unchanged"
            inevitability = reason

        evidence.update({
            "repository_url": old["repository_url"],
            "commit_sha": old["commit_sha"],
            "draw_rank": int(old["draw_rank"]),
            "decisive_file_line": old["decisive_file_line"],
            "split_risk_before_section12": old["split_risk_final"],
            "split_risk_after_section12": final_risk,
        })
        output_rel = f"equivalence/{rid}.json"
        write_json(RESULT_ROOT / output_rel, evidence)
        new_outputs[rid] = evidence

        new_row = dict(old)
        new_row.update({
            "evidence_tier": after_tier,
            "validation_method": validation_method,
            "candidate_code_executed": candidate_executed,
            "synthetic_rows_evaluated": synthetic_rows,
            "synthetic_speakers_evaluated": synthetic_speakers,
            "evaluation_pairs": evaluation_pairs,
            "max_speaker_intersection": max_intersection,
            "split_risk_final": final_risk,
            "changed_from_prior": str(old["prior_split_risk"] != final_risk).lower(),
            "inevitability_or_limit": inevitability,
            "reproduction_output": output_rel,
            "evidence_tier_before_section12": before_tier,
            "original_source_fragment_executed": str(bool(evidence["original_source_fragment_executed"])).lower(),
            "candidate_module_imported": "false",
            "property_cases": evidence["property_case_count"],
            "property_cases_passed": evidence["property_cases_passed"],
            "adapter_equivalence": evidence["adapter_equivalence"],
            "tier_gap_decision": decision,
            "baseline_reproduction_output": old["reproduction_output"],
        })
        new_rows.append(new_row)
        equivalence_rows.append({
            "repository_id": rid,
            "draw_rank": old["draw_rank"],
            "repository_url": old["repository_url"],
            "commit_sha": old["commit_sha"],
            "tier_before": before_tier,
            "tier_after": after_tier,
            "original_source_fragment_executed": str(bool(evidence["original_source_fragment_executed"])).lower(),
            "candidate_module_imported": "false",
            "property_cases": evidence["property_case_count"],
            "property_cases_passed": evidence["property_cases_passed"],
            "adapter_equivalence": evidence["adapter_equivalence"],
            "split_before": old["split_risk_final"],
            "split_after": final_risk,
            "general_mathematical_proof": "false",
            "input_domain": evidence["input_domain"],
            "output_contract": evidence["output_contract"],
            "coverage_boundary": evidence["coverage_boundary"],
            "decisive_file_line": old["decisive_file_line"],
            "evidence_output": output_rel,
        })

    evidence_fields = list(old_rows[0].keys()) + [
        "evidence_tier_before_section12", "original_source_fragment_executed",
        "candidate_module_imported", "property_cases", "property_cases_passed",
        "adapter_equivalence", "tier_gap_decision", "baseline_reproduction_output",
    ]
    write_csv(RESULT_ROOT / "repository_evidence.csv", evidence_fields, new_rows)
    write_csv(RESULT_ROOT / "equivalence_registry.csv", list(equivalence_rows[0].keys()), equivalence_rows)

    tier_counts = Counter(row["evidence_tier"] for row in new_rows)
    outcome_counts = Counter(row["split_risk_final"] for row in new_rows)
    summary_rows = [
        {"dimension": "evidence_tier", "category": key, "count": tier_counts[key], "denominator": 30, "note": "mutually exclusive repository tier after section-12 closure"}
        for key in ("tier1_execution_confirmed", "tier2_static_execution_inference", "tier3_unknown")
    ] + [
        {"dimension": "split_risk_final", "category": key, "count": outcome_counts[key], "denominator": 30, "note": "yes=exact-fragment overlap or decisive static non-enforcement; no=speaker-disjoint; unknown=insufficient frozen evidence"}
        for key in ("yes", "no", "unknown")
    ]
    write_csv(RESULT_ROOT / "tier_summary.csv", ["dimension", "category", "count", "denominator", "note"], summary_rows)

    changes = [
        {
            "repository_id": "P05R015", "change_type": "prior_stage_outcome_correction", "field": "split_risk_final",
            "old_value": "yes", "new_value": "no", "outcome_changed": "true",
            "reason": "Speaker is parsed from each filename; the exact frozen lines 57-61 select two whole speaker IDs and mask all rows by those IDs.",
            "decisive_file_line": "modules/formatter.py:L18-L33 | modules/formatter.py:L47-L64 | app.py:L12-L14",
            "evidence_output": "equivalence/P05R015.json",
        },
        {
            "repository_id": "P05R008", "change_type": "section12_evidence_and_outcome_downgrade", "field": "evidence_tier|split_risk_final",
            "old_value": "tier1_execution_confirmed|no", "new_value": "tier3_unknown|unknown", "outcome_changed": "true",
            "reason": "External Kaggle Train/Test manifests are absent. Comments and recorded counts do not make cross-directory speaker disjointness logically identifiable from the frozen repository.",
            "decisive_file_line": "cnn-final-ee708-project.ipynb:cell5 | Evaluation_Test_Data.ipynb:cells2,5,7,8",
            "evidence_output": "equivalence/P05R008.json",
        },
    ]
    for rid in ("P05R014", "P05R017", "P05R021", "P05R029"):
        changes.append({
            "repository_id": rid, "change_type": "section12_adapter_contract_clarification", "field": "adapter_output_contract",
            "old_value": "outer split only", "new_value": "exact outer split plus inner validation removal", "outcome_changed": "false",
            "reason": "The earlier replica stopped after the outer split. The section-12 exact fragment executes the subsequent validation split before defining the final training rows; the final test arm and speaker-overlap verdict are unchanged.",
            "decisive_file_line": old_by_id[rid]["decisive_file_line"],
            "evidence_output": f"equivalence/{rid}.json",
        })
    write_csv(RESULT_ROOT / "change_registry.csv", list(changes[0].keys()), changes)

    readme = """# P0.5 evidence-tier execution validation — section-12 closure

This directory is the self-contained machine-readable record for the frozen 30-repository probability sample drawn from the **public-code candidate frame generated by the preregistered search queries**. It is not a probability sample of all SER papers, all SER repositories, or deployed systems.

## Evidence meaning

- `tier1_execution_confirmed`: only the frozen decisive split AST statement(s) were executed on a synthetic manifest with explicit speaker IDs. No candidate module or whole repository was imported. Each retained tier-1 repository has 16/16 exact-index matches against a separately written adapter over the stated manifest/order cases.
- `tier2_static_execution_inference`: decisive source lines make non-enforcement inferable, but the original path could not safely be run in the frozen environment.
- `tier3_unknown`: frozen evidence does not identify the final speaker relationship.

The finite property tests are **empirical equivalence evidence under the explicitly recorded input domain, seeds and cases**. They are not a general mathematical proof and do not cover upstream audio/feature extraction, external manifests, model execution, or performance.

`P05R008` is downgraded to `unknown`: the frozen notebooks refer to external Kaggle `Train`/`Test` directories whose manifests and construction rule are absent. A locally invented folder tree would not establish what the authors' external directories contained.

## Files

- `repository_evidence.csv`: final one-row-per-repository decision after section-12 closure.
- `equivalence_registry.csv`: concise per-repository execution/equivalence/downgrade basis.
- `equivalence/*.json`: exact finite-case records and primary train/test speaker intersections, or the explicit downgrade boundary.
- `tier_summary.csv`: final tier and risk counts.
- `change_registry.csv`: outcome, tier and adapter-contract changes across the evidence upgrade.
- `source_inventory.csv`, `synthetic_manifest.csv`, `per_repository/*.json`: preserved baseline inputs/history from the earlier evidence pass.
- `analysis_manifest.json`, `verification.json`, `artifact_qa.json`: provenance and QA.

Safety boundary: no network, GPU, training, installation, candidate binary, whole-repository execution, P1 rerun or P2 rerun occurred.
"""
    (RESULT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    exclusions = {"analysis_manifest.json", "verification.json", "artifact_qa.json"}
    output_hashes = {}
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path.name not in exclusions:
            output_hashes[path.relative_to(RESULT_ROOT).as_posix()] = sha256(path)
    manifest_payload = {
        "stage": "P0.5 section-12 evidence-tier timing-gap closure",
        "authority_root": str(ROOT),
        "analysis_script": "tools/close_p0_5_evidence_tier_gap.py",
        "analysis_script_sha256": sha256(Path(__file__)),
        "baseline_analysis_script": "tools/run_p0_5_execution_validation.py",
        "baseline_analysis_script_sha256": sha256(BASELINE_SCRIPT),
        "verifier_script": "tools/verify_p0_5_execution_validation.py",
        "verifier_script_sha256": sha256(ROOT / "tools" / "verify_p0_5_execution_validation.py"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sys.modules["sklearn"].__version__,
        "sample_size": 30,
        "property_case_seeds": CASE_SEEDS,
        "tier_counts": dict(tier_counts),
        "outcome_counts": dict(outcome_counts),
        "tier1_exact_fragment_count": len(TIER1_IDS),
        "tier1_property_cases_total": len(TIER1_IDS) * len(CASE_SEEDS),
        "tier1_property_cases_passed": sum(int(row["property_cases_passed"]) for row in new_rows),
        "section12_changes": ["P05R008", "P05R014", "P05R017", "P05R021", "P05R029"],
        "safety": {
            "network_used": False, "gpu_used": False, "candidate_module_imported": False,
            "whole_repository_executed": False, "models_trained": False,
            "p1_rerun": False, "p2_rerun": False, "ser_modified": False,
        },
        "inference_scope": "public-code candidate frame generated by the preregistered search queries; no extrapolation to all SER literature",
        "finite_property_test_is_general_proof": False,
        "output_hash_exclusions": sorted(exclusions),
        "output_sha256": output_hashes,
    }
    write_json(RESULT_ROOT / "analysis_manifest.json", manifest_payload)
    print(json.dumps({"tier_counts": dict(tier_counts), "outcome_counts": dict(outcome_counts), "tier1_cases": f"{manifest_payload['tier1_property_cases_passed']}/{manifest_payload['tier1_property_cases_total']}"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

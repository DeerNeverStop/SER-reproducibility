"""Independent CPU-only plan gate and result integrity seal.

The verifier replays the design from raw metadata, rather than accepting a
compiler's success flag. Plan mode reads no SER predictions. Result mode
checks prediction arrays without computing scientific scores. Neither mode
starts training; a passed technical gate is not a scientific result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import numpy as np

from v3.data_design.core_verify import (
    VerificationError, _csv, read_json, representatives,
)

SEED = 20260906
SEXES = ("Female", "Male")
POLICIES = ("U", "R", "C")
WEIGHTS_SHA = "d8e745fb761c56d209431413bcc373f92784dc2c8d850199b6caa5cf05ca2c36"
HEX = re.compile(r"[0-9a-f]{64}\Z")
CONFIGS = {
    "ridge_wavlm": {"model": "ridge_a1_wavlm_base_plus", "feature_state": 12,
                    "alpha": 1.0, "class_weight": "balanced", "solver": "lsqr", "tol": 0.0001},
    "cnn": {"model": "cnn", "engine": "p1_frozen", "batch_size": 32, "epochs": 100,
            "patience": 101, "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1,
            "early_stopping": False, "checkpoint_selection": "min_val_loss"},
    "wavlm_ft": {"model": "wavlm_base_plus_ft", "engine": "wavlm_partial_ft", "batch_size": 16,
                 "crop_seconds": 3.0, "eval_cap_seconds": 10.0, "epochs": 15, "patience": 16,
                 "fp16": True, "lr_encoder": 5e-5, "lr_head": 0.001, "weight_decay": 0.01,
                 "trainable_layers": "top4+pool+head", "early_stopping": False,
                 "checkpoint_selection": "min_val_loss"},
}
BASE_SOURCES = {
    "v3/speaker_coverage/EXECUTION_PLAN.md",
    "v3/data_design/core_plan.py", "v3/data_design/core_run.py", "v3/data_design/core_verify.py",
    "v3/deploy/engines_deploy.py", "v2/plan_rc2/unit_configs.json", "v2/ser_v2/train.py",
    "v2/ser_v2/corpora.py", "v2/ser_v2/features.py", "v2/ser_v2/common.py", "v2/ser_v2/stats.py",
    "advanced_models.py", "tuned_standard_experiment.py", "advanced_experiment_utils.py",
    "experiment_utils.py", "features.py", "dataset.py", "fno_model.py",
}
UNIT_FIELDS = {
    "policy", "fold", "rotation", "draw", "fit", "val", "test", "block", "model",
    "seed_index", "train_seed", "B", "S", "P_global", "P_per_speaker", "R", "scenario",
    "config", "fit_manifest_sha256", "unit_id",
}


def require(ok, message):
    if not ok:
        raise VerificationError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def byte_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local(root, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "invalid relative path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == name, "unsafe relative path")
    root = Path(root).resolve()
    target = (root / name).resolve()
    require(target.is_relative_to(root), "path escapes supplied root")
    return target


def pinned(path, expected, label):
    require(isinstance(expected, str) and HEX.fullmatch(expected), f"invalid hash: {label}")
    require(Path(path).is_file() and byte_hash(path) == expected, f"byte hash mismatch: {label}")


def rank(values, salt):
    return sorted(values, key=lambda x: (digest([salt, x]), x))


def seed(*parts):
    return int(digest([SEED, *parts])[:16], 16) % (2**31 - 1)


def normalize(value):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 1 and value.size and np.isfinite(value).all(), "invalid reference vector")
    norm = np.linalg.norm(value)
    require(np.isfinite(norm) and norm > 1e-12, "zero reference norm")
    return value / norm


def roles_and_prompts(population, prompts, sex, fold, rotation):
    query, stop = set(), set()
    for group in SEXES:
        people = [s for s in population if sex[s] == group]
        query.update(rank(people, f"core:0:{group}:outer")[fold::5])
    for group in SEXES:
        eligible = [s for s in population if s not in query and sex[s] == group]
        stop.update(rank(eligible, f"core:0:{fold}:{group}:stop")[:4])
    prompt_order = rank(prompts, "core:prompt-pairs:20260905")
    query_prompts = prompt_order[rotation * 2:rotation * 2 + 2]
    remainder = rank(set(prompts) - set(query_prompts), f"core:prompt-common:{rotation}")
    return query, stop, remainder[:8], remainder[:6], query_prompts


def policy_reference(candidates, centers, sex, policy, fold, rotation, draw):
    """Brute-force candidate objective evaluation, independent of planner code."""
    people = sorted(candidates)
    vectors = np.stack([normalize(centers[s]) for s in people])
    distances = np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    order = rank(people, f"coverage:{SEED}:{fold}:{rotation}:{policy}")
    priority = {s: i for i, s in enumerate(order)}
    positions = {s: i for i, s in enumerate(people)}

    def metrics(selected):
        nearest = distances[:, [positions[s] for s in selected]].min(axis=1)
        return {"mean_nn1": float(nearest.mean()), "max_nn1": float(nearest.max())}

    trace = []
    if policy == "U":
        selected = []
        for group in SEXES:
            pool = [s for s in people if sex[s] == group]
            rng = np.random.Generator(np.random.PCG64(seed("U", fold, rotation, draw, group)))
            selected.extend(rng.choice(pool, 12, replace=False).tolist())
        termination = "uniform_quota_complete"
    else:
        selected = [order[0]]
        trace.append({"step": "initialize", "speaker": selected[0], **metrics(selected)})
        while len(selected) < 24:
            counts = Counter(sex[s] for s in selected)
            choices = [s for s in people if s not in selected and counts[sex[s]] < 12]
            require(choices, "policy exhausted sex quota")
            if policy == "R":
                chosen = min(choices, key=lambda s: (metrics(selected + [s])["mean_nn1"], priority[s]))
            else:
                chosen = min(choices, key=lambda s: (
                    -float(distances[positions[s], [positions[t] for t in selected]].min()), priority[s]))
            selected.append(chosen)
            trace.append({"step": "add", "speaker": chosen, **metrics(selected)})
        termination = "quota_complete"
        if policy == "R":
            termination = "no_strict_swap_improvement"
            for step in range(100):
                current, offers = metrics(selected)["mean_nn1"], []
                for outgoing in selected:
                    for incoming in people:
                        if incoming in selected or sex[incoming] != sex[outgoing]:
                            continue
                        trial = [incoming if s == outgoing else s for s in selected]
                        score = metrics(trial)["mean_nn1"]
                        if score < current - 1e-12:
                            offers.append((score, priority[outgoing], priority[incoming], outgoing, incoming))
                if not offers:
                    break
                _, _, _, outgoing, incoming = min(offers)
                selected[selected.index(outgoing)] = incoming
                trace.append({"step": "same_sex_swap", "swap": step + 1, "out": outgoing,
                              "in": incoming, **metrics(selected)})
            else:
                termination = "swap_limit_100"
    return sorted(selected), {"selection_trace": trace, "termination": termination, **metrics(selected)}


def layout_reference(selected, train_prompts, sex, fold, rotation, draw, attempt):
    require(type(attempt) is int and 0 <= attempt < 100, "invalid layout attempt")
    graph, assignment = {}, {}
    for group in SEXES:
        graph[group] = []
        for pair in range(6):
            chosen = rank(list(combinations(range(8), 4)),
                          f"layout:{SEED}:{fold}:{rotation}:{draw}:{attempt}:{group}:{pair}")[0]
            graph[group].extend([list(chosen), [i for i in range(8) if i not in chosen]])
        order = rank([s for s in selected if sex[s] == group],
                     f"layout-mapping:{SEED}:{fold}:{rotation}:{group}")
        for person, indices in zip(order, graph[group]):
            assignment[person] = [train_prompts[i] for i in indices]
    return graph, assignment


def compare_trace(actual, expected):
    require(actual.get("termination") == expected["termination"], "selection termination mismatch")
    for field in ("mean_nn1", "max_nn1"):
        require(np.isclose(actual.get(field, np.nan), expected[field], rtol=0, atol=1e-10),
                "selection objective mismatch")
    a, b = actual.get("selection_trace"), expected["selection_trace"]
    require(isinstance(a, list) and len(a) == len(b), "selection trace length mismatch")
    for left, right in zip(a, b):
        require(set(left) == set(right), "selection trace fields mismatch")
        for key, value in right.items():
            if key in {"mean_nn1", "max_nn1"}:
                require(np.isclose(left[key], value, rtol=0, atol=1e-10), "trace objective mismatch")
            else:
                require(left[key] == value, f"selection trajectory mismatch: {key}")


def validate_panel(paths, rep, sex):
    require(isinstance(paths, list) and paths == sorted(set(paths)) and len(paths) == 576,
            "training panel needs 576 distinct sorted recordings")
    require(all(p in rep for p in paths), "fit contains unknown/nonrepresentative recording")
    rows = [rep[p] for p in paths]
    people = {r["speaker"] for r in rows}
    require(len(people) == 24 and Counter(sex[s] for s in people) == {"Female": 12, "Male": 12},
            "training speaker/sex quota mismatch")
    require(len({r["sha256"] for r in rows}) == 576, "training contains duplicate bytes")
    require(set(Counter(r["speaker"] for r in rows).values()) == {24}, "per-speaker audio quota")
    counts = Counter(r["prompt"] for r in rows)
    require(len(counts) == 8 and set(counts.values()) == {72}, "eight-prompt equal exposure quota")
    cells = defaultdict(set)
    for row in rows:
        cells[row["speaker"], row["prompt"]].add(row["label"])
    require(len(cells) == 96 and all(v == set(range(6)) for v in cells.values()), "six-emotion cell quota")
    require(set(Counter(s for s, _ in cells).values()) == {4}, "four prompts per speaker required")
    sex_prompt = Counter((sex[s], p) for s, p in cells)
    require(len(sex_prompt) == 16 and set(sex_prompt.values()) == {6}, "six people per sex/prompt required")
    return people


def verify_document(plan, manifest_rows, sex, embeddings, capacity, diagnostics, *, complete=True):
    """Independent metadata/selector gate; fixtures can never permit a formal run."""
    require(plan.get("schema") == "ser-speaker-coverage-1"
            and plan.get("program") == "SER26-SPEAKER-COVERAGE-1", "wrong plan schema/program")
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "plan hash mismatch")
    design = plan.get("design", {})
    constants = {"master_seed": SEED, "speaker_folds": 5, "prompt_rotations": 6, "draws": 3,
                 "status": "prospectively_specified_estimation_historically_exposed_corpus",
                 "outer_roles": "core_plan.roles(meta,0,fold); fixed across draw",
                 "policies": list(POLICIES), "blocks": {"core": 540, "ft": 180}, "expected_units": 720,
                 "ft_draws": [0], "ft_train_seeds": 2,
                 "pilot_units": {"core": 18, "ft": 1, "included_in_formal_units": False},
                 "B": 576, "S": 24, "stop_speakers": 8, "R_swap_limit": 100,
                 "R_strict_improvement_tolerance": 1e-12, "layout_attempt_limit": 100,
                 "reference": "eight_train_prompt_neutral_representatives",
                 "centroid": "normalize_each_then_mean_then_normalize_float64",
                 "selection_metric": "cosine_nn1_on_same_eligible_training_pool_including_selected",
                 "selection_queries": "none; no test/stop embeddings or scores used",
                 "primary_contrasts": {
                     "primary": {"model": "cnn", "contrast": "C-U", "endpoint": "person_mean_uar_pp",
                                 "inference": "two_sided_estimation"},
                     "required_second_model": {"model": "wavlm_ft", "contrast": "C-U"},
                     "secondary": "R-U; Ridge contrasts; other fixed contrasts",
                     "q25": "descriptive_distribution_quantile_difference", "sign_dependent_gate": False},
                 "scientific_scores_hidden_until_complete": True}
    for key, value in constants.items():
        require(design.get(key) == value, f"design constant mismatch: {key}")
    if complete:
        require(not design.get("fixture") and not plan["input"].get("fixture"), "fixture cannot pass formal gate")
        require(all(design.get(k) is True for k in ("runtime_ready", "sourcefreeze_complete", "ft_runtime_ready")),
                "runtime/source freeze incomplete")
    rep, population_report = representatives(manifest_rows, complete=complete)
    for row in manifest_rows:
        expected = ("angry", "disgust", "fearful", "happy", "neutral", "sad")[int(row["label_index"])]
        require(row.get("label") == expected, "manifest label name/index mismatch")
    population = sorted({r["speaker"] for r in rep.values()})
    prompts = sorted({r["prompt"] for r in rep.values()})
    require(len(prompts) == 12 and all(sex.get(s) in SEXES for s in population), "metadata dimensions/sex invalid")
    cells = {(r["speaker"], r["prompt"], r["label"]): p for p, r in rep.items()}
    units = plan.get("units", [])
    require(len(units) == 720, "formal design requires 720 units")
    expected = set(product(("ridge_wavlm", "cnn"), range(5), range(6), range(3), POLICIES, (0,)))
    expected |= set(product(("wavlm_ft",), range(5), range(6), (0,), POLICIES, (0, 1)))
    grid, contexts, ids = set(), defaultdict(list), set()
    for unit in units:
        require(set(unit) == UNIT_FIELDS, "unit field set mismatch")
        require(unit["unit_id"] == digest({k: v for k, v in unit.items() if k != "unit_id"}), "unit hash mismatch")
        require(unit["unit_id"] not in ids, "duplicate unit identity")
        ids.add(unit["unit_id"])
        for key in ("fold", "rotation", "draw", "seed_index", "train_seed"):
            require(type(unit[key]) is int, f"unit integer field invalid: {key}")
        coordinate = tuple(unit[k] for k in ("model", "fold", "rotation", "draw", "policy", "seed_index"))
        require(coordinate in expected and coordinate not in grid, "missing/duplicate factorial coordinate")
        grid.add(coordinate)
        require(unit["config"] == CONFIGS[unit["model"]], "model config differs from frozen contract")
        block = "ft" if unit["model"] == "wavlm_ft" else "core"
        require(unit["block"] == block and unit["train_seed"] == seed(
            "training", block, unit["fold"], unit["rotation"], unit["draw"], unit["seed_index"]), "seed/block mismatch")
        require(all(unit[k] == v for k, v in {"B": 576, "S": 24, "P_global": 8,
                    "P_per_speaker": 4, "R": 1, "scenario": "prompt_new"}.items()), "panel constants mismatch")
        require(unit["fit_manifest_sha256"] == digest(unit["fit"]), "fit manifest hash mismatch")
        contexts[unit["fold"], unit["rotation"]].append(unit)
    require(grid == expected, "incomplete factorial grid")
    for document, schema in ((capacity, "ser-speaker-coverage-capacity-1"),
                             (diagnostics, "ser-speaker-coverage-selection-1")):
        require(document.get("schema") == schema and document.get("plan_sha256") == plan["plan_sha256"],
                "auxiliary document plan identity mismatch")
        require(len(document.get("conditions", [])) == 30, "auxiliary conditions incomplete")
    cap = {(c["fold"], c["rotation"]): c for c in capacity["conditions"]}
    diag = {(c["fold"], c["rotation"]): c for c in diagnostics["conditions"]}
    require(set(cap) == set(diag) == set(contexts), "duplicate/missing auxiliary condition")
    require(capacity.get("people") == len(population) and capacity.get("representative_cells") == len(rep),
            "capacity population mismatch")
    require(capacity.get("by_model") == dict(Counter(u["model"] for u in units)), "capacity model counts mismatch")
    require(diagnostics.get("no_heldout_distance_gate") is True, "heldout-distance gate not prohibited")
    require(diagnostics.get("metric") == "pool_in_sample_cosine_nn1", "diagnostic metric mismatch")
    for (fold, rotation), group in sorted(contexts.items()):
        query, stop, train_prompts, stop_prompts, query_prompts = roles_and_prompts(population, prompts, sex, fold, rotation)
        candidate = sorted(s for s in population if s not in query | stop and all(
            (s, p, c) in cells for p in train_prompts for c in range(6)))
        require(len(stop) == 8 and all(sum(sex[s] == g for s in candidate) >= 12 for g in SEXES), "candidate capacity invalid")
        refs = {s: [cells[s, p, 4] for p in sorted(train_prompts)] for s in candidate}
        centers = {s: normalize(np.mean([normalize(embeddings[p]) for p in refs[s]], axis=0)) for s in candidate}
        expected_val = sorted(p for p, r in rep.items() if r["speaker"] in stop and r["prompt"] in stop_prompts)
        expected_test = sorted(p for p, r in rep.items() if r["speaker"] in query and r["prompt"] in query_prompts)
        c = cap[fold, rotation]
        checks = {"eligible_speakers": candidate, "test_speakers": sorted(query), "stop_speakers": sorted(stop),
                  "train_prompts": train_prompts, "stop_prompts": stop_prompts, "query_prompts": query_prompts,
                  "reference_paths": refs, "reference_sha256": digest(refs), "eligible": len(candidate),
                  "eligible_female": sum(sex[s] == "Female" for s in candidate),
                  "eligible_male": sum(sex[s] == "Male" for s in candidate),
                  "train_pool": len(population) - len(query | stop), "val_rows": len(expected_val), "test_rows": len(expected_test)}
        for key, value in checks.items():
            require(c.get(key) == value, f"candidate/reference metadata mismatch: {key}")
        require(c.get("centroid_sha256") == digest({s: centers[s].tolist() for s in candidate}), "centroid hash mismatch")
        for evaluation, people in ((expected_val, stop), (expected_test, query)):
            require(all({rep[p]["label"] for p in evaluation if rep[p]["speaker"] == s} == set(range(6))
                        for s in people), "evaluation speaker lacks emotion class")
        query_count = Counter((rep[p]["speaker"], rep[p]["label"]) for p in expected_test)
        require(c.get("minimum_query_per_person_class") == min(query_count.values()), "query class capacity mismatch")
        selection_rows = diag[fold, rotation].get("selections", [])
        selections = {(s["policy"], s["draw"]): s for s in selection_rows}
        require(len(selection_rows) == len(selections) == 5 and set(selections) == {
            ("U", 0), ("U", 1), ("U", 2), ("R", None), ("C", None)}, "selection diagnostic grid mismatch")
        chosen = {}
        for (policy, draw), record in selections.items():
            selected, trace = policy_reference(candidate, centers, sex, policy, fold, rotation, draw or 0)
            require(record.get("selected_speakers") == selected, "selected policy speakers mismatch")
            require(record.get("fold") == fold and record.get("rotation") == rotation
                    and record.get("evaluation_population") == "same_complete_training_candidate_pool_including_selected"
                    and record.get("distance_definition") == "cosine_nn1", "selection objective population mismatch")
            compare_trace(record, trace)
            for d in (range(3) if policy != "U" else (draw,)):
                chosen[policy, d] = selected
        layout_rows = diag[fold, rotation].get("layouts", [])
        layouts = {x["draw"]: x for x in layout_rows}
        require(len(layout_rows) == len(layouts) == 3 and set(layouts) == {0, 1, 2}, "layout draw grid mismatch")
        previous = {p: set() for p in POLICIES}
        for draw in range(3):
            attempt = layouts[draw]["attempt"]
            require(type(attempt) is int and 0 <= attempt < 100, "invalid layout attempt")
            for trial in range(attempt + 1):
                hashes = {}
                for policy in POLICIES:
                    _, assignment = layout_reference(chosen[policy, draw], train_prompts, sex,
                                                     fold, rotation, draw, trial)
                    paths = sorted(cells[s, p, label] for s, texts in assignment.items()
                                   for p in texts for label in range(6))
                    hashes[policy] = digest(paths)
                collision = any(hashes[p] in previous[p] for p in POLICIES)
                require(collision if trial < attempt else not collision,
                        "layout did not use first predefined nonduplicate attempt")
            for policy in POLICIES:
                previous[policy].add(hashes[policy])
        unique_fits = defaultdict(set)
        for unit in group:
            require(unit["val"] == expected_val and unit["test"] == expected_test,
                    "fixed outer/stop evaluation differs across policy, draw or model")
            people = validate_panel(unit["fit"], rep, sex)
            require(people == set(chosen[unit["policy"], unit["draw"]]), "fit people differ from policy selection")
            require(people.isdisjoint(query | stop) and query.isdisjoint(stop), "speaker role overlap")
            graph, assignment = layout_reference(sorted(people), train_prompts, sex, fold, rotation,
                                                unit["draw"], layouts[unit["draw"]]["attempt"])
            require(layouts[unit["draw"]].get("abstract_slots") == graph, "shared abstract layout mismatch")
            expected_fit = sorted(cells[s, p, label] for s, texts in assignment.items() for p in texts for label in range(6))
            require(unit["fit"] == expected_fit, "training assignment differs from fixed shared slot graph")
            role_hashes = [{rep[p]["sha256"] for p in unit[k]} for k in ("fit", "val", "test")]
            require(all(a.isdisjoint(b) for a, b in combinations(role_hashes, 2)), "bytes shared across roles")
            require({rep[p]["prompt"] for p in unit["test"]}.isdisjoint(
                {rep[p]["prompt"] for p in unit["fit"] + unit["val"]}), "query text leakage")
            unique_fits[unit["policy"]].add(tuple(unit["fit"]))
        require(all(len(unique_fits[p]) == 3 for p in POLICIES), "draws repeat identical fit manifests")
    return {"pass": True, "formal_allowed": False, "scope": "plan_structure_only" if complete else "test_fixture_only",
            "plan_sha256": plan["plan_sha256"], "n_units": 720, "blocks": {"core": 540, "ft": 180},
            "units_by_model": dict(Counter(u["model"] for u in units)), "reference_contexts_checked": 30,
            "policy_trajectories_checked": 150, "representatives": population_report,
            "scientific_outcomes_read": False, "scientific_validity_established": False}


def clean_rows(raw):
    groups = defaultdict(list)
    for row in raw:
        groups[row["sha256"]].append(row)
    return sorted((min(group, key=lambda r: r["relative_path"]) for group in groups.values()
                   if len({r["label_index"] for r in group}) == 1
                   and min(r["relative_path"] for r in group) != "1076_MTI_SAD_XX.wav"),
                  key=lambda r: r["relative_path"])


def verify_audio(clean, root):
    require(root is not None, "formal gate requires --audio-root to independently verify raw audio bytes")
    total = 0
    for row in clean:
        path = local(root, row["relative_path"])
        pinned(path, row["sha256"], "raw audio " + row["relative_path"])
        require(path.stat().st_size == int(row["bytes"]), "raw audio byte length mismatch")
        total += path.stat().st_size
    return {"audio_files_verified": len(clean), "audio_bytes_verified": total}


def verify_features(root, inp, expected_paths):
    require(set(inp["feature_files"]) == set(inp["feature_sha256"]) == {"logmel", "wavlm_base_plus"}, "feature pin set mismatch")
    for kind, name in inp["feature_files"].items():
        require(PurePosixPath(name).name == name and "SYNTHETIC" not in name.upper(), "invalid feature basename")
        path = local(root, name)
        pinned(path, inp["feature_sha256"][kind], kind)
        side = read_json(path.with_suffix(".json"))
        require(side.get("sha256") == inp["feature_sha256"][kind] and side.get("corpus") == "cremad", "feature receipt mismatch")
        if kind == "wavlm_base_plus":
            require(side.get("encoder") == kind and side.get("weights_sha256") == WEIGHTS_SHA, "WavLM feature weights mismatch")
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == {"X", "paths"}, "unexpected feature arrays")
            paths = data["paths"]
            require(paths.ndim == 1 and paths.dtype.kind == "U" and paths.tolist() == expected_paths, "feature path population mismatch")
        with zipfile.ZipFile(path) as archive, archive.open("X.npy") as stream:
            version = np.lib.format.read_magic(stream)
            require(version in {(1, 0), (2, 0)}, "unsupported numpy header")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = reader(stream)
            suffix, required_dtype = ((64, 128), np.dtype("float32")) if kind == "logmel" else ((13, 768), np.dtype("float16"))
            require(shape == (len(expected_paths), *suffix) and dtype == required_dtype, "feature shape/dtype mismatch")
            remaining = int(np.prod(shape)) * dtype.itemsize
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                require(chunk and len(chunk) % dtype.itemsize == 0, "truncated feature payload")
                require(np.isfinite(np.frombuffer(chunk, dtype=dtype)).all(), "nonfinite feature values")
                remaining -= len(chunk)
            require(not stream.read(1), "trailing feature payload bytes")


def verify_asv(path, inp, clean, repo):
    path = Path(path)
    require(path.name == inp.get("asv_path"), "ASV basename mismatch")
    pinned(path, inp.get("asv_sha256"), "ASV NPZ")
    receipt_path = path.with_suffix(".json")
    require(inp.get("asv_receipt_file") == receipt_path.name, "ASV receipt filename mismatch")
    pinned(receipt_path, inp.get("asv_receipt_sha256"), "ASV receipt")
    receipt = read_json(receipt_path)
    identity = receipt.get("identity", {})
    require(receipt.get("sha256") == inp["asv_sha256"] and identity.get("schema") == "ser-asv-embeddings-1"
            and identity.get("manifest_sha256") == inp["manifest_sha256"] and identity.get("corpus") == "cremad", "ASV provenance mismatch")
    require(identity.get("model_id") == "speechbrain/spkrec-ecapa-voxceleb"
            and re.fullmatch(r"[0-9a-f]{40}", identity.get("model_revision", "")), "ASV model/revision invalid")
    pins = identity.get("model_sha256", {})
    require("embedding_model.ckpt" in pins and all(HEX.fullmatch(v) for v in pins.values()), "ASV model weight pins invalid")
    require(identity.get("source_sha256") == byte_hash(repo / "v3/speaker_coverage/extract_asv.py"), "ASV extraction source mismatch")
    require(receipt.get("scientific_SER_scores_computed") is False and identity.get("rows") == len(clean), "ASV extraction population/scope mismatch")
    quality_path = local(path.parent, receipt.get("audio_quality_file"))
    pinned(quality_path, receipt.get("audio_quality_sha256"), "ASV audio receipt")
    quality = _csv(quality_path)
    require(len(quality) == len(clean) and [r["path"] for r in quality] == [r["relative_path"] for r in clean], "ASV audio receipt population mismatch")
    for actual, source in zip(quality, clean):
        require(actual.get("sha256") == source["sha256"], "ASV source audio hash mismatch")
    require(receipt.get("source_audio_bytes_verified") == sum(int(r["bytes"]) for r in clean), "ASV source byte total mismatch")
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"paths", "embeddings"}, "ASV array set mismatch")
        names, values = data["paths"], data["embeddings"]
        require(names.ndim == 1 and names.dtype.kind == "U" and names.tolist() == [r["relative_path"] for r in clean], "ASV path population/order mismatch")
        require(values.ndim == 2 and values.dtype.kind == "f" and values.shape == (len(clean), 192), "ECAPA embedding shape mismatch")
        require(inp.get("asv_rows") == len(clean) and inp.get("asv_dimension") == 192 and receipt.get("shape") == [len(clean), 192], "ASV declared shape mismatch")
        require(np.isfinite(values).all(), "nonfinite ASV vectors")
        return {name: values[i].astype(np.float64) for i, name in enumerate(names.tolist())}


def verify_wavlm(path, inp):
    require(path is not None and Path(path).name == inp.get("wavlm_model_file"), "formal FT gate requires pinned --wavlm-model")
    pinned(path, inp.get("wavlm_model_sha256"), "WavLM model file")
    require(inp.get("wavlm_weights_sha256") == WEIGHTS_SHA, "WavLM semantic pin mismatch")
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and state and all(isinstance(v, torch.Tensor) for v in state.values()), "WavLM file is not a tensor state dictionary")
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(state[key].detach().cpu().numpy().tobytes())
    require(h.hexdigest() == WEIGHTS_SHA, "WavLM semantic state hash mismatch")


def audit_reference_access(raw, sex, embeddings, capacity, *, complete=True):
    """Secondary instrumented source check, after independent reconstruction."""
    from v3.speaker_coverage import plan as compiler
    rep, _ = representatives(raw, complete=complete)
    cells = {(r["speaker"], r["prompt"], r["label"]): {
        "relative_path": p, "speaker": r["speaker"], "sentence": r["prompt"], "label_index": r["label"],
        "label": ("angry", "disgust", "fearful", "happy", "neutral", "sad")[r["label"]], "sha256": r["sha256"]}
        for p, r in rep.items()}
    meta = {"cells": cells, "sex": sex, "speakers": sorted({r["speaker"] for r in rep.values()}),
            "prompts": sorted({r["prompt"] for r in rep.values()})}

    class ReferenceOnly(Mapping):
        def __init__(self, permitted):
            self.permitted, self.reads = set(permitted), set()
        def __getitem__(self, key):
            require(key in self.permitted, "planner attempted nontraining reference read")
            self.reads.add(key)
            return embeddings[key]
        def __iter__(self):
            raise VerificationError("planner attempted global embedding iteration")
        def __len__(self):
            return len(self.permitted)

    for context in capacity["conditions"]:
        allowed = {p for paths in context["reference_paths"].values() for p in paths}
        guard = ReferenceOnly(allowed)
        compiler.plan_context(meta, guard, context["fold"], context["rotation"])
        require(guard.reads == allowed, "planner reference read coverage mismatch")
    return 30


def verify_plan(repo, plan_path, features, asv_path, wavlm_model, audio_root=None):
    repo, plan_path = Path(repo).resolve(), Path(plan_path).resolve()
    require(repo == Path(__file__).resolve().parents[2], "verifier import checkout differs from --repo")
    document = read_json(plan_path)
    inp = document["input"]
    for name in ("manifest", "demographics"):
        pinned(local(repo, inp[name + "_path"]), inp[name + "_sha256"], name)
    sources = document.get("source_sha256", {})
    required_sources = BASE_SOURCES | {p.relative_to(repo).as_posix() for p in (repo / "v3/speaker_coverage").glob("*.py")}
    require(required_sources <= set(sources) and {"v3/speaker_coverage/run.py", "v3/speaker_coverage/verify.py"} <= set(sources), "incomplete source freeze")
    for name, sha in sources.items():
        pinned(local(repo, name), sha, "source " + name)
    raw = _csv(local(repo, inp["manifest_path"]))
    representatives(raw, complete=True)
    clean = clean_rows(raw)
    require(len(clean) == 7435, "hygiene population mismatch")
    sex = {}
    for row in _csv(local(repo, inp["demographics_path"])):
        require(row.get("ActorID") not in sex and row.get("Sex") in SEXES, "duplicate/invalid demographics")
        sex[row["ActorID"]] = row["Sex"]
    audio = verify_audio(clean, audio_root)
    verify_features(features, inp, [r["relative_path"] for r in clean])
    embeddings = verify_asv(asv_path, inp, clean, repo)
    verify_wavlm(wavlm_model, inp)
    capacity = read_json(plan_path.parent / "capacity.json")
    diagnostics = read_json(plan_path.parent / "selection_diagnostics.json")
    report = verify_document(document, raw, sex, embeddings, capacity, diagnostics)
    access_contexts = audit_reference_access(raw, sex, embeddings, capacity)
    report.update(audio, formal_allowed=True, scope="complete_real_plan_and_actual_input_bytes",
                  source_files_checked=len(sources), feature_bytes_checked=True, asv_receipt_checked=True,
                  wavlm_file_and_state_hash_checked=True, planner_reference_access_audited_contexts=access_contexts,
                  authorization_scope="technical_gate_only; user authorization and runtime cost guard are separate")
    return report


def result_predictions(path, unit, metadata):
    """Check alignment and arithmetic without aggregating any SER metric."""
    expected = {"paths", "labels", "logits", "pred", "proba", "last_logits", "last_pred", "last_proba"}
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == expected, "prediction artifact key mismatch")
        require(arrays["paths"].dtype.kind == "U" and arrays["paths"].tolist() == unit["test"], "prediction path order mismatch")
        y = arrays["labels"]
        require(y.dtype == np.int64 and y.shape == (len(unit["test"]),) and
                y.tolist() == [int(metadata[p]["label_index"]) for p in unit["test"]], "prediction labels differ from actual manifest")
        for prefix in ("", "last_"):
            logits, proba, pred = (arrays[prefix + name] for name in ("logits", "proba", "pred"))
            require(logits.dtype == np.float64 and logits.shape == (len(y), 6) and np.isfinite(logits).all(), "invalid logits")
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            expected_proba = exp / exp.sum(axis=1, keepdims=True)
            require(proba.dtype == np.float64 and proba.shape == logits.shape and np.isfinite(proba).all()
                    and np.allclose(proba, expected_proba, atol=1e-12, rtol=1e-12), "invalid probability arithmetic")
            require(pred.dtype == np.int64 and pred.shape == y.shape and np.array_equal(pred, logits.argmax(axis=1)), "invalid argmax predictions")


@lru_cache(maxsize=1)
def ft_delta_specification():
    """Derive complete expected keys offline; never call bundle.get_model()."""
    import torchaudio
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    encoder = torchaudio.models.wavlm_model(**bundle._params)
    allowed = tuple(f"encoder.transformer.layers.{i}." for i in range(8, 12))
    parameters = {"enc." + name: (tuple(value.shape), value.dtype) for name, value in encoder.named_parameters()
                  if name.startswith(allowed)}
    import torch
    parameters.update({"fc.weight": ((6, 768), torch.float32), "fc.bias": ((6,), torch.float32)})
    buffers = {"enc." + name: (tuple(value.shape), value.dtype) for name, value in encoder.named_buffers()}
    return parameters, buffers, digest(bundle._params)


def result_checkpoint(path, unit, receipt, plan_input):
    if unit["model"] == "ridge_wavlm":
        with np.load(path, allow_pickle=False) as arrays:
            required = {"coef", "intercept", "classes", "scaler_mean", "scaler_scale", "scaler_var", "n_features_in", "n_samples_seen"}
            require(set(arrays.files) == required, "ridge checkpoint field mismatch")
            require(arrays["coef"].shape == (6, 768) and arrays["intercept"].shape == (6,)
                    and np.array_equal(arrays["classes"], np.arange(6)), "ridge checkpoint classifier shape/classes mismatch")
            require(all(arrays[k].shape == (768,) for k in ("scaler_mean", "scaler_scale", "scaler_var")), "ridge scaler shape mismatch")
            require(all(np.isfinite(arrays[k]).all() for k in arrays.files), "nonfinite ridge checkpoint")
            require((arrays["scaler_scale"] > 0).all() and (arrays["scaler_var"] >= 0).all(), "invalid ridge scaler")
            require(arrays["n_features_in"].shape == arrays["n_samples_seen"].shape == ()
                    and arrays["n_features_in"].dtype == arrays["n_samples_seen"].dtype == np.int64
                    and int(arrays["n_features_in"]) == 768 and int(arrays["n_samples_seen"]) == len(unit["fit"]), "ridge scaler training population mismatch")
        return
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict), "invalid tensor checkpoint container")
    schema = "ser-speaker-coverage-cnn-checkpoint-1" if unit["model"] == "cnn" else "ser-speaker-coverage-ft-delta-1"
    require(checkpoint.get("schema") == schema and checkpoint.get("config") == unit["config"] and
            checkpoint.get("train_seed") == unit["train_seed"] and checkpoint.get("best_epoch") == receipt["best_epoch"]
            and checkpoint.get("epochs_run") == receipt["epochs_run"], "checkpoint identity mismatch")
    best, last = checkpoint.get("best_state"), checkpoint.get("last_state")
    require(isinstance(best, dict) and best and isinstance(last, dict) and best.keys() == last.keys(), "missing/inconsistent best and last states")
    for key in best:
        a, b = best[key], last[key]
        require(isinstance(key, str) and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.device.type == b.device.type == "cpu" and a.dtype == b.dtype and a.shape == b.shape
                and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "invalid best/last checkpoint tensor")
    if unit["model"] == "cnn":
        import advanced_models
        architecture = {"width": 48, "n_blocks": 3, "kernel_size": 5, "dilation_base": 1, "dropout": 0.1}
        reference = advanced_models.build_neural_model("cnn", n_mels=64, n_classes=6, config=architecture).state_dict()
        require(best.keys() == reference.keys() and all(best[k].shape == reference[k].shape and best[k].dtype == reference[k].dtype
                                                       for k in reference), "CNN checkpoint is not a complete fixed-architecture state")
    else:
        require(checkpoint.get("base_file_sha256") == plan_input["wavlm_model_sha256"]
                and checkpoint.get("base_state_sha256") == WEIGHTS_SHA, "FT delta base mismatch")
        trainable, buffers = checkpoint.get("trainable_parameter_names"), checkpoint.get("buffer_names")
        require(isinstance(trainable, list) and isinstance(buffers, list) and len(set(trainable)) == len(trainable)
                and len(set(buffers)) == len(buffers) and not set(trainable).intersection(buffers)
                and set(best) == set(trainable) | set(buffers), "FT delta parameter/buffer coverage mismatch")
        require({n.split(".")[4] for n in trainable if n.startswith("enc.encoder.transformer.layers.")} == {"8", "9", "10", "11"}
                and {"fc.weight", "fc.bias"} <= set(trainable)
                and all(n.startswith(tuple(f"enc.encoder.transformer.layers.{i}." for i in range(8, 12)))
                        or n in ("fc.weight", "fc.bias") for n in trainable), "FT delta trainability differs from top4 plus head")
        require(best["fc.weight"].shape == (6, 768) and best["fc.bias"].shape == (6,), "FT head shape mismatch")
        require(all(isinstance(checkpoint.get(k), str) and HEX.fullmatch(checkpoint[k])
                    for k in ("bundle_params_sha256", "frozen_parameter_sha256")), "FT base/bundle provenance missing")
        expected_parameters, expected_buffers, bundle_sha = ft_delta_specification()
        expected_state = {**expected_parameters, **expected_buffers}
        require(set(trainable) == expected_parameters.keys() and set(buffers) == expected_buffers.keys()
                and checkpoint["bundle_params_sha256"] == bundle_sha
                and all(tuple(best[k].shape) == shape and best[k].dtype == dtype for k, (shape, dtype) in expected_state.items()),
                "FT delta omits or changes an expected parameter/buffer")


def verify_result_unit(unit, unit_root, plan, phase, metadata):
    """Only the single DONE-selected attempt counts toward completion."""
    unit_root = Path(unit_root)
    done_path = unit_root / "DONE"
    done = read_json(done_path)
    identity = {"unit_id": unit["unit_id"], "plan_sha256": plan["plan_sha256"], "phase": phase,
                "model": unit["model"], "block": unit["block"]}
    require(done.get("schema") == "ser-speaker-coverage-done-1" and all(done.get(k) == v for k, v in identity.items()), "DONE identity mismatch")
    attempt = done.get("attempt")
    require(type(attempt) is int and attempt > 0, "invalid selected attempt")
    attempt_dirs = list((unit_root / "attempts").iterdir())
    require(all(p.is_dir() and re.fullmatch(r"[0-9]{4,}", p.name) and int(p.name) > 0
                and p.name == f"{int(p.name):04d}" for p in attempt_dirs), "invalid attempt directory")
    require(attempt_dirs and max(int(p.name) for p in attempt_dirs) == attempt, "unsealed later attempt exists")
    prefix = f"attempts/{attempt:04d}/"
    checkpoint_name = "checkpoint.npz" if unit["model"] == "ridge_wavlm" else "checkpoint.pt"
    names = {prefix + n for n in ("predictions.npz", "receipt.json", "history.json", checkpoint_name)}
    artifacts = done.get("artifacts", {})
    require(set(artifacts) == names, "DONE artifact set mismatch")
    for name, sha in artifacts.items():
        pinned(local(unit_root, name), sha, "result " + name)
    receipt = read_json(unit_root / prefix / "receipt.json")
    require(receipt.get("schema") == "ser-speaker-coverage-result-1" and all(receipt.get(k) == v for k, v in identity.items())
            and receipt.get("attempt") == attempt, "result receipt identity mismatch")
    require(receipt.get("unit_config_sha256") == digest(unit["config"]) and receipt.get("input_sha256") == digest(plan["input"]), "result receipt input/config hash mismatch")
    require(isinstance(receipt.get("environment"), dict) and receipt["environment"], "missing execution environment")
    for k in ("fit_seconds", "wall_seconds", "peak_cuda_bytes"):
        require(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) and receipt[k] >= 0, "invalid runtime receipt")
    require(receipt["wall_seconds"] + 1e-6 >= receipt["fit_seconds"], "fit duration exceeds unit wall duration")
    require(receipt.get("checkpoint_reload_verified") is True, "checkpoint replay not confirmed")
    formats = {"cnn": "cnn_full_best_last", "ridge_wavlm": "ridge_npz", "wavlm_ft": "wavlm_delta_best_last"}
    require(receipt.get("checkpoint_format") == formats[unit["model"]], "checkpoint format mismatch")
    reload_keys = ("checkpoint_reload_best_max_abs_diff", "checkpoint_reload_last_max_abs_diff", "checkpoint_reload_max_abs_diff")
    require(all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) and receipt[k] >= 0 for k in reload_keys)
            and receipt[reload_keys[2]] == max(receipt[k] for k in reload_keys[:2]), "invalid best/last checkpoint replay receipt")
    history = read_json(unit_root / prefix / "history.json")
    epochs = 0 if unit["model"] == "ridge_wavlm" else unit["config"]["epochs"]
    require(type(receipt.get("epochs_run")) is int and receipt["epochs_run"] == epochs and isinstance(history, list)
            and [r.get("epoch") for r in history] == list(range(1, epochs + 1)), "fixed epoch/history budget mismatch")
    require(all(set(r) == {"epoch", "train_loss", "val_loss", "optimizer_steps"} for r in history), "unexpected history columns/scientific scores")
    for row in history:
        require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in ("train_loss", "val_loss")), "invalid epoch loss")
        require(type(row.get("optimizer_steps")) is int and row["optimizer_steps"] == math.ceil(len(unit["fit"]) / unit["config"]["batch_size"]), "optimizer step budget mismatch")
    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"] if history else None
    require(receipt.get("best_epoch") == best_epoch, "best checkpoint does not follow minimum validation loss")
    result_predictions(unit_root / prefix / "predictions.npz", unit, metadata)
    result_checkpoint(unit_root / prefix / checkpoint_name, unit, receipt, plan["input"])
    return {"done_sha256": byte_hash(done_path), "attempt": attempt, "ignored_uncommitted_attempts": len(attempt_dirs) - 1,
            "fit_seconds": receipt["fit_seconds"], "wall_seconds": receipt["wall_seconds"],
            "artifact_bytes": sum((unit_root / name).stat().st_size for name in artifacts)}


def verify_phase_receipts(phase_root, plan, phase, units, entries):
    identity = read_json(phase_root / "identity.json")
    require(identity == {"schema": "ser-speaker-coverage-output-1", "plan_sha256": plan["plan_sha256"],
                         "phase": phase, "program": "SER26-SPEAKER-COVERAGE-1"}, "output phase identity mismatch")
    require(read_json(phase_root / "plan_snapshot.json") == plan, "executed plan snapshot mismatch")
    gate = read_json(phase_root / "verification.json")
    require(gate.get("pass") is True and gate.get("formal_allowed") is True and gate.get("plan_sha256") == plan["plan_sha256"]
            and gate.get("scope") == "complete_real_plan_and_actual_input_bytes", "missing complete independent plan gate receipt")
    ledger_path = phase_root / "ledger.jsonl"
    raw = ledger_path.read_bytes()
    require(raw and raw.endswith(b"\n"), "empty/truncated runtime ledger")
    events = [json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(VerificationError("nonfinite ledger")))
              for line in raw.decode("utf-8").splitlines()]
    known, active, starts, done, failed = {u["unit_id"]: u for u in units}, set(), {}, {}, set()
    for row in events:
        event = row.get("event")
        require(event in ("invocation_start", "unit_start", "unit_done", "unit_failed", "invocation_end", "stale_lock_recovered"), "unknown ledger event")
        require(datetime.fromisoformat(row["at"]).tzinfo is not None, "ledger timestamp must include timezone")
        if event == "stale_lock_recovered":
            require(type(row.get("previous_pid")) is int and row["previous_pid"] > 0, "invalid stale lock recovery record")
        elif event == "invocation_start":
            selected = row.get("selected_unit_ids", [])
            require(row.get("plan_sha256") == plan["plan_sha256"] and row.get("phase") == phase
                    and isinstance(selected, list) and selected and len(selected) == len(set(selected))
                    and set(selected) <= known.keys() and row.get("selection_sha256") == digest(selected)
                    and all(known[i]["model"] == row.get("model") for i in selected), "ledger invocation identity/selection mismatch")
            active = set(selected)
        elif event in ("unit_start", "unit_done", "unit_failed"):
            uid, attempt = row.get("unit_id"), row.get("attempt")
            require(uid in active and type(attempt) is int and attempt > 0, "ledger unit outside invocation")
            key = uid, attempt
            if event == "unit_start":
                require(key not in starts and uid not in done, "duplicate/restarted successful ledger unit")
                starts[key] = row
            else:
                require(key in starts and key not in failed and uid not in done, "ledger terminal event without unique start")
                if event == "unit_done":
                    if row.get("recovered_from_done") is not None:
                        require(row["recovered_from_done"] is True and isinstance(row.get("done_sha256"), str)
                                and HEX.fullmatch(row["done_sha256"]), "invalid recovered DONE receipt")
                    done[uid] = row
                else:
                    failed.add(key)
        else:
            require(row.get("phase") == phase and row.get("scores_computed") is False, "ledger invocation end mismatch")
            active = set()
    for uid, entry in entries.items():
        require(uid in done and done[uid].get("attempt") == entry["attempt"]
                and done[uid].get("fit_seconds") == entry["fit_seconds"], "DONE-selected attempt missing/different in runtime ledger")
        if done[uid].get("recovered_from_done"):
            require(done[uid]["done_sha256"] == entry["done_sha256"], "recovered ledger DONE hash mismatch")
    return {"ledger_sha256": byte_hash(ledger_path), "ledger_event_count": len(events), "ledger_failed_attempts": len(failed),
            "output_identity_sha256": byte_hash(phase_root / "identity.json"),
            "executed_plan_snapshot_sha256": byte_hash(phase_root / "plan_snapshot.json"),
            "independent_plan_gate_receipt_sha256": byte_hash(phase_root / "verification.json"),
            "ledger_provenance_limit": "append-only runner event consistency checked and final bytes hashed; no external signature or hash chain"}


def verify_result(repo, plan_path, outroot, phase="formal", block="core", *, consolidate=False):
    """Full block integrity, not model quality or evidence of causal validity."""
    repo, plan_path = Path(repo).resolve(), Path(plan_path).resolve()
    require(repo == Path(__file__).resolve().parents[2], "result verifier import checkout differs from --repo")
    plan = read_json(plan_path)
    require(phase in ("formal", "pilot") and block in ("core", "ft"), "invalid result phase/block")
    require(plan.get("schema") == "ser-speaker-coverage-1" and plan.get("program") == "SER26-SPEAKER-COVERAGE-1"
            and plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "result plan identity/hash mismatch")
    require(plan["design"].get("fixture") is False and plan["design"].get("runtime_ready") is True
            and plan["design"].get("sourcefreeze_complete") is True, "result plan is not a complete real source freeze")
    if block == "ft":
        require(plan["design"].get("ft_runtime_ready") is True, "FT runtime not frozen")
    sources = plan.get("source_sha256", {})
    require(BASE_SOURCES | {p.relative_to(repo).as_posix() for p in (repo / "v3/speaker_coverage").glob("*.py")} <= sources.keys(), "result source set incomplete")
    for name, sha in sources.items():
        pinned(local(repo, name), sha, "result source " + name)
    manifest = local(repo, plan["input"]["manifest_path"])
    pinned(manifest, plan["input"]["manifest_sha256"], "result manifest")
    manifest_rows = _csv(manifest)
    rep, _ = representatives(manifest_rows, complete=True)
    metadata = {r["relative_path"]: r for r in manifest_rows}
    units = plan["units"]
    require(len(units) == len({u["unit_id"] for u in units}) == 720 and Counter(u["block"] for u in units) == {"core": 540, "ft": 180}
            and Counter(u["model"] for u in units) == {"cnn": 270, "ridge_wavlm": 270, "wavlm_ft": 180}, "result plan factorial counts mismatch")
    expected_grid = set(product(("ridge_wavlm", "cnn"), range(5), range(6), range(3), POLICIES, (0,)))
    expected_grid |= set(product(("wavlm_ft",), range(5), range(6), (0,), POLICIES, (0, 1)))
    observed_grid = set()
    for unit in units:
        require(unit["unit_id"] == digest({k: v for k, v in unit.items() if k != "unit_id"}), "result unit hash mismatch")
        coordinate = tuple(unit[k] for k in ("model", "fold", "rotation", "draw", "policy", "seed_index"))
        require(coordinate in expected_grid and coordinate not in observed_grid and unit["config"] == CONFIGS[unit["model"]], "result unit grid/config mismatch")
        observed_grid.add(coordinate)
        require(all(len(unit[role]) == len(set(unit[role])) and set(unit[role]) <= rep.keys() for role in ("fit", "val", "test"))
                and len(unit["fit"]) == 576 and unit["B"] == 576 and unit["S"] == 24, "result unit sample population mismatch")
    if phase == "pilot":
        units = [u for u in units if u["fold"] == u["rotation"] == 0 and (u["block"] == "core" or
                 (u["draw"] == 0 and u["policy"] == "U" and u["seed_index"] == 0))]
    selected = [u for u in units if u["block"] == block]
    expected = {("formal", "core"): 540, ("formal", "ft"): 180, ("pilot", "core"): 18, ("pilot", "ft"): 1}[phase, block]
    require(len(selected) == expected, "selected block count mismatch")
    phase_root = Path(outroot).resolve() / phase
    unit_dirs = list((phase_root / "units").iterdir())
    require(all(p.is_dir() and p.name in {u["unit_id"] for u in units} for p in unit_dirs), "unexpected result unit directory")
    entries = {u["unit_id"]: verify_result_unit(u, local(phase_root / "units", u["unit_id"]), plan, phase, metadata) for u in selected}
    phase_receipts = verify_phase_receipts(phase_root, plan, phase, units, entries)
    done_shas = {k: v["done_sha256"] for k, v in entries.items()}
    marker_path = phase_root / f"completion_{block}.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        require(marker == {"schema": "ser-speaker-coverage-block-complete-1", "plan_sha256": plan["plan_sha256"], "phase": phase,
                           "block": block, "count": expected, "done_sha256": done_shas, "scores_computed": False,
                           "independent_verification_required": True}, "block completion marker mismatch")
    else:
        require(consolidate is True, "missing block completion marker; use explicit consolidate mode for fully copied shards")
    report = {"schema": "ser-speaker-coverage-independent-result-seal-1", "pass": True, "block_complete": True,
              "plan_sha256": plan["plan_sha256"], "phase": phase, "block": block, "count": expected, "done_sha256": done_shas,
              "runner_completion_marker": "verified" if marker_path.exists() else "absent; complete shards independently consolidated",
              "counting_unit": "one DONE-selected successful attempt per planned unit; failed/partial attempts are not counted",
              "ignored_uncommitted_attempts": sum(v["ignored_uncommitted_attempts"] for v in entries.values()),
              "summed_successful_fit_seconds": sum(v["fit_seconds"] for v in entries.values()),
              "summed_successful_unit_wall_seconds": sum(v["wall_seconds"] for v in entries.values()),
              "timing_scope": "sums of successful unit timings; not billed pod elapsed time, GPU utilization, or total cost; failed attempts excluded",
              "artifact_bytes_checked": sum(v["artifact_bytes"] for v in entries.values()), "prediction_arrays_read_for_integrity": True,
              "scientific_scores_computed": False, "scientific_validity_established": False,
              "checkpoint_replay_scope": "independent CPU content/identity/shape checks plus pinned runner reload receipt; no independent inference replay here"}
    report.update(phase_receipts)
    report["seal_sha256"] = digest(report)
    return report


def verify_results(repo, plan_path, outroot, *, phase="formal", block="all", consolidate=False):
    if block != "all":
        return verify_result(repo, plan_path, outroot, phase, block, consolidate=consolidate)
    audits = {b: verify_result(repo, plan_path, outroot, phase, b, consolidate=consolidate) for b in ("core", "ft")}
    report = {"schema": "ser-speaker-coverage-all-result-seal-1", "pass": True, "all_blocks_complete": True,
              "plan_sha256": audits["core"]["plan_sha256"], "phase": phase, "count": sum(a["count"] for a in audits.values()),
              "blocks": audits, "scientific_scores_computed": False, "scientific_validity_established": False}
    report["seal_sha256"] = digest(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "features", "asv", "wavlm-model", "audio-root"):
        parser.add_argument("--" + name, type=Path, required=name in ("repo", "plan"))
    parser.add_argument("--outroot", type=Path, help="result mode: runner output base, containing pilot/formal")
    parser.add_argument("--phase", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--block", choices=("core", "ft", "all"), default="core")
    parser.add_argument("--consolidate", action="store_true", help="allow absent runner block marker after independently verifying every copied unit")
    parser.add_argument("--out", type=Path, help="optional machine-readable gate receipt")
    args = parser.parse_args(argv)
    try:
        if args.outroot:
            report = verify_results(args.repo, args.plan, args.outroot, phase=args.phase, block=args.block, consolidate=args.consolidate)
        else:
            require(all(getattr(args, name) is not None for name in ("features", "asv", "wavlm_model", "audio_root")), "plan gate requires features, ASV, WavLM and audio-root")
            report = verify_plan(args.repo, args.plan, args.features, args.asv, args.wavlm_model, args.audio_root)
    except Exception as error:
        report = {"pass": False, "formal_allowed": False, "error": str(error),
                  "scientific_outcomes_read": False, "scientific_validity_established": False}
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

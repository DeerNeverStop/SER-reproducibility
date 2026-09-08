"""Independent, read-only verification of Study II plans; never reads outcomes.

This module does not import the plan builder, runner, legacy split code, or a
scientific scorer.  The CLI accepts only the complete real 1,440-unit design.
Small in-memory fixtures may call verify_document(..., complete=False) in tests;
that mode is explicitly labelled and is not exposed by the CLI.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


class VerificationError(ValueError):
    pass


CLASSES = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
LABELS = set(range(6))
MODELS = {"ridge_wavlm", "cnn"}
SCENARIOS = {"prompt_seen", "prompt_new"}
HEX = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"(\d{4})_([A-Z]{3})_(ANG|DIS|FEA|HAP|NEU|SAD)_([A-Z]{1,2})\.wav\Z")
UNIT_FIELDS = {
    "model", "fold", "rotation", "draw", "scenario", "B", "S", "P_global",
    "P_per_speaker", "R", "train_seed", "fit", "val", "test", "config", "unit_id",
}
CONFIGS = {
    "ridge_wavlm": {"model": "ridge_a1_wavlm_base_plus", "feature_state": 12,
                    "alpha": 1.0, "class_weight": "balanced", "solver": "lsqr", "tol": 0.0001},
    "cnn": {"model": "cnn", "batch_size": 32, "epochs": 100, "patience": 15,
            "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1},
}


def require(value, message):
    if not value:
        raise VerificationError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value, omit=None):
    return hashlib.sha256(canonical({k: v for k, v in value.items() if k != omit})).hexdigest()


def byte_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=_object,
                         parse_constant=lambda x: (_ for _ in ()).throw(VerificationError(f"nonfinite JSON: {x}")))


def _relative(root, value):
    require(isinstance(value, str) and value, "missing relative path")
    p = PurePosixPath(value.replace("\\", "/"))
    require(not p.is_absolute() and ".." not in p.parts and ":" not in value, f"unsafe relative path: {value}")
    resolved = Path(root).resolve().joinpath(*p.parts).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), f"path escapes supplied root: {value}")
    return resolved


def _hash_field(value, name):
    require(isinstance(value, str) and HEX.fullmatch(value), f"invalid SHA256: {name}")


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        require(names and len(names) == len(set(names)), "missing/duplicate CSV columns")
        return list(reader)


def representatives(rows, *, complete=True):
    """Independently replay H1/H2/H3, then MD > XX > unavailable."""
    require(rows, "empty real manifest")
    paths, normalized, sha_groups = set(), {}, collections.defaultdict(list)
    for row in rows:
        path = row.get("relative_path")
        require(isinstance(path, str) and path not in paths, "missing/duplicate manifest path")
        require(not PurePosixPath(path.replace("\\", "/")).is_absolute() and ".." not in path.replace("\\", "/").split("/"), "unsafe audio path")
        match = NAME.fullmatch(PurePosixPath(path.replace("\\", "/")).name)
        require(match is not None, f"invalid CREMA-D filename: {path}")
        speaker, prompt, emotion, intensity = match.groups()
        require(str(row.get("speaker")) == speaker and row.get("sentence") == prompt,
                f"manifest/filename identity mismatch: {path}")
        try:
            label = int(row["label_index"])
        except (KeyError, TypeError, ValueError):
            raise VerificationError(f"invalid label index: {path}")
        require(label == CLASSES[emotion] and row.get("corpus") == "cremad", f"manifest label/corpus mismatch: {path}")
        require(row.get("intensity") == intensity, f"manifest intensity mismatch: {path}")
        _hash_field(row.get("sha256"), "audio sha256")
        item = {"path": path, "speaker": speaker, "prompt": prompt, "label": label,
                "intensity": intensity, "sha256": row["sha256"]}
        normalized[path] = item
        sha_groups[item["sha256"]].append(item)
        paths.add(path)
    if complete:
        require(len(rows) == 7442, "complete core requires the 7,442-row raw CREMA-D manifest")
        require(len({r["speaker"] for r in normalized.values()}) == 91, "raw manifest must contain 91 speakers")
        require(len({r["prompt"] for r in normalized.values()}) == 12, "raw manifest must contain 12 prompts")
    clean = []
    for group in sha_groups.values():
        if len({r["label"] for r in group}) > 1:
            continue
        winner = min(group, key=lambda r: r["path"])
        if PurePosixPath(winner["path"]).name != "1076_MTI_SAD_XX.wav":
            clean.append(winner)
    cells = collections.defaultdict(list)
    for row in clean:
        cells[(row["speaker"], row["prompt"], row["label"])].append(row)
    selected, unavailable = {}, []
    for cell, group in cells.items():
        allowed = [r for r in group if r["intensity"] in {"MD", "XX"}]
        if not allowed:
            unavailable.append(cell)
            continue
        row = min(allowed, key=lambda r: (0 if r["intensity"] == "MD" else 1, r["path"]))
        selected[row["path"]] = row
    if complete:
        require(len(clean) == 7435, "hygiene population differs from the 7,435-row contract")
    return selected, {"raw_rows": len(rows), "hygienic_rows": len(clean),
                      "representative_rows": len(selected), "unavailable_cells": [list(c) for c in sorted(unavailable)],
                      "representative_intensity_counts": dict(collections.Counter(r["intensity"] for r in selected.values()))}


def _balanced(speakers, sex, expected_each, message):
    counts = collections.Counter(sex.get(s) for s in speakers)
    require(counts == {"Female": expected_each, "Male": expected_each}, message)


def _group_unit(unit, rep, sex):
    require(set(unit) == UNIT_FIELDS, "unit fields differ from the execution contract")
    uid = unit["unit_id"]
    _hash_field(uid, "unit_id")
    require(uid == digest(unit, "unit_id"), "unit identity hash mismatch")
    require(unit["model"] in MODELS and unit["scenario"] in SCENARIOS, "unrecognized model/scenario")
    for key, allowed in {"B": {288, 576}, "S": {12, 48}, "fold": set(range(5)),
                         "draw": set(range(3)), "rotation": set(range(6))}.items():
        require(type(unit[key]) is int and unit[key] in allowed, f"invalid unit {key}")
    B, S = unit["B"], unit["S"]
    require(unit["P_global"] == 8 and unit["R"] == 1 and unit["P_per_speaker"] == B // (6 * S), "invalid panel constants")
    require(type(unit["train_seed"]) is int and 0 <= unit["train_seed"] < 2**32, "invalid training seed")
    require(unit["config"] == CONFIGS[unit["model"]], "model configuration differs from fixed core")
    role_rows, role_speakers, role_prompts = {}, {}, {}
    all_paths = []
    for role in ("fit", "val", "test"):
        panel = unit[role]
        require(isinstance(panel, list) and panel and all(isinstance(p, str) for p in panel), f"empty/invalid {role}")
        require(panel == sorted(set(panel)), f"{role} paths are not unique and sorted")
        require(all(p in rep for p in panel), f"{role} contains nonrepresentative, unhygienic or unknown recording")
        group = [rep[p] for p in panel]
        role_rows[role] = group
        role_speakers[role] = {r["speaker"] for r in group}
        role_prompts[role] = {r["prompt"] for r in group}
        require({r["label"] for r in group} == LABELS, f"{role} has missing global classes")
        all_paths.extend(panel)
    require(len(all_paths) == len(set(all_paths)), "recording crosses role boundaries")
    require(len({rep[p]["sha256"] for p in all_paths}) == len(all_paths), "byte duplicate within/across role boundaries")
    for a, b in itertools.combinations(("fit", "val", "test"), 2):
        require(not (role_speakers[a] & role_speakers[b]), f"speaker overlap between {a} and {b}")
    require(len(unit["fit"]) == B and len(role_speakers["fit"]) == S, "fit budget/speaker count mismatch")
    require(len(role_prompts["fit"]) == 8 and len(role_prompts["val"]) == 6 and len(role_prompts["test"]) == 2,
            "fit/val/test prompt count mismatch")
    require(len(role_speakers["val"]) == 8, "early-stop speaker count mismatch")
    require(role_prompts["val"] <= role_prompts["fit"] and not (role_prompts["val"] & role_prompts["test"]), "early-stop prompt boundary mismatch")
    _balanced(role_speakers["fit"], sex, S // 2, "training sex strata are unbalanced")
    _balanced(role_speakers["val"], sex, 4, "early-stop sex strata are unbalanced")
    if unit["scenario"] == "prompt_new":
        require(not (role_prompts["fit"] & role_prompts["test"]), "unseen test prompt leaked into fit")
    else:
        require(role_prompts["test"] <= role_prompts["fit"], "seen test prompts absent from fit")
    incidence = collections.defaultdict(set)
    by_cell = collections.defaultdict(set)
    by_speaker = collections.defaultdict(set)
    for r in role_rows["fit"]:
        by_cell[(r["speaker"], r["prompt"])].add(r["label"])
        incidence[r["prompt"]].add(r["speaker"])
        by_speaker[r["speaker"]].add(r["prompt"])
    require(all(labels == LABELS for labels in by_cell.values()), "fit speaker-prompt block lacks classes")
    require(all(len(prompts) == unit["P_per_speaker"] for prompts in by_speaker.values()), "per-speaker prompt degree mismatch")
    require(collections.Counter(r["label"] for r in role_rows["fit"]) == {c: B // 6 for c in LABELS}, "class quota mismatch")
    for speakers in incidence.values():
        require(len(speakers) == B // 48, "per-prompt speaker exposure mismatch")
        _balanced(speakers, sex, B // 96, "per-prompt sex strata are unbalanced")
    expected_val = {p for p, r in rep.items() if r["speaker"] in role_speakers["val"] and r["prompt"] in role_prompts["val"]}
    expected_test = {p for p, r in rep.items() if r["speaker"] in role_speakers["test"] and r["prompt"] in role_prompts["test"]}
    require(set(unit["val"]) == expected_val and len(unit["val"]) <= 288, "early-stop exact available six-prompt cover mismatch")
    require(all({r["label"] for r in role_rows["val"] if r["speaker"] == s} == LABELS for s in role_speakers["val"]),
            "early-stop speaker lacks a fixed target class")
    require(set(unit["test"]) == expected_test, "test is not the exact representative cross-product cover")
    return {"unit": unit, "speakers": role_speakers, "prompts": role_prompts, "incidence": incidence}


def verify_document(plan, manifest_rows, demographics, *, complete=True):
    """Pure metadata verification; no file loading and no scientific outcomes."""
    require(plan.get("schema") == "ser-study2-core-1" and plan.get("program") == "SER26-STUDY2-CORE-1", "wrong core schema/program")
    _hash_field(plan.get("plan_sha256"), "plan_sha256")
    require(plan["plan_sha256"] == digest(plan, "plan_sha256"), "plan hash mismatch")
    units = plan.get("units")
    require(isinstance(units, list) and units, "missing units")
    if complete:
        require(len(units) == 1440, "complete core requires exactly 1440 units")
        design = plan.get("design", {})
        for key, value in {"speaker_folds": 5, "prompt_rotations": 6, "draws": 3,
                           "budgets": [288, 576], "speakers": [12, 48], "global_prompts": 8,
                           "stop_speakers": 8, "no_selection": True,
                           "scientific_scores_hidden_until_complete": True, "sesoi_pp": 3.0}.items():
            require(design.get(key) == value, f"design constant mismatch: {key}")
        require(design.get("status") == "prospectively_specified_estimation_not_new_confirmatory_evidence", "estimation status mismatch")
    rep, rep_report = representatives(manifest_rows, complete=complete)
    population = {r["speaker"] for r in rep.values()}
    prompts = {r["prompt"] for r in rep.values()}
    require(population <= set(demographics) and all(demographics[s] in {"Female", "Male"} for s in population), "missing/invalid demographics")
    metas, ids, coordinates = [], set(), set()
    for unit in units:
        meta = _group_unit(unit, rep, demographics)
        require(unit["unit_id"] not in ids, "duplicate unit id")
        ids.add(unit["unit_id"])
        coordinate = tuple(unit[k] for k in ("draw", "fold", "rotation", "B", "S", "scenario", "model"))
        require(coordinate not in coordinates, "duplicate design coordinate")
        coordinates.add(coordinate)
        metas.append(meta)
    if complete:
        expected = set(itertools.product(range(3), range(5), range(6), (288, 576), (12, 48), SCENARIOS, MODELS))
        require(coordinates == expected, "missing/extra factorial coordinates")
    base_groups = collections.defaultdict(list)
    scenario_pairs = collections.defaultdict(dict)
    model_pairs = collections.defaultdict(list)
    for meta in metas:
        u = meta["unit"]
        base = (u["draw"], u["fold"], u["rotation"])
        base_groups[base].append(meta)
        scenario_pairs[base + (u["B"], u["S"], u["model"])][u["scenario"]] = meta
        model_pairs[base + (u["B"], u["S"], u["scenario"])].append(meta)
    for group in base_groups.values():
        baseline = group[0]["unit"]
        for meta in group[1:]:
            require(meta["unit"]["val"] == baseline["val"], "paired policies/scenarios do not share early-stop recordings")
            require(meta["unit"]["test"] == baseline["test"], "paired policies/scenarios do not share test recordings")
    for group in model_pairs.values():
        require(len(group) == 2 and {m["unit"]["model"] for m in group} == MODELS, "missing paired model")
        a, b = (m["unit"] for m in group)
        require(a["fit"] == b["fit"] and a["train_seed"] == b["train_seed"], "model comparison changes fit data/seed")
    for pair in scenario_pairs.values():
        require(set(pair) == SCENARIOS, "missing paired scenario")
        seen, new = pair["prompt_seen"], pair["prompt_new"]
        require(seen["speakers"]["fit"] == new["speakers"]["fit"], "scenario pairing changes training speakers")
        require(seen["unit"]["train_seed"] == new["unit"]["train_seed"], "scenario pairing changes training seed")
        shared = seen["prompts"]["fit"] & new["prompts"]["fit"]
        removed = seen["prompts"]["fit"] - new["prompts"]["fit"]
        added = new["prompts"]["fit"] - seen["prompts"]["fit"]
        require(len(shared) == 6 and removed == seen["prompts"]["test"] and len(added) == 2,
                "scenario contrast is not a common-six/two-prompt substitution")
        require(shared == seen["prompts"]["val"], "stop prompts are not exactly the common six")
        require(all(seen["incidence"][p] == new["incidence"][p] for p in shared), "common-prompt adjacency changed between scenarios")
        matched = any(all(seen["incidence"][a] == new["incidence"][b] for a, b in zip(sorted(removed), order))
                      for order in itertools.permutations(sorted(added)))
        require(matched, "changed-prompt speaker adjacency is not a paired slot substitution")
    if complete:
        draw_fold = collections.defaultdict(list)
        for (draw, fold, rotation), group in base_groups.items():
            draw_fold[(draw, fold)].append((rotation, group[0]))
        fold_maps = []
        missing_test = []
        by_cell = {(r["speaker"], r["prompt"], r["label"]): r for r in rep.values()}
        for draw in range(3):
            owners = {}
            sex_fold_counts = collections.defaultdict(list)
            for fold in range(5):
                rotations = sorted(draw_fold[(draw, fold)], key=lambda x: x[0])
                require([r for r, _ in rotations] == list(range(6)), "missing prompt rotation")
                test_people = rotations[0][1]["speakers"]["test"]
                stop_people = rotations[0][1]["speakers"]["val"]
                prompt_counts = collections.Counter()
                for _, meta in rotations:
                    require(meta["speakers"]["test"] == test_people and meta["speakers"]["val"] == stop_people,
                            "test/stop speakers change across prompt rotations")
                    prompt_counts.update(meta["prompts"]["test"])
                require(prompt_counts == {p: 1 for p in prompts} and len(prompts) == 12, "six rotations do not cover all twelve prompts exactly once")
                for s in test_people:
                    require(s not in owners, "test speaker appears in more than one outer fold")
                    owners[s] = fold
                    for label in LABELS:
                        covered = [p for p in prompts if (s, p, label) in by_cell]
                        require(covered, "test speaker lacks a fixed class after all rotations")
                        if draw == 0 and len(covered) != 12:
                            missing_test.append({"speaker": s, "label": label, "available_prompts": len(covered)})
                for gender in ("Female", "Male"):
                    sex_fold_counts[gender].append(sum(demographics[s] == gender for s in test_people))
            require(set(owners) == population and len(owners) == 91, "every draw must test all 91 speakers exactly once")
            require(all(max(v) - min(v) <= 1 for v in sex_fold_counts.values()), "outer folds are not stratified by demographic sex")
            fold_maps.append(tuple(sorted(owners.items())))
        require(len(set(fold_maps)) == 3, "draws repeat an identical speaker partition")
    else:
        missing_test = []
    return {"pass": True, "schema": "ser-study2-independent-plan-verification-1",
            "scope": "complete_real_core_metadata" if complete else "test_fixture_only",
            "n_units": len(units), "units_by_model": dict(collections.Counter(u["model"] for u in units)),
            "n_speakers": len(population), "n_prompts": len(prompts),
            "early_stop_rows_range": [min(len(u["val"]) for u in units), max(len(u["val"]) for u in units)],
            "representatives": rep_report, "test_class_prompt_shortfalls": missing_test,
            "scientific_outcomes_read": False}


def verify_plan_file(repo, plan_path, *, features_dir=None, metadata_only=False):
    repo = Path(repo).resolve()
    plan = read_json(plan_path)
    require(isinstance(plan, dict), "plan is not an object")
    inp = plan.get("input", {})
    manifest = _relative(repo, inp.get("manifest_path"))
    demographics_path = _relative(repo, inp.get("demographics_path"))
    for path, expected, name in [(manifest, inp.get("manifest_sha256"), "manifest"),
                                  (demographics_path, inp.get("demographics_sha256"), "demographics")]:
        _hash_field(expected, name)
        require(path.is_file() and byte_hash(path) == expected, f"{name} byte hash mismatch")
    sources = plan.get("source_sha256")
    require(isinstance(sources, dict) and sources, "missing pinned sources")
    for relative, expected in sources.items():
        _hash_field(expected, relative)
        source = _relative(repo, relative)
        require(source.is_file() and byte_hash(source) == expected, f"pinned source byte hash mismatch: {relative}")
    feature_files, feature_hashes = inp.get("feature_files", {}), inp.get("feature_sha256", {})
    require(set(feature_files) == set(feature_hashes) == {"logmel", "wavlm_base_plus"}, "feature identity mapping mismatch")
    for kind, basename in feature_files.items():
        require(isinstance(basename, str) and PurePosixPath(basename).name == basename and "\\" not in basename,
                "feature cache must be an explicit basename")
        _hash_field(feature_hashes[kind], kind)
        if not metadata_only:
            require(features_dir is not None, "full input hash verification requires --features; otherwise use --metadata-only")
            path = _relative(features_dir, basename)
            require(path.is_file() and byte_hash(path) == feature_hashes[kind], f"feature cache byte hash mismatch: {kind}")
    demographics = {}
    for row in _csv(demographics_path):
        speaker = row.get("ActorID")
        require(speaker and speaker not in demographics and row.get("Sex") in {"Female", "Male"}, "invalid/duplicate demographic row")
        demographics[speaker] = row["Sex"]
    result = verify_document(plan, _csv(manifest), demographics, complete=True)
    result.update({"plan_sha256": plan["plan_sha256"], "manifest_sha256": inp["manifest_sha256"],
                   "source_files_checked": len(sources), "feature_bytes_checked": not metadata_only,
                   "runtime_ready_claimed": bool(plan.get("design", {}).get("runtime_complete", False)),
                   "training_authorization": "not_granted_by_metadata_verification"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--metadata-only", action="store_true", help="check plan/manifest/source identities; skip feature payload hashes; never read run results")
    args = parser.parse_args(argv)
    try:
        result = verify_plan_file(args.repo, args.plan, features_dir=args.features, metadata_only=args.metadata_only)
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"pass": False, "error": str(exc), "scientific_outcomes_read": False}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

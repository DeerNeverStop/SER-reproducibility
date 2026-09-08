#!/usr/bin/env python3
"""Independent numerical replay of an already completed formal 720-unit study.

This external audit tool imports the integrity verifier, never score.py. It
does not train models, use a GPU, access the network, or accept pilot results.
Its second implementation pools explicit truth/guess lists, computes each
class's Boolean recall, and then averages people. Bootstrap draws use the
fixed PCG64 stream, but are evaluated as people-count weights rather than
indexing the primary scorer's matrix. Quantiles use an explicit linear rule.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
from itertools import product
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import csv

import numpy as np

MODELS = ("cnn", "ridge_wavlm", "wavlm_ft")
POLICIES = ("U", "R", "C")
CHECKPOINTS = ("best", "last")
LABELS = {"angry": 0, "disgust": 1, "fearful": 2, "happy": 3, "neutral": 4, "sad": 5}
CODES = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
SEED, BOOTSTRAPS = 20260906, 10000
TOLERANCE = 1e-9
TABLE_KEYS = {
    "policy_means.csv": ("model", "policy", "checkpoint"),
    "contrasts.csv": ("model", "checkpoint", "contrast"),
    "fold_contrasts.csv": ("model", "checkpoint", "contrast", "fold"),
    "per_speaker.csv": ("model", "policy", "checkpoint", "speaker"),
    "per_repeat_speaker.csv": ("model", "policy", "checkpoint", "repeat", "speaker"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        before = Path(path).stat()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
        after = Path(path).stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed during hashing")
    return result.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "invalid CSV header")
        rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows), "malformed CSV row")
    return rows


def local(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative, "invalid relative artifact path")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == relative, "escaping artifact path")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    require(result.is_relative_to(root), "artifact escapes its supplied root")
    return result


def pin(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "invalid SHA256 pin")
    require(file_hash(path) == expected, "file differs from pinned SHA256: " + Path(path).name)


def repeats(model):
    return 2 if model == "wavlm_ft" else 3


def load_plan(path):
    plan = read_json(path)
    require(plan.get("schema") == "ser-speaker-coverage-1" and plan.get("program") == "SER26-SPEAKER-COVERAGE-1", "wrong plan schema/program")
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "plan content hash mismatch")
    units = plan.get("units", [])
    require(len(units) == len({u["unit_id"] for u in units}) == 720, "numerical replay requires all 720 formal units")
    expected = set(product(("cnn", "ridge_wavlm"), range(5), range(6), range(3), POLICIES, (0,)))
    expected |= set(product(("wavlm_ft",), range(5), range(6), (0,), POLICIES, (0, 1)))
    actual = set()
    for unit in units:
        require(unit["unit_id"] == digest({k: v for k, v in unit.items() if k != "unit_id"}), "unit content hash mismatch")
        coordinate = tuple(unit[k] for k in ("model", "fold", "rotation", "draw", "policy", "seed_index"))
        require(coordinate in expected and coordinate not in actual, "formal factorial grid incomplete/duplicated")
        actual.add(coordinate)
    require(Counter(u["block"] for u in units) == {"core": 540, "ft": 180}, "formal block counts differ")
    return plan


def load_raw_manifest(repo, plan):
    manifest = local(repo, plan["input"]["manifest_path"])
    pin(manifest, plan["input"]["manifest_sha256"])
    rows = read_csv(manifest)
    require(len(rows) == 7442 and len({r["relative_path"] for r in rows}) == 7442, "unexpected raw manifest population")
    duplicates = defaultdict(list)
    for row in rows:
        path = row["relative_path"]
        local(repo, path)
        parts = Path(path).stem.split("_")
        require(len(parts) == 4 and parts[0] == row["speaker"] and parts[1] == row["sentence"]
                and parts[2] in CODES and LABELS.get(row["label"]) == CODES[parts[2]] == int(row["label_index"]),
                "filename/raw manifest class or speaker disagreement")
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "invalid raw audio pin")
        duplicates[row["sha256"]].append(row)
    excluded = {"1076_MTI_SAD_XX.wav"}
    for group in duplicates.values():
        if len({r["label"] for r in group}) > 1:
            excluded.update(r["relative_path"] for r in group)
        else:
            excluded.update(sorted(r["relative_path"] for r in group)[1:])
    clean = {r["relative_path"]: r for r in rows if r["relative_path"] not in excluded}
    require(len(clean) == 7435, "hygiene exclusions differ from the frozen corpus")
    speakers = sorted({r["speaker"] for r in clean.values()})
    require(len(speakers) == 91, "all 91 formal people are required")
    return clean, speakers


def inspect_closure(plan, outroot):
    """No prediction values are read until every formal DONE exists."""
    base = Path(outroot) / "formal" / "units"
    require(base.is_dir(), "formal result directory missing; pilot is not accepted")
    ids = {u["unit_id"] for u in plan["units"]}
    require(all(p.is_dir() and p.name in ids for p in base.iterdir()), "unexpected formal unit directory")
    closure, predictions = {}, {}
    for unit in plan["units"]:
        directory = local(base, unit["unit_id"])
        done_path = directory / "DONE"
        done = read_json(done_path)
        require(done.get("schema") == "ser-speaker-coverage-done-1" and done.get("unit_id") == unit["unit_id"]
                and done.get("plan_sha256") == plan["plan_sha256"] and done.get("phase") == "formal"
                and done.get("model") == unit["model"] and done.get("block") == unit["block"], "formal DONE identity mismatch")
        attempt = done.get("attempt")
        require(type(attempt) is int and attempt > 0, "invalid committed attempt")
        prefix = f"attempts/{attempt:04d}/"
        checkpoint = "checkpoint.npz" if unit["model"] == "ridge_wavlm" else "checkpoint.pt"
        require(set(done.get("artifacts", {})) == {prefix + n for n in ("predictions.npz", "receipt.json", "history.json", checkpoint)},
                "DONE does not commit exactly the required artifacts")
        name = prefix + "predictions.npz"
        path = local(directory, name)
        pin(path, done["artifacts"][name])
        predictions[unit["unit_id"]] = (path, done["artifacts"][name])
        closure[unit["unit_id"]] = file_hash(done_path)
    return closure, predictions


def boolean_recalls(observations):
    """Explicit observed labels, six fixed denominators; no confusion helper."""
    recalls, support = [], []
    require(observations and all(type(y) is int and type(p) is int and y in range(6) and p in range(6)
                                 for y, p in observations), "invalid truth/guess list")
    for label in range(6):
        decisions = [guess == label for truth, guess in observations if truth == label]
        require(decisions, "an acted class is absent; fixed-six UAR cannot drop it")
        support.append(len(decisions))
        recalls.append(sum(decisions) / len(decisions))
    return math.fsum(recalls) * (100 / 6), min(support)


def read_guesses(path, unit, metadata):
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == {"paths", "labels", "logits", "pred", "proba", "last_logits", "last_pred", "last_proba"},
                "prediction key set differs")
        paths = arrays["paths"].tolist()
        require(arrays["paths"].dtype.kind == "U" and paths == unit["test"] and len(paths) == len(set(paths))
                and set(paths) <= metadata.keys(), "actual query paths differ from plan/manifest")
        truth = [int(metadata[p]["label_index"]) for p in paths]
        require(arrays["labels"].dtype == np.int64 and arrays["labels"].tolist() == truth, "sealed labels differ from raw manifest")
        guesses = {}
        for checkpoint, field in (("best", "logits"), ("last", "last_logits")):
            scores = arrays[field]
            require(scores.dtype == np.float64 and scores.shape == (len(paths), 6) and np.isfinite(scores).all(), "invalid sealed logits")
            # Python max chooses the first class on a tie, independently of the
            # primary scorer's ndarray.argmax implementation.
            guess = [max(range(6), key=lambda k: float(row[k])) for row in scores]
            pred_field = "pred" if checkpoint == "best" else "last_pred"
            require(arrays[pred_field].dtype == np.int64 and arrays[pred_field].tolist() == guess, "saved predictions differ from independent argmax")
            guesses[checkpoint] = guess
    return paths, truth, guesses


def collect_observations(plan, prediction_files, metadata, speakers):
    buckets, seen_paths, rotations, folds = defaultdict(list), defaultdict(set), defaultdict(set), defaultdict(set)
    for unit in plan["units"]:
        path, sha = prediction_files[unit["unit_id"]]
        pin(path, sha)
        paths, truth, guesses = read_guesses(path, unit, metadata)
        repetition = unit["seed_index"] if unit["model"] == "wavlm_ft" else unit["draw"]
        groups = defaultdict(list)
        for index, name in enumerate(paths):
            groups[metadata[name]["speaker"]].append(index)
        for person, indexes in groups.items():
            names = {paths[i] for i in indexes}
            for checkpoint in CHECKPOINTS:
                key = unit["model"], unit["policy"], checkpoint, repetition, person
                require(unit["rotation"] not in rotations[key] and not names.intersection(seen_paths[key]), "repeated query recording/rotation")
                buckets[key].extend((truth[i], guesses[checkpoint][i]) for i in indexes)
                seen_paths[key].update(names)
                rotations[key].add(unit["rotation"])
                folds[key].add(unit["fold"])
    expected = {(m, p, c, r, s) for m in MODELS for p in POLICIES for c in CHECKPOINTS for r in range(repeats(m)) for s in speakers}
    require(set(buckets) == expected, "incomplete per-person repetition grid")
    values, detail, person_fold = {}, [], {}
    for key in sorted(buckets):
        model, policy, checkpoint, repetition, person = key
        require(rotations[key] == set(range(6)) and len(folds[key]) == 1, "six actual rotations/fixed outer fold not covered")
        fold = next(iter(folds[key]))
        require(person not in person_fold or person_fold[person] == fold, "person's outer fold changed")
        person_fold[person] = fold
        value, minimum = boolean_recalls(buckets[key])
        values[key] = value
        detail.append({"model": model, "policy": policy, "checkpoint": checkpoint, "repeat": repetition,
                       "repeat_type": "training_seed" if model == "wavlm_ft" else "panel_draw", "speaker": person,
                       "fold": fold, "uar_percent": value, "query_records": len(seen_paths[key]), "minimum_class_support": minimum})
    return values, detail, person_fold


def average(values):
    values = list(values)
    require(values and all(math.isfinite(v) for v in values), "empty/nonfinite average")
    return math.fsum(values) / len(values)


def linear_quantile(values, probability):
    ordered = sorted(float(x) for x in values)
    require(ordered and all(math.isfinite(x) for x in ordered) and 0 <= probability <= 1, "invalid quantile")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_people_weights(n, repetitions=BOOTSTRAPS):
    require(type(n) is int and n > 1, "invalid bootstrap person count")
    generator = np.random.Generator(np.random.PCG64(SEED))
    weights = np.empty((repetitions, n), dtype=np.int64)
    for row in range(repetitions):
        weights[row] = np.bincount(generator.integers(0, n, size=n), minlength=n)
    require((weights.sum(axis=1) == n).all(), "bootstrap weights lost people")
    return weights


def summary_from_person_values(speakers, values, person_fold, *, bootstrap_repetitions=BOOTSTRAPS):
    weights = bootstrap_people_weights(len(speakers), bootstrap_repetitions)
    people, policies, contrasts, by_fold = [], [], [], []
    means = {}
    for model, checkpoint, policy in product(MODELS, CHECKPOINTS, POLICIES):
        nrep = repeats(model)
        individual = [average(values[model, policy, checkpoint, r, person] for r in range(nrep)) for person in speakers]
        means[model, policy, checkpoint] = individual
        policies.append({"model": model, "policy": policy, "checkpoint": checkpoint, "n_speakers": len(speakers),
                         "n_repetitions": nrep, "mean_speaker_uar_percent": average(individual),
                         "q25_speaker_uar_percent_descriptive": linear_quantile(individual, .25),
                         "repeat_population_means_percent": [average(values[model, policy, checkpoint, r, s] for s in speakers) for r in range(nrep)]})
        people.extend({"model": model, "policy": policy, "checkpoint": checkpoint, "speaker": s, "fold": person_fold[s], "uar_percent": v}
                      for s, v in zip(speakers, individual))
    for model, checkpoint, policy in product(MODELS, CHECKPOINTS, ("C", "R")):
        # Average within-person repeat differences before bootstrap weighting.
        difference = [average(values[model, policy, checkpoint, r, s] - values[model, "U", checkpoint, r, s]
                              for r in range(repeats(model))) for s in speakers]
        boot = weights @ np.asarray(difference, dtype=np.float64) / len(speakers)
        contrasts.append({"model": model, "checkpoint": checkpoint, "contrast": policy + "-U",
                          "role": "primary" if (model, checkpoint, policy) == ("cnn", "best", "C") else "secondary_or_sensitivity",
                          "n_speakers": len(speakers), "difference_pp": average(difference),
                          "ci95_low_pp": linear_quantile(boot, .025), "ci95_high_pp": linear_quantile(boot, .975),
                          "repeat_differences_pp": [average(values[model, policy, checkpoint, r, s] - values[model, "U", checkpoint, r, s]
                                                             for s in speakers) for r in range(repeats(model))]})
        for fold in range(5):
            selected = [d for s, d in zip(speakers, difference) if person_fold[s] == fold]
            by_fold.append({"model": model, "checkpoint": checkpoint, "contrast": policy + "-U", "fold": fold,
                            "n_speakers": len(selected), "difference_pp_descriptive": average(selected)})
    return {"policy_means.csv": policies, "contrasts.csv": contrasts, "fold_contrasts.csv": by_fold, "per_speaker.csv": people}


class Comparator:
    def __init__(self):
        self.numeric_values_checked = 0
        self.maximum_absolute_error = 0.0

    def value(self, actual, expected, location):
        if isinstance(expected, bool) or expected is None:
            require(actual is expected, "disagreement at " + location)
        elif isinstance(expected, (int, float)):
            require(type(actual) in (int, float) and math.isfinite(actual), "invalid numeric value at " + location)
            error = abs(actual - expected)
            require(error <= (0 if isinstance(expected, int) else TOLERANCE), "numeric disagreement at " + location)
            self.numeric_values_checked += 1
            self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        elif isinstance(expected, str):
            require(actual == expected, "text disagreement at " + location)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), "list length disagreement at " + location)
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.value(a, e, f"{location}[{i}]")
        else:
            require(isinstance(actual, dict) and actual.keys() == expected.keys(), "field disagreement at " + location)
            for key, value in expected.items():
                self.value(actual[key], value, location + "." + key)

    def rows(self, actual_rows, expected_rows, keys, location, *, csv_input=False):
        require(len(actual_rows) == len(expected_rows), "row count disagreement at " + location)
        expected_map = {tuple(str(r[k]) for k in keys): r for r in expected_rows}
        require(len(expected_map) == len(expected_rows), "replay generated duplicate rows")
        seen = set()
        for row in actual_rows:
            identity = tuple(str(row[k]) for k in keys)
            require(identity in expected_map and identity not in seen, "unknown/duplicate row at " + location)
            seen.add(identity)
            expected = expected_map[identity]
            require(row.keys() == expected.keys(), "row columns disagree at " + location)
            if csv_input:
                converted = {}
                for key, value in expected.items():
                    if isinstance(value, str):
                        converted[key] = row[key]
                    elif isinstance(value, list):
                        converted[key] = parse_json(row[key])
                    elif isinstance(value, int):
                        converted[key] = int(row[key])
                    else:
                        converted[key] = float(row[key])
                row = converted
            self.value(row, expected, location + ":" + "/".join(identity))


def compare_score_artifacts(results_dir, report, tables, closure, plan, repo):
    require(report.get("schema") == "ser-speaker-coverage-analysis-1" and report.get("plan_sha256") == plan["plan_sha256"]
            and report.get("done_mapping_sha256") == digest(closure), "analysis does not bind the actual formal DONE closure")
    require(report.get("verified_formal_units") == 720 and report.get("excluded_pilot_units") == 19
            and report.get("bootstrap_seed") == SEED and report.get("bootstrap_repetitions") == BOOTSTRAPS,
            "analysis count/bootstrap protocol differs")
    require(report.get("all_comparisons_reported") is True and report.get("equivalence_claim") is False
            and report.get("causal_mediation_claim") is False, "analysis interpretation flags differ")
    pin(Path(repo) / "v3/speaker_coverage/score.py", report.get("source_sha256"))
    require(set(report.get("table_sha256", {})) == TABLE_KEYS.keys(), "analysis table inventory differs")
    for block, count in (("core", 540), ("ft", 180)):
        audit = report.get("result_audits", {}).get(block, {})
        subset = {u["unit_id"]: closure[u["unit_id"]] for u in plan["units"] if u["block"] == block}
        require(audit.get("pass") is True and audit.get("count") == count and audit.get("phase") == "formal"
                and audit.get("plan_sha256") == plan["plan_sha256"] and audit.get("done_sha256") == subset,
                "analysis block audit does not match actual closure")
    compare = Comparator()
    for name, keys in TABLE_KEYS.items():
        path = local(results_dir, name)
        pin(path, report["table_sha256"][name])
        compare.rows(read_csv(path), tables[name], keys, name, csv_input=True)
    compare.rows(report.get("policies", []), tables["policy_means.csv"], TABLE_KEYS["policy_means.csv"], "results.policies")
    compare.rows(report.get("contrasts", []), tables["contrasts.csv"], TABLE_KEYS["contrasts.csv"], "results.contrasts")
    primary = [r for r in tables["contrasts.csv"] if r["role"] == "primary"]
    require(len(primary) == 1 and len(tables["contrasts.csv"]) == 12, "all 12 predeclared comparisons required")
    compare.value(report.get("primary"), primary[0], "results.primary")
    return compare


def replay(repo, plan_path, outroot, results_dir):
    repo, plan_path, outroot, results_dir = (Path(p).resolve() for p in (repo, plan_path, outroot, results_dir))
    plan = load_plan(plan_path)
    metadata, speakers = load_raw_manifest(repo, plan)
    closure, predictions = inspect_closure(plan, outroot)
    report_path = results_dir / "results.json"
    score_report_sha = file_hash(report_path)
    report = read_json(report_path)
    require(report.get("plan_sha256") == plan["plan_sha256"] and report.get("done_mapping_sha256") == digest(closure),
            "main result report belongs to another plan or set of committed attempts")
    # Full checkpoint/history/ledger gate precedes all new logit-value reads.
    sys.path.insert(0, str(repo))
    verifier = importlib.import_module("v3.speaker_coverage.verify")
    require(Path(verifier.__file__).resolve() == repo / "v3/speaker_coverage/verify.py",
            "cached integrity verifier comes from another checkout")
    pin(verifier.__file__, plan["source_sha256"]["v3/speaker_coverage/verify.py"])
    gate = verifier.verify_results(repo, plan_path, outroot, phase="formal", block="all", consolidate=True)
    require(gate.get("pass") is True and gate.get("all_blocks_complete") is True and gate.get("count") == 720,
            "complete formal integrity verification did not pass")
    require(gate.get("plan_sha256") == plan["plan_sha256"], "integrity gate reports another plan")
    values, detail, folds = collect_observations(plan, predictions, metadata, speakers)
    tables = summary_from_person_values(speakers, values, folds)
    tables["per_repeat_speaker.csv"] = detail
    comparison = compare_score_artifacts(results_dir, report, tables, closure, plan, repo)
    require(file_hash(report_path) == score_report_sha, "main result report changed during replay")
    for unit in plan["units"]:
        pin(outroot / "formal/units" / unit["unit_id"] / "DONE", closure[unit["unit_id"]])
        pin(*predictions[unit["unit_id"]])
    for name, sha in report["table_sha256"].items():
        pin(results_dir / name, sha)
    audit = {"schema": "ser-coverage-independent-numerical-replay-1", "pass": True, "phase": "formal", "formal_units": 720,
             "people": 91, "excluded_pilot_units": 19, "plan_sha256": plan["plan_sha256"], "plan_file_sha256": file_hash(plan_path),
             "manifest_sha256": plan["input"]["manifest_sha256"], "done_mapping_sha256": digest(closure),
             "integrity_gate_seal_sha256": gate["seal_sha256"], "main_results_file_sha256": score_report_sha,
             "main_table_sha256": report["table_sha256"], "replay_source_sha256": file_hash(__file__),
             "score_source_sha256": report["source_sha256"], "bootstrap_seed": SEED, "bootstrap_repetitions": BOOTSTRAPS,
             "table_rows_checked": {name: len(rows) for name, rows in tables.items()}, "contrasts_checked": 12,
             "numeric_values_checked": comparison.numeric_values_checked, "absolute_tolerance": TOLERANCE,
             "maximum_absolute_error": comparison.maximum_absolute_error,
             "independent_aggregation": "actual query truth/guess lists pooled over six rotations; Boolean recall per fixed acted class; average repetitions within person",
             "independent_bootstrap": "PCG64 fixed people draws represented by multiplicity weights; linear quantiles explicitly interpolated",
             "primary_score_functions_imported": False, "new_model_fits": 0, "network_used": False,
             "scope": "numerical and artifact consistency only; no causal validity, significance threshold, or additional comparison is certified"}
    audit["audit_sha256"] = digest(audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "outroot", "results", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        audit = replay(args.repo, args.plan, args.outroot, args.results)
    except Exception as error:
        audit = {"schema": "ser-coverage-independent-numerical-replay-1", "pass": False, "phase": "formal",
                 "error_type": type(error).__name__, "error": str(error), "new_model_fits": 0,
                 "replay_source_sha256": file_hash(__file__)}
    # The requested audit must not replace any input or sealed unit evidence.
    target = args.out.resolve()
    require(not target.is_relative_to(args.outroot.resolve()) and not target.is_relative_to(args.repo.resolve())
            and not target.is_relative_to(args.results.resolve()) and target != args.plan.resolve(), "audit output must be outside all input directories")
    target.parent.mkdir(parents=True, exist_ok=True)
    audit["verified_at"] = datetime.now(timezone.utc).isoformat()
    # Rebind the final receipt, including its timestamp.
    audit.pop("audit_sha256", None)
    audit["audit_sha256"] = digest(audit)
    target.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pass": audit["pass"], "phase": "formal", "audit": str(target), "new_model_fits": 0}))
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

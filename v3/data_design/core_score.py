"""Score only the complete locked Study II; no model selection or partial scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sys
import os
import uuid

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.data_design import core_verify

EXPECTED_UNITS = 1440
PROGRAM = "SER26-STUDY2-CORE-1"
MODELS = ("ridge_wavlm", "cnn")
POLICIES = tuple(itertools.product(MODELS, (288, 576), (12, 48), ("prompt_seen", "prompt_new")))
require = core_verify.require
read_json = core_verify.read_json
file_sha = core_verify.byte_hash


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replay_ledger(path, known_units, plan_sha):
    """Independently enforce at most two starts and one final successful attempt."""
    path = Path(path)
    require(path.is_file(), "incomplete core: missing attempt ledger")
    payload = path.read_bytes()
    counts = {uid: 0 for uid in known_units}
    states = {uid: None for uid in known_units}
    for line in payload.decode("utf-8").splitlines():
        event = json.loads(line, object_pairs_hook=core_verify._object)
        require(isinstance(event, dict), "invalid ledger event")
        if "plan_sha256" in event:
            require(event["plan_sha256"] == plan_sha, "ledger plan identity mismatch")
        kind = event.get("event")
        if kind not in {"unit_start", "unit_failed", "unit_done"}:
            continue
        uid, attempt = event.get("unit_id"), event.get("attempt")
        require(uid in counts, "ledger has an unplanned unit")
        require(type(attempt) is int and 1 <= attempt <= 2, "ledger attempt exceeds allowed range")
        if kind == "unit_start":
            require(states[uid] in {None, "unit_failed"}, "illegal repeated/start-after-done ledger transition")
            require(attempt == counts[uid] + 1, "ledger start attempt number mismatch")
            counts[uid] = attempt
        else:
            require(states[uid] == "unit_start", "terminal ledger event without active start")
            require(attempt == counts[uid], "ledger terminal attempt number mismatch")
        states[uid] = kind
    require(all(states[uid] == "unit_done" and counts[uid] in (1, 2) for uid in known_units),
            "incomplete ledger: every planned unit needs a terminal done")
    return counts, hashlib.sha256(payload).hexdigest()


def verify_unit_artifacts(plan_sha, unit, directory, expected_done_sha, environment, expected_attempt):
    done_path = directory / "DONE"
    require(done_path.is_file() and file_sha(done_path) == expected_done_sha, "DONE missing or hash mismatch")
    done = read_json(done_path)
    require(done.get("schema") == "ser-study2-unit-done-1" and done.get("unit_id") == unit["unit_id"]
            and done.get("plan_sha256") == plan_sha, "DONE identity mismatch")
    for name, field in (("unit.json", "unit_json_sha256"), ("predictions.npz", "predictions_sha256")):
        target = directory / name
        require(target.is_file() and file_sha(target) == done.get(field), f"{name} hash mismatch")
    receipt = read_json(directory / "unit.json")
    require(receipt.get("schema") == "ser-study2-unit-1" and receipt.get("unit_id") == unit["unit_id"]
            and receipt.get("plan_sha256") == plan_sha and receipt.get("model") == unit["model"]
            and receipt.get("predictions_sha256") == done["predictions_sha256"], "unit receipt identity mismatch")
    require(receipt.get("environment") == environment, "unit environment differs from model run identity")
    require(type(expected_attempt) is int and expected_attempt in (1, 2)
            and type(receipt.get("attempt")) is int and receipt["attempt"] == expected_attempt,
            "receipt attempt must match the valid terminal ledger attempt")
    require(all(type(receipt.get(k)) in (float, int) and math.isfinite(receipt[k]) and receipt[k] >= 0
                for k in ("fit_seconds", "wall_seconds")), "invalid timing receipt")
    return done["predictions_sha256"]


def verify_closed(plan, run):
    """Hash every required artifact before any prediction array is interpreted."""
    run = Path(run)
    require(not (run / ".core-run.lock").exists(), "active or unresolved runner lock")
    units = plan["units"]
    known = {u["unit_id"] for u in units}
    require(len(units) == len(known) == EXPECTED_UNITS, "incomplete core plan")
    for filename in ("analysis_lock.json", "completion.json", "run_identity.json"):
        require((run / filename).is_file(), f"incomplete core: missing {filename}")
    plan_sha = plan["plan_sha256"]
    require(read_json(run / "run_identity.json") == {"schema": "ser-study2-run-1", "program": PROGRAM,
                                                   "plan_sha256": plan_sha}, "run identity mismatch")
    lock, completion = read_json(run / "analysis_lock.json"), read_json(run / "completion.json")
    done = completion.get("done_sha256", {})
    require(isinstance(done, dict) and set(done) == known, "incomplete DONE closure")
    require(completion == {"schema": "ser-study2-completion-1", "program": PROGRAM, "plan_sha256": plan_sha,
                           "n_units": EXPECTED_UNITS, "units_done": EXPECTED_UNITS, "done_sha256": done}, "completion identity mismatch")
    expected_lock = {"schema": "ser-study2-analysis-lock-1", "program": PROGRAM, "plan_sha256": plan_sha,
                     "n_units": EXPECTED_UNITS, "done_sha256": done, "completion_sha256": file_sha(run / "completion.json")}
    require({k: v for k, v in lock.items() if k != "created_at"} == expected_lock, "analysis lock mismatch")
    directory = run / "units"
    require(directory.is_dir() and {p.name for p in directory.iterdir()} == known, "missing or unplanned unit directories")
    # Presence is checked for all units first, before even opening one NPZ.
    require(all((directory / uid / name).is_file() for uid in known
                for name in ("DONE", "unit.json", "predictions.npz")), "incomplete unit artifacts")
    attempts, ledger_sha = replay_ledger(run / "ledger.jsonl", known, plan_sha)
    environments = {model: read_json(run / f"environment_{model}.json") for model in MODELS}
    prediction_hashes = {u["unit_id"]: verify_unit_artifacts(plan_sha, u, directory / u["unit_id"],
                         done[u["unit_id"]], environments[u["model"]], attempts[u["unit_id"]]) for u in units}
    identity = {"plan_sha256": plan_sha, "analysis_lock_sha256": file_sha(run / "analysis_lock.json"),
                "completion_sha256": file_sha(run / "completion.json"), "verified_units": EXPECTED_UNITS,
                "done_mapping_sha256": hashlib.sha256(canonical(done)).hexdigest(), "ledger_sha256": ledger_sha}
    return identity, prediction_hashes


def verify_closure_unchanged(run, identity):
    run = Path(run)
    require(not (run / ".core-run.lock").exists(), "runner restarted during scoring")
    for name, field in (("analysis_lock.json", "analysis_lock_sha256"),
                        ("completion.json", "completion_sha256"), ("ledger.jsonl", "ledger_sha256")):
        require((run / name).is_file() and file_sha(run / name) == identity[field],
                f"{name} changed during scoring")


def fixed_uar(correct, support):
    correct, support = np.asarray(correct), np.asarray(support)
    require(correct.shape == support.shape == (6,) and np.isfinite(correct).all() and np.isfinite(support).all()
            and (support > 0).all() and (correct >= 0).all() and (correct <= support).all(), "missing or invalid fixed-class support")
    return float(np.mean(correct / support) * 100.0)


def mean_draws(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] == 3 and np.isfinite(values).all(), "three complete draw vectors required")
    return values.mean(axis=0)


def bootstrap_indices():
    return np.random.default_rng(20260905).integers(0, 91, size=(10000, 91))


def estimate(draw_values, indices):
    values = mean_draws(draw_values)
    require(values.shape == (91,) and indices.shape == (10000, 91), "wrong speaker/bootstrap population")
    sampled = values[indices].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5], method="linear")
    draws = np.asarray(draw_values).mean(axis=1)
    return {"estimate_pp": float(values.mean()), "ci95_percentile_pp": [float(low), float(high)],
            "n_speakers": 91, "draw_estimates_pp": draws.tolist(),
            "draw_range_pp": [float(draws.min()), float(draws.max())]}


def collect_draws(plan, manifest, run, prediction_hashes):
    speakers = sorted({r["speaker"] for r in manifest.values()})
    require(len(speakers) == 91, "analysis requires the complete 91-speaker population")
    stats = {}
    for unit in plan["units"]:
        payload = (Path(run) / "units" / unit["unit_id"] / "predictions.npz").read_bytes()
        require(hashlib.sha256(payload).hexdigest() == prediction_hashes[unit["unit_id"]], "prediction changed after closure verification")
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            require(set(z.files) == {"paths", "logits"}, "unexpected prediction columns")
            paths, logits = z["paths"], z["logits"]
        require(paths.ndim == 1 and paths.dtype.kind == "U" and paths.tolist() == unit["test"], "prediction path identity mismatch")
        require(logits.shape == (len(paths), 6) and logits.dtype.kind == "f" and np.isfinite(logits).all(), "invalid logits")
        predictions = logits.argmax(axis=1)  # Ties choose the lowest fixed class index.
        grouped = defaultdict(list)
        for i, path in enumerate(paths.tolist()):
            grouped[manifest[path]["speaker"]].append(i)
        policy = tuple(unit[k] for k in ("model", "B", "S", "scenario"))
        for speaker, positions in grouped.items():
            key = policy + (unit["draw"], speaker)
            if key not in stats:
                stats[key] = [np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64), set(), set(), set()]
            correct, support, rotations, folds, seen_paths = stats[key]
            require(unit["rotation"] not in rotations, "duplicate speaker/rotation predictions")
            rotations.add(unit["rotation"])
            folds.add(unit["fold"])
            selected_paths = [str(paths[i]) for i in positions]
            require(not seen_paths.intersection(selected_paths), "duplicated test recording across rotations")
            seen_paths.update(selected_paths)
            truth = np.asarray([int(manifest[p]["label_index"]) for p in selected_paths])
            predicted = predictions[positions]
            support += np.bincount(truth, minlength=6)
            correct += np.bincount(truth[truth == predicted], minlength=6)
    expected = {policy + (draw, speaker) for policy in POLICIES for draw in range(3) for speaker in speakers}
    require(set(stats) == expected, "incomplete policy/draw/speaker prediction population")
    values = {policy: np.empty((3, 91), dtype=np.float64) for policy in POLICIES}
    for policy in POLICIES:
        for draw in range(3):
            for i, speaker in enumerate(speakers):
                correct, support, rotations, folds, _ = stats[policy + (draw, speaker)]
                require(rotations == set(range(6)) and len(folds) == 1, "incomplete six-rotation speaker aggregation")
                values[policy][draw, i] = fixed_uar(correct, support)
    return speakers, values


def summarize(values, indices):
    require(set(values) == set(POLICIES), "all 16 policies must be reported")
    absolute, contrasts = [], []
    for model, budget, count, scenario in POLICIES:
        absolute.append({"model": model, "B": budget, "S": count, "scenario": scenario,
                         **estimate(values[(model, budget, count, scenario)], indices)})
    for model, budget in itertools.product(MODELS, (288, 576)):
        differences = {}
        for scenario in ("prompt_seen", "prompt_new"):
            differences[scenario] = values[(model, budget, 48, scenario)] - values[(model, budget, 12, scenario)]
            contrasts.append({"model": model, "B": budget, "contrast": "S48_minus_S12", "scenario": scenario,
                              "primary": scenario == "prompt_new", **estimate(differences[scenario], indices)})
        contrasts.append({"model": model, "B": budget, "contrast": "new_minus_seen_of_S48_minus_S12",
                          "primary": False, **estimate(differences["prompt_new"] - differences["prompt_seen"], indices)})
    return {"absolute_uar": absolute, "contrasts": contrasts}


def score(repo, plan_path, run, out):
    repo, plan_path, run, out = map(lambda p: Path(p).resolve(), (repo, plan_path, run, out))
    require(out not in (repo, run) and not any(out.is_relative_to(repo / p) for p in ("v2", "v2_1")), "unsafe analysis output directory")
    verification = core_verify.verify_plan_file(repo, plan_path, metadata_only=True)
    plan = read_json(plan_path)
    require(plan["plan_sha256"] == verification["plan_sha256"]
            == hashlib.sha256(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})).hexdigest(),
            "plan changed after metadata verification")
    require(plan.get("design", {}).get("runtime_complete") is True, "draft plan cannot be scored")
    identity, hashes = verify_closed(plan, run)
    identity.update(scorer_sha256=file_sha(__file__), numpy_version=np.__version__, python_version=sys.version.split()[0])
    if out.exists() and any(out.iterdir()):
        require((out / "score_identity.json").is_file() and read_json(out / "score_identity.json") == identity, "existing output has another analysis identity")
    manifest_bytes = (repo / plan["input"]["manifest_path"]).read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == plan["input"]["manifest_sha256"], "manifest changed after metadata verification")
    manifest = {r["relative_path"]: r for r in csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig")))}
    speakers, values = collect_draws(plan, manifest, run, hashes)
    indices = bootstrap_indices()
    report = {"schema": "ser-study2-score-1", "identity": identity, "metadata_verification": verification,
              "method": {"classes": list(range(6)), "units": "UAR percentage points", "speaker_order": speakers,
                         "order": "pool six rotations per speaker/draw, fixed-class recall mean, then average three draws",
                         "bootstrap_seed": 20260905, "bootstrap_replicates": 10000, "shared_indices": True,
                         "index_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
                         "interval": "pointwise 95% percentile; conditional on executed fits and partitions",
                         "scorer_versioning": "separate code SHA implementing the plan-pinned scoring specification",
                         "draws_are_independent_samples": False, "confirmatory_p_values": False,
                         "difference_in_differences": "(new S48-new S12)-(seen S48-seen S12)"},
              **summarize(values, indices)}
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["model", "B", "S", "scenario", "speaker", "draw0_uar_pp", "draw1_uar_pp", "draw2_uar_pp", "mean_uar_pp"])
    for policy in POLICIES:
        means = mean_draws(values[policy])
        for i, speaker in enumerate(speakers):
            writer.writerow([*policy, speaker, *values[policy][:, i].tolist(), float(means[i])])
    verify_closure_unchanged(run, identity)
    atomic_bytes(out / "score_identity.json", canonical(identity) + b"\n")
    atomic_bytes(out / "per_speaker.csv", buffer.getvalue().encode("utf-8"))
    report["per_speaker_sha256"] = file_sha(out / "per_speaker.csv")
    atomic_bytes(out / "score.json", canonical(report) + b"\n")
    atomic_bytes(out / "SCORE_DONE", canonical({"identity": identity, "score_sha256": file_sha(out / "score.json"),
                 "per_speaker_sha256": report["per_speaker_sha256"]}) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "run", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    report = score(args.repo, args.plan, args.run, args.out)
    print(json.dumps({"scored": True, "plan_sha256": report["identity"]["plan_sha256"],
                      "absolute_estimates": len(report["absolute_uar"]), "contrasts": len(report["contrasts"])}))


if __name__ == "__main__":
    main()

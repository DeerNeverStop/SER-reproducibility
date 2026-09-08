"""Independent Study II result replay, after all 1,440 units are closed.

Never imports core_score or core_run.  Only the frozen metadata verifier is
shared.  --analysis points to core_score's directory containing score.json,
per_speaker.csv, score_identity.json and SCORE_DONE (not results.json).
--out must be the analysis directory's verification.json. Incomplete runs are rejected
before any NPZ is parsed.  CPU-only; this module contains no fitting code.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

# Set these before importing numerical libraries. No torch/CUDA import occurs.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"

import numpy as np

from v3.data_design import core_verify

PROGRAM = "SER26-STUDY2-CORE-1"
MODELS = ("ridge_wavlm", "cnn")
POLICIES = tuple(itertools.product(MODELS, (288, 576), (12, 48), ("prompt_seen", "prompt_new")))
N_UNITS, N_SPEAKERS, N_DRAWS, N_RESAMPLES = 1440, 91, 3, 10000
TOLERANCE = 1e-10
HEX = re.compile(r"[0-9a-f]{64}\Z")
CSV_COLUMNS = ["model", "B", "S", "scenario", "speaker", "draw0_uar_pp", "draw1_uar_pp", "draw2_uar_pp", "mean_uar_pp"]


class ResultVerificationError(ValueError):
    pass


def demand(condition, message):
    if not condition:
        raise ResultVerificationError(message)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        demand(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ResultVerificationError(f"nonfinite JSON token: {value}")


def parse_json(payload):
    return json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_pairs, parse_constant=_nonfinite)


def checked_read(path, snapshot):
    path = Path(path).resolve()
    demand(path.is_file(), f"required file missing: {path.name}")
    payload = path.read_bytes()
    snapshot[str(path)] = sha_bytes(payload)
    return payload


def _json(path, snapshot):
    value = parse_json(checked_read(path, snapshot))
    demand(isinstance(value, dict), f"JSON object required: {Path(path).name}")
    return value


def attempts_from_bytes(payload, known, plan_sha):
    """Separate state-machine implementation: idle/failed -> open -> done."""
    state = {uid: [0, "idle"] for uid in known}
    for line in payload.decode("utf-8").splitlines():
        event = parse_json(line.encode("utf-8"))
        demand(isinstance(event, dict), "invalid attempt ledger object")
        if "plan_sha256" in event:
            demand(event["plan_sha256"] == plan_sha, "attempt ledger plan mismatch")
        kind = event.get("event")
        if kind not in ("unit_start", "unit_failed", "unit_done"):
            continue
        uid, number = event.get("unit_id"), event.get("attempt")
        demand(uid in state, "unplanned unit in attempt ledger")
        demand(type(number) is int and number in (1, 2), "attempt limit/type violated")
        previous_number, phase = state[uid]
        if kind == "unit_start":
            demand(phase in ("idle", "failed") and number == previous_number + 1, "illegal attempt start transition")
            state[uid] = [number, "open"]
        else:
            demand(phase == "open" and number == previous_number, "illegal attempt terminal transition")
            state[uid][1] = "done" if kind == "unit_done" else "failed"
    demand(all(phase == "done" for _, phase in state.values()), "incomplete attempt ledger")
    return {uid: value[0] for uid, value in state.items()}


def check_closed(plan, run, snapshot):
    """Complete presence/identity/hash audit, with no NPZ interpretation."""
    run = Path(run).resolve()
    demand(not (run / ".core-run.lock").exists(), "active or unresolved run lock")
    units = plan.get("units", [])
    known = {unit.get("unit_id") for unit in units}
    demand(len(units) == len(known) == N_UNITS and all(isinstance(uid, str) and HEX.fullmatch(uid) for uid in known),
           "full 1440-unit plan required")
    coordinates = {tuple(u[k] for k in ("model", "B", "S", "scenario", "draw", "fold", "rotation")) for u in units}
    expected = {policy + (draw, fold, rotation) for policy in POLICIES for draw in range(3) for fold in range(5) for rotation in range(6)}
    demand(coordinates == expected, "incomplete factorial coordinates")
    # Reject the presently incomplete 720-unit run before opening any arrays.
    required = [run / name for name in ("completion.json", "analysis_lock.json", "run_identity.json", "ledger.jsonl")]
    required += [run / f"environment_{model}.json" for model in MODELS]
    demand(all(path.is_file() for path in required), "incomplete closure: required receipt absent")
    base = run / "units"
    demand(base.is_dir() and {p.name for p in base.iterdir()} == known, "missing or extra unit directories")
    demand(all((base / uid / name).is_file() for uid in known for name in ("DONE", "unit.json", "predictions.npz")),
           "incomplete unit artifact set")
    plan_sha = plan["plan_sha256"]
    identity = _json(run / "run_identity.json", snapshot)
    demand(identity == {"schema": "ser-study2-run-1", "program": PROGRAM, "plan_sha256": plan_sha}, "run identity mismatch")
    completion = _json(run / "completion.json", snapshot)
    mapping = completion.get("done_sha256", {})
    demand(isinstance(mapping, dict) and set(mapping) == known and all(isinstance(v, str) and HEX.fullmatch(v) for v in mapping.values()), "DONE closure population mismatch")
    demand(completion == {"schema": "ser-study2-completion-1", "program": PROGRAM, "plan_sha256": plan_sha,
                          "n_units": N_UNITS, "units_done": N_UNITS, "done_sha256": mapping}, "completion identity mismatch")
    lock = _json(run / "analysis_lock.json", snapshot)
    demand({k: v for k, v in lock.items() if k != "created_at"} == {
        "schema": "ser-study2-analysis-lock-1", "program": PROGRAM, "plan_sha256": plan_sha,
        "n_units": N_UNITS, "done_sha256": mapping,
        "completion_sha256": snapshot[str((run / "completion.json").resolve())]}, "analysis lock mismatch")
    ledger = checked_read(run / "ledger.jsonl", snapshot)
    attempts = attempts_from_bytes(ledger, known, plan_sha)
    environments = {m: _json(run / f"environment_{m}.json", snapshot) for m in MODELS}
    prediction_hashes = {}
    for unit in units:
        uid, folder = unit["unit_id"], base / unit["unit_id"]
        done = _json(folder / "DONE", snapshot)
        demand(snapshot[str((folder / "DONE").resolve())] == mapping[uid], "DONE byte hash mismatch")
        demand(set(done) == {"schema", "unit_id", "plan_sha256", "unit_json_sha256", "predictions_sha256"}
               and done["schema"] == "ser-study2-unit-done-1" and done["unit_id"] == uid and done["plan_sha256"] == plan_sha,
               "DONE identity/fields mismatch")
        receipt = _json(folder / "unit.json", snapshot)
        demand(snapshot[str((folder / "unit.json").resolve())] == done["unit_json_sha256"], "unit receipt hash mismatch")
        demand(receipt.get("schema") == "ser-study2-unit-1" and receipt.get("unit_id") == uid
               and receipt.get("model") == unit["model"] and receipt.get("plan_sha256") == plan_sha,
               "unit receipt identity mismatch")
        demand(receipt.get("environment") == environments[unit["model"]], "unit/model environment mismatch")
        demand(type(receipt.get("attempt")) is int and receipt["attempt"] == attempts[uid], "receipt/ledger attempt mismatch")
        demand(all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) and receipt[k] >= 0
                   for k in ("fit_seconds", "wall_seconds")), "invalid unit timing")
        expected_prediction = done["predictions_sha256"]
        demand(isinstance(expected_prediction, str) and HEX.fullmatch(expected_prediction)
               and receipt.get("predictions_sha256") == expected_prediction, "prediction receipt identity mismatch")
        prediction_path = (folder / "predictions.npz").resolve()
        actual = sha_file(prediction_path)
        demand(actual == expected_prediction, "prediction byte hash mismatch")
        snapshot[str(prediction_path)] = actual
        prediction_hashes[uid] = actual
    result_identity = {"plan_sha256": plan_sha,
                       "analysis_lock_sha256": snapshot[str((run / "analysis_lock.json").resolve())],
                       "completion_sha256": snapshot[str((run / "completion.json").resolve())],
                       "verified_units": N_UNITS, "done_mapping_sha256": sha_bytes(json_bytes(mapping)),
                       "ledger_sha256": sha_bytes(ledger)}
    return result_identity, prediction_hashes


def reconstruct(plan, manifest, run, prediction_hashes):
    """Accumulate truth-by-prediction confusion matrices, then fixed-class UAR."""
    speakers = sorted({row["speaker"] for row in manifest.values()})
    demand(len(speakers) == N_SPEAKERS, "91 test speakers required")
    sp_index = {speaker: i for i, speaker in enumerate(speakers)}
    policy_index = {policy: i for i, policy in enumerate(POLICIES)}
    matrices = np.zeros((16, 3, 91, 6, 6), dtype=np.int64)
    masks = np.zeros((16, 3, 91), dtype=np.uint8)
    folds = np.full((16, 3, 91), -1, dtype=np.int8)
    used = collections.defaultdict(set)
    for unit in sorted(plan["units"], key=lambda u: u["unit_id"]):
        payload = (Path(run) / "units" / unit["unit_id"] / "predictions.npz").read_bytes()
        demand(sha_bytes(payload) == prediction_hashes[unit["unit_id"]], "predictions changed after closure audit")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            demand(set(archive.files) == {"paths", "logits"}, "prediction array columns mismatch")
            paths, logits = archive["paths"], archive["logits"]
        demand(paths.ndim == 1 and paths.dtype.kind == "U" and paths.tolist() == unit["test"], "prediction path order/identity mismatch")
        demand(logits.shape == (len(paths), 6) and logits.dtype.kind == "f" and np.isfinite(logits).all(), "nonfinite or malformed logits")
        policy = tuple(unit[k] for k in ("model", "B", "S", "scenario"))
        pi, draw = policy_index[policy], unit["draw"]
        selected_paths = paths.tolist()
        demand(not used[(pi, draw)].intersection(selected_paths), "test path repeated across rotations/folds")
        used[(pi, draw)].update(selected_paths)
        demand(all(path in manifest for path in selected_paths), "unknown test path")
        ids = np.array([sp_index[manifest[p]["speaker"]] for p in selected_paths], dtype=np.int64)
        truth = np.array([int(manifest[p]["label_index"]) for p in selected_paths], dtype=np.int64)
        demand(np.isin(truth, np.arange(6)).all(), "noncontract test label")
        predicted = np.argmax(logits, axis=1)
        rotation_bit = 1 << unit["rotation"]
        for si in set(ids.tolist()):
            demand(not (int(masks[pi, draw, si]) & rotation_bit), "duplicate speaker rotation")
            demand(folds[pi, draw, si] in (-1, unit["fold"]), "speaker crosses outer folds")
            masks[pi, draw, si] |= rotation_bit
            folds[pi, draw, si] = unit["fold"]
        np.add.at(matrices[pi, draw], (ids, truth, predicted), 1)
    demand((masks == 63).all(), "incomplete six-rotation speaker cover")
    support = matrices.sum(axis=-1)
    demand((support > 0).all(), "missing fixed-class speaker support")
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    values = np.add.reduce(diagonal / support, axis=-1) * (100.0 / 6.0)
    return speakers, values


def bootstrap_design():
    indices = np.random.default_rng(20260905).integers(0, 91, size=(10000, 91))
    weights = np.zeros((10000, 91), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(10000), 91), indices.reshape(-1)), 1.0)
    return indices, weights


def linear_quantile(sorted_values, probability):
    location = (len(sorted_values) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    return float(sorted_values[lower] + (location - lower) * (sorted_values[upper] - sorted_values[lower]))


def one_estimate(draw_vectors, weights):
    values = np.asarray(draw_vectors, dtype=np.float64)
    demand(values.shape == (3, 91) and np.isfinite(values).all(), "wrong estimation population")
    averaged = np.array([math.fsum(values[:, i]) / 3.0 for i in range(91)])
    simulated = np.sort(weights @ averaged / 91.0)
    draw_means = [math.fsum(row) / 91.0 for row in values]
    return {"estimate_pp": math.fsum(averaged) / 91.0,
            "ci95_percentile_pp": [linear_quantile(simulated, .025), linear_quantile(simulated, .975)],
            "n_speakers": 91, "draw_estimates_pp": draw_means,
            "draw_range_pp": [min(draw_means), max(draw_means)]}


def independent_summaries(values, weights):
    demand(values.shape == (16, 3, 91), "missing policy vectors")
    lookup = {policy: values[i] for i, policy in enumerate(POLICIES)}
    absolute = []
    for policy in POLICIES:
        model, budget, count, condition = policy
        absolute.append({"model": model, "B": budget, "S": count, "scenario": condition,
                         **one_estimate(lookup[policy], weights)})
    differences = []
    for model in MODELS:
        for budget in (288, 576):
            seen = lookup[(model, budget, 48, "prompt_seen")] - lookup[(model, budget, 12, "prompt_seen")]
            new = lookup[(model, budget, 48, "prompt_new")] - lookup[(model, budget, 12, "prompt_new")]
            for name, vector in (("prompt_seen", seen), ("prompt_new", new)):
                differences.append({"model": model, "B": budget, "contrast": "S48_minus_S12", "scenario": name,
                                    "primary": name == "prompt_new", **one_estimate(vector, weights)})
            differences.append({"model": model, "B": budget, "contrast": "new_minus_seen_of_S48_minus_S12",
                                "primary": False, **one_estimate(new - seen, weights)})
    return {"absolute_uar": absolute, "contrasts": differences}


class Comparator:
    def __init__(self):
        self.maximum = 0.0
        self.numeric_fields = 0

    def compare(self, expected, observed, path="$"):
        if isinstance(expected, dict):
            demand(isinstance(observed, dict) and set(expected) == set(observed), f"result keys mismatch: {path}")
            for key in expected:
                self.compare(expected[key], observed[key], path + "." + key)
        elif isinstance(expected, list):
            demand(isinstance(observed, list) and len(expected) == len(observed), f"result list mismatch: {path}")
            for i, (a, b) in enumerate(zip(expected, observed)):
                self.compare(a, b, f"{path}[{i}]")
        elif type(expected) is bool or expected is None or isinstance(expected, str):
            demand(type(expected) is type(observed) and expected == observed, f"result value/type mismatch: {path}")
        elif type(expected) is int:
            demand(type(observed) is int and expected == observed, f"integer population/identity mismatch: {path}")
            self.numeric_fields += 1
        else:
            demand(type(observed) in (int, float) and math.isfinite(observed), f"nonfinite/nonnumeric result: {path}")
            error = abs(float(expected) - float(observed))
            self.maximum = max(self.maximum, error)
            self.numeric_fields += 1
            demand(error <= TOLERANCE, f"numeric mismatch: {path}")


def compare_csv(payload, speakers, values, comparison):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    demand(reader.fieldnames == CSV_COLUMNS, "per-speaker CSV columns mismatch")
    lookup = {policy: i for i, policy in enumerate(POLICIES)}
    ids = {s: i for i, s in enumerate(speakers)}
    seen = set()
    for row in reader:
        demand(None not in row and all(v is not None for v in row.values()), "malformed per-speaker CSV row")
        policy = (row["model"], int(row["B"]), int(row["S"]), row["scenario"])
        speaker = row["speaker"]
        demand(policy in lookup and speaker in ids and (policy, speaker) not in seen, "duplicate/unknown per-speaker row")
        seen.add((policy, speaker))
        vector = values[lookup[policy], :, ids[speaker]]
        targets = [*vector.tolist(), math.fsum(vector) / 3.0]
        for column, target in zip(CSV_COLUMNS[5:], targets):
            comparison.compare(float(target), float(row[column]), "per_speaker." + column)
    demand(len(seen) == 16 * 91, "incomplete per-speaker CSV")


def _method(speakers, indices):
    return {"classes": list(range(6)), "units": "UAR percentage points", "speaker_order": speakers,
            "order": "pool six rotations per speaker/draw, fixed-class recall mean, then average three draws",
            "bootstrap_seed": 20260905, "bootstrap_replicates": 10000, "shared_indices": True,
            "index_sha256": sha_bytes(indices.astype("<i8", copy=False).tobytes()),
            "interval": "pointwise 95% percentile; conditional on executed fits and partitions",
            "scorer_versioning": "separate code SHA implementing the plan-pinned scoring specification",
            "draws_are_independent_samples": False, "confirmatory_p_values": False,
            "difference_in_differences": "(new S48-new S12)-(seen S48-seen S12)"}


def audit_analysis(repo, plan, run, analysis, manifest, metadata_report, snapshot, comparison, activity=None):
    """Complete replay; caller must have performed real metadata verification."""
    base_identity, prediction_hashes = check_closed(plan, run, snapshot)
    analysis = Path(analysis)
    for name in ("score.json", "per_speaker.csv", "score_identity.json", "SCORE_DONE"):
        demand((analysis / name).is_file(), f"complete analysis missing {name}")
    identity = _json(analysis / "score_identity.json", snapshot)
    demand(set(identity) == set(base_identity) | {"scorer_sha256", "numpy_version", "python_version"}, "score identity fields mismatch")
    for key, expected in base_identity.items():
        demand(identity.get(key) == expected, f"score identity differs from closed run: {key}")
    scorer_path = Path(repo) / "v3/data_design/core_score.py"
    scorer_sha = sha_bytes(checked_read(scorer_path, snapshot))
    demand(identity.get("scorer_sha256") == scorer_sha, "scorer version hash mismatch")
    demand(all(isinstance(identity.get(k), str) and identity[k] for k in ("numpy_version", "python_version")), "missing scorer environment versions")
    result = _json(analysis / "score.json", snapshot)
    csv_payload = checked_read(analysis / "per_speaker.csv", snapshot)
    csv_sha = sha_bytes(csv_payload)
    score_done = _json(analysis / "SCORE_DONE", snapshot)
    demand(score_done == {"identity": identity, "score_sha256": snapshot[str((analysis / "score.json").resolve())],
                         "per_speaker_sha256": csv_sha}, "SCORE_DONE binding mismatch")
    # No prediction array is interpreted until both complete closures pass.
    if activity is not None:
        activity["scientific_arrays_read"] = True
    speakers, values = reconstruct(plan, manifest, run, prediction_hashes)
    indices, weights = bootstrap_design()
    expected = {"schema": "ser-study2-score-1", "identity": identity,
                "metadata_verification": metadata_report, "method": _method(speakers, indices),
                **independent_summaries(values, weights), "per_speaker_sha256": csv_sha}
    comparison.compare(expected, result)
    compare_csv(csv_payload, speakers, values, comparison)
    return {"identity": identity, "n_units": 1440, "n_speakers": 91, "absolute_estimates": 16,
            "speaker_count_contrasts": 8, "interaction_contrasts": 4, "per_speaker_rows": 1456}


def unchanged(snapshot, run):
    demand(not (Path(run) / ".core-run.lock").exists(), "runner restarted during verification")
    for name, expected in snapshot.items():
        demand(Path(name).is_file() and sha_file(name) == expected, f"artifact changed during verification: {Path(name).name}")


def atomic_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(json_bytes(report) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify(repo, plan_path, run, analysis, out):
    repo, plan_path, run, analysis, out = (Path(p).resolve() for p in (repo, plan_path, run, analysis, out))
    demand(out == analysis / "verification.json" and out != plan_path and analysis != repo
           and not analysis.is_relative_to(run) and not analysis.is_relative_to(repo / "v2")
           and not analysis.is_relative_to(repo / "v2_1"), "unsafe verification output path")
    verifier_path = Path(__file__).resolve()
    verifier_sha = sha_file(verifier_path)
    snapshot, comparison = {str(verifier_path): verifier_sha}, Comparator()
    report = {"schema": "ser-study2-independent-result-verification-1", "pass": False,
              "tolerance_pp": TOLERANCE, "scientific_arrays_read": False}
    try:
        metadata = core_verify.verify_plan_file(repo, plan_path, metadata_only=True)
        plan = _json(plan_path, snapshot)
        demand(plan.get("plan_sha256") == metadata["plan_sha256"]
               == sha_bytes(json_bytes({k: v for k, v in plan.items() if k != "plan_sha256"})), "plan identity changed after metadata check")
        demand(plan.get("design", {}).get("runtime_complete") is True, "draft plan cannot be result-verified")
        manifest_path = (repo / plan["input"]["manifest_path"]).resolve()
        manifest_payload = checked_read(manifest_path, snapshot)
        demand(sha_bytes(manifest_payload) == plan["input"]["manifest_sha256"], "manifest identity changed after metadata check")
        reader = csv.DictReader(io.StringIO(manifest_payload.decode("utf-8-sig")))
        rows = list(reader)
        manifest = {row["relative_path"]: row for row in rows}
        demand(len(manifest) == len(rows), "duplicate manifest path during replay")
        # Capture every pinned source as well as demographics against mid-audit changes.
        for relative, expected in plan["source_sha256"].items():
            path = (repo / relative).resolve()
            demand(sha_file(path) == expected, "source changed after metadata check")
            snapshot[str(path)] = expected
        demographics = (repo / plan["input"]["demographics_path"]).resolve()
        snapshot[str(demographics)] = plan["input"]["demographics_sha256"]
        # This gate is intentionally repeated inside audit_analysis; before it,
        # the current 720-unit run causes no NPZ interpretation whatsoever.
        check_closed(plan, run, snapshot)
        details = audit_analysis(repo, plan, run, analysis, manifest, metadata, snapshot, comparison, report)
        unchanged(snapshot, run)
        report.update(details)
        report["pass"] = True
    except (ResultVerificationError, core_verify.VerificationError, OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        report["error"] = str(exc)
    report.update({"max_numeric_error": comparison.maximum, "numeric_fields_checked": comparison.numeric_fields,
                   "verifier_sha256": verifier_sha, "numpy_version": np.__version__,
                   "python_version": sys.version.split()[0], "file_hashes": snapshot,
                   "results_approved_for_paper": report["pass"]})
    atomic_report(out, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "run", "analysis", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    report = verify(args.repo, args.plan, args.run, args.analysis, args.out)
    print(json.dumps({k: report[k] for k in ("pass", "max_numeric_error", "numeric_fields_checked", "results_approved_for_paper")}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

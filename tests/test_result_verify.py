"""Generated TEST data only; never open a real work/core or pilot artifact.

The production verifier does not import the scorer. Tests use the scorer only
as a second implementation to produce reports from these synthetic fixtures.
"""
import ast
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import pytest

from v3.data_design import core_score as reference
from v3.data_design import result_verify as verify


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def ledger(sequence):
    return b"".join(verify.json_bytes({"event": kind, "unit_id": "TEST", "attempt": number}) + b"\n"
                    for kind, number in sequence)


@pytest.mark.parametrize("sequence", [
    [("unit_start", 1)],
    [("unit_done", 1)],
    [("unit_start", 2), ("unit_done", 2)],
    [("unit_start", 1), ("unit_done", 2)],
    [("unit_start", 1), ("unit_start", 2), ("unit_done", 2)],
    [("unit_start", 1), ("unit_done", 1), ("unit_start", 2)],
    [("unit_start", 1), ("unit_failed", 1), ("unit_start", 2), ("unit_failed", 2), ("unit_start", 3)],
    [("unit_start", True), ("unit_done", True)],
])
def test_strict_attempt_failures(sequence):
    with pytest.raises(verify.ResultVerificationError):
        verify.attempts_from_bytes(ledger(sequence), {"TEST"}, "TEST_plan")


def test_orphan_consumes_attempt_and_recovery_is_allowed():
    events = [("unit_start", 1), ("unit_failed", 1), ("unit_start", 2), ("unit_done", 2)]
    assert verify.attempts_from_bytes(ledger(events), {"TEST"}, "TEST_plan") == {"TEST": 2}


@pytest.fixture(scope="module")
def closed_TEST(tmp_path_factory):
    repo = tmp_path_factory.mktemp("generated_TEST_result_verification")
    run, analysis = repo / "TEST_run", repo / "TEST_analysis"
    run.mkdir(); analysis.mkdir()
    source = repo / "v3/data_design/core_score.py"
    source.parent.mkdir(parents=True)
    source.write_text("# TEST scorer source identity only\n", encoding="utf-8")
    environments = {m: {"device": "cpu", "fixture": "TEST", "model": m} for m in verify.MODELS}
    manifest, units = {}, []
    # Unequal supports deliberately distinguish pooling from rotation averaging.
    for policy in verify.POLICIES:
        for draw in range(3):
            for fold in range(5):
                for rotation in range(6):
                    paths, predictions = [], []
                    for speaker in range(fold, 91, 5):
                        for label in range(6):
                            for repeat in range(3 if rotation == 1 else 1):
                                p = f"TEST_s{speaker:03d}_p{rotation}_c{label}_r{repeat}.wav"
                                manifest[p] = {"speaker": f"s{speaker:03d}", "label_index": str(label), "relative_path": p}
                                paths.append(p)
                                shift = (2 if policy[3] == "prompt_new" else 1) if policy[2] == 48 else 0
                                correct = ((speaker + label + rotation + draw) % 6) < 2 + shift
                                predictions.append(label if correct else (label + 1) % 6)
                    unit = dict(zip(("model", "B", "S", "scenario"), policy))
                    unit.update(draw=draw, fold=fold, rotation=rotation, test=paths)
                    uid = verify.sha_bytes(verify.json_bytes(unit))
                    unit["unit_id"] = uid
                    folder = run / "units" / uid
                    folder.mkdir(parents=True)
                    np.savez(folder / "predictions.npz", paths=np.asarray(paths), logits=np.eye(6, dtype=np.float32)[predictions])
                    units.append(unit)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["relative_path", "speaker", "label_index"])
    writer.writeheader(); writer.writerows(manifest.values())
    manifest_path = repo / "TEST_manifest.csv"
    manifest_path.write_text(buffer.getvalue(), encoding="utf-8")
    demographics = repo / "TEST_demographics.csv"
    demographics.write_text("TEST only\n", encoding="utf-8")
    plan = {"program": verify.PROGRAM, "units": units, "design": {"runtime_complete": True},
            "input": {"manifest_path": manifest_path.name, "manifest_sha256": verify.sha_file(manifest_path),
                      "demographics_path": demographics.name, "demographics_sha256": verify.sha_file(demographics)},
            "source_sha256": {"v3/data_design/core_score.py": verify.sha_file(source)}}
    plan["plan_sha256"] = verify.sha_bytes(verify.json_bytes(plan))
    plan_path = repo / "TEST_plan.json"
    put(plan_path, plan)
    mapping, hashes, events = {}, {}, []
    for unit in units:
        uid = unit["unit_id"]; folder = run / "units" / uid
        hashes[uid] = verify.sha_file(folder / "predictions.npz")
        receipt = {"schema": "ser-study2-unit-1", "unit_id": uid, "model": unit["model"],
                   "plan_sha256": plan["plan_sha256"], "environment": environments[unit["model"]],
                   "predictions_sha256": hashes[uid], "attempt": 1, "fit_seconds": 1.0, "wall_seconds": 1.0}
        put(folder / "unit.json", receipt)
        put(folder / "DONE", {"schema": "ser-study2-unit-done-1", "unit_id": uid, "plan_sha256": plan["plan_sha256"],
                              "unit_json_sha256": verify.sha_file(folder / "unit.json"), "predictions_sha256": hashes[uid]})
        mapping[uid] = verify.sha_file(folder / "DONE")
        for event in ("unit_start", "unit_done"):
            events.append({"event": event, "unit_id": uid, "attempt": 1, "plan_sha256": plan["plan_sha256"]})
    (run / "ledger.jsonl").write_bytes(b"".join(verify.json_bytes(e) + b"\n" for e in events))
    for model, environment in environments.items():
        put(run / f"environment_{model}.json", environment)
    put(run / "run_identity.json", {"schema": "ser-study2-run-1", "program": verify.PROGRAM, "plan_sha256": plan["plan_sha256"]})
    put(run / "completion.json", {"schema": "ser-study2-completion-1", "program": verify.PROGRAM,
                                  "plan_sha256": plan["plan_sha256"], "n_units": 1440, "units_done": 1440, "done_sha256": mapping})
    put(run / "analysis_lock.json", {"schema": "ser-study2-analysis-lock-1", "program": verify.PROGRAM,
                                     "plan_sha256": plan["plan_sha256"], "n_units": 1440, "done_sha256": mapping,
                                     "completion_sha256": verify.sha_file(run / "completion.json")})
    metadata = {"pass": True, "plan_sha256": plan["plan_sha256"], "scope": "generated_TEST_fixture"}
    # Compute every reported scientific quantity through the other implementation.
    speaker_order, vectors = reference.collect_draws(plan, manifest, run, hashes)
    indices = reference.bootstrap_indices()
    identity, _ = verify.check_closed(plan, run, {})
    identity.update(scorer_sha256=verify.sha_file(source), numpy_version=np.__version__, python_version=sys.version.split()[0])
    put(analysis / "score_identity.json", identity)
    report = {"schema": "ser-study2-score-1", "identity": identity, "metadata_verification": metadata,
              "method": verify._method(speaker_order, indices), **reference.summarize(vectors, indices)}
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer); writer.writerow(verify.CSV_COLUMNS)
    for policy in reference.POLICIES:
        for si, speaker in enumerate(speaker_order):
            row = vectors[policy][:, si]
            writer.writerow([*policy, speaker, *row, row.mean()])
    (analysis / "per_speaker.csv").write_text(csv_buffer.getvalue(), encoding="utf-8", newline="")
    report["per_speaker_sha256"] = verify.sha_file(analysis / "per_speaker.csv")
    put(analysis / "score.json", report)
    reseal_analysis(analysis)
    return repo, plan, plan_path, run, analysis, manifest, metadata


def reseal_analysis(analysis):
    report = json.loads((analysis / "score.json").read_text())
    report["per_speaker_sha256"] = verify.sha_file(analysis / "per_speaker.csv")
    put(analysis / "score.json", report)
    put(analysis / "SCORE_DONE", {"identity": report["identity"], "score_sha256": verify.sha_file(analysis / "score.json"),
                                 "per_speaker_sha256": report["per_speaker_sha256"]})


def test_complete_raw_replay_matches_other_implementation(closed_TEST):
    repo, plan, _, run, analysis, manifest, metadata = closed_TEST
    snapshot, comparison = {}, verify.Comparator()
    result = verify.audit_analysis(repo, plan, run, analysis, manifest, metadata, snapshot, comparison)
    verify.unchanged(snapshot, run)
    assert result["n_speakers"] == 91 and result["n_units"] == 1440
    assert result["absolute_estimates"] == 16 and result["interaction_contrasts"] == 4
    assert comparison.maximum < 1e-10 and comparison.numeric_fields > 5800


def test_public_report_passes_only_with_metadata_gate(closed_TEST, monkeypatch, tmp_path):
    repo, _, plan_path, run, analysis, _, metadata = closed_TEST
    monkeypatch.setattr(verify.core_verify, "verify_plan_file", lambda *a, **k: metadata)
    report = verify.verify(repo, plan_path, run, analysis, analysis / "verification.json")
    assert report["pass"] and report["results_approved_for_paper"] and report["scientific_arrays_read"]
    assert report["file_hashes"] and report["max_numeric_error"] < 1e-10


def test_incomplete_closure_rejects_before_any_array(closed_TEST, tmp_path, monkeypatch):
    _, plan, _, _, _, _, _ = closed_TEST
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("must not parse prediction arrays"))
    with pytest.raises(ValueError, match="incomplete closure"):
        verify.check_closed(plan, tmp_path, {})


@pytest.mark.parametrize("change", ["estimate", "sample_size", "did_sign", "csv", "method"])
def test_resealed_scientific_tampering_is_rejected(closed_TEST, tmp_path, change):
    repo, plan, _, run, original, manifest, metadata = closed_TEST
    analysis = tmp_path / "TEST_changed_analysis"
    shutil.copytree(original, analysis)
    report = json.loads((analysis / "score.json").read_text())
    if change == "estimate": report["absolute_uar"][0]["estimate_pp"] += 0.01
    elif change == "sample_size": report["absolute_uar"][0]["n_speakers"] = 1440
    elif change == "did_sign": report["contrasts"][2]["estimate_pp"] *= -1
    elif change == "method": report["method"]["bootstrap_seed"] += 1
    else:
        rows = list(csv.reader(io.StringIO((analysis / "per_speaker.csv").read_text())))
        rows[1][-1] = str(float(rows[1][-1]) + .01)
        buffer = io.StringIO(); csv.writer(buffer).writerows(rows)
        (analysis / "per_speaker.csv").write_text(buffer.getvalue())
    put(analysis / "score.json", report); reseal_analysis(analysis)
    with pytest.raises(ValueError, match="mismatch"):
        verify.audit_analysis(repo, plan, run, analysis, manifest, metadata, {}, verify.Comparator())


def test_weighted_bootstrap_and_contrast_sign_have_analytic_reference():
    indices, weights = verify.bootstrap_design()
    base = np.vstack([np.linspace(11 + draw, 44 + draw, 91) for draw in range(3)])
    result = verify.one_estimate(base, weights)
    expected = np.percentile(base.mean(axis=0)[indices].mean(axis=1), [2.5, 97.5], method="linear")
    np.testing.assert_allclose(result["ci95_percentile_pp"], expected, atol=1e-12)
    values = np.array([base + ((7 if condition == "prompt_new" else 2) if count == 48 else 0)
                       for _, _, count, condition in verify.POLICIES])
    contrasts = verify.independent_summaries(values, weights)["contrasts"]
    for i, contrast in enumerate(contrasts):
        assert contrast["estimate_pp"] == pytest.approx((2, 7, 5)[i % 3])
        assert contrast["n_speakers"] == 91


def test_source_is_independent_and_json_is_strict():
    tree = ast.parse(Path(verify.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = " ".join(alias.name for alias in node.names) + " " + str(getattr(node, "module", ""))
            assert "core_score" not in imported and "core_run" not in imported
    for payload in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises(ValueError): verify.parse_json(payload)


def test_post_audit_file_changes_are_rejected(tmp_path):
    artifact = tmp_path / "TEST.txt"; artifact.write_bytes(b"before")
    snapshot = {str(artifact): verify.sha_file(artifact)}
    artifact.write_bytes(b"after")
    with pytest.raises(ValueError, match="changed during verification"):
        verify.unchanged(snapshot, tmp_path)


@pytest.mark.parametrize("target", ["v3/data_design/evidence/PLAN_LOCK.json", "v2/verification.json", "verification.json"])
def test_output_cannot_overwrite_frozen_or_uncontrolled_files(tmp_path, target):
    protected = tmp_path / target
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_bytes(b"TEST protected bytes")
    with pytest.raises(ValueError, match="unsafe verification output"):
        verify.verify(tmp_path, tmp_path / "plan.json", tmp_path / "run", protected.parent, protected)
    assert protected.read_bytes() == b"TEST protected bytes"


@pytest.mark.parametrize("field,value", [("attempt", 2), ("environment", {"device": "forged"}), ("plan_sha256", "e" * 64)])
def test_resealed_receipt_cannot_bypass_binding(closed_TEST, field, value, monkeypatch):
    _, plan, _, run, _, _, _ = closed_TEST
    unit = plan["units"][0]; folder = run / "units" / unit["unit_id"]
    paths = [folder / "unit.json", folder / "DONE", run / "completion.json", run / "analysis_lock.json"]
    original = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("invalid receipt must precede array parsing"))
    try:
        receipt = json.loads(original[paths[0]]); receipt[field] = value; put(paths[0], receipt)
        done = json.loads(original[paths[1]]); done["unit_json_sha256"] = verify.sha_file(paths[0]); put(paths[1], done)
        completion = json.loads(original[paths[2]]); completion["done_sha256"][unit["unit_id"]] = verify.sha_file(paths[1]); put(paths[2], completion)
        lock = json.loads(original[paths[3]]); lock["done_sha256"] = completion["done_sha256"]
        lock["completion_sha256"] = verify.sha_file(paths[2]); put(paths[3], lock)
        with pytest.raises(ValueError, match="mismatch"):
            verify.check_closed(plan, run, {})
    finally:
        for path, payload in original.items(): path.write_bytes(payload)

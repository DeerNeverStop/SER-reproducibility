"""Generated TEST fixtures only. Never access real work predictions or scores."""
import hashlib

import numpy as np
import pytest

from v3.data_design import core_score as score


def test_fixed_class_uar_differs_from_sample_accuracy():
    correct = np.array([90, 0, 0, 0, 0, 0])
    support = np.array([90, 1, 1, 1, 1, 1])
    assert score.fixed_uar(correct, support) == pytest.approx(100 / 6)
    assert score.fixed_uar(correct, support) != pytest.approx(100 * correct.sum() / support.sum())
    with pytest.raises(ValueError, match="fixed-class support"):
        score.fixed_uar(correct, np.array([90, 1, 1, 1, 1, 0]))


def test_pool_rotations_before_uar_then_average_draws():
    supports = [np.ones(6, dtype=int), np.full(6, 9)] + [np.ones(6, dtype=int)] * 4
    correct = [np.ones(6, dtype=int)] + [np.zeros(6, dtype=int)] * 5
    pooled = score.fixed_uar(np.sum(correct, axis=0), np.sum(supports, axis=0))
    rotation_average = np.mean([score.fixed_uar(c, n) for c, n in zip(correct, supports)])
    assert pooled == pytest.approx(100 / 14) and pooled != pytest.approx(rotation_average)
    draws = np.array([[pooled], [50.0], [100.0]])
    assert score.mean_draws(draws)[0] == pytest.approx((100 / 14 + 50 + 100) / 3)


def test_paired_signs_draw_reporting_and_shared_bootstrap():
    indices = score.bootstrap_indices()
    expected = np.random.default_rng(20260905).integers(0, 91, size=(10000, 91))
    np.testing.assert_array_equal(indices, expected)
    base = np.vstack([np.linspace(20 + draw, 40 + draw, 91) for draw in range(3)])
    values = {}
    for model, budget, count, scenario in score.POLICIES:
        effect = (7 if scenario == "prompt_new" else 2) if count == 48 else 0
        values[(model, budget, count, scenario)] = base + effect
    report = score.summarize(values, indices)
    assert len(report["absolute_uar"]) == 16 and len(report["contrasts"]) == 12
    assert sum(c["primary"] for c in report["contrasts"]) == 4
    for contrast in report["contrasts"]:
        expected_effect = 5 if contrast["contrast"].startswith("new_minus_seen") else (7 if contrast["scenario"] == "prompt_new" else 2)
        assert contrast["estimate_pp"] == pytest.approx(expected_effect)
        assert contrast["ci95_percentile_pp"] == pytest.approx([expected_effect, expected_effect])
        assert contrast["draw_estimates_pp"] == pytest.approx([expected_effect] * 3)
        assert contrast["n_speakers"] == 91
    first = score.estimate(base, indices)
    doubled = score.estimate(2 * base, indices)
    assert doubled["ci95_percentile_pp"] == pytest.approx(np.array(first["ci95_percentile_pp"]) * 2)


def test_incomplete_core_is_rejected_before_numpy_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prediction arrays must remain unread")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="incomplete core plan"):
        score.verify_closed({"units": [], "plan_sha256": "0" * 64}, tmp_path)
    units = [{"unit_id": str(i)} for i in range(1440)]
    with pytest.raises(ValueError, match="missing analysis_lock"):
        score.verify_closed({"units": units, "plan_sha256": "0" * 64}, tmp_path)


@pytest.fixture
def one_completed_TEST_unit(tmp_path):
    unit, plan_sha, environment = {"unit_id": "a" * 64, "model": "ridge_wavlm"}, "b" * 64, {"device": "cpu", "fixture": "TEST"}
    (tmp_path / "predictions.npz").write_bytes(b"TEST opaque bytes: integrity test never interprets this payload")
    prediction_sha = score.file_sha(tmp_path / "predictions.npz")
    receipt = {"schema": "ser-study2-unit-1", "unit_id": unit["unit_id"], "plan_sha256": plan_sha,
               "model": unit["model"], "environment": environment, "predictions_sha256": prediction_sha,
               "fit_seconds": 1.0, "wall_seconds": 2.0, "attempt": 1}
    score.atomic_bytes(tmp_path / "unit.json", score.canonical(receipt))
    done = {"schema": "ser-study2-unit-done-1", "unit_id": unit["unit_id"], "plan_sha256": plan_sha,
            "unit_json_sha256": score.file_sha(tmp_path / "unit.json"), "predictions_sha256": prediction_sha}
    score.atomic_bytes(tmp_path / "DONE", score.canonical(done))
    return plan_sha, unit, tmp_path, score.file_sha(tmp_path / "DONE"), environment, 1


@pytest.mark.parametrize("target", ["DONE", "unit.json", "predictions.npz"])
def test_each_artifact_hash_is_verified(one_completed_TEST_unit, target):
    args = one_completed_TEST_unit
    assert score.verify_unit_artifacts(*args) == score.file_sha(args[2] / "predictions.npz")
    with (args[2] / target).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        score.verify_unit_artifacts(*args)


def test_wrong_plan_or_environment_is_rejected(one_completed_TEST_unit):
    plan_sha, unit, path, done_sha, environment, attempt = one_completed_TEST_unit
    with pytest.raises(ValueError, match="identity mismatch"):
        score.verify_unit_artifacts("c" * 64, unit, path, done_sha, environment, attempt)
    with pytest.raises(ValueError, match="environment"):
        score.verify_unit_artifacts(plan_sha, unit, path, done_sha, {"device": "cuda"}, attempt)


@pytest.mark.parametrize("forged_attempt", [3, 2, True])
def test_resealed_receipt_attempt_cannot_bypass_ledger(one_completed_TEST_unit, forged_attempt):
    plan_sha, unit, path, _, environment, attempt = one_completed_TEST_unit
    receipt = score.read_json(path / "unit.json")
    receipt["attempt"] = forged_attempt
    score.atomic_bytes(path / "unit.json", score.canonical(receipt))
    done = score.read_json(path / "DONE")
    done["unit_json_sha256"] = score.file_sha(path / "unit.json")
    score.atomic_bytes(path / "DONE", score.canonical(done))
    with pytest.raises(ValueError, match="receipt attempt"):
        score.verify_unit_artifacts(plan_sha, unit, path, score.file_sha(path / "DONE"), environment, attempt)


def write_TEST_ledger(path, sequence):
    payload = b"".join(score.canonical({"event": kind, "unit_id": "TEST_unit", "attempt": attempt}) + b"\n"
                       for kind, attempt in sequence)
    path.write_bytes(payload)


@pytest.mark.parametrize("sequence,message", [
    ([("unit_start", 1), ("unit_start", 2), ("unit_done", 2)], "transition"),
    ([("unit_start", 1), ("unit_failed", 1), ("unit_start", 2), ("unit_failed", 2), ("unit_start", 3)], "allowed range"),
    ([("unit_start", 1), ("unit_done", 1), ("unit_start", 2), ("unit_done", 2)], "transition"),
    ([("unit_start", 1), ("unit_failed", 1)], "terminal done"),
    ([("unit_start", 1)], "terminal done"),
    ([("unit_start", 1), ("unit_done", 2)], "number mismatch"),
    ([("unit_done", 1)], "without active start"),
    ([("unit_start", 2), ("unit_done", 2)], "number mismatch"),
])
def test_attempt_ledger_counterexamples(tmp_path, sequence, message):
    path = tmp_path / "ledger.jsonl"
    write_TEST_ledger(path, sequence)
    with pytest.raises(ValueError, match=message):
        score.replay_ledger(path, {"TEST_unit"}, "TEST_plan")


@pytest.mark.parametrize("retry", [False, True])
def test_valid_attempt_ledger_including_orphan_recovery(tmp_path, retry):
    path = tmp_path / "ledger.jsonl"
    sequence = [("unit_start", 1)]
    if retry:
        sequence.extend([("unit_failed", 1), ("unit_start", 2)])
    write_TEST_ledger(path, sequence)
    with path.open("ab") as handle:
        handle.write(score.canonical({"event": "unit_done", "unit_id": "TEST_unit",
                                      "attempt": 2 if retry else 1, "recovered_from_done": True}) + b"\n")
    counts, ledger_sha = score.replay_ledger(path, {"TEST_unit"}, "TEST_plan")
    assert counts == {"TEST_unit": 2 if retry else 1}
    assert ledger_sha == score.file_sha(path)


def test_ledger_mutation_after_verification_blocks_output(tmp_path):
    identity = {}
    for name, field in (("analysis_lock.json", "analysis_lock_sha256"),
                        ("completion.json", "completion_sha256"), ("ledger.jsonl", "ledger_sha256")):
        (tmp_path / name).write_bytes(b"TEST identity fixture")
        identity[field] = score.file_sha(tmp_path / name)
    score.verify_closure_unchanged(tmp_path, identity)
    with (tmp_path / "ledger.jsonl").open("ab") as handle:
        handle.write(b"\nchanged")
    with pytest.raises(ValueError, match="ledger.jsonl changed"):
        score.verify_closure_unchanged(tmp_path, identity)


def test_raw_TEST_predictions_pool_rotations_before_three_draws(tmp_path, monkeypatch):
    # A small helper-level fixture uses one policy, keeping the complete
    # 91-person/3-draw/6-rotation aggregation. The public CLI has no such override.
    policy = ("ridge_wavlm", 288, 12, "prompt_new")
    monkeypatch.setattr(score, "POLICIES", (policy,))
    manifest, units, hashes = {}, [], {}
    for draw in range(3):
        for rotation in range(6):
            paths, truth = [], []
            for speaker in range(91):
                for label in range(6):
                    for repeat in range(9 if rotation == 1 else 1):
                        path = f"TEST_s{speaker:03d}_p{rotation}_c{label}_r{repeat}.wav"
                        manifest[path] = {"speaker": f"s{speaker:03d}", "label_index": str(label)}
                        paths.append(path)
                        truth.append(label)
            truth = np.array(truth)
            correct = draw == 2 or (draw == 0 and rotation == 0)
            predicted = truth if correct else (truth + 1) % 6
            logits = np.eye(6, dtype=np.float32)[predicted]
            uid = hashlib.sha256(f"TEST:{draw}:{rotation}".encode()).hexdigest()
            directory = tmp_path / "units" / uid
            directory.mkdir(parents=True)
            np.savez(directory / "predictions.npz", paths=np.asarray(paths), logits=logits)
            hashes[uid] = score.file_sha(directory / "predictions.npz")
            units.append({"unit_id": uid, "model": policy[0], "B": policy[1], "S": policy[2],
                          "scenario": policy[3], "draw": draw, "fold": 0, "rotation": rotation, "test": paths})
    speakers, values = score.collect_draws({"units": units}, manifest, tmp_path, hashes)
    assert len(speakers) == 91
    np.testing.assert_allclose(values[policy][0], 100 / 14)
    np.testing.assert_allclose(values[policy][1], 0)
    np.testing.assert_allclose(values[policy][2], 100)
    np.testing.assert_allclose(score.mean_draws(values[policy]), (100 / 14 + 100) / 3)
    with pytest.raises(ValueError, match="six-rotation"):
        score.collect_draws({"units": units[:-1]}, manifest, tmp_path, hashes)

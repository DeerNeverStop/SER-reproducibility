"""Recovery protocol fixtures with real NPZ artifacts; no training or scoring.

The tiny checkpoint is an opaque synthetic blob: this tests the runner's
committed artifact hash contract, not neural checkpoint semantics (test_engine).
"""
import json

import numpy as np
import pytest

from v3.inner_validation import run


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def committed_fixture(base, *, start=True, failed=False):
    root = base / "formal"
    unit = {"fit": [f"SYNTHETIC_fit/{i}.wav" for i in range(576)], "train_seed": 2,
            "config": {"fixture_only": True}}
    rows, arrays = {}, {}
    for role in run.ROLES:
        paths = [f"SYNTHETIC_{role}/{i}.wav" for i in range(6)]
        unit[role] = paths
        rows.update({p: {"label_index": i} for i, p in enumerate(paths)})
        arrays[role + "__paths"] = np.asarray(paths)
        arrays[role + "__labels"] = np.arange(6, dtype=np.int64)
        for name in run.CHECKPOINTS:
            arrays[f"{role}__{name}__logits"] = np.zeros((6, 6), dtype=np.float64)
    unit["unit_id"] = run.digest(unit)
    plan_sha = "SYNTHETIC_PLAN_IDENTITY"
    directory = root / "units" / unit["unit_id"]
    attempt = directory / "attempts/0001"
    attempt.mkdir(parents=True)
    np.savez(attempt / "predictions.npz", **arrays)
    (attempt / "checkpoint.pt").write_bytes(b"SYNTHETIC_HASH_CONTRACT_NOT_A_REAL_MODEL")
    history = [{"epoch": e, "train_loss": 2. / e, "val_seen_loss": 1. + e / 100,
                "val_unseen_loss": .1 + (e - 4) ** 2, "optimizer_steps": 36,
                "scaler_skipped_steps": 0} for e in range(1, 16)]
    write_json(attempt / "history.json", history)
    receipt = {"schema": "ser-dual-validation-receipt-1", "unit_id": unit["unit_id"],
               "unit_sha256": run.digest(unit), "plan_sha256": plan_sha, "phase": "formal",
               "epochs_run": 15, "checkpoint_reload_verified": True, "fit_seconds": .25,
               "peak_cuda_bytes": 0, "checkpoint_reload_max_abs_diff": 0.,
               "best_seen_epoch": 1, "best_unseen_epoch": 4,
               "checkpoint_unique_epochs": [1, 4, 15], "checkpoint_unique_epoch_count": 3}
    write_json(attempt / "receipt.json", receipt)
    done = {"schema": "ser-dual-validation-done-1", "unit_id": unit["unit_id"],
            "unit_sha256": run.digest(unit), "plan_sha256": plan_sha, "phase": "formal",
            "artifacts": {f"attempts/0001/{name}": run.file_sha(attempt / name) for name in run.ARTIFACTS}}
    write_json(directory / "DONE", done)
    run.event(root, "batch_start", unit_ids=[unit["unit_id"]])
    if start:
        run.event(root, "unit_start", unit_id=unit["unit_id"])
    if failed:
        run.event(root, "unit_failed", unit_id=unit["unit_id"], error_type="SyntheticFailure")
    return root, unit, plan_sha, rows, attempt


def events(root):
    return [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines()]


def test_crash_after_done_before_ledger_recovers_with_exact_done_hash(tmp_path):
    root, unit, plan_sha, rows, attempt = committed_fixture(tmp_path)
    assert run.verify_unit(unit, root, plan_sha, rows) is not None
    assert not any(e["event"] == "unit_done" for e in events(root))
    run.reconcile_done_event(root, unit, plan_sha, rows)
    recovered = [e for e in events(root) if e["event"] == "unit_done"]
    assert len(recovered) == 1
    assert recovered[0]["unit_id"] == unit["unit_id"]
    assert recovered[0]["recovered_from_verified_done"] is True
    assert recovered[0]["done_sha256"] == run.file_sha(root / "units" / unit["unit_id"] / "DONE")
    plan = {"units": [unit], "plan_sha256": plan_sha}
    assert run.result_gate(plan, tmp_path, "formal", rows)["pass"] is True


def test_repeated_recovery_does_not_append_duplicate_completion(tmp_path):
    root, unit, plan_sha, rows, attempt = committed_fixture(tmp_path)
    run.reconcile_done_event(root, unit, plan_sha, rows)
    before = (root / "ledger.jsonl").read_bytes()
    run.reconcile_done_event(root, unit, plan_sha, rows)
    assert (root / "ledger.jsonl").read_bytes() == before
    assert sum(e["event"] == "unit_done" for e in events(root)) == 1


def test_missing_start_or_prior_failure_cannot_be_recovered(tmp_path):
    for name, start, failed in (("no_start", False, False), ("prior_failure", True, True)):
        root, unit, plan_sha, rows, attempt = committed_fixture(tmp_path / name, start=start, failed=failed)
        before = (root / "ledger.jsonl").read_bytes()
        with pytest.raises(ValueError, match="unique nonfailed start"):
            run.reconcile_done_event(root, unit, plan_sha, rows)
        assert (root / "ledger.jsonl").read_bytes() == before
        assert not any(e["event"] == "unit_done" for e in events(root))


def test_corrupt_artifact_or_rehashed_bad_npz_cannot_gain_recovered_event(tmp_path):
    for mode in ("corrupt_bytes", "rehashed_wrong_order"):
        root, unit, plan_sha, rows, attempt = committed_fixture(tmp_path / mode)
        predictions = attempt / "predictions.npz"
        if mode == "corrupt_bytes":
            with predictions.open("ab") as handle:
                handle.write(b"SYNTHETIC_CORRUPTION")
            error = "artifact missing/corrupt"
        else:
            with np.load(predictions, allow_pickle=False) as saved:
                changed = {key: saved[key].copy() for key in saved.files}
            changed["val_seen__paths"] = changed["val_seen__paths"][::-1]
            np.savez(predictions, **changed)
            done_path = root / "units" / unit["unit_id"] / "DONE"
            done = run.read(done_path)
            done["artifacts"]["attempts/0001/predictions.npz"] = run.file_sha(predictions)
            write_json(done_path, done)
            error = "prediction order differs"
        before = (root / "ledger.jsonl").read_bytes()
        with pytest.raises(ValueError, match=error):
            run.reconcile_done_event(root, unit, plan_sha, rows)
        assert (root / "ledger.jsonl").read_bytes() == before
        assert not any(e["event"] == "unit_done" for e in events(root))

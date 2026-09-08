from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from v2_1.n14r import make_pins, run as n14r_run
from v2_1.n14r.spec import canonical_json


STUDY_ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = STUDY_ROOT / "plan"


@pytest.fixture(scope="module")
def frozen_plan():
    """Load the real, generated 2,240-unit plan rather than a hand-made stub."""
    return n14r_run.load_plan(PLAN_DIR)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_old_train_unit(
    row: dict[str, str], cfg: dict, run_dir: Path,
) -> Path:
    """Write the nested four-file artifact emitted by ``ser_v2.train``."""
    unit_dir = run_dir / "units" / row["unit_id"]
    unit_dir.mkdir(parents=True)
    predictions = (
        "sample_index,relative_path,speaker,y_true,y_pred,"
        "logit_0,logit_1,logit_2,logit_3,logit_4,logit_5\n"
        "0,a.wav,1001,0,0,1.0,0.0,0.0,0.0,0.0,0.0\n"
    ).encode("utf-8")
    prediction_sha = _sha(predictions)
    (unit_dir / "predictions.csv").write_bytes(predictions)
    (unit_dir / "history.json").write_text(
        json.dumps([{"epoch": 1, "val_loss": 1.0, "val_uar": 16.7}]) + "\n",
        encoding="utf-8",
    )
    unit = {
        "unit_id": row["unit_id"],
        "status": "done",
        "config": cfg,
        "predictions_sha256": prediction_sha,
        "best_epoch": 1,
        "epochs_run": 1,
        "val_loss_best": 1.0,
        "val_uar_best": 16.7,
    }
    for field in n14r_run.IDENTITY_FIELDS:
        value = row[field]
        if field in n14r_run.INTEGER_IDENTITY_FIELDS:
            value = int(value)
        unit[field] = value
    (unit_dir / "unit.json").write_text(
        json.dumps(unit, sort_keys=True) + "\n", encoding="utf-8",
    )
    (unit_dir / "DONE").write_text(prediction_sha + "\n", encoding="ascii", newline="")
    return unit_dir


def _json_writer(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_lock_run_dir(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "execution_environment.json").write_text(
        json.dumps({"external_feature_cache": {"kind": "test-fixture", "sha256": "f" * 64}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "ledger.jsonl").write_text("", encoding="utf-8")
    audit = run_dir / "n14r_execution_ledger.jsonl"
    audit.write_text("", encoding="utf-8")
    return audit


def test_load_plan_accepts_real_generated_plan(frozen_plan):
    rows, configs = frozen_plan
    assert len(rows) == 2240
    assert len(configs) == 8
    assert len({row["unit_id"] for row in rows}) == 2240
    assert {int(row["draw_id"]) for row in rows} == set(range(28))
    assert all(int(row["draw_id"]) == int(row["r"]) for row in rows)
    assert all(configs[row["config_sha256"]]["hpo_config_index"] == int(row["config_index"])
               for row in rows)


def test_pins_cover_every_local_module_imported_by_the_frozen_engine() -> None:
    required = {
        "v2_1/__init__.py", "advanced_experiment_utils.py", "advanced_models.py",
        "dataset.py", "evaluate.py", "experiment_utils.py", "features.py",
        "fno_data.py", "fno_model.py", "tuned_standard_experiment.py",
        "v2/ser_v2/__init__.py",
        "v2/ser_v2/common.py", "v2/ser_v2/corpora.py", "v2/ser_v2/features.py",
        "v2/ser_v2/stats.py", "v2/ser_v2/train.py",
    }
    assert required.issubset(set(make_pins.REUSED_FILES))


def test_preflight_accepts_nested_old_train_artifact_and_checks_identity(tmp_path: Path, frozen_plan):
    rows, configs = frozen_plan
    row = {**rows[0], "n_fit": "2", "n_val": "1", "n_test": "1"}
    unit_dir = _write_old_train_unit(row, configs[row["config_sha256"]], tmp_path)
    assert n14r_run.preflight_completed_unit(row, tmp_path, configs) == "done"

    unit = json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))
    unit["train_seed"] += 1
    (unit_dir / "unit.json").write_text(json.dumps(unit) + "\n", encoding="utf-8")
    with pytest.raises(n14r_run.IntegrityError, match="train_seed"):
        n14r_run.preflight_completed_unit(row, tmp_path, configs)


def test_preflight_rejects_corrupt_done_hash(tmp_path: Path, frozen_plan):
    rows, configs = frozen_plan
    row = {**rows[1], "n_fit": "2", "n_val": "1", "n_test": "1"}
    unit_dir = _write_old_train_unit(row, configs[row["config_sha256"]], tmp_path)
    (unit_dir / "DONE").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(n14r_run.IntegrityError, match="prediction/DONE/unit hash mismatch"):
        n14r_run.preflight_completed_unit(row, tmp_path, configs)


def test_preflight_requires_exact_done_receipt_and_valid_history(tmp_path: Path, frozen_plan):
    rows, configs = frozen_plan
    row = {**rows[2], "n_fit": "2", "n_val": "1", "n_test": "1"}
    unit_dir = _write_old_train_unit(row, configs[row["config_sha256"]], tmp_path)
    (unit_dir / "DONE").write_text(
        (unit_dir / "DONE").read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="",
    )
    with pytest.raises(n14r_run.IntegrityError, match="prediction/DONE/unit hash mismatch"):
        n14r_run.preflight_completed_unit(row, tmp_path, configs)

    unit_dir = tmp_path / "other" / "units" / row["unit_id"]
    _write_old_train_unit(row, configs[row["config_sha256"]], tmp_path / "other")
    (unit_dir / "history.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(n14r_run.IntegrityError, match="non-empty"):
        n14r_run.preflight_completed_unit(row, tmp_path / "other", configs)


def test_preflight_parses_prediction_structure_after_hashes_match(tmp_path: Path, frozen_plan):
    rows, configs = frozen_plan
    row = {**rows[3], "n_fit": "2", "n_val": "1", "n_test": "1"}
    unit_dir = _write_old_train_unit(row, configs[row["config_sha256"]], tmp_path)
    broken = (unit_dir / "predictions.csv").read_text(encoding="utf-8").replace(
        ",0,1.0,0.0,0.0,0.0,0.0,0.0", ",1,1.0,0.0,0.0,0.0,0.0,0.0",
    )
    (unit_dir / "predictions.csv").write_text(broken, encoding="utf-8", newline="")
    sha = _sha((unit_dir / "predictions.csv").read_bytes())
    unit = json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))
    unit["predictions_sha256"] = sha
    (unit_dir / "unit.json").write_text(json.dumps(unit) + "\n", encoding="utf-8")
    (unit_dir / "DONE").write_text(sha + "\n", encoding="ascii", newline="")
    with pytest.raises(n14r_run.IntegrityError, match="argmax"):
        n14r_run.preflight_completed_unit(row, tmp_path, configs)


def test_second_failure_permanently_exhausts_even_a_completed_unit(frozen_plan):
    rows, _ = frozen_plan
    draw = [row for row in rows if row["draw_id"] == "0"]
    uid = draw[0]["unit_id"]
    assert n14r_run._exhausted_unit_ids(draw, {uid: 1}) == []
    assert n14r_run._exhausted_unit_ids(draw, {uid: 2}) == [uid]


def test_orphan_start_is_closed_as_failure_or_recovered_success(tmp_path: Path, frozen_plan):
    rows, configs = frozen_plan

    class FakePlanIO:
        manifest = {"cremad": {"a.wav": {"sample_index": 0, "speaker": "1001", "label_index": 0}}}

        @staticmethod
        def fold(_row):
            return {"fit": ["f1", "f2"], "val": ["v1"], "test": ["a.wav"]}

    failed_row = {**rows[4], "n_fit": "2", "n_val": "1", "n_test": "1"}
    run_failed = tmp_path / "failed"
    run_failed.mkdir()
    (run_failed / "ledger.jsonl").write_text(
        json.dumps({"unit_id": failed_row["unit_id"], "event": "start"}) + "\n",
        encoding="utf-8", newline="",
    )
    n14r_run._reconcile_orphan_attempts([failed_row], run_failed, configs, FakePlanIO())
    events = [json.loads(line) for line in (run_failed / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["start", "failed"]
    assert events[-1]["recovered_after_interruption"] is True

    done_row = {**rows[5], "n_fit": "2", "n_val": "1", "n_test": "1"}
    run_done = tmp_path / "done"
    run_done.mkdir()
    (run_done / "ledger.jsonl").write_text(
        json.dumps({"unit_id": done_row["unit_id"], "event": "start"}) + "\n",
        encoding="utf-8", newline="",
    )
    _write_old_train_unit(done_row, configs[done_row["config_sha256"]], run_done)
    n14r_run._reconcile_orphan_attempts([done_row], run_done, configs, FakePlanIO())
    events = [json.loads(line) for line in (run_done / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["start", "done"]
    assert events[-1]["recovered_after_interruption"] is True


def test_pins_verification_rebuilds_inventory_and_rejects_extra_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    study = repo / "v2_1" / "n14r"
    features = tmp_path / "features"
    study.mkdir(parents=True)
    features.mkdir()
    (study / "contract.txt").write_text("frozen\n", encoding="utf-8")
    (features / "cremad__logmel__one.npz").write_bytes(b"npz")
    (features / "cremad__logmel__one.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(make_pins, "REUSED_FILES", ())
    record = make_pins.build_record(repo, features)
    assert make_pins.verify_record(record, repo, features) == []

    evidence = study / "evidence" / "postrun-verification.json"
    evidence.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")
    assert make_pins.verify_record(record, repo, features) == []

    removed = json.loads(json.dumps(record))
    removed["files"].pop("v2_1/n14r/contract.txt")
    payload = {key: value for key, value in removed.items() if key != "content_manifest_sha256"}
    removed["content_manifest_sha256"] = _sha(make_pins.canonical_json(payload).encode("utf-8"))
    assert "files" in make_pins.verify_record(removed, repo, features)

    (features / "cremad__logmel__two.npz").write_bytes(b"npz2")
    (features / "cremad__logmel__two.json").write_text("{}\n", encoding="utf-8")
    failures = make_pins.verify_record(record, repo, features)
    assert failures and failures[0].startswith("inventory_rebuild:RuntimeError")


def test_analysis_lock_hashes_24_by_80_units_and_self_hash(
    tmp_path: Path, frozen_plan, monkeypatch: pytest.MonkeyPatch,
):
    rows, _ = frozen_plan

    # The lock function is outcome-blind.  A deterministic path hasher lets the
    # test exercise all 1,920 x 4 entries without manufacturing 7,680 large files.
    def path_hash(path: Path) -> str:
        return _sha(path.as_posix().encode("utf-8"))

    monkeypatch.setattr(n14r_run, "_sha256_file", path_hash)
    monkeypatch.setattr(n14r_run, "_write_json_atomic", _json_writer)
    run_dir = tmp_path / "run"
    audit = _prepare_lock_run_dir(run_dir)
    lock = n14r_run.create_analysis_lock(
        PLAN_DIR, run_dir, rows, list(range(24)), "env-fingerprint", audit,
    )

    assert lock["complete_draw_ids"] == list(range(24))
    assert lock["n_complete_draws"] == 24
    assert lock["n_locked_units"] == 24 * 80
    assert len(lock["unit_artifact_hashes"]) == 24 * 80
    assert all(set(hashes) == {"unit.json", "history.json", "predictions.csv", "DONE"}
               for hashes in lock["unit_artifact_hashes"].values())
    without_self = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    assert lock["lock_payload_sha256"] == _sha(canonical_json(without_self).encode("utf-8"))
    on_disk = json.loads((run_dir / "analysis_lock.json").read_text(encoding="utf-8"))
    assert on_disk == lock


def test_existing_analysis_lock_cannot_silently_drift(
    tmp_path: Path, frozen_plan, monkeypatch: pytest.MonkeyPatch,
):
    rows, _ = frozen_plan
    monkeypatch.setattr(
        n14r_run, "_sha256_file", lambda path: _sha(path.as_posix().encode("utf-8")),
    )
    monkeypatch.setattr(n14r_run, "_write_json_atomic", _json_writer)
    run_dir = tmp_path / "run"
    audit = _prepare_lock_run_dir(run_dir)
    n14r_run.create_analysis_lock(PLAN_DIR, run_dir, rows, list(range(24)), "environment-A", audit)

    with pytest.raises(n14r_run.IntegrityError, match="environment_fingerprint"):
        n14r_run.create_analysis_lock(
            PLAN_DIR, run_dir, rows, list(range(24)), "environment-B", audit,
        )


def test_runner_lock_is_mutually_exclusive_and_released(tmp_path: Path):
    run_dir = tmp_path / "run"
    with n14r_run.exclusive_runner_lock(run_dir):
        assert (run_dir / ".n14r-run.lock").exists()
        with pytest.raises(n14r_run.IntegrityError, match="runner lock exists"):
            with n14r_run.exclusive_runner_lock(run_dir):
                pass
    assert not (run_dir / ".n14r-run.lock").exists()
    with n14r_run.exclusive_runner_lock(run_dir):
        assert (run_dir / ".n14r-run.lock").exists()

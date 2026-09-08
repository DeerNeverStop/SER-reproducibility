"""Mutation tests for the implementation-independent N14R verifier."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from v2_1.n14r.verify import (
    BASE_COMMIT,
    BASE_TAG_COMMITS,
    HPO_GRID,
    PIN_EXCLUSIONS,
    REUSED_PINNED_FILES,
    VerificationError,
    _apply_cremad_hygiene,
    _expected_analysis_draw_ids,
    _load_manifest,
    _refs,
    _replay_attempt_ledger,
    _select_episode,
    _unit_candidate,
    _validate_hygiene,
    _validate_pins,
    compare_results,
    macro_uar_6,
    sha256_file,
    sha256_json,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_reserve_void_uses_next_nonvoid_reserve() -> None:
    assert _expected_analysis_draw_ids({0, 24}) == [*range(1, 24), 25]


def test_attempt_ledger_replay_enforces_two_closed_attempts() -> None:
    rows = [{"unit_id": "u0"}, {"unit_id": "u1"}]
    replay = _replay_attempt_ledger([
        {"unit_id": "u0", "event": "start"},
        {"unit_id": "u0", "event": "failed"},
        {"unit_id": "u0", "event": "start"},
        {"unit_id": "u0", "event": "done", "predictions_sha256": "a" * 64},
    ], rows)
    assert replay["u0"] == {
        "state": "done", "starts": 2, "done": 1, "failed": 1,
        "done_receipt": "a" * 64,
    }
    with pytest.raises(VerificationError, match="out-of-sequence start"):
        _replay_attempt_ledger([
            {"unit_id": "u0", "event": "start"},
            {"unit_id": "u0", "event": "start"},
        ], rows)
    with pytest.raises(VerificationError, match="third training attempt"):
        _replay_attempt_ledger([
            {"unit_id": "u0", "event": "start"},
            {"unit_id": "u0", "event": "failed"},
            {"unit_id": "u0", "event": "start"},
            {"unit_id": "u0", "event": "failed"},
            {"unit_id": "u0", "event": "start"},
        ], rows)
    with pytest.raises(VerificationError, match="unterminated training attempt"):
        _replay_attempt_ledger([{"unit_id": "u1", "event": "start"}], rows)


def _receipt(unit_dir: Path) -> dict[str, str]:
    return {name: sha256_file(unit_dir / name)
            for name in ("unit.json", "history.json", "predictions.csv", "DONE")}


def _rewrite_receipt(unit_dir: Path, unit: dict[str, object]) -> dict[str, str]:
    pred_sha = sha256_file(unit_dir / "predictions.csv")
    unit["predictions_sha256"] = pred_sha
    _write_json(unit_dir / "unit.json", unit)
    (unit_dir / "DONE").write_text(pred_sha + "\n", encoding="utf-8")
    return _receipt(unit_dir)


@pytest.fixture()
def atomic_unit(tmp_path: Path):
    unit_id = "u" * 64
    unit_dir = tmp_path / "units" / unit_id
    unit_dir.mkdir(parents=True)
    refs = [
        {"sample_index": i, "relative_path": f"audio/{i}.wav", "speaker": f"S{i}",
         "label_index": i}
        for i in range(6)
    ]
    pred_path = unit_dir / "predictions.csv"
    fields = ["sample_index", "relative_path", "speaker", "y_true", "y_pred"] + [f"logit_{i}" for i in range(6)]
    with pred_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for ref in refs:
            label = ref["label_index"]
            writer.writerow({"sample_index": ref["sample_index"], "relative_path": ref["relative_path"],
                             "speaker": ref["speaker"], "y_true": label, "y_pred": label,
                             **{f"logit_{i}": 1.0 if i == label else 0.0 for i in range(6)}})
    history = [{"epoch": 0, "val_loss": 1.0, "val_uar": 12.0},
               {"epoch": 1, "val_loss": 0.5, "val_uar": 34.0}]
    _write_json(unit_dir / "history.json", history)
    row = {"unit_id": unit_id, "arm": "N14R", "corpus_level": "cremad", "model": "resnet_se",
           "cell": "GR_hpo", "fold": 0, "r": 0, "seed_index": 0, "train_seed": 123,
           "config_sha256": "c" * 64, "split_sha256": "s" * 64, "config_index": 0,
           "n_fit": 12, "n_val": 6, "n_test": 6}
    unit: dict[str, object] = {**row, "status": "done", "config": HPO_GRID[0],
                              "best_epoch": 1, "val_loss_best": 0.5, "val_uar_best": 34.0}
    receipt = _rewrite_receipt(unit_dir, unit)
    return tmp_path, row, {"test": refs}, unit, receipt


def test_six_class_uar_and_lowest_index_tie_break() -> None:
    assert macro_uar_6(range(6), range(6)) == 100.0
    candidates = [{"config_index": i, "val_uar": 50.0 if i in (2, 5) else 40.0,
                   "test_uar": 0.0, "unit_id": str(i)} for i in range(8)]
    assert _select_episode(candidates)["config_index"] == 2


def test_mutation_missing_config_is_rejected() -> None:
    candidates = [{"config_index": i, "val_uar": float(i), "test_uar": 0.0}
                  for i in range(8) if i != 6]
    with pytest.raises(VerificationError, match="0..7"):
        _select_episode(candidates)


def test_atomic_unit_baseline(atomic_unit) -> None:
    run, row, scope, _, receipt = atomic_unit
    candidate = _unit_candidate(row, run, scope, receipt)
    assert candidate["val_uar"] == 34.0
    assert candidate["test_uar"] == 100.0


def test_mutation_prediction_hash_is_rejected(atomic_unit) -> None:
    run, row, scope, _, receipt = atomic_unit
    pred = run / "units" / row["unit_id"] / "predictions.csv"
    pred.write_text(pred.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VerificationError, match="artifact hash"):
        _unit_candidate(row, run, scope, receipt)


def test_mutation_missing_test_row_is_rejected_even_when_resealed(atomic_unit) -> None:
    run, row, scope, unit, _ = atomic_unit
    unit_dir = run / "units" / row["unit_id"]
    pred = unit_dir / "predictions.csv"
    lines = pred.read_text(encoding="utf-8").splitlines()
    pred.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    receipt = _rewrite_receipt(unit_dir, unit)
    with pytest.raises(VerificationError, match="row count"):
        _unit_candidate(row, run, scope, receipt)


def test_mutation_train_seed_is_rejected_even_when_resealed(atomic_unit) -> None:
    run, row, scope, unit, _ = atomic_unit
    unit_dir = run / "units" / row["unit_id"]
    unit["train_seed"] = 124
    receipt = _rewrite_receipt(unit_dir, unit)
    with pytest.raises(VerificationError, match="train_seed"):
        _unit_candidate(row, run, scope, receipt)


@pytest.mark.parametrize(
    ("field", "value", "expected_path"),
    [("values", [9.0, 2.0], "$.primary.values[0]"),
     ("p_two_sided", 0.0001, "$.primary.p_two_sided"),
     ("verdict", "supported", "$.primary.verdict")],
)
def test_mutation_scored_draw_p_or_verdict_is_detected(field, value, expected_path) -> None:
    expected = {"primary": {"values": [1.0, 2.0], "p_two_sided": 0.4, "verdict": "not supported"}}
    actual = json.loads(json.dumps(expected))
    actual["primary"][field] = value
    differences = compare_results(expected, actual)
    assert differences
    assert differences[0]["path"] == expected_path


def test_extra_scorer_field_is_not_silently_ignored() -> None:
    differences = compare_results({"tested": True}, {"tested": True, "invented": 1})
    assert differences == [{"path": "$.invented", "expected": "<absent>", "actual": 1}]


def test_external_cache_receipts_are_portable_but_pinned_sources_are_not(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    study = tmp_path / "v2_1" / "n14r"
    study.mkdir(parents=True)
    source = study / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    reused = []
    for rel in REUSED_PINNED_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {rel}\n", encoding="utf-8")
        reused.append(path)
    external = [
        {"name": "cremad__logmel__x.npz", "sha256": "a" * 64, "size_bytes": 100},
        {"name": "cremad__logmel__x.json", "sha256": "b" * 64, "size_bytes": 200},
    ]
    files = {
        path.relative_to(tmp_path).as_posix(): sha256_file(path)
        for path in [source, *reused]
    }
    pins = {
        "schema": "ser26-n14r-pins-1", "base_commit": BASE_COMMIT,
        "base_tag_commits": BASE_TAG_COMMITS, "files": files,
        "external_feature_cache": external, "exclusions": PIN_EXCLUSIONS,
    }
    pins["content_manifest_sha256"] = sha256_json(pins)
    pins_path = study / "PINS.json"
    _write_json(pins_path, pins)
    runtime = [{"basename": item["name"], "path": str(tmp_path / "missing" / item["name"]),
                "sha256": item["sha256"], "size_bytes": item["size_bytes"]} for item in external]
    lock = {"external_feature_cache": runtime}
    environment = {"external_feature_cache": runtime}
    assert _validate_pins(pins_path, lock, environment) is False
    (tmp_path / ".git").rmdir()
    assert _validate_pins(pins_path, lock, environment) is False

    windows_runtime = [
        {**item, "path": f"E:\\frozen-cache\\{item['basename']}"}
        for item in runtime
    ]
    assert _validate_pins(
        pins_path, {"external_feature_cache": windows_runtime}, environment
    ) is False

    bad_runtime = json.loads(json.dumps(runtime))
    bad_runtime[0]["basename"] = "substituted.npz"
    with pytest.raises(VerificationError, match="path/basename"):
        _validate_pins(pins_path, {"external_feature_cache": bad_runtime}, environment)

    duplicate_runtime = [runtime[0], dict(runtime[0])]
    with pytest.raises(VerificationError, match="duplicated or incomplete"):
        _validate_pins(
            pins_path, {"external_feature_cache": duplicate_runtime}, environment
        )

    omitted = json.loads(json.dumps(pins))
    omitted["files"].pop(source.relative_to(tmp_path).as_posix())
    omitted.pop("content_manifest_sha256")
    omitted["content_manifest_sha256"] = sha256_json(omitted)
    _write_json(pins_path, omitted)
    with pytest.raises(VerificationError, match="source inventory"):
        _validate_pins(pins_path, lock, environment)
    _write_json(pins_path, pins)

    source.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(VerificationError, match="pinned source hash"):
        _validate_pins(pins_path, lock, environment)


def _manifest_row(path: str, digest: str, label: int, index: int) -> dict[str, object]:
    return {"sample_index": index, "relative_path": path, "speaker": f"S{index:03d}",
            "label_index": label, "sha256": digest}


def test_independent_h1_h2_h3_replay_sorts_and_reindexes() -> None:
    rows = [
        _manifest_row("z_conflict.wav", "a" * 64, 0, 0),
        _manifest_row("y_conflict.wav", "a" * 64, 1, 1),
        _manifest_row("b_same.wav", "b" * 64, 2, 2),
        _manifest_row("a_same.wav", "b" * 64, 2, 3),
        _manifest_row("folder/1076_MTI_SAD_XX.wav", "c" * 64, 5, 4),
        _manifest_row("m_keep.wav", "d" * 64, 3, 5),
    ]
    kept, hygiene = _apply_cremad_hygiene(rows)
    assert [row["relative_path"] for row in kept] == ["a_same.wav", "m_keep.wav"]
    assert [row["sample_index"] for row in kept] == [0, 1]
    reasons = {item["relative_path"]: item["reason"] for item in hygiene["dropped"]}
    assert reasons["z_conflict.wav"].startswith("H1 ")
    assert reasons["y_conflict.wav"].startswith("H1 ")
    assert reasons["b_same.wav"].startswith("H2 ")
    assert reasons["folder/1076_MTI_SAD_XX.wav"] == "H3 registered unusable file"


def test_raw_manifest_hash_precedes_hygiene_and_split_refs_use_new_indices(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    rows = [_manifest_row(f"safe_{label}.wav", f"{label + 1:064x}", label, label)
            for label in range(6)]
    rows += [
        _manifest_row("h1_a.wav", "e" * 64, 0, 6),
        _manifest_row("h1_b.wav", "e" * 64, 1, 7),
        _manifest_row("1076_MTI_SAD_XX.wav", "f" * 64, 5, 8),
    ]
    manifest_path = plan_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    raw_sha = sha256_file(manifest_path)
    kept, located, hygiene = _load_manifest(plan_dir, raw_sha)
    assert located == manifest_path
    assert hygiene["n_input"] == 9 and hygiene["n_kept"] == 6
    assert [row["sample_index"] for row in kept] == list(range(6))
    by_path = {row["relative_path"]: row for row in kept}
    refs = _refs([row["relative_path"] for row in kept], by_path)
    assert [ref["sample_index"] for ref in refs] == list(range(6))
    with pytest.raises(VerificationError, match="split path absent"):
        _refs(["h1_a.wav"], by_path)
    with pytest.raises(VerificationError, match="byte-identical"):
        _load_manifest(plan_dir, "0" * 64)


def test_hygiene_file_exact_hash_and_semantics_are_both_enforced(tmp_path: Path) -> None:
    dropped = [
        {"relative_path": path, "reason": "H3 registered unusable file" if i == 6
         else f"H1 label-conflicting duplicate group {i:012x}"}
        for i, path in enumerate([
            "1006_TIE_HAP_XX.wav", "1006_TIE_NEU_XX.wav", "1013_WSI_DIS_XX.wav",
            "1013_WSI_SAD_XX.wav", "1017_IWW_ANG_XX.wav", "1017_IWW_FEA_XX.wav",
            "1076_MTI_SAD_XX.wav",
        ])
    ]
    computed = {"corpus": "cremad", "n_input": 7442, "n_kept": 7435,
                "n_dropped": 7, "dropped": dropped}
    manifest_sha = "a" * 64
    hygiene_path = tmp_path / "hygiene.json"
    document = {"schema": "ser26-n14r-hygiene-1", "manifest_sha256": manifest_sha,
                **computed}
    _write_json(hygiene_path, document)
    _write_json(tmp_path / "plan_summary.json", {
        "hygiene_sha256": sha256_file(hygiene_path), "manifest_input_rows": 7442,
        "analysis_population_rows": 7435, "manifest_sha256": manifest_sha,
    })
    assert _validate_hygiene(tmp_path, manifest_sha, computed) == sha256_file(hygiene_path)

    hygiene_path.write_text(hygiene_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VerificationError, match="exact-byte hash"):
        _validate_hygiene(tmp_path, manifest_sha, computed)
    _write_json(hygiene_path, document)

    # Re-seal the superficial file hash after changing a deletion: the
    # independent H1/H2/H3 reconstruction must still reject it.
    document["dropped"] = document["dropped"][:-1]
    document["n_dropped"] = 6
    _write_json(hygiene_path, document)
    summary = _json_for_test(tmp_path / "plan_summary.json")
    summary["hygiene_sha256"] = sha256_file(hygiene_path)
    _write_json(tmp_path / "plan_summary.json", summary)
    with pytest.raises(VerificationError, match="independent H1/H2/H3"):
        _validate_hygiene(tmp_path, manifest_sha, computed)


def _json_for_test(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

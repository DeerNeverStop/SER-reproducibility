"""Synthetic fixtures only: no Qwen import, download, GPU call, or real TTS."""
from collections import Counter
from copy import deepcopy
import csv
from pathlib import Path

import numpy as np
import pytest

from v3.speaker_coverage.tts import probe as p


def synthetic_plan():
    codes = sorted(p.TEXTS)
    unit = {"model": "cnn", "policy": "U", "fold": 0, "rotation": 0, "draw": 0,
            "fit": [f"1001_{x}_NEU_XX.wav" for x in codes[:8]],
            "val": [f"1002_{x}_NEU_XX.wav" for x in codes[:6]],
            "test": [f"1003_{x}_NEU_XX.wav" for x in codes[8:10]]}
    unit["unit_id"] = p.digest(unit)
    plan = {"schema": "ser-speaker-coverage-1", "design": {"fixture": True}, "units": [unit]}
    plan["plan_sha256"] = p.digest(plan)
    return plan


@pytest.fixture
def frozen(tmp_path):
    model = tmp_path / "fake_model_NOT_REAL_WEIGHTS"
    model.mkdir()
    (model / "config.json").write_text('{"synthetic_fixture": true}', encoding="utf-8")
    (model / ".cache").mkdir()
    (model / ".cache" / "irrelevant.txt").write_text("synthetic download metadata", encoding="utf-8")
    pins = {"model_id": p.MODEL_ID, "revision": "1" * 40, "package": p.PACKAGE,
            "prepared_only": True, "files": {"config.json": {
                "bytes": (model / "config.json").stat().st_size, "sha256": p.file_sha(model / "config.json")}}}
    pin_path = tmp_path / "pins.json"
    p.immutable_json(pin_path, pins)
    plan_path = tmp_path / "synthetic_plan.json"
    p.immutable_json(plan_path, synthetic_plan())
    out = tmp_path / "probe_synthetic_only"
    manifest = p.freeze(plan_path, model, pin_path, out)
    return manifest, out, model, pin_path, plan_path


def test_manifest_exact_cells_roles_inputs_and_determinism(frozen):
    manifest, out, model, pins, plan = frozen
    attempts = manifest["attempts"]
    assert len(attempts) == len({a["seed"] for a in attempts}) == 100
    assert Counter(a["role"] for a in attempts) == {"probe": 96, "anchor": 4}
    assert Counter(v["sex_description"] for v in manifest["voices"]) == {"female": 2, "male": 2}
    assert Counter((a["voice_id"], a["emotion"]) for a in attempts if a["role"] == "probe") == {
        (v["voice_id"], emotion): 4 for v in p.VOICES for emotion in p.EMOTIONS}
    assert {a["prompt"] for a in attempts}.isdisjoint(manifest["context"]["query_prompts"])
    assert all(a["conditioning_audio"] is None and a["generation"]["max_new_tokens"] == 512 for a in attempts)
    assert all((out / "attempts" / a["attempt_id"] / "config.json").is_file() for a in attempts)
    assert not list(out.rglob("*.wav"))
    assert p.freeze(plan, model, pins, out) == manifest
    assert p.load_manifest(out / "manifest.json") == manifest
    changed = deepcopy(manifest["model"])
    changed["revision"] = "2" * 40
    assert p.build_manifest(synthetic_plan(), p.file_sha(plan), changed, p.file_sha(p.__file__))["manifest_sha256"] != manifest["manifest_sha256"]
    with pytest.raises(ValueError, match="synthetic fixture"):
        p.generate(out / "manifest.json", model, pins)


def test_fail_closed_for_changed_snapshot_plan_source_and_query_leak(frozen):
    manifest, out, model, pins, _ = frozen
    (model / "extra.bin").write_bytes(b"UNPINNED SYNTHETIC")
    with pytest.raises(ValueError, match="inventory"):
        p.verify_model(model, pins)
    (model / "extra.bin").unlink()
    (model / "config.json").write_bytes(b"CHANGED SYNTHETIC")
    with pytest.raises(ValueError, match="bytes"):
        p.verify_model(model, pins)
    plan = synthetic_plan()
    plan["units"][0]["test"][0] = plan["units"][0]["fit"][0]
    with pytest.raises(ValueError, match="plan hash"):
        p.prompt_context(plan)
    unit = plan["units"][0]
    unit["unit_id"] = p.digest({k: v for k, v in unit.items() if k != "unit_id"})
    plan["plan_sha256"] = p.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="prompt roles"):
        p.prompt_context(plan)
    manifest["source_sha256"] = "0" * 64
    manifest["manifest_sha256"] = p.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    alternate = out / "changed_source_manifest.json"
    p.immutable_json(alternate, manifest)
    with pytest.raises(ValueError, match="source changed"):
        p.load_manifest(alternate)


class FakeBackend:
    environment = {"backend": "SYNTHETIC_TEST_FIXTURE_NO_TTS"}

    def __init__(self):
        self.calls = []
        self.seeds = []

    def start_attempt(self, seed):
        self.seeds.append(seed)

    def generate(self, attempt):
        self.calls.append(attempt["attempt_id"])
        if attempt["attempt_id"] == "a002":
            raise RuntimeError("SYNTHETIC injected generation failure")
        if attempt["attempt_id"] == "a003":
            return np.array([np.nan], dtype=np.float32), 24000
        return np.array([0, 0.25, -0.25, 1.01], dtype=np.float32), 24000

    def memory(self):
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "fixture": True}


def test_resume_never_retries_success_failure_or_interrupted_and_checks_wav(frozen):
    manifest, out, *_ = frozen
    pending = manifest["attempts"][3]
    p.immutable_json(out / "attempts" / pending["attempt_id"] / "started.json", {
        "started_utc": "synthetic_previous_start", "manifest_sha256": manifest["manifest_sha256"],
        "attempt_sha256": pending["attempt_sha256"]})
    backend = FakeBackend()
    receipts = p.run_attempts(manifest, out, backend)
    assert len(backend.calls) == len(set(backend.calls)) == 99
    assert "a004" not in backend.calls
    assert Counter(r["status"] for r in receipts) == {"success": 97, "failed": 2, "interrupted": 1}
    assert receipts[0]["metrics"]["clip_fraction_abs_ge_0_999"] == 0.25
    assert receipts[2]["metrics"]["finite"] is False
    assert receipts[2]["metrics"]["rms"] is None
    assert receipts[1]["wav_file"] is None
    second = FakeBackend()
    assert p.run_attempts(manifest, out, second) == receipts
    assert second.calls == second.seeds == []
    (out / "attempts" / "a001" / "audio.wav").write_bytes(b"SYNTHETIC TAMPER")
    with pytest.raises(ValueError, match="WAV changed"):
        p.run_attempts(manifest, out, FakeBackend())


def test_blind_export_is_unrated_private_key_separate_and_no_overwrite(frozen):
    manifest, out, *_ = frozen
    with pytest.raises(ValueError, match="not terminal"):
        p.blind(out / "manifest.json")
    p.run_attempts(manifest, out, FakeBackend())
    summary = p.blind(out / "manifest.json")
    assert summary["raters_completed"] == 0
    assert summary["success_clips_per_rater"] == 98
    assert summary["omitted_attempts"] == ["a002", "a003"]
    assert summary["missing_anchors"] == []
    orders = []
    key = p.read_json(out / "blind_private_key.json")
    assert len(key["items"]) == 98 * 3
    for rater in range(1, 4):
        folder = out / "blind" / f"rater_{rater}"
        with (folder / "quality_ratings.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        orders.append([r["clip_id"] for r in rows])
        assert len(rows) == 98
        for row in rows:
            assert row["clip_id"].startswith("clip_")
            assert (folder / row["audio_file"]).exists()
            assert not {"voice_id", "intended_emotion", "attempt_id"}.intersection(row)
            assert all(v == "" for k, v in row.items() if k not in ("clip_id", "audio_file"))
        with (folder / "identity_ratings.csv").open(encoding="utf-8", newline="") as handle:
            identities = list(csv.DictReader(handle))
        assert len(identities) == 94
        assert all(len(r["reference_clip_ids"].split(";")) == 4 for r in identities)
        assert all(r["best_matching_reference_id_or_NONE_UNCLEAR"] == "" for r in identities)
    assert len({tuple(o) for o in orders}) == 3
    with pytest.raises(ValueError, match="already exists"):
        p.blind(out / "manifest.json")


@pytest.mark.parametrize("value,rate,writable", [([], 24000, False), ([[.2]], 24000, False),
    ([np.inf], 24000, False), ([0, 0], 24000, True), ([.1], 0, False), ([.1], True, False)])
def test_metrics_do_not_repair_or_filter_audio(value, rate, writable):
    _, metrics = p.waveform_metrics(value, rate)
    assert metrics["technically_writable"] is writable
    p.canonical(metrics)  # strict JSON: never NaN/Infinity

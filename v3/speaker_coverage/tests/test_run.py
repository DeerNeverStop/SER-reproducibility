"""CPU toy checks for budget, checkpoint integrity, phase isolation and resumability."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from v3.speaker_coverage import run
from v3.speaker_coverage.plan import model_config


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def unit_fixture(model="ridge_wavlm"):
    rows = {f"x{i}.wav": {"label_index": str(i % 6)} for i in range(30)}
    unit = {"model": model, "block": "ft" if model == "wavlm_ft" else "core",
            "config": model_config(model), "train_seed": 13, "policy": "U", "fold": 0,
            "rotation": 0, "draw": 0, "seed_index": 0,
            "fit": list(rows)[:18], "val": list(rows)[18:24], "test": list(rows)[24:]}
    unit["unit_id"] = run.digest(unit)
    return unit, rows


class Features:
    def __init__(self):
        self.x = np.random.RandomState(2).normal(size=(30, 8)).astype(np.float32)
    def get(self, corpus, kind, paths, state=None):
        return self.x[[int(x[1:-4]) for x in paths]]


def test_ridge_serialization_and_train_only_scaler(tmp_path):
    unit, rows = unit_fixture()
    feats = Features()
    before = feats.x[:18].mean(axis=0, dtype=np.float64)
    feats.x[18:] += 10000
    best, last, history, info = run.fit_ridge(unit, rows, feats, tmp_path / "checkpoint.npz")
    assert np.array_equal(best, last)
    assert history == [] and info["epochs_run"] == 0 and info["checkpoint_reload_verified"]
    with np.load(tmp_path / "checkpoint.npz") as saved:
        assert np.allclose(saved["scaler_mean"], before)
        assert saved["classes"].tolist() == list(range(6))


def test_cnn_full_budget_even_when_validation_deteriorates():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(6, 6, bias=False)
            torch.nn.init.zeros_(self.fc.weight)
        def forward(self, x):
            return self.fc(x)
    cfg = {"batch_size": 6, "epochs": 100, "patience": 1, "lr": 0.05,
           "weight_decay": 0.0, "early_stopping": False}
    x, y = np.eye(6, dtype=np.float32), np.arange(6, dtype=np.int64)
    model = Toy()
    best, last, history, best_epoch = run.full_cnn_loop(model, x, y, x, (y + 1) % 6,
                                                       cfg, 13, torch.device("cpu"))
    assert len(history) == 100
    assert sum(h["optimizer_steps"] for h in history) == 100
    assert best_epoch == 1
    assert not torch.equal(best["fc.weight"], last["fc.weight"])
    assert torch.equal(last["fc.weight"], model.state_dict()["fc.weight"])
    assert all(set(h) == {"epoch", "train_loss", "val_loss", "optimizer_steps"} for h in history)


class TinyFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Linear(6, 6))
        for p in self.enc[0].parameters():
            p.requires_grad = False
        self.fc = torch.nn.Linear(6, 6)
        self.register_buffer("counter", torch.tensor(2, dtype=torch.int64))
    def forward(self, x):
        return self.fc(self.enc(x))


def test_ft_full_budget_delta_includes_buffers_and_restores(tmp_path):
    unit, rows = unit_fixture("wavlm_ft")
    def batch(paths, train):
        return torch.eye(6)[[int(p[1:-4]) % 6 for p in paths]]
    batch.rng = np.random.RandomState(unit["train_seed"])
    model = TinyFT()
    original = copy.deepcopy(model)
    best, last, history, best_epoch, frozen = run.full_ft_loop(model, unit, rows, batch, torch.device("cpu"))
    assert len(history) == 15 and sum(h["optimizer_steps"] for h in history) == 30
    assert frozen == run.frozen_hash(original) == run.frozen_hash(model)
    assert "counter" in best and "enc.0.weight" not in best
    path = tmp_path / "delta.pt"
    torch.save({"best": best, "last": last}, path)
    saved = torch.load(path, weights_only=True)
    run.load_delta(original, saved["last"], frozen)
    assert np.array_equal(run.predict_wave(original, unit["test"], batch, torch.device("cpu")),
                          run.predict_wave(model, unit["test"], batch, torch.device("cpu")))
    assert best_epoch == 1 + np.argmin([h["val_loss"] for h in history])
    bad = dict(best)
    del bad["counter"]
    with pytest.raises(run.IntegrityError, match="delta omits"):
        run.load_delta(original, bad, frozen)
    with torch.no_grad():
        original.enc[0].weight.add_(1)
    with pytest.raises(run.IntegrityError, match="frozen base"):
        run.load_delta(original, best, frozen)


def test_replay_rejects_difference_and_nan():
    x = np.zeros((3, 6))
    with pytest.raises(run.IntegrityError, match="differs"):
        run.reload_difference(x, x + 0.1)
    with pytest.raises(run.IntegrityError, match="invalid"):
        run.reload_difference(x, x + np.nan)


def make_done(tmp_path, phase="formal"):
    unit, rows = unit_fixture()
    plan = {"units": [unit], "input": {"cache": "fixed"}, "plan_sha256": "f" * 64}
    out = tmp_path / phase
    attempt, directory = run.next_attempt(out / "units" / unit["unit_id"], False, 2)
    logits, last, history, info = run.fit_ridge(unit, rows, Features(), directory / "checkpoint.npz")
    info.update(fit_seconds=0.1, wall_seconds=0.2, peak_cuda_bytes=0)
    run.write_completed(unit, out, plan, phase, attempt, run.prediction_arrays(unit, rows, logits, last),
                        history, info, {"python": "test", "torch": torch.__version__})
    return unit, rows, plan, out


def test_done_resume_tamper_and_phase_separation(tmp_path):
    unit, rows, plan, out = make_done(tmp_path)
    assert run.verify_completed(unit, out, plan["plan_sha256"], "formal", rows)
    assert not run.verify_completed(unit, tmp_path / "pilot", plan["plan_sha256"], "pilot", rows)
    with pytest.raises(run.IntegrityError, match="identity"):
        run.verify_completed(unit, out, "changed", "formal", rows)
    checkpoint = out / "units" / unit["unit_id"] / "attempts/0001/checkpoint.npz"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(run.IntegrityError, match="artifact"):
        run.verify_completed(unit, out, plan["plan_sha256"], "formal", rows)


def test_partial_attempt_requires_explicit_bounded_fresh_retry(tmp_path):
    first, path = run.next_attempt(tmp_path, False, 2)
    (path / "checkpoint.pt").write_bytes(b"partial")
    with pytest.raises(run.IntegrityError, match="retry-failed"):
        run.next_attempt(tmp_path, False, 2)
    second, newpath = run.next_attempt(tmp_path, True, 2)
    assert (first, second) == (1, 2) and newpath != path and not list(newpath.iterdir())
    assert (path / "checkpoint.pt").read_bytes() == b"partial"
    with pytest.raises(run.IntegrityError, match="budget exhausted"):
        run.next_attempt(tmp_path, True, 2)


def test_fixed_unit_shards_preserve_plan_order():
    a, _ = unit_fixture()
    b = {**a, "unit_id": "b", "fold": 1}
    plan = {"units": [a, b]}
    assert run.select_units(plan, "formal", "ridge_wavlm", ["b", a["unit_id"]]) == [a, b]
    for ids in (["unknown"], [a["unit_id"], a["unit_id"]]):
        with pytest.raises(run.IntegrityError):
            run.select_units(plan, "formal", "ridge_wavlm", ids)
    with pytest.raises(run.IntegrityError, match="different model"):
        run.select_units(plan, "formal", "cnn", [a["unit_id"]])


def test_pilot_exact_18_plus_one_excludes_second_ft_seed():
    units = []
    for model in run.MODELS:
        for policy in "URC":
            for fold in range(5):
                for rotation in range(6):
                    for draw in (range(3) if model != "wavlm_ft" else [0]):
                        for seed_index in (range(2) if model == "wavlm_ft" else [0]):
                            units.append({"block": "ft" if model == "wavlm_ft" else "core", "model": model,
                                          "policy": policy, "fold": fold, "rotation": rotation,
                                          "draw": draw, "seed_index": seed_index})
    assert len(run.phase_units({"units": units}, "formal")) == 720
    pilot = run.phase_units({"units": units}, "pilot")
    assert len(pilot) == 19
    ft = [u for u in pilot if u["block"] == "ft"]
    assert ft[0]["policy"] == "U" and ft[0]["seed_index"] == 0


def test_incomplete_block_never_sealed(tmp_path):
    unit, rows, plan, out = make_done(tmp_path)
    missing = {**unit, "unit_id": "a" * 64}
    plan["units"].append(missing)
    result = run.finish_blocks(plan, out, "formal", rows)
    assert result["core"] == {"complete": 1, "expected": 2}
    assert not (out / "completion_core.json").exists()


def test_predictions_reject_label_reordering(tmp_path):
    unit, rows = unit_fixture()
    values = run.prediction_arrays(unit, rows, np.zeros((6, 6)), np.zeros((6, 6)))
    values["labels"] = values["labels"][::-1]
    path = tmp_path / "bad.npz"
    np.savez(path, **values)
    with pytest.raises(run.IntegrityError, match="labels"):
        run.validate_predictions(path, unit, rows)


def test_commit_crash_window_recovers_once_and_offload_never_retrains(tmp_path):
    unit, rows, plan, out = make_done(tmp_path)
    run.core.append_event(out, "unit_start", unit_id=unit["unit_id"], attempt=1)
    assert run.verify_completed(unit, out, plan["plan_sha256"], "formal", rows)
    run.record_verified_done(unit, out, recovered=True)
    run.record_verified_done(unit, out, recovered=True)
    events = [e for e in run.ledger_events(out) if e["event"] == "unit_done"]
    assert len(events) == 1 and events[0]["recovered_from_done"]
    assert events[0]["done_sha256"] == run.file_sha(out / "units" / unit["unit_id"] / "DONE")
    with pytest.raises(run.IntegrityError, match="previously completed"):
        run.reject_offloaded_repeat(unit, out)


def test_recovery_cannot_convert_failed_attempt_to_success(tmp_path):
    unit, _, _, out = make_done(tmp_path)
    run.core.append_event(out, "unit_start", unit_id=unit["unit_id"], attempt=1)
    run.core.append_event(out, "unit_failed", unit_id=unit["unit_id"], attempt=1)
    with pytest.raises(run.IntegrityError, match="nonfailed"):
        run.record_verified_done(unit, out, recovered=True)


def test_execute_refuses_independent_gate_failure_before_training(tmp_path, monkeypatch):
    from v3.speaker_coverage import verify
    unit, _ = unit_fixture()
    plan = {"schema": run.SCHEMA, "program": run.PROGRAM, "units": [unit], "input": {"asv_path": "asv.npz"}}
    plan["plan_sha256"] = run.digest(plan)
    plan_path = tmp_path / "plan.json"
    run.atomic_json(plan_path, plan)
    args = run.parser().parse_args(["--repo", str(run.REPO), "--plan", str(plan_path),
        "--features", str(tmp_path), "--audio-root", str(tmp_path), "--wavlm-model", str(tmp_path / "base.pt"),
        "--out", str(tmp_path / "output"), "--phase", "formal", "--model", "ridge_wavlm"])
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    calls = []
    def gate(*args, **kwargs):
        calls.append(kwargs)
        return {"pass": True, "formal_allowed": False}
    monkeypatch.setattr(verify, "verify_plan", gate)
    monkeypatch.setattr(run, "fit_ridge", lambda *args: pytest.fail("training before gate"))
    with pytest.raises(run.IntegrityError, match="did not authorize"):
        run.execute(args)
    assert len(calls) == 1 and calls[0]["audio_root"] == tmp_path
    assert not (tmp_path / "output").exists()

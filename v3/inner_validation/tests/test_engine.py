"""Synthetic CPU checks only: no study predictions, real WavLM or downloads."""
import copy

import numpy as np
import pytest
import torch

from v3.inner_validation import engine


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def fixture():
    cfg = {"epochs": 15, "batch_size": 16, "fp16": True, "early_stopping": False,
           "weight_decay": .01, "crop_seconds": 3., "eval_cap_seconds": 10.,
           "lr_encoder": 5e-5, "lr_head": .001, "trainable_layers": "top4+head"}
    unit = {"config": cfg, "train_seed": 71, "unit_id": "SYNTHETIC_ONLY"}
    rows = {}
    for role, count in (("fit", 18), ("val_seen", 12), ("val_unseen", 18), ("test", 6)):
        unit[role] = [f"{role}/{i}.wav" for i in range(count)]
        rows.update({p: {"label_index": str(i % 6)} for i, p in enumerate(unit[role])})
    return unit, rows


def batch_fixture(unit, calls=None):
    rng = np.random.RandomState(unit["train_seed"])
    def batch(paths, train):
        if calls is not None:
            calls.append((tuple(paths), train))
        indexes = [int(p.rsplit("/", 1)[1][:-4]) for p in paths]
        x = np.eye(6, dtype=np.float32)[np.asarray(indexes) % 6]
        if train:
            x = x + rng.normal(0, .05, size=x.shape).astype(np.float32)
        return torch.from_numpy(x)
    batch.rng = rng
    return batch


class TinyFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Dropout(.2), torch.nn.Linear(6, 6))
        for parameter in self.enc[0].parameters():
            parameter.requires_grad = False
        self.fc = torch.nn.Linear(6, 6)
        self.register_buffer("counter", torch.tensor(2, dtype=torch.int64))
    def forward(self, x):
        return self.fc(self.enc(x))


def controlled_losses(monkeypatch, shared=False):
    calls = {"val_seen": 0, "val_unseen": 0}
    def loss(model, paths, labels, batch, device):
        role = paths[0].split("/")[0]
        calls[role] += 1
        epoch = calls[role]
        if shared:
            return float(16 - epoch)
        # Seen has a tie at epochs 2/3: keep the earliest. Unseen selects 4.
        low = (2, 3) if role == "val_seen" else (4,)
        return .1 if epoch in low else 2. + epoch / 100
    monkeypatch.setattr(engine, "_validation_loss", loss)
    return calls


def test_full_trajectory_is_unchanged_by_validation_labels():
    unit, rows = fixture()
    torch.manual_seed(6)
    original = TinyFT()
    model1, model2 = copy.deepcopy(original), copy.deepcopy(original)
    states1, selected1, hist1, frozen1 = engine.full_dual_loop(
        model1, unit, rows, batch_fixture(unit), torch.device("cpu"))
    altered = copy.deepcopy(rows)
    for role in ("val_seen", "val_unseen"):
        for path in unit[role]:
            altered[path]["label_index"] = str((int(altered[path]["label_index"]) + 1) % 6)
    states2, selected2, hist2, frozen2 = engine.full_dual_loop(
        model2, unit, altered, batch_fixture(unit), torch.device("cpu"))
    assert len(hist1) == len(hist2) == 15
    assert sum(h["optimizer_steps"] for h in hist1) == 30
    assert all(h["scaler_skipped_steps"] == 0 for h in hist1)
    assert [h["train_loss"] for h in hist1] == [h["train_loss"] for h in hist2]
    assert [h["val_seen_loss"] for h in hist1] != [h["val_seen_loss"] for h in hist2]
    assert all(torch.equal(states1["15"][k], states2["15"][k]) for k in states1["15"])
    assert frozen1 == frozen2 == engine.frozen_hash(original)
    assert all(set(h) == {"epoch", "train_loss", "val_seen_loss", "val_unseen_loss",
                          "optimizer_steps", "scaler_skipped_steps"} for h in hist1)
    assert selected1["best_seen"] == min(hist1, key=lambda h: h["val_seen_loss"])["epoch"]
    assert selected2["best_unseen"] == min(hist2, key=lambda h: h["val_unseen_loss"])["epoch"]


def test_validation_ce_is_sum_over_all_rows_including_short_batch():
    unit, rows = fixture()
    paths = unit["val_unseen"]
    logits = torch.arange(108, dtype=torch.float32).reshape(18, 6) / 40
    logits[16:] *= -6
    labels = np.arange(18) % 6
    lookup = {p: i for i, p in enumerate(paths)}
    def batch(subset, train):
        assert train is False
        return logits[[lookup[p] for p in subset]]
    actual = engine._validation_loss(torch.nn.Identity(), paths, labels, batch, torch.device("cpu"))
    wanted = float(torch.nn.functional.cross_entropy(logits, torch.tensor(labels), reduction="sum")) / 18
    wrong = (float(torch.nn.functional.cross_entropy(logits[:16], torch.tensor(labels[:16])))
             + float(torch.nn.functional.cross_entropy(logits[16:], torch.tensor(labels[16:])))) / 2
    assert actual == pytest.approx(wanted, abs=1e-6)
    assert abs(actual - wrong) > .01


def test_separate_minima_first_tie_and_unique_epoch_deltas(monkeypatch):
    unit, rows = fixture()
    calls = controlled_losses(monkeypatch)
    states, selected, history, frozen = engine.full_dual_loop(
        TinyFT(), unit, rows, batch_fixture(unit), torch.device("cpu"))
    assert calls == {"val_seen": 15, "val_unseen": 15}
    assert selected == {"best_seen": 2, "best_unseen": 4, "last": 15}
    assert set(states) == {"2", "4", "15"}
    assert "counter" in states["2"] and "enc.0.weight" not in states["2"]
    assert not torch.equal(states["2"]["fc.weight"], states["15"]["fc.weight"])


def install_toy_builder(monkeypatch, unit):
    builds, prediction_calls = [], []
    def builder(path, seed):
        builds.append((str(path), seed))
        torch.manual_seed(seed)
        return TinyFT(), "synthetic-bundle-params"
    monkeypatch.setattr(engine, "build_ft_model", builder)
    monkeypatch.setattr(engine, "_make_batch", lambda u, root, cache: batch_fixture(u))
    native_predict = engine.predict_wave
    def predict(model, paths, batch, device):
        prediction_calls.append(tuple(paths))
        return native_predict(model, paths, batch, device)
    monkeypatch.setattr(engine, "predict_wave", predict)
    return builds, prediction_calls


@pytest.mark.parametrize("shared", [False, True])
def test_fit_disk_rebuild_all_roles_and_epoch_dedup(tmp_path, monkeypatch, shared):
    unit, rows = fixture()
    controlled_losses(monkeypatch, shared=shared)
    builds, calls = install_toy_builder(monkeypatch, unit)
    base, checkpoint = tmp_path / "SYNTHETIC_BASE", tmp_path / "checkpoint.pt"
    base.write_bytes(b"SYNTHETIC_NO_REAL_MODEL")
    arrays, history, info = engine.fit_dual(unit, rows, base, tmp_path, torch.device("cpu"), checkpoint)
    assert len(builds) == 2 and len(history) == 15
    expected_epochs = [15] if shared else [2, 4, 15]
    assert info["checkpoint_unique_epochs"] == expected_epochs
    assert info["checkpoint_unique_epoch_count"] == len(expected_epochs)
    assert len(calls) == len(expected_epochs) * 3 * 2
    assert all(paths in [tuple(unit[r]) for r in engine.ROLES] for paths in calls)
    assert set(arrays) == engine.PREDICTION_KEYS and len(arrays) == 15
    assert info["checkpoint_reload_verified"] and info["checkpoint_reload_max_abs_diff"] == 0
    assert len(info["checkpoint_reload_max_abs_diff_by_prediction"]) == 9
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["schema"] == engine.SCHEMA
    assert set(saved["epoch_states"]) == {str(x) for x in expected_epochs}
    assert saved["base_file_sha256"] == engine.legacy.file_sha(base)
    for state in saved["epoch_states"].values():
        assert all(t.dtype == torch.float32 for k, t in state.items() if k != "counter")
    for role in engine.ROLES:
        assert arrays[role + "__paths"].tolist() == unit[role]
        assert arrays[role + "__labels"].dtype == np.int64
        for name in engine.CHECKPOINTS:
            values = arrays[f"{role}__{name}__logits"]
            assert values.dtype == np.float64 and values.shape == (len(unit[role]), 6)
        if shared:
            assert np.array_equal(arrays[f"{role}__best_seen__logits"], arrays[f"{role}__last__logits"])
    with pytest.raises(ValueError, match="already exists"):
        engine.fit_dual(unit, rows, base, tmp_path, torch.device("cpu"), checkpoint)


@pytest.mark.parametrize("corruption", ["logits", "dtype", "buffer"])
def test_disk_restore_mutations_fail_closed(tmp_path, monkeypatch, corruption):
    unit, rows = fixture()
    controlled_losses(monkeypatch, shared=True)
    install_toy_builder(monkeypatch, unit)
    base = tmp_path / "SYNTHETIC_BASE"
    base.write_bytes(b"SYNTHETIC_NO_REAL_MODEL")
    native_load = torch.load
    def load(*args, **kwargs):
        saved = native_load(*args, **kwargs)
        state = saved["epoch_states"]["15"]
        if corruption == "logits":
            state["fc.bias"] += 1
        elif corruption == "dtype":
            state["fc.weight"] = state["fc.weight"].half()
        else:
            del state["counter"]
        return saved
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="replay differs|lossless dtype|delta coverage"):
        engine.fit_dual(unit, rows, base, tmp_path, torch.device("cpu"), tmp_path / "bad.pt")


def test_validation_cannot_consume_training_rng():
    unit, rows = fixture()
    original = batch_fixture(unit)
    def batch(paths, train):
        if not train:
            original.rng.rand()
        return original(paths, train)
    batch.rng = original.rng
    with pytest.raises(ValueError, match="consumed training random state"):
        engine.full_dual_loop(TinyFT(), unit, rows, batch, torch.device("cpu"))


def test_invalid_budget_overlap_and_unbalanced_labels_rejected():
    unit, rows = fixture()
    for key, value in (("epochs", 14), ("batch_size", 8), ("fp16", False),
                       ("lr_encoder", .001), ("trainable_layers", "top12+head")):
        changed = copy.deepcopy(unit)
        changed["config"][key] = value
        with pytest.raises(ValueError):
            engine.validate_unit(changed, rows)
    changed = copy.deepcopy(unit)
    changed["val_seen"][0] = changed["fit"][0]
    with pytest.raises(ValueError, match="cross roles"):
        engine.validate_unit(changed, rows)
    for role in ("fit", "val_seen", "val_unseen"):
        changed_rows = copy.deepcopy(rows)
        changed_rows[unit[role][0]]["label_index"] = "1"
        with pytest.raises(ValueError, match="class-balanced"):
            engine.validate_unit(unit, changed_rows)


def test_scaler_skip_is_counted_separately_from_attempt():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    class OverflowScaler:
        current = 8.
        def get_scale(self): return self.current
        def scale(self, loss): return loss
        def step(self, optimizer): pass
        def update(self): self.current /= 2
    assert engine._optimizer_step(parameter * 3, optimizer, OverflowScaler()) == 1
    assert parameter.item() == 1.
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    assert engine._optimizer_step(parameter * 3, optimizer, scaler) == 0
    assert parameter.item() == pytest.approx(.7)

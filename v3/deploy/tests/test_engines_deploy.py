"""CPU tests for v3/deploy/engines_deploy.py and calib_gpu.py (SER26-DEPLOY-1, block B2). No GPU, no downloads.

    E:/科研/SER/ser_gpu/Scripts/python.exe -m pytest v3/deploy/tests/test_engines_deploy.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SER_V2_DATA_ROOT", "E:/科研/essay/SER-v2/v2")

from v3.deploy import common  # noqa: E402

GUARD = common.cpu_guard(2)          # CUDA hidden, 2 threads, BelowNormal - before torch is imported anywhere

from v3.deploy import calib_gpu, engines_deploy  # noqa: E402

torch = pytest.importorskip("torch")
DEVICE = torch.device("cpu")
N_CLASSES, N_MELS, FRAMES = 3, 64, 128


# ----------------------------------------------------------------------------- synthetic log-mel data

class SyntheticData:
    """40 utterances, 3 classes, 5 speakers: class template + speaker offset + noise, per-utterance z-scored."""

    def __init__(self, n=40, seed=7):
        rng = np.random.RandomState(seed)
        self.paths = [f"spk{i % 5:02d}/utt_{i:03d}.wav" for i in range(n)]
        self.y = {p: i % N_CLASSES for i, p in enumerate(self.paths)}
        self.spk = {p: p.split("/")[0] for p in self.paths}
        mel_axis = np.linspace(0, 1, N_MELS)[:, None]
        spk_off = {s: rng.normal(0, 0.6, size=(N_MELS, 1)) for s in set(self.spk.values())}
        self.X = {}
        for p in self.paths:
            x = np.sin(2 * np.pi * (self.y[p] + 1) * mel_axis) + spk_off[self.spk[p]] + rng.normal(0, 1.0, size=(N_MELS, FRAMES))
            self.X[p] = ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)
        self.n_classes = N_CLASSES
        self.audio_root = None

    def labels(self, paths):
        return np.asarray([self.y[p] for p in paths], dtype=np.int64)

    def speakers(self, paths):
        return [self.spk[p] for p in paths]

    def logmel(self, paths):
        return np.stack([self.X[p] for p in paths])

    def fold(self):
        # speaker-grouped: spk03 -> val, spk04 -> test, the rest fit (24 / 8 / 8)
        fit = [p for p in self.paths if self.spk[p] in ("spk00", "spk01", "spk02")]
        val = [p for p in self.paths if self.spk[p] == "spk03"]
        test = [p for p in self.paths if self.spk[p] == "spk04"]
        return {"fit": fit, "val": val, "test": test}


CNN_CFG = {"batch_size": 8, "dropout": 0.1, "engine": "p1_frozen", "epochs": 2, "lr": 0.001, "model": "cnn", "patience": 15,
           "weight_decay": 0.0001, "config_source": "tuned_standard_experiment.candidate_configs(model)[2]"}
ROW = {"train_seed": 20260905, "unit_id": "test-unit"}


def run_cnn(data, cfg, seed=20260905):
    return engines_deploy.engine_p1_frozen({**ROW, "train_seed": seed}, cfg, data.fold(), data, DEVICE)


# ----------------------------------------------------------------------------- p1_frozen copy

def test_cpu_guard_and_no_cuda():
    assert GUARD["cuda_visible_devices"] == "-1" and GUARD["threads"] == 2
    assert not torch.cuda.is_initialized()


def test_p1_frozen_shapes_and_val_export():
    data = SyntheticData()
    fold = data.fold()
    test_logits, val_logits, history, info = run_cnn(data, CNN_CFG)
    assert test_logits.shape == (len(fold["test"]), N_CLASSES) and test_logits.dtype == np.float64
    assert val_logits.shape == (len(fold["val"]), N_CLASSES) and val_logits.dtype == np.float64
    assert np.isfinite(test_logits).all() and np.isfinite(val_logits).all()
    assert len(history) == 2 and info["epochs_run"] == 2 and info["best_epoch"] in (1, 2)
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "val_uar"}
    assert info["arch"]["width"] == 48 and info["arch"]["n_blocks"] == 3 and info["arch"]["augment"] is True
    assert info["arch"]["lr"] == 0.001 and info["arch"]["weight_decay"] == 0.0001 and info["arch"]["dropout"] == 0.1


def test_p1_frozen_deterministic_for_same_seed():
    data = SyntheticData()
    a = run_cnn(data, CNN_CFG)
    b = run_cnn(data, CNN_CFG)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert [h["val_loss"] for h in a[2]] == [h["val_loss"] for h in b[2]]
    c = run_cnn(data, CNN_CFG, seed=1)
    assert not np.array_equal(a[0], c[0])


def test_p1_frozen_restores_best_checkpoint_and_stops_on_patience():
    data = SyntheticData()
    fold = data.fold()
    cfg = {**CNN_CFG, "epochs": 12, "patience": 2, "lr": 0.02}          # aggressive lr so val loss is non-monotone
    test_logits, val_logits, history, info = run_cnn(data, cfg)
    losses = [h["val_loss"] for h in history]
    best = int(np.argmin(losses)) + 1
    assert info["best_epoch"] == best and info["val_loss_best"] == losses[best - 1]
    # stopping rule: either patience exhausted exactly `patience` epochs after the best, or all epochs run
    assert info["epochs_run"] == min(cfg["epochs"], best + cfg["patience"])
    # the exported val logits belong to the restored checkpoint: recomputing the weighted CE reproduces val_loss_best
    y_fit, y_val = data.labels(fold["fit"]), data.labels(fold["val"])
    counts = np.bincount(y_fit, minlength=N_CLASSES).astype(np.float64)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / N_CLASSES, dtype=torch.float32)
    ce = torch.nn.functional.cross_entropy(torch.from_numpy(val_logits).float(), torch.from_numpy(y_val), weight=w).item()
    assert abs(ce - info["val_loss_best"]) < 1e-4
    assert abs(common.uar(y_val, val_logits.argmax(1), N_CLASSES) - info["val_uar_best"]) < 1e-9


def test_spec_augment_masks_and_noise_only_in_copy():
    x = torch.ones(3, N_MELS, FRAMES)
    rng = np.random.RandomState(0)
    out = engines_deploy.spec_augment(x, rng, time_mask=16, freq_mask=8, noise_std=0.0)
    assert torch.all(x == 1.0)                                  # input untouched
    assert out.shape == x.shape and (out == 0).any()            # some masked cells
    rng2 = np.random.RandomState(0)
    assert torch.equal(engines_deploy.spec_augment(x, rng2, 16, 8, 0.0), out)


def test_p1_arch_matches_frozen_constant():
    arch, source = engines_deploy.p1_arch({"model": "cnn", "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1})
    assert arch == engines_deploy.P1_CNN_ARCH_FROZEN
    assert source.startswith("tuned_standard_experiment") or source.startswith("frozen_constant")


# ----------------------------------------------------------------------------- WavLM batching / cropping helper

def test_wavlm_batch_train_crop_and_eval_cap():
    sr = 16000
    crop, cap = 3 * sr, 10 * sr
    lengths = {"a.wav": 2 * sr, "b.wav": 5 * sr, "c.wav": 12 * sr}
    store = {k: np.random.RandomState(i).normal(0, 0.1, size=n).astype(np.float16) for i, (k, n) in enumerate(lengths.items())}

    def wav(p):
        return store[p]

    rng = np.random.RandomState(3)
    xb = engines_deploy.wavlm_batch(list(lengths), True, wav, crop, cap, rng)
    assert xb.shape == (3, crop) and xb.dtype == torch.float32
    assert torch.all(xb[0, 2 * sr:] == 0)                       # short clip zero-padded to crop
    assert torch.count_nonzero(xb[1]) > 0.9 * crop              # long clips cropped, not padded
    rng_a, rng_b = np.random.RandomState(11), np.random.RandomState(11)
    assert torch.equal(engines_deploy.wavlm_batch(["b.wav", "c.wav"], True, wav, crop, cap, rng_a),
                       engines_deploy.wavlm_batch(["b.wav", "c.wav"], True, wav, crop, cap, rng_b))
    # crop position is drawn from the rng: a different rng state changes the crop of a long clip
    assert not torch.equal(engines_deploy.wavlm_batch(["c.wav"], True, wav, crop, cap, np.random.RandomState(1)),
                           engines_deploy.wavlm_batch(["c.wav"], True, wav, crop, cap, np.random.RandomState(2)))
    ev = engines_deploy.wavlm_batch(list(lengths), False, wav, crop, cap, rng)
    assert ev.shape == (3, cap)                                 # capped at 10 s, padded to the batch max
    assert torch.all(ev[0, 2 * sr:] == 0) and torch.all(ev[1, 5 * sr:] == 0)
    assert torch.allclose(ev[2], torch.from_numpy(store["c.wav"][:cap].astype(np.float32)))
    ev2 = engines_deploy.wavlm_batch(["a.wav", "b.wav"], False, wav, crop, cap, rng)
    assert ev2.shape == (2, 5 * sr)                             # eval batches pad only to their own longest member


def test_wave_cache_loads_trims_and_caches(tmp_path):
    sf = pytest.importorskip("soundfile")
    sr = 16000
    y = np.zeros(sr, dtype=np.float32)
    y[4000:12000] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(8000) / sr)
    (tmp_path / "s").mkdir()
    sf.write(tmp_path / "s" / "u.wav", y, 22050)                 # resampled to 16 kHz on load, trimmed at 30 dB
    cache = engines_deploy.WaveCache(tmp_path, sr)
    w = cache("s/u.wav")
    assert w.dtype == np.float16 and 0 < len(w) < sr and len(cache) == 1
    assert cache("s/u.wav") is w


# ----------------------------------------------------------------------------- calib_gpu contract smoke test

def fake_engine(row, cfg, fold, data, device):
    rng = np.random.RandomState(int(row["train_seed"]))
    test = rng.normal(size=(len(fold["test"]), data.n_classes))
    val = rng.normal(size=(len(fold["val"]), data.n_classes))
    history = [{"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_uar": 50.0}, {"epoch": 2, "train_loss": 0.8, "val_loss": 0.95, "val_uar": 40.0}]
    return test, val, history, {"best_epoch": 1, "epochs_run": 2, "val_loss_best": 0.9, "val_uar_best": 50.0, "train_seconds": 0.01}


def test_calib_gpu_writes_contract_and_is_resumable(tmp_path):
    from v3.deploy import calib
    data = SyntheticData()
    fold = data.fold()
    u = {"unit_id": "u" * 64, "program": "SER26-DEPLOY-1", "module": "B2", "level": "ravdess", "base": "ravdess", "model": "cnn",
         "engine": "p1_frozen", "cell": "GG", "r": 0, "fold": 0, "seed_index": 0, "train_seed": 123, "split_key": "ctrl__ravdess__GG__r0",
         "split_sha256": "x", "config_sha256": "y", "config": {"model": "cnn", "engine": "p1_frozen"}, "n_classes": N_CLASSES,
         "n_fit": len(fold["fit"]), "n_val": len(fold["val"]), "n_test": len(fold["test"])}
    udir = calib_gpu.unit_dir(tmp_path, u["unit_id"])
    env = {"hostname": "test", "device": "cpu"}
    unit = calib_gpu.run_unit(u, fold, data, fake_engine, DEVICE, udir, env, {"classes": ["a", "b", "c"]})
    assert calib.done_status(udir, N_CLASSES, u["n_val"], u["n_test"]) == "ok"
    done = (udir / "DONE").read_text(encoding="utf-8")
    assert done == f"{unit['val_sha256']} {unit['test_sha256']}\n"
    assert done.split() == [common.sha256_file(udir / "val_predictions.csv"), common.sha256_file(udir / "test_predictions.csv")]
    with open(udir / "val_predictions.csv", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\n")
        body = fh.read()
    assert header == "relative_path,speaker,y_true,y_pred,logit_0,logit_1,logit_2"
    assert "\r" not in body and body.count("\n") == len(fold["val"])
    val = common.read_predictions_csv(udir / "val_predictions.csv", "logit")
    assert val["paths"] == fold["val"] and np.array_equal(val["y_pred"], val["scores"].argmax(1))
    assert np.array_equal(val["y_true"], data.labels(fold["val"]))
    uj = json.loads((udir / "unit.json").read_text(encoding="utf-8"))
    for k in ("status", "engine", "environment", "timing", "best_epoch", "epochs_run", "val_sha256", "test_sha256", "n_val_rows", "n_test_rows"):
        assert k in uj
    assert uj["status"] == "done" and uj["best_epoch"] == 1 and uj["epochs_run"] == 2 and uj["n_val_rows"] == len(fold["val"])
    assert all(uj[k] == v for k, v in u.items())                # plan row verbatim
    assert json.loads((udir / "history.json").read_text(encoding="utf-8"))[0]["epoch"] == 1
    # resumability: a corrupted csv invalidates DONE; an intact directory is "ok"
    (udir / "test_predictions.csv").write_text("relative_path,speaker,y_true,y_pred,logit_0,logit_1,logit_2\n", encoding="utf-8")
    assert calib.done_status(udir, N_CLASSES, u["n_val"], u["n_test"]) == "DONE sha mismatch"


def test_calib_gpu_select_units_and_audio_roots(tmp_path):
    plan = {"module": "B2", "units": [{"unit_id": "a", "model": "cnn"}, {"unit_id": "b", "model": "wavlm_ft"}, {"unit_id": "c", "model": "ridge_hubert_base"},
                                      {"unit_id": "d", "model": "cnn"}]}
    assert [u["unit_id"] for u in calib_gpu.select_units(plan, ["wavlm_ft", "cnn"])] == ["b", "a", "d"]
    assert [u["unit_id"] for u in calib_gpu.select_units(plan, ["cnn"], ["d"])] == ["d"]
    with pytest.raises(common.IntegrityError):
        calib_gpu.select_units(plan, ["ridge_hubert_base"])
    with pytest.raises(common.IntegrityError):
        calib_gpu.select_units(plan, ["cnn"], ["b"])
    roots = calib_gpu.resolve_audio_roots(tmp_path, "pod")
    assert roots == {"ravdess": tmp_path / "ravdess", "cremad": tmp_path / "cremad", "subesco": tmp_path / "subesco"}
    local = calib_gpu.resolve_audio_roots(None, "local_windows")
    assert all(p.is_absolute() for p in local.values()) and set(local) == {"ravdess", "cremad", "subesco"}
    with pytest.raises(common.IntegrityError):
        calib_gpu.resolve_audio_roots(None, "pod")

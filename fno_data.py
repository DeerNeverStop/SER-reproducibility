"""
fno_data.py
===========
Turn audio files into fixed-size log-mel spectrograms for the FNO model.
Reuses the label parsing from the traditional pipeline (dataset.list_audio).

Each clip -> a (n_mels, n_frames) log-mel spectrogram, per-sample standardized.
All spectrograms are precomputed once and cached to a .npz so training epochs
don't recompute them.
"""
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm

from dataset import list_audio   # reuse RAVDESS / folder label parsing

SR = 22050
N_MELS = 64
N_FRAMES = 128       # ~3 s at hop_length=512


def load_logmel(path, sr=SR, n_mels=N_MELS, n_frames=N_FRAMES):
    """Load one file -> RAW log-mel spectrogram of shape (n_mels, n_frames).

    Normalization is applied later by normalize_spectrograms() so the
    per-speaker option can pool statistics across each speaker's clips.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size < sr // 10:
        y = np.pad(y, (0, sr // 10 - y.size + 1))
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512)
    logmel = librosa.power_to_db(mel, ref=np.max)          # (n_mels, T)
    # pad or truncate the time axis to a fixed length
    if logmel.shape[1] < n_frames:
        pad = n_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)),
                        mode="constant", constant_values=logmel.min())
    else:
        logmel = logmel[:, :n_frames]
    return logmel.astype(np.float32)


def normalize_spectrograms(X, groups=None, mode="speaker"):
    """Normalize log-mel spectrograms. X: (N, n_mels, T). Returns a new array.

    mode='sample'  : per-clip z-score (one scalar mean/std over the whole
                     spectrogram). Removes per-clip loudness offset only.
    mode='speaker' : per-speaker, per-mel-bin CMVN. For each speaker, pool all of
                     that speaker's frames and z-score every mel band by that
                     speaker's own mean/std. This strips the speaker's average
                     spectral profile (their timbre/pitch/loudness baseline),
                     leaving the emotional deviation. Label-free, so usable even
                     for held-out speakers. Falls back to 'sample' if no groups.
    """
    Xn = X.astype(np.float32).copy()
    if mode == "sample" or groups is None:
        for i in range(len(Xn)):
            Xn[i] = (Xn[i] - Xn[i].mean()) / (Xn[i].std() + 1e-6)
        return Xn
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sub = Xn[idx]                                   # (n_clips, n_mels, T)
        mu = sub.mean(axis=(0, 2), keepdims=True)       # (1, n_mels, 1)
        sd = sub.std(axis=(0, 2), keepdims=True)
        sd[sd < 1e-6] = 1.0
        Xn[idx] = (sub - mu) / sd
    return Xn


def spec_augment(spec, n_time_masks=2, time_mask_width=16,
                 n_freq_masks=2, freq_mask_width=8, noise_std=0.0, rng=None):
    """SpecAugment (+ optional Gaussian noise) on ONE spectrogram (n_mels, T).

    Randomly zeroes `n_time_masks` time bands (up to `time_mask_width` frames
    each) and `n_freq_masks` frequency bands (up to `freq_mask_width` mel bins
    each), then optionally adds light noise. Masks are filled with 0, which
    equals the mean of a normalized spectrogram (standard "mask to mean").

    USE ON TRAINING DATA ONLY, freshly each epoch. Never on the test set.
    """
    rng = rng if rng is not None else np.random.default_rng()
    spec = spec.copy()
    n_mels, T = spec.shape
    for _ in range(n_time_masks):
        w = int(rng.integers(0, time_mask_width + 1))
        if 0 < w < T:
            t0 = int(rng.integers(0, T - w))
            spec[:, t0:t0 + w] = 0.0
    for _ in range(n_freq_masks):
        w = int(rng.integers(0, freq_mask_width + 1))
        if 0 < w < n_mels:
            f0 = int(rng.integers(0, n_mels - w))
            spec[f0:f0 + w, :] = 0.0
    if noise_std > 0:
        spec = spec + rng.normal(0, noise_std, spec.shape).astype(spec.dtype)
    return spec


def build_mel_dataset(data_dir, scheme="ravdess", cache="mel_cache.npz"):
    """Return (X, y, groups). X has shape (N, n_mels, n_frames)."""
    if cache and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        if "kind" not in d.files or str(d["kind"]) != "raw_logmel_v2":
            raise RuntimeError(
                f"'{cache}' is an old/incompatible cache (it held normalized "
                "spectrograms). Delete it and rerun so RAW spectrograms are "
                "recomputed.")
        print(f"Loaded cached spectrograms from {cache}")
        return d["X"], d["y"], d["groups"]

    items = list_audio(data_dir, scheme)
    if not items:
        raise RuntimeError(f"No labelled audio under '{data_dir}' (scheme='{scheme}').")

    X, y, groups = [], [], []
    for path, emo, group in tqdm(items, desc="Computing mel-spectrograms"):
        try:
            X.append(load_logmel(path))
            y.append(emo)
            groups.append(group)
        except Exception as exc:
            print(f"  ! skipping {path.name}: {exc}")
    X = np.stack(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if cache:
        np.savez_compressed(cache, X=X, y=y, groups=groups, kind="raw_logmel_v2")
        print(f"Cached spectrograms -> {cache}")
    return X, y, groups

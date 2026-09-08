"""
features.py
===========
Hand-crafted acoustic feature extraction for *traditional* speech emotion
recognition.

Pipeline implemented here (per audio file):
    1. load + silence trimming            -> rough endpoint detection
    2. frame-level features               -> prosodic / spectral / voice-quality
    3. statistical functionals over time  -> fixed-length vector

The three feature families correspond to what is usually used in classical SER:
    - Prosodic    : F0 (pitch), energy/RMS, zero-crossing rate
    - Spectral    : MFCC (+delta), chroma, spectral contrast, centroid,
                    bandwidth, rolloff, flatness
    - Voice quality: jitter, shimmer, HNR  (via Praat / parselmouth, optional)

Frame-level features are variable length, so we collapse each feature's time
series with a set of statistical functionals (mean, std, min, max, range,
median, skew). This is what turns a variable-length utterance into one
fixed-length feature vector that a classical classifier can consume.
"""
from pathlib import Path

import numpy as np
import librosa
from scipy import stats

# --- optional Praat backend for voice-quality features -----------------------
try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except Exception:  # pragma: no cover - environment dependent
    HAS_PRAAT = False

SR = 22050          # working sample rate
N_MFCC = 13         # classic MFCC order

_FUNC_NAMES = ["mean", "std", "min", "max", "range", "median", "skew"]


# -----------------------------------------------------------------------------
# Statistical functionals
# -----------------------------------------------------------------------------
def _functionals(mat: np.ndarray) -> np.ndarray:
    """Apply the functional set to each row of a (n_feat, T) matrix.

    Returns a flat 1-D array of length ``n_feat * len(_FUNC_NAMES)``.
    NaN / inf values are dropped per row before computing statistics.
    """
    mat = np.atleast_2d(np.asarray(mat, dtype=np.float64))
    out = []
    for row in mat:
        row = row[np.isfinite(row)]
        if row.size == 0:
            out.extend([0.0] * len(_FUNC_NAMES))
            continue
        rng = float(row.max() - row.min())
        sk = float(stats.skew(row)) if (row.size > 2 and row.std() > 1e-8) else 0.0
        out.extend([
            float(row.mean()), float(row.std()), float(row.min()),
            float(row.max()), rng, float(np.median(row)), sk,
        ])
    return np.asarray(out, dtype=np.float64)


def _func_names(prefix: str, n_rows: int):
    names = []
    for i in range(n_rows):
        tag = f"{prefix}{i}" if n_rows > 1 else prefix
        names.extend(f"{tag}_{f}" for f in _FUNC_NAMES)
    return names


# -----------------------------------------------------------------------------
# Frame-level feature matrices (spectral + energy)
# -----------------------------------------------------------------------------
def extract_frame_features(y: np.ndarray, sr: int = SR) -> dict:
    """Return an ordered dict of name -> (n_rows, T) feature matrices."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats = {
        "mfcc": mfcc,
        "mfcc_delta": librosa.feature.delta(mfcc),
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
        # fmin=100, n_bands=6 keeps the top band edge below the Nyquist freq.
        "contrast": librosa.feature.spectral_contrast(y=y, sr=sr, fmin=100.0, n_bands=6),
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "flatness": librosa.feature.spectral_flatness(y=y),
        "rms": librosa.feature.rms(y=y),                  # energy / intensity
        "zcr": librosa.feature.zero_crossing_rate(y),
    }
    return feats


# -----------------------------------------------------------------------------
# Prosodic pitch (F0) statistics
# -----------------------------------------------------------------------------
def extract_pitch_stats(y: np.ndarray, sr: int = SR):
    """F0 statistics over the *voiced* frames using probabilistic YIN."""
    names = ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_voiced_ratio"]
    try:
        f0, _, _ = librosa.pyin(y, fmin=65.0, fmax=2093.0, sr=sr)
    except Exception:
        return np.zeros(len(names)), names
    total = f0.size
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        return np.zeros(len(names)), names
    vals = [
        float(voiced.mean()), float(voiced.std()),
        float(voiced.min()), float(voiced.max()),
        float(voiced.max() - voiced.min()),
        float(voiced.size / max(total, 1)),
    ]
    return np.asarray(vals, dtype=np.float64), names


# -----------------------------------------------------------------------------
# Voice-quality features (jitter / shimmer / HNR) via Praat
# -----------------------------------------------------------------------------
def extract_voice_quality(path):
    """Return (values, names) or (None, names) if the Praat backend is absent."""
    names = ["jitter_local", "shimmer_local", "hnr_mean"]
    if not HAS_PRAAT:
        return None, names
    try:
        snd = parselmouth.Sound(str(path))
        pitch = snd.to_pitch()
        pp = call([snd, pitch], "To PointProcess (cc)")
        jitter = call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
        harm = snd.to_harmonicity()
        hv = harm.values[harm.values != -200.0]
        hnr = float(hv.mean()) if hv.size else 0.0
        vals = [jitter, shimmer, hnr]
        vals = [0.0 if (v is None or not np.isfinite(v)) else float(v) for v in vals]
        return np.asarray(vals, dtype=np.float64), names
    except Exception:
        return np.zeros(len(names), dtype=np.float64), names


# -----------------------------------------------------------------------------
# Top-level: one file -> one fixed-length feature vector
# -----------------------------------------------------------------------------
def extract_features(path, sr: int = SR, include_voice_quality: bool = True):
    """Extract the full hand-crafted feature vector for a single audio file.

    Returns (vector, feature_names). ``feature_names`` is deterministic for a
    given environment, so every file in a run yields the same vector length.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    # Rough endpoint detection: drop leading/trailing near-silence.
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size < sr // 10:                       # guard against empty/silent clips
        y = np.pad(y, (0, sr // 10 - y.size + 1))

    parts, names = [], []

    # Spectral + energy frame features -> functionals
    for name, mat in extract_frame_features(y, sr).items():
        mat = np.atleast_2d(mat)
        parts.append(_functionals(mat))
        names.extend(_func_names(name, mat.shape[0]))

    # Prosodic pitch statistics
    p_vals, p_names = extract_pitch_stats(y, sr)
    parts.append(p_vals)
    names.extend(p_names)

    # Voice-quality (only if Praat is available, kept consistent across a run)
    if include_voice_quality and HAS_PRAAT:
        vq_vals, vq_names = extract_voice_quality(path)
        if vq_vals is not None:
            parts.append(vq_vals)
            names.extend(vq_names)

    vec = np.concatenate(parts)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, names


if __name__ == "__main__":
    import sys
    v, n = extract_features(sys.argv[1])
    print(f"vector length = {len(v)} (praat={'on' if HAS_PRAAT else 'off'})")
    print("first 8 features:", list(zip(n[:8], np.round(v[:8], 4))))

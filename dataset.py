"""
dataset.py
==========
Discover audio files, parse their emotion labels, and turn the whole corpus
into an (X, y, groups) feature table.

Two labelling schemes are supported:

  * ``ravdess`` (default) -- label is read from the RAVDESS 7-field filename
    e.g. ``03-01-05-01-02-01-12.wav`` -> emotion field = "05" = angry,
    actor field = "12" (kept as the group id for speaker-aware splits).

  * ``folder`` -- label is the name of the directory the file sits in,
    e.g. ``data/happy/xxx.wav`` -> label "happy".
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from features import extract_features

# RAVDESS emotion code -> name
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def parse_ravdess(filename: str):
    """Return (emotion, actor) from a RAVDESS filename, or (None, None)."""
    parts = Path(filename).stem.split("-")
    if len(parts) != 7:
        return None, None
    return RAVDESS_EMOTIONS.get(parts[2]), parts[6]


def list_audio(data_dir, scheme: str = "ravdess"):
    """Walk ``data_dir`` and return a list of (path, emotion, group) tuples."""
    data_dir = Path(data_dir)
    items = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        if scheme == "ravdess":
            emo, actor = parse_ravdess(p.name)
            if emo is None:
                continue
            items.append((p, emo, actor))
        else:  # folder-per-class
            items.append((p, p.parent.name, "unknown"))
    return items


def build_dataset(data_dir, scheme: str = "ravdess", cache: str | None = None):
    """Extract features for every file. Returns (X, y, groups, feature_names).

    If ``cache`` points to an existing .npz file it is loaded instead of
    recomputing; otherwise the result is written there.
    """
    if cache and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        print(f"Loaded cached features from {cache}")
        return d["X"], d["y"], d["groups"], list(d["feature_names"])

    items = list_audio(data_dir, scheme)
    if not items:
        raise RuntimeError(
            f"No labelled audio found under '{data_dir}' (scheme='{scheme}')."
        )

    X, y, groups, names = [], [], [], None
    for path, emo, group in tqdm(items, desc="Extracting features"):
        try:
            vec, names = extract_features(path)
        except Exception as exc:  # keep going if one file is broken
            print(f"  ! skipping {path.name}: {exc}")
            continue
        X.append(vec)
        y.append(emo)
        groups.append(group)

    X = np.vstack(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if cache:
        np.savez_compressed(
            cache, X=X, y=y, groups=groups, feature_names=np.asarray(names)
        )
        print(f"Cached features -> {cache}")
    return X, y, groups, names

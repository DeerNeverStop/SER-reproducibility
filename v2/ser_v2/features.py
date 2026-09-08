"""Feature caches.

log-mel cache (exactly P1): librosa.load sr=22050 mono; trim top_db=30; pad to >=0.1 s; 64 mel,
n_fft 1024, hop 512, fmax 11025; power_to_db(ref=max); fixed 128 frames (right pad with the
spectrogram minimum / truncate); per-utterance z-score with eps 1e-6; float32 [64, 128].

SSL cache: torchaudio bundles HUBERT_BASE, WAVLM_BASE_PLUS, WAV2VEC2_BASE; 16 kHz mono; trim
30 dB; 13 hidden states (post-projection input + 12 transformer layers), time-mean pooled,
float16 [13, 768]. File names are content-addressed:
    <corpus>__<encoder>__sr16000-trim30-13state-time-mean__m<manifest sha[:8]>__w<weights sha[:8]>__c<code sha[:8]>.npz

Synthetic caches (dry run only) carry class and speaker structure so models learn something.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import atomic_write_json, sha256_file, sha256_text, stable_u32

LOGMEL = {"sr": 22050, "trim_top_db": 30.0, "minimum_seconds": 0.1, "n_mels": 64, "n_fft": 1024, "hop_length": 512,
          "fmax": 11025.0, "frames": 128, "eps": 1e-6}
SSL = {"sr": 16000, "trim_top_db": 30.0, "n_states": 13, "dim": 768, "pooling": "time_mean",
       "bundles": {"hubert_base": "HUBERT_BASE", "wavlm_base_plus": "WAVLM_BASE_PLUS", "wav2vec2_base": "WAV2VEC2_BASE"}}


def code_hash() -> str:
    return sha256_file(Path(__file__))[:8]


def logmel_from_wav(path: Path) -> np.ndarray:
    import librosa
    cfg = LOGMEL
    y, _ = librosa.load(str(path), sr=cfg["sr"], mono=True)
    y, _ = librosa.effects.trim(y, top_db=cfg["trim_top_db"])
    minimum = int(round(cfg["minimum_seconds"] * cfg["sr"]))
    if y.size < minimum:
        y = np.pad(y, (0, minimum - y.size))
    mel = librosa.feature.melspectrogram(y=y, sr=cfg["sr"], n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
                                         hop_length=cfg["hop_length"], fmax=cfg["fmax"])
    lm = librosa.power_to_db(mel, ref=np.max)
    frames = cfg["frames"]
    if lm.shape[1] < frames:
        lm = np.pad(lm, ((0, 0), (0, frames - lm.shape[1])), mode="constant", constant_values=float(lm.min()))
    else:
        lm = lm[:, :frames]
    lm = lm.astype(np.float32, copy=False)
    lm = (lm - lm.mean(dtype=np.float64)) / (lm.std(dtype=np.float64) + cfg["eps"])
    return np.asarray(lm, dtype=np.float32)


def build_logmel_cache(corpus: str, root: Path, manifest_rows: list[dict], out_dir: Path) -> Path:
    paths = [r["relative_path"] for r in manifest_rows]
    X = np.stack([logmel_from_wav(root / p) for p in paths]).astype(np.float32)
    msha = sha256_text("\n".join(paths))[:8]
    name = f"{corpus}__logmel__sr22050-mel64-nfft1024-hop512-fmax11025-frames128-trim30-zscore__m{msha}__c{code_hash()}.npz"
    out = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, paths=np.asarray(paths))
    atomic_write_json(out.with_suffix(".json"), {"corpus": corpus, "n": len(paths), "config": LOGMEL, "sha256": sha256_file(out)})
    return out


def build_ssl_cache(corpus: str, encoder: str, root: Path, manifest_rows: list[dict], out_dir: Path, device: str = "cuda") -> Path:
    import torch
    import torchaudio
    bundle = getattr(torchaudio.pipelines, SSL["bundles"][encoder])
    model = bundle.get_model().to(device).eval()
    wsha = hashlib.sha256()
    for _, t in sorted(model.state_dict().items()):
        wsha.update(t.detach().cpu().numpy().tobytes())
    paths = [r["relative_path"] for r in manifest_rows]
    feats = np.zeros((len(paths), SSL["n_states"], SSL["dim"]), dtype=np.float16)
    import librosa
    with torch.no_grad():
        for i, p in enumerate(paths):
            y, _ = librosa.load(str(root / p), sr=SSL["sr"], mono=True)
            y, _ = librosa.effects.trim(y, top_db=SSL["trim_top_db"])
            wav = torch.from_numpy(y).float().unsqueeze(0).to(device)
            states, _ = model.extract_features(wav)          # 12 transformer outputs
            proj = model.encoder.feature_projection(model.feature_extractor(wav, None)[0])  # [1, T, 768] post-projection input (torchaudio: Encoder.feature_projection)
            hs = [proj] + list(states)
            feats[i] = torch.stack([h.mean(dim=1)[0] for h in hs]).to(torch.float16).cpu().numpy()
    msha = sha256_text("\n".join(paths))[:8]
    name = f"{corpus}__{encoder}__sr16000-trim30-13state-time-mean__m{msha}__w{wsha.hexdigest()[:8]}__c{code_hash()}.npz"
    out = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=feats, paths=np.asarray(paths))
    atomic_write_json(out.with_suffix(".json"), {"corpus": corpus, "encoder": encoder, "n": len(paths), "config": SSL,
                                                 "weights_sha256": wsha.hexdigest(), "sha256": sha256_file(out)})
    return out


def synthetic_logmel(manifest_rows: list[dict], seed_text: str = "synthetic-logmel") -> np.ndarray:
    """[N, 64, 128] with a class-specific spectral template plus a speaker offset plus noise."""
    n = len(manifest_rows)
    X = np.zeros((n, LOGMEL["n_mels"], LOGMEL["frames"]), dtype=np.float32)
    mel_axis = np.linspace(0, 1, LOGMEL["n_mels"])[:, None]
    for i, r in enumerate(manifest_rows):
        rng = np.random.RandomState(stable_u32(f"{seed_text}|{r['relative_path']}"))
        c = int(r["label_index"])
        spk = np.random.RandomState(stable_u32(f"{seed_text}|spk|{r['speaker']}")).normal(0, 0.6, size=(LOGMEL["n_mels"], 1))
        template = np.sin(2 * np.pi * (c + 1) * mel_axis) * 1.0
        X[i] = template + spk + rng.normal(0, 1.0, size=X[i].shape)
        X[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-6)
    return X


def synthetic_ssl(manifest_rows: list[dict], seed_text: str = "synthetic-ssl") -> np.ndarray:
    n = len(manifest_rows)
    X = np.zeros((n, SSL["n_states"], SSL["dim"]), dtype=np.float16)
    for i, r in enumerate(manifest_rows):
        rng = np.random.RandomState(stable_u32(f"{seed_text}|{r['relative_path']}"))
        c = int(r["label_index"])
        cls = np.random.RandomState(stable_u32(f"{seed_text}|cls|{c}")).normal(0, 1, size=SSL["dim"])
        spk = np.random.RandomState(stable_u32(f"{seed_text}|spk|{r['speaker']}")).normal(0, 1, size=SSL["dim"])
        base = 0.8 * cls + 0.8 * spk + rng.normal(0, 1.5, size=SSL["dim"])
        for s in range(SSL["n_states"]):
            X[i, s] = (base + 0.1 * s * rng.normal(0, 1, size=SSL["dim"])).astype(np.float16)
    return X


def write_synthetic_caches(corpus: str, manifest_rows: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = np.asarray([r["relative_path"] for r in manifest_rows])
    written = {}
    lm = out_dir / f"{corpus}__logmel__SYNTHETIC.npz"
    np.savez(lm, X=synthetic_logmel(manifest_rows), paths=paths)
    written["logmel"] = str(lm)
    ssl = synthetic_ssl(manifest_rows)
    for enc in SSL["bundles"]:
        p = out_dir / f"{corpus}__{enc}__SYNTHETIC.npz"
        np.savez(p, X=ssl, paths=paths)
        written[enc] = str(p)
    return written


class FeatureStore:
    """Resolve (corpus, kind) -> arrays indexed by relative path."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, tuple[np.ndarray, dict[str, int]]] = {}

    def load(self, corpus: str, kind: str) -> tuple[np.ndarray, dict[str, int]]:
        key = f"{corpus}__{kind}"
        if key not in self._cache:
            cands = sorted(self.root.glob(f"{corpus}__{kind}__*.npz"))
            if not cands:
                raise FileNotFoundError(f"no feature cache for {key} under {self.root}")
            z = np.load(cands[-1], allow_pickle=False)
            X, paths = z["X"], [str(p) for p in z["paths"]]
            self._cache[key] = (X, {p: i for i, p in enumerate(paths)})
        return self._cache[key]

    def get(self, corpus: str, kind: str, paths: list[str], state: int | None = None) -> np.ndarray:
        X, index = self.load(corpus, kind)
        idx = np.asarray([index[p] for p in paths])
        arr = X[idx]
        if state is not None:
            arr = arr[:, state, :]
        return np.asarray(arr, dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, help="corpus audio root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=["logmel", "ssl", "synthetic"], required=True)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from .corpora import load_manifest, apply_hygiene
    rows, _ = apply_hygiene("subesco" if a.corpus.startswith("subesco") else a.corpus, load_manifest(a.manifest))
    if a.kind == "synthetic":
        print(json.dumps(write_synthetic_caches(a.corpus, rows, a.out)))
    elif a.kind == "logmel":
        print(build_logmel_cache(a.corpus, a.root, rows, a.out))
    else:
        print(build_ssl_cache(a.corpus, a.encoder, a.root, rows, a.out, a.device))


if __name__ == "__main__":
    main()

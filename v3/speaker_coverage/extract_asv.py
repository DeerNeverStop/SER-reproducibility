"""Extract frozen SpeechBrain embeddings from verified real corpus audio.

No SER training or scientific arm scoring. Download the official model separately,
pin its revision, and supply its local snapshot. Every source waveform is hashed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--model-id", choices=("speechbrain/spkrec-ecapa-voxceleb",
                                         "speechbrain/spkrec-xvect-voxceleb"), required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    assert len(args.revision) == 40 and all(c in "0123456789abcdef" for c in args.revision)
    assert args.batch_size > 0 and args.threads > 0
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from v3.data_design.core_plan import read_metadata

    meta = read_metadata(repo)
    rows = sorted(meta["clean"], key=lambda r: r["relative_path"])
    model_dir = args.model_dir.resolve()
    assert (model_dir / "hyperparams.yaml").is_file()
    model_sha = {p.relative_to(model_dir).as_posix(): sha(p) for p in sorted(model_dir.rglob("*"))
                 if p.is_file() and ".cache" not in p.parts and p.suffix in {".ckpt", ".yaml", ".txt"}}
    assert "embedding_model.ckpt" in model_sha
    identity = {"schema": "ser-asv-embeddings-1", "corpus": "cremad", "model_id": args.model_id,
                "model_revision": args.revision, "model_sha256": model_sha,
                "manifest_sha256": meta["input"]["manifest_sha256"], "rows": len(rows),
                "preprocessing": "soundfile float32; channel mean; scipy polyphase resample to 16000; no trim or loudness change",
                "embedding": "SpeechBrain EncoderClassifier.encode_batch normalize=False; eval mode",
                "batch_size": args.batch_size, "source_sha256": sha(Path(__file__))}
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.with_suffix(".json").exists():
        assert out.is_file() and out.with_suffix(".json").is_file(), "incomplete existing embedding artifact"
        old = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert old["identity"] == identity and old["sha256"] == sha(out), "existing artifact identity differs"
        print(json.dumps({"status": "verified_existing", "rows": len(rows), "sha256": old["sha256"]}))
        return

    torch.set_num_threads(args.threads)
    torch.manual_seed(20260906)
    np.random.seed(20260906)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    classifier = EncoderClassifier.from_hparams(
        source=str(model_dir), savedir=str(model_dir),
        run_opts={"device": args.device}, overrides={"pretrained_path": str(model_dir)})
    classifier.eval()
    vectors, audio_quality = [], []
    audio_root = args.audio_root.resolve()
    source_bytes, duration_total = 0, 0.0
    for start in range(0, len(rows), args.batch_size):
        block = rows[start:start + args.batch_size]
        waves = []
        for row in block:
            path = (audio_root / row["relative_path"]).resolve()
            assert path.is_relative_to(audio_root) and path.is_file()
            assert sha(path) == row["sha256"], f"audio byte mismatch: {row['relative_path']}"
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            assert len(wav) > 0 and np.isfinite(wav).all() and sr > 0
            wav = wav.mean(axis=1)
            duration = len(wav) / sr
            audio_quality.append({"path": row["relative_path"], "sha256": row["sha256"],
                                  "seconds": duration, "original_sr": sr,
                                  "rms": float(np.sqrt(np.mean(np.square(wav.astype(np.float64))))),
                                  "clipped_fraction": float(np.mean(np.abs(wav) >= .999))})
            if sr != 16000:
                gcd = math.gcd(sr, 16000)
                wav = resample_poly(wav, 16000 // gcd, sr // gcd).astype(np.float32)
            waves.append(torch.from_numpy(np.ascontiguousarray(wav)))
            source_bytes += path.stat().st_size
            duration_total += duration
        lengths = torch.tensor([len(w) for w in waves], dtype=torch.float32)
        batch = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        lengths = lengths / batch.shape[1]
        with torch.inference_mode():
            emb = classifier.encode_batch(batch.to(args.device), lengths.to(args.device), normalize=False)
        block_emb = emb.squeeze(1).float().cpu().numpy()
        assert block_emb.ndim == 2 and len(block_emb) == len(block)
        assert np.isfinite(block_emb).all() and (np.linalg.norm(block_emb, axis=1) > 0).all()
        vectors.append(block_emb)
        if start % (args.batch_size * 32) == 0:
            print(json.dumps({"event": "embedding_progress", "completed": start + len(block),
                              "total": len(rows), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    values = np.concatenate(vectors)
    assert len(values) == len(rows)
    temp = out.with_suffix(".tmp.npz")
    np.savez_compressed(temp, paths=np.asarray([r["relative_path"] for r in rows]), embeddings=values)
    temp.replace(out)
    quality_path = out.with_name(out.stem + "_audio_quality.csv")
    with quality_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audio_quality[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audio_quality)
    receipt = {"identity": identity, "sha256": sha(out), "shape": list(values.shape),
               "source_audio_bytes_verified": source_bytes, "source_audio_seconds": duration_total,
               "elapsed_seconds": elapsed, "extraction_RTF": elapsed / duration_total,
               "audio_quality_file": quality_path.name, "audio_quality_sha256": sha(quality_path),
               "versions": {p: importlib.metadata.version(p) for p in
                            ("torch", "torchaudio", "speechbrain", "numpy", "scipy", "soundfile")},
               "device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
               "scientific_SER_scores_computed": False}
    atomic_json(out.with_suffix(".json"), receipt)
    print(json.dumps({"status": "complete", "rows": len(rows), "shape": list(values.shape),
                      "elapsed_seconds": elapsed, "sha256": receipt["sha256"]}))


if __name__ == "__main__":
    main()

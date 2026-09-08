"""P4-D1: build the two per-channel log-mel feature caches.

Gate-8 discipline: feature extraction is NOT reimplemented -- every sample is
processed by the frozen `p1_protocol_core.compute_logmel` (imported, not
copied), so the D1 front-end is bit-identical in code to P1's.

Index contract (critical): cache row i holds the features of pair_manifest.csv
row i for the given channel. The 99 anchored split NPZs index into exactly
this order. The cache signature embeds the pair_manifest SHA-256 to pin that
contract.

Fail-closed: every wav is SHA-256-verified against pair_manifest.csv BEFORE
feature extraction; any mismatch aborts with no cache written.

Outputs (E drive only, per assignment section 3):
  E:\claudework_data\ICASSP2027-corpora\ravdess-song\cache\d1_logmel_song.npz
  E:\claudework_data\ICASSP2027-corpora\ravdess-speech-common6\cache\d1_logmel_speech.npz

Note: speech wavs are read in place from the read-only P1 source tree
(E:\科研\SER\data), anchored per-file by SHA; they are not duplicated into
ravdess-speech-common6\ (only the derived cache lives there). This keeps a
single authoritative copy of the audio while still pinning provenance.
"""
import csv
import os
import sys
from pathlib import Path

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import p1_protocol_core as core  # noqa: E402  (read-only reuse)

PAIR_MANIFEST = Path(r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched\pair_manifest.csv")
SONG_ROOT = Path(r"E:\claudework_data\ICASSP2027-corpora\ravdess-song")
SPEECH_ROOT = Path(r"E:\科研\SER\data")
OUT = {
    "song": Path(r"E:\claudework_data\ICASSP2027-corpora\ravdess-song\cache\d1_logmel_song.npz"),
    "speech": Path(r"E:\claudework_data\ICASSP2027-corpora\ravdess-speech-common6\cache\d1_logmel_speech.npz"),
}
ROOT = {"song": SONG_ROOT, "speech": SPEECH_ROOT}


def main():
    import numpy as np

    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1012:
        raise RuntimeError(f"pair_manifest rows {len(rows)} != 1012")
    pair_sha = core.sha256_file(PAIR_MANIFEST)

    for channel in ("song", "speech"):
        out = OUT[channel]
        signature = {
            "channel": channel,
            "pair_manifest_sha256": pair_sha,
            "preprocess": core.PREPROCESS_CONFIG,
            "preprocessing_code_hash": core.preprocessing_code_hash(),
            "index_contract": "row i == pair_manifest.csv row i",
        }
        if out.exists():
            with np.load(out, allow_pickle=False) as data:
                observed = str(data["signature_json"].item())
                if observed != core.canonical_json(signature):
                    raise RuntimeError(f"existing cache signature drift: {out}")
                if tuple(data["X"].shape) != (1012, 64, 128):
                    raise RuntimeError(f"existing cache shape drift: {out}")
            print(f"[{channel}] cache exists and signature matches: {out}")
            print(f"[{channel}] sha256 {core.sha256_file(out)}")
            continue

        # fail-closed provenance check on every file BEFORE any extraction
        for i, row in enumerate(rows):
            path = ROOT[channel] / row[f"{channel}_relative_path"]
            if core.sha256_file(path) != row[f"{channel}_sha256"]:
                raise RuntimeError(f"wav SHA drift (fail-closed, no cache written): {path}")
            if (i + 1) % 200 == 0 or i + 1 == 1012:
                print(f"[{channel}] verified {i + 1}/1012", flush=True)

        X = np.empty((1012, 64, 128), dtype=np.float32)
        for i, row in enumerate(rows):
            X[i] = core.compute_logmel(ROOT[channel] / row[f"{channel}_relative_path"])
            if (i + 1) % 100 == 0 or i + 1 == 1012:
                print(f"[{channel}] extracted {i + 1}/1012", flush=True)
        if not np.isfinite(X).all():
            raise RuntimeError(f"{channel}: non-finite features")
        core.atomic_save_npz(out, X=X, signature_json=np.asarray(core.canonical_json(signature)))
        print(f"[{channel}] wrote {out}")
        print(f"[{channel}] sha256 {core.sha256_file(out)}")


if __name__ == "__main__":
    main()

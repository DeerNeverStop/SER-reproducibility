"""Corpus specifications, filename parsers, byte manifests, and synthetic manifests.

Manifest columns (one row per utterance):
  sample_index, corpus, relative_path, bytes, sha256, speaker, sex, label, label_index,
  sentence, take, intensity
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import sha256_file, stable_u32, write_csv, read_csv

MANIFEST_FIELDS = [
    "sample_index", "corpus", "relative_path", "bytes", "sha256", "speaker", "sex",
    "label", "label_index", "sentence", "take", "intensity",
]


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    labels: tuple[str, ...]
    expected_utterances: int | None
    expected_speakers: int | None
    n_sentences: int | None
    notes: str = ""
    license: str = ""


RAVDESS_EMOTIONS = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
CREMAD_CODES = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CREMAD_EMOTIONS = ("angry", "disgust", "fearful", "happy", "neutral", "sad")   # P1 label_index order
SUBESCO_EMOTIONS = ("ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE")

CORPORA: dict[str, CorpusSpec] = {
    "ravdess": CorpusSpec("ravdess", RAVDESS_EMOTIONS, 1440, 24, 2,
                          "Speech only (modality 03, vocal channel 01); 2 statements x 2 reps x 2 intensities.",
                          "CC BY-NC-SA 4.0 (Zenodo 10.5281/zenodo.1188976)"),
    "cremad": CorpusSpec("cremad", CREMAD_EMOTIONS, 7442, 91, 12,
                         "12 sentences x 6 emotions; intensity levels LO/MD/HI/XX (siblings only for IEO).",
                         "ODbL 1.0 / DbCL 1.0"),
    "subesco": CorpusSpec("subesco", SUBESCO_EMOTIONS, 7000, 20, 10,
                          "20 speakers x 10 sentences x 7 emotions x 5 takes.",
                          "see SUBESCO record (PLOS ONE 2021); pin DOI and license text at tag-1"),
}

_RAVDESS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$")
_CREMAD_RE = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{1,2})\.wav$")
_SUBESCO_RE = re.compile(r"^([FM])_(\d{2})_([A-Z]+)_S_?(\d{1,2})_([A-Z]+)_(\d+)\]?\.wav$")


def parse_filename(corpus: str, filename: str) -> dict | None:
    """Return speaker/sex/label/sentence/take/intensity or None when the file is not in scope."""
    name = Path(filename).name
    if corpus == "ravdess":
        m = _RAVDESS_RE.match(name)
        if not m:
            return None
        modality, channel, emotion, intensity, statement, repetition, actor = m.groups()
        if modality != "03" or channel != "01":
            return None  # audio-only speech
        actor_i = int(actor)
        return {
            "speaker": f"{actor_i:02d}", "sex": "M" if actor_i % 2 == 1 else "F",
            "label": RAVDESS_EMOTIONS[int(emotion) - 1], "sentence": f"S{int(statement)}",
            "take": f"R{int(repetition)}", "intensity": {"01": "normal", "02": "strong"}[intensity],
        }
    if corpus == "cremad":
        m = _CREMAD_RE.match(name)
        if not m:
            return None
        actor, sentence, emotion, level = m.groups()
        if emotion not in CREMAD_CODES:
            return None
        return {"speaker": actor, "sex": "", "label": CREMAD_CODES[emotion], "sentence": sentence,
                "take": level, "intensity": level}
    if corpus == "subesco":
        m = _SUBESCO_RE.match(name)
        if not m:
            return None
        sex, num, _speaker_name, sentence, emotion, take = m.groups()
        if emotion not in SUBESCO_EMOTIONS:
            return None
        return {"speaker": f"{sex}{int(num):02d}", "sex": sex, "label": emotion,
                "sentence": f"S{int(sentence)}", "take": f"T{int(take)}", "intensity": ""}
    raise ValueError(f"unknown corpus {corpus}")


def build_manifest(corpus: str, root: Path, out_csv: Path, hash_files: bool = True) -> list[dict]:
    """Walk `root`, parse every in-scope WAV, and write a byte manifest sorted by relative path."""
    spec = CORPORA[corpus]
    label_index = {lab: i for i, lab in enumerate(spec.labels)}
    rows = []
    for path in sorted(root.rglob("*.wav")):
        parsed = parse_filename(corpus, path.name)
        if parsed is None:
            continue
        rel = path.relative_to(root).as_posix()
        rows.append({
            "corpus": corpus, "relative_path": rel, "bytes": path.stat().st_size,
            "sha256": sha256_file(path) if hash_files else "",
            **parsed, "label_index": label_index[parsed["label"]],
        })
    rows.sort(key=lambda r: r["relative_path"])
    for i, r in enumerate(rows):
        r["sample_index"] = i
    write_csv(out_csv, MANIFEST_FIELDS, rows)
    return rows


def load_manifest(path: Path | str) -> list[dict]:
    rows = read_csv(path)
    for r in rows:
        r["sample_index"] = int(r["sample_index"])
        r["label_index"] = int(r["label_index"])
        r["bytes"] = int(r["bytes"]) if r.get("bytes") else 0
    return rows


def manifest_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "y": np.asarray([r["label_index"] for r in rows], dtype=np.int64),
        "speaker": np.asarray([r["speaker"] for r in rows]),
        "sentence": np.asarray([r["sentence"] for r in rows]),
        "take": np.asarray([r["take"] for r in rows]),
        "sex": np.asarray([r.get("sex", "") for r in rows]),
        "cell": np.asarray([f"{r['speaker']}|{r['sentence']}|{r['label']}" for r in rows]),
    }


def validate_manifest(corpus: str, rows: list[dict]) -> dict:
    """Return a hygiene report; raise on hard violations (duplicate paths, unknown labels)."""
    spec = CORPORA[corpus]
    paths = [r["relative_path"] for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate relative_path in manifest")
    labels = {r["label"] for r in rows}
    unknown = labels - set(spec.labels)
    if unknown:
        raise ValueError(f"unknown labels {sorted(unknown)}")
    by_sha: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("sha256"):
            by_sha.setdefault(r["sha256"], []).append(r)
    dup_groups = [g for g in by_sha.values() if len(g) > 1]
    conflicting = [g for g in dup_groups if len({x["label"] for x in g}) > 1]
    report = {
        "corpus": corpus, "n_utterances": len(rows), "n_speakers": len({r["speaker"] for r in rows}),
        "n_sentences": len({r["sentence"] for r in rows}), "labels": sorted(labels),
        "expected_utterances": spec.expected_utterances, "expected_speakers": spec.expected_speakers,
        "exact_duplicate_groups": len(dup_groups), "label_conflicting_duplicate_groups": len(conflicting),
        "count_matches_expected": (spec.expected_utterances is None or len(rows) == spec.expected_utterances),
    }
    return report


def synthetic_manifest(corpus: str, n_speakers: int, n_sentences: int, n_takes: int,
                       labels: tuple[str, ...] | None = None, seed_text: str = "synthetic") -> list[dict]:
    """Deterministic synthetic manifest with the real column layout (no audio, sha256 = hash of name)."""
    spec = CORPORA.get(corpus)
    labs = labels or (spec.labels if spec else ("a", "b", "c", "d"))
    rows = []
    for s in range(n_speakers):
        sex = "M" if s % 2 == 0 else "F"
        for sent in range(n_sentences):
            for li, lab in enumerate(labs):
                for t in range(n_takes):
                    rel = f"spk{s:02d}/{corpus}_s{sent:02d}_{lab}_t{t}.wav"
                    rows.append({
                        "corpus": corpus, "relative_path": rel, "bytes": 1000 + stable_u32(rel) % 1000,
                        "sha256": f"{stable_u32(seed_text + rel):08x}" * 8, "speaker": f"{s:02d}", "sex": sex,
                        "label": lab, "label_index": li, "sentence": f"S{sent + 1}", "take": f"T{t + 1}",
                        "intensity": "",
                    })
    rows.sort(key=lambda r: r["relative_path"])
    for i, r in enumerate(rows):
        r["sample_index"] = i
    return rows


# Preregistered hygiene (applied to every real population before any split is generated):
#  H1 exact-byte duplicate groups with conflicting labels: drop every member;
#  H2 exact-byte duplicate groups with one label: keep the lexically first path;
#  H3 registered unusable files (verified on the audio at PREP; provenance in the registry).
KNOWN_UNUSABLE = {"cremad": ["1076_MTI_SAD_XX.wav"]}   # zero-variance audio reported by the 2026 A-design hygiene


def apply_hygiene(corpus: str, rows: list[dict]) -> tuple[list[dict], dict]:
    by_sha: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("sha256"):
            by_sha.setdefault(r["sha256"], []).append(r)
    drop: dict[str, str] = {}
    for sha, group in by_sha.items():
        if len(group) < 2:
            continue
        if len({g["label"] for g in group}) > 1:
            for g in group:
                drop[g["relative_path"]] = f"H1 label-conflicting duplicate group {sha[:12]}"
        else:
            for g in sorted(group, key=lambda x: x["relative_path"])[1:]:
                drop[g["relative_path"]] = f"H2 exact duplicate of lexically first path ({sha[:12]})"
    for name in KNOWN_UNUSABLE.get(corpus, []):
        for r in rows:
            if Path(r["relative_path"]).name == name:
                drop[r["relative_path"]] = "H3 registered unusable file"
    kept = [dict(r) for r in rows if r["relative_path"] not in drop]
    kept.sort(key=lambda r: r["relative_path"])
    for i, r in enumerate(kept):
        r["sample_index"] = i
    log = {"corpus": corpus, "n_input": len(rows), "n_kept": len(kept), "n_dropped": len(drop),
           "dropped": [{"relative_path": k, "reason": v} for k, v in sorted(drop.items())]}
    return kept, log

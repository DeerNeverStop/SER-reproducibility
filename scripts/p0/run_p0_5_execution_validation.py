#!/usr/bin/env python3
"""Evidence-strength execution validation for the frozen P0.5 sample.

Safety boundary: this program never imports or executes candidate repository code.
It byte-reads frozen sources, re-implements only the decisive split operation, and
feeds that operation a synthetic manifest with explicit speaker IDs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle as sklearn_shuffle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "sources_p0_5"
RESULT_ROOT = ROOT / "results" / "execution_validation"
PER_REPO = RESULT_ROOT / "per_repository"

EXPECTED_INPUTS = {
    "p0_5_random_survey.csv": "d8745960363fff38aac7081b28fceee04238d79f971508d7c34c28b2183899e9",
    "p0_5_random_order.csv": "8df60755279887a72c288e161373d30fef6542d82b0a2828172d36a1beec5626",
    "P0_5_PROTOCOL.md": "210969e24e7b250dda0af3d5479754bc86888d3110bb8d410c50f4d65a669481",
    "results/p0_5/random_repository_risk.csv": "b0f1c1ca7a9d5def32619697c2d8b0f0087c95de45ce529e1f3e5faed7dfad3a",
}

TEXT_CODE_SUFFIXES = {
    ".py", ".ipynb", ".r", ".m", ".jl", ".sh", ".ps1", ".bat",
    ".cmd", ".yaml", ".yml", ".toml", ".json", ".md", ".txt",
}


@dataclass(frozen=True)
class RepoSpec:
    repository_id: str
    rank: int
    source_dir: str
    tier: str
    expected: str
    method: str
    source_refs: tuple[str, ...]
    rationale: str
    params: dict[str, Any]


def spec(
    rid: str,
    rank: int,
    source_dir: str,
    tier: str,
    expected: str,
    method: str,
    refs: Iterable[str],
    rationale: str,
    **params: Any,
) -> RepoSpec:
    return RepoSpec(rid, rank, source_dir, tier, expected, method, tuple(refs), rationale, params)


SPECS = [
    spec("P05R001", 1, "0001__Vansh-187__speech-emotion-recognition", "tier1_execution_confirmed", "yes", "random_rows", ["notebooks/emotion_classifier_final.ipynb:L510-L522"], "Feature rows are split by ordinary train_test_split; speaker is not a split key.", corpora=["RAVDESS"], test_size=0.2, seed=0, stratify=False),
    spec("P05R002", 2, "0002__fdhlhdc12__ravdess-dtw-emotion-recognition", "tier3_unknown", "unknown", "not_executable_no_split_source", ["app.py:L733-L752", "feature_extraction_ml.py:L1-L47"], "All executable Python was byte-read. The app loads serialized models/scaler, while the frozen tree contains no training or split constructor; pickle contents cannot recover the original split membership."),
    spec("P05R003", 4, "0004__Shiva9855__CodeAlpha_Emotion-Recognition-from-Speech", "tier1_execution_confirmed", "yes", "random_rows", ["config.py:L29-L36", "train.py:L148-L159"], "The pooled target-corpus feature rows are stratified only by emotion; speaker is absent from the call.", corpora=["RAVDESS", "EmoDB"], test_size=0.2, seed=42, stratify=True),
    spec("P05R004", 5, "0005__dingdongwang__EmotionThinker", "tier3_unknown", "unknown", "not_executable_inference_only", ["scripts/emotionthinker_infer.py:L1-L39"], "The frozen repository exposes model download/inference only and no IEMOCAP/RAVDESS training manifest or split path. Running it would require network/model execution and still would not reveal the absent training split."),
    spec("P05R005", 6, "0006__jobinjoy12__Speech-Emotion-Recognition-using-CREMA-D-dataset", "tier1_execution_confirmed", "yes", "random_rows", ["emotion speech recognition code (2).ipynb:L1 (cell 15)"], "CREMA-D feature rows are passed to ordinary shuffled train_test_split without actor grouping.", corpora=["CREMA-D"], test_size=0.2, seed=0, stratify=False),
    spec("P05R006", 8, "0008__ShaheenPerveen__Speech_Emotion_Recognition_IEMOCAP", "tier1_execution_confirmed", "yes", "random_rows", ["Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1563-L1572"], "The representative normalized-MFCC path stratifies utterance rows only by target label; no IEMOCAP speaker/session group enters the split. The source leaves random_state unset, so the isolated probe fixes 20260813 only to make the witness reproducible.", corpora=["IEMOCAP"], test_size=0.2, seed=20260813, source_random_state="unset", stratify=True),
    spec("P05R007", 9, "0009__sss3799__audio_emotion_recognition", "tier1_execution_confirmed", "yes", "random_rows", ["hsl621_emotion_recognition.ipynb:L1 (cell 39)"], "The notebook uses shuffled train_test_split on pooled examples with random_state=0 and no speaker key.", corpora=["RAVDESS"], test_size=0.25, seed=0, stratify=False),
    spec("P05R008", 10, "0010__neerajk23-cmd__Speech-Emotion-Recognition-on-RAVDESS-Dataset", "tier1_execution_confirmed", "no", "fixed_speakers", ["cnn-final-ee708-project.ipynb:L1 (cells 5-6)", "Evaluation_Test_Data.ipynb:L1 (cells 3,6,8)"], "The reported outer evaluation holds out actors 20-24; the random inner validation is not the final evaluation pair.", corpus="RAVDESS", train_speakers=[f"RAVDESS:{i:02d}" for i in range(1, 20)], test_speakers=[f"RAVDESS:{i:02d}" for i in range(20, 25)]),
    spec("P05R009", 12, "0012__shreyash-alt__Emotion_Recognition_RAVDESS", "tier1_execution_confirmed", "yes", "random_rows", ["emotionravdess2.ipynb:L537-L560"], "RAVDESS examples are split by ordinary label-stratified train_test_split, not actor.", corpora=["RAVDESS"], test_size=0.2, seed=42, stratify=True),
    spec("P05R010", 13, "0013__KangHyunWook__Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset", "tier1_execution_confirmed", "yes", "shuffled_fixed_slices", ["train.py:L17-L41", "train.py:L119-L139"], "preprocess shuffles rows without a frozen seed and then takes fixed row slices; speaker is never consulted.", corpus="RAVDESS", probe_seed=20260813),
    spec("P05R011", 14, "0014__AnkushMalaker__pretrained-dcnn-attention-ser", "tier1_execution_confirmed", "yes", "random_rows", ["utils.py:L73-L88"], "The only active target split (RAVDESS) is ordinary 90/10 train_test_split on filenames/labels without actor grouping; other named target paths are absent.", corpora=["RAVDESS"], test_size=0.1, seed=20260813, stratify=False),
    spec("P05R012", 15, "0015__dori2063__SER_Augmentation_CycleGAN", "tier3_unknown", "unknown", "missing_external_partition_generator", ["run.py:L38-L45", "run.py:L140-L184"], "The code consumes an external section_list and pre-generated arrays, but the frozen tree omits the list contents/generator that maps IEMOCAP speakers to sections. Ten-fold labels alone do not identify whether speakers cross partitions."),
    spec("P05R013", 16, "0016__sanskardherange__codeAlpha_EmotionRecognition", "tier1_execution_confirmed", "yes", "random_rows", ["extract_features.py:L17-L35", "train_model.py:L18-L31"], "Actor identity is dropped during feature-table construction and the remaining rows receive ordinary 80/20 train_test_split.", corpora=["RAVDESS"], test_size=0.2, seed=42, stratify=False),
    spec("P05R014", 17, "0017__didar-ali-deed__A-hybrid-Approach-To-SER", "tier1_execution_confirmed", "yes", "random_rows", ["scripts/train_emotion_classifier.py:L52-L60"], "The combined RAVDESS/CREMA-D feature dataset is split by row twice, with no speaker field supplied.", corpora=["RAVDESS", "CREMA-D"], test_size=0.2, seed=42, stratify=False),
    spec("P05R015", 18, "0018__fdebrain__Speech-Emotion-Recognition-Emo-DB", "tier1_execution_confirmed", "no", "random_whole_speakers", ["modules/formatter.py:L18-L33", "modules/formatter.py:L47-L64", "app.py:L12-L14"], "Speaker is parsed from each filename, two complete speaker IDs are sampled, and row membership is filtered by that speaker set. Random selection of groups is still speaker-disjoint.", corpus="EmoDB", seed=42, n_test_speakers=2),
    spec("P05R016", 19, "0019__Susi-brambilla__emotion-recognition-RAVDESS", "tier1_execution_confirmed", "no", "fixed_speakers", ["train_val_set.py:L51-L60", "train_val_set.py:L105-L114"], "Actors 23-24 are assigned wholly to test, actors 21-22 to validation, and the remaining actors to training.", corpus="RAVDESS", train_speakers=[f"RAVDESS:{i:02d}" for i in range(1, 21)], test_speakers=["RAVDESS:23", "RAVDESS:24"]),
    spec("P05R017", 20, "0020__Omkar-Gode__Speech-Emotion-Recognition", "tier1_execution_confirmed", "yes", "augmented_random_rows", ["Speech_Emotion_Recognition.ipynb:L1946-L1952", "Speech_Emotion_Recognition.ipynb:L2640-L2687"], "Multiple rows derived from the same CREMA-D utterance are pooled before an ordinary random row split; neither source utterance nor actor is grouped.", corpus="CREMA-D", variants=3, test_size=0.2, seed=42),
    spec("P05R018", 22, "0022__daribdb__speech-emotion-recognition-iemocap", "tier1_execution_confirmed", "no", "loso_speakers", ["iemocap_emotion_recognition_csv.ipynb:L793-L841"], "Each fold holds out one complete IEMOCAP speaker for test and a distinct speaker for validation; both are excluded from train.", corpus="IEMOCAP"),
    spec("P05R019", 23, "0023__Yaowan410__Emotion-Aware-subtitles", "tier1_execution_confirmed", "yes", "random_rows", ["Model.py:L90-L101"], "IEMOCAP row indices are stratified by emotion only; speaker/session IDs do not enter train_test_split.", corpora=["IEMOCAP"], test_size=0.1, seed=42, stratify=True),
    spec("P05R020", 24, "0024__Mayureshkore07__Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning", "tier1_execution_confirmed", "no", "random_whole_speakers_70_15_15", ["notebooks/01_Data_Preparation/04_dataset_split.ipynb:L232-L286"], "The notebook first splits the unique speaker list and then selects all rows by speaker, with explicit disjointness assertions.", corpora=["RAVDESS", "CREMA-D"], seed=42),
    spec("P05R021", 25, "0025__VeerR13__Speech-Emotion-Recognition", "tier1_execution_confirmed", "yes", "random_rows", ["SER_v2_kaggle.ipynb:L191"], "The pooled RAVDESS/CREMA-D table is stratified by emotion into row-level train/validation/test sets; actor is not a grouping argument.", corpora=["RAVDESS", "CREMA-D"], test_size=0.075, seed=42, stratify=True),
    spec("P05R022", 26, "0026__yipenglai__Audio-Emotion-Classification", "tier1_execution_confirmed", "yes", "stratified_kfold_rows", ["crnn/train.ipynb:L4686-L4706"], "Four-fold StratifiedKFold receives labels but no actor groups, so each fold partitions utterance rows.", corpus="RAVDESS", folds=4, seed=42),
    spec("P05R023", 27, "0027__VishHUB1__Multimodal-SER", "tier1_execution_confirmed", "yes", "windowed_random_rows", ["Mel_framing_window.py:L64-L125", "Inception_framing.py:L156-L176"], "Each source utterance yields several framed images and train_test_split then partitions image paths, not utterances or speakers.", corpus="CREMA-D", windows=4, test_size=0.3, seed=42),
    spec("P05R024", 28, "0028__VINUVPOTTY__voice-emotion-recognition-ravdess", "tier1_execution_confirmed", "yes", "random_rows", ["train.py:L53-L79", "train.py:L103-L109"], "Features retain no actor group and are stratified by emotion in an ordinary 80/20 row split.", corpora=["RAVDESS"], test_size=0.2, seed=42, stratify=True),
    spec("P05R025", 29, "0029__flaviorainhoavila__IEMOCAPspeechEmotionRecognition", "tier2_static_execution_inference", "yes", "fastai_random_items_static", ["prep_data.py:L38-L45", "melspec_extrac.py:L42-L57", "emo_rec.ipynb:L122-L133"], "The full path copies utterances into emotion folders, preserves utterance filenames for images, and calls ImageDataBunch.from_folder(valid_pct=0.2). That API partitions items, and the call supplies neither speaker groups nor a group-aware splitter. Fastai is absent from the frozen isolated environment and installation is forbidden, so this is static execution inference: non-enforcement is decisive, while an observed synthetic intersection is not claimed."),
    spec("P05R026", 30, "0030__vikrant-3009__SpeechEmotionRecognition", "tier1_execution_confirmed", "yes", "random_rows", ["main.ipynb:L77-L110"], "RAVDESS rows receive one 75/25 random split with random_state=9 and no actor grouping.", corpora=["RAVDESS"], test_size=0.25, seed=9, stratify=False),
    spec("P05R027", 31, "0031__AitanaESCI__speech-emotion-recognition", "tier1_execution_confirmed", "no", "fixed_iemocap_sessions", ["shared-dataset-loader/nbdev-upc-aidl-iemocap-datasets/src/nbdev_upc_aidl_iemocap_datasets/core.py:L531-L547", "shared-dataset-loader/nbdev-upc-aidl-iemocap-datasets/src/nbdev_upc_aidl_iemocap_datasets/core.py:L775-L785", "shared-dataset-loader/nbdev-upc-aidl-iemocap-datasets/src/nbdev_upc_aidl_iemocap_datasets/core.py:L820-L824"], "The loader fixes Ses03-Ses05 speakers to train, Ses02 speakers to validation, and Ses01 speakers to test.", corpus="IEMOCAP"),
    spec("P05R028", 32, "0032__GaybsGimenez__RAVDESS-Speech-Emotion-Recognition-SER", "tier1_execution_confirmed", "yes", "random_rows", ["Reconhecimento_de_Emoções_pela_Fala_RAVDESS.ipynb:L12719"], "RAVDESS utterance rows receive ordinary 70/30 train_test_split with random_state=1 and no actor groups.", corpora=["RAVDESS"], test_size=0.3, seed=1, stratify=False),
    spec("P05R029", 33, "0033__Abhiramkura__Speech-Emotion-recognition-using-deformable-convolutional-neural-networks", "tier1_execution_confirmed", "yes", "random_rows", ["DCNN_model.ipynb:L898-L904"], "The pooled RAVDESS/CREMA-D feature rows are split by emotion-stratified train_test_split, without a speaker grouping key.", corpora=["RAVDESS", "CREMA-D"], test_size=0.2, seed=42, stratify=True),
    spec("P05R030", 34, "0034__mzarvandi__SER-wav2vec", "tier3_unknown", "unknown", "external_kaggle_partition", ["SER-wav2vec.ipynb:L408-L418", "SER-wav2vec.ipynb:L565-L566"], "The notebook infers membership from external Kaggle Train/Train and Test/Test directories. Their manifests and construction rule are absent, so actor overlap cannot be recovered from the frozen repository."),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def synthetic_manifest() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    definitions = {
        "RAVDESS": ([f"{i:02d}" for i in range(1, 25)], [f"emotion_{i}" for i in range(8)], 60),
        "CREMA-D": ([str(i) for i in range(1001, 1092)], [f"emotion_{i}" for i in range(6)], 24),
        "IEMOCAP": ([f"Ses{s:02d}{g}" for s in range(1, 6) for g in ("F", "M")], [f"emotion_{i}" for i in range(4)], 40),
        "EmoDB": (["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"], [f"emotion_{i}" for i in range(7)], 42),
    }
    for corpus, (speakers, labels, per_speaker) in definitions.items():
        for speaker in speakers:
            for ordinal in range(per_speaker):
                label = labels[ordinal % len(labels)]
                uid = f"{corpus}:{speaker}:u{ordinal:03d}"
                rows.append({
                    "row_id": f"SYN{len(rows):06d}",
                    "corpus": corpus,
                    "path": f"synthetic/{corpus}/{speaker}/{uid}.wav",
                    "speaker_id": f"{corpus}:{speaker}",
                    "emotion": label,
                    "utterance_id": uid,
                })
    return rows


def subset(rows: list[dict[str, str]], corpora: Iterable[str]) -> list[dict[str, str]]:
    wanted = set(corpora)
    return [row for row in rows if row["corpus"] in wanted]


def pair_payload(label: str, train: list[dict[str, str]], test: list[dict[str, str]]) -> dict[str, Any]:
    train_speakers = sorted({row["speaker_id"] for row in train})
    test_speakers = sorted({row["speaker_id"] for row in test})
    intersection = sorted(set(train_speakers) & set(test_speakers))
    return {
        "pair": label,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_speakers": train_speakers,
        "test_speakers": test_speakers,
        "speaker_intersection": intersection,
        "speaker_intersection_count": len(intersection),
    }


def execute_probe(repo: RepoSpec, manifest: list[dict[str, str]]) -> list[dict[str, Any]]:
    p = repo.params
    if repo.method == "random_rows":
        rows = subset(manifest, p["corpora"])
        indices = np.arange(len(rows))
        labels = [row["emotion"] for row in rows] if p["stratify"] else None
        train_i, test_i = train_test_split(indices, test_size=p["test_size"], random_state=p["seed"], shuffle=True, stratify=labels)
        return [pair_payload("final_evaluation", [rows[i] for i in train_i], [rows[i] for i in test_i])]

    if repo.method == "shuffled_fixed_slices":
        rows = subset(manifest, [p["corpus"]])
        shuffled = list(sklearn_shuffle(rows, random_state=p["probe_seed"]))
        test = shuffled[-181:-1]
        remaining = shuffled[:-180]
        train = remaining[:1020]
        return [pair_payload("final_evaluation", train, test)]

    if repo.method == "fixed_speakers":
        rows = subset(manifest, [p["corpus"]])
        train_set, test_set = set(p["train_speakers"]), set(p["test_speakers"])
        return [pair_payload("final_evaluation", [r for r in rows if r["speaker_id"] in train_set], [r for r in rows if r["speaker_id"] in test_set])]

    if repo.method == "random_whole_speakers":
        rows = subset(manifest, [p["corpus"]])
        speakers = np.array(sorted({r["speaker_id"] for r in rows}))
        rng = np.random.RandomState(p["seed"])
        test_speakers = set(rng.choice(speakers, size=p["n_test_speakers"], replace=False).tolist())
        return [pair_payload("final_evaluation", [r for r in rows if r["speaker_id"] not in test_speakers], [r for r in rows if r["speaker_id"] in test_speakers])]

    if repo.method == "augmented_random_rows":
        base = subset(manifest, [p["corpus"]])
        rows = []
        for row in base:
            for variant in range(p["variants"]):
                clone = dict(row)
                clone["row_id"] = f"{row['row_id']}:aug{variant}"
                rows.append(clone)
        train_i, test_i = train_test_split(np.arange(len(rows)), test_size=p["test_size"], random_state=p["seed"], shuffle=True)
        return [pair_payload("final_evaluation", [rows[i] for i in train_i], [rows[i] for i in test_i])]

    if repo.method == "loso_speakers":
        rows = subset(manifest, [p["corpus"]])
        speakers = sorted({r["speaker_id"] for r in rows})
        pairs = []
        for fold, test_speaker in enumerate(speakers):
            val_speaker = speakers[(fold + 1) % len(speakers)]
            train = [r for r in rows if r["speaker_id"] not in {test_speaker, val_speaker}]
            test = [r for r in rows if r["speaker_id"] == test_speaker]
            pairs.append(pair_payload(f"fold_{fold:02d}", train, test))
        return pairs

    if repo.method == "random_whole_speakers_70_15_15":
        rows = subset(manifest, p["corpora"])
        speakers = sorted({r["speaker_id"] for r in rows})
        train_s, temp_s = train_test_split(speakers, test_size=0.30, random_state=p["seed"])
        _val_s, test_s = train_test_split(temp_s, test_size=0.50, random_state=p["seed"])
        train_set, test_set = set(train_s), set(test_s)
        return [pair_payload("final_evaluation", [r for r in rows if r["speaker_id"] in train_set], [r for r in rows if r["speaker_id"] in test_set])]

    if repo.method == "stratified_kfold_rows":
        rows = subset(manifest, [p["corpus"]])
        labels = np.array([r["emotion"] for r in rows])
        splitter = StratifiedKFold(n_splits=p["folds"], shuffle=True, random_state=p["seed"])
        return [pair_payload(f"fold_{fold:02d}", [rows[i] for i in train_i], [rows[i] for i in test_i]) for fold, (train_i, test_i) in enumerate(splitter.split(np.arange(len(rows)), labels))]

    if repo.method == "windowed_random_rows":
        base = subset(manifest, [p["corpus"]])
        rows = []
        for row in base:
            for window in range(p["windows"]):
                clone = dict(row)
                clone["row_id"] = f"{row['row_id']}:window{window}"
                rows.append(clone)
        labels = [r["emotion"] for r in rows]
        train_i, test_i = train_test_split(np.arange(len(rows)), test_size=p["test_size"], random_state=p["seed"], stratify=labels)
        return [pair_payload("final_evaluation", [rows[i] for i in train_i], [rows[i] for i in test_i])]

    if repo.method == "fixed_iemocap_sessions":
        rows = subset(manifest, [p["corpus"]])
        train = [r for r in rows if r["speaker_id"].startswith(("IEMOCAP:Ses03", "IEMOCAP:Ses04", "IEMOCAP:Ses05"))]
        test = [r for r in rows if r["speaker_id"].startswith("IEMOCAP:Ses01")]
        return [pair_payload("final_evaluation", train, test)]

    raise ValueError(f"No execution probe for {repo.repository_id}: {repo.method}")


def parse_ref(ref: str) -> tuple[str, int | None, int | None]:
    file_part, marker, line_part = ref.partition(":L")
    if not marker:
        return file_part, None, None
    token = line_part.split()[0].split(",")[0]
    if token.isdigit():
        value = int(token)
        return file_part, value, value
    if "-L" in token:
        start, end = token.split("-L", 1)
        if start.isdigit() and end.isdigit():
            return file_part, int(start), int(end)
    return file_part, None, None


def source_inventory(specs: list[RepoSpec]) -> list[dict[str, Any]]:
    inventory = []
    for repo in specs:
        repo_root = SOURCE_ROOT / repo.source_dir
        for path in sorted(repo_root.rglob("*")):
            if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_CODE_SUFFIXES:
                continue
            data = path.read_bytes()
            inventory.append({
                "repository_id": repo.repository_id,
                "draw_rank": repo.rank,
                "relative_path": path.relative_to(repo_root).as_posix(),
                "bytes": len(data),
                "line_count": data.count(b"\n") + (1 if data else 0),
                "sha256": hashlib.sha256(data).hexdigest(),
                "read_mode": "full_byte_read_no_execution",
                "candidate_code_executed": "false",
            })
    return inventory


def git_head(repo_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def main() -> int:
    for rel, expected_hash in EXPECTED_INPUTS.items():
        actual = sha256(ROOT / rel)
        if actual != expected_hash:
            raise RuntimeError(f"Frozen input hash mismatch: {rel}: {actual} != {expected_hash}")

    survey = read_csv(ROOT / "p0_5_random_survey.csv")
    prior = {row["repository_id"]: row["split"] for row in read_csv(ROOT / "results" / "p0_5" / "random_repository_risk.csv")}
    survey_by_id = {row["repository_id"]: row for row in survey}
    if len(SPECS) != 30 or set(survey_by_id) != {repo.repository_id for repo in SPECS}:
        raise RuntimeError("Frozen 30-repository roster does not match specification")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PER_REPO.mkdir(parents=True, exist_ok=True)
    manifest = synthetic_manifest()
    write_csv(RESULT_ROOT / "synthetic_manifest.csv", ["row_id", "corpus", "path", "speaker_id", "emotion", "utterance_id"], manifest)

    inventory = source_inventory(SPECS)
    write_csv(RESULT_ROOT / "source_inventory.csv", ["repository_id", "draw_rank", "relative_path", "bytes", "line_count", "sha256", "read_mode", "candidate_code_executed"], inventory)

    evidence_rows = []
    changes = []
    for repo in SPECS:
        row = survey_by_id[repo.repository_id]
        repo_root = SOURCE_ROOT / repo.source_dir
        head = git_head(repo_root)
        if head != row["commit_sha"].lower():
            raise RuntimeError(f"HEAD mismatch for {repo.repository_id}: {head}")

        for ref in repo.source_refs:
            rel_path, start, end = parse_ref(ref)
            source_path = repo_root / rel_path
            if not source_path.is_file():
                raise RuntimeError(f"Missing decisive source for {repo.repository_id}: {rel_path}")
            if start is not None and end is not None:
                line_count = source_path.read_bytes().count(b"\n") + 1
                if not (1 <= start <= end <= line_count):
                    raise RuntimeError(f"Invalid line anchor for {repo.repository_id}: {ref} / {line_count}")

        if repo.tier == "tier1_execution_confirmed":
            pairs = execute_probe(repo, manifest)
            max_overlap = max(p["speaker_intersection_count"] for p in pairs)
            final = "yes" if max_overlap > 0 else "no"
            if final != repo.expected:
                raise RuntimeError(f"Unexpected probe result for {repo.repository_id}: {final} != {repo.expected}")
            validation_method = "isolated_split_logic_on_synthetic_speaker_manifest"
            synthetic_rows = sum(p["train_rows"] + p["test_rows"] for p in pairs)
            synthetic_speakers = len(set().union(*(set(p["train_speakers"]) | set(p["test_speakers"]) for p in pairs)))
        else:
            pairs = []
            max_overlap = None
            final = repo.expected
            validation_method = repo.method
            synthetic_rows = 0
            synthetic_speakers = 0

        output_rel = f"per_repository/{repo.repository_id}.json"
        payload = {
            "repository_id": repo.repository_id,
            "repository_url": row["repository_url"],
            "commit_sha": row["commit_sha"],
            "draw_rank": repo.rank,
            "evidence_tier": repo.tier,
            "candidate_code_executed": False,
            "candidate_source_imported": False,
            "network_used": False,
            "whole_repository_executed": False,
            "probe_method": repo.method,
            "probe_parameters": repo.params,
            "decisive_source": list(repo.source_refs),
            "rationale": repo.rationale,
            "pairs": pairs,
            "max_speaker_intersection": max_overlap,
            "split_risk_final": final,
            "prior_split_risk": prior[repo.repository_id],
        }
        out_path = RESULT_ROOT / output_rel
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        changed = prior[repo.repository_id] != final
        permalink_refs = []
        for ref in repo.source_refs:
            rel_path, start, end = parse_ref(ref)
            encoded_path = "/".join(part.replace(" ", "%20") for part in Path(rel_path).as_posix().split("/"))
            anchor = f"#L{start}" if start is not None else ""
            if start is not None and end is not None and end != start:
                anchor += f"-L{end}"
            permalink_refs.append(f"{row['repository_url']}/blob/{row['commit_sha']}/{encoded_path}{anchor}")
        evidence_rows.append({
            "repository_id": repo.repository_id,
            "draw_rank": repo.rank,
            "repository_url": row["repository_url"],
            "commit_sha": row["commit_sha"],
            "target_datasets": row["target_datasets"],
            "prior_split_risk": prior[repo.repository_id],
            "evidence_tier": repo.tier,
            "validation_method": validation_method,
            "candidate_code_executed": "false",
            "synthetic_rows_evaluated": synthetic_rows,
            "synthetic_speakers_evaluated": synthetic_speakers,
            "evaluation_pairs": len(pairs),
            "max_speaker_intersection": "" if max_overlap is None else max_overlap,
            "split_risk_final": final,
            "changed_from_prior": str(changed).lower(),
            "decisive_file_line": " | ".join(repo.source_refs),
            "frozen_permalinks": " | ".join(permalink_refs),
            "inevitability_or_limit": repo.rationale,
            "reproduction_output": output_rel,
        })
        if changed:
            changes.append({
                "repository_id": repo.repository_id,
                "repository_url": row["repository_url"],
                "old_value": prior[repo.repository_id],
                "new_value": final,
                "reason": repo.rationale,
                "decisive_file_line": " | ".join(repo.source_refs),
                "reproduction_output": output_rel,
            })

    evidence_fields = [
        "repository_id", "draw_rank", "repository_url", "commit_sha", "target_datasets",
        "prior_split_risk", "evidence_tier", "validation_method", "candidate_code_executed",
        "synthetic_rows_evaluated", "synthetic_speakers_evaluated", "evaluation_pairs",
        "max_speaker_intersection", "split_risk_final", "changed_from_prior",
        "decisive_file_line", "frozen_permalinks", "inevitability_or_limit", "reproduction_output",
    ]
    write_csv(RESULT_ROOT / "repository_evidence.csv", evidence_fields, evidence_rows)
    write_csv(RESULT_ROOT / "change_registry.csv", ["repository_id", "repository_url", "old_value", "new_value", "reason", "decisive_file_line", "reproduction_output"], changes)

    tier_counts = Counter(row["evidence_tier"] for row in evidence_rows)
    outcome_counts = Counter(row["split_risk_final"] for row in evidence_rows)
    summary_rows = [
        {"dimension": "evidence_tier", "category": key, "count": tier_counts[key], "denominator": 30, "note": "mutually exclusive repository tier"}
        for key in ("tier1_execution_confirmed", "tier2_static_execution_inference", "tier3_unknown")
    ] + [
        {"dimension": "split_risk_final", "category": key, "count": outcome_counts[key], "denominator": 30, "note": "yes=synthetic overlap or decisive non-enforcement; no=speaker-disjoint; unknown=insufficient frozen evidence"}
        for key in ("yes", "no", "unknown")
    ]
    write_csv(RESULT_ROOT / "tier_summary.csv", ["dimension", "category", "count", "denominator", "note"], summary_rows)

    readme = """# P0.5 evidence-strength execution validation artifacts

This directory is the self-contained machine-readable record for the frozen 30-repository probability sample.

- `repository_evidence.csv`: one row per repository and the final evidence-tier decision.
- `per_repository/*.json`: parameters, direct train/test speaker sets, intersections, or the exact static/unknown boundary.
- `synthetic_manifest.csv`: deterministic manifest with explicit corpus-qualified speaker IDs.
- `source_inventory.csv`: every text/code artifact byte-read from each frozen repository; candidate code was never executed.
- `tier_summary.csv`: required evidence-tier and final-risk counts.
- `change_registry.csv`: changes from the pre-existing P0.5 split-risk table.
- `analysis_manifest.json`: frozen input/output hashes and environment provenance.

The execution probe re-implements only the decisive split primitive. It does not import candidate modules, execute a repository, access the network, train a model, or infer performance. A positive synthetic intersection validates the shortcut mechanism under the repository's split rule; it is not an estimate of the original authors' realized overlap count.
"""
    (RESULT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    output_hashes = {}
    hash_exclusions = {"analysis_manifest.json", "verification.json", "artifact_qa.json"}
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path.name not in hash_exclusions:
            output_hashes[path.relative_to(RESULT_ROOT).as_posix()] = sha256(path)
    analysis_manifest = {
        "stage": "P0.5 evidence-strength execution validation",
        "authority_root": str(ROOT),
        "analysis_script": "tools/run_p0_5_execution_validation.py",
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "safety": {
            "network_used": False,
            "candidate_code_executed": False,
            "candidate_source_imported": False,
            "whole_repository_executed": False,
            "models_trained": False,
            "p1_or_p2_rerun": False,
        },
        "expected_input_sha256": EXPECTED_INPUTS,
        "sample_size": 30,
        "tier_counts": dict(tier_counts),
        "outcome_counts": dict(outcome_counts),
        "changed_repositories": [row["repository_id"] for row in changes],
        "output_hash_exclusions": sorted(hash_exclusions),
        "output_sha256": output_hashes,
    }
    (RESULT_ROOT / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tier_counts": dict(tier_counts), "outcome_counts": dict(outcome_counts), "changes": [row["repository_id"] for row in changes]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write the root reviewer's preregistered P0.5 ranks 1--11 batch.

This is a mechanical serializer for decisions made by static inspection.  It
does not import or execute any candidate repository code.
"""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
HEADER = next(csv.reader((ROOT / "survey_table.csv").open(encoding="utf-8-sig", newline="")))
AUDITED_AT = "2026-08-10T18:19:10-04:00"


def rec(**values: str) -> dict[str, str]:
    row = {key: "" for key in HEADER}
    row.update(values)
    return row


ROWS = [
    rec(
        repository_url="https://github.com/Vansh-187/speech-emotion-recognition",
        commit_sha="f21f3700413ea8e5c39921fb102c10672cae69a8",
        commit_date="2025-06-24T19:07:01+05:30", audited_at=AUDITED_AT,
        license="Apache-2.0", publication_year="2025",
        repo_relation="third_party_reproduction", target_datasets="RAVDESS",
        reported_results="RAVDESS 8-class holdout accuracy=82.61%; F1=82.00%",
        result_evidence="README.md:L50-L56; notebooks/emotion_classifier_final.ipynb:L1 cell 30 output accuracy=0.8261",
        result_code_link="direct", split_category="random",
        split_by_dataset="RAVDESS=utterance-level train_test_split(test_size=0.2, random_state=0)",
        speaker_disjointness="not_enforced",
        split_evidence="notebooks/emotion_classifier_final.ipynb:L1 cells 6 and 16 parse emotion but do not group actor and call train_test_split",
        normalization_leakage="no", normalization_by_dataset="RAVDESS=per-utterance deterministic features; no corpus-fitted scaler on the audited path",
        normalization_evidence="notebooks/emotion_classifier_final.ipynb:L1 cells 13-16 compute per-utterance features before splitting without a fitted cross-sample normalizer",
        test_selection="yes_explicit", test_selection_by_dataset="RAVDESS=the test loader is evaluated every epoch and best checkpoint is chosen by test accuracy",
        test_selection_evidence="notebooks/emotion_classifier_final.ipynb:L1 cells 25,27,30 build test_loader, save on best test_acc, then reload it",
        augmentation_leakage="no_train_only", augmentation_by_dataset="RAVDESS=noise and stretch-plus-pitch are appended only when the base sample is in x_train",
        augmentation_evidence="notebooks/emotion_classifier_final.ipynb:L1 cells 16-18 split base features first and append augmentations only for matched training rows",
        evaluation_repetition="single_split_single_seed", repetition_by_dataset="RAVDESS=one 80/20 split with random_state=0 and one training run",
        n_folds="1", n_seeds="1", variance_reported="none",
        repetition_evidence="notebooks/emotion_classifier_final.ipynb:L1 cells 16 and 27 contain one fixed split and one training loop",
        paper_code_discrepancy="not_applicable_no_paper", evidence_strength="high",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=1; test-selected checkpoint makes the displayed metric optimistically selected on the same holdout."
    ),
    rec(
        repository_url="https://github.com/fdhlhdc12/ravdess-dtw-emotion-recognition",
        commit_sha="c20123028d7e718e624eb56599ae7436ca9991e4",
        commit_date="2026-06-27T12:47:50+07:00", audited_at=AUDITED_AT,
        license="no license file found", publication_year="2026",
        repo_relation="third_party_reproduction", target_datasets="RAVDESS",
        reported_results="dashboard claims accuracy=90.28%, precision=90.11%, recall=89.56%, F1=90.22%, with 5-fold GridSearchCV",
        result_evidence="app.py:L1983-L2010", result_code_link="uncertain",
        split_category="unknown", split_by_dataset="RAVDESS=training split and fold construction absent from frozen repository",
        speaker_disjointness="unknown", split_evidence="app.py:L735-L752 loads serialized models/scaler; no training or split-generation code is present",
        normalization_leakage="unknown", normalization_by_dataset="RAVDESS=serialized scaler is applied at inference but its fit population is unavailable",
        normalization_evidence="app.py:L737-L743|L779-L784 loads scaler.pkl and transforms a feature; fit code/provenance is absent",
        test_selection="unknown", test_selection_by_dataset="RAVDESS=model selection and final-test data flow unavailable",
        test_selection_evidence="app.py:L1983-L2010 contains static displayed metrics and a GridSearchCV claim but no training/evaluation path",
        augmentation_leakage="not_applicable", augmentation_by_dataset="RAVDESS=auditable inference feature path has no augmentation",
        augmentation_evidence="feature_extraction_ml.py:L4-L45 extracts MFCC/delta summaries only; app.py:L770-L784 calls that path",
        evaluation_repetition="kfold_single_run", repetition_by_dataset="RAVDESS=dashboard states one 5-fold GridSearchCV; seed schedule unavailable",
        n_folds="5", n_seeds="unknown", variance_reported="none",
        repetition_evidence="app.py:L2005-L2010 states 5-fold GridSearchCV; no fold outputs or repeated seeds are supplied",
        paper_code_discrepancy="not_applicable_no_paper", evidence_strength="low",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=2; low evidence converts the five risk endpoints to unknown in primary-risk aggregation. Serialized pkl files were not loaded or executed."
    ),
    rec(
        repository_url="https://github.com/Shiva9855/CodeAlpha_Emotion-Recognition-from-Speech",
        commit_sha="cbc84f527fc14b375d34ce2c8aec9b00cabaf459",
        commit_date="2026-07-20T01:15:32+05:30", audited_at=AUDITED_AT,
        license="MIT", publication_year="2026",
        repo_relation="third_party_reproduction", target_datasets="RAVDESS|EmoDB",
        reported_results="no numeric RAVDESS or EmoDB result reported in frozen repository",
        result_evidence="README.md:L129-L137 discusses expected outputs and recommends actor-wise splitting but gives no metric",
        result_code_link="uncertain", split_category="random",
        split_by_dataset="RAVDESS=random utterance holdout | EmoDB=random utterance holdout in one merged dataset",
        speaker_disjointness="not_enforced",
        split_evidence="utils/dataset_loader.py:L34-L43 retains only path and emotion for RAVDESS; train.py:L145-L159 stratifies by label only",
        normalization_leakage="no", normalization_by_dataset="RAVDESS=train-only StandardScaler | EmoDB=train-only StandardScaler",
        normalization_evidence="train.py:L66-L74 fits StandardScaler on X_train and transforms X_test",
        test_selection="yes_explicit", test_selection_by_dataset="RAVDESS=yes_explicit | EmoDB=yes_explicit",
        test_selection_evidence="train.py:L158-L159|L169-L196 uses the sole test_loader each epoch, early-stops and saves on its accuracy",
        augmentation_leakage="yes", augmentation_by_dataset="RAVDESS=augment-before-split | EmoDB=augment-before-split",
        augmentation_evidence="train.py:L47-L63 builds original plus three augmentations for all rows; train.py:L136-L153 then splits the combined matrix",
        evaluation_repetition="single_split_single_seed", repetition_by_dataset="RAVDESS=one fixed random split | EmoDB=one fixed random split",
        n_folds="1", n_seeds="1", variance_reported="none",
        repetition_evidence="train.py:L151-L154 uses one random_state; train.py:L174-L196 contains one training run",
        paper_code_discrepancy="not_applicable_no_paper", evidence_strength="high",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=4; README.md:L137 itself suggests speaker-wise splitting as a stricter alternative."
    ),
    rec(
        repository_url="https://github.com/dingdongwang/EmotionThinker",
        commit_sha="afcc5c578bc69df436fbb2317957fe4897e33342",
        commit_date="2026-08-07T13:27:05-07:00", audited_at=AUDITED_AT,
        license="no license file found", paper_title="EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning",
        paper_url="https://arxiv.org/abs/2601.15668", publication_year="2026",
        repo_relation="author_official", target_datasets="IEMOCAP|RAVDESS",
        reported_results="paper Table 2 reports IEMOCAP accuracy=77.68% and RAVDESS accuracy=71.56%",
        result_evidence="paper arXiv:2601.15668 Table 2; README.md:L5-L8 identifies the official paper and README.md:L79-L85 lists IEMOCAP annotations",
        result_code_link="uncertain", split_category="unknown",
        split_by_dataset="IEMOCAP=training/evaluation split absent | RAVDESS=training/evaluation split absent",
        speaker_disjointness="unknown",
        split_evidence="scripts/emotionthinker_infer.py:L1-L18 provides model inference only; no target training or evaluation split code exists",
        normalization_leakage="unknown", normalization_by_dataset="IEMOCAP=unknown | RAVDESS=unknown",
        normalization_evidence="scripts/emotionthinker_infer.py:L1-L18 delegates preprocessing to a downloaded pretrained model; target training statistics are absent",
        test_selection="unknown", test_selection_by_dataset="IEMOCAP=unknown | RAVDESS=unknown",
        test_selection_evidence="README.md:L54-L69 documents inference only; target training/checkpoint-selection code is absent",
        augmentation_leakage="unknown", augmentation_by_dataset="IEMOCAP=unknown | RAVDESS=unknown",
        augmentation_evidence="README.md:L74-L89 describes released annotations/prosody pipeline, not the target benchmark training loaders",
        evaluation_repetition="unknown", repetition_by_dataset="IEMOCAP=unknown in frozen code | RAVDESS=unknown in frozen code",
        n_folds="unknown", n_seeds="unknown", variance_reported="none",
        repetition_evidence="scripts/emotionthinker_infer.py:L1-L18 has no fold/seed loop and the frozen repository lacks target evaluation code",
        paper_code_discrepancy="unassessable_missing_training_evaluation_code", evidence_strength="low",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=5; official paper/repository relation is clear, but all preregistered implementation-risk fields remain unknown under the code-first rule."
    ),
    rec(
        repository_url="https://github.com/jobinjoy12/Speech-Emotion-Recognition-using-CREMA-D-dataset",
        commit_sha="10f7e5102b86904ea3b3bb40708238988834ccd0",
        commit_date="2024-02-22T09:13:09+05:30", audited_at=AUDITED_AT,
        license="no license file found", publication_year="2024",
        repo_relation="third_party_reproduction", target_datasets="CREMA-D",
        reported_results="CREMA-D holdout accuracy=52.10%",
        result_evidence="emotion speech recognition code (2).ipynb:L1 cell gHy_VEo2btF3 output",
        result_code_link="direct", split_category="random",
        split_by_dataset="CREMA-D=utterance-level train_test_split(test_size=0.2) without speaker groups",
        speaker_disjointness="not_enforced",
        split_evidence="emotion speech recognition code (2).ipynb:L1 cells TmZwQAAia6dU and xQYo6vSXbkTy parse labels and call ordinary train_test_split",
        normalization_leakage="no", normalization_by_dataset="CREMA-D=StandardScaler fitted on x_train only",
        normalization_evidence="emotion speech recognition code (2).ipynb:L1 cell BBnu4lqxboil fits x_train then transforms x_test",
        test_selection="no_fixed_training", test_selection_by_dataset="CREMA-D=one fixed MLP fit followed by one holdout prediction",
        test_selection_evidence="emotion speech recognition code (2).ipynb:L1 cells HLTsAspAbqR1 and gHy_VEo2btF3 fit once then predict once",
        augmentation_leakage="yes", augmentation_by_dataset="CREMA-D=noise is applied to every utterance before the holdout split, including eventual test items",
        augmentation_evidence="emotion speech recognition code (2).ipynb:L1 cell xyJe_5Q1-xcV injects noise while building all X; cell xQYo6vSXbkTy splits afterward",
        evaluation_repetition="single_split_single_seed", repetition_by_dataset="CREMA-D=one unseeded holdout and one MLP fit",
        n_folds="1", n_seeds="1", variance_reported="none",
        repetition_evidence="emotion speech recognition code (2).ipynb:L1 cells xQYo6vSXbkTy and HLTsAspAbqR1 contain one split and one fit",
        paper_code_discrepancy="not_applicable_no_paper", evidence_strength="high",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=6; test audio itself is noise-augmented before splitting, satisfying the frozen augmentation-risk definition."
    ),
    rec(
        repository_url="https://github.com/ShaheenPerveen/Speech_Emotion_Recognition_IEMOCAP",
        commit_sha="1f2a835abbf1fbce138b7c26e7e2eb74b35b8147",
        commit_date="2020-01-01T15:42:44+05:30", audited_at=AUDITED_AT,
        license="no license file found", publication_year="2020",
        repo_relation="third_party_reproduction", target_datasets="IEMOCAP",
        reported_results="representative normalized-MFCC path reports best internal CV score=0.66499 (gradient boosting); multiple holdout reports are retained in notebook outputs",
        result_evidence="Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 35-38",
        result_code_link="direct", split_category="random",
        split_by_dataset="IEMOCAP=utterance-level stratified 80/20 split; ordinary 2- or 3-fold CV inside training subset",
        speaker_disjointness="not_enforced",
        split_evidence="Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cell 16 uses train_test_split stratified only by target",
        normalization_leakage="yes", normalization_by_dataset="IEMOCAP=MinMaxScaler fitted on the full feature table before train/test split",
        normalization_evidence="Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 14-16 call fit_transform(train) before train_test_split",
        test_selection="test_exposed_each_epoch", test_selection_by_dataset="IEMOCAP=the same X_test is repeatedly reported across many candidate classifiers without an outer untouched set",
        test_selection_evidence="Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 22-65 tune several models then repeatedly print X_test reports",
        augmentation_leakage="not_applicable", augmentation_by_dataset="IEMOCAP=no augmentation in the representative normalized-MFCC path",
        augmentation_evidence="Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 1-16 load fixed features, normalize, and split without augmentation",
        evaluation_repetition="single_split_single_seed|kfold_single_run", repetition_by_dataset="IEMOCAP=one unseeded holdout; 2-fold grids for most models and one 3-fold grid",
        n_folds="2|3", n_seeds="1", variance_reported="none",
        repetition_evidence="Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 16,22,27,35,40,44 show one holdout plus cv=2/3",
        paper_code_discrepancy="not_applicable_no_paper", evidence_strength="high",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=8; representative auditable path selected before aggregate results were computed."
    ),
    rec(
        repository_url="https://github.com/sss3799/audio_emotion_recognition",
        commit_sha="d83a682e7b0abe95fa983ce0323bb71010cd2af5",
        commit_date="2023-04-22T21:40:34+05:30", audited_at=AUDITED_AT,
        license="no license file found", publication_year="2023",
        repo_relation="third_party_reproduction", target_datasets="RAVDESS",
        reported_results="RAVDESS holdout accuracy=60.7433%",
        result_evidence="hsl621_emotion_recognition.ipynb:L1 cell 46 output",
        result_code_link="direct", split_category="random",
        split_by_dataset="RAVDESS=utterance-level train_test_split(random_state=0, shuffle=True)",
        speaker_disjointness="not_enforced",
        split_evidence="hsl621_emotion_recognition.ipynb:L1 cells 7 and 40 retain emotion labels but call ordinary train_test_split without actor groups",
        normalization_leakage="no", normalization_by_dataset="RAVDESS=StandardScaler fitted on x_train then applied to x_test",
        normalization_evidence="hsl621_emotion_recognition.ipynb:L1 cell 41",
        test_selection="test_exposed_each_epoch", test_selection_by_dataset="RAVDESS=the sole test split is passed as validation_data for all 50 epochs",
        test_selection_evidence="hsl621_emotion_recognition.ipynb:L1 cells 45-46 train with validation_data=(x_test,y_test) then report it as test",
        augmentation_leakage="yes", augmentation_by_dataset="RAVDESS=original, noise, and stretch-plus-pitch variants are generated before random splitting",
        augmentation_evidence="hsl621_emotion_recognition.ipynb:L1 cells 33-34 generate three variants for all paths; cell 40 splits afterward",
        evaluation_repetition="single_split_single_seed", repetition_by_dataset="RAVDESS=one split with random_state=0 and one 50-epoch fit",
        n_folds="1", n_seeds="1", variance_reported="none",
        repetition_evidence="hsl621_emotion_recognition.ipynb:L1 cells 40 and 45",
        paper_code_discrepancy="not_applicable_no_paper", evidence_strength="high",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=9; augmented counterparts can cross the random split and the holdout is exposed every epoch."
    ),
    rec(
        repository_url="https://github.com/neerajk23-cmd/Speech-Emotion-Recognition-on-RAVDESS-Dataset",
        commit_sha="516bc2802eddc65dc382c7d51cd0384410c3e056",
        commit_date="2026-08-06T22:25:03+05:30", audited_at=AUDITED_AT,
        license="no license file found", publication_year="2026",
        repo_relation="third_party_reproduction", target_datasets="RAVDESS",
        reported_results="held-out actors 20-24: accuracy=62.33% on 300 utterances",
        result_evidence="Evaluation_Test_Data.ipynb:L1 cells 8-10 outputs found 5 actors/300 files and accuracy=0.6233",
        result_code_link="direct", split_category="predefined_speaker_independent",
        split_by_dataset="RAVDESS=actors 1-19 training pool and actors 20-24 final test; utterance-random inner train/validation split within actors 1-19",
        speaker_disjointness="enforced",
        split_evidence="cnn-final-ee708-project.ipynb:L1 cells 5-6 show 19 train actors and inner random validation; Evaluation_Test_Data.ipynb:L1 cells 3,6,8 show five separate test actor directories",
        normalization_leakage="no", normalization_by_dataset="RAVDESS=per-utterance fixed mel-spectrogram/decibel transforms; no cross-sample fitted scaler",
        normalization_evidence="Evaluation_Test_Data.ipynb:L1 cell 3 constructs each test spectrogram independently with fixed parameters",
        test_selection="no_separate_validation", test_selection_by_dataset="RAVDESS=best checkpoint chosen on an inner validation subset; actor-held-out test evaluated once in separate notebook",
        test_selection_evidence="cnn-final-ee708-project.ipynb:L1 cell 6 saves on validation loss; Evaluation_Test_Data.ipynb:L1 cells 8-10 load and evaluate final test once",
        augmentation_leakage="not_applicable", augmentation_by_dataset="RAVDESS=no augmentation in the final train or held-out-test notebook paths",
        augmentation_evidence="cnn-final-ee708-project.ipynb:L1 cells 4-6 and Evaluation_Test_Data.ipynb:L1 cell 3 contain fixed mel transforms only",
        evaluation_repetition="single_split_single_seed", repetition_by_dataset="RAVDESS=one actor holdout and one inner split with seed=42",
        n_folds="1", n_seeds="1", variance_reported="none",
        repetition_evidence="cnn-final-ee708-project.ipynb:L1 cells 5-6; Evaluation_Test_Data.ipynb:L1 cells 8-10",
        paper_code_discrepancy="unassessable_report_not_mapped_to_claims", evidence_strength="high",
        primary_reviewer="root", verification_status="pending",
        notes="P0.5 draw_rank=10; the final test is speaker-disjoint even though the inner validation split is not grouped. Static inspection only; candidate torchaudio code was not run."
    ),
]


EXCLUSIONS = [
    {
        "draw_rank": "3", "canonical_repo_url": "https://github.com/leecool9669/emotion-recognition-wav2vec2-IEMOCAP-WebUI",
        "commit_sha": "c9c809cd2a863765f52d86d8a254fe38a5c9a610", "exclusion_code": "E2_target_not_evaluated",
        "checked_at": AUDITED_AT,
        "exclusion_reason": "Frozen repository is an explicitly fake/demo-only WebUI: it neither loads real weights nor performs target-data evaluation.",
        "evidence": "app.py:L2-L13|L24|L67; README.md:L60|L76-L101",
    },
    {
        "draw_rank": "7", "canonical_repo_url": "https://github.com/Aakash200/speech-emotion-recognition-ravdess-data",
        "commit_sha": "7033167b79af57e92c8a9a9b34920ce8ce9235a7", "exclusion_code": "E3_no_relevant_code",
        "checked_at": AUDITED_AT,
        "exclusion_reason": "Frozen tree contains the 1,440 RAVDESS WAV files but no source code or evaluation implementation.",
        "evidence": "repository tree at frozen commit: 1,440 .wav files; zero code/notebook/README files",
    },
    {
        "draw_rank": "11", "canonical_repo_url": "https://github.com/revatipimparkar/Facial-Emotion-Recognition-in-Humans",
        "commit_sha": "9056c8458889a1e810700b74d8c87700bab15c36", "exclusion_code": "E1_not_SER",
        "checked_at": AUDITED_AT,
        "exclusion_reason": "Repository implements facial/image emotion recognition rather than speech emotion recognition.",
        "evidence": "README.md:L1-L4; Final_Emotion_recognition_cnn.ipynb:L1",
    },
]


def blob(url: str, sha: str, path: str, lines: str) -> str:
    return f"{url}/blob/{sha}/{quote(path, safe='/')}#{lines}"


def main() -> None:
    out_csv = ROOT / "work" / "p0_5_audit_root.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ROWS)

    ex_csv = ROOT / "work" / "p0_5_exclusions_root.csv"
    ex_header = list(EXCLUSIONS[0])
    with ex_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ex_header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(EXCLUSIONS)

    md = [
        "# P0.5 随机样本静态审计：root 批次（抽签位 1–11）",
        "",
        f"- 审计时间：`{AUDITED_AT}`",
        "- 只读静态审计；未运行候选代码、未加载其 pickle/权重、未安装其依赖、未使用 GPU。",
        "- 本批次：抽签位 1–11；纳入 8，E1/E2/E3 各排除 1。主表判定仍待独立全字段复核。",
        "",
        "## 纳入",
        "",
    ]
    ranks = [1, 2, 4, 5, 6, 8, 9, 10]
    for rank, row in zip(ranks, ROWS):
        url, sha = row["repository_url"], row["commit_sha"]
        md += [
            f"### 抽签位 {rank}: `{url}`",
            "",
            f"- 冻结 commit：`{sha}`（`{row['commit_date']}`）",
            f"- 数据集/划分：`{row['target_datasets']}` / `{row['split_category']}`；说话人互斥：`{row['speaker_disjointness']}`。",
            f"- 主风险：归一化 `{row['normalization_leakage']}`；测试选择 `{row['test_selection']}`；增强 `{row['augmentation_leakage']}`；重复 `{row['evaluation_repetition']}`。",
            f"- 结果：{row['reported_results']}",
            f"- 可追溯证据（CSV）：{row['split_evidence']}；{row['normalization_evidence']}；{row['test_selection_evidence']}；{row['augmentation_evidence']}。",
            "",
        ]
    md += ["## 排除与补抽依据", ""]
    for row in EXCLUSIONS:
        md += [
            f"- 抽签位 {row['draw_rank']} — `{row['exclusion_code']}`：{row['exclusion_reason']} 证据：{row['evidence']}。",
        ]
    md += [
        "",
        "## 冻结永久链接抽查索引",
        "",
        f"- rank 1 split/test: {blob(ROWS[0]['repository_url'], ROWS[0]['commit_sha'], 'notebooks/emotion_classifier_final.ipynb', 'L1')}；results: {blob(ROWS[0]['repository_url'], ROWS[0]['commit_sha'], 'README.md', 'L50-L56')}。",
        f"- rank 2 inference/scaler: {blob(ROWS[1]['repository_url'], ROWS[1]['commit_sha'], 'app.py', 'L735-L784')}；metrics: {blob(ROWS[1]['repository_url'], ROWS[1]['commit_sha'], 'app.py', 'L1983-L2010')}。",
        f"- rank 4 split/selection: {blob(ROWS[2]['repository_url'], ROWS[2]['commit_sha'], 'train.py', 'L145-L196')}；augmentation: {blob(ROWS[2]['repository_url'], ROWS[2]['commit_sha'], 'train.py', 'L47-L63')}。",
        f"- rank 5 official relation: {blob(ROWS[3]['repository_url'], ROWS[3]['commit_sha'], 'README.md', 'L1-L18')}；inference-only path: {blob(ROWS[3]['repository_url'], ROWS[3]['commit_sha'], 'scripts/emotionthinker_infer.py', 'L1-L18')}。",
        f"- rank 6 complete notebook: {blob(ROWS[4]['repository_url'], ROWS[4]['commit_sha'], 'emotion speech recognition code (2).ipynb', 'L1')}。",
        f"- rank 8 normalized MFCC path: {blob(ROWS[5]['repository_url'], ROWS[5]['commit_sha'], 'Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb', 'L1')}。",
        f"- rank 9 complete notebook: {blob(ROWS[6]['repository_url'], ROWS[6]['commit_sha'], 'hsl621_emotion_recognition.ipynb', 'L1')}。",
        f"- rank 10 train/validation: {blob(ROWS[7]['repository_url'], ROWS[7]['commit_sha'], 'cnn-final-ee708-project.ipynb', 'L1')}；held-out actor test: {blob(ROWS[7]['repository_url'], ROWS[7]['commit_sha'], 'Evaluation_Test_Data.ipynb', 'L1')}。",
        "",
        "Notebook 仓库的 GitHub 文件在冻结 commit 下为单个 JSON 文本行，故永久链接锚为 `L1`；具体 cell id/index 已保留在 CSV 证据字段。",
    ]
    (ROOT / "work" / "p0_5_audit_root.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"included={len(ROWS)} excluded={len(EXCLUSIONS)}")


if __name__ == "__main__":
    main()

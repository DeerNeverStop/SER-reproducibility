# P0.5 随机样本逐库证据

按冻结顺序检查抽签位 1–34，补抽后得到 30 个合格库；其后 331-34 个位置保持 `not_reached`。
本文件拼接各初审批次；最终结论须在 30/30 全字段交叉复核与主审裁决后读取主表。

<!-- source-batch: p0_5_audit_21_35.md -->

# P0.5 ranks 21–35 顺序静态审计（隔离批次）

- 审计时间：`2026-08-10T18:45:02-04:00`
- 冻结规则：沿用 `P0_PROTOCOL.md` 与 `P0_5_PROTOCOL.md`；只读静态审计，未运行候选代码、未安装候选依赖、未使用 GPU。
- 上一批状态：ranks 1–20 已有 17 个纳入。rank 21 排除后，ranks 22–34 连续纳入 13 个；在 rank 34 达到总体 n=30，立即停止。
- rank 35：`not_reached`，未做 Stage-B 审计。
- 本批输出：13 include、1 exclude（`E2_target_not_evaluated`）、1 not_reached。
- `repository_id` 按委托要求留空；全部 `verification_status=pending`。

## 顺序结局

| draw rank | 结局 | 仓库 | 关键说明 |
|---:|---|---|---|
| 21 | exclude | Kani2k06/CODEALPHA-TASK-2 | E2：README 声称 EmoDB/RAVDESS，但唯一训练代码只加载 TESS。 |
| 22 | include | https://github.com/daribdb/speech-emotion-recognition-iemocap | IEMOCAP 4-class 10-fold leave-one-speaker-out: WA=52.71% ± 4.39% SD; UA=52.36% ± 4.81% SD |
| 23 | include | https://github.com/Yaowan410/Emotion-Aware-subtitles | IEMOCAP 5-class validation: paper reports accuracy about 0.66 and macro-F1 about 0.57; committed best-epoch confusion matrix implies accuracy=653/980=66.63% at epoch 16 |
| 24 | include | https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning | pooled RAVDESS+CREMA-D 3-class test: CNN accuracy=82.60% (stored 0.8260234); secondary CRNN accuracy=80.56% |
| 25 | include | https://github.com/VeerR13/Speech-Emotion-Recognition | pooled RAVDESS+TESS+CREMA-D+SAVEE 8-class held-out standard test: accuracy=89.29%; macro-F1=91.09% |
| 26 | include | https://github.com/yipenglai/Audio-Emotion-Classification | RAVDESS 7-class four-fold label-stratified CV: macro recall (called accuracy)=0.68155; across-fold SD=0.01009 |
| 27 | include | https://github.com/VishHUB1/Multimodal-SER | CREMA-D framed log-Mel Inception: README claims accuracy improved from 75% to 94% |
| 28 | include | https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess | RAVDESS 8-class evaluation code is present, but the frozen repository reports no numeric score |
| 29 | include | https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition | IEMOCAP 4-class sole validation holdout: final epoch accuracy=0.653132; best observed epoch accuracy=0.664733 |
| 30 | include | https://github.com/vikrant-3009/SpeechEmotionRecognition | RAVDESS 4-class random holdout accuracy=56.77% (n_test=192) |
| 31 | include | https://github.com/AitanaESCI/speech-emotion-recognition | IEMOCAP 4-class fixed-session holdout, selected WavLM audio-only model: test accuracy=0.641; test macro-F1=0.603 |
| 32 | include | https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER | RAVDESS speech+song 8-class random holdout: accuracy=0.8471153975 (~84.71%); macro-F1=0.82; weighted-F1=0.85; n_test=1040 |
| 33 | include | https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks | pooled RAVDESS+CREMA-D+TESS 8-class random holdout: test accuracy=71.46%; macro-F1=0.7483; weighted-F1=0.7146; n_test=2337 |
| 34 | include | https://github.com/mzarvandi/SER-wav2vec | IEMOCAP 11-class external predefined holdout: mean test cross-entropy loss=1.9321030474 after an interrupted training run |
| 35 | not_reached | horikita-99/Speech_Emotion_Recognition | rank 34 已满足累计 n=30，按停止规则不审计。 |

## 排除证据

### Rank 21 — Kani2k06/CODEALPHA-TASK-2 — E2_target_not_evaluated

README 声称使用 RAVDESS、TESS 与 EMO-DB，但冻结 commit 的唯一训练入口 `training.py:L18-L33` 只实现并调用 TESS loader；`training.py:L48-L74` 的划分、训练与评估也只消费该 TESS 数据流。故目标 EmoDB/RAVDESS 仅为 README 声称，没有目标数据集评估代码，按冻结 E2 排除。

- README：https://github.com/Kani2k06/CODEALPHA-TASK-2/blob/2eebb43d86c9121b7fb700c540704f512617d3f1/README.md#L1-L8
- 活动代码：https://github.com/Kani2k06/CODEALPHA-TASK-2/blob/2eebb43d86c9121b7fb700c540704f512617d3f1/training.py#L18-L74

## 纳入库逐字段摘要与证据

### Rank 22 — https://github.com/daribdb/speech-emotion-recognition-iemocap

- 冻结：`785c4433a31ff0bfe86eee5a6230a4fa5aebc6dc`（2026-04-14T16:14:48+05:00）；许可：`none_found`；证据强度：`high`。
- 结果：IEMOCAP 4-class 10-fold leave-one-speaker-out: WA=52.71% ± 4.39% SD; UA=52.36% ± 4.81% SD（link=`direct`）。
- 划分：`LOSO`；speaker=`enforced`。IEMOCAP=10-fold leave-one-speaker-out with a distinct circular validation speaker in each fold。
- 归一化：`no`。IEMOCAP=BatchNorm running statistics are learned on the training loader and used in eval mode; no corpus-fitted external scaler。
- 测试选择：`no_separate_validation`。IEMOCAP=validation speaker selects the best checkpoint; the held-out test speaker is evaluated only after reloading it。
- 增强：`no_train_only`。IEMOCAP=train loader uses augmentation; validation and test loaders explicitly disable it。
- 重复：`LOSO`，n_folds=10，n_seeds=1，variance=`sd`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：iemocap_emotion_recognition_csv.ipynb:L2239-L2240 stores the final WA/UA mean ± SD; https://github.com/daribdb/speech-emotion-recognition-iemocap/blob/785c4433a31ff0bfe86eee5a6230a4fa5aebc6dc/iemocap_emotion_recognition_csv.ipynb#L2239-L2240
- 划分证据：iemocap_emotion_recognition_csv.ipynb:L793-L841 enumerates speakers, assigns one test and one validation speaker, and excludes both from train; https://github.com/daribdb/speech-emotion-recognition-iemocap/blob/785c4433a31ff0bfe86eee5a6230a4fa5aebc6dc/iemocap_emotion_recognition_csv.ipynb#L793-L841
- 归一化证据：iemocap_emotion_recognition_csv.ipynb:L643-L674 defines model BatchNorm and L711/L748 switches train/eval modes; https://github.com/daribdb/speech-emotion-recognition-iemocap/blob/785c4433a31ff0bfe86eee5a6230a4fa5aebc6dc/iemocap_emotion_recognition_csv.ipynb#L643-L674
- 选择证据：iemocap_emotion_recognition_csv.ipynb:L867-L899 selects/saves on validation accuracy before the test evaluation at L899-L924; https://github.com/daribdb/speech-emotion-recognition-iemocap/blob/785c4433a31ff0bfe86eee5a6230a4fa5aebc6dc/iemocap_emotion_recognition_csv.ipynb#L867-L924
- 增强证据：iemocap_emotion_recognition_csv.ipynb:L434-L483 defines noise/pitch/stretch and L839-L841 enables it only for train; https://github.com/daribdb/speech-emotion-recognition-iemocap/blob/785c4433a31ff0bfe86eee5a6230a4fa5aebc6dc/iemocap_emotion_recognition_csv.ipynb#L839-L841
- 重复证据：iemocap_emotion_recognition_csv.ipynb:L793-L815 creates ten folds and L943-L946 computes across-fold SD; https://github.com/daribdb/speech-emotion-recognition-iemocap/blob/785c4433a31ff0bfe86eee5a6230a4fa5aebc6dc/iemocap_emotion_recognition_csv.ipynb#L943-L946
- 备注：README designates this notebook as the main training path. A separate iemocap_test.ipynb contains a random-split baseline and is not used for the primary judgment.

### Rank 23 — https://github.com/Yaowan410/Emotion-Aware-subtitles

- 冻结：`dc64ec9a268d2229211b51b9c76bf6fff01f6c49`（2026-01-20T23:52:22+08:00）；许可：`MIT`；证据强度：`medium`。
- 结果：IEMOCAP 5-class validation: paper reports accuracy about 0.66 and macro-F1 about 0.57; committed best-epoch confusion matrix implies accuracy=653/980=66.63% at epoch 16（link=`partial`）。
- 划分：`random`；speaker=`not_enforced`。IEMOCAP=random utterance-level 90/10 stratified holdout, random_state=42。
- 归一化：`no`。IEMOCAP=fixed pretrained WavLM feature-extractor/per-utterance waveform processing; no cross-sample fitted scaler。
- 测试选择：`yes_explicit`。IEMOCAP=the sole 10% holdout is evaluated every epoch, selects the saved best checkpoint, and supplies the reported validation result。
- 增强：`not_applicable`。IEMOCAP=current active code only resamples and deterministically truncates to the first six seconds; it defines no stochastic augmentation。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`yes`。
- 结果证据：bundled PDF pp.1,3,7 reports the approximate metrics and epoch-16 selection; Result/WechatIMG1657.jpg contains the 5x5 best-epoch matrix; https://github.com/Yaowan410/Emotion-Aware-subtitles/blob/dc64ec9a268d2229211b51b9c76bf6fff01f6c49/Result/WechatIMG1657.jpg
- 划分证据：Model.py:L90-L101 stratifies labels only and carries no speaker/session group into train_test_split; https://github.com/Yaowan410/Emotion-Aware-subtitles/blob/dc64ec9a268d2229211b51b9c76bf6fff01f6c49/Model.py#L90-L101
- 归一化证据：Model.py:L177-L215 instantiates the fixed pretrained feature extractor and processes each waveform independently; https://github.com/Yaowan410/Emotion-Aware-subtitles/blob/dc64ec9a268d2229211b51b9c76bf6fff01f6c49/Model.py#L177-L215
- 选择证据：Model.py:L368-L481 evaluates val each epoch and saves the maximum val_acc; https://github.com/Yaowan410/Emotion-Aware-subtitles/blob/dc64ec9a268d2229211b51b9c76bf6fff01f6c49/Model.py#L368-L481
- 增强证据：Model.py:L184-L215 is the complete collate path and contains no random crop or noise operation; https://github.com/Yaowan410/Emotion-Aware-subtitles/blob/dc64ec9a268d2229211b51b9c76bf6fff01f6c49/Model.py#L184-L215
- 重复证据：Model.py:L38-L64 fixes seed 42 and L93-L101 creates one split; no fold/seed loop exists; https://github.com/Yaowan410/Emotion-Aware-subtitles/blob/dc64ec9a268d2229211b51b9c76bf6fff01f6c49/Model.py#L38-L64
- 备注：The PDF says training uses random six-second crops plus Gaussian noise, whereas the frozen Model.py deterministically keeps waveform[:max_len_samples] for both loaders and adds no noise. The committed matrix title also differs from the current plotting title, so the exact result-to-code link is partial, not direct.

### Rank 24 — https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning

- 冻结：`7172379c53268a4f0b7fceacb91792efeddca247`（2026-06-30T10:51:41+05:30）；许可：`MIT`；证据强度：`high`。
- 结果：pooled RAVDESS+CREMA-D 3-class test: CNN accuracy=82.60% (stored 0.8260234); secondary CRNN accuracy=80.56%（link=`direct`）。
- 划分：`speaker_independent`；speaker=`enforced`。RAVDESS=pooled speaker-disjoint 70/15/15 train/validation/test; CREMA-D=same pooled partition。
- 归一化：`no`。RAVDESS=pooled mean/std fitted on X_train only; CREMA-D=same。
- 测试选择：`no_separate_validation`。RAVDESS=validation drives early stopping and test is evaluated afterward once; CREMA-D=same pooled path。
- 增强：`not_applicable`。RAVDESS=no active augmentation in the final CNN path; CREMA-D=same。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：README.md:L215-L244 reports both final models; final CNN notebook stores 0.8260234 at L392-L398; https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning/blob/7172379c53268a4f0b7fceacb91792efeddca247/notebooks/04_Final_Model/23_CNN_3emotions.ipynb#L392-L398
- 划分证据：04_dataset_split.ipynb:L232-L286 splits unique speakers at random_state=42, selects rows by speaker, and asserts all sets disjoint; https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning/blob/7172379c53268a4f0b7fceacb91792efeddca247/notebooks/01_Data_Preparation/04_dataset_split.ipynb#L232-L286
- 归一化证据：23_CNN_3emotions.ipynb:L140-L141 computes mean/std only from X_train before applying them to held-out arrays; https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning/blob/7172379c53268a4f0b7fceacb91792efeddca247/notebooks/04_Final_Model/23_CNN_3emotions.ipynb#L140-L141
- 选择证据：23_CNN_3emotions.ipynb:L285-L375 configures validation-based early stopping/training and L392-L398 separately evaluates test; https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning/blob/7172379c53268a4f0b7fceacb91792efeddca247/notebooks/04_Final_Model/23_CNN_3emotions.ipynb#L285-L398
- 增强证据：final notebook's loaded train/val/test arrays pass directly to model.fit without an augmentation generator (L274-L398); https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning/blob/7172379c53268a4f0b7fceacb91792efeddca247/notebooks/04_Final_Model/23_CNN_3emotions.ipynb#L274-L398
- 重复证据：04_dataset_split.ipynb:L232-L239 fixes the only split seed and the final notebook contains one model.fit; https://github.com/Mayureshkore07/Cross-Dataset-Speech-Emotion-Recognition-Using-Deep-Learning/blob/7172379c53268a4f0b7fceacb91792efeddca247/notebooks/01_Data_Preparation/04_dataset_split.ipynb#L232-L239
- 备注：The 82.60% score is a pooled cross-corpus, three-emotion estimate; it is not a dataset-specific RAVDESS or CREMA-D score.

### Rank 25 — https://github.com/VeerR13/Speech-Emotion-Recognition

- 冻结：`39473a22021ffe5a661bf6c53380309ba7a533ef`（2026-04-23T12:16:47+05:30）；许可：`MIT`；证据强度：`high`。
- 结果：pooled RAVDESS+TESS+CREMA-D+SAVEE 8-class held-out standard test: accuracy=89.29%; macro-F1=91.09%（link=`direct`）。
- 划分：`random`；speaker=`not_enforced`。RAVDESS=pooled utterance-level 85/7.5/7.5 label-stratified split; CREMA-D=same。
- 归一化：`no`。RAVDESS=pooled StandardScaler fit only on idx_tr; CREMA-D=same。
- 测试选择：`no_separate_validation`。RAVDESS=validation selects/restores the checkpoint; standard test is evaluated after training; CREMA-D=same pooled path。
- 增强：`no_train_only`。RAVDESS=SpecAugment/noise/mixup generator receives idx_tr only; CREMA-D=same; reported primary result is standard no-TTA test。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：README.md:L4-L24 reports the exact metrics and identifies the CPU/no-TTA result; https://github.com/VeerR13/Speech-Emotion-Recognition/blob/39473a22021ffe5a661bf6c53380309ba7a533ef/README.md#L4-L24
- 划分证据：SER_v2_kaggle.ipynb:L191 performs ordinary stratified train_test_split calls and the parsers do not retain speaker groups; https://github.com/VeerR13/Speech-Emotion-Recognition/blob/39473a22021ffe5a661bf6c53380309ba7a533ef/SER_v2_kaggle.ipynb#L191
- 归一化证据：SER_v2_kaggle.ipynb:L191 fits StandardScaler on X_flat_base[idx_tr] and only transforms val/test; https://github.com/VeerR13/Speech-Emotion-Recognition/blob/39473a22021ffe5a661bf6c53380309ba7a533ef/SER_v2_kaggle.ipynb#L191
- 选择证据：SER_v2_kaggle.ipynb:L245 selects by val_accuracy and L252 evaluates the loaded best model on test; https://github.com/VeerR13/Speech-Emotion-Recognition/blob/39473a22021ffe5a661bf6c53380309ba7a533ef/SER_v2_kaggle.ipynb#L245-L252
- 增强证据：SER_v2_kaggle.ipynb:L198 constructs AugMixupGen only from X_2d_base[idx_tr]/Xf_tr_scaled; https://github.com/VeerR13/Speech-Emotion-Recognition/blob/39473a22021ffe5a661bf6c53380309ba7a533ef/SER_v2_kaggle.ipynb#L198
- 重复证据：SER_v2_kaggle.ipynb:L191 fixes the only split seed and L245 contains one fit; https://github.com/VeerR13/Speech-Emotion-Recognition/blob/39473a22021ffe5a661bf6c53380309ba7a533ef/SER_v2_kaggle.ipynb#L191
- 备注：Metrics pool four corpora and are not target-dataset-specific. A secondary TTA evaluation exists, but README explicitly identifies the reported 89.29%/91.09% as full held-out CPU evaluation without TTA.

### Rank 26 — https://github.com/yipenglai/Audio-Emotion-Classification

- 冻结：`426d7d473238777ef4e33fdb8d8e053f8cdaae48`（2019-12-07T21:15:08-05:00）；许可：`none_found`；证据强度：`high`。
- 结果：RAVDESS 7-class four-fold label-stratified CV: macro recall (called accuracy)=0.68155; across-fold SD=0.01009（link=`direct`）。
- 划分：`random`；speaker=`not_enforced`。RAVDESS=4-fold label-stratified shuffled KFold over utterances。
- 归一化：`no`。RAVDESS=feature mean/std recomputed from each CV training fold and applied to all observations。
- 测试选择：`yes_explicit`。RAVDESS=each held-out fold is evaluated every ten epochs and its maximum macro recall is the fold result。
- 增强：`not_applicable`。RAVDESS=no stochastic augmentation in the active precomputed-feature CV path。
- 重复：`kfold_single_run`，n_folds=4，n_seeds=1，variance=`sd`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：crnn/train.ipynb:L14427-L14436 stores the mean and SD; L161/L267 aliases macro recall as accuracy; https://github.com/yipenglai/Audio-Emotion-Classification/blob/426d7d473238777ef4e33fdb8d8e053f8cdaae48/crnn/train.ipynb#L14427-L14436
- 划分证据：crnn/train.ipynb:L4686-L4706 uses StratifiedKFold on labels only; actor IDs are not supplied to the splitter; https://github.com/yipenglai/Audio-Emotion-Classification/blob/426d7d473238777ef4e33fdb8d8e053f8cdaae48/crnn/train.ipynb#L4686-L4706
- 归一化证据：crnn/train.ipynb:L4635-L4643 defines training-index-only normalization and L4695-L4700 calls it within each fold; https://github.com/yipenglai/Audio-Emotion-Classification/blob/426d7d473238777ef4e33fdb8d8e053f8cdaae48/crnn/train.ipynb#L4635-L4643
- 选择证据：crnn/train.ipynb:L161-L267 evaluates the held-out fold repeatedly and updates best_acc_val from it; https://github.com/yipenglai/Audio-Emotion-Classification/blob/426d7d473238777ef4e33fdb8d8e053f8cdaae48/crnn/train.ipynb#L161-L267
- 增强证据：crnn/train.ipynb:L4686-L4706 directly indexes X_raw and contains no augmentation call; https://github.com/yipenglai/Audio-Emotion-Classification/blob/426d7d473238777ef4e33fdb8d8e053f8cdaae48/crnn/train.ipynb#L4686-L4706
- 重复证据：crnn/train.ipynb:L4686-L4706 defines four folds and L14427-L14436 summarizes their SD; https://github.com/yipenglai/Audio-Emotion-Classification/blob/426d7d473238777ef4e33fdb8d8e053f8cdaae48/crnn/train.ipynb#L4686-L4706
- 备注：The splitter uses shuffle=True without a random_state and the training notebook fixes no RNG seed. The reported 'accuracy' is sklearn macro recall, not ordinary micro accuracy.

### Rank 27 — https://github.com/VishHUB1/Multimodal-SER

- 冻结：`f268b8309889cfc59a2dc8e1cf4aa4c47eb631fc`（2025-07-29T23:28:12+05:30）；许可：`none_found`；证据强度：`medium`。
- 结果：CREMA-D framed log-Mel Inception: README claims accuracy improved from 75% to 94%（link=`partial`）。
- 划分：`random`；speaker=`not_enforced`。CREMA-D=random image/utterance-level 70/30 stratified holdout, random_state=42。
- 归一化：`no`。CREMA-D=fixed ImageNet mean/std constants; no corpus-fitted current-data scaler。
- 测试选择：`test_exposed_each_epoch`。CREMA-D=the sole test loader is passed as val_loader and scored every epoch; no automatic best-epoch selection statement is present。
- 增强：`not_applicable`。CREMA-D=framed Inception path uses the same deterministic resize/to-tensor/fixed-normalize transform without stochastic augmentation。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：README.md:L1-L2 contains the dataset, 31-frame path, and claimed accuracy; https://github.com/VishHUB1/Multimodal-SER/blob/f268b8309889cfc59a2dc8e1cf4aa4c47eb631fc/README.md#L1-L2
- 划分证据：Inception_framing.py:L156-L176 performs ordinary train_test_split over spectrogram paths and labels; https://github.com/VishHUB1/Multimodal-SER/blob/f268b8309889cfc59a2dc8e1cf4aa4c47eb631fc/Inception_framing.py#L156-L176
- 归一化证据：Inception_framing.py:L164-L169 applies fixed ImageNet normalization to both sets; https://github.com/VishHUB1/Multimodal-SER/blob/f268b8309889cfc59a2dc8e1cf4aa4c47eb631fc/Inception_framing.py#L164-L169
- 选择证据：Inception_framing.py:L45-L109 evaluates val_loader each epoch and L188-L206 passes/reuses test_loader; https://github.com/VishHUB1/Multimodal-SER/blob/f268b8309889cfc59a2dc8e1cf4aa4c47eb631fc/Inception_framing.py#L45-L109
- 增强证据：Inception_framing.py:L164-L176 constructs both loaders from the same deterministic transform; https://github.com/VishHUB1/Multimodal-SER/blob/f268b8309889cfc59a2dc8e1cf4aa4c47eb631fc/Inception_framing.py#L164-L176
- 重复证据：Inception_framing.py:L156-L191 contains one split and one training invocation; https://github.com/VishHUB1/Multimodal-SER/blob/f268b8309889cfc59a2dc8e1cf4aa4c47eb631fc/Inception_framing.py#L156-L191
- 备注：The README's exact 31-frame wording maps most closely to Inception_framing.py, but no frozen output artifact contains 94% and the repository has several alternative model scripts; therefore result linkage is partial.

### Rank 28 — https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess

- 冻结：`6a861f77a01ca7ad75336219f220920ce0a36af3`（2026-06-04T12:00:52+05:30）；许可：`none_found`；证据强度：`high`。
- 结果：RAVDESS 8-class evaluation code is present, but the frozen repository reports no numeric score（link=`direct`）。
- 划分：`random`；speaker=`not_enforced`。RAVDESS=random utterance-level 80/20 label-stratified holdout, random_state=42。
- 归一化：`yes`。RAVDESS=StandardScaler is fit on all utterances before the train/test split。
- 测试选择：`yes_explicit`。RAVDESS=the sole test holdout is validation_data for EarlyStopping and is evaluated again for the final metric。
- 增强：`not_applicable`。RAVDESS=no active augmentation in the complete MFCC extraction and dense-model training path。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：README.md:L165-L175 names evaluated metrics without values; train.py:L153-L157 defines and prints test accuracy; https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess/blob/6a861f77a01ca7ad75336219f220920ce0a36af3/train.py#L153-L157
- 划分证据：train.py:L54-L79 loads actor folders but discards actor IDs; L103-L109 splits samples by labels only; https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess/blob/6a861f77a01ca7ad75336219f220920ce0a36af3/train.py#L54-L109
- 归一化证据：train.py:L95-L109 calls scaler.fit_transform(X) before train_test_split; https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess/blob/6a861f77a01ca7ad75336219f220920ce0a36af3/train.py#L95-L109
- 选择证据：train.py:L138-L156 monitors val_loss on X_test/y_test and then evaluates the same arrays; https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess/blob/6a861f77a01ca7ad75336219f220920ce0a36af3/train.py#L138-L156
- 增强证据：train.py:L33-L49 is the complete feature extractor and L114-L151 the training path; neither generates augmented samples; https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess/blob/6a861f77a01ca7ad75336219f220920ce0a36af3/train.py#L33-L49
- 重复证据：train.py:L103-L109 creates one split and L144-L151 calls one fit; https://github.com/VINUVPOTTY/voice-emotion-recognition-ravdess/blob/6a861f77a01ca7ad75336219f220920ce0a36af3/train.py#L103-L109
- 备注：The committed Keras model/scaler are not treated as a numeric report. Inclusion rests on the explicit RAVDESS evaluation path, not on the README's illustrative 92.5% prediction-confidence example.

### Rank 29 — https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition

- 冻结：`ac73e9f95b23f132cdf60ccbd39d0807b60e2db0`（2022-03-11T12:14:40-08:00）；许可：`none_found`；证据强度：`high`。
- 结果：IEMOCAP 4-class sole validation holdout: final epoch accuracy=0.653132; best observed epoch accuracy=0.664733（link=`direct`）。
- 划分：`random`；speaker=`not_enforced`。IEMOCAP=random spectrogram/utterance-level 80/20 validation split, numpy seed=42。
- 归一化：`no`。IEMOCAP=fixed ImageNet normalization constants, not statistics fitted from the current corpus。
- 测试选择：`test_exposed_each_epoch`。IEMOCAP=the sole validation holdout is scored for all 15 epochs and then used for interpretation; no distinct final test exists。
- 增强：`not_applicable`。IEMOCAP=the DataBunch explicitly sets ds_tfms=None。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：emo_rec.ipynb:L250-L339 stores the two fit_one_cycle histories, including the final and best validation accuracies; https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition/blob/ac73e9f95b23f132cdf60ccbd39d0807b60e2db0/emo_rec.ipynb#L250-L339
- 划分证据：emo_rec.ipynb:L122-L133 uses ImageDataBunch.from_folder(valid_pct=0.2) with no speaker/session grouping; https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition/blob/ac73e9f95b23f132cdf60ccbd39d0807b60e2db0/emo_rec.ipynb#L122-L133
- 归一化证据：emo_rec.ipynb:L131-L133 calls normalize(imagenet_stats); https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition/blob/ac73e9f95b23f132cdf60ccbd39d0807b60e2db0/emo_rec.ipynb#L131-L133
- 选择证据：emo_rec.ipynb:L191-L339 fits on the sole validation DataBunch and records validation accuracy each epoch; https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition/blob/ac73e9f95b23f132cdf60ccbd39d0807b60e2db0/emo_rec.ipynb#L191-L339
- 增强证据：emo_rec.ipynb:L131-L133 sets ds_tfms=None; https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition/blob/ac73e9f95b23f132cdf60ccbd39d0807b60e2db0/emo_rec.ipynb#L131-L133
- 重复证据：emo_rec.ipynb:L131-L133 fixes the only split seed and L250-L339 contains the single training history; https://github.com/flaviorainhoavila/IEMOCAPspeechEmotionRecognition/blob/ac73e9f95b23f132cdf60ccbd39d0807b60e2db0/emo_rec.ipynb#L131-L133
- 备注：The repository labels this set validation and has no independent test set. The reported best observed accuracy is descriptive exposure, not a separately validated checkpoint estimate.

### Rank 30 — https://github.com/vikrant-3009/SpeechEmotionRecognition

- 冻结：`8268ac652980817c71da0215da5fc2a5f949711c`（2022-03-26T18:25:02+05:30）；许可：`none_found`；证据强度：`high`。
- 结果：RAVDESS 4-class random holdout accuracy=56.77% (n_test=192)（link=`direct`）。
- 划分：`random`；speaker=`not_enforced`。RAVDESS=random utterance-level 75/25 holdout, random_state=9。
- 归一化：`not_applicable`。RAVDESS=per-utterance fixed feature extraction only; no fitted cross-sample scaler。
- 测试选择：`no_fixed_training`。RAVDESS=MLP uses one fixed max_iter schedule and test predictions occur only after model.fit。
- 增强：`not_applicable`。RAVDESS=no active augmentation in the notebook's feature/load/train path。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：main.ipynb:L229-L232 stores Accuracy: 56.77%; L156-L165 stores the 576/192 train/test counts; https://github.com/vikrant-3009/SpeechEmotionRecognition/blob/8268ac652980817c71da0215da5fc2a5f949711c/main.ipynb#L229-L232
- 划分证据：main.ipynb:L85-L110 extracts labels but not actor groups and calls ordinary train_test_split; https://github.com/vikrant-3009/SpeechEmotionRecognition/blob/8268ac652980817c71da0215da5fc2a5f949711c/main.ipynb#L85-L110
- 归一化证据：main.ipynb:L1-L110 imports no scaler and passes extracted per-file features directly to the split/model; https://github.com/vikrant-3009/SpeechEmotionRecognition/blob/8268ac652980817c71da0215da5fc2a5f949711c/main.ipynb#L1-L110
- 选择证据：main.ipynb:L193-L232 fits once, then predicts/evaluates test without test-driven checkpoint selection; https://github.com/vikrant-3009/SpeechEmotionRecognition/blob/8268ac652980817c71da0215da5fc2a5f949711c/main.ipynb#L193-L232
- 增强证据：main.ipynb:L1-L110 defines the complete data path and creates no augmented samples; https://github.com/vikrant-3009/SpeechEmotionRecognition/blob/8268ac652980817c71da0215da5fc2a5f949711c/main.ipynb#L1-L110
- 重复证据：main.ipynb:L85-L110 creates one split and L193-L232 one fit/evaluation; https://github.com/vikrant-3009/SpeechEmotionRecognition/blob/8268ac652980817c71da0215da5fc2a5f949711c/main.ipynb#L85-L110
- 备注：Only calm, happy, fearful, and disgust are retained. The split seed is fixed, but MLPClassifier random_state is left at its stochastic default.

### Rank 31 — https://github.com/AitanaESCI/speech-emotion-recognition

- 冻结：`ccec93c7605cd4b962a561b1d0e90e2e9af429e5`（2026-07-06T16:51:56+02:00）；许可：`none_found_root; Apache-2.0_nested_loader_only`；证据强度：`medium`。
- 结果：IEMOCAP 4-class fixed-session holdout, selected WavLM audio-only model: test accuracy=0.641; test macro-F1=0.603（link=`partial`）。
- 划分：`speaker_independent`；speaker=`enforced`。IEMOCAP=fixed session/speaker holdout: train Ses03-Ses05 actors, validation Ses02 actors, test Ses01 actors。
- 归一化：`no`。IEMOCAP=raw-waveform WavLM path uses fixed pretrained/internal normalization and per-utterance resampling/truncation; no corpus-fitted scaler。
- 测试选择：`yes_explicit`。IEMOCAP=fixed test metrics are compared across eight experiment categories and explicitly determine the documented final model; within one CLI run validation remains separate。
- 增强：`not_applicable`。IEMOCAP=final WavLM wrapper reads raw audio and bypasses the spectrogram/pitch augmentation of the underlying dataset。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：README.md:L167-L170 and L419-L435 reports the selected WavLM test metrics; https://github.com/AitanaESCI/speech-emotion-recognition/blob/ccec93c7605cd4b962a561b1d0e90e2e9af429e5/README.md#L419-L435
- 划分证据：loader core.py:L531-L547,L775-L785,L820-L824 constructs fixed session-specific actor partitions; https://github.com/AitanaESCI/speech-emotion-recognition/blob/ccec93c7605cd4b962a561b1d0e90e2e9af429e5/shared-dataset-loader/nbdev-upc-aidl-iemocap-datasets/src/nbdev_upc_aidl_iemocap_datasets/core.py#L531-L547
- 归一化证据：train-cli/models/wavlm.py:L64-L107 and L140-L202 implements the raw-waveform path without a current-corpus scaler; https://github.com/AitanaESCI/speech-emotion-recognition/blob/ccec93c7605cd4b962a561b1d0e90e2e9af429e5/train-cli/models/wavlm.py#L64-L107
- 选择证据：README.md:L415-L435 and L478-L489 selects the final model from test comparisons; train-cli/train.py:L604-L673 evaluates a run; https://github.com/AitanaESCI/speech-emotion-recognition/blob/ccec93c7605cd4b962a561b1d0e90e2e9af429e5/README.md#L415-L435
- 增强证据：train-cli/models/wavlm.py:L64-L137 uses the raw-audio wrapper; https://github.com/AitanaESCI/speech-emotion-recognition/blob/ccec93c7605cd4b962a561b1d0e90e2e9af429e5/train-cli/models/wavlm.py#L64-L137
- 重复证据：train-cli/train.py:L99-L106,L239,L410-L418,L573-L673 shows one seed and one fixed-partition run; https://github.com/AitanaESCI/speech-emotion-recognition/blob/ccec93c7605cd4b962a561b1d0e90e2e9af429e5/train-cli/train.py#L99-L106
- 备注：README reports Macro-F1, but current CLI computes weighted F1 at train-cli/train.py:L688-L702 and no frozen run artifact links the exact README values to this commit, hence partial linkage. Apache-2.0 is present only in the nested shared dataset loader, not at repository root.

### Rank 32 — https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER

- 冻结：`9557d621afb724d3dd3b4fe16918c6dbd484c527`（2024-03-18T10:58:42-03:00）；许可：`none_found`；证据强度：`high`。
- 结果：RAVDESS speech+song 8-class random holdout: accuracy=0.8471153975 (~84.71%); macro-F1=0.82; weighted-F1=0.85; n_test=1040（link=`direct`）。
- 划分：`random`；speaker=`not_enforced`。RAVDESS=ordinary utterance-level 70/30 train_test_split(random_state=1) without actor grouping。
- 归一化：`not_applicable`。RAVDESS=per-utterance MFCC temporal mean only; no fitted cross-sample normalizer。
- 测试选择：`yes_explicit`。RAVDESS=the sole test holdout is validation_data for all 50 epochs, drives save_best_only checkpointing, and supplies final evaluation。
- 增强：`not_applicable`。RAVDESS=no active augmentation in feature extraction or training。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：notebook:L13370-L13379 and L13979-L13993 stores the evaluation and classification report; https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER/blob/9557d621afb724d3dd3b4fe16918c6dbd484c527/Reconhecimento_de_Emo%C3%A7%C3%B5es_pela_Fala_RAVDESS.ipynb#L13370-L13379
- 划分证据：notebook:L12719 performs the ordinary split; https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER/blob/9557d621afb724d3dd3b4fe16918c6dbd484c527/Reconhecimento_de_Emo%C3%A7%C3%B5es_pela_Fala_RAVDESS.ipynb#L12719
- 归一化证据：notebook:L1262-L1267 contains the per-file feature path; https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER/blob/9557d621afb724d3dd3b4fe16918c6dbd484c527/Reconhecimento_de_Emo%C3%A7%C3%B5es_pela_Fala_RAVDESS.ipynb#L1262-L1267
- 选择证据：notebook:L13076,L13327-L13338,L13370-L13379 reuses the test set for checkpoint selection and final scoring; https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER/blob/9557d621afb724d3dd3b4fe16918c6dbd484c527/Reconhecimento_de_Emo%C3%A7%C3%B5es_pela_Fala_RAVDESS.ipynb#L13327-L13338
- 增强证据：notebook:L1262-L1267 and L13327-L13338 contains the complete primary path without augmentation; https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER/blob/9557d621afb724d3dd3b4fe16918c6dbd484c527/Reconhecimento_de_Emo%C3%A7%C3%B5es_pela_Fala_RAVDESS.ipynb#L1262-L1267
- 重复证据：notebook:L12719 and L13327-L13338 shows one split and one fit; https://github.com/GaybsGimenez/RAVDESS-Speech-Emotion-Recognition-SER/blob/9557d621afb724d3dd3b4fe16918c6dbd484c527/Reconhecimento_de_Emo%C3%A7%C3%B5es_pela_Fala_RAVDESS.ipynb#L12719
- 备注：The displayed 84.71% comes from the last in-memory epoch rather than a reloaded best checkpoint, but the same test set was exposed every epoch and controlled checkpoint saving.

### Rank 33 — https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks

- 冻结：`3e73295a4446ec1a708485c27ff4570e9afdff95`（2025-07-09T19:29:44+05:30）；许可：`none_found`；证据强度：`high`。
- 结果：pooled RAVDESS+CREMA-D+TESS 8-class random holdout: test accuracy=71.46%; macro-F1=0.7483; weighted-F1=0.7146; n_test=2337（link=`direct`）。
- 划分：`random`；speaker=`not_enforced`。RAVDESS=pooled utterance split with CREMA-D and TESS: 20% test then 10% of remaining train for validation, labels stratified, random_state=42; CREMA-D=same。
- 归一化：`no`。RAVDESS=per-utterance waveform peak normalization and MFCC/mel z-normalization; CREMA-D=same; no cross-sample statistics。
- 测试选择：`no_separate_validation`。RAVDESS=separate validation split drives early stopping/checkpointing and test is evaluated afterward; CREMA-D=same pooled path。
- 增强：`not_applicable`。RAVDESS=deterministic denoising/resampling/padding before splitting but no additional stochastic samples; CREMA-D=same。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：DCNN_model.ipynb:L1604 and L1683-L1706 stores test metrics; https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks/blob/3e73295a4446ec1a708485c27ff4570e9afdff95/DCNN_model.ipynb#L1683-L1706
- 划分证据：DCNN_model.ipynb:L898-L904 performs label-stratified splits without speaker groups; https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks/blob/3e73295a4446ec1a708485c27ff4570e9afdff95/DCNN_model.ipynb#L898-L904
- 归一化证据：DCNN_model.ipynb:L351 and L770-L781 contains per-utterance normalization; https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks/blob/3e73295a4446ec1a708485c27ff4570e9afdff95/DCNN_model.ipynb#L770-L781
- 选择证据：DCNN_model.ipynb:L1004-L1020 configures validation selection and L1604 evaluates test; https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks/blob/3e73295a4446ec1a708485c27ff4570e9afdff95/DCNN_model.ipynb#L1004-L1020
- 增强证据：DCNN_model.ipynb:L330-L354 and L898-L904 shows deterministic preprocessing followed by the split; https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks/blob/3e73295a4446ec1a708485c27ff4570e9afdff95/DCNN_model.ipynb#L330-L354
- 重复证据：DCNN_model.ipynb:L898-L904 and L1004-L1020 contains one split/run; https://github.com/Abhiramkura/Speech-Emotion-recognition-using-deformable-convolutional-neural-networks/blob/3e73295a4446ec1a708485c27ff4570e9afdff95/DCNN_model.ipynb#L898-L904
- 备注：Metrics are pooled across three corpora, not dataset-specific. README claims strong speaker generalization, but the active code uses label-stratified utterance splitting without speaker groups.

### Rank 34 — https://github.com/mzarvandi/SER-wav2vec

- 冻结：`3a78aa177edc4e847bcfccbffc2cd88d0aab96f6`（2021-08-08T17:10:24+04:30）；许可：`none_found`；证据强度：`medium`。
- 结果：IEMOCAP 11-class external predefined holdout: mean test cross-entropy loss=1.9321030474 after an interrupted training run（link=`direct`）。
- 划分：`unknown`；speaker=`unknown`。IEMOCAP=external Kaggle Train/Train and Test/Test directories; split generator and file manifests are absent。
- 归一化：`no`。IEMOCAP=raw waveforms feed a fixed pretrained wav2vec2 model with internal pretrained GroupNorm/LayerNorm; no corpus-fitted current-data scaler。
- 测试选择：`no_fixed_training`。IEMOCAP=test loader is invoked once after the manual training loop terminates; no test-driven checkpoint/hyperparameter selection appears on the executed path。
- 增强：`not_applicable`。IEMOCAP=no augmentation operation in the raw-waveform dataset or executed training path。
- 重复：`single_split_single_seed`，n_folds=1，n_seeds=1，variance=`none`。
- 论文—代码：`not_applicable_no_paper`。
- 结果证据：SER-wav2vec.ipynb:L1356-L1388 stores the post-interruption test loss; https://github.com/mzarvandi/SER-wav2vec/blob/3a78aa177edc4e847bcfccbffc2cd88d0aab96f6/SER-wav2vec.ipynb#L1356-L1388
- 划分证据：SER-wav2vec.ipynb:L409-L418 and L565-L566 consumes external directory labels without exposing split construction; https://github.com/mzarvandi/SER-wav2vec/blob/3a78aa177edc4e847bcfccbffc2cd88d0aab96f6/SER-wav2vec.ipynb#L409-L418
- 归一化证据：SER-wav2vec.ipynb:L119-L151 and L565-L587 shows fixed pretrained processing without a current-data scaler; https://github.com/mzarvandi/SER-wav2vec/blob/3a78aa177edc4e847bcfccbffc2cd88d0aab96f6/SER-wav2vec.ipynb#L119-L151
- 选择证据：SER-wav2vec.ipynb:L625-L646,L1059-L1067,L1356-L1388 shows training then a single executed test pass; https://github.com/mzarvandi/SER-wav2vec/blob/3a78aa177edc4e847bcfccbffc2cd88d0aab96f6/SER-wav2vec.ipynb#L1356-L1388
- 增强证据：SER-wav2vec.ipynb:L409-L418 and L565-L646 contains the complete dataset/training path without augmentation; https://github.com/mzarvandi/SER-wav2vec/blob/3a78aa177edc4e847bcfccbffc2cd88d0aab96f6/SER-wav2vec.ipynb#L565-L646
- 重复证据：SER-wav2vec.ipynb:L565-L646 and L1356-L1388 contains one attempt/evaluation; https://github.com/mzarvandi/SER-wav2vec/blob/3a78aa177edc4e847bcfccbffc2cd88d0aab96f6/SER-wav2vec.ipynb#L565-L646
- 备注：Training was interrupted by KeyboardInterrupt during the planned five-epoch loop, so the loss is a partial-run quantitative evaluation, not a completed benchmark. Later Trainer code assigns test as eval_dataset but has no stored execution output.

## 停止位

rank 34 是本轮第 13 个新增纳入库，加上 ranks 1–20 的 17 个纳入库后总体恰为 30。按预注册的排除即补抽/达到 n=30 立即停止规则，rank 35 及以后不进入 Stage-B；这里没有把停止后的仓库误写成排除。

<!-- source-batch: p0_5_audit_batch_12_20.md -->

# P0.5 random-sample initial audit: draw ranks 12–20

- Audit time: 2026-08-10 18:26:04 UTC-04:00
- Reviewer: `p0_7_reliability`
- Protocols applied: frozen `P0_5_PROTOCOL.md` and inherited `P0_PROTOCOL.md`
- Scope: static Stage-B screening followed by the 38-field initial audit; no candidate code, model, pickle, notebook cell, dependency installer, or GPU workload was executed.
- Output rule: included repositories appear in `p0_5_audit_batch_12_20.csv`; `repository_id` is deliberately blank and `verification_status=pending` for later independent review.

## Stage-B screening (performed before field extraction)

The nine canonical URLs are distinct from each other and from the 32 repositories already present in `survey_table.csv`. No same-content mirror or unmodified fork was found within the audited set. Rank 18 credits an upstream repository, but its frozen tree contains substantive speaker-aware split and GroupKFold revisions, so it is not treated as an E5 mirror.

| draw rank | candidate | frozen repository / HEAD | Stage-B decision | first E-code | eligibility evidence |
|---:|---|---|---|---|---|
| 12 | C147 | `shreyash-alt/Emotion_Recognition_RAVDESS` / `36743de708023c2ae1a5aa7b538e79b79c3337cd` | include | — | RAVDESS scope and executable notebook workflow are documented in [README.md:L9-L22](https://github.com/shreyash-alt/Emotion_Recognition_RAVDESS/blob/36743de708023c2ae1a5aa7b538e79b79c3337cd/README.md#L9-L22); the notebook trains and reports held-out predictions at [emotionravdess2.ipynb:L1103-L1216](https://github.com/shreyash-alt/Emotion_Recognition_RAVDESS/blob/36743de708023c2ae1a5aa7b538e79b79c3337cd/emotionravdess2.ipynb#L1103-L1216). |
| 13 | C132 | `KangHyunWook/Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset` / `dd366fbcde5a8248195a47b7c0efd974f25f74c2` | include | — | README identifies RAVDESS audio-visual SER and the feature artifact at [README.md:L3-L17](https://github.com/KangHyunWook/Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset/blob/dd366fbcde5a8248195a47b7c0efd974f25f74c2/README.md#L3-L17); `train.py` contains train/dev/test execution through [train.py:L119-L252](https://github.com/KangHyunWook/Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset/blob/dd366fbcde5a8248195a47b7c0efd974f25f74c2/train.py#L119-L252). |
| 14 | C345 | `AnkushMalaker/pretrained-dcnn-attention-ser` / `6d5e27b6cb17d92c9d61c2c023faf518f651bcef` | include | — | Although the candidate note targeted EmoDB, the frozen repository explicitly supplies a RAVDESS SER path at [README.md:L27-L37](https://github.com/AnkushMalaker/pretrained-dcnn-attention-ser/blob/6d5e27b6cb17d92c9d61c2c023faf518f651bcef/README.md#L27-L37), which is also a protocol target. Missing EmoDB/IEMOCAP paths are retained as unknown fields, not used to exclude the repository. |
| 15 | C252 | `dori2063/SER_Augmentation_CycleGAN` / `86648637f4fd5dcb564ff2c579fd7cbbe7816f0d` | include | — | README names IEMOCAP, openSMILE, CycleGAN augmentation and SVM at [README.md:L1-L4](https://github.com/dori2063/SER_Augmentation_CycleGAN/blob/86648637f4fd5dcb564ff2c579fd7cbbe7816f0d/README.md#L1-L4); the frozen script contains feature loading, augmentation training and SVM evaluation at [run.py:L69-L225](https://github.com/dori2063/SER_Augmentation_CycleGAN/blob/86648637f4fd5dcb564ff2c579fd7cbbe7816f0d/run.py#L69-L225). External split arrays are missing, so split semantics remain unknown. |
| 16 | C010 | `sanskardherange/codeAlpha_EmotionRecognition` / `9b8d1fc934a62c0b1b155ab2feb9d113d85a70f3` | include | — | README identifies RAVDESS and MFCC SER at [README.md:L12-L28](https://github.com/sanskardherange/codeAlpha_EmotionRecognition/blob/9b8d1fc934a62c0b1b155ab2feb9d113d85a70f3/README.md#L12-L28); feature extraction and model fitting are present in [extract_features.py:L12-L35](https://github.com/sanskardherange/codeAlpha_EmotionRecognition/blob/9b8d1fc934a62c0b1b155ab2feb9d113d85a70f3/extract_features.py#L12-L35) and [train_model.py:L8-L34](https://github.com/sanskardherange/codeAlpha_EmotionRecognition/blob/9b8d1fc934a62c0b1b155ab2feb9d113d85a70f3/train_model.py#L8-L34). |
| 17 | C202 | `didar-ali-deed/A-hybrid-Approach-To-SER` / `5f704dd990a586d0485dca16aeb66048e049a8ee` | include | — | README names RAVDESS, CREMA-D, TESS and SAVEE at [README.md:L11-L30](https://github.com/didar-ali-deed/A-hybrid-Approach-To-SER/blob/5f704dd990a586d0485dca16aeb66048e049a8ee/README.md#L11-L30); the frozen classifier implements train/validation/test and evaluation at [scripts/train_emotion_classifier.py:L52-L175](https://github.com/didar-ali-deed/A-hybrid-Approach-To-SER/blob/5f704dd990a586d0485dca16aeb66048e049a8ee/scripts/train_emotion_classifier.py#L52-L175). |
| 18 | C348 | `fdebrain/Speech-Emotion-Recognition-Emo-DB` / `55bc617644d61d8c5ce9d46df954776ece525f45` | include | — | README identifies EmoDB SER and the revised speaker-aware design at [README.md:L1-L18](https://github.com/fdebrain/Speech-Emotion-Recognition-Emo-DB/blob/55bc617644d61d8c5ce9d46df954776ece525f45/README.md#L1-L18); the application trains the full pipeline at [app.py:L24-L40](https://github.com/fdebrain/Speech-Emotion-Recognition-Emo-DB/blob/55bc617644d61d8c5ce9d46df954776ece525f45/app.py#L24-L40). |
| 19 | C145 | `Susi-brambilla/emotion-recognition-RAVDESS` / `7f56fb9ca521c37a06c0a856641536b547eca483` | include | — | README identifies RAVDESS audio/video SER and train/test entrypoints at [README.md:L1-L21](https://github.com/Susi-brambilla/emotion-recognition-RAVDESS/blob/7f56fb9ca521c37a06c0a856641536b547eca483/README.md#L1-L21); model training/test functions are present despite preprocessing-name defects, e.g. [audio_video_model.py:L74-L241](https://github.com/Susi-brambilla/emotion-recognition-RAVDESS/blob/7f56fb9ca521c37a06c0a856641536b547eca483/audio_video_model.py#L74-L241). |
| 20 | C171 | `Omkar-Gode/Speech-Emotion-Recognition` / `ee0719fab738c68914a71ababe5a914a29fc6cbd` | include | — | README identifies CREMA-D SER and CNN training at [README.md:L3-L7](https://github.com/Omkar-Gode/Speech-Emotion-Recognition/blob/ee0719fab738c68914a71ababe5a914a29fc6cbd/README.md#L3-L7); the notebook contains augmentation, splitting, training and evaluation at [Speech_Emotion_Recognition.ipynb:L1894-L1952](https://github.com/Omkar-Gode/Speech-Emotion-Recognition/blob/ee0719fab738c68914a71ababe5a914a29fc6cbd/Speech_Emotion_Recognition.ipynb#L1894-L1952) and [L2640-L3197](https://github.com/Omkar-Gode/Speech-Emotion-Recognition/blob/ee0719fab738c68914a71ababe5a914a29fc6cbd/Speech_Emotion_Recognition.ipynb#L2640-L3197). |

### Excluded positions and first E-code

None. This batch has **9 included and 0 excluded** positions, so there is no exclusion rank or first E-code to report. This is a screening outcome, not an assumption that all nine implementations are rigorous or runnable without repair.

## Initial field judgments

| rank | split / speaker | normalization | test selection | augmentation | repetition |
|---:|---|---|---|---|---|
| 12 | random / not enforced | no (train-only scaler) | yes, explicit selection on the only holdout | not applicable | one split, one run |
| 13 | random / not enforced | unknown (raw preprocessing absent) | separate dev then test | unknown (raw preprocessing absent) | one split, one run |
| 14 | mixed: missing EmoDB/IEMOCAP paths; random RAVDESS / not enforced | mixed | mixed; RAVDESS-only holdout drives early stopping | mixed; RAVDESS code says not implemented | mixed; one RAVDESS split |
| 15 | unknown (external `section_list` semantics) | unknown (upstream arrays absent) | fixed training, test after fitting | train-only synthetic samples | same held section repeated three unseeded times |
| 16 | random / not enforced | not applicable | test exposed every epoch as `validation_data` | not applicable | one split, one run |
| 17 | pooled random / not enforced | not applicable | separate validation then test | not applicable | one split, one run |
| 18 | speaker-independent / enforced | **yes for inner GroupKFold validation**; outer test remains isolated | inner validation then outer test | not applicable | one outer split plus 8-fold inner GroupKFold |
| 19 | predefined actors / enforced | no (train-only audio stats; fixed video stats) | validation early stopping then distinct test | **yes: augmentation occurs before split and enters val/test** | one fixed split, one run |
| 20 | random augmented rows / not enforced | no (train-only scaler) | validation callbacks then distinct test | **yes: source derivatives are pooled before row split and can cross partitions** | one split, one run |

## Unknown-value search record

- **Rank 13:** inspected `README.md`, `train.py`, `models.py`, and the frozen file list. `README.md:L15-L17` says raw MP4 preprocessing will be uploaded; the committed `au_mfcc.pkl` was treated as data and was **not unpickled/executed**. Consequently upstream normalization and augmentation are unknown.
- **Rank 14:** inspected `README.md`, `train.py`, `utils.py`, `SpeechModel.py`, `infer.py`, and the frozen file list. `README.md:L62-L65` explicitly says only RAVDESS preparation exists. EmoDB/IEMOCAP split, speaker, normalization, selection, augmentation and repetition fields therefore remain unknown; they are not inherited from paper prose.
- **Rank 15:** inspected `README.md`, `run.py`, `plot_history.py`, `model/utils.py`, `model/mainmodel.py`, `model/mainmodel_1582.py`, and the frozen file list. The `CSV_DIR`/`NPY_DIR` declared at `run.py:L38-L41`, their section-name definitions, upstream openSMILE generator and numeric logs are absent. Visible train-only normalization does not close the provenance of those precomputed arrays.
- **Rank 17 result provenance:** inspected both classifier scripts, preprocessing/feature scripts, `results/results.txt`, `results/evaluation_results.json`, and the frozen file list. The 48.91% text result has a matching writer; no frozen script writes the 99.88095% JSON result, so linkage is partial rather than assumed.

## Salient static findings

- Rank 18 is speaker-disjoint at the outer test boundary, but `MinMaxScaler` is fit once on the entire outer-training matrix before eight-speaker GroupKFold model selection. This is a narrowly scoped inner-validation normalization leak, not outer-test leakage.
- Rank 19 preserves actor disjointness, yet creates and saves augmented audio/video files before assigning actors 21–22 to validation and 23–24 to test. The validation and test partitions therefore contain augmented material, violating the preregistered train-only augmentation rule.
- Rank 20 creates original, noise, pitch, and noise+pitch derivatives for each utterance, pools all derivative rows, and only then randomly splits. Derivatives of a single source utterance can occur across train, validation, and test, so the reported 5,954-row test support is not 5,954 independent utterances.
- Rank 16's frozen `data/` and `features.csv` contain 180 rows from only `Actor_01`–`Actor_03`, whereas the README describes the full 24-actor/7,356-file RAVDESS corpus. The approximate 85% statement has no committed metric output tying it to a split.

## Mechanical QA

- CSV physical records: one header + 9 data rows.
- Width: exactly 38 fields for the header and every data row.
- Frozen revisions: all 9 `commit_sha` values are 40 lowercase hexadecimal characters and match the checked-out HEADs; all `commit_date` values came from those HEAD commits.
- Identity/status: all 9 `repository_id` values are blank; all reviewers are `p0_7_reliability`; all statuses are `pending`.
- Enumerations: all top-level risk/repetition values are drawn from values already used by `survey_table.csv`; dataset-specific details remain in the corresponding `*_by_dataset` columns.
- Evidence: every non-unknown risk judgment has a frozen `blob/<40-char-sha>/...#Lx-Ly` permalink. The sole binary-paper result source (`Susanna-Brambilla.pdf`) uses a frozen commit permalink plus page/table locator because GitHub cannot provide text-line anchors for that PDF.
- Prohibited actions: candidate code/notebooks/models were not run; pickles/checkpoints were not loaded; no dependency was installed; no GPU work was started; `p0_5_random_order.csv`, `survey_table.csv`, `REPORT.md`, and `PROGRESS.md` were not modified.

<!-- source-batch: p0_5_audit_root.md -->

# P0.5 随机样本静态审计：root 批次（抽签位 1–11）

- 审计时间：`2026-08-10T18:19:10-04:00`
- 只读静态审计；未运行候选代码、未加载其 pickle/权重、未安装其依赖、未使用 GPU。
- 本批次：抽签位 1–11；纳入 8，E1/E2/E3 各排除 1。主表判定仍待独立全字段复核。

## 纳入

### 抽签位 1: `https://github.com/Vansh-187/speech-emotion-recognition`

- 冻结 commit：`f21f3700413ea8e5c39921fb102c10672cae69a8`（`2025-06-24T19:07:01+05:30`）
- 数据集/划分：`RAVDESS` / `random`；说话人互斥：`not_enforced`。
- 主风险：归一化 `no`；测试选择 `yes_explicit`；增强 `no_train_only`；重复 `single_split_single_seed`。
- 结果：RAVDESS 8-class holdout accuracy=82.61%; F1=82.00%
- 可追溯证据（CSV）：notebooks/emotion_classifier_final.ipynb:L1 cells 6 and 16 parse emotion but do not group actor and call train_test_split；notebooks/emotion_classifier_final.ipynb:L1 cells 13-16 compute per-utterance features before splitting without a fitted cross-sample normalizer；notebooks/emotion_classifier_final.ipynb:L1 cells 25,27,30 build test_loader, save on best test_acc, then reload it；notebooks/emotion_classifier_final.ipynb:L1 cells 16-18 split base features first and append augmentations only for matched training rows。

### 抽签位 2: `https://github.com/fdhlhdc12/ravdess-dtw-emotion-recognition`

- 冻结 commit：`c20123028d7e718e624eb56599ae7436ca9991e4`（`2026-06-27T12:47:50+07:00`）
- 数据集/划分：`RAVDESS` / `unknown`；说话人互斥：`unknown`。
- 主风险：归一化 `unknown`；测试选择 `unknown`；增强 `not_applicable`；重复 `kfold_single_run`。
- 结果：dashboard claims accuracy=90.28%, precision=90.11%, recall=89.56%, F1=90.22%, with 5-fold GridSearchCV
- 可追溯证据（CSV）：app.py:L735-L752 loads serialized models/scaler; no training or split-generation code is present；app.py:L737-L743|L779-L784 loads scaler.pkl and transforms a feature; fit code/provenance is absent；app.py:L1983-L2010 contains static displayed metrics and a GridSearchCV claim but no training/evaluation path；feature_extraction_ml.py:L4-L45 extracts MFCC/delta summaries only; app.py:L770-L784 calls that path。

### 抽签位 4: `https://github.com/Shiva9855/CodeAlpha_Emotion-Recognition-from-Speech`

- 冻结 commit：`cbc84f527fc14b375d34ce2c8aec9b00cabaf459`（`2026-07-20T01:15:32+05:30`）
- 数据集/划分：`RAVDESS|EmoDB` / `random`；说话人互斥：`not_enforced`。
- 主风险：归一化 `no`；测试选择 `yes_explicit`；增强 `yes`；重复 `single_split_single_seed`。
- 结果：no numeric RAVDESS or EmoDB result reported in frozen repository
- 可追溯证据（CSV）：utils/dataset_loader.py:L34-L43 retains only path and emotion for RAVDESS; train.py:L145-L159 stratifies by label only；train.py:L66-L74 fits StandardScaler on X_train and transforms X_test；train.py:L158-L159|L169-L196 uses the sole test_loader each epoch, early-stops and saves on its accuracy；train.py:L47-L63 builds original plus three augmentations for all rows; train.py:L136-L153 then splits the combined matrix。

### 抽签位 5: `https://github.com/dingdongwang/EmotionThinker`

- 冻结 commit：`afcc5c578bc69df436fbb2317957fe4897e33342`（`2026-08-07T13:27:05-07:00`）
- 数据集/划分：`IEMOCAP|RAVDESS` / `unknown`；说话人互斥：`unknown`。
- 主风险：归一化 `unknown`；测试选择 `unknown`；增强 `unknown`；重复 `unknown`。
- 结果：paper Table 2 reports IEMOCAP accuracy=77.68% and RAVDESS accuracy=71.56%
- 可追溯证据（CSV）：scripts/emotionthinker_infer.py:L1-L18 provides model inference only; no target training or evaluation split code exists；scripts/emotionthinker_infer.py:L1-L18 delegates preprocessing to a downloaded pretrained model; target training statistics are absent；README.md:L54-L69 documents inference only; target training/checkpoint-selection code is absent；README.md:L74-L89 describes released annotations/prosody pipeline, not the target benchmark training loaders。

### 抽签位 6: `https://github.com/jobinjoy12/Speech-Emotion-Recognition-using-CREMA-D-dataset`

- 冻结 commit：`10f7e5102b86904ea3b3bb40708238988834ccd0`（`2024-02-22T09:13:09+05:30`）
- 数据集/划分：`CREMA-D` / `random`；说话人互斥：`not_enforced`。
- 主风险：归一化 `no`；测试选择 `no_fixed_training`；增强 `yes`；重复 `single_split_single_seed`。
- 结果：CREMA-D holdout accuracy=52.10%
- 可追溯证据（CSV）：emotion speech recognition code (2).ipynb:L1 cells TmZwQAAia6dU and xQYo6vSXbkTy parse labels and call ordinary train_test_split；emotion speech recognition code (2).ipynb:L1 cell BBnu4lqxboil fits x_train then transforms x_test；emotion speech recognition code (2).ipynb:L1 cells HLTsAspAbqR1 and gHy_VEo2btF3 fit once then predict once；emotion speech recognition code (2).ipynb:L1 cell xyJe_5Q1-xcV injects noise while building all X; cell xQYo6vSXbkTy splits afterward。

### 抽签位 8: `https://github.com/ShaheenPerveen/Speech_Emotion_Recognition_IEMOCAP`

- 冻结 commit：`1f2a835abbf1fbce138b7c26e7e2eb74b35b8147`（`2020-01-01T15:42:44+05:30`）
- 数据集/划分：`IEMOCAP` / `random`；说话人互斥：`not_enforced`。
- 主风险：归一化 `yes`；测试选择 `test_exposed_each_epoch`；增强 `not_applicable`；重复 `single_split_single_seed|kfold_single_run`。
- 结果：representative normalized-MFCC path reports best internal CV score=0.66499 (gradient boosting); multiple holdout reports are retained in notebook outputs
- 可追溯证据（CSV）：Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cell 16 uses train_test_split stratified only by target；Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 14-16 call fit_transform(train) before train_test_split；Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 22-65 tune several models then repeatedly print X_test reports；Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb:L1 cells 1-16 load fixed features, normalize, and split without augmentation。

### 抽签位 9: `https://github.com/sss3799/audio_emotion_recognition`

- 冻结 commit：`d83a682e7b0abe95fa983ce0323bb71010cd2af5`（`2023-04-22T21:40:34+05:30`）
- 数据集/划分：`RAVDESS` / `random`；说话人互斥：`not_enforced`。
- 主风险：归一化 `no`；测试选择 `test_exposed_each_epoch`；增强 `yes`；重复 `single_split_single_seed`。
- 结果：RAVDESS holdout accuracy=60.7433%
- 可追溯证据（CSV）：hsl621_emotion_recognition.ipynb:L1 cells 7 and 40 retain emotion labels but call ordinary train_test_split without actor groups；hsl621_emotion_recognition.ipynb:L1 cell 41；hsl621_emotion_recognition.ipynb:L1 cells 45-46 train with validation_data=(x_test,y_test) then report it as test；hsl621_emotion_recognition.ipynb:L1 cells 33-34 generate three variants for all paths; cell 40 splits afterward。

### 抽签位 10: `https://github.com/neerajk23-cmd/Speech-Emotion-Recognition-on-RAVDESS-Dataset`

- 冻结 commit：`516bc2802eddc65dc382c7d51cd0384410c3e056`（`2026-08-06T22:25:03+05:30`）
- 数据集/划分：`RAVDESS` / `predefined_speaker_independent`；说话人互斥：`enforced`。
- 主风险：归一化 `no`；测试选择 `no_separate_validation`；增强 `not_applicable`；重复 `single_split_single_seed`。
- 结果：held-out actors 20-24: accuracy=62.33% on 300 utterances
- 可追溯证据（CSV）：cnn-final-ee708-project.ipynb:L1 cells 5-6 show 19 train actors and inner random validation; Evaluation_Test_Data.ipynb:L1 cells 3,6,8 show five separate test actor directories；Evaluation_Test_Data.ipynb:L1 cell 3 constructs each test spectrogram independently with fixed parameters；cnn-final-ee708-project.ipynb:L1 cell 6 saves on validation loss; Evaluation_Test_Data.ipynb:L1 cells 8-10 load and evaluate final test once；cnn-final-ee708-project.ipynb:L1 cells 4-6 and Evaluation_Test_Data.ipynb:L1 cell 3 contain fixed mel transforms only。

## 排除与补抽依据

- 抽签位 3 — `E2_target_not_evaluated`：Frozen repository is an explicitly fake/demo-only WebUI: it neither loads real weights nor performs target-data evaluation. 证据：app.py:L2-L13|L24|L67; README.md:L60|L76-L101。
- 抽签位 7 — `E3_no_relevant_code`：Frozen tree contains the 1,440 RAVDESS WAV files but no source code or evaluation implementation. 证据：repository tree at frozen commit: 1,440 .wav files; zero code/notebook/README files。
- 抽签位 11 — `E1_not_SER`：Repository implements facial/image emotion recognition rather than speech emotion recognition. 证据：README.md:L1-L4; Final_Emotion_recognition_cnn.ipynb:L1。

## 冻结永久链接抽查索引

- rank 1 split/test: https://github.com/Vansh-187/speech-emotion-recognition/blob/f21f3700413ea8e5c39921fb102c10672cae69a8/notebooks/emotion_classifier_final.ipynb#L1；results: https://github.com/Vansh-187/speech-emotion-recognition/blob/f21f3700413ea8e5c39921fb102c10672cae69a8/README.md#L50-L56。
- rank 2 inference/scaler: https://github.com/fdhlhdc12/ravdess-dtw-emotion-recognition/blob/c20123028d7e718e624eb56599ae7436ca9991e4/app.py#L735-L784；metrics: https://github.com/fdhlhdc12/ravdess-dtw-emotion-recognition/blob/c20123028d7e718e624eb56599ae7436ca9991e4/app.py#L1983-L2010。
- rank 4 split/selection: https://github.com/Shiva9855/CodeAlpha_Emotion-Recognition-from-Speech/blob/cbc84f527fc14b375d34ce2c8aec9b00cabaf459/train.py#L145-L196；augmentation: https://github.com/Shiva9855/CodeAlpha_Emotion-Recognition-from-Speech/blob/cbc84f527fc14b375d34ce2c8aec9b00cabaf459/train.py#L47-L63。
- rank 5 official relation: https://github.com/dingdongwang/EmotionThinker/blob/afcc5c578bc69df436fbb2317957fe4897e33342/README.md#L1-L18；inference-only path: https://github.com/dingdongwang/EmotionThinker/blob/afcc5c578bc69df436fbb2317957fe4897e33342/scripts/emotionthinker_infer.py#L1-L18。
- rank 6 complete notebook: https://github.com/jobinjoy12/Speech-Emotion-Recognition-using-CREMA-D-dataset/blob/10f7e5102b86904ea3b3bb40708238988834ccd0/emotion%20speech%20recognition%20code%20%282%29.ipynb#L1。
- rank 8 normalized MFCC path: https://github.com/ShaheenPerveen/Speech_Emotion_Recognition_IEMOCAP/blob/1f2a835abbf1fbce138b7c26e7e2eb74b35b8147/Model_Codes/ML_DL_Models_Using_MFCC/ML_with_MFCC_Normalized.ipynb#L1。
- rank 9 complete notebook: https://github.com/sss3799/audio_emotion_recognition/blob/d83a682e7b0abe95fa983ce0323bb71010cd2af5/hsl621_emotion_recognition.ipynb#L1。
- rank 10 train/validation: https://github.com/neerajk23-cmd/Speech-Emotion-Recognition-on-RAVDESS-Dataset/blob/516bc2802eddc65dc382c7d51cd0384410c3e056/cnn-final-ee708-project.ipynb#L1；held-out actor test: https://github.com/neerajk23-cmd/Speech-Emotion-Recognition-on-RAVDESS-Dataset/blob/516bc2802eddc65dc382c7d51cd0384410c3e056/Evaluation_Test_Data.ipynb#L1。

Notebook 仓库的 GitHub 文件在冻结 commit 下为单个 JSON 文本行，故永久链接锚为 `L1`；具体 cell id/index 已保留在 CSV 证据字段。

<!-- P0_5_ADJUDICATIONS_START -->
## P0.5 全量交叉复核与主审裁决

裁决时间：`2026-08-11 00:16:56 UTC-04:00`；合并复核 SHA-256：`7b630043fcded142f174fb0984705e95db436ab4cfeafee8cc7d66ab44c287c6`。
30/30 纳入库与 4/4 顺序排除项均由不同审计遍次复核；37 个字段经主审裁决。
旧值、建议值、最终值、证据与理由完整保存在 `p0_5_review.csv` 和 `p0_5_adjudications.csv`。
这不是两名独立人类编码者；作者盲编码 8–10 库仍为待作者事项。

# SPEAKER-NONEXCLUSIVE EVALUATION IN PUBLIC SER CODE: A PREREGISTERED AUDIT AND CONTROLLED STUDY OF PROTOCOL EFFECTS

[AUTHOR NAME] - [AFFILIATION] - [EMAIL]

## Abstract

Speaker-independent generalization is often intended in speech emotion recognition (SER), but public implementations may use speaker-nonexclusive evaluation. We separate how often this risk appears in public code from how strongly protocol choice changes measured performance. A preregistered probability sample included 30 eligible repositories. Twenty showed speaker-nonexclusive split risk. Five did not, and five were unresolved. The observed rate was 66.7% (Wilson 95% CI [48.8%, 80.8%]). We then ran 1,620 fits on two corpora with four architectures, three protocols, and three seeds. The speaker-paired Random-minus-GroupKFold UAR difference ranged from 15.39 percentage points (FNO) to 19.68 (ResNet-SE) on RAVDESS, and from 1.82 (FNO) to 3.06 (Transformer) on CREMA-D. All eight contrasts survived Holm correction. The results characterize a sampled practice and two tested corpora, not a universal corpus law.

**Index Terms-** speech emotion recognition, evaluation protocol, speaker independence, reproducibility, empirical audit

## 1. INTRODUCTION

Speech emotion recognition systems are often interpreted as measuring generalization to unseen speakers. That interpretation requires training and evaluation partitions to be speaker-exclusive. If one speaker can occur on both sides, the estimand instead concerns new utterances from speakers represented during training. Such a protocol can be legitimate for a speaker-dependent application, but it does not alone support an unseen-speaker claim.

Prior studies show that SER performance depends on partition construction [1]-[5], while benchmark efforts increasingly distribute fixed speaker-independent splits [7], [8]. The closest prior study, from 2026, already contrasts random and speaker-independent evaluation on RAVDESS and CREMA-D and reports a larger change on RAVDESS [5]. Therefore, neither the existence of a split effect nor its occurrence on these corpora is claimed as a first discovery here.

We connect two complementary measurements. First, we estimate how often speaker-nonexclusive split risk appears in a preregistered probability sample from a frozen public-code candidate frame. Evidence is tied to executable split behavior or directly traceable code. Second, we quantify protocol sensitivity under one implementation: two corpora, four fixed architectures, three evaluation protocols, three seeds, complete out-of-fold prediction, and speaker-paired inference.

We contribute a preregistered audit of 30 eligible public SER repositories. We add a controlled study with 1,620 valid fits. Finally, speaker-cluster intervals and family-wise multiplicity control show consistently positive, but sharply different, Random-GroupKFold UAR effects in the two tested corpora.

We use speaker-nonexclusive split risk descriptively. It does not imply that a model demonstrably exploited identity, that every reported result is invalid, or that every repository intended unseen-speaker generalization.

## 2. RELATION TO PRIOR WORK

SER evaluation has long distinguished speaker-dependent from speaker-independent settings [1]. Direct comparisons later showed that fold criteria alter absolute performance and sometimes model rankings [2]-[5]. Reproduction work has also found that public SER systems can be hard to reconstruct under one common protocol [6]. Antoniou et al. [6] audit a purposive IEMOCAP case study; our sampling estimand is instead a frozen public-code candidate frame.

SERAB [7] and Open-Emotion/EMO-SUPERB [8] reduce ambiguity by publishing standardized speaker-independent partitions. They address benchmark construction rather than the prevalence of observable split behavior in sampled implementations. Meyer et al. [9] provide a related warning that dataset splits can expose non-emotion shortcuts. Our study is narrower: speaker exclusivity, code-practice prevalence, and paired protocol effects.

## 3. METHODS

### 3.1. Public-code audit

The repository candidate frame, probability sample, and classification procedure were frozen before outcome analysis. The unit was a unique repository. Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions.

A repository was positive when the evaluated path did not enforce speaker exclusivity or produced speaker overlap on the frozen synthetic manifest; negative when exclusivity was verified; and unknown when available evidence could not resolve the property. Twenty-four repositories had highest-tier executable evidence, one had traceable static evidence, and five remained unknown. Highest-tier fragments passed 384 finite property cases. Each repository contributed 16 cases. We executed only the decisive frozen split fragment, not the entire repository.

The audit denominator was 30 repositories. We report the positive fraction with a two-sided Wilson interval. Unresolved repositories stay unresolved; a separate sensitivity bound treats all five as positive. The target is the frozen public-code candidate frame, not all SER papers or software.

### 3.2. Corpora and preprocessing

RAVDESS contributed 1,440 utterances from 24 speakers over eight classes [10]; CREMA-D contributed 7,442 utterances from 91 speakers over six classes [11]. Audio was converted to 22.05 kHz mono and trimmed at 30 dB. Inputs were 64-bin log-mel features (FFT 1,024; hop 512; upper frequency 11,025 Hz; 128 frames) with per-utterance standardization. SpecAugment was restricted to training data.

### 3.3. Models, protocols, and inference

Four fixed probes were used: CNN, ResNet-SE, Transformer, and Fourier neural operator. The architecture and optimization contracts were fixed across protocols. AdamW used batch size 64, at most 100 epochs, early stopping patience 15, and seeds 0, 1, and 2. No test partition guided model selection. These probes do not exhaust modern SER model space; their role is to hold implementation constant while changing protocol.

Random used five-fold StratifiedKFold (seed 42) without a speaker grouping constraint. Grouped used five-fold GroupKFold by speaker. LOSO held out one speaker at a time (24 RAVDESS folds; 91 CREMA-D folds). Inner validation mirrored each protocol: ungrouped under Random, speaker-grouped under the strict protocols. Thus Random-Grouped changes both outer speaker overlap and inner-validation grouping; it is a pipeline-level protocol contrast, not a pure causal estimate of outer-test overlap.

Full out-of-fold predictions were retained for every corpus-model-protocol-seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for RAVDESS; 1,212 for CREMA-D). UAR was primary. The principal contrast was Random minus GroupKFold; Random-LOSO and Grouped-LOSO were secondary members of the same frozen family.

Speaker was the paired unit. Each speaker was first averaged over three seeds. Confidence intervals used 10,000 whole-speaker bootstrap replicates. Two-sided Wilcoxon tests were adjusted by Holm. The multiplicity family size was 24 preregistered UAR tests. Accuracy and macro-F1 were descriptive only.

All Random folds placed every corpus speaker on both sides of the outer split, whereas Grouped and LOSO outer overlap was zero. This verifies the protocol property; it does not by itself show that a trained classifier used speaker identity.

## 4. RESULTS

### 4.1. Audit prevalence

The sample contained 30 repositories. Twenty showed speaker-nonexclusive split risk. Five were verified negative, and five were unresolved. The observed positive fraction was 66.7% (Wilson 95% CI [48.8%, 80.8%]). Under an unknown-as-positive sensitivity analysis, the positive count was 25. The denominator remained 30. The sensitivity rate was 83.3% (Wilson 95% CI [66.4%, 92.7%]). The primary estimate retains unknown cases in the denominator rather than silently converting them to negatives.

![Protocol effects](figure_protocol_effects.png)

*Fig. 1. (A) Frozen audit outcomes for the preregistered probability sample; the estimand is the frozen public-code candidate frame, not all SER work. The interval is the Wilson 95% CI for 20/30. (B) Speaker-paired Random-minus-GroupKFold UAR differences with 10,000-replicate speaker-cluster intervals. These eight contrasts belong to the original frozen 24-test Holm family; all eight survived correction, and the corpus contrast is descriptive for these pipelines.*

### 4.2. Controlled protocol effects

Random five-fold UAR exceeded GroupKFold UAR in all eight combinations. The speaker-paired difference ranged from 15.39 percentage points (FNO) to 19.68 (ResNet-SE) on RAVDESS, and from 1.82 (FNO) to 3.06 (Transformer) on CREMA-D; the remaining models lay between these endpoints. All eight Random-Grouped tests survived correction. Across all three protocol contrasts, the Holm rejections numbered 19. The multiplicity family size was 24 tests.

| Corpus | Model | Random UAR | GroupKFold UAR | LOSO UAR | R-G delta [95% CI] | Holm p |
|---|---|---:|---:|---:|---:|---:|
| RAVDESS speech | CNN | 56.86 ± 1.17 | 39.15 ± 0.52 | 45.79 ± 1.81 | 17.71 [15.04, 20.55] | 0.000309 |
| RAVDESS speech | ResNet-SE | 66.04 ± 1.53 | 46.35 ± 2.50 | 50.76 ± 2.51 | 19.68 [16.62, 22.94] | 0.000309 |
| RAVDESS speech | Transformer | 57.36 ± 1.73 | 41.02 ± 1.33 | 42.53 ± 2.40 | 16.34 [12.59, 20.33] | 0.000322 |
| RAVDESS speech | FNO | 55.79 ± 1.24 | 40.41 ± 2.13 | 42.51 ± 0.69 | 15.39 [12.28, 18.82] | 0.000322 |
| CREMA-D | CNN | 58.63 ± 0.59 | 55.84 ± 0.84 | 57.59 ± 0.31 | 2.80 [2.07, 3.53] | 4.05e-08 |
| CREMA-D | ResNet-SE | 61.24 ± 0.64 | 58.79 ± 0.79 | 60.34 ± 0.43 | 2.46 [1.65, 3.27] | 6.99e-06 |
| CREMA-D | Transformer | 52.01 ± 0.58 | 48.95 ± 0.31 | 51.54 ± 0.40 | 3.06 [2.29, 3.82] | 1.09e-08 |
| CREMA-D | FNO | 53.37 ± 0.48 | 51.55 ± 0.54 | 53.86 ± 0.11 | 1.82 [1.09, 2.51] | 0.000132 |

*Table 1. UAR (%) mean ± sample SD over three seeds. Delta is the speaker-paired Random-Grouped difference with 95% cluster-bootstrap CI.*

LOSO was not uniformly below both five-fold protocols; for example, CREMA-D FNO had slightly higher mean UAR under LOSO than Random. Protocols are therefore not a simple ordinal scale of strictness. Grouped is the most directly matched outer-fold comparison to Random; LOSO also changes fold count, test composition, and training size.

The nonoverlapping effect ranges show sharply different protocol sensitivity between the two tested corpora. This does not identify a cause. Speaker count, recording design, acted content, and within-speaker structure are candidate moderators, but none is isolated by a two-corpus design.

On a matched 23-actor RAVDESS speech-song panel (six shared emotions, 1,012 items per channel, 792 fits), the Random-minus-GroupKFold protocol-package difference was positive in both channels for all four models (speech +11.87 to +16.12, song +9.45 to +11.59 UAR points). The song-minus-speech difference of these premiums was negative for all four models as a point estimate; it was directionally stable in this finite panel for ResNet-SE, Transformer, and FNO under seed, leave-one-actor-out, and actor-reweighting diagnostics (stability descriptions, not confidence intervals), and not conclusive for CNN; the four secondary song-minus-speech differences of the Random-minus-LOSO premiums were also not conclusive.

## 5. DISCUSSION

The audit and experiment answer different questions. The audit estimates how frequently a verifiable split risk appears in one sampled public-code frame. The experiment estimates how much one frozen pipeline changes when the evaluation protocol changes. Neither substitutes for the other: a common practice need not have a large effect in every corpus, and a large controlled effect does not establish prevalence.

The RAVDESS-CREMA-D pattern is directionally consistent with the closest exact-corpus prior study [5]. We consequently frame this result as a replication and extension, not discovery. The added value is the linked design: probability sampling for practice prevalence, executable repository evidence, a common four-model implementation, complete OOF prediction, speaker-paired uncertainty, and preregistered multiplicity control.

The operational recommendation is narrow. If a reported SER score is intended to describe unseen speakers, the evaluation should enforce speaker exclusivity and publish grouping information sufficient to reproduce that property. Random utterance splitting may still answer a legitimate speaker-dependent question, but the target population should be explicit.

No universal correction should be subtracted from published scores. The reported protocol-effect ranges arise from specific corpora, models, preprocessing, training recipe, and split constructions. Multiplying the audit prevalence by these effects would combine incompatible estimands and is intentionally excluded.

### 5.1. Limitations

The audit covers one GitHub/arXiv-dominated candidate frame and 30 sampled repositories, with five unresolved. Search availability and eligibility limit extrapolation. A positive label describes the inspected split mechanism or its behavior on a frozen synthetic manifest; it does not prove the exact speaker overlap in an associated paper's reported run.

Speaker-nonexclusive evaluation makes identity-related information available across partitions, but does not show that a classifier used identity or that score differences are caused solely by identity. The controlled experiment uses two acted English corpora, four non-SSL probes, three optimization seeds, and one frozen outer partition per protocol. Its confidence intervals do not cover alternative split draws or other corpora.

Hyperparameters were fixed across protocols, so we estimate a protocol difference under one training recipe rather than each protocol's independently optimized best score. Random-Grouped also changes inner validation grouping. The speaker-cluster intervals quantify uncertainty within the observed actors, not a superpopulation of recording designs.

## 6. CONCLUSION

A preregistered probability audit found speaker-nonexclusive split risk in 20 sampled public SER repositories. The sample contained 30 repositories, with five verified negatives and five unresolved cases. In a controlled 1,620-fit study, Random five-fold evaluation produced higher UAR than GroupKFold for all four probes on both corpora, but the premium was much larger on RAVDESS than CREMA-D. These findings support explicit speaker grouping when unseen-speaker generalization is the target. They characterize a sampled coding practice and specified conditions, not all SER systems.

## 7. REPRODUCIBILITY STATEMENT

Every number, table, and figure in this manuscript was generated directly from frozen audit and experiment tables. The build fails closed unless the final adjudication, successful-fit total, complete OOF cells, verification checks, and eight RG-UAR rows match their registered sources. Repository-level audit evidence, split manifests, speaker-level predictions, and independent statistic verification are retained in the project artifact bundle. A release URL will be inserted after anonymization and archival packaging are finalized.

Submission note. ICASSP 2027 uses single-anonymous review. The placeholder author and affiliation above must be replaced, and the official 2027 template must supersede this provisional layout before submission.

## References

[1] B. W. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009 Emotion Challenge," in Proc. Interspeech, pp. 312-315, 2009.

[2] L. Pepino, P. Riera, L. Ferrer, and A. Gravano, "Fusion approaches for emotion recognition from speech using acoustic and text-based features," in Proc. ICASSP, pp. 6484-6488, 2020.

[3] B. T. Atmaja and A. Sasou, "Effect of different splitting criteria on the performance of speech emotion recognition," in Proc. TENCON, pp. 760-764, 2021.

[4] M. Zielonka et al., "Recognition of emotions in speech using convolutional neural networks on different datasets," Electronics, vol. 11, no. 22, Art. 3831, 2022.

[5] E. Ibrahim, M. E. Ghoraba, and A. E. Ghoraba, "Multimodal emotion recognition using hybrid deep feature fusion under speaker-independent evaluation," Scientific Reports, vol. 16, Art. 19584, 2026.

[6] N. Antoniou, A. Katsamanis, T. Giannakopoulos, and S. Narayanan, "Designing and evaluating speech emotion recognition systems: A reality check case study with IEMOCAP," in Proc. ICASSP, pp. 1-5, 2023.

[7] N. Scheidwasser-Clow, M. Kegler, P. Beckmann, and M. Cernak, "SERAB: A multi-lingual benchmark for speech emotion recognition," in Proc. ICASSP, pp. 7697-7701, 2022.

[8] H. Wu et al., "Open-Emotion: A reproducible EMO-SUPERB for speech emotion recognition systems," in Proc. IEEE SLT, pp. 510-517, 2024.

[9] P. Meyer, E. Buschermohle, and T. Fingscheidt, "What do classifiers actually learn? A case study on emotion recognition datasets," in Proc. Interspeech, pp. 262-266, 2018.

[10] S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)," PLOS ONE, vol. 13, no. 5, Art. e0196391, 2018.

[11] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, "CREMA-D: Crowd-sourced emotional multimodal actors dataset," IEEE Trans. Affect. Comput., vol. 5, no. 4, pp. 377-390, 2014.

> **历史状态（2026-09-07补记）：**下文保留2026-08-15的公共代码审计设想及后续原始批注，不再作为当前论文贡献列表。当前定位请读[PRIOR_ART_POSITIONING_20260907.md](PRIOR_ART_POSITIONING_20260907.md)。旧文中关于Ibrahim纯音频分支及两库落差的正确事实、2026-08-17的书目更正均保留；本状态说明不撤回这些事实。概率抽样代码审计及代码实践流行度不属于当前稿贡献，新程序仍待正式结果。

# Prior-art collision matrix for the ICASSP 2027 demo

Search date: 2026-08-15. This matrix records the closest primary studies found before drafting. It is not a systematic review. Risk is assessed against the proposed paper, not against the underlying experiments.

## Bottom line

The broad claim that random, speaker-nonexclusive splits can inflate SER performance is already established. A June 2026 *Scientific Reports* article also reports a much larger split effect on RAVDESS than on CREMA-D, so neither the split effect nor two-corpus heterogeneity is defensible as a first discovery.

The remaining defensible contribution is narrower:

1. a preregistered probability sample from a frozen public-code candidate frame;
2. file/line and executable-fragment evidence for split behavior;
3. a same-implementation 2-corpus x 4-model x 3-protocol x 3-seed controlled study;
4. speaker-paired inference with cluster bootstrap and a frozen Holm family.

## Risk-ranked overlaps

| Risk | Work | Direct overlap | What remains distinct in this demo |
|---|---|---|---|
| **Critical** | Ibrahim, Ghoraba, and Ghoraba, "Multimodal emotion recognition using hybrid deep feature fusion under speaker-independent evaluation," *Scientific Reports*, 2026. [DOI](https://doi.org/10.1038/s41598-026-58836-w) / [PubMed](https://pubmed.ncbi.nlm.nih.gov/42342860/) | Uses RAVDESS and CREMA-D and contrasts random with speaker-independent evaluation. Its audio branch already shows a much larger degradation on RAVDESS than CREMA-D. | No probability-sampled code-practice prevalence estimate; no four-architecture paired protocol study with speaker-cluster inference. |
| **Critical / high** | Zielonka et al., "Recognition of Emotions in Speech Using Convolutional Neural Networks on Different Datasets," *Electronics*, 2022. [Article](https://www.mdpi.com/2079-9292/11/22/3831) / [DOI](https://doi.org/10.3390/electronics11223831) | Includes RAVDESS and CREMA-D and explicitly contrasts random splits with actor-disjoint preparation for CNN/ResNet systems. | The demo adds a frozen code audit, GroupKFold versus LOSO, multiple seeds, and speaker-paired uncertainty. |
| **High** | Pepino et al., "Fusion Approaches for Emotion Recognition from Speech Using Acoustic and Text-Based Features," ICASSP 2020. [DOI](https://doi.org/10.1109/ICASSP40776.2020.9054709) / [full text](https://arxiv.org/abs/2403.18635) | Shows that fold construction can change absolute SER performance and model conclusions; compares random, speaker, and speaker-plus-script folds. | Different corpora and multimodal focus; no public-code probability audit or prevalence interval. |
| **High** | Atmaja and Sasou, "Effect of Different Splitting Criteria on the Performance of Speech Emotion Recognition," TENCON 2021. [DOI](https://doi.org/10.1109/TENCON54134.2021.9707265) / [full text](https://arxiv.org/abs/2210.14501) | Directly studies speaker-dependent, speaker-independent, text-independent, and joint criteria over repeated trials. | Single JTES corpus and MLP pipeline; no code-practice sampling or same-implementation multi-architecture effect estimation. |
| **High** | Antoniou et al., "Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check Case Study with IEMOCAP," ICASSP 2023. [full text](https://arxiv.org/abs/2304.00860) | Reviews SER evaluation assumptions and attempts to reproduce public implementations under a common speaker-independent protocol. | Purposive IEMOCAP case study rather than a frozen-frame probability sample; does not estimate prevalence with a confidence interval. |
| **High / medium** | Wagner et al., "Open-Emotion: A Reproducible EMO-SUPERB for Speech Emotion Recognition Systems," SLT 2024. [DOI](https://doi.org/10.1109/SLT61566.2024.10832296) / [extended paper](https://arxiv.org/abs/2402.13018) | Standardizes speaker-independent SER partitions and reproducible benchmarking over several corpora and SSL models. | Benchmark construction rather than an audit of current public-code practice or paired random-versus-grouped protocol effects. |
| **Medium** | Arora et al., "SERAB: A Multi-Lingual Benchmark for Speech Emotion Recognition," ICASSP 2022. [DOI](https://doi.org/10.1109/ICASSP43922.2022.9747348) / [full text](https://arxiv.org/abs/2110.03414) | Includes RAVDESS and CREMA-D under fixed speaker-independent partitions. | No random-split control, code audit, or protocol-effect inference. |
| **Medium** | Schuller et al., "What Do Classifiers Actually Learn? A Case Study on Emotion Recognition Datasets," Interspeech 2018. [DOI](https://doi.org/10.21437/Interspeech.2018-1851) | Establishes that a dataset split can expose a strong non-emotion shortcut and radically change performance. | Focuses on lexical/content shortcuts rather than speaker exclusivity and public-code practice. |
| **Medium / low** | Atassi and Esposito, "Comparison of Speaker Dependent and Speaker Independent Emotion Recognition," 2013. [DOI](https://doi.org/10.2478/amcs-2013-0060) | Direct historical precedent for speaker-dependent versus speaker-independent SER. | Does not overlap the audit design or current controlled benchmark scale. |
| **Low; mechanism citation** | Champion et al., "Utility-Preserving Privacy-Enabled Speech Embeddings for Emotion Detection," Interspeech 2023. [DOI](https://doi.org/10.21437/Interspeech.2023-1075) | Shows that speaker identity can remain recoverable from emotion-oriented speech representations on RAVDESS/CREMA-D. | Studies privacy and representation sanitization, not evaluation protocols. |

## Claims explicitly removed from the demo

- "We are the first to show speaker leakage inflates SER."
- "RAVDESS is uniquely vulnerable" or any universal corpus law from two corpora.
- "66.7% of SER papers leak speakers." The denominator is a frozen public-code candidate frame, not the literature.
- A universal correction factor or the historical P2 quantity `Theta = pi * Delta`.
- A causal claim that all of the Random-GroupKFold difference comes from outer-test speaker overlap; inner-validation grouping changes at the same time.

## Recommended positioning

**Working title:** *Speaker-Nonexclusive Evaluation in Public SER Code: A Preregistered Audit and Controlled Study of Protocol Effects*

**One-sentence contribution:** A probability-sampled code audit estimates how often speaker exclusivity is not enforced within a frozen public-code frame, while a controlled paired benchmark quantifies how sharply protocol sensitivity differs between two tested corpora without proposing a universal correction.


---

## 更正批注（2026-08-17 16:00 UTC-04:00，Claude；追加式，不改上表原文）

写作合规审计中发现上表两处**署名错误**（已联网核实原始文献页）：

1. 表中 "Wagner et al., Open-Emotion: A Reproducible EMO-SUPERB..., SLT 2024"
   ——实际作者为 **H. Wu, H.-C. Chou, K.-W. Chang, L. Goncalves, J. Du,
   J.-S. R. Jang, C.-C. Lee, H.-Y. Lee**（IEEE Xplore 10832296 与
   arXiv:2402.13018 一致）。无 Wagner。
2. 表中 "Arora et al., SERAB: A Multi-Lingual Benchmark..., ICASSP 2022"
   ——实际作者为 **N. Scheidwasser-Clow, M. Kegler, P. Beckmann, M. Cernak**
   （arXiv:2110.03414 官方页）。无 Arora。

**影响评估**：论文 `paper/main.tex` 参考文献两条署名本来就是对的
（wu2024openemotion、scheidwasserclow2022serab），六篇必引文献全部在文中被
显式引用并正面比较——作者 2026-08-15 10:52 裁决的**实质要求已满足**；
裁决原文与本表沿用的 "Wagner 2024 / Arora 2022" 两个人名系本表当初的
元数据错误，随裁决转写扩散。DOI 与工作本身的识别自始正确，
不涉及任何科研结论。作者如对此更正有异议可否决。

Completed Study II, updated 2026-09-05. All 1,440 planned units are closed, and the complete scientific results have passed independent recomputation from the original predictions. This section reports every specified endpoint and contrast.

# Study II: Speaker allocation and within-speaker sentence coverage under fixed recording budgets

**Manuscript section for integration.** The study is complete. The methods describe the locked design; the results and discussion below use the independently verified full analysis.

## Motivation and relation to previous work

A practical data-collection question concerns how to allocate a limited number of labelled recordings across speakers and sentences. For example, with six emotion classes and eight available training sentences, a budget of 288 recordings can support either 12 speakers producing four sentences per emotion or 48 speakers producing one sentence per emotion. Both arrangements can expose each training sentence through six speakers. The decision therefore concerns a tradeoff between the number of training speakers and sentence coverage within each speaker. A positive effect of either allocation is an empirical question.

Protocol sensitivity is an established motivation for this question. Kumari et al. compare random and actor-wise evaluation on CREMA-D using classical, convolutional and hybrid systems; their prominently reported random/actor-wise comparison is expressed in **Macro-F1**, which must not be substituted for the UAR endpoint used here. Their work provides a close precedent for explicit protocol comparison. [Kumari et al., 2026](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0355238)

Fixed-budget data composition also has precedents in related speech tasks. Vaessen and van Leeuwen restrict training subsets to 50,000 recordings while varying speaker and session diversity. Their task is speaker recognition, so the study motivates a controlled allocation experiment without establishing the direction of an effect for emotion recognition. [Vaessen and van Leeuwen, 2022](https://www.isca-archive.org/interspeech_2022/vaessen22_interspeech.html)

Sentence content requires separate attention. Pešán et al. examine lexical overlap and text dependency in paralinguistic datasets including CLSE and IEMOCAP. This motivates explicit sentence conditions, while leaving the present fixed-budget allocation comparison to be evaluated. [Pešán et al., 2024](https://arxiv.org/abs/2403.07767)

Study II compares two specified allocations on speakers unseen during supervised training, under paired seen- and unseen-sentence conditions. Its contribution is a controlled estimate of an allocation tradeoff under a reproducible, finite-corpus design. The study evaluates fixed policies without introducing a new classifier or a configuration-search algorithm.

## Methods

### Design status and prior exposure

Study II was specified after the researchers had viewed earlier v2 results and an external review of the proposed extension. CREMA-D had already been used in the earlier work. This is a prospectively specified estimation study informed by prior analyses. It does not inherit the confirmatory or preregistration status of v2 or N14R, and the corpus is not a newly collected, previously untouched population.

The core unit lists, model settings, source identities, data identities and statistical specification were locked before core execution. Publication of these materials provides an auditable version history; it is not an independent preregistration review. The separate engineering pilot was used to check inputs, execution and timing without computing or inspecting its selection-set performance or rankings. Study II scientific outcomes were not inspected until all 1,440 required units were closed; manuscript interpretation began after independent result verification.

### Corpus preparation and representative recordings

The pinned CREMA-D manifest contains 7,442 recordings. The inherited hygiene procedure produces 7,435 recordings, after which this study selects one representative recording per speaker × sentence × emotion cell. The rule gives priority to `MD`, followed by `XX`; a cell with neither is unavailable. Ties within the same priority are resolved by the lexicographically first relative path. No `HI` or `LO` substitute is used to complete a cell, and repeated files are not used to meet a recording budget.

The metadata audit identifies **6,524 representative recordings: 455 MD and 6,069 XX**. These are metadata and capacity findings, not recognition results. The MD/XX rule avoids systematic selection of high-intensity IEO recordings through file ordering. It does not establish that IEO and other sentences have identical emotional intensity, and unspecified intensity is not treated as equivalent to a controlled medium-intensity condition.

The analysis population contains 91 speakers and 12 sentences, with six fixed emotion classes. After combining the six test-sentence pairs, individual speaker × class cells contain **11–12 available representative sentences**. Twelve is the intended full support; unavailable recordings remain absent. No class is dropped from a speaker's UAR calculation, and no missing recording is duplicated or imputed. The missing-cell inventory accompanies the analysis.

### Fixed recording budgets and allocations

Let $B$ denote the number of distinct final-training recordings and $S$ the number of training speakers. Every selected speaker–sentence edge contains one recording from each of the six emotion classes. The global number of training sentences is fixed at eight.

| Training recordings $B$ | Training speakers $S$ | Sentences per speaker | Speakers per training sentence | Recordings per speaker–sentence–class cell |
|---:|---:|---:|---:|---:|
| 288 | 12 | 4 | 6 | 1 |
| 288 | 48 | 1 | 6 | 1 |
| 576 | 12 | 8 | 12 | 1 |
| 576 | 48 | 2 | 12 | 1 |

Thus $B=6S P_{\mathrm{per\ speaker}}$, and each sentence is represented by $B/(8\times6)$ speakers. Within a budget, the allocations have the same total recordings, global sentence count, class quotas and number of speakers contributing to each sentence. Increasing $S$ necessarily decreases within-speaker sentence coverage in this design; the estimand is this joint allocation change.

Both allocations have equal numbers in the corpus-provided Female and Male metadata strata. The same balance holds among the speakers contributing to every training sentence. These strata are taken from the official demographic table, rather than inferred from audio. They address one potential composition imbalance without controlling every demographic or acoustic attribute.

### Speaker roles and paired sentence conditions

Each of three draws specifies a complete five-fold outer speaker partition, stratified using the demographic metadata. Every speaker belongs to exactly one outer test fold within a draw. From the speakers outside that fold, eight speakers—four from each metadata sex stratum—form the early-stopping population. The remaining speakers form the training candidate pool. Test, early-stopping and training speakers are mutually exclusive.

Ridge uses the same candidate pool as CNN even though Ridge does not use early stopping. There is no additional configuration-selection set and no validation-driven choice of speaker allocation. Training eligibility requires complete class support across the union of the paired training-sentence sets. The smaller 12-speaker set is nested within the corresponding 48-speaker set, as verified in the frozen design.

The 12 sentences are partitioned into six disjoint test pairs. For each pair, the design fixes six common sentences and two replacement sentences from the other ten sentences. The conditions are:

- **Seen sentence:** the training-sentence set comprises the six common sentences and the two test sentences.
- **Unseen sentence:** the training-sentence set comprises the same six common sentences and the two replacement sentences.

Each condition therefore uses eight training sentences. The same speakers and speaker–sentence allocation graph are retained within a matched pair; the two test-sentence identities are replaced at corresponding graph positions. This preserves each speaker's degree, each sentence's exposure and the class blocks. Test sentences never appear in training or early-stopping recordings under the unseen-sentence condition.

For a given draw, outer fold and sentence pair, the test paths are identical across models, budgets, speaker allocations and sentence conditions. The early-stopping set uses the eight designated speakers and the six common sentences. Its available size is **279–288 recordings**, with the exact paths fixed in the plan and shared across all matched conditions. The ideal 288-recording Cartesian product is not claimed when cells are unavailable.

The complete design comprised 2 models × 2 budgets × 2 speaker allocations × 2 sentence conditions × 5 speaker folds × 6 sentence pairs × 3 draws, or **1,440 fit units**, with **720 per model**. Six sentence-pair predictions are combined for each test speaker; they are not treated as six independent test populations.

### Models and execution rules

The first model is a Ridge classifier on the existing time-mean WavLM-base+ feature cache, using fixed hidden state 12. StandardScaler is fitted only on the current training subset. Ridge uses `alpha=1.0`, balanced class weights, `solver=lsqr` and `tol=0.0001`, and runs on CPU.

The second model is the fixed log-mel CNN inherited from the pinned `p1_frozen` implementation, using architecture candidate index 2. Its specified settings are a maximum of 100 epochs, early-stopping patience 15, batch size 32, learning rate 0.001, weight decay 0.0001, dropout 0.1, AdamW and a cosine schedule. Class weights are calculated from training data. The checkpoint is selected by minimum early-stopping validation loss. Training augmentation follows the pinned implementation; evaluation uses no augmentation. Architecture and preprocessing details are supplied through the source versions rather than selected using Study II outcomes.

Actual epochs, timing and execution environments are recorded. The common epoch limit does not imply equal optimizer-update counts across recording budgets. The implementation does not extend training for a particular configuration in response to its test result.

Each planned unit permits at most two fitting attempts with the same configuration, seed and data. An interrupted attempt with a start event but no successful completion consumes an attempt. Successful units are resumed only after identity and hash verification. Failures are neither assigned a zero score nor silently omitted; exhausted attempts leave the core incomplete. An execution ledger, per-unit receipts, prediction hashes and a complete analysis lock provide the audit trail. Timing-only pilot fits and failed attempts are accounted for separately from the 1,440 specified units.

### Estimands and aggregation order

For speaker $i$, policy $q=(\mathrm{model},B,S,\mathrm{condition})$ and draw $d$, predictions from the six sentence-pair models are pooled. If $n_{ic}$ is the number of available recordings of class $c$ for that speaker and $a_{icqd}$ the number classified correctly, the speaker-level score in percentage points is

$$
u_i(q,d)=\frac{100}{6}\sum_{c=0}^{5}\frac{a_{icqd}}{n_{ic}}.
$$

All six class denominators must be positive. This calculation precedes averaging over draws:

$$
\bar u_i(q)=\frac{1}{3}\sum_{d=0}^{2}u_i(q,d),\qquad
\mu(q)=\frac{1}{91}\sum_{i=1}^{91}\bar u_i(q).
$$

The overall endpoint gives equal weight to speakers and, within each speaker, to the six classes. It is not utterance accuracy, and equal weighting of sentence-pair UAR values cannot replace pooling their class counts when recordings are missing.

For each model and budget, the primary contrast is

$$
\Delta_{\mathrm{new}}=\frac{1}{91}\sum_i
\left[\bar u_i(S=48,\mathrm{new})-\bar u_i(S=12,\mathrm{new})\right].
$$

There are four primary estimates: two budgets for each of two models. Corresponding seen-sentence contrasts, $\Delta_{\mathrm{seen}}$, are reported alongside the four differences in differences, $\Delta_{\mathrm{new}}-\Delta_{\mathrm{seen}}$. A positive difference in differences denotes a larger allocation contrast under unseen sentences; it does not by itself establish an improvement in either condition's absolute performance. All 16 absolute UAR estimates and all 12 specified contrasts are retained.

### Conditional uncertainty

The bootstrap resamples 91 speaker IDs with replacement, retaining each selected speaker's complete paired vector. Speaker IDs are sorted as strings. NumPy `default_rng(20260905)` generates one `10000 × 91` resampling matrix, shared across every absolute and contrast vector. The 2.5th and 97.5th percentiles use linear quantile interpolation.

The resulting intervals describe approximate test-speaker uncertainty conditional on the executed training subsets, fitted models, sentence arrangement and training procedure. They do not integrate retraining under resampled datasets or uncertainty over new corpora. Cross-validation models share training speakers, and the three draws reuse the same finite corpus. The inferential sample size remains 91 speakers; folds, sentence pairs, draws and fit units do not enlarge it. Each complete draw's overall estimate and the three-draw range are also reported as descriptive sensitivity information.

Intervals are pointwise, not simultaneous across models, budgets and conditions. No confirmatory p-values, Holm decisions or family-level significance claims are specified. An interval spanning zero is not evidence of equivalence. The three-percentage-point reference in the planning documents is an effect-size discussion scale, not demonstrated detection power or an empirically validated acceptance threshold.

## Results

### Complete execution and independent verification

All 1,440 specified units completed: 720 Ridge–WavLM and 720 CNN units. The final ledger contains 1,440 starts and 1,440 successful completions, with no failed attempts; every receipt records attempt 1. The CNN run began with eight timing-only monitored core units, then resumed the same environment and plan for the remaining 712. Those eight units are part of the 720 CNN units and are counted once. The separate nine-fit engineering pilot is excluded from these core counts.

The two model blocks were merged into a new offline run directory, retaining their original receipts, prediction bytes, model environments and source ledgers. Complete closure and the analysis lock were verified before scoring. An independent implementation then reconstructed speaker-level quantities and intervals from the original predictions. It passed all 6,199 numeric-field comparisons, with a maximum discrepancy of 2.13 × 10⁻¹⁴ percentage points. This establishes agreement with the specified computation, while the inferential limitations remain those described above.

### Absolute speaker-weighted UAR

Table 1 reports all 16 absolute endpoints. Values are UAR percentages with pointwise 95% conditional percentile intervals; all estimates average the same 91 test speakers after pooling sentence pairs and averaging draws.

| Model | Recording budget | Sentence condition | S12 UAR [95% interval] | S48 UAR [95% interval] |
|---|---:|---|---|---|
| Ridge–WavLM | 288 | Seen | 44.74 [43.45, 45.99] | 44.54 [43.20, 45.86] |
| Ridge–WavLM | 288 | Unseen | 28.54 [27.75, 29.30] | 29.27 [28.57, 30.00] |
| Ridge–WavLM | 576 | Seen | 46.96 [45.51, 48.41] | 46.50 [45.12, 47.84] |
| Ridge–WavLM | 576 | Unseen | 28.33 [27.67, 28.99] | 28.39 [27.69, 29.08] |
| CNN | 288 | Seen | 41.27 [39.73, 42.83] | 42.57 [40.85, 44.29] |
| CNN | 288 | Unseen | 40.24 [38.69, 41.83] | 41.32 [39.66, 42.96] |
| CNN | 576 | Seen | 43.14 [41.48, 44.80] | 45.29 [43.55, 47.07] |
| CNN | 576 | Unseen | 41.06 [39.42, 42.67] | 43.22 [41.55, 44.87] |

The absolute estimates show substantial sentence-condition sensitivity in the Ridge–WavLM pipeline: its unseen-sentence values lie between 28.33% and 29.27%, whereas its seen-sentence values lie between 44.54% and 46.96%. CNN's corresponding values lie between 40.24% and 43.22% for unseen sentences and between 41.27% and 45.29% for seen sentences. These are descriptive summaries of the reported endpoints; no additional cross-model or sentence-gap test was performed. The two pipelines use different feature front ends, pretraining and training procedures, so this pattern cannot be attributed to classifier architecture alone or treated as a test of a memorization mechanism.

[Figure 1: all absolute UAR estimates](results/absolute_uar.png) · [PDF](results/absolute_uar.pdf).

### Paired allocation contrasts

Table 2 retains all 12 specified contrasts. The four unseen-sentence contrasts are primary. Differences are UAR percentage points, with pointwise 95% conditional intervals.

| Model | Recording budget | Δnew: S48 − S12 [95% interval] | Δseen: S48 − S12 [95% interval] | Δnew − Δseen [95% interval] |
|---|---:|---|---|---|
| Ridge–WavLM | 288 | +0.73 [0.07, 1.43] | -0.20 [-1.05, 0.62] | +0.94 [-0.09, 1.98] |
| Ridge–WavLM | 576 | +0.06 [-0.64, 0.76] | -0.46 [-1.40, 0.44] | +0.53 [-0.55, 1.62] |
| CNN | 288 | +1.08 [0.50, 1.67] | +1.30 [0.70, 1.91] | -0.22 [-0.92, 0.49] |
| CNN | 576 | +2.16 [1.54, 2.78] | +2.15 [1.57, 2.72] | +0.01 [-0.60, 0.65] |

For CNN under unseen sentences, changing from S12 to S48 while reducing each speaker's sentence coverage increased the estimated UAR by 1.08 points at B=288 [0.50, 1.67] and 2.16 points at B=576 [1.54, 2.78]. The corresponding absolute values were 40.24% to 41.32% and 41.06% to 43.22%. Both pointwise intervals exclude zero under the conditional bootstrap procedure. These results support a modest positive allocation contrast for this CNN configuration at both specified recording budgets.

For Ridge–WavLM, the unseen-sentence contrast was 0.73 points at B=288 [0.07, 1.43], with absolute UAR changing from 28.54% to 29.27%. At B=576 it was 0.06 points [−0.64, 0.76], with absolute UAR changing from 28.33% to 28.39%. The latter interval includes decreases and increases; it does not establish equivalence. The positive B=288 estimate also needs to be read alongside its draw sensitivity below. A difference in whether individual model-specific intervals exclude zero is not a formal test of a difference between models.

The paired seen-sentence results preserve the full pattern. CNN's contrasts were 1.30 [0.70, 1.91] and 2.15 [1.57, 2.72] points at B=288 and B=576, respectively. Ridge–WavLM's were −0.20 [−1.05, 0.62] and −0.46 [−1.40, 0.44]. Thus CNN's positive estimated benefit was also present when test sentence identities were represented during training. All four differences-in-differences intervals span zero: 0.94 [−0.09, 1.98] and 0.53 [−0.55, 1.62] for Ridge–WavLM, and −0.22 [−0.92, 0.49] and 0.01 [−0.60, 0.65] for CNN. The study does not establish that unseen sentences benefit more from the allocation change, and these intervals do not establish equality of the sentence-condition effects.

[Figure 2: all allocation and interaction contrasts](results/allocation_contrasts.png) · [PDF](results/allocation_contrasts.pdf). The full-precision values for both tables are in [results.csv](results/results.csv); [tables.md](results/tables.md) provides the complete tabular export.

### Sensitivity to the three specified draws

Table 3 reports the four primary contrasts in each complete draw. These values describe variation across the three executed partitions and training subsets, all drawn from the same corpus. They are not three independent replications and do not increase the sample size beyond 91 speakers.

| Model | Recording budget | Draw 0 Δnew | Draw 1 Δnew | Draw 2 Δnew | Three-draw range |
|---|---:|---:|---:|---:|---|
| Ridge–WavLM | 288 | +0.70 | +2.04 | -0.54 | [-0.54, 2.04] |
| Ridge–WavLM | 576 | -0.48 | +0.91 | -0.24 | [-0.48, 0.91] |
| CNN | 288 | +0.81 | +0.98 | +1.45 | [0.81, 1.45] |
| CNN | 576 | +1.86 | +2.56 | +2.07 | [1.86, 2.56] |

CNN's primary contrasts were positive in all three draws at both budgets; the corresponding seen-sentence contrasts were also positive in every draw. Ridge–WavLM's unseen-sentence contrasts changed sign across draws at both budgets, as did its seen-sentence contrasts. In particular, the aggregate positive Ridge–WavLM result at B=288 coexists with one negative draw and a range from −0.54 to 2.04 points. Its narrow conditional interval therefore must not be read as evidence that the benefit will persist across newly sampled training populations.

The complete [draw supplement](results/draw_estimates.csv) retains all three draw estimates and their ranges for **all 28 quantities**: 16 absolute endpoints and 12 contrasts. No draw, model, budget or contrast was removed because of its sign or magnitude.

### Execution characteristics

| Model | Completed units | Failed/retried fits | Sum of recorded fit seconds | Sum of per-unit wall seconds | Epochs run per unit, median [range] |
|---|---:|---:|---:|---:|---|
| Ridge–WavLM | 720 | 0 / 0 | 61.10 | 65.51 | Not applicable; no neural-network epochs |
| CNN | 720 | 0 / 0 | 6,974.70 | 7,009.27 | 55 [22, 100] |

These are sums of receipt-level durations. They exclude feature precomputation, engineering-pilot work, provisioning, data transfer, verification and reporting, and must not be interpreted as cloud-billing hours or end-to-end study time. CNN's median recorded best-epoch index was 40 [7, 93]. Epoch counts differ among fits because of the fixed early-stopping procedure; no configuration received a result-driven extension.

Ridge–WavLM ran on Windows with two CPU threads, Python 3.13.9 and scikit-learn 1.9.0. CNN ran on an independently authorized Linux host with an NVIDIA GeForce RTX 5090, eight CPU threads, Python 3.12.3, PyTorch 2.8.0+cu128 and CUDA 12.8. Each model retained one recorded environment across its 720 units. Model, representation and runtime differences preclude attributing between-pipeline outcome differences to architecture alone.

## Discussion

The experiment extends the evaluation-protocol question to a controlled data-allocation question. In this finite CREMA-D design, reallocating a fixed recording count from greater within-speaker sentence coverage toward more training speakers yielded modest positive CNN estimates in both sentence conditions. Ridge–WavLM showed smaller unseen-sentence estimates that varied in sign across draws, while its seen-sentence estimates were negative with intervals spanning zero. The complete pattern supports a configuration-specific account of the allocation tradeoff rather than one uniform collection rule. It also shows why absolute performance and repeated-partition sensitivity should accompany a mean gain.

The effect sizes matter for the scope of the practical claim. All four primary point estimates are below the three-percentage-point planning reference. That reference was a discussion scale, not a validated threshold for collection utility, detection power or equivalence. The observed gains can motivate further assessment of speaker coverage for a comparable CNN workflow, but their value cannot be converted into a recruitment recommendation from recording counts alone.

### Identifiability and collection costs

The estimand is a constrained joint allocation change. At fixed B, increasing the number of training speakers from 12 to 48 necessarily reduces each speaker's sentence coverage, from four sentences to one at B=288 and from eight to two at B=576. It cannot isolate a speaker-count effect while holding per-speaker data constant. Global sentence count, per-sentence exposure, class quotas and the specified demographic balance were held fixed, but unmeasured speaker attributes and particular sentence identities can still influence the finite-corpus estimates.

The budget counts distinct recordings, not recording duration, recruitment effort, annotation expense or currency. Recruiting four times as many speakers can entail costs not represented by B. Only two speaker counts and two recording budgets were examined. The experiment therefore does not identify an optimum, determine an economically efficient collection policy or validate a general-purpose adviser.

### Sentence conditions and model interpretation

The paired sentence construction controls graph degrees and exposures while substituting two sentence identities. It supports comparisons for these particular corpus sentences and arrangements, not a universal effect of arbitrary unseen linguistic content. The four differences-in-differences intervals span zero, so no sentence-condition-specific allocation benefit is established. This also prevents interpreting the positive CNN unseen-sentence contrasts as evidence that additional speaker coverage uniquely solves a novel-content problem.

Both conditions keep supervised-training and test speakers disjoint. Study II therefore does not re-estimate random-split speaker leakage or establish a mechanism for an earlier protocol gap. In addition, the WavLM-pretrained Ridge pipeline and the log-mel CNN differ in representation, pretraining, optimization and runtime; their descriptive outcome differences do not isolate a classifier-architecture effect. Speaker separation within this study does not establish that these individuals were absent from the pretrained encoder's original training data.

### Population and uncertainty limits

The corpus has a limited set of acted emotions, speakers and sentences. MD/XX selection, incomplete cells and eligibility for complete training panels restrict the population to which an allocation result could apply. Balancing the supplied sex strata does not establish fairness across demographic groups. No noise-proportion intervention, synthetic-speaker experiment, genuine-repeat intervention or cross-corpus validation was included. A benefit in real customer-service deployment, clinical assessment or other natural emotional-state applications has not been demonstrated.

Historical exposure and conditional uncertainty remain relevant after correct execution. The design was specified after earlier v2 analyses on an already used corpus; complete execution does not convert it into independent confirmatory evidence. The 95% intervals are pointwise and approximate, conditional on the executed training subsets, fitted models and sentence arrangement. Cross-validation models share training speakers, and the three draws reuse the same finite corpus. The Ridge draw reversals illustrate an uncertainty component that a bootstrap of the existing test-speaker vectors does not fully capture. Independent collection and external validation would be needed to assess transfer of any allocation recommendation.

## Conclusion

With recording count and global sentence exposure controlled, reallocating training data from 12 to 48 speakers increased CNN UAR on unseen speakers and sentences by 1.08 and 2.16 percentage points at the two specified budgets. Ridge–WavLM yielded smaller, draw-sensitive estimates of 0.73 and 0.06 points. The complete results provide evidence about a modest allocation tradeoff within these pipelines and this corpus; they do not establish a universal optimum or a benefit specific to unseen sentences.

## Reproducibility and accompanying files

This integration record accompanies the manuscript section and need not appear verbatim in the submitted paper.

- Study identifier: `SER26-STUDY2-CORE-1`.
- Locked plan SHA256: `f258cabe582d97b3a386a666b566730d00241a5ded0e14ad61d590cb3c861503`.
- Pre-execution commit recorded in the execution evidence: `6bf62b9be3ffebd669bec1dc312e724566d42d48`.
- The [frozen scientific specification](../../v3/data_design/SPEC_CORE_ZH.md), [prior-exposure disclosure](../../v3/data_design/PRIOR_EXPOSURE.md) and [execution contract](../../v3/data_design/CONTRACT.md) define the original design and analysis rules; they retain their historical wording.
- The [plan lock](../../v3/data_design/evidence/PLAN_LOCK.json), [metadata preflight](../../v3/data_design/evidence/preflight_verification.json) and [capacity audit](../../v3/data_design/evidence/capacity.json) preserve the pre-execution identities and feasibility checks.
- The archived full [completion record](results/completion.json.gz), [analysis lock](results/analysis_lock.json.gz), [merged-source provenance](results/merge_provenance.json.gz) and [attempt ledger](results/ledger.jsonl.gz) bind all 1,440 units to their unchanged model-block artifacts. The compressed records preserve the original bytes when decompressed.
- The [full-precision score](results/score.json), [analysis identity](results/score_identity.json), [per-speaker table](results/per_speaker.csv) and [independent verification](results/verification.json.gz) establish the results-to-text provenance. Verification passed for 91 speakers and 1,456 policy–speaker rows, with 6,199 numeric fields compared and maximum error 2.13 × 10⁻¹⁴ percentage points. Original execution paths remain recorded in the verification snapshot; the [bundle manifest](results/BUNDLE_MANIFEST.json) retains artifact hashes for the portable copies.
- Portable manuscript outputs comprise [all estimates and intervals](results/results.csv), [all draw estimates and ranges](results/draw_estimates.csv), [formatted tables](results/tables.md), [absolute-UAR figure](results/absolute_uar.pdf) and [allocation-contrast figure](results/allocation_contrasts.pdf). Tables in this section use two decimal places; the source exports retain full precision.

This completed section was prepared after full closure and independent scientific verification. It reports all 16 absolute endpoints, all 12 specified contrasts and the complete draw supplement, without adding exploratory confidence intervals, p-values or result-driven configuration searches.

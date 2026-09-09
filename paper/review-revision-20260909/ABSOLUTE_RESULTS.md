# Complete absolute outer-test results

This supplement preserves every setting in the manuscript's absolute-results table: three original-study settings and eight supplement settings. The results are descriptive. No new hypothesis tests, confidence intervals, or outcome-based exclusions are introduced on this page.

## Reading the tables

Each candidate model follows the same training trajectory within a setting. **Seen CE** and **Unseen CE** select the epoch with the lowest validation cross-entropy using speakers present or absent, respectively, from the fit set. **Seen UAR** and **Unseen UAR** select the epoch with the highest validation UAR using those same respective groups. Exact ties select the earliest epoch. **Last** uses the final epoch of the stated candidate window, without validation-based epoch selection. All five retained models are evaluated on the same unfamiliar-speaker outer test for that context; outer performance never selects the checkpoint.

The five absolute columns report **outer-test UAR in percent**, not validation scores or CE losses. UAR is macro recall computed over the complete outer-test role and the corpus's native classes: six for CREMA-D, seven for SUBESCO, and eight for RAVDESS. It is not the average of individual speakers' UAR. Each reported mean first weights the five outer folds equally within a draw, then weights the 24 draws equally. The 120 contexts per setting are not treated as 120 independent experimental samples. Different native tasks are not pooled.

**δUAR = outer UAR of Seen UAR − outer UAR of Unseen UAR**, in **percentage points (pp)**. A positive value favors the seen-based UAR selection rule on the common outer test. The difference is computed before rounding; subtracting two displayed three-decimal scores need not reproduce its final digit.

## Original study: all three main settings

These rows summarize the original 360 main A-trained trajectories. The 24 additional whole-group exchange controls are a separate descriptive analysis and do not enter these means.

| Corpus | Backbone | Window (epochs) | Seen CE (%) | Unseen CE (%) | Seen UAR (%) | Unseen UAR (%) | Last (%) | δUAR (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CREMA-D | WavLM | 15 | 56.253 | 55.318 | 57.154 | 57.155 | 53.715 | −0.002 |
| SUBESCO | WavLM | 15 | 47.113 | 44.866 | 47.426 | 47.738 | 47.292 | −0.313 |
| RAVDESS | WavLM | 15 | 33.941 | 33.455 | 32.943 | 32.925 | 32.821 | +0.017 |

Sources: [original aggregate results](../../v3/final_program_20260907/reports/scores/results.json) and [all original selected epochs and outer scores](../../v3/final_program_20260907/reports/scores/selected_epochs.csv).

## Supplement: all eight model–window settings

The supplement contains 360 new WavLM trajectories and 360 HuBERT trajectories. Original and supplement WavLM runs are distinct; their scores are not interchangeable paired baselines. Supplement SUBESCO and RAVDESS WavLM each train for 45 epochs. For these two tasks, the 15- and 45-epoch windows select from the prefix and full candidate set of the **same long trajectory**, respectively. The two windows are therefore paired descriptions, not additional independent fits. Supplement CREMA-D WavLM and all HuBERT trajectories train for 15 epochs. Every trajectory completes its fixed training budget; checkpoint selection does not stop training early.

| Corpus | Backbone | Window (epochs) | Seen CE (%) | Unseen CE (%) | Seen UAR (%) | Unseen UAR (%) | Last (%) | δUAR (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CREMA-D | WavLM | 15 | 55.121 | 54.103 | 56.541 | 56.430 | 53.240 | +0.111 |
| SUBESCO | WavLM | 15 | 48.110 | 45.774 | 47.946 | 47.470 | 47.009 | +0.476 |
| SUBESCO | WavLM | 45 | 48.393 | 45.774 | 49.613 | 49.122 | 48.839 | +0.491 |
| RAVDESS | WavLM | 15 | 34.983 | 34.766 | 34.028 | 33.325 | 34.141 | +0.703 |
| RAVDESS | WavLM | 45 | 42.700 | 40.226 | 43.516 | 42.405 | 42.491 | +1.111 |
| CREMA-D | HuBERT | 15 | 57.002 | 55.426 | 58.783 | 58.503 | 56.446 | +0.280 |
| SUBESCO | HuBERT | 15 | 48.616 | 44.598 | 49.836 | 49.435 | 49.048 | +0.402 |
| RAVDESS | HuBERT | 15 | 42.066 | 40.668 | 41.450 | 40.755 | 41.615 | +0.694 |

Sources: [complete supplement descriptive aggregates](../../docs/autodl-supplement-20260908/results/descriptive.csv), using `comparison=model_window`, endpoints `T_seen_ce`, `T_unseen_ce`, `T_seen_uar`, `T_unseen_uar`, `T_last`, and `D_UAR`; and [all supplement selected epochs and outer scores](../../docs/autodl-supplement-20260908/results/selected_epochs.csv). The `T_*` values are absolute UAR percentages; `D_UAR` is a difference in pp.

## Fixed-last comparison: complete post hoc description

The following eight contrasts were assembled after reviewing the completed results to make the fixed-last baseline explicit. Each is **Last − Unseen CE**, computed on the same context's outer test and averaged with the same five-fold-then-24-draw rule. These are post hoc descriptive differences, not additional prespecified hypothesis tests. No p values or confidence intervals are assigned here.

| Corpus | Backbone | Window (epochs) | Last − Unseen CE (pp) |
|---|---|---:|---:|
| CREMA-D | WavLM | 15 | −0.863 |
| SUBESCO | WavLM | 15 | +1.235 |
| SUBESCO | WavLM | 45 | +3.065 |
| RAVDESS | WavLM | 15 | −0.625 |
| RAVDESS | WavLM | 45 | +2.266 |
| CREMA-D | HuBERT | 15 | +1.020 |
| SUBESCO | HuBERT | 15 | +4.449 |
| RAVDESS | HuBERT | 15 | +0.946 |

The fixed-last mean is higher in four of the six 15-epoch settings and six of all eight settings. These are counts of observed mean directions, not counts of statistically established improvements or an estimated probability that a strategy will win on a new task. They do not recommend always using the last epoch. Both negative contrasts are retained. Likewise, no validation criterion is uniformly best: supplement RAVDESS WavLM15 favors CE over UAR for both validation roles, whereas SUBESCO WavLM15 and HuBERT RAVDESS have opposite CE-versus-UAR rankings across the two roles.

The 45-epoch last comparison remains a fixed-budget comparison within each long trajectory. Selecting an earlier stored checkpoint is distinct from actually terminating training earlier. These observations do not estimate the benefits of an online early-stopping algorithm or a different allocation of speakers between fit and validation sets.

## Numerical provenance and scope

The 55 absolute means were cross-checked against the complete original and supplement selected-epoch CSVs, averaging five folds per draw and then 24 draws. Their largest difference from the archived aggregate values was approximately 7.1 × 10⁻¹⁵ percentage points. The six displayed values in each of the 11 main-table rows match the archived manuscript table after three-decimal **decimal ROUND_HALF_UP** formatting. Archived unrounded δUAR values determine its displayed rounding, including original SUBESCO −0.3125 → −0.313 pp. The post hoc last differences use unrounded values, not subtraction of the printed table cells.

All original and supplement main settings are retained, including negative differences. These descriptive tables do not add tests to, or replace, the [original six-test results](../../v3/final_program_20260907/reports/scores/results.json) or the [supplement ten-test family](../../docs/autodl-supplement-20260908/results/primary_tests.csv). The conclusions remain conditional on the finite corpora, panels, model configurations, and candidate windows studied.

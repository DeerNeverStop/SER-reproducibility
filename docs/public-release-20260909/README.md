# Manuscript and numerical-presentation revision: v2026.09.09

This package accompanies [release v2026.09.09](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.09). The [v2026.09.08 release](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.08) and its original prediction bundle remain available separately.

## Revision contents

- [Revised English manuscript](../../paper/review-revision-20260909/english/xie.pdf) and [LaTeX source](../../paper/review-revision-20260909/english/main.tex).
- [Complete absolute-results supplement](../../paper/review-revision-20260909/ABSOLUTE_RESULTS.md), retaining all three original and eight supplemental settings, four selection rules, each window's fixed last epoch, and δUAR. Its complete eight-setting last-minus-unseen-CE comparison is explicitly post hoc and descriptive, not a recommendation to always use the last epoch.
- [Post hoc window figure](../../paper/review-revision-20260909/figures/window_selection.pdf) and [CSV-based generation script](../../paper/review-revision-20260909/scripts/make_window_figure.py). The figure summarizes the already completed supplemental WavLM trajectories on SUBESCO and RAVDESS. It changes the eligible checkpoint window from 1 to 45; all underlying trajectories completed 45 training epochs. It does not represent new training or actual early stopping.

The figure first averages five paired fold differences per draw, then uses 24 draw means. Its intervals are pointwise 95% t intervals with 23 degrees of freedom, conditional on the studied corpora and programs. The scan is post hoc: the bands are not simultaneous, and the figure adds no new confirmatory hypothesis tests or established population thresholds.

The revision makes the selection problem and practical examples more explicit while retaining the scope of the evidence. **All 16 primary tests are unchanged:** the original six-test Holm family and the supplement's separate ten-test Holm family, including uncertain and non-rejected outcomes. Descriptive examples retain reversals in policy ranking. There are no new model fits, prediction runs, or primary statistical results in this revision. The underlying completed studies remain 384 original formal fits and 720 supplemental formal fits; these counts are not numbers of independent speakers or draws.

## Public numerical evidence and its limits

| Material | Public entry and scope |
|---|---|
| Original 384-fit study | [Aggregate results](../../v3/final_program_20260907/reports/scores/results.json) and [score tables](../../v3/final_program_20260907/reports/scores/) |
| Supplemental 720-fit study | [All ten primary tests](../../docs/autodl-supplement-20260908/results/primary_tests.csv), [complete descriptions](../../docs/autodl-supplement-20260908/results/descriptive.csv), [selected epochs](../../docs/autodl-supplement-20260908/results/selected_epochs.csv), and [all 18,000 curve rows](../../docs/autodl-supplement-20260908/results/curves.csv) |
| Table-based CPU reproduction | [Existing commands and reproduction levels](../public-release-20260908/REPRODUCING.md); the original and supplemental tests retain their original inputs and separate families |
| Original prediction replay bundle | `formal_numeric_bundle.zip` remains attached to [v2026.09.08](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.08); [its identity and scope](../public-release-20260908/AVAILABILITY.md) are unchanged |
| Earlier manuscript | [2026-09-08 English PDF](../../paper/supplement-results-20260908/english/xie.pdf) and [source](../../paper/supplement-results-20260908/english/main.tex) remain historical entries |

The original ZIP supplies prediction replay for the **384-fit study**, not the supplemental 720-fit full prediction archive. For the supplement, the complete reported unit, draw, selected-epoch and curve tables are public; full per-epoch logits and retained checkpoints remain in a controlled archive. Table replay, original prediction replay, full-archive verification and independent retraining have different scopes. This revision does not equate them.

No raw audio, model weights, credentials, operational records or bundled conference style are added. Original dataset sources and reuse boundaries remain documented in [RIGHTS_AND_DATA.md](../../RIGHTS_AND_DATA.md). Earlier release notes are preserved with their original version labels. This publication does not turn historical pre-execution freezes into public preregistration or claim conference acceptance.

## 中文简述

本次修订只改善论文表述和图表呈现，没有新增训练，也没有改变原六项与新十项主检验。全部 11 个设置的绝对成绩迁入独立补充页；新图是在既有完整轨迹上进行的事后窗口描述。原 384 次拟合的预测回放 ZIP 仍从旧版发布下载，新 720 次拟合公开的是完整结果表与曲线，不代表其全部预测和权重已经公开。

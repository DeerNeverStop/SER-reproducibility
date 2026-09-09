# Manuscript revision: 9 September 2026

[English manuscript](english/xie.pdf) · [中文解释版](chinese/README.md) · [LaTeX source](english/main.tex) · [Complete absolute results](ABSOLUTE_RESULTS.md) · [Window figure and reproduction](figures/README.md)

This revision accompanies public artifact version `v2026.09.09`. It is an author-review manuscript, not a submission receipt or an acceptance claim. The [8 September manuscript](../supplement-results-20260908/english/xie.pdf) remains a historical version.

## Changes

- The abstract, contributions and conclusion directly state the observed dependence on validation population, selection criterion and candidate window.
- All 16 primary result rows retain their previous numbers and separate Holm6/Holm10 families. No training or confirmatory tests were added.
- A new two-panel figure shows post hoc W1--45 CE/UAR window profiles on the supplemental WavLM45 trajectories. Its pointwise intervals average five folds within each of 24 draws. The original 384-fit trajectories are not substituted for these prefixes.
- The complete absolute table moves to `ABSOLUTE_RESULTS.md`: all 11 original/supplemental settings, all four policies, last epochs, and descriptive UAR exposure contrasts. The manuscript retains absolute ranges and examples with different policy rankings.
- Wording distinguishes checkpoint choice from training termination, original B-trained controls from main trajectories, and the prespecified DUAL last-epoch description from later post hoc analyses.
- The artifact reference now names the public version. Public curves/tables/code and the original 384 prediction bundle are distinguished from controlled supplemental logits/weights.
- Author-confirmed funding and conflict declarations, the basis for the secondary-data ethics assessment, and the scope of AI assistance are included. No institutional ethics approval or waiver is claimed.

## Verification

The PDF has four technical pages, including AI disclosure, plus one page for ethics/funding/conflict statements and references. It retains the existing conference style without modifying its margins or font settings. The final figure uses text at least 9.2 pt after embedding. All five rendered pages were visually inspected; machine checks found no overfull boxes, missing references, unembedded fonts, Type 3 fonts, or text outside the checked layout.

[Machine checks](qa/MACHINE_CHECKS.json), [visual/file receipt](qa/BUILD_QA.json), and [scientific review](qa/FINAL_SCIENCE_REVIEW.md) bind the delivered version. The [figure audit](figures/window_selection_audit.json) checks W15/W45 selections, draw values and intervals against the archived results. These checks do not constitute independent training or an institutional ethics determination.

## Build and inspect

Obtain `spconf.sty` from the [official ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php) and place it beside `english/main.tex`. The third-party style is not redistributed in the public repository. With Tectonic 0.17.0, from this directory:

```text
tectonic -X compile english/main.tex --outdir /path/to/new-build-directory --keep-logs --keep-intermediates
```

The figure is already included. To regenerate it from the repository root, install Matplotlib and SciPy and follow [the figure instructions](figures/README.md). PDF byte identities can differ with renderer versions or timestamps; the delivered byte identity is recorded in QA.

For the PDF checks, install `pypdf` and `pdfplumber`, then run from the repository root:

```text
python -B paper/review-revision-20260909/scripts/check_manuscript.py --repo . --pdf /path/to/new-build-directory/main.pdf --log /path/to/new-build-directory/main.log --out /path/to/new-build-directory/checks.json
```

The checker imports the existing PDF font-inspection helper and preserves the original numerical rows. Its output does not replace inspection of the rendered pages. The public [16-test replay](../../docs/public-release-20260908/REPRODUCING.md) remains the numerical verification entry point.

## 中文说明

这版把论文的发现讲得更直接：同样的训练和陌生人测试，仅改变验证人群、选模指标或候选轮数，最后拿来报告的模型与成绩就会变化。新增图方便看懂这种变化；完整成绩表仍公开保留。没有新增GPU训练，也没有把事后图包装成新的显著性证据。原始16项主检验、负向例子和不确定结果都保留。

资助和利益冲突声明依据作者确认填写；伦理部分说明公开资料二次分析的政策依据及没有取得机构审批/豁免，AI辅助披露保留。此前中文PDF仍为历史译稿，本次没有将其冒称为新版全文翻译。

现另附[对应本版的7页中文解释版](chinese/README.md)，解释方法、指标和结论边界，保留全部16项主检验及11种设置的绝对成绩。

# 纳入 AutoDL 补充结果的英文论文

2026-09-08。当前稿为 [英文 PDF](english/xie.pdf) 和 [LaTeX 源码](english/main.tex)，配套[中文实验汇报](../../docs/autodl-supplement-20260908/RESULTS_REPORT_ZH.md)。本稿包含已验收的原 384 次正式训练和另行冻结的 720 次 AutoDL 补充训练；此前英文稿与中译 PDF 保留为历史版本，未冒称同步更新。

论文保留原六项主检验及全部新十项主检验。新实验支持 CREMA-D、SUBESCO 在第二骨干上的 CE/UAR 交互，以及 RAVDESS 的 CE 曝光差随候选窗口改变；RAVDESS 的窗口交互未通过 Holm10，正文明确保留。所有模型、窗口和四种选模规则的绝对成绩均进入表格，结论没有扩展为 UAR 一贯最好或已经消除说话人差距。

当前 PDF 为 5 页：4 页正文（含 AI 使用披露）和 1 页参考文献。沿用已有官方 `spconf.sty`，没有压缩字号或改边距。已编译并逐页检查，具体源文件、PDF、编译器及检查记录见 [QA 清单](qa/BUILD_QA.json)。科学终审见 [逐项审查](qa/FINAL_MANUSCRIPT_CLAIM_REVIEW.md)。

为在正文完整保留两组主检验及绝对成绩，本版用三张表呈现证据。旧完整曲线图保留在[此前图稿](../figure-revision-20260907/README.md)和旧研究归档；本轮全部 18,000 个 epoch 的数据保留在[结果目录](../../docs/autodl-supplement-20260908/results/curves.csv)，没有按显著性筛选曲线。

用 Tectonic 0.17.0 从本目录编译，输出到一个新目录：

```text
tectonic -X compile english/main.tex --outdir NEW_BUILD --keep-logs --keep-intermediates
```

原始科学源码和计划冻结身份见[完整结果与证据](../../docs/autodl-supplement-20260908/evidence/README.md)。排版编译可能因 PDF 时间戳产生不同文件 SHA；交付 PDF 的实际 SHA 以 QA 清单为准。当前是供作者审阅的稿件，尚未投稿。

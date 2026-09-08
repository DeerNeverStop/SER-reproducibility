# 最终双语论文与完成实验

2026-09-07 交付稿。384 次正式训练、完整工件/账本验收、独立数值复算、预定检查点恢复及预测包迁移验收均已完成。论文已按完整结果重写：英文 5 页（前四页技术内容、第五页仅参考文献），中文完整对应 8 页。两版通过结构和全页视觉检查。**未向会议投稿。**

- [英文 PDF](english/xie.pdf) · [LaTeX](english/main.tex)
- [中文 PDF](chinese/explainer.pdf) · [可编辑正文](chinese/中文解读.md)
- [实验方法、结果和论文取舍的人话汇报](RESEARCH_REPORT_中文.md)
- [完整科学解读](review/FINAL_SCIENCE_INTERPRETATION.md) · [独立主稿红队](review/ACTUAL_MANUSCRIPT_SCIENCE_REVIEW.md)
- [完整轻量证据](../../v3/final_program_20260907/reports/README.md) · [共用精确事实](../../v3/final_program_20260907/reports/paper_facts/facts.json)
- [源码与轻量证据包](source_bundle.zip) · [交付清单](DELIVERY_MANIFEST.json)
- [PDF 结构检查](qa/structural.json) · [实际嵌图字号](qa/figure_geometry.json) · [逐页目检](qa/visual_review.json)

核心结果是**验证说话人曝光的选模后果依赖 CE/UAR 准则**。共同轨迹与外测下，CREMA-D、SUBESCO 的 CE 已见验证优势为 +0.935/+2.247 pp，两库准则交互也通过预定六项 Holm 控制；RAVDESS 仍不确定。UAR 规则的描述性近零差不是等效性证明。新稿保留全部三库、六个主检验、绝对成绩及完整曲线，不声称分组验证或 UAR 在所有任务中更优。

TTS 正式 E3 是条件候选：未满足准入，未执行正式合成训练；人评未完成不能写成负结果。旧 Study II、E2、N14R2、DUAL 和诊断分析保留各自估计对象与推断身份，不拼成独立样本。原 submission-20260906、旧工作稿检查及 prepared/ 下标记的模拟稿保留历史身份，不替代最终 PDF。

## 构建

作者为 Tian Xie、University of Toronto、tianjack.xie@mail.utoronto.ca。英文保留原始 spconf.sty，SHA dc5d632639040cb183f2ab62780f314845aa021be73048c8a9ec9c2072d64a86，未压缩官方版式。实际使用 Tectonic 0.17.0；Python 3.12.14、ReportLab 4.4.9、pypdf 6.10.0、pdfplumber 0.11.9、pdfminer.six 20251230；Poppler 渲染。中文使用具有使用权限的 Windows simsun.ttc（子字体 0）与 simhei.ttf，哈希见中文构建收据；字体文件与编译器缓存不再分发。

在仓库根目录或源码包解压根目录执行，工具须已安装在 PATH；先创建新的 build/english 输出目录。

```text
tectonic -X compile paper/final-20260907/english/main.tex --outdir build/english --keep-logs --keep-intermediates
python -X utf8 -B paper/final-20260907/chinese/build_chinese.py --source paper/final-20260907/chinese/中文解读.md --output build/chinese/explainer.pdf --font-dir C:/Windows/Fonts --figure-manifest paper/final-20260907/figures/FIGURE_MANIFEST.json
python -X utf8 -B paper/final-20260907/tools/check_pdfs.py --english build/english/main.pdf --chinese build/chinese/explainer.pdf --output build/structural.json
python -X utf8 -B paper/final-20260907/tools/check_actual_figures.py --english build/english/main.pdf --chinese build/chinese/explainer.pdf --curve paper/final-20260907/figures/complete_epoch_curves.pdf --out build/figure_geometry.json
```

Tectonic 首次构建可能需要获取 TeX 资源。本次实际构建和渲染记录在 qa/；中文图为矢量嵌入，图字最小值英文 9.533、中文 9.205 个物理 PDF 点。程序检查不替代目检。review/build_actual_manuscript.py 是红队修订前的历史整合工具，**不是最终论文重建命令**。

## 实验重放与访问

科学冻结提交为 976297d824ea98816bdc0948befd704e612a54c1，计划 SHA 为 393109434af0bfb6d18205f8f3713aa5e08d08f0f0ac3ffc4a63f8e635422c25。论文提交和完成标签不是预执行冻结；384 次拟合不是独立人群样本。

预测包 formal_numeric_bundle.zip 为 59,897,568 字节，SHA c3f14905050f13401788a926d21dbb754cf28031499bfa77abef913315b14e6f，在私有仓库 SER26-final-program-complete-20260907 release 单独提供，不进入普通 Git 历史或论文源码 ZIP。源码包用于论文重建与轻量证据阅读；科学重放使用完整匹配的仓库 checkout 和预测包，见 [PORTABLE_BUNDLE.md](../../v3/final_program_20260907/PORTABLE_BUNDLE.md)。音频、基座及全部检查点保留受控原归档；预测包不能代替原始音频重训或全权重验收。

仓库私有，release 不改变访问权限；须安排审稿人访问，不能称公开复现或公开预注册。实际投稿前由作者审阅最终内容、确认署名及真实伦理/资助披露、填写 ORCID 等资料，并确认届时会议要求。本任务未代签、伪造批准或提交会议。

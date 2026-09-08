# SER 论文交付与投稿前检查

生成日期：2026-09-06。目标：ICASSP 2027 regular conference paper。

作者：Tian Xie；University of Toronto；tianjack.xie@mail.utoronto.ca。

## 两版稿件

英文使用 ICASSP 2027 官方 `spconf.sty` 模板：4 页技术正文加 1 页仅参考文献，US Letter、双栏、正文 10 pt、表格 9 pt、不印页码。作者信息、摘要、关键词、11 条参考文献和 AI 使用披露均已填入。中文为 8 页逐节意译和解释，保留相同实验、数值和结论边界，不是另一套科学结果。

最终两版 PDF 均已渲染并逐页目检，未见截断、重叠、乱码或破损表格；中文表格保持整表同页。两版的字体均嵌入，未加密，文件均小于 5 MB。英文第 5 页只含参考文献。自动报告中要求人工复查的项目已由本轮页面检查补足，但不替代作者对科学内容和投稿的批准。

可编辑文件：`english/main.tex` 与 `chinese/中文解读.md`。`chinese/build_chinese.py` 是中文 PDF 排版脚本。`tools/check_pdfs.py` 是只读 PDF 结构检查器；自动检查不能代替逐页目检。`pdf_qa.json` 记录最终 PDF 的结构检查和 SHA-256。

英文采用官方模板和本地检查，不代表投稿系统已经接收或通过最终验证。本次没有投稿、发布论文、更改 GitHub 可见性，也没有重新运行实验。

## 投稿前尚需作者处理

1. 审阅并批准全文：尤其作者资格、单位、所有实验的证据链、引用，以及 AI 使用披露。没有替作者作出最终批准。
2. 准备作者 ORCID，并确认投稿系统里的姓名、顺序、单位、邮箱与稿件一致。当前只有一位作者，依据本次提供的信息填写。
3. 决定如何向审稿人提供实验材料。SER 仓库在本次检查时为 **PRIVATE**，未登录者不能访问。冻结 N14R2 规范要求执行前的公开收据，但目前不能核实该收据当时是否公众可见。因此稿件使用更保守的“执行前冻结方案／预先指定分析”描述，而非声称已完成公开预注册，并明确披露这项限制。可以另行决定公开经过审查的非敏感复现材料，或使用会议允许的补充材料渠道；这需要新的明确授权。公开今天的材料，不能倒推证明它在实验前就公开过。
4. 确认是否有需要披露的真实资助、利益冲突、额外作者或伦理说明。没有编造资助项目、IRB 批准或伦理豁免；稿中只说明未新收集参与者数据。
5. 在正式上传时再次检查会议页面、提交 PDF 与系统提取结果。任何后续内容修改后都应重新编译、检查页数和逐页查看。

## 核对的官方规则

英文技术内容最多 4 页，可另加第 5 页仅放参考文献；字体和 PDF 技术要求按官方套件检查。会议并非匿名审稿，需真实署名；AI 辅助应在致谢中如实披露。

- [ICASSP 2027 Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/)
- [ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
- [Submission Instructions](https://2027.ieeeicassp.org/paper-submission-instructions/)

官方模板文件保持原样：

- `spconf.sty` SHA-256：`dc5d632639040cb183f2ab62780f314845aa021be73048c8a9ec9c2072d64a86`
- `IEEEbib.bst` SHA-256：`7e6ca0c8b72158d504a12bb091f817c07032a021ba41ca783cca4c2dd80d570b`

## 科学内容的重要边界

- 原始 v2 的多语料结果使用说话人配对统计与整名说话人 bootstrap；新 N14R2 的推断单位是 24 个完整 draw。两套区间没有合并。
- N14R2 的 1,920 次训练均完成；统计样本量不是 1,920。中断的 N14R 数据没有混入新实验。
- 新主要结果是验证乐观偏差增加 1.419 个百分点，95% t 区间 [0.901, 1.937]。不是测试性能下降 1.419 点。D09R2 为描述性结果，不构成等效性或无损害证明。
- 保留并披露 `n_planned_draws` 的输出元数据勘误（24 改 28）；科学数值没有改写。
- 冻结验证器中的缓存字节检查 false 与另行完成的云端 PINS 核验被分开说明，没有伪称前者通过。
- 访问受限的仓库记录：`SER26-prereg-1`、`SER26-N14R2-prereg-1`、`SER26-N14R2-complete-20260906`。

源证据工作树固定在提交 `4fe0bb52f15b56a6626363be31a378b0f9661293`。本次论文编辑位于独立交付目录，没有修改冻结实验源码、计划、锁或结果文件。

核对的结果文件 SHA-256：

- 原始 v2 `results.json`：`7e09783aed6003e1fbf47e89f839d061e5d3f5d1655bb9de2c8c02508577a016`
- 原始 v2 `verification.json`：`002756e8e59626d600bec54eb6f3f69c5425d66be5a9b8ab675361400b836cd7`
- N14R2 `n14r2_results.metadata-corrected.json`：`92146e7dcf01cc8fd923f78d08017e638d501732ff23110008caa64b5f6ef6c6`
- N14R2 `n14r2_verification.metadata-corrected.json`：`e82f088d88770e6a201ad93b581a5460aa9fba27658a9810ba63a9ed115cf843`

## 重新生成

英文用 Tectonic 0.17.0 编译，T1 字体编码放在 `spconf` 之前，避免 XeTeX 的字体回退。参考文献在 `main.tex` 的 `thebibliography` 内，当前编译不需要 BibTeX。提供的 `IEEEbib.bst` 是未修改的官方配套文件，留作后续编辑。

从解压后的源目录运行（需相应工具已安装并可用）：

```powershell
tectonic --outdir . english/main.tex
python chinese/build_chinese.py --output SER_中文解读.pdf
python tools/check_pdfs.py --english main.pdf --chinese SER_中文解读.pdf --output pdf_qa.json
```

中文生成依赖 `reportlab`；检查依赖 `pypdf` 和 `pdfplumber`。中文脚本默认使用 Windows 的 `C:/Windows/Fonts/simsun.ttc` 与 `simhei.ttf`，支持 `--font-dir`。源包不附带 Windows 字体、编译器缓存、原始音频、特征缓存、训练输出、账户信息或其他私有资料。

需要进行版面复查时，使用 Poppler `pdftoppm -png` 渲染每一页。所有实质改动后都需重新检查，不能沿用旧 QA 报告作为新 PDF 的通过记录。

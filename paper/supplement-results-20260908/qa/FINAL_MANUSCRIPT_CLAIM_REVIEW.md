# 最终英文稿科学主张复核收据

结论：**通过本次有限范围的科学/统计文字核对，当前未发现剩余阻断问题。** 不包含PDF分页/视觉、完整引用原文或实际投稿行政验收；未修改稿件、冻结源或结果，也未重新运行实验/主评分/全归档审计。

- 核对时间（UTC）：2026-09-08T17:00:34.875625+00:00
- 主稿：`C:\Users\jock8\Documents\ChatGPT\论文\SER-speaker-execution-20260906\paper\supplement-results-20260908\english\main.tex`
- 通过版本 SHA256：`8c8d28974e6facc46f6d5a08375360ba455bb64e842c309839c119ccb5eb3ae8`

## 核对内容

- 主检验表（tab:primary）完整16行：旧6行与已验旧稿一致；新10行逐项对CSV核对均值、pointwise CI、raw p及Holm p，按实际显示精度及ROUND_HALF_UP一致。绝对表现表（tab:absolute）完整11行：旧3行与旧稿一致，新8行四规则/Last/D_UAR与描述CSV的显示值全部一致。没有重算原预测。
- 第68、74、79–98行：原Holm6独立，新B/C两个面板共同Holm10；不是各自6/4或事后Holm16。CI明确为pointwise、非同时区间、非等效检验；未扩大独立样本量。
- 第15、131、134–136行：摘要所述6个通过的新效应数目正确；HuBERT三个D_CE及CREMA/SUB两个J，加RAV窗口L_CE。RAV L_J虽raw p=.0183、pointwise CI排零，Holm=.0731未通过；SUB两个L与HuBERT RAV J的未明确结果均保留。
- 第139–141行：UAR次序按验证角色限定，未声称普遍优于CE或消除差距；D_UAR的0.111–1.111范围及RAV45描述CI[0.168,2.054]准确，没有新增确认性检验。
- 第54、56、66–70、151行：15/45为同一新长轨迹内的候选窗口，新WavLM配对基线未换成旧Windows结果；骨干/预训练混杂、不完整矩阵、有限语料和复用人/面板、政策比较而非声纹/语言机制均保留。第56行两个独立15轮技术companion的history/logits最大差0、选轮相同，得到已保存pilot gate支持，未冒充推断重复。
- 第148行：DUAL固定末轮已存在的optimism仍限定为历史不同估计量，没有归因到全部checkpoint选择。
- 第153–155行：score.py的audit_inputs在科学聚合前逐720正式单元核输入/source/plan/env及保留或释放证据；之后独立closeout审计确认720正式、8pilot、90个正常退出批次和20个保留检查点。独立数值复算报告passed，199,780个scalar comparisons最大差7.105427357601002e-15，对应稿中7.1×10^-15。保留12正式+8pilot的fresh-process报告，不声称全部权重永久保存或独立重训。

## 阅读期间已由根代理修正的三处

1. 第139行原“SUBESCO WavLM15在both roles中CE均更高”和“HuBERT RAVDESS CE均更高”不实：两者均为Seen CE较高、Unseen UAR较高。当前版本已逐角色改正；表本身原来即正确。
2. 第153行已明确其他权重是在predictions和metadata完成异地验证后释放，避免暗示所有权重均异地备份。
3. 第155行已改为主评分准入检查在先、独立全归档审计在后。原笼统“full archive checks preceded scoring”会混淆时间及范围，当前版本已消除。

上述均已在本收据绑定的SHA版本中确认修复，不再列为未解决问题。未重哈希大checkpoint或重放199,780项；这部分引用已完成的独立审计原件，而不是把本次文字核查称为新独立数值复算。

## 绑定证据

- `C:\Users\jock8\Documents\ChatGPT\论文\SER-speaker-execution-20260906\paper\supplement-results-20260908\english\main.tex` — `8c8d28974e6facc46f6d5a08375360ba455bb64e842c309839c119ccb5eb3ae8`
- `C:\Users\jock8\Documents\ChatGPT\论文\SER-speaker-execution-20260906\paper\review-revision-20260908\english\main.tex` — `021fbeb933ef42debcd44e9f47074ce30247d38847d3fe516cc7e62dd6573cab`
- `D:\SER-final-program-20260908\final_scores\primary_tests.csv` — `19f3adb14410076acf1e4ca4b5b155b5fe4a0888408d69c8cef1511ef526a004`
- `D:\SER-final-program-20260908\final_scores\descriptive.csv` — `1899a7ef7d6b49be7bd640ff717909b47c938ed9b0a65bf4a772ba661b15660b`
- `D:\SER-final-program-20260908\final_scores\results.json` — `bf6e41c9ef39465c566da737add9e2cf0c2b4e707e751fa42f3fe2991131630f`
- `D:\SER-final-program-20260908\final_scores\input_audit.json` — `872884c9575d6a49c7520c858e75a937eeb9693c8c80c29dd2960ab45d7b0962`
- `D:\SER-final-program-20260908\autodl_operations\closeout_verified_v1.json` — `3537f9ad050c048575617efbc53fdf561f29037adeb15dcb6754626cacc7605b`
- `D:\SER-final-program-20260908\independent_supplement_check\FINAL_NEW\independent_replay.json` — `c54bae0c2ed7f3d8f81ae8fb3eb17093021e1890e0335d90acfd0eb33ddf5cb0`
- `E:\SER-autodl-supplement-20260908\backup\pilot_gate.json` — `686e7abfa0156f64bd55a3276b9f11602964c59a6e55a8262b2aef907c1b4741`

# Ibrahim 2026：直接相关工作补核

核查日期：2026-09-07。Ibrahim、Mohamed Ezzat Ghoraba、Ahmed Ezzat Ghoraba 的 *Multimodal emotion recognition using hybrid deep feature fusion under speaker-independent evaluation* 确实存在，Scientific Reports 16:19584，2026-06-25 发表，DOI **10.1038/s41598-026-58836-w**。[出版社全文](https://www.nature.com/articles/s41598-026-58836-w?error=cookies_not_supported)；[PubMed 元数据](https://pubmed.ncbi.nlm.nih.gov/42342860/)。

本轮实际重开了出版社完整 HTML 与表 6。无参数链接间歇返回身份重定向错误，公开的 cookies_not_supported 参数链接成功；不能把短暂访问错误说成论文不存在或只能看到摘要。2026-09-06 的工作区研究笔记也已读过该文，但新 RESEARCH_DIRECTION 六篇表没有列出它。**应补列这一直接相关来源；“此前做过有界检索”不足以解释遗漏已明确相关论文。**

论文主系统融合手工音频特征与深度视觉特征。§3.5.3“Model training and evaluation”明确：两语料随机分层 72/8/20；RAVDESS 用 LOSO，CREMA-D 用人物互斥五折。§3.6 模型参数说明使用经验验证并提到 validation macro-F1。它直接覆盖人物重叠与人物独立评估的分数差。[方法全文](https://www.nature.com/articles/s41598-026-58836-w?error=cookies_not_supported)

**不能用“它是多模态”将其排除：表 6 同时提供 audio-only 消融。**

| 表 6 分支 | RAVDESS 随机 accuracy | RAVDESS LOSO accuracy | CREMA-D 随机 accuracy | CREMA-D 人物互斥五折 accuracy |
|---|---:|---:|---:|---:|
| Audio-only | 70.83% | 39.72% ± 10.99 | 54.06% | 47.10% ± 3.39 |
| Full multimodal | 95.83% | 48.06% ± 9.76 | 73.54% | 53.12% ± 2.65 |

来源：[出版社表 6](https://www.nature.com/articles/s41598-026-58836-w/tables/6?error=cookies_not_supported)。表中还同时列 macro-F1；以上为 accuracy，**不是 audio-only UAR**。不能把摘要的 full-model 数字借给音频分支，也不能将这些“±”未经定义就写成 95% CI。其 audio-only 的随机−独立差为 RAVDESS 31.11、CREMA-D 6.96 个 accuracy 百分点，定性上已经包含前者差距较大的观察。

本次读到的方法没有报告：共同拟合录音／同一组候选训练轨迹下独立操纵内层 seen/unseen 选择；固定同一外测后比较 selected-model V−T；或本项目固定末轮、人物对半的方向性选择与报告分析。其学习曲线改变训练规模，不等价于协议间严格等量配对；独立评估方案的内层验证分组细节不能从“speaker-independent”总标签自行补全。这里的“没有报告”以原文为界，不断言作者代码不存在这些操作。[方法与结果](https://www.nature.com/articles/s41598-026-58836-w?error=cookies_not_supported)

所以它是“重叠评估会高估陌生人物泛化”的直接先例，也有音频单模态证据；它不自动等同于本项目具体内层估计量。应明确承认碰撞并缩窄增量，不能由这些差别直接保证论文新颖性。


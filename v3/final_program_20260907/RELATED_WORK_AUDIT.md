# 六组直接相关原文核查

2026-09-07。用于最终论文定位；不修改冻结研究或原稿，不读取新正式结果。科学内容依据出版社全文、作者论文，不以Claude评语或其他论文的引用代替原文。IEEE页码/DOI另核对出版社登记的Crossref元数据。下列“未报告”只限实际查读的方法与结果，既不证明作者代码从未做过，也不是全领域首次的证明。

## 1. Ibrahim et al., 2026：纯音频外层比较已存在

Elhossiny Ibrahim, Mohamed Ezzat Ghoraba, and Ahmed Ezzat Ghoraba. “Multimodal emotion recognition using hybrid deep feature fusion under speaker-independent evaluation.” **Scientific Reports 16, 19584**, 25 June 2026. DOI **10.1038/s41598-026-58836-w**。[出版社全文](https://www.nature.com/articles/s41598-026-58836-w?error=cookies_not_supported)、[表6](https://www.nature.com/articles/s41598-026-58836-w/tables/6?error=cookies_not_supported)。

§3.5.3比较随机72/8/20、RAVDESS LOSO与CREMA-D人物互斥五折。表6的Audio-Only准确率分别为70.83→39.72、54.06→47.10；同时列macro-F1，不能当UAR或把“±”直接命名95%CI。§3.6提验证调参，但未报告SI协议内验证如何按人物分组。没有所读原文中的共同轨迹双验证或固定同一外测的Δ(V−T)干预。

**定位判断：**必须承认纯音频SD/SI分数差及两库异质性的先例；不能以其主系统多模态将之排除。区分点应是具体内层实验，而非重新发现外层落差。

## 2. Zielonka et al., 2022：直接划分先例，勿扩大表3的语料范围

Marta Zielonka, Artur Piastowski, Andrzej Czyżewski, Paweł Nadachowski, Maksymilian Operlejn, and Kamil Kaczor. “Recognition of Emotions in Speech Using Convolutional Neural Networks on Different Datasets.” **Electronics 11(22), 3831**, 21 November 2022. DOI **10.3390/electronics11223831**。[出版社](https://www.mdpi.com/2079-9292/11/22/3831)、[作者机构保存的原论文](https://mostwiedzy.pl/pl/publication/download/1/recognition-of-emotions-in-speech-using-convolutional-neural-networks-on-different-datasets_75193.pdf)。

§4–7采用五个语料、CNN/ResNet18、四或六类任务，表3比较随机与演员不重叠的**test accuracy**。其行是TESS、IEMOCAP、合并数据，以及ResNet18的CREMA-D/RAVDESS/SAVEE/TESS四库合并。**不是分别报告CREMA-D与RAVDESS各自的两协议落差或其排序。**所读方法没有独立inner因素、同一外测V−T或匹配训练N的曝光实验。

**定位判断：**支持“外层演员划分影响识别表现”已有实证；当前碰撞矩阵可保留直接相关级别，但不能把合并语料行误写成单库复现。它的人类CREMA-D标注问卷也不等于本项目新模型结果的主标签。

## 3. Kumari et al., 2026：同处理流程、三模型、五种子已有人做

Parveen Kumari, Yogita Yashveer Raghav, Vimmi Kochher, et al. “Hybrid CNN-embedding fusion with MFCC-SVM for speech emotion recognition: Random vs actor-wise evaluation on CREMA-D.” **PLOS One 21(8), e0355238**, 24 August 2026. DOI **10.1371/journal.pone.0355238**。[出版社全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0355238)。

§3–4明确纯音频MFCC-SVM、log-mel CNN和特征融合SVM；CREMA-D六类，随机与actor-wise外层划分，共同预处理、五seed均值±SD。§4.4–5区分accuracy、UAR、macro-F1。所读主文没有给共同外测下的inner随机/分组干预、独立双验证检查点对比；这些不能从“protocol-aware”标题自行补全。正文固定超参和五种子也不证明跨协议训练语句成员/条数严格匹配。

**定位判断：**不能仅凭“统一预处理＋多模型＋多种子＋SD/SI”主张新框架。我们的差异需落到可识别估计量及控制条件；该文未报告特定内层组合，不代表其作者没有验证集。

## 4. SERAB：三个分区全部人物独立已有明确规范

Neil Scheidwasser-Clow, Mikolaj Kegler, Pierre Beckmann, and Milos Cernak. “SERAB: A Multi-Lingual Benchmark for Speech Emotion Recognition.” **ICASSP 2022, pp.7697–7701**. DOI **10.1109/ICASSP43922.2022.9747348**。[DOI](https://doi.org/10.1109/ICASSP43922.2022.9747348)、[作者原论文](https://arxiv.org/pdf/2110.03414)、[IEEE登记元数据](https://api.crossref.org/works/10.1109/ICASSP43922.2022.9747348)。方法阅读的是2021年上传的作者稿；不能将上传年当会议年。

§2.1–2.2覆盖九库六语言。一般60/20/20，CREMA-D70/10/20，train/validation/test人物互斥。冻结特征提取器后，以train/validation网格选分类器，在独立test评价。指标是各任务accuracy，再作跨任务UM/WM/GM；**UM不是类宏召回UAR**。没有报告inner曝光的随机干预或共同轨迹双验证。

**定位判断：**“validation也必须考虑人物独立”与多语料统一评估都非首次；本项目采用不同划分是为了配对干预，并不等于SERAB基准成绩或语言因果比较。

## 5. Atmaja & Sasou：speaker×text四格且匹配训练N已有先例

Bagus Tris Atmaja and Akira Sasou. “Effect of different splitting criteria on the performance of speech emotion recognition.” **TENCON 2021, pp.760–764**. DOI **10.1109/TENCON54134.2021.9707265**。[作者全文](https://arxiv.org/html/2210.14501v1)、[作者书目信息](https://arxiv.org/abs/2210.14501)、[DOI](https://doi.org/10.1109/TENCON54134.2021.9707265)。arXiv上传于2022，不改变TENCON2021归属。

§III、表III–IV：JTES100人四类、声学特征MLP；SD/SI/TI/STI四条件。实验3各条件**14,400训练、400测试**；内部五折训练/验证。指标WA即整体accuracy，报告SE，不能说没有重复或不确定性。§IV-B明确不同条件的测试成员不同。没有独立outer×inner交叉或固定同一外测的V−T。

**定位判断：**匹配训练条数与speaker×text四格是扩展先例。可区分的是本项目固定test及共同val的训练曝光替换；这种替换仍改变训练组成，不能独自识别纯声纹机制。

## 6. Antoniou et al.：分组验证与复现审计有明确ICASSP先例

Nikolaos Antoniou, Athanasios Katsamanis, Theodoros Giannakopoulos, and Shrikanth Narayanan. “Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check Case Study with IEMOCAP.” **ICASSP 2023, pp.1–5**. DOI **10.1109/ICASSP49357.2023.10096808**。[USC保存的IEEE原论文](https://sail.usc.edu/publications/files/Antoniou-ICASSP2023.pdf)、[DOI](https://doi.org/10.1109/ICASSP49357.2023.10096808)、[IEEE登记元数据](https://api.crossref.org/works/10.1109/ICASSP49357.2023.10096808)。

§3.2–3.3讨论SD/SI、脚本/即兴、happy/excited标签处理；推荐IEMOCAP四类十折，8人训练、1验证、1测试。WA是accuracy，UA是类平均recall。§4/Table3实际复现公开模型，说明标签与评估假设影响结果；该表主要报WA。没有共同候选训练轨迹下单独操纵inner、固定外测测Δ(V−T)的实验。

**定位判断：**不能写它只重视test、忽略validation；“评估协议影响比较、代码公开不等于可复现”也已有先例。本项目增量是特定对照的量化和估计对象区分。

## 最终稿可用的定位边界

上述核查支持将论文定位为**受控评估研究**，不支持领域首次发现speaker leakage。外层复现、曝光棋盘、内层N14R2、共同轨迹DUAL与新384程序必须分别标明估计对象；不能将它们合为一个“偏差”数字。

相对于本次查读的六组原文，可以具体解释：保持共同SI外测；分别改变训练曝光或验证选择规则；在同一完整轨迹比较seen/unseen与CE/UAR，并保存全部轮次的外测预测作事前限定的诊断。这是**本次对照后的定位判断**，不是已完成的全领域新颖性证明。新384结果尚未知，不能提前写成“证实检查点机制”；三库差异也不是语言因果效应。

所有引用宜进入最终英文稿及中文解读，且以相同的限定语描述。不应再把书目遗漏修正成“相关工作不重要”，也不应因为存在先例而抹掉更严格控制的增量。

## 查读与可及性说明

- Ibrahim、Kumari出版社完整HTML及关键方法段已重开；Ibrahim表6直接重开。Kumari表题与正文已核，S1/S2补充CSV链接本次重定向失败，未据此宣称其所有补充材料不存在inner信息。
- Zielonka出版社搜索收录全文与此前保存的12页原PDF交叉读取，重点§7/Table3；本次直接HTML触发429，以原论文内容为依据，不以ResearchGate摘要代替。
- SERAB、Antoniou作者/机构PDF的评估方法页已直接读取；SERAB会议版本没有逐字比对作者稿，出版年/页码/DOI另核IEEE登记元数据。Atmaja作者HTML方法与表格及自报会议书目均已读取。
- 本轮没有运行六篇论文的训练代码，没有给论文检测AI使用率，也没有据文字风格推断作者研究诚信。

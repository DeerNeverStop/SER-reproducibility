# 验证说话人曝光与选模后报告：研究定位

记录日期：2026-09-07，收到 Claude PR #16 后修订。本文件梳理已有证据、探索设计和下一项确认问题，不改变此前冻结协议，也不把事后分析写成预执行设计。当日交叉报告探索及其 CPU 复核均已完成，实际结果见 [本轮报告](REPORT.md)与[完整评审回应](../review_response_20260907/RESPONSE.md)。原拟议240次角色互换训练已延期，当前新增GPU预算为0。

要回答的问题是：**用于挑模型的验证成绩中，有多少差异与说话人是否参与拟合有关，又有多少与选模后重复报告同一验证数据有关？** 这两个来源可能同时存在，还可能混入不同人物群体的难度差。直接比较验证与外测分数，不能自动把它们分开。

已有共同拟合实验提供了第一层证据。480 条正式 WavLM 训练轨迹中，两种验证规则共享拟合录音和候选训练轨迹，分别用自身验证 CE 选轮次、验证 UAR 选配置，再评价共同人物独立外测。主差分 `(V_seen−T_seen)−(V_unseen−T_unseen)` 为 **+2.706 pp，95% t CI [1.802, 3.610]**。它排除了“两条规则实际拟合成员不同”这一解释，但同时涉及验证人群和自适应选模，不能全部称为选模偏差。固定 config 3、第 15 轮时，两种规则对应同一模型，外测分数严格相同，验证差仍为 **+2.83854 pp**；这说明差距并非只在适应性选择后出现。主差减此对照也不是纯选模效应。完整数值与边界见[完成报告](../inner_validation/reports/RUN_REPORT.md)及[结果 JSON](../inner_validation/reports/results/results.json)。

**纠正此前对人物难度的描述：这不是两组始终不变的人。** 冻结随机化已使91/91名说话人跨draw进入过seen和unseen角色。事后逐人、joint(draw,fold)固定效应下，cfg3系数为2.870941 pp；先对四配置末轮平均再调整为2.899325 pp，均未消除正差。Claude的未调整2.801可复现，但FE 2.967及区间缺少完整实现，不能作为已复现结果引用。有限随机化的不平衡、人物×文本/模型/曝光交互，以及角色变化伴随整个拟合人群变化仍需限定，但不再把完全未控制的静态人物难度当新增240次训练的主要理由。详见[角色与FE复核](../review_response_20260907/dual_science_review.md)。

相关工作已经明确覆盖 SD/SI 差异、验证人物隔离，以及选模与最终评估分开。以下七个来源均核查到开放原文的相关方法部分；没有把只可见摘要的候选作为本表证据。此次补列此前遗漏、且项目已有笔记提及的 Ibrahim，不能以有界检索为遗漏直接先例辩解。

| 原始来源 | 可核查的已有覆盖 | 对本方向的限制 |
|---|---|---|
| Atmaja & Sasou，*Effect of different splitting criteria on the performance of speech emotion recognition*，TENCON 2021（预印本于 2022 上传）。[全文](https://arxiv.org/pdf/2210.14501) | 实验 3、Table III 将 SD、SI、文本独立、人物与文本独立四种条件都设为 14,400 条训练、400 条测试；Table IV 报告 WA。 | 人物／文本依赖和匹配训练条数已有先例。其训练与测试成员随条件变化，不能等同于共同拟合轨迹；WA 也不能当作本项目 UAR。 |
| Scheidwasser-Clow et al.，*SERAB: A multi-lingual benchmark for speech emotion recognition*，ICASSP 2022。[全文](https://mkegler.github.io/publication/scheidwasser-clow-2022/scheidwasser-clow-2022.pdf) | §2.1 的 train/validation/test 人物互斥；§2.2 用训练与验证数据进行分类器网格搜索，再在保留测试集评估。 | “验证集也应按人物隔离”和“调参后独立测试”已有直接 SER 先例。 |
| Antoniou et al.，*Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check Case Study with IEMOCAP*，ICASSP 2023。[全文](https://sail.usc.edu/publications/files/Antoniou-ICASSP2023.pdf) | §3.2.1 讨论人物重叠造成的评估问题；PDF 第 4 页建议 8 人训练、1 人验证、1 人测试的 10 折人物独立评估。 | 不能把验证人物隔离称为新发现。文献间结果表也不是共同训练条件下的因果差值。 |
| Kumari et al.，*Hybrid CNN-embedding fusion with MFCC-SVM for speech emotion recognition: Random vs actor-wise evaluation on CREMA-D*，PLOS One，2026-08-24。[官方全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0355238) | §4.4–4.5 比较随机录音与 actor-wise train/test 划分，使用相同处理流程，报告 5 个种子。 | 这是近期直接相关比较；重复种子和 SD/SI 对比本身不足以构成强创新。所述方法没有报告共同轨迹双验证与额外独立报告集的完整组合。 |
| Ibrahim et al.，*Multimodal emotion recognition using hybrid deep feature fusion under speaker-independent evaluation*，Scientific Reports，2026-06-25。[官方全文](https://www.nature.com/articles/s41598-026-58836-w?error=cookies_not_supported)、[表6](https://www.nature.com/articles/s41598-026-58836-w/tables/6?error=cookies_not_supported) | RAVDESS随机与LOSO、CREMA-D随机与人物互斥五折；表6含audio-only，accuracy分别70.83/39.72与54.06/47.10。 | 直接覆盖重叠与人物独立差距，不能以多模态排除。所读方法未报告本项目共同候选训练下的内层双验证与共同外测差分；accuracy不是UAR，“±”不是可自行认定的95% CI。详见[来源核查](../review_response_20260907/diag_ibrahim_source.md)。 |
| Naini et al.，*The Interspeech 2025 Challenge on Speech Emotion Recognition in Naturalistic Conditions*。[官方全文](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.pdf) | §2.1 使用人物独立训练、开发、测试；§3 的基线包括 WavLM-large 微调。 | 现代预训练编码器配合 SI 开发集已有应用。挑战允许有限重复提交，不能把其测试描述成完全没有反馈的单次盲测。 |
| Cawley & Talbot，*On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*，JMLR 2010。[官方全文](https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf) | §1、§5 解释有限验证样本上的选择过拟合，并将选模纳入完整训练程序，在外层重新选择和评估。 | 这是通用模型选择／嵌套评估先例。独立报告集本身不是新理论，也不能据此推断 SER 中曝光效应的大小。 |

因此，可争取的增量是把这些问题放入同一受控实验，给出可以核对的量化区分，而非首次发现 SD/SI 差异。**本次有界检索未在上述方法中看到完全相同的组合，不等于不存在其他先行工作，也不保证新颖性。** 面向已知用户的 SD 目标本身并非自动构成泄漏；关键是估计量是否对应声称的泛化目标。

近期探索使用已封存的第 15 轮预测，不增加训练，也不重新推理。候选配置固定为 config 0–3，不使用按原验证 CE 选择的 best checkpoint。每个原 seen/unseen 验证组的 24 人各按性别分成 12＋12 人，每半 6 女、6 男；两向交换选择半与报告半。每个方向只用选择半的 UAR 挑配置，精确并列取最低配置索引，再在未参与**本方向**选择的另一半报告。切片沿用存档预测，保留原推理 batch 上下文，避免以重新分批推理引入另一项变化。

这一 cross-report 可以描述同一选择规则的“用于选择的成绩”与“未用于该方向选择的报告成绩”相差多少，以及曝光相关差异是否仍出现在报告半。每个人在另一方向会参与选择，全部录音也已经进入原研究开发与分析；因此，这仍是旧数据上的事后探索，**不是全新盲测或新的独立确认实验**。两方向、同人物和重复划分之间存在依赖；只报告描述性结果，不新增 p 值、置信区间或因果调整。固定末轮排除了本探索中的轮次选择，但四配置选择仍然存在；半组样本量变化也可能改变选模稳定性，不能把任何差值直接解释成原主流程的纯选择偏差。

**当前先完成已有证据的解释和论文定位，不新增训练。** CPU复核已得到四配置等权的仅检查点选择外测差+0.835625 pp，事后95% t区间[0.569718,1.101532]。这是可考虑独立确认的线索；训练完整运行15轮，没有真正提前停训，现有预测也不足以重建完整逐轮测试曲线。不能从检查点截面声称第8–10轮达峰、选更晚导致更好，或把事后区间提升为原主发现。

若下一项确需GPU，先为**检查点选择规则在独立预定条件下的稳健性**写单一主对比：分别界定验证人物曝光与CE/UAR目标，不混改多项因素；固定候选数、训练预算、配对层级、最小关心效应及多重比较范围，再估算算力。不能先拿旧显著结果当保证，再为凑结果加配置、种子或语料。

原角色互换草案保留为延期的框架控制/精度检查：相同报告录音、文本和耦合随机数确有额外控制价值，但已不是首次实现角色轮换，也不能隔离逐人加入效应。跨语料若使用SUBESCO，其20人规模会限制人物泛化精度，增加种子不能补足独立人物；语言与录制条件、人物和类别混杂仍须承认。现有跨语料容量检查不是足够功效或语言因果识别的证明。

对论文而言，当前证据支持一个更具体的评估方法问题：共同训练条件下的验证人群差异，与验证分数被用于选择后的报告方式，如何共同影响性能估计。后续结论必须随实际结果调整；本方向目前不提供哪种验证策略能提高部署性能、消除差距或普适适用的未经测试建议。

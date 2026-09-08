# 当前论文定位与直接先例对照

2026-09-07。本文替代旧碰撞矩阵作为**当前定位入口**，已纳入新程序384条正式轨迹及实际验收完成后的结果。旧矩阵保留为历史记录。以下只纳入六组已查读原文的直接证据；不是系统综述，也不是全领域首次性证明。“未报告该对照”仅指所查方法和结果，不能改写为“作者未做”。完整书目信息、章节、指标及可及性限制见[原文核查](../v3/final_program_20260907/RELATED_WORK_AUDIT.md)。

当前最可辩护的定位是：**在已知说话人划分影响SER评估的基础上，用不同层次的受控比较，区分训练/验证协议带来的分数差、验证相对外测的乐观差，以及检查点选择规则对共同外测表现的影响。**新增结果在CREMA-D和SUBESCO支持限定的CE/UAR准则敏感性，RAVDESS仍不确定；这些估计对象不能合称一个已识别的“泄漏机制”，也不支持UAR选模普适优胜。[最终科学解释](final-20260907/review/FINAL_SCIENCE_INTERPRETATION.md)给出全部六项结果与推断边界。

## 六组直接证据

| 已核先例 | 原文已提供的证据 | 本稿可以具体区分的对照及其边界 |
|---|---|---|
| **Ibrahim et al., Scientific Reports 2026**；§3.5.3、§3.6、表6；[出版社表6](https://www.nature.com/articles/s41598-026-58836-w/tables/6?error=cookies_not_supported) | Audio-Only分支已比较RAVDESS/CREMA-D随机与人物独立外层协议；accuracy为70.83→39.72、54.06→47.10。纯音频落差及RAVDESS更大的落差均已有先例。 | 所读方法提验证调参，但未明确报告SI内验证的分组规则，也未报告共同训练轨迹、共同SI外测下的双验证对照。应保留其直接相关性；不能以主系统多模态将其排除，更不能把其accuracy改称本稿UAR。 |
| **Zielonka et al., Electronics 2022**；§4–7、表3；[出版社](https://www.mdpi.com/2079-9292/11/22/3831)、[作者机构原文](https://mostwiedzy.pl/pl/publication/download/1/recognition-of-emotions-in-speech-using-convolutional-neural-networks-on-different-datasets_75193.pdf) | CNN/ResNet、多语料、随机与演员不重叠准备的test accuracy比较。表3有单库及合并语料行。 | 表3不是分别报告CREMA-D和RAVDESS各自的协议落差。所读原文未报告独立inner因素、同一外测V−T或匹配训练条数的曝光替换。增量是具体控制，不是首次证明划分会影响结果。 |
| **Kumari et al., PLOS One 2026**；§3–4、§6；[出版社全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0355238) | CREMA-D上MFCC-SVM、CNN及融合SVM，共同预处理，随机/actor-wise外层比较，五种子和accuracy/UAR/macro-F1。 | 统一流程、多模型、多种子不足以单独构成本稿新意。所读主文未报告共同fit双验证、共同外测下的inner干预；补充CSV本轮未成功取回，不能宣称所有补充材料均无相关信息。 |
| **Scheidwasser-Clow et al., SERAB, ICASSP 2022**；§2.1–2.2；[作者论文](https://arxiv.org/pdf/2110.03414)、[DOI](https://doi.org/10.1109/ICASSP43922.2022.9747348) | 九库六语言，train/validation/test三分区人物互斥；冻结特征后用train/validation选择分类器，在test评价。跨任务UM是accuracy的任务平均，不是类宏召回UAR。 | 人物独立验证、多语料统一评估已有规范先例。未报告共同轨迹上seen/unseen选择规则的配对干预。本稿自定义划分用于识别特定差值，不是SERAB排行榜复现或语言因果检验。 |
| **Atmaja & Sasou, TENCON 2021**；§III、表III–IV、§IV-B；[作者全文](https://arxiv.org/html/2210.14501v1) | JTES的SD/SI/TI/STI四条件；实验3各为14,400训练、400测试，并有内部验证、重复及SE。§IV-B明确不同条件测试成员不同。 | speaker×text四格及匹配训练N已有先例。本稿曝光块在每个配对面板固定test/val录音、fit条数及选模规则，是对先例的扩展；但fit–val人物曝光关系与加权CE的类别权重并非全部相同，详见[实际控制审计](EXPOSURE_CLAIM_AUDIT.md)。其原文未报告本稿这种固定同一外测的outer×inner/V−T对照。训练组成仍改变，不是纯声纹效应。 |
| **Antoniou et al., ICASSP 2023**；§3.2–3.3、§4、表3；[机构保存的IEEE原文](https://sail.usc.edu/publications/files/Antoniou-ICASSP2023.pdf) | IEMOCAP划分、脚本/即兴、标签处理及公开模型复现；四类十折8人训练、1人验证、1人测试。 | 不能说其忽略验证或只有外测分组；协议可比性和复现审计也已有先例。所读原文未报告共同候选轨迹下改变验证人群/准则，并在同一外测上量化所选模型差异的对照。 |

## 本项目各层回答不同问题

| 证据层 | 保持与改变的内容 | 可以支持的定位 | 不能据此声称 |
|---|---|---|---|
| **原v2 outer×inner及曝光块，已完成** | 协议四格改变outer/inner边界；曝光块另固定测试交集、控制训练录音数，替换训练曝光。 | 在指定实现中量化完整协议依赖；把prompt、speaker及sibling-take的对照写清楚。 | 四格差值恒等式不是因果中介分解；匹配条数不是匹配全部训练属性；三库、六模型族本身不是充分新意。 |
| **N14R2，已完成** | GR/GG共享同一人物独立outer test；但random/grouped inner划分重新分配开发录音，**fit成员随inner改变**。同时选择检查点及配置。 | 估计两套完整训练/验证选择程序的`Δ(V−T)`，即随机inner相对分组inner的验证乐观差。 | 不能说已固定同一fit、只改变验证人群；不能把该差直接叫外测性能损失，或完全归于复用验证选模。 |
| **DUAL，已完成** | 同一fit和每配置完整训练轨迹，共享seen/unseen两套验证及outer test；两验证有相同query文本，但人物不同；各自选择检查点和配置。 | 将训练fit改变这一因素移出两验证规则的比较；在现代模型上区分`ΔV`、`ΔT`与`Δ(V−T)`。 | 不是N14R2的等价复现；验证人群难度、选择规则和所选模型差异仍须区分。正的gap差不说明seen选模外测更差。 |
| **固定last交叉报告、人物固定效应等，已完成的事后诊断** | 复用既有预测，按新问题拆分报告/选择、重加权或调整固定人物效应。 | 提供关于验证复用和人群组成的敏感性证据，帮助形成后续设计。 | 不替换原主要终点，不变成新的确认研究；固定效应调整不是纯曝光或逐人因果识别。 |
| **新final program 384，正式训练与实际验收已完成** | 360条WavLM主轨迹＝三库×24draw×5fold；每条共同fit轨迹比较seen/unseen×CE/UAR四条检查点规则，保存全部15轮预测。另24次CREMA-D B组拟合，只以固定last作整组角色互换描述。 | 共同SI外测上的`δCE`及准则交互`J=δCE−δUAR`在CREMA-D和SUBESCO获预定Holm六检验支持，RAVDESS两项不确定；完整曲线补足旧存档只有选中检查点的限制。 | 不能把两库结果写成三库一致、机制识别或UAR普适优胜；RAVDESS不确定不等于零效应或等价。跨库不是语言因果，角色互换不是逐人曝光效应；A/B query参与主研究选择，不能称全局未用于选择的独立报告集。 |

上述“共同fit”指同一主训练轨迹上的选择规则比较，不指补充A/B两个模型的fit录音相同。新程序三库保留原生类别，但人数、文本与训练N不同；稳健性是跨指定语料的概念重复，不是同预算语言比较。新384的六项检验族、24完整draw单位、描述性控制及完整性门槛见[冻结科学设计](../v3/final_program_20260907/SCIENCE_DESIGN.md)。旧稿对应依据为[原文方法与N14R2](submission-20260906/english/main.tex)，DUAL依据为[冻结协议](../v3/inner_validation/PROTOCOL.md)及[完成报告](../v3/inner_validation/reports/RUN_REPORT.md)。

新程序的[完整gate](../v3/final_program_20260907/reports/scores/complete_gate.json)、[全部主结果](../v3/final_program_20260907/reports/scores/results.json)、[独立数值复算](../v3/final_program_20260907/reports/audits/numeric_original.json)、[异路径复算](../v3/final_program_20260907/reports/audits/numeric_relocated.json)和[指定四单位恢复](../v3/final_program_20260907/reports/audits/formal_first_panel_restore_recovery_v1.json)均已完成；详细范围与原后处理失败后的接续记录见[证据归档](../v3/final_program_20260907/reports/README.md)。执行前的私有Git冻结提供本程序的内部预定记录，不是公开预注册；对这些既有语料的先前研究经验也未因此消失。

## 术语与贡献的取舍

Cawley与Talbot指出，有限样本的模型选择准则本身可以被过拟合，继而影响性能评价；这是一般模型选择偏差的先例，不是本稿新发现。[JMLR官方原文与书目信息](https://www.jmlr.org/papers/v11/cawley10a.html)：Gavin C. Cawley and Nicola L. C. Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” **JMLR 11(70):2079–2107, 2010**。

因此本稿应把三个词分开：**说话人曝光**是数据/人群边界；**选模复用**是同一有限验证信息参与选择和报告；**外测选择效果**是不同选择规则所得模型在共同外测上的表现差。`V−T`可同时受报告人群、训练组成和选择影响，不能直接全部命名为winner's curse。既有理论支持准确用词，不自动赋予本稿机制识别。

当前贡献列表应删除旧设想中的“概率抽样公共代码审计”“代码实践流行度估计”；这些不是当前论文提供的证据。代码、哈希、恢复与独立复算支撑结果可核查性，不单独充当科学新意。三库、多模型、很多拟合也是证据覆盖范围，不能代替一个此前原文未报告的明确对照及其结果。

论文现在围绕**估计对象的区分及检查点规则的受控外测比较**组织：在固定WavLM配置及本次原生任务下，两库支持说话人曝光对检查点选择效果的影响依赖CE/UAR准则，RAVDESS仍不确定。`δCE`是seen与unseen两条CE选模规则所得模型的外测UAR差，不是CE数值差或`Δ(V−T)`；`J`比较该差与UAR选模规则的对应差。全部六项预定检验均须保留，δUAR与角色控制只作描述，具体数字以[最终科学解释](final-20260907/review/FINAL_SCIENCE_INTERPRETATION.md)为准。不能写“首次证明speaker leakage”“分组选择必然更好”“CE普遍更差”“消除偏差”或“已识别声纹/语言机制”。旧N14R2、DUAL和新增程序各自保留时间、主要终点和推断单位，不拼接为一个更大的独立样本。

TTS的100次技术尝试不等于正式E3训练。正式合成分支的条件资格未满足，已按原规范关闭本轮分支且未执行；这既不是全部合成实验完成，也不是合成训练无效的证据，更不能写“AI合成解决差距”。[TTS范围审计](../v3/final_program_20260907/TTS_SCOPE_DECISION_AUDIT.md)保留了该边界。

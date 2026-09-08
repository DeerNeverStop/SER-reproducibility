# 研究与交付完成清单

本文件区分已经完成的研究、新正式批次和未满足准入的候选。状态必须由对应证据更新，不能因时间已用完、代码测试通过或论文页数有限而写成实验完成。

| 研究/交付 | 当前事实 | 本轮处理与完成依据 |
|---|---|---|
| 原v2外层/内层、匹配曝光与FT | 已有4,943完成单位及原verification | 不重复训练；新稿只引用原终点及原推断身份，不能沿用失效旧N14数字 |
| N14R2 | 24draw、1,920fits完成，原planned字段勘误和独立验收保留 | 若引用使用1.419[.901,1.937]；旧N14的确认性推断失效、N14R未完整完成，两者均不混入 |
| Study II录音预算配置 | 1,440单位完成 | 人数与每人句子覆盖同时变化；适合作为数据构建边界证据，不称纯人数因果 |
| Study II原配置导出包 | 已完成工具、真实独立17项测试与ZIP异路径新进程验收 | 原计划不重生成；720/90/90清单无损重建1,440unit；[下载与用法](../data_recipes_20260907/README.md)。无音频或模型，不冒充新实验/最优数据集 |
| E0/E1 | 既有ASV几何、弱关联、voice-only人评对齐完成 | 只作诊断，非证明声纹距离决定SER泛化 |
| E2选人策略 | 720formal+19pilot已完成，事后外测几何已独立重算 | 具体C策略CNN主效应-.109pp，不能说所有覆盖算法无效或已消除gap；C外测平均NN1/3在本次观测中反而增加；R方向另报 |
| DUAL | 480formal+4pilot已完成 | 保留预定theta2.706身份及固定模型2.839分解；+.836检查点对比为事后动机 |
| DIAG | 零新增训练、完整既有logit再分析及独立算术复核 | 只作同一材料的方法诊断，不算新的独立重复 |
| TTS技术探针 | 100尝试完成；四锚检索51/96；盲听完成0/3 | **0/3是评审未完成，不是3人都判不合格**。正式E3未获准入，未执行；本轮有限方案不继续正式合成训练。不能用AI代替人评后冒称人类情绪/身份验证通过，也不能称合成训练无效已被证实 |
| 新技术pilot | 4完成，完整工件gate、新完整账本复验、独立CUDA磁盘恢复33组全部pass | 排除科学统计；资格证据在qualification |
| 新主矩阵 | 360条正式A轨迹已完成，全部384完整gate及主评分通过 | 3原生语料×24draw×5fold，每轨迹4选择规则和全15轮外测logits；pilot不计入正式科学结果。[完整gate](reports/scores/complete_gate.json)、[主结果](reports/scores/results.json) |
| 新角色控制 | 24条正式B已完成 | CREMA每draw预定fold0；固定last整组角色互换仅作描述，不冒充旧240完整角色互换或纯逐人因果。[实际结果](reports/scores/controls.csv) |
| 六项主检验 | 六项预定检验均完成；CREMA-D与SUBESCO的δCE及J获Holm6支持，RAVDESS两项仍不确定 | 3语料×δCE/J，24draw的t推断与Holm6；δUAR及E等只描述。δCE是两条CE选模规则的外测UAR差，J是其与UAR选模规则差的交互，不能改称CE数值差或泄漏机制。[完整解释](../../paper/final-20260907/review/FINAL_SCIENCE_INTERPRETATION.md) |
| 完整数值/恢复复核 | 真实384原地、导出内部及异路径数值复算均通过；指定4UID恢复30组，最大logit差0 | 原地与异路径各核63,901个数值，最大差4.44×10⁻¹⁶；恢复覆盖预定首面板的三native heads及A/B，不是恢复全部384或重新训练。原包装器恢复阶段失败后已修复CUDA可见性并接续完成，原失败保留。[原数值审计](reports/audits/numeric_original.json)、[异路径审计](reports/audits/numeric_relocated.json)、[恢复](reports/audits/formal_first_panel_restore_recovery_v1.json)、[接续完成](reports/operations/recovery_cuda_visibility_v1/90_recovery_complete.json)、[非检查点包记录](reports/LOCAL_LARGE_ARTIFACTS.json) |
| 新颖性复核 | 已读远端新增评价与六组直接原文 | 正面补Ibrahim/Zielonka/Kumari；分组验证、多模型、多seed、匹配训练N均已有先例，不独占这些一般做法 |
| 正文取舍 | 新完整结果已形成限定的准则敏感性定位 | 保留全部六项检验及RAVDESS不确定结果；两库支持不能写成三库一致、已识别机制或UAR普适优胜；不同研究估计量和CI不合并。[最终科学解释](../../paper/final-20260907/review/FINAL_SCIENCE_INTERPRETATION.md) |
| 中英文论文 | 已按实际384结果整体重写并完成双语科学核对 | [英文5页](../../paper/final-20260907/english/xie.pdf)、[中文完整对应8页](../../paper/final-20260907/chinese/explainer.pdf)，同一未舍入事实源，准确AI披露；保留旧submission原字节。[最终交付](../../paper/final-20260907/README.md) |
| PDF与源码包 | 最终PDF、全页视觉/结构/图字号QA、源码ZIP及实际新目录重建均通过 | 英文4页技术+第5页仅参考文献；中文8页；源码ZIP解压后实际重新编译，两版共13张页面PNG与已审交付字节相同。[验收与清单](../../paper/final-20260907/DELIVERY_MANIFEST.json)。未实际投稿 |

本项目没有因这些研究证明了“构造一个能够消除SD/SI差距的数据集”。说话人曝光、目标任务、文本覆盖和评估规则需要分别界定；通过降低seen表现让差距缩小也不等于改善unseen表现。最终汇报必须说明哪些问题得到有限证据、哪些仍未解，不把没有实施的E3藏在“全部完成”里。

数据构建方向的已完成证据与具体建议见[综合判断](../../paper/DATASET_DIRECTION_SYNTHESIS_20260907.md)。[TTS范围审计](TTS_SCOPE_DECISION_AUDIT.md)追溯原方案中正式E3的条件资格：当前不仅人评未完成，稳定六情绪配方和正式音色库也未建立，因此不跳过条件启动正式合成臂。这关闭了“是否必须补正式fits”的原规范解释，不等于关闭合成研究问题本身。

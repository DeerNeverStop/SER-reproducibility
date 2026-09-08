# 最终实验与双语论文：执行入口

**完成附注，2026-09-08：** 本目录下述运行中/尚未导出/尚未生成论文等表述均为当时的准备和运行记录。正式 384 次拟合、真实预测复算包、异路径复算及论文已完成，见[完成归档](reports/README.md)与[中文研究报告](../../paper/final-20260907/RESEARCH_REPORT_中文.md)。后续另做的 720 次 AutoDL 实验见[补充结果](../../docs/autodl-supplement-20260908/RESULTS_REPORT_ZH.md)，不计入本程序的 384 次。

目标：完成剩余研究的设计、独立红队、实际执行与分析；按完整证据判断正文/附录取舍，交付符合原要求的中英文论文及可编辑源码。此入口是进度记录，不是完成声明。

本轮正式设计为 **360 条主要轨迹 + 24 条配对控制**，另4条技术pilot排除于科学统计。三库保留原生6/7/8类，每条主要轨迹同时评价seen/unseen×CE/UAR四种检查点规则。原延期240次草案由这套有明确增量的完整设计取代；没有假称已重跑原120对。

先读 [科学规范](SCIENCE_DESIGN.md)、[初始红队](RED_TEAM_INITIAL.md)、[交付要求](DELIVERY_REQUIREMENTS.md)。代码审查分别见 [引擎审查](ENGINE_REVIEW.md)、[运行器审查](RUNNER_REVIEW.md)。原有完整研究、空结果及TTS未通过正式准入的状态仍保留；不会为填满新论文而隐藏旧结果或假称合成数据有效。

2026-09-07当前状态：全15,882条原始录音字节核对通过；清洗后15,872条完整解码通过。真实元数据计划含384正式+4pilot，全部结构与精确重放通过。**四条真实GPU技术pilot已完成，完整工件门禁及独立进程恢复通过；正式384已于14:52 UTC从976297d冻结后启动，正在本机运行。** 恢复的33个epoch×报告组最大logit差为0。后续完成证据须来自phase目录的SOURCE_LOCK、DONE、完整账本、COMPLETE_GATE及独立评分；进程退出、跑完一个子集不算完整实验。见[运行状态](RUN_STATUS.md)与[完整交付清单](COMPLETION_MATRIX.md)。

完整25.55MB计划保存在 `D:/SER-final-program-20260907/inputs/plan.json`，可用 `python -m v3.final_program_20260907.plan --repo . --out <新路径>` 精确重建，身份与输入预检见[PREPARATION.json](PREPARATION.json)。大预测和检查点不进Git。旧冻结源码、原计划和数值保持不变。

本机RTX5070的四条pilot实测约92.6/36.4/36.4/17.1秒，峰值allocated均低于1.6GiB。按正式矩阵加权约5.11小时，规划留5–7小时；这是每种情况仅一条技术观察的估算，不是保证。选择本机正式执行，新增云租费0，D盘完整归档空间足够。证据见[技术资格与预算](qualification/QUALIFICATION.json)、[实际独立恢复](qualification/pilot_fresh_process_replay.json)。正式384全部封存前不计算外测科学成绩；主要检验族固定为三库×(delta_CE,J)共六项，统一Holm，另24个角色控制只作描述。

正式运行器在pilot后补强了完整账本门禁：所有reservation和尝试均须有合法顺序与明确终态；中断重试须显式记录abandoned，原失败保留。全384合成门禁反例均已修复，不以最终384个DONE掩盖额外或悬空尝试。科学计划和训练引擎与pilot保持相同字节。用户要求再次查看Git后取得的Claude新评价以原提交字节保存在[remote_reviews](remote_reviews/MANIFEST.json)；评价对象是旧c6386d0程序，并非本轮新正式结果。

此研究的预执行锁在私有Git中可追溯，不称公开预注册或从未见过这些语料。最终中英文稿将使用同一数值源与主张—证据映射；尚未生成新最终稿。

完成后的数值交付工具现已准备：[非检查点复算包](PORTABLE_BUNDLE.md)可将384套已封存预测及账本迁移后实际独立复算；它不包含音频或权重、不替代完整权重gate。本次仅合成测试通过，真实包尚未导出。[作图脚本](plot_results.py)要求完整384 gate及独立数值通过，保留六项原检验、全部15轮曲线与24个描述控制；目前仅用明确标记的合成数据检查布局，没有查看真实新曲线。

[收尾命令审计](POSTRUN_COMMAND_AUDIT.md)给出核过实际接口的顺序：直接运行score，由其内部先做唯一必需的完整权重/账本gate，再独立数值与预定首面板恢复、绘图、导出和异路径复算。训练run子命令退出并不自动完成该gate，也不能仅凭exit0判定全384完成。

[方法图](../../paper/final-20260907/figures/README.md)已按冻结规范绘制并验收，尚未读取科学结果或决定正文取舍。原数据集问题的边界见[数据构建综合判断](../../paper/DATASET_DIRECTION_SYNTHESIS_20260907.md)和[TTS条件范围](TTS_SCOPE_DECISION_AUDIT.md)。

[原配置工具包](../data_recipes_20260907/README.md)现可直接下载和重放全部已执行Study II清单，含真实17项独立测试及ZIP异路径验收；它不含音频、权重或预测，不是新384的评分工件，也不代表新实验已经完成。

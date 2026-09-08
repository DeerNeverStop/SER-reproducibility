# 后续研究的当前入口

**完成状态更新，2026-09-08：** 原 384 次正式训练及另行冻结的 720 次 AutoDL 补充训练均已完成并验收。当前稿见[最新英文论文](../paper/supplement-results-20260908/english/xie.pdf)，完整新结果见[补充实验报告](../docs/autodl-supplement-20260908/RESULTS_REPORT_ZH.md)，全项目见[实验总目录](../docs/public-release-20260908/EXPERIMENTS_ZH.md)。下段“准备/执行中、pilot 尚未完成、没有 HuBERT 或 45 轮结果”是此前时点的历史状态，已被本附注取代；原记录保留以便追溯。

2026-09-08。当前正文基于已完成的 [384 次正式拟合及验收](final_program_20260907/reports/README.md)，最新英文编辑稿见 [review-revision-20260908](../paper/review-revision-20260908/english/main.tex)。[720 次 AutoDL 补充方案](../docs/autodl-supplement-20260908/README.md)及[实现目录](autodl_supplement_20260908/)属于另一个仍在准备/执行中的阶段。云实例及 SSH 已可用，但云端 pilot 尚未完成；目前没有可写入本稿的 HuBERT 或 45 轮结果。以下三项研究及 [PR #16 复核](review_response_20260907/RESPONSE.md)保留各自历史身份；其中“240 次延期、该轮新增 GPU 为 0”描述的是当时决定，不代表当前全项目状态。

| 已完成研究 | 实际工作 | 指标与推断单位 | 当前解读 |
|---|---|---|---|
| [E2 选人](speaker_coverage/reports/RUN_REPORT.md) | 720次正式拟合；U随机、R代表性、C配额farthest-first | 逐人UAR，再91人等权；人物配对bootstrap条件于已拟合模型 | CNN主C−U为−0.109 pp，没有检出收益；池内最坏距离改善不等于外测平均接近，见[新外测几何](review_response_20260907/e2_review_notes.md)。 |
| [DUAL 双验证](inner_validation/reports/RUN_REPORT.md) | 480次正式共同候选训练 | 每折全体录音宏UAR，5折先平均再24draw；原主t检验 | 主θ2.706 pp是所选模型验证−SI外测差异，不能全部称选模偏差；固定末轮同模型差2.839。见[角色与检查点复核](review_response_20260907/dual_science_review.md)。 |
| [DIAG 交叉报告](validation_reporting/REPORT.md) | 复用DUAL固定末轮预测；零新增训练/推理 | 两方向→5折→24draw；纯描述，不新增p/CI | 用于解释选择与报告程序，非第三次独立证据。见[精确加权及噪声假设复核](review_response_20260907/diag_review.md)。 |

同一语料、说话人、面板和训练程序的复用意味着这些历史结果不能被计为彼此独立的三次重复。不同UAR加权与推断层级也不能直接合并。PR #16 阶段的探索没有追认统一的确认性检验族；原主结果、次要结果、事后区间及描述量分别保留身份，没有等效性或排除阈值结论。另见[旧 15 轮轨迹的 W=8/10/12/15 完整诊断](autodl_supplement_20260908/diagnostics/RECONSTRUCTION_REPORT.md)：它复现原 W15 结果，但不能替代新 45 轮训练，不新增检验或独立样本。

E2、DUAL 和已完成的 384 程序采用私有仓库中的预执行冻结，不等于公开预注册。上述 CPU 复核没有重新审核全部权重或独立重训；代码重算吻合不替代研究设计审查。旧稿状态保留在[paper/README.md](../paper/README.md)，当前英文修订及限制见[变更说明](../paper/review-revision-20260908/CHANGELOG.md)，不把历史 N14 或事后发现误列为当前确认性结果。

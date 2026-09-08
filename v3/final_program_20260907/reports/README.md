# 正式 384 单元：机械后处理交付归档

本目录按原字节保存完成的轻量产物；没有重新训练、评分或运行验收。原始证据中的本地路径、时间戳和状态字段均保留。文件大小、SHA-256 及复制来源见 [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。清单覆盖本次归档，不覆盖后续另增的论文或说明。

| 目录 | 内容 |
|---|---|
| [scores](scores/results.json) | 7 件：主评分 JSON、完整 gate、5 个 CSV；384 正式拟合，其中 360 个主单元、24 个对照单元，72 个语料×draw 汇总及 6 项预定检验；pilot 不进入正式评分 |
| [audits](audits/numeric_original.json) | 原路径数值复算、4 单元 checkpoint 恢复、迁移后数值复算，各 1 件 |
| [paper_facts](paper_facts/facts.json) | 共享事实 JSON 与中英文表格，共 3 件 |
| [figures](figures/FIGURE_MANIFEST.json) | 3 张图的 PDF/PNG/SVG 与来源清单，共 10 件 |
| [operations/original_pipeline](operations/original_pipeline/99_failed.json) | 原始启动合同、阶段事件及日志，包括第一次恢复阶段失败 |
| [operations/recovery_cuda_visibility_v1](operations/recovery_cuda_visibility_v1/90_recovery_complete.json) | 环境问题诊断、修复后接续的事件及日志 |
| [operations/training](operations/training/OPERATIONS_AFTER_TRAINING.md) | 全部 384 收据的运行汇总、JSON 与只读复算脚本 |

原路径和迁移后的独立数值复算各比较 63,901 个数值，均无不匹配，最大绝对差为 `4.440892098500626e-16`。独立聚合没有调用主评分器的统计或 engine metrics；来源和计划门禁仍共享冻结运行器。这是保存预测的数值复算，不是独立训练复现。

ZIP 导出器也实际执行了一次完整 384 单元的独立数值复算及账本约束检查，之后封装小工件。解包器检查精确成员、SHA 和深层身份，再在新的目录中真正执行数值复算；其显式路径迁移例外仅处理原 `run_dir` 元数据，不改封存结果或放宽数值身份。ZIP 不含原始音频、基座或 checkpoint；其中保存的 full gate 绑定历史权重 SHA，数值复算不会重新读取这些缺省权重。

[恢复记录](audits/formal_first_panel_restore_recovery_v1.json) 只覆盖预定首 panel 的 CREMA-D A/B、SUBESCO A、RAVDESS A 四个单元，恢复其全部已保存 winner/last 状态并检查 A/B/outer 预测：30 个去重 epoch×group 比较的最大差为 0。它使用共享模型结构、预处理和预测代码，不等于另外 380 单元的独立进程恢复，也不等于新训练。

原流水线的 full gate、主评分和原路径数值复算先已通过，随后恢复阶段因进程内 `CUDA_VISIBLE_DEVICES` 成为空字符串而退出。诊断及接续记录保留了这一失败；移除该环境项使其成为 UNSET 后，恢复、制图、事实表、ZIP 导出与迁移复算均完成。未重跑训练。训练运维汇总是恢复前的状态快照，其中“30_restore 失败”没有被事后改写；最终机械状态应结合新的恢复完成事件阅读。

原冻结科学源码提交为 `976297d824ea98816bdc0948befd704e612a54c1`。本次交付归档及之后的提交不是该预执行冻结；私有 Git 冻结也不是公开预注册。

59,897,568 字节的预测复算 ZIP 保留在本地，未放入 Git；位置、SHA 和包内证明范围见 [LOCAL_LARGE_ARTIFACTS.json](LOCAL_LARGE_ARTIFACTS.json)。后续是否上传为 release 附件由主任务处理，本目录不声称附件已经发布。完整 D 盘原始归档仍需保留，以支持权重恢复与原始证据核查。这里是阅读与归档入口；原证据的绝对路径不会因复制而重写，迁移执行应使用已验收的可移植包。

最终科学解释、中英论文及人工图形验收由主任务完成；本次复制不新增科学判断，也不自动宣称用户的完整目标已经完成。

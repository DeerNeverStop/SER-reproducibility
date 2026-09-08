# 说话人泛化研究：交给 Claude 的方案审阅包

2026-09-06，草案 v2。这里是新实验的设计、自审与预算，**没有本轮新实验结果，也不是已冻结的执行计划**。

建议按以下顺序阅读：

1. [Claude 审核任务](CLAUDE_REVIEW_REQUEST.md)：需要质疑的问题、预期反馈格式。
2. [自审结论和修订](SELF_REVIEW.md)：已经改正什么，哪些风险仍待实测。
3. [研究计划](research_plan.md)：研究价值、对照、人力、算力和时间。
4. [统一执行约束](execution_contract.md)：角色、选样、统计量与进入条件；旧的两份技术笔记不属于本包的有效规范。
5. [预算表](budget_table.md)、[AutoDL 比较](cost_comparison_autodl.md)、[历史运行证据](evidence/runtime_evidence.md)。

推荐主线是 E0/E1 测量与已有预测诊断、E2 真实训练数据选样。E3 合成干预须先证明情绪控制和音色稳定性；G 缩差检验及 E4 真人双语采集是可选扩展。

| 状态 | 内容 |
|---|---|
| 已有证据 | Study II 结果、旧数据容量、旧训练计时；通过固定 commit 链接追溯 |
| 本次完成 | 文档自审与修改、预算复算、可移植的审阅材料 |
| 未完成 | 新声纹、新角色/面板求解、模型质量/显存/RTF 小试、正式训练、听评 |
| 不能据此宣称 | 已证明创新、已消除泛化差距、达到统计功效、确定总账单 |

本包只在 `docs/research-plan-20260906/` 下新增文件，不修改既有论文、训练代码或历史实验结果。历史运行的选择性汇总不替代完整原始数据重放，原始音频、模型权重、云端环境收据不随本包上传。

预算可用 Python 3.10+ 标准库离线重算，在本目录运行：

```bash
python budget_plan.py
python validate_review_packet.py
```

第一条只重写本目录预算 JSON/Markdown。第二条只读检查文件清单、SHA256、相对链接与数量/算术；**这些检查不认证科学有效性或运行就绪**。文件版本见 [manifest.json](manifest.json)，来源见 [evidence/sources.json](evidence/sources.json)。

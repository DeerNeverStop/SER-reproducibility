# SER 后续数据配置研究策划案

2026-09-04，可行性复核 v0.2。研究方向：在固定训练录音预算下，为陌生说话人情绪识别选择人数、句子和真实重复录音的配置。

- [简要策划案：给作者](BRIEF_ZH.md)
- [详细策划案：给 Claude 与实施者](PLAN_FOR_CLAUDE_ZH.md)
- [可行性评估：价值、风险、算力与已验证范围](FEASIBILITY_REVIEW_ZH.md)
- [元数据检查代码](feasibility_check.py) 与 [确定性验证快照](feasibility_snapshot.json)

状态：已完成方案复核和 540 组元数据面板构造检查。建议器、正式规划器及新模型实验尚待实现；未产生新增识别率结论。

在仓库根目录复现检查，需 Python 3.10+，仅使用标准库：

```shell
python docs/data-design/feasibility_check.py --check docs/data-design/feasibility_snapshot.json
```

交接时先阅读可行性评估，再按详细策划案 P0/P1 推进。保持旧 `v2/`、`v2_1/n14r/` 的冻结研究与运行目录不变。

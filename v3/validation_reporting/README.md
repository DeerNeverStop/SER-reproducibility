# Validation reporting diagnostic

中文入口：[完成报告](REPORT.md)、[研究定位与原始文献](RESEARCH_DIRECTION.md)、[下一实验草案](NEXT_EXPERIMENT.md)、[语料容量](DATASET_FEASIBILITY.md)。

这是已知旧结果后的探索，只分析 `SER26-DUAL-VALIDATION-1` 的固定第15轮存档预测。不运行 GPU，不改旧实验输出。[SPEC.md](SPEC.md) 固定本轮拆分、选模、聚合和解释边界；`evidence/` 是新分析目录的字节副本，报告不是事前注册。

在 repo 根目录，用现有包含 NumPy 的 Python：

```powershell
python -m unittest v3.validation_reporting.test_core v3.validation_reporting.test_run -v
python -m v3.validation_reporting.run prepare --repo . --archive D:/SER-dual-validation-20260906 --out D:/SER-validation-report-diagnostic-replay
python -m v3.validation_reporting.run analyze --repo . --archive D:/SER-dual-validation-20260906 --out D:/SER-validation-report-diagnostic-replay
```

输出目录必须尚不存在，不能复用或覆盖已经完成的目录。复算需要原 480 个 NPZ、receipt、DONE、完整计划、ledger、已完成完整性检查及旧结果；Git 中不重复上传模型权重与预测。源码/metadata 会同时对旧冻结计划校验。新生成计划包含时间，哈希随时间变化，但固定盐产生的分半和最终数字应一致。

本轮正式计划文件 `evidence/plan.json` 绑定当时的 SPEC 与执行源码字节。`analysis/FILE_SHA256.json` 绑定分析输出，`evidence/FILE_SHA256.json` 绑定所复制的完整新证据文件。独立 CSV 复算放在 `review/`，不会改变已绑定的源码或分析目录。

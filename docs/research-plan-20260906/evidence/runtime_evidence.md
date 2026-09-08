# 运行时间证据与解释边界

日期：2026-09-06。机器可读的选择性汇总见 [runtime_summary.json](runtime_summary.json)。这是历史记录与规划算术的摘要，新研究尚未训练、提取声纹或生成音频。完整原始收据、模型与大归档未全部装入此评阅包。

| 历史运行 | 可核查的实测摘要 | 预算用途 |
|---|---|---|
| Study II | 已完成 720 CNN + 720 CPU Ridge。RTX5090、B576 CNN n=360：均值 12.831 秒/fit、P90 18.724 秒；Ridge 均值 0.1147 秒 | 新 coverage 的相邻配置参考，不能把既有 1,440 单元重新收费 |
| DEPLOY2 | 已完成 540 CPU + 120 GPU 单元。RTX5090 CREMA-D partial FT n=10：均值 65.574 秒、P90 84.124 秒；实际 4–9 epochs | 证明旧管线可运行；不是新配置固定 15 epochs 的实测 |
| N14R2 | 1,920 fits，4 张 RTX4090、每卡 8 worker。worker elapsed 和 101.4667 h；各卡任务跨度和 12.7952 h；所有卡首末跨度 3.41895 h | 不得用 101.47 h 直接乘单卡租价 |

`gpu_seconds` 的实现是 `time.perf_counter()` 记录的进程 elapsed，包含 CPU 等待、验证和预测，不是 GPU 利用率积分。各卡 first-to-last unit 跨度仍不包括全部准备、搬运、备份和停机尾部，不能冒称完整账单。P90 是已观测样本的经验分位数，不是未来耗时的置信上界。

新 E2 的 CNN 12–30 秒、FT 75–180 秒，以及 E3 的生成/QA/特征 RTF、合格率和固定满程训练耗时均是规划假设。RTF 应按指定管线的 elapsed 除以输出音频时长测量；同卡型、不同实例仍可能不同速。独立 FT 峰值显存尚无足够历史证据。新生成音频的 WavLM 特征成本已在[预算](../budget_table.md)中单列，CPU Ridge 不等于首次特征提取免费。

已有原始音频约 4.02 GB、16 套特征及元数据约 1.569 GB、三份 SSL 基座权重约 1.133 GB，可按原输入身份复用。N14R2 的外部 log-mel 缓存曾经内容哈希核对通过。新参考、合成数据及其特征仍需准备；复用资产不代表新科学问题已得到验证。

## 固定科学来源

- Study II：[完整执行收据](https://github.com/DeerNeverStop/SER/blob/55b585efe9ea898f8c1717162cefef1ffda8cd65/v3/data_design/evidence/core_execution_20260905.json)、[计时实现](https://github.com/DeerNeverStop/SER/blob/55b585efe9ea898f8c1717162cefef1ffda8cd65/v3/data_design/core_run.py)。B 分层统计由全部 unit 与冻结 plan 连接后重算。
- DEPLOY2：[完成说明](https://github.com/DeerNeverStop/SER/blob/55b585efe9ea898f8c1717162cefef1ffda8cd65/v3/deploy/evidence/EXECUTION_STATUS_ZH.md)、[先导计时审计](https://github.com/DeerNeverStop/SER/blob/55b585efe9ea898f8c1717162cefef1ffda8cd65/v3/deploy/evidence/pilot_timing_audit.json)、[训练实现](https://github.com/DeerNeverStop/SER/blob/55b585efe9ea898f8c1717162cefef1ffda8cd65/v3/deploy/engines_deploy.py)。分组统计来自全部 120 GPU unit；旧 B1 原冻结 FAIL 与补充 PASS 均保留，不能由运行完成推断所有科学验证通过。
- v2：[训练 timer 定义](https://github.com/DeerNeverStop/SER/blob/55b585efe9ea898f8c1717162cefef1ffda8cd65/v2/ser_v2/train.py)、[历史并行队列记录](https://github.com/DeerNeverStop/SER/blob/55b585efe9ea898f8c1717162cefef1ffda8cd65/v2/RUN_RECORD_main.md)。
- N14R2：[冻结 PINS](https://github.com/DeerNeverStop/SER/blob/4fe0bb52f15b56a6626363be31a378b0f9661293/v2_1/n14r2/PINS.json)。另行保管的 `n14r2-closed-run.tar` 在 release 清单中记载 SHA256 `162e209301252397c6d7bf79ae8c3a781fca93f8ef7dbe2babace94d4f44f9e9`。本包没有复制该大归档，也没有执行完整原始证据重放。

摘要只保留科学配置、统计量和来源定位；固定链接不代表本次重新联网检查其内容。文件完整性和预算算术校验不认证论文设计、统计功效或任何尚未运行的实验结果。

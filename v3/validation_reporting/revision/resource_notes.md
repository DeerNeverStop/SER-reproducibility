# 历史 RTX 5090 运行性能核查

2026-09-07；仅核现有本地记录，没有 GPU/API 调用、价格查询、新训练或科学打分。可复算数据见 [resource_history.json](resource_history.json)，重算程序见 [resource_recompute.py](resource_recompute.py)。程序核对 480 正式＋4 pilot 的小型收据、history、DONE、账本及已有完整 gate 的绑定；不读取音频、模型权重或预测数组，大文件只检查大小。JSON 保留逐次用时、每轮更新尝试/AMP 跳过、60 批传输及 1,588 个小型输入文件 SHA；没有复制 loss 数值。

## 实测用时及下一固定配置

| RTX 5090 记录 | n | 最小 | 中位数 | p90 | 最大 | 累计 |
|---|---:|---:|---:|---:|---:|---:|
| 旧双验证全部正式 fit_seconds | 480 | 29.182 s | 34.195 s | 37.010 s | 48.105 s | 16,505.549 s |
| 其中 cfg3 fit_seconds | 120 | 29.870 s | 34.044 s | 36.299 s | 48.105 s | 4,098.941 s |
| cfg3 账本 start→done | 120 | 30.148 s | 34.375 s | 36.710 s | 48.398 s | 4,138.816 s |
| 更早 coverage 实验的正式 FT | 180 | 21.912 s | 23.487 s | 24.224 s | 26.677 s | 4,239.255 s |

p90 使用排序后 `(n−1)×0.9` 的线性插值。cfg3 为 encoder LR `5e-5`、head LR `1e-3`、weight decay `0.01`，top4＋head，batch 16，完整 15 轮，训练 AMP float16，参数保存 FP32，crop 3 s／eval cap 10 s。历史环境统一为 Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 12.8、NumPy 2.1.2，PyTorch CPU threads=1，matmul TF32 关闭、cuDNN TF32 开启。

**fit_seconds 不是纯训练耗时。** 冻结 runner 的 perf_counter 包围 CUDA cache/reset 与整个引擎调用，包含模型加载/哈希、尚未缓存的音频预处理、15 轮训练与每轮双验证、最终选中/末轮状态的预测、检查点写盘/fsync、全新模型回载及预测重放。它不含返回后的 predictions/history/receipt 序列化、DONE 封存和 runner 后续核验。账本 start→done 另外覆盖这部分单元包装；正式 480 次的账本区间累计 16,657.079 s，比 fit 累计多 151.530 s。它仍不含共享初始 gate 或批次间等待。

**没有逐轮时间或 train/eval/save 分项计时。** history 只记录 epoch、loss、optimizer_steps 和 scaler_skipped_steps，不能从中逆推每轮验证花多少秒。cfg3 共 64,800 次 minibatch 更新尝试，AMP 记录跳过 27 次；正式全体为 259,200 次尝试、134 次跳过。coverage FT 采用不同验证、预测与保存流程，只能辅助说明现代 FT 在这张卡上可以很短，不能用两个历史均值之差充当严格验证消融的节省量。

## 显存、CPU 与主机 RAM

- 正式 480 次最大 **allocated** 为 2,030,068,736 B（1.891 GiB）；cfg3 最大为 2,028,756,992 B（1.889 GiB）。这是每次 reset 后的 `torch.cuda.max_memory_allocated`，不是 reserved 或整进程/整卡峰值。
- reserved 峰值、CPU 型号/分配核数、进程 RAM 峰值及主机总 RAM **没有在这些运行收据中记录**。threads=1 是软件线程设置，不是主机只有一个 CPU 核。
- 另有一个早期瞬时 nvidia-smi 快照：利用率 9%、已用 1,070 MiB、总量 32,607 MiB；它不是峰值，也不能补出整个训练的显存/利用率曲线。
- **24 GB 卡是本方案合理的容量候选**：同 batch 16、crop/eval 上限、top4＋head 下，已测 allocated 留有很大余量；现有证据不支持必须租 32/80 GB。与此同时，没有跨卡实测或 reserved 峰值，不能把“24 GB 完整执行已通过”写成事实。应以新引擎固定技术 pilot 核验实际显存，不能为显存悄改 batch。

## 传输、恢复和租用区间

旧双验证 60 个正式批次使用每批四条并行 SSH tar 流，合计传输 **124,106,209,280 B**，传输计时累计 **6,202.448 s（1.723 h）**；每批中位数 98.496 s，最短 71.744 s、最长 164.555 s，按总字节/总计时加权为 **19.082 MiB/s**。原流程让上一批传输与下一批训练重叠，这 1.723 h 不能完整再加到训练或租用总时长；它也不包含最初环境、基座和数据上传。

最初四个正式模型的额外独立进程恢复检查：总 wall **24.661 s**，其中 forward **8.849 s**、waveform prepare **1.505 s**，覆盖 36 个 checkpoint-role 比较；这些是小范围恢复实测，不能当全 480 模型回载成本。更早 coverage 的九模型恢复为总 wall 12.961 s、forward 1.998 s，也不应混算成双验证恢复耗时。

双验证实际租用从 `2026-09-07T00:08:39.625Z` 到确认 EXITED 的 `05:46:34Z`，**20,274.375 s＝5 h 37 min 54.375 s**。正式第一个 unit_start 到最后一个 unit_done 为 **18,460.194 s＝5.128 h**，与 fit 累计 4.585 h 的差值还含批次间 gate、调度、传输等待等，不能统称 GPU 空闲。更早 coverage 租期为 `2026-09-06T19:43:17.616Z` 至 `23:29:11Z`，13,553.384 s；该租期还承担过其他准备/诊断，不能当 180 FT 的独立租用时间。本次不重报账单或报价。

## 下一轮 240 fits／120 pairs 的容量与时间计划

这一规模与新引擎尚未实测。保持 576 fit clips、batch 16、15 轮时，240 fits 按旧 cfg3 的均值、p90、历史最大值分别机械外推为 **2.277 h、2.420 h、3.207 h 的 fit 区间**；这是三个历史比例情景，不是概率上界或租期保证。新方案固定第 15 轮、只做末轮 query，移除中间双验证和额外检查点，应减少工作，但当前没有可据以量化节省的分项计时。

**建议按 3–4 小时安排，6 小时作为故障及异常延迟的停止上限；不是固定购买 6 小时。** 正式启动前用新程序的固定技术配对 pilot 实测速度、allocated/reserved、RAM 和末轮保存/回载/传输时间，再确认预算；不查看科学效果决定是否继续。硬件单价与不同卡吞吐由另一次独立询价/基准评估决定，本核查不据显卡名称猜速度。

存储可以比“旧 125 GB 减半”明显小：旧正式结果中，**29 个仅保存一个 unique epoch 的 checkpoint 都是 113,459,049 B**。按相同 FP32 差分格式，240 个末轮 checkpoint 为 **27,230,171,760 B＝27.230 GB，约 25.360 GiB**，另加少量预测、收据和审计文件。旧 cfg3 的 120 fits 有 71 个双状态、49 个三状态；照旧扩大到 240 fits 会有 65.577 GB checkpoint。27.230 GB 是基于旧单状态序列化的体积估算，不含基座、音频/缓存、传输临时文件；若新格式增加优化器状态或改变架构，必须重新测量。保留完整 FP32 语义，不用降精度凑存储。

复算命令（输出必须不存在）：

```powershell
Set-Location -LiteralPath 'C:/Users/jock8/Documents/ChatGPT/论文/SER-speaker-execution-20260906'
& 'E:/科研/SER/ser_gpu/Scripts/python.exe' -B `
  'v3/validation_reporting/revision/resource_recompute.py' `
  --repo . `
  --dual-root 'D:/SER-dual-validation-20260906' `
  --coverage-root 'D:/SER-speaker-coverage-20260906' `
  --out 'D:/SER-role-swap-design-20260907/resource_history_recheck.json'
```

元数据检查实际用 Windows 本地 Python 3.13.9 执行，退出 0；没有使用训练环境进行新的性能测量。

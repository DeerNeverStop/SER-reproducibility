# 新执行引擎的独立工程审查

2026-09-07。只读审查 [engine.py](engine.py)，独立新增 [test_engine.py](test_engine.py)。没有修改引擎、计划或旧冻结代码，没有读取真实研究预测，没有启动 GPU、下载模型或运行真实 WavLM 训练。

最终检查的引擎 SHA256：`c5d2618884c2c67b8e5c932ee4506634a6f101e8f8bfed3f4675123448fe90f6`；所引用计划模块 SHA256：`7c4b65113e1c2f9a43606d654814ffa101e1276a21f35a123e9e29d7e1ba1ba0`。本记录覆盖该字节版本；后续修改不能自动继承此结论。测试按最新合同运行：A 臂允许四条规则选择且保存第 1–15 轮 logits；B 臂完成 15 轮、只保存末轮，不进行任何检查点选择。正式 384 与 pilot 4 的全局计划数量由计划/运行验收器负责，这组 toy 测试不证明正式单元已经存在或完成。

## 发现与处置

**独立发现一项必须修正的问题，已由主任务修复并验证。** 原 `waveform_batch` 只裁剪长训练音频；短音频直接进入 `pad_sequence`，导致全短 batch 的输入宽度取该 batch 最长音频，而非固定三秒。这样不同训练人群的短音频长度可以改变有效输入宽度，违背当前裁剪合同，也偏离继承实现的固定补零行为。当前短 crop 分支补零至 crop；测试以 2/17 个样本的全短 batch 要求输出严格为 `(2, 48000)`，同时核对补零值和原缓存未变。

随后主任务收到另一项交叉审查发现：原引擎 `crop_uniform` 的字符串键/53 位映射与计划 `crop_start` 的 JSON 键/64 位整数映射并非同一合同。主任务在冻结前删除前者，统一调用计划 `crop_start`；本测试已随之更新，并以独立 JSON 编码和整数有理数重算验证实际采用的新合同。没有 GPU 在上述两个修正前执行；本记录不将其他审查者的发现冒称本组首次发现。

最终版本未发现另一个需要阻止技术 pilot 的引擎问题。这个判断限于下列 CPU 合成证据和源码核对，不等于整个实验、云端环境、统计设计或正式结果已验收。

## 实际通过的检查

| 检查 | 独立证据与结论 |
|---|---|
| CE 与 UAR | 6/7/8 类分别构造类别支持不等的 logits；CE 与 PyTorch float64 全样本交叉熵相符，UAR 与独立整数混淆矩阵/有理数宏召回严格一致。该 fixture 的 UAR 与整体 accuracy 不同，能发现误用 accuracy。大共同偏移不会使普通范围 CE 溢出。 |
| 并列与四规则 | 实际运行 15 轮小模型，通过替身验证量制造 CE 与 UAR 在不同轮次的精确并列。四条规则分别选第 2/4/5/3 轮，保留最早并列，末轮固定第 15 轮；验证保存状态、去重 epoch 集合、重建后 logits 均吻合。普通 logits 并列选最小类索引，有理数 UAR 不引入类别求和浮点破坏并列。 |
| A/B/outer 独立批次 | 保留完整 batch16 加末批的实际 `predict_group` 和变长 padding；将全部 outer 波形改成大幅不同长度/幅值，同时循环改 outer 标签。两次 toy 拟合的 A/B 全轮 logits、完整 history、selected epochs、初始 SHA、保存的每个 tensor 都完全相同，而 outer logits/labels 确实改变。调用记录始终等于各自冻结角色批次，角色串入、顺序改变会拒绝。 |
| slot 裁剪 | 对 15 轮 × 100 slot × 7 种长度，以独立 canonical JSON/SHA256 和 `Fraction(h64,2**64)` 重算全部整数偏移；隐含共享位置严格 `0 ≤ u < 1`，同键不同长度只改变合法裁剪区间，访问顺序不改变结果。确认引擎直接引用计划 `crop_start`，短波形返回起点 0、固定补零三秒。此项检验的是位置 slot 机制，跨臂人物/情绪/文本 slot 的语义配对仍由计划验收保证。 |
| 每轮 train 与随机数 | 小模型含 Dropout，逐次记录训练/梯度状态；A 臂每轮 eval 后均重新进入 train。纯推理前后 Python、NumPy 全局与 Torch CPU RNG 均未改变。另向真实 toy 执行路径注入 Python/Torch 随机消耗，均在首次评估后拒绝，且没有产出 checkpoint/predictions。 |
| B 臂限制 | 将 `metrics` 替换成一调用即报错，B 仍完成 15 轮并通过回载；其 selected epochs 仅 `last:15`，NPZ `epochs=[15]`、logits 时间轴长 1，history 没有验证分数。错误 selection_enabled/prediction_epochs 会拒绝。 |
| outer 只存不评 | A 两次 toy 共 60 次 metrics 调用，严格等于每次 15 轮 × A/B；没有 outer 指标调用。NPZ 仅含 epochs、三个角色 paths/labels/all_epoch_logits；history 仅训练字段和四条验证规则，无 outer UAR/CE/accuracy。`outer_scores_computed=false` 与实际调用路径一致。 |
| 原生类别头与冻结范围 | 用本地极小 encoder 替身执行实际 `build_model` 结构代码，分别验证 6/7/8 输出维度、只解冻顶部四层与分类头、同 seed 初始状态和 SHA 一致；不支持的类别数拒绝。替身不下载权重，不证明真实基座字节或实际 WavLM 输出已验收。 |
| 拒绝无效输入 | 缺少验证类别、NaN、越界标签、fit/query 文本重叠、报告人物泄漏、角色批次混入和顺序改变均拒绝。 |

CE/UAR 的改善判定目前内嵌在循环中，以严格 `<`/`>` 保留最早轮次，实际 toy 执行已覆盖；没有必要仅为了测试再抽函数。若后续抽出选择函数，必须保留有理数 UAR 比较，不能先转 float 再决定并列。

## 运行记录与范围

执行环境：本地 `E:/科研/SER/ser_gpu/Scripts/python.exe`，CPU，单线程；CUDA 不可见、禁止写 bytecode，关闭 pytest cache。最后一轮 **26 passed in 3.64s**，命令退出码 0。

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'E:/科研/SER/ser_gpu/Scripts/python.exe' -m pytest -q -p no:cacheprovider v3/final_program_20260907/test_engine.py
```

这些测试没有核验 CUDA FP16/GradScaler、真实 torchaudio WavLM、本机与云端数值一致性或真实训练吞吐，应由预定技术 pilot/独立恢复覆盖。`rng_state` 的运行时保护覆盖 Python 与 Torch CPU/当前 CUDA RNG；训练顺序使用局部 `RandomState`、裁剪使用无状态哈希。当前推理没有消费 NumPy 全局随机数，测试额外检查了这一点，但运行时 guard 本身没有声称检查任意缓存内部 RNG。

独立角色 batching 排除了 outer 波形参与 A/B padding 的路径，不代表消除每个角色自身的 padding 上下文。训练计数、配额、完整 source/plan 身份、原音频字节、DONE/账本/收据闭合及正式 384/pilot 4 数量不由这个引擎单独证明；应继续由主任务的冻结计划与完整结果门槛承担。不得把本记录或 toy checkpoint 当成真实研究完成/科学成绩证据。

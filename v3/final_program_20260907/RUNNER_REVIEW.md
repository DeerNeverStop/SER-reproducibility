# 新运行器执行前红队

2026-09-07。审查 `run.py` 与同目录 `plan.py`、`engine.py`；只读源代码，并运行临时目录中的纯 CPU 合成样例与元数据规划，未启动 pilot/GPU、未读取新科学成绩、未改运行器。

**修订结论：R1–R6 的所列实际漏洞已由根节点修复，独立新增 `test_run.py` 的 31 项 CPU 回归全部通过（2.23 秒）。当前未发现阻止固定四个技术 pilot 的剩余运行器缺陷。** 这不替代实际部署的源／输入／模型门禁和 pilot 自身恢复验收。新 `audit_phase` 已落盘；正式执行前还须单独验收全 384 集合与 ledger 门禁，不能把这 31 项单位测试说成全块验收已完成。

- 本次修复后复验 `run.py` SHA256：`9ca940c616aa1ef4f4116d7e77892c98f87d1aaa57873571bf1c2c2b6a5cdda6`。
- 本次 `test_run.py` SHA256：`17cd1449f2d2c666e8a51883c07c3cd09566abcd3055e8bf818d12f5278cc21f`。
- 环境：本机 `E:/科研/SER/ser_gpu/Scripts/python.exe -B -m pytest -q -p no:cacheprovider v3/final_program_20260907/test_run.py`；仅临时小型 CPU 张量／NPZ／JSON，未加载大 WavLM。

已确认的修复为：freeze/checked_plan 调用完整 planner replay；实际普通模块的路径与 namespace package 的每项路径均精确对应所选 repo；逐实际引用 wav 检查 bytes/SHA；四 artifact 严格限定到同一 attempt；DONE 前完整读取并验 NPZ/history/小型 checkpoint 结构；先持久 reservation 后 start；仅有唯一同 attempt start、正确 reservation 且无同 attempt failed 才恢复缺失 done；提交后异常不会写 unit_failed；裁剪直接调用 planner 的精确整数接口。下面保留问题的**修复前实证和修复要求**，用作回归测试的依据，不代表它们仍未修复。

- `run.py` 被实证审查的 SHA256：`775d2b04705ca21bba73ffecd9b422c16a2dd1f8bf176bbeef1424d33572e7c1`。
- 元数据规划复核时 `plan.py` SHA256：`7c4b65113e1c2f9a43606d654814ffa101e1276a21f35a123e9e29d7e1ba1ba0`。

## 1. 已修复问题的原始实证与处理要求

### R1：计划自洽哈希不等于规定实验的计划

`freeze()`、`checked_plan()` 只检查计划哈希、SOURCE_LOCK 与现存文件自洽，没有调用本程序 `plan.validate(plan, repo)`。自洽但改变角色、配置、单位总数或分析约定的计划可被锁定；engine 的逐单位验证不能替代完整 360＋24＋4 清单及确定性重建。`phase_units` 也只来自该计划自身。

修复：在首次冻结及每次执行时调用新 planner 的 metadata 校验与完整 semantic replay；必须验证实际加载的 runner、planner、engine 和关键继承模块 `__file__` 都来自 `--repo`，并将这些实际文件与 source lock 对照。仅 hash `--repo` 磁盘上的另一个 checkout 不能证明进程执行的就是该副本。运行完成再检查源／计划未变；语料根目录与模型位置可以按部署环境改变，内容身份不能改变。

### R2：运行输入音频未绑定到计划中的字节

`execute()` 从 `--roots` 创建 `WaveCache`，只调用 `validate_unit()` 验证内嵌 metadata 与角色，没有按 `rows[path].sha256/bytes` 检查实际根目录中的 wav。把 roots 指到具有同名路径、不同声音或标签内容的目录仍会训练，并写出自洽 DONE。先前本机 inventory 的通过不能自动证明上传后的云目录内容相同。

修复：首次推理/训练前验证本次所需音频并记录 input gate，至少覆盖拟合与 A/B/outer 全部引用路径的路径安全、字节 SHA、长度与解码。运行时读取的路径必须绑定到该 gate；不能只信一份来自另一根目录的历史收据。模型当前已有文件 SHA 固定检查，engine 另核编码器 state SHA，这部分方向正确。

### R3：DONE 的哈希文件与实际读取的 receipt 可来自不同 attempt

`verify_done()` 只检查 artifact 的 basename 为四个固定名称；随后另外按 `DONE.attempt/receipt.json` 读取语义，并不要求该 receipt 就是 artifacts 中被 hash 的那一份。

**CPU 实证：**四个 artifact SHA 都绑定 `attempts/0001`；将 DONE.attempt 改为 `attempts/0002`，在 0002 写另一份未被 hash 的有效 receipt，即可通过。0001 中真正被 hash 的 receipt 可以是无关 JSON。临时 fixture 已随上下文退出清除。

修复：严格校验 attempt 为本 unit 下合法的固定格式相对路径；artifact key 集合必须恰好是该 attempt 的四个文件，不允许跨 attempt；读取 receipt/history/NPZ 的路径必须来自这一精确集合。DONE、receipt 的 schema/unit/phase/lock/attempt 必须相互一致。

### R4：当前 `verify_done` 并未验证产物的科学结构或训练完成

**CPU 实证：**将 checkpoint 内容设为 `not a checkpoint`，predictions 内容设为 `not an npz`，history 设为 `[]`；receipt 填 `epochs=0`、`wall_seconds=-5`、`reload_max_abs_diff=-1`。只要重新写入对应四文件 SHA，当前 `verify_done()` 全部接受。现有 `reload_max_abs_diff <= 1e-5` 没有非负性校验，且仅信任自述 `outer_scores_computed=False`。

修复：跳过 DONE 及完整 phase seal 时至少检查以下结构，且真正读取被 pin 的文件：

- history 恰好 epoch 1–15，训练损失有限，attempted steps 每轮等于 `ceil(len(fit)/16)`，AMP skipped 为合法非负整数且不超过尝试；A 有四指标，B 无选轮指标。
- NPZ 的严格 key、原生类别数、预测 epoch 集合（A 为 15，B 为 last）、每组 paths/labels 与计划同序、logits shape/dtype/有限性。
- checkpoint/schema、基座和初始化身份、selected epochs、可训练参数和全部 buffers、各 unique delta 的 name/shape/FP32/有限性；规则选择只从预定验证指标重建，并列规则一致。完整独立恢复另验，不用 JSON 布尔值替代。
- receipt 的 unit/phase/lock、15 轮、非负有限计时/显存/最大恢复差及每个应恢复 epoch×group 的完整列表。

不计算 outer UAR 也能完成这些结构验收。若轻量模式不重新 hash 权重，必须将这一范围明确列出，不能称完整 checkpoint gate。

### R5：DONE 提交与 ledger 完成之间的崩溃窗口没有恢复闭包

正常顺序是写 DONE→`verify_done()`→追加 unit_done。两者之间被中断时，恢复分支仅 verify＋continue，永远不会补 unit_done。若异常发生在 DONE 提交后，外层 `except BaseException` 还可能写 unit_failed，造成同一有效 DONE 与失败事件并存。

另一个窗口是 unit_start 已持久化，但 attempt 目录尚未创建：目录计数为零，下一次执行甚至不需 `--retry-failed` 就重新创建相同编号；`validate_unit` 等早期失败也可能落入此路径。ledger 中重复 start 与目录计数不一致，无法据此称“每 unit 至多两次尝试”。

修复：将 attempt 预留与 ledger 状态共同管理，执行前检查部分 JSON 行、孤 start 与既有失败。仅对**完整核验 DONE＋唯一对应 start＋没有该 attempt 的 failed/done** 的窗口追加受限恢复 unit_done，保存当前 DONE SHA 和 `recovered_from_done`；重复恢复不得重复事件。若 DONE 已完成核验，不应在后续日志写失败时反向宣告模型训练失败。真实失败保留证据，显式重试使用下一个已预留编号；不要用“目录数”推断完整尝试历史。

### R6：当前裁剪实现与 planner 的冻结约定不同

planner 的 `crop_start` 使用紧凑 JSON `[PROGRAM, seed, epoch, slot]` 的 SHA 前 64 位，按整数公式映射合法起点。初版 engine 的 `crop_uniform` 使用 `seed:epoch:slot` 字符串，再转 53 位浮点数。这不是同一确定性程序。

**CPU 实证：**`seed=3, epoch=1, slot=7, n_samples=64000`，planner 起点为 **763**，engine 起点为 **3223**。

修复：engine 调用已冻结的同一整数接口，或在开训前统一计划、协议与引擎；测试应覆盖长短录音、首末 slot、重排及配对臂，不接受只有“数值在合法区间内”的测试。这项是源／协议一致性问题，不需要读取 query 成绩来决定实现。

## 2. 全块完成门控仍需独立验收，不能依赖退出码

`process_complete` 仅表示本次选中的单位循环结束；`--unit-id` 可只指定一个单位。`--max-units` 或 `--max-hours` 到限会 operational_yield 并正常返回 0。**exit 0、process_complete 或 DONE 数量都不能证明完整 384 完成。** 新 `audit_phase` 已包含完整 phase 数量、unit 目录集合、所有权重/预测/history 和配对初始化检查，另调用 reconcile 检查提交 attempt。正式前应补针对意外单位、额外或未闭合 ledger 记录、历史失败／重试及实际全块闭合的测试；pilot 四个另验，不加入 formal。

`--max-hours` 是当前进程的软限制，逐 unit 边界检查，不是累计租期或全程序故障上限；重启会重新计时。CLI 应拒绝非有限／非正时限及非法 max_units。总预算与停卡须由运行协调层依据持久 start/预算/备份状态执行，不能把 runner 的 12 小时默认值视为已实现全程预算约束。限时 yield 后不得据 exit 0 清理未核验产物或开始科学评分。

已有 OS advisory lock 能在进程退出时释放，优于仅凭锁文件存在判断活跃；它只保护同 out 的执行过程，不能替代 freeze 竞态、跨 phase GPU 活跃检查或完整 ledger 验收。写入失败留下 `.partial` 应显式报错并保留，不应静默覆盖旧 attempt。

## 3. 当前没有发现的错误与已完成检查

新 planner 在纯 CPU 上完成 `generate` 与 `validate`，得到 semantic SHA `393109434af0bfb6d18205f8f3713aa5e08d08f0f0ac3ffc4a63f8e635422c25`；用时约 1.65 秒。结果为 CREMA A 120＋B 24、SUBESCO A 120、RAVDESS A 120，pilot 为 CREMA A/B 和另外两库 A 各一，共 4。fit 长度分别为 **576、224、64**。此处调用 planner 自己的 replay 是接口与实际容量验证，不冒称另一套独立生成算法。

只读结构审查确认：原生 6/7/8 类；SUBESCO T1、RAVDESS normal/R1 均不作结果驱动 fallback；A/B fit 的 slot 匹配；fit/query 文本互斥；A/B/outer 人互斥；report batch 按角色隔开；每 outer 人有全部原生类别；各 draw 的五折外测覆盖全部人物。未发现通过元数据路径选择科学结果的逻辑。所有这些约定仍需由 runner 真正调用 replay，不能只依赖手工先运行一次 planner。

engine 当前直接调用 `metrics()` 的 group 只有 A/B，outer 只 forward 和保存，未看到从 outer 准确率或损失选择规则的直接路径。四规则从同一份验证 logits 计算 CE 与精确 Fraction 宏召回；比较用严格 `<`/`>`，保留最早并列。角色分开的 batch 已解决此前指出的 outer 长度改变验证 padding 的路径。训练每轮重新 `model.train()`，报告与恢复用 eval。这里只核调用结构，不因此宣称完整输入隔离/运行状态测试已通过。

资源收据目前保存 allocated 和 reserved 两种峰值；engine 的 `wall_seconds` 包含模型构建、实际用到的音频预处理、训练、验证、存档与磁盘恢复，排除 runner 后续 receipt/DONE/最终验收。没有逐 epoch 的 wall 分量，不能在报告中把训练与验证成本精确拆开。更不能将拟合区间和当作 GPU busy 时间或租期。B 仅 last，保存状态的数量可能不同，资源汇总必须按实际 unique states 与 payload bytes 计。

## 4. 已执行与后续验收的分界

31 项已运行测试覆盖：实际调用 semantic replay；本地 namespace 通过而外部模块拒绝；同名同长度异字节音频拒绝；跨 attempt receipt 拒绝；损坏 NPZ/checkpoint、空或残缺 history、错误 step/skip、非有限或负计时拒绝；预测 path/label/dtype/shape/epoch 拒绝；重写 receipt 不能改验证获胜轮；实际小型 delta 缺键/shape/dtype/NaN 拒绝；DONE→ledger 仅恢复一次；无 start、有同 attempt failed 或重复 start 的 DONE 拒绝恢复；半行 ledger 保留并拒绝。裁剪同接口在独立 engine 测试中覆盖，此文件不重复实现。

reservation 与提交后异常路径已读代码核对，尚未在本文件中模拟整个 GPU `execute` 进程被强杀。subset/yield 不等于 full phase 的边界也需由后续完整门禁测试落实。可继续四个技术 pilot，并在实际产物上执行独立门禁与预定磁盘恢复；这些后续项目没有被写成已通过。

本文没有要求新增科学对比、改动旧冻结研究或重复训练以争取正结果。所列事项是当前执行合同的正确性与来源闭合要求。

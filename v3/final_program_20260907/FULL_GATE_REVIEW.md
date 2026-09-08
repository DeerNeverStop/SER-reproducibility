# 完整 384 单位门禁：独立集合与账本测试

2026-09-07。对应 pilot 冻结版本 `3980a97` 中的 `run.py`，SHA256：`9ca940c616aa1ef4f4116d7e77892c98f87d1aaa57873571bf1c2c2b6a5cdda6`。本轮仅新增本记录及 `test_full_gate.py`，没有修改正在运行的 plan/engine/run/SCIENCE_DESIGN，没有运行 GPU 或读取真实新预测。

**结论：单位完整性已有保护，但当前完整门禁未覆盖全部账本与 attempt 生命周期。正式版修复前，不应据它释放新科学成绩或声称全账本闭合。** 这不表示正在执行的四个技术 pilot 已发生训练错误，也不要求中断它们；应在 pilot 结束后修正正式版并重新冻结。

## 测试范围与实际结果

本测试先在 CPU 上运行真实 planner 的 generate/validate，使用实际的 360 个主 A、24 个 CREMA-D B 及其角色／配对元数据。随后在临时目录创建 384 个玩具 DONE、reservation 和 ledger。

**`verify_done` 被 mock**，未创建或读取 checkpoint、NPZ、音频；`checked_plan` 的正常值引用已验证计划，计划遭修改时调用真实 planner replay。因此这些测试检验完整集合、账本及 pair 初始化比较的连接逻辑，不能称为权重、训练过程、预测数值或完整运行环境的验证。

运行环境：`E:/科研/SER/ser_gpu/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --tb=short v3/final_program_20260907/test_full_gate.py`。

- 首批 15 项：**8 通过、7 未能拒绝故障，33.77 秒**。
- 追加两项真实失败／重试边界：**1 通过、1 未能拒绝故障，6.76 秒**。
- 合计 17 项已分别执行，**9 通过、8 个拒绝预期未满足**；不是一轮完整 17 项测试的计时。

故障测试保留为普通的失败回归测试，没有 xfail 或忽略。修复后应全部通过；不能通过修改这些拒绝预期来消除报告中的失败。

## 已正确保护的路径

| 条件 | 当前结果 |
|---|---|
| 完整 384，包括全部 24 个 B，全部单位开始/完成事件闭合 | 通过；每个单位调用 payload 验证，gate 保留 384 个 DONE SHA |
| 缺主 A 或缺 B 的 DONE | 拒绝，不写完整 gate |
| 多出一个 unit 目录 | 拒绝 |
| 同单位重复 unit_done | 拒绝 |
| 配对 A/B 初始化 SHA 不同 | 拒绝 |
| 将 B 改称 checkpoint_main，并重算自洽 plan SHA | 真实 planner replay 拒绝，在 payload 验证前结束 |
| ledger 最后只有半行 JSON | 拒绝，原 bytes 保留 |
| 第一次 attempt 明确 unit_failed，第二次完整成功且 DONE 指向第二次 | 通过；合法失败重试不应被自动丢弃，但失败必须留在资源／运行记录中 |

这些通过项不能推出“任何额外账本事件都会拒绝”。以下反例均有真实测试。

## 必须修复的完整门禁漏洞

### FG1：不属于冻结计划的单位事件完全被忽略

在其他 384 个 DONE 都正确时，分别加入未知 UID 的 `unit_start`、`unit_failed` 或 `unit_done`。三个样例中 `audit_phase()` 都继续返回 `pass=True`，写出 COMPLETE_GATE。

原因是 `reconcile_done()` 每次只筛当前预定 UID，全局没有检查 ledger 所有带 unit_id 的事件是否属于冻结 phase 清单。未知 UID 的完成事件甚至没有 DONE SHA，也未被读取。

**最小修正：**一次解析完整 ledger，检查所有事件的已知类型及字段，任何 unit 事件必须属于当前冻结 phase；不允许靠遍历预定 UID 时自然跳过其他事件。

### FG2：只检查最终 DONE 指向的 attempt，其余尝试可悬空

在某个已成功单位之后加入 `attempts/0002` 的 start（无终态）或 failed（无 start），当前门禁都接受。另一个更贴近中断恢复的反例是：attempt 0001 有 start 但没有 failed/中断关闭，attempt 0002 已成功并提交 DONE；当前门禁仍接受，因为只核 0002。

**最小修正：**按 `(unit_id, attempt)` 建状态机，核所有 reservation、start、failed/明确中断终态和最终成功的关联；同 attempt 只能按许可顺序发生一次。已记录的失败后重试可以合法闭合，不能简单禁止全账本任何 failure，也不能把有失败的程序写成零失败。对于中断后没有终态的旧 attempt，应有明确、保留证据的恢复/中断关闭记录；不能让第二次成功自动抹去第一次状态。

### FG3：事件存在但顺序不可能，仍可完成

仅把某单位 ledger 中 done 放到 start 前面，当前门禁接受。它只比较筛出的数量和 SHA，没有检查序列先后。

**最小修正：**按 ledger 的持久记录顺序验证 reservation/start/terminal 的合法转换。DONE 提交后缺少 unit_done 的允许恢复，应当发生在已有唯一 start 之后，且保留 `recovered_after_commit` 和 DONE SHA。不要以任意排序事件后再计数的办法掩盖原顺序。

### FG4：没有任何解释的 reservation 可被忽略

在一个已经提交 attempt 0001 的单位下额外放入 `reservations/0002.json`，没有 start 或终态，当前门禁接受。当前代码只读取最终 DONE 对应的一个 reservation。

**最小修正：**检查全部 reservation 与实际 attempt／ledger 的精确对应，不能多也不能少。reserve 后、start 前的真实崩溃应作为显式恢复边界处理；若保留该预留号，需记录为何关闭，不能删除证据或靠目录数量推测从未发生。

## 正式版验收顺序

先完整解析和验证静态 ledger／reservation 清单，再验证当前 committed 单位及允许的 DONE→unit_done 恢复窗口，最后再次核对完整账本并写全 phase seal。若恢复会追加 unit_done，gate 必须绑定追加后的 ledger SHA；重复审计不得继续追加重复成功。全局校验失败不得留下新的 `pass=True` gate。

现有 `process_complete` 或进程 exit 0 只说明本次调用结束，不替代该检查。对正式程序必须保持 360 main＋24 control、384 个不同 UID 的精确集合；pilot 四个另验。所有故障／重试应在汇总中有准确范围，不能以最终 384 成功等同历史零失败。

完成修复后重新执行本 17 项及既有 31 项单位测试，并记录新的源码 SHA。还需在实际 pilot/正式产物上调用真实 `verify_done` 与独立恢复；本文件的 payload mock 不覆盖这些步骤。新的 gate 若扩展 schema 或进程事件合同，应补合理 fixture，而非把旧 mock 的简化字段当作正式协议。

# 非检查点数值复算包

本工具用于正式 **384 = 360 A + 24 B** 全部完成、完整权重 gate 通过且主评分已完成之后。准备本工具时，正式训练仍在运行；本次仅运行合成测试，**没有导出真实正式包，也没有读取其科学成绩**。

实现见 [portable_bundle.py](portable_bundle.py)，数值复核见 [independent_numeric_audit.py](independent_numeric_audit.py)。工具不是新的训练实验，也不改变冻结评分规则。

## 包含与边界

ZIP 精确收录 source lock、计划快照、完整 ledger、全部 attempt reservations、384 个 DONE、每单位最终成功 attempt 的 predictions/history/receipt，以及主评分的五 CSV、results JSON 和原始 gate 快照。另包含导出时新运行的独立数值审计、说明和逐文件 SHA/字节数清单。

失败或放弃 attempt 的状态保留在完整 ledger/reservations 中；其未提交 payload 不收录。四个 pilot 不进入本包。原始音频、预训练基座及 checkpoint 字节均不收录。源码本身不收录：manifest 记录冻结来源与两个工具的 SHA，接收者须使用匹配的源码 checkout。

**原始完整权重 gate 仍是前提。** 本工具复查小工件、完整账本及数值，不重新核验缺省的权重字节、不执行模型推理或重训。原完整运行档案、权重、输入音频和基座仍须另行保留。私有 Git 冻结不是公开预注册；manifest 自哈希不是数字签名，交付时应另行传递并核对 ZIP SHA。

## 导出与迁移验收

在源码 checkout 中，使用已有 Python 环境运行。输出均须为新路径；ZIP 不得放入 phase 或评分输入目录，接收 audit JSON 不得放入解压目录。

```text
python -B -m v3.final_program_20260907.portable_bundle export --repo . --run-dir D:/SER-final-program-20260907/formal --results MAIN_SCORE_DIRECTORY --out NEW_BUNDLE.zip

python -B -m v3.final_program_20260907.portable_bundle extract --repo . --archive NEW_BUNDLE.zip --out DIFFERENT_NEW_DIRECTORY --out-json NEW_NUMERICAL_AUDIT.json
```

导出先核 384 完整 gate、冻结来源和 reservation/ledger 闭包，再实际调用独立数值审计。小工件在审计、复制、压缩前后核对 SHA。导出命令输出 ZIP SHA、字节数与 manifest SHA。

验收先拒绝 ZIP 越界、重复路径、符号链接、权重及额外文件，再核对精确计划衍生清单与原 gate/DONE/评分承诺，最后在新目录实际运行数值复核。只有全部验收通过后才发布指定的外部 audit JSON。数值检查失败可能留下已解压目录；这不代表验收通过，命令失败且不会发布成功验收报告。已有 ZIP、目录和报告均不覆盖。

也可对已验收的目录直接复算：

```text
python -B -m v3.final_program_20260907.independent_numeric_audit --repo . --run-dir DIFFERENT_NEW_DIRECTORY/formal --results DIFFERENT_NEW_DIRECTORY/scores --out-json ANOTHER_NEW_AUDIT.json --allow-relocated-inputs
```

`--allow-relocated-inputs` 只把封存 `results.inputs.run_dir` 中的原绝对路径视为位置元数据。它不改写任何封存文件，不豁免相对清单、SHA、标签、预测或数值检查；新审计同时记录原目录与实际目录。包内旧收据的绝对路径保持原样，不是匿名化发布包。

## 已完成的合成验证

[test_portable_bundle.py](test_portable_bundle.py) 最终受影响的 **12 项测试通过（66.93 秒）**；此前与独立数值审计的 12 项测试联合运行，共 **24 项通过**。

端到端 fixture 使用完整 384 单位的微型合成 logits，并导出、迁移解压后实际运行独立数值审计。五 CSV、六项 t/CI/Holm 与控制值通过，封存结果字节未改变。反例覆盖缺 B 臂、真实修改合成 logits 后仅重签弱 manifest、越界路径、包内权重、reservation 篡改、部分 gate、覆盖已有输出，以及重定位标志不能豁免其他 SHA。

这些 fixture 的源码/计划核验入口由明确的合成替身提供，ledger 读取入口使用简短 JSONL 解析替身；**不宣称合成计划经过真实科学设计或完整模型 gate**。冻结的 `ledger_contract.audit_events`、ZIP/文件/SHA 检查、预测路径标签核对和独立数值计算实际运行，未替换。假 checkpoint 仅是标记字节，既不反序列化也不入包。

另一次只读实际来源核查确认正式 lock `cd32459cef1e09c5efe5ef4c9f3a918aeef5a0aac15f2e98731a60daa3f4d258` 的 **43 个冻结文件全部 SHA 一致**，冻结提交为 `976297d824ea98816bdc0948befd704e612a54c1`。这项检查只读取源码和 SOURCE_LOCK 元数据，与尚未完成的科学结果无关。

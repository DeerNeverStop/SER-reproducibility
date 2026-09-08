# 正式完成后的命令与依赖审计

2026-09-07，只读准备。核对了当前 [run.py](run.py)、[score.py](score.py)、[independent_numeric_audit.py](independent_numeric_audit.py)、[replay_checkpoints.py](replay_checkpoints.py)、[portable_bundle.py](portable_bundle.py) 和 [plot_results.py](plot_results.py) 的真实接口；另只读了正式 SOURCE_LOCK/plan_snapshot 元数据，以确定抽样 UID。

**本次没有执行真实全量 gate、评分、数值复算、检查点恢复、绘图或导出，没有读取正式 predictions/history 或新科学成绩。没有改动 43 个冻结文件。现有恢复工具支持 formal，不需要修复或新增 helper。** 以下命令只能在原训练进程确认退出、384 个正式单位全部完成后由根任务顺序执行，不构成自动续训或轮询程序。

## 真实依赖与最少执行顺序

| 操作 | 代码实际要求及副作用 | 不能代替什么 |
|---|---|---|
| `run audit`（可选单独预检） | 根据 phase 内 SOURCE_LOCK 确定 formal；核全部 384 权重文件、预测、history、receipt、DONE、reservations 和 ledger。获取 OS 执行锁，必要时修复缺失的 ledger completion，写 COMPLETE_GATE.json。 | 不是纯只读调用，不计算 outer 科学成绩，也不是独立模型重训。 |
| `score` | **内部必先调用完整 `audit_phase`**，通过后从全量 logits 评分；保存该次 gate 的原字节快照。 | 若已另跑 `run audit`，仍会再次读取全量权重，不能通过传旧 gate 跳过。 |
| `independent_numeric_audit` | 既有 384 gate + 完整主评分；核来源和全部小工件，再独立重算五表、六项 t/CI/Holm、控制值。 | 不加载或重新 hash 大 checkpoint；依赖先前完整权重 gate。 |
| `replay_checkpoints` | 独立新 Python 进程；核所选四单位的冻结身份、实际 checkpoint、基座及报告音频字节，重建保存状态并比较 logits。 | 软件本身只要求所选 DONE，并不强制完整 384 gate；本交付流程额外要求全量完成后才运行。四单位通过不能替代全量 gate 或独立数值复核。 |
| `plot_results` | 完整主评分、原 gate 快照、绑定这份评分文件的通过数值审计。 | 只绘既定结果，不新增检验；脚本不执行视觉 QA。 |
| `paper_facts` | 共用绘图准入门槛，并核完整384单位的DONE、小工件与审计绑定；复制既有结果生成中英事实表。 | 不读取预测或重新计算统计，也不重新核验大权重；不能替代前序验收。 |
| `portable_bundle export` | 全量 gate + 主评分；复核真实完整账本，并再运行一次独立数值审计后复制小工件。 | 不包含音频、基座及 checkpoint；仍须保留原完整档案。 |
| `portable_bundle extract` | 精确 ZIP 白名单、manifest 和深层 gate/账本身份通过后，在新目录实际运行允许路径迁移的数值审计。 | manifest 自哈希不是外部认证；另行传递 ZIP SHA。 |

推荐顺序：**训练退出且完整 → score 内的全量 gate 与主评分 → 独立数值复算 → 预定四单位 fresh-process 恢复 → 绘图与逐图 QA → 导出与异路径验收 → 论文更新与交付封存。** 这样只需一次主评分必需的全量权重 gate；若希望在评分前先单独确认技术门禁，可加下方可选预检并接受随后的重复权重检查。

已核实际 `run run` 的结束行为：每单位提交前调用 `verify_done`，循环结束后只复查 `checked_plan` 并追加 `process_complete`，**不自动调用 `audit_phase`**。`run.py` 的全量门禁调用只在 `audit` 子命令分支；主评分另有内置调用。逐单位检验不能替代全量 reservation/ledger gate。达到 `max-hours` 时也可能写 `operational_yield` 后正常返回，因此进程 exit 0 本身不是完成 384 的证明；须核精确全量目录/单位集合，最终由 score 内部门禁判定。全过程结束后不启动新的训练。

首次完整 gate 后不要重新调用 `run run` 来“确认状态”，因为运行入口会追加 process 账本事件，改变主评分绑定的 ledger。若合法重新执行了 gate，时间戳可能改变；数值审计保留主评分目录的原 gate 快照，要求新的 phase gate 仍有相同 DONE/artifact/ledger 闭包。

## 路径与公共设置

以下均为经过接口核对的 PowerShell 命令模板，**本次未执行**。`postrun` 下的目标文件或输出子目录须未存在；重复验证应使用新名字，不覆盖原报告。Python、roots JSON 和基座路径只做了存在性核对，实际输入字节由恢复命令再验。

```powershell
$postrunPython = 'E:/科研/SER/ser_gpu/Scripts/python.exe'
$postrunRepo = 'C:/Users/jock8/Documents/ChatGPT/论文/SER-speaker-execution-20260906'
$postrunPhase = 'D:/SER-final-program-20260907/formal'
$postrunRoot = 'D:/SER-final-program-20260907/postrun'
$postrunScores = Join-Path $postrunRoot 'scores'
$postrunAudits = Join-Path $postrunRoot 'audits'
$postrunFigures = Join-Path $postrunRoot 'figures'
$postrunRoots = 'D:/SER-final-program-20260907/inputs/audio_roots.json'
$postrunBase = 'C:/Users/jock8/.cache/torch/hub/checkpoints/wavlm_base_plus.pth'
Set-Location -LiteralPath $postrunRepo
```

采用 `scores`、`audits`、`figures` 兄弟目录：`plot_results` 除了禁止进入 score 和 phase，还禁止输出目录处于 **audit JSON 的父目录之内**。例如把数值审计直接放到 `postrun/numeric.json`，再输出 `postrun/figures` 会被当前绘图门槛拒绝；上述布局避免该问题。

### 可选：单独完整预检

```powershell
& $postrunPython -X utf8 -B -m v3.final_program_20260907.run audit --repo $postrunRepo --out $postrunPhase
if ($LASTEXITCODE -ne 0) { throw 'Full formal gate failed; stop postrun work.' }
```

此 `audit` 子命令使用的是 `--out`，不是 `--run-dir`。不需要 `--phase formal`、`--roots`、`--model-path` 或 `--device`：实际 phase 来自锁；权重读取使用 CPU，不执行模型前向。即使省略本可选步骤，下一步也会执行相同门禁。

### 1. 全量门禁与主评分

```powershell
& $postrunPython -X utf8 -B -m v3.final_program_20260907.score --repo $postrunRepo --run-dir $postrunPhase --out $postrunScores
if ($LASTEXITCODE -ne 0) { throw 'Full gate or main scoring failed; do not proceed.' }
```

`score` 输出为全新的目录；不得放进 formal 任意子目录或让它成为 formal 的祖先。完成产物是五个 CSV、results.json 和 complete_gate.json，共七件；不能把中途已出现的目录视为成功。原 phase 的 COMPLETE_GATE.json 由内部门禁生成，评分目录中的小写 complete_gate.json 是本次评分绑定的原字节快照。

### 2. 独立数值复算

```powershell
$postrunNumeric = Join-Path $postrunAudits 'numeric_original.json'
& $postrunPython -X utf8 -B -m v3.final_program_20260907.independent_numeric_audit --repo $postrunRepo --run-dir $postrunPhase --results $postrunScores --out-json $postrunNumeric
if ($LASTEXITCODE -ne 0) { throw 'Independent numerical audit failed.' }
```

原路径审计不加 `--allow-relocated-inputs`。验收报告需 `pass=true`、384 单位、五 CSV、六项检验及全部数值在 1e-9 容差内，并有来源/输入前后 SHA。报告中的 checkpoint 项只是历史 gate pins 和当前 stat，不能称又验了一遍大权重。

### 3. 固定首面板的四单位独立恢复

从当前正式计划直接核出的集合如下，没有根据任何效应或 winner epoch 选单位：

| 单位 ID | 原生头 | 抽样坐标 | 恢复范围 |
|---|---:|---|---|
| `fpc_formal_cremad_d00_f0_A` | 6 | CREMA-D / draw 0 / fold 0 / A | 四规则 winner 与 last 的所有去重保存状态 |
| `fpc_formal_cremad_d00_f0_B` | 6 | 同一 CREMA-D 面板 / B | 固定 last=15；其 NPZ 时间轴只有一个位置 |
| `fpc_formal_subesco_d00_f0_A` | 7 | SUBESCO / draw 0 / fold 0 / A | 四规则 winner 与 last 的所有去重保存状态 |
| `fpc_formal_ravdess_d00_f0_A` | 8 | RAVDESS / draw 0 / fold 0 / A | 四规则 winner 与 last 的所有去重保存状态 |

```powershell
$postrunRestore = Join-Path $postrunAudits 'formal_first_panel_restore.json'
& $postrunPython -X utf8 -B -m v3.final_program_20260907.replay_checkpoints --repo $postrunRepo --run-dir $postrunPhase --roots $postrunRoots --model-path $postrunBase --unit-id fpc_formal_cremad_d00_f0_A --unit-id fpc_formal_cremad_d00_f0_B --unit-id fpc_formal_subesco_d00_f0_A --unit-id fpc_formal_ravdess_d00_f0_A --out-json $postrunRestore --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Independent formal checkpoint replay failed.' }
```

必须等原训练 Python 进程退出、同一块本机 GPU 空闲后新开此进程；不要与现有训练争资源。该调用不租云、不新训练。若此前人为屏蔽了 CUDA，需要先恢复原 GPU 可见设置；不要用更换设备来掩盖环境不匹配。CUDA 恢复要求 torch/torchaudio/CUDA/device/threads 与原收据一致；CPU 模式虽被工具支持，但记录环境差异，不能自动替代预定的同环境 GPU 恢复。

代码按 `lock.phase` 取正式单位，不存在 pilot-only 限制；B 使用保存的 `[15]` 时间轴，A 使用保存的 1–15 时间轴，因此无需硬编码 index 14。每个去重 epoch 都恢复 A/B/outer 三组，保留原 role batch 和数据预处理。总去重状态数依保存 winner 是否重合而定，现在不能预写为 pilot 的 11 状态或 33 组。四单位通过的标准为实际所有保存状态×三组的 logits 绝对最大差不超过 1e-5，rtol=0；它不计算 outer UAR/CE/检验。

工具只强制报告位于 phase 的 `units` 子树和音频 roots 之外；本命令进一步放到整个 phase 之外的 audits 目录。它重读基座、所选 checkpoint 和报告音频，不应对非检查点 ZIP 的解压目录调用，因为该包故意缺这些输入。

### 4. 固定绘图及人工视觉检查

```powershell
& $postrunPython -X utf8 -B -m v3.final_program_20260907.plot_results --results $postrunScores --audit $postrunNumeric --out $postrunFigures
if ($LASTEXITCODE -ne 0) { throw 'Figure contract failed.' }
```

输出三张图，每张 PNG/PDF/SVG，加 FIGURE_MANIFEST.json。图中沿用六项主检验、点态 t 区间和描述性曲线/控制，不新增统计终点。脚本明确标注 `visual_review_required=true`，仍需逐图查看标签、区间、图例、页边与缺字。对迁移包的评分作图，应先使用迁移后新生成的数值审计，因为图脚本要求审计绑定当前评分文件的实际绝对路径。

### 4b. 中英共用事实表

```powershell
$postrunFacts = Join-Path $postrunRoot 'paper_facts'
& $postrunPython -X utf8 -B -m v3.final_program_20260907.paper_facts --results $postrunScores --audit $postrunNumeric --out $postrunFacts
if ($LASTEXITCODE -ne 0) { throw 'Shared bilingual fact admission failed.' }
```

生成同源的 `facts.json`、`tables_zh.md` 与 `tables_en.md`，保留全部六项预定检验、四种选择规则和末轮、以及预定描述，不按显著性筛选。JSON保留来源原始数值，Markdown只做显示舍入。正式结果不能加 `--synthetic`；该开关仅供有显式模拟标记的接口演练。输出目录必须为新的兄弟目录，不能进入评分、审计或原phase。说明与已完成模拟验证见[PAPER_FACTS.md](PAPER_FACTS.md)。

### 5. 非检查点包导出及异路径实测验收

```powershell
$postrunZip = Join-Path $postrunRoot 'bundles/formal_numeric_bundle.zip'
& $postrunPython -X utf8 -B -m v3.final_program_20260907.portable_bundle export --repo $postrunRepo --run-dir $postrunPhase --results $postrunScores --out $postrunZip
if ($LASTEXITCODE -ne 0) { throw 'Portable export failed.' }

$postrunUnpacked = Join-Path $postrunRoot 'portable_accept/bundle'
$postrunRelocatedAudit = Join-Path $postrunAudits 'numeric_relocated.json'
& $postrunPython -X utf8 -B -m v3.final_program_20260907.portable_bundle extract --repo $postrunRepo --archive $postrunZip --out $postrunUnpacked --out-json $postrunRelocatedAudit
if ($LASTEXITCODE -ne 0) { throw 'Relocated numerical acceptance failed.' }
```

导出会再执行一次小工件数值复核；不会再加载全量大权重。接收者使用相同冻结源码和工具 checkout；accept 内部加明确的 relocation 标志，只把 results.inputs.run_dir 视为原路径元数据，其他 SHA 和数值承诺不豁免。原始封存评分文件不会被改写。

ZIP 的精确清单包含导出时新生成的数值审计，**不包含**另行生成的 formal_first_panel_restore.json、原路径数值审计或图件；它们应在最终论文交付包另行归档，不能硬塞入该 ZIP 而使白名单失效。完整保留 D 盘原 checkpoint 档案、音频与基座，不能因便携包通过就清理原始证据。其他说明见 [PORTABLE_BUNDLE.md](PORTABLE_BUNDLE.md)。

## 本次核对的来源身份

正式锁为 `cd32459cef1e09c5efe5ef4c9f3a918aeef5a0aac15f2e98731a60daa3f4d258`，冻结提交为 `976297d824ea98816bdc0948befd704e612a54c1`，计划 SHA 为 `393109434af0bfb6d18205f8f3713aa5e08d08f0f0ac3ffc4a63f8e635422c25`。这些是既有运行的来源身份，不是当前完成证明，也不是公开预注册证明。

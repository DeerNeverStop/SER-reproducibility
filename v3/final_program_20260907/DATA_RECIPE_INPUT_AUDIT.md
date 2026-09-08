# 已执行 Study II 配置包输入审计（2026-09-07）

本文件仅清点已经执行并封存的 `SER26-STUDY2-CORE-1`。没有重新生成计划、重新选样或新增模型训练；未读取新 384 单元研究的任何分数。这里的“重放”首先指从原计划导出同一录音清单，与重训模型、从保存预测复算结果分开。

仓库根为 `C:/Users/jock8/Documents/ChatGPT/论文/SER-speaker-execution-20260906`。下列相对路径均相对于此根，另行注明的旧归档路径除外。

## 1. 已存在的真实输入与字节身份

| 文件 | 已保存字节数 | 解压后字节数 | 用途 |
|---|---:|---:|---|
| `v3/data_design/evidence/core_plan.json.gz` | 3,006,697 | 42,665,048 | 原 1,440 个训练单元的完整有序 fit/val/test 路径、配置、种子、UID |
| `v3/data_design/evidence/representative_manifest.csv.gz` | 299,746 | 706,489 | 6,524 个代表录音的路径、说话人、文本、类别、强度、音频 SHA、选择原因 |
| `v3/data_design/evidence/PLAN_LOCK.json` | 3,369 | — | 原计划语义 SHA、原 JSON/压缩包字节 SHA、输入与 16 源码 SHA |
| `v2/manifests/cremad_manifest.csv` | 949,725 | — | 原 7,442 行清单与音频字节身份 |
| `v3/data_design/metadata/VideoDemographics.csv` | 4,033 | — | 官方 Female/Male 元数据分层 |
| `v3/data_design/evidence/unrepresented_cells.json` | 442 | — | 清洗后无 MD/XX 代表的实际记录 |
| `v3/data_design/evidence/preflight_verification.json` | 3,399 | — | 原完整元数据预检记录 |

这七个现有文件合计 **4,267,411 bytes**，没有音频和特征缓存。原计划绑定的 16 个源码文件目前均存在，合计 **221,155 bytes**，本次逐文件 SHA 全部匹配旧锁。

关键身份已在本次读取中核对：

- 原计划语义 SHA：`f258cabe582d97b3a386a666b566730d00241a5ded0e14ad61d590cb3c861503`。
- `core_plan.json.gz` SHA：`76a84b56189af5e2a9888f8690a912f1902f7db0eb7f51ede9f5bd4da179cf43`。
- 解压的原 JSON 字节 SHA：`27d42104e9df3b031c11293caec096f714eb4f13caedde8c9419fc517156ada9`。
- `representative_manifest.csv.gz` SHA：`7d3f2eeed0e7993704e272a36006533365c79df61f2353b581622dfbaeea6740`。
- 解压的代表 CSV 字节 SHA：`19f03f6cdd17b91e35b9385b56641908b21875b07219236f9e80f7a9ae7e463e`。
- `PLAN_LOCK.json` SHA：`9861f122e12fabdea9d53a86f12461f9d8822990603baad1d906b207baa0fb2b`。
- 原 manifest SHA：`e43224f032d474ac5e0d1b46d82cb7580c88bb08bac87e989fc767fd8eef552b`；demographics SHA：`e33ddc8b60bedf98541735801127cbb67098b5957271adbdd5dea8a44294eec3`。

原未压缩文件仍在 `C:/Users/jock8/Documents/ChatGPT/论文/SER-study2/v3/data_design/work/plan/`。本次确认 `core_plan.json` 和 `representative_manifest.csv` 分别与上列解压字节完全一致。因此应直接复制/解压这份计划，不能调用当前或旧版 `build()` 再生成一份“等价计划”替代原件。

**锁的边界：**原 `PLAN_LOCK` 直接绑定计划与原 manifest/规则源码，未独立列出代表 CSV 的文件 SHA。本次通过原输入及旧选择规则复核代表表，完全匹配；新交付包可另外记录上述 CSV SHA，但不能把新记录说成原来已有的冻结字段。

## 2. 6,524 行究竟是什么

代表表列为 `relative_path,speaker,sentence,label_index,intensity,sha256,selection_reason`。每行对应一个真实录音，不含虚构样本或补抽样本。本次核对：

- 原 7,442 行按旧卫生规则剩 7,435 行；再按 speaker × prompt × class 优先 MD、否则 XX 得 6,524 行。
- 6,524 个路径、音频 SHA 和 speaker × prompt × class 单元均唯一；覆盖 91 人、12 个文本、6 个情绪类别。
- 455 行为 `MD_preferred`，6,069 行为 `XX_no_MD`；没有 HI/LO 替代。
- 每行的说话人、文本、类别、强度、SHA 均与原 manifest 对应行一致；冻结计划所有 fit/val/test 路径均在这份代表表内。
- `unrepresented_cells.json` 记的是清洗后仍有原录音、但无 MD/XX 可选的 `1040_ITH_SAD_X.wav`，不等于完整笛卡尔网格中所有缺格的总清单。理论 91 × 12 × 6 = 6,552 格与 6,524 行相差 28，不能把该单条记录解释为“全语料只缺一格”。

依据：`v3/data_design/core_plan.py:44–80` 的卫生/代表规则；`core_verify.py:293–315` 的测试缺格检查。此处只按已保存元数据核对，没有扫描或重新生成音频。

## 3. 唯一面板数：录音清单与人群集合分开

这里以**实际有序路径列表完全相同**定义同一录音面板；另用排序后列表复核，得到相同的唯一数，不存在只是顺序不同造成的虚增。

| 对象 | 独特数量 | 在 1,440 个 unit 中的重复情况 |
|---|---:|---|
| fit 录音清单 | **720** | 每份 2 次，分别用于 CNN/Ridge |
| val（early-stop）录音清单 | **90** | 每份 16 次 |
| test 录音清单 | **90** | 每份 16 次 |
| 完整 `(fit,val,test)` 三元组 | **720** | 每份 2 次，分别用于 CNN/Ridge |
| fit 说话人集合 | **180** | 90 组 12 人、90 组 48 人；每组用于 8 个 unit |
| val 说话人集合 | **15** | 每组 8 人；每组用于 96 个 unit |
| test 说话人集合 | **15** | 9 组 19 人、6 组 17 人；每组用于 96 个 unit |

90 个 context 是 3 draw × 5 fold × 6 rotation。每个 context 有 2 budget × 2 speaker-count × 2 scenario = 8 个数据配置，乘 2 模型得到 16 个 unit。预算和人数不是新增独立外测人群。

fit 中 360 份有 288 行，360 份有 576 行。val 面板长度为 279–288，test 为 197–228；缺格保留实际可用录音，不能在导出时擅自补齐到理论长度。所有 fit 清单的音频并集是 6,342 条，val 并集 4,730 条，test 并集 6,524 条；这些是跨许多不同角色划分的并集，**不能合成一个训练集后继续声称测试未见人**。

同一 draw 中，六个 rotation × 五折的 30 份唯一 test 清单恰好覆盖全部 6,524 条代表录音，每条一次；三个 draw 会再次使用这些录音，绝不是 19,572 条不同测试样本。

## 4. 哪些配对真的相同

以下均直接由原 unit 列表比较，而非仅根据设计文档推断。

1. **跨模型：**固定 draw/fold/rotation/B/S/scenario 后，CNN 与 Ridge 的 fit、val、test 有序路径完全相同，`train_seed` 也相同；配置和模型不同，因此 UID 不同、训练不能合并。
2. **同 context 的所有配置：**16 个 unit 的 val、test 完全相同，且同一个训练种子；全计划恰有 90 个不同 `train_seed`。val 是独立 early-stop 说话人，不是最终外测集合。
3. **seen/new：**固定其余轴，两臂训练说话人和每人的文本槽图相同，共享六个 fit 文本，把另两个 seen 文本换成两个非 query 文本。fit 文件共享严格 75%：B=288 时 216 条，B=576 时 432 条；不是同一个 fit 清单。val/test 不变。
4. **S=12/48：**12 人集合嵌套于对应的 48 人集合，**录音清单不嵌套**。固定条数时人数增加会降低每人的文本覆盖，不能说这是只给同一训练集增加 36 人。
5. **B=288/576：**对应人数/scenario/context 的训练人集合相同，但具体文本曝光量变了；不能把种子相同当作两预算的同一模型轨迹。
6. **跨 rotation/draw：**同 draw/fold 的 val/test 人群在六个 rotation 固定，但文本不同，所以仍是六份不同录音清单；draw 改变说话人分折。所有这些配对共享有限原语料，不是新的独立语料重复。

依据：`core_plan.py:83–99,130–152,175–205`；独立检查器 `core_verify.py:263–321`。导出时应保留原 `val` 字段并解释为 early-stop，不能改名为“独立最终测试”。

每人的 fit 文本覆盖应在包内明确展示：

| B | S | 每人文本数 | 每文本人数 | 每人录音数 |
|---:|---:|---:|---:|---:|
| 288 | 12 | 4 | 6 | 24 |
| 288 | 48 | 1 | 6 | 6 |
| 576 | 12 | 8 | 12 | 48 |
| 576 | 48 | 2 | 12 | 12 |

每个实际 speaker × prompt 边均含六类各一条。两性总人数与每个文本上的人数均等分。这是已执行的四个预算配置，不是依据新分数挑出的“最佳配方”。

## 5. 如何对应已经执行的记录

当前 Git 小证据包在 `paper/study2/results/`：`completion.json.gz`、`analysis_lock.json.gz`、`verification.json.gz`、`merge_provenance.json.gz`、`MERGE_DONE.gz`、`ledger.jsonl.gz` 及 `BUNDLE_MANIFEST.json`。本次重新核对：

- 原计划语义 SHA 与所有 1,440 个 unit 的自哈希正确；completion 的 UID 集合与计划恰好相同。
- `analysis_lock.done_sha256` 与 completion 的映射相同；其 completion 字节 SHA 正确。
- 原 completion JSON 字节 SHA 为 `dc52a4f60e4e00ad67cad74c9eef304a7d96ffa092584dd402692ec74d62e209`；analysis-lock 原 JSON 字节 SHA 为 `191ed52748d87690d5d755c17b0ca5c6bd2d160f48541c9712a04aa8c86fc9d2`。

完整原记录仍在 `C:/Users/jock8/Documents/ChatGPT/论文/SER-study2/v3/data_design/work/merged_core/units/<unit_id>/`，不是新 384 单元运行目录：

| 文件名 | 文件数 | 合计 bytes |
|---|---:|---:|
| `DONE` | 1,440 | 529,920 |
| `unit.json` | 1,440 | 1,447,624 |
| `predictions.npz` | 1,440 | 39,562,368 |
| **总计** | **4,320** | **41,539,912** |

本次读这 4,320 个旧文件的字节并重新计算 SHA，全部通过 `completion → DONE → unit.json/predictions.npz` 链，并核对 unit/plan/model 身份；**没有打开 NPZ 数组或重新计算科学分数**。原 runner 的记录合同见 `core_run.py:534–584`。若交付方要附保存预测以支持数值复算，现有内容约 39.6 MB，无需模型/GPU。

旧 `verification.json` 包含原电脑绝对路径。复制它保留的是历史验收证据，不会自动把旧路径重新绑定成交付位置。新包必须另有当前相对路径/字节 SHA 映射，不能覆盖或改写原验收 JSON。

## 6. 最小可用交付建议

建议交付一个**现有配方导出包**，以原 `core_plan.json.gz` 为唯一真源，保留全部四种 B/S、两种 seen/new、两个模型。不要生成新 split、筛掉不利配置，或把其中某个面板包装成推荐器。

- **原件层：**第 1 节七个文件；补入原 SPEC/PRIOR_EXPOSURE 说明，以及原 PLAN_LOCK 绑定的 16 源码文件或精确可取的版本和 SHA。gzip 应保留原字节，避免重压缩改变身份。
- **便用层：**从原 1,440 个 unit 原样导出可读索引，包含 unit_id、model、draw/fold/rotation、B/S/scenario、P_per_speaker、train_seed、完整 config，以及三个清单的内容哈希。可只存 720 个唯一 fit 和各 90 个 val/test 清单，用引用去重；必须验证能够无损重建每个原 unit 的有序列表和原 unit_id。
- **录音层：**代表表已带音频 SHA；可从同一表连接官方 sex 字段方便查看，但新增字段注明是便用视图，不改原代表表。导出保持原相对音频文件名，不绑定某台机器的盘符，不复制/转换音频，不填补缺格。
- **执行层：**附原 completion、analysis_lock、ledger、verification、merge 证据和原 BUNDLE_MANIFEST。若要让接收者独立核查到实际单元而不是只读历史映射，另附原 1,440 个 DONE + unit.json（合计 1,977,544 bytes），新 manifest 记录复制文件的 SHA。若还提供预测复算功能，则把已存在的 NPZ 也一并原样附入。
- **验收层：**新包写自己的 manifest 和范围声明；检查原件 SHA、720/90/90 去重后重建、1,440 UID/轴集合、所有路径有唯一代表录音、原 completion 对应、用户音频的 SHA（仅在其提供音频时）。新包不是新预注册或新实验结果。

不要承诺“拿这约 4 MB 就可立即重训”。录音清单重放不需要 GPU；重新训练还需合法取得并哈希匹配的真实音频或原计划指定的 logmel/WavLM 特征缓存、旧代码环境及预算。原计划保存了两个缓存文件名与 byte SHA，却没有把缓存数据塞进这个小包。Ridge 和 CNN 的计算流程也不同，不能把相同数据面板称为相同模型的重复预测。

包的核心用途是让别人精确回答“这次固定条数下，究竟选了哪些人、哪些句子、哪些录音，如何对应已执行记录”。它不证明这四种配置在别的语料/模型上最优，也不证明它们消除了 speaker-dependent / speaker-independent 差距。

## 7. 本次检查方法与范围

使用标准库 `gzip/json/csv/hashlib/collections` 只读解压原件；按旧 canonical JSON 规则去掉自哈希字段核验 plan/unit；按有序路径元组与路径集合分别去重；连接已锁元数据检查录音身份；读取旧完成文件字节复核 SHA 链。没有调用 planner `build()` 或 `roles()` 产生新计划，没有读取特征/模型权重，没有新增统计检验，没有改动任何冻结源码或历史结果。

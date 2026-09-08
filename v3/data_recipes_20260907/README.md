# 已执行数据配置的可复现清单

这里提供 Study II 原配置的导出工具，回答“固定录音条数时，具体选了哪些人物、句子和录音，怎样对应已执行的模型”。唯一计划来源是已封存的 `SER26-STUDY2-CORE-1`；工具不重新划分、不重新选样、不读取新 384 单元结果，也不按旧结果筛选所谓最佳配置。

这是 **录音清单和历史执行身份的复现包**，不是音频数据集或可立即重训的模型包。原录音、特征缓存、模型权重和预测数组均不随包分发。导出清单不需要 GPU；重新训练仍需取得匹配的音频或原缓存，并准备原环境。原研究方法、结果和限制见 [Study II](../data_design/README.md)，精确输入审计见 [DATA_RECIPE_INPUT_AUDIT.md](../final_program_20260907/DATA_RECIPE_INPUT_AUDIT.md)。

可直接下载 [4.51 MB ZIP](releases/study2_recording_recipes_v1.zip)，内含独立工具与使用说明；解压后按包内命令校验，无需绑定本仓库的绝对路径。ZIP 已实际解压到另一目录，并在新 Python 进程中通过完整配置校验与示例导出。另有 17 项使用真实旧输入的独立测试通过，包括全部 1,440 个 unit 的手动无损重建及损坏反例；范围和身份见 [ACCEPTANCE.json](ACCEPTANCE.json)。

ZIP SHA-256：`1252d3b28a0b1a8db3cdfe9c58ede0d22033f0d6bf95de7691a98e137494afd5`。这是已完成的配置交付，不是仍在运行的新 384 单元实验的完成证明。

## 配置有哪些

| 拟合录音数 B | 训练人数 S | 每人句子数 | 每人录音数 | 每个训练句的贡献人数 |
|---:|---:|---:|---:|---:|
| 288 | 12 | 4 | 24 | 6 |
| 288 | 48 | 1 | 6 | 6 |
| 576 | 12 | 8 | 48 | 12 |
| 576 | 48 | 2 | 12 | 12 |

每个人—句子组合含六类情绪各一条；每个配置使用八个训练句。各配置性别人数均分，同预算下每句的人数与性别配额匹配，总人数有意变化。因此人数增加同时减少每人的句子覆盖，并不是只增加人物、其他条件全部不动。

每个配置又包含 `prompt_seen` / `prompt_new` 两种文本条件。两臂共用验证和测试录音，训练录音严格共享 75%；这里的 seen/new 指 **文本**，两臂最终测试人物都未参与模型拟合。S=12 的人物集合嵌套于 S=48，但录音清单不嵌套。不要把多个划分的录音合并后仍声称保持陌生人物测试。

完整设计有 3 draw × 5 fold × 6 rotation = 90 个上下文，每个上下文包含八个数据配置和两个模型，合计 1,440 个原单元。CNN 和 Ridge–WavLM 共用同一配置的有序路径与训练种子，模型流程不同。去重后有 720 份 fit、90 份 val、90 份 test 清单；清单数量、模型拟合数量均不是独立人群样本量。

## 使用

从仓库根目录，使用 Python 标准库即可。每次为输出选择不存在或为空的目录；工具不会覆盖已有非空输出。

```shell
python -m v3.data_recipes_20260907.recipe_bundle build --repo . --out /path/to/study2_recipes
python -m v3.data_recipes_20260907.recipe_bundle verify --bundle /path/to/study2_recipes
python -m v3.data_recipes_20260907.recipe_bundle export --bundle /path/to/study2_recipes --unit-id e890d24810c9f91a11f1fcf8b78d5904d497b78ed0bda3b75b4642b335fe5fc5 --out /path/to/one_unit
```

上述 UID 是原计划首个单元：Ridge–WavLM、draw 0 / fold 0 / rotation 0、288 条、12 人、已见文本。它只是使用示例，不是根据结果推荐的配置。全部原 UID 和对应轴、种子、模型配置在 `units.json` 中。

`build` 原样复制计划和元数据，校验旧计划及绑定源码，再生成去重清单和当前包的文件 manifest。`verify` 在不依赖原仓库绝对路径的情况下校验包内原件及全部配置，确认能无损重建原 unit 的完整路径顺序和身份。`export` 从已通过校验的包导出指定单元的 `fit.csv`、`val.csv`、`test.csv` 与完整 `unit.json`。CSV 的 `row_index` 保留原顺序，另附原代表表字段和来源 sex 元数据；导出 CSV 不改变原代表表。

fit 用于拟合，val 用于原模型的早停/选择，test 用于最终报告。缺格按原清单保留，不能为了整齐补录音或重复采样。6,524 行代表表来自 91 人 × 12 句 × 6 类理论网格中的实际可用录音；与完整网格相差 28 格。`unrepresented_cells.json` 只记录清洗后有录音却没有合适 MD/XX 代表的一种情况，不是这 28 个缺格的完整清单。

## 身份与边界

- 原计划语义 SHA：`f258cabe582d97b3a386a666b566730d00241a5ded0e14ad61d590cb3c861503`。原计划、单位 ID、音频相对路径和 SHA 均不改写。
- 新 manifest 证明当前包装的字节身份；它的生成发生在旧 Study II 结果已知之后，不是新的预注册或原来已存在的冻结证明。原 `PLAN_LOCK` 的历史字段按原字节保存。
- 包附历史完成与审计记录，保留旧记录中的原路径和原统计含义。没有附预测数组，因而 `verify` 不声称重新计算了科学结果，也不声称重新验证了全部模型运行。
- 清单携带音频 SHA，但本工具不读取用户音频，因此校验包通过不代表本机音频已通过哈希检查。还需逐文件验证音频后才能重训。
- 这些配置没有证明普遍最优，也没有证明 SD/SI 差距已经缩小或消失。已知结果及其对后续数据构建的有限启发见 [综合判断](../../paper/DATASET_DIRECTION_SYNTHESIS_20260907.md)。

在当前本机正式训练期间，新增 CPU 命令应屏蔽 CUDA、限制至两线程并使用 Windows BelowNormal 优先级，避免与训练争用资源。这不改变原训练进程。

## English summary

This tool exports the exact, previously executed Study II recording configurations. It does not generate splits, select configurations by their outcomes, train models, or access the new 384-unit study's results. All 1,440 original units reconstruct losslessly from 720 fit, 90 validation, and 90 test lists. Each export preserves the original order, configuration, seed, unit ID, and recording metadata.

The package contains metadata and historical execution records, not audio, feature caches, weights, or prediction arrays. Validation establishes package and configuration identity; it is neither model retraining nor a new numerical result audit. Speaker count and per-speaker prompt coverage vary jointly. The text conditions share held-out speakers and test recordings. No optimal dataset or elimination of the speaker-dependent/independent gap is established.

# Public reproduction guide / 公开材料复现说明

本公开快照提供不同深度的核查。各层不能相互冒充：读表、从预测重算、重新训练以及检查原始归档是不同工作。

## 1. Read the complete results

- 当前论文的原 384 次：[scores](../../v3/final_program_20260907/reports/scores/)；六项原主检验在 `results.json` 的 `tests`。
- 新 720 次：[results](../autodl-supplement-20260908/results/)；十项新主检验在 `primary_tests.csv`，保留全部十项。
- 两套原始 `draws.csv`、全部 `selected_epochs.csv`、完整 `curves.csv` 均在相应目录；新 720 的 45 轮轨迹各自提供 15/45 两个候选窗口，因此其 960 行 context 结果不代表 960 次新训练。
- 全历史的规模、结论和来源见[实验目录](EXPERIMENTS_ZH.md)。

## 2. Recompute all 16 primary tests from public tables

在仓库根目录执行，Python 3.11 或更新版本：

```text
python -m pip install -r docs/public-release-20260908/requirements-replay.txt
python docs/public-release-20260908/verify_public_manifest.py --repo .
python docs/public-release-20260908/verify_public_results.py --repo .
```

第一条验证命令核对公开文件清单的大小和 SHA-256。第二条从已公开 draw 表独立重算均值、样本标准差、标准误、df=23 的双侧 t/p、点估计 95% 区间和 Holm 校正，分别保留原六项与新十项家族。它不导入原训练器或评分器，不使用 GPU、音频或权重，也不宣称由此核验了未下载的逐条 logits。

## 3. Replay the original 384 fits from saved predictions

从[固定版本 Release](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.08)下载 `formal_numeric_bundle.zip`，先与 [AVAILABILITY.md](AVAILABILITY.md) 的字节数和 SHA 对照。不要把输出放在仓库、输入包或任何已存在的目录内。

```text
python -m v3.final_program_20260907.portable_bundle extract --repo . --archive /path/to/formal_numeric_bundle.zip --out /new/path/replayed_384 --out-json /new/path/replay_audit.json
```

这条更深的命令使用旧冻结实现及其导入链。除轻量依赖外，还需要 NumPy、SciPy、PyTorch、Transformers、scikit-learn、librosa、soundfile 等研究环境依赖；可使用原有研究环境。它运行在 CPU，不运行模型推理或训练。原环境记录和精确冻结文件在研究报告与包内 `SOURCE_LOCK.json` 中。

包内包含 384 次正式训练的成功预测/历史/收据、原账本、计划、原评分及逐文件哈希，排除四个 pilot、原始音频、基座和 checkpoint 字节。工具在新目录检查包成员、SHA、身份和账本闭包，再实际重算 63,901 个数值比较。历史完整权重 gate 是保存的证据，缺省权重不会被本命令重新哈希。

旧 JSON 中的少量本机路径保留原字节作为来源信息；该包不是匿名包。工具的明确迁移例外仅针对路径元数据，不改动结果或降低数值门槛。详见[工具范围](../../v3/final_program_20260907/PORTABLE_BUNDLE.md)。

## 4. Rebuild data recipes and plans

[Study II 配置包](../../v3/data_recipes_20260907/README.md)及其 ZIP 可重建原研究的 1,440 个配置，包含训练/验证/测试录音清单，不含音频，不是新增 1,440 次实验。

新 720 的完整便携参考和计划位于 `v3/autodl_supplement_20260908/plans/`。从仓库根目录将计划生成到仓库外的新位置：

```text
python -B -m v3.autodl_supplement_20260908.plan generate --reference v3/autodl_supplement_20260908/plans/reference.json.gz --out /new/path/PLAN.json.gz
```

计划结构生成不等于原音频字节通过验证，也不等于完成训练。

## 5. Retrain or audit the full original archives

完整训练需要自行从原提供方取得授权音频和基座权重、核对清单中的身份、安装相应研究环境，并准备所需硬件。原 384 与新 720 的 Linux/Windows 执行环境不同；不能把另一环境上的新训练冒充历史字节相同运行。

科学训练代码保持冻结字节。部分运行器有原主机、环境、输入路径和提供商身份门禁；公开副本不是任意机器上一条命令即可重训的安装器。移植需另立配置、保留偏离记录，并区分新运行与原来源锁。公开仓库不提供账户凭据或一键租机动作。

720 的全逐轮预测及保留权重未作为本次 Release 附件发布。公开的完整表格、数值验收摘要及原核验代码允许检查报告和方法，但不能取代这些缺失字节。相关完整归档检查命令需要档案持有人在受控环境运行。历史私有 Release 的 SHA 或本地路径仅是来源记录，不代表公众已经有权或能够下载其全部内容。

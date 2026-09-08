# 独立 CSV 算术复算

`independent_csv_replay.json` 是首次只读复算的记录；`independent_csv_replay.py` 保存其独立算法，并增加输入/输出命令行参数。脚本仅依赖 Python 标准库，不导入探索的 `core.py` 或 `run.py`。

在仓库根目录执行下列命令，输出路径必须尚不存在：

```powershell
python v3/validation_reporting/review/independent_csv_replay.py --repo . --archive D:/SER-validation-report-diagnostic-20260907 --out v3/validation_reporting/review/independent_csv_replay_reproduced.json
```

再次执行时改用新的 `--out` 文件名；脚本拒绝覆盖任何已有输出，也拒绝写入输入 archive。不要使用 `python -O`。将 `--archive` 指向含 `plan.json` 和 `analysis/` 的目录；`--repo` 必须保有该计划记录的冻结源码。

复算先验证计划、输出清单和冻结源码哈希，从 1,920 个等类支持半区的 CSV 百分数恢复整数正确数/144；480 个外测 UAR 则以 Decimal 精确保留，不能换成普通准确率。之后用 Fraction 独立选择四配置、处理最低索引并列、生成逐次差值，并按两个方向→五折→24 draws 聚合。9,506 项数值比较允许误差 1e-9 pp；首次实际最大误差为 1.7763568394002505e-14 pp。

这验证的是从 CSV 到选择及聚合的下游计算，不重读原始 logits，不构成独立模型推理或科学重复；没有新增 p 值、置信区间或实验终点。首次记录在脚本保存前通过 stdin 执行；随后保存版的复跑记录为 `independent_csv_replay_reproduced.json`，额外含脚本字节 SHA256，数字检查和结果保持相同。

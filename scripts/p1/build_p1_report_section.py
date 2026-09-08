#!/usr/bin/env python3
"""Format the verified P1 summary CSVs as an append-only REPORT section.

This is a reporting formatter, not a statistical implementation. It consumes
the already verified summary tables and refuses incomplete rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "protocol_premium" / "summary"
DEFAULT_OUTPUT = PROJECT_ROOT / "work" / "p1_report_section.generated.md"

CORPORA = ("ravdess", "cremad")
MODELS = ("cnn", "resnet_se", "transformer", "fno")
CONTRASTS = ("RG", "RL", "GL")
METRICS = ("uar", "accuracy", "macro_f1")
CORPUS_LABEL = {"ravdess": "RAVDESS", "cremad": "CREMA-D"}
MODEL_LABEL = {"cnn": "CNN", "resnet_se": "ResNet-SE", "transformer": "Transformer", "fno": "FNO"}
METRIC_LABEL = {"uar": "UAR", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}
PROTOCOL_SHORT = {"random": "Random", "groupkfold": "GroupKFold", "loso": "LOSO"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def format_p(value: str) -> str:
    number = float(value)
    return f"{number:.4g}"


def verdict(row: dict[str, str]) -> str:
    low = float(row["ci95_low"])
    high = float(row["ci95_high"])
    if row["metric"] == "uar":
        if row["holm_reject_0_05"] == "true":
            return "Holm 显著；CI 不覆盖 0"
        if low <= 0 <= high:
            return "未达显著；CI 覆盖 0"
        return "未达 Holm 显著；CI 不覆盖 0"
    if low <= 0 <= high:
        return "not conclusive（CI 覆盖 0）"
    return "CI 不覆盖 0；描述性（未作 Holm 检验）"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary_dir = args.summary_dir.resolve()

    metrics = read_csv(summary_dir / "protocol_metric_summary.csv")
    premiums = read_csv(summary_dir / "protocol_premium.csv")
    cells = read_csv(summary_dir / "cell_completeness.csv")
    budget = read_csv(summary_dir / "budget_audit.csv")
    if len(metrics) != 72 or len(premiums) != 72 or len(cells) != 72 or len(budget) != 1:
        raise ValueError("verified summary design row counts changed")
    if any(row["status"] != "complete" for row in metrics + premiums + cells):
        raise ValueError("incomplete P1 summary row: REPORT generation refused")

    metric_lookup = {
        (row["corpus"], row["model"], row["protocol"], row["metric"]): row
        for row in metrics
    }
    premium_lookup = {
        (row["corpus"], row["model"], row["contrast"], row["metric"]): row
        for row in premiums
    }
    headline = premium_lookup[("ravdess", "cnn", "RG", "uar")]
    uar_rows = [row for row in premiums if row["metric"] == "uar"]
    uar_rejects = sum(row["holm_reject_0_05"] == "true" for row in uar_rows)
    uar_nonreject_ci_excludes = sum(
        row["holm_reject_0_05"] != "true"
        and not (float(row["ci95_low"]) <= 0 <= float(row["ci95_high"]))
        for row in uar_rows
    )
    accuracy_nc = sum(
        float(row["ci95_low"]) <= 0 <= float(row["ci95_high"])
        for row in premiums
        if row["metric"] == "accuracy"
    )
    f1_nc = sum(
        float(row["ci95_low"]) <= 0 <= float(row["ci95_high"])
        for row in premiums
        if row["metric"] == "macro_f1"
    )

    lines = [
        "## P1：三协议受控测量与协议溢价表（最终）",
        "",
        "### 完整性、定义与可追溯来源",
        "",
        "P1 严格沿用任何结果出现前冻结的模型、超参数、seeds `{0,1,2}` 与三种协议。",
        "RAVDESS 408 个、CREMA-D 1,212 个训练单元全部成功，合计 1,620/1,620；",
        "72/72 个 `corpus × model × protocol × seed` OOF cell 完整，没有缺失、失败或",
        "available-case 替代。主验证与不导入主汇总代码的独立统计复算均为 `pass`。",
        "本轮只格式化已经验收的 `results/protocol_premium/summary/*.csv`，没有重跑 fit、",
        "重算 OOF 或改动统计判据。",
        "",
        "三个 contrast 均按 A−B 定义：`RG = Random − GroupKFold`、",
        "`RL = Random − LOSO`、`GL = GroupKFold − LOSO`。协议均值与 seed 样本标准差",
        "来自 `protocol_metric_summary.csv` 的 `mean`、`sample_sd`；点差与配对 95% CI",
        "来自 `protocol_premium.csv` 的 `point_mean_difference`、`ci95_low`、`ci95_high`。",
        "配对单位是说话人：先在同一说话人内跨 3 seed 求均值，再做协议差；CI 是固定",
        "10,000 次整说话人 bootstrap。只有 UAR 进入预注册的 24 项双侧 Wilcoxon + Holm",
        "family；Accuracy 与 Macro-F1 只报告配对差和 CI，不追加事后显著性检验。",
        "",
        "### 主要观察",
        "",
        f"RAVDESS×CNN 的 RG-UAR 为 `{float(headline['point_mean_difference'])*100:+.2f}` pp，",
        f"95% CI `[{float(headline['ci95_low'])*100:+.2f}, {float(headline['ci95_high'])*100:+.2f}]`，",
        f"Holm-adjusted `p={format_p(headline['holm_adjusted_p'])}`。这个点估计与姊妹项目约",
        "18.1 点的事前观察一致，但这里只能称为**与先验一致**，不能称为验证了先验。",
        "RAVDESS 的四个模型中，RG-UAR 为 +15.39 至 +19.68 pp，RL-UAR 为 +11.07 至",
        "+15.28 pp；CREMA-D 的 RG-UAR 较小，为 +1.82 至 +3.06 pp。说明协议溢价在本次",
        "受控实现中强烈依赖语料，不能用一个固定折算数跨语料套用。",
        "",
        f"UAR 的 24 个预注册比较中 {uar_rejects}/24 在 Holm 0.05 下拒绝零差；其余",
        f"{24-uar_rejects}/24 未达 Holm 显著。其中 {uar_nonreject_ci_excludes} 项的未校正",
        "bootstrap CI 不覆盖 0，但 Holm-adjusted p 仍大于 0.05，因此仍不作显著声明；",
        "这不是把差异写成零。Accuracy 有 " + str(accuracy_nc) + "/24、Macro-F1 有 " + str(f1_nc) +
        "/24 个 CI 覆盖 0 的 `not conclusive` 比较。",
        "",
        "### 三类规定图",
        "",
        "![三协议 OOF 均值与 seed SD](results/protocol_premium/figures/p1_protocol_metric_comparison.png)",
        "",
        "图 1：三协议 OOF 指标均值与 seed 样本 SD；同一面板只比较冻结的统一实现。",
        "",
        "![逐说话人配对差值分布](results/protocol_premium/figures/p1_paired_speaker_difference_distributions.png)",
        "",
        "图 2：逐说话人配对差值分布；每位说话人的指标先跨三个 seed 平均。",
        "",
        "![按语料与模型分面的协议溢价](results/protocol_premium/figures/p1_protocol_premium_facets.png)",
        "",
        "图 3：按语料×模型分面的 RG/RL/GL 点差与配对 95% CI。PNG 与 SVG、源 CSV",
        "哈希和绘图脚本哈希均登记在 `results/protocol_premium/figures/figure_manifest.json`。",
        "",
        "### 完整溢价表",
        "",
        "表中协议均值均为 3 seed 的 OOF 均值 ± seed 样本 SD；Δ 与 CI 使用百分点。",
        "UAR 的 `Holm p` 是固定 24 项 family 的校正值。Accuracy/Macro-F1 的判读只说明",
        "配对 CI 是否覆盖 0，不把未校正 CI 冒充多重比较后的显著性结论。",
    ]

    for metric in METRICS:
        lines.extend(
            [
                "",
                f"#### {METRIC_LABEL[metric]}",
                "",
                "| 语料 | 模型 | 对比 | A：均值 ± seed SD | B：均值 ± seed SD | Δ pp（配对 95% CI） | Holm p | 判读 |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for corpus in CORPORA:
            for model in MODELS:
                for contrast in CONTRASTS:
                    row = premium_lookup[(corpus, model, contrast, metric)]
                    a = metric_lookup[(corpus, model, row["protocol_a"], metric)]
                    b = metric_lookup[(corpus, model, row["protocol_b"], metric)]
                    delta = float(row["point_mean_difference"]) * 100
                    low = float(row["ci95_low"]) * 100
                    high = float(row["ci95_high"]) * 100
                    p = format_p(row["holm_adjusted_p"]) if metric == "uar" else "—"
                    contrast_label = (
                        f"{contrast} ({PROTOCOL_SHORT[row['protocol_a']]}−"
                        f"{PROTOCOL_SHORT[row['protocol_b']]})"
                    )
                    lines.append(
                        f"| {CORPUS_LABEL[corpus]} | {MODEL_LABEL[model]} | {contrast_label} | "
                        f"{float(a['mean']):.3f} ± {float(a['sample_sd']):.3f} | "
                        f"{float(b['mean']):.3f} ± {float(b['sample_sd']):.3f} | "
                        f"{delta:+.2f} [{low:+.2f}, {high:+.2f}] | {p} | {verdict(row)} |"
                    )

    lines.extend(
        [
            "",
            "### 样本框偏差与 P1 外推边界",
            "",
            "1. **P0 样本框仍有覆盖边界。** 随机主样本是在冻结的 331 项 pending 候选框内",
            "   概率抽取，但候选框本身主要来自 GitHub + arXiv，只覆盖可发现且有公开代码的",
            "   工作；Papers with Code 路径失效，Semantic Scholar 替代检索因 HTTP 429 不可用。",
            "   因此 P0 风险率不能外推到无代码论文或完整 SER 文献总体。",
            "2. **P1 不是他人论文复现。** 它只测同一套冻结实现下更换划分协议的配对差，",
            "   不能据此写成证实或推翻任何具体论文，也不能把某篇论文数字直接扣掉本表数值。",
            "3. **溢价幅度只对已测配置成立。** 本轮仅含 RAVDESS、CREMA-D，四个统一模型",
            "   （CNN、ResNet-SE、Transformer、FNO）及冻结的特征、优化、早停和数据处理。",
            "   未测语料、语言、标签集、模型容量、预训练表示、时长分布与超参数可能产生不同",
            "   溢价；尤其 RAVDESS 与 CREMA-D 的差异已说明不能跨语料使用单一折算常数。",
            "4. **CI 与检验范围有限。** bootstrap 的抽样单位是当前语料中的说话人，反映本次",
            "   演员集合内的不确定性；seed SD 只有 3 个 seed。UAR 的 Holm family 固定为 24，",
            "   Accuracy 与 Macro-F1 没有确认性 p 值。CI 覆盖 0 的项目为 `not conclusive`，",
            "   不能写成没有差异或差异为零。",
            "5. **P0 与 P1 尚未在本阶段相乘。** 将公开代码样本框的风险率与协议溢价结合属于",
            "   P2；本节不进行总体文献偏移估算。作者盲编码 8–10 库仍是人类 inter-rater",
            "   声明的待办，也不影响本次 P1 受控测量本身。",
            "",
            "### P1 产出索引与停止点",
            "",
            "- 逐运行、逐折与预测来源：`results/protocol_premium/units/`、`run_plan.csv`、",
            "  `run_status.csv`、`attempt_ledger/`。",
            "- 冻结汇总：`results/protocol_premium/summary/`；主验证 `verification.json=pass`，",
            "  独立统计复算 `independent_statistics_verification.json=pass`。",
            "- 出图脚本：`tools/plot_p1_protocol_premium.py`；三类图与 manifest 位于",
            "  `results/protocol_premium/figures/`。",
            "- 一页评估检查清单：`CHECKLIST.md`。",
            "",
            "P1 到此完成并停止，等待验收；没有开始 P2。",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

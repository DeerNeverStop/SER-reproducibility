#!/usr/bin/env python3
"""Produce preregistered P0.5 random-sample estimates and P0 comparison."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import summarize_p0 as p0


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SURVEY = ROOT / "p0_5_random_survey.csv"
PURPOSIVE_SURVEY = ROOT / "survey_table.csv"
ORDER = ROOT / "p0_5_random_order.csv"
OUT = ROOT / "results" / "p0_5"
PURPOSIVE_SHA256 = "27a552cedfe8361049c6c0c346fa89a410d481a9f8b0bcb454432f8f1028fa99"
DOMAINS = tuple(p0.DOMAIN_FIELDS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float, digits: int = 6) -> str:
    return "" if math.isnan(x) else f"{x:.{digits}f}"


def pct(x: float) -> str:
    return "" if math.isnan(x) else f"{100 * x:.3f}"


def canonical_repository(url: str) -> str:
    value = url.strip().rstrip("/")
    if value.casefold().endswith(".git"):
        value = value[:-4]
    return value.casefold()


def validate_target_datasets(rows: list[dict[str, str]], sample: str) -> None:
    allowed = set(p0.TARGET_DATASETS)
    for row in rows:
        tokens = [token.strip() for token in row["target_datasets"].split("|") if token.strip()]
        if not tokens or len(tokens) != len(set(tokens)) or any(token not in allowed for token in tokens):
            raise ValueError(
                f"invalid target_datasets in {sample} {row.get('repository_id', '')}: {tokens!r}"
            )


def repo_risk(rows: list[dict[str, str]], sample: str) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    out: list[dict[str, object]] = []
    by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for row in rows:
        statuses: dict[str, str] = {}
        for domain in DOMAINS:
            values = [p0.dataset_status(row, dataset, domain) for dataset in p0.datasets_for(row)]
            statuses[domain] = p0.combine(values)
            by_domain[domain].append(statuses[domain])
        out.append({
            "sample": sample,
            "repository_id": row["repository_id"],
            "repository_url": row["repository_url"],
            "evidence_strength": row["evidence_strength"],
            "target_datasets": row["target_datasets"],
            **statuses,
        })
    return out, by_domain


def dataset_risk(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        for dataset in p0.datasets_for(row):
            out.append({
                "repository_id": row["repository_id"],
                "repository_url": row["repository_url"],
                "dataset": dataset,
                "evidence_strength": row["evidence_strength"],
                **{domain: p0.dataset_status(row, dataset, domain) for domain in DOMAINS},
            })
    return out


def summary(sample: str, statuses: dict[str, list[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        counts = Counter(statuses[domain])
        n = len(statuses[domain])
        yes, no, unknown = counts["yes"], counts["no"], counts["unknown"]
        known = yes + no
        main = yes / n
        upper = (yes + unknown) / n
        known_rate = yes / known if known else math.nan
        main_ci = p0.wilson(yes, n)
        upper_ci = p0.wilson(yes + unknown, n)
        known_ci = p0.wilson(yes, known) if known else (math.nan, math.nan)
        rows.append({
            "sample": sample, "metric": domain, "N": n,
            "yes": yes, "no": no, "unknown": unknown,
            "main_rate": fmt(main), "main_percent": pct(main),
            "main_wilson95_low": fmt(main_ci[0]), "main_wilson95_high": fmt(main_ci[1]),
            "main_wilson95_percent": f"{pct(main_ci[0])}–{pct(main_ci[1])}",
            "upper_rate_unknown_as_yes": fmt(upper), "upper_percent_unknown_as_yes": pct(upper),
            "upper_wilson95_low": fmt(upper_ci[0]), "upper_wilson95_high": fmt(upper_ci[1]),
            "upper_wilson95_percent": f"{pct(upper_ci[0])}–{pct(upper_ci[1])}",
            "unknown_rate": fmt(unknown / n), "unknown_percent": pct(unknown / n),
            "known_N": known, "known_only_rate": fmt(known_rate), "known_only_percent": pct(known_rate),
            "known_only_wilson95_low": fmt(known_ci[0]), "known_only_wilson95_high": fmt(known_ci[1]),
            "known_only_wilson95_percent": f"{pct(known_ci[0])}–{pct(known_ci[1])}" if known else "",
        })
    return rows


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p by summing tables no more likely than observed."""
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    total = row1 + row2

    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(col2, row1 - x) / math.comb(total, row1)

    lo = max(0, row1 - col2)
    hi = min(row1, col1)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed + 1e-15))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def comparison(
    random_status: dict[str, list[str]], purposive_status: dict[str, list[str]]
) -> list[dict[str, object]]:
    partial: list[dict[str, object]] = []
    pvalues: list[float] = []
    for domain in DOMAINS:
        r = Counter(random_status[domain])
        p = Counter(purposive_status[domain])
        nr, np_ = len(random_status[domain]), len(purposive_status[domain])
        r_known, p_known = r["yes"] + r["no"], p["yes"] + p["no"]
        r_rate, p_rate = r["yes"] / nr, p["yes"] / np_
        r_ci, p_ci = p0.wilson(r["yes"], nr), p0.wilson(p["yes"], np_)
        # Newcombe's hybrid-score interval for the difference between two
        # independent proportions (method 10).  This combines the two Wilson
        # score intervals quadratically; simply subtracting opposite endpoints
        # is a different, needlessly conservative interval.
        difference = r_rate - p_rate
        diff_low = difference - math.sqrt((r_rate - r_ci[0]) ** 2 + (p_ci[1] - p_rate) ** 2)
        diff_high = difference + math.sqrt((r_ci[1] - r_rate) ** 2 + (p_rate - p_ci[0]) ** 2)
        pval = fisher_two_sided(r["yes"], nr - r["yes"], p["yes"], np_ - p["yes"])
        pvalues.append(pval)
        partial.append({
            "metric": domain,
            "random_N": nr, "random_yes": r["yes"], "random_no": r["no"], "random_unknown": r["unknown"],
            "random_main_percent": pct(r_rate), "random_upper_percent": pct((r["yes"] + r["unknown"]) / nr),
            "random_unknown_percent": pct(r["unknown"] / nr),
            "random_known_N": r_known,
            "random_known_only_percent": pct(r["yes"] / r_known) if r_known else "",
            "purposive_N": np_, "purposive_yes": p["yes"], "purposive_no": p["no"], "purposive_unknown": p["unknown"],
            "purposive_main_percent": pct(p_rate), "purposive_upper_percent": pct((p["yes"] + p["unknown"]) / np_),
            "purposive_unknown_percent": pct(p["unknown"] / np_),
            "purposive_known_N": p_known,
            "purposive_known_only_percent": pct(p["yes"] / p_known) if p_known else "",
            "main_difference_random_minus_purposive": fmt(difference),
            "main_difference_percentage_points": f"{100 * difference:.3f}",
            "newcombe95_low": fmt(diff_low), "newcombe95_high": fmt(diff_high),
            "newcombe95_percentage_points": f"{100 * diff_low:.3f}–{100 * diff_high:.3f}",
            "upper_difference_percentage_points": f"{100 * (((r['yes'] + r['unknown']) / nr) - ((p['yes'] + p['unknown']) / np_)):.3f}",
            "unknown_difference_percentage_points": f"{100 * ((r['unknown'] / nr) - (p['unknown'] / np_)):.3f}",
            "fisher_two_sided_p": fmt(pval, 9),
        })
    adjusted = holm(pvalues)
    for row, adj in zip(partial, adjusted):
        row["holm_adjusted_p"] = fmt(adj, 9)
        row["holm_below_0_05_working_model"] = "yes" if adj < 0.05 else "no"
        row["comparison_inference_scope"] = "working_model_only_purposive_sample_nonprobability"
    return partial


def dataset_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in p0.TARGET_DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        for domain in DOMAINS:
            statuses = [str(row[domain]) for row in subset]
            counts = Counter(statuses)
            n = len(statuses)
            if not n:
                continue
            ci = p0.wilson(counts["yes"], n)
            output.append({
                "dataset": dataset, "metric": domain, "N": n,
                "yes": counts["yes"], "no": counts["no"], "unknown": counts["unknown"],
                "main_percent": pct(counts["yes"] / n),
                "main_wilson95_percent": f"{pct(ci[0])}–{pct(ci[1])}",
                "upper_percent_unknown_as_yes": pct((counts["yes"] + counts["unknown"]) / n),
            })
    return output


def main() -> None:
    if sha256(PURPOSIVE_SURVEY) != PURPOSIVE_SHA256:
        raise ValueError("purposeful P0 survey changed after P0.5 preregistration")
    random_rows = p0.read_csv(RANDOM_SURVEY)
    purposive_rows = p0.read_csv(PURPOSIVE_SURVEY)
    if len(random_rows) != 30 or len(purposive_rows) != 32:
        raise ValueError(f"expected random n=30 and purposive n=32, got {len(random_rows)} and {len(purposive_rows)}")
    if any(row["verification_status"] not in {"verified_agree", "verified_corrected"} for row in random_rows):
        raise ValueError("P0.5 survey contains unverified rows")
    validate_target_datasets(random_rows, "random")
    validate_target_datasets(purposive_rows, "purposive")

    _, order = read_order()
    if len(order) != 331 or [int(row["draw_rank"]) for row in order] != list(range(1, 332)):
        raise ValueError("random order must retain the complete ranked 331-item frame")
    reached = [row for row in order if row["selection_status"] != "not_reached"]
    selected = [row for row in reached if row["screening_outcome"] == "include"]
    excluded = [row for row in reached if row["screening_outcome"] == "exclude"]
    if [int(row["draw_rank"]) for row in reached] != list(range(1, len(reached) + 1)):
        raise ValueError("screening must be a continuous prefix of the frozen random order")
    if len(reached) != 34 or len(selected) != 30 or len(excluded) != 4:
        raise ValueError("expected stop rank 34 with 30 eligible and 4 excluded candidates")
    if any(row["selection_status"] != "not_reached" for row in order[len(reached):]):
        raise ValueError("all rows after the stopping rank must remain not_reached")
    if any(not row["exclusion_code"].strip() for row in excluded):
        raise ValueError("every sequential exclusion requires an E-code")
    selected_identity = {
        (canonical_repository(row["canonical_repo_url"]), row["frozen_commit"])
        for row in selected
    }
    survey_identity = {
        (canonical_repository(row["repository_url"]), row["commit_sha"])
        for row in random_rows
    }
    if selected_identity != survey_identity:
        raise ValueError("random survey URL/commit identities do not match the 30 selected order rows")

    random_repo, random_status = repo_risk(random_rows, "random_probability_sample")
    purposive_repo, purposive_status = repo_risk(purposive_rows, "purposive_depth_sample")
    random_dataset = dataset_risk(random_rows)
    summaries = summary("random_probability_sample", random_status) + summary("purposive_depth_sample", purposive_status)
    comparisons = comparison(random_status, purposive_status)
    dataset_summaries = dataset_summary(random_dataset)

    flow = [
        {"stage": "frozen_pending_frame", "category": "all", "count": len(order)},
        {"stage": "sequential_screening", "category": "reached", "count": len(reached)},
        {"stage": "sequential_screening", "category": "included_complete_audit", "count": len(selected)},
        {"stage": "sequential_screening", "category": "excluded", "count": len(excluded)},
        {"stage": "sequential_screening", "category": "not_reached", "count": len(order) - len(reached)},
    ]
    for code, count in sorted(Counter(r["exclusion_code"] for r in excluded).items()):
        flow.append({"stage": "exclusion_code", "category": code, "count": count})

    write_csv(OUT / "random_repository_risk.csv", random_repo, list(random_repo[0]))
    write_csv(OUT / "purposive_repository_risk.csv", purposive_repo, list(purposive_repo[0]))
    write_csv(OUT / "random_repository_dataset_risk.csv", random_dataset, list(random_dataset[0]))
    write_csv(OUT / "sample_summary.csv", summaries, list(summaries[0]))
    write_csv(OUT / "sample_comparison.csv", comparisons, list(comparisons[0]))
    write_csv(OUT / "random_dataset_summary.csv", dataset_summaries, list(dataset_summaries[0]))
    write_csv(OUT / "sample_flow.csv", flow, ["stage", "category", "count"])

    label = {
        "split": "说话人非互斥/随机划分",
        "normalization": "归一化泄漏",
        "test_explicit": "显式测试集选模",
        "test_any_exposure": "任意测试集暴露",
        "augmentation": "增强进入测试路径",
        "single_split_single_seed": "单次划分单 seed",
        "variance_absent": "无方差报告",
    }
    exclusion_counts = Counter(row["exclusion_code"] for row in excluded)
    exclusion_text = "、".join(f"`{code}`={count}" for code, count in sorted(exclusion_counts.items()))
    md = [
        "# P0.5 随机概率样本报告",
        "",
        "论文主估计来自冻结 331 项 pending 框的种子化随机顺序，经顺序补抽得到的 30 个合格仓库；目的性 32 库仅作并排深度案例比较，不简单池化。",
        "",
        "## 抽样流",
        "",
        f"冻结框 331 项；顺序检查 draw ranks 1–{len(reached)} 后停止，其中完整纳入 30、排除 {len(excluded)}、未到达 {len(order) - len(reached)}。排除构成为 {exclusion_text}。停止位之后没有做 Stage-B，也没有把它们写成排除。",
        "",
        "30 个纳入单位在随机样本内等权。完整合格总体大小未知，故不做有限总体修正；目的性 32 库没有已知纳入概率，不能构造 inverse-probability 权重。",
        "",
        "## 随机样本七项主估计",
        "",
        "| 风险 | yes/no/unknown | 主率（Wilson 95% CI） | unknown 全作风险上界（Wilson 95% CI） | 可判定子集（Wilson 95% CI） | unknown 率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    random_summary = {r["metric"]: r for r in summaries if r["sample"] == "random_probability_sample"}
    purposive_summary = {r["metric"]: r for r in summaries if r["sample"] == "purposive_depth_sample"}
    for metric in DOMAINS:
        s = random_summary[metric]
        md.append(
            f"| {label[metric]} | {s['yes']}/{s['no']}/{s['unknown']} | "
            f"{s['main_percent']}% ({s['main_wilson95_percent']}%) | "
            f"{s['upper_percent_unknown_as_yes']}% ({s['upper_wilson95_percent']}%) | "
            f"{s['yes']}/{s['known_N']} = {s['known_only_percent']}% ({s['known_only_wilson95_percent']}%) | "
            f"{s['unknown_percent']}% |"
        )
    md += [
        "",
        "主率把 unknown 留在总分母并按未确认风险处理；上界把 unknown 全按风险。",
        "",
        "## 目的性样本并排比较",
        "",
        "| 风险 | 随机 yes/no/unknown；主率 | 目的性 yes/no/unknown；主率 | 随机−目的性 pp（Newcombe 95% CI） | unknown 差 pp | Fisher p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        md.append(
            f"| {label[row['metric']]} | {row['random_yes']}/{row['random_no']}/{row['random_unknown']}; {row['random_main_percent']}% | "
            f"{row['purposive_yes']}/{row['purposive_no']}/{row['purposive_unknown']}; {row['purposive_main_percent']}% | "
            f"{row['main_difference_percentage_points']} ({row['newcombe95_percentage_points']}) | "
            f"{row['unknown_difference_percentage_points']} | {row['fisher_two_sided_p']} | {row['holm_adjusted_p']} |"
        )
    md += [
        "",
        "下列敏感性列补齐双方的 unknown 上界与可判定子集率；上界差不做第二套显著性检验。",
        "",
        "| 风险 | 随机上界 / 可判定率 | 目的性上界 / 可判定率 | 上界差 pp |",
        "|---|---:|---:|---:|",
    ]
    for row in comparisons:
        rs = random_summary[row["metric"]]
        ps = purposive_summary[row["metric"]]
        md.append(
            f"| {label[row['metric']]} | {rs['upper_percent_unknown_as_yes']}% / {rs['known_only_percent']}% | "
            f"{ps['upper_percent_unknown_as_yes']}% / {ps['known_only_percent']}% | "
            f"{row['upper_difference_percentage_points']} |"
        )
    md += [
        "",
        "Newcombe 使用独立比例差的 hybrid-score method 10（无连续性校正），方向为随机减目的性；Fisher 双侧采用 probability-ordering，2×2 为 yes 对 no+unknown；七项组成一个 Holm family。",
        "",
        "目的性 32 库不是概率样本，因此其 Wilson、Newcombe、Fisher 和 Holm 没有设计型总体覆盖率或Ⅰ类错误保证。这里按预注册保留它们，但只解释为把目的性样本暂视作 iid binomial 时的**工作模型诊断**，不宣称两个总体显著不同；`holm_adjusted_p` 是否低于 0.05 也不升级为总体推断。无差异同样不能证明目的性样本无偏，且 n=30/32 的比较功效有限。",
        "",
        "## 合并规则与权重",
        "",
        "逐数据集结局按原 P0 字典编码；仓库内按 `yes > unknown > no` 合并，任一目标语料为 yes 即为仓库 yes。`evidence_strength=low` 的仓库在全部七项主汇总中降为 unknown。可判定子集率可能受信息性 unknown 影响，不能替代总分母主率。",
        "",
        "论文主估计只使用随机样本 n=30。目的性样本 n=32 是深度案例层；两者不简单池化为 62 库总体，不报告池化 Wilson 区间，也不声称存在可用的加权总体估计。",
        "",
        "## 推断边界",
        "",
        "随机性只成立于冻结 GitHub/arXiv pending 框及顺序资格筛选；不覆盖无公开代码、未被检索捕获或只在其他平台托管的工作。资格/可访问性若与风险结局相关，仍可能造成偏差。目的性与随机样本的差异只是选择机制相关描述，不作因果解释。Papers with Code 与 Semantic Scholar 在检索日均未贡献可检查结果，详见 P0.6。",
        "",
        "## 复核身份",
        "",
        "30/30 的复核是同一自动化代理系统不同审计遍次的交叉核对，不是两名独立人类编码者。作者盲编码 8–10 库仍为待作者事项。",
    ]
    (ROOT / "P0_5_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"random_n=30 purposive_n=32 reached={len(reached)} excluded={len(excluded)}")


def read_order() -> tuple[list[str], list[dict[str, str]]]:
    with ORDER.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


if __name__ == "__main__":
    main()

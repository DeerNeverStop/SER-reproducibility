"""Preserve B1 frozen outputs and export complete-column tables/readable report.

The frozen CSV writer used its first six-class row as the column schema. This
companion export restores the already-computed remaining class-support columns
from preserved per-repeat JSON, without changing any frozen source or raw file.

Reproduction from the repository root (CPU thread limit is 2):
    python v3/deploy/analysis/render_b1_report.py

Requires the frozen B1 summary.csv/endpoints.json and all 72 saved repeat JSONs
in v3/deploy/work/B1_analysis. The preceding frozen scoring command is:
    python -m v3.deploy.review_budget run --out v3/deploy/results/B1 --work v3/deploy/work/B1_analysis

This reruns the export, not model fitting; it regenerates companion CSVs,
provenance and the pre-audit Markdown report. Existing independent audit receipts
must be rechecked against regenerated artifacts; this script never grants PASS.
The historical receipt identifies the original external writer; this repository
copy only changes repository discovery and documents reproduction.
Original writer SHA256: 2140ac4946ca1f4dccc2c906872cab746045f47026f23da06f11cdb4d608b28a
"""
import csv
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from v3.deploy import common as c, review_budget as rb


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pct(v):
    return "NA" if not np.isfinite(float(v)) else f"{100*float(v):.2f}"


def main():
    c.cpu_guard(2)
    out = REPO / "v3/deploy/results/B1"
    work = REPO / "v3/deploy/work/B1_analysis"
    original_sha = c.sha256_file(out / "summary.csv")
    original_ep_sha = c.sha256_file(out / "endpoints.json")
    original = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
    result = c.read_json(out / "endpoints.json")
    groups, source_shas, raw_metrics, raw_speakers = {}, {}, [], []
    for name in result["inputs"]["replicates"]:
        level, model, rep = name.split("|")
        p = work / f"{level}__{model}__{rep}.json"
        source_shas[p.relative_to(REPO).as_posix()] = c.sha256_file(p)
        scored = c.read_json(p)
        for key in ("budgets", "fold_budgets"):
            scored[key] = {float(budget): value for budget, value in scored[key].items()}
            for budget, value in scored[key].items():
                variant = "global_oof" if key == "budgets" else "within_fold"
                raw_metrics.append(dict(level=level, model=model, replicate=rep, variant=variant, **value["metrics"]))
                for speaker, stats in value["speakers"].items():
                    raw_speakers.append(dict(level=level, model=model, replicate=rep, variant=variant, b=budget, speaker=speaker, **stats))
        groups.setdefault((level, model), {})[rep] = scored
    full, fold, all_endpoints, person_coverage = [], [], [], {}
    for (level, model), reps in sorted(groups.items()):
        classes = c.class_table(c.load_manifest(c.LEVELS[level]))
        agg = rb.aggregate_group(level, model, reps, len(classes), classes)
        full += agg["summary"]
        fold += agg["fold_sensitivity"]["summary"]
        all_endpoints += agg["endpoints"]
        for budget in rb.BUDGETS:
            person_coverage[(level,model,budget)] = float(np.mean([r["coverage"] for r in agg["per_speaker"] if r["b"]==budget]))
    # Verify that the companion export preserves every existing primary CSV number.
    original_by = {(r["level"], r["model"], float(r["b"])): r for r in original}
    numeric_checked = 0
    missing = set()
    for row in full:
        saved = original_by[(row["level"], row["model"], row["b"])]
        missing.update(set(row) - set(saved))
        for k, v in saved.items():
            if k in ("level", "model"):
                assert row[k] == v
            else:
                a, b = float(row[k]), float(v)
                assert (np.isnan(a) and np.isnan(b)) or abs(a-b) <= 1e-12, (k,a,b)
                numeric_checked += 1
    assert c.canonical(all_endpoints) == c.canonical(result["endpoints"])
    write_csv(out / "summary_complete.csv", full)
    write_csv(out / "fold_sensitivity_summary.csv", fold)
    write_csv(out / "per_repeat_metrics_complete.csv", raw_metrics)
    write_csv(out / "per_repeat_speakers_complete.csv", raw_speakers)
    c.atomic_write_json(out / "complete_export_provenance.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "program": c.PROGRAM,
        "original_summary_sha256": original_sha, "original_endpoints_sha256": original_ep_sha,
        "omitted_original_csv_columns": sorted(missing),
        "original_csv_numbers_preserved": numeric_checked, "all_84_primary_endpoints_preserved_exactly": True,
        "source_replicate_json_sha256": source_shas,
        "writer": str(Path(__file__).resolve()), "writer_sha256": c.sha256_file(Path(__file__)),
        "frozen_scorer_sha256": c.sha256_file(REPO / "v3/deploy/review_budget.py"),
        "note": "Export-only companion: reconstructs frozen aggregate from saved per-repeat scores; does not change raw files, frozen code, original outputs, selection or estimands. This is not the independent scientific verification."})
    ep = {(e["level"], e["model"], e["quantity"]): e for e in result["endpoints"]}
    fold_by = {(r["level"], r["model"], r["b"]): r for r in fold}
    primary = []
    for row in full:
        if row["b"] != .2:
            continue
        f = fold_by[(row["level"], row["model"], .2)]
        primary.append(dict(row, speaker_equal_coverage=person_coverage[(row["level"],row["model"],.2)],
                            fold_accepted_error=f["accepted_error"], fold_actual_review_rate=f["actual_review_rate"],
                            fold_speakers_below_0_70=f["speakers_below_0.70"],
                            accepted_error_ci_low=ep[(row["level"],row["model"],"accepted_error")]["ci95"][0],
                            accepted_error_ci_high=ep[(row["level"],row["model"],"accepted_error")]["ci95"][1],
                            capture_ci_low=ep[(row["level"],row["model"],"error_capture_rate")]["ci95"][0],
                            capture_ci_high=ep[(row["level"],row["model"],"error_capture_rate")]["ci95"][1]))
    write_csv(out / "primary_budget20_complete.csv", primary)
    low_shares = [r["speakers_below_0.70"] / r["n_speakers"] for r in primary]
    max_fold_risk_delta = max(abs(r["fold_accepted_error"]-r["accepted_error"]) for r in primary)
    zero_rows = [r for r in raw_speakers if r["n_acc"] == 0]
    lines = ["# B1 固定复核额度分析", "",
        "状态：冻结主评分器已完成。独立验证另行验收；已保留并披露冻结验证器的逐人缺失值口径问题，不能把本报告称为独立验证已通过。", "",
        "来源为旧 v2 的完整 GG 预测：360 个单元、72 个五折 OOF 重复、21 个语料×模型组；CREMA-D 100 个，RAVDESS 与 SUBESCO-980 各 130 个。CNN 在后两语料保留全部 3×3 划分/种子组合；其他 CTRL 每组 3 个重复，FT 每语料 2 个种子。这里没有重新训练模型，也没有使用正在运行的 N14R 预测。", "",
        "这是一项批处理描述：使用待处理批次的无标签置信度排序，复核最低 floor(b×n) 条；并列按路径 SHA 排序。它不等同于用独立校准集设置在线阈值，也不保证客服场景中的错误率。", "",
        "## 预定 20% 主结果", "",
        f"21 个模型/语料组合中，约 20% 的人工复核量捕获 {min(p['error_capture_rate'] for p in primary)*100:.2f}%–{max(p['error_capture_rate'] for p in primary)*100:.2f}% 的原始错误，仍有 {min(1-p['error_capture_rate'] for p in primary)*100:.2f}%–{max(1-p['error_capture_rate'] for p in primary)*100:.2f}% 错误留在接受集。接受集错误率为 {min(p['accepted_error'] for p in primary)*100:.2f}%–{max(p['accepted_error'] for p in primary)*100:.2f}%。各组均低于各自全体录音错误率，但残余错误仍不可忽略。", "",
        f"总体约 80% 的接受覆盖没有保证每个人达到同等覆盖：每次重复中覆盖不足 70% 的人数平均占 {min(low_shares)*100:.2f}%–{max(low_shares)*100:.2f}%。20% 主预算中没有零接受说话人。逐折分别分配同一名义预算时，21 组的低覆盖人数均减少或持平，接受错误率与全局 OOF 排序的最大绝对差为 {max_fold_risk_delta*100:.2f} 个百分点。逐折 floor 会使实际总复核额略有不同，完整表保留双方实际额度；这是预定敏感性分析，不是独立因果对照。", "",
        "下表错误率和捕获率单位为 %；括号为固定预测/固定选择下的 95% 说话人聚类 bootstrap 区间。低覆盖人数为先在每次重复计数、再平均，因此可为小数；不能解读为同一固定人群或新增独立样本。"]
    labels = {"cremad":"CREMA-D（91 人）", "ravdess":"RAVDESS（24 人）", "subesco_980":"SUBESCO-980（20 人）"}
    for lv in labels:
        lines += ["", "### "+labels[lv], "", "| 模型 | 原错误率 | 接受错误率 [95% CI] | 捕获错误 [95% CI] | <70% 覆盖人数 | 逐折接受错误率 | 逐折低覆盖人数 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in primary:
            if r["level"] != lv:
                continue
            lines.append(f"| {r['model']} | {pct(r['overall_error_rate'])} | {pct(r['accepted_error'])} [{pct(r['accepted_error_ci_low'])}, {pct(r['accepted_error_ci_high'])}] | {pct(r['error_capture_rate'])} [{pct(r['capture_ci_low'])}, {pct(r['capture_ci_high'])}] | {r['speakers_below_0.70']:.2f} | {pct(r['fold_accepted_error'])} | {r['fold_speakers_below_0_70']:.2f} |")
    lines += ["", "### 20% 预算下的逐人风险与覆盖", "",
              "这里“人均覆盖/风险”为说话人等权量；min/p10 先在每次重复计算后再平均，不是把所有重复合成后挑同一个最差人。百分数均为 %。", "",
              "| 语料 | 模型 | 人均接受风险 | 人均覆盖 | 平均最小覆盖 | 平均 p10 覆盖 | 全局实际复核 | 逐折实际复核 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append(f"| {r['level']} | {r['model']} | {pct(r['speaker_equal_accepted_error'])} | {pct(r['speaker_equal_coverage'])} | {pct(r['speaker_coverage_min'])} | {pct(r['speaker_coverage_p10'])} | {pct(r['actual_review_rate'])} | {pct(r['fold_actual_review_rate'])} |")
    lines += ["", "## 完整预算与缺失值", "",
        "10%、20%、30% 三档全部保留在 summary_complete.csv 与 fold_sensitivity_summary.csv；逐人等额变体、说话人等权风险、覆盖 min/p10、AURC（0–1）与接受类别支持也在表内。全局/逐折原始每次重复指标和逐人计数分别在 per_repeat_metrics_complete.csv、per_repeat_speakers_complete.csv。", "",
        "21 组的平均接受风险在 10%→20%→30% 预算中均下降；增加人工复核也伴随自动接受覆盖下降。这是预定三个工作点的观察，不是全阈值单调性保证。下表保留两个次预算及 AURC。AURC 为全覆盖范围的离散风险均值，尺度 0–1，越小表示该排序下整体风险越低；它同时受分类错误总量和置信度排序质量影响，不能代替某一固定预算风险。", "",
        "| 语料 | 模型 | 10% 接受风险 | 10% 捕获错误 | 30% 接受风险 | 30% 捕获错误 | AURC (0–1) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    by = {(r["level"],r["model"],r["b"]):r for r in full}
    for r in primary:
        low,high=by[(r["level"],r["model"],.1)],by[(r["level"],r["model"],.3)]
        lines.append(f"| {r['level']} | {r['model']} | {pct(low['accepted_error'])} | {pct(low['error_capture_rate'])} | {pct(high['accepted_error'])} | {pct(high['error_capture_rate'])} | {r['aurc']:.4f} |")
    lines += ["",
        f"全局及逐折所有保存的逐人/重复/预算记录共 {len(raw_speakers):,} 行，零接受记录 {len(zero_rows)} 行。全局 SUBESCO-980/CNN 在 b=0.30 时，F09 的 9 次重复中有 1 次无接受录音，所以严格跨重复聚合风险为 NA（8/9 次可定义），覆盖值仍保留。不能用其余 8 次代替预定 9 次，也不能把这次风险填零。", "",
        "接受集 UAR 需要全类别支持；缺支持时为 NA。跨重复 UAR 的平均仅针对支持完整的可定义重复，不能称完整固定类总体 UAR。原 summary.csv 的六类首行列模板没有导出高类别索引支持列（accepted_support_6/7）；完整信息仍在逐重复 JSON。本次仅新增完整列导出并保存来源 SHA，原 CSV/JSON 与冻结代码未改动。", "",
        "## 解释边界", "",
        "结果支持报告复核额度与残余错误、逐人覆盖分布的关系。它不能证明独立校准阈值在陌生人群中同样可信；该问题由 B2/FIX 回答。各模型和语料的预处理、类别集合、录音数、训练次数不同，不把跨模型或跨语料差值归因为架构或真实部署收益。未对方向或显著性筛选组合，全部 21 组和三个预算保留。", "",
        "文件索引：endpoints.json（原始 84 主端点及全部逐折敏感性）；summary.csv（原始未改动 CSV）；summary_complete.csv（完整列）；primary_budget20_complete.csv（主表）；rejected_class_mix.csv；per_speaker.csv；complete_export_provenance.json（原输出及 72 个来源 JSON 的 SHA）。", ""]
    (out / "B1_ANALYSIS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    assert c.sha256_file(out / "summary.csv") == original_sha and c.sha256_file(out / "endpoints.json") == original_ep_sha
    print(json.dumps({"groups":len(groups),"summary_rows":len(full),"fold_summary_rows":len(fold),"raw_metric_rows":len(raw_metrics),
                      "raw_speaker_rows":len(raw_speakers),"missing_columns_restored":sorted(missing),"original_numeric_fields_checked":numeric_checked,
                      "zero_accept_records":len(zero_rows),"max_fold_risk_delta_pp":max_fold_risk_delta*100,"report":str(out/'B1_ANALYSIS_ZH.md')}))


if __name__=="__main__":
    main()

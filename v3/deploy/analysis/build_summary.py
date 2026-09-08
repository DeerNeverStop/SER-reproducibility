"""Descriptive presentation only; never fits, selects, or edits frozen results.

Run from the repository root after all four frozen scorers. This file is
outside the frozen runtime inventory. Primary intervals are copied verbatim.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "analysis"
OUT.mkdir(exist_ok=True)
LEVELS = ("cremad", "ravdess", "subesco_980")
ENCODERS = ("wavlm_base_plus", "hubert_base", "wav2vec2_base")
MODELS = ("cnn", "ridge_wavlm_base_plus", "ridge_hubert_base", "ridge_wav2vec2_base", "wavlm_ft")
LABELS = {"cremad": "CREMA-D", "ravdess": "RAVDESS", "subesco_980": "SUBESCO-980",
          "cnn": "CNN", "wavlm_base_plus": "WavLM", "hubert_base": "HuBERT", "wav2vec2_base": "wav2vec2",
          "ridge_wavlm_base_plus": "WavLM-Ridge", "ridge_hubert_base": "HuBERT-Ridge",
          "ridge_wav2vec2_base": "wav2vec2-Ridge", "wavlm_ft": "WavLM-FT"}
COLORS = {"cremad": "#176b87", "ravdess": "#b86125", "subesco_980": "#7855a4"}

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def pp(e):
    return f"{100*e['estimate']:+.2f} [{100*e['ci95'][0]:+.2f}, {100*e['ci95'][1]:+.2f}]"

def label(level, model):
    return f"{LABELS[level]} / {LABELS[model]}"

def save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

data = {m: read_json(RESULTS / m / "endpoints.json") for m in ("A", "B1", "B2", "B2-FIX")}
assert [len(data[m]["endpoints"]) for m in data] == [903, 84, 105, 36]
all_endpoints = []
for module, d in data.items():
    for e in d["endpoints"]:
        row = {"module": module, "original_scale": "UAR percentage points; harm endpoints are proportions" if module == "A" else "proportion"}
        row.update({k: v for k, v in e.items() if not isinstance(v, (dict, list))})
        interval = e.get("ci95") or [None, None]
        row.update(ci95_low=interval[0], ci95_high=interval[1])
        all_endpoints.append(row)
write_csv("all_1128_endpoints.csv", all_endpoints)

b2_index = {(e["level"], e["model"], e["quantity"]): e for e in data["B2"]["endpoints"]}
fix_index = {(e["level"], e["model"], e["quantity"]): e for e in data["B2-FIX"]["endpoints"]}
primary = []
for level in LEVELS:
    for model in MODELS:
        row = dict(level=level, model=model)
        for q in ("coverage", "accepted_error", "gap_coverage", "gap_accepted_error"):
            e = b2_index[level, model, q]
            for k, v in (("pp", e["estimate"]), ("lo_pp", e["ci95"][0]), ("hi_pp", e["ci95"][1])):
                row[q + "_" + k] = 100 * v
            row[q + "_n"] = e["n"]
        # Review deviation is exactly minus coverage; not a new endpoint test.
        row["review_deviation_pp"] = -row["coverage_pp"]
        primary.append(row)
write_csv("B2_primary_all15.csv", primary)

absolute = []
for a in data["B2"]["absolute"]:
    row = {k: a[k] for k in ("level", "model", "cell", "n_r", "n_speakers", "n_units")}
    for q in ("coverage", "accepted_error", "gap_coverage", "gap_accepted_error", "coverage_secondary", "accepted_error_secondary", "gap_accepted_error_secondary"):
        e = a["speaker." + q]
        row.update({q + "_pct_or_pp": 100 * e["estimate"], q + "_lo": 100 * e["ci95"][0], q + "_hi": 100 * e["ci95"][1], q + "_n": e["n"]})
    row["realized_review_pct"] = 100 - row["coverage_pct_or_pp"]
    row["review_minus_20_pp"] = row["realized_review_pct"] - 20
    row.update({"coverage_profile_" + k + ("_pct" if k in ("min", "p10") else ""): (100*v if k in ("min", "p10") else v)
                for k, v in a["speaker.coverage_profile"].items()})
    absolute.append(row)
write_csv("B2_absolute_all30.csv", absolute)
write_csv("B2_secondary_all45.csv", [
    {**{k:v for k,v in e.items() if k not in ("estimate", "ci95")}, "unit":"percentage_points",
     "estimate_pp":100*e["estimate"], "ci95_low_pp":100*e["ci95"][0], "ci95_high_pp":100*e["ci95"][1]}
    for e in data["B2"]["endpoints"] if e["quantity"].endswith("secondary")])
pooled = []
for a in data["B2"]["absolute"]:
    for p in a["pooled_by_repeat"]:
        pooled.append({**{k: a[k] for k in ("level", "model", "cell")}, **p})
write_csv("B2_pooled_by_repeat.csv", pooled)

fixed = []
for level in LEVELS:
    for enc in ENCODERS:
        model = "ridge_" + enc
        row = dict(level=level, model=model)
        for q in ("review_deviation", "gap_coverage", "accepted_error", "gap_accepted_error"):
            e = fix_index[level, model, q]
            row.update({q + "_pp": 100*e["estimate"], q + "_lo_pp": 100*e["ci95"][0], q + "_hi_pp": 100*e["ci95"][1], q + "_n": e["n"]})
        # Descriptive absolute means from scorer's per-person records; no new CI.
        units = [u for u in data["B2-FIX"]["units"] if (u["level"], u["model"]) == (level, model)]
        for role in ("seen", "new"):
            persons = defaultdict(list)
            for u in units:
                for speaker, metrics in u[role]["speakers"].items():
                    persons[speaker].append(metrics)
            assert all(len(v) == 3 for v in persons.values())
            for q in ("coverage", "accepted_error", "gap_accepted_error"):
                row[f"{role}_{q}_pct_or_pp_no_ci"] = 100 * np.mean([np.mean([m[q] for m in v]) for v in persons.values()])
        assert np.isclose(row["seen_accepted_error_pct_or_pp_no_ci"] - row["new_accepted_error_pct_or_pp_no_ci"], row["accepted_error_pp"], atol=1e-10)
        fixed.append(row)
write_csv("B2_FIX_primary_all9.csv", fixed)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.1, 1]})
for ax, index, pairs, title in (
    (axes[0], b2_index, [(l,m) for l in LEVELS for m in MODELS], "B2: complete GR minus GG protocol"),
    (axes[1], fix_index, [(l,"ridge_"+m) for l in LEVELS for m in ENCODERS], "B2-FIX: seen minus new calibration speakers"),
):
    for i, (level, model) in enumerate(pairs):
        e = index[level, model, "gap_accepted_error"]
        x, lo, hi = 100*np.asarray([e["estimate"], *e["ci95"]])
        ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", color=COLORS[level], capsize=3)
    ax.set(yticks=range(len(pairs)), yticklabels=[label(*p) for p in pairs], title=title, xlabel="Change in test minus calibration accepted error (pp)")
    ax.axvline(0, color="#888888", linewidth=1, linestyle="--")
    ax.set_xlim(-7, 22)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
fig.suptitle("Calibration optimism varies across corpora and protocols", fontsize=16, y=1.01)
fig.text(.5, -.015, "Frozen estimates and 95% speaker bootstrap intervals; conditional on fitted models and thresholds. All groups shown.", ha="center", fontsize=10)
fig.tight_layout(w_pad=3)
save(fig, "calibration_gap_all_groups")

b1 = read_csv(RESULTS / "B1" / "primary_budget20_complete.csv")
assert len(b1) == 21
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
for ax, level in zip(axes, LEVELS):
    rows = [r for r in b1 if r["level"] == level]
    for i, r in enumerate(rows):
        before, after = 100*float(r["overall_error_rate"]), 100*float(r["accepted_error"])
        lo, hi = 100*float(r["accepted_error_ci_low"]), 100*float(r["accepted_error_ci_high"])
        ax.plot([after, before], [i, i], color="#bbbbbb")
        ax.plot(before, i, "x", color="#555555", label="All recordings" if i == 0 else None)
        ax.errorbar(after, i, xerr=[[after-lo], [hi-after]], fmt="o", color=COLORS[level], capsize=2, label="Accepted after 20% review" if i == 0 else None)
    ax.set(title=LABELS[level], xlabel="Recording-pooled error (%)", yticks=range(len(rows)), yticklabels=[r["model"].replace("_base_plus", "").replace("_base", "") for r in rows])
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.suptitle("B1: residual errors after a fixed review quota", fontsize=16)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .025), ncol=2, frameon=False)
fig.text(.5, -.015, "21 groups; means over OOF repeats. Accepted-risk 95% CIs cluster speakers. Historical models differ from B2.", ha="center", fontsize=10)
fig.tight_layout(rect=(0, .09, 1, 1))
save(fig, "review_budget20_all_groups")

lines = ["# B2 与 B2-FIX：完整校准分析", "", "所有差值均以百分点计，括号为原冻结95%说话人bootstrap区间。B2含390单元/15组，FIX含135单元/9组；FT仅一次五折重复。未按结果筛选组合。", "", "## B2：完整 GR−GG 管线对照", "", "GR和GG使用同一测试集合，但训练集合不同；不能将差异全部归因于校准说话人重叠。coverage为自动接受覆盖，错误率指接受部分的错误率。gap为测试错误率减去校准错误率，正的gap差表示GR的校准估计相对更加乐观。", "", "|语料 / 模型|接受覆盖差|接受错误差|测试−校准错误gap差|", "|---|---:|---:|---:|"]
for level in LEVELS:
    for model in MODELS:
        lines.append("|" + label(level,model) + "|" + "|".join(pp(b2_index[level,model,q]) for q in ("coverage","accepted_error","gap_accepted_error")) + "|")
lines += ["", "B2的15个gap差点估计全部为正（+0.44至+18.44点），但SUBESCO/CNN区间跨零；接受错误差正负均有（−9.03至+2.49点）。因此不能写成‘GR必然使测试识别更差’。RAVDESS/CNN甚至接受错误下降9.03点，同时校准乐观gap增大7.86点，两者并不矛盾。", "", "主阈值在校准集按20%分位数设置。30个条件的测试逐人平均实际复核率为15.44%–23.76%；所有主量保留全体说话人，没有无接受风险缺失。每个条件的绝对值及其CI在B2_absolute_all30.csv。录音池化量另存B2_pooled_by_repeat.csv，不能与逐人等权量混用。", "", "## B2-FIX：同一模型与测试预测，仅更换匹配校准集", "", "seen/new两条件共享模型权重、标准化器和测试logits；校准录音量及句子×类别支持匹配，实际seen校准录音已从训练集中移除。它比B2更直接检查固定模型下校准人群差异，但仅涵盖三种Ridge；不能替CNN/FT作同样的因果断言。", "", "|语料 / 模型|实际复核率差 seen−new|接受错误差 seen−new|测试−校准错误gap差|", "|---|---:|---:|---:|"]
for level in LEVELS:
    for enc in ENCODERS:
        model = "ridge_"+enc
        lines.append("|" + label(level,model) + "|" + "|".join(pp(fix_index[level,model,q]) for q in ("review_deviation","accepted_error","gap_accepted_error")) + "|")
lines += ["", "固定模型对照的8/9个gap差为正（+2.04至+15.63点）；SUBESCO/WavLM为−2.59点，区间[−4.96,−0.38]，方向相反且必须保留。RAVDESS三种Ridge最一致，gap增加13.27–15.63点，测试接受错误本身仅增加0.16–0.88点：主要变化体现在校准对风险的估计，而非同等幅度的测试风险恶化。seen复核率在9组点估计均更低（−0.33至−5.03点），对应较宽松接受；其接受错误差均为小的正点估计，但并非各组区间都排除零。", "", "## 预定次要风险阈值与缺失", "", "次要阈值把校准接受错误目标设为原校准错误的一半，不是20%固定复核量，也不是分布无关风险保证。B2中其逐人平均接受覆盖仅6.63%–59.70%。严格要求全部重复和GR/GG风险均有定义时，SUBESCO/HuBERT-Ridge只剩2/20人，CNN为3/20，WavLM-Ridge为6/20，wav2vec2-Ridge为8/20；RAVDESS部分Ridge也只有10–19/24人。不能把这些条件性风险比较推广成全体人的改善，不能因未接受而将风险置零。45个次要端点与有效n完整保留B2_secondary_all45.csv，覆盖端点仍保留全体人。", "", "所有区间条件于固定训练模型和已选校准阈值；未重训模型bootstrap，未计入完整训练/阈值选择不确定性。20人与24人语料的泛化证据有限，且已有先前结果暴露，当前为估计与描述，不以未校正的多端点区间作为验证性显著结论。", ""]
(OUT / "B2_ANALYSIS_ZH.md").write_text("\n".join(lines), encoding="utf-8")

paths = [RESULTS / m / "endpoints.json" for m in data] + [RESULTS / "B1" / "primary_budget20_complete.csv"]
receipt = {"purpose": "presentation of complete frozen scores; no fitting or endpoint selection", "endpoint_count": len(all_endpoints),
           "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "scale_note": "Original endpoint values retained in all_1128_endpoints.csv; B2/FIX presentation converts proportions to percentage points.",
           "verification": "See separate original frozen and supplemental audit receipts; presentation does not override them."}
(OUT / "presentation_provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(f"Wrote {len(all_endpoints)} endpoints, complete B2/FIX tables, two scientific figures, and Chinese analysis to {OUT}")

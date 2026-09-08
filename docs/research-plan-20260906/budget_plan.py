"""Offline budget arithmetic: no GPU, model loading, cloud or network calls."""
from pathlib import Path
from statistics import NormalDist
import json, math

outdir = Path(__file__).resolve().parent
P = {
 "date":"2026-09-06", "status":"planning scenarios; not measured new runtimes",
 "gpu":"single RTX 5090 sequential jobs", "hourly_rate_USD":0.99,
 "rate_basis":"Runpod Secure Cloud on-demand catalog snapshot",
 "platform_comparison":{"autodl_5090_CNY_h":2.78,"USD_to_CNY_budget_assumption":6.7},
 "E4":{"train_language_conditions":3,"speaker_folds":5,"draws":3,"CNN_seeds":2,"Ridge_seeds":1,"status":"optional language extension; not GPU-priced"},
 "retry_slowdown_multiplier":1.5,
 "E0_E1_E2pilot_allocated_GPU_hours":[3,6],
 "E2":{"pilot_CNN":9,"pilot_Ridge":9,"core_CNN":270,"core_Ridge":270,"optional_FT":180,
       "CNN_seconds_scenario":[12,30],"FT_seconds_scenario":[75,180],
       "feature_GPU_hours":[0.5,1.5],"setup_transfer_GPU_hours":[2,4]},
 "E3":{"arms":6,"synthetic_arms":3,"folds":5,"draws":3,"accepted_clips_per_arm_split":576,
       "seconds_per_clip_assumed":5,"acceptance_probability_assumed":0.7,
       "one_time_anchor_candidates":96,"anchor_emotions":6,"anchor_seconds_assumed":5,
       "generation_RTF_scenarios":[0.3,1,3,10],"QA_RTF_all_attempts_assumed":0.05,
       "feature_RTF_accepted_training_audio_assumed":0.05,"conventional_aug_audio_hours_assumed":12,
       "CNN":90,"Ridge":90,"optional_FT":60,"CNN_seconds_assumed":40,"FT_seconds_assumed":150,
       "setup_transfer_GPU_hours":4,"technical_benchmark_attempted_clips":100,
       "benchmark_GPU_hour_cap_proposal":2,
       "pilot_accepted_clips_per_synthetic_arm":120,"pilot_synthesis_splits":1,
       "pilot_CNN":18,"pilot_Ridge":18,"pilot_rating_clips":480,"formal_rating_clips":1440,
       "raters":3,"rating_seconds_assumed":30},
 "G":{"strategies":2,"exposures":2,"blocks":5,"draws":3,"models":2,"distinct_query_speakers":60},
 "storage":{"network_volume_GB":200,"USD_per_GB_month":0.07,"days_per_month_assumed":30}
}
result = {"parameters":P}
m = P["retry_slowdown_multiplier"]
rate = P["hourly_rate_USD"]
a = P["E2"]
core, full = [], []
for j in range(2):
    cnn = a["core_CNN"]*a["CNN_seconds_scenario"][j]/3600
    ft = a["optional_FT"]*a["FT_seconds_scenario"][j]/3600
    feature, setup = a["feature_GPU_hours"][j], a["setup_transfer_GPU_hours"][j]
    core.append((cnn+feature)*m+setup)
    full.append((cnn+ft+feature)*m+setup)
result["E2"] = {
 "core_fits":a["core_CNN"]+a["core_Ridge"],
 "with_FT_fits":a["core_CNN"]+a["core_Ridge"]+a["optional_FT"],
 "core_GPU_hours":core,"with_FT_GPU_hours":full,"with_FT_GPU_USD":[x*rate for x in full],
 "E0_E1_pilot_plus_core_GPU_hours":[core[j]+P["E0_E1_E2pilot_allocated_GPU_hours"][j] for j in range(2)],
 "E0_E1_pilot_plus_full_GPU_hours":[full[j]+P["E0_E1_E2pilot_allocated_GPU_hours"][j] for j in range(2)]
}
a = P["E3"]
clips = a["synthetic_arms"]*a["folds"]*a["draws"]*a["accepted_clips_per_arm_split"]
accepted_h = clips*a["seconds_per_clip_assumed"]/3600
anchor_h = a["one_time_anchor_candidates"]*a["anchor_emotions"]*a["anchor_seconds_assumed"]/3600
attempted_h = (accepted_h+anchor_h)/a["acceptance_probability_assumed"]
qa = attempted_h*a["QA_RTF_all_attempts_assumed"]
feature_h = (accepted_h+a["conventional_aug_audio_hours_assumed"])*a["feature_RTF_accepted_training_audio_assumed"]
training = (a["CNN"]*a["CNN_seconds_assumed"]+a["optional_FT"]*a["FT_seconds_assumed"])/3600
rows = []
for rtf in a["generation_RTF_scenarios"]:
    gen = attempted_h*rtf
    total = (gen+qa+feature_h+training)*m+a["setup_transfer_GPU_hours"]
    rows.append({"assumed_RTF":rtf,"generation_GPU_h":gen,"QA_GPU_h":qa,
                 "CNN_and_optional_FT_GPU_h":training,"new_feature_GPU_h":feature_h,"total_GPU_h":total,
                 "GPU_USD":total*rate,"continuous_single_GPU_days":total/24})
result["E3"] = {
 "accepted_unique_clips_no_cross_split_reuse":clips,"accepted_audio_hours":accepted_h,
 "one_time_accepted_anchor_audio_hours":anchor_h,
 "expected_attempted_audio_hours":attempted_h,"core_fits":a["CNN"]+a["Ridge"],
 "with_optional_FT_fits":a["CNN"]+a["Ridge"]+a["optional_FT"],"scenarios":rows,
 "pilot_synthetic_clips":a["synthetic_arms"]*a["pilot_accepted_clips_per_synthetic_arm"],
 "pilot_rating_person_hours_raw":a["pilot_rating_clips"]*a["raters"]*a["rating_seconds_assumed"]/3600,
 "formal_rating_person_hours_raw":a["formal_rating_clips"]*a["raters"]*a["rating_seconds_assumed"]/3600
}
a = P["G"]
result["G"] = {"all_fits":math.prod(a[k] for k in ["strategies","exposures","blocks","draws","models"]),
               "CNN":60,"Ridge":60,"allocated_GPU_hours":[1,3]}
a = P["storage"]
month = a["network_volume_GB"]*a["USD_per_GB_month"]
result["storage_USD"] = {"30_days":month,"7_days":month*7/a["days_per_month_assumed"]}
z_ci = NormalDist().inv_cdf(.975)
z_power = z_ci+NormalDist().inv_cdf(.8)
result["precision_sensitivity"] = [
 {"n_speakers":n,"paired_SD_assumed_pp":sd,
  "approx_80pct_MDE_pp":z_power*sd/math.sqrt(n),"approx_CI_halfwidth_pp":z_ci*sd/math.sqrt(n)}
 for n in [20,24,60,91] for sd in [5,10]]
a=P["E4"]
panels=a["train_language_conditions"]*a["speaker_folds"]*a["draws"]
result["E4"]={"data_panels":panels,"Ridge_fits":panels,"CNN_fits":panels*a["CNN_seeds"],"total_fits":panels*(1+a["CNN_seeds"]),"GPU_hours":None}
price=P["platform_comparison"]["autodl_5090_CNY_h"]
result["AutoDL"]={"condition":"same RTX5090 task time assumed; actual throughput unmeasured","E0_E1_E2_full_CNY":[h*price for h in result["E2"]["E0_E1_pilot_plus_full_GPU_hours"]],"E3_scenarios":[{"assumed_RTF":r["assumed_RTF"],"GPU_CNY":r["total_GPU_h"]*price} for r in rows]}
result["warnings"] = [
 "RTF and acceptance rate are unknown. These are arithmetic scenarios, not runtime predictions.",
 "FT75-180s for E2 is a planning assumption, not a matched historical benchmark.",
 "QA RTF is charged on ALL attempted audio including rejected generations.",
 "New WavLM features budget covers 36h accepted synthetic and 12h conventional augmentation; assumed feature RTF .05, not benchmarked.",
 "Same GPU type does not ensure equal throughput; benchmark host and pipeline first.",
 "Independent-speaker normal approximation is not established power for overlapping CV models.",
 "Excluded: labor, dataset fees, taxes, local CPU time, language collection and development delays."
]
(outdir/"budget_plan.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
lines = [
 "# 可重算预算明细","",
 "仅做方案算术；未运行新训练或生成。单卡 RTX5090 Secure Cloud 计价 0.99 USD/h。RTF=生成用时/输出音频时长，未实测。","",
 "| 范围 | GPU 小时 | 说明 |","|---|---:|---|",
 "| E0/E1 + E2 技术先导 | 3–6 | 待测量的规划额度 |",
 f"| E2 核心540 fits | {core[0]:.2f}–{core[1]:.2f} | 270 CNN + 270 CPU Ridge |",
 f"| E2 含180 FT | {full[0]:.2f}–{full[1]:.2f} | 包含上一行，不重复相加 |",
 "", "## E3 生成速度敏感性","",
 "三合成臂×576条×15split×5秒=36小时合格训练音频，另有96音色×6情绪×5秒=0.8小时候选锚点。假设合格率0.7，含失败尝试约52.57小时音频。所有尝试计入自动质检。新合成36h与常规增强12h的WavLM特征按RTF .05另计2.4GPUh再乘缓冲。表内含90CNN+可选60FT、1.5倍缓冲、4小时准备。",
 "", "| 假设RTF | 原始生成GPU h | 合计GPU h | GPU美元 | 连续单卡天数 |",
 "|---:|---:|---:|---:|---:|"]
for row in rows:
    lines.append(f"| {row['assumed_RTF']:g} | {row['generation_GPU_h']:.2f} | {row['total_GPU_h']:.2f} | {row['GPU_USD']:.2f} | {row['continuous_single_GPU_days']:.2f} |")
lines += [
 "", "未测RTF和合格率前，不将本表当预计账单。RTF10是压力情景，不是建议支出。合格率由0.7降到0.35，生成和QA项翻倍，训练项不翻倍。",
 "", f"200GB network volume：30天 {month:.2f} USD；按30天月近似7天 {result['storage_USD']['7_days']:.2f} USD。其他磁盘、税费及人工另计。",
 "", "## 精度敏感性","",
 "理想独立说话人的正态近似，不能当交叉验证设计已经具备的统计功效。",
 "", "| 人数 | 假设逐人配对差SD(pp) | 约80%检出所需效应(pp) | 95%区间半宽(pp) |",
 "|---:|---:|---:|---:|"]
for row in result["precision_sensitivity"]:
    lines.append(f"| {row['n_speakers']} | {row['paired_SD_assumed_pp']} | {row['approx_80pct_MDE_pp']:.2f} | {row['approx_CI_halfwidth_pp']:.2f} |")
lines += ["", "## AutoDL 同时长换价与语言扩展", "", "AutoDL5090按2.78元/h，未验证跨平台吞吐。E0/E1/E2含FT约35–81元，E3 RTF1/3情景约266/704元；额外先导、人工与磁盘另计。", "", "E4为45个数据面板：45个Ridge+90个CNN=135 fits。确定性Ridge不重复训练seed；语言扩展GPU时长尚未定价。"]
(outdir/"budget_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({k:result[k] for k in ["E2","E3","G","storage_USD"]},ensure_ascii=False,indent=2))

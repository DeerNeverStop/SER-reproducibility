#!/usr/bin/env python3
"""Pre-scoring descriptive raw-audio budget audit; no SER results are inputs.

This was added during execution, before formal SER scoring. It is not a new
endpoint, hypothesis test, confidence interval, or causal adjustment. Selection
and training remain frozen. The input CSV summarizes original channel-mean
audio before resampling/trim/loudness processing; RMS and amplitude-threshold
fractions are not validated human-perceived quality scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics

PLAN_SHA = "8ea69b57b7030e4b431f503d8562de230cefed2cfa7d4d092f703e8b164cae4f"
POLICIES = ("U", "R", "C")
SOURCE_FILES = ("v3/speaker_coverage/extract_asv.py", "v2/ser_v2/features.py", "v3/deploy/engines_deploy.py", "v3/speaker_coverage/run.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def pin(path, expected):
    require(sha(path) == expected, "pinned file changed: " + Path(path).name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "duplicate/empty CSV header")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), "malformed CSV row")
    return rows


def local(root, name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and str(p) == name and not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and ":" not in name, "unsafe input-relative path")
    root = Path(root).resolve()
    target = (root / name).resolve()
    require(target.is_relative_to(root), "input escapes root")
    return target


def parse_quality(rows, manifest):
    output = {}
    for row in rows:
        require(set(row) == {"path", "sha256", "seconds", "original_sr", "rms", "clipped_fraction"}, "quality schema changed")
        path = row["path"]
        require(path not in output and path in manifest and row["sha256"] == manifest[path]["sha256"], "quality WAV identity differs from frozen manifest")
        seconds, sr, rms, fraction = float(row["seconds"]), int(row["original_sr"]), float(row["rms"]), float(row["clipped_fraction"])
        require(all(math.isfinite(x) for x in (seconds, rms, fraction)) and seconds > 0 and sr > 0 and rms >= 0 and 0 <= fraction <= 1,
                "invalid raw audio statistic")
        samples = round(seconds * sr)
        require(samples > 0 and abs(samples - seconds * sr) < 1e-6, "duration/sample-rate does not identify an integer sample count")
        clipped = round(fraction * samples)
        require(abs(clipped - fraction * samples) < 1e-6, "threshold proportion does not identify an integer sample count")
        output[path] = {"seconds": seconds, "original_sr": sr, "rms": rms, "clipped_fraction": fraction,
                        "samples": samples, "clipped_samples": clipped}
    return output


def panel_statistics(paths, quality):
    require(len(paths) == len(set(paths)) and paths and set(paths) <= quality.keys(), "missing/duplicated panel quality rows")
    records = [quality[path] for path in paths]
    seconds = math.fsum(r["seconds"] for r in records)
    samples = sum(r["samples"] for r in records)
    return {"recordings": len(records), "total_seconds": seconds, "total_hours": seconds / 3600,
            "mean_record_seconds": seconds / len(records),
            "mean_record_rms": math.fsum(r["rms"] for r in records) / len(records),
            "pooled_sample_rms": math.sqrt(math.fsum(r["rms"] ** 2 * r["samples"] for r in records) / samples),
            "amplitude_ge_0_999_sample_fraction": sum(r["clipped_samples"] for r in records) / samples,
            "mean_record_amplitude_ge_0_999_fraction": math.fsum(r["clipped_fraction"] for r in records) / len(records),
            "recordings_with_any_amplitude_ge_0_999": sum(r["clipped_samples"] > 0 for r in records), "original_samples": samples}


def panel_groups(plan, quality):
    grouped = defaultdict(list)
    require(len(plan["units"]) == 720 and Counter(u["block"] for u in plan["units"]) == {"core": 540, "ft": 180}, "wrong frozen unit counts")
    for unit in plan["units"]:
        require(unit["unit_id"] == digest({k: v for k, v in unit.items() if k != "unit_id"}), "frozen unit hash mismatch")
        require(unit["fit_manifest_sha256"] == digest(unit["fit"]) and len(unit["fit"]) == 576 and unit["B"] == 576 and unit["S"] == 24,
                "frozen recording/person budget differs")
        grouped[unit["block"], unit["fold"], unit["rotation"], unit["draw"], unit["policy"]].append(unit)
    expected = {("core", f, r, d, p) for f in range(5) for r in range(6) for d in range(3) for p in POLICIES}
    expected |= {("ft", f, r, 0, p) for f in range(5) for r in range(6) for p in POLICIES}
    require(grouped.keys() == expected, "panel contexts are incomplete/duplicated")
    output, by_key = [], {}
    for key in sorted(grouped):
        block, fold, rotation, draw, policy = key
        units = grouped[key]
        require(len(units) == 2 and units[0]["fit"] == units[1]["fit"], "models/seeds do not share identical panel")
        if block == "core":
            require({u["model"] for u in units} == {"cnn", "ridge_wavlm"}, "core models differ")
        else:
            require({u["model"] for u in units} == {"wavlm_ft"} and {u["seed_index"] for u in units} == {0, 1}, "FT seed pairing differs")
            require(units[0]["fit"] == grouped["core", fold, rotation, 0, policy][0]["fit"], "FT is not the fixed core draw0 panel")
        row = {"scope": block, "fold": fold, "rotation": rotation, "draw": draw, "policy": policy,
               "fit_manifest_sha256": units[0]["fit_manifest_sha256"], "unit_ids_sharing_panel": sorted(u["unit_id"] for u in units),
               **panel_statistics(units[0]["fit"], quality)}
        output.append(row)
        by_key[key] = row
    return output, by_key


def stats(values):
    values = list(values)
    require(values, "empty descriptive statistic")
    return {"n": len(values), "mean": math.fsum(values) / len(values), "median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def descriptions(panels, by_key):
    metrics = ("total_seconds", "total_hours", "mean_record_seconds", "mean_record_rms", "pooled_sample_rms",
               "amplitude_ge_0_999_sample_fraction", "mean_record_amplitude_ge_0_999_fraction", "recordings_with_any_amplitude_ge_0_999")
    policies, paired, differences = [], [], []
    for scope in ("core", "ft"):
        expected = 90 if scope == "core" else 30
        for policy in POLICIES:
            rows = [row for row in panels if row["scope"] == scope and row["policy"] == policy]
            require(len(rows) == expected, "policy panel count differs")
            policies.append({"scope": scope, "policy": policy, "n_panels": len(rows), "recordings_per_panel": 576,
                             "metrics": {name: stats(row[name] for row in rows) for name in metrics}})
        contexts = sorted({(row["fold"], row["rotation"], row["draw"]) for row in panels if row["scope"] == scope})
        for first, second in (("R", "U"), ("C", "U"), ("C", "R")):
            rows = []
            for fold, rotation, draw in contexts:
                a, b = by_key[scope, fold, rotation, draw, first], by_key[scope, fold, rotation, draw, second]
                row = {"scope": scope, "difference": first + "-" + second, "fold": fold, "rotation": rotation, "draw": draw,
                       "total_seconds_delta": a["total_seconds"] - b["total_seconds"],
                       "duration_relative_percent": (a["total_seconds"] / b["total_seconds"] - 1) * 100,
                       "mean_record_seconds_delta": a["mean_record_seconds"] - b["mean_record_seconds"],
                       "mean_record_rms_delta": a["mean_record_rms"] - b["mean_record_rms"],
                       "pooled_sample_rms_delta": a["pooled_sample_rms"] - b["pooled_sample_rms"],
                       "amplitude_ge_0_999_sample_fraction_delta_pp": (a["amplitude_ge_0_999_sample_fraction"] - b["amplitude_ge_0_999_sample_fraction"]) * 100}
                rows.append(row)
            names = [name for name in rows[0] if name not in ("scope", "difference", "fold", "rotation", "draw")]
            paired.append({"scope": scope, "difference": first + "-" + second, "matched_panels": len(rows),
                           "metrics": {name: stats(row[name] for row in rows) for name in names},
                           "maximum_absolute_duration_difference_context": max(rows, key=lambda row: abs(row["total_seconds_delta"]))})
            differences.extend(rows)
    return policies, paired, differences


def audit(repo, plan_path, features):
    repo, features = Path(repo).resolve(), Path(features).resolve()
    plan = read_json(plan_path)
    require(plan.get("schema") == "ser-speaker-coverage-1" and plan.get("plan_sha256") == PLAN_SHA
            and plan["plan_sha256"] == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "not the exact frozen 8ea plan")
    inp = plan["input"]
    manifest_path = local(repo, inp["manifest_path"])
    pin(manifest_path, inp["manifest_sha256"])
    raw = read_csv(manifest_path)
    require(len(raw) == 7442 and len({r["relative_path"] for r in raw}) == 7442, "raw manifest population mismatch")
    manifest = {r["relative_path"]: r for r in raw}
    receipt_path = local(features, inp["asv_receipt_file"])
    pin(receipt_path, inp["asv_receipt_sha256"])
    receipt = read_json(receipt_path)
    require(receipt["identity"]["manifest_sha256"] == inp["manifest_sha256"] and receipt["sha256"] == inp["asv_sha256"]
            and receipt["identity"]["rows"] == 7435, "ECAPA receipt differs from plan input identity")
    quality_path = local(features, receipt["audio_quality_file"])
    pin(quality_path, receipt["audio_quality_sha256"])
    quality = parse_quality(read_csv(quality_path), manifest)
    duplicate_groups = defaultdict(list)
    for row in raw:
        duplicate_groups[row["sha256"]].append(row)
    excluded = {"1076_MTI_SAD_XX.wav"}
    for group in duplicate_groups.values():
        if len({r["label"] for r in group}) > 1:
            excluded.update(r["relative_path"] for r in group)
        else:
            excluded.update(sorted(r["relative_path"] for r in group)[1:])
    require(len(quality) == 7435 and set(quality) == set(manifest) - excluded, "quality population differs from the complete clean manifest")
    source_sha = {name: plan["source_sha256"][name] for name in SOURCE_FILES}
    for name, expected in source_sha.items():
        pin(local(repo, name), expected)
    require(receipt["identity"]["source_sha256"] == source_sha[SOURCE_FILES[0]], "quality extraction source differs")
    # Assert the exact interpretation against pinned extractor source, without
    # running extraction or loading any WAV/ASV vector/prediction payload.
    source = (repo / SOURCE_FILES[0]).read_text(encoding="utf-8")
    require('wav = wav.mean(axis=1)' in source and 'float(np.mean(np.abs(wav) >= .999))' in source
            and source.index('audio_quality.append') < source.index('if sr != 16000:'), "raw quality measurement semantics changed")
    panels, by_key = panel_groups(plan, quality)
    policies, paired, differences = descriptions(panels, by_key)
    result = {"schema": "ser-coverage-pre-score-duration-description-1", "status": "completed_descriptive_audit", "plan_sha256": PLAN_SHA,
              "protocol_timing": "added during execution before formal SER scoring; not a preregistered outcome or test",
              "input_sha256": {"plan_file": sha(plan_path), "manifest": inp["manifest_sha256"], "ecapa_receipt": inp["asv_receipt_sha256"],
                               "audio_quality_csv": receipt["audio_quality_sha256"], "audit_script": sha(__file__)},
              "frozen_source_sha256": source_sha, "corpus_quality_rows": len(quality), "source_sample_rates": dict(Counter(r["original_sr"] for r in quality.values())),
              "wav_hash_binding": "every quality row matched the frozen manifest WAV SHA; all 7435 clean paths covered; no audio was rehashed in this audit",
              "core_panel_count": 270, "core_panels_per_policy": 90, "core_units_sharing_panels": 540,
              "ft_panel_count": 90, "ft_panels_per_policy": 30, "ft_units_sharing_panels": 180,
              "ft_scope": "same fixed draw0 panels as core; the two training seeds are not counted as new panels",
              "fixed_budget": "576 distinct recordings and 24 speakers per panel; not an equal number of raw audio hours",
              "measurement_definitions": {
                  "raw_audio": "soundfile float32 decoded audio, channel-mean mono, before resampling or trim/loudness changes",
                  "mean_record_rms": "equal-record arithmetic mean of original mono-waveform RMS",
                  "pooled_sample_rms": "sqrt(sum(record RMS squared * original samples) / sum(original samples))",
                  "amplitude_ge_0_999_sample_fraction": "number of original mono samples with absolute amplitude >= 0.999 divided by all original mono samples",
                  "policy_summary": "equal-panel descriptive summaries; each policy has 90 core or 30 draw0 FT panels",
                  "matched_differences": "same fold/rotation/draw, first-minus-second policy; ranges describe observed panels only"},
              "limitations": ["RMS and a 0.999 amplitude threshold are not validated human-perceived quality scores or proof of audible clipping",
                              "recordings and speakers recur across panels; panel counts are not independent statistical sample sizes",
                              "logmel/SSL caches use trimming and model-specific processing; CNN logmel normalization and FT random 3s crops/padding change effective inputs",
                              "raw source duration is not post-trim speech duration, model exposure duration, GPU work, or an alternative controlled budget",
                              "no label quality judgment, exclusion, resampling, covariate adjustment, new hypothesis test or confidence interval is introduced"],
              "SER_results_read": False, "ASV_embedding_values_read": False, "waveforms_read": False, "selection_or_training_changed": False,
              "hypothesis_tests": [], "confidence_intervals": [], "causal_adjustments": [],
              "policy_descriptions": policies, "matched_difference_descriptions": paired, "panels": panels, "matched_panel_differences": differences}
    result["audit_sha256"] = digest(result)
    return result


def markdown(report):
    lines = ["# 冻结训练面板的原始音频预算描述", "",
             "这是在正式 SER 评分前、执行期间补做的描述性检查，不是预注册终点，也没有增加检验、置信区间或因果调整。没有读取 E2 结果、重新选人或修改训练。", "",
             f"冻结计划：`{report['plan_sha256']}`。CSV 的 7,435 条 WAV SHA 均与冻结 manifest 一致，CSV 本身由 ECAPA 收据及计划逐级锁定。", "",
             "每个面板固定 576 条、24 人；原始音频总时长并不相等。核心部分每个策略有 90 个面板（5 fold × 6 rotation × 3 draw），CNN 与 Ridge 共用面板。FT 每策略只列 30 个 draw0 面板，与核心 draw0 完全一致；两个种子不增加面板数。", "",
             "下表对面板等权汇总。幅值比例在每个面板内按原始采样点数加权；RMS 为各条原始单声道音频 RMS 的算术均值。", "",
             "|范围/策略|面板数|每面板总分钟：均值 [最小, 最大]|每条秒数均值|每条 RMS 均值|幅值≥0.999比例均值（%）|",
             "|---|---:|---:|---:|---:|---:|"]
    for item in report["policy_descriptions"]:
        m = item["metrics"]
        t = m["total_seconds"]
        lines.append(f"|{item['scope']}/{item['policy']}|{item['n_panels']}|{t['mean']/60:.3f} [{t['minimum']/60:.3f}, {t['maximum']/60:.3f}]|{m['mean_record_seconds']['mean']:.5f}|{m['mean_record_rms']['mean']:.6f}|{m['amplitude_ge_0_999_sample_fraction']['mean']*100:.6f}|")
    lines += ["", "同一 fold/rotation/draw 的匹配差异如下，方向均为前者减后者。范围是观察到的面板差异，不是置信区间。", "",
              "|范围/差异|配对数|总时长差秒：均值 [最小, 最大]|相对时长差%：均值 [最小, 最大]|RMS差均值|幅值比例差均值（百分点）|",
              "|---|---:|---:|---:|---:|---:|"]
    for item in report["matched_difference_descriptions"]:
        m = item["metrics"]
        s, p = m["total_seconds_delta"], m["duration_relative_percent"]
        lines.append(f"|{item['scope']}/{item['difference']}|{item['matched_panels']}|{s['mean']:.3f} [{s['minimum']:.3f}, {s['maximum']:.3f}]|{p['mean']:.3f} [{p['minimum']:.3f}, {p['maximum']:.3f}]|{m['mean_record_rms_delta']['mean']:.6f}|{m['amplitude_ge_0_999_sample_fraction_delta_pp']['mean']:.6f}|")
    lines += ["", "完整 JSON 还保留每个面板、逐对差异、样本加权 pooled RMS，以及所有指标的均值/中位数/最小值/最大值。", "",
              "原始 RMS 和幅值阈值不是经过验证的人类感知质量分数，也不能单凭该比例断言存在可听见的削波失真。这里的原始时长包含静音；特征缓存的 trim、CNN 特征归一化，以及 FT 的随机 3 秒裁剪/补零都会改变有效输入，不能把这些原始秒数当作模型实际看到的语音时长或 GPU 预算。面板共享录音和说话人，因此 90/30 不是独立推断样本量。", "",
              f"数据来源 SHA：CSV `{report['input_sha256']['audio_quality_csv']}`；ECAPA 收据 `{report['input_sha256']['ecapa_receipt']}`；审计脚本 `{report['input_sha256']['audit_script']}`。", ""]
    return "\n".join(lines)


def self_test():
    manifest = {"a.wav": {"sha256": "a"}, "b.wav": {"sha256": "b"}}
    rows = [{"path": "a.wav", "sha256": "a", "seconds": "1", "original_sr": "10", "rms": "0.1", "clipped_fraction": "0.1"},
            {"path": "b.wav", "sha256": "b", "seconds": "3", "original_sr": "10", "rms": "0.3", "clipped_fraction": "0.0"}]
    quality = parse_quality(rows, manifest)
    panel = panel_statistics(["a.wav", "b.wav"], quality)
    require(panel["total_seconds"] == 4 and panel["mean_record_seconds"] == 2 and panel["mean_record_rms"] == .2, "duration/record mean test failed")
    require(abs(panel["pooled_sample_rms"] - math.sqrt(.07)) < 1e-12 and panel["amplitude_ge_0_999_sample_fraction"] == .025
            and panel["mean_record_amplitude_ge_0_999_fraction"] == .05, "sample weighting test failed")
    bad = [dict(rows[0], sha256="changed")]
    try:
        parse_quality(bad, manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched WAV SHA accepted")
    for paths in (["missing.wav"], ["a.wav", "a.wav"]):
        try:
            panel_statistics(paths, quality)
        except ValueError:
            pass
        else:
            raise AssertionError("missing/duplicate recording accepted")
    print(json.dumps({"self_test": "passed", "scope": "synthetic arithmetic and identity only"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    for name in ("repo", "plan", "features"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--out-prefix", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    require(args.repo and args.plan and args.features, "repo, plan and features are required")
    report = audit(args.repo, args.plan, args.features)
    report["audited_at_utc"] = datetime.now(timezone.utc).isoformat()
    report.pop("audit_sha256", None)
    report["audit_sha256"] = digest(report)
    base = args.out_prefix.resolve()
    require(base.parent == Path(__file__).resolve().parent, "outputs must remain in the external feasibility directory")
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    base.with_suffix(".md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "plan_sha256": PLAN_SHA, "core_panels": 270, "ft_panels": 90, "SER_results_read": False}))


if __name__ == "__main__":
    main()

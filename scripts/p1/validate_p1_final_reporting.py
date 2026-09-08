#!/usr/bin/env python3
"""Mechanical QA for project-07 P1 final reporting artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "results" / "protocol_premium" / "summary"
FIGURES = PROJECT_ROOT / "results" / "protocol_premium" / "figures"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (SUMMARY / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_mean_sd(text: str) -> tuple[float, float]:
    match = re.fullmatch(r"([+-]?\d+\.\d+) ± ([+-]?\d+\.\d+)", text)
    if not match:
        raise ValueError(f"invalid mean ± SD cell: {text}")
    return float(match.group(1)), float(match.group(2))


def parse_delta(text: str) -> tuple[float, float, float]:
    match = re.fullmatch(
        r"([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]", text
    )
    if not match:
        raise ValueError(f"invalid delta/CI cell: {text}")
    return tuple(float(match.group(index)) for index in (1, 2, 3))


def main() -> int:
    errors: list[str] = []
    report = (PROJECT_ROOT / "REPORT.md").read_text(encoding="utf-8")
    checklist = (PROJECT_ROOT / "CHECKLIST.md").read_text(encoding="utf-8")
    marker = "## P1：三协议受控测量与协议溢价表（最终）"
    if report.count(marker) != 1:
        errors.append(f"REPORT P1 section count={report.count(marker)}")
        p1 = ""
    else:
        p1 = report.split(marker, 1)[1]

    for phrase in (
        "1,620/1,620",
        "72/72",
        "没有缺失、失败",
        "P1 到此完成并停止，等待验收；没有开始 P2。",
        "样本框偏差与 P1 外推边界",
    ):
        if phrase not in p1:
            errors.append(f"REPORT missing phrase: {phrase}")

    metrics = read_csv("protocol_metric_summary.csv")
    premiums = read_csv("protocol_premium.csv")
    metric_lookup = {
        (row["corpus"], row["model"], row["protocol"], row["metric"]): row
        for row in metrics
    }
    corpus_key = {"RAVDESS": "ravdess", "CREMA-D": "cremad"}
    model_key = {"CNN": "cnn", "ResNet-SE": "resnet_se", "Transformer": "transformer", "FNO": "fno"}
    metric_key = {"UAR": "uar", "Accuracy": "accuracy", "Macro-F1": "macro_f1"}
    current_metric: str | None = None
    parsed_rows = []
    for line in p1.splitlines():
        if line.startswith("#### "):
            current_metric = metric_key.get(line.removeprefix("#### "))
        if not (line.startswith("| RAVDESS |") or line.startswith("| CREMA-D |")):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or current_metric is None:
            errors.append(f"malformed REPORT table row: {line}")
            continue
        corpus, model, contrast_cell, a_cell, b_cell, delta_cell, p_cell, verdict = cells
        contrast = contrast_cell[:2]
        row = next(
            (
                item
                for item in premiums
                if item["corpus"] == corpus_key[corpus]
                and item["model"] == model_key[model]
                and item["contrast"] == contrast
                and item["metric"] == current_metric
            ),
            None,
        )
        if row is None:
            errors.append(f"unmapped REPORT row: {line}")
            continue
        a = metric_lookup[(row["corpus"], row["model"], row["protocol_a"], current_metric)]
        b = metric_lookup[(row["corpus"], row["model"], row["protocol_b"], current_metric)]
        report_a = parse_mean_sd(a_cell)
        report_b = parse_mean_sd(b_cell)
        report_delta = parse_delta(delta_cell)
        expected_a = (float(a["mean"]), float(a["sample_sd"]))
        expected_b = (float(b["mean"]), float(b["sample_sd"]))
        expected_delta = (
            float(row["point_mean_difference"]) * 100,
            float(row["ci95_low"]) * 100,
            float(row["ci95_high"]) * 100,
        )
        if any(abs(x - y) > 0.00051 for x, y in zip(report_a, expected_a)):
            errors.append(f"mean/SD A mismatch: {corpus}/{model}/{contrast}/{current_metric}")
        if any(abs(x - y) > 0.00051 for x, y in zip(report_b, expected_b)):
            errors.append(f"mean/SD B mismatch: {corpus}/{model}/{contrast}/{current_metric}")
        if any(abs(x - y) > 0.0051 for x, y in zip(report_delta, expected_delta)):
            errors.append(f"delta/CI mismatch: {corpus}/{model}/{contrast}/{current_metric}")
        if current_metric == "uar":
            if p_cell == "—" or abs(float(p_cell) - float(row["holm_adjusted_p"])) > max(5e-8, abs(float(row["holm_adjusted_p"])) * 0.0002):
                errors.append(f"Holm p mismatch: {corpus}/{model}/{contrast}")
        elif p_cell != "—":
            errors.append(f"unexpected p for {current_metric}: {corpus}/{model}/{contrast}")
        low, high = float(row["ci95_low"]), float(row["ci95_high"])
        if current_metric != "uar" and low <= 0 <= high and "not conclusive" not in verdict:
            errors.append(f"missing not conclusive: {corpus}/{model}/{contrast}/{current_metric}")
        parsed_rows.append((corpus, model, contrast, current_metric))

    if len(parsed_rows) != 72 or len(set(parsed_rows)) != 72:
        errors.append(f"REPORT premium table coverage rows={len(parsed_rows)} unique={len(set(parsed_rows))}")

    manifest_path = FIGURES / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("statistics_recomputed") is not False:
        errors.append("figure manifest status/reporting boundary invalid")
    for item in manifest.get("sources", {}).values():
        path = PROJECT_ROOT / item["relative_path"]
        if sha256(path) != item["sha256"]:
            errors.append(f"figure source hash mismatch: {path.name}")
    for item in manifest.get("outputs", []):
        path = PROJECT_ROOT / item["relative_path"]
        if sha256(path) != item["sha256"]:
            errors.append(f"figure output hash mismatch: {path.name}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                if image.width < 1800 or image.height < 1000:
                    errors.append(f"figure too small: {path.name} {image.size}")
                stats = ImageStat.Stat(image.convert("L"))
                if stats.var[0] < 100:
                    errors.append(f"figure appears blank: {path.name}")

    for heading in range(1, 8):
        if f"## {heading}." not in checklist:
            errors.append(f"CHECKLIST missing section {heading}")
    if "最低可比门槛" not in checklist or "speaker-independent" not in checklist:
        errors.append("CHECKLIST missing minimum-comparability gate")

    live_new = [
        PROJECT_ROOT / "tools" / "plot_p1_protocol_premium.py",
        PROJECT_ROOT / "tools" / "build_p1_report_section.py",
        PROJECT_ROOT / "tools" / "validate_p1_final_reporting.py",
        PROJECT_ROOT / "CHECKLIST.md",
    ]
    old_prefix = "C:\\Users\\jock8\\Desktop\\科研"
    for path in live_new:
        if old_prefix in path.read_text(encoding="utf-8"):
            errors.append(f"new active file contains old path: {path.name}")

    result = {
        "status": "pass" if not errors else "fail",
        "checks": {
            "report_premium_rows": len(parsed_rows),
            "report_unique_premium_rows": len(set(parsed_rows)),
            "figure_outputs": len(manifest.get("outputs", [])),
            "checklist_sections": 7,
        },
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

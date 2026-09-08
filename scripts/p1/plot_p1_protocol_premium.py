#!/usr/bin/env python3
"""Render the three preregistered P1 reporting figure families.

This script is reporting-only. It reads the already verified summary CSVs and
does not recompute OOF metrics, paired estimates, confidence intervals, tests,
or Holm adjustments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "protocol_premium" / "summary"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "protocol_premium" / "figures"

CORPORA = ("ravdess", "cremad")
CORPUS_LABEL = {"ravdess": "RAVDESS", "cremad": "CREMA-D"}
MODELS = ("cnn", "resnet_se", "transformer", "fno")
MODEL_LABEL = {
    "cnn": "CNN",
    "resnet_se": "ResNet-SE",
    "transformer": "Transformer",
    "fno": "FNO",
}
PROTOCOLS = ("random", "groupkfold", "loso")
PROTOCOL_LABEL = {
    "random": "Random",
    "groupkfold": "GroupKFold",
    "loso": "LOSO",
}
CONTRASTS = ("RG", "RL", "GL")
METRICS = ("uar", "accuracy", "macro_f1")
METRIC_LABEL = {"uar": "UAR", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}
MODEL_COLORS = {
    "cnn": "#1B4965",
    "resnet_se": "#087E8B",
    "transformer": "#D97706",
    "fno": "#8B5CF6",
}
METRIC_COLORS = {"uar": "#1B4965", "accuracy": "#D97706", "macro_f1": "#087E8B"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_complete(rows: list[dict[str, str]], expected: int, name: str) -> None:
    if len(rows) != expected:
        raise ValueError(f"{name}: expected {expected} rows, observed {len(rows)}")
    bad = [row for row in rows if row.get("status") != "complete"]
    if bad:
        raise ValueError(f"{name}: {len(bad)} non-complete rows; reporting plot refused")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        metadata = {"Creator": "plot_p1_protocol_premium.py"}
        if suffix == "svg":
            metadata["Date"] = None
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white", metadata=metadata)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_protocol_comparison(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    lookup = {
        (row["corpus"], row["model"], row["protocol"], row["metric"]): row
        for row in rows
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.7), sharex=True, sharey="row")
    x = np.arange(len(PROTOCOLS), dtype=float)
    for i, corpus in enumerate(CORPORA):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for model in MODELS:
                selected = [lookup[(corpus, model, protocol, metric)] for protocol in PROTOCOLS]
                means = np.array([float(row["mean"]) for row in selected])
                sds = np.array([float(row["sample_sd"]) for row in selected])
                ax.errorbar(
                    x,
                    means,
                    yerr=sds,
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    markersize=4.8,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABEL[model],
                )
            ax.set_xticks(x, [PROTOCOL_LABEL[p] for p in PROTOCOLS])
            ax.grid(axis="y", color="#D8DEE4", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f"{CORPUS_LABEL[corpus]} · {METRIC_LABEL[metric]}", fontsize=11)
            ax.set_ylim(0.15, 0.80)
            if j == 0:
                ax.set_ylabel("OOF score (mean ± seed SD)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("P1 three-protocol comparison", fontsize=14, fontweight="bold", y=1.055)
    fig.text(
        0.5,
        -0.01,
        "Each point is the OOF metric mean across seeds {0,1,2}; error bars are sample SD across seeds.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "p1_protocol_metric_comparison")


def plot_speaker_differences(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0), sharey=True)
    offsets = {"uar": -0.22, "accuracy": 0.0, "macro_f1": 0.22}
    width = 0.18
    for i, corpus in enumerate(CORPORA):
        for j, contrast in enumerate(CONTRASTS):
            ax = axes[i, j]
            for metric in METRICS:
                data = []
                positions = []
                for k, model in enumerate(MODELS):
                    values = [
                        float(row[f"{metric}_difference"])
                        for row in rows
                        if row["corpus"] == corpus
                        and row["model"] == model
                        and row["contrast"] == contrast
                        and row["status"] == "complete"
                    ]
                    data.append(values)
                    positions.append(k + offsets[metric])
                box = ax.boxplot(
                    data,
                    positions=positions,
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.1},
                    whiskerprops={"color": METRIC_COLORS[metric], "linewidth": 1.0},
                    capprops={"color": METRIC_COLORS[metric], "linewidth": 1.0},
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(METRIC_COLORS[metric])
                    patch.set_edgecolor(METRIC_COLORS[metric])
                    patch.set_alpha(0.80)
            ax.axhline(0.0, color="#111827", linewidth=0.9, linestyle="--")
            ax.set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS], rotation=20, ha="right")
            ax.grid(axis="y", color="#D8DEE4", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f"{CORPUS_LABEL[corpus]} · {contrast}", fontsize=11)
            if j == 0:
                ax.set_ylabel("Speaker-paired difference (A − B)")
    legend_handles = [
        plt.Line2D([0], [0], color=METRIC_COLORS[m], linewidth=7, label=METRIC_LABEL[m])
        for m in METRICS
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Per-speaker paired protocol-difference distributions", fontsize=14, fontweight="bold", y=1.055)
    fig.text(
        0.5,
        -0.015,
        "RG = Random − GroupKFold; RL = Random − LOSO; GL = GroupKFold − LOSO. Boxes summarize speakers after three-seed averaging.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "p1_paired_speaker_difference_distributions")


def plot_premium_facets(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    complete = [row for row in rows if row["status"] == "complete"]
    lows = [float(row["ci95_low"]) for row in complete]
    highs = [float(row["ci95_high"]) for row in complete]
    span = max(abs(min(lows)), abs(max(highs)))
    xlim = (-span * 1.08, span * 1.08)
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.7), sharex=True, sharey=True)
    row_order = [(contrast, metric) for contrast in CONTRASTS for metric in METRICS]
    y = np.arange(len(row_order))[::-1]
    labels = [f"{c} {METRIC_LABEL[m]}" for c, m in row_order]
    lookup = {
        (row["corpus"], row["model"], row["contrast"], row["metric"]): row
        for row in complete
    }
    for i, corpus in enumerate(CORPORA):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            for y_pos, (contrast, metric) in zip(y, row_order):
                row = lookup[(corpus, model, contrast, metric)]
                point = float(row["point_mean_difference"])
                low = float(row["ci95_low"])
                high = float(row["ci95_high"])
                ax.errorbar(
                    point,
                    y_pos,
                    xerr=[[point - low], [high - point]],
                    fmt="o",
                    color=METRIC_COLORS[metric],
                    capsize=2.5,
                    markersize=4.6,
                    linewidth=1.2,
                )
            ax.axvline(0.0, color="#111827", linewidth=0.9, linestyle="--")
            ax.grid(axis="x", color="#D8DEE4", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f"{CORPUS_LABEL[corpus]} · {MODEL_LABEL[model]}", fontsize=10.5)
            ax.set_xlim(*xlim)
            ax.set_yticks(y, labels)
            if i == 1:
                ax.set_xlabel("Protocol premium Δ (A − B), 95% CI")
    fig.suptitle("Protocol premiums by corpus and model", fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "CIs are the frozen 10,000-replicate speaker-cluster bootstrap intervals; UAR multiplicity decisions remain in protocol_premium.csv.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "p1_protocol_premium_facets")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary_dir = args.summary_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "protocol_metric_summary.csv": summary_dir / "protocol_metric_summary.csv",
        "paired_speaker_differences.csv": summary_dir / "paired_speaker_differences.csv",
        "protocol_premium.csv": summary_dir / "protocol_premium.csv",
    }
    metric_rows = read_csv(inputs["protocol_metric_summary.csv"])
    speaker_rows = read_csv(inputs["paired_speaker_differences.csv"])
    premium_rows = read_csv(inputs["protocol_premium.csv"])
    require_complete(metric_rows, 72, "protocol_metric_summary.csv")
    require_complete(speaker_rows, 1380, "paired_speaker_differences.csv")
    require_complete(premium_rows, 72, "protocol_premium.csv")

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "svg.hashsalt": "project07-p1-protocol-premium-v1",
        }
    )

    outputs: list[Path] = []
    outputs.extend(plot_protocol_comparison(metric_rows, output_dir))
    outputs.extend(plot_speaker_differences(speaker_rows, output_dir))
    outputs.extend(plot_premium_facets(premium_rows, output_dir))

    manifest_path = output_dir / "figure_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "status": "complete",
        "reporting_only": True,
        "statistics_recomputed": False,
        "script": {
            "relative_path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "sources": {
            name: {
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in inputs.items()
        },
        "outputs": [
            {"relative_path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(path)}
            for path in outputs
        ],
        "figure_families": [
            "three_protocol_seed_summary",
            "paired_speaker_difference_distribution",
            "corpus_model_faceted_premium_ci",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": len(outputs), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

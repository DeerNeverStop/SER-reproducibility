"""Translate the existing window figure; read all 180 summaries without new analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

WIDTH_MM, HEIGHT_MM = 172.0, 62.0
CORPORA = ("subesco", "ravdess")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/simhei.ttf"),
                        help="Installed CJK font; the original font file is not distributed")
    parser.add_argument("--out", type=Path, help="New/empty output directory")
    args = parser.parse_args()
    repo = args.repo.resolve()
    base = repo / "paper/review-revision-20260909"
    out = args.out.resolve() if args.out else base / "chinese/figures"
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "Output must be new or empty")
    inputs = {"summary_csv": base / "figures/window_selection_summary.csv",
              "english_pdf": base / "figures/window_selection.pdf",
              "english_audit": base / "figures/window_selection_audit.json"}
    raw = {name: path.read_bytes() for name, path in inputs.items()}
    source_raw = Path(__file__).read_bytes()
    font_raw = args.font.read_bytes()
    english_audit = json.loads(raw["english_audit"])
    require(record(raw["summary_csv"]) == english_audit["outputs"]["window_selection_summary.csv"],
            "Summary differs from the existing English figure's source")
    require(record(raw["english_pdf"]) == english_audit["outputs"]["window_selection.pdf"],
            "English figure differs from its generation record")
    rows = list(csv.DictReader(io.StringIO(raw["summary_csv"].decode("utf-8"))))
    require(len(rows) == 180, "Expected all 180 existing summary rows")
    seen = set()
    for row in rows:
        row["window"] = int(row["window"])
        key = row["corpus"], row["criterion"], row["window"]
        require(key not in seen, "Duplicate summary row")
        seen.add(key)
        require(row["model"] == "wavlm_base_plus", "Unexpected model")
        require((int(row["n_draws"]), int(row["folds_per_draw"]), int(row["df"])) == (24, 5, 23),
                "Unexpected summary aggregation")
        for field in ("mean_pp", "pointwise_95_low_pp", "pointwise_95_high_pp"):
            row[field] = float(row[field])
            require(math.isfinite(row[field]), "Nonfinite plot value")
        require(row["pointwise_95_low_pp"] <= row["mean_pp"] <= row["pointwise_95_high_pp"], "Invalid interval")
    require(seen == {(c, criterion, w) for c in CORPORA for criterion in ("CE", "UAR") for w in range(1, 46)},
            "Incomplete summary grid")

    font_manager.fontManager.addfont(str(args.font))
    family = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10.4, "axes.labelsize": 10.4,
                         "axes.titlesize": 10.6, "xtick.labelsize": 10.2, "ytick.labelsize": 10.2,
                         "legend.fontsize": 10.4, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": False, "axes.linewidth": .65})
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), sharey=True)
    fig.subplots_adjust(left=.118, right=.987, bottom=.27, top=.775, wspace=.23)
    handles = []
    for index, (corpus, ax) in enumerate(zip(CORPORA, axes)):
        ax.set_title(f"({chr(97 + index)}) {corpus.upper()}", pad=6)
        ax.axhline(0, color=".45", linewidth=.6, zorder=0)
        for window in (15, 45):
            ax.axvline(window, color=".6", linewidth=.65, linestyle=(0, (2, 3)), zorder=0)
        for criterion, color, linestyle in (("CE", "#0072B2", "-"), ("UAR", "#D55E00", (0, (4, 2)))):
            selected = sorted((r for r in rows if r["corpus"] == corpus and r["criterion"] == criterion),
                              key=lambda r: r["window"])
            x = [r["window"] for r in selected]
            ax.fill_between(x, [r["pointwise_95_low_pp"] for r in selected],
                            [r["pointwise_95_high_pp"] for r in selected], color=color, alpha=.16, linewidth=0)
            line, = ax.plot(x, [r["mean_pp"] for r in selected], color=color, linestyle=linestyle,
                            linewidth=1.55, zorder=3)
            if index == 0:
                handles.append(line)
        ax.set_xlim(1, 45.8)
        ax.set_xticks((1, 15, 30, 45))
        ax.tick_params(axis="both", length=3, pad=2, width=.6)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=".88", linewidth=.45)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(math.floor(min(r["pointwise_95_low_pp"] for r in rows)),
                     math.ceil(max(r["pointwise_95_high_pp"] for r in rows)))
    axes[0].set_ylabel("外测UAR差（百分点）\n熟悉验证 - 陌生验证", labelpad=8)
    fig.supxlabel("候选轮数W", x=.552, y=.063, fontsize=10.4)
    fig.legend(handles=handles, labels=("按CE选模", "按UAR选模"), ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(.55, 1.005), handlelength=2.6,
               columnspacing=2.0, borderaxespad=0)
    fig.canvas.draw()
    texts = [item for item in fig.findobj(Text) if item.get_visible() and item.get_text()]
    minimum = min(item.get_fontsize() for item in texts)
    require(minimum >= 10.2, "Text smaller than 10.2 pt")
    renderer = fig.canvas.get_renderer()
    for item in texts:
        box = item.get_window_extent(renderer)
        require(box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.x1 + .5 and box.y1 <= fig.bbox.y1 + .5,
                "Text outside native canvas: " + item.get_text())
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "window_selection_zh.pdf", metadata={"Title": "候选轮数与外测UAR差",
                "Author": "", "Subject": "Existing post hoc profiles; unchanged pointwise intervals; no new tests",
                "CreationDate": None, "ModDate": None})
    fig.savefig(out / "window_selection_zh.png", dpi=300)
    plt.close(fig)
    require(Path(__file__).read_bytes() == source_raw, "Figure script changed")
    require(args.font.read_bytes() == font_raw, "Installed font changed")
    for name, path in inputs.items():
        require(path.read_bytes() == raw[name], "Input changed: " + name)
    manifest = {"schema": "ser-chinese-window-figure-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": {name: record((out / name).read_bytes())
                          for name in ("window_selection_zh.pdf", "window_selection_zh.png")},
                "inputs": {name: {"path": path.relative_to(repo).as_posix(), **record(raw[name])}
                           for name, path in inputs.items()},
                "script": {"path": "paper/review-revision-20260909/chinese/make_chinese_figure.py", **record(source_raw)},
                "font": {"filename": args.font.name, "family": family, **record(font_raw),
                         "distribution": "Installed font used for rendering; original font file not included. PDF embeds the used subset."},
                "layout": {"width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "minimum_native_font_pt": minimum,
                           "text_artists_checked": len(texts), "text_inside_canvas": True, "png_dpi": 300,
                           "pdf_fonttype": 42, "cropping": "native canvas; no tight bounding-box resize"},
                "data_scope": "All 180 existing English summary rows reused without statistical recalculation or rounding; 2 corpora x 2 criteria x 45 windows.",
                "interval_scope": "Existing pointwise 95% t intervals from 24 five-fold draw means, df23; not simultaneous bands; post hoc windows; no additional hypothesis tests.",
                "label_scope": "Seen/unseen validation translated as 熟悉验证/陌生验证; all trajectories still completed 45 epochs.",
                "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__}}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                                            ensure_ascii=False).encode()).hexdigest()
    with (out / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"pass": True, "summary_rows": len(rows), "layout": manifest["layout"],
                      "files": manifest["files"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

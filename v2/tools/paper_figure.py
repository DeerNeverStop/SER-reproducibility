"""Figure for the v2 manuscript (three panels), generated from results.json only.

A  protocol factorial: per primary level, the three speaker-paired contrasts RR-GG, RG-GG, RR-RG for
   the scratch group (registry rows N01/N03/N04, R01-R06) and the D01 GR-GG allocation descriptor.
B  mechanism block: N05/N06 (CREMA-D), N07/N08/N09 (SUBESCO-full), N10 (CREMA-D 24 vs utterance-matched 91).
C  fine-tuning: R09-R11 (fine-tuned premium) and N11-N13 (re-exposure = FT premium - frozen premium).
Rows whose a-priori status is `estimate` (N07, N08, N11-N13) are drawn as open (hatched) bars.

    python -m tools.paper_figure --run runs/main --out paper/v2/figure_v2.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import read_json  # noqa: E402

FILL, NEG = "#4C72B0", "#C44E52"


def bar(ax, items, title, ylabel):
    xs = np.arange(len(items))
    for i, (label, h) in enumerate(items):
        if h is None or not h.get("tested"):
            ax.text(i, 0, "n.t.", ha="center", va="bottom", fontsize=7)
            continue
        m, lo, hi = h["mean"], h["ci_low"], h["ci_high"]
        est = h.get("a_priori_status", "confirmatory") != "confirmatory" or h.get("verdict", "").startswith("estimate")
        color = FILL if m >= 0 else NEG
        if est:
            ax.bar(i, m, width=0.7, facecolor="white", edgecolor=color, hatch="////", linewidth=1.0)
        else:
            ax.bar(i, m, width=0.7, color=color)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _ in items], rotation=90, fontsize=6)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panels", default="ABC", help="subset of ABC to draw (single-column figure when only A)")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = read_json(a.run / "results.json")
    H = {**R["hypotheses"], **R["tost"]}
    D = R["descriptors"]
    if a.panels == "A":
        fig, ax0 = plt.subplots(1, 1, figsize=(3.4, 1.7)); axes = [ax0, None, None]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), gridspec_kw={"width_ratios": [4.2, 2.4, 2.4]})
    items = []
    for lv, key, ids in (("RAV", "ravdess", ("R01", "R02", "R03")), ("CRE", "cremad", ("R04", "R05", "R06")), ("SUB-980", "subesco_980", ("N01", "N03", "N04"))):
        for hid, name in zip(ids, ("RR-GG", "RG-GG", "RR-RG")):
            items.append((f"{lv} {name}", H.get(hid)))
        items.append((f"{lv} GR-GG", D["D01"][key]["scratch"]))
    bar(axes[0], items, "Protocol factorial (scratch models)" if a.panels == "A" else "A. Protocol factorial (scratch models)", "UAR difference (points)")
    items = [("CRE prm (N05)", H.get("N05")), ("CRE spk (N06)", H.get("N06")), ("SUB-full prm (N07)", H.get("N07")),
             ("SUB-full spk (N08)", H.get("N08")), ("SUB-full sib (N09)", H.get("N09")), ("CRE 24 vs 91 (N10)", H.get("N10"))]
    if axes[1] is not None: bar(axes[1], items, "B. Mechanism block", "")
    items = [("RAV FT (R09)", H.get("R09")), ("CRE FT (R10)", H.get("R10")), ("SUB-980 FT (R11)", H.get("R11")),
             ("RAV re-exp (N11)", H.get("N11")), ("CRE re-exp (N12)", H.get("N12")), ("SUB-980 re-exp (N13)", H.get("N13"))]
    if axes[2] is not None: bar(axes[2], items, "C. WavLM-base+ fine-tuning", "")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

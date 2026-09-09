"""Post hoc window profile from archived CSV; no fits, prediction loads or new tests.

Run from any directory with --repo pointing at the reproducibility checkout.
Only curves.csv supplies values/selection; the other tables check W=15 and 45.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import scipy
from scipy.stats import t

CORPORA = ("subesco", "ravdess")
MODEL = "wavlm_base_plus"
RULES = ("seen_ce", "unseen_ce", "seen_uar", "unseen_uar")
RESULTS = Path("docs/autodl-supplement-20260908/results")
REL_OUT = Path("paper/review-revision-20260909/figures")
WIDTH_MM, HEIGHT_MM = 178.0, 58.0
TOL = 1e-9


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(raw):
    return {"bytes": len(raw), "sha256": digest(raw)}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def parse_csv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(repo):
    names = ("results.json", "curves.csv", "selected_epochs.csv", "units.csv", "draws.csv")
    raw = {name: (repo / RESULTS / name).read_bytes() for name in names}
    results = json.loads(raw["results.json"])
    require(digest(canonical({k: v for k, v in results.items() if k != "result_sha256"}))
            == results["result_sha256"], "Result self-seal mismatch")
    for name in names[1:]:
        require(record(raw[name]) == results["tables"][name], "Input table hash mismatch: " + name)
    require(results["counts"]["formal_units"] == 720, "Expected complete 720-unit results")
    all_curves = parse_csv(raw["curves.csv"])
    require(len(all_curves) == 18000, "Expected complete 18000-row curve table")
    require(len({r["unit_id"] for r in all_curves}) == 720, "Curve UID count mismatch")
    curves = defaultdict(list)
    identity = {}
    for r0 in all_curves:
        if r0["corpus"] not in CORPORA or r0["model"] != MODEL:
            continue
        r = dict(r0)
        for key in ("draw", "fold", "epoch"):
            r[key] = int(r[key])
        for key in (*RULES, "outer_uar"):
            r[key] = float(r[key])
            require(math.isfinite(r[key]), "Nonfinite curve value")
        require(r["uar_unit"] == "UAR_percent" and r["ce_unit"] == "natural_log_loss", "Curve units mismatch")
        ident = (r["corpus"], r["draw"], r["fold"])
        require(identity.setdefault(r["unit_id"], ident) == ident, "Inconsistent UID identity")
        curves[r["unit_id"]].append(r)
    require(len(curves) == 240, "Expected 240 long trajectories")
    expected = {(c, d, f) for c in CORPORA for d in range(24) for f in range(5)}
    require(set(identity.values()) == expected and len(identity) == len(expected), "Panel identity mismatch")
    for history in curves.values():
        history.sort(key=lambda r: r["epoch"])
        require([r["epoch"] for r in history] == list(range(1, 46)), "Missing/duplicate epoch")

    selected = {}
    for r in parse_csv(raw["selected_epochs.csv"]):
        key = (r["unit_id"], int(r["window"]), r["rule"])
        require(key not in selected, "Duplicate selected-epoch row")
        selected[key] = r
    unit_ref = {(r["unit_id"], int(r["window"])): r for r in parse_csv(raw["units.csv"])}
    draw_ref = {(r["corpus"], int(r["window"]), int(r["draw"])): r
                for r in parse_csv(raw["draws.csv"])
                if r["comparison"] == "model_window" and r["model"] == MODEL}
    description_ref = {(r["corpus"], r["window"], r["endpoint"]): r
                       for r in results["descriptive"]
                       if r["comparison"] == "model_window" and r["model"] == MODEL}

    by_draw = defaultdict(list)
    checks = {"selected_epochs_checked": 0, "selected_outer_values_checked": 0,
              "unit_differences_checked": 0, "draw_values_checked": 0,
              "summary_values_checked": 0, "max_numeric_abs_error_pp": 0.0}

    def compare(a, b, kind):
        error = abs(float(a) - float(b))
        require(error <= TOL, "Endpoint mismatch: " + kind)
        checks[kind] += 1
        checks["max_numeric_abs_error_pp"] = max(checks["max_numeric_abs_error_pp"], error)

    for uid in sorted(curves):
        history = curves[uid]
        corpus, draw, fold = identity[uid]
        for window in range(1, 46):
            prefix = history[:window]
            # Earliest epoch wins exact ties in the archived float representation.
            # W15/45 are also checked against the original Fraction-based scorer.
            winners = {rule: min(prefix, key=lambda r: ((r[rule] if rule.endswith("ce") else -r[rule]), r["epoch"]))
                       for rule in RULES}
            for criterion in ("CE", "UAR"):
                low = criterion.lower()
                difference = winners["seen_" + low]["outer_uar"] - winners["unseen_" + low]["outer_uar"]
                by_draw[(corpus, window, criterion, draw)].append((fold, difference))
                if window in (15, 45):
                    compare(difference, unit_ref[uid, window]["D_" + criterion + "_pp"], "unit_differences_checked")
            if window in (15, 45):
                for rule, winner in winners.items():
                    ref = selected[uid, window, rule]
                    require(winner["epoch"] == int(ref["selected_epoch"]), "Selected epoch mismatch")
                    checks["selected_epochs_checked"] += 1
                    compare(winner["outer_uar"], ref["outer_uar"], "selected_outer_values_checked")

    draws, summary, endpoint_values = [], [], []
    t_critical = float(t.ppf(.975, 23))
    for corpus in CORPORA:
        for window in range(1, 46):
            for criterion in ("CE", "UAR"):
                values = []
                for draw in range(24):
                    pairs = by_draw[corpus, window, criterion, draw]
                    require(sorted(f for f, _ in pairs) == list(range(5)), "Incomplete five-fold draw")
                    value = statistics.mean(v for _, v in pairs)
                    values.append(value)
                    draws.append({"corpus": corpus, "model": MODEL, "window": window,
                                  "criterion": criterion, "draw": draw, "folds": 5,
                                  "seen_minus_unseen_outer_uar_pp": value})
                    if window in (15, 45):
                        compare(value, draw_ref[corpus, window, draw]["D_" + criterion + "_pp"], "draw_values_checked")
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                se = sd / math.sqrt(24)
                lo, hi = mean - t_critical * se, mean + t_critical * se
                row = {"corpus": corpus, "model": MODEL, "window": window, "criterion": criterion,
                       "n_draws": 24, "folds_per_draw": 5, "df": 23, "mean_pp": mean,
                       "sd_draw_pp": sd, "se_pp": se, "pointwise_95_low_pp": lo, "pointwise_95_high_pp": hi}
                summary.append(row)
                if window in (15, 45):
                    ref = description_ref[corpus, window, "D_" + criterion]
                    for a, b in ((mean, ref["mean_pp"]), (lo, ref["pointwise_95_ci_pp"][0]),
                                 (hi, ref["pointwise_95_ci_pp"][1])):
                        compare(a, b, "summary_values_checked")
                    endpoint_values.append(row)
    for name in names:
        require((repo / RESULTS / name).read_bytes() == raw[name], "Source changed during calculation")
    return draws, summary, {"inputs": {str(RESULTS / n).replace("\\", "/"): record(raw[n]) for n in names},
                            "result_sha256": results["result_sha256"], "plan_sha256": results["plan_sha256"],
                            "counts": {"source_units": 720, "source_curve_rows": 18000,
                                       "long_trajectories_used": 240, "curve_rows_used": 10800,
                                       "draw_rows_output": len(draws), "summary_rows_output": len(summary)},
                            "endpoint_checks": checks, "endpoint_values": endpoint_values,
                            "t_critical_df23": t_critical}


def plot(summary, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.4,
                         "axes.labelsize": 9.4, "axes.titlesize": 9.6,
                         "xtick.labelsize": 9.2, "ytick.labelsize": 9.2,
                         "legend.fontsize": 9.4, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": .65, "xtick.major.width": .6,
                         "ytick.major.width": .6, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), sharey=True)
    fig.subplots_adjust(left=.092, right=.987, bottom=.252, top=.782, wspace=.24)
    styles = {"CE": ("#0072B2", "-"), "UAR": ("#D55E00", (0, (4, 2)))}
    handles = []
    for index, (corpus, ax) in enumerate(zip(CORPORA, axes)):
        ax.set_title(f"({chr(97 + index)}) {corpus.upper()}", pad=5.0)
        ax.axhline(0, color=".45", lw=.6, zorder=0)
        ax.axvline(15, color=".6", lw=.65, ls=(0, (2, 3)), zorder=0)
        ax.axvline(45, color=".6", lw=.65, ls=(0, (2, 3)), zorder=0)
        for criterion, (color, style) in styles.items():
            rows = [r for r in summary if r["corpus"] == corpus and r["criterion"] == criterion]
            x = [r["window"] for r in rows]
            ax.fill_between(x, [r["pointwise_95_low_pp"] for r in rows],
                            [r["pointwise_95_high_pp"] for r in rows], color=color, alpha=.16, linewidth=0)
            line, = ax.plot(x, [r["mean_pp"] for r in rows], color=color, linestyle=style,
                            linewidth=1.5, label=criterion, zorder=3)
            if index == 0:
                handles.append(line)
        ax.set_xlim(1, 45.8)
        ax.set_xticks([1, 15, 30, 45])
        ax.tick_params(axis="both", length=3, pad=2)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=".88", linewidth=.45)
        ax.spines[["top", "right"]].set_visible(False)
    lo = min(r["pointwise_95_low_pp"] for r in summary)
    hi = max(r["pointwise_95_high_pp"] for r in summary)
    axes[0].set_ylim(math.floor(lo), math.ceil(hi))
    axes[0].set_ylabel("Outer UAR difference (pp)\nseen - unseen", labelpad=6)
    fig.supxlabel("Selection window W (epochs)", x=.54, y=.058, fontsize=9.4)
    fig.legend(handles=handles, labels=["CE criterion", "UAR criterion"], loc="upper center",
               bbox_to_anchor=(.54, 1.002), ncol=2, frameon=False, handlelength=2.8,
               columnspacing=2.0, borderaxespad=0)
    fig.canvas.draw()
    texts = [artist for artist in fig.findobj(Text) if artist.get_visible() and artist.get_text()]
    minimum = min(artist.get_fontsize() for artist in texts)
    require(minimum >= 9, "Figure includes text below 9 pt")
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for artist in texts:
        box = artist.get_window_extent(renderer)
        require(box.x0 >= bounds.x0 - .5 and box.x1 <= bounds.x1 + .5
                and box.y0 >= bounds.y0 - .5 and box.y1 <= bounds.y1 + .5,
                "Figure text extends outside native canvas: " + artist.get_text())
    fig.savefig(out / "window_selection.pdf", metadata={"Title": "Post hoc selection-window profile",
                "Author": "", "Subject": "Pointwise 95% t intervals over 24 five-fold draw means; no additional tests",
                "CreationDate": None, "ModDate": None})
    fig.savefig(out / "window_selection.png", dpi=300)
    plt.close(fig)
    return {"width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "native_minimum_text_pt": minimum,
            "visible_text_artists_checked": len(texts), "native_text_inside_canvas": True,
            "pdf_fonttype": 42, "png_dpi": 300, "save_bbox": "native canvas; no tight cropping"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, help="New/empty output directory; default: repo figures directory")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve() if args.out else (repo / REL_OUT).resolve()
    require(out != repo and repo not in out.parents or out == (repo / REL_OUT).resolve(),
            "Inside the checkout only the dedicated figures directory is allowed")
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "Refusing nonempty output directory")
    source_before = Path(__file__).read_bytes()
    draws, summary, audit = analyze(repo)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "window_selection_draws.csv", draws)
    write_csv(out / "window_selection_summary.csv", summary)
    audit["layout"] = plot(summary, out)
    audit.update({"schema": "ser-posthoc-window-figure-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "script": {"path": "paper/review-revision-20260909/scripts/make_window_figure.py", **record(source_before)},
                  "analysis_scope": "Post hoc nested candidate windows on archived WavLM 45-epoch trajectories; not new training or a new confirmatory test family.",
                  "aggregation": "Within each corpus/window/criterion, mean five paired fold differences per draw, then mean 24 draws.",
                  "interval_scope": "Pointwise mean +/- t(.975,23) * SD(24 draw means)/sqrt(24); conditional on fixed corpus, panels and program. Not simultaneous bands.",
                  "selection": "Minimize own validation CE or maximize own validation UAR over epochs 1..W; earliest ties. Outer UAR is used only after selection.",
                  "csv_scope": "Archived full-precision floating serialization; original scorer uses Fraction UAR. All W15/45 choices and values checked against original tables.",
                  "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "scipy": scipy.__version__}})
    require(Path(__file__).read_bytes() == source_before, "Script changed during generation")
    for name, info in audit["inputs"].items():
        require(record((repo / name).read_bytes()) == info, "Input changed during generation")
    audit["outputs"] = {p.name: record(p.read_bytes()) for p in sorted(out.iterdir())}
    audit["audit_sha256"] = digest(canonical(audit))
    with (out / "window_selection_audit.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"pass": True, "counts": audit["counts"], "endpoint_checks": audit["endpoint_checks"],
                      "layout": audit["layout"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

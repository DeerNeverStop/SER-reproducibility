"""Descriptors D15, D16 and D18 (estimation only) from a scored run.

D15  RAVDESS CNN under LOSO, size-matched LOSO (LOSOSUB) and Group5 (= CTRL GG): mean over
     replicates of the mean per-speaker UAR, with the speaker-paired LOSO-Group5 difference.
D16  P1 reconciliation: CNN on the six frozen P1 splits (P1RR / P1GG, seeds 0-2) next to the
     2026 P1 rows (RAVDESS CNN Random 56.86 / Grouped 39.15; CREMA-D CNN 58.63 / 55.84, UAR %),
     and the v2 RR / GG cells of the same corpus.
D18  accuracy and macro-F1 per level x model x cell (CTRL main replicates), released without inference.

    python -m tools.summarize_descriptors --plan plan_rc2 --manifests manifests --run runs/main --out runs/main/descriptors/d15_d16_d18.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2 import stats  # noqa: E402
from ser_v2.common import atomic_write_json, read_csv, read_json  # noqa: E402

P1_ROWS = {"ravdess": {"random": 56.86, "groupkfold": 39.15}, "cremad": {"random": 58.63, "groupkfold": 55.84}}
N_CLASSES = {"ravdess": 8, "cremad": 6, "subesco": 7, "subesco_980": 7}


def load_predictions(run: Path, uid: str):
    with open(run / "units" / uid / "predictions.csv", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh); next(rd)
        rows = list(rd)
    y = np.asarray([int(r[3]) for r in rows]); p = np.asarray([int(r[4]) for r in rows]); s = np.asarray([r[2] for r in rows])
    return y, p, s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    plan = read_csv(a.plan / "run_plan.csv")
    results = read_json(a.run / "results.json") if (a.run / "results.json").exists() else None
    done = [r for r in plan if (a.run / "units" / r["unit_id"] / "DONE").exists()]
    # ---- cell-run OOF assembly (main replicates only: seed_index == r for CTRL)
    cells = defaultdict(list)
    for r in done:
        if r["arm"] != "CTRL" or int(r["seed_index"]) != int(r["r"]):
            continue
        cells[(r["corpus_level"], r["model"], r["cell"], int(r["r"]))].append(r)
    speaker_uar, acc_f1 = {}, {}
    for key, rows in cells.items():
        Y, P, S = [], [], []
        for r in rows:
            y, p, s = load_predictions(a.run, r["unit_id"]); Y.append(y); P.append(p); S.append(s)
        y, p, s = np.concatenate(Y), np.concatenate(P), np.concatenate(S)
        K = N_CLASSES[rows[0]["base_corpus"]]
        speaker_uar[key] = {sp: stats.uar(y[s == sp], p[s == sp], K) for sp in sorted(set(s.tolist()))}
        acc_f1[key] = {"n_rows": int(len(y)), "n_folds": len(rows), "accuracy": stats.accuracy(y, p), "macro_f1": stats.macro_f1(y, p, K),
                       "uar_pooled": stats.uar(y, p, K), "mean_speaker_uar": float(np.mean(list(speaker_uar[key].values())))}

    def rep_mean(level, model, cell):
        vals = [speaker_uar[k] for k in speaker_uar if k[0] == level and k[1] == model and k[2] == cell]
        if not vals:
            return None
        spk = sorted(set.intersection(*[set(v) for v in vals]))
        return {s_: float(np.mean([v[s_] for v in vals])) for s_ in spk}, len(vals)

    def summ(d):
        if d is None:
            return {"tested": False}
        arr = np.asarray([d[s_] for s_ in sorted(d)])
        lo, hi = stats.speaker_bootstrap_ci(arr)
        return {"n": int(len(arr)), "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None, "ci_low": lo, "ci_high": hi}

    out = {"D15": {}, "D16": {}, "D18": {}}
    # ---- D15
    v = {c: rep_mean("ravdess", "cnn", c) for c in ("LOSO", "LOSOSUB", "GG")}
    for c, x in v.items():
        out["D15"][c] = {"tested": x is not None, "mean_speaker_uar": float(np.mean(list(x[0].values()))) if x else None, "n_replicates": x[1] if x else 0}
    if v["LOSO"] and v["GG"]:
        d = {s_: v["LOSO"][0][s_] - v["GG"][0][s_] for s_ in v["GG"][0] if s_ in v["LOSO"][0]}
        out["D15"]["LOSO_minus_Group5"] = summ(d)
    if v["LOSOSUB"] and v["GG"]:
        d = {s_: v["LOSOSUB"][0][s_] - v["GG"][0][s_] for s_ in v["GG"][0] if s_ in v["LOSOSUB"][0]}
        out["D15"]["LOSOSUB_minus_Group5"] = summ(d)
    # ---- D16
    for corpus in ("ravdess", "cremad"):
        entry = {"p1_2026": P1_ROWS[corpus]}
        for cell, proto in (("P1RR", "random"), ("P1GG", "groupkfold")):
            vals = [acc_f1[k] for k in acc_f1 if k[0] == corpus and k[1] == "cnn" and k[2] == cell]
            entry[cell] = {"n_seeds": len(vals), "uar_pooled_mean": float(np.mean([x["uar_pooled"] for x in vals])) if vals else None,
                           "mean_speaker_uar": float(np.mean([x["mean_speaker_uar"] for x in vals])) if vals else None,
                           "p1_row": P1_ROWS[corpus][proto]}
        for cell in ("RR", "GG"):
            vals = [acc_f1[k] for k in acc_f1 if k[0] == corpus and k[1] == "cnn" and k[2] == cell]
            entry[f"v2_{cell}"] = {"n_replicates": len(vals), "uar_pooled_mean": float(np.mean([x["uar_pooled"] for x in vals])) if vals else None,
                                   "mean_speaker_uar": float(np.mean([x["mean_speaker_uar"] for x in vals])) if vals else None}
        if entry["P1RR"]["n_seeds"] and entry["P1GG"]["n_seeds"]:
            entry["premium_v2_on_P1_splits"] = entry["P1RR"]["uar_pooled_mean"] - entry["P1GG"]["uar_pooled_mean"]
            entry["premium_p1_2026"] = P1_ROWS[corpus]["random"] - P1_ROWS[corpus]["groupkfold"]
        out["D16"][corpus] = entry
    # ---- D18
    agg = defaultdict(list)
    for (level, model, cell, r), x in acc_f1.items():
        agg[(level, model, cell)].append(x)
    for (level, model, cell), xs in sorted(agg.items()):
        out["D18"][f"{level}|{model}|{cell}"] = {"n_replicates": len(xs),
                                                  "accuracy": float(np.mean([x["accuracy"] for x in xs])), "macro_f1": float(np.mean([x["macro_f1"] for x in xs])),
                                                  "uar_pooled": float(np.mean([x["uar_pooled"] for x in xs])), "mean_speaker_uar": float(np.mean([x["mean_speaker_uar"] for x in xs]))}
    out["n_cell_runs"] = len(acc_f1)
    atomic_write_json(a.out, out)
    print(json.dumps({"D15": out["D15"], "D16": out["D16"]}, indent=1)[:3000])


if __name__ == "__main__":
    main()

"""Deliberately naive (loop-based) second implementation of the SPEC endpoints for the synthetic fixture.
Written independently of v3/deploy/verify.py's vectorised code; shares only the SPEC's definitions and the fixed
bootstrap index rule (default_rng(20260905).integers(0, n, size=(10000, n)), percentile CI, linear)."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

PROGRAM = "SER26-DEPLOY-1"


def boot_idx(n):
    return np.random.default_rng(20260905).integers(0, n, size=(10000, n))


def ci_mean(values):
    vals = [v for v in values]
    idx = boot_idx(len(vals))
    means = []
    for row in idx:
        picked = [vals[i] for i in row if not math.isnan(vals[i])]
        means.append(sum(picked) / len(picked) if picked else float("nan"))
    lo, hi = np.percentile(means, [2.5, 97.5], method="linear")
    good = [v for v in vals if not math.isnan(v)]
    return sum(good) / len(good), [float(lo), float(hi)], len(good)


def ep(**kw):
    rec = {k: v for k, v in kw.items() if v is not None}
    rec["id"] = "naive:" + "|".join(f"{k}={v}" for k, v in sorted(rec.items()) if k not in ("estimate", "ci95", "n"))
    return rec


def recall_table(y_true, y_pred, C):
    """per class recall (percent) or None when the class is absent."""
    out = []
    for c in range(C):
        tot = sum(1 for t in y_true if t == c)
        hit = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        out.append(None if tot == 0 else 100.0 * hit / tot)
    return out


def uar_of(y_true, y_pred, C):
    rs = [r for r in recall_table(y_true, y_pred, C) if r is not None]
    return sum(rs) / len(rs)


def softmax_conf(row):
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return max(e / s for e in ex)


def load_manifest(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["relative_path"]: r for r in rows}, sorted({(int(r["label_index"]), r["label"]) for r in rows})


# ----------------------------------------------------------------------------- module A

def score_a(fx, plan_a, conds_cfg, test_speakers, enc):
    """Per (level): per speaker UAR per (condition, estimator) from the gz file, mean over draws; single r."""
    endpoints, absolute, per_speaker = [], [], []
    for u in plan_a["units"]:
        lvl = u["level"]
        man, classes = load_manifest(fx["manifests_dir"] / f"{fx['level_map'][lvl]}_manifest.csv")
        C = len(classes)
        cls_names = [c[1] for c in classes]
        rows = []
        with gzip.open(fx["run_dir"] / "A" / "units" / u["unit_id"] / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
        U, R = {}, {}   # (cond, est) -> {spk: uar}, (cond, est) -> {spk: [recall_c]}
        keys = sorted({(r["condition"], r["estimator"]) for r in rows})
        for (c, e) in keys:
            U[(c, e)], R[(c, e)] = {}, {}
            draws = sorted({int(r["draw"]) for r in rows if r["condition"] == c and r["estimator"] == e})
            for s in test_speakers:
                us, recs = [], []
                for d in draws:
                    sub = [r for r in rows if r["condition"] == c and r["estimator"] == e and int(r["draw"]) == d and r["speaker"] == s]
                    yt = [int(r["y_true"]) for r in sub]; yp = [int(r["y_pred"]) for r in sub]
                    us.append(uar_of(yt, yp, C)); recs.append(recall_table(yt, yp, C))
                U[(c, e)][s] = sum(us) / len(us)
                R[(c, e)][s] = [sum(rc[k] for rc in recs) / len(recs) for k in range(C)]
        spk = sorted(test_speakers)
        for (c, e) in keys:
            est, ci, n = ci_mean([U[(c, e)][s] for s in spk])
            absolute.append(ep(level=lvl, model=enc, condition=c, estimator=e, quantity="uar", estimate=est, ci95=ci, n=n))
            for s in spk:
                per_speaker.append({"level": lvl, "enc": enc, "condition": c, "estimator": e, "speaker": s, "uar": U[(c, e)][s]})
        none = U[("none", "none")]
        emit = lambda comparison, vals, condition, estimator="naive", quantity="uar": endpoints.append(
            ep(level=lvl, model=enc, comparison=comparison, condition=condition, estimator=estimator, quantity=quantity, **dict(zip(("estimate", "ci95", "n"), ci_mean(vals)))))
        emit("balanced@5-none", [U[("balanced@5", "naive")][s] - none[s] for s in spk], "balanced@5")
        feasible = [c for c, cfg in conds_cfg.items() if cfg.get("feasible", True)]
        single_N = sorted({conds_cfg[c]["N"] for c in feasible if conds_cfg[c]["composition"] == "single"})
        Nmax = single_N[-1]
        singles = lambda N, comp="single": sorted(c for c in feasible if conds_cfg[c]["composition"] == comp and conds_cfg[c]["N"] == N)
        mean_single = {s: sum(U[(c, "naive")][s] for c in singles(Nmax)) / len(singles(Nmax)) for s in spk}
        emit(f"mean_single@{Nmax}-none", [mean_single[s] - none[s] for s in spk], f"single:*@{Nmax}")
        rdiffs = {s: [] for s in spk}
        for c in singles(Nmax):
            k = cls_names.index(c.split(":")[1].split("@")[0])
            d = {s: R[(c, "naive")][s][k] - R[("none", "none")][s][k] for s in spk}
            emit(f"recall:{c}-none", [d[s] for s in spk], c, quantity="recall")
            for s in spk:
                rdiffs[s].append(d[s])
        emit(f"recall:mean_single@{Nmax}-none", [sum(rdiffs[s]) / len(rdiffs[s]) for s in spk], f"single:*@{Nmax}", quantity="recall")
        emit(f"harm_share:single@{Nmax}-none", [1.0 if mean_single[s] - none[s] < -2.0 else 0.0 for s in spk], f"single:*@{Nmax}", quantity="harm_share")
        for c in singles(Nmax):
            emit(f"harm_share:{c}-none", [1.0 if U[(c, "naive")][s] - none[s] < -2.0 else 0.0 for s in spk], c, quantity="harm_share")
        for N in single_N:
            sc = singles(N)
            if f"balanced@{N}" in feasible:
                emit(f"mean_single@{N}-balanced@{N}", [sum(U[(c, "naive")][s] for c in sc) / len(sc) - U[(f"balanced@{N}", "naive")][s] for s in spk], f"single:*@{N}")
            for est in ("shrink", "neutral_filter", "prior_corrected"):
                emit("est-naive@single", [sum(U[(c, est)][s] - U[(c, "naive")][s] for c in sc) / len(sc) for s in spk], f"single:*@{N}", est)
            oc = singles(N, "other_single")
            if oc:
                emit(f"other_single@{N}-single@{N}", [sum(U[(c, "naive")][s] - U[(c[6:], "naive")][s] for c in oc) / len(oc) for s in spk], f"single:*@{N}")
        for N in sorted({conds_cfg[c]["N"] for c in feasible if conds_cfg[c]["composition"] == "balanced"}):
            if f"other_balanced@{N}" in feasible:
                emit(f"other_balanced@{N}-balanced@{N}", [U[(f"other_balanced@{N}", "naive")][s] - U[(f"balanced@{N}", "naive")][s] for s in spk], f"balanced@{N}")
    return endpoints, absolute, per_speaker


# ----------------------------------------------------------------------------- module B1 (one group per level/model)

def read_logits(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    lc = sorted([c for c in rows[0] if c.startswith("logit_")], key=lambda c: int(c.split("_")[1]))
    return [{"path": r["relative_path"], "spk": r["speaker"], "yt": int(r["y_true"]), "yp": int(r["y_pred"]),
             "conf": softmax_conf([float(r[c]) for c in lc])} for r in rows]


def accept_stats(rows, tau, speakers):
    acc = [r for r in rows if r["conf"] >= tau]
    per = {}
    for s in speakers:
        mine = [r for r in rows if r["spk"] == s]; ok = [r for r in mine if r["conf"] >= tau]
        per[s] = {"n": len(mine), "n_acc": len(ok), "n_acc_err": sum(1 for r in ok if r["yt"] != r["yp"]), "n_err": sum(1 for r in mine if r["yt"] != r["yp"]),
                  "cov": len(ok) / len(mine), "err": (sum(1 for r in ok if r["yt"] != r["yp"]) / len(ok)) if ok else float("nan")}
    return acc, per


def pooled_ci(per, num, den):
    spk = sorted(per); idx = boot_idx(len(spk))
    est = sum(per[s][num] for s in spk) / sum(per[s][den] for s in spk)
    samp = [sum(per[spk[i]][num] for i in row) / sum(per[spk[i]][den] for i in row) for row in idx]
    lo, hi = np.percentile(samp, [2.5, 97.5], method="linear")
    return est, [float(lo), float(hi)], len(spk)


def count_ci(per, pred):
    spk = sorted(per); idx = boot_idx(len(spk))
    est = float(sum(1 for s in spk if pred(per[s])))
    samp = [sum(1 for i in row if pred(per[spk[i]])) for row in idx]
    lo, hi = np.percentile(samp, [2.5, 97.5], method="linear")
    return est, [float(lo), float(hi)], len(spk)


def score_b1(fx, level_models):
    endpoints, per_speaker = [], []
    for lvl, model, uid in level_models:
        man, classes = load_manifest(fx["manifests_dir"] / f"{fx['level_map'][lvl]}_manifest.csv")
        C = len(classes)
        rows = read_logits(fx["v2root"] / "runs" / "main" / "units" / uid / "predictions.csv")
        speakers = sorted({r["spk"] for r in rows})
        for b in (0.1, 0.2, 0.3):
            tau = float(np.quantile([r["conf"] for r in rows], b, method="linear"))
            acc, per = accept_stats(rows, tau, speakers)
            add = lambda q, e, ci, n: endpoints.append(ep(level=lvl, model=model, comparison="absolute", b=b, quantity=q, estimate=e, ci95=ci, n=n))
            add("accepted_error", *pooled_ci(per, "n_acc_err", "n_acc"))
            e_, ci, n = pooled_ci(per, "n_err", "n_err")  # placeholder shape; capture computed below
            rej_err = {s: per[s]["n_err"] - per[s]["n_acc_err"] for s in speakers}
            for s in speakers:
                per[s]["n_rej_err"] = rej_err[s]
            add("error_capture_rate", *pooled_ci(per, "n_rej_err", "n_err"))
            qper = {}
            for s in speakers:
                mine = [r for r in rows if r["spk"] == s]
                t_s = float(np.quantile([r["conf"] for r in mine], b, method="linear"))
                ok = [r for r in mine if r["conf"] >= t_s]
                qper[s] = {"n_acc": len(ok), "n_acc_err": sum(1 for r in ok if r["yt"] != r["yp"])}
            add("quota_accepted_error", *pooled_ci(qper, "n_acc_err", "n_acc"))
            cnt, cci, cn = count_ci(per, lambda p: p["cov"] < 0.70)
            add("count_speakers_coverage_below_0.70", cnt, cci, cn)
            add("share_speakers_coverage_below_0.70", cnt / cn, [cci[0] / cn, cci[1] / cn], cn)
            add("count_speakers_coverage_zero", *count_ci(per, lambda p: p["n_acc"] == 0))
            covs = [per[s]["cov"] for s in speakers]; idx = boot_idx(len(speakers))
            mins = [min(covs[i] for i in row) for row in idx]; p10s = [float(np.percentile([covs[i] for i in row], 10, method="linear")) for row in idx]
            add("coverage_min", min(covs), np.percentile(mins, [2.5, 97.5], method="linear").tolist(), len(speakers))
            add("coverage_p10", float(np.percentile(covs, 10, method="linear")), np.percentile(p10s, [2.5, 97.5], method="linear").tolist(), len(speakers))
            add("accepted_uar", uar_of([r["yt"] for r in acc], [r["yp"] for r in acc], C), [float("nan"), float("nan")], len(speakers))
            rej = [r for r in rows if r["conf"] < tau]
            for k, (_, name) in enumerate(classes):
                add(f"rejected_share_{name}", sum(1 for r in rej if r["yt"] == k) / len(rej), [float("nan"), float("nan")], len(speakers))
            for s in speakers:
                per_speaker.append({"level": lvl, "model": model, "b": b, "speaker": s, "coverage": per[s]["cov"], "accepted_error": per[s]["err"], "n_utts": per[s]["n"]})
        ordered = sorted(range(len(rows)), key=lambda i: (-rows[i]["conf"], i))
        risks, errs = [], 0
        for k, i in enumerate(ordered, start=1):
            errs += 1 if rows[i]["yt"] != rows[i]["yp"] else 0
            risks.append(errs / k)
        n_corr = sum(1 for r in rows if r["yt"] == r["yp"])
        endpoints.append(ep(level=lvl, model=model, comparison="absolute", quantity="aurc", estimate=sum(risks) / len(risks), ci95=[float("nan"), float("nan")], n=len(speakers)))
        endpoints.append(ep(level=lvl, model=model, comparison="absolute", quantity="oracle_aurc", estimate=sum(max(0, k - n_corr) / k for k in range(1, len(rows) + 1)) / len(rows), ci95=[float("nan"), float("nan")], n=len(speakers)))
    return endpoints, per_speaker


# ----------------------------------------------------------------------------- module B2 (one r, one fold)

def b2_unit(fx, u):
    udir = fx["run_dir"] / "B2" / "units" / u["unit_id"]
    val, test = read_logits(udir / "val_predictions.csv"), read_logits(udir / "test_predictions.csv")
    speakers = sorted({r["spk"] for r in test})
    tau = float(np.quantile([r["conf"] for r in val], 0.2, method="linear"))
    val_acc = [r for r in val if r["conf"] >= tau]
    r_hat = sum(1 for r in val_acc if r["yt"] != r["yp"]) / len(val_acc)
    _, per = accept_stats(test, tau, speakers)
    test_acc = [r for r in test if r["conf"] >= tau]
    # secondary rule
    r_star = 0.5 * sum(1 for r in val if r["yt"] != r["yp"]) / len(val)
    srt = sorted(val, key=lambda r: r["conf"])
    k = None
    for kk in range(len(val) + 1):
        rest = srt[kk:]
        err = (sum(1 for r in rest if r["yt"] != r["yp"]) / len(rest)) if rest else 0.0
        if err <= r_star + 1e-15:
            k = kk; break
    tau2 = srt[k - 1]["conf"] if k > 0 else -math.inf
    _, per2 = accept_stats(test, tau2, speakers)
    test_acc2 = [r for r in test if r["conf"] >= tau2]
    return {"speakers": speakers, "r_hat": r_hat, "per": per, "per2": per2, "r_star": r_star, "k_frac": k / len(val),
            "cov_test": len(test_acc) / len(test), "err_test": sum(1 for r in test_acc if r["yt"] != r["yp"]) / len(test_acc),
            "cov_test2": len(test_acc2) / len(test), "err_test2": (sum(1 for r in test_acc2 if r["yt"] != r["yp"]) / len(test_acc2)) if test_acc2 else float("nan")}


def score_b2(fx, plan_b2):
    endpoints, absolute, per_speaker = [], [], []
    by = {}
    for u in plan_b2["units"]:
        by.setdefault((u["level"], u["model"]), {})[u["cell"]] = b2_unit(fx, u)
    for (lvl, model), cells in sorted(by.items()):
        gg, gr = cells["GG"], cells["GR"]
        spk = gg["speakers"]
        def diff(fn):
            return [fn(gr, s) - fn(gg, s) for s in spk]
        for q, fn in (("realised_coverage", lambda c, s: c["per"][s]["cov"]), ("realised_accepted_error", lambda c, s: c["per"][s]["err"]),
                      ("gap_error", lambda c, s: c["per"][s]["err"] - c["r_hat"]), ("secondary_test_coverage", lambda c, s: c["per2"][s]["cov"]),
                      ("secondary_test_accepted_error", lambda c, s: c["per2"][s]["err"])):
            e, ci, n = ci_mean(diff(fn))
            endpoints.append(ep(level=lvl, model=model, comparison="GR-GG", quantity=q, estimate=e, ci95=ci, n=n))
        for cell, c in (("GG", gg), ("GR", gr)):
            add = lambda q, e, ci, n: absolute.append(ep(level=lvl, model=model, cell=cell, quantity=q, estimate=e, ci95=ci, n=n))
            add("realised_coverage", *ci_mean([c["per"][s]["cov"] for s in spk]))
            add("realised_accepted_error", *ci_mean([c["per"][s]["err"] for s in spk]))
            add("secondary_test_coverage", *ci_mean([c["per2"][s]["cov"] for s in spk]))
            add("secondary_test_accepted_error", *ci_mean([c["per2"][s]["err"] for s in spk]))
            nan = [float("nan"), float("nan")]
            add("promised_accepted_error", c["r_hat"], nan, len(spk)); add("realised_coverage_pooled", c["cov_test"], nan, len(spk))
            add("realised_accepted_error_pooled", c["err_test"], nan, len(spk)); add("target_risk", c["r_star"], nan, len(spk))
            add("secondary_val_reject_fraction", c["k_frac"], nan, len(spk)); add("gap_coverage_pooled", c["cov_test"] - 0.8, nan, len(spk))
            add("gap_error_pooled", c["err_test"] - c["r_hat"], nan, len(spk))
            covs = [c["per"][s]["cov"] for s in spk]
            add("coverage_min", min(covs), nan, len(spk)); add("coverage_p10", float(np.percentile(covs, 10, method="linear")), nan, len(spk))
            add("n_cov_below_70", float(sum(1 for v in covs if v < 0.7)), nan, len(spk)); add("n_cov_zero", float(sum(1 for v in covs if v == 0)), nan, len(spk))
            for s in spk:
                per_speaker.append({"level": lvl, "model": model, "cell": cell, "speaker": s, "coverage": c["per"][s]["cov"], "accepted_error": c["per"][s]["err"]})
    return endpoints, absolute, per_speaker


def write_results(results_dir: Path, module: str, endpoints, absolute, per_speaker, columns):
    d = results_dir / module
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "endpoints.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"program": PROGRAM, "module": module, "endpoints": endpoints, "absolute": absolute, "inputs": {"fixture": True}}, fh, indent=1, sort_keys=True)
    with open(d / "per_speaker.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(columns) + "\n")
        for r in per_speaker:
            fh.write(",".join(repr(r[c]) if isinstance(r[c], float) else str(r[c]) for c in columns) + "\n")

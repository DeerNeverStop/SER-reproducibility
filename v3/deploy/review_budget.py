"""B1 - human review budget on existing speaker-exclusive (GG) v2 predictions (SER26-DEPLOY-1, SPEC section 3).

CLI (worktree root, env SER_V2_DATA_ROOT, CPU only)
  python -m v3.deploy.review_budget run --out v3/deploy/results/B1 [--work v3/deploy/work/B1]
                                        [--levels ravdess,cremad] [--models cnn,wavlm_base_plus_ft]

Sources : v2 runs/main/units with arm=CTRL, cell=GG, level in {ravdess, cremad, subesco_980} (all models) and
          arm=FT, cell=GG, model=wavlm_base_plus_ft (seed_index 0,1).  predictions.csv is verified against
          unit.json["predictions_sha256"] before use.
Pooling : one replicate = (level, model, r, seed_index); its five outer folds are concatenated (OOF predictions,
          every population speaker exactly once).
Rules   : confidence = max softmax(logits); for b in {0.10, 0.20, 0.30}: tau_b = numpy.quantile(conf, b, "linear"),
          accepted <=> conf >= tau_b.  Reported per replicate then averaged over replicates of the same (level, model):
          accepted-set error, error capture rate (errors rejected / all errors), accepted-set UAR (common.uar),
          per-speaker coverage profile (min, p10, count < 0.70, count == 0), rejected-set class composition,
          per-speaker equal-quota variant (each speaker rejects its own bottom b), AURC (max-prob ranking, ties broken
          by stable order) and oracle AURC (correct predictions ranked first).
CIs     : speaker-cluster percentile bootstrap (common.bootstrap_indices) for accepted error at b=0.20 and the error
          capture rate (ratio of per-speaker sums, threshold held fixed, averaged over replicates), and for the share
          of speakers with coverage < 0.70 (per-speaker indicator mean; the count is share x n_speakers).
Outputs : <out>/per_speaker.csv, summary.csv, rejected_class_mix.csv, endpoints.json;
          <work>/<level>__<model>__r<r>s<seed>.csv per-replicate per-speaker tables (SPEC 5.2).
"""
from __future__ import annotations

import argparse
import hashlib
import csv
from pathlib import Path

import numpy as np

from . import common
from .common import (DEPLOY_ROOT, PROGRAM, REPO_ROOT, V2_DATA_ROOT, atomic_write_json, atomic_write_text,
                     bootstrap_indices, class_table, load_manifest, LEVELS, now, read_json, read_predictions_csv,
                     require, sha256_file, softmax, uar)

MODULE = "B1"
SPEC_VERSION = "DEPLOY-2"
BUDGETS = (0.10, 0.20, 0.30)
PRIMARY_BUDGET = 0.20
B1_LEVELS = ("ravdess", "cremad", "subesco_980")
FT_MODEL = "wavlm_base_plus_ft"
LOW_COVERAGE = 0.70


# ============================================================================= v2 unit selection / loading

def read_v2_plan_rows() -> list:
    with open(REPO_ROOT / "v2" / "plan_rc2" / "run_plan.csv", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def select_v2_units(v2_rows: list, levels=B1_LEVELS, models=None) -> list:
    out = []
    for r in v2_rows:
        if r["cell"] != "GG" or r["corpus_level"] not in levels:
            continue
        if not ((r["arm"] == "CTRL") or (r["arm"] == "FT" and r["model"] == FT_MODEL)):
            continue
        if models and r["model"] not in models:
            continue
        out.append(r)
    out.sort(key=lambda r: (r["corpus_level"], r["model"], int(r["r"]), int(r["seed_index"]), int(r["fold"])))
    return out


def load_v2_unit(row: dict) -> dict:
    udir = V2_DATA_ROOT / "runs" / "main" / "units" / row["unit_id"]
    unit = read_json(udir / "unit.json")
    require(unit.get("status") == "done" and unit["unit_id"] == row["unit_id"], f"v2 unit {row['unit_id']} not done")
    sha = sha256_file(udir / "predictions.csv")
    require(sha == unit["predictions_sha256"], f"v2 unit {row['unit_id']}: predictions.csv sha mismatch")
    pred = read_predictions_csv(udir / "predictions.csv", "logit")
    require(np.array_equal(pred["scores"].argmax(axis=1), pred["y_pred"]), f"v2 unit {row['unit_id']}: y_pred != argmax")
    return {"row": row, "unit": unit, "pred": pred, "predictions_sha256": sha}


def pool_replicates(loaded: list) -> dict:
    """(level, model, r, seed_index) -> pooled OOF dict (paths, speakers, y_true, y_pred, scores, folds)."""
    groups = {}
    for L in loaded:
        r = L["row"]
        groups.setdefault((r["corpus_level"], r["model"], int(r["r"]), int(r["seed_index"])), []).append(L)
    pooled = {}
    for key, items in sorted(groups.items()):
        folds = sorted(int(L["row"]["fold"]) for L in items)
        require(folds == list(range(5)), f"{key}: expected folds 0..4, got {folds}")
        items.sort(key=lambda L: int(L["row"]["fold"]))
        d = {"paths": [], "speakers": [], "y_true": [], "y_pred": [], "scores": [], "fold": []}
        spk_fold = {}
        for L in items:
            p = L["pred"]
            for s in set(p["speakers"]):
                require(s not in spk_fold, f"{key}: speaker {s} in two test folds")
                spk_fold[s] = int(L["row"]["fold"])
            d["paths"] += p["paths"]
            d["speakers"] += p["speakers"]
            d["y_true"].append(p["y_true"])
            d["y_pred"].append(p["y_pred"])
            d["scores"].append(p["scores"])
            d["fold"] += [int(L["row"]["fold"])] * len(p["paths"])
        require(len(set(d["paths"])) == len(d["paths"]), f"{key}: duplicate paths across folds")
        pooled[key] = {"paths": d["paths"], "speakers": np.asarray(d["speakers"]), "y_true": np.concatenate(d["y_true"]),
                       "y_pred": np.concatenate(d["y_pred"]), "scores": np.vstack(d["scores"]), "fold": np.asarray(d["fold"]),
                       "unit_ids": [L["row"]["unit_id"] for L in items]}
    return pooled


# ============================================================================= metrics (pure)

def confidence_of(logits: np.ndarray) -> np.ndarray:
    return softmax(logits).max(axis=1)


def quantile_tau(conf: np.ndarray, b: float) -> float:
    return float(np.quantile(np.asarray(conf, dtype=np.float64), b, method="linear"))


def quota_accept(conf, b, paths=None):
    """Reject floor(b*n), ties by SHA256(UTF8 relative_path), then path.

    The ordinal fallback supports synthetic pure-function tests only; production
    score_replicate always receives recorded paths from pool_replicates.
    """
    conf = np.asarray(conf, dtype=float)
    require(conf.ndim == 1 and conf.size and np.isfinite(conf).all() and 0 <= b <= 1, "invalid quota inputs")
    paths = list(paths) if paths is not None else [str(i) for i in range(len(conf))]
    require(len(paths) == len(conf) and len(set(paths)) == len(paths), "quota paths must be unique and aligned")
    order = sorted(range(len(conf)), key=lambda i: (conf[i], hashlib.sha256(paths[i].encode("utf-8")).hexdigest(), paths[i]))
    acc = np.ones(len(conf), dtype=bool)
    acc[order[:int(np.floor(b * len(conf)))]] = False
    return acc


def safe_ratio(num, den) -> float:
    return float(num) / float(den) if den else float("nan")


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def per_speaker_table(speakers: np.ndarray, acc: np.ndarray, err: np.ndarray) -> dict:
    """speaker -> n, n_acc, n_err, n_err_acc, n_err_rej, coverage, accepted_error."""
    out = {}
    for s in sorted(set(speakers.tolist())):
        m = speakers == s
        a, e = acc[m], err[m]
        n_acc, n_err, n_err_acc = int(a.sum()), int(e.sum()), int((e & a).sum())
        out[s] = {"n": int(m.sum()), "n_acc": n_acc, "n_err": n_err, "n_err_acc": n_err_acc, "n_err_rej": n_err - n_err_acc,
                  "coverage": n_acc / int(m.sum()), "accepted_error": safe_ratio(n_err_acc, n_acc)}
    return out


def coverage_profile(cov: np.ndarray) -> dict:
    cov = np.asarray(cov, dtype=np.float64)
    return {"speaker_coverage_min": float(cov.min()), "speaker_coverage_p10": float(np.percentile(cov, 10, method="linear")),
            "speakers_below_0.70": int((cov < LOW_COVERAGE).sum()), "speakers_zero": int((cov == 0).sum())}


def quota_variant(conf: np.ndarray, err: np.ndarray, speakers: np.ndarray, b: float, paths=None) -> dict:
    """Each speaker rejects floor(b*n_s), resolving ties by path SHA."""
    acc = np.zeros(conf.shape[0], dtype=bool)
    paths = np.asarray(paths if paths is not None else [str(i) for i in range(len(conf))])
    for s in set(speakers.tolist()):
        m = speakers == s
        acc[m] = quota_accept(conf[m], b, paths[m])
    return {"quota_accepted_error": safe_ratio((err & acc).sum(), acc.sum()), "quota_coverage": float(acc.mean()),
            "quota_error_capture_rate": safe_ratio((err & ~acc).sum(), err.sum())}


def aurc(conf: np.ndarray, err: np.ndarray, paths=None) -> dict:
    """Area under the risk-coverage curve, mean over k=1..n of (errors among the k most confident)/k.
    oracle: correct predictions first, so risk_k = max(0, k - n_correct)/k."""
    err = np.asarray(err, dtype=np.float64)
    n = err.shape[0]
    require(n > 0, "empty AURC")
    paths = list(paths) if paths is not None else [str(i) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-float(conf[i]), hashlib.sha256(paths[i].encode("utf-8")).hexdigest(), paths[i]))
    k = np.arange(1, n + 1, dtype=np.float64)
    risk = np.cumsum(err[order]) / k
    n_correct = n - err.sum()
    oracle = np.maximum(0.0, k - n_correct) / k
    a, o = float(risk.mean()), float(oracle.mean())
    return {"aurc": a, "oracle_aurc": o, "e_aurc": a - o}


def budget_metrics(conf, err, y_true, y_pred, speakers, b: float, n_classes: int, paths=None, folds=None) -> dict:
    conf, err = np.asarray(conf, dtype=np.float64), np.asarray(err, dtype=bool)
    y_true, y_pred, speakers = np.asarray(y_true), np.asarray(y_pred), np.asarray(speakers)
    tau = quantile_tau(conf, b)
    if folds is None:
        acc = quota_accept(conf, b, paths)
    else:
        folds = np.asarray(folds)
        path_array = np.asarray(paths if paths is not None else [str(i) for i in range(len(conf))])
        acc = np.zeros(len(conf), bool)
        for fold in sorted(set(folds.tolist())):
            mask = folds == fold
            acc[mask] = quota_accept(conf[mask], b, path_array[mask])
    n_err = int(err.sum())
    support = np.bincount(y_true[acc], minlength=n_classes)
    m = {"b": b, "tau": tau, "coverage": float(acc.mean()), "accepted_error": safe_ratio((err & acc).sum(), acc.sum()),
         "error_capture_rate": safe_ratio((err & ~acc).sum(), n_err),
         "accepted_uar": uar(y_true[acc], y_pred[acc], n_classes) if (support > 0).all() else float("nan"),
         "accepted_present_class_uar": uar(y_true[acc], y_pred[acc], n_classes) if acc.any() else float("nan"),
         "accepted_classes_present": int((support > 0).sum()), "n_rejected": int((~acc).sum()),
         "actual_review_rate": float((~acc).mean()),
         "overall_error_rate": float(err.mean()), "overall_uar": uar(y_true, y_pred, n_classes), "n_utts": int(conf.shape[0])}
    m.update({f"accepted_support_{c}": int(support[c]) for c in range(n_classes)})
    spk = per_speaker_table(speakers, acc, err)
    risks = np.asarray([v["accepted_error"] for v in spk.values()])
    m["speaker_equal_accepted_error"] = float(risks[np.isfinite(risks)].mean()) if np.isfinite(risks).any() else float("nan")
    m["speaker_risk_undefined_count"] = int((~np.isfinite(risks)).sum())
    m.update(coverage_profile(np.asarray([v["coverage"] for v in spk.values()])))
    m.update(quota_variant(conf, err, speakers, b, paths))
    rej = ~acc
    n_rej = int(rej.sum())
    mix = []
    for c in range(n_classes):
        in_c = y_true == c
        mix.append({"class_index": c, "rejected_count": int((rej & in_c).sum()), "rejected_share": safe_ratio((rej & in_c).sum(), n_rej),
                    "class_share_overall": float(in_c.mean()), "class_rejection_rate": safe_ratio((rej & in_c).sum(), in_c.sum())})
    return {"metrics": m, "speakers": spk, "rejected_mix": mix}


def score_replicate(pooled: dict, n_classes: int) -> dict:
    conf = confidence_of(pooled["scores"])
    err = pooled["y_pred"] != pooled["y_true"]
    paths = pooled.get("paths")
    out = {"budgets": {}, "aurc": aurc(conf, err, paths), "n_utts": int(conf.shape[0]), "n_speakers": len(set(pooled["speakers"].tolist()))}
    for b in BUDGETS:
        out["budgets"][b] = budget_metrics(conf, err, pooled["y_true"], pooled["y_pred"], pooled["speakers"], b, n_classes, paths)
    if "fold" in pooled:
        out["fold_budgets"] = {b: budget_metrics(conf, err, pooled["y_true"], pooled["y_pred"], pooled["speakers"], b, n_classes, paths, pooled["fold"]) for b in BUDGETS}
        fold_results = [(int((pooled["fold"] == f).sum()), aurc(conf[pooled["fold"] == f], err[pooled["fold"] == f],
                         np.asarray(paths)[pooled["fold"] == f])) for f in sorted(set(pooled["fold"].tolist()))]
        out["fold_aurc"] = {k: sum(n * a[k] for n, a in fold_results) / len(conf) for k in out["aurc"]}
    return out


# ============================================================================= aggregation over replicates

def bootstrap_ratio(nums: np.ndarray, dens: np.ndarray, indices: np.ndarray) -> dict:
    """nums/dens: (n_rep, n_speakers). Estimate = mean over replicates of pooled ratio; CI from speaker resampling
    (ratio of resampled sums per replicate, then mean over replicates)."""
    nums, dens = np.asarray(nums, dtype=np.float64), np.asarray(dens, dtype=np.float64)
    est = float(np.mean([safe_ratio(nums[i].sum(), dens[i].sum()) for i in range(nums.shape[0])]))
    acc = np.zeros(indices.shape[0], dtype=np.float64)
    for i in range(nums.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            acc += nums[i][indices].sum(axis=1) / dens[i][indices].sum(axis=1)
    acc /= nums.shape[0]
    finite = np.isfinite(acc)
    lo, hi = np.percentile(acc[finite], [2.5, 97.5], method="linear") if finite.any() else (np.nan, np.nan)
    return {"estimate": est, "ci95": [float(lo), float(hi)], "n": int(nums.shape[1]), "n_undefined_bootstrap": int((~finite).sum())}


def bootstrap_indicator(ind: np.ndarray, indices: np.ndarray) -> dict:
    """ind: (n_rep, n_speakers) 0/1; per-speaker indicator averaged over replicates, then speaker bootstrap of the mean."""
    v = np.asarray(ind, dtype=np.float64).mean(axis=0)
    return common.summarize_paired(v, indices)


def aggregate_group(level: str, model: str, reps: dict, n_classes: int, classes: list) -> dict:
    """reps: rep_name -> score_replicate output (same speakers in every replicate)."""
    names = sorted(reps)
    speakers = sorted(reps[names[0]]["budgets"][BUDGETS[0]]["speakers"])
    for nm in names:
        require(sorted(reps[nm]["budgets"][BUDGETS[0]]["speakers"]) == speakers, f"{level}/{model}: replicate speakers differ")
    idx = bootstrap_indices(len(speakers))
    summary_rows, per_speaker_rows, mix_rows, endpoints = [], [], [], []
    a = {k: float(np.mean([reps[nm]["aurc"][k] for nm in names])) for k in ("aurc", "oracle_aurc", "e_aurc")}
    for b in BUDGETS:
        ms = [reps[nm]["budgets"][b]["metrics"] for nm in names]
        row = {"level": level, "model": model, "b": b, "n_reps": len(names), "n_speakers": len(speakers)}
        for k in ms[0]:
            if k == "b":
                continue
            vals = np.asarray([m[k] for m in ms], dtype=np.float64)
            row[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        row.update(a)
        summary_rows.append(row)
        tabs = [reps[nm]["budgets"][b]["speakers"] for nm in names]
        cov = np.asarray([[t[s]["coverage"] for s in speakers] for t in tabs])
        n_acc = np.asarray([[t[s]["n_acc"] for s in speakers] for t in tabs])
        n_err_acc = np.asarray([[t[s]["n_err_acc"] for s in speakers] for t in tabs])
        n_err = np.asarray([[t[s]["n_err"] for s in speakers] for t in tabs])
        n_err_rej = np.asarray([[t[s]["n_err_rej"] for s in speakers] for t in tabs])
        n_utts = [tabs[0][s]["n"] for s in speakers]
        with np.errstate(invalid="ignore", divide="ignore"):
            acc_err = np.where(n_acc > 0, n_err_acc / np.maximum(n_acc, 1), np.nan)
            acc_err_mean = acc_err.mean(axis=0)
        for i, s in enumerate(speakers):
            per_speaker_rows.append({"level": level, "model": model, "b": b, "speaker": s, "coverage": float(cov[:, i].mean()),
                                     "accepted_error": float(acc_err_mean[i]), "n_utts": n_utts[i],
                                     "n_reps": len(names), "n_defined_risk_reps": int(np.isfinite(acc_err[:, i]).sum())})
        for c in range(n_classes):
            vals = [reps[nm]["budgets"][b]["rejected_mix"][c] for nm in names]
            mix_rows.append({"level": level, "model": model, "b": b, "class_index": c, "class_label": classes[c],
                             "rejected_count": float(np.mean([v["rejected_count"] for v in vals])),
                             "rejected_share": finite_mean([v["rejected_share"] for v in vals]),
                             "class_share_overall": float(np.mean([v["class_share_overall"] for v in vals])),
                             "class_rejection_rate": finite_mean([v["class_rejection_rate"] for v in vals])})
        if b == PRIMARY_BUDGET:
            base = {"level": level, "model": model, "comparison": "absolute", "b": b, "n_reps": len(names)}
            e = bootstrap_ratio(n_err_acc, n_acc, idx)
            endpoints.append(dict(base, id=f"B1:{level}:{model}:accepted_error@b={b}", quantity="accepted_error", **e))
            e = bootstrap_ratio(n_err_rej, n_err, idx)
            endpoints.append(dict(base, id=f"B1:{level}:{model}:error_capture_rate@b={b}", quantity="error_capture_rate", **e))
            e = bootstrap_indicator(cov < LOW_COVERAGE, idx)
            endpoints.append(dict(base, id=f"B1:{level}:{model}:share_speakers_coverage_below_0.70@b={b}",
                                  quantity="share_speakers_coverage_below_0.70", **e))
            n = len(speakers)
            endpoints.append(dict(base, id=f"B1:{level}:{model}:count_speakers_coverage_below_0.70@b={b}",
                                  quantity="count_speakers_coverage_below_0.70", estimate=e["estimate"] * n,
                                  ci95=[e["ci95"][0] * n, e["ci95"][1] * n], n=n))
    result = {"summary": summary_rows, "per_speaker": per_speaker_rows, "mix": mix_rows, "endpoints": endpoints}
    if all("fold_budgets" in reps[nm] for nm in names):
        fold_reps = {nm: {"budgets": reps[nm]["fold_budgets"], "aurc": reps[nm]["fold_aurc"]} for nm in names}
        result["fold_sensitivity"] = aggregate_group(level, model, fold_reps, n_classes, classes)
    return result


# ============================================================================= io

def write_csv(path: Path, rows: list, columns: list) -> None:
    lines = [",".join(columns)]
    for r in rows:
        lines.append(",".join(_fmt(r.get(c)) for c in columns))
    atomic_write_text(path, "\n".join(lines) + "\n")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def write_replicate_table(work: Path, level: str, model: str, rep: str, scored: dict) -> None:
    atomic_write_json(work / f"{level}__{model}__{rep}.json", scored)
    rows = []
    for b in BUDGETS:
        for s, v in scored["budgets"][b]["speakers"].items():
            rows.append(dict({"level": level, "model": model, "rep": rep, "b": b, "speaker": s}, **v))
    cols = ["level", "model", "rep", "b", "speaker", "coverage", "accepted_error", "n", "n_acc", "n_err", "n_err_acc", "n_err_rej"]
    write_csv(work / f"{level}__{model}__{rep}.csv", rows, cols)


def run(out: Path, work: Path, levels=B1_LEVELS, models=None) -> dict:
    v2_rows = read_v2_plan_rows()
    selected = select_v2_units(v2_rows, levels, models)
    require(selected, "no v2 units selected")
    print(f"[B1] {len(selected)} v2 units selected", flush=True)
    loaded = []
    for i, r in enumerate(selected):
        loaded.append(load_v2_unit(r))
        if (i + 1) % 50 == 0:
            print(f"[B1] loaded {i + 1}/{len(selected)}", flush=True)
    pooled = pool_replicates(loaded)
    manifests = {lv: load_manifest(LEVELS[lv]) for lv in sorted({k[0] for k in pooled})}
    classes = {lv: class_table(m) for lv, m in manifests.items()}
    groups = {}
    work.mkdir(parents=True, exist_ok=True)
    for (level, model, r, seed), P in pooled.items():
        n_classes = len(classes[level])
        require(P["scores"].shape[1] == n_classes, f"{level}/{model}: logit width != n_classes")
        rep = f"r{r}s{seed}"
        scored = score_replicate(P, n_classes)
        write_replicate_table(work, level, model, rep, scored)
        groups.setdefault((level, model), {})[rep] = scored
    print(f"[B1] {len(pooled)} replicates scored in {len(groups)} (level, model) groups", flush=True)
    summary, per_speaker, mix, endpoints, fold_sensitivity = [], [], [], [], []
    for (level, model), reps in sorted(groups.items()):
        agg = aggregate_group(level, model, reps, len(classes[level]), classes[level])
        summary += agg["summary"]
        per_speaker += agg["per_speaker"]
        mix += agg["mix"]
        endpoints += agg["endpoints"]
        fold_sensitivity.append(dict(level=level, model=model, **agg["fold_sensitivity"]))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_speaker.csv", per_speaker, list(per_speaker[0]))
    write_csv(out / "summary.csv", summary, list(summary[0].keys()))
    write_csv(out / "rejected_class_mix.csv", mix, list(mix[0].keys()))
    result = {
        "program": PROGRAM, "module": MODULE, "spec": SPEC_VERSION, "scored_at": now(), "endpoints": endpoints,
        "fold_sensitivity": fold_sensitivity,
        "inputs": {"levels": list(levels), "models": sorted({r["model"] for r in selected}), "budgets": list(BUDGETS),
                   "primary_budget": PRIMARY_BUDGET, "low_coverage": LOW_COVERAGE, "n_units": len(selected),
                   "units": [{"unit_id": L["row"]["unit_id"], "predictions_sha256": L["predictions_sha256"]} for L in loaded],
                   "replicates": {f"{k[0]}|{k[1]}|r{k[2]}s{k[3]}": P["unit_ids"] for k, P in pooled.items()},
                   "v2_run_plan_sha256": sha256_file(REPO_ROOT / "v2" / "plan_rc2" / "run_plan.csv"),
                   "bootstrap": {"seed": common.BOOTSTRAP_SEED, "reps": common.BOOTSTRAP_REPS, "unit": "speaker",
                                 "ci": "percentile 2.5/97.5", "threshold": "full-sample quota masks held fixed; tau is diagnostic only"},
                   "quota_rule": "floor(b*n); reject sorted(confidence,SHA256(path),path) ascending; labels not used",
                   "aurc_ties": "confidence descending, SHA256(path) ascending, path ascending; scale 0..1",
                   "accepted_uar_rule": "fixed class support required, otherwise NA; present-class UAR explicitly separate",
                   "code_sha256": {"review_budget.py": sha256_file(Path(__file__)), "common.py": sha256_file(DEPLOY_ROOT / "common.py")}},
    }
    atomic_write_json(out / "endpoints.json", result)
    print(f"[B1] wrote {len(endpoints)} endpoints, {len(summary)} summary rows -> {out}", flush=True)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SER26-DEPLOY-1 module B1 (review budget on v2 GG predictions)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", default=str(DEPLOY_ROOT / "results" / "B1"))
    p.add_argument("--work", default=str(DEPLOY_ROOT / "work" / "B1"))
    p.add_argument("--levels", default=",".join(B1_LEVELS))
    p.add_argument("--models", default="", help="comma list restriction (pilot / format checks only)")
    args = ap.parse_args(argv)
    common.cpu_guard(2)
    levels = tuple(x for x in args.levels.split(",") if x)
    models = [x for x in args.models.split(",") if x] or None
    run(Path(args.out), Path(args.work), levels, models)


if __name__ == "__main__":
    main()

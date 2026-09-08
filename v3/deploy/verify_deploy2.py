"""Independent DEPLOY-2 metadata and raw-prediction checks.

This module never imports a runner, estimator, calibration, or scoring module.
Shared verify.py contains independent CSV/manifest readers and confusion counts.
Numeric rules are implemented here with numpy and Python; result identifiers are
only adapters, never a source of thresholds, predictions, or bootstrap indices.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from . import verify as V

NAN = float("nan")


def quota(conf, paths, budget):
    """Exactly floor(b*n) rejections; label-blind deterministic SHA path ties."""
    conf = np.asarray(conf, dtype=float)
    if len(paths) != len(conf) or len(set(paths)) != len(paths):
        raise V.VerificationError("quota requires unique aligned paths")
    order = sorted(range(len(paths)), key=lambda i: (conf[i], hashlib.sha256(paths[i].encode()).hexdigest(), paths[i]))
    accepted = np.ones(len(paths), dtype=bool)
    accepted[order[:int(np.floor(budget * len(paths)))]] = False
    return accepted


def risk_threshold(conf, incorrect, minimum=1):
    """Enumerate deployable confidence thresholds, honoring ties and nonempty acceptance."""
    conf, incorrect = np.asarray(conf, dtype=float), np.asarray(incorrect, dtype=bool)
    target = 0.5 * float(incorrect.mean())
    for threshold in sorted(set(conf.tolist())):
        accepted = conf >= threshold
        if accepted.sum() >= minimum and float(incorrect[accepted].mean()) <= target:
            return dict(r_star=target, tau=threshold, val_rejection=float((~accepted).mean()),
                        val_coverage=float(accepted.mean()), val_accepted_error=float(incorrect[accepted].mean()),
                        feasible=True, min_accepted=minimum)
    return dict(r_star=target, tau=float("inf"), val_rejection=1., val_coverage=0., val_accepted_error=NAN,
                feasible=False, min_accepted=minimum)


def interval(values):
    """One common full-speaker index matrix; undefined speakers remain in index frame."""
    values = np.asarray(values, dtype=float)
    indices = V.bootstrap_indices(len(values))
    finite = np.isfinite(values)
    if not finite.any():
        return {"estimate": NAN, "ci95": [NAN, NAN], "n": 0, "n_dropped": len(values)}
    draws = values[indices]
    counts = np.isfinite(draws).sum(axis=1)
    means = np.divide(np.nansum(draws, axis=1), counts, out=np.full(len(counts), np.nan), where=counts > 0)
    bounds = np.nanpercentile(means, [2.5, 97.5]).tolist()
    return {"estimate": float(values[finite].mean()), "ci95": bounds, "n": int(finite.sum()), "n_dropped": int((~finite).sum()),
            "n_population": len(values), "n_undefined_bootstrap": int((counts == 0).sum())}


def assert_numeric(expected, actual, where="", tolerance=1e-9):
    """Compare a declared independent numeric subtree. NaNs and same-sign infinities are explicit."""
    count = 0
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise V.VerificationError(f"{where}.{key}: missing")
            count += assert_numeric(value, actual[key], f"{where}.{key}", tolerance)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise V.VerificationError(f"{where}: list shape mismatch")
        for i, value in enumerate(expected):
            count += assert_numeric(value, actual[i], f"{where}[{i}]", tolerance)
    elif isinstance(expected, (int, float, np.number)):
        try:
            a, b = float(expected), float(actual)
        except (TypeError, ValueError) as exc:
            raise V.VerificationError(f"{where}: not numeric") from exc
        if not ((np.isnan(a) and np.isnan(b)) or a == b or (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tolerance)):
            raise V.VerificationError(f"{where}: numeric mismatch")
        count = 1
    elif actual != expected:
        raise V.VerificationError(f"{where}: identity mismatch")
    return count


def _error(err, accept):
    return float(np.asarray(err)[accept].mean()) if accept.any() else NAN


def _prediction_scores(pred):
    return pred["logits"] if "logits" in pred else pred["scores"]


def calibration_unit(val, test):
    """Both rules, no test truth used in calibration or selection."""
    cv = V.softmax_max(_prediction_scores(val)); ct = V.softmax_max(_prediction_scores(test))
    ev = val["y_true"] != val["y_pred"]; et = test["y_true"] != test["y_pred"]
    tau = float(np.quantile(cv, 0.2, method="linear"))
    av, at = cv >= tau, ct >= tau
    p = dict(tau=tau, budget=.2, promised_coverage=.8, val_coverage=float(av.mean()),
             val_accepted_error=_error(ev, av), val_error_rate=float(ev.mean()), test_coverage=float(at.mean()),
             test_accepted_error=_error(et, at), test_error_rate=float(et.mean()))
    p.update(gap_coverage=p["test_coverage"]-p["val_coverage"],
             test_review_deviation=1-p["test_coverage"]-.2,
             gap_accepted_error=p["test_accepted_error"]-p["val_accepted_error"])
    p.update(n_val=len(ev), n_val_accepted=int(av.sum()), n_val_accepted_errors=int(ev[av].sum()),
             n_test=len(et), n_test_accepted=int(at.sum()), n_test_accepted_errors=int(et[at].sum()))
    secondary = risk_threshold(cv, ev)
    at2 = ct >= secondary["tau"]
    av2 = cv >= secondary["tau"]
    secondary.update(n_val=len(ev), n_val_accepted=int(av2.sum()), n_val_accepted_errors=int(ev[av2].sum()),
                     n_test=len(et), n_test_accepted=int(at2.sum()), n_test_accepted_errors=int(et[at2].sum()))
    secondary.update(test_coverage=float(at2.mean()), test_accepted_error=_error(et, at2))
    secondary.update(gap_coverage=secondary["test_coverage"]-secondary["val_coverage"],
                     gap_accepted_error=secondary["test_accepted_error"]-secondary["val_accepted_error"])
    speakers = {}
    for speaker in sorted(set(test["speakers"])):
        m = np.asarray(test["speakers"]) == speaker
        row = dict(coverage=float(at[m].mean()), accepted_error=_error(et[m], at[m]), n=int(m.sum()),
                   n_acc=int(at[m].sum()), n_err_acc=int(et[m][at[m]].sum()))
        row.update(gap_coverage=row["coverage"]-p["val_coverage"],
                   gap_accepted_error=row["accepted_error"]-p["val_accepted_error"],
                   coverage_secondary=float(at2[m].mean()), accepted_error_secondary=_error(et[m], at2[m]),
                   gap_accepted_error_secondary=_error(et[m], at2[m])-secondary["val_accepted_error"])
        speakers[speaker] = row
    return {"primary": p, "secondary": secondary, "speakers": speakers}


def fixed_unit(preds):
    out = {}
    for role in ("seen", "new"):
        score = calibration_unit(preds[role], preds["test"])
        primary = score["primary"]
        out[role] = {k: primary[k] for k in ("tau", "val_coverage", "val_accepted_error", "test_coverage", "test_accepted_error",
                                           "test_review_deviation", "gap_coverage", "gap_accepted_error")}
        speakers = {}
        for speaker, sp in score["speakers"].items():
            row = {k: sp[k] for k in ("coverage", "accepted_error", "n", "n_acc", "n_err_acc", "gap_coverage", "gap_accepted_error")}
            row["review_deviation"] = 1 - row["coverage"] - .2
            speakers[speaker] = row
        out[role]["speakers"] = speakers
    return out


def verify_fixed_raw(plan, run_dir, ctx):
    """Raw identity checks; one stored test file and one model for both calibrators."""
    records = []
    seen_ids = set()
    for u in plan["units"]:
        uid = u["unit_id"]
        if uid in seen_ids:
            raise V.VerificationError("FIX duplicate planned unit")
        seen_ids.add(uid)
        identity = {k: u[k] for k in ("program", "module", "level", "model", "r", "fold", "split_sha256", "config_sha256", "feature_sha256")}
        if V.digest(identity) != uid or V.digest(u["config"]) != u["config_sha256"]:
            raise V.VerificationError("FIX unit/config digest mismatch")
        split = plan["splits"][u["split_key"]]
        if V.digest(split) != u["split_sha256"]:
            raise V.VerificationError("FIX partition digest mismatch")
        man = ctx.manifest(u["level"])
        orig = ctx.split(split["source_key"], split["source_sha256"])["folds"][u["fold"]]
        V.verify_fixed_partition(split, man, orig)
        replay_fixed_partition(split, man, orig)
        fixed_root = run_dir / "B2_FIXED" if (run_dir / "B2_FIXED").is_dir() else run_dir / "B2-FIX"
        path = fixed_root / "units" / uid
        files = [f"{r}_predictions.csv" for r in ("seen", "new", "test")] + ["model.npz"]
        if any(not (path / f).is_file() for f in files + ["unit.json", "DONE"]):
            raise V.VerificationError("FIX missing output")
        shas = [V.sha256_file(path / f) for f in files]
        if (path / "DONE").read_text().split() != shas:
            raise V.VerificationError("FIX output SHA mismatch")
        meta = V.read_json(path / "unit.json")
        if any(meta.get(k) != v for k, v in u.items()) or meta.get("status") != "done":
            raise V.VerificationError("FIX output identity mismatch")
        if [meta.get(f"{r}_sha256") for r in ("seen", "new", "test")] + [meta.get("model_sha256")] != shas:
            raise V.VerificationError("FIX metadata SHA mismatch")
        preds = {}
        for role in ("seen", "new", "test"):
            pred = V.read_logit_predictions(path / f"{role}_predictions.csv", False)
            report = V.Report()
            V.check_rows_against("B2-FIX", report, uid, role, pred, split[role], man, man.n_classes)
            if report.failed() or pred["paths"] != split[role] or not np.array_equal(pred["logits"].argmax(1), pred["y_pred"]):
                raise V.VerificationError(f"FIX {role} paths/labels/logits mismatch")
            preds[role] = pred
        records.append(dict(fixed_unit(preds), **{k: u[k] for k in ("unit_id", "level", "model", "r", "fold")}))
    return records


def fixed_endpoints(scored):
    groups = defaultdict(list)
    for unit in scored:
        groups[(unit["level"], unit["model"])].append(unit)
    endpoints = []
    for (level, model), units in sorted(groups.items()):
        speakers = sorted({s for u in units for s in u["seen"]["speakers"]})
        rs = sorted({u["r"] for u in units})
        for quantity in ("review_deviation", "gap_coverage", "accepted_error", "gap_accepted_error"):
            by_s = {s: {} for s in speakers}
            for u in units:
                for s in u["seen"]["speakers"]:
                    if u["r"] in by_s[s]:
                        raise V.VerificationError("FIX duplicate speaker/repeat")
                    by_s[s][u["r"]] = u["seen"]["speakers"][s][quantity] - u["new"]["speakers"][s][quantity]
            if any(set(rows) != set(rs) for rows in by_s.values()):
                raise V.VerificationError("FIX incomplete speaker/repeat")
            values = [np.mean([by_s[s][r] for r in rs]) for s in speakers]
            endpoints.append(dict(level=level, model=model, comparison="seen-new", quantity=quantity,
                                  n_repeats=len(rs), **interval(values)))
    return endpoints


def a_expected_vectors(plan, uar, recalls, classes):
    """Scientific A contrast definitions keyed by ep/condition/estimator/N/class.

    Endpoint display names are deliberately absent. Keys enumerate the design,
    so deleting an endpoint or changing its reference cannot evade verification.
    """
    conditions = {c["id"]: c for c in plan["conditions"]}
    usable = {c for c, row in conditions.items() if row["feasible"]}
    none = uar[("none", "none")]
    vectors = {}
    def add(ep, cond, est, n, cls, values, primary):
        vectors[(ep, cond, est, n, cls)] = (np.asarray(values), primary)
    ns = sorted({c["N"] for c in conditions.values() if c["feasible"] and c["composition"].startswith("single:")})
    for c in conditions.values():
        if c["feasible"] and c["composition"] == "balanced":
            add(1, c["id"], "naive", c["N"], None, uar[(c["id"], "naive")]-none, c["N"] == 3)
    for n in ns:
        cs = [c for c in classes if f"single:{c}@{n}" in usable]
        single = [f"single:{c}@{n}" for c in cs]
        mean = np.mean([uar[(c, "naive")] for c in single], axis=0)
        add(2, f"single:*@{n}", "naive", n, None, mean-none, n == 3)
        if f"balanced@{n}" in usable:
            add(3, f"single:*@{n}", "naive", n, None, mean-uar[(f"balanced@{n}", "naive")], n == 3)
            if n == 3:
                subset = [c for c in classes if c in conditions["balanced@3"]["need"] and f"single:{c}@3" in usable]
                if subset:
                    add(3, "single:balanced_classes@3", "naive", 3, None,
                        np.mean([uar[(f"single:{c}@3", "naive")] for c in subset], axis=0)-uar[("balanced@3", "naive")], False)
        rd = []
        for cls, cond in zip(cs, single):
            j = classes.index(cls)
            diff = recalls[(cond, "naive")][:, j]-recalls[("none", "none")][:, j]
            add(4, cond, "naive", n, cls, diff, False); rd.append(diff)
        add(4, f"single:*@{n}", "naive", n, None, np.mean(rd, axis=0), n == 3)
        for est in ("shrink", "neutral_filter", "prior_corrected"):
            dd = [uar[(c, est)]-uar[(c, "naive")] for c in single]
            add(5, f"single:*@{n}", est, n, None, np.mean(dd, axis=0), n == 3)
            for cls, cond, diff in zip(cs, single, dd):
                add(5, cond, est, n, cls, diff, False)
        for est in ("naive", "shrink", "neutral_filter", "prior_corrected"):
            harms = []
            for cls, cond in zip(cs, single):
                add(7, cond, est, n, cls, (uar[(cond, est)]-none < -2).astype(float), False)
                harms.append(uar[(cond, est)])
            add(7, f"single:*@{n}", est, n, None, (np.mean(harms, axis=0)-none < -2).astype(float), n == 3 and est == "naive")
    for n in (3, 5):
        if f"other_balanced@{n}" in usable and f"balanced@{n}" in usable:
            add(6, f"other_balanced@{n}", "naive", n, None,
                uar[(f"other_balanced@{n}", "naive")]-uar[(f"balanced@{n}", "naive")], n == 3)
        cs = [c for c in classes if f"other_single:{c}@{n}" in usable and f"single:{c}@{n}" in usable]
        if cs:
            dd = [uar[(f"other_single:{c}@{n}", "naive")]-uar[(f"single:{c}@{n}", "naive")] for c in cs]
            add(6, f"other_single:*@{n}", "naive", n, None, np.mean(dd, axis=0), n == 3)
            for cls, diff in zip(cs, dd):
                add(6, f"other_single:{cls}@{n}", "naive", n, cls, diff, False)
    return vectors


def verify_a_results(raw_plan, acc, seen, ctx, result, ps_rows):
    count = 0; expected_endpoint_keys = set(); expected_absolute_keys = set(); expected_ps = {}
    if result.get("program") != V.PROGRAM or result.get("module") != "A":
        raise V.VerificationError("A result study identity mismatch")
    if "endpoint_manifest" in raw_plan:
        declared = {r["id"]: r for r in result["endpoints"]}
        manifest = {r["id"]: r for r in raw_plan["endpoint_manifest"]}
        if len(declared) != len(result["endpoints"]) or set(declared) != set(manifest):
            raise V.VerificationError("A missing endpoint/duplicate/unexpected identifier")
        for key, row in manifest.items():
            actual = declared[key]
            for field, expected in row.items():
                actual_value = actual.get(field, True if field == "available" else False if field == "primary" else None)
                if actual_value != expected:
                    raise V.VerificationError(f"A endpoint metadata differs: {field}")
    ep_map = {}
    for e in result["endpoints"]:
        if e.get("available") is False:
            if e.get("estimate") is not None or e.get("n") != 0:
                raise V.VerificationError("A unavailable endpoint contains a score")
            continue
        k = (e["level"], e["model"], int(e["endpoint"]), e["condition"], e["estimator"], e.get("N"), e.get("target_class"))
        if k in ep_map:
            raise V.VerificationError("A duplicated endpoint")
        ep_map[k] = e
    abs_map = {(e["level"], e["model"], e["condition"], e["estimator"]): e for e in result["absolute"]}
    if len(abs_map) != len(result["absolute"]):
        raise V.VerificationError("A duplicated absolute")
    for (level, encoder), accumulated in sorted(acc.items()):
        man = ctx.manifest(level); report = V.Report(); speakers = sorted(seen[(level, encoder)])
        uar, rec = V.a_speaker_tables(accumulated, speakers, man.n_classes, report, level, encoder)
        if report.failed():
            raise V.VerificationError("A invalid confusion support")
        for (condition, estimator), values in uar.items():
            key = (level, encoder, condition, estimator); expected_absolute_keys.add(key)
            target = abs_map.get(key)
            if target is None:
                raise V.VerificationError("A missing absolute")
            iv = interval(values); count += assert_numeric({k: iv[k] for k in ("estimate", "ci95", "n")}, target)
            for speaker, value in zip(speakers, values):
                expected_ps[(level, encoder, condition, estimator, speaker)] = float(value)
        for key, (values, primary) in a_expected_vectors(raw_plan["levels"][level], uar, rec, man.classes).items():
            key = (level, encoder) + key; expected_endpoint_keys.add(key)
            if key not in ep_map:
                raise V.VerificationError(f"A missing endpoint {key}")
            iv = interval(values); target = ep_map[key]
            count += assert_numeric({k: iv[k] for k in ("estimate", "ci95", "n")}, target)
            if target.get("primary") != primary:
                raise V.VerificationError("A primary budget changed")
    if set(ep_map) != expected_endpoint_keys or set(abs_map) != expected_absolute_keys:
        raise V.VerificationError("A unexpected endpoint/absolute")
    actual_ps = {(r["level"], r["enc"], r["condition"], r["estimator"], r["speaker"]): float(r["uar"]) for r in ps_rows}
    if len(actual_ps) != len(ps_rows) or set(actual_ps) != set(expected_ps):
        raise V.VerificationError("A per-speaker identity set mismatch")
    for key, value in expected_ps.items():
        count += assert_numeric(value, actual_ps[key], "A.per_speaker")
    return count


B2_QUANTITIES = ("coverage", "accepted_error", "gap_coverage", "gap_accepted_error", "coverage_secondary",
                 "accepted_error_secondary", "gap_accepted_error_secondary")


def profile(values):
    values = np.asarray(values, float)
    return {"min": float(values.min()), "p10": float(np.percentile(values, 10)),
            "count_below_0.70": int((values < .7).sum()), "count_zero": int((values == 0).sum()), "n_speakers": len(values)}


def b2_aggregate(scored):
    groups = defaultdict(list)
    for row in scored:
        groups[(row["level"], row["model"])].append(row)
    endpoints, absolute, ps = [], [], []
    for (level, model), units in sorted(groups.items()):
        speakers = sorted({s for u in units for s in u["speakers"]}); rs = sorted({u["r"] for u in units})
        matrices = {c: {q: np.full((len(rs), len(speakers)), np.nan) for q in B2_QUANTITIES} for c in ("GG", "GR")}
        present = set()
        for u in units:
            for speaker, values in u["speakers"].items():
                key = (u["cell"], u["r"], speaker)
                if key in present:
                    raise V.VerificationError("B2 duplicate speaker/repeat")
                present.add(key)
                for q in B2_QUANTITIES:
                    matrices[u["cell"]][q][rs.index(u["r"]), speakers.index(speaker)] = values[q]
        if len(present) != 2*len(rs)*len(speakers):
            raise V.VerificationError("B2 incomplete paired speaker/repeat")
        for c in ("GG", "GR"):
            cell = [u for u in units if u["cell"] == c]
            reduced = {q: a.mean(0) for q, a in matrices[c].items()}
            ab = dict(level=level, model=model, cell=c, n_units=len(cell), n_r=len(rs), n_speakers=len(speakers))
            for rule in ("primary", "secondary"):
                for key in cell[0][rule]:
                    values = np.asarray([u[rule][key] for u in cell], float)
                    ab[f"{rule}.{key}.unit_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else NAN
            for q, values in reduced.items():
                ab[f"speaker.{q}"] = interval(values)
            ab["speaker.coverage_profile"] = profile(reduced["coverage"])
            ab["speaker.coverage_profile_secondary"] = profile(reduced["coverage_secondary"])
            ab["profiles_by_repeat"] = {rule: [dict(r=r, **profile(matrices[c][q][i])) for i, r in enumerate(rs)]
                                        for rule, q in (("primary", "coverage"), ("secondary", "coverage_secondary"))}
            ab["mean_repeat_count_below_0.70"] = float(np.mean([a["count_below_0.70"] for a in ab["profiles_by_repeat"]["primary"]]))
            pooled = []
            for repeat in rs:
                for rule in ("primary", "secondary"):
                    pp = {k: int(sum(u[rule][k] for u in cell if u["r"] == repeat)) for k in
                          ("n_val", "n_val_accepted", "n_val_accepted_errors", "n_test", "n_test_accepted", "n_test_accepted_errors")}
                    pp.update(r=repeat, rule=rule)
                    for role in ("val", "test"):
                        pp[f"{role}_coverage"] = _ratio(pp[f"n_{role}_accepted"], pp[f"n_{role}"])
                        pp[f"{role}_accepted_error"] = _ratio(pp[f"n_{role}_accepted_errors"], pp[f"n_{role}_accepted"])
                    pp["gap_coverage"] = pp["test_coverage"]-pp["val_coverage"]
                    pp["gap_accepted_error"] = pp["test_accepted_error"]-pp["val_accepted_error"]
                    pp["test_review_deviation"] = 1-pp["test_coverage"]-.2
                    pooled.append(pp)
            ab["pooled_by_repeat"] = pooled
            absolute.append(ab)
            for i, speaker in enumerate(speakers):
                ps.append(dict(level=level, model=model, cell=c, speaker=speaker, n_r=len(rs), **{q: float(a[i]) for q, a in reduced.items()}))
        for q in B2_QUANTITIES:
            values = (matrices["GR"][q]-matrices["GG"][q]).mean(0)
            endpoints.append(dict(level=level, model=model, comparison="GR-GG", quantity=q,
                                  rule="secondary" if q.endswith("_secondary") else "primary", **interval(values)))
    return dict(endpoints=endpoints, absolute=absolute, per_speaker_rows=ps)


def _ratio(n, d):
    return float(n / d) if d else NAN


def b1_ranking(conf, incorrect, paths):
    ordering = sorted(range(len(conf)), key=lambda i: (-float(conf[i]), hashlib.sha256(paths[i].encode()).hexdigest(), paths[i]))
    running, values = 0, []
    for k, index in enumerate(ordering, 1):
        running += int(incorrect[index]); values.append(running/k)
    n_correct = len(conf)-int(np.sum(incorrect))
    oracle = np.mean([max(0, k-n_correct)/k for k in range(1, len(conf)+1)])
    area = float(np.mean(values))
    return {"aurc": area, "oracle_aurc": float(oracle), "e_aurc": area-float(oracle)}


def b1_budget(group, budget, by_fold=False):
    conf, paths = group["conf"], group["paths"]
    truth, guesses, speakers = group["y_true"], group["y_pred"], np.asarray(group["speakers"])
    incorrect = truth != guesses; C = group["C"]
    accepted = np.zeros(len(paths), bool)
    if by_fold:
        for f in sorted(set(group["fold"])):
            mask = group["fold"] == f
            accepted[mask] = quota(conf[mask], np.asarray(paths)[mask].tolist(), budget)
    else:
        accepted = quota(conf, paths, budget)
    sp_names = sorted(set(speakers)); equalquota = np.zeros(len(paths), bool); table = {}
    for s in sp_names:
        m = speakers == s; a = accepted[m]; e = incorrect[m]
        table[s] = dict(n=int(m.sum()), n_acc=int(a.sum()), n_err=int(e.sum()), n_err_acc=int((a&e).sum()),
                        n_err_rej=int((~a&e).sum()), coverage=float(a.mean()), accepted_error=_error(e, a))
        equalquota[m] = quota(conf[m], np.asarray(paths)[m].tolist(), budget)
    cm = V.confusion(truth[accepted], guesses[accepted], C); support = cm.sum(1)
    risks = np.asarray([table[s]["accepted_error"] for s in sp_names]); cov = np.asarray([table[s]["coverage"] for s in sp_names])
    metrics = dict(b=budget, tau=float(np.quantile(conf, budget)), coverage=float(accepted.mean()),
                   accepted_error=_error(incorrect, accepted), error_capture_rate=_ratio(int((incorrect&~accepted).sum()), incorrect.sum()),
                   accepted_uar=V.uar_from_confusion(cm) if (support > 0).all() else NAN,
                   accepted_present_class_uar=V.uar_from_confusion(cm), accepted_classes_present=int((support > 0).sum()),
                   n_rejected=int((~accepted).sum()), actual_review_rate=float((~accepted).mean()),
                   overall_error_rate=float(incorrect.mean()), overall_uar=V.uar_from_confusion(V.confusion(truth, guesses, C)),
                   n_utts=len(paths), speaker_equal_accepted_error=float(np.nanmean(risks)) if np.isfinite(risks).any() else NAN,
                   speaker_risk_undefined_count=int((~np.isfinite(risks)).sum()), speaker_coverage_min=float(cov.min()),
                   speaker_coverage_p10=float(np.percentile(cov, 10)), speakers_below_0_70=int((cov < .7).sum()),
                   speakers_zero=int((cov == 0).sum()), quota_accepted_error=_error(incorrect, equalquota),
                   quota_coverage=float(equalquota.mean()), quota_error_capture_rate=_ratio(int((incorrect&~equalquota).sum()), incorrect.sum()))
    metrics["speakers_below_0.70"] = metrics.pop("speakers_below_0_70")
    metrics.update({f"accepted_support_{c}": int(support[c]) for c in range(C)})
    mix = [dict(class_index=c, rejected_count=int((~accepted & (truth == c)).sum()),
                rejected_share=_ratio(int((~accepted & (truth == c)).sum()), (~accepted).sum()),
                class_share_overall=float((truth == c).mean()),
                class_rejection_rate=_ratio(int((~accepted & (truth == c)).sum()), (truth == c).sum())) for c in range(C)]
    return dict(metrics=metrics, speakers=table, rejected_mix=mix)


def pooled_interval(numerators, denominators):
    numerators, denominators = np.asarray(numerators), np.asarray(denominators)
    idx = V.bootstrap_indices(numerators.shape[1])
    estimates = [_ratio(a.sum(), b.sum()) for a, b in zip(numerators, denominators)]
    with np.errstate(invalid="ignore", divide="ignore"):
        sampled = np.mean([a[idx].sum(1)/b[idx].sum(1) for a, b in zip(numerators, denominators)], axis=0)
    finite = np.isfinite(sampled)
    return dict(estimate=float(np.mean(estimates)), ci95=np.percentile(sampled[finite], [2.5, 97.5]).tolist() if finite.any() else [NAN, NAN],
                n=numerators.shape[1], n_undefined_bootstrap=int((~finite).sum()))


def b1_aggregate(groups, ctx, by_fold=False):
    by_lm = defaultdict(list)
    for key, group in sorted(groups.items()):
        by_lm[key[:2]].append(group)
    summary, per_speaker, mix, endpoints = [], [], [], []
    for (level, model), reps in sorted(by_lm.items()):
        classes = ctx.manifest(level).classes; n_classes = len(classes)
        speakers = sorted(set(reps[0]["speakers"])); n = len(speakers)
        ranking = []
        for g in reps:
            if by_fold:
                result = dict.fromkeys(("aurc", "oracle_aurc", "e_aurc"), 0.)
                for fold in sorted(set(g["fold"])):
                    m = g["fold"] == fold
                    ar = b1_ranking(g["conf"][m], (g["y_true"] != g["y_pred"])[m], np.asarray(g["paths"])[m].tolist())
                    for k in result:
                        result[k] += float(m.mean())*ar[k]
                ranking.append(result)
            else:
                ranking.append(b1_ranking(g["conf"], g["y_true"] != g["y_pred"], g["paths"]))
        rank_means = {k: float(np.mean([r[k] for r in ranking])) for k in ranking[0]}
        for b in (.1, .2, .3):
            raw = [b1_budget(g, b, by_fold) for g in reps]
            row = dict(level=level, model=model, b=b, n_reps=len(reps), n_speakers=n)
            for k in raw[0]["metrics"]:
                if k == "b": continue
                vals = [r["metrics"][k] for r in raw]
                row[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else NAN
            row.update(rank_means); summary.append(row)
            for speaker in speakers:
                vals = [r["speakers"][speaker] for r in raw]
                risks = [v["accepted_error"] for v in vals]
                per_speaker.append(dict(level=level, model=model, b=b, speaker=speaker, coverage=float(np.mean([v["coverage"] for v in vals])),
                                        accepted_error=float(np.nanmean(risks)) if np.isfinite(risks).any() else NAN, n_utts=vals[0]["n"]))
            for c, label in enumerate(classes):
                ms = [r["rejected_mix"][c] for r in raw]
                mix.append(dict(level=level, model=model, b=b, class_index=c, class_label=label,
                                **{k: float(np.nanmean([m[k] for m in ms])) if np.isfinite([m[k] for m in ms]).any() else NAN
                                   for k in ("rejected_count", "rejected_share", "class_share_overall", "class_rejection_rate")}))
            if b == .2:
                base = dict(level=level, model=model, comparison="absolute", b=b, n_reps=len(reps))
                for quantity, num, den in (("accepted_error", "n_err_acc", "n_acc"), ("error_capture_rate", "n_err_rej", "n_err")):
                    nums = [[r["speakers"][s][num] for s in speakers] for r in raw]
                    dens = [[r["speakers"][s][den] for s in speakers] for r in raw]
                    endpoints.append(dict(base, quantity=quantity, **pooled_interval(nums, dens)))
                flags = np.asarray([[r["speakers"][s]["coverage"] < .7 for s in speakers] for r in raw]).mean(0)
                iv = interval(flags)
                bare = {k: iv[k] for k in ("estimate", "ci95", "n")}
                endpoints.append(dict(base, quantity="share_speakers_coverage_below_0.70", **bare))
                endpoints.append(dict(base, quantity="count_speakers_coverage_below_0.70", estimate=iv["estimate"]*n,
                                      ci95=[v*n for v in iv["ci95"]], n=n))
    return dict(summary=summary, per_speaker=per_speaker, mix=mix, endpoints=endpoints)


def compare_records(expected, actual, fields, label):
    def identity(row):
        return tuple(str(row[k]) for k in fields)
    em = {identity(r): r for r in expected}; am = {identity(r): r for r in actual}
    if len(em) != len(expected) or len(am) != len(actual) or set(em) != set(am):
        raise V.VerificationError(f"{label}: missing/duplicate/unexpected rows")
    return sum(assert_numeric(r, am[k], label) for k, r in em.items())


def metadata_audit(plan_dir, ctx):
    """No feature arrays and no predictions: validate scientific assignment contracts."""
    report = {"program": V.PROGRAM, "modules": {}, "outcomes_read": False}
    report["plan_file_sha256"] = {n: V.sha256_file(plan_dir / n) for n in ("A.json", "B2.json", "B2_FIXED.json")}
    report["code_sha256"] = {p.name: V.sha256_file(p) for p in (Path(__file__), Path(V.__file__))}
    ap = V.read_json(plan_dir / "A.json")
    if ap.get("program") != V.PROGRAM:
        raise V.VerificationError("A wrong study identity")
    norm = V.normalise_plan_a(ap); a_counts = {}
    for level, block in norm["levels"].items():
        man = ctx.manifest(level)
        for speaker, queries in block["query"].items():
            enroll = block["enrollment"][speaker]
            if {man.sentence(p) for p in queries} & {man.sentence(p) for p in enroll}:
                raise V.VerificationError("A query/reference sentence overlap")
            if set(man.label(p) for p in queries) != set(range(man.n_classes)):
                raise V.VerificationError("A missing fixed query class")
        a_counts[level] = dict(speakers=len(block["query"]), feasible=sum(c["feasible"] for c in block["conditions"].values()))
        if not block["conditions"]["balanced@3"]["feasible"]:
            raise V.VerificationError("A lacks primary same-N control")
        for u in (u for u in norm["units"] if u["level"] == level):
            split = ctx.split(u["split_key"], u["split_sha256"])
            expected_q = V.derived_query_sets(level, man, split["population"], V.PROGRAM)
            if expected_q and not any(all(set(block["query"][s]) == q[s] for s in block["query"]) for q in expected_q):
                raise V.VerificationError("A query sentence hash derivation mismatch")
    report["modules"]["A"] = dict(units=len(ap["units"]), levels=a_counts)
    bp = V.read_json(plan_dir / "B2.json"); seen_ids = set(); pairs = defaultdict(dict)
    for u in bp["units"]:
        if u["unit_id"] in seen_ids: raise V.VerificationError("B2 duplicate unit")
        seen_ids.add(u["unit_id"])
        identity = {"program": V.PROGRAM, "module": "B2", **{k: u[k] for k in ("level", "base", "model", "cell", "r", "fold", "seed_index", "train_seed", "split_key", "split_sha256", "config_sha256")}}
        if V.digest(identity) != u["unit_id"] or V.digest(u["config"]) != u["config_sha256"]:
            raise V.VerificationError("B2 unit/config identity mismatch")
        man = ctx.manifest(u["level"]); fold = ctx.split(u["split_key"], u["split_sha256"])["folds"][u["fold"]]
        sp = {r: {man.speaker(p) for p in fold[r]} for r in ("fit", "val", "test")}
        if sp["test"] & (sp["fit"] | sp["val"]): raise V.VerificationError("B2 test speaker leakage")
        if u["cell"] == "GG" and sp["fit"] & sp["val"]: raise V.VerificationError("B2 GG val speaker leakage")
        if u["cell"] == "GR" and not sp["val"] <= sp["fit"]: raise V.VerificationError("B2 GR missing intended overlap")
        pairs[(u["level"], u["model"], u["r"], u["fold"])][u["cell"]] = fold
    for pair in pairs.values():
        if set(pair) != {"GG", "GR"} or pair["GG"]["test"] != pair["GR"]["test"]:
            raise V.VerificationError("B2 unpaired outer fold")
    report["modules"]["B2"] = dict(units=len(bp["units"]), paired_folds=len(pairs),
                                      fit_differs=sum(set(p["GG"]["fit"]) != set(p["GR"]["fit"]) for p in pairs.values()))
    fp = V.read_json(plan_dir / "B2_FIXED.json"); fixed = {}
    for key, split in fp["splits"].items():
        level = next(u["level"] for u in fp["units"] if u["split_key"] == key)
        orig = ctx.split(split["source_key"], split["source_sha256"])["folds"][split["fold"]]
        fixed[key] = V.verify_fixed_partition(split, ctx.manifest(level), orig)
        replay_fixed_partition(split, ctx.manifest(level), orig)
    report["modules"]["B2-FIX"] = dict(units=len(fp["units"]), matched_splits=len(fixed), partitions=fixed)
    report["pass"] = True
    return report


def replay_fixed_partition(split, man, original):
    """Re-derive donor selection, half-recording eligibility and matched cells from metadata."""
    salt = split["salt"]
    novel = V._ranked({man.speaker(p) for p in original["val"]}, salt+"|new-speaker")
    donors = V._ranked({man.speaker(p) for p in original["fit"]}, salt+"|seen-speaker")[:len(novel)]
    eligible = []
    for donor in donors:
        pp = [p for p in original["fit"] if man.speaker(p) == donor]
        eligible.extend(V._ranked(pp, f"{salt}|{donor}|seen-holdout")[:len(pp)//2])
    if sorted(eligible) != split["seen_eligible_paths"] or split["donor_pairs"] != [{"seen": a, "new": b} for a, b in zip(donors, novel)]:
        raise V.VerificationError("FIX donor/half-recording assignment changed")
    expected_seen, expected_new, expected_cells = [], [], []
    for donor, newcomer in zip(donors, novel):
        a = defaultdict(list); b = defaultdict(list)
        for p in original["fit"]:
            if man.speaker(p) == donor: a[(man.sentence(p), man.label(p))].append(p)
        for p in original["val"]:
            if man.speaker(p) == newcomer: b[(man.sentence(p), man.label(p))].append(p)
        for sentence, label in sorted(set(a) | set(b)):
            aa, bb = a[(sentence, label)], b[(sentence, label)]
            eligible_cell = [p for p in aa if p in set(eligible)]
            n = min(len(eligible_cell), len(bb)); prefix = f"{salt}|{donor}|{newcomer}|{sentence}|{label}"
            expected_seen.extend(V._ranked(eligible_cell, prefix+"|seen-recording")[:n])
            expected_new.extend(V._ranked(bb, prefix+"|new-recording")[:n])
            expected_cells.append(dict(seen_speaker=donor, new_speaker=newcomer, sentence=sentence, class_index=label,
                                       n_seen_available=len(aa), n_new_available=len(bb), n_seen_eligible=len(eligible_cell),
                                       n_each_selected=n, status="matched" if n else "dropped_whole_cell"))
    if sorted(expected_seen) != split["seen"] or sorted(expected_new) != split["new"]:
        raise V.VerificationError("FIX matched recording replay differs")
    if sorted(set(original["fit"])-set(expected_seen)) != split["fit"] or expected_cells != split["cell_feasibility"]:
        raise V.VerificationError("FIX retained training set/feasibility receipt differs")


def run(modules, plan_dir, run_dir, results_dir, out, sources, tolerance=1e-9):
    ctx = V.Context(sources); report = {"program": V.PROGRAM, "kind": "independent-verification", "modules": {}, "reasons": [],
                                       "verify_sha256": V.sha256_file(Path(__file__)), "tolerance": tolerance}
    for module in modules:
        rep = V.Report(); count = 0
        try:
            if module == "A":
                plan = V.read_json(plan_dir / "A.json")
                acc, seen = V.verify_a_integrity(V.normalise_plan_a(plan), run_dir, ctx, rep)
                if rep.failed(): raise V.VerificationError(str(rep.failures))
                count = verify_a_results(plan, acc, seen, ctx, V.read_json(results_dir / "A" / "endpoints.json"),
                                         V.read_csv_rows(results_dir / "A" / "per_speaker.csv"))
            elif module == "B2":
                plan = V.read_json(plan_dir / "B2.json")
                data = V.verify_b2_integrity(V.normalise_plan_b2(plan), run_dir, ctx, rep)
                if rep.failed(): raise V.VerificationError(str(rep.failures))
                by_uid = {u["unit_id"]: u for u in plan["units"]}; scored = []
                for u in by_uid.values():
                    path = V.b2_unit_path(run_dir, u["unit_id"], u["model"])
                    meta = V.read_json(path / "unit.json")
                    if any(meta.get(k) != value for k, value in u.items()):
                        raise V.VerificationError("B2 output identity/config differs from plan")
                    if u["model"] in ("cnn", "wavlm_ft"):
                        cp = path / "checkpoint.pt"
                        if not cp.is_file() or V.sha256_file(cp) != meta.get("engine_info", {}).get("checkpoint_sha256"):
                            raise V.VerificationError("B2 restored model checkpoint absent/corrupt")
                    d = data[(u["level"], u["model"], u["r"], u["fold"])][u["cell"]]
                    for p in d["val"], d["test"]:
                        if not np.array_equal(p["logits"].argmax(1), p["y_pred"]):
                            raise V.VerificationError("B2 prediction/argmax mismatch")
                    scored.append(dict(calibration_unit(d["val"], d["test"]), **{k: u[k] for k in ("unit_id", "level", "model", "cell", "r", "fold")}))
                expected = b2_aggregate(scored); actual = V.read_json(results_dir / "B2" / "endpoints.json")
                assert_numeric({"program": V.PROGRAM, "module": module}, actual, "result header")
                count += compare_records(expected["endpoints"], actual["endpoints"], ("level", "model", "quantity"), "B2 endpoints")
                count += compare_records(expected["absolute"], actual["absolute"], ("level", "model", "cell"), "B2 absolute")
                count += compare_records(expected["per_speaker_rows"], V.read_csv_rows(results_dir / "B2" / "per_speaker.csv"),
                                         ("level", "model", "cell", "speaker"), "B2 per-speaker")
            elif module == "B2-FIX":
                plan = V.read_json(plan_dir / "B2_FIXED.json"); scored = verify_fixed_raw(plan, run_dir, ctx)
                path = results_dir / "B2-FIX" / "endpoints.json"
                if not path.is_file() and (results_dir / "B2_FIXED" / "endpoints.json").is_file():
                    path = results_dir / "B2_FIXED" / "endpoints.json"
                actual = V.read_json(path)
                assert_numeric({"program": V.PROGRAM, "module": module}, actual, "result header")
                count += compare_records(scored, actual["units"], ("unit_id",), "FIX units")
                count += compare_records(fixed_endpoints(scored), actual["endpoints"], ("level", "model", "quantity"), "FIX endpoints")
            elif module == "B1":
                groups = V.verify_b1_integrity(ctx, rep)
                if rep.failed(): raise V.VerificationError(str(rep.failures))
                expected = b1_aggregate(groups, ctx); actual = V.read_json(results_dir / "B1" / "endpoints.json")
                assert_numeric({"program": V.PROGRAM, "module": module}, actual, "result header")
                count += compare_records(expected["endpoints"], actual["endpoints"], ("level", "model", "quantity"), "B1 endpoints")
                for key, filename, ids in (("summary", "summary.csv", ("level", "model", "b")),
                                           ("per_speaker", "per_speaker.csv", ("level", "model", "b", "speaker")),
                                           ("mix", "rejected_class_mix.csv", ("level", "model", "b", "class_index"))):
                    count += compare_records(expected[key], V.read_csv_rows(results_dir / "B1" / filename), ids, f"B1 {key}")
                fold = b1_aggregate(groups, ctx, True)
                for key, ids in (("endpoints", ("level", "model", "quantity")), ("summary", ("level", "model", "b")),
                                  ("per_speaker", ("level", "model", "b", "speaker")), ("mix", ("level", "model", "b", "class_index"))):
                    pooled = [r for group in actual["fold_sensitivity"] for r in group[key]]
                    count += compare_records(fold[key], pooled, ids, f"B1 fold {key}")
            else:
                raise V.VerificationError(f"unknown module {module}")
            report["modules"][module] = {"integrity": {"pass": True}, "comparison": {"pass": True, "compared": count, "failed": 0}}
        except (OSError, KeyError, ValueError, TypeError) as exc:
            report["modules"][module] = {"integrity": {"pass": False}, "comparison": {"pass": False, "compared": count, "failed": 1}}
            report["reasons"].append(f"{module}: {type(exc).__name__}: {exc}")
    report["pass"] = not report["reasons"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return report

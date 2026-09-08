"""B2-FIX: one fixed Ridge model, familiar versus unseen calibration speakers.

Only metadata constructs the split. Primary thresholds use no calibration or test
labels; labels in prediction files are for subsequent error evaluation. A pair of
calibrators shares the exact same model artifact and single test prediction file.
All 135 units must close before ``score`` is allowed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

from . import calib, common
from .common import (DEPLOY_ROOT, ENCODERS, LEVELS, PROGRAM, atomic_write_json,
                     atomic_write_text, class_table, digest, labels_for, load_manifest,
                     load_split, now, ranked, read_json, read_predictions_csv, require,
                     sha256_file, speakers_for, write_predictions_csv)

MODULE = "B2-FIX"
SPEC_VERSION = "DEPLOY-2"
ROLES = ("fit", "seen", "new", "test")
PRED_ROLES = ("seen", "new", "test")
SPLIT_RULE = {
    "donors": "hash-rank GG-fit speakers; first n(GG-val speakers), paired by independently ranked GG-val speakers",
    "strata": "paired donor x sentence x label_index",
    "seen_eligibility": "first floor(n_donor/2) hash-ranked GG-fit donor recordings; the remainder is reserved for F",
    "count": "k=min(n_eligible_seen_cell,n_new_cell); select first k independently hash-ranked paths per side",
    "zero_cell": "drop the entire cell on both sides; never substitute another cell",
    "unused_new": "remain outside fit and test",
    "fail": "empty calibrator speaker, absent calibration class, lost donor in fit, non-disjoint roles, unequal cell counts",
}
RIDGE_CONFIG = dict(calib.RIDGE_CONFIG, solver="lsqr", tol=1e-4,
                    transform="float32 subtract mean cast to float32, then divide scale cast to float32")


def role_summary(paths, manifest):
    return {"n": len(paths), "speakers": sorted(set(speakers_for(manifest, paths))),
            "classes": dict(sorted(Counter(str(manifest[p]["label_index"]) for p in paths).items())),
            "sentences": dict(sorted(Counter(manifest[p]["sentence"] for p in paths).items()))}


def build_split(manifest: dict, fold: dict, salt: str) -> dict:
    """Pure metadata construction; per-pair matching guarantees equal speaker weights/sizes."""
    for role in ("fit", "val", "test"):
        require(len(set(fold[role])) == len(fold[role]), f"duplicate source {role} path")
    source_speakers = {r: set(speakers_for(manifest, fold[r])) for r in ("fit", "val", "test")}
    require(not(source_speakers["fit"] & (source_speakers["val"] | source_speakers["test"]))
            and not(source_speakers["val"] & source_speakers["test"]), "source GG speaker overlap")
    new_speakers = ranked(source_speakers["val"], salt + "|new-speaker")
    require(new_speakers and len(source_speakers["fit"]) >= len(new_speakers), "insufficient donor speakers")
    donors = ranked(source_speakers["fit"], salt + "|seen-speaker")[:len(new_speakers)]
    candidates = {r: defaultdict(list) for r in ("fit", "val")}
    eligible_seen = []
    for donor in donors:
        recordings = [p for p in fold["fit"] if manifest[p]["speaker"] == donor]
        eligible_seen += ranked(recordings, f"{salt}|{donor}|seen-holdout")[:len(recordings)//2]
    eligible_set = set(eligible_seen)
    for r in candidates:
        for p in fold[r]:
            m = manifest[p]
            candidates[r][(m["speaker"], m["sentence"], m["label_index"])].append(p)
    seen_paths, new_paths, cells = [], [], []
    for seen_s, new_s in zip(donors, new_speakers):
        keys = sorted({(t, c) for s, t, c in candidates["fit"] if s == seen_s}
                      | {(t, c) for s, t, c in candidates["val"] if s == new_s})
        for sentence, cls in keys:
            a = candidates["fit"][(seen_s, sentence, cls)]
            b = candidates["val"][(new_s, sentence, cls)]
            eligible = [p for p in a if p in eligible_set]
            k = min(len(eligible), len(b))
            cell_salt = f"{salt}|{seen_s}|{new_s}|{sentence}|{cls}"
            seen_paths += ranked(eligible, cell_salt + "|seen-recording")[:k]
            new_paths += ranked(b, cell_salt + "|new-recording")[:k]
            cells.append({"seen_speaker": seen_s, "new_speaker": new_s, "sentence": sentence,
                          "class_index": cls, "n_seen_available": len(a), "n_new_available": len(b),
                          "n_seen_eligible": len(eligible),
                          "n_each_selected": k, "status": "matched" if k else "dropped_whole_cell"})
    split = {"fit": sorted(set(fold["fit"]) - set(seen_paths)), "seen": sorted(seen_paths),
             "new": sorted(new_paths), "test": list(fold["test"]),
             "donor_pairs": [{"seen": a, "new": b} for a, b in zip(donors, new_speakers)],
             "seen_eligible_paths": sorted(eligible_seen),
             "cell_feasibility": cells, "salt": salt}
    validate_split(split, manifest)
    split["summaries"] = {r: role_summary(split[r], manifest) for r in ROLES}
    return split


def validate_split(split: dict, manifest: dict) -> None:
    sets = {r: set(split[r]) for r in ROLES}
    spks = {r: set(speakers_for(manifest, split[r])) for r in ROLES}
    for i, r in enumerate(ROLES):
        require(sets[r] and len(sets[r]) == len(split[r]), f"empty/duplicate {r}")
        require(all(not(sets[r] & sets[s]) for s in ROLES[i + 1:]), "four-way recording overlap")
    require(not(spks["fit"] & (spks["new"] | spks["test"])) and not(spks["new"] & spks["test"]), "unseen speaker overlap")
    require(spks["seen"] <= spks["fit"], "a seen donor has no fit recordings")
    pairs = split["donor_pairs"]
    require({p["seen"] for p in pairs} == spks["seen"] and {p["new"] for p in pairs} == spks["new"]
            and len(pairs) == len(spks["seen"]) == len(spks["new"]), "calibrator speaker missing or duplicated")
    all_classes = set(range(len(class_table(manifest))))
    for r in ("fit", "seen", "new"):
        require(set(labels_for(manifest, split[r])) == all_classes, f"{r} lacks a class")
    for pair in pairs:
        counts = []
        for r in ("seen", "new"):
            counts.append(Counter((manifest[p]["sentence"], manifest[p]["label_index"])
                                  for p in split[r] if manifest[p]["speaker"] == pair[r]))
        require(counts[0] and counts[0] == counts[1], "paired calibrator cell counts differ")


def make_unit_id(row):
    return digest({k: row[k] for k in ("program", "module", "level", "model", "r", "fold",
                                      "split_sha256", "config_sha256", "feature_sha256")})


def build_plan(inputs: dict) -> dict:
    units, splits, manifests = [], {}, {}
    for level in calib.B2_LEVELS:
        L = inputs["levels"][level]
        manifest = load_manifest(L["base"])
        manifests[L["base"]] = L["manifest_sha256"]
        for r in calib.B2_RS:
            source_key = f"ctrl__{level}__GG__r{r}"
            source = L["splits"][source_key]
            for f in calib.B2_FOLDS:
                key = f"{level}__r{r}__f{f}"
                split = build_split(manifest, source["table"]["folds"][f], f"{PROGRAM}|{MODULE}|{key}")
                split.update(source_key=source_key, source_sha256=source["sha256"], fold=f)
                splits[key] = split
                for encoder in ENCODERS:
                    feat = L["features"][encoder]
                    row = {"program": PROGRAM, "module": MODULE, "spec": SPEC_VERSION,
                           "level": level, "base": L["base"], "feature_corpus": L["feature_corpus"],
                           "model": f"ridge_{encoder}", "engine": "ridge", "cell": "FIX", "r": r, "fold": f,
                           "train_seed": 0, "split_key": key, "split_sha256": digest(split),
                           "config": RIDGE_CONFIG, "config_sha256": digest(RIDGE_CONFIG),
                           "feature_kind": encoder, "feature_file": feat["file"], "feature_sha256": feat["sha256"],
                           "n_classes": len(L["classes"]), **{f"n_{role}": len(split[role]) for role in ROLES}}
                    row["unit_id"] = make_unit_id(row)
                    units.append(row)
    units.sort(key=lambda u: (u["level"], u["model"], u["r"], u["fold"]))
    require(len({u["unit_id"] for u in units}) == len(units), "duplicate unit id")
    return {"program": PROGRAM, "module": MODULE, "spec": SPEC_VERSION, "generated_at": now(),
            "split_rule": SPLIT_RULE, "threshold_rule": {"budget": 0.20, "quantile_method": "linear", "accept": ">=tau"},
            "splits": splits, "manifests": manifests, "units": units, "n_units": len(units),
            "code_sha256": {n: sha256_file(DEPLOY_ROOT / n) for n in ("calib_fixed.py", "calib.py", "common.py")}}


def validate_plan(plan, verify_sources=True):
    require(plan["program"] == PROGRAM and plan["module"] == MODULE, "wrong program/module")
    require(plan["n_units"] == len(plan["units"]) and len({u["unit_id"] for u in plan["units"]}) == len(plan["units"]), "plan unit count")
    manifests = {base: load_manifest(base) for base in plan["manifests"]}
    if verify_sources:
        for base, sha in plan["manifests"].items():
            require(sha256_file(common.manifest_path(base)) == sha, "manifest changed")
        for name, sha in plan["code_sha256"].items():
            require(sha256_file(DEPLOY_ROOT / name) == sha, f"plan code changed: {name}")
    for u in plan["units"]:
        require(make_unit_id(u) == u["unit_id"] and digest(u["config"]) == u["config_sha256"], "unit identity/config mismatch")
        split = plan["splits"][u["split_key"]]
        require(digest(split) == u["split_sha256"], "split digest mismatch")
        require(all(len(split[r]) == u[f"n_{r}"] for r in ROLES), "split sizes mismatch")
        validate_split(split, manifests[u["base"]])
    if verify_sources:
        checked = set()
        for u in plan["units"]:
            if u["split_key"] in checked:
                continue
            s = plan["splits"][u["split_key"]]
            source = load_split(s["source_key"], verify=True)
            require(common.split_index()[s["source_key"]]["sha256"] == s["source_sha256"], "source split changed")
            fresh = build_split(manifests[u["base"]], source["folds"][u["fold"]], s["salt"])
            fresh.update(source_key=s["source_key"], source_sha256=s["source_sha256"], fold=u["fold"])
            require(fresh == s, "split differs from metadata rule")
            checked.add(u["split_key"])
    return manifests


def fit_model(X, y, n_classes, config):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    z = np.asarray(X, dtype=np.float32).copy()
    z -= scaler.mean_.astype(np.float32)
    z /= scaler.scale_.astype(np.float32)
    model = RidgeClassifier(alpha=config["alpha"], class_weight=config["class_weight"],
                            solver=config["solver"], tol=config["tol"]).fit(z, y)
    require(model.classes_.tolist() == list(range(n_classes)), "fit missing classes")
    return {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "coef": model.coef_,
            "intercept": model.intercept_, "classes": model.classes_}


def predict_model(model, X):
    # Explicit casts freeze rounding behavior across sklearn transformer versions.
    z = np.array(X, dtype=np.float32, copy=True)
    z -= model["scaler_mean"].astype(np.float32)
    z /= model["scaler_scale"].astype(np.float32)
    out = z @ model["coef"].T + model["intercept"]
    if out.shape[1] == 1 and len(model["classes"]) == 2:
        out = np.column_stack([-out[:, 0], out[:, 0]])
    return np.asarray(out, dtype=np.float64)


def done_status(udir, row, split=None, manifest=None):
    if not (udir / "DONE").exists():
        return "missing DONE"
    files = [f"{r}_predictions.csv" for r in PRED_ROLES] + ["model.npz"]
    if not all((udir / n).is_file() for n in files + ["unit.json"]):
        return "missing files"
    shas = [sha256_file(udir / n) for n in files]
    if (udir / "DONE").read_text(encoding="utf-8").split() != shas:
        return "DONE sha mismatch"
    meta = read_json(udir / "unit.json")
    if meta.get("status") != "done" or any(meta.get(k) != v for k, v in row.items()):
        return "unit identity mismatch"
    if [meta.get(f"{r}_sha256") for r in PRED_ROLES] + [meta.get("model_sha256")] != shas:
        return "unit sha mismatch"
    for role in PRED_ROLES:
        pred = read_predictions_csv(udir / f"{role}_predictions.csv", "logit")
        if pred["scores"].shape != (row[f"n_{role}"], row["n_classes"]) or not np.isfinite(pred["scores"]).all():
            return "prediction shape/nonfinite"
        if not np.array_equal(pred["scores"].argmax(axis=1), pred["y_pred"]):
            return "prediction argmax mismatch"
        if split is not None and pred["paths"] != split[role]:
            return "prediction paths mismatch"
        if manifest is not None and (pred["speakers"] != speakers_for(manifest, pred["paths"])
                                    or not np.array_equal(pred["y_true"], labels_for(manifest, pred["paths"]))):
            return "prediction labels/speakers mismatch"
    return "ok"


def run_plan(plan, out, limit=None):
    guard = common.cpu_guard(2)
    manifests = validate_plan(plan)
    out.mkdir(parents=True, exist_ok=True)
    env = calib.environment_info(guard["threads"])
    envfile = out / "environment.json"
    if envfile.exists():
        require(read_json(envfile) == env, "run environment drift")
    else:
        atomic_write_json(envfile, env)
    lock = out / "runner.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    completed, cache, cache_key = 0, None, None
    try:
        for u in plan["units"]:
            s, manifest = plan["splits"][u["split_key"]], manifests[u["base"]]
            udir = out / "units" / u["unit_id"]
            status = done_status(udir, u, s, manifest)
            if status == "ok":
                continue
            require(status == "missing DONE", f"refusing corrupt completed unit: {status}")
            if limit is not None and completed >= limit:
                break
            attempts = read_json(udir / "attempts.json") if (udir / "attempts.json").exists() else []
            require(len(attempts) < 2, f"unit exhausted two attempts: {u['unit_id']}")
            attempts.append({"attempt": len(attempts) + 1, "started_at": now(), "status": "started"})
            atomic_write_json(udir / "attempts.json", attempts)
            try:
                key = (u["feature_corpus"], u["feature_kind"])
                if cache_key != key:
                    cache = common.FeatureCache(*key)
                    require(cache.sha256 == u["feature_sha256"], "cache sha mismatch")
                    cache_key = key
                meta = run_one_unit(plan, u, out, manifest, cache, env, attempts[-1]["started_at"])
                attempts[-1].update(status="done", finished_at=now(), seconds=meta["timing"]["seconds"])
                completed += 1
                print(f"[B2-FIX] completed {completed} new units; last={meta['timing']['seconds']:.2f}s", flush=True)
            except Exception as exc:
                attempts[-1].update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                atomic_write_json(udir / "attempts.json", attempts)
                with open(out / "ledger.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(dict(attempts[-1], unit_id=u["unit_id"]), sort_keys=True) + "\n")
    finally:
        lock.unlink()
    return completed


def run_one_unit(plan, u, out, manifest, cache, env, started_at):
    """Supervisor API: compute/write one unit; supervisor owns locks/retries/seals."""
    s = plan["splits"][u["split_key"]]
    require(digest(s) == u["split_sha256"] and cache.sha256 == u["feature_sha256"], "unit input hash mismatch")
    validate_split(s, manifest)
    udir = Path(out) / "units" / u["unit_id"]
    udir.mkdir(parents=True, exist_ok=True)
    require(not (udir / "DONE").exists(), "refusing overwrite of completed unit")
    with common.Timer() as timer:
        X = cache.get(s["fit"], state=u["config"]["state"])
        model = fit_model(X, labels_for(manifest, s["fit"]), u["n_classes"], u["config"])
        with open(udir / "model.npz.tmp", "wb") as fh:
            np.savez_compressed(fh, **model)
        os.replace(udir / "model.npz.tmp", udir / "model.npz")
        shas = {}
        for role in PRED_ROLES:
            scores = predict_model(model, cache.get(s[role], state=u["config"]["state"]))
            shas[role] = write_predictions_csv(udir / f"{role}_predictions.csv", s[role],
                speakers_for(manifest, s[role]), labels_for(manifest, s[role]), scores.argmax(axis=1), scores, "logit")
    model_sha = sha256_file(udir / "model.npz")
    meta = dict(u, status="done", environment=env, model_sha256=model_sha,
                timing={"seconds": timer.seconds, "started_at": started_at, "finished_at": now()},
                **{f"{r}_sha256": shas[r] for r in PRED_ROLES})
    atomic_write_json(udir / "unit.json", meta)
    atomic_write_text(udir / "DONE", " ".join([shas[r] for r in PRED_ROLES] + [model_sha]) + "\n")
    return meta


def score_unit(preds):
    # Primary only: thresholds independent of labels. Secondary risk-target rule is not used in FIX.
    test = preds["test"]
    result = {}
    ct = calib.confidence_of(test["scores"])
    et = test["y_true"] != test["y_pred"]
    for role in ("seen", "new"):
        p = preds[role]
        conf = calib.confidence_of(p["scores"])
        tau = calib.primary_tau(conf)
        av, at = conf >= tau, ct >= tau
        ev = p["y_true"] != p["y_pred"]
        val_error = calib.accepted_error(ev, av)
        speakers = calib.per_speaker_stats(test["speakers"], at, et)
        for v in speakers.values():
            v.update(review_deviation=1 - v["coverage"] - calib.BUDGET,
                     gap_coverage=v["coverage"] - float(av.mean()),
                     gap_accepted_error=v["accepted_error"] - val_error)
        result[role] = {"tau": tau, "val_coverage": float(av.mean()), "val_accepted_error": val_error,
                        "test_coverage": float(at.mean()), "test_accepted_error": calib.accepted_error(et, at),
                        "test_review_deviation": 1 - float(at.mean()) - calib.BUDGET,
                        "gap_coverage": float(at.mean()) - float(av.mean()),
                        "gap_accepted_error": calib.accepted_error(et, at) - val_error, "speakers": speakers}
    return result


def aggregate(scored):
    """Strict complete-r speaker averages; undefined risks stay NA with explicit count."""
    groups = defaultdict(list)
    for u in scored:
        groups[(u["level"], u["model"])].append(u)
    endpoints = []
    quantities = ("review_deviation", "gap_coverage", "accepted_error", "gap_accepted_error")
    for (level, model), units in sorted(groups.items()):
        speakers = sorted({s for u in units for s in u["seen"]["speakers"]})
        rs = sorted({u["r"] for u in units})
        matrices = {role: {q: np.full((len(rs), len(speakers)), np.nan) for q in quantities} for role in ("seen", "new")}
        counted = set()
        for u in units:
            require(set(u["seen"]["speakers"]) == set(u["new"]["speakers"]), "test speaker mismatch")
            for sp in u["seen"]["speakers"]:
                require((u["r"], sp) not in counted, "duplicate test speaker per repeat")
                counted.add((u["r"], sp))
                for role in matrices:
                    for q in quantities:
                        matrices[role][q][rs.index(u["r"]), speakers.index(sp)] = u[role]["speakers"][sp][q]
        require(len(counted) == len(rs) * len(speakers), "incomplete repeat speaker coverage")
        for q in quantities:
            values = (matrices["seen"][q] - matrices["new"][q]).mean(axis=0)
            ep = calib.paired_endpoint(values)
            ep.update(id=f"B2-FIX:{level}:{model}:seen-new:{q}", level=level, model=model,
                      comparison="seen-new", quantity=q, n_repeats=len(rs),
                      interval="conditional on fixed fitted models/calibration thresholds; strict complete-repeat paired speakers")
            endpoints.append(ep)
    return endpoints


def score_plan(plan, run):
    manifests = validate_plan(plan)
    for u in plan["units"]:
        status = done_status(run / "units" / u["unit_id"], u, plan["splits"][u["split_key"]], manifests[u["base"]])
        require(status == "ok", f"all-or-nothing scoring: {u['unit_id']} {status}")
    scored = []
    for u in plan["units"]:
        preds = {r: read_predictions_csv(run / "units" / u["unit_id"] / f"{r}_predictions.csv", "logit") for r in PRED_ROLES}
        scored.append(dict(score_unit(preds), **{k: u[k] for k in ("unit_id", "level", "model", "r", "fold")}))
    return {"program": PROGRAM, "module": MODULE, "spec": SPEC_VERSION, "scored_at": now(),
            "n_units": len(scored), "units": scored, "endpoints": aggregate(scored)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default=str(DEPLOY_ROOT / "work" / "plan" / "B2_FIXED.json"))
    for name in ("run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--limit", type=int)
        else:
            p.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    common.cpu_guard(2)
    if args.cmd == "plan":
        plan = build_plan(calib.collect_inputs(verify_caches=True))
        validate_plan(plan)
        atomic_write_json(Path(args.out), plan)
        print(f"[B2-FIX plan] {plan['n_units']} units; {len(plan['splits'])} matched splits")
    elif args.cmd == "run":
        run_plan(read_json(Path(args.plan)), Path(args.out), args.limit)
    else:
        atomic_write_json(Path(args.out), score_plan(read_json(Path(args.plan)), Path(args.run)))


if __name__ == "__main__":
    main()

"""Synthetic SER26-DEPLOY-1 study for verifier tests: 2 levels, 3 test speakers, 2 classes, 1 r, 1 fold.
Builds manifests, split tables, plans, raw outputs for A / B1 / B2 and a NAIVE (loop-based) second scorer whose
endpoints.json / per_speaker.csv the verifier must reproduce to 1e-9. No scorer module is imported."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

LEVELS = {"lvA": "alpha", "lvB": "beta"}
CLASSES = ["neutral", "angry"]
SPEAKERS = ["s1", "s2", "s3", "s4", "s5"]      # s1..s3 = test fold, s4 = GG val, s5 = fit
TEST_SPK, VAL_SPK, FIT_SPK = ["s1", "s2", "s3"], ["s4"], ["s5"]
SENTENCES = ["Q1", "Q2", "E1", "E2", "E3"]    # Q1,Q2 -> query set; E* -> enrollment pool
TAKES = 3
ENC = "wavlm_base_plus"
B2_MODELS = ["ridge_wavlm_base_plus"]
B1_MODELS = ["cnn"]
CONDITIONS = {"none": {"composition": "none", "N": None, "estimators": ["none"], "draws": 1},
              "balanced@5": {"composition": "balanced", "N": 5, "estimators": ["naive", "shrink", "neutral_filter", "prior_corrected"], "draws": 2},
              "single:neutral@3": {"composition": "single", "N": 3, "estimators": ["naive", "shrink", "neutral_filter", "prior_corrected"], "draws": 2},
              "single:angry@3": {"composition": "single", "N": 3, "estimators": ["naive", "shrink", "neutral_filter", "prior_corrected"], "draws": 2},
              "single:neutral@5": {"composition": "single", "N": 5, "estimators": ["naive", "shrink", "neutral_filter", "prior_corrected"], "draws": 2},
              "single:angry@5": {"composition": "single", "N": 5, "estimators": ["naive", "shrink", "neutral_filter", "prior_corrected"], "draws": 2},
              "other_balanced@5": {"composition": "other_balanced", "N": 5, "estimators": ["naive"], "draws": 2},
              "other_single:neutral@5": {"composition": "other_single", "N": 5, "estimators": ["naive"], "draws": 2},
              "other_single:angry@5": {"composition": "other_single", "N": 5, "estimators": ["naive"], "draws": 2},
              "balanced@10": {"composition": "balanced", "N": 10, "estimators": ["naive"], "draws": 2, "feasible": False},
              "oracle_all": {"composition": "oracle_all", "N": None, "estimators": ["naive"], "draws": 1}}


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical_digest(v) -> str:
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_json(p: Path, v):
    write_text(p, json.dumps(v, indent=1, sort_keys=True) + "\n")


def corpus_rows(base: str):
    """Deterministic synthetic manifest rows: speaker x sentence x class x take."""
    rows = []
    i = 0
    for s in SPEAKERS:
        for sent in SENTENCES:
            for ci, c in enumerate(CLASSES):
                for t in range(TAKES):
                    rows.append({"sample_index": i, "corpus": base, "relative_path": f"{s}/{sent}_{c}_{t}.wav", "bytes": 1000 + i,
                                 "sha256": "0" * 64, "speaker": s, "sex": "F", "label": c, "label_index": ci, "sentence": sent, "take": f"T{t}", "intensity": ""})
                    i += 1
    return rows


def write_manifest(root: Path, base: str, rows):
    cols = ["sample_index", "corpus", "relative_path", "bytes", "sha256", "speaker", "sex", "label", "label_index", "sentence", "take", "intensity"]
    write_text(root / "manifests" / f"{base}_manifest.csv", ",".join(cols) + "\n" + "".join(",".join(str(r[c]) for c in cols) + "\n" for r in rows))


def build_splits(v2root: Path, rows_by_level: dict):
    """One fold per level and cell. GG: val = s4 (speaker-exclusive); GR: val = half of s4+s5 utterances (overlaps fit)."""
    index = {}
    for lvl, rows in rows_by_level.items():
        pop = [r["relative_path"] for r in rows]
        test = [p for p in pop if p.split("/")[0] in TEST_SPK]
        for cell in ("GG", "GR"):
            if cell == "GG":
                val = [p for p in pop if p.split("/")[0] in VAL_SPK]
            else:
                rest = [p for p in pop if p.split("/")[0] not in TEST_SPK]
                val = rest[::2]
            key = f"ctrl__{lvl}__{cell}__r0"
            table = {"key": key, "population": pop, "folds": [{"fold": 0, "test": test, "val": val, "meta": {"cell": cell, "r": 0}}]}
            path = v2root / "plan_rc2" / "splits" / f"{key}.json"
            write_json(path, table)
            index[key] = {"n_folds": 1, "path": f"splits/{key}.json", "sha256": sha_file(path)}
    return index


def query_and_pool(rows):
    q, e = {}, {}
    for r in rows:
        (q if r["sentence"].startswith("Q") else e).setdefault(r["speaker"], []).append(r["relative_path"])
    return q, e


def unit_identity(module, **kw):
    ident = {"program": "SER26-DEPLOY-1", "module": module, **kw}
    return canonical_digest(ident), ident


def build_plans(plan_dir: Path, rows_by_level: dict, index: dict):
    a_units, b2_units, levels = [], [], {}
    for lvl, rows in rows_by_level.items():
        q, e = query_and_pool(rows)
        conds = [{"condition": name, "composition": c["composition"], "N": c["N"], "feasible": c.get("feasible", True),
                  "estimators": c["estimators"], "draws": c["draws"]} for name, c in CONDITIONS.items()]
        levels[lvl] = {"query": q, "enrollment": e, "conditions": conds, "draws": 2}
        key = f"ctrl__{lvl}__GG__r0"
        uid, ident = unit_identity("A", level=lvl, enc=ENC, cell="GG", r=0, fold=0, split_key=key, split_sha256=index[key]["sha256"])
        a_units.append({"unit_id": uid, "level": lvl, "enc": ENC, "r": 0, "fold": 0, "split_key": key, "split_sha256": index[key]["sha256"], "identity": ident})
        for model in B2_MODELS:
            for cell in ("GG", "GR"):
                key = f"ctrl__{lvl}__{cell}__r0"
                uid, ident = unit_identity("B2", level=lvl, model=model, cell=cell, r=0, fold=0, split_key=key, split_sha256=index[key]["sha256"])
                b2_units.append({"unit_id": uid, "level": lvl, "model": model, "cell": cell, "r": 0, "fold": 0, "split_key": key,
                                 "split_sha256": index[key]["sha256"], "identity": ident})
    write_json(plan_dir / "A.json", {"program": "SER26-DEPLOY-1", "module": "A", "units": a_units, "levels": levels})
    write_json(plan_dir / "B2.json", {"program": "SER26-DEPLOY-1", "module": "B2", "units": b2_units})
    return a_units, b2_units


def synth_predictions(rng, y_true, acc_p):
    """Random predictions with class-dependent accuracy; logits whose argmax equals y_pred."""
    y_true = np.asarray(y_true)
    correct = rng.random(len(y_true)) < acc_p
    y_pred = np.where(correct, y_true, 1 - y_true)
    logits = rng.normal(size=(len(y_true), 2))
    margin = rng.random(len(y_true)) * 2.0 + 0.05
    logits[np.arange(len(y_true)), y_pred] = logits[np.arange(len(y_true)), 1 - y_pred] + margin
    return y_pred, logits


def write_a_units(run_dir: Path, a_units, rows_by_level: dict, rng):
    for u in a_units:
        rows = rows_by_level[u["level"]]
        man = {r["relative_path"]: r for r in rows}
        q, _ = query_and_pool(rows)
        lines = ["condition,estimator,draw,relative_path,speaker,y_true,y_pred"]
        for name, c in CONDITIONS.items():
            if not c.get("feasible", True):
                continue
            for est in c["estimators"]:
                acc_p = {"none": 0.7, "naive": 0.75, "shrink": 0.72, "neutral_filter": 0.68, "prior_corrected": 0.74}[est]
                if c["composition"].startswith("single") or c["composition"] == "other_single":
                    acc_p -= 0.15
                for d in range(c["draws"]):
                    for s in TEST_SPK:
                        paths = q[s]
                        yt = [man[p]["label_index"] for p in paths]
                        yp, _ = synth_predictions(rng, yt, acc_p)
                        for p, a, b in zip(paths, yt, yp):
                            lines.append(f"{name},{est},{d},{p},{s},{a},{int(b)}")
        udir = run_dir / "A" / "units" / u["unit_id"]
        udir.mkdir(parents=True, exist_ok=True)
        with gzip.open(udir / "predictions.csv.gz", "wt", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        write_json(udir / "model.json", {k: u[k] for k in ("unit_id", "level", "enc", "r", "fold", "split_key", "split_sha256")})
        write_text(udir / "DONE", sha_file(udir / "predictions.csv.gz") + "\n")


def logit_csv(paths, man, y_pred, logits, sample_index=False):
    head = (["sample_index"] if sample_index else []) + ["relative_path", "speaker", "y_true", "y_pred", "logit_0", "logit_1"]
    lines = [",".join(head)]
    for i, (p, yp, lg) in enumerate(zip(paths, y_pred, logits)):
        r = man[p]
        lines.append(",".join(([str(i)] if sample_index else []) + [p, r["speaker"], str(r["label_index"]), str(int(yp)), repr(float(lg[0])), repr(float(lg[1]))]))
    return "\n".join(lines) + "\n"


def write_b2_units(run_dir: Path, b2_units, rows_by_level: dict, v2root: Path, rng):
    for u in b2_units:
        rows = rows_by_level[u["level"]]
        man = {r["relative_path"]: r for r in rows}
        table = json.load(open(v2root / "plan_rc2" / "splits" / f"{u['split_key']}.json", encoding="utf-8"))
        fold = table["folds"][0]
        udir = run_dir / "B2" / "units" / u["unit_id"]
        udir.mkdir(parents=True, exist_ok=True)
        acc = 0.85 if u["cell"] == "GR" else 0.7           # GR val is optimistic (overlaps fit)
        for part, acc_p in (("val", acc), ("test", 0.7)):
            paths = fold[part]
            yp, lg = synth_predictions(rng, [man[p]["label_index"] for p in paths], acc_p)
            write_text(udir / f"{part}_predictions.csv", logit_csv(paths, man, yp, lg))
        write_json(udir / "unit.json", {k: u[k] for k in ("unit_id", "level", "model", "cell", "r", "fold", "split_key", "split_sha256")})
        write_text(udir / "DONE", f"{sha_file(udir / 'val_predictions.csv')} {sha_file(udir / 'test_predictions.csv')}\n")


def write_b1_units(v2root: Path, rows_by_level: dict, index: dict, rng):
    """v2-style run_plan.csv + runs/main/units/<uid>/{unit.json, predictions.csv} for arm CTRL, cell GG, models B1_MODELS."""
    cols = ["unit_id", "arm", "corpus_level", "base_corpus", "panel_draw", "model", "cell", "fold", "r", "seed_index", "train_seed",
            "config_sha256", "split_sha256", "n_fit", "n_val", "n_test", "est_gpu_sec", "cap_group", "conditional", "truncation_rank", "status"]
    lines = [",".join(cols)]
    for lvl, rows in rows_by_level.items():
        man = {r["relative_path"]: r for r in rows}
        key = f"ctrl__{lvl}__GG__r0"
        table = json.load(open(v2root / "plan_rc2" / "splits" / f"{key}.json", encoding="utf-8"))
        test = table["folds"][0]["test"]
        for model in B1_MODELS:
            uid = canonical_digest(["v2-fixture", lvl, model, 0])
            udir = v2root / "runs" / "main" / "units" / uid
            udir.mkdir(parents=True, exist_ok=True)
            yp, lg = synth_predictions(rng, [man[p]["label_index"] for p in test], 0.7)
            write_text(udir / "predictions.csv", logit_csv(test, man, yp, lg, sample_index=True))
            cfg = canonical_digest({"model": model})
            write_json(udir / "unit.json", {"unit_id": uid, "arm": "CTRL", "model": model, "cell": "GG", "corpus_level": lvl, "fold": 0, "r": 0,
                                            "seed_index": 0, "split_sha256": index[key]["sha256"], "config_sha256": cfg, "status": "done",
                                            "predictions_sha256": sha_file(udir / "predictions.csv")})
            lines.append(",".join([uid, "CTRL", lvl, LEVELS[lvl], "", model, "GG", "0", "0", "0", "1", cfg, index[key]["sha256"],
                                   "0", "0", str(len(test)), "1", "CTRL", "0", "99", "pending"]))
        # a decoy row the selection must ignore (other cell)
        lines.append(",".join(["deadbeef", "CTRL", lvl, LEVELS[lvl], "", "cnn", "RR", "0", "0", "0", "1", "x", "y", "0", "0", "0", "1", "CTRL", "0", "99", "pending"]))
    write_text(v2root / "plan_rc2" / "run_plan.csv", "\n".join(lines) + "\n")


def build_study(root: Path, seed: int = 7) -> dict:
    """Creates the whole synthetic study under root. Returns paths and in-memory tables used by the naive scorer."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    v2root, plan_dir, run_dir, results_dir = root / "v2data", root / "work" / "plan", root / "work", root / "results"
    rows_by_level = {lvl: corpus_rows(base) for lvl, base in LEVELS.items()}
    for lvl, base in LEVELS.items():
        write_manifest(root, base, rows_by_level[lvl])
    index = build_splits(v2root, rows_by_level)
    write_json(root / "split_index.json", index)
    a_units, b2_units = build_plans(plan_dir, rows_by_level, index)
    write_a_units(run_dir, a_units, rows_by_level, rng)
    write_b2_units(run_dir, b2_units, rows_by_level, v2root, rng)
    write_b1_units(v2root, rows_by_level, index, rng)
    return {"root": root, "v2root": v2root, "plan_dir": plan_dir, "run_dir": run_dir, "results_dir": results_dir,
            "manifests_dir": root / "manifests", "split_index": root / "split_index.json", "run_plan": v2root / "plan_rc2" / "run_plan.csv",
            "rows_by_level": rows_by_level, "index": index, "a_units": a_units, "b2_units": b2_units}

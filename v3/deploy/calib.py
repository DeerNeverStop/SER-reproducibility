"""B2 - calibration-set speaker overlap (SER26-DEPLOY-1, SPEC_DEPLOY_ZH.md section 4).

CLI (run from the worktree root, env SER_V2_DATA_ROOT set, CPU only)
  python -m v3.deploy.calib plan      --out v3/deploy/work/plan/B2.json
  python -m v3.deploy.calib run-ridge --plan v3/deploy/work/plan/B2.json --out v3/deploy/work/B2 [--limit N] [--unit ID ...]
  python -m v3.deploy.calib score     --plan v3/deploy/work/plan/B2.json --run v3/deploy/work/B2 --out v3/deploy/results/B2
                                      [--models ridge_wavlm_base_plus,ridge_hubert_base,...]

Plan rows (B2.json["units"]) carry everything a runner needs: level, base (manifest base corpus), feature_corpus
(v2 base_corpus used for feature caches), model, engine ("ridge" | "p1_frozen" | "wavlm_partial_ft"), cell (GR|GG), r,
fold, seed_index, train_seed, split_key, split_sha256, config_sha256, config (full dict), feature_kind, feature_file,
feature_sha256, n_fit/n_val/n_test, n_classes.  Ridge units are run here on CPU; cnn / wavlm_ft units are run by the
GPU runner (calib_gpu.py, another author) and must produce exactly the files below.

================================  OUTPUT CONTRACT (all B2 units: ridge, cnn, wavlm_ft)  ================================
Directory: <run>/units/<unit_id>/   (default <run> = v3/deploy/work/B2)

  val_predictions.csv, test_predictions.csv
      header EXACTLY: relative_path,speaker,y_true,y_pred,logit_0,...,logit_{C-1}   (C = n_classes of the level)
      one row per path of the split fold's "val" / "test" list, IN THE ORDER OF THE SPLIT TABLE:
          table = common.load_split(row["split_key"]); fold = table["folds"][row["fold"]]; paths = fold["val"] / fold["test"]
      speaker / y_true come from the v2 manifest (common.load_manifest(row["base"]), labels_for / speakers_for);
      logits = raw model outputs (ridge: decision_function; cnn / wavlm_ft: pre-softmax logits of the best-epoch model);
      y_pred = argmax over the logits (int);
      write with common.write_predictions_csv(path, paths, speakers, y_true, y_pred, logits, score_prefix="logit")
      (repr(float) formatting, LF newlines, atomic write) - it returns the file sha256.

  unit.json
      every plan-row field verbatim, plus
          "status": "done",
          "engine": as in the plan row,
          "environment": free-form dict (hostname, python, torch/sklearn versions, device name, cuda flag, ...),
          "timing": {"started_at": iso-utc, "finished_at": iso-utc, "seconds": float},
          "best_epoch": int or null, "epochs_run": int or null   (null for ridge),
          "val_sha256", "test_sha256": sha256 of the two csv files,
          "n_val_rows", "n_test_rows": row counts (must equal n_val / n_test of the plan row).

  DONE
      exactly one line: "<val_sha256> <test_sha256>\n".  Written LAST (after both csvs and unit.json).

The scorer treats a unit as complete only if DONE exists, both shas in DONE match the csv files on disk, unit.json
carries the same shas, and the csv headers/row counts match the contract.  A partial directory without DONE is
re-run from scratch by the runner (resumable).  Runners append one JSON line per finished unit to <run>/ledger.jsonl.
========================================================================================================================

Threshold rules (SPEC 4) implemented in the scorer
  primary   : b = 0.20; tau = numpy.quantile(val_confidence, 0.20, method="linear"); accepted <=> confidence >= tau.
              promised = (val accepted-set error at tau, coverage 0.80); realised on test = coverage, accepted-set error,
              per-speaker coverage; gap = realised - promised.
  secondary : r* = 0.5 x val error rate; tau = the smallest val rejection fraction (grid = sorted val confidences,
              threshold semantics confidence >= tau) whose val accepted-set error <= r*; report test coverage / risk.
  confidence = max softmax(logits) (temperature 1).
Endpoints  : GR - GG paired by (level, model, r, fold), reduced to test speakers (each speaker is a test speaker in
             exactly one fold per r), averaged over r, speaker-equal-weight mean with 10,000-rep speaker bootstrap
             (common.bootstrap_indices(n_speakers)).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from . import common
from .common import (DEPLOY_ROOT, ENCODERS, LEVELS, PROGRAM, REPO_ROOT, STATE, V2_DATA_ROOT, IntegrityError,
                     atomic_write_json, atomic_write_text, bootstrap_indices, class_table, digest, labels_for,
                     load_manifest, load_split, manifest_path, now, read_json, read_predictions_csv, require,
                     sha256_file, softmax, speakers_for, split_index, summarize_paired, write_predictions_csv)

MODULE = "B2"
SPEC_VERSION = "DEPLOY-2"
B2_LEVELS = ("ravdess", "cremad", "subesco_980")
B2_CELLS = ("GR", "GG")
B2_RS = (0, 1, 2)
B2_FOLDS = (0, 1, 2, 3, 4)
RIDGE_CONFIG = {"kind": "ridge", "state": STATE, "alpha": 1.0, "class_weight": "balanced", "scaler": "standard"}
RIDGE_MODELS = {f"ridge_{enc}": enc for enc in ENCODERS}
ENGINE_OF = {"cnn": "p1_frozen", "wavlm_ft": "wavlm_partial_ft"}
V2_FT_MODEL = "wavlm_base_plus_ft"
V2_SEED_PROGRAM = "SER26"
BUDGET = 0.20
PROMISED_COVERAGE = 1.0 - BUDGET
SECONDARY_RISK_FACTOR = 0.5
IDENTITY_FIELDS = ("level", "base", "model", "cell", "r", "fold", "seed_index", "train_seed", "split_key",
                   "split_sha256", "config_sha256")
PREDICTION_COLUMNS = ("relative_path", "speaker", "y_true", "y_pred")
CONTRACT = {
    "files": ["val_predictions.csv", "test_predictions.csv", "unit.json", "DONE"],
    "prediction_columns": "relative_path,speaker,y_true,y_pred,logit_0..logit_{C-1}",
    "row_order": "split table order (fold['val'] / fold['test'])",
    "writer": "common.write_predictions_csv(..., score_prefix='logit')",
    "unit_json_extra": ["status", "engine", "environment", "timing", "best_epoch", "epochs_run", "val_sha256",
                        "test_sha256", "n_val_rows", "n_test_rows"],
    "done_line": "<val_sha256> <test_sha256>\\n",
}


# ============================================================================= seeds (v2 rule, re-implemented)

def stable_u32(text: str) -> int:
    """v2 seed primitive: first 32 bits of SHA-256(text), big-endian."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def v2_train_seed(corpus: str, fold: int, seed_index: int) -> int:
    """v2 seeds.train_seed / crossing_train_seed share this key (r and seed_index occupy the same slot)."""
    return stable_u32(f"{V2_SEED_PROGRAM}|train|{corpus}|{fold}|{seed_index}")


def v2_ft_train_seed(corpus: str, fold: int, seed_index: int) -> int:
    return stable_u32(f"{V2_SEED_PROGRAM}|train|{corpus}_ft|{fold}|{seed_index}")


SEED_RULE = ("cnn: train_seed = v2 CTRL unit (same level, cell, r, fold, seed_index) when present, else "
             "stable_u32('SER26|train|<level>|<fold>|<seed_index>') (identical to v2 seeds.crossing_train_seed; "
             "coincides with every present v2 unit); wavlm_ft: v2 FT GG unit of the same level/fold, seed_index 0 "
             "(= stable_u32('SER26|train|<level>_ft|<fold>|0')), used for both cells; ridge: 0 (deterministic solver).")


# ============================================================================= plan inputs

def unit_identity(row: dict) -> dict:
    ident = {"program": PROGRAM, "module": MODULE}
    ident.update({k: row[k] for k in IDENTITY_FIELDS})
    return ident


def make_unit_id(row: dict) -> str:
    return digest(unit_identity(row))


def read_v2_plan_rows() -> list:
    with open(REPO_ROOT / "v2" / "plan_rc2" / "run_plan.csv", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def find_feature_file(corpus: str, kind: str) -> Path:
    cands = sorted((V2_DATA_ROOT / "features").glob(f"{corpus}__{kind}__*.npz"))
    require(len(cands) == 1, f"expected exactly one feature cache for {corpus}/{kind}, found {len(cands)}")
    return cands[0]


def collect_inputs(verify_caches: bool = True) -> dict:
    """Everything the plan needs from v2 (read-only). Split files are verified against split_index.json."""
    v2_rows = read_v2_plan_rows()
    configs = read_json(REPO_ROOT / "v2" / "plan_rc2" / "unit_configs.json")
    sidx = split_index()
    levels = {}
    for level in B2_LEVELS:
        base = LEVELS[level]
        manifest = load_manifest(base)
        classes = class_table(manifest)
        bases = sorted({r["base_corpus"] for r in v2_rows if r["corpus_level"] == level})
        require(len(bases) == 1, f"ambiguous v2 base_corpus for {level}: {bases}")
        feature_corpus = bases[0]
        splits = {}
        for cell in B2_CELLS:
            for r in B2_RS:
                key = f"ctrl__{level}__{cell}__r{r}"
                table = load_split(key, verify=True)
                require(len(table["folds"]) == len(B2_FOLDS), f"{key}: expected {len(B2_FOLDS)} folds")
                for k, f in enumerate(table["folds"]):
                    require(int(f["fold"]) == k, f"{key}: fold order broken")
                splits[key] = {"sha256": sidx[key]["sha256"], "table": table}
        # GG and GR share outer test folds per r (SPEC 1)
        for r in B2_RS:
            gg, gr = splits[f"ctrl__{level}__GG__r{r}"]["table"], splits[f"ctrl__{level}__GR__r{r}"]["table"]
            require(gg["population"] == gr["population"], f"{level} r{r}: GG/GR populations differ")
            for fg, fr in zip(gg["folds"], gr["folds"]):
                require(fg["test"] == fr["test"], f"{level} r{r} fold {fg['fold']}: GG/GR outer test folds differ")
        population = set(next(iter(splits.values()))["table"]["population"])
        require(population <= set(manifest), f"{level}: population paths missing from manifest {base}")
        features = {}
        for kind in list(ENCODERS) + ["logmel"]:
            path = find_feature_file(feature_corpus, kind)
            entry = {"file": path.name, "sha256": sha256_file(path)}
            if verify_caches:
                paths = set(np.load(path, allow_pickle=False)["paths"].tolist())
                require(population <= paths, f"{level}: population paths missing from cache {path.name}")
            features[kind] = entry
        levels[level] = {"base": base, "feature_corpus": feature_corpus, "manifest_sha256": sha256_file(manifest_path(base)),
                         "classes": classes, "n_speakers": len({manifest[p]["speaker"] for p in population}),
                         "splits": splits, "features": features}
    return {"v2_rows": v2_rows, "configs": configs, "levels": levels,
            "v2_run_plan_sha256": sha256_file(REPO_ROOT / "v2" / "plan_rc2" / "run_plan.csv"),
            "v2_unit_configs_sha256": sha256_file(REPO_ROOT / "v2" / "plan_rc2" / "unit_configs.json"),
            "split_index_sha256": sha256_file(REPO_ROOT / "v2" / "plan_rc2" / "split_index.json")}


# ============================================================================= plan construction (pure)

def _v2_lookup(v2_rows: list, **where) -> list:
    return [r for r in v2_rows if all(str(r[k]) == str(v) for k, v in where.items())]


def _cnn_reference(v2_rows: list) -> str:
    shas = {r["config_sha256"] for r in v2_rows if r["arm"] == "CTRL" and r["model"] == "cnn"
            and r["cell"] in B2_CELLS and r["corpus_level"] in B2_LEVELS}
    require(len(shas) == 1, f"v2 CTRL cnn config sha not unique: {shas}")
    return shas.pop()


def _ft_reference(v2_rows: list) -> dict:
    """(level, fold) -> v2 FT GG seed-0 unit (train_seed, config_sha256, split_sha256, unit_id)."""
    out = {}
    shas = set()
    for r in v2_rows:
        if r["arm"] == "FT" and r["model"] == V2_FT_MODEL and r["cell"] == "GG" and r["seed_index"] == "0" \
                and r["corpus_level"] in B2_LEVELS:
            require(r["r"] == "0", "v2 FT units are expected at r0 only")
            out[(r["corpus_level"], int(r["fold"]))] = r
            shas.add(r["config_sha256"])
    require(len(shas) == 1, f"v2 FT seed-0 config sha not unique: {shas}")
    return out


def build_units(inputs: dict) -> list:
    v2_rows, configs, levels = inputs["v2_rows"], inputs["configs"], inputs["levels"]
    cnn_sha = _cnn_reference(v2_rows)
    cnn_cfg = configs[cnn_sha]
    require(digest(cnn_cfg) == cnn_sha, "v2 cnn config body does not hash to its config_sha256")
    ft_ref = _ft_reference(v2_rows)
    ft_sha = next(iter(ft_ref.values()))["config_sha256"]
    ft_cfg = configs[ft_sha]
    require(digest(ft_cfg) == ft_sha, "v2 FT config body does not hash to its config_sha256")
    ridge_sha = digest(RIDGE_CONFIG)
    units = []
    for level in B2_LEVELS:
        L = levels[level]
        C = len(L["classes"])
        for cell in B2_CELLS:
            for r in B2_RS:
                key = f"ctrl__{level}__{cell}__r{r}"
                split = L["splits"][key]
                for fold in B2_FOLDS:
                    f = split["table"]["folds"][fold]
                    base_row = {"program": PROGRAM, "module": MODULE, "spec": SPEC_VERSION, "level": level,
                                "base": L["base"], "feature_corpus": L["feature_corpus"], "cell": cell, "r": r,
                                "fold": fold, "split_key": key, "split_sha256": split["sha256"],
                                "n_fit": len(f["fit"]), "n_val": len(f["val"]), "n_test": len(f["test"]),
                                "n_classes": C, "manifest_sha256": L["manifest_sha256"]}
                    # --- ridge x 3 encoders
                    for model, enc in RIDGE_MODELS.items():
                        row = dict(base_row, model=model, engine="ridge", seed_index=0, train_seed=0,
                                   train_seed_source="unused:deterministic_solver", v2_reference_unit=None,
                                   config_sha256=ridge_sha, config=dict(RIDGE_CONFIG), feature_kind=enc,
                                   feature_file=L["features"][enc]["file"], feature_sha256=L["features"][enc]["sha256"])
                        units.append(row)
                    # --- cnn (v2 p1_frozen), seed_index 0
                    ref = _v2_lookup(v2_rows, arm="CTRL", model="cnn", corpus_level=level, cell=cell, r=r, fold=fold,
                                     seed_index=0)
                    derived = v2_train_seed(level, fold, 0)
                    if ref:
                        require(len(ref) == 1, "duplicate v2 cnn reference")
                        require(int(ref[0]["train_seed"]) == derived, f"v2 cnn train_seed differs from the seed rule at {level} {cell} r{r} f{fold}")
                        require(ref[0]["split_sha256"] == split["sha256"], f"v2 cnn split sha mismatch at {level} {cell} r{r}")
                        seed, source, ref_id = int(ref[0]["train_seed"]), "v2_unit", ref[0]["unit_id"]
                    else:
                        seed, source, ref_id = derived, "derived:SER26|train|<level>|<fold>|0", None
                    units.append(dict(base_row, model="cnn", engine=ENGINE_OF["cnn"], seed_index=0, train_seed=seed,
                                      train_seed_source=source, v2_reference_unit=ref_id, config_sha256=cnn_sha,
                                      config=dict(cnn_cfg), feature_kind="logmel",
                                      feature_file=L["features"]["logmel"]["file"],
                                      feature_sha256=L["features"]["logmel"]["sha256"]))
                    # --- wavlm_ft (v2 wavlm_partial_ft), r0 and seed_index 0 only
                    if r == 0:
                        fr = ft_ref[(level, fold)]
                        require(int(fr["train_seed"]) == v2_ft_train_seed(level, fold, 0), "v2 FT train_seed differs from the seed rule")
                        require(fr["split_sha256"] == L["splits"][f"ctrl__{level}__GG__r0"]["sha256"], "v2 FT GG split sha mismatch")
                        units.append(dict(base_row, model="wavlm_ft", engine=ENGINE_OF["wavlm_ft"], seed_index=0,
                                          train_seed=int(fr["train_seed"]), train_seed_source="v2_ft_gg_unit",
                                          v2_reference_unit=fr["unit_id"], config_sha256=ft_sha, config=dict(ft_cfg),
                                          feature_kind=None, feature_file=None, feature_sha256=None))
    for row in units:
        row["unit_id"] = make_unit_id(row)
    units.sort(key=lambda u: (u["level"], u["model"], u["cell"], u["r"], u["fold"]))
    require(len({u["unit_id"] for u in units}) == len(units), "duplicate unit ids")
    return units


def build_plan(inputs: dict) -> dict:
    units = build_units(inputs)
    counts = {}
    for u in units:
        counts[u["model"]] = counts.get(u["model"], 0) + 1
    levels = {lv: {k: v for k, v in L.items() if k != "splits"} | {"split_sha256": {k: s["sha256"] for k, s in L["splits"].items()}}
              for lv, L in inputs["levels"].items()}
    return {
        "program": PROGRAM, "module": MODULE, "spec": SPEC_VERSION, "generated_at": now(),
        "code_sha256": {"calib.py": sha256_file(Path(__file__)), "common.py": sha256_file(DEPLOY_ROOT / "common.py")},
        "v2_inputs": {"run_plan.csv": inputs["v2_run_plan_sha256"], "unit_configs.json": inputs["v2_unit_configs_sha256"],
                      "split_index.json": inputs["split_index_sha256"]},
        "manifests": {L["base"]: L["manifest_sha256"] for L in inputs["levels"].values()},
        "levels": levels,
        "seed_rule": SEED_RULE,
        "ridge_config": RIDGE_CONFIG,
        "threshold_rules": {"primary": {"budget": BUDGET, "quantile_method": "linear", "accept": "confidence >= tau",
                                        "nominal_coverage": PROMISED_COVERAGE, "coverage_gap": "test minus actual val coverage"},
                            "secondary": {"risk_factor": SECONDARY_RISK_FACTOR,
                                          "min_accepted": 1, "grid": "unique observed validation confidences ascending",
                                          "no_feasible": "tau=+inf, coverage=0, accepted risk=NA, feasible=false",
                                          "rule": "smallest val rejection fraction with val accepted error <= r*"}},
        "interpretation": "whole inner-validation protocol effect; models/training data differ between GR and GG",
        "contract": CONTRACT,
        "counts": counts, "n_units": len(units), "units": units,
    }


def write_summary_md(plan: dict, path: Path) -> None:
    lines = [f"# B2 plan summary ({PROGRAM}, spec {SPEC_VERSION})", "", f"generated_at: {plan['generated_at']}",
             f"n_units: {plan['n_units']}", "", "## Units per model", "", "| model | engine | units |", "|---|---|---|"]
    engines = {u["model"]: u["engine"] for u in plan["units"]}
    for m, n in sorted(plan["counts"].items()):
        lines.append(f"| {m} | {engines[m]} | {n} |")
    lines += ["", "## Units per level x model", "", "| level | " + " | ".join(sorted(plan["counts"])) + " |",
              "|---|" + "---|" * len(plan["counts"])]
    for lv in B2_LEVELS:
        row = [str(sum(1 for u in plan["units"] if u["level"] == lv and u["model"] == m)) for m in sorted(plan["counts"])]
        lines.append(f"| {lv} | " + " | ".join(row) + " |")
    derived = sum(1 for u in plan["units"] if u["model"] == "cnn" and u["train_seed_source"].startswith("derived"))
    lines += ["", "## Notes", "", f"- seed rule: {plan['seed_rule']}", f"- cnn units with derived train_seed: {derived}",
              "- feature caches: v2 base_corpus caches (subesco_980 level uses the subesco_980__* caches, as v2 did); "
              "labels/speakers from the v2 base manifest (common.LEVELS).",
              "- output contract: see the docstring of v3/deploy/calib.py or plan['contract'].", ""]
    atomic_write_text(path, "\n".join(lines))


def cmd_plan(args) -> None:
    common.cpu_guard(2)
    inputs = collect_inputs(verify_caches=not args.no_cache_check)
    plan = build_plan(inputs)
    out = Path(args.out)
    atomic_write_json(out, plan)
    write_summary_md(plan, out.with_name("B2_summary.md"))
    print(f"[plan] wrote {out} ({plan['n_units']} units): {plan['counts']}")


# ============================================================================= unit directory helpers

def unit_dir(run_root: Path, unit_id: str) -> Path:
    return run_root / "units" / unit_id


def done_status(udir: Path, n_classes: int | None = None, n_val: int | None = None, n_test: int | None = None) -> str:
    """'ok' when the unit is complete and self-consistent, else a short reason."""
    done, vp, tp, uj = udir / "DONE", udir / "val_predictions.csv", udir / "test_predictions.csv", udir / "unit.json"
    if not done.is_file():
        return "missing DONE"
    if not (vp.is_file() and tp.is_file() and uj.is_file()):
        return "missing files"
    parts = done.read_text(encoding="utf-8").split()
    if len(parts) != 2:
        return "malformed DONE"
    if parts[0] != sha256_file(vp) or parts[1] != sha256_file(tp):
        return "DONE sha mismatch"
    unit = read_json(uj)
    if unit.get("val_sha256") != parts[0] or unit.get("test_sha256") != parts[1]:
        return "unit.json sha mismatch"
    for path, n in ((vp, n_val), (tp, n_test)):
        with open(path, encoding="utf-8", newline="") as fh:
            header = fh.readline().rstrip("\n").split(",")
            rows = sum(1 for _ in fh)
        if n_classes is not None and header != list(PREDICTION_COLUMNS) + [f"logit_{i}" for i in range(n_classes)]:
            return f"bad header in {path.name}"
        if n is not None and rows != n:
            return f"row count {rows} != {n} in {path.name}"
    return "ok"


def write_unit_outputs(udir: Path, row: dict, extra: dict, val: dict, test: dict) -> dict:
    """val/test: dicts with paths, speakers, y_true, y_pred, scores. Writes csvs, unit.json, then DONE."""
    udir.mkdir(parents=True, exist_ok=True)
    shas = {}
    for role, d in (("val", val), ("test", test)):
        require(d["scores"].shape == (len(d["paths"]), row["n_classes"]), f"{role} score shape mismatch")
        shas[role] = write_predictions_csv(udir / f"{role}_predictions.csv", d["paths"], d["speakers"], d["y_true"],
                                           d["y_pred"], d["scores"], score_prefix="logit")
    unit = dict(row)
    unit.update(extra)
    unit.update({"status": "done", "val_sha256": shas["val"], "test_sha256": shas["test"],
                 "n_val_rows": len(val["paths"]), "n_test_rows": len(test["paths"])})
    atomic_write_json(udir / "unit.json", unit)
    atomic_write_text(udir / "DONE", f"{shas['val']} {shas['test']}\n")
    return unit


# ============================================================================= ridge runner (CPU)

def fit_ridge_unit(X_fit, y_fit, X_val, X_test, n_classes: int, config: dict):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    require(config["scaler"] == "standard" and config["kind"] == "ridge", "unsupported ridge config")
    scaler = StandardScaler().fit(X_fit)
    clf = RidgeClassifier(alpha=config["alpha"], class_weight=config["class_weight"]).fit(scaler.transform(X_fit), y_fit)
    require(clf.classes_.tolist() == list(range(n_classes)), "fit set does not contain every class")
    out = []
    for X in (X_val, X_test):
        scores = np.asarray(clf.decision_function(scaler.transform(X)), dtype=np.float64)
        require(scores.ndim == 2 and scores.shape[1] == n_classes, "decision_function shape")
        out.append(scores)
    return out


def environment_info(threads: int) -> dict:
    import sklearn
    return {"hostname": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__,
            "sklearn": sklearn.__version__, "device": "cpu", "threads": threads, "platform": platform.platform()}


def cmd_run_ridge(args) -> None:
    guard = common.cpu_guard(2)
    plan = read_json(args.plan)
    run_root = Path(args.out)
    units = [u for u in plan["units"] if u["engine"] == "ridge"]
    if args.unit:
        wanted = set(args.unit)
        units = [u for u in units if u["unit_id"] in wanted]
        require(len(units) == len(wanted), "unknown --unit id")
    ledger = run_root / "ledger.jsonl"
    env = environment_info(guard["threads"])
    manifests, splits, cache, cache_key = {}, {}, None, None
    n_done = n_skipped = 0
    for i, u in enumerate(units):
        udir = unit_dir(run_root, u["unit_id"])
        if done_status(udir, u["n_classes"], u["n_val"], u["n_test"]) == "ok":
            n_skipped += 1
            continue
        if args.limit is not None and n_done >= args.limit:
            break
        with common.Timer() as t:
            started = now()
            if u["base"] not in manifests:
                manifests[u["base"]] = load_manifest(u["base"])
            manifest = manifests[u["base"]]
            require(len(class_table(manifest)) == u["n_classes"], "class count mismatch")
            if u["split_key"] not in splits:
                splits[u["split_key"]] = load_split(u["split_key"], verify=True)
            fold = splits[u["split_key"]]["folds"][u["fold"]]
            require((len(fold["fit"]), len(fold["val"]), len(fold["test"])) == (u["n_fit"], u["n_val"], u["n_test"]), "fold sizes")
            key = (u["feature_corpus"], u["feature_kind"])
            if cache_key != key:
                cache = common.FeatureCache(*key)
                require(cache.sha256 == u["feature_sha256"], f"feature cache sha mismatch for {key}")
                cache_key = key
            state = u["config"]["state"]
            X = {role: cache.get(fold[role], state=state) for role in ("fit", "val", "test")}
            y = {role: labels_for(manifest, fold[role]) for role in ("fit", "val", "test")}
            s_val, s_test = fit_ridge_unit(X["fit"], y["fit"], X["val"], X["test"], u["n_classes"], u["config"])
            preds = {}
            for role, s in (("val", s_val), ("test", s_test)):
                preds[role] = {"paths": fold[role], "speakers": speakers_for(manifest, fold[role]), "y_true": y[role],
                               "y_pred": s.argmax(axis=1), "scores": s}
        extra = {"environment": env, "timing": {"started_at": started, "finished_at": now(), "seconds": t.seconds},
                 "best_epoch": None, "epochs_run": None}
        unit = write_unit_outputs(udir, u, extra, preds["val"], preds["test"])
        with open(ledger, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"unit_id": u["unit_id"], "status": "done", "seconds": t.seconds, "finished_at": unit["timing"]["finished_at"],
                                 "val_sha256": unit["val_sha256"], "test_sha256": unit["test_sha256"]}, sort_keys=True) + "\n")
        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            print(f"[run-ridge] {i + 1}/{len(units)} done={n_done} skipped={n_skipped} last={t.seconds:.2f}s", flush=True)
    print(f"[run-ridge] finished: done={n_done} skipped={n_skipped} of {len(units)} ridge units", flush=True)


# ============================================================================= scoring primitives (pure numpy)

def confidence_of(logits: np.ndarray) -> np.ndarray:
    return softmax(logits).max(axis=1)


def primary_tau(conf_val: np.ndarray, budget: float = BUDGET) -> float:
    conf = np.asarray(conf_val, dtype=np.float64)
    require(conf.ndim == 1 and conf.size > 0 and np.isfinite(conf).all() and 0 <= budget < 1, "invalid calibration confidences/budget")
    return float(np.quantile(conf, budget, method="linear"))


def accepted_error(err: np.ndarray, acc: np.ndarray) -> float:
    return float(err[acc].mean()) if acc.any() else float("nan")


def secondary_tau(conf_val: np.ndarray, err_val: np.ndarray, factor: float = SECONDARY_RISK_FACTOR) -> dict:
    """Smallest rejection fraction (threshold grid = sorted val confidences) with val accepted error <= r*.
    Threshold semantics: accepted <=> confidence >= tau. If no finite threshold satisfies it, tau = +inf (reject all)."""
    conf = np.asarray(conf_val, dtype=np.float64)
    err = np.asarray(err_val, dtype=np.float64)
    n = conf.shape[0]
    require(n > 0 and conf.shape == err.shape and np.isfinite(conf).all(), "invalid secondary inputs")
    r_star = factor * float(err.mean())
    order = np.argsort(conf, kind="stable")
    cs, es = conf[order], err[order]
    suffix = np.concatenate([np.cumsum(es[::-1])[::-1], [0.0]])          # suffix[j] = errors among sorted[j:]
    first = np.searchsorted(cs, cs, side="left")                          # accepted set for tau=cs[k] starts at first[k]
    n_acc = n - first
    acc_err = suffix[first] / n_acc                                       # n_acc >= 1 always
    ok = np.nonzero(acc_err <= r_star)[0]
    if ok.size:
        k = int(ok[0])
        j = int(first[k])
        return {"r_star": r_star, "tau": float(cs[k]), "val_rejection": j / n, "val_coverage": 1.0 - j / n,
                "val_accepted_error": float(acc_err[k]), "feasible": True, "min_accepted": 1}
    return {"r_star": r_star, "tau": float("inf"), "val_rejection": 1.0, "val_coverage": 0.0,
            "val_accepted_error": float("nan"), "feasible": False, "min_accepted": 1}


def per_speaker_stats(speakers, acc: np.ndarray, err: np.ndarray) -> dict:
    """speaker -> coverage, accepted_error (nan if nothing accepted), n, n_acc, n_err_acc."""
    speakers = np.asarray(speakers)
    out = {}
    for s in sorted(set(speakers.tolist())):
        m = speakers == s
        a = acc[m]
        out[s] = {"coverage": float(a.mean()), "accepted_error": accepted_error(err[m], a), "n": int(m.sum()),
                  "n_acc": int(a.sum()), "n_err_acc": int((err[m] & a).sum())}
    return out


def score_unit(val: dict, test: dict, budget: float = BUDGET) -> dict:
    """Apply both threshold rules; val/test are read_predictions_csv dicts (scores = logits)."""
    cv, ct = confidence_of(val["scores"]), confidence_of(test["scores"])
    ev, et = val["y_pred"] != val["y_true"], test["y_pred"] != test["y_true"]
    tau = primary_tau(cv, budget)
    acc_v, acc_t = cv >= tau, ct >= tau
    promised_err = accepted_error(ev, acc_v)
    primary = {"tau": tau, "budget": budget, "promised_coverage": 1.0 - budget, "val_coverage": float(acc_v.mean()),
               "val_accepted_error": promised_err, "val_error_rate": float(ev.mean()),
               "test_coverage": float(acc_t.mean()), "test_accepted_error": accepted_error(et, acc_t),
               "test_error_rate": float(et.mean()),
               "n_val": int(ev.size), "n_val_accepted": int(acc_v.sum()), "n_val_accepted_errors": int((ev & acc_v).sum()),
               "n_test": int(et.size), "n_test_accepted": int(acc_t.sum()), "n_test_accepted_errors": int((et & acc_t).sum())}
    primary["gap_coverage"] = primary["test_coverage"] - primary["val_coverage"]
    primary["test_review_deviation"] = 1.0 - primary["test_coverage"] - budget
    primary["gap_accepted_error"] = primary["test_accepted_error"] - promised_err
    sec = secondary_tau(cv, ev)
    acc_t2 = ct >= sec["tau"]
    sec.update({"test_coverage": float(acc_t2.mean()), "test_accepted_error": accepted_error(et, acc_t2)})
    acc_v2 = cv >= sec["tau"]
    sec.update(n_val=int(ev.size), n_val_accepted=int(acc_v2.sum()), n_val_accepted_errors=int((ev & acc_v2).sum()),
               n_test=int(et.size), n_test_accepted=int(acc_t2.sum()), n_test_accepted_errors=int((et & acc_t2).sum()))
    sec["gap_accepted_error"] = sec["test_accepted_error"] - sec["val_accepted_error"]
    sec["gap_coverage"] = sec["test_coverage"] - sec["val_coverage"]
    sp1 = per_speaker_stats(test["speakers"], acc_t, et)
    sp2 = per_speaker_stats(test["speakers"], acc_t2, et)
    speakers = {}
    for s in sp1:
        speakers[s] = dict(sp1[s])
        speakers[s]["gap_coverage"] = sp1[s]["coverage"] - primary["val_coverage"]
        speakers[s]["gap_accepted_error"] = sp1[s]["accepted_error"] - promised_err
        speakers[s]["coverage_secondary"] = sp2[s]["coverage"]
        speakers[s]["accepted_error_secondary"] = sp2[s]["accepted_error"]
        speakers[s]["gap_accepted_error_secondary"] = sp2[s]["accepted_error"] - sec["val_accepted_error"]
    return {"primary": primary, "secondary": sec, "speakers": speakers}


SPEAKER_QUANTITIES = ("coverage", "accepted_error", "gap_coverage", "gap_accepted_error", "coverage_secondary",
                      "accepted_error_secondary", "gap_accepted_error_secondary")


def speaker_matrix(unit_results: list, speakers: list, quantity: str) -> np.ndarray:
    """(n_r, n_speakers) matrix from a list of scored units (one cell, one level/model); each speaker exactly once per r."""
    rs = sorted({u["r"] for u in unit_results})
    pos = {s: i for i, s in enumerate(speakers)}
    M = np.full((len(rs), len(speakers)), np.nan)
    seen = {r: set() for r in rs}
    for u in unit_results:
        i = rs.index(u["r"])
        for s, v in u["speakers"].items():
            require(s in pos, f"speaker {s} of unit {u['unit_id']} is not in the population speaker list")
            require(s not in seen[u["r"]], f"speaker {s} appears in two test folds of r{u['r']}")
            seen[u["r"]].add(s)
            M[i, pos[s]] = v[quantity]
    for r in rs:
        require(seen[r] == set(speakers), f"r{r}: test speakers do not cover the population")
    return M


def reduce_over_r(M: np.ndarray) -> np.ndarray:
    """Strict mean over all predeclared repeats; any undefined risk remains NA."""
    return np.asarray(M, dtype=np.float64).mean(axis=0)


def paired_endpoint(values: np.ndarray) -> dict:
    """Equal-weight defined-speaker risk; shared FULL-population bootstrap indices.

    Undefined speaker risks never become zero; each resample averages its finite
    entries, with undefined draws recorded and excluded only from CI quantiles.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    kept = values[finite]
    if kept.size == 0:
        return {"estimate": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0, "n_dropped": int((~finite).sum())}
    idx = bootstrap_indices(values.size)
    counts = finite[idx].sum(axis=1)
    sums = np.where(finite, values, 0.0)[idx].sum(axis=1)
    valid = counts > 0
    draws = sums[valid] / counts[valid]
    lo, hi = np.percentile(draws, [2.5, 97.5], method="linear")
    res = {"estimate": float(kept.mean()), "ci95": [float(lo), float(hi)], "n": int(kept.size),
           "n_dropped": int((~finite).sum()), "n_population": int(values.size),
           "n_undefined_bootstrap": int((~valid).sum())}
    return res


def coverage_profile(cov: np.ndarray) -> dict:
    cov = np.asarray(cov, dtype=np.float64)
    return {"min": float(cov.min()), "p10": float(np.percentile(cov, 10, method="linear")),
            "count_below_0.70": int((cov < 0.70).sum()), "count_zero": int((cov == 0).sum()), "n_speakers": int(cov.size)}


# ============================================================================= scorer

def load_unit_predictions(udir: Path, row: dict) -> tuple:
    status = done_status(udir, row["n_classes"], row["n_val"], row["n_test"])
    require(status == "ok", f"unit {row['unit_id']} not scorable: {status}")
    val = read_predictions_csv(udir / "val_predictions.csv", "logit")
    test = read_predictions_csv(udir / "test_predictions.csv", "logit")
    meta = read_json(udir / "unit.json")
    require(meta.get("status") == "done" and all(meta.get(k) == v for k, v in row.items()), "unit metadata differs from plan")
    for d in (val, test):
        require(np.isfinite(d["scores"]).all(), "nonfinite prediction scores")
        require(np.array_equal(d["scores"].argmax(axis=1), d["y_pred"]), f"unit {row['unit_id']}: y_pred is not argmax of logits")
    return val, test


def score_plan(plan: dict, run_root: Path, models: list | None) -> dict:
    units = plan["units"]
    if models:
        unknown = set(models) - {u["model"] for u in units}
        require(not unknown, f"unknown models {sorted(unknown)}")
        units = [u for u in units if u["model"] in models]
    incomplete = [(u["unit_id"], s) for u in units
                  for s in [done_status(unit_dir(run_root, u["unit_id"]), u["n_classes"], u["n_val"], u["n_test"])] if s != "ok"]
    require(not incomplete, f"{len(incomplete)} of {len(units)} planned units are not complete (first: {incomplete[:3]}); "
                            "scoring is all-or-nothing per model - use --models to restrict")
    scored, inputs_units = [], []
    for i, u in enumerate(units):
        udir = unit_dir(run_root, u["unit_id"])
        val, test = load_unit_predictions(udir, u)
        res = score_unit(val, test)
        res.update({k: u[k] for k in ("unit_id", "level", "model", "cell", "r", "fold")})
        scored.append(res)
        d = read_json(udir / "unit.json")
        inputs_units.append({"unit_id": u["unit_id"], "val_sha256": d["val_sha256"], "test_sha256": d["test_sha256"]})
        if (i + 1) % 50 == 0:
            print(f"[score] {i + 1}/{len(units)} units scored", flush=True)
    return {"scored": scored, "inputs_units": inputs_units, "units": units}


def aggregate(plan: dict, scored: list) -> dict:
    """Endpoints (GR-GG), absolute per (level, model, cell), per-speaker table rows."""
    by = {}
    for s in scored:
        by.setdefault((s["level"], s["model"]), {}).setdefault(s["cell"], []).append(s)
    endpoints, absolute, per_speaker_rows = [], [], []
    for (level, model), cells in sorted(by.items()):
        require(set(cells) == set(B2_CELLS), f"{level}/{model}: both cells are needed for GR-GG")
        speakers = sorted({s for u in cells["GG"] for s in u["speakers"]})
        reduced = {}
        for cell in B2_CELLS:
            reduced[cell] = {q: reduce_over_r(speaker_matrix(cells[cell], speakers, q)) for q in SPEAKER_QUANTITIES}
            n_r = len({u["r"] for u in cells[cell]})
            for i, spk in enumerate(speakers):
                row = {"level": level, "model": model, "cell": cell, "speaker": spk,
                       "coverage": reduced[cell]["coverage"][i], "accepted_error": reduced[cell]["accepted_error"][i], "n_r": n_r}
                for q in SPEAKER_QUANTITIES[2:]:
                    row[q] = reduced[cell][q][i]
                per_speaker_rows.append(row)
            units_c = cells[cell]
            abs_entry = {"level": level, "model": model, "cell": cell, "n_units": len(units_c), "n_r": n_r,
                         "n_speakers": len(speakers)}
            for rule in ("primary", "secondary"):
                for k in units_c[0][rule]:
                    vals = np.asarray([u[rule][k] for u in units_c], dtype=np.float64)
                    abs_entry[f"{rule}.{k}.unit_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            for q in SPEAKER_QUANTITIES:
                abs_entry[f"speaker.{q}"] = paired_endpoint(reduced[cell][q])
            abs_entry["speaker.coverage_profile"] = coverage_profile(reduced[cell]["coverage"])
            abs_entry["speaker.coverage_profile_secondary"] = coverage_profile(reduced[cell]["coverage_secondary"])
            abs_entry["coverage_profile_definition"] = "profile of mean per-speaker coverage; see profiles_by_repeat for required per-repeat counts"
            abs_entry["profiles_by_repeat"] = {
                rule: [dict(r=int(r), **coverage_profile(speaker_matrix(units_c, speakers, q)[j]))
                       for j, r in enumerate(sorted({u["r"] for u in units_c}))]
                for rule, q in (("primary", "coverage"), ("secondary", "coverage_secondary"))}
            abs_entry["mean_repeat_count_below_0.70"] = float(np.mean([p["count_below_0.70"] for p in abs_entry["profiles_by_repeat"]["primary"]]))
            if all("n_test_accepted" in u["primary"] for u in units_c):
                abs_entry["pooled_by_repeat"] = pooled_by_repeat(units_c)
            absolute.append(abs_entry)
        for q in SPEAKER_QUANTITIES:
            require(sorted({u["r"] for u in cells["GR"]}) == sorted({u["r"] for u in cells["GG"]}), "paired repeats differ")
            diff = reduce_over_r(speaker_matrix(cells["GR"], speakers, q) - speaker_matrix(cells["GG"], speakers, q))
            ep = {"id": f"B2:{level}:{model}:GR-GG:{q}", "level": level, "model": model, "comparison": "GR-GG",
                  "quantity": q, "rule": "secondary" if q.endswith("_secondary") else "primary"}
            ep.update(paired_endpoint(diff))
            endpoints.append(ep)
    return {"endpoints": endpoints, "absolute": absolute, "per_speaker_rows": per_speaker_rows}


def pooled_by_repeat(units):
    """Recording-pooled risks per repeat, separate from equal-speaker endpoints.

    Calibration pools concatenate fold-specific calibration predictions; the same
    calibration speaker may recur under different fitted models. No independent
    sample-size claim is made for that concatenated calibration reference.
    """
    rows = []
    for r in sorted({u["r"] for u in units}):
        for rule in ("primary", "secondary"):
            subset = [u[rule] for u in units if u["r"] == r]
            sums = {k: sum(u[k] for u in subset) for k in
                    ("n_val", "n_val_accepted", "n_val_accepted_errors", "n_test", "n_test_accepted", "n_test_accepted_errors")}
            row = dict(r=r, rule=rule, **sums)
            for role in ("val", "test"):
                n, a, e = (sums[f"n_{role}"], sums[f"n_{role}_accepted"], sums[f"n_{role}_accepted_errors"])
                row[f"{role}_coverage"] = a / n
                row[f"{role}_accepted_error"] = e / a if a else float("nan")
            row["test_review_deviation"] = 1 - row["test_coverage"] - BUDGET
            row["gap_coverage"] = row["test_coverage"] - row["val_coverage"]
            row["gap_accepted_error"] = row["test_accepted_error"] - row["val_accepted_error"]
            rows.append(row)
    return rows


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


def cmd_score(args) -> None:
    common.cpu_guard(2)
    plan_path = Path(args.plan)
    plan = read_json(plan_path)
    models = [m for m in args.models.split(",") if m] if args.models else None
    run_root, out = Path(args.run), Path(args.out)
    res = score_plan(plan, run_root, models)
    agg = aggregate(plan, res["scored"])
    out.mkdir(parents=True, exist_ok=True)
    ps_cols = ["level", "model", "cell", "speaker", "coverage", "accepted_error", "n_r"] + list(SPEAKER_QUANTITIES[2:])
    write_csv(out / "per_speaker.csv", agg["per_speaker_rows"], ps_cols)
    unit_rows = []
    for s in res["scored"]:
        row = {k: s[k] for k in ("unit_id", "level", "model", "cell", "r", "fold")}
        for rule in ("primary", "secondary"):
            for k, v in s[rule].items():
                row[f"{rule}.{k}"] = v
        unit_rows.append(row)
    write_csv(out / "units.csv", unit_rows, list(unit_rows[0].keys()))
    endpoints = {
        "program": PROGRAM, "module": MODULE, "spec": SPEC_VERSION, "scored_at": now(),
        "endpoints": agg["endpoints"], "absolute": agg["absolute"],
        "inputs": {"plan": str(plan_path), "plan_sha256": sha256_file(plan_path), "run_root": str(run_root),
                   "models": sorted({u["model"] for u in res["units"]}), "n_units": len(res["units"]),
                   "units": res["inputs_units"], "bootstrap": {"seed": common.BOOTSTRAP_SEED, "reps": common.BOOTSTRAP_REPS,
                                                                "unit": "speaker", "ci": "percentile 2.5/97.5"},
                   "threshold_rules": plan["threshold_rules"],
                   "code_sha256": {"calib.py": sha256_file(Path(__file__)), "common.py": sha256_file(DEPLOY_ROOT / "common.py")}},
    }
    atomic_write_json(out / "endpoints.json", endpoints)
    print(f"[score] {len(res['units'])} units, {len(agg['endpoints'])} endpoints -> {out}")


# ============================================================================= CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SER26-DEPLOY-1 module B2 (calibration-set speaker overlap)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write the B2 plan (unit table) from v2 inputs")
    p.add_argument("--out", default=str(DEPLOY_ROOT / "work" / "plan" / "B2.json"))
    p.add_argument("--no-cache-check", action="store_true", help="skip loading cache path lists (still hashes files)")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("run-ridge", help="run the ridge units on CPU (resumable)")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=str(DEPLOY_ROOT / "work" / "B2"))
    p.add_argument("--limit", type=int, default=None, help="run at most N new units")
    p.add_argument("--unit", action="append", default=None, help="restrict to these unit ids")
    p.set_defaults(func=cmd_run_ridge)
    p = sub.add_parser("score", help="score complete units and write endpoints")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", default=str(DEPLOY_ROOT / "work" / "B2"))
    p.add_argument("--out", default=str(DEPLOY_ROOT / "results" / "B2"))
    p.add_argument("--models", default="", help="comma list; default = every planned model (all units must be DONE)")
    p.set_defaults(func=cmd_score)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

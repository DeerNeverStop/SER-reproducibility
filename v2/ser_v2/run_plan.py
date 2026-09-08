"""Enumerate every unit of the program from manifests + registry, write split tables and
run_plan.csv (unit_id = SHA-256 of the unit's full configuration), and report the budget.

    python -m ser_v2.run_plan --manifests <dir with {ravdess,cremad,subesco}_manifest.csv> --out <plan dir>
    python -m ser_v2.run_plan --synthetic --out <plan dir>      # synthetic manifests, same enumeration
"""
from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import corpora, registry as R, seeds, splits
from .common import atomic_write_json, atomic_write_text, canonical_json, sha256_text, sha256_json, read_csv

HPO_GRID = [{"lr": lr, "weight_decay": wd, "dropout": do} for lr in (3e-4, 1e-3) for wd in (1e-4, 1e-3) for do in (0.1, 0.3)]
FT_SEEDS = {"ravdess": [0, 1], "cremad": [0, 1], "subesco_980": [0, 1]}
FT_CONDITIONAL = {("cremad", 1)}          # runs only if the timing probe allows; first truncation item
CROSSING_LEVELS = ("ravdess", "subesco_980")
PLAN_FIELDS = ["unit_id", "arm", "corpus_level", "base_corpus", "panel_draw", "model", "cell", "fold", "r", "seed_index",
               "train_seed", "config_sha256", "split_sha256", "n_fit", "n_val", "n_test", "est_gpu_sec", "cap_group",
               "conditional", "truncation_rank", "status"]


@dataclass
class Unit:
    arm: str
    corpus_level: str
    base_corpus: str
    panel_draw: str
    model: str
    cell: str
    fold: int
    r: int
    seed_index: int
    train_seed: int
    config: dict
    split_sha256: str
    n_fit: int
    n_val: int
    n_test: int
    est_gpu_sec: float
    cap_group: str
    conditional: bool = False
    truncation_rank: int = 99

    def row(self) -> dict:
        cfg_sha = sha256_json(self.config)
        uid = sha256_text(canonical_json({"arm": self.arm, "corpus_level": self.corpus_level, "panel_draw": self.panel_draw,
                                          "model": self.model, "cell": self.cell, "fold": self.fold, "r": self.r,
                                          "seed_index": self.seed_index, "config_sha256": cfg_sha,
                                          "split_sha256": self.split_sha256}))
        return {"unit_id": uid, "arm": self.arm, "corpus_level": self.corpus_level, "base_corpus": self.base_corpus,
                "panel_draw": self.panel_draw, "model": self.model, "cell": self.cell, "fold": self.fold, "r": self.r,
                "seed_index": self.seed_index, "train_seed": self.train_seed, "config_sha256": cfg_sha,
                "split_sha256": self.split_sha256, "n_fit": self.n_fit, "n_val": self.n_val, "n_test": self.n_test,
                "est_gpu_sec": round(self.est_gpu_sec, 3), "cap_group": self.cap_group,
                "conditional": int(self.conditional), "truncation_rank": self.truncation_rank, "status": "pending"}


# ------------------------------------------------------------------ cost model

def scratch_cost(n_fit: int, model: str) -> float:
    return R.SEC_PER_TRAIN_UTT * n_fit * R.MODEL_COST_MULT[model]


def model_config(model: str, overrides: dict | None = None) -> dict:
    base = {"engine": "p1_frozen", "model": model, "epochs": 100, "patience": 15, "batch_size": 64,
            "lr": 3e-4 if model == "transformer" else 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
            "config_source": "tuned_standard_experiment.candidate_configs(model)[2]"}
    if model in R.PROBE_MODELS:
        base = {"engine": "linear_probe", "model": model, "feature_state": 12, "pooling": "time_mean",
                "epochs": 100, "patience": 15, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4}
    if overrides:
        base.update(overrides)
    return base


# ------------------------------------------------------------------ split table writer

class SplitStore:
    """Split tables are stored and hashed by relative path so that the hash of a panel's
    partition does not depend on the row order of the surrounding base manifest."""

    def __init__(self, root: Path):
        self.root = root
        self.index: dict[str, dict] = {}
        self.paths: list[str] = []

    def bind(self, rows: list[dict]) -> None:
        self.paths = [r["relative_path"] for r in rows]

    def put(self, key: str, folds: list[splits.Fold], population: np.ndarray, store_fit: bool) -> str:
        P = self.paths
        payload = {"key": key, "population": [P[i] for i in population.tolist()], "folds": []}
        for f in folds:
            entry = {"fold": f.fold, "test": [P[i] for i in f.test.tolist()], "val": [P[i] for i in f.val.tolist()],
                     "meta": _json_safe(f.meta)}
            if store_fit:
                entry["fit"] = [P[i] for i in f.fit.tolist()]
            payload["folds"].append(entry)
        text = canonical_json(payload)
        sha = sha256_text(text)
        path = self.root / f"{key}.json"
        atomic_write_text(path, text)
        self.index[key] = {"sha256": sha, "path": path.relative_to(self.root.parent).as_posix(), "n_folds": len(folds)}
        return sha


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ------------------------------------------------------------------ populations

def load_populations(manifest_dir: Path | None, synthetic: bool) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Return populations and a provenance map: 'real' or 'synthetic_placeholder' per base."""
    synth = {"ravdess": lambda: corpora.synthetic_manifest("ravdess", 24, 2, 4),
             "cremad": lambda: corpora.synthetic_manifest("cremad", 91, 12, 1),
             "subesco": lambda: corpora.synthetic_manifest("subesco", 20, 10, 5)}
    out, prov, hygiene = {}, {}, {}
    for c in ("ravdess", "cremad", "subesco", "subesco_980"):
        path = (manifest_dir / f"{c}_manifest.csv") if manifest_dir else None
        if not synthetic and path is not None and path.exists():
            rows = corpora.load_manifest(path)
            base = "subesco" if c == "subesco_980" else c
            rows, hygiene[c] = corpora.apply_hygiene(base, rows)
            out[c] = rows
            prov[c] = "real"
        elif c == "subesco_980":
            continue   # derived from the subesco base when no pinned manifest exists
        else:
            out[c] = synth[c]()
            prov[c] = "synthetic_placeholder"
    if "subesco_980" not in out:
        out["subesco_980"] = None
        prov["subesco_980"] = "derived_from_subesco_base"
    prov["_hygiene"] = hygiene
    return out, prov


SUBESCO_980_PROMPTS = {"S1", "S2", "S3", "S6", "S8", "S9", "S10"}


def level_subset(level: str, arrays: dict, rows: list[dict], draw: int, pinned_980: set[str] | None) -> np.ndarray:
    """Population indices of a corpus level (panel) on the base corpus."""
    spec = R.CORPUS_LEVELS[level]
    n = len(arrays["y"])
    if spec["panel"] is None:
        return np.arange(n, dtype=np.int64)
    if level == "subesco_980":
        if pinned_980:
            return np.asarray([i for i, r in enumerate(rows) if r["relative_path"] in pinned_980], dtype=np.int64)
        mask = np.isin(arrays["sentence"], sorted(SUBESCO_980_PROMPTS))
        idx = np.where(mask)[0]
        sel = splits.panel_one_take_per_cell(arrays["cell"][idx], arrays["take"][idx], "subesco_980")
        return np.sort(idx[sel])
    if level == "sub_1400_one":
        return splits.panel_one_take_per_cell(arrays["cell"], arrays["take"], "sub_1400_one")
    if level in ("sub_700_one", "sub_700_two"):
        cells = splits.select_cells(arrays["cell"], 700, "sub_700")
        return splits.panel_one_take_per_cell(arrays["cell"], arrays["take"], "sub_700", 2 if level.endswith("two") else 1, cells)
    if level == "cremad_24":
        return splits.panel_speakers(arrays["speaker"], arrays["sex"], 24, "cremad_24", draw)
    if level == "cremad_91m":
        target = len(splits.panel_speakers(arrays["speaker"], arrays["sex"], 24, "cremad_24", draw))
        return splits.panel_utterance_matched(arrays["y"], arrays["speaker"], target, "cremad_91m", draw)
    raise ValueError(level)


# ------------------------------------------------------------------ enumeration

def build(manifest_dir: Path | None, out: Path, synthetic: bool, pinned_980_paths: Path | None = None,
          p1_split_dir: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    store = SplitStore(out / "splits")
    pops, provenance = load_populations(manifest_dir, synthetic)
    hygiene = provenance.pop("_hygiene")
    atomic_write_json(out / "hygiene_log.json", hygiene)
    if pops.get("subesco_980") is None:
        pops.pop("subesco_980")
    arrays = {c: corpora.manifest_arrays(rows) for c, rows in pops.items()}
    pinned = None
    if pinned_980_paths and pinned_980_paths.exists():
        pinned = {ln.strip() for ln in pinned_980_paths.read_text(encoding="utf-8").splitlines() if ln.strip()}
    units: list[Unit] = []
    level_info: dict[str, dict] = {}

    def base_of(level: str) -> str:
        if level == "subesco_980" and "subesco_980" in pops:
            return "subesco_980"
        return R.CORPUS_LEVELS[level]["base"]

    def subset_of(level: str, k: int) -> np.ndarray:
        base = base_of(level)
        if base == "subesco_980":
            return np.arange(len(arrays[base]["y"]), dtype=np.int64)
        return level_subset(level, arrays[base], pops[base], k, pinned)

    def level_pops(level: str):
        base = base_of(level)
        draws = R.CORPUS_LEVELS[level].get("draws")
        for k in (range(draws) if draws else [None]):
            sub = subset_of(level, k if k is not None else 0)
            tag = f"{level}" + (f"_d{k}" if k is not None else "")
            level_info[tag] = {"base": base, "provenance": provenance[base], "n_utt": int(len(sub)),
                               "n_spk": int(len(set(arrays[base]["speaker"][sub])))}
            yield tag, base, k, sub

    # ---- CTRL ------------------------------------------------------------------
    family = [l for l, s in R.CORPUS_LEVELS.items() if s["role"] == "family"]
    secondary = [l for l, s in R.CORPUS_LEVELS.items() if s["role"] == "secondary"]
    for level in family + secondary:
        is_family = level in family
        models = list(R.SCRATCH_MODELS if is_family else ("cnn", "resnet_se")) + list(R.PROBE_MODELS)
        cells = list(splits.CELLS) if is_family else ["RR", "GG"]
        reps = list(R.REPLICATES) if R.CORPUS_LEVELS[level].get("draws") is None else [0, 1]
        for tag, base, k, sub in level_pops(level):
            A = arrays[base]
            store.bind(pops[base])
            for r in reps:
                for cell in cells:
                    folds = splits.ctrl_cell_folds(cell, A["y"], A["speaker"], tag, r, subset=sub)
                    sha = store.put(f"ctrl__{tag}__{cell}__r{r}", folds, sub, store_fit=False)
                    for model in models:
                        for f in folds:
                            cost = scratch_cost(len(f.fit), model) if model in R.SCRATCH_MODELS else R.PROBE_UNIT_SEC
                            rank = 99
                            if level in ("cremad_24", "cremad_91m") and k is not None and k >= 3:
                                rank = 4
                            if level == "subesco_980" and model == "transformer" and r == 2:
                                rank = 6
                            units.append(Unit("CTRL", tag, base, "" if k is None else str(k), model, cell, f.fold, r, r,
                                              seeds.train_seed(tag, f.fold, r), model_config(model), sha, len(f.fit), len(f.val),
                                              len(f.test), cost, "CTRL", False, rank))
            # take-grouped split on subesco_full (cnn only; inner grouped by cell)
            if level == "subesco_full":
                for r in reps:
                    tg = splits.take_grouped_folds(A["y"], A["cell"], seeds.split_seed(tag + "_tg", r))
                    folds = []
                    labels = set(np.unique(A["y"]).tolist())
                    for kf, (tr, te) in enumerate(tg):
                        fit, val, att = splits.inner_grouped(tr, A["y"], A["cell"], seeds.inner_seed(tag, "TG", kf, r), labels)
                        folds.append(splits.Fold(kf, tr, te, fit, val, {"cell": "TG", "inner_attempt": att}))
                    sha = store.put(f"ctrl__{tag}__TG__r{r}", folds, sub, store_fit=False)
                    for f in folds:
                        units.append(Unit("CTRL", tag, base, "", "cnn", "TG", f.fold, r, r, seeds.train_seed(tag, f.fold, r),
                                          model_config("cnn"), sha, len(f.fit), len(f.val), len(f.test),
                                          scratch_cost(len(f.fit), "cnn"), "CTRL"))
    # crossing: off-diagonal draw x seed, cnn, 4 cells
    for level in CROSSING_LEVELS:
        base = base_of(level)
        A = arrays[base]
        sub = subset_of(level, 0)
        store.bind(pops[base])
        for r in R.REPLICATES:
            for cell in splits.CELLS:
                sha = store.index[f"ctrl__{level}__{cell}__r{r}"]["sha256"]
                folds = splits.ctrl_cell_folds(cell, A["y"], A["speaker"], level, r, subset=sub)
                for s in R.REPLICATES:
                    if s == r:
                        continue
                    for f in folds:
                        units.append(Unit("CTRL", level, base, "", "cnn", cell, f.fold, r, s,
                                          seeds.crossing_train_seed(level, f.fold, s), model_config("cnn"), sha,
                                          len(f.fit), len(f.val), len(f.test), scratch_cost(len(f.fit), "cnn"), "CTRL"))
    # LOSO pair on ravdess cnn
    A = arrays["ravdess"]
    store.bind(pops["ravdess"])
    labels = set(np.unique(A["y"]).tolist())
    for r in R.REPLICATES:
        for proto, fl in (("LOSO", splits.loso_folds(A["speaker"])), ("LOSOSUB", splits.loso_sub_folds(A["speaker"], seeds.split_seed("ravdess_lososub", r)))):
            folds = []
            for kf, (tr, te) in enumerate(fl):
                fit, val, att = splits.inner_grouped(tr, A["y"], A["speaker"], seeds.inner_seed("ravdess", proto, kf, r), labels)
                folds.append(splits.Fold(kf, tr, te, fit, val, {"cell": proto, "inner_attempt": att}))
            sha = store.put(f"ctrl__ravdess__{proto}__r{r}", folds, np.arange(len(A["y"])), store_fit=(proto == "LOSOSUB"))
            for f in folds:
                units.append(Unit("CTRL", "ravdess", "ravdess", "", "cnn", proto, f.fold, r, r, seeds.train_seed("ravdess", f.fold, r),
                                  model_config("cnn"), sha, len(f.fit), len(f.val), len(f.test), scratch_cost(len(f.fit), "cnn"), "CTRL"))
    # P1 reconciliation (real manifests + P1 split files only)
    if p1_split_dir is not None:
        for corpus in ("ravdess", "cremad"):
            csv_path = p1_split_dir / f"{corpus}_outer_test_assignments.csv"
            if not csv_path.exists():
                continue
            A = arrays[corpus]
            store.bind(pops[corpus])
            n_labels = len(corpora.CORPORA[corpus].labels)
            p1 = splits.p1_frozen_folds(csv_path, pops[corpus])
            if "random" not in p1 or "groupkfold" not in p1:
                level_info[f"p1_reconciliation_{corpus}"] = {"skipped": True, "reason": "P1 paths not found in manifest",
                                                              "missing_paths": p1.get("_missing_paths")}
                continue
            for proto, cell in (("random", "P1RR"), ("groupkfold", "P1GG")):
                for seed in (0, 1, 2):
                    folds = []
                    for kf, (tr, te) in enumerate(p1[proto]):
                        fit, val, base_seed, att = splits.p1_inner_split(corpus, proto, kf, seed, tr, A["y"], A["speaker"], n_labels)
                        folds.append(splits.Fold(kf, tr, te, fit, val, {"cell": cell, "p1_inner_base": base_seed, "inner_attempt": att,
                                                                       "p1_missing_paths": p1["_missing_paths"]}))
                    sha = store.put(f"ctrl__{corpus}__{cell}__s{seed}", folds, np.arange(len(A["y"])), store_fit=False)
                    for f in folds:
                        units.append(Unit("CTRL", corpus, corpus, "", "cnn", cell, f.fold, seed, seed, seed,
                                          model_config("cnn", {"p1_seed": seed}), sha, len(f.fit), len(f.val), len(f.test),
                                          scratch_cost(len(f.fit), "cnn"), "CTRL"))
    # ---- MECH2X2 --------------------------------------------------------------
    for level in ("cremad", "subesco_full"):
        base = R.CORPUS_LEVELS[level]["base"]
        A = arrays[base]
        store.bind(pops[base])
        conds = tuple(c for c in splits.MECH_CONDITIONS if not (level == "cremad" and c == "both_sib"))
        for r in R.MECH_REPLICATES:
            block = splits.mechanism_block(A["y"], A["speaker"], A["sentence"], A["cell"], A["take"], A["sex"], level, r,
                                           conditions=conds, include_siblings_condition=(level == "subesco_full"),
                                           rotations=R.MECH_ROTATIONS)
            for cond, folds in block.items():
                sha = store.put(f"mech__{level}__{cond}__r{r}", folds, np.arange(len(A["y"])), store_fit=True)
                for model in ("cnn", "resnet_se"):
                    for f in folds:
                        units.append(Unit("MECH2X2", level, base, "", model, cond, f.fold, r, r, seeds.train_seed(level + "_mech", f.fold, r),
                                          model_config(model), sha, len(f.fit), len(f.val), len(f.test), scratch_cost(len(f.fit), model),
                                          "MECH2X2", False, 3 if r == 1 else 99))
    # ---- FT ---------------------------------------------------------------------
    for level in ("ravdess", "cremad", "subesco_980"):
        base = base_of(level)
        for cell in ("RR", "GG"):
            sha = store.index[f"ctrl__{level}__{cell}__r0"]["sha256"]
            A = arrays[base]
            sub = subset_of(level, 0)
            folds = splits.ctrl_cell_folds(cell, A["y"], A["speaker"], level, 0, subset=sub)
            for seed in FT_SEEDS[level]:
                for model, frac in (("wavlm_base_plus_ft", 1.0), ("wavlm_base_plus_frozen_sr", R.FROZEN_SAME_REGIME_FRACTION)):
                    cfg = {"engine": "wavlm_partial_ft" if frac == 1.0 else "wavlm_frozen_same_regime", "model": model,
                           "trainable_layers": "top4+pool+head" if frac == 1.0 else "pool+head", "fp16": True, "batch_size": 16,
                           "crop_seconds": 3.0, "eval_cap_seconds": 10.0, "epochs": 15, "patience": 3,
                           "lr_encoder": 5e-5, "lr_head": 1e-3, "weight_decay": 1e-2, "seed": seed}
                    cond = (level, seed) in FT_CONDITIONAL
                    rank = 1 if cond else (5 if (seed == 1 and frac < 1.0) else 99)
                    for f in folds:
                        units.append(Unit("FT", level, base, "", model, cell, f.fold, 0, seed, seeds.crossing_train_seed(level + "_ft", f.fold, seed),
                                          cfg, sha, len(f.fit), len(f.val), len(f.test), R.FT_FOLD_MIN[level] * 60.0 * frac, "FT", cond, rank))
    # ---- HPO --------------------------------------------------------------------
    for level, rank in (("cremad", 99), ("ravdess", 2)):
        base = base_of(level)
        A = arrays[base]
        for cell in ("GR", "GG", "RR"):
            sha = store.index[f"ctrl__{level}__{cell}__r0"]["sha256"]
            folds = splits.ctrl_cell_folds(cell, A["y"], A["speaker"], level, 0)
            for ci, g in enumerate(HPO_GRID):
                cfg = model_config("resnet_se", {**g, "hpo_config_index": ci})
                for f in folds:
                    cost = scratch_cost(len(f.fit), "resnet_se") * (1.3 if g["lr"] < 1e-3 else 1.0)
                    units.append(Unit("HPO", level, base, "", "resnet_se", f"{cell}_hpo", f.fold, 0, 0, seeds.train_seed(level, f.fold, 0),
                                      cfg, sha, len(f.fit), len(f.val), len(f.test), cost, "HPO", False, rank))
    # ---- PROBECPU contract units (CPU Ridge; the 20-draw / layer sweeps are separate tools) ----
    RIDGE_MODELS = tuple(f"ridge_a1_{m}" for m in R.PROBE_MODELS)
    ridge_cfg = {"engine": "ridge_a1", "scaler": "train_fold_standard", "alpha": 1.0, "class_weight": "balanced", "feature_state": 12}
    A = arrays["ravdess"]
    store.bind(pops["ravdess"])
    for r in R.REPLICATES:
        for cell, src in (("RO", "RR"), ("GO", "GG")):
            src_folds = splits.ctrl_cell_folds(src, A["y"], A["speaker"], "ravdess", r)
            folds = [splits.Fold(f.fold, f.train, f.test, f.train, np.asarray([], dtype=np.int64), {"cell": cell, "outer_only": True}) for f in src_folds]
            sha = store.put(f"probecpu__ravdess__{cell}__r{r}", folds, np.arange(len(A["y"])), store_fit=False)
            for model in RIDGE_MODELS:
                for f in folds:
                    units.append(Unit("PROBECPU", "ravdess", "ravdess", "", model, cell, f.fold, r, r, 0, ridge_cfg, sha,
                                      len(f.fit), 0, len(f.test), R.RIDGE_UNIT_SEC, "PROBECPU"))
    A = arrays["cremad"]
    store.bind(pops["cremad"])
    g5 = splits.ctrl_cell_folds("GG", A["y"], A["speaker"], "cremad", 0)
    folds = [splits.Fold(f.fold, f.train, f.test, f.train, np.asarray([], dtype=np.int64), {"cell": "G5", "outer_only": True}) for f in g5]
    sha = store.put("probecpu__cremad__G5__r0", folds, np.arange(len(A["y"])), store_fit=False)
    for model in RIDGE_MODELS:
        for f in folds:
            units.append(Unit("PROBECPU", "cremad", "cremad", "", model, "G5", f.fold, 0, 0, 0, ridge_cfg, sha, len(f.fit), 0, len(f.test), R.RIDGE_UNIT_SEC, "PROBECPU"))
    for si in (0, 1, 2):
        ls = splits.loso_sub_folds(A["speaker"], seeds.split_seed("cremad_lososub", si))
        folds = [splits.Fold(kf, tr, te, tr, np.asarray([], dtype=np.int64), {"cell": "LOSOSUB", "outer_only": True, "subsample": si}) for kf, (tr, te) in enumerate(ls)]
        sha = store.put(f"probecpu__cremad__LOSOSUB__s{si}", folds, np.arange(len(A["y"])), store_fit=True)
        for model in RIDGE_MODELS:
            for f in folds:
                units.append(Unit("PROBECPU", "cremad", "cremad", "", model, "LOSOSUB", f.fold, 0, si, 0, ridge_cfg, sha, len(f.fit), 0, len(f.test), R.RIDGE_UNIT_SEC, "PROBECPU"))
    # ---- MECHID (inference) & PROBECPU (CPU) counted for the record ---------------
    mechid_count = 0
    for u in units:
        if u.arm == "CTRL" and u.r == 0 and u.seed_index == 0 and u.model in ("cnn", "resnet_se") and u.cell in ("RR", "GG") and u.corpus_level in family:
            mechid_count += 1
        if u.arm == "FT" and u.cell in ("RR", "GG"):
            mechid_count += 1
    ridge_levels = 7
    ridge_units = ridge_levels * len(R.PROBE_MODELS) * 13 * 2 * 20 * 5
    # ---- write ------------------------------------------------------------------
    rows = [u.row() for u in units]
    configs = {}
    for u in units:
        configs[sha256_json(u.config)] = u.config
    atomic_write_json(out / "unit_configs.json", configs)
    ids = [r_["unit_id"] for r_ in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate unit ids")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=PLAN_FIELDS, lineterminator="\n")
    w.writeheader()
    for r_ in rows:
        w.writerow(r_)
    atomic_write_text(out / "run_plan.csv", buf.getvalue())
    atomic_write_json(out / "split_index.json", store.index)
    summary = budget_summary(rows, mechid_count, ridge_units)
    summary["level_info"] = level_info
    summary["synthetic"] = synthetic
    summary["population_provenance"] = provenance
    summary["run_plan_sha256"] = sha256_text(buf.getvalue())
    atomic_write_json(out / "plan_summary.json", summary)
    return summary


def budget_summary(rows: list[dict], mechid_count: int, ridge_units: int) -> dict:
    per_arm: dict[str, dict] = {}
    for r_ in rows:
        a = per_arm.setdefault(r_["arm"], {"units": 0, "gpu_hours": 0.0, "conditional_gpu_hours": 0.0})
        a["units"] += 1
        if r_["conditional"]:
            a["conditional_gpu_hours"] += r_["est_gpu_sec"] / 3600
        else:
            a["gpu_hours"] += r_["est_gpu_sec"] / 3600
    for arm in ("PREP", "STAT"):
        per_arm[arm] = {"units": 0, "gpu_hours": R.ARMS[arm]["gpu_hours_plan"], "conditional_gpu_hours": 0.0}
    per_arm["MECHID"] = {"units": mechid_count, "gpu_hours": 0.3, "conditional_gpu_hours": 0.0}
    pc = per_arm.setdefault("PROBECPU", {"units": 0, "gpu_hours": 0.0, "conditional_gpu_hours": 0.0})
    pc["gpu_hours"] = 0.0
    pc["sweep_units_not_in_plan"] = ridge_units
    pc["cpu_hours"] = (ridge_units + pc["units"]) * R.RIDGE_UNIT_SEC / 3600
    for arm, a in per_arm.items():
        a["cap"] = R.ARMS[arm]["cap"]
        a["within_cap"] = a["gpu_hours"] + a["conditional_gpu_hours"] <= a["cap"] + 1e-9
        a["gpu_hours"] = round(a["gpu_hours"], 2)
        a["conditional_gpu_hours"] = round(a["conditional_gpu_hours"], 2)
    total = sum(a["gpu_hours"] for a in per_arm.values())
    cond = sum(a["conditional_gpu_hours"] for a in per_arm.values())
    by_rank: dict[int, float] = {}
    for r_ in rows:
        if r_["truncation_rank"] < 99:
            by_rank[r_["truncation_rank"]] = by_rank.get(r_["truncation_rank"], 0.0) + r_["est_gpu_sec"] / 3600
    return {"per_arm": per_arm, "total_gpu_hours": round(total, 2), "conditional_gpu_hours": round(cond, 2),
            "total_with_conditional": round(total + cond, 2), "program_cap": R.PROGRAM_CAP_GPU_HOURS,
            "within_program_cap": total + cond <= R.PROGRAM_CAP_GPU_HOURS,
            "truncation_savings_by_rank_gpu_hours": {str(k): round(v, 2) for k, v in sorted(by_rank.items())},
            "truncation_order": [t.strip() for t in R.TRUNCATION_ORDER], "n_units": len(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pinned-980", type=Path, default=None)
    ap.add_argument("--p1-splits", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.synthetic and a.manifests is None:
        ap.error("--manifests or --synthetic")
    s = build(a.manifests, a.out, a.synthetic, a.pinned_980, a.p1_splits)
    import json
    print(json.dumps({k: v for k, v in s.items() if k != "level_info"}, indent=1))


if __name__ == "__main__":
    main()

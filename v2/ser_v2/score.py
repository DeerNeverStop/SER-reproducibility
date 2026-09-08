"""One-shot scorer. Reads a run (plan + manifests + unit outputs), computes every registered
hypothesis, TOST, descriptor and claim verdict exactly as in SPEC_SCORING_CONTRACT.md, and
writes results.json + numeric_insert.tex. The independent verifier (verify.py) recomputes the
same quantities from the spec without importing this module.

    python -m ser_v2.score --plan PLAN --manifests MANIFESTS --run RUN --registry REGISTRY
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import corpora, stats
from .common import atomic_write_json, atomic_write_text, canonical_json, read_csv, read_json, sha256_file, sha256_text

SCRATCH_FAMILY = ("cnn", "resnet_se", "transformer")
SCRATCH_SECONDARY = ("cnn", "resnet_se")
PROBES = ("hubert_base", "wavlm_base_plus", "wav2vec2_base")
RIDGE = tuple(f"ridge_a1_{m}" for m in PROBES)
FAMILY_LEVELS = ("ravdess", "cremad", "subesco_980")
BOOT_SEED = 20260903


# ----------------------------------------------------------------------------- loading

@dataclass
class UnitOut:
    row: dict
    unit: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    speaker: np.ndarray
    paths: list[str]
    logits: np.ndarray
    pred_sha: str


@dataclass
class CellRun:
    key: str
    arm: str
    level: str
    model: str
    cell: str
    r: int
    seed: int
    units: list[UnitOut] = field(default_factory=list)
    complete: bool = False
    void_reason: str = ""
    speaker_uar: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0


class Run:
    def __init__(self, plan_dir: Path, manifests_dir: Path, run_dir: Path, registry_dir: Path):
        self.plan_dir, self.manifests_dir, self.run_dir, self.registry_dir = plan_dir, manifests_dir, run_dir, registry_dir
        self.plan = read_csv(plan_dir / "run_plan.csv")
        self.split_index = read_json(plan_dir / "split_index.json")
        self.sha_to_key = {v["sha256"]: k for k, v in self.split_index.items()}
        self.configs = read_json(plan_dir / "unit_configs.json")
        self.hygiene = read_json(plan_dir / "hygiene_log.json")
        self.registry = read_csv(registry_dir / "hypothesis_registry.csv")
        self.claim_map = read_json(registry_dir / "claim_map.json")
        self.manifest: dict[str, dict[str, dict]] = {}
        self.n_classes: dict[str, int] = {}
        for base in ("ravdess", "cremad", "subesco", "subesco_980"):
            p = manifests_dir / f"{base}_manifest.csv"
            if p.exists():
                rows, _ = corpora.apply_hygiene("subesco" if base == "subesco_980" else base, corpora.load_manifest(p))
            else:
                rows = corpora.synthetic_manifest(base, 20, 10, 5)
            self.manifest[base] = {r["relative_path"]: r for r in rows}
            self.n_classes[base] = len(corpora.CORPORA["subesco" if base == "subesco_980" else base].labels)
        self._splits: dict[str, dict] = {}
        self.integrity: dict[str, dict] = {k: {"pass": True, "failures": []} for k in ("I1", "I2", "I3", "I4", "I5", "I6", "I7")}
        self.units: dict[str, UnitOut] = {}
        self.cell_runs: dict[str, CellRun] = {}

    def split(self, sha: str) -> dict:
        key = self.sha_to_key[sha]
        if key not in self._splits:
            path = self.plan_dir / "splits" / f"{key}.json"
            if sha256_file(path) != self.split_index[key]["sha256"]:
                self.fail("I2", f"split table {key}: file sha differs from split_index (I2b)")
            self._splits[key] = read_json(path)
        return self._splits[key]

    def fail(self, check: str, msg: str) -> None:
        self.integrity[check]["pass"] = False
        if len(self.integrity[check]["failures"]) < 200:
            self.integrity[check]["failures"].append(msg)

    # ---- units ---------------------------------------------------------------
    def load_units(self) -> None:
        units_dir = self.run_dir / "units"
        for row in self.plan:
            udir = units_dir / row["unit_id"]
            if not (udir / "DONE").exists():
                continue
            unit = read_json(udir / "unit.json")
            pred_path = udir / "predictions.csv"
            sha = sha256_file(pred_path)
            done = (udir / "DONE").read_text(encoding="utf-8").strip()
            if not (done == unit.get("predictions_sha256") == sha):
                self.fail("I1", f"{row['unit_id']}: DONE/unit.json/predictions sha mismatch")
            if unit.get("split_sha256") != row["split_sha256"]:
                self.fail("I2", f"{row['unit_id']}: split sha differs from plan")
            cfg = self.configs.get(row["config_sha256"])
            if canonical_json(unit.get("config")) != canonical_json(cfg):
                self.fail("I7", f"{row['unit_id']}: unit config differs from plan")
            with open(pred_path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                K = len(header) - 5
                paths, spk, yt, yp, lg = [], [], [], [], []
                for rec in reader:
                    paths.append(rec[1]); spk.append(rec[2]); yt.append(int(rec[3])); yp.append(int(rec[4]))
                    lg.append([float(v) for v in rec[5:5 + K]])
            y_true, y_pred, logits = np.asarray(yt), np.asarray(yp), np.asarray(lg, dtype=float)
            speaker = np.asarray(spk)
            base = row["base_corpus"]
            table = self.split(row["split_sha256"])
            fold = next(f for f in table["folds"] if f["fold"] == int(row["fold"]))
            if set(paths) != set(fold["test"]):
                self.fail("I2", f"{row['unit_id']}: prediction paths != fold test set")
            if len(set(paths)) != len(paths):
                self.fail("I4", f"{row['unit_id']}: duplicate paths")
            if logits.size and not np.array_equal(np.argmax(logits, axis=1), y_pred):
                self.fail("I4", f"{row['unit_id']}: y_pred != argmax(logits)")
            man = self.manifest[base]
            missing = [p for p in table["population"] if p not in man]
            if missing:
                self.fail("I3", f"{row['unit_id']}: {len(missing)} population paths absent from the hygiene-filtered manifest (I3b)")
            for i, p in enumerate(paths):
                m = man.get(p)
                if m is None or int(m["label_index"]) != y_true[i] or m["speaker"] != speaker[i]:
                    self.fail("I3", f"{row['unit_id']}: {p} label/speaker mismatch")
                    break
            self.units[row["unit_id"]] = UnitOut(row, unit, y_true, y_pred, speaker, paths, logits, sha)

    # ---- cell-runs --------------------------------------------------------------
    def build_cell_runs(self) -> None:
        groups: dict[str, list[UnitOut]] = defaultdict(list)
        for u in self.units.values():
            r = u.row
            key = f"{r['arm']}|{r['corpus_level']}|{r['model']}|{r['cell']}|r{r['r']}|s{r['seed_index']}"
            groups[key].append(u)
        for key, us in groups.items():
            r0 = us[0].row
            cr = CellRun(key, r0["arm"], r0["corpus_level"], r0["model"], r0["cell"], int(r0["r"]), int(r0["seed_index"]))
            if cr.arm == "HPO":
                self._assemble_hpo(cr, us)
            else:
                cr.units = us
                self._assemble(cr, [(u.paths, u.y_true, u.y_pred, u.speaker) for u in us])
            self.cell_runs[key] = cr

    def _population(self, u: UnitOut) -> list[str]:
        """The table's test population: union of every fold's test list (== population for
        CTRL/FT/HPO/PROBECPU; the checkerboard test set for MECH2X2)."""
        table = self.split(u.row["split_sha256"])
        return [p for f in table["folds"] for p in f["test"]]

    def _assemble(self, cr: CellRun, parts: list) -> None:
        pop = self._population(cr.units[0]) if cr.units else []
        paths = [p for part in parts for p in part[0]]
        if len(set(paths)) != len(paths) or set(paths) != set(pop) or len(set(pop)) != len(pop):
            cr.complete, cr.void_reason = False, "OOF cover is not exactly-once over the table's test population"
            self.fail("I5", f"{cr.key}: {cr.void_reason}")
            return
        y_true = np.concatenate([p[1] for p in parts]); y_pred = np.concatenate([p[2] for p in parts])
        speaker = np.concatenate([p[3] for p in parts])
        base = cr.units[0].row["base_corpus"]
        cr.speaker_uar = per_speaker_uar(y_true, y_pred, speaker, self.n_classes[base])
        cr.complete, cr.n_rows = True, len(paths)

    def _assemble_hpo(self, cr: CellRun, us: list[UnitOut]) -> None:
        by_fold: dict[int, list[UnitOut]] = defaultdict(list)
        for u in us:
            by_fold[int(u.row["fold"])].append(u)
        n_folds = self.split_index[self.sha_to_key[us[0].row["split_sha256"]]]["n_folds"]
        cr.hpo = {"selected": {}, "val_sel": {}, "test_sel": {}, "test_max": {}}
        parts = []
        for f in range(n_folds):
            cands = by_fold.get(f, [])
            if len(cands) != 8:
                cr.complete, cr.void_reason = False, f"fold {f} has {len(cands)}/8 configs"
                self.fail("I6", f"{cr.key}: {cr.void_reason}")
                return
            cands.sort(key=lambda u: (-float(u.unit["val_uar_best"]), int(u.unit["config"]["hpo_config_index"])))
            sel = cands[0]
            K = self.n_classes[sel.row["base_corpus"]]
            cr.hpo["selected"][f] = int(sel.unit["config"]["hpo_config_index"])
            cr.hpo["val_sel"][f] = float(sel.unit["val_uar_best"])
            cr.hpo["test_sel"][f] = stats.uar(sel.y_true, sel.y_pred, K)
            cr.hpo["test_max"][f] = max(stats.uar(c.y_true, c.y_pred, K) for c in cands)
            parts.append((sel.paths, sel.y_true, sel.y_pred, sel.speaker))
            cr.units.append(sel)
        cr.fold_of_speaker = {}
        for f, part in enumerate(parts):
            for s in set(part[3].tolist()):
                cr.fold_of_speaker[s] = f
        self._assemble(cr, parts)


def per_speaker_uar(y_true, y_pred, speaker, K) -> dict[str, float]:
    out = {}
    for s in sorted(set(speaker.tolist())):
        m = speaker == s
        out[s] = stats.uar(y_true[m], y_pred[m], K)
    return out


# ----------------------------------------------------------------------------- values

class Values:
    def __init__(self, run: Run):
        self.run = run

    def cell_runs(self, arm: str, level: str, model: str, cell: str) -> list[CellRun]:
        out = []
        for cr in self.run.cell_runs.values():
            if cr.arm == arm and cr.level == level and cr.model == model and cr.cell == cell and cr.complete:
                if arm in ("CTRL", "MECH2X2", "HPO") and cr.seed != cr.r:
                    continue
                out.append(cr)
        return out

    def V(self, arm: str, level: str, model: str, cell: str, seeds: set[int] | None = None) -> dict[str, float] | None:
        crs = self.cell_runs(arm, level, model, cell)
        if seeds is not None:
            crs = [c for c in crs if c.seed in seeds]
        if not crs:
            return None
        acc: dict[str, list[float]] = defaultdict(list)
        for cr in crs:
            for s, v in cr.speaker_uar.items():
                acc[s].append(v)
        return {s: float(np.mean(v)) for s, v in acc.items()}

    def group(self, arm: str, level: str, models: tuple, cell: str, seeds: set[int] | None = None) -> dict[str, float] | None:
        vals = [self.V(arm, level, m, cell, seeds) for m in models]
        if any(v is None for v in vals):
            return None
        speakers = set.intersection(*[set(v) for v in vals])
        return {s: float(np.mean([v[s] for v in vals])) for s in speakers}


def diff(a: dict | None, b: dict | None) -> dict[str, float] | None:
    if a is None or b is None:
        return None
    return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}


def scratch_models(level: str) -> tuple:
    return SCRATCH_FAMILY if level in FAMILY_LEVELS else SCRATCH_SECONDARY


# ----------------------------------------------------------------------------- statistics on a contrast

def summarize(d: dict[str, float] | None, margin: float | None = None) -> dict:
    if d is None or len(d) == 0:
        return {"n": 0, "tested": False, "n_nonzero": 0, "mean": None, "median": None, "sd": None, "ci_low": None,
                "ci_high": None, "wilcoxon_p": 1.0, "sign_test_p": None, "speakers": [], "d": []}
    speakers = sorted(d)
    arr = np.asarray([d[s] for s in speakers], dtype=float)
    t = stats.paired_test(arr)
    lo, hi = stats.speaker_bootstrap_ci(arr, 10000, BOOT_SEED)
    out = {"tested": True, "n": int(t["n"]), "n_nonzero": int(t["n_nonzero"]), "mean": t["mean"], "median": t["median"],
           "sd": t["sd"], "ci_low": lo, "ci_high": hi, "wilcoxon_p": t["wilcoxon_p"],
           "sign_test_p": t.get("sign_test_p") if t["sign_test_used"] else None, "speakers": speakers, "d": arr.tolist()}
    if margin is not None:
        t2 = stats.tost_paired(arr, margin)
        out.update({"p_low": t2["p_low"], "p_high": t2["p_high"], "p": t2["p"], "ci90": t2["ci90"], "se": t2["se"]})
    return out


# ----------------------------------------------------------------------------- scorer

class Scorer:
    def __init__(self, run: Run):
        self.run = run
        self.v = Values(run)

    # -- contrasts -----------------------------------------------------------
    def pipe(self, level, models, a="RR", b="GG", arm="CTRL"):
        return diff(self.v.group(arm, level, models, a), self.v.group(arm, level, models, b))

    def n10(self):
        run = self.run
        p24: dict[str, list[float]] = defaultdict(list)
        p91: dict[str, list[float]] = defaultdict(list)
        draws = sorted({cr.level for cr in run.cell_runs.values() if cr.level.startswith("cremad_24_d")})
        for lv in draws:
            d = self.pipe(lv, SCRATCH_SECONDARY)
            if d is None:
                continue
            for s, val in d.items():
                p24[s].append(val)
        for lv in sorted({cr.level for cr in run.cell_runs.values() if cr.level.startswith("cremad_91m_d")}):
            d = self.pipe(lv, SCRATCH_SECONDARY)
            if d is None:
                continue
            for s, val in d.items():
                p91[s].append(val)
        if not p24 or not p91:
            return None, {}
        out = {s: float(np.mean(p24[s])) - float(np.mean(p91[s])) for s in sorted(p24) if s in p91}
        return out, {"n_draws_24": len(draws), "n_speakers_in_any_draw": len(p24)}

    def reexposure(self, level):
        sets = [{cr.seed for cr in self.v.cell_runs("FT", level, m, c)} for m in ("wavlm_base_plus_ft", "wavlm_base_plus_frozen_sr") for c in ("RR", "GG")]
        common = set.intersection(*sets) if sets else set()
        if not common:
            return None
        a = diff(self.v.V("FT", level, "wavlm_base_plus_ft", "RR", common), self.v.V("FT", level, "wavlm_base_plus_ft", "GG", common))
        b = diff(self.v.V("FT", level, "wavlm_base_plus_frozen_sr", "RR", common), self.v.V("FT", level, "wavlm_base_plus_frozen_sr", "GG", common))
        return diff(a, b)

    def ft_pipe(self, level):
        return diff(self.v.V("FT", level, "wavlm_base_plus_ft", "RR"), self.v.V("FT", level, "wavlm_base_plus_ft", "GG"))

    def hpo_cell(self, cell) -> CellRun | None:
        crs = self.v.cell_runs("HPO", "cremad", "resnet_se", cell)
        return crs[0] if crs else None

    def n14(self):
        gr, gg = self.hpo_cell("GR_hpo"), self.hpo_cell("GG_hpo")
        if gr is None or gg is None:
            return None
        out = {}
        for s in sorted(set(gr.speaker_uar) & set(gg.speaker_uar)):
            opt_gr = gr.hpo["val_sel"][gr.fold_of_speaker[s]] - gr.speaker_uar[s]
            opt_gg = gg.hpo["val_sel"][gg.fold_of_speaker[s]] - gg.speaker_uar[s]
            out[s] = opt_gr - opt_gg
        return out

    def n15(self):
        ridge = [diff(self.v.V("PROBECPU", "ravdess", m, "RO"), self.v.V("PROBECPU", "ravdess", m, "GO")) for m in RIDGE]
        probe = [diff(self.v.V("CTRL", "ravdess", m, "RG"), self.v.V("CTRL", "ravdess", m, "GG")) for m in PROBES]
        if any(x is None for x in ridge + probe):
            return None
        speakers = set.intersection(*[set(x) for x in ridge + probe])
        return {s: float(np.mean([x[s] for x in ridge])) - float(np.mean([x[s] for x in probe])) for s in sorted(speakers)}

    def t01(self):
        ds = [diff(self.v.V("PROBECPU", "cremad", m, "LOSOSUB"), self.v.V("PROBECPU", "cremad", m, "G5")) for m in RIDGE]
        if any(x is None for x in ds):
            return None
        speakers = set.intersection(*[set(x) for x in ds])
        return {s: float(np.mean([x[s] for x in ds])) for s in sorted(speakers)}

    def mech(self, level, a, b):
        return self.pipe(level, SCRATCH_SECONDARY, a, b, arm="MECH2X2")

    def mech_half(self, level, exposed: bool):
        """Per speaker: UAR under the half rotation where the speaker is (un)exposed minus none.
        The exposed half is read per replicate r from that replicate's spk_half_h1 table
        (halves are hashed per replicate), averaged over the replicates where all three cell-runs
        (none, h1, h2) are complete for every model of the scratch group."""
        per_r: dict[int, dict[str, float]] = {}
        reps = sorted({cr.r for cr in self.run.cell_runs.values() if cr.arm == "MECH2X2" and cr.level == level})
        for r in reps:
            vals: list[dict[str, float]] = []
            ok = True
            for m in SCRATCH_SECONDARY:
                none = self.v.V("MECH2X2", level, m, "none", {r}); h1 = self.v.V("MECH2X2", level, m, "spk_half_h1", {r}); h2 = self.v.V("MECH2X2", level, m, "spk_half_h2", {r})
                crs = [c for c in self.v.cell_runs("MECH2X2", level, m, "spk_half_h1") if c.r == r]
                if none is None or h1 is None or h2 is None or not crs:
                    ok = False
                    break
                exposed_h1: set[str] = set()
                for f in self.run.split(crs[0].units[0].row["split_sha256"])["folds"]:
                    exposed_h1.update(f["meta"].get("exposed_speakers", []))
                d = {}
                for s_ in sorted(set(none) & set(h1) & set(h2)):
                    in_h1 = s_ in exposed_h1
                    pick = (h1 if in_h1 else h2) if exposed else (h2 if in_h1 else h1)
                    d[s_] = pick[s_] - none[s_]
                vals.append(d)
            if not ok:
                continue
            speakers = set.intersection(*[set(v) for v in vals])
            per_r[r] = {s_: float(np.mean([v[s_] for v in vals])) for s_ in speakers}
        if not per_r:
            return None
        speakers = set.intersection(*[set(v) for v in per_r.values()])
        return {s_: float(np.mean([per_r[r][s_] for r in sorted(per_r)])) for s_ in sorted(speakers)}

    # -- assembly ------------------------------------------------------------
    def contrasts(self) -> dict[str, dict | None]:
        c = {}
        c["N01"] = self.pipe("subesco_980", SCRATCH_FAMILY)
        c["N02"] = self.pipe("subesco_980", PROBES)
        c["N03"] = self.pipe("subesco_980", SCRATCH_FAMILY, "RG", "GG")
        c["N04"] = self.pipe("subesco_980", SCRATCH_FAMILY, "RR", "RG")
        c["N05"] = self.mech("cremad", "prm", "none")
        c["N06"] = self.mech("cremad", "spk", "none")
        c["N07"] = self.mech("subesco_full", "prm", "none")
        c["N08"] = self.mech("subesco_full", "spk", "none")
        c["N09"] = self.mech("subesco_full", "both_sib", "both")
        c["N10"], self.n10_meta = self.n10()
        c["N11"], c["N12"], c["N13"] = self.reexposure("ravdess"), self.reexposure("cremad"), self.reexposure("subesco_980")
        c["N14"] = self.n14()
        c["N15"] = self.n15()
        c["R01"], c["R02"], c["R03"] = self.pipe("ravdess", SCRATCH_FAMILY), self.pipe("ravdess", SCRATCH_FAMILY, "RG", "GG"), self.pipe("ravdess", SCRATCH_FAMILY, "RR", "RG")
        c["R04"], c["R05"], c["R06"] = self.pipe("cremad", SCRATCH_FAMILY), self.pipe("cremad", SCRATCH_FAMILY, "RG", "GG"), self.pipe("cremad", SCRATCH_FAMILY, "RR", "RG")
        c["R07"], c["R08"] = self.pipe("ravdess", PROBES), self.pipe("cremad", PROBES)
        c["R09"], c["R10"], c["R11"] = self.ft_pipe("ravdess"), self.ft_pipe("cremad"), self.ft_pipe("subesco_980")
        c["T01"] = self.t01()
        c["T02"] = self.mech_half("cremad", exposed=False)
        return c

    def descriptors(self) -> dict:
        D = {}
        D["D01"] = {lv: {"scratch": summarize(self.pipe(lv, scratch_models(lv), "GR", "GG")), "probe": summarize(self.pipe(lv, PROBES, "GR", "GG"))}
                    for lv in FAMILY_LEVELS}
        d02 = {}
        for lv in FAMILY_LEVELS:
            entry = {}
            for grp, models in (("scratch", scratch_models(lv)), ("probe", PROBES)):
                signs = {}
                for m in models:
                    d = diff(self.v.V("CTRL", lv, m, "RR"), self.v.V("CTRL", lv, m, "GG"))
                    if d is not None:
                        signs[m] = float(np.mean([d[s] for s in sorted(d)]))
                entry[grp] = {"means": signs, "positive_models": sum(1 for v in signs.values() if v > 0), "total": len(models)}
            d02[lv] = entry
        D["D02"] = d02
        D["D03"] = {}
        base = self.pipe("ravdess", SCRATCH_FAMILY)
        for other in ("cremad", "subesco_980"):
            o = self.pipe(other, SCRATCH_FAMILY)
            if base is None or o is None:
                D["D03"][f"ravdess_minus_{other}"] = {"tested": False}
                continue
            a = np.asarray([base[s] for s in sorted(base)]); b = np.asarray([o[s] for s in sorted(o)])
            rng = np.random.RandomState(BOOT_SEED)
            ia = rng.randint(0, len(a), size=(10000, len(a))); ib = rng.randint(0, len(b), size=(10000, len(b)))
            dm = a[ia].mean(axis=1) - b[ib].mean(axis=1)
            lo, hi = np.percentile(dm, [2.5, 97.5], method="linear")
            D["D03"][f"ravdess_minus_{other}"] = {"tested": True, "diff": float(a.mean() - b.mean()), "ci_low": float(lo), "ci_high": float(hi),
                                                  "band": 3.0, "label": "larger" if lo > 3.0 else ("smaller" if hi < -3.0 else "within_band")}
        D["D04"] = {"sub_1400_one_minus_sub_700_one": summarize(diff(self.pipe("sub_1400_one", SCRATCH_SECONDARY), self.pipe("sub_700_one", SCRATCH_SECONDARY))),
                    "sub_700_two_minus_sub_700_one": summarize(diff(self.pipe("sub_700_two", SCRATCH_SECONDARY), self.pipe("sub_700_one", SCRATCH_SECONDARY)))}
        D["D05"] = summarize(diff(self.v.V("CTRL", "subesco_full", "cnn", "RR"), self.v.V("CTRL", "subesco_full", "cnn", "TG")))
        D["D06"] = {}
        for lv in ("cremad", "subesco_full"):
            both = self.mech(lv, "both", "none"); spk = self.mech(lv, "spk", "none"); prm = self.mech(lv, "prm", "none")
            inter = None
            if both is not None and spk is not None and prm is not None:
                inter = {s: both[s] - spk[s] - prm[s] for s in sorted(set(both) & set(spk) & set(prm))}
            g1 = spk
            g05 = self.mech_half(lv, exposed=True)
            crowd = diff(g1, g05)
            share = None
            if g1 and g05 and np.mean(list(g1.values())) > 0:
                share = float(np.mean([g05[s] for s in g05]) / np.mean([g1[s] for s in g1]))
            D["D06"][lv] = {"both_minus_none": summarize(both), "interaction": summarize(inter), "G1": summarize(g1),
                            "G05": summarize(g05), "crowding_G1_minus_G05": summarize(crowd), "share_G05_over_G1": share}
        d07 = {"cremad_24": {}, "cremad_91m": {}}
        for cr in self.run.cell_runs.values():
            for fam in ("cremad_24", "cremad_91m"):
                if cr.level.startswith(fam + "_d") and cr.level not in d07[fam]:
                    d = self.pipe(cr.level, SCRATCH_SECONDARY)
                    if d is not None:
                        d07[fam][cr.level] = float(np.mean([d[s] for s in sorted(d)]))
        for fam in d07:
            vals = list(d07[fam].values())
            d07[fam]["_sd_over_draws"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        D["D07"] = d07
        rr, gg, gr = self.hpo_cell("RR_hpo"), self.hpo_cell("GG_hpo"), self.hpo_cell("GR_hpo")
        d08 = {"rr_minus_gg": summarize(diff(rr.speaker_uar, gg.speaker_uar) if rr and gg else None)}
        for name, cr in (("RR_hpo", rr), ("GG_hpo", gg), ("GR_hpo", gr)):
            if cr is not None:
                d08[name] = {"selected": {str(k): v for k, v in cr.hpo["selected"].items()},
                             "test_selection_optimism_mean": float(np.mean([cr.hpo["test_max"][f] - cr.hpo["test_sel"][f] for f in cr.hpo["test_sel"]]))}
        D["D08"] = d08
        if gr and gg:
            fixed = diff(self.v.V("CTRL", "cremad", "resnet_se", "GR", {0}), self.v.V("CTRL", "cremad", "resnet_se", "GG", {0}))
            hpo = diff(gr.speaker_uar, gg.speaker_uar)
            D["D09"] = summarize(diff(hpo, fixed))
        else:
            D["D09"] = {"tested": False}
        D["D14"] = {}
        for lv in ("ravdess", "subesco_980"):
            M = np.full((3, 3), np.nan)
            for r in range(3):
                for s in range(3):
                    a = b = None
                    for cr in self.run.cell_runs.values():
                        if cr.arm == "CTRL" and cr.level == lv and cr.model == "cnn" and cr.r == r and cr.seed == s and cr.complete:
                            if cr.cell == "RR":
                                a = cr.speaker_uar
                            elif cr.cell == "GG":
                                b = cr.speaker_uar
                    if a is not None and b is not None:
                        M[r, s] = float(np.mean([a[k] - b[k] for k in sorted(set(a) & set(b))]))
            if np.isnan(M).any():
                D["D14"][lv] = {"tested": False}
            else:
                rows_m, cols_m = M.mean(axis=1), M.mean(axis=0)
                D["D14"][lv] = {"tested": True, "M": M.tolist(), "row_means": rows_m.tolist(), "col_means": cols_m.tolist(),
                                "grand_mean": float(M.mean()), "sd_draw": float(np.std(rows_m, ddof=1)), "sd_seed": float(np.std(cols_m, ddof=1))}
        return D

    def run_all(self) -> dict:
        run = self.run
        run.load_units()
        run.build_cell_runs()
        contrasts = self.contrasts()
        reg = {h["hypothesis_id"]: h for h in run.registry}
        hyps: dict[str, dict] = {}
        tost: dict[str, dict] = {}
        for hid, h in reg.items():
            fam = h["family"]
            d = contrasts.get(hid)
            status = h.get("a_priori_status", "confirmatory")
            if fam == "T":
                margin = float(h["sesoi_or_margin"].replace("+/-", "").replace("pp", "").strip())
                s = summarize(d, margin=margin)
                if not s["tested"]:
                    s.update({"p": 1.0, "p_low": None, "p_high": None, "ci90": None, "se": None})
                s.update({"family": fam, "m": int(h["family_size"] or 0), "margin": margin, "a_priori_status": status})
                tost[hid] = s
            else:
                s = summarize(d)
                s.update({"family": fam, "m": int(h["family_size"] or 0), "direction": h["direction"], "a_priori_status": status})
                hyps[hid] = s
        families = sorted({s["family"] for s in hyps.values()})
        for fam in families:
            members = {hid: (s["wilcoxon_p"] if s["tested"] else 1.0) for hid, s in hyps.items()
                       if s["family"] == fam and s["a_priori_status"] == "confirmatory"}
            for hid, res in stats.holm(members).items():
                s = hyps[hid]
                s["p_holm"], s["reject"] = res["p_holm"], res["reject"]
                if not s["tested"]:
                    s["verdict"] = "not tested"
                elif s["direction"] == ">0":
                    s["verdict"] = "supported" if (s["reject"] and s["mean"] > 0) else "not supported"
                else:
                    s["verdict"] = "supported" if s["reject"] else "not supported"
        for hid, s in hyps.items():
            if s["a_priori_status"] != "confirmatory":
                s["p_holm"], s["reject"] = None, False
                s["verdict"] = "estimate (not tested)" if s["tested"] else "not tested"
        tmembers = {hid: (s["p"] if s["tested"] else 1.0) for hid, s in tost.items() if s["a_priori_status"] == "confirmatory"}
        for hid, res in stats.holm(tmembers).items():
            s = tost[hid]
            s["p_holm"], s["reject"] = res["p_holm"], res["reject"]
            s["equivalent"] = bool(s["tested"] and res["reject"])
            s["verdict"] = "not tested" if not s["tested"] else ("supported" if s["equivalent"] else "not supported")
        for hid, s in tost.items():
            if s["a_priori_status"] != "confirmatory":
                s["p_holm"], s["reject"], s["equivalent"] = None, False, False
                s["verdict"] = "estimate (not tested)" if s["tested"] else "not tested"
        dev = self.deviation_index()
        for hid, s in {**hyps, **tost}.items():
            if s["verdict"] == "not tested" and hid in dev:
                s["verdict"] = f"not tested (truncated, deviation entry {dev[hid]})"
        claims = self.claims({**hyps, **tost}, dev)
        descriptors = self.descriptors()
        results = {"schema": "ser-v2-results-1", "run_id": run.run_dir.name,
                   "plan_sha256": sha256_file(run.plan_dir / "run_plan.csv"),
                   "registry_sha256": sha256_file(run.registry_dir / "hypothesis_registry.csv"),
                   "n_units_done": len(run.units), "n_units_void": sum(len(c.units) for c in run.cell_runs.values() if not c.complete),
                   "cell_runs": {k: {"complete": c.complete, "void_reason": c.void_reason, "n_folds": len(c.units), "n_rows": c.n_rows,
                                     "speaker_uar": c.speaker_uar} for k, c in run.cell_runs.items()},
                   "hypotheses": hyps, "tost": tost, "claims": claims, "descriptors": descriptors,
                   "n10_meta": getattr(self, "n10_meta", {}), "integrity": run.integrity}
        return results

    def deviation_index(self) -> dict[str, int]:
        """hypothesis id -> deviation entry number, from runs/<run_id>/deviations.jsonl."""
        path = self.run.run_dir / "deviations.jsonl"
        out: dict[str, int] = {}
        if not path.exists():
            return out
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = int(e.get("entry", i))
            for hid in e.get("hypotheses", []):
                out.setdefault(hid, n)
        return out

    def claims(self, results: dict[str, dict], dev: dict[str, int] | None = None) -> dict:
        out = {}
        dev = dev or {}
        base_verdict = lambda hid: results.get(hid, {}).get("verdict", "not tested").split(" (")[0]
        mean_of = lambda hid: results.get(hid, {}).get("mean")

        def not_tested_label(ids):
            entries = sorted({dev[h] for h in ids if h in dev and base_verdict(h) == "not tested"})
            return f"not tested (truncated, deviation entry {entries[0]})" if entries else "not tested"

        for cid, c in self.run.claim_map["claims"].items():
            ev = {}
            if c.get("type") == "estimation":
                for h in c.get("reported", []):
                    ev[h] = results.get(h, {}).get("verdict", "not tested")
                out[cid] = {"verdict": "reported (estimation claim)", "rule": c["verdict_rule"], "evidence": ev}
                continue
            if "requires_k_of" in c:
                k, ids = c["requires_k_of"]
                ev = {h: base_verdict(h) for h in ids}
                n_sup = sum(1 for v in ev.values() if v == "supported")
                n_nt = sum(1 for v in ev.values() if v == "not tested")
                verdict = "supported" if n_sup >= k else (not_tested_label(ids) if n_sup + n_nt >= k else "not supported")
            elif "requires_any" in c:
                ids = c["requires_any"]
                ev = {h: base_verdict(h) for h in ids}
                opposite = any(v == "not supported" and (mean_of(h) or 0) < 0 for h, v in ev.items())
                if any(v == "supported" for v in ev.values()) and not opposite:
                    verdict = "supported"
                elif any(v == "not supported" for v in ev.values()):
                    verdict = "not supported"
                else:
                    verdict = not_tested_label(ids)
            else:
                ids = c["requires"]
                ev = {h: base_verdict(h) for h in ids}
                if all(v == "supported" for v in ev.values()):
                    verdict = "supported"
                elif any(v == "not supported" for v in ev.values()):
                    verdict = "not supported"
                else:
                    verdict = not_tested_label(ids)
            for h in c.get("reported", []):
                ev[h] = results.get(h, {}).get("verdict", "not tested")
            out[cid] = {"verdict": verdict, "rule": c["verdict_rule"], "evidence": ev}
        return out


# ----------------------------------------------------------------------------- tex insert

def fmt_p(p: float | None) -> str:
    if p is None:
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def numeric_insert(results: dict) -> str:
    lines = ["% generated by ser_v2.score from results.json; do not edit by hand"]
    for hid, s in {**results["hypotheses"], **results["tost"]}.items():
        macro = "H" + hid
        if not s.get("tested"):
            lines.append(f"\\newcommand{{\\{macro}Verdict}}{{not tested}}")
            continue
        lines.append(f"\\newcommand{{\\{macro}Mean}}{{{s['mean']:.2f}}}")
        lines.append(f"\\newcommand{{\\{macro}CILow}}{{{s['ci_low']:.2f}}}")
        lines.append(f"\\newcommand{{\\{macro}CIHigh}}{{{s['ci_high']:.2f}}}")
        lines.append(f"\\newcommand{{\\{macro}N}}{{{s['n']}}}")
        p = s.get("wilcoxon_p", s.get("tost_p"))
        lines.append(f"\\newcommand{{\\{macro}P}}{{{fmt_p(p)}}}")
        lines.append(f"\\newcommand{{\\{macro}PHolm}}{{{fmt_p(s.get('p_holm'))}}}")
        lines.append(f"\\newcommand{{\\{macro}Verdict}}{{{s['verdict']}}}")
    for cid, c in results["claims"].items():
        lines.append(f"\\newcommand{{\\Claim{cid.replace('-', '')}}}{{{c['verdict']}}}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    a = ap.parse_args(argv)
    run = Run(a.plan, a.manifests, a.run, a.registry)
    results = Scorer(run).run_all()
    atomic_write_text(a.run / "results.json", json.dumps(results, indent=1, sort_keys=True) + "\n")
    atomic_write_text(a.run / "numeric_insert.tex", numeric_insert(results))
    summary = {hid: (s["verdict"], round(s["mean"], 3) if s.get("tested") else None) for hid, s in {**results["hypotheses"], **results["tost"]}.items()}
    print(json.dumps({"n_units_done": results["n_units_done"], "n_void": results["n_units_void"],
                      "integrity": {k: v["pass"] for k, v in results["integrity"].items()}, "hypotheses": summary,
                      "claims": {k: v["verdict"] for k, v in results["claims"].items()}}, indent=1))


if __name__ == "__main__":
    main()

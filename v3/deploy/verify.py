"""Independent verifier for SER26-DEPLOY-1 (v3/deploy). Recomputes every predefined quantity of modules A, B1 and B2
from the raw prediction files, using only SPEC_DEPLOY_ZH.md as the definition, and compares them with the scorer
outputs in results/<module>/endpoints.json and per_speaker.csv.

Independence: this module does not import enroll.py / review_budget.py / calib.py / calib_gpu.py / engines_deploy.py.
From common.py only I/O helpers are used (read_json, sha256_file, digest, cpu_guard). All statistics (UAR, per-speaker
aggregation, quantile thresholds, coverage/error, capture rate, AURC, speaker bootstrap) are re-implemented here with
numpy only.

CLI
  python -m v3.deploy.verify --module A|B1|B2|all --plan-dir v3/deploy/work/plan --run-dir v3/deploy/work \
      --results-dir v3/deploy/results --v2-data $SER_V2_DATA_ROOT --out v3/deploy/results/verification.json [--tolerance 1e-9]

Fail-closed policy: any integrity failure (missing unit, DONE/sha mismatch, wrong row set, manifest disagreement)
sets pass=false; the module's endpoints are then NOT recomputed (no partial statistics) and the reasons are written.
Any endpoint present on one side only, any |difference| > tolerance in estimate or CI bound, and any per-speaker row
missing on one side also sets pass=false.

PLAN SCHEMA assumed (A.json / B2.json); the reader tolerates the listed synonyms and fails closed otherwise.
  common      : {"program", "module", "units": [unit, ...]}
  unit (A)    : {"unit_id", "level", "enc"|"model", "r", "fold", "split_key", "split_sha256", ["identity": {...}]}
  unit (B2)   : {"unit_id", "level", "model", "cell", "r", "fold", "split_key", "split_sha256", ["identity": {...}]}
  A per level : plan["levels"][level]["per_speaker"][spk] = {"Q": [paths], "E_by_class": {cls: [paths]}, ...}  (plan.py layout)
                or {"query"|"Q": {spk: [paths]}, "enrollment"|"E": {spk: [paths]}} (fixture layout);
                plan["levels"][level]["conditions"] = [{"id"|"condition": "single:neutral@3", "composition": "single:neutral"|"single",
                "N": int|0|null, "feasible": bool, "estimators": [...], "draws": int}, ...]
  A unit_id   : recomputed as sha256(canonical({f: unit[f] for f in plan["config"]["unit_id_fields"]})) when that list exists
                (plan.py: program, module, level, enc, r, fold, split_key, split_sha256, feature_sha256, state, ridge, shrink_k).
  A model.json: identity read from model.json["unit"] (fallback: top level); model.json["predictions_sha256"] must equal DONE.
  B2 unit.json: identity fields at top level; "val_sha256"/"test_sha256" must equal the DONE line; status must be "done".
  condition names: "none", "balanced@N", "single:<e>@N", "other_balanced@N", "other_single:<e>@N", "oracle_all".
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import read_json, sha256_file, digest, cpu_guard

PROGRAM = "SER26-DEPLOY-2"
LEGACY_PROGRAM = "SER26-DEPLOY-1"
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPS = 10000
B_GRID = (0.1, 0.2, 0.3)
B2_BUDGET = 0.2
B2_TARGET_RISK_FACTOR = 0.5
HARM_POINTS = 2.0
COVERAGE_FLOOR = 0.70
LEVEL_BASE = {"ravdess": "ravdess", "cremad": "cremad", "subesco_980": "subesco"}
QUERY_RULE = {"cremad": ("hash", 5, ("SER26-DEPLOY-1:cremad:sent",)),
              "subesco_980": ("hash", 3, ("SER26-DEPLOY-1:subesco_980:sent", "SER26-DEPLOY-1:subesco:sent")),
              "ravdess": ("sentence", None, ("S1",))}
ESTIMATORS_ALL = ("naive", "shrink", "neutral_filter", "prior_corrected")
RECOVERY_ESTIMATORS = ("shrink", "neutral_filter", "prior_corrected")
B1_MODELS_CTRL = ("cnn", "resnet_se", "transformer", "hubert_base", "wavlm_base_plus", "wav2vec2_base")
B1_MODEL_FT = "wavlm_base_plus_ft"
NON_IDENT = {"id", "estimate", "ci95", "n", "note", "notes", "se", "ci_kind", "n_reps", "scored_at", "spec", "unit", "n_units", "n_groups"}
REPO_ROOT = Path(__file__).resolve().parents[2]

AMBIGUITIES = [
    "A/UAR: 'fixed class set' implemented as mean recall over classes PRESENT in the speaker's Q rows (v2 convention in "
    "common.uar docstring); on the real Q sets every speaker has all classes so the choice is moot there; any absent class "
    "is recorded as a note.",
    "A/none: E0 'none' is condition-independent; expected rows follow the plan condition table (default: condition 'none', "
    "estimator 'none', 1 draw). 'oracle_all' defaults to estimator 'naive', 1 draw. Other conditions default to 5 draws and "
    "estimators naive/shrink/neutral_filter/prior_corrected (other_*: naive only).",
    "A/item 3 'single - balanced at the same N': computed only for N feasible for BOTH single:* and balanced (CREMA-D N=5).",
    "A/items 2,4,7 use N = largest feasible single N of the level; item 5 is emitted for every feasible single N; item 6 for "
    "every N where both other_* and the base condition are feasible.",
    "A/item 4 recall: per reference emotion e (condition single:e@N, quantity recall) and the mean over e (single:*@N).",
    "A/item 7 harm share: indicator[mean_e UAR(single:e@N,naive) - UAR(none) < -2]; per-e variants also emitted; CI = "
    "speaker bootstrap of the indicator mean.",
    "A/averaging: per speaker mean over draws, then over r, then speaker-equal mean; other_* conditions use estimator naive.",
    "B1/threshold: tau_b = np.quantile(conf, b, linear); accepted iff conf >= tau_b.",
    "B1/rates: accepted error, capture rate and quota-variant accepted error are utterance-pooled over the group's OOF pool; "
    "coverage p10 = np.percentile(cov, 10, linear); 'coverage < 70%' means cov < 0.70.",
    "B1/aggregation: flat mean over all (r, seed_index) groups of a (level, model); bootstrap keeps tau fixed, resamples "
    "speakers, recomputes the pooled ratio/count per group and averages over groups; CI = percentile 2.5/97.5.",
    "B1/AURC: sort by confidence descending (ties by file order), AURC = mean_k risk(k); oracle = mean_k max(0,k-n_correct)/k; "
    "no CI (ci95 = [nan, nan]).",
    "B1/per_speaker.csv: coverage/accepted_error = per-speaker values averaged nan-aware over groups.",
    "B2/threshold: tau = np.quantile(conf_val, 0.20, linear), accepted iff conf >= tau on val and on test.",
    "B2/secondary rule: r* = 0.5 x val error; k = smallest number of lowest-confidence val rows whose rejection gives val "
    "accepted error <= r*; tau2 = k-th lowest val confidence (k>0) else -inf; accepted iff conf >= tau2.",
    "B2/pairing: GR - GG per test speaker within (level, model, r, fold) for coverage_s, accepted_error_s (NaN if none "
    "accepted) and error gap_s = accepted_error_s - unit r_hat; nan-aware mean over r, then speaker-equal mean + bootstrap. "
    "Coverage-gap difference equals the coverage difference and is not emitted separately.",
    "B2/absolute per (level, model, cell): speaker-equal realised coverage/error, unit-mean promised error, pooled realised "
    "values, gaps, secondary-rule results; B1-style coverage statistics per r then averaged.",
    "Unit identity: consistency between plan, directory and model.json/unit.json is checked; the unit_id hash is recomputed "
    "only when the plan carries an explicit 'identity' object (SPEC 5.2 leaves the key set open).",
    "Comparison: endpoints matched on every identifying field (all keys except id/estimate/ci95/n); NaN equals NaN; CI bounds "
    "of endpoints for which this verifier emits no CI are reported as 'ci_unverified' notes, not failures.",
]


class VerificationError(ValueError):
    pass


def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _first(d: dict, *keys, default=None, required=False, where=""):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    if required:
        raise VerificationError(f"plan schema: none of {keys} found{(' in ' + where) if where else ''}")
    return default


def _ranked(items, salt):
    return sorted(items, key=lambda it: (digest([salt, it]), it))


# ----------------------------------------------------------------------------- report

@dataclass
class Report:
    failures: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def fail(self, module, check, detail, unit_id=None):
        self.failures.append({"module": module, "unit_id": unit_id, "check": check, "detail": str(detail)})

    def note(self, module, text):
        rec = {"module": module, "note": str(text)}
        if rec not in self.notes:
            self.notes.append(rec)

    def failed(self, module=None):
        return any(f for f in self.failures if module is None or f["module"] == module)


# ----------------------------------------------------------------------------- sources (manifests / splits / v2 plan)

@dataclass
class Sources:
    manifests_dir: Path
    split_index: Path
    v2_data_root: Path
    run_plan: Path
    level_base: dict

    @staticmethod
    def real(v2_data_root: Path) -> "Sources":
        return Sources(REPO_ROOT / "v2" / "manifests", REPO_ROOT / "v2" / "plan_rc2" / "split_index.json",
                       Path(v2_data_root), REPO_ROOT / "v2" / "plan_rc2" / "run_plan.csv", dict(LEVEL_BASE))


class Manifest:
    """relative_path -> (speaker, label_index, sentence, label); classes fixed by label_index order."""

    def __init__(self, path: Path):
        self.rows = {}
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                self.rows[r["relative_path"]] = (r["speaker"], int(r["label_index"]), r.get("sentence", ""), r["label"])
        if not self.rows:
            raise VerificationError(f"empty manifest {path}")
        pairs = sorted({(v[1], v[3]) for v in self.rows.values()})
        if [p[0] for p in pairs] != list(range(len(pairs))):
            raise VerificationError(f"label_index not contiguous in {path}")
        self.classes = [p[1] for p in pairs]
        self.n_classes = len(pairs)

    def speaker(self, p):
        return self.rows[p][0]

    def label(self, p):
        return self.rows[p][1]

    def sentence(self, p):
        return self.rows[p][2]


class Context:
    def __init__(self, sources: Sources):
        self.src = sources
        self._manifests = {}
        self._splits = {}
        self._index = None

    def manifest(self, level) -> Manifest:
        base = self.src.level_base.get(level)
        if base is None:
            raise VerificationError(f"unknown level {level!r} (no base corpus mapping)")
        if base not in self._manifests:
            self._manifests[base] = Manifest(self.src.manifests_dir / f"{base}_manifest.csv")
        return self._manifests[base]

    def split_index(self):
        if self._index is None:
            self._index = read_json(self.src.split_index)
        return self._index

    def split(self, key: str, expected_sha=None):
        """{population:[paths], folds:{fold:{test:[..], val:[..]}}, sha256}; sha verified vs split_index and plan."""
        if key not in self._splits:
            idx = self.split_index()
            if key not in idx:
                raise VerificationError(f"split key {key!r} not in split_index")
            path = self.src.v2_data_root / "plan_rc2" / idx[key]["path"]
            if not path.is_file():
                raise VerificationError(f"split file missing: {path}")
            sha = sha256_file(path)
            if sha != idx[key]["sha256"]:
                raise VerificationError(f"split file sha mismatch vs split_index for {key}")
            t = read_json(path)
            folds = {}
            pop = set(t["population"])
            for f in t["folds"]:
                test, val = list(f["test"]), list(f["val"])
                if set(test) - pop or set(val) - pop or set(test) & set(val):
                    raise VerificationError(f"malformed fold {f['fold']} in {key}")
                if len(set(test)) != len(test) or len(set(val)) != len(val):
                    raise VerificationError(f"duplicate paths in fold {f['fold']} of {key}")
                fit = list(f.get("fit", sorted(pop - set(test) - set(val))))
                if set(fit) & (set(test) | set(val)) or len(set(fit)) != len(fit):
                    raise VerificationError(f"fit recording leakage in fold {f['fold']} of {key}")
                folds[int(f["fold"])] = {"test": test, "val": val, "fit": fit}
            self._splits[key] = {"population": list(t["population"]), "folds": folds, "sha256": sha}
        s = self._splits[key]
        if expected_sha is not None and expected_sha != s["sha256"]:
            raise VerificationError(f"split sha declared by plan ({str(expected_sha)[:12]}..) != file sha for {key}")
        return s


# ----------------------------------------------------------------------------- prediction readers

def read_label_predictions_gz(path: Path):
    """A: gz csv with condition,estimator,draw,relative_path,speaker,y_true,y_pred."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        need = ["condition", "estimator", "draw", "relative_path", "speaker", "y_true", "y_pred"]
        if header != need:
            raise VerificationError(f"unexpected header {header} in {path}")
        cond, est, draw, rel, spk, yt, yp = [], [], [], [], [], [], []
        for row in rd:
            if len(row) != 7:
                raise VerificationError(f"malformed row (len {len(row)}) in {path}")
            cond.append(row[0]); est.append(row[1]); draw.append(int(row[2])); rel.append(row[3]); spk.append(row[4])
            yt.append(int(row[5])); yp.append(int(row[6]))
    return {"condition": cond, "estimator": est, "draw": np.asarray(draw, dtype=np.int64), "paths": rel,
            "speakers": spk, "y_true": np.asarray(yt, dtype=np.int64), "y_pred": np.asarray(yp, dtype=np.int64)}


def read_logit_predictions(path: Path, allow_sample_index: bool):
    """B1/B2: csv with [sample_index,]relative_path,speaker,y_true,y_pred,logit_0..logit_{C-1}."""
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        off = 0
        if header and header[0] == "sample_index":
            if not allow_sample_index:
                raise VerificationError(f"unexpected sample_index column in {path}")
            off = 1
        if header[off:off + 4] != ["relative_path", "speaker", "y_true", "y_pred"]:
            raise VerificationError(f"unexpected header {header} in {path}")
        lcols = header[off + 4:]
        if not lcols or lcols != [f"logit_{i}" for i in range(len(lcols))]:
            raise VerificationError(f"logit columns malformed in {path}: {lcols}")
        rel, spk, yt, yp, lg = [], [], [], [], []
        for row in rd:
            if len(row) != len(header):
                raise VerificationError(f"malformed row in {path}")
            rel.append(row[off]); spk.append(row[off + 1]); yt.append(int(row[off + 2])); yp.append(int(row[off + 3]))
            lg.append([float(v) for v in row[off + 4:]])
    return {"paths": rel, "speakers": spk, "y_true": np.asarray(yt, dtype=np.int64),
            "y_pred": np.asarray(yp, dtype=np.int64), "logits": np.asarray(lg, dtype=np.float64).reshape(len(rel), len(lcols))}


def check_rows_against(module, rep, unit_id, tag, pred, expected_paths, man: Manifest, n_logits=None):
    """Exact path set, no duplicates, manifest speaker/label agreement, label ranges. Returns True when clean."""
    ok = True
    paths = pred["paths"]
    if len(set(paths)) != len(paths):
        rep.fail(module, f"{tag}:duplicate_rows", f"{len(paths) - len(set(paths))} duplicated relative_path rows", unit_id); ok = False
    exp, got = set(expected_paths), set(paths)
    if got != exp:
        rep.fail(module, f"{tag}:row_set", f"missing={len(exp - got)} extra={len(got - exp)}; e.g. missing {sorted(exp - got)[:3]} extra {sorted(got - exp)[:3]}", unit_id); ok = False
    C = man.n_classes
    bad_spk = bad_lab = 0
    for p, s, y in zip(paths, pred["speakers"], pred["y_true"].tolist()):
        if p not in man.rows or man.speaker(p) != s:
            bad_spk += 1
        elif man.label(p) != y:
            bad_lab += 1
    if bad_spk:
        rep.fail(module, f"{tag}:speaker_mismatch", f"{bad_spk} rows disagree with manifest speaker/path", unit_id); ok = False
    if bad_lab:
        rep.fail(module, f"{tag}:label_mismatch", f"{bad_lab} rows have y_true != manifest label_index", unit_id); ok = False
    yp = pred["y_pred"]
    if yp.size and (int(yp.min()) < 0 or int(yp.max()) >= C):
        rep.fail(module, f"{tag}:y_pred_range", f"y_pred outside 0..{C - 1}", unit_id); ok = False
    if n_logits is not None and n_logits != C:
        rep.fail(module, f"{tag}:n_logits", f"{n_logits} logit columns, corpus has {C} classes", unit_id); ok = False
    return ok


def check_identity(module, rep, unit_id, declared: dict, planned: dict, keys):
    """Fields present in both the unit's json and the plan entry must agree (string comparison)."""
    for k in keys:
        if k in declared and k in planned and str(declared[k]) != str(planned[k]):
            rep.fail(module, f"identity:{k}", f"unit json says {declared[k]!r}, plan says {planned[k]!r}", unit_id)
    if "unit_id" in declared and str(declared["unit_id"]) != unit_id:
        rep.fail(module, "identity:unit_id", f"unit json unit_id {declared['unit_id']!r} != {unit_id}", unit_id)
    ident = planned.get("identity")
    if isinstance(ident, dict) and digest(ident) != unit_id:
        rep.fail(module, "identity:digest", "sha256(canonical identity) != unit_id", unit_id)


# ----------------------------------------------------------------------------- statistics primitives (numpy only)

def softmax_max(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def confusion(y_true, y_pred, C) -> np.ndarray:
    m = np.zeros((C, C), dtype=np.int64)
    np.add.at(m, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return m


def uar_from_confusion(m: np.ndarray) -> float:
    """Mean recall (percent) over classes with support > 0; NaN when there is no support at all."""
    sup = m.sum(axis=1)
    keep = sup > 0
    if not keep.any():
        return float("nan")
    return 100.0 * float(np.mean(np.diag(m)[keep] / sup[keep]))


def recalls_from_confusion(m: np.ndarray) -> np.ndarray:
    sup = m.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 100.0 * np.diag(m) / sup
    r[sup == 0] = np.nan
    return r


def bootstrap_indices(n_speakers: int) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_speakers, size=(BOOTSTRAP_REPS, n_speakers))


def ci_of_mean(values, idx: np.ndarray):
    """Speaker-equal mean and percentile CI of the bootstrap means (nan-aware per replicate)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or np.all(np.isnan(v)):
        return float("nan"), [float("nan"), float("nan")], 0
    with np.errstate(invalid="ignore"):
        samp = np.nanmean(v[idx], axis=1)
    lo, hi = np.percentile(samp, [2.5, 97.5], method="linear")
    return float(np.nanmean(v)), [float(lo), float(hi)], int(np.sum(~np.isnan(v)))


def ci_of_pooled(num_by_group, den_by_group, idx: np.ndarray):
    """num/den: (G, n_speakers) per-speaker sufficient statistics. estimate = mean_g(sum num / sum den);
    replicate = mean_g(resampled sum num / resampled sum den)."""
    num = np.asarray(num_by_group, dtype=np.float64); den = np.asarray(den_by_group, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = float(np.mean(num.sum(axis=1) / den.sum(axis=1)))
        samp = np.mean(num[:, idx].sum(axis=2) / den[:, idx].sum(axis=2), axis=0)
    lo, hi = np.percentile(samp, [2.5, 97.5], method="linear")
    return est, [float(lo), float(hi)], int(num.shape[1])


def ci_of_count(flags_by_group, idx: np.ndarray):
    fl = np.asarray(flags_by_group, dtype=np.float64)
    est = float(np.mean(fl.sum(axis=1)))
    samp = np.mean(fl[:, idx].sum(axis=2), axis=0)
    lo, hi = np.percentile(samp, [2.5, 97.5], method="linear")
    return est, [float(lo), float(hi)], int(fl.shape[1])


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-conf, kind="stable")
    err = (~correct[order]).astype(np.float64)
    k = np.arange(1, err.size + 1, dtype=np.float64)
    return float(np.mean(np.cumsum(err) / k))


def oracle_aurc(n: int, n_correct: int) -> float:
    k = np.arange(1, n + 1, dtype=np.float64)
    return float(np.mean(np.maximum(0.0, k - n_correct) / k))


NAN_CI = [float("nan"), float("nan")]


def endpoint(**kw):
    """Endpoint record; identifying fields = every key except id/estimate/ci95/n (None-valued keys dropped)."""
    est = kw.pop("estimate"); ci = kw.pop("ci95"); n = kw.pop("n")
    rec = {k: v for k, v in kw.items() if v is not None}
    rec["id"] = "|".join(f"{k}={rec[k]}" for k in sorted(rec))
    rec.update({"estimate": float(est), "ci95": [float(ci[0]), float(ci[1])], "n": int(n)})
    return rec


# ============================================================================= MODULE A: enrollment composition

def parse_condition(name: str):
    """'single:angry@3' -> (composition='single', emotion='angry', N=3); 'none' -> ('none', None, None)."""
    comp, N = name, None
    if "@" in name:
        comp, n_s = name.rsplit("@", 1)
        N = int(n_s)
    emotion = None
    for pref in ("other_single:", "single:"):
        if comp.startswith(pref):
            emotion = comp[len(pref):]
            comp = pref[:-1]
            break
    return comp, emotion, N


def default_estimators(comp: str):
    if comp == "none":
        return ["none"]
    if comp == "oracle_all":
        return ["naive"]
    if comp in ("other_balanced", "other_single"):
        return ["naive"]
    return list(ESTIMATORS_ALL)


def normalise_plan_a(plan: dict) -> dict:
    """-> {"units": [...], "levels": {level: {"query": {spk: [..]}, "enrollment": {spk: [..]}, "conditions": {name: entry}}}}
    entry = {"composition", "emotion", "N", "feasible", "estimators", "draws"}."""
    units = _first(plan, "units", required=True, where="A.json")
    level_blocks = _first(plan, "levels", default=None)
    out_levels = {}
    level_names = sorted({str(u["level"]) for u in units})
    default_draws = int(_first(plan, "draws", "n_draws", default=5))
    for lvl in level_names:
        if isinstance(level_blocks, dict) and lvl in level_blocks:
            blk = level_blocks[lvl]
            per_spk = _first(blk, "per_speaker", default=None)
            if isinstance(per_spk, dict):        # real plan.py layout: per_speaker[spk] = {Q: [...], E_by_class: {cls: [...]}, ...}
                query = {s: list(_first(v, "Q", "query", required=True, where=f"per_speaker[{s}]")) for s, v in per_spk.items()}
                enrol = {}
                for s, v in per_spk.items():
                    e = _first(v, "E", "enrollment", default=None)
                    if e is None:
                        e = [pth for lst in _first(v, "E_by_class", required=True, where=f"per_speaker[{s}]").values() for pth in lst]
                    enrol[s] = list(e)
            else:
                query = _first(blk, "query", "Q", "query_set", required=True, where=f"levels[{lvl}]")
                enrol = _first(blk, "enrollment", "E", "enrollment_pool", "enrolment", required=True, where=f"levels[{lvl}]")
            conds = _first(blk, "conditions", "condition_table", required=True, where=f"levels[{lvl}]")
            lvl_draws = int(_first(blk, "draws", "n_draws", default=default_draws))
        else:
            query = _first(_first(plan, "query", "Q", default={}), lvl, required=True, where="plan.query")
            enrol = _first(_first(plan, "enrollment", "E", default={}), lvl, required=True, where="plan.enrollment")
            conds = _first(_first(plan, "conditions", "condition_table", default={}), lvl, required=True, where="plan.conditions")
            lvl_draws = default_draws
        if isinstance(conds, dict):
            conds = [dict(v, condition=k) if isinstance(v, dict) else {"condition": k, "feasible": bool(v)} for k, v in conds.items()]
        table = {}
        for e in conds:
            name = str(_first(e, "condition", "name", "id", required=True, where=f"condition entry of {lvl}"))
            comp, emo, N = parse_condition(name)
            comp_decl = _first(e, "composition", default=None)
            if comp_decl is not None:            # may carry the emotion ("single:neutral") as in plan.py
                comp, emo2, _ = parse_condition(str(comp_decl))
                emo = emo2 if emo2 is not None else emo
            N = _first(e, "N", "n", default=N)
            if comp in ("none", "oracle_all") or N in (0, "0"):
                N = None
            feas = _first(e, "feasible", "feasibility", default=True)
            if isinstance(feas, str):
                feas = feas.lower() not in ("infeasible", "false", "0", "no")
            ests = list(_first(e, "estimators", default=default_estimators(comp)))
            seeds = _first(e, "draw_seeds", "seeds", default=None)
            if seeds is not None:
                draws = len(seeds)
            else:
                draws = int(_first(e, "draws", "n_draws", default=(1 if comp in ("none", "oracle_all") else lvl_draws)))
            table[name] = {"composition": comp, "emotion": emo, "N": (None if N is None else int(N)),
                           "feasible": bool(feas), "estimators": ests, "draws": draws}
        out_levels[lvl] = {"query": {str(k): list(v) for k, v in query.items()},
                           "enrollment": {str(k): list(v) for k, v in enrol.items()}, "conditions": table}
    norm_units = []
    id_fields = _first(_first(plan, "config", default={}), "unit_id_fields", default=None)
    for u in units:
        ident = u.get("identity")
        if ident is None and id_fields:      # plan.py convention: unit_id = sha256(canonical({field: unit[field]})) with 'ridge{...}' nested
            ident = {}
            for f_ in id_fields:
                base_ = f_.split("{", 1)[0]
                if base_ in u:
                    ident[base_] = u[base_]
                else:
                    ident = None; break
        norm_units.append({"identity": ident, "unit_id": str(u["unit_id"]), "level": str(u["level"]),
                           "enc": str(_first(u, "enc", "model", "encoder", required=True, where="A unit")),
                           "r": int(u["r"]), "fold": int(u["fold"]), "split_key": str(u["split_key"]),
                           "split_sha256": _first(u, "split_sha256", default=None), "test_speakers": u.get("test_speakers")})
    return {"program": plan.get("program", LEGACY_PROGRAM), "units": norm_units, "levels": out_levels,
            "raw": plan}


def derived_query_sets(level: str, man: Manifest, population, program=LEGACY_PROGRAM):
    """Independent derivation of Q per speaker from SPEC 2.2; returns list of candidate {spk: set(paths)} (one per salt)."""
    rule = QUERY_RULE.get(level)
    if rule is None:
        return []
    kind, k, salts = rule
    if program == PROGRAM:
        salts = tuple(s.replace(LEGACY_PROGRAM, PROGRAM) for s in salts)
    pop = list(population)
    cands = []
    if kind == "sentence":
        q_sents = [set(salts)]
    else:
        sents = sorted({man.sentence(p) for p in pop})
        q_sents = [set(_ranked(sents, salt)[:k]) for salt in salts]
    for qs in q_sents:
        d = defaultdict(set)
        for p in pop:
            if man.sentence(p) in qs:
                d[man.speaker(p)].add(p)
        cands.append(dict(d))
    return cands


def verify_reference_records(raw_plan, u, records, man, rep):
    """Reconstruct DEPLOY-2 reference draws from the frozen plan, without importing enrollment code."""
    uid, level, r, fold = u["unit_id"], u["level"], int(u["r"]), int(u["fold"])
    lp = raw_plan["levels"][level]
    speakers = list(u["test_speakers"])
    order = _ranked(sorted(speakers), f"{PROGRAM}:{level}:pair:r{r}:f{fold}")
    pairs = {s: order[(i + 1) % len(order)] for i, s in enumerate(order)} if len(order) >= 2 else {}
    if lp["pairs"][f"r{r}"][f"f{fold}"] != pairs:
        rep.fail("A", "reference:derangement", "donor map differs from frozen hash cycle", uid)
    index = {}
    for row in records:
        key = (row.get("speaker"), row.get("condition"), row.get("draw"))
        if key in index:
            rep.fail("A", "reference:duplicate", key, uid)
        index[key] = row
    expected_keys = set()
    for speaker in speakers:
        for cond in lp["conditions"]:
            comp = cond["composition"]
            if comp == "none" or not cond["feasible"]:
                continue
            other = comp.startswith("other_")
            donor = pairs[speaker] if other else speaker
            base_comp = comp[6:] if other else comp
            entry = lp["per_speaker"][donor]
            pool = entry["E_by_class"]
            for draw in range(int(cond["draws"])):
                key = (speaker, cond["id"], draw); expected_keys.add(key)
                if comp == "oracle_all":
                    expected = sorted(entry["Q"] + [p for c in lp["classes"] for p in pool[c]])
                else:
                    n = int(cond["N"])
                    seed = int(digest([PROGRAM, level, r, donor, base_comp, n, draw])[:16], 16)
                    rng = np.random.default_rng(seed)
                    if base_comp == "balanced":
                        permutations = {c: rng.permutation(pool[c]).tolist() for c in lp["class_order"]}
                        expected = [permutations[lp["class_order"][i % len(lp["class_order"])]]
                                    [i // len(lp["class_order"])] for i in range(n)]
                    else:
                        expected = rng.permutation(pool[base_comp.split(":", 1)[1]]).tolist()[:n]
                got = index.get(key)
                if got is None or got.get("source_speaker") != donor or got.get("paths") != expected:
                    rep.fail("A", "reference:draw_replay", key, uid)
                    continue
                paths = got["paths"]
                if len(paths) != len(set(paths)) or any(p not in man.rows or man.speaker(p) != donor for p in paths):
                    rep.fail("A", "reference:paths", key, uid)
                if comp != "oracle_all" and ({man.sentence(p) for p in paths} & {man.sentence(p) for p in entry["Q"]}):
                    rep.fail("A", "reference:query_sentence_leak", key, uid)
    if set(index) != expected_keys:
        rep.fail("A", "reference:key_set", f"missing={len(expected_keys-set(index))} extra={len(set(index)-expected_keys)}", uid)


def verify_fixed_partition(partition, man, original_fold=None):
    """Four-way recording exclusion, overlap assignment, and paired cell counts. Returns metadata only."""
    roles = ("fit", "seen", "new", "test")
    paths = {r: list(partition[r]) for r in roles}
    for role, pp in paths.items():
        if not pp or len(pp) != len(set(pp)) or any(p not in man.rows for p in pp):
            raise VerificationError(f"FIX {role}: empty/duplicate/unknown recording")
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            if set(paths[a]) & set(paths[b]):
                raise VerificationError(f"FIX recording overlap {a}/{b}")
    sp = {r: {man.speaker(p) for p in pp} for r, pp in paths.items()}
    if not sp["seen"] <= sp["fit"]:
        raise VerificationError("FIX seen donors absent from model fit")
    if sp["fit"] & (sp["new"] | sp["test"]) or sp["new"] & sp["test"]:
        raise VerificationError("FIX novel calibration/test speaker leakage")
    if len(sp["seen"]) != len(sp["new"]):
        raise VerificationError("FIX calibration speaker counts differ")
    if set(man.label(p) for p in paths["fit"]) != set(range(man.n_classes)):
        raise VerificationError("FIX training loses evaluation class")
    def counts(role):
        result = defaultdict(int)
        for p in paths[role]:
            result[(man.sentence(p), man.label(p))] += 1
        return dict(result)
    if counts("seen") != counts("new"):
        raise VerificationError("FIX sentence/emotion cells not matched")
    if original_fold is not None:
        if set(paths["test"]) != set(original_fold["test"]) or not set(paths["new"]) <= set(original_fold["val"]):
            raise VerificationError("FIX new/test assignments differ from outer GG fold")
        if "fit" in original_fold and not set(paths["fit"] + paths["seen"]) <= set(original_fold["fit"]):
            raise VerificationError("FIX training/seen recording outside original GG fit")
    return {"rows": {r: len(p) for r, p in paths.items()}, "speakers": {r: len(s) for r, s in sp.items()},
            "matched_cells": len(counts("seen"))}


def verify_a_integrity(plan: dict, run_dir: Path, ctx: Context, rep: Report):
    """Checks every planned A unit and accumulates per-speaker confusion matrices.
    Returns acc[(level, enc)][(condition, estimator)][r][speaker] = ndarray (draws, C, C)."""
    M = "A"
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    units_dir = run_dir / "A" / "units"
    seen_speakers = defaultdict(set)
    for lvl, blk in plan["levels"].items():
        man = ctx.manifest(lvl)
        q, e = blk["query"], blk["enrollment"]
        for s in q:
            if set(q[s]) & set(e.get(s, [])):
                rep.fail(M, "plan:Q_E_overlap", f"level {lvl} speaker {s}: query and enrollment share sentences")
            bad = [p for p in q[s] if p not in man.rows or man.speaker(p) != s]
            if bad:
                rep.fail(M, "plan:Q_manifest", f"level {lvl} speaker {s}: {len(bad)} query paths not this speaker's")
        if not blk["conditions"]:
            rep.fail(M, "plan:conditions", f"level {lvl}: empty condition table")
    for u in plan["units"]:
        uid, lvl, enc, r, fold = u["unit_id"], u["level"], u["enc"], u["r"], u["fold"]
        udir = units_dir / uid
        try:
            man = ctx.manifest(lvl)
            split = ctx.split(u["split_key"], u["split_sha256"])
        except VerificationError as ex:
            rep.fail(M, "split", ex, uid); continue
        if fold not in split["folds"]:
            rep.fail(M, "split:fold", f"fold {fold} not in {u['split_key']}", uid); continue
        test_paths = split["folds"][fold]["test"]
        test_speakers = sorted({man.speaker(p) for p in test_paths})
        if u.get("test_speakers") is not None and sorted(map(str, u["test_speakers"])) != test_speakers:
            rep.fail(M, "plan:test_speakers", f"plan lists {len(u['test_speakers'])} test speakers, split table gives {len(test_speakers)}", uid)
        # independent Q derivation (real levels only) against the plan's Q for this fold's test speakers
        blk = plan["levels"][lvl]
        cands = derived_query_sets(lvl, man, split["population"], plan.get("program", LEGACY_PROGRAM))
        if cands:
            match = [all(set(blk["query"].get(s, [])) == c.get(s, set()) for s in test_speakers) for c in cands]
            if not any(match):
                rep.fail(M, "plan:Q_derivation", f"plan query set differs from SPEC 2.2 derivation for level {lvl}", uid)
            elif len(cands) > 1 and not match[0]:
                rep.note(M, f"level {lvl}: query set matches alternative salt {QUERY_RULE[lvl][2][match.index(True)]}")
        for f in ("DONE", "predictions.csv.gz", "model.json"):
            if not (udir / f).is_file():
                rep.fail(M, "missing_file", f"{f} absent in {udir}", uid)
        if rep.failed(M) and any(x["unit_id"] == uid and x["check"] == "missing_file" for x in rep.failures):
            continue
        done = (udir / "DONE").read_text(encoding="utf-8").strip()
        sha = sha256_file(udir / "predictions.csv.gz")
        strict = plan.get("program") == PROGRAM
        want_done = f"{sha} {sha256_file(udir / 'model.json')}" if strict else sha
        if done != want_done:
            rep.fail(M, "DONE_sha", f"DONE={done[:16]}.. file={sha[:16]}..", uid)
        try:
            mj = read_json(udir / "model.json")
        except Exception as ex:
            rep.fail(M, "model_json", ex, uid); mj = {}
        declared = mj.get("unit", mj) if isinstance(mj, dict) else {}
        check_identity(M, rep, uid, declared if isinstance(declared, dict) else {}, u, ("level", "enc", "r", "fold", "split_key", "split_sha256"))
        if isinstance(mj, dict) and mj.get("predictions_sha256") not in (None, sha):
            rep.fail(M, "model_json:predictions_sha256", f"model.json {str(mj.get('predictions_sha256'))[:16]}.. != file {sha[:16]}..", uid)
        if strict:
            ref_path = udir / "references.json.gz"
            try:
                if sha256_file(ref_path) != mj.get("references_sha256"):
                    raise VerificationError("reference receipt sha mismatch")
                with gzip.open(ref_path, "rt", encoding="utf-8") as fh:
                    receipt = json.load(fh)
                if receipt.get("unit_id") != uid or receipt.get("program") != PROGRAM or receipt.get("schema_version") != 2:
                    raise VerificationError("reference receipt identity mismatch")
                if receipt.get("plan_sha256") != plan["raw"].get("plan_sha256") or mj.get("plan_sha256") != plan["raw"].get("plan_sha256"):
                    raise VerificationError("reference/model plan digest mismatch")
                verify_reference_records(plan["raw"], u, receipt["records"], man, rep)
            except (OSError, KeyError, ValueError) as ex:
                rep.fail(M, "reference:receipt", ex, uid)
        try:
            pred = read_label_predictions_gz(udir / "predictions.csv.gz")
        except Exception as ex:
            rep.fail(M, "predictions_read", ex, uid); continue
        expected_keys = {(c, est, d) for c, ent in blk["conditions"].items() if ent["feasible"]
                         for est in ent["estimators"] for d in range(ent["draws"])}
        q_paths = [p for s in test_speakers for p in blk["query"].get(s, [])]
        missing_q = [s for s in test_speakers if s not in blk["query"]]
        if missing_q:
            rep.fail(M, "plan:Q_missing_speaker", f"no query set for test speakers {missing_q}", uid)
        keys = list(zip(pred["condition"], pred["estimator"], pred["draw"].tolist()))
        got_keys = set(keys)
        if got_keys != expected_keys:
            rep.fail(M, "condition_set", f"missing {sorted(expected_keys - got_keys)[:4]} extra {sorted(got_keys - expected_keys)[:4]}", uid)
        # per key: exact Q coverage without duplicates
        by_key = defaultdict(list)
        for i, k in enumerate(keys):
            by_key[k].append(i)
        C = man.n_classes
        clean = True
        for k, idxs in by_key.items():
            sub = {"paths": [pred["paths"][i] for i in idxs], "speakers": [pred["speakers"][i] for i in idxs],
                   "y_true": pred["y_true"][idxs], "y_pred": pred["y_pred"][idxs]}
            if not check_rows_against(M, rep, uid, f"rows[{k[0]}/{k[1]}/{k[2]}]", sub, q_paths, man):
                clean = False
        if not clean or got_keys != expected_keys or missing_q:
            continue
        # accumulate confusions (draws stacked)
        spk_arr = np.asarray(pred["speakers"])
        for c, ent in blk["conditions"].items():
            if not ent["feasible"]:
                continue
            for est in ent["estimators"]:
                for s in test_speakers:
                    mats = np.zeros((ent["draws"], C, C), dtype=np.int64)
                    for d in range(ent["draws"]):
                        idxs = np.asarray(by_key[(c, est, d)])
                        m = idxs[spk_arr[idxs] == s]
                        mats[d] = confusion(pred["y_true"][m], pred["y_pred"][m], C)
                    if s in acc[(lvl, enc)][(c, est)][r]:
                        rep.fail(M, "duplicate_speaker_r", f"speaker {s} appears twice as test speaker for r={r}", uid)
                    acc[(lvl, enc)][(c, est)][r][s] = mats
        seen_speakers[(lvl, enc)].update(test_speakers)
    return acc, seen_speakers


def a_speaker_tables(acc_le: dict, speakers: list, C: int, rep: Report, lvl: str, enc: str):
    """-> uar[(cond, est)] = (n_spk,) ; rec[(cond, est)] = (n_spk, C). Per speaker: mean over draws, then over r."""
    uar, rec = {}, {}
    for key, by_r in acc_le.items():
        U = np.full(len(speakers), np.nan); R = np.full((len(speakers), C), np.nan)
        for i, s in enumerate(speakers):
            per_r_u, per_r_rec = [], []
            for r, by_s in sorted(by_r.items()):
                if s not in by_s:
                    continue
                mats = by_s[s]
                us = [uar_from_confusion(m) for m in mats]
                rs = np.stack([recalls_from_confusion(m) for m in mats])
                if any(np.isnan(u) for u in us) or np.isnan(rs).any():
                    rep.note("A", f"{lvl}/{enc} {key} r={r} speaker {s}: a class is absent from Q (skipped in UAR)")
                per_r_u.append(float(np.nanmean(us))); per_r_rec.append(np.nanmean(rs, axis=0))
            if per_r_u:
                U[i] = float(np.mean(per_r_u)); R[i] = np.mean(np.stack(per_r_rec), axis=0)
        if np.isnan(U).any():
            rep.fail("A", "speaker_missing", f"{lvl}/{enc} {key}: {int(np.isnan(U).sum())} speakers without any r")
        uar[key], rec[key] = U, R
    return uar, rec


def build_a_endpoints(plan: dict, acc, seen_speakers, ctx: Context, rep: Report):
    """Endpoint naming (identifying fields level, model=enc, comparison, condition, estimator, quantity):
      absolute : comparison=None, condition=<cond>, estimator=<est>, quantity=uar
      item 1   : comparison='balanced@5-none', condition='balanced@5', estimator=naive, quantity=uar
      item 2   : 'mean_single@N-none', condition='single:*@N'
      item 3   : 'mean_single@N-balanced@N', condition='single:*@N'
      item 4   : 'recall:single:e@N-none' (condition single:e@N) and 'recall:mean_single@N-none' (single:*@N), quantity=recall
      item 5   : 'est-naive@single', condition='single:*@N', estimator=<est>
      item 6   : 'other_single@N-single@N' / 'other_balanced@N-balanced@N'
      item 7   : 'harm_share:single@N-none' (single:*@N) and 'harm_share:single:e@N-none', quantity=harm_share."""
    endpoints, absolute, per_speaker = [], [], []
    for (lvl, enc) in sorted(acc):
        man = ctx.manifest(lvl); C = man.n_classes; cls = man.classes
        speakers = sorted(seen_speakers[(lvl, enc)])
        idx = bootstrap_indices(len(speakers))
        uar, rec = a_speaker_tables(acc[(lvl, enc)], speakers, C, rep, lvl, enc)
        conds = plan["levels"][lvl]["conditions"]
        feas = {c for c, e in conds.items() if e["feasible"]}

        def has(c, est="naive"):
            return (c, est) in uar

        def emit(comparison, values, condition=None, estimator=None, quantity="uar"):
            est_, ci, n = ci_of_mean(values, idx)
            endpoints.append(endpoint(level=lvl, model=enc, comparison=comparison, condition=condition,
                                      estimator=estimator, quantity=quantity, estimate=est_, ci95=ci, n=n))

        for (c, est), U in sorted(uar.items()):
            e_, ci, n = ci_of_mean(U, idx)
            absolute.append(endpoint(level=lvl, model=enc, condition=c, estimator=est, quantity="uar", estimate=e_, ci95=ci, n=n))
            for s, v in zip(speakers, U):
                per_speaker.append({"level": lvl, "enc": enc, "condition": c, "estimator": est, "speaker": s, "uar": float(v)})
        none_key = next((k for k in uar if conds.get(k[0], {}).get("composition") == "none"), None)
        if none_key is None:
            rep.fail("A", "none_missing", f"{lvl}/{enc}: no 'none' condition in the accumulated predictions"); continue
        U0, R0 = uar[none_key], rec[none_key]
        single_N = sorted({e["N"] for c, e in conds.items() if e["composition"] == "single" and c in feas})
        bal_N = sorted({e["N"] for c, e in conds.items() if e["composition"] == "balanced" and c in feas})

        def single_conds(N, comp="single"):
            return sorted(c for c, e in conds.items() if e["composition"] == comp and e["N"] == N and c in feas)

        if has("balanced@5"):
            emit("balanced@5-none", uar[("balanced@5", "naive")] - U0, "balanced@5", "naive")
        if single_N:
            N = single_N[-1]
            sc = single_conds(N)
            mean_single = np.mean(np.stack([uar[(c, "naive")] for c in sc]), axis=0)
            emit(f"mean_single@{N}-none", mean_single - U0, f"single:*@{N}", "naive")
            recall_diffs = []
            for c in sc:
                k = cls.index(conds[c]["emotion"])
                d = rec[(c, "naive")][:, k] - R0[:, k]
                recall_diffs.append(d)
                emit(f"recall:{c}-none", d, c, "naive", "recall")
            emit(f"recall:mean_single@{N}-none", np.mean(np.stack(recall_diffs), axis=0), f"single:*@{N}", "naive", "recall")
            harm = (mean_single - U0 < -HARM_POINTS).astype(np.float64)
            emit(f"harm_share:single@{N}-none", harm, f"single:*@{N}", "naive", "harm_share")
            for c in sc:
                emit(f"harm_share:{c}-none", (uar[(c, "naive")] - U0 < -HARM_POINTS).astype(np.float64), c, "naive", "harm_share")
        for N in single_N:
            sc = single_conds(N)
            if N in bal_N and has(f"balanced@{N}"):
                emit(f"mean_single@{N}-balanced@{N}", np.mean(np.stack([uar[(c, "naive")] for c in sc]), axis=0) - uar[(f"balanced@{N}", "naive")], f"single:*@{N}", "naive")
            for est in RECOVERY_ESTIMATORS:
                pairs = [c for c in sc if has(c, est)]
                if pairs:
                    emit("est-naive@single", np.mean(np.stack([uar[(c, est)] - uar[(c, "naive")] for c in pairs]), axis=0), f"single:*@{N}", est)
            oc = [c for c in single_conds(N, "other_single") if has(c) and has(c.replace("other_", "", 1))]
            if oc:
                emit(f"other_single@{N}-single@{N}", np.mean(np.stack([uar[(c, "naive")] - uar[(c.replace("other_", "", 1), "naive")] for c in oc]), axis=0), f"single:*@{N}", "naive")
        for N in bal_N:
            if has(f"other_balanced@{N}") and has(f"balanced@{N}"):
                emit(f"other_balanced@{N}-balanced@{N}", uar[(f"other_balanced@{N}", "naive")] - uar[(f"balanced@{N}", "naive")], f"balanced@{N}", "naive")
    return endpoints, absolute, per_speaker


# ============================================================================= MODULE B1: review budget on v2 units

def select_b1_rows(run_plan: Path, level_base: dict):
    with open(run_plan, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    sel = []
    for r in rows:
        if r["cell"] != "GG" or r["corpus_level"] not in level_base:
            continue
        if (r["arm"] == "CTRL" and r["model"] in B1_MODELS_CTRL) or (r["arm"] == "FT" and r["model"] == B1_MODEL_FT):
            sel.append(r)
    return sel


def verify_b1_integrity(ctx: Context, rep: Report):
    """-> groups[(level, model, r, seed)] = {"fold": {test paths...}, "conf", "correct", "y_true", "y_pred", "speakers"} pooled OOF."""
    M = "B1"
    rows = select_b1_rows(ctx.src.run_plan, ctx.src.level_base)
    if not rows:
        rep.fail(M, "run_plan", "no B1 source rows selected from run_plan.csv"); return {}
    units_root = ctx.src.v2_data_root / "runs" / "main" / "units"
    groups = defaultdict(lambda: {"parts": {}})
    for r in rows:
        uid, lvl, model, rr, seed, fold = r["unit_id"], r["corpus_level"], r["model"], int(r["r"]), int(r["seed_index"]), int(r["fold"])
        gkey = (lvl, model, rr, seed)
        udir = units_root / uid
        try:
            man = ctx.manifest(lvl)
            split = ctx.split(f"ctrl__{lvl}__GG__r{rr}", r.get("split_sha256") or None)
        except VerificationError as ex:
            rep.fail(M, "split", ex, uid); continue
        if fold not in split["folds"]:
            rep.fail(M, "split:fold", f"fold {fold} absent", uid); continue
        if not (udir / "unit.json").is_file() or not (udir / "predictions.csv").is_file():
            rep.fail(M, "missing_file", f"unit.json/predictions.csv absent in {udir}", uid); continue
        uj = read_json(udir / "unit.json")
        sha = sha256_file(udir / "predictions.csv")
        if uj.get("predictions_sha256") != sha:
            rep.fail(M, "predictions_sha256", f"unit.json {str(uj.get('predictions_sha256'))[:16]}.. file {sha[:16]}..", uid)
        planned = {"unit_id": uid, "arm": r["arm"], "model": model, "cell": "GG", "corpus_level": lvl, "fold": fold, "r": rr,
                   "seed_index": seed, "split_sha256": r["split_sha256"], "config_sha256": r["config_sha256"]}
        check_identity(M, rep, uid, uj, planned, tuple(planned))
        if str(uj.get("status", "done")) != "done":
            rep.fail(M, "status", f"unit status {uj.get('status')!r}", uid)
        try:
            pred = read_logit_predictions(udir / "predictions.csv", allow_sample_index=True)
        except Exception as ex:
            rep.fail(M, "predictions_read", ex, uid); continue
        if not check_rows_against(M, rep, uid, "test", pred, split["folds"][fold]["test"], man, pred["logits"].shape[1]):
            continue
        if fold in groups[gkey]["parts"]:
            rep.fail(M, "duplicate_fold", f"fold {fold} appears twice in group {gkey}", uid); continue
        groups[gkey]["parts"][fold] = pred
    # every group must have all folds of its split and cover the population exactly once
    out = {}
    for gkey, g in sorted(groups.items()):
        lvl, model, rr, seed = gkey
        split = ctx.split(f"ctrl__{lvl}__GG__r{rr}")
        want = set(split["folds"])
        if set(g["parts"]) != want:
            rep.fail(M, "group_incomplete", f"group {gkey}: folds {sorted(g['parts'])} != {sorted(want)}"); continue
        parts = [g["parts"][f] for f in sorted(g["parts"])]
        paths = [p for pr in parts for p in pr["paths"]]
        union = sorted(p for f in split["folds"].values() for p in f["test"])
        if sorted(paths) != union:
            rep.fail(M, "group_population", f"group {gkey}: pooled OOF rows != union of the split's test folds"); continue
        if union != sorted(split["population"]):
            rep.note(M, f"group {gkey}: test folds do not partition the population ({len(union)} of {len(split['population'])} paths)")
        logits = np.concatenate([pr["logits"] for pr in parts])
        y_true = np.concatenate([pr["y_true"] for pr in parts]); y_pred = np.concatenate([pr["y_pred"] for pr in parts])
        if not np.array_equal(logits.argmax(axis=1), y_pred):
            rep.note(M, f"group {gkey}: y_pred != argmax(logits) for {int((logits.argmax(axis=1) != y_pred).sum())} rows (using y_pred)")
        out[gkey] = {"conf": softmax_max(logits), "y_true": y_true, "y_pred": y_pred, "correct": y_true == y_pred,
                     "speakers": np.asarray([s for pr in parts for s in pr["speakers"]]), "C": logits.shape[1],
                     "paths": paths, "fold": np.concatenate([np.full(len(g["parts"][f]["paths"]), f) for f in sorted(g["parts"])])}
    return out


def speaker_stats(conf, correct, speakers_arr, speakers, accept_mask):
    """Per-speaker sufficient statistics for an accept mask: n, n_acc, n_acc_err, n_err, n_rej_err, coverage, acc_err."""
    n = np.zeros(len(speakers)); n_acc = np.zeros(len(speakers)); n_acc_err = np.zeros(len(speakers))
    n_err = np.zeros(len(speakers)); n_rej_err = np.zeros(len(speakers))
    for i, s in enumerate(speakers):
        m = speakers_arr == s
        a = accept_mask[m]; c = correct[m]
        n[i] = m.sum(); n_acc[i] = a.sum(); n_acc_err[i] = (~c[a]).sum(); n_err[i] = (~c).sum(); n_rej_err[i] = (~c[~a]).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = n_acc / n
        acc_err = np.where(n_acc > 0, n_acc_err / np.maximum(n_acc, 1), np.nan)
    return {"n": n, "n_acc": n_acc, "n_acc_err": n_acc_err, "n_err": n_err, "n_rej_err": n_rej_err, "coverage": cov, "acc_err": acc_err}


def b1_group_metrics(g: dict, speakers: list):
    """All SPEC 3 quantities for one pooled OOF group. Returns {"b": {b: {...}}, "aurc", "oracle_aurc"}."""
    conf, correct, y_true, y_pred, spk, C = g["conf"], g["correct"], g["y_true"], g["y_pred"], g["speakers"], g["C"]
    out = {"b": {}, "aurc": aurc(conf, correct), "oracle_aurc": oracle_aurc(conf.size, int(correct.sum()))}
    for b in B_GRID:
        tau = float(np.quantile(conf, b, method="linear"))
        acc = conf >= tau
        st = speaker_stats(conf, correct, spk, speakers, acc)
        rej_true = y_true[~acc]
        share = np.asarray([(rej_true == k).mean() if rej_true.size else np.nan for k in range(C)])
        # per-speaker equal quota: each speaker rejects its own lowest-confidence b fraction
        qacc = np.zeros_like(acc)
        for s in speakers:
            m = spk == s
            qacc[m] = conf[m] >= float(np.quantile(conf[m], b, method="linear"))
        qst = speaker_stats(conf, correct, spk, speakers, qacc)
        out["b"][b] = {"tau": tau, "stats": st, "quota": qst, "rejected_class_share": share,
                       "accepted_uar": uar_from_confusion(confusion(y_true[acc], y_pred[acc], C)),
                       "accepted_confusion_by_speaker": np.stack([confusion(y_true[acc & (spk == s)], y_pred[acc & (spk == s)], C) for s in speakers])}
    return out


def build_b1_endpoints(groups: dict, ctx: Context, rep: Report):
    """Naming (level, model, comparison='absolute', b, quantity): quantities accepted_error, error_capture_rate, accepted_uar,
    coverage_min, coverage_p10, count_speakers_coverage_below_0.70, share_speakers_coverage_below_0.70, count_speakers_coverage_zero,
    quota_accepted_error, rejected_share_<class>, and (no b) aurc, oracle_aurc.
    Estimates are flat means over (r, seed) groups; CIs via speaker bootstrap where speaker-decomposable."""
    endpoints, per_speaker = [], []
    by_lm = defaultdict(list)
    for gkey in sorted(groups):
        by_lm[(gkey[0], gkey[1])].append(gkey)
    for (lvl, model), gkeys in sorted(by_lm.items()):
        man = ctx.manifest(lvl); cls = man.classes
        speakers = sorted(set().union(*[set(groups[k]["speakers"].tolist()) for k in gkeys]))
        idx = bootstrap_indices(len(speakers))
        mets = [b1_group_metrics(groups[k], speakers) for k in gkeys]

        def add(quantity, estimate, ci, n, b=None):
            endpoints.append(endpoint(level=lvl, model=model, comparison="absolute", b=(None if b is None else float(b)), quantity=quantity,
                                      estimate=estimate, ci95=ci, n=n))

        for b in B_GRID:
            S = [m["b"][b]["stats"] for m in mets]; Q = [m["b"][b]["quota"] for m in mets]
            e, ci, n = ci_of_pooled([s["n_acc_err"] for s in S], [s["n_acc"] for s in S], idx); add("accepted_error", e, ci, n, b)
            e, ci, n = ci_of_pooled([s["n_rej_err"] for s in S], [s["n_err"] for s in S], idx); add("error_capture_rate", e, ci, n, b)
            e, ci, n = ci_of_pooled([q["n_acc_err"] for q in Q], [q["n_acc"] for q in Q], idx); add("quota_accepted_error", e, ci, n, b)
            e, ci, n = ci_of_count([s["coverage"] < COVERAGE_FLOOR for s in S], idx); add("count_speakers_coverage_below_0.70", e, ci, n, b)
            add("share_speakers_coverage_below_0.70", e / n, [ci[0] / n, ci[1] / n], n, b)
            e, ci, n = ci_of_count([s["coverage"] == 0 for s in S], idx); add("count_speakers_coverage_zero", e, ci, n, b)
            cov = np.stack([s["coverage"] for s in S])
            samp_min = np.mean(cov[:, idx].min(axis=2), axis=0); samp_p10 = np.mean(np.percentile(cov[:, idx], 10, axis=2, method="linear"), axis=0)
            add("coverage_min", float(np.mean(cov.min(axis=1))), np.percentile(samp_min, [2.5, 97.5], method="linear").tolist(), len(speakers), b)
            add("coverage_p10", float(np.mean(np.percentile(cov, 10, axis=1, method="linear"))), np.percentile(samp_p10, [2.5, 97.5], method="linear").tolist(), len(speakers), b)
            add("accepted_uar", float(np.mean([m["b"][b]["accepted_uar"] for m in mets])), NAN_CI, len(speakers), b)
            shares = np.stack([m["b"][b]["rejected_class_share"] for m in mets])
            for k, name in enumerate(cls):
                add(f"rejected_share_{name}", float(np.nanmean(shares[:, k])), NAN_CI, len(speakers), b)
            with np.errstate(invalid="ignore"):
                cov_mean = np.mean(cov, axis=0); err_mean = np.nanmean(np.stack([s["acc_err"] for s in S]), axis=0)
            n_utts = S[0]["n"]
            for i, s in enumerate(speakers):
                per_speaker.append({"level": lvl, "model": model, "b": float(b), "speaker": s, "coverage": float(cov_mean[i]),
                                    "accepted_error": float(err_mean[i]), "n_utts": int(n_utts[i])})
        add("aurc", float(np.mean([m["aurc"] for m in mets])), NAN_CI, len(speakers))
        add("oracle_aurc", float(np.mean([m["oracle_aurc"] for m in mets])), NAN_CI, len(speakers))
    return endpoints, per_speaker


# ============================================================================= MODULE B2: calibration-set speaker overlap

def normalise_plan_b2(plan: dict) -> list:
    units = _first(plan, "units", required=True, where="B2.json")
    out = []
    for u in units:
        out.append({"unit_id": str(u["unit_id"]), "level": str(u["level"]), "model": str(_first(u, "model", "enc", required=True, where="B2 unit")),
                    "cell": str(u["cell"]), "r": int(u["r"]), "fold": int(u["fold"]), "split_key": str(u["split_key"]),
                    "split_sha256": _first(u, "split_sha256", default=None), "identity": u.get("identity")})
    return out


def verify_b2_integrity(units: list, run_dir: Path, ctx: Context, rep: Report):
    """-> data[(level, model, r, fold)][cell] = {"val": pred, "test": pred}; also checks GR/GG test-fold identity."""
    M = "B2"
    data = defaultdict(dict)
    for u in units:
        uid, lvl, model, cell, r, fold = u["unit_id"], u["level"], u["model"], u["cell"], u["r"], u["fold"]
        udir = b2_unit_path(run_dir, uid, model)
        if u["split_key"] != f"ctrl__{lvl}__{cell}__r{r}":
            rep.fail(M, "plan:split_key", f"split_key {u['split_key']} != ctrl__{lvl}__{cell}__r{r}", uid)
        try:
            man = ctx.manifest(lvl)
            split = ctx.split(u["split_key"], u["split_sha256"])
        except VerificationError as ex:
            rep.fail(M, "split", ex, uid); continue
        if fold not in split["folds"]:
            rep.fail(M, "split:fold", f"fold {fold} absent", uid); continue
        missing = [f for f in ("DONE", "unit.json", "val_predictions.csv", "test_predictions.csv") if not (udir / f).is_file()]
        if missing:
            rep.fail(M, "missing_file", f"{missing} absent in {udir}", uid); continue
        done = (udir / "DONE").read_text(encoding="utf-8").split()
        vs, ts = sha256_file(udir / "val_predictions.csv"), sha256_file(udir / "test_predictions.csv")
        if done != [vs, ts]:
            rep.fail(M, "DONE_sha", f"DONE={[d[:12] for d in done]} files=[{vs[:12]}, {ts[:12]}]", uid)
        try:
            uj = read_json(udir / "unit.json")
        except Exception as ex:
            rep.fail(M, "unit_json", ex, uid); uj = {}
        check_identity(M, rep, uid, uj if isinstance(uj, dict) else {}, u, ("level", "model", "cell", "r", "fold", "split_key", "split_sha256"))
        if isinstance(uj, dict):
            for k_, sha_ in (("val_sha256", vs), ("test_sha256", ts)):
                if uj.get(k_) not in (None, sha_):
                    rep.fail(M, f"unit_json:{k_}", f"unit.json {str(uj.get(k_))[:16]}.. != file {sha_[:16]}..", uid)
            if str(uj.get("status", "done")) != "done":
                rep.fail(M, "status", f"unit status {uj.get('status')!r}", uid)
        try:
            pv = read_logit_predictions(udir / "val_predictions.csv", allow_sample_index=False)
            pt = read_logit_predictions(udir / "test_predictions.csv", allow_sample_index=False)
        except Exception as ex:
            rep.fail(M, "predictions_read", ex, uid); continue
        ok_v = check_rows_against(M, rep, uid, "val", pv, split["folds"][fold]["val"], man, pv["logits"].shape[1])
        ok_t = check_rows_against(M, rep, uid, "test", pt, split["folds"][fold]["test"], man, pt["logits"].shape[1])
        if not (ok_v and ok_t):
            continue
        if cell in data[(lvl, model, r, fold)]:
            rep.fail(M, "duplicate_unit", f"cell {cell} planned twice for {(lvl, model, r, fold)}", uid); continue
        data[(lvl, model, r, fold)][cell] = {"val": pv, "test": pt, "C": man.n_classes}
    for key, cells in sorted(data.items()):
        if set(cells) != {"GG", "GR"}:
            rep.fail(M, "pair_incomplete", f"{key}: cells {sorted(cells)} (need GG and GR)"); continue
        if sorted(cells["GG"]["test"]["paths"]) != sorted(cells["GR"]["test"]["paths"]):
            rep.fail(M, "pair_test_mismatch", f"{key}: GR and GG test rows differ")
    return data


def b2_unit_path(run_dir, unit_id, model):
    """Accept one consolidated tree or distinct managed CPU/GPU trees, never both copies."""
    candidates = [run_dir / "B2" / "units" / unit_id,
                  run_dir / ("B2_gpu" if model in ("cnn", "wavlm_ft") else "B2_cpu") / "units" / unit_id]
    existing = [p for p in candidates if p.is_dir()]
    if len(existing) > 1:
        raise VerificationError(f"duplicate B2 raw artifact location for {unit_id}")
    return existing[0] if existing else candidates[0]


def b2_unit_rule(pv: dict, pt: dict, speakers: list):
    """Primary rule (b=0.20 on val) and secondary rule (target risk); per-speaker realised quantities on test."""
    cv, ct = softmax_max(pv["logits"]), softmax_max(pt["logits"])
    corr_v, corr_t = pv["y_true"] == pv["y_pred"], pt["y_true"] == pt["y_pred"]
    spk_t = np.asarray(pt["speakers"])
    tau = float(np.quantile(cv, B2_BUDGET, method="linear"))
    acc_v = cv >= tau
    r_hat = float((~corr_v[acc_v]).mean()) if acc_v.any() else float("nan")
    acc_t = ct >= tau
    st = speaker_stats(ct, corr_t, spk_t, speakers, acc_t)
    # secondary: r* = 0.5 x val error; smallest k lowest-confidence val rows rejected s.t. accepted error <= r*
    r_star = B2_TARGET_RISK_FACTOR * float((~corr_v).mean())
    order = np.argsort(cv, kind="stable")
    err_sorted = (~corr_v[order]).astype(np.float64)
    n = cv.size
    remaining_err = err_sorted.sum() - np.concatenate([[0.0], np.cumsum(err_sorted)])          # errors left after rejecting k
    remaining_n = n - np.arange(n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_err_k = np.where(remaining_n > 0, remaining_err / np.maximum(remaining_n, 1), 0.0)
    k = int(np.argmax(acc_err_k <= r_star + 1e-15))
    tau2 = float(cv[order][k - 1]) if k > 0 else -np.inf
    acc_v2 = cv >= tau2; acc_t2 = ct >= tau2
    st2 = speaker_stats(ct, corr_t, spk_t, speakers, acc_t2)
    return {"tau": tau, "r_hat": r_hat, "cov_test": float(acc_t.mean()), "err_test": float((~corr_t[acc_t]).mean()) if acc_t.any() else float("nan"),
            "stats": st, "r_star": r_star, "val_reject_fraction": k / n, "tau2": tau2,
            "cov_test2": float(acc_t2.mean()), "err_test2": float((~corr_t[acc_t2]).mean()) if acc_t2.any() else float("nan"), "stats2": st2,
            "val_err2": float((~corr_v[acc_v2]).mean()) if acc_v2.any() else float("nan")}


def build_b2_endpoints(data: dict, ctx: Context, rep: Report):
    """Naming: endpoints (level, model, comparison='GR-GG', quantity in {realised_coverage, realised_accepted_error, gap_error,
    secondary_test_coverage, secondary_test_accepted_error}); absolute (level, model, cell, quantity in {promised_accepted_error,
    realised_coverage, realised_accepted_error, realised_coverage_pooled, realised_accepted_error_pooled, gap_coverage_pooled,
    gap_error_pooled, target_risk, secondary_val_reject_fraction, secondary_test_coverage, secondary_test_accepted_error,
    coverage_min, coverage_p10, n_cov_below_70, n_cov_zero}). Speaker-level values are nan-aware means over r."""
    endpoints, absolute, per_speaker = [], [], []
    by_lm = defaultdict(list)
    for key in data:
        by_lm[(key[0], key[1])].append(key)
    for (lvl, model), keys in sorted(by_lm.items()):
        speakers = sorted(set().union(*[set(data[k]["GG"]["test"]["speakers"]) for k in keys]))
        n_s = len(speakers); idx = bootstrap_indices(n_s)
        rs = sorted({k[2] for k in keys})
        cellv = {c: {"cov": np.full((len(rs), n_s), np.nan), "err": np.full((len(rs), n_s), np.nan), "gap": np.full((len(rs), n_s), np.nan),
                     "cov2": np.full((len(rs), n_s), np.nan), "err2": np.full((len(rs), n_s), np.nan), "units": []} for c in ("GG", "GR")}
        for key in sorted(keys):
            ri = rs.index(key[2])
            for cell in ("GG", "GR"):
                d = data[key][cell]
                spk_in = sorted(set(d["test"]["speakers"]))
                res = b2_unit_rule(d["val"], d["test"], spk_in)
                cellv[cell]["units"].append(res)
                for j, s in enumerate(spk_in):
                    i = speakers.index(s)
                    if not np.isnan(cellv[cell]["cov"][ri, i]):
                        rep.fail("B2", "speaker_twice", f"{lvl}/{model}/{cell} r={key[2]}: speaker {s} tested twice")
                    cellv[cell]["cov"][ri, i] = res["stats"]["coverage"][j]; cellv[cell]["err"][ri, i] = res["stats"]["acc_err"][j]
                    cellv[cell]["gap"][ri, i] = res["stats"]["acc_err"][j] - res["r_hat"]
                    cellv[cell]["cov2"][ri, i] = res["stats2"]["coverage"][j]; cellv[cell]["err2"][ri, i] = res["stats2"]["acc_err"][j]
        with np.errstate(invalid="ignore"):
            sp = {c: {q: np.nanmean(cellv[c][q], axis=0) for q in ("cov", "err", "gap", "cov2", "err2")} for c in ("GG", "GR")}
            diff = {q: np.nanmean(cellv["GR"][q] - cellv["GG"][q], axis=0) for q in ("cov", "err", "gap", "cov2", "err2")}
        for q, name in (("cov", "realised_coverage"), ("err", "realised_accepted_error"), ("gap", "gap_error"),
                        ("cov2", "secondary_test_coverage"), ("err2", "secondary_test_accepted_error")):
            e, ci, n = ci_of_mean(diff[q], idx)
            endpoints.append(endpoint(level=lvl, model=model, comparison="GR-GG", quantity=name, estimate=e, ci95=ci, n=n))
        for cell in ("GG", "GR"):
            U = cellv[cell]["units"]

            def add(quantity, estimate, ci, n):
                absolute.append(endpoint(level=lvl, model=model, cell=cell, quantity=quantity, estimate=estimate, ci95=ci, n=n))

            for q, name in (("cov", "realised_coverage"), ("err", "realised_accepted_error"), ("cov2", "secondary_test_coverage"), ("err2", "secondary_test_accepted_error")):
                e, ci, n = ci_of_mean(sp[cell][q], idx); add(name, e, ci, n)
            for attr, name in (("r_hat", "promised_accepted_error"), ("cov_test", "realised_coverage_pooled"), ("err_test", "realised_accepted_error_pooled"),
                               ("r_star", "target_risk"), ("val_reject_fraction", "secondary_val_reject_fraction")):
                add(name, float(np.nanmean([u[attr] for u in U])), NAN_CI, n_s)
            add("gap_coverage_pooled", float(np.nanmean([u["cov_test"] - (1.0 - B2_BUDGET) for u in U])), NAN_CI, n_s)
            add("gap_error_pooled", float(np.nanmean([u["err_test"] - u["r_hat"] for u in U])), NAN_CI, n_s)
            cov = cellv[cell]["cov"]
            add("coverage_min", float(np.nanmean(np.nanmin(cov, axis=1))), NAN_CI, n_s)
            add("coverage_p10", float(np.nanmean([np.percentile(row[~np.isnan(row)], 10, method="linear") for row in cov])), NAN_CI, n_s)
            add("n_cov_below_70", float(np.mean(np.sum(cov < COVERAGE_FLOOR, axis=1))), NAN_CI, n_s)
            add("n_cov_zero", float(np.mean(np.sum(cov == 0, axis=1))), NAN_CI, n_s)
            for i, s in enumerate(speakers):
                per_speaker.append({"level": lvl, "model": model, "cell": cell, "speaker": s, "coverage": float(sp[cell]["cov"][i]), "accepted_error": float(sp[cell]["err"][i])})
    return endpoints, absolute, per_speaker


# ============================================================================= comparison with scorer outputs

def _ident_key(rec: dict):
    items = []
    for k, v in rec.items():
        if k in NON_IDENT or isinstance(v, (list, dict)):
            continue
        if isinstance(v, float):
            v = format(v, ".12g")
        items.append((k, str(v)))
    return tuple(sorted(items))


def _close(a, b, tol):
    if _is_nan(a) and _is_nan(b):
        return True, 0.0
    try:
        d = abs(float(a) - float(b))
    except (TypeError, ValueError):
        return False, float("inf")
    return (d <= tol), d


def compare_endpoints(module, mine: list, theirs: list, tol: float, table: str):
    """Match on identifying fields; compare estimate, ci95 bounds and n. Returns summary dict."""
    m_map = {_ident_key(e): e for e in mine}
    t_map = {_ident_key(e): e for e in theirs}
    if len(m_map) != len(mine) or len(t_map) != len(theirs):
        raise VerificationError(f"{module}/{table}: duplicate endpoint identities (mine {len(mine) - len(m_map)}, scorer {len(theirs) - len(t_map)})")
    res = {"table": table, "compared": 0, "failed": 0, "max_abs_diff": 0.0, "mismatches": [], "only_in_scorer": [], "only_in_verifier": [], "ci_unverified": []}
    for k in sorted(set(m_map) | set(t_map)):
        if k not in m_map:
            res["only_in_scorer"].append(dict(k)); continue
        if k not in t_map:
            res["only_in_verifier"].append(dict(k)); continue
        a, b = m_map[k], t_map[k]
        res["compared"] += 1
        bad = []
        ok, d = _close(a["estimate"], b.get("estimate"), tol); res["max_abs_diff"] = max(res["max_abs_diff"], d if math.isfinite(d) else 0.0)
        if not ok:
            bad.append(("estimate", a["estimate"], b.get("estimate"), d))
        their_ci = b.get("ci95") or [float("nan"), float("nan")]
        for j, side in enumerate(("ci_lo", "ci_hi")):
            if _is_nan(a["ci95"][j]) and not _is_nan(their_ci[j]):
                res["ci_unverified"].append({**dict(k), "side": side}); continue
            ok, d = _close(a["ci95"][j], their_ci[j], tol); res["max_abs_diff"] = max(res["max_abs_diff"], d if math.isfinite(d) else 0.0)
            if not ok:
                bad.append((side, a["ci95"][j], their_ci[j], d))
        if "n" in b and int(b["n"]) != int(a["n"]):
            bad.append(("n", a["n"], b["n"], abs(int(b["n"]) - int(a["n"]))))
        if bad:
            res["failed"] += 1
            res["mismatches"].append({"identity": dict(k), "fields": [{"field": f, "verifier": v, "scorer": s, "abs_diff": dd} for f, v, s, dd in bad]})
    return res


def read_csv_rows(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def compare_per_speaker(module, mine: list, theirs: list, tol: float, numeric: tuple):
    def key(row):
        return tuple((k, str(row[k])) for k in sorted(row) if k not in numeric)
    def norm(row):
        out = dict(row)
        for k in ("b",):
            if k in out:
                out[k] = format(float(out[k]), ".12g")
        return out
    m_map = {key(norm(r)): r for r in mine}; t_map = {key(norm(r)): r for r in theirs}
    res = {"table": "per_speaker", "compared": 0, "failed": 0, "max_abs_diff": 0.0, "mismatches": [], "only_in_scorer": [], "only_in_verifier": []}
    for k in sorted(set(m_map) | set(t_map)):
        if k not in m_map:
            res["only_in_scorer"].append(dict(k)); continue
        if k not in t_map:
            res["only_in_verifier"].append(dict(k)); continue
        res["compared"] += 1
        bad = []
        for col in numeric:
            if col not in t_map[k]:
                bad.append((col, m_map[k][col], None, float("inf"))); continue
            ok, d = _close(m_map[k][col], t_map[k][col], tol); res["max_abs_diff"] = max(res["max_abs_diff"], d if math.isfinite(d) else 0.0)
            if not ok:
                bad.append((col, m_map[k][col], t_map[k][col], d))
        if bad:
            res["failed"] += 1
            res["mismatches"].append({"identity": dict(k), "fields": [{"field": f, "verifier": v, "scorer": s, "abs_diff": dd} for f, v, s, dd in bad]})
    return res


def write_csv(path: Path, rows: list, columns: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(columns) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else (repr(float(r[c])) if isinstance(r.get(c), float) else str(r[c])) for c in columns) + "\n")


# ============================================================================= orchestration

A_PS_COLS = ["level", "enc", "condition", "estimator", "speaker", "uar"]
B1_PS_COLS = ["level", "model", "b", "speaker", "coverage", "accepted_error", "n_utts"]
B2_PS_COLS = ["level", "model", "cell", "speaker", "coverage", "accepted_error"]
PS_NUMERIC = {"A": ("uar",), "B1": ("coverage", "accepted_error", "n_utts"), "B2": ("coverage", "accepted_error")}
EP_COLS = ["table", "level", "model", "cell", "comparison", "condition", "estimator", "b", "quantity", "estimate", "ci_lo", "ci_hi", "n"]


def recompute(module: str, plan_dir: Path, run_dir: Path, ctx: Context, rep: Report):
    """Integrity + recomputation for one module. Returns (endpoints, absolute, per_speaker) or None when integrity failed."""
    if module == "A":
        plan_path = plan_dir / "A.json"
        if not plan_path.is_file():
            rep.fail("A", "plan_missing", f"{plan_path} absent"); return None
        plan = normalise_plan_a(read_json(plan_path))
        acc, seen = verify_a_integrity(plan, run_dir, ctx, rep)
        if rep.failed("A"):
            return None
        return build_a_endpoints(plan, acc, seen, ctx, rep)
    if module == "B1":
        groups = verify_b1_integrity(ctx, rep)
        if rep.failed("B1"):
            return None
        ep, ps = build_b1_endpoints(groups, ctx, rep)
        return ep, [], ps
    if module == "B2":
        plan_path = plan_dir / "B2.json"
        if not plan_path.is_file():
            rep.fail("B2", "plan_missing", f"{plan_path} absent"); return None
        units = normalise_plan_b2(read_json(plan_path))
        data = verify_b2_integrity(units, run_dir, ctx, rep)
        if rep.failed("B2"):
            return None
        return build_b2_endpoints(data, ctx, rep)
    raise VerificationError(f"unknown module {module}")


def endpoint_rows(table, eps):
    return [{"table": table, **{c: e.get(c) for c in EP_COLS[1:9]}, "estimate": e["estimate"], "ci_lo": e["ci95"][0], "ci_hi": e["ci95"][1], "n": e["n"]} for e in eps]


def run(modules, plan_dir: Path, run_dir: Path, results_dir: Path, out: Path, sources: Sources, tolerance: float = 1e-9) -> dict:
    plans = [plan_dir / f"{m}.json" for m in ("A", "B2", "B2-FIX")]
    if any(p.is_file() and read_json(p).get("program") == PROGRAM for p in plans):
        from . import verify_deploy2
        return verify_deploy2.run(modules, plan_dir, run_dir, results_dir, out, sources, tolerance)
    ctx = Context(sources)
    rep = Report()
    report = {"program": PROGRAM, "kind": "independent-verification", "tolerance": tolerance, "modules": {}, "ambiguities": AMBIGUITIES}
    overall = True
    for module in modules:
        block = {"integrity": {"pass": False, "failures": []}, "comparison": None, "notes": []}
        try:
            got = recompute(module, plan_dir, run_dir, ctx, rep)
        except VerificationError as ex:
            rep.fail(module, "exception", ex); got = None
        block["integrity"]["failures"] = [f for f in rep.failures if f["module"] == module]
        block["integrity"]["pass"] = not block["integrity"]["failures"]
        block["notes"] = [n["note"] for n in rep.notes if n["module"] == module]
        if got is None:
            overall = False
            report["modules"][module] = block
            continue
        eps, absolute, ps = got
        ps_cols = {"A": A_PS_COLS, "B1": B1_PS_COLS, "B2": B2_PS_COLS}[module]
        write_csv(results_dir / f"verification_recomputed_{module}_endpoints.csv", endpoint_rows("endpoints", eps) + endpoint_rows("absolute", absolute), EP_COLS)
        write_csv(results_dir / f"verification_recomputed_{module}_per_speaker.csv", ps, ps_cols)
        comp = {"pass": False, "compared": 0, "failed": 0, "max_abs_diff": 0.0, "tables": []}
        ep_path, ps_path = results_dir / module / "endpoints.json", results_dir / module / "per_speaker.csv"
        if not ep_path.is_file() or not ps_path.is_file():
            comp["error"] = f"scorer results missing: {ep_path if not ep_path.is_file() else ps_path}"
        else:
            theirs = read_json(ep_path)
            if theirs.get("program") not in (PROGRAM, LEGACY_PROGRAM) or str(theirs.get("module")) != module:
                comp["error"] = f"scorer endpoints.json header mismatch: {theirs.get('program')}/{theirs.get('module')}"
            tables = [compare_endpoints(module, eps, list(theirs.get("endpoints", [])), tolerance, "endpoints"),
                      compare_endpoints(module, absolute, list(theirs.get("absolute", [])), tolerance, "absolute"),
                      compare_per_speaker(module, [{k: (repr(v) if isinstance(v, float) else str(v)) for k, v in r.items()} for r in ps],
                                          read_csv_rows(ps_path), tolerance, PS_NUMERIC[module])]
            comp["tables"] = tables
            comp["compared"] = sum(t["compared"] for t in tables); comp["failed"] = sum(t["failed"] for t in tables)
            comp["max_abs_diff"] = max(t["max_abs_diff"] for t in tables)
            comp["one_side_only"] = sum(len(t["only_in_scorer"]) + len(t["only_in_verifier"]) for t in tables)
            comp["pass"] = comp["failed"] == 0 and comp["one_side_only"] == 0 and "error" not in comp
        block["comparison"] = comp
        block["recomputed"] = {"endpoints": eps, "absolute": absolute, "n_per_speaker_rows": len(ps)}
        overall = overall and block["integrity"]["pass"] and comp["pass"]
        report["modules"][module] = block
    report["pass"] = bool(overall)
    report["reasons"] = [f"{f['module']}:{f['check']}:{f['detail']}" for f in rep.failures] + \
                        [f"{m}:comparison:{b['comparison'].get('error', 'failed=%d one_side_only=%d' % (b['comparison']['failed'], b['comparison'].get('one_side_only', 0)))}"
                         for m, b in report["modules"].items() if b["comparison"] is not None and not b["comparison"]["pass"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, sort_keys=True, default=float)
        fh.write("\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independent verifier for SER26-DEPLOY-1")
    ap.add_argument("--module", choices=["A", "B1", "B2", "B2-FIX", "all"], required=True)
    ap.add_argument("--plan-dir", type=Path, default=Path("v3/deploy/work/plan"))
    ap.add_argument("--run-dir", type=Path, default=Path("v3/deploy/work"))
    ap.add_argument("--results-dir", type=Path, default=Path("v3/deploy/results"))
    ap.add_argument("--v2-data", type=Path, default=Path(os.environ.get("SER_V2_DATA_ROOT", r"E:\科研\essay\SER-v2\v2")))
    ap.add_argument("--out", type=Path, default=Path("v3/deploy/results/verification.json"))
    ap.add_argument("--tolerance", type=float, default=1e-9)
    a = ap.parse_args(argv)
    cpu_guard(2)
    modules = ["A", "B1", "B2", "B2-FIX"] if a.module == "all" else [a.module]
    rep = run(modules, a.plan_dir, a.run_dir, a.results_dir, a.out, Sources.real(a.v2_data), a.tolerance)
    summary = {m: {"integrity": b["integrity"]["pass"], "comparison": (b["comparison"] or {}).get("pass"),
                   "compared": (b["comparison"] or {}).get("compared"), "failed": (b["comparison"] or {}).get("failed")} for m, b in rep["modules"].items()}
    print(json.dumps({"pass": rep["pass"], "out": str(a.out), "modules": summary}, ensure_ascii=False))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

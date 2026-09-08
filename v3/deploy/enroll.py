"""Block A of SER26-DEPLOY-2: enrollment composition (plan / run / score).

    python -m v3.deploy.enroll plan  --out v3/deploy/work/plan/A.json
    python -m v3.deploy.enroll run   --plan v3/deploy/work/plan/A.json --out v3/deploy/work/A [--units k1,k2 | --limit n]
    python -m v3.deploy.enroll score --plan v3/deploy/work/plan/A.json --run v3/deploy/work/A --out v3/deploy/results/A

CPU only (common.cpu_guard(2) is the first call of every command). Reads v2 data read-only; never edits v2/.
All plan-level logic (sentence split, condition table, feasibility, reference draws) is pure and reproducible
from the plan JSON alone; the run stage needs the plan + manifest labels + feature cache.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import csv
import gzip
import io
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v3.deploy import common as C

MODULE = "A"
ESTIMATOR_NONE = "none"
ESTIMATORS_REF = ("naive", "shrink", "neutral_filter", "prior_corrected")
RECOVERY_ESTIMATORS = ("shrink", "neutral_filter", "prior_corrected")
PRED_COLUMNS = ("condition", "estimator", "draw", "relative_path", "speaker", "y_true", "y_pred")

CONFIG = {
    "state": C.STATE,
    "ridge": {"alpha": 1.0, "class_weight": "balanced", "solver": "auto"},
    "shrink_k": 8,
    "draws": 5,
    "std_floor": 1e-3,
    "ddof": 0,
    "softmax_temperature": 1.0,
    "neutral_filter": {"min_keep": 2, "iterations": 2,
                       "variance": "E2 formula over all N references (mean from kept references only)",
                       "round2_normalisation": "naive",
                       "round2_options": {"round1": "references z-scored with the round-1 (kept-neutral mean, E2 variance) stats",
                                          "naive": "references z-scored with the E1 (all-reference mean/variance) stats"}},
    "prior_corrected": {"max_iter": 50, "tol": 1e-8, "probability_floor": 1e-12,
                        "init": "training class prior (utterance frequencies)"},
    "primary_N": 3,
    "variance_rule": "linear variance shrinkage; excludes between-mean mixture term",
    "other_reference_rule": "reuse the donor's own same-composition, N, r, draw reference sample",
    "harm_rule": "threshold the speaker's mean-over-feasible-emotions single UAR minus none",
    "balanced_N": [3, 5, 10, 20],
    "single_N": [3, 5],
    "other_N": [3, 5],
    "bootstrap": {"seed": C.BOOTSTRAP_SEED, "reps": C.BOOTSTRAP_REPS},
    "harm_threshold_points": 2.0,
    "unit_id_fields": ["program", "module", "level", "enc", "r", "fold", "split_key", "split_sha256",
                       "feature_sha256", "state", "ridge{alpha,class_weight}", "shrink_k"],
}

# base corpus -> sentence split rule (SPEC 2.2). "hash": first n of the salted ranking are Q; "fixed": explicit lists.
SENTENCE_RULES = {
    "cremad": ("hash", 5),
    "subesco": ("hash", 3),
    "ravdess": ("fixed", ["S1"], ["S2"]),      # statement 01 -> Q, statement 02 -> E
}
R_VALUES = (0, 1, 2)


# ============================================================================= plan: pure helpers

def sentence_salt(base: str) -> str:
    return f"{C.PROGRAM}:{base}:sent"


def sentence_split(base: str, sentences, rule=None) -> dict:
    sentences = sorted(set(sentences))
    rule = SENTENCE_RULES[base] if rule is None else rule
    salt = sentence_salt(base)
    ranked = C.ranked(sentences, salt)
    if rule[0] == "hash":
        n_q = int(rule[1])
        C.require(0 < n_q < len(sentences), f"{base}: cannot take {n_q} query sentences out of {len(sentences)}")
        q, e = ranked[:n_q], ranked[n_q:]
    elif rule[0] == "fixed":
        q, e = list(rule[1]), list(rule[2])
        C.require(set(q) | set(e) == set(sentences) and not (set(q) & set(e)),
                  f"{base}: fixed sentence rule {rule} does not partition {sentences}")
    else:
        raise C.IntegrityError(f"unknown sentence rule {rule}")
    C.require(not (set(q) & set(e)) and set(q) | set(e) == set(sentences), "Q/E must partition the sentences")
    return {"salt": salt, "rule": list(rule), "all": sentences, "ranked": ranked, "Q": q, "E": e}


def neutral_index(classes) -> int:
    hits = [i for i, c in enumerate(classes) if c.lower() == "neutral"]
    C.require(len(hits) == 1, f"expected exactly one neutral class in {classes}")
    return hits[0]


def balanced_need(class_order, n: int) -> dict:
    """Round-robin over class_order until n references: per-class requirement (only classes reached)."""
    return dict(Counter(class_order[i % len(class_order)] for i in range(n)))


def feasibility_reason(e_counts: dict, need: dict):
    """e_counts: speaker -> {class: count in E}. Returns None if EVERY speaker can supply `need`."""
    for spk in sorted(e_counts):
        for cls, n in need.items():
            have = int(e_counts[spk].get(cls, 0))
            if have < n:
                return f"speaker {spk}: class {cls} has {have} in E, needs {n}"
    return None


def build_conditions(classes, class_order, e_counts, min_fold_test_speakers: int, cfg=CONFIG) -> list:
    conds = [{"id": "none", "composition": "none", "N": 0, "source": "none", "draws": 1,
              "estimators": [ESTIMATOR_NONE], "feasible": True, "reason": None}]

    def add(cid, comp, n, source, estimators, reason, **extra):
        conds.append({"id": cid, "composition": comp, "N": n, "source": source, "draws": cfg["draws"],
                      "estimators": list(estimators), "feasible": reason is None, "reason": reason, **extra})

    for n in cfg["balanced_N"]:
        need = balanced_need(class_order, n)
        add(f"balanced@{n}", "balanced", n, "self", ESTIMATORS_REF, feasibility_reason(e_counts, need), need=need)
    for n in cfg["single_N"]:
        for cls in classes:
            add(f"single:{cls}@{n}", f"single:{cls}", n, "self", ESTIMATORS_REF,
                feasibility_reason(e_counts, {cls: n}), target_class=cls, need={cls: n})
    other_reason = None if min_fold_test_speakers >= 2 else \
        f"an outer fold has only {min_fold_test_speakers} test speaker(s); no partner available"
    for n in cfg["other_N"]:
        need = balanced_need(class_order, n)
        add(f"other_balanced@{n}", "other_balanced", n, "partner", ESTIMATORS_REF,
            other_reason or feasibility_reason(e_counts, need), need=need)
        for cls in classes:
            add(f"other_single:{cls}@{n}", f"other_single:{cls}", n, "partner", ESTIMATORS_REF,
                other_reason or feasibility_reason(e_counts, {cls: n}), target_class=cls, need={cls: n})
    conds.append({"id": "oracle_all", "composition": "oracle_all", "N": None, "source": "self_all", "draws": 1,
                  "estimators": ["naive"], "feasible": True, "reason": None,
                  "interpretation": "descriptive transductive reference; uses Q features; not a guaranteed upper bound"})
    C.require(len({c["id"] for c in conds}) == len(conds), "duplicate condition ids")
    return conds


def pair_salt(level: str, r: int, fold: int) -> str:
    return f"{C.PROGRAM}:{level}:pair:r{r}:f{fold}"


def pair_speakers(test_speakers, salt: str) -> dict:
    rk = C.ranked(sorted(test_speakers), salt)
    return {rk[i]: rk[(i + 1) % len(rk)] for i in range(len(rk))} if len(rk) >= 2 else {}


def build_level_plan(level: str, base: str, manifest: dict, tables: dict, classes, rule=None, cfg=CONFIG) -> dict:
    """tables: {r: split table with 'population' and 'folds' (each with test/val/fit path lists)}."""
    pops = {r: tuple(sorted(t["population"])) for r, t in tables.items()}
    first = next(iter(pops.values()))
    C.require(all(p == first for p in pops.values()), f"{level}: population differs across r")
    population = list(first)
    C.require(all(p in manifest for p in population), f"{level}: population path missing from manifest")
    classes = list(classes)
    cls_index = {c: i for i, c in enumerate(classes)}
    speakers = sorted({manifest[p]["speaker"] for p in population})
    sentences = sorted({manifest[p]["sentence"] for p in population})
    split = sentence_split(base, sentences, rule)
    q_set = set(split["Q"])
    per_speaker = {s: {"Q": [], "Q_labels": [], "E_by_class": {c: [] for c in classes}} for s in speakers}
    for p in population:
        row = manifest[p]
        C.require(row["label"] == classes[row["label_index"]], f"label/label_index mismatch for {p}")
        entry = per_speaker[row["speaker"]]
        if row["sentence"] in q_set:
            entry["Q"].append(p)
            entry["Q_labels"].append(int(row["label_index"]))
        else:
            entry["E_by_class"][row["label"]].append(p)
    e_counts = {}
    q_all_classes = True
    for s, entry in per_speaker.items():
        entry["E_class_counts"] = {c: len(entry["E_by_class"][c]) for c in classes}
        entry["Q_class_counts"] = {c: int(sum(1 for y in entry["Q_labels"] if y == cls_index[c])) for c in classes}
        entry["n_Q"], entry["n_E"] = len(entry["Q"]), int(sum(entry["E_class_counts"].values()))
        C.require(entry["n_Q"] > 0 and entry["n_E"] > 0, f"{level}: speaker {s} has empty Q or E")
        e_counts[s] = entry["E_class_counts"]
        q_all_classes &= all(v > 0 for v in entry["Q_class_counts"].values())
    C.require(q_all_classes, f"{level}: every speaker's Q must contain every fixed evaluation class")
    class_order = C.ranked(classes, f"{C.PROGRAM}:{base}:order")

    pairs, fold_test_speakers, min_fold = {}, {}, None
    for r, table in sorted(tables.items()):
        pairs[f"r{r}"], fold_test_speakers[f"r{r}"] = {}, {}
        for fold in table["folds"]:
            k = int(fold["fold"])
            test_paths = set(fold["test"])
            test_spk = sorted({manifest[p]["speaker"] for p in test_paths})
            spk_paths = {p for p in population if manifest[p]["speaker"] in set(test_spk)}
            C.require(spk_paths == test_paths, f"{level} r{r} fold{k}: test fold is not speaker-closed")
            fold_test_speakers[f"r{r}"][f"f{k}"] = test_spk
            pairs[f"r{r}"][f"f{k}"] = pair_speakers(test_spk, pair_salt(level, r, k))
            min_fold = len(test_spk) if min_fold is None else min(min_fold, len(test_spk))
        seen = [s for f in fold_test_speakers[f"r{r}"].values() for s in f]
        C.require(sorted(seen) == speakers, f"{level} r{r}: outer folds do not partition the speakers")

    conditions = build_conditions(classes, class_order, e_counts, min_fold or 0, cfg)
    return {
        "level": level, "base": base, "classes": classes, "neutral_index": neutral_index(classes),
        "class_order": class_order, "speakers": speakers, "n_population": len(population),
        "sentences": split, "q_all_classes_every_speaker": bool(q_all_classes),
        "per_speaker": per_speaker, "fold_test_speakers": fold_test_speakers, "pairs": pairs,
        "conditions": conditions,
    }


def unit_identity(level, enc, r, fold, split_key, split_sha256, feature_sha256, cfg=CONFIG) -> dict:
    return {"program": C.PROGRAM, "module": MODULE, "level": level, "enc": enc, "r": int(r), "fold": int(fold),
            "split_key": split_key, "split_sha256": split_sha256, "feature_sha256": feature_sha256,
            "state": cfg["state"], "ridge": {"alpha": cfg["ridge"]["alpha"], "class_weight": cfg["ridge"]["class_weight"]},
            "shrink_k": cfg["shrink_k"]}


def build_units(level_plans: dict, encoders, split_keys: dict, split_shas: dict, feature_shas: dict, cfg=CONFIG) -> list:
    """split_keys/split_shas: {level: {r: ...}}; feature_shas: {level: {enc: sha}}."""
    units = []
    for level in sorted(level_plans):
        lp = level_plans[level]
        for enc in encoders:
            for r_key in sorted(lp["fold_test_speakers"]):
                r = int(r_key[1:])
                for f_key in sorted(lp["fold_test_speakers"][r_key], key=lambda k: int(k[1:])):
                    fold = int(f_key[1:])
                    ident = unit_identity(level, enc, r, fold, split_keys[level][r], split_shas[level][r],
                                          feature_shas[level][enc], cfg)
                    test_spk = lp["fold_test_speakers"][r_key][f_key]
                    units.append({**ident, "unit_id": C.digest(ident), "test_speakers": test_spk,
                                  "n_test_speakers": len(test_spk)})
    C.require(len({u["unit_id"] for u in units}) == len(units), "duplicate unit ids")
    return units


def seal_plan(plan: dict) -> dict:
    plan["endpoint_manifest"] = expected_endpoint_manifest(plan)
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    plan["plan_sha256"] = C.digest(payload)
    return plan


def verify_plan_digest(plan: dict) -> None:
    C.require(plan.get("program") == C.PROGRAM and plan.get("module") == MODULE, "A program/module mismatch")
    C.require(plan.get("plan_sha256") == C.digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
              "A plan digest mismatch")
    C.require(len({u["unit_id"] for u in plan["units"]}) == len(plan["units"]), "duplicate planned A unit")


# ============================================================================= reference draws

def draw_seed(level: str, r: int, speaker: str, composition: str, n, draw: int) -> int:
    return int(C.digest([C.PROGRAM, level, int(r), speaker, composition, n, int(draw)])[:16], 16)


def speaker_all_paths(lp: dict, speaker: str) -> list:
    entry = lp["per_speaker"][speaker]
    return sorted(entry["Q"] + [p for c in lp["classes"] for p in entry["E_by_class"][c]])


def draw_references(lp: dict, level: str, r: int, fold: int, speaker: str, cond: dict, draw: int) -> list:
    """Deterministic reference sample for (speaker, condition, draw); reproducible from the plan alone."""
    comp = cond["composition"]
    if comp == "oracle_all":
        return speaker_all_paths(lp, speaker)
    if comp.startswith("other_"):
        source = lp["pairs"][f"r{r}"][f"f{fold}"][speaker]
        C.require(source != speaker, "partner must differ from the speaker")
        base_comp = comp[len("other_"):]
    else:
        source, base_comp = speaker, comp
    pool = lp["per_speaker"][source]["E_by_class"]
    n = int(cond["N"])
    # One frozen sample per donor/composition/draw, reused by the donor and its
    # derangement recipient. Other-speaker contrasts therefore cannot acquire
    # an extra independent reference-sampling difference.
    rng = np.random.default_rng(draw_seed(level, r, source, base_comp, n, draw))
    if base_comp == "balanced":
        order = lp["class_order"]
        perms = {c: [pool[c][i] for i in rng.permutation(len(pool[c]))] for c in order}
        taken, out = Counter(), []
        for i in range(n):
            c = order[i % len(order)]
            C.require(taken[c] < len(perms[c]), f"pool exhausted for class {c}")
            out.append(perms[c][taken[c]])
            taken[c] += 1
        return out
    if base_comp.startswith("single:"):
        cls = base_comp.split(":", 1)[1]
        items = pool[cls]
        C.require(len(items) >= n, f"pool exhausted for class {cls}")
        return [items[i] for i in rng.permutation(len(items))[:n]]
    raise C.IntegrityError(f"unknown composition {comp}")


# ============================================================================= model + estimators

class UnitModel:
    def __init__(self, ridge_norm, ridge_glob, scaler_mean, scaler_scale, m_glob, v_glob, delta, prior, n_classes, temperature=1.0):
        self.ridge_norm, self.ridge_glob = ridge_norm, ridge_glob
        self.scaler_mean, self.scaler_scale = scaler_mean, scaler_scale
        self.m_glob, self.v_glob, self.delta, self.prior = m_glob, v_glob, delta, prior
        self.n_classes, self.temperature = int(n_classes), float(temperature)

    @staticmethod
    def _decision(ridge, X):
        d = np.asarray(ridge.decision_function(X), dtype=np.float64)
        return np.stack([-d, d], axis=1) if d.ndim == 1 else d

    def scores_glob(self, X_raw):
        return self._decision(self.ridge_glob, (np.asarray(X_raw, dtype=np.float64) - self.scaler_mean) / self.scaler_scale)

    def posteriors_glob(self, X_raw):
        return C.softmax(self.scores_glob(X_raw), self.temperature)

    def scores_norm(self, X_norm):
        return self._decision(self.ridge_norm, X_norm)

    def posteriors_norm(self, X_norm):
        return C.softmax(self.scores_norm(X_norm), self.temperature)

    def predict_norm(self, X_norm):
        return self.scores_norm(X_norm).argmax(axis=1)


def fit_unit_model(X_tr, y_tr, spk_tr, n_classes: int, cfg=CONFIG) -> UnitModel:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    spk_tr = np.asarray(spk_tr)
    floor = cfg["std_floor"]
    speakers = sorted(set(spk_tr.tolist()))
    X_norm, X_cent = np.empty_like(X_tr), np.empty_like(X_tr)
    for s in speakers:
        m = spk_tr == s
        mu = X_tr[m].mean(axis=0)
        sd = np.maximum(X_tr[m].std(axis=0), floor)
        X_cent[m] = X_tr[m] - mu
        X_norm[m] = X_cent[m] / sd
    delta = np.zeros((n_classes, X_tr.shape[1]))
    for e in range(n_classes):
        per = [X_cent[(spk_tr == s) & (y_tr == e)].mean(axis=0) for s in speakers if ((spk_tr == s) & (y_tr == e)).any()]
        C.require(per, f"class {e} absent from the training set")
        delta[e] = np.mean(per, axis=0)
    kw = dict(alpha=cfg["ridge"]["alpha"], class_weight=cfg["ridge"]["class_weight"], solver=cfg["ridge"]["solver"])
    ridge_norm = RidgeClassifier(**kw).fit(X_norm, y_tr)
    scaler = StandardScaler().fit(X_tr)
    ridge_glob = RidgeClassifier(**kw).fit(scaler.transform(X_tr), y_tr)
    for rc in (ridge_norm, ridge_glob):
        C.require(rc.classes_.tolist() == list(range(n_classes)), "ridge classes must be 0..C-1")
    prior = np.bincount(y_tr, minlength=n_classes) / float(len(y_tr))
    C.require((prior > 0).all(), "every class must be present in training")
    return UnitModel(ridge_norm, ridge_glob, scaler.mean_.copy(), scaler.scale_.copy(), scaler.mean_.copy(),
                     scaler.var_.copy(), delta, prior, n_classes, cfg["softmax_temperature"])


def zscore(X, mean, var, floor: float):
    sd = np.maximum(np.sqrt(np.maximum(np.asarray(var, dtype=np.float64), 0.0)), floor)
    return (np.asarray(X, dtype=np.float64) - mean) / sd


def naive_stats(X_ref):
    X_ref = np.asarray(X_ref, dtype=np.float64)
    return X_ref.mean(axis=0), X_ref.var(axis=0)


def shrink_stats(m_ref, v_ref, n: int, m_glob, v_glob, k: float):
    """E2: separate linear shrinkage of means and variances, not mixture moments."""
    return (n * m_ref + k * m_glob) / (n + k), (n * v_ref + k * v_glob) / (n + k)


def select_neutral(P, neutral_idx: int, min_keep: int):
    keep = np.flatnonzero(P.argmax(axis=1) == neutral_idx)
    fallback = keep.size < min_keep
    if fallback:
        keep = np.argsort(-P[:, neutral_idx], kind="stable")[:min_keep]
    return np.sort(keep), bool(fallback)


def neutral_filter_stats(model: UnitModel, X_ref, neutral_idx: int, cfg=CONFIG):
    """E3: mean from references predicted neutral (fallback top-2 by neutral posterior); variance via E2 (all refs);
    Frozen default: the second pass re-scores E1-normalised references with M_norm.
    This is a two-pass heuristic, not an iterative distribution-fitting method."""
    X_ref = np.asarray(X_ref, dtype=np.float64)
    n, k, floor = X_ref.shape[0], cfg["shrink_k"], cfg["std_floor"]
    nf = cfg["neutral_filter"]
    m_ref, v_ref = naive_stats(X_ref)
    _, var = shrink_stats(m_ref, v_ref, n, model.m_glob, model.v_glob, k)
    keep, fb = select_neutral(model.posteriors_glob(X_ref), neutral_idx, nf["min_keep"])
    mean, fallbacks, kept = X_ref[keep].mean(axis=0), [fb], [int(keep.size)]
    mode = nf.get("round2_normalisation", "naive")
    C.require(mode in ("round1", "naive"), f"unknown neutral_filter round2_normalisation {mode!r}")
    for _ in range(int(nf["iterations"]) - 1):
        m_n, v_n = (m_ref, v_ref) if mode == "naive" else (mean, var)
        keep, fb = select_neutral(model.posteriors_norm(zscore(X_ref, m_n, v_n, floor)), neutral_idx, nf["min_keep"])
        mean = X_ref[keep].mean(axis=0)
        fallbacks.append(fb)
        kept.append(int(keep.size))
    return mean, var, {"kept": kept, "fallback": fallbacks}


def em_prior(P, prior0, max_iter: int = 50, tol: float = 1e-8, probability_floor: float = 1e-12):
    """EM-style prior adjustment on an unlabeled batch of Ridge pseudo-posteriors.

    Clip each input pseudo-probability then renormalise each row, so softmax
    underflow cannot cause a zero denominator. This does not validate label shift
    or calibrate class-balanced Ridge probabilities.
    """
    P = np.asarray(P, dtype=np.float64)
    prior0 = np.asarray(prior0, dtype=np.float64)
    C.require(P.ndim == 2 and P.shape[0] > 0 and P.shape[1] == prior0.size,
              "pseudo-posteriors must be a nonempty n-by-C matrix")
    C.require(np.isfinite(P).all() and (P >= 0).all() and (P.sum(axis=1) > 0).all(),
              "pseudo-posteriors must be finite, nonnegative and have positive row sums")
    C.require(0 < probability_floor < 1 and max_iter >= 1 and tol > 0, "invalid EM numerical controls")
    C.require((prior0 > 0).all() and abs(prior0.sum() - 1.0) < 1e-9, "training prior must be positive and sum to 1")
    P = np.maximum(P, probability_floor)
    P /= P.sum(axis=1, keepdims=True)
    pi, it = prior0.copy(), 0
    for it in range(1, int(max_iter) + 1):
        w = P * (pi / prior0)
        w /= w.sum(axis=1, keepdims=True)
        new = w.mean(axis=0)
        change = float(np.abs(new - pi).max())
        pi = new
        if change < tol:
            break
    return pi / pi.sum(), it


def prior_corrected_stats(model: UnitModel, X_ref, cfg=CONFIG):
    """E4: mean(refs) - sum_e pi_hat_e * delta_e with EM prior on M_glob pseudo-posteriors; variance via E2."""
    X_ref = np.asarray(X_ref, dtype=np.float64)
    n, k = X_ref.shape[0], cfg["shrink_k"]
    m_ref, v_ref = naive_stats(X_ref)
    _, var = shrink_stats(m_ref, v_ref, n, model.m_glob, model.v_glob, k)
    pc = cfg["prior_corrected"]
    pi, it = em_prior(model.posteriors_glob(X_ref), model.prior, pc["max_iter"], pc["tol"], pc["probability_floor"])
    return m_ref - pi @ model.delta, var, {"em_iter": int(it), "pi": pi}


def estimator_stats(model: UnitModel, name: str, X_ref, neutral_idx: int, cfg=CONFIG):
    """Returns (mean, var, info) for a reference-based estimator (E1..E4)."""
    n = np.asarray(X_ref).shape[0]
    m_ref, v_ref = naive_stats(X_ref)
    if name == "naive":
        return m_ref, v_ref, {}
    if name == "shrink":
        m, v = shrink_stats(m_ref, v_ref, n, model.m_glob, model.v_glob, cfg["shrink_k"])
        return m, v, {}
    if name == "neutral_filter":
        return neutral_filter_stats(model, X_ref, neutral_idx, cfg)
    if name == "prior_corrected":
        return prior_corrected_stats(model, X_ref, cfg)
    raise C.IntegrityError(f"unknown estimator {name}")


# ============================================================================= run: one unit

def unit_train_paths(lp: dict, test_speakers) -> list:
    test = set(test_speakers)
    return sorted(p for s in lp["speakers"] if s not in test for p in speaker_all_paths(lp, s))


def reference_records(plan: dict, unit: dict) -> list:
    """Replayable path-level evidence, independent of features and predictions."""
    level, r, fold = unit["level"], int(unit["r"]), int(unit["fold"])
    lp = plan["levels"][level]
    records = []
    for speaker in unit["test_speakers"]:
        for cond in lp["conditions"]:
            if cond["composition"] == "none" or not cond["feasible"]:
                continue
            source = (lp["pairs"][f"r{r}"][f"f{fold}"][speaker]
                      if cond["composition"].startswith("other_") else speaker)
            for draw in range(int(cond["draws"])):
                paths = draw_references(lp, level, r, fold, speaker, cond, draw)
                C.require(len(paths) == len(set(paths)), "a reference draw repeats a recording")
                if cond["composition"] != "oracle_all":
                    C.require(len(paths) == int(cond["N"]), "reference count differs from N")
                    C.require(not (set(paths) & set(lp["per_speaker"][source]["Q"])), "reference includes donor Q")
                records.append({"condition": cond["id"], "draw": draw, "speaker": speaker,
                                "source_speaker": source, "paths": paths})
    return records


def compute_unit(plan: dict, unit: dict, manifest: dict, X_of, cfg=None):
    """Fit M_norm/M_glob for the unit and produce prediction rows for every test speaker.
    X_of(paths) -> float array (n, D). Returns (rows, info); rows are PRED_COLUMNS tuples."""
    cfg = cfg or plan["config"]
    level, r, fold = unit["level"], int(unit["r"]), int(unit["fold"])
    lp = plan["levels"][level]
    n_classes, neutral_idx, floor = len(lp["classes"]), int(lp["neutral_index"]), cfg["std_floor"]
    t0 = time.perf_counter()
    references = reference_records(plan, unit)
    ref_index = {(rec["speaker"], rec["condition"], rec["draw"]): rec["paths"] for rec in references}
    train_paths = unit_train_paths(lp, unit["test_speakers"])
    X_tr = X_of(train_paths)
    y_tr = C.labels_for(manifest, train_paths)
    spk_tr = C.speakers_for(manifest, train_paths)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    model = fit_unit_model(X_tr, y_tr, spk_tr, n_classes, cfg)
    t_fit = time.perf_counter() - t1
    del X_tr

    t2 = time.perf_counter()
    rows, sets, diag = [], 0, Counter()
    by_cond = defaultdict(Counter)
    em_iters = []
    speakers_info = {}
    for spk in unit["test_speakers"]:
        entry = lp["per_speaker"][spk]
        q_paths, y_q = entry["Q"], np.asarray(entry["Q_labels"], dtype=np.int64)
        C.require(len(q_paths) == len(y_q) and
                  all(int(manifest[p]["label_index"]) == int(y) for p, y in zip(q_paths, y_q)), "Q labels drifted from manifest")
        C.require(set(y_q.tolist()) == set(range(n_classes)), "Q must contain all fixed evaluation classes")
        X_q = X_of(q_paths)
        speakers_info[spk] = {"n_Q": len(q_paths), "n_E": entry["n_E"],
                              "partner": lp["pairs"][f"r{r}"][f"f{fold}"].get(spk)}
        y_pred = model.posteriors_glob(X_q).argmax(axis=1)
        rows.extend(("none", ESTIMATOR_NONE, 0, p, spk, int(yt), int(yp)) for p, yt, yp in zip(q_paths, y_q, y_pred))
        sets += 1
        for cond in lp["conditions"]:
            if cond["composition"] == "none" or not cond["feasible"]:
                continue
            for draw in range(int(cond["draws"])):
                refs = ref_index[(spk, cond["id"], draw)]
                X_ref = X_of(refs)
                for est in cond["estimators"]:
                    mean, var, info = estimator_stats(model, est, X_ref, neutral_idx, cfg)
                    if est == "neutral_filter":
                        diag["nf_sets"] += 1
                        diag["nf_fallback_iter1"] += int(info["fallback"][0])
                        diag["nf_fallback_iter2"] += int(info["fallback"][-1])
                        by_cond[cond["id"]]["sets"] += 1
                        by_cond[cond["id"]]["fallback_iter1"] += int(info["fallback"][0])
                        by_cond[cond["id"]]["fallback_iter2"] += int(info["fallback"][-1])
                    elif est == "prior_corrected":
                        em_iters.append(info["em_iter"])
                    y_pred = model.predict_norm(zscore(X_q, mean, var, floor))
                    rows.extend((cond["id"], est, draw, p, spk, int(yt), int(yp)) for p, yt, yp in zip(q_paths, y_q, y_pred))
                    sets += 1
    t_pred = time.perf_counter() - t2
    info = {
        "timing": {"load_s": round(t_load, 3), "fit_s": round(t_fit, 3), "predict_s": round(t_pred, 3),
                   "total_s": round(time.perf_counter() - t0, 3)},
        "train": {"n_paths": len(train_paths), "n_speakers": len(set(spk_tr)),
                  "class_prior": [float(v) for v in model.prior]},
        "test_speakers": speakers_info,
        "references": references,
        "rows": len(rows), "sets": sets,
        "diagnostics": {**{k: int(v) for k, v in diag.items()},
                        "em_iter_mean": float(np.mean(em_iters)) if em_iters else None,
                        "em_iter_max": int(max(em_iters)) if em_iters else None,
                        "neutral_filter_round2_normalisation": cfg["neutral_filter"].get("round2_normalisation", "naive"),
                        "neutral_filter_by_condition": {k: dict(v) for k, v in sorted(by_cond.items())}},
    }
    return rows, info


def predictions_bytes(rows) -> bytes:
    buf = io.StringIO()
    buf.write(",".join(PRED_COLUMNS) + "\n")
    for cond, est, draw, path, spk, yt, yp in rows:
        buf.write(f"{cond},{est},{draw},{path},{spk},{yt},{yp}\n")
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
        gz.write(buf.getvalue().encode("utf-8"))
    return raw.getvalue()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def environment_info(guard=None) -> dict:
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(), "cpu_guard": guard}


def write_unit(unit_dir: Path, plan: dict, unit: dict, rows, info: dict, guard=None) -> str:
    data = predictions_bytes(rows)
    atomic_write_bytes(unit_dir / "predictions.csv.gz", data)
    sha = C.sha256_file(unit_dir / "predictions.csv.gz")
    references = {"schema_version": 2, "program": C.PROGRAM, "module": MODULE,
                  "unit_id": unit["unit_id"], "plan_sha256": plan.get("plan_sha256"),
                  "records": info["references"]}
    atomic_write_bytes(unit_dir / "references.json.gz", gzip.compress(C.canonical(references).encode("utf-8"), mtime=0))
    model = {"program": C.PROGRAM, "module": MODULE, "unit": unit, "plan_sha256": plan.get("plan_sha256"),
             "config": plan["config"], "environment": environment_info(guard), "written": C.now(),
             "predictions_sha256": sha, "predictions_bytes": len(data),
             "references_sha256": C.sha256_file(unit_dir / "references.json.gz"),
             **{k: v for k, v in info.items() if k != "references"}}
    C.atomic_write_json(unit_dir / "model.json", model)
    C.atomic_write_text(unit_dir / "DONE", sha + " " + C.sha256_file(unit_dir / "model.json") + "\n")
    return sha


def done_verified(unit_dir: Path) -> bool:
    done, pred = unit_dir / "DONE", unit_dir / "predictions.csv.gz"
    model_path, refs = unit_dir / "model.json", unit_dir / "references.json.gz"
    if not all(p.is_file() for p in (done, pred, model_path, refs)):
        return False
    recorded = done.read_text(encoding="utf-8").split()
    if len(recorded) != 2 or any(len(h) != 64 for h in recorded):
        return False
    if C.sha256_file(pred) != recorded[0] or C.sha256_file(model_path) != recorded[1]:
        return False
    model = C.read_json(model_path)
    return model.get("predictions_sha256") == recorded[0] and model.get("references_sha256") == C.sha256_file(refs)


def append_ledger(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(C.canonical({"ts": C.now(), **event}) + "\n")


def run_units(plan: dict, units, out_dir: Path, manifests: dict, feature_getters: dict, guard=None, log=print) -> dict:
    """manifests: {level: manifest}; feature_getters: {(level, enc): X_of}. Skips units with a verified DONE."""
    verify_plan_digest(plan)
    planned = {u["unit_id"]: u for u in plan["units"]}
    C.require(all(planned.get(u["unit_id"]) == u for u in units), "requested A unit is not in the sealed plan")
    ledger = out_dir / "ledger.jsonl"
    counts = Counter()
    for unit in units:
        unit_dir = out_dir / "units" / unit["unit_id"]
        ident = {k: unit[k] for k in ("unit_id", "level", "enc", "r", "fold")}
        if done_verified(unit_dir):
            model = C.read_json(unit_dir / "model.json")
            C.require(model.get("plan_sha256") == plan.get("plan_sha256") and model.get("unit") == unit,
                      "completed A unit belongs to another plan or unit identity")
            append_ledger(ledger, {"event": "skip", **ident})
            counts["skipped"] += 1
            log(f"skip  {unit['unit_id'][:12]} {unit['level']} {unit['enc']} r{unit['r']} f{unit['fold']} (DONE verified)")
            continue
        append_ledger(ledger, {"event": "start", **ident})
        t0 = time.perf_counter()
        try:
            rows, info = compute_unit(plan, unit, manifests[unit["level"]], feature_getters[(unit["level"], unit["enc"])])
            sha = write_unit(unit_dir, plan, unit, rows, info, guard)
        except Exception as exc:  # noqa: BLE001 - recorded in the ledger, then re-raised
            append_ledger(ledger, {"event": "failed", **ident, "error": f"{type(exc).__name__}: {exc}"})
            counts["failed"] += 1
            raise
        secs = time.perf_counter() - t0
        append_ledger(ledger, {"event": "done", **ident, "seconds": round(secs, 3), "rows": info["rows"],
                               "sets": info["sets"], "predictions_sha256": sha})
        counts["done"] += 1
        log(f"done  {unit['unit_id'][:12]} {unit['level']} {unit['enc']} r{unit['r']} f{unit['fold']} "
            f"rows={info['rows']} sets={info['sets']} bytes={(unit_dir / 'predictions.csv.gz').stat().st_size} "
            f"fit={info['timing']['fit_s']:.1f}s predict={info['timing']['predict_s']:.1f}s wall={secs:.1f}s")
    return dict(counts)


# ============================================================================= score

def verify_run_complete(plan: dict, run_dir: Path) -> dict:
    verify_plan_digest(plan)
    bad = []
    shas = {}
    for unit in plan["units"]:
        unit_dir = run_dir / "units" / unit["unit_id"]
        if not done_verified(unit_dir):
            bad.append(unit["unit_id"])
        else:
            model = C.read_json(unit_dir / "model.json")
            C.require(model.get("plan_sha256") == plan.get("plan_sha256") and model.get("unit") == unit,
                      "A metadata differs from the sealed plan")
            with gzip.open(unit_dir / "references.json.gz", "rt", encoding="utf-8") as fh:
                receipt = json.load(fh)
            C.require(receipt.get("unit_id") == unit["unit_id"] and receipt.get("plan_sha256") == plan.get("plan_sha256")
                      and receipt.get("records") == reference_records(plan, unit), "A reference receipt differs from plan replay")
            shas[unit["unit_id"]] = (unit_dir / "DONE").read_text(encoding="utf-8").split()[0]
    C.require(not bad, f"{len(bad)} of {len(plan['units'])} planned units lack a verified DONE, e.g. {bad[:3]}")
    return shas


def read_unit_predictions(unit_dir: Path) -> dict:
    """(condition, estimator, draw, speaker) -> {'paths': [...], 'y_true': array, 'y_pred': array}"""
    groups = defaultdict(lambda: {"paths": [], "y_true": [], "y_pred": []})
    with gzip.open(unit_dir / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        C.require(header == list(PRED_COLUMNS), f"unexpected prediction columns {header}")
        for cond, est, draw, path, spk, yt, yp in reader:
            g = groups[(cond, est, int(draw), spk)]
            g["paths"].append(path)
            g["y_true"].append(int(yt))
            g["y_pred"].append(int(yp))
    return {k: {"paths": v["paths"], "y_true": np.asarray(v["y_true"]), "y_pred": np.asarray(v["y_pred"])}
            for k, v in groups.items()}


def nest_mean(values_by_r: dict) -> float:
    """Average over draws within each r, then over r (SPEC 2.5 order)."""
    C.require(values_by_r and all(len(v) for v in values_by_r.values()), "empty draw list")
    return float(np.mean([np.mean(v) for _, v in sorted(values_by_r.items())]))


def nest_mean_vec(values_by_r: dict) -> np.ndarray:
    return np.mean([np.mean(np.asarray(v, dtype=np.float64), axis=0) for _, v in sorted(values_by_r.items())], axis=0)


def collect_per_speaker(plan: dict, run_dir: Path):
    """Returns (uar, recall): {(level, enc, cond, est): {speaker: value}} with the draw->r->speaker nesting."""
    raw_uar = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))     # key -> spk -> r -> [uar]
    raw_rec = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for unit in plan["units"]:
        level, enc, r = unit["level"], unit["enc"], int(unit["r"])
        lp = plan["levels"][level]
        n_classes = len(lp["classes"])
        cond_by_id = {c["id"]: c for c in lp["conditions"]}
        groups = read_unit_predictions(run_dir / "units" / unit["unit_id"])
        expected = set()
        for spk in unit["test_speakers"]:
            for cond in lp["conditions"]:
                if cond["feasible"]:
                    for est in cond["estimators"]:
                        for d in range(int(cond["draws"])):
                            expected.add((cond["id"], est, d, spk))
        C.require(set(groups) == expected, f"unit {unit['unit_id'][:12]}: prediction sets differ from the plan "
                                           f"(missing {len(expected - set(groups))}, extra {len(set(groups) - expected)})")
        for (cond_id, est, draw, spk), g in groups.items():
            entry = lp["per_speaker"][spk]
            C.require(sorted(g["paths"]) == sorted(entry["Q"]), f"unit {unit['unit_id'][:12]}: Q differs for {spk}/{cond_id}")
            lab = dict(zip(entry["Q"], entry["Q_labels"]))
            C.require(all(lab[p] == int(y) for p, y in zip(g["paths"], g["y_true"])), "y_true drifted from plan labels")
            key = (level, enc, cond_id, est)
            raw_uar[key][spk][r].append(C.uar(g["y_true"], g["y_pred"], n_classes))
            raw_rec[key][spk][r].append([C.class_recall(g["y_true"], g["y_pred"], c) for c in range(n_classes)])
            C.require(cond_by_id[cond_id]["feasible"], "predictions for an infeasible condition")
    uar, recall = {}, {}
    for key, by_spk in raw_uar.items():
        level = key[0]
        r_keys = {int(k[1:]) for k in plan["levels"][level]["fold_test_speakers"]}
        for spk, by_r in by_spk.items():
            C.require(set(by_r) == r_keys, f"{key}: speaker {spk} missing some r")
        uar[key] = {spk: nest_mean(by_r) for spk, by_r in by_spk.items()}
        recall[key] = {spk: nest_mean_vec(by_r) for spk, by_r in raw_rec[key].items()}
        C.require(sorted(uar[key]) == plan["levels"][level]["speakers"], f"{key}: speaker set incomplete")
    return uar, recall


def summarize(values, indices):
    values = np.asarray(values, dtype=np.float64)
    if np.isnan(values).any():
        ok = ~np.isnan(values)
        sampled = np.nanmean(values[indices], axis=1)
        lo, hi = np.percentile(sampled, [2.5, 97.5], method="linear")
        return {"estimate": float(values[ok].mean()), "ci95": [float(lo), float(hi)], "n": int(ok.sum()), "nan_dropped": int((~ok).sum())}
    return C.summarize_paired(values, indices)


def score_run(plan: dict, run_dir: Path, out_dir: Path, plan_path: Path | None = None) -> dict:
    unit_shas = verify_run_complete(plan, run_dir)
    uar, recall = collect_per_speaker(plan, run_dir)
    cfg = plan["config"]
    harm_thr = float(cfg["harm_threshold_points"])
    encoders = sorted({u["enc"] for u in plan["units"]})
    endpoints, absolute, harm_rows, recall_rows, ps_rows = [], [], [], [], []
    feas_summary = {}

    for level in sorted(plan["levels"]):
        lp = plan["levels"][level]
        speakers = lp["speakers"]
        idx = C.bootstrap_indices(len(speakers), cfg["bootstrap"]["seed"], cfg["bootstrap"]["reps"])
        classes = lp["classes"]
        conds = {c["id"]: c for c in lp["conditions"]}
        feas = {c["id"]: c["feasible"] for c in lp["conditions"]}
        feas_summary[level] = {c["id"]: {"feasible": c["feasible"], "reason": c["reason"]} for c in lp["conditions"]}
        single_ns = sorted({c["N"] for c in lp["conditions"] if c["composition"].startswith("single:") and c["feasible"]})
        n_primary = int(cfg["primary_N"])

        def single_classes(n, prefix="single"):
            return [c for c in classes if feas.get(f"{prefix}:{c}@{n}", False)]

        for enc in encoders:
            def vec(cond_id, est):
                key = (level, enc, cond_id, est)
                C.require(key in uar, f"missing scores for {key}")
                return np.asarray([uar[key][s] for s in speakers], dtype=np.float64)

            def rec(cond_id, est, cls_idx):
                key = (level, enc, cond_id, est)
                return np.asarray([recall[key][s][cls_idx] for s in speakers], dtype=np.float64)

            def entry(ep, comparison, cond_id, est, values, **extra):
                out = {"id": f"ep{ep}:{level}:{enc}:{comparison}", "endpoint": ep, "level": level, "model": enc,
                       "comparison": comparison, "condition": cond_id, "estimator": est, **summarize(values, idx), **extra}
                endpoints.append(out)
                return out

            def unavailable(ep, comparison, reason, **extra):
                endpoints.append({"id": f"ep{ep}:{level}:{enc}:{comparison}", "endpoint": ep, "level": level, "model": enc,
                                  "comparison": comparison, "condition": None, "estimator": None, "estimate": None,
                                  "ci95": None, "n": 0, "available": False, "reason": reason, **extra})

            # absolute UAR of every feasible (condition, estimator) + per_speaker rows
            for cond in lp["conditions"]:
                if not cond["feasible"]:
                    continue
                for est in cond["estimators"]:
                    v = vec(cond["id"], est)
                    absolute.append({"id": f"abs:{level}:{enc}:{cond['id']}:{est}", "level": level, "model": enc,
                                     "comparison": "absolute", "condition": cond["id"], "estimator": est, **summarize(v, idx)})
                    ps_rows.extend((level, enc, cond["id"], est, s, repr(float(x))) for s, x in zip(speakers, v))
            none = vec("none", ESTIMATOR_NONE)

            # 1. Shared N=3 is primary; all other feasible budgets remain secondary.
            for n in cfg["balanced_N"]:
                cid = f"balanced@{n}"
                if feas.get(cid):
                    entry(1, f"{cid}[naive]-none", cid, "naive", vec(cid, "naive") - none, N=n,
                          reference_condition="none", reference_estimator=ESTIMATOR_NONE, primary=(n == n_primary))
                else:
                    unavailable(1, f"{cid}[naive]-none", conds[cid]["reason"], N=n)

            # 2. Mean over feasible emotions at each N; shared N=3 is primary.
            if not single_ns:
                unavailable(2, "mean_e(single:e@N)[naive]-none", "no feasible single:* condition")
            for n in single_ns:
                cls = single_classes(n)
                diff = np.mean([vec(f"single:{c}@{n}", "naive") for c in cls], axis=0) - none
                entry(2, f"mean_e(single:e@{n})[naive]-none", f"single:*@{n}", "naive", diff, N=n, classes=cls,
                      reference_condition="none", reference_estimator=ESTIMATOR_NONE, primary=(n == n_primary))

            # 3. single - balanced at the same N
            common_n = [n for n in single_ns if feas.get(f"balanced@{n}")]
            if not common_n:
                unavailable(3, "mean_e(single:e@N)[naive]-balanced@N[naive]",
                            f"no N with both single:* and balanced feasible (single N={single_ns}, "
                            f"balanced N={[c['N'] for c in lp['conditions'] if c['composition'] == 'balanced' and c['feasible']]})")
            for n in common_n:
                cls = single_classes(n)
                diff = np.mean([vec(f"single:{c}@{n}", "naive") for c in cls], axis=0) - vec(f"balanced@{n}", "naive")
                entry(3, f"mean_e(single:e@{n})[naive]-balanced@{n}[naive]", f"single:*@{n}", "naive", diff, N=n, classes=cls,
                      reference_condition=f"balanced@{n}", reference_estimator="naive", primary=(n == n_primary))
            # Secondary comparison equalizes the average class mixture with the
            # three equally represented classes of the frozen balanced@3 pool.
            matched = [c for c in classes if c in conds[f"balanced@{n_primary}"]["need"]
                       and feas.get(f"single:{c}@{n_primary}")]
            if (feas.get(f"balanced@{n_primary}") and len(matched) == n_primary
                    and all(v == 1 for v in conds[f"balanced@{n_primary}"]["need"].values())):
                diff = np.mean([vec(f"single:{c}@{n_primary}", "naive") for c in matched], axis=0) - vec(f"balanced@{n_primary}", "naive")
                entry(3, f"mean_e_in_balanced_classes(single:e@{n_primary})[naive]-balanced@{n_primary}[naive]",
                      f"single:balanced_classes@{n_primary}", "naive", diff, N=n_primary, classes=matched,
                      reference_condition=f"balanced@{n_primary}", reference_estimator="naive", primary=False,
                      interpretation="secondary matched class mixture across the fixed balanced subset")

            # 4. Recall of the reference emotion, all feasible budgets/classes.
            if single_ns:
                for n in single_ns:
                    cls = single_classes(n)
                    recall_diffs = []
                    for c in cls:
                        ci = classes.index(c)
                        diff = rec(f"single:{c}@{n}", "naive", ci) - rec("none", ESTIMATOR_NONE, ci)
                        recall_diffs.append(diff)
                        e = entry(4, f"recall[{c}](single:{c}@{n}[naive])-recall[{c}](none)", f"single:{c}@{n}", "naive",
                                  diff, N=n, target_class=c, reference_condition="none", reference_estimator=ESTIMATOR_NONE,
                                  primary=False)
                        r_single = summarize(rec(f"single:{c}@{n}", "naive", ci), idx)
                        r_none = summarize(rec("none", ESTIMATOR_NONE, ci), idx)
                        recall_rows.append((level, enc, c, n, r_single["estimate"], r_none["estimate"], e["estimate"],
                                            e["ci95"][0], e["ci95"][1], e["n"]))
                    entry(4, f"mean_e(recall[e](single:e@{n}[naive])-recall[e](none))", f"single:*@{n}", "naive",
                          np.mean(recall_diffs, axis=0), N=n, classes=cls, reference_condition="none",
                          reference_estimator=ESTIMATOR_NONE, primary=(n == n_primary))
            else:
                unavailable(4, "recall[e](single:e@N)-recall[e](none)", "no feasible single:* condition")

            # 5. estimator recovery E2/E3/E4 - E1 under single:* (mean over e, plus per-class entries)
            for n in single_ns:
                cls = single_classes(n)
                for est in RECOVERY_ESTIMATORS:
                    diffs = [vec(f"single:{c}@{n}", est) - vec(f"single:{c}@{n}", "naive") for c in cls]
                    entry(5, f"mean_e(single:e@{n})[{est}]-[naive]", f"single:*@{n}", est, np.mean(diffs, axis=0), N=n,
                          classes=cls, reference_condition=f"single:*@{n}", reference_estimator="naive", primary=(n == n_primary))
                    for c, d in zip(cls, diffs):
                        entry(5, f"single:{c}@{n}[{est}]-[naive]", f"single:{c}@{n}", est, d, N=n, target_class=c,
                              reference_condition=f"single:{c}@{n}", reference_estimator="naive", primary=False)

            # 6. other_* - own references (N in other_N, E1); primary at shared N=3.
            for n_o in cfg["other_N"]:
                if feas.get(f"other_balanced@{n_o}") and feas.get(f"balanced@{n_o}"):
                    entry(6, f"other_balanced@{n_o}[naive]-balanced@{n_o}[naive]", f"other_balanced@{n_o}", "naive",
                          vec(f"other_balanced@{n_o}", "naive") - vec(f"balanced@{n_o}", "naive"), N=n_o,
                          reference_condition=f"balanced@{n_o}", reference_estimator="naive", primary=(n_o == n_primary))
                else:
                    unavailable(6, f"other_balanced@{n_o}[naive]-balanced@{n_o}[naive]",
                                conds[f"other_balanced@{n_o}"]["reason"] or conds[f"balanced@{n_o}"]["reason"])
                cls_o = [c for c in classes if feas.get(f"other_single:{c}@{n_o}") and feas.get(f"single:{c}@{n_o}")]
                if cls_o:
                    diffs = [vec(f"other_single:{c}@{n_o}", "naive") - vec(f"single:{c}@{n_o}", "naive") for c in cls_o]
                    entry(6, f"mean_e(other_single:e@{n_o})[naive]-mean_e(single:e@{n_o})[naive]", f"other_single:*@{n_o}", "naive",
                          np.mean(diffs, axis=0), N=n_o, classes=cls_o, reference_condition=f"single:*@{n_o}",
                          reference_estimator="naive", primary=(n_o == n_primary))
                    for c, d in zip(cls_o, diffs):
                        entry(6, f"other_single:{c}@{n_o}[naive]-single:{c}@{n_o}[naive]", f"other_single:{c}@{n_o}", "naive", d,
                              N=n_o, target_class=c, reference_condition=f"single:{c}@{n_o}", reference_estimator="naive",
                              primary=False)
                else:
                    reasons = sorted({conds[f"other_single:{c}@{n_o}"]["reason"] or conds[f"single:{c}@{n_o}"]["reason"] or ""
                                      for c in classes})
                    unavailable(6, f"mean_e(other_single:e@{n_o})[naive]-mean_e(single:e@{n_o})[naive]", "; ".join(r for r in reasons if r))

            # 7. harm share: fraction of speakers whose UAR under single:e@N is > 2 points below none
            for n in single_ns:
                cls = single_classes(n)
                for est in ESTIMATORS_REF:
                    ind = [(vec(f"single:{c}@{n}", est) - none < -harm_thr).astype(np.float64) for c in cls]
                    mean_diff = np.mean([vec(f"single:{c}@{n}", est) for c in cls], axis=0) - none
                    harm = (mean_diff < -harm_thr).astype(np.float64)
                    e = entry(7, f"harm_share(mean_e single:e@{n}[{est}] vs none)", f"single:*@{n}", est, harm,
                              N=n, classes=cls, reference_condition="none", reference_estimator=ESTIMATOR_NONE,
                              threshold_points=harm_thr, primary=(n == n_primary and est == "naive"))
                    harm_rows.append((level, enc, f"single:*@{n}", est, e["estimate"], e["ci95"][0], e["ci95"][1], e["n"]))
                    for c, v in zip(cls, ind):
                        e = entry(7, f"harm_share(single:{c}@{n}[{est}] vs none)", f"single:{c}@{n}", est, v, N=n, target_class=c,
                                  reference_condition="none", reference_estimator=ESTIMATOR_NONE, threshold_points=harm_thr,
                                  primary=False)
                        harm_rows.append((level, enc, f"single:{c}@{n}", est, e["estimate"], e["ci95"][0], e["ci95"][1], e["n"]))

    expected_ids = [row["id"] for row in plan["endpoint_manifest"]]
    observed_ids = [row["id"] for row in endpoints]
    C.require(len(observed_ids) == len(set(observed_ids)) and set(observed_ids) == set(expected_ids),
              "A scored endpoint set differs from frozen metadata manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    C.atomic_write_text(out_dir / "per_speaker.csv", "level,enc,condition,estimator,speaker,uar\n" +
                        "".join(",".join(map(str, r)) + "\n" for r in ps_rows))
    C.atomic_write_text(out_dir / "harm_share.csv", "level,enc,condition,estimator,share,ci_lo,ci_hi,n\n" +
                        "".join(",".join(map(str, r)) + "\n" for r in harm_rows))
    C.atomic_write_text(out_dir / "recall.csv", "level,enc,class,N,recall_single,recall_none,diff,ci_lo,ci_hi,n\n" +
                        "".join(",".join(map(str, r)) + "\n" for r in recall_rows))
    ledger = run_dir / "ledger.jsonl"
    result = {
        "program": C.PROGRAM, "module": MODULE, "written": C.now(),
        "endpoints": endpoints, "absolute": absolute, "feasibility": feas_summary,
        "inputs": {"plan_sha256": plan.get("plan_sha256"),
                   "plan_file_sha256": C.sha256_file(plan_path) if plan_path else None,
                   "run_ledger_sha256": C.sha256_file(ledger) if ledger.is_file() else None,
                   "units_predictions_sha256": C.digest(sorted(unit_shas.items())), "n_units": len(unit_shas)},
        "config": cfg,
    }
    C.atomic_write_json(out_dir / "endpoints.json", result)
    return result


# ============================================================================= CLI

def expected_endpoint_manifest(plan: dict) -> list:
    """Outcome-free enumeration, including unavailable endpoints, frozen in A.json."""
    cfg, result = plan["config"], []
    primary_n = int(cfg["primary_N"])
    encoders = sorted({u["enc"] for u in plan["units"]})
    for level, lp in sorted(plan["levels"].items()):
        conds = {c["id"]: c for c in lp["conditions"]}
        feasible = {k for k, c in conds.items() if c["feasible"]}
        ns = sorted({c["N"] for c in lp["conditions"] if c["composition"].startswith("single:") and c["feasible"]})
        def classes_for(n):
            return [c for c in lp["classes"] if f"single:{c}@{n}" in feasible]
        for enc in encoders:
            def add(ep, comparison, condition=None, estimator=None, available=True, primary=False, **kw):
                result.append({"id": f"ep{ep}:{level}:{enc}:{comparison}", "endpoint": ep, "level": level,
                               "model": enc, "comparison": comparison, "condition": condition, "estimator": estimator,
                               "available": available, "primary": primary, **kw})
            for n in cfg["balanced_N"]:
                name = f"balanced@{n}"
                ok = name in feasible
                add(1, f"{name}[naive]-none", name if ok else None, "naive" if ok else None,
                    available=ok, primary=ok and n == primary_n, N=n)
            if not ns:
                add(2, "mean_e(single:e@N)[naive]-none", available=False)
            for n in ns:
                add(2, f"mean_e(single:e@{n})[naive]-none", f"single:*@{n}", "naive", primary=n == primary_n,
                    N=n, classes=classes_for(n), reference_condition="none", reference_estimator=ESTIMATOR_NONE)
            common = [n for n in ns if f"balanced@{n}" in feasible]
            if not common:
                add(3, "mean_e(single:e@N)[naive]-balanced@N[naive]", available=False)
            for n in common:
                add(3, f"mean_e(single:e@{n})[naive]-balanced@{n}[naive]", f"single:*@{n}", "naive",
                    primary=n == primary_n, N=n, classes=classes_for(n), reference_condition=f"balanced@{n}", reference_estimator="naive")
            balanced = conds[f"balanced@{primary_n}"]
            matched = [c for c in lp["classes"] if c in balanced["need"] and f"single:{c}@{primary_n}" in feasible]
            if balanced["feasible"] and len(matched) == primary_n and all(v == 1 for v in balanced["need"].values()):
                add(3, f"mean_e_in_balanced_classes(single:e@{primary_n})[naive]-balanced@{primary_n}[naive]",
                    f"single:balanced_classes@{primary_n}", "naive", N=primary_n, classes=matched,
                    reference_condition=f"balanced@{primary_n}", reference_estimator="naive")
            if not ns:
                add(4, "recall[e](single:e@N)-recall[e](none)", available=False)
            for n in ns:
                classes = classes_for(n)
                for c in classes:
                    add(4, f"recall[{c}](single:{c}@{n}[naive])-recall[{c}](none)", f"single:{c}@{n}", "naive",
                        N=n, target_class=c, reference_condition="none", reference_estimator=ESTIMATOR_NONE)
                add(4, f"mean_e(recall[e](single:e@{n}[naive])-recall[e](none))", f"single:*@{n}", "naive",
                    primary=n == primary_n, N=n, classes=classes, reference_condition="none", reference_estimator=ESTIMATOR_NONE)
                for est in RECOVERY_ESTIMATORS:
                    add(5, f"mean_e(single:e@{n})[{est}]-[naive]", f"single:*@{n}", est, primary=n == primary_n,
                        N=n, classes=classes, reference_condition=f"single:*@{n}", reference_estimator="naive")
                    for c in classes:
                        add(5, f"single:{c}@{n}[{est}]-[naive]", f"single:{c}@{n}", est, N=n, target_class=c,
                            reference_condition=f"single:{c}@{n}", reference_estimator="naive")
            for n in cfg["other_N"]:
                ok = f"other_balanced@{n}" in feasible and f"balanced@{n}" in feasible
                add(6, f"other_balanced@{n}[naive]-balanced@{n}[naive]", f"other_balanced@{n}" if ok else None,
                    "naive" if ok else None, available=ok, primary=ok and n == primary_n,
                    **({"N": n, "reference_condition": f"balanced@{n}", "reference_estimator": "naive"} if ok else {}))
                classes = [c for c in lp["classes"] if f"other_single:{c}@{n}" in feasible and f"single:{c}@{n}" in feasible]
                add(6, f"mean_e(other_single:e@{n})[naive]-mean_e(single:e@{n})[naive]", f"other_single:*@{n}" if classes else None,
                    "naive" if classes else None, available=bool(classes), primary=bool(classes) and n == primary_n,
                    **({"N": n, "classes": classes, "reference_condition": f"single:*@{n}", "reference_estimator": "naive"} if classes else {}))
                for c in classes:
                    add(6, f"other_single:{c}@{n}[naive]-single:{c}@{n}[naive]", f"other_single:{c}@{n}", "naive",
                        N=n, target_class=c, reference_condition=f"single:{c}@{n}", reference_estimator="naive")
            for n in ns:
                classes = classes_for(n)
                for est in ESTIMATORS_REF:
                    add(7, f"harm_share(mean_e single:e@{n}[{est}] vs none)", f"single:*@{n}", est,
                        primary=n == primary_n and est == "naive", N=n, classes=classes,
                        reference_condition="none", reference_estimator=ESTIMATOR_NONE)
                    for c in classes:
                        add(7, f"harm_share(single:{c}@{n}[{est}] vs none)", f"single:{c}@{n}", est,
                            N=n, target_class=c, reference_condition="none", reference_estimator=ESTIMATOR_NONE)
    C.require(len({row["id"] for row in result}) == len(result), "duplicate A endpoint identity")
    return result


def feasibility_markdown(plan: dict) -> str:
    lines = [f"# Block A feasibility ({C.PROGRAM})", "", f"plan_sha256: `{plan['plan_sha256']}`", "",
             "| level | condition | N | feasible | reason |", "|---|---|---|---|---|"]
    for level in sorted(plan["levels"]):
        for c in plan["levels"][level]["conditions"]:
            lines.append(f"| {level} | `{c['id']}` | {c['N'] if c['N'] is not None else '-'} | "
                         f"{'yes' if c['feasible'] else 'NO'} | {c['reason'] or ''} |")
    lines += ["", "| level | base | classes | class order | Q sentences | E sentences | speakers | Q has all classes for every speaker |",
              "|---|---|---|---|---|---|---|---|"]
    for level in sorted(plan["levels"]):
        lp = plan["levels"][level]
        lines.append(f"| {level} | {lp['base']} | {len(lp['classes'])} | {' > '.join(lp['class_order'])} | "
                     f"{' '.join(lp['sentences']['Q'])} | {' '.join(lp['sentences']['E'])} | {len(lp['speakers'])} | "
                     f"{lp['q_all_classes_every_speaker']} |")
    lines += ["", f"units: {len(plan['units'])}", ""]
    return "\n".join(lines)


def cmd_plan(args, guard) -> None:
    level_plans, split_keys, split_shas, feature_shas, manifest_shas = {}, {}, {}, {}, {}
    idx = C.split_index()
    for level, base in C.LEVELS.items():
        manifest = C.load_manifest(base)
        classes = C.class_table(manifest)
        tables, split_keys[level], split_shas[level] = {}, {}, {}
        for r in R_VALUES:
            key = f"ctrl__{level}__GG__r{r}"
            tables[r] = C.load_split(key)
            split_keys[level][r], split_shas[level][r] = key, idx[key]["sha256"]
        level_plans[level] = build_level_plan(level, base, manifest, tables, classes)
        manifest_shas[level] = C.sha256_file(C.manifest_path(base))
        feature_shas[level] = {}
        for enc in C.ENCODERS:
            cands = sorted((C.V2_DATA_ROOT / "features").glob(f"{base}__{enc}__*.npz"))
            C.require(len(cands) == 1, f"expected exactly one cache for {base}/{enc}, found {len(cands)}")
            feature_shas[level][enc] = C.sha256_file(cands[0])
            level_plans[level].setdefault("feature_files", {})[enc] = cands[0].name
        level_plans[level]["split_keys"] = {f"r{r}": k for r, k in split_keys[level].items()}
        level_plans[level]["split_sha256"] = {f"r{r}": s for r, s in split_shas[level].items()}
        level_plans[level]["manifest_sha256"] = manifest_shas[level]
        level_plans[level]["feature_sha256"] = feature_shas[level]
        print(f"plan  {level}: {len(level_plans[level]['speakers'])} speakers, "
              f"{sum(c['feasible'] for c in level_plans[level]['conditions'])}/{len(level_plans[level]['conditions'])} conditions feasible")
    units = build_units(level_plans, C.ENCODERS, split_keys, split_shas, feature_shas)
    here = Path(__file__).resolve()
    plan = {
        "program": C.PROGRAM, "module": MODULE, "created": C.now(), "config": CONFIG,
        "encoders": list(C.ENCODERS), "r_values": list(R_VALUES),
        "draw_seed_rule": "int(sha256(canonical([program, level, r, source_speaker, base_composition_without_other_prefix, N, draw]))[:16], 16)",
        "code_sha256": {"enroll.py": C.sha256_file(here), "common.py": C.sha256_file(here.parent / "common.py")},
        "environment": environment_info(guard), "levels": level_plans, "units": units,
    }
    seal_plan(plan)
    out = Path(args.out)
    C.atomic_write_json(out, plan)
    C.atomic_write_text(out.parent / "A_feasibility.md", feasibility_markdown(plan))
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(units)} units, plan_sha256 {plan['plan_sha256'][:12]})")
    print(f"wrote {out.parent / 'A_feasibility.md'}")


def select_units(plan: dict, spec: str | None, limit: int | None) -> list:
    units = plan["units"]
    if spec:
        wanted = []
        for tok in [t.strip() for t in spec.split(",") if t.strip()]:
            if "/" in tok:
                level, enc, r, fold = tok.split("/")
                hits = [u for u in units if u["level"] == level and u["enc"] == enc and u["r"] == int(r) and u["fold"] == int(fold)]
            else:
                hits = [u for u in units if u["unit_id"] == tok or u["unit_id"].startswith(tok)]
            C.require(len(hits) == 1, f"unit selector {tok!r} matched {len(hits)} units")
            wanted.append(hits[0])
        units = wanted
    if limit is not None:
        units = units[:limit]
    return units


def cmd_run(args, guard) -> None:
    plan_path = Path(args.plan)
    plan = C.read_json(plan_path)
    C.require(plan.get("module") == MODULE and plan.get("program") == C.PROGRAM, "not a Block A plan")
    units = select_units(plan, args.units, args.limit)
    out_dir = Path(args.out)
    print(f"run   {len(units)} unit(s) -> {out_dir}")
    manifests, getters = {}, {}
    total = Counter()
    t0 = time.perf_counter()
    for (level, enc) in sorted({(u["level"], u["enc"]) for u in units}):
        lp = plan["levels"][level]
        if level not in manifests:
            manifests[level] = C.load_manifest(lp["base"])
            C.require(C.sha256_file(C.manifest_path(lp["base"])) == lp["manifest_sha256"], f"manifest sha changed for {level}")
        cache = C.FeatureCache(lp["base"], enc)
        C.require(cache.sha256 == lp["feature_sha256"][enc], f"feature cache sha changed for {level}/{enc}")
        state = plan["config"]["state"]
        getters[(level, enc)] = lambda paths, _c=cache, _s=state: _c.get(paths, state=_s).astype(np.float64)
        group = [u for u in units if u["level"] == level and u["enc"] == enc]
        counts = run_units(plan, group, out_dir, manifests, getters, guard)
        total.update(counts)
        del getters[(level, enc)], cache
    print(f"run   finished: {dict(total)} in {time.perf_counter() - t0:.1f}s")


def cmd_score(args, guard) -> None:
    plan_path = Path(args.plan)
    plan = C.read_json(plan_path)
    result = score_run(plan, Path(args.run), Path(args.out), plan_path)
    print(f"score wrote {Path(args.out)}: {len(result['endpoints'])} endpoint entries, {len(result['absolute'])} absolute entries, "
          f"{result['inputs']['n_units']} units")


def main(argv=None) -> None:
    guard = C.cpu_guard(2)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default=str(C.DEPLOY_ROOT / "work" / "plan" / "A.json"))
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=str(C.WORK_ROOT / "A"))
    p.add_argument("--units", help="comma list of unit ids (or prefixes) or level/enc/r/fold selectors")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=str(C.DEPLOY_ROOT / "results" / "A"))
    p.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    args.func(args, guard)


if __name__ == "__main__":
    main()

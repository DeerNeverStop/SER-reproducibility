"""
ser_v2.verify — INDEPENDENT VERIFIER for the SER v2 scoring contract.

This module was written from ``SPEC_SCORING_CONTRACT.md`` (frozen at tag-1, with the
two amendments: bootstrap seed 20260903 and integrity sub-check I2b) and from the
registry files only.  It deliberately shares NO code with the scorer
(``ser_v2/score.py``) or with any other ``ser_v2`` module: every loader, every
integrity check and every statistic below is re-implemented from the specification
text and from the documented file formats.  Only the Python standard library, numpy
and scipy are used.

What it does
------------
1. Loads the plan (run_plan.csv, split_index.json, unit_configs.json, splits/*.json),
   the manifests (re-applying hygiene rules H1/H2/H3 and checking the result against
   plan/hygiene_log.json), the registry and every unit of a run.
2. Performs integrity checks I1-I7 (incl. I2b) exactly as specified in Section 2 —
   failures are reported, never patched.
3. Builds cell-runs, per-speaker UARs and replicate averages (Section 3), the
   contrasts of Section 4 (N01-N15, R01-R11, T01-T02), the statistics of Section 5
   (speaker bootstrap, Wilcoxon, sign test, Holm, TOST, directional reading), the
   claim verdicts of claim_map.json and descriptors D01-D09 and D14.
4. Writes its own full result object (Section 6 schema) to RUN_DIR/verifier_results.json.
5. If ``--results`` (the scorer's results.json) is given, compares item by item and
   writes verification.json (Section 7).  results.json is never modified.

Exit codes: 0 = everything passed; 1 = any comparison mismatch or integrity failure;
2 = fatal input error.

Interpretations of under-specified points are marked with ``INTERPRETATION`` comments
and are listed in the ``verifier.interpretations`` block of the output.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats as sps

# --------------------------------------------------------------------------------------
# Constants fixed by the specification / registry
# --------------------------------------------------------------------------------------
SCHEMA = "ser-v2-results-1"
VERIFICATION_SCHEMA = "ser-v2-verification-1"
SPEC_ID = "SPEC_SCORING_CONTRACT.md (tag-1; amendments: seed 20260903, I2b)"

BOOT_SEED = 20260903          # Section 5 (amended)
N_BOOT = 10000                # Section 5
ALPHA = 0.05                  # Section 5 / claim_map alpha
TOL = 1e-12                   # Section 7 comparison tolerance (absolute)
D03_BAND = 3.0                # D03 reading band (pp)

FAMILY_LEVELS = ("ravdess", "cremad", "subesco_980")
SCRATCH_FAMILY = ("cnn", "resnet_se", "transformer")
SCRATCH_SECONDARY = ("cnn", "resnet_se")
PROBE_MODELS = ("hubert_base", "wavlm_base_plus", "wav2vec2_base")
FT_MODEL = "wavlm_base_plus_ft"
FROZEN_MODEL = "wavlm_base_plus_frozen_sr"
RIDGE_MODELS = ("ridge_a1_hubert_base", "ridge_a1_wavlm_base_plus", "ridge_a1_wav2vec2_base")
# INTERPRETATION: the mechanism block (arm MECH2X2) contains no transformer units and the
# registry lists "cnn,resnet_se" for N05-N09/T02, so its scratch group is {cnn, resnet_se}
# on both of its levels (cremad is a family level, but the family rule of Section 3 is read
# as applying to the CTRL factorial only).
MECH_SCRATCH = ("cnn", "resnet_se")
MECH_CELLS = ("none", "spk", "prm", "both", "both_sib", "spk_half_h1", "spk_half_h2")
HPO_CELLS = ("GR_hpo", "GG_hpo", "RR_hpo")
HPO_N_CONFIGS = 8
HPO_ARM, CTRL_ARM, MECH_ARM, FT_ARM, PROBECPU_ARM = "HPO", "CTRL", "MECH2X2", "FT", "PROBECPU"
SEED_EQ_R_ARMS = (CTRL_ARM, MECH_ARM, HPO_ARM)      # Section 3: main replicate units
# H3 (hygiene): the registered unusable file, per plan/hygiene_log.json and the task brief.
REGISTERED_UNUSABLE = {"cremad": ("1076_MTI_SAD_XX.wav",)}
PRED_FIXED_COLUMNS = ("sample_index", "relative_path", "speaker", "y_true", "y_pred")

HYP_COMPARE_FIELDS = ("n", "n_nonzero", "mean", "median", "sd", "ci_low", "ci_high",
                      "wilcoxon_p", "sign_test_p", "p_holm", "reject", "verdict")
TOST_COMPARE_FIELDS = HYP_COMPARE_FIELDS + ("p_low", "p_high", "p", "equivalent", "ci90")

INTERPRETATIONS = [
    "I5 is applied as amended: COMPLETE iff the OOF path multiset equals, exactly once, the union "
    "of the test lists of ALL folds of the cell-run's split table (its test population). The "
    "per-cell-run flag covers_population (test population == population) is informational only "
    "and is never used for completeness (it is False for MECH2X2 tables by design).",
    "I3b: a split table whose population contains paths outside the hygiene-filtered manifest is "
    "reported under I3 (naming the key) and the units referencing it are treated as unusable.",
    "Units failing any of I1/I2/I2b/I3/I3b/I4/I7 are treated as unusable: a cell-run that needs "
    "them is VOID (never patched, never silently used).",
    "Families and membership come from the registry: `family` column; only rows with "
    "a_priori_status == 'confirmatory' enter the Holm family of that label; m = the family_size "
    "column (cross-checked against the confirmatory count and claim_map.family_sizes; a mismatch "
    "is an I0 failure). Rows with a_priori_status == 'estimate' are computed and reported like the "
    "others, with p_holm = null, reject = false and verdict 'estimate (not tested)' ('not tested' "
    "when not computable); this applies to T rows too (TOST p reported, no Holm).",
    "Claims of type 'estimation' (claim_map.json) get verdict 'reported (estimation claim)' "
    "regardless of their (possibly empty) requirement list; hypotheses in `demoted_a_priori` / "
    "`reported` appear in the evidence block but never enter the rule.",
    "I5/I6 pass flags: a check passes when no cell-run with at least one done unit is VOID for "
    "that reason; void cell-runs are listed as failures.",
    "FT seed eligibility ('seed_index values present for BOTH models on that level') is applied "
    "per (level, cell): a seed enters the replicate average of V[L, ft|frozen_sr, cell] iff the "
    "cell-runs of both models for that seed and cell are COMPLETE.",
    "MECH half cells (T02, D06 G05): for each replicate r, exposed_half_r(s) is read from "
    "meta.exposed_speakers of the spk_half_h1 table of that r (fold f(s) = fold whose test rows "
    "contain s); U_{half}(s) at r is taken from cell-run (spk_half_<half>, r) when COMPLETE; the "
    "replicate average is over the r values where that value is defined, exactly like V.",
    "N14: f(s) is the unique fold whose OOF rows contain s under the respective HPO cell-run; a "
    "speaker present in more than one fold (impossible for G outer splits) would be skipped.",
    "N10/D07: P24(s)/P91(s) average over the draws for which the respective value is defined "
    "(truncated draws are simply absent); D07 SDs are sample SDs (ddof=1) over draws, null with "
    "fewer than two draws.",
    "Tie determinism: the Wilcoxon test, the sign test and n_nonzero use D rounded to 10 "
    "decimals (numpy.round(D, 10)); mean/median/sd/CI use the unrounded D.",
    "Not-tested hypotheses are written with n=0, n_nonzero=0, wilcoxon_p=1.0 (the p that enters "
    "the family), sign_test_p=null, mean/median/sd/ci=null, p_holm from the Holm pass, "
    "reject=false; not-tested T rows carry p=1.0, p_low=p_high=null, ci90=null. The verdict is "
    "'not tested (truncated, deviation entry N)' iff runs/<run_id>/deviations.jsonl has an entry "
    "whose 'hypotheses' list names the hypothesis (N = its 'entry' field, else the 1-based line "
    "number), otherwise the plain string 'not tested'; matching is on ids only. The extra field "
    "'reason' (missing / void / truncated by plan rank) is informational.",
    "Sign test with n_zero > 0.2 n but n_nonzero == 0 (binomtest undefined): sign_test_p = 1.0.",
    "sd is the sample SD (ddof=1); null when n < 2. TOST is 'not tested' when n < 2.",
    "Claims: every non-estimation claim (including type 'replication') follows the structural "
    "requires/requires_any/requires_k_of rule of claim_map.json as defined in Section 5; the "
    "evidence dict carries the per-test verdicts. requires_any blocks on any member that is "
    "'not supported' with a mean of the sign opposite to its registered direction. If a required "
    "member is not tested and the tested members alone do not satisfy the rule, the claim is "
    "'not tested', suffixed ' (truncated, deviation entry N)' only when deviations.jsonl names "
    "the claim or one of its untested members (lowest entry number). A non-estimation claim with "
    "an empty requirement list gets verdict 'no verdict'.",
    "D03 label: 'larger' iff ci_low > 3, 'smaller' iff ci_high < -3, else 'within_band'.",
    "D14: matrix entries need both cell-runs (RR and GG for that (r, seed)) COMPLETE; row/column "
    "means are over the available entries; sd_draw/sd_seed are sample SDs (ddof=1), null with "
    "fewer than two values.",
    "plan_sha256 = SHA-256 of plan/run_plan.csv bytes (matches plan_summary.run_plan_sha256); "
    "registry_sha256 = SHA-256 of registry/hypothesis_registry.csv bytes.",
    "n_units_done = units with unit.json.status == 'done'; n_units_void = done units that are "
    "unusable or belong to a VOID cell-run.",
    "The scorer's numeric_insert.tex (if present next to results.json) is checked best-effort: "
    "macros whose name spells a hypothesis id (digits as words) must match a field of that "
    "hypothesis after the stated rounding; other macros are matched against any value and "
    "reported for information only.",
]


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
class FatalInputError(Exception):
    """Raised for unreadable / structurally invalid inputs (exit code 2)."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise FatalInputError("cannot read JSON %s: %s" % (path, exc))


def canonical_json(obj) -> str:
    """Canonical JSON text used for I7 equality (sorted keys, minimal separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fnum(x):
    """Convert to a plain float (None for non-finite)."""
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def to_jsonable(o):
    """Recursively convert numpy scalars/arrays and non-finite floats for json.dumps."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return fnum(o)
    return o


def write_json(path: str, obj) -> None:
    # Floats are written by json with repr precision (no rounding).
    text = json.dumps(to_jsonable(obj), indent=1, allow_nan=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.write("\n")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Manifests + hygiene
# --------------------------------------------------------------------------------------
class Manifest:
    """A hygiene-filtered manifest of one base corpus."""

    def __init__(self, base: str, rows: list, dropped: list, n_input: int):
        self.base = base
        self.n_input = n_input
        self.dropped = dropped                       # [(relative_path, reason)]
        self.paths = [r["relative_path"] for r in rows]
        self.path_index = {p: i for i, p in enumerate(self.paths)}
        spk = [r["speaker"] for r in rows]
        self.speakers = sorted(set(spk))             # string sort
        code = {s: i for i, s in enumerate(self.speakers)}
        self.spk_code = np.array([code[s] for s in spk], dtype=np.int32)
        self.spk_of_code = self.speakers
        try:
            self.label_index = np.array([int(r["label_index"]) for r in rows], dtype=np.int64)
        except ValueError as exc:
            raise FatalInputError("manifest %s: non-integer label_index (%s)" % (base, exc))
        self.n_classes = int(len(set(self.label_index.tolist())))
        self.n_kept = len(rows)


def apply_hygiene(base: str, rows: list):
    """H1: drop every member of an exact-sha256 duplicate group whose labels conflict;
    H2: in a same-label duplicate group keep the lexically first path; H3: drop the
    registered unusable file(s)."""
    drop = {}
    by_sha = collections.defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r)
    for sha, grp in by_sha.items():
        if len(grp) < 2:
            continue
        labels = {g["label_index"] for g in grp}
        if len(labels) > 1:
            for g in grp:
                drop.setdefault(g["relative_path"], "H1 label-conflicting duplicate group %s" % sha[:12])
        else:
            keep = min(g["relative_path"] for g in grp)
            for g in grp:
                if g["relative_path"] != keep:
                    drop.setdefault(g["relative_path"],
                                    "H2 exact duplicate of lexically first path (%s)" % sha[:12])
    present = {r["relative_path"] for r in rows}
    for p in REGISTERED_UNUSABLE.get(base, ()):
        if p in present:
            drop.setdefault(p, "H3 registered unusable file")
    kept = [r for r in rows if r["relative_path"] not in drop]
    dropped = sorted(drop.items())
    return kept, dropped


def load_manifest(path: str, base: str) -> Manifest:
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise FatalInputError("cannot read manifest %s: %s" % (path, exc))
    needed = {"relative_path", "sha256", "speaker", "label_index"}
    if not rows or not needed.issubset(rows[0].keys()):
        raise FatalInputError("manifest %s lacks required columns %s" % (path, sorted(needed)))
    kept, dropped = apply_hygiene(base, rows)
    return Manifest(base, kept, dropped, len(rows))


# --------------------------------------------------------------------------------------
# Split tables
# --------------------------------------------------------------------------------------
class SplitTable:
    def __init__(self, key: str, obj: dict, manifest: Manifest, sha_ok: bool):
        self.key = key
        self.sha_ok = sha_ok
        self.population = list(obj.get("population", []))
        pop_idx = [manifest.path_index.get(p) for p in self.population]
        self.pop_missing = sum(1 for i in pop_idx if i is None)
        self.pop_idx = np.array(sorted(i for i in pop_idx if i is not None), dtype=np.int64)
        self.folds = {}                               # fold -> dict(test_idx, meta, n_missing, n_test)
        for f in obj.get("folds", []):
            fid = int(f["fold"])
            test = f.get("test", [])
            idx = [manifest.path_index.get(p) for p in test]
            missing = sum(1 for i in idx if i is None)
            arr = np.array(sorted(i for i in idx if i is not None), dtype=np.int64)
            self.folds[fid] = {"test_idx": arr, "meta": f.get("meta", {}) or {},
                               "n_missing": missing, "n_test": len(test),
                               "unique": int(len(np.unique(arr))) == len(arr)}
        self.fold_ids = sorted(self.folds)
        if self.folds:
            allidx = np.concatenate([self.folds[f]["test_idx"] for f in self.fold_ids])
            uniq, cnt = np.unique(allidx, return_counts=True)
            self.union_idx = uniq
            self.union_max_mult = int(cnt.max()) if len(cnt) else 0
        else:
            self.union_idx = np.array([], dtype=np.int64)
            self.union_max_mult = 0
        self.covers_population = (self.union_max_mult == 1 and self.pop_missing == 0
                                  and np.array_equal(self.union_idx, self.pop_idx))


# --------------------------------------------------------------------------------------
# Units and cell-runs
# --------------------------------------------------------------------------------------
class Unit:
    __slots__ = ("uid", "row", "arm", "level", "model", "cell", "fold", "r", "seed", "base",
                 "split_key", "usable", "idx", "yt", "yp", "val_uar_best", "hpo_index", "n_rows")

    def __init__(self, uid, row):
        self.uid = uid
        self.row = row
        self.arm = row["arm"]
        self.level = row["corpus_level"]
        self.model = row["model"]
        self.cell = row["cell"]
        self.fold = row["fold"]
        self.r = row["r"]
        self.seed = row["seed_index"]
        self.base = row["base_corpus"]
        self.split_key = None
        self.usable = True
        self.idx = self.yt = self.yp = None
        self.val_uar_best = None
        self.hpo_index = None
        self.n_rows = 0


def cell_run_key(arm, level, model, cell, r, seed) -> str:
    return "%s|%s|%s|%s|r%d|s%d" % (arm, level, model, cell, r, seed)


def uar_percent(yt: np.ndarray, yp: np.ndarray) -> float:
    """UAR in percent over the classes PRESENT in yt (Section 3)."""
    classes = np.unique(yt)
    recalls = []
    for c in classes:
        mc = yt == c
        recalls.append(float(np.count_nonzero(yp[mc] == c)) / float(np.count_nonzero(mc)))
    return 100.0 * float(np.mean(recalls))


class CellRun:
    def __init__(self, arm, level, model, cell, r, seed):
        self.arm, self.level, self.model, self.cell, self.r, self.seed = arm, level, model, cell, r, seed
        self.key = cell_run_key(arm, level, model, cell, r, seed)
        self.units = collections.defaultdict(list)    # fold -> [Unit] (usable + unusable)
        self.complete = False
        self.reasons = []
        self.split_key = None
        self.speaker_uar = {}
        self.fold_of_speaker = {}
        self.n_rows = 0
        self.n_folds_present = 0
        self.n_folds_expected = None
        self.covers_population = None
        # HPO extras
        self.selected = {}      # fold -> hpo_config_index
        self.val_sel = {}       # fold -> val_uar_best of the selected config
        self.test_sel = {}      # fold -> fold-level test UAR of the selected config
        self.test_max = {}      # fold -> max over configs of fold-level test UAR
        self.per_config_test_uar = {}  # fold -> {cfg: uar}

    @property
    def is_hpo(self):
        return self.arm == HPO_ARM

    def add(self, u: Unit):
        self.units[u.fold].append(u)


# --------------------------------------------------------------------------------------
# The verifier context: loads everything and runs the integrity checks
# --------------------------------------------------------------------------------------
class Context:
    def __init__(self, plan_dir, manifest_dir, run_dir, registry_dir):
        self.plan_dir = plan_dir
        self.manifest_dir = manifest_dir
        self.run_dir = run_dir
        self.registry_dir = registry_dir
        self.integrity = collections.OrderedDict(
            (k, {"pass": True, "failures": []}) for k in ("I0", "I1", "I2", "I3", "I4", "I5", "I6", "I7"))
        self.manifests = {}          # base -> Manifest | None (missing file)
        self.splits = {}             # key -> SplitTable
        self.split_base = {}         # key -> base corpus used to resolve paths
        self.units = {}              # uid -> Unit (status done only)
        self.n_units_done = 0
        self.n_units_not_done = 0
        self.cell_runs = {}          # key -> CellRun
        self.plan_by_cell = collections.defaultdict(list)   # (arm, level, model, cell) -> [rows]
        self.deviations = []         # [(entry_number, text, obj)]
        self.i2b_checked = set()

    # -- failure bookkeeping ----------------------------------------------------------
    def fail(self, check: str, message: str) -> None:
        item = self.integrity[check]
        item["pass"] = False
        item["failures"].append(message)

    # -- loading -----------------------------------------------------------------------
    def load_registry(self):
        path = os.path.join(self.registry_dir, "hypothesis_registry.csv")
        try:
            raw = read_bytes(path)
        except OSError as exc:
            raise FatalInputError("cannot read %s: %s" % (path, exc))
        self.registry_sha256 = sha256_bytes(raw)
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        self.hypotheses = collections.OrderedDict()
        for r in rows:
            hid = r["hypothesis_id"].strip()
            margin = None
            m = re.search(r"([0-9]*\.?[0-9]+)", r.get("sesoi_or_margin", "") or "")
            if m:
                margin = float(m.group(1))
            try:
                fam_size = int(r["family_size"])
            except (TypeError, ValueError):
                raise FatalInputError("hypothesis_registry.csv: bad family_size for %s" % hid)
            ph = (r.get("power_holm") or "").strip()
            self.hypotheses[hid] = {
                "family": r["family"].strip(),
                "m": fam_size,
                "direction": (r.get("direction") or "").strip(),
                "claim_id": (r.get("claim_id") or "").strip(),
                "margin": margin,
                "a_priori_status": (r.get("a_priori_status") or "").strip(),
                "power_holm": float(ph) if ph else None,
                "corpus_level": r.get("corpus_level", ""),
                "model_levels": r.get("model_levels", ""),
            }
        self.claim_map = load_json(os.path.join(self.registry_dir, "claim_map.json"))
        arms_path = os.path.join(self.registry_dir, "arms.json")
        self.arms = load_json(arms_path) if os.path.exists(arms_path) else {}
        # Family sizes: m = confirmatory count per family label (family_size column), cross-checked.
        conf = collections.Counter(h["family"] for h in self.hypotheses.values() if h["a_priori_status"] == "confirmatory")
        self.family_m = {}
        for hid, h in self.hypotheses.items():
            fam = h["family"]
            self.family_m.setdefault(fam, h["m"])
            if h["m"] != self.family_m[fam]:
                self.fail("I0", "registry: family_size of %s (%d) differs within family %s (%d)" % (hid, h["m"], fam, self.family_m[fam]))
        for fam, m in self.family_m.items():
            if conf.get(fam, 0) != m:
                self.fail("I0", "registry: family %s family_size=%d but %d confirmatory rows" % (fam, m, conf.get(fam, 0)))
        for fam, m in (self.claim_map.get("family_sizes") or {}).items():
            if fam in self.family_m and self.family_m[fam] != int(m):
                self.fail("I0", "claim_map family_sizes[%s]=%s != registry family_size %d" % (fam, m, self.family_m[fam]))

    def load_plan(self):
        path = os.path.join(self.plan_dir, "run_plan.csv")
        try:
            raw = read_bytes(path)
        except OSError as exc:
            raise FatalInputError("cannot read %s: %s" % (path, exc))
        self.plan_sha256 = sha256_bytes(raw)
        self.plan = collections.OrderedDict()
        for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
            try:
                for k in ("fold", "r", "seed_index"):
                    r[k] = int(r[k])
                r["truncation_rank"] = int(r.get("truncation_rank") or 99)
            except ValueError as exc:
                raise FatalInputError("run_plan.csv: bad integer field in unit %s (%s)" % (r.get("unit_id"), exc))
            self.plan[r["unit_id"]] = r
            self.plan_by_cell[(r["arm"], r["corpus_level"], r["model"], r["cell"])].append(r)
        self.split_index = load_json(os.path.join(self.plan_dir, "split_index.json"))
        self.sha2key = {}
        for k, v in self.split_index.items():
            self.sha2key.setdefault(v["sha256"], k)
        self.unit_configs = load_json(os.path.join(self.plan_dir, "unit_configs.json"))
        hl = os.path.join(self.plan_dir, "hygiene_log.json")
        self.hygiene_log = load_json(hl) if os.path.exists(hl) else None
        self.levels = sorted({r["corpus_level"] for r in self.plan.values()})
        self.level_base = {}
        for r in self.plan.values():
            self.level_base.setdefault(r["corpus_level"], r["base_corpus"])
        # (arm, level, cell, r, seed) -> split key, from the plan (used for MECH meta lookups)
        self.plan_split_key = {}
        for r in self.plan.values():
            key = self.sha2key.get(r["split_sha256"])
            self.plan_split_key.setdefault((r["arm"], r["corpus_level"], r["cell"], r["r"], r["seed_index"]), key)

    def load_manifests(self):
        bases = sorted({r["base_corpus"] for r in self.plan.values()})
        for base in bases:
            path = os.path.join(self.manifest_dir, "%s_manifest.csv" % base)
            if not os.path.exists(path):
                self.manifests[base] = None
                continue
            man = load_manifest(path, base)
            self.manifests[base] = man
            # Hygiene cross-check against plan/hygiene_log.json (task brief: mismatch = failure).
            if self.hygiene_log is not None:
                entry = self.hygiene_log.get(base)
                if entry is None:
                    self.fail("I0", "hygiene_log.json has no entry for base corpus %s" % base)
                else:
                    if int(entry.get("n_kept", -1)) != man.n_kept:
                        self.fail("I0", "hygiene: %s n_kept %d != hygiene_log n_kept %s"
                                  % (base, man.n_kept, entry.get("n_kept")))
                    if int(entry.get("n_input", -1)) != man.n_input:
                        self.fail("I0", "hygiene: %s n_input %d != hygiene_log n_input %s"
                                  % (base, man.n_input, entry.get("n_input")))
                    logged = {d["relative_path"] for d in entry.get("dropped", [])}
                    mine = {p for p, _ in man.dropped}
                    if logged != mine:
                        self.fail("I0", "hygiene: %s dropped set differs from hygiene_log (mine-only=%s, log-only=%s)"
                                  % (base, sorted(mine - logged), sorted(logged - mine)))
        if self.hygiene_log is not None:
            for base in self.hygiene_log:
                if base not in self.manifests:
                    self.fail("I0", "hygiene_log.json lists %s which no plan row uses" % base)

    def load_deviations(self):
        path = os.path.join(self.run_dir, "deviations.jsonl")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    obj = {"raw": line}
                # entry number = the entry's "entry" field, else its 1-based line number
                num = obj["entry"] if isinstance(obj, dict) and isinstance(obj.get("entry"), int) else i
                self.deviations.append((num, line, obj))

    def get_split(self, key: str, base: str) -> SplitTable:
        st = self.splits.get(key)
        if st is not None:
            return st
        info = self.split_index.get(key)
        if info is None:
            raise FatalInputError("split key %s not in split_index.json" % key)
        path = os.path.join(self.plan_dir, info["path"])
        try:
            raw = read_bytes(path)
        except OSError as exc:
            raise FatalInputError("cannot read split table %s: %s" % (path, exc))
        sha_ok = sha256_bytes(raw) == info["sha256"]
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise FatalInputError("split table %s is not valid JSON: %s" % (path, exc))
        man = self.manifests.get(base)
        if man is None:
            raise FatalInputError("split table %s needs manifest for base %s which is missing" % (key, base))
        st = SplitTable(key, obj, man, sha_ok)
        self.splits[key] = st
        self.split_base[key] = base
        return st

    # -- units -------------------------------------------------------------------------
    def load_units(self):
        udir = os.path.join(self.run_dir, "units")
        if not os.path.isdir(udir):
            raise FatalInputError("run directory has no units/ folder: %s" % udir)
        names = sorted(os.listdir(udir))
        log("loading %d unit directories ..." % len(names))
        for n, uid in enumerate(names, start=1):
            d = os.path.join(udir, uid)
            if not os.path.isdir(d):
                continue
            self._load_unit(uid, d)
            if n % 500 == 0:
                log("  %d/%d units" % (n, len(names)))
        log("units: %d done (%d not done), %d usable" % (
            self.n_units_done, self.n_units_not_done, sum(1 for u in self.units.values() if u.usable)))

    def _load_unit(self, uid: str, d: str):
        ujpath = os.path.join(d, "unit.json")
        if not os.path.exists(ujpath):
            self.fail("I0", "unit %s: no unit.json" % uid)
            return
        try:
            uj = load_json(ujpath)
        except FatalInputError as exc:
            self.fail("I0", "unit %s: %s" % (uid, exc))
            return
        if uj.get("status") != "done":
            self.n_units_not_done += 1
            return
        self.n_units_done += 1
        row = self.plan.get(uid)
        if row is None:
            self.fail("I0", "unit %s is done but not in run_plan.csv" % uid)
            return
        u = Unit(uid, row)
        self.units[uid] = u
        # plan/unit.json agreement on the identity fields (membership uses the plan row)
        for k in ("arm", "corpus_level", "model", "cell", "fold", "r", "seed_index"):
            if k in uj and uj[k] != row[k]:
                self.fail("I0", "unit %s: unit.json %s=%r differs from plan %r" % (uid, k, uj[k], row[k]))
        try:
            u.val_uar_best = float(uj["val_uar_best"]) if uj.get("val_uar_best") is not None else None
        except (TypeError, ValueError):
            u.val_uar_best = None

        # ---- I7: config equality ------------------------------------------------------
        plan_cfg = self.unit_configs.get(row["config_sha256"])
        if plan_cfg is None:
            self.fail("I7", "unit %s: plan config_sha256 %s not in unit_configs.json" % (uid, row["config_sha256"]))
            u.usable = False
        elif canonical_json(uj.get("config")) != canonical_json(plan_cfg):
            self.fail("I7", "unit %s: unit.json.config differs from unit_configs[%s]" % (uid, row["config_sha256"][:12]))
            u.usable = False
        if plan_cfg is not None and u.arm == HPO_ARM:
            hi = plan_cfg.get("hpo_config_index")
            if isinstance(hi, int):
                u.hpo_index = hi
            else:
                self.fail("I6", "unit %s: HPO unit config lacks hpo_config_index" % uid)
                u.usable = False

        # ---- I1: DONE == unit.json.predictions_sha256 == sha256(predictions.csv) -------
        ppath = os.path.join(d, "predictions.csv")
        if not os.path.exists(ppath):
            self.fail("I1", "unit %s: predictions.csv missing" % uid)
            u.usable = False
            return
        raw = read_bytes(ppath)
        sha = sha256_bytes(raw)
        dpath = os.path.join(d, "DONE")
        done_txt = None
        if os.path.exists(dpath):
            done_txt = read_bytes(dpath).decode("utf-8", "replace").strip()
        else:
            self.fail("I1", "unit %s: DONE file missing" % uid)
            u.usable = False
        if done_txt is not None and not (done_txt == uj.get("predictions_sha256") == sha):
            self.fail("I1", "unit %s: DONE=%s unit.json=%s sha256(predictions.csv)=%s"
                      % (uid, done_txt[:12], str(uj.get("predictions_sha256"))[:12], sha[:12]))
            u.usable = False

        # ---- I2: split sha agreement + I2b split table bytes ------------------------
        if uj.get("split_sha256") != row["split_sha256"]:
            self.fail("I2", "unit %s: unit.json.split_sha256 %s != plan %s"
                      % (uid, str(uj.get("split_sha256"))[:12], row["split_sha256"][:12]))
            u.usable = False
        key = self.sha2key.get(row["split_sha256"])
        if key is None:
            self.fail("I2", "unit %s: plan split_sha256 %s not in split_index.json" % (uid, row["split_sha256"][:12]))
            u.usable = False
            return
        u.split_key = key
        man = self.manifests.get(u.base)
        if man is None:
            self.fail("I3", "unit %s: no manifest for base corpus %s (level %s)" % (uid, u.base, u.level))
            u.usable = False
            return
        st = self.get_split(key, u.base)
        if key not in self.i2b_checked:
            self.i2b_checked.add(key)
            if not st.sha_ok:
                self.fail("I2", "I2b: split table %s bytes do not hash to split_index sha256" % key)
            if st.pop_missing:
                self.fail("I3", "I3b: split table %s: %d population paths not in hygiene-filtered manifest %s"
                          % (key, st.pop_missing, u.base))
        if not st.sha_ok or st.pop_missing:
            u.usable = False

        # ---- parse predictions --------------------------------------------------------
        parsed = self._parse_predictions(uid, raw, man)
        if parsed is None:
            u.usable = False
            return
        paths, spk, yt, yp, logits = parsed
        n = len(paths)
        u.n_rows = n

        # ---- I4: duplicates + argmax + logit width -------------------------------------
        if len(set(paths)) != n:
            self.fail("I4", "unit %s: duplicate relative_path in predictions.csv" % uid)
            u.usable = False
        if logits.shape[1] != man.n_classes:
            self.fail("I4", "unit %s: %d logit columns but n_classes=%d" % (uid, logits.shape[1], man.n_classes))
            u.usable = False
        if n:
            am = np.argmax(logits, axis=1)          # ties -> lowest index
            bad = int(np.count_nonzero(am != yp))
            if bad:
                self.fail("I4", "unit %s: y_pred != argmax(logits) in %d rows" % (uid, bad))
                u.usable = False

        # ---- I3: manifest agreement -------------------------------------------------
        idx = np.array([man.path_index.get(p, -1) for p in paths], dtype=np.int64)
        unknown = int(np.count_nonzero(idx < 0))
        if unknown:
            self.fail("I3", "unit %s: %d relative_path values not in hygiene-filtered manifest %s" % (uid, unknown, u.base))
            u.usable = False
        known = idx >= 0
        if np.any(known):
            bad_y = int(np.count_nonzero(yt[known] != man.label_index[idx[known]]))
            if bad_y:
                self.fail("I3", "unit %s: y_true != manifest label_index in %d rows" % (uid, bad_y))
                u.usable = False
            man_spk = [man.spk_of_code[c] for c in man.spk_code[idx[known]]]
            pred_spk = [s for s, k in zip(spk, known) if k]
            bad_s = sum(1 for a, b in zip(man_spk, pred_spk) if a != b)
            if bad_s:
                self.fail("I3", "unit %s: speaker != manifest speaker in %d rows" % (uid, bad_s))
                u.usable = False

        # ---- I2 (second half): path set == fold test list -----------------------------
        fold = st.folds.get(u.fold)
        if fold is None:
            self.fail("I2", "unit %s: fold %d not in split table %s" % (uid, u.fold, key))
            u.usable = False
        else:
            if fold["n_missing"]:
                self.fail("I2", "split table %s fold %d: %d test paths not in manifest" % (key, u.fold, fold["n_missing"]))
            if unknown or not np.array_equal(np.unique(idx[known]), fold["test_idx"]) or fold["n_missing"]:
                self.fail("I2", "unit %s: set(relative_path) != test list of fold %d in %s" % (uid, u.fold, key))
                u.usable = False
        u.idx = idx.astype(np.int32)
        u.yt = yt.astype(np.int16)
        u.yp = yp.astype(np.int16)

    def _parse_predictions(self, uid, raw: bytes, man: Manifest):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self.fail("I4", "unit %s: predictions.csv is not UTF-8" % uid)
            return None
        if '"' in text:
            rows = list(csv.reader(io.StringIO(text)))
            rows = [r for r in rows if r]
        else:
            rows = [ln.split(",") for ln in text.splitlines() if ln]
        if not rows:
            self.fail("I4", "unit %s: predictions.csv is empty" % uid)
            return None
        header = [h.strip() for h in rows[0]]
        k = len(header) - len(PRED_FIXED_COLUMNS)
        expected = list(PRED_FIXED_COLUMNS) + ["logit_%d" % i for i in range(max(k, 0))]
        if header != expected:
            self.fail("I4", "unit %s: unexpected predictions.csv header %s" % (uid, header[:8]))
            return None
        body = rows[1:]
        width = len(header)
        if any(len(r) != width for r in body):
            self.fail("I4", "unit %s: ragged rows in predictions.csv" % uid)
            return None
        try:
            paths = [r[1] for r in body]
            spk = [r[2] for r in body]
            yt = np.array([int(r[3]) for r in body], dtype=np.int64)
            yp = np.array([int(r[4]) for r in body], dtype=np.int64)
            logits = np.array([r[5:] for r in body], dtype=np.float64).reshape(len(body), k)
        except ValueError as exc:
            self.fail("I4", "unit %s: malformed numeric field in predictions.csv (%s)" % (uid, exc))
            return None
        return paths, spk, yt, yp, logits

    # -- cell-runs -----------------------------------------------------------------------
    def build_cell_runs(self):
        for u in self.units.values():
            cr = self.cell_runs.get(cell_run_key(u.arm, u.level, u.model, u.cell, u.r, u.seed))
            if cr is None:
                cr = CellRun(u.arm, u.level, u.model, u.cell, u.r, u.seed)
                self.cell_runs[cr.key] = cr
            cr.add(u)
        for key in sorted(self.cell_runs):
            self._evaluate_cell_run(self.cell_runs[key])
        n_ok = sum(1 for c in self.cell_runs.values() if c.complete)
        log("cell-runs: %d complete, %d void" % (n_ok, len(self.cell_runs) - n_ok))

    def _evaluate_cell_run(self, cr: CellRun):
        units = [u for us in cr.units.values() for u in us]
        man = self.manifests.get(units[0].base)
        keys = {u.split_key for u in units if u.split_key}
        if len(keys) != 1 or man is None:
            cr.reasons.append("units reference %d split tables / manifest missing" % len(keys))
            self.fail("I5", "%s VOID: %s" % (cr.key, cr.reasons[-1]))
            return
        cr.split_key = keys.pop()
        st = self.splits[cr.split_key]
        cr.n_folds_expected = len(st.fold_ids)
        cr.covers_population = bool(st.covers_population)
        unusable = [u.uid for u in units if not u.usable]
        if unusable:
            cr.reasons.append("%d unit(s) failed integrity checks" % len(unusable))
        chosen = {}   # fold -> Unit whose predictions enter the OOF set
        missing_folds = []
        for f in st.fold_ids:
            us = [u for u in cr.units.get(f, []) if u.usable]
            if cr.is_hpo:
                by_cfg = {}
                for u in us:
                    if u.hpo_index in by_cfg:
                        cr.reasons.append("fold %d: duplicate config %d" % (f, u.hpo_index))
                    by_cfg[u.hpo_index] = u
                if sorted(by_cfg) != list(range(HPO_N_CONFIGS)):
                    missing_folds.append(f)
                    continue
                # HPO selection: largest val_uar_best; ties -> smallest hpo_config_index
                order = sorted(by_cfg.values(), key=lambda u: (-(u.val_uar_best if u.val_uar_best is not None else -math.inf), u.hpo_index))
                sel = order[0]
                chosen[f] = sel
                cr.selected[f] = sel.hpo_index
                cr.val_sel[f] = sel.val_uar_best
                per = {u.hpo_index: uar_percent(u.yt.astype(np.int64), u.yp.astype(np.int64)) for u in by_cfg.values()}
                cr.per_config_test_uar[f] = per
                cr.test_sel[f] = per[sel.hpo_index]
                cr.test_max[f] = max(per.values())
            else:
                if len(us) == 0:
                    missing_folds.append(f)
                    continue
                if len(us) > 1:
                    cr.reasons.append("fold %d has %d usable units" % (f, len(us)))
                chosen[f] = us[0]
        cr.n_folds_present = len(chosen)
        if missing_folds:
            if cr.is_hpo:
                cr.reasons.append("I6: folds without all %d configs: %s" % (HPO_N_CONFIGS, missing_folds))
                self.fail("I6", "%s VOID: %s" % (cr.key, cr.reasons[-1]))
            else:
                cr.reasons.append("missing folds %s" % missing_folds)
        extra = sorted(set(cr.units) - set(st.fold_ids))
        if extra:
            cr.reasons.append("units for folds %s not in split table" % extra)
        # I5: union of the folds' test paths equals the population exactly once
        if chosen:
            idx = np.concatenate([chosen[f].idx.astype(np.int64) for f in sorted(chosen)])
            uniq, cnt = np.unique(idx, return_counts=True)
            if cnt.size and cnt.max() > 1:
                cr.reasons.append("I5: %d paths appear more than once in the OOF set" % int(np.count_nonzero(cnt > 1)))
            if not np.array_equal(uniq, st.union_idx):
                cr.reasons.append("I5: OOF path set != union of the split table's test lists")
        if cr.reasons:
            if not cr.is_hpo or not any(r.startswith("I6") for r in cr.reasons):
                self.fail("I5", "%s VOID: %s" % (cr.key, "; ".join(cr.reasons)))
            return
        cr.complete = True
        yt = np.concatenate([chosen[f].yt for f in sorted(chosen)]).astype(np.int64)
        yp = np.concatenate([chosen[f].yp for f in sorted(chosen)]).astype(np.int64)
        idx = np.concatenate([chosen[f].idx for f in sorted(chosen)]).astype(np.int64)
        folds = np.concatenate([np.full(len(chosen[f].idx), f, dtype=np.int64) for f in sorted(chosen)])
        codes = man.spk_code[idx]
        cr.n_rows = int(len(idx))
        for c in np.unique(codes):
            m = codes == c
            s = man.spk_of_code[int(c)]
            cr.speaker_uar[s] = uar_percent(yt[m], yp[m])
            fs = np.unique(folds[m])
            cr.fold_of_speaker[s] = int(fs[0]) if len(fs) == 1 else None

    # -- helpers for the contrasts ---------------------------------------------------------
    def complete_runs(self, arm, level, model, cell):
        return [c for c in self.cell_runs.values()
                if c.complete and c.arm == arm and c.level == level and c.model == model and c.cell == cell]

    def run(self, arm, level, model, cell, r, seed):
        c = self.cell_runs.get(cell_run_key(arm, level, model, cell, r, seed))
        return c if (c is not None and c.complete) else None


# --------------------------------------------------------------------------------------
# Statistics (Section 5)
# --------------------------------------------------------------------------------------
def bootstrap_ci(D: np.ndarray):
    rng = np.random.RandomState(BOOT_SEED)
    n = len(D)
    idx = rng.randint(0, n, size=(N_BOOT, n))
    means = D[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], method="linear")
    return float(lo), float(hi)


def contrast_stats(dmap):
    """Base statistics of a per-speaker contrast; dmap = {speaker: D(s)} or None."""
    if not dmap:
        return {"n": 0, "n_nonzero": 0, "mean": None, "median": None, "sd": None,
                "ci_low": None, "ci_high": None, "wilcoxon_p": 1.0, "sign_test_p": None,
                "speakers": [], "d": []}
    speakers = sorted(dmap)                                  # speaker id string sort
    D = np.array([float(dmap[s]) for s in speakers], dtype=np.float64)
    n = int(len(D))
    mean = float(np.mean(D))
    median = float(np.median(D))
    sd = float(np.std(D, ddof=1)) if n > 1 else None
    lo, hi = bootstrap_ci(D)
    # Tie determinism (Section 5 clarification): the rank-based tests and the zero count use
    # D rounded to 10 decimals; mean/median/sd/CI above stay on the unrounded vector.
    D_r = np.round(D, 10)
    nz = D_r[D_r != 0]
    n_nonzero = int(len(nz))
    n_zero = n - n_nonzero
    if n_nonzero > 0:
        wp = float(sps.wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue)
    else:
        wp = 1.0
    sign_p = None
    if n_zero > 0.2 * n:
        n_pos = int(np.count_nonzero(D_r > 0))
        sign_p = float(sps.binomtest(n_pos, n_nonzero, 0.5, alternative="two-sided").pvalue) if n_nonzero > 0 else 1.0
    return {"n": n, "n_nonzero": n_nonzero, "mean": mean, "median": median, "sd": sd,
            "ci_low": lo, "ci_high": hi, "wilcoxon_p": wp, "sign_test_p": sign_p,
            "speakers": speakers, "d": [float(x) for x in D]}


def holm(pvals: "collections.OrderedDict[str, float]", m: int):
    """Holm step-down: returns ({id: p_holm}, {id: reject})."""
    order = sorted(pvals.keys(), key=lambda k: (pvals[k], list(pvals).index(k)))
    adj, rej = {}, {}
    running = 0.0
    stopped = False
    for i, hid in enumerate(order):
        j = i + 1
        val = min(1.0, (m - j + 1) * pvals[hid])
        running = max(running, val)
        adj[hid] = float(running)
        if not stopped and running <= ALPHA:
            rej[hid] = True
        else:
            stopped = True
            rej[hid] = False
    return adj, rej


def tost(D: np.ndarray, margin: float):
    n = len(D)
    mean = float(np.mean(D))
    se = float(np.std(D, ddof=1)) / math.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = (mean + margin) / se if se > 0 else (math.inf if mean + margin > 0 else (-math.inf if mean + margin < 0 else math.nan))
        t_high = (mean - margin) / se if se > 0 else (math.inf if mean - margin > 0 else (-math.inf if mean - margin < 0 else math.nan))
        p_low = 1.0 - float(sps.t.cdf(t_low, n - 1))
        p_high = float(sps.t.cdf(t_high, n - 1))
    if math.isnan(p_low):
        p_low = 1.0
    if math.isnan(p_high):
        p_high = 1.0
    p = max(p_low, p_high)
    if se > 0:
        lo, hi = sps.t.interval(0.90, n - 1, loc=mean, scale=se)
        ci90 = [float(lo), float(hi)]
    else:
        ci90 = [mean, mean]
    return {"p_low": p_low, "p_high": p_high, "p": p, "ci90": ci90, "se": se, "margin": margin}


def dmean(values):
    return float(np.mean(np.array(values, dtype=np.float64)))


# --------------------------------------------------------------------------------------
# Contrasts (Sections 3-4)
# --------------------------------------------------------------------------------------
class Contrasts:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._v = {}
        self._ft_seeds = {}

    # ---- replicate averages V ----------------------------------------------------------
    def ft_seeds(self, level, cell):
        """Seeds present (COMPLETE) for BOTH ft and frozen_sr on (level, cell) — Section 3."""
        k = (level, cell)
        if k not in self._ft_seeds:
            a = {c.seed for c in self.ctx.complete_runs(FT_ARM, level, FT_MODEL, cell)}
            b = {c.seed for c in self.ctx.complete_runs(FT_ARM, level, FROZEN_MODEL, cell)}
            self._ft_seeds[k] = a & b
        return self._ft_seeds[k]

    def V(self, arm, level, model, cell, missing=None):
        key = (arm, level, model, cell)
        if key not in self._v:
            runs = self.ctx.complete_runs(arm, level, model, cell)
            if arm in SEED_EQ_R_ARMS:
                runs = [c for c in runs if c.seed == c.r]
            elif arm == FT_ARM:
                elig = self.ft_seeds(level, cell)
                runs = [c for c in runs if c.seed in elig]
            if not runs:
                self._v[key] = None
            else:
                acc = collections.defaultdict(list)
                for c in runs:
                    for s, u in c.speaker_uar.items():
                        acc[s].append(u)
                self._v[key] = {s: dmean(v) for s, v in acc.items()}
        v = self._v[key]
        if v is None and missing is not None:
            missing.append(key)
        return v

    def Vg(self, arm, level, models, cell, missing=None):
        vs = [self.V(arm, level, m, cell, missing) for m in models]
        if any(v is None for v in vs):
            return None
        common = set(vs[0])
        for v in vs[1:]:
            common &= set(v)
        return {s: dmean([v[s] for v in vs]) for s in common}

    @staticmethod
    def diff(A, B):
        if A is None or B is None:
            return None
        return {s: A[s] - B[s] for s in A if s in B}

    @staticmethod
    def mean_maps(maps):
        """Per-speaker mean over a list of maps (all maps must be defined; common speakers)."""
        if not maps or any(m is None for m in maps):
            return None
        common = set(maps[0])
        for m in maps[1:]:
            common &= set(m)
        return {s: dmean([m[s] for m in maps]) for s in common}

    def scratch(self, level):
        return SCRATCH_FAMILY if level in FAMILY_LEVELS else SCRATCH_SECONDARY

    # ---- generic contrasts -------------------------------------------------------------
    def premium(self, arm, level, models, cA, cB, missing):
        return self.diff(self.Vg(arm, level, models, cA, missing), self.Vg(arm, level, models, cB, missing))

    def mech(self, level, cA, cB, missing):
        return self.premium(MECH_ARM, level, MECH_SCRATCH, cA, cB, missing)

    def ft_did(self, level, missing):
        ft = self.diff(self.V(FT_ARM, level, FT_MODEL, "RR", missing), self.V(FT_ARM, level, FT_MODEL, "GG", missing))
        fr = self.diff(self.V(FT_ARM, level, FROZEN_MODEL, "RR", missing), self.V(FT_ARM, level, FROZEN_MODEL, "GG", missing))
        return self.diff(ft, fr)

    # ---- N10 / D07 ------------------------------------------------------------------------
    def draws(self):
        ks = set()
        for lv in self.ctx.levels:
            m = re.match(r"^cremad_24_d(\d+)$", lv)
            if m:
                ks.add(int(m.group(1)))
        return sorted(ks)

    def panel_premia(self, missing):
        """Returns ({k: P24_k map}, {k: P91_k map})."""
        p24, p91 = {}, {}
        for k in self.draws():
            d24 = self.premium(CTRL_ARM, "cremad_24_d%d" % k, SCRATCH_SECONDARY, "RR", "GG", missing)
            d91 = self.premium(CTRL_ARM, "cremad_91m_d%d" % k, SCRATCH_SECONDARY, "RR", "GG", missing)
            if d24 is not None:
                p24[k] = d24
            if d91 is not None:
                p91[k] = d91
        return p24, p91

    def n10(self, missing):
        p24, p91 = self.panel_premia(missing)
        acc24, acc91 = collections.defaultdict(list), collections.defaultdict(list)
        for k, m in p24.items():
            for s, v in m.items():
                acc24[s].append(v)
        for k, m in p91.items():
            for s, v in m.items():
                acc91[s].append(v)
        D = {s: dmean(acc24[s]) - dmean(acc91[s]) for s in acc24 if s in acc91}
        return D if D else None

    # ---- MECH half cells ------------------------------------------------------------------
    def mech_half(self, level, model, which, missing):
        """V of the virtual cell 'exposed half' / 'unexposed half' (Section 3 mechanism block)."""
        ctx = self.ctx
        rs = sorted({c.r for c in ctx.cell_runs.values() if c.arm == MECH_ARM and c.level == level
                     and c.model == model and c.cell in ("spk_half_h1", "spk_half_h2")})
        acc = collections.defaultdict(list)
        man = ctx.manifests.get(ctx.level_base.get(level))
        for r in rs:
            key = ctx.plan_split_key.get((MECH_ARM, level, "spk_half_h1", r, r))
            if key is None or man is None:
                continue
            st = ctx.get_split(key, man.base)
            runs = {h: ctx.run(MECH_ARM, level, model, "spk_half_" + h, r, r) for h in ("h1", "h2")}
            for f in st.fold_ids:
                fold = st.folds[f]
                exposed = set(fold["meta"].get("exposed_speakers", []))
                test_speakers = {man.spk_of_code[int(c)] for c in np.unique(man.spk_code[fold["test_idx"]])}
                for s in test_speakers:
                    exposed_half = "h1" if s in exposed else "h2"
                    half = exposed_half if which == "exposed" else ("h2" if exposed_half == "h1" else "h1")
                    c = runs[half]
                    if c is not None and s in c.speaker_uar:
                        acc[s].append(c.speaker_uar[s])
        if not acc:
            missing.append((MECH_ARM, level, model, "spk_half_%s" % which))
            return None
        return {s: dmean(v) for s, v in acc.items()}

    def mech_half_group(self, level, which, missing):
        return self.mean_maps([self.mech_half(level, m, which, missing) for m in MECH_SCRATCH])

    # ---- HPO -----------------------------------------------------------------------------
    def hpo_run(self, level, cell, missing):
        c = self.ctx.run(HPO_ARM, level, "resnet_se", cell, 0, 0)
        if c is None:
            missing.append((HPO_ARM, level, "resnet_se", cell))
        return c

    def n14(self, missing):
        gr = self.hpo_run("cremad", "GR_hpo", missing)
        gg = self.hpo_run("cremad", "GG_hpo", missing)
        if gr is None or gg is None:
            return None
        D = {}
        for s in gr.speaker_uar:
            if s not in gg.speaker_uar:
                continue
            f1, f2 = gr.fold_of_speaker.get(s), gg.fold_of_speaker.get(s)
            if f1 is None or f2 is None or gr.val_sel.get(f1) is None or gg.val_sel.get(f2) is None:
                continue
            D[s] = (gr.val_sel[f1] - gr.speaker_uar[s]) - (gg.val_sel[f2] - gg.speaker_uar[s])
        return D if D else None

    # ---- N15 / T01 -------------------------------------------------------------------------
    def n15(self, missing):
        ridge = self.mean_maps([self.diff(self.V(PROBECPU_ARM, "ravdess", m, "RO", missing),
                                          self.V(PROBECPU_ARM, "ravdess", m, "GO", missing)) for m in RIDGE_MODELS])
        probe = self.mean_maps([self.diff(self.V(CTRL_ARM, "ravdess", m, "RG", missing),
                                          self.V(CTRL_ARM, "ravdess", m, "GG", missing)) for m in PROBE_MODELS])
        return self.diff(ridge, probe)

    def t01(self, missing):
        return self.mean_maps([self.diff(self.V(PROBECPU_ARM, "cremad", m, "LOSOSUB", missing),
                                         self.V(PROBECPU_ARM, "cremad", m, "G5", missing)) for m in RIDGE_MODELS])

    def t02(self, missing):
        return self.diff(self.mech_half_group("cremad", "unexposed", missing),
                         self.Vg(MECH_ARM, "cremad", MECH_SCRATCH, "none", missing))

    # ---- table of Section 4 -----------------------------------------------------------------
    def compute(self, hid, missing):
        S = self.scratch
        table = {
            "N01": lambda: self.premium(CTRL_ARM, "subesco_980", S("subesco_980"), "RR", "GG", missing),
            "N02": lambda: self.premium(CTRL_ARM, "subesco_980", PROBE_MODELS, "RR", "GG", missing),
            "N03": lambda: self.premium(CTRL_ARM, "subesco_980", S("subesco_980"), "RG", "GG", missing),
            "N04": lambda: self.premium(CTRL_ARM, "subesco_980", S("subesco_980"), "RR", "RG", missing),
            "N05": lambda: self.mech("cremad", "prm", "none", missing),
            "N06": lambda: self.mech("cremad", "spk", "none", missing),
            "N07": lambda: self.mech("subesco_full", "prm", "none", missing),
            "N08": lambda: self.mech("subesco_full", "spk", "none", missing),
            "N09": lambda: self.mech("subesco_full", "both_sib", "both", missing),
            "N10": lambda: self.n10(missing),
            "N11": lambda: self.ft_did("ravdess", missing),
            "N12": lambda: self.ft_did("cremad", missing),
            "N13": lambda: self.ft_did("subesco_980", missing),
            "N14": lambda: self.n14(missing),
            "N15": lambda: self.n15(missing),
            "R01": lambda: self.premium(CTRL_ARM, "ravdess", S("ravdess"), "RR", "GG", missing),
            "R02": lambda: self.premium(CTRL_ARM, "ravdess", S("ravdess"), "RG", "GG", missing),
            "R03": lambda: self.premium(CTRL_ARM, "ravdess", S("ravdess"), "RR", "RG", missing),
            "R04": lambda: self.premium(CTRL_ARM, "cremad", S("cremad"), "RR", "GG", missing),
            "R05": lambda: self.premium(CTRL_ARM, "cremad", S("cremad"), "RG", "GG", missing),
            "R06": lambda: self.premium(CTRL_ARM, "cremad", S("cremad"), "RR", "RG", missing),
            "R07": lambda: self.premium(CTRL_ARM, "ravdess", PROBE_MODELS, "RR", "GG", missing),
            "R08": lambda: self.premium(CTRL_ARM, "cremad", PROBE_MODELS, "RR", "GG", missing),
            "R09": lambda: self.diff(self.V(FT_ARM, "ravdess", FT_MODEL, "RR", missing), self.V(FT_ARM, "ravdess", FT_MODEL, "GG", missing)),
            "R10": lambda: self.diff(self.V(FT_ARM, "cremad", FT_MODEL, "RR", missing), self.V(FT_ARM, "cremad", FT_MODEL, "GG", missing)),
            "R11": lambda: self.diff(self.V(FT_ARM, "subesco_980", FT_MODEL, "RR", missing), self.V(FT_ARM, "subesco_980", FT_MODEL, "GG", missing)),
            "T01": lambda: self.t01(missing),
            "T02": lambda: self.t02(missing),
        }
        fn = table.get(hid)
        if fn is None:
            missing.append(("unknown hypothesis", hid, "", ""))
            return None
        D = fn()
        return D if D else None


# --------------------------------------------------------------------------------------
# Truncation / deviation bookkeeping for not-tested hypotheses
# --------------------------------------------------------------------------------------
def classify_missing(ctx: Context, missing_terms):
    """For the undefined V terms of a hypothesis, decide whether they are 'truncated'
    (all planned units absent from the run and truncation-ranked), 'void' (units present
    but the cell-run is VOID) or 'missing' (absent without a truncation rank)."""
    kinds, ranks, uids = [], set(), []
    for term in missing_terms:
        arm, level, model, cell = term
        rows = ctx.plan_by_cell.get((arm, level, model, cell), [])
        if arm in SEED_EQ_R_ARMS:
            rows = [r for r in rows if r["seed_index"] == r["r"]]
        if not rows:
            kinds.append("missing")
            continue
        present = [r for r in rows if r["unit_id"] in ctx.units]
        if present:
            kinds.append("void")
        elif all(r["truncation_rank"] != 99 for r in rows) or all(
                (r.get("status") or "").lower() in ("truncated", "skipped", "cancelled", "dropped") for r in rows):
            kinds.append("truncated")
            ranks.update(r["truncation_rank"] for r in rows)
            uids.extend(r["unit_id"] for r in rows)
        else:
            kinds.append("missing")
    if not kinds:
        return "undefined", None
    if all(k == "truncated" for k in kinds):
        return "truncated", (ranks, uids)
    if any(k == "void" for k in kinds):
        return "void", None
    return "missing", None


def find_deviation_entry(ctx: Context, item_id):
    """Number of the first deviations.jsonl entry whose "hypotheses" list names item_id
    (hypothesis or claim id); None when no entry names it.  Matching is on ids only."""
    for num, _text, obj in ctx.deviations:
        if isinstance(obj, dict):
            names = obj.get("hypotheses")
            if isinstance(names, str):
                names = [names]
            if isinstance(names, list) and item_id in names:
                return num
    return None


def not_tested_verdict(entry_number):
    if entry_number is None:
        return "not tested"
    return "not tested (truncated, deviation entry %s)" % entry_number


# --------------------------------------------------------------------------------------
# Results assembly (Section 6)
# --------------------------------------------------------------------------------------
def is_tost_family(fam: str) -> bool:
    """T rows are the equivalence (TOST) hypotheses; the family label starts with 'T'."""
    return fam == "T" or fam.startswith("T-")


def compute_hypotheses(ctx: Context, con: Contrasts):
    hyps = collections.OrderedDict()
    tosts = collections.OrderedDict()
    fam_p = collections.defaultdict(collections.OrderedDict)   # family -> {hid: p} (confirmatory only)
    for hid, reg in ctx.hypotheses.items():
        fam = reg["family"]
        is_t = is_tost_family(fam) or reg["direction"] == "equivalence"
        confirmatory = reg["a_priori_status"] == "confirmatory"
        missing = []
        D = con.compute(hid, missing)
        st = contrast_stats(D)
        entry = dict(st)
        entry.update({"family": fam, "m": ctx.family_m.get(fam, reg["m"]), "direction": reg["direction"] or None,
                      "claim_id": reg["claim_id"], "a_priori_status": reg["a_priori_status"],
                      "power_holm": reg["power_holm"], "reason": None, "deviation_entry": None})
        tested = st["n"] > 0
        if is_t:
            entry["margin"] = reg["margin"]
            tested = st["n"] >= 2 and reg["margin"] is not None
            if tested:
                D_arr = np.array([D[s] for s in st["speakers"]], dtype=np.float64)
                entry.update(tost(D_arr, reg["margin"]))
            else:
                entry.update({"p_low": None, "p_high": None, "p": 1.0, "ci90": None, "se": None})
        if not tested:
            kind, _info = classify_missing(ctx, missing)
            entry["reason"] = kind if missing else ("no speakers with a defined contrast" if D is None else "insufficient n")
            # the '(truncated, deviation entry N)' suffix depends only on deviations.jsonl naming the id
            entry["deviation_entry"] = find_deviation_entry(ctx, hid)
            entry["missing_terms"] = ["|".join(str(x) for x in t) for t in missing]
        entry["tested"] = bool(tested)
        entry["is_tost"] = bool(is_t)
        if confirmatory:
            p_family = (entry["p"] if is_t else st["wilcoxon_p"]) if tested else 1.0
            fam_p[fam][hid] = float(p_family)
        else:
            # a priori demoted / estimate rows: reported, never Holm-adjusted, never a verdict
            entry["p_holm"] = None
            entry["reject"] = False
            entry["verdict"] = "estimate (not tested)" if tested else not_tested_verdict(entry["deviation_entry"])
            if is_t:
                entry["equivalent"] = False
        (tosts if is_t else hyps)[hid] = entry
    # Holm within each family (missing confirmatory hypotheses enter with p = 1)
    for fam, pmap in fam_p.items():
        adj, rej = holm(pmap, ctx.family_m.get(fam, len(pmap)))
        for hid in pmap:
            entry = tosts[hid] if hid in tosts else hyps[hid]
            entry["p_holm"] = adj[hid]
            entry["reject"] = bool(rej[hid]) and entry["tested"]
            if not entry["tested"]:
                entry["verdict"] = not_tested_verdict(entry["deviation_entry"])
                if entry["is_tost"]:
                    entry["equivalent"] = False
                continue
            if entry["is_tost"]:
                entry["equivalent"] = bool(entry["reject"])
                entry["verdict"] = "supported" if entry["equivalent"] else "not supported"
            else:
                direction = entry["direction"]
                if direction == ">0":
                    ok = entry["reject"] and entry["mean"] > 0
                elif direction == "<0":
                    ok = entry["reject"] and entry["mean"] < 0
                else:                       # e.g. N15: no direction
                    ok = entry["reject"]
                entry["verdict"] = "supported" if ok else "not supported"
    return hyps, tosts


def compute_claims(ctx: Context, hyps, tosts):
    allh = dict(hyps)
    allh.update(tosts)
    claims = collections.OrderedDict()
    for cid, spec in ctx.claim_map.get("claims", {}).items():
        if "requires_any" in spec:
            rule, members, k = "requires_any", list(spec["requires_any"]), 1
        elif "requires_k_of" in spec:
            rule, k, members = "requires_k_of", int(spec["requires_k_of"][0]), list(spec["requires_k_of"][1])
        else:
            rule, members, k = "requires", list(spec.get("requires", [])), None
        evidence = collections.OrderedDict()
        for hid in members + list(spec.get("reported", [])) + list(spec.get("demoted_a_priori", [])):
            evidence[hid] = allh[hid]["verdict"] if hid in allh else "not tested"
        rule_text = "%s(%s)%s" % (rule, ",".join(members), "" if k is None or rule != "requires_k_of" else " k=%d" % k)
        base = {"rule": rule_text, "evidence": evidence, "type": spec.get("type"),
                "verdict_rule": spec.get("verdict_rule"),
                "demoted_a_priori": list(spec.get("demoted_a_priori", []))}
        if spec.get("type") == "estimation":
            claims[cid] = dict(verdict="reported (estimation claim)", **base)
            continue
        if not members:
            claims[cid] = dict(verdict="no verdict", **base)
            continue
        verdicts = {hid: evidence[hid] for hid in members}
        # anything that is neither 'supported' nor 'not supported' cannot satisfy or fail a rule
        untested = [hid for hid in members if verdicts[hid] not in ("supported", "not supported")]
        supported = [hid for hid in members if verdicts[hid] == "supported"]

        def opposite_sign_not_supported(hid):
            e = allh.get(hid)
            if e is None or e["verdict"] != "not supported" or e.get("mean") is None:
                return False
            d = e.get("direction") or ""
            if d == ">0":
                return e["mean"] < 0
            if d == "<0":
                return e["mean"] > 0
            return False

        if rule == "requires":
            satisfied = len(supported) == len(members)
        elif rule == "requires_any":
            satisfied = len(supported) >= 1 and not any(opposite_sign_not_supported(h) for h in members)
        else:
            satisfied = len(supported) >= k
        if satisfied:
            verdict = "supported"
        elif untested:
            # suffix only when deviations.jsonl names the claim itself or an untested member
            entries = {find_deviation_entry(ctx, cid)}
            entries |= {allh[h].get("deviation_entry") for h in untested if h in allh}
            entries.discard(None)
            verdict = not_tested_verdict(min(entries) if entries else None)
        else:
            verdict = "not supported"
        claims[cid] = dict(verdict=verdict, **base)
    return claims


def desc_stats(dmap):
    st = contrast_stats(dmap)
    return {k: st[k] for k in ("n", "mean", "median", "sd", "ci_low", "ci_high", "speakers", "d")}


def sign_of(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def compute_descriptors(ctx: Context, con: Contrasts):
    D = collections.OrderedDict()
    # D01: GR - GG allocation package per family level (scratch and probe groups)
    d01 = collections.OrderedDict()
    for lv in FAMILY_LEVELS:
        d01[lv] = {"scratch": desc_stats(con.premium(CTRL_ARM, lv, con.scratch(lv), "GR", "GG", [])),
                   "probe": desc_stats(con.premium(CTRL_ARM, lv, PROBE_MODELS, "GR", "GG", []))}
    D["D01"] = d01
    # D02: per-model RR-GG sign counts per family level
    d02 = collections.OrderedDict()
    for lv in FAMILY_LEVELS:
        models = collections.OrderedDict()
        counts = {}
        for gname, group in (("scratch", con.scratch(lv)), ("probe", PROBE_MODELS)):
            pos, tot = 0, 0
            for m in group:
                d = con.diff(con.V(CTRL_ARM, lv, m, "RR"), con.V(CTRL_ARM, lv, m, "GG"))
                if d:
                    mean = dmean(list(d.values()))
                    models[m] = {"n": len(d), "mean": mean, "sign": sign_of(mean), "group": gname}
                    tot += 1
                    pos += int(mean > 0)
                else:
                    models[m] = {"n": 0, "mean": None, "sign": None, "group": gname}
            counts[gname] = {"positive": pos, "total": len(group), "defined": tot,
                             "positive_models_over_total": "%d/%d" % (pos, len(group))}
        # per-replicate estimates (main replicates r == seed) for information
        per_rep = collections.OrderedDict()
        for m in tuple(con.scratch(lv)) + tuple(PROBE_MODELS):
            reps = {}
            for c in ctx.complete_runs(CTRL_ARM, lv, m, "RR"):
                if c.seed != c.r:
                    continue
                g = ctx.run(CTRL_ARM, lv, m, "GG", c.r, c.seed)
                if g is None:
                    continue
                common = sorted(set(c.speaker_uar) & set(g.speaker_uar))
                if common:
                    reps["r%d" % c.r] = dmean([c.speaker_uar[s] - g.speaker_uar[s] for s in common])
            per_rep[m] = reps
        d02[lv] = {"models": models, "scratch": counts["scratch"], "probe": counts["probe"], "per_replicate": per_rep}
    D["D02"] = d02
    # D03: corpus differences of the scratch RR-GG mean with two-sample speaker bootstrap
    d03 = collections.OrderedDict()
    base = con.premium(CTRL_ARM, "ravdess", con.scratch("ravdess"), "RR", "GG", [])
    for other in ("cremad", "subesco_980"):
        o = con.premium(CTRL_ARM, other, con.scratch(other), "RR", "GG", [])
        name = "ravdess_minus_%s" % other
        if not base or not o:
            d03[name] = {"n_a": len(base) if base else 0, "n_b": len(o) if o else 0, "diff": None,
                         "ci_low": None, "ci_high": None, "label": None}
            continue
        A = np.array([base[s] for s in sorted(base)], dtype=np.float64)
        B = np.array([o[s] for s in sorted(o)], dtype=np.float64)
        rng = np.random.RandomState(BOOT_SEED)
        ia = rng.randint(0, len(A), size=(N_BOOT, len(A)))
        ib = rng.randint(0, len(B), size=(N_BOOT, len(B)))
        diffs = A[ia].mean(axis=1) - B[ib].mean(axis=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5], method="linear")
        lo, hi = float(lo), float(hi)
        label = "larger" if lo > D03_BAND else ("smaller" if hi < -D03_BAND else "within_band")
        d03[name] = {"n_a": int(len(A)), "n_b": int(len(B)), "mean_a": float(np.mean(A)), "mean_b": float(np.mean(B)),
                     "diff": float(np.mean(A)) - float(np.mean(B)), "ci_low": lo, "ci_high": hi,
                     "band": D03_BAND, "label": label}
    D["D03"] = d03
    # D04: sub_1400_one vs sub_700_one; sub_700_two vs sub_700_one (scratch RR-GG, paired by speaker)
    def sub_prem(lv):
        return con.premium(CTRL_ARM, lv, SCRATCH_SECONDARY, "RR", "GG", [])
    p700 = sub_prem("sub_700_one")
    D["D04"] = {"sub_1400_one_vs_sub_700_one": desc_stats(con.diff(sub_prem("sub_1400_one"), p700)),
                "sub_700_two_vs_sub_700_one": desc_stats(con.diff(sub_prem("sub_700_two"), p700))}
    # D05: subesco_full cnn RR - TG
    D["D05"] = desc_stats(con.diff(con.V(CTRL_ARM, "subesco_full", "cnn", "RR"), con.V(CTRL_ARM, "subesco_full", "cnn", "TG")))
    # D06: mechanism block per level
    d06 = collections.OrderedDict()
    for lv in ("cremad", "subesco_full"):
        none = con.Vg(MECH_ARM, lv, MECH_SCRATCH, "none")
        both = con.Vg(MECH_ARM, lv, MECH_SCRATCH, "both")
        spk = con.Vg(MECH_ARM, lv, MECH_SCRATCH, "spk")
        prm = con.Vg(MECH_ARM, lv, MECH_SCRATCH, "prm")
        both_none = con.diff(both, none)
        spk_none = con.diff(spk, none)
        prm_none = con.diff(prm, none)
        inter = con.diff(con.diff(both_none, spk_none), prm_none)
        g1 = spk_none
        g05 = con.diff(con.mech_half_group(lv, "exposed", []), none)
        crowd = con.diff(g1, g05)
        share = None
        if g1 and g05:
            m1 = dmean(list(g1.values()))
            if m1 > 0:
                share = dmean(list(g05.values())) / m1
        d06[lv] = {"both_minus_none": desc_stats(both_none), "interaction": desc_stats(inter),
                   "G1": desc_stats(g1), "G05": desc_stats(g05), "crowding": desc_stats(crowd), "share": share}
    D["D06"] = d06
    # D07: per-draw panel premia and SD over draws
    p24, p91 = con.panel_premia([])
    draws = collections.OrderedDict()
    m24, m91 = [], []
    for k in con.draws():
        e = {"p24_mean": None, "p24_n": 0, "p91_mean": None, "p91_n": 0}
        if k in p24:
            e["p24_mean"] = dmean(list(p24[k].values()))
            e["p24_n"] = len(p24[k])
            m24.append(e["p24_mean"])
        if k in p91:
            e["p91_mean"] = dmean(list(p91[k].values()))
            e["p91_n"] = len(p91[k])
            m91.append(e["p91_mean"])
        draws["d%d" % k] = e
    D["D07"] = {"draws": draws,
                "sd_p24": float(np.std(m24, ddof=1)) if len(m24) > 1 else None,
                "sd_p91": float(np.std(m91, ddof=1)) if len(m91) > 1 else None,
                "n_draws_p24": len(m24), "n_draws_p91": len(m91)}
    # D08: RR_hpo premium, selected configs, test-selection optimism (cremad)
    d08 = collections.OrderedDict()
    rr = ctx.run(HPO_ARM, "cremad", "resnet_se", "RR_hpo", 0, 0)
    gg = ctx.run(HPO_ARM, "cremad", "resnet_se", "GG_hpo", 0, 0)
    d08["premium_RR_hpo_minus_GG_hpo"] = desc_stats(con.diff(rr.speaker_uar if rr else None, gg.speaker_uar if gg else None))
    sel, opt = collections.OrderedDict(), collections.OrderedDict()
    for cell in HPO_CELLS:
        c = ctx.run(HPO_ARM, "cremad", "resnet_se", cell, 0, 0)
        if c is None:
            sel[cell] = None
            opt[cell] = {"per_fold": None, "mean": None}
            continue
        sel[cell] = {"f%d" % f: c.selected[f] for f in sorted(c.selected)}
        per = {"f%d" % f: c.test_max[f] - c.test_sel[f] for f in sorted(c.selected)}
        opt[cell] = {"per_fold": per, "mean": dmean(list(per.values())) if per else None,
                     "val_sel": {"f%d" % f: c.val_sel[f] for f in sorted(c.selected)},
                     "test_sel": {"f%d" % f: c.test_sel[f] for f in sorted(c.selected)},
                     "test_max": {"f%d" % f: c.test_max[f] for f in sorted(c.selected)}}
    d08["selected_config_index"] = sel
    d08["test_selection_optimism"] = opt
    D["D08"] = d08
    # D09: honest DiD (HPO GR-GG) - (CTRL r=0 resnet_se GR-GG) on cremad
    gr = ctx.run(HPO_ARM, "cremad", "resnet_se", "GR_hpo", 0, 0)
    cgr = ctx.run(CTRL_ARM, "cremad", "resnet_se", "GR", 0, 0)
    cgg = ctx.run(CTRL_ARM, "cremad", "resnet_se", "GG", 0, 0)
    hpo_d = con.diff(gr.speaker_uar if gr else None, gg.speaker_uar if gg else None)
    fixed_d = con.diff(cgr.speaker_uar if cgr else None, cgg.speaker_uar if cgg else None)
    D["D09"] = desc_stats(con.diff(hpo_d, fixed_d))
    # D14: crossing matrices for ravdess and subesco_980, cnn
    d14 = collections.OrderedDict()
    reps = sorted({r["r"] for r in ctx.plan.values() if r["arm"] == CTRL_ARM})
    seeds = sorted({r["seed_index"] for r in ctx.plan.values() if r["arm"] == CTRL_ARM})
    for lv in ("ravdess", "subesco_980"):
        M = []
        for r in reps:
            row = []
            for sd in seeds:
                a = ctx.run(CTRL_ARM, lv, "cnn", "RR", r, sd)
                b = ctx.run(CTRL_ARM, lv, "cnn", "GG", r, sd)
                d = con.diff(a.speaker_uar if a else None, b.speaker_uar if b else None)
                row.append(dmean(list(d.values())) if d else None)
            M.append(row)
        def mean_or_none(vals):
            v = [x for x in vals if x is not None]
            return dmean(v) if v else None
        row_means = [mean_or_none(row) for row in M]
        col_means = [mean_or_none([M[i][j] for i in range(len(reps))]) for j in range(len(seeds))]
        allv = [x for row in M for x in row if x is not None]
        rm = [x for x in row_means if x is not None]
        cm = [x for x in col_means if x is not None]
        d14[lv] = {"r": reps, "seeds": seeds, "matrix": M, "row_means": row_means, "col_means": col_means,
                   "grand_mean": dmean(allv) if allv else None,
                   "sd_draw": float(np.std(rm, ddof=1)) if len(rm) > 1 else None,
                   "sd_seed": float(np.std(cm, ddof=1)) if len(cm) > 1 else None}
    D["D14"] = d14
    return D


def build_results(ctx: Context):
    con = Contrasts(ctx)
    hyps, tosts = compute_hypotheses(ctx, con)
    claims = compute_claims(ctx, hyps, tosts)
    descriptors = compute_descriptors(ctx, con)
    cell_runs = collections.OrderedDict()
    for key in sorted(ctx.cell_runs):
        c = ctx.cell_runs[key]
        e = {"complete": bool(c.complete), "n_folds": int(c.n_folds_present), "n_rows": int(c.n_rows),
             "speaker_uar": collections.OrderedDict((s, c.speaker_uar[s]) for s in sorted(c.speaker_uar)),
             "n_folds_expected": c.n_folds_expected, "split_key": c.split_key,
             "covers_population": c.covers_population, "void_reason": "; ".join(c.reasons) or None}
        if c.is_hpo:
            e["selected_config_index"] = {"f%d" % f: c.selected[f] for f in sorted(c.selected)}
            e["val_sel"] = {"f%d" % f: c.val_sel[f] for f in sorted(c.val_sel)}
            e["test_sel"] = {"f%d" % f: c.test_sel[f] for f in sorted(c.test_sel)}
            e["test_max"] = {"f%d" % f: c.test_max[f] for f in sorted(c.test_max)}
        cell_runs[key] = e
    void_units = 0
    for u in ctx.units.values():
        c = ctx.cell_runs.get(cell_run_key(u.arm, u.level, u.model, u.cell, u.r, u.seed))
        if not u.usable or c is None or not c.complete:
            void_units += 1
    run_json = os.path.join(ctx.run_dir, "run.json")
    run_id = None
    if os.path.exists(run_json):
        try:
            run_id = load_json(run_json).get("run_id")
        except FatalInputError:
            run_id = None
    if not run_id:
        run_id = os.path.basename(os.path.normpath(ctx.run_dir))
    integrity = collections.OrderedDict()
    for k, v in ctx.integrity.items():
        integrity[k] = {"pass": bool(v["pass"]), "failures": list(v["failures"]), "n_failures": len(v["failures"])}
    return collections.OrderedDict([
        ("schema", SCHEMA),
        ("run_id", run_id),
        ("plan_sha256", ctx.plan_sha256),
        ("registry_sha256", ctx.registry_sha256),
        ("n_units_done", int(ctx.n_units_done)),
        ("n_units_void", int(void_units)),
        ("cell_runs", cell_runs),
        ("hypotheses", hyps),
        ("tost", tosts),
        ("claims", claims),
        ("descriptors", descriptors),
        ("integrity", integrity),
        ("verifier", collections.OrderedDict([
            ("module", "ser_v2.verify"), ("spec", SPEC_ID), ("bootstrap_seed", BOOT_SEED),
            ("n_boot", N_BOOT), ("alpha", ALPHA), ("numpy", np.__version__),
            ("scipy", __import__("scipy").__version__),
            ("n_units_not_done", int(ctx.n_units_not_done)),
            ("n_units_usable", int(sum(1 for u in ctx.units.values() if u.usable))),
            ("n_cell_runs", len(ctx.cell_runs)),
            ("n_cell_runs_complete", sum(1 for c in ctx.cell_runs.values() if c.complete)),
            ("manifests", {b: (None if m is None else {"n_input": m.n_input, "n_kept": m.n_kept,
                                                       "n_classes": m.n_classes, "n_speakers": len(m.speakers),
                                                       "dropped": [{"relative_path": p, "reason": r} for p, r in m.dropped]})
                           for b, m in ctx.manifests.items()}),
            ("interpretations", INTERPRETATIONS),
        ])),
    ])


# --------------------------------------------------------------------------------------
# Comparison with the scorer (Section 7)
# --------------------------------------------------------------------------------------
def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_values(a, b):
    """Returns (pass, abs_diff|None)."""
    if a is None and b is None:
        return True, 0.0
    if is_number(a) and is_number(b):
        if isinstance(a, float) or isinstance(b, float):
            if math.isnan(a) or math.isnan(b):
                return False, None
            d = abs(float(a) - float(b))
            return d <= TOL, d
        return a == b, float(abs(a - b))
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b, 0.0 if a == b else None
    if isinstance(a, str) and isinstance(b, str):
        return a == b, 0.0 if a == b else None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, None
        worst = 0.0
        for x, y in zip(a, b):
            ok, d = compare_values(x, y)
            if not ok:
                return False, d
            worst = max(worst, d or 0.0)
        return True, worst
    return False, None


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, "%s.%d" % (prefix, i) if prefix else str(i)))
    else:
        out[prefix] = obj
    return out


def compare_results(scorer: dict, mine: dict, results_path: str):
    items = collections.OrderedDict()

    def add(name, a, b):
        ok, d = compare_values(a, b)
        items[name] = {"scorer": a, "verifier": b, "abs_diff": d, "pass": bool(ok)}

    for block, fields in (("hypotheses", HYP_COMPARE_FIELDS), ("tost", TOST_COMPARE_FIELDS)):
        sb, mb = scorer.get(block, {}) or {}, mine.get(block, {}) or {}
        for hid in sorted(set(sb) | set(mb)):
            if hid not in sb or hid not in mb:
                items["%s.%s" % (block, hid)] = {"scorer": "present" if hid in sb else "absent",
                                                 "verifier": "present" if hid in mb else "absent",
                                                 "abs_diff": None, "pass": False}
                continue
            for f in fields:
                add("%s.%s.%s" % (block, hid, f), sb[hid].get(f), mb[hid].get(f))
    sc, mc = scorer.get("claims", {}) or {}, mine.get("claims", {}) or {}
    for cid in sorted(set(sc) | set(mc)):
        a = sc.get(cid, {}).get("verdict") if cid in sc else "absent"
        b = mc.get(cid, {}).get("verdict") if cid in mc else "absent"
        add("claims.%s.verdict" % cid, a, b)
    sd, md = flatten(scorer.get("descriptors", {}) or {}), flatten(mine.get("descriptors", {}) or {})
    for path in sorted(set(sd) & set(md)):
        a, b = sd[path], md[path]
        if (is_number(a) or a is None) and (is_number(b) or b is None) and not (a is None and b is None):
            add("descriptors.%s" % path, a, b)
    si, mi = scorer.get("integrity", {}) or {}, mine.get("integrity", {}) or {}
    for k in sorted(set(si) & set(mi)):
        if k == "I0":
            continue
        add("integrity.%s.pass" % k, si[k].get("pass"), mi[k].get("pass"))
    failed = [k for k, v in items.items() if not v["pass"]]
    diffs = [v["abs_diff"] for v in items.values() if v["abs_diff"] is not None]
    return collections.OrderedDict([
        ("schema", VERIFICATION_SCHEMA),
        ("results_path", results_path),
        ("run_id", {"scorer": scorer.get("run_id"), "verifier": mine.get("run_id")}),
        ("tolerance", TOL),
        ("pass", not failed),
        ("max_abs_diff", max(diffs) if diffs else 0.0),
        ("n_items", len(items)),
        ("n_failed", len(failed)),
        ("failed_items", failed),
        ("verifier_integrity_pass", all(v["pass"] for v in mi.values())),
        ("items", items),
    ])


# --------------------------------------------------------------------------------------
# numeric_insert.tex (best effort, Section 6 last paragraph)
# --------------------------------------------------------------------------------------
_DIGIT_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def _macro_ids(name, ids):
    low = name.lower()
    for w, d in sorted(_DIGIT_WORDS.items(), key=lambda kv: -len(kv[0])):
        low = low.replace(w, d)
    return [h for h in ids if h.lower() in low]


def _rounded_forms(v):
    if isinstance(v, bool) or v is None:
        return set()
    if isinstance(v, int):
        return {str(v)}
    forms = {"%.2f" % v, "%.3f" % v, repr(v)}
    return forms


def check_tex(tex_path, scorer):
    text = read_bytes(tex_path).decode("utf-8", "replace")
    pat = re.compile(r"\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z@]+)\}?\s*(?:\[\d+\])?\s*\{([^{}]*)\}")
    ids = list((scorer.get("hypotheses") or {}).keys()) + list((scorer.get("tost") or {}).keys())
    allvals = flatten({"hypotheses": scorer.get("hypotheses"), "tost": scorer.get("tost"),
                       "descriptors": scorer.get("descriptors")})
    numeric_all = [v for v in allvals.values() if is_number(v)]
    macros, unmatched_specific, unmatched_any = 0, [], []
    for name, raw in pat.findall(text):
        macros += 1
        val = raw.replace("\\%", "").replace("$", "").replace("\\textless", "<").replace("\\,", "").strip()
        hids = _macro_ids(name, ids)
        if hids:
            cand = []
            for h in hids:
                e = (scorer.get("hypotheses") or {}).get(h) or (scorer.get("tost") or {}).get(h) or {}
                cand.extend(v for v in flatten(e).values() if is_number(v))
            target = unmatched_specific
        else:
            cand = numeric_all
            target = unmatched_any
        if val.startswith("<"):
            thr = float(val[1:]) if re.match(r"^<\s*[0-9.]+$", val) else 0.001
            ok = any(isinstance(v, float) and 0 <= v < thr for v in cand)
        else:
            try:
                fval = float(val)
            except ValueError:
                ok = True     # non-numeric macro: not a numeric claim
                continue
            ok = any(val in _rounded_forms(v) or (is_number(v) and abs(round(float(v), 2) - fval) < 1e-9)
                     or (is_number(v) and abs(round(float(v), 3) - fval) < 1e-9) for v in cand)
        if not ok:
            target.append({"macro": name, "value": raw})
    return {"path": tex_path, "n_macros": macros, "pass": not unmatched_specific,
            "unmatched_hypothesis_macros": unmatched_specific,
            "unmatched_other_macros_info": unmatched_any}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def summarize(results, verification):
    hy = results["hypotheses"]
    tos = results["tost"]
    cnt = collections.Counter(e["verdict"] for e in list(hy.values()) + list(tos.values()))
    integ = results["integrity"]
    print("=== ser_v2.verify summary ===")
    print("run_id=%s  units done=%d usable=%d void=%d  cell-runs=%d complete=%d" % (
        results["run_id"], results["n_units_done"], results["verifier"]["n_units_usable"], results["n_units_void"],
        results["verifier"]["n_cell_runs"], results["verifier"]["n_cell_runs_complete"]))
    print("integrity: " + "  ".join("%s=%s%s" % (k, "pass" if v["pass"] else "FAIL",
                                                 "" if v["pass"] else "(%d)" % v["n_failures"]) for k, v in integ.items()))
    print("hypotheses: " + ", ".join("%s=%d" % (k, v) for k, v in sorted(cnt.items())))
    for hid, e in list(hy.items()) + list(tos.items()):
        if e["verdict"].startswith("not tested"):
            print("  %s [%s]: %s (%s)" % (hid, e["family"], e["verdict"], e.get("reason")))
        else:
            extra = " p=%.4g" % e["p"] if "p" in e and e.get("p") is not None else ""
            ph = "null" if e.get("p_holm") is None else "%.4g" % e["p_holm"]
            print("  %s [%s]: n=%d mean=%.4f ci=[%.4f, %.4f] wilcoxon_p=%.4g%s p_holm=%s -> %s" % (
                hid, e["family"], e["n"], e["mean"], e["ci_low"], e["ci_high"], e["wilcoxon_p"], extra, ph, e["verdict"]))
    print("claims: " + ", ".join("%s=%s" % (c, v["verdict"]) for c, v in results["claims"].items()))
    if verification is not None:
        print("verification: pass=%s items=%d failed=%d max_abs_diff=%s" % (
            verification["pass"], verification["n_items"], verification["n_failed"], verification["max_abs_diff"]))
        for k in verification["failed_items"][:40]:
            it = verification["items"][k]
            print("  MISMATCH %s: scorer=%r verifier=%r" % (k, it["scorer"], it["verifier"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ser_v2.verify",
                                 description="Independent verifier for the SER v2 scoring contract.")
    ap.add_argument("--plan", required=True, help="plan directory (run_plan.csv, split_index.json, ...)")
    ap.add_argument("--manifests", required=True, help="directory with <base>_manifest.csv files")
    ap.add_argument("--run", required=True, help="run directory (units/<unit_id>/...)")
    ap.add_argument("--registry", required=True, help="registry directory (hypothesis_registry.csv, claim_map.json)")
    ap.add_argument("--results", default=None, help="scorer results.json to compare against (never modified)")
    ap.add_argument("--out", default=None, help="verification.json path (default RUN_DIR/verification.json)")
    ap.add_argument("--tex", default=None, help="numeric_insert.tex to check (default: next to --results if present)")
    args = ap.parse_args(argv)

    try:
        for p in (args.plan, args.manifests, args.run, args.registry):
            if not os.path.isdir(p):
                raise FatalInputError("not a directory: %s" % p)
        ctx = Context(args.plan, args.manifests, args.run, args.registry)
        ctx.load_registry()
        ctx.load_plan()
        ctx.load_manifests()
        ctx.load_deviations()
        ctx.load_units()
        ctx.build_cell_runs()
        results = build_results(ctx)
    except FatalInputError as exc:
        print("FATAL: %s" % exc, file=sys.stderr)
        return 2

    out_results = os.path.join(args.run, "verifier_results.json")
    write_json(out_results, results)
    log("wrote %s" % out_results)
    rc = 0 if all(v["pass"] for v in results["integrity"].values()) else 1

    verification = None
    if args.results:
        try:
            scorer = load_json(args.results)
        except FatalInputError as exc:
            print("FATAL: %s" % exc, file=sys.stderr)
            return 2
        verification = compare_results(scorer, to_jsonable(results), args.results)
        tex_path = args.tex or os.path.join(os.path.dirname(os.path.abspath(args.results)), "numeric_insert.tex")
        if os.path.exists(tex_path):
            verification["tex"] = check_tex(tex_path, scorer)
            if not verification["tex"]["pass"]:
                verification["pass"] = False
                verification["failed_items"].append("tex.unmatched_hypothesis_macros")
                verification["n_failed"] += 1
        if not verification["verifier_integrity_pass"]:
            verification["pass"] = False
        out = args.out or os.path.join(args.run, "verification.json")
        write_json(out, verification)
        log("wrote %s" % out)
        if not verification["pass"]:
            rc = 1
    summarize(results, verification)
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""Single source of truth for the v2 program: arms, unit enumeration rules, cost model,
hypothesis registry, claim map, budget and truncation order. `export()` writes the
machine-readable files under v2/registry/ that are frozen at tag-1.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from .common import atomic_write_json, atomic_write_text, sha256_json

PROGRAM_VERSION = "2.0.0-rc1"

# ------------------------------------------------------------------ corpus levels
CORPUS_LEVELS = {
    # family levels (enter Family N / R)
    "ravdess":      {"base": "ravdess", "panel": None, "role": "family", "n_utt": 1440, "n_spk": 24, "n_classes": 8},
    "cremad":       {"base": "cremad", "panel": None, "role": "family", "n_utt": 7442, "n_spk": 91, "n_classes": 6},
    "subesco_980":  {"base": "subesco", "panel": "subesco_980", "role": "family", "n_utt": 980, "n_spk": 20, "n_classes": 7,
                     "definition": "prompts {1,2,3,6,8,9,10} x one take per speaker x sentence x emotion cell; the pinned S3 release_v10 selected manifest (sha256 in corpora.json) is authoritative when present"},
    # secondary levels (estimates only)
    "subesco_full": {"base": "subesco", "panel": None, "role": "secondary", "n_utt": 7000, "n_spk": 20, "n_classes": 7},
    "sub_1400_one": {"base": "subesco", "panel": "one_take_all_prompts", "role": "secondary", "n_utt": 1400, "n_spk": 20, "n_classes": 7},
    "sub_700_one":  {"base": "subesco", "panel": "700_cells_one_take", "role": "secondary", "n_utt": 700, "n_spk": 20, "n_classes": 7},
    "sub_700_two":  {"base": "subesco", "panel": "700_cells_two_takes", "role": "secondary", "n_utt": 1400, "n_spk": 20, "n_classes": 7},
    "cremad_24":    {"base": "cremad", "panel": "speakers_24", "role": "secondary", "n_utt": 1963, "n_spk": 24, "n_classes": 6, "draws": 5},
    "cremad_91m":   {"base": "cremad", "panel": "utterance_matched_91", "role": "secondary", "n_utt": 1963, "n_spk": 91, "n_classes": 6, "draws": 5},
}

SCRATCH_MODELS = ("cnn", "resnet_se", "transformer")
PROBE_MODELS = ("hubert_base", "wavlm_base_plus", "wav2vec2_base")
MODEL_COST_MULT = {"cnn": 1.0, "resnet_se": 1.4, "transformer": 1.1}
SEC_PER_TRAIN_UTT = 0.0095          # scratch 5-fold fit, RTX 5070, from P1/B/D1 records
PROBE_UNIT_SEC = 3.2                # early-stopped linear probe on cached features
RIDGE_UNIT_SEC = 0.07               # CPU
FT_FOLD_MIN = {"ravdess": 12.0, "cremad": 36.0, "subesco_980": 9.0}   # planning values; replaced by timing probe
FROZEN_SAME_REGIME_FRACTION = 1.0 / 3.0
REPLICATES = (0, 1, 2)
MECH_REPLICATES = (0, 1)          # two replicates x two prompt-group rotations = 20 fits per condition and model
MECH_ROTATIONS = (0, 1)

# ------------------------------------------------------------------ arms
ARMS = {
    "PREP": {"gpu_hours_plan": 3.0, "cap": 3.5, "kind": "infrastructure",
             "description": "manifests, log-mel cache (CPU), 13-state SSL cache for 3 encoders, split tables + assertions, synthetic dry run + mutation battery, off-family timing probe"},
    "CTRL": {"cap": 16.0, "kind": "gpu", "description": "outer x inner protocol factorial; scratch models + frozen probes; panels; crossings; LOSO pair; P1 reconciliation"},
    "MECH2X2": {"cap": 7.0, "kind": "gpu", "description": "speaker-overlap x prompt-overlap checkerboard (two prompt-group rotations) with matched training size; sibling condition; half-exposure rotations"},
    "FT": {"cap": 26.0, "kind": "gpu", "description": "partial fine-tune of WavLM-base+ vs frozen same-regime comparator, RR/GG, r=0 partitions"},
    "HPO": {"cap": 5.0, "kind": "gpu", "description": "8-config hyperparameter search on random vs grouped inner validation with fixed speaker-exclusive outer test (GR/GG) plus RR descriptor cell; ResNet-SE"},
    "MECHID": {"cap": 0.5, "kind": "gpu", "description": "inference-only: speaker-ID probes per checkpoint on a common panel; nearest-neighbour composition"},
    "PROBECPU": {"cap": 0.0, "kind": "cpu", "description": "Ridge factorial on 13-state features: head x partition family x layer x 20 draws; LOSO / Group5 / LOSO-sub TOST"},
    "STAT": {"gpu_hours_plan": 0.5, "cap": 0.5, "kind": "gpu", "description": "rolling 1%+1% bitwise retrain checks; one-shot scoring; independent verifier replay"},
}
PROGRAM_CAP_GPU_HOURS = 50.0

# preregistered truncation order, applied only if the live projection exceeds PROGRAM_CAP
TRUNCATION_ORDER = [
    "FT: cremad seed 1 (fine-tune and frozen comparator)          [conditional unit; first to go]",
    "HPO: ravdess cells (all)                                       ",
    "MECH2X2: replicate r=1 on cremad and subesco_full              ",
    "CTRL: cremad_24 / cremad_91m draws 3-4                         ",
    "FT: ravdess and subesco_980 frozen-comparator seed 1           ",
    "CTRL: transformer on subesco_980 replicate r=2                 ",
]

# ------------------------------------------------------------------ hypothesis registry
_COMMON = {
    "unit": "speaker", "statistic": "mean of speaker-paired differences (pp UAR)",
    "ci_method": "whole-speaker percentile bootstrap, 10000 reps, seed 20260903",
    "test": "two-sided Wilcoxon signed-rank (zero_method=wilcox); exact sign test added when >20% of speakers tie at zero",
    "aggregation": "average within speaker over replicates and over the named model levels BEFORE differencing",
    "missing_rule": "a voided cell (missing outer fold after one retry) voids its contrast; p=1 placeholder, family size unchanged, verdict 'not tested'",
    "truncation_rule": "units removed by the preregistered truncation order enter as p=1 with verdict 'not tested (truncated, deviation entry N)'",
}


def _h(hid, claim, family, level, models, contrast, definition, direction, status, sesoi="", n_units=None, note=""):
    return {"hypothesis_id": hid, "claim_id": claim, "family": family, "corpus_level": level,
            "model_levels": models, "contrast": contrast, "definition": definition,
            "direction": direction, "sidedness": "two-sided test, directional hypothesis" if direction != "none" else "two-sided",
            "a_priori_status": status, "sesoi_or_margin": sesoi, "n_units_expected": n_units, "power_note": note, **_COMMON}


HYPOTHESES = [
    # ---- Family N-decomp: SUBESCO-980 premium and its outer/inner decomposition (claims C2a, C2b)
    _h("N01", "C2a", "N-decomp", "subesco_980", "cnn,resnet_se,transformer", "RR-GG", "UAR(RR)-UAR(GG), scratch mean", ">0", "confirmatory", n_units=20),
    _h("N02", "C2a", "N-decomp", "subesco_980", "hubert_base,wavlm_base_plus,wav2vec2_base", "RR-GG", "UAR(RR)-UAR(GG), probe mean", ">0", "confirmatory", n_units=20, note="weakest member; S3 observed ~3 pp with WavLM fragile"),
    _h("N03", "C2b", "N-decomp", "subesco_980", "cnn,resnet_se,transformer", "RG-GG", "outer test leakage", ">0", "confirmatory", n_units=20),
    _h("N04", "C2b", "N-decomp", "subesco_980", "cnn,resnet_se,transformer", "RR-RG", "inner selection leakage under a leaky outer split", ">0", "confirmatory", n_units=20),
    _h("N05", "C3a", "N-mech", "cremad", "cnn,resnet_se", "mech prm-none", "prompt overlap at speaker-exclusive training, matched N", ">0", "confirmatory", n_units=91),
    _h("N06", "C3b", "N-mech", "cremad", "cnn,resnet_se", "mech spk-none", "cross-prompt speaker overlap at prompt-exclusive training, matched N", ">0", "confirmatory", n_units=91),
    _h("N07", "C3a", "N-mech", "subesco_full", "cnn,resnet_se", "mech prm-none", "prompt overlap at speaker-exclusive training, matched N", ">0", "confirmatory", n_units=20),
    _h("N08", "C3b", "N-mech", "subesco_full", "cnn,resnet_se", "mech spk-none", "cross-prompt speaker overlap at prompt-exclusive training, matched N", ">0", "confirmatory", n_units=20),
    _h("N09", "C3c", "N-mech", "subesco_full", "cnn,resnet_se", "mech both_sib-both", "sibling-take increment at matched N", ">0", "confirmatory", n_units=20),
    _h("N10", "C3d", "N-mech", "cremad_24 vs cremad_91m", "cnn,resnet_se", "[RR-GG](cremad_24)-[RR-GG](cremad_91m)", "speaker-count/density moderator at matched N; paired over speakers present in >=1 cremad_24 draw, averaged over draws within speaker", ">0", "confirmatory", n_units=None, note="expected ~70 paired speakers; sensitivity: bootstrap over draws"),
    _h("N11", "C4b", "N-modern", "ravdess", "wavlm_base_plus", "[RR-GG](ft)-[RR-GG](frozen_same_regime)", "re-exposure: fine-tuned premium minus frozen premium, same encoder/head/folds/seeds", ">0", "confirmatory", n_units=24, note="single partition; two seeds"),
    _h("N12", "C4b", "N-modern", "cremad", "wavlm_base_plus", "[RR-GG](ft)-[RR-GG](frozen_same_regime)", "re-exposure", ">0", "confirmatory", n_units=91),
    _h("N13", "C4b", "N-modern", "subesco_980", "wavlm_base_plus", "[RR-GG](ft)-[RR-GG](frozen_same_regime)", "re-exposure", ">0", "confirmatory", n_units=20),
    _h("N14", "S1", "N-supp", "cremad", "resnet_se", "[val-test](GR_hpo)-[val-test](GG_hpo)", "validation optimism of hyperparameter selection on a speaker-overlapping vs grouped inner validation, fixed speaker-exclusive outer test", ">0", "confirmatory", n_units=91),
    _h("N15", "S2", "N-supp", "ravdess", "hubert_base,wavlm_base_plus,wav2vec2_base", "[Ridge Random-Grouped]-[probe RG-GG]", "head effect at the outer boundary on identical CTRL partitions", "none", "confirmatory", n_units=24),
    # ---- Family R: preregistered re-tests of directions already observed on these speaker panels
    _h("R01", "C2-rep", "R", "ravdess", "cnn,resnet_se,transformer", "RR-GG", "pipeline premium", ">0", "confirmatory", n_units=24),
    _h("R02", "C2-rep", "R", "ravdess", "cnn,resnet_se,transformer", "RG-GG", "outer leakage", ">0", "confirmatory", n_units=24),
    _h("R03", "C2-rep", "R", "ravdess", "cnn,resnet_se,transformer", "RR-RG", "inner leakage", ">0", "confirmatory", n_units=24),
    _h("R04", "C2-rep", "R", "cremad", "cnn,resnet_se,transformer", "RR-GG", "pipeline premium", ">0", "confirmatory", n_units=91),
    _h("R05", "C2-rep", "R", "cremad", "cnn,resnet_se,transformer", "RG-GG", "outer leakage", ">0", "confirmatory", n_units=91),
    _h("R06", "C2-rep", "R", "cremad", "cnn,resnet_se,transformer", "RR-RG", "inner leakage", ">0", "confirmatory", n_units=91),
    _h("R07", "C2-rep", "R", "ravdess", "hubert_base,wavlm_base_plus,wav2vec2_base", "RR-GG", "probe premium", ">0", "confirmatory", n_units=24),
    _h("R08", "C2-rep", "R", "cremad", "hubert_base,wavlm_base_plus,wav2vec2_base", "RR-GG", "probe premium", ">0", "confirmatory", n_units=91),
    _h("R09", "C4a", "R", "ravdess", "wavlm_base_plus_ft", "RR-GG", "fine-tuned premium", ">0", "confirmatory", n_units=24),
    _h("R10", "C4a", "R", "cremad", "wavlm_base_plus_ft", "RR-GG", "fine-tuned premium", ">0", "confirmatory", n_units=91),
    _h("R11", "C4a", "R", "subesco_980", "wavlm_base_plus_ft", "RR-GG", "fine-tuned premium", ">0", "confirmatory", n_units=20),
    # ---- TOST pair (equivalence)
    _h("T01", "S3", "T", "cremad", "hubert_base,wavlm_base_plus,wav2vec2_base (Ridge A1)", "LOSO_sub-Group5", "size-matched LOSO equals 5-fold grouped", "equivalence", "confirmatory", sesoi="+/-1.0 pp", n_units=91),
    _h("T02", "S4", "T", "cremad", "cnn,resnet_se", "mech unexposed-half - none", "no spillover to unexposed co-fold speakers at half exposure", "equivalence", "confirmatory", sesoi="+/-1.5 pp", n_units=91),
]

# descriptors: estimates with CIs, never tested
DESCRIPTORS = [
    ("D01", "GR-GG allocation package on every family level (costs fit speakers; not inflation)"),
    ("D02", "per-level RR-GG sign counts (x/9 scratch, x/9 probe) and per-replicate estimates"),
    ("D03", "corpus differences RAVDESS-CREMA-D and RAVDESS-SUBESCO-980 with two-sample speaker bootstrap; +/-3 pp reading band"),
    ("D04", "sub_1400_one vs sub_700_one (cell-count effect at one take) and sub_700_two vs sub_700_one (take effect at fixed cells); the fixed-N one/two contrast is an allocation effect"),
    ("D05", "subesco_full random minus take-grouped premium on identical data"),
    ("D06", "mechanism both-none, interaction (both-none)-(spk-none)-(prm-none); crowding contrast G1-G05; share G05/G1"),
    ("D07", "cremad_24 vs cremad_91m raw premia per draw; variance over draws"),
    ("D08", "RR_hpo reported-number premium; selected-config distribution; test-set-selection optimism (max over configs of outer-test UAR minus honestly selected)"),
    ("D09", "honest outer-test DiD [GR_hpo-GG_hpo]-[GR_fixed-GG_fixed] (direction not prespecified)"),
    ("D10", "speaker-ID probe accuracy on test-fold speakers under RR vs GG checkpoints (scratch, frozen WavLM, fine-tuned WavLM)"),
    ("D11", "nearest-neighbour composition of training folds for test items (sibling / same-speaker / same-prompt / other)"),
    ("D12", "premium-by-layer (Ridge) per encoder x corpus with Spearman over state index"),
    ("D13", "Ridge direction stability over 20 draws (>=18/20 positive and q05>0)"),
    ("D14", "RAVDESS CNN and SUBESCO-980 CNN draw x seed variance components"),
    ("D15", "RAVDESS CNN LOSO, LOSO-sub, Group5 (like-for-like with Ibrahim et al. 2026)"),
    ("D16", "CNN RR/GG on the six P1 frozen splits next to the 2026 P1 rows (reconciliation)"),
    ("D17", "audit: Y_any and Y_test proportions, Wilson and partial-identification intervals, hypergeometric frame envelope, U bounds, execution-eligible fraction, realized overlap on the eligible subset, kappa with CI"),
    ("D18", "accuracy and macro-F1 released without inference"),
]

CLAIMS = {
    "C1": {"text": "Audit prevalence of speaker-nonexclusivity risk (Y_any primary, Y_test secondary) in a frozen-rule probability sample of public SER repository lineages",
           "type": "estimation", "requires": [], "verdict_rule": "always reported with intervals; no verdict"},
    "C2a": {"text": "Speaker-nonexclusive splitting inflates UAR on the repetition-clean SUBESCO-980 panel (scratch and probes)",
            "type": "confirmatory", "requires": ["N01"], "reported": ["N02"], "verdict_rule": "supported iff N01 rejects; N02 reported"},
    "C2b": {"text": "The premium decomposes into positive outer test leakage and positive inner selection leakage on SUBESCO-980",
            "type": "confirmatory", "requires": ["N03", "N04"], "verdict_rule": "supported iff both reject"},
    "C2-rep": {"text": "Same-implementation replication of the RAVDESS/CREMA-D direction (Ibrahim et al. 2026; Zielonka et al. 2022; our 2026 P1)",
               "type": "replication", "requires": ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08"], "verdict_rule": "supported iff all eight reject; per-test verdicts in evidence; not a headline claim"},
    "C3a": {"text": "Prompt/content overlap alone inflates UAR at speaker-exclusive training", "type": "confirmatory",
            "requires_any": ["N05", "N07"], "verdict_rule": "supported iff >=1 rejects and no opposite-sign rejection"},
    "C3b": {"text": "Cross-prompt speaker overlap alone inflates UAR at prompt-exclusive training", "type": "confirmatory",
            "requires_any": ["N06", "N08"], "verdict_rule": "supported iff >=1 rejects and no opposite-sign rejection"},
    "C3c": {"text": "Sibling takes add a further increment at matched N", "type": "confirmatory", "requires": ["N09"], "verdict_rule": "supported iff N09 rejects"},
    "C3d": {"text": "At matched N, fewer speakers with higher per-speaker density inflate more", "type": "confirmatory", "requires": ["N10"], "verdict_rule": "supported iff N10 rejects"},
    "C4a": {"text": "The premium persists under partial fine-tuning of WavLM-base+", "type": "replication", "requires_k_of": (2, ["R09", "R10", "R11"]), "verdict_rule": "supported iff >=2 of 3 reject"},
    "C4b": {"text": "Fine-tuning re-exposes speaker information relative to the frozen same-regime comparator", "type": "confirmatory", "requires_k_of": (2, ["N11", "N12", "N13"]), "verdict_rule": "supported iff >=2 of 3 reject; wording without D10 is 'fine-tuning changes the premium'"},
    "S1": {"text": "Supplement: hyperparameter selection on a speaker-overlapping validation set is optimistic", "type": "confirmatory", "requires": ["N14"], "verdict_rule": "supported iff N14 rejects"},
    "S2": {"text": "Supplement: the frozen-probe outer-boundary premium is head-dependent", "type": "confirmatory", "requires": ["N15"], "verdict_rule": "supported iff N15 rejects"},
    "S3": {"text": "Supplement: LOSO advantage equals a training-set-size effect on CREMA-D frozen features", "type": "equivalence", "requires": ["T01"], "verdict_rule": "supported iff TOST rejects both one-sided nulls"},
    "S4": {"text": "Supplement: no spillover to unexposed co-fold speakers", "type": "equivalence", "requires": ["T02"], "verdict_rule": "supported iff TOST rejects both one-sided nulls"},
}

FAMILY_ALPHA = 0.05
FORBIDDEN_CLAIMS = [
    "prevalence x effect multiplication or a universal correction factor",
    "'more than half' or any threshold statement about the audit",
    "'uses speaker identity' / 'uses voiceprint' / mediation shares",
    "'monotone dose curve'",
    "language-causal statements for SUBESCO",
    "reuse of 2026 P1/S0/S2/S3 numbers as updates (cited once as prior replication; reconciled via D16)",
    "any per-level (single model) test; any accuracy/macro-F1 inference",
]


FAMILY_POLICY = ("Families are claim-wise: one Holm family per headline claim group; FWER 0.05 within each family; "
                 "no program-wise error rate is claimed. Only hypotheses with a_priori_status == 'confirmatory' enter a family; "
                 "hypotheses demoted a priori by the simulated power table (power_holm < 0.5) are reported as estimates with CIs.")


def apply_power_table(power_table: dict | None) -> None:
    """Set a_priori_status from registry/power_table.json (frozen at tag-1)."""
    if not power_table:
        return
    for h in HYPOTHESES:
        row = power_table.get("table", {}).get(h["hypothesis_id"])
        if row is None:
            continue
        h["power_holm"] = row["power_holm"]
        h["a_priori_status"] = "confirmatory" if row["power_holm"] >= 0.5 else "estimate"


def adapt_claims_to_status() -> dict:
    """Demoted hypotheses leave the `requires` lists (they are still reported); a claim with no
    confirmatory member left becomes an estimation claim. Recorded in the claim map at tag-1."""
    status = {h["hypothesis_id"]: h["a_priori_status"] for h in HYPOTHESES}
    out = {}
    for cid, c in CLAIMS.items():
        c2 = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in c.items()}
        demoted = []
        for key in ("requires", "requires_any"):
            if key in c2:
                keep = [h for h in c2[key] if status[h] == "confirmatory"]
                demoted += [h for h in c2[key] if status[h] != "confirmatory"]
                c2[key] = keep
        if "requires_k_of" in c2:
            k, ids = c2["requires_k_of"]
            keep = [h for h in ids if status[h] == "confirmatory"]
            demoted += [h for h in ids if status[h] != "confirmatory"]
            c2["requires_k_of"] = [min(k, len(keep)), keep] if keep else [0, []]
        if demoted:
            c2["reported"] = sorted(set(c2.get("reported", [])) | set(demoted))
            c2["demoted_a_priori"] = demoted
        has_conf = bool(c2.get("requires")) or bool(c2.get("requires_any")) or (c2.get("requires_k_of", [0, []])[1])
        if c2.get("type") != "estimation" and not has_conf:
            c2["type"] = "estimation"
            c2["verdict_rule"] = "estimation claim after a priori demotion: reported with intervals; no verdict"
        out[cid] = c2
    return out


def family_sizes() -> dict[str, int]:
    out: dict[str, int] = {}
    for h in HYPOTHESES:
        if h["a_priori_status"] == "confirmatory":
            out[h["family"]] = out.get(h["family"], 0) + 1
    return out


def validate() -> dict:
    sizes = family_sizes()
    ids = [h["hypothesis_id"] for h in HYPOTHESES]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate hypothesis ids")
    referenced = set()
    for c in CLAIMS.values():
        referenced.update(c.get("requires", []))
        referenced.update(c.get("requires_any", []))
        referenced.update(c.get("reported", []))
        if "requires_k_of" in c:
            referenced.update(c["requires_k_of"][1])
    unknown = referenced - set(ids)
    if unknown:
        raise ValueError(f"claims reference unknown hypotheses {sorted(unknown)}")
    unreferenced = set(ids) - referenced
    if unreferenced:
        raise ValueError(f"hypotheses not bound to any claim {sorted(unreferenced)}")
    return {"family_sizes": sizes, "n_hypotheses": len(ids), "n_claims": len(CLAIMS), "n_descriptors": len(DESCRIPTORS)}


def export(root: Path, power_table: dict | None = None) -> dict:
    apply_power_table(power_table)
    info = validate()
    sizes = info["family_sizes"]
    fields = ["hypothesis_id", "claim_id", "family", "family_size", "corpus_level", "model_levels", "contrast", "definition",
              "direction", "sidedness", "unit", "n_units_expected", "statistic", "ci_method", "test", "aggregation",
              "missing_rule", "truncation_rule", "sesoi_or_margin", "a_priori_status", "power_holm", "power_note"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for h in HYPOTHESES:
        row = dict(h)
        row["family_size"] = sizes.get(h["family"], 0)
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    atomic_write_text(root / "hypothesis_registry.csv", buf.getvalue())
    claims = adapt_claims_to_status()
    atomic_write_json(root / "claim_map.json", {"alpha": FAMILY_ALPHA, "family_sizes": sizes, "family_policy": FAMILY_POLICY, "claims": claims,
                                                "descriptors": DESCRIPTORS, "forbidden_claims": FORBIDDEN_CLAIMS})
    atomic_write_json(root / "arms.json", {"program_version": PROGRAM_VERSION, "arms": ARMS,
                                           "program_cap_gpu_hours": PROGRAM_CAP_GPU_HOURS,
                                           "truncation_order": [t.strip() for t in TRUNCATION_ORDER],
                                           "corpus_levels": CORPUS_LEVELS, "scratch_models": SCRATCH_MODELS,
                                           "probe_models": PROBE_MODELS, "replicates": REPLICATES,
                                           "cost_model": {"sec_per_train_utt": SEC_PER_TRAIN_UTT, "model_mult": MODEL_COST_MULT,
                                                          "probe_unit_sec": PROBE_UNIT_SEC, "ridge_unit_sec": RIDGE_UNIT_SEC,
                                                          "ft_fold_min_planning": FT_FOLD_MIN,
                                                          "frozen_same_regime_fraction": FROZEN_SAME_REGIME_FRACTION}})
    info["registry_sha256"] = sha256_json({"h": HYPOTHESES, "c": claims, "a": ARMS})
    return info

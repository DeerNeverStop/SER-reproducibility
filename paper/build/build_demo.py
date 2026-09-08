from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab import rl_config
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image as RLImage,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]          # E:\科研\ICASSP2027
# frozen evidence stays in the 07 project (read-only); paths absolute since
# the 2026-08-15 migration broke the old relative layout
PROJECT_07 = Path(r"E:\科研\claudework\07-ser-repro-protocol-audit")
# new build outputs use a new name so the migrated 2026-08-15 demo PDF is
# never overwritten (its sha is recorded in build_manifest.json)
OUTPUT_PDF = HERE / "icassp2027_protocol_audit_m2.pdf"

TIER_CSV = PROJECT_07 / "results" / "execution_validation" / "tier_summary.csv"
RUN_STATUS_JSON = PROJECT_07 / "results" / "protocol_premium" / "run_status_summary.json"
VERIFICATION_JSON = (
    PROJECT_07 / "results" / "protocol_premium" / "summary" / "verification.json"
)
PREMIUM_CSV = (
    PROJECT_07 / "results" / "protocol_premium" / "summary" / "protocol_premium.csv"
)
METRIC_CSV = (
    PROJECT_07
    / "results"
    / "protocol_premium"
    / "summary"
    / "protocol_metric_summary.csv"
)
PREREG = PROJECT_07 / "P1_PREREGISTRATION.md"
# frozen one-shot D1 scoring report (dialogue records 64-66); the builder
# re-hashes and parses it live -- D1 facts derive from certified bytes
D1_REPORT = (
    PROJECT / "results" / "corpora" / "ravdess-mode-matched" / "scoring"
    / "d1_scoring_report.json"
)
D1_REPORT_SHA = "aaab7a0be4feedf190d65953651c121a230fe8bd43e8d45b9f1e084c7214e142"
# frozen D1 completion manifest v6 (certifies the 792-fit design count)
D1_COMPLETION = (
    PROJECT / "results" / "corpora" / "ravdess-mode-matched" / "training"
    / "completion_manifest_v6.json"
)
D1_COMPLETION_SHA = "f3b5ce8c785cddd463b8bbd60e7cb2b9e328e91641b6e114a433030cbec9aa9d"

FIGURE_PNG = PROJECT / "figures" / "figure_protocol_effects.png"
MARKDOWN_OUT = HERE / "paper_demo.md"
LATEX_OUT = HERE / "main_provisional.tex"
NUMBERCHECK_OUT = HERE / "main_numbercheck.tex"
FACTS_OUT = HERE / "paper_facts_07.json"
MANIFEST_OUT = HERE / "build_manifest.json"

TITLE = (
    "SPEAKER-NONEXCLUSIVE EVALUATION IN PUBLIC SER CODE: "
    "A PREREGISTERED AUDIT AND CONTROLLED STUDY OF PROTOCOL EFFECTS"
)
AUTHOR = "[AUTHOR NAME]"
AFFILIATION = "[AFFILIATION]"
EMAIL = "[EMAIL]"

MODEL_ORDER = ["cnn", "resnet_se", "transformer", "fno"]
MODEL_LABEL = {
    "cnn": "CNN",
    "resnet_se": "ResNet-SE",
    "transformer": "Transformer",
    "fno": "FNO",
}
CORPUS_ORDER = ["ravdess", "cremad"]
CORPUS_LABEL = {"ravdess": "RAVDESS", "cremad": "CREMA-D"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_evidence() -> dict:
    tier_rows = load_csv(TIER_CSV)
    tier = {
        (r["dimension"], r["category"]): int(r["count"]) for r in tier_rows
    }
    expected_tier = {
        ("evidence_tier", "tier1_execution_confirmed"): 24,
        ("evidence_tier", "tier2_static_execution_inference"): 1,
        ("evidence_tier", "tier3_unknown"): 5,
        ("split_risk_final", "yes"): 20,
        ("split_risk_final", "no"): 5,
        ("split_risk_final", "unknown"): 5,
    }
    if any(tier.get(k) != v for k, v in expected_tier.items()):
        raise RuntimeError(f"Frozen audit contract changed: {tier}")

    run = json.loads(RUN_STATUS_JSON.read_text(encoding="utf-8"))
    if run.get("planned") != 1620 or run.get("success") != 1620:
        raise RuntimeError(f"Frozen fit contract changed: {run}")
    if run.get("failed_after_retry") != 0 or run.get("pending") != 0:
        raise RuntimeError(f"Frozen run is not complete: {run}")

    verification = json.loads(VERIFICATION_JSON.read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise RuntimeError(f"Verification is not PASS: {verification}")
    if verification.get("counts", {}).get("complete_cells") != 72:
        raise RuntimeError(f"OOF cell contract changed: {verification}")
    required_checks = {
        "all_output_hashes_match",
        "fixed_holm_family_24",
        "full_cell_missingness_propagation",
        "raw_utterance_reanalysis_exact_match",
        "speaker_paired_direction",
        "strict_protocol_speaker_disjointness",
    }
    if any(not verification["checks"].get(k) for k in required_checks):
        raise RuntimeError(f"Verification check failed: {verification['checks']}")

    premium_rows = load_csv(PREMIUM_CSV)
    rg = {}
    for row in premium_rows:
        if (
            row["contrast"] == "RG"
            and row["metric"] == "uar"
            and row["status"] == "complete"
        ):
            key = (row["corpus"], row["model"])
            rg[key] = {
                "delta": float(row["point_mean_difference"]),
                "ci_low": float(row["ci95_low"]),
                "ci_high": float(row["ci95_high"]),
                "p_holm": float(row["holm_adjusted_p"]),
                "holm_reject": row["holm_reject_0_05"].lower() == "true",
                "n_speakers": int(row["n_speakers"]),
                "bootstrap_replicates": int(row["bootstrap_replicates"]),
            }
    expected_keys = {(c, m) for c in CORPUS_ORDER for m in MODEL_ORDER}
    if set(rg) != expected_keys or not all(v["holm_reject"] for v in rg.values()):
        raise RuntimeError(f"Frozen RG-UAR contract changed: {sorted(rg)}")

    holm_significant = sum(
        1
        for row in premium_rows
        if row["metric"] == "uar"
        and row["status"] == "complete"
        and row.get("holm_reject_0_05", "").lower() == "true"
    )
    if holm_significant != 19:
        raise RuntimeError(f"Expected 19/24 Holm-significant UAR tests, got {holm_significant}")

    metric_rows = load_csv(METRIC_CSV)
    metrics = {}
    for row in metric_rows:
        if row["metric"] != "uar" or row["status"] != "complete":
            continue
        metrics[(row["corpus"], row["model"], row["protocol"])] = {
            "mean": float(row["mean"]),
            "sd": float(row["sample_sd"]),
        }
    expected_metric_keys = {
        (c, m, p)
        for c in CORPUS_ORDER
        for m in MODEL_ORDER
        for p in ("random", "groupkfold", "loso")
    }
    if set(metrics) != expected_metric_keys:
        raise RuntimeError("Frozen UAR metric table is incomplete")

    prereg_text = PREREG.read_text(encoding="utf-8")
    prereg_contracts = [
        "1,440",
        "7,442",
        "24",
        "91",
        "22050",
        "64",
        "1024",
        "512",
        "128",
        "10,000",
    ]
    missing = [token for token in prereg_contracts if token not in prereg_text]
    if missing:
        raise RuntimeError(f"Preregistration contract tokens missing: {missing}")

    # -- frozen D1 report: certify bytes, then derive every D1 fact -------
    import hashlib as _hashlib
    d1_bytes = D1_REPORT.read_bytes()
    d1_observed = _hashlib.sha256(d1_bytes).hexdigest()
    if d1_observed != D1_REPORT_SHA:
        raise RuntimeError(
            f"frozen D1 report sha {d1_observed} != expected {D1_REPORT_SHA}")
    d1_doc = json.loads(d1_bytes.decode("utf-8"))
    models = d1_doc["models"]
    if set(models) != {"cnn", "resnet_se", "transformer", "fno"}:
        raise RuntimeError("D1 report model set drifted")
    prg_speech = {m: models[m]["point_estimates"]["P_RG_speech"] * 100 for m in models}
    prg_song = {m: models[m]["point_estimates"]["P_RG_song"] * 100 for m in models}
    for m in models:
        if not (prg_speech[m] > 0 and prg_song[m] > 0):
            raise RuntimeError(f"D1 {m}: P_RG not positive in both channels")
        if not models[m]["point_estimates"]["D_mode_RG"] < 0:
            raise RuntimeError(f"D1 {m}: D_mode_RG point estimate not negative")
        rw_rl = models[m]["actor_reweighting"]["D_mode_RL"]
        if not (rw_rl["q2_5"] < 0 < rw_rl["q97_5"]):
            raise RuntimeError(f"D1 {m}: D_mode_RL stability range does not "
                                "cross zero; not_conclusive wording invalid")
    # the paper sentence names THREE diagnostics (seed, leave-one-actor-out,
    # actor-reweighting); all three are asserted here (record 79 actions 1-2)
    rw = {m: models[m]["actor_reweighting"]["D_mode_RG"] for m in models}
    sw = {m: models[m]["seedwise_panel_estimates"]["D_mode_RG"] for m in models}
    lo = {m: models[m]["leave_one_actor_out"]["D_mode_RG"] for m in models}
    if not (rw["cnn"]["q2_5"] < 0 < rw["cnn"]["q97_5"]):
        raise RuntimeError("D1 cnn: reweighting range does not cross zero; "
                            "not_conclusive wording invalid")
    if not (min(sw["cnn"]) < 0 < max(sw["cnn"])):
        raise RuntimeError("D1 cnn: seedwise values do not cross zero; "
                            "not_conclusive wording invalid")
    if not (lo["cnn"]["min"] < 0 < lo["cnn"]["max"]):
        raise RuntimeError("D1 cnn: LOAO range does not cross zero; "
                            "not_conclusive wording invalid")
    for m in ("resnet_se", "transformer", "fno"):
        if not rw[m]["q97_5"] < 0:
            raise RuntimeError(f"D1 {m}: reweighting range crosses zero; "
                                "'directionally stable' wording invalid")
        if not all(v < 0 for v in sw[m]):
            raise RuntimeError(f"D1 {m}: a seedwise value is non-negative; "
                                "'directionally stable' wording invalid")
        if not lo[m]["max"] < 0:
            raise RuntimeError(f"D1 {m}: LOAO upper bound non-negative; "
                                "'directionally stable' wording invalid")
    d1c_bytes = D1_COMPLETION.read_bytes()
    d1c_observed = _hashlib.sha256(d1c_bytes).hexdigest()
    if d1c_observed != D1_COMPLETION_SHA:
        raise RuntimeError(
            f"frozen D1 completion sha {d1c_observed} != expected {D1_COMPLETION_SHA}")
    d1c_doc = json.loads(d1c_bytes.decode("utf-8"))
    d1 = {
        "actors": len(d1_doc["actors"]),
        "items_per_channel": int(d1_doc["n_samples_per_channel"]),
        "fits": len(d1c_doc["units"]),
        "prg_speech_low": round(min(prg_speech.values()), 2),
        "prg_speech_high": round(max(prg_speech.values()), 2),
        "prg_song_low": round(min(prg_song.values()), 2),
        "prg_song_high": round(max(prg_song.values()), 2),
    }
    if (d1["actors"], d1["items_per_channel"], d1["fits"]) != (23, 1012, 792):
        raise RuntimeError(f"D1 panel constants drifted: {d1}")

    return {
        "tier": tier,
        "run": run,
        "verification": verification,
        "rg": rg,
        "metrics": metrics,
        "holm_significant": holm_significant,
        "d1": d1,
    }


def p(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}"


def manuscript(e: dict) -> dict:
    rg = e["rg"]
    rav = [100 * rg[("ravdess", m)]["delta"] for m in MODEL_ORDER]
    cre = [100 * rg[("cremad", m)]["delta"] for m in MODEL_ORDER]

    abstract = (
        "Speaker-independent generalization is often intended in speech emotion "
        "recognition (SER), but public implementations may use speaker-nonexclusive "
        "evaluation. We separate how often this risk appears in public code from how "
        "strongly protocol choice changes measured performance. A preregistered probability "
        "sample included 30 eligible repositories. Twenty showed speaker-nonexclusive split "
        "risk. Five did not, and five were unresolved. The observed rate was 66.7% (Wilson "
        "95% CI [48.8%, 80.8%]). We then ran 1,620 fits on two corpora with four architectures, "
        "three protocols, and three seeds. The speaker-paired Random-minus-GroupKFold UAR "
        f"difference ranged from {min(rav):.2f} percentage points (FNO) to {max(rav):.2f} "
        f"(ResNet-SE) on RAVDESS, and from {min(cre):.2f} (FNO) to {max(cre):.2f} "
        "(Transformer) on CREMA-D. All eight contrasts survived Holm correction. The results "
        "characterize a sampled practice and two tested corpora, not a universal corpus law."
    )
    if not (100 <= len(re.findall(r"\b[\w-]+\b", abstract)) <= 150):
        raise RuntimeError("Abstract must contain 100-150 words")

    columns = [
        [
            ("center", "ABSTRACT"),
            ("body", abstract),
            (
                "body",
                "<b><i>Index Terms-</i></b> speech emotion recognition, evaluation "
                "protocol, speaker independence, reproducibility, empirical audit",
            ),
            ("section", "1. INTRODUCTION"),
            (
                "body",
                "Speech emotion recognition systems are often interpreted as measuring "
                "generalization to unseen speakers. That interpretation requires training "
                "and evaluation partitions to be speaker-exclusive. If one speaker can "
                "occur on both sides, the estimand instead concerns new utterances from "
                "speakers represented during training. Such a protocol can be legitimate "
                "for a speaker-dependent application, but it does not alone support an "
                "unseen-speaker claim.",
            ),
            (
                "body",
                "Prior studies show that SER performance depends on partition construction "
                "[1]-[5], while benchmark efforts increasingly distribute fixed "
                "speaker-independent splits [7], [8]. The closest prior study, from 2026, "
                "already contrasts random and speaker-independent evaluation on RAVDESS "
                "and CREMA-D and reports a larger change on RAVDESS [5]. Therefore, neither "
                "the existence of a split effect nor its occurrence on these corpora is "
                "claimed as a first discovery here.",
            ),
        ],
        [
            (
                "body",
                "We connect two complementary measurements. First, we estimate how often "
                "speaker-nonexclusive split risk appears in a preregistered probability "
                "sample from a frozen public-code candidate frame. Evidence is tied to "
                "executable split behavior or directly traceable code. Second, we quantify "
                "protocol sensitivity under one implementation: two corpora, four fixed "
                "architectures, three evaluation protocols, three seeds, complete "
                "out-of-fold prediction, and speaker-paired inference.",
            ),
            (
                "body",
                "We contribute a preregistered audit of 30 eligible public SER "
                "repositories. We add a controlled study "
                "with 1,620 valid fits. Finally, speaker-cluster intervals and family-wise "
                "multiplicity control show consistently positive, but sharply different, "
                "Random-GroupKFold UAR effects in the two tested corpora.",
            ),
            (
                "body",
                "We use <i>speaker-nonexclusive split risk</i> descriptively. It does not "
                "imply that a model demonstrably exploited identity, that every reported "
                "result is invalid, or that every repository intended unseen-speaker "
                "generalization.",
            ),
            ("section", "2. RELATION TO PRIOR WORK"),
            (
                "body",
                "SER evaluation has long distinguished speaker-dependent from "
                "speaker-independent settings [1]. Direct comparisons later showed that "
                "fold criteria alter absolute performance and sometimes model rankings "
                "[2]-[5]. Reproduction work has also found that public SER systems can be "
                "hard to reconstruct under one common protocol [6]. Antoniou et al. [6] "
                "audit a purposive IEMOCAP case study; our sampling estimand is instead a "
                "frozen public-code candidate frame.",
            ),
            (
                "body",
                "SERAB [7] and Open-Emotion/EMO-SUPERB [8] reduce ambiguity by publishing "
                "standardized speaker-independent partitions. They address benchmark "
                "construction rather than the prevalence of observable split behavior in "
                "sampled implementations. Meyer et al. [9] provide a related warning that "
                "dataset splits can expose non-emotion shortcuts. Our study is narrower: "
                "speaker exclusivity, code-practice prevalence, and paired protocol effects.",
            ),
        ],
        [
            ("section", "3. METHODS"),
            ("subsection", "3.1. Public-code audit"),
            (
                "body",
                "The repository candidate frame, probability sample, and classification "
                "procedure were frozen before outcome analysis. The unit was a unique "
                "repository. Deterministic SHA-256 ordering with seed 202608101744 was "
                "applied to 331 pending candidates from a 383-item deduplicated frame; "
                "screening ranks 1-34 yielded 30 eligible repositories and four exclusions.",
            ),
            (
                "body",
                "A repository was positive when the evaluated path did not enforce speaker "
                "exclusivity or produced speaker overlap on the frozen synthetic manifest; "
                "negative when exclusivity was verified; and unknown when available evidence "
                "could not resolve the property. Twenty-four repositories had highest-tier "
                "executable evidence, one had traceable static evidence, and five remained "
                "unknown. Highest-tier fragments passed 384 finite property cases. Each "
                "repository contributed 16 cases. We executed only the decisive frozen split fragment, not the "
                "entire repository.",
            ),
            (
                "body",
                "The audit denominator was 30 repositories. We report the positive fraction "
                "with a two-sided Wilson interval. Unresolved repositories stay unresolved; a "
                "separate sensitivity bound treats all five as positive. The target is the "
                "frozen public-code candidate frame, not all SER papers or software.",
            ),
            ("subsection", "3.2. Corpora and preprocessing"),
            (
                "body",
                "RAVDESS contributed 1,440 utterances from 24 speakers over eight "
                "classes [10]; CREMA-D contributed 7,442 utterances from 91 speakers "
                "over six classes [11]. Audio was converted to 22.05 kHz mono and trimmed at 30 dB. "
                "Inputs were 64-bin log-mel features (FFT 1,024; hop 512; upper frequency "
                "11,025 Hz; 128 frames) with per-utterance standardization. SpecAugment was "
                "restricted to training data.",
            ),
        ],
        [
            ("subsection", "3.3. Models, protocols, and inference"),
            (
                "body",
                "Four fixed probes were used: CNN, ResNet-SE, Transformer, and Fourier "
                "neural operator. The architecture and optimization contracts were fixed "
                "across protocols. AdamW used batch size 64, at most 100 epochs, early "
                "stopping patience 15, and seeds 0, 1, and 2. No test partition guided model "
                "selection. These probes do not exhaust modern SER model space; their role "
                "is to hold implementation constant while changing protocol.",
            ),
            (
                "body",
                "Random used five-fold StratifiedKFold (seed 42) without a speaker grouping "
                "constraint. Grouped used five-fold GroupKFold by speaker. LOSO held out one "
                "speaker at a time (24 RAVDESS folds; 91 CREMA-D folds). Inner validation "
                "mirrored each protocol: ungrouped under Random, speaker-grouped under the "
                "strict protocols. Thus Random-Grouped changes both outer speaker overlap and "
                "inner-validation grouping; it is a pipeline-level protocol contrast, not a "
                "pure causal estimate of outer-test overlap.",
            ),
            (
                "body",
                "Full out-of-fold predictions were retained for every corpus-model-protocol-"
                "seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for "
                "RAVDESS; 1,212 for CREMA-D). UAR was primary. The principal contrast was "
                "Random minus GroupKFold; Random-LOSO and Grouped-LOSO were secondary "
                "members of the same frozen family.",
            ),
            (
                "body",
                "Speaker was the paired unit. Each speaker was first averaged over three "
                "seeds. Confidence intervals used 10,000 whole-speaker bootstrap replicates. "
                "Two-sided Wilcoxon tests were adjusted by Holm. The multiplicity family size "
                "was 24 preregistered UAR tests. Accuracy and macro-F1 were "
                "descriptive only.",
            ),
            (
                "body",
                "All Random folds placed every corpus speaker on both sides of the outer "
                "split, whereas Grouped and LOSO outer overlap was zero. This verifies the "
                "protocol property; it does not by itself show that a trained classifier used "
                "speaker identity.",
            ),
        ],
        [
            ("section", "4. RESULTS"),
            ("subsection", "4.1. Audit prevalence"),
            (
                "body",
                "The sample contained 30 repositories. Twenty showed speaker-nonexclusive "
                "split risk. Five were verified negative, and five were unresolved. The observed "
                "positive fraction was 66.7% (Wilson 95% CI [48.8%, 80.8%]). Under an "
                "unknown-as-positive sensitivity analysis, the positive count was 25. The "
                "denominator remained 30. The sensitivity rate was 83.3% (Wilson 95% CI "
                "[66.4%, 92.7%]). The primary estimate retains unknown cases in the "
                "denominator rather than silently converting them to negatives.",
            ),
            ("figure", "protocol"),
            (
                "caption",
                "Fig. 1. (A) Frozen audit outcomes for the preregistered probability "
                "sample; the estimand is the frozen public-code candidate frame, not "
                "all SER work. The interval is the Wilson 95% CI for 20/30. (B) "
                "Speaker-paired Random-minus-GroupKFold UAR differences with "
                "10,000-replicate speaker-cluster intervals. These eight contrasts "
                "belong to the original frozen 24-test Holm family; all eight survived "
                "correction, and the corpus contrast is descriptive for these pipelines.",
            ),
        ],
        [
            ("subsection", "4.2. Controlled protocol effects"),
            (
                "body",
                "Random five-fold UAR exceeded GroupKFold UAR in all eight combinations. "
                f"The speaker-paired difference ranged from {min(rav):.2f} percentage "
                f"points (FNO) to {max(rav):.2f} (ResNet-SE) on RAVDESS, and from "
                f"{min(cre):.2f} (FNO) to {max(cre):.2f} (Transformer) on CREMA-D; the "
                "remaining models lay between these endpoints. All eight Random-Grouped "
                "tests survived correction. Across all three protocol contrasts, the Holm "
                "rejections numbered 19. The multiplicity family size was 24 tests.",
            ),
            ("table", "uar"),
            (
                "caption",
                "Table 1. UAR (%) mean ± sample SD over three seeds. Delta is the "
                "speaker-paired Random-Grouped difference with 95% cluster-bootstrap CI.",
            ),
            (
                "body",
                "LOSO was not uniformly below both five-fold protocols; for example, "
                "CREMA-D FNO had slightly higher mean UAR under LOSO than Random. Protocols "
                "are therefore not a simple ordinal scale of strictness. Grouped is the most "
                "directly matched outer-fold comparison to Random; LOSO also changes fold "
                "count, test composition, and training size.",
            ),
            (
                "body",
                "The nonoverlapping effect ranges show sharply different protocol "
                "sensitivity between the two tested corpora. This does not identify a cause. "
                "Speaker count, recording design, acted content, and within-speaker structure "
                "are candidate moderators, but none is isolated by a two-corpus design.",
            ),
            (
                "body",
                "On a matched 23-actor RAVDESS speech-song panel (six shared emotions, "
                "1,012 items per channel, 792 fits), the Random-minus-GroupKFold "
                "protocol-package difference was positive in both channels for all four "
                "models (speech +11.87 to +16.12, song +9.45 to +11.59 UAR points). The "
                "song-minus-speech difference of these premiums was negative for all four "
                "models as a point estimate; it was directionally stable in this finite "
                "panel for ResNet-SE, Transformer, and FNO under seed, leave-one-actor-out, "
                "and actor-reweighting diagnostics (stability descriptions, not confidence "
                "intervals), and not conclusive for CNN; the four secondary "
                "song-minus-speech differences of the Random-minus-LOSO premiums were "
                "also not conclusive.",
            ),
        ],
        [
            ("section", "5. DISCUSSION"),
            (
                "body",
                "The audit and experiment answer different questions. The audit estimates "
                "how frequently a verifiable split risk appears in one sampled public-code "
                "frame. The experiment estimates how much one frozen pipeline changes when "
                "the evaluation protocol changes. Neither substitutes for the other: a common "
                "practice need not have a large effect in every corpus, and a large controlled "
                "effect does not establish prevalence.",
            ),
            (
                "body",
                "The RAVDESS-CREMA-D pattern is directionally consistent with the closest "
                "exact-corpus prior study [5]. We consequently frame this result as a "
                "replication and extension, not discovery. The added value is the linked "
                "design: probability sampling for practice prevalence, executable repository "
                "evidence, a common four-model implementation, complete OOF prediction, "
                "speaker-paired uncertainty, and preregistered multiplicity control.",
            ),
            (
                "body",
                "The operational recommendation is narrow. If a reported SER score is "
                "intended to describe unseen speakers, the evaluation should enforce "
                "speaker exclusivity and publish grouping information sufficient to reproduce "
                "that property. Random utterance splitting may still answer a legitimate "
                "speaker-dependent question, but the target population should be explicit.",
            ),
            (
                "body",
                "No universal correction should be subtracted from published scores. The "
                "reported protocol-effect ranges arise from specific corpora, models, "
                "preprocessing, training recipe, and split constructions. Multiplying the "
                "audit prevalence by these effects would combine incompatible estimands and "
                "is intentionally excluded.",
            ),
            ("subsection", "5.1. Limitations"),
            (
                "body",
                "The audit covers one GitHub/arXiv-dominated candidate frame and 30 sampled "
                "repositories, with five unresolved. Search availability and eligibility "
                "limit extrapolation. A positive label describes the inspected split mechanism "
                "or its behavior on a frozen synthetic manifest; it does not prove the exact "
                "speaker overlap in an associated paper's reported run.",
            ),
        ],
        [
            (
                "body",
                "Speaker-nonexclusive evaluation makes identity-related information available "
                "across partitions, but does not show that a classifier used identity or that "
                "score differences are caused solely by identity. The controlled experiment "
                "uses two acted English corpora, four non-SSL probes, three optimization "
                "seeds, and one frozen outer partition per protocol. Its confidence intervals "
                "do not cover alternative split draws or other corpora.",
            ),
            (
                "body",
                "Hyperparameters were fixed across protocols, so we estimate a protocol "
                "difference under one training recipe rather than each protocol's independently "
                "optimized best score. Random-Grouped also changes inner validation grouping. "
                "The speaker-cluster intervals quantify uncertainty within the observed actors, "
                "not a superpopulation of recording designs.",
            ),
            ("section", "6. CONCLUSION"),
            (
                "body",
                "A preregistered probability audit found speaker-nonexclusive split risk in "
                "20 sampled public SER repositories. The sample contained 30 repositories, "
                "with five verified negatives and "
                "five unresolved cases. In a controlled 1,620-fit study, Random five-fold "
                "evaluation produced higher UAR than GroupKFold for all four probes on both "
                "corpora, but the premium was much larger on RAVDESS than CREMA-D. These "
                "findings support explicit speaker grouping when unseen-speaker generalization "
                "is the target. They characterize a sampled coding practice and specified "
                "conditions, not all SER systems.",
            ),
            ("section", "7. REPRODUCIBILITY STATEMENT"),
            (
                "body",
                "Every number, table, and figure in this manuscript was generated "
                "directly from frozen audit and experiment tables. "
                "The build fails closed unless the final adjudication, successful-fit total, "
                "complete OOF cells, verification checks, and eight RG-UAR rows match their "
                "registered sources. Repository-level audit evidence, split manifests, "
                "speaker-level predictions, and independent statistic verification are "
                "retained in the project artifact bundle. A release URL will be inserted "
                "after anonymization and archival packaging are finalized.",
            ),
            (
                "body",
                "<b>Submission note.</b> ICASSP 2027 uses single-anonymous review. The "
                "placeholder author and affiliation above must be replaced, and the official "
                "2027 template must supersede this provisional layout before submission.",
            ),
        ],
    ]

    references = [
        "B. W. Schuller, S. Steidl, and A. Batliner, \"The INTERSPEECH 2009 Emotion Challenge,\" in <i>Proc. Interspeech</i>, pp. 312-315, 2009.",
        "L. Pepino, P. Riera, L. Ferrer, and A. Gravano, \"Fusion approaches for emotion recognition from speech using acoustic and text-based features,\" in <i>Proc. ICASSP</i>, pp. 6484-6488, 2020.",
        "B. T. Atmaja and A. Sasou, \"Effect of different splitting criteria on the performance of speech emotion recognition,\" in <i>Proc. TENCON</i>, pp. 760-764, 2021.",
        "M. Zielonka et al., \"Recognition of emotions in speech using convolutional neural networks on different datasets,\" <i>Electronics</i>, vol. 11, no. 22, Art. 3831, 2022.",
        "E. Ibrahim, M. E. Ghoraba, and A. E. Ghoraba, \"Multimodal emotion recognition using hybrid deep feature fusion under speaker-independent evaluation,\" <i>Scientific Reports</i>, vol. 16, Art. 19584, 2026.",
        "N. Antoniou, A. Katsamanis, T. Giannakopoulos, and S. Narayanan, \"Designing and evaluating speech emotion recognition systems: A reality check case study with IEMOCAP,\" in <i>Proc. ICASSP</i>, pp. 1-5, 2023.",
        "N. Scheidwasser-Clow, M. Kegler, P. Beckmann, and M. Cernak, \"SERAB: A multi-lingual benchmark for speech emotion recognition,\" in <i>Proc. ICASSP</i>, pp. 7697-7701, 2022.",
        "H. Wu et al., \"Open-Emotion: A reproducible EMO-SUPERB for speech emotion recognition systems,\" in <i>Proc. IEEE SLT</i>, pp. 510-517, 2024.",
        "P. Meyer, E. Buschermohle, and T. Fingscheidt, \"What do classifiers actually learn? A case study on emotion recognition datasets,\" in <i>Proc. Interspeech</i>, pp. 262-266, 2018.",
        "S. R. Livingstone and F. A. Russo, \"The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS),\" <i>PLOS ONE</i>, vol. 13, no. 5, Art. e0196391, 2018.",
        "H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, \"CREMA-D: Crowd-sourced emotional multimodal actors dataset,\" <i>IEEE Trans. Affect. Comput.</i>, vol. 5, no. 4, pp. 377-390, 2014.",
    ]

    return {
        "abstract": abstract,
        "columns": columns,
        "references": references,
        "rav_range": (min(rav), max(rav)),
        "cre_range": (min(cre), max(cre)),
    }


def register_fonts() -> None:
    font_dir = Path(r"C:\Windows\Fonts")
    fonts = {
        "TimesNewRoman": font_dir / "times.ttf",
        "TimesNewRoman-Bold": font_dir / "timesbd.ttf",
        "TimesNewRoman-Italic": font_dir / "timesi.ttf",
        "TimesNewRoman-BoldItalic": font_dir / "timesbi.ttf",
    }
    for name, path in fonts.items():
        if not path.is_file():
            raise RuntimeError(f"Required embedded font missing: {path}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "TimesNewRoman",
        normal="TimesNewRoman",
        bold="TimesNewRoman-Bold",
        italic="TimesNewRoman-Italic",
        boldItalic="TimesNewRoman-BoldItalic",
    )
    rl_config.canvas_basefontname = "TimesNewRoman"


def pil_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size=size)


def draw_figure(e: dict) -> None:
    width, height = 1700, 1540
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    regular = pil_font("times.ttf", 48)
    small = pil_font("times.ttf", 42)
    tiny = pil_font("times.ttf", 36)
    bold = pil_font("timesbd.ttf", 52)
    bold_small = pil_font("timesbd.ttf", 42)

    left, right = 120, width - 70
    d.text((left, 25), "A  PUBLIC-CODE AUDIT", font=bold, fill="black")
    bar_y0, bar_y1 = 135, 260
    bar_w = right - left
    counts = [(20, "YES", "#3569a8"), (5, "NO", "#4c956c"), (5, "UNKNOWN", "#b8b8b8")]
    x = left
    for count, label, color in counts:
        w = bar_w * count / 30
        d.rectangle((x, bar_y0, x + w, bar_y1), fill=color, outline="black", width=3)
        label_box = d.textbbox((0, 0), label, font=bold_small)
        count_text = f"{count}/30"
        count_box = d.textbbox((0, 0), count_text, font=tiny)
        text_color = "white" if label != "UNKNOWN" else "black"
        d.text(
            (x + (w - (label_box[2] - label_box[0])) / 2, bar_y0 + 14),
            label,
            font=bold_small,
            fill=text_color,
        )
        d.text(
            (x + (w - (count_box[2] - count_box[0])) / 2, bar_y0 + 68),
            count_text,
            font=tiny,
            fill=text_color,
        )
        x += w
    axis_y = 365
    d.line((left, axis_y, right, axis_y), fill="black", width=3)
    for pct in (0, 25, 50, 75, 100):
        xx = left + bar_w * pct / 100
        d.line((xx, axis_y - 12, xx, axis_y + 12), fill="black", width=3)
        label = f"{pct}%"
        box = d.textbbox((0, 0), label, font=tiny)
        d.text((xx - (box[2] - box[0]) / 2, axis_y + 18), label, font=tiny, fill="black")
    ci_x0 = left + bar_w * 0.488
    ci_x1 = left + bar_w * 0.808
    point_x = left + bar_w * (20 / 30)
    d.line((ci_x0, axis_y - 48, ci_x1, axis_y - 48), fill="#b33a3a", width=9)
    d.line((ci_x0, axis_y - 65, ci_x0, axis_y - 31), fill="#b33a3a", width=6)
    d.line((ci_x1, axis_y - 65, ci_x1, axis_y - 31), fill="#b33a3a", width=6)
    d.ellipse((point_x - 13, axis_y - 61, point_x + 13, axis_y - 35), fill="#b33a3a")
    d.text(
        (left, 440),
        "Observed positive: 66.7%   Wilson 95% CI: 48.8-80.8%",
        font=regular,
        fill="black",
    )

    d.text((left, 545), "B  RANDOM - GROUPKFOLD UAR", font=bold, fill="black")
    plot_x0, plot_x1 = 480, right
    plot_y0, plot_y1 = 675, 1350
    max_x = 24.0
    for tick in (0, 5, 10, 15, 20):
        xx = plot_x0 + (plot_x1 - plot_x0) * tick / max_x
        d.line((xx, plot_y0, xx, plot_y1), fill="#d7d7d7", width=3)
        label = str(tick)
        box = d.textbbox((0, 0), label, font=tiny)
        d.text((xx - (box[2] - box[0]) / 2, plot_y1 + 15), label, font=tiny, fill="black")
    d.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="black", width=4)

    rows = []
    for corpus in CORPUS_ORDER:
        for model in MODEL_ORDER:
            rows.append((corpus, model, e["rg"][(corpus, model)]))
    row_gap = (plot_y1 - plot_y0) / len(rows)
    colors_by_corpus = {"ravdess": "#3569a8", "cremad": "#d77b28"}
    for i, (corpus, model, stat) in enumerate(rows):
        yy = plot_y0 + (i + 0.5) * row_gap
        label = f"{CORPUS_LABEL[corpus]}  {MODEL_LABEL[model]}"
        d.text((left, yy - 24), label, font=small, fill="black")
        lo = 100 * stat["ci_low"]
        hi = 100 * stat["ci_high"]
        est = 100 * stat["delta"]
        xlo = plot_x0 + (plot_x1 - plot_x0) * lo / max_x
        xhi = plot_x0 + (plot_x1 - plot_x0) * hi / max_x
        xe = plot_x0 + (plot_x1 - plot_x0) * est / max_x
        color = colors_by_corpus[corpus]
        d.line((xlo, yy, xhi, yy), fill=color, width=9)
        d.line((xlo, yy - 17, xlo, yy + 17), fill=color, width=6)
        d.line((xhi, yy - 17, xhi, yy + 17), fill=color, width=6)
        d.ellipse((xe - 14, yy - 14, xe + 14, yy + 14), fill=color, outline="black", width=2)
    d.text((plot_x0 + 250, 1450), "UAR difference (percentage points)", font=regular, fill="black")
    img.save(FIGURE_PNG, dpi=(300, 300), optimize=True)


def paper_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "paper_body",
            parent=base["BodyText"],
            fontName="TimesNewRoman",
            fontSize=9.0,
            leading=10.0,
            alignment=TA_JUSTIFY,
            spaceAfter=3.0,
            allowWidows=0,
            allowOrphans=0,
        ),
        "section": ParagraphStyle(
            "paper_section",
            parent=base["Heading2"],
            fontName="TimesNewRoman-Bold",
            fontSize=9.4,
            leading=10.4,
            alignment=TA_CENTER,
            spaceBefore=5.5,
            spaceAfter=3.0,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "paper_subsection",
            parent=base["Heading3"],
            fontName="TimesNewRoman-BoldItalic",
            fontSize=8.9,
            leading=9.8,
            alignment=TA_LEFT,
            spaceBefore=4.0,
            spaceAfter=2.0,
            keepWithNext=True,
        ),
        "center": ParagraphStyle(
            "paper_center",
            parent=base["BodyText"],
            fontName="TimesNewRoman-Bold",
            fontSize=8.9,
            leading=9.8,
            alignment=TA_CENTER,
            spaceAfter=3.0,
        ),
        "caption": ParagraphStyle(
            "paper_caption",
            parent=base["BodyText"],
            fontName="TimesNewRoman",
            fontSize=7.8,
            leading=8.6,
            alignment=TA_JUSTIFY,
            spaceBefore=1.5,
            spaceAfter=3.0,
        ),
        "reference": ParagraphStyle(
            "paper_reference",
            parent=base["BodyText"],
            fontName="TimesNewRoman",
            fontSize=8.0,
            leading=8.8,
            alignment=TA_JUSTIFY,
            leftIndent=11,
            firstLineIndent=-11,
            spaceAfter=2.0,
        ),
    }


def table_flowable(e: dict) -> Table:
    header = ["Corpus / model", "Rand.", "Group", "LOSO", "R-G [95% CI]"]
    rows = [header]
    for corpus in CORPUS_ORDER:
        for model in MODEL_ORDER:
            m = e["metrics"]
            rg = e["rg"][(corpus, model)]
            random = m[(corpus, model, "random")]
            grouped = m[(corpus, model, "groupkfold")]
            loso = m[(corpus, model, "loso")]
            rows.append(
                [
                    f"{CORPUS_LABEL[corpus]} / {MODEL_LABEL[model]}",
                    f"{p(random['mean'])}\n±{p(random['sd'])}",
                    f"{p(grouped['mean'])}\n±{p(grouped['sd'])}",
                    f"{p(loso['mean'])}\n±{p(loso['sd'])}",
                    f"{p(rg['delta'])} [{p(rg['ci_low'])}, {p(rg['ci_high'])}]",
                ]
            )
    table = Table(rows, colWidths=[68, 37, 37, 37, 76], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "TimesNewRoman"),
                ("FONTNAME", (0, 0), (-1, 0), "TimesNewRoman-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.2),
                ("LEADING", (0, 0), (-1, -1), 6.8),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.black),
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
            ]
        )
    )
    return table


def make_pdf(e: dict, spec: dict) -> None:
    register_fonts()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = letter
    margin_x = 0.58 * inch
    bottom = 0.48 * inch
    gap = 0.22 * inch
    col_w = (page_w - 2 * margin_x - gap) / 2
    first_top = 1.55 * inch
    later_top = 0.47 * inch

    first_frames = [
        Frame(
            margin_x,
            bottom,
            col_w,
            page_h - bottom - first_top,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="first_left",
        ),
        Frame(
            margin_x + col_w + gap,
            bottom,
            col_w,
            page_h - bottom - first_top,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="first_right",
        ),
    ]
    later_frames = [
        Frame(
            margin_x,
            bottom,
            col_w,
            page_h - bottom - later_top,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="later_left",
        ),
        Frame(
            margin_x + col_w + gap,
            bottom,
            col_w,
            page_h - bottom - later_top,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="later_right",
        ),
    ]

    def draw_title(canvas, doc):
        canvas.saveState()
        title_style = ParagraphStyle(
            "title",
            fontName="TimesNewRoman-Bold",
            fontSize=13.0,
            leading=14.2,
            alignment=TA_CENTER,
        )
        title = Paragraph(TITLE, title_style)
        _, th = title.wrap(page_w - 2 * margin_x, 70)
        title.drawOn(canvas, margin_x, page_h - 35 - th)
        canvas.setFont("TimesNewRoman", 9.2)
        canvas.drawCentredString(page_w / 2, page_h - 79, AUTHOR)
        canvas.setFont("TimesNewRoman-Italic", 8.2)
        canvas.drawCentredString(page_w / 2, page_h - 93, f"{AFFILIATION}  |  {EMAIL}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=later_top,
        bottomMargin=bottom,
        title=TITLE,
        author=AUTHOR,
        subject="ICASSP 2027 paper demo generated from frozen project-07 evidence",
        creator="Project 07 reproducible paper builder",
        pageTemplates=[
            PageTemplate("First", frames=first_frames, onPage=draw_title, autoNextPageTemplate="Later"),
            PageTemplate("Later", frames=later_frames),
        ],
    )

    styles = paper_styles()
    story = []
    for idx, column in enumerate(spec["columns"]):
        if idx and idx % 2 == 0:
            story.append(PageBreak())
        elif idx:
            story.append(FrameBreak())
        for kind, content in column:
            if kind in {"body", "section", "subsection", "center", "caption"}:
                story.append(Paragraph(content, styles[kind]))
            elif kind == "figure":
                fig = RLImage(str(FIGURE_PNG), width=col_w, height=col_w * 1540 / 1700)
                story.append(Spacer(1, 2))
                story.append(fig)
            elif kind == "table":
                story.append(table_flowable(e))
            else:
                raise RuntimeError(f"Unknown flow kind: {kind}")

    story.append(PageBreak())
    story.append(Paragraph("REFERENCES", styles["section"]))
    for i, ref in enumerate(spec["references"], 1):
        story.append(Paragraph(f"[{i}] {ref}", styles["reference"]))
    doc.build(story)


def markdown_table(e: dict) -> str:
    lines = [
        "| Corpus | Model | Random UAR | GroupKFold UAR | LOSO UAR | R-G delta [95% CI] | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in CORPUS_ORDER:
        for model in MODEL_ORDER:
            m = e["metrics"]
            rg = e["rg"][(corpus, model)]
            random = m[(corpus, model, "random")]
            grouped = m[(corpus, model, "groupkfold")]
            loso = m[(corpus, model, "loso")]
            lines.append(
                f"| {'RAVDESS speech' if corpus == 'ravdess' else CORPUS_LABEL[corpus]} | {MODEL_LABEL[model]} | "
                f"{p(random['mean'])} ± {p(random['sd'])} | "
                f"{p(grouped['mean'])} ± {p(grouped['sd'])} | "
                f"{p(loso['mean'])} ± {p(loso['sd'])} | "
                f"{p(rg['delta'])} [{p(rg['ci_low'])}, {p(rg['ci_high'])}] | "
                f"{rg['p_holm']:.3g} |"
            )
    return "\n".join(lines)


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def write_markdown(e: dict, spec: dict) -> None:
    lines = [
        f"# {TITLE}",
        "",
        f"{AUTHOR} - {AFFILIATION} - {EMAIL}",
        "",
        "## Abstract",
        "",
        spec["abstract"],
        "",
        "**Index Terms-** speech emotion recognition, evaluation protocol, speaker independence, reproducibility, empirical audit",
        "",
    ]
    for column in spec["columns"]:
        for kind, content in column:
            clean = strip_html(content)
            if kind == "center" and clean == "ABSTRACT":
                continue
            if kind == "section":
                lines.extend([f"## {clean}", ""])
            elif kind == "subsection":
                lines.extend([f"### {clean}", ""])
            elif kind == "body":
                if clean == spec["abstract"] or clean.startswith("Index Terms-"):
                    continue
                lines.extend([clean, ""])
            elif kind == "figure":
                lines.extend(["![Protocol effects](figure_protocol_effects.png)", ""])
            elif kind == "table":
                lines.extend([markdown_table(e), ""])
            elif kind == "caption":
                lines.extend([f"*{clean}*", ""])
    lines.extend(["## References", ""])
    for i, ref in enumerate(spec["references"], 1):
        lines.extend([f"[{i}] {strip_html(ref)}", ""])
    MARKDOWN_OUT.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "+/-": r"$\pm$",
        "±": r"$\pm$",
    }
    out = strip_html(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    # Keep one claim sentence per source line so the semantic number checker
    # can bind context before comparing values. TeX collapses this whitespace.
    out = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n", out)
    return out


def latex_table(e: dict) -> str:
    rows = []
    for corpus in CORPUS_ORDER:
        for model in MODEL_ORDER:
            m = e["metrics"]
            rg = e["rg"][(corpus, model)]
            random = m[(corpus, model, "random")]
            grouped = m[(corpus, model, "groupkfold")]
            loso = m[(corpus, model, "loso")]
            rows.append(
                f"{'RAVDESS speech' if corpus == 'ravdess' else CORPUS_LABEL[corpus]} & {MODEL_LABEL[model]} & "
                f"{p(random['mean'])} $\\pm$ {p(random['sd'])} & "
                f"{p(grouped['mean'])} $\\pm$ {p(grouped['sd'])} & "
                f"{p(loso['mean'])} $\\pm$ {p(loso['sd'])} & "
                f"{p(rg['delta'])} [{p(rg['ci_low'])}, {p(rg['ci_high'])}] \\\\"
            )
    return "\n".join(rows)


def write_latex(e: dict, spec: dict) -> None:
    section_text = []
    seen = set()
    for column in spec["columns"]:
        for kind, content in column:
            clean = strip_html(content)
            if kind == "center" and clean == "ABSTRACT":
                continue
            if kind == "body":
                if clean == spec["abstract"] or clean.startswith("Index Terms-"):
                    continue
                section_text.append(latex_escape(clean) + "\n")
            elif kind == "section":
                if clean in seen:
                    continue
                seen.add(clean)
                title = re.sub(r"^\d+\.\s*", "", clean).title()
                section_text.append(f"\\section{{{latex_escape(title)}}}\n")
            elif kind == "subsection":
                title = re.sub(r"^\d+\.\d+\.\s*", "", clean)
                section_text.append(f"\\subsection{{{latex_escape(title)}}}\n")
            elif kind == "figure":
                section_text.append(
                    "\\begin{figure}[t]\n"
                    "\\centering\n"
                    "\\includegraphics[width=\\columnwidth]{figure_protocol_effects.png}\n"
                    "\\caption{Frozen audit outcomes and speaker-paired Random-minus-GroupKFold UAR differences. Panel A summarizes the preregistered probability sample from the frozen public-code candidate frame (the estimand is this frame, not all SER work); panel B shows the eight corpus-model contrasts, which belong to the original frozen 24-test Holm family.}\n"
                    "\\label{fig:protocol}\n"
                    "\\end{figure}\n"
                )
            elif kind == "table":
                section_text.append(
                    "\\begin{table*}[t]\n"
                    "\\centering\\scriptsize\n"
                    "\\caption{UAR (\\%) mean $\\pm$ sample SD over three seeds. The final column is the speaker-paired Random-Grouped difference with 95\\% cluster-bootstrap CI.}\n"
                    "\\begin{tabular}{llrrrr}\n\\toprule\n"
                    "Corpus & Model & Random UAR & Grouped UAR & LOSO UAR & Delta UAR [95\\% CI] \\\\ \n"
                    "\\midrule\n"
                    + latex_table(e)
                    + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
                )
            elif kind == "caption":
                continue

    bibitems = []
    keys = [
        "schuller2009challenge",
        "pepino2020fusion",
        "atmaja2021splitting",
        "zielonka2022recognition",
        "ibrahim2026multimodal",
        "antoniou2023designing",
        "scheidwasserclow2022serab",
        "wu2024openemotion",
        "meyer2018classifiers",
        "livingstone2018ravdess",
        "cao2014cremad",
    ]
    for key, ref in zip(keys, spec["references"], strict=True):
        bibitems.append(f"\\bibitem{{{key}}} {latex_escape(ref)}")

    tex = r"""\documentclass{article}
\usepackage{spconf,amsmath,graphicx,booktabs}
\title{SPEAKER-NONEXCLUSIVE EVALUATION IN PUBLIC SER CODE:\\
A PREREGISTERED AUDIT AND CONTROLLED STUDY OF PROTOCOL EFFECTS}
\name{[AUTHOR NAME]}
\address{[AFFILIATION] \\ [EMAIL]}
\begin{document}
\maketitle
\begin{abstract}
""" + latex_escape(spec["abstract"]) + r"""
\end{abstract}
\begin{keywords}
speech emotion recognition, evaluation protocol, speaker independence, reproducibility, empirical audit
\end{keywords}
""" + "\n".join(section_text) + r"""
\begin{thebibliography}{11}
""" + "\n".join(bibitems) + r"""
\end{thebibliography}
\end{document}
"""
    LATEX_OUT.write_text(tex, encoding="utf-8")
    # The shared number auditor is designed for scientific claims, not years,
    # volumes, pages, or DOI numerals in bibliography metadata.  Keep a
    # mechanically derived technical-body view so the audit can fail closed on
    # every experimental number without generating bibliography false alarms.
    numbercheck_tex = re.sub(
        r"\\begin\{thebibliography\}\{[^}]*\}.*?\\end\{thebibliography\}",
        "",
        tex,
        flags=re.DOTALL,
    )
    NUMBERCHECK_OUT.write_text(numbercheck_tex, encoding="utf-8")


def fact(
    fact_id: str,
    value: float,
    statistic: str,
    metric: str,
    source_id: str,
    pointer: str,
    *,
    dimensions: dict[str, str] | None = None,
    aliases: list[str] | None = None,
    unit: str = "count",
    tolerance: float = 0.0005,
) -> dict:
    return {
        "id": fact_id,
        "value": value,
        "statistic": statistic,
        "metric": metric,
        "dimensions": dimensions or {"experiment": "protocol audit"},
        "aliases": aliases or [],
        "unit": unit,
        "source_id": source_id,
        "pointer": pointer,
        "track_unused": False,
        "tolerance": tolerance,
    }


def write_fact_catalog(e: dict) -> None:
    sources = []
    for source_id, path, authority in [
        ("tier_summary", TIER_CSV, "final-audit-adjudication"),
        ("run_status", RUN_STATUS_JSON, "final-run-status"),
        ("verification", VERIFICATION_JSON, "independent-verification"),
        ("premium", PREMIUM_CSV, "final-paired-statistics"),
        ("metric_summary", METRIC_CSV, "final-oof-metrics"),
        ("prereg", PREREG, "frozen-preregistration"),
        ("d1_report", D1_REPORT, "frozen-one-shot-scoring"),
        ("d1_completion", D1_COMPLETION, "frozen-completion-manifest"),
    ]:
        sources.append(
            {
                "id": source_id,
                "path": str(path),
                "sha256": sha256(path),
                "priority": 100,
                "authority": authority,
                "required": True,
            }
        )

    audit_dims = {"experiment": "public code protocol audit"}
    facts = [
        fact("audit.n", 30, "count", "repository sample", "tier_summary", "split_risk_final denominator", dimensions={**audit_dims, "outcome": "sample"}, aliases=["sample included", "sample contained", "sample size", "audit denominator", "denominator remained"]),
        fact("audit.yes", 20, "count", "repository audit", "tier_summary", "split_risk_final=yes", dimensions={**audit_dims, "outcome": "speaker nonexclusive"}, aliases=["showed speaker-nonexclusive split risk", "found speaker-nonexclusive split risk"]),
        fact("audit.no", 5, "count", "repository audit", "tier_summary", "split_risk_final=no", dimensions={**audit_dims, "outcome": "verified negative"}, aliases=["verified negative", "did not"]),
        fact("audit.unknown", 5, "count", "repository audit", "tier_summary", "split_risk_final=unknown", dimensions={**audit_dims, "outcome": "unresolved"}, aliases=["were unresolved", "unresolved cases"]),
        fact("audit.rate_pct", 66.7, "mean", "repository audit", "tier_summary", "20/30 displayed", dimensions={**audit_dims, "outcome": "speaker nonexclusive"}, aliases=["observed rate", "observed positive fraction"], unit="percent", tolerance=0.05),
        fact("audit.ci_low_pct", 48.8, "ci_low", "repository audit", "tier_summary", "Wilson interval from 20/30", dimensions={**audit_dims, "outcome": "speaker nonexclusive"}, aliases=["observed rate", "observed positive fraction"], unit="percent", tolerance=0.05),
        fact("audit.ci_high_pct", 80.8, "ci_high", "repository audit", "tier_summary", "Wilson interval from 20/30", dimensions={**audit_dims, "outcome": "speaker nonexclusive"}, aliases=["observed rate", "observed positive fraction"], unit="percent", tolerance=0.05),
        fact("audit.sensitivity_count", 25, "count", "repository audit sensitivity", "tier_summary", "20+5 unknown", dimensions={**audit_dims, "outcome": "unknown as risk"}, aliases=["positive count"], unit="count"),
        fact("audit.sensitivity_pct", 83.3, "mean", "repository audit sensitivity", "tier_summary", "25/30 displayed", dimensions={**audit_dims, "outcome": "unknown as risk"}, aliases=["sensitivity rate"], unit="percent", tolerance=0.05),
        fact("audit.sensitivity_ci_low_pct", 66.4, "ci_low", "repository audit sensitivity", "tier_summary", "Wilson interval from 25/30", dimensions={**audit_dims, "outcome": "unknown as risk"}, aliases=["sensitivity rate"], unit="percent", tolerance=0.05),
        fact("audit.sensitivity_ci_high_pct", 92.7, "ci_high", "repository audit sensitivity", "tier_summary", "Wilson interval from 25/30", dimensions={**audit_dims, "outcome": "unknown as risk"}, aliases=["sensitivity rate"], unit="percent", tolerance=0.05),
        fact("audit.tier1", 24, "count", "execution evidence", "tier_summary", "evidence_tier=tier1_execution_confirmed", dimensions={**audit_dims, "evidence_tier": "highest tier"}, aliases=["highest-tier executable evidence"]),
        fact("audit.property_checks", 384, "count", "property checks", "tier_summary", "24*16 derived", dimensions={**audit_dims, "evidence_tier": "highest tier"}, aliases=["finite property cases", "fragments passed"]),
        fact("audit.property_checks_per_repository", 16, "count", "property checks", "tier_summary", "384/24 derived", dimensions={**audit_dims, "evidence_tier": "highest tier"}, aliases=["repository contributed", "cases per repository"]),
        fact("run.fits", 1620, "count", "fit count", "run_status", "/success", dimensions={"experiment": "protocol premium", "status": "successful"}, aliases=["valid fits", "successful fits", "ran 1,620 fits", "1,620-fit study"]),
        fact("run.oof_cells", 72, "count", "oof cell count", "verification", "/counts/complete_cells", dimensions={"experiment": "protocol premium", "status": "complete"}, aliases=["complete cells", "OOF cells"]),
        fact("stats.holm_significant", 19, "count", "Holm rejections", "premium", "count holm_reject_0_05=true for UAR", dimensions={"experiment": "protocol premium", "family": "all protocol contrasts"}, aliases=["Holm rejections numbered"]),
        fact("stats.holm_family", 24, "count", "multiplicity family size", "premium", "fixed family size", dimensions={"experiment": "protocol premium", "family": "all protocol contrasts"}, aliases=["multiplicity family size"]),
        fact("stats.bootstrap", 10000, "count", "bootstrap replicates", "premium", "bootstrap_replicates", dimensions={"experiment": "protocol premium"}, aliases=["whole-speaker bootstrap replicates", "speaker-cluster intervals"]),
        fact("dataset.ravdess.utterances", 1440, "count", "utterance count", "prereg", "RAVDESS utterances", dimensions={"experiment": "protocol premium", "corpus": "ravdess speech"}, aliases=["RAVDESS contributed"]),
        fact("dataset.ravdess.speakers", 24, "count", "speaker count", "prereg", "RAVDESS speakers", dimensions={"experiment": "protocol premium", "corpus": "ravdess speech"}, aliases=["RAVDESS contains"]),
        fact("dataset.cremad.utterances", 7442, "count", "utterance count", "prereg", "CREMA-D utterances", dimensions={"experiment": "protocol premium", "corpus": "cremad"}, aliases=["CREMA-D contributed"]),
        fact("dataset.cremad.speakers", 91, "count", "speaker count", "prereg", "CREMA-D speakers", dimensions={"experiment": "protocol premium", "corpus": "cremad"}, aliases=["CREMA-D contains"]),
        # D1 matched speech-song panel: every value below derives from the
        # sha-certified frozen report parsed in load_evidence() (record 77
        # action 2); wording per dialogue record 66
        fact("d1.actors", e["d1"]["actors"], "count", "matched panel actors", "d1_report", "len(actors)", dimensions={"experiment": "d1 matched panel"}, aliases=["matched 23-actor"]),
        fact("d1.items_per_channel", e["d1"]["items_per_channel"], "count", "items per channel", "d1_report", "n_samples_per_channel", dimensions={"experiment": "d1 matched panel"}, aliases=["items per channel"]),
        fact("d1.fits", e["d1"]["fits"], "count", "fit count", "d1_completion", "len(units)", dimensions={"experiment": "d1 matched panel", "status": "successful"}, aliases=["792 fits"]),
        fact("d1.premium_speech_low", e["d1"]["prg_speech_low"], "mean", "RG premium speech low", "d1_report", "min over models of point_estimates.P_RG_speech*100", dimensions={"experiment": "d1 matched panel", "channel": "speech"}, aliases=["speech"], unit="uar points", tolerance=0.005),
        fact("d1.premium_speech_high", e["d1"]["prg_speech_high"], "mean", "RG premium speech high", "d1_report", "max over models of point_estimates.P_RG_speech*100", dimensions={"experiment": "d1 matched panel", "channel": "speech"}, aliases=["speech"], unit="uar points", tolerance=0.005),
        fact("d1.premium_song_low", e["d1"]["prg_song_low"], "mean", "RG premium song low", "d1_report", "min over models of point_estimates.P_RG_song*100", dimensions={"experiment": "d1 matched panel", "channel": "song"}, aliases=["song"], unit="uar points", tolerance=0.005),
        fact("d1.premium_song_high", e["d1"]["prg_song_high"], "mean", "RG premium song high", "d1_report", "max over models of point_estimates.P_RG_song*100", dimensions={"experiment": "d1 matched panel", "channel": "song"}, aliases=["song"], unit="uar points", tolerance=0.005),
    ]
    registered_sources = {"tier_summary", "run_status", "verification", "premium",
                           "metric_summary", "prereg", "d1_report", "d1_completion"}
    dangling = sorted({f["source_id"] for f in facts} - registered_sources)
    if dangling:
        raise RuntimeError(f"fact catalog has dangling source ids: {dangling}")

    for corpus in CORPUS_ORDER:
        for model in MODEL_ORDER:
            key = (corpus, model)
            stat = e["rg"][key]
            prefix = f"{corpus}.{model}"
            corpus_dim = "ravdess speech" if corpus == "ravdess" else "cremad"
            model_dim = MODEL_LABEL[model]
            common_dims = {
                "experiment": "protocol premium",
                "corpus": corpus_dim,
                "model": model_dim,
            }
            facts.extend(
                [
                    fact(f"{prefix}.delta_pct", 100 * stat["delta"], "difference", "delta balanced accuracy", "premium", f"{corpus}/{model}/RG/uar/point_mean_difference", dimensions={**common_dims, "comparison": "random minus groupkfold"}, aliases=["Delta UAR", "Random-Grouped difference", f"{CORPUS_LABEL[corpus]} {MODEL_LABEL[model]}"], unit="percentage_points", tolerance=0.005),
                    fact(f"{prefix}.ci_low_pct", 100 * stat["ci_low"], "ci_low", "delta balanced accuracy", "premium", f"{corpus}/{model}/RG/uar/ci95_low", dimensions={**common_dims, "comparison": "random minus groupkfold"}, aliases=["Delta UAR", "95% CI"], unit="percentage_points", tolerance=0.005),
                    fact(f"{prefix}.ci_high_pct", 100 * stat["ci_high"], "ci_high", "delta balanced accuracy", "premium", f"{corpus}/{model}/RG/uar/ci95_high", dimensions={**common_dims, "comparison": "random minus groupkfold"}, aliases=["Delta UAR", "95% CI"], unit="percentage_points", tolerance=0.005),
                ]
            )
            for protocol, label in [("random", "Random UAR"), ("groupkfold", "Grouped UAR"), ("loso", "LOSO UAR")]:
                metric = e["metrics"][(corpus, model, protocol)]
                metric_dims = {**common_dims, "condition": protocol}
                facts.extend(
                    [
                        fact(f"{prefix}.{protocol}.mean_pct", 100 * metric["mean"], "mean", "balanced_accuracy", "metric_summary", f"{corpus}/{model}/{protocol}/uar/mean", dimensions=metric_dims, aliases=[label], unit="percent", tolerance=0.005),
                        fact(f"{prefix}.{protocol}.sd_pct", 100 * metric["sd"], "sd", "balanced_accuracy", "metric_summary", f"{corpus}/{model}/{protocol}/uar/sample_sd", dimensions=metric_dims, aliases=[label, "sample SD"], unit="percent", tolerance=0.005),
                    ]
                )

    catalog = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Project-07 paper facts derived from frozen audit and P1 tables.",
        "sources": sources,
        "facts": facts,
    }
    FACTS_OUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")


def pdf_font_audit(reader: PdfReader) -> dict:
    seen = {}
    for page in reader.pages:
        resources = page.get("/Resources", {})
        font_map = resources.get("/Font", {})
        if hasattr(font_map, "get_object"):
            font_map = font_map.get_object()
        for name, ref in font_map.items():
            font = ref.get_object()
            base = str(font.get("/BaseFont", name))
            descriptor = font.get("/FontDescriptor")
            embedded = False
            if descriptor is not None:
                descriptor = descriptor.get_object()
                embedded = any(k in descriptor for k in ("/FontFile", "/FontFile2", "/FontFile3"))
            seen[base] = embedded
    return seen


def validate_pdf(spec: dict) -> dict:
    reader = PdfReader(str(OUTPUT_PDF))
    if len(reader.pages) != 5:
        raise RuntimeError(f"Expected 5 pages (4 technical + references), got {len(reader.pages)}")
    for i, page in enumerate(reader.pages, 1):
        box = page.mediabox
        if abs(float(box.width) - 612) > 0.5 or abs(float(box.height) - 792) > 0.5:
            raise RuntimeError(f"Page {i} is not US Letter: {box}")
    texts = [(page.extract_text() or "") for page in reader.pages]
    normalized = [re.sub(r"\s+", " ", text) for text in texts]
    if "REFERENCES" in "\n".join(normalized[:4]):
        raise RuntimeError("References leaked into the four technical pages")
    if "REFERENCES" not in normalized[4]:
        raise RuntimeError("Reference-only fifth page is missing")
    if "CONCLUSION" not in normalized[3]:
        raise RuntimeError("Technical content did not fill through page 4")
    if "Multimodal emotion recognition" not in normalized[4]:
        raise RuntimeError("Critical 2026 collision is missing from references")
    fonts = pdf_font_audit(reader)
    if not fonts or any(not embedded for embedded in fonts.values()):
        raise RuntimeError(f"All PDF fonts must be embedded: {fonts}")
    for token in ("20 sampled public SER repositories", "sample contained 30 repositories", "1,620-fit", "15.39 percentage points (FNO)", "3.06 (Transformer)"):
        if token not in "\n".join(normalized):
            raise RuntimeError(f"Expected paper token missing from PDF: {token}")
    return {
        "pages": len(reader.pages),
        "page_size_points": [612, 792],
        "fonts": fonts,
        "abstract_words": len(re.findall(r"\b[\w-]+\b", spec["abstract"])),
        "references_only_page_5": True,
    }


def main() -> None:
    evidence = load_evidence()
    spec = manuscript(evidence)
    draw_figure(evidence)
    write_markdown(evidence, spec)
    write_latex(evidence, spec)
    write_fact_catalog(evidence)
    make_pdf(evidence, spec)
    pdf_audit = validate_pdf(spec)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "output_pdf": str(OUTPUT_PDF),
        "output_sha256": sha256(OUTPUT_PDF),
        "inputs": {
            str(path): sha256(path)
            for path in (TIER_CSV, RUN_STATUS_JSON, VERIFICATION_JSON, PREMIUM_CSV, METRIC_CSV, PREREG, D1_REPORT, D1_COMPLETION)
        },
        "generated": {
            str(path): sha256(path)
            for path in (FIGURE_PNG, MARKDOWN_OUT, LATEX_OUT, NUMBERCHECK_OUT, FACTS_OUT)
        },
        "pdf_audit": pdf_audit,
        "scientific_scope": "P0 final 20/5/5 plus P1 controlled protocol effects "
                             "plus D1 compact matched-panel diagnostic (record-66 "
                             "wording); P2 and P5 excluded",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""D1 completion manifest v6 (prescore amendment) + read-only verifier.

Closes the two gaps Codex dialogue record 60 identified in v5:
  1. Receipt identity closure: verify_postscore_receipt() now enforces the
     full required-field set, FIXED output/claim file names, live content
     hashes, and three-way identity cross-checking -- the receipt, the
     parsed claim, and the parsed output must agree on the scorer/unlock/v6
     hashes AND each must equal the live hash of the actual scorer, unlock,
     and v6 manifest. Mutations covered: each identity field wrong, all
     three wrong, missing field, substituted file name, 1-byte claim/output
     change, live unlock swap, lying claim.
  2. Verifier honesty: this builder's verify mode rebuilds the ENTIRE
     manifest from disk facts and deep-compares it against the frozen file
     (only created_at/created_at_note are exempt), so every keyed field --
     including spec text blocks -- is explicitly covered, matching the
     printed claim.

All v5 closure checks are re-run from disk. v1..v5/corrigendum/attempts
stay byte-identical; v5 is verified and recorded as superseded. The runbook
wording corrections required by record 60 action 3 live in
D1_SCORING_RUNBOOK.md (revision log there).

Self-contained (stdlib only), fail-closed, outcome-blind.

Usage:
    python tools/build_d1_completion_manifest_v6.py            # build once
    python tools/build_d1_completion_manifest_v6.py verify     # re-check frozen v6
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

RESULT_ROOT = Path(r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched")
TRAIN_ROOT = RESULT_ROOT / "training"
PLAN_PATH = RESULT_ROOT / "run_plan.csv"
PREPARED = TRAIN_ROOT / "d1_prepared.json"
DEVIATION = TRAIN_ROOT / "d1_deviation_corrigendum_20260816.json"
V1_PATH = TRAIN_ROOT / "completion_manifest.json"
V3_PATH = TRAIN_ROOT / "completion_manifest_v3.json"
V4_PATH = TRAIN_ROOT / "completion_manifest_v4.json"
V5_PATH = TRAIN_ROOT / "completion_manifest_v5.json"
OUT_PATH = TRAIN_ROOT / "completion_manifest_v6.json"
ANOMALY_REGISTRY = RESULT_ROOT / "anomaly_registry.csv"
PAIR_MANIFEST = RESULT_ROOT / "pair_manifest.csv"
NPZ_DIR = Path(r"E:\科研\claudework\07-ser-repro-protocol-audit\work\d1_splits")
TOOLS_DIR = Path(__file__).resolve().parent
GPU_SECONDS_CAP = 14400.0
STALE_UNIT = "d1__transformer__groupkfold__s1__f004__speech"
SCORER_FILES = ("score_d1_results.py", "test_d1_scorer.py", "test_d1_scorer_assembly.py")
REVIEW_REQUEST_RECORD = 61  # the dialogue record submitting THIS v6 for review
PRED_HEADER = ("unit_id", "channel", "model", "protocol", "seed", "outer_fold",
                "sample_index", "canonical_actor", "relative_path",
                "true_label_index", "true_label_name",
                "predicted_label_index", "predicted_label_name",
                "logit_neutral", "logit_calm", "logit_happy", "logit_sad",
                "logit_angry", "logit_fearful")
EMOTION6 = ("neutral", "calm", "happy", "sad", "angry", "fearful")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def unit_dir_for(unit_id: str) -> Path:
    parts = unit_id.split("__")  # d1, model, protocol, sN, fNNN, channel
    return (TRAIN_ROOT / "units" / parts[5] / parts[1] / parts[2]
             / f"seed_{int(parts[3][1:])}" / f"fold_{int(parts[4][1:]):03d}")


def gather() -> dict:
    """Re-derive every checkable fact from disk (read-only)."""
    prepared = json.loads(PREPARED.read_text(encoding="utf-8"))
    dev = json.loads(DEVIATION.read_text(encoding="utf-8"))
    v4 = json.loads(V4_PATH.read_text(encoding="utf-8"))
    v5 = json.loads(V5_PATH.read_text(encoding="utf-8"))
    if v5["units"] != v4["units"]:
        fail("v5 unit table diverged from v4")

    anchor_files = {
        "run.lock.stale_audited_20260816": TRAIN_ROOT / "run.lock.stale_audited_20260816",
        "stale_lock_audit_20260816.json": TRAIN_ROOT / "stale_lock_audit_20260816.json",
        "attempt_01_corrected": TRAIN_ROOT / "attempt_ledger" / f"{STALE_UNIT}__attempt_01.json",
        "attempt_02_success": TRAIN_ROOT / "attempt_ledger" / f"{STALE_UNIT}__attempt_02.json",
        "completion_manifest_v1_superseded": V1_PATH,
    }
    for key, path in anchor_files.items():
        observed = sha256_file(path)
        if observed != dev["file_anchors_sha256"][key]:
            fail(f"corrigendum anchor {key}: {observed} != {dev['file_anchors_sha256'][key]}")

    plan_sha = sha256_file(PLAN_PATH)
    if plan_sha != prepared["run_plan_sha256"]:
        fail("run_plan.csv drifted from d1_prepared.json")
    for name, frozen in prepared["tool_sha256"].items():
        if sha256_file(TOOLS_DIR / name) != frozen:
            fail(f"tool {name} drifted from frozen anchor")
    with open(PLAN_PATH, newline="", encoding="utf-8") as f:
        plan_rows = list(csv.DictReader(f))
    if len(plan_rows) != 792:
        fail(f"run plan rows {len(plan_rows)} != 792")

    npz_shas = {}
    for r in plan_rows:
        rel = r["inner_split_path"]
        if rel not in npz_shas:
            observed = sha256_file(NPZ_DIR / rel)
            if observed != r["inner_split_sha256"]:
                fail(f"split NPZ {rel} drifted from run plan")
            npz_shas[rel] = observed
        elif npz_shas[rel] != r["inner_split_sha256"]:
            fail(f"run plan inconsistent sha for {rel}")

    attempt_rows = []
    total = 0.0
    stale_count = 0
    per_unit = Counter()
    for path in sorted((TRAIN_ROOT / "attempt_ledger").glob("*.json"),
                        key=lambda p: p.name):
        att = json.loads(path.read_text(encoding="utf-8"))
        total += float(att["elapsed_seconds"])
        per_unit[att["unit_id"]] += 1
        if att["status"] == "stale_crashed":
            stale_count += 1
            if att["unit_id"] != STALE_UNIT or "manual_correction" not in att:
                fail(f"unexpected stale attempt {path.name}")
        elif att["status"] != "success":
            fail(f"{path.name}: status {att['status']}")
        attempt_rows.append((path.name, sha256_file(path)))
    digest_input = b"".join(name.encode("utf-8") + b"\x00" + sha.encode("ascii") + b"\n"
                              for name, sha in attempt_rows)
    tree_digest = hashlib.sha256(digest_input).hexdigest()
    if stale_count != 1:
        fail(f"stale_crashed count {stale_count} != 1")
    if any(v > 2 for v in per_unit.values()):
        fail("a unit has >2 attempts")
    if total > GPU_SECONDS_CAP:
        fail(f"charged {total} exceeds cap")
    if abs(total - float(dev["settlement"]["final_ledger_total_seconds"])) > 0.001:
        fail("ledger total != corrigendum final figure")
    if tree_digest != v5["attempt_ledger_freeze"]["tree_digest"]:
        fail("attempt tree digest changed since v5")

    units = v5["units"]
    if set(units) != {r["unit_id"] for r in plan_rows}:
        fail("unit set != run plan unit set")
    pred_rows_total = 0
    for r in plan_rows:
        unit_id = r["unit_id"]
        path = unit_dir_for(unit_id) / "predictions.csv"
        observed = sha256_file(path)
        if observed != units[unit_id]["predictions_sha256"]:
            fail(f"{unit_id}: predictions.csv drifted from frozen anchor")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = tuple(next(reader))
            n_rows = sum(1 for _ in reader)
        if header != PRED_HEADER:
            fail(f"{unit_id}: predictions header drift {header}")
        if n_rows != int(r["n_outer_test"]):
            fail(f"{unit_id}: physical rows {n_rows} != plan n_outer_test {r['n_outer_test']}")
        pred_rows_total += n_rows

    scorer_shas = {name: sha256_file(TOOLS_DIR / name) for name in SCORER_FILES}

    return {
        "prepared": prepared, "dev": dev, "v5": v5,
        "corrigendum_anchor_observed": {k: sha256_file(p) for k, p in anchor_files.items()},
        "plan_sha": plan_sha, "npz_shas": npz_shas,
        "attempt_rows": attempt_rows, "tree_digest": tree_digest,
        "charged_total": round(total, 3),
        "pred_rows_total": pred_rows_total,
        "scorer_shas": scorer_shas,
        "static_shas": {
            "completion_manifest_v3": sha256_file(V3_PATH),
            "completion_manifest_v4": sha256_file(V4_PATH),
            "completion_manifest_v5": sha256_file(V5_PATH),
            "deviation_corrigendum": sha256_file(DEVIATION),
            "d1_prepared": sha256_file(PREPARED),
            "smoke_test": sha256_file(TRAIN_ROOT / "smoke_test.json"),
            "progress_ledger": sha256_file(TRAIN_ROOT / "progress_ledger.csv"),
            "anomaly_registry": sha256_file(ANOMALY_REGISTRY),
            "pair_manifest": sha256_file(PAIR_MANIFEST),
        },
    }


UNLOCK_REQUIREMENTS = {
    "file": str(TRAIN_ROOT / "SCORING_UNLOCKED.json"),
    "schema": {
        "completion_manifest_v6_sha256": "MUST equal the live hash of completion_manifest_v6.json",
        "scorer_sha256": "MUST equal the live hash of score_d1_results.py (also frozen in v6)",
        "output_path": "MUST equal scorer.output_path exactly",
        "prescore_pass_record": f"int, STRICTLY greater than review_request_record "
                                  f"({REVIEW_REQUEST_RECORD}); the Codex PASS record number",
        "author_authorization": "non-empty string quoting the author's approval",
        "executor": "non-empty string naming who runs the scorer",
    },
    "enforcement": "scorer parses and exactly compares every field; empty file, wrong "
                    "hash, wrong record, or missing field all refuse (validate_unlock, "
                    "8 mutation tests)",
    "creation_rule": "created only after Codex's pre-score review PASS, with author "
                      "authorization; Codex records the verdict but never creates it",
}


def make_manifest(facts: dict) -> dict:
    """Everything except created_at/created_at_note, rebuilt purely from
    disk facts + module constants -- shared by build() and verify()."""
    return {
        "builder": {"file": Path(__file__).name,
                     "sha256": sha256_file(Path(__file__).resolve())},
        "role": "prescore amendment per dialogue record 60; supersedes v5 as the "
                 "scoring gate document while keeping v1..v5/corrigendum unchanged",
        "review_request_record": REVIEW_REQUEST_RECORD,
        "supersedes": {"file": V5_PATH.name,
                        "sha256": facts["static_shas"]["completion_manifest_v5"]},
        "deviation_corrigendum": {"file": DEVIATION.name,
                                    "sha256": facts["static_shas"]["deviation_corrigendum"],
                                    "all_file_anchors_verified": facts["corrigendum_anchor_observed"]},
        "unlock_requirements": UNLOCK_REQUIREMENTS,
        "anchors": {
            "run_plan_sha256": facts["plan_sha"],
            "d1_prepared_sha256": facts["static_shas"]["d1_prepared"],
            "smoke_test_sha256": facts["static_shas"]["smoke_test"],
            "progress_ledger_sha256": facts["static_shas"]["progress_ledger"],
            "anomaly_registry_sha256": facts["static_shas"]["anomaly_registry"],
            "completion_manifest_v3_sha256": facts["static_shas"]["completion_manifest_v3"],
            "completion_manifest_v4_sha256": facts["static_shas"]["completion_manifest_v4"],
            "tool_sha256": facts["prepared"]["tool_sha256"],
            "split_npz_sha256": facts["npz_shas"],
        },
        "attempt_ledger_freeze": {
            "digest_spec": "sha256 over filename-sorted entries of "
                            "(filename + NUL + sha256hex + LF)",
            "tree_digest": facts["tree_digest"],
            "n_files": len(facts["attempt_rows"]),
            "charged_seconds_total": facts["charged_total"],
            "cap_seconds": GPU_SECONDS_CAP,
            "files": {name: sha for name, sha in facts["attempt_rows"]},
        },
        "predictions_structural_gate": {
            "method": "physical data-row count per file == run plan n_outer_test; "
                       "header tuple equality; byte hash == frozen anchor; no value cell read",
            "units_checked": 792,
            "total_prediction_rows": facts["pred_rows_total"],
        },
        "scorer": {
            **facts["scorer_shas"],
            "pair_manifest_sha256": facts["static_shas"]["pair_manifest"],
            "expected_input_root": str(TRAIN_ROOT / "units"),
            "npz_dir": str(NPZ_DIR),
            "output_path": str(RESULT_ROOT / "scoring" / "d1_scoring_report.json"),
            "scoring_claim_path": str(RESULT_ROOT / "scoring" / "scoring_claim.json"),
            "postscore_receipt_path": str(RESULT_ROOT / "scoring" / "postscore_receipt.json"),
            "config": {
                "models": ["cnn", "resnet_se", "transformer", "fno"],
                "channels": ["speech", "song"],
                "protocols": ["random", "groupkfold", "loso"],
                "seeds": [0, 1, 2],
                "n_classes": 6,
                "class_order": list(EMOTION6),
                "reweighting_draws": 10000,
                "reweighting_seed": 202608160300,
                "uncertainty_terminology": "actor_reweighting_95_stability_quantile_range; never CI",
                "p_values": None,
                "output_schema": "per model: status, point_estimates{P_RG_speech, P_RG_song, "
                                  "D_mode_RG, P_RL_speech, P_RL_song, D_mode_RL}, "
                                  "seedwise_panel_estimates (3 paired-seed panel values per "
                                  "quantity), seed_mean_sd (mean, sample SD ddof=1, n_seeds=3; "
                                  "training randomness only, panel n stays 23 actors), "
                                  "per_actor (incl. per-seed premiums), leave_one_actor_out, "
                                  "actor_reweighting quantiles",
            },
            "identity_closure": "per row: true label + actor + channel path vs pair manifest; "
                                 "per unit: sample set == frozen split NPZ outer_test_idx "
                                 "(NPZ re-hashed against run plan first)",
            "one_shot": "O_CREAT|O_EXCL|O_BINARY scoring claim before first outcome read; "
                         "write_all (looped os.write, 0-byte write raises) + length/sha "
                         "verification against the serialized buffer before atomic replace; "
                         "postscore receipt keys output_sha256, claim_sha256 and "
                         "completion_manifest_v6_sha256; verify_postscore_receipt enforces "
                         "required fields, fixed file names, and three-way identity "
                         "cross-checks (receipt == parsed claim == parsed output == live "
                         "v6/unlock/scorer hashes); failures after the output commit leave "
                         "claim/output/temp as audit evidence and refuse silent rescoring",
        },
        "units": facts["v5"]["units"],
        "outcome_blindness": "this builder hashed and line-counted predictions files; "
                              "no outcome value was read or interpreted",
    }


def build() -> int:
    if OUT_PATH.exists():
        fail(f"{OUT_PATH} exists; refusing to overwrite a frozen manifest")
    facts = gather()
    tz = timezone(timedelta(hours=-4))
    manifest = {
        "created_at": datetime.now(tz).isoformat(timespec="seconds"),
        "created_at_note": "generated from the process clock immediately before write; "
                            "the filesystem timestamp is authoritative",
        **make_manifest(facts),
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PASS: tree digest {facts['tree_digest'][:16]}... unchanged over "
           f"{len(facts['attempt_rows'])} files; {len(facts['npz_shas'])} split NPZs "
           f"anchored; identity-closing scorer + unlock schema frozen; "
           f"{facts['pred_rows_total']} prediction rows verified")
    print(f"wrote {OUT_PATH} (sha256 {sha256_file(OUT_PATH)})")
    return 0


def verify() -> int:
    if not OUT_PATH.exists():
        fail("no frozen v6 to verify")
    frozen = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    rebuilt = make_manifest(gather())
    frozen_cmp = {k: v for k, v in frozen.items()
                   if k not in ("created_at", "created_at_note")}
    if frozen_cmp == rebuilt:
        print("verify PASS: frozen v6 deep-equals the manifest rebuilt from live disk "
              "facts and builder constants (every field except created_at)")
        return 0
    keys = sorted(set(frozen_cmp) | set(rebuilt))
    for key in keys:
        if frozen_cmp.get(key) != rebuilt.get(key):
            print(f"MISMATCH in top-level field {key!r}")
    print("verify FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(verify() if (len(sys.argv) > 1 and sys.argv[1] == "verify") else build())

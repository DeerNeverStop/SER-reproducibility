"""Generate a CPU-only, non-executable AutoDL supplement design inventory."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

PROGRAM = "SER27-AUTODL-SUPPLEMENT-DESIGN-1"
REFERENCE_BYTE_SHA = "6bee99d32fe5932a6dfe6d22708ac2db6ec445b03afc339be50c649e3cf37c14"
CORPORA = ("cremad", "subesco", "ravdess")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def seed(corpus, draw, fold, stream):
    return int(digest([PROGRAM, "formal", corpus, draw, fold, stream])[:16], 16) % (2**31 - 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.reference_plan.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_BYTE_SHA, "reference plan byte identity differs"
    ref = json.loads(raw)
    anchors = {(u["corpus"], u["draw"], u["fold"]): u for u in ref["units"]
               if u["phase"] == "formal" and u["arm"] == "A"}
    assert len(anchors) == 360
    jobs = []
    for draw in range(24):
        for fold in range(5):
            for corpus in CORPORA:
                old = anchors[(corpus, draw, fold)]
                panel = {key: old[key] for key in (
                    "group_speakers", "test_speakers", "fit_prompts",
                    "query_prompts", "report", "report_batches")}
                combinations = [("hubert_base", "A", 15), ("wavlm_base_plus", "A", 15 if corpus == "cremad" else 45)]
                if corpus == "cremad":
                    combinations.append(("wavlm_base_plus", "B", 15))
                # Metadata-only order avoids running an entire model or direction in a separate time block.
                combinations.sort(key=lambda c: digest([PROGRAM, "order", corpus, draw, fold, list(c)]))
                for model, arm, epochs in combinations:
                    uid = f"ads1_formal_{corpus}_d{draw:02d}_f{fold}_{model}_{arm}"
                    jobs.append({
                        "unit_id": uid, "phase": "formal", "backend_required": "autodl",
                        "corpus": corpus, "draw": draw, "fold": fold,
                        "model": model, "arm": arm, "epochs": epochs,
                        "windows": [15] if epochs == 15 else [15, 45],
                        "classes": old["n_classes"],
                        "fit_rows_expected": {"cremad": 576, "subesco": 224, "ravdess": 64}[corpus],
                        "seen_group": arm, "unseen_group": "B" if arm == "A" else "A",
                        "reference_A_unit_id": old["unit_id"],
                        "reference_panel_sha256": digest(panel),
                        "seeds": {s: seed(corpus, draw, fold, s) for s in ("encoder_init", "head_init", "order", "crop", "torch_training")},
                        "permanent_checkpoint_sample": draw in (0, 12) and fold == 0,
                        "max_retained_delta_count": 5 if epochs == 15 else 10,
                        "fit_materialization": "Must derive and validate role slots and audio SHA before executable source freeze",
                    })
    assert len(jobs) == len({j["unit_id"] for j in jobs}) == 840
    assert Counter(j["epochs"] for j in jobs) == {15: 600, 45: 240}
    assert sum(j["epochs"] for j in jobs) == 19800
    by_key = {(j["corpus"], j["draw"], j["fold"], j["model"], j["arm"]): j for j in jobs}
    for j in jobs:
        counterpart = by_key[(j["corpus"], j["draw"], j["fold"], "wavlm_base_plus", "A")]
        assert j["reference_panel_sha256"] == counterpart["reference_panel_sha256"]
        assert j["seeds"] == counterpart["seeds"]
    samples = [j["unit_id"] for j in jobs if j["permanent_checkpoint_sample"]]
    assert len(samples) == 14
    assert sum(j["max_retained_delta_count"] for j in jobs if j["permanent_checkpoint_sample"]) == 90
    groups = [{"model": model, "corpus": corpus, "arm": arm, "epochs": epochs, "units": n}
              for (model, corpus, arm, epochs), n in sorted(Counter(
                  (j["model"], j["corpus"], j["arm"], j["epochs"]) for j in jobs).items())]
    assert len(groups) == 7 and all(g["units"] == 120 for g in groups)
    tests = []
    for corpus in CORPORA:
        tests += [{"id": f"P1_{corpus}_{endpoint}", "corpus": corpus, "endpoint": endpoint,
                   "comparison": "HuBERT A, window15, seen minus unseen"} for endpoint in ("D_CE", "J")]
    for corpus in ("subesco", "ravdess"):
        tests += [{"id": f"P2_{corpus}_{endpoint}", "corpus": corpus, "endpoint": endpoint,
                   "comparison": "WavLM A, window45 minus window15 on same trajectory"} for endpoint in ("L_CE", "L_J")]
    tests += [{"id": f"P3_cremad_{endpoint}", "corpus": "cremad", "endpoint": endpoint,
               "comparison": "Equal mean of new-cloud WavLM A/B directional effects"} for endpoint in ("S_CE", "S_J")]
    assert len(tests) == 12
    plan = {
        "schema": "ser-autodl-supplement-design-1", "program": PROGRAM,
        "status": "planning_only_not_an_execution_lock", "execution_ready": False,
        "reference_source_commit": "3ae0014d27a0af369f6bbe40da79b7f4d670b2c2",
        "reference_plan_byte_sha256": REFERENCE_BYTE_SHA,
        "reference_plan_semantic_sha256": ref["plan_sha256"],
        "formal_units": 840, "formal_epoch_passes": 19800, "groups": groups,
        "all_future_gpu_work_backend": "autodl",
        "same_host_and_environment_for_formal": True,
        "historical_local_results_enter_primary_pairs": False,
        "rng_domain": PROGRAM,
        "head_initialization": "Separate seed after encoder construction; same class head tensors across backbones, full initialization matched within WavLM A/B",
        "checkpoint_retention": {
            "policy": "all_epoch_predictions_and_receipts; all saved states validated while present; fixed14_units_weights_permanent",
            "fixed_sample_unit_ids": samples, "max_sample_deltas": 90,
            "max_all_deltas_if_retained": 5400,
            "require_verified_backup_before_non_sample_weights_removed": True,
            "old_archive_changed": False,
        },
        "statistics": {"replicate": "24 draw means after five-fold means; directions averaged first for P3",
                       "test": "two-sided one-sample t", "df": 23, "family": tests,
                       "multiplicity": "Holm, fixed 12 slots, alpha0.05", "intervals": "pointwise95%"},
        "cloud_candidate": {
            "gpu_spec_uuid": "5090-p", "gpu_count": 1, "vram_gb": 32,
            "cuda_v_from": 128, "expand_system_disk_by_gb_candidate": 100,
            "image_uuid_candidate": "base-image-mbr2n4urrc",
            "environment": "new Python3.11, torch2.11.0+cu128, torchaudio2.11.0+cu128",
            "hourly_price_yuan": None, "storage_price_yuan": None,
            "planning_gpu_window_hours": 24, "proposed_total_budget_yuan": 80,
            "budget_is_not_a_measured_quote_or_user_approved_spend_cap": True,
            "created": False, "real_name_status_current": "unconfirmed; earlier create returned TORealName",
        },
        "not_yet_complete": [
            "derive all fit slots including B folds1-4 and check audio union",
            "implement independent multi-backbone dual-window runner and new retention-aware gates",
            "hash model files, resolve full Linux dependency and source lock",
            "verify current real-name eligibility, stock and actual Pro prices",
            "AutoDL-only technical pilots and blinded resource admission",
        ],
        "job_matrix_semantic_sha256": digest(jobs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {"DESIGN.json": json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                "JOB_MATRIX.jsonl": "".join(json.dumps(j, ensure_ascii=False, sort_keys=True) + "\n" for j in jobs)}
    for name, text in payloads.items():
        target = args.output_dir / name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing design: {target}")
        target.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"formal_units": len(jobs), "formal_epoch_passes": 19800, "groups": groups,
                      "recovery_samples": len(samples), "new_primary_tests": len(tests),
                      "execution_ready": False, "job_matrix_semantic_sha256": digest(jobs)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

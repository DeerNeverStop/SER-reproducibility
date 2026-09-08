"""P4-D1: build the frozen 792-fit run_plan.csv.

Consumes ONLY anchored artifacts (fail-closed on any hash drift):
  - split_v2/split_manifest_v2.json  (gate-5 v2 manifest; tier-B hash pinned)
  - inner_split_inventory.csv        (99 NPZ paths + SHA-256, re-verified here)
  - pair_manifest.csv                (population; SHA pinned into every row)
  - the two channel caches           (SHA pinned into every row)

Ordering (assignment section 4): matched units -- the speech and song fits of
the same (model, seed, protocol, fold) are consecutive rows (speech first),
so an 8/22 hard stop cannot strand a half-matched grid. Unit-level order:
model -> protocol -> seed -> fold, mirroring P1's plan ordering.

train_seed is channel-INDEPENDENT by construction:
  stable_u32("D1-train|{model}|{protocol}|{fold}|{seed}")
so both channels of a matched unit build identical initial weights under
set_reproducible_seed -> BUILD_MODEL (asserted at run time via paired
init-state hashes, P5-style).

Expected totals: 4 models x 3 seeds x (5+5+23) folds x 2 channels = 792 rows.
"""
import csv
import json
import os
import sys
from pathlib import Path

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import p1_protocol_core as core  # noqa: E402

OUT_ROOT = Path(r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched")
V2_MANIFEST = OUT_ROOT / "split_v2" / "split_manifest_v2.json"
INVENTORY = OUT_ROOT / "inner_split_inventory.csv"
PAIR_MANIFEST = OUT_ROOT / "pair_manifest.csv"
NPZ_DIR = Path(r"E:\科研\claudework\07-ser-repro-protocol-audit\work\d1_splits")
CACHE = {
    "song": Path(r"E:\claudework_data\ICASSP2027-corpora\ravdess-song\cache\d1_logmel_song.npz"),
    "speech": Path(r"E:\claudework_data\ICASSP2027-corpora\ravdess-speech-common6\cache\d1_logmel_speech.npz"),
}
PLAN_PATH = OUT_ROOT / "run_plan.csv"

MODELS = ("cnn", "resnet_se", "transformer", "fno")
PROTOCOL_FOLDS = {"random": 5, "groupkfold": 5, "loso": 23}
SEEDS = (0, 1, 2)
CHANNELS = ("speech", "song")  # speech first inside each matched unit

FIELDS = [
    "sequence", "unit_id", "matched_unit_id", "channel", "model", "protocol", "seed",
    "outer_fold", "n_outer_folds", "n_fit", "n_validation", "n_outer_test",
    "train_seed_key", "train_seed_u32",
    "inner_split_path", "inner_split_sha256", "outer_fold_sha256",
    "pair_manifest_sha256", "cache_sha256", "split_manifest_v2_sha256", "tier_b_sha256",
]


def main():
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    tier_b = manifest["gate5_two_tier_hash"]["tier_b_canonical_slot_assignment_sha256"]
    if not tier_b["equal"]:
        raise RuntimeError("v2 manifest tier-B not equal -- refuse to plan")
    tier_b_sha = tier_b["derived_from_song_provenance_file"]
    v2_sha = core.sha256_file(V2_MANIFEST)
    pair_sha = core.sha256_file(PAIR_MANIFEST)
    if manifest["source_of_truth"]["pair_manifest_sha256"] != pair_sha:
        raise RuntimeError("pair_manifest drifted since v2 manifest was frozen")
    inv_sha = core.sha256_file(INVENTORY)
    if manifest["source_of_truth"]["inventory_sha256"] != inv_sha:
        raise RuntimeError("inner_split_inventory drifted since v2 manifest was frozen")

    cache_sha = {}
    for channel, path in CACHE.items():
        if not path.exists():
            raise RuntimeError(f"cache missing (build_d1_feature_cache.py first): {path}")
        cache_sha[channel] = core.sha256_file(path)

    inventory = {}
    with INVENTORY.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["protocol"], int(row["outer_fold"]), int(row["seed"]))
            npz_path = NPZ_DIR / row["inner_split_path"].replace("/", os.sep)
            if core.sha256_file(npz_path) != row["inner_split_sha256"]:
                raise RuntimeError(f"NPZ SHA drift (fail-closed): {npz_path}")
            inventory[key] = row
    if len(inventory) != 99:
        raise RuntimeError(f"inventory units {len(inventory)} != 99")

    rows = []
    sequence = 0
    for model in MODELS:
        for protocol, n_folds in PROTOCOL_FOLDS.items():
            for seed in SEEDS:
                for fold in range(n_folds):
                    inv = inventory[(protocol, fold, seed)]
                    seed_key = f"D1-train|{model}|{protocol}|{fold}|{seed}"
                    seed_u32 = core.stable_u32(seed_key)
                    matched_id = f"d1__{model}__{protocol}__s{seed}__f{fold:03d}"
                    for channel in CHANNELS:
                        sequence += 1
                        rows.append({
                            "sequence": str(sequence),
                            "unit_id": f"{matched_id}__{channel}",
                            "matched_unit_id": matched_id,
                            "channel": channel,
                            "model": model,
                            "protocol": protocol,
                            "seed": str(seed),
                            "outer_fold": str(fold),
                            "n_outer_folds": str(n_folds),
                            "n_fit": inv["n_fit"],
                            "n_validation": inv["n_validation"],
                            "n_outer_test": inv["n_outer_test"],
                            "train_seed_key": seed_key,
                            "train_seed_u32": str(seed_u32),
                            "inner_split_path": inv["inner_split_path"],
                            "inner_split_sha256": inv["inner_split_sha256"],
                            "outer_fold_sha256": inv["outer_fold_sha256"],
                            "pair_manifest_sha256": pair_sha,
                            "cache_sha256": cache_sha[channel],
                            "split_manifest_v2_sha256": v2_sha,
                            "tier_b_sha256": tier_b_sha,
                        })
    if len(rows) != 792:
        raise RuntimeError(f"run plan rows {len(rows)} != 792")
    core.atomic_write_csv(PLAN_PATH, FIELDS, rows)
    print("wrote", PLAN_PATH)
    print("run_plan_sha256", core.sha256_file(PLAN_PATH))
    print("rows", len(rows))


if __name__ == "__main__":
    main()

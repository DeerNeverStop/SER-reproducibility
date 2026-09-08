"""P4-D1 gate 5/9 v2 self-contained verifier, implementing exactly the checks
Codex's record-40 review specified:

  1. re-hash every NPZ against inner_split_inventory.csv (fail on drift);
  2. read the 99 NPZs directly; verify index mutual exclusion, fit+val+test
     partitioning, OOF exactly-once per protocol, grouped-protocol speaker
     zero-crossing (fit/val/test all disjoint by actor), and 6-class coverage
     of every fit and val subset;
  3. REBUILD the expected outer/inner roles from the NPZs and compare
     row-by-row against provenance_song_v2.json / provenance_speech_v2.json
     (zero mismatch required);
  4. mutation tests assert that an NPZ-vs-provenance INCONSISTENCY is caught
     by the comparison itself (fail-closed at the semantic level), not merely
     that editing a JSON changes a hash;
  5. recompute both tiers of the gate-5 hash from the on-disk provenance
     files and compare with split_manifest_v2.json.

Self-contained on purpose: imports neither the builder nor p1_protocol_core;
hashing and canonical JSON are re-implemented here so a shared bug cannot
hide a real defect.
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

OUT_ROOT = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched"
V2_DIR = os.path.join(OUT_ROOT, "split_v2")
NPZ_DIR = r"E:\科研\claudework\07-ser-repro-protocol-audit\work\d1_splits"
PROTOCOL_FOLDS = {"random": 5, "groupkfold": 5, "loso": 23}
SEEDS = (0, 1, 2)

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(("PASS  " if ok else "FAIL  ") + name)
    passed, failed = passed + (1 if ok else 0), failed + (0 if ok else 1)


def canonical_json(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(v):
    return hashlib.sha256(canonical_json(v).encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_channel(rows_):
    return [{"canonical_key": r["canonical_key"], "actor": r["actor"],
             "label6_index": r["label6_index"], "outer": r["outer"], "inner": r["inner"]}
            for r in rows_]


def compare_provenance_to_truth(prov, idx_of, outer_truth, inner_truth):
    """Return the number of provenance rows whose outer or inner records
    disagree with the NPZ-derived truth. This is the semantic fail-closed
    comparison the mutation tests exercise."""
    mismatch = 0
    for r in prov:
        i = idx_of[r["canonical_key"]]
        if any(r["outer"][p] != outer_truth[p][i] for p in PROTOCOL_FOLDS):
            mismatch += 1
            continue
        if any(r["inner"][tag] != role for tag, role in inner_truth[i].items()):
            mismatch += 1
    return mismatch


def main():
    # ---- load pair manifest (row order defines sample index) ----
    with open(os.path.join(OUT_ROOT, "pair_manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check("pair_manifest has 1012 rows", len(rows) == 1012)
    canonical_keys = ["-".join([r["actor"].zfill(2), r["emotion_code"].zfill(2), r["intensity"].zfill(2),
                                 r["statement"].zfill(2), r["repetition"].zfill(2)]) for r in rows]
    idx_of = {ck: i for i, ck in enumerate(canonical_keys)}
    actors = [r["actor"].zfill(2) for r in rows]
    labels6 = [int(r["emotion_code"]) - 1 for r in rows]

    # ---- 1. inventory SHA verification ----
    with open(os.path.join(OUT_ROOT, "inner_split_inventory.csv"), newline="", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))
    check("inventory has 99 rows", len(inventory) == 99)
    sha_ok = True
    npz = {}
    for row in inventory:
        path = os.path.join(NPZ_DIR, row["inner_split_path"].replace("/", os.sep))
        if sha256_file(path) != row["inner_split_sha256"]:
            sha_ok = False
        with np.load(path, allow_pickle=False) as d:
            npz[(row["protocol"], int(row["outer_fold"]), int(row["seed"]))] = {
                "fit": np.sort(d["fit_idx"]).astype(np.int64),
                "val": np.sort(d["val_idx"]).astype(np.int64),
                "test": np.sort(d["outer_test_idx"]).astype(np.int64),
            }
    check("all 99 NPZ SHA-256 match inventory (fail-closed source anchoring)", sha_ok)

    # ---- 2. NPZ-level invariants ----
    all_idx = np.arange(1012, dtype=np.int64)
    disjoint_ok = partition_ok = speaker_ok = class_ok = seeds_consistent = True
    for (p, fold, seed), d in npz.items():
        if np.intersect1d(d["fit"], d["val"]).size or np.intersect1d(d["fit"], d["test"]).size \
                or np.intersect1d(d["val"], d["test"]).size:
            disjoint_ok = False
        if not np.array_equal(np.sort(np.concatenate([d["fit"], d["val"], d["test"]])), all_idx):
            partition_ok = False
        if p != "random":
            fa = {actors[i] for i in d["fit"].tolist()}
            va = {actors[i] for i in d["val"].tolist()}
            ta = {actors[i] for i in d["test"].tolist()}
            if (fa | va) & ta or fa & va:
                speaker_ok = False
        for part in ("fit", "val"):
            if {labels6[i] for i in d[part].tolist()} != set(range(6)):
                class_ok = False
        if seed != 0 and not np.array_equal(d["test"], npz[(p, fold, 0)]["test"]):
            seeds_consistent = False
    check("every NPZ: fit/val/test mutually disjoint", disjoint_ok)
    check("every NPZ: fit+val+test partitions all 1012 samples", partition_ok)
    check("groupkfold/loso: fit/val/test actor sets pairwise disjoint (zero speaker crossing)", speaker_ok)
    check("every fit and val subset covers all 6 classes", class_ok)
    check("outer test identical across the 3 seeds of each fold", seeds_consistent)

    oof_ok = True
    outer_truth = {p: {} for p in PROTOCOL_FOLDS}
    for p, n_folds in PROTOCOL_FOLDS.items():
        for fold in range(n_folds):
            for i in npz[(p, fold, 0)]["test"].tolist():
                if i in outer_truth[p]:
                    oof_ok = False
                outer_truth[p][i] = fold
        if len(outer_truth[p]) != 1012:
            oof_ok = False
    check("OOF exactly-once per protocol (rebuilt from NPZs)", oof_ok)

    inner_truth = [dict() for _ in range(1012)]
    for (p, fold, seed), d in npz.items():
        tag = f"{p}|{fold}|{seed}"
        fit_set = set(d["fit"].tolist())
        val_set = set(d["val"].tolist())
        for i in range(1012):
            inner_truth[i][tag] = "fit" if i in fit_set else ("val" if i in val_set else "outer_test")

    # ---- 3. row-by-row provenance vs NPZ truth ----
    with open(os.path.join(V2_DIR, "provenance_song_v2.json"), encoding="utf-8") as f:
        prov_song = json.load(f)
    with open(os.path.join(V2_DIR, "provenance_speech_v2.json"), encoding="utf-8") as f:
        prov_speech = json.load(f)
    check("provenance_song_v2 has 1012 rows", len(prov_song) == 1012)
    check("provenance_speech_v2 has 1012 rows", len(prov_speech) == 1012)
    mm_song = compare_provenance_to_truth(prov_song, idx_of, outer_truth, inner_truth)
    mm_speech = compare_provenance_to_truth(prov_speech, idx_of, outer_truth, inner_truth)
    check("song provenance vs NPZ truth: 0 row mismatches", mm_song == 0)
    check("speech provenance vs NPZ truth: 0 row mismatches", mm_speech == 0)

    # regression guard against the exact v1 defect: outer folds must span every fold,
    # not collapse to the last one
    from collections import Counter
    fold_span_ok = True
    for p, n_folds in PROTOCOL_FOLDS.items():
        seen = Counter(r["outer"][p] for r in prov_song)
        if set(seen) != set(range(n_folds)):
            fold_span_ok = False
    check("regression (v1 defect): recorded outer folds span ALL folds of each protocol, "
          "not only the last one", fold_span_ok)

    # ---- 4. mutation tests at the semantic (NPZ-vs-provenance) level ----
    mutated = json.loads(json.dumps(prov_song))
    r0 = mutated[0]
    p0 = "groupkfold"
    r0["outer"][p0] = (r0["outer"][p0] + 1) % PROTOCOL_FOLDS[p0]
    check("FAIL-CLOSED: corrupting one row's outer_test_fold IS caught by the "
          "NPZ-vs-provenance comparison",
          compare_provenance_to_truth(mutated, idx_of, outer_truth, inner_truth) > 0)

    mutated2 = json.loads(json.dumps(prov_song))
    tag0 = "random|0|0"
    cur = mutated2[5]["inner"][tag0]
    mutated2[5]["inner"][tag0] = "fit" if cur != "fit" else "val"
    check("FAIL-CLOSED: corrupting one row's inner role IS caught by the "
          "NPZ-vs-provenance comparison",
          compare_provenance_to_truth(mutated2, idx_of, outer_truth, inner_truth) > 0)

    # simulate the v1 defect wholesale and confirm this verifier would have caught it
    v1_style = json.loads(json.dumps(prov_song))
    last = {"random": 4, "groupkfold": 4, "loso": 22}
    for r in v1_style:
        for p in PROTOCOL_FOLDS:
            r["outer"][p] = last[p]
    v1_mm = compare_provenance_to_truth(v1_style, idx_of, outer_truth, inner_truth)
    check(f"FAIL-CLOSED: a v1-style last-fold-only provenance is rejected "
          f"({v1_mm} rows flagged)", v1_mm > 0)

    # ---- 5. hash recomputation vs manifest ----
    with open(os.path.join(V2_DIR, "split_manifest_v2.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    g5 = manifest["gate5_two_tier_hash"]
    check("tier-A song hash matches manifest",
          sha256_json(prov_song) == g5["tier_a_per_channel_provenance_sha256"]["song"])
    check("tier-A speech hash matches manifest",
          sha256_json(prov_speech) == g5["tier_a_per_channel_provenance_sha256"]["speech"])
    check("tier-A song != speech", sha256_json(prov_song) != sha256_json(prov_speech))
    tb_song = sha256_json(strip_channel(prov_song))
    tb_speech = sha256_json(strip_channel(prov_speech))
    check("tier-B from song file matches manifest",
          tb_song == g5["tier_b_canonical_slot_assignment_sha256"]["derived_from_song_provenance_file"])
    check("tier-B from speech file matches manifest",
          tb_speech == g5["tier_b_canonical_slot_assignment_sha256"]["derived_from_speech_provenance_file"])
    check("tier-B equal across channels", tb_song == tb_speech)
    check("superseded v1 tier-B hash recorded and differs from v2",
          manifest["supersedes"]["tier_b_v1"] != tb_song and len(manifest["supersedes"]["tier_b_v1"]) == 64)

    print(f"\n{passed} PASS, {failed} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

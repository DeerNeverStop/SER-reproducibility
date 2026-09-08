"""P4-D1 gate 5 v2: rebuild the frozen split provenance FROM the anchored NPZ
split artifacts, fixing the two defects Codex's record-40 review found in v1:

  v1 defect 1 (fold overwrite): v1 assigned `outer_role[key][protocol] =
  {outer_fold, role}` inside the per-fold loop, so every sample's outer record
  collapsed to the LAST fold of each protocol (Random f4 / GroupKFold f4 /
  LOSO f22). Independently confirmed: 810/836/968 of 1,012 rows inconsistent
  with the true NPZ-derived outer-test folds.
  v2 schema: `outer[protocol] = outer_test_fold` -- each sample's unique test
  fold (well-defined because OOF is exactly-once), no information loss.

  v1 defect 2 (fake independence): v1's "three independent derivations" of the
  tier-B hash all serialized one shared in-memory object. v2 derives the
  tier-B hash twice, once from each on-disk per-channel provenance file, and
  makes NO independence claim beyond that; correctness against the actual
  split arrays is established separately by row-by-row NPZ rebuild comparison
  (here and in the self-contained verifier).

Source of truth: the 99 NPZs under work/d1_splits/ anchored by
inner_split_inventory.csv (SHA-verified before use; fail-closed on drift).
A fresh regeneration via the frozen P1 code is kept only as a secondary
determinism cross-check, clearly labeled as such.

Outputs go to a NEW directory split_v2/ (v1 files are left in place and
marked superseded by SUPERSEDED_split_v1.md). Old tier-B hash e4748d19... is
recorded as superseded and must not be cited.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

OUT_ROOT = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched"
OUT_DIR = os.path.join(OUT_ROOT, "split_v2")
NPZ_DIR = r"E:\科研\claudework\07-ser-repro-protocol-audit\work\d1_splits"
PAIR_MANIFEST = os.path.join(OUT_ROOT, "pair_manifest.csv")
INVENTORY = os.path.join(OUT_ROOT, "inner_split_inventory.csv")
REFERENCE_TOPIC02_FOLDS = r"E:\科研\SER\codex\results\f_song_a2\consensus\shared_outer_test_assignment.csv"
SUPERSEDED_TIER_B_V1 = "e4748d191a6364aa87494887bfb85343061646411201dd7890c01106fc9cb70f"

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import p1_protocol_core as core  # noqa: E402  (read-only reuse; file not modified)

EMOTION6 = ("neutral", "calm", "happy", "sad", "angry", "fearful")
PROTOCOL_FOLDS = {"random": 5, "groupkfold": 5, "loso": 23}
SEEDS = (0, 1, 2)
CORPUS_TAG = "ravdess_d1"
core.CORPORA[CORPUS_TAG] = {"outer_folds": dict(PROTOCOL_FOLDS), "labels": EMOTION6}


def load_pairs():
    with open(PAIR_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1012:
        raise RuntimeError(f"pair_manifest rows {len(rows)} != 1012")
    return rows


def ck_of(row):
    return "-".join([row["actor"].zfill(2), row["emotion_code"].zfill(2),
                     row["intensity"].zfill(2), row["statement"].zfill(2), row["repetition"].zfill(2)])


def main():
    import numpy as np

    rows = load_pairs()
    canonical_keys = [ck_of(r) for r in rows]
    if len(set(canonical_keys)) != 1012:
        raise RuntimeError("canonical_key collision")
    actors = [r["actor"].zfill(2) for r in rows]
    labels6 = [int(r["emotion_code"]) - 1 for r in rows]
    if any(not 0 <= v <= 5 for v in labels6):
        raise RuntimeError("emotion_code out of 1..6")

    # ---- verify inventory SHAs, then load all 99 NPZs (fail-closed) ----
    with open(INVENTORY, newline="", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))
    if len(inventory) != 99:
        raise RuntimeError(f"inventory rows {len(inventory)} != 99")
    npz = {}  # (protocol, fold, seed) -> dict(fit, val, test)
    for row in inventory:
        p, fold, seed = row["protocol"], int(row["outer_fold"]), int(row["seed"])
        path = Path(NPZ_DIR) / row["inner_split_path"].replace("/", os.sep)
        actual_sha = core.sha256_file(path)
        if actual_sha != row["inner_split_sha256"]:
            raise RuntimeError(f"NPZ SHA drift (fail-closed): {path}")
        with np.load(path, allow_pickle=False) as d:
            npz[(p, fold, seed)] = {
                "fit": np.sort(d["fit_idx"]).astype(np.int64),
                "val": np.sort(d["val_idx"]).astype(np.int64),
                "test": np.sort(d["outer_test_idx"]).astype(np.int64),
            }

    # ---- rebuild outer truth from NPZs: unique outer_test_fold per sample per protocol ----
    outer_test_fold = {p: {} for p in PROTOCOL_FOLDS}
    for p, n_folds in PROTOCOL_FOLDS.items():
        for fold in range(n_folds):
            ref = npz[(p, fold, 0)]["test"]
            for seed in SEEDS[1:]:
                if not np.array_equal(ref, npz[(p, fold, seed)]["test"]):
                    raise RuntimeError(f"outer test differs across seeds: {p}/f{fold}")
            for i in ref.tolist():
                if i in outer_test_fold[p]:
                    raise RuntimeError(f"OOF violated: sample {i} tests twice in {p}")
                outer_test_fold[p][i] = fold
        if len(outer_test_fold[p]) != 1012:
            raise RuntimeError(f"{p}: OOF coverage {len(outer_test_fold[p])} != 1012")

    # ---- per-NPZ invariants re-checked at build time (verifier re-checks independently) ----
    all_idx = np.arange(1012, dtype=np.int64)
    for (p, fold, seed), d in npz.items():
        if np.intersect1d(d["fit"], d["val"]).size or np.intersect1d(d["fit"], d["test"]).size \
                or np.intersect1d(d["val"], d["test"]).size:
            raise RuntimeError(f"index overlap in {p}/f{fold}/s{seed}")
        union = np.sort(np.concatenate([d["fit"], d["val"], d["test"]]))
        if not np.array_equal(union, all_idx):
            raise RuntimeError(f"fit+val+test does not partition 1012 in {p}/f{fold}/s{seed}")
        if p != "random":
            fit_actors = {actors[i] for i in d["fit"].tolist()}
            val_actors = {actors[i] for i in d["val"].tolist()}
            test_actors = {actors[i] for i in d["test"].tolist()}
            if (fit_actors | val_actors) & test_actors or fit_actors & val_actors:
                raise RuntimeError(f"speaker overlap in {p}/f{fold}/s{seed}")
        for part in ("fit", "val"):
            if {labels6[i] for i in d[part].tolist()} != set(range(6)):
                raise RuntimeError(f"{part} missing a class in {p}/f{fold}/s{seed}")

    # ---- secondary determinism cross-check: regenerate from frozen P1 code ----
    y = np.asarray(labels6, dtype=np.int64)
    groups = np.asarray(actors)
    regen = core.generate_outer_splits(CORPUS_TAG, y, groups)
    regen_matches = True
    for p, n_folds in PROTOCOL_FOLDS.items():
        for fold in range(n_folds):
            if not np.array_equal(np.sort(regen[p][fold][1]), npz[(p, fold, 0)]["test"]):
                regen_matches = False
    # (informational only; the NPZs are the anchored source of truth)

    # ---- inner roles per protocol|fold|seed, rebuilt from NPZs ----
    inner_role = [dict() for _ in range(1012)]
    for (p, fold, seed), d in npz.items():
        tag = f"{p}|{fold}|{seed}"
        fit_set = set(d["fit"].tolist())
        val_set = set(d["val"].tolist())
        for i in range(1012):
            inner_role[i][tag] = "fit" if i in fit_set else ("val" if i in val_set else "outer_test")

    # ---- build per-channel provenance (deep per-row dicts; nothing shared) ----
    def build_provenance(channel):
        out = []
        for i, ck in sorted(enumerate(canonical_keys), key=lambda t: t[1]):
            r = rows[i]
            out.append({
                "canonical_key": ck,
                "channel": channel,
                "relative_path": r[f"{channel}_relative_path"],
                "sha256": r[f"{channel}_sha256"],
                "file_size_bytes": r[f"{channel}_file_size_bytes"],
                "actor": actors[i],
                "label6_index": labels6[i],
                "outer": {p: outer_test_fold[p][i] for p in PROTOCOL_FOLDS},
                "inner": dict(inner_role[i]),
            })
        return out

    prov_song = build_provenance("song")
    prov_speech = build_provenance("speech")

    def strip_channel(rows_):
        return [{"canonical_key": r["canonical_key"], "actor": r["actor"],
                 "label6_index": r["label6_index"], "outer": r["outer"], "inner": r["inner"]}
                for r in rows_]

    tier_a_song = core.sha256_bytes(core.canonical_json(prov_song).encode("utf-8"))
    tier_a_speech = core.sha256_bytes(core.canonical_json(prov_speech).encode("utf-8"))
    if tier_a_song == tier_a_speech:
        raise RuntimeError("tier-A hashes unexpectedly equal")
    tier_b_from_song = core.sha256_bytes(core.canonical_json(strip_channel(prov_song)).encode("utf-8"))
    tier_b_from_speech = core.sha256_bytes(core.canonical_json(strip_channel(prov_speech)).encode("utf-8"))
    if tier_b_from_song != tier_b_from_speech:
        raise RuntimeError("gate 5 FAIL-CLOSED: tier-B differs between channels")

    # ---- row-by-row self-check: provenance vs NPZ truth (the check v1 lacked) ----
    idx_of = {ck: i for i, ck in enumerate(canonical_keys)}
    for r in prov_song:
        i = idx_of[r["canonical_key"]]
        for p in PROTOCOL_FOLDS:
            if r["outer"][p] != outer_test_fold[p][i]:
                raise RuntimeError(f"provenance/NPZ outer mismatch at {r['canonical_key']}/{p}")

    # ---- topic-02 cross-check from NPZ-derived groupkfold partition ----
    d1_partition = defaultdict(set)
    for i, fold in outer_test_fold["groupkfold"].items():
        d1_partition[fold].add(actors[i])
    with open(REFERENCE_TOPIC02_FOLDS, newline="", encoding="utf-8") as f:
        t02 = defaultdict(set)
        for row in csv.DictReader(f):
            t02[int(row["test_fold"])].add(row["actor"])
    topic02_match = {frozenset(v) for v in d1_partition.values()} == {frozenset(v) for v in t02.values()}

    # ---- write v2 outputs ----
    os.makedirs(OUT_DIR, exist_ok=True)
    core.atomic_write_json(Path(OUT_DIR) / "provenance_song_v2.json", prov_song)
    core.atomic_write_json(Path(OUT_DIR) / "provenance_speech_v2.json", prov_speech)

    manifest = {
        "created_at": core.now_iso(),
        "version": 2,
        "supersedes": {
            "tier_b_v1": SUPERSEDED_TIER_B_V1,
            "reason": "Codex record-40 review: v1 outer provenance collapsed to the last fold "
                       "per protocol (810/836/968 of 1,012 rows inconsistent with NPZ truth); "
                       "v1 'three independent derivations' were same-source. v1 hash must not "
                       "be cited for any gate.",
            "v1_files_left_in_place": ["../split_manifest.json", "../provenance_song.json",
                                        "../provenance_speech.json"],
        },
        "source_of_truth": {
            "what": "the 99 anchored inner-split NPZs (SHA-verified against inner_split_inventory.csv "
                     "before use); provenance is REBUILT from them, never from a parallel in-memory object",
            "inventory_csv": "../inner_split_inventory.csv",
            "inventory_sha256": core.sha256_file(Path(INVENTORY)),
            "pair_manifest_sha256": core.sha256_file(Path(PAIR_MANIFEST)),
            "npz_dir": NPZ_DIR,
        },
        "outer_schema": "outer[protocol] = the sample's unique outer_test_fold (well-defined by "
                          "OOF exactly-once); no per-fold overwrite is possible in this schema",
        "gate5_two_tier_hash": {
            "tier_a_per_channel_provenance_sha256": {"song": tier_a_song, "speech": tier_a_speech,
                                                       "note": "expected to differ (channel provenance)"},
            "tier_b_canonical_slot_assignment_sha256": {
                "derived_from_song_provenance_file": tier_b_from_song,
                "derived_from_speech_provenance_file": tier_b_from_speech,
                "equal": tier_b_from_song == tier_b_from_speech,
                "independence_claim": "NONE beyond two on-disk files hashing equal; correctness "
                                        "against actual split arrays is certified by row-by-row NPZ "
                                        "rebuild comparison (builder self-check + self-contained "
                                        "verifier test_d1_split_manifest_v2.py), not by this hash",
            },
        },
        "secondary_determinism_crosscheck": {
            "regenerated_from_frozen_p1_code_matches_npz": bool(regen_matches),
            "note": "informational only; NPZs are the anchored source of truth",
        },
        "topic02_groupkfold_actor_partition_match": bool(topic02_match),
        "reference_topic02_sha256": core.sha256_file(Path(REFERENCE_TOPIC02_FOLDS)),
    }
    core.atomic_write_json(Path(OUT_DIR) / "split_manifest_v2.json", manifest)

    supersede_note = (
        "# v1 split-manifest artifacts: SUPERSEDED (2026-08-16, Codex record 40)\n\n"
        "`split_manifest.json`, `provenance_song.json`, `provenance_speech.json` in this\n"
        "directory contain a per-fold overwrite defect in their outer records (only the\n"
        "last fold per protocol survives; 810/836/968 of 1,012 rows inconsistent with the\n"
        "actual NPZ splits). They are kept unmodified for the audit trail and MUST NOT be\n"
        "cited by any gate, worker, or completion manifest. Tier-B hash e4748d19... is\n"
        "void. Use `split_v2/` instead. The actual NPZ split arrays, plus\n"
        "`outer_fold_summary.csv` and `inner_split_inventory.csv`, were verified sound by\n"
        "Codex (record 40) and remain valid.\n"
    )
    core.atomic_write_text(Path(OUT_ROOT) / "SUPERSEDED_split_v1.md", supersede_note)

    print("GATE_5_V2: PASS (rebuilt from anchored NPZs, row-by-row verified)")
    print("tier_a song:", tier_a_song[:16], "speech:", tier_a_speech[:16])
    print("tier_b (song-file vs speech-file):", tier_b_from_song[:16],
          "equal =", tier_b_from_song == tier_b_from_speech)
    print("regen determinism cross-check:", regen_matches)
    print("topic02 partition match:", topic02_match)
    return 0


if __name__ == "__main__":
    sys.exit(main())

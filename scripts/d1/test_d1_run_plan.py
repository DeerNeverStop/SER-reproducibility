"""P4-D1 run-plan self-contained verifier.

Independent by construction: imports neither the plan builder, the runner,
p1_protocol_core, nor run_p1_protocol_premium. Hashing (file SHA-256 and the
stable-u32 seed derivation) is re-implemented locally so a shared bug cannot
mask a defect.

Checks:
  - 792 rows; 396 matched units; each matched unit is exactly two ADJACENT
    rows (speech then song) -- the 8/22 hard-stop cannot strand half a unit;
  - train_seed is channel-independent within a matched unit and matches an
    independent re-derivation of stable_u32("D1-train|model|protocol|fold|seed");
  - per-protocol fold counts 5/5/23, 4 models, 3 seeds, all unit_ids unique;
  - every distinct NPZ referenced exists and its SHA-256 matches BOTH the plan
    row and inner_split_inventory.csv;
  - anchoring constants (pair manifest, v2 manifest, tier-B) are uniform
    across rows and match the actual files on disk;
  - the two channels reference DIFFERENT cache files with different hashes,
    each matching the real cache on disk;
  - mutation (fail-closed): corrupting a row's seed / channel order / NPZ sha
    in memory is detected by the corresponding check.
"""
import csv
import hashlib
import json
import os
import sys

PLAN = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched\run_plan.csv"
OUT_ROOT = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched"
NPZ_DIR = r"E:\科研\claudework\07-ser-repro-protocol-audit\work\d1_splits"
CACHE = {
    "song": r"E:\claudework_data\ICASSP2027-corpora\ravdess-song\cache\d1_logmel_song.npz",
    "speech": r"E:\claudework_data\ICASSP2027-corpora\ravdess-speech-common6\cache\d1_logmel_speech.npz",
}
PROTOCOL_FOLDS = {"random": 5, "groupkfold": 5, "loso": 23}

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(("PASS  " if ok else "FAIL  ") + name)
    passed, failed = passed + (1 if ok else 0), failed + (0 if ok else 1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_u32(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def pair_adjacency_ok(rows):
    for i in range(0, len(rows), 2):
        a, b = rows[i], rows[i + 1]
        if a["matched_unit_id"] != b["matched_unit_id"]:
            return False
        if a["channel"] != "speech" or b["channel"] != "song":
            return False
    return True


def seeds_ok(rows):
    for row in rows:
        expected = stable_u32(
            f"D1-train|{row['model']}|{row['protocol']}|{int(row['outer_fold'])}|{int(row['seed'])}")
        if int(row["train_seed_u32"]) != expected:
            return False
    for i in range(0, len(rows), 2):
        if rows[i]["train_seed_u32"] != rows[i + 1]["train_seed_u32"]:
            return False
    return True


def main():
    with open(PLAN, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check("792 rows", len(rows) == 792)
    check("396 unique matched units", len({r["matched_unit_id"] for r in rows}) == 396)
    check("all 792 unit_ids unique", len({r["unit_id"] for r in rows}) == 792)
    check("matched units are adjacent speech-then-song pairs", pair_adjacency_ok(rows))
    check("train_seed: channel-independent AND matches independent re-derivation", seeds_ok(rows))

    combos = {(r["model"], r["protocol"], r["seed"], r["outer_fold"]) for r in rows}
    check("396 unique (model,protocol,seed,fold) combos", len(combos) == 396)
    from collections import Counter
    per_protocol = Counter(r["protocol"] for r in rows)
    check("fold counts per protocol (x4 models x3 seeds x2 channels)",
          per_protocol == Counter({"loso": 23 * 24, "random": 5 * 24, "groupkfold": 5 * 24}))
    check("4 models present", len({r["model"] for r in rows}) == 4)
    check("3 seeds present", {r["seed"] for r in rows} == {"0", "1", "2"})

    # NPZ anchoring vs plan AND vs inventory
    with open(os.path.join(OUT_ROOT, "inner_split_inventory.csv"), newline="", encoding="utf-8") as f:
        inv = {r["inner_split_path"]: r["inner_split_sha256"] for r in csv.DictReader(f)}
    plan_npz = {r["inner_split_path"]: r["inner_split_sha256"] for r in rows}
    check("plan references exactly the 99 inventoried NPZs", set(plan_npz) == set(inv))
    check("plan NPZ SHAs match inventory", all(plan_npz[k] == inv[k] for k in inv))
    disk_ok = all(sha256_file(os.path.join(NPZ_DIR, k.replace("/", os.sep))) == v
                  for k, v in plan_npz.items())
    check("all 99 NPZ SHAs verified on disk", disk_ok)

    # anchoring constants uniform + match disk
    v2_sha = sha256_file(os.path.join(OUT_ROOT, "split_v2", "split_manifest_v2.json"))
    pair_sha = sha256_file(os.path.join(OUT_ROOT, "pair_manifest.csv"))
    check("v2 manifest sha uniform and matches disk",
          {r["split_manifest_v2_sha256"] for r in rows} == {v2_sha})
    check("pair manifest sha uniform and matches disk",
          {r["pair_manifest_sha256"] for r in rows} == {pair_sha})
    with open(os.path.join(OUT_ROOT, "split_v2", "split_manifest_v2.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tb = manifest["gate5_two_tier_hash"]["tier_b_canonical_slot_assignment_sha256"]
    check("tier-B sha uniform and matches v2 manifest",
          {r["tier_b_sha256"] for r in rows} == {tb["derived_from_song_provenance_file"]})

    cache_by_channel = {ch: {r["cache_sha256"] for r in rows if r["channel"] == ch}
                        for ch in ("speech", "song")}
    check("each channel pins exactly one cache sha",
          all(len(v) == 1 for v in cache_by_channel.values()))
    check("speech and song pin DIFFERENT caches",
          cache_by_channel["speech"] != cache_by_channel["song"])
    check("song cache sha matches disk",
          cache_by_channel["song"] == {sha256_file(CACHE["song"])})
    check("speech cache sha matches disk",
          cache_by_channel["speech"] == {sha256_file(CACHE["speech"])})

    # mutations (in memory; fail-closed demonstrations)
    m1 = [dict(r) for r in rows]
    m1[10]["train_seed_u32"] = str(int(m1[10]["train_seed_u32"]) ^ 1)
    check("FAIL-CLOSED: corrupted train_seed detected", not seeds_ok(m1))
    m2 = [dict(r) for r in rows]
    m2[0], m2[1] = m2[1], m2[0]  # song before speech
    check("FAIL-CLOSED: broken channel order detected", not pair_adjacency_ok(m2))
    m3 = dict(plan_npz)
    first = next(iter(m3))
    m3[first] = "0" * 64
    check("FAIL-CLOSED: corrupted NPZ sha detected", not all(m3[k] == inv[k] for k in inv))

    print(f"\n{passed} PASS, {failed} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

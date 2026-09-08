"""P4-D1 scorer: assignment section-6 aggregation, written BEFORE any real
predictions are read.

HARD GATE (fail-closed, dialogue records 54 + 56 + 58 + 60): scoring the real
training tree requires BOTH
  training/completion_manifest_v6.json   (prescore amendment keying attempt
                                          tree digest + this scorer's hash)
  training/SCORING_UNLOCKED.json         (STRUCTURED authorization, written
                                          only after Codex's pre-score review
                                          PASSes; see validate_unlock)
The unlock is parsed and every field exactly compared -- an empty file, a
wrong hash, or a wrong review-record number all refuse. main() additionally
re-hashes THIS file against the sha frozen in v6, re-hashes pair_manifest.csv
and run_plan.csv against v6, acquires an O_CREAT|O_EXCL scoring claim BEFORE
the first outcome byte is read (a crashed claim is audit evidence, never
silently cleared), writes claim/output/receipt through write_all (looped
os.write; a 0-byte write raises) with post-write length+sha verification
against the serialized buffer before the atomic os.replace commit, and leaves
a postscore receipt anchoring the output AND claim content hashes plus the
v6 hash. verify_postscore_receipt() (record 60 action 1) then re-checks the
closure read-only with FIXED file names and full identity cross-checking:
the receipt, the parsed claim, and the parsed output must agree on the
v6/unlock/scorer hashes, and each must equal the live hash of the actual
v6 manifest, unlock file, and this scorer; output/claim live bytes must
match the receipt's content hashes. Failures after the output commit leave
claim/output/temp as audit evidence and refuse silent rescoring -- they are
NOT rolled back (see the runbook's corrected wording).

Input identity closure (record 56 action 3): every predictions.csv is
byte-hash-gated against the completion manifest, then each row's true label,
canonical actor and relative path are checked against the frozen pair
manifest, and each unit's sample set must equal the outer_test_idx of the
frozen split NPZ named by the run plan (NPZ re-hashed first).

Aggregation per model (assignment section 6, in order):
  1. actor-level OOF UAR per (actor, channel, protocol, seed);
  2. within actor: P_RG = UAR(random) - UAR(groupkfold)   [P_RL secondary];
  3. within actor: mean over the 3 seeds;
  4. within actor: D_mode = P_RG(song) - P_RG(speech);
  5. equal-weight mean over the 23 actors;
  6. actor-cluster reweighting: 10,000 draws of 23 actors with replacement
     (fixed seed), complete per-actor vectors kept together;
  7. report: point estimates, all 23 per-actor contrasts, leave-one-actor-out
     stability range, the 2.5/97.5 quantiles of step 6 -- named
     "actor_reweighting_95_stability_quantile_range", NEVER "CI" -- and
     (record 56 action 4) the three paired-seed panel estimates plus
     seed mean +/- sample SD for each headline quantity. Seed SD describes
     training randomness only; the panel n remains the 23 actors.

Missing units: any absent (channel, protocol, seed) cell marks every contrast
that needs it as incomplete/not_conclusive for that model; partial cells are
never silently pooled. No p-values are produced.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

RESULT_ROOT = Path(r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched")
TRAIN_ROOT = RESULT_ROOT / "training"
COMPLETION_V6 = TRAIN_ROOT / "completion_manifest_v6.json"
UNLOCK = TRAIN_ROOT / "SCORING_UNLOCKED.json"
PAIR_MANIFEST = RESULT_ROOT / "pair_manifest.csv"
RUN_PLAN = RESULT_ROOT / "run_plan.csv"
NPZ_DIR = Path(r"E:\科研\claudework\07-ser-repro-protocol-audit\work\d1_splits")
SCORING_DIR = RESULT_ROOT / "scoring"
SCORING_OUT = SCORING_DIR / "d1_scoring_report.json"
SCORING_CLAIM = SCORING_DIR / "scoring_claim.json"
RECEIPT = SCORING_DIR / "postscore_receipt.json"
EMOTION6 = ("neutral", "calm", "happy", "sad", "angry", "fearful")
PRED_HEADER = ("unit_id", "channel", "model", "protocol", "seed", "outer_fold",
                "sample_index", "canonical_actor", "relative_path",
                "true_label_index", "true_label_name",
                "predicted_label_index", "predicted_label_name",
                "logit_neutral", "logit_calm", "logit_happy", "logit_sad",
                "logit_angry", "logit_fearful")

MODELS = ("cnn", "resnet_se", "transformer", "fno")
CHANNELS = ("speech", "song")
PROTOCOLS = ("random", "groupkfold", "loso")
SEEDS = (0, 1, 2)
N_CLASSES = 6
REWEIGHT_DRAWS = 10_000
REWEIGHT_SEED = 202608160300  # frozen before any real predictions are read
HEADLINE_KEYS = ("P_RG_speech", "P_RG_song", "D_mode_RG",
                  "P_RL_speech", "P_RL_song", "D_mode_RL")


def actor_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy over the classes actually present for this actor."""
    recalls = []
    for cls in range(N_CLASSES):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == cls).mean()))
    if not recalls:
        raise ValueError("actor has no samples")
    return float(np.mean(recalls))


def collect_actor_uars(predictions: dict, actors_of_sample: dict) -> dict:
    """predictions[(channel, protocol, seed)] = {sample_index: (true, pred)}.

    Verifies each cell covers every sample exactly once, then returns
    uar[(actor, channel, protocol, seed)].
    """
    n_samples = len(actors_of_sample)
    out = {}
    for (channel, protocol, seed), cell in predictions.items():
        if len(cell) != n_samples:
            raise ValueError(
                f"cell {channel}/{protocol}/s{seed} covers {len(cell)} samples, "
                f"expected {n_samples} (OOF exactly-once violated)")
        by_actor: dict[str, list[tuple[int, int]]] = {}
        for sample_index, (t, p) in cell.items():
            by_actor.setdefault(actors_of_sample[sample_index], []).append((t, p))
        for actor, pairs in by_actor.items():
            arr = np.asarray(pairs, dtype=np.int64)
            out[(actor, channel, protocol, seed)] = actor_uar(arr[:, 0], arr[:, 1])
    return out


def aggregate_model(uars: dict, actors: list[str], secondary_protocol: str = "loso") -> dict:
    """Steps 2-7 for one model. `uars` is the dict from collect_actor_uars.

    Returns a report dict; if any needed (actor, channel, protocol, seed) is
    missing, returns {"status": "incomplete/not_conclusive", "missing": [...]}
    without partial pooling.
    """
    missing = []
    for actor in actors:
        for channel in CHANNELS:
            for protocol in PROTOCOLS:
                for seed in SEEDS:
                    if (actor, channel, protocol, seed) not in uars:
                        missing.append(f"{actor}/{channel}/{protocol}/s{seed}")
    if missing:
        return {"status": "incomplete/not_conclusive", "missing": missing}

    def premium(actor, channel, grouped_protocol):
        per_seed = [uars[(actor, channel, "random", s)] - uars[(actor, channel, grouped_protocol, s)]
                    for s in SEEDS]
        return float(np.mean(per_seed)), [float(v) for v in per_seed]

    per_actor = {}
    for actor in actors:
        rg_speech, rg_speech_seeds = premium(actor, "speech", "groupkfold")
        rg_song, rg_song_seeds = premium(actor, "song", "groupkfold")
        rl_speech, rl_speech_seeds = premium(actor, "speech", secondary_protocol)
        rl_song, rl_song_seeds = premium(actor, "song", secondary_protocol)
        per_actor[actor] = {
            "P_RG_speech": rg_speech, "P_RG_song": rg_song,
            "P_RG_speech_per_seed": rg_speech_seeds, "P_RG_song_per_seed": rg_song_seeds,
            "D_mode_RG": rg_song - rg_speech,
            "P_RL_speech": rl_speech, "P_RL_song": rl_song,
            "P_RL_speech_per_seed": rl_speech_seeds, "P_RL_song_per_seed": rl_song_seeds,
            "D_mode_RL": rl_song - rl_speech,
        }

    def panel_mean(key):
        return float(np.mean([per_actor[a][key] for a in actors]))

    def loao_range(key):
        values = []
        for left_out in actors:
            rest = [per_actor[a][key] for a in actors if a != left_out]
            values.append(float(np.mean(rest)))
        return {"min": min(values), "max": max(values)}

    # record 56 action 4: paired-seed panel estimates. For seed s, each
    # actor's premium at that seed is paired across channels (both arms of a
    # matched unit share the channel-independent train seed), so D_mode at
    # seed s is the panel mean of (song_s - speech_s).
    def seed_panel(key: str, seed_pos: int) -> float:
        if key == "D_mode_RG":
            vals = [per_actor[a]["P_RG_song_per_seed"][seed_pos]
                     - per_actor[a]["P_RG_speech_per_seed"][seed_pos] for a in actors]
        elif key == "D_mode_RL":
            vals = [per_actor[a]["P_RL_song_per_seed"][seed_pos]
                     - per_actor[a]["P_RL_speech_per_seed"][seed_pos] for a in actors]
        else:
            vals = [per_actor[a][f"{key}_per_seed"][seed_pos] for a in actors]
        return float(np.mean(vals))

    seedwise = {key: [seed_panel(key, i) for i in range(len(SEEDS))]
                 for key in HEADLINE_KEYS}
    seed_mean_sd = {key: {"mean": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)),
                            "n_seeds": len(SEEDS)}
                     for key, vals in seedwise.items()}

    rng = np.random.default_rng(REWEIGHT_SEED)
    actor_arr = np.asarray(actors)
    draws = {"D_mode_RG": [], "D_mode_RL": [], "P_RG_speech": [], "P_RG_song": []}
    for _ in range(REWEIGHT_DRAWS):
        drawn = rng.choice(actor_arr, size=len(actors), replace=True)
        for key in draws:
            draws[key].append(float(np.mean([per_actor[a][key] for a in drawn])))

    def stability(key):
        q = np.quantile(np.asarray(draws[key]), [0.025, 0.975])
        return {"q2_5": float(q[0]), "q97_5": float(q[1]),
                "name": "actor_reweighting_95_stability_quantile_range",
                "not_a_confidence_interval": True}

    return {
        "status": "complete",
        "n_actors": len(actors),
        "point_estimates": {key: panel_mean(key) for key in HEADLINE_KEYS},
        "seedwise_panel_estimates": seedwise,
        "seed_mean_sd": seed_mean_sd,
        "seed_sd_note": "SD over the 3 paired training seeds describes training "
                         "randomness only; the panel n remains the actors and no "
                         "new speaker-level inference is implied",
        "per_actor": per_actor,
        "leave_one_actor_out": {key: loao_range(key)
                                  for key in ("D_mode_RG", "D_mode_RL", "P_RG_speech", "P_RG_song")},
        "actor_reweighting": {key: stability(key)
                                for key in ("D_mode_RG", "D_mode_RL", "P_RG_speech", "P_RG_song")},
        "reweighting_draws": REWEIGHT_DRAWS,
        "reweighting_seed": REWEIGHT_SEED,
        "p_values": None,
        "terminology_note": "quantile ranges are finite-panel stability descriptions, not CIs",
    }


# --------------------------------------------------------------------------
# real-tree assembly (root-parameterized so tests drive synthetic trees)
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_unit_id(unit_id: str) -> dict:
    """d1__<model>__<protocol>__s<seed>__f<fold>__<channel> -> parts."""
    parts = unit_id.split("__")
    if (len(parts) != 6 or parts[0] != "d1" or parts[1] not in MODELS
            or parts[2] not in PROTOCOLS or not parts[3].startswith("s")
            or not parts[4].startswith("f") or parts[5] not in CHANNELS):
        raise ValueError(f"malformed unit_id {unit_id!r}")
    return {"model": parts[1], "protocol": parts[2], "seed": int(parts[3][1:]),
            "fold": int(parts[4][1:]), "channel": parts[5]}


def unit_dir_for(train_root: Path, unit_id: str) -> Path:
    p = parse_unit_id(unit_id)
    return (train_root / "units" / p["channel"] / p["model"] / p["protocol"]
             / f"seed_{p['seed']}" / f"fold_{p['fold']:03d}")


def load_pair_rows(pair_manifest_path: Path) -> list[dict]:
    """sample_index (0-based row order) -> identity row from the frozen pair
    manifest: canonical actor, frozen emotion label index, and the
    channel-specific relative path (record 56 action 3)."""
    out = []
    with open(pair_manifest_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["emotion_name"] not in EMOTION6:
                raise ValueError(f"pair manifest emotion {r['emotion_name']!r} "
                                  "outside frozen EMOTION6")
            out.append({"actor": f"{int(r['actor']):02d}",
                         "emotion_idx": EMOTION6.index(r["emotion_name"]),
                         "path": {"song": r["song_relative_path"],
                                   "speech": r["speech_relative_path"]}})
    return out


def load_plan_index(plan_path: Path) -> dict[str, dict]:
    """unit_id -> {n_outer_test, inner_split_path, inner_split_sha256}."""
    out = {}
    with open(plan_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["unit_id"]] = {"n_outer_test": int(r["n_outer_test"]),
                                   "inner_split_path": r["inner_split_path"],
                                   "inner_split_sha256": r["inner_split_sha256"]}
    return out


def load_outer_test_idx(npz_dir: Path, rel_path: str, expected_sha: str) -> frozenset[int]:
    """Hash-gate the frozen split NPZ, then return its outer_test_idx set."""
    path = npz_dir / rel_path
    observed = sha256_file(path)
    if observed != expected_sha:
        raise RuntimeError(f"split NPZ {rel_path}: sha {observed} != run plan {expected_sha}")
    with np.load(path) as z:
        return frozenset(int(i) for i in z["outer_test_idx"])


def read_unit_predictions(train_root: Path, unit_id: str, expected_sha: str,
                            pair_rows: list[dict],
                            expected_idx: frozenset[int]) -> dict[int, tuple[int, int]]:
    """Hash-gate then parse ONE unit's predictions.csv.

    Fail-closed on: byte-hash drift from the completion manifest, header
    schema drift, identity columns disagreeing with the unit_id, label
    name/index violating the frozen EMOTION6 order, true label / actor /
    relative path disagreeing with the frozen pair manifest, duplicate or
    out-of-range sample_index, and a sample set differing from the frozen
    split NPZ's outer_test_idx. Returns {sample_index: (true_idx, pred_idx)}.
    """
    p = parse_unit_id(unit_id)
    path = unit_dir_for(train_root, unit_id) / "predictions.csv"
    observed = sha256_file(path)
    if observed != expected_sha:
        raise RuntimeError(f"{unit_id}: predictions.csv sha {observed} != "
                            f"completion manifest {expected_sha}")
    out: dict[int, tuple[int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != PRED_HEADER:
            raise RuntimeError(f"{unit_id}: predictions header drift {reader.fieldnames}")
        for row in reader:
            if (row["unit_id"] != unit_id or row["channel"] != p["channel"]
                    or row["model"] != p["model"] or row["protocol"] != p["protocol"]
                    or int(row["seed"]) != p["seed"] or int(row["outer_fold"]) != p["fold"]):
                raise RuntimeError(f"{unit_id}: identity columns disagree in row {row['sample_index']}")
            idx = int(row["sample_index"])
            if not (0 <= idx < len(pair_rows)):
                raise RuntimeError(f"{unit_id}: sample_index {idx} outside pair manifest")
            ref = pair_rows[idx]
            if row["canonical_actor"] != ref["actor"]:
                raise RuntimeError(f"{unit_id}: sample {idx} actor {row['canonical_actor']} "
                                    f"!= pair manifest {ref['actor']}")
            t, pr = int(row["true_label_index"]), int(row["predicted_label_index"])
            for which, li, name in (("true", t, row["true_label_name"]),
                                      ("predicted", pr, row["predicted_label_name"])):
                if not (0 <= li < N_CLASSES) or EMOTION6[li] != name:
                    raise RuntimeError(f"{unit_id}: sample {idx} {which} label {li}/{name} "
                                        "violates frozen EMOTION6 order")
            if t != ref["emotion_idx"]:
                raise RuntimeError(f"{unit_id}: sample {idx} true label {t} != pair "
                                    f"manifest emotion {ref['emotion_idx']}")
            if row["relative_path"] != ref["path"][p["channel"]]:
                raise RuntimeError(f"{unit_id}: sample {idx} path {row['relative_path']!r} "
                                    f"!= pair manifest {ref['path'][p['channel']]!r}")
            if idx in out:
                raise RuntimeError(f"{unit_id}: duplicate sample_index {idx} within unit")
            out[idx] = (t, pr)
    if frozenset(out) != expected_idx:
        raise RuntimeError(f"{unit_id}: sample set != frozen split outer_test_idx "
                            f"({len(out)} rows vs {len(expected_idx)} expected; "
                            f"symmetric diff {len(frozenset(out) ^ expected_idx)})")
    return out


def assemble_model_predictions(train_root: Path, unit_anchors: dict[str, dict],
                                 pair_rows: list[dict], plan_index: dict[str, dict],
                                 npz_dir: Path, model: str) -> dict:
    """Assemble predictions[(channel, protocol, seed)] = {sample_index: (t, p)}
    for one model. Each unit is verified against its frozen split NPZ; a
    duplicate sample_index across the folds of one cell is a hard error;
    completeness is enforced by collect_actor_uars."""
    cells: dict[tuple[str, str, int], dict[int, tuple[int, int]]] = {}
    for unit_id, anchor in sorted(unit_anchors.items()):
        p = parse_unit_id(unit_id)
        if p["model"] != model:
            continue
        plan = plan_index[unit_id]
        expected_idx = load_outer_test_idx(npz_dir, plan["inner_split_path"],
                                             plan["inner_split_sha256"])
        if len(expected_idx) != plan["n_outer_test"]:
            raise RuntimeError(f"{unit_id}: NPZ outer_test_idx size {len(expected_idx)} "
                                f"!= run plan n_outer_test {plan['n_outer_test']}")
        preds = read_unit_predictions(train_root, unit_id, anchor["predictions_sha256"],
                                        pair_rows, expected_idx)
        cell = cells.setdefault((p["channel"], p["protocol"], p["seed"]), {})
        for idx, tp in preds.items():
            if idx in cell:
                raise RuntimeError(f"{model}/{p['channel']}/{p['protocol']}/s{p['seed']}: "
                                    f"sample {idx} predicted by more than one fold")
            cell[idx] = tp
    return cells


def score_tree(train_root: Path, manifest: dict, pair_rows: list[dict],
                plan_index: dict[str, dict], npz_dir: Path) -> dict:
    """Assemble + aggregate every model. Pure given its inputs; no gates."""
    unit_anchors = manifest["units"]
    actors = sorted({r["actor"] for r in pair_rows})
    actors_of_sample = {i: r["actor"] for i, r in enumerate(pair_rows)}
    reports = {}
    for model in MODELS:
        cells = assemble_model_predictions(train_root, unit_anchors, pair_rows,
                                             plan_index, npz_dir, model)
        uars = collect_actor_uars(cells, actors_of_sample)
        reports[model] = aggregate_model(uars, actors)
    return {"actors": actors, "n_samples_per_channel": len(pair_rows),
             "models": reports}


# --------------------------------------------------------------------------
# authorization, claim, atomic output (record 56 actions 1-2)
# --------------------------------------------------------------------------

def validate_unlock(unlock, manifest_sha: str, scorer_sha: str,
                     review_request_record: int, output_path: str) -> None:
    """Structured-unlock validation: exact comparison, everything fail-closed.

    Requires a dict with: completion_manifest_v6_sha256 == the LIVE v6 hash,
    scorer_sha256 == this file's hash, output_path == the unique allowed
    output, prescore_pass_record an int STRICTLY greater than the frozen
    review-request record (the PASS can only postdate the request), and
    non-empty author_authorization / executor strings. Raises on any gap."""
    if not isinstance(unlock, dict):
        raise RuntimeError("unlock is not a JSON object")
    for key in ("completion_manifest_v6_sha256", "scorer_sha256", "output_path",
                 "prescore_pass_record", "author_authorization", "executor"):
        if key not in unlock:
            raise RuntimeError(f"unlock missing field {key!r}")
    if unlock["completion_manifest_v6_sha256"] != manifest_sha:
        raise RuntimeError("unlock v6 sha mismatch")
    if unlock["scorer_sha256"] != scorer_sha:
        raise RuntimeError("unlock scorer sha mismatch")
    if unlock["output_path"] != output_path:
        raise RuntimeError("unlock output path mismatch")
    rec = unlock["prescore_pass_record"]
    if not isinstance(rec, int) or rec <= review_request_record:
        raise RuntimeError(f"unlock prescore_pass_record {rec!r} must be an int > "
                            f"review request record {review_request_record}")
    for key in ("author_authorization", "executor"):
        if not isinstance(unlock[key], str) or not unlock[key].strip():
            raise RuntimeError(f"unlock {key} empty")


def write_all(fd: int, data: bytes) -> None:
    """Loop os.write until every byte is accepted (record 58 action 1).
    A partial write is retried from the boundary; a 0-byte write raises --
    a short write must never be committed as success."""
    view = memoryview(data)
    written = 0
    while written < len(view):
        n = os.write(fd, view[written:])
        if n <= 0:
            raise RuntimeError(f"short write: os.write returned {n} at byte {written}/{len(view)}")
        written += n


def acquire_scoring_claim(claim_path: Path, payload: dict) -> str:
    """O_CREAT|O_EXCL claim BEFORE the first outcome read. An existing claim
    (live or crashed) always refuses -- crashed claims are audit evidence and
    may only be removed manually after an explicit audit note.
    The written bytes are re-read and hash-verified; returns the claim's
    sha256 so the postscore receipt can anchor the claim CONTENT."""
    data = json.dumps(payload, indent=2).encode("utf-8")
    expected = hashlib.sha256(data).hexdigest()
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    # O_BINARY: without it the Windows CRT text mode rewrites \n as \r\n,
    # so the committed bytes would not match the serialized buffer.
    fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                  | getattr(os, "O_BINARY", 0))
    try:
        write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    if claim_path.stat().st_size != len(data) or sha256_file(claim_path) != expected:
        raise RuntimeError("claim content verification failed after write")
    return expected


def atomic_write_json(final_path: Path, obj: dict) -> str:
    """Exclusive same-directory temp file -> write_all -> fsync -> length and
    sha verification against the serialized buffer -> atomic replace.
    A failed verification leaves the temp file as evidence and never touches
    final_path. Returns the committed file's sha256 (== the buffer's)."""
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    expected = hashlib.sha256(data).hexdigest()
    tmp = final_path.with_name(final_path.name + ".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                  | getattr(os, "O_BINARY", 0))
    try:
        write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    if tmp.stat().st_size != len(data) or sha256_file(tmp) != expected:
        raise RuntimeError("temp content verification failed; temp left as evidence")
    os.replace(tmp, final_path)
    return expected


RECEIPT_REQUIRED_FIELDS = ("written_at", "output", "output_sha256", "claim",
                            "claim_sha256", "scorer_sha256", "unlock_sha256",
                            "completion_manifest_v6_sha256")
IDENTITY_FIELDS = ("scorer_sha256", "unlock_sha256", "completion_manifest_v6_sha256")


def verify_postscore_receipt(scoring_dir: Path,
                               manifest_path: Path | None = None,
                               unlock_path: Path | None = None,
                               scorer_path: Path | None = None) -> None:
    """Read-only postscore closure (record 60 action 1). Raises on any gap:

    - every RECEIPT_REQUIRED_FIELDS key must be present;
    - file names are FIXED: output must be d1_scoring_report.json and claim
      must be scoring_claim.json -- substituted names refuse;
    - live re-hashes of the committed output and claim must equal the
      receipt's content hashes;
    - the receipt, the PARSED claim, and the PARSED output must agree on all
      three identity fields (scorer/unlock/v6), and each identity value must
      equal the live hash of the actual scorer file, unlock file, and v6
      manifest.
    The default anchor paths are the real ones; tests inject synthetic paths.
    """
    manifest_path = COMPLETION_V6 if manifest_path is None else manifest_path
    unlock_path = UNLOCK if unlock_path is None else unlock_path
    scorer_path = Path(__file__).resolve() if scorer_path is None else scorer_path
    receipt = json.loads((scoring_dir / "postscore_receipt.json").read_text(encoding="utf-8"))
    for key in RECEIPT_REQUIRED_FIELDS:
        if key not in receipt:
            raise RuntimeError(f"receipt missing required field {key!r}")
    if receipt["output"] != "d1_scoring_report.json":
        raise RuntimeError(f"receipt output name {receipt['output']!r} != fixed "
                            "d1_scoring_report.json")
    if receipt["claim"] != "scoring_claim.json":
        raise RuntimeError(f"receipt claim name {receipt['claim']!r} != fixed "
                            "scoring_claim.json")
    for name_key, sha_key in (("output", "output_sha256"), ("claim", "claim_sha256")):
        path = scoring_dir / receipt[name_key]
        observed = sha256_file(path)
        if observed != receipt[sha_key]:
            raise RuntimeError(f"receipt {sha_key} {receipt[sha_key]} != live {observed} "
                                f"for {path.name}")
    output_doc = json.loads((scoring_dir / receipt["output"]).read_text(encoding="utf-8"))
    claim_doc = json.loads((scoring_dir / receipt["claim"]).read_text(encoding="utf-8"))
    live = {"scorer_sha256": sha256_file(scorer_path),
             "unlock_sha256": sha256_file(unlock_path),
             "completion_manifest_v6_sha256": sha256_file(manifest_path)}
    for field in IDENTITY_FIELDS:
        values = {"receipt": receipt.get(field), "claim": claim_doc.get(field),
                   "output": output_doc.get(field), "live": live[field]}
        if len(set(values.values())) != 1:
            raise RuntimeError(f"identity field {field} disagreement: {values}")


def main() -> int:
    if not COMPLETION_V6.exists():
        print("REFUSED: training/completion_manifest_v6.json absent -- prescore "
              "amendment not frozen (dialogue record 60 action 2).")
        return 1
    if not UNLOCK.exists():
        print("REFUSED: training/SCORING_UNLOCKED.json absent -- Codex pre-score review "
              "has not PASSed. Scoring the real tree is forbidden (assignment section 6).")
        return 1
    manifest = json.loads(COMPLETION_V6.read_text(encoding="utf-8"))
    self_sha = sha256_file(Path(__file__).resolve())
    if self_sha != manifest["scorer"]["score_d1_results.py"]:
        print("REFUSED: this scorer drifted from the sha frozen in completion_manifest_v6.")
        return 1
    try:
        unlock = json.loads(UNLOCK.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"REFUSED: unlock is not valid JSON ({exc}).")
        return 1
    manifest_sha = sha256_file(COMPLETION_V6)
    try:
        validate_unlock(unlock, manifest_sha, self_sha,
                         int(manifest["review_request_record"]), str(SCORING_OUT))
    except RuntimeError as exc:
        print(f"REFUSED: {exc}")
        return 1
    if sha256_file(PAIR_MANIFEST) != manifest["scorer"]["pair_manifest_sha256"]:
        print("REFUSED: pair_manifest.csv drifted from completion_manifest_v6.")
        return 1
    if sha256_file(RUN_PLAN) != manifest["anchors"]["run_plan_sha256"]:
        print("REFUSED: run_plan.csv drifted from completion_manifest_v6.")
        return 1
    if SCORING_OUT.exists() or RECEIPT.exists():
        print("REFUSED: scoring output/receipt already present; scoring is one-shot.")
        return 1
    tz = timezone(timedelta(hours=-4))
    unlock_sha = sha256_file(UNLOCK)
    try:
        claim_sha = acquire_scoring_claim(SCORING_CLAIM, {
            "claimed_at": datetime.now(tz).isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "scorer_sha256": self_sha,
            "unlock_sha256": unlock_sha,
            "completion_manifest_v6_sha256": manifest_sha,
        })
    except FileExistsError:
        print(f"REFUSED: {SCORING_CLAIM} exists -- another scorer ran or crashed; "
              "manual audit required, never silently rescored.")
        return 1
    pair_rows = load_pair_rows(PAIR_MANIFEST)
    plan_index = load_plan_index(RUN_PLAN)
    result = score_tree(TRAIN_ROOT, manifest, pair_rows, plan_index, NPZ_DIR)
    result = {"created_at": datetime.now(tz).isoformat(timespec="seconds"),
               "scorer_sha256": self_sha,
               "completion_manifest_v6_sha256": manifest_sha,
               "unlock_sha256": unlock_sha,
               **result}
    out_sha = atomic_write_json(SCORING_OUT, result)
    atomic_write_json(RECEIPT, {
        "written_at": datetime.now(tz).isoformat(timespec="seconds"),
        "output": SCORING_OUT.name, "output_sha256": out_sha,
        "claim": SCORING_CLAIM.name, "claim_sha256": claim_sha,
        "scorer_sha256": self_sha, "unlock_sha256": unlock_sha,
        "completion_manifest_v6_sha256": manifest_sha,
    })
    verify_postscore_receipt(SCORING_DIR)
    print(f"scoring complete -> {SCORING_OUT} (sha256 {out_sha})")
    for model, rep in result["models"].items():
        print(f"  {model}: {rep['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

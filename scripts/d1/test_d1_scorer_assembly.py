"""Synthetic tests for score_d1_results.py real-tree ASSEMBLY path and the
record-56 gates (structured unlock, atomic one-shot, input identity closure,
seed variability reporting).

Never touches the real training tree: every test builds its own miniature
tree in a temp directory (3 actors x 6 samples, all 4 models, 2 channels,
3 protocols, 3 seeds, plus frozen synthetic split NPZs and a pair manifest)
and drives the root-parameterized helpers directly. The one subprocess test
runs the scorer's gated main() against the real repo and asserts it REFUSES.

Run: python tools/test_d1_scorer_assembly.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import score_d1_results as sc

ACTORS = ("01", "02", "03")
SAMPLES_PER_ACTOR = 6
N_SAMPLES = len(ACTORS) * SAMPLES_PER_ACTOR
FOLDS = {"random": 2, "groupkfold": 2, "loso": 3}

PASSED = 0
FAILED = []


def check(name: str, fn) -> None:
    global PASSED
    try:
        fn()
        PASSED += 1
        print(f"  PASS {name}")
    except Exception as exc:  # noqa: BLE001 - report and continue
        FAILED.append(name)
        print(f"  FAIL {name}: {exc}")


def expect_raises(fn, needle: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        if needle not in str(exc):
            raise AssertionError(f"raised, but message lacks {needle!r}: {exc}") from exc
        return
    raise AssertionError(f"expected an exception containing {needle!r}")


def sample_actor(idx: int) -> str:
    return ACTORS[idx // SAMPLES_PER_ACTOR]


def sample_fold(protocol: str, idx: int) -> int:
    if protocol == "loso":
        return idx // SAMPLES_PER_ACTOR          # one fold per actor
    return idx % FOLDS[protocol]                   # round-robin


def true_label(idx: int) -> int:
    return idx % sc.N_CLASSES


def rel_path(channel: str, idx: int) -> str:
    return f"{channel}_{idx}.wav"


def write_pair_manifest(root: Path) -> Path:
    path = root / "pair_manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["actor", "emotion_code", "emotion_name", "intensity",
                     "statement", "repetition", "song_relative_path", "song_sha256",
                     "song_file_size_bytes", "speech_relative_path", "speech_sha256",
                     "speech_file_size_bytes"])
        for idx in range(N_SAMPLES):
            w.writerow([sample_actor(idx), 1, sc.EMOTION6[true_label(idx)], 1, 1, 1,
                         rel_path("song", idx), "0" * 64, 1,
                         rel_path("speech", idx), "0" * 64, 1])
    return path


def default_rows(unit_id: str, wrong_pred: bool):
    p = sc.parse_unit_id(unit_id)
    rows = []
    for idx in range(N_SAMPLES):
        if sample_fold(p["protocol"], idx) != p["fold"]:
            continue
        t = true_label(idx)
        pred = (t + 1) % sc.N_CLASSES if wrong_pred else t
        rows.append([unit_id, p["channel"], p["model"], p["protocol"], p["seed"],
                      p["fold"], idx, sample_actor(idx), rel_path(p["channel"], idx),
                      t, sc.EMOTION6[t], pred, sc.EMOTION6[pred],
                      0, 0, 0, 0, 0, 0])
    return rows


def write_unit(root: Path, unit_id: str, wrong_pred: bool = False,
                rows_override=None) -> str:
    unit_dir = sc.unit_dir_for(root, unit_id)
    unit_dir.mkdir(parents=True, exist_ok=True)
    path = unit_dir / "predictions.csv"
    rows = default_rows(unit_id, wrong_pred) if rows_override is None else rows_override
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(sc.PRED_HEADER)
        w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_npzs(root: Path) -> Path:
    npz_dir = root / "npz"
    for protocol, n_folds in FOLDS.items():
        (npz_dir / protocol).mkdir(parents=True, exist_ok=True)
        for fold in range(n_folds):
            idx = np.asarray([i for i in range(N_SAMPLES)
                                if sample_fold(protocol, i) == fold], dtype=np.int64)
            np.savez(npz_dir / protocol / f"fold_{fold:03d}.npz", outer_test_idx=idx)
    return npz_dir


def build_tree(root: Path, wrong_cells: set[tuple[str, str, str]] = frozenset(),
                skip_cells: set[tuple[str, str, str, int]] = frozenset()):
    """Returns (manifest, plan_index, npz_dir)."""
    npz_dir = write_npzs(root)
    anchors, plan_index = {}, {}
    for model in sc.MODELS:
        for channel in sc.CHANNELS:
            for protocol in sc.PROTOCOLS:
                for seed in sc.SEEDS:
                    if (model, channel, protocol, seed) in skip_cells:
                        continue
                    for fold in range(FOLDS[protocol]):
                        unit_id = f"d1__{model}__{protocol}__s{seed}__f{fold:03d}__{channel}"
                        sha = write_unit(root, unit_id,
                                          wrong_pred=(model, channel, protocol) in wrong_cells)
                        anchors[unit_id] = {"predictions_sha256": sha}
                        split_rel = f"{protocol}/fold_{fold:03d}.npz"
                        plan_index[unit_id] = {
                            "n_outer_test": sum(1 for i in range(N_SAMPLES)
                                                  if sample_fold(protocol, i) == fold),
                            "inner_split_path": split_rel,
                            "inner_split_sha256": sc.sha256_file(npz_dir / split_rel),
                        }
    return {"units": anchors}, plan_index, npz_dir


def main() -> int:
    print("test_d1_scorer_assembly (synthetic only)")
    tmp = Path(tempfile.mkdtemp(prefix="d1_scorer_asm_"))

    pair_path = write_pair_manifest(tmp)
    pair_rows = sc.load_pair_rows(pair_path)

    def t_pair_rows():
        assert len(pair_rows) == N_SAMPLES
        assert pair_rows[0]["actor"] == "01" and pair_rows[N_SAMPLES - 1]["actor"] == "03"
        assert pair_rows[7]["emotion_idx"] == true_label(7)
        assert pair_rows[7]["path"]["song"] == rel_path("song", 7)
    check("pair manifest -> actor/emotion/path rows", t_pair_rows)

    perfect_root = tmp / "perfect"
    manifest, plan_index, npz_dir = build_tree(perfect_root)

    def t_perfect_complete():
        result = sc.score_tree(perfect_root, manifest, pair_rows, plan_index, npz_dir)
        assert result["actors"] == list(ACTORS)
        for model in sc.MODELS:
            rep = result["models"][model]
            assert rep["status"] == "complete", rep
            for key, val in rep["point_estimates"].items():
                assert abs(val) < 1e-12, (model, key, val)
            for key in sc.HEADLINE_KEYS:
                sw = rep["seedwise_panel_estimates"][key]
                assert len(sw) == 3 and all(abs(v) < 1e-12 for v in sw)
                assert abs(rep["seed_mean_sd"][key]["sd"]) < 1e-12
    check("perfect tree -> complete, premiums 0, seedwise 0", t_perfect_complete)

    def t_determinism():
        a = json.dumps(sc.score_tree(perfect_root, manifest, pair_rows, plan_index, npz_dir),
                        sort_keys=True)
        b = json.dumps(sc.score_tree(perfect_root, manifest, pair_rows, plan_index, npz_dir),
                        sort_keys=True)
        assert a == b
    check("assembly + scoring deterministic", t_determinism)

    wrong_root = tmp / "wrong_random_speech"
    w_manifest, w_plan, w_npz = build_tree(wrong_root,
                                             wrong_cells={("fno", "speech", "random")})

    def t_wrong_cell_math():
        result = sc.score_tree(wrong_root, w_manifest, pair_rows, w_plan, w_npz)
        rep = result["models"]["fno"]
        assert rep["status"] == "complete"
        pe = rep["point_estimates"]
        assert abs(pe["P_RG_speech"] - (-1.0)) < 1e-12, pe
        assert abs(pe["P_RG_song"]) < 1e-12, pe
        assert abs(pe["D_mode_RG"] - 1.0) < 1e-12, pe
        cnn = result["models"]["cnn"]["point_estimates"]
        assert all(abs(v) < 1e-12 for v in cnn.values()), "cnn contaminated"
    check("wrong random/speech cell -> P_RG_speech=-1, D_mode=+1, others untouched", t_wrong_cell_math)

    # ---- seed variability (record 56 action 4) -------------------------
    def t_seed_mean_sd_handcalc():
        uars = {}
        speech_random = {0: 0.7, 1: 0.8, 2: 0.9}   # per-seed UAR, all actors alike
        for actor in ACTORS:
            for seed in sc.SEEDS:
                for channel in sc.CHANNELS:
                    for protocol in sc.PROTOCOLS:
                        val = 0.5
                        if channel == "speech" and protocol == "random":
                            val = speech_random[seed]
                        uars[(actor, channel, protocol, seed)] = val
        rep = sc.aggregate_model(uars, list(ACTORS))
        sw = rep["seedwise_panel_estimates"]["P_RG_speech"]
        assert all(abs(a - b) < 1e-12 for a, b in zip(sw, [0.2, 0.3, 0.4])), sw
        msd = rep["seed_mean_sd"]["P_RG_speech"]
        assert abs(msd["mean"] - 0.3) < 1e-12
        assert abs(msd["sd"] - 0.1) < 1e-12          # sample SD of [.2,.3,.4]
        assert abs(rep["point_estimates"]["P_RG_speech"] - 0.3) < 1e-12
        dsw = rep["seedwise_panel_estimates"]["D_mode_RG"]
        assert all(abs(a - b) < 1e-12 for a, b in zip(dsw, [-0.2, -0.3, -0.4])), dsw
    check("seed mean±SD hand-calc (mean .3, SD .1; D_mode paired per seed)", t_seed_mean_sd_handcalc)

    def t_seed_pairing_mutation():
        base, permuted = {}, {}
        song_random = {0: 0.6, 1: 0.75, 2: 0.9}
        perm = {0: song_random[2], 1: song_random[0], 2: song_random[1]}
        for target, table in ((base, song_random), (permuted, perm)):
            for actor in ACTORS:
                for seed in sc.SEEDS:
                    for channel in sc.CHANNELS:
                        for protocol in sc.PROTOCOLS:
                            val = 0.5
                            if channel == "song" and protocol == "random":
                                val = table[seed]
                            target[(actor, channel, protocol, seed)] = val
        rep_a = sc.aggregate_model(base, list(ACTORS))
        rep_b = sc.aggregate_model(permuted, list(ACTORS))
        assert abs(rep_a["point_estimates"]["D_mode_RG"]
                    - rep_b["point_estimates"]["D_mode_RG"]) < 1e-12
        assert rep_a["seedwise_panel_estimates"]["D_mode_RG"] != \
                rep_b["seedwise_panel_estimates"]["D_mode_RG"]
    check("seed-pairing mutation: means equal, seedwise differs (pairing detected)", t_seed_pairing_mutation)

    # ---- identity closure mutations (record 56 action 3) ---------------
    unit0 = "d1__cnn__random__s0__f000__speech"
    p0 = sc.parse_unit_id(unit0)
    idx0 = next(i for i in range(N_SAMPLES) if sample_fold("random", i) == 0)
    expected0 = sc.load_outer_test_idx(npz_dir, plan_index[unit0]["inner_split_path"],
                                         plan_index[unit0]["inner_split_sha256"])

    def one_row(idx: int, **overrides):
        t = true_label(idx)
        row = {"unit_id": unit0, "channel": p0["channel"], "model": p0["model"],
                "protocol": p0["protocol"], "seed": p0["seed"], "fold": p0["fold"],
                "idx": idx, "actor": sample_actor(idx), "path": rel_path("speech", idx),
                "t": t, "t_name": sc.EMOTION6[t], "p": t, "p_name": sc.EMOTION6[t]}
        row.update(overrides)
        return [row["unit_id"], row["channel"], row["model"], row["protocol"],
                 row["seed"], row["fold"], row["idx"], row["actor"], row["path"],
                 row["t"], row["t_name"], row["p"], row["p_name"], 0, 0, 0, 0, 0, 0]

    def t_wrong_true_label():
        root = tmp / "bad_true_label"
        wrong_t = (true_label(idx0) + 1) % sc.N_CLASSES
        sha = write_unit(root, unit0, rows_override=[
            one_row(idx0, t=wrong_t, t_name=sc.EMOTION6[wrong_t])])
        expect_raises(lambda: sc.read_unit_predictions(root, unit0, sha, pair_rows,
                                                          expected0),
                       "pair manifest emotion")
    check("true label != pair manifest emotion -> refusal", t_wrong_true_label)

    def t_wrong_path():
        root = tmp / "bad_path"
        sha = write_unit(root, unit0, rows_override=[one_row(idx0, path="evil.wav")])
        expect_raises(lambda: sc.read_unit_predictions(root, unit0, sha, pair_rows,
                                                          expected0),
                       "pair manifest")
    check("relative path != pair manifest -> refusal", t_wrong_path)

    def t_wrong_fold_sample_set():
        root = tmp / "bad_fold"
        rows = default_rows(unit0, wrong_pred=False)
        swap_in = next(i for i in range(N_SAMPLES) if sample_fold("random", i) == 1)
        rows[0] = one_row(swap_in)                      # count preserved, set wrong
        sha = write_unit(root, unit0, rows_override=rows)
        expect_raises(lambda: sc.read_unit_predictions(root, unit0, sha, pair_rows,
                                                          expected0),
                       "sample set")
    check("sample from another fold (count preserved) -> sample-set refusal", t_wrong_fold_sample_set)

    def t_npz_tamper():
        root = tmp / "npz_tamper"
        m2, plan2, npz2 = build_tree(root)
        target = npz2 / plan2[unit0]["inner_split_path"]
        target.write_bytes(target.read_bytes() + b"#")
        expect_raises(lambda: sc.assemble_model_predictions(root, m2["units"], pair_rows,
                                                               plan2, npz2, "cnn"),
                       "split NPZ")
    check("tampered split NPZ -> hash refusal", t_npz_tamper)

    def t_tamper_predictions():
        path = sc.unit_dir_for(perfect_root, unit0) / "predictions.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"#")
            expect_raises(lambda: sc.read_unit_predictions(
                perfect_root, unit0, manifest["units"][unit0]["predictions_sha256"],
                pair_rows, expected0), "sha")
        finally:
            path.write_bytes(original)
    check("tampered predictions bytes -> hash refusal", t_tamper_predictions)

    def t_missing_cell():
        root = tmp / "skip_cell"
        m2, plan2, npz2 = build_tree(root, skip_cells={("cnn", "song", "loso", 2)})
        result = sc.score_tree(root, m2, pair_rows, plan2, npz2)
        rep = result["models"]["cnn"]
        assert rep["status"] == "incomplete/not_conclusive"
        assert any("song/loso/s2" in m for m in rep["missing"])
        assert result["models"]["fno"]["status"] == "complete"
    check("missing cell -> incomplete/not_conclusive, others complete", t_missing_cell)

    def t_partial_cell():
        root = tmp / "partial_cell"
        m2, plan2, npz2 = build_tree(root)
        removed = "d1__cnn__random__s1__f001__speech"
        (sc.unit_dir_for(root, removed) / "predictions.csv").unlink()
        del m2["units"][removed]
        del plan2[removed]
        expect_raises(lambda: sc.score_tree(root, m2, pair_rows, plan2, npz2),
                       "OOF exactly-once violated")
    check("partial cell (one fold gone) -> OOF exactly-once error", t_partial_cell)

    def t_duplicate_across_folds():
        root = tmp / "dup"
        m2, plan2, npz2 = build_tree(root)
        unit_b = "d1__cnn__random__s0__f001__speech"
        rows = default_rows(unit_b, wrong_pred=False) + [one_row(
            idx0, unit_id=unit_b, fold=1)]
        m2["units"][unit_b] = {"predictions_sha256": write_unit(root, unit_b,
                                                                    rows_override=rows)}
        expect_raises(lambda: sc.assemble_model_predictions(root, m2["units"], pair_rows,
                                                               plan2, npz2, "cnn"),
                       "sample set")
    check("smuggled duplicate across folds -> refusal (set check)", t_duplicate_across_folds)

    # ---- structured unlock (record 56 action 1) ------------------------
    V4_SHA, SC_SHA, REQ, OUT = "a" * 64, "b" * 64, 57, r"E:\out\report.json"

    def good_unlock():
        return {"completion_manifest_v6_sha256": V4_SHA, "scorer_sha256": SC_SHA,
                 "output_path": OUT, "prescore_pass_record": 58,
                 "author_authorization": "author approved in session", "executor": "Claude"}

    def t_unlock_valid():
        sc.validate_unlock(good_unlock(), V4_SHA, SC_SHA, REQ, OUT)
    check("structured unlock: valid case passes", t_unlock_valid)

    def t_unlock_mutations():
        expect_raises(lambda: sc.validate_unlock("not a dict", V4_SHA, SC_SHA, REQ, OUT),
                       "not a JSON object")
        expect_raises(lambda: sc.validate_unlock({}, V4_SHA, SC_SHA, REQ, OUT), "missing field")
        u = good_unlock(); u["completion_manifest_v6_sha256"] = "c" * 64
        expect_raises(lambda: sc.validate_unlock(u, V4_SHA, SC_SHA, REQ, OUT), "v6 sha")
        u = good_unlock(); u["scorer_sha256"] = "c" * 64
        expect_raises(lambda: sc.validate_unlock(u, V4_SHA, SC_SHA, REQ, OUT), "scorer sha")
        u = good_unlock(); u["output_path"] = r"E:\elsewhere.json"
        expect_raises(lambda: sc.validate_unlock(u, V4_SHA, SC_SHA, REQ, OUT), "output path")
        u = good_unlock(); u["prescore_pass_record"] = 57
        expect_raises(lambda: sc.validate_unlock(u, V4_SHA, SC_SHA, REQ, OUT), "must be an int >")
        u = good_unlock(); u["prescore_pass_record"] = "58"
        expect_raises(lambda: sc.validate_unlock(u, V4_SHA, SC_SHA, REQ, OUT), "must be an int >")
        u = good_unlock(); u["author_authorization"] = "  "
        expect_raises(lambda: sc.validate_unlock(u, V4_SHA, SC_SHA, REQ, OUT), "empty")
    check("structured unlock: 8 mutations all refuse", t_unlock_mutations)

    # ---- atomic one-shot (record 56 action 2) --------------------------
    def t_claim_exactly_one():
        claim = tmp / "claims" / "scoring_claim.json"
        results = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            try:
                sc.acquire_scoring_claim(claim, {"who": threading.get_ident()})
                results.append("won")
            except FileExistsError:
                results.append("refused")
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results) == ["refused", "won"], results
        assert claim.exists()
        expect_raises(lambda: sc.acquire_scoring_claim(claim, {}), "")
    check("claim race: exactly one winner; existing claim always refuses", t_claim_exactly_one)

    def t_atomic_write_and_crash_trace():
        out = tmp / "atomic" / "report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        sha = sc.atomic_write_json(out, {"x": 1})
        assert out.exists() and sha == sc.sha256_file(out)
        assert not out.with_name(out.name + ".tmp").exists()
        crashed = out.with_name(out.name + ".tmp")
        crashed.write_text("partial", encoding="utf-8")   # simulate mid-write crash
        expect_raises(lambda: sc.atomic_write_json(out, {"x": 2}), "")
        assert json.loads(out.read_text(encoding="utf-8")) == {"x": 1}
    check("atomic write: fsync+replace, no tmp residue; crashed tmp blocks rerun", t_atomic_write_and_crash_trace)

    # ---- short-write injection (record 58 action 1) --------------------
    class ChunkedWriter:
        """Patch os.write inside the scorer to accept at most k bytes per
        call -- a correct write_all loops to completion."""
        def __init__(self, k: int):
            self.k, self.real = k, sc.os.write
        def __enter__(self):
            sc.os.write = lambda fd, data: self.real(fd, bytes(data)[: self.k])
            return self
        def __exit__(self, *exc):
            sc.os.write = self.real

    class ZeroWriter:
        """Patch os.write to accept nothing -- write_all must raise, and no
        partial file may ever be committed as the final output."""
        def __init__(self):
            self.real = sc.os.write
        def __enter__(self):
            sc.os.write = lambda fd, data: 0
            return self
        def __exit__(self, *exc):
            sc.os.write = self.real

    def t_short_write_output():
        out = tmp / "shortw" / "report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": {"cnn": {"status": "complete", "filler": "z" * 5000}}}
        with ChunkedWriter(18):
            sha = sc.atomic_write_json(out, payload)
        assert json.loads(out.read_text(encoding="utf-8")) == payload
        assert sha == sc.sha256_file(out)
    check("output short-write (18B/call): write_all loops, full JSON committed", t_short_write_output)

    def t_short_write_claim():
        claim = tmp / "shortw" / "scoring_claim.json"
        with ChunkedWriter(7):
            sha = sc.acquire_scoring_claim(claim, {"pid": 1, "note": "y" * 500})
        assert sha == sc.sha256_file(claim)
        assert json.loads(claim.read_text(encoding="utf-8"))["note"] == "y" * 500
    check("claim short-write (7B/call): full content committed and hash-verified", t_short_write_claim)

    def t_zero_write_fail_closed():
        out = tmp / "zerow" / "report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        sc.atomic_write_json(out, {"x": 1})
        with ZeroWriter():
            expect_raises(lambda: sc.atomic_write_json(
                tmp / "zerow" / "other.json", {"x": 2}), "short write")
            expect_raises(lambda: sc.acquire_scoring_claim(
                tmp / "zerow" / "claim.json", {"x": 3}), "short write")
        assert json.loads(out.read_text(encoding="utf-8")) == {"x": 1}
        assert not (tmp / "zerow" / "other.json").exists()
    check("0-byte write: claim and output both fail-closed, nothing committed", t_zero_write_fail_closed)

    # ---- postscore receipt identity closure (record 60 action 1) -------
    def make_closure(sdir: Path):
        """Build a full synthetic postscore closure: fake v6 manifest, fake
        unlock, fake scorer file, claim + output carrying the identity
        fields, and a complete receipt. Returns the anchor paths."""
        sdir.mkdir(parents=True, exist_ok=True)
        anchors = {"manifest": sdir / "fake_v6.json", "unlock": sdir / "fake_unlock.json",
                    "scorer": sdir / "fake_scorer.py"}
        anchors["manifest"].write_text('{"fake": "v6"}', encoding="utf-8")
        anchors["unlock"].write_text('{"fake": "unlock"}', encoding="utf-8")
        anchors["scorer"].write_text("# fake scorer", encoding="utf-8")
        ids = {"scorer_sha256": sc.sha256_file(anchors["scorer"]),
                "unlock_sha256": sc.sha256_file(anchors["unlock"]),
                "completion_manifest_v6_sha256": sc.sha256_file(anchors["manifest"])}
        claim_sha = sc.acquire_scoring_claim(sdir / "scoring_claim.json",
                                               {"pid": 42, **ids})
        out_sha = sc.atomic_write_json(sdir / "d1_scoring_report.json",
                                         {"ok": True, **ids})
        sc.atomic_write_json(sdir / "postscore_receipt.json", {
            "written_at": "t", "output": "d1_scoring_report.json",
            "output_sha256": out_sha, "claim": "scoring_claim.json",
            "claim_sha256": claim_sha, **ids})
        return anchors

    def closure_verify(sdir: Path, anchors: dict):
        return sc.verify_postscore_receipt(sdir, manifest_path=anchors["manifest"],
                                             unlock_path=anchors["unlock"],
                                             scorer_path=anchors["scorer"])

    def t_receipt_clean_closure():
        sdir = tmp / "postscore_clean"
        anchors = make_closure(sdir)
        closure_verify(sdir, anchors)
    check("postscore closure: clean case passes", t_receipt_clean_closure)

    def rebuild_receipt(sdir: Path, mutate):
        receipt_path = sdir / "postscore_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        mutate(receipt)
        receipt_path.unlink()
        sc.atomic_write_json(receipt_path, receipt)

    def t_receipt_identity_mutations():
        for field in ("scorer_sha256", "unlock_sha256", "completion_manifest_v6_sha256"):
            sdir = tmp / f"postscore_wrong_{field}"
            anchors = make_closure(sdir)
            def set_wrong(r, f=field):
                r[f] = "WRONG"
            rebuild_receipt(sdir, set_wrong)
            expect_raises(lambda: closure_verify(sdir, anchors), "disagreement")
        sdir = tmp / "postscore_all_wrong"
        anchors = make_closure(sdir)
        def all_wrong(r):
            for f in ("scorer_sha256", "unlock_sha256", "completion_manifest_v6_sha256"):
                r[f] = "WRONG"
        rebuild_receipt(sdir, all_wrong)
        expect_raises(lambda: closure_verify(sdir, anchors), "disagreement")
    check("receipt identity fields wrong (each + all three) -> refusal", t_receipt_identity_mutations)

    def t_receipt_missing_fields_and_names():
        sdir = tmp / "postscore_missing"
        anchors = make_closure(sdir)
        def drop(r):
            del r["completion_manifest_v6_sha256"]
        rebuild_receipt(sdir, drop)
        expect_raises(lambda: closure_verify(sdir, anchors), "missing required field")
        sdir = tmp / "postscore_rename"
        anchors = make_closure(sdir)
        (sdir / "d1_scoring_report.json").rename(sdir / "other.json")
        def rename(r):
            r["output"] = "other.json"
        rebuild_receipt(sdir, rename)
        expect_raises(lambda: closure_verify(sdir, anchors), "fixed")
    check("receipt missing field / substituted output name -> refusal", t_receipt_missing_fields_and_names)

    def t_receipt_object_mutations():
        sdir = tmp / "postscore_claim_flip"
        anchors = make_closure(sdir)
        claim_path = sdir / "scoring_claim.json"
        claim_path.write_text(claim_path.read_text(encoding="utf-8") + " ",
                                encoding="utf-8")
        expect_raises(lambda: closure_verify(sdir, anchors), "claim_sha256")
        sdir = tmp / "postscore_output_flip"
        anchors = make_closure(sdir)
        out_path = sdir / "d1_scoring_report.json"
        out_path.write_text(out_path.read_text(encoding="utf-8") + " ",
                              encoding="utf-8")
        expect_raises(lambda: closure_verify(sdir, anchors), "output_sha256")
        sdir = tmp / "postscore_unlock_swap"
        anchors = make_closure(sdir)
        anchors["unlock"].write_text('{"fake": "another unlock"}', encoding="utf-8")
        expect_raises(lambda: closure_verify(sdir, anchors), "disagreement")
        sdir = tmp / "postscore_claim_lies"
        anchors = make_closure(sdir)
        claim_path = sdir / "scoring_claim.json"
        doc = json.loads(claim_path.read_text(encoding="utf-8"))
        doc["unlock_sha256"] = "WRONG"
        claim_path.write_text(json.dumps(doc), encoding="utf-8")
        expect_raises(lambda: closure_verify(sdir, anchors), "")
    check("1-byte claim/output change, live unlock swap, lying claim -> all refused", t_receipt_object_mutations)

    def t_real_tree_gate():
        proc = subprocess.run([sys.executable, str(TOOLS_DIR / "score_d1_results.py")],
                                capture_output=True, text=True, timeout=120)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "REFUSED" in proc.stdout, proc.stdout
    check("real tree: gated main refuses (subprocess exit 1)", t_real_tree_gate)

    print(f"{PASSED} passed, {len(FAILED)} failed{': ' + ', '.join(FAILED) if FAILED else ''}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

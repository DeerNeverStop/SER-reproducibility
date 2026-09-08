"""P4-D1 training runner: RAVDESS Speech-Song matched protocol study.

Usage (SER environment):

    python tools/run_d1_training.py prepare
    python tools/run_d1_training.py smoke
    python tools/run_d1_training.py run
    python tools/run_d1_training.py status

Gate-8 discipline -- this file reuses P1's frozen training code by IMPORT and
implements no training math of its own:
  set_reproducible_seed, make_train_loader (-> TrainDataset -> SPEC_AUGMENT,
  fit-subset-only augmentation), make_eval_loader (no augmentation),
  class_balanced_criterion, train_one_fit (checkpoint = strict min validation
  loss, patience 15, max 100 epochs, AdamW + cosine), and
  evaluate_outer_test_once (exactly one pass) are all imported from
  run_p1_protocol_premium.py. Normalization is per-sample z-score baked into
  the cache by p1_protocol_core.compute_logmel at cache-build time.

P5 lesson applied: the seed is set BEFORE model construction in this file
(set_reproducible_seed(train_seed) -> BUILD_MODEL), matching P1's formal path;
both channels of a matched unit share one channel-independent train_seed, and
their initial state hashes are asserted equal after the pair completes
(fail-closed halt file on mismatch).

Anchoring: consumes ONLY run_plan.csv rows (which pin NPZ/caches/pair-manifest
/v2-manifest hashes); every artifact is re-hashed before use. No split or
population data is regenerated here.

Budget: D1 charged-GPU hard cap 4.0 h; hard stop 2026-08-22 23:59:59 UTC-04.
Matched-unit ordering: the speech and song fits of one unit run back-to-back
before the next unit starts, so a hard stop cannot strand a half-matched grid.

No outer-test aggregate is computed or printed; per-utterance logits go into
immutable unit directories, scoring happens in a later gated stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)

import numpy as np
import torch

import p1_protocol_core as core
import run_p1_protocol_premium as p1

RESULT_ROOT = Path(r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched")
TRAIN_ROOT = RESULT_ROOT / "training"
PLAN_PATH = RESULT_ROOT / "run_plan.csv"
V2_MANIFEST = RESULT_ROOT / "split_v2" / "split_manifest_v2.json"
PAIR_MANIFEST = RESULT_ROOT / "pair_manifest.csv"
NPZ_DIR = Path(r"E:\科研\claudework\07-ser-repro-protocol-audit\work\d1_splits")
CACHE = {
    "song": Path(r"E:\claudework_data\ICASSP2027-corpora\ravdess-song\cache\d1_logmel_song.npz"),
    "speech": Path(r"E:\claudework_data\ICASSP2027-corpora\ravdess-speech-common6\cache\d1_logmel_speech.npz"),
}
RUNNER_PATH = Path(__file__).resolve()

EMOTION6 = ("neutral", "calm", "happy", "sad", "angry", "fearful")
N_CLASSES = 6
GPU_SECONDS_CAP = 4.0 * 3600  # D1 charged-GPU hard cap (assignment section 4)
HARD_STOP = datetime(2026, 8, 22, 23, 59, 59, tzinfo=timezone(timedelta(hours=-4)))
MAX_ATTEMPTS = 2  # per unit, same config; P1 engineering-lock retry semantics

# Substitute anchors backing the "P1 prereg prefix check inapplicable" ruling
# (record 44 action 3): D1 does not merely trust "imported from P1" -- it pins
# the P1 official runner file, the frozen front-end code hash, and (at prepare
# and run time) re-derives candidate_configs to compare against
# EXPECTED_MODEL_CONFIGS. Values match P1_ENGINEERING_LOCK.md.
EXPECTED_P1_RUNNER_SHA256 = "8cf2232854919eacd676c860ce8ff990ab74e2fa69be2f2826db645cd17b2cbc"
EXPECTED_FRONTEND_CODE_HASH = "33bc3b5e4b6a7a9693c3ffa41d63166bb9ed827e662ff83ee747cfcd9a883d61"


class BudgetStop(Exception):
    """Raised at an epoch boundary when the 4 GPU-h cap or the 8/22 hard stop
    is reached. Training stops immediately; the affected cell is recorded as
    incomplete/not_conclusive, never silently continued."""


def ensure_ser_symbols_loaded_d1() -> dict:
    """D1's own SER hash gate + symbol import.

    P1's `ensure_ser_symbols_loaded` calls `assert_frozen_inputs`, which also
    checks the frozen prefix of P1_PREREGISTRATION.md. That prefix check now
    fails for a DOCUMENTED, non-substantive reason: the 2026-08-12 workspace
    migration rewrote two read-only corpus paths inside the frozen region
    (P1_PREREGISTRATION.md deviation log, 2026-08-12 17:01 entry -- bytes
    verified identical via symlink; P1 results unaffected). D1 therefore
    verifies exactly the parts that protect THIS experiment -- the SER commit
    and the five frozen model-code file hashes -- and records the P1-prereg
    prefix check as inapplicable rather than silently skipping it.
    Fail-closed on any SER commit/file-hash mismatch.
    """
    observed = {
        "ser_commit": core.git_head(core.SER_ROOT),
        "ser_file_sha256": {name: core.sha256_file(core.SER_ROOT / name)
                             for name in core.SER_FILE_SHA256},
        "p1_prereg_prefix_check": "inapplicable_for_d1: documented 2026-08-12 migration "
                                    "path edit inside frozen prefix (see P1_PREREGISTRATION.md "
                                    "deviation log); SER code identity verified independently",
    }
    errors = []
    if observed["ser_commit"] != core.SER_COMMIT:
        errors.append("SER commit drift")
    for name, expected in core.SER_FILE_SHA256.items():
        if observed["ser_file_sha256"][name] != expected:
            errors.append(f"SER file hash drift: {name}")
    if errors:
        raise RuntimeError("; ".join(errors))
    if p1.BUILD_MODEL is None:
        sys.path.insert(0, str(core.SER_ROOT))
        try:
            import advanced_models
            import fno_data
        finally:
            if sys.path and sys.path[0] == str(core.SER_ROOT):
                sys.path.pop(0)
        for name, module in (("advanced_models", advanced_models), ("fno_data", fno_data)):
            if Path(module.__file__).resolve() != (core.SER_ROOT / f"{name}.py").resolve():
                raise RuntimeError(f"unexpected module source for {name}")
        p1.BUILD_MODEL = advanced_models.build_neural_model
        p1.PARAMETER_COUNT = advanced_models.parameter_count
        p1.SPEC_AUGMENT = fno_data.spec_augment
    return observed


def sha256_state_dict(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_plan() -> list[dict[str, str]]:
    rows = core.read_csv(PLAN_PATH)
    if len(rows) != 792:
        raise RuntimeError(f"run plan rows {len(rows)} != 792")
    return rows


def load_labels() -> np.ndarray:
    pair_rows = core.read_csv(PAIR_MANIFEST)
    if len(pair_rows) != 1012:
        raise RuntimeError("pair_manifest rows != 1012")
    y = np.asarray([int(r["emotion_code"]) - 1 for r in pair_rows], dtype=np.int64)
    if not ((0 <= y) & (y < N_CLASSES)).all():
        raise RuntimeError("label out of range")
    return y


def load_pair_rows() -> list[dict[str, str]]:
    return core.read_csv(PAIR_MANIFEST)


def verify_anchors(rows: list[dict[str, str]]) -> dict:
    """Re-hash every anchored artifact the plan references. Fail-closed."""
    v2_sha = core.sha256_file(V2_MANIFEST)
    pair_sha = core.sha256_file(PAIR_MANIFEST)
    cache_sha = {ch: core.sha256_file(path) for ch, path in CACHE.items()}
    npz_seen: dict[str, str] = {}
    for row in rows:
        if row["split_manifest_v2_sha256"] != v2_sha:
            raise RuntimeError("v2 manifest hash drift vs run plan")
        if row["pair_manifest_sha256"] != pair_sha:
            raise RuntimeError("pair manifest hash drift vs run plan")
        if row["cache_sha256"] != cache_sha[row["channel"]]:
            raise RuntimeError(f"cache hash drift vs run plan ({row['channel']})")
        rel = row["inner_split_path"]
        if rel not in npz_seen:
            npz_seen[rel] = core.sha256_file(NPZ_DIR / rel.replace("/", os.sep))
        if npz_seen[rel] != row["inner_split_sha256"]:
            raise RuntimeError(f"NPZ hash drift vs run plan: {rel}")
    return {"v2_manifest_sha256": v2_sha, "pair_manifest_sha256": pair_sha,
            "cache_sha256": cache_sha, "n_npz_verified": len(npz_seen)}


def prepared_path() -> Path:
    return TRAIN_ROOT / "d1_prepared.json"


def ledger_path() -> Path:
    return TRAIN_ROOT / "progress_ledger.csv"


def halt_path() -> Path:
    return TRAIN_ROOT / "HALT_PAIRED_INIT_MISMATCH.json"


def unit_dir(row: dict[str, str]) -> Path:
    return (TRAIN_ROOT / "units" / row["channel"] / row["model"] / row["protocol"]
            / f"seed_{int(row['seed'])}" / f"fold_{int(row['outer_fold']):03d}")


def unit_status(row: dict[str, str]) -> str | None:
    p = unit_dir(row) / "run.json"
    if not p.exists():
        return None
    try:
        return str(json.loads(p.read_text(encoding="utf-8"))["status"])
    except Exception:
        return None


def attempt_dir() -> Path:
    return TRAIN_ROOT / "attempt_ledger"


def attempt_files_for(unit_id: str) -> list[Path]:
    if not attempt_dir().exists():
        return []
    return sorted(attempt_dir().glob(f"{unit_id}__attempt_*.json"))


def read_attempt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_excl_json(path: Path, payload: dict) -> None:
    """O_CREAT|O_EXCL creation -- a concurrent creator of the same path gets
    FileExistsError instead of silently clobbering (record 46 action 2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def run_lock_path() -> Path:
    return TRAIN_ROOT / "run.lock"


def acquire_run_lock() -> Path:
    """Whole-run exclusive lock (record 46 action 2). A live OR stale lock
    refuses the second runner; a stale lock may only be removed manually
    after an explicit audit note -- never auto-broken."""
    lock = run_lock_path()
    try:
        create_excl_json(lock, {"pid": os.getpid(), "token": uuid.uuid4().hex,
                                  "created_at": core.now_iso()})
    except FileExistsError:
        raise RuntimeError(
            f"run lock present ({lock}): another runner is live, or a stale lock "
            "needs explicit audited manual removal. Refusing to start.")
    return lock


def release_run_lock(lock: Path) -> None:
    try:
        os.unlink(lock)
    except FileNotFoundError:
        pass


def open_attempt(unit_id: str) -> Path:
    """Atomically open a per-attempt ledger entry BEFORE any GPU compute.

    Fail-closed rules (P1 engineering-lock semantics, records 44/46):
      - at most MAX_ATTEMPTS attempts per unit;
      - if any prior attempt reached the outer-test phase without a committed
        unit directory, retraining is permanently refused (test-once rule);
      - the attempt file itself is created O_EXCL, so a concurrent process
        racing for the same attempt number loses with an error instead of
        silently sharing/overwriting the ledger entry.
    """
    prior = [read_attempt(p) for p in attempt_files_for(unit_id)]
    for att in prior:
        if att.get("phase") in ("outer_test", "commit") and att["status"] != "success":
            raise RuntimeError(
                f"test-once violation guard: {unit_id} has a non-committed attempt that "
                "already accessed the outer test; retraining is forbidden, unit is lost")
    if len(prior) >= MAX_ATTEMPTS:
        raise RuntimeError(f"attempt cap: {unit_id} already has {len(prior)} attempts")
    attempt_id = len(prior) + 1
    path = attempt_dir() / f"{unit_id}__attempt_{attempt_id:02d}.json"
    try:
        create_excl_json(path, {
            "unit_id": unit_id, "attempt": attempt_id, "status": "running",
            "phase": "training", "started_at": core.now_iso(),
            "heartbeat_at": core.now_iso(), "elapsed_seconds": 0.0,
        })
    except FileExistsError:
        raise RuntimeError(
            f"attempt race lost (fail-closed): {path.name} was created concurrently")
    return path


def update_attempt(path: Path, **fields) -> None:
    record = read_attempt(path)
    record.update(fields)
    record["heartbeat_at"] = core.now_iso()
    core.atomic_write_json(path, record)


def reconcile_stale_attempts() -> list[str]:
    """On startup, close any 'running' attempt left by a crashed process.

    Conservative charging (record 46 action 1, P1 semantics): the closed
    attempt is charged max(last-heartbeat elapsed, wall clock from started_at
    to reconciliation time) -- the tail between the last heartbeat and the
    crash/kill is never free. If the attempt had reached the outer-test
    phase, the unit is permanently lost (test-once, enforced by open_attempt
    and the loss ledger)."""
    notes = []
    for path in sorted(attempt_dir().glob("*.json")) if attempt_dir().exists() else []:
        att = read_attempt(path)
        if att["status"] == "running":
            started = datetime.fromisoformat(att["started_at"])
            wall = (datetime.now(started.tzinfo) - started).total_seconds()
            charged = max(float(att["elapsed_seconds"]), wall)
            update_attempt(path, status="stale_crashed", elapsed_seconds=round(charged, 3))
            notes.append(f"{att['unit_id']} attempt {att['attempt']} "
                          f"(phase {att['phase']}, charged {charged:.1f}s = "
                          f"max(heartbeat {att['elapsed_seconds']}s, wall {wall:.1f}s))")
    return notes


def charged_seconds() -> float:
    """Budget = sum over ALL attempt-ledger entries regardless of outcome
    (success, failed, stale_crashed, stopped_budget). Failures and crashes
    are charged at their last heartbeat -- never free."""
    total = 0.0
    if attempt_dir().exists():
        for path in attempt_dir().glob("*.json"):
            total += float(read_attempt(path)["elapsed_seconds"])
    return total


def budget_gate() -> None:
    """Called before each pair AND at every epoch boundary via heartbeat."""
    if datetime.now(HARD_STOP.tzinfo) >= HARD_STOP:
        raise BudgetStop("hard stop 2026-08-22 23:59:59 UTC-04 reached")
    if charged_seconds() >= GPU_SECONDS_CAP:
        raise BudgetStop(f"charged GPU seconds >= cap {GPU_SECONDS_CAP}")


def epoch_budget_check(base_charged: float, elapsed: float,
                        now: datetime | None = None) -> None:
    """The exact check the per-epoch heartbeat applies (factored out so the
    regression suite can drive it without a GPU). base_charged = all OTHER
    attempts' charged seconds at fit start; elapsed = this fit so far.
    Also invoked immediately BEFORE outer-test access (record 46 action 4):
    the last epoch callback may predate crossing the cap/deadline."""
    if (now or datetime.now(HARD_STOP.tzinfo)) >= HARD_STOP:
        raise BudgetStop("hard stop 2026-08-22 23:59:59 UTC-04 reached at epoch boundary")
    if base_charged + elapsed >= GPU_SECONDS_CAP:
        raise BudgetStop("charged GPU cap 4.0 h reached at epoch boundary")


def loss_ledger_path() -> Path:
    return TRAIN_ROOT / "loss_ledger.csv"


def record_unit_loss(unit_id: str, reason: str) -> None:
    """Append-only persistent record of a permanently lost unit (record 46
    action 3). Written the moment the loss is decided, survives restarts."""
    fields = ["at", "unit_id", "reason"]
    exists = loss_ledger_path().exists()
    loss_ledger_path().parent.mkdir(parents=True, exist_ok=True)
    with loss_ledger_path().open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow({"at": core.now_iso(), "unit_id": unit_id, "reason": reason})
        f.flush()
        os.fsync(f.fileno())


def permanently_lost_units() -> set[str]:
    if not loss_ledger_path().exists():
        return set()
    return {r["unit_id"] for r in core.read_csv(loss_ledger_path())}


def final_status(rows: list[dict[str, str]]) -> dict:
    """Terminal state semantics (record 46 action 3): `all_units_complete`
    requires ALL 792 units committed successfully; anything less is
    `incomplete/not_conclusive` with the missing/lost units listed."""
    successes = [r["unit_id"] for r in rows if unit_status(r) == "success"]
    lost = sorted(permanently_lost_units())
    missing = [r["unit_id"] for r in rows if unit_status(r) != "success"]
    if len(successes) == len(rows) and not lost:
        return {"status": "all_units_complete", "units_success": len(successes),
                "charged_seconds": charged_seconds()}
    return {"status": "incomplete/not_conclusive", "units_success": len(successes),
            "units_total": len(rows), "lost_units": lost,
            "missing_units": missing[:50],
            "n_missing": len(missing), "charged_seconds": charged_seconds()}


def verify_smoke_gate() -> None:
    """run() precondition (record 46 action 4): the recorded smoke test must
    be a PASS produced by exactly the frozen runner now on disk."""
    smoke_path = TRAIN_ROOT / "smoke_test.json"
    if not smoke_path.exists():
        raise RuntimeError("smoke_test.json absent -- run `smoke` first")
    record = json.loads(smoke_path.read_text(encoding="utf-8"))
    if record.get("status") != "pass":
        raise RuntimeError("recorded smoke test is not a pass")
    if record.get("runner_sha256") != core.sha256_file(RUNNER_PATH):
        raise RuntimeError("smoke_test.json was produced by a different runner "
                            "revision -- re-run `smoke` on the frozen runner")


def append_ledger(entry: dict) -> None:
    fields = ["at", "unit_id", "status", "elapsed_seconds", "epochs_run", "best_epoch",
              "init_state_sha256", "checkpoint_sha256"]
    exists = ledger_path().exists()
    ledger_path().parent.mkdir(parents=True, exist_ok=True)
    with ledger_path().open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(entry)
        f.flush()
        os.fsync(f.fileno())


def verify_substitute_anchors() -> dict:
    """Record-44 action 3: strong anchors replacing the inapplicable P1 prereg
    prefix check. Fail-closed on any mismatch."""
    p1_runner_sha = core.sha256_file(Path(p1.__file__))
    if p1_runner_sha != EXPECTED_P1_RUNNER_SHA256:
        raise RuntimeError("P1 official runner drifted from engineering-lock hash")
    frontend_hash = core.preprocessing_code_hash()
    if frontend_hash != EXPECTED_FRONTEND_CODE_HASH:
        raise RuntimeError("frozen front-end code hash drifted")
    ensure_ser_symbols_loaded_d1()
    sys.path.insert(0, str(core.SER_ROOT))
    try:
        from tuned_standard_experiment import candidate_configs
        observed_configs = {m: candidate_configs(m)[2] for m in
                             ("cnn", "resnet_se", "transformer", "fno")}
    finally:
        if sys.path and sys.path[0] == str(core.SER_ROOT):
            sys.path.pop(0)
    if core.canonical_json(observed_configs) != core.canonical_json(core.EXPECTED_MODEL_CONFIGS):
        raise RuntimeError("candidate_configs drifted from preregistered model configs")
    return {
        "p1_official_runner_sha256": p1_runner_sha,
        "frontend_code_hash": frontend_hash,
        "candidate_configs_match_preregistration": True,
        "core_full_file_sha256": core.sha256_file(Path(core.__file__)),
        "note": "core full-file hash differs from P1 lock era by documented non-frontend "
                 "edits; frontend + configs verified equal above (record 44 evidence 3)",
    }


def training_started() -> bool:
    return (TRAIN_ROOT / "units").exists() or attempt_dir().exists()


def prepare(refreeze: bool = False) -> None:
    if prepared_path().exists():
        if not refreeze:
            raise RuntimeError("d1_prepared.json already frozen; overwrite refused. "
                                "Use `refreeze` (allowed only before training starts).")
        if training_started():
            raise RuntimeError("cannot refreeze after training has started")
    rows = load_plan()
    anchors = verify_anchors(rows)
    ser_gate = ensure_ser_symbols_loaded_d1()
    substitute = verify_substitute_anchors()
    plan_sha = core.sha256_file(PLAN_PATH)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    prepared = {
        "prepared_at": core.now_iso(),
        "run_plan_sha256": plan_sha,
        "anchors": anchors,
        "ser_hash_gate": ser_gate,
        "substitute_anchors": substitute,
        "max_attempts_per_unit": MAX_ATTEMPTS,
        "n_units": len(rows),
        "n_matched_units": len({r["matched_unit_id"] for r in rows}),
        "label_order": list(EMOTION6),
        "n_classes": N_CLASSES,
        "gpu_seconds_cap": GPU_SECONDS_CAP,
        "hard_stop": HARD_STOP.isoformat(),
        "training_hyperparameters": {
            "reused_from": "run_p1_protocol_premium.py (imported, not copied)",
            "batch_size": p1.BATCH_SIZE, "max_epochs": p1.MAX_EPOCHS, "patience": p1.PATIENCE,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
            "checkpoint_rule": "strict minimum validation loss",
            "augmentation_scope": "fit subset only (p1.make_train_loader)",
            "validation_augmentation": False,
            "outer_test_augmentation": False,
            "normalization": "per-sample z-score at cache build (core.compute_logmel)",
        },
        "tool_sha256": {
            RUNNER_PATH.name: core.sha256_file(RUNNER_PATH),
            "run_p1_protocol_premium.py": core.sha256_file(Path(p1.__file__)),
            "p1_protocol_core.py": core.sha256_file(Path(core.__file__)),
        },
        "environment": core.environment_record(),
    }
    core.atomic_write_json(prepared_path(), prepared)
    print(core.canonical_json({"status": "prepared", "run_plan_sha256": plan_sha,
                                "n_units": len(rows)}))


def load_prepared() -> dict:
    if not prepared_path().exists():
        raise RuntimeError("run `prepare` first")
    prepared = json.loads(prepared_path().read_text(encoding="utf-8"))
    if core.sha256_file(PLAN_PATH) != prepared["run_plan_sha256"]:
        raise RuntimeError("run_plan.csv drifted since prepare")
    return prepared


def verify_frozen_at_run(prepared: dict) -> None:
    """Record-44 action 3: every `run` start re-verifies ALL frozen code and
    data anchors against d1_prepared.json. Fail-closed on any drift."""
    observed = {
        RUNNER_PATH.name: core.sha256_file(RUNNER_PATH),
        "run_p1_protocol_premium.py": core.sha256_file(Path(p1.__file__)),
        "p1_protocol_core.py": core.sha256_file(Path(core.__file__)),
    }
    for name, expected in prepared["tool_sha256"].items():
        if observed.get(name) != expected:
            raise RuntimeError(f"frozen code drift since prepare (fail-closed): {name}")
    verify_substitute_anchors()
    ensure_ser_symbols_loaded_d1()  # re-runs SER commit + 5-file hash gate


def smoke() -> None:
    """CPU-only synthetic checks; no D1 data metrics produced."""
    ensure_ser_symbols_loaded_d1()
    checks = []
    X = np.random.default_rng(7).normal(size=(4, 64, 128)).astype(np.float32)
    for model_name in ("cnn", "resnet_se", "transformer", "fno"):
        config = dict(core.EXPECTED_MODEL_CONFIGS[model_name])
        p1.set_reproducible_seed(1234)
        m1 = p1.BUILD_MODEL(model_name, 64, N_CLASSES, config)
        h1 = sha256_state_dict(m1)
        p1.set_reproducible_seed(1234)
        m2 = p1.BUILD_MODEL(model_name, 64, N_CLASSES, config)
        h2 = sha256_state_dict(m2)
        checks.append({"check": f"{model_name}_paired_init_reproducible", "pass": h1 == h2})
        with torch.no_grad():
            out = m1(torch.tensor(X))
        checks.append({"check": f"{model_name}_output_shape_4x6", "pass": tuple(out.shape) == (4, N_CLASSES)})
        p1.set_reproducible_seed(9999)
        m3 = p1.BUILD_MODEL(model_name, 64, N_CLASSES, config)
        checks.append({"check": f"{model_name}_different_seed_different_init",
                        "pass": sha256_state_dict(m3) != h1})
    import inspect
    src = inspect.getsource(attempt_fit)
    checks.append({"check": "attempt_fit_has_one_outer_test_call_site",
                    "pass": src.count("evaluate_outer_test_once(") == 1})
    checks.append({"check": "seed_set_before_model_build",
                    "pass": src.index("set_reproducible_seed") < src.index("BUILD_MODEL(")})
    failed = [c["check"] for c in checks if not c["pass"]]
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    core.atomic_write_json(TRAIN_ROOT / "smoke_test.json", {
        "created_at": core.now_iso(), "synthetic_only": True,
        "status": "pass" if not failed else "fail",
        "n_checks": len(checks), "failed": failed, "checks": checks,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
    })
    if failed:
        raise RuntimeError(f"smoke failed: {failed}")
    print(core.canonical_json({"status": "pass", "checks": len(checks), "synthetic_only": True}))


def attempt_fit(row: dict[str, str], X: np.ndarray, y_all: np.ndarray,
                 pair_rows: list[dict[str, str]], device: torch.device,
                 epoch_callback=None, phase_callback=None) -> dict:
    ensure_ser_symbols_loaded_d1()
    rel = row["inner_split_path"]
    npz_path = NPZ_DIR / rel.replace("/", os.sep)
    if core.sha256_file(npz_path) != row["inner_split_sha256"]:
        raise RuntimeError(f"inner split hash drift: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as split:
        fit_idx = np.asarray(split["fit_idx"], dtype=np.int64)
        val_idx = np.asarray(split["val_idx"], dtype=np.int64)
        test_idx = np.asarray(split["outer_test_idx"], dtype=np.int64)
    train_seed = int(row["train_seed_u32"])
    if core.stable_u32(row["train_seed_key"]) != train_seed:
        raise RuntimeError("train_seed_u32 does not match its key")
    config = dict(core.EXPECTED_MODEL_CONFIGS[row["model"]])
    # P5 lesson: seed BEFORE build, explicitly, in this file.
    p1.set_reproducible_seed(train_seed)
    model = p1.BUILD_MODEL(row["model"], 64, N_CLASSES, config)
    init_state_sha = sha256_state_dict(model)
    model, history, fit_summary = p1.train_one_fit(
        model, X[fit_idx], y_all[fit_idx], X[val_idx], y_all[val_idx],
        config, train_seed, N_CLASSES, device,
        epoch_callback,  # heartbeat + budget/deadline gate at every epoch boundary
    )
    if phase_callback is not None:
        phase_callback("outer_test")  # ledger marks test access BEFORE it happens
    logits, predictions = p1.evaluate_outer_test_once(model, X[test_idx], y_all[test_idx], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "row": row, "config": config, "history": history, "fit_summary": fit_summary,
        "init_state_sha256": init_state_sha, "model": model,
        "test_idx": test_idx, "y_true": y_all[test_idx],
        "predictions": predictions, "logits": logits,
    }


def write_unit(result: dict, pair_rows: list[dict[str, str]]) -> dict:
    row = result["row"]
    final = unit_dir(row)
    if final.exists():
        raise RuntimeError(f"refusing to overwrite unit dir: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.parent / f".{final.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    torch.save(result["model"].state_dict(), stage / "checkpoint.pt")
    checkpoint_sha = core.sha256_file(stage / "checkpoint.pt")
    config_record = {
        **{k: row[k] for k in row},
        "model_config": result["config"],
        "init_state_sha256": result["init_state_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "label_order": list(EMOTION6),
        "n_classes": N_CLASSES,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "completed_at": core.now_iso(),
    }
    core.atomic_write_json(stage / "config.json", config_record)
    history_fields = ["epoch", "train_loss", "validation_loss", "validation_uar",
                       "validation_accuracy", "validation_macro_f1", "learning_rate"]
    core.atomic_write_csv(stage / "history.csv", history_fields, [
        {k: (str(item[k]) if k == "epoch" else format(float(item[k]), ".17g"))
         for k in history_fields} for item in result["history"]
    ])
    channel = row["channel"]
    pred_fields = ["unit_id", "channel", "model", "protocol", "seed", "outer_fold",
                    "sample_index", "canonical_actor", "relative_path", "true_label_index",
                    "true_label_name", "predicted_label_index", "predicted_label_name"] + \
                   [f"logit_{l}" for l in EMOTION6]
    pred_rows = []
    for local, sample_index in enumerate(result["test_idx"].tolist()):
        t = int(result["y_true"][local]); p = int(result["predictions"][local])
        item = {
            "unit_id": row["unit_id"], "channel": channel, "model": row["model"],
            "protocol": row["protocol"], "seed": row["seed"], "outer_fold": row["outer_fold"],
            "sample_index": str(sample_index),
            "canonical_actor": pair_rows[sample_index]["actor"].zfill(2),
            "relative_path": pair_rows[sample_index][f"{channel}_relative_path"],
            "true_label_index": str(t), "true_label_name": EMOTION6[t],
            "predicted_label_index": str(p), "predicted_label_name": EMOTION6[p],
        }
        for li, l in enumerate(EMOTION6):
            item[f"logit_{l}"] = format(float(result["logits"][local, li]), ".17g")
        pred_rows.append(item)
    core.atomic_write_csv(stage / "predictions.csv", pred_fields, pred_rows)
    run_record = {
        "unit_id": row["unit_id"], "status": "success", "completed_at": core.now_iso(),
        **result["fit_summary"],
        "init_state_sha256": result["init_state_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "n_predictions": len(pred_rows),
        "config_sha256": core.sha256_file(stage / "config.json"),
        "history_sha256": core.sha256_file(stage / "history.csv"),
        "predictions_sha256": core.sha256_file(stage / "predictions.csv"),
    }
    core.atomic_write_json(stage / "run.json", run_record)
    os.replace(stage, final)
    return run_record


def run_one_unit(row: dict[str, str], caches: dict, y_all: np.ndarray,
                  pair_rows: list[dict[str, str]], device: torch.device) -> dict:
    """Full attempt lifecycle for one fit: attempt-before-compute ledger,
    epoch heartbeat + budget gate, test-once phase marker, commit retried
    only from the in-memory result. Returns the unit run record."""
    attempt_path = open_attempt(row["unit_id"])
    base_charged = charged_seconds() - float(read_attempt(attempt_path)["elapsed_seconds"])
    started = time.perf_counter()

    def epoch_heartbeat(epoch: int, elapsed: float) -> None:
        update_attempt(attempt_path, elapsed_seconds=round(elapsed, 3), last_epoch=epoch)
        epoch_budget_check(base_charged, elapsed)

    def phase_marker(phase: str) -> None:
        if phase == "outer_test":
            # record 46 action 4: re-gate cap/deadline immediately BEFORE any
            # outer-test access -- the last epoch callback may be stale. A stop
            # here leaves phase="training", so the unit remains retryable.
            epoch_budget_check(base_charged, time.perf_counter() - started)
        update_attempt(attempt_path, phase=phase,
                        elapsed_seconds=round(time.perf_counter() - started, 3))

    try:
        result = attempt_fit(row, caches[row["channel"]], y_all, pair_rows, device,
                              epoch_callback=epoch_heartbeat, phase_callback=phase_marker)
    except BudgetStop:
        update_attempt(attempt_path, status="stopped_budget",
                        elapsed_seconds=round(time.perf_counter() - started, 3))
        raise
    except BaseException:
        update_attempt(attempt_path, status="failed",
                        elapsed_seconds=round(time.perf_counter() - started, 3))
        raise
    # outer test has been accessed; commit may be retried, retraining may not.
    phase_marker("commit")
    record = None
    commit_error = None
    for _ in range(2):  # in-memory commit retry only (test-once rule)
        try:
            record = write_unit(result, pair_rows)
            break
        except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised below
            commit_error = exc
    if record is None:
        update_attempt(attempt_path, status="failed_commit",
                        elapsed_seconds=round(time.perf_counter() - started, 3))
        raise RuntimeError(f"unit commit failed after in-memory retries: {commit_error}")
    elapsed = time.perf_counter() - started
    update_attempt(attempt_path, status="success", elapsed_seconds=round(elapsed, 3))
    append_ledger({
        "at": core.now_iso(), "unit_id": row["unit_id"], "status": "success",
        "elapsed_seconds": format(elapsed, ".3f"),
        "epochs_run": str(record["epochs_run"]), "best_epoch": str(record["best_epoch"]),
        "init_state_sha256": result["init_state_sha256"],
        "checkpoint_sha256": record["checkpoint_sha256"],
    })
    return {"record": record, "init_state_sha256": result["init_state_sha256"]}


def run() -> None:
    if halt_path().exists():
        raise RuntimeError(f"paired-init halt file present, resolve first: {halt_path()}")
    prepared = load_prepared()
    verify_frozen_at_run(prepared)  # record-44 action 3: fail-closed code anchors
    verify_smoke_gate()             # record-46 action 4: smoke pass by THIS runner
    rows = load_plan()
    verify_anchors(rows)
    lock = acquire_run_lock()       # record-46 action 2: whole-run exclusivity
    try:
        _run_locked(rows)
    finally:
        release_run_lock(lock)


def _run_locked(rows: list[dict[str, str]]) -> None:
    stale = reconcile_stale_attempts()
    for note in stale:
        print(f"[stale attempt reconciled, conservative charge] {note}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; D1 formal training requires the GPU")
    device = torch.device("cuda")
    y_all = load_labels()
    pair_rows = load_pair_rows()
    caches = {}
    for ch, path in CACHE.items():
        with np.load(path, allow_pickle=False) as data:
            caches[ch] = np.asarray(data["X"], dtype=np.float32)

    by_matched: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for row in rows:
        if row["matched_unit_id"] not in by_matched:
            order.append(row["matched_unit_id"])
            by_matched[row["matched_unit_id"]] = []
        by_matched[row["matched_unit_id"]].append(row)

    done = skipped = 0
    already_lost = permanently_lost_units()
    for matched_id in order:
        pair = by_matched[matched_id]
        if len(pair) != 2 or {r["channel"] for r in pair} != {"speech", "song"}:
            raise RuntimeError(f"matched unit malformed: {matched_id}")
        pair.sort(key=lambda r: 0 if r["channel"] == "speech" else 1)
        if all(unit_status(r) == "success" for r in pair):
            skipped += 2
            continue
        if any(r["unit_id"] in already_lost for r in pair):
            # record 46 action 3: a persisted loss is terminal -- skip without
            # re-raising it as a fresh failure on every restart.
            continue
        try:
            budget_gate()  # pair-level gate (epoch-level gate lives in the heartbeat)
        except BudgetStop as stop:
            print(core.canonical_json({"status": "stopped", "reason": str(stop),
                                        "completed_fits_this_run": done,
                                        **final_status(rows)}))
            return
        init_hashes = {}
        pair_abort = False
        for row in pair:
            if unit_status(row) == "success":
                cfg = json.loads((unit_dir(row) / "config.json").read_text(encoding="utf-8"))
                init_hashes[row["channel"]] = cfg["init_state_sha256"]
                continue
            try:
                outcome = run_one_unit(row, caches, y_all, pair_rows, device)
            except BudgetStop as stop:
                print(core.canonical_json({"status": "stopped_mid_pair", "reason": str(stop),
                                            "incomplete_matched_unit": matched_id,
                                            "completed_fits_this_run": done,
                                            **final_status(rows)}))
                return
            except RuntimeError as exc:
                # attempt-cap / test-once refusals and commit failures land here.
                # Persist the loss (append-only, survives restarts), then continue
                # with the next matched unit.
                record_unit_loss(row["unit_id"], str(exc))
                pair_abort = True
                break
            init_hashes[row["channel"]] = outcome["init_state_sha256"]
            done += 1
        if pair_abort:
            print(f"[matched unit lost -> persisted to loss ledger] {matched_id}", flush=True)
            continue
        if init_hashes["speech"] != init_hashes["song"]:
            core.atomic_write_json(halt_path(), {
                "at": core.now_iso(), "matched_unit_id": matched_id,
                "init_speech": init_hashes["speech"], "init_song": init_hashes["song"],
                "action": "fail-closed halt: paired-initialization invariant violated",
            })
            raise RuntimeError(f"paired-init mismatch in {matched_id}; halt file written")
        print(f"[matched unit done] {matched_id} (completed fits: {done}, skipped: {skipped})",
              flush=True)
    # record 46 action 3: `all_units_complete` ONLY on 792/792 success.
    print(core.canonical_json({"completed_fits_this_run": done,
                                "skipped_existing": skipped, **final_status(rows)}))


def status() -> None:
    rows = load_plan()
    print(core.canonical_json({
        **final_status(rows),
        "gpu_seconds_cap": GPU_SECONDS_CAP,
        "halt_file_present": halt_path().exists(),
        "run_lock_present": run_lock_path().exists(),
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "refreeze", "smoke", "run", "status"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(refreeze=False)
    elif args.command == "refreeze":
        prepare(refreeze=True)
    else:
        {"smoke": smoke, "run": run, "status": status}[args.command]()


if __name__ == "__main__":
    main()

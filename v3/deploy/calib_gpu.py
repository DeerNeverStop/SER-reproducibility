"""GPU runner for the B2 cnn / wavlm_ft units (SER26-DEPLOY-1, SPEC_DEPLOY_ZH.md section 4).

Consumes the plan written by `python -m v3.deploy.calib plan` (work/plan/B2.json), runs the engines_deploy copies of
the v2 engines (p1_frozen CNN on the v2 log-mel cache; WavLM-base+ partial fine-tune on the corpus audio) and writes
the unit contract defined in v3/deploy/calib.py (val_predictions.csv, test_predictions.csv, unit.json, DONE).

    python -m v3.deploy.calib_gpu run --plan v3/deploy/work/plan/B2.json --out v3/deploy/work/B2 \
        --models cnn[,wavlm_ft] --device cuda --audio-root /workspace/ser-deploy-data/audio [--audio-profile pod] \
        [--limit N] [--units ID,ID,...] [--threads 2] [--no-verify-audio]
    python -m v3.deploy.calib_gpu status --plan v3/deploy/work/plan/B2.json --out v3/deploy/work/B2

Environment: SER_V2_DATA_ROOT must point at the v2 data root (features/, plan_rc2/splits/). --device cpu installs
common.cpu_guard (CUDA hidden, 2 threads, BelowNormal) before torch is imported; --device cuda refuses to fall back
to the CPU silently. Resumable: units whose DONE verifies (calib.done_status == "ok") are skipped. Only counts,
epochs and seconds are printed. Events (start / done / failed / skipped / redo) go to <out>/ledger_gpu.jsonl and
every finished unit also appends one line to <out>/ledger.jsonl (contract).
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

from . import common
from .common import (DEPLOY_ROOT, LEVELS, IntegrityError, Timer, atomic_write_json, class_table, digest, labels_for,
                     load_manifest, load_split, now, read_json, require, sha256_file, speakers_for)
from .calib import done_status, unit_dir, write_unit_outputs

GPU_MODELS = ("cnn", "wavlm_ft")
AUDIO_ROOTS_FILE = DEPLOY_ROOT / "pod" / "audio_roots.json"


# ----------------------------------------------------------------------------- plan / data access

def load_plan(path: Path) -> dict:
    plan = read_json(path)
    require(plan.get("module") == "B2" and isinstance(plan.get("units"), list), f"{path} is not a B2 plan")
    return plan


def select_units(plan: dict, models, unit_ids=None) -> list:
    """Requested models in the requested order; within a model the plan order is kept."""
    for m in models:
        require(m in GPU_MODELS, f"unknown GPU model {m!r}; expected one of {GPU_MODELS}")
    by_model = {m: [u for u in plan["units"] if u["model"] == m] for m in models}
    units = [u for m in models for u in by_model[m]]
    if unit_ids:
        wanted = set(unit_ids)
        units = [u for u in units if u["unit_id"] in wanted]
        require(len(units) == len(wanted), "unknown --units id (or id not among the requested models)")
    return units


def resolve_audio_roots(audio_root: Path | None, profile: str) -> dict:
    """base corpus -> directory, from pod/audio_roots.json (relative entries hang under --audio-root)."""
    table = read_json(AUDIO_ROOTS_FILE)
    require(profile in table["profiles"], f"unknown audio profile {profile!r}; have {sorted(table['profiles'])}")
    roots = {}
    for base, entry in table["profiles"][profile].items():
        p = Path(entry)
        if not p.is_absolute():
            require(audio_root is not None, f"--audio-root is required for relative audio profile entry {base}={entry}")
            p = Path(audio_root) / p
        roots[base] = p
    return roots


class UnitData:
    """What the engines read: labels / speakers from the v2 manifest, log-mel rows from the v2 cache, audio root."""

    def __init__(self, base: str, manifest: dict, n_classes: int, feature_cache=None, audio_root: Path | None = None):
        self.base, self.manifest, self.n_classes = base, manifest, n_classes
        self.cache, self.audio_root = feature_cache, audio_root

    def labels(self, paths) -> np.ndarray:
        return labels_for(self.manifest, paths)

    def speakers(self, paths) -> list:
        return speakers_for(self.manifest, paths)

    def logmel(self, paths) -> np.ndarray:
        require(self.cache is not None, "log-mel cache not loaded for this unit")
        return self.cache.get(paths)


def verify_audio(manifest: dict, root: Path, paths) -> int:
    """sha256 of every listed wav against the v2 manifest; returns the number verified (raises on any mismatch)."""
    bad = []
    for p in paths:
        f = root / p
        if not f.is_file() or sha256_file(f) != manifest[p]["sha256"]:
            bad.append(p)
    require(not bad, f"{len(bad)} audio files missing or sha-mismatched under {root} (first: {bad[:3]})")
    return len(paths)


# ----------------------------------------------------------------------------- environment

def make_device(name: str, threads: int):
    guard = None
    if name == "cpu":
        guard = common.cpu_guard(threads)
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=threads)
    torch.set_num_threads(threads)
    if name.startswith("cuda"):
        require(torch.cuda.is_available(), "--device cuda requested but torch.cuda.is_available() is False (no silent CPU fallback)")
    device = torch.device(name)
    if name == "cpu":
        torch.set_num_threads(threads)
    return device, guard


def environment_info(device, threads: int, guard) -> dict:
    import torch
    from . import engines_deploy
    env = {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "device": str(device), "threads": threads, "cpu_guard": guard,
           "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "gpu_total_memory_gb": (round(torch.cuda.get_device_properties(device).total_memory / 2 ** 30, 2)
                                   if device.type == "cuda" else None),
           "engine_version": engines_deploy.ENGINE_VERSION, "v2_engine_version": engines_deploy.V2_ENGINE_VERSION,
           "engines_deploy_sha256": sha256_file(Path(engines_deploy.__file__)),
           "advanced_models_sha256": sha256_file(common.REPO_ROOT / "advanced_models.py"),
           "v2_train_py_sha256": sha256_file(common.REPO_ROOT / "v2" / "ser_v2" / "train.py")}
    for mod in ("torchaudio", "librosa", "sklearn", "scipy"):
        try:
            env[mod] = importlib.import_module(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = None
    return env


# ----------------------------------------------------------------------------- one unit

def build_fold(table: dict, u: dict) -> dict:
    fold = table["folds"][u["fold"]]
    require(fold["fold"] == u["fold"], "fold index mismatch in split table")
    require((len(fold["fit"]), len(fold["val"]), len(fold["test"])) == (u["n_fit"], u["n_val"], u["n_test"]), "fold sizes differ from the plan")
    return {"fit": list(fold["fit"]), "val": list(fold["val"]), "test": list(fold["test"])}


def run_unit(u: dict, fold: dict, data: UnitData, engine_fn, device, udir: Path, env: dict, extra_info: dict | None = None) -> dict:
    """Run one engine call and write the contract files (csvs, unit.json, history.json, then DONE)."""
    started = now()
    with Timer() as t:
        test_logits, val_logits, history, info = engine_fn(u, u["config"], fold, data, device)
    test_logits, val_logits = np.asarray(test_logits, dtype=np.float64), np.asarray(val_logits, dtype=np.float64)
    require(test_logits.shape == (len(fold["test"]), u["n_classes"]), f"test logits shape {test_logits.shape}")
    require(val_logits.shape == (len(fold["val"]), u["n_classes"]), f"val logits shape {val_logits.shape}")
    require(np.isfinite(test_logits).all() and np.isfinite(val_logits).all(), "non-finite logits")
    preds = {}
    for role, s in (("val", val_logits), ("test", test_logits)):
        preds[role] = {"paths": fold[role], "speakers": data.speakers(fold[role]), "y_true": data.labels(fold[role]),
                       "y_pred": s.argmax(axis=1), "scores": s}
    checkpoint_state = info.pop("_checkpoint_state", None)
    if checkpoint_state is not None:
        import torch
        udir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = udir / "checkpoint.pt"
        torch.save({"unit_id": u["unit_id"], "config": u["config"], "state_dict": checkpoint_state}, checkpoint_path)
        info["checkpoint_sha256"] = sha256_file(checkpoint_path)
        del checkpoint_state
    engine_info = {k: v for k, v in info.items() if k not in ("best_epoch", "epochs_run")}
    extra = {"environment": env, "timing": {"started_at": started, "finished_at": now(), "seconds": t.seconds},
             "best_epoch": info.get("best_epoch"), "epochs_run": info.get("epochs_run"),
             "engine_info": engine_info, "n_history": len(history), **(extra_info or {})}
    udir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(udir / "history.json", history)
    return write_unit_outputs(udir, u, extra, preds["val"], preds["test"])


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------- run

def cmd_run(args) -> int:
    models = [m for m in args.models.split(",") if m]
    plan = load_plan(Path(args.plan))
    units = select_units(plan, models, [x for x in (args.units or "").split(",") if x])
    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    run_guard = getattr(args, "run_guard", None)
    device, guard = make_device(args.device, args.threads)
    from . import engines_deploy
    env = environment_info(device, args.threads, guard)
    audio_roots = resolve_audio_roots(Path(args.audio_root) if args.audio_root else None, args.audio_profile) if "wavlm_ft" in models else {}
    events, ledger = run_root / "ledger_gpu.jsonl", run_root / "ledger.jsonl"
    append_jsonl(events, {"event": "session_start", "at": now(), "models": models, "n_selected": len(units), "device": str(device),
                          "hostname": env["hostname"], "gpu_name": env["gpu_name"], "torch": env["torch"]})
    print(f"[calib_gpu] {len(units)} unit(s) selected for {models} on {device} ({env['gpu_name'] or 'cpu'})", flush=True)
    manifests, splits, audio_ok = {}, {}, {}
    cache, cache_key = None, None
    n_done = n_skipped = n_failed = 0
    for i, u in enumerate(units):
        udir = unit_dir(run_root, u["unit_id"])
        if run_guard and run_guard.sealed(u, udir):
            n_skipped += 1
            continue
        status = done_status(udir, u["n_classes"], u["n_val"], u["n_test"])
        if status == "ok":
            n_skipped += 1
            continue
        if (udir / "DONE").exists():
            append_jsonl(events, {"event": "redo", "unit_id": u["unit_id"], "reason": status, "at": now()})
        if args.limit is not None and n_done >= args.limit:
            break
        if run_guard:
            run_guard.before(u)
        append_jsonl(events, {"event": "start", "unit_id": u["unit_id"], "model": u["model"], "level": u["level"], "cell": u["cell"],
                              "r": u["r"], "fold": u["fold"], "at": now()})
        try:
            require(digest(u["config"]) == u["config_sha256"], "plan config body does not hash to config_sha256")
            require(u["engine"] == engines_deploy.PLAN_MODEL_ENGINE[u["model"]] == u["config"]["engine"], "engine name mismatch")
            base = u["base"]
            if base not in manifests:
                manifests[base] = load_manifest(base)
                require(sha256_file(common.manifest_path(base)) == u["manifest_sha256"], f"manifest sha mismatch for {base}")
            manifest = manifests[base]
            classes = class_table(manifest)
            require(len(classes) == u["n_classes"], "class count mismatch")
            if u["split_key"] not in splits:
                table = load_split(u["split_key"], verify=True)
                require(common.split_index()[u["split_key"]]["sha256"] == u["split_sha256"], "split sha differs from the plan")
                splits[u["split_key"]] = table
            fold = build_fold(splits[u["split_key"]], u)
            extra_info = {"classes": classes}
            if u["model"] == "cnn":
                key = (u["feature_corpus"], u["feature_kind"])
                if cache_key != key:
                    cache = common.FeatureCache(*key)
                    require(cache.path.name == u["feature_file"] and cache.sha256 == u["feature_sha256"], f"feature cache mismatch for {key}")
                    cache_key = key
                data = UnitData(base, manifest, u["n_classes"], feature_cache=cache)
                extra_info["feature_cache_sha256"] = cache.sha256
            else:
                root = audio_roots[base]
                if args.verify_audio and base not in audio_ok:
                    audio_ok[base] = verify_audio(manifest, root, splits[u["split_key"]]["population"])
                    print(f"[calib_gpu] audio verified for {base}: {audio_ok[base]} files", flush=True)
                data = UnitData(base, manifest, u["n_classes"], audio_root=root)
                extra_info["audio_root"] = str(root)
                extra_info["audio_sha_verified"] = audio_ok.get(base)
            engine_fn = engines_deploy.ENGINES[u["engine"]]
            unit = run_unit(u, fold, data, engine_fn, device, udir, env, extra_info)
            if run_guard:
                run_guard.finish(u, udir)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            append_jsonl(events, {"event": "failed", "unit_id": u["unit_id"], "error": repr(exc)[:500],
                                  "traceback": traceback.format_exc()[-2000:], "at": now()})
            print(f"[calib_gpu] unit {u['unit_id'][:12]} ({u['level']} {u['model']} {u['cell']} r{u['r']} f{u['fold']}) FAILED: {exc!r}",
                  file=sys.stderr, flush=True)
            if args.fail_fast:
                break
            continue
        n_done += 1
        secs = unit["timing"]["seconds"]
        append_jsonl(events, {"event": "done", "unit_id": u["unit_id"], "seconds": secs, "train_seconds": unit["engine_info"].get("train_seconds"),
                              "best_epoch": unit["best_epoch"], "epochs_run": unit["epochs_run"], "at": now()})
        append_jsonl(ledger, {"unit_id": u["unit_id"], "status": "done", "seconds": secs, "finished_at": unit["timing"]["finished_at"],
                              "val_sha256": unit["val_sha256"], "test_sha256": unit["test_sha256"], "runner": "calib_gpu"})
        print(f"[calib_gpu] {i + 1}/{len(units)} {u['level']} {u['model']} {u['cell']} r{u['r']} f{u['fold']}: "
              f"epochs={unit['epochs_run']} best={unit['best_epoch']} {secs:.1f}s (done={n_done} skipped={n_skipped} failed={n_failed})", flush=True)
    summary = {"models": models, "selected": len(units), "done": n_done, "skipped_ok": n_skipped, "failed": n_failed, "device": str(device)}
    append_jsonl(events, {"event": "session_end", "at": now(), **summary})
    print("[calib_gpu] " + json.dumps(summary), flush=True)
    return 1 if n_failed else 0


# ----------------------------------------------------------------------------- status

def cmd_status(args) -> int:
    plan = load_plan(Path(args.plan))
    run_root = Path(args.out)
    counts = {}
    for u in plan["units"]:
        if u["model"] not in GPU_MODELS:
            continue
        st = done_status(unit_dir(run_root, u["unit_id"]), u["n_classes"], u["n_val"], u["n_test"])
        key = (u["model"], u["level"])
        counts.setdefault(key, {"units": 0, "ok": 0, "other": {}})
        counts[key]["units"] += 1
        if st == "ok":
            counts[key]["ok"] += 1
        else:
            counts[key]["other"][st] = counts[key]["other"].get(st, 0) + 1
    failed = set()
    events = run_root / "ledger_gpu.jsonl"
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if e.get("event") == "failed":
                    failed.add(e["unit_id"])
                elif e.get("event") == "done":
                    failed.discard(e["unit_id"])
    total_ok = sum(c["ok"] for c in counts.values())
    total = sum(c["units"] for c in counts.values())
    for (model, level), c in sorted(counts.items()):
        print(f"{model:9s} {level:12s} ok={c['ok']:3d}/{c['units']:3d} " + (json.dumps(c["other"]) if c["other"] else ""))
    print(f"total ok={total_ok}/{total} failed_last={len(failed)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run cnn / wavlm_ft units")
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--models", default="cnn", help="comma list drawn from cnn,wavlm_ft (run order)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--audio-root", default=None, help="directory holding the per-corpus audio subdirs (see pod/audio_roots.json)")
    r.add_argument("--audio-profile", default="pod", help="profile in pod/audio_roots.json (pod | local_windows)")
    r.add_argument("--limit", type=int, default=None, help="stop after N newly finished units")
    r.add_argument("--units", default=None, help="comma list of unit ids")
    r.add_argument("--threads", type=int, default=2, help="CPU threads (only enforced for --device cpu)")
    r.add_argument("--no-verify-audio", dest="verify_audio", action="store_false", help="skip the per-corpus audio sha check")
    r.add_argument("--fail-fast", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", help="DONE counts per model x level")
    s.add_argument("--plan", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except IntegrityError as exc:
        print(f"[calib_gpu] integrity error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

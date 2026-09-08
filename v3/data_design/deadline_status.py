"""Read-only deadline forecast from execution metadata; never read predictions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

UTC = timezone.utc
GPU_GATE = datetime.fromisoformat("2026-09-07T18:00:00-04:00")
COMPUTE_GATE = datetime.fromisoformat("2026-09-10T20:00:00-04:00")


def moment(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


def read_events(path):
    # A partial concurrent last write is not silently treated as a valid ledger.
    payload = Path(path).read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError("ledger snapshot ends with an unfinished line; retry the read")
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def old_status(events, now):
    done = {}
    failures = Counter()
    for event in events:
        if event.get("event") == "done":
            done.setdefault(event["unit_id"], moment(event["at"]))
        elif event.get("event") == "failed":
            failures[event["unit_id"]] += 1
    times = sorted(done.values())
    if any(t > now for t in times):
        raise ValueError("ledger contains future completion timestamps")
    # Two failures can invalidate a whole old-study draw. Leave its accounting
    # to the old owner and withhold a naive 1920-minus-done forecast in that case.
    ambiguous = sorted(uid for uid, n in failures.items() if n >= 2)
    remaining = max(0, 1920 - len(done)) if not ambiguous else None
    windows = []
    for count in (20, 50, 100):
        if len(times) < count:
            continue
        elapsed = (times[-1] - times[-count]).total_seconds()
        if elapsed > 0:
            rate = (count - 1) * 3600 / elapsed
            windows.append({"recent_completions": count, "units_per_hour": rate})
    stale = bool(remaining) and (not times or (now - times[-1]).total_seconds() > 3600)
    eta = None
    if remaining == 0:
        eta = now
    elif remaining is not None and windows and not stale:
        eta = now + timedelta(hours=remaining / min(x["units_per_hour"] for x in windows))
    return {"unique_done": len(done), "remaining_primary_estimate": remaining,
            "failed_events": sum(failures.values()), "ambiguous_draw_accounting_units": ambiguous,
            "unrecovered_failed_units": sorted(set(failures) - set(done)),
            "last_done_at": times[-1].isoformat() if times else None,
            "stale_over_one_hour": stale, "throughput": windows,
            "training_eta_utc": eta.isoformat() if eta else None,
            "counts_are_ledger_progress_not_artifact_verification": True}


def study_status(events, recovered_metadata=None):
    recovered_metadata = recovered_metadata or {}
    done = {"ridge_wavlm": {}, "cnn": {}}
    latest = {}
    for event in events:
        if event.get("event") in ("unit_start", "unit_failed", "unit_done"):
            latest[event["unit_id"]] = event
        if event.get("event") != "unit_done":
            continue
        metadata = recovered_metadata.get(event["unit_id"], {})
        model = event.get("model", metadata.get("model"))
        if model not in done:
            raise ValueError("unknown Study II model")
        duration = float(event.get("wall_seconds", metadata.get("wall_seconds")))
        if not 0 <= duration < float("inf"):
            raise ValueError("invalid unit duration")
        done[model].setdefault(event["unit_id"], duration)
    if any(len(v) > 720 for v in done.values()):
        raise ValueError("Study II unit count exceeds plan")
    durations = list(done["cnn"].values())
    # The first eight ordered units need not represent both budgets. Preserve
    # a 120 s/unit planning allowance until the whole run finishes; raise it
    # when observed runtimes are slower. This is not a measured GPU benchmark.
    seconds = max(120.0, max(durations[-50:], default=0.0) * 1.5)
    return {"done_by_model": {m: len(v) for m, v in done.items()},
            "remaining_ridge": 720 - len(done["ridge_wavlm"]),
            "remaining_cnn": 720 - len(durations), "cnn_seconds_per_unit_allowance": seconds,
            "exhausted_failed_units": sorted(uid for uid, e in latest.items()
                                              if e["event"] == "unit_failed" and e.get("attempt", 0) >= 2),
            "timing_basis": "max(120-second planning assumption, 1.5 times slowest of last 50 CNN units)",
            "representative_cnn_benchmark_available": False}


def recovery_metadata(events, run):
    result = {}
    for event in events:
        if event.get("event") != "unit_done" or all(k in event for k in ("model", "wall_seconds")):
            continue
        uid = event["unit_id"]
        if not isinstance(uid, str) or len(uid) != 64 or any(c not in "0123456789abcdef" for c in uid):
            raise ValueError("invalid recovered unit identifier")
        folder = run / "units" / uid
        payload = (folder / "unit.json").read_bytes()
        done = json.loads((folder / "DONE").read_text(encoding="utf-8"))
        receipt = json.loads(payload)
        if hashlib.sha256(payload).hexdigest() != done.get("unit_json_sha256") or receipt.get("unit_id") != uid:
            raise ValueError("recovered unit metadata hash/identity mismatch")
        result[uid] = {k: receipt[k] for k in ("model", "wall_seconds")}
    return result


def forecast(now, old, study, gpu_ready=False, handoff_hours=6.0, verification_hours=6.0):
    if handoff_hours < 0 or verification_hours < 0:
        raise ValueError("buffers must be nonnegative")
    reasons = []
    if study["remaining_ridge"]:
        reasons.append("Ridge block incomplete; CNN-only forecast cannot establish full-core completion")
    if study["exhausted_failed_units"]:
        reasons.append("Study II has exhausted fit attempts; investigate before making a compute/rental estimate")
    if not gpu_ready and study["remaining_cnn"] and now >= GPU_GATE:
        reasons.append("GPU unavailable past September 7 18:00 Toronto")
    available = now if gpu_ready or not study["remaining_cnn"] else None
    if available is None and old["training_eta_utc"]:
        available = moment(old["training_eta_utc"]) + timedelta(hours=handoff_hours)
    finish = None
    if available is not None and not study["remaining_ridge"] and not study["exhausted_failed_units"]:
        finish = max(now, available) + timedelta(
            seconds=study["remaining_cnn"] * study["cnn_seconds_per_unit_allowance"],
            hours=verification_hours)
        if finish > COMPUTE_GATE:
            reasons.append("projected new compute plus verification exceeds September 10 20:00 Toronto")
    elif available is None and study["remaining_cnn"]:
        reasons.append("GPU availability cannot be forecast reliably; inspect old-task operational state")
    return {"needs_attention": bool(reasons), "reasons": reasons,
            "gpu_ready_operator_confirmed": gpu_ready,
            "projected_compute_and_verification_end_utc": finish.isoformat() if finish else None,
            "handoff_buffer_hours_assumed": handoff_hours, "verification_buffer_hours_assumed": verification_hours,
            "gpu_availability_gate": GPU_GATE.isoformat(), "compute_gate": COMPUTE_GATE.isoformat(),
            "interpretation": "planning estimate, not a guarantee or permission to launch/rent; inspect any alert before recommending a purchase"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--n14r-root", required=True, type=Path)
    parser.add_argument("--gpu-ready", action="store_true", help="operator confirmed released local or authorized independent GPU; this flag cannot authorize execution")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    try:
        locked = json.loads((root / "v3/data_design/evidence/PLAN_LOCK.json").read_text(encoding="utf-8"))
        mismatch = [name for name, sha in locked["source_sha256"].items()
                    if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha]
        old_events = read_events(args.n14r_root / "v2_1/n14r/runs/n14r/ledger.jsonl")
        run = root / "v3/data_design/work/core"
        study_events = read_events(run / "ledger.jsonl")
        metadata = recovery_metadata(study_events, run)
        # Read the clock after the snapshots, so a concurrent valid completion
        # cannot become a fictitious future event simply because I/O took time.
        now = datetime.now(UTC)
        old = old_status(old_events, now)
        study = study_status(study_events, metadata)
        prediction = forecast(now, old, study, args.gpu_ready)
        if mismatch:
            prediction["needs_attention"] = True
            prediction["reasons"].append("frozen source identity mismatch; hold Study II and investigate")
        report = {"schema": "ser-study2-deadline-status-1", "recorded_at": now.isoformat(),
                  "scientific_metrics_read": False, "source_identity_mismatches": mismatch,
                  "n14r": old, "study2": study, "forecast": prediction}
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"schema": "ser-study2-deadline-status-1", "recorded_at": datetime.now(UTC).isoformat(),
                  "scientific_metrics_read": False, "input_error": f"{type(error).__name__}: {error}",
                  "forecast": {"needs_attention": True,
                               "projected_compute_and_verification_end_utc": None,
                               "reasons": ["operational snapshot unavailable; retry/inspect inputs before a cloud decision"]}}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        target = args.out.resolve()
        allowed = root / "v3/data_design/evidence"
        if not target.is_relative_to(allowed) or not target.name.startswith("deadline_status_") or target.suffix != ".json":
            raise ValueError("output must be evidence/deadline_status_*.json in the new study")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

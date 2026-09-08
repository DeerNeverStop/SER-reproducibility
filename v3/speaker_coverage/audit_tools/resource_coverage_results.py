#!/usr/bin/env python3
"""Resource-only audit of accepted pilot 19 and formal 720 result seals.

Requires the existing verifier's --block all receipts for both phases. This
standard-library tool does not import the verifier, score.py, NumPy, or torch;
it never loads predictions or checkpoint tensors. It rebinds accepted DONE
maps to the current receipt/history bytes and ledger. Large artifact payloads
are not rehashed: their content verification comes from the accepted gate,
and their current existence/byte counts are checked here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics

MODELS = ("cnn", "ridge_wavlm", "wavlm_ft")
EXPECTED = {"pilot": {"cnn": 9, "ridge_wavlm": 9, "wavlm_ft": 1},
            "formal": {"cnn": 270, "ridge_wavlm": 270, "wavlm_ft": 180}}
EPOCHS = {"cnn": 100, "ridge_wavlm": 0, "wavlm_ft": 15}
FORMATS = {"cnn": "cnn_full_best_last", "ridge_wavlm": "ridge_npz", "wavlm_ft": "wavlm_delta_best_last"}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(text):
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))


def read_json(path):
    return parse(Path(path).read_text(encoding="utf-8"))


def local(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative, "invalid relative artifact path")
    p = PurePosixPath(relative)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == relative, "unsafe artifact path")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    require(target.is_relative_to(root), "artifact escapes supplied root")
    return target


def pin(path, sha):
    require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), "invalid SHA256 pin")
    require(file_sha(path) == sha, "current bytes differ from accepted identity: " + Path(path).name)


def finite_nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid resource value: " + name)
    return value


def moment(value):
    time = datetime.fromisoformat(value)
    require(time.tzinfo is not None, "event timestamp lacks timezone")
    return time.astimezone(timezone.utc)


def phase_units(plan, phase):
    if phase == "formal":
        return plan["units"]
    require(phase == "pilot", "invalid phase")
    return [u for u in plan["units"] if u["fold"] == u["rotation"] == 0 and
            (u["block"] == "core" or (u["draw"] == 0 and u["policy"] == "U" and u["seed_index"] == 0))]


def load_frozen_plan(repo, path):
    plan = read_json(path)
    require(plan.get("schema") == "ser-speaker-coverage-1" and plan.get("program") == "SER26-SPEAKER-COVERAGE-1"
            and plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "frozen plan identity mismatch")
    require(len(plan["units"]) == len({u["unit_id"] for u in plan["units"]}) == 720, "all 720 formal units required")
    for unit in plan["units"]:
        require(unit["unit_id"] == digest({k: v for k, v in unit.items() if k != "unit_id"}), "unit hash mismatch")
    for phase in EXPECTED:
        require(Counter(u["model"] for u in phase_units(plan, phase)) == EXPECTED[phase], "pilot/formal model counts differ")
    # Bind the accepted-gate implementation to the frozen source without
    # reimplementing, importing, or rerunning its scientific-design gate.
    for name in ("v3/speaker_coverage/verify.py", "v3/speaker_coverage/run.py", "v3/speaker_coverage/EXECUTION_PLAN.md"):
        pin(local(repo, name), plan["source_sha256"][name])
    return plan


def validate_gate(gate, plan, phase, units, phase_root):
    require(gate.get("schema") == "ser-speaker-coverage-all-result-seal-1" and gate.get("pass") is True
            and gate.get("all_blocks_complete") is True and gate.get("phase") == phase
            and gate.get("plan_sha256") == plan["plan_sha256"] and gate.get("count") == len(units), "accepted all-block gate identity mismatch")
    require(gate.get("seal_sha256") == digest({k: v for k, v in gate.items() if k != "seal_sha256"}), "accepted gate content seal mismatch")
    require(set(gate.get("blocks", {})) == {"core", "ft"}, "both accepted blocks required")
    closure = {}
    for block in ("core", "ft"):
        audit = gate["blocks"][block]
        expected = {u["unit_id"] for u in units if u["block"] == block}
        require(audit.get("schema") == "ser-speaker-coverage-independent-result-seal-1" and audit.get("pass") is True
                and audit.get("block_complete") is True and audit.get("block") == block and audit.get("phase") == phase
                and audit.get("plan_sha256") == plan["plan_sha256"] and audit.get("count") == len(expected)
                and set(audit.get("done_sha256", {})) == expected, "accepted block closure differs")
        require(audit.get("seal_sha256") == digest({k: v for k, v in audit.items() if k != "seal_sha256"}), "accepted block content seal mismatch")
        for name, key in (("ledger.jsonl", "ledger_sha256"), ("identity.json", "output_identity_sha256"),
                          ("plan_snapshot.json", "executed_plan_snapshot_sha256"), ("verification.json", "independent_plan_gate_receipt_sha256")):
            pin(phase_root / name, audit[key])
        closure.update(audit["done_sha256"])
    return closure


def inspect_success(unit, phase, phase_root, plan, done_sha):
    directory = local(phase_root / "units", unit["unit_id"])
    done_path = directory / "DONE"
    pin(done_path, done_sha)
    done = read_json(done_path)
    identity = {"unit_id": unit["unit_id"], "plan_sha256": plan["plan_sha256"], "phase": phase,
                "model": unit["model"], "block": unit["block"]}
    require(all(done.get(k) == v for k, v in identity.items()), "DONE resource identity differs")
    attempt = done.get("attempt")
    require(type(attempt) is int and attempt > 0, "invalid successful attempt")
    prefix = f"attempts/{attempt:04d}/"
    checkpoint = "checkpoint.npz" if unit["model"] == "ridge_wavlm" else "checkpoint.pt"
    expected = {prefix + name for name in (checkpoint, "predictions.npz", "receipt.json", "history.json")}
    require(set(done.get("artifacts", {})) == expected, "committed artifact inventory differs")
    sizes = {}
    for name in done["artifacts"]:
        target = local(directory, name)
        require(target.is_file(), "accepted artifact is now missing")
        sizes[name] = target.stat().st_size
    for name in ("receipt.json", "history.json"):
        pin(directory / prefix / name, done["artifacts"][prefix + name])
    receipt = read_json(directory / prefix / "receipt.json")
    require(receipt.get("schema") == "ser-speaker-coverage-result-1" and all(receipt.get(k) == v for k, v in identity.items())
            and receipt.get("attempt") == attempt and receipt.get("input_sha256") == digest(plan["input"])
            and receipt.get("unit_config_sha256") == digest(unit["config"]), "resource receipt does not bind the accepted unit")
    model, epochs = unit["model"], EPOCHS[unit["model"]]
    require(receipt.get("epochs_run") == epochs and (epochs == 0 or unit["config"].get("epochs") == epochs), "model fixed epoch budget differs")
    history = read_json(directory / prefix / "history.json")
    require(isinstance(history, list) and [row.get("epoch") for row in history] == list(range(1, epochs + 1)), "resource history epoch sequence differs")
    for row in history:
        require(set(row) == {"epoch", "train_loss", "val_loss", "optimizer_steps"}, "unexpected history field; resource audit excludes scientific metrics")
        finite_nonnegative(row["val_loss"], "validation loss")
        require(type(row["optimizer_steps"]) is int and row["optimizer_steps"] == math.ceil(len(unit["fit"]) / unit["config"]["batch_size"]),
                "optimizer step count differs from fixed budget")
    best = min(history, key=lambda row: row["val_loss"])["epoch"] if history else None
    require(receipt.get("best_epoch") == best and receipt.get("checkpoint_reload_verified") is True
            and receipt.get("checkpoint_format") == FORMATS[model], "best/last checkpoint receipt is incomplete")
    reload_keys = ("checkpoint_reload_best_max_abs_diff", "checkpoint_reload_last_max_abs_diff", "checkpoint_reload_max_abs_diff")
    for key in reload_keys:
        finite_nonnegative(receipt.get(key), key)
    require(receipt[reload_keys[2]] == max(receipt[k] for k in reload_keys[:2]), "best/last reload maximum is inconsistent")
    fit = finite_nonnegative(receipt.get("fit_seconds"), "fit_seconds")
    wall = finite_nonnegative(receipt.get("wall_seconds"), "wall_seconds")
    peak = finite_nonnegative(receipt.get("peak_cuda_bytes"), "peak_cuda_bytes")
    require(type(peak) is int, "CUDA allocated byte count must be an integer")
    require(wall + 1e-6 >= fit, "unit wall seconds are shorter than fit seconds")
    environment = receipt.get("environment", {})
    require(environment.get("device") in ("cpu", "cuda"), "unknown execution device")
    require(isinstance(environment.get("fit_timing"), str) and isinstance(environment.get("wall_timing"), str), "missing timing definitions")
    finished = moment(receipt["finished_at"])
    return {"phase": phase, "model": model, "block": unit["block"], "unit_id": unit["unit_id"], "attempt": attempt,
            "epochs_run": epochs, "best_epoch": best, "optimizer_steps": sum(row["optimizer_steps"] for row in history),
            "fit_seconds": fit, "unit_wall_seconds": wall, "device": environment["device"],
            "gpu": environment.get("gpu"), "peak_cuda_allocated_bytes": peak,
            "checkpoint_reload_best_max_abs_diff": receipt[reload_keys[0]], "checkpoint_reload_last_max_abs_diff": receipt[reload_keys[1]],
            "checkpoint_reload_max_abs_diff": receipt[reload_keys[2]], "checkpoint_format": receipt["checkpoint_format"],
            "fit_timing": environment["fit_timing"], "wall_timing": environment["wall_timing"],
            "finished_at": finished.isoformat(), "artifact_bytes": sum(sizes.values()), "done_bytes": done_path.stat().st_size,
            "checkpoint_bytes": sizes[prefix + checkpoint], "predictions_bytes": sizes[prefix + "predictions.npz"],
            "receipt_history_bytes": sizes[prefix + "receipt.json"] + sizes[prefix + "history.json"], "done_sha256": done_sha,
            "receipt_sha256": done["artifacts"][prefix + "receipt.json"], "history_sha256": done["artifacts"][prefix + "history.json"]}


def ledger_attempts(phase_root, units, successful_rows):
    raw = (phase_root / "ledger.jsonl").read_bytes()
    require(raw and raw.endswith(b"\n"), "partial runtime ledger")
    known = {u["unit_id"]: u for u in units}
    observations, times, invocations = {}, [], 0
    for line in raw.decode("utf-8").splitlines():
        row = parse(line)
        when = moment(row["at"])
        times.append(when)
        event = row["event"]
        if event == "invocation_start":
            invocations += 1
        if event not in ("unit_start", "unit_done", "unit_failed"):
            continue  # Ordering/identity/stale-lock rules already passed the pinned gate.
        key = row["unit_id"], row["attempt"]
        require(key[0] in known, "ledger includes an unknown planned unit")
        if event == "unit_start":
            require(key not in observations, "attempt start is duplicated")
            observations[key] = {"model": known[key[0]]["model"], "unit_id": key[0], "attempt": key[1],
                                 "status": "no_terminal_event", "start_at": when.isoformat(), "terminal_at": None,
                                 "observed_event_span_seconds": None, "fit_seconds": None}
        else:
            require(key in observations and observations[key]["status"] == "no_terminal_event", "attempt terminal event is inconsistent")
            entry = observations[key]
            span = (when - moment(entry["start_at"])).total_seconds()
            require(span >= 0, "negative attempt event span; wall clock is inconsistent")
            entry.update(status="success" if event == "unit_done" else "failed", terminal_at=when.isoformat(),
                         observed_event_span_seconds=span, recovered_from_done=bool(row.get("recovered_from_done", False)))
            if event == "unit_done":
                entry["fit_seconds"] = finite_nonnegative(row["fit_seconds"], "ledger fit_seconds")
            else:
                entry["error_type"] = row.get("error_type", "unrecorded")
    successful = {(r["unit_id"], r["attempt"]): r for r in successful_rows}
    require({k for k, v in observations.items() if v["status"] == "success"} == successful.keys(), "successful ledger and accepted DONE attempts differ")
    for key, receipt in successful.items():
        require(observations[key]["fit_seconds"] == receipt["fit_seconds"], "ledger fit duration differs from receipt")
    entries = list(observations.values())
    return entries, {"invocations_started": invocations, "first_recorded_event_at": min(times).isoformat(),
                     "last_recorded_event_at": max(times).isoformat(),
                     "event_envelope_seconds": (max(times) - min(times)).total_seconds(),
                     "event_envelope_scope": "first-to-last ledger timestamps; includes gaps, may overlap other phases, excludes pre-gate setup; never equated with GPU rental duration"}


def distribution(values, *, include_sum=True):
    values = list(values)
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None, **({"sum": 0.0} if include_sum else {})}
    return {"count": len(values), "minimum": min(values), "median": statistics.median(values),
            "maximum": max(values), **({"sum": math.fsum(values)} if include_sum else {})}


def summarize(rows, attempts):
    failed = [a for a in attempts if a["status"] == "failed"]
    incomplete = [a for a in attempts if a["status"] == "no_terminal_event"]
    successes = [a for a in attempts if a["status"] == "success"]
    return {"successful_units": len(rows), "started_attempts": len(attempts), "successful_attempts": len(successes),
            "failed_attempts": len(failed), "attempts_without_terminal_event": len(incomplete),
            "successful_units_requiring_retry": sum(r["attempt"] > 1 for r in rows),
            "recovered_success_ledger_events": sum(a.get("recovered_from_done", False) for a in successes),
            "epochs_run_total": sum(r["epochs_run"] for r in rows), "optimizer_steps_total": sum(r["optimizer_steps"] for r in rows),
            "optimizer_step_count_scope": "runner counter of completed batches/update attempts; FT GradScaler may skip an effective optimizer step on overflow",
            "effective_optimizer_steps_total": None,
            "fit_wall_seconds": distribution(r["fit_seconds"] for r in rows),
            "successful_unit_wall_seconds": distribution(r["unit_wall_seconds"] for r in rows),
            "summed_successful_fit_hours": math.fsum(r["fit_seconds"] for r in rows) / 3600,
            "summed_cuda_successful_fit_wall_seconds": math.fsum(r["fit_seconds"] for r in rows if r["device"] == "cuda"),
            "summed_cpu_successful_fit_wall_seconds": math.fsum(r["fit_seconds"] for r in rows if r["device"] == "cpu"),
            "peak_cuda_allocated_bytes": distribution((r["peak_cuda_allocated_bytes"] for r in rows), include_sum=False),
            "peak_cuda_allocated_bytes_scope": "torch.cuda.max_memory_allocated per successful unit; not total device VRAM or reserved memory",
            "gpu_names": sorted({r["gpu"] for r in rows if r["gpu"] is not None}),
            "best_epoch": distribution((r["best_epoch"] for r in rows if r["best_epoch"] is not None), include_sum=False),
            "all_best_last_reloads_verified": True, "largest_checkpoint_reload_max_abs_diff": max((r["checkpoint_reload_max_abs_diff"] for r in rows), default=None),
            "successful_artifact_bytes": sum(r["artifact_bytes"] for r in rows), "done_bytes": sum(r["done_bytes"] for r in rows),
            "checkpoint_bytes": sum(r["checkpoint_bytes"] for r in rows), "predictions_bytes": sum(r["predictions_bytes"] for r in rows),
            "receipt_history_bytes": sum(r["receipt_history_bytes"] for r in rows),
            "failed_attempt_observed_event_span_seconds": distribution(a["observed_event_span_seconds"] for a in failed),
            "failed_attempt_fit_seconds": None, "unclosed_attempt_cost": None,
            "failure_time_scope": "start-to-failure wall-clock event spans only; failed fit timers were not saved; unclosed attempts have unknown duration",
            "fit_timing_definitions": sorted({r["fit_timing"] for r in rows}), "unit_wall_timing_definitions": sorted({r["wall_timing"] for r in rows})}


def audit_phase(plan, phase, units, outroot, gate):
    phase_root = Path(outroot) / phase
    closure = validate_gate(gate, plan, phase, units, phase_root)
    rows = [inspect_success(u, phase, phase_root, plan, closure[u["unit_id"]]) for u in units]
    for block in ("core", "ft"):
        accepted = gate["blocks"][block]
        selected = [r for r in rows if r["block"] == block]
        require(sum(r["artifact_bytes"] for r in selected) == accepted["artifact_bytes_checked"], "artifact byte count changed since accepted gate")
        require(math.isclose(math.fsum(r["fit_seconds"] for r in selected), accepted["summed_successful_fit_seconds"], rel_tol=1e-12, abs_tol=1e-8),
                "successful fit total differs from accepted gate")
        require(math.isclose(math.fsum(r["unit_wall_seconds"] for r in selected), accepted["summed_successful_unit_wall_seconds"], rel_tol=1e-12, abs_tol=1e-8),
                "successful unit wall total differs from accepted gate")
    attempts, envelope = ledger_attempts(phase_root, units, rows)
    report = {"phase": phase, "accepted_gate_seal_sha256": gate["seal_sha256"], "done_mapping_sha256": digest(closure),
              "summary": summarize(rows, attempts), "ledger_envelope": envelope,
              "by_model": {m: summarize([r for r in rows if r["model"] == m], [a for a in attempts if a["model"] == m]) for m in MODELS},
              "non_successful_attempts": [a for a in attempts if a["status"] != "success"], "per_successful_unit": rows}
    # Catch receipt, history, DONE, and ledger changes during aggregation.
    for row in rows:
        directory = phase_root / "units" / row["unit_id"]
        pin(directory / "DONE", row["done_sha256"])
        for name, key in (("receipt.json", "receipt_sha256"), ("history.json", "history_sha256")):
            pin(directory / f"attempts/{row['attempt']:04d}" / name, row[key])
    pin(phase_root / "ledger.jsonl", gate["blocks"]["core"]["ledger_sha256"])
    return report, rows, attempts


def audit(repo, plan_path, outroot, pilot_gate_path, formal_gate_path):
    repo, plan_path, outroot = map(lambda p: Path(p).resolve(), (repo, plan_path, outroot))
    plan_file_sha = file_sha(plan_path)
    plan = load_frozen_plan(repo, plan_path)
    phases, all_rows, all_attempts, gate_inputs = {}, [], [], {}
    for phase, path in (("pilot", pilot_gate_path), ("formal", formal_gate_path)):
        gate_sha = file_sha(path)
        phases[phase], rows, attempts = audit_phase(plan, phase, phase_units(plan, phase), outroot, read_json(path))
        pin(path, gate_sha)
        gate_inputs[phase] = gate_sha
        all_rows.extend(rows)
        all_attempts.extend(attempts)
    require(len(all_rows) == 739 and len(phases["pilot"]["per_successful_unit"]) == 19
            and len(phases["formal"]["per_successful_unit"]) == 720, "pilot and formal must remain separate complete populations")
    pin(plan_path, plan_file_sha)
    result = {"schema": "ser-coverage-resource-audit-1", "pass": True, "plan_sha256": plan["plan_sha256"],
              "plan_file_sha256": plan_file_sha, "source_sha256": file_sha(__file__), "accepted_gate_file_sha256": gate_inputs,
              "phases": phases, "combined_successful_runs": 739, "combined_summary": summarize(all_rows, all_attempts),
              "combined_by_model": {m: summarize([r for r in all_rows if r["model"] == m], [a for a in all_attempts if a["model"] == m]) for m in MODELS},
              "timing_distinctions": {
                  "successful_fit_seconds": "sum of per-unit monotonic fit_seconds; includes model construction, full training, prediction and checkpoint save/reload/replay",
                  "successful_unit_wall_seconds": "sum of per-unit wall_seconds measured before result receipt writing; excludes initial shared integrity gate",
                  "gpu_utilization_seconds": None, "gpu_rental_elapsed_seconds": None, "actual_billed_amount": None,
                  "billing_currency": None, "rental_and_bill_status": "not evidenced by result receipts or ledger; no price or duration was inferred",
                  "combined_event_envelope": None, "phase_envelopes_must_not_be_added_as_rental_time": True},
              "scope_excludes": ["ASV extraction", "TTS pilot", "shared feature generation", "initial verification/setup", "cloud idle periods",
                                 "storage/network charges", "historical experiments outside these pilot/formal units"],
              "artifact_integrity_scope": "accepted full result seal plus current DONE/receipt/history/ledger hashes and current artifact byte counts; large prediction/checkpoint payloads not rehashed here",
              "accepted_gate_trust_boundary": "provided locally accepted content-hashed audit receipts; no external signature or fresh tensor replay is claimed",
              "prediction_arrays_loaded": False, "checkpoint_tensors_loaded": False, "scientific_metrics_read_or_computed": False,
              "new_fits_started": 0, "network_used": False}
    result["audit_sha256"] = digest(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "outroot", "pilot-gate", "formal-gate", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit(args.repo, args.plan, args.outroot, args.pilot_gate, args.formal_gate)
    except Exception as error:
        report = {"schema": "ser-coverage-resource-audit-1", "pass": False, "error_type": type(error).__name__, "error": str(error),
                  "source_sha256": file_sha(__file__), "prediction_arrays_loaded": False, "scientific_metrics_read_or_computed": False}
    output = args.out.resolve()
    require(not output.is_relative_to(args.repo.resolve()) and not output.is_relative_to(args.outroot.resolve())
            and output not in {p.resolve() for p in (args.plan, args.pilot_gate, args.formal_gate)}, "resource audit must not replace input evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    report["audited_at"] = datetime.now(timezone.utc).isoformat()
    report.pop("audit_sha256", None)
    report["audit_sha256"] = digest(report)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pass": report["pass"], "audit": str(output), "scientific_metrics_read_or_computed": False}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

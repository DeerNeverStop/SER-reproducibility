"""Outcome-blind global closure for the four-node N14R2 execution.

This module never parses unit.json, history.json, predictions.csv, or the
trainer DONE body.  It replays authorization and attempt ledgers, validates
outcome-free receipts, and hashes opaque artifacts.  Only a fully valid set of
24 complete draws can produce ``analysis_lock.json``.  A pause or blocked draw
is nonterminal; terminal incompleteness requires all 28 eligible draws to have
resolved without reaching 24 complete draws.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from . import make_pins


STUDY_ID = "N14R2"
SPEC_VERSION = "2.0.0"
N_DRAWS = 28
N_PRIMARY = 24
N_NODES = 4
UNITS_PER_DRAW = 80
ARTIFACT_NAMES = ("unit.json", "history.json", "predictions.csv", "DONE")
PLAN_HASH_NAMES = ("run_plan.csv", "split_index.json", "unit_configs.json", "hygiene.json", "plan_summary.json")
COMPLETION_SCHEMA = "ser26-n14r2-completion-1"
LOCK_SCHEMA = "ser26-n14r2-analysis-lock-1"
RECEIPT_SCHEMA = "ser26-n14r2-artifact-receipt-1"
DRAW_SCHEMA = "ser26-n14r2-node-draw-1"
FAILURE_CLASSES = {
    "training", "cuda", "infrastructure", "worker_exit", "timeout",
    "interrupted", "circuit_cancelled", "integrity",
}
TERMINAL_RUN_STATUSES = {"ready_for_closure", "not_tested_incomplete", "complete"}


class ClosureError(ValueError):
    """Execution provenance cannot safely be closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"cannot read JSON receipt {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    raise ClosureError(f"blank ledger line {line_no}: {path}")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ClosureError(f"malformed ledger line {line_no}: {path}") from exc
                if not isinstance(event, dict):
                    raise ClosureError(f"ledger line {line_no} is not an object: {path}")
                events.append(event)
    except OSError as exc:
        raise ClosureError(f"cannot read ledger {path}: {exc}") from exc
    return events


def _atomic_immutable_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ClosureError(f"refusing to replace immutable closure receipt: {path}")
        return
    fd, raw_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_plan(plan_dir: Path) -> tuple[list[dict[str, str]], dict[int, list[dict[str, str]]]]:
    path = plan_dir / "run_plan.csv"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ClosureError(f"cannot load frozen plan: {exc}") from exc
    required = {
        "study_id", "spec_version", "unit_id", "draw_id", "draw_role", "analysis_order",
        "fold", "train_rep", "cell", "config_index", "manifest_sha256", "config_sha256",
        "split_sha256", "train_seed", "status", "arm",
    }
    if not rows or not required.issubset(rows[0]):
        raise ClosureError("run_plan.csv is empty or lacks closure identity fields")
    if len(rows) != N_DRAWS * UNITS_PER_DRAW or len({row["unit_id"] for row in rows}) != len(rows):
        raise ClosureError("plan must contain 2,240 unique units")
    by_draw: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            draw = int(row["draw_id"])
            order = int(row["analysis_order"])
        except (TypeError, ValueError) as exc:
            raise ClosureError("invalid plan draw/order") from exc
        role = "primary" if draw < N_PRIMARY else "reserve"
        if (row["study_id"] != STUDY_ID or row["spec_version"] != SPEC_VERSION
                or row["arm"] != STUDY_ID or row["status"] != "pending"
                or draw not in range(N_DRAWS) or order != draw or row["draw_role"] != role):
            raise ClosureError(f"frozen plan identity changed for {row['unit_id']}")
        by_draw[draw].append(row)
    if set(by_draw) != set(range(N_DRAWS)) or any(len(items) != UNITS_PER_DRAW for items in by_draw.values()):
        raise ClosureError("plan draw grid must be 28 draws of 80 units")
    return rows, dict(by_draw)


def _validate_event_sequence(
    events: list[dict[str, Any]], *, node_id: str | None = None, require_chain: bool = False,
) -> None:
    if not events:
        raise ClosureError("ledger is empty")
    chained = ["event_sha256" in event or "prev_event_sha256" in event for event in events]
    if any(chained) and not all(chained):
        raise ClosureError("ledger hash-chain fields are only partially present")
    if require_chain and not all(chained):
        raise ClosureError("global ledger must be hash chained")
    previous: str | None = None
    for index, event in enumerate(events, 1):
        if event.get("seq") != index or event.get("study_id") != STUDY_ID:
            raise ClosureError(f"ledger sequence/study mismatch at event {index}")
        if node_id is not None and event.get("node_id") != node_id:
            raise ClosureError(f"node ledger identity mismatch at event {index}")
        if all(chained):
            actual = event.get("event_sha256")
            payload = {key: value for key, value in event.items() if key != "event_sha256"}
            if event.get("prev_event_sha256") != previous or actual != sha256_json(payload):
                raise ClosureError(f"ledger hash chain mismatch at event {index}")
            previous = actual


def validate_global_ledger(
    events: list[dict[str, Any]], by_draw: dict[int, list[dict[str, str]]],
    *, plan_run_sha256: str | None = None, pins_sha256: str | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    _validate_event_sequence(events, require_chain=True)
    if events[-1].get("event") != "run_stop" or events[-1].get("status") not in TERMINAL_RUN_STATUSES:
        raise ClosureError("global ledger lacks a terminal run_stop")
    if events[0].get("event") != "run_start" or sum(e.get("event") == "run_start" for e in events) != 1:
        raise ClosureError("global ledger must begin with exactly one run_start")
    if sum(e.get("event") == "run_stop" for e in events) != 1:
        raise ClosureError("global ledger contains an early or duplicate run_stop")
    if (plan_run_sha256 is not None and events[0].get("plan_run_sha256") != plan_run_sha256
            or pins_sha256 is not None and events[0].get("pins_sha256") != pins_sha256):
        raise ClosureError("global run_start is not bound to frozen plan/PINS")
    start = events[0]
    if (not re.fullmatch(r"[0-9a-f]{40}", str(start.get("preregistration_commit", "")))
            or not str(start.get("preregistration_tag", "")).startswith("SER26-N14R2-prereg-")
            or not str(start.get("public_receipt_url", "")).startswith(
                "https://github.com/DeerNeverStop/SER/pull/6#issuecomment-"
            ) or not isinstance(start.get("public_receipt_created_at"), str)
            or start.get("contains_outcome_statistics") is not False):
        raise ClosureError("global run_start lacks the prospective public registration receipt")
    authorized: dict[int, dict[str, Any]] = {}
    terminal: dict[int, dict[str, Any]] = {}
    reserve_order: list[int] = []
    registered: set[str] = set()
    recovery_ids: set[str] = set()
    for event in events:
        kind = event.get("event")
        if kind == "draw_authorized":
            try:
                draw = int(event["draw_id"])
                remainder = int(event["shard_remainder"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ClosureError("malformed global draw authorization") from exc
            role = "primary" if draw < N_PRIMARY else "reserve"
            if (draw not in by_draw or draw in authorized or event.get("draw_role") != role
                    or event.get("node_id") != f"node{draw % N_NODES}" or remainder != draw % N_NODES
                    or not isinstance(event.get("authorization_id"), str)
                    or not event["authorization_id"]):
                raise ClosureError(f"invalid or duplicate draw authorization: {draw}")
            if (plan_run_sha256 is not None and event.get("plan_run_sha256") != plan_run_sha256
                    or pins_sha256 is not None and event.get("pins_sha256") != pins_sha256):
                raise ClosureError(f"draw authorization is not bound to frozen inputs: {draw}")
            if draw < N_PRIMARY:
                if reserve_order:
                    raise ClosureError("primary draw was authorized after a reserve")
            else:
                expected = N_PRIMARY + len(reserve_order)
                if draw != expected:
                    raise ClosureError("reserves were not authorized in global ascending order")
                if set(range(N_PRIMARY)) - set(terminal):
                    raise ClosureError("reserve authorized before all primary draws were terminal")
                if any(terminal[d].get("status") not in {"complete", "void", "void_training_failure"}
                       for d in range(N_PRIMARY)):
                    raise ClosureError("reserve authorized while a primary draw was blocked")
                prior_voids = sum(
                    event_.get("status") in {"void", "void_training_failure"}
                    for event_ in terminal.values()
                )
                if prior_voids < draw - (N_PRIMARY - 1):
                    raise ClosureError("reserve authorized without enough prior training-failure voids")
                if draw > N_PRIMARY and draw - 1 not in terminal:
                    raise ClosureError("reserve authorized before the preceding reserve was terminal")
                if sum(event_.get("status") == "complete" for event_ in terminal.values()) >= N_PRIMARY:
                    raise ClosureError("reserve authorized after 24 complete draws already existed")
                reserve_order.append(draw)
            authorized[draw] = event
        elif kind == "draw_terminal":
            try:
                draw = int(event["draw_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ClosureError("malformed global draw terminal event") from exc
            if draw not in authorized or draw in terminal:
                raise ClosureError(f"draw terminal is unauthorized or duplicated: {draw}")
            if (event.get("node_id") != f"node{draw % N_NODES}"
                    or event.get("status") not in {"complete", "void", "void_training_failure"}
                    or not isinstance(event.get("node_draw_receipt_sha256"), str)):
                raise ClosureError(f"invalid global draw terminal event: {draw}")
            terminal[draw] = event
        elif kind == "node_recovery_authorized":
            auth_id = event.get("authorization_id")
            node_id = event.get("node_id")
            if (not isinstance(auth_id, str) or not auth_id or auth_id in recovery_ids
                    or node_id not in {f"node{i}" for i in range(N_NODES)}
                    or event.get("shard_remainder") != int(node_id[-1])
                    or not isinstance(event.get("prior_node_ledger_sha256"), str)
                    or len(event["prior_node_ledger_sha256"]) != 64
                    or (plan_run_sha256 is not None and event.get("plan_run_sha256") != plan_run_sha256)
                    or (pins_sha256 is not None and event.get("pins_sha256") != pins_sha256)):
                raise ClosureError("invalid node recovery authorization")
            recovery_ids.add(auth_id)
        elif kind == "node_registered":
            node_id = event.get("node_id")
            if (node_id not in {f"node{i}" for i in range(N_NODES)} or node_id in registered
                    or event.get("shard_remainder") != int(node_id[-1])):
                raise ClosureError("invalid or duplicate node registration")
            registered.add(node_id)
        elif kind not in {"run_start", "node_registered", "circuit_break", "run_stop"}:
            raise ClosureError(f"unknown global ledger event: {kind!r}")
    if set(range(N_PRIMARY)) - set(authorized):
        raise ClosureError("all 24 primary draws must be authorized before terminal closure")
    if registered != {f"node{i}" for i in range(N_NODES)}:
        raise ClosureError("global ledger lacks exactly four node registrations")
    return authorized, terminal, events[-1]


def _safe_attempt_path(run_dir: Path, raw: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts:
        raise ClosureError(f"unsafe attempt path in receipt: {raw!r}")
    path = run_dir.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ClosureError(f"attempt path escapes run root: {raw!r}") from exc
    return path


def _validate_attempt_receipt(
    receipt: Any, row: dict[str, str], node_id: str, attempt: int, attempt_dir: Path,
) -> dict[str, str]:
    if not isinstance(receipt, dict):
        raise ClosureError("attempt receipt is not an object")
    required = {
        "schema", "study_id", "node_id", "shard_remainder", "unit_id", "draw_id",
        "attempt", "status", "failure_class", "plan_row_sha256", "artifact_hashes",
        "started_at", "finished_at",
    }
    if set(receipt) != required:
        raise ClosureError(f"attempt receipt fields changed: {row['unit_id']} attempt {attempt}")
    expected_identity = (
        receipt.get("schema") == RECEIPT_SCHEMA,
        receipt.get("study_id") == STUDY_ID,
        receipt.get("node_id") == node_id,
        receipt.get("shard_remainder") == int(row["draw_id"]) % N_NODES,
        receipt.get("unit_id") == row["unit_id"],
        receipt.get("draw_id") == int(row["draw_id"]),
        receipt.get("attempt") == attempt,
        receipt.get("plan_row_sha256") == sha256_json(row),
    )
    if not all(expected_identity):
        raise ClosureError(f"attempt receipt identity mismatch: {row['unit_id']} attempt {attempt}")
    if not isinstance(receipt.get("started_at"), str) or not isinstance(receipt.get("finished_at"), str):
        raise ClosureError(f"attempt receipt timestamps are missing: {row['unit_id']} attempt {attempt}")
    status, failure_class = receipt.get("status"), receipt.get("failure_class")
    artifacts = receipt.get("artifact_hashes")
    if status == "done":
        if failure_class is not None or not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
            raise ClosureError(f"completed attempt receipt is malformed: {row['unit_id']}")
        observed: dict[str, str] = {}
        for name in ARTIFACT_NAMES:
            path = attempt_dir / name
            if not path.is_file():
                raise ClosureError(f"completed attempt artifact is missing: {path}")
            observed[name] = sha256_file(path)
        if observed != artifacts:
            raise ClosureError(f"completed attempt artifact hash mismatch: {row['unit_id']}")
        return observed
    if status != "failed" or failure_class not in FAILURE_CLASSES:
        raise ClosureError(f"failed attempt classification is invalid: {row['unit_id']}")
    if artifacts not in ({}, None):
        raise ClosureError(f"failed attempt unexpectedly declares scientific artifacts: {row['unit_id']}")
    return {}


def validate_node(
    run_dir: Path, node_id: str, rows_by_id: dict[str, dict[str, str]],
    authorized: set[int] | dict[int, dict[str, Any]],
    recovery_authorizations: dict[str, dict[str, Any]] | None = None,
    runner_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node_dir = run_dir / "nodes" / node_id
    ledger_path = node_dir / "ledger.jsonl"
    environment_path = node_dir / "execution_environment.json"
    if not environment_path.is_file():
        raise ClosureError(f"missing execution environment for {node_id}")
    events = read_jsonl(ledger_path)
    _validate_event_sequence(events, node_id=node_id)
    raw_lines = ledger_path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != len(events) or any(not line.endswith(b"\n") for line in raw_lines):
        raise ClosureError(f"{node_id}: ledger must be newline-terminated JSONL")
    prefix_hashes: list[str] = []
    prefix = b""
    for line in raw_lines:
        prefix_hashes.append(hashlib.sha256(prefix).hexdigest())
        prefix += line
    states: dict[str, str] = {}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    open_attempts: dict[str, int] = {}
    blocked_invocations: set[str] = set()
    failed_invocations: set[str] = set()
    ready_slots: dict[str, set[int]] = defaultdict(set)
    current_invocation: str | None = None
    previous_event: dict[str, Any] | None = None
    used_recoveries: set[str] = set()
    enforce_recovery = recovery_authorizations is not None
    recoveries = recovery_authorizations or {}
    for event_index, event in enumerate(events):
        kind = event.get("event")
        invocation = event.get("invocation_id")
        if not isinstance(invocation, str) or not invocation:
            raise ClosureError(f"{node_id}: ledger event lacks invocation_id")
        if kind == "runner_start":
            if current_invocation is not None and invocation == current_invocation:
                raise ClosureError(f"{node_id}: runner invocation ID was reused")
            if current_invocation is not None:
                clean = (previous_event is not None and previous_event.get("event") == "runner_stop"
                         and previous_event.get("status") in {"authorized_draws_terminal", "paused_operator"})
                recovery_id = event.get("recovery_authorization_id")
                prior_hash = event.get("prior_node_ledger_sha256")
                if clean:
                    if recovery_id is not None or prior_hash is not None:
                        raise ClosureError(f"{node_id}: clean restart must not claim recovery authorization")
                else:
                    recovery = recoveries.get(str(recovery_id))
                    if not enforce_recovery:
                        if not isinstance(recovery_id, str) or not recovery_id or not isinstance(prior_hash, str):
                            raise ClosureError(f"{node_id}: restart lacks recovery identity")
                    elif recovery is None or recovery_id in used_recoveries:
                        raise ClosureError(f"{node_id}: restart lacks exact global recovery authorization")
                    else:
                        source_hash = recovery.get("prior_node_ledger_sha256")
                        boundaries = [i for i in range(event_index + 1) if prefix_hashes[i] == source_hash]
                        allowed_reconciliation = {"done", "failed", "circuit_break"}
                        if (recovery.get("node_id") != node_id
                                or recovery.get("shard_remainder") != int(node_id[-1])
                                or prior_hash != source_hash or len(boundaries) != 1
                                or any(item.get("event") not in allowed_reconciliation
                                       for item in events[boundaries[0]:event_index])):
                            raise ClosureError(f"{node_id}: recovery prefix/reconciliation is invalid")
                        used_recoveries.add(str(recovery_id))
            elif event.get("recovery_authorization_id") is not None or event.get("prior_node_ledger_sha256") is not None:
                raise ClosureError(f"{node_id}: initial invocation must not claim recovery authorization")
            current_invocation = invocation
            if runner_bindings is not None:
                global_sha = event.get("global_ledger_sha256")
                expected_authorizations = {
                    str(draw): auth_id
                    for draw, auth_id in runner_bindings.get("authorizations_by_prefix", {}).get(
                        global_sha, {}
                    ).items()
                    if int(draw) % N_NODES == int(node_id[-1])
                }
                if (event.get("plan_run_sha256") != runner_bindings.get("plan_run_sha256")
                        or event.get("pins_sha256") != runner_bindings.get("pins_sha256")
                        or event.get("execution_environment_sha256") != runner_bindings.get("environment_sha256")
                        or global_sha not in runner_bindings.get("global_prefix_hashes", set())
                        or event.get("authorization_ids") != expected_authorizations
                        or event.get("workers") != 8):
                    raise ClosureError(f"{node_id}: runner_start provenance binding mismatch")
            previous_event = event
            continue
        if current_invocation is None:
            raise ClosureError(f"{node_id}: event precedes runner_start")
        if invocation != current_invocation:
            raise ClosureError(f"{node_id}: invocation ID changed without runner_start")
        if kind == "circuit_break":
            blocked_invocations.add(invocation)
            previous_event = event
            continue
        if kind == "worker_ready":
            slot = event.get("slot")
            if (type(slot) is not int or slot not in range(8) or slot in ready_slots[invocation]
                    or event.get("cuda_healthy") is not True or event.get("health_checksum") != 4096.0
                    or (runner_bindings is not None
                        and event.get("backend_precision") != runner_bindings.get("backend_precision"))):
                raise ClosureError(f"{node_id}: invalid or duplicate worker health receipt")
            ready_slots[invocation].add(slot)
            previous_event = event
            continue
        if kind == "runner_stop":
            if event.get("status") not in {"authorized_draws_terminal", "paused_operator"}:
                raise ClosureError(f"{node_id}: invalid clean runner_stop status")
            previous_event = event
            continue
        if kind not in {"start", "done", "failed"}:
            raise ClosureError(f"{node_id}: unknown ledger event {kind!r}")
        uid = event.get("unit_id")
        row = rows_by_id.get(str(uid))
        try:
            draw, attempt = int(event["draw_id"]), int(event["attempt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClosureError(f"{node_id}: malformed attempt event") from exc
        if (row is None or draw != int(row["draw_id"]) or draw % N_NODES != int(node_id[-1])
                or draw not in authorized or attempt not in (1, 2)):
            raise ClosureError(f"{node_id}: attempt event violates plan/shard/authorization")
        if kind == "start":
            expected_auth = authorized.get(draw) if isinstance(authorized, dict) else None
            if (expected_auth is not None
                    and (event.get("authorization_id") != expected_auth.get("authorization_id")
                         or event.get("plan_row_sha256") != sha256_json(row))):
                raise ClosureError(f"{node_id}: start is not bound to draw authorization/plan row")
            if len(ready_slots[invocation]) != 8:
                raise ClosureError(f"{node_id}: dispatch preceded eight healthy workers")
            if invocation in blocked_invocations or invocation in failed_invocations:
                raise ClosureError(f"{node_id}: dispatch occurred after circuit break in one invocation")
            if uid in open_attempts or states.get(str(uid)) == "done" or attempt != len(attempts[str(uid)]) + 1:
                raise ClosureError(f"{node_id}: invalid or third attempt start for {uid}")
            open_attempts[str(uid)] = attempt
            previous_event = event
            continue
        if open_attempts.get(str(uid)) != attempt:
            raise ClosureError(f"{node_id}: unmatched terminal event for {uid}")
        attempt_dir = node_dir / "units" / str(uid) / f"attempt_{attempt:02d}"
        receipt_path = attempt_dir / "artifact_receipt.json"
        if not receipt_path.is_file() or event.get("receipt_sha256") != sha256_file(receipt_path):
            raise ClosureError(f"{node_id}: terminal event/attempt receipt hash mismatch for {uid}")
        receipt = read_json(receipt_path)
        artifact_hashes = _validate_attempt_receipt(receipt, row, node_id, attempt, attempt_dir)
        if receipt.get("status") != kind:
            raise ClosureError(f"{node_id}: ledger/receipt terminal status mismatch for {uid}")
        record = {
            "attempt": attempt,
            "status": kind,
            "failure_class": receipt.get("failure_class"),
            "artifact_receipt_sha256": sha256_file(receipt_path),
            "attempt_path": attempt_dir.relative_to(run_dir).as_posix(),
            "artifact_hashes": artifact_hashes,
        }
        attempts[str(uid)].append(record)
        del open_attempts[str(uid)]
        if kind == "done":
            if states.get(str(uid)) == "done":
                raise ClosureError(f"{node_id}: duplicate completed unit {uid}")
            states[str(uid)] = "done"
            pointer_path = node_dir / "units" / str(uid) / "DONE.json"
            pointer = read_json(pointer_path)
            if (not isinstance(pointer, dict)
                    or set(pointer) != {"attempt", "artifact_receipt_sha256"}
                    or int(pointer.get("attempt", -1)) != attempt
                    or pointer.get("artifact_receipt_sha256") != record["artifact_receipt_sha256"]):
                raise ClosureError(f"{node_id}: immutable DONE pointer mismatch for {uid}")
            record["done_pointer_sha256"] = sha256_file(pointer_path)
        else:
            states[str(uid)] = "failed"
            failed_invocations.add(invocation)
            if receipt.get("failure_class") != "training" and invocation not in blocked_invocations:
                # The circuit-break can be written after concurrent terminal
                # messages, but must exist for the same invocation by closure.
                pass
        previous_event = event
    if open_attempts:
        raise ClosureError(f"{node_id}: terminal closure has orphan starts")
    infra_invocations = {
        event["invocation_id"] for event in events
        if event.get("event") == "failed" and event.get("failure_class") != "training"
    }
    if not infra_invocations.issubset(blocked_invocations):
        raise ClosureError(f"{node_id}: infrastructure/integrity failure lacks circuit break")
    units_dir = node_dir / "units"
    if units_dir.exists():
        actual_units = {path.name for path in units_dir.iterdir() if path.is_dir()}
        if not actual_units.issubset(rows_by_id):
            raise ClosureError(f"{node_id}: unknown unit directory exists")
        for uid in actual_units:
            actual_attempts = {
                int(path.name[-2:]) for path in (units_dir / uid).glob("attempt_[0-9][0-9]") if path.is_dir()
            }
            if actual_attempts != {item["attempt"] for item in attempts.get(uid, [])}:
                raise ClosureError(f"{node_id}: attempt directory/ledger inventory mismatch for {uid}")
    return {
        "node_id": node_id,
        "ledger_path": ledger_path,
        "ledger_sha256": sha256_file(ledger_path),
        "environment_path": environment_path,
        "environment_sha256": sha256_file(environment_path),
        "attempts": dict(attempts),
        "states": states,
        "used_recovery_authorization_ids": sorted(used_recoveries),
        "last_event": events[-1],
    }


def validate_draw_receipt(
    run_dir: Path, draw: int, event: dict[str, Any], draw_rows: list[dict[str, str]],
    node: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    node_id = f"node{draw % N_NODES}"
    path = run_dir / "nodes" / node_id / "draws" / f"draw_{draw:02d}.json"
    if not path.is_file() or sha256_file(path) != event.get("node_draw_receipt_sha256"):
        raise ClosureError(f"draw {draw}: global/node draw receipt hash mismatch")
    receipt = read_json(path)
    planned = {row["unit_id"] for row in draw_rows}
    if (not isinstance(receipt, dict) or receipt.get("schema") != DRAW_SCHEMA
            or receipt.get("study_id") != STUDY_ID or receipt.get("node_id") != node_id
            or receipt.get("shard_remainder") != draw % N_NODES or receipt.get("draw_id") != draw
            or set(receipt.get("unit_ids", [])) != planned):
        raise ClosureError(f"draw {draw}: node receipt identity/population mismatch")
    status = receipt.get("status")
    if status == "void_training_failure":
        status = "void"
    global_status = event.get("status")
    if global_status == "void_training_failure":
        global_status = "void"
    if status != global_status or status not in {"complete", "void"}:
        raise ClosureError(f"draw {draw}: node/global terminal status mismatch")
    selected = receipt.get("selected_attempts")
    selected_receipts = receipt.get("artifact_receipt_sha256")
    exhausted = set(receipt.get("exhausted_unit_ids", []))
    if not isinstance(selected, dict) or not isinstance(selected_receipts, dict):
        raise ClosureError(f"draw {draw}: selected-attempt maps are malformed")
    done_units = {uid for uid in planned if node["states"].get(uid) == "done"}
    if set(selected) != done_units or set(selected_receipts) != done_units:
        raise ClosureError(f"draw {draw}: selected-attempt inventory differs from completed units")
    for uid in done_units:
        record = next((item for item in node["attempts"][uid] if item["status"] == "done"), None)
        if (record is None or int(selected[uid]) != record["attempt"]
                or selected_receipts[uid] != record["artifact_receipt_sha256"]):
            raise ClosureError(f"draw {draw}: selected attempt/hash mismatch for {uid}")
    expected_exhausted = {
        uid for uid in planned
        if len(node["attempts"].get(uid, [])) == 2
        and all(item["status"] == "failed" and item["failure_class"] == "training"
                for item in node["attempts"][uid])
    }
    blocked_exhausted = {
        uid for uid in planned
        if len(node["attempts"].get(uid, [])) == 2 and node["states"].get(uid) != "done"
        and uid not in expected_exhausted
    }
    if blocked_exhausted:
        raise ClosureError(f"draw {draw}: mixed/infrastructure exhaustion cannot be declared terminal")
    if status == "complete":
        if done_units != planned or exhausted:
            raise ClosureError(f"draw {draw}: incomplete population declared complete")
    elif not expected_exhausted or exhausted != expected_exhausted:
        raise ClosureError(f"draw {draw}: void is not backed by two training failures")
    if status == "void":
        if (set(event.get("exhausted_unit_ids", [])) != exhausted
                or not isinstance(event.get("reason"), str) or not event["reason"]):
            raise ClosureError(f"draw {draw}: global void receipt does not match exhausted units")
    return status, {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": sha256_file(path),
        "status": status,
        "done_units": sorted(done_units),
        "exhausted_unit_ids": sorted(exhausted),
    }


def _plan_hashes(plan_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in PLAN_HASH_NAMES:
        path = plan_dir / name
        if not path.is_file():
            raise ClosureError(f"missing frozen plan file: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def close_run(
    plan_dir: str | Path, run_dir: str | Path, *, repo: str | Path, features: str | Path,
) -> dict[str, Any]:
    plan_dir, run_dir, repo, features = map(Path, (plan_dir, run_dir, repo, features))
    rows, by_draw = load_plan(plan_dir)
    rows_by_id = {row["unit_id"]: row for row in rows}
    pins_path = plan_dir.parent / "PINS.json"
    pins = read_json(pins_path)
    failures = make_pins.verify_record(pins, repo, features, plan_dir.parent / "environment")
    if failures:
        raise ClosureError(f"PINS verification failed: {failures}")
    global_path = run_dir / "global_ledger.jsonl"
    global_events = read_jsonl(global_path)
    authorized, terminal, stop = validate_global_ledger(
        global_events, by_draw,
        plan_run_sha256=sha256_file(plan_dir / "run_plan.csv"),
        pins_sha256=sha256_file(pins_path),
    )
    recovery_authorizations: dict[str, dict[str, Any]] = {}
    for event in global_events:
        if event.get("event") != "node_recovery_authorized":
            continue
        auth_id = event["authorization_id"]
        if auth_id in recovery_authorizations:
            raise ClosureError("duplicate global node recovery authorization")
        recovery_authorizations[auth_id] = event
    bindings_path = plan_dir.parent / "ENVIRONMENT_BINDINGS.json"
    bindings = read_json(bindings_path)
    global_prefix_hashes: set[str] = set()
    authorizations_by_prefix: dict[str, dict[str, str]] = {}
    seen_authorizations: dict[int, str] = {}
    prefix = b""
    for line, event in zip(global_path.read_bytes().splitlines(keepends=True), global_events):
        prefix += line
        if event.get("event") == "draw_authorized":
            seen_authorizations[int(event["draw_id"])] = event["authorization_id"]
        digest = hashlib.sha256(prefix).hexdigest()
        global_prefix_hashes.add(digest)
        authorizations_by_prefix[digest] = {
            str(draw): auth_id for draw, auth_id in seen_authorizations.items()
        }
    nodes = {
        f"node{i}": validate_node(
            run_dir, f"node{i}", rows_by_id, authorized, recovery_authorizations,
            {
                "plan_run_sha256": sha256_file(plan_dir / "run_plan.csv"),
                "pins_sha256": sha256_file(pins_path),
                "environment_sha256": sha256_file(
                    run_dir / "nodes" / f"node{i}" / "execution_environment.json"
                ),
                "global_prefix_hashes": global_prefix_hashes,
                "authorizations_by_prefix": authorizations_by_prefix,
                "backend_precision": read_json(
                    run_dir / "nodes" / f"node{i}" / "execution_environment.json"
                ).get("contract", {}).get("backend_precision"),
            },
        )
        for i in range(N_NODES)
    }
    used_recoveries = {
        auth_id for node in nodes.values() for auth_id in node["used_recovery_authorization_ids"]
    }
    if used_recoveries != set(recovery_authorizations):
        raise ClosureError("global recovery authorizations are unused or multiply consumed")
    if any(node["last_event"].get("event") != "runner_stop" or
           node["last_event"].get("status") not in {"authorized_draws_terminal", "paused_operator"}
           for node in nodes.values()):
        raise ClosureError("terminal global closure requires every node to stop cleanly")
    if pins.get("environment_bindings") != bindings or pins.get("environment_bindings_sha256") != sha256_file(bindings_path):
        raise ClosureError("PINS/environment binding mismatch")
    for node_id, node in nodes.items():
        expected = bindings.get("nodes", {}).get(node_id, {})
        if (expected.get("shard_remainder") != int(node_id[-1])
                or expected.get("environment_sha256") != node["environment_sha256"]):
            raise ClosureError(f"run environment differs from prospective binding: {node_id}")
    registered = {
        event["node_id"]: event for event in global_events if event.get("event") == "node_registered"
    }
    for node_id, binding in bindings.get("nodes", {}).items():
        event = registered.get(node_id, {})
        if any(event.get(key) != binding.get(key) for key in
               ("shard_remainder", "environment_sha256", "gpu_uuid", "driver_version")):
            raise ClosureError(f"global node registration differs from prospective binding: {node_id}")
    draw_receipts: dict[int, dict[str, Any]] = {}
    complete_ids: list[int] = []
    voids: dict[str, str] = {}
    for draw, event in terminal.items():
        status, receipt = validate_draw_receipt(run_dir, draw, event, by_draw[draw], nodes[f"node{draw % 4}"])
        draw_receipts[draw] = receipt
        if status == "complete":
            complete_ids.append(draw)
        else:
            voids[str(draw)] = f"{len(receipt['exhausted_unit_ids'])} unit(s) exhausted two allowlisted training failures"
    complete_ids.sort()
    analysis_ids = complete_ids[:N_PRIMARY]
    status = "complete" if len(analysis_ids) == N_PRIMARY else "not_tested_incomplete"
    if status == "complete" and len(complete_ids) != N_PRIMARY:
        raise ClosureError("execution continued after 24 complete draws")
    blocked_draw_ids = sorted(set(authorized) - set(terminal))
    if status == "complete" and blocked_draw_ids:
        raise ClosureError("complete closure cannot contain authorized unresolved draws")
    if status != "complete" and (set(authorized) != set(range(N_DRAWS))
                                 or set(terminal) != set(range(N_DRAWS))
                                 or blocked_draw_ids):
        raise ClosureError(
            "paused/blocked execution is nonterminal; incomplete closure requires all 28 draws resolved"
        )
    plan_hashes = _plan_hashes(plan_dir)
    prereg_hashes = {
        "spec.json": sha256_file(plan_dir.parent / "spec.json"),
        "PINS.json": sha256_file(pins_path),
        "ENVIRONMENT_BINDINGS.json": sha256_file(bindings_path),
    }
    completion: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "study_id": STUDY_ID,
        "status": status,
        "required_complete_draws": N_PRIMARY,
        "n_complete_draws": len(analysis_ids),
        "complete_draw_ids": analysis_ids,
        "void_draws": voids,
        "blocked_draw_ids": blocked_draw_ids,
        "authorized_draw_ids": list(authorized),
        "contains_outcome_statistics": False,
        "analysis_lock_payload_sha256": None,
        "global_ledger_sha256": sha256_file(global_path),
        "environment_bindings_sha256": sha256_file(bindings_path),
        "written_at": stop.get("at"),
    }
    if stop.get("status") == "ready_for_closure" and status != "complete":
        raise ClosureError("global run_stop claimed readiness without 24 complete draws")
    if status == "complete":
        selected_ids = {
            row["unit_id"] for draw in analysis_ids for row in by_draw[draw]
        }
        selected: dict[str, Any] = {}
        all_attempts: dict[str, Any] = {}
        for node in nodes.values():
            for uid, records in node["attempts"].items():
                all_attempts[uid] = records
                if uid in selected_ids:
                    done = [item for item in records if item["status"] == "done"]
                    if len(done) != 1:
                        raise ClosureError(f"selected unit lacks exactly one immutable success: {uid}")
                    selected[uid] = done[0]
        if set(selected) != selected_ids:
            raise ClosureError("selected lock population does not contain exactly 1,920 units")
        lock: dict[str, Any] = {
            "schema": LOCK_SCHEMA,
            "study_id": STUDY_ID,
            "spec_version": SPEC_VERSION,
            "contains_outcome_statistics": False,
            "complete_draw_ids": analysis_ids,
            "n_complete_draws": N_PRIMARY,
            "n_locked_units": len(selected),
            "plan_hashes": plan_hashes,
            "preregistration_hashes": prereg_hashes,
            "environment_bindings": bindings,
            "external_feature_cache": [
                {
                    "basename": entry["name"],
                    "path": str(features / entry["name"]),
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                }
                for entry in pins["external_feature_cache"]
            ],
            "environment_receipt_hashes": {
                node_id: node["environment_sha256"] for node_id, node in nodes.items()
            },
            "global_ledger_sha256": sha256_file(global_path),
            "node_ledger_hashes": {node_id: node["ledger_sha256"] for node_id, node in nodes.items()},
            "draw_receipt_hashes": {str(draw): receipt["sha256"] for draw, receipt in draw_receipts.items()},
            "all_attempt_receipts": all_attempts,
            "unit_artifact_hashes": selected,
            "created_at": stop.get("at"),
        }
        lock["lock_payload_sha256"] = sha256_json(lock)
        _atomic_immutable_json(run_dir / "analysis_lock.json", lock)
        completion["analysis_lock_payload_sha256"] = lock["lock_payload_sha256"]
        completion["analysis_lock_sha256"] = sha256_file(run_dir / "analysis_lock.json")
    elif (run_dir / "analysis_lock.json").exists():
        raise ClosureError("incomplete execution must not have an analysis lock")
    _atomic_immutable_json(run_dir / "completion.json", completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args(argv)
    result = close_run(args.plan, args.run, repo=args.repo, features=args.features)
    print(json.dumps({
        "status": result["status"], "n_complete_draws": result["n_complete_draws"],
        "analysis_lock_created": result["analysis_lock_payload_sha256"] is not None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

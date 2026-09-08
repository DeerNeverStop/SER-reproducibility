"""Outcome-blind, single-writer coordinator for one of four N14R2 GPU shards.

Start is the conservative actual-attempt boundary: it is fsynced immediately
before allocation/dispatch. A crash in that narrow interval still consumes one
attempt. Neither initialization nor CUDA health probes consume fit attempts.
An infrastructure circuit break ends this invocation; a subsequent explicit
invocation creates and healthchecks entirely new processes before any retry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import queue
import re
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from .worker import (ARTIFACT_NAMES, INFRA_FAILURES, RECEIPT_SCHEMA, IntegrityError,
                     N14R2PlanIO, canonical, file_lock, hash_file, hash_json, now,
                     publish_json, read_json, receipt_for, worker_main)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NODE_EVENTS = {"runner_start", "worker_ready", "start", "done", "failed", "circuit_break", "runner_stop"}


def load_plan(plan_dir):
    from .spec import SPEC
    plan_dir = Path(plan_dir)
    with (plan_dir / "run_plan.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    configs = read_json(plan_dir / "unit_configs.json")
    if len(rows) != SPEC["design"]["maximum_units"] or len({r["unit_id"] for r in rows}) != len(rows):
        raise IntegrityError("plan has missing or duplicate units")
    for row in rows:
        if not HEX64.fullmatch(row["unit_id"]):
            raise IntegrityError("unsafe unit identifier")
        draw = int(row["draw_id"])
        if row["arm"] != "N14R2" or row["study_id"] != "N14R2" or draw not in range(28):
            raise IntegrityError("old or unexpected study/draw in plan")
        if draw != int(row["r"]) or int(row["train_rep"]) != 0 or int(row["seed_index"]) != 0:
            raise IntegrityError("compatibility seed/draw mirrors disagree")
        if row["draw_role"] != ("primary" if draw < 24 else "reserve"):
            raise IntegrityError("draw role mismatch")
        cfg = configs.get(row["config_sha256"])
        if cfg is None or hash_json(cfg) != row["config_sha256"] or cfg.get("engine") != "p1_frozen":
            raise IntegrityError("configuration identity mismatch")
        if int(cfg.get("hpo_config_index", -1)) != int(row["config_index"]):
            raise IntegrityError("configuration index mismatch")
    for draw in range(28):
        keys = [(int(r["fold"]), r["cell"], int(r["config_index"])) for r in rows if int(r["draw_id"]) == draw]
        expected = {(f, c, k) for f in range(5) for c in ("GR_hpo", "GG_hpo") for k in range(8)}
        if len(keys) != 80 or set(keys) != expected:
            raise IntegrityError("draw must contain the complete 80-unit grid")
    return rows, configs


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return parse_jsonl_bytes(path.read_bytes(), path.name)


def parse_jsonl_bytes(data, name):
    if data and not data.endswith(b"\n"):
        raise IntegrityError(f"torn JSONL record: {name}; preserve and repair explicitly")
    import json
    try:
        records = [json.loads(line) for line in data.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise IntegrityError(f"malformed JSONL: {name}") from exc
    if any(not isinstance(r, dict) or r.get("seq") != i for i, r in enumerate(records, 1)):
        raise IntegrityError(f"missing/duplicate ledger sequence: {name}")
    return records


class AuthorizationMap(dict):
    """A verified immutable snapshot plus its provenance, not a live ledger."""


def authorized_draws(path, *, shard, plan_run_sha256, pins_sha256, reserve_draw=None, recovery_ledger_sha256=None):
    """Read a frozen global authorization snapshot; never authorize locally."""
    snapshot = Path(path).read_bytes()
    records = parse_jsonl_bytes(snapshot, Path(path).name)
    if not records:
        raise IntegrityError("global authorization ledger required")
    previous, authorized, terminal, last_reserve = None, {}, {}, 23
    stopped, recoveries = False, {}
    for event in records:
        payload = {k: v for k, v in event.items() if k != "event_sha256"}
        if event.get("study_id") != "N14R2" or event.get("prev_event_sha256") != previous or event.get("event_sha256") != hash_json(payload):
            raise IntegrityError("global authorization hash chain invalid")
        previous = event["event_sha256"]
        if stopped:
            raise IntegrityError("event after global run_stop")
        kind = event.get("event")
        if kind == "run_stop":
            stopped = True
        elif kind == "node_recovery_authorized":
            node = event.get("shard_remainder")
            if (type(node) is not int or node not in range(4) or event.get("node_id") != f"node{node}"
                    or event.get("plan_run_sha256") != plan_run_sha256 or event.get("pins_sha256") != pins_sha256
                    or not HEX64.fullmatch(str(event.get("prior_node_ledger_sha256", "")))
                    or not event.get("authorization_id")):
                raise IntegrityError("invalid node recovery authorization")
            recoveries[node] = event
        elif kind in {"draw_authorized", "draw_terminal"}:
            draw = event.get("draw_id")
            if type(draw) is not int or draw not in range(28) or event.get("node_id") != f"node{draw % 4}":
                raise IntegrityError("global draw routing invalid")
            if kind == "draw_authorized":
                if draw in authorized or event.get("shard_remainder") != draw % 4:
                    raise IntegrityError("duplicate/misrouted draw authorization")
                if event.get("plan_run_sha256") != plan_run_sha256 or event.get("pins_sha256") != pins_sha256:
                    raise IntegrityError("authorization not bound to frozen inputs")
                if not isinstance(event.get("authorization_id"), str) or not event["authorization_id"]:
                    raise IntegrityError("authorization identity missing")
                if event.get("draw_role") != ("primary" if draw < 24 else "reserve"):
                    raise IntegrityError("authorization role mismatch")
                if draw >= 24:
                    if not all(d in terminal for d in range(24)) or draw != last_reserve + 1:
                        raise IntegrityError("reserve requires all primary terminal statuses and ascending order")
                    if last_reserve >= 24 and last_reserve not in terminal:
                        raise IntegrityError("previous reserve is not terminal")
                    if sum(t["status"] == "void" for t in terminal.values()) < draw - 23:
                        raise IntegrityError("reserve is not justified by terminal voids")
                    last_reserve = draw
                authorized[draw] = event
            else:
                if draw not in authorized or draw in terminal or event.get("status") not in {"complete", "void"}:
                    raise IntegrityError("duplicate/unauthorized global terminal draw")
                if not HEX64.fullmatch(str(event.get("node_draw_receipt_sha256", ""))):
                    raise IntegrityError("global terminal receipt hash missing")
                if event["status"] == "void" and (not event.get("exhausted_unit_ids") or not event.get("reason")):
                    raise IntegrityError("void needs exhausted units and reason")
                terminal[draw] = event
        elif kind not in {"run_start", "node_registered", "circuit_break"}:
            raise IntegrityError("unknown global ledger event")
    if stopped:
        raise IntegrityError("global run is stopped; new dispatch forbidden")
    if recovery_ledger_sha256 is not None and recoveries.get(shard, {}).get("prior_node_ledger_sha256") != recovery_ledger_sha256:
        raise IntegrityError("explicit global recovery authorization for this exact node ledger required")
    selected = AuthorizationMap({d: a for d, a in authorized.items() if d < 24 and d % 4 == shard})
    if reserve_draw is not None:
        if reserve_draw not in range(24, 28) or reserve_draw % 4 != shard or reserve_draw not in authorized:
            raise IntegrityError("explicit reserve is not authorized for this shard")
        selected[reserve_draw] = authorized[reserve_draw]
    if not selected:
        raise IntegrityError("no authorized primary/reserve draws for this shard")
    selected.global_ledger_sha256 = hashlib.sha256(snapshot).hexdigest()
    selected.recovery_authorization_id = recoveries[shard]["authorization_id"] if recovery_ledger_sha256 else None
    return selected


class Ledger:
    def __init__(self, node_dir, shard):
        self.path = Path(node_dir) / "ledger.jsonl"
        self.audit_path = Path(node_dir) / "node_audit.jsonl"
        self.shard, self.node_id = shard, f"node{shard}"
        self.records = read_jsonl(self.path)
        self.audit = read_jsonl(self.audit_path)
        self.invocation_id = self.records[-1].get("invocation_id") if self.records else None

    def append(self, event, **fields):
        record = {"seq": len(self.records) + 1, "event": event, "study_id": "N14R2",
                  "node_id": self.node_id, "shard_remainder": self.shard, "at": now(),
                  "invocation_id": self.invocation_id, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        return record

    def audit_event(self, event, **fields):
        record = {"seq": len(self.audit) + 1, "event": event, "study_id": "N14R2",
                  "node_id": self.node_id, "at": now(), "invocation_id": self.invocation_id, **fields}
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.audit.append(record)


def parse_attempts(records, rows, shard):
    by_id, attempts, active, completed = {r["unit_id"]: r for r in rows}, {}, {}, set()
    blocked, circuit_seen, invocation = False, False, None
    for event in records:
        kind = event.get("event")
        if kind not in NODE_EVENTS or event.get("study_id") != "N14R2" or event.get("node_id") != f"node{shard}":
            raise IntegrityError("unknown or foreign node event")
        if kind == "runner_start":
            if active or (blocked and not circuit_seen):
                raise IntegrityError("runner restart before attempts/circuit were reconciled")
            invocation = event.get("invocation_id")
            if not invocation:
                raise IntegrityError("runner invocation identity missing")
            blocked, circuit_seen = False, False
        elif kind == "circuit_break":
            if active:
                raise IntegrityError("circuit break must seal all dispatched attempts first")
            blocked, circuit_seen = True, True
        elif kind in {"start", "done", "failed"}:
            uid, attempt = event.get("unit_id"), event.get("attempt")
            row = by_id.get(uid)
            if row is None or int(row["draw_id"]) % 4 != shard or event.get("draw_id") != int(row["draw_id"]):
                raise IntegrityError("unplanned or misrouted attempt")
            if type(attempt) is not int or attempt not in (1, 2):
                raise IntegrityError("attempt cap or type invalid")
            if kind == "start":
                if blocked or invocation is None or uid in active or uid in completed:
                    raise IntegrityError("duplicate or post-failure dispatch")
                history = attempts.setdefault(uid, [])
                if attempt != len(history) + 1:
                    raise IntegrityError("missing/duplicate attempt ordinal")
                item = {"start": event, "terminal": None}
                history.append(item)
                active[uid] = item
            else:
                item = active.pop(uid, None)
                if item is None or item["start"]["attempt"] != attempt:
                    raise IntegrityError("terminal without matching open attempt")
                if not HEX64.fullmatch(str(event.get("receipt_sha256", ""))):
                    raise IntegrityError("terminal receipt hash missing")
                if kind == "failed" and event.get("failure_class") not in INFRA_FAILURES | {"training"}:
                    raise IntegrityError("unknown failure class")
                item["terminal"] = event
                if kind == "done":
                    completed.add(uid)
                else:
                    blocked = True
        elif kind == "runner_stop" and active:
            raise IntegrityError("runner stopped with unsealed attempts")
    return attempts


def attempt_path(node_dir, uid, attempt):
    return Path(node_dir) / "units" / uid / f"attempt_{attempt:02d}"


def task_for(row, start, node_dir, shard):
    return {"row": row, "attempt": start["attempt"], "started_at": start["at"],
            "node_id": f"node{shard}", "shard_remainder": shard,
            "attempt_dir": str(attempt_path(node_dir, row["unit_id"], start["attempt"]))}


def verify_receipt(path, task, configs, io, validator=None):
    receipt = read_json(path)
    if not isinstance(receipt, dict):
        raise IntegrityError("artifact receipt is not an object")
    expected = receipt_for(task, status=receipt.get("status"))
    required = set(expected)
    if set(receipt) != required:
        raise IntegrityError("artifact receipt fields changed")
    if any(type(receipt.get(k)) is not int for k in ("attempt", "draw_id", "shard_remainder")):
        raise IntegrityError("artifact receipt integer identity type mismatch")
    if not isinstance(receipt.get("finished_at"), str) or not receipt["finished_at"]:
        raise IntegrityError("artifact receipt finish time missing")
    for field in required - {"finished_at", "status", "failure_class", "artifact_hashes"}:
        if receipt.get(field) != expected[field]:
            raise IntegrityError(f"receipt identity mismatch: {field}")
    if receipt["status"] == "done":
        if receipt["failure_class"] is not None or set(receipt["artifact_hashes"]) != set(ARTIFACT_NAMES):
            raise IntegrityError("successful receipt contract invalid")
        for name, digest in receipt["artifact_hashes"].items():
            file = Path(path).parent / name
            if not file.is_file() or file.is_symlink() or hash_file(file) != digest:
                raise IntegrityError("missing/changed completed artifact")
        # Raw artifacts are hashed as bytes only. The isolated fit worker has
        # already performed local structure/checkpoint validation before seal.
    elif receipt["status"] == "failed":
        if receipt["failure_class"] not in INFRA_FAILURES | {"training"} or receipt["artifact_hashes"]:
            raise IntegrityError("failed receipt contract invalid")
    else:
        raise IntegrityError("unknown artifact receipt status")
    return receipt


def publish_pointer(node_dir, uid, attempt, receipt_sha256):
    path = Path(node_dir) / "units" / uid / "DONE.json"
    value = {"attempt": attempt, "artifact_receipt_sha256": receipt_sha256}
    if path.exists():
        if read_json(path) != value:
            raise IntegrityError("selected successful attempt cannot change")
    else:
        publish_json(path, value)


def seal_attempt(ledger, row, start, node_dir, shard, configs, io, *, failure_class="interrupted", validator=None):
    task = task_for(row, start, node_dir, shard)
    directory = Path(task["attempt_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "artifact_receipt.json"
    if not path.exists():
        # Receipt is the commit boundary. Even a raw DONE without its worker
        # receipt is conservatively interrupted, not outcome-inspected/revived.
        receipt = receipt_for(task, status="failed", failure_class=failure_class)
        publish_json(path, receipt)
    receipt = verify_receipt(path, task, configs, io, validator)
    digest = hash_file(path)
    ledger.append(receipt["status"], unit_id=row["unit_id"], draw_id=int(row["draw_id"]),
                  attempt=start["attempt"], receipt_sha256=digest, failure_class=receipt["failure_class"])
    if receipt["status"] == "done":
        publish_pointer(node_dir, row["unit_id"], start["attempt"], digest)
    return receipt


def reconcile(node_dir, rows, shard, ledger, configs, io, validator=None):
    """Run only while coordinator lock is held and ALL old worker locks are free."""
    attempts = parse_attempts(ledger.records, rows, shard)
    by_id = {r["unit_id"]: r for r in rows}
    units_root = Path(node_dir) / "units"
    if units_root.exists():
        for directory in units_root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or directory.name not in attempts:
                raise IntegrityError("unledgered or foreign unit directory")
            expected = {f"attempt_{i + 1:02d}" for i in range(len(attempts[directory.name]))} | {"DONE.json"}
            if any(p.name not in expected or p.is_symlink() for p in directory.iterdir()):
                raise IntegrityError("orphan/extra attempt directory")
    recovered_failure = False
    for uid, history in attempts.items():
        pointer_expected = None
        for item in history:
            task = task_for(by_id[uid], item["start"], node_dir, shard)
            path = Path(task["attempt_dir"]) / "artifact_receipt.json"
            terminal = item["terminal"]
            if terminal is None:
                result = seal_attempt(ledger, by_id[uid], item["start"], node_dir, shard, configs, io, validator=validator)
                recovered_failure |= result["status"] == "failed"
                terminal = ledger.records[-1]
                ledger.audit_event("orphan_reconciled", unit_id=uid, attempt=task["attempt"], status=result["status"])
            else:
                if not path.is_file() or hash_file(path) != terminal["receipt_sha256"]:
                    raise IntegrityError("terminal has missing or changed receipt")
                result = verify_receipt(path, task, configs, io, validator)
                if result["status"] != terminal["event"] or result["failure_class"] != terminal.get("failure_class"):
                    raise IntegrityError("receipt contradicts terminal ledger")
            if result["status"] == "done":
                pointer_expected = {"attempt": task["attempt"], "artifact_receipt_sha256": hash_file(path)}
                publish_pointer(node_dir, uid, task["attempt"], hash_file(path))
        pointer = units_root / uid / "DONE.json"
        if pointer.exists() and read_json(pointer) != pointer_expected:
            raise IntegrityError("orphan or contradictory success pointer")
    # A crash can happen after a failed terminal but before the circuit record.
    failure_since_boundary = False
    for event in ledger.records:
        if event["event"] in {"runner_start", "circuit_break"}:
            failure_since_boundary = False
        elif event["event"] == "failed":
            failure_since_boundary = True
    if recovered_failure or failure_since_boundary:
        ledger.append("circuit_break", reason="orphan_recovery", contains_outcome_statistics=False)
    return parse_attempts(ledger.records, rows, shard)


def draw_states(rows, attempts):
    result = {}
    for draw in sorted({int(r["draw_id"]) for r in rows}):
        members = [r for r in rows if int(r["draw_id"]) == draw]
        done, exhausted, voidable = [], [], []
        for row in members:
            history = attempts.get(row["unit_id"], [])
            terminals = [a["terminal"] for a in history]
            if any(t and t["event"] == "done" for t in terminals):
                done.append(row["unit_id"])
            elif len(history) >= 2:
                exhausted.append(row["unit_id"])
                if all(t and t["event"] == "failed" and t["failure_class"] == "training" for t in terminals):
                    voidable.append(row["unit_id"])
        status = "complete" if len(done) == len(members) else "blocked" if set(exhausted) - set(voidable) else "void" if voidable else "pending"
        result[draw] = {"status": status, "done_units": len(done), "total_units": len(members), "exhausted_unit_ids": exhausted}
    return result


def publish_draw_receipts(node_dir, rows, attempts, shard):
    states = draw_states(rows, attempts)
    for draw, state in states.items():
        if state["status"] not in {"complete", "void"}:
            continue
        uids = sorted(r["unit_id"] for r in rows if int(r["draw_id"]) == draw)
        selected, digests = {}, {}
        for uid in uids:
            for item in attempts.get(uid, []):
                terminal = item["terminal"]
                if terminal and terminal["event"] == "done":
                    selected[uid] = terminal["attempt"]
                    digests[uid] = terminal["receipt_sha256"]
        receipt = {"schema": "ser26-n14r2-node-draw-1", "study_id": "N14R2", "node_id": f"node{shard}",
                   "shard_remainder": shard, "draw_id": draw, "status": state["status"], "unit_ids": uids,
                   "selected_attempts": selected, "artifact_receipt_sha256": digests,
                   "exhausted_unit_ids": state["exhausted_unit_ids"]}
        path = Path(node_dir) / "draws" / f"draw_{draw:02d}.json"
        if path.exists():
            if read_json(path) != receipt:
                raise IntegrityError("immutable draw receipt changed")
        else:
            publish_json(path, receipt)
    return states


class ProcessPool:
    """One mailbox per spawned process; no futures executor hidden task queue."""
    def __init__(self, count, settings):
        context = mp.get_context("spawn")
        self.stop_event = context.Event()
        self.outgoing = context.Queue()
        self.incoming = [context.Queue() for _ in range(count)]
        self.processes = [context.Process(target=worker_main, args=(i, self.incoming[i], self.outgoing, settings, self.stop_event), daemon=True) for i in range(count)]
        try:
            for process in self.processes:
                process.start()
        except BaseException:
            self.close(abort=True)
            raise

    def receive(self, timeout=0.25):
        try:
            return self.outgoing.get(timeout=timeout)
        except queue.Empty:
            return None

    def send(self, slot, task):
        if self.stop_event.is_set():
            raise RuntimeError("worker circuit is open")
        self.incoming[slot].put(task)

    def dead(self):
        return [i for i, p in enumerate(self.processes) if not p.is_alive()]

    def stopped(self):
        return self.stop_event.is_set()

    def close(self, abort=False):
        self.stop_event.set()
        for incoming in self.incoming:
            incoming.put(None)
        if abort:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.is_alive():
                raise IntegrityError("worker did not stop; recovery unsafe")
        for channel in [self.outgoing, *self.incoming]:
            channel.cancel_join_thread()
            channel.close()


def coordinate(node_dir, rows, configs, io, shard, ledger, authorizations, settings,
               *, workers=8, health_timeout=180, unit_timeout=3600, pool_factory=ProcessPool,
               validator=None):
    """Testable orchestration core. Production caller provides the lock/freeze."""
    attempts = reconcile(node_dir, rows, shard, ledger, configs, io, validator)
    states = publish_draw_receipts(node_dir, rows, attempts, shard)
    if any(s["status"] == "blocked" for s in states.values()):
        # Exhausted infrastructure attempts cannot be unlocked by ordinary
        # recovery authorization; a prospective public amendment is required.
        return {"status": "blocked_not_void", "draws": states, "contains_outcome_statistics": False}
    ledger.invocation_id = uuid.uuid4().hex
    ledger.append("runner_start", workers=workers, authorization_ids={str(d): a["authorization_id"] for d, a in authorizations.items()},
                  contains_outcome_statistics=False, **settings.get("runner_identity", {}))
    pending = [r for r in rows if int(r["draw_id"]) in authorizations and states[int(r["draw_id"])]["status"] == "pending"
               and not any(a["terminal"] and a["terminal"]["event"] == "done" for a in attempts.get(r["unit_id"], []))]
    pending.sort(key=lambda r: (int(r["draw_id"]), int(r["fold"]), r["cell"], int(r["config_index"])))
    if not pending:
        ledger.append("runner_stop", status="authorized_draws_terminal", contains_outcome_statistics=False)
        return {"status": "authorized_draws_terminal", "draws": states}
    pool, active, failure, finished, paused = None, {}, None, False, False
    pause_file = Path(settings["pause_file"]) if settings.get("pause_file") else None
    try:
        pool = pool_factory(workers, settings)
        ready, deadline = set(), time.monotonic() + health_timeout
        while len(ready) < workers:
            message = pool.receive()
            if message:
                if (message.get("event") != "ready" or message.get("slot") in ready or message.get("cuda_healthy") is not True
                        or message.get("health_checksum") != 4096.0
                        or ("backend_precision" in settings and message.get("backend_precision") != settings["backend_precision"])):
                    failure = "healthcheck"
                    break
                slot = message["slot"]
                if type(slot) is not int or slot not in range(workers):
                    raise IntegrityError("invalid healthcheck worker slot")
                ready.add(slot)
                ledger.append("worker_ready", slot=slot, pid=message.get("pid"), cuda_healthy=True,
                              health_operation=message.get("health_operation"), health_device=message.get("health_device"),
                              health_checksum=message.get("health_checksum"), backend_precision=message.get("backend_precision"))
            if pool.dead() or pool.stopped() or time.monotonic() >= deadline:
                failure = "healthcheck"
                break
        idle = set(ready)
        while failure is None and (pending or active):
            if pause_file is not None and pause_file.exists():
                paused = True
            if paused and not active:
                break
            # Stop is shared directly by workers: check before EVERY dispatch,
            # not only after eventually receiving a queued failure message.
            if pool.stopped() or pool.dead():
                failure = "worker_exit"
                break
            for slot in sorted(idle):
                if pause_file is not None and pause_file.exists():
                    paused = True
                if not pending or paused:
                    break
                if pool.stopped() or pool.dead():
                    failure = "worker_exit"
                    break
                row = pending.pop(0)
                number = len(attempts.get(row["unit_id"], [])) + 1
                if number > 2:
                    raise IntegrityError("attempt cap exceeded")
                start = ledger.append("start", unit_id=row["unit_id"], draw_id=int(row["draw_id"]), attempt=number,
                                      authorization_id=authorizations[int(row["draw_id"])]["authorization_id"], plan_row_sha256=hash_json(row))
                task = task_for(row, start, node_dir, shard)
                active[slot] = (row, start, time.monotonic())
                Path(task["attempt_dir"]).mkdir(parents=True, exist_ok=False)
                try:
                    pool.send(slot, task)
                except RuntimeError:
                    failure = "worker_exit"
                    break
                idle.remove(slot)
            if failure:
                break
            message = pool.receive()
            if message:
                slot = message.get("slot")
                if message.get("event") != "result" or slot not in active:
                    failure = "worker_exit"
                    break
                row, start, _ = active[slot]
                if message.get("unit_id") != row["unit_id"] or message.get("attempt") != start["attempt"]:
                    raise IntegrityError("worker result identity mismatch")
                receipt_path = attempt_path(node_dir, row["unit_id"], start["attempt"]) / "artifact_receipt.json"
                if not receipt_path.exists() or hash_file(receipt_path) != message.get("receipt_sha256"):
                    raise IntegrityError("worker receipt missing or hash mismatch")
                result = seal_attempt(ledger, row, start, node_dir, shard, configs, io, validator=validator)
                del active[slot]
                idle.add(slot)
                if result["status"] != "done" or message.get("postfit_failure"):
                    failure = result.get("failure_class") or message["postfit_failure"]
            if unit_timeout and any(time.monotonic() - item[2] > unit_timeout for item in active.values()):
                failure = "timeout"
        finished = failure is None
    except KeyboardInterrupt:
        failure = "interrupted"
    except BaseException:
        failure = "integrity"
        raise
    finally:
        if pool is not None:
            pool.close(abort=not finished)
        # Workers are dead before touching their attempts. Complete receipts
        # survive a circuit break; incomplete peers are conservatively failed.
        for slot, (row, start, _) in sorted(active.items()):
            seal_attempt(ledger, row, start, node_dir, shard, configs, io,
                         failure_class="timeout" if failure == "timeout" else "circuit_cancelled", validator=validator)
        if failure:
            ledger.append("circuit_break", reason=failure, contains_outcome_statistics=False)
            ledger.audit_event("dispatch_stopped", reason=failure, fresh_processes_required_for_retry=True)
    attempts = parse_attempts(ledger.records, rows, shard)
    states = publish_draw_receipts(node_dir, rows, attempts, shard)
    status = "blocked_not_void" if any(s["status"] == "blocked" for s in states.values()) else "circuit_open_resume_required" if failure else "paused_operator" if paused else "authorized_draws_terminal"
    if not failure:
        ledger.append("runner_stop", status=status, contains_outcome_statistics=False,
                      pause_request_sha256=hash_file(pause_file) if paused and pause_file and pause_file.exists() else None)
    return {"status": status, "draws": states, "contains_outcome_statistics": False}


def preflight_pool(node_dir, settings, *, workers=8, health_timeout=180, pool_factory=ProcessPool):
    """Initialize the full pool and storage publication, with zero fit dispatch."""
    checks, pool = {}, None
    try:
        pool = pool_factory(workers, settings)
        deadline = time.monotonic() + health_timeout
        while len(checks) < workers:
            message = pool.receive()
            if message:
                slot = message.get("slot")
                if (message.get("event") != "ready" or type(slot) is not int or slot not in range(workers)
                        or slot in checks or message.get("cuda_healthy") is not True
                        or message.get("health_checksum") != 4096.0):
                    raise IntegrityError("pool preflight healthcheck failed")
                checks[slot] = message
            if pool.dead() or pool.stopped() or time.monotonic() >= deadline:
                raise IntegrityError("pool preflight failed or timed out")
        result = {"schema": "ser26-n14r2-pool-preflight-1", "study_id": "N14R2", "status": "passed",
                  "node_id": Path(node_dir).name, "at": now(), "workers": [checks[i] for i in sorted(checks)],
                  "fit_attempts_dispatched": 0, "contains_outcome_statistics": False}
        # Exercises the exact hard-link + fsync path used for DONE/receipts.
        path = Path(node_dir) / "preflight" / f"pool-{uuid.uuid4().hex}.json"
        publish_json(path, result)
        if read_json(path) != result:
            raise IntegrityError("storage preflight readback mismatch")
        return result
    finally:
        if pool is not None:
            pool.close(abort=len(checks) != workers)


def verify_pins(plan_dir, feature_dir):
    from .make_pins import verify_record
    package = Path(plan_dir).parent
    failures = verify_record(read_json(package / "PINS.json"), Path(__file__).resolve().parents[2],
                             Path(feature_dir), package / "environment")
    if failures:
        raise IntegrityError("frozen source/environment/cache inventory mismatch")


def execute(plan_dir, manifest_dir, feature_dir, run_root, *, shard, workers=8,
            global_ledger=None, reserve_draw=None, health_timeout=180, unit_timeout=3600, preflight_only=False):
    if shard not in range(4) or workers != 8:
        raise IntegrityError("frozen allocation requires shard 0..3 and exactly eight workers")
    plan_dir, manifest_dir, feature_dir, run_root = map(Path, (plan_dir, manifest_dir, feature_dir, run_root))
    node_dir = run_root / "nodes" / f"node{shard}"
    rows, configs = load_plan(plan_dir)
    rows = [r for r in rows if int(r["draw_id"]) % 4 == shard]
    with file_lock(node_dir / ".coordinator.lock"):
        # Never reconcile while an orphan child of an old coordinator is alive.
        with ExitStack() as stack:
            for slot in range(8):
                stack.enter_context(file_lock(node_dir / ".worker-locks" / f"slot{slot}.lock"))
        verify_pins(plan_dir, feature_dir)
        from .ops.environment import freeze_execution_environment
        environment = freeze_execution_environment(plan_dir, manifest_dir, feature_dir, node_dir,
                                                   node_id=f"node{shard}", shard_remainder=shard)
        io = N14R2PlanIO(plan_dir, manifest_dir, node_dir)
        # Validate every assigned fold before the first worker can run a fit.
        for row in rows:
            io.fold(row)
        settings = {"plan_dir": str(plan_dir.resolve()), "manifest_dir": str(manifest_dir.resolve()),
                    "feature_dir": str(feature_dir.resolve()), "node_dir": str(node_dir.resolve()),
                    "coordinator_pid": os.getpid(), "backend_precision": environment["contract"]["backend_precision"],
                    "pause_file": str((run_root / "PAUSE_REQUEST").resolve())}
        if preflight_only:
            result = preflight_pool(node_dir, settings, workers=workers, health_timeout=health_timeout)
        else:
            ledger_path = node_dir / "ledger.jsonl"
            prior_records = read_jsonl(ledger_path)
            clean_stop = prior_records and prior_records[-1]["event"] == "runner_stop" and prior_records[-1].get("status") in {"authorized_draws_terminal", "paused_operator"}
            recovery_sha = hash_file(ledger_path) if prior_records and not clean_stop else None
            authorizations = authorized_draws(global_ledger or run_root / "global_ledger.jsonl", shard=shard,
                                             plan_run_sha256=hash_file(plan_dir / "run_plan.csv"),
                                             pins_sha256=hash_file(plan_dir.parent / "PINS.json"), reserve_draw=reserve_draw,
                                             recovery_ledger_sha256=recovery_sha)
            settings["runner_identity"] = {
                "plan_run_sha256": hash_file(plan_dir / "run_plan.csv"), "pins_sha256": hash_file(plan_dir.parent / "PINS.json"),
                "execution_environment_sha256": hash_file(node_dir / "execution_environment.json"),
                "global_ledger_sha256": authorizations.global_ledger_sha256,
                "recovery_authorization_id": authorizations.recovery_authorization_id,
                "prior_node_ledger_sha256": recovery_sha}
            ledger = Ledger(node_dir, shard)
            result = coordinate(node_dir, rows, configs, io, shard, ledger, authorizations, settings,
                                workers=workers, health_timeout=health_timeout, unit_timeout=unit_timeout)
        verify_pins(plan_dir, feature_dir)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--global-ledger", type=Path)
    parser.add_argument("--reserve-draw", type=int, choices=range(24, 28))
    parser.add_argument("--preflight-only", action="store_true", help="validate all eight spawned workers and storage; dispatch zero fits")
    parser.add_argument("--health-timeout", type=float, default=180)
    parser.add_argument("--unit-timeout", type=float, default=3600, help="infrastructure watchdog seconds; 0 disables (never a draw-void rule)")
    args = parser.parse_args(argv)
    if args.health_timeout <= 0 or args.unit_timeout < 0:
        parser.error("invalid watchdog duration")
    try:
        result = execute(**vars(args))
    except Exception as exc:
        print(canonical({"study_id": "N14R2", "status": "blocked_integrity", "exception_type": type(exc).__name__, "contains_outcome_statistics": False}))
        return 2
    print(canonical(result))
    return 0 if result["status"] in {"authorized_draws_terminal", "passed"} else 3


if __name__ == "__main__":
    raise SystemExit(main())

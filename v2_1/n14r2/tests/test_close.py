from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from v2_1.n14r2 import close, make_pins


PLAN_SHA = "a" * 64
PINS_SHA = "b" * 64


def _chain(items: list[dict]) -> list[dict]:
    records = []
    previous = None
    for fields in items:
        value = {
            "seq": len(records) + 1,
            "study_id": "N14R2",
            "at": f"2026-09-05T00:00:{len(records):02d}Z",
            "prev_event_sha256": previous,
            **fields,
        }
        value["event_sha256"] = close.sha256_json(value)
        previous = value["event_sha256"]
        records.append(value)
    return records


def _global_items(*, void_draw: int | None = None, reserve: bool = False) -> list[dict]:
    items = [{
        "event": "run_start", "plan_run_sha256": PLAN_SHA, "pins_sha256": PINS_SHA,
        "preregistration_commit": "1" * 40, "preregistration_tag": "SER26-N14R2-prereg-1",
        "public_receipt_url": "https://github.com/DeerNeverStop/SER/pull/6#issuecomment-1",
        "public_receipt_created_at": "2026-09-05T00:00:00Z", "contains_outcome_statistics": False,
    }]
    items += [
        {
            "event": "node_registered", "node_id": f"node{i}", "shard_remainder": i,
            "environment_sha256": str(i) * 64, "gpu_uuid": f"gpu-{i}", "driver_version": "x",
        }
        for i in range(4)
    ]
    for draw in range(24):
        items.append({
            "event": "draw_authorized", "draw_id": draw, "draw_role": "primary",
            "node_id": f"node{draw % 4}", "shard_remainder": draw % 4,
            "authorization_id": f"auth-{draw}", "plan_run_sha256": PLAN_SHA,
            "pins_sha256": PINS_SHA,
        })
    for draw in range(24):
        status = "void" if draw == void_draw else "complete"
        items.append({
            "event": "draw_terminal", "draw_id": draw, "node_id": f"node{draw % 4}",
            "status": status, "node_draw_receipt_sha256": "c" * 64,
            **({"exhausted_unit_ids": ["u"], "reason": "two failures"} if status == "void" else {}),
        })
    if reserve:
        items += [{
            "event": "draw_authorized", "draw_id": 24, "draw_role": "reserve",
            "node_id": "node0", "shard_remainder": 0, "authorization_id": "auth-24",
            "plan_run_sha256": PLAN_SHA, "pins_sha256": PINS_SHA,
        }, {
            "event": "draw_terminal", "draw_id": 24, "node_id": "node0", "status": "complete",
            "node_draw_receipt_sha256": "d" * 64,
        }]
    items.append({"event": "run_stop", "status": "ready_for_closure"})
    return items


def test_global_ledger_accepts_primary_then_needed_reserve() -> None:
    events = _chain(_global_items(void_draw=3, reserve=True))
    authorized, terminal, stop = close.validate_global_ledger(
        events, {draw: [{}] for draw in range(28)},
        plan_run_sha256=PLAN_SHA, pins_sha256=PINS_SHA,
    )
    assert set(authorized) == set(range(25))
    assert terminal[3]["status"] == "void"
    assert stop["status"] == "ready_for_closure"


def test_global_ledger_rejects_unhashed_or_early_reserve() -> None:
    items = _global_items(void_draw=3, reserve=True)
    events = _chain(items)
    with pytest.raises(close.ClosureError, match="hash chained"):
        close.validate_global_ledger(
            [{k: v for k, v in e.items() if k not in {"event_sha256", "prev_event_sha256"}} for e in events],
            {draw: [{}] for draw in range(28)}, plan_run_sha256=PLAN_SHA, pins_sha256=PINS_SHA,
        )
    reserve = next(item for item in items if item.get("draw_id") == 24 and item["event"] == "draw_authorized")
    items.remove(reserve)
    items.insert(6, reserve)
    with pytest.raises(close.ClosureError, match="before all primary"):
        close.validate_global_ledger(
            _chain(items), {draw: [{}] for draw in range(28)},
            plan_run_sha256=PLAN_SHA, pins_sha256=PINS_SHA,
        )


def _node_event(seq: int, kind: str, invocation: str, **extra) -> dict:
    return {
        "seq": seq, "event": kind, "study_id": "N14R2", "node_id": "node0",
        "shard_remainder": 0, "at": f"2026-09-05T00:00:{seq:02d}Z",
        "invocation_id": invocation, **extra,
    }


def _write_node(tmp_path: Path, events: list[dict]) -> Path:
    node = tmp_path / "nodes" / "node0"
    node.mkdir(parents=True)
    (node / "execution_environment.json").write_text("{}\n", encoding="utf-8")
    (node / "ledger.jsonl").write_text(
        "".join(close.canonical_json(event) + "\n" for event in events), encoding="utf-8", newline="\n",
    )
    return node


def test_recovery_binds_pre_reconciliation_byte_prefix(tmp_path: Path) -> None:
    first = _node_event(1, "runner_start", "old")
    old_bytes = (close.canonical_json(first) + "\n").encode()
    source_sha = hashlib.sha256(old_bytes).hexdigest()
    events = [
        first,
        _node_event(2, "circuit_break", "old", reason="orphan_recovery"),
        _node_event(3, "runner_start", "new", recovery_authorization_id="recover-1",
                    prior_node_ledger_sha256=source_sha),
        _node_event(4, "runner_stop", "new", status="authorized_draws_terminal"),
    ]
    _write_node(tmp_path, events)
    recovery = {
        "recover-1": {"node_id": "node0", "shard_remainder": 0,
                      "prior_node_ledger_sha256": source_sha}
    }
    result = close.validate_node(tmp_path, "node0", {}, set(), recovery)
    assert result["used_recovery_authorization_ids"] == ["recover-1"]
    recovery["recover-1"]["prior_node_ledger_sha256"] = "f" * 64
    with pytest.raises(close.ClosureError, match="recovery prefix"):
        close.validate_node(tmp_path, "node0", {}, set(), recovery)


def test_clean_paused_restart_needs_no_recovery(tmp_path: Path) -> None:
    events = [
        _node_event(1, "runner_start", "one"),
        _node_event(2, "runner_stop", "one", status="paused_operator"),
        _node_event(3, "runner_start", "two"),
        _node_event(4, "runner_stop", "two", status="authorized_draws_terminal"),
    ]
    _write_node(tmp_path, events)
    close.validate_node(tmp_path, "node0", {}, set(), {})


def test_pins_rebuilds_full_inventory_and_rejects_extra_cache(tmp_path: Path, monkeypatch) -> None:
    repo, features = tmp_path / "repo", tmp_path / "features"
    study, environment = repo / "v2_1" / "n14r2", repo / "v2_1" / "n14r2" / "environment"
    environment.mkdir(parents=True); features.mkdir()
    source = study / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(make_pins, "REUSED_FILES", ())
    monkeypatch.setattr(make_pins, "_tracked_study_files", lambda _: [source, *sorted(environment.glob("*.json"))])
    for index in range(4):
        contract = {
            "python": "3.12.3", "system": "Linux", "machine": "x86_64",
            "packages": dict(make_pins.CORE_PACKAGES), "torch_cuda": "12.8", "cudnn": 91900,
            "cpu_threads_per_worker": 1,
            "gpu": {"name": "NVIDIA GeForce RTX 4090", "uuid": f"GPU-{index}",
                    "driver_version": "580.1", "memory_total_mib": 24564},
            "cuda_health": {"device": "cuda:0", "checksum": 4.0}, "cuda_health_checksum": 4.0,
        }
        receipt = {
            "schema": make_pins.ENVIRONMENT_SCHEMA, "study_id": "N14R2", "node_id": f"node{index}",
            "shard_modulus": 4, "shard_remainder": index, "recorded_at": "2026-09-05T00:00:00Z",
            "contract": contract, "contract_sha256": make_pins.sha256_json(contract),
        }
        (environment / f"node{index}.json").write_text(json.dumps(receipt), encoding="utf-8")
    bindings = make_pins.build_environment_bindings(repo, environment)
    (study / "ENVIRONMENT_BINDINGS.json").write_text(json.dumps(bindings), encoding="utf-8")
    (features / "cremad__logmel__one.npz").write_bytes(b"npz")
    (features / "cremad__logmel__one.json").write_text("{}", encoding="utf-8")
    record = make_pins.build_record(repo, features, environment)
    assert make_pins.verify_record(record, repo, features, environment) == []
    source.write_text("changed\n", encoding="utf-8")
    assert make_pins.verify_record(record, repo, features, environment) == ["content_manifest_sha256", "files"]
    source.write_text("frozen\n", encoding="utf-8")
    (features / "cremad__logmel__extra.npz").write_bytes(b"extra")
    assert make_pins.verify_record(record, repo, features, environment)[0].startswith("inventory_rebuild:RuntimeError")

"""Versioned operator erratum; never modifies frozen source or original artifacts.

Only the output-only planned-draw count is corrected. Scientific verification
must still be rerun with the unmodified preregistered independent verifier.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ORIGINAL_SHA = "a49a9cd56f2fe64df0be96b568c24be63779e1d6126e38b791310b0fad506ec6"
FAILURE_SHA = "bdaf7745fd218115d038f852137b3aefbf84053403f0ebe5a5a1a1b53f646a0c"
PLAN_SHA = "165f79d939efbba52946d70c222d6df643c419ceb670c5e4b6fd1c184d0385c7"
DIFFERENCE = [{"actual": 24, "expected": 28, "path": "$.n_planned_draws"}]
RESULT_SCHEMA = "ser-v2.1-n14r2-results-1"
VERIFY_SCHEMA = "ser-v2.1-n14r2-verification-1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reject_links(path, *, regular=False):
    """Reject symlinks/junctions/reparse ancestors without resolving them."""
    path = Path(path).absolute()
    for current in reversed((path, *path.parents)):
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Symlink/junction/reparse path forbidden: {current}")
    if regular:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"Required input is missing: {path}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Required input is not a regular file: {path}")


def strict_int(value, expected):
    return type(value) is int and value == expected


def correct_bytes(original, failure, plan):
    for data, expected in ((original, ORIGINAL_SHA), (failure, FAILURE_SHA), (plan, PLAN_SHA)):
        if sha(data) != expected:
            raise ValueError("Input SHA-256 mismatch")
    report = json.loads(failure)
    differences = report.get("differences")
    exact_difference = (
        isinstance(differences, list) and len(differences) == 1
        and isinstance(differences[0], dict) and set(differences[0]) == {"actual", "expected", "path"}
        and differences[0].get("path") == "$.n_planned_draws"
        and strict_int(differences[0].get("actual"), 24)
        and strict_int(differences[0].get("expected"), 28)
    )
    if (not isinstance(report, dict) or report.get("schema") != VERIFY_SCHEMA
            or report.get("study_id") != "N14R2" or report.get("pass") is not False
            or not exact_difference):
        raise ValueError("Not the exact known single metadata verification failure")
    rows = list(csv.DictReader(io.StringIO(plan.decode("utf-8"))))
    try:
        counts = Counter(int(r["draw_id"]) for r in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Plan draw IDs are malformed") from exc
    if len(rows) != 2240 or counts != Counter({draw: 80 for draw in range(28)}):
        raise ValueError("Plan is not the exact 28-draw inventory")
    before = json.loads(original)
    if (not isinstance(before, dict) or before.get("schema") != RESULT_SCHEMA
            or before.get("spec_version") != "2.0.0"
            or before.get("tested") is not True
            or not strict_int(before.get("expected_complete_draws"), 24)
            or not strict_int(before.get("n_planned_draws"), 24)
            or not strict_int(before.get("n_complete_draws"), 24)
            or not isinstance(before.get("analysis_draw_ids"), list)
            or len(before["analysis_draw_ids"]) != 24
            or any(not strict_int(value, expected)
                   for expected, value in enumerate(before["analysis_draw_ids"]))
            or before.get("study_id") != "N14R2" or before.get("status") != "tested"):
        raise ValueError("Unexpected original result metadata")
    old, new = b'"n_planned_draws": 24', b'"n_planned_draws": 28'
    if original.count(old) != 1:
        raise ValueError("Replacement target is not unique")
    corrected = original.replace(old, new, 1)
    expected_object = dict(before)
    expected_object["n_planned_draws"] = 28
    after = json.loads(corrected)
    if (not strict_int(after.get("n_planned_draws"), 28)
            or after != expected_object):
        raise ValueError("Unexpected additional semantic change")
    changed = [i for i, (a, b) in enumerate(zip(original, corrected)) if a != b]
    if len(original) != len(corrected) or len(changed) != 1:
        raise ValueError("Expected exactly one changed byte")
    return corrected, changed[0]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = (args.original, args.failure, args.plan)
    targets = (args.output, args.receipt)
    for path in inputs:
        reject_links(path, regular=True)
    for path in targets:
        reject_links(path)
        reject_links(path.parent)
        if not path.parent.is_dir():
            raise ValueError("Target parent must already be a regular directory")
    identities = [os.path.normcase(os.path.abspath(p)) for p in inputs + targets]
    if len(set(identities)) != 5 or any(p.exists() or p.is_symlink() for p in targets):
        raise ValueError("Targets must be distinct, absent and not input aliases")
    original, failure, plan = [p.read_bytes() for p in inputs]
    corrected, offset = correct_bytes(original, failure, plan)
    receipt = {
        "schema": "ser26-n14r2-metadata-erratum-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_commit": "4fe0bb52f15b56a6626363be31a378b0f9661293",
        "original_result_sha256": ORIGINAL_SHA,
        "original_failed_verification_sha256": FAILURE_SHA,
        "run_plan_sha256": PLAN_SHA,
        "corrected_result_sha256": sha(corrected),
        "operator_script_sha256": sha(Path(__file__).read_bytes()),
        "change": {"path": "$.n_planned_draws", "from": 24, "to": 28,
                   "changed_bytes": 1, "zero_based_byte_offset": offset},
        "reason": "Frozen score.py:1256 counts the locked 24 analysis draws after filtering at 1061-1069; the frozen plan contains 28 planned draws (24 primary + 4 reserves), as verify.py:1292 correctly expects. This field is output-only.",
        "preserved": ["frozen code and PINS", "plan", "analysis lock", "original result",
                      "original failed verifier report", "all numerical and inferential fields"],
        "contains_outcome_statistics": False,
        "objects_equal_except_declared_field": True,
        "original_inputs_unchanged": True,
        "original_verification_pass": False,
        "corrected_verification": "pending; rerun the unmodified frozen independent verifier",
        "disclosure": "Correction was selected from the singleton structural mismatch and source inspection before scientific interpretation. During subsequent original-report inspection, numerical values were incidentally displayed after the analysis lock and after the metadata-only correction decision. No design, selection, calculation or stopping rule was changed."
    }
    with args.output.open("xb") as handle:
        handle.write(corrected)
        handle.flush()
        os.fsync(handle.fileno())
    if args.output.read_bytes() != corrected:
        raise ValueError("Corrected output readback mismatch")
    if [p.read_bytes() for p in inputs] != [original, failure, plan]:
        raise ValueError("Original inputs changed")
    with args.receipt.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(args.receipt.read_text(encoding="utf-8")) != receipt:
        raise ValueError("Erratum receipt readback mismatch")
    if [sha(p.read_bytes()) for p in inputs] != [ORIGINAL_SHA, FAILURE_SHA, PLAN_SHA]:
        raise ValueError("Original input SHA-256 changed after publication")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

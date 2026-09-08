"""Verify P1 protocol-premium summaries against every utterance artifact.

The verifier is CPU-only.  It re-runs the raw-artifact analysis in memory,
compares every published CSV byte-semantically, and applies additional
cross-table invariants for full-cell missingness, paired directions, the fixed
24-test Holm family, and strict-protocol speaker disjointness.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "protocol_premium"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import summarize_p1_protocol_premium as summary  # noqa: E402


class VerificationError(RuntimeError):
    """A published summary or a cross-table invariant failed verification."""


def load_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise VerificationError(f"missing summary table: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(expected_fields):
                raise VerificationError(
                    f"schema mismatch for {path.name}: observed={reader.fieldnames}, expected={list(expected_fields)}"
                )
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"JSON root is not an object: {path}")
    return value


def canonical_expected_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, str]]:
    return [{field: str(row.get(field, "")) for field in fields} for row in rows]


def assert_close(observed: str, expected: float, context: str, tolerance: float = 5e-15) -> None:
    if observed == "":
        raise VerificationError(f"missing numeric value: {context}")
    try:
        value = float(observed)
    except ValueError as exc:
        raise VerificationError(f"invalid numeric value {observed!r}: {context}") from exc
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise VerificationError(f"numeric mismatch for {context}: observed={value}, expected={expected}")


def assert_blank(row: Mapping[str, str], fields: Sequence[str], context: str) -> None:
    nonblank = {field: row[field] for field in fields if row.get(field, "") != ""}
    if nonblank:
        raise VerificationError(f"missing-cell propagation failed for {context}: {nonblank}")


def independent_holm(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    uar = [row for row in rows if row["metric"] == "uar"]
    if len(uar) != 24:
        raise VerificationError(f"expected fixed 24-row UAR family, found {len(uar)}")
    values: list[tuple[float, tuple[str, str, str], bool]] = []
    for row in uar:
        key = (row["corpus"], row["model"], row["contrast"])
        observed = row["status"] == "complete" and row["wilcoxon_p_raw"] != ""
        p = float(row["wilcoxon_p_raw"]) if observed else 1.0
        if not (0.0 <= p <= 1.0):
            raise VerificationError(f"invalid Wilcoxon p for {key}: {p}")
        values.append((p, key, observed))
    values.sort(key=lambda item: (item[0], item[1]))
    adjusted: dict[tuple[str, str, str], float] = {}
    running = 0.0
    for rank, (p, key, observed) in enumerate(values):
        running = max(running, min(1.0, (24 - rank) * p))
        if observed:
            adjusted[key] = running
    return adjusted


def semantic_checks(tables: Mapping[str, list[dict[str, str]]], manifest: Mapping[str, Any]) -> dict[str, int]:
    expected_counts = {
        "unit_completeness.csv": 1620,
        "cell_completeness.csv": 72,
        "oof_seed_metrics.csv": 72,
        "protocol_metric_summary.csv": 72,
        "speaker_seed_metrics.csv": sum(cfg["expected_speakers"] for cfg in summary.CORPORA.values()) * len(summary.MODELS) * len(summary.PROTOCOLS) * len(summary.SEEDS),
        "speaker_protocol_metrics.csv": sum(cfg["expected_speakers"] for cfg in summary.CORPORA.values()) * len(summary.MODELS) * len(summary.PROTOCOLS),
        "paired_speaker_differences.csv": sum(cfg["expected_speakers"] for cfg in summary.CORPORA.values()) * len(summary.MODELS) * len(summary.CONTRASTS),
        "protocol_premium.csv": len(summary.CORPORA) * len(summary.MODELS) * len(summary.CONTRASTS) * len(summary.METRICS),
        "outer_fold_speaker_overlap.csv": sum(sum(cfg["folds"].values()) for cfg in summary.CORPORA.values()),
        "budget_audit.csv": 1,
    }
    for name, count in expected_counts.items():
        if len(tables[name]) != count:
            raise VerificationError(f"{name} row count={len(tables[name])}, expected={count}")

    units_by_cell: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tables["unit_completeness.csv"]:
        units_by_cell[(row["corpus"], row["model"], row["protocol"], row["seed"])].append(row)
    cell_rows = {
        (row["corpus"], row["model"], row["protocol"], row["seed"]): row
        for row in tables["cell_completeness.csv"]
    }
    if len(cell_rows) != 72:
        raise VerificationError("duplicate cell-completeness key")
    complete_cell_keys: set[tuple[str, str, str, str]] = set()
    for key, cell in cell_rows.items():
        unit_rows = units_by_cell[key]
        expected_complete = bool(unit_rows) and all(row["status"] == "success" for row in unit_rows)
        if (cell["status"] == "complete") != expected_complete:
            raise VerificationError(f"cell status does not propagate unit status: {key}")
        if expected_complete:
            complete_cell_keys.add(key)

    oof_rows = {
        (row["corpus"], row["model"], row["protocol"], row["seed"]): row
        for row in tables["oof_seed_metrics.csv"]
    }
    if set(oof_rows) != set(cell_rows):
        raise VerificationError("OOF/cell key sets differ")
    for key, row in oof_rows.items():
        if key in complete_cell_keys:
            if row["status"] != "complete":
                raise VerificationError(f"complete cell absent from OOF metrics: {key}")
            for metric in summary.METRICS:
                value = float(row[metric])
                if not 0.0 <= value <= 1.0:
                    raise VerificationError(f"OOF {metric} out of range for {key}: {value}")
        else:
            if row["status"] == "complete":
                raise VerificationError(f"incomplete cell marked complete in OOF: {key}")
            assert_blank(row, ["n_utterances", "n_speakers", *summary.METRICS], f"OOF {key}")

    protocol_summary = tables["protocol_metric_summary.csv"]
    for row in protocol_summary:
        seed_keys = [(row["corpus"], row["model"], row["protocol"], str(seed)) for seed in summary.SEEDS]
        complete = all(key in complete_cell_keys for key in seed_keys)
        if (row["status"] == "complete") != complete:
            raise VerificationError(f"three-seed propagation mismatch: {row}")
        if complete:
            values = [float(oof_rows[key][row["metric"]]) for key in seed_keys]
            assert_close(row["mean"], sum(values) / 3.0, f"protocol mean {row}")
            mean = sum(values) / 3.0
            sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / 2.0)
            assert_close(row["sample_sd"], sample_sd, f"protocol sample SD {row}")
        else:
            assert_blank(row, ["mean", "sample_sd"], f"protocol summary {row}")

    speaker_protocol = {
        (row["corpus"], row["model"], row["protocol"], row["speaker_id"]): row
        for row in tables["speaker_protocol_metrics.csv"]
    }
    paired = tables["paired_speaker_differences.csv"]
    paired_by_group: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paired:
        group = (row["corpus"], row["model"], row["contrast"])
        paired_by_group[group].append(row)
        required = [
            (row["corpus"], row["model"], protocol, str(seed))
            for protocol in (row["protocol_a"], row["protocol_b"])
            for seed in summary.SEEDS
        ]
        complete = all(key in complete_cell_keys for key in required)
        if (row["status"] == "complete") != complete:
            raise VerificationError(f"paired missingness propagation mismatch: {group}/{row['speaker_id']}")
        if not complete:
            assert_blank(row, [f"{metric}_difference" for metric in summary.METRICS], f"paired {group}/{row['speaker_id']}")
            continue
        left = speaker_protocol[(row["corpus"], row["model"], row["protocol_a"], row["speaker_id"])]
        right = speaker_protocol[(row["corpus"], row["model"], row["protocol_b"], row["speaker_id"])]
        for metric in summary.METRICS:
            expected = float(left[f"{metric}_seed_mean"]) - float(right[f"{metric}_seed_mean"])
            assert_close(row[f"{metric}_difference"], expected, f"paired direction {group}/{row['speaker_id']}/{metric}")

    premium = tables["protocol_premium.csv"]
    holm = independent_holm(premium)
    for row in premium:
        group = (row["corpus"], row["model"], row["contrast"])
        paired_rows = paired_by_group[group]
        complete = bool(paired_rows) and all(item["status"] == "complete" for item in paired_rows)
        if (row["status"] == "complete") != complete:
            raise VerificationError(f"premium missingness propagation mismatch: {group}/{row['metric']}")
        if not complete:
            assert_blank(row, [
                "point_mean_difference", "speaker_difference_sample_sd", "ci95_low", "ci95_high",
                "wilcoxon_statistic", "wilcoxon_p_raw", "holm_adjusted_p", "holm_reject_0_05",
            ], f"premium {group}/{row['metric']}")
            continue
        values = [float(item[f"{row['metric']}_difference"]) for item in paired_rows]
        expected_mean = sum(values) / len(values)
        expected_sd = math.sqrt(sum((value - expected_mean) ** 2 for value in values) / (len(values) - 1))
        assert_close(row["point_mean_difference"], expected_mean, f"premium mean {group}/{row['metric']}")
        assert_close(row["speaker_difference_sample_sd"], expected_sd, f"premium SD {group}/{row['metric']}")
        if int(row["n_speakers"]) != summary.CORPORA[row["corpus"]]["expected_speakers"]:
            raise VerificationError(f"premium speaker n mismatch: {group}/{row['metric']}")
        if row["metric"] == "uar":
            if row["holm_family_size"] != "24":
                raise VerificationError(f"Holm family size drift: {group}")
            if row["wilcoxon_p_raw"] != "":
                assert_close(row["holm_adjusted_p"], holm[group], f"Holm adjusted p {group}")
                expected_reject = str(holm[group] <= summary.HOLM_ALPHA).lower()
                if row["holm_reject_0_05"] != expected_reject:
                    raise VerificationError(f"Holm decision mismatch: {group}")
            elif row["sign_test_p_sensitivity"] == "" or row["wilcoxon_error"] == "":
                raise VerificationError(f"failed Wilcoxon lacks preregistered sign-test sensitivity: {group}")
        else:
            assert_blank(row, [
                "wilcoxon_statistic", "wilcoxon_p_raw", "wilcoxon_error",
                "sign_test_p_sensitivity", "n_zero_differences", "holm_family_size",
                "holm_adjusted_p", "holm_reject_0_05",
            ], f"secondary inferential fields {group}/{row['metric']}")

    for row in tables["outer_fold_speaker_overlap.csv"]:
        if row["protocol"] != "random" and (row["n_overlap"] != "0" or row["assertion_pass"] != "true"):
            raise VerificationError(f"strict protocol speaker-overlap assertion failed: {row}")

    complete_count = len(complete_cell_keys)
    if int(manifest.get("complete_cells", -1)) != complete_count:
        raise VerificationError("analysis_manifest complete_cells mismatch")
    budget_pass = tables["budget_audit.csv"][0]["status"] == "pass"
    expected_status = "complete" if complete_count == 72 and budget_pass else "invalid_budget" if not budget_pass else "incomplete"
    if manifest.get("status") != expected_status:
        raise VerificationError("analysis_manifest status mismatch")
    return {"complete_cells": complete_count, **expected_counts}


def verify(result_root: Path, project_root: Path, summary_dir: Path) -> dict[str, Any]:
    manifest_path = summary_dir / "analysis_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "p1_protocol_premium_summary_v1":
        raise VerificationError(f"unsupported summary schema: {manifest.get('schema_version')!r}")
    if manifest.get("preregistration_frozen_prefix_sha256") != summary.PREREG_SHA256:
        raise VerificationError("analysis manifest preregistration hash mismatch")
    if manifest.get("summarizer_sha256") != summary.sha256_file(Path(summary.__file__).resolve()):
        raise VerificationError("summarizer changed since analysis_manifest was written")
    output_hashes = manifest.get("outputs")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(summary.TABLE_FIELDS):
        raise VerificationError("analysis_manifest output inventory mismatch")

    observed_tables: dict[str, list[dict[str, str]]] = {}
    for name, fields in summary.TABLE_FIELDS.items():
        path = summary_dir / name
        if summary.sha256_file(path) != output_hashes[name]:
            raise VerificationError(f"published table hash mismatch: {name}")
        observed_tables[name] = load_csv(path, fields)

    try:
        fresh = summary.analyze(result_root, project_root)
    except summary.AnalysisError as exc:
        raise VerificationError(f"raw-artifact reanalysis failed: {exc}") from exc
    for name, fields in summary.TABLE_FIELDS.items():
        expected = canonical_expected_rows(fresh.tables[name], fields)
        if observed_tables[name] != expected:
            mismatch = next(
                (
                    index,
                    observed_tables[name][index] if index < len(observed_tables[name]) else None,
                    expected[index] if index < len(expected) else None,
                )
                for index in range(max(len(observed_tables[name]), len(expected)))
                if index >= len(observed_tables[name]) or index >= len(expected) or observed_tables[name][index] != expected[index]
            )
            raise VerificationError(f"raw reanalysis mismatch in {name} at row {mismatch[0] + 2}: observed={mismatch[1]}, expected={mismatch[2]}")

    counts = semantic_checks(observed_tables, manifest)
    return {
        "schema_version": "p1_protocol_premium_verification_v1",
        "verified_at": summary.now_iso(),
        "status": "pass",
        "analysis_status": manifest["status"],
        "analysis_manifest_sha256": summary.sha256_file(manifest_path),
        "summarizer_sha256": summary.sha256_file(Path(summary.__file__).resolve()),
        "verifier_sha256": summary.sha256_file(Path(__file__).resolve()),
        "checks": {
            "all_output_hashes_match": True,
            "raw_utterance_reanalysis_exact_match": True,
            "full_cell_missingness_propagation": True,
            "speaker_paired_direction": True,
            "strict_protocol_speaker_disjointness": True,
            "fixed_holm_family_24": True,
        },
        "counts": counts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_dir = args.summary_dir or args.result_root / "summary"
    report_path = args.report or summary_dir / "verification.json"
    try:
        report = verify(args.result_root.resolve(), args.project_root.resolve(), summary_dir.resolve())
        summary.atomic_write_json(report_path, report)
    except (VerificationError, summary.AnalysisError) as exc:
        print(json.dumps({"status": "verification_failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "pass",
        "analysis_status": report["analysis_status"],
        "report": str(report_path),
    }, ensure_ascii=False, sort_keys=True))
    if args.require_complete and report["analysis_status"] != "complete":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

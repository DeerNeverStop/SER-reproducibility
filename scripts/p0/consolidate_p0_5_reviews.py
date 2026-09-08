#!/usr/bin/env python3
"""Consolidate the three frozen P0.5 cross-review batches.

This script only copies review records after checking their identity against the
frozen random order and the initial survey.  It does not adjudicate proposals.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
ORDER = ROOT / "p0_5_random_order.csv"
SURVEY = ROOT / "p0_5_random_survey.csv"
OUT = ROOT / "p0_5_review.csv"
HEADER = [
    "repository_url", "commit_sha", "draw_rank", "reviewer",
    "eligibility_verdict", "verdict", "field", "initial_value",
    "proposed_value", "evidence", "reason",
]
EXPECTED_BATCHES = {
    "p0_5_review_root_by_p06.csv",
    "p0_5_review_12_20_by_p08.csv",
    "p0_5_review_21_23_by_qa.csv",
    "p0_5_review_24_26_by_qb.csv",
    "p0_5_review_27_by_qc.csv",
    "p0_5_review_28_34_by_p07.csv",
}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def canonical(url: str) -> str:
    parsed = urlsplit(url.strip().rstrip("/"))
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.casefold() != "github.com" or len(parts) != 2:
        raise ValueError(f"invalid canonical repository URL: {url!r}")
    return f"https://github.com/{parts[0]}/{parts[1]}".casefold()


def main() -> None:
    paths = sorted(WORK / name for name in EXPECTED_BATCHES if (WORK / name).is_file())
    names = {path.name for path in paths}
    if names != EXPECTED_BATCHES:
        raise ValueError(f"review batch set mismatch: {sorted(names)}")

    _, order = read(ORDER)
    reached = [row for row in order if row["selection_status"] != "not_reached"]
    order_by_rank = {int(row["draw_rank"]): row for row in reached}
    _, survey = read(SURVEY)
    survey_by_url = {canonical(row["repository_url"]): row for row in survey}

    rows: list[dict[str, str]] = []
    for path in paths:
        header, batch = read(path)
        if header != HEADER:
            raise ValueError(f"header mismatch: {path.name}")
        rows.extend(batch)

    seen_repositories: set[str] = set()
    seen_proposals: set[tuple[str, str]] = set()
    for row in rows:
        rank = int(row["draw_rank"])
        if rank not in order_by_rank:
            raise ValueError(f"reviewed rank outside reached prefix: {rank}")
        frozen = order_by_rank[rank]
        key = canonical(row["repository_url"])
        if key != canonical(frozen["canonical_repo_url"]):
            raise ValueError(f"URL/rank mismatch at rank {rank}")
        if row["commit_sha"] != frozen["frozen_commit"]:
            raise ValueError(f"SHA/rank mismatch at rank {rank}")
        if row["eligibility_verdict"] != "agree":
            raise ValueError(f"unresolved eligibility disagreement at rank {rank}")
        if row["verdict"] not in {"agree", "disagree"}:
            raise ValueError(f"invalid verdict at rank {rank}")
        seen_repositories.add(key)

        if row["verdict"] == "agree":
            if any(row[field] for field in ("field", "initial_value", "proposed_value")):
                raise ValueError(f"agree row has proposal payload at rank {rank}")
            continue

        if not all(row[field] for field in ("field", "proposed_value", "evidence", "reason")):
            raise ValueError(f"incomplete disagreement at rank {rank}")
        if key not in survey_by_url:
            raise ValueError(f"field proposal targets excluded repository at rank {rank}")
        survey_row = survey_by_url[key]
        field = row["field"]
        if field not in survey_row:
            raise ValueError(f"unknown survey field {field!r} at rank {rank}")
        # Batch reviews were made against the pre-assembly audit rows.  The
        # assembler subsequently appended a deterministic provenance suffix to
        # ``notes`` only.  Reconcile that administrative suffix here so the
        # consolidated old value exactly matches the random survey and the
        # proposed value cannot erase assembly history.  Raw review files remain
        # unchanged and preserve what the second pass originally saw.
        if (
            field == "notes"
            and survey_row[field] != row["initial_value"]
            and survey_row[field].startswith(row["initial_value"] + " || draw_rank=")
        ):
            suffix = survey_row[field][len(row["initial_value"]):]
            row["initial_value"] = survey_row[field]
            if not row["proposed_value"].endswith(suffix):
                row["proposed_value"] += suffix
            row["reason"] += " Consolidation preserves the deterministic assembly-provenance suffix."
        if survey_row[field] != row["initial_value"]:
            raise ValueError(f"initial value mismatch at rank {rank}, field {field}")
        proposal_key = (key, field)
        if proposal_key in seen_proposals:
            raise ValueError(f"duplicate proposal at rank {rank}, field {field}")
        seen_proposals.add(proposal_key)

    expected_repositories = {canonical(row["canonical_repo_url"]) for row in reached}
    if seen_repositories != expected_repositories:
        missing = sorted(expected_repositories - seen_repositories)
        extra = sorted(seen_repositories - expected_repositories)
        raise ValueError(f"review coverage mismatch: missing={missing}, extra={extra}")

    rows.sort(key=lambda row: (int(row["draw_rank"]), row["verdict"] != "agree", row["field"]))
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    verdicts = Counter(row["verdict"] for row in rows)
    print(
        f"reviewed_repositories={len(seen_repositories)} rows={len(rows)} "
        f"agree_rows={verdicts['agree']} proposals={verdicts['disagree']} output={OUT.name}"
    )


if __name__ == "__main__":
    main()

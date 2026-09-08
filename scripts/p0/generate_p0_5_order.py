#!/usr/bin/env python3
"""Generate the preregistered deterministic-random P0.5 sampling order."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "candidate_log.csv"
FRAME = ROOT / "work" / "p0_5_frame_snapshot.csv"
ORDER = ROOT / "p0_5_random_order.csv"

EXPECTED_INPUT_SHA256 = "9f46ef0e0dc3a40910fa7bc030cbe393f1f60f0401ec30254a2a2d5514306938"
EXPECTED_ROWS = 383
EXPECTED_PENDING = 331
SEED = "202608101744"
PREFIX = f"P0.5|{SEED}|"

ORDER_FIELDS = [
    "draw_rank",
    "sampling_key_sha256",
    "candidate_id",
    "canonical_repo_url",
    "target_datasets",
    "discovery_source",
    "search_ids",
    "repo_relation",
    "original_notes",
    "selection_status",
    "screening_outcome",
    "exclusion_code",
    "exclusion_reason",
    "frozen_commit",
    "checked_at",
    "eligible_sequence",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(url: str) -> str:
    raw = url.strip().split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if raw.casefold().endswith(".git"):
        raw = raw[:-4]
    parsed = urlsplit(raw)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.casefold() != "github.com" or len(parts) != 2:
        raise ValueError(f"not a canonical GitHub repository URL: {url!r}")
    return f"https://github.com/{parts[0]}/{parts[1]}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    actual_hash = sha256(INPUT)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise SystemExit(f"candidate_log.csv SHA drift: {actual_hash}")
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"candidate row drift: {len(rows)}")
    counts = Counter(row["screening_status"] for row in rows)
    if counts != Counter({"pending": 331, "include": 32, "exclude": 20}):
        raise SystemExit(f"candidate status drift: {dict(counts)}")

    pending = [row.copy() for row in rows if row["screening_status"] == "pending"]
    if len(pending) != EXPECTED_PENDING:
        raise SystemExit(f"pending frame drift: {len(pending)}")
    normalized = [canonical(row["canonical_repo_url"]) for row in pending]
    keys = [value.casefold() for value in normalized]
    if len(set(keys)) != EXPECTED_PENDING:
        raise SystemExit("casefolded canonical URL duplication in pending frame")
    for row, value in zip(pending, normalized):
        if row["canonical_repo_url"].rstrip("/") != value:
            raise SystemExit(f"stored URL is not canonical: {row['canonical_repo_url']!r}")

    for row in pending:
        material = (PREFIX + row["canonical_repo_url"].casefold()).encode("utf-8")
        row["sampling_key_sha256"] = hashlib.sha256(material).hexdigest()
    pending.sort(
        key=lambda row: (
            row["sampling_key_sha256"],
            row["canonical_repo_url"].casefold(),
            row["canonical_repo_url"],
        )
    )

    frame_rows = sorted(
        ({field: row[field] for field in input_fields} for row in pending),
        key=lambda row: int(row["candidate_id"][1:]),
    )
    write_csv(FRAME, input_fields, frame_rows)

    order_rows: list[dict[str, str]] = []
    for rank, row in enumerate(pending, 1):
        order_rows.append(
            {
                "draw_rank": str(rank),
                "sampling_key_sha256": row["sampling_key_sha256"],
                "candidate_id": row["candidate_id"],
                "canonical_repo_url": row["canonical_repo_url"],
                "target_datasets": row["target_datasets"],
                "discovery_source": row["discovery_source"],
                "search_ids": row["search_ids"],
                "repo_relation": row["repo_relation"],
                "original_notes": row["notes"],
                "selection_status": "not_reached",
                "screening_outcome": "",
                "exclusion_code": "",
                "exclusion_reason": "",
                "frozen_commit": "",
                "checked_at": "",
                "eligible_sequence": "",
            }
        )
    write_csv(ORDER, ORDER_FIELDS, order_rows)

    print(
        f"input_sha256={actual_hash} frame={len(frame_rows)} order={len(order_rows)} "
        f"order_sha256={sha256(ORDER)}"
    )


if __name__ == "__main__":
    main()

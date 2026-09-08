#!/usr/bin/env python3
"""Assemble preregistered P0.5 batch decisions without adjudicating them.

The script requires a complete, gap-free sequence from draw rank 1 through the
rank at which the 30th eligible repository is reached.  It updates the sampling
log, assigns P05R IDs, and preserves every batch note.  Candidate code is never
imported or executed.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
SURVEY_HEADER = next(csv.reader((ROOT / "survey_table.csv").open(encoding="utf-8-sig", newline="")))
ORDER_PATH = ROOT / "p0_5_random_order.csv"
INITIAL_ORDER = WORK / "p0_5_random_order_initial.csv"
OUT_SURVEY = ROOT / "p0_5_random_survey.csv"
OUT_NOTES = ROOT / "p0_5_random_notes.md"

EXPECTED_INITIAL_SHA256 = "6b7c1b063033e85ee4051ac976392878c624fd4821372f5fd2483fb07a2ec2c6"
SIMPLE_ENUMS = {
    "split_category": {"random", "speaker_independent", "LOSO", "predefined_speaker_independent", "mixed", "unknown"},
    "speaker_disjointness": {"enforced", "not_enforced", "observed_overlap", "mixed", "unknown"},
    "normalization_leakage": {"yes", "no", "not_applicable", "mixed", "unknown"},
    "test_selection": {"yes_explicit", "test_exposed_each_epoch", "no_separate_validation", "no_fixed_training", "mixed", "unknown"},
    "augmentation_leakage": {"yes", "no_train_only", "not_applicable", "mixed", "unknown"},
    "variance_reported": {"sd", "se", "ci", "other", "none", "mixed", "unknown"},
    "evidence_strength": {"high", "medium", "low"},
}
EVIDENCE = (
    "result_evidence", "split_evidence", "normalization_evidence",
    "test_selection_evidence", "augmentation_evidence", "repetition_evidence",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def canonical(url: str) -> str:
    raw = url.strip().split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if raw.lower().endswith(".git"):
        raw = raw[:-4]
    parts = urlsplit(raw)
    path = [part for part in parts.path.split("/") if part]
    if parts.netloc.lower() != "github.com" or len(path) != 2:
        raise ValueError(f"noncanonical GitHub repository URL: {url!r}")
    return f"https://github.com/{path[0]}/{path[1]}".casefold()


def validate_audit(row: dict[str, str], source: Path) -> None:
    canonical(row["repository_url"])
    if len(row["commit_sha"]) != 40 or any(c not in "0123456789abcdef" for c in row["commit_sha"]):
        raise ValueError(f"{source.name}: invalid commit SHA for {row['repository_url']}")
    if row["repository_id"].strip():
        raise ValueError(f"{source.name}: batch repository_id must be blank")
    if row["verification_status"] != "pending":
        raise ValueError(f"{source.name}: expected verification_status=pending")
    for field, allowed in SIMPLE_ENUMS.items():
        if row[field] not in allowed:
            raise ValueError(f"{source.name}: invalid {field}={row[field]!r}")
    for field in EVIDENCE:
        if not row[field].strip():
            raise ValueError(f"{source.name}: blank {field} for {row['repository_url']}")
    for field in ("commit_date", "audited_at", "target_datasets", "primary_reviewer"):
        if not row[field].strip():
            raise ValueError(f"{source.name}: blank {field} for {row['repository_url']}")


def main() -> None:
    if not INITIAL_ORDER.exists():
        raise ValueError("missing immutable work/p0_5_random_order_initial.csv")
    if sha256(INITIAL_ORDER) != EXPECTED_INITIAL_SHA256:
        raise ValueError("initial order hash differs from preregistered file")
    initial_header, initial_rows = read_csv(INITIAL_ORDER)
    if len(initial_rows) != 331 or len({canonical(r["canonical_repo_url"]) for r in initial_rows}) != 331:
        raise ValueError("initial random frame must contain 331 unique repositories")

    audit_by_url: dict[str, tuple[dict[str, str], Path]] = {}
    for path in sorted(WORK.glob("p0_5_audit_*.csv")):
        if "review" in path.stem or "adjud" in path.stem:
            continue
        header, rows = read_csv(path)
        if header != SURVEY_HEADER:
            raise ValueError(f"{path.name}: expected exact 38-column survey header")
        for row in rows:
            validate_audit(row, path)
            key = canonical(row["repository_url"])
            if key in audit_by_url:
                raise ValueError(f"duplicate audited URL across batches: {row['repository_url']}")
            audit_by_url[key] = (row, path)

    exclusion_by_url: dict[str, tuple[dict[str, str], Path]] = {}
    for path in sorted(WORK.glob("p0_5_exclusions_*.csv")):
        _, rows = read_csv(path)
        for row in rows:
            required = {"draw_rank", "canonical_repo_url", "commit_sha", "checked_at", "exclusion_code", "exclusion_reason", "evidence"}
            if not required.issubset(row):
                raise ValueError(f"{path.name}: missing exclusion columns")
            key = canonical(row["canonical_repo_url"])
            if key in exclusion_by_url or key in audit_by_url:
                raise ValueError(f"duplicate/conflicting decision: {row['canonical_repo_url']}")
            if row["exclusion_code"] not in {f"E{i}_{suffix}" for i, suffix in [
                (1, "not_SER"), (2, "target_not_evaluated"), (3, "no_relevant_code"),
                (4, "unavailable"), (5, "duplicate_or_fork"), (6, "not_repository"),
                (7, "non_target_modality"), (8, "other"),
            ]}:
                raise ValueError(f"{path.name}: invalid exclusion code {row['exclusion_code']!r}")
            exclusion_by_url[key] = (row, path)

    cumulative = 0
    stop_rank: int | None = None
    selected: list[tuple[int, dict[str, str], Path]] = []
    updated: list[dict[str, str]] = []
    for raw in initial_rows:
        row = dict(raw)
        rank = int(row["draw_rank"])
        key = canonical(row["canonical_repo_url"])
        audit = audit_by_url.get(key)
        exclusion = exclusion_by_url.get(key)
        if stop_rank is None:
            if audit is None and exclusion is None:
                raise ValueError(f"decision gap before reaching n=30: draw_rank={rank} {row['canonical_repo_url']}")
            if audit is not None:
                audit_row, source = audit
                if audit_row["commit_sha"] == "" or key != canonical(audit_row["repository_url"]):
                    raise ValueError(f"audit/order mismatch at rank {rank}")
                cumulative += 1
                row.update({
                    "selection_status": "selected_eligible",
                    "screening_outcome": "include",
                    "exclusion_code": "", "exclusion_reason": "",
                    "frozen_commit": audit_row["commit_sha"],
                    "checked_at": audit_row["audited_at"],
                    "eligible_sequence": str(cumulative),
                })
                selected.append((rank, audit_row, source))
                if cumulative == 30:
                    stop_rank = rank
            else:
                ex_row, _ = exclusion  # type: ignore[misc]
                if int(ex_row["draw_rank"]) != rank:
                    raise ValueError(f"exclusion rank mismatch for {row['canonical_repo_url']}")
                row.update({
                    "selection_status": "screened_excluded",
                    "screening_outcome": "exclude",
                    "exclusion_code": ex_row["exclusion_code"],
                    "exclusion_reason": ex_row["exclusion_reason"],
                    "frozen_commit": ex_row["commit_sha"],
                    "checked_at": ex_row["checked_at"],
                    "eligible_sequence": "",
                })
        updated.append(row)

    if stop_rank is None or len(selected) != 30:
        raise ValueError(f"sample incomplete: eligible={len(selected)}")
    extra_decisions = []
    for raw in initial_rows[stop_rank:]:
        key = canonical(raw["canonical_repo_url"])
        if key in audit_by_url or key in exclusion_by_url:
            extra_decisions.append(raw["draw_rank"])
    if extra_decisions:
        raise ValueError(f"decisions after preregistered stop rank {stop_rank}: {extra_decisions}")
    decided_keys = {canonical(r["canonical_repo_url"]) for r in initial_rows[:stop_rank]}
    supplied_keys = set(audit_by_url) | set(exclusion_by_url)
    if decided_keys != supplied_keys:
        raise ValueError("batch decisions do not equal the gap-free prefix through stop rank")

    survey_rows: list[dict[str, str]] = []
    for sequence, (rank, raw, source) in enumerate(selected, start=1):
        row = dict(raw)
        row["repository_id"] = f"P05R{sequence:03d}"
        row["notes"] = (row["notes"] + " || " if row["notes"] else "") + f"draw_rank={rank}; initial_batch={source.name}"
        survey_rows.append(row)
    with OUT_SURVEY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SURVEY_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(survey_rows)
    with ORDER_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=initial_header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(updated)

    notes = [
        "# P0.5 随机样本逐库证据",
        "",
        f"按冻结顺序检查抽签位 1–{stop_rank}，补抽后得到 30 个合格库；其后 331-{stop_rank} 个位置保持 `not_reached`。",
        "本文件拼接各初审批次；最终结论须在 30/30 全字段交叉复核与主审裁决后读取主表。",
        "",
    ]
    for path in sorted(WORK.glob("p0_5_audit_*.md")):
        if "review" in path.stem or "adjud" in path.stem:
            continue
        notes += [f"<!-- source-batch: {path.name} -->", "", path.read_text(encoding="utf-8-sig").strip(), ""]
    OUT_NOTES.write_text("\n".join(notes).rstrip() + "\n", encoding="utf-8")
    print(f"stop_rank={stop_rank} eligible=30 excluded={stop_rank - 30} tail_not_reached={331 - stop_rank}")


if __name__ == "__main__":
    main()

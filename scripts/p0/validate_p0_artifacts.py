#!/usr/bin/env python3
"""Cross-file structural and traceability checks for the P0 audit artifacts."""

from __future__ import annotations

import csv
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]

SURVEY_ENUMS = {
    "split_category": {
        "random", "speaker_independent", "LOSO",
        "predefined_speaker_independent", "mixed", "unknown",
    },
    "normalization_leakage": {"yes", "no", "not_applicable", "mixed", "unknown"},
    "test_selection": {
        "yes_explicit", "test_exposed_each_epoch", "no_separate_validation",
        "no_fixed_training", "mixed", "unknown",
    },
    "augmentation_leakage": {"yes", "no_train_only", "not_applicable", "mixed", "unknown"},
    "variance_reported": {"sd", "se", "ci", "other", "none", "unknown"},
    "evidence_strength": {"high", "medium", "low"},
    "verification_status": {"verified_agree", "verified_corrected"},
}

REPETITION_ENUMS = {
    "single_split_single_seed", "single_split_multi_seed", "kfold_single_run",
    "kfold_multi_seed", "LOSO", "repeated_holdout", "mixed", "unknown",
}

TARGET_DATASETS = {"RAVDESS", "CREMA-D", "IEMOCAP", "EmoDB"}


def read_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def canonical_key(url: str) -> str:
    raw = url.strip().split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if raw.lower().endswith(".git"):
        raw = raw[:-4]
    parts = urlsplit(raw)
    segments = [part for part in parts.path.split("/") if part]
    if parts.netloc.lower() != "github.com" or len(segments) != 2:
        raise ValueError(f"noncanonical GitHub URL: {url!r}")
    return f"https://github.com/{segments[0]}/{segments[1]}".lower()


def duplicates(values: list[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if count > 1)


def local_source_map() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory in sorted((ROOT / "sources").iterdir()):
        if not directory.is_dir() or not (directory / ".git").exists():
            continue
        completed = subprocess.run(
            ["git", "-C", str(directory), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        result[canonical_key(completed.stdout.strip())] = directory
    return result


def resolve_evidence_path(source_root: Path, raw_relative: str) -> tuple[Path | None, str]:
    relative = raw_relative.strip().replace("\\", "/")
    candidates = [relative]
    words = relative.split()
    candidates.extend(" ".join(words[index:]) for index in range(1, len(words)))
    for item in candidates:
        path = source_root / Path(item)
        if path.exists():
            return path, item
    basename = Path(words[-1]).name if words else ""
    matches = list(source_root.rglob(basename)) if basename else []
    if len(matches) == 1:
        return matches[0], matches[0].relative_to(source_root).as_posix()
    return None, relative


def main() -> None:
    search_header, searches = read_csv("search_log.csv")
    candidate_header, candidates = read_csv("candidate_log.csv")
    survey_header, survey = read_csv("survey_table.csv")
    roster_header, roster = read_csv("P0_AUDIT_ROSTER.csv")

    errors: list[str] = []
    warnings: list[str] = []
    if len(search_header) != 9:
        errors.append(f"search_log header has {len(search_header)} columns, expected 9")
    if len(candidate_header) != 15:
        errors.append(f"candidate_log header has {len(candidate_header)} columns, expected 15")
    if len(survey_header) != 38:
        errors.append(f"survey_table header has {len(survey_header)} columns, expected 38")
    if len(roster_header) != 6:
        errors.append(f"P0_AUDIT_ROSTER header has {len(roster_header)} columns, expected 6")

    search_ids = [row["search_id"] for row in searches]
    errors.extend(f"duplicate search_id: {value}" for value in duplicates(search_ids))
    known_search_ids = set(search_ids)

    candidate_urls = [canonical_key(row["canonical_repo_url"]) for row in candidates]
    errors.extend(f"duplicate candidate URL: {value}" for value in duplicates(candidate_urls))
    expected_candidate_ids = [f"C{index:03d}" for index in range(1, len(candidates) + 1)]
    if [row["candidate_id"] for row in candidates] != expected_candidate_ids:
        errors.append("candidate_id sequence is not contiguous in file order")
    for row in candidates:
        dangling = [item for item in row["search_ids"].split("|") if item and item not in known_search_ids]
        if dangling:
            errors.append(f"{row['candidate_id']} has dangling search IDs: {dangling}")
        if row["screening_status"] not in {"pending", "include", "exclude", "awaiting_access"}:
            errors.append(f"{row['candidate_id']} invalid screening_status={row['screening_status']!r}")
        if row["screening_status"] == "exclude" and not row["exclusion_code"]:
            errors.append(f"{row['candidate_id']} is excluded without an exclusion code")

    roster_urls = [canonical_key(row["canonical_repo_url"]) for row in roster]
    errors.extend(f"duplicate roster URL: {value}" for value in duplicates(roster_urls))
    if [row["audit_order"] for row in roster] != [str(index) for index in range(1, len(roster) + 1)]:
        errors.append("audit_order sequence is not contiguous in file order")

    survey_urls = [canonical_key(row["repository_url"]) for row in survey]
    errors.extend(f"duplicate survey URL: {value}" for value in duplicates(survey_urls))
    expected_repo_ids = [f"R{index:03d}" for index in range(1, len(survey) + 1)]
    if [row["repository_id"] for row in survey] != expected_repo_ids:
        errors.append("repository_id sequence is not contiguous in file order")
    candidate_by_url = {canonical_key(row["canonical_repo_url"]): row for row in candidates}
    roster_by_url = {canonical_key(row["canonical_repo_url"]): row for row in roster}
    for url_key, roster_row in roster_by_url.items():
        status = roster_row["status"]
        if status == "included_audited":
            if url_key not in set(survey_urls):
                errors.append(f"included roster URL lacks survey row: {roster_row['canonical_repo_url']}")
        elif status.startswith("excluded_"):
            candidate = candidate_by_url.get(url_key)
            if not candidate or candidate["screening_status"] != "exclude":
                errors.append(f"excluded roster URL lacks excluded candidate: {roster_row['canonical_repo_url']}")
            elif status != f"excluded_{candidate['exclusion_code']}":
                errors.append(f"roster exclusion status mismatches candidate: {roster_row['canonical_repo_url']}")
        else:
            warnings.append(f"roster URL has nonfinal status {status}: {roster_row['canonical_repo_url']}")
    for url_key in survey_urls:
        if url_key not in roster_by_url:
            errors.append(f"survey URL absent from frozen roster: {url_key}")
    sources = local_source_map()
    line_pattern = re.compile(
        r"(?P<path>[A-Za-z0-9_.() /\\-]+\.(?:py|ipynb|md|yaml|yml|json|csv|txt|sh|ps1)):L(?P<start>\d+)(?:-L?(?P<end>\d+))?"
    )
    evidence_fields = (
        "result_evidence",
        "split_evidence",
        "normalization_evidence",
        "test_selection_evidence",
        "augmentation_evidence",
        "repetition_evidence",
    )
    for row in survey:
        url_key = canonical_key(row["repository_url"])
        if url_key not in candidate_by_url:
            errors.append(f"{row['repository_id']} is missing from candidate_log")
        elif candidate_by_url[url_key]["screening_status"] != "include":
            errors.append(f"{row['repository_id']} candidate status is not include")
        if not re.fullmatch(r"[0-9a-f]{40}", row["commit_sha"], re.IGNORECASE):
            errors.append(f"{row['repository_id']} invalid commit SHA")
        for field in evidence_fields:
            if not row[field].strip():
                errors.append(f"{row['repository_id']} blank {field}")
        for field, allowed in SURVEY_ENUMS.items():
            if row[field] not in allowed:
                errors.append(f"{row['repository_id']} invalid {field}={row[field]!r}")
        repetition_values = [item for item in row["evaluation_repetition"].split("|") if item]
        if not repetition_values or any(item not in REPETITION_ENUMS for item in repetition_values):
            errors.append(
                f"{row['repository_id']} invalid evaluation_repetition={row['evaluation_repetition']!r}"
            )
        for count_field in ("n_folds", "n_seeds"):
            if not re.fullmatch(r"(?:\d+(?:\|\d+)*)|unknown", row[count_field]):
                errors.append(f"{row['repository_id']} invalid {count_field}={row[count_field]!r}")
        target_values = row["target_datasets"].split("|")
        if (
            not target_values
            or len(target_values) != len(set(target_values))
            or any(value not in TARGET_DATASETS for value in target_values)
        ):
            errors.append(f"{row['repository_id']} invalid target_datasets={row['target_datasets']!r}")
        speaker_value = row["speaker_disjointness"]
        if speaker_value not in {"enforced", "not_enforced", "observed_overlap", "mixed", "unknown"}:
            mapped_values = [
                segment.split("=", 1)[1].strip()
                for segment in speaker_value.split("|")
                if "=" in segment
            ]
            if not mapped_values or any(
                value not in {"enforced", "not_enforced", "observed_overlap", "unknown"}
                for value in mapped_values
            ):
                errors.append(
                    f"{row['repository_id']} invalid speaker_disjointness={speaker_value!r}"
                )
        if url_key not in sources:
            warnings.append(f"{row['repository_id']} has no local git clone mapped by origin URL")
            continue
        source_root = sources[url_key]
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        if head != row["commit_sha"].lower():
            errors.append(
                f"{row['repository_id']} clone HEAD {head} differs from frozen commit {row['commit_sha']}"
            )
        worktree_status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if worktree_status:
            warnings.append(f"{row['repository_id']} frozen clone has local worktree changes")
        resolved_any = False
        for field in evidence_fields:
            for match in line_pattern.finditer(row[field]):
                candidate_path, relative = resolve_evidence_path(source_root, match.group("path"))
                if candidate_path is None:
                    # The permissive expression can see prose immediately before
                    # a genuine ``file.py:Lx`` reference as a second path (for
                    # example ``passes test but mlmodel.py:L41``).  Do not report
                    # that prose-only prefix as a missing file; the explicit path
                    # and frozen permalink are checked separately.
                    if " " in relative and "/" not in relative and "\\" not in relative:
                        continue
                    warnings.append(f"{row['repository_id']} unresolved evidence path: {relative}")
                    continue
                resolved_any = True
                try:
                    line_count = sum(1 for _ in candidate_path.open(encoding="utf-8", errors="replace"))
                except OSError as exc:
                    warnings.append(f"{row['repository_id']} could not read evidence path {relative}: {exc}")
                    continue
                end_line = int(match.group("end") or match.group("start"))
                if end_line > line_count:
                    warnings.append(
                        f"{row['repository_id']} evidence line {end_line} exceeds {relative} length {line_count}"
                    )
        if not resolved_any:
            errors.append(f"{row['repository_id']} has no machine-recognizable path:Lx evidence")

    included_candidates = {canonical_key(row["canonical_repo_url"]) for row in candidates if row["screening_status"] == "include"}
    for url in sorted(included_candidates - set(survey_urls)):
        errors.append(f"included candidate lacks survey row: {url}")

    roster_urls = [canonical_key(row["canonical_repo_url"]) for row in roster]
    errors.extend(f"duplicate roster URL: {value}" for value in duplicates(roster_urls))
    for row, url in zip(roster, roster_urls):
        if url not in candidate_by_url:
            errors.append(f"roster item {row['audit_order']} is missing from candidate_log: {url}")

    print(
        f"searches={len(searches)} candidates={len(candidates)} survey={len(survey)} "
        f"roster={len(roster)} clones={len(sources)}"
    )
    print(f"errors={len(errors)} warnings={len(warnings)}")
    for item in errors:
        print(f"ERROR: {item}")
    for item in warnings:
        print(f"WARN: {item}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

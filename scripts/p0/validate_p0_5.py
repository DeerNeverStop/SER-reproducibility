#!/usr/bin/env python3
"""Independent structural and evidence validation for P0.5 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
SURVEY = ROOT / "p0_5_random_survey.csv"
ORDER = ROOT / "p0_5_random_order.csv"
INITIAL = WORK / "p0_5_random_order_initial.csv"
NOTES = ROOT / "p0_5_random_notes.md"
EXPECTED_INITIAL_SHA = "6b7c1b063033e85ee4051ac976392878c624fd4821372f5fd2483fb07a2ec2c6"
EXPECTED_P1_SHA = "91db3edc544f76fc412f4dfc5f35adef23934635ca3ede36c8089c38bd14b782"
EVIDENCE_FIELDS = (
    "result_evidence", "split_evidence", "normalization_evidence",
    "test_selection_evidence", "augmentation_evidence", "repetition_evidence",
)
LINK_RE = re.compile(
    r"https://github\.com/([^/\s<>]+)/([^/\s<>]+)/blob/([0-9a-f]{40})/([^\s<>#]+)#L(\d+)(?:-L?(\d+))?",
    re.IGNORECASE,
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def canonical(url: str) -> str:
    raw = url.strip().split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if raw.lower().endswith(".git"):
        raw = raw[:-4]
    p = urlsplit(raw)
    parts = [x for x in p.path.split("/") if x]
    if p.netloc.lower() != "github.com" or len(parts) != 2:
        raise ValueError(f"bad repository URL {url!r}")
    return f"https://github.com/{parts[0]}/{parts[1]}".casefold()


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    check(sha(INITIAL) == EXPECTED_INITIAL_SHA, "initial random order SHA mismatch")
    check(sha(ROOT / "P1_PREREGISTRATION.md") == EXPECTED_P1_SHA, "P1 preregistration changed after freeze")
    initial_header, initial = read(INITIAL)
    order_header, order = read(ORDER)
    check(initial_header == order_header, "order header changed")
    check(len(initial) == len(order) == 331, "random order must retain 331 rows")
    frozen_fields = [
        "draw_rank", "sampling_key_sha256", "candidate_id", "canonical_repo_url",
        "target_datasets", "discovery_source", "search_ids", "repo_relation", "original_notes",
    ]
    for a, b in zip(initial, order):
        check(all(a[f] == b[f] for f in frozen_fields), f"frozen order identity changed at rank {a['draw_rank']}")
    reached = [r for r in order if r["selection_status"] != "not_reached"]
    included_order = [r for r in reached if r["screening_outcome"] == "include"]
    excluded_order = [r for r in reached if r["screening_outcome"] == "exclude"]
    check(len(reached) == 34 and [int(r["draw_rank"]) for r in reached] == list(range(1, 35)), "screened prefix must be ranks 1-34")
    check(len(included_order) == 30 and len(excluded_order) == 4, "expected 30 include and 4 exclude")
    check(all(r["selection_status"] == "not_reached" for r in order[34:]), "tail after rank34 must be not_reached")
    check([int(r["eligible_sequence"]) for r in included_order] == list(range(1, 31)), "eligible sequence must be 1..30")

    central_header = next(csv.reader((ROOT / "survey_table.csv").open(encoding="utf-8-sig", newline="")))
    survey_header, survey = read(SURVEY)
    check(survey_header == central_header and len(survey_header) == 38, "random survey must use exact 38-column schema")
    check(len(survey) == 30, "random survey n must be 30")
    check(len({canonical(r["repository_url"]) for r in survey}) == 30, "random survey URLs not unique")
    check([r["repository_id"] for r in survey] == [f"P05R{i:03d}" for i in range(1, 31)], "P05 repository IDs not sequential")
    allowed_status = {"pending", "verified_agree", "verified_corrected"}
    check(all(r["verification_status"] in allowed_status for r in survey), "invalid verification status")
    for row in survey:
        check(re.fullmatch(r"[0-9a-f]{40}", row["commit_sha"]) is not None, f"bad SHA {row['repository_id']}")
        check(all(row[f].strip() for f in EVIDENCE_FIELDS), f"blank evidence field {row['repository_id']}")
        traceable = re.compile(r"(?::L\d+|#L\d+|\.pdf:p(?:\.|\d)|\bPDF pp?\.)", re.IGNORECASE)
        check(all(traceable.search(row[f]) for f in EVIDENCE_FIELDS), f"evidence lacks file:line/page {row['repository_id']}")

    order_by_url = {canonical(r["canonical_repo_url"]): r for r in order}
    inventory = {canonical(r["canonical_repo_url"]): r for r in read(WORK / "p0_5_inventory.csv")[1]}
    for row in survey:
        key = canonical(row["repository_url"])
        check(key in order_by_url and order_by_url[key]["screening_outcome"] == "include", f"survey/order mismatch {row['repository_id']}")
        check(key in inventory and inventory[key]["commit_sha"] == row["commit_sha"], f"inventory SHA mismatch {row['repository_id']}")
        rank = int(order_by_url[key]["draw_rank"])
        matches = list((ROOT / "sources_p0_5").glob(f"{rank:04d}__*"))
        check(len(matches) == 1, f"missing/ambiguous clone rank {rank}")
        if len(matches) == 1:
            proc = subprocess.run(
                ["git", "-c", "safe.directory=*", "-C", str(matches[0]), "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            check(proc.returncode == 0 and proc.stdout.strip() == row["commit_sha"], f"clone HEAD mismatch rank {rank}")

    text = NOTES.read_text(encoding="utf-8-sig")
    links_by_repo: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    valid_link_count = 0
    for match in LINK_RE.finditer(text):
        owner, repo, commit, encoded_path, start, end = match.groups()
        key = canonical(f"https://github.com/{owner}/{repo}")
        path = unquote(encoded_path)
        links_by_repo[(key, commit.lower())].append((path, int(start), int(end or start)))
        order_row = order_by_url.get(key)
        if not order_row:
            continue
        rank = int(order_row["draw_rank"])
        dirs = list((ROOT / "sources_p0_5").glob(f"{rank:04d}__*"))
        if len(dirs) != 1:
            continue
        local = dirs[0] / Path(path)
        if not local.is_file():
            warnings.append(f"historical permalink path missing rank {rank}: {path}")
            continue
        line_count = sum(1 for _ in local.open(encoding="utf-8", errors="replace"))
        if int(start) < 1 or int(end or start) > max(1, line_count):
            warnings.append(f"historical permalink line out of range rank {rank}: {path}#{start}-{end}")
            continue
        valid_link_count += 1
    for row in survey:
        key = (canonical(row["repository_url"]), row["commit_sha"])
        check(bool(links_by_repo.get(key)), f"no frozen permalink in notes for {row['repository_id']}")
    check(valid_link_count >= 30, f"too few valid frozen permalinks: {valid_link_count}")

    review_paths = sorted(WORK.glob("p0_5_review_*.csv"))
    if review_paths:
        expected_review_header = [
            "repository_url", "commit_sha", "draw_rank", "reviewer", "eligibility_verdict",
            "verdict", "field", "initial_value", "proposed_value", "evidence", "reason",
        ]
        reviewed: set[str] = set()
        proposals = 0
        for path in review_paths:
            header, rows = read(path)
            check(header == expected_review_header, f"review header mismatch {path.name}")
            for row in rows:
                key = canonical(row["repository_url"])
                reviewed.add(key)
                check(row["eligibility_verdict"] in {"agree", "disagree"}, f"bad eligibility verdict {path.name}")
                check(row["verdict"] in {"agree", "disagree"}, f"bad field verdict {path.name}")
                if row["verdict"] == "disagree":
                    proposals += 1
                    check(bool(row["field"] and row["proposed_value"] and row["evidence"] and row["reason"]), f"incomplete proposal {path.name}")
        expected = {canonical(r["canonical_repo_url"]) for r in reached}
        check(reviewed == expected, f"review coverage mismatch reviewed={len(reviewed)} expected=34")
        check(len(review_paths) == 6, f"expected six independent review batch files, got {len(review_paths)}")
    else:
        warnings.append("cross-review files not yet present")
        proposals = 0

    consolidated_path = ROOT / "p0_5_review.csv"
    adjudication_path = ROOT / "p0_5_adjudications.csv"
    if consolidated_path.is_file() and adjudication_path.is_file():
        consolidated_header, consolidated = read(consolidated_path)
        check(consolidated_header == expected_review_header, "consolidated review header mismatch")
        consolidated_proposals = [row for row in consolidated if row["verdict"] == "disagree"]
        check(len({canonical(row["repository_url"]) for row in consolidated}) == 34, "consolidated review repository coverage mismatch")
        check(len(consolidated_proposals) == proposals, "raw/consolidated proposal count mismatch")

        expected_adjudication_header = [
            "repository_id", "repository_url", "commit_sha", "draw_rank", "field",
            "initial_value", "review_proposed_value", "adjudicated_value", "decision",
            "adjudicated_at", "reviewer", "review_sha256", "review_evidence", "rationale",
        ]
        adjudication_header, adjudications = read(adjudication_path)
        check(adjudication_header == expected_adjudication_header, "adjudication header mismatch")
        check(len(adjudications) == len(consolidated_proposals), "adjudication proposal count mismatch")
        review_by_key = {
            (canonical(row["repository_url"]), row["field"]): row
            for row in consolidated_proposals
        }
        adjudication_by_key = {
            (canonical(row["repository_url"]), row["field"]): row
            for row in adjudications
        }
        check(len(review_by_key) == len(consolidated_proposals), "duplicate consolidated field proposal")
        check(len(adjudication_by_key) == len(adjudications), "duplicate adjudication field")
        check(set(review_by_key) == set(adjudication_by_key), "review/adjudication key mismatch")
        final_survey_by_url = {canonical(row["repository_url"]): row for row in survey}
        for key, adjudication in adjudication_by_key.items():
            review_row = review_by_key[key]
            check(adjudication["initial_value"] == review_row["initial_value"], f"adjudication initial mismatch {key}")
            check(adjudication["review_proposed_value"] == review_row["proposed_value"], f"adjudication proposal mismatch {key}")
            check(adjudication["decision"] in {"accepted", "rejected", "refined"}, f"invalid adjudication decision {key}")
            check(final_survey_by_url[key[0]][key[1]] == adjudication["adjudicated_value"], f"final survey/adjudication mismatch {key}")
        status_counts = Counter(row["verification_status"] for row in survey)
        check(status_counts == Counter({"verified_corrected": 10, "verified_agree": 20}), f"unexpected final verification statuses: {status_counts}")
        check("<!-- P0_5_ADJUDICATIONS_START -->" in text, "adjudication marker absent from notes")
    else:
        warnings.append("consolidated review/adjudication artifacts not yet present")

    p06_header, p06 = read(ROOT / "p0_6_search_log.csv")
    prereg = [r for r in p06 if r["run_role"] == "preregistered"]
    check(len(prereg) == 16 and len(p06) == 19, "P0.6 request accounting mismatch")
    check(all(r["http_status"] == "429" for r in prereg), "P0.6 prereg requests not all 429")
    check(len({r["request_id"] for r in prereg}) == 8, "P0.6 query IDs not 8")
    check(all(sum(x["request_id"] == r["request_id"] for x in prereg) == 2 for r in prereg), "P0.6 query attempt count not 2")
    check(len(read(ROOT / "p0_6_alternative_candidates.csv")[1]) == 0, "P0.6 candidate table should be empty after unavailable API")

    p08 = json.loads((ROOT / "results" / "p0_8" / "verification.json").read_text(encoding="utf-8"))
    check(p08["verdict"] == "pass" and p08["errors"] == 0 and p08["warnings"] == 1, "P0.8 verification not pass/0/1")

    verdict = "PASS" if not errors else "FAIL"
    print(f"P0.5 validation {verdict}: checks={checks} errors={len(errors)} warnings={len(warnings)} proposals={proposals}")
    for item in errors:
        print(f"ERROR: {item}")
    for item in warnings:
        print(f"WARNING: {item}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

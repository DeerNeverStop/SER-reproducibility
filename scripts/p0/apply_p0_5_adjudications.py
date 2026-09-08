#!/usr/bin/env python3
"""Apply explicitly reviewed P0.5 field adjudications once.

The consolidated review SHA is required on the command line so a main-audit
decision cannot silently target a moving review file.  By default every field
proposal is accepted; any rejected/refined proposal must be listed in the
optional override CSV before this script is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "p0_5_review.csv"
SURVEY = ROOT / "p0_5_random_survey.csv"
NOTES = ROOT / "p0_5_random_notes.md"
ADJUDICATIONS = ROOT / "p0_5_adjudications.csv"
ADJUDICATION_MD = ROOT / "p0_5_adjudications.md"
OVERRIDES = ROOT / "work" / "p0_5_adjudication_overrides.csv"
MARKER = "<!-- P0_5_ADJUDICATIONS_START -->"
OUTPUT_FIELDS = [
    "repository_id", "repository_url", "commit_sha", "draw_rank", "field",
    "initial_value", "review_proposed_value", "adjudicated_value", "decision",
    "adjudicated_at", "reviewer", "review_sha256", "review_evidence", "rationale",
]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def canonical(url: str) -> str:
    parsed = urlsplit(url.strip().rstrip("/"))
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.casefold() != "github.com" or len(parts) != 2:
        raise ValueError(f"invalid repository URL: {url!r}")
    return f"https://github.com/{parts[0]}/{parts[1]}".casefold()


def load_overrides() -> dict[tuple[int, str], dict[str, str]]:
    if not OVERRIDES.is_file():
        return {}
    expected = ["draw_rank", "field", "decision", "adjudicated_value", "rationale"]
    header, rows = read(OVERRIDES)
    if header != expected:
        raise ValueError("adjudication override header mismatch")
    output: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["draw_rank"]), row["field"])
        if key in output or row["decision"] not in {"accepted", "rejected", "refined"}:
            raise ValueError(f"invalid or duplicate adjudication override: {key}")
        if not row["adjudicated_value"] or not row["rationale"]:
            raise ValueError(f"incomplete adjudication override: {key}")
        output[key] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-review-sha", required=True)
    args = parser.parse_args()
    actual_review_sha = file_sha(REVIEW)
    if actual_review_sha != args.expected_review_sha.casefold():
        raise ValueError(
            f"review SHA mismatch: expected {args.expected_review_sha}, got {actual_review_sha}"
        )
    if ADJUDICATIONS.exists() or ADJUDICATION_MD.exists():
        raise FileExistsError("adjudication output already exists; refuse to overwrite history")
    notes_text = NOTES.read_text(encoding="utf-8-sig")
    if MARKER in notes_text:
        raise ValueError("adjudication marker already exists in notes")

    _, review = read(REVIEW)
    survey_header, survey = read(SURVEY)
    survey_by_url = {canonical(row["repository_url"]): row for row in survey}
    overrides = load_overrides()
    proposals = [row for row in review if row["verdict"] == "disagree"]
    if not proposals:
        raise ValueError("no review proposals to adjudicate")
    if any(row["eligibility_verdict"] != "agree" for row in review):
        raise ValueError("eligibility disagreement must be resolved before field adjudication")

    adjudicated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S UTC%z")
    adjudicated_at = adjudicated_at[:-2] + ":" + adjudicated_at[-2:]
    output: list[dict[str, str]] = []
    corrected_urls: set[str] = set()
    proposal_keys: set[tuple[int, str]] = set()
    for proposal in proposals:
        rank = int(proposal["draw_rank"])
        field = proposal["field"]
        key = (rank, field)
        if key in proposal_keys:
            raise ValueError(f"duplicate review proposal: {key}")
        proposal_keys.add(key)
        repo_key = canonical(proposal["repository_url"])
        if repo_key not in survey_by_url:
            raise ValueError(f"proposal targets non-included repository: rank {rank}")
        row = survey_by_url[repo_key]
        if row["commit_sha"] != proposal["commit_sha"] or row[field] != proposal["initial_value"]:
            raise ValueError(f"proposal no longer matches survey: rank {rank}, field {field}")

        override = overrides.get(key)
        if override:
            decision = override["decision"]
            value = override["adjudicated_value"]
            rationale = f"{proposal['reason']} | Main adjudication: {override['rationale']}"
        else:
            decision = "accepted"
            value = proposal["proposed_value"]
            rationale = proposal["reason"]

        row[field] = value
        corrected_urls.add(repo_key)
        output.append({
            "repository_id": row["repository_id"],
            "repository_url": row["repository_url"],
            "commit_sha": row["commit_sha"],
            "draw_rank": str(rank),
            "field": field,
            "initial_value": proposal["initial_value"],
            "review_proposed_value": proposal["proposed_value"],
            "adjudicated_value": value,
            "decision": decision,
            "adjudicated_at": adjudicated_at,
            "reviewer": proposal["reviewer"],
            "review_sha256": actual_review_sha,
            "review_evidence": proposal["evidence"],
            "rationale": rationale,
        })

    unused_overrides = set(overrides) - proposal_keys
    if unused_overrides:
        raise ValueError(f"override does not match a proposal: {sorted(unused_overrides)}")

    reviewed_included = {
        canonical(row["repository_url"])
        for row in review
        if canonical(row["repository_url"]) in survey_by_url
    }
    if reviewed_included != set(survey_by_url):
        raise ValueError("not all 30 included repositories have a cross-review record")
    for repo_key, row in survey_by_url.items():
        row["verification_status"] = (
            "verified_corrected" if repo_key in corrected_urls else "verified_agree"
        )

    with SURVEY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=survey_header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(survey)
    with ADJUDICATIONS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)

    by_repo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in output:
        by_repo[row["repository_id"]].append(row)
    md = [
        "# P0.5 交叉复核裁决记录",
        "",
        f"- 裁决时间：`{adjudicated_at}`",
        f"- 合并复核 SHA-256：`{actual_review_sha}`",
        f"- 覆盖：30/30 纳入库及 4/4 顺序排除项；字段裁决 {len(output)} 项。",
        "- 原值永久保留在 `p0_5_review.csv` 与本表 `initial_value`；随机主表只写裁决后值。",
        "- 复核是同一自动化代理系统的不同审计遍次，不是独立人类编码者。",
        "",
    ]
    for repository_id in sorted(by_repo):
        rows = sorted(by_repo[repository_id], key=lambda row: row["field"])
        md.extend([f"## {repository_id} — {rows[0]['repository_url']}", ""])
        for row in rows:
            md.extend([
                f"- `{row['field']}`: `{row['initial_value']}` → `{row['adjudicated_value']}` (`{row['decision']}`)",
                f"  - 依据：{row['review_evidence']}",
                f"  - 理由：{row['rationale']}",
            ])
        md.append("")
    ADJUDICATION_MD.write_text("\n".join(md), encoding="utf-8")

    note_lines = [
        "",
        MARKER,
        "## P0.5 全量交叉复核与主审裁决",
        "",
        f"裁决时间：`{adjudicated_at}`；合并复核 SHA-256：`{actual_review_sha}`。",
        f"30/30 纳入库与 4/4 顺序排除项均由不同审计遍次复核；{len(output)} 个字段经主审裁决。",
        "旧值、建议值、最终值、证据与理由完整保存在 `p0_5_review.csv` 和 `p0_5_adjudications.csv`。",
        "这不是两名独立人类编码者；作者盲编码 8–10 库仍为待作者事项。",
        "",
    ]
    NOTES.write_text(notes_text.rstrip() + "\n" + "\n".join(note_lines), encoding="utf-8")

    status_counts = Counter(row["verification_status"] for row in survey)
    decision_counts = Counter(row["decision"] for row in output)
    print(
        f"adjudicated_fields={len(output)} corrected_repositories={len(corrected_urls)} "
        f"statuses={dict(status_counts)} decisions={dict(decision_counts)}"
    )


if __name__ == "__main__":
    main()

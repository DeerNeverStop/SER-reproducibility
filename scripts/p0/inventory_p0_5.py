#!/usr/bin/env python3
"""Create a static, non-executing inventory of already cloned P0.5 candidates."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDER = ROOT / "p0_5_random_order.csv"
SOURCES = ROOT / "sources_p0_5"
OUTPUT = ROOT / "work" / "p0_5_inventory.csv"
CODE_SUFFIXES = {".py", ".ipynb", ".m", ".r", ".R", ".jl", ".sh", ".ps1"}
SKIP_PARTS = {".git", "__pycache__", ".ipynb_checkpoints"}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return result.stdout.strip()


def main() -> None:
    with ORDER.open(encoding="utf-8", newline="") as handle:
        by_rank = {int(row["draw_rank"]): row for row in csv.DictReader(handle)}

    records: list[dict[str, str | int]] = []
    for repo in sorted(SOURCES.glob("[0-9][0-9][0-9][0-9]__*")):
        rank = int(repo.name[:4])
        row = by_rank[rank]
        files = [
            p
            for p in repo.rglob("*")
            if p.is_file() and not any(part in SKIP_PARTS for part in p.relative_to(repo).parts)
        ]
        code = [p for p in files if p.suffix in CODE_SUFFIXES]
        readmes = [p for p in files if p.name.lower().startswith("readme")]
        licenses = [
            p
            for p in files
            if p.name.lower().startswith(("license", "copying"))
        ]
        records.append(
            {
                "draw_rank": rank,
                "candidate_id": row["candidate_id"],
                "canonical_repo_url": row["canonical_repo_url"],
                "target_datasets": row["target_datasets"],
                "commit_sha": git(repo, "rev-parse", "HEAD"),
                "commit_date": git(repo, "show", "-s", "--format=%cI", "HEAD"),
                "file_count": len(files),
                "code_file_count": len(code),
                "code_paths": "|".join(str(p.relative_to(repo)).replace("\\", "/") for p in code[:30]),
                "readme_paths": "|".join(str(p.relative_to(repo)).replace("\\", "/") for p in readmes),
                "license_paths": "|".join(str(p.relative_to(repo)).replace("\\", "/") for p in licenses),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {OUTPUT} rows={len(records)}")


if __name__ == "__main__":
    main()

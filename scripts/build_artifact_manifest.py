#!/usr/bin/env python3
"""Build a deterministic SHA-256 inventory for the paper artifact tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifact_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versioned_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.STDOUT
    )
    relative_paths = [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]
    return [ROOT / path for path in relative_paths if ROOT / path != OUTPUT]


def main() -> None:
    files = versioned_files()
    entries = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )

    document = {
        "schema": "ser-icassp2027-artifact-manifest-v1",
        "source_commit": "d11768ed31a91f611472b89e52f3f6bb97030c40",
        "n_files": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT}: {len(entries)} files, {total_bytes} bytes")


if __name__ == "__main__":
    main()

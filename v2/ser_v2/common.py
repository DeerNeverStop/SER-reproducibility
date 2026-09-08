"""Shared helpers: canonical JSON, hashing, atomic writes, stable seeds."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_u32(text: str) -> int:
    """First 32 bits of SHA-256(text) as an unsigned integer (seed table primitive)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def atomic_write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path | str, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: Path | str, fieldnames: list[str], rows: Iterable[dict]) -> None:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    atomic_write_text(path, buf.getvalue())


def read_csv(path: Path | str) -> list[dict[str, str]]:
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

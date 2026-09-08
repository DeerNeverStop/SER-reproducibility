"""Convert the frozen P1 manifests (results/protocol_premium/manifests/*.csv) into v2 manifests.

The v2 split tables depend only on filename metadata (speaker, label, sentence, take, sex),
so the RAVDESS and CREMA-D plan can be generated and pinned from these files before any
audio is touched; byte sizes and SHA-256 are carried over from the P1 manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_csv, write_csv
from .corpora import CORPORA, MANIFEST_FIELDS, parse_filename


def convert(corpus: str, src: Path, dst: Path) -> int:
    spec = CORPORA[corpus]
    label_index = {lab: i for i, lab in enumerate(spec.labels)}
    rows = []
    for r in read_csv(src):
        if r.get("parse_status", "ok") != "ok" or r.get("read_status", "ok") != "ok":
            continue
        parsed = parse_filename(corpus, r["relative_path"])
        if parsed is None:
            raise ValueError(f"unparseable {r['relative_path']}")
        if corpus == "ravdess":
            # P1 labels are the same eight names; keep P1's label_index to stay byte-comparable
            assert parsed["label"] == r["label_name"], (parsed, r)
        elif corpus == "cremad":
            assert parsed["label"] == r["label_name"], (parsed, r)
        rows.append({"corpus": corpus, "relative_path": r["relative_path"], "bytes": int(r["file_size_bytes"]),
                     "sha256": r["sha256"], **parsed, "label_index": label_index[parsed["label"]]})
    rows.sort(key=lambda x: x["relative_path"])
    for i, r in enumerate(rows):
        r["sample_index"] = i
    write_csv(dst, MANIFEST_FIELDS, rows)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1-manifests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for corpus in ("ravdess", "cremad"):
        n = convert(corpus, a.p1_manifests / f"{corpus}_manifest.csv", a.out / f"{corpus}_manifest.csv")
        print(corpus, n, "rows")


if __name__ == "__main__":
    main()

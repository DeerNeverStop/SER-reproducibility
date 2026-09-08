"""Make the scorer's numeric_insert.tex usable by LaTeX.

The scorer (frozen at tag-1) writes macro names such as \\HN01Mean; LaTeX control sequences
cannot contain digits. This tool rewrites ONLY the digits inside macro names to words
(\\HN01Mean -> \\HNZeroOneMean, \\ClaimC2a -> \\ClaimCTwoa) and leaves every value byte-identical,
so the verifier's check of the original file still applies and the paper file is derived
mechanically from it.

    python -m tools.tex_macros --in runs/main/numeric_insert.tex --out paper/v2/numeric_insert_paper.tex
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
PAT = re.compile(r"\\newcommand\{\\([A-Za-z0-9]+)\}")


def rename(name: str) -> str:
    return "".join(WORDS.get(ch, ch) for ch in name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    out, n = [], 0
    for line in a.src.read_text(encoding="utf-8").splitlines():
        m = PAT.match(line)
        if m:
            new = rename(m.group(1))
            if new != m.group(1):
                n += 1
            line = line.replace("{\\" + m.group(1) + "}", "{\\" + new + "}", 1)
        out.append(line)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(("% derived from " + a.src.name + " by tools.tex_macros (digit->word in macro NAMES only; values untouched)\n" + "\n".join(out) + "\n").encode("utf-8"))
    print(f"{n} macro names renamed -> {a.out}")


if __name__ == "__main__":
    main()

"""Number check for paper/v2/main.tex.

1. Every macro of the families \\H..., \\Claim..., \\D..., \\S..., \\Tab... used in main.tex is defined in one
   of the generated inserts (numeric_insert_paper.tex, descriptor_macros.tex, tables_v2.tex, carried_audit.tex).
2. The generated inserts are byte-identical to a fresh regeneration from runs/<run>/results.json and the
   descriptor JSON files (regeneration determinism; nothing was hand-edited).
3. numeric_insert_paper.tex carries exactly the values of the scorer's numeric_insert.tex (only macro names differ).
4. Digit literals that appear in the prose of main.tex outside macros are listed for the author (they must be
   design constants such as 4,943 or 10,000, never results).

    python -m tools.check_paper_numbers --run runs/main --paper ../paper/v2 --plan plan_rc2 --registry registry
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import sha256_file  # noqa: E402

MACRO_USE = re.compile(r"\\((?:H[NRT]|Claim|D|S|Tab)[A-Za-z]+)")
MACRO_DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--paper", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    a = ap.parse_args(argv)
    py = sys.executable
    tex = (a.paper / "main.tex").read_text(encoding="utf-8")
    body = re.sub(r"(?m)^%.*$", "", tex)
    used = sorted(set(MACRO_USE.findall(body)))
    defined = set()
    for f in ("numeric_insert_paper.tex", "descriptor_macros.tex", "tables_v2.tex", "carried_audit.tex"):
        defined |= set(MACRO_DEF.findall((a.paper / f).read_text(encoding="utf-8")))
    undefined = [m for m in used if m not in defined]
    print(f"[1] macros used {len(used)}, undefined {len(undefined)}: {undefined}")
    # [2] regeneration determinism
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        subprocess.run([py, "-m", "tools.tex_macros", "--in", str(a.run / "numeric_insert.tex"), "--out", str(td / "numeric_insert_paper.tex")], cwd=V2, check=True, capture_output=True)
        subprocess.run([py, "-m", "tools.paper_tables", "--plan", str(a.plan), "--registry", str(a.registry), "--run", str(a.run), "--out", str(td)], cwd=V2, check=True, capture_output=True)
        subprocess.run([py, "-m", "tools.paper_descriptors", "--run", str(a.run), "--out", str(td / "descriptor_macros.tex")], cwd=V2, check=True, capture_output=True)
        ok = True
        for f in ("numeric_insert_paper.tex", "tables_v2.tex", "descriptor_macros.tex"):
            same = sha256_file(td / f) == sha256_file(a.paper / f)
            ok &= same
            print(f"[2] {f}: {'identical' if same else 'DIFFERS from regeneration'}")
    # [3] values equal the scorer insert
    orig = dict(re.findall(r"\\newcommand\{\\([A-Za-z0-9]+)\}\{(.*)\}", (a.run / "numeric_insert.tex").read_text(encoding="utf-8")))
    paper = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (a.paper / "numeric_insert_paper.tex").read_text(encoding="utf-8")))
    from tools.tex_macros import rename
    bad = [k for k, v in orig.items() if paper.get(rename(k)) != v]
    print(f"[3] scorer insert values carried: {len(orig) - len(bad)}/{len(orig)}; mismatches: {bad}")
    # [4] digit literals in prose
    prose = re.sub(r"\\begin\{thebibliography\}.*", "", body, flags=re.S)
    prose = re.sub(r"\\(?:cite|ref|label|texttt|includegraphics)\{[^}]*\}", "", prose)
    lits = sorted(set(re.findall(r"(?<![A-Za-z\\{])\d[\d,\.]*", prose)))
    print(f"[4] digit literals in prose ({len(lits)}): {lits}")
    print("RESULT", "PASS" if (not undefined and ok and not bad) else "FAIL")


if __name__ == "__main__":
    main()

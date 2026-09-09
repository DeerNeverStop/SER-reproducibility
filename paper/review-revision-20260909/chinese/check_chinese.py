"""Check a Chinese explanation against its fixed English numeric source and PDF."""
from pathlib import Path
from decimal import Decimal
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import re

import pdfplumber
from pypdf import PdfReader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_values(text):
    text = text.replace("−", "-").replace("＋", "+").replace("\\!", "")
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^" + m[0].translate(superscripts), text)
    text = text.replace("\\times", "×").replace("$", "")
    text = re.sub(r"([0-9.]+)\s*[×x]\s*10\s*\^\s*\{?([+-]?\d+)\}?", r"\1e\2", text)
    return [Decimal(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parents[2]
    en_path = folder.parent / "english/main.tex"
    md_path = folder / "中文解读.md"
    source = en_path.read_text(encoding="utf-8")
    md = md_path.read_text(encoding="utf-8")
    start = source.index(r"\label{tab:primary}")
    stop = source.index(r"\end{table*}", start)
    en_rows = [line.split(" & ") for line in source[start:stop].splitlines()
               if re.match(r"^(CREMA-D|SUBESCO|RAVDESS) &", line)]
    md_rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
               for line in md.splitlines() if line.startswith("|")]
    prim_rows = [r for r in md_rows if r[0] in {"CREMA-D", "SUBESCO", "RAVDESS"}
                 and len(r) == 6 and "[" in r[3]]
    primary_same = len(prim_rows) == len(en_rows) == 16 and all(
        a[0] == b[0] and numeric_values(" ".join(a[2:])) == numeric_values(" ".join(b[2:]))
        for a, b in zip(en_rows, prim_rows))
    absolute = (folder.parent / "ABSOLUTE_RESULTS.md").read_text(encoding="utf-8")
    abs_rows = [[c.strip() for c in line.strip().strip("|").split("|")]
                for line in absolute.splitlines() if line.startswith("|")]
    abs_rows = [r for r in abs_rows if len(r) == 9 and r[0] in {"CREMA-D", "SUBESCO", "RAVDESS"}]
    zh_abs = [r for r in md_rows if len(r) == 9 and r[0] in {"CREMA-D", "SUBESCO", "RAVDESS"}]
    absolute_same = len(abs_rows) == len(zh_abs) == 11 and all(
        a[:3] == b[:3] and numeric_values(" ".join(a[3:])) == numeric_values(" ".join(b[3:]))
        for a, b in zip(abs_rows, zh_abs))
    spec = importlib.util.spec_from_file_location("font_inspection", root / "paper/submission-20260906/tools/check_pdfs.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reader = PdfReader(args.pdf)
    fonts, hidden, font_errors = helper.font_inventory(reader)
    with pdfplumber.open(args.pdf) as document:
        texts = [p.extract_text() or "" for p in document.pages]
        violations = []
        page_reports = []
        for n, page in enumerate(document.pages, 1):
            chars = [c for c in page.chars if c["text"].strip()]
            for c in chars:
                if c["x0"] < 40 or c["x1"] > page.width - 40 or c["top"] < 10 or c["bottom"] > page.height - 10:
                    violations.append({"page": n, "text": c["text"], "bbox": [c["x0"], c["top"], c["x1"], c["bottom"]]})
            page_reports.append({"page": n, "text_characters": len(texts[n-1]),
                                 "size_pt": [page.width, page.height],
                                 "font_sizes": dict(Counter(str(round(c["size"],2)) for c in chars))})
    checks = {
        "all_16_primary_numeric_rows": primary_same,
        "all_11_absolute_numeric_rows": absolute_same,
        "final_english_source_identity": digest(en_path) == "043ebfaddb925910b74005fff77b61192d50b5d798931e09fa2512fff6b957a7",
        "fonts_embedded_no_type3": bool(fonts) and all(f["embedded"] and not f["type3"] for f in fonts) and not font_errors,
        "no_invisible_text": not hidden,
        "no_layout_escape": not violations,
        "nonempty_pages": all(len(t.strip()) > 30 for t in texts),
        "unicode_chinese_text": all(any("\u4e00" <= ch <= "\u9fff" for ch in t) for t in texts),
        "no_replacement_characters": not any("\ufffd" in t or "(cid:" in t for t in texts),
    }
    report = {"schema": "ser-chinese-companion-qa-20260909-1", "pass": all(checks.values()),
              "source_markdown_sha256": digest(md_path), "english_source_sha256": digest(en_path),
              "pdf_sha256": digest(args.pdf), "checker_sha256": digest(Path(__file__)),
              "pages": len(reader.pages), "checks": checks, "fonts": fonts,
              "page_reports": page_reports, "layout_violations": violations,
              "scope": "Numeric translation checks and machine PDF inspection. Does not replace human-language review or visual page inspection; no new scientific tests or training."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pass": report["pass"], "pages": report["pages"], "failed": [k for k,v in checks.items() if not v],
                      "primary_rows": len(prim_rows), "absolute_rows": len(zh_abs), "pdf_sha256": report["pdf_sha256"]}))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

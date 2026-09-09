"""Read-only numeric and PDF checks; visual inspection remains a separate step."""
import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_rows(source):
    start = source.index(r"\label{tab:primary}")
    end = source.index(r"\end{table*}", start)
    return [line for line in source[start:end].splitlines()
            if re.match(r"^(CREMA-D|SUBESCO|RAVDESS) &", line)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    current = root / "paper/review-revision-20260909/english/main.tex"
    previous = root / "paper/supplement-results-20260908/english/main.tex"
    source = current.read_text(encoding="utf-8")
    old_source = previous.read_text(encoding="utf-8")
    log = args.log.read_text(encoding="utf-8", errors="replace")
    # Reuse the existing recursive font inventory, which inspects embedded forms.
    module_path = root / "paper/submission-20260906/tools/check_pdfs.py"
    spec = importlib.util.spec_from_file_location("existing_pdf_qa", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = PdfReader(args.pdf)
    fonts, invisible, font_errors = module.font_inventory(reader)
    with pdfplumber.open(args.pdf) as pdf:
        page_texts = [page.extract_text(x_tolerance=1, y_tolerance=2) or "" for page in pdf.pages]
        figure_chars = [c for page in pdf.pages for c in page.chars
                        if "DejaVuSans" in c.get("fontname", "") and c["text"].strip()]
        figure_sizes = []
        for c in figure_chars:
            if c["upright"]:
                figure_sizes.append(c["size"])
            else:
                # pdfminer reports the advance of a 90-degree rotated glyph as
                # `size`; the physical x extent is its transformed font height.
                a, b, cc, d, _, _ = c["matrix"]
                assert abs(a) < 1e-8 and abs(d) < 1e-8 and b * cc < 0
                figure_sizes.append(c["width"])
        positions = []
        geometry_errors = []
        for index, page in enumerate(pdf.pages, 1):
            characters = [c for c in page.chars if c["text"].strip()]
            positions.append({"page": index,
                              "size_pt": [page.width, page.height],
                              "min_x_pt": min(c["x0"] for c in characters),
                              "max_x_pt": max(c["x1"] for c in characters),
                              "min_top_pt": min(c["top"] for c in characters),
                              "max_bottom_pt": max(c["bottom"] for c in characters),
                              "font_size_counts": dict(Counter(str(round(c["size"], 3)) for c in characters))})
            for c in characters:
                # Official 178 mm text area and small font-metric tolerance.
                if c["x0"] < 51.8 or c["x1"] > 561.0 or c["top"] < 69.5 or c["bottom"] > 726.0:
                    geometry_errors.append({"page": index, "text": c["text"],
                                            "bbox": [c["x0"], c["top"], c["x1"], c["bottom"]]})
        checks = {}
        module.abstract_and_authors(pdf.pages[0], checks, "Tian Xie",
                                    "tianjack.xie@mail.utoronto.ca", "University of Toronto")
    plain_last = page_texts[-1].upper()
    checks.update({
        "five_pages": {"pass": len(reader.pages) == 5, "pages": len(reader.pages)},
        "all_16_primary_rows_unchanged": {"pass": len(primary_rows(source)) == 16
                                         and primary_rows(source) == primary_rows(old_source)},
        "compiler_no_overfull_or_missing_references": {
            "pass": not re.search(r"Overfull|Undefined control sequence|undefined references|Citation .* undefined|Missing character", log)},
        "all_fonts_embedded_no_type3": {"pass": bool(fonts)
                                        and all(f["embedded"] and not f["type3"] for f in fonts)
                                        and not font_errors},
        "no_invisible_text": {"pass": not invisible},
        "figure_final_font_at_least_9pt": {"pass": bool(figure_sizes) and min(figure_sizes) >= 9.0,
                                           "min_size_pt": min(figure_sizes, default=None),
                                           "method": "Final PDF transformed font height; upright size or x extent for checked 90-degree y-axis labels."},
        "text_inside_layout": {"pass": not geometry_errors, "violations": geometry_errors},
        "AI_disclosure_on_technical_pages": {"pass": any("ACKNOWLEDGMENTS AND AI DISCLOSURE" in t
                                                           for t in page_texts[:4])
                                             and "AI DISCLOSURE" not in plain_last},
        "page5_declarations_and_references": {
            "pass": all(t in plain_last for t in ["COMPLIANCE WITH ETHICAL STANDARDS", "FUNDING AND CONFLICTS OF INTEREST", "REFERENCES"])
                    and not any(t in plain_last for t in ["Fig. 1".upper(), "2. EXPERIMENTAL DESIGN", "3. RESULTS", "5. CONCLUSION"]),
            "limitation": "Heading check only; final page contents require visual review."},
        "public_artifact_reference": {"pass": "SER-reproducibility/releases/tag/v2026.09.09" in source
                                      and "https://github.com/DeerNeverStop/SER}" not in source},
        "separate_Holm_families": {"pass": "Holm6" in source and "Holm10" in source},
    })
    report = {"schema": "ser-manuscript-checks-20260909-1",
              "pass": all(v["pass"] for v in checks.values()),
              "source_sha256": sha(current), "previous_source_sha256": sha(previous),
              "pdf_sha256": sha(args.pdf), "compiler_log_sha256": sha(args.log),
              "checker_sha256": sha(Path(__file__)),
              "checks": checks, "fonts": fonts, "page_geometry": positions,
              "scope": "Manuscript numeric preservation and machine PDF checks. Not submission, ethics approval, new statistical evidence, or independent retraining. Visual review and public-link availability are checked separately."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "failed": [k for k, v in checks.items() if not v["pass"]],
                      "pages": len(reader.pages), "pdf_sha256": report["pdf_sha256"],
                      "figure_font_min_pt": checks["figure_final_font_at_least_9pt"]["min_size_pt"]}))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

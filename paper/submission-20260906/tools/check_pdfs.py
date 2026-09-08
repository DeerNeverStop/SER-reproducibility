"""Read-only PDF delivery QA; machine checks do not replace page-by-page review."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
from pypdf import PdfReader
from pypdf.generic import ContentStream

LEFT = 72 - 6.2 * 72 / 25.4
WIDTH = 178 * 72 / 25.4
GUTTER = 6 * 72 / 25.4
RIGHT = LEFT + WIDTH
COL_RIGHT = LEFT + (WIDTH - GUTTER) / 2
COL_LEFT = COL_RIGHT + GUTTER
TOP = 72
BOTTOM = TOP + 229 * 72 / 25.4
TOLERANCE = 2.0


def obj(value):
    return value.get_object() if hasattr(value, "get_object") else value


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def text(page):
    # Times word spaces can be narrower than pdfplumber's default 3pt tolerance.
    return page.extract_text(x_tolerance=1.0, y_tolerance=2.0) or ""


def words(page):
    return page.extract_words(x_tolerance=1.0, y_tolerance=2.0)


def column_text(page):
    return "\n".join(text(page.crop((left, 0, right, page.height))) for left, right in (
        (LEFT - TOLERANCE, COL_RIGHT + TOLERANCE),
        (COL_LEFT - TOLERANCE, RIGHT + TOLERANCE)))


def coordinate(item):
    return {k: round(float(item[k]), 3) for k in ("x0", "top", "x1", "bottom")}


def check(checks, name, passed, **evidence):
    checks[name] = {"pass": bool(passed), **evidence}


def font_inventory(reader):
    inventory = []
    invisible = []
    resource_errors = []

    def visit(resources, content, page_number, label, seen):
        resources = obj(resources or {})
        for name, reference in obj(resources.get("/Font", {})).items():
            font = obj(reference)
            subtype = str(font.get("/Subtype", ""))
            descendants = [obj(f) for f in obj(font.get("/DescendantFonts", []))]
            leaves = descendants or [font]
            descriptors = [obj(f.get("/FontDescriptor", {})) for f in leaves]
            embedded = all(any(key in d and hasattr(obj(d[key]), "get_data")
                               and bool(obj(d[key]).get_data())
                               for key in ("/FontFile", "/FontFile2", "/FontFile3"))
                           for d in descriptors)
            inventory.append({"page": page_number, "resource": f"{label}/{name}",
                              "name": str(font.get("/BaseFont", "")),
                              "subtype": subtype, "embedded": embedded,
                              "type3": subtype == "/Type3" or any(
                                  str(f.get("/Subtype")) == "/Type3" for f in leaves)})
        try:
            stream = ContentStream(content, reader) if content is not None else None
            mode, state_stack = 0, []
            for args, op in ([] if stream is None else stream.operations):
                if op == b"q":
                    state_stack.append(mode)
                elif op == b"Q" and state_stack:
                    mode = state_stack.pop()
                elif op == b"Tr":
                    mode = int(args[0])
                elif op in (b"Tj", b"TJ", b"'", b'"') and mode in (3, 7):
                    invisible.append({"page": page_number, "resource": label,
                                      "text_rendering_mode": mode})
        except Exception as exc:
            resource_errors.append({"page": page_number, "resource": label,
                                    "error": f"{type(exc).__name__}: {exc}"})
        for name, reference in obj(resources.get("/XObject", {})).items():
            form = obj(reference)
            if form.get("/Subtype") != "/Form":
                continue
            identity = (getattr(reference, "idnum", None), id(form))
            if identity in seen:
                continue
            visit(form.get("/Resources", resources), form, page_number,
                  f"{label}/{name}", seen | {identity})

    for number, page in enumerate(reader.pages, 1):
        visit(page.get("/Resources", {}), page.get_contents(), number, "page", set())
    return inventory, invisible, resource_errors


def abstract_and_authors(page, checks, expected_author, expected_email, affiliation):
    items = words(page)
    headings = [w for w in items if w["text"].strip().upper() == "ABSTRACT"]
    heading = next((w for w in headings if w["x1"] <= COL_RIGHT + TOLERANCE), None)
    boundary = heading["top"] if heading else page.height * 0.30
    panel = text(page.crop((0, 0, page.width, boundary)))
    check(checks, "expected_author", compact(expected_author) in compact(panel),
          expected=expected_author, panel_text=panel)
    check(checks, "expected_email", compact(expected_email) in compact(panel),
          expected=expected_email)
    check(checks, "expected_affiliation", compact(affiliation) in compact(panel),
          expected=affiliation)
    keywords = [w for w in items if w["text"].casefold() == "index" and heading
                and w["top"] > heading["bottom"] and w["x0"] < COL_RIGHT]
    if not heading or not keywords:
        check(checks, "abstract_100_to_150_words", False,
              reason="Could not locate left-column ABSTRACT and Index Terms boundaries")
        return boundary
    stop = min(keywords, key=lambda w: w["top"])
    region = (LEFT - TOLERANCE, heading["bottom"] + 0.1,
              COL_RIGHT + TOLERANCE, stop["top"] - 0.1)
    abstract = text(page.crop(region))
    # Join end-of-line hyphenation; retain intra-line compound words as one word.
    abstract = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    count = len(re.findall(r"\S*[A-Za-z0-9]\S*", abstract))
    check(checks, "abstract_100_to_150_words", 100 <= count <= 150,
          word_count=count, method="whitespace tokens containing letters/digits, after line-end dehyphenation",
          extracted_abstract=abstract, region=list(region))
    return boundary


def references_page_five(page, checks, manual):
    columns = [page.crop((LEFT - TOLERANCE, 0, COL_RIGHT + TOLERANCE, page.height)),
               page.crop((COL_LEFT - TOLERANCE, 0, RIGHT + TOLERANCE, page.height))]
    texts = [text(c) for c in columns]
    first_line = next((line.strip() for line in texts[0].splitlines() if line.strip()), "")
    heading_ok = bool(re.fullmatch(r"(?:\d+\.?\s*)?REFERENCES", first_line, re.I))
    forbidden = re.findall(r"(?im)^\s*(?:\d+\.?\s*)?(?:ABSTRACT|INTRODUCTION|METHODS?|RESULTS|"
                           r"DISCUSSION|CONCLUSION|ACKNOWLEDGMENTS?|AUTHOR RESPONSIBILITY)\s*$",
                           "\n".join(texts))
    citation_labels = re.findall(r"(?m)^\s*\[(\d+)\]", "\n".join(texts))
    check(checks, "page5_references_only_heuristic", heading_ok and not forbidden and bool(citation_labels),
          first_left_column_line=first_line, citation_labels=citation_labels,
          forbidden_nonreference_headings=forbidden,
          limitation="Column-order heading/citation checks cannot prove every line is bibliographic content")
    manual.append("Visually confirm all page 5 content is references; automatic checks are heuristic.")


def layout_checks(pages, checks, title_bottom, english):
    page_reports, physical_outside, layout_outside, gutter_items, footers = [], [], [], [], []
    for number, page in enumerate(pages, 1):
        chars = [c for c in page.chars if c.get("text", "").strip()]
        sizes = Counter(round(float(c["size"]), 2) for c in chars)
        boxes = []
        for c in chars:
            if c["x0"] < -0.5 or c["top"] < -0.5 or c["x1"] > page.width + .5 or c["bottom"] > page.height + .5:
                physical_outside.append({"page": number, "text": c["text"], **coordinate(c)})
            # LaTeX textheight ends at the final baseline, not the bottom of
            # font-descender metrics. Nimbus 10pt extends ~2.162pt below it.
            # Retain strict glyph checks against the physical page separately.
            matrix = c.get("matrix")
            baseline = page.height - float(matrix[5]) if matrix and c.get("upright") else c["bottom"]
            if english and (c["x0"] < LEFT - TOLERANCE or c["x1"] > RIGHT + TOLERANCE
                            or c["top"] < TOP - TOLERANCE or baseline > BOTTOM + TOLERANCE):
                layout_outside.append({"page": number, "text": c["text"],
                                       "baseline_top_pt": round(baseline, 3), **coordinate(c)})
        for w in words(page):
            if english and not (number == 1 and w["top"] < title_bottom):
                if w["x0"] < COL_LEFT - TOLERANCE and w["x1"] > COL_RIGHT + TOLERANCE:
                    gutter_items.append({"page": number, "text": w["text"], **coordinate(w)})
        # A bottom-most strip, deliberately below the spconf body box. Never
        # classify ordinary numbered equations/tables/footnotes in the body.
        footer_page = page.crop((0, page.height - 45, page.width, page.height))
        for line in text(footer_page).splitlines():
            value = line.strip()
            if re.fullmatch(r"(?:page\s*)?(?:\d{1,3}|[ivxlcdm]{1,8})(?:\s*(?:/|of)\s*\d{1,3})?", value, re.I):
                footers.append({"page": number, "text": value, "region_top": page.height - 45})
        if chars:
            boxes = [round(min(c["x0"] for c in chars), 3), round(min(c["top"] for c in chars), 3),
                     round(max(c["x1"] for c in chars), 3), round(max(c["bottom"] for c in chars), 3)]
        page_reports.append({"page": number, "width_pt": float(page.width), "height_pt": float(page.height),
                             "text_bbox": boxes, "font_size_character_counts": dict(sorted(sizes.items()))})
    check(checks, "no_text_outside_physical_pages", not physical_outside,
          count=len(physical_outside), examples=physical_outside[:30])
    if english:
        check(checks, "text_inside_spconf_layout_box", not layout_outside,
              bounds_pt=[LEFT, TOP, RIGHT, BOTTOM], tolerance_pt=TOLERANCE,
              vertical_rule="Glyph top must fit; final baseline must fit textheight; font descenders may extend below that baseline",
              count=len(layout_outside), examples=layout_outside[:30])
        check(checks, "no_printed_footer_page_numbers_heuristic", not footers,
              candidates=footers, limitation="Checks standalone numeral/page labels only in the bottom 45pt strip")
    return page_reports, gutter_items


def inspect(path, *, english, args):
    path = path.absolute()
    checks, manual = {}, ["Root must inspect every rendered page; this report is not visual approval."]
    result = {"path": str(path), "kind": "english" if english else "chinese", "checks": checks,
              "manual_review": manual}
    if not path.is_file():
        result.update(status="missing", error="PDF does not exist")
        return result
    before = path.read_bytes()
    result.update(sha256=hashlib.sha256(before).hexdigest(), size_bytes=len(before))
    check(checks, "size_below_5MB", len(before) < 5_000_000, maximum_exclusive_bytes=5_000_000)
    reader = PdfReader(path)
    check(checks, "unencrypted", not reader.is_encrypted)
    if reader.is_encrypted:
        result["status"] = "fail"
        return result
    inventory, invisible, resource_errors = font_inventory(reader)
    result["fonts"] = inventory
    check(checks, "all_fonts_embedded", bool(inventory) and all(f["embedded"] for f in inventory))
    check(checks, "no_type3_fonts", not any(f["type3"] for f in inventory))
    check(checks, "no_nonpainting_text_render_modes_detected", not invisible,
          occurrences=invisible, limitation="Modes 3 and 7 checked; transparency/occlusion are not exhaustively tested")
    check(checks, "pdf_resource_inspection_succeeded", not resource_errors, errors=resource_errors)
    with pdfplumber.open(path) as pdf:
        result["page_count"] = len(pdf.pages)
        title_bottom = 0
        if english:
            check(checks, "at_most_five_pages", 1 <= len(pdf.pages) <= 5, maximum=5)
            check(checks, "all_pages_US_letter", all(abs(p.width - 612) <= .5 and abs(p.height - 792) <= .5 for p in pdf.pages))
            title_bottom = abstract_and_authors(pdf.pages[0], checks, args.expected_author,
                                               args.expected_email, args.expected_affiliation)
            if len(pdf.pages) == 5:
                references_page_five(pdf.pages[4], checks, manual)
            else:
                checks["page5_references_only_heuristic"] = {"pass": True, "not_applicable": True}
            all_text = "\n".join(text(p) for p in pdf.pages)
            pending_patterns = (r"awaiting\s+(?:the\s+)?author(?:s)?\s+(?:review|approval)",
                                r"(?:author\s+review|author\s+approval)\s+pending", r"pending\s+author\s+review",
                                r"not\s+yet\s+reviewed\s+by\s+the\s+author", r"author\s+to\s+review",
                                r"待作者", r"作者待阅", r"作者尚未")
            found = [m.group(0) for pattern in pending_patterns for m in re.finditer(pattern, all_text, re.I)]
            check(checks, "no_pending_author_review_placeholders", not found, matches=found)
            responsibility = [line.strip() for line in column_text(pdf.pages[3]).splitlines()
                              if "responsib" in line.casefold() or "author" in line.casefold()] if len(pdf.pages) >= 4 else []
            result["page4_author_responsibility_lines"] = responsibility
            manual.append("Confirm the responsibility statement is truthful; text matching cannot certify author review or consent.")
        result["pages"], gutter = layout_checks(pdf.pages, checks, title_bottom, english)
        result["gutter_crossing_word_candidates"] = gutter
        if gutter:
            manual.append("Inspect gutter-crossing words: full-width titles/tables/figures may be legitimate; do not treat candidates as automatic violations.")
    after = path.read_bytes()
    check(checks, "input_unchanged_during_inspection", after == before)
    result["failed_checks"] = [name for name, value in checks.items() if not value["pass"]]
    result["status"] = "fail" if result["failed_checks"] else "machine_checks_pass_manual_review_required"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--chinese", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "tmp/pdfs/ser-20260906/qa.json")
    parser.add_argument("--expected-author", default="Tian Xie")
    parser.add_argument("--expected-email", default="tianjack.xie@mail.utoronto.ca")
    parser.add_argument("--expected-affiliation", default="University of Toronto")
    args = parser.parse_args(argv)
    inputs = [p for p in (args.english, args.chinese) if p]
    if (args.output.suffix.casefold() != ".json" or
            args.output.resolve() in [p.resolve() for p in inputs] or
            (args.output.exists() and any(p.exists() and args.output.samefile(p) for p in inputs))):
        parser.error("The report must not overwrite an input PDF")
    report = {"schema": "ser-paper-pdf-qa-1", "created_at": datetime.now(timezone.utc).isoformat(),
              "visual_review_performed_by_this_script": False, "pdfs": []}
    for path, english in ((args.english, True), (args.chinese, False)):
        if path is None:
            continue
        try:
            report["pdfs"].append(inspect(path, english=english, args=args))
        except Exception as exc:
            report["pdfs"].append({"path": str(path.absolute()), "status": "error",
                                   "error": f"{type(exc).__name__}: {exc}"})
    report["machine_checks_pass"] = all(p["status"] == "machine_checks_pass_manual_review_required" for p in report["pdfs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.absolute()), "machine_checks_pass": report["machine_checks_pass"],
                      "pdfs": [{k: p.get(k) for k in ("path", "status", "failed_checks", "error")} for p in report["pdfs"]]}, ensure_ascii=False))
    raise SystemExit(0 if report["machine_checks_pass"] else 1)


if __name__ == "__main__":
    main()

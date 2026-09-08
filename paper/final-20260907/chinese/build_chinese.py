"""Build the readable Chinese companion from its editable Markdown source.

Requires reportlab and the Windows SimSun/SimHei fonts; local PDF figures
also require pypdf. Font files are used locally and are not redistributed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import textsplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
import reportlab.platypus.paragraph as paragraph_module


INK = colors.HexColor("#172C3C")
ACCENT = colors.HexColor("#126F79")
MUTED = colors.HexColor("#576974")
RULE = colors.HexColor("#CEDADF")
PALE = colors.HexColor("#EFF5F6")
PAGE_W, PAGE_H = A4
MARGIN = 54
TEXT_W = PAGE_W - MARGIN * 2
FIGURE_LINE = re.compile(r"!\[([^\]]+)\]\(([^)]+)\)")

# ReportLab's built-in kinsoku list is Japanese-oriented. Include Chinese
# closing punctuation in this process only; do not modify installed libraries.
textsplit.ALL_CANNOT_START += "，；：！？）】》〉”’"
paragraph_module.ALL_CANNOT_START = textsplit.ALL_CANNOT_START


def normalize(text: str) -> str:
    # Retain mathematical meaning without relying on uncommon superscript glyphs.
    for old, new in {
        "2²⁴": "2^24", "10⁻⁶": "10^-6", "−": "-", "—": "-",
        "–": "-", "‑": "-", "≤": "<=",
    }.items():
        text = text.replace(old, new)
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^" + m[0].translate(superscripts), text)
    return text


def inline(text: str) -> str:
    text = html.escape(normalize(text), quote=False)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda m: f'<link href="{html.escape(m[2], quote=True)}" color="#126F79">{m[1]}</link>',
        text,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r'<font name="CN-Bold">\1</font>', text)
    text = re.sub(r"`([^`]+)`", r'<font color="#126F79">\1</font>', text)
    return text


def styles() -> dict[str, ParagraphStyle]:
    common = dict(fontName="CN", textColor=INK, wordWrap="CJK", splitLongWords=1,
                  allowWidows=0, allowOrphans=0)
    result = {
        "body": ParagraphStyle("Body", fontSize=11.5, leading=18.4, spaceAfter=9, **common),
        "meta": ParagraphStyle("Meta", fontSize=10, leading=16, spaceAfter=13,
                               **{**common, "textColor": MUTED}),
        "intro": ParagraphStyle("Intro", fontSize=10.5, leading=17, spaceAfter=16,
                                borderColor=RULE, borderWidth=0.5, borderPadding=10,
                                backColor=PALE, **common),
        "title": ParagraphStyle("Title", fontName="CN-Bold", fontSize=21, leading=30,
                                 textColor=INK, wordWrap="CJK", spaceAfter=16,
                                 keepWithNext=True),
        "h1": ParagraphStyle("Heading1", fontName="CN-Bold", fontSize=16, leading=24,
                              textColor=ACCENT, wordWrap="CJK", spaceBefore=18,
                              spaceAfter=10, keepWithNext=True),
        "h2": ParagraphStyle("Heading2", fontName="CN-Bold", fontSize=12.5, leading=20,
                              textColor=INK, wordWrap="CJK", spaceBefore=12,
                              spaceAfter=8, keepWithNext=True),
        "bullet": ParagraphStyle("Bullet", fontSize=11.5, leading=18.4,
                                  leftIndent=11, firstLineIndent=-8, spaceAfter=5, **common),
        "reference-intro": ParagraphStyle("ReferenceIntro", fontSize=11.5, leading=18.4,
                                          spaceAfter=9, keepWithNext=True, **common),
        "formula": ParagraphStyle("Formula", fontName="CN-Bold", fontSize=12,
                                   leading=22, textColor=ACCENT, alignment=TA_CENTER,
                                   spaceBefore=4, spaceAfter=12),
        "cell": ParagraphStyle("Cell", fontSize=9.7, leading=14.7, **common),
        "cell-center": ParagraphStyle("CellCenter", fontSize=9.7, leading=14.7,
                                       alignment=TA_CENTER, **common),
        "cell-head": ParagraphStyle("CellHead", fontName="CN-Bold", fontSize=9.5,
                                     leading=14.5, textColor=INK, wordWrap="CJK"),
        "caption": ParagraphStyle("Caption", fontSize=9.5, leading=14.5,
                                   spaceAfter=12, **common),
    }
    return result


class CompanionDoc(BaseDocTemplate):
    def __init__(self, destination, *, title="英文论文中文对应稿"):
        target = str(destination) if isinstance(destination, (str, Path)) else destination
        super().__init__(target, pagesize=A4, leftMargin=MARGIN,
                         rightMargin=MARGIN, topMargin=55, bottomMargin=51,
                         title=title,
                         author="Tian Xie", pageCompression=1,
                         initialFontName="CN", initialFontSize=11.5, lang="zh-CN")
        frame = Frame(MARGIN, 51, TEXT_W, PAGE_H - 106,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="reading", frames=frame, onPage=self.decorate))
        self._bookmark_count = 0

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(MUTED)
        canvas.setFont("CN", 8)
        canvas.drawString(MARGIN, PAGE_H - 32, "SER  |  英文论文中文解读")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 32, "Tian Xie  ·  2026-09-07")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - 40, PAGE_W - MARGIN, PAGE_H - 40)
        canvas.line(MARGIN, 38, PAGE_W - MARGIN, 38)
        canvas.drawString(MARGIN, 24, "解释版：与英文稿使用同一组实验和结论")
        canvas.drawRightString(PAGE_W - MARGIN, 24, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("Heading1", "Heading2"):
            self._bookmark_count += 1
            key = f"section-{self._bookmark_count}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key,
                                      level=0 if flowable.style.name == "Heading1" else 1,
                                      closed=False)


def markdown_table(lines: list[str], st: dict) -> Table:
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")]
             for line in lines]
    cells = [row for row in cells if not all(re.fullmatch(r"[:\- ]+", c) for c in row)]
    n = len(cells[0])
    if n == 5:
        widths = [91, *([(TEXT_W - 91) / 4] * 4)]
    elif n == 6 and cells[0][1] == "对比" and "CI" in cells[0][3]:
        # Keep signed confidence-interval endpoints on the same line.
        widths = [82, 52, 70, 130, (TEXT_W - 334) / 2, (TEXT_W - 334) / 2]
    elif n == 4:
        widths = [99, 88, 88, TEXT_W - 275]
    elif n == 3 and cells[0][0] == "记号":
        widths = [57, 194, TEXT_W - 251]
    elif n == 3:
        widths = [237, 164, TEXT_W - 401]
    elif n == 2:
        widths = [TEXT_W * 0.49, TEXT_W * 0.51]
    else:
        widths = [TEXT_W / n] * n
    data = []
    for rownum, row in enumerate(cells):
        parsed = []
        for colnum, content in enumerate(row):
            value = inline(content)
            if n == 5 and rownum:
                if colnum:
                    value = value.replace(" [", "<br/>[")
                else:
                    value = value.replace("／", "<br/>")
            if n == 5 and not rownum and colnum:
                value = value.replace(" ", "<br/>", 1)
            pstyle = st["cell-head"] if not rownum else st["cell-center"] if n >= 4 and colnum else st["cell"]
            parsed.append(Paragraph(value, pstyle))
        data.append(parsed)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=0,
                  spaceBefore=3, spaceAfter=13)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "CN"),
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("LINEABOVE", (0, 0), (-1, 0), 1, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFB")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def require(ok, message):
    if not ok:
        raise ValueError(message)


def byte_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def figure_assets(lines, source, manifest_path, pins):
    """Read each PDF once, then retain that exact verified buffer for merging."""
    image_lines = [line.strip() for line in lines if line.strip().startswith("![")]
    require(all(FIGURE_LINE.fullmatch(line) for line in image_lines),
            "figures must be standalone ![caption](relative.pdf) lines")
    if not image_lines:
        return {}
    require(manifest_path is not None, "PDF figures require --figure-manifest")
    from pypdf import PdfReader

    manifest_path = manifest_path.resolve()
    raw = manifest_path.read_bytes()
    pins[str(manifest_path)] = byte_sha(raw)
    manifest = json.loads(raw)
    require(isinstance(manifest.get("files"), dict), "figure manifest has no files mapping")
    assets = {}
    for line in image_lines:
        relative = FIGURE_LINE.fullmatch(line)[2]
        url = urlsplit(relative)
        require(not any((url.scheme, url.netloc, url.query, url.fragment))
                and not Path(relative).is_absolute() and "\\" not in relative,
                "figure must be a relative local PDF, not a URL")
        path = (source.parent / relative).resolve()
        require(path.suffix.lower() == ".pdf" and path.parent == manifest_path.parent
                and path == (manifest_path.parent / path.name).resolve(),
                "figure must be directly in the manifest directory")
        if relative in assets:
            continue
        expected = manifest["files"].get(path.name)
        require(isinstance(expected, dict) and re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", "")),
                "figure basename has no SHA-256 commitment")
        # Different relative spellings of one PDF reuse the same byte buffer.
        reused = next((asset for asset in assets.values() if asset["path"] == path), None)
        if reused is not None:
            assets[relative] = reused
            continue
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "figure changed while reading")
        sha = byte_sha(data)
        require(sha == expected["sha256"] and expected.get("bytes", len(data)) == len(data),
                "figure bytes differ from manifest")
        reader = PdfReader(BytesIO(data), strict=True)
        require(not reader.is_encrypted and len(reader.pages) == 1, "figure must be a single unencrypted PDF page")
        page = reader.pages[0]
        require(page.rotation == 0 and float(page.get("/UserUnit", 1)) == 1,
                "rotated or nonstandard-unit PDF figures are unsupported")
        box = tuple(float(x) for x in page.cropbox)
        media = tuple(float(x) for x in page.mediabox)
        require(all(math.isfinite(x) for x in (*box, *media))
                and media[0] <= box[0] < box[2] <= media[2]
                and media[1] <= box[1] < box[3] <= media[3], "invalid figure page boxes")
        assets[relative] = dict(path=path, data=data, reader=reader, page=page,
                                sha256=sha, box=box, stat=(after.st_size, after.st_mtime_ns))
    return assets


class PDFSlot(Flowable):
    """Reserve real layout space; vector content is merged after pagination."""
    def __init__(self, asset, caption, placements):
        super().__init__()
        self.asset, self.caption, self.placements = asset, caption, placements
        x0, y0, x1, y1 = asset["box"]
        self.width = TEXT_W
        self.scale = TEXT_W / (x1 - x0)
        self.height = (y1 - y0) * self.scale
        self.hAlign = "LEFT"

    def wrap(self, avail_width, avail_height):
        require(self.width <= avail_width + 1e-6, "PDF figure exceeds text frame width")
        return self.width, self.height

    def draw(self):
        x, y = self.canv.absolutePosition(0, 0)
        require(MARGIN - 1e-6 <= x and x + self.width <= PAGE_W - MARGIN + 1e-6
                and 51 - 1e-6 <= y and y + self.height <= PAGE_H - 55 + 1e-6,
                "PDF figure placement escapes the text frame")
        x0, y0, _, _ = self.asset["box"]
        self.placements.append(dict(asset=self.asset, caption=self.caption,
            page=self.canv.getPageNumber(), x=x, y=y, width=self.width,
            height=self.height, scale=self.scale,
            matrix=[self.scale, 0, 0, self.scale, x-self.scale*x0, y-self.scale*y0]))


def build(source: Path, destination: Path, font_dir: Path,
          figure_manifest: Path | None = None) -> None:
    source, destination, font_dir = source.resolve(), destination.resolve(), font_dir.resolve()
    source_bytes = source.read_bytes()
    pins = {str(source): byte_sha(source_bytes),
            str(Path(__file__).resolve()): byte_sha(Path(__file__).read_bytes())}
    for name in ("simsun.ttc", "simhei.ttf"):
        path = font_dir / name
        pins[str(path)] = byte_sha(path.read_bytes())
    lines = source_bytes.decode("utf-8-sig").splitlines()
    assets = figure_assets(lines, source, figure_manifest, pins)
    protected = {Path(path) for path in pins} | {asset["path"] for asset in assets.values()}
    receipt_path = destination.with_suffix(destination.suffix + ".build.json")
    require(destination not in protected and receipt_path not in protected,
            "output or receipt would overwrite a build input")
    pdfmetrics.registerFont(TTFont("CN", str(font_dir / "simsun.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(font_dir / "simhei.ttf")))
    pdfmetrics.registerFontFamily("CN", normal="CN", bold="CN-Bold", italic="CN", boldItalic="CN-Bold")
    st = styles()
    heading_title = next((line.strip()[2:] for line in lines if line.strip().startswith("# ")),
                         "英文论文中文对应稿")
    document_title = Paragraph(inline(heading_title), st["title"]).getPlainText()
    story = []
    placements = []
    i = 0
    in_references = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            story.append(markdown_table(block, st))
            continue
        if line.startswith("!["):
            matched = FIGURE_LINE.fullmatch(line)
            caption, relative = matched[1], matched[2]
            figure = PDFSlot(assets[relative], caption, placements)
            caption_flow = Paragraph(inline(caption), st["caption"])
            _, caption_height = caption_flow.wrap(TEXT_W, PAGE_H)
            require(figure.height + 6 + caption_height + st["caption"].spaceAfter <= PAGE_H - 106,
                    "figure and caption cannot fit together on one page")
            story.append(KeepTogether([figure, Spacer(1, 6), caption_flow]))
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), st["title"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), st["h2"]))
        elif line.startswith("## "):
            in_references = line[3:] == "参考文献"
            if in_references:
                story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]), st["h1"]))
        elif line.startswith("- "):
            entry = Paragraph("· " + inline(line[2:]), st["bullet"])
            story.append(KeepTogether([entry]) if in_references else entry)
        elif line.startswith("`") and line.endswith("`"):
            story.append(Paragraph(inline(line), st["formula"]))
        else:
            block = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith(("#", "|", "- ", "![")):
                i += 1
                block.append(lines[i].strip())
            if line.startswith("作者："):
                story.append(Paragraph("<br/>".join(inline(x) for x in block), st["meta"]))
            else:
                pstyle = (st["reference-intro"] if in_references else
                          st["intro"] if line.startswith("本文是对应英文论文") else st["body"])
                story.append(Paragraph(inline("".join(block)), pstyle))
        i += 1
    base = BytesIO()
    CompanionDoc(base, title=document_title).build(story)
    pdf_bytes = base.getvalue()
    if placements:
        from pypdf import PdfReader, PdfWriter
        writer = PdfWriter(clone_from=PdfReader(BytesIO(pdf_bytes)))
        for placed in placements:
            writer.pages[placed["page"] - 1].merge_transformed_page(
                placed["asset"]["page"], tuple(placed["matrix"]), over=True, expand=False)
        merged = BytesIO()
        writer.write(merged)
        pdf_bytes = merged.getvalue()
    for path, expected in pins.items():
        require(byte_sha(Path(path).read_bytes()) == expected, "build input changed: " + path)
    for asset in assets.values():
        current = asset["path"].stat()
        require((current.st_size, current.st_mtime_ns) == asset["stat"], "figure changed during build")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pdf_bytes)
    receipt = dict(schema="ser-chinese-companion-build-1",
        created_at=datetime.now(timezone.utc).isoformat(), inputs_sha256=pins,
        document_metadata=dict(title=document_title, author="Tian Xie", language="zh-CN"),
        figure_manifest=str(figure_manifest.resolve()) if figure_manifest and assets else None,
        figures=[dict(path=str(p["asset"]["path"]), sha256=p["asset"]["sha256"],
            bytes=len(p["asset"]["data"]), source_cropbox=list(p["asset"]["box"]),
            **{k:v for k,v in p.items() if k != "asset"}) for p in placements],
        output=dict(path=str(destination), bytes=len(pdf_bytes), sha256=byte_sha(pdf_bytes)),
        figure_embedding="vector PDF content merge; no rasterization",
        scope="Document layout only; no model inference or new scientific analysis.",
        visual_review_performed=False)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("中文解读.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--figure-manifest", type=Path,
                        help="SHA manifest beside the single-page PDF figures; required only when figures occur")
    args = parser.parse_args()
    build(args.source, args.output, args.font_dir, args.figure_manifest)

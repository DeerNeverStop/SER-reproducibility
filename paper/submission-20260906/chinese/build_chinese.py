"""Build the readable Chinese companion from its editable Markdown source.

Requires reportlab and the Windows SimSun/SimHei fonts. The font files are
used locally and are not redistributed with the source package.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import textsplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
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
        "formula": ParagraphStyle("Formula", fontName="CN-Bold", fontSize=12,
                                   leading=22, textColor=ACCENT, alignment=TA_CENTER,
                                   spaceBefore=4, spaceAfter=12),
        "cell": ParagraphStyle("Cell", fontSize=9.7, leading=14.7, **common),
        "cell-center": ParagraphStyle("CellCenter", fontSize=9.7, leading=14.7,
                                       alignment=TA_CENTER, **common),
        "cell-head": ParagraphStyle("CellHead", fontName="CN-Bold", fontSize=9.5,
                                     leading=14.5, textColor=INK, wordWrap="CJK"),
    }
    return result


class CompanionDoc(BaseDocTemplate):
    def __init__(self, destination: Path):
        super().__init__(str(destination), pagesize=A4, leftMargin=MARGIN,
                         rightMargin=MARGIN, topMargin=55, bottomMargin=51,
                         title="语音情感识别中的说话人重叠：英文论文中文解读",
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
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 32, "Tian Xie  ·  2026-09-06")
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
    elif n == 4:
        widths = [99, 88, 88, TEXT_W - 275]
    elif n == 3 and cells[0][0] == "记号":
        widths = [57, 194, TEXT_W - 251]
    elif n == 3:
        widths = [237, 164, TEXT_W - 401]
    else:
        widths = [TEXT_W * 0.49, TEXT_W * 0.51]
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


def build(source: Path, destination: Path, font_dir: Path) -> None:
    pdfmetrics.registerFont(TTFont("CN", str(font_dir / "simsun.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(font_dir / "simhei.ttf")))
    pdfmetrics.registerFontFamily("CN", normal="CN", bold="CN-Bold", italic="CN", boldItalic="CN-Bold")
    st = styles()
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    story = []
    i = 0
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
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), st["title"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), st["h2"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), st["h1"]))
        elif line.startswith("- "):
            story.append(Paragraph("· " + inline(line[2:]), st["bullet"]))
        elif line.startswith("`") and line.endswith("`"):
            story.append(Paragraph(inline(line), st["formula"]))
        else:
            block = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith(("#", "|", "- ")):
                i += 1
                block.append(lines[i].strip())
            if line.startswith("作者："):
                story.append(Paragraph("<br/>".join(inline(x) for x in block), st["meta"]))
            else:
                pstyle = st["intro"] if line.startswith("本文是对应英文论文") else st["body"]
                story.append(Paragraph(inline("".join(block)), pstyle))
        i += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    CompanionDoc(destination).build(story)
    print(destination.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("中文解读.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    build(args.source, args.output, args.font_dir)

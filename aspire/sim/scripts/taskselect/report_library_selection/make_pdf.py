#!/usr/bin/env python3
"""Render a Markdown report (headings, paragraphs, lists, pipe tables, images, inline code) to PDF with reportlab.

  /home/jm/miniconda3/envs/diffrag/bin/python scripts/taskselect/report_library_selection/make_pdf.py [REPORT.md] [out.pdf]

Needs markdown-it-py and reportlab, plus the Liberation fonts (all four sans styles and a mono face).
Images are resolved relative to the Markdown file.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

FONTS = Path("/usr/share/fonts/truetype/liberation")
for name, file in [("Body", "LiberationSans-Regular.ttf"), ("Body-Bold", "LiberationSans-Bold.ttf"),
                   ("Body-Italic", "LiberationSans-Italic.ttf"), ("Body-BoldItalic", "LiberationSans-BoldItalic.ttf"),
                   ("Mono", "LiberationMono-Regular.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")

INK, MUTED, RULE, HEAD_BG = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d3"), colors.HexColor("#eceae4")
BASE = ParagraphStyle("base", fontName="Body", fontSize=9, leading=12.6, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
H = {1: ParagraphStyle("h1", parent=BASE, fontName="Body-Bold", fontSize=16, leading=20, spaceBefore=2, spaceAfter=8, keepWithNext=1),
     2: ParagraphStyle("h2", parent=BASE, fontName="Body-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=5, keepWithNext=1),
     3: ParagraphStyle("h3", parent=BASE, fontName="Body-Bold", fontSize=10, leading=13, spaceBefore=7, spaceAfter=3, keepWithNext=1)}
CELL = ParagraphStyle("cell", parent=BASE, fontSize=7.6, leading=9.8, spaceAfter=0)
CELL_HEAD = ParagraphStyle("cellhead", parent=CELL, fontName="Body-Bold")


def inline(token) -> str:
    """Inline children to reportlab paragraph markup."""
    out = []
    for c in token.children or []:
        t = c.type
        if t == "text":
            out.append(html.escape(c.content))
        elif t == "code_inline":
            out.append(f'<font face="Mono" size="8" color="#7a3b00">{html.escape(c.content)}</font>')
        elif t in ("strong_open", "strong_close"):
            out.append("<b>" if t.endswith("open") else "</b>")
        elif t in ("em_open", "em_close"):
            out.append("<i>" if t.endswith("open") else "</i>")
        elif t in ("softbreak", "hardbreak"):
            out.append(" " if t == "softbreak" else "<br/>")
        elif t in ("link_open", "link_close"):
            out.append("<u>" if t.endswith("open") else "</u>")
        elif t == "image":
            out.append(html.escape(c.content))
    return "".join(out)


class Builder:
    def __init__(self, md_dir: Path, width: float):
        self.dir, self.width = md_dir, width

    def walk(self, tokens, i=0, stop=None):
        """Return (flowables, next_index) until a closing token named `stop`."""
        flow = []
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == stop:
                return flow, i + 1
            t = tok.type
            if t == "heading_open":
                level = min(int(tok.tag[1]), 3)
                flow.append(Paragraph(inline(tokens[i + 1]), H[level]))
                i += 3
            elif t == "paragraph_open":
                inl = tokens[i + 1]
                imgs = [c for c in (inl.children or []) if c.type == "image"]
                if imgs and len(inl.children) == 1:
                    flow.append(self.image(imgs[0]))
                else:
                    flow.append(Paragraph(inline(inl), BASE))
                i += 3
            elif t in ("bullet_list_open", "ordered_list_open"):
                items, i = self.list_items(tokens, i + 1, t.replace("open", "close"))
                kind = "bullet" if t.startswith("bullet") else "1"
                flow.append(ListFlowable(items, bulletType=kind, bulletFontName="Body", bulletFontSize=8.5, leftIndent=14,
                                         bulletDedent=12, spaceAfter=5, start=1 if kind == "1" else None))
            elif t == "table_open":
                tbl, i = self.table(tokens, i + 1)
                flow.append(tbl)
                flow.append(Spacer(1, 6))
            elif t == "fence" or t == "code_block":
                from reportlab.platypus import Preformatted
                flow.append(Preformatted(tok.content.rstrip("\n"), ParagraphStyle("pre", parent=BASE, fontName="Mono", fontSize=7.5,
                                                                                 leading=10, backColor=HEAD_BG, borderPadding=4)))
                i += 1
            elif t == "hr":
                flow.append(Spacer(1, 6)); i += 1
            else:
                i += 1
        return flow, i

    def list_items(self, tokens, i, stop):
        items = []
        while i < len(tokens) and tokens[i].type != stop:
            if tokens[i].type == "list_item_open":
                inner, i = self.walk(tokens, i + 1, "list_item_close")
                for f in inner:
                    if isinstance(f, Paragraph):
                        f.style = ParagraphStyle("li", parent=BASE, spaceAfter=2)
                items.append(ListItem(inner))
            else:
                i += 1
        return items, i + 1

    def table(self, tokens, i):
        rows, head_rows = [], 0
        while i < len(tokens) and tokens[i].type != "table_close":
            if tokens[i].type == "tr_open":
                cells, is_head = [], False
                i += 1
                while tokens[i].type != "tr_close":
                    if tokens[i].type in ("th_open", "td_open"):
                        is_head = is_head or tokens[i].type == "th_open"
                        cells.append(Paragraph(inline(tokens[i + 1]), CELL_HEAD if tokens[i].type == "th_open" else CELL))
                        i += 3
                    else:
                        i += 1
                head_rows += is_head and not rows or 0
                rows.append(cells)
            i += 1
        ncol = max(len(r) for r in rows)
        rows = [r + [Paragraph("", CELL)] * (ncol - len(r)) for r in rows]
        # column widths: proportional to the plain text length (clamped), never narrower than the longest word
        def words(c):
            return [w for r in rows for w in r[c].getPlainText().split()]
        longest = [max([len(w) for w in words(c)] + [1]) for c in range(ncol)]
        length = [max(len(r[c].getPlainText()) for r in rows) for c in range(ncol)]
        weight = [max(longest[c] + 2, min(length[c], 45)) for c in range(ncol)]
        floor = [longest[c] * 4.1 + 8 for c in range(ncol)]
        widths = [self.width * w / sum(weight) for w in weight]
        for _ in range(ncol):                       # lift narrow columns to their floor, take the space from the wide ones
            low = [c for c in range(ncol) if widths[c] < floor[c]]
            if not low:
                break
            need = sum(floor[c] - widths[c] for c in low)
            rest = [c for c in range(ncol) if c not in low]
            spare = sum(widths[c] for c in rest)
            for c in low:
                widths[c] = floor[c]
            for c in rest:
                widths[c] -= need * widths[c] / spare
        tbl = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                                 ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                                 ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        return tbl, i + 1

    def image(self, tok):
        path = (self.dir / tok.attrs["src"]).resolve()
        from reportlab.lib.utils import ImageReader
        w, h = ImageReader(str(path)).getSize()
        scale = min(self.width / w, 1.0) * 1.0
        img = Image(str(path), width=w * scale * self.width / (w * scale), height=h * self.width / w)
        img.hAlign = "LEFT"
        return KeepTogether([img, Spacer(1, 6)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 10 * mm, "Library build with task selection - first comparison and noise floor")
    canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("REPORT.md")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    md = MarkdownIt("commonmark", {"breaks": False}).enable("table")
    tokens = md.parse(src.read_text())
    doc = SimpleDocTemplate(str(dst), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=15 * mm,
                            bottomMargin=17 * mm, title=src.stem, author="ASPIRE")
    width = A4[0] - doc.leftMargin - doc.rightMargin
    flow, _ = Builder(src.parent, width).walk(tokens)
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

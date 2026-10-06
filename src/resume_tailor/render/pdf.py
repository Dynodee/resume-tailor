"""Node 6 -- render an ATS-safe PDF.

What "ATS-safe" means in practice, and why this file looks plain:

- One column, one text flow. Multi-column layouts and text boxes get read in
  the wrong order, which scrambles your dates into your skills.
- No tables. Right-aligned dates are usually done with a table or a tab stop;
  both are parsed unreliably, so dates go on their own line instead.
- No headers, footers, or images. Anything in the margin region is commonly
  dropped, which is how people lose their phone number.
- A standard font with real text, never outlines. The parser reads the text
  layer; if the glyphs are vectorised there is nothing to read.
- A real bullet glyph followed by a space, with the indent done by the layout
  rather than by whitespace characters.

The companion .txt file written next to the PDF is the extraction check: if it
reads correctly, a parser will see it correctly.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
)

from ..state import TailoredResume

INK = HexColor("#111111")
RULE = HexColor("#999999")


class Rule(Flowable):
    """A hairline under a section heading. Vector, so no text is displaced."""

    def __init__(self, width: float, thickness: float = 0.6) -> None:
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.height = thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(RULE)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


def _styles(scale: float) -> dict[str, ParagraphStyle]:
    def size(base: float) -> float:
        return round(base * scale, 2)

    body = size(9.6)
    return {
        "name": ParagraphStyle(
            "name", fontName="Helvetica-Bold", fontSize=size(17), leading=size(20),
            alignment=TA_CENTER, textColor=INK, spaceAfter=size(2)),
        "headline": ParagraphStyle(
            "headline", fontName="Helvetica", fontSize=size(9), leading=size(11.5),
            alignment=TA_CENTER, textColor=INK, spaceAfter=size(1.5)),
        "contact": ParagraphStyle(
            "contact", fontName="Helvetica", fontSize=size(9), leading=size(11.5),
            alignment=TA_CENTER, textColor=INK, spaceAfter=size(9)),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=size(10.5), leading=size(12.5),
            textColor=INK, spaceBefore=size(7), spaceAfter=size(2.5)),
        "entry": ParagraphStyle(
            "entry", fontName="Helvetica-Bold", fontSize=body + 0.4, leading=size(12),
            textColor=INK, spaceBefore=size(4), spaceAfter=0),
        "meta": ParagraphStyle(
            "meta", fontName="Helvetica-Oblique", fontSize=body - 0.6, leading=size(11),
            textColor=INK, spaceAfter=size(2)),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=body, leading=size(12.2),
            textColor=INK, leftIndent=size(11), firstLineIndent=size(-11),
            spaceAfter=size(1.6), alignment=TA_LEFT),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=body, leading=size(12.2),
            textColor=INK, spaceAfter=size(2), alignment=TA_LEFT),
    }


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _rich(text: str, spans: list | None) -> str:
    """Escape `text` for a Paragraph, wrapping the given spans in <b>.

    Bold changes the font of a run, not the characters in it, so the text a
    parser extracts is identical with or without emphasis.
    """
    if not spans:
        return _escape(text)
    out, pos = [], 0
    for start, end in sorted(spans):
        if start < pos or end > len(text):
            continue
        out.append(_escape(text[pos:start]))
        out.append(f"<b>{_escape(text[start:end])}</b>")
        pos = end
    out.append(_escape(text[pos:]))
    return "".join(out)


def _flowables(resume: TailoredResume, contact: dict, styles, frame_width: float):
    story: list = []
    name = contact.get("name", "")
    story.append(Paragraph(_escape(name).upper(), styles["name"]))
    if resume.headline:
        story.append(Paragraph(_escape(resume.headline), styles["headline"]))
    bits = [contact.get(k, "") for k in ("phone", "email", "linkedin", "location")]
    story.append(Paragraph(" | ".join(_escape(b) for b in bits if b), styles["contact"]))

    def heading(text: str) -> list:
        return [Paragraph(_escape(text).upper(), styles["section"]), Rule(frame_width),
                Spacer(1, 3)]

    if resume.summary:
        story += heading("Professional Summary")
        story.append(Paragraph(_rich(resume.summary, resume.emphasis.get("summary")),
                               styles["body"]))

    for si, section in enumerate(resume.sections):
        story += heading(section.heading)
        for ei, entry in enumerate(section.entries):
            block: list = []
            if section.kind == "experience":
                title = entry.get("title", "")
                company = entry.get("company", "")
                block.append(Paragraph(
                    f"{_escape(title)}" + (f", {_escape(company)}" if company else ""),
                    styles["entry"]))
                meta = " | ".join(x for x in (
                    f"{entry.get('start','')} - {entry.get('end','')}".strip(" -"),
                    entry.get("location", ""),
                ) if x)
                if meta:
                    block.append(Paragraph(_escape(meta), styles["meta"]))
            elif section.kind == "projects":
                head = entry.get("name", "")
                sub = entry.get("subtitle", "")
                block.append(Paragraph(
                    f"{_escape(head)}" + (f", {_escape(sub)}" if sub else ""),
                    styles["entry"]))
                if entry.get("year"):
                    block.append(Paragraph(_escape(entry["year"]), styles["meta"]))
            elif section.kind == "education":
                block.append(Paragraph(_escape(entry.get("degree", "")), styles["entry"]))
                meta = " | ".join(x for x in (entry.get("school", ""),
                                              entry.get("dates", "")) if x)
                if meta:
                    block.append(Paragraph(_escape(meta), styles["meta"]))
            elif section.kind == "skills":
                label = entry.get("label", "")
                items = ", ".join(str(i) for i in entry.get("items", []))
                block.append(Paragraph(
                    f"<b>{_escape(label)}:</b> {_escape(items)}", styles["body"]))

            for bi, bullet in enumerate(entry.get("bullets", [])):
                text = bullet["text"] if isinstance(bullet, dict) else str(bullet)
                spans = resume.emphasis.get(f"s{si}e{ei}b{bi}")
                block.append(Paragraph(f"• {_rich(text, spans)}", styles["bullet"]))

            # Keep a heading with at least its first bullet, so a page break
            # never orphans a job title from its content.
            story.append(KeepTogether(block[:2]) if len(block) > 1 else block[0])
            story.extend(block[2:])

    return story


def _build(resume, contact, path: Path, scale: float, margin: float) -> int:
    doc = BaseDocTemplate(
        str(path), pagesize=LETTER,
        leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin,
        title=f"{contact.get('name','Resume')} - Resume",
        author=contact.get("name", ""), subject="Resume",
        creator="resume-tailor",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                  id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    styles = _styles(scale)
    doc.build(_flowables(resume, contact, styles, doc.width))
    return doc.page


def render(resume: TailoredResume, master: dict, out_path: Path,
           max_pages: int = 2) -> tuple[Path, int]:
    """Render, shrinking slightly until it fits `max_pages`.

    Auto-fit rather than truncation: dropping content to fit a page is a
    decision the tailor node should have made deliberately, not something the
    renderer does behind its back.
    """
    contact = master.get("contact", {})
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages = 0
    for scale, margin in ((1.0, 0.62 * inch), (0.96, 0.58 * inch),
                          (0.92, 0.54 * inch), (0.88, 0.5 * inch),
                          (0.85, 0.45 * inch)):
        pages = _build(resume, contact, out_path, scale, margin)
        if pages <= max_pages:
            break

    _write_extraction_check(resume, contact, out_path.with_suffix(".txt"))
    return out_path, pages


def _write_extraction_check(resume: TailoredResume, contact: dict, path: Path) -> None:
    """Plain-text twin of the PDF: what a parser should come away with."""
    lines = [contact.get("name", "")]
    if resume.headline:
        lines.append(resume.headline)
    lines.append(" | ".join(v for k, v in contact.items() if k != "name" and v))
    if resume.summary:
        lines += ["", "PROFESSIONAL SUMMARY", resume.summary]
    for section in resume.sections:
        lines += ["", section.heading]
        for entry in section.entries:
            if section.kind == "experience":
                lines.append(f"{entry.get('title','')}, {entry.get('company','')}")
                lines.append(f"{entry.get('start','')} - {entry.get('end','')} | "
                             f"{entry.get('location','')}")
            elif section.kind == "projects":
                lines.append(f"{entry.get('name','')}, {entry.get('subtitle','')}")
                if entry.get("year"):
                    lines.append(str(entry["year"]))
            elif section.kind == "education":
                lines.append(entry.get("degree", ""))
                meta = " | ".join(x for x in (entry.get("school", ""),
                                              entry.get("dates", "")) if x)
                if meta:
                    lines.append(meta)
            elif section.kind == "skills":
                lines.append(f"{entry.get('label','')}: "
                             f"{', '.join(str(i) for i in entry.get('items', []))}")
            for bullet in entry.get("bullets", []):
                text = bullet["text"] if isinstance(bullet, dict) else str(bullet)
                lines.append(f"- {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

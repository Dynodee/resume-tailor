"""Render the cover letter: a plain one-page PDF plus a text copy.

Same header as the resume (name, then contact line), so the two read as a set.
Many application forms want the letter pasted into a box, which is what the
.txt copy is for.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from ..state import CoverLetter
from .pdf import INK, _escape


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "name": ParagraphStyle("name", fontName="Helvetica-Bold", fontSize=16, leading=19,
                               alignment=TA_CENTER, textColor=INK, spaceAfter=2),
        "contact": ParagraphStyle("contact", fontName="Helvetica", fontSize=9, leading=11.5,
                                  alignment=TA_CENTER, textColor=INK, spaceAfter=18),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=14.5,
                               textColor=INK, spaceAfter=10),
    }


def _contact_line(contact: dict) -> str:
    return " | ".join(str(contact.get(k, "")) for k in ("phone", "email", "linkedin", "location")
                      if contact.get(k))


def render(letter: CoverLetter, master: dict, out_path: Path, name: str,
           date: dt.date | None = None) -> Path:
    contact = master.get("contact", {}) or {}
    today = (date or dt.date.today()).strftime("%B %d, %Y").replace(" 0", " ")
    styles = _styles()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    story = [Paragraph(_escape(name or contact.get("name", "")).upper(), styles["name"]),
             Paragraph(_escape(_contact_line(contact)), styles["contact"]),
             Paragraph(_escape(today), styles["body"])]
    if letter.greeting:
        story.append(Paragraph(_escape(letter.greeting), styles["body"]))
    story += [Paragraph(_escape(p.text.strip()), styles["body"])
              for p in letter.paragraphs if p.text.strip()]
    story.append(Spacer(1, 4))
    if letter.sign_off:
        story.append(Paragraph(_escape(letter.sign_off), styles["body"]))
    story.append(Paragraph(_escape(name), styles["body"]))

    doc = SimpleDocTemplate(str(out_path), pagesize=LETTER, leftMargin=0.9 * inch,
                            rightMargin=0.9 * inch, topMargin=0.8 * inch,
                            bottomMargin=0.8 * inch, title=f"{name} - Cover Letter",
                            author=name, subject="Cover Letter", creator="resume-tailor")
    doc.build(story)

    text = [name, _contact_line(contact), "", today, ""]
    if letter.greeting:
        text += [letter.greeting, ""]
    text += [p.text.strip() + "\n" for p in letter.paragraphs if p.text.strip()]
    text += [letter.sign_off, name] if letter.sign_off else [name]
    out_path.with_suffix(".txt").write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")
    return out_path

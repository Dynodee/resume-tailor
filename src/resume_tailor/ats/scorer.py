"""Keyword coverage scoring.

This is the number the revision loop steers on. Two scores are reported because
they answer different questions: the overall score is "how well does this read
to a keyword matcher", and the required score is "would a rules-based screen
reject this outright". The second one is the one that gets people filtered.
"""

from __future__ import annotations

from ..state import CoverageItem, CoverageReport, Keyword, Priority, TailoredResume
from .lexicon import _pattern, surface_forms


def is_soft(kw: Keyword) -> bool:
    return kw.category == "soft"


def _section_texts(resume: TailoredResume) -> dict[str, str]:
    """Flatten the resume into named blocks so we can say *where* a term hit.

    Placement matters to real screens: a keyword that only appears in the skills
    list is weaker evidence than the same keyword inside a dated job bullet.
    """
    out = {"headline": resume.headline, "summary": resume.summary}
    for section in resume.sections:
        chunks: list[str] = []
        for entry in section.entries:
            for key, value in entry.items():
                if key == "bullets":
                    for b in value:
                        chunks.append(b["text"] if isinstance(b, dict) else str(b))
                elif isinstance(value, str):
                    chunks.append(value)
                elif isinstance(value, list):
                    chunks.extend(str(v) for v in value)
        out[section.kind] = "\n".join(chunks)
    return out


def score(resume: TailoredResume, keywords: list[Keyword],
          master_text: str | None = None) -> CoverageReport:
    """Score keyword coverage.

    `master_text` is the whole fact base flattened to text. When given, required
    terms that appear nowhere in it are split out as unattainable, and the
    attainable score is computed over the rest.
    """
    blocks = _section_texts(resume)
    items: list[CoverageItem] = []

    for kw in keywords:
        forms = surface_forms(kw.term)
        if kw.aliases:
            forms = list(dict.fromkeys([*forms, *kw.aliases]))
        where: list[str] = []
        for block_name, block_text in blocks.items():
            if not block_text:
                continue
            if any(_pattern(f).search(block_text) for f in forms):
                where.append(block_name)
        items.append(CoverageItem(keyword=kw, covered=bool(where), where=where))

    def weight(item: CoverageItem) -> float:
        # Soft skills count for little: a screen cannot verify "communication",
        # and chasing the literal string is what produces bullets like
        # "Applied problem-solving to...". Show it, don't name it.
        return item.keyword.score * (0.25 if is_soft(item.keyword) else 1.0)

    def pct(subset: list[CoverageItem]) -> float:
        total = sum(weight(i) for i in subset)
        if total == 0:
            return 100.0
        got = sum(weight(i) for i in subset if i.covered)
        return round(100.0 * got / total, 1)

    # Required coverage is what drives the revision loop, so soft skills are
    # left out of it entirely -- otherwise the loop keeps asking the model to
    # name-drop them.
    required = [i for i in items
                if i.keyword.priority is Priority.REQUIRED and not is_soft(i.keyword)]
    attainable = required
    unattainable: list[str] = []
    if master_text is not None:
        attainable, unattainable = [], []
        for i in required:
            forms = list(dict.fromkeys([*surface_forms(i.keyword.term), *i.keyword.aliases]))
            if any(_pattern(f).search(master_text) for f in forms):
                attainable.append(i)
            else:
                unattainable.append(i.keyword.term)

    return CoverageReport(
        items=items,
        score=pct(items),
        required_score=pct(required) if required else pct(items),
        attainable_score=pct(attainable) if attainable else 100.0,
        unattainable=unattainable,
    )


def explain(report: CoverageReport, limit: int = 12) -> str:
    """Human-readable gap list, used in the CLI output and the match report."""
    lines = [
        f"Overall keyword coverage: {report.score}%",
        f"Required-term coverage:   {report.required_score}%",
    ]
    for priority, label in (
        (Priority.REQUIRED, "MISSING (required)"),
        (Priority.PREFERRED, "MISSING (preferred)"),
    ):
        gaps = [k for k in report.missing(priority) if not is_soft(k)][:limit]
        if gaps:
            lines.append("")
            lines.append(label + ":")
            lines.extend(f"  - {k.term}" + (f"  <- {k.evidence[:90]}" if k.evidence else "")
                         for k in gaps)
    return "\n".join(lines)

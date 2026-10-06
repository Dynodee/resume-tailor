"""Bolding the posting's terms on the page."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FIXTURES, ScriptedLLM, allstate_jd  # noqa: E402

from resume_tailor.config import Config  # noqa: E402
from resume_tailor.graph import run  # noqa: E402
from resume_tailor.render import emphasis  # noqa: E402
from resume_tailor.state import (  # noqa: E402
    Keyword,
    PipelineState,
    Priority,
    ResumeSection,
    TailoredResume,
)


def kw(term, priority=Priority.REQUIRED, category="skill"):
    from resume_tailor.ats import lexicon
    return Keyword(term=term, normalized=term.lower(), priority=priority, category=category,
                   aliases=lexicon.LEXICON.get(term, {}).get("aliases", []))


def resume_of(*bullets, summary="", end="Present"):
    return TailoredResume(summary=summary, sections=[
        ResumeSection(heading="WORK EXPERIENCE", kind="experience", entries=[
            {"title": "Analyst", "end": end,
             "bullets": [{"source_id": f"b{i}", "text": t} for i, t in enumerate(bullets)]}]),
        ResumeSection(heading="SKILLS", kind="skills",
                      entries=[{"label": "Data", "items": ["SQL", "Tableau"]}]),
    ])


def bolded(resume, plan) -> dict[str, list[str]]:
    texts = {k: t for k, _, t in emphasis._units(resume)}
    return {k: [texts[k][s:e] for s, e in spans] for k, spans in plan.items()}


def test_bolds_posting_terms_and_nothing_else():
    r = resume_of("Write SQL and Python to validate data quality.")
    got = bolded(r, emphasis.plan(r, [kw("SQL"), kw("Data Quality")]))
    assert got == {"s0e0b0": ["SQL", "data quality"]}       # Python: not in the posting


def test_at_most_two_per_bullet_and_three_in_summary():
    terms = [kw(t) for t in ("SQL", "Tableau", "Power BI", "Snowflake", "dbt")]
    r = resume_of("Used SQL, Tableau, Power BI, Snowflake, and dbt daily.",
                  summary="Analyst using SQL, Tableau, Power BI, Snowflake, and dbt.")
    plan = emphasis.plan(r, terms)
    assert len(plan["summary"]) <= 3
    assert sum(1 for _ in plan["s0e0b0"]) <= 2


def test_required_beats_mentioned_then_reading_order():
    r = resume_of("Built Snowflake models and SQL reports in Tableau.")
    terms = [kw("Snowflake", Priority.MENTIONED), kw("SQL"), kw("Tableau")]
    got = bolded(r, emphasis.plan(r, terms))["s0e0b0"]
    assert got == ["SQL", "Tableau"]


def test_once_per_role():
    r = resume_of("Write SQL to validate data.", "Write SQL to build reports.")
    got = bolded(r, emphasis.plan(r, [kw("SQL")]))
    assert got == {"s0e0b0": ["SQL"]}


def test_longest_phrase_wins_and_adjacent_spans_merge():
    r = resume_of("Built Tableau dashboards that 15+ regions relied on.")
    got = bolded(r, emphasis.plan(r, [kw("Tableau"), kw("Dashboards")]))
    assert got["s0e0b0"] == ["Tableau dashboards"]


def test_stakeholder_meetings_bolds_as_a_phrase_when_the_posting_asks():
    r = resume_of("Manage delivery like a project, running stakeholder meetings.")
    got = bolded(r, emphasis.plan(r, [kw("Stakeholder Management", category="soft")]))
    assert got["s0e0b0"] == ["stakeholder meetings"]


def test_skills_section_and_headings_are_never_bold():
    r = resume_of("Nothing relevant here.")
    plan = emphasis.plan(r, [kw("SQL"), kw("Tableau")])
    assert plan == {}


def _render(tmp_path, bold=True):
    payload = json.loads((FIXTURES / "allstate_payload_v2.json").read_text())
    cfg = Config(out_dir=tmp_path, max_revisions=0)
    cfg.offline = False
    cfg.bold = bold
    return run(PipelineState(job_text=allstate_jd(), max_revisions=0), cfg,
               ScriptedLLM([payload]))


def _font_runs(pdf_path: str) -> list[tuple[str, str]]:
    from pypdf import PdfReader

    runs: list[tuple[str, str]] = []

    def visit(text, cm, tm, font, size):
        if text.strip() and font:
            runs.append((text, str(font.get("/BaseFont", ""))))

    for page in PdfReader(pdf_path).pages:
        page.extract_text(visitor_text=visit)
    return runs


def test_pdf_draws_the_terms_in_bold(tmp_path):
    pytest.importorskip("pypdf")
    final = _render(tmp_path)
    bold_text = " ".join(t for t, f in _font_runs(final.pdf_path) if "Bold" in f)
    assert "data quality" in bold_text
    assert "Tableau dashboards" in bold_text


def test_bold_does_not_change_the_extracted_text(tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfReader

    final = _render(tmp_path)
    pdf_text = re.sub(r"\s+", " ", " ".join(
        p.extract_text() for p in PdfReader(final.pdf_path).pages))
    twin = Path(final.pdf_path).with_suffix(".txt").read_text()
    for line in twin.splitlines():
        if line.startswith("- "):
            assert re.sub(r"\s+", " ", line[2:]) in pdf_text


def test_no_bold_flag(tmp_path):
    final = _render(tmp_path, bold=False)
    assert final.resume.emphasis == {}
    assert not any(l.startswith("bold:") for l in final.log)


def test_change_report_lists_bolded_terms(tmp_path):
    final = _render(tmp_path)
    pdf = Path(final.pdf_path)
    body = pdf.with_name(pdf.stem + "_changes.md").read_text()
    assert "## Bolded on the page" in body and "data quality" in body

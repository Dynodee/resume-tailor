"""Regression tests built from a real run: a public Allstate Data Analytics
posting, and the model output the first version of the tool produced for it,
replayed against the fictional example fact base.

That output had four problems a reader caught and the pipeline did not:
keyword stuffing, an overlong summary, projects placed above a work history
that matched the role better, and no way to tell what tailoring had changed.
Each test below pins one of those down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import ScriptedLLM, allstate_jd, allstate_payload  # noqa: E402

from resume_tailor import report  # noqa: E402
from resume_tailor.ats import linter, scorer  # noqa: E402
from resume_tailor.config import Config  # noqa: E402
from resume_tailor.graph import run  # noqa: E402
from resume_tailor.llm import LLM  # noqa: E402
from resume_tailor.nodes import extract_keywords, ingest, tailor  # noqa: E402
from resume_tailor.sources import job_posting, resume_source  # noqa: E402
from resume_tailor.state import Keyword, PipelineState, Priority  # noqa: E402


@pytest.fixture(scope="module")
def master():
    return resume_source.load(None, Config(offline=True).master_path)


@pytest.fixture(scope="module")
def keywords():
    cfg = Config(offline=True)
    s = PipelineState(job_text=allstate_jd())
    s = s.model_copy(update=ingest.ingest(s, cfg))
    s = s.model_copy(update=extract_keywords.extract(s, cfg, LLM(cfg)))
    return s.keywords


@pytest.fixture(scope="module")
def stuffed(master):
    resume, _ = tailor._assemble(allstate_payload(), master)
    return resume


# --- posting parsing ------------------------------------------------------------

def test_workday_style_headings_are_recognised():
    sections = job_posting.from_text(allstate_jd()).sections
    assert "essential qualifications" in sections
    assert "skills" in sections
    assert "key responsibilities" in sections


def test_bullets_are_not_mistaken_for_headings():
    assert not job_posting._is_heading("- 2 years of data and analytics experience")
    assert not job_posting._is_heading("Experience working with Big Data and SQL daily.")
    assert job_posting._is_heading("Essential Qualifications")
    assert job_posting._is_heading("What you'll bring:")


def test_essential_and_skills_sections_are_required(keywords):
    by_term = {k.term: k for k in keywords}
    assert by_term["SQL"].priority is Priority.REQUIRED
    assert by_term["Microsoft Fabric"].priority is Priority.REQUIRED
    assert by_term["dbt"].priority is Priority.REQUIRED      # from the Skills list


def test_compensation_section_is_ignored():
    p = job_posting.from_text("Analyst\n\nRequirements\n- SQL\n\nBenefits\n- Python lunch club\n")
    from resume_tailor.nodes.extract_keywords import _deterministic
    assert "Python" not in _deterministic(p)


def test_title_with_consultant_is_found():
    assert "Consultant" in job_posting.from_text(allstate_jd()).title


# --- coverage --------------------------------------------------------------------

def test_terms_outside_the_fact_base_are_unattainable(stuffed, keywords, master):
    cov = scorer.score(stuffed, keywords, tailor.master_text(master))
    assert "Microsoft Fabric" in cov.unattainable
    assert "Insurance" in cov.unattainable
    assert "SQL" not in cov.unattainable


def test_soft_skills_do_not_drive_required_coverage():
    from resume_tailor.state import ResumeSection, TailoredResume
    resume = TailoredResume(sections=[ResumeSection(
        heading="X", kind="experience",
        entries=[{"end": "Present", "bullets": [{"source_id": "a", "text": "Write SQL."}]}])])
    kws = [Keyword(term="SQL", normalized="sql", priority=Priority.REQUIRED, category="skill"),
           Keyword(term="Communication", normalized="communication",
                   priority=Priority.REQUIRED, category="soft")]
    assert scorer.score(resume, kws).required_score == 100.0


# --- the writing problems in the real output --------------------------------------

def _rules(resume) -> dict[str, list]:
    out: dict[str, list] = {}
    for i in linter.lint(resume):
        out.setdefault(i.rule, []).append(i)
    return out


def test_linter_catches_the_keyword_gloss(stuffed):
    hits = _rules(stuffed)["keyword-gloss"]
    assert any("large, complex datasets" in i.excerpt for i in hits)


def test_linter_catches_soft_skill_name_drops(stuffed):
    assert len(_rules(stuffed)["soft-skill-name-drop"]) == 2


def test_linter_catches_repeated_word_in_summary(stuffed):
    assert any("business" in i.message for i in _rules(stuffed)["repetition"])


def test_long_summary_is_an_error_that_forces_a_revision(stuffed):
    issue = _rules(stuffed)["summary-length"][0]
    assert issue.severity == "error"


def test_acronym_expansions_are_not_flagged_as_glosses():
    from resume_tailor.state import TailoredResume
    r = TailoredResume(summary="Exposed tools over Model Context Protocol (MCP). "
                               "M.S. in progress (expected June 2027).")
    assert "keyword-gloss" not in _rules(r)


# --- section order ------------------------------------------------------------------

def test_experience_leads_for_an_analytics_role(stuffed, keywords):
    ordered = tailor.order_sections(stuffed, keywords)
    assert [s.kind for s in ordered.sections][:2] == ["experience", "projects"]


def test_projects_lead_for_an_agentic_ai_role(master):
    kws = [Keyword(term=t, normalized=t.lower(), priority=Priority.REQUIRED, category="tool")
           for t in ("LangGraph", "Model Context Protocol (MCP)", "Agentic AI", "LLM")]
    resume = tailor._offline_tailor(master, kws)
    assert resume.sections[0].kind == "projects"


# --- the online path ------------------------------------------------------------------

def _run_scripted(tmp_path, payloads, revisions=1):
    cfg = Config(out_dir=tmp_path, max_revisions=revisions)
    cfg.offline = False
    llm = ScriptedLLM(payloads)
    final = run(PipelineState(job_text=allstate_jd(), max_revisions=revisions), cfg, llm)
    return final, llm


def test_tailor_prompt_contains_the_role_not_just_keywords(tmp_path):
    _, llm = _run_scripted(tmp_path, [allstate_payload()], revisions=0)
    tailor_prompt = next(u for s, u in llm.prompts if "tailor an existing resume" in s)
    assert "Monitors and evaluates business initiatives" in tailor_prompt
    assert "Microsoft Fabric" in tailor_prompt.split("do not attempt these:")[1].split("\n")[0]
    assert "$75,100" not in tailor_prompt     # compensation stays out


def test_revision_shows_the_model_its_previous_draft(tmp_path):
    final, llm = _run_scripted(tmp_path, [allstate_payload()], revisions=1)
    assert final.revisions == 1            # the 64-word summary forced a second pass
    second = [u for s, u in llm.prompts if "tailor an existing resume" in s][1]
    assert "Previous draft" in second
    assert "applying problem-solving" in second
    assert "summary-length" in second


def _phrase_complete(payload: dict) -> dict:
    """The fixes exact-phrase coverage asks for, made the honest way."""
    payload["headline"] = "Data Analytics | SQL - Tableau - Power BI - dbt - Snowflake"
    payload["summary"] = ("Data analyst with 6+ years of SQL, Tableau, and Power BI "
                          "experience turning large datasets into decisions. Cut dashboard "
                          "load time 75% for 40+ users and built reporting 15+ regions rely on.")
    payload["skills"] = [
        {"label": "Data Visualization & BI", "items": ["Tableau", "Power BI", "MicroStrategy"]},
        {"label": "SQL Development & Data", "items": ["SQL", "dbt (Data Build Tool)", "Snowflake",
                                                      "Python"]},
    ]
    return payload


def test_a_clean_revision_ends_the_loop(tmp_path):
    fixed = _phrase_complete(allstate_payload())
    final, _ = _run_scripted(tmp_path, [allstate_payload(), fixed], revisions=2)
    assert final.revisions == 1
    assert not [i for i in final.grammar_issues if i.severity == "error"]
    assert final.coverage.phrase_score >= 70


# --- the change report -------------------------------------------------------------

def test_change_report_says_what_changed(tmp_path):
    final, _ = _run_scripted(tmp_path, [allstate_payload()], revisions=0)
    pdf = Path(final.pdf_path)
    changes = pdf.with_name(pdf.stem + "_changes.md")
    body = changes.read_text()
    assert "7 unchanged, 4 edited, 4 rewritten; 4 left out" in body
    assert "Microsoft Fabric" in body
    assert "`harbor_category`" in body      # left out, and listed
    assert "before:" in body and "after:" in body


def test_change_counts_match_the_real_run(master, keywords, stuffed):
    cs = report.compute(stuffed, master, keywords)
    assert cs.counts() == {"unchanged": 7, "edited": 4, "rewritten": 4}
    assert {sid for _, sid, _ in cs.left_out} == {"ag_api", "ag_mcp", "ag_sandbox",
                                                   "harbor_category"}

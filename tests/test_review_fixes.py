"""Guards added after reviewing a real run (Google.org BI Generalist): a bullet
restating another in the same role, a posting term spread across bullets, a
greeting addressed to a sub-team, and a resume that spilled onto page two."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import ScriptedLLM, allstate_jd, allstate_payload  # noqa: E402

from resume_tailor.ats import linter  # noqa: E402
from resume_tailor.config import EXAMPLE_MASTER, Config  # noqa: E402
from resume_tailor.graph import run  # noqa: E402
from resume_tailor.llm import LLM  # noqa: E402
from resume_tailor.nodes import letter as letter_node, render as render_node  # noqa: E402
from resume_tailor.nodes import tailor as tailor_node  # noqa: E402
from resume_tailor.sources import resume_source  # noqa: E402
from resume_tailor.state import Keyword, PipelineState, Priority, ResumeSection, TailoredResume  # noqa: E402


@pytest.fixture
def master():
    return resume_source.load(None, EXAMPLE_MASTER)


def _role(*bullets: tuple[str, str]) -> TailoredResume:
    return TailoredResume(sections=[ResumeSection(
        heading="WORK EXPERIENCE", kind="experience",
        entries=[{"title": "Senior Business Analyst", "company": "Northwind", "end": "Present",
                  "bullets": [{"source_id": s, "text": t} for s, t in bullets]}])])


def test_a_bullet_restating_another_in_the_same_role_is_an_error():
    # The two Ross bullets from the real run, reduced to their shape.
    resume = _role(
        ("northwind_sql", "Design data structures for longitudinal tracking, writing SQL and "
                          "MDX for data validation across datasets and cubes in large reporting "
                          "systems."),
        ("confirmed_data_profiling", "Apply data profiling to the structures mapped in SQL and "
                                     "MDX across large reporting systems, informing fixes."))
    issues = linter.repeated_work(resume)
    assert [i.rule for i in issues] == ["repeated-work"]
    assert issues[0].severity == "error" and "bullets[1]" in issues[0].location


def test_different_work_in_one_role_is_fine(master):
    resume = tailor_node._offline_tailor(master, [])
    assert linter.repeated_work(resume) == []


BI = Keyword(term="Business Intelligence", normalized="business intelligence",
             priority=Priority.REQUIRED)


def test_a_posting_term_written_into_two_bullets_is_an_error(master):
    resume = _role(
        ("northwind_reqs", "Translate ambiguous business requirements into Business "
                           "Intelligence specifications, partnering with IT."),
        ("harbor_share", "Gave 15+ regions real-time Business Intelligence through Tableau "
                         "dashboards."))
    issues = linter.spread_terms(resume, master, [BI])
    assert [i.rule for i in issues] == ["term-spread"] and issues[0].severity == "error"
    assert "'Business Intelligence' was written into 2 bullets" in issues[0].message


def test_once_is_fine_and_hinted_terms_do_not_count(master):
    once = _role(("harbor_share", "Built Tableau business intelligence dashboards that 15+ "
                                  "regions relied on."),
                 ("northwind_reqs", "Translate ambiguous business requirements into technical "
                                    "specifications."))
    assert linter.spread_terms(once, master, [BI]) == []
    # "data quality" is a keyword hint on both of these facts: supported, not stuffing.
    dq = Keyword(term="Data Quality", normalized="data quality", priority=Priority.REQUIRED)
    hinted = _role(("harbor_stack", "Protected data quality with Python and SQL."),
                   ("state_queries", "Wrote queries that caught data quality errors."))
    assert linter.spread_terms(hinted, master, [dq]) == []


def test_tailor_prompt_carries_the_rules_and_a_one_page_budget(master, tmp_path):
    llm = ScriptedLLM([allstate_payload()])
    cfg = Config(out_dir=tmp_path, offline=True, max_revisions=0)
    run(PipelineState(job_text=allstate_jd(), max_revisions=0), cfg, llm)
    prompt = next(u for s, u in llm.prompts if "tailor an existing resume" in s)
    system = next(s for s, _ in llm.prompts if "tailor an existing resume" in s)
    assert "ONE PAGE." in prompt and "At most 11 bullets" in prompt
    assert "more than one bullet whose fact does not already" in system
    assert "ONE BULLET PER PIECE OF WORK" in system


def test_one_page_is_the_default():
    assert Config(offline=True).max_pages == 1
    assert "At most 15 bullets" in tailor_node.budget(2)


def test_render_cuts_the_least_relevant_bullets_to_fit(master, tmp_path):
    cfg = Config(out_dir=tmp_path, offline=True, max_revisions=0)
    resume = tailor_node._offline_tailor(master, [])     # 16 bullets: two pages
    before = {b["source_id"] for s in resume.sections for e in s.entries
              for b in e.get("bullets", []) if isinstance(b, dict)}
    out = render_node.render(PipelineState(resume=resume, master=master), cfg)
    after = {b["source_id"] for s in out["resume"].sections for e in s.entries
             for b in e.get("bullets", []) if isinstance(b, dict)}
    assert not any(e.startswith("layout:") for e in out["errors"])
    assert after < before
    cut_line = next(line for line in out["log"] if "bullet(s) to fit 1 page" in line)
    assert all(sid in cut_line for sid in before - after)
    # Projects lose bullets first; every entry keeps at least one.
    assert all(e.get("bullets") for s in out["resume"].sections
               if s.kind in ("experience", "projects") for e in s.entries)


def test_letter_greets_the_company_not_a_sub_team():
    assert "never a\nsub-team, program or department name" in letter_node.SYSTEM
    assert '"Hey Google team,"' in letter_node.SYSTEM

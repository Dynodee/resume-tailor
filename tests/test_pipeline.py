from __future__ import annotations

from pathlib import Path

import pytest

from resume_tailor.ats import lexicon, linter, scorer
from resume_tailor.config import Config
from resume_tailor.graph import run
from resume_tailor.llm import LLM, _extract_json
from resume_tailor.nodes import tailor as tailor_node
from resume_tailor.sources import job_posting, resume_source
from resume_tailor.state import (
    Keyword,
    PipelineState,
    Priority,
    ResumeSection,
    TailoredResume,
)

POSTING = """
Senior AI Engineer

About the role
You will build agentic workflows and LLM-powered products using Python.

Requirements
- 5+ years of Python
- Experience with LLMs and multi-agent orchestration
- Strong SQL and data modeling
- Bachelor's degree in Computer Science or a quantitative field

Preferred qualifications
- LangGraph or LangChain
- Familiarity with MCP
- Kubernetes

Benefits
- Free lunch and a fast-paced environment
"""


@pytest.fixture(scope="module")
def master():
    return resume_source.load(None, Config().master_path)


@pytest.fixture(scope="module")
def posting():
    return job_posting.from_text(POSTING)


# --- lexicon -----------------------------------------------------------------

def test_lexicon_finds_canonical_terms():
    hits = lexicon.find_terms("We use Python, SQL and Tableau daily.")
    assert {"Python", "SQL", "Tableau"} <= set(hits)


def test_lexicon_matches_aliases_to_canonical_form():
    hits = lexicon.find_terms("Experience with large language models is required.")
    assert "LLM" in hits
    assert "large language models" in hits["LLM"]["matched_aliases"]


def test_lexicon_respects_token_boundaries():
    # "R" should not match inside other words, and "Go" should not match "Google".
    assert "R" not in lexicon.find_terms("Great work on the report.")
    assert "Go" not in lexicon.find_terms("We use Google Cloud.")


def test_lexicon_handles_punctuated_terms():
    assert "CI/CD" in lexicon.find_terms("Own the CI/CD pipeline.")


def test_evidence_is_captured():
    hits = lexicon.find_terms("Line one.\nStrong SQL skills required.\n")
    assert "SQL" in hits["SQL"]["evidence"]


# --- posting parsing ----------------------------------------------------------

def test_sections_are_split(posting):
    keys = set(posting.sections)
    assert any("requirement" in k for k in keys)
    assert any("preferred" in k for k in keys)


def test_title_is_guessed(posting):
    assert "Engineer" in posting.title


def test_priority_comes_from_section(posting):
    from resume_tailor.nodes.extract_keywords import _deterministic

    found = _deterministic(posting)
    assert found["Python"].priority is Priority.REQUIRED
    assert found["Kubernetes"].priority is Priority.PREFERRED


def test_title_terms_are_forced_required():
    p = job_posting.from_text("Staff Python Engineer\n\nNice to have\n- Python\n")
    from resume_tailor.nodes.extract_keywords import _deterministic

    assert _deterministic(p)["Python"].priority is Priority.REQUIRED


# --- coverage scoring ---------------------------------------------------------

def _resume_with(text: str) -> TailoredResume:
    return TailoredResume(
        headline="", summary="",
        sections=[ResumeSection(
            heading="WORK EXPERIENCE", kind="experience",
            entries=[{"title": "Analyst", "company": "X", "end": "Present",
                      "bullets": [{"source_id": "a", "text": text}]}],
        )],
    )


def test_coverage_counts_alias_matches():
    kw = Keyword(term="LLM", normalized="llm", aliases=["large language models"],
                 priority=Priority.REQUIRED)
    report = scorer.score(_resume_with("Shipped large language models to production."),
                          [kw])
    assert report.required_score == 100.0


def test_coverage_reports_gaps():
    kws = [
        Keyword(term="Python", normalized="python", priority=Priority.REQUIRED),
        Keyword(term="Kubernetes", normalized="kubernetes", priority=Priority.REQUIRED),
    ]
    report = scorer.score(_resume_with("Wrote Python services."), kws)
    assert [k.term for k in report.missing()] == ["Kubernetes"]
    assert 0 < report.required_score < 100


def test_required_terms_outweigh_mentioned():
    req = Keyword(term="Python", normalized="python", priority=Priority.REQUIRED)
    men = Keyword(term="Docker", normalized="docker", priority=Priority.MENTIONED)
    hit_required = scorer.score(_resume_with("Wrote Python services."), [req, men])
    hit_mentioned = scorer.score(_resume_with("Ran Docker containers."), [req, men])
    assert hit_required.score > hit_mentioned.score


def test_coverage_records_where_a_term_hit():
    kw = Keyword(term="Python", normalized="python", priority=Priority.REQUIRED)
    report = scorer.score(_resume_with("Wrote Python services."), [kw])
    assert report.items[0].where == ["experience"]


# --- linter ------------------------------------------------------------------

def _rules(resume) -> set[str]:
    return {i.rule for i in linter.lint(resume)}


def test_linter_flags_first_person():
    assert "first-person" in _rules(_resume_with("I built the pipeline."))


def test_linter_flags_weak_opener():
    assert "weak-opener" in _rules(_resume_with("Responsible for reporting."))


def test_linter_flags_tense_mismatch_in_current_role():
    # "end" is Present in the fixture, so a past-tense verb is wrong.
    assert "tense-consistency" in _rules(_resume_with("Built the reporting stack."))


def test_linter_accepts_a_clean_bullet():
    clean = _resume_with("Build reporting pipelines that serve 40+ users daily.")
    assert not [i for i in linter.lint(clean) if i.severity == "error"]


def test_linter_flags_repeated_words_and_spacing():
    rules = _rules(_resume_with("Build build the  pipeline ."))
    assert {"repeated-word", "double-space", "space-before-punctuation"} <= rules


def test_linter_flags_missing_terminal_period():
    assert "terminal-punctuation" in _rules(_resume_with("Build reporting pipelines"))


# --- fact-base discipline -----------------------------------------------------

def test_assemble_drops_untraceable_bullets(master):
    payload = {
        "headline": "H", "summary": "S",
        "experience": [{"id": "northwind", "bullets": [
            {"source_id": "northwind_sql", "text": "Write SQL and MDX daily."},
            {"source_id": "does_not_exist", "text": "Invented a time machine."},
        ]}],
        "skills": [{"label": "Programming", "items": ["Python", "Fortran"]}],
    }
    resume, dropped = tailor_node._assemble(payload, master)
    text = resume.all_text()
    assert "time machine" not in text
    assert "Fortran" not in text
    assert "Python" in text
    assert any("does_not_exist" in d for d in dropped)
    assert any("Fortran" in d for d in dropped)


def test_audit_flags_terms_absent_from_fact_base(master):
    resume = _resume_with("Deployed models on Kubernetes at scale.")
    assert any("Kubernetes" in w for w in tailor_node.audit(resume, master))


def test_audit_is_quiet_on_faithful_text(master):
    resume = _resume_with("Write SQL and MDX to validate data structures.")
    assert tailor_node.audit(resume, master) == []


# --- llm helpers --------------------------------------------------------------

def test_json_extraction_survives_fences_and_prose():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Here you go:\n[1, 2]\nHope that helps!') == [1, 2]
    assert _extract_json("not json at all") is None


# --- end to end ---------------------------------------------------------------

def test_offline_run_produces_a_pdf(tmp_path: Path):
    cfg = Config(out_dir=tmp_path, offline=True, max_revisions=1)
    state = PipelineState(job_text=POSTING, max_revisions=1)
    final = run(state, cfg, LLM(cfg))

    assert final.pdf_path is not None
    pdf = Path(final.pdf_path)
    assert pdf.exists() and pdf.stat().st_size > 2000
    assert pdf.read_bytes().startswith(b"%PDF")

    twin = pdf.with_suffix(".txt")
    assert twin.exists()
    body = twin.read_text()
    assert "JORDAN RIVERA" in body.upper()
    assert "WORK EXPERIENCE" in body
    assert "Northwind Retail Group" in body


def test_offline_run_scores_and_terminates(tmp_path: Path):
    cfg = Config(out_dir=tmp_path, offline=True, max_revisions=2)
    final = run(PipelineState(job_text=POSTING, max_revisions=2), cfg, LLM(cfg))
    assert final.coverage is not None
    assert final.coverage.score > 0
    assert final.revisions <= 2


def test_offline_tailoring_prefers_relevant_projects(master):
    kws = [Keyword(term="LangGraph", normalized="langgraph", priority=Priority.REQUIRED),
           Keyword(term="Agentic AI", normalized="agentic ai", priority=Priority.REQUIRED)]
    resume = tailor_node._offline_tailor(master, kws)
    projects = next(s for s in resume.sections if s.kind == "projects")
    assert "Multi-Agent" in projects.entries[0]["name"]


def test_langgraph_and_fallback_agree(tmp_path: Path):
    """Both executors must produce the same document from the same input."""
    pytest.importorskip("langgraph")
    from resume_tailor.graph import _run_langgraph, _run_simple, build_nodes

    results = []
    for runner in (_run_langgraph, _run_simple):
        cfg = Config(out_dir=tmp_path / runner.__name__, offline=True, max_revisions=1)
        nodes = build_nodes(cfg, LLM(cfg))
        final = runner(PipelineState(job_text=POSTING, max_revisions=1), nodes, None)
        assert final.pdf_path is not None
        results.append((final.resume.all_text(), final.coverage.score))
    assert results[0] == results[1]


def test_short_posting_raises_a_warning(tmp_path: Path):
    cfg = Config(out_dir=tmp_path, offline=True, max_revisions=0)
    final = run(PipelineState(job_text="Engineer wanted.", max_revisions=0),
                cfg, LLM(cfg))
    assert any("too short" in e for e in final.errors)


# --- resume parsing -----------------------------------------------------------

def test_parse_resume_text_recovers_structure():
    doc = """JORDAN RIVERA
Applied AI | Python
555-555-0142 | jordan.rivera@example.com | linkedin.com/in/jordan-rivera-example

PROFESSIONAL SUMMARY
Analytics professional with 6+ years of experience building things.

WORK EXPERIENCE
Senior Business Analyst - Northwind Retail Group
March 2024 - Present
- Translate requirements into technical specifications.
- Write SQL and MDX to validate data structures.

EDUCATION
Master's Degree, Computer Science - Cascade State University
January 2024 - 2027

SKILLS
Programming & Data: Python, SQL, MDX
"""
    parsed = resume_source.parse_resume_text(doc)
    assert parsed["contact"]["email"] == "jordan.rivera@example.com"
    assert parsed["experience"][0]["title"] == "Senior Business Analyst"
    assert parsed["experience"][0]["end"].lower() == "present"
    assert len(parsed["experience"][0]["bullets"]) == 2
    assert "Python" in parsed["skills"]["programming_data"]["items"]


def test_merge_keeps_curated_guardrails(master):
    parsed = resume_source.parse_resume_text("JORDAN RIVERA\n\nSKILLS\nX: Y\n")
    merged = resume_source.merge(master, parsed)
    assert merged["do_not_claim"] == master["do_not_claim"]
    assert merged["experience"] == master["experience"]


# --- config -------------------------------------------------------------------

def test_dotenv_is_loaded_and_shell_wins(tmp_path, monkeypatch):
    from resume_tailor.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text('# comment\nRT_TEST_A="from-file"\nexport RT_TEST_B=b\nRT_TEST_C=file\n')
    monkeypatch.delenv("RT_TEST_A", raising=False)
    monkeypatch.delenv("RT_TEST_B", raising=False)
    monkeypatch.setenv("RT_TEST_C", "shell")
    load_dotenv(env)
    import os
    assert os.environ["RT_TEST_A"] == "from-file"
    assert os.environ["RT_TEST_B"] == "b"
    assert os.environ["RT_TEST_C"] == "shell"


def test_placeholder_key_means_offline(tmp_path):
    cfg = Config(api_key="sk-ant-...", out_dir=tmp_path)
    assert cfg.offline and cfg.api_key is None


def test_certifications_render_under_education(master, tmp_path):
    from resume_tailor.nodes.tailor import _offline_tailor
    resume = _offline_tailor(master, [])
    edu = next(s for s in resume.sections if s.kind == "education")
    assert edu.heading == "EDUCATION & CERTIFICATIONS"
    assert any("Tableau" in e["degree"] for e in edu.entries)


def test_unsourced_skills_stay_off(master):
    items = {i for g in master["skills"].values() for i in g["items"]}
    assert "Retrieval & Context Engineering" not in items
    assert "Git/GitHub" in items

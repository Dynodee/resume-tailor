"""Exact-phrase coverage: the posting's own words, matched literally."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FIXTURES, ScriptedLLM, allstate_jd, allstate_payload  # noqa: E402

from resume_tailor.ats import phrases  # noqa: E402
from resume_tailor.config import Config  # noqa: E402
from resume_tailor.graph import run  # noqa: E402
from resume_tailor.llm import LLM  # noqa: E402
from resume_tailor.nodes import extract_keywords, ingest, tailor  # noqa: E402
from resume_tailor.sources import job_posting, resume_source  # noqa: E402
from resume_tailor.state import ExactPhrase, PipelineState, Priority, TailoredResume  # noqa: E402


@pytest.fixture(scope="module")
def master():
    return resume_source.load(None, Config(offline=True).master_path)


@pytest.fixture(scope="module")
def extracted():
    cfg = Config(offline=True)
    s = PipelineState(job_text=allstate_jd())
    s = s.model_copy(update=ingest.ingest(s, cfg))
    return s.model_copy(update=extract_keywords.extract(s, cfg, LLM(cfg)))


@pytest.fixture(scope="module")
def v2(master):
    import json
    payload = json.loads((FIXTURES / "allstate_payload_v2.json").read_text())
    return tailor._assemble(payload, master)[0]


def _status(resume, extracted, master) -> dict[str, str]:
    items, _ = phrases.score(resume, extracted.phrases, tailor.master_text(master))
    return {i.phrase.text: i.status for i in items}


# --- literal matching rules -----------------------------------------------------------

@pytest.mark.parametrize("phrase,text,expected", [
    ("Data Visualization", "Data Visualization & BI", True),
    ("data visualization", "DATA VISUALIZATIONS", True),          # case and plural
    ("Ad Hoc Analysis", "ad-hoc analysis for finance", True),     # hyphen = space
    ("Data Visualization", "dashboards and visualizations", False),
    ("Data Analytics", "Data & Business Analytics", False),       # not contiguous
    ("SQL", "MySQL administrator", False),                        # token boundary
    ("Structured Query Language (SQL)", "Write SQL daily.", True),  # acronym form
    ("Structured Query Language (SQL) Development", "Owns SQL development.", True),
    ("Structured Query Language (SQL) Development", "Write SQL daily.", False),
])
def test_literal_matching(phrase, text, expected):
    assert phrases.literal_in(phrase, text) is expected


def test_alternatives_for_tagged_acronyms():
    assert phrases.alternatives("Structured Query Language (SQL) Development") == [
        "Structured Query Language (SQL) Development",
        "Structured Query Language Development",
        "SQL Development",
    ]


# --- collecting phrases ------------------------------------------------------------------

def test_workday_skills_list_becomes_exact_phrases(extracted):
    tags = {p.text for p in extracted.phrases if p.source == "skills list"}
    assert {"Ad Hoc Analysis", "Data Build Tool", "Data Visualization",
            "Structured Query Language (SQL) Development"} <= tags


def test_skills_tags_are_required(extracted):
    dbt = next(p for p in extracted.phrases if p.text == "Data Build Tool")
    assert dbt.priority is Priority.REQUIRED
    assert dbt.concept == "dbt"


def test_requirement_wording_is_taken_from_its_own_line(extracted):
    texts = {p.text for p in extracted.phrases}
    assert "Microsoft Fabric" in texts
    # "SQL" is already accepted by the "Structured Query Language (SQL)" tag,
    # so it is not listed a second time.
    assert "SQL" not in texts and "Structured Query Language" not in texts


def test_prose_under_a_skills_heading_is_not_split_into_tags():
    p = job_posting.from_text(
        "Analyst\n\nSkills\nYou will spend most of your week building complex models "
        "in SQL for finance partners, and presenting them to leadership every month.\n")
    assert not [x for x in phrases.collect(p, []) if x.source == "skills list"]


def test_soft_tags_are_marked_soft(extracted):
    assert next(p for p in extracted.phrases if p.text == "Business Influence").soft


# --- scoring the real second Allstate output -------------------------------------------

def test_v2_statuses_match_a_literal_search(v2, extracted, master):
    st = _status(v2, extracted, master)
    assert st["Data Analysis"] == "exact"            # inside "Exploratory Data Analysis"
    assert st["Tableau"] == "exact"
    assert st["Data Build Tool"] == "concept"        # page says "dbt"
    assert st["Data Visualization"] == "concept"     # page says "dashboards"
    assert st["Data Analytics"] == "concept"         # page says "Data & Business Analytics"
    assert st["Microsoft Fabric"] == "unsupported"
    assert st["Ad Hoc Analysis"] == "unsupported"


def test_v2_phrase_score(v2, extracted, master):
    _, pct = phrases.score(v2, extracted.phrases, tailor.master_text(master))
    assert 40 <= pct <= 50       # 4 of 9 supported phrases word for word


def test_unsupported_and_soft_phrases_do_not_count(extracted):
    only_bad = [ExactPhrase(text="Microsoft Fabric", priority=Priority.REQUIRED),
                ExactPhrase(text="Business Influence", priority=Priority.REQUIRED, soft=True)]
    _, pct = phrases.score(TailoredResume(), only_bad, "nothing relevant here")
    assert pct == 100.0


# --- the honest ways to get a phrase on the page ----------------------------------------

def test_skill_item_may_carry_the_postings_name():
    allowed = {"dbt", "sql"}
    assert tailor.skill_allowed("dbt (Data Build Tool)", allowed)
    assert tailor.skill_allowed("Structured Query Language (SQL)", allowed)
    assert not tailor.skill_allowed("dbt (Microsoft Fabric)", allowed)
    assert not tailor.skill_allowed("Data Build Tool", allowed)   # bare rename: not listed


def test_group_label_cannot_name_an_unsupported_skill(master):
    payload = allstate_payload()
    payload["skills"] = [{"label": "Insurance Analytics", "items": ["SQL"]}]
    resume, dropped = tailor._assemble(payload, master, forbidden=["Insurance"])
    skills = next(s for s in resume.sections if s.kind == "skills")
    assert skills.entries[0]["label"] == "Skills"
    assert any("Insurance Analytics" in d for d in dropped)


# --- through the graph -------------------------------------------------------------------

def _run(tmp_path, payloads, revisions):
    cfg = Config(out_dir=tmp_path, max_revisions=revisions)
    cfg.offline = False
    llm = ScriptedLLM(payloads)
    return run(PipelineState(job_text=allstate_jd(), max_revisions=revisions), cfg, llm), llm


def test_prompt_lists_supported_phrases_and_bans_unsupported(tmp_path):
    _, llm = _run(tmp_path, [allstate_payload()], 0)
    prompt = next(u for s, u in llm.prompts if "tailor an existing resume" in s)
    block = prompt.split("# EXACT PHRASES A RECRUITER WILL SEARCH FOR")[1].split("#")[0]
    assert "- Data Build Tool" in block
    assert "Microsoft Fabric" not in block
    banned = prompt.split("do not attempt these:")[1].split("\n")[0]
    assert "Ad Hoc Analysis" in banned and "Microsoft Fabric" in banned


def test_low_phrase_coverage_triggers_a_revision_with_specifics(tmp_path):
    final, llm = _run(tmp_path, [allstate_payload()], 1)
    assert final.revisions == 1
    second = [u for s, u in llm.prompts if "tailor an existing resume" in s][1]
    feedback = second.split("only says in other words")[1]
    assert "Data Build Tool" in feedback and "Data Visualization" in feedback


def test_change_report_has_the_phrase_table(tmp_path):
    final, _ = _run(tmp_path, [allstate_payload()], 0)
    pdf = Path(final.pdf_path)
    body = pdf.with_name(pdf.stem + "_changes.md").read_text()
    assert "## Exact-phrase coverage" in body
    assert "in other words only" in body
    assert "Data Build Tool" in body

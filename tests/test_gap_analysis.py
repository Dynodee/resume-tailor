"""Gap analysis: sorting unstated requirements, the reviewer, interview mode,
and the rule that a reframed term stays on the bullets that earned it."""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import ScriptedLLM  # noqa: E402

from resume_tailor import report  # noqa: E402
from resume_tailor.ats import scorer  # noqa: E402
from resume_tailor.config import EXAMPLE_MASTER, Config  # noqa: E402
from resume_tailor.graph import run  # noqa: E402
from resume_tailor.llm import LLM  # noqa: E402
from resume_tailor.nodes import gap_analysis, tailor as tailor_node  # noqa: E402
from resume_tailor.sources import job_posting, resume_source  # noqa: E402
from resume_tailor.state import GapItem, Keyword, PipelineState, Priority, TailoredResume  # noqa: E402

POSTING = """Analytics Engineer

Requirements
- Python and SQL
- CI/CD for data pipelines
- Kubernetes
- Trade Promotion Management experience

Preferred
- Looker
"""


def kw(term, priority=Priority.REQUIRED, category="skill"):
    return Keyword(term=term, normalized=term.lower(), priority=priority, category=category)


KEYWORDS = [kw("Python"), kw("CI/CD"), kw("Kubernetes", category="tool"),
            kw("Trade Promotion Management"), kw("Looker", Priority.PREFERRED, "tool"),
            kw("Communication", category="soft")]


@pytest.fixture
def master():
    return resume_source.load(None, EXAMPLE_MASTER)


def _state(master, **kw_) -> PipelineState:
    return PipelineState(posting=job_posting.from_text(POSTING), keywords=KEYWORDS,
                         master=master, **kw_)


def _cfg(tmp_path, **kw_) -> Config:
    return Config(out_dir=tmp_path / "out", offline=True, **kw_)


SORTED = {"items": [
    {"term": "CI/CD", "disposition": "reframe", "source_ids": ["ag_eval"],
     "rationale": "'30 unit tests gating' every release", "question": ""},
    {"term": "Kubernetes", "disposition": "gap", "source_ids": [], "rationale": "", "question": ""},
    {"term": "Looker", "disposition": "adjacent", "source_ids": ["harbor_share"],
     "rationale": "built Tableau dashboards", "question": ""},
]}


def test_candidates_are_the_unstated_requirements(master):
    terms = {g.term for g in gap_analysis.candidates(_state(master))}
    assert terms == {"CI/CD", "Kubernetes", "Trade Promotion Management", "Looker"}


def test_reframe_survives_review_and_becomes_evidence(master, tmp_path):
    llm = ScriptedLLM(gaps=SORTED, verdicts={"verdicts": [
        {"term": "CI/CD", "source_id": "ag_eval", "supported": True, "reason": "gates releases"}]})
    out = gap_analysis.analyze(_state(master), _cfg(tmp_path), llm)
    by = {g.term: g for g in out["gaps"]}
    assert by["CI/CD"].disposition == "reframe" and by["CI/CD"].source_ids == ["ag_eval"]
    assert by["Looker"].disposition == "adjacent"
    assert by["Kubernetes"].disposition == "gap"
    # On the do-not-claim list, so ruled out before the model ever sees it.
    assert by["Trade Promotion Management"].disposition == "gap"
    assert "rules it out" in by["Trade Promotion Management"].rationale
    sorted_prompt = next(u for s, u in llm.prompts if "sort job requirements" in s)
    to_sort = sorted_prompt.split("# REQUIREMENTS TO SORT")[1].split("# CLAIMS")[0]
    assert "Trade Promotion Management" not in to_sort and "Looker" in to_sort

    # The reframe is now a hint on its bullet, so the term is attainable...
    new_master = out["master"]
    bullet = tailor_node._bullet_index(new_master)["ag_eval"]
    assert "CI/CD" in bullet["keywords"]
    unattainable = scorer.score(TailoredResume(), KEYWORDS,
                                tailor_node.master_text(new_master)).unattainable
    assert "CI/CD" not in unattainable and "Kubernetes" in unattainable
    # ...and the original fact base is untouched.
    assert "CI/CD" not in tailor_node._bullet_index(master)["ag_eval"]["keywords"]


def test_rejected_reframe_becomes_related_experience(master, tmp_path):
    llm = ScriptedLLM(gaps=SORTED, verdicts={"verdicts": [
        {"term": "CI/CD", "source_id": "ag_eval", "supported": False,
         "reason": "unit tests are not a deployment pipeline"}]})
    out = gap_analysis.analyze(_state(master), _cfg(tmp_path), llm)
    g = next(g for g in out["gaps"] if g.term == "CI/CD")
    assert g.disposition == "adjacent" and "reviewer" in g.rationale
    assert "CI/CD" not in tailor_node._bullet_index(out["master"])["ag_eval"]["keywords"]


def test_no_verdict_counts_as_rejection(master, tmp_path):
    out = gap_analysis.analyze(_state(master), _cfg(tmp_path), ScriptedLLM(gaps=SORTED))
    assert next(g for g in out["gaps"] if g.term == "CI/CD").disposition == "adjacent"


def test_evidence_that_points_nowhere_becomes_a_question(master, tmp_path):
    bad = copy.deepcopy(SORTED)
    bad["items"][0]["source_ids"] = ["no_such_bullet"]
    out = gap_analysis.analyze(_state(master), _cfg(tmp_path), ScriptedLLM(gaps=bad))
    g = next(g for g in out["gaps"] if g.term == "CI/CD")
    assert g.disposition == "ask" and "CI/CD" in g.question


def test_terms_the_model_skips_become_questions(master, tmp_path):
    out = gap_analysis.analyze(_state(master), _cfg(tmp_path), ScriptedLLM())
    open_ = [g for g in out["gaps"] if g.term != "Trade Promotion Management"]
    assert {g.disposition for g in open_} == {"ask"}


def test_offline_everything_open_is_a_question(master, tmp_path):
    cfg = _cfg(tmp_path)
    out = gap_analysis.analyze(_state(master), cfg, LLM(cfg))
    assert {g.disposition for g in out["gaps"] if g.term != "Trade Promotion Management"} == {"ask"}


# --- interview mode -------------------------------------------------------------------------

def _scripted(*answers):
    queue = list(answers)
    asked = []

    def ask(question: str) -> str:
        asked.append(question)
        return queue.pop(0)
    ask.asked = asked
    return ask


def test_interview_confirms_declines_and_remembers(tmp_path):
    fact_base = tmp_path / "fact_base.yaml"
    shutil.copy(EXAMPLE_MASTER, fact_base)
    # Order: required first (CI/CD, Kubernetes), then preferred (Looker).
    # Entries menu: northwind, harbor, state, agents, basket -> "4" is agents.
    ask = _scripted("yes, I set up GitHub Actions to run our pytest suite on every push", "4",
                    "no", "")
    cfg = _cfg(tmp_path, master_path=fact_base, ask=ask)
    state = _state(resume_source.load(None, fact_base))
    out = gap_analysis.analyze(state, cfg, LLM(cfg))

    by = {g.term: g for g in out["gaps"]}
    assert by["CI/CD"].disposition == "confirmed"
    assert by["Kubernetes"].disposition == "declined"
    assert by["Looker"].disposition == "ask"
    assert len(ask.asked) == 4                     # three questions plus "where was that?"

    saved = resume_source.load_confirmed(cfg.confirmed_path)
    assert saved["confirmed"][0]["entry_id"] == "agents"
    assert saved["confirmed"][0]["text"] == ("Set up GitHub Actions to run our pytest suite on "
                                             "every push.")
    assert saved["declined"][0]["term"] == "Kubernetes"

    # Next run: the confirmed fact is evidence, the declined term is not asked again.
    ask2 = _scripted("")
    cfg2 = _cfg(tmp_path, master_path=fact_base, ask=ask2)
    loaded = tailor_node.load_master(PipelineState(), cfg2)["master"]
    agents = next(e for e in loaded["projects"] if e["id"] == "agents")
    assert agents["bullets"][-1]["keywords"] == ["CI/CD"]
    out2 = gap_analysis.analyze(_state(loaded), cfg2, LLM(cfg2))
    by2 = {g.term: g for g in out2["gaps"]}
    assert "CI/CD" not in by2
    assert by2["Kubernetes"].disposition == "declined"
    assert len(ask2.asked) == 1 and "Looker" in ask2.asked[0]


def test_interview_never_writes_next_to_the_example(master, tmp_path):
    ask = _scripted("yes, I used Looker for a month", "1", "", "")
    cfg = _cfg(tmp_path, ask=ask)
    gap_analysis.analyze(_state(master), cfg, LLM(cfg))
    assert not cfg.confirmed_path.exists()


def test_polished_answer_may_not_add_names_or_numbers(master):
    g = GapItem(term="CI/CD", question="Have you built CI pipelines?")
    llm = ScriptedLLM(facts=[{"text": "Built Jenkins pipelines deploying 40 services."}])
    text = gap_analysis._bullet_from_answer(g, "I set up GitHub Actions for our tests", None, llm)
    assert text == "Set up GitHub Actions for our tests."
    llm = ScriptedLLM(facts=[{"text": "Set up GitHub Actions to run tests on every push."}])
    text = gap_analysis._bullet_from_answer(g, "set up github actions to run tests on every push",
                                            None, llm)
    assert text == "Set up GitHub Actions to run tests on every push."


# --- the tailor honours the sorting -----------------------------------------------------------

def test_reframed_terms_stay_on_their_bullets(master):
    resume = tailor_node._offline_tailor(master, [])
    bullets = {b["source_id"]: b for s in resume.sections for e in s.entries
               for b in e.get("bullets", []) if isinstance(b, dict)}
    bullets["ag_graph"]["text"] = "Built a LangGraph system whose CI/CD gates every release."
    bullets["northwind_sql"]["text"] = "Write SQL deployed through CI/CD pipelines."
    resume.summary = "Analyst who owns CI/CD."
    gaps = [GapItem(term="CI/CD", disposition="reframe", source_ids=["ag_graph"])]

    notes = tailor_node.enforce_reframes(resume, master, gaps)
    assert "CI/CD" in bullets["ag_graph"]["text"]
    assert bullets["northwind_sql"]["text"] == tailor_node._bullet_index(master)["northwind_sql"]["text"]
    assert any(n.startswith("reverted: bullet northwind_sql") for n in notes)
    assert any(n.startswith("fabrication check") and "summary" in n for n in notes)


def test_adjacent_terms_never_appear_on_rewritten_bullets(master):
    resume = tailor_node._offline_tailor(master, [])
    b = next(b for s in resume.sections for e in s.entries for b in e.get("bullets", [])
             if isinstance(b, dict) and b["source_id"] == "harbor_share")
    b["text"] = "Built Looker dashboards that 15+ regions relied on."
    gaps = [GapItem(term="Looker", disposition="adjacent", source_ids=["harbor_share"])]
    tailor_node.enforce_reframes(resume, master, gaps)
    assert "Looker" not in b["text"]


def test_tailor_prompt_carries_the_sorting(master, tmp_path):
    gaps = [GapItem(term="CI/CD", disposition="reframe", source_ids=["ag_eval"],
                    rationale="tests gate every release"),
            GapItem(term="Looker", disposition="adjacent", source_ids=["harbor_share"],
                    rationale="Tableau dashboards")]
    llm = ScriptedLLM([{"headline": "H", "summary": "S."}])
    state = _state(master, gaps=gaps)
    tailor_node.tailor(state, _cfg(tmp_path), llm)
    prompt = next(u for s, u in llm.prompts if "tailor an existing resume" in s)
    assert "- CI/CD: bullets ag_eval -- tests gate every release" in prompt
    assert "- Looker: bullets harbor_share -- Tableau dashboards" in prompt
    assert llm.calls[-1]["effort"] == "high"


def test_skills_label_cannot_name_a_reframed_term(master):
    payload = {"skills": [{"label": "CI/CD & Testing", "items": ["pytest"]}]}
    resume, dropped = tailor_node._assemble(payload, master, ["CI/CD"])
    skills = next(s for s in resume.sections if s.kind == "skills")
    assert skills.entries[0]["label"] == "Skills"
    assert any("CI/CD" in d for d in dropped)


# --- the report accounts for everything ------------------------------------------------------

def test_every_requirement_is_accounted_for(tmp_path):
    cfg = Config(out_dir=tmp_path, offline=True, max_revisions=0)
    final = run(PipelineState(job_text=POSTING + "\nAbout the role\n" + "We build data "
                              "products for retail teams. " * 20, max_revisions=0), cfg, LLM(cfg))
    rows = report.accounting(final)
    wanted = {k.term for k in final.keywords
              if k.priority is not Priority.MENTIONED and k.category != "soft"}
    assert wanted <= {t for t, _, _, _ in rows}
    assert all(status in {"on_page", "unused", "reframe", "confirmed", "adjacent", "ask",
                          "declined", "gap"} for _, _, status, _ in rows)
    changes = Path(final.pdf_path).with_name(Path(final.pdf_path).stem + "_changes.md")
    assert "## Every requirement, accounted for" in changes.read_text()
    assert report.accounting_line(rows).startswith(f"{len(rows)} requirements:")

"""Profiles, importing a resume, lexicon packs, headings and extra sections."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from fakes import ScriptedLLM  # noqa: E402

from resume_tailor import cli  # noqa: E402
from resume_tailor.ats import lexicon  # noqa: E402
from resume_tailor.config import PROFILES_DIR, Config  # noqa: E402
from resume_tailor.graph import run  # noqa: E402
from resume_tailor.llm import LLM  # noqa: E402
from resume_tailor.nodes import tailor as tailor_node  # noqa: E402
from resume_tailor.render import pdf  # noqa: E402
from resume_tailor.sources import importer, resume_source  # noqa: E402
from resume_tailor.state import PipelineState  # noqa: E402

# A fictional nurse: nothing in the built-in lexicon, which is tech and data.
RESUME = """MORGAN LEE
Registered Nurse | Patient Care - Epic EHR - Triage
555-555-0199 | morgan.lee@example.com | Austin, TX

SUMMARY
Registered nurse with 6 years of emergency and critical care experience. Known for calm triage under pressure.

PROFESSIONAL EXPERIENCE
Charge Nurse, Example Medical Center
June 2021 - Present | Austin, TX
- Lead a team of 8 nurses per shift in a 40-bed emergency department,
  coordinating triage and patient flow.
- Document care in Epic and train new hires on charting standards.
Staff Nurse - Sample County Hospital
May 2018 - June 2021 | Round Rock, TX
- Administered medications and monitored vital signs for up to 6 ICU patients.

EDUCATION
Bachelor of Science in Nursing - Example State University
2014 - 2018

LICENSES & CERTIFICATIONS
Registered Nurse, Texas Board of Nursing
BLS and ACLS, American Heart Association

SKILLS
Clinical: Triage, Medication Administration, Patient Education
Systems: Epic, Microsoft Office
"""

NURSE_POSTING = """Emergency Department Registered Nurse

About the role
Join our emergency department caring for patients across a 30-bed unit.

Requirements
- Active Registered Nurse license
- BLS and ACLS certification
- Experience with triage and medication administration
- Epic EHR charting

Preferred
- Critical care experience
- Patient education

Benefits
- Tuition support
"""


@pytest.fixture
def resume_file(tmp_path) -> Path:
    path = tmp_path / "morgan_resume.txt"
    path.write_text(RESUME, encoding="utf-8")
    return path


def _offline(tmp_path, **kw) -> LLM:
    return LLM(Config(offline=True, out_dir=tmp_path / "out", **kw))


# --- profiles -----------------------------------------------------------------------

def test_a_profile_keeps_one_persons_files_together(tmp_path):
    cfg = Config(profile="jane", out_dir=tmp_path)
    assert cfg.master_path == PROFILES_DIR / "jane" / "fact_base.yaml"
    assert cfg.samples_dir == PROFILES_DIR / "jane" / "writing_samples"
    assert cfg.style_path == PROFILES_DIR / "jane" / "style.yaml"
    assert cfg.confirmed_path.name == "fact_base.confirmed.yaml"


def test_explicit_master_beats_profile(tmp_path):
    cfg = Config(profile="jane", master_path=tmp_path / "x.yaml", out_dir=tmp_path)
    assert cfg.master_path == tmp_path / "x.yaml"


def test_the_example_is_never_written_to(tmp_path):
    assert not Config(out_dir=tmp_path).writable_fact_base
    assert Config(master_path=tmp_path / "mine.yaml", out_dir=tmp_path).writable_fact_base


# --- import ---------------------------------------------------------------------------

def test_offline_import_reads_a_non_tech_resume(resume_file, tmp_path):
    result = importer.import_resume(resume_file, _offline(tmp_path))
    m = result.master
    assert result.drafted_by == "offline parser"
    assert [(e["title"], e["company"]) for e in m["experience"]] == [
        ("Charge Nurse", "Example Medical Center"), ("Staff Nurse", "Sample County Hospital")]
    first = m["experience"][0]
    assert first["location"] == "Austin, TX" and first["end"] == "Present"
    assert "coordinating triage and patient flow" in first["bullets"][0]["text"]  # wrapped line
    assert m["contact"]["email"] == "morgan.lee@example.com"
    assert m["contact"]["location"] == "Austin, TX"
    assert any("LICENSES" in s["heading"] for s in m["extra_sections"])
    assert "healthcare" in result.packs and m["lexicon_packs"] == result.packs
    assert result.unverified == []


def _model_draft() -> dict:
    return {
        "contact": {"name": "MORGAN LEE", "phone": "555-555-0199",
                    "email": "morgan@not-in-the-resume.com", "linkedin": "", "location": "Austin, TX"},
        "headlines": [{"text": "Registered Nurse | Triage - Epic", "fits": ["nurse"]},
                      {"text": "Clinical Data | Python - SQL", "fits": ["analyst"]}],
        "summary_facts": ["Registered nurse with 6 years of emergency and critical care experience.",
                          "Recognised as nurse of the year in 2023."],
        "experience": [{
            "id": "example_medical", "title": "Charge Nurse", "company": "Example Medical Center",
            "location": "Austin, TX", "start": "June 2021", "end": "Present",
            "bullets": [
                {"text": "Lead a team of 8 nurses per shift in a 40-bed emergency department, "
                         "coordinating triage and patient flow.",
                 "keywords": ["triage", "team leadership", "Kubernetes"]},
                {"text": "Lead a team of 12 nurses per shift across two hospitals.",
                 "keywords": ["leadership"]},
            ]}],
        "projects": [], "projects_heading": "PROJECTS",
        "education": [{"degree": "Bachelor of Science in Nursing",
                       "school": "Example State University", "dates": "2014 - 2018"}],
        "certifications": [],
        "skills": [{"label": "Clinical", "items": ["Triage", "Medication Administration",
                                                   "Robotic Surgery"]}],
        "extra_sections": [], "do_not_claim": [],
    }


def test_model_import_is_checked_against_the_original(resume_file, tmp_path):
    llm = ScriptedLLM(imported=_model_draft())
    result = importer.import_resume(resume_file, llm)
    m = result.master
    bullets = m["experience"][0]["bullets"]

    # The verbatim bullet stays; the one with an invented number is parked.
    assert [b["text"][:20] for b in bullets] == ["Lead a team of 8 nur"]
    assert any("12 nurses" in u for u in result.unverified)
    assert any("nurse of the year" in u for u in result.unverified)
    assert any("Robotic Surgery" in u for u in result.unverified)
    assert all("12 nurses" not in b["text"] for e in m["experience"] for b in e["bullets"])
    # Hints must be grounded in their bullet.
    assert "triage" in bullets[0]["keywords"]
    assert "Kubernetes" not in bullets[0]["keywords"]
    assert result.dropped_hints >= 1
    # A guessed email is cleared; a headline naming tools the resume lacks is dropped.
    assert m["contact"]["email"] == ""
    assert [h["text"] for h in m["headlines"]] == ["Registered Nurse | Triage - Epic"]
    # The PDF path sends the document itself; a text resume goes in as text.
    assert isinstance(llm.calls[0]["user"], str) and "<resume>" in llm.calls[0]["user"]
    assert llm.calls[0]["schema"] is importer.ImportedResume


def test_pdf_import_sends_the_document(tmp_path):
    fake_pdf = tmp_path / "resume.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 not really")
    llm = ScriptedLLM(imported=_model_draft())
    importer.import_resume(fake_pdf, llm)
    block = llm.calls[0]["user"][0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"


def test_imported_ids_are_unique():
    fb = importer.to_fact_base({"experience": [
        {"id": "acme", "title": "A", "bullets": [{"text": "One.", "keywords": []}]},
        {"id": "acme", "title": "B", "bullets": [{"text": "Two.", "keywords": []}]}]})
    assert [e["id"] for e in fb["experience"]] == ["acme", "acme_2"]
    assert fb["experience"][1]["bullets"][0]["id"] == "acme_2_1"


def test_imported_fact_base_runs_end_to_end(resume_file, tmp_path):
    result = importer.import_resume(resume_file, _offline(tmp_path))
    saved = importer.save(result, tmp_path / "fact_base.yaml", resume_file)
    assert saved.read_text().startswith("# Fact base imported from morgan_resume.txt")

    cfg = Config(master_path=saved, out_dir=tmp_path / "out", offline=True, max_revisions=1)
    final = run(PipelineState(job_text=NURSE_POSTING, max_revisions=1), cfg, LLM(cfg))
    assert final.pdf_path
    text = Path(final.pdf_path).with_suffix(".txt").read_text()
    assert "Charge Nurse" in text and "LICENSES & CERTIFICATIONS" in text
    # The healthcare pack was loaded from the fact base, so its terms were found.
    terms = {k.term for k in final.keywords}
    assert {"Triage", "BLS", "ACLS", "Registered Nurse"} <= terms
    assert any("lexicon packs healthcare" in line for line in final.log)


def test_cli_import_refuses_to_overwrite(resume_file, tmp_path, capsys):
    dest = tmp_path / "fb.yaml"
    args = ["import", str(resume_file), "--out", str(dest), "--offline", "--no-color"]
    assert cli.main(args) == 0 and dest.exists()
    assert cli.main(args) == 1
    assert "already exists" in capsys.readouterr().out
    assert cli.main([*args, "--force"]) == 0


# --- lexicon packs ---------------------------------------------------------------------

def test_packs_load_and_reset():
    assert "Triage" not in lexicon.find_terms("Experienced in triage.")
    assert lexicon.load_pack("healthcare") > 0
    assert "Triage" in lexicon.find_terms("Experienced in triage.")
    lexicon.reset()
    assert "Triage" not in lexicon.find_terms("Experienced in triage.")


def test_packs_never_override_the_core():
    before = dict(lexicon.LEXICON["SQL"])
    lexicon.load_terms([{"term": "SQL", "category": "soft", "aliases": ["x"]}])
    assert lexicon.LEXICON["SQL"] == before


def test_unknown_pack_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown lexicon pack"):
        lexicon.load_pack("astrology")


def test_learned_terms_are_saved_and_reloaded(tmp_path):
    from resume_tailor.state import Keyword

    path = tmp_path / "lexicon.yaml"
    kws = [Keyword(term="Guidewire", normalized="guidewire", category="tool",
                   aliases=["guidewire policycenter"]),
           Keyword(term="Python", normalized="python"),
           Keyword(term="Grit", normalized="grit", category="soft")]
    assert lexicon.remember(path, kws) == 1             # Python known, soft skipped
    assert lexicon.remember(path, kws) == 0             # no duplicates
    assert lexicon.load_learned(path) == 1
    assert "Guidewire" in lexicon.find_terms("Configured Guidewire PolicyCenter rules.")


# --- headings and extra sections --------------------------------------------------------

@pytest.fixture(scope="module")
def master():
    return resume_source.load(None, Config().master_path)


def test_projects_heading_comes_from_the_fact_base(master):
    resume = tailor_node._offline_tailor(master, [])
    assert any(s.heading == "AI & ANALYTICS PROJECTS" for s in resume.sections)
    plain = {k: v for k, v in master.items() if k != "projects_heading"}
    resume = tailor_node._offline_tailor(plain, [])
    assert any(s.heading == "PROJECTS" for s in resume.sections)


def test_extra_sections_render_as_written(master, tmp_path):
    m = copy.deepcopy(master)
    m["extra_sections"] = [{"heading": "Volunteering", "entries": [
        {"title": "Data Volunteer", "subtitle": "Portland Food Bank", "dates": "2022 - Present",
         "bullets": ["Built the weekly donations dashboard."]}]}]
    resume = tailor_node._offline_tailor(m, [])
    other = next(s for s in resume.sections if s.kind == "other")
    assert other.heading == "VOLUNTEERING"
    path, _ = pdf.render(resume, m, tmp_path / "r.pdf")
    text = path.with_suffix(".txt").read_text()
    assert "VOLUNTEERING" in text and "Data Volunteer, Portland Food Bank" in text
    assert "- Built the weekly donations dashboard." in text


def test_confirmed_facts_merge_into_a_copy(master):
    data = {"confirmed": [{"id": "confirmed_ci_cd", "entry_id": "agents", "term": "CI/CD",
                           "text": "Set up GitHub Actions to run the test suite on every push."}],
            "declined": [{"term": "Kubernetes"}]}
    data["confirmed"].append({"id": "confirmed_looker", "entry_id": "skills", "term": "Looker",
                              "text": "Looker"})
    merged = resume_source.apply_confirmed(master, data)
    agents = next(e for e in merged["projects"] if e["id"] == "agents")
    assert agents["bullets"][-1]["keywords"] == ["CI/CD"]
    assert merged["skills"]["interview"] == {"label": "Additional Skills", "items": ["Looker"]}
    assert "interview" not in master["skills"]
    assert any("Kubernetes" in line for line in merged["do_not_claim"])
    assert len(next(e for e in master["projects"] if e["id"] == "agents")["bullets"]) == 5


def test_unverified_items_are_never_evidence(master):
    m = copy.deepcopy(master)
    m["unverified"] = [{"where": "northwind", "text": "Ran Kubernetes clusters for 200 services."}]
    assert "Kubernetes" not in tailor_node.master_text(m)
    resume = tailor_node._offline_tailor(master, [])
    resume.summary = "Analyst who ran Kubernetes clusters."
    assert any("Kubernetes" in w for w in tailor_node.audit(resume, m))


def test_copy_edits_never_touch_extra_sections(master):
    from resume_tailor.nodes import grammar

    m = copy.deepcopy(master)
    m["extra_sections"] = [{"heading": "Awards", "entries": [
        {"title": "Analyst of the Year", "subtitle": "", "dates": "2023",
         "bullets": ["voted by peers"]}]}]
    resume = tailor_node._offline_tailor(m, [])
    units = grammar._units(resume)
    assert "voted by peers" not in units.values()
    grammar._mechanical_fixes(resume)
    other = next(s for s in resume.sections if s.kind == "other")
    assert other.entries[0]["bullets"][0]["text"] == "voted by peers"

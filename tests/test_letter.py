"""Cover letters: learning a voice, the checker, and the write-check-revise loop."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import ScriptedLLM, allstate_jd, allstate_payload  # noqa: E402

from resume_tailor.config import EXAMPLE_MASTER, Config  # noqa: E402
from resume_tailor.graph import run  # noqa: E402
from resume_tailor.letter import style  # noqa: E402
from resume_tailor.llm import LLM  # noqa: E402
from resume_tailor.nodes import letter as letter_node  # noqa: E402
from resume_tailor.render import letter_pdf  # noqa: E402
from resume_tailor.sources import resume_source  # noqa: E402
from resume_tailor.state import (  # noqa: E402
    CoverLetter,
    GapItem,
    JobPosting,
    LetterClaim,
    LetterParagraph,
    PipelineState,
)

# Two letters the fictional Jordan Rivera "wrote": short sentences, contractions,
# no exclamation marks -- and old employers that must never leak into a new letter.
SAMPLE_1 = """Jordan Rivera
Portland, OR

Dear Hiring Manager,

I'm applying for the Senior Analyst role at Bluefin Logistics. I've spent six years turning messy data into decisions people act on. That's the work I want to keep doing.

At my current job I own reporting from the first requirement to sign-off. I write the SQL, I build the dashboards, and I sit with the people who use them. When a number looks wrong, I'm the one who finds out why.

I'd like to bring that to Bluefin. Thanks for reading. I'd be glad to talk.

Best,
Jordan Rivera
"""

SAMPLE_2 = """Dear Hiring Team,

I'm writing about the Reporting Lead opening at Harborview Foods. My work sits where the business question meets the data. I like it there.

Most of what I do is translation. A manager asks why sales dipped. I find the three tables that know, I check them against each other, and I bring back an answer they can use. It isn't glamorous, but it's useful.

Harborview's push into new channels needs that kind of reporting. I'd welcome a conversation.

Best,
Jordan
"""


@pytest.fixture
def home(tmp_path) -> Path:
    d = tmp_path / "home"
    (d / "writing_samples").mkdir(parents=True)
    (d / "writing_samples" / "bluefin.txt").write_text(SAMPLE_1)
    (d / "writing_samples" / "harborview.txt").write_text(SAMPLE_2)
    return d


@pytest.fixture
def master():
    return resume_source.load(None, EXAMPLE_MASTER)


def _cfg(tmp_path, home, **kw) -> Config:
    return Config(out_dir=tmp_path / "out", offline=True, home_dir=home, **kw)


# --- learning a voice -------------------------------------------------------------------

def test_body_drops_address_greeting_and_sign_off():
    b = style.body(SAMPLE_1)
    assert b.startswith("I'm applying") and b.rstrip().endswith("I'd be glad to talk.")


def test_measure_reads_the_style():
    stats = style.measure(SAMPLE_1)
    assert stats["contractions_per_100_words"] > 3
    assert 5 <= stats["avg_sentence_words"] <= 14
    assert stats["exclamations_per_1000_words"] == 0
    assert 70 <= stats["words"] <= 120


def test_samples_and_an_offline_profile(tmp_path):
    letters = []
    for name, text in (("a.txt", SAMPLE_1), ("b.md", SAMPLE_2)):
        p = tmp_path / name
        p.write_text(text)
        letters.append(p)
    saved = style.add_samples(letters, tmp_path / "samples")
    assert [p.name for p in saved] == ["a.txt", "b.txt"]
    profile = style.build_profile(style.load_samples(tmp_path / "samples"),
                                  LLM(Config(offline=True, out_dir=tmp_path)))
    assert profile["voice"] == {} and profile["stats"]["words"] > 60
    style.save_profile(profile, tmp_path / "style.yaml")
    assert style.load_profile(tmp_path / "style.yaml") == profile
    assert "measured: sentences average" in style.describe(profile)


def test_too_short_to_be_a_letter(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("Thanks for the chat today.")
    with pytest.raises(ValueError, match="under 60 words"):
        style.add_samples([p], tmp_path / "samples")


def test_signature_phrases_must_really_be_yours():
    voice = {"tone": "plain", "formality": "conversational", "opening": "role first",
             "closing": "short", "self_presentation": "modest", "structure": "3 short paragraphs",
             "signature_phrases": ["glad to talk", "synergy-driven leader"], "avoids": [],
             "notes": ""}
    profile = style.build_profile([("a", SAMPLE_1)], ScriptedLLM(voice=voice))
    assert profile["voice"]["signature_phrases"] == ["glad to talk"]


def test_distance_flags_a_draft_that_does_not_sound_like_you():
    stats = style.average([style.measure(SAMPLE_1), style.measure(SAMPLE_2)])
    stiff = ("Dear Sir or Madam, I am writing in order to formally express my considerable "
             "interest in the position that has been advertised by your organisation, and I "
             "would respectfully submit that my qualifications and extensive experience render "
             "me exceptionally well suited to the requirements of this demanding role! "
             "Furthermore, I do not hesitate to state that my analytical capabilities are "
             "considerable and have been demonstrated repeatedly over a number of years.")
    rules = {rule for rule, _ in style.distance(stiff, stats)}
    assert {"style-sentence-length", "style-contractions", "style-exclamations"} <= rules


# --- the checker ----------------------------------------------------------------------------

POSTING = JobPosting(title="Data Analyst", company="Acme Analytics",
                     raw_text="Data Analyst at Acme Analytics. Build Tableau dashboards and "
                              "own reporting for our retail clients. Kubernetes a plus.")


def _letter(*paragraphs: tuple[str, list[tuple[str, str]]]) -> CoverLetter:
    return CoverLetter(greeting="Dear Hiring Manager,", sign_off="Best,", paragraphs=[
        LetterParagraph(text=t, claims=[LetterClaim(source_id=s, claim=c) for s, c in claims])
        for t, claims in paragraphs])


def _check(master, letter, cfg, gaps=()) -> set[str]:
    state = PipelineState(posting=POSTING, master=master, letter=letter, gaps=list(gaps))
    return {i.rule for i in letter_node.check(state, cfg)["letter_issues"]}


def test_checker_catches_invented_numbers_and_fact_ids(master, tmp_path, home):
    letter = _letter(("At Acme Analytics I'd cut load time by 90% for 40+ users.",
                      [("harbor_perf", "cut load time"), ("made_up_id", "something")]))
    rules = _check(master, letter, _cfg(tmp_path, home))
    assert {"number-not-in-facts", "unknown-fact"} <= rules


def test_checker_catches_copying_and_old_employers(master, tmp_path, home):
    letter = _letter(("I've spent six years turning messy data into decisions people act on, "
                      "much as I did for Bluefin.", []))
    rules = _check(master, letter, _cfg(tmp_path, home))
    assert {"copied-from-sample", "old-letter-detail"} <= rules


def test_stock_phrases_are_flagged_only_if_you_never_use_them(master, tmp_path, home):
    letter = _letter(("I'm passionate about reporting at Acme Analytics.", []))
    assert "stock-phrase" in _check(master, letter, _cfg(tmp_path, home))
    (home / "writing_samples" / "third.txt").write_text(
        SAMPLE_2.replace("I like it there.", "I'm passionate about it."))
    assert "stock-phrase" not in _check(master, letter, _cfg(tmp_path, home))


def test_checker_flags_claims_about_gaps(master, tmp_path, home):
    letter = _letter(("I've run Kubernetes clusters at Acme Analytics scale.",
                      [("northwind_sql", "Ran Kubernetes clusters")]))
    gaps = [GapItem(term="Kubernetes", disposition="gap")]
    assert "claim-overreach" in _check(master, letter, _cfg(tmp_path, home), gaps)


def test_checker_flags_placeholders_and_a_missing_company(master, tmp_path, home):
    letter = _letter(("I'd love to join [Company] as an analyst.", []))
    rules = _check(master, letter, _cfg(tmp_path, home))
    assert {"placeholder", "company-missing"} <= rules


GOOD = [
    ("I'm applying for the Data Analyst role at Acme Analytics. Reporting is the work I do "
     "best, and your retail clients are the kind of people I build it for.",
     [("northwind_reqs", "turns requirements into reporting")]),
    ("At Harbor Beverage Distributors I built Tableau dashboards that 15+ regions relied on. "
     "One of them was slow, so I rebuilt it at the data source and cut load time by 75% for "
     "40+ users. I check numbers before anyone else has to.",
     [("harbor_share", "Tableau dashboards for 15+ regions"),
      ("harbor_perf", "cut load time by 75% for 40+ users")]),
    ("Today I write SQL and MDX to validate large reporting systems, and I own the test cycle "
     "through sign-off. That habit of proving a number is right is what I'd bring to Acme "
     "Analytics. I'd be glad to talk.",
     [("northwind_sql", "validates reporting systems"), ("northwind_pilot", "owns UAT")]),
]


def test_a_faithful_letter_has_no_errors(master, tmp_path, home):
    state = PipelineState(posting=POSTING, master=master, letter=_letter(*GOOD))
    issues = letter_node.check(state, _cfg(tmp_path, home))["letter_issues"]
    assert not [i for i in issues if i.severity == "error"], issues


# --- the loop, end to end -------------------------------------------------------------------

def _letter_payload(paragraphs) -> dict:
    return {"greeting": "Dear Hiring Manager,", "sign_off": "Best,", "gaps_addressed": [],
            "paragraphs": [{"text": t, "claims": [{"source_id": s, "claim": c} for s, c in cl]}
                           for t, cl in paragraphs]}


def test_letter_loop_revises_then_renders(tmp_path, home):
    bad = _letter_payload([("I cut costs by 99% at Bluefin.", [("nope", "cost")])])
    good = _letter_payload(GOOD)
    llm = ScriptedLLM([allstate_payload()], letters=[bad, good])
    cfg = _cfg(tmp_path, home, max_revisions=0)
    final = run(PipelineState(job_text=allstate_jd(), max_revisions=0, want_letter=True),
                cfg, llm)

    assert final.letter_revisions >= 1
    assert final.letter_path and Path(final.letter_path).exists()
    assert Path(final.letter_path).read_bytes().startswith(b"%PDF")
    text = Path(final.letter_path).with_suffix(".txt").read_text()
    assert "Dear Hiring Manager," in text and text.rstrip().endswith("Jordan Rivera")
    assert final.letter_path.endswith("_Cover_Letter.pdf")

    letter_calls = [c for c in llm.calls if "write cover letters" in c["system"]]
    assert "WRITING SAMPLES" in letter_calls[0]["cached"]
    assert "Bluefin" in letter_calls[0]["cached"]            # the samples, verbatim
    assert "[harbor_perf]" in letter_calls[0]["cached"]      # the fact list
    assert letter_calls[0]["effort"] == "high"
    revision = letter_calls[1]["user"]
    assert "# REVISION" in revision and "number-not-in-facts" in revision


def test_offline_skips_the_letter(tmp_path, home):
    cfg = _cfg(tmp_path, home, max_revisions=0)
    final = run(PipelineState(job_text=allstate_jd(), max_revisions=0, want_letter=True),
                cfg, LLM(cfg))
    assert final.pdf_path and final.letter is None and final.letter_path is None
    assert any(line.startswith("letter: skipped") for line in final.log)


def test_both_executors_write_the_same_letter(tmp_path, home):
    pytest.importorskip("langgraph")
    from resume_tailor.graph import _run_langgraph, _run_simple, build_nodes

    texts = []
    for runner in (_run_langgraph, _run_simple):
        cfg = _cfg(tmp_path / runner.__name__, home, max_revisions=0)
        llm = ScriptedLLM([allstate_payload()], letters=[_letter_payload(GOOD)])
        final = runner(PipelineState(job_text=allstate_jd(), max_revisions=0,
                                     want_letter=True), build_nodes(cfg, llm), None)
        texts.append((final.letter.all_text(), Path(final.letter_path).name))
    assert texts[0] == texts[1]


def test_letter_pdf_and_text_copy(master, tmp_path):
    out = letter_pdf.render(_letter(*GOOD), master, tmp_path / "letter.pdf", "Jordan Rivera")
    assert out.read_bytes().startswith(b"%PDF")
    text = out.with_suffix(".txt").read_text()
    assert text.startswith("Jordan Rivera\n555-555-0142")
    assert "Best,\nJordan Rivera" in text


def test_style_add_takes_a_folder_or_a_name_without_extension(tmp_path):
    folder = tmp_path / "Cover letter"
    folder.mkdir()
    (folder / "bluefin.txt").write_text(SAMPLE_1)
    (folder / "harborview.md").write_text(SAMPLE_2)
    (folder / "notes.jpg").write_bytes(b"not a letter")
    saved = style.add_samples([folder], tmp_path / "samples")
    assert sorted(p.name for p in saved) == ["bluefin.txt", "harborview.txt"]
    # Windows hides extensions, so "My letter" should find "My letter.txt".
    (tmp_path / "My letter.txt").write_text(SAMPLE_1)
    assert style.resolve_letters([tmp_path / "My letter"]) == [tmp_path / "My letter.txt"]
    with pytest.raises(FileNotFoundError, match="no such file"):
        style.resolve_letters([tmp_path / "missing"])


def test_style_add_reports_problems_without_a_traceback(tmp_path, capsys):
    from resume_tailor import cli

    empty = tmp_path / "empty folder"
    empty.mkdir()
    assert cli.main(["style", "add", str(empty)]) == 1
    assert "no PDF, DOCX, TXT or MD files" in capsys.readouterr().err

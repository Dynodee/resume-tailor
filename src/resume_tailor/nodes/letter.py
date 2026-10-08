"""Nodes 8-10 -- write a cover letter in the user's voice, check it, revise it.

    resume-tailor <url> --letter

    ... render -> letter -> letter_check -+-> render_letter -> END
                    ^                     |
                    +--- revise_letter <--+

The letter is held to the resume's rule: every claim about the candidate comes
from the fact base. It also carries a job the resume cannot: the gap analysis
says which requirements are only related experience or true gaps, and the
letter is where those get addressed honestly instead of being worked onto the
page.

The voice comes from the user's own letters (see letter/style.py). The samples
go into the prompt verbatim, with the measured and described style, as a cached
prefix -- the same for every posting and every revision, so it is paid for at
full price once.

The checker is deterministic, like the resume linter, and its findings drive
the revision loop:

- claims cite fact ids that exist; numbers appear in the fact base or posting
- no sentence copied from a sample, and no employer or detail carried over
  from an old letter
- the draft's measured style is close to the user's (sentence length,
  contractions, exclamation marks, sentences starting with "I")
- length near the user's usual; no stock phrases the user never uses; the
  company is named; no unfilled placeholders
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from ..ats import lexicon, phrases as phrase_mod
from ..config import Config
from ..letter import style
from ..llm import LLM
from ..render import letter_pdf
from ..state import CoverLetter, GrammarIssue, PipelineState, Priority
from .tailor import _STOP, _role_brief, master_text

SYSTEM = """You write cover letters in the candidate's own voice.

VOICE. The writing samples are letters the candidate wrote. Match how they
write: sentence length and rhythm, formality, how they open and close, how they
talk about their own work, the words they reach for and the ones they avoid.
Write the letter they would write for this job. Borrow the voice only: never
reuse a sentence from a sample, and never mention an employer, role or detail
from a sample unless the fact list contains it too.

FACTS. Every claim about the candidate comes from the FACT LIST. In "claims",
list each fact a paragraph relies on, by id, with the claim it supports. Never
add a number, tool, employer, credential or responsibility the fact list does
not contain, and copy numbers exactly.

REQUIREMENTS. Use the requirement notes to decide what to cover:
- shown on the resume, reframed, or confirmed: evidence the candidate has it.
  Pick the two or three that matter most to this employer and back them with
  facts. Complement the resume; do not restate it line by line.
- related experience: name the connection honestly -- what they have done that
  transfers, and that it is related rather than the same thing.
- gaps and unanswered questions: never claim them. Address at most one, only if
  it is central to the role, briefly and in the candidate's voice -- what they
  bring instead or how they are closing it. No apologising.

SHAPE. A greeting line; three or four short paragraphs; a closing line such as
"Sincerely," (the name is added separately). Name the company and the role. No
headings, bullet points or placeholders such as [Company]. If no hiring
manager is named, greet the company by the name in THE ROLE heading -- "<word
the samples use> <Company> team," (for example "Hey Google team,") -- never a
sub-team, program or department name from the posting body.
gaps_addressed: the requirement terms the letter speaks to that are related
experience or gaps."""

CACHED = """# WRITING SAMPLES -- the candidate's own letters (voice only, never content)
{samples}

# HOW THEY WRITE
{voice}

# FACT LIST
{facts}

# CANDIDATE
{name}"""

USER = """# THE ROLE
{title} at {company}

{role_brief}

# REQUIREMENT NOTES
{notes}

# THE TAILORED RESUME SENT WITH THIS LETTER
{resume}

# LENGTH
About {words} words across the paragraphs.
{feedback}"""


class _Claim(BaseModel):
    source_id: str
    claim: str


class _Paragraph(BaseModel):
    text: str
    claims: list[_Claim]


class _LetterOut(BaseModel):
    greeting: str
    paragraphs: list[_Paragraph]
    sign_off: str
    gaps_addressed: list[str]


# Stock phrases flagged only when the user's own letters never use them -- if
# they do, it is their voice and stays.
CLICHES = (
    "i am writing to express my interest", "i am excited to apply", "perfect fit",
    "great fit", "i believe i would be", "hit the ground running", "think outside the box",
    "team player", "results-driven", "detail-oriented", "go-getter", "synergy",
    "passionate about", "i am confident that", "thrilled", "dynamic environment",
    "fast-paced environment", "esteemed", "leverage my skills",
)
_PLACEHOLDER = re.compile(r"\[[^\]]{2,40}\]|\{[^}]{2,40}\}|<[A-Za-z ]{2,40}>|\bXX+\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


# --- inputs the letter draws on ---------------------------------------------------------

def fact_list(master: dict) -> dict[str, str]:
    """Every fact the letter may cite, by id."""
    facts: dict[str, str] = {}
    for i, f in enumerate(master.get("summary_facts", []) or [], 1):
        facts[f"fact{i}"] = str(f)
    for group in ("experience", "projects"):
        for e in master.get(group, []) or []:
            label = ", ".join(x for x in (e.get("title") or e.get("name", ""),
                                          e.get("company") or e.get("subtitle", ""),
                                          f"{e.get('start', '')} - {e.get('end', '')}".strip(" -")
                                          or e.get("year", "")) if x)
            for b in e.get("bullets", []):
                facts[b["id"]] = f"{b['text']} ({label})"
    for i, e in enumerate(master.get("education", []) or [], 1):
        facts[f"edu{i}"] = ", ".join(x for x in (e.get("degree", ""), e.get("school", ""),
                                                 e.get("dates", "")) if x)
    for i, c in enumerate(master.get("certifications", []) or [], 1):
        facts[f"cert{i}"] = ", ".join(x for x in (c.get("name", ""), c.get("issuer", ""),
                                                  c.get("date", "")) if x)
    for si, s in enumerate(master.get("extra_sections", []) or []):
        for ei, e in enumerate(s.get("entries") or []):
            facts[f"extra{si}_{ei}"] = f"{s.get('heading', '')}: " + ", ".join(
                x for x in (e.get("title", ""), e.get("subtitle", ""), e.get("dates", "")) if x)
    items = [i for g in (master.get("skills") or {}).values() for i in g.get("items", [])]
    if items:
        facts["skills"] = "Skills: " + ", ".join(items)
    return facts


def requirement_notes(state: PipelineState) -> str:
    lines = []
    cov = state.coverage
    if cov:
        shown = [i.keyword.term for i in cov.items if i.covered and i.keyword.category != "soft"
                 and i.keyword.priority is not Priority.MENTIONED]
        if shown:
            lines.append("Shown on the resume: " + ", ".join(shown))
    for disposition, label in (("reframe", "Reframed (the cited facts show it)"),
                               ("confirmed", "Confirmed by the candidate"),
                               ("adjacent", "Related experience only -- say so honestly")):
        group = [g for g in state.gaps if g.disposition == disposition]
        for g in group:
            lines.append(f"{label}: {g.term} -- facts {', '.join(g.source_ids) or '(none)'}"
                         + (f"; {g.rationale}" if g.rationale else ""))
    gaps = [g.term for g in state.gaps if g.disposition in ("gap", "declined")]
    if gaps:
        lines.append("Gaps -- never claim: " + ", ".join(gaps))
    asks = [g.term for g in state.gaps if g.disposition == "ask"]
    if asks:
        lines.append("Unanswered (treat as gaps): " + ", ".join(asks))
    return "\n".join(lines) or "(no requirement notes)"


def _resume_digest(state: PipelineState) -> str:
    resume = state.resume
    if resume is None:
        return "(no resume)"
    parts = [resume.headline, resume.summary]
    for section in resume.sections:
        if section.kind in ("experience", "projects"):
            for e in section.entries:
                parts.append(f"{e.get('title') or e.get('name', '')}, "
                             f"{e.get('company') or e.get('subtitle', '')}")
                parts.extend(f"- {b['text']}" for b in e.get("bullets", []))
    return "\n".join(p for p in parts if p)


def _target_words(profile: dict) -> int:
    words = (profile.get("stats") or {}).get("words") or 0
    return int(min(max(words, 200), 450)) if words else 320


def display_name(master: dict) -> str:
    name = str((master.get("contact") or {}).get("name", "")).strip()
    return name.title() if name.isupper() else name


# --- the writer --------------------------------------------------------------------------

def write(state: PipelineState, cfg: Config, llm: LLM) -> dict:
    log = list(state.log)
    if llm.offline:
        log.append("letter: skipped -- writing in your voice needs ANTHROPIC_API_KEY")
        return {"letter": None, "log": log}
    samples = style.load_samples(cfg.samples_dir)
    profile = style.load_profile(cfg.style_path)
    if samples and not profile.get("stats"):
        profile = {**profile, "stats": style.average([style.measure(t) for _, t in samples])}
    posting = state.posting
    facts = fact_list(state.master)

    cached = CACHED.format(
        samples="\n\n".join(f'<letter name="{n}">\n{t[:6000]}\n</letter>' for n, t in samples[:3])
                or "(none provided -- write plainly and professionally)",
        voice=style.describe(profile),
        facts="\n".join(f"[{k}] {v}" for k, v in facts.items()),
        name=display_name(state.master) or "(unknown)",
    )
    payload = llm.complete_json(
        SYSTEM,
        USER.format(
            title=posting.title or "(not stated)", company=posting.company or "(not stated)",
            role_brief=_role_brief(posting, limit=4000), notes=requirement_notes(state),
            resume=_resume_digest(state), words=_target_words(profile),
            feedback=_feedback(state)),
        schema=_LetterOut, max_tokens=12000, effort="high", cached=cached,
    ) or {}
    letter = CoverLetter.model_validate({
        "greeting": payload.get("greeting", ""),
        "paragraphs": payload.get("paragraphs", []),
        "sign_off": payload.get("sign_off", ""),
        "gaps_addressed": payload.get("gaps_addressed", []),
    })
    words = len(letter.body().split())
    log.append(f"letter: draft {state.letter_revisions} -- {len(letter.paragraphs)} "
               f"paragraph(s), {words} words"
               + ("" if samples else " (no writing samples -- add some with "
                                    "`resume-tailor style add`)"))
    return {"letter": letter, "log": log}


def _feedback(state: PipelineState) -> str:
    if state.letter_revisions == 0 or state.letter is None:
        return ""
    lines = ["", "# REVISION",
             "Your previous draft is below, with the problems a checker found. Fix every "
             "problem and keep what works.", "", state.letter.all_text(), "", "Problems:"]
    for i in state.letter_issues[:15]:
        lines.append(f"- [{i.rule}] {i.message}" + (f' ("{i.excerpt}")' if i.excerpt else ""))
    return "\n".join(lines)


# --- the checker ----------------------------------------------------------------------------

def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def _shingles(words: list[str], n: int = 8) -> dict[tuple, int]:
    return {tuple(words[i:i + n]): i for i in range(len(words) - n + 1)}


_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][A-Za-z0-9&]+(?:\s+[A-Z][A-Za-z0-9&]+)*|"
                   r"[A-Z]{2,})\b", re.MULTILINE)


def _names(text: str) -> set[str]:
    """Capitalised words and runs mid-sentence -- the likely proper nouns."""
    out = set()
    for m in _NAME.finditer(text):
        for word in m.group(1).split():
            if word not in _STOP and len(word) >= 3:
                out.add(word)
    return out


def check(state: PipelineState, cfg: Config) -> dict:
    letter = state.letter
    log = list(state.log)
    if letter is None:
        return {"letter_issues": []}
    issues: list[GrammarIssue] = []

    def add(rule: str, message: str, severity: str = "warning", excerpt: str = "") -> None:
        issues.append(GrammarIssue(location="letter", rule=rule, severity=severity,
                                   message=message, excerpt=excerpt[:160]))

    body = letter.body()
    posting = state.posting
    company = getattr(posting, "company", "") or ""
    title = getattr(posting, "title", "") or ""
    reference = "\n".join([master_text(state.master), getattr(posting, "raw_text", "") or "",
                           company, title, display_name(state.master)])
    ref_low = reference.lower()
    samples = style.load_samples(cfg.samples_dir)
    sample_bodies = [style.body(t) for _, t in samples]
    sample_low = " ".join(sample_bodies).lower().replace("’", "'")
    facts = fact_list(state.master)

    if not letter.paragraphs or not body.strip():
        add("empty", "The letter has no paragraphs.", "error")
        return {"letter_issues": issues, "log": log + ["letter check: empty draft"]}

    for p in letter.paragraphs:
        for c in p.claims:
            if c.source_id not in facts:
                add("unknown-fact", f"cites fact '{c.source_id}', which does not exist", "error",
                    c.claim)

    known_numbers = set(_NUMBER.findall(reference.replace(",", "")))
    for n in dict.fromkeys(_NUMBER.findall(body.replace(",", ""))):
        if n not in known_numbers:
            add("number-not-in-facts", f"the number {n} is not in your fact base or the "
                "posting -- use the exact figures from your facts", "error",
                _around(body, n))

    # Details carried over from an old letter: names in a sample that neither
    # the fact base nor the posting mention.
    stale = {w for w in set().union(*(_names(b) for b in sample_bodies))
             if w.lower() not in ref_low} if sample_bodies else set()
    letter_names = _names(body)
    for w in sorted(letter_names & stale):
        add("old-letter-detail", f"mentions '{w}' from one of your old letters", "error",
            _around(body, w))
    for w in sorted(letter_names - stale):
        stems = {w.lower(), w.lower()[:-1], w.lower()[:-2]} if len(w) > 5 else {w.lower()}
        if not any(st in ref_low for st in stems if len(st) >= 3):
            add("fabrication-check", f"'{w}' is in the letter but not in your fact base or "
                "the posting -- check it before sending", "warning", _around(body, w))

    words = _norm_words(body)
    sample_shingles: dict[tuple, int] = {}
    for b in sample_bodies:
        sample_shingles.update(_shingles(_norm_words(b)))
    copied = [" ".join(s) for s in _shingles(words) if s in sample_shingles]
    if copied:
        add("copied-from-sample", "repeats a passage from one of your old letters word for "
            "word -- say it fresh", "error", copied[0])

    stats = (style.load_profile(cfg.style_path).get("stats")
             or (style.average([style.measure(t) for _, t in samples]) if samples else {}))
    for rule, message in style.distance(body, stats):
        add(rule, message, "warning")

    target = _target_words({"stats": stats})
    if not 0.7 * target <= len(words) <= 1.3 * target:
        add("length", f"{len(words)} words; aim for about {target}", "warning")

    low = body.lower().replace("’", "'")
    for phrase in CLICHES:
        if phrase in low and phrase not in sample_low:
            add("stock-phrase", f"'{phrase}' is a stock phrase your own letters never use",
                "warning", _around(body, phrase))

    if company and not phrase_mod.literal_in(company, letter.all_text()):
        add("company-missing", f"does not name the company ({company})", "warning")
    if m := _PLACEHOLDER.search(letter.all_text()):
        add("placeholder", "contains an unfilled placeholder", "error", m.group(0))

    # A claim about a related-only or missing skill must rest on a fact that
    # actually mentions it.
    weak = [g for g in state.gaps if g.disposition in ("adjacent", "gap", "declined", "ask")]
    for p in letter.paragraphs:
        for c in p.claims:
            fact = facts.get(c.source_id, "")
            for g in weak:
                forms = [g.term, *lexicon.surface_forms(g.term)]
                if any(phrase_mod.literal_in(f, c.claim) for f in forms) and not any(
                        phrase_mod.literal_in(f, fact) for f in forms):
                    add("claim-overreach", f"claims '{g.term}' on the strength of "
                        f"{c.source_id}, which does not mention it", "error", c.claim)

    errors = sum(1 for i in issues if i.severity == "error")
    log.append(f"letter check: {len(issues)} issue(s)" + (f", {errors} error(s)" if errors else ""))
    return {"letter_issues": issues, "log": log}


def _around(text: str, needle: str, width: int = 70) -> str:
    pos = text.lower().find(needle.lower())
    if pos < 0:
        return ""
    start = max(0, pos - width // 2)
    return text[start:start + width].replace("\n", " ")


# --- routing and rendering --------------------------------------------------------------------

def should_revise(state: PipelineState) -> str:
    if state.letter is None or state.letter_revisions >= state.letter_max_revisions:
        return "render_letter"
    if any(i.severity in ("error", "warning") for i in state.letter_issues):
        return "revise_letter"
    return "render_letter"


def bump_revision(state: PipelineState) -> dict:
    log = list(state.log)
    rules = sorted({i.rule for i in state.letter_issues if i.severity in ("error", "warning")})
    log.append(f"revise letter: pass {state.letter_revisions + 1} because of "
               + ", ".join(rules))
    return {"letter_revisions": state.letter_revisions + 1, "log": log}


def render(state: PipelineState, cfg: Config) -> dict:
    if state.letter is None:
        return {}
    log = list(state.log)
    resume_pdf = Path(state.pdf_path) if state.pdf_path else cfg.out_dir / "Resume.pdf"
    out_path = resume_pdf.with_name(resume_pdf.stem + "_Cover_Letter.pdf")
    path = letter_pdf.render(state.letter, state.master, out_path, display_name(state.master))
    log.append(f"render letter: {path.name}, text copy at {path.with_suffix('.txt').name}")
    return {"letter_path": str(path), "log": log}

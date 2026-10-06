"""Exact-phrase coverage: the posting's own words, matched literally.

Keyword coverage (scorer.py) is deliberately generous -- aliases mean "dbt"
satisfies "data build tool". That is the right question for "is this skill on
the page?", and the wrong one for a recruiter typing "Data Build Tool" into the
applicant tracking system's search box. Many systems match those words, not
the meaning, so a resume can score well on keywords and still not come up.

This module asks the stricter question. It collects phrases exactly as the
posting writes them -- every item in a skills tag list (Workday, LinkedIn), plus
the posting's own wording of each required or preferred skill -- and checks
each one literally against the page.

"Literally" is defined to match how search boxes tokenise, and no looser:
case-insensitive, hyphens and slashes treated as spaces, and an optional plural
"s" on the last word. "Structured Query Language (SQL)" is satisfied by either
the long form or the acronym, since a search on the tag would accept both.
"""

from __future__ import annotations

import re

from ..state import (
    CoverageReport,
    ExactPhrase,
    JobPosting,
    Keyword,
    PhraseItem,
    Priority,
    TailoredResume,
)
from . import lexicon

# A skills list item is a short label, not a sentence.
_MAX_TAG_WORDS = 6
_SPLIT = re.compile(r"\s*(?:,|;|\n|•|\s-\s|^\s*-\s*)\s*", re.MULTILINE)


# --- literal matching -------------------------------------------------------------

def _norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[-/_]", " ", text)
    text = re.sub(r"[^a-z0-9+#&.' ]+", " ", text)
    text = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", text)   # drop sentence dots
    return re.sub(r"\s+", " ", text).strip()


def alternatives(phrase: str) -> list[str]:
    """Literal forms that count as the phrase.

    "Structured Query Language (SQL) Development" ->
        ["Structured Query Language (SQL) Development",
         "Structured Query Language Development", "SQL Development"]
    """
    out = [phrase]
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*(.*)$", phrase)
    if m:
        before, inner, after = (x.strip() for x in m.groups())
        for form in (f"{before} {after}", f"{inner} {after}"):
            form = form.strip()
            if form and form not in out:
                out.append(form)
    return out


def _literal_pattern(phrase: str) -> re.Pattern[str] | None:
    tokens = _norm(phrase).split()
    if not tokens:
        return None
    body = r"\s+".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<![a-z0-9]){body}(?:s|es)?(?![a-z0-9])")


def literal_in(phrase: str, text: str) -> bool:
    haystack = _norm(text)
    for form in alternatives(phrase):
        pat = _literal_pattern(form)
        if pat and pat.search(haystack):
            return True
    return False


def _concepts(phrase: str) -> set[str]:
    """Lexicon terms the phrase names, using every literal alternative."""
    found: set[str] = set()
    for form in alternatives(phrase):
        found |= set(lexicon.find_terms(form))
        # A tag like "Data Build Tool" is an alias, so look it up directly too.
        norm = lexicon.normalize(form)
        for term, meta in lexicon.LEXICON.items():
            if norm == lexicon.normalize(term) or norm in {
                lexicon.normalize(a) for a in meta["aliases"]
            }:
                found.add(term)
    return found


def concept_in(phrase: ExactPhrase, text: str) -> bool:
    terms = _concepts(phrase.text)
    if phrase.concept:
        terms.add(phrase.concept)
    return any(
        lexicon._pattern(form).search(text)
        for term in terms
        for form in lexicon.surface_forms(term)
    )


# --- collecting phrases from a posting --------------------------------------------

def _skills_sections(posting: JobPosting) -> list[tuple[str, str]]:
    return [(name, body) for name, body in posting.sections.items()
            if "skill" in name.lower()
            and not any(m in name.lower() for m in lexicon.SKIP_MARKERS)]


def _tags(body: str) -> list[str]:
    """Split a skills list into its tags; give up if it is prose, not a list."""
    items = [x.strip(" .-•*") for x in _SPLIT.split(body)]
    items = [x for x in items if x]
    if not items:
        return []
    short = [x for x in items if len(x.split()) <= _MAX_TAG_WORDS]
    # Mostly short items means a tag list. Mostly long ones means sentences,
    # which the keyword extractor already handles.
    if len(short) < max(2, int(0.7 * len(items))):
        return []
    return short


def _section_priority(name: str) -> Priority:
    low = name.lower()
    if any(m in low for m in lexicon.PREFERRED_MARKERS):
        return Priority.PREFERRED
    if any(m in low for m in lexicon.REQUIRED_MARKERS):
        return Priority.REQUIRED
    return Priority.MENTIONED


def _posting_surface(kw: Keyword, text: str) -> str | None:
    """The keyword exactly as it appears in the posting, longest form first."""
    forms = sorted({kw.term, *lexicon.surface_forms(kw.term), *kw.aliases},
                   key=len, reverse=True)
    for form in forms:
        if m := lexicon._pattern(form).search(text):
            return m.group(0)
    return None


def _already_covered_by(text: str, existing) -> bool:
    """True if `text` is one of the literal alternatives of a collected phrase.

    "SQL" is already accepted for the tag "Structured Query Language (SQL)", so
    listing it again would double-count the same search.
    """
    key = _norm(text)
    return any(key == _norm(alt) for ph in existing for alt in alternatives(ph.text))


def collect(posting: JobPosting, keywords: list[Keyword]) -> list[ExactPhrase]:
    phrases: dict[str, ExactPhrase] = {}

    def add(text: str, priority: Priority, source: str, concept: str | None,
            soft: bool) -> None:
        key = _norm(text)
        if not key:
            return
        existing = phrases.get(key)
        if existing is None:
            phrases[key] = ExactPhrase(text=text.strip(), priority=priority,
                                       source=source, concept=concept, soft=soft)
        elif priority.weight > existing.priority.weight:
            existing.priority = priority

    # 1. Skills tag lists -- the literal search vocabulary.
    for name, body in _skills_sections(posting):
        priority = _section_priority(name)
        for tag in _tags(body):
            concepts = _concepts(tag)
            concept = sorted(concepts)[0] if len(concepts) == 1 else None
            soft = bool(concepts) and all(
                lexicon.LEXICON[c]["category"] == "soft" for c in concepts)
            add(tag, priority, "skills list", concept, soft)

    # 2. The posting's own wording of each required or preferred skill, taken
    #    from the line it was found on so "SQL" in the requirements wins over
    #    "Structured Query Language" further down.
    for kw in keywords:
        if kw.priority is Priority.MENTIONED:
            continue
        surface = (_posting_surface(kw, kw.evidence)
                   or _posting_surface(kw, posting.raw_text) or kw.term)
        if _already_covered_by(surface, phrases.values()):
            continue
        add(surface, kw.priority, "requirements", kw.term, kw.category == "soft")

    return sorted(phrases.values(), key=lambda p: (-p.weight, p.text.lower()))


# --- scoring -----------------------------------------------------------------------

def score(resume: TailoredResume, phrases: list[ExactPhrase],
          master_text: str | None = None) -> tuple[list[PhraseItem], float]:
    page = resume.all_text()
    items: list[PhraseItem] = []
    for ph in phrases:
        if literal_in(ph.text, page):
            status = "exact"
        elif concept_in(ph, page):
            status = "concept"
        elif master_text is None or literal_in(ph.text, master_text) or concept_in(
                ph, master_text):
            status = "missing"
        else:
            status = "unsupported"
        items.append(PhraseItem(phrase=ph, status=status))

    counted = [i for i in items if i.status != "unsupported" and not i.phrase.soft]
    total = sum(i.phrase.weight for i in counted)
    got = sum(i.phrase.weight for i in counted if i.status == "exact")
    pct = round(100.0 * got / total, 1) if total else 100.0
    return items, pct


def attach(report: CoverageReport, resume: TailoredResume, phrases: list[ExactPhrase],
           master_text: str | None = None) -> CoverageReport:
    items, pct = score(resume, phrases, master_text)
    report.phrase_items = items
    report.phrase_score = pct
    return report


def unsupported_terms(phrases: list[ExactPhrase], master_text: str) -> list[str]:
    """Phrases nothing in the fact base backs -- off limits for the tailor."""
    empty = TailoredResume()
    items, _ = score(empty, phrases, master_text)
    return [i.phrase.text for i in items if i.status == "unsupported"]


def summary_line(report: CoverageReport) -> str:
    counted = [i for i in report.phrase_items
               if i.status != "unsupported" and not i.phrase.soft]
    exact = sum(1 for i in counted if i.status == "exact")
    return f"{exact} of {len(counted)} supported phrases word for word ({report.phrase_score}%)"

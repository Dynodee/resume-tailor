"""Decide which posting terms to bold on the page.

Bolding is for the human skimming after the screen: their eye lands on the
words they were hired to look for. It does nothing for the ATS, which reads the
text layer and ignores weight, so it can neither help nor hurt parsing.

The failure mode is bolding too much. A page where every third word is bold
has no emphasis at all, so the plan is deliberately stingy:

- only the summary and bullets -- never the skills list, where every item would
  qualify, and never headings
- at most two bold spans per bullet and three in the summary
- each term bolded at most once per role or project, so "SQL" lights up once in
  each job rather than in every line
- required terms first, then preferred, then the rest; longer phrases beat
  shorter ones, so "data quality" wins over "data"

The output is character spans per text unit, keyed the same way the grammar
node addresses units ("summary", "s0e1b2"). The renderer applies them; nothing
here touches the words themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ats import lexicon
from ..ats.phrases import alternatives
from ..state import ExactPhrase, Keyword, TailoredResume

PER_BULLET = 2
PER_SUMMARY = 3


@dataclass(frozen=True)
class _Matcher:
    concept: str          # what counts as "the same term" for the once-per-role rule
    pattern: re.Pattern[str]
    rank: float


def _span_pattern(form: str) -> re.Pattern[str] | None:
    """Tolerant of hyphen-vs-space and a trailing plural, bounded on both sides."""
    tokens = re.findall(r"[A-Za-z0-9+#&./']+", form)
    if not tokens:
        return None
    body = r"[\s\-/]+".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?:s|es)?(?![A-Za-z0-9])", re.IGNORECASE)


def _matchers(keywords: list[Keyword], phrases: list[ExactPhrase]) -> list[_Matcher]:
    out: list[_Matcher] = []
    seen: set[tuple[str, str]] = set()

    def add(concept: str, form: str, weight: float) -> None:
        key = (concept.lower(), form.lower())
        if key in seen or len(form.strip()) < 2:
            return
        seen.add(key)
        if pat := _span_pattern(form):
            out.append(_Matcher(concept.lower(), pat, weight))

    for kw in keywords:
        for form in {kw.term, *lexicon.surface_forms(kw.term), *kw.aliases}:
            add(kw.term, form, kw.priority.weight)
    for ph in phrases:
        for form in alternatives(ph.text):
            add(ph.concept or ph.text, form, ph.priority.weight)
    return out


def _units(resume: TailoredResume):
    """(key, entry-id, text) for every unit that may carry bold."""
    if resume.summary:
        yield "summary", "summary", resume.summary
    for si, section in enumerate(resume.sections):
        if section.kind not in ("experience", "projects"):
            continue
        for ei, entry in enumerate(section.entries):
            for bi, bullet in enumerate(entry.get("bullets", [])):
                text = bullet["text"] if isinstance(bullet, dict) else str(bullet)
                yield f"s{si}e{ei}b{bi}", f"s{si}e{ei}", text


def plan(resume: TailoredResume, keywords: list[Keyword],
         phrases: list[ExactPhrase] | None = None,
         per_bullet: int = PER_BULLET, per_summary: int = PER_SUMMARY,
         ) -> dict[str, list[tuple[int, int]]]:
    matchers = _matchers(keywords, phrases or [])
    used_in_entry: dict[str, set[str]] = {}
    result: dict[str, list[tuple[int, int]]] = {}

    for key, entry_id, text in _units(resume):
        candidates: list[tuple[float, int, int, int, str]] = []
        for m in matchers:
            for hit in m.pattern.finditer(text):
                # rank: priority, then length, then earlier position
                candidates.append((m.rank, hit.end() - hit.start(), -hit.start(),
                                   hit.start(), m.concept))
        # Priority first, then reading order (the first mention is the one the
        # eye meets), then length so a longer phrase beats a shorter one that
        # starts at the same place.
        candidates.sort(key=lambda c: (-c[0], c[3], -c[1]))

        cap = per_summary if key == "summary" else per_bullet
        taken: list[tuple[int, int]] = []
        concepts_here = used_in_entry.setdefault(entry_id, set())
        for rank, length, _neg, start, concept in candidates:
            if len(taken) >= cap:
                break
            end = start + length
            if concept in concepts_here:
                continue
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            concepts_here.add(concept)
        if taken:
            result[key] = _merge_adjacent(sorted(taken), text)
    return result


def _merge_adjacent(spans: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    """Join spans separated only by a space: "**Tableau dashboards**", not
    "**Tableau** **dashboards**", which reads as a rendering glitch."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and not text[merged[-1][1]:start].strip():
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def bolded_terms(resume: TailoredResume) -> list[str]:
    """The distinct words that ended up bold, for the change report."""
    texts = dict((k, t) for k, _, t in _units(resume))
    seen: dict[str, str] = {}
    for key, spans in resume.emphasis.items():
        for s, e in spans:
            word = texts.get(key, "")[s:e]
            seen.setdefault(word.lower(), word)
    return sorted(seen.values(), key=str.lower)


__all__ = ["plan", "bolded_terms", "PER_BULLET", "PER_SUMMARY"]

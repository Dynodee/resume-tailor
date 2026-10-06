"""Node 2 -- work out which terms the screen is actually looking for.

Hybrid by design. The lexicon pass is exact, repeatable, and free; it catches
the terms that matter most because they are the ones everyone lists. The LLM
pass is there for the rest: domain vocabulary, in-house tool names, phrasings
no dictionary anticipated. Neither alone is enough -- the dictionary misses
novelty and the model paraphrases.

Priority comes from which section of the posting a term appeared in, not from
the model's opinion, because that mapping is the one thing the posting states
outright.
"""

from __future__ import annotations

from ..ats import lexicon, phrases as phrase_mod
from ..config import Config
from ..llm import LLM
from ..state import JobPosting, Keyword, PipelineState, Priority

SYSTEM = """You extract applicant-tracking-system keywords from job postings.

You are looking for terms a keyword screen would match on: named technologies,
methods, certifications, domain vocabulary, and role titles. Return the term as
the posting writes it, plus the obvious surface variants a resume might use.

Rules:
- Skip generic filler ("fast-paced environment", "team player", "passion").
- Skip company perks, benefits, and EEO boilerplate.
- Prefer the specific over the general: "LangGraph" over "frameworks".
- Multi-word terms are fine; keep them short enough to appear verbatim in a bullet.
- At most 30 terms."""

USER = """Job title: {title}
Company: {company}

Posting sections:
{sections}

Return JSON: a list of objects with keys
  term       (string, canonical form as the posting writes it)
  aliases    (list of strings, other ways a resume might phrase it; may be empty)
  category   (one of: skill, tool, credential, title, soft)
  priority   (one of: required, preferred, mentioned)
  evidence   (the phrase from the posting it came from, under 150 chars)"""


def _priority_for_section(section_name: str) -> Priority:
    name = section_name.lower()
    if any(m in name for m in lexicon.PREFERRED_MARKERS):
        return Priority.PREFERRED
    if any(m in name for m in lexicon.REQUIRED_MARKERS):
        return Priority.REQUIRED
    return Priority.MENTIONED


def _deterministic(posting: JobPosting) -> dict[str, Keyword]:
    """Lexicon pass, with priority taken from the section a term appeared in."""
    found: dict[str, Keyword] = {}

    # Highest-priority section wins if a term shows up in more than one.
    ordered = sorted(
        posting.sections.items(),
        key=lambda kv: -_priority_for_section(kv[0]).weight,
    )
    for name, body in ordered:
        if any(m in name.lower() for m in lexicon.SKIP_MARKERS):
            continue
        priority = _priority_for_section(name)
        for term, meta in lexicon.find_terms(body).items():
            if term in found:
                found[term].frequency += meta["frequency"]
                continue
            found[term] = Keyword(
                term=term,
                normalized=lexicon.normalize(term),
                aliases=meta["aliases"],
                category=meta["category"],
                priority=priority,
                frequency=meta["frequency"],
                evidence=meta["evidence"],
            )

    # Terms in the job title itself are always required -- a screen that filters
    # on anything filters on the title.
    if posting.title:
        for term, meta in lexicon.find_terms(posting.title).items():
            kw = found.get(term)
            if kw:
                kw.priority = Priority.REQUIRED
            else:
                found[term] = Keyword(
                    term=term, normalized=lexicon.normalize(term),
                    aliases=meta["aliases"], category=meta["category"],
                    priority=Priority.REQUIRED, frequency=meta["frequency"],
                    evidence=posting.title,
                )
    return found


def _from_llm(llm: LLM, posting: JobPosting) -> list[Keyword]:
    if llm.offline:
        return []
    sections = "\n\n".join(
        f"## {name}\n{body[:3000]}" for name, body in posting.sections.items()
    )
    payload = llm.complete_json(
        SYSTEM,
        USER.format(title=posting.title or "(not stated)",
                    company=posting.company or "(not stated)",
                    sections=sections or posting.raw_text[:8000]),
        max_tokens=8000,
        effort="low",
    )
    out: list[Keyword] = []
    for item in payload or []:
        if not isinstance(item, dict) or not item.get("term"):
            continue
        try:
            priority = Priority(str(item.get("priority", "mentioned")).lower())
        except ValueError:
            priority = Priority.MENTIONED
        out.append(Keyword(
            term=str(item["term"]).strip(),
            normalized=lexicon.normalize(str(item["term"])),
            aliases=[str(a) for a in item.get("aliases", []) if a][:6],
            category=str(item.get("category", "other")).lower(),
            priority=priority,
            evidence=str(item.get("evidence", ""))[:200],
        ))
    return out


def extract(state: PipelineState, cfg: Config, llm: LLM) -> dict:
    posting = state.posting
    if posting is None:
        return {"errors": state.errors + ["extract: no posting in state"]}

    found = _deterministic(posting)
    n_lexicon = len(found)

    for kw in _from_llm(llm, posting):
        existing = found.get(kw.term)
        if existing is None:
            # Dedupe against aliases too, so "LLMs" does not become a second
            # entry next to "LLM".
            norm = kw.normalized
            alias_hit = next(
                (k for k in found.values()
                 if norm == k.normalized
                 or norm in {lexicon.normalize(a) for a in k.aliases}),
                None,
            )
            if alias_hit:
                # The model saw it too; trust the stronger priority.
                if kw.priority.weight > alias_hit.priority.weight:
                    alias_hit.priority = kw.priority
                alias_hit.aliases = list(dict.fromkeys([*alias_hit.aliases, *kw.aliases]))
                continue
            found[kw.term] = kw
        else:
            existing.aliases = list(dict.fromkeys([*existing.aliases, *kw.aliases]))
            if kw.priority.weight > existing.priority.weight:
                existing.priority = kw.priority

    keywords = sorted(found.values(), key=lambda k: (-k.score, k.term.lower()))

    log = list(state.log)
    counts = {p.value: sum(1 for k in keywords if k.priority is p) for p in Priority}
    log.append(
        f"extract: {len(keywords)} terms ({n_lexicon} from lexicon, "
        f"{len(keywords) - n_lexicon} new from model) | "
        f"required={counts['required']} preferred={counts['preferred']} "
        f"mentioned={counts['mentioned']}"
    )
    exact = phrase_mod.collect(posting, keywords)
    tags = sum(1 for ph in exact if ph.source == "skills list")
    log.append(f"extract: {len(exact)} exact phrases to match word for word"
               + (f" ({tags} from the posting's skills list)" if tags else ""))
    return {"keywords": keywords, "phrases": exact, "log": log}

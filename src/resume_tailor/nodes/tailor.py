"""Nodes 3 and 4 -- load the fact base, then tailor against the posting.

The hard constraint here is that tailoring is a *rephrasing and selection*
problem, not a generation problem. The model may reorder, re-emphasise, and
rewrite the wording of a fact; it may not add a fact. Every bullet it emits
carries the id of the master fact it came from, and anything that fails to
trace back is dropped before the document is assembled. That check is what
keeps an honest resume honest when the posting asks for something you have
never done.
"""

from __future__ import annotations

import re
from typing import Any

from ..ats import lexicon, phrases as phrase_mod, scorer
from ..config import Config
from ..llm import LLM
from ..sources import resume_source
from ..state import (
    Keyword,
    PipelineState,
    Priority,
    ResumeSection,
    TailoredResume,
)

SYSTEM = """You tailor an existing resume to a specific job posting.

ABSOLUTE RULE: you may only restate facts that appear in the MASTER FACT BASE.
You may reorder, re-emphasise, merge, trim, and rewrite the wording of a fact.
You may not introduce a tool, employer, metric, date, credential, or
responsibility that is not in the fact base. If the posting asks for something
absent from the fact base, leave it out -- do not soften it into an implied
claim. Dropping a true detail that does not matter for this role is fine.

HOW TO TAILOR -- relevance first, keywords second:
1. Read THE ROLE section. Work out the three or four things this job actually
   does day to day.
2. For each bullet you keep, decide which of those things it is evidence for,
   and rewrite it so that part leads. Cut clauses that do not serve this role.
   Example: for a role about data quality and reporting standards, "Write SQL
   and MDX to map and validate data structures across large reporting systems"
   becomes "Write SQL to validate data quality across large reporting
   systems..." -- same fact, emphasis on what this employer needs.
3. Drop bullets that are evidence for nothing in the posting. A shorter,
   sharper resume beats a complete one.
4. Only then use the posting's exact terminology, and only where it fits the
   sentence naturally. A keyword screen matches strings, so prefer the
   posting's phrase over a synonym when the fact supports it.
5. Recruiters search the applicant tracking system for the EXACT PHRASES
   listed, word for word -- "dbt" does not match a search for "Data Build
   Tool". Get each one onto the page verbatim where the fact base supports it,
   using the least intrusive place first:
   - a skills group label ("Data Visualization & BI")
   - a skill item written with the posting's name in parentheses, for a skill
     already in the fact base: "dbt (Data Build Tool)"
   - the headline
   - a bullet, only where the phrase reads naturally
   The no-stuffing rules below still apply.

KEYWORD STUFFING IS A FAILURE, NOT A TRADE-OFF. A human reads this after the
screen, and forced keywords are the fastest way to lose them. Never:
- gloss a keyword in parentheses: "big data (large, complex datasets)"
- name a soft skill: "applied problem-solving", "leveraged communication" --
  show it through what was done instead
- use the same word twice in one sentence
- add a keyword to a bullet where it does not change the meaning

STYLE: lead each bullet with a concrete action verb; keep the numbers already
in the fact base; put the result before the mechanism where both are known.
Present tense for a role whose end date is "Present", past tense otherwise. No
first-person pronouns. One or two sentences per bullet, ending with a period."""

USER = """# TARGET ROLE
{title} at {company}

# THE ROLE (from the posting)
{role_brief}

# KEYWORDS THE SCREEN IS LOOKING FOR
Required: {required}
Preferred: {preferred}
Also mentioned: {mentioned}
Not supported by the fact base -- do not attempt these: {unattainable}

# EXACT PHRASES A RECRUITER WILL SEARCH FOR
{exact_phrases}

# CLAIMS THAT ARE NOT TRUE -- never imply these
{do_not_claim}

# MASTER FACT BASE
{master}

{feedback}
# TASK
Produce the tailored resume as JSON:

{{
  "headline": "string, the pipe-separated headline line",
  "summary": "exactly 2 sentences, at most 45 words in total",
  "experience": [
    {{"id": "<experience id from fact base>",
      "bullets": [{{"source_id": "<bullet id from fact base>", "text": "rewritten bullet"}}]}}
  ],
  "projects": [
    {{"id": "<project id>",
      "bullets": [{{"source_id": "<bullet id>", "text": "rewritten bullet"}}]}}
  ],
  "skills": [
    {{"label": "group label", "items": ["skill", "skill"]}}
  ],
  "notes": "one line on what you emphasised and what you had to leave out"
}}

SUMMARY: sentence one says who the candidate is for THIS role, with years of
experience and at most three of the posting's tools. Sentence two gives one
concrete, numeric proof point that matters to this employer. Mention the M.S.
only if the posting asks for a degree or the role is AI-focused. No lists of
five tools -- the skills section does that.

BUDGET: keep every role (gaps read worse than an imperfect fit), newest first,
with two to four bullets each -- most relevant first. Projects get at most three
bullets; a project irrelevant to this posting gets one bullet or is omitted.
At most 15 bullets in total. In "skills", only list items that appear in the
fact base's skills section, ordered so the posting's terms come first, and omit
groups irrelevant to this role."""


def load_master(state: PipelineState, cfg: Config) -> dict:
    master = resume_source.load(state.resume_ref, cfg.master_path)
    counts = (
        f"{len(master.get('experience', []))} role(s), "
        f"{len(master.get('projects', []))} project(s), "
        f"{sum(len(g.get('items', [])) for g in (master.get('skills') or {}).values())} skills"
    )
    log = list(state.log)
    log.append(f"master: loaded {counts}"
               + (f" from {state.resume_ref}" if state.resume_ref else ""))
    return {"master": master, "log": log}


# --- fact-base helpers --------------------------------------------------------

def _bullet_index(master: dict) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for group in ("experience", "projects"):
        for entry in master.get(group, []):
            for bullet in entry.get("bullets", []):
                idx[bullet["id"]] = bullet
    return idx


def _entry_index(master: dict) -> dict[str, dict]:
    return {e["id"]: e for group in ("experience", "projects")
            for e in master.get(group, [])}


def _allowed_skills(master: dict) -> set[str]:
    return {s.strip().lower()
            for group in (master.get("skills") or {}).values()
            for s in group.get("items", [])}


def _education_section(master: dict) -> ResumeSection:
    """Degrees, then certifications, under one heading.

    Certifications reuse the education entry shape (title / issuer / date) so
    the renderer, linter and scorer need no special case for them.
    """
    entries = [dict(e) for e in master.get("education", [])]
    certs = master.get("certifications") or []
    for c in certs:
        entries.append({"degree": c.get("name", ""), "school": c.get("issuer", ""),
                        "dates": c.get("date", "")})
    heading = "EDUCATION & CERTIFICATIONS" if certs else "EDUCATION"
    return ResumeSection(heading=heading, kind="education", entries=entries)


def _render_master(master: dict) -> str:
    """Compact view of the fact base for the prompt -- ids included, so the
    model can only reference facts that exist."""
    lines: list[str] = []
    lines.append("## Headline options")
    for h in master.get("headlines", []):
        lines.append(f"- [{h['id']}] {h['text']}  (fits: {', '.join(h.get('fits', []))})")
    lines.append("\n## Facts the summary may use")
    lines.extend(f"- {f}" for f in master.get("summary_facts", []))
    for group, label in (("experience", "Experience"), ("projects", "Projects")):
        lines.append(f"\n## {label}")
        for entry in master.get(group, []):
            head = entry.get("title") or entry.get("name", "")
            sub = entry.get("company") or entry.get("subtitle", "")
            dates = f"{entry.get('start', '')} - {entry.get('end', '')}".strip(" -")
            lines.append(f"\n[{entry['id']}] {head} | {sub} | {dates} "
                         f"| tags: {', '.join(entry.get('tags', []))}")
            for b in entry.get("bullets", []):
                hint = ", ".join(b.get("keywords", []))
                lines.append(f"  - [{b['id']}] {b['text']}"
                             + (f"\n      (relates to: {hint})" if hint else ""))
    if master.get("certifications"):
        lines.append("\n## Certifications")
        lines.extend(f"- {c.get('name', '')} ({c.get('issuer', '')})"
                     for c in master["certifications"])
    lines.append("\n## Skills available")
    for group in (master.get("skills") or {}).values():
        lines.append(f"- {group['label']}: {', '.join(group.get('items', []))}")
    return "\n".join(lines)


def _role_brief(posting, limit: int = 5000) -> str:
    """The parts of the posting that describe the job, without perks and EEO."""
    if posting is None:
        return "(no posting text)"
    parts = []
    for name, body in posting.sections.items():
        if any(m in name.lower() for m in lexicon.SKIP_MARKERS):
            continue
        parts.append(f"## {name}\n{body.strip()}")
    text = "\n\n".join(parts) or posting.raw_text
    return text[:limit]


def master_text(master: dict) -> str:
    import yaml as _yaml

    return _yaml.safe_dump({k: v for k, v in master.items() if k != "do_not_claim"})


def _hard_required(keywords: list[Keyword]) -> list[Keyword]:
    return [k for k in keywords
            if k.priority is Priority.REQUIRED and k.category != "soft"]


def order_sections(resume: TailoredResume, keywords: list[Keyword]) -> TailoredResume:
    """Put work experience first unless the projects carry the match.

    Leading with projects is the right call for a career pivot (an agentic AI
    role where LangGraph only appears in a side project) and the wrong one for a
    role your day job already matches -- there it buries your strongest
    evidence under your weakest.
    """
    kinds = {s.kind: s for s in resume.sections}
    exp, proj = kinds.get("experience"), kinds.get("projects")
    if not (exp and proj):
        return resume
    wanted = _hard_required(keywords) or [k for k in keywords if k.category != "soft"]

    def hits(section: ResumeSection) -> int:
        probe = TailoredResume(sections=[section])
        return sum(1 for i in scorer.score(probe, wanted).items if i.covered)

    projects_first = hits(proj) > hits(exp)
    order = (["projects", "experience"] if projects_first else ["experience", "projects"])
    head = [kinds[k] for k in order]
    rest = [s for s in resume.sections if s.kind not in order]
    resume.sections = head + rest
    return resume


def _feedback_block(state: PipelineState) -> str:
    """On a revision pass, show the model its previous draft and what to fix.

    The draft has to be included: the lint findings point at locations like
    "experience[1].bullets[2]", which mean nothing to a model that cannot see
    the document they refer to.
    """
    if state.revisions == 0 or state.resume is None:
        return ""
    import json

    parts = ["# REVISION FEEDBACK",
             f"This is revision {state.revisions}. Below is your previous draft. "
             "Fix the problems listed and keep everything else as it was."]
    parts.append("\n## Previous draft\n" + json.dumps(_draft_payload(state.resume), indent=1))
    if state.coverage:
        unattainable = set(state.coverage.unattainable)
        missing = [k for k in state.coverage.missing(Priority.REQUIRED)
                   if k.category != "soft" and k.term not in unattainable][:8]
        if missing:
            parts.append("\nRequired terms the fact base supports but the draft does not "
                         "use. Work each in where it reads naturally; skip any that "
                         "would need forcing:")
            parts.extend(f"  - {k.term}" for k in missing)
    if state.coverage and state.coverage.phrase_items:
        other_words = [p.text for p in state.coverage.phrases("concept") if not p.soft]
        absent = [p.text for p in state.coverage.phrases("missing") if not p.soft]
        if other_words:
            parts.append("\nExact phrases the draft only says in other words -- put the "
                         "posting's wording on the page (label, skill item with the "
                         "posting's name in parentheses, headline, or a natural bullet):")
            parts.extend(f"  - {t}" for t in other_words)
        if absent:
            parts.append("Exact phrases the fact base supports but the draft omits:")
            parts.extend(f"  - {t}" for t in absent)
    if state.grammar_issues:
        parts.append("\nWriting problems to fix:")
        for issue in state.grammar_issues[:20]:
            parts.append(f"  - [{issue.rule}] {issue.location}: {issue.message}")
            if issue.excerpt:
                parts.append(f"      > {issue.excerpt}")
    parts.append("")
    return "\n".join(parts)


def _draft_payload(resume: TailoredResume) -> dict:
    out: dict[str, Any] = {"headline": resume.headline, "summary": resume.summary}
    for section in resume.sections:
        if section.kind in ("experience", "projects"):
            out[section.kind] = [
                {"entry": e.get("title") or e.get("name"),
                 "bullets": [{"source_id": b.get("source_id"), "text": b.get("text")}
                             for b in e.get("bullets", [])]}
                for e in section.entries
            ]
        elif section.kind == "skills":
            out["skills"] = section.entries
    return out


# --- offline fallback ---------------------------------------------------------

def _offline_tailor(master: dict, keywords: list[Keyword]) -> TailoredResume:
    """Deterministic selection and ordering, no rewriting.

    Bullets keep their master wording; what changes is which ones appear and in
    what order. That is a real (if blunt) tailoring strategy, and it means the
    pipeline produces a valid, honest document with no model available.
    """
    wanted = {k.normalized for k in keywords}
    wanted |= {lexicon.normalize(a) for k in keywords for a in k.aliases}

    def _overlaps(hint: str) -> bool:
        """Loose match between a bullet's keyword hint and a wanted term.

        Exact equality is too strict here: a posting asking for "unit testing"
        should be matched by a bullet hinted "testing". Containment in either
        direction is the right level of looseness for ranking -- it never
        changes what the document claims, only which true bullet gets shown.
        """
        norm = lexicon.normalize(hint)
        if not norm:
            return False
        return any(norm == w or norm in w or w in norm for w in wanted)

    def bullet_score(bullet: dict) -> float:
        hits = sum(1 for kw in bullet.get("keywords", []) if _overlaps(kw))
        text_hits = len(set(lexicon.find_terms(bullet["text"])) &
                        {k.term for k in keywords})
        return hits * 2 + text_hits

    def entry_score(entry: dict) -> float:
        tag_hits = sum(1 for t in entry.get("tags", []) if _overlaps(t))
        return tag_hits * 3 + sum(bullet_score(b) for b in entry.get("bullets", []))

    # Headline: whichever option shares the most vocabulary with the posting.
    headlines = master.get("headlines", [])
    headline = ""
    if headlines:
        headline = max(
            headlines,
            key=lambda h: sum(1 for f in h.get("fits", []) if lexicon.normalize(f) in wanted)
            + len(set(lexicon.find_terms(h["text"])) & {k.term for k in keywords}),
        )["text"]

    facts = master.get("summary_facts", [])
    scored_facts = sorted(
        facts, key=lambda f: -len(set(lexicon.find_terms(f)) & {k.term for k in keywords})
    )
    # Pick the two most relevant facts, but keep them in fact-base order so the
    # "who you are" sentence still comes first.
    picked = sorted(scored_facts[:2], key=facts.index)
    summary = " ".join(s.rstrip(".") + "." for s in picked)

    sections: list[ResumeSection] = []

    projects = sorted(master.get("projects", []), key=entry_score, reverse=True)
    if projects:
        sections.append(ResumeSection(
            heading="AI & ANALYTICS PROJECTS", kind="projects",
            entries=[{
                "name": p.get("name", ""), "subtitle": p.get("subtitle", ""),
                "year": p.get("year", ""), "end": "",
                "bullets": [{"source_id": b["id"], "text": b["text"]}
                            for b in sorted(p.get("bullets", []),
                                            key=bullet_score, reverse=True)[:3]],
            } for p in projects],
        ))

    experience = master.get("experience", [])  # already newest-first in the YAML
    sections.append(ResumeSection(
        heading="WORK EXPERIENCE", kind="experience",
        entries=[{
            "title": e.get("title", ""), "company": e.get("company", ""),
            "location": e.get("location", ""), "start": e.get("start", ""),
            "end": e.get("end", ""),
            "bullets": [{"source_id": b["id"], "text": b["text"]}
                        for b in sorted(e.get("bullets", []),
                                        key=bullet_score, reverse=True)[:4]],
        } for e in experience],
    ))

    sections.append(_education_section(master))

    skill_groups = []
    for group in (master.get("skills") or {}).values():
        items = sorted(group.get("items", []),
                       key=lambda s: lexicon.normalize(s) not in wanted)
        skill_groups.append({"label": group["label"], "items": items})
    skill_groups.sort(
        key=lambda g: -sum(1 for i in g["items"] if lexicon.normalize(i) in wanted)
    )
    sections.append(ResumeSection(heading="SKILLS", kind="skills", entries=skill_groups))

    return order_sections(
        TailoredResume(headline=headline, summary=summary, sections=sections), keywords)


# --- assembly -----------------------------------------------------------------

def skill_allowed(item: str, allowed: set[str]) -> bool:
    """A skills-section item must be one you listed -- or that same skill under
    the posting's name.

    "dbt (Data Build Tool)" is allowed when "dbt" is in the fact base, because
    the lexicon says the two name the same tool; that is how a skills section
    picks up the exact phrase a recruiter searches for without claiming
    anything new. "dbt (Microsoft Fabric)" is not, because they do not.
    """
    if item.strip().lower() in allowed:
        return True
    m = re.match(r"^(.+?)\s*\((.+)\)\s*$", item.strip())
    if not m:
        return False
    outer, inner = (x.strip() for x in m.groups())
    for listed, other in ((outer, inner), (inner, outer)):
        if listed.lower() in allowed:
            same = phrase_mod._concepts(listed) & phrase_mod._concepts(other)
            if same:
                return True
    return False


def _assemble(payload: dict, master: dict,
              forbidden: list[str] | tuple = ()) -> tuple[TailoredResume, list[str]]:
    """Turn the model's JSON into a TailoredResume, dropping anything untraceable."""
    bullets = _bullet_index(master)
    entries = _entry_index(master)
    allowed_skills = _allowed_skills(master)
    dropped: list[str] = []

    def build(group_key: str, kind: str, heading: str) -> ResumeSection | None:
        out_entries: list[dict[str, Any]] = []
        for item in payload.get(group_key, []) or []:
            src = entries.get(str(item.get("id", "")))
            if src is None:
                dropped.append(f"unknown {kind} id '{item.get('id')}'")
                continue
            kept = []
            for b in item.get("bullets", []) or []:
                sid = str(b.get("source_id", ""))
                text = str(b.get("text", "")).strip()
                if sid not in bullets:
                    dropped.append(f"bullet with unknown source_id '{sid}'")
                    continue
                if not text:
                    continue
                kept.append({"source_id": sid, "text": text})
            if not kept:
                kept = [{"source_id": b["id"], "text": b["text"]}
                        for b in src.get("bullets", [])[:2]]
            base = {
                "location": src.get("location", ""),
                "start": src.get("start", ""),
                "end": src.get("end", ""),
                "bullets": kept,
            }
            if kind == "experience":
                base |= {"title": src.get("title", ""), "company": src.get("company", "")}
            else:
                base |= {"name": src.get("name", ""), "subtitle": src.get("subtitle", ""),
                         "year": src.get("year", "")}
            out_entries.append(base)
        return ResumeSection(heading=heading, kind=kind, entries=out_entries) if out_entries else None

    sections: list[ResumeSection] = []
    if proj := build("projects", "projects", "AI & ANALYTICS PROJECTS"):
        sections.append(proj)
    if exp := build("experience", "experience", "WORK EXPERIENCE"):
        sections.append(exp)

    sections.append(_education_section(master))

    skill_entries = []
    for group in payload.get("skills", []) or []:
        raw_items = [str(x).strip() for x in group.get("items", [])]
        items = [x for x in raw_items if skill_allowed(x, allowed_skills)]
        dropped.extend(f"invented skill '{x}'" for x in raw_items
                       if not skill_allowed(x, allowed_skills))
        label = str(group.get("label", "Skills")).strip() or "Skills"
        if any(phrase_mod.literal_in(term, label) for term in forbidden):
            # A group label is free text, which makes it the easiest place to
            # smuggle in a claim ("Insurance Analytics"). It gets the same
            # check as everything else.
            dropped.append(f"label '{label}' names something not in the fact base")
            label = "Skills"
        if items:
            skill_entries.append({"label": label, "items": items})
    if not skill_entries:
        skill_entries = [{"label": g["label"], "items": g.get("items", [])}
                         for g in (master.get("skills") or {}).values()]
    sections.append(ResumeSection(heading="SKILLS", kind="skills", entries=skill_entries))

    resume = TailoredResume(
        headline=str(payload.get("headline", "")).strip(),
        summary=str(payload.get("summary", "")).strip(),
        sections=sections,
    )
    return resume, dropped


# --- fabrication guard --------------------------------------------------------

_TECHY = re.compile(r"\b(?:[A-Z][A-Za-z0-9]*(?:\.[a-z]+)?|[A-Z]{2,})\b")
_STOP = {
    "A", "An", "The", "And", "Or", "But", "For", "With", "From", "Into", "Over",
    "Built", "Wrote", "Led", "Designed", "Owned", "Drove", "Used", "Turned",
    "Tracked", "Rebuilt", "Managed", "Translate", "Write", "Lead", "Manage",
    "Own", "Partner", "Present", "Implemented", "Exposed", "Served", "Supported",
    "Diagnosed", "Wrote", "Analytics", "Data", "Python", "SQL",
}


def audit(resume: TailoredResume, master: dict) -> list[str]:
    """Flag proper nouns in the output that do not appear in the fact base.

    This is a blunt instrument and it is meant to be: it produces warnings for a
    human, not automatic edits. It exists because the expensive failure mode of
    this whole pipeline is a plausible sentence about something you have not
    done, and that failure is invisible in a coverage score.
    """
    import yaml as _yaml

    master_text = _yaml.safe_dump(master).lower()
    suspects: dict[str, int] = {}
    for token in _TECHY.findall(resume.all_text()):
        if token in _STOP or len(token) < 3:
            continue
        low = token.lower()
        # Inflections of a word the fact base uses are not new claims:
        # "Translated" is the past tense of "Translate".
        stems = {low, low[:-1], low[:-2], low[:-3]} if len(low) > 5 else {low}
        if low not in master_text and not any(
                len(st) >= 4 and st in master_text for st in stems):
            suspects[token] = suspects.get(token, 0) + 1
    return [f"'{t}' appears in the tailored resume but not in the fact base"
            for t in sorted(suspects)]


# --- the node -----------------------------------------------------------------

def tailor(state: PipelineState, cfg: Config, llm: LLM) -> dict:
    master = state.master
    keywords = state.keywords
    posting = state.posting
    log = list(state.log)
    errors = list(state.errors)

    if llm.offline:
        resume = _offline_tailor(master, keywords)
        log.append(f"tailor: offline mode -- selected and ordered {_count(resume)} "
                   "bullets from the fact base without rewriting")
    else:
        by_priority = {p: [k.term for k in keywords if k.priority is p] for p in Priority}
        unattainable = scorer.score(TailoredResume(), keywords,
                                    master_text(master)).unattainable
        unsupported = phrase_mod.unsupported_terms(state.phrases, master_text(master))
        forbidden = list(dict.fromkeys([*unattainable, *unsupported]))
        exact_lines = [f"- {p.text}" for p in state.phrases
                       if not p.soft and p.text not in unsupported]
        payload = llm.complete_json(
            SYSTEM,
            USER.format(
                title=posting.title or "(not stated)",
                company=posting.company or "(not stated)",
                role_brief=_role_brief(posting),
                unattainable=", ".join(forbidden) or "(none)",
                exact_phrases="\n".join(exact_lines) or "(none identified)",
                required=", ".join(by_priority[Priority.REQUIRED]) or "(none identified)",
                preferred=", ".join(by_priority[Priority.PREFERRED]) or "(none)",
                mentioned=", ".join(by_priority[Priority.MENTIONED][:25]) or "(none)",
                do_not_claim="\n".join(f"- {c}" for c in master.get("do_not_claim", []))
                             or "(none listed)",
                master=_render_master(master),
                feedback=_feedback_block(state),
            ),
            max_tokens=6000,
        )
        resume, dropped = _assemble(payload or {}, master, forbidden)
        resume = order_sections(resume, keywords)
        note = str((payload or {}).get("notes", "")).strip()
        log.append(f"tailor: revision {state.revisions} -- {_count(resume)} bullets"
                   + (f" | {len(dropped)} dropped as untraceable" if dropped else ""))
        if note:
            log.append(f"tailor note: {note}")
        errors.extend(f"dropped: {d}" for d in dropped[:10])

    for warning in audit(resume, master):
        errors.append(f"fabrication check: {warning}")

    coverage = scorer.score(resume, keywords, master_text(master))
    phrase_mod.attach(coverage, resume, state.phrases, master_text(master))
    log.append(f"tailor: coverage {coverage.score}% overall, "
               f"{coverage.attainable_score}% of the required terms your fact base "
               f"supports"
               + (f" | not in your fact base: {', '.join(coverage.unattainable)}"
                  if coverage.unattainable else ""))
    if coverage.phrase_items:
        log.append(f"tailor: exact phrases -- {phrase_mod.summary_line(coverage)}")

    return {"resume": resume, "coverage": coverage, "log": log, "errors": errors}


def _count(resume: TailoredResume) -> int:
    return sum(len(e.get("bullets", [])) for s in resume.sections for e in s.entries)

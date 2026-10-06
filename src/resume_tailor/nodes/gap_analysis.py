"""Node 3b -- sort every requirement the fact base does not state in so many words.

Most "gaps" are not gaps. The scorer calls a term unattainable when none of its
spellings appears in the fact base, so "CI/CD" against a bullet about tests that
run on every merge counts as missing -- and the tailor is then told never to
touch it. This node looks at each such term and sorts it:

    reframe   a bullet shows the skill in other words -> the posting's term may
              go on that bullet (and only that bullet)
    adjacent  related, not the same -> the bullet stresses what transfers, the
              cover letter names the difference
    ask       plausible but unknown -> a question for the user
    gap       nothing backs it -> a cover-letter topic

Reframes are the dangerous call, so they get a second, independent review that
asks only "does this bullet, as written, show this skill?". A reframe the
reviewer rejects becomes adjacent. Seniority, credentials and named tools never
reframe, and anything on the do-not-claim list is a gap before the model sees
it.

In interview mode (`--interview`) the open items become questions. A yes turns
into a new fact, saved beside the fact base so it is there on every later run;
a no is recorded so the term is never claimed or asked about again.
"""

from __future__ import annotations

import copy
import datetime as dt
import re
from typing import Literal

from pydantic import BaseModel

from ..ats import lexicon, phrases as phrase_mod
from ..config import Config
from ..llm import LLM
from ..sources import resume_source
from ..state import GapItem, PipelineState, Priority
from .tailor import _bullet_index, _render_master, _role_brief, master_text

MAX_CANDIDATES = 20
MAX_QUESTIONS = 10

SORT_SYSTEM = """You sort job requirements against a candidate's fact base.

Each requirement below is something the posting asks for that the fact base
does not state in those words. Decide which of these applies:

- reframe: one or more bullets already show this skill in different words, so
  the posting's term is an accurate description of what the bullet says was
  done. Cite the bullet ids. Example: "Set up automated tests that run on every
  merge" shows "CI/CD".
- adjacent: the bullets show related experience that transfers, but calling it
  the posting's term would overstate it. Example: building Tableau dashboards is
  adjacent to "Power BI". Cite the bullet ids.
- ask: nothing in the fact base shows it, but someone with this background
  plausibly did it and never wrote it down. Write one short, specific question
  that would establish it, naming a concrete example of what would count.
- gap: nothing in the fact base suggests it.

Be strict about reframe. It must be true of the bullet as written, not of what
the person probably also did. Seniority, years of experience, degrees,
licenses, certifications and named tools or vendors are never reframes: the
fact base either says them or they are ask or gap. When torn between reframe
and adjacent, choose adjacent.

rationale: one sentence, quoting the words in the bullet that justify the call.
source_ids: bullet ids for reframe and adjacent, empty otherwise.
question: for ask only, empty otherwise.
Return every requirement exactly once, spelled exactly as given."""

SORT_USER = """# THE ROLE
{title} at {company}

{role_brief}

# REQUIREMENTS TO SORT
{requirements}

# CLAIMS THAT ARE NOT TRUE
{do_not_claim}

# FACT BASE
{master}"""

VERIFY_SYSTEM = """You are a skeptical reviewer checking claimed evidence on a resume.

Each item pairs a skill a job posting asks for with one bullet from the
candidate's own records. Decide whether a hiring manager reading the bullet
would agree the candidate has done that skill as stated -- not something
similar, not something they probably also did.

supported: true only if the bullet's own words demonstrate the skill.
reason: one sentence."""

FACT_SYSTEM = """You turn an interview answer into one resume bullet.

Use only what the answer says. Keep every number, tool and name it mentions and
add none it does not. Lead with an action verb; no first-person pronouns; one
sentence ending with a period. Use present tense if the role is current,
otherwise past tense. If the answer is too vague for a bullet, return it
cleaned up rather than embellished."""


class _Sorted(BaseModel):
    class Item(BaseModel):
        term: str
        disposition: Literal["reframe", "adjacent", "ask", "gap"]
        source_ids: list[str]
        rationale: str
        question: str

    items: list[Item]


class _Verdicts(BaseModel):
    class Verdict(BaseModel):
        term: str
        source_id: str
        supported: bool
        reason: str

    verdicts: list[Verdict]


class _Fact(BaseModel):
    text: str


# --- which terms need sorting -------------------------------------------------------

def _forms(term: str, aliases: list[str] = ()) -> list[str]:
    return list(dict.fromkeys([term, *lexicon.surface_forms(term), *aliases]))


def _supported(forms: list[str], text: str) -> bool:
    return any(lexicon._pattern(f).search(text) for f in forms if f)


def candidates(state: PipelineState) -> list[GapItem]:
    """Required and preferred terms (and exact phrases) the fact base never states."""
    text = master_text(state.master)
    out: dict[str, GapItem] = {}
    for kw in state.keywords:
        if kw.priority is Priority.MENTIONED or kw.category == "soft":
            continue
        if _supported(_forms(kw.term, kw.aliases), text):
            continue
        out[lexicon.normalize(kw.term)] = GapItem(term=kw.term, priority=kw.priority,
                                                  evidence=kw.evidence[:200])
    unsupported = set(phrase_mod.unsupported_terms(state.phrases, text))
    for ph in state.phrases:
        if ph.soft or ph.priority is Priority.MENTIONED or ph.text not in unsupported:
            continue
        key = lexicon.normalize(ph.text)
        if key in out or (ph.concept and lexicon.normalize(ph.concept) in out):
            continue
        out[key] = GapItem(term=ph.text, priority=ph.priority,
                           evidence=f"from the posting's {ph.source}")
    items = sorted(out.values(), key=lambda g: -g.priority.weight)
    return items[:MAX_CANDIDATES]


def _blocked(term: str, do_not_claim: list[str]) -> str | None:
    """The do-not-claim line that rules this term out, if any."""
    for line in do_not_claim:
        if any(phrase_mod.literal_in(f, line) for f in _forms(term)):
            return line
    return None


# --- the node -------------------------------------------------------------------------

def analyze(state: PipelineState, cfg: Config, llm: LLM) -> dict:
    if state.posting is None:
        return {}
    master = copy.deepcopy(state.master)
    log = list(state.log)
    items = candidates(state)
    if not items:
        log.append("gaps: every required term is already in your fact base")
        return {"gaps": [], "log": log}

    do_not_claim = [str(x) for x in master.get("do_not_claim") or []]
    open_items, said_no, ruled_out = [], set(), set()
    for g in items:
        line = _blocked(g.term, do_not_claim)
        if line and "interview mode" in line:
            # You said no before: never a reframe and never asked again, but
            # related experience can still be shown and named in the letter.
            said_no.add(g.term)
            open_items.append(g)
        elif line:
            ruled_out.add(g.term)             # you wrote it is not true: never ask
            g.disposition = "gap"
            g.rationale = f"your fact base rules it out: {line}"
        else:
            open_items.append(g)

    if open_items and not llm.offline:
        _sort(open_items, state, master, llm)
        rejected = _review(open_items, master, llm)
        if rejected:
            log.append(f"gaps: reviewer rejected {rejected} reframe(s) as overstated "
                       "-- treated as related experience instead")
    else:
        for g in open_items:
            g.disposition = "ask"
            g.question = _default_question(g)
    for g in open_items:
        if g.term in said_no:
            g.disposition = "adjacent" if g.disposition in ("reframe", "adjacent") else "declined"

    if cfg.ask is not None:
        askable = [g for g in items if g.term not in said_no | ruled_out]
        confirmed, declined = _interview(askable, master, cfg, llm)
        new = {"confirmed": confirmed, "declined": declined}
        if confirmed or declined:
            master = resume_source.apply_confirmed(master, new)
            if cfg.writable_fact_base:
                saved = resume_source.load_confirmed(cfg.confirmed_path)
                saved["confirmed"] += confirmed
                saved["declined"] += declined
                resume_source.save_confirmed(cfg.confirmed_path, saved)
        log.append(f"interview: {len(confirmed)} confirmed, {len(declined)} declined"
                   + (f" -- saved to {cfg.confirmed_path.name}"
                      if (confirmed or declined) and cfg.writable_fact_base else ""))

    # A reframe makes its term a keyword hint on the cited bullets. Hints are
    # what the scorer and the tailor treat as evidence, so this is the single
    # switch that turns "unattainable" into "attainable, from bullet X".
    bullets = _bullet_index(master)
    for g in items:
        if g.disposition == "reframe":
            for sid in g.source_ids:
                hints = bullets[sid].setdefault("keywords", [])
                if g.term not in hints:
                    hints.append(g.term)

    counts: dict[str, int] = {}
    for g in items:
        counts[g.disposition] = counts.get(g.disposition, 0) + 1
    log.append(f"gaps: {len(items)} requirement(s) not stated in your fact base -- "
               + ", ".join(f"{n} {d}" for d, n in sorted(counts.items())))
    return {"gaps": items, "master": master, "log": log}


def _sort(items: list[GapItem], state: PipelineState, master: dict, llm: LLM) -> None:
    posting = state.posting
    requirements = "\n".join(
        f"- {g.term} ({g.priority.value})" + (f' -- posting: "{g.evidence}"' if g.evidence else "")
        for g in items)
    payload = llm.complete_json(
        SORT_SYSTEM,
        SORT_USER.format(
            title=posting.title or "(not stated)", company=posting.company or "(not stated)",
            role_brief=_role_brief(posting, limit=3000), requirements=requirements,
            do_not_claim="\n".join(f"- {c}" for c in master.get("do_not_claim", []))
                         or "(none listed)",
            master=_render_master(master)),
        schema=_Sorted, max_tokens=12000, effort="high",
    ) or {}
    by_term = {str(i.get("term", "")).strip().lower(): i for i in payload.get("items", [])
               if isinstance(i, dict)}
    known = _bullet_index(master)
    for g in items:
        got = by_term.get(g.term.lower())
        if got is None:
            # The model skipped it: asking is the safe default.
            g.disposition, g.question = "ask", _default_question(g)
            continue
        g.disposition = got.get("disposition", "gap")
        g.rationale = str(got.get("rationale", "")).strip()
        g.source_ids = [s for s in got.get("source_ids") or [] if s in known]
        g.question = str(got.get("question", "")).strip()
        if g.disposition in ("reframe", "adjacent") and not g.source_ids:
            g.disposition = "ask"             # evidence that points at nothing
        if g.disposition == "ask" and not g.question:
            g.question = _default_question(g)


def _review(items: list[GapItem], master: dict, llm: LLM) -> int:
    """Second opinion on every reframe; returns how many were rejected."""
    known = _bullet_index(master)
    pairs = [(g, sid) for g in items if g.disposition == "reframe" for sid in g.source_ids]
    if not pairs:
        return 0
    listing = "\n".join(f'- skill: {g.term} | source_id: {sid} | bullet: "{known[sid]["text"]}"'
                        for g, sid in pairs)
    payload = llm.complete_json(VERIFY_SYSTEM, f"Review each pair:\n{listing}",
                                schema=_Verdicts, max_tokens=8000, effort="medium") or {}
    supported = {(str(v.get("term", "")).lower(), v.get("source_id"))
                 for v in payload.get("verdicts", [])
                 if isinstance(v, dict) and v.get("supported")}
    reasons = {(str(v.get("term", "")).lower(), v.get("source_id")): v.get("reason", "")
               for v in payload.get("verdicts", []) if isinstance(v, dict)}
    rejected = 0
    for g in items:
        if g.disposition != "reframe":
            continue
        kept = [sid for sid in g.source_ids if (g.term.lower(), sid) in supported]
        if kept:
            g.source_ids = kept
        else:
            # No verdict at all counts as a rejection: unreviewed is not approved.
            rejected += 1
            g.disposition = "adjacent"
            why = reasons.get((g.term.lower(), g.source_ids[0]), "")
            g.rationale = f"reviewer: {why}" if why else g.rationale
    return rejected


# --- interview mode -----------------------------------------------------------------------

_NO = {"no", "n", "nope", "never", "none", "not really"}
_SKIP = {"", "skip", "s", "pass", "?"}


def _default_question(g: GapItem) -> str:
    return (f"The posting asks for {g.term}. Have you done this -- at work, in a "
            "project, or a course? If so, what did you do?")


def _interview(items: list[GapItem], master: dict, cfg: Config,
               llm: LLM) -> tuple[list[dict], list[dict]]:
    entries = [(e["id"], " - ".join(x for x in (e.get("title") or e.get("name", ""),
                                                e.get("company") or e.get("subtitle", "")) if x))
               for group in ("experience", "projects") for e in master.get(group, []) or []]
    today = dt.date.today().isoformat()
    confirmed: list[dict] = []
    declined: list[dict] = []
    asked = 0
    for g in items:
        if g.disposition not in ("ask", "adjacent", "gap") or asked >= MAX_QUESTIONS:
            continue
        question = g.question or _default_question(g)
        answer = cfg.ask(f"{question}\n  (describe it, 'no', or Enter to skip) > ").strip()
        asked += 1
        low = answer.lower().strip(" .!")
        if low in _SKIP:
            continue
        g.answer = answer
        if low in _NO:
            # "No" to a related-experience item means "not directly": the
            # related experience is still real and still worth showing.
            if g.disposition != "adjacent":
                g.disposition = "declined"
            declined.append({"term": g.term, "date": today})
            continue
        menu = "\n".join(f"  {i}. {label}" for i, (_, label) in enumerate(entries, 1))
        choice = cfg.ask(f"Where was that?\n{menu}\n  0. Not tied to one role\n  number > ")
        choice = choice.strip()
        entry_id = (entries[int(choice) - 1][0]
                    if choice.isdigit() and 1 <= int(choice) <= len(entries) else "")
        entry = next((e for group in ("experience", "projects")
                      for e in master.get(group, []) or [] if e["id"] == entry_id), None)
        text = _bullet_from_answer(g, answer, entry, llm)
        fid = "confirmed_" + (re.sub(r"[^a-z0-9]+", "_", g.term.lower()).strip("_") or "fact")
        confirmed.append({"id": fid, "entry_id": entry_id, "term": g.term, "text": text,
                          "question": question, "answer": answer, "date": today})
        g.disposition = "confirmed"
        g.source_ids = [fid]
    return confirmed, declined


def _clean_answer(answer: str) -> str:
    text = re.sub(r"^(?:yes|yeah|yep|sure)\b[,.!]?\s*", "", answer.strip(), flags=re.I)
    text = re.sub(r"^i\s+", "", text, flags=re.I)
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _bullet_from_answer(g: GapItem, answer: str, entry: dict | None, llm: LLM) -> str:
    fallback = _clean_answer(answer)
    if llm.offline:
        return fallback
    current = bool(entry) and str(entry.get("end", "")).lower() in {"present", "current"}
    where = (f"{entry.get('title') or entry.get('name', '')} "
             f"({'current role' if current else 'past role'})") if entry else "no specific role"
    payload = llm.complete_json(
        FACT_SYSTEM,
        f"Requirement: {g.term}\nRole: {where}\nQuestion: {g.question}\nAnswer: {answer}",
        schema=_Fact, max_tokens=4000, effort="low") or {}
    text = str(payload.get("text", "")).strip()
    # The polished bullet may not introduce a number or a name the answer lacks.
    nums = set(re.findall(r"\d+(?:\.\d+)?", answer))
    names = {w.lower() for w in re.findall(r"[A-Za-z][\w.+#/-]*", answer)}
    added_num = any(n not in nums for n in re.findall(r"\d+(?:\.\d+)?", text))
    added_name = any(w.lower() not in names
                     for w in re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][\w.+#/-]+", text))
    return fallback if not text or added_num or added_name else text

"""Turn anyone's resume into a fact base.

    resume-tailor import my_resume.pdf --profile jane

The fact base used to be written by hand, which meant the tool only worked for
someone willing to type out a YAML file. Import does that first draft: the model
reads the PDF or Word file and returns the fact base as a validated structure,
including the ids, keyword hints and headline options a person would otherwise
write themselves.

The fact base is the one thing the tailor may claim about a person, so import is
held to the same rule as everything else -- nothing goes in that the original
does not say. Three checks run on the model's draft before it is saved:

1. **Every bullet and summary fact is found in the original** -- word for word,
   or nearly (line-wrap and punctuation differences), with every number present.
   Anything that fails moves to an `unverified:` list the tailor never reads.
2. **Keyword hints must be grounded in their bullet.** Hints count as evidence
   when the tailor decides what it may claim, so a hint like "Kubernetes" on a
   bullet that never mentions it would be a back door. A hint survives only if
   the bullet names it, or names something the lexicon says is the same thing.
3. **Contact details must appear in the original.** A guessed email is worse
   than a blank one.

Without an API key, the old rule-based parser does the draft instead; it is
weaker on unusual layouts but the same checks apply.
"""

from __future__ import annotations

import base64
import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ..ats import lexicon
from ..llm import LLM
from . import resume_source


# --- the shape the model returns ---------------------------------------------------

class _Bullet(BaseModel):
    text: str
    keywords: list[str]


class _Role(BaseModel):
    id: str
    title: str
    company: str
    location: str
    start: str
    end: str
    bullets: list[_Bullet]


class _Project(BaseModel):
    id: str
    name: str
    subtitle: str
    year: str
    bullets: list[_Bullet]


class _Degree(BaseModel):
    degree: str
    school: str
    dates: str


class _Cert(BaseModel):
    name: str
    issuer: str
    date: str


class _SkillGroup(BaseModel):
    label: str
    items: list[str]


class _Headline(BaseModel):
    text: str
    fits: list[str]


class _ExtraEntry(BaseModel):
    title: str
    subtitle: str
    dates: str
    bullets: list[str]


class _ExtraSection(BaseModel):
    heading: str
    entries: list[_ExtraEntry]


class _Contact(BaseModel):
    name: str
    phone: str
    email: str
    linkedin: str
    location: str


class ImportedResume(BaseModel):
    contact: _Contact
    headlines: list[_Headline]
    summary_facts: list[str]
    experience: list[_Role]
    projects: list[_Project]
    projects_heading: str
    education: list[_Degree]
    certifications: list[_Cert]
    skills: list[_SkillGroup]
    extra_sections: list[_ExtraSection]
    do_not_claim: list[str]


SYSTEM = """You convert a resume into a structured fact base for a resume-tailoring tool.

The fact base is the only thing the tool may ever claim about this person, so it
must contain exactly what the resume says -- nothing more, nothing improved.

- Copy every bullet word for word. The only edits allowed: join a line the PDF
  broke in two, drop the bullet glyph, repair a word hyphenated across a line
  break. Never reword, merge, summarise or polish a bullet.
- Keep every number, date, employer, title and tool exactly as written.
- Put each bullet under the role or project it appears under.
- id: a short, unique snake_case id per role and project ("acme" for Acme Corp).
- keywords: for each bullet, 3 to 8 short terms an applicant tracking system
  would match that the bullet itself demonstrates -- named in it, or a plain
  synonym of something named in it. Never a skill the bullet does not show.
- headlines: one to three lines shaped "Focus | Skill - Skill - Skill", built
  only from titles and skills the resume contains. fits: job-title words that
  each headline suits.
- summary_facts: the resume's summary split into independent statements, each
  in the resume's own words. Never compute a new fact ("8 years of experience")
  the resume does not state.
- skills: the resume's skill groups and items as written. A single flat list may
  be grouped sensibly, but keep every item's wording.
- extra_sections: anything else -- licenses, volunteering, awards, publications,
  languages -- as written.
- projects_heading: the resume's own heading for projects, or "PROJECTS".
- do_not_claim: things a reader might wrongly assume that the resume shows are
  not true -- a degree in progress is not completed, a certification "in
  progress" is not held, a contract role is not a full-time one. Empty if none.
- Use "" or [] for anything the resume does not contain. Never guess contact
  details."""


@dataclass
class ImportResult:
    master: dict[str, Any]
    drafted_by: str                       # "model" or "offline parser"
    verified: int = 0
    unverified: list[str] = field(default_factory=list)
    dropped_hints: int = 0
    packs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def import_resume(path: Path, llm: LLM) -> ImportResult:
    path = Path(path)
    try:
        text = resume_source.read_document(path)
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 - unreadable PDF: the model may still read it
        text = ""

    if llm.offline:
        if not text.strip():
            raise ValueError(f"could not read any text from {path.name}; without an API "
                             "key there is nothing to parse")
        raw = _from_parser(text)
        drafted_by = "offline parser"
    else:
        raw = llm.complete_json(SYSTEM, _user_content(path, text), schema=ImportedResume,
                                max_tokens=16000, effort="medium")
        drafted_by = "model"

    master = to_fact_base(raw or {})
    result = ImportResult(master=master, drafted_by=drafted_by)
    verify(result, text)
    result.packs = lexicon.suggest_packs(text or yaml.safe_dump(master))
    if result.packs:
        master["lexicon_packs"] = result.packs
    return result


def _user_content(path: Path, text: str):
    if path.suffix.lower() == ".pdf":
        # The PDF itself, not its extracted text: the model reads the layout,
        # which survives multi-column templates that scramble text extraction.
        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return [{"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
                {"type": "text", "text": "Convert this resume into the fact base."}]
    return f"Convert this resume into the fact base.\n\n<resume>\n{text}\n</resume>"


# --- building the fact base -----------------------------------------------------------

def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:24].strip("_")
    return s or fallback


def to_fact_base(raw: dict[str, Any]) -> dict[str, Any]:
    """The model's (or parser's) draft, reshaped into the fact-base layout with
    unique ids. Bullet ids are derived, never taken from the model."""
    used: set[str] = set()

    def unique(base: str) -> str:
        out, n = base, 2
        while out in used:
            out, n = f"{base}_{n}", n + 1
        used.add(out)
        return out

    def bullets(eid: str, items) -> list[dict]:
        out = []
        for i, b in enumerate(items or [], 1):
            if isinstance(b, str):
                b = {"text": b, "keywords": []}
            text = str(b.get("text", "")).strip()
            if text:
                out.append({"id": f"{eid}_{i}", "text": text,
                            "keywords": [str(k).strip() for k in b.get("keywords") or []
                                         if str(k).strip()]})
        return out

    master: dict[str, Any] = {}
    contact = raw.get("contact") or {}
    master["contact"] = {k: str(contact.get(k, "") or "").strip()
                         for k in ("name", "phone", "email", "linkedin", "location")}
    master["headlines"] = [
        {"id": f"h{i}", "text": str(h.get("text", "")).strip(),
         "fits": [str(f) for f in h.get("fits") or []]}
        for i, h in enumerate(raw.get("headlines") or [], 1) if str(h.get("text", "")).strip()
    ]
    master["summary_facts"] = [str(s).strip() for s in raw.get("summary_facts") or []
                               if str(s).strip()]

    master["experience"] = []
    for i, r in enumerate(raw.get("experience") or [], 1):
        eid = unique(_slug(r.get("id") or r.get("company") or r.get("title"), f"role{i}"))
        master["experience"].append({
            "id": eid, "title": str(r.get("title", "")).strip(),
            "company": str(r.get("company", "")).strip(),
            "location": str(r.get("location", "")).strip(),
            "start": str(r.get("start", "")).strip(), "end": str(r.get("end", "")).strip(),
            "tags": [], "bullets": bullets(eid, r.get("bullets")),
        })

    master["projects_heading"] = str(raw.get("projects_heading") or "PROJECTS").strip()
    master["projects"] = []
    for i, p in enumerate(raw.get("projects") or [], 1):
        pid = unique(_slug(p.get("id") or p.get("name"), f"project{i}"))
        master["projects"].append({
            "id": pid, "name": str(p.get("name", "")).strip(),
            "subtitle": str(p.get("subtitle", "")).strip(),
            "year": str(p.get("year", "")).strip(),
            "tags": [], "bullets": bullets(pid, p.get("bullets")),
        })

    master["education"] = [{k: str(e.get(k, "")).strip() for k in ("degree", "school", "dates")}
                           for e in raw.get("education") or []]
    master["certifications"] = [{k: str(c.get(k, "")).strip()
                                 for k in ("name", "issuer", "date")}
                                for c in raw.get("certifications") or []]

    skills: dict[str, Any] = {}
    for i, g in enumerate(raw.get("skills") or [], 1):
        items = [str(x).strip() for x in g.get("items") or [] if str(x).strip()]
        if items:
            key = _slug(g.get("label"), f"group{i}")
            while key in skills:
                key += "_"
            skills[key] = {"label": str(g.get("label", "Skills")).strip() or "Skills",
                           "items": items}
    master["skills"] = skills

    master["extra_sections"] = [
        {"heading": str(s.get("heading", "")).strip().upper(),
         "entries": [{"title": str(e.get("title", "")).strip(),
                      "subtitle": str(e.get("subtitle", "")).strip(),
                      "dates": str(e.get("dates", "")).strip(),
                      "bullets": [str(b).strip() for b in e.get("bullets") or [] if str(b).strip()]}
                     for e in s.get("entries") or []]}
        for s in raw.get("extra_sections") or [] if s.get("entries")
    ]
    master["do_not_claim"] = [str(x).strip() for x in raw.get("do_not_claim") or []
                              if str(x).strip()]
    return master


def _from_parser(text: str) -> dict[str, Any]:
    """The rule-based parser's output, in the same shape the model returns."""
    parsed = resume_source.parse_resume_text(text)

    def hinted(entry_bullets):
        return [{"text": b["text"],
                 "keywords": [t.lower() for t in lexicon.find_terms(b["text"])]}
                for b in entry_bullets]

    extra = [{"heading": key, "entries": [{"title": line, "subtitle": "", "dates": "",
                                          "bullets": []} for line in lines]}
             for key, lines in (parsed.get("unparsed") or {}).items() if lines]
    return {
        "contact": parsed.get("contact", {}),
        "headlines": [{"text": h["text"], "fits": []} for h in parsed.get("headlines", [])],
        "summary_facts": parsed.get("summary_facts", []),
        "experience": [{**e, "bullets": hinted(e.get("bullets", []))}
                       for e in parsed.get("experience", [])],
        "projects": [{**p, "bullets": hinted(p.get("bullets", []))}
                     for p in parsed.get("projects", [])],
        "projects_heading": "PROJECTS",
        "education": parsed.get("education", []),
        "certifications": [],
        "skills": [{"label": g["label"], "items": g["items"]}
                   for g in (parsed.get("skills") or {}).values()],
        "extra_sections": extra,
        "do_not_claim": [],
    }


# --- verification against the original ------------------------------------------------

def _norm(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)          # hyphen at a line break
    text = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9%+#.' ]+", " ", text)
    text = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class _Source:
    norm: str
    tokens: set[str]
    numbers: set[str]
    digits: str
    terms: set[str]

    @classmethod
    def of(cls, text: str) -> "_Source":
        norm = _norm(text)
        return cls(norm, set(norm.split()), set(_NUMBER.findall(text.replace(",", ""))),
                   re.sub(r"\D", "", text), set(lexicon.find_terms(text)))


def found_in(candidate: str, src: _Source) -> bool:
    """Is `candidate` in the original, allowing for line wraps and punctuation?"""
    c = _norm(candidate)
    if not c:
        return False
    if c in src.norm:
        return True
    if any(n not in src.numbers for n in _NUMBER.findall(candidate.replace(",", ""))):
        return False                      # a number the original never states
    words = [w for w in c.split() if len(w) > 2]
    if not words:
        return False
    return sum(w in src.tokens for w in words) / len(words) >= 0.85


def grounded(hint: str, bullet: str) -> bool:
    """A keyword hint the bullet itself backs up."""
    h = lexicon.normalize(hint)
    b = lexicon.normalize(bullet)
    if not h:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(h)}(?![a-z0-9])", b):
        return True
    for term in lexicon.find_terms(bullet):
        if h in {lexicon.normalize(f) for f in lexicon.surface_forms(term)}:
            return True
    words = [w for w in h.split() if len(w) >= 4]
    b_words = b.split()
    return bool(words) and all(any(bw.startswith(w[:5]) for bw in b_words) for w in words)


def verify(result: ImportResult, text: str) -> None:
    master = result.master
    if not text.strip():
        result.warnings.append(
            "could not read the text layer of the original (a scanned PDF?), so the "
            "import could not be checked against it -- read the fact base carefully")
        for group in ("experience", "projects"):
            for entry in master.get(group, []):
                for b in entry["bullets"]:
                    result.dropped_hints += _ground_hints(b)
        return
    src = _Source.of(text)
    unverified: list[dict[str, str]] = []

    for group in ("experience", "projects"):
        for entry in master.get(group, []):
            kept = []
            for b in entry["bullets"]:
                if found_in(b["text"], src):
                    result.dropped_hints += _ground_hints(b)
                    kept.append(b)
                    result.verified += 1
                else:
                    unverified.append({"where": entry["id"], "text": b["text"]})
            entry["bullets"] = kept

    facts = []
    for fact in master.get("summary_facts", []):
        if found_in(fact, src):
            facts.append(fact)
            result.verified += 1
        else:
            unverified.append({"where": "summary_facts", "text": fact})
    master["summary_facts"] = facts

    for key, group in list(master.get("skills", {}).items()):
        items = []
        for item in group["items"]:
            if found_in(item, src) or set(lexicon.find_terms(item)) & src.terms:
                items.append(item)
            else:
                unverified.append({"where": f"skills.{key}", "text": item})
        group["items"] = items
        if not items:
            del master["skills"][key]

    headlines = []
    for h in master.get("headlines", []):
        extra_terms = set(lexicon.find_terms(h["text"])) - src.terms
        if extra_terms:
            result.warnings.append(f"dropped headline naming {', '.join(sorted(extra_terms))}, "
                                   "which the original does not mention")
        else:
            headlines.append(h)
    master["headlines"] = headlines

    contact = master.get("contact", {})
    for key in ("email", "linkedin", "name"):
        value = contact.get(key, "")
        if value and _norm(value) not in src.norm:
            result.warnings.append(f"cleared contact {key} '{value}': not in the original")
            contact[key] = ""
    phone_digits = re.sub(r"\D", "", contact.get("phone", ""))
    if phone_digits and phone_digits[-10:] not in src.digits:
        result.warnings.append("cleared the phone number: not in the original")
        contact["phone"] = ""

    if unverified:
        master["unverified"] = unverified
        result.unverified = [f"{u['where']}: {u['text']}" for u in unverified]


def _ground_hints(bullet: dict) -> int:
    before = bullet.get("keywords", [])
    bullet["keywords"] = [k for k in before if grounded(k, bullet["text"])]
    return len(before) - len(bullet["keywords"])


# --- writing it out -----------------------------------------------------------------------

def header(source: Path, drafted_by: str) -> str:
    today = dt.date.today().isoformat()
    return f"""# Fact base imported from {source.name} on {today} ({drafted_by}).
#
# This file is the only thing resume-tailor may claim about you. Read it once
# before your first run:
#   - every bullet should be something you did, worded as you would defend it
#   - `unverified` lists anything the import could not find in the original;
#     move each item into the right place if it is correct, otherwise delete
#     it -- nothing there is used while it sits there
#   - add to do_not_claim anything a posting might tempt the tool to imply
#   - add true facts that are not on your resume yet: the tailor can only use
#     what is here, and a fact you leave out is a gap it cannot close
"""


def save(result: ImportResult, path: Path, source: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header(source, result.drafted_by) + "\n")
        yaml.safe_dump(result.master, fh, sort_keys=False, allow_unicode=True, width=100)
    return path

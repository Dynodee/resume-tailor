"""Typed state that flows through the graph.

Every node takes the state and returns a partial update. Nothing else is
shared between nodes -- if a value is not in here, downstream nodes cannot
see it. That constraint is what makes the pipeline debuggable: dump the state
after any node and you have the complete picture at that point.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """How badly the posting wants a term.

    REQUIRED terms come out of a "must have"/"requirements" block or a
    minimum-qualifications list. PREFERRED comes from "nice to have". MENTIONED
    is everything else the posting says.
    """

    REQUIRED = "required"
    PREFERRED = "preferred"
    MENTIONED = "mentioned"

    @property
    def weight(self) -> float:
        return {"required": 3.0, "preferred": 1.5, "mentioned": 1.0}[self.value]


class JobPosting(BaseModel):
    """A posting after fetching and cleaning, before any interpretation."""

    url: str | None = None
    source: str = "unknown"          # greenhouse / lever / ashby / generic / paste
    title: str = ""
    company: str = ""
    location: str = ""
    raw_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)

    def char_count(self) -> int:
        return len(self.raw_text)


class Keyword(BaseModel):
    """One ATS term the posting cares about."""

    term: str                         # canonical surface form, e.g. "Python"
    normalized: str                   # lowercased match key, e.g. "python"
    aliases: list[str] = Field(default_factory=list)
    category: str = "other"           # skill / tool / credential / title / soft
    priority: Priority = Priority.MENTIONED
    frequency: int = 1
    evidence: str = ""                # the posting line it came from

    @property
    def score(self) -> float:
        # Frequency helps, but with sharply diminishing returns -- a term
        # repeated eight times is not eight times as important as one said once.
        return self.priority.weight * (1.0 + min(self.frequency - 1, 4) * 0.25)


class ExactPhrase(BaseModel):
    """A phrase exactly as the posting writes it.

    Keyword coverage asks "does the resume mention this skill in any wording?".
    A recruiter typing into the applicant tracking system's search box asks a
    stricter question: "does the resume contain these words?". "Dashboards" and
    "data visualization" are the same skill to the first and a miss to the second.
    """

    text: str                         # verbatim from the posting, e.g. "Data Build Tool"
    priority: Priority = Priority.MENTIONED
    source: str = ""                  # "skills list" or the section it came from
    concept: str | None = None        # canonical lexicon term, e.g. "dbt"
    soft: bool = False

    @property
    def weight(self) -> float:
        return self.priority.weight


class PhraseItem(BaseModel):
    phrase: ExactPhrase
    # exact       -- the words appear on the page
    # concept     -- the skill is on the page in other words ("dbt" for "Data Build Tool")
    # missing     -- not on the page, but the fact base supports it
    # unsupported -- nothing in the fact base backs it
    status: Literal["exact", "concept", "missing", "unsupported"]


class CoverageItem(BaseModel):
    keyword: Keyword
    covered: bool
    where: list[str] = Field(default_factory=list)   # section names that hit it


class CoverageReport(BaseModel):
    items: list[CoverageItem] = Field(default_factory=list)
    score: float = 0.0                # 0-100, weighted by priority
    required_score: float = 0.0       # 0-100, REQUIRED terms only
    # Required terms the fact base can actually support. This is what the
    # revision loop steers on: chasing a term you have never used ("Microsoft
    # Fabric") can only end in a false claim or in keyword stuffing.
    attainable_score: float = 100.0
    unattainable: list[str] = Field(default_factory=list)
    # Literal phrase coverage, over the phrases the fact base can support.
    phrase_items: list[PhraseItem] = Field(default_factory=list)
    phrase_score: float = 100.0

    def phrases(self, status: str) -> list[ExactPhrase]:
        return [i.phrase for i in self.phrase_items if i.status == status]

    def missing(self, priority: Priority | None = None) -> list[Keyword]:
        out = [i.keyword for i in self.items if not i.covered]
        if priority is not None:
            out = [k for k in out if k.priority is priority]
        return sorted(out, key=lambda k: -k.score)

    def covered(self) -> list[Keyword]:
        return [i.keyword for i in self.items if i.covered]


class ResumeBullet(BaseModel):
    """A single tailored bullet, with the master fact it came from."""

    source_id: str                    # bullet id in master_resume.yaml
    text: str


class ResumeSection(BaseModel):
    heading: str
    kind: Literal["summary", "experience", "projects", "education", "skills"]
    entries: list[dict[str, Any]] = Field(default_factory=list)


class TailoredResume(BaseModel):
    """The tailored document as structured data, not as a blob of text.

    Keeping it structured means the grammar pass can edit a single bullet
    without reflowing the document, and the renderer never has to parse
    anything back out of prose.
    """

    headline: str = ""
    summary: str = ""
    sections: list[ResumeSection] = Field(default_factory=list)
    # Character spans to bold, keyed by unit ("summary", "s0e1b2"). Set by the
    # render node just before drawing; it never changes the words.
    emphasis: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)

    def all_text(self) -> str:
        parts = [self.headline, self.summary]
        for s in self.sections:
            parts.append(s.heading)
            for e in s.entries:
                parts.append(" ".join(str(v) for k, v in e.items() if k != "bullets"))
                for b in e.get("bullets", []):
                    parts.append(b["text"] if isinstance(b, dict) else str(b))
        return "\n".join(p for p in parts if p)


class GrammarIssue(BaseModel):
    location: str                     # human-readable path, e.g. "experience[0].bullets[2]"
    rule: str                         # rule id, e.g. "tense-consistency"
    severity: Literal["error", "warning", "style"] = "warning"
    message: str
    excerpt: str = ""
    suggestion: str = ""


class PipelineState(BaseModel):
    """Everything the graph carries from start to finish."""

    # inputs
    job_url: str | None = None
    job_text: str | None = None       # paste path, bypasses the fetcher
    resume_ref: str | None = None     # path or Drive file id, backend decides

    # stage outputs
    posting: JobPosting | None = None
    keywords: list[Keyword] = Field(default_factory=list)
    phrases: list[ExactPhrase] = Field(default_factory=list)
    master: dict[str, Any] = Field(default_factory=dict)
    resume: TailoredResume | None = None
    coverage: CoverageReport | None = None
    grammar_issues: list[GrammarIssue] = Field(default_factory=list)
    pdf_path: str | None = None

    # control
    revisions: int = 0
    max_revisions: int = 2
    coverage_target: float = 80.0     # stop revising once required coverage clears this
    phrase_target: float = 70.0       # ...and once exact-phrase coverage clears this
    log: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def note(self, msg: str) -> None:
        self.log.append(msg)

"""The change report: what tailoring actually did to the resume.

Written next to every PDF so the question "did it change anything?" has an
answer you can read instead of having to diff two PDFs by eye. It shows, per
bullet, the fact-base wording against what went on the page, which posting
keywords each edit picked up, which facts were left out, and which required
terms the fact base cannot support at all.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from .ats import scorer
from .state import Keyword, PipelineState, ResumeSection, TailoredResume

UNCHANGED, EDITED, REWRITTEN = "unchanged", "edited", "rewritten"


@dataclass
class BulletChange:
    source_id: str
    entry: str
    before: str
    after: str
    similarity: float
    gained: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.similarity >= 0.97:
            return UNCHANGED
        if self.similarity >= 0.75:
            return EDITED
        return REWRITTEN


@dataclass
class ChangeSet:
    bullets: list[BulletChange]
    left_out: list[tuple[str, str, str]]          # (entry, source_id, text)
    baseline_score: float
    baseline_attainable: float

    def counts(self) -> dict[str, int]:
        out = {UNCHANGED: 0, EDITED: 0, REWRITTEN: 0}
        for b in self.bullets:
            out[b.status] += 1
        return out

    def one_line(self) -> str:
        c = self.counts()
        return (f"{len(self.bullets)} bullets on the page: {c[UNCHANGED]} unchanged, "
                f"{c[EDITED]} edited, {c[REWRITTEN]} rewritten; "
                f"{len(self.left_out)} left out")


def _terms_in(text: str, keywords: list[Keyword]) -> set[str]:
    probe = TailoredResume(summary=text)
    return {i.keyword.term for i in scorer.score(probe, keywords).items if i.covered}


def baseline(master: dict) -> TailoredResume:
    """The fact base exactly as written: every bullet, every skill, no edits."""
    sections = []
    for kind, heading in (("experience", "WORK EXPERIENCE"), ("projects", "PROJECTS")):
        sections.append(ResumeSection(heading=heading, kind=kind, entries=[
            {**{k: v for k, v in e.items() if k != "bullets"},
             "bullets": [{"source_id": b["id"], "text": b["text"]}
                         for b in e.get("bullets", [])]}
            for e in master.get(kind, [])
        ]))
    sections.append(ResumeSection(heading="SKILLS", kind="skills", entries=[
        {"label": g.get("label", ""), "items": g.get("items", [])}
        for g in (master.get("skills") or {}).values()
    ]))
    headline = (master.get("headlines") or [{}])[0].get("text", "")
    return TailoredResume(headline=headline, sections=sections)


def compute(resume: TailoredResume, master: dict, keywords: list[Keyword],
            master_text: str | None = None) -> ChangeSet:
    facts: dict[str, tuple[str, str]] = {}
    for kind in ("experience", "projects"):
        for e in master.get(kind, []):
            label = e.get("title") or e.get("name", "")
            if e.get("company"):
                label = f"{label}, {e['company']}"
            for b in e.get("bullets", []):
                facts[b["id"]] = (label, b["text"])

    changes: list[BulletChange] = []
    used: set[str] = set()
    for section in resume.sections:
        if section.kind not in ("experience", "projects"):
            continue
        for e in section.entries:
            for b in e.get("bullets", []):
                sid = b.get("source_id", "") if isinstance(b, dict) else ""
                after = b["text"] if isinstance(b, dict) else str(b)
                entry, before = facts.get(sid, ("(unknown)", ""))
                used.add(sid)
                ratio = difflib.SequenceMatcher(None, before.lower(), after.lower()).ratio()
                gained = sorted(_terms_in(after, keywords) - _terms_in(before, keywords))
                changes.append(BulletChange(sid, entry, before, after, round(ratio, 2), gained))

    left_out = [(entry, sid, text) for sid, (entry, text) in facts.items() if sid not in used]
    base = scorer.score(baseline(master), keywords, master_text)
    return ChangeSet(changes, left_out, base.score, base.attainable_score)


def write(state: PipelineState, changes: ChangeSet, path: Path) -> Path:
    cov = state.coverage
    posting = state.posting
    title = " at ".join(x for x in (getattr(posting, "title", ""),
                                     getattr(posting, "company", "")) if x)
    lines = [f"# What changed{' -- ' + title if title else ''}", ""]

    if cov:
        lines += [
            "## Keyword coverage",
            "",
            f"- Tailored resume: {cov.score}% overall, "
            f"{cov.attainable_score}% of the required terms your fact base supports",
            f"- Fact base as written (every bullet, no edits): {changes.baseline_score}% "
            f"overall, {changes.baseline_attainable}% of supported required terms",
        ]
        if cov.unattainable:
            lines += [
                f"- Required by the posting but not in your fact base: "
                f"{', '.join(cov.unattainable)}",
                "  - If any of these are true for you, add them to "
                "data/master_resume.yaml and re-run. If not, they are cover-letter or "
                "interview topics -- never put them on the page.",
            ]
        lines.append("")

    if cov and cov.phrase_items:
        lines += _phrase_section(cov)

    lines += ["## Bullets", "", changes.one_line() + ".", ""]
    current = None
    for b in changes.bullets:
        if b.entry != current:
            lines += [f"### {b.entry}", ""]
            current = b.entry
        gained = f" -- picked up: {', '.join(b.gained)}" if b.gained else ""
        lines.append(f"- **{b.status.upper()}** `{b.source_id}`{gained}")
        if b.status == UNCHANGED:
            lines.append(f"  - {b.after}")
        else:
            lines.append(f"  - before: {b.before}")
            lines.append(f"  - after:  {b.after}")
        lines.append("")

    if changes.left_out:
        lines += ["## Left out for this posting", ""]
        for entry, sid, text in changes.left_out:
            lines.append(f"- `{sid}` ({entry}): {text}")
        lines.append("")

    if state.resume and state.resume.summary:
        lines += ["## Summary as written", "", state.resume.summary, ""]

    if state.resume and state.resume.emphasis:
        from .render.emphasis import bolded_terms

        lines += ["## Bolded on the page", "",
                  "Posting terms highlighted for the recruiter skimming after the screen "
                  "(the ATS ignores bold): " + ", ".join(bolded_terms(state.resume)) + ".",
                  "Turn off with --no-bold.", ""]

    remaining = [i for i in state.grammar_issues if i.severity in ("error", "warning")]
    if remaining:
        lines += ["## Writing checks still flagged", ""]
        for i in remaining:
            lines.append(f"- [{i.rule}] {i.location}: {i.message}")
            if i.excerpt:
                lines.append(f"  > {i.excerpt}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_PHRASE_LABELS = (
    ("exact", "On the page word for word"),
    ("concept", "On the page in other words only -- a search for the posting's wording misses these"),
    ("missing", "Supported by your fact base but not on the page"),
    ("unsupported", "Not in your fact base -- add any that are true to master_resume.yaml"),
)


def _phrase_section(cov) -> list[str]:
    from .ats.phrases import summary_line

    out = ["## Exact-phrase coverage", "",
           "Recruiters search the applicant tracking system for these words. "
           f"{summary_line(cov)}.", ""]
    for status, label in _PHRASE_LABELS:
        group = [i.phrase for i in cov.phrase_items if i.status == status]
        if not group:
            continue
        out.append(f"**{label}**")
        out.append("")
        for ph in group:
            note = " (soft skill -- not scored)" if ph.soft else ""
            src = f", from the {ph.source}" if ph.source else ""
            out.append(f"- {ph.text}{note}  _{ph.priority.value}{src}_")
        out.append("")
    return out


__all__ = ["BulletChange", "ChangeSet", "baseline", "compute", "write"]

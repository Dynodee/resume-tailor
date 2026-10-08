"""Node 6 -- write the PDF and name it something a recruiter can file."""

from __future__ import annotations

import re
from pathlib import Path

from ..config import Config
from .. import report
from ..render import emphasis, pdf
from ..state import PipelineState


def _slug(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")
    return (cleaned or fallback)[:48]


def render(state: PipelineState, cfg: Config) -> dict:
    if state.resume is None:
        return {"errors": state.errors + ["render: no resume in state"]}

    posting = state.posting
    name = _slug(state.master.get("contact", {}).get("name", "Resume"), "Resume")
    company = _slug(getattr(posting, "company", "") or "", "")
    role = _slug(getattr(posting, "title", "") or "", "Tailored")
    parts = [p for p in (name, company, role) if p]
    out_path = cfg.out_dir / ("_".join(parts) + ".pdf")

    resume = state.resume

    def draw():
        resume.emphasis = (emphasis.plan(resume, state.keywords, state.phrases)
                           if cfg.bold else {})
        return pdf.render(resume, state.master, out_path, max_pages=cfg.max_pages)

    path, pages = draw()
    # Still over at the smallest readable type: cut the least relevant bullet
    # (the tailor orders each entry most-relevant first) and draw again.
    cut: list[str] = []
    while pages > cfg.max_pages and (victim := _least_relevant(resume)):
        cut.append(victim)
        path, pages = draw()
    log = list(state.log)
    log.append(f"render: {path.name} ({pages} page{'s' if pages != 1 else ''}), "
               f"extraction check at {path.with_suffix('.txt').name}")
    if cut:
        log.append(f"render: cut {len(cut)} bullet(s) to fit {cfg.max_pages} page(s): "
                   + ", ".join(cut))
    from .tailor import master_text

    changes = report.compute(state.resume, state.master, state.keywords,
                             master_text(state.master))
    changes_path = report.write(state, changes, path.with_name(path.stem + "_changes.md"))
    log.append(f"changes: {changes.one_line()} -- see {changes_path.name}")
    if resume.emphasis:
        spans = sum(len(v) for v in resume.emphasis.values())
        log.append(f"bold: {spans} posting term(s) highlighted in "
                   f"{len(resume.emphasis)} line(s): "
                   + ", ".join(emphasis.bolded_terms(resume)))

    errors = list(state.errors)
    if pages > cfg.max_pages:
        # The renderer stops shrinking at a readable size rather than going
        # smaller; past that, fitting is a content decision, not a layout one.
        errors.append(f"layout: {pages} pages, over the {cfg.max_pages}-page limit at "
                      "the smallest readable size -- cut a bullet or two")
    return {"pdf_path": str(path), "resume": resume, "log": log, "errors": errors}


def _least_relevant(resume) -> str | None:
    """Remove and return the id of the bullet that matters least.

    Projects go first, then the oldest roles; within an entry the last bullet,
    since the tailor puts the most relevant first. Every entry keeps at least
    one bullet (two for the most recent role), so cutting never erases a job.
    """
    sections = sorted((s for s in resume.sections if s.kind in ("experience", "projects")),
                      key=lambda s: 0 if s.kind == "projects" else 1)
    for section in sections:
        for ei in reversed(range(len(section.entries))):
            entry = section.entries[ei]
            keep = 2 if section.kind == "experience" and ei == 0 else 1
            bullets = entry.get("bullets", [])
            if len(bullets) > keep:
                gone = bullets.pop()
                return gone.get("source_id", "?") if isinstance(gone, dict) else "?"
    return None

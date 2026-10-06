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
    resume.emphasis = (emphasis.plan(resume, state.keywords, state.phrases)
                       if cfg.bold else {})

    path, pages = pdf.render(resume, state.master, out_path,
                             max_pages=cfg.max_pages)
    log = list(state.log)
    log.append(f"render: {path.name} ({pages} page{'s' if pages != 1 else ''}), "
               f"extraction check at {path.with_suffix('.txt').name}")
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

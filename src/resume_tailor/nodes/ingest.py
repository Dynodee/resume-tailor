"""Node 1 -- turn a URL (or pasted text) into a clean JobPosting."""

from __future__ import annotations

from ..config import Config
from ..sources import job_posting
from ..state import PipelineState


def ingest(state: PipelineState, cfg: Config) -> dict:
    if state.job_text:
        posting = job_posting.from_text(state.job_text)
    elif state.job_url:
        posting = job_posting.fetch(state.job_url, timeout=cfg.timeout_s)
    else:
        return {"errors": state.errors + ["no job_url or job_text supplied"]}

    if posting.char_count() > cfg.max_posting_chars:
        posting.raw_text = posting.raw_text[: cfg.max_posting_chars]
        posting.sections = job_posting.split_sections(posting.raw_text)

    log = list(state.log)
    log.append(
        f"ingest: {posting.source} | {posting.char_count()} chars | "
        f"{len(posting.sections)} section(s) | title={posting.title or '?'}"
    )

    errors = list(state.errors)
    if posting.char_count() < 400:
        # Almost always a client-rendered page that returned a shell. Better to
        # say so than to tailor against 200 characters of boilerplate.
        errors.append(
            "posting text looks too short to be the real description "
            f"({posting.char_count()} chars) -- the page may be JavaScript-rendered. "
            "Re-run with --text and paste the description."
        )

    return {"posting": posting, "log": log, "errors": errors}

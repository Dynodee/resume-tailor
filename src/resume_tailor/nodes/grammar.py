"""Node 5 -- review the grammar, then actually apply the fixes.

Reviewing and revising are separate steps on purpose. The review is a critique
pass that names each problem and where it is; the revision applies only the
edits that came back with a concrete replacement, and re-lints afterwards so the
report reflects the document that will be rendered, not the one that went in.

Edits are applied per bullet rather than by regenerating the document, because a
regeneration can quietly undo the keyword work done upstream.
"""

from __future__ import annotations

from ..ats import linter
from ..config import Config
from ..llm import LLM
from ..state import GrammarIssue, PipelineState, TailoredResume

SYSTEM = """You are a meticulous copy editor for resumes.

Correct grammar, spelling, punctuation, tense, parallel structure, and awkward
phrasing. Preserve meaning exactly: do not add, remove, or soften any claim,
number, tool name, or company name. Preserve the existing vocabulary -- the
wording has been chosen to match a keyword screen, so replacing a term with a
synonym is a regression, not an improvement.

Resume conventions that are correct and must not be "fixed":
- Bullets are verb-led fragments, not full sentences with a subject.
- No first-person pronouns.
- Present tense for the current role, past tense for previous roles.
- Terminal periods on every bullet.

Return only the items you are actually changing."""

USER = """Automated checks flagged these (they may be incomplete or wrong -- use
your judgement):
{lint}

Here is the document, one editable unit per line, keyed by id:

{units}

Return JSON: a list of objects with keys
  id       (the unit id exactly as given)
  text     (the corrected text, complete -- not a diff)
  reason   (short, what you fixed)

Omit any unit you are not changing."""


def _units(resume: TailoredResume) -> dict[str, str]:
    """Addressable text units. Ids are stable within a single pass."""
    units: dict[str, str] = {}
    if resume.headline:
        units["headline"] = resume.headline
    if resume.summary:
        units["summary"] = resume.summary
    for si, section in enumerate(resume.sections):
        if section.kind == "other":
            continue                      # extra sections are shown as written
        for ei, entry in enumerate(section.entries):
            for bi, bullet in enumerate(entry.get("bullets", [])):
                text = bullet["text"] if isinstance(bullet, dict) else str(bullet)
                units[f"s{si}e{ei}b{bi}"] = text
    return units


def _apply(resume: TailoredResume, edits: dict[str, str]) -> int:
    applied = 0
    if "headline" in edits:
        resume.headline = edits["headline"]
        applied += 1
    if "summary" in edits:
        resume.summary = edits["summary"]
        applied += 1
    for si, section in enumerate(resume.sections):
        for ei, entry in enumerate(section.entries):
            for bi, bullet in enumerate(entry.get("bullets", [])):
                key = f"s{si}e{ei}b{bi}"
                if key not in edits:
                    continue
                if isinstance(bullet, dict):
                    bullet["text"] = edits[key]
                else:
                    entry["bullets"][bi] = edits[key]
                applied += 1
    return applied


def _mechanical_fixes(resume: TailoredResume) -> int:
    """Fixes that need no judgement at all -- do them before spending a call."""
    import re

    def fix(text: str) -> str:
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("—", " - ").replace("–", "-")
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = text.strip()
        if text and not text.endswith((".", ")", ":")):
            text += "."
        return text

    changed = 0
    for section in resume.sections:
        if section.kind == "other":
            continue
        for entry in section.entries:
            for bullet in entry.get("bullets", []):
                if not isinstance(bullet, dict):
                    continue
                new = fix(bullet["text"])
                if new != bullet["text"]:
                    bullet["text"] = new
                    changed += 1
    if resume.summary:
        new = fix(resume.summary)
        if new != resume.summary:
            resume.summary = new
            changed += 1
    return changed


def review_and_revise(state: PipelineState, cfg: Config, llm: LLM) -> dict:
    resume = state.resume
    log = list(state.log)
    if resume is None:
        return {"errors": state.errors + ["grammar: no resume in state"]}

    mechanical = _mechanical_fixes(resume)
    issues = linter.lint(resume)
    log.append(f"grammar: {mechanical} mechanical fix(es), "
               f"{len(issues)} issue(s) from the rule checks")

    applied = 0
    if not llm.offline:
        units = _units(resume)
        lint_text = "\n".join(
            f"- [{i.severity}] {i.rule} at {i.location}: {i.message}"
            + (f"\n    > {i.excerpt}" if i.excerpt else "")
            for i in issues
        ) or "(no automated findings)"
        unit_text = "\n".join(f"[{k}] {v}" for k, v in units.items())
        payload = llm.complete_json(
            SYSTEM, USER.format(lint=lint_text, units=unit_text), max_tokens=12000,
            effort="medium",
        )
        edits: dict[str, str] = {}
        for item in payload or []:
            if isinstance(item, dict) and item.get("id") in units and item.get("text"):
                text = str(item["text"]).strip()
                if text and text != units[item["id"]]:
                    edits[str(item["id"])] = text
        applied = _apply(resume, edits)
        _mechanical_fixes(resume)
        log.append(f"grammar: applied {applied} copy edit(s) from the review pass")

    remaining = linter.lint(resume)
    fixed = len(issues) - len(remaining)
    log.append(
        f"grammar: {len(remaining)} issue(s) remain"
        + (f" ({fixed} resolved this pass)" if fixed > 0 else "")
    )
    return {"resume": resume, "grammar_issues": remaining, "log": log}


def should_revise(state: PipelineState) -> str:
    """Router: back to tailoring, or on to rendering.

    Bounded by max_revisions so a posting asking for things the fact base cannot
    support terminates instead of looping. Errors (severity "error") count too --
    a document with real grammar errors is worth one more pass even at good
    coverage.
    """
    if state.revisions >= state.max_revisions:
        return "render"
    coverage_short = (
        state.coverage is not None
        and state.coverage.attainable_score < state.coverage_target
    )
    phrases_short = (
        state.coverage is not None
        and state.coverage.phrase_items
        and state.coverage.phrase_score < state.phrase_target
    )
    hard_errors = [i for i in state.grammar_issues if i.severity == "error"]
    if coverage_short or phrases_short or hard_errors:
        return "revise"
    return "render"


def bump_revision(state: PipelineState) -> dict:
    log = list(state.log)
    reasons = []
    if state.coverage and state.coverage.attainable_score < state.coverage_target:
        reasons.append(f"attainable required coverage {state.coverage.attainable_score}% "
                       f"< {state.coverage_target}%")
    if (state.coverage and state.coverage.phrase_items
            and state.coverage.phrase_score < state.phrase_target):
        reasons.append(f"exact-phrase coverage {state.coverage.phrase_score}% "
                       f"< {state.phrase_target}%")
    errs = [i for i in state.grammar_issues if i.severity == "error"]
    if errs:
        reasons.append(f"{len(errs)} grammar error(s)")
    log.append(f"revise: pass {state.revisions + 1} because " + "; ".join(reasons))
    return {"revisions": state.revisions + 1, "log": log}


def issues_to_text(issues: list[GrammarIssue]) -> str:
    return linter.summarize(issues)

"""Deterministic resume grammar and style checks.

These are the errors a general-purpose grammar checker misses because resume
bullets are not sentences: fragments are correct here, first person is wrong
here, and past-vs-present tense is decided by whether you still hold the job.

Running these before the LLM grammar pass means the model spends its attention
on real prose problems instead of re-deriving the same mechanical rules.
"""

from __future__ import annotations

import re

from ..state import GrammarIssue, TailoredResume

# Case matters here. Bare "I" is first person; the "I" inside "EXPERIENCE" is
# not, and "US" is a country while "us" is a pronoun.
FIRST_PERSON = re.compile(
    r"(?<![A-Za-z])(I|[Ww]e|[Mm]y|[Mm]e|[Mm]ine|[Oo]ur|[Oo]urs|us)(?![A-Za-z])"
)
WEAK_OPENERS = {
    "responsible for", "helped", "assisted with", "worked on", "involved in",
    "tasked with", "participated in", "duties included", "in charge of",
}
FILLER = {
    "very", "really", "various", "several", "numerous", "a number of",
    "utilize", "utilized", "leverage synergies", "team player", "detail-oriented",
    "hard worker", "go-getter", "think outside the box", "results-driven",
}
# Common past-tense verbs that signal a bullet is written in past tense.
PAST_VERB = re.compile(r"^(\w+ed|built|wrote|led|ran|drove|made|took|won|grew|"
                       r"cut|sold|kept|taught|spoke|chose|set|met|sent|held|"
                       r"began|brought|found|gave|showed)\b", re.IGNORECASE)
PRESENT_VERB = re.compile(r"^(build|write|lead|run|drive|make|take|own|manage|"
                          r"design|deliver|partner|translate|maintain|support|"
                          r"develop|create|analyze|present|coordinate)\b", re.IGNORECASE)
# Signs that a keyword was forced in rather than written in.
SOFT_NAME_DROP = re.compile(
    r"\b(?:appl(?:y|ying|ied)|leverag(?:e|ing|ed)|utiliz(?:e|ing|ed)|"
    r"demonstrat(?:e|ing|ed)|us(?:e|ing|ed)|show(?:ing|ed)?)\s+(?:strong\s+|excellent\s+)?"
    r"(problem[- ]solving|communication|critical thinking|analytical thinking|"
    r"attention to detail|leadership|teamwork|collaboration|business influence)\b",
    re.IGNORECASE,
)
# "big data (large, complex datasets)": a keyword followed by a parenthetical
# that glosses it. Acronym expansions like "Model Context Protocol (MCP)" and
# figures like "(expected December 2027)" are left alone.
GLOSS = re.compile(r"\(([a-z][^()\d]*?(?:,|\s\w+\s)[^()\d]*?)\)")
SUMMARY_MAX_WORDS = 55
SUMMARY_WARN_WORDS = 45
_STOP = {"and", "the", "with", "for", "from", "that", "into", "data", "their",
         "through", "across", "before", "after", "while"}
DOUBLE_SPACE = re.compile(r"[^\S\n]{2,}")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
REPEATED_WORD = re.compile(r"(?<![a-z])(\w+)\s+\1(?![a-z])", re.IGNORECASE)
STRAIGHT_QUOTE_OK = re.compile(r"[‘’“”]")
NUMERIC = re.compile(r"\d")


def _iter_bullets(resume: TailoredResume):
    """Yield (location, text, is_current_role) for every bullet in the doc."""
    for si, section in enumerate(resume.sections):
        for ei, entry in enumerate(section.entries):
            current = str(entry.get("end", "")).strip().lower() in {"present", "current"}
            for bi, bullet in enumerate(entry.get("bullets", [])):
                text = bullet["text"] if isinstance(bullet, dict) else str(bullet)
                yield f"{section.kind}[{ei}].bullets[{bi}]", text, current, (si, ei, bi)


def lint(resume: TailoredResume) -> list[GrammarIssue]:
    issues: list[GrammarIssue] = []

    def add(loc, rule, msg, excerpt="", severity="warning", suggestion=""):
        issues.append(GrammarIssue(location=loc, rule=rule, severity=severity,
                                   message=msg, excerpt=excerpt[:160],
                                   suggestion=suggestion))

    # --- whole-document checks -------------------------------------------
    # Prose only. Section headings and company names are not sentences, and
    # checking them produces noise ("EXPERIENCE" is not a first-person "I").
    whole = "\n".join([resume.summary,
                       *(t for _, t, _, _ in _iter_bullets(resume))])
    if m := FIRST_PERSON.search(whole):
        add("document", "first-person", "Resume prose should not use first-person "
            "pronouns; drop the subject and lead with the verb.",
            _context(whole, m.start()), "error")
    for m in REPEATED_WORD.finditer(whole):
        if m.group(1).lower() not in {"had", "that"}:
            add("document", "repeated-word", f"Repeated word: '{m.group(1)}'.",
                _context(whole, m.start()), "error")
    if STRAIGHT_QUOTE_OK.search(whole):
        add("document", "smart-punctuation",
            "Curly quotes and dashes can garble in older ATS parsers; prefer "
            "straight ASCII punctuation.", severity="style")

    # --- keyword-stuffing tells ------------------------------------------
    for loc, text in _prose_units(resume):
        if m := SOFT_NAME_DROP.search(text):
            add(loc, "soft-skill-name-drop",
                f"Names a soft skill ('{m.group(1)}') instead of showing it. Cut the "
                "phrase -- the rest of the sentence is the evidence.",
                _context(text, m.start()), "warning")
        if m := GLOSS.search(text):
            add(loc, "keyword-gloss",
                "Parenthetical restates the phrase before it -- this reads as a "
                "keyword inserted for a screen. Keep one wording.",
                _context(text, m.start()), "warning")
        for word in _repeated_in_sentence(text):
            add(loc, "repetition",
                f"'{word}' is used twice in one sentence; vary or cut one.",
                text, "style")

    # --- summary shape ----------------------------------------------------
    if resume.summary:
        words = len(resume.summary.split())
        sentences = len([x for x in re.split(r"(?<=[.!?])\s+", resume.summary.strip()) if x])
        if words > SUMMARY_MAX_WORDS:
            add("summary", "summary-length",
                f"Summary is {words} words; recruiters skim it in seconds. Two "
                f"sentences, under {SUMMARY_WARN_WORDS} words.", resume.summary, "error")
        elif words > SUMMARY_WARN_WORDS or sentences > 2:
            add("summary", "summary-length",
                f"Summary is {words} words in {sentences} sentences; aim for two "
                f"sentences under {SUMMARY_WARN_WORDS} words.", resume.summary, "warning")

    # --- per-bullet checks -------------------------------------------------
    tense_by_entry: dict[tuple, list[str]] = {}
    for loc, text, is_current, key in _iter_bullets(resume):
        stripped = text.strip()
        if not stripped:
            add(loc, "empty-bullet", "Bullet is empty.", severity="error")
            continue

        if not stripped[0].isupper() and not stripped[0].isdigit():
            add(loc, "capitalization", "Bullet should start with a capital letter.",
                stripped, "error")
        if not stripped.endswith((".", ")")):
            add(loc, "terminal-punctuation",
                "Bullet does not end with a period; keep terminal punctuation "
                "consistent across the document.", stripped, "style")
        if DOUBLE_SPACE.search(stripped):
            add(loc, "double-space", "Multiple consecutive spaces.", stripped, "error")
        if SPACE_BEFORE_PUNCT.search(stripped):
            add(loc, "space-before-punctuation", "Space before punctuation.",
                stripped, "error")

        low = stripped.lower()
        for opener in WEAK_OPENERS:
            if low.startswith(opener):
                add(loc, "weak-opener",
                    f"Opens with '{opener}' -- lead with a concrete action verb instead.",
                    stripped, "warning")
                break
        for word in FILLER:
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", low):
                add(loc, "filler",
                    f"'{word}' adds no information; cut it or replace with a specific.",
                    stripped, "style")
                break

        if len(stripped) > 300:
            add(loc, "bullet-length",
                f"Bullet is {len(stripped)} characters; over ~300 it stops being "
                "scannable. Split it or cut the weaker clause.", stripped, "warning")

        first_word = stripped.split()[0] if stripped.split() else ""
        if PAST_VERB.match(first_word):
            tense_by_entry.setdefault(key[:2], []).append("past")
            if is_current:
                add(loc, "tense-consistency",
                    "Current role should be written in present tense.", stripped,
                    "warning", "Rewrite the leading verb in present tense.")
        elif PRESENT_VERB.match(first_word):
            tense_by_entry.setdefault(key[:2], []).append("present")
            if not is_current:
                add(loc, "tense-consistency",
                    "Past role should be written in past tense.", stripped,
                    "warning", "Rewrite the leading verb in past tense.")

    for entry_key, tenses in tense_by_entry.items():
        if len(set(tenses)) > 1:
            add(f"section[{entry_key[0]}].entry[{entry_key[1]}]", "tense-mixed",
                "Bullets within one role mix past and present tense.",
                severity="warning")

    # --- quantification ---------------------------------------------------
    bullets = [t for _, t, _, _ in _iter_bullets(resume)]
    if bullets:
        quantified = sum(1 for b in bullets if NUMERIC.search(b))
        if quantified / len(bullets) < 0.3:
            add("document", "quantification",
                f"Only {quantified} of {len(bullets)} bullets contain a number. "
                "Numbers are the cheapest credibility signal on the page.",
                severity="style")

    return issues


def _context(text: str, pos: int, width: int = 60) -> str:
    start = max(0, pos - width // 2)
    return text[start:start + width].replace("\n", " ")


def _prose_units(resume: TailoredResume):
    if resume.summary:
        yield "summary", resume.summary
    for loc, text, _, _ in _iter_bullets(resume):
        yield loc, text


def _repeated_in_sentence(text: str) -> list[str]:
    """Content words of 5+ letters used twice within one sentence."""
    found: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        seen: set[str] = set()
        for raw in re.findall(r"[A-Za-z][A-Za-z-]+", sentence):
            w = raw.lower()
            if len(w) < 5 or w in _STOP:
                continue
            if w in seen and w not in found:
                found.append(w)
            seen.add(w)
    return found


def summarize(issues: list[GrammarIssue]) -> str:
    if not issues:
        return "No grammar or style issues found."
    by_sev: dict[str, int] = {}
    for i in issues:
        by_sev[i.severity] = by_sev.get(i.severity, 0) + 1
    head = ", ".join(f"{v} {k}" for k, v in sorted(by_sev.items()))
    lines = [f"{len(issues)} issue(s): {head}"]
    for i in issues[:20]:
        lines.append(f"  [{i.severity}] {i.location} ({i.rule}): {i.message}")
        if i.excerpt:
            lines.append(f"      > {i.excerpt}")
    return "\n".join(lines)

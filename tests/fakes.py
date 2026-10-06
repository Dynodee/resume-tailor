"""A scripted stand-in for the model, so the online code path is testable."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptedLLM:
    """Returns canned JSON per pipeline stage and records every prompt.

    Each stage is recognised by a phrase in its system prompt. Stages a test
    does not script get an empty-but-valid reply, so a test about tailoring is
    not forced to script the gap analysis too.
    """

    def __init__(self, tailor_payloads: list[dict] | None = None, edits: list | None = None,
                 *, gaps: dict | None = None, verdicts: dict | None = None,
                 letters: list[dict] | None = None, voice: dict | None = None,
                 facts: list[dict] | None = None, imported: dict | None = None):
        self.tailor_payloads = list(tailor_payloads or [{}])
        self.edits = edits or []
        self.gaps = gaps or {"items": []}
        self.verdicts = verdicts or {"verdicts": []}
        self.letters = list(letters or [])
        self.voice = voice
        self.facts = list(facts or [])
        self.imported = imported
        self.prompts: list[tuple[str, str]] = []
        self.calls: list[dict] = []

    @property
    def offline(self) -> bool:
        return False

    def complete_json(self, system: str, user, max_tokens: int = 4000, retries: int = 1,
                      schema=None, effort=None, cached: str = ""):
        text = user if isinstance(user, str) else json.dumps(user)
        if cached:
            text = cached + "\n" + text
        self.prompts.append((system, text))
        self.calls.append({"system": system, "user": user, "schema": schema,
                           "effort": effort, "cached": cached})
        if "applicant-tracking-system keywords" in system:
            return []
        if "tailor an existing resume" in system:
            if len(self.tailor_payloads) > 1:
                return self.tailor_payloads.pop(0)
            return self.tailor_payloads[0]
        if "copy editor" in system:
            return self.edits
        if "sort job requirements" in system:
            return self.gaps
        if "skeptical reviewer" in system:
            return self.verdicts
        if "turn an interview answer" in system:
            return self.facts.pop(0) if self.facts else {"text": "", "entry_id": ""}
        if "describe how a person writes" in system:
            return self.voice or {}
        if "write cover letters" in system:
            if not self.letters:
                raise AssertionError("no scripted cover letter")
            return self.letters.pop(0) if len(self.letters) > 1 else self.letters[0]
        if "convert a resume" in system:
            return self.imported
        raise AssertionError("unexpected prompt")


def allstate_payload() -> dict:
    return json.loads((FIXTURES / "allstate_payload.json").read_text())


def allstate_jd() -> str:
    return (FIXTURES / "allstate_jd.txt").read_text()

"""A scripted stand-in for the model, so the online code path is testable."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptedLLM:
    """Returns canned JSON per pipeline stage and records every prompt."""

    def __init__(self, tailor_payloads: list[dict], edits: list | None = None):
        self.tailor_payloads = list(tailor_payloads)
        self.edits = edits or []
        self.prompts: list[tuple[str, str]] = []

    @property
    def offline(self) -> bool:
        return False

    def complete_json(self, system: str, user: str, max_tokens: int = 4000, retries: int = 1):
        self.prompts.append((system, user))
        if "applicant-tracking-system keywords" in system:
            return []
        if "tailor an existing resume" in system:
            if len(self.tailor_payloads) > 1:
                return self.tailor_payloads.pop(0)
            return self.tailor_payloads[0]
        if "copy editor" in system:
            return self.edits
        raise AssertionError("unexpected prompt")


def allstate_payload() -> dict:
    return json.loads((FIXTURES / "allstate_payload.json").read_text())


def allstate_jd() -> str:
    return (FIXTURES / "allstate_jd.txt").read_text()

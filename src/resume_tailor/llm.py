"""LLM access, with a deterministic offline mode.

Two reasons the offline path exists. First, the whole graph -- ingestion,
extraction, coverage scoring, linting, PDF rendering -- is testable in CI
without a key or a network call. Second, when a run fails you can re-run it
offline and see immediately whether the bug was in the orchestration or in the
model output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Config


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = None
        if not cfg.offline:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=cfg.api_key, timeout=cfg.timeout_s
                )
            except Exception as exc:  # pragma: no cover - import/credential path
                raise LLMError(f"could not initialise Anthropic client: {exc}") from exc

    @property
    def offline(self) -> bool:
        return self._client is None

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        if self._client is None:
            return ""
        msg = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def complete_json(
        self, system: str, user: str, max_tokens: int = 4000, retries: int = 1
    ) -> Any:
        """Ask for JSON and insist on getting it.

        Models wrap JSON in prose or fences often enough that parsing has to be
        defensive; one bounded retry that shows the model its own broken output
        recovers nearly all of the rest.
        """
        if self._client is None:
            return None
        system = system + (
            "\n\nRespond with a single valid JSON value and nothing else. "
            "No prose, no markdown fences."
        )
        last = ""
        for attempt in range(retries + 1):
            raw = self.complete(system, user if attempt == 0 else
                                f"{user}\n\nYour previous reply was not valid JSON:\n"
                                f"{last[:1500]}\n\nReturn valid JSON only.",
                                max_tokens=max_tokens)
            last = raw
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
        raise LLMError(f"model did not return parseable JSON after {retries + 1} attempts")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> Any | None:
    if not raw:
        return None
    candidates = [raw.strip()]
    if m := _FENCE.search(raw):
        candidates.insert(0, m.group(1).strip())
    # Last resort: the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if 0 <= start < end:
            candidates.append(raw[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None

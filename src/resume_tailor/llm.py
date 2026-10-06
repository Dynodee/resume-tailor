"""LLM access, with a deterministic offline mode.

Two reasons the offline path exists. First, the whole graph -- ingestion,
extraction, coverage scoring, linting, PDF rendering -- is testable in CI
without a key or a network call. Second, when a run fails you can re-run it
offline and see immediately whether the bug was in the orchestration or in the
model output.

Three things the wrapper does for every call:

- **Schema-checked output.** Given a Pydantic model, the call uses structured
  outputs, so the reply is guaranteed to match it. Models that do not support
  that fall back to asking for JSON in the prompt and validating afterwards.
- **Effort per call.** Keyword extraction does not need the depth that
  tailoring does. Each node says how hard to think; ``RT_EFFORT`` overrides all
  of them at once.
- **Refusal fallback.** On the models that support it, a safety decline is
  retried server-side on another model instead of failing the run.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Config


class LLMError(RuntimeError):
    pass


# Models that accept output_config.effort with the full low..max range.
_EFFORT_MODELS = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5",
                  "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
                  "claude-sonnet-4-6")
# Models that take the server-side refusal fallback in its "default" form.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
                    "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

Content = str | list[dict[str, Any]]


class LLM:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = None
        self._structured_ok = True
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

    # --- request plumbing ------------------------------------------------------

    def _kwargs(self, system: str, user: Content, max_tokens: int,
                effort: str | None, cached: str) -> dict[str, Any]:
        content: Content = user
        if cached:
            # The stable part goes first, marked for caching, so a revision pass
            # that resends it pays a fraction of the input price.
            blocks = user if isinstance(user, list) else [{"type": "text", "text": user}]
            content = [{"type": "text", "text": cached,
                        "cache_control": {"type": "ephemeral"}}, *blocks]
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        effort = self.cfg.effort or effort
        if effort and self.cfg.model.startswith(_EFFORT_MODELS):
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    def _messages(self, kwargs: dict[str, Any]):
        """The messages resource to call, with the refusal fallback when available."""
        if self.cfg.fallbacks and self.cfg.model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
            return self._client.beta.messages
        return self._client.messages

    @staticmethod
    def _check_stop(msg) -> None:
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMError(f"the model declined this request (category: {category})")

    # --- public API --------------------------------------------------------------

    def complete(self, system: str, user: Content, max_tokens: int = 4000,
                 effort: str | None = None, cached: str = "") -> str:
        if self._client is None:
            return ""
        kwargs = self._kwargs(system, user, max_tokens, effort, cached)
        msg = self._messages(kwargs).create(**kwargs)
        self._check_stop(msg)
        return "".join(block.text for block in msg.content if block.type == "text")

    def complete_json(
        self, system: str, user: Content, max_tokens: int = 4000, retries: int = 1,
        schema: type | None = None, effort: str | None = None, cached: str = "",
    ) -> Any:
        """Ask for JSON and insist on getting it.

        With a Pydantic ``schema`` the reply is schema-checked by the API and
        comes back as plain dicts and lists, so callers handle both paths the
        same way. Without one -- or on a model that cannot do structured output
        -- the reply is parsed defensively, with one bounded retry that shows
        the model its own broken output.
        """
        if self._client is None:
            return None
        if schema is not None and self._structured_ok:
            parsed = self._parse(system, user, max_tokens, schema, effort, cached)
            if parsed is not None:
                return parsed

        system = system + (
            "\n\nRespond with a single valid JSON value and nothing else. "
            "No prose, no markdown fences."
        )
        last = ""
        for attempt in range(retries + 1):
            prompt = user
            if attempt:
                note = (f"\n\nYour previous reply was not valid JSON:\n{last[:1500]}"
                        "\n\nReturn valid JSON only.")
                prompt = (user + note if isinstance(user, str)
                          else [*user, {"type": "text", "text": note}])
            raw = self.complete(system, prompt, max_tokens=max_tokens, effort=effort,
                                cached=cached)
            last = raw
            value = _extract_json(raw)
            if value is None:
                continue
            if schema is None:
                return value
            try:
                return schema.model_validate(value).model_dump()
            except Exception:  # noqa: BLE001 - any validation failure means retry
                continue
        raise LLMError(f"model did not return parseable JSON after {retries + 1} attempts")

    def _parse(self, system: str, user: Content, max_tokens: int, schema: type,
               effort: str | None, cached: str) -> Any | None:
        import anthropic

        kwargs = self._kwargs(system, user, max_tokens, effort, cached)
        try:
            msg = self._messages(kwargs).parse(output_format=schema, **kwargs)
        except anthropic.BadRequestError as exc:
            # Model or schema not supported by structured outputs: remember that
            # and use the prompt-and-validate path for the rest of the run.
            if "output" in str(exc).lower() or "schema" in str(exc).lower():
                self._structured_ok = False
                return None
            raise
        self._check_stop(msg)
        if msg.parsed_output is None:
            if msg.stop_reason == "max_tokens":
                raise LLMError(f"reply was cut off at max_tokens={max_tokens}")
            return None
        return msg.parsed_output.model_dump()


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

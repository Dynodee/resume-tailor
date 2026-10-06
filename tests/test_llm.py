"""Request shapes the LLM wrapper sends, checked against a fake client."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import pytest
from pydantic import BaseModel

from resume_tailor.config import Config
from resume_tailor.llm import LLM, LLMError


class Answer(BaseModel):
    value: int


class _BadRequest(anthropic.BadRequestError):
    def __init__(self, message: str) -> None:  # skip the HTTP plumbing
        Exception.__init__(self, message)


def _msg(text="", parsed=None, stop="end_turn", category=None):
    return SimpleNamespace(
        stop_reason=stop,
        stop_details=SimpleNamespace(category=category) if category else None,
        content=[SimpleNamespace(type="text", text=text)],
        parsed_output=parsed,
    )


class _Resource:
    def __init__(self, replies, log, name):
        self.replies, self.log, self.name = replies, log, name

    def _next(self, kind, kwargs):
        self.log.append((self.name, kind, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def create(self, **kwargs):
        return self._next("create", kwargs)

    def parse(self, **kwargs):
        return self._next("parse", kwargs)


def _llm(tmp_path, model="claude-opus-5-5", replies=(), **cfg):
    llm = LLM(Config(out_dir=tmp_path, model=model, offline=True, **cfg))
    log: list = []
    replies = list(replies)
    llm._client = SimpleNamespace(
        messages=_Resource(replies, log, "messages"),
        beta=SimpleNamespace(messages=_Resource(replies, log, "beta")),
    )
    return llm, log


def test_effort_and_fallback_on_a_current_model(tmp_path):
    llm, log = _llm(tmp_path, replies=[_msg("hi")])
    assert llm.complete("sys", "user", effort="high") == "hi"
    resource, kind, kw = log[0]
    assert resource == "beta" and kind == "create"
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"] == {"effort": "high"}


def test_no_effort_or_fallback_on_an_older_model(tmp_path):
    llm, log = _llm(tmp_path, model="claude-haiku-4-5", replies=[_msg("hi")])
    llm.complete("sys", "user", effort="high")
    resource, _, kw = log[0]
    assert resource == "messages"
    assert "output_config" not in kw and "fallbacks" not in kw


def test_rt_effort_overrides_every_node(tmp_path):
    llm, log = _llm(tmp_path, replies=[_msg("hi")], effort="low")
    llm.complete("sys", "user", effort="high")
    assert log[0][2]["output_config"] == {"effort": "low"}


def test_fallbacks_can_be_switched_off(tmp_path):
    llm, log = _llm(tmp_path, replies=[_msg("hi")], fallbacks=False)
    llm.complete("sys", "user")
    assert log[0][0] == "messages"


def test_schema_uses_structured_output(tmp_path):
    llm, log = _llm(tmp_path, replies=[_msg(parsed=Answer(value=3))])
    assert llm.complete_json("sys", "user", schema=Answer) == {"value": 3}
    _, kind, kw = log[0]
    assert kind == "parse" and kw["output_format"] is Answer


def test_unsupported_structured_output_falls_back_to_prompted_json(tmp_path):
    llm, log = _llm(tmp_path, replies=[_BadRequest("output_format is not supported"),
                                       _msg('{"value": 4}')])
    assert llm.complete_json("sys", "user", schema=Answer) == {"value": 4}
    assert [k for _, k, _ in log] == ["parse", "create"]
    assert "single valid JSON value" in log[1][2]["system"]
    assert llm._structured_ok is False


def test_prompted_json_is_validated_against_the_schema(tmp_path):
    llm, _ = _llm(tmp_path, replies=[_BadRequest("schema not supported"),
                                     _msg('{"wrong": 1}'), _msg('{"value": 5}')])
    assert llm.complete_json("sys", "user", schema=Answer) == {"value": 5}


def test_refusal_raises_a_clear_error(tmp_path):
    llm, _ = _llm(tmp_path, replies=[_msg(stop="refusal", category="cyber")])
    with pytest.raises(LLMError, match="declined"):
        llm.complete("sys", "user")


def test_cached_prefix_goes_first_with_cache_control(tmp_path):
    llm, log = _llm(tmp_path, replies=[_msg("ok")])
    llm.complete("sys", "the variable part", cached="the stable part")
    content = log[0][2]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "the stable part",
                          "cache_control": {"type": "ephemeral"}}
    assert content[1] == {"type": "text", "text": "the variable part"}


def test_document_blocks_pass_through(tmp_path):
    llm, log = _llm(tmp_path, replies=[_msg("ok")])
    doc = {"type": "document", "source": {"type": "base64",
                                          "media_type": "application/pdf", "data": "AA=="}}
    llm.complete("sys", [doc, {"type": "text", "text": "read this"}])
    assert log[0][2]["messages"][0]["content"][0] is doc


def test_offline_returns_nothing(tmp_path):
    llm = LLM(Config(out_dir=tmp_path, offline=True))
    assert llm.offline
    assert llm.complete_json("sys", "user", schema=Answer) is None

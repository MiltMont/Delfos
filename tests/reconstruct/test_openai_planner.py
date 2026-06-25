"""Tests for the OpenAI-backed HopPlanner.

The planner is the LLM in the reconstruct loop. These tests drive it against a
fake OpenAI client so no network is touched: they pin the request the planner
sends (model, structured `response_format`) and how it maps the parsed response
back onto :class:`HopDecision`.
"""

from __future__ import annotations

from typing import Any, cast

from openai import OpenAI

from delfos.reconstruct.planner import (
    CandidateSummary,
    Collected,
    HopDecision,
    HopPlanner,
    HopRequest,
)
from delfos.reconstruct.planners.openai import OpenAIHopPlanner


class _FakeMessage:
    def __init__(self, parsed: HopDecision | None) -> None:
        self.parsed = parsed
        self.refusal: str | None = None


class _FakeChoice:
    def __init__(self, parsed: HopDecision | None) -> None:
        self.message = _FakeMessage(parsed)


class _FakeParseResult:
    def __init__(self, parsed: HopDecision | None) -> None:
        self.choices = [_FakeChoice(parsed)]


class _FakeCompletions:
    def __init__(self, parsed: HopDecision | None) -> None:
        self._parsed = parsed
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> _FakeParseResult:
        self.calls.append(kwargs)
        return _FakeParseResult(self._parsed)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, parsed: HopDecision | None) -> None:
        self.completions = _FakeCompletions(parsed)
        self.chat = _FakeChat(self.completions)


def _request() -> HopRequest:
    cur = CandidateSummary(id="cue-1", node_kind="cue", label="auth", snippet=None, tags=[])
    cand = CandidateSummary(
        id="content-1",
        node_kind="content",
        label="def login()",
        snippet="Authenticate a user.",
        tags=["arch_layer=service"],
    )
    return HopRequest(query="how does auth work", current=cur, candidates=[cand], hops_remaining=3)


def test_decide_returns_parsed_decision() -> None:
    decision = HopDecision(
        collect=[Collected(id="content-1", relevance=0.8)],
        descend_into=None,
        stop=True,
    )
    fake = _FakeClient(decision)
    planner = OpenAIHopPlanner(model="gpt-4o-mini", client=cast(OpenAI, fake))

    result = planner.decide(_request())

    assert result is decision


def test_decide_sends_model_and_structured_format() -> None:
    decision = HopDecision(collect=[], descend_into=None, stop=True)
    fake = _FakeClient(decision)
    planner = OpenAIHopPlanner(model="gpt-4o-mini", client=cast(OpenAI, fake))

    planner.decide(_request())

    assert len(fake.completions.calls) == 1
    call = fake.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["response_format"] is HopDecision


def test_decide_includes_query_and_candidate_ids_in_prompt() -> None:
    decision = HopDecision(collect=[], descend_into=None, stop=True)
    fake = _FakeClient(decision)
    planner = OpenAIHopPlanner(model="gpt-4o-mini", client=cast(OpenAI, fake))

    planner.decide(_request())

    messages = fake.completions.calls[0]["messages"]
    blob = " ".join(str(m.get("content", "")) for m in messages)
    assert "how does auth work" in blob
    assert "content-1" in blob


def test_decide_stops_when_model_returns_no_parsed_object() -> None:
    fake = _FakeClient(None)
    planner = OpenAIHopPlanner(model="gpt-4o-mini", client=cast(OpenAI, fake))

    result = planner.decide(_request())

    assert result.stop is True
    assert result.collect == []
    assert result.descend_into is None


def test_satisfies_hop_planner_protocol() -> None:
    fake = _FakeClient(HopDecision())
    planner = OpenAIHopPlanner(model="gpt-4o-mini", client=cast(OpenAI, fake))
    assert isinstance(planner, HopPlanner)

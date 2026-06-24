from __future__ import annotations

import pytest
from pydantic import ValidationError

from delfos.reconstruct.planner import (
    CandidateSummary,
    Collected,
    HopDecision,
    HopPlanner,
    HopRequest,
)
from delfos.reconstruct.planners.fake import FakeHopPlanner


def test_candidate_summary_roundtrips() -> None:
    c = CandidateSummary(
        id="content-1",
        node_kind="content",
        label="def load_config()",
        snippet="Load the config.",
        tags=["language=python"],
    )
    assert c.id == "content-1"
    assert c.node_kind == "content"


def test_collected_relevance_must_be_in_unit_range() -> None:
    with pytest.raises(ValidationError):
        Collected(id="x", relevance=1.5)
    with pytest.raises(ValidationError):
        Collected(id="x", relevance=-0.1)


def test_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HopDecision(collect=[], descend_into=None, stop=False, bogus=1)  # type: ignore[call-arg]


def test_hop_request_holds_candidates() -> None:
    cur = CandidateSummary(id="cue-1", node_kind="cue", label="auth", snippet=None, tags=[])
    req = HopRequest(query="how does auth work", current=cur, candidates=[cur], hops_remaining=3)
    assert req.hops_remaining == 3
    assert req.candidates[0].id == "cue-1"


def test_fake_planner_returns_scripted_decisions_then_stops() -> None:
    d1 = HopDecision(collect=[Collected(id="a", relevance=0.9)], descend_into="a", stop=False)
    planner = FakeHopPlanner([d1])
    cur = CandidateSummary(id="cue-1", node_kind="cue", label="x", snippet=None, tags=[])
    req = HopRequest(query="q", current=cur, candidates=[cur], hops_remaining=3)

    assert planner.decide(req) is d1
    # Once the script is exhausted, it returns a terminal stop decision.
    after = planner.decide(req)
    assert after.stop is True
    assert planner.call_count == 2
    assert len(planner.requests) == 2


def test_fake_planner_raises_after_error_after() -> None:
    planner = FakeHopPlanner([], error_after=0)
    cur = CandidateSummary(id="cue-1", node_kind="cue", label="x", snippet=None, tags=[])
    req = HopRequest(query="q", current=cur, candidates=[cur], hops_remaining=1)

    with pytest.raises(RuntimeError):
        planner.decide(req)


def test_fake_planner_satisfies_protocol() -> None:
    assert isinstance(FakeHopPlanner([]), HopPlanner)

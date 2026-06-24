"""Provider-agnostic interface for the per-hop reconstruction planner.

The planner is the LLM in the `reconstruct` loop. It sees the current node and
its candidate neighbors and returns which to collect plus which single one to
descend into. Concrete backends (OpenAI/Anthropic) are added at implementation
time; this module fixes only the data contract.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class CandidateSummary(BaseModel):
    """The compact view of a node the planner reasons over."""

    model_config = ConfigDict(extra="forbid")

    id: str
    node_kind: Literal["cue", "content"]
    label: str
    snippet: str | None = None
    tags: list[str] = Field(default_factory=list)


class Collected(BaseModel):
    """A node the planner chose to include, with its relevance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    relevance: float = Field(ge=0.0, le=1.0)


class HopRequest(BaseModel):
    """Everything the planner needs to decide a single hop."""

    model_config = ConfigDict(extra="forbid")

    query: str
    current: CandidateSummary
    candidates: list[CandidateSummary]
    hops_remaining: int


class HopDecision(BaseModel):
    """The planner's decision for one hop."""

    model_config = ConfigDict(extra="forbid")

    collect: list[Collected] = Field(default_factory=lambda: [])
    descend_into: str | None = None
    stop: bool = False


@runtime_checkable
class HopPlanner(Protocol):
    """Decides one hop of the reconstruction walk."""

    def decide(self, request: HopRequest) -> HopDecision: ...

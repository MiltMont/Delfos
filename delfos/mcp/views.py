"""MCP-facing serialization models for graph nodes.

Dedicated to the MCP layer (not reused from ``reconstruct.planner``) so the tool
surface stays decoupled from the planner's ``CandidateSummary`` contract. Walk
tools return :class:`NodeSummary` (cheap); ``fetch`` returns :class:`ContentDetail`
(full body). Embeddings are never serialized back to the agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from delfos.schema import ContentNode, CueNode

SNIPPET_LIMIT = 500  # mirrors delfos.reconstruct.summaries._SNIPPET_LIMIT


def _truncate(text: str, limit: int = SNIPPET_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class NodeSummary(BaseModel):
    """Compact, walk-time view of a node. Cheap enough to fan out over."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["cue", "content"]
    label: str
    snippet: str | None = None
    tags: list[str] = Field(default_factory=list)


class ContentDetail(BaseModel):
    """Full content payload returned by ``fetch``. No embedding."""

    model_config = ConfigDict(extra="forbid")

    id: str
    symbol_name: str | None
    signature: str | None
    docstring: str | None
    body: str
    memory_layer: str
    source_file: str
    git_sha: str


def cue_to_summary(node: CueNode) -> NodeSummary:
    """Summarize a cue: its text is the label; cues carry no content tags."""
    return NodeSummary(id=node.id, kind="cue", label=node.text, snippet=None, tags=[])


def content_to_summary(node: ContentNode, tags: list[str]) -> NodeSummary:
    """Summarize content: signature/symbol/kind as label, docstring-or-body snippet."""
    label = node.signature or node.symbol_name or node.kind.value
    snippet = node.docstring if node.docstring is not None else _truncate(node.body)
    return NodeSummary(id=node.id, kind="content", label=label, snippet=snippet, tags=list(tags))


def content_to_detail(node: ContentNode) -> ContentDetail:
    """Full content view, embedding intentionally dropped."""
    return ContentDetail(
        id=node.id,
        symbol_name=node.symbol_name,
        signature=node.signature,
        docstring=node.docstring,
        body=node.body,
        memory_layer=node.memory_layer.value,
        source_file=node.source_file,
        git_sha=node.git_sha,
    )

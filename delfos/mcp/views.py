"""MCP-facing serialization models for graph nodes.

Dedicated to the MCP layer (not reused from ``reconstruct.planner``) so the tool
surface stays decoupled from the planner's ``CandidateSummary`` contract. Walk
tools return :class:`NodeSummary` (cheap); ``fetch`` returns :class:`ContentDetail`
(full body). Embeddings are never serialized back to the agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from delfos.enrich import AnnotationOutcome
from delfos.schema import ContentNode, CueNode
from delfos.scip.reader import Occurrence, Relationship

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


class ScipReference(BaseModel):
    """A single SCIP occurrence of a symbol: where it is used (SCIP 0-based lines)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    relative_path: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


class ScipRelation(BaseModel):
    """A SCIP relationship target symbol (e.g. an implemented or type symbol)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str


def occurrence_to_reference(relative_path: str, occ: Occurrence) -> ScipReference:
    """Serialize a ``(relative_path, Occurrence)`` pair for the MCP surface."""
    return ScipReference(
        symbol=occ.symbol,
        relative_path=relative_path,
        start_line=occ.start_line,
        start_col=occ.start_col,
        end_line=occ.end_line,
        end_col=occ.end_col,
    )


def relationship_to_relation(rel: Relationship) -> ScipRelation:
    """Serialize a SCIP relationship target symbol for the MCP surface."""
    return ScipRelation(symbol=rel.symbol)


class AnnotateResult(BaseModel):
    """What ``annotate`` wrote, dropped, and the existing tag vocabulary to reuse."""

    model_config = ConfigDict(extra="forbid")

    content_id: str
    written_cue_ids: list[str]
    written_tag_ids: list[str]
    dropped_phrases: list[str]
    existing_values: dict[str, list[str]]


def outcome_to_result(outcome: AnnotationOutcome) -> AnnotateResult:
    """Serialize an annotation outcome for the MCP surface."""
    return AnnotateResult(
        content_id=outcome.content_id,
        written_cue_ids=outcome.written_cue_ids,
        written_tag_ids=outcome.written_tag_ids,
        dropped_phrases=outcome.dropped_phrases,
        existing_values=outcome.existing_values,
    )

"""FastMCP read server: four graph tools + a reconstruct prompt.

Tool logic lives in plain ``_``-prefixed functions so it is unit-testable
without an MCP transport; :func:`build_server` registers thin wrappers. The
calling agent is the planner — the server runs no planner LLM. The ``reconstruct``
prompt teaches the depth-first walk the agent drives.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from delfos.reconstruct import ReconstructionService, TagFilter
from delfos.schema import TagCategory
from delfos.scip.service import ScipService

from .views import (
    ContentDetail,
    NodeSummary,
    ScipReference,
    ScipRelation,
    content_to_detail,
    content_to_summary,
    cue_to_summary,
    occurrence_to_reference,
    relationship_to_relation,
)

_SCIP_UNAVAILABLE = (
    "SCIP index not available: the server was started without an index.scip. "
    "Run `delfos index <repo>` with scip-python installed to generate one."
)


def _to_tag_filters(pairs: list[tuple[str, str]]) -> list[TagFilter]:
    out: list[TagFilter] = []
    for category, value in pairs:
        try:
            cat = TagCategory(category)
        except ValueError as exc:
            valid = ", ".join(c.value for c in TagCategory)
            raise ValueError(
                f"unknown tag category {category!r}; valid categories: {valid}"
            ) from exc
        out.append((cat, value))
    return out


def _search(service: ReconstructionService, query: str, k: int = 5) -> list[NodeSummary]:
    try:
        cues = service.search(query, k)
    except Exception as exc:  # noqa: BLE001 - tool boundary: translate endpoint failures into actionable errors
        raise RuntimeError(
            "search failed: could not reach the embedding endpoint. "
            "Check the embedding endpoint is up and the model is pulled."
        ) from exc
    return [cue_to_summary(c) for c in cues]


def _traverse_forward(
    service: ReconstructionService,
    cue_ids: list[str],
    tag_filters: list[tuple[str, str]] | None = None,
) -> list[NodeSummary]:
    filters = _to_tag_filters(tag_filters) if tag_filters else None
    contents = service.traverse_forward(cue_ids, filters)
    return [content_to_summary(c, service.content_tags(c.id)) for c in contents]


def _traverse_reverse(service: ReconstructionService, content_ids: list[str]) -> list[NodeSummary]:
    return [cue_to_summary(c) for c in service.traverse_reverse(content_ids)]


def _fetch(service: ReconstructionService, ids: list[str]) -> list[ContentDetail]:
    return [content_to_detail(c) for c in service.fetch(ids)]


def _require_scip(scip: ScipService | None) -> ScipService:
    if scip is None:
        raise RuntimeError(_SCIP_UNAVAILABLE)
    return scip


def _references(scip: ScipService | None, content_id: str) -> list[ScipReference]:
    pairs = _require_scip(scip).references(content_id)
    return [occurrence_to_reference(path, occ) for path, occ in pairs]


def _implementations(scip: ScipService | None, content_id: str) -> list[ScipRelation]:
    rels = _require_scip(scip).implementations(content_id)
    return [relationship_to_relation(r) for r in rels]


def _type_definition(scip: ScipService | None, content_id: str) -> list[ScipRelation]:
    rels = _require_scip(scip).type_definition(content_id)
    return [relationship_to_relation(r) for r in rels]


def reconstruct_prompt(query: str, budget: int = 3) -> str:
    """Protocol text teaching the agent to drive a depth-first reconstruction."""
    return (
        f"Reconstruct memory for this query by walking the graph yourself; you are "
        f"the planner.\n\n"
        f"Query: {query}\n"
        f"Budget: {budget} traversal steps.\n\n"
        f"Protocol:\n"
        f"1. Call `search` with the query to get seed cue nodes.\n"
        f"2. Call `traverse_forward` on the most promising cues to reach content; "
        f"use tag_filters to narrow when a category is obviously relevant.\n"
        f"3. Descend depth-first: expand the single most relevant candidate one hop "
        f"at a time rather than fanning out. Use `traverse_reverse` to discover "
        f"sibling cues when a content node looks central.\n"
        f"4. Spend at most {budget} traversal steps; backtrack when a branch stops "
        f"yielding relevant nodes.\n"
        f"5. Call `fetch` with the ids worth keeping to get their full bodies.\n"
        f"6. Optionally expand a content node with SCIP code intelligence: "
        f"`references` (where its symbol is used), `implementations`, or "
        f"`type_definition`. Invoke these only when cross-references help answer "
        f"the query.\n"
        f"7. Stop when relevance drops or the budget is exhausted, then answer from "
        f"the fetched content."
    )


def build_server(service: ReconstructionService, scip: ScipService | None = None) -> FastMCP:
    """Build the FastMCP app, registering the graph + SCIP tools and the prompt.

    ``scip`` is optional: when ``None`` (no ``index.scip`` was loaded) the SCIP
    tools are still registered but return an actionable error, so the rest of
    the server keeps working.
    """
    mcp = FastMCP("delfos")

    @mcp.tool()
    def search(query: str, k: int = 5) -> list[NodeSummary]:  # pyright: ignore[reportUnusedFunction]
        """Find cue entry points by semantic similarity. Start a walk here."""
        return _search(service, query, k)

    @mcp.tool()
    def traverse_forward(  # pyright: ignore[reportUnusedFunction]
        cue_ids: list[str], tag_filters: list[tuple[str, str]] | None = None
    ) -> list[NodeSummary]:
        """Expand cues to their content. tag_filters are (category, value) pairs."""
        return _traverse_forward(service, cue_ids, tag_filters)

    @mcp.tool()
    def traverse_reverse(content_ids: list[str]) -> list[NodeSummary]:  # pyright: ignore[reportUnusedFunction]
        """Discover sibling cues that point at the given content nodes."""
        return _traverse_reverse(service, content_ids)

    @mcp.tool()
    def fetch(ids: list[str]) -> list[ContentDetail]:  # pyright: ignore[reportUnusedFunction]
        """Fetch full content bodies for the given node ids."""
        return _fetch(service, ids)

    @mcp.tool()
    def references(content_id: str) -> list[ScipReference]:  # pyright: ignore[reportUnusedFunction]
        """SCIP cross-references: where the content node's symbol is used."""
        return _references(scip, content_id)

    @mcp.tool()
    def implementations(content_id: str) -> list[ScipRelation]:  # pyright: ignore[reportUnusedFunction]
        """SCIP implementations: symbols the content node's symbol implements."""
        return _implementations(scip, content_id)

    @mcp.tool()
    def type_definition(content_id: str) -> list[ScipRelation]:  # pyright: ignore[reportUnusedFunction]
        """SCIP type definitions for the content node's symbol."""
        return _type_definition(scip, content_id)

    @mcp.prompt()
    def reconstruct(query: str, budget: int = 3) -> str:  # pyright: ignore[reportUnusedFunction]
        """Drive a depth-first memory reconstruction over the graph."""
        return reconstruct_prompt(query, budget)

    return mcp

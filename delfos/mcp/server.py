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

from .views import (
    ContentDetail,
    NodeSummary,
    content_to_detail,
    content_to_summary,
    cue_to_summary,
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
    return [cue_to_summary(c) for c in service.search(query, k)]


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
        f"6. Stop when relevance drops or the budget is exhausted, then answer from "
        f"the fetched content."
    )


def build_server(service: ReconstructionService) -> FastMCP:
    """Build the FastMCP app, registering the four tools and the prompt."""
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

    @mcp.prompt()
    def reconstruct(query: str, budget: int = 3) -> str:  # pyright: ignore[reportUnusedFunction]
        """Drive a depth-first memory reconstruction over the graph."""
        return reconstruct_prompt(query, budget)

    return mcp

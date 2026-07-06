"""FastMCP read server: graph + annotate tools, a reconstruct and enrich prompt.

Tool logic lives in plain ``_``-prefixed functions so it is unit-testable
without an MCP transport; :func:`build_server` registers thin wrappers. The
calling agent is the planner — the server runs no planner LLM. The ``reconstruct``
prompt teaches the depth-first walk the agent drives; the ``enrich`` prompt
teaches the agent to write back concept cues and semantic tags.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from delfos.enrich import EnrichmentService
from delfos.reconstruct import ReconstructionService, TagFilter
from delfos.schema import TagCategory
from delfos.scip.service import ScipService

from .views import (
    AnnotateResult,
    ContentDetail,
    NodeSummary,
    ScipReference,
    ScipRelation,
    content_to_detail,
    content_to_summary,
    cue_to_summary,
    occurrence_to_reference,
    outcome_to_result,
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


_ENRICH_UNAVAILABLE = "enrichment unavailable: the server was built without an EnrichmentService."


def _annotate(
    enrich: EnrichmentService | None,
    content_id: str,
    concepts: list[str] | None,
    *,
    arch_layer: str | None,
    pattern_type: str | None,
) -> AnnotateResult:
    if enrich is None:
        raise RuntimeError(_ENRICH_UNAVAILABLE)
    outcome = enrich.annotate(
        content_id, concepts or [], arch_layer=arch_layer, pattern_type=pattern_type
    )
    return outcome_to_result(outcome)


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


def enrich_prompt(focus: str = "") -> str:
    """Protocol text teaching the agent to enrich content it has actually read."""
    scope = f" Focus on: {focus}." if focus else ""
    return (
        f"Enrich the code-memory graph with what you learned; you are the "
        f"extractor.{scope}\n\n"
        f"Protocol:\n"
        f"1. Only annotate content nodes whose bodies you have read via `fetch`.\n"
        f"2. Call `annotate` with 1-5 concept phrases per node describing what the "
        f"code is about (e.g. 'rate limiting', 'crash recovery'). Never restate the "
        f"symbol name — that cue already exists.\n"
        f"3. Optionally set arch_layer (which architectural layer the code belongs "
        f"to) and pattern_type (the recurring pattern it embodies).\n"
        f"4. The result echoes existing tag values: reuse one unless none fits.\n"
        f"5. Annotations are wiped when their file is re-indexed, so do not "
        f"annotate code you are about to change."
    )


def build_server(
    service: ReconstructionService,
    scip: ScipService | None = None,
    enrich: EnrichmentService | None = None,
) -> FastMCP:
    """Build the FastMCP app, registering the graph + SCIP + annotate tools and prompts.

    ``scip`` is optional: when ``None`` (no ``index.scip`` was loaded) the SCIP
    tools are still registered but return an actionable error, so the rest of
    the server keeps working. ``enrich`` is optional the same way: when ``None``
    the ``annotate`` tool is still registered but returns an actionable error.
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

    @mcp.tool()
    def annotate(  # pyright: ignore[reportUnusedFunction]
        content_id: str,
        concepts: list[str] | None = None,
        arch_layer: str | None = None,
        pattern_type: str | None = None,
    ) -> AnnotateResult:
        """Attach concept cues and semantic tags to a content node you have read.

        Concepts are short phrases describing what the code is about. The result
        echoes existing arch_layer/pattern_type values — reuse them when they fit.
        Call with only content_id to just see the current vocabulary.
        """
        return _annotate(
            enrich, content_id, concepts, arch_layer=arch_layer, pattern_type=pattern_type
        )

    @mcp.prompt()
    def reconstruct(query: str, budget: int = 3) -> str:  # pyright: ignore[reportUnusedFunction]
        """Drive a depth-first memory reconstruction over the graph."""
        return reconstruct_prompt(query, budget)

    @mcp.prompt(name="enrich")
    def enrich_memory(focus: str = "") -> str:  # pyright: ignore[reportUnusedFunction]
        """Teach the agent to write concept cues and semantic tags for code it read."""
        return enrich_prompt(focus)

    return mcp

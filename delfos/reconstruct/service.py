"""The read-path service: search, traversal, and LLM-driven reconstruction.

Sits entirely on top of :class:`~delfos.store.base.GraphStore`; it never touches
the database directly. The three primitives are pure graph operations;
``reconstruct`` additionally drives a :class:`~delfos.reconstruct.planner.HopPlanner`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from delfos.indexer.embedder import Embedder
from delfos.schema import (
    ContentNode,
    CueNode,
    Direction,
    EdgeType,
    Node,
    NodeStatus,
    NodeType,
    TagCategory,
    TagNode,
)
from delfos.store import GraphStore

from .planner import CandidateSummary, HopDecision, HopPlanner, HopRequest
from .summaries import summarize

logger = logging.getLogger(__name__)

TagFilter = tuple[TagCategory, str]


class ReconstructionService:
    """Read-path operations over the Cue-Tag-Content graph."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        planner: HopPlanner,
        *,
        seed_k: int = 5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._planner = planner
        self._seed_k = seed_k

    def search(self, query: str, k: int = 5) -> list[CueNode]:
        """Embed ``query`` and return the ``k`` nearest cue nodes."""
        embedding = self._embedder.embed([query])[0]
        hits = self._store.vector_search(embedding, k, node_type=NodeType.CUE)
        cues: list[CueNode] = []
        for hit in hits:
            node = self._store.get_node(hit.node_id)
            if isinstance(node, CueNode):
                cues.append(node)
        return cues

    def traverse_forward(
        self,
        cue_ids: Sequence[str],
        tag_filters: Sequence[TagFilter] | None = None,
    ) -> list[ContentNode]:
        """Expand cues to their ACTIVE content, optionally tag-filtered."""
        wanted = set(tag_filters) if tag_filters else None
        out: list[ContentNode] = []
        seen: set[str] = set()
        for cue_id in cue_ids:
            for neighbor in self._store.neighbors(
                cue_id, edge_type=EdgeType.CUE_OF, direction=Direction.OUTGOING
            ):
                content = self._eligible_content(neighbor, wanted, seen)
                if content is not None:
                    seen.add(content.id)
                    out.append(content)
        return out

    def traverse_reverse(self, content_ids: Sequence[str]) -> list[CueNode]:
        """Discover sibling cues that point at the given content nodes."""
        out: list[CueNode] = []
        seen: set[str] = set()
        for content_id in content_ids:
            for neighbor in self._store.neighbors(
                content_id, edge_type=EdgeType.CUE_OF, direction=Direction.INCOMING
            ):
                if isinstance(neighbor, CueNode) and neighbor.id not in seen:
                    seen.add(neighbor.id)
                    out.append(neighbor)
        return out

    def reconstruct(
        self,
        query: str,
        budget: int = 3,
        tag_filters: Sequence[TagFilter] | None = None,
    ) -> list[ContentNode]:
        """Reconstruct a relevant content set via LLM-driven depth-first walk.

        Seeds with :meth:`search`, then at each hop asks the planner which
        neighbors to collect and which single one to descend into. ``budget``
        caps the total number of planner calls. Returns content ordered by
        planner-assigned relevance (descending; ties keep discovery order).
        """
        wanted = set(tag_filters) if tag_filters else None
        seeds = self.search(query, k=self._seed_k)
        if not seeds:
            return []

        seed_queue: list[Node] = list(seeds)
        current: Node = seed_queue.pop(0)
        stack: list[Node] = []
        visited: set[str] = {current.id}
        result: dict[str, tuple[ContentNode, float]] = {}
        budget_remaining = budget

        while budget_remaining > 0:
            candidates = self._candidates_for(current, wanted)
            request = HopRequest(
                query=query,
                current=self._to_summary(current),
                candidates=[self._to_summary(c) for c in candidates],
                hops_remaining=budget_remaining,
            )
            try:
                decision = self._planner.decide(request)
            except Exception:
                logger.warning(
                    "hop planner failed; returning partial reconstruction",
                    exc_info=True,
                )
                break
            budget_remaining -= 1

            by_id = {c.id: c for c in candidates}
            self._collect(decision, by_id, result)

            if decision.stop:
                break

            nxt = by_id.get(decision.descend_into) if decision.descend_into else None
            if nxt is not None and nxt.id not in visited:
                stack.append(current)
                current = nxt
                visited.add(nxt.id)
            elif stack:
                current = stack.pop()
            elif seed_queue:
                current = seed_queue.pop(0)
                visited.add(current.id)
            else:
                break

        ordered = sorted(result.values(), key=lambda pair: pair[1], reverse=True)
        return [content for content, _ in ordered]

    def _to_summary(self, node: Node) -> CandidateSummary:
        """Build the planner-facing summary, attaching tags for content."""
        if isinstance(node, ContentNode):
            tags = sorted(f"{cat.value}={val}" for cat, val in self._content_tags(node.id))
            return summarize(node, tags)
        if isinstance(node, CueNode):
            return summarize(node)
        raise TypeError(f"cannot summarize node type: {type(node).__name__}")

    def _collect(
        self,
        decision: HopDecision,
        by_id: dict[str, Node],
        result: dict[str, tuple[ContentNode, float]],
    ) -> None:
        for item in decision.collect:
            node = by_id.get(item.id)
            if node is None:
                continue
            content = self._as_content(node)
            if content is None:
                continue
            existing = result.get(content.id)
            if existing is None or item.relevance > existing[1]:
                result[content.id] = (content, item.relevance)

    def _candidates_for(self, node: Node, wanted: set[TagFilter] | None) -> list[Node]:
        if isinstance(node, CueNode):
            return self._content_candidates(node.id, wanted)
        if isinstance(node, ContentNode):
            out: list[Node] = []
            for cue in self._store.neighbors(
                node.id, edge_type=EdgeType.CUE_OF, direction=Direction.INCOMING
            ):
                if isinstance(cue, CueNode) and cue.status == NodeStatus.ACTIVE:
                    out.append(cue)
            for peer in self._content_candidates_via(node.id, EdgeType.PART_OF_TOPIC, wanted):
                out.append(peer)
            return out
        return []

    def _content_candidates(self, node_id: str, wanted: set[TagFilter] | None) -> list[Node]:
        return self._content_candidates_via(node_id, EdgeType.CUE_OF, wanted)

    def _content_candidates_via(
        self, node_id: str, edge_type: EdgeType, wanted: set[TagFilter] | None
    ) -> list[Node]:
        out: list[Node] = []
        seen: set[str] = set()
        for neighbor in self._store.neighbors(
            node_id, edge_type=edge_type, direction=Direction.OUTGOING
        ):
            content = self._eligible_content(neighbor, wanted, seen)
            if content is not None:
                seen.add(content.id)
                out.append(content)
        return out

    def _as_content(self, node: Node) -> ContentNode | None:
        if isinstance(node, ContentNode):
            return node
        if isinstance(node, CueNode):
            for neighbor in self._store.neighbors(
                node.id, edge_type=EdgeType.CUE_OF, direction=Direction.OUTGOING
            ):
                content = self._resolve_redirect(neighbor)
                if isinstance(content, ContentNode) and content.status == NodeStatus.ACTIVE:
                    return content
        return None

    def _eligible_content(
        self, node: Node, wanted: set[TagFilter] | None, seen: set[str]
    ) -> ContentNode | None:
        content = self._resolve_redirect(node)
        if not isinstance(content, ContentNode):
            return None
        if content.status != NodeStatus.ACTIVE:
            return None
        if content.id in seen:
            return None
        if wanted is not None and not wanted <= self._content_tags(content.id):
            return None
        return content

    def _content_tags(self, content_id: str) -> set[TagFilter]:
        tags: set[TagFilter] = set()
        for node in self._store.neighbors(
            content_id, edge_type=EdgeType.TAGGED_WITH, direction=Direction.OUTGOING
        ):
            if isinstance(node, TagNode):
                tags.add((node.category, node.value))
        return tags

    def _resolve_redirect(self, node: Node) -> Node:
        targets = self._store.neighbors(
            node.id, edge_type=EdgeType.REDIRECTS_TO, direction=Direction.OUTGOING
        )
        return targets[0] if targets else node

"""The read-path service: search, traversal, and LLM-driven reconstruction.

Sits entirely on top of :class:`~delfos.store.base.GraphStore`; it never touches
the database directly. The three primitives are pure graph operations;
``reconstruct`` additionally drives a :class:`~delfos.reconstruct.planner.HopPlanner`.
"""

from __future__ import annotations

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

from .planner import HopPlanner

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

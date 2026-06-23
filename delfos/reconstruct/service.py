"""The read-path service: search, traversal, and LLM-driven reconstruction.

Sits entirely on top of :class:`~delfos.store.base.GraphStore`; it never touches
the database directly. The three primitives are pure graph operations;
``reconstruct`` additionally drives a :class:`~delfos.reconstruct.planner.HopPlanner`.
"""

from __future__ import annotations

from delfos.indexer.embedder import Embedder
from delfos.schema import CueNode, NodeType, TagCategory
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

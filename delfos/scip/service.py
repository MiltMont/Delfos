"""Read-path bridge from a graph ContentNode to SCIP cross-references.

ContentNode IDs assigned during indexing are the SCIP symbol string when SCIP
coverage was present, so ScipService passes content_id directly to the index —
no secondary FK dereference needed. Nodes indexed without SCIP use a fallback
id scheme and return empty results naturally (the id is not in the SCIP index).

v1 scope: results are SCIP-native (relative path + line range + symbol). We do
not resolve a referencing occurrence back to its enclosing ContentNode.
"""

from __future__ import annotations

from collections.abc import Callable

from delfos.schema import ContentNode
from delfos.scip.reader import Occurrence, Relationship, ScipIndex
from delfos.store import GraphStore


class ScipService:
    """Resolve a content node's SCIP cross-references using its id as the symbol."""

    def __init__(self, store: GraphStore, index: ScipIndex) -> None:
        self._store = store
        self._index = index

    def references(self, content_id: str) -> list[tuple[str, Occurrence]]:
        """All non-definition usages of the node's symbol across the repo.

        Returns ``(relative_path, occurrence)`` pairs. Empty when ``content_id``
        is not a SCIP symbol (node was indexed without SCIP coverage).
        """
        self._content_node(content_id)  # validate node exists and is ContentNode
        return self._index.references(content_id)

    def implementations(self, content_id: str) -> list[Relationship]:
        """Symbols the node's symbol implements (SCIP ``is_implementation``)."""
        return self._relationships(content_id, lambda r: r.is_implementation)

    def type_definition(self, content_id: str) -> list[Relationship]:
        """Type-definition symbols for the node's symbol (``is_type_definition``)."""
        return self._relationships(content_id, lambda r: r.is_type_definition)

    def _content_node(self, content_id: str) -> ContentNode:
        node = self._store.get_node(content_id)
        if not isinstance(node, ContentNode):
            raise ValueError(f"no content node with id {content_id!r}")
        return node

    def _relationships(
        self, content_id: str, predicate: Callable[[Relationship], bool]
    ) -> list[Relationship]:
        node = self._content_node(content_id)
        # source_file scopes the symbol_info lookup to document-local symbols first.
        info = self._index.symbol_info(content_id, relative_path=node.source_file)
        if info is None:
            return []
        return [rel for rel in info.relationships if predicate(rel)]

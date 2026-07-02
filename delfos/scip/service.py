"""Read-path bridge from a graph ``ContentNode`` to SCIP cross-references.

:class:`ScipService` mirrors :class:`~delfos.reconstruct.service.ReconstructionService`:
it wraps a :class:`~delfos.store.base.GraphStore` (to resolve a ``content_id`` to
its node) and a :class:`~delfos.scip.reader.ScipIndex` (to resolve that node's
``scip_symbol`` foreign key to cross-reference data). Nothing about
references/implementations/types is materialized as graph edges — the index
stays the single authoritative source and is queried on demand.

v1 scope: results are SCIP-native (relative path + line range + symbol). We do
not resolve a referencing occurrence back to its enclosing ``ContentNode`` —
that needs an enclosing-range index and a ``scip_symbol``→``content_id`` reverse
lookup that does not exist yet (future enhancement).
"""

from __future__ import annotations

from collections.abc import Callable

from delfos.schema import ContentNode
from delfos.scip.reader import Occurrence, Relationship, ScipIndex
from delfos.store import GraphStore


class ScipService:
    """Resolve a content node's SCIP foreign key to cross-reference data."""

    def __init__(self, store: GraphStore, index: ScipIndex) -> None:
        self._store = store
        self._index = index

    def references(self, content_id: str) -> list[tuple[str, Occurrence]]:
        """All non-definition usages of the node's symbol across the repo.

        Returns ``(relative_path, occurrence)`` pairs. The definition occurrence
        is excluded — the caller already knows the definition site from the
        ``ContentNode`` itself. Empty when the node carries no SCIP symbol.
        """
        node = self._content_node(content_id)
        if not node.scip_symbol:
            return []
        return self._index.references(node.scip_symbol)

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
        if not node.scip_symbol:
            return []
        # The symbol's SymbolInformation lives in its defining document, which is
        # the content node's own source file (with external_symbols as fallback).
        info = self._index.symbol_info(node.scip_symbol, relative_path=node.source_file)
        if info is None:
            return []
        return [rel for rel in info.relationships if predicate(rel)]

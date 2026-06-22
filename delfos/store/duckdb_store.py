"""DuckDB-backed :class:`GraphStore` implementation.

Stub only: this fixes the constructor shape and confirms the class satisfies
the :class:`GraphStore` interface. Method bodies are intentionally absent and
raise :class:`NotImplementedError`; they will be filled in once the interface
is reviewed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from delfos.schema import Direction, Edge, EdgeType, Node, NodeType

from .base import GraphStore, IndexedFile, VectorSearchResult


class DuckDBGraphStore(GraphStore):
    """Single-file DuckDB store using the VSS extension for vector search.

    All vectors in the store share one embedding model; ``embedding_dim`` and
    ``embedding_model`` are fixed at construction and enforced on write.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_dim: int,
        embedding_model: str,
    ) -> None:
        self.path = Path(path)
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

    # --- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # --- transactions ------------------------------------------------------

    def begin_transaction(self) -> None:
        raise NotImplementedError

    def commit(self) -> None:
        raise NotImplementedError

    def rollback(self) -> None:
        raise NotImplementedError

    # --- node / edge writes ------------------------------------------------

    def upsert_node(self, node: Node) -> None:
        raise NotImplementedError

    def upsert_edge(self, edge: Edge) -> None:
        raise NotImplementedError

    def delete_node(self, node_id: str) -> None:
        raise NotImplementedError

    def delete_nodes_for_file(self, source_file: str) -> None:
        raise NotImplementedError

    # --- reads -------------------------------------------------------------

    def get_node(self, node_id: str) -> Node | None:
        raise NotImplementedError

    def neighbors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
        direction: Direction = Direction.OUTGOING,
    ) -> list[Node]:
        raise NotImplementedError

    def vector_search(
        self,
        embedding: list[float],
        k: int,
        *,
        node_type: NodeType | None = None,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError

    # --- checkpoint manifest ----------------------------------------------

    def record_indexed_file(self, file_path: str, git_sha: str, indexed_at: datetime) -> None:
        raise NotImplementedError

    def indexed_file_sha(self, file_path: str) -> str | None:
        raise NotImplementedError

    def list_indexed_files(self) -> list[IndexedFile]:
        raise NotImplementedError

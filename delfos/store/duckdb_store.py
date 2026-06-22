"""DuckDB-backed :class:`GraphStore` implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from delfos.schema import Direction, Edge, EdgeType, Node, NodeType

from .base import GraphStore, IndexedFile, VectorSearchResult

_NODE_COLUMNS: list[str] = [
    "id",
    "node_type",
    "source_file",
    "git_sha",
    "indexed_at",
    "status",
    "deleted_at",
    "deleted_by_commit",
    "embedding",
    "embedding_model",
    "embedding_model_version",
    "cue_type",
    "text",
    "category",
    "value",
    "kind",
    "memory_layer",
    "symbol_name",
    "signature",
    "docstring",
    "body",
]

_EDGE_COLUMNS: list[str] = [
    "source_id",
    "target_id",
    "edge_type",
    "source_file",
    "git_sha",
    "indexed_at",
]


class DuckDBGraphStore(GraphStore):
    """Single-file DuckDB store; brute-force cosine vector search.

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
        self._in_txn = False
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(str(self.path))

    # --- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        dim = self.embedding_dim
        self._con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT,
                source_file TEXT,
                git_sha TEXT,
                indexed_at TIMESTAMP,
                status TEXT,
                deleted_at TIMESTAMP,
                deleted_by_commit TEXT,
                embedding DOUBLE[{dim}],
                embedding_model TEXT,
                embedding_model_version TEXT,
                cue_type TEXT,
                text TEXT,
                category TEXT,
                value TEXT,
                kind TEXT,
                memory_layer TEXT,
                symbol_name TEXT,
                signature TEXT,
                docstring TEXT,
                body TEXT
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT,
                target_id TEXT,
                edge_type TEXT,
                source_file TEXT,
                git_sha TEXT,
                indexed_at TIMESTAMP,
                PRIMARY KEY (source_id, target_id, edge_type)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                git_sha TEXT,
                indexed_at TIMESTAMP
            )
            """
        )

    def close(self) -> None:
        self._con.close()

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

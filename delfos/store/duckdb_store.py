"""DuckDB-backed :class:`GraphStore` implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
from pydantic import TypeAdapter

from delfos.schema import Direction, Edge, EdgeType, EmbeddedMixin, Node, NodeType

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

_NODE_ADAPTER: TypeAdapter[Node] = TypeAdapter(Node)


def _node_params(node: Node) -> list[object]:
    data = node.model_dump()
    return [data.get(col) for col in _NODE_COLUMNS]


def _edge_params(edge: Edge) -> list[object]:
    data = edge.model_dump()
    return [data.get(col) for col in _EDGE_COLUMNS]


def _row_to_node(row: tuple[object, ...]) -> Node:
    raw = dict(zip(_NODE_COLUMNS, row, strict=True))
    # extra="forbid" on the models: keep only populated columns so a row for
    # one node type never carries another type's columns into validation.
    data = {k: v for k, v in raw.items() if v is not None}
    return _NODE_ADAPTER.validate_python(data)


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
        # isinstance narrows to CueNode | ContentNode (the EmbeddedMixin types),
        # so node.embedding / node.embedding_model are typed for pyright strict.
        if isinstance(node, EmbeddedMixin) and node.embedding is not None:
            if node.embedding_model != self.embedding_model:
                raise ValueError(
                    f"embedding_model {node.embedding_model!r} "
                    f"does not match store model {self.embedding_model!r}"
                )
            if len(node.embedding) != self.embedding_dim:
                raise ValueError(
                    f"embedding length {len(node.embedding)} != store dim {self.embedding_dim}"
                )
        cols = ", ".join(_NODE_COLUMNS)
        placeholders = ", ".join(["?"] * len(_NODE_COLUMNS))
        self._con.execute(
            f"INSERT OR REPLACE INTO nodes ({cols}) VALUES ({placeholders})",
            _node_params(node),
        )

    def upsert_edge(self, edge: Edge) -> None:
        cols = ", ".join(_EDGE_COLUMNS)
        placeholders = ", ".join(["?"] * len(_EDGE_COLUMNS))
        self._con.execute(
            f"INSERT OR REPLACE INTO edges ({cols}) VALUES ({placeholders})",
            _edge_params(edge),
        )

    def delete_node(self, node_id: str) -> None:
        self._con.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            [node_id, node_id],
        )
        self._con.execute("DELETE FROM nodes WHERE id = ?", [node_id])

    def delete_nodes_for_file(self, source_file: str) -> None:
        self._con.execute(
            """
            DELETE FROM edges
            WHERE source_file = ?
               OR source_id IN (SELECT id FROM nodes WHERE source_file = ?)
               OR target_id IN (SELECT id FROM nodes WHERE source_file = ?)
            """,
            [source_file, source_file, source_file],
        )
        self._con.execute("DELETE FROM nodes WHERE source_file = ?", [source_file])

    # --- reads -------------------------------------------------------------

    def get_node(self, node_id: str) -> Node | None:
        cols = ", ".join(_NODE_COLUMNS)
        row = self._con.execute(f"SELECT {cols} FROM nodes WHERE id = ?", [node_id]).fetchone()
        if row is None:
            return None
        return _row_to_node(row)

    def neighbors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
        direction: Direction = Direction.OUTGOING,
    ) -> list[Node]:
        if direction == Direction.OUTGOING:
            match_col, return_col = "source_id", "target_id"
        else:
            match_col, return_col = "target_id", "source_id"
        n_cols = ", ".join(f"n.{c}" for c in _NODE_COLUMNS)
        sql = (
            f"SELECT {n_cols} FROM edges e "
            f"JOIN nodes n ON n.id = e.{return_col} "
            f"WHERE e.{match_col} = ?"
        )
        params: list[object] = [node_id]
        if edge_type is not None:
            sql += " AND e.edge_type = ?"
            params.append(edge_type)
        rows = self._con.execute(sql, params).fetchall()
        return [_row_to_node(row) for row in rows]

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

"""The ``GraphStore`` abstraction.

No component (indexer, MCP tools, CLI) should ever touch the database directly;
they all go through this interface. Its shape is fully determined by the schema
in :mod:`delfos.schema` and by the prototype decisions in ``decisions.md``.

This module defines signatures and contracts only. Concrete backends live
alongside it (see :mod:`delfos.store.native_store`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from delfos.schema import Direction, Edge, EdgeType, Node, NodeType


class VectorSearchResult(BaseModel):
    """A single hit from :meth:`GraphStore.vector_search`.

    ``score`` is a similarity score where higher means closer. ``node`` is
    populated only when the backend is asked to hydrate full nodes; otherwise
    callers resolve ``node_id`` via :meth:`GraphStore.get_node`.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    score: float
    node: Node | None = None


class IndexedFile(BaseModel):
    """A row of the checkpoint manifest (``decisions.md`` section 6)."""

    model_config = ConfigDict(extra="forbid")

    file_path: str
    git_sha: str
    indexed_at: datetime


class GraphStore(ABC):
    """Single-writer / multi-reader store for the Cue-Tag-Content graph.

    Implementations must operate in WAL-style concurrency: many readers while a
    single writer indexes. All write methods must be safe to call inside an
    open transaction (see :meth:`transaction`).

    Embedding-model invariant: every vector in a given store must come from the
    same embedding model. Backends validate this at initialization and reject
    nodes whose ``embedding_model`` disagrees with the store's configured model.
    """

    # --- lifecycle ---------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Create tables, indexes, and the vector index if absent.

        Idempotent: safe to call on an already-initialized store.
        """

    @abstractmethod
    def close(self) -> None:
        """Flush and release the underlying connection."""

    # --- transactions ------------------------------------------------------

    @abstractmethod
    def begin_transaction(self) -> None:
        """Open a write transaction. Nested transactions are not supported."""

    @abstractmethod
    def commit(self) -> None:
        """Commit the open transaction."""

    @abstractmethod
    def rollback(self) -> None:
        """Roll back the open transaction."""

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Scope a unit of work (one file, per ``decisions.md`` section 6).

        Commits on clean exit, rolls back on exception. This is the atomic unit
        the indexer relies on for crash recovery.
        """
        self.begin_transaction()
        try:
            yield
        except BaseException:
            self.rollback()
            raise
        else:
            self.commit()

    # --- node / edge writes ------------------------------------------------

    @abstractmethod
    def upsert_node(self, node: Node) -> None:
        """Insert ``node`` or replace the existing node with the same ``id``."""

    @abstractmethod
    def upsert_edge(self, edge: Edge) -> None:
        """Insert ``edge`` or replace the matching ``(source, target, type)``."""

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Hard-delete a node and its incident edges.

        Tombstoning (soft delete) is done via :meth:`upsert_node` with
        ``status=NodeStatus.DELETED``; this method is the hard removal used when
        purging a re-indexed file.
        """

    @abstractmethod
    def delete_nodes_for_file(self, source_file: str) -> None:
        """Drop every node and edge sourced from ``source_file``.

        Backs the delete-and-reindex strategy: called before re-indexing a file
        whose ``git_sha`` changed.
        """

    # --- reads -------------------------------------------------------------

    @abstractmethod
    def get_node(self, node_id: str) -> Node | None:
        """Return the node with ``node_id``, or ``None`` if absent."""

    @abstractmethod
    def neighbors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
        direction: Direction = Direction.OUTGOING,
    ) -> list[Node]:
        """Return nodes adjacent to ``node_id``.

        Filtered by ``edge_type`` when given, and by traversal ``direction``.
        """

    @abstractmethod
    def vector_search(
        self,
        embedding: list[float],
        k: int,
        *,
        node_type: NodeType | None = None,
    ) -> list[VectorSearchResult]:
        """Approximate k-NN over stored vectors, best match first.

        ``node_type`` restricts the search (cues are the usual target). The
        query ``embedding`` must come from the store's configured model.
        """

    # --- checkpoint manifest ----------------------------------------------

    @abstractmethod
    def record_indexed_file(self, file_path: str, git_sha: str, indexed_at: datetime) -> None:
        """Mark ``(file_path, git_sha)`` as fully committed in the manifest.

        Must be written within the same transaction as the file's nodes so the
        manifest never claims a file that was not actually committed.
        """

    @abstractmethod
    def indexed_file_sha(self, file_path: str) -> str | None:
        """Return the last committed ``git_sha`` for ``file_path``, if any."""

    @abstractmethod
    def list_indexed_files(self) -> list[IndexedFile]:
        """Return the full checkpoint manifest."""

"""NativeGraphStore — GraphStore backed by the libdelfos C++ engine.

Drop-in replacement for DuckDBGraphStore.  All product code (indexer, MCP
tools) must talk through the GraphStore ABC; this module is an implementation
detail.

The _delfos extension module is built via scikit-build-core (pip install -e .)
and lives at delfos/_delfos.so.
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from pathlib import Path

from delfos import _delfos
from delfos.schema import (
    ContentKind,
    ContentNode,
    CueNode,
    CueType,
    Direction,
    Edge,
    EdgeType,
    MemoryLayer,
    Node,
    NodeStatus,
    NodeType,
    TagCategory,
    TagNode,
)
from delfos.store.base import GraphStore, IndexedFile, VectorSearchResult

# ─────────────────────────────────────────────────────────────────────────────
# Enum conversion tables
# ─────────────────────────────────────────────────────────────────────────────

_NODE_TYPE_TO_NATIVE: dict[NodeType, int] = {
    NodeType.CUE: _delfos.NODE_TYPE_CUE,
    NodeType.TAG: _delfos.NODE_TYPE_TAG,
    NodeType.CONTENT: _delfos.NODE_TYPE_CONTENT,
}
_NODE_TYPE_FROM_NATIVE: dict[int, NodeType] = {v: k for k, v in _NODE_TYPE_TO_NATIVE.items()}

_STATUS_TO_NATIVE: dict[NodeStatus, int] = {
    NodeStatus.ACTIVE: _delfos.NODE_STATUS_ACTIVE,
    NodeStatus.DELETED: _delfos.NODE_STATUS_DELETED,
}
_STATUS_FROM_NATIVE: dict[int, NodeStatus] = {v: k for k, v in _STATUS_TO_NATIVE.items()}

_EDGE_TYPE_TO_NATIVE: dict[EdgeType, int] = {
    EdgeType.CUE_OF: _delfos.EDGE_TYPE_CUE_OF,
    EdgeType.TAGGED_WITH: _delfos.EDGE_TYPE_TAGGED_WITH,
    EdgeType.PART_OF_TOPIC: _delfos.EDGE_TYPE_PART_OF_TOPIC,
    EdgeType.REDIRECTS_TO: _delfos.EDGE_TYPE_REDIRECTS_TO,
}
_EDGE_TYPE_FROM_NATIVE: dict[int, EdgeType] = {v: k for k, v in _EDGE_TYPE_TO_NATIVE.items()}

_CUE_TYPE_TO_NATIVE: dict[CueType, int] = {
    CueType.SYMBOL: _delfos.CUE_TYPE_SYMBOL,
    CueType.CONCEPT: _delfos.CUE_TYPE_CONCEPT,
    CueType.ERROR_MESSAGE: _delfos.CUE_TYPE_ERROR_MESSAGE,
}
_CUE_TYPE_FROM_NATIVE: dict[int, CueType] = {v: k for k, v in _CUE_TYPE_TO_NATIVE.items()}

_TAG_CAT_TO_NATIVE: dict[TagCategory, int] = {
    TagCategory.MODULE_PATH: _delfos.TAG_CATEGORY_MODULE_PATH,
    TagCategory.ARCH_LAYER: _delfos.TAG_CATEGORY_ARCH_LAYER,
    TagCategory.PATTERN_TYPE: _delfos.TAG_CATEGORY_PATTERN_TYPE,
    TagCategory.LANG_CONSTRUCT: _delfos.TAG_CATEGORY_LANG_CONSTRUCT,
    TagCategory.LANGUAGE: _delfos.TAG_CATEGORY_LANGUAGE,
}
_TAG_CAT_FROM_NATIVE: dict[int, TagCategory] = {v: k for k, v in _TAG_CAT_TO_NATIVE.items()}

_KIND_TO_NATIVE: dict[ContentKind, int] = {
    ContentKind.FUNCTION: _delfos.CONTENT_KIND_FUNCTION,
    ContentKind.CLASS: _delfos.CONTENT_KIND_CLASS,
    ContentKind.MODULE: _delfos.CONTENT_KIND_MODULE,
    ContentKind.COMMIT: _delfos.CONTENT_KIND_COMMIT,
    ContentKind.TEST: _delfos.CONTENT_KIND_TEST,
}
_KIND_FROM_NATIVE: dict[int, ContentKind] = {v: k for k, v in _KIND_TO_NATIVE.items()}

_LAYER_TO_NATIVE: dict[MemoryLayer, int] = {
    MemoryLayer.EPISODIC: _delfos.MEMORY_LAYER_EPISODIC,
    MemoryLayer.SEMANTIC: _delfos.MEMORY_LAYER_SEMANTIC,
    MemoryLayer.TOPIC: _delfos.MEMORY_LAYER_TOPIC,
}
_LAYER_FROM_NATIVE: dict[int, MemoryLayer] = {v: k for k, v in _LAYER_TO_NATIVE.items()}

_DIR_TO_NATIVE: dict[Direction, int] = {
    Direction.OUTGOING: _delfos.DIRECTION_OUTGOING,
    Direction.INCOMING: _delfos.DIRECTION_INCOMING,
}

# ─────────────────────────────────────────────────────────────────────────────
# Datetime ↔ microseconds (UTC-stable: calendar.timegm treats input as UTC)
# ─────────────────────────────────────────────────────────────────────────────


def _dt_to_us(dt: datetime) -> int:
    return calendar.timegm(dt.timetuple()) * 1_000_000 + dt.microsecond


def _us_to_dt(us: int) -> datetime:
    # Return a naive datetime (no tzinfo) matching DuckDB's TIMESTAMP behaviour.
    return datetime.fromtimestamp(us / 1_000_000, tz=UTC).replace(tzinfo=None)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic ↔ native NodeData conversion helpers
# ─────────────────────────────────────────────────────────────────────────────


def _pydantic_to_native(node: Node) -> _delfos.NodeData:
    nd = _delfos.NodeData()
    nd.id = node.id
    nd.status = _STATUS_TO_NATIVE[node.status]
    nd.indexed_at = _dt_to_us(node.indexed_at)
    nd.deleted_at = _dt_to_us(node.deleted_at) if node.deleted_at is not None else 0
    nd.deleted_by_commit = node.deleted_by_commit or ""

    if isinstance(node, CueNode):
        nd.type = _delfos.NODE_TYPE_CUE
        nd.source_file = node.source_file
        nd.git_sha = node.git_sha
        nd.cue_type = _CUE_TYPE_TO_NATIVE[node.cue_type]
        nd.text = node.text
        nd.embedding = list(node.embedding) if node.embedding is not None else []
        nd.embedding_model = node.embedding_model or ""
        nd.embedding_model_version = node.embedding_model_version or ""

    elif isinstance(node, TagNode):
        nd.type = _delfos.NODE_TYPE_TAG
        nd.source_file = node.source_file or ""
        nd.git_sha = node.git_sha or ""
        nd.category = _TAG_CAT_TO_NATIVE[node.category]
        nd.value = node.value

    else:
        # ContentNode — the only remaining branch of the discriminated union
        assert isinstance(node, ContentNode)
        nd.type = _delfos.NODE_TYPE_CONTENT
        nd.source_file = node.source_file
        nd.git_sha = node.git_sha
        nd.kind = _KIND_TO_NATIVE[node.kind]
        nd.memory_layer = _LAYER_TO_NATIVE[node.memory_layer]
        nd.symbol_name = node.symbol_name or ""
        nd.scip_symbol = node.scip_symbol or ""
        nd.signature = node.signature or ""
        nd.docstring = node.docstring or ""
        nd.body = node.body
        nd.embedding = list(node.embedding) if node.embedding is not None else []
        nd.embedding_model = node.embedding_model or ""
        nd.embedding_model_version = node.embedding_model_version or ""

    return nd


def _native_to_pydantic(nd: _delfos.NodeData) -> Node:
    status = _STATUS_FROM_NATIVE[nd.status]
    indexed_at = _us_to_dt(nd.indexed_at)
    deleted_at = _us_to_dt(nd.deleted_at) if nd.deleted_at != 0 else None
    deleted_by_commit = nd.deleted_by_commit or None

    common: dict[str, object] = {
        "id": nd.id,
        "indexed_at": indexed_at,
        "status": status,
        "deleted_at": deleted_at,
        "deleted_by_commit": deleted_by_commit,
    }

    node_type = _NODE_TYPE_FROM_NATIVE[nd.type]

    if node_type == NodeType.CUE:
        emb = list(nd.embedding) if nd.embedding else None
        return CueNode(
            **common,  # type: ignore[arg-type]
            source_file=nd.source_file,
            git_sha=nd.git_sha,
            cue_type=_CUE_TYPE_FROM_NATIVE[nd.cue_type],
            text=nd.text,
            embedding=emb,
            embedding_model=nd.embedding_model or None,
            embedding_model_version=nd.embedding_model_version or None,
        )

    if node_type == NodeType.TAG:
        return TagNode(
            **common,  # type: ignore[arg-type]
            source_file=nd.source_file or None,
            git_sha=nd.git_sha or None,
            category=_TAG_CAT_FROM_NATIVE[nd.category],
            value=nd.value,
        )

    # ContentNode
    emb = list(nd.embedding) if nd.embedding else None
    return ContentNode(
        **common,  # type: ignore[arg-type]
        source_file=nd.source_file,
        git_sha=nd.git_sha,
        kind=_KIND_FROM_NATIVE[nd.kind],
        memory_layer=_LAYER_FROM_NATIVE[nd.memory_layer],
        symbol_name=nd.symbol_name or None,
        scip_symbol=nd.scip_symbol or None,
        signature=nd.signature or None,
        docstring=nd.docstring or None,
        body=nd.body,
        embedding=emb,
        embedding_model=nd.embedding_model or None,
        embedding_model_version=nd.embedding_model_version or None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NativeGraphStore
# ─────────────────────────────────────────────────────────────────────────────


class NativeGraphStore(GraphStore):
    """GraphStore backed by the libdelfos C++ engine.

    Parameters
    ----------
    path:
        Directory where the snapshot files are persisted.  Created on first
        use.  Safe to pass an existing directory — ``initialize()`` loads the
        stored graph automatically.
    embedding_dim:
        Expected dimensionality of every vector written to this store.
    embedding_model:
        Identifier for the embedding model.  Every node embedding stored must
        have been produced by this model; ``upsert_node`` rejects mismatches.
    """

    def __init__(
        self,
        path: Path,
        *,
        embedding_dim: int,
        embedding_model: str,
    ) -> None:
        self._store = _delfos.Store(str(path), embedding_dim, embedding_model)
        self._embedding_dim = embedding_dim
        self._embedding_model = embedding_model

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self._store.initialize()

    def close(self) -> None:
        self._store.close()

    # ── transactions ───────────────────────────────────────────────────────────

    def begin_transaction(self) -> None:
        self._store.begin_transaction()

    def commit(self) -> None:
        self._store.commit()

    def rollback(self) -> None:
        self._store.rollback()

    # ── writes ─────────────────────────────────────────────────────────────────

    def upsert_node(self, node: Node) -> None:
        emb = getattr(node, "embedding", None)
        if emb is not None:
            emb_model = getattr(node, "embedding_model", None)
            if emb_model != self._embedding_model:
                raise ValueError(
                    f"Node embedding_model {emb_model!r} does not match store model "
                    f"{self._embedding_model!r}"
                )
            if len(emb) != self._embedding_dim:
                raise ValueError(
                    f"Node embedding dim {len(emb)} != store dim {self._embedding_dim}"
                )
        self._store.upsert_node(_pydantic_to_native(node))

    def upsert_edge(self, edge: Edge) -> None:
        indexed_at = _dt_to_us(edge.indexed_at) if edge.indexed_at is not None else 0
        self._store.upsert_edge(
            edge.source_id,
            edge.target_id,
            _EDGE_TYPE_TO_NATIVE[edge.edge_type],
            edge.source_file or "",
            edge.git_sha or "",
            indexed_at,
        )

    def delete_node(self, node_id: str) -> None:
        self._store.delete_node(node_id)

    def delete_nodes_for_file(self, source_file: str) -> None:
        self._store.delete_nodes_for_file(source_file)

    # ── reads ──────────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Node | None:
        nd = self._store.get_node(node_id)
        if nd is None:
            return None
        return _native_to_pydantic(nd)

    def neighbors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
        direction: Direction = Direction.OUTGOING,
    ) -> list[Node]:
        et_int: int | None = _EDGE_TYPE_TO_NATIVE[edge_type] if edge_type is not None else None
        native_nodes = self._store.neighbors(
            node_id,
            _DIR_TO_NATIVE[direction],
            et_int,
        )
        return [_native_to_pydantic(nd) for nd in native_nodes]

    def vector_search(
        self,
        embedding: list[float],
        k: int,
        *,
        node_type: NodeType | None = None,
    ) -> list[VectorSearchResult]:
        nt_int: int | None = _NODE_TYPE_TO_NATIVE[node_type] if node_type is not None else None
        pairs = self._store.vector_search(embedding, k, nt_int)
        return [VectorSearchResult(node_id=nid, score=score) for nid, score in pairs]

    # ── manifest ───────────────────────────────────────────────────────────────

    def record_indexed_file(self, file_path: str, git_sha: str, indexed_at: datetime) -> None:
        self._store.record_indexed_file(file_path, git_sha, _dt_to_us(indexed_at))

    def indexed_file_sha(self, file_path: str) -> str | None:
        return self._store.indexed_file_sha(file_path)  # type: ignore[return-value]

    def list_indexed_files(self) -> list[IndexedFile]:
        rows = self._store.list_indexed_files()
        return [
            IndexedFile(file_path=fp, git_sha=sha, indexed_at=_us_to_dt(us)) for fp, sha, us in rows
        ]

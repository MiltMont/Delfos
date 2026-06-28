# Type stubs for the _delfos nanobind extension module.
# These allow pyright (strict mode) to type-check native_store.py.

# ── Enum integer constants ─────────────────────────────────────────────────────
NODE_TYPE_CUE: int
NODE_TYPE_TAG: int
NODE_TYPE_CONTENT: int

NODE_STATUS_ACTIVE: int
NODE_STATUS_DELETED: int

EDGE_TYPE_CUE_OF: int
EDGE_TYPE_TAGGED_WITH: int
EDGE_TYPE_PART_OF_TOPIC: int
EDGE_TYPE_REDIRECTS_TO: int

CUE_TYPE_SYMBOL: int
CUE_TYPE_CONCEPT: int
CUE_TYPE_ERROR_MESSAGE: int

TAG_CATEGORY_MODULE_PATH: int
TAG_CATEGORY_ARCH_LAYER: int
TAG_CATEGORY_PATTERN_TYPE: int
TAG_CATEGORY_LANG_CONSTRUCT: int
TAG_CATEGORY_LANGUAGE: int

CONTENT_KIND_FUNCTION: int
CONTENT_KIND_CLASS: int
CONTENT_KIND_MODULE: int
CONTENT_KIND_COMMIT: int
CONTENT_KIND_TEST: int

MEMORY_LAYER_EPISODIC: int
MEMORY_LAYER_SEMANTIC: int
MEMORY_LAYER_TOPIC: int

DIRECTION_OUTGOING: int
DIRECTION_INCOMING: int

INVALID_NODE: int

# ── NodeData ───────────────────────────────────────────────────────────────────

class NodeData:
    id: str
    type: int
    status: int
    indexed_at: int
    deleted_at: int
    deleted_by_commit: str
    source_file: str
    git_sha: str
    cue_type: int
    text: str
    category: int
    value: str
    kind: int
    memory_layer: int
    symbol_name: str
    signature: str
    docstring: str
    body: str
    embedding: list[float]
    embedding_model: str
    embedding_model_version: str

    def __init__(self) -> None: ...

# ── Store ──────────────────────────────────────────────────────────────────────

class Store:
    def __init__(self, path: str, embedding_dim: int, embedding_model: str) -> None: ...

    # Lifecycle
    def initialize(self) -> None: ...
    def close(self) -> None: ...

    # Transactions
    def begin_transaction(self) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...

    # Writes
    def upsert_node(self, node: NodeData) -> None: ...
    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: int,
        source_file: str,
        git_sha: str,
        indexed_at: int,
    ) -> None: ...
    def delete_node(self, node_id: str) -> None: ...
    def delete_nodes_for_file(self, source_file: str) -> None: ...

    # Reads
    def get_node(self, node_id: str) -> NodeData | None: ...
    def neighbors(
        self,
        node_id: str,
        direction: int,
        edge_type: int | None = None,
    ) -> list[NodeData]: ...
    def vector_search(
        self,
        embedding: list[float],
        k: int,
        node_type: int | None = None,
    ) -> list[tuple[str, float]]: ...

    # Manifest
    def record_indexed_file(self, file_path: str, git_sha: str, indexed_at: int) -> None: ...
    def indexed_file_sha(self, file_path: str) -> str | None: ...
    def list_indexed_files(self) -> list[tuple[str, str, int]]: ...

    # Config accessors
    def embedding_dim(self) -> int: ...
    def embedding_model(self) -> str: ...

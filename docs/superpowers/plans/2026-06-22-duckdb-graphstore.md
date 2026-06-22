# DuckDBGraphStore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the concrete `DuckDBGraphStore` backend so every `GraphStore` method works against a single-file DuckDB database, built test-first.

**Architecture:** One `nodes` table (typed columns, `node_type` discriminator, `embedding` as a fixed-size `DOUBLE[dim]` column), one `edges` table keyed on `(source_id, target_id, edge_type)`, and one `indexed_files` manifest table — all on a single read-write DuckDB connection. `vector_search` ranks brute-force with `array_cosine_distance` (the documented swap seam for a future HNSW index). Rows hydrate back to Pydantic models via `TypeAdapter(Node)`.

**Tech Stack:** Python 3.12+, DuckDB (`duckdb>=1.1.0`), Pydantic v2, pytest.

## Global Constraints

- Python `>=3.12`; target `py312`.
- Pyright runs in **strict** mode — all new code must be fully typed. Run `uv run pyright` (expect `0 errors`) as the last check of every task.
- Ruff lint (`E`, `F`, `I`, `UP`, `B`) and format must pass: `uv run ruff check .` and `uv run ruff format .`.
- Line length 100.
- All Pydantic models use `extra="forbid"` — row→model hydration must drop columns that don't belong to the target type (see the `_row_to_node` helper in Task 2).
- No component may touch DuckDB except this class.
- Embedding-model invariant: every stored vector must match the store's configured `embedding_model`; reject on mismatch.
- Spec: `docs/superpowers/specs/2026-06-22-duckdb-graphstore-design.md`.

---

### Task 1: Test scaffolding + lifecycle (initialize / close / construction)

Adds pytest, the temp-file fixture, the three table schemas, and idempotent `initialize`.

**Files:**
- Modify: `pyproject.toml` (add `pytest` to `[dependency-groups].dev`)
- Modify: `delfos/store/duckdb_store.py`
- Create: `tests/__init__.py`, `tests/store/__init__.py` (empty package markers)
- Create: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `GraphStore` (base), `delfos.schema` models.
- Produces: `DuckDBGraphStore(path, *, embedding_dim, embedding_model)` with a live `self._con: duckdb.DuckDBPyConnection`, `self._in_txn: bool`, working `initialize()` (idempotent) and `close()`. Module constant `_NODE_COLUMNS: list[str]` and `_EDGE_COLUMNS: list[str]` (used by later tasks).

- [ ] **Step 1: Add pytest to dev dependencies**

In `pyproject.toml`, change the dev group to:

```toml
[dependency-groups]
dev = [
    "ruff>=0.6.0",
    "pyright>=1.1.380",
    "pytest>=8.0.0",
]
```

Then run: `uv sync`
Expected: pytest installed.

Create empty package markers so pytest and pyright treat the test tree unambiguously:

```bash
mkdir -p tests/store && touch tests/__init__.py tests/store/__init__.py
```

- [ ] **Step 2: Write the failing lifecycle test**

Create `tests/store/test_duckdb_store.py`:

```python
from __future__ import annotations

from collections.abc import Iterator

import pytest

from delfos.store.duckdb_store import DuckDBGraphStore

EMBEDDING_DIM = 8
EMBEDDING_MODEL = "fake-v1"


@pytest.fixture
def store(tmp_path) -> Iterator[DuckDBGraphStore]:
    s = DuckDBGraphStore(
        tmp_path / "t.duckdb",
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
    )
    s.initialize()
    yield s
    s.close()


def test_initialize_is_idempotent(store: DuckDBGraphStore) -> None:
    # Second call must not raise.
    store.initialize()
    tables = {
        row[0]
        for row in store._con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    assert {"nodes", "edges", "indexed_files"} <= tables


def test_constructor_stores_config(store: DuckDBGraphStore) -> None:
    assert store.embedding_dim == EMBEDDING_DIM
    assert store.embedding_model == EMBEDDING_MODEL
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/store/test_duckdb_store.py -v`
Expected: FAIL — `initialize` raises `NotImplementedError`.

- [ ] **Step 4: Implement lifecycle + schema**

Replace the body of `delfos/store/duckdb_store.py` with the constructor, column constants, and lifecycle methods (leave the other methods raising `NotImplementedError` for now):

```python
"""DuckDB-backed :class:`GraphStore` implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

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


class DuckDBGraphStore(GraphStore):
    """Single-file DuckDB store; brute-force cosine vector search."""

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
```

Keep the remaining `GraphStore` methods (`begin_transaction`, `commit`, `rollback`, `upsert_node`, `upsert_edge`, `delete_node`, `delete_nodes_for_file`, `get_node`, `neighbors`, `vector_search`, `record_indexed_file`, `indexed_file_sha`, `list_indexed_files`) as stubs that `raise NotImplementedError` for now.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/store/test_duckdb_store.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add pyproject.toml uv.lock delfos/store/duckdb_store.py tests/store/test_duckdb_store.py
git commit -m "feat(store): DuckDBGraphStore lifecycle + schema"
```

Expected: ruff clean, pyright `0 errors`.

---

### Task 2: upsert_node + get_node round-trip

Adds the node↔row helpers and the first read/write pair for all three node types.

**Files:**
- Modify: `delfos/store/duckdb_store.py`
- Modify: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `_NODE_COLUMNS`, `self._con` (Task 1).
- Produces: `upsert_node(node) -> None`, `get_node(node_id) -> Node | None`, module-level `_NODE_ADAPTER: TypeAdapter[Node]`, helpers `_node_params(node) -> list[object]` and `_row_to_node(row) -> Node`. Test helpers `make_cue`, `make_tag`, `make_content` (defined in the test file) reused by later tasks.

- [ ] **Step 1: Write failing round-trip tests**

Append to `tests/store/test_duckdb_store.py` (add imports at the top of the file):

```python
from datetime import datetime

from delfos.schema import (
    ContentKind,
    ContentNode,
    CueNode,
    CueType,
    MemoryLayer,
    TagCategory,
    TagNode,
)

NOW = datetime(2026, 6, 22, 12, 0, 0)


def make_cue(node_id: str = "cue-1", embedding: list[float] | None = None) -> CueNode:
    return CueNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text="load_config",
        embedding=embedding,
        embedding_model=EMBEDDING_MODEL if embedding is not None else None,
    )


def make_tag(node_id: str = "tag-1") -> TagNode:
    return TagNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        category=TagCategory.MODULE_PATH,
        value="delfos.config",
    )


def make_content(
    node_id: str = "content-1", embedding: list[float] | None = None
) -> ContentNode:
    return ContentNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="load_config",
        body="def load_config(): ...",
        embedding=embedding,
        embedding_model=EMBEDDING_MODEL if embedding is not None else None,
    )


def vec(seed: float) -> list[float]:
    return [seed + i for i in range(EMBEDDING_DIM)]


def test_roundtrip_cue(store: DuckDBGraphStore) -> None:
    node = make_cue(embedding=vec(0.1))
    store.upsert_node(node)
    assert store.get_node("cue-1") == node


def test_roundtrip_tag(store: DuckDBGraphStore) -> None:
    node = make_tag()
    store.upsert_node(node)
    assert store.get_node("tag-1") == node


def test_roundtrip_content(store: DuckDBGraphStore) -> None:
    node = make_content(embedding=vec(0.2))
    store.upsert_node(node)
    assert store.get_node("content-1") == node


def test_get_node_missing_returns_none(store: DuckDBGraphStore) -> None:
    assert store.get_node("nope") is None


def test_upsert_node_replaces(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue(embedding=vec(0.1)))
    updated = make_cue(embedding=vec(0.9))
    store.upsert_node(updated)
    assert store.get_node("cue-1") == updated
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "roundtrip or missing or replaces" -v`
Expected: FAIL — `upsert_node` raises `NotImplementedError`.

- [ ] **Step 3: Implement helpers + upsert_node + get_node**

In `delfos/store/duckdb_store.py`, add `TypeAdapter` to the pydantic import and define the adapter + helpers at module level (below the column constants):

```python
from pydantic import TypeAdapter

_NODE_ADAPTER: TypeAdapter[Node] = TypeAdapter(Node)


def _node_params(node: Node) -> list[object]:
    data = node.model_dump()
    return [data.get(col) for col in _NODE_COLUMNS]


def _row_to_node(row: tuple[object, ...]) -> Node:
    raw = dict(zip(_NODE_COLUMNS, row, strict=True))
    # extra="forbid" on the models: keep only populated columns so a row for
    # one node type never carries another type's columns into validation.
    data = {k: v for k, v in raw.items() if v is not None}
    return _NODE_ADAPTER.validate_python(data)
```

Then implement the methods on the class:

```python
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

    def get_node(self, node_id: str) -> Node | None:
        cols = ", ".join(_NODE_COLUMNS)
        row = self._con.execute(
            f"SELECT {cols} FROM nodes WHERE id = ?", [node_id]
        ).fetchone()
        if row is None:
            return None
        return _row_to_node(row)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "roundtrip or missing or replaces" -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add delfos/store/duckdb_store.py tests/store/test_duckdb_store.py
git commit -m "feat(store): upsert_node + get_node round-trip"
```

Expected: ruff clean, pyright `0 errors`.

---

### Task 3: Embedding-invariant validation — tests

Locks the embedding-model and dimension guards with explicit tests. **No production code in this task** — the guards already ship in Task 2's `upsert_node`; this task only proves them, so Step 2 verifies a *pass*, not a red phase.

**Files:**
- Modify: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `upsert_node` validation (Task 2), `make_cue` / `vec` helpers.
- Produces: nothing new — validation already exists; this task proves it.

- [ ] **Step 1: Write failing validation tests**

Append to `tests/store/test_duckdb_store.py`:

```python
def test_upsert_rejects_wrong_embedding_model(store: DuckDBGraphStore) -> None:
    bad = CueNode(
        id="cue-x",
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text="x",
        embedding=vec(0.1),
        embedding_model="other-model",
    )
    with pytest.raises(ValueError, match="does not match store model"):
        store.upsert_node(bad)


def test_upsert_rejects_wrong_embedding_dim(store: DuckDBGraphStore) -> None:
    bad = make_cue(embedding=[0.1, 0.2, 0.3])  # dim 3, store expects 8
    with pytest.raises(ValueError, match="!= store dim"):
        store.upsert_node(bad)
```

- [ ] **Step 2: Run to verify pass**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "rejects" -v`
Expected: PASS (2 passed). (Validation was implemented in Task 2; these tests confirm it.)

- [ ] **Step 3: Lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add tests/store/test_duckdb_store.py
git commit -m "test(store): embedding-invariant validation"
```

Expected: ruff clean, pyright `0 errors`.

---

### Task 4: upsert_edge + neighbors

Adds edge writes and directional neighbor traversal with an optional type filter.

**Files:**
- Modify: `delfos/store/duckdb_store.py`
- Modify: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `_EDGE_COLUMNS`, `_NODE_COLUMNS`, `_row_to_node`, `upsert_node` / `get_node`.
- Produces: `upsert_edge(edge) -> None`, `neighbors(node_id, *, edge_type=None, direction=Direction.OUTGOING) -> list[Node]`.

- [ ] **Step 1: Write failing edge/neighbor tests**

Append to `tests/store/test_duckdb_store.py`. **Merge** `Direction`, `Edge`, `EdgeType` into the existing `from delfos.schema import (...)` block from Task 2 — do **not** add a second `from delfos.schema import` line, or `ruff check` will fail with `I001` (it is not auto-fixed by `ruff format`).

```python
# (Direction, Edge, EdgeType now come from the merged Task 2 import block.)


def _edge(src: str, tgt: str, etype: EdgeType = EdgeType.CUE_OF) -> Edge:
    return Edge(source_id=src, target_id=tgt, edge_type=etype, source_file="a.py")


def test_neighbors_outgoing(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    result = store.neighbors("cue-1", direction=Direction.OUTGOING)
    assert [n.id for n in result] == ["content-1"]


def test_neighbors_incoming(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    result = store.neighbors("content-1", direction=Direction.INCOMING)
    assert [n.id for n in result] == ["cue-1"]


def test_neighbors_filters_by_edge_type(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_node(make_tag("tag-1"))
    store.upsert_edge(_edge("cue-1", "content-1", EdgeType.CUE_OF))
    store.upsert_edge(_edge("cue-1", "tag-1", EdgeType.TAGGED_WITH))
    result = store.neighbors("cue-1", edge_type=EdgeType.TAGGED_WITH)
    assert [n.id for n in result] == ["tag-1"]


def test_upsert_edge_replaces(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))  # same triple
    count = store._con.execute("SELECT count(*) FROM edges").fetchone()
    assert count is not None and count[0] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "neighbors or upsert_edge" -v`
Expected: FAIL — `upsert_edge` raises `NotImplementedError`.

- [ ] **Step 3: Implement upsert_edge + neighbors**

Add a module-level edge param helper next to `_node_params`:

```python
def _edge_params(edge: Edge) -> list[object]:
    data = edge.model_dump()
    return [data.get(col) for col in _EDGE_COLUMNS]
```

Implement the methods on the class:

```python
    def upsert_edge(self, edge: Edge) -> None:
        cols = ", ".join(_EDGE_COLUMNS)
        placeholders = ", ".join(["?"] * len(_EDGE_COLUMNS))
        self._con.execute(
            f"INSERT OR REPLACE INTO edges ({cols}) VALUES ({placeholders})",
            _edge_params(edge),
        )

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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "neighbors or upsert_edge" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add delfos/store/duckdb_store.py tests/store/test_duckdb_store.py
git commit -m "feat(store): upsert_edge + neighbors traversal"
```

Expected: ruff clean, pyright `0 errors`.

---

### Task 5: delete_node + delete_nodes_for_file

Adds hard deletion of single nodes and whole-file purges, both cascading to edges.

**Files:**
- Modify: `delfos/store/duckdb_store.py`
- Modify: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `upsert_node` / `upsert_edge` / `get_node` / `neighbors`.
- Produces: `delete_node(node_id) -> None`, `delete_nodes_for_file(source_file) -> None`.
- **Contract:** `delete_nodes_for_file` issues two statements (edges, then nodes) and deliberately does **not** open its own transaction — the indexer calls it inside `transaction()` so the delete and the subsequent re-insert commit as one atomic file re-index (the one-file-per-transaction model). Self-wrapping would break that composition.

- [ ] **Step 1: Write failing deletion tests**

Append to `tests/store/test_duckdb_store.py`:

```python
def test_delete_node_removes_node_and_incident_edges(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    store.delete_node("cue-1")
    assert store.get_node("cue-1") is None
    edge_count = store._con.execute("SELECT count(*) FROM edges").fetchone()
    assert edge_count is not None and edge_count[0] == 0


def test_delete_nodes_for_file_removes_nodes_and_edges(store: DuckDBGraphStore) -> None:
    # Two nodes from a.py, one cross-file REDIRECTS_TO edge into a.py from b.py.
    store.upsert_node(make_cue("cue-1"))  # source_file a.py
    store.upsert_node(make_content("content-1"))  # source_file a.py
    keep = make_tag("tag-keep")
    keep_in_b = keep.model_copy(update={"source_file": "b.py"})
    store.upsert_node(keep_in_b)
    store.upsert_edge(_edge("cue-1", "content-1"))  # file-scoped, a.py
    cross = Edge(
        source_id="tag-keep",
        target_id="cue-1",
        edge_type=EdgeType.REDIRECTS_TO,
        source_file="b.py",
    )
    store.upsert_edge(cross)  # provenance b.py but touches a node in a.py

    store.delete_nodes_for_file("a.py")

    assert store.get_node("cue-1") is None
    assert store.get_node("content-1") is None
    assert store.get_node("tag-keep") is not None  # b.py node survives
    edge_count = store._con.execute("SELECT count(*) FROM edges").fetchone()
    assert edge_count is not None and edge_count[0] == 0  # both edges gone


def test_delete_nodes_for_file_clears_null_provenance_edge(store: DuckDBGraphStore) -> None:
    # An edge with source_file=None is removed via the source_id/target_id
    # fallback, not the source_file clause.
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(
        Edge(source_id="cue-1", target_id="content-1", edge_type=EdgeType.CUE_OF)
    )  # source_file defaults to None
    store.delete_nodes_for_file("a.py")
    edge_count = store._con.execute("SELECT count(*) FROM edges").fetchone()
    assert edge_count is not None and edge_count[0] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "delete" -v`
Expected: FAIL — `delete_node` raises `NotImplementedError`.

- [ ] **Step 3: Implement deletes**

Implement on the class:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "delete" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add delfos/store/duckdb_store.py tests/store/test_duckdb_store.py
git commit -m "feat(store): delete_node + delete_nodes_for_file"
```

Expected: ruff clean, pyright `0 errors`.

---

### Task 6: vector_search (brute-force cosine)

Adds k-NN over cue/content embeddings, ordered by cosine similarity, with a `node_type` filter.

**Files:**
- Modify: `delfos/store/duckdb_store.py`
- Modify: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `self.embedding_dim`, `upsert_node`.
- Produces: `vector_search(embedding, k, *, node_type=None) -> list[VectorSearchResult]`. Results ordered best-first; `score = 1 - array_cosine_distance`; `node` left `None`; NULL-embedding nodes excluded.

- [ ] **Step 1: Write failing vector-search tests**

Append to `tests/store/test_duckdb_store.py`:

```python
def _unit(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


def test_vector_search_orders_by_similarity(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-a", embedding=_unit(0)))
    store.upsert_node(make_cue("cue-b", embedding=_unit(1)))
    results = store.vector_search(_unit(0), k=2)
    assert [r.node_id for r in results] == ["cue-a", "cue-b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].node is None


def test_vector_search_respects_k(store: DuckDBGraphStore) -> None:
    for i in range(5):
        store.upsert_node(make_cue(f"cue-{i}", embedding=_unit(i % EMBEDDING_DIM)))
    assert len(store.vector_search(_unit(0), k=2)) == 2


def test_vector_search_filters_by_node_type(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1", embedding=_unit(0)))
    store.upsert_node(make_content("content-1", embedding=_unit(0)))
    results = store.vector_search(_unit(0), k=5, node_type=NodeType.CUE)
    assert [r.node_id for r in results] == ["cue-1"]


def test_vector_search_skips_null_embeddings(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_tag("tag-1"))  # no embedding
    store.upsert_node(make_cue("cue-1", embedding=_unit(0)))
    results = store.vector_search(_unit(0), k=5)
    assert [r.node_id for r in results] == ["cue-1"]
```

Merge `NodeType` into the existing `from delfos.schema import (...)` block — do not add a separate import line (see the `I001` note in Task 4).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "vector_search" -v`
Expected: FAIL — `vector_search` raises `NotImplementedError`.

- [ ] **Step 3: Implement vector_search**

Implement on the class:

```python
    def vector_search(
        self,
        embedding: list[float],
        k: int,
        *,
        node_type: NodeType | None = None,
    ) -> list[VectorSearchResult]:
        dim = self.embedding_dim
        sql = (
            f"SELECT id, array_cosine_distance(embedding, ?::DOUBLE[{dim}]) AS dist "
            f"FROM nodes WHERE embedding IS NOT NULL"
        )
        params: list[object] = [embedding]
        if node_type is not None:
            sql += " AND node_type = ?"
            params.append(node_type)
        sql += " ORDER BY dist ASC LIMIT ?"
        params.append(k)
        rows = self._con.execute(sql, params).fetchall()
        return [
            VectorSearchResult(node_id=str(row[0]), score=1.0 - float(row[1]))
            for row in rows
        ]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "vector_search" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add delfos/store/duckdb_store.py tests/store/test_duckdb_store.py
git commit -m "feat(store): brute-force cosine vector_search"
```

Expected: ruff clean, pyright `0 errors`.

---

### Task 7: Checkpoint manifest

Adds the `indexed_files` read/write methods backing crash recovery.

**Files:**
- Modify: `delfos/store/duckdb_store.py`
- Modify: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `self._con`, `IndexedFile`.
- Produces: `record_indexed_file(file_path, git_sha, indexed_at) -> None`, `indexed_file_sha(file_path) -> str | None`, `list_indexed_files() -> list[IndexedFile]`.

- [ ] **Step 1: Write failing manifest tests**

Append to `tests/store/test_duckdb_store.py`:

```python
from delfos.store.base import IndexedFile


def test_manifest_record_and_read_sha(store: DuckDBGraphStore) -> None:
    store.record_indexed_file("a.py", "sha1", NOW)
    assert store.indexed_file_sha("a.py") == "sha1"
    assert store.indexed_file_sha("missing.py") is None


def test_manifest_record_replaces_sha(store: DuckDBGraphStore) -> None:
    store.record_indexed_file("a.py", "sha1", NOW)
    store.record_indexed_file("a.py", "sha2", NOW)
    assert store.indexed_file_sha("a.py") == "sha2"


def test_manifest_list(store: DuckDBGraphStore) -> None:
    store.record_indexed_file("a.py", "sha1", NOW)
    store.record_indexed_file("b.py", "sha2", NOW)
    listed = store.list_indexed_files()
    assert {f.file_path for f in listed} == {"a.py", "b.py"}
    assert all(isinstance(f, IndexedFile) for f in listed)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "manifest" -v`
Expected: FAIL — `record_indexed_file` raises `NotImplementedError`.

- [ ] **Step 3: Implement manifest methods**

Implement on the class:

```python
    def record_indexed_file(
        self, file_path: str, git_sha: str, indexed_at: datetime
    ) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO indexed_files "
            "(file_path, git_sha, indexed_at) VALUES (?, ?, ?)",
            [file_path, git_sha, indexed_at],
        )

    def indexed_file_sha(self, file_path: str) -> str | None:
        row = self._con.execute(
            "SELECT git_sha FROM indexed_files WHERE file_path = ?", [file_path]
        ).fetchone()
        return None if row is None else str(row[0])

    def list_indexed_files(self) -> list[IndexedFile]:
        rows = self._con.execute(
            "SELECT file_path, git_sha, indexed_at FROM indexed_files"
        ).fetchall()
        return [
            IndexedFile(file_path=str(r[0]), git_sha=str(r[1]), indexed_at=r[2])
            for r in rows
        ]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "manifest" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add delfos/store/duckdb_store.py tests/store/test_duckdb_store.py
git commit -m "feat(store): indexed_files checkpoint manifest"
```

Expected: ruff clean, pyright `0 errors`.

---

### Task 8: Transactions + crash recovery

Adds the transaction primitives and proves commit/rollback and reopen-after-close persistence.

**Files:**
- Modify: `delfos/store/duckdb_store.py`
- Modify: `tests/store/test_duckdb_store.py`

**Interfaces:**
- Consumes: `self._con`, `self._in_txn`, the base-class `transaction()` context manager.
- Produces: `begin_transaction() -> None` (raises `RuntimeError` if already in a transaction), `commit() -> None`, `rollback() -> None`.

- [ ] **Step 1: Write failing transaction/recovery tests**

Append to `tests/store/test_duckdb_store.py`:

```python
def test_commit_persists(store: DuckDBGraphStore) -> None:
    with store.transaction():
        store.upsert_node(make_cue("cue-1"))
    assert store.get_node("cue-1") is not None


def test_rollback_discards(store: DuckDBGraphStore) -> None:
    try:
        with store.transaction():
            store.upsert_node(make_cue("cue-1"))
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store.get_node("cue-1") is None


def test_nested_transaction_raises(store: DuckDBGraphStore) -> None:
    store.begin_transaction()
    with pytest.raises(RuntimeError, match="transaction already open"):
        store.begin_transaction()
    store.rollback()


def test_committed_state_survives_reopen(tmp_path) -> None:
    path = tmp_path / "persist.duckdb"
    s1 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s1.initialize()
    with s1.transaction():
        s1.upsert_node(make_cue("cue-1"))
    s1.close()

    s2 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s2.initialize()
    assert s2.get_node("cue-1") is not None
    s2.close()


def test_uncommitted_state_lost_on_reopen(tmp_path) -> None:
    path = tmp_path / "crash.duckdb"
    s1 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s1.initialize()
    s1.begin_transaction()
    s1.upsert_node(make_cue("cue-1"))
    s1.close()  # die mid-file: never committed

    s2 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s2.initialize()
    assert s2.get_node("cue-1") is None
    s2.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store/test_duckdb_store.py -k "commit or rollback or nested or reopen or uncommitted" -v`
Expected: FAIL — `begin_transaction` raises `NotImplementedError`.

- [ ] **Step 3: Implement transaction primitives**

Implement on the class:

```python
    def begin_transaction(self) -> None:
        if self._in_txn:
            raise RuntimeError("transaction already open; nesting is not supported")
        self._con.execute("BEGIN TRANSACTION")
        self._in_txn = True

    def commit(self) -> None:
        self._con.execute("COMMIT")
        self._in_txn = False

    def rollback(self) -> None:
        self._con.execute("ROLLBACK")
        self._in_txn = False
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/store/test_duckdb_store.py -v`
Expected: PASS (all tests green).

- [ ] **Step 5: Final lint, type-check, commit**

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
git add delfos/store/duckdb_store.py tests/store/test_duckdb_store.py
git commit -m "feat(store): transactions + crash-recovery persistence"
```

Expected: ruff clean, pyright `0 errors`, full suite green. No `NotImplementedError` remains in `DuckDBGraphStore`.

---

## Notes for the implementer

- **DuckDB typing under pyright strict:** `duckdb.connect` returns `duckdb.DuckDBPyConnection`; `.execute(...).fetchone()` is typed as `tuple[Any, ...] | None` and `.fetchall()` as `list[Any]`. The code above annotates the connection explicitly. If pyright flags an `Any`-return in a specific spot, prefer an explicit local annotation (e.g. `row: tuple[object, ...] | None = ...`) over a blanket `# type: ignore`.
- **StrEnum binding:** all schema enums are `StrEnum` (a `str` subclass), so binding `node_type`, `status`, `cue_type`, etc. as parameters sends their string value, and reading them back as strings re-validates cleanly through `TypeAdapter(Node)`.
- **Array columns come back as tuples:** DuckDB returns a `DOUBLE[dim]` column value as a Python `tuple`, not a `list`. `_row_to_node` passes it to `TypeAdapter(Node).validate_python`, which coerces the tuple to `list[float]` for the `embedding` field — so `get_node(...) == node` holds. (Verified: `(0.1, 0.2, 0.3)` → `[0.1, 0.2, 0.3]`, equal to the original.) Don't compare a raw fetched row's array against a `list` directly; go through the model.
- **`close()` rolls back an open transaction:** verified on DuckDB 1.5.4 — closing a connection with an uncommitted `BEGIN` does not raise and discards the open transaction, which is what `test_uncommitted_state_lost_on_reopen` relies on. No explicit `rollback()` before `close()` is needed to make that test pass.
- **`DOUBLE[dim]`, not `FLOAT[dim]`:** the store receives Python `float` (64-bit). Storing as `DOUBLE` round-trips embeddings exactly, so `get_node(...) == node` holds for arbitrary embedding values; a `FLOAT` (32-bit) column would lose precision and break round-trip equality. `array_cosine_distance` accepts `DOUBLE` arrays. Revisit to `FLOAT` only if a future HNSW index requires 32-bit vectors.
- **`INSERT OR REPLACE`** relies on the PRIMARY KEY of each table (declared in Task 1); don't drop those constraints.
- Run the whole file (`uv run pytest tests/store/test_duckdb_store.py -v`) at the end of each task, not just the filtered subset, once you're past Task 4 — cheap regression insurance.

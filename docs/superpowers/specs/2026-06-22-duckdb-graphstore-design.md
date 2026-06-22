# DuckDBGraphStore — Implementation Design

**Date:** 2026-06-22
**Status:** Approved for implementation
**Scope:** Concrete `DuckDBGraphStore` backend for the `GraphStore` contract, built test-first.

## Purpose

`DuckDBGraphStore` (`delfos/store/duckdb_store.py`) is the only concrete `GraphStore`
backend. Every other component (indexer, MCP tools, future CLI) reaches the database
exclusively through this class. Today all 13 of its methods raise `NotImplementedError`;
this design fills them in, under TDD, against a single-file DuckDB database.

## Settled decisions

These four forks were chosen during brainstorming, each over explicit alternatives:

1. **Vector search: brute-force now, HNSW-ready.** `vector_search` ranks with DuckDB's
   `array_cosine_distance` in a plain `ORDER BY ... LIMIT k`. Exact and deterministic
   (easy to assert in tests), no extension to load, no experimental HNSW persistence flag.
   The method is the documented single swap seam: HNSW slots in behind the same signature
   when scale demands it. (YAGNI — see `decisions.md` §3/§4 scale caveat.)
2. **Tests: pytest + temp-file DB per test.** A `pytest` dev-dependency is added. Each
   test gets a fresh on-disk DuckDB file via `tmp_path`, so the real persistence contract
   (close → reopen → assert committed state survived) can be exercised. This is required to
   test crash recovery and `indexed_files` manifest survival.
3. **Scope: all 13 methods, single connection.** Implement the full interface against one
   read-write connection. The single-writer / multi-reader read-only-attach concurrency
   story is deferred to the MCP-server layer that will own how sessions connect. Also
   deferred: the HNSW index, and tombstone/`status` filtering.
4. **Storage: one `nodes` table, typed columns.** A single `nodes` table holds the common
   `BaseNode` fields plus every type-specific field as its own nullable, typed column
   (`embedding` as a real `FLOAT[dim]` column). The `node_type` discriminator drives
   row → correct Pydantic model on read. Single id space, single-table reads, and
   `vector_search` is a plain column scan.

Two smaller decisions, made without a separate question:

- **Distance metric: cosine** (`array_cosine_distance`), the standard for text embeddings.
- **`vector_search` returns `node=None` in v1.** Callers resolve `node_id` via `get_node`,
  matching the contract's hydration note. No hydrate flag exists on the signature.

## Storage schema (3 tables, one id space)

```sql
nodes(
  id TEXT PRIMARY KEY,
  node_type TEXT,                    -- discriminator: cue | tag | content
  source_file TEXT, git_sha TEXT, indexed_at TIMESTAMP,
  status TEXT, deleted_at TIMESTAMP, deleted_by_commit TEXT,
  embedding FLOAT[<dim>],            -- NULL for tags; fixed-size enables cosine
  embedding_model TEXT, embedding_model_version TEXT,
  cue_type TEXT, text TEXT,                                 -- cue-only
  category TEXT, value TEXT,                                -- tag-only
  kind TEXT, memory_layer TEXT, symbol_name TEXT,
  signature TEXT, docstring TEXT, body TEXT                 -- content-only
)

edges(
  source_id TEXT, target_id TEXT, edge_type TEXT,
  source_file TEXT, git_sha TEXT, indexed_at TIMESTAMP,
  PRIMARY KEY (source_id, target_id, edge_type)            -- the upsert key
)

indexed_files(
  file_path TEXT PRIMARY KEY, git_sha TEXT, indexed_at TIMESTAMP
)
```

`embedding` is a **fixed-size `FLOAT[dim]`**, with `dim` baked from the constructor's
`embedding_dim`, so `array_cosine_distance` applies. Type-specific columns are nullable and
populated only for their node type.

## Lifecycle & write validation

- **`__init__`** opens the DuckDB connection on `path` (so a temp-file store can be reopened
  later), stores `embedding_dim` / `embedding_model`, and tracks an `_in_txn` flag.
- **`initialize()`** runs `CREATE TABLE IF NOT EXISTS` for all three tables — idempotent,
  safe on an already-initialized store. **`close()`** closes the connection.
- **`upsert_node`** validates before writing, raising `ValueError` when: an embedding is
  present but `embedding_model != self.embedding_model`, or `len(embedding) != self.embedding_dim`.
  Then `INSERT OR REPLACE INTO nodes`, mapping type-specific columns from the Pydantic model
  and leaving the rest NULL.
- **`upsert_edge`** runs `INSERT OR REPLACE INTO edges`, keyed on `(source_id, target_id, edge_type)`.

## Read paths

- **`get_node`** — `SELECT *` by id, rebuild a dict, hand to `TypeAdapter(Node).validate_python`
  so the `node_type` discriminator selects the right model. Missing id → `None`.
- **`neighbors`** — `OUTGOING`: edges where `source_id = id` → return target nodes;
  `INCOMING`: edges where `target_id = id` → return source nodes. Optional `edge_type`
  filter. Joins to `nodes` and hydrates each row to a `Node`.
- **`vector_search`** —
  `SELECT id, array_cosine_distance(embedding, $q) AS dist FROM nodes
   WHERE embedding IS NOT NULL [AND node_type = ?] ORDER BY dist LIMIT k`.
  `score = 1 - dist` (higher = closer, per the contract). NULL-embedding nodes (tags) are
  skipped. `VectorSearchResult.node` stays `None` in v1. **This method is the documented
  swap seam for HNSW.**

## Deletes, manifest, transactions

- **`delete_node`** — delete incident edges (`source_id = id OR target_id = id`), then the node.
- **`delete_nodes_for_file`** — collect ids where `source_file = ?`; delete edges where
  `source_file = ? OR source_id IN ids OR target_id IN ids` (catches both file-scoped edges
  and cross-file edges such as `REDIRECTS_TO` that touch a dropped node); then delete the
  nodes where `source_file = ?`. Backs the delete-and-reindex strategy.
- **manifest** — `record_indexed_file` = `INSERT OR REPLACE`; `indexed_file_sha` = scalar
  select returning `git_sha | None`; `list_indexed_files` → `list[IndexedFile]`.
- **transactions** — `begin_transaction` issues `BEGIN TRANSACTION` and raises if `_in_txn`
  is already set (no nesting); `commit` / `rollback` issue the SQL and clear the flag. The
  base-class `transaction()` context manager drives these.

## Out of scope (deferred)

- Multi-reader read-only attach / WAL concurrency orchestration (MCP-server layer's job).
- HNSW vector index (the `vector_search` body is the swap point).
- Tombstone / `status` filtering — v1 reads return nodes regardless of `status`, since
  nothing creates tombstones yet (delete-and-reindex hard-deletes).

## TDD test plan

`tests/store/test_duckdb_store.py`, pytest with a temp-file DB fixture:

```python
@pytest.fixture
def store(tmp_path):
    s = DuckDBGraphStore(tmp_path / "t.duckdb", embedding_dim=8, embedding_model="fake-v1")
    s.initialize()
    yield s
    s.close()
```

Written red → green in this order:

1. `initialize` idempotent; tables exist; constructor stores config.
2. `upsert_node` + `get_node` round-trip for **each** type (cue, tag, content); missing id → `None`.
3. embedding **model** mismatch → `ValueError`; embedding **dim** mismatch → `ValueError`.
4. `upsert_node` replace semantics (same id overwrites).
5. `upsert_edge` + `neighbors` (outgoing, incoming, `edge_type` filter); edge replace semantics.
6. `delete_node` removes node + incident edges.
7. `delete_nodes_for_file` removes matching nodes + file-scoped and incident edges.
8. `vector_search` ordering, `k` limit, `node_type` filter, skips NULL-embedding nodes.
9. manifest: `record_indexed_file` → `indexed_file_sha` → `list_indexed_files`.
10. transaction: commit persists; rollback discards.
11. **crash recovery**: write inside a transaction, roll back (uncommitted) → reopen file →
    nothing; then commit → survives reopen.

## Definition of done

- All 13 methods implemented; no `NotImplementedError` remains.
- `tests/store/test_duckdb_store.py` covers the plan above and passes.
- `pytest` added to the dev dependency group.
- `uv run ruff check .`, `uv run ruff format .`, and `uv run pyright` (strict) all pass.

# DuckDBGraphStore Implementation Plan — Issues to Flag

**Source:** review of `docs/superpowers/plans/2026-06-22-duckdb-graphstore.md`
**Date:** 2026-06-22

## 1. Risk: `close()` with an open transaction (Task 8, `test_uncommitted_state_lost_on_reopen`)

The test calls `s1.close()` after `s1.begin_transaction()` + `upsert_node` but before any commit/rollback, and asserts the row is gone on reopen. This **assumes DuckDB rolls back open transactions on connection close**. That's standard DB behavior, but DuckDB's Python binding may instead raise or leave the transaction in an undefined state. The implementer should verify this empirically before relying on the test as written — if DuckDB raises on close-with-open-txn, the test needs an explicit `s1.rollback()` before `close()` (which would weaken what the test proves). Recommend adding a note to Task 8 to confirm the behavior first.

## 2. Ruff `I` (isort) will flag duplicate `from delfos.schema import` lines

- Task 2 adds a multi-line `from delfos.schema import (ContentKind, ContentNode, ...)`.
- Task 4 adds `from delfos.schema import Direction, Edge, EdgeType`.
- Task 6 says "Add `NodeType` to the test imports."

Each of these would create a second/third import from the same module, which ruff `I` will reformat into a single merged block. The plan's "run `ruff format`" step will fix it automatically, but the implementer should be aware the pasted code won't be lint-clean until that step runs. The plan could say "merge into the existing `delfos.schema` import block" to avoid confusion.

## 3. Missing `tests/__init__.py` / `tests/store/__init__.py`

The plan never creates these. Pytest will discover tests fine without them (rootdir-based), and pyright follows PEP 420 namespace packages, so this likely works. But pyright strict mode with `include = ["."]` sometimes complains about implicit namespace packages in edge cases. Worth a one-line note saying "create empty `tests/__init__.py` and `tests/store/__init__.py` if pyright flags the test dir" — or just create them proactively in Task 1.

## 4. `upsert_node` validation uses `getattr` instead of `isinstance`

```python
embedding = getattr(node, "embedding", None)
if getattr(node, "embedding_model", None) != self.embedding_model:
```

`TagNode` doesn't have `embedding`/`embedding_model` fields (it doesn't inherit `EmbeddedMixin`), and `extra="forbid"` means `getattr` on a missing field returns the default `None` — so this works. But it's fragile: it relies on `getattr` returning `None` for absent fields rather than raising. An `isinstance(node, EmbeddedMixin)` guard would be more explicit and self-documenting. Minor style point, not a bug.

## 5. No test for `upsert_edge` with `source_file=None`

`Edge.source_file` is `str | None = None`, and `delete_nodes_for_file`'s `source_file = ?` clause would miss such edges (relying on the `source_id IN (...) / target_id IN (...)` fallback). The plan never tests an edge with null provenance. The fallback handles it correctly, but a test would lock that down. Minor gap.

## 6. `delete_nodes_for_file` is not wrapped in a transaction by the store

The method issues two separate SQL statements (edges, then nodes) with no internal `BEGIN`/`COMMIT`. If called outside a transaction and the process dies between the two statements, you get orphaned edges referencing deleted nodes. The spec says the indexer wraps file re-indexing in `transaction()`, so this is fine by contract — but the plan doesn't state this dependency. A one-line note on `delete_nodes_for_file` saying "must be called inside `transaction()`" would make the contract explicit.

## 7. Task 3 is tests-only with no red phase

Task 3's Step 2 says "Run to verify pass" (not "verify failure"). This is intentional — validation was implemented in Task 2 — but it breaks the otherwise-consistent red→green rhythm. The task header "Embedding-invariant validation" could mislead a rote implementer into thinking they need to add code. The body clarifies, but consider renaming to "Embedding-invariant validation **tests**" for clarity.

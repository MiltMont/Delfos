# SCIP Symbol as ContentNode ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `scip_symbol` foreign-key field on `ContentNode` with using the SCIP symbol string directly as the node's `id`, enabling O(1) reverse lookup via the existing `id_index_` without any new data structures.

**Architecture:** When SCIP is available, the extractor computes `content_id = scip_symbol` instead of `content:{source_file}::{qualified_name}`; module-level nodes and definitions without SCIP coverage keep the fallback scheme. `ScipService` stops dereferencing `node.scip_symbol` and passes `content_id` directly to the index — the indirection is gone. The `scip_symbol` field is then dead and removed from schema, store, C++ struct, FlatBuffers schema, and bindings.

**Tech Stack:** Python 3.12, Pydantic v2, nanobind C++ extension (`libdelfos`), FlatBuffers 24.3.25, pytest.

## Global Constraints

- Pyright strict mode must pass on all changed Python files: `uv run pyright`
- `extra="forbid"` on all Pydantic models — no new optional fields
- Run tests with `uv run pytest`; run C++ tests with `ctest --test-dir build/debug --output-on-failure`
- Rebuild the Python extension after any C++ change: `uv pip install -e .`
- FlatBuffers regeneration command (requires flatc 24.3.25): `flatc --cpp -o libdelfos/flatbuffers/ libdelfos/flatbuffers/delfos.fbs`
  - Install flatc if missing: `brew install flatbuffers` (macOS)

---

## File Map

| File | Change |
|------|--------|
| `delfos/indexer/extractor.py` | `_definition_content_id` prefers SCIP symbol; `_add_content` drops `scip_symbol` param |
| `tests/indexer/test_pipeline_scip.py` | Assert node ID = SCIP symbol instead of checking `scip_symbol` FK |
| `delfos/scip/service.py` | Use `content_id` directly; drop `node.scip_symbol` dereference |
| `tests/scip/test_service.py` | Node ID = SCIP symbol; drop `scip_symbol` constructor arg |
| `tests/mcp/test_server.py` | Node ID = `SCIP_SYM`; call SCIP tools with `SCIP_SYM` |
| `delfos/schema/nodes.py` | Remove `scip_symbol: str \| None = None` from `ContentNode` |
| `delfos/store/native_store.py` | Remove two lines that read/write `scip_symbol` |
| `delfos/_delfos.pyi` | Remove `scip_symbol: str` from `NodeData` stub |
| `libdelfos/include/delfos/node.hpp` | Remove `std::string scip_symbol` field |
| `libdelfos/bindings/py_delfos.cpp` | Remove `.def_rw("scip_symbol", ...)` binding |
| `libdelfos/flatbuffers/delfos.fbs` | Remove `scip_symbol: string` from `Node` table |
| `libdelfos/flatbuffers/delfos_generated.h` | Regenerate with `flatc` after editing `.fbs` |
| `tests/store/test_native_store.py` | Remove two now-obsolete `scip_symbol` roundtrip tests |

---

## Task 1: Use SCIP symbol as ContentNode ID in the extractor

**Files:**
- Modify: `delfos/indexer/extractor.py`
- Modify: `tests/indexer/test_pipeline_scip.py`

**Interfaces:**
- Produces: `extract(module, ...)` returns `ContentNode` nodes whose `.id` equals the SCIP symbol string when SCIP coverage exists, and `content:{source_file}::{qualified_name}` otherwise. The `scip_symbol` field on `ContentNode` is no longer set (stays `None`).

- [ ] **Step 1: Write the failing test**

Rename `test_index_populates_scip_symbol_by_lineno` to `test_index_uses_scip_symbol_as_content_id` and update its assertions in `tests/indexer/test_pipeline_scip.py`:

```python
def test_index_uses_scip_symbol_as_content_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def foo():\n    return 1\n")

    idx = write_index(
        tmp_path / "index.scip",
        documents=[document("mod.py", occurrences=[occurrence(SYM, 0, definition=True)])],
    )
    scip = ScipIndex(idx)

    def _fake_load(_self: Indexer, _root: Path, _ws: Workspace) -> tuple[ScipIndex, ScipStatus]:
        return scip, ScipStatus.PRESENT

    monkeypatch.setattr(Indexer, "_load_scip_index", _fake_load)

    store = NativeGraphStore(
        tmp_path / "snap", embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    store.initialize()
    indexer = Indexer(store, _HashEmbedder())
    stats = indexer.index(repo, workspace=Workspace(tmp_path / "ws"))
    assert stats.indexed_files == 1

    # The SCIP symbol IS the content node id.
    foo = store.get_node(SYM)
    assert isinstance(foo, ContentNode)
    assert foo.id == SYM

    # The module node has no SCIP definition → fallback id unchanged.
    module = store.get_node("content:mod.py::<module>")
    assert isinstance(module, ContentNode)
    assert module.id == "content:mod.py::<module>"

    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/test_pipeline_scip.py::test_index_uses_scip_symbol_as_content_id -v
```

Expected: `FAILED` — `store.get_node(SYM)` returns `None` (node lives at old fallback id).

- [ ] **Step 3: Implement the change in `delfos/indexer/extractor.py`**

Replace `_definition_content_id` (line 79) with:

```python
def _definition_content_id(source_file: str, qualified_name: str, scip_symbol: str | None) -> str:
    if scip_symbol:
        return scip_symbol
    return f"content:{source_file}::{qualified_name}"
```

Replace `_Builder._add_definition` (lines 131–154) with:

```python
def _add_definition(self, definition: ParsedDefinition, module_id: str) -> None:
    source_file = self._module.source_file
    scip_sym = self._scip_symbols.get(definition.lineno)
    content_id = _definition_content_id(source_file, definition.qualified_name, scip_sym)
    self._add_content(
        node_id=content_id,
        kind=ContentKind.TEST if definition.is_test else _CONTENT_KINDS[definition.kind],
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name=definition.qualified_name,
        signature=definition.signature,
        docstring=definition.docstring,
        body=definition.source,
    )
    self._tag_content(content_id, definition.kind.value)
    self._add_edge(content_id, module_id, EdgeType.PART_OF_TOPIC)

    symbol_cue_id = _symbol_cue_id(source_file, definition.qualified_name)
    self._add_cue(symbol_cue_id, CueType.SYMBOL, definition.name)
    self._add_edge(symbol_cue_id, content_id, EdgeType.CUE_OF)

    for message in definition.error_messages:
        error_cue_id = _error_cue_id(source_file, message)
        self._add_cue(error_cue_id, CueType.ERROR_MESSAGE, message)
        self._add_edge(error_cue_id, content_id, EdgeType.CUE_OF)
```

Replace `_add_content` (lines 156–180) — remove the `scip_symbol` parameter and the field in the `ContentNode` constructor:

```python
def _add_content(
    self,
    *,
    node_id: str,
    kind: ContentKind,
    memory_layer: MemoryLayer,
    symbol_name: str | None,
    signature: str | None,
    docstring: str | None,
    body: str,
) -> None:
    self._nodes[node_id] = ContentNode(
        id=node_id,
        source_file=self._module.source_file,
        git_sha=self._git_sha,
        indexed_at=self._indexed_at,
        kind=kind,
        memory_layer=memory_layer,
        symbol_name=symbol_name,
        signature=signature,
        docstring=docstring,
        body=body,
    )
```

Also update the docstring of `extract()` (lines 227–236) — replace the `scip_symbols` description:

```python
    """Extract all nodes and edges for ``module``.

    ``git_sha`` is the per-file content SHA stamped on every sourced node and
    edge; ``indexed_at`` is the shared timestamp for this indexing pass. Cue
    nodes are returned without embeddings — the pipeline attaches those inside
    the file's transaction.

    ``scip_symbols`` optionally maps a definition's 1-based ``lineno`` to its
    SCIP symbol string. When a match exists, that symbol becomes the
    ``ContentNode.id`` directly, enabling O(1) reverse lookup from any SCIP
    occurrence. Definitions without a match use the fallback id scheme
    ``content:{source_file}::{qualified_name}``.
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/test_pipeline_scip.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run pyright**

```bash
uv run pyright delfos/indexer/extractor.py
```

Expected: `0 errors, 0 warnings`.

- [ ] **Step 6: Commit**

```bash
git add delfos/indexer/extractor.py tests/indexer/test_pipeline_scip.py
git commit -m "feat(indexer): use SCIP symbol as ContentNode id when available"
```

---

## Task 2: Simplify ScipService to use content_id directly

**Files:**
- Modify: `delfos/scip/service.py`
- Modify: `tests/scip/test_service.py`
- Modify: `tests/mcp/test_server.py`

**Interfaces:**
- Consumes: `ContentNode.id` is the SCIP symbol when indexed with SCIP (from Task 1). The `scip_symbol` field still exists on the model but is always `None` after Task 1.
- Produces: `ScipService.references(content_id)` passes `content_id` directly to `ScipIndex.references()`. Nodes with fallback IDs return `[]` naturally (the symbol isn't in the index).

- [ ] **Step 1: Write the failing tests**

Replace the entire body of `tests/scip/test_service.py` with:

```python
"""Tests for the SCIP read-path bridge (delfos.scip.service.ScipService)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from delfos.schema import ContentKind, ContentNode, MemoryLayer
from delfos.scip.reader import ScipIndex
from delfos.scip.service import ScipService
from delfos.store.native_store import NativeGraphStore

from .builders import (
    document,
    occurrence,
    relationship,
    symbol_information,
    write_index,
)

EMBEDDING_DIM = 8
EMBEDDING_MODEL = "fake-v1"
NOW = datetime(2026, 6, 22, 12, 0, 0)

SYM = "scip-python python . a/foo()."


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(
        tmp_path / "snap", embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    s.initialize()
    yield s
    s.close()


def _content(node_id: str) -> ContentNode:
    return ContentNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="foo",
        body="def foo(): ...",
    )


def _index(path: Path) -> ScipIndex:
    write_index(
        path,
        documents=[
            document(
                "a.py",
                occurrences=[
                    occurrence(SYM, 0, definition=True),
                    occurrence(SYM, 20),
                ],
                symbols=[
                    symbol_information(
                        SYM,
                        [
                            relationship("iface#", is_implementation=True),
                            relationship("Type#", is_type_definition=True),
                        ],
                    )
                ],
            ),
            document("b.py", occurrences=[occurrence(SYM, 5)]),
        ],
    )
    return ScipIndex(path)


def test_references_resolves_symbol_and_excludes_definition(
    store: NativeGraphStore, tmp_path: Path
) -> None:
    store.upsert_node(_content(SYM))
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    refs = svc.references(SYM)
    assert {(path, occ.start_line) for path, occ in refs} == {("a.py", 20), ("b.py", 5)}


def test_implementations_and_type_definition(store: NativeGraphStore, tmp_path: Path) -> None:
    store.upsert_node(_content(SYM))
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    assert [r.symbol for r in svc.implementations(SYM)] == ["iface#"]
    assert [r.symbol for r in svc.type_definition(SYM)] == ["Type#"]


def test_node_with_fallback_id_returns_empty(store: NativeGraphStore, tmp_path: Path) -> None:
    # A node indexed without SCIP has a fallback id, not in the SCIP index → [].
    store.upsert_node(_content("content:a.py::foo"))
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    assert svc.references("content:a.py::foo") == []
    assert svc.implementations("content:a.py::foo") == []
    assert svc.type_definition("content:a.py::foo") == []


def test_unknown_content_id_raises(store: NativeGraphStore, tmp_path: Path) -> None:
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    with pytest.raises(ValueError, match="no content node"):
        svc.references("content:missing")
```

In `tests/mcp/test_server.py`, update `test_scip_tools_resolve_references_and_relationships` (around line 197):

```python
def test_scip_tools_resolve_references_and_relationships(
    store: NativeGraphStore, tmp_path: Path
) -> None:
    # Node id IS the SCIP symbol — no separate scip_symbol field.
    content = make_content(SCIP_SYM, "login")
    load(store, [content], [])
    scip = _scip_service(store, tmp_path)

    refs = _references(scip, SCIP_SYM)
    assert [(r.relative_path, r.start_line) for r in refs] == [("a.py", 9)]
    assert [r.symbol for r in _implementations(scip, SCIP_SYM)] == ["iface#"]
    assert [r.symbol for r in _type_definition(scip, SCIP_SYM)] == ["Type#"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scip/test_service.py tests/mcp/test_server.py::test_scip_tools_resolve_references_and_relationships -v
```

Expected: service tests fail — `ScipService` still looks up `node.scip_symbol` (which is `None` after Task 1) and returns `[]` instead of results.

- [ ] **Step 3: Implement the change in `delfos/scip/service.py`**

Replace the entire file with:

```python
"""Read-path bridge from a graph ContentNode to SCIP cross-references.

ContentNode IDs assigned during indexing are the SCIP symbol string when SCIP
coverage was present, so ScipService passes content_id directly to the index —
no secondary FK dereference needed. Nodes indexed without SCIP use a fallback
id scheme and return empty results naturally (the id is not in the SCIP index).

v1 scope: results are SCIP-native (relative path + line range + symbol). We do
not resolve a referencing occurrence back to its enclosing ContentNode.
"""

from __future__ import annotations

from collections.abc import Callable

from delfos.schema import ContentNode
from delfos.scip.reader import Occurrence, Relationship, ScipIndex
from delfos.store import GraphStore


class ScipService:
    """Resolve a content node's SCIP cross-references using its id as the symbol."""

    def __init__(self, store: GraphStore, index: ScipIndex) -> None:
        self._store = store
        self._index = index

    def references(self, content_id: str) -> list[tuple[str, Occurrence]]:
        """All non-definition usages of the node's symbol across the repo.

        Returns ``(relative_path, occurrence)`` pairs. Empty when ``content_id``
        is not a SCIP symbol (node was indexed without SCIP coverage).
        """
        self._content_node(content_id)  # validate node exists and is ContentNode
        return self._index.references(content_id)

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
        # source_file scopes the symbol_info lookup to document-local symbols first.
        info = self._index.symbol_info(content_id, relative_path=node.source_file)
        if info is None:
            return []
        return [rel for rel in info.relationships if predicate(rel)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scip/test_service.py tests/mcp/test_server.py -v
```

Expected: all tests in both files pass.

- [ ] **Step 5: Run pyright**

```bash
uv run pyright delfos/scip/service.py
```

Expected: `0 errors, 0 warnings`.

- [ ] **Step 6: Commit**

```bash
git add delfos/scip/service.py tests/scip/test_service.py tests/mcp/test_server.py
git commit -m "refactor(scip): use content_id directly as SCIP symbol in ScipService"
```

---

## Task 3: Remove the `scip_symbol` field from schema, store, and C++

**Files:**
- Modify: `delfos/schema/nodes.py`
- Modify: `delfos/store/native_store.py`
- Modify: `delfos/_delfos.pyi`
- Modify: `libdelfos/include/delfos/node.hpp`
- Modify: `libdelfos/bindings/py_delfos.cpp`
- Modify: `libdelfos/flatbuffers/delfos.fbs`
- Regenerate: `libdelfos/flatbuffers/delfos_generated.h`
- Modify: `tests/store/test_native_store.py`

**Interfaces:**
- Consumes: After Tasks 1 and 2, `scip_symbol` is always `None` on every node in the store and never read by any live code path.

- [ ] **Step 1: Remove `scip_symbol` from the Python schema**

In `delfos/schema/nodes.py`, delete line 124:

```python
    scip_symbol: str | None = None
```

The `ContentNode` class (lines 114–128) becomes:

```python
class ContentNode(SourcedNode, EmbeddedMixin):
    """The actual implementation artifact returned to agents.

    ``reconstruct`` returns a flat, score-ordered list of these.
    """

    node_type: Literal[NodeType.CONTENT] = NodeType.CONTENT
    kind: ContentKind
    memory_layer: MemoryLayer
    symbol_name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    body: str
```

- [ ] **Step 2: Remove `scip_symbol` from `native_store.py`**

In `delfos/store/native_store.py`, delete line 150:

```python
        nd.scip_symbol = node.scip_symbol or ""
```

And delete line 208:

```python
        scip_symbol=nd.scip_symbol or None,
```

- [ ] **Step 3: Remove `scip_symbol` from the pyi stub**

In `delfos/_delfos.pyi`, delete line 63:

```python
    scip_symbol: str
```

- [ ] **Step 4: Run Python tests to verify the schema change is clean**

```bash
uv run pytest tests/store/test_native_store.py tests/scip/ tests/indexer/ -v
```

Expected: all pass except the two obsolete roundtrip tests (`test_roundtrip_content_scip_symbol` and `test_roundtrip_content_scip_symbol_default_none`) which will error because `ContentNode` no longer accepts `scip_symbol`.

- [ ] **Step 5: Delete the two obsolete store roundtrip tests**

In `tests/store/test_native_store.py`, delete the functions `test_roundtrip_content_scip_symbol` (lines 143–151) and `test_roundtrip_content_scip_symbol_default_none` (lines 154–159).

- [ ] **Step 6: Run Python tests again to confirm clean**

```bash
uv run pytest tests/store/test_native_store.py tests/scip/ tests/indexer/ -v
```

Expected: all pass.

- [ ] **Step 7: Remove `scip_symbol` from `node.hpp`**

In `libdelfos/include/delfos/node.hpp`, delete line 44:

```cpp
    std::string scip_symbol;   // SCIP symbol FK (empty when no SCIP symbol)
```

- [ ] **Step 8: Remove `scip_symbol` from `py_delfos.cpp`**

In `libdelfos/bindings/py_delfos.cpp`, delete line 365:

```cpp
        .def_rw("scip_symbol", &NodeData::scip_symbol)
```

- [ ] **Step 9: Remove `scip_symbol` from `delfos.fbs`**

In `libdelfos/flatbuffers/delfos.fbs`, delete line 37:

```
    scip_symbol:           string;
```

The `Node` table (lines 15–38) becomes:

```
table Node {
    id:                    string;
    type:                  NodeType;
    status:                NodeStatus;
    indexed_at:            long;
    deleted_at:            long;
    deleted_by_commit:     string;
    source_file:           string;
    git_sha:               string;
    cue_type:              CueType;
    text:                  string;
    category:              TagCategory;
    value:                 string;
    kind:                  ContentKind;
    memory_layer:          MemoryLayer;
    symbol_name:           string;
    signature:             string;
    docstring:             string;
    body:                  string;
    embedding:             [double];
    embedding_model:       string;
    embedding_model_version: string;
}
```

- [ ] **Step 10: Regenerate `delfos_generated.h`**

```bash
flatc --cpp -o libdelfos/flatbuffers/ libdelfos/flatbuffers/delfos.fbs
```

If `flatc` is not on PATH: `brew install flatbuffers` then retry. Verify the generated file no longer contains `scip_symbol`:

```bash
grep -c "scip_symbol" libdelfos/flatbuffers/delfos_generated.h
```

Expected: `0`.

- [ ] **Step 11: Rebuild the Python extension**

```bash
uv pip install -e .
```

Expected: builds successfully, extension module reloaded.

- [ ] **Step 12: Run C++ tests**

```bash
cmake --preset debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure
```

Expected: all C++ tests pass.

- [ ] **Step 13: Run full Python test suite**

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 14: Run pyright on changed files**

```bash
uv run pyright delfos/schema/nodes.py delfos/store/native_store.py
```

Expected: `0 errors, 0 warnings`.

- [ ] **Step 15: Commit**

```bash
git add delfos/schema/nodes.py delfos/store/native_store.py delfos/_delfos.pyi \
        libdelfos/include/delfos/node.hpp libdelfos/bindings/py_delfos.cpp \
        libdelfos/flatbuffers/delfos.fbs libdelfos/flatbuffers/delfos_generated.h \
        tests/store/test_native_store.py
git commit -m "chore: remove scip_symbol FK field — content id is now the SCIP symbol"
```

---

## Self-Review

**Spec coverage:**
- ✅ Extractor uses SCIP symbol as node ID (Task 1)
- ✅ Fallback to `content:{file}::{name}` when no SCIP coverage (Task 1)
- ✅ Module-level ContentNode uses fallback scheme — no SCIP symbol for file-level (Task 1 — module node call site unchanged)
- ✅ ScipService uses `content_id` directly, no FK dereference (Task 2)
- ✅ Nodes with fallback IDs return `[]` from SCIP tools naturally (Task 2 — `test_node_with_fallback_id_returns_empty`)
- ✅ `scip_symbol` field removed from Python schema, store translation, pyi stub, C++ struct, bindings, FlatBuffers (Task 3)
- ✅ Delete-and-reindex stale handling unaffected — `delete_nodes_for_file` uses `source_file`, not ID

**Placeholder scan:** None found.

**Type consistency:** `content_id: str` passed to `ScipService` methods in Tasks 1–3; `_content_node(content_id)` validates and returns `ContentNode`; `symbol_info(content_id, relative_path=node.source_file)` consistent with `ScipIndex.symbol_info` signature throughout.
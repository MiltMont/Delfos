# Agent-Driven Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the calling agent write `CONCEPT` cues and `ARCH_LAYER`/`PATTERN_TYPE` tags into the graph via a new MCP `annotate` tool and `enrich` prompt, per the approved spec `docs/superpowers/specs/2026-07-05-agent-driven-enrichment-design.md`.

**Architecture:** A new write-path `EnrichmentService` (`delfos/enrich/`) sits on top of `GraphStore` + `Embedder`, mirroring `ReconstructionService`. The MCP server registers one `annotate` tool and one `enrich` prompt. One store addition (`list_tag_values`) powers the vocabulary echo, backed by a new C++ node scan (`Graph::nodes_by_type` → `Store::list_nodes_by_type` binding). Annotations carry the target content node's `source_file`/`git_sha`, so `delete_nodes_for_file` wipes them on re-index with no new deletion logic.

**Tech Stack:** Python 3 (Pydantic v2, FastMCP, pytest), C++20 (Catch2, nanobind), uv, cmake presets.

## Global Constraints

- Pyright **strict** mode: all new code fully typed (`uv run pyright` must pass).
- `extra="forbid"` on all new Pydantic models.
- Everything goes through `GraphStore` — no component touches `delfos._delfos` except `NativeGraphStore`.
- Every embedding must carry `embedding_model` (enforced by `EmbeddedMixin`).
- C++ compiled with `-Wall -Wextra -Werror` — no unused variables.
- After any C++ change, rebuild the extension: `uv pip install -e .` (Python tests load the installed extension, not the cmake build).
- Lint/format: `uv run ruff check .` and `uv run ruff format .` must be clean before each commit.
- Commit style: conventional commits, lowercase (`feat:`, `test:`, `docs:`), matching `git log`.
- Work on a feature branch (e.g. `feat/agent-driven-enrichment`), not `main`.

---

### Task 1: C++ node scan — `Graph::nodes_by_type` + `Store::list_nodes_by_type` binding

**Files:**
- Modify: `libdelfos/include/delfos/graph.hpp` (public section, after `nodes_for_file`, ~line 96)
- Modify: `libdelfos/bindings/py_delfos.cpp` (Store struct ~line 255; module `.def` ~line 402)
- Test: `libdelfos/tests/test_graph.cpp` (append)

**Interfaces:**
- Consumes: existing `Graph` internals (`id_index_`, `nodes_`), `NodeData`, `NodeType` (`libdelfos/include/delfos/types.hpp`).
- Produces: `std::vector<NodeData> Graph::nodes_by_type(NodeType) const` and the Python-visible `Store.list_nodes_by_type(node_type: int) -> list[NodeData]`, used by Task 2.

- [ ] **Step 1: Write the failing Catch2 tests**

Append to `libdelfos/tests/test_graph.cpp` (helpers `make_cue`, `make_tag`, `make_content` already exist at the top of the file):

```cpp
// ─────────────────────────────────────────────────────────────────────────────
// nodes_by_type
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("nodes_by_type returns live nodes of the requested type") {
    Graph g;
    g.upsert_node(make_cue("cue:1"));
    g.upsert_node(make_tag("tag:language:cpp"));
    g.upsert_node(make_tag("tag:arch_layer:storage"));
    g.upsert_node(make_content("content:1"));

    auto tags = g.nodes_by_type(NodeType::Tag);

    REQUIRE(tags.size() == 2);
    for (const auto& n : tags) {
        REQUIRE(n.type == NodeType::Tag);
    }
}

TEST_CASE("nodes_by_type excludes soft-deleted and hard-deleted nodes") {
    Graph g;
    g.upsert_node(make_tag("tag:a"));
    NodeData soft = make_tag("tag:b");
    soft.status = NodeStatus::Deleted;
    g.upsert_node(soft);
    g.upsert_node(make_tag("tag:c"));
    g.delete_node(g.find("tag:c"));

    auto tags = g.nodes_by_type(NodeType::Tag);

    REQUIRE(tags.size() == 1);
    REQUIRE(tags[0].id == "tag:a");
}

TEST_CASE("nodes_by_type works on a dirty graph without rebuild") {
    Graph g;
    g.upsert_node(make_tag("tag:a"));
    REQUIRE(g.dirty());

    auto tags = g.nodes_by_type(NodeType::Tag);

    REQUIRE(tags.size() == 1);
}
```

- [ ] **Step 2: Build and run to verify failure**

```bash
cmake --preset debug && cmake --build build/debug
```
Expected: **compile error** — `nodes_by_type` is not a member of `Graph`.

- [ ] **Step 3: Implement `Graph::nodes_by_type`**

In `libdelfos/include/delfos/graph.hpp`, public section, directly after the `nodes_for_file` method (ends ~line 96):

```cpp
    // Live nodes of one type, in unspecified order. Iterates id_index_ (kept
    // current through mutation), so it works on dirty graphs — no rebuild()
    // needed, unlike nodes_view().
    std::vector<NodeData> nodes_by_type(NodeType type) const {
        std::vector<NodeData> out;
        for (const auto& entry : id_index_) {
            const NodeData& nd = nodes_[entry.second];
            if (nd.type == type && nd.status == NodeStatus::Active) {
                out.push_back(nd);
            }
        }
        return out;
    }
```

- [ ] **Step 4: Build and run C++ tests**

```bash
cmake --build build/debug && ctest --test-dir build/debug --output-on-failure
```
Expected: all tests PASS (including the three new ones).

- [ ] **Step 5: Expose it on the `Store` binding**

In `libdelfos/bindings/py_delfos.cpp`, inside `struct Store`, after `list_indexed_files()` (~line 255):

```cpp
    // Live nodes of one type — backs NativeGraphStore.list_tag_values().
    std::vector<NodeData> list_nodes_by_type(int node_type) {
        return graph_.nodes_by_type(static_cast<NodeType>(node_type));
    }
```

And in `NB_MODULE`, next to the other read bindings (after the `vector_search` `.def`, ~line 397):

```cpp
        .def("list_nodes_by_type", &Store::list_nodes_by_type, nb::arg("node_type"))
```

- [ ] **Step 6: Rebuild everything and verify**

```bash
cmake --build build/debug && ctest --test-dir build/debug --output-on-failure
uv pip install -e .
uv run python -c "from delfos import _delfos; s=_delfos.Store('/tmp/x', 8, 'm'); s.initialize(); print(s.list_nodes_by_type(_delfos.NODE_TYPE_TAG))"
```
Expected: ctest PASS; the Python one-liner prints `[]`.

- [ ] **Step 7: Commit**

```bash
git add libdelfos/include/delfos/graph.hpp libdelfos/bindings/py_delfos.cpp libdelfos/tests/test_graph.cpp
git commit -m "feat: add node scan by type to the C++ store"
```

---

### Task 2: `GraphStore.list_tag_values`

**Files:**
- Modify: `delfos/store/base.py` (abstract method after `vector_search`, ~line 160; add `TagCategory` to the `from delfos.schema import` block)
- Modify: `delfos/store/native_store.py` (implementation after `vector_search`, ~line 341)
- Test: `tests/store/test_native_store.py` (append)

**Interfaces:**
- Consumes: `Store.list_nodes_by_type(node_type: int)` from Task 1; existing `_native_to_pydantic`, `_delfos.NODE_TYPE_TAG`, `TagNode`.
- Produces: `GraphStore.list_tag_values(category: TagCategory) -> list[str]` (sorted, distinct), used by Task 4.

- [ ] **Step 1: Write the failing test**

Append to `tests/store/test_native_store.py`:

```python
def test_list_tag_values_returns_sorted_distinct_values_for_category(tmp_path: Path) -> None:
    store = NativeGraphStore(tmp_path / "g", embedding_dim=8, embedding_model="m")
    store.initialize()
    now = datetime(2026, 7, 5, 12, 0, 0)
    with store.transaction():
        store.upsert_node(
            TagNode(id="tag:arch_layer:storage", indexed_at=now,
                    category=TagCategory.ARCH_LAYER, value="storage")
        )
        store.upsert_node(
            TagNode(id="tag:arch_layer:cli", indexed_at=now,
                    category=TagCategory.ARCH_LAYER, value="cli")
        )
        store.upsert_node(
            TagNode(id="tag:pattern_type:factory", indexed_at=now,
                    category=TagCategory.PATTERN_TYPE, value="factory")
        )

    assert store.list_tag_values(TagCategory.ARCH_LAYER) == ["cli", "storage"]
    assert store.list_tag_values(TagCategory.PATTERN_TYPE) == ["factory"]
    assert store.list_tag_values(TagCategory.LANGUAGE) == []
    store.close()
```

Match the file's existing imports; add any of `datetime`, `TagCategory`, `TagNode` that are missing to its import block.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store/test_native_store.py::test_list_tag_values_returns_sorted_distinct_values_for_category -v`
Expected: FAIL — `AttributeError: ... has no attribute 'list_tag_values'` (or pytest collection TypeError once the ABC method exists but the impl doesn't).

- [ ] **Step 3: Add the abstract method**

In `delfos/store/base.py`, add `TagCategory` to the existing `from delfos.schema import` block, then after `vector_search`:

```python
    @abstractmethod
    def list_tag_values(self, category: TagCategory) -> list[str]:
        """Return the sorted distinct values of ACTIVE tag nodes in ``category``.

        Powers the ``annotate`` tool's vocabulary echo: agents are nudged to
        reuse an existing value instead of coining a near-synonym.
        """
```

- [ ] **Step 4: Implement in `NativeGraphStore`**

In `delfos/store/native_store.py`, after `vector_search`:

```python
    def list_tag_values(self, category: TagCategory) -> list[str]:
        values: set[str] = set()
        for nd in self._store.list_nodes_by_type(_NODE_TYPE_TO_NATIVE[NodeType.TAG]):
            node = _native_to_pydantic(nd)
            if isinstance(node, TagNode) and node.category == category:
                values.add(node.value)
        return sorted(values)
```

(`_NODE_TYPE_TO_NATIVE`, `NodeType`, `TagNode`, and `_native_to_pydantic` already exist in this module.)

- [ ] **Step 5: Run the test and full suite checks**

```bash
uv run pytest tests/store/ -v
uv run pyright
uv run ruff check .
```
Expected: PASS / 0 errors / clean.

- [ ] **Step 6: Commit**

```bash
git add delfos/store/base.py delfos/store/native_store.py tests/store/test_native_store.py
git commit -m "feat: expose distinct tag values per category on GraphStore"
```

---

### Task 3: `delfos/enrich` package — normalization and id helpers

**Files:**
- Create: `delfos/enrich/__init__.py`
- Create: `delfos/enrich/service.py` (helpers only in this task; service class in Task 4)
- Create: `tests/enrich/__init__.py` (empty)
- Test: `tests/enrich/test_service.py`

**Interfaces:**
- Consumes: nothing outside stdlib.
- Produces (used by Task 4 and Task 6): `_normalize_phrase(str) -> str`, `_normalize_tag_value(str) -> str`, `_concept_cue_id(source_file: str, phrase: str) -> str`, constants `MAX_CONCEPTS_PER_CALL = 10`, `MAX_PHRASE_LENGTH = 100`, and `class EnrichmentError(ValueError)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/enrich/__init__.py` (empty) and `tests/enrich/test_service.py`:

```python
"""Tests for the agent-driven enrichment write path."""

from __future__ import annotations

from delfos.enrich.service import (
    _concept_cue_id,  # pyright: ignore[reportPrivateUsage]
    _normalize_phrase,  # pyright: ignore[reportPrivateUsage]
    _normalize_tag_value,  # pyright: ignore[reportPrivateUsage]
)


def test_normalize_phrase_lowercases_and_collapses_whitespace() -> None:
    assert _normalize_phrase("  Crash \t Recovery ") == "crash recovery"
    assert _normalize_phrase("") == ""
    assert _normalize_phrase("   ") == ""


def test_normalize_tag_value_lowercases_and_hyphenates() -> None:
    assert _normalize_tag_value("Storage Engine") == "storage-engine"
    assert _normalize_tag_value("  CLI  ") == "cli"
    assert _normalize_tag_value("   ") == ""


def test_concept_cue_id_mirrors_error_cue_scheme_and_is_stable() -> None:
    a = _concept_cue_id("a.py", "crash recovery")
    b = _concept_cue_id("a.py", "crash recovery")
    assert a == b
    assert a.startswith("cue:concept:a.py::")
    assert len(a.split("::")[-1]) == 12
    assert _concept_cue_id("b.py", "crash recovery") != a
    assert _concept_cue_id("a.py", "rate limiting") != a
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/enrich/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'delfos.enrich'`.

- [ ] **Step 3: Implement the helpers**

Create `delfos/enrich/service.py`:

```python
"""The write path: agent-driven enrichment of content nodes.

The calling agent is the extractor (the write-path extension of the read
path's "the calling agent is the planner"): it supplies concept phrases and
semantic tag values for content it has actually read, via the MCP ``annotate``
tool. This module never calls a chat LLM. Sits entirely on top of
:class:`~delfos.store.base.GraphStore`.
"""

from __future__ import annotations

import hashlib

MAX_CONCEPTS_PER_CALL = 10
MAX_PHRASE_LENGTH = 100


class EnrichmentError(ValueError):
    """Raised when an annotate call is invalid; the message is agent-facing."""


def _normalize_phrase(phrase: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return " ".join(phrase.split()).lower()


def _normalize_tag_value(value: str) -> str:
    """Lowercase and replace whitespace runs with hyphens (``Storage Engine`` -> ``storage-engine``)."""
    return "-".join(value.split()).lower()


def _concept_cue_id(source_file: str, phrase: str) -> str:
    """Deterministic concept-cue id; mirrors the error-cue scheme in the extractor."""
    slug = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:12]
    return f"cue:concept:{source_file}::{slug}"
```

Create `delfos/enrich/__init__.py`:

```python
"""Delfos write path: agent-driven enrichment (concept cues + semantic tags)."""

from .service import EnrichmentError

__all__ = ["EnrichmentError"]
```

(Task 4 extends this `__init__` with the service and outcome types.)

- [ ] **Step 4: Run tests and checks**

```bash
uv run pytest tests/enrich/ -v && uv run pyright && uv run ruff check .
```
Expected: PASS / 0 errors / clean.

- [ ] **Step 5: Commit**

```bash
git add delfos/enrich tests/enrich
git commit -m "feat: add enrich package with normalization and cue-id helpers"
```

---

### Task 4: `EnrichmentService.annotate`

**Files:**
- Modify: `delfos/enrich/service.py` (add `AnnotationOutcome`, `EnrichmentService`)
- Modify: `delfos/enrich/__init__.py` (export them)
- Create: `tests/enrich/conftest.py`
- Test: `tests/enrich/test_service.py` (append)

**Interfaces:**
- Consumes: `GraphStore` (incl. Task 2's `list_tag_values`), `Embedder` protocol, schema types.
- Produces (used by Task 5):

```python
@dataclass(frozen=True)
class AnnotationOutcome:
    content_id: str
    written_cue_ids: list[str]
    written_tag_ids: list[str]
    dropped_phrases: list[str]
    existing_values: dict[str, list[str]]  # keyed by TagCategory value: "arch_layer", "pattern_type"

class EnrichmentService:
    def __init__(self, store: GraphStore, embedder: Embedder) -> None: ...
    def annotate(
        self,
        content_id: str,
        concepts: Sequence[str] = (),
        *,
        arch_layer: str | None = None,
        pattern_type: str | None = None,
    ) -> AnnotationOutcome: ...
```

- [ ] **Step 1: Create the test fixtures**

Create `tests/enrich/conftest.py` (reuses the reconstruct fixture helpers rather than duplicating them):

```python
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import (
    EMB_DIM,
    EMB_MODEL,
    FakeEmbedder,
    load,
    make_content,
    vec,
)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL)
    s.initialize()
    yield s
    s.close()


__all__ = ["EMB_DIM", "EMB_MODEL", "FakeEmbedder", "load", "make_content", "store", "vec"]
```

- [ ] **Step 2: Write the failing service tests**

Append to `tests/enrich/test_service.py`:

```python
import pytest

from delfos.enrich import AnnotationOutcome, EnrichmentError, EnrichmentService
from delfos.schema import CueNode, CueType, Direction, EdgeType, TagCategory
from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import FakeEmbedder, load, make_content, make_cue, vec


def _service(store: NativeGraphStore, mapping: dict[str, list[float]]) -> EnrichmentService:
    return EnrichmentService(store, FakeEmbedder(mapping))


def test_annotate_writes_concept_cues_edges_and_tags(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    outcome = svc.annotate(
        "content:1", ["Crash  Recovery"], arch_layer="Storage Engine", pattern_type=None
    )

    assert len(outcome.written_cue_ids) == 1
    cue = store.get_node(outcome.written_cue_ids[0])
    assert isinstance(cue, CueNode)
    assert cue.cue_type == CueType.CONCEPT
    assert cue.text == "crash recovery"
    assert cue.embedding is not None

    # CUE_OF edge: cue -> content
    neighbors = store.neighbors(cue.id, edge_type=EdgeType.CUE_OF, direction=Direction.OUTGOING)
    assert [n.id for n in neighbors] == ["content:1"]

    # TAGGED_WITH edge: content -> normalized tag
    assert outcome.written_tag_ids == ["tag:arch_layer:storage-engine"]
    tags = store.neighbors("content:1", edge_type=EdgeType.TAGGED_WITH, direction=Direction.OUTGOING)
    assert "tag:arch_layer:storage-engine" in [t.id for t in tags]


def test_annotate_stamps_target_provenance_on_cues(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    outcome = svc.annotate("content:1", ["crash recovery"])

    cue = store.get_node(outcome.written_cue_ids[0])
    assert isinstance(cue, CueNode)
    assert cue.source_file == "a.py"  # make_content uses source_file="a.py"
    assert cue.git_sha == "s"


def test_annotate_is_idempotent(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    first = svc.annotate("content:1", ["crash recovery"], arch_layer="storage")
    second = svc.annotate("content:1", ["crash recovery"], arch_layer="storage")

    assert first.written_cue_ids == second.written_cue_ids
    assert first.written_tag_ids == second.written_tag_ids


def test_annotate_drops_bad_phrases_but_keeps_good_ones(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    outcome = svc.annotate(
        "content:1",
        ["crash recovery", "Save_Snapshot", "   ", "x" * 101, "crash  recovery"],
    )

    assert len(outcome.written_cue_ids) == 1  # only "crash recovery"
    assert outcome.dropped_phrases == ["Save_Snapshot", "   ", "x" * 101, "crash  recovery"]


def test_annotate_rejects_unknown_and_non_content_ids(store: NativeGraphStore) -> None:
    load(store, [make_cue("cue:symbol:a.py::f", "f", embedding=vec(2.0))], [])
    svc = _service(store, {})

    with pytest.raises(EnrichmentError, match="unknown node id"):
        svc.annotate("content:nope", ["a concept"])
    with pytest.raises(EnrichmentError, match="not a content node"):
        svc.annotate("cue:symbol:a.py::f", ["a concept"])


def test_annotate_rejects_too_many_concepts(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])
    svc = _service(store, {})

    with pytest.raises(EnrichmentError, match="at most 10"):
        svc.annotate("content:1", [f"concept {i}" for i in range(11)])


def test_annotate_rejects_empty_tag_value(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])
    svc = _service(store, {})

    with pytest.raises(EnrichmentError, match="empty tag value"):
        svc.annotate("content:1", arch_layer="   ")


def test_annotate_with_only_content_id_is_a_vocab_query(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})
    svc.annotate("content:1", ["crash recovery"], arch_layer="storage", pattern_type="validation")

    outcome = svc.annotate("content:1")

    assert outcome.written_cue_ids == []
    assert outcome.written_tag_ids == []
    assert outcome.existing_values == {
        "arch_layer": ["storage"],
        "pattern_type": ["validation"],
    }


def test_annotate_writes_nothing_when_embedder_fails(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])

    class Boom:
        @property
        def model(self) -> str:
            return "fake-v1"

        @property
        def model_version(self) -> str | None:
            return None

        @property
        def dimensions(self) -> int:
            return 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("dead endpoint")

    svc = EnrichmentService(store, Boom())
    with pytest.raises(RuntimeError, match="dead endpoint"):
        svc.annotate("content:1", ["crash recovery"], arch_layer="storage")

    assert store.neighbors("content:1", edge_type=EdgeType.TAGGED_WITH) == []
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/enrich/test_service.py -v`
Expected: FAIL — `ImportError: cannot import name 'AnnotationOutcome' from 'delfos.enrich'`.

- [ ] **Step 4: Implement `AnnotationOutcome` and `EnrichmentService`**

Append to `delfos/enrich/service.py` (extend the import block at the top accordingly):

```python
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from delfos.indexer.embedder import Embedder
from delfos.schema import ContentNode, CueNode, CueType, Edge, EdgeType, TagCategory, TagNode
from delfos.store import GraphStore


@dataclass(frozen=True)
class AnnotationOutcome:
    """What one ``annotate`` call wrote, dropped, and found already in the graph."""

    content_id: str
    written_cue_ids: list[str]
    written_tag_ids: list[str]
    dropped_phrases: list[str]
    existing_values: dict[str, list[str]]


class EnrichmentService:
    """Write-path operations: agent-supplied concept cues and semantic tags."""

    def __init__(self, store: GraphStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def annotate(
        self,
        content_id: str,
        concepts: Sequence[str] = (),
        *,
        arch_layer: str | None = None,
        pattern_type: str | None = None,
    ) -> AnnotationOutcome:
        """Attach concept cues and semantic tags to one content node.

        Idempotent: ids are deterministic, so retries upsert the same nodes.
        Embedding happens before the transaction opens — an embedder failure
        writes nothing. Everything written carries the target's provenance,
        so a re-index of the file wipes it (delete-and-reindex).
        """
        if len(concepts) > MAX_CONCEPTS_PER_CALL:
            raise EnrichmentError(
                f"got {len(concepts)} concepts; pass at most {MAX_CONCEPTS_PER_CALL} per call"
            )
        target = self._store.get_node(content_id)
        if target is None:
            raise EnrichmentError(f"unknown node id {content_id!r}")
        if not isinstance(target, ContentNode):
            raise EnrichmentError(
                f"{content_id!r} is a {target.node_type.value} node, not a content node"
            )

        accepted, dropped = self._screen_phrases(concepts, target.symbol_name)
        tags = self._screen_tags(arch_layer=arch_layer, pattern_type=pattern_type)

        vectors = self._embedder.embed(accepted) if accepted else []

        indexed_at = datetime.now(tz=UTC)
        cue_ids: list[str] = []
        tag_ids: list[str] = []
        with self._store.transaction():
            for phrase, vector in zip(accepted, vectors, strict=True):
                cue_id = _concept_cue_id(target.source_file, phrase)
                self._store.upsert_node(
                    CueNode(
                        id=cue_id,
                        source_file=target.source_file,
                        git_sha=target.git_sha,
                        indexed_at=indexed_at,
                        cue_type=CueType.CONCEPT,
                        text=phrase,
                        embedding=vector,
                        embedding_model=self._embedder.model,
                        embedding_model_version=self._embedder.model_version,
                    )
                )
                self._store.upsert_edge(
                    Edge(
                        source_id=cue_id,
                        target_id=content_id,
                        edge_type=EdgeType.CUE_OF,
                        source_file=target.source_file,
                        git_sha=target.git_sha,
                        indexed_at=indexed_at,
                    )
                )
                cue_ids.append(cue_id)
            for category, value in tags:
                tag_id = f"tag:{category.value}:{value}"
                self._store.upsert_node(
                    TagNode(id=tag_id, indexed_at=indexed_at, category=category, value=value)
                )
                self._store.upsert_edge(
                    Edge(
                        source_id=content_id,
                        target_id=tag_id,
                        edge_type=EdgeType.TAGGED_WITH,
                        source_file=target.source_file,
                        git_sha=target.git_sha,
                        indexed_at=indexed_at,
                    )
                )
                tag_ids.append(tag_id)

        return AnnotationOutcome(
            content_id=content_id,
            written_cue_ids=cue_ids,
            written_tag_ids=tag_ids,
            dropped_phrases=dropped,
            existing_values={
                TagCategory.ARCH_LAYER.value: self._store.list_tag_values(TagCategory.ARCH_LAYER),
                TagCategory.PATTERN_TYPE.value: self._store.list_tag_values(
                    TagCategory.PATTERN_TYPE
                ),
            },
        )

    @staticmethod
    def _screen_phrases(
        concepts: Sequence[str], symbol_name: str | None
    ) -> tuple[list[str], list[str]]:
        """Normalize phrases; drop empties, overlong ones, duplicates, and the symbol name."""
        symbol = (symbol_name or "").lower()
        accepted: list[str] = []
        dropped: list[str] = []
        seen: set[str] = set()
        for raw in concepts:
            phrase = _normalize_phrase(raw)
            if not phrase or len(phrase) > MAX_PHRASE_LENGTH or phrase == symbol or phrase in seen:
                dropped.append(raw)
                continue
            seen.add(phrase)
            accepted.append(phrase)
        return accepted, dropped

    @staticmethod
    def _screen_tags(
        *, arch_layer: str | None, pattern_type: str | None
    ) -> list[tuple[TagCategory, str]]:
        """Normalize the two agent-writable tag values; reject ones that normalize to empty."""
        out: list[tuple[TagCategory, str]] = []
        pairs = ((TagCategory.ARCH_LAYER, arch_layer), (TagCategory.PATTERN_TYPE, pattern_type))
        for category, raw in pairs:
            if raw is None:
                continue
            value = _normalize_tag_value(raw)
            if not value:
                raise EnrichmentError(f"empty tag value for {category.value}")
            out.append((category, value))
        return out
```

Note the phrase-vs-symbol check compares against the normalized phrase, so `"Save_Snapshot"` must be dropped when `symbol_name == "save_snapshot"`: normalize the comparison by also replacing nothing — `_normalize_phrase("Save_Snapshot")` is `"save_snapshot"`, which equals the lowercased symbol name. (This is why the test uses `Save_Snapshot`.)

Update `delfos/enrich/__init__.py`:

```python
"""Delfos write path: agent-driven enrichment (concept cues + semantic tags)."""

from .service import AnnotationOutcome, EnrichmentError, EnrichmentService

__all__ = ["AnnotationOutcome", "EnrichmentError", "EnrichmentService"]
```

- [ ] **Step 5: Run tests and checks**

```bash
uv run pytest tests/enrich/ -v && uv run pyright && uv run ruff check .
```
Expected: PASS / 0 errors / clean. If the `Boom` embedder fails pyright's `Embedder` protocol check, add `_: Embedder = Boom()`-style assertions are NOT needed — the protocol is structural; just ensure `Boom` has all four members shown above.

- [ ] **Step 6: Commit**

```bash
git add delfos/enrich tests/enrich
git commit -m "feat: add EnrichmentService.annotate write path"
```

---

### Task 5: MCP surface — `annotate` tool, `enrich` prompt, wiring

**Files:**
- Modify: `delfos/mcp/views.py` (add `AnnotateResult` + converter)
- Modify: `delfos/mcp/server.py` (add `_annotate`, `enrich_prompt`, register tool + prompt, extend `build_server`)
- Modify: `delfos/mcp/__main__.py` (construct and pass `EnrichmentService`)
- Test: `tests/mcp/test_views.py`, `tests/mcp/test_server.py` (append)

**Interfaces:**
- Consumes: `EnrichmentService`, `AnnotationOutcome`, `EnrichmentError` from Task 4.
- Produces: MCP tool `annotate(content_id, concepts=None, arch_layer=None, pattern_type=None) -> AnnotateResult`; MCP prompt `enrich(focus="")`; new signature `build_server(service, scip=None, enrich=None)`.

- [ ] **Step 1: Write the failing view test**

Append to `tests/mcp/test_views.py`:

```python
def test_outcome_to_result_maps_all_fields() -> None:
    outcome = AnnotationOutcome(
        content_id="content:1",
        written_cue_ids=["cue:concept:a.py::abc123def456"],
        written_tag_ids=["tag:arch_layer:storage"],
        dropped_phrases=["   "],
        existing_values={"arch_layer": ["storage"], "pattern_type": []},
    )

    result = outcome_to_result(outcome)

    assert result.content_id == "content:1"
    assert result.written_cue_ids == ["cue:concept:a.py::abc123def456"]
    assert result.written_tag_ids == ["tag:arch_layer:storage"]
    assert result.dropped_phrases == ["   "]
    assert result.existing_values == {"arch_layer": ["storage"], "pattern_type": []}
```

Add to that file's imports: `from delfos.enrich import AnnotationOutcome` and `outcome_to_result`, `AnnotateResult` from `delfos.mcp.views`.

Run: `uv run pytest tests/mcp/test_views.py -v` — Expected: FAIL (ImportError).

- [ ] **Step 2: Implement the view**

Append to `delfos/mcp/views.py` (add `from delfos.enrich import AnnotationOutcome` to imports):

```python
class AnnotateResult(BaseModel):
    """What ``annotate`` wrote, dropped, and the existing tag vocabulary to reuse."""

    model_config = ConfigDict(extra="forbid")

    content_id: str
    written_cue_ids: list[str]
    written_tag_ids: list[str]
    dropped_phrases: list[str]
    existing_values: dict[str, list[str]]


def outcome_to_result(outcome: AnnotationOutcome) -> AnnotateResult:
    """Serialize an annotation outcome for the MCP surface."""
    return AnnotateResult(
        content_id=outcome.content_id,
        written_cue_ids=outcome.written_cue_ids,
        written_tag_ids=outcome.written_tag_ids,
        dropped_phrases=outcome.dropped_phrases,
        existing_values=outcome.existing_values,
    )
```

Run: `uv run pytest tests/mcp/test_views.py -v` — Expected: PASS.

- [ ] **Step 3: Write the failing server tests**

Append to `tests/mcp/test_server.py` (add imports: `_annotate` and `enrich_prompt` from `delfos.mcp.server`, `EnrichmentService` from `delfos.enrich`, `FakeEmbedder` from `tests.reconstruct.conftest`, `make_content` and `load` are already imported):

```python
def test_annotate_tool_writes_and_echoes_vocab(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    enrich = EnrichmentService(store, FakeEmbedder({"crash recovery": vec(3.0)}))

    result = _annotate(
        enrich, "content:1", ["crash recovery"], arch_layer="storage", pattern_type=None
    )

    assert result.content_id == "content:1"
    assert len(result.written_cue_ids) == 1
    assert result.written_tag_ids == ["tag:arch_layer:storage"]
    assert result.existing_values["arch_layer"] == ["storage"]


def test_annotate_tool_errors_without_service() -> None:
    with pytest.raises(RuntimeError, match="enrichment unavailable"):
        _annotate(None, "content:1", ["x"], arch_layer=None, pattern_type=None)


def test_enrich_prompt_teaches_the_annotate_discipline() -> None:
    text = enrich_prompt()
    assert "annotate" in text
    assert "1-5 concept phrases" in text
    assert "reuse" in text.lower()

    focused = enrich_prompt(focus="the storage layer")
    assert "the storage layer" in focused


def test_build_server_registers_annotate_and_enrich(store: NativeGraphStore) -> None:
    svc = make_service(store, vec(0.0))
    enrich = EnrichmentService(store, FakeEmbedder({}))

    server = build_server(svc, None, enrich)

    tools = asyncio.run(server.list_tools())
    assert "annotate" in [t.name for t in tools]
    prompts = asyncio.run(server.list_prompts())
    assert "enrich" in [p.name for p in prompts]
```

(`asyncio.run(server.list_tools())` / `list_prompts()` is the idiom the existing tests in this file already use.)

Run: `uv run pytest tests/mcp/test_server.py -v` — Expected: FAIL (ImportError: `_annotate`).

- [ ] **Step 4: Implement `_annotate`, `enrich_prompt`, and registration**

In `delfos/mcp/server.py`:

Add imports: `from delfos.enrich import EnrichmentService` and extend the `.views` import with `AnnotateResult, outcome_to_result`.

Add near `_require_scip`:

```python
_ENRICH_UNAVAILABLE = (
    "enrichment unavailable: the server was built without an EnrichmentService."
)


def _annotate(
    enrich: EnrichmentService | None,
    content_id: str,
    concepts: list[str] | None,
    *,
    arch_layer: str | None,
    pattern_type: str | None,
) -> AnnotateResult:
    if enrich is None:
        raise RuntimeError(_ENRICH_UNAVAILABLE)
    outcome = enrich.annotate(
        content_id, concepts or [], arch_layer=arch_layer, pattern_type=pattern_type
    )
    return outcome_to_result(outcome)
```

Add next to `reconstruct_prompt`:

```python
def enrich_prompt(focus: str = "") -> str:
    """Protocol text teaching the agent to enrich content it has actually read."""
    scope = f" Focus on: {focus}." if focus else ""
    return (
        f"Enrich the code-memory graph with what you learned; you are the "
        f"extractor.{scope}\n\n"
        f"Protocol:\n"
        f"1. Only annotate content nodes whose bodies you have read via `fetch`.\n"
        f"2. Call `annotate` with 1-5 concept phrases per node describing what the "
        f"code is about (e.g. 'rate limiting', 'crash recovery'). Never restate the "
        f"symbol name — that cue already exists.\n"
        f"3. Optionally set arch_layer (which architectural layer the code belongs "
        f"to) and pattern_type (the recurring pattern it embodies).\n"
        f"4. The result echoes existing tag values: reuse one unless none fits.\n"
        f"5. Annotations are wiped when their file is re-indexed, so do not "
        f"annotate code you are about to change."
    )
```

Extend `build_server` — new signature and registrations (keep all existing ones):

```python
def build_server(
    service: ReconstructionService,
    scip: ScipService | None = None,
    enrich: EnrichmentService | None = None,
) -> FastMCP:
```

Inside, after the `type_definition` tool:

```python
    @mcp.tool()
    def annotate(  # pyright: ignore[reportUnusedFunction]
        content_id: str,
        concepts: list[str] | None = None,
        arch_layer: str | None = None,
        pattern_type: str | None = None,
    ) -> AnnotateResult:
        """Attach concept cues and semantic tags to a content node you have read.

        Concepts are short phrases describing what the code is about. The result
        echoes existing arch_layer/pattern_type values — reuse them when they fit.
        Call with only content_id to just see the current vocabulary.
        """
        return _annotate(
            enrich, content_id, concepts, arch_layer=arch_layer, pattern_type=pattern_type
        )
```

And after the `reconstruct` prompt:

```python
    @mcp.prompt()
    def enrich_memory(focus: str = "") -> str:  # pyright: ignore[reportUnusedFunction]
        """Teach the agent to write concept cues and semantic tags for code it read."""
        return enrich_prompt(focus)
```

**Naming note:** FastMCP registers the prompt under the function name. The spec calls the prompt `enrich`, but the local variable `enrich` (the service parameter) would shadow it. Register with an explicit name instead: `@mcp.prompt(name="enrich")` on `def enrich_memory(...)` — verify FastMCP's `prompt()` decorator accepts `name=` (it does in current versions); if not, rename the `enrich` parameter to `enrich_service` and name the function `enrich`.

Update the module docstring's first line ("four graph tools + a reconstruct prompt") to reflect the new tool and prompt.

- [ ] **Step 5: Wire `__main__.py`**

In `delfos/mcp/__main__.py`, add `from delfos.enrich import EnrichmentService` and replace the last two lines of `main`:

```python
    enrich = EnrichmentService(store, embedder)
    logger.info("[5/5] serving MCP over stdio (Ctrl-C to stop)")
    build_server(service, scip, enrich).run()
```

- [ ] **Step 6: Run the full suite and checks**

```bash
uv run pytest && uv run pyright && uv run ruff check .
```
Expected: all PASS / 0 errors / clean.

- [ ] **Step 7: Commit**

```bash
git add delfos/mcp delfos/enrich tests/mcp
git commit -m "feat: expose annotate tool and enrich prompt on the MCP server"
```

---

### Task 6: Integration tests — staleness and retrieval win

**Files:**
- Test: `tests/enrich/test_integration.py` (create)
- Modify: `tests/enrich/conftest.py` (add `HashEmbedder`)

**Interfaces:**
- Consumes: `Indexer` pipeline, `EnrichmentService`, `ReconstructionService`, `Workspace`.
- Produces: nothing — proves the two spec-mandated end-to-end behaviors.

- [ ] **Step 1: Add a deterministic embedder to the conftest**

Append to `tests/enrich/conftest.py` (mirrors `_HashEmbedder` in `tests/indexer/test_pipeline_scip.py`; duplicated here because that one is module-private):

```python
import hashlib
import math

HASH_DIM = 32
HASH_MODEL = "hash-sha256-d32"


class HashEmbedder:
    """Deterministic embedder: same text -> same unit vector. No network."""

    @property
    def model(self) -> str:
        return HASH_MODEL

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return HASH_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(b) - 128.0 for b in digest]
            length = math.sqrt(sum(x * x for x in raw)) or 1.0
            out.append([x / length for x in raw])
        return out
```

- [ ] **Step 2: Write the two failing integration tests**

Create `tests/enrich/test_integration.py`:

```python
"""End-to-end proof of the two enrichment guarantees.

1. Staleness: annotations die with their file on re-index (delete-and-reindex).
2. Retrieval win: an annotated concept phrase is findable via `search` and
   leads to the content node via `traverse_forward`.

SCIP generation is forced to fail so content ids use the deterministic
fallback scheme ``content:{source_file}::{qualified_name}`` regardless of
whether scip-python is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from delfos.enrich import EnrichmentService
from delfos.indexer import Indexer
from delfos.indexer import pipeline as pipeline_mod
from delfos.reconstruct import ReconstructionService
from delfos.scip.generate import ScipGenerationError
from delfos.store.native_store import NativeGraphStore
from delfos.workspace import Workspace

from .conftest import HASH_DIM, HASH_MODEL, HashEmbedder

CONTENT_ID = "content:mod.py::save_snapshot"


@pytest.fixture(autouse=True)
def _no_scip(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_k: object) -> object:
        raise ScipGenerationError("scip disabled for deterministic ids")

    monkeypatch.setattr(pipeline_mod, "generate_scip_index", _raise)


def _index(repo: Path, store: NativeGraphStore) -> None:
    Indexer(store, HashEmbedder()).index(repo, workspace=Workspace(repo))


def _make_repo(tmp_path: Path, body: str) -> tuple[Path, NativeGraphStore]:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "mod.py").write_text(body)
    store = NativeGraphStore(
        tmp_path / "graph", embedding_dim=HASH_DIM, embedding_model=HASH_MODEL
    )
    store.initialize()
    return repo, store


def test_annotations_die_when_their_file_is_reindexed(tmp_path: Path) -> None:
    repo, store = _make_repo(tmp_path, "def save_snapshot():\n    return 1\n")
    _index(repo, store)
    enrich = EnrichmentService(store, HashEmbedder())

    outcome = enrich.annotate(CONTENT_ID, ["crash recovery"], arch_layer="storage")
    cue_id = outcome.written_cue_ids[0]
    assert store.get_node(cue_id) is not None

    (repo / "mod.py").write_text("def save_snapshot():\n    return 2\n")
    _index(repo, store)

    assert store.get_node(cue_id) is None  # annotation died with the file
    assert store.get_node(CONTENT_ID) is not None  # content was re-indexed
    store.close()


def test_search_finds_content_via_concept_cue(tmp_path: Path) -> None:
    repo, store = _make_repo(tmp_path, "def save_snapshot():\n    return 1\n")
    _index(repo, store)
    EnrichmentService(store, HashEmbedder()).annotate(CONTENT_ID, ["crash recovery"])

    service = ReconstructionService(store, HashEmbedder())
    cues = service.search("crash recovery", k=1)

    assert len(cues) == 1
    assert cues[0].text == "crash recovery"
    contents = service.traverse_forward([cues[0].id])
    assert [c.id for c in contents] == [CONTENT_ID]
    store.close()
```

- [ ] **Step 3: Run to verify state**

Run: `uv run pytest tests/enrich/test_integration.py -v`
Expected: PASS (Tasks 1–4 are already implemented; if anything fails here it is a real integration bug — debug it, do not weaken the test).

- [ ] **Step 4: Full suite and checks**

```bash
uv run pytest && uv run pyright && uv run ruff check .
```
Expected: all PASS / 0 errors / clean.

- [ ] **Step 5: Commit**

```bash
git add tests/enrich
git commit -m "test: prove enrichment staleness and retrieval end to end"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/decisions.md` (new decision under "Read path" section area — add a "Write path" section)
- Modify: `ARCHITECTURE.md` (MCP surface + new `enrich` package)
- Modify: `README.md` (tool list in the MCP section)
- Modify: `CLAUDE.md` (MCP surface + packages list)

**Interfaces:** none — documentation of Tasks 1–6.

- [ ] **Step 1: Add the decision record**

In `docs/decisions.md`, after the "Read path" section, add:

```markdown
## Write path

### Enrichment is agent-driven; the calling agent is the extractor

`CONCEPT` cues and `ARCH_LAYER`/`PATTERN_TYPE` tags are written by the calling
agent through the MCP `annotate` tool (taught by the `enrich` prompt) — the
write-path extension of "the calling agent is the planner". Delfos never calls
a chat LLM at index time and never holds a chat-model API key. Tag values are
open vocabulary, normalized (lowercase, hyphenated); the tool echoes existing
values per category so agents converge on a shared vocabulary instead of
coining near-synonyms. Annotations carry the target content node's
`source_file`/`git_sha`, so delete-and-reindex wipes them when the file
changes — a concept extracted from old code may be wrong for the new code, and
a stale concept cue is worse than a missing one.
```

- [ ] **Step 2: Update `ARCHITECTURE.md`, `README.md`, `CLAUDE.md`**

- `ARCHITECTURE.md`: add the `enrich` package to the package walkthrough; add `annotate` + `enrich` to the MCP surface description; mention `GraphStore.list_tag_values` and the C++ `list_nodes_by_type` binding where the store API is described.
- `README.md`: in the MCP tool list, add `annotate` under a "Write tool" bullet and `enrich` next to the `reconstruct` prompt.
- `CLAUDE.md`: in the "MCP surface" section, add: `annotate` (write tool: agent-supplied concept cues + arch_layer/pattern_type tags, provenance-stamped so re-index wipes them) and the `enrich` prompt; add `enrich` to the Python package list.

Match each file's existing tone and depth — one to three sentences per mention, not a re-explanation of the design (the spec and decisions.md carry that).

- [ ] **Step 3: Verify docs are consistent**

Re-read the diff of all four files:
```bash
git diff docs/decisions.md ARCHITECTURE.md README.md CLAUDE.md
```
Check: tool named `annotate` everywhere, prompt named `enrich` everywhere, no claim that indexing calls an LLM.

- [ ] **Step 4: Commit**

```bash
git add docs/decisions.md ARCHITECTURE.md README.md CLAUDE.md
git commit -m "docs: document agent-driven enrichment surface"
```

---

## Final verification (after all tasks)

```bash
cmake --preset debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure
uv pip install -e .
uv run pytest
uv run pyright
uv run ruff check .
uv run ruff format --check .
```

All green → follow superpowers:finishing-a-development-branch (PR against `main`; note the spec branch `docs/agent-driven-enrichment-spec` should merge first or be folded into this branch).

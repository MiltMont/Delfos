# Reconstruction Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Delfos's read path — three pure graph primitives (`search`, `traverse_forward`, `traverse_reverse`) and an LLM-driven depth-first `reconstruct` — as a service layer on top of `GraphStore`.

**Architecture:** A new `delfos/reconstruct/` package exposes `ReconstructionService`, constructed with a `GraphStore`, an `Embedder`, and a provider-agnostic `HopPlanner`. The primitives are pure graph operations; `reconstruct` runs a sequential DFS where an LLM (`HopPlanner`) decides, at each hop, which neighbors to collect and which single neighbor to descend into, bounded by a budget on total planner calls. Tests inject a `FakeHopPlanner` and a fake embedder, so the whole layer is deterministic and offline.

**Tech Stack:** Python 3.12, Pydantic v2, `NativeGraphStore` (libdelfos C++ backend), pytest, pyright (strict), ruff.

## Global Constraints

- Python `>=3.12`; target `py312`.
- Pyright **strict mode** — all new code fully typed, no implicit `Any`.
- All Pydantic models set `model_config = ConfigDict(extra="forbid")`.
- Ruff lint select = `E, F, I, UP, B`; line length 100. Run `uv run ruff format .` and `uv run ruff check .`.
- **Never bypass `GraphStore`** — every data access goes through the store interface.
- `status=ACTIVE` nodes only in the read path; follow a single `REDIRECTS_TO` edge transparently when present.
- Type-check command: `uv run pyright`. Test command: `uv run pytest`.

---

## File Structure

- Create `delfos/reconstruct/__init__.py` — exports `ReconstructionService`, `TagFilter`.
- Create `delfos/reconstruct/planner.py` — `CandidateSummary`, `HopRequest`, `Collected`, `HopDecision`, `HopPlanner` Protocol.
- Create `delfos/reconstruct/summaries.py` — `summarize(node, tags)` → `CandidateSummary`.
- Create `delfos/reconstruct/service.py` — `ReconstructionService` + `TagFilter` alias.
- Create `delfos/reconstruct/planners/__init__.py`
- Create `delfos/reconstruct/planners/fake.py` — `FakeHopPlanner` (scripted; shipped so it can back demos/tests).
- Create `tests/reconstruct/__init__.py`
- Create `tests/reconstruct/conftest.py` — `FakeEmbedder`, graph-fixture builders.
- Create `tests/reconstruct/test_planner.py`, `test_summaries.py`, `test_search.py`, `test_traverse.py`, `test_reconstruct.py`.

---

### Task 1: Planner models and `HopPlanner` protocol

**Files:**
- Create: `delfos/reconstruct/__init__.py`
- Create: `delfos/reconstruct/planner.py`
- Create: `tests/reconstruct/__init__.py`
- Test: `tests/reconstruct/test_planner.py`

**Interfaces:**
- Produces:
  - `CandidateSummary(id: str, node_kind: Literal["cue","content"], label: str, snippet: str | None, tags: list[str])`
  - `Collected(id: str, relevance: float)` — `relevance` constrained `0.0..1.0`
  - `HopRequest(query: str, current: CandidateSummary, candidates: list[CandidateSummary], hops_remaining: int)`
  - `HopDecision(collect: list[Collected], descend_into: str | None, stop: bool)`
  - `HopPlanner` Protocol with `decide(self, request: HopRequest) -> HopDecision`

- [ ] **Step 1: Create the empty package marker files**

Create `delfos/reconstruct/__init__.py` with a module docstring only (exports are added in Task 9):

```python
"""Delfos read path: search, traversal, and LLM-driven reconstruction."""
```

Create `tests/reconstruct/__init__.py` as an empty file (no content).

- [ ] **Step 2: Write the failing test**

Create `tests/reconstruct/test_planner.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from delfos.reconstruct.planner import (
    CandidateSummary,
    Collected,
    HopDecision,
    HopRequest,
)


def test_candidate_summary_roundtrips() -> None:
    c = CandidateSummary(
        id="content-1",
        node_kind="content",
        label="def load_config()",
        snippet="Load the config.",
        tags=["language=python"],
    )
    assert c.id == "content-1"
    assert c.node_kind == "content"


def test_collected_relevance_must_be_in_unit_range() -> None:
    with pytest.raises(ValidationError):
        Collected(id="x", relevance=1.5)
    with pytest.raises(ValidationError):
        Collected(id="x", relevance=-0.1)


def test_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HopDecision(collect=[], descend_into=None, stop=False, bogus=1)  # type: ignore[call-arg]


def test_hop_request_holds_candidates() -> None:
    cur = CandidateSummary(id="cue-1", node_kind="cue", label="auth", snippet=None, tags=[])
    req = HopRequest(query="how does auth work", current=cur, candidates=[cur], hops_remaining=3)
    assert req.hops_remaining == 3
    assert req.candidates[0].id == "cue-1"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'delfos.reconstruct.planner'`

- [ ] **Step 4: Write the implementation**

Create `delfos/reconstruct/planner.py`:

```python
"""Provider-agnostic interface for the per-hop reconstruction planner.

The planner is the LLM in the `reconstruct` loop. It sees the current node and
its candidate neighbors and returns which to collect plus which single one to
descend into. Concrete backends (OpenAI/Anthropic) are added at implementation
time; this module fixes only the data contract.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class CandidateSummary(BaseModel):
    """The compact view of a node the planner reasons over."""

    model_config = ConfigDict(extra="forbid")

    id: str
    node_kind: Literal["cue", "content"]
    label: str
    snippet: str | None = None
    tags: list[str] = Field(default_factory=list)


class Collected(BaseModel):
    """A node the planner chose to include, with its relevance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    relevance: float = Field(ge=0.0, le=1.0)


class HopRequest(BaseModel):
    """Everything the planner needs to decide a single hop."""

    model_config = ConfigDict(extra="forbid")

    query: str
    current: CandidateSummary
    candidates: list[CandidateSummary]
    hops_remaining: int


class HopDecision(BaseModel):
    """The planner's decision for one hop."""

    model_config = ConfigDict(extra="forbid")

    collect: list[Collected] = Field(default_factory=list)
    descend_into: str | None = None
    stop: bool = False


@runtime_checkable
class HopPlanner(Protocol):
    """Decides one hop of the reconstruction walk."""

    def decide(self, request: HopRequest) -> HopDecision: ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/reconstruct/test_planner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Type-check and commit**

```bash
uv run ruff format delfos/reconstruct tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add delfos/reconstruct tests/reconstruct
git commit -m "feat(reconstruct): planner data contract + HopPlanner protocol"
```

Expected: pyright reports 0 errors.

---

### Task 2: Node summaries

**Files:**
- Create: `delfos/reconstruct/summaries.py`
- Test: `tests/reconstruct/test_summaries.py`

**Interfaces:**
- Consumes: `CandidateSummary` (Task 1); `CueNode`, `ContentNode` from `delfos.schema`.
- Produces: `summarize(node: CueNode | ContentNode, tags: Sequence[str] = ()) -> CandidateSummary`

- [ ] **Step 1: Write the failing test**

Create `tests/reconstruct/test_summaries.py`:

```python
from __future__ import annotations

from datetime import datetime

from delfos.reconstruct.summaries import summarize
from delfos.schema import ContentKind, ContentNode, CueNode, CueType, MemoryLayer

NOW = datetime(2026, 6, 23, 12, 0, 0)


def _cue() -> CueNode:
    return CueNode(
        id="cue-1", source_file="a.py", git_sha="s", indexed_at=NOW,
        cue_type=CueType.SYMBOL, text="load_config",
    )


def _content(body: str) -> ContentNode:
    return ContentNode(
        id="content-1", source_file="a.py", git_sha="s", indexed_at=NOW,
        kind=ContentKind.FUNCTION, memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="load_config", signature="def load_config() -> Config",
        docstring="Load it.", body=body,
    )


def test_summarize_cue_uses_text_as_label() -> None:
    s = summarize(_cue())
    assert s.node_kind == "cue"
    assert s.label == "load_config"
    assert s.snippet is None


def test_summarize_content_prefers_signature_and_docstring() -> None:
    s = summarize(_content(body="def load_config(): ..."))
    assert s.node_kind == "content"
    assert s.label == "def load_config() -> Config"
    assert s.snippet == "Load it."


def test_summarize_truncates_long_body_when_no_docstring() -> None:
    long_body = "x" * 1000
    content = _content(body=long_body)
    content.docstring = None
    s = summarize(content)
    assert s.snippet is not None
    assert len(s.snippet) <= 501  # 500 chars + ellipsis
    assert s.snippet.endswith("…")


def test_summarize_passes_through_tags() -> None:
    s = summarize(_cue(), tags=["language=python"])
    assert s.tags == ["language=python"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_summaries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'delfos.reconstruct.summaries'`

- [ ] **Step 3: Write the implementation**

Create `delfos/reconstruct/summaries.py`:

```python
"""Map a graph node to the compact `CandidateSummary` the planner sees.

This is the single place that decides how much of each node is exposed to the
LLM, which keeps token cost controlled and prompt-shaping out of the traversal
loop.
"""

from __future__ import annotations

from collections.abc import Sequence

from delfos.schema import ContentNode, CueNode

from .planner import CandidateSummary

_SNIPPET_LIMIT = 500


def _truncate(text: str, limit: int = _SNIPPET_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def summarize(node: CueNode | ContentNode, tags: Sequence[str] = ()) -> CandidateSummary:
    """Build the planner-facing summary of ``node``."""
    if isinstance(node, CueNode):
        return CandidateSummary(
            id=node.id, node_kind="cue", label=node.text, snippet=None, tags=list(tags)
        )

    label = node.signature or node.symbol_name or node.kind.value
    snippet = node.docstring if node.docstring is not None else _truncate(node.body)
    return CandidateSummary(
        id=node.id, node_kind="content", label=label, snippet=snippet, tags=list(tags)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/reconstruct/test_summaries.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Type-check and commit**

```bash
uv run ruff format delfos/reconstruct tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add delfos/reconstruct/summaries.py tests/reconstruct/test_summaries.py
git commit -m "feat(reconstruct): node summarization for the planner"
```

Expected: pyright 0 errors.

---

### Task 3: `FakeHopPlanner` and shared test doubles

**Files:**
- Create: `delfos/reconstruct/planners/__init__.py`
- Create: `delfos/reconstruct/planners/fake.py`
- Create: `tests/reconstruct/conftest.py`
- Test: `tests/reconstruct/test_planner.py` (append)

**Interfaces:**
- Consumes: `HopRequest`, `HopDecision` (Task 1).
- Produces:
  - `FakeHopPlanner(decisions: Sequence[HopDecision], *, error_after: int | None = None)` with `.decide(request) -> HopDecision`, `.call_count: int`, `.requests: list[HopRequest]`.
  - `FakeEmbedder(mapping: dict[str, list[float]], *, model: str = "fake-v1", dimensions: int)` satisfying the `Embedder` protocol.
  - pytest fixtures: `EMB_DIM` constant (8), `NOW` constant, node builders `make_cue/make_content/make_tag`, and a `built_store` factory described in Task 4.

- [ ] **Step 1: Write the failing test (append to test_planner.py)**

Append to `tests/reconstruct/test_planner.py`:

```python
from delfos.reconstruct.planner import HopPlanner
from delfos.reconstruct.planners.fake import FakeHopPlanner


def test_fake_planner_returns_scripted_decisions_then_stops() -> None:
    d1 = HopDecision(collect=[Collected(id="a", relevance=0.9)], descend_into="a", stop=False)
    planner = FakeHopPlanner([d1])
    cur = CandidateSummary(id="cue-1", node_kind="cue", label="x", snippet=None, tags=[])
    req = HopRequest(query="q", current=cur, candidates=[cur], hops_remaining=3)

    assert planner.decide(req) is d1
    # Once the script is exhausted, it returns a terminal stop decision.
    after = planner.decide(req)
    assert after.stop is True
    assert planner.call_count == 2
    assert len(planner.requests) == 2


def test_fake_planner_raises_after_error_after() -> None:
    planner = FakeHopPlanner([], error_after=0)
    cur = CandidateSummary(id="cue-1", node_kind="cue", label="x", snippet=None, tags=[])
    req = HopRequest(query="q", current=cur, candidates=[cur], hops_remaining=1)
    import pytest

    with pytest.raises(RuntimeError):
        planner.decide(req)


def test_fake_planner_satisfies_protocol() -> None:
    assert isinstance(FakeHopPlanner([]), HopPlanner)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'delfos.reconstruct.planners'`

- [ ] **Step 3: Write the FakeHopPlanner implementation**

Create `delfos/reconstruct/planners/__init__.py`:

```python
"""Concrete `HopPlanner` backends."""
```

Create `delfos/reconstruct/planners/fake.py`:

```python
"""A scripted `HopPlanner` for tests and demos — no network, fully deterministic."""

from __future__ import annotations

from collections.abc import Sequence

from delfos.reconstruct.planner import HopDecision, HopRequest


class FakeHopPlanner:
    """Returns pre-scripted decisions in order.

    After the script is exhausted it returns a terminal ``stop`` decision, so a
    walk always halts. With ``error_after=n`` the ``n``-th call (0-indexed) and
    every call after it raise ``RuntimeError``, to exercise partial-result
    handling.
    """

    def __init__(
        self, decisions: Sequence[HopDecision], *, error_after: int | None = None
    ) -> None:
        self._decisions = list(decisions)
        self._error_after = error_after
        self._calls = 0
        self.requests: list[HopRequest] = []

    def decide(self, request: HopRequest) -> HopDecision:
        index = self._calls
        self._calls += 1
        self.requests.append(request)
        if self._error_after is not None and index >= self._error_after:
            raise RuntimeError("FakeHopPlanner scripted failure")
        if index < len(self._decisions):
            return self._decisions[index]
        return HopDecision(collect=[], descend_into=None, stop=True)

    @property
    def call_count(self) -> int:
        return self._calls
```

- [ ] **Step 4: Write the shared conftest test doubles**

Create `tests/reconstruct/conftest.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from delfos.schema import (
    ContentKind,
    ContentNode,
    CueNode,
    CueType,
    Edge,
    EdgeType,
    MemoryLayer,
    Node,
    TagCategory,
    TagNode,
)
from delfos.store.native_store import NativeGraphStore

EMB_DIM = 8
EMB_MODEL = "fake-v1"
NOW = datetime(2026, 6, 23, 12, 0, 0)


class FakeEmbedder:
    """Embedder protocol double: maps known texts to fixed vectors."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    @property
    def model(self) -> str:
        return EMB_MODEL

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return EMB_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping[t] for t in texts]


def vec(seed: float) -> list[float]:
    return [seed + i for i in range(EMB_DIM)]


def make_cue(
    node_id: str, text: str, *, embedding: list[float] | None = None
) -> CueNode:
    return CueNode(
        id=node_id, source_file="a.py", git_sha="s", indexed_at=NOW,
        cue_type=CueType.SYMBOL, text=text,
        embedding=embedding,
        embedding_model=EMB_MODEL if embedding is not None else None,
    )


def make_content(node_id: str, symbol: str) -> ContentNode:
    return ContentNode(
        id=node_id, source_file="a.py", git_sha="s", indexed_at=NOW,
        kind=ContentKind.FUNCTION, memory_layer=MemoryLayer.SEMANTIC,
        symbol_name=symbol, signature=f"def {symbol}()", docstring=None,
        body=f"def {symbol}(): ...",
    )


def make_tag(node_id: str, category: TagCategory, value: str) -> TagNode:
    return TagNode(id=node_id, indexed_at=NOW, category=category, value=value)


def edge(source: str, target: str, edge_type: EdgeType) -> Edge:
    return Edge(source_id=source, target_id=target, edge_type=edge_type)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(
        tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL
    )
    s.initialize()
    yield s
    s.close()


def load(store: NativeGraphStore, nodes: list[Node], edges: list[Edge]) -> None:
    """Persist a fixture graph in one transaction."""
    with store.transaction():
        for node in nodes:
            store.upsert_node(node)
        for e in edges:
            store.upsert_edge(e)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/reconstruct/test_planner.py -v`
Expected: PASS (all planner tests, including the 3 new ones)

- [ ] **Step 6: Type-check and commit**

```bash
uv run ruff format delfos/reconstruct tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add delfos/reconstruct/planners tests/reconstruct
git commit -m "feat(reconstruct): FakeHopPlanner + shared test doubles"
```

Expected: pyright 0 errors.

---

### Task 4: `ReconstructionService.search`

**Files:**
- Create: `delfos/reconstruct/service.py`
- Test: `tests/reconstruct/test_search.py`

**Interfaces:**
- Consumes: `GraphStore` (`delfos.store`), `Embedder` (`delfos.indexer.embedder`), `NodeType`, `CueNode` (`delfos.schema`); `HopPlanner` (Task 1); `FakeEmbedder`, `store`, `load`, builders (Task 3).
- Produces:
  - `TagFilter = tuple[TagCategory, str]`
  - `ReconstructionService(store: GraphStore, embedder: Embedder, planner: HopPlanner, *, seed_k: int = 5)`
  - `ReconstructionService.search(self, query: str, k: int = 5) -> list[CueNode]`

- [ ] **Step 1: Write the failing test**

Create `tests/reconstruct/test_search.py`:

```python
from __future__ import annotations

from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.store.native_store import NativeGraphStore

from .conftest import FakeEmbedder, load, make_cue, vec


def test_search_returns_nearest_cues(store: NativeGraphStore) -> None:
    near = make_cue("cue-near", "auth", embedding=vec(0.10))
    far = make_cue("cue-far", "billing", embedding=vec(9.0))
    load(store, [near, far], [])

    embedder = FakeEmbedder({"how does auth work": vec(0.10)})
    service = ReconstructionService(store, embedder, FakeHopPlanner([]))

    hits = service.search("how does auth work", k=2)

    assert [c.id for c in hits] == ["cue-near", "cue-far"]


def test_search_only_returns_cue_nodes(store: NativeGraphStore) -> None:
    near = make_cue("cue-near", "auth", embedding=vec(0.10))
    load(store, [near], [])
    embedder = FakeEmbedder({"q": vec(0.10)})
    service = ReconstructionService(store, embedder, FakeHopPlanner([]))

    hits = service.search("q", k=5)

    assert all(isinstance(c, type(near)) for c in hits)
    assert hits[0].id == "cue-near"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'delfos.reconstruct.service'`

- [ ] **Step 3: Write the service with search only**

Create `delfos/reconstruct/service.py`:

```python
"""The read-path service: search, traversal, and LLM-driven reconstruction.

Sits entirely on top of :class:`~delfos.store.base.GraphStore`; it never touches
the database directly. The three primitives are pure graph operations;
``reconstruct`` additionally drives a :class:`~delfos.reconstruct.planner.HopPlanner`.
"""

from __future__ import annotations

from delfos.indexer.embedder import Embedder
from delfos.schema import CueNode, NodeType, TagCategory
from delfos.store import GraphStore

from .planner import HopPlanner

TagFilter = tuple[TagCategory, str]


class ReconstructionService:
    """Read-path operations over the Cue-Tag-Content graph."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        planner: HopPlanner,
        *,
        seed_k: int = 5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._planner = planner
        self._seed_k = seed_k

    def search(self, query: str, k: int = 5) -> list[CueNode]:
        """Embed ``query`` and return the ``k`` nearest cue nodes."""
        embedding = self._embedder.embed([query])[0]
        hits = self._store.vector_search(embedding, k, node_type=NodeType.CUE)
        cues: list[CueNode] = []
        for hit in hits:
            node = self._store.get_node(hit.node_id)
            if isinstance(node, CueNode):
                cues.append(node)
        return cues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/reconstruct/test_search.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Type-check and commit**

```bash
uv run ruff format delfos/reconstruct tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add delfos/reconstruct/service.py tests/reconstruct/test_search.py
git commit -m "feat(reconstruct): ReconstructionService.search"
```

Expected: pyright 0 errors.

---

### Task 5: `traverse_forward` + tag helpers

**Files:**
- Modify: `delfos/reconstruct/service.py`
- Test: `tests/reconstruct/test_traverse.py`

**Interfaces:**
- Consumes: `Direction`, `EdgeType`, `ContentNode`, `TagNode`, `NodeStatus`, `Node` (`delfos.schema`).
- Produces:
  - `ReconstructionService.traverse_forward(self, cue_ids: Sequence[str], tag_filters: Sequence[TagFilter] | None = None) -> list[ContentNode]`
  - private helpers `_content_tags(content_id: str) -> set[TagFilter]` and `_resolve_redirect(node: Node) -> Node`

- [ ] **Step 1: Write the failing test**

Create `tests/reconstruct/test_traverse.py`:

```python
from __future__ import annotations

from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType, TagCategory
from delfos.store.native_store import NativeGraphStore

from .conftest import FakeEmbedder, edge, load, make_content, make_cue, make_tag


def _service(store: NativeGraphStore) -> ReconstructionService:
    return ReconstructionService(store, FakeEmbedder({}), FakeHopPlanner([]))


def test_traverse_forward_follows_cue_of(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth")
    content = make_content("content-1", "login")
    load(store, [cue, content], [edge("cue-1", "content-1", EdgeType.CUE_OF)])

    result = _service(store).traverse_forward(["cue-1"])

    assert [c.id for c in result] == ["content-1"]


def test_traverse_forward_filters_by_tag(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth")
    py = make_content("content-py", "login")
    js = make_content("content-js", "logon")
    tag_py = make_tag("tag-py", TagCategory.LANGUAGE, "python")
    edges = [
        edge("cue-1", "content-py", EdgeType.CUE_OF),
        edge("cue-1", "content-js", EdgeType.CUE_OF),
        edge("content-py", "tag-py", EdgeType.TAGGED_WITH),
    ]
    load(store, [cue, py, js, tag_py], edges)

    result = _service(store).traverse_forward(
        ["cue-1"], tag_filters=[(TagCategory.LANGUAGE, "python")]
    )

    assert [c.id for c in result] == ["content-py"]


def test_traverse_forward_dedups_across_cues(store: NativeGraphStore) -> None:
    cues = [make_cue("cue-1", "a"), make_cue("cue-2", "b")]
    content = make_content("content-1", "login")
    edges = [
        edge("cue-1", "content-1", EdgeType.CUE_OF),
        edge("cue-2", "content-1", EdgeType.CUE_OF),
    ]
    load(store, [*cues, content], edges)

    result = _service(store).traverse_forward(["cue-1", "cue-2"])

    assert [c.id for c in result] == ["content-1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_traverse.py -v`
Expected: FAIL with `AttributeError: 'ReconstructionService' object has no attribute 'traverse_forward'`

- [ ] **Step 3: Add imports and the implementation**

In `delfos/reconstruct/service.py`, replace the import block and add methods. Change the imports to:

```python
from __future__ import annotations

from collections.abc import Sequence

from delfos.indexer.embedder import Embedder
from delfos.schema import (
    ContentNode,
    CueNode,
    Direction,
    EdgeType,
    Node,
    NodeStatus,
    NodeType,
    TagCategory,
    TagNode,
)
from delfos.store import GraphStore

from .planner import HopPlanner
```

Then add these methods to `ReconstructionService` (after `search`):

```python
    def traverse_forward(
        self,
        cue_ids: Sequence[str],
        tag_filters: Sequence[TagFilter] | None = None,
    ) -> list[ContentNode]:
        """Expand cues to their ACTIVE content, optionally tag-filtered."""
        wanted = set(tag_filters) if tag_filters else None
        out: list[ContentNode] = []
        seen: set[str] = set()
        for cue_id in cue_ids:
            for neighbor in self._store.neighbors(
                cue_id, edge_type=EdgeType.CUE_OF, direction=Direction.OUTGOING
            ):
                content = self._eligible_content(neighbor, wanted, seen)
                if content is not None:
                    seen.add(content.id)
                    out.append(content)
        return out

    def _eligible_content(
        self, node: Node, wanted: set[TagFilter] | None, seen: set[str]
    ) -> ContentNode | None:
        content = self._resolve_redirect(node)
        if not isinstance(content, ContentNode):
            return None
        if content.status != NodeStatus.ACTIVE:
            return None
        if content.id in seen:
            return None
        if wanted is not None and not wanted <= self._content_tags(content.id):
            return None
        return content

    def _content_tags(self, content_id: str) -> set[TagFilter]:
        tags: set[TagFilter] = set()
        for node in self._store.neighbors(
            content_id, edge_type=EdgeType.TAGGED_WITH, direction=Direction.OUTGOING
        ):
            if isinstance(node, TagNode):
                tags.add((node.category, node.value))
        return tags

    def _resolve_redirect(self, node: Node) -> Node:
        targets = self._store.neighbors(
            node.id, edge_type=EdgeType.REDIRECTS_TO, direction=Direction.OUTGOING
        )
        return targets[0] if targets else node
```

Note: `TagCategory` and `NodeType` remain imported because `TagFilter` and `search` use them; keep all listed imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/reconstruct/test_traverse.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Type-check and commit**

```bash
uv run ruff format delfos/reconstruct tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add delfos/reconstruct/service.py tests/reconstruct/test_traverse.py
git commit -m "feat(reconstruct): traverse_forward with tag filtering"
```

Expected: pyright 0 errors.

---

### Task 6: `traverse_reverse`

**Files:**
- Modify: `delfos/reconstruct/service.py`
- Test: `tests/reconstruct/test_traverse.py` (append)

**Interfaces:**
- Produces: `ReconstructionService.traverse_reverse(self, content_ids: Sequence[str]) -> list[CueNode]`

- [ ] **Step 1: Write the failing test (append to test_traverse.py)**

Append to `tests/reconstruct/test_traverse.py`:

```python
def test_traverse_reverse_finds_sibling_cues(store: NativeGraphStore) -> None:
    content = make_content("content-1", "login")
    cue_a = make_cue("cue-a", "auth")
    cue_b = make_cue("cue-b", "signin")
    edges = [
        edge("cue-a", "content-1", EdgeType.CUE_OF),
        edge("cue-b", "content-1", EdgeType.CUE_OF),
    ]
    load(store, [content, cue_a, cue_b], edges)

    result = _service(store).traverse_reverse(["content-1"])

    assert {c.id for c in result} == {"cue-a", "cue-b"}


def test_traverse_reverse_dedups(store: NativeGraphStore) -> None:
    c1 = make_content("content-1", "login")
    c2 = make_content("content-2", "logout")
    cue = make_cue("cue-a", "auth")
    edges = [
        edge("cue-a", "content-1", EdgeType.CUE_OF),
        edge("cue-a", "content-2", EdgeType.CUE_OF),
    ]
    load(store, [c1, c2, cue], edges)

    result = _service(store).traverse_reverse(["content-1", "content-2"])

    assert [c.id for c in result] == ["cue-a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_traverse.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'traverse_reverse'`

- [ ] **Step 3: Add the implementation**

Add this method to `ReconstructionService` (after `traverse_forward`):

```python
    def traverse_reverse(self, content_ids: Sequence[str]) -> list[CueNode]:
        """Discover sibling cues that point at the given content nodes."""
        out: list[CueNode] = []
        seen: set[str] = set()
        for content_id in content_ids:
            for neighbor in self._store.neighbors(
                content_id, edge_type=EdgeType.CUE_OF, direction=Direction.INCOMING
            ):
                if isinstance(neighbor, CueNode) and neighbor.id not in seen:
                    seen.add(neighbor.id)
                    out.append(neighbor)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/reconstruct/test_traverse.py -v`
Expected: PASS (5 tests total in file)

- [ ] **Step 5: Type-check and commit**

```bash
uv run ruff format delfos/reconstruct tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add delfos/reconstruct/service.py tests/reconstruct/test_traverse.py
git commit -m "feat(reconstruct): traverse_reverse"
```

Expected: pyright 0 errors.

---

### Task 7: `reconstruct` core walk

**Files:**
- Modify: `delfos/reconstruct/service.py`
- Test: `tests/reconstruct/test_reconstruct.py`

**Interfaces:**
- Consumes: `HopRequest`, `HopDecision`, `Collected` (Task 1); `summarize` (Task 2); all helpers from Tasks 4–6.
- Produces:
  - `ReconstructionService.reconstruct(self, query: str, budget: int = 3, tag_filters: Sequence[TagFilter] | None = None) -> list[ContentNode]`
  - private helpers `_candidates_for(node, wanted) -> list[Node]` and `_as_content(node) -> ContentNode | None`

- [ ] **Step 1: Write the failing test**

Create `tests/reconstruct/test_reconstruct.py`:

```python
from __future__ import annotations

from delfos.reconstruct.planner import Collected, HopDecision
from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType
from delfos.store.native_store import NativeGraphStore

from .conftest import FakeEmbedder, edge, load, make_content, make_cue, vec


def _build_two_hop_graph(store: NativeGraphStore) -> None:
    # cue-auth -> content-login -> (sibling) cue-session -> content-token
    seed = make_cue("cue-auth", "auth", embedding=vec(0.1))
    login = make_content("content-login", "login")
    session = make_cue("cue-session", "session")
    token = make_content("content-token", "make_token")
    edges = [
        edge("cue-auth", "content-login", EdgeType.CUE_OF),
        edge("cue-session", "content-login", EdgeType.CUE_OF),
        edge("cue-session", "content-token", EdgeType.CUE_OF),
    ]
    load(store, [seed, login, session, token], edges)


def _service(store: NativeGraphStore, planner: FakeHopPlanner) -> ReconstructionService:
    embedder = FakeEmbedder({"q": vec(0.1)})
    return ReconstructionService(store, embedder, planner, seed_k=5)


def test_reconstruct_collects_from_first_hop(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    planner = FakeHopPlanner(
        [HopDecision(collect=[Collected(id="content-login", relevance=0.9)], stop=True)]
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 1


def test_reconstruct_descends_and_orders_by_relevance(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    planner = FakeHopPlanner(
        [
            # Hop 1 at cue-auth: collect login (0.5), descend into content-login
            HopDecision(
                collect=[Collected(id="content-login", relevance=0.5)],
                descend_into="content-login",
            ),
            # Hop 2 at content-login: sibling cue-session is a candidate; collect it
            # (resolves to its first content) with higher relevance, then stop.
            HopDecision(
                collect=[Collected(id="cue-session", relevance=0.95)], stop=True
            ),
        ]
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    # cue-session resolves to content-login (its first CUE_OF target); already
    # collected, so relevance is upgraded to 0.95. Single deduped result.
    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 2


def test_reconstruct_stops_at_budget(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    # Always descend, never stop: only budget halts the walk.
    planner = FakeHopPlanner(
        [
            HopDecision(collect=[], descend_into="content-login"),
            HopDecision(collect=[], descend_into="cue-session"),
            HopDecision(collect=[], descend_into="content-token"),
            HopDecision(collect=[], descend_into="content-login"),
        ]
    )
    _service(store, planner).reconstruct("q", budget=2)

    assert planner.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_reconstruct.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'reconstruct'`

- [ ] **Step 3: Add imports and the implementation**

In `delfos/reconstruct/service.py`, extend the local imports to add the planner models, summaries, and logging. The `delfos.reconstruct` imports become:

```python
import logging

from .planner import CandidateSummary, HopDecision, HopPlanner, HopRequest
from .summaries import summarize

logger = logging.getLogger(__name__)
```

(Place `import logging` with the stdlib imports at the top, and `logger = ...` after the imports. `CandidateSummary`/`HopDecision`/`HopRequest` join the existing `from .planner import HopPlanner` line.)

Add these methods to `ReconstructionService` (after `traverse_reverse`):

```python
    def reconstruct(
        self,
        query: str,
        budget: int = 3,
        tag_filters: Sequence[TagFilter] | None = None,
    ) -> list[ContentNode]:
        """Reconstruct a relevant content set via LLM-driven depth-first walk.

        Seeds with :meth:`search`, then at each hop asks the planner which
        neighbors to collect and which single one to descend into. ``budget``
        caps the total number of planner calls. Returns content ordered by
        planner-assigned relevance (descending; ties keep discovery order).
        """
        wanted = set(tag_filters) if tag_filters else None
        seeds = self.search(query, k=self._seed_k)
        if not seeds:
            return []

        seed_queue: list[Node] = list(seeds)
        current: Node = seed_queue.pop(0)
        stack: list[Node] = []
        visited: set[str] = {current.id}
        result: dict[str, tuple[ContentNode, float]] = {}
        budget_remaining = budget

        while budget_remaining > 0:
            candidates = self._candidates_for(current, wanted)
            request = HopRequest(
                query=query,
                current=self._to_summary(current),
                candidates=[self._to_summary(c) for c in candidates],
                hops_remaining=budget_remaining,
            )
            try:
                decision = self._planner.decide(request)
            except Exception:
                logger.warning(
                    "hop planner failed; returning partial reconstruction",
                    exc_info=True,
                )
                break
            budget_remaining -= 1

            by_id = {c.id: c for c in candidates}
            self._collect(decision, by_id, result)

            if decision.stop:
                break

            nxt = by_id.get(decision.descend_into) if decision.descend_into else None
            if nxt is not None and nxt.id not in visited:
                stack.append(current)
                current = nxt
                visited.add(nxt.id)
            elif stack:
                current = stack.pop()
            elif seed_queue:
                current = seed_queue.pop(0)
                visited.add(current.id)
            else:
                break

        ordered = sorted(result.values(), key=lambda pair: pair[1], reverse=True)
        return [content for content, _ in ordered]

    def _to_summary(self, node: Node) -> CandidateSummary:
        """Build the planner-facing summary, attaching tags for content."""
        if isinstance(node, ContentNode):
            tags = sorted(f"{cat.value}={val}" for cat, val in self._content_tags(node.id))
            return summarize(node, tags)
        if isinstance(node, CueNode):
            return summarize(node)
        raise TypeError(f"cannot summarize node type: {type(node).__name__}")

    def _collect(
        self,
        decision: HopDecision,
        by_id: dict[str, Node],
        result: dict[str, tuple[ContentNode, float]],
    ) -> None:
        for item in decision.collect:
            node = by_id.get(item.id)
            if node is None:
                continue
            content = self._as_content(node)
            if content is None:
                continue
            existing = result.get(content.id)
            if existing is None or item.relevance > existing[1]:
                result[content.id] = (content, item.relevance)

    def _candidates_for(self, node: Node, wanted: set[TagFilter] | None) -> list[Node]:
        if isinstance(node, CueNode):
            return self._content_candidates(node.id, wanted)
        if isinstance(node, ContentNode):
            out: list[Node] = []
            for cue in self._store.neighbors(
                node.id, edge_type=EdgeType.CUE_OF, direction=Direction.INCOMING
            ):
                if isinstance(cue, CueNode) and cue.status == NodeStatus.ACTIVE:
                    out.append(cue)
            for peer in self._content_candidates_via(
                node.id, EdgeType.PART_OF_TOPIC, wanted
            ):
                out.append(peer)
            return out
        return []

    def _content_candidates(
        self, node_id: str, wanted: set[TagFilter] | None
    ) -> list[Node]:
        return self._content_candidates_via(node_id, EdgeType.CUE_OF, wanted)

    def _content_candidates_via(
        self, node_id: str, edge_type: EdgeType, wanted: set[TagFilter] | None
    ) -> list[Node]:
        out: list[Node] = []
        seen: set[str] = set()
        for neighbor in self._store.neighbors(
            node_id, edge_type=edge_type, direction=Direction.OUTGOING
        ):
            content = self._eligible_content(neighbor, wanted, seen)
            if content is not None:
                seen.add(content.id)
                out.append(content)
        return out

    def _as_content(self, node: Node) -> ContentNode | None:
        if isinstance(node, ContentNode):
            return node
        if isinstance(node, CueNode):
            for neighbor in self._store.neighbors(
                node.id, edge_type=EdgeType.CUE_OF, direction=Direction.OUTGOING
            ):
                content = self._resolve_redirect(neighbor)
                if isinstance(content, ContentNode) and content.status == NodeStatus.ACTIVE:
                    return content
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/reconstruct/test_reconstruct.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Type-check and commit**

```bash
uv run ruff format delfos/reconstruct tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add delfos/reconstruct/service.py tests/reconstruct/test_reconstruct.py
git commit -m "feat(reconstruct): LLM-driven depth-first reconstruct walk"
```

Expected: pyright 0 errors.

---

### Task 8: `reconstruct` edge cases (robustness)

**Files:**
- Test: `tests/reconstruct/test_reconstruct.py` (append)

No production code changes are expected — Task 7's implementation already covers these paths. This task is the reviewer gate that proves it. If a test fails, fix `service.py` minimally to satisfy it.

**Interfaces:**
- Consumes: everything from Task 7.

- [ ] **Step 1: Write the failing tests (append to test_reconstruct.py)**

Append to `tests/reconstruct/test_reconstruct.py`:

```python
def test_reconstruct_empty_when_no_seed_cues(store: NativeGraphStore) -> None:
    # No cue carries an embedding, so vector_search returns nothing.
    load(store, [make_content("content-1", "login")], [])
    planner = FakeHopPlanner([])
    result = _service(store, planner).reconstruct("q", budget=3)

    assert result == []
    assert planner.call_count == 0


def test_reconstruct_ignores_hallucinated_ids(store: NativeGraphStore) -> None:
    # Two embedded seeds: when hop 1's descend_into is invalid and the stack is
    # empty, the walk falls back to the second seed and keeps going.
    seed1 = make_cue("cue-auth", "auth", embedding=vec(0.1))
    seed2 = make_cue("cue-extra", "extra", embedding=vec(0.11))
    login = make_content("content-login", "login")
    edges = [
        edge("cue-auth", "content-login", EdgeType.CUE_OF),
        edge("cue-extra", "content-login", EdgeType.CUE_OF),
    ]
    load(store, [seed1, seed2, login], edges)

    planner = FakeHopPlanner(
        [
            HopDecision(
                collect=[Collected(id="does-not-exist", relevance=0.9)],
                descend_into="also-fake",
                stop=False,
            ),
            HopDecision(collect=[Collected(id="content-login", relevance=0.4)], stop=True),
        ]
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    # Hallucinated collect dropped; invalid descend_into forced a fallback to the
    # second seed, where a real collect succeeded.
    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 2


def test_reconstruct_returns_partial_on_planner_error(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    planner = FakeHopPlanner(
        [HopDecision(collect=[Collected(id="content-login", relevance=0.7)],
                     descend_into="content-login")],
        error_after=1,  # 2nd call raises
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    # First hop collected before the second call blew up.
    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 2
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/reconstruct/test_reconstruct.py -v`
Expected: PASS (6 tests total). If `test_reconstruct_returns_partial_on_planner_error` or the hallucinated-id test fails, the bug is in `reconstruct`'s backtrack/seed-fallback or its try/except placement — fix `service.py` so collected results survive a later planner exception and an invalid `descend_into` falls through to the seed queue.

- [ ] **Step 3: Type-check and commit**

```bash
uv run ruff format tests/reconstruct
uv run ruff check delfos/reconstruct tests/reconstruct
uv run pyright
git add tests/reconstruct/test_reconstruct.py delfos/reconstruct/service.py
git commit -m "test(reconstruct): budget, hallucinated ids, partial-on-error"
```

Expected: pyright 0 errors.

---

### Task 9: Package exports and full-suite verification

**Files:**
- Modify: `delfos/reconstruct/__init__.py`
- Test: `tests/reconstruct/test_exports.py`

**Interfaces:**
- Produces: `from delfos.reconstruct import ReconstructionService, TagFilter`

- [ ] **Step 1: Write the failing test**

Create `tests/reconstruct/test_exports.py`:

```python
from __future__ import annotations


def test_public_exports_importable() -> None:
    from delfos.reconstruct import ReconstructionService, TagFilter

    assert ReconstructionService.__name__ == "ReconstructionService"
    # TagFilter is a tuple type alias; just confirm it is importable and usable.
    _: TagFilter = ("language", "python")  # type: ignore[assignment]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_exports.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReconstructionService' from 'delfos.reconstruct'`

- [ ] **Step 3: Write the exports**

Replace `delfos/reconstruct/__init__.py` with:

```python
"""Delfos read path: search, traversal, and LLM-driven reconstruction."""

from .planner import (
    CandidateSummary,
    Collected,
    HopDecision,
    HopPlanner,
    HopRequest,
)
from .service import ReconstructionService, TagFilter

__all__ = [
    "CandidateSummary",
    "Collected",
    "HopDecision",
    "HopPlanner",
    "HopRequest",
    "ReconstructionService",
    "TagFilter",
]
```

- [ ] **Step 4: Run the full suite and type-check**

Run: `uv run pytest -v`
Expected: PASS (all store tests + all reconstruct tests).

Run: `uv run pyright`
Expected: 0 errors, strict mode.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add delfos/reconstruct/__init__.py tests/reconstruct/test_exports.py
git commit -m "feat(reconstruct): public exports + full-suite green"
```

---

## Notes for the implementer

- **Why content↔cue alternation:** from a cue you go *out* along `CUE_OF` to content; from content you go *in* along `CUE_OF` to find sibling cues, and *out* along `PART_OF_TOPIC` to topic peers. This is why `_candidates_for` branches on node type.
- **Relevance tie-break:** `sorted` is stable and `result` is a dict (insertion-ordered), so equal-relevance content keeps discovery order. This is the documented v1 behavior; the spec's "reaching-cue similarity" tie-break is deliberately deferred to avoid threading similarity scores through the walk.
- **Collecting a cue:** the planner may collect a cue candidate; `_as_content` resolves it to that cue's first ACTIVE `CUE_OF` content. A cue with no content resolves to nothing and is skipped.
- **`REDIRECTS_TO`:** the indexer emits none in v1, but `_resolve_redirect` follows one transparently if present, so the read path is forward-compatible.

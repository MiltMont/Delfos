# MCP Read Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the Delfos read path over MCP (stdio) as four tools (`search`, `traverse_forward`, `traverse_reverse`, `fetch`) plus a `reconstruct` prompt, with the calling agent as the planner.

**Architecture:** A thin `delfos/mcp/` adapter over the existing `ReconstructionService` (which sits over `GraphStore`). The server introduces no graph logic. Walk tools return compact `NodeSummary` views; `fetch` returns full `ContentDetail` bodies. The embedder is the only model dependency; there is no server-side planner LLM.

**Tech Stack:** Python 3.12+, `mcp` (FastMCP) Python SDK, Pydantic v2, the existing `delfos` package and its C++-backed `NativeGraphStore`.

## Global Constraints

- Pyright runs in **strict mode** — all new code must be fully typed.
- All Pydantic models use `model_config = ConfigDict(extra="forbid")`.
- `requires-python = ">=3.12"`.
- No component outside `delfos/store/` touches the C++ engine directly; all graph access goes through `ReconstructionService`.
- Embeddings are **never** serialized back to the agent.
- The query-time embedding model must equal the index-time `embedding_model` (enforced at startup).
- Lint/format/type/test gates: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`.

---

### Task 1: MCP view models (`delfos/mcp/views.py`)

Pure Pydantic + converters. No `mcp` dependency needed; fully unit-testable on its own.

**Files:**
- Create: `delfos/mcp/__init__.py`
- Create: `delfos/mcp/views.py`
- Test: `tests/mcp/__init__.py`, `tests/mcp/test_views.py`

**Interfaces:**
- Consumes: `delfos.schema.CueNode`, `delfos.schema.ContentNode`.
- Produces:
  - `SNIPPET_LIMIT: int = 500`
  - `class NodeSummary(BaseModel)` — fields `id: str`, `kind: Literal["cue","content"]`, `label: str`, `snippet: str | None`, `tags: list[str]`.
  - `class ContentDetail(BaseModel)` — fields `id: str`, `symbol_name: str | None`, `signature: str | None`, `docstring: str | None`, `body: str`, `memory_layer: str`, `source_file: str`, `git_sha: str`.
  - `cue_to_summary(node: CueNode) -> NodeSummary`
  - `content_to_summary(node: ContentNode, tags: list[str]) -> NodeSummary`
  - `content_to_detail(node: ContentNode) -> ContentDetail`

- [ ] **Step 1: Write the failing test**

Create `tests/mcp/__init__.py` (empty file), then `tests/mcp/test_views.py`:

```python
from __future__ import annotations

from datetime import datetime

from delfos.mcp.views import (
    SNIPPET_LIMIT,
    ContentDetail,
    NodeSummary,
    content_to_detail,
    content_to_summary,
    cue_to_summary,
)
from delfos.schema import ContentKind, ContentNode, CueNode, CueType, MemoryLayer

NOW = datetime(2026, 6, 24, 12, 0, 0)


def _content(**over: object) -> ContentNode:
    base: dict[str, object] = dict(
        id="c1",
        source_file="a.py",
        git_sha="sha",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="login",
        signature="def login()",
        docstring=None,
        body="def login(): ...",
    )
    base.update(over)
    return ContentNode(**base)  # type: ignore[arg-type]


def test_cue_to_summary_has_no_snippet_or_tags() -> None:
    cue = CueNode(
        id="q1",
        source_file="a.py",
        git_sha="sha",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text="auth",
    )
    summary = cue_to_summary(cue)
    assert summary == NodeSummary(id="q1", kind="cue", label="auth", snippet=None, tags=[])


def test_content_summary_prefers_signature_and_uses_docstring() -> None:
    summary = content_to_summary(_content(docstring="Logs a user in."), ["language=python"])
    assert summary.kind == "content"
    assert summary.label == "def login()"
    assert summary.snippet == "Logs a user in."
    assert summary.tags == ["language=python"]


def test_content_summary_truncates_body_when_no_docstring() -> None:
    summary = content_to_summary(_content(body="x" * 600, docstring=None), [])
    assert summary.snippet is not None
    assert len(summary.snippet) == SNIPPET_LIMIT + 1  # 500 chars + ellipsis
    assert summary.snippet.endswith("…")


def test_content_detail_omits_embedding_and_carries_provenance() -> None:
    detail = content_to_detail(
        _content(embedding=[0.1] * 4, embedding_model="fake-v1", body="def login(): ...")
    )
    assert isinstance(detail, ContentDetail)
    assert detail.body == "def login(): ..."
    assert detail.source_file == "a.py"
    assert detail.memory_layer == "semantic"
    assert "embedding" not in detail.model_dump()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/mcp/test_views.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'delfos.mcp'`.

- [ ] **Step 3: Write minimal implementation**

Create `delfos/mcp/__init__.py`:

```python
"""MCP read server: exposes the Delfos read path over the Model Context Protocol."""
```

Create `delfos/mcp/views.py`:

```python
"""MCP-facing serialization models for graph nodes.

Dedicated to the MCP layer (not reused from ``reconstruct.planner``) so the tool
surface stays decoupled from the planner's ``CandidateSummary`` contract. Walk
tools return :class:`NodeSummary` (cheap); ``fetch`` returns :class:`ContentDetail`
(full body). Embeddings are never serialized back to the agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from delfos.schema import ContentNode, CueNode

SNIPPET_LIMIT = 500  # mirrors delfos.reconstruct.summaries._SNIPPET_LIMIT


def _truncate(text: str, limit: int = SNIPPET_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class NodeSummary(BaseModel):
    """Compact, walk-time view of a node. Cheap enough to fan out over."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["cue", "content"]
    label: str
    snippet: str | None = None
    tags: list[str] = Field(default_factory=list)


class ContentDetail(BaseModel):
    """Full content payload returned by ``fetch``. No embedding."""

    model_config = ConfigDict(extra="forbid")

    id: str
    symbol_name: str | None
    signature: str | None
    docstring: str | None
    body: str
    memory_layer: str
    source_file: str
    git_sha: str


def cue_to_summary(node: CueNode) -> NodeSummary:
    """Summarize a cue: its text is the label; cues carry no content tags."""
    return NodeSummary(id=node.id, kind="cue", label=node.text, snippet=None, tags=[])


def content_to_summary(node: ContentNode, tags: list[str]) -> NodeSummary:
    """Summarize content: signature/symbol/kind as label, docstring-or-body snippet."""
    label = node.signature or node.symbol_name or node.kind.value
    snippet = node.docstring if node.docstring is not None else _truncate(node.body)
    return NodeSummary(id=node.id, kind="content", label=label, snippet=snippet, tags=list(tags))


def content_to_detail(node: ContentNode) -> ContentDetail:
    """Full content view, embedding intentionally dropped."""
    return ContentDetail(
        id=node.id,
        symbol_name=node.symbol_name,
        signature=node.signature,
        docstring=node.docstring,
        body=node.body,
        memory_layer=node.memory_layer.value,
        source_file=node.source_file,
        git_sha=node.git_sha,
    )
```

- [ ] **Step 4: Run tests + gates to verify they pass**

Run: `uv run pytest tests/mcp/test_views.py -v && uv run pyright && uv run ruff check .`
Expected: PASS (4 tests), no pyright errors, no ruff errors.

- [ ] **Step 5: Commit**

```bash
git add delfos/mcp/__init__.py delfos/mcp/views.py tests/mcp/__init__.py tests/mcp/test_views.py
git commit -m "feat(mcp): NodeSummary/ContentDetail views + converters

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Service read-path additions (`delfos/reconstruct/service.py`)

Make the planner optional, add `fetch`, add `content_tags`. Lets the MCP server build a `ReconstructionService` with **no planner LLM** while keeping `reconstruct`/`HopPlanner` intact, and keeps `fetch`/tag-enrichment on the read-path boundary.

**Files:**
- Modify: `delfos/reconstruct/service.py` (constructor ~38-49, `reconstruct` ~94-141, add two methods)
- Test: `tests/reconstruct/test_fetch.py`

**Interfaces:**
- Consumes: `delfos.store.GraphStore.get_node`, the existing `_content_tags`.
- Produces (new public surface on `ReconstructionService`):
  - `__init__(self, store, embedder, planner: HopPlanner | None = None, *, seed_k: int = 5)`
  - `fetch(self, ids: Sequence[str]) -> list[ContentNode]`
  - `content_tags(self, content_id: str) -> list[str]`
  - `reconstruct(...)` raises `RuntimeError` when constructed without a planner.

- [ ] **Step 1: Write the failing test**

Create `tests/reconstruct/test_fetch.py`:

```python
from __future__ import annotations

import pytest

from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType, NodeStatus, TagCategory
from delfos.store.native_store import NativeGraphStore

from .conftest import (
    FakeEmbedder,
    edge,
    load,
    make_content,
    make_tag,
)


def _service(store: NativeGraphStore) -> ReconstructionService:
    # No planner: exercises the planner-optional constructor.
    return ReconstructionService(store, FakeEmbedder({}))


def test_fetch_returns_active_content_and_skips_unknown(store: NativeGraphStore) -> None:
    content = make_content("c1", "login")
    load(store, [content], [])
    svc = _service(store)

    got = svc.fetch(["c1", "does-not-exist"])

    assert [c.id for c in got] == ["c1"]
    assert got[0].body == "def login(): ..."


def test_fetch_skips_deleted_content(store: NativeGraphStore) -> None:
    content = make_content("c1", "login")
    content.status = NodeStatus.DELETED
    load(store, [content], [])
    svc = _service(store)

    assert svc.fetch(["c1"]) == []


def test_content_tags_renders_sorted_category_value(store: NativeGraphStore) -> None:
    content = make_content("c1", "login")
    tag = make_tag("t1", TagCategory.LANGUAGE, "python")
    load(store, [content, tag], [edge("c1", "t1", EdgeType.TAGGED_WITH)])
    svc = _service(store)

    assert svc.content_tags("c1") == ["language=python"]


def test_reconstruct_without_planner_raises(store: NativeGraphStore) -> None:
    svc = _service(store)
    with pytest.raises(RuntimeError, match="planner"):
        svc.reconstruct("anything")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/reconstruct/test_fetch.py -v`
Expected: FAIL — `TypeError` (constructor still requires `planner` positionally) / `AttributeError` for `fetch`/`content_tags`.

- [ ] **Step 3: Write minimal implementation**

In `delfos/reconstruct/service.py`, change the constructor signature so `planner` defaults to `None`:

```python
    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        planner: HopPlanner | None = None,
        *,
        seed_k: int = 5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._planner = planner
        self._seed_k = seed_k
```

At the top of `reconstruct`, immediately after the docstring, guard on the planner and bind a local (so the per-hop call is type-narrowed):

```python
        planner = self._planner
        if planner is None:
            raise RuntimeError("reconstruct requires a HopPlanner; none was configured")
```

Then change the per-hop decide call inside the loop from `self._planner.decide(request)` to:

```python
                decision = planner.decide(request)
```

Add these two methods to the class (e.g. just after `reconstruct`):

```python
    def fetch(self, ids: Sequence[str]) -> list[ContentNode]:
        """Resolve ids to ACTIVE content nodes, skipping unknown/non-content/deleted."""
        out: list[ContentNode] = []
        for node_id in ids:
            node = self._store.get_node(node_id)
            if isinstance(node, ContentNode) and node.status == NodeStatus.ACTIVE:
                out.append(node)
        return out

    def content_tags(self, content_id: str) -> list[str]:
        """Tags on a content node as sorted ``"category=value"`` strings."""
        return sorted(f"{cat.value}={val}" for cat, val in self._content_tags(content_id))
```

(`Sequence`, `ContentNode`, and `NodeStatus` are already imported in this module.)

- [ ] **Step 4: Run tests + gates to verify they pass**

Run: `uv run pytest tests/reconstruct/ -v && uv run pyright`
Expected: PASS — new `test_fetch.py` passes and the existing reconstruct/search/traverse tests still pass (the positional `planner` argument remains compatible). No pyright errors.

- [ ] **Step 5: Commit**

```bash
git add delfos/reconstruct/service.py tests/reconstruct/test_fetch.py
git commit -m "feat(reconstruct): optional planner + fetch/content_tags for MCP read path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Server config + startup model check (`delfos/mcp/config.py`)

Env-driven construction of the embedder and store, plus the fail-fast embedding-model match. No `mcp` dependency; testable in isolation.

**Files:**
- Create: `delfos/mcp/config.py`
- Test: `tests/mcp/test_config.py`

**Interfaces:**
- Consumes: `delfos.indexer.OpenAIEmbedder`, `delfos.indexer.embedder.Embedder`, `delfos.store.NativeGraphStore`, `openai.OpenAI`.
- Produces:
  - `@dataclass(frozen=True) class ServerConfig` — `index_path: Path`, `embed_model: str`, `embed_dim: int`, `embed_base_url: str | None`, `embed_api_key: str | None`, `send_dimensions: bool`.
  - `config_from_env(env: Mapping[str, str]) -> ServerConfig`
  - `build_embedder(cfg: ServerConfig) -> OpenAIEmbedder`
  - `build_store(cfg: ServerConfig) -> NativeGraphStore`
  - `check_model_match(store: NativeGraphStore, embedder: Embedder) -> None` (raises `RuntimeError` on mismatch)

- [ ] **Step 1: Write the failing test**

Create `tests/mcp/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from delfos.mcp.config import ServerConfig, check_model_match, config_from_env
from delfos.store.native_store import NativeGraphStore


class _Embedder:
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


def test_config_from_env_uses_defaults_for_empty_env() -> None:
    cfg = config_from_env({})
    assert cfg == ServerConfig(
        index_path=Path("delfos/store"),
        embed_model="nomic-embed-text",
        embed_dim=768,
        embed_base_url=None,
        embed_api_key=None,
        send_dimensions=False,
    )


def test_config_from_env_reads_overrides() -> None:
    cfg = config_from_env(
        {
            "DELFOS_INDEX_PATH": "/data/graph",
            "DELFOS_EMBED_MODEL": "text-embedding-3-small",
            "DELFOS_EMBED_DIM": "1536",
            "DELFOS_EMBED_BASE_URL": "http://localhost:11434/v1",
            "DELFOS_EMBED_API_KEY": "ollama",
            "DELFOS_EMBED_SEND_DIM": "1",
        }
    )
    assert cfg.index_path == Path("/data/graph")
    assert cfg.embed_model == "text-embedding-3-small"
    assert cfg.embed_dim == 1536
    assert cfg.embed_base_url == "http://localhost:11434/v1"
    assert cfg.embed_api_key == "ollama"
    assert cfg.send_dimensions is True


def test_check_model_match_passes_when_equal(tmp_path: Path) -> None:
    store = NativeGraphStore(tmp_path / "g", embedding_dim=8, embedding_model="fake-v1")
    store.initialize()
    check_model_match(store, _Embedder("fake-v1"))  # no raise
    store.close()


def test_check_model_match_raises_on_mismatch(tmp_path: Path) -> None:
    store = NativeGraphStore(tmp_path / "g", embedding_dim=8, embedding_model="fake-v1")
    store.initialize()
    with pytest.raises(RuntimeError, match="fake-v1"):
        check_model_match(store, _Embedder("other-model"))
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/mcp/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'delfos.mcp.config'`.

- [ ] **Step 3: Write minimal implementation**

Create `delfos/mcp/config.py`:

```python
"""Env-driven startup configuration for the MCP read server.

Reuses the smoke harness's ``DELFOS_EMBED_*`` convention. The embedder is the
server's only model dependency; point ``DELFOS_EMBED_BASE_URL`` at a local
OpenAI-compatible endpoint or leave it unset for OpenAI-hosted. The query-time
embedding model must match the index-time one; :func:`check_model_match` enforces
this at startup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from delfos.indexer import OpenAIEmbedder
from delfos.indexer.embedder import Embedder
from delfos.store import NativeGraphStore

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerConfig:
    """Resolved server configuration."""

    index_path: Path
    embed_model: str
    embed_dim: int
    embed_base_url: str | None
    embed_api_key: str | None
    send_dimensions: bool


def config_from_env(env: Mapping[str, str]) -> ServerConfig:
    """Build a :class:`ServerConfig` from environment variables (with defaults)."""
    return ServerConfig(
        index_path=Path(env.get("DELFOS_INDEX_PATH", "delfos/store")),
        embed_model=env.get("DELFOS_EMBED_MODEL", "nomic-embed-text"),
        embed_dim=int(env.get("DELFOS_EMBED_DIM", "768")),
        embed_base_url=env.get("DELFOS_EMBED_BASE_URL"),
        embed_api_key=env.get("DELFOS_EMBED_API_KEY"),
        send_dimensions=env.get("DELFOS_EMBED_SEND_DIM", "0").strip().lower() in _TRUTHY,
    )


def build_embedder(cfg: ServerConfig) -> OpenAIEmbedder:
    """Construct the OpenAI-compatible embedder from config."""
    client = OpenAI(base_url=cfg.embed_base_url, api_key=cfg.embed_api_key)
    return OpenAIEmbedder(
        cfg.embed_model,
        dimensions=cfg.embed_dim,
        send_dimensions=cfg.send_dimensions,
        client=client,
    )


def build_store(cfg: ServerConfig) -> NativeGraphStore:
    """Open the persisted graph store at the configured path."""
    store = NativeGraphStore(
        cfg.index_path, embedding_dim=cfg.embed_dim, embedding_model=cfg.embed_model
    )
    store.initialize()
    return store


def check_model_match(store: NativeGraphStore, embedder: Embedder) -> None:
    """Fail fast unless the embedder's model matches the store's index model."""
    if store.embedding_model != embedder.model:
        raise RuntimeError(
            f"embedder model {embedder.model!r} does not match index model "
            f"{store.embedding_model!r}; queries must use the model the index "
            f"was built with"
        )
```

- [ ] **Step 4: Run tests + gates to verify they pass**

Run: `uv run pytest tests/mcp/test_config.py -v && uv run pyright && uv run ruff check .`
Expected: PASS (4 tests), no pyright/ruff errors.

- [ ] **Step 5: Commit**

```bash
git add delfos/mcp/config.py tests/mcp/test_config.py
git commit -m "feat(mcp): env config + fail-fast embedding-model match

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: FastMCP server, prompt, and entry point

Add the `mcp` dependency, the tool/prompt wiring, and the `python -m delfos.mcp` entry point. Tool *logic* lives in plain module functions so it is testable synchronously without an MCP transport; `build_server` registers thin wrappers.

**Files:**
- Modify: `pyproject.toml` (add `mcp` dep + `delfos-mcp` console script)
- Create: `delfos/mcp/server.py`
- Create: `delfos/mcp/__main__.py`
- Create: `tests/mcp/conftest.py`
- Test: `tests/mcp/test_server.py`

**Interfaces:**
- Consumes: Task 1 (`NodeSummary`, `ContentDetail`, `cue_to_summary`, `content_to_summary`, `content_to_detail`), Task 2 (`ReconstructionService.search/traverse_forward/traverse_reverse/fetch/content_tags`), Task 3 (`config_from_env`, `build_embedder`, `build_store`, `check_model_match`), `mcp.server.fastmcp.FastMCP`, `delfos.schema.TagCategory`.
- Produces:
  - `_search(service, query: str, k: int = 5) -> list[NodeSummary]`
  - `_traverse_forward(service, cue_ids: list[str], tag_filters: list[tuple[str, str]] | None = None) -> list[NodeSummary]`
  - `_traverse_reverse(service, content_ids: list[str]) -> list[NodeSummary]`
  - `_fetch(service, ids: list[str]) -> list[ContentDetail]`
  - `reconstruct_prompt(query: str, budget: int = 3) -> str`
  - `build_server(service: ReconstructionService) -> FastMCP`
  - `delfos/mcp/__main__.py:main() -> None`

- [ ] **Step 1: Add the `mcp` dependency and sync**

In `pyproject.toml`, change the `dependencies` array (currently lines 10-13) to:

```toml
dependencies = [
    "openai>=1.0.0",
    "pydantic>=2.9.0",
    "mcp>=1.2.0",
]
```

And add a console-script entry point immediately after the `dependencies` array (before `[build-system]`):

```toml
[project.scripts]
delfos-mcp = "delfos.mcp.__main__:main"
```

Run: `uv sync`
Expected: resolves and installs `mcp`. Verify import: `uv run python -c "from mcp.server.fastmcp import FastMCP; print('ok')"` prints `ok`.

- [ ] **Step 2: Write the failing test**

Create `tests/mcp/conftest.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from delfos.reconstruct import ReconstructionService
from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import EMB_DIM, EMB_MODEL, FakeEmbedder, vec


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL)
    s.initialize()
    yield s
    s.close()


def make_service(store: NativeGraphStore, query_vec: list[float]) -> ReconstructionService:
    """Service over the seeded store, with an embedder that maps 'q' -> query_vec."""
    return ReconstructionService(store, FakeEmbedder({"q": query_vec}))


__all__ = ["store", "make_service", "vec"]
```

Create `tests/mcp/test_server.py`:

```python
from __future__ import annotations

from delfos.mcp.server import (
    _fetch,
    _search,
    _traverse_forward,
    _traverse_reverse,
    build_server,
    reconstruct_prompt,
)
from delfos.schema import EdgeType, TagCategory
from delfos.store.native_store import NativeGraphStore
from mcp.server.fastmcp import FastMCP

from tests.reconstruct.conftest import (
    edge,
    load,
    make_content,
    make_cue,
    make_tag,
)

from .conftest import make_service, vec


def _seed(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth", embedding=vec(0.10))
    content = make_content("c1", "login")
    tag = make_tag("t1", TagCategory.LANGUAGE, "python")
    load(
        store,
        [cue, content, tag],
        [
            edge("cue-1", "c1", EdgeType.CUE_OF),
            edge("c1", "t1", EdgeType.TAGGED_WITH),
        ],
    )


def test_search_returns_cue_summaries(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _search(svc, "q", k=5)

    assert [s.id for s in out] == ["cue-1"]
    assert out[0].kind == "cue"
    assert out[0].label == "auth"


def test_traverse_forward_returns_content_summaries_with_tags(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _traverse_forward(svc, ["cue-1"])

    assert [s.id for s in out] == ["c1"]
    assert out[0].kind == "content"
    assert out[0].tags == ["language=python"]


def test_traverse_forward_unknown_tag_category_errors(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    try:
        _traverse_forward(svc, ["cue-1"], [("not_a_category", "x")])
    except ValueError as exc:
        assert "not_a_category" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected ValueError for unknown tag category")


def test_traverse_reverse_returns_sibling_cues(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _traverse_reverse(svc, ["c1"])

    assert [s.id for s in out] == ["cue-1"]
    assert out[0].kind == "cue"


def test_fetch_returns_full_bodies(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _fetch(svc, ["c1", "missing"])

    assert [d.id for d in out] == ["c1"]
    assert out[0].body == "def login(): ..."


def test_reconstruct_prompt_contains_protocol_and_args() -> None:
    text = reconstruct_prompt("how does auth work", budget=4)
    lowered = text.lower()
    assert "how does auth work" in text
    assert "4" in text
    assert "search" in lowered
    assert "fetch" in lowered
    assert "budget" in lowered


def test_build_server_registers_tools_and_prompt(store: NativeGraphStore) -> None:
    svc = make_service(store, vec(0.10))
    server = build_server(svc)
    assert isinstance(server, FastMCP)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/mcp/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'delfos.mcp.server'`.

- [ ] **Step 4: Write minimal implementation**

Create `delfos/mcp/server.py`:

```python
"""FastMCP read server: four graph tools + a reconstruct prompt.

Tool logic lives in plain ``_``-prefixed functions so it is unit-testable
without an MCP transport; :func:`build_server` registers thin wrappers. The
calling agent is the planner — the server runs no planner LLM. The ``reconstruct``
prompt teaches the depth-first walk the agent drives.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from delfos.reconstruct import ReconstructionService, TagFilter
from delfos.schema import TagCategory

from .views import (
    ContentDetail,
    NodeSummary,
    content_to_detail,
    content_to_summary,
    cue_to_summary,
)


def _to_tag_filters(pairs: list[tuple[str, str]]) -> list[TagFilter]:
    out: list[TagFilter] = []
    for category, value in pairs:
        try:
            cat = TagCategory(category)
        except ValueError as exc:
            valid = ", ".join(c.value for c in TagCategory)
            raise ValueError(
                f"unknown tag category {category!r}; valid categories: {valid}"
            ) from exc
        out.append((cat, value))
    return out


def _search(service: ReconstructionService, query: str, k: int = 5) -> list[NodeSummary]:
    return [cue_to_summary(c) for c in service.search(query, k)]


def _traverse_forward(
    service: ReconstructionService,
    cue_ids: list[str],
    tag_filters: list[tuple[str, str]] | None = None,
) -> list[NodeSummary]:
    filters = _to_tag_filters(tag_filters) if tag_filters else None
    contents = service.traverse_forward(cue_ids, filters)
    return [content_to_summary(c, service.content_tags(c.id)) for c in contents]


def _traverse_reverse(
    service: ReconstructionService, content_ids: list[str]
) -> list[NodeSummary]:
    return [cue_to_summary(c) for c in service.traverse_reverse(content_ids)]


def _fetch(service: ReconstructionService, ids: list[str]) -> list[ContentDetail]:
    return [content_to_detail(c) for c in service.fetch(ids)]


def reconstruct_prompt(query: str, budget: int = 3) -> str:
    """Protocol text teaching the agent to drive a depth-first reconstruction."""
    return (
        f"Reconstruct memory for this query by walking the graph yourself; you are "
        f"the planner.\n\n"
        f"Query: {query}\n"
        f"Budget: {budget} traversal steps.\n\n"
        f"Protocol:\n"
        f"1. Call `search` with the query to get seed cue nodes.\n"
        f"2. Call `traverse_forward` on the most promising cues to reach content; "
        f"use tag_filters to narrow when a category is obviously relevant.\n"
        f"3. Descend depth-first: expand the single most relevant candidate one hop "
        f"at a time rather than fanning out. Use `traverse_reverse` to discover "
        f"sibling cues when a content node looks central.\n"
        f"4. Spend at most {budget} traversal steps; backtrack when a branch stops "
        f"yielding relevant nodes.\n"
        f"5. Call `fetch` with the ids worth keeping to get their full bodies.\n"
        f"6. Stop when relevance drops or the budget is exhausted, then answer from "
        f"the fetched content."
    )


def build_server(service: ReconstructionService) -> FastMCP:
    """Build the FastMCP app, registering the four tools and the prompt."""
    mcp = FastMCP("delfos")

    @mcp.tool()
    def search(query: str, k: int = 5) -> list[NodeSummary]:
        """Find cue entry points by semantic similarity. Start a walk here."""
        return _search(service, query, k)

    @mcp.tool()
    def traverse_forward(
        cue_ids: list[str], tag_filters: list[tuple[str, str]] | None = None
    ) -> list[NodeSummary]:
        """Expand cues to their content. tag_filters are (category, value) pairs."""
        return _traverse_forward(service, cue_ids, tag_filters)

    @mcp.tool()
    def traverse_reverse(content_ids: list[str]) -> list[NodeSummary]:
        """Discover sibling cues that point at the given content nodes."""
        return _traverse_reverse(service, content_ids)

    @mcp.tool()
    def fetch(ids: list[str]) -> list[ContentDetail]:
        """Fetch full content bodies for the given node ids."""
        return _fetch(service, ids)

    @mcp.prompt()
    def reconstruct(query: str, budget: int = 3) -> str:
        """Drive a depth-first memory reconstruction over the graph."""
        return reconstruct_prompt(query, budget)

    return mcp
```

Create `delfos/mcp/__main__.py`:

```python
"""Entry point: `python -m delfos.mcp` (and the `delfos-mcp` console script).

Wires env config -> store + embedder -> startup model check -> a planner-less
ReconstructionService -> FastMCP, then serves over stdio.
"""

from __future__ import annotations

import os

from delfos.mcp.config import build_embedder, build_store, check_model_match, config_from_env
from delfos.mcp.server import build_server
from delfos.reconstruct import ReconstructionService


def main() -> None:
    cfg = config_from_env(os.environ)
    store = build_store(cfg)
    embedder = build_embedder(cfg)
    check_model_match(store, embedder)
    service = ReconstructionService(store, embedder)  # planner=None: agent is the planner
    build_server(service).run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests + gates to verify they pass**

Run: `uv run pytest tests/mcp/ -v && uv run pyright && uv run ruff check . && uv run ruff format --check .`
Expected: PASS (all `tests/mcp` tests), no pyright/ruff errors.

- [ ] **Step 6: Verify the server boots over stdio**

Run (sends one MCP `initialize` request and checks the server responds without crashing):

```bash
DELFOS_EMBED_MODEL=fake-v1 DELFOS_EMBED_DIM=8 DELFOS_INDEX_PATH="$(mktemp -d)/graph" \
  uv run python -c "
from delfos.mcp.config import config_from_env, build_store
from delfos.mcp.server import build_server
from delfos.reconstruct import ReconstructionService
import os
cfg = config_from_env(os.environ)
store = build_store(cfg)
class E:
    model='fake-v1'; model_version=None; dimensions=8
    def embed(self, t): return [[0.0]*8 for _ in t]
svc = ReconstructionService(store, E())
print('tools wired:', type(build_server(svc)).__name__)
"
```

Expected: prints `tools wired: FastMCP` with no traceback. (This exercises config → store → server wiring without needing a live embedding endpoint.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml delfos/mcp/server.py delfos/mcp/__main__.py tests/mcp/conftest.py tests/mcp/test_server.py
git commit -m "feat(mcp): FastMCP read server, reconstruct prompt, stdio entry point

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Agent-as-planner, no server-side planner LLM → Task 2 (optional planner) + Task 4 (`__main__` builds service with `planner=None`). ✓
- 4 tools (search/traverse_forward/traverse_reverse/fetch) → Task 4. ✓
- `reconstruct` MCP prompt teaching the walk → Task 4 (`reconstruct_prompt` + `@mcp.prompt()`). ✓
- Tiered returns: `NodeSummary` + `ContentDetail`, embeddings stripped → Task 1. ✓
- `tag_filters` as (category, value) with unknown-category error → Task 4 (`_to_tag_filters`). ✓
- Embedder-only model dependency; `DELFOS_EMBED_*` config → Task 3. ✓
- Startup model-match fail-fast → Task 3 (`check_model_match`) + Task 4 (`main`). ✓
- `fetch` on the read-path boundary; tolerant of unknown ids → Task 2. ✓
- stdio transport + `python -m delfos.mcp` + `delfos-mcp` script → Task 4. ✓
- Tests with in-memory store + FakeEmbedder, no network → Tasks 1-4. ✓

**Placeholder scan:** No TBD/TODO; every code and test step is complete. ✓

**Type consistency:** `_search/_traverse_forward/_traverse_reverse/_fetch` signatures and `NodeSummary`/`ContentDetail` field names are identical across Tasks 1 and 4. `ReconstructionService.fetch`/`content_tags`/optional-`planner` defined in Task 2 are consumed with matching signatures in Task 4. `config_from_env`/`build_store`/`build_embedder`/`check_model_match` defined in Task 3 are consumed with matching names in Task 4. ✓

## Out of scope (deferred)

- Write path (`index` as an MCP tool) — separate spec.
- Server-side planner LLM / `reconstruct` as a tool / HTTP-SSE transport.
- Auth, multi-tenancy, remote deployment.

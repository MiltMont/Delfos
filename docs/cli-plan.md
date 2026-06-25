# Delfos CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `delfos` command-line interface — built on stdlib `argparse`, no new
dependencies — that unblocks the **write path** (`delfos index`) and gives a human a
terminal-side window into a store (`status`, `search`, `reconstruct`) plus a `serve`
alias for the existing MCP server.

**Why:** The read path is fully served by `delfos-mcp`, but the `Indexer`
(`delfos/indexer/pipeline.py`) has **no entry point** — the only way to build a persistent
store today is to write Python or run `scripts/smoke_local_llm.py` (which indexes into a
throwaway tempdir). `delfos index` is the one command that clears a real blocker; the
inspection commands are thin wrappers over `ReconstructionService` that make a store
debuggable without an MCP client.

**Architecture:** A thin `delfos/cli/` adapter. It introduces **no graph logic**: it wires
env/flags → store + embedder (+ optional planner) → existing `Indexer` /
`ReconstructionService` calls → **pure render functions** → stdout. Command logic stays in
plain, injectable functions (mirroring the MCP server's `_`-prefixed helpers) so it is
unit-testable without a network or MCP transport; `main()` does the env-driven wiring.

**Tech Stack:** Python 3.12+, stdlib `argparse`, Pydantic v2, the existing `delfos` package
and its C++-backed `NativeGraphStore`. **No runtime dependency is added.** (`.env` autoload
is intentionally out of scope — the CLI reads `os.environ` only, same as a normal Unix tool.)

## Command surface (this plan)

```
delfos index <repo> [--index-path PATH]          # build/update a persistent store
delfos status       [--index-path PATH]          # inspect a store (no network)
delfos search <query> [-k N] [--index-path PATH] # semantic seed lookup
delfos reconstruct <query> [--budget N] [-k N] [--index-path PATH]
delfos serve                                     # alias -> delfos.mcp.__main__:main
```

All commands resolve store + embedder from the existing `DELFOS_EMBED_*` env (see
`delfos/mcp/config.py`); `--index-path` overrides `DELFOS_INDEX_PATH`. `reconstruct`
additionally needs a planner, resolved from `DELFOS_LLM_*` (Task 1). `index`/`status`
never need a planner; `status` needs no embedder endpoint at all.

## Global Constraints

- Pyright runs in **strict mode** — all new code must be fully typed.
- All Pydantic models use `model_config = ConfigDict(extra="forbid")`.
- `requires-python = ">=3.12"`; no new runtime dependency.
- No component outside `delfos/store/` touches the C++ engine directly; all graph access
  goes through `Indexer` / `ReconstructionService`.
- The query/index-time embedding model must match the store's `embedding_model`
  (`check_model_match`, enforced before any embedder-backed command runs).
- Lint/format/type/test gates: `uv run ruff check .`, `uv run ruff format .`,
  `uv run pyright`, `uv run pytest`.

---

### Task 1: Shared config + planner builder (`delfos/config.py`)

Promote the env-driven config out of the MCP package into a neutral `delfos/config.py` so
both MCP and CLI share one source of truth, and add a planner builder for `reconstruct`.
`delfos/mcp/config.py` becomes a re-export shim so existing MCP imports/tests are untouched.
(Decision locked: single source of truth in `delfos/config.py`, not a CLI→MCP import.)

**Files:**

- Create: `delfos/config.py` (moved embed config + new planner config)
- Modify: `delfos/mcp/config.py` → re-export shim
- Modify: `delfos/mcp/__main__.py` import line (point at `delfos.config`)
- Test: `tests/test_config.py` (planner additions; existing `tests/mcp/test_config.py` keeps passing via the shim)

**Interfaces (new, additive):**

- `@dataclass(frozen=True) class PlannerConfig` — `llm_model: str | None`, `llm_base_url: str | None`, `llm_api_key: str | None`.
- `planner_config_from_env(env: Mapping[str, str]) -> PlannerConfig`
- `build_planner(cfg: PlannerConfig) -> OpenAIHopPlanner` — raises `RuntimeError` when `llm_model` is `None`.
- Re-exported unchanged from `delfos.config`: `ServerConfig`, `config_from_env`, `build_embedder`, `build_store`, `check_model_match`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from __future__ import annotations

import pytest

from delfos.config import (
    PlannerConfig,
    build_planner,
    config_from_env,
    planner_config_from_env,
)


def test_config_from_env_still_importable_from_delfos_config() -> None:
    # The embed config moved here; the MCP shim must keep re-exporting it.
    cfg = config_from_env({})
    assert cfg.embed_model == "nomic-embed-text"


def test_planner_config_defaults_to_none_model() -> None:
    assert planner_config_from_env({}) == PlannerConfig(
        llm_model=None, llm_base_url=None, llm_api_key=None
    )


def test_planner_config_reads_overrides() -> None:
    cfg = planner_config_from_env(
        {
            "DELFOS_LLM_MODEL": "local-chat",
            "DELFOS_LLM_BASE_URL": "http://localhost:8080/v1",
            "DELFOS_LLM_API_KEY": "local",
        }
    )
    assert cfg == PlannerConfig(
        llm_model="local-chat",
        llm_base_url="http://localhost:8080/v1",
        llm_api_key="local",
    )


def test_build_planner_without_model_raises() -> None:
    with pytest.raises(RuntimeError, match="DELFOS_LLM_MODEL"):
        build_planner(PlannerConfig(llm_model=None, llm_base_url=None, llm_api_key=None))


def test_build_planner_returns_planner_for_model() -> None:
    planner = build_planner(
        PlannerConfig(llm_model="local-chat", llm_base_url="http://x/v1", llm_api_key="k")
    )
    assert planner.model == "local-chat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError` for `delfos.config`.

- [ ] **Step 3: Write minimal implementation**

Move the entire current body of `delfos/mcp/config.py` into a new `delfos/config.py`
(the `ServerConfig` dataclass, `config_from_env`, `build_embedder`, `build_store`,
`check_model_match`, `_TRUTHY`). Then append the planner additions:

```python
from delfos.reconstruct.planners.openai import OpenAIHopPlanner


@dataclass(frozen=True)
class PlannerConfig:
    """Hop-planner (chat LLM) configuration for `reconstruct`."""

    llm_model: str | None
    llm_base_url: str | None
    llm_api_key: str | None


def planner_config_from_env(env: Mapping[str, str]) -> PlannerConfig:
    """Read the `DELFOS_LLM_*` chat-model settings (all optional)."""
    return PlannerConfig(
        llm_model=env.get("DELFOS_LLM_MODEL"),
        llm_base_url=env.get("DELFOS_LLM_BASE_URL"),
        llm_api_key=env.get("DELFOS_LLM_API_KEY"),
    )


def build_planner(cfg: PlannerConfig) -> OpenAIHopPlanner:
    """Construct the OpenAI-compatible hop planner; require a model name."""
    if cfg.llm_model is None:
        raise RuntimeError(
            "reconstruct needs a chat model; set DELFOS_LLM_MODEL "
            "(and DELFOS_LLM_BASE_URL for a local endpoint)"
        )
    client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    return OpenAIHopPlanner(cfg.llm_model, client=client)
```

Replace `delfos/mcp/config.py` with a shim that preserves its public surface:

```python
"""Backwards-compatible re-export of the shared config (now in :mod:`delfos.config`)."""

from __future__ import annotations

from delfos.config import (
    ServerConfig,
    build_embedder,
    build_store,
    check_model_match,
    config_from_env,
)

__all__ = [
    "ServerConfig",
    "build_embedder",
    "build_store",
    "check_model_match",
    "config_from_env",
]
```

Update the import line in `delfos/mcp/__main__.py` to read from `delfos.config`
(functionally identical; keeps the dependency pointing at the canonical module):

```python
from delfos.config import build_embedder, build_store, check_model_match, config_from_env
```

- [ ] **Step 4: Run tests + gates to verify they pass**

Run: `uv run pytest tests/test_config.py tests/mcp/test_config.py -v && uv run pyright && uv run ruff check .`
Expected: PASS — new planner tests pass, the existing `tests/mcp/test_config.py` still passes through the shim, no pyright/ruff errors.

- [ ] **Step 5: Commit**

```bash
git add delfos/config.py delfos/mcp/config.py delfos/mcp/__main__.py tests/test_config.py
git commit -m "refactor(config): promote shared config to delfos.config + add planner builder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: CLI core — parser, renderers, `index` + `status` (the MVP)

The write-path-unblocking slice. Pure render functions (no I/O) plus injectable command
runners, wired by `main()`. `status` needs no network; `index` needs only the embedder.

**Files:**

- Create: `delfos/cli/__init__.py`
- Create: `delfos/cli/render.py` (pure formatters)
- Create: `delfos/cli/app.py` (`build_parser`, command runners, `main`)
- Create: `delfos/cli/__main__.py` (`python -m delfos.cli`)
- Modify: `pyproject.toml` (`delfos` console script)
- Test: `tests/cli/__init__.py`, `tests/cli/conftest.py` (a `FixedEmbedder`), `tests/cli/test_render.py`, `tests/cli/test_parser.py`, `tests/cli/test_commands.py`

**Interfaces:**

- `render_index_stats(stats: IndexStats) -> str`
- `render_status(embed_model: str, embed_dim: int, files: list[IndexedFile]) -> str`
- `build_parser() -> argparse.ArgumentParser`
- `run_index(repo_path: str, store: GraphStore, embedder: Embedder) -> IndexStats`
- `run_status(store: GraphStore, embed_model: str, embed_dim: int) -> str`
- `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/cli/__init__.py` (empty), then `tests/cli/test_render.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from delfos.cli.render import render_index_stats, render_status
from delfos.indexer import IndexStats
from delfos.store.base import IndexedFile


def test_render_index_stats_reports_all_counters() -> None:
    stats = IndexStats(
        indexed_files=3, skipped_files=1, failed_files=["bad.py"],
        nodes_written=42, edges_written=40,
    )
    out = render_index_stats(stats)
    assert "indexed: 3" in out
    assert "skipped: 1" in out
    assert "nodes: 42" in out
    assert "edges: 40" in out
    assert "bad.py" in out  # failures are named, not just counted


def test_render_status_lists_model_and_files() -> None:
    files = [
        IndexedFile(file_path="a.py", git_sha="abcdef123456", indexed_at=datetime(2026, 6, 25, tzinfo=UTC)),
    ]
    out = render_status("nomic-embed-text", 768, files)
    assert "nomic-embed-text" in out
    assert "768" in out
    assert "a.py" in out
    assert "abcdef1" in out  # short sha
    assert "1 file" in out  # count summary


def test_render_status_handles_empty_store() -> None:
    out = render_status("m", 8, [])
    assert "0 files" in out or "empty" in out.lower()
```

Create `tests/cli/test_parser.py`:

```python
from __future__ import annotations

import pytest

from delfos.cli.app import build_parser


def test_index_requires_repo() -> None:
    parser = build_parser()
    ns = parser.parse_args(["index", "some/repo"])
    assert ns.command == "index"
    assert ns.repo == "some/repo"


def test_index_path_flag_overrides() -> None:
    parser = build_parser()
    ns = parser.parse_args(["status", "--index-path", "/data/g"])
    assert ns.index_path == "/data/g"


def test_no_command_errors() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
```

Create `tests/cli/conftest.py` with a `FixedEmbedder`. The reconstruct suite's
`FakeEmbedder` only does **keyed lookups** (`self._mapping[t]`) and raises `KeyError` on any
unmapped text, so it cannot embed the arbitrary cue strings the real `Indexer` extracts from
a file. `run_index` needs an embedder that returns a constant vector for *any* text:

```python
from __future__ import annotations

from tests.reconstruct.conftest import EMB_DIM, EMB_MODEL


class FixedEmbedder:
    """Embedder protocol double: one fixed vector for any text (model matches the store)."""

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
        return [[0.0] * EMB_DIM for _ in texts]
```

Create `tests/cli/test_commands.py` (integration: real temp store + `FixedEmbedder`, no network):

```python
from __future__ import annotations

from pathlib import Path

from delfos.cli.app import run_index, run_status
from delfos.store.native_store import NativeGraphStore
from tests.cli.conftest import FixedEmbedder
from tests.reconstruct.conftest import EMB_DIM, EMB_MODEL


def _store(tmp_path: Path) -> NativeGraphStore:
    s = NativeGraphStore(tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL)
    s.initialize()
    return s


def test_index_then_status_reflects_written_files(tmp_path: Path) -> None:
    # A tiny repo to index.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def hello():\n    return 1\n")

    store = _store(tmp_path)
    # FixedEmbedder returns a constant vector for any cue text it is asked to embed.
    embedder = FixedEmbedder()

    stats = run_index(str(repo), store, embedder)
    assert stats.indexed_files == 1
    assert stats.failed_files == []

    out = run_status(store, EMB_MODEL, EMB_DIM)
    assert "mod.py" in out
    assert "1 file" in out
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'delfos.cli'`.

- [ ] **Step 3: Write minimal implementation**

Create `delfos/cli/__init__.py`:

```python
"""Delfos command-line interface: write path (`index`) + store inspection."""
```

Create `delfos/cli/render.py`:

```python
"""Pure stdout formatters for CLI commands. No I/O, no graph access — trivially testable."""

from __future__ import annotations

from delfos.indexer import IndexStats
from delfos.store.base import IndexedFile


def render_index_stats(stats: IndexStats) -> str:
    lines = [
        f"indexed: {stats.indexed_files}  skipped: {stats.skipped_files}  "
        f"failed: {len(stats.failed_files)}",
        f"nodes: {stats.nodes_written}  edges: {stats.edges_written}",
    ]
    if stats.failed_files:
        lines.append("failed files:")
        lines.extend(f"  {path}" for path in stats.failed_files)
    return "\n".join(lines)


def render_status(embed_model: str, embed_dim: int, files: list[IndexedFile]) -> str:
    header = f"embedding model: {embed_model} (dim {embed_dim})"
    count = f"{len(files)} file{'s' if len(files) != 1 else ''} indexed"
    if not files:
        return f"{header}\n{count} (empty store)"
    rows = [
        f"  {f.git_sha[:7]}  {f.indexed_at.isoformat()}  {f.file_path}"
        for f in sorted(files, key=lambda f: f.file_path)
    ]
    return "\n".join([header, count, *rows])
```

Create `delfos/cli/app.py`:

```python
"""Argument parsing, env-driven wiring, and command dispatch for the `delfos` CLI."""

from __future__ import annotations

import argparse
import os

from delfos.config import build_embedder, build_store, check_model_match, config_from_env
from delfos.indexer import Indexer
from delfos.indexer.embedder import Embedder
from delfos.store import GraphStore

from .render import render_index_stats, render_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="delfos", description="Delfos graph-memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build/update a persistent store from a repo")
    p_index.add_argument("repo", help="path to the repository to index")
    p_index.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    p_status = sub.add_parser("status", help="inspect a store's manifest")
    p_status.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    # search / reconstruct / serve are added in Task 3.
    return parser


def run_index(repo_path: str, store: GraphStore, embedder: Embedder):
    return Indexer(store, embedder).index(repo_path)


def run_status(store: GraphStore, embed_model: str, embed_dim: int) -> str:
    return render_status(embed_model, embed_dim, store.list_indexed_files())


def _override_index_path(args: argparse.Namespace) -> None:
    if getattr(args, "index_path", None):
        os.environ["DELFOS_INDEX_PATH"] = args.index_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _override_index_path(args)
    cfg = config_from_env(os.environ)
    store = build_store(cfg)
    try:
        if args.command == "index":
            embedder = build_embedder(cfg)
            check_model_match(store, embedder)
            print(render_index_stats(run_index(args.repo, store, embedder)))
        elif args.command == "status":
            print(run_status(store, cfg.embed_model, cfg.embed_dim))
        else:  # pragma: no cover - argparse `required=True` guards this
            return 2
    finally:
        store.close()
    return 0
```

Create `delfos/cli/__main__.py`:

```python
"""Entry point: `python -m delfos.cli` and the `delfos` console script."""

from __future__ import annotations

import sys

from delfos.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
```

In `pyproject.toml`, add to `[project.scripts]` (alongside the existing `delfos-mcp`):

```toml
[project.scripts]
delfos = "delfos.cli.app:main"
delfos-mcp = "delfos.mcp.__main__:main"
```

- [ ] **Step 4: Run tests + gates to verify they pass**

Run: `uv run pytest tests/cli/ -v && uv run pyright && uv run ruff check . && uv run ruff format --check .`
Expected: PASS — render, parser, and the index→status integration test pass; no pyright/ruff errors.

- [ ] **Step 5: Smoke the real binary**

Run (indexes this repo into a temp store with a fake-model store; status reads it back):

```bash
TMP=$(mktemp -d)
uv run delfos status --index-path "$TMP/graph"   # empty store
# index requires a live embedder endpoint; smoke just the offline path here.
```

Expected: prints the embedding-model header and `0 files indexed (empty store)` with no traceback.

- [ ] **Step 6: Commit**

```bash
git add delfos/cli/ tests/cli/ pyproject.toml
git commit -m "feat(cli): argparse CLI with index + status commands

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Read commands — `search`, `reconstruct`, `serve`

Layer the inspection commands on top of `ReconstructionService`, plus a `serve` alias for
the MCP server. `search` needs the embedder; `reconstruct` additionally builds a planner
via Task 1; `serve` delegates to the existing MCP entry point.

**Files:**

- Modify: `delfos/cli/render.py` (add `render_search`, `render_reconstruct`)
- Modify: `delfos/cli/app.py` (subparsers + dispatch branches)
- Test: `tests/cli/test_parser.py` (extend), `tests/cli/test_render.py` (extend), `tests/cli/test_commands.py` (extend)

**Interfaces:**

- `render_search(cues: list[CueNode]) -> str`
- `render_reconstruct(contents: list[ContentNode]) -> str`
- `run_search(query: str, k: int, service: ReconstructionService) -> list[CueNode]`
- New subparsers: `search <query> [-k N] [--index-path]`, `reconstruct <query> [--budget N] [-k N] [--index-path]`, `serve`. `-k`/`--budget` parse as `int` (`type=int`).

- [ ] **Step 1: Write the failing test**

Extend `tests/cli/test_parser.py` (the `search`/`reconstruct` subparsers land in this task,
so the `-k`/`--budget` parsing tests belong here):

```python
def test_search_parses_k() -> None:
    parser = build_parser()
    ns = parser.parse_args(["search", "how does auth work", "-k", "7"])
    assert ns.command == "search"
    assert ns.query == "how does auth work"
    assert ns.k == 7  # parsed as int, not "7"


def test_reconstruct_parses_budget() -> None:
    parser = build_parser()
    ns = parser.parse_args(["reconstruct", "x", "--budget", "2"])
    assert ns.command == "reconstruct"
    assert ns.budget == 2  # parsed as int


def test_serve_takes_no_args() -> None:
    parser = build_parser()
    ns = parser.parse_args(["serve"])
    assert ns.command == "serve"
```

Extend `tests/cli/test_render.py`:

```python
def test_render_search_lists_cue_ids_and_text() -> None:
    from delfos.cli.render import render_search
    from tests.reconstruct.conftest import make_cue

    out = render_search([make_cue("cue-1", "auth"), make_cue("cue-2", "login")])
    assert "cue-1" in out and "auth" in out
    assert "cue-2" in out and "login" in out


def test_render_search_handles_no_hits() -> None:
    from delfos.cli.render import render_search

    assert "no" in render_search([]).lower()


def test_render_reconstruct_shows_content_provenance() -> None:
    from delfos.cli.render import render_reconstruct
    from tests.reconstruct.conftest import make_content

    out = render_reconstruct([make_content("c1", "login")])
    assert "c1" in out
    assert "login" in out  # symbol_name / signature surfaced
```

Extend `tests/cli/test_commands.py` with a `search` integration test that seeds a store
(reuse `tests/reconstruct/conftest` `load`/`make_cue`/`edge`) and a keyed `FakeEmbedder`
(the query text is known here, so keyed lookups are fine), then calls the
`run_search(query, k, service)` helper and asserts the rendered output names the seeded cue.
(`reconstruct` is exercised by the existing planner tests + the smoke harness; the CLI
wiring is covered by the boot smoke in Step 4 rather than a network test.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/ -v`
Expected: FAIL — `ImportError` for `render_search`/`render_reconstruct` (and `run_search`).

- [ ] **Step 3: Write minimal implementation**

Add to `delfos/cli/render.py`:

```python
from delfos.schema import ContentNode, CueNode


def render_search(cues: list[CueNode]) -> str:
    if not cues:
        return "no matching cues"
    return "\n".join(f"  {c.id}  {c.text}" for c in cues)


def render_reconstruct(contents: list[ContentNode]) -> str:
    if not contents:
        return "no content reconstructed"
    lines: list[str] = []
    for c in contents:
        label = c.signature or c.symbol_name or c.kind.value
        lines.append(f"  {c.id}  [{c.memory_layer.value}] {c.source_file}: {label}")
    return "\n".join(lines)
```

In `delfos/cli/app.py`, register the new subparsers in `build_parser` (`search` with
`-k` `type=int` default 5; `reconstruct` with `--budget` `type=int` default 3 and `-k`
`type=int` default 5; and a no-arg `serve`), add a `run_search(query, k, service)` helper,
and extend `main`'s dispatch:

```python
# search: embedder only
embedder = build_embedder(cfg)
check_model_match(store, embedder)
service = ReconstructionService(store, embedder)
print(render_search(service.search(args.query, args.k)))

# reconstruct: embedder + planner
embedder = build_embedder(cfg)
check_model_match(store, embedder)
planner = build_planner(planner_config_from_env(os.environ))
service = ReconstructionService(store, embedder, planner)
print(render_reconstruct(service.reconstruct(args.query, args.budget)))

# serve: hand off to the MCP entry point (no store opened here)
from delfos.mcp.__main__ import main as serve_main
serve_main()
```

> The `serve` branch should run **before** `build_store`/`store.close` wiring (it owns its
> own lifecycle via the MCP `main`); structure `main()` so `serve` returns early.

- [ ] **Step 4: Run tests + gates + boot smoke**

Run: `uv run pytest tests/cli/ -v && uv run pyright && uv run ruff check . && uv run ruff format --check .`

Then smoke the wiring without a live endpoint (mirrors the MCP plan's boot check):

```bash
uv run python -c "
from delfos.cli.app import build_parser
ns = build_parser().parse_args(['reconstruct', 'x', '--budget', '2'])
print('parsed:', ns.command, ns.budget)
"
```

Expected: tests PASS, no pyright/ruff errors, smoke prints `parsed: reconstruct 2`.

- [ ] **Step 5: Commit**

```bash
git add delfos/cli/render.py delfos/cli/app.py tests/cli/
git commit -m "feat(cli): search, reconstruct, and serve commands

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Goal coverage:**

- Write path unblocked (`delfos index`) → Task 2. ✓
- Store inspection without an MCP client (`status`/`search`/`reconstruct`) → Tasks 2–3. ✓
- stdlib `argparse`, no new runtime dependency → Tasks 2–3 (only stdlib imported). ✓
- Reuses existing `Indexer`/`ReconstructionService`/config; no new graph logic → all tasks. ✓
- Embedding-model match enforced before embedder-backed commands → Task 2/3 (`check_model_match`). ✓
- `reconstruct` planner resolved from `DELFOS_LLM_*` with a clear error when unset → Task 1 (`build_planner`) + Task 3. ✓
- `serve` reaches the existing MCP server → Task 3. ✓

**Testability:** pure renderers (no I/O) + injectable `run_*` helpers + offline integration
tests over a temp `NativeGraphStore` (a constant `FixedEmbedder` for `index`, the keyed
`FakeEmbedder` for `search`); no command test requires a network. Argparse covered
directly. ✓

**Placeholder scan:** none — the embedder-on-arbitrary-text gap is closed by the
`FixedEmbedder` in `tests/cli/conftest.py` (Task 2 Step 1); everything else is concrete. ✓

## Out of scope (deferred)

- Non-Python indexing (the walker is `*.py`-only in `pipeline.py`).
- `--watch`/daemon incremental indexing (re-running `index` is already SHA-skip cheap).
- Store reset/compact/`--json` output, progress bars, color.
- `.env` autoload in the runtime CLI (kept a dev-only convenience of the smoke harness).
- A `typer`/`click` dependency (revisit only if subcommand ergonomics outgrow argparse).

```

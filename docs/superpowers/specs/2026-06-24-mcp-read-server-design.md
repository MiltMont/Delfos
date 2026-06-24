# MCP Read Server — Design

**Date:** 2026-06-24
**Status:** Approved (design); pending implementation plan
**Component:** `delfos/mcp/`

## Purpose

Expose the Delfos read path over the **Model Context Protocol** so an agent can
query the persisted Cue→Tag→Content graph as memory. The reconstruction engine
([2026-06-23 spec](./2026-06-23-reconstruction-engine-design.md)) already
implements the read-path operations on top of `GraphStore`. This component wraps
those primitives in an MCP server.

The defining choice: **the calling agent is the planner.** In an MCP world the
LLM that issues the tool call is itself capable of the per-hop reasoning the
paper calls "LLM-driven traversal," so the server exposes graph *primitives* and
a *prompt* that teaches the walk — it does **not** run a server-side planner LLM.

This is the layer that makes the package name (`delfos`, "Graph memory MCP
server for codebases") true.

## Key decisions (resolved during brainstorming)

1. **Planner role — agent-as-planner.** The MCP server exposes graph primitives;
   the calling agent drives the depth-first walk itself. `reconstruct` + the
   `HopPlanner` protocol + `OpenAIHopPlanner` remain in the tree as an internal
   engine (smoke harness, tests, a possible future CLI) but are **not** exposed
   as MCP tools.
2. **Scope — read path only.** `index()` (write path) stays a library/CLI entry
   point and gets its own spec later. This server operates over an
   already-indexed store.
3. **Walk guidance — MCP prompt.** The paper's discipline (seed → expand →
   descend one → respect budget → stop) ships as a reusable MCP **prompt**
   (`reconstruct`). Tool descriptions stay terse; the algorithm lives in the
   prompt, not in code.
4. **Return shape — tiered (summaries + fetch).** The walk tools return compact
   summaries; a `fetch` tool returns full content bodies for the agent's final
   selection. Mirrors the standard MCP search+fetch convention: cheap hops, full
   payload only when asked. Embeddings are always stripped.
5. **Model dependency — embedder only.** The embedder is the server's single
   mandatory model dependency. There is no server-side planner LLM. The embedder
   may be OpenAI-hosted or a local OpenAI-compatible endpoint (a config choice).
6. **Transport — stdio.** Standard for local MCP servers and Claude Code.

## Architecture

```
agent (planner)  ──MCP/stdio──▶  delfos.mcp.server
                                   ├─ tools  ──▶ ReconstructionService ──▶ GraphStore
                                   └─ prompt: reconstruct (protocol text)
                                 embedder (only model dependency)
```

The server is a thin, fully-typed adapter. It introduces **no new graph logic**;
every graph operation goes through `ReconstructionService`, which goes through
`GraphStore` — preserving the project's single-boundary rule (no component
touches the C++ engine directly).

## Surface

### Tools (4)

| Tool | Signature | Returns |
| --- | --- | --- |
| `search` | `search(query: str, k: int = 5)` | `list[NodeSummary]` — cue entry points by vector similarity |
| `traverse_forward` | `traverse_forward(cue_ids: list[str], tag_filters: list[tuple[str, str]] \| None = None)` | `list[NodeSummary]` — cues expanded to content |
| `traverse_reverse` | `traverse_reverse(content_ids: list[str])` | `list[NodeSummary]` — sibling cues pointing at the content |
| `fetch` | `fetch(ids: list[str])` | `list[ContentDetail]` — full content bodies |

`tag_filters` arrive as `(category, value)` string pairs and are mapped to the
service's `TagFilter` (`TagCategory`, `str`); an unknown category is a tool error
with the valid categories listed.

### Prompt (1)

`reconstruct(query: str, budget: int = 3)` — returns protocol text instructing
the agent to:

1. `search(query)` for seed cues.
2. `traverse_forward` on the most promising cues.
3. Descend one hop at a time (depth-first), expanding the single best candidate
   rather than fanning out.
4. Respect `budget` as a cap on traversal steps; backtrack when a branch dries
   up.
5. `fetch` the ids worth keeping.
6. Stop when relevance drops or budget is exhausted, then synthesize from the
   fetched bodies.

## Data shapes (`delfos/mcp/views.py`)

Dedicated MCP-facing models, **not** reused from `reconstruct.planner` — this
keeps the MCP layer decoupled from the planner's `CandidateSummary` contract.
The 500-char truncation constant is shared/reused.

```python
class NodeSummary(BaseModel):
    id: str
    kind: Literal["cue", "content"]
    label: str            # cue text, or content signature/symbol/kind
    snippet: str | None   # content docstring or body truncated to ≤500 chars
    tags: list[str]       # "CATEGORY=value" for content; empty for cues

class ContentDetail(BaseModel):
    id: str
    symbol_name: str | None
    signature: str | None
    docstring: str | None
    body: str
    memory_layer: str
    source_file: str
    git_sha: str
    # embedding intentionally absent
```

Both use `extra="forbid"`, consistent with the rest of the schema.

## Components

- `delfos/mcp/server.py` — the FastMCP app: tool + prompt registration, wiring a
  `ReconstructionService` (built without a planner) and the configured embedder.
- `delfos/mcp/views.py` — `NodeSummary` / `ContentDetail` and the converters from
  `CueNode` / `ContentNode`.
- `delfos/mcp/config.py` — env-driven startup: store snapshot path + embedder
  construction, reusing the smoke harness's `DELFOS_EMBED_*` convention
  (`DELFOS_EMBED_BASE_URL`, `DELFOS_EMBED_MODEL`, `DELFOS_EMBED_DIM`,
  `DELFOS_EMBED_API_KEY`, send-dimensions flag) plus `DELFOS_INDEX_PATH`.
- `delfos/mcp/__main__.py` — `python -m delfos.mcp` entry point; a `delfos-mcp`
  console script in `pyproject.toml`.

## Changes to existing code (targeted)

1. `ReconstructionService.__init__` — relax `planner` to
   `planner: HopPlanner | None = None`. `reconstruct()` raises a clear
   `RuntimeError` if invoked without a planner. This lets the server construct
   the service with **no planner LLM** while leaving `reconstruct` / `HopPlanner`
   fully intact for the smoke harness and tests.
2. Add `ReconstructionService.fetch(ids: Sequence[str]) -> list[ContentNode]` —
   resolves each id via `store.get_node`, keeping only ACTIVE `ContentNode`s
   (skipping unknown/non-content/deleted ids). Keeps `fetch` on the read-path
   boundary instead of letting the MCP layer touch the store directly.
3. `pyproject.toml` — add the `mcp` dependency and the `delfos-mcp` console
   script.

## Configuration & startup contract

The server is constructed with a store snapshot path and an embedder, both from
the environment (defaults mirror the smoke harness; OpenAI-hosted is reached by
pointing `DELFOS_EMBED_BASE_URL` at the OpenAI endpoint).

**Invariant promoted to a startup guarantee:** the embedder's model must equal
the store's recorded `embedding_model`. The server checks this when it opens the
store and **fails fast** with an actionable message on mismatch (read-time
queries must live in the same vector space as the indexed cues). This moves the
index-time/read-time embedding-model match from smoke-script lore to a hard
server contract.

## Error handling

- Empty graph results → `[]` (every read tool).
- `fetch` of unknown / non-content / deleted ids → silently skipped, matching
  the service's existing tolerant filtering.
- Unknown `tag_filters` category → tool error naming the valid `TagCategory`
  values.
- Embedder / endpoint failure (in `search`) → MCP tool error carrying "check the
  embedding endpoint is up and the model is pulled" guidance.
- Embedding-model mismatch at startup → fail fast before serving any request.

## Testing

No network; mirrors the existing test style (in-memory `NativeGraphStore` +
`FakeEmbedder`).

- `views` conversion: embedding stripped from `ContentDetail`; snippet truncated
  at 500 chars; tags rendered `CATEGORY=value`.
- Each tool against a tiny seeded Cue–Tag–Content graph: `search` returns cue
  summaries, `traverse_forward` respects `tag_filters`, `traverse_reverse`
  returns sibling cues, `fetch` returns full bodies and skips unknown ids.
- Startup mismatch check raises on an embedder/store model disagreement.
- The `reconstruct` prompt returns text containing the protocol steps and echoes
  `query` / `budget`.
- `ReconstructionService.reconstruct()` raises when constructed without a
  planner.

## Out of scope

- The write path (`index` as an MCP tool) — separate spec.
- Any server-side planner LLM, `reconstruct` as an MCP tool, or HTTP/SSE
  transport.
- Auth, multi-tenant, or remote deployment concerns.

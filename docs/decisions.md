# Key Decisions

This is Delfos's decision log — the *why* behind the architecture, for
contributors who need to understand intent that isn't recoverable from the code
alone. It consolidates the design rationale that was previously scattered across
per-milestone design specs. For *what* the system is and *how* the pieces fit,
see [`CLAUDE.md`](../CLAUDE.md) and the [`README`](../README.md).

Each entry states the decision, the reasoning, and (where useful) what was
explicitly ruled out.

---

## Foundational model

### Active reconstruction, not one-shot retrieval

Delfos implements the active *reconstruction* model from ["Memory is
Reconstructed, Not Retrieved: Graph Memory for LLM
Agents"](https://arxiv.org/abs/2606.06036) (Ji, Li & Hooi; arXiv 2606.06036,
ICML 2026). Memory access is an iterative, LLM-driven traversal of a persistent
graph rather than a single similarity lookup.

**Why:** the paper's premise — that reconstructing a relevant set by walking
semantic connections beats static retrieval — is the entire reason this project
exists. Every layer is shaped by it.

### `reconstruct` uses depth-first sequential traversal

The walk deepens one path — expanding the single best candidate per hop and
backtracking when a branch dries up — rather than fanning out breadth-first.

**Why:** faithful to the paper's LLM-driven traversal. One expansion per planner step keeps
the walk cheap and the reasoning legible, and makes `budget` a hard ceiling on
planner calls.

### Graph shape: Cue → Tag → Content

Three node types with directional, typed edges (`CUE_OF`, `TAGGED_WITH`,
`PART_OF_TOPIC`, `REDIRECTS_TO`). Only **cues** are embedded and vector-searched;
**tags** are filtered categorically and never embedded; **content** is the
artifact returned to agents.

**Why:** separating the searchable entry points (cues) from the payload
(content) keeps the vector index small and the returned artifacts rich, and lets
tags act as cheap categorical bridges instead of more embeddings.

---

## Read path

### Primitives are pure; only `reconstruct` is LLM-in-the-loop

`search`, `traverse_forward`, and `traverse_reverse` are pure, deterministic
graph operations. Only `reconstruct` invokes an LLM (via the `HopPlanner`
abstraction), which decides, at each hop, which neighbors to collect and which
single neighbor to descend into.

**Why:** keeps the cheap, deterministic parts independently testable and
reusable, and confines all non-determinism (and token cost) to the one operation
that needs it. Tests inject a `FakeHopPlanner` and a fake embedder, so the whole
layer runs offline and deterministically.

### `budget` caps total planner (LLM) calls

`budget` limits the total number of planner calls across the entire walk;
backtracking spends budget too. Default 3.

**Why:** a single, predictable knob that bounds cost regardless of graph shape.

### MCP server: the calling agent *is* the planner

The MCP server exposes graph *primitives* plus a *prompt* that teaches the walk.
It does **not** run a server-side planner LLM. The `reconstruct` engine, the
`HopPlanner` protocol, and concrete planners remain in the tree as an internal
engine (smoke harness, tests, a possible future CLI) but are not exposed as MCP
tools.

**Why:** in an MCP world the agent issuing the tool call is already capable of
the per-hop reasoning the paper calls "LLM-driven traversal." Making it the
planner avoids a redundant second LLM, keeps the server a thin adapter, and lets
the walk discipline (seed → expand → descend one → respect budget → stop) ship
as a reusable prompt rather than server-side code.

### Tiered returns: summaries + `fetch`; embeddings always stripped

Walk tools return compact `NodeSummary`s; a separate `fetch` tool returns full
`ContentDetail` bodies for the agent's final selection. Embeddings are never
returned.

**Why:** mirrors the standard MCP search+fetch convention — cheap hops, full
payload only when asked — which controls token cost. Embeddings are internal and
useless to the agent.

---

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

---

## Storage & provenance

### One boundary: everything goes through `GraphStore`

No component (indexer, MCP tools, future CLI) touches the C++ engine directly.
`NativeGraphStore` is the concrete backend.

**Why:** a single seam keeps the Python layers decoupled from the storage engine
and makes the backend swappable. The earlier `DuckDBGraphStore` was removed once
the native engine landed; the interface is what everything codes against.

### Provenance on every node/edge; delete-and-reindex for staleness

Every node and edge records `source_file` and `git_sha`. When a file's SHA
changes, all of its nodes/edges are hard-deleted and the file is re-processed
from scratch.

**Why:** delete-and-reindex is dramatically simpler than symbol-level diffing
and can't leave partial/inconsistent state. The atomic unit is one file per
transaction, with the checkpoint manifest entry written inside the same
transaction — so a crash mid-file leaves the store untouched and the file is
retried on restart.

### Embedding model is configured at store construction, never hard-coded

`NativeGraphStore` is built with a single `embedding_model` and rejects nodes
whose model differs; the `EmbeddedMixin` validator enforces that
`embedding_model` is present whenever `embedding` is set. The MCP server
promotes this to a **startup guarantee**: the embedder's model must equal the
store's recorded `embedding_model`, checked at open time with fail-fast on
mismatch.

**Why:** read-time queries must live in the same vector space as the indexed
cues. Catching a mismatch at startup turns a silent garbage-results failure into
an actionable error.

### `NodeData.embedding` is float64

The C++ `NodeData.embedding` is `std::vector<double>`, even though USearch uses
float32 internally.

**Why:** preserves exact round-trips of Python floats across the boundary;
narrowing to float32 happens only inside the index.

---

## Explicitly out of scope for v1

- **No symbol-level diffing, tombstones, or rename detection.** The read path
  filters to `status=ACTIVE` and follows a `REDIRECTS_TO` edge transparently *if
  present*, so the model is forward-compatible at near-zero cost — but the
  indexer emits no tombstones or redirects yet. Historical/tombstone queries are
  deferred.

---

## Engineering standards

- **Pyright strict mode is non-negotiable** — all code fully typed, no implicit
  `Any`.
- **`extra="forbid"` on all Pydantic models** — reject unknown fields rather than
  silently accept them, so schema drift surfaces immediately.

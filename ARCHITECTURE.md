# Architecture

This document is the orientation pass for contributors: what Delfos is built
around, how the pieces fit, and which invariants hold the whole thing together.
It is deliberately not exhaustive — the *why* behind each decision lives in
[`docs/decisions.md`](docs/decisions.md), and the module docstrings are the
source of truth for details. Read this first, then go where your change takes
you.

## The one idea everything serves

Delfos implements the active *reconstruction* model from ["Memory is
Reconstructed, Not Retrieved: Graph Memory for LLM Agents"](https://arxiv.org/abs/2606.06036)
(arXiv 2606.06036) over a code repository, and exposes it as an MCP server.

The premise: an agent looking for relevant code shouldn't get one shot at a
similarity search. It should *walk* — start from a semantic entry point, look at
the neighbors, decide what to keep, descend into the most promising branch,
backtrack when a path dries up. Memory as an iterative, reasoning-driven
traversal of a persistent graph, not a top-k lookup.

Every layer of this codebase is shaped by that premise. If a change makes the
walk less legible, less cheap, or less deterministic where it should be
deterministic, it's probably fighting the architecture.

## The graph: Cue → Tag → Content

Three node types, defined as Pydantic models in `delfos/schema/`:

- **`CueNode`** — the entry points agents query by: function names, concept
  strings, error messages. These are the *only* nodes that carry embeddings and
  the only ones reachable by vector search.
- **`TagNode`** — categorical bridges (`MODULE_PATH`, `ARCH_LAYER`,
  `PATTERN_TYPE`, `LANG_CONSTRUCT`, `LANGUAGE`). Filtered by category, never
  embedded. Shared across files.
- **`ContentNode`** — the payload: functions, classes, modules, commits, tests.
  What actually gets returned to the agent.

Edges are directional and typed: `CUE_OF`, `TAGGED_WITH`, `PART_OF_TOPIC`,
`REDIRECTS_TO`.

The separation is the point. Keeping cues (small, embedded, searchable) apart
from content (rich, returned, never embedded) keeps the vector index tiny and
the payloads unconstrained, and tags give the walk cheap categorical filtering
without more embeddings.

## Two layers, one boundary

The codebase is Python orchestration over a C++ storage engine, and they meet
at exactly one seam: the `GraphStore` abstract base class
(`delfos/store/base.py`).

```
  indexer      mcp server      reconstruct      cli
      \             |               |            /
       \            |               |           /
        +---------- GraphStore (ABC) ----------+     ← the only boundary
                         |
                 NativeGraphStore
                         |
                 delfos._delfos (nanobind)
                         |
                    libdelfos (C++)
```

**No component ever touches the C++ engine directly.** Not the indexer, not the
MCP tools, not the CLI. Everything codes against `GraphStore`;
`NativeGraphStore` (`delfos/store/native_store.py`) is the concrete backend.
This seam is what made it possible to delete the earlier DuckDB backend without
touching anything above it — keep it that way.

### The engine: `libdelfos/`

Header-only C++ under `libdelfos/include/delfos/`:

- **`graph.hpp`** — an in-memory CSR directed property graph: O(1) ID lookup,
  cache-friendly adjacency.
- **`vector_index.hpp`** — HNSW cosine similarity over USearch; <1ms for
  50K × dim=1536 k=5 queries.
- **`snapshot.hpp`** — persistence via FlatBuffers + USearch's native format,
  with atomic rename for crash safety.

`libdelfos/bindings/py_delfos.cpp` exposes `Store` and `NodeData` to Python as
the `delfos._delfos` extension. It has its own tests (`libdelfos/tests/`,
Catch2, built with ASan+UBSan in the debug preset) and a nanobench benchmark
(`libdelfos/bench/`).

## The write path: indexing

`delfos/indexer/` turns a repository into the graph. Four modules, in pipeline
order:

1. **`parser.py`** — tree-sitter parse of Python source into a `ParsedModule`
   IR. Error-tolerant: files with syntax errors are parsed partially instead of
   raising.
2. **`extractor.py`** — the pure, side-effect-free heart: `ParsedModule` in,
   nodes and edges out. No I/O, no embedding, no persistence. Per file it emits
   one module `ContentNode`, one `ContentNode` per definition, `CueNode`s for
   symbols and raised error messages, and shared `TagNode`s.
3. **`embedder.py`** — attaches vectors to cue nodes via an OpenAI-compatible
   endpoint.
4. **`pipeline.py`** — the `Indexer` that ties it together and owns all the
   transactional discipline (below).

The crash-recovery model is simple and strict: **one file per transaction**.
For each file the pipeline computes the git blob SHA, skips it if the checkpoint
manifest already has that SHA, and otherwise deletes the file's prior
nodes/edges and writes the new ones — embeddings included — inside a single
transaction, together with the manifest entry. A crash mid-file leaves the store
untouched and the file is retried on the next run. Staleness handling is
**delete-and-reindex**: no symbol-level diffing, no tombstones, no rename
detection in v1.

Everything a run produces lands in one self-describing directory per repo, the
`.delfos/` workspace (`delfos/workspace.py`):

```
<repo>/.delfos/
├── store/          # NativeGraphStore snapshot (graph + vectors)
├── index.scip      # SCIP cross-reference index
├── manifest.json   # provenance + consistency metadata
└── config.toml     # optional non-secret config
```

The manifest records which run produced the graph and the SCIP index (so
inconsistency is detectable) and the embedding model/dimension the index was
built with (so queries don't need to re-specify them).

## The write path: enrichment

`delfos/enrich/` is the other write path: agent-driven, not index-time.
`EnrichmentService.annotate` lets the calling agent — the extractor, the
write-path counterpart to "the calling agent is the planner" — attach
`CONCEPT` cues and `ARCH_LAYER`/`PATTERN_TYPE` tags to a content node it has
actually read. Concepts are embedded like any other cue; tag values are open
vocabulary, normalized, and the call echoes back existing values per category
(`GraphStore.list_tag_values`, backed by the C++ `Store.list_nodes_by_type`
binding over `Graph::nodes_by_type`) so agents converge on shared terms rather
than coining near-synonyms. Everything written carries the target's
`source_file`/`git_sha`, so the same delete-and-reindex that handles the index
wipes stale annotations too — no new storage concept required.

## The read path: reconstruction

`delfos/reconstruct/` is the read-path service, sitting entirely on
`GraphStore`. The split that matters:

- **Three pure primitives** — `search` (vector search over cues),
  `traverse_forward` (cues → content, with tag filters), `traverse_reverse`
  (content → cues). Deterministic graph operations, independently testable,
  no LLM anywhere near them.
- **One LLM-in-the-loop operation** — `reconstruct`, the depth-first walk. At
  each hop a `HopPlanner` (`planner.py`, a Protocol) sees compact
  `CandidateSummary`s of the neighbors and decides what to collect and which
  *single* neighbor to descend into. `budget` is a hard ceiling on planner
  calls (backtracking spends it too). `summaries.py` is the single place that
  decides how much of a node the LLM sees.

All non-determinism and all token cost are confined to `reconstruct`. Tests
inject `FakeHopPlanner` and a fake embedder, so the whole layer runs offline.

### The MCP server: the calling agent *is* the planner

Here's the twist worth understanding before touching `delfos/mcp/`: the MCP
server does **not** run a server-side planner LLM. The agent issuing the tool
calls (Claude Code, Cursor, …) is already an LLM capable of per-hop reasoning —
so the server exposes the *primitives* (`search`, `traverse_forward`,
`traverse_reverse`, `fetch`) plus a *prompt* that teaches the walk discipline
(seed → expand → descend one → respect budget → stop). The in-tree
`reconstruct` engine and its planners remain as an internal engine used by the
CLI, tests, and the smoke harness; they are not MCP tools.

Returns are tiered, following the MCP search+fetch convention: walk tools
return compact `NodeSummary`s, and a separate `fetch` returns full
`ContentDetail` bodies for the agent's final picks. Embeddings are never
serialized back to the agent. The MCP view models live in `mcp/views.py`,
deliberately separate from the planner's `CandidateSummary` so the tool surface
and the planner contract can evolve independently.

The same "calling agent" framing extends to the write path: the `annotate`
tool wraps `EnrichmentService` so the agent can write `CONCEPT` cues and
`ARCH_LAYER`/`PATTERN_TYPE` tags for content it has read, taught by the
`enrich` prompt (see [The write path: enrichment](#the-write-path-enrichment)).
`annotate` is always registered; called with only `content_id` it's a
vocabulary query.

### SCIP cross-references

`delfos/scip/` adds precise code navigation next to the semantic graph: at
index time, `scip-python` generates `index.scip`; at read time, `ScipService`
resolves a `ContentNode` to its references, implementations, and type
definitions. The trick that keeps this cheap: content node IDs *are* SCIP
symbol strings when SCIP coverage exists, so lookup is a direct key access with
no foreign-key indirection. The MCP server exposes these as three more tools
(`references`, `implementations`, `type_definition`).

## Entry points and wiring

- **`delfos/cli/`** — the `delfos` command: `index`, `status`, `doctor`,
  `search`, `reconstruct`, `serve`. Every command anchors on a repo's
  `.delfos/` workspace.
- **`delfos/mcp/`** — the FastMCP graph server (`delfos-mcp` / `delfos serve`):
  the read tools plus the `annotate` write tool. Tool logic lives in plain
  `_`-prefixed functions so it's unit-testable without an MCP transport;
  `build_server` registers thin wrappers.
- **`delfos/config.py`** — env-driven startup configuration (`DELFOS_*`
  variables; precedence documented in the README). This is also where the
  embedding-model startup check lives.

## Invariants — the short list

Break one of these and things fail in ways that are hard to see:

1. **Everything goes through `GraphStore`.** No reaching past the seam into
   `delfos._delfos`.
2. **Cue and content nodes always carry `source_file` + `git_sha`.**
   Provenance is what makes delete-and-reindex possible. Tags are the
   deliberate exception — they are shared across files and carry no
   provenance; a re-index drops the file-scoped edges, not the tags.
3. **One file per transaction**, manifest entry included. This is the entire
   crash-recovery story.
4. **One embedding model per store.** `NativeGraphStore` is constructed with
   an `embedding_model` and rejects nodes whose model differs; the server
   fail-fasts at startup if the query-time embedder doesn't match the index.
   A mismatch here doesn't error at query time — it silently returns garbage,
   which is why it's caught at the door.
5. **Primitives stay pure.** LLM calls belong only in `reconstruct` (or in the
   calling agent, in the MCP case).
6. **Pyright strict mode and `extra="forbid"` on every Pydantic model** are
   non-negotiable. Schema drift and untyped code surface immediately, by
   design.

## Where to go next

- [`docs/decisions.md`](docs/decisions.md) — the decision log: what was chosen,
  why, and what was explicitly ruled out.
- [`README.md`](README.md) — setup, configuration, and build commands.
- The module docstrings — each pipeline stage and service opens with a
  docstring stating its contract; they are kept accurate and are the fastest
  way to understand a module before editing it.
- `tests/` mirrors the package layout; the reconstruct tests
  (`tests/reconstruct/`) are the best executable walkthrough of the read path.

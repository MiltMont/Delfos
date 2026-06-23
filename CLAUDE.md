# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                 # install Python deps + dev tools
uv pip install -e .     # build + install the _delfos C++ extension (build isolation pulls scikit-build-core + nanobind)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type-check (strict mode)
uv run pytest           # run tests

# C++ build (debug, with ASan+UBSan)
cmake --preset debug
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure

# Python wheel
uv build
```

Pyright runs in strict mode — all new code must be fully typed.

## What This Is

**Delfos** is a graph-memory MCP server for codebases. It implements the active
*reconstruction* model from ["Memory is Reconstructed, Not Retrieved: Graph
Memory for LLM Agents"](https://arxiv.org/abs/2606.06036) (arXiv 2606.06036):
memory access is an iterative, LLM-driven traversal of a persistent graph, not
one-shot retrieval.

## Architecture

### Two layers, one boundary

Everything goes through `GraphStore` (`delfos/store/base.py`). No component
(indexer, MCP tools, future CLI) should ever touch the C++ engine directly.
`NativeGraphStore` is the concrete backend; `DuckDBGraphStore` has been removed.

### The graph: Cue → Tag → Content

Three node types (`delfos/schema/`):

- **`CueNode`** — entry points agents query by (function names, concept strings,
  error messages). The *only* nodes searched by vector similarity.
- **`TagNode`** — semantic bridges between cues and content; filtered categorically,
  never embedded. Categories: `MODULE_PATH`, `ARCH_LAYER`, `PATTERN_TYPE`,
  `LANG_CONSTRUCT`, `LANGUAGE`.
- **`ContentNode`** — the actual implementation artifact returned to agents
  (functions, classes, modules, commits, tests). Carries `memory_layer`
  (episodic / semantic / topic).

Edges (`Edge`) are directional and typed: `CUE_OF`, `TAGGED_WITH`,
`PART_OF_TOPIC`, `REDIRECTS_TO`.

### Storage engine (`libdelfos`)

The C++ engine (`libdelfos/`) provides:

1. **CSR graph** (`graph.hpp`) — in-memory directed property graph with O(1)
   ID lookup and cache-friendly adjacency.
2. **HNSW vector index** (`vector_index.hpp`) — USearch cosine similarity, <1ms
   for 50K × dim=1536 k=5 queries.
3. **Snapshot persistence** (`snapshot.hpp`) — FlatBuffers + USearch native
   format, atomic rename for crash safety.

The Python extension `delfos._delfos` (nanobind) exposes `Store` and `NodeData`
to `NativeGraphStore`.

### MCP tool shape (target state)

**Read path (active reconstruction):**
- `search(query) → List[CueNode]` — vector search on cue nodes
- `traverse_forward(cue_ids, tag_filters) → List[ContentNode]`
- `traverse_reverse(content_ids) → List[CueNode]`
- `reconstruct(query, budget) → List[ContentNode]` — depth-first traversal

**Write path:**
- `index(repo_path)` — trigger the construction pipeline

### Provenance and stale-handling

Every node and edge stores `source_file` and `git_sha`. The stale strategy is
**delete-and-reindex**: when a file's git SHA changes, all nodes/edges from that
file are hard-deleted and the file is re-processed from scratch.

### Crash recovery

The indexer's atomic unit is **one file per transaction**. The checkpoint
manifest records `(file_path, git_sha, indexed_at)`. The manifest entry is
written inside the same transaction as the file's nodes — a crash mid-file
leaves the store untouched and the file retried on restart.

### Embedding invariant

Every vector must carry `embedding_model`. `NativeGraphStore` is constructed
with a single `embedding_model` and rejects nodes whose model differs. The
`EmbeddedMixin` validator (`schema/nodes.py`) enforces `embedding_model` is
present whenever `embedding` is set.

## Key Decisions (see `docs/decisions.md`)

- `reconstruct` uses **depth-first sequential** traversal (per Figure 9 of arXiv 2606.06036)
- No symbol-level diffing, tombstones, or rename detection in v1
- Embedding model is configured at store construction; never hard-coded
- Pyright strict mode is non-negotiable; `extra="forbid"` on all Pydantic models
- `NodeData.embedding` is `std::vector<double>` (float64) to preserve Python
  float round-trips; USearch internally uses float32

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                 # install deps + dev tools
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type-check (strict mode)
```

There are no tests yet. Pyright runs in strict mode — all new code must be fully typed.

## What This Is

**Delfos** is a graph-memory MCP server for codebases. It implements the active *reconstruction* model from ["Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"](https://arxiv.org/abs/2606.06036) (arXiv 2606.06036): memory access is an iterative, LLM-driven traversal of a persistent graph, not one-shot retrieval.

The project is early-stage. What currently exists is the schema and storage abstraction — the MCP server layer, indexer, and CLI are not yet built.

## Architecture

### Two layers, one boundary

Everything goes through `GraphStore` (`src/delfos/store/base.py`). No component (indexer, MCP tools, future CLI) should ever touch DuckDB directly. `DuckDBGraphStore` is the only concrete backend; its methods currently all raise `NotImplementedError`.

### The graph: Cue → Tag → Content

Three node types (`src/delfos/schema/`):

- **`CueNode`** — entry points agents query by (function names, concept strings, error messages). These are the *only* nodes searched by vector similarity.
- **`TagNode`** — semantic bridges between cues and content; filtered categorically, never embedded. Categories: `MODULE_PATH`, `ARCH_LAYER`, `PATTERN_TYPE`, `LANG_CONSTRUCT`, `LANGUAGE`.
- **`ContentNode`** — the actual implementation artifact returned to agents (functions, classes, modules, commits, tests). Carries `memory_layer` (episodic / semantic / topic).

Edges (`Edge`) are directional and typed: `CUE_OF`, `TAGGED_WITH`, `PART_OF_TOPIC`, `REDIRECTS_TO`.

### MCP tool shape (target state)

**Read path (active reconstruction):**
- `search(query) → List[CueNode]` — vector search on cue nodes
- `traverse_forward(cue_ids, tag_filters) → List[ContentNode]` — expand active set
- `traverse_reverse(content_ids) → List[CueNode]` — redirect from retrieved content
- `reconstruct(query, budget) → List[ContentNode]` — server-side depth-first traversal (budget = hops, default 3); this is the primary tool agents use

**Write path:**
- `index(repo_path)` — trigger the construction pipeline

### Provenance and stale-handling

Every node and edge stores `source_file` and `git_sha`. The stale strategy for v1 is **delete-and-reindex**: when a file's `git_sha` changes, all nodes/edges from that file are hard-deleted and the file is re-processed from scratch. `delete_nodes_for_file(source_file)` backs this.

### Crash recovery

The indexer's atomic unit is **one file per transaction**. An `indexed_files` table (the checkpoint manifest) records `(file_path, git_sha, indexed_at)`. The manifest entry is written inside the same transaction as the file's nodes — if the indexer dies mid-file, the transaction is never committed and the file is cleanly retried on restart. `record_indexed_file` / `indexed_file_sha` back this.

### Embedding invariant

Every vector must carry `embedding_model` (and optionally `embedding_model_version`). A `DuckDBGraphStore` instance is configured with a single `embedding_model` at construction, and writes are rejected when the node's model disagrees. This is enforced from day one to make future re-embedding migrations feasible. The `EmbeddedMixin` validator (`nodes.py`) enforces `embedding_model` is present whenever `embedding` is set.

## Key Decisions (see `decisions.md`)

- `reconstruct` uses **depth-first sequential** traversal, not parallel breadth (per Figure 9 of the paper)
- No symbol-level diffing, tombstones, or rename detection in v1 — revisit before any production use
- Embedding model is configured via environment variable or config; never hard-coded
- Pyright strict mode is non-negotiable; `extra="forbid"` on all Pydantic models

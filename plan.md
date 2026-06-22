# MRAgent Implementation — Session Summary

The Paper

"Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents" (arXiv 2606.06036)

The core idea: instead of passive one-shot retrieval, memory access is an iterative, LLM-driven traversal of a persistent graph. Memory is never rebuilt from scratch — the graph persists, and per-query "reconstruction" means traversing it.

Two key operations (Section 4.1):

- Forward traversal: cue → tag → content (expand from what you know)
- Reverse traversal: content → cue/tag (redirect based on what you found)

Reconstruction state = (active set of candidates, accumulated evidence from prior steps)

Construction pipeline (Appendix B.1) — the write path, runs once then incrementally:

- Processes raw input (dialogue or code) into three memory layers:
  - Episodic: concrete events/chunks with Cue–Tag–Content triples
  - Semantic: stable facts and abstractions
  - Topic: high-level clusters of related episodes

---
What We're Building

An MCP server (not a standalone agent) that implements MRAgent's architecture over a codebase. Any MCP-compatible agent (Claude Code, Cursor, Delfos, etc.) can consume it as tools — they don't need to know anything about the graph internals.

Why MCP: clean separation, language-agnostic, works across agents you don't own.

---
Architecture Decisions

Database: SQLite + sqlite-vec

- Single embedded file, zero-dependency deployment
- sqlite-vec for ANN vector search on cue nodes
- Sufficient for local prototype; no separate process required

Language: Python

- Good tree-sitter + embedding tooling
- MCP server is a separate process — doesn't need to match the consuming agent's language

Concurrency: shared DB + single writer

- One indexer process owns all writes (triggered by git commits or file watcher)
- All agent sessions are read-only consumers
- SQLite WAL mode gives concurrent reads while writer is active
- No copy-per-agent (consistency nightmare)

Codebase change tracking: git-diff based

- Re-run B.1 pipeline only on changed files per commit
- Soft-invalidate stale nodes, reprocess incrementally
- Tree-sitter for incremental AST parsing

---

MCP Server Shape

Tools (write path):

- index(repo_path) — trigger B.1 construction/update pipeline

Tools (read path / active reconstruction):

- search(query) — vector search on cue nodes, returns candidates
- traverse_forward(cues, tags) — expand active set
- traverse_reverse(content_ids) — activate new cues from retrieved content

Resources:

- codebase://architecture — topic layer, high-level map
- codebase://module/{name} — module docs
- codebase://symbol/{name} — function/class nodes

---
Getting Agents to Use It

1. CLAUDE.md / system prompt — instruct agents to call search() before reading files
2. Tool quality — if reconstruction returns better context than raw file reads, agents prefer it naturally
3. Permission restrictions — for agents you own, remove file-read permissions and force all codebase access through MCP

---

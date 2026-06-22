# Foundations — What Must Be Resolved Before Implementation

This document identifies the decisions that are **blocking** — things that cannot be deferred without forcing a rewrite later. They are ordered by dependency: each item depends on the ones before it being settled first.

---

## 1. Code-specific Cue–Tag–Content schema

**Why it comes first:** Every other component depends on this. The DB schema, the indexer pipeline, the reconstruction algorithm, and the tool return types are all direct expressions of this schema. Building any of them without a settled schema means rebuilding them when it changes.

**What must be decided:**

- **Cues:** What constitutes a valid cue for code? Candidates: function names, class names, error message strings, concept phrases extracted by LLM ("rate limiting", "token refresh"). How are cues generated — purely from AST, from LLM extraction, or both? Are cues typed (symbol-cue vs concept-cue)?

- **Tags:** What are the valid tag categories? Proposed set: `module_path`, `arch_layer` (api / service / data / infra), `pattern_type` (factory, middleware, singleton), `lang_construct` (decorator, context_manager, generic), `language`. Are tags free-form strings or a closed enum per category? How are tags assigned — heuristic rules, AST analysis, LLM labeling?

- **Content:** What goes in a content node? Minimum: function/class body + signature + docstring. Extended: git commit messages that touched the symbol, associated test cases. Which are required at indexing time vs. lazily enriched?

- **Memory layers:** What node types map to each layer?
  - Episodic: commits, PRs, code-change events. What is the granularity — per-file, per-symbol, per-commit?
  - Semantic: exported API surface (interfaces, types, constants). How is "exported" defined across languages? How is the semantic layer kept distinct from episodic content for the same symbol?
  - Topic: modules and packages as architectural clusters. What determines cluster membership — directory structure, explicit annotation, LLM clustering?

- **Edge types:** What named relationships connect nodes? Minimum: `CUE_OF`, `TAGGED_WITH`, `PART_OF_TOPIC`, `REDIRECTS_TO` (for renames). Are edges typed and directional? Can a cue belong to multiple content nodes?

**Output of this step:** A graph schema document with node types, required properties per type, edge types, and at least two worked examples (one function, one module) showing the full subgraph that would be created.

---

## 2. Database abstraction interface

**Why it comes second:** The database is DuckDB (with the VSS extension for vector search). No component should ever call it directly — the interface must be defined before the indexer or tools are implemented, because both layers call into it. Its shape is fully determined by the schema from step 1.

**What must be decided:**

- **Interface boundaries:** What are the primitive operations the interface must expose? Candidates: `upsert_node`, `upsert_edge`, `vector_search(embedding, k)`, `neighbors(node_id, edge_type, direction)`, `get_node(id)`, `delete_node(id)`, `begin_transaction / commit / rollback`. Are transactions first-class in the interface, or handled differently per backend?

- **Vector search contract:** DuckDB's VSS extension provides ANN search via HNSW indexes. What does the interface contract promise — exact k-NN or approximate? What is the return type (node_id + score, or full node)?

- **Backend choice (resolved):** DuckDB. A single embedded file holds the relational tables (nodes, edges, checkpoint manifest) and, via the VSS extension, the vector index — keeping the prototype dependency-free while giving a richer query language than sqlite-vec for traversal queries.

**Output of this step:** A Python abstract base class (`GraphStore`) with all method signatures, return types, and documented contracts. A single DuckDB backend stub should be written (no implementation, just the class skeleton satisfying the interface).

---

## 3. `reconstruct(query, budget)` tool contract

**Why it comes third:** This is the primary interface between the MCP server and any consuming agent. The reconstruction algorithm, the tool's return type, and the trace format must all be specified before the server is built. The primitive tools (`search`, `traverse_forward`, `traverse_reverse`) are also defined as a byproduct of this step — their signatures follow from what `reconstruct` needs to call internally.

**What must be decided:**

- **Budget definition:** What does `budget` control — number of traversal steps, total nodes visited, wall-clock time, or a combination? What is a sensible default? What happens when budget is exhausted — return partial results or error?

- **Traversal policy:** The paper (Figure 9) shows sequential depth outperforms parallel breadth. What is the default traversal order — depth-first, best-first by score, or iterative deepening? How does the algorithm decide when to switch from forward to reverse traversal?

- **Evidence set format:** What does `reconstruct` return to the agent? Candidates: a list of content nodes with scores, a structured object with episodic / semantic / topic sections, a flat markdown summary. The format directly determines how useful the tool is to an agent with no knowledge of the graph.

- **Reconstruction trace:** Every call to `reconstruct` should log its traversal path. What is the structure of this trace — a list of `(step, node_id, edge_type, score)` tuples? Where is it written — log file, in-memory ring buffer, DB table? Is it included in the tool response (verbose mode) or only accessible via `mragent inspect`?

- **Primitive tool contracts:** Once `reconstruct` is defined, specify `search(query) → List[CueNode]`, `traverse_forward(cues, tags) → List[ContentNode]`, and `traverse_reverse(content_ids) → List[CueNode]` as explicit types. These are the debugging surface; their return types must be consistent with what `reconstruct` returns internally.

**Output of this step:** A tool contract document with function signatures, argument types, return types, error conditions, and a worked example showing a full `reconstruct` call trace for a concrete query against a hypothetical codebase.

---

## 4. Stale-handling strategy

**Why it comes fourth:** The indexer is the write path. Writing it without a stale-handling strategy produces code that must be replaced entirely when stale content accumulates. The decisions here affect both the graph schema (requires fields added in step 1) and the indexer logic.

**What must be decided:**

- **Node stamping:** Every node must carry `git_sha` (the commit SHA at which it was last indexed) and `indexed_at` (timestamp). These fields must be added to the schema from step 1 if not already present.

- **Symbol-level diffing:** When a file changes, the indexer must diff at the symbol level, not the file level. This requires the indexer to parse the old AST (from the last indexed SHA) and the new AST, compare exported symbols, and classify each change as: unchanged, modified, added, or deleted. What is the source of the "old AST" — stored in the DB, re-parsed from git history, or diffed from git's object store?

- **Deleted symbols:** A deleted function must not simply be removed from the graph. It becomes a tombstone node with `status: deleted`, retaining its cues and edges so agents asking about historical behavior still find it. What metadata does a tombstone carry — the last known content, the commit that deleted it, a `deleted_at` timestamp?

- **Renamed symbols:** A rename must produce a `REDIRECTS_TO` edge from the old cue node to the new one. Traversal must follow redirect edges transparently. How is a rename detected — by heuristic (same signature, different name), by git blame, or by explicit annotation?

- **Modified symbols:** A significantly changed implementation needs re-extraction of cues and tags, not just content replacement. What threshold defines "significant" — any change, a diff exceeding N lines, or LLM-judged semantic distance above a threshold?

- **Merge semantics:** The indexer must merge into the existing graph, not replace it. What is the merge algorithm — upsert by stable ID, upsert by (file_path, symbol_name), or content-addressed by hash?

**Output of this step:** A stale-handling spec with the merge algorithm, tombstone schema, redirect edge semantics, and a decision on how old ASTs are sourced for diffing.

---

## 5. Embedding model versioning

**Why it comes fifth:** Every node with a vector embedding is permanently tied to the model that produced it. This must be in the schema and the indexer from day one — adding it later requires re-reading every node to know which model was used.

**What must be decided:**

- **Per-node metadata:** Each node that carries an embedding must store `embedding_model` (e.g. `text-embedding-3-small`) and `embedding_model_version` (e.g. `2024-01`). These fields must be added to the schema from step 1.

- **Model selection:** Which embedding model is used at indexing time — configured globally, per-project, or per-node-type? Is there a default that works without API keys (local model via `sentence-transformers`)?

- **Migration command:** `mragent migrate-embeddings --to <model>` must be specified now even if implemented later. What does it do — re-embed all nodes, compare old and new embeddings, and update in a single transaction? Is re-indexing the whole codebase an acceptable alternative to per-node migration?

- **Cross-model search:** If a partial migration is in progress (or if two projects use different models), can vector search still work? The answer is almost certainly "no" — the interface should enforce that all nodes in a search index use the same model. The DB backend must validate this invariant at index creation.

**Output of this step:** An addendum to the schema document specifying the embedding metadata fields, the model selection config format, and the migration command signature.

---

## 6. Indexer atomicity

**Why it comes sixth:** The indexer is the only writer. If it crashes mid-file, the graph is in an inconsistent state with no recovery path. This must be designed before the indexer is implemented — adding transactional guarantees after the fact requires rewriting the entire write path.

**What must be decided:**

- **Transaction granularity:** What is the atomic unit — one file, one symbol, one commit? Per-file transactions are the practical choice: a file is small enough to fit in a single transaction, and the diff unit from git is already per-file.

- **Checkpoint manifest:** A persistent manifest records which file SHAs have been fully committed to the graph. On restart, the indexer skips files whose SHA appears in the manifest with `status: committed`. What is the manifest format — a dedicated DuckDB table, or a JSON file?

- **Partial write recovery:** If a transaction fails after some nodes are written but before commit, DuckDB never commits the transaction so rollback is automatic. Are there cases where auto-rollback is not sufficient (e.g. vector index updates that happen outside the transaction)?

- **Crash-safe manifest update:** The manifest entry for a file must be written atomically after the DB transaction commits, not before. What mechanism ensures this ordering — a two-phase write, a WAL-style approach, or treating the DB transaction as the manifest?

**Output of this step:** An indexer atomicity spec with the checkpoint manifest schema, the per-file transaction protocol, and the restart/recovery algorithm written in pseudocode.

---

## Summary: dependency order

```
1. Cue–Tag–Content schema for code
        ↓
2. GraphStore abstract interface
        ↓
3. reconstruct() tool contract + primitive tool signatures
        ↓
4. Stale-handling strategy      ←── depends on schema fields from (1)
        ↓
5. Embedding model versioning   ←── depends on schema fields from (1)
        ↓
6. Indexer atomicity            ←── depends on transaction model from (2)
```

None of these are implementation tasks. Each one produces a specification document or interface definition. The implementation plan cannot be written until all six exist, because the implementation of every component (indexer, GraphStore backends, MCP server, CLI) is a direct translation of these specifications into code.

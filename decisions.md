# Prototype Decisions — Sections 3–6

Decisions made for the bare-bones implementation. Sections 1 and 2 are settled separately in their own documents.

---

## Section 3 — `reconstruct(query, budget)` tool contract

- **Budget:** number of traversal hops (integer, default 3). Simplest control knob; easy to tune later.
- **Traversal order:** depth-first, sequential. The paper (Figure 9) shows this outperforms parallel breadth.
- **Return format:** flat list of `ContentNode` objects ordered by cumulative score. No episodic/semantic/topic sectioning for v1.
- **Reconstruction trace:** not included in v1. Add when debugging requires it.
- **Primitive signatures:**
  - `search(query: str) → List[CueNode]`
  - `traverse_forward(cue_ids: List[str], tag_filters: List[str]) → List[ContentNode]`
  - `traverse_reverse(content_ids: List[str]) → List[CueNode]`

---

## Section 4 — Stale-handling strategy

- **Approach:** when a file's `git_sha` changes, delete all nodes sourced from that file and re-index it from scratch.
- **No** symbol-level diffing, tombstones, rename detection, or merge logic for v1.
- **Required node fields:** every node must store `source_file` (file path) and `git_sha` (commit SHA at index time). These are the only fields needed to implement the delete-and-reindex strategy.

> ⚠️ **Revisit before any production use.** Delete-and-reindex is correct but does not scale — a large codebase with frequent commits will re-process entire files on every change. Symbol-level diffing with tombstone nodes is the right long-term approach (see Section 4 of foundations.md for the full spec).

---

## Section 5 — Embedding model versioning

- **Per-node field:** `embedding_model` (string) is mandatory on every node that carries a vector. Required from day one so future migration can identify which nodes need re-embedding.
- **Model selection:** configured via environment variable or config file. No hard-coded default — the choice must be explicit.
- **Migration:** not implemented for v1. If the model changes, re-index the codebase.

---

## Section 6 — Indexer atomicity

- **Atomic unit:** one file per transaction. Per-file is small enough to fit in a single transaction and matches the git diff unit.
- **Checkpoint manifest:** an `indexed_files` table in the same SQLite database.
  - Columns: `file_path TEXT`, `git_sha TEXT`, `indexed_at TIMESTAMP`
  - On startup: load this table and skip any file whose `(file_path, git_sha)` is already present.
- **Recovery:** if the indexer dies mid-file, the SQLite transaction is never committed, so nothing is written. On restart, the file is processed cleanly. No additional recovery logic needed.

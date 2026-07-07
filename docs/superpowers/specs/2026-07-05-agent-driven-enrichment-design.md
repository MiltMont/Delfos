# Agent-Driven Enrichment: Concept Cues and Semantic Tags

**Date:** 2026-07-05
**Status:** Approved

## Problem

The schema defines three cue types (`SYMBOL`, `CONCEPT`, `ERROR_MESSAGE`) and
five tag categories, but the indexer only emits `SYMBOL` and `ERROR_MESSAGE`
cues and the three mechanical tag categories (`LANGUAGE`, `MODULE_PATH`,
`LANG_CONSTRUCT`). `CONCEPT` cues and the two semantic tag categories
(`ARCH_LAYER`, `PATTERN_TYPE`) are never produced.

Concept cues are the core idea of the reconstruction model (arXiv 2606.06036):
they are the only entry points a symbol-ignorant agent can use. Without them,
vector search over cues degenerates into fuzzy symbol-name lookup, and Delfos
offers little over ripgrep + LSP. An agent asking *"where is stale handling
done?"* finds nothing unless a function happens to be named that.

## Decision

**Enrichment is agent-driven.** Delfos does not call a chat LLM at index time.
Instead, the MCP surface gains a write tool so the calling agent — which
already is the planner on the read path — supplies the semantics. This extends
the existing stance in `docs/decisions.md` ("the calling agent *is* the
planner; no server-side planner LLM runs") to the write path: **the calling
agent is the extractor.** Delfos never holds a chat-model API key.

Decisions made during design:

| Question | Decision |
|---|---|
| How does the agent participate? | New MCP write tool + prompt (not MCP sampling, not CLI batch LLM) |
| Tool surface shape | One `annotate` tool covering concepts + both tag categories |
| Tag vocabulary control | Open values, normalized, with a reuse nudge (echo existing values) |
| Annotations on file change | Die with the file via the existing delete-and-reindex rule |

## Architecture

### New component: `EnrichmentService` (`delfos/enrich/service.py`)

The write-path sibling of `ReconstructionService`. Constructed with the same
`GraphStore` and `Embedder` the MCP server already builds (search embeds
queries today, so enrichment needs no new configuration). All access goes
through the `GraphStore` boundary; the C++ engine is never touched directly.

Responsibilities:

- Validate the target `content_id` exists and is a `ContentNode`.
- Normalize concept phrases and tag values.
- Embed concept phrases with the store's embedding model.
- Upsert cue nodes, tag nodes, and edges in one transaction.
- Return a summary of what was written plus the existing vocabulary for
  `ARCH_LAYER` and `PATTERN_TYPE`.

### MCP surface additions (`delfos/mcp/server.py`)

**Tool `annotate`:**

```
annotate(
    content_id: str,
    concepts: list[str] = [],
    arch_layer: str | None = None,
    pattern_type: str | None = None,
) -> AnnotateResult
```

The parameter *names* `arch_layer` and `pattern_type` encode the tag
*category*: the service maps them to `TagCategory.ARCH_LAYER` /
`TagCategory.PATTERN_TYPE` internally, so agents can only write the two
semantic categories (the three mechanical ones remain indexer-owned). The
`str` is the tag *value*, which is an open string by the vocabulary decision
above — `TagNode.value` is untyped (`str`) in both the Pydantic schema and the
C++ engine; only the category is an enum.

`AnnotateResult` (a `delfos/mcp/views.py` view model) reports the cue/tag ids
written, phrases dropped by normalization, and
`existing_values: dict[str, list[str]]` for both semantic tag categories.
Calling `annotate` with only a `content_id` writes nothing and serves as a
cheap "show me the current vocabulary" query.

**Prompt `enrich`:** sibling of `reconstruct`. Teaches the discipline:

- Annotate after `fetch`, only code you genuinely understood.
- 1–5 concept phrases per node; phrases describe *what the code is about*
  ("rate limiting", "crash recovery"), not restate its name.
- Reuse an existing tag value unless none fits.

### Graph writes per `annotate` call

- One `CueNode` per accepted phrase: `cue_type=CONCEPT`,
  id `cue:concept:{source_file}::{sha1(normalized_phrase)[:12]}` (mirrors the
  error-cue scheme in `delfos/indexer/extractor.py`), embedded text, and a
  `CUE_OF` edge to the content node.
- `TagNode` per tag value, reused when it already exists
  (`tag:{category}:{value}`), plus a `TAGGED_WITH` edge from the content node.
  Tag nodes stay provenance-free and shared across files, as today.

### Provenance and staleness

Cue nodes and **all** new edges are stamped with the *target content node's*
`source_file` and `git_sha`. Because `delete_nodes_for_file` deletes by
`source_file`, annotations are wiped automatically when the file is
re-indexed — the "die with the file" semantics fall out of the existing
delete-and-reindex rule with zero new deletion logic. A concept extracted from
old code may be wrong for the new code; a stale concept cue is worse than a
missing one. Agents re-enrich on demand.

### Normalization

- Concept phrases: strip, collapse internal whitespace, lowercase. Reject
  empty results. A phrase that exactly equals the node's `symbol_name`
  (case-insensitive) is silently dropped — the `SYMBOL` cue already covers it.
- Tag values: lowercase, whitespace → hyphens. Reject empty results.

### Guardrails

- Max 10 concept phrases per call.
- Max 100 characters per phrase.
- Max 100 characters per tag value (`arch_layer` / `pattern_type`), checked after normalization.
- Idempotent: re-annotating produces the same ids; upserts, never duplicates.

The guardrails are a spam brake; the quality discipline lives in the `enrich`
prompt.

### Store addition

`GraphStore.list_tag_values(category: TagCategory) -> list[str]` to power the
vocabulary echo, added to the ABC (`delfos/store/base.py`) and
`NativeGraphStore`. The current native API exposes no node enumeration (only
`get_node` / `neighbors` / `vector_search`), so this requires one new C++
binding: a tag-node scan (e.g. `Store::list_tag_nodes()`) over the CSR graph in
`graph.hpp`, exposed via `libdelfos/bindings/py_delfos.cpp`, with a C++ unit
test alongside the existing ones in `libdelfos/tests/`. Python-side,
`NativeGraphStore.list_tag_values` filters the scan by category and returns
sorted distinct values.

### Read path

**Zero changes.** `search` already runs vector similarity over all cue nodes,
and tag filtering is categorical — `CONCEPT` cues and `ARCH_LAYER` /
`PATTERN_TYPE` tags light up existing machinery.

## Data flow

```
agent → annotate(content_id, concepts, arch_layer, pattern_type)
  1. validate content_id resolves to a ContentNode      (error if not)
  2. normalize phrases + tag values                      (drop/reject)
  3. embed accepted phrases in one batch call            (error → nothing written)
  4. transaction:
       upsert CueNodes (CONCEPT, embedded, provenance from target)
       upsert TagNodes (reuse existing)
       upsert CUE_OF + TAGGED_WITH edges (provenance from target)
  5. return AnnotateResult (written ids, dropped phrases, existing vocab)
```

Embedding happens *before* the transaction opens so an embedder failure
writes nothing.

## Error handling

- Unknown `content_id`, or an id resolving to a cue/tag node → tool error
  naming the id and its actual node type; nothing written.
- Embedder failure → tool error; nothing written.
- All phrases dropped by normalization → success with empty `written`,
  populated `dropped`, and the vocab echo (not an error).

## Testing

- **Unit:** normalization, id scheme, symbol-name filtering (pure functions).
- **Service** (fake embedder, real `NativeGraphStore`, following
  `tests/reconstruct/` conventions): round-trip, idempotency, provenance
  stamping, vocab echo, guardrails, error cases.
- **MCP** (following `tests/mcp/` conventions): tool registration, result
  shape, error surfaces.
- **Integration:**
  1. *Staleness* — index a file → annotate a node → modify the file →
     re-index → annotations are gone.
  2. *Retrieval win* — annotate a node with a concept phrase → `search` for
     that phrase returns the content node via the concept cue.

## Out of scope (v1)

- Batch `annotate_many` tool (add later if bulk enrichment proves common).
- CLI enrichment command.
- Server-side LLM extraction of any kind.
- Un-annotate / delete tool (re-index is the reset).
- Enrichment-coverage tracking or status reporting.

Each can be added later without changing this design.

## Documentation updates

- `docs/decisions.md`: new entry — enrichment is agent-driven; the calling
  agent is the extractor; annotations die with the file.
- `ARCHITECTURE.md`, `README.md`, `CLAUDE.md`: document the `annotate` tool,
  the `enrich` prompt, and the `EnrichmentService`.

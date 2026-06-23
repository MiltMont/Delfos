# Reconstruction Engine (Read Path) — Design

**Date:** 2026-06-23
**Status:** Approved (design); pending implementation plan
**Component:** `delfos/reconstruct/`

## Purpose

Implement the **read path** of Delfos: the active *reconstruction* model from
["Memory is Reconstructed, Not Retrieved"](https://arxiv.org/abs/2606.06036).
The indexer (write path) already turns a repo into a persisted Cue→Tag→Content
graph. This component makes that graph *readable* by agents, exposing the four
read-path operations Delfos is built around:

- `search(query) -> list[CueNode]` — vector search over cue nodes.
- `traverse_forward(cue_ids, tag_filters) -> list[ContentNode]` — expand cues to content.
- `traverse_reverse(content_ids) -> list[CueNode]` — discover sibling cues from content.
- `reconstruct(query, budget) -> list[ContentNode]` — the primary tool: an
  LLM-driven, depth-first traversal that reconstructs a relevant content set.

This is the keystone the MCP server and CLI sit on top of. Those layers are out
of scope here.

## Key decisions (resolved during brainstorming)

1. **Scope:** design the full read path (all four operations) as one cohesive
   layer, since `reconstruct` reuses the primitives.
2. **Traversal brain:** `reconstruct` is **LLM-in-the-loop** — an LLM decides
   each hop, faithful to the paper's Figure 9. (The primitives remain pure,
   LLM-free graph operations.)
3. **Hop decision:** *local pick-next + collect*. At each hop the LLM sees the
   current node and its immediate neighbors, and returns which neighbors to
   collect plus which single neighbor to descend into (DFS, with backtracking).
4. **Budget:** `budget` caps the **total number of planner (LLM) calls** across
   the whole walk. Backtracking spends budget too. Default 3.
5. **Result scoring:** **LLM-assigned relevance** — the planner emits a 0–1
   relevance for each collected node; results are ordered by relevance desc.
6. **Planner provider:** chosen at implementation-plan time. The brainstorm
   fixes only the provider-agnostic `HopPlanner` interface.

Smaller policy decisions:

- **Tag filters:** `reconstruct(query, budget, tag_filters=None)` — caller tags
  hard-constrain eligible content (mirroring `traverse_forward`); the LLM also
  sees tags as signal.
- **Tombstones / renames:** per the project's v1 "Key Decisions" (no tombstones
  or rename detection in v1), the walk filters to `status=ACTIVE`. Traversal
  follows a `REDIRECTS_TO` edge transparently *if present*, so it is
  forward-compatible at near-zero cost, even though the indexer emits none yet.

## Architecture & module layout

A read-path service layer that sits on top of `GraphStore` and **never touches
DuckDB directly** (same boundary as the indexer). It depends on the existing
`Embedder` (to embed the query) and a new `HopPlanner` abstraction.

```
delfos/reconstruct/
  __init__.py     # exports ReconstructionService
  service.py      # ReconstructionService: the 4 read-path methods
  planner.py      # HopPlanner Protocol + HopRequest / HopDecision / Candidate models
  summaries.py    # build the compact node summaries the planner sees
  planners/
    __init__.py
    fake.py       # FakeHopPlanner (scripted) — for tests
    # openai.py / anthropic.py added at plan time
```

`ReconstructionService` is constructed with a `GraphStore`, an `Embedder`, and a
`HopPlanner`. The three primitives are **pure, LLM-free** graph operations; only
`reconstruct` invokes the planner. This keeps the cheap, deterministic parts
independently testable and reusable.

## The primitives

All operate purely through `GraphStore`. `status=ACTIVE` nodes only.

- **`search(query, k=...) -> list[CueNode]`**: embed `query` via `Embedder`,
  call `store.vector_search(emb, k, node_type=CUE)`, hydrate results to
  `CueNode`s. No LLM.
- **`traverse_forward(cue_ids, tag_filters=None) -> list[ContentNode]`**: for
  each cue, follow `CUE_OF` *outgoing* to content; keep content whose
  `TAGGED_WITH` tags satisfy `tag_filters`. Follows `REDIRECTS_TO` transparently.
- **`traverse_reverse(content_ids) -> list[CueNode]`**: for each content node,
  follow `CUE_OF` *incoming* to discover sibling cues.

## The `reconstruct` algorithm (LLM-driven depth-first)

**Signature:** `reconstruct(query: str, budget: int = 3, tag_filters=None) -> list[ContentNode]`

**State:**
- `result` — `id -> (ContentNode, relevance)`, deduped, keeping max relevance.
- `stack` — DFS backtrack stack of visited positions.
- `visited` — set of expanded node ids (cycle guard).
- `budget_remaining` — total planner calls left.

**Seed:** `seeds = search(query, k=seed_k)` (config, e.g. 5). Current position =
best seed cue; remaining seeds held as backtrack alternatives.

**Loop** (while `budget_remaining > 0` and a position exists):

1. **Gather candidates** for current node X via the store:
   - X is a **Cue** → `CUE_OF` outgoing content (ACTIVE; follow `REDIRECTS_TO`).
   - X is **Content** → sibling cues (`CUE_OF` incoming) + topic peers
     (`PART_OF_TOPIC`); its tags travel along as signal. `tag_filters`
     hard-filter eligible content.
2. **Plan the hop:** build a `HopRequest` (query + current summary + candidate
   summaries) → `HopPlanner.decide` → `HopDecision`. Validate returned ids
   against the candidate set; drop hallucinations.
3. **Collect:** add collected **content** nodes to `result` with their
   relevance (if the planner collects a cue, resolve its `CUE_OF` content).
   `budget_remaining -= 1`.
4. **Move:** if `stop` → break. Else if `descend_into` is a valid, unvisited
   node → push X, descend. Else **backtrack**: pop the stack; if empty, pull the
   next unused seed; if none remain, break.

**Finalize:** order `result` by relevance desc (tie-break: reaching cue's
query-similarity); return `list[ContentNode]`.

The walk is strictly sequential — one expansion per planner call, deepening one
path before backtracking — and `budget` is a hard ceiling on planner calls.

## The `HopPlanner` interface

Provider-agnostic; concrete LLM chosen at plan time; tests inject a fake. All
Pydantic models use `extra="forbid"` (project convention).

```python
class CandidateSummary(BaseModel):      # what the planner sees per neighbor
    id: str
    node_kind: Literal["cue", "content"]
    label: str            # cue.text, or content symbol_name/signature
    snippet: str | None   # docstring or truncated body for content
    tags: list[str]       # "category=value" strings, as signal

class HopRequest(BaseModel):
    query: str
    current: CandidateSummary
    candidates: list[CandidateSummary]
    hops_remaining: int

class Collected(BaseModel):
    id: str
    relevance: float = Field(ge=0.0, le=1.0)

class HopDecision(BaseModel):
    collect: list[Collected]
    descend_into: str | None
    stop: bool

class HopPlanner(Protocol):
    def decide(self, request: HopRequest) -> HopDecision: ...
```

Concrete planners use function/tool-calling with `HopDecision` as the schema, so
model output is validated rather than free-text parsed. `summaries.py` owns the
node→`CandidateSummary` mapping — the single place that decides how much of each
node the LLM sees, keeping token cost controlled and prompt logic out of the
traversal loop.

## Error handling

- **No seed cues** → return `[]` (empty reconstruction, not an error).
- **Hallucinated ids** in `collect` / `descend_into` → filtered against the
  candidate set; an invalid `descend_into` is treated as a backtrack signal.
- **Planner call fails / raises** (network, rate limit) → abort the walk
  gracefully and return what is accumulated so far (a **partial
  reconstruction**), with the failure surfaced via a logged warning.
- **Cycles** → `visited` set prevents re-expansion.
- **Embedding-model mismatch** → already enforced by the store; surfaced as a
  clear error.

## Testing (no network, fully deterministic)

- **Primitives** (`search` / `traverse_forward` / `traverse_reverse`) → run
  against an in-memory `DuckDBGraphStore` seeded with a small fixture graph;
  assert exact node sets. The `Embedder` is stubbed with a fixed query vector.
- **`reconstruct`** → inject `FakeHopPlanner` with **scripted decisions**;
  assert: traversal order, budget enforcement (≤ N planner calls), dedup,
  relevance ordering, backtracking when a path dead-ends, hallucinated-id
  filtering, and partial-result-on-planner-error.
- Gives the repo its first non-store tests; everything stays pyright-strict.

## Out of scope

- MCP server layer (exposes these as tools) — separate milestone.
- CLI.
- Concrete provider SDK choice for the planner (decided at plan time).
- Historical/tombstone queries and rename detection (deferred per v1 decisions).

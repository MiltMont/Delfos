# libdelfos — Full Implementation Plan

> This document is the single source of truth for implementing `libdelfos`, the C++ graph core that replaces DuckDB in Delfos. Any agent or developer reading this should have complete context to implement the library without additional information.

---

## 1. Project Context

### What is Delfos?

Delfos is a graph-memory MCP server for codebases. It implements the **active reconstruction** model from [arXiv 2606.06036](https://arxiv.org/abs/2606.06036): memory access is iterative LLM-driven traversal of a persistent graph, not one-shot vector retrieval.

### What exists today (Python)

```
delfos/
├── schema/          # Pydantic models: CueNode, TagNode, ContentNode, Edge, enums
├── indexer/         # AST parser → extractor → embedder → pipeline
├── store/
│   ├── base.py      # GraphStore ABC (the interface everything talks to)
│   └── duckdb_store.py  # Current backend (DuckDB, brute-force cosine)
```

The indexer parses Python files, extracts a Cue-Tag-Content graph, embeds cue nodes via OpenAI, and persists everything through the `GraphStore` ABC.

### Why replace DuckDB with C++?

The read and write hot paths are **`GraphStore` primitives** — especially `vector_search`, `neighbors`, and `get_node`. In DuckDB, each call costs a full SQL round-trip (parse/plan/execute/deserialize). In a native C++ CSR graph, adjacency is a pointer dereference and vector search uses HNSW in-process. Measured difference for a 10K-file repo: **~50–200ms per DuckDB hop vs. < 1μs per C++ neighbor lookup** and **< 1ms for k-NN search**.

libdelfos does **not** own traversal orchestration. Future read-path code (MCP tools, reconstruction) stays in Python and calls these primitives through the unchanged `GraphStore` ABC.

### What libdelfos is

A **header-only C++ storage engine** that provides:

1. An in-memory directed property graph (CSR layout)
2. An HNSW vector index for cosine similarity search
3. FlatBuffers-based snapshot persistence
4. A thin Python extension (`_delfos`) used exclusively by `NativeGraphStore`

### Architectural boundaries

libdelfos is a **storage engine**, not a port of the Python product API.

```
┌─────────────────────────────────────────────────────────┐
│  Python product code (indexer, future MCP read path)    │
│  depends on GraphStore ABC only                         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  NativeGraphStore (delfos/store/native_store.py)        │
│  Pydantic ↔ native conversion, txn semantics, validation│
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  libdelfos (C++): Graph, VectorIndex, snapshot          │
│  engine-native API — indices, CSR, batch rebuild          │
└─────────────────────────────────────────────────────────┘
```

| Layer | Mimic Python? | Notes |
|---|---|---|
| **`GraphStore` boundary** (`native_store.py`) | **Yes — exactly** | Drop-in replacement for `DuckDBGraphStore`; must pass existing store tests |
| **C++ core** (`Graph`, `VectorIndex`, `snapshot`) | **No — optimize for the engine** | Flat `NodeData`, integer indices, batch `rebuild()` |
| **`_delfos` nanobind exports** | **Minimal** | Only what `NativeGraphStore` needs; not a second public graph API |

**Mimic at the boundary:** `GraphStore` method signatures, Pydantic schema field semantics, upsert/delete semantics, manifest shape.

**Differ internally:** unified node struct, `NodeIdx`/`EdgeIdx`, CSR adjacency, separate HNSW index, snapshot files instead of SQL.

**Out of scope for libdelfos:** query embedding, traversal orchestration, MCP tool logic, any product-level `reconstruct` implementation.

---

## 2. The Graph Schema (Source of Truth)

The C++ types must map exactly to the Python schema in `delfos/schema/`. Field semantics match; representation may differ (flat struct vs. discriminated union).

### 2.1 Enums

| C++ Enum | Values | Python Source |
|---|---|---|
| `NodeType` | `Cue`, `Tag`, `Content` | `schema/enums.py::NodeType` |
| `NodeStatus` | `Active`, `Deleted` | `schema/enums.py::NodeStatus` |
| `CueType` | `Symbol`, `Concept`, `ErrorMessage` | `schema/enums.py::CueType` |
| `TagCategory` | `ModulePath`, `ArchLayer`, `PatternType`, `LangConstruct`, `Language` | `schema/enums.py::TagCategory` |
| `ContentKind` | `Function`, `Class`, `Module`, `Commit`, `Test` | `schema/enums.py::ContentKind` |
| `MemoryLayer` | `Episodic`, `Semantic`, `Topic` | `schema/enums.py::MemoryLayer` |
| `EdgeType` | `CueOf`, `TaggedWith`, `PartOfTopic`, `RedirectsTo` | `schema/enums.py::EdgeType` |
| `Direction` | `Outgoing`, `Incoming` | `schema/enums.py::Direction` |

All enums are `uint8_t` backed.

### 2.2 NodeData (unified struct)

In Python there are three discriminated node types (`CueNode`, `TagNode`, `ContentNode`). In C++ we use a single struct with a `type` discriminator — fields irrelevant to a given type remain default-initialized.

Timestamps are stored as **Unix epoch microseconds** (`int64_t`) in C++; the binding layer converts to/from Python `datetime`.

```cpp
struct NodeData {
    // Identity & lifecycle (all nodes)
    std::string id;
    NodeType    type;
    NodeStatus  status = NodeStatus::Active;
    int64_t     indexed_at = 0;       // required on all nodes
    int64_t     deleted_at = 0;       // 0 = unset; set when status == Deleted
    std::string deleted_by_commit;    // optional tombstone metadata

    // Provenance (SourcedNode: Cue and Content; optional on Tag)
    std::string source_file;
    std::string git_sha;

    // Cue-specific (populated when type == Cue)
    CueType     cue_type{};
    std::string text;

    // Tag-specific (populated when type == Tag)
    TagCategory category{};
    std::string value;

    // Content-specific (populated when type == Content)
    ContentKind  kind{};
    MemoryLayer  memory_layer{};
    std::string  symbol_name;
    std::string  signature;
    std::string  docstring;
    std::string  body;

    // Embedding (Cue nodes usually; Content optionally)
    std::vector<float> embedding;
    std::string embedding_model;          // required when embedding non-empty
    std::string embedding_model_version;  // optional
};
```

### 2.3 EdgeData

Edges use integer indices internally. The Python `Edge` model uses string `source_id` / `target_id`; `NativeGraphStore` resolves those to `NodeIdx` before calling the C++ graph.

```cpp
struct EdgeData {
    NodeIdx     source;     // index into Graph's node array
    NodeIdx     target;     // index into Graph's node array
    EdgeType    type;
    std::string source_file;  // provenance (optional, nullable in Python)
    std::string git_sha;      // provenance (optional)
    int64_t     indexed_at = 0;
};
```

### 2.4 Node ID Format (how the Python indexer generates IDs)

The Python extractor (`delfos/indexer/extractor.py`) produces IDs in these formats:
- Content: `"content:{source_file}::{qualified_name}"` (e.g. `"content:delfos/store/base.py::GraphStore"`)
- Content (module-level): `"content:{source_file}::<module>"`
- Cue (symbol): `"cue:symbol:{source_file}::{qualified_name}"`
- Cue (error): `"cue:error:{source_file}::{sha1_slug_12}"`
- Tag: `"tag:{category}:{value}"` (e.g. `"tag:language:python"`)

These string IDs are the stable keys for upsert/lookup. The C++ graph must support O(1) lookup by string ID via `id_index_`, including while `dirty_ == true`.

### 2.5 Edge Semantics

| Edge Type | Source → Target | Meaning |
|---|---|---|
| `CueOf` | CueNode → ContentNode | "this cue is an entry point to this content" |
| `TaggedWith` | ContentNode → TagNode | "this content is categorized by this tag" |
| `PartOfTopic` | ContentNode → ContentNode | "this definition belongs to this module" |
| `RedirectsTo` | Node → Node | "this was renamed to that" (rename chains) |

### 2.6 Graph Shape Per File (produced by the indexer)

For a Python file `foo.py` containing class `Bar` with method `baz`:

```
CueNode("cue:symbol:foo.py::Bar")
    ──CUE_OF──► ContentNode("content:foo.py::Bar")    [kind=Class, layer=Semantic]
                    ──TAGGED_WITH──► TagNode("tag:language:python")
                    ──TAGGED_WITH──► TagNode("tag:module_path:foo")
                    ──TAGGED_WITH──► TagNode("tag:lang_construct:class")
                    ──PART_OF_TOPIC──► ContentNode("content:foo.py::<module>")  [kind=Module, layer=Topic]

CueNode("cue:symbol:foo.py::Bar.baz")
    ──CUE_OF──► ContentNode("content:foo.py::Bar.baz")  [kind=Function, layer=Semantic]
                    ──TAGGED_WITH──► TagNode("tag:language:python")
                    ──TAGGED_WITH──► TagNode("tag:module_path:foo")
                    ──TAGGED_WITH──► TagNode("tag:lang_construct:method")
                    ──PART_OF_TOPIC──► ContentNode("content:foo.py::<module>")
```

---

## 3. Architecture

### 3.1 Header-Only, No Compilation Units

The entire library lives in `libdelfos/include/delfos/`. There are no `.cpp` files for the library itself. Only benchmarks, tests, and Python bindings produce compilation units.

### 3.2 Header Dependency Graph (strict — no cycles, lower never includes higher)

```
types.hpp          ← leaf (no dependencies)
    ↑
node.hpp           ← depends on types.hpp
    ↑
edge.hpp           ← depends on types.hpp
    ↑
graph.hpp          ← depends on types.hpp, node.hpp, edge.hpp
    ↑
vector_index.hpp   ← depends on types.hpp, <usearch/index.hpp>
    ↑
snapshot.hpp       ← depends on graph.hpp, vector_index.hpp, <flatbuffers>
    ↑
delfos.hpp         ← umbrella, includes all above
```

### 3.3 Directory Layout

```
Delfos/
├── libdelfos/
│   ├── include/delfos/
│   │   ├── delfos.hpp            # umbrella include
│   │   ├── types.hpp             # enums, NodeIdx, EdgeIdx, constants
│   │   ├── node.hpp              # NodeData struct
│   │   ├── edge.hpp              # EdgeData struct
│   │   ├── graph.hpp             # CSR graph class
│   │   ├── vector_index.hpp      # HNSW wrapper (usearch)
│   │   └── snapshot.hpp          # FlatBuffers persistence
│   │
│   ├── flatbuffers/
│   │   ├── delfos.fbs            # FlatBuffers schema
│   │   └── delfos_generated.h    # generated, committed to repo
│   │
│   ├── bench/
│   │   ├── CMakeLists.txt
│   │   ├── bench_graph.cpp
│   │   └── bench_query.cpp       # neighbors + vector_search microbench
│   │
│   ├── tests/
│   │   ├── CMakeLists.txt
│   │   ├── test_graph.cpp
│   │   ├── test_vector_index.cpp
│   │   └── test_snapshot.cpp
│   │
│   └── bindings/
│       ├── CMakeLists.txt
│       └── py_delfos.cpp         # nanobind module (adapter surface only)
│
├── delfos/
│   └── store/
│       ├── base.py               # UNCHANGED — GraphStore ABC
│       ├── duckdb_store.py       # REMOVED in Phase 5
│       └── native_store.py       # NEW — wrapper around _delfos
│
├── CMakeLists.txt                # NEW — top-level
├── CMakePresets.json             # NEW — debug/profile/release presets
└── pyproject.toml                # MODIFIED — scikit-build-core backend
```

---

## 4. Component Specifications

### 4.1 `types.hpp`

**Purpose:** Foundation types used by all other headers.

```cpp
#pragma once
#include <cstdint>
#include <limits>

namespace delfos {

using NodeIdx = uint32_t;
using EdgeIdx = uint32_t;
inline constexpr NodeIdx INVALID_NODE = std::numeric_limits<NodeIdx>::max();
inline constexpr EdgeIdx INVALID_EDGE = std::numeric_limits<EdgeIdx>::max();

enum class NodeType : uint8_t { Cue, Tag, Content };
enum class NodeStatus : uint8_t { Active, Deleted };
enum class EdgeType : uint8_t { CueOf, TaggedWith, PartOfTopic, RedirectsTo };
enum class CueType : uint8_t { Symbol, Concept, ErrorMessage };
enum class TagCategory : uint8_t { ModulePath, ArchLayer, PatternType, LangConstruct, Language };
enum class ContentKind : uint8_t { Function, Class, Module, Commit, Test };
enum class MemoryLayer : uint8_t { Episodic, Semantic, Topic };
enum class Direction : uint8_t { Outgoing, Incoming };

} // namespace delfos
```

**Invariants:**
- All enums fit in `uint8_t` (< 256 variants).
- `INVALID_NODE` / `INVALID_EDGE` are sentinels, never used as real indices.

---

### 4.2 `node.hpp`

**Purpose:** The `NodeData` struct. See Section 2.2.

**Invariants:**
- A `NodeData` with `type == Cue` always has `cue_type` and `text` populated.
- A `NodeData` with `type == Tag` always has `category` and `value` populated.
- A `NodeData` with `type == Content` always has `kind`, `memory_layer`, and `body` populated.
- `id` is never empty.
- `indexed_at` is always set on write.
- `embedding` may be empty (no embedding yet) or have exactly `dim` floats.
- When `embedding` is non-empty, `embedding_model` must be non-empty.

---

### 4.3 `edge.hpp`

**Purpose:** The `EdgeData` struct. See Section 2.3.

**Invariants:**
- `source` and `target` are valid `NodeIdx` values (< `graph.node_count()` after rebuild).
- `type` is always set.
- `source_file` / `git_sha` may be empty for cross-file edges (`RedirectsTo`).

**Upsert key:** `(source, target, type)` — matches `GraphStore.upsert_edge`.

---

### 4.4 `graph.hpp`

**Purpose:** The core CSR graph. Handles storage, mutation, compaction, adjacency queries.

**Public Interface:**

```cpp
class Graph {
public:
    // ── Mutation (NOT thread-safe) ──────────────────────────────────────
    NodeIdx upsert_node(NodeData node);   // replace by id if present
    EdgeIdx upsert_edge(EdgeData edge);   // replace by (source, target, type)
    void    delete_node(NodeIdx idx);     // hard-delete node + incident edges
    void    delete_nodes_for_file(std::string_view source_file);
    void    rebuild();                    // compact tombstones + build CSR

    // ── ID lookup (valid while dirty_) ──────────────────────────────────
    NodeIdx find(std::string_view id) const;

    // ── Adjacency queries (require dirty_ == false) ───────────────────────
    const NodeData&      node(NodeIdx idx) const;
    const EdgeData&      edge(EdgeIdx idx) const;
    std::span<const EdgeIdx> outgoing(NodeIdx idx) const;
    std::span<const EdgeIdx> incoming(NodeIdx idx) const;
    std::vector<NodeIdx> neighbors(NodeIdx idx, Direction dir,
                                   EdgeType* type_filter = nullptr) const;
    size_t node_count() const;
    size_t edge_count() const;
};
```

**Upsert semantics:**
- `upsert_node`: if `id` exists, replace the slot in `nodes_` and update `id_index_` / `file_index_`. If new, append. Does **not** trigger `rebuild()`.
- `upsert_edge`: if `(source, target, type)` exists, replace; else append. Does **not** trigger `rebuild()`.
- Vector index updates are **not** handled here — `NativeGraphStore` calls `VectorIndex::insert` / `remove` alongside node upserts.

**Delete semantics (match `DuckDBGraphStore`):**
- `delete_node`: hard-remove the node and all incident edges immediately.
- `delete_nodes_for_file`: hard-remove every node with matching `source_file` and every edge where `edge.source_file == file`, or either endpoint is a node being removed. Must take effect **immediately** so same-transaction re-index can upsert the same IDs without conflict.
- Soft tombstones (`status == Deleted` via `upsert_node`) remain in storage until the next `rebuild()` compacts them.

**Internal Data:**

| Field | Type | Purpose |
|---|---|---|
| `nodes_` | `vector<NodeData>` | All node data, indexed by `NodeIdx` |
| `edges_` | `vector<EdgeData>` | All edge data, indexed by `EdgeIdx` |
| `out_offset_` | `vector<uint32_t>` | CSR row pointers for outgoing edges (size = nodes+1) |
| `out_edges_` | `vector<EdgeIdx>` | CSR column indices for outgoing edges |
| `in_offset_` | `vector<uint32_t>` | CSR row pointers for incoming edges |
| `in_edges_` | `vector<EdgeIdx>` | CSR column indices for incoming edges |
| `id_index_` | `unordered_map<string, NodeIdx>` | String ID → node index (kept current during mutation) |
| `file_index_` | `unordered_map<string, vector<NodeIdx>>` | Source file → node indices |
| `edge_key_index_` | `unordered_map<EdgeKey, EdgeIdx>` | `(source, target, type)` → edge index |
| `dirty_` | `bool` | True after mutation; false after `rebuild()` |

**Preconditions:**
- `find()` works regardless of `dirty_`.
- `outgoing()`, `incoming()`, `neighbors()` require `dirty_ == false`. Asserted in debug builds.
- During an open write transaction, callers may upsert/delete freely; adjacency is unavailable until `commit()` → `rebuild()`. This matches the indexer pattern (write a whole file, then commit).

**`rebuild()` Algorithm:**
1. Compact: remove nodes with `status == Deleted`, build `remap[old_idx] → new_idx`
2. Remap: update `edges_[i].source` and `edges_[i].target` via remap table
3. Remove orphaned edges (where source or target mapped to `INVALID_NODE`)
4. Build outgoing CSR: count per-source, prefix-sum, scatter
5. Build incoming CSR: count per-target, prefix-sum, scatter
6. Rebuild `id_index_`, `file_index_`, and `edge_key_index_` from compacted storage
7. Set `dirty_ = false`

---

### 4.5 `vector_index.hpp`

**Purpose:** HNSW vector index wrapping [USearch](https://github.com/unum-cloud/usearch).

**Public Interface:**

```cpp
struct VectorHit {
    NodeIdx node;
    float   score;  // cosine similarity ∈ [0, 1], higher = more similar
};

class VectorIndex {
public:
    explicit VectorIndex(size_t dim, size_t capacity = 100'000);
    void insert(NodeIdx node, NodeType type, std::span<const float> embedding);
    void remove(NodeIdx node);
    std::vector<VectorHit> search(std::span<const float> query, size_t k,
                                  NodeType* type_filter = nullptr) const;
    size_t size() const;
    size_t dim() const;

    // Persistence
    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);
};
```

**Invariants:**
- All embeddings must have exactly `dim` dimensions.
- `search()` returns results sorted by descending `score`.
- When `type_filter` is set, only nodes of that type are returned (implements `GraphStore.vector_search(..., node_type=...)`).
- `search()` is thread-safe (usearch guarantees concurrent reads).
- `insert()` / `remove()` are NOT thread-safe with concurrent reads.

**Implementation Notes:**
- USearch's `index_dense_t` with `metric_kind_t::cos_k` (cosine distance).
- Maintain bidirectional map: `NodeIdx ↔ usearch_key` (uint64_t auto-incrementing).
- Store `NodeType` per indexed node for post-filtering (or filter during insert by skipping non-matching types).
- `save()`/`load()` use USearch's native `index.save()` / `index.load()`.

---

### 4.6 `snapshot.hpp`

**Purpose:** Serialize/deserialize the graph + vector index + checkpoint manifest to disk.

**Public Interface:**

```cpp
struct ManifestEntry {
    std::string file_path;
    std::string git_sha;
    int64_t     indexed_at = 0;
};

namespace snapshot {
    void save(const std::filesystem::path& dir,
              const Graph& graph,
              const VectorIndex& vectors,
              const std::vector<ManifestEntry>& manifest);

    void load(const std::filesystem::path& dir,
              Graph& graph,
              VectorIndex& vectors,
              std::vector<ManifestEntry>& manifest);
}
```

**File Layout:**

```
snapshot_dir/
├── graph.fb           # FlatBuffers: all nodes + all edges
├── vectors.usearch    # USearch native index format
└── manifest.fb        # FlatBuffers: indexed file paths + git SHAs
```

**Atomicity:**
- `save()` writes to a temporary directory (e.g. `snapshot_dir.tmp.{pid}`), then atomically `rename()`s it over the target directory. If the process dies mid-write, the old snapshot remains intact.

**FlatBuffers Schema (`delfos.fbs`):**

```fbs
namespace delfos.fb;

enum NodeType : byte { Cue = 0, Tag, Content }
enum EdgeType : byte { CueOf = 0, TaggedWith, PartOfTopic, RedirectsTo }
enum NodeStatus : byte { Active = 0, Deleted }
enum CueType : byte { Symbol = 0, Concept, ErrorMessage }
enum TagCategory : byte { ModulePath = 0, ArchLayer, PatternType, LangConstruct, Language }
enum ContentKind : byte { Function = 0, Class, Module, Commit, Test }
enum MemoryLayer : byte { Episodic = 0, Semantic, Topic }

table Node {
    id: string;
    type: NodeType;
    status: NodeStatus;
    indexed_at: int64;
    deleted_at: int64;
    deleted_by_commit: string;
    source_file: string;
    git_sha: string;
    cue_type: CueType;
    text: string;
    category: TagCategory;
    value: string;
    kind: ContentKind;
    memory_layer: MemoryLayer;
    symbol_name: string;
    signature: string;
    docstring: string;
    body: string;
    embedding: [float];
    embedding_model: string;
    embedding_model_version: string;
}

table Edge {
    source: uint32;
    target: uint32;
    type: EdgeType;
    source_file: string;
    git_sha: string;
    indexed_at: int64;
}

table ManifestEntry {
    file_path: string;
    git_sha: string;
    indexed_at: int64;
}

table Snapshot {
    nodes: [Node];
    edges: [Edge];
    manifest: [ManifestEntry];
}

root_type Snapshot;
```

---

### 4.7 `delfos.hpp` (Umbrella)

```cpp
#pragma once
#include "delfos/types.hpp"
#include "delfos/node.hpp"
#include "delfos/edge.hpp"
#include "delfos/graph.hpp"
#include "delfos/vector_index.hpp"
#include "delfos/snapshot.hpp"
```

---

## 5. Python Integration

### 5.1 nanobind Module (`bindings/py_delfos.cpp`)

The `_delfos` module is an **implementation detail** of `NativeGraphStore`. Application code must not import it directly.

Exposed surface (minimal):

- Internal `Store` class bundling `Graph`, `VectorIndex`, manifest, and txn state — constructed with `(path, embedding_dim, embedding_model)`
- Methods mirroring what the adapter needs (not a raw public `Graph` API for product code)
- Enum values for conversion helpers
- `INVALID_NODE` constant

The binding layer converts between `NodeData` / `EdgeData` and Pydantic models. All embedding-model and dimension validation happens in `NativeGraphStore` (same rules as `DuckDBGraphStore`).

Optional: expose low-level `Graph` / `VectorIndex` behind a `DELFOS_DEBUG=1` build flag for C++ unit-test parity and microbenchmarks.

### 5.2 `native_store.py`

Implements `GraphStore` ABC (from `delfos/store/base.py`). Maps each ABC method to `_delfos` calls:

| ABC Method | Native Calls |
|---|---|
| `initialize()` | `snapshot::load()` if snapshot exists; else empty graph |
| `close()` | `snapshot::save()` if dirty |
| `upsert_node(node)` | validate embedding model/dim → `upsert_node` + `vectors.insert/remove` |
| `upsert_edge(edge)` | resolve string IDs → indices → `upsert_edge` |
| `delete_node(node_id)` | `find` → `delete_node` + `vectors.remove` |
| `delete_nodes_for_file(f)` | `delete_nodes_for_file(f)` + remove affected vectors |
| `get_node(id)` | `find(id)` → `node(idx)` → `to_pydantic()` |
| `neighbors(id, ...)` | `find` → `neighbors(...)` → convert (requires post-commit / rebuilt graph) |
| `vector_search(emb, k, node_type=...)` | `vectors.search(emb, k, type_filter=...)` → map indices to `VectorSearchResult` |
| `record_indexed_file(...)` | append/replace in internal manifest |
| `indexed_file_sha(f)` | lookup in manifest |
| `list_indexed_files()` | return full manifest as `IndexedFile` list |
| `begin_transaction()` | mark txn open; adjacency queries should not run mid-txn |
| `commit()` | `graph.rebuild()` + sync vector index keys if needed |
| `rollback()` | reload from last committed snapshot state |

**Parity requirement:** `NativeGraphStore` must pass the same test suite as `DuckDBGraphStore` (`tests/store/test_duckdb_store.py`, parametrized or duplicated as `test_native_store.py`).

### 5.3 `pyproject.toml` Changes

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.build-type = "Release"
wheel.packages = ["delfos"]
```

---

## 6. Build System

### 6.1 Top-Level `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)
project(delfos LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# USearch (header-only HNSW)
FetchContent_Declare(usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch
    GIT_TAG main
    GIT_SHALLOW ON)
FetchContent_MakeAvailable(usearch)

# FlatBuffers (for snapshot serialization)
FetchContent_Declare(flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers
    GIT_TAG v24.3.25
    GIT_SHALLOW ON)
set(FLATBUFFERS_BUILD_TESTS OFF)
FetchContent_MakeAvailable(flatbuffers)

# Interface library (header-only)
add_library(delfos_core INTERFACE)
target_include_directories(delfos_core INTERFACE libdelfos/include)
target_link_libraries(delfos_core INTERFACE usearch flatbuffers)

# Tests
option(DELFOS_BUILD_TESTS "Build unit tests" ON)
if(DELFOS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(libdelfos/tests)
endif()

# Benchmarks
option(DELFOS_BUILD_BENCH "Build benchmarks" ON)
if(DELFOS_BUILD_BENCH)
    add_subdirectory(libdelfos/bench)
endif()

# Python bindings (only when building wheel)
option(DELFOS_BUILD_BINDINGS "Build nanobind Python module" OFF)
if(DELFOS_BUILD_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG main
        GIT_SHALLOW ON)
    FetchContent_MakeAvailable(nanobind)
    add_subdirectory(libdelfos/bindings)
endif()
```

### 6.2 `CMakePresets.json`

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "debug",
      "generator": "Ninja",
      "binaryDir": "build/debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-fsanitize=address,undefined -fno-omit-frame-pointer"
      }
    },
    {
      "name": "profile",
      "generator": "Ninja",
      "binaryDir": "build/profile",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-fno-omit-frame-pointer -fxray-instrument"
      }
    },
    {
      "name": "release",
      "generator": "Ninja",
      "binaryDir": "build/release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-O3 -flto=thin -march=native"
      }
    }
  ]
}
```

---

## 7. Testing Strategy

### 7.1 Unit Tests (Catch2 or doctest)

| Test File | What It Covers |
|---|---|
| `test_graph.cpp` | upsert_node/edge, delete_node, delete_nodes_for_file, rebuild, find (while dirty), outgoing/incoming/neighbors (after rebuild), edge-type filtering, compaction, upsert replace semantics |
| `test_vector_index.cpp` | insert, remove, search accuracy, node_type filter, empty index, duplicate insert |
| `test_snapshot.cpp` | save/load round-trip fidelity, all schema fields including timestamps and embedding metadata, crash safety |

### 7.2 Benchmarks (nanobench)

| Benchmark | Target |
|---|---|
| `bench_graph_rebuild` | 10K nodes / 50K edges rebuild < 10ms |
| `bench_neighbors` | Single `neighbors()` call < 1μs |
| `bench_vector_search` | 50K vectors dim=1536, k=5 < 1ms |
| `bench_query_path` | 100 sequential `neighbors()` + 1 `vector_search` < 1ms total |
| `bench_snapshot_save` | 100K nodes serialize < 2s |
| `bench_snapshot_load` | 100K nodes deserialize < 1s |

### 7.3 Python Store Parity Tests

| Test File | What It Covers |
|---|---|
| `tests/store/test_native_store.py` | Same cases as `test_duckdb_store.py` against `NativeGraphStore` |

### 7.4 Sanitizers

All C++ tests run under ASan + UBSan in CI. TSan added when concurrency paths are exercised.

---

## 8. Concurrency Model

```
┌───────────────────────────────┐
│  MCP Server (read-only)       │
│  shared_ptr<const Graph>      │──── multiple threads can read concurrently
│  VectorIndex::search() is     │     (no locks)
│  thread-safe                  │
└───────────────┬───────────────┘
                │ atomic swap
┌───────────────┴───────────────┐
│  Indexer (single writer)      │
│  Builds new Graph + VectorIdx │──── runs in isolation
│  Calls rebuild()              │
│  Saves snapshot               │
│  Swaps shared_ptr atomically  │
└───────────────────────────────┘
```

- After `rebuild()`, the `Graph` is immutable. All adjacency query methods are `const`.
- The MCP server holds a `std::shared_ptr<const Graph>`. The indexer builds a completely new `Graph`, saves a snapshot, and does an atomic `std::atomic_store` of the new `shared_ptr`.
- No mutexes on the read path. Zero contention.

---

## 9. Constraints & Non-Goals

### Constraints
- **C++20 minimum.** Uses `std::span`, designated initializers, `std::erase_if`, `contains()`.
- **clang++ only.** The project standardizes on LLVM (see toolchain skill). GCC compatibility is not a goal.
- **No exceptions in query hot paths.** `graph.hpp` adjacency queries and `vector_index.hpp` search use asserts and return values.
- **`inline` on all function definitions.** Required for header-only ODR correctness.
- **Every vector in the index uses the same embedding model.** Enforced by `NativeGraphStore` at write time (same as `DuckDBGraphStore`).

### Non-Goals (explicitly out of scope)
- **Traversal orchestration** — no `reconstruct`, DFS, or MCP read-path logic in C++
- Multi-model embedding support (one model per store, same as DuckDB backend)
- Incremental CSR maintenance (batched rebuild is fine for write-rarely)
- GPU acceleration (future consideration, not now)
- Custom allocator (use standard allocator; profile first)
- Windows support (Linux-only for now; macOS is nice-to-have)
- Python < 3.12 support
- A second public Python graph API parallel to `GraphStore`

---

## 10. Phased Implementation

### Phase 1: Core Graph

**Files:** `types.hpp`, `node.hpp`, `edge.hpp`, `graph.hpp`, `CMakeLists.txt`, `CMakePresets.json`, `tests/test_graph.cpp`

**Acceptance:**
- All enums and node/edge fields map 1:1 to `delfos/schema/`
- `upsert_node` / `upsert_edge` replace-by-key semantics work correctly
- `delete_node` and `delete_nodes_for_file` hard-delete immediately
- `find()` works while `dirty_ == true`
- `rebuild()` builds CSR; `outgoing` / `incoming` / `neighbors` work after rebuild
- Edge-type filtering in `neighbors()` works
- `rebuild()` compacts soft-deleted (`status == Deleted`) nodes
- Compiles with `clang++ -std=c++20 -Wall -Wextra -Werror` — zero warnings
- ASan + UBSan clean on all tests

**PR scope:** Graph data structure only. No vector index, no persistence, no Python bindings.

---

### Phase 2: Vector Index

**Files:** `vector_index.hpp`, `tests/test_vector_index.cpp`, `bench/bench_vector.cpp`

**Depends on:** Phase 1

**Acceptance:**
- `VectorIndex(1536)` constructs successfully
- `insert` / `remove` / `search` work correctly
- `search(..., type_filter=Cue)` restricts results to cue nodes
- Search returns results sorted by descending cosine similarity
- Empty-index search returns empty vector (no crash)
- Removing a non-existent node is a no-op (no crash)
- Benchmark: 50K random dim=1536 vectors, search k=5, p99 < 1ms
- ASan clean

---

### Phase 3: Persistence

**Files:** `snapshot.hpp`, `flatbuffers/delfos.fbs`, `flatbuffers/delfos_generated.h`, `tests/test_snapshot.cpp`

**Depends on:** Phase 1 + 2

**Acceptance:**
- FlatBuffers schema compiles with `flatc --cpp`
- `snapshot::save()` writes three files atomically (graph.fb, vectors.usearch, manifest.fb)
- `snapshot::load()` populates Graph + VectorIndex + manifest
- Round-trip preserves all fields including `indexed_at`, tombstone metadata, and embedding model
- Partial write (kill process mid-save) does not corrupt existing snapshot
- Benchmark: 100K nodes save < 2s, load < 1s
- ASan clean

---

### Phase 4: Python Bindings & NativeGraphStore

**Files:** `bindings/py_delfos.cpp`, `bindings/CMakeLists.txt`, `delfos/store/native_store.py`, `tests/store/test_native_store.py`, updated `pyproject.toml`

**Depends on:** Phase 1 + 2 + 3

**Acceptance:**
- `import _delfos` succeeds after `pip install -e .`
- `NativeGraphStore` implements **all** `GraphStore` ABC methods
- Passes store parity tests (same coverage as `test_duckdb_store.py`)
- `Indexer(NativeGraphStore(...), embedder).index(repo)` succeeds end-to-end
- `pyright --strict` passes on `native_store.py`
- `uv build` produces a wheel

---

### Phase 5: Integration & DuckDB Removal

**Files:** Remove `duckdb_store.py`, remove `duckdb` from `pyproject.toml`, update `CLAUDE.md`

**Depends on:** Phase 4

**Acceptance:**
- `duckdb` gone from dependencies
- `delfos/store/duckdb_store.py` deleted
- Indexer + NativeGraphStore can index the Delfos repo itself end-to-end
- `pyright --strict` passes
- `ruff check .` passes
- End-to-end benchmark: index Delfos repo < 30s; `vector_search` + 100× `neighbors` < 5ms

---

## 11. Build & Test Commands

```bash
# Configure debug build (with sanitizers)
cmake --preset debug

# Build
cmake --build build/debug

# Run tests
ctest --test-dir build/debug --output-on-failure

# Run benchmarks (release build)
cmake --preset release
cmake --build build/release
./build/release/libdelfos/bench/bench_query

# Build Python wheel
uv build

# Lint + typecheck (Python side)
uv run ruff check .
uv run pyright

# Store parity tests
uv run pytest tests/store/test_native_store.py -v
```

---

## 12. Dependencies (External)

| Dependency | Version | How Fetched | License |
|---|---|---|---|
| [USearch](https://github.com/unum-cloud/usearch) | latest stable | CMake FetchContent | Apache 2.0 |
| [FlatBuffers](https://github.com/google/flatbuffers) | v24.3.25 | CMake FetchContent | Apache 2.0 |
| [nanobind](https://github.com/wjakob/nanobind) | v2.x | CMake FetchContent | BSD-3 |
| [Catch2](https://github.com/catchorg/Catch2) | v3.x | CMake FetchContent | BSL-1.0 |
| [nanobench](https://github.com/martinus/nanobench) | latest | Single header vendored | MIT |

All fetched at configure time. No system-level package installs needed beyond the LLVM toolchain (already in blueprint).

---

## 13. Glossary

| Term | Definition |
|---|---|
| **CSR** | Compressed Sparse Row — cache-friendly graph representation where edges for each node are stored contiguously |
| **HNSW** | Hierarchical Navigable Small World — approximate nearest neighbor algorithm with O(log n) query time |
| **Cue** | Entry-point node agents query by (function name, error message, concept) |
| **Tag** | Categorical bridge node connecting cues to content (language, module path, etc.) |
| **Content** | Terminal node containing actual code (function body, class, module) |
| **GraphStore** | The Python ABC all product code uses; libdelfos is its native backend |
| **Snapshot** | Serialized graph + vectors + manifest on disk |
| **Manifest** | Table of `(file_path, git_sha, indexed_at)` tracking which files have been indexed |
| **Tombstone** | A node with `status == Deleted` — compacted on `rebuild()`, distinct from hard-delete |

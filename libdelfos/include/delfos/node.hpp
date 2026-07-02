#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "delfos/types.hpp"

namespace delfos {

// Unified node struct — one discriminated type for all three Python node kinds
// (CueNode, TagNode, ContentNode). Fields irrelevant to a given type are
// default-initialised and must be ignored by callers.
//
// Timestamps are Unix epoch microseconds (int64_t); the Python binding layer
// converts to/from datetime objects.
struct NodeData {
    // ── Identity & lifecycle (all nodes) ──────────────────────────────────
    std::string id;
    NodeType    type{NodeType::Cue};
    NodeStatus  status{NodeStatus::Active};
    int64_t     indexed_at{0};
    int64_t     deleted_at{0};         // 0 = unset
    std::string deleted_by_commit;

    // ── Provenance (SourcedNode: Cue and Content; optional on Tag) ────────
    std::string source_file;
    std::string git_sha;

    // ── Cue-specific (populated when type == Cue) ─────────────────────────
    CueType     cue_type{CueType::Symbol};
    std::string text;

    // ── Tag-specific (populated when type == Tag) ─────────────────────────
    TagCategory category{TagCategory::Language};
    std::string value;

    // ── Content-specific (populated when type == Content) ─────────────────
    ContentKind kind{ContentKind::Function};
    MemoryLayer memory_layer{MemoryLayer::Semantic};
    std::string symbol_name;
    std::string signature;
    std::string docstring;
    std::string body;
    std::string scip_symbol;   // SCIP symbol FK (empty when no SCIP symbol)

    // ── Embedding (Cue nodes primarily; Content optionally) ───────────────
    // Stored as float64 (double) to match DuckDB precision and allow
    // lossless Python round-trips. Downcast to float32 when inserting
    // into VectorIndex (USearch uses float32 internally).
    std::vector<double> embedding;
    std::string         embedding_model;
    std::string         embedding_model_version;
};

} // namespace delfos

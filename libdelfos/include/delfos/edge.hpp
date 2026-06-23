#pragma once
#include <cstdint>
#include <string>

#include "delfos/types.hpp"

namespace delfos {

// Directed, typed edge between two nodes identified by integer indices.
//
// The Python Edge model uses string source_id/target_id; NativeGraphStore
// resolves those to NodeIdx before calling into the C++ graph.
//
// Upsert key: (source, target, type) — matches GraphStore.upsert_edge.
struct EdgeData {
    NodeIdx     source{INVALID_NODE};
    NodeIdx     target{INVALID_NODE};
    EdgeType    type{EdgeType::CueOf};
    std::string source_file;   // provenance (empty for cross-file REDIRECTS_TO)
    std::string git_sha;
    int64_t     indexed_at{0};
};

} // namespace delfos

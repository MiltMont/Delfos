#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "delfos/types.hpp"

namespace delfos {

struct NodeData {
    std::string id;
    NodeType type{};
    NodeStatus status = NodeStatus::Active;
    std::int64_t indexed_at = 0;
    std::int64_t deleted_at = 0;
    std::string deleted_by_commit;

    std::string source_file;
    std::string git_sha;

    CueType cue_type{};
    std::string text;

    TagCategory category{};
    std::string value;

    ContentKind kind{};
    MemoryLayer memory_layer{};
    std::string symbol_name;
    std::string signature;
    std::string docstring;
    std::string body;

    std::vector<float> embedding;
    std::string embedding_model;
    std::string embedding_model_version;
};

}  // namespace delfos

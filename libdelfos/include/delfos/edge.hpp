#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "delfos/types.hpp"

namespace delfos {

struct EdgeData {
    NodeIdx source = INVALID_NODE;
    NodeIdx target = INVALID_NODE;
    EdgeType type{};
    std::string source_file;
    std::string git_sha;
    std::int64_t indexed_at = 0;
};

struct EdgeKey {
    NodeIdx source = INVALID_NODE;
    NodeIdx target = INVALID_NODE;
    EdgeType type{};

    friend bool operator==(const EdgeKey& lhs, const EdgeKey& rhs) {
        return lhs.source == rhs.source && lhs.target == rhs.target && lhs.type == rhs.type;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const noexcept {
        const auto type = static_cast<std::size_t>(key.type);
        const auto src = static_cast<std::size_t>(key.source);
        const auto dst = static_cast<std::size_t>(key.target);
        return (src * 1315423911u) ^ (dst * 2654435761u) ^ (type + 0x9e3779b97f4a7c15ULL);
    }
};

}  // namespace delfos

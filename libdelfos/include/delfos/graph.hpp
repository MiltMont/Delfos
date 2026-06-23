#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "delfos/edge.hpp"
#include "delfos/node.hpp"
#include "delfos/types.hpp"

namespace delfos {

class Graph {
public:
    inline NodeIdx upsert_node(NodeData node) {
        const auto existing = id_index_.find(node.id);
        if (existing != id_index_.end()) {
            const auto idx = existing->second;
            if (idx < nodes_.size()) {
                erase_node_from_file_index_(idx, nodes_[idx].source_file);
                nodes_[idx] = std::move(node);
                add_node_to_file_index_(idx, nodes_[idx].source_file);
                dirty_ = true;
                return idx;
            }
        }

        const auto idx = static_cast<NodeIdx>(nodes_.size());
        id_index_.emplace(node.id, idx);
        nodes_.push_back(std::move(node));
        add_node_to_file_index_(idx, nodes_.back().source_file);
        dirty_ = true;
        return idx;
    }

    inline EdgeIdx upsert_edge(EdgeData edge) {
        const EdgeKey key{edge.source, edge.target, edge.type};
        const auto existing = edge_key_index_.find(key);
        if (existing != edge_key_index_.end()) {
            const auto idx = existing->second;
            if (idx < edges_.size()) {
                edges_[idx] = std::move(edge);
                dirty_ = true;
                return idx;
            }
        }

        const auto idx = static_cast<EdgeIdx>(edges_.size());
        edges_.push_back(std::move(edge));
        edge_key_index_[key] = idx;
        dirty_ = true;
        return idx;
    }

    inline void delete_node(NodeIdx idx) {
        if (idx >= nodes_.size()) {
            return;
        }

        auto& node = nodes_[idx];
        if (!node.id.empty()) {
            id_index_.erase(node.id);
        }
        erase_node_from_file_index_(idx, node.source_file);

        node.id.clear();
        node.source_file.clear();
        node.git_sha.clear();
        node.status = NodeStatus::Deleted;

        remove_incident_edges_(idx);
        dirty_ = true;
    }

    inline void delete_nodes_for_file(std::string_view source_file) {
        std::vector<NodeIdx> to_delete;
        if (const auto found = file_index_.find(std::string(source_file)); found != file_index_.end()) {
            to_delete = found->second;
        }

        for (const auto idx : to_delete) {
            delete_node(idx);
        }

        for (std::size_t i = 0; i < edges_.size();) {
            const auto& edge = edges_[i];
            const bool edge_matches_file = edge.source_file == source_file;
            bool endpoint_matches_file = false;
            if (edge.source < nodes_.size()) {
                endpoint_matches_file = endpoint_matches_file || (nodes_[edge.source].source_file == source_file);
            }
            if (edge.target < nodes_.size()) {
                endpoint_matches_file = endpoint_matches_file || (nodes_[edge.target].source_file == source_file);
            }
            if (edge_matches_file || endpoint_matches_file) {
                edges_.erase(edges_.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
        rebuild_edge_key_index_();
        dirty_ = true;
    }

    inline void rebuild() {
        std::vector<NodeData> compacted_nodes;
        compacted_nodes.reserve(nodes_.size());

        std::vector<NodeIdx> remap(nodes_.size(), INVALID_NODE);
        for (NodeIdx old_idx = 0; old_idx < nodes_.size(); ++old_idx) {
            const auto& node = nodes_[old_idx];
            const bool should_keep = node.status != NodeStatus::Deleted && !node.id.empty();
            if (!should_keep) {
                continue;
            }
            const auto new_idx = static_cast<NodeIdx>(compacted_nodes.size());
            remap[old_idx] = new_idx;
            compacted_nodes.push_back(node);
        }

        std::vector<EdgeData> compacted_edges;
        compacted_edges.reserve(edges_.size());
        for (const auto& edge : edges_) {
            if (edge.source >= remap.size() || edge.target >= remap.size()) {
                continue;
            }
            const auto new_source = remap[edge.source];
            const auto new_target = remap[edge.target];
            if (new_source == INVALID_NODE || new_target == INVALID_NODE) {
                continue;
            }
            EdgeData remapped = edge;
            remapped.source = new_source;
            remapped.target = new_target;
            compacted_edges.push_back(std::move(remapped));
        }

        nodes_ = std::move(compacted_nodes);
        edges_ = std::move(compacted_edges);

        id_index_.clear();
        file_index_.clear();
        for (NodeIdx idx = 0; idx < nodes_.size(); ++idx) {
            id_index_[nodes_[idx].id] = idx;
            add_node_to_file_index_(idx, nodes_[idx].source_file);
        }
        rebuild_edge_key_index_();

        const auto n = nodes_.size();
        out_offset_.assign(n + 1, 0);
        in_offset_.assign(n + 1, 0);

        for (const auto& edge : edges_) {
            if (edge.source < n) {
                ++out_offset_[edge.source + 1];
            }
            if (edge.target < n) {
                ++in_offset_[edge.target + 1];
            }
        }
        for (std::size_t i = 1; i < out_offset_.size(); ++i) {
            out_offset_[i] += out_offset_[i - 1];
            in_offset_[i] += in_offset_[i - 1];
        }

        out_edges_.assign(edges_.size(), INVALID_EDGE);
        in_edges_.assign(edges_.size(), INVALID_EDGE);
        auto out_write = out_offset_;
        auto in_write = in_offset_;
        for (EdgeIdx edge_idx = 0; edge_idx < edges_.size(); ++edge_idx) {
            const auto& edge = edges_[edge_idx];
            out_edges_[out_write[edge.source]++] = edge_idx;
            in_edges_[in_write[edge.target]++] = edge_idx;
        }

        dirty_ = false;
    }

    inline NodeIdx find(std::string_view id) const {
        const auto found = id_index_.find(std::string(id));
        if (found == id_index_.end()) {
            return INVALID_NODE;
        }
        return found->second;
    }

    inline const NodeData& node(NodeIdx idx) const {
        return nodes_.at(idx);
    }

    inline const EdgeData& edge(EdgeIdx idx) const {
        return edges_.at(idx);
    }

    inline std::span<const EdgeIdx> outgoing(NodeIdx idx) const {
        require_clean_();
        assert(idx < nodes_.size());
        const auto begin = out_offset_[idx];
        const auto end = out_offset_[idx + 1];
        return std::span<const EdgeIdx>(out_edges_.data() + begin, end - begin);
    }

    inline std::span<const EdgeIdx> incoming(NodeIdx idx) const {
        require_clean_();
        assert(idx < nodes_.size());
        const auto begin = in_offset_[idx];
        const auto end = in_offset_[idx + 1];
        return std::span<const EdgeIdx>(in_edges_.data() + begin, end - begin);
    }

    inline std::vector<NodeIdx> neighbors(
        NodeIdx idx,
        Direction dir,
        EdgeType* type_filter = nullptr
    ) const {
        require_clean_();
        assert(idx < nodes_.size());

        std::vector<NodeIdx> result;
        const auto edge_ids = (dir == Direction::Outgoing) ? outgoing(idx) : incoming(idx);
        result.reserve(edge_ids.size());
        for (const auto edge_idx : edge_ids) {
            const auto& e = edge(edge_idx);
            if (type_filter != nullptr && e.type != *type_filter) {
                continue;
            }
            result.push_back(dir == Direction::Outgoing ? e.target : e.source);
        }
        return result;
    }

    [[nodiscard]] inline std::size_t node_count() const {
        return nodes_.size();
    }

    [[nodiscard]] inline std::size_t edge_count() const {
        return edges_.size();
    }

private:
    inline void require_clean_() const {
        assert(!dirty_ && "Adjacency queries require rebuild() after mutation.");
    }

    inline void add_node_to_file_index_(NodeIdx idx, const std::string& file) {
        if (file.empty()) {
            return;
        }
        file_index_[file].push_back(idx);
    }

    inline void erase_node_from_file_index_(NodeIdx idx, const std::string& file) {
        if (file.empty()) {
            return;
        }
        const auto it = file_index_.find(file);
        if (it == file_index_.end()) {
            return;
        }
        auto& indices = it->second;
        indices.erase(
            std::remove(indices.begin(), indices.end(), idx),
            indices.end()
        );
        if (indices.empty()) {
            file_index_.erase(it);
        }
    }

    inline void remove_incident_edges_(NodeIdx idx) {
        for (std::size_t i = 0; i < edges_.size();) {
            const auto& e = edges_[i];
            if (e.source == idx || e.target == idx) {
                edges_.erase(edges_.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
        rebuild_edge_key_index_();
    }

    inline void rebuild_edge_key_index_() {
        edge_key_index_.clear();
        edge_key_index_.reserve(edges_.size());
        for (EdgeIdx idx = 0; idx < edges_.size(); ++idx) {
            const auto& edge = edges_[idx];
            edge_key_index_[EdgeKey{edge.source, edge.target, edge.type}] = idx;
        }
    }

    std::vector<NodeData> nodes_;
    std::vector<EdgeData> edges_;

    std::vector<std::uint32_t> out_offset_;
    std::vector<EdgeIdx> out_edges_;
    std::vector<std::uint32_t> in_offset_;
    std::vector<EdgeIdx> in_edges_;

    std::unordered_map<std::string, NodeIdx> id_index_;
    std::unordered_map<std::string, std::vector<NodeIdx>> file_index_;
    std::unordered_map<EdgeKey, EdgeIdx, EdgeKeyHash> edge_key_index_;

    bool dirty_ = true;
};

}  // namespace delfos

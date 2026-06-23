#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "delfos/edge.hpp"
#include "delfos/node.hpp"
#include "delfos/types.hpp"

namespace delfos {

// Key used to deduplicate edges: (source, target, type) is the natural upsert
// key matching GraphStore.upsert_edge semantics.
struct EdgeKey {
    NodeIdx  source;
    NodeIdx  target;
    EdgeType type;

    bool operator==(const EdgeKey&) const noexcept = default;
};

struct EdgeKeyHash {
    using is_transparent = void;

    std::size_t operator()(const EdgeKey& k) const noexcept {
        // FNV-style mixing — good distribution, no stdlib dependencies
        std::size_t h = std::hash<uint32_t>{}(k.source);
        h ^= std::hash<uint32_t>{}(k.target) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.type)) + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Graph — in-memory directed property graph with CSR adjacency.
//
// Mutation phase (dirty_ == true):
//   upsert_node / upsert_edge / delete_node / delete_nodes_for_file
//   find() works at all times (id_index_ is kept current).
//
// Query phase (dirty_ == false, after rebuild()):
//   outgoing / incoming / neighbors are valid pointer-dereference fast paths.
//
// Lifecycle per indexer transaction:
//   1. begin_transaction (caller)  → free to call upsert/delete
//   2. rebuild()                   → compact tombstones, build CSR
//   3. adjacency queries available
// ─────────────────────────────────────────────────────────────────────────────
class Graph {
public:
    // ── Mutation (NOT thread-safe) ─────────────────────────────────────────
    NodeIdx upsert_node(NodeData node);
    EdgeIdx upsert_edge(EdgeData edge);
    void    delete_node(NodeIdx idx);
    void    delete_nodes_for_file(std::string_view source_file);
    void    rebuild();

    // ── ID lookup (valid while dirty_) ────────────────────────────────────
    NodeIdx find(std::string_view id) const;

    // ── Accessors (always valid once index is in range) ───────────────────
    const NodeData& node(NodeIdx idx) const;
    const EdgeData& edge(EdgeIdx idx) const;

    // ── Adjacency (require dirty_ == false) ───────────────────────────────
    std::span<const EdgeIdx> outgoing(NodeIdx idx) const;
    std::span<const EdgeIdx> incoming(NodeIdx idx) const;

    // Returns neighbouring NodeIdx values (not edge indices).
    // type_filter is an optional edge-type predicate; pass nullptr to return
    // all neighbours regardless of edge type.
    std::vector<NodeIdx> neighbors(NodeIdx idx, Direction dir,
                                   EdgeType* type_filter = nullptr) const;

    // Counts — node_count is live nodes (soft-deleted until rebuild();
    // hard-deleted are removed immediately). edge_count is live edges only.
    std::size_t node_count() const noexcept { return id_index_.size(); }
    std::size_t edge_count() const noexcept { return edge_key_index_.size(); }

    bool dirty() const noexcept { return dirty_; }

private:
    // ── Storage ───────────────────────────────────────────────────────────
    std::vector<NodeData> nodes_;
    std::vector<EdgeData> edges_;

    // ── CSR (valid only when dirty_ == false) ─────────────────────────────
    std::vector<uint32_t> out_offset_; // size = node_count + 1
    std::vector<EdgeIdx>  out_edges_;
    std::vector<uint32_t> in_offset_;
    std::vector<EdgeIdx>  in_edges_;

    // ── Indices (maintained during mutation) ──────────────────────────────
    std::unordered_map<std::string, NodeIdx>               id_index_;
    std::unordered_map<std::string, std::vector<NodeIdx>>  file_index_;
    std::unordered_map<EdgeKey, EdgeIdx, EdgeKeyHash>      edge_key_index_;

    // Hard-deleted node slots — holes compacted away on rebuild().
    // Distinct from soft-delete (status == NodeStatus::Deleted via upsert_node).
    std::unordered_set<NodeIdx> hard_deleted_;

    bool dirty_{false};
};

// ─────────────────────────────────────────────────────────────────────────────
// Inline implementations
// ─────────────────────────────────────────────────────────────────────────────

inline NodeIdx Graph::upsert_node(NodeData node) {
    auto it = id_index_.find(node.id);
    if (it != id_index_.end()) {
        NodeIdx idx = it->second;

        // Update file_index_ only when source_file changes.
        if (nodes_[idx].source_file != node.source_file) {
            if (!nodes_[idx].source_file.empty()) {
                auto fit = file_index_.find(nodes_[idx].source_file);
                if (fit != file_index_.end()) {
                    std::erase(fit->second, idx);
                    if (fit->second.empty()) file_index_.erase(fit);
                }
            }
            if (!node.source_file.empty()) {
                file_index_[node.source_file].push_back(idx);
            }
        }

        nodes_[idx] = std::move(node);
        dirty_ = true;
        return idx;
    }

    NodeIdx idx = static_cast<NodeIdx>(nodes_.size());
    // Access node fields before moving.
    id_index_.emplace(node.id, idx);
    if (!node.source_file.empty()) {
        file_index_[node.source_file].push_back(idx);
    }
    nodes_.push_back(std::move(node));
    dirty_ = true;
    return idx;
}

inline EdgeIdx Graph::upsert_edge(EdgeData edge) {
    EdgeKey key{edge.source, edge.target, edge.type};
    auto it = edge_key_index_.find(key);
    if (it != edge_key_index_.end()) {
        EdgeIdx idx = it->second;
        edges_[idx] = std::move(edge);
        dirty_ = true;
        return idx;
    }
    EdgeIdx idx = static_cast<EdgeIdx>(edges_.size());
    edge_key_index_.emplace(key, idx);
    edges_.push_back(std::move(edge));
    dirty_ = true;
    return idx;
}

inline void Graph::delete_node(NodeIdx idx) {
    if (idx >= static_cast<NodeIdx>(nodes_.size()) || hard_deleted_.contains(idx)) {
        return; // already gone or out of range
    }

    // Remove from id lookup immediately so find() returns INVALID_NODE.
    id_index_.erase(nodes_[idx].id);

    // Remove from file index.
    if (!nodes_[idx].source_file.empty()) {
        auto fit = file_index_.find(nodes_[idx].source_file);
        if (fit != file_index_.end()) {
            std::erase(fit->second, idx);
            if (fit->second.empty()) file_index_.erase(fit);
        }
    }

    hard_deleted_.insert(idx);

    // Kill all incident edges.
    for (auto& e : edges_) {
        if (e.source == INVALID_NODE) continue;
        if (e.source == idx || e.target == idx) {
            edge_key_index_.erase({e.source, e.target, e.type});
            e.source = INVALID_NODE; // sentinel for dead edge
        }
    }

    dirty_ = true;
}

inline void Graph::delete_nodes_for_file(std::string_view source_file) {
    std::string file_str(source_file);

    // Collect node indices sourced from this file.
    std::unordered_set<NodeIdx> dying;
    if (auto it = file_index_.find(file_str); it != file_index_.end()) {
        for (NodeIdx idx : it->second) dying.insert(idx);
        file_index_.erase(it);
    }

    // Hard-delete each collected node.
    for (NodeIdx idx : dying) {
        id_index_.erase(nodes_[idx].id);
        hard_deleted_.insert(idx);
    }

    // Kill edges: provenance matches file, or either endpoint is being deleted.
    for (auto& e : edges_) {
        if (e.source == INVALID_NODE) continue;
        bool kill = (e.source_file == file_str)
                 || dying.contains(e.source)
                 || dying.contains(e.target);
        if (kill) {
            edge_key_index_.erase({e.source, e.target, e.type});
            e.source = INVALID_NODE;
        }
    }

    if (!dying.empty()) dirty_ = true;
}

inline void Graph::rebuild() {
    const std::size_t old_n = nodes_.size();

    // ── Step 1: Build remap table and compact nodes ────────────────────────
    // A node is removed if it was hard-deleted OR soft-deleted (status==Deleted).
    std::vector<NodeIdx> remap(old_n, INVALID_NODE);
    std::vector<NodeData> new_nodes;
    new_nodes.reserve(old_n);

    for (NodeIdx i = 0; i < static_cast<NodeIdx>(old_n); ++i) {
        if (hard_deleted_.contains(i)) continue;
        if (nodes_[i].status == NodeStatus::Deleted) continue;
        remap[i] = static_cast<NodeIdx>(new_nodes.size());
        new_nodes.push_back(std::move(nodes_[i]));
    }
    nodes_ = std::move(new_nodes);
    hard_deleted_.clear();

    // ── Step 2: Compact and remap edges ───────────────────────────────────
    std::vector<EdgeData> new_edges;
    new_edges.reserve(edges_.size());

    for (auto& e : edges_) {
        if (e.source == INVALID_NODE) continue; // hard-deleted edge
        // Both endpoints must have mapped to valid new indices.
        if (e.source >= old_n || e.target >= old_n) continue;
        NodeIdx ns = remap[e.source];
        NodeIdx nt = remap[e.target];
        if (ns == INVALID_NODE || nt == INVALID_NODE) continue;
        e.source = ns;
        e.target = nt;
        new_edges.push_back(std::move(e));
    }
    edges_ = std::move(new_edges);

    // ── Step 3: Rebuild all indices ───────────────────────────────────────
    id_index_.clear();
    file_index_.clear();
    edge_key_index_.clear();

    for (NodeIdx i = 0; i < static_cast<NodeIdx>(nodes_.size()); ++i) {
        id_index_.emplace(nodes_[i].id, i);
        if (!nodes_[i].source_file.empty()) {
            file_index_[nodes_[i].source_file].push_back(i);
        }
    }
    for (EdgeIdx i = 0; i < static_cast<EdgeIdx>(edges_.size()); ++i) {
        edge_key_index_.emplace(EdgeKey{edges_[i].source, edges_[i].target, edges_[i].type}, i);
    }

    // ── Step 4: Build CSR ─────────────────────────────────────────────────
    const std::size_t n = nodes_.size();
    const std::size_t m = edges_.size();

    // Outgoing
    out_offset_.assign(n + 1, 0u);
    for (const auto& e : edges_) out_offset_[e.source + 1]++;
    for (std::size_t i = 1; i <= n; ++i) out_offset_[i] += out_offset_[i - 1];
    out_edges_.resize(m);
    {
        std::vector<uint32_t> pos = out_offset_;
        for (EdgeIdx i = 0; i < static_cast<EdgeIdx>(m); ++i) {
            out_edges_[pos[edges_[i].source]++] = i;
        }
    }

    // Incoming
    in_offset_.assign(n + 1, 0u);
    for (const auto& e : edges_) in_offset_[e.target + 1]++;
    for (std::size_t i = 1; i <= n; ++i) in_offset_[i] += in_offset_[i - 1];
    in_edges_.resize(m);
    {
        std::vector<uint32_t> pos = in_offset_;
        for (EdgeIdx i = 0; i < static_cast<EdgeIdx>(m); ++i) {
            in_edges_[pos[edges_[i].target]++] = i;
        }
    }

    dirty_ = false;
}

inline NodeIdx Graph::find(std::string_view id) const {
    auto it = id_index_.find(std::string(id));
    return (it != id_index_.end()) ? it->second : INVALID_NODE;
}

inline const NodeData& Graph::node(NodeIdx idx) const {
    assert(idx < nodes_.size() && "node index out of range");
    assert(!hard_deleted_.contains(idx) && "accessing a hard-deleted node slot");
    return nodes_[idx];
}

inline const EdgeData& Graph::edge(EdgeIdx idx) const {
    assert(idx < edges_.size() && "edge index out of range");
    return edges_[idx];
}

inline std::span<const EdgeIdx> Graph::outgoing(NodeIdx idx) const {
    assert(!dirty_ && "outgoing() called on dirty graph — call rebuild() first");
    assert(idx < nodes_.size() && "node index out of range");
    return {out_edges_.data() + out_offset_[idx],
            out_edges_.data() + out_offset_[idx + 1]};
}

inline std::span<const EdgeIdx> Graph::incoming(NodeIdx idx) const {
    assert(!dirty_ && "incoming() called on dirty graph — call rebuild() first");
    assert(idx < nodes_.size() && "node index out of range");
    return {in_edges_.data() + in_offset_[idx],
            in_edges_.data() + in_offset_[idx + 1]};
}

inline std::vector<NodeIdx> Graph::neighbors(NodeIdx idx, Direction dir,
                                              EdgeType* type_filter) const {
    assert(!dirty_ && "neighbors() called on dirty graph — call rebuild() first");
    std::vector<NodeIdx> result;
    const auto edge_span = (dir == Direction::Outgoing) ? outgoing(idx) : incoming(idx);
    for (EdgeIdx ei : edge_span) {
        const EdgeData& e = edges_[ei];
        if (type_filter == nullptr || e.type == *type_filter) {
            result.push_back((dir == Direction::Outgoing) ? e.target : e.source);
        }
    }
    return result;
}

} // namespace delfos

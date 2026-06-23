#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// USearch: header-only HNSW library (fetched via FetchContent).
// Suppress warnings from third-party headers — we build our own code with
// -Wall -Wextra -Werror but cannot hold vendored libraries to the same standard.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-W#warnings"
#include <usearch/index_dense.hpp>
#include <usearch/index_plugins.hpp>
#pragma clang diagnostic pop

#include "delfos/types.hpp"

namespace delfos {

namespace us = unum::usearch;

// A single k-NN result from VectorIndex::search().
// score is cosine similarity ∈ [0, 1]; higher = more similar.
struct VectorHit {
    NodeIdx node;
    float   score;
};

// ─────────────────────────────────────────────────────────────────────────────
// VectorIndex — thin HNSW wrapper for cosine similarity search.
//
// Threading model (per plan section 8):
//   search() is thread-safe (USearch guarantees concurrent reads).
//   insert() / remove() are NOT thread-safe with concurrent reads or writes.
//
// Node identity:
//   NodeIdx (uint32_t) is used directly as the USearch uint64_t key.
//   After graph.rebuild() compacts indices, NativeGraphStore (Phase 4) is
//   responsible for removing stale keys and reinserting with new indices.
//
// Type filtering:
//   Each inserted vector records its NodeType.
//   search(query, k, type_filter) uses USearch's filtered_search() so that
//   non-matching nodes are skipped during the HNSW walk itself, not post-hoc.
// ─────────────────────────────────────────────────────────────────────────────
class VectorIndex {
public:
    // Construct an empty index for dim-dimensional float32 vectors.
    // capacity is the initial reservation hint (can grow dynamically).
    explicit VectorIndex(std::size_t dim, std::size_t capacity = 100'000)
        : dim_(dim)
        , index_(make_index(dim, capacity))
    {}

    // Insert (or replace) a vector for node.
    // embedding must have exactly dim() elements.
    // Both float32 and float64 overloads are provided; double is downcasted
    // to float32 before passing to USearch (which uses float32 internally).
    inline void insert(NodeIdx node, NodeType type, std::span<const double> embedding) {
        std::vector<float> f32(embedding.size());
        for (std::size_t i = 0; i < embedding.size(); ++i)
            f32[i] = static_cast<float>(embedding[i]);
        insert(node, type, std::span<const float>(f32));
    }

    inline void insert(NodeIdx node, NodeType type, std::span<const float> embedding) {
        assert(embedding.size() == dim_ && "embedding dimension mismatch");

        // If this node already has a vector, remove the old one first.
        if (node_types_.contains(node)) {
            auto res = index_.remove(static_cast<us::default_key_t>(node));
            (void)res; // ignore labeling_result_t; node may not actually be in index
        }

        auto res = index_.add(static_cast<us::default_key_t>(node), embedding.data());
        if (!res) {
            throw std::runtime_error(std::string("VectorIndex::insert failed: ") + res.error.what());
        }
        node_types_[node] = type;
    }

    // Remove the vector for node. No-op if node is not in the index.
    inline void remove(NodeIdx node) {
        auto it = node_types_.find(node);
        if (it == node_types_.end()) return; // not present — no-op

        index_.remove(static_cast<us::default_key_t>(node));
        node_types_.erase(it);
    }

    // double overload — converts to float32 before searching.
    inline std::vector<VectorHit> search(std::span<const double> query, std::size_t k,
                                         NodeType* type_filter = nullptr) const {
        std::vector<float> f32(query.size());
        for (std::size_t i = 0; i < query.size(); ++i)
            f32[i] = static_cast<float>(query[i]);
        return search(std::span<const float>(f32), k, type_filter);
    }

    // k-NN search. Returns up to k hits sorted by descending cosine similarity.
    // When type_filter is non-null, only nodes of that type are returned.
    // query must have exactly dim() elements.
    inline std::vector<VectorHit> search(std::span<const float> query, std::size_t k,
                                         NodeType* type_filter = nullptr) const {
        assert(query.size() == dim_ && "query dimension mismatch");

        if (index_.size() == 0 || k == 0) return {};

        std::size_t k_request = std::min(k, index_.size());

        if (type_filter == nullptr) {
            // No filtering — plain search.
            auto raw = index_.search(query.data(), k_request);
            return collect_hits(raw, k);
        }

        // Use USearch filtered_search so the HNSW walk itself skips
        // non-matching nodes — more accurate than post-hoc filtering.
        const NodeType wanted_type = *type_filter;
        auto predicate = [this, wanted_type](us::default_key_t key) -> bool {
            auto it = node_types_.find(static_cast<NodeIdx>(key));
            return it != node_types_.end() && it->second == wanted_type;
        };

        // Over-request to account for nodes filtered out during traversal.
        // We ask for all vectors and let the predicate prune; USearch still
        // only returns up to k_request that pass the predicate.
        std::size_t k_over = std::min(k_request * 4 + 16, index_.size());
        auto raw = index_.filtered_search(query.data(), k_over, predicate);
        return collect_hits(raw, k);
    }

    std::size_t size() const noexcept { return index_.size(); }
    std::size_t dim()  const noexcept { return dim_; }

    // Persistence — USearch native binary format (.usearch file).
    inline void save(const std::filesystem::path& path) const {
        auto res = index_.save(path.string().c_str());
        if (!res) {
            throw std::runtime_error(std::string("VectorIndex::save failed: ") + res.error.what());
        }
    }

    inline void load(const std::filesystem::path& path) {
        auto loaded = us::index_dense_t::make(path.string().c_str());
        if (!loaded) {
            throw std::runtime_error(std::string("VectorIndex::load failed: ") + loaded.error.what());
        }
        index_ = std::move(loaded.index);
        dim_   = index_.dimensions();
        // Note: node_types_ is NOT serialized in the .usearch file —
        // it is the caller's responsibility to restore node types after load.
        // NativeGraphStore (Phase 4) rebuilds node_types_ from the graph on load.
        node_types_.clear();
    }

    // Direct access to the per-node type map (needed by NativeGraphStore
    // to restore node_types_ after snapshot load).
    std::unordered_map<NodeIdx, NodeType>& node_type_map() noexcept { return node_types_; }
    const std::unordered_map<NodeIdx, NodeType>& node_type_map() const noexcept { return node_types_; }

private:
    std::size_t dim_;
    us::index_dense_t index_;
    std::unordered_map<NodeIdx, NodeType> node_types_;

    // Factory helper — avoids repeating metric construction.
    static us::index_dense_t make_index(std::size_t dim, std::size_t capacity) {
        us::metric_punned_t metric{dim, us::metric_kind_t::cos_k, us::scalar_kind_t::f32_k};
        auto result = us::index_dense_t::make(metric);
        if (!result) {
            throw std::runtime_error(
                std::string("VectorIndex: failed to create USearch index: ") + result.error.what());
        }
        us::index_dense_t idx = std::move(result.index);
        us::index_limits_t limits;
        limits.members = capacity;
        idx.reserve(limits);
        return idx;
    }

    // Convert a raw USearch search_result_t to VectorHit vector.
    // USearch returns hits in ascending distance order (closest first),
    // which is also descending similarity order — exactly what we want.
    template <typename SearchResult>
    static std::vector<VectorHit> collect_hits(const SearchResult& raw, std::size_t k) {
        std::vector<VectorHit> hits;
        hits.reserve(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            auto match = raw[i];
            float score = 1.0f - match.distance; // distance = 1 - cos_similarity
            hits.push_back({static_cast<NodeIdx>(match.member.key), score});
        }
        // Trim to k in case we got more (filtered_search over-requested).
        if (hits.size() > k) hits.resize(k);
        return hits;
    }
};

} // namespace delfos

// nanobind Python extension module for libdelfos.
// Imported by NativeGraphStore as: from delfos._delfos import Store, NodeData, ...
//
// This file defines:
//   - NodeData  — Python-accessible struct mirroring delfos::NodeData
//   - Store     — bundles Graph + VectorIndex + manifest + txn state
//   - Integer constants for all delfos:: enums (NODE_TYPE_CUE, etc.)
//   - INVALID_NODE constant

#include <cassert>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-W#warnings"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#pragma clang diagnostic pop

#include "delfos/delfos.hpp"

namespace nb = nanobind;
using namespace delfos;

// ─────────────────────────────────────────────────────────────────────────────
// Store — Python-facing bundle of Graph + VectorIndex + manifest + txn state.
//
// NativeGraphStore (Python) never touches Graph/VectorIndex/snapshot directly;
// everything goes through this class.
// ─────────────────────────────────────────────────────────────────────────────
struct Store {
    std::filesystem::path path_;
    std::size_t           embedding_dim_;
    std::string           embedding_model_;
    Graph                 graph_;
    VectorIndex           vectors_;
    std::vector<ManifestEntry> manifest_;
    bool                  in_txn_{false};
    bool                  dirty_{false};  // any write since last snapshot save

    Store(std::string path_str, std::size_t dim, std::string model)
        : path_(std::move(path_str))
        , embedding_dim_(dim)
        , embedding_model_(std::move(model))
        , vectors_(dim)
    {}

    // ── Lifecycle ────────────────────────────────────────────────────────────

    void initialize() {
        graph_    = Graph{};
        vectors_  = VectorIndex{embedding_dim_};
        manifest_.clear();
        in_txn_   = false;
        dirty_    = false;
        if (std::filesystem::exists(path_ / "graph.fb")) {
            snapshot::load(path_, graph_, vectors_, manifest_);
        }
    }

    void close() {
        if (in_txn_) {
            // Uncommitted transaction: discard in-memory state without saving.
            in_txn_ = false;
            dirty_  = false;
            return;
        }
        if (dirty_) {
            _save();
        }
    }

    // ── Transactions ─────────────────────────────────────────────────────────

    void begin_transaction() {
        if (in_txn_) throw std::runtime_error("transaction already open");
        in_txn_ = true;
    }

    void commit() {
        if (!in_txn_) throw std::runtime_error("no open transaction");
        _rebuild_if_dirty();
        _save();
        in_txn_ = false;
        dirty_  = false;
    }

    void rollback() {
        if (!in_txn_) throw std::runtime_error("no open transaction");
        // Restore to last committed snapshot (or empty if none exists).
        graph_   = Graph{};
        vectors_ = VectorIndex{embedding_dim_};
        manifest_.clear();
        if (std::filesystem::exists(path_ / "graph.fb")) {
            snapshot::load(path_, graph_, vectors_, manifest_);
        }
        in_txn_ = false;
        dirty_  = false;
    }

    // ── Node / Edge writes ───────────────────────────────────────────────────

    void upsert_node(const NodeData& node) {
        NodeIdx idx = graph_.upsert_node(node);
        if (!node.embedding.empty()) {
            vectors_.insert(idx, node.type, node.embedding);
        } else {
            // If there was a previous embedding, remove it.
            vectors_.remove(idx);
        }
        dirty_ = true;
    }

    // Edge upsert by string IDs (NativeGraphStore resolves IDs externally).
    void upsert_edge(const std::string& source_id, const std::string& target_id,
                     int edge_type_int, const std::string& source_file,
                     const std::string& git_sha, int64_t indexed_at) {
        _rebuild_if_dirty();
        NodeIdx src = graph_.find(source_id);
        NodeIdx tgt = graph_.find(target_id);
        if (src == INVALID_NODE || tgt == INVALID_NODE) {
            throw std::runtime_error(
                "upsert_edge: source or target node not found: " +
                source_id + " -> " + target_id);
        }
        EdgeData e;
        e.source      = src;
        e.target      = tgt;
        e.type        = static_cast<EdgeType>(static_cast<uint8_t>(edge_type_int));
        e.source_file = source_file;
        e.git_sha     = git_sha;
        e.indexed_at  = indexed_at;
        graph_.upsert_edge(e);
        dirty_ = true;
    }

    void delete_node(const std::string& node_id) {
        NodeIdx idx = graph_.find(node_id);
        if (idx == INVALID_NODE) return;
        vectors_.remove(idx);
        graph_.delete_node(idx);
        dirty_ = true;
    }

    void delete_nodes_for_file(const std::string& source_file) {
        // Collect indices before deletion so we can clean up VectorIndex.
        for (NodeIdx idx : graph_.nodes_for_file(source_file)) {
            vectors_.remove(idx);
        }
        graph_.delete_nodes_for_file(source_file);
        dirty_ = true;
    }

    // ── Reads ────────────────────────────────────────────────────────────────

    std::optional<NodeData> get_node(const std::string& node_id) {
        NodeIdx idx = graph_.find(node_id);
        if (idx == INVALID_NODE) return std::nullopt;
        // Hard-deleted slots have been removed from id_index_ immediately,
        // so find() returning a valid index means the node is live.
        return graph_.node(idx);
    }

    std::vector<NodeData> neighbors(const std::string& node_id,
                                    int direction_int,
                                    std::optional<int> edge_type_int) {
        _rebuild_if_dirty();
        NodeIdx idx = graph_.find(node_id);
        if (idx == INVALID_NODE) return {};

        Direction dir = static_cast<Direction>(static_cast<uint8_t>(direction_int));
        EdgeType* tf  = nullptr;
        EdgeType  et;
        if (edge_type_int) {
            et = static_cast<EdgeType>(static_cast<uint8_t>(*edge_type_int));
            tf = &et;
        }
        auto neighbor_idxs = graph_.neighbors(idx, dir, tf);

        std::vector<NodeData> result;
        result.reserve(neighbor_idxs.size());
        for (NodeIdx ni : neighbor_idxs) {
            result.push_back(graph_.node(ni));
        }
        return result;
    }

    // Returns (node_id, score) pairs; filters out soft-deleted nodes.
    std::vector<std::pair<std::string, float>> vector_search(
            const std::vector<float>& embedding, std::size_t k,
            std::optional<int> node_type_int) {
        _rebuild_if_dirty();
        NodeType* tf = nullptr;
        NodeType  nt;
        if (node_type_int) {
            nt = static_cast<NodeType>(static_cast<uint8_t>(*node_type_int));
            tf = &nt;
        }
        auto hits = vectors_.search(embedding, k, tf);

        std::vector<std::pair<std::string, float>> results;
        results.reserve(hits.size());
        for (const auto& hit : hits) {
            if (hit.node < static_cast<NodeIdx>(graph_.node_count())) {
                const NodeData& n = graph_.node(hit.node);
                if (n.status == NodeStatus::Active) {
                    results.emplace_back(n.id, hit.score);
                }
            }
        }
        return results;
    }

    // ── Manifest ─────────────────────────────────────────────────────────────

    void record_indexed_file(const std::string& file_path,
                             const std::string& git_sha,
                             int64_t indexed_at) {
        for (auto& e : manifest_) {
            if (e.file_path == file_path) {
                e.git_sha    = git_sha;
                e.indexed_at = indexed_at;
                dirty_       = true;
                return;
            }
        }
        manifest_.push_back({file_path, git_sha, indexed_at});
        dirty_ = true;
    }

    std::optional<std::string> indexed_file_sha(const std::string& file_path) {
        for (const auto& e : manifest_) {
            if (e.file_path == file_path) return e.git_sha;
        }
        return std::nullopt;
    }

    // Returns list of (file_path, git_sha, indexed_at_us) triples.
    std::vector<std::tuple<std::string, std::string, int64_t>> list_indexed_files() {
        std::vector<std::tuple<std::string, std::string, int64_t>> result;
        result.reserve(manifest_.size());
        for (const auto& e : manifest_) {
            result.emplace_back(e.file_path, e.git_sha, e.indexed_at);
        }
        return result;
    }

    // ── Accessors used by NativeGraphStore for config validation ─────────────
    std::size_t embedding_dim()   const noexcept { return embedding_dim_; }
    std::string embedding_model() const          { return embedding_model_; }

private:
    void _rebuild_if_dirty() {
        if (graph_.dirty()) graph_.rebuild();
    }

    void _save() {
        if (graph_.dirty()) graph_.rebuild();
        std::filesystem::create_directories(path_);
        snapshot::save(path_, graph_, vectors_, manifest_);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// nanobind module definition
// ─────────────────────────────────────────────────────────────────────────────

NB_MODULE(_delfos, m) {
    m.doc() = "libdelfos Python bindings — implementation detail of NativeGraphStore.";

    // ── Enum integer constants ────────────────────────────────────────────────
    // NodeType
    m.attr("NODE_TYPE_CUE")     = static_cast<int>(NodeType::Cue);
    m.attr("NODE_TYPE_TAG")     = static_cast<int>(NodeType::Tag);
    m.attr("NODE_TYPE_CONTENT") = static_cast<int>(NodeType::Content);

    // NodeStatus
    m.attr("NODE_STATUS_ACTIVE")  = static_cast<int>(NodeStatus::Active);
    m.attr("NODE_STATUS_DELETED") = static_cast<int>(NodeStatus::Deleted);

    // EdgeType
    m.attr("EDGE_TYPE_CUE_OF")         = static_cast<int>(EdgeType::CueOf);
    m.attr("EDGE_TYPE_TAGGED_WITH")    = static_cast<int>(EdgeType::TaggedWith);
    m.attr("EDGE_TYPE_PART_OF_TOPIC")  = static_cast<int>(EdgeType::PartOfTopic);
    m.attr("EDGE_TYPE_REDIRECTS_TO")   = static_cast<int>(EdgeType::RedirectsTo);

    // CueType
    m.attr("CUE_TYPE_SYMBOL")        = static_cast<int>(CueType::Symbol);
    m.attr("CUE_TYPE_CONCEPT")       = static_cast<int>(CueType::Concept);
    m.attr("CUE_TYPE_ERROR_MESSAGE") = static_cast<int>(CueType::ErrorMessage);

    // TagCategory
    m.attr("TAG_CATEGORY_MODULE_PATH")    = static_cast<int>(TagCategory::ModulePath);
    m.attr("TAG_CATEGORY_ARCH_LAYER")     = static_cast<int>(TagCategory::ArchLayer);
    m.attr("TAG_CATEGORY_PATTERN_TYPE")   = static_cast<int>(TagCategory::PatternType);
    m.attr("TAG_CATEGORY_LANG_CONSTRUCT") = static_cast<int>(TagCategory::LangConstruct);
    m.attr("TAG_CATEGORY_LANGUAGE")       = static_cast<int>(TagCategory::Language);

    // ContentKind
    m.attr("CONTENT_KIND_FUNCTION") = static_cast<int>(ContentKind::Function);
    m.attr("CONTENT_KIND_CLASS")    = static_cast<int>(ContentKind::Class);
    m.attr("CONTENT_KIND_MODULE")   = static_cast<int>(ContentKind::Module);
    m.attr("CONTENT_KIND_COMMIT")   = static_cast<int>(ContentKind::Commit);
    m.attr("CONTENT_KIND_TEST")     = static_cast<int>(ContentKind::Test);

    // MemoryLayer
    m.attr("MEMORY_LAYER_EPISODIC") = static_cast<int>(MemoryLayer::Episodic);
    m.attr("MEMORY_LAYER_SEMANTIC") = static_cast<int>(MemoryLayer::Semantic);
    m.attr("MEMORY_LAYER_TOPIC")    = static_cast<int>(MemoryLayer::Topic);

    // Direction
    m.attr("DIRECTION_OUTGOING") = static_cast<int>(Direction::Outgoing);
    m.attr("DIRECTION_INCOMING") = static_cast<int>(Direction::Incoming);

    // Sentinel
    m.attr("INVALID_NODE") = static_cast<uint32_t>(INVALID_NODE);

    // ── NodeData class ────────────────────────────────────────────────────────
    nb::class_<NodeData>(m, "NodeData")
        .def(nb::init<>())
        // Identity & lifecycle
        .def_rw("id",          &NodeData::id)
        .def_prop_rw("type",
            [](const NodeData& n){ return static_cast<int>(n.type); },
            [](NodeData& n, int v){ n.type = static_cast<NodeType>(v); })
        .def_prop_rw("status",
            [](const NodeData& n){ return static_cast<int>(n.status); },
            [](NodeData& n, int v){ n.status = static_cast<NodeStatus>(v); })
        .def_rw("indexed_at",        &NodeData::indexed_at)
        .def_rw("deleted_at",        &NodeData::deleted_at)
        .def_rw("deleted_by_commit", &NodeData::deleted_by_commit)
        // Provenance
        .def_rw("source_file", &NodeData::source_file)
        .def_rw("git_sha",     &NodeData::git_sha)
        // Cue-specific
        .def_prop_rw("cue_type",
            [](const NodeData& n){ return static_cast<int>(n.cue_type); },
            [](NodeData& n, int v){ n.cue_type = static_cast<CueType>(v); })
        .def_rw("text", &NodeData::text)
        // Tag-specific
        .def_prop_rw("category",
            [](const NodeData& n){ return static_cast<int>(n.category); },
            [](NodeData& n, int v){ n.category = static_cast<TagCategory>(v); })
        .def_rw("value", &NodeData::value)
        // Content-specific
        .def_prop_rw("kind",
            [](const NodeData& n){ return static_cast<int>(n.kind); },
            [](NodeData& n, int v){ n.kind = static_cast<ContentKind>(v); })
        .def_prop_rw("memory_layer",
            [](const NodeData& n){ return static_cast<int>(n.memory_layer); },
            [](NodeData& n, int v){ n.memory_layer = static_cast<MemoryLayer>(v); })
        .def_rw("symbol_name", &NodeData::symbol_name)
        .def_rw("signature",   &NodeData::signature)
        .def_rw("docstring",   &NodeData::docstring)
        .def_rw("body",        &NodeData::body)
        // Embedding
        .def_rw("embedding",               &NodeData::embedding)
        .def_rw("embedding_model",         &NodeData::embedding_model)
        .def_rw("embedding_model_version", &NodeData::embedding_model_version)
    ;

    // ── Store class ───────────────────────────────────────────────────────────
    nb::class_<Store>(m, "Store")
        .def(nb::init<std::string, std::size_t, std::string>(),
             nb::arg("path"), nb::arg("embedding_dim"), nb::arg("embedding_model"))
        // Lifecycle
        .def("initialize",         &Store::initialize)
        .def("close",              &Store::close)
        // Transactions
        .def("begin_transaction",  &Store::begin_transaction)
        .def("commit",             &Store::commit)
        .def("rollback",           &Store::rollback)
        // Writes
        .def("upsert_node",        &Store::upsert_node,        nb::arg("node"))
        .def("upsert_edge",        &Store::upsert_edge,
             nb::arg("source_id"), nb::arg("target_id"),
             nb::arg("edge_type"), nb::arg("source_file"),
             nb::arg("git_sha"),   nb::arg("indexed_at"))
        .def("delete_node",             &Store::delete_node,             nb::arg("node_id"))
        .def("delete_nodes_for_file",   &Store::delete_nodes_for_file,   nb::arg("source_file"))
        // Reads
        .def("get_node",           &Store::get_node,           nb::arg("node_id"))
        .def("neighbors",          &Store::neighbors,
             nb::arg("node_id"), nb::arg("direction"),
             nb::arg("edge_type") = nb::none())
        .def("vector_search",      &Store::vector_search,
             nb::arg("embedding"), nb::arg("k"),
             nb::arg("node_type") = nb::none())
        // Manifest
        .def("record_indexed_file",  &Store::record_indexed_file,
             nb::arg("file_path"), nb::arg("git_sha"), nb::arg("indexed_at"))
        .def("indexed_file_sha",     &Store::indexed_file_sha,   nb::arg("file_path"))
        .def("list_indexed_files",   &Store::list_indexed_files)
        // Config accessors
        .def("embedding_dim",        &Store::embedding_dim)
        .def("embedding_model",      &Store::embedding_model)
    ;
}

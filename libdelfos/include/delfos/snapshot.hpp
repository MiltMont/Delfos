#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>   // ::getpid()
#include <vector>

// FlatBuffers runtime (suppress third-party warnings).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-W#warnings"
#include <flatbuffers/flatbuffers.h>
#pragma clang diagnostic pop

// Generated FlatBuffers header (libdelfos/flatbuffers/delfos_generated.h).
// The include path libdelfos/flatbuffers/ is added to delfos_core by CMake.
#include "delfos_generated.h"

#include "delfos/graph.hpp"
#include "delfos/vector_index.hpp"

namespace delfos {

// A row in the checkpoint manifest — one entry per indexed source file.
// Stored alongside the graph so the indexer can detect stale files on restart.
struct ManifestEntry {
    std::string file_path;
    std::string git_sha;
    int64_t     indexed_at{0};
};

namespace snapshot {

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace detail {

// Write raw bytes to a file, throwing on error.
inline void write_file(const std::filesystem::path& path,
                       const uint8_t* data, std::size_t size) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("snapshot: cannot open for write: " + path.string());
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) throw std::runtime_error("snapshot: write failed: " + path.string());
}

// Read an entire binary file into a byte vector, throwing on error.
inline std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("snapshot: cannot open for read: " + path.string());
    std::size_t size = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::vector<uint8_t> buf(size);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error("snapshot: read failed: " + path.string());
    return buf;
}

// Convert delfos::NodeData → delfos::fb::NodeT (object-API struct).
inline delfos::fb::NodeT to_fb_node(const NodeData& n) {
    delfos::fb::NodeT t;
    t.id                      = n.id;
    t.type                    = static_cast<delfos::fb::NodeType>(static_cast<uint8_t>(n.type));
    t.status                  = static_cast<delfos::fb::NodeStatus>(static_cast<uint8_t>(n.status));
    t.indexed_at              = n.indexed_at;
    t.deleted_at              = n.deleted_at;
    t.deleted_by_commit       = n.deleted_by_commit;
    t.source_file             = n.source_file;
    t.git_sha                 = n.git_sha;
    t.cue_type                = static_cast<delfos::fb::CueType>(static_cast<uint8_t>(n.cue_type));
    t.text                    = n.text;
    t.category                = static_cast<delfos::fb::TagCategory>(static_cast<uint8_t>(n.category));
    t.value                   = n.value;
    t.kind                    = static_cast<delfos::fb::ContentKind>(static_cast<uint8_t>(n.kind));
    t.memory_layer            = static_cast<delfos::fb::MemoryLayer>(static_cast<uint8_t>(n.memory_layer));
    t.symbol_name             = n.symbol_name;
    t.signature               = n.signature;
    t.docstring               = n.docstring;
    t.body                    = n.body;
    t.embedding               = n.embedding;
    t.embedding_model         = n.embedding_model;
    t.embedding_model_version = n.embedding_model_version;
    return t;
}

// Convert delfos::fb::NodeT → delfos::NodeData.
inline NodeData from_fb_node(const delfos::fb::NodeT& t) {
    NodeData n;
    n.id                      = t.id;
    n.type                    = static_cast<NodeType>(static_cast<uint8_t>(t.type));
    n.status                  = static_cast<NodeStatus>(static_cast<uint8_t>(t.status));
    n.indexed_at              = t.indexed_at;
    n.deleted_at              = t.deleted_at;
    n.deleted_by_commit       = t.deleted_by_commit;
    n.source_file             = t.source_file;
    n.git_sha                 = t.git_sha;
    n.cue_type                = static_cast<CueType>(static_cast<uint8_t>(t.cue_type));
    n.text                    = t.text;
    n.category                = static_cast<TagCategory>(static_cast<uint8_t>(t.category));
    n.value                   = t.value;
    n.kind                    = static_cast<ContentKind>(static_cast<uint8_t>(t.kind));
    n.memory_layer            = static_cast<MemoryLayer>(static_cast<uint8_t>(t.memory_layer));
    n.symbol_name             = t.symbol_name;
    n.signature               = t.signature;
    n.docstring               = t.docstring;
    n.body                    = t.body;
    n.embedding               = t.embedding;
    n.embedding_model         = t.embedding_model;
    n.embedding_model_version = t.embedding_model_version;
    return n;
}

// Convert delfos::EdgeData → delfos::fb::EdgeT.
inline delfos::fb::EdgeT to_fb_edge(const EdgeData& e) {
    delfos::fb::EdgeT t;
    t.source      = e.source;
    t.target      = e.target;
    t.type        = static_cast<delfos::fb::EdgeType>(static_cast<uint8_t>(e.type));
    t.source_file = e.source_file;
    t.git_sha     = e.git_sha;
    t.indexed_at  = e.indexed_at;
    return t;
}

// Convert delfos::fb::EdgeT → delfos::EdgeData.
inline EdgeData from_fb_edge(const delfos::fb::EdgeT& t) {
    EdgeData e;
    e.source      = t.source;
    e.target      = t.target;
    e.type        = static_cast<EdgeType>(static_cast<uint8_t>(t.type));
    e.source_file = t.source_file;
    e.git_sha     = t.git_sha;
    e.indexed_at  = t.indexed_at;
    return e;
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// save() — serialise graph + vectors + manifest to dir atomically.
//
// Writes to a temp directory (dir.parent / dir.name + ".tmp." + PID) first,
// then renames it over the target. A crash mid-write leaves the existing
// snapshot intact (rename is atomic on the same filesystem).
//
// Precondition: graph.dirty() == false (call rebuild() before save).
// ─────────────────────────────────────────────────────────────────────────────
inline void save(const std::filesystem::path& dir,
                 const Graph& graph,
                 const VectorIndex& vectors,
                 const std::vector<ManifestEntry>& manifest) {
    if (graph.dirty())
        throw std::logic_error("snapshot::save: graph is dirty — call rebuild() first");

    // ── Prepare temp directory ─────────────────────────────────────────────
    std::filesystem::path tmp_dir = dir.parent_path() /
        (dir.filename().string() + ".tmp." + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);

    // ── 1. Serialize graph (nodes + edges) to graph.fb ────────────────────
    {
        delfos::fb::SnapshotT snap;
        snap.nodes.reserve(graph.node_count());
        snap.edges.reserve(graph.edge_count());

        for (const NodeData& n : graph.nodes_view())
            snap.nodes.push_back(std::make_unique<delfos::fb::NodeT>(detail::to_fb_node(n)));

        for (const EdgeData& e : graph.edges_view())
            snap.edges.push_back(std::make_unique<delfos::fb::EdgeT>(detail::to_fb_edge(e)));

        flatbuffers::FlatBufferBuilder fbb;
        fbb.Finish(delfos::fb::Snapshot::Pack(fbb, &snap));
        detail::write_file(tmp_dir / "graph.fb", fbb.GetBufferPointer(), fbb.GetSize());
    }

    // ── 2. Save vector index (USearch native format) ──────────────────────
    vectors.save(tmp_dir / "vectors.usearch");

    // ── 3. Serialize manifest to manifest.fb ──────────────────────────────
    {
        delfos::fb::SnapshotT snap;
        snap.manifest.reserve(manifest.size());
        for (const ManifestEntry& m : manifest) {
            auto mt = std::make_unique<delfos::fb::ManifestEntryT>();
            mt->file_path  = m.file_path;
            mt->git_sha    = m.git_sha;
            mt->indexed_at = m.indexed_at;
            snap.manifest.push_back(std::move(mt));
        }
        flatbuffers::FlatBufferBuilder fbb;
        fbb.Finish(delfos::fb::Snapshot::Pack(fbb, &snap));
        detail::write_file(tmp_dir / "manifest.fb", fbb.GetBufferPointer(), fbb.GetSize());
    }

    // ── 4. Atomic rename (replaces existing snapshot if any) ─────────────
    if (std::filesystem::exists(dir)) std::filesystem::remove_all(dir);
    std::filesystem::rename(tmp_dir, dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// load() — deserialise snapshot dir into graph, vectors, and manifest.
//
// After loading:
//   - graph is rebuilt (dirty() == false, adjacency queries work)
//   - vectors is populated with the saved HNSW index
//   - vectors.node_type_map() is rebuilt from the loaded graph nodes
//   - manifest contains the checkpoint rows
// ─────────────────────────────────────────────────────────────────────────────
inline void load(const std::filesystem::path& dir,
                 Graph& graph,
                 VectorIndex& vectors,
                 std::vector<ManifestEntry>& manifest) {
    // ── 1. Load graph (nodes + edges) from graph.fb ───────────────────────
    {
        auto buf = detail::read_file(dir / "graph.fb");
        auto snap = delfos::fb::UnPackSnapshot(buf.data());

        for (const auto& nt : snap->nodes)
            graph.upsert_node(detail::from_fb_node(*nt));
        for (const auto& et : snap->edges)
            graph.upsert_edge(detail::from_fb_edge(*et));
        graph.rebuild();
    }

    // ── 2. Load vector index (USearch native format) ──────────────────────
    vectors.load(dir / "vectors.usearch");

    // Restore node_type_map from the loaded graph: any node whose embedding
    // is non-empty was indexed in the VectorIndex before the snapshot was saved.
    // This is safe because: (a) we compacted the graph, so NodeIdx values are
    // contiguous; (b) snapshot::save() is always called after rebuild().
    {
        auto& type_map = vectors.node_type_map();
        type_map.clear();
        const auto nodes = graph.nodes_view();
        for (NodeIdx i = 0; i < static_cast<NodeIdx>(nodes.size()); ++i) {
            if (!nodes[i].embedding.empty()) {
                type_map.emplace(i, nodes[i].type);
            }
        }
    }

    // ── 3. Load manifest from manifest.fb ────────────────────────────────
    {
        auto buf = detail::read_file(dir / "manifest.fb");
        auto snap = delfos::fb::UnPackSnapshot(buf.data());

        manifest.clear();
        manifest.reserve(snap->manifest.size());
        for (const auto& mt : snap->manifest) {
            manifest.push_back({mt->file_path, mt->git_sha, mt->indexed_at});
        }
    }
}

} // namespace snapshot
} // namespace delfos

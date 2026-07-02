#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "delfos/delfos.hpp"

using namespace delfos;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM = 4;
static constexpr char MODEL[]    = "test-model";

static std::vector<double> vec(double v0, double v1, double v2, double v3) {
    // normalise so cosine similarity is well-defined
    double len = std::sqrt(v0*v0 + v1*v1 + v2*v2 + v3*v3);
    if (len == 0.0) len = 1.0;
    return {v0/len, v1/len, v2/len, v3/len};
}

// Build a minimal but complete graph+VectorIndex+manifest and verify it
// works before we save.
struct Fixture {
    Graph        graph;
    VectorIndex  vi{DIM};
    std::vector<ManifestEntry> manifest;

    NodeIdx cue_idx{};
    NodeIdx tag_idx{};
    NodeIdx con_idx{};

    Fixture() {
        // Cue node with embedding
        NodeData cue;
        cue.id           = "cue:symbol:a.cpp::foo";
        cue.type         = NodeType::Cue;
        cue.status       = NodeStatus::Active;
        cue.indexed_at   = 111000LL;
        cue.source_file  = "a.cpp";
        cue.git_sha      = "deadbeef";
        cue.cue_type     = CueType::Symbol;
        cue.text         = "foo";
        cue.embedding    = vec(1,0,0,0);
        cue.embedding_model         = MODEL;
        cue.embedding_model_version = "v1";
        cue_idx = graph.upsert_node(cue);
        vi.insert(cue_idx, NodeType::Cue, cue.embedding);

        // Tag node (no embedding)
        NodeData tag;
        tag.id         = "tag:language:cpp";
        tag.type       = NodeType::Tag;
        tag.status     = NodeStatus::Active;
        tag.indexed_at = 222000LL;
        tag.category   = TagCategory::Language;
        tag.value      = "cpp";
        tag_idx = graph.upsert_node(tag);

        // Content node with embedding
        NodeData con;
        con.id           = "content:a.cpp::foo";
        con.type         = NodeType::Content;
        con.status       = NodeStatus::Active;
        con.indexed_at   = 333000LL;
        con.source_file  = "a.cpp";
        con.git_sha      = "deadbeef";
        con.kind         = ContentKind::Function;
        con.memory_layer = MemoryLayer::Semantic;
        con.symbol_name  = "foo";
        con.signature    = "void foo()";
        con.docstring    = "A test function.";
        con.body         = "void foo() {}";
        con.scip_symbol  = "scip-python python . a.cpp/foo().";
        con.embedding    = vec(0,1,0,0);
        con.embedding_model = MODEL;
        con_idx = graph.upsert_node(con);
        vi.insert(con_idx, NodeType::Content, con.embedding);

        // Edges
        graph.upsert_edge({cue_idx, con_idx, EdgeType::CueOf,       "a.cpp", "deadbeef", 444000LL});
        graph.upsert_edge({con_idx, tag_idx, EdgeType::TaggedWith,   "a.cpp", "deadbeef", 555000LL});

        graph.rebuild();

        // Manifest entry
        manifest.push_back({"a.cpp", "deadbeef", 666000LL});
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Basic round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("save creates three files in snapshot directory", "[snapshot][files]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_files";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    REQUIRE(std::filesystem::exists(dir / "graph.fb"));
    REQUIRE(std::filesystem::exists(dir / "vectors.usearch"));
    REQUIRE(std::filesystem::exists(dir / "manifest.fb"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: node count and edge count are preserved", "[snapshot][roundtrip]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_counts";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    Graph g2;
    VectorIndex vi2(DIM);
    std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    REQUIRE(g2.node_count() == f.graph.node_count());
    REQUIRE(g2.edge_count() == f.graph.edge_count());
    REQUIRE(vi2.size()      == f.vi.size());
    REQUIRE(m2.size()       == f.manifest.size());

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: graph find works after load", "[snapshot][roundtrip]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_find";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    REQUIRE(g2.find("cue:symbol:a.cpp::foo") != INVALID_NODE);
    REQUIRE(g2.find("tag:language:cpp")      != INVALID_NODE);
    REQUIRE(g2.find("content:a.cpp::foo")    != INVALID_NODE);
    REQUIRE(g2.find("nonexistent")           == INVALID_NODE);

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: all node fields are preserved", "[snapshot][fields]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_nodefields";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    // Cue node fields
    NodeIdx ci = g2.find("cue:symbol:a.cpp::foo");
    const NodeData& c = g2.node(ci);
    REQUIRE(c.type         == NodeType::Cue);
    REQUIRE(c.status       == NodeStatus::Active);
    REQUIRE(c.indexed_at   == 111000LL);
    REQUIRE(c.source_file  == "a.cpp");
    REQUIRE(c.git_sha      == "deadbeef");
    REQUIRE(c.cue_type     == CueType::Symbol);
    REQUIRE(c.text         == "foo");
    REQUIRE(c.embedding.size()        == DIM);
    REQUIRE(c.embedding_model         == MODEL);
    REQUIRE(c.embedding_model_version == "v1");

    // Content node fields
    NodeIdx ni = g2.find("content:a.cpp::foo");
    const NodeData& n = g2.node(ni);
    REQUIRE(n.type         == NodeType::Content);
    REQUIRE(n.indexed_at   == 333000LL);
    REQUIRE(n.kind         == ContentKind::Function);
    REQUIRE(n.memory_layer == MemoryLayer::Semantic);
    REQUIRE(n.symbol_name  == "foo");
    REQUIRE(n.signature    == "void foo()");
    REQUIRE(n.docstring    == "A test function.");
    REQUIRE(n.body         == "void foo() {}");
    REQUIRE(n.scip_symbol  == "scip-python python . a.cpp/foo().");

    // Tag node fields
    NodeIdx ti = g2.find("tag:language:cpp");
    const NodeData& t = g2.node(ti);
    REQUIRE(t.type     == NodeType::Tag);
    REQUIRE(t.category == TagCategory::Language);
    REQUIRE(t.value    == "cpp");
    REQUIRE(t.indexed_at == 222000LL);

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: edge fields are preserved", "[snapshot][fields]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_edgefields";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    NodeIdx ci  = g2.find("cue:symbol:a.cpp::foo");
    NodeIdx con = g2.find("content:a.cpp::foo");

    auto out = g2.outgoing(ci);
    REQUIRE(out.size() == 1u);
    const EdgeData& e = g2.edge(out[0]);
    REQUIRE(e.type        == EdgeType::CueOf);
    REQUIRE(e.target      == con);
    REQUIRE(e.source_file == "a.cpp");
    REQUIRE(e.git_sha     == "deadbeef");
    REQUIRE(e.indexed_at  == 444000LL);

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: adjacency works after load", "[snapshot][adjacency]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_adj";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    NodeIdx ci  = g2.find("cue:symbol:a.cpp::foo");
    NodeIdx con = g2.find("content:a.cpp::foo");
    NodeIdx tag = g2.find("tag:language:cpp");

    auto cue_out = g2.outgoing(ci);
    REQUIRE(cue_out.size() == 1u);
    REQUIRE(g2.edge(cue_out[0]).target == con);

    auto con_out = g2.outgoing(con);
    REQUIRE(con_out.size() == 1u);
    REQUIRE(g2.edge(con_out[0]).target == tag);

    auto con_inc = g2.incoming(con);
    REQUIRE(con_inc.size() == 1u);

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: vector search works after load (type filter preserved)", "[snapshot][vectors]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_vec";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    // Unfiltered: both cue (vec[1,0,0,0]) and content (vec[0,1,0,0]) indexed
    auto hits = vi2.search(vec(1,0,0,0), 2);
    REQUIRE(hits.size() == 2u);
    REQUIRE(hits[0].score == Catch::Approx(1.0f).epsilon(1e-4f));

    // Cue-only filter: only the cue node should be returned
    NodeType cue_filter = NodeType::Cue;
    auto cue_hits = vi2.search(vec(1,0,0,0), 5, &cue_filter);
    REQUIRE(cue_hits.size() == 1u);
    // The returned NodeIdx should correspond to the cue node in g2
    NodeIdx loaded_cue = g2.find("cue:symbol:a.cpp::foo");
    REQUIRE(cue_hits[0].node == loaded_cue);

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: manifest is preserved", "[snapshot][manifest]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_manifest";
    Fixture f;
    snapshot::save(dir, f.graph, f.vi, f.manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    REQUIRE(m2.size() == 1u);
    REQUIRE(m2[0].file_path  == "a.cpp");
    REQUIRE(m2[0].git_sha    == "deadbeef");
    REQUIRE(m2[0].indexed_at == 666000LL);

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: timestamps are preserved exactly", "[snapshot][fields]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_ts";

    Graph g;
    NodeData n;
    n.id             = "cue:ts-test";
    n.type           = NodeType::Cue;
    n.indexed_at     = 9876543210987654LL;
    n.deleted_at     = 1234567890123456LL;
    n.deleted_by_commit = "tombstone-sha";
    n.cue_type       = CueType::ErrorMessage;
    n.text           = "RuntimeError: out of memory";
    n.source_file    = "x.py";
    n.git_sha        = "cafecafe";
    g.upsert_node(n);
    g.rebuild();

    VectorIndex vi(DIM);
    std::vector<ManifestEntry> manifest;
    snapshot::save(dir, g, vi, manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    NodeIdx idx = g2.find("cue:ts-test");
    const NodeData& loaded = g2.node(idx);
    REQUIRE(loaded.indexed_at       == 9876543210987654LL);
    REQUIRE(loaded.deleted_at       == 1234567890123456LL);
    REQUIRE(loaded.deleted_by_commit == "tombstone-sha");
    REQUIRE(loaded.cue_type         == CueType::ErrorMessage);

    std::filesystem::remove_all(dir);
}

TEST_CASE("save/load: empty graph and empty manifest round-trip", "[snapshot][edge_cases]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_empty";

    Graph g;
    g.rebuild(); // already not dirty, but make it explicit
    VectorIndex vi(DIM);
    std::vector<ManifestEntry> manifest;
    snapshot::save(dir, g, vi, manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    REQUIRE_NOTHROW(snapshot::load(dir, g2, vi2, m2));

    REQUIRE(g2.node_count() == 0u);
    REQUIRE(g2.edge_count() == 0u);
    REQUIRE(vi2.size()      == 0u);
    REQUIRE(m2.empty());

    std::filesystem::remove_all(dir);
}

TEST_CASE("save is atomic: second save replaces first snapshot cleanly", "[snapshot][atomic]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_atomic";

    // First save: 1 cue node
    {
        Graph g;
        NodeData n;
        n.id = "cue:v1";  n.type = NodeType::Cue;
        n.cue_type = CueType::Symbol;  n.text = "v1";
        n.indexed_at = 1LL;  n.source_file = "a.cpp";  n.git_sha = "sha1";
        g.upsert_node(n);
        g.rebuild();
        VectorIndex vi(DIM);
        snapshot::save(dir, g, vi, {});
    }

    // Verify first snapshot
    {
        Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
        snapshot::load(dir, g2, vi2, m2);
        REQUIRE(g2.find("cue:v1") != INVALID_NODE);
        REQUIRE(g2.find("cue:v2") == INVALID_NODE);
    }

    // Second save: different node
    {
        Graph g;
        NodeData n;
        n.id = "cue:v2";  n.type = NodeType::Cue;
        n.cue_type = CueType::Symbol;  n.text = "v2";
        n.indexed_at = 2LL;  n.source_file = "b.cpp";  n.git_sha = "sha2";
        g.upsert_node(n);
        g.rebuild();
        VectorIndex vi(DIM);
        snapshot::save(dir, g, vi, {});
    }

    // Verify the second snapshot completely replaced the first
    {
        Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
        snapshot::load(dir, g2, vi2, m2);
        REQUIRE(g2.find("cue:v1") == INVALID_NODE);
        REQUIRE(g2.find("cue:v2") != INVALID_NODE);
    }

    std::filesystem::remove_all(dir);
}

TEST_CASE("save on dirty graph throws", "[snapshot][precondition]") {
    Graph g;
    NodeData n;
    n.id = "x";  n.type = NodeType::Cue;  n.cue_type = CueType::Symbol;
    n.text = "x";  n.indexed_at = 1LL;
    g.upsert_node(n);  // sets dirty_
    REQUIRE(g.dirty());

    VectorIndex vi(DIM);
    REQUIRE_THROWS_AS(
        snapshot::save(std::filesystem::temp_directory_path() / "should_not_exist",
                       g, vi, {}),
        std::logic_error);
}

TEST_CASE("save/load: multiple manifest entries preserved", "[snapshot][manifest]") {
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "delfos_snap_multiman";

    Graph g;
    g.rebuild();
    VectorIndex vi(DIM);
    std::vector<ManifestEntry> manifest = {
        {"a.cpp", "sha_a", 1000LL},
        {"b.cpp", "sha_b", 2000LL},
        {"c.cpp", "sha_c", 3000LL},
    };
    snapshot::save(dir, g, vi, manifest);

    Graph g2;  VectorIndex vi2(DIM);  std::vector<ManifestEntry> m2;
    snapshot::load(dir, g2, vi2, m2);

    REQUIRE(m2.size() == 3u);
    std::unordered_map<std::string, std::string> sha_map;
    for (const auto& e : m2) sha_map[e.file_path] = e.git_sha;
    REQUIRE(sha_map["a.cpp"] == "sha_a");
    REQUIRE(sha_map["b.cpp"] == "sha_b");
    REQUIRE(sha_map["c.cpp"] == "sha_c");

    std::filesystem::remove_all(dir);
}

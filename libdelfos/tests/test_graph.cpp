#include <catch2/catch_test_macros.hpp>

#include "delfos/delfos.hpp"

using namespace delfos;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static NodeData make_cue(std::string id, std::string source_file = "a.cpp") {
    NodeData n;
    n.id          = std::move(id);
    n.type        = NodeType::Cue;
    n.status      = NodeStatus::Active;
    n.indexed_at  = 1000000LL;
    n.source_file = std::move(source_file);
    n.git_sha     = "abc123";
    n.cue_type    = CueType::Symbol;
    n.text        = "some_function";
    return n;
}

static NodeData make_tag(std::string id, std::string source_file = "a.cpp") {
    NodeData n;
    n.id          = std::move(id);
    n.type        = NodeType::Tag;
    n.status      = NodeStatus::Active;
    n.indexed_at  = 1000000LL;
    n.source_file = std::move(source_file);
    n.category    = TagCategory::Language;
    n.value       = "cpp";
    return n;
}

static NodeData make_content(std::string id, std::string source_file = "a.cpp") {
    NodeData n;
    n.id          = std::move(id);
    n.type        = NodeType::Content;
    n.status      = NodeStatus::Active;
    n.indexed_at  = 1000000LL;
    n.source_file = std::move(source_file);
    n.git_sha     = "abc123";
    n.kind        = ContentKind::Function;
    n.memory_layer = MemoryLayer::Semantic;
    n.body        = "void foo() {}";
    return n;
}

static EdgeData make_edge(NodeIdx src, NodeIdx tgt,
                          EdgeType t = EdgeType::CueOf,
                          std::string source_file = "a.cpp") {
    EdgeData e;
    e.source      = src;
    e.target      = tgt;
    e.type        = t;
    e.source_file = std::move(source_file);
    e.indexed_at  = 1000000LL;
    return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// Enum completeness (compile-time sanity check — just instantiate each enum)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("enum values are all reachable", "[types]") {
    // If any enum value is missing a case the compiler warns; this just
    // exercises the full set so the enum/value mapping is explicit.
    static_assert(static_cast<uint8_t>(NodeType::Cue)     == 0u);
    static_assert(static_cast<uint8_t>(NodeType::Tag)     == 1u);
    static_assert(static_cast<uint8_t>(NodeType::Content) == 2u);

    static_assert(static_cast<uint8_t>(NodeStatus::Active)  == 0u);
    static_assert(static_cast<uint8_t>(NodeStatus::Deleted) == 1u);

    static_assert(static_cast<uint8_t>(EdgeType::CueOf)       == 0u);
    static_assert(static_cast<uint8_t>(EdgeType::TaggedWith)   == 1u);
    static_assert(static_cast<uint8_t>(EdgeType::PartOfTopic)  == 2u);
    static_assert(static_cast<uint8_t>(EdgeType::RedirectsTo)  == 3u);

    static_assert(static_cast<uint8_t>(CueType::Symbol)       == 0u);
    static_assert(static_cast<uint8_t>(CueType::Concept)      == 1u);
    static_assert(static_cast<uint8_t>(CueType::ErrorMessage) == 2u);

    static_assert(static_cast<uint8_t>(TagCategory::ModulePath)   == 0u);
    static_assert(static_cast<uint8_t>(TagCategory::ArchLayer)    == 1u);
    static_assert(static_cast<uint8_t>(TagCategory::PatternType)  == 2u);
    static_assert(static_cast<uint8_t>(TagCategory::LangConstruct)== 3u);
    static_assert(static_cast<uint8_t>(TagCategory::Language)     == 4u);

    static_assert(static_cast<uint8_t>(ContentKind::Function) == 0u);
    static_assert(static_cast<uint8_t>(ContentKind::Class)    == 1u);
    static_assert(static_cast<uint8_t>(ContentKind::Module)   == 2u);
    static_assert(static_cast<uint8_t>(ContentKind::Commit)   == 3u);
    static_assert(static_cast<uint8_t>(ContentKind::Test)     == 4u);

    static_assert(static_cast<uint8_t>(MemoryLayer::Episodic) == 0u);
    static_assert(static_cast<uint8_t>(MemoryLayer::Semantic) == 1u);
    static_assert(static_cast<uint8_t>(MemoryLayer::Topic)    == 2u);

    static_assert(static_cast<uint8_t>(Direction::Outgoing) == 0u);
    static_assert(static_cast<uint8_t>(Direction::Incoming) == 1u);

    SUCCEED("all enum values accessible");
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty-graph invariants
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("empty graph has zero counts", "[graph][empty]") {
    Graph g;
    REQUIRE(g.node_count() == 0u);
    REQUIRE(g.edge_count() == 0u);
    REQUIRE(!g.dirty());
}

TEST_CASE("find on empty graph returns INVALID_NODE", "[graph][empty]") {
    Graph g;
    REQUIRE(g.find("anything") == INVALID_NODE);
}

TEST_CASE("rebuild on empty graph is a no-op", "[graph][empty]") {
    Graph g;
    g.rebuild();
    REQUIRE(g.node_count() == 0u);
    REQUIRE(!g.dirty());
}

// ─────────────────────────────────────────────────────────────────────────────
// upsert_node
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("upsert_node inserts a new node", "[graph][upsert]") {
    Graph g;
    NodeIdx idx = g.upsert_node(make_cue("cue-1"));
    REQUIRE(idx != INVALID_NODE);
    REQUIRE(g.node_count() == 1u);
}

TEST_CASE("upsert_node returns consistent index", "[graph][upsert]") {
    Graph g;
    NodeIdx a = g.upsert_node(make_cue("cue-a"));
    NodeIdx b = g.upsert_node(make_cue("cue-b"));
    REQUIRE(a != b);
    REQUIRE(a != INVALID_NODE);
    REQUIRE(b != INVALID_NODE);
}

TEST_CASE("upsert_node replace-by-id semantics", "[graph][upsert]") {
    Graph g;
    g.upsert_node(make_cue("cue-1"));
    REQUIRE(g.node_count() == 1u);

    // Replace with updated text.
    NodeData updated = make_cue("cue-1");
    updated.text = "updated_name";
    g.upsert_node(updated);

    REQUIRE(g.node_count() == 1u); // still one node

    g.rebuild();
    NodeIdx idx = g.find("cue-1");
    REQUIRE(idx != INVALID_NODE);
    REQUIRE(g.node(idx).text == "updated_name");
}

TEST_CASE("upsert_node sets dirty flag", "[graph][upsert]") {
    Graph g;
    REQUIRE(!g.dirty());
    g.upsert_node(make_cue("cue-1"));
    REQUIRE(g.dirty());
    g.rebuild();
    REQUIRE(!g.dirty());
}

TEST_CASE("upsert_node tracks all node types", "[graph][upsert]") {
    Graph g;
    g.upsert_node(make_cue("cue-1"));
    g.upsert_node(make_tag("tag-1"));
    g.upsert_node(make_content("content-1"));
    REQUIRE(g.node_count() == 3u);

    g.rebuild();
    REQUIRE(g.node(g.find("cue-1")).type     == NodeType::Cue);
    REQUIRE(g.node(g.find("tag-1")).type     == NodeType::Tag);
    REQUIRE(g.node(g.find("content-1")).type == NodeType::Content);
}

// ─────────────────────────────────────────────────────────────────────────────
// find() while dirty
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("find works while dirty", "[graph][find]") {
    Graph g;
    g.upsert_node(make_cue("cue-1"));
    REQUIRE(g.dirty());
    REQUIRE(g.find("cue-1") != INVALID_NODE);
    REQUIRE(g.find("nonexistent") == INVALID_NODE);
}

TEST_CASE("find works after rebuild", "[graph][find]") {
    Graph g;
    g.upsert_node(make_cue("cue-1"));
    g.rebuild();
    REQUIRE(!g.dirty());
    REQUIRE(g.find("cue-1") != INVALID_NODE);
}

// ─────────────────────────────────────────────────────────────────────────────
// upsert_edge
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("upsert_edge inserts a new edge", "[graph][edge]") {
    Graph g;
    NodeIdx a = g.upsert_node(make_cue("cue-1"));
    NodeIdx b = g.upsert_node(make_content("content-1"));
    EdgeIdx ei = g.upsert_edge(make_edge(a, b));
    REQUIRE(ei != INVALID_EDGE);
    REQUIRE(g.edge_count() == 1u);
}

TEST_CASE("upsert_edge replace-by-triple semantics", "[graph][edge]") {
    Graph g;
    NodeIdx a = g.upsert_node(make_cue("cue-1"));
    NodeIdx b = g.upsert_node(make_content("content-1"));

    EdgeData e1 = make_edge(a, b);
    e1.git_sha = "sha-old";
    g.upsert_edge(e1);

    EdgeData e2 = make_edge(a, b);
    e2.git_sha = "sha-new";
    g.upsert_edge(e2);

    REQUIRE(g.edge_count() == 1u); // same triple → replace, not append

    g.rebuild();
    // The edge after rebuild should be the second one (sha-new).
    auto out = g.outgoing(a);
    REQUIRE(out.size() == 1u);
    REQUIRE(g.edge(out[0]).git_sha == "sha-new");
}

TEST_CASE("distinct edge triples are both stored", "[graph][edge]") {
    Graph g;
    NodeIdx cue  = g.upsert_node(make_cue("cue-1"));
    NodeIdx con  = g.upsert_node(make_content("content-1"));
    NodeIdx tag  = g.upsert_node(make_tag("tag-1"));
    g.upsert_edge(make_edge(cue, con, EdgeType::CueOf));
    g.upsert_edge(make_edge(cue, tag, EdgeType::TaggedWith));
    REQUIRE(g.edge_count() == 2u);
}

// ─────────────────────────────────────────────────────────────────────────────
// CSR adjacency after rebuild()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("outgoing edges after rebuild", "[graph][csr]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    g.upsert_edge(make_edge(cue, con));
    g.rebuild();

    auto out = g.outgoing(cue);
    REQUIRE(out.size() == 1u);
    REQUIRE(g.edge(out[0]).target == con);

    auto empty_out = g.outgoing(con);
    REQUIRE(empty_out.empty());
}

TEST_CASE("incoming edges after rebuild", "[graph][csr]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    g.upsert_edge(make_edge(cue, con));
    g.rebuild();

    auto inc = g.incoming(con);
    REQUIRE(inc.size() == 1u);
    REQUIRE(g.edge(inc[0]).source == cue);

    auto empty_inc = g.incoming(cue);
    REQUIRE(empty_inc.empty());
}

TEST_CASE("neighbors outgoing direction", "[graph][neighbors]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    g.upsert_edge(make_edge(cue, con));
    g.rebuild();

    auto nb = g.neighbors(cue, Direction::Outgoing);
    REQUIRE(nb.size() == 1u);
    REQUIRE(nb[0] == con);
}

TEST_CASE("neighbors incoming direction", "[graph][neighbors]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    g.upsert_edge(make_edge(cue, con));
    g.rebuild();

    auto nb = g.neighbors(con, Direction::Incoming);
    REQUIRE(nb.size() == 1u);
    REQUIRE(nb[0] == cue);
}

TEST_CASE("neighbors edge-type filter", "[graph][neighbors]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    NodeIdx tag = g.upsert_node(make_tag("tag-1"));
    g.upsert_edge(make_edge(cue, con, EdgeType::CueOf));
    g.upsert_edge(make_edge(cue, tag, EdgeType::TaggedWith));
    g.rebuild();

    EdgeType tw = EdgeType::TaggedWith;
    auto nb = g.neighbors(cue, Direction::Outgoing, &tw);
    REQUIRE(nb.size() == 1u);
    REQUIRE(nb[0] == tag);

    EdgeType co = EdgeType::CueOf;
    auto nb2 = g.neighbors(cue, Direction::Outgoing, &co);
    REQUIRE(nb2.size() == 1u);
    REQUIRE(nb2[0] == con);
}

TEST_CASE("neighbors returns all without type filter", "[graph][neighbors]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    NodeIdx tag = g.upsert_node(make_tag("tag-1"));
    g.upsert_edge(make_edge(cue, con, EdgeType::CueOf));
    g.upsert_edge(make_edge(cue, tag, EdgeType::TaggedWith));
    g.rebuild();

    auto nb = g.neighbors(cue, Direction::Outgoing); // no filter
    REQUIRE(nb.size() == 2u);
}

// ─────────────────────────────────────────────────────────────────────────────
// delete_node (hard-delete)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("delete_node removes node from find immediately", "[graph][delete]") {
    Graph g;
    NodeIdx idx = g.upsert_node(make_cue("cue-1"));
    REQUIRE(g.find("cue-1") == idx);
    g.delete_node(idx);
    REQUIRE(g.find("cue-1") == INVALID_NODE);
    REQUIRE(g.node_count() == 0u);
}

TEST_CASE("delete_node removes incident edges immediately", "[graph][delete]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    g.upsert_edge(make_edge(cue, con));
    REQUIRE(g.edge_count() == 1u);

    g.delete_node(cue);
    REQUIRE(g.edge_count() == 0u);
}

TEST_CASE("delete_node leaves non-incident nodes intact", "[graph][delete]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    g.upsert_node(make_content("content-1"));
    g.delete_node(cue);

    REQUIRE(g.find("content-1") != INVALID_NODE);
    REQUIRE(g.node_count() == 1u);
}

TEST_CASE("delete_node on INVALID_NODE is a no-op", "[graph][delete]") {
    Graph g;
    g.upsert_node(make_cue("cue-1"));
    REQUIRE_NOTHROW(g.delete_node(INVALID_NODE));
    REQUIRE(g.node_count() == 1u);
}

TEST_CASE("delete_node followed by upsert same id creates fresh node", "[graph][delete]") {
    Graph g;
    NodeIdx old_idx = g.upsert_node(make_cue("cue-1"));
    g.delete_node(old_idx);
    REQUIRE(g.find("cue-1") == INVALID_NODE);

    NodeIdx new_idx = g.upsert_node(make_cue("cue-1"));
    REQUIRE(g.find("cue-1") == new_idx);
    REQUIRE(g.node_count() == 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// delete_nodes_for_file
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("delete_nodes_for_file removes all nodes from that file", "[graph][delete_file]") {
    Graph g;
    g.upsert_node(make_cue("cue-1", "a.cpp"));
    g.upsert_node(make_content("content-1", "a.cpp"));
    g.upsert_node(make_tag("tag-1", "b.cpp"));

    g.delete_nodes_for_file("a.cpp");

    REQUIRE(g.find("cue-1")     == INVALID_NODE);
    REQUIRE(g.find("content-1") == INVALID_NODE);
    REQUIRE(g.find("tag-1")     != INVALID_NODE); // b.cpp survives
    REQUIRE(g.node_count() == 1u);
}

TEST_CASE("delete_nodes_for_file removes file-scoped edges", "[graph][delete_file]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1", "a.cpp"));
    NodeIdx con = g.upsert_node(make_content("content-1", "a.cpp"));
    g.upsert_edge(make_edge(cue, con, EdgeType::CueOf, "a.cpp"));
    REQUIRE(g.edge_count() == 1u);

    g.delete_nodes_for_file("a.cpp");
    REQUIRE(g.edge_count() == 0u);
}

TEST_CASE("delete_nodes_for_file removes cross-file edge touching deleted node", "[graph][delete_file]") {
    // tag-keep is in b.cpp, but the REDIRECTS_TO edge points at cue-1 in a.cpp.
    Graph g;
    NodeIdx cue  = g.upsert_node(make_cue("cue-1",    "a.cpp"));
    NodeIdx keep = g.upsert_node(make_tag("tag-keep",  "b.cpp"));

    EdgeData cross;
    cross.source      = keep;
    cross.target      = cue;
    cross.type        = EdgeType::RedirectsTo;
    cross.source_file = "b.cpp";
    g.upsert_edge(cross);
    REQUIRE(g.edge_count() == 1u);

    g.delete_nodes_for_file("a.cpp");

    REQUIRE(g.find("cue-1")    == INVALID_NODE);
    REQUIRE(g.find("tag-keep") != INVALID_NODE); // b.cpp node survives
    REQUIRE(g.edge_count()     == 0u);           // cross-file edge gone
}

TEST_CASE("delete_nodes_for_file removes edge with null provenance via endpoint", "[graph][delete_file]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1", "a.cpp"));
    NodeIdx con = g.upsert_node(make_content("content-1", "a.cpp"));

    EdgeData e;
    e.source = cue;
    e.target = con;
    e.type   = EdgeType::CueOf;
    // No source_file set (empty string = null provenance).
    g.upsert_edge(e);
    REQUIRE(g.edge_count() == 1u);

    g.delete_nodes_for_file("a.cpp");
    REQUIRE(g.edge_count() == 0u);
}

TEST_CASE("delete_nodes_for_file on unknown file is a no-op", "[graph][delete_file]") {
    Graph g;
    g.upsert_node(make_cue("cue-1", "a.cpp"));
    REQUIRE_NOTHROW(g.delete_nodes_for_file("nonexistent.cpp"));
    REQUIRE(g.node_count() == 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// rebuild() compaction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("rebuild compacts soft-deleted (tombstone) nodes", "[graph][compaction]") {
    Graph g;
    g.upsert_node(make_cue("cue-1"));
    g.upsert_node(make_cue("cue-2"));

    // Soft-delete cue-1 via upsert with status=Deleted.
    NodeData tomb = make_cue("cue-1");
    tomb.status = NodeStatus::Deleted;
    g.upsert_node(tomb);

    // Before rebuild: cue-1 is still findable (tombstone in index).
    REQUIRE(g.find("cue-1") != INVALID_NODE);

    g.rebuild();

    // After rebuild: cue-1 is gone.
    REQUIRE(g.find("cue-1") == INVALID_NODE);
    REQUIRE(g.node_count() == 1u);
    REQUIRE(g.find("cue-2") != INVALID_NODE);
}

TEST_CASE("rebuild compacts hard-deleted node slots", "[graph][compaction]") {
    Graph g;
    NodeIdx idx = g.upsert_node(make_cue("cue-1"));
    g.upsert_node(make_cue("cue-2"));
    g.delete_node(idx);
    g.rebuild();

    REQUIRE(g.node_count() == 1u);
    REQUIRE(g.find("cue-2") != INVALID_NODE);
    // After compaction, cue-2 must map to a valid internal index.
    NodeIdx i2 = g.find("cue-2");
    REQUIRE(i2 != INVALID_NODE);
    REQUIRE(g.node(i2).id == "cue-2");
}

TEST_CASE("rebuild remaps indices after compaction", "[graph][compaction]") {
    // Three nodes; delete the first. After rebuild the second and third
    // get remapped to indices 0 and 1.
    Graph g;
    NodeIdx n0 = g.upsert_node(make_cue("n0"));
    NodeIdx n1 = g.upsert_node(make_cue("n1"));
    NodeIdx n2 = g.upsert_node(make_cue("n2"));
    g.upsert_edge(make_edge(n1, n2, EdgeType::CueOf));

    g.delete_node(n0);
    g.rebuild();

    REQUIRE(g.node_count() == 2u);
    NodeIdx i1 = g.find("n1");
    NodeIdx i2 = g.find("n2");
    REQUIRE(i1 != INVALID_NODE);
    REQUIRE(i2 != INVALID_NODE);

    // The edge must still work after remapping.
    REQUIRE(g.outgoing(i1).size() == 1u);
    REQUIRE(g.edge(g.outgoing(i1)[0]).target == i2);
}

TEST_CASE("rebuild removes edges whose endpoint was compacted away", "[graph][compaction]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    g.upsert_edge(make_edge(cue, con));

    // Soft-delete the content node.
    NodeData tomb = make_content("content-1");
    tomb.status = NodeStatus::Deleted;
    g.upsert_node(tomb);

    g.rebuild();

    REQUIRE(g.node_count() == 1u); // only cue survives
    REQUIRE(g.edge_count() == 0u); // orphaned edge removed
    NodeIdx cue_new = g.find("cue-1");
    REQUIRE(g.outgoing(cue_new).empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Multiple rebuild cycles
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("multiple rebuild cycles are idempotent on stable data", "[graph][rebuild]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));
    g.upsert_edge(make_edge(cue, con));

    g.rebuild();
    NodeIdx c1 = g.find("cue-1");
    g.rebuild(); // second rebuild on unchanged data
    NodeIdx c2 = g.find("cue-1");

    REQUIRE(c1 == c2);
    REQUIRE(g.node_count() == 2u);
    REQUIRE(g.edge_count() == 1u);
}

TEST_CASE("delete and re-index cycle within same transaction scope", "[graph][rebuild]") {
    Graph g;

    // First index pass.
    NodeIdx old_cue = g.upsert_node(make_cue("cue:symbol:a.cpp::foo", "a.cpp"));
    NodeIdx old_con = g.upsert_node(make_content("content:a.cpp::foo", "a.cpp"));
    g.upsert_edge(make_edge(old_cue, old_con, EdgeType::CueOf, "a.cpp"));
    g.rebuild();

    REQUIRE(g.node_count() == 2u);
    REQUIRE(g.edge_count() == 1u);

    // Simulate file change: delete and reindex.
    g.delete_nodes_for_file("a.cpp");
    REQUIRE(g.find("cue:symbol:a.cpp::foo") == INVALID_NODE);
    REQUIRE(g.find("content:a.cpp::foo")    == INVALID_NODE);

    NodeIdx new_cue = g.upsert_node(make_cue("cue:symbol:a.cpp::foo", "a.cpp"));
    NodeIdx new_con = g.upsert_node(make_content("content:a.cpp::foo", "a.cpp"));
    g.upsert_edge(make_edge(new_cue, new_con, EdgeType::CueOf, "a.cpp"));
    g.rebuild();

    REQUIRE(g.node_count() == 2u);
    REQUIRE(g.edge_count() == 1u);
    REQUIRE(g.find("cue:symbol:a.cpp::foo") != INVALID_NODE);
    REQUIRE(g.find("content:a.cpp::foo")    != INVALID_NODE);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge properties round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("edge fields are preserved through rebuild", "[graph][roundtrip]") {
    Graph g;
    NodeIdx cue = g.upsert_node(make_cue("cue-1"));
    NodeIdx con = g.upsert_node(make_content("content-1"));

    EdgeData e;
    e.source      = cue;
    e.target      = con;
    e.type        = EdgeType::CueOf;
    e.source_file = "foo.cpp";
    e.git_sha     = "deadbeef";
    e.indexed_at  = 9999LL;
    g.upsert_edge(e);
    g.rebuild();

    EdgeIdx ei = g.outgoing(cue)[0];
    const EdgeData& stored = g.edge(ei);
    REQUIRE(stored.type        == EdgeType::CueOf);
    REQUIRE(stored.source_file == "foo.cpp");
    REQUIRE(stored.git_sha     == "deadbeef");
    REQUIRE(stored.indexed_at  == 9999LL);
}

TEST_CASE("node fields are preserved through rebuild", "[graph][roundtrip]") {
    Graph g;
    NodeData n;
    n.id                      = "cue:symbol:a.cpp::Bar";
    n.type                    = NodeType::Cue;
    n.status                  = NodeStatus::Active;
    n.indexed_at              = 12345678LL;
    n.source_file             = "a.cpp";
    n.git_sha                 = "cafe1234";
    n.cue_type                = CueType::Symbol;
    n.text                    = "Bar";
    n.embedding               = {0.1f, 0.2f, 0.3f};
    n.embedding_model         = "test-model-v1";
    n.embedding_model_version = "2026-01";
    g.upsert_node(n);
    g.rebuild();

    NodeIdx idx = g.find("cue:symbol:a.cpp::Bar");
    const NodeData& s = g.node(idx);
    REQUIRE(s.id                      == "cue:symbol:a.cpp::Bar");
    REQUIRE(s.type                    == NodeType::Cue);
    REQUIRE(s.indexed_at              == 12345678LL);
    REQUIRE(s.source_file             == "a.cpp");
    REQUIRE(s.git_sha                 == "cafe1234");
    REQUIRE(s.cue_type                == CueType::Symbol);
    REQUIRE(s.text                    == "Bar");
    REQUIRE(s.embedding.size()        == 3u);
    REQUIRE(s.embedding[0]            == 0.1f);
    REQUIRE(s.embedding_model         == "test-model-v1");
    REQUIRE(s.embedding_model_version == "2026-01");
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph shape matching the paper schema (Cue→Content with Tags)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("full cue-tag-content graph shape survives rebuild", "[graph][schema]") {
    // Models the indexer output described in section 2.6 of the plan.
    Graph g;

    NodeIdx cue = g.upsert_node(make_cue("cue:symbol:foo.py::Bar", "foo.py"));
    NodeIdx con = g.upsert_node(make_content("content:foo.py::Bar", "foo.py"));

    NodeData mod_node;
    mod_node.id          = "content:foo.py::<module>";
    mod_node.type        = NodeType::Content;
    mod_node.kind        = ContentKind::Module;
    mod_node.memory_layer = MemoryLayer::Topic;
    mod_node.body        = "";
    mod_node.indexed_at  = 1LL;
    NodeIdx mod  = g.upsert_node(mod_node);

    NodeData lang_tag;
    lang_tag.id       = "tag:language:python";
    lang_tag.type     = NodeType::Tag;
    lang_tag.category = TagCategory::Language;
    lang_tag.value    = "python";
    lang_tag.indexed_at = 1LL;
    NodeIdx lang = g.upsert_node(lang_tag);

    g.upsert_edge(make_edge(cue, con,  EdgeType::CueOf,       "foo.py"));
    g.upsert_edge(make_edge(con, lang, EdgeType::TaggedWith,  "foo.py"));
    g.upsert_edge(make_edge(con, mod,  EdgeType::PartOfTopic, "foo.py"));
    g.rebuild();

    REQUIRE(g.node_count() == 4u);
    REQUIRE(g.edge_count() == 3u);

    // cue should have one outgoing edge to con
    auto cue_out = g.outgoing(g.find("cue:symbol:foo.py::Bar"));
    REQUIRE(cue_out.size() == 1u);
    REQUIRE(g.edge(cue_out[0]).type == EdgeType::CueOf);

    // con should have two outgoing edges
    NodeIdx con_idx = g.find("content:foo.py::Bar");
    auto con_out = g.outgoing(con_idx);
    REQUIRE(con_out.size() == 2u);

    // lang tag should have one incoming edge
    auto lang_inc = g.incoming(g.find("tag:language:python"));
    REQUIRE(lang_inc.size() == 1u);
    REQUIRE(g.edge(lang_inc[0]).type == EdgeType::TaggedWith);
}

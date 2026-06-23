#include <catch2/catch_test_macros.hpp>

#include "delfos/graph.hpp"

namespace {

delfos::NodeData make_cue(const char* id, const char* source_file, const char* text) {
    delfos::NodeData node{};
    node.id = id;
    node.type = delfos::NodeType::Cue;
    node.cue_type = delfos::CueType::Symbol;
    node.text = text;
    node.source_file = source_file;
    node.git_sha = "deadbeef";
    node.indexed_at = 1;
    return node;
}

}  // namespace

TEST_CASE("upsert_node replaces existing node by id", "[graph]") {
    delfos::Graph graph;

    const auto first = graph.upsert_node(make_cue("cue:1", "a.py", "old"));
    auto updated = make_cue("cue:1", "a.py", "new");
    const auto second = graph.upsert_node(updated);

    REQUIRE(first == second);
    const auto idx = graph.find("cue:1");
    REQUIRE(idx != delfos::INVALID_NODE);
    REQUIRE(graph.node(idx).text == "new");
}

TEST_CASE("rebuild creates traversable adjacency", "[graph]") {
    delfos::Graph graph;

    const auto cue = graph.upsert_node(make_cue("cue:1", "a.py", "symbol"));
    auto content = make_cue("content:1", "a.py", "content");
    content.type = delfos::NodeType::Content;
    const auto content_idx = graph.upsert_node(content);
    auto tag = make_cue("tag:language:python", "", "python");
    tag.type = delfos::NodeType::Tag;
    const auto tag_idx = graph.upsert_node(tag);

    delfos::EdgeData cue_of{};
    cue_of.source = cue;
    cue_of.target = content_idx;
    cue_of.type = delfos::EdgeType::CueOf;
    graph.upsert_edge(cue_of);

    delfos::EdgeData tagged{};
    tagged.source = content_idx;
    tagged.target = tag_idx;
    tagged.type = delfos::EdgeType::TaggedWith;
    graph.upsert_edge(tagged);

    graph.rebuild();

    const auto from_cue = graph.neighbors(cue, delfos::Direction::Outgoing);
    REQUIRE(from_cue.size() == 1);
    REQUIRE(from_cue.front() == content_idx);

    auto filter = delfos::EdgeType::TaggedWith;
    const auto from_content = graph.neighbors(content_idx, delfos::Direction::Outgoing, &filter);
    REQUIRE(from_content.size() == 1);
    REQUIRE(from_content.front() == tag_idx);
}

TEST_CASE("delete_nodes_for_file removes file-scoped graph data", "[graph]") {
    delfos::Graph graph;

    const auto file_node = graph.upsert_node(make_cue("cue:file", "tracked.py", "x"));
    const auto keep_node = graph.upsert_node(make_cue("cue:keep", "other.py", "y"));

    delfos::EdgeData edge{};
    edge.source = file_node;
    edge.target = keep_node;
    edge.type = delfos::EdgeType::RedirectsTo;
    edge.source_file = "tracked.py";
    graph.upsert_edge(edge);

    graph.delete_nodes_for_file("tracked.py");
    graph.rebuild();

    REQUIRE(graph.find("cue:file") == delfos::INVALID_NODE);
    REQUIRE(graph.find("cue:keep") != delfos::INVALID_NODE);
    REQUIRE(graph.edge_count() == 0);
}

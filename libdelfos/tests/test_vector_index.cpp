#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "delfos/vector_index.hpp"

using namespace delfos;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM = 8;

// Unit vector with 1.0 in position i and 0.0 elsewhere.
static std::vector<float> unit(std::size_t i, std::size_t dim = DIM) {
    std::vector<float> v(dim, 0.0f);
    v[i % dim] = 1.0f;
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("VectorIndex constructs with given dim", "[vector_index][construct]") {
    VectorIndex vi(DIM);
    REQUIRE(vi.dim()  == DIM);
    REQUIRE(vi.size() == 0u);
}

TEST_CASE("VectorIndex constructs with large dim (1536)", "[vector_index][construct]") {
    VectorIndex vi(1536);
    REQUIRE(vi.dim()  == 1536u);
    REQUIRE(vi.size() == 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert / size
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("insert increases size", "[vector_index][insert]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    REQUIRE(vi.size() == 1u);
    vi.insert(1, NodeType::Cue, unit(1));
    REQUIRE(vi.size() == 2u);
}

TEST_CASE("duplicate insert (same NodeIdx) does not grow size", "[vector_index][insert]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    vi.insert(0, NodeType::Cue, unit(1)); // same key, updated vector
    // size may stay 1 or remain the same after replacement
    REQUIRE(vi.size() == 1u);
}

TEST_CASE("insert nodes of different types", "[vector_index][insert]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue,     unit(0));
    vi.insert(1, NodeType::Content, unit(1));
    vi.insert(2, NodeType::Tag,     unit(2));
    REQUIRE(vi.size() == 3u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Remove
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("remove decreases size", "[vector_index][remove]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    vi.insert(1, NodeType::Cue, unit(1));
    vi.remove(0);
    REQUIRE(vi.size() == 1u);
}

TEST_CASE("remove non-existent node is a no-op", "[vector_index][remove]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    REQUIRE_NOTHROW(vi.remove(42));
    REQUIRE(vi.size() == 1u);
}

TEST_CASE("remove on empty index is a no-op", "[vector_index][remove]") {
    VectorIndex vi(DIM);
    REQUIRE_NOTHROW(vi.remove(0));
    REQUIRE(vi.size() == 0u);
}

TEST_CASE("removed node does not appear in search results", "[vector_index][remove]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    vi.insert(1, NodeType::Cue, unit(1));
    vi.remove(0);

    auto hits = vi.search(unit(0), 5);
    for (const auto& h : hits) {
        REQUIRE(h.node != 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search — empty index
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("search on empty index returns empty result (no crash)", "[vector_index][search]") {
    VectorIndex vi(DIM);
    auto hits = vi.search(unit(0), 5);
    REQUIRE(hits.empty());
}

TEST_CASE("search with k=0 returns empty result", "[vector_index][search]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    auto hits = vi.search(unit(0), 0);
    REQUIRE(hits.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Search — correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("search returns correct nearest neighbour", "[vector_index][search]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    vi.insert(1, NodeType::Cue, unit(1));
    vi.insert(2, NodeType::Cue, unit(2));

    auto hits = vi.search(unit(0), 1);
    REQUIRE(hits.size() == 1u);
    REQUIRE(hits[0].node == 0u);
    REQUIRE(hits[0].score == Catch::Approx(1.0f).epsilon(1e-4f));
}

TEST_CASE("search results sorted by descending score", "[vector_index][search]") {
    VectorIndex vi(DIM);
    // unit(0) is most similar to query unit(0); unit(1) is orthogonal.
    vi.insert(0, NodeType::Cue, unit(0));
    vi.insert(1, NodeType::Cue, unit(1));

    auto hits = vi.search(unit(0), 2);
    REQUIRE(hits.size() == 2u);
    REQUIRE(hits[0].score >= hits[1].score);
    REQUIRE(hits[0].node  == 0u);
    REQUIRE(hits[0].score == Catch::Approx(1.0f).epsilon(1e-4f));
}

TEST_CASE("search with k larger than index size returns all nodes", "[vector_index][search]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    vi.insert(1, NodeType::Cue, unit(1));

    auto hits = vi.search(unit(0), 100);
    REQUIRE(hits.size() == 2u);
}

TEST_CASE("search respects k limit", "[vector_index][search]") {
    VectorIndex vi(DIM);
    for (NodeIdx i = 0; i < 8; ++i)
        vi.insert(i, NodeType::Cue, unit(i));

    auto hits = vi.search(unit(0), 3);
    REQUIRE(hits.size() == 3u);
}

TEST_CASE("cosine score of identical vectors is ~1.0", "[vector_index][search]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    auto hits = vi.search(unit(0), 1);
    REQUIRE(!hits.empty());
    REQUIRE(hits[0].score == Catch::Approx(1.0f).epsilon(1e-4f));
}

TEST_CASE("cosine score of orthogonal vectors is ~0.0", "[vector_index][search]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0)); // e0
    vi.insert(1, NodeType::Cue, unit(1)); // e1 — orthogonal to e0

    auto hits = vi.search(unit(0), 2);
    REQUIRE(hits.size() == 2u);
    // The second hit (e1) should have near-zero similarity.
    float second_score = hits[1].score;
    REQUIRE(second_score == Catch::Approx(0.0f).margin(0.01f));
}

// ─────────────────────────────────────────────────────────────────────────────
// Search — type filter
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("type filter restricts results to cue nodes", "[vector_index][filter]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue,     unit(0));
    vi.insert(1, NodeType::Content, unit(0)); // same direction, but Content type

    NodeType cue_filter = NodeType::Cue;
    auto hits = vi.search(unit(0), 5, &cue_filter);
    for (const auto& h : hits) {
        REQUIRE(h.node == 0u); // only cue-typed node
    }
    REQUIRE(hits.size() == 1u);
}

TEST_CASE("type filter restricts results to content nodes", "[vector_index][filter]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue,     unit(0));
    vi.insert(1, NodeType::Content, unit(0));

    NodeType con_filter = NodeType::Content;
    auto hits = vi.search(unit(0), 5, &con_filter);
    REQUIRE(hits.size() == 1u);
    REQUIRE(hits[0].node == 1u);
}

TEST_CASE("type filter returns empty when no nodes of that type exist", "[vector_index][filter]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));

    NodeType tag_filter = NodeType::Tag;
    auto hits = vi.search(unit(0), 5, &tag_filter);
    REQUIRE(hits.empty());
}

TEST_CASE("type filter with multiple nodes of target type", "[vector_index][filter]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue, unit(0));
    vi.insert(1, NodeType::Cue, unit(1));
    vi.insert(2, NodeType::Cue, unit(2));
    vi.insert(3, NodeType::Content, unit(0)); // same direction, different type

    NodeType cue_filter = NodeType::Cue;
    auto hits = vi.search(unit(0), 5, &cue_filter);
    for (const auto& h : hits) {
        // All returned nodes must be Cue type (indices 0, 1, 2).
        REQUIRE(h.node != 3u);
    }
    REQUIRE(hits.size() == 3u);
}

TEST_CASE("null type_filter returns all types", "[vector_index][filter]") {
    VectorIndex vi(DIM);
    vi.insert(0, NodeType::Cue,     unit(0));
    vi.insert(1, NodeType::Content, unit(1));
    vi.insert(2, NodeType::Tag,     unit(2));

    auto hits = vi.search(unit(0), 5, nullptr);
    REQUIRE(hits.size() == 3u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistence (save / load round-trip)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("save and load preserves search results", "[vector_index][persistence]") {
    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "delfos_test_vi.usearch";

    {
        VectorIndex vi(DIM);
        vi.insert(0, NodeType::Cue, unit(0));
        vi.insert(1, NodeType::Cue, unit(1));
        vi.save(tmp);
    }

    VectorIndex vi2(DIM);
    vi2.load(tmp);
    REQUIRE(vi2.dim()  == DIM);
    REQUIRE(vi2.size() == 2u);

    auto hits = vi2.search(unit(0), 2);
    REQUIRE(!hits.empty());
    REQUIRE(hits[0].node == 0u);
    REQUIRE(hits[0].score == Catch::Approx(1.0f).epsilon(1e-4f));

    std::filesystem::remove(tmp);
}

TEST_CASE("load into existing index replaces it", "[vector_index][persistence]") {
    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "delfos_test_vi2.usearch";

    {
        VectorIndex vi(DIM);
        vi.insert(0, NodeType::Cue, unit(0));
        vi.save(tmp);
    }

    VectorIndex vi2(DIM);
    vi2.insert(99, NodeType::Content, unit(3));  // populate with different data
    vi2.load(tmp);

    // After load, the index should reflect the saved state (1 vector, not 2).
    REQUIRE(vi2.size() == 1u);

    std::filesystem::remove(tmp);
}

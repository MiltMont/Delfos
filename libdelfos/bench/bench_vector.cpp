// Microbenchmarks for VectorIndex (Phase 2).
//
// Targets from docs/libdelfos-plan.md §7.2:
//   bench_vector_search  50K random dim=1536 vectors, k=5, p99 < 1ms
//
// Run with:
//   cmake --preset release && cmake --build build/release
//   ./build/release/libdelfos/bench/bench_vector

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <cstdlib>
#include <random>
#include <vector>

#include "delfos/vector_index.hpp"

using namespace delfos;

static std::vector<float> random_unit(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    float len = 0.0f;
    for (auto& x : v) {
        x = dist(rng);
        len += x * x;
    }
    len = std::sqrt(len);
    if (len > 0.0f)
        for (auto& x : v) x /= len;
    return v;
}

int main() {
    std::mt19937 rng(42);
    constexpr std::size_t N   = 50'000;
    constexpr std::size_t DIM = 1536;
    constexpr std::size_t K   = 5;

    // ── Build index ────────────────────────────────────────────────────────
    VectorIndex vi(DIM, N + 1024);

    std::vector<std::vector<float>> vectors;
    vectors.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        vectors.push_back(random_unit(DIM, rng));
        vi.insert(static_cast<NodeIdx>(i), NodeType::Cue, vectors.back());
    }

    // Prepare a random query vector.
    std::vector<float> query = random_unit(DIM, rng);

    ankerl::nanobench::Bench bench;
    bench.title("VectorIndex")
         .unit("query")
         .warmup(50)
         .minEpochIterations(200)
         .performanceCounters(true);

    // ── Benchmark: unfiltered k-NN search ─────────────────────────────────
    bench.run("search k=5 | 50K × dim=1536", [&] {
        auto hits = vi.search(query, K);
        ankerl::nanobench::doNotOptimizeAway(hits);
    });

    // ── Benchmark: type-filtered search (Cue only, all nodes are Cue) ─────
    NodeType cue_filter = NodeType::Cue;
    bench.run("filtered_search k=5 Cue | 50K × dim=1536", [&] {
        auto hits = vi.search(query, K, &cue_filter);
        ankerl::nanobench::doNotOptimizeAway(hits);
    });

    return 0;
}

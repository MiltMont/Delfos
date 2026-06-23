# ── Phase 2: vector index benchmark ──────────────────────────────────────────
add_executable(bench_vector bench_vector.cpp)
target_link_libraries(bench_vector PRIVATE delfos_core nanobench::nanobench)
target_compile_options(bench_vector PRIVATE -Wall -Wextra)
# No -Werror on bench target so nanobench headers don't break with
# future compiler versions.

# Phase 1 benchmarks (graph rebuild, neighbors) will be added in a follow-up.
# bench_graph_rebuild  (10K nodes / 50K edges rebuild < 10ms)
# bench_neighbors      (single neighbors() call < 1μs)
# bench_query_path     (100 × neighbors() + 1 × vector_search < 1ms)

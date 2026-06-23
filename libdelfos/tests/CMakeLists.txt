include(CTest)
include(Catch)

# ── Phase 1: graph tests ──────────────────────────────────────────────────────
add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE delfos_core Catch2::Catch2WithMain)
target_compile_options(test_graph PRIVATE -Wall -Wextra -Werror)
catch_discover_tests(test_graph)

# ── Phase 2: vector index tests ───────────────────────────────────────────────
add_executable(test_vector_index test_vector_index.cpp)
target_link_libraries(test_vector_index PRIVATE delfos_core Catch2::Catch2WithMain)
target_compile_options(test_vector_index PRIVATE -Wall -Wextra -Werror)
catch_discover_tests(test_vector_index)

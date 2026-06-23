add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE delfos_core Catch2::Catch2WithMain)

# Enforce strict warning flags on our own test code.
target_compile_options(test_graph PRIVATE -Wall -Wextra -Werror)

include(CTest)
include(Catch)
catch_discover_tests(test_graph)

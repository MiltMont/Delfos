cmake_minimum_required(VERSION 3.20)
project(delfos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# ── Phase 1 test dependency ───────────────────────────────────────────────────
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    ON)
FetchContent_MakeAvailable(Catch2)

# ── Phase 2: USearch (HNSW vector index) — fetched when vector_index.hpp lands
# FetchContent_Declare(usearch
#     GIT_REPOSITORY https://github.com/unum-cloud/usearch
#     GIT_TAG        main
#     GIT_SHALLOW    ON)
# FetchContent_MakeAvailable(usearch)

# ── Phase 3: FlatBuffers (snapshot serialisation) — fetched when snapshot.hpp lands
# FetchContent_Declare(flatbuffers
#     GIT_REPOSITORY https://github.com/google/flatbuffers
#     GIT_TAG        v24.3.25
#     GIT_SHALLOW    ON)
# set(FLATBUFFERS_BUILD_TESTS OFF)
# FetchContent_MakeAvailable(flatbuffers)

# ── Core interface library (header-only) ──────────────────────────────────────
add_library(delfos_core INTERFACE)
target_include_directories(delfos_core INTERFACE libdelfos/include)
# Phase 2/3 will add: target_link_libraries(delfos_core INTERFACE usearch flatbuffers)

# ── Tests ─────────────────────────────────────────────────────────────────────
option(DELFOS_BUILD_TESTS "Build unit tests" ON)
if(DELFOS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(libdelfos/tests)
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
option(DELFOS_BUILD_BENCH "Build benchmarks" ON)
if(DELFOS_BUILD_BENCH)
    add_subdirectory(libdelfos/bench)
endif()

# ── Python bindings ───────────────────────────────────────────────────────────
# Phase 4: nanobind module (_delfos) and NativeGraphStore.
option(DELFOS_BUILD_BINDINGS "Build nanobind Python module" OFF)
if(DELFOS_BUILD_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    # FetchContent_Declare(nanobind ...) will be added in Phase 4
    add_subdirectory(libdelfos/bindings)
endif()

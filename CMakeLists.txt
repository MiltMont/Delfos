cmake_minimum_required(VERSION 3.20)
project(delfos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# ── Catch2 (unit tests) ───────────────────────────────────────────────────────
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    ON)
FetchContent_MakeAvailable(Catch2)

# ── USearch (HNSW vector index, Phase 2) ─────────────────────────────────────
FetchContent_Declare(usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG        v2.16.5
    GIT_SHALLOW    ON)
set(USEARCH_BUILD_TEST_CPP   OFF CACHE BOOL "" FORCE)
set(USEARCH_BUILD_BENCH_CPP  OFF CACHE BOOL "" FORCE)
set(USEARCH_BUILD_LIB_C      OFF CACHE BOOL "" FORCE)
set(USEARCH_BUILD_SQLITE     OFF CACHE BOOL "" FORCE)
# Disable the fp16 library shim — it pulls in a header that isn't installed.
# We don't use float16 vectors, so this is safe to turn off.
set(USEARCH_USE_FP16LIB      OFF CACHE BOOL "" FORCE)
set(USEARCH_USE_SIMSIMD      OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(usearch)

# ── nanobench (microbenchmarks, Phase 2) ─────────────────────────────────────
FetchContent_Declare(nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG        v4.3.11
    GIT_SHALLOW    ON)
FetchContent_MakeAvailable(nanobench)

# ── FlatBuffers (snapshot serialisation, Phase 3) ─────────────────────────────
FetchContent_Declare(flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG        v24.3.25
    GIT_SHALLOW    ON)
set(FLATBUFFERS_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL        OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB  ON  CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC    ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flatbuffers)

# ── Core interface library (header-only) ──────────────────────────────────────
add_library(delfos_core INTERFACE)
target_include_directories(delfos_core INTERFACE
    libdelfos/include
    libdelfos/flatbuffers   # for delfos_generated.h
)
target_link_libraries(delfos_core INTERFACE usearch flatbuffers)

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

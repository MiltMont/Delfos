cmake_minimum_required(VERSION 3.28)

project(delfos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_library(delfos_core INTERFACE)
target_include_directories(
    delfos_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/libdelfos/include
)
target_compile_options(
    delfos_core
    INTERFACE
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -Werror>
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -Werror>
)

option(DELFOS_BUILD_TESTS "Build libdelfos unit tests" ON)
if(DELFOS_BUILD_TESTS)
    include(CTest)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.6.0
        GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(catch2)

    add_subdirectory(libdelfos/tests)
endif()

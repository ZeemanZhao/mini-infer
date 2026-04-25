cmake_minimum_required(VERSION 3.20)
project(mini_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS          "Build GoogleTest suite"          ON)
option(BUILD_PYTHON_BINDING "Build pybind11 Python binding"   OFF)

# ── Core library ─────────────────────────────────────────────────────────────
add_library(mini_infer_core
    src/tensor.cpp
    src/ops/matmul.cpp
    src/ops/relu.cpp
)
target_include_directories(mini_infer_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

# ── Demo executable ───────────────────────────────────────────────────────────
add_executable(mini_infer src/main.cpp)
target_link_libraries(mini_infer PRIVATE mini_infer_core)
target_compile_definitions(mini_infer PRIVATE
    DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(test_mlp_accuracy tests/test_mlp_accuracy.cpp)
    target_link_libraries(test_mlp_accuracy PRIVATE mini_infer_core GTest::gtest_main)
    target_compile_definitions(test_mlp_accuracy PRIVATE
        DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    )

    include(GoogleTest)
    gtest_discover_tests(test_mlp_accuracy)
endif()

# ── Python binding — reserved for Phase 1 ────────────────────────────────────
if(BUILD_PYTHON_BINDING)
    message(FATAL_ERROR "Python binding not yet implemented (Phase 1).")
endif()

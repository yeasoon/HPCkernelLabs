cmake_minimum_required(VERSION 3.14)
project(KernelBench VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(
        -O3
        -march=native
        -mfma
        -mavx2
        -funroll-loops
        -fno-omit-frame-pointer   # needed for perf call-graph
    )
endif()

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Main benchmark executable ─────────────────────────────────────────────────
add_executable(bench example.cpp)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ── Optional: link your kernel.cpp if it exists ───────────────────────────────
# add_executable(bench bench_kernel.cpp kernel.cpp)

# ── Optional: PAPI hardware counters ─────────────────────────────────────────
# find_library(PAPI_LIB papi)
# if(PAPI_LIB)
#     target_compile_definitions(bench PRIVATE USE_PAPI)
#     target_link_libraries(bench PRIVATE ${PAPI_LIB})
#     message(STATUS "PAPI found: ${PAPI_LIB}")
# endif()

# ── Optional: OpenMP for parallel benchmarks ─────────────────────────────────
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)
# endif()

# ── Print config ──────────────────────────────────────────────────────────────
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
cmake_minimum_required(VERSION 3.10)

# Project metadata
project(NaiveMatmul VERSION 1.0 LANGUAGES CXX)

# Use C++17 standard for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CRITICAL: Enable high-level optimizations
# Without this, the compiler won't try to vectorize your loops.
if(MSVC)
    # Optimizations for Visual Studio
    add_compile_options(/O2 /arch:AVX2)
else()
    # Optimizations for GCC/Clang
    # add_compile_options(-O3 -march=native -ffast-math)
    add_compile_options(-O3 -march=native -mfma -mavx2 -funroll-loops)
endif()

# Define the executable
# Replace 'main.cpp' with the actual name of your source file
add_executable(matmul_cpu naive.cpp)
add_executable(microkernel microkernel.cpp)
# Optional: Print build type for confirmation
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
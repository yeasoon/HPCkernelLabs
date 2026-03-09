cmake_minimum_required(VERSION 3.18)
project(naive_matmul CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Auto-detect GPU arch, or set manually e.g. 80=A100, 89=RTX4090, 90=H100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(matmul naive.cu)

target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
)
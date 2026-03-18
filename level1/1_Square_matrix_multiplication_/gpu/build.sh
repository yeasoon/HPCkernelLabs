# nvcc -O2 -o matmul src/naive.cu && ./matmul
# nvcc -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_101a,code=sm_101a   -o matmul src/naive.cu
rm -rf build && mkdir -p build && cd build
cmake ..
cmake --build .
./matmul

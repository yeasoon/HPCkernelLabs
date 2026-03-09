# nvcc -O2 -o matmul naive.cu && ./matmul
rm -rf build && mkdir -p build && cd build
cmake ..
cmake --build .
./matmul
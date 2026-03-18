# 1. Create a build directory to keep the source clean
rm -rf build && mkdir -p build && cd build

# 2. Configure the project
cmake ..

# 3. Build the executable
cmake --build . --config Release

# 4. Run it
# ./matmul_cpu   # (or .\Release\matmul_cpu.exe on Windows)
./microkernel


# g++ -O3 -march=native -mfma -funroll-loops -std=c++17 -pthread -o roofline src/roofline.cpp && ./roofline && numactl --cpunodebind=0 --membind=0 ./roofline
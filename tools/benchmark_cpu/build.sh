mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

./bench                        # defaults
./bench --iters 20             # 20 minimum iterations
./bench --duration 3.0         # run for at least 3 seconds each
./bench --csv results.csv      # export to CSV
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>
#include <numeric>
#include <algorithm>
#include <cstring>

// ============================================================
// PEAK COMPUTE BENCHMARK (FMA-heavy floating point ops)
// ============================================================
double benchmark_peak_compute(int duration_seconds = 3)
{
    const int UNROLL = 8;
    volatile double a[UNROLL], b[UNROLL], c[UNROLL];
    for (int i = 0; i < UNROLL; i++)
    {
        a[i] = 1.0001;
        b[i] = 1.0002;
        c[i] = 0.5;
    }

    long long ops = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end_time = start + std::chrono::seconds(duration_seconds);

    while (std::chrono::high_resolution_clock::now() < end_time)
    {
        for (int i = 0; i < 1000; i++)
        {
            // FMA: 2 FLOPS per iteration * UNROLL
            for (int j = 0; j < UNROLL; j++)
                a[j] = a[j] * b[j] + c[j];
        }
        ops += 2LL * 1000 * UNROLL;
    }

    auto elapsed = std::chrono::duration<double>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count();

    return (double)ops / elapsed / 1e9; // GFLOPS
}

// ============================================================
// MEMORY BANDWIDTH BENCHMARK (sequential read + write)
// ============================================================
double benchmark_memory_bandwidth(int duration_seconds = 3)
{
    // Use a large buffer (256 MB) to overflow caches
    const size_t N = 32 * 1024 * 1024; // 32M doubles = 256 MB
    std::vector<double> src(N, 1.23456), dst(N, 0.0);

    long long bytes = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end_time = start + std::chrono::seconds(duration_seconds);

    while (std::chrono::high_resolution_clock::now() < end_time)
    {
        // STREAM-style copy: read + write
        std::memcpy(dst.data(), src.data(), N * sizeof(double));
        bytes += 2LL * N * sizeof(double); // read + write
    }

    auto elapsed = std::chrono::duration<double>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count();

    return (double)bytes / elapsed / 1e9; // GB/s
}

// ============================================================
// MULTI-THREADED WRAPPERS
// ============================================================
double mt_peak_compute(int nthreads, int duration_seconds = 3)
{
    std::vector<double> results(nthreads, 0.0);
    std::vector<std::thread> threads;

    for (int t = 0; t < nthreads; t++)
    {
        threads.emplace_back([&, t]()
                             { results[t] = benchmark_peak_compute(duration_seconds); });
    }
    for (auto &th : threads)
        th.join();

    return std::accumulate(results.begin(), results.end(), 0.0);
}

double mt_memory_bandwidth(int nthreads, int duration_seconds = 3)
{
    std::vector<double> results(nthreads, 0.0);
    std::vector<std::thread> threads;

    for (int t = 0; t < nthreads; t++)
    {
        threads.emplace_back([&, t]()
                             { results[t] = benchmark_memory_bandwidth(duration_seconds); });
    }
    for (auto &th : threads)
        th.join();

    return std::accumulate(results.begin(), results.end(), 0.0);
}

// ============================================================
// MAIN
// ============================================================
int main()
{
    int nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 1;

    std::cout << "================================================\n";
    std::cout << "  CPU Peak Benchmark\n";
    std::cout << "================================================\n";
    std::cout << "Logical cores detected : " << nthreads << "\n\n";

    // --- Single-core ---
    std::cout << "[1/4] Single-core peak compute (3s)...\n";
    double sc_gflops = benchmark_peak_compute(3);
    std::cout << "  -> " << sc_gflops << " GFLOPS\n\n";

    std::cout << "[2/4] Single-core memory bandwidth (3s)...\n";
    double sc_bw = benchmark_memory_bandwidth(3);
    std::cout << "  -> " << sc_bw << " GB/s\n\n";

    // --- Multi-core ---
    std::cout << "[3/4] Multi-core peak compute (" << nthreads << " threads, 3s)...\n";
    double mc_gflops = mt_peak_compute(nthreads, 3);
    std::cout << "  -> " << mc_gflops << " GFLOPS\n\n";

    std::cout << "[4/4] Multi-core memory bandwidth (" << nthreads << " threads, 3s)...\n";
    double mc_bw = mt_memory_bandwidth(nthreads, 3);
    std::cout << "  -> " << mc_bw << " GB/s\n\n";

    // --- Summary ---
    std::cout << "================================================\n";
    std::cout << "  RESULTS SUMMARY\n";
    std::cout << "================================================\n";
    std::cout << "  Single-core compute  : " << sc_gflops << " GFLOPS\n";
    std::cout << "  Multi-core compute   : " << mc_gflops << " GFLOPS\n";
    std::cout << "  Single-core BW       : " << sc_bw << " GB/s\n";
    std::cout << "  Multi-core BW        : " << mc_bw << " GB/s\n";
    std::cout << "================================================\n";
    std::cout << "\nNote: GFLOPS measures scalar FMA throughput.\n";
    std::cout << "For SIMD/AVX peak, compile with -O3 -march=native.\n";

    return 0;
}
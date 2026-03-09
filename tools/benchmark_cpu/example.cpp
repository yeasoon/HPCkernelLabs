// =============================================================================
// bench_kernel.cpp  —  Example: benchmark your kernel.h / kernel.cpp
//
// Compile (single file):
//   g++ -O3 -march=native -mfma -mavx2 -std=c++17 \
//       -o bench bench_kernel.cpp -I.
//
// Compile (separate kernel):
//   g++ -O3 -march=native -mfma -mavx2 -std=c++17 \
//       -o bench bench_kernel.cpp kernel.cpp -I.
//
// Run:
//   ./bench                          # default settings
//   ./bench --iters 20               # minimum 20 iterations
//   ./bench --duration 3.0           # run each kernel for at least 3s
//   ./bench --warmup 5               # 5 warmup iterations
//   ./bench --csv results.csv        # export results to CSV
//   ./bench --no-hist                # skip latency histogram
// =============================================================================

#define BENCH_MAIN // activate built-in main() in benchmark.hpp
#include "benchmark.hpp"

// ── Include YOUR kernel here ─────────────────────────────────────────────────
// #include "kernel.h"
// If your kernel is header-only, just include it.
// If it has a .cpp, add it to the compile command.

// For this example we inline simple kernels to make it self-contained:
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma")
#include <immintrin.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// YOUR KERNEL SIGNATURES — replace with your actual kernel headers
// ─────────────────────────────────────────────────────────────────────────────
static void naive_matmul(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
        {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

static void blocked_matmul(const float *A, const float *B, float *C,
                           int N, int BS = 1024)
{
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
            {
                int ie = std::min(ii + BS, N), ke = std::min(kk + BS, N), je = std::min(jj + BS, N);
                for (int i = ii; i < ie; i++)
                    for (int k = kk; k < ke; k++)
                    {
                        float tmp = A[i * N + k];
                        for (int j = jj; j < je; j++)
                            C[i * N + j] += tmp * B[k * N + j];
                    }
            }
}

static void avx2_matmul(const float *A, const float *B, float *C,
                        int N, int BS = 1024)
{
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
            {
                int ie = std::min(ii + BS, N), ke = std::min(kk + BS, N), je = std::min(jj + BS, N);
                for (int i = ii; i < ie; i++)
                    for (int k = kk; k < ke; k++)
                    {
                        __m256 tmp = _mm256_set1_ps(A[i * N + k]);
                        __builtin_prefetch(&B[(k + 1) * N + jj], 0, 1);
                        int j = jj, w = je - jj;
                        for (; j <= jj + w - 32; j += 32)
                        {
                            __m256 c0 = _mm256_loadu_ps(C + i * N + j), c1 = _mm256_loadu_ps(C + i * N + j + 8);
                            __m256 c2 = _mm256_loadu_ps(C + i * N + j + 16), c3 = _mm256_loadu_ps(C + i * N + j + 24);
                            c0 = _mm256_fmadd_ps(tmp, _mm256_loadu_ps(B + k * N + j), c0);
                            c1 = _mm256_fmadd_ps(tmp, _mm256_loadu_ps(B + k * N + j + 8), c1);
                            c2 = _mm256_fmadd_ps(tmp, _mm256_loadu_ps(B + k * N + j + 16), c2);
                            c3 = _mm256_fmadd_ps(tmp, _mm256_loadu_ps(B + k * N + j + 24), c3);
                            _mm256_storeu_ps(C + i * N + j, c0);
                            _mm256_storeu_ps(C + i * N + j + 8, c1);
                            _mm256_storeu_ps(C + i * N + j + 16, c2);
                            _mm256_storeu_ps(C + i * N + j + 24, c3);
                        }
                        for (; j < je; j++)
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
            }
}

// ─────────────────────────────────────────────────────────────────────────────
// BENCHMARK REGISTRATION  (called by BENCH_MAIN's main())
// ─────────────────────────────────────────────────────────────────────────────
// Note: benchmark.hpp was already included above.
// BENCH_MAIN must be defined BEFORE the first include to activate main().
// Move the #define BENCH_MAIN to the very top of this file if needed.

void register_benchmarks(bench::Suite &suite, int /*argc*/, char ** /*argv*/)
{

    // ── problem size ─────────────────────────────────────────────────────────
    const int N = 4096; // ← change to your kernel's problem size
    const int BS = 1024;

    // pre-allocate inputs / outputs  (allocation is NOT measured)
    static std::vector<float> A(N * N), B(N * N), C(N * N);
    static std::vector<float> Cref(N * N);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = (i % 7) * 0.1f;
        B[i] = (i % 5) * 0.1f;
    }

    // FLOPs and bytes for roofline numbers
    double flops = 2.0 * N * N * N;             // matmul: 2*N^3
    double bytes = 3.0 * N * N * sizeof(float); // read A,B + write C

    // ── correctness reference ─────────────────────────────────────────────────
    // Run once to build reference output
    std::fill(Cref.begin(), Cref.end(), 0.0f);
    naive_matmul(A.data(), B.data(), Cref.data(), N);

    // ── ADD YOUR KERNELS HERE ─────────────────────────────────────────────────
    // Pattern:
    //   suite.add("name", lambda, flops_per_call, bytes_per_call, "group");
    //
    // The lambda captures everything it needs. Reset output arrays inside if
    // your kernel accumulates (C += A*B) so results stay consistent.

    suite.add("naive_matmul", [&]
              {
            std::fill(C.begin(), C.end(), 0.0f);
            naive_matmul(A.data(), B.data(), C.data(), N);
            bench::do_not_optimise(C[0]); }, flops, bytes, "matmul");

    suite.add("blocked_matmul BS=1024", [&]
              {
            std::fill(C.begin(), C.end(), 0.0f);
            blocked_matmul(A.data(), B.data(), C.data(), N, BS);
            bench::do_not_optimise(C[0]); }, flops, bytes, "matmul");

    suite.add_checked(
        "avx2_matmul BS=1024",
        // kernel lambda
        [&]
        {
            std::fill(C.begin(), C.end(), 0.0f);
            avx2_matmul(A.data(), B.data(), C.data(), N, BS);
            bench::do_not_optimise(C[0]);
        },
        // correctness checker lambda — compare against reference
        [&]() -> bool
        {
            std::fill(C.begin(), C.end(), 0.0f);
            avx2_matmul(A.data(), B.data(), C.data(), N, BS);
            float max_err = 0.0f;
            for (int i = 0; i < N * N; i++)
                max_err = std::max(max_err, std::abs(C[i] - Cref[i]));
            if (max_err > 1e-2f)
            {
                std::cerr << "  max_err = " << max_err << "\n";
                return false;
            }
            return true;
        },
        flops, bytes, "matmul");

    // ── EXAMPLE: benchmark a memory-bound kernel ─────────────────────────────
    static std::vector<float> src(N * N, 1.0f), dst(N * N, 0.0f);
    double copy_bytes = 2.0 * N * N * sizeof(float); // read + write

    suite.add("memcpy baseline", [&]
              {
            std::memcpy(dst.data(), src.data(), N*N*sizeof(float));
            bench::do_not_optimise(dst[0]); },
              /*flops=*/0, copy_bytes, "memory");

    // ── TEMPLATE: plug in your own kernel ────────────────────────────────────
    // Uncomment and adapt:
    //
    // suite.add("my_kernel",
    //     [&]{
    //         my_kernel(input.data(), output.data(), N);
    //         bench::do_not_optimise(output[0]);
    //     },
    //     /*flops=*/ 2.0 * N * N * N,    // set to 0 if unknown
    //     /*bytes=*/ 3.0 * N * N * sizeof(float),
    //     "my_group");
}
#include <iostream>
#include <vector>
#include <chrono>

/**
 * Naive Matrix Multiplication (C = A * B)
 * @param A Input matrix of size N x N
 * @param B Input matrix of size N x N
 * @param C Output matrix of size N x N
 * @param N Dimension of the square matrices
 */
void naive_matmul(const std::vector<float> &A,
                  const std::vector<float> &B,
                  std::vector<float> &C, int N)
{
    for (int i = 0; i < N; ++i)
    { // Row of A
        for (int j = 0; j < N; ++j)
        { // Column of B
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            { // Dot product inner loop
                // Indexing: [row * N + col]
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Naive Matrix Multiplication (C = A * B)
 * @param A Input matrix of size N x N
 * @param B Input matrix of size N x N
 * @param C Output matrix of size N x N
 * @param N Dimension of the square matrices
 */
void cache_friendly_matmul(const std::vector<float> &A,
                           const std::vector<float> &B,
                           std::vector<float> &C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < N; ++k)
        { // Swap J and K
            float temp = A[i * N + k];
            for (int j = 0; j < N; ++j)
            {
                C[i * N + j] += temp * B[k * N + j];
            }
        }
    }
}

void cache_blocked_matmul(const std::vector<float> &A,
                          const std::vector<float> &B,
                          std::vector<float> &C,
                          int N, int BS = 64)
{
    // const int BS = 64; // block size (tune for cache)

    for (int ii = 0; ii < N; ii += BS)
    {
        for (int kk = 0; kk < N; kk += BS)
        {
            for (int jj = 0; jj < N; jj += BS)
            {

                for (int i = ii; i < std::min(ii + BS, N); ++i)
                {
                    for (int k = kk; k < std::min(kk + BS, N); ++k)
                    {
                        float temp = A[i * N + k];

                        for (int j = jj; j < std::min(jj + BS, N); ++j)
                        {
                            C[i * N + j] += temp * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma")

#include <immintrin.h>
#include <algorithm>
#include <cstring>

void cache_blocked_matmul_avx2(const std::vector<float> &A,
                               const std::vector<float> &B,
                               std::vector<float> &C,
                               int N, int BS = 1024)
{
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
            {
                int i_end = std::min(ii + BS, N);
                int k_end = std::min(kk + BS, N);
                int j_end = std::min(jj + BS, N);

                for (int i = ii; i < i_end; ++i)
                    for (int k = kk; k < k_end; ++k)
                    {
                        __m256 temp = _mm256_set1_ps(A[i * N + k]);

                        int j = jj;
                        for (; j <= j_end - 8; j += 8)
                        {
                            __m256 c = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                            _mm256_storeu_ps(&C[i * N + j],
                                             _mm256_fmadd_ps(temp, b, c));
                        }
                        for (; j < j_end; ++j)
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
            }
}

void cache_blocked_matmul_avx2_no_looptrans(const std::vector<float> &A,
                                            const std::vector<float> &B,
                                            std::vector<float> &C,
                                            int N, int BS = 1024)
{
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
            {
                int i_end = std::min(ii + BS, N);
                int k_end = std::min(kk + BS, N);
                int j_end = std::min(jj + BS, N);

                for (int i = ii; i < i_end; ++i)
                {
                    int j = jj;

                    // ── process 4 rows of k × 8 floats of j simultaneously ──
                    // accumulate in registers, write C only ONCE at the end
                    for (; j <= j_end - 8; j += 8)
                    {
                        // Load C once into register
                        __m256 c0 = _mm256_loadu_ps(&C[i * N + j]);

                        // Accumulate over k entirely in registers
                        for (int k = kk; k < k_end; ++k)
                        {
                            __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 temp = _mm256_set1_ps(A[i * N + k]);
                            c0 = _mm256_fmadd_ps(temp, b, c0);
                        }

                        // Write C only once (was: write every k iteration)
                        _mm256_storeu_ps(&C[i * N + j], c0);
                    }

                    // scalar tail
                    for (; j < j_end; ++j)
                    {
                        float acc = C[i * N + j];
                        for (int k = kk; k < k_end; ++k)
                            acc += A[i * N + k] * B[k * N + j];
                        C[i * N + j] = acc;
                    }
                }
            }
}

void cache_blocked_matmul_avx2_unroll(const std::vector<float> &A,
                                      const std::vector<float> &B,
                                      std::vector<float> &C,
                                      int N, int BS = 1024)
{
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
            {
                int i_end = std::min(ii + BS, N);
                int k_end = std::min(kk + BS, N);
                int j_end = std::min(jj + BS, N);

                for (int i = ii; i < i_end; ++i)
                    for (int k = kk; k < k_end; ++k)
                    {
                        const float *Aik = &A[i * N + k];  // scalar, hoisted
                        const float *Bkj = &B[k * N + jj]; // row pointer
                        float *Cij = &C[i * N + jj];       // row pointer

                        __m256 temp = _mm256_set1_ps(*Aik);

                        int j = 0;
                        int width = j_end - jj;

                        // 4x unrolled: 32 floats per iteration
                        for (; j <= width - 32; j += 32)
                        {
                            __m256 c0 = _mm256_loadu_ps(Cij + j);
                            __m256 c1 = _mm256_loadu_ps(Cij + j + 8);
                            __m256 c2 = _mm256_loadu_ps(Cij + j + 16);
                            __m256 c3 = _mm256_loadu_ps(Cij + j + 24);

                            __m256 b0 = _mm256_loadu_ps(Bkj + j);
                            __m256 b1 = _mm256_loadu_ps(Bkj + j + 8);
                            __m256 b2 = _mm256_loadu_ps(Bkj + j + 16);
                            __m256 b3 = _mm256_loadu_ps(Bkj + j + 24);

                            c0 = _mm256_fmadd_ps(temp, b0, c0);
                            c1 = _mm256_fmadd_ps(temp, b1, c1);
                            c2 = _mm256_fmadd_ps(temp, b2, c2);
                            c3 = _mm256_fmadd_ps(temp, b3, c3);

                            _mm256_storeu_ps(Cij + j, c0);
                            _mm256_storeu_ps(Cij + j + 8, c1);
                            _mm256_storeu_ps(Cij + j + 16, c2);
                            _mm256_storeu_ps(Cij + j + 24, c3);
                        }

                        // 1x: remaining 8-float chunks
                        for (; j <= width - 8; j += 8)
                        {
                            __m256 c = _mm256_loadu_ps(Cij + j);
                            __m256 b = _mm256_loadu_ps(Bkj + j);
                            _mm256_storeu_ps(Cij + j, _mm256_fmadd_ps(temp, b, c));
                        }

                        // scalar tail
                        for (; j < width; ++j)
                            Cij[j] += (*Aik) * Bkj[j];
                    }
            }
}
void cache_blocked_matmul_avx2_unroll_prefench(const std::vector<float> &A,
                                               const std::vector<float> &B,
                                               std::vector<float> &C,
                                               int N, int BS = 256) // ← smaller BS
{

    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
            {
                int i_end = std::min(ii + BS, N);
                int k_end = std::min(kk + BS, N);
                int j_end = std::min(jj + BS, N);

                for (int i = ii; i < i_end; ++i)
                    for (int k = kk; k < k_end; ++k)
                    {
                        const float *Aik = &A[i * N + k];
                        const float *Bkj = &B[k * N + jj];
                        float *Cij = &C[i * N + jj];

                        __m256 temp = _mm256_set1_ps(*Aik);

                        int width = j_end - jj;
                        int j = 0;

                        // Prefetch next k row of B into L1
                        if (k + 1 < k_end)
                            __builtin_prefetch(&B[(k + 1) * N + jj], 0, 1);

                        // 2x unroll only (was 4x — too aggressive for L3)
                        for (; j <= width - 16; j += 16)
                        {
                            __m256 c0 = _mm256_loadu_ps(Cij + j);
                            __m256 c1 = _mm256_loadu_ps(Cij + j + 8);
                            __m256 b0 = _mm256_loadu_ps(Bkj + j);
                            __m256 b1 = _mm256_loadu_ps(Bkj + j + 8);
                            c0 = _mm256_fmadd_ps(temp, b0, c0);
                            c1 = _mm256_fmadd_ps(temp, b1, c1);
                            _mm256_storeu_ps(Cij + j, c0);
                            _mm256_storeu_ps(Cij + j + 8, c1);
                        }

                        // 1x
                        for (; j <= width - 8; j += 8)
                        {
                            __m256 c = _mm256_loadu_ps(Cij + j);
                            __m256 b = _mm256_loadu_ps(Bkj + j);
                            _mm256_storeu_ps(Cij + j, _mm256_fmadd_ps(temp, b, c));
                        }

                        // scalar tail
                        for (; j < width; ++j)
                            Cij[j] += (*Aik) * Bkj[j];
                    }
            }
}

void cache_blocked_matmul_avx2_unroll_prefench_rowCaccum(const std::vector<float> &A,
                                                         const std::vector<float> &B,
                                                         std::vector<float> &C,
                                                         int N, int BS = 1024)
{

    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
            {
                int i_end = std::min(ii + BS, N);
                int k_end = std::min(kk + BS, N);
                int j_end = std::min(jj + BS, N);
                int width = j_end - jj;

                for (int i = ii; i < i_end; ++i)
                    for (int k = kk; k < k_end; ++k)
                    {
                        // ── identical to your original, just vectorized j ──
                        const float *Bkj = &B[k * N + jj];
                        float *Cij = &C[i * N + jj];
                        __m256 temp = _mm256_set1_ps(A[i * N + k]);

                        // prefetch next k's B row while processing this one
                        __builtin_prefetch(&B[(k + 1) * N + jj], 0, 1);

                        int j = 0;
                        for (; j <= width - 8; j += 8)
                        {
                            __m256 c = _mm256_loadu_ps(Cij + j);
                            __m256 b = _mm256_loadu_ps(Bkj + j);
                            _mm256_storeu_ps(Cij + j, _mm256_fmadd_ps(temp, b, c));
                        }
                        for (; j < width; ++j)
                            Cij[j] += A[i * N + k] * Bkj[j];
                    }
            }
}

void cache_blocked_matmul_avx2_packB(const std::vector<float> &A,
                                     const std::vector<float> &B,
                                     std::vector<float> &C,
                                     int N, int BS = 1024)
{
    // Pack buffer for one B block: BS×BS floats, reused across ii
    // Keeps B block access stride-1, fits in L2 (1024×1024×4 = 4MB ~ L3)
    std::vector<float> Bpack(BS * BS);

    for (int kk = 0; kk < N; kk += BS)
        for (int jj = 0; jj < N; jj += BS)
        {
            int k_end = std::min(kk + BS, N);
            int j_end = std::min(jj + BS, N);
            int kb = k_end - kk;
            int jb = j_end - jj;

            // ── Pack B[kk:k_end, jj:j_end] into contiguous Bpack ──
            // Bpack[k_local * jb + j_local] → stride-1 in j
            for (int k = 0; k < kb; ++k)
                std::memcpy(&Bpack[k * jb],
                            &B[(kk + k) * N + jj],
                            jb * sizeof(float));

            // ── Now sweep all ii blocks reusing packed B ──
            for (int ii = 0; ii < N; ii += BS)
            {
                int i_end = std::min(ii + BS, N);

                for (int i = ii; i < i_end; ++i)
                    for (int k = 0; k < kb; ++k)
                    {
                        const float *Bkj = &Bpack[k * jb]; // packed, stride-1 ✅
                        float *Cij = &C[i * N + jj];
                        __m256 temp = _mm256_set1_ps(A[i * N + kk + k]);

                        __builtin_prefetch(Bkj + 64, 0, 1); // prefetch ahead in Bpack

                        int j = 0;
                        for (; j <= jb - 8; j += 8)
                        {
                            __m256 c = _mm256_loadu_ps(Cij + j);
                            __m256 b = _mm256_loadu_ps(Bkj + j);
                            _mm256_storeu_ps(Cij + j, _mm256_fmadd_ps(temp, b, c));
                        }
                        for (; j < jb; ++j)
                            Cij[j] += A[i * N + kk + k] * Bkj[j];
                    }
            }
        }
}

void cache_blocked_matmul_avx2_korder_packAB(const std::vector<float> &A,
                                             const std::vector<float> &B,
                                             std::vector<float> &C,
                                             int N, int BS = 1024)
{
    std::vector<float> Bpack(BS * BS);

    for (int kk = 0; kk < N; kk += BS)
        for (int jj = 0; jj < N; jj += BS)
        {
            int k_end = std::min(kk + BS, N);
            int j_end = std::min(jj + BS, N);
            int kb = k_end - kk;
            int jb = j_end - jj;

            // ── packB (known to help) ──
            for (int k = 0; k < kb; ++k)
                std::memcpy(&Bpack[k * jb],
                            &B[(kk + k) * N + jj],
                            jb * sizeof(float));

            for (int ii = 0; ii < N; ii += BS)
            {
                int i_end = std::min(ii + BS, N);

                for (int i = ii; i < i_end; ++i)
                    for (int k = 0; k < kb; ++k)
                    {
                        const float *Bkj = &Bpack[k * jb];
                        float *Cij = &C[i * N + jj];
                        __m256 temp = _mm256_set1_ps(A[i * N + kk + k]);

                        // ── prefetch next k row of Bpack (known to help) ──
                        __builtin_prefetch(Bkj + jb, 0, 1);

                        int j = 0;

                        // ── 4 independent accumulators (known to help) ──
                        for (; j <= jb - 32; j += 32)
                        {
                            __m256 c0 = _mm256_loadu_ps(Cij + j);
                            __m256 c1 = _mm256_loadu_ps(Cij + j + 8);
                            __m256 c2 = _mm256_loadu_ps(Cij + j + 16);
                            __m256 c3 = _mm256_loadu_ps(Cij + j + 24);

                            c0 = _mm256_fmadd_ps(temp, _mm256_loadu_ps(Bkj + j), c0);
                            c1 = _mm256_fmadd_ps(temp, _mm256_loadu_ps(Bkj + j + 8), c1);
                            c2 = _mm256_fmadd_ps(temp, _mm256_loadu_ps(Bkj + j + 16), c2);
                            c3 = _mm256_fmadd_ps(temp, _mm256_loadu_ps(Bkj + j + 24), c3);

                            _mm256_storeu_ps(Cij + j, c0);
                            _mm256_storeu_ps(Cij + j + 8, c1);
                            _mm256_storeu_ps(Cij + j + 16, c2);
                            _mm256_storeu_ps(Cij + j + 24, c3);
                        }

                        for (; j <= jb - 8; j += 8)
                        {
                            __m256 c = _mm256_loadu_ps(Cij + j);
                            c = _mm256_fmadd_ps(temp, _mm256_loadu_ps(Bkj + j), c);
                            _mm256_storeu_ps(Cij + j, c);
                        }

                        for (; j < jb; ++j)
                            Cij[j] += A[i * N + kk + k] * Bkj[j];
                    }
            }
        }
}

int main()
{
    // Add inside main() before the benchmark
#ifdef __AVX2__
    std::cout << "AVX2 : YES\n";
#else
    std::cout << "AVX2 : NO  ← this is your problem\n";
#endif
#ifdef __FMA__
    std::cout << "FMA  : YES\n";
#else
    std::cout << "FMA  : NO\n";
#endif
    int N = 2048 * 2; // Lowering N for a quick test; 4096 will take a while
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    std::cout << "Starting Naive Matmul for N=" << N << "..." << std::endl;
    // for (auto BS : {16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096})
    // for (int BS : {128, 256, 512, 1024})
    for (auto BS : {1024})
    {
        std::cout << "Block Size: " << BS << std::endl;
        // cache_blocked_matmul_avx2(A, B, C, N, BS);
        auto start = std::chrono::high_resolution_clock::now();

        // naive_matmul(A, B, C, N);
        // cache_friendly_matmul(A, B, C, N);
        // cache_blocked_matmul(A, B, C, N, BS);
        // cache_blocked_matmul_avx2(A, B, C, N, BS);
        // cache_blocked_matmul_avx2_no_looptrans(A, B, C, N, BS);
        // cache_blocked_matmul_avx2_unroll(A, B, C, N, BS);
        // cache_blocked_matmul_avx2_unroll_prefench(A, B, C, N, BS);
        // cache_blocked_matmul_avx2_unroll_prefench_rowCaccum(A, B, C, N, BS);
        // cache_blocked_matmul_avx2_packB(A, B, C, N, BS);
        cache_blocked_matmul_avx2_korder_packAB(A, B, C, N, BS);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double seconds = diff.count();

        // Calculate GFLOPS (Giga-Floating Point Operations Per Second)
        // For NxN matmul, operations = 2 * N^3 (multiplications + additions)
        double gflops = (2.0 * N * N * N) / (seconds * 1e9);

        std::cout << "naive: " << std::endl;
        std::cout << "Time: " << seconds << " s" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    }

    // start = std::chrono::high_resolution_clock::now();

    // cache_friendly_matmul(A, B, C, N);

    // end = std::chrono::high_resolution_clock::now();
    // diff = end - start;
    // seconds = diff.count();

    // // Calculate GFLOPS (Giga-Floating Point Operations Per Second)
    // // For NxN matmul, operations = 2 * N^3 (multiplications + additions)
    // gflops = (2.0 * N * N * N) / (seconds * 1e9);
    // std::cout << "Loop Transposition: "<< std::endl;
    // std::cout << "Time: " << seconds << " s" << std::endl;
    // std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    return 0;
}
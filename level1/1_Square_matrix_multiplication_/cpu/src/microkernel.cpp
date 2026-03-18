// matmul_microkernel.c
// True BLIS-style microkernel for E5-2620 v4 (Broadwell AVX2)
// Compile: gcc -O3 -march=native -mfma -mavx2 -std=c11 -o matmul matmul_microkernel.c

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma")

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// ============================================================
// TUNING PARAMETERS for E5-2620 v4 (Broadwell)
//
//  MR x NR  = register tile = fits in AVX2 registers
//  MR=8     → 8 rows,  each row is 1 float = 1 lane of ymm
//  NR=16    → 16 cols, 2 AVX2 registers wide
//  MR*NR    = 8*16 = 128 floats = 16 ymm regs (exactly fills file)
//
//  MC x KC  = A panel fits in L2 (256KB)
//  NC x KC  = B panel fits in L3 (20MB per socket)
// ============================================================
#define MR 8  // micro-kernel rows
#define NR 16 // micro-kernel cols (2 x AVX2)

#define MC 256  // L2 panel rows    (MC*KC*4 < 256KB)
#define KC 256  // L2 panel depth
#define NC 4096 // L3 panel cols    (KC*NC*4 < 20MB)

// ============================================================
// PACK ROUTINES
// Pack into column-major MR strips (A) and row-major NR strips (B)
// This gives stride-1 access inside the microkernel
// ============================================================

// Pack A[MC x KC] into Ap[KC/MR panels of MR x KC]
// Each panel: MR rows packed contiguously, column by column
static void pack_A(int M, int K,
                   const float *A, int lda,
                   float *Ap)
{
    int i, k, ii;
    for (i = 0; i < M; i += MR)
    {
        int ib = (i + MR <= M) ? MR : M - i;
        for (k = 0; k < K; k++)
        {
            for (ii = 0; ii < ib; ii++)
                Ap[ii] = A[(i + ii) * lda + k];
            for (; ii < MR; ii++) // zero-pad if needed
                Ap[ii] = 0.0f;
            Ap += MR;
        }
    }
}

// Pack B[KC x NC] into Bp[NC/NR panels of KC x NR]
// Each panel: NR cols packed contiguously, row by row
static void pack_B(int K, int N,
                   const float *B, int ldb,
                   float *Bp)
{
    int j, k, jj;
    for (j = 0; j < N; j += NR)
    {
        int jb = (j + NR <= N) ? NR : N - j;
        for (k = 0; k < K; k++)
        {
            for (jj = 0; jj < jb; jj++)
                Bp[jj] = B[k * ldb + (j + jj)];
            for (; jj < NR; jj++) // zero-pad if needed
                Bp[jj] = 0.0f;
            Bp += NR;
        }
    }
}

// ============================================================
// MICROKERNEL: 8 x 16
//
// Computes C[MR x NR] += A_packed[MR x K] * B_packed[K x NR]
//
// Register layout (16 ymm regs for C tile):
//   c_row0:  ymm0  ymm1   (cols 0-7,  cols 8-15)
//   c_row1:  ymm2  ymm3
//   ...
//   c_row7:  ymm14 ymm15
//
// A is stored as K panels of MR floats (column-major in panel)
// B is stored as K panels of NR floats (row-major in panel)
// ============================================================
static void microkernel_8x16(int K,
                             const float *Ap, // MR x K packed
                             const float *Bp, // K x NR packed
                             float *C,
                             int ldc)
{
    // Load C tile into registers (8 rows x 16 cols = 16 ymm regs)
    __m256 c00 = _mm256_loadu_ps(C + 0 * ldc + 0);
    __m256 c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * ldc + 0);
    __m256 c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc + 0);
    __m256 c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc + 0);
    __m256 c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4 * ldc + 0);
    __m256 c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5 * ldc + 0);
    __m256 c51 = _mm256_loadu_ps(C + 5 * ldc + 8);
    __m256 c60 = _mm256_loadu_ps(C + 6 * ldc + 0);
    __m256 c61 = _mm256_loadu_ps(C + 6 * ldc + 8);
    __m256 c70 = _mm256_loadu_ps(C + 7 * ldc + 0);
    __m256 c71 = _mm256_loadu_ps(C + 7 * ldc + 8);

    // Main K loop — both Ap and Bp accessed stride-1
    for (int k = 0; k < K; k++)
    {
        // Load 1 column of B panel: NR=16 floats = 2 ymm regs
        __m256 b0 = _mm256_loadu_ps(Bp);     // cols 0-7
        __m256 b1 = _mm256_loadu_ps(Bp + 8); // cols 8-15
        Bp += NR;

        // Broadcast each of the MR=8 A values, FMA against both B regs
        // broadcast_ss: 1 scalar → all 8 lanes of ymm (zero extra cost)
        __m256 a;

        a = _mm256_broadcast_ss(Ap + 0);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);

        a = _mm256_broadcast_ss(Ap + 1);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);

        a = _mm256_broadcast_ss(Ap + 2);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);

        a = _mm256_broadcast_ss(Ap + 3);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);

        a = _mm256_broadcast_ss(Ap + 4);
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);

        a = _mm256_broadcast_ss(Ap + 5);
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);

        a = _mm256_broadcast_ss(Ap + 6);
        c60 = _mm256_fmadd_ps(a, b0, c60);
        c61 = _mm256_fmadd_ps(a, b1, c61);

        a = _mm256_broadcast_ss(Ap + 7);
        c70 = _mm256_fmadd_ps(a, b0, c70);
        c71 = _mm256_fmadd_ps(a, b1, c71);

        Ap += MR;
    }

    // Store C tile back
    _mm256_storeu_ps(C + 0 * ldc + 0, c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc + 0, c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc + 0, c50);
    _mm256_storeu_ps(C + 5 * ldc + 8, c51);
    _mm256_storeu_ps(C + 6 * ldc + 0, c60);
    _mm256_storeu_ps(C + 6 * ldc + 8, c61);
    _mm256_storeu_ps(C + 7 * ldc + 0, c70);
    _mm256_storeu_ps(C + 7 * ldc + 8, c71);
}

// ============================================================
// 5-LOOP BLIS GEMM
// Loop order (outer to inner):
//   Loop 5: jc  — NC strips of B (fits B panel in L3)
//   Loop 4: kc  — KC strips      (fits A panel in L2)
//   Loop 3: ic  — MC strips      (fits A block in L2)
//   Loop 2: jr  — NR micro-cols  (register tile)
//   Loop 1: ir  — MR micro-rows  (register tile)
//   Kernel: microkernel_8x16
// ============================================================
void sgemm_blis(int M, int N, int K,
                const float *A, int lda,
                const float *B, int ldb,
                float *C, int ldc)
{
    // Allocate packing buffers
    float *Ap = (float *)aligned_alloc(32, MC * KC * sizeof(float));
    float *Bp = (float *)aligned_alloc(32, KC * NC * sizeof(float));

    for (int jc = 0; jc < N; jc += NC)
    {
        int nb = (jc + NC <= N) ? NC : N - jc;

        for (int kc = 0; kc < K; kc += KC)
        {
            int kb = (kc + KC <= K) ? KC : K - kc;

            // Pack B panel [kb x nb] → Bp  (stays in L3)
            pack_B(kb, nb, B + kc * ldb + jc, ldb, Bp);

            for (int ic = 0; ic < M; ic += MC)
            {
                int mb = (ic + MC <= M) ? MC : M - ic;

                // Pack A panel [mb x kb] → Ap  (stays in L2)
                pack_A(mb, kb, A + ic * lda + kc, lda, Ap);

                // Micro-kernel loop
                float *Ap_ptr = Ap;
                for (int ir = 0; ir < mb; ir += MR)
                {
                    float *Bp_ptr = Bp;
                    for (int jr = 0; jr < nb; jr += NR)
                    {
                        microkernel_8x16(
                            kb,
                            Ap_ptr,
                            Bp_ptr,
                            C + (ic + ir) * ldc + (jc + jr),
                            ldc);
                        Bp_ptr += KC * NR;
                    }
                    Ap_ptr += KC * MR;
                }
            }
        }
    }

    free(Ap);
    free(Bp);
}

// ============================================================
// BENCHMARK
// ============================================================
int main(void)
{
    const int N = 4096;

    float *A = (float *)aligned_alloc(32, N * N * sizeof(float));
    float *B = (float *)aligned_alloc(32, N * N * sizeof(float));
    float *C = (float *)aligned_alloc(32, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)(i % 7) * 0.1f;
        B[i] = (float)(i % 5) * 0.1f;
        C[i] = 0.0f;
    }

    // Warmup
    sgemm_blis(N, N, N, A, N, B, N, C, N);
    memset(C, 0, N * N * sizeof(float));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    sgemm_blis(N, N, N, A, N, B, N, C, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double gflops = 2.0 * N * N * N / sec / 1e9;
    printf("Time   : %.4f s\n", sec);
    printf("GFLOPS : %.2f\n", gflops);

    free(A);
    free(B);
    free(C);
    return 0;
}
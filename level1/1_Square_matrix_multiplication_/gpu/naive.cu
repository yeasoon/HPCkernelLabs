#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdint.h> // or #include <cstdint>

#define N 4096 // Matrix size N x N

// Naive CUDA kernel: one thread per output element
__global__ void matmul_naive(float *A, float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

#define TILE 32 // 32×32 tile → 32² × 4B × 2 = 8KB smem (fits easily)

__global__ void matmul_tiled(float *A, float *B, float *C, int n)
{
    // Shared memory tiles — each block loads ONE tile of A and B
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Sweep tiles across the K dimension
    for (int t = 0; t < (n + TILE - 1) / TILE; t++)
    {
        // ── Collaborative load: each thread loads ONE element into smem ──
        // A tile: row stays fixed, column advances with tile index t
        if (row < n && (t * TILE + threadIdx.x) < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // B tile: column stays fixed, row advances with tile index t
        if (col < n && (t * TILE + threadIdx.y) < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // wait for ALL threads to finish loading

// ── Compute: each thread does TILE multiply-adds from smem ──
#pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads(); // wait before loading next tile (prevent overwrite)
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

__global__ void matmul_tiled_v2(float *A, float *B, float *C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1]; // padding: eliminates bank conflicts

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++)
    {
        As[threadIdx.y][threadIdx.x] =
            (row < n && t * TILE + threadIdx.x < n)
                ? A[row * n + t * TILE + threadIdx.x]
                : 0.f;

        Bs[threadIdx.y][threadIdx.x] =
            (col < n && t * TILE + threadIdx.y < n)
                ? B[(t * TILE + threadIdx.y) * n + col]
                : 0.f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

#define TW 16    // thread block width  (TILE/2)
#define TH 16    // thread block height (TILE/2)
#define COARSE 2 // each thread computes COARSE×COARSE outputs

__global__ void matmul_coarse(float *A, float *B, float *C, int n)
{
    // Smem: TILE×TILE tiles, loaded by 16×16=256 threads (4 elements each)
    __shared__ float As[TILE][TILE + 1]; // +1 eliminates ALL bank conflicts
    __shared__ float Bs[TILE][TILE + 1];

    // Each thread is responsible for a 2×2 patch in the output tile
    int ty = threadIdx.y; // 0..15
    int tx = threadIdx.x; // 0..15

    // Base output coordinates for this thread's 2×2 patch
    int row0 = blockIdx.y * TILE + ty * COARSE;
    int col0 = blockIdx.x * TILE + tx * COARSE;

    // 2×2 accumulator in registers
    float acc[COARSE][COARSE] = {{0.f, 0.f}, {0.f, 0.f}};

    for (int t = 0; t < n; t += TILE)
    {
// ── Load As: each thread loads COARSE×COARSE elements ──
// Thread (ty,tx) loads rows [ty*2, ty*2+1], cols [tx*2, tx*2+1]
#pragma unroll
        for (int i = 0; i < COARSE; i++)
#pragma unroll
            for (int j = 0; j < COARSE; j++)
            {
                int r = ty * COARSE + i;
                int c = tx * COARSE + j;
                As[r][c] = (blockIdx.y * TILE + r < n && t + c < n)
                               ? A[(blockIdx.y * TILE + r) * n + t + c]
                               : 0.f;
                Bs[r][c] = (t + r < n && blockIdx.x * TILE + c < n)
                               ? B[(t + r) * n + blockIdx.x * TILE + c]
                               : 0.f;
            }

        __syncthreads();

// ── Compute: 2×2 outputs × TILE k-steps ──
#pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            // Load 2 A values into registers once
            float a0 = As[ty * COARSE + 0][k];
            float a1 = As[ty * COARSE + 1][k];

            // Load 2 B values into registers once
            float b0 = Bs[k][tx * COARSE + 0];
            float b1 = Bs[k][tx * COARSE + 1];

            // 4 FMAs — all from registers, zero smem re-reads
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }

        __syncthreads();
    }

// ── Write 2×2 outputs ──
#pragma unroll
    for (int i = 0; i < COARSE; i++)
#pragma unroll
        for (int j = 0; j < COARSE; j++)
            if (row0 + i < n && col0 + j < n)
                C[(row0 + i) * n + col0 + j] = acc[i][j];
}

#define NTHREADS 256 // 16×16
// Each thread loads 2 contiguous rows of As and 2 contiguous cols of Bs

__global__ void matmul_coarse_v2(float *A, float *B, float *C, int n)
{
    __shared__ float As[TILE][TILE]; // no padding needed with this pattern
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx; // 0..255

    // ── Flat load: thread tid loads elements tid, tid+256, tid+512, tid+768 ──
    // These map to row-major positions in TILE×TILE = 1024 element tile
    // Consecutive tids → consecutive columns → stride-1 → zero bank conflicts

    int baseA = blockIdx.y * TILE * n; // start of A tile rows
    int baseB = blockIdx.x * TILE;     // start of B tile cols

    // 2×2 accumulator — compute index uses ty,tx independently of load
    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

    for (int t = 0; t < n; t += TILE)
    {
// Load 4 elements each using flat tid
// Element e → As[e/TILE][e%TILE]
// e = tid + k*NTHREADS for k=0..3
#pragma unroll
        for (int e = tid; e < TILE * TILE; e += NTHREADS)
        {
            int r = e / TILE; // 0..31
            int c = e % TILE; // 0..31  ← consecutive across warp ✅
            As[r][c] = A[(blockIdx.y * TILE + r) * n + t + c];
            Bs[r][c] = B[(t + r) * n + blockIdx.x * TILE + c];
        }

        __syncthreads();

// Compute: thread (ty,tx) → output patch (ty*2, tx*2) to (ty*2+1, tx*2+1)
#pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    int row0 = blockIdx.y * TILE + ty * 2;
    int col0 = blockIdx.x * TILE + tx * 2;
    C[(row0)*n + col0] = acc00;
    C[(row0)*n + col0 + 1] = acc01;
    C[(row0 + 1) * n + col0] = acc10;
    C[(row0 + 1) * n + col0 + 1] = acc11;
}
#define BX 16
#define BY 16
__global__ __launch_bounds__(256, 6) void matmul_v3(float *__restrict__ A,
                                                    float *__restrict__ B,
                                                    float *__restrict__ C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y;   // 0..15
    int tx = threadIdx.x;   // 0..15
    int tid = ty * BX + tx; // 0..255

    // Flat load coordinates — precomputed, NO division/modulo
    // 256 threads × 4 = 1024 = 32×32
    // Map: thread tid → 4 elements at rows {tid>>2} ... cleaner:
    // Split tid into (row_group, col_group):
    //   load_row = tid / 8   → 0..31  (32 rows, 8 threads share each row... no)
    //
    // Cleanest: 256 threads load 32 rows × 32 cols
    // 8 threads per row (32 cols / 4 floats per thread)
    // load_row = tid / 8,  load_col_base = (tid % 8) * 4

    int ld_row = tid >> 3;       // tid / 8  → 0..31
    int ld_col = (tid & 7) << 2; // (tid % 8)*4 → 0,4,8,...,28

    // These 4 consecutive columns → bank = col%32 → 4 consecutive banks ✅
    // Warp (32 threads): ld_col = {0,0,0,0, 4,4,4,4, 8,...,28,28,28,28}
    //                    × 4 consecutive each → full 32-bank coverage ✅

    int rowA = blockIdx.y * TILE + ld_row;
    int rowB_t_offset = ld_row; // row within B tile
    int colB = blockIdx.x * TILE + ld_col;

    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

    for (int t = 0; t < n; t += TILE)
    {
        // ── Load As: 4 consecutive floats per thread, coalesced ──
        // Use float4 for single 128-bit instruction
        float4 a4 = *reinterpret_cast<const float4 *>(&A[rowA * n + t + ld_col]);
        As[ld_row][ld_col] = a4.x;
        As[ld_row][ld_col + 1] = a4.y;
        As[ld_row][ld_col + 2] = a4.z;
        As[ld_row][ld_col + 3] = a4.w;

        float4 b4 = *reinterpret_cast<const float4 *>(&B[(t + ld_row) * n + colB]);
        Bs[ld_row][ld_col] = b4.x;
        Bs[ld_row][ld_col + 1] = b4.y;
        Bs[ld_row][ld_col + 2] = b4.z;
        Bs[ld_row][ld_col + 3] = b4.w;

        __syncthreads();

// ── Compute: thread (ty,tx) → 2×2 output at (ty*2, tx*2) ──
#pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    int row0 = blockIdx.y * TILE + ty * 2;
    int col0 = blockIdx.x * TILE + tx * 2;
    C[(row0)*n + col0] = acc00;
    C[(row0)*n + col0 + 1] = acc01;
    C[(row0 + 1) * n + col0] = acc10;
    C[(row0 + 1) * n + col0 + 1] = acc11;
}

__global__ __launch_bounds__(256, 6) void matmul_v4(float *__restrict__ A,
                                                    float *__restrict__ B,
                                                    float *__restrict__ C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty * BX + tx;

    const int ld_row = tid >> 3;
    const int ld_col = (tid & 7) << 2;
    const int row_a0 = ty * 2;
    const int row_a1 = ty * 2 + 1;
    const int col_b0 = tx * 2;
    const int col_b1 = tx * 2 + 1;

    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

    // Precompute base pointers outside loop
    const float *__restrict__ pA_base =
        A + (blockIdx.y * TILE + ld_row) * n + ld_col;
    const float *__restrict__ pB_base =
        B + ld_row * n + blockIdx.x * TILE + ld_col;

    for (int t = 0; t < n; t += TILE)
    {
        // ── Explicit register staging before smem write ──
        // Load into registers first, THEN write to smem
        // This prevents compiler from fusing load+compute paths
        float ra0, ra1, ra2, ra3;
        float rb0, rb1, rb2, rb3;

        // LDG.128 — single 128-bit read-only cache instruction
        asm volatile(
            "ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(ra0), "=f"(ra1), "=f"(ra2), "=f"(ra3)
            : "l"(pA_base + t));
        asm volatile(
            "ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(rb0), "=f"(rb1), "=f"(rb2), "=f"(rb3)
            : "l"(pB_base + t * n));

        // Write to smem — compiler MUST emit STS here
        As[ld_row][ld_col] = ra0;
        As[ld_row][ld_col + 1] = ra1;
        As[ld_row][ld_col + 2] = ra2;
        As[ld_row][ld_col + 3] = ra3;

        Bs[ld_row][ld_col] = rb0;
        Bs[ld_row][ld_col + 1] = rb1;
        Bs[ld_row][ld_col + 2] = rb2;
        Bs[ld_row][ld_col + 3] = rb3;

        __syncthreads();

// ── Compute loop: only LDS and FFMA should appear here ──
#pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            float a0 = As[row_a0][k];
            float a1 = As[row_a1][k];
            float b0 = Bs[k][col_b0];
            float b1 = Bs[k][col_b1];
            acc00 = __fmaf_rn(a0, b0, acc00);
            acc01 = __fmaf_rn(a0, b1, acc01);
            acc10 = __fmaf_rn(a1, b0, acc10);
            acc11 = __fmaf_rn(a1, b1, acc11);
        }

        __syncthreads();
    }

    const int row0 = blockIdx.y * TILE + row_a0;
    const int col0 = blockIdx.x * TILE + col_b0;
    C[(row0)*n + col0] = acc00;
    C[(row0)*n + col0 + 1] = acc01;
    C[(row0 + 1) * n + col0] = acc10;
    C[(row0 + 1) * n + col0 + 1] = acc11;
}
#define TILE 64 // output tile per block (64×64)
#define TM 4    // outputs per thread in y (TILE/BY = 64/16 = 4)
#define TN 4    // outputs per thread in x (TILE/BX = 64/16 = 4)

__global__ __launch_bounds__(256, 3) void matmul_v5(float *__restrict__ A,
                                                    float *__restrict__ B,
                                                    float *__restrict__ C, int n)
{
    __shared__ float As[TILE][TILE]; // 64×64 × 4B = 16KB
    __shared__ float Bs[TILE][TILE]; // 16KB  total = 32KB per block

    const int ty = threadIdx.y;   // 0..15
    const int tx = threadIdx.x;   // 0..15
    const int tid = ty * BX + tx; // 0..255

    // ── Flat load indices (no div/mod in loop) ──
    // 256 threads load 64×64=4096 elements → 16 each
    // Each thread loads a 4×4 contiguous patch in row-major
    // thread tid → row = tid/16, col_base = (tid%16)*4
    const int ld_row = tid >> 4;             // tid/16 → 0..15 (only 16 rows?!)
    const int ld_col_base = (tid & 15) << 2; // (tid%16)*4 → 0,4,...,60

    // Wait — 256 threads × 16 elements = 4096 = 64×64 ✅
    // But ld_row = 0..15, we need rows 0..63
    // Each thread loads rows {ld_row, ld_row+16, ld_row+32, ld_row+48}

    // Precompute output base coordinates
    const int row_base = blockIdx.y * TILE + ty * TM; // 4 consecutive output rows
    const int col_base = blockIdx.x * TILE + tx * TN; // 4 consecutive output cols

    // 4×4 accumulator
    float acc[TM][TN] = {0};

    for (int t = 0; t < n; t += TILE)
    {
// ── Load As: thread loads 4 rows × 4 cols = 16 elements ──
// rows {ld_row, ld_row+16, ld_row+32, ld_row+48}, cols {ld_col_base..+3}
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int r = ld_row + i * 16;
            int global_row = blockIdx.y * TILE + r;
            int global_col = t + ld_col_base;
            float4 a4 = *reinterpret_cast<const float4 *>(&A[global_row * n + global_col]);
            As[r][ld_col_base] = a4.x;
            As[r][ld_col_base + 1] = a4.y;
            As[r][ld_col_base + 2] = a4.z;
            As[r][ld_col_base + 3] = a4.w;

            float4 b4 = *reinterpret_cast<const float4 *>(
                &B[(t + r) * n + blockIdx.x * TILE + ld_col_base]);
            Bs[r][ld_col_base] = b4.x;
            Bs[r][ld_col_base + 1] = b4.y;
            Bs[r][ld_col_base + 2] = b4.z;
            Bs[r][ld_col_base + 3] = b4.w;
        }

        __syncthreads();

// ── Compute: thread (ty,tx) → 4×4 output patch ──
// Output rows: ty*4, ty*4+1, ty*4+2, ty*4+3
// Output cols: tx*4, tx*4+1, tx*4+2, tx*4+3
#pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            // Load 4 A values (one column of our output row-patch)
            float a[TM];
#pragma unroll
            for (int m = 0; m < TM; m++)
                a[m] = As[ty * TM + m][k];

            // Load 4 B values (one row of our output col-patch)
            float b[TN];
#pragma unroll
            for (int nn = 0; nn < TN; nn++)
                b[nn] = Bs[k][tx * TN + nn];

// 4×4 = 16 FMAs — all from registers
#pragma unroll
            for (int m = 0; m < TM; m++)
#pragma unroll
                for (int nn = 0; nn < TN; nn++)
                    acc[m][nn] = __fmaf_rn(a[m], b[nn], acc[m][nn]);
        }

        __syncthreads();
    }

// ── Write 4×4 outputs ──
#pragma unroll
    for (int m = 0; m < TM; m++)
#pragma unroll
        for (int nn = 0; nn < TN; nn++)
            C[(row_base + m) * n + col_base + nn] = acc[m][nn];
}

#define TILE 32 // reduce tile to fit 2 buffers in smem
#define BX 16
#define BY 16
#define TM 4
#define TN 4
// 2 buffers × 2 matrices × 32×32×4B = 16KB total ← fits 6 blocks/SM!

__global__ __launch_bounds__(256, 4) void matmul_v6(float *__restrict__ A,
                                                    float *__restrict__ B,
                                                    float *__restrict__ C, int n)
{
    // Double buffer ping-pong
    __shared__ float As[2][TILE][TILE]; // 2×4KB = 8KB
    __shared__ float Bs[2][TILE][TILE]; // 2×4KB = 8KB  total=16KB

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty * BX + tx;

    // TILE=32: 256 threads load 32×32=1024 elements → 4 each
    // ld_row = tid/8 → 0..31, ld_col_base = (tid%8)*4 → 0,4,...,28
    const int ld_row = tid >> 3;            // 0..31
    const int ld_col_base = (tid & 7) << 2; // 0,4,...,28

    const int row_base = blockIdx.y * TILE + ty * TM;
    const int col_base = blockIdx.x * TILE + tx * TN;

    float acc[TM][TN] = {0};

    // ── Prefetch tile 0 into buffer 0 ──
    int buf = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int row = ld_row + i * (TILE / 4); // TILE=32: stride=8...
    }
    // Actually with TILE=32 and 256 threads: 256×4=1024=32×32 ✅
    // ld_row = tid>>3 = 0..31 (one row per group of 8 threads)
    // Each thread loads 4 consecutive cols: ld_col_base..+3
    // NO loop needed — one store per thread covers entire tile!

    auto load_tile = [&](int buf_idx, int t)
    {
        int row = blockIdx.y * TILE + ld_row;
        int col = t + ld_col_base;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(&As[buf_idx][ld_row][ld_col_base])),
                     "l"(&A[row * n + col]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(&Bs[buf_idx][ld_row][ld_col_base])),
                     "l"(&B[(t + ld_row) * n + blockIdx.x * TILE + ld_col_base]));
    };

    // Prefetch first tile
    load_tile(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    for (int t = 0; t < n; t += TILE)
    {
        int cur = t / TILE % 2;
        int nxt = 1 - cur;

        // Issue next tile load (async, no stall)
        if (t + TILE < n)
        {
            load_tile(nxt, t + TILE);
            asm volatile("cp.async.commit_group;\n" ::);
        }

// Compute current tile from cur buffer
#pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            float a[TM], b[TN];
#pragma unroll
            for (int m = 0; m < TM; m++)
                a[m] = As[cur][ty * TM + m][k];
#pragma unroll
            for (int nn = 0; nn < TN; nn++)
                b[nn] = Bs[cur][k][tx * TN + nn];
#pragma unroll
            for (int m = 0; m < TM; m++)
#pragma unroll
                for (int nn = 0; nn < TN; nn++)
                    acc[m][nn] = __fmaf_rn(a[m], b[nn], acc[m][nn]);
        }

        // Wait for next tile before proceeding
        if (t + TILE < n)
        {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }
    }

#pragma unroll
    for (int m = 0; m < TM; m++)
#pragma unroll
        for (int nn = 0; nn < TN; nn++)
            C[(row_base + m) * n + col_base + nn] = acc[m][nn];
}

void fill_random(float *mat, int n)
{
    for (int i = 0; i < n * n; i++)
        mat[i] = (float)rand() / RAND_MAX;
}

int main()
{
    int n = N;
    size_t bytes = n * n * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    srand(42);
    fill_random(h_A, n);
    fill_random(h_B, n);

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Grid / block dims
    // dim3 threads(32, 32);
    // dim3 blocks((n + threads.x - 1) / threads.x,
    //             (n + threads.y - 1) / threads.y);
    dim3 threads(TW, TH); // 16×16 = 256 threads
    // dim3 threads(BX, BY);                                      // 32×8 = 256 threads (full warp in x-dim)
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE); // 128×128 blocks

    // Warm-up
    // matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
    // matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    // matmul_tiled_v2<<<blocks, threads>>>(d_A, d_B, d_C, n);
    // matmul_coarse<<<blocks, threads>>>(d_A, d_B, d_C, n);
    // matmul_coarse_v2<<<blocks, threads>>>(d_A, d_B, d_C, n);
    // matmul_v3<<<blocks, threads>>>(d_A, d_B, d_C, n);
    // matmul_v4<<<blocks, threads>>>(d_A, d_B, d_C, n);
    // matmul_v5<<<blocks, threads>>>(d_A, d_B, d_C, n);
    matmul_v6<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Benchmark with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 10;
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
        // matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
        // matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
        // matmul_tiled_v2<<<blocks, threads>>>(d_A, d_B, d_C, n);
        // matmul_coarse<<<blocks, threads>>>(d_A, d_B, d_C, n);
        // matmul_coarse_v2<<<blocks, threads>>>(d_A, d_B, d_C, n);
        // matmul_v3<<<blocks, threads>>>(d_A, d_B, d_C, n);
        // matmul_v4<<<blocks, threads>>>(d_A, d_B, d_C, n);
        // matmul_v5<<<blocks, threads>>>(d_A, d_B, d_C, n);
        matmul_v6<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / runs;

    // FLOPS: 2*N^3 (multiply-add per element, N^2 elements)
    double flops = 2.0 * (double)n * n * n;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    printf("Matrix size : %d x %d\n", n, n);
    printf("Block size  : %d x %d\n", threads.x, threads.y);
    printf("Avg time    : %.3f ms  (over %d runs)\n", avg_ms, runs);
    printf("Throughput  : %.2f GFLOP/s\n", gflops);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Quick sanity check: print C[0][0]
    printf("C[0][0]     : %f\n", h_C[0]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
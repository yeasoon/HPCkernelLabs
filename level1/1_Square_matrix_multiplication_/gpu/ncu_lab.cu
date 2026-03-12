// cuda_methodology_lab.cu
//
// Step-by-step CUDA performance lab.
// Designed to help learn a profiling-first optimization methodology.
//
// Build:
//   nvcc -O3 -std=c++17 cuda_methodology_lab.cu -o cuda_lab
//
// Example runs:
//   ./cuda_lab info
//   ./cuda_lab coalesce 16777216 256
//   ./cuda_lab divergence 16777216 256
//   ./cuda_lab coarsen 16777216 256 1
//   ./cuda_lab coarsen 16777216 256 2
//   ./cuda_lab coarsen 16777216 256 4
//   ./cuda_lab privatize 16777216 256
//   ./cuda_lab bound pointwise 16777216 256
//   ./cuda_lab bound gemm 1024 1024 1024 16
//   ./cuda_lab tile_sweep 1024 1024 1024
//
// Nsight Compute examples:
//   ncu --set full ./cuda_lab coalesce 16777216 256
//   ncu --set full ./cuda_lab divergence 16777216 256
//   ncu --set full ./cuda_lab coarsen 16777216 256 4
//   ncu --set full ./cuda_lab bound gemm 1024 1024 1024 16
//
// Notes:
// - The "tile_sweep" experiment is a better replacement for the previous
//   incorrect occupancy helper usage.
// - For 2D GEMM threadblocks, sweeping tile size and measuring time is more
//   meaningful than asking CUDA for a generic 1D block recommendation.

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess)                                             \
        {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

static void fill_f32(std::vector<float> &v, float scale = 1.0f)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = scale * static_cast<float>(i % 1024) / 1024.0f;
    }
}

static void fill_i32(std::vector<int> &v)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = static_cast<int>(i);
    }
}

static void print_device_info()
{
    int dev = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n";
    std::cout << "Max threads/block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Shared memory/block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << "Registers/block: " << prop.regsPerBlock << "\n";
    std::cout << "Max threads/SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Memory clock rate: " << prop.memoryClockRate << " kHz\n";
    std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits\n";

    double peak_bw_gbps =
        2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) / 1e9;
    std::cout << "Approx peak DRAM bandwidth: " << peak_bw_gbps << " GB/s\n";
}

template <typename LaunchFn>
static float benchmark_kernel(LaunchFn launch_fn, int warmup = 5, int iters = 20)
{
    for (int i = 0; i < warmup; ++i)
    {
        launch_fn();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        launch_fn();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= static_cast<float>(iters);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

// ============================================================
// STEP 1: Coalescing
// ============================================================
__global__ void copy_coalesced_kernel(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = in[i];
    }
}

// Intentionally poor access pattern: each thread reads with a stride.
__global__ void copy_strided_kernel(const float *in, float *out, int n, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int src = (i * stride) % n;
        out[i] = in[src];
    }
}

// ============================================================
// STEP 2: Divergence
// ============================================================
__global__ void divergence_branchy_kernel(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int x = data[i];
        if (x & 1)
        {
            data[i] = x * 3 + 1;
        }
        else
        {
            data[i] = x / 2;
        }
    }
}

__global__ void divergence_reduced_kernel(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int x = data[i];
        int odd = x & 1;
        int a = x * 3 + 1;
        int b = x / 2;
        data[i] = odd * a + (1 - odd) * b;
    }
}

// ============================================================
// STEP 3: Thread coarsening
// ============================================================
template <int COARSEN>
__global__ void vec_add_coarsened_kernel(const float *a, const float *b, float *c, int n)
{
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;

#pragma unroll
    for (int k = 0; k < COARSEN; ++k)
    {
        int i = base + k;
        if (i < n)
        {
            c[i] = a[i] + b[i];
        }
    }
}

// ============================================================
// STEP 4: Privatization / register reuse
// ============================================================
__global__ void axpy_reload_kernel(const float *a, const float *b, float *out, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Intentionally redundant reloads.
        out[i] = alpha * (a[i] + a[i]) + (b[i] + b[i]);
    }
}

__global__ void axpy_privatized_kernel(const float *a, const float *b, float *out, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = a[i];
        float y = b[i];
        out[i] = alpha * (x + x) + (y + y);
    }
}

// ============================================================
// STEP 5: Memory-bound pointwise
// ============================================================
__global__ void relu_kernel(const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

// ============================================================
// STEP 6: GEMM for compute-vs-memory and tile sweep
// ============================================================
// Naive GEMM
__global__ void gemm_naive_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Shared-memory tiled GEMM for clearer tile experiments.
template <int TILE>
__global__ void gemm_tiled_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t)
    {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = acc;
    }
}

static double estimate_relu_intensity_f32()
{
    // Approximate: one compare/select for ~8 bytes of traffic.
    return 1.0 / 8.0;
}

static double estimate_gemm_intensity(int M, int N, int K)
{
    // Approximate operational intensity:
    // 2MNK / (MK + KN + MN)
    double num = 2.0 * static_cast<double>(M) * N * K;
    double den = static_cast<double>(M) * K +
                 static_cast<double>(K) * N +
                 static_cast<double>(M) * N;
    return num / den;
}

static void print_experiment_header(const std::string &name)
{
    std::cout << "\n============================================================\n";
    std::cout << name << "\n";
    std::cout << "============================================================\n";
}

static void experiment_coalesce(int n, int block_size)
{
    print_experiment_header("STEP 1: Coalescing");

    std::vector<float> h_in(n), h_out(n);
    fill_f32(h_in);

    float *d_in = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (n + block_size - 1) / block_size;
    int stride = 33;

    float t_strided = benchmark_kernel([&]
                                       { copy_strided_kernel<<<grid, block_size>>>(d_in, d_out, n, stride); });

    float t_coalesced = benchmark_kernel([&]
                                         { copy_coalesced_kernel<<<grid, block_size>>>(d_in, d_out, n); });

    double bytes = 2.0 * n * sizeof(float);
    double bw_strided = bytes / (t_strided * 1e6);
    double bw_coalesced = bytes / (t_coalesced * 1e6);

    std::cout << "n = " << n << ", block_size = " << block_size << "\n";
    std::cout << "strided copy      : " << t_strided << " ms, " << bw_strided << " GB/s\n";
    std::cout << "coalesced copy    : " << t_coalesced << " ms, " << bw_coalesced << " GB/s\n";
    std::cout << "speedup           : " << (t_strided / t_coalesced) << "x\n";
    std::cout << "Hypothesis        : coalesced should reduce wasted memory traffic.\n";
    std::cout << "Profile next      : DRAM throughput, L1/TEX behavior, duration.\n";

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}
__global__ void divergence_heavy_branchy_kernel(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = data[i];
        if (i & 1)
        {
#pragma unroll 32
            for (int k = 0; k < 64; ++k)
            {
                x = x * 1.0001f + 0.0001f;
            }
        }
        else
        {
#pragma unroll 32
            for (int k = 0; k < 64; ++k)
            {
                x = x * 0.9999f - 0.0001f;
            }
        }
        data[i] = x;
    }
}
__global__ void divergence_warp_uniform_kernel(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = data[i];
        int warp_id = i / warpSize;

        if (warp_id & 1)
        {
#pragma unroll 32
            for (int k = 0; k < 64; ++k)
            {
                x = x * 1.0001f + 0.0001f;
            }
        }
        else
        {
#pragma unroll 32
            for (int k = 0; k < 64; ++k)
            {
                x = x * 0.9999f - 0.0001f;
            }
        }
        data[i] = x;
    }
}
static void experiment_divergence(int n, int block_size)
{
    // print_experiment_header("STEP 2: Control Divergence");

    // std::vector<int> h(n);
    // fill_i32(h);

    // int *d = nullptr;
    // CHECK_CUDA(cudaMalloc(&d, n * sizeof(int)));

    // int grid = (n + block_size - 1) / block_size;

    // CHECK_CUDA(cudaMemcpy(d, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    // float t_branchy = benchmark_kernel([&]
    //                                    { divergence_branchy_kernel<<<grid, block_size>>>(d, n); });

    // CHECK_CUDA(cudaMemcpy(d, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    // float t_reduced = benchmark_kernel([&]
    //                                    { divergence_reduced_kernel<<<grid, block_size>>>(d, n); });

    // std::cout << "n = " << n << ", block_size = " << block_size << "\n";
    // std::cout << "branchy kernel    : " << t_branchy << " ms\n";
    // std::cout << "branch-reduced    : " << t_reduced << " ms\n";
    // std::cout << "speedup           : " << (t_branchy / t_reduced) << "x\n";
    // std::cout << "Hypothesis        : reduced divergence may improve warp efficiency.\n";
    // std::cout << "Profile next      : branch metrics, warp execution efficiency, duration.\n";

    // CHECK_CUDA(cudaFree(d));
    print_experiment_header("STEP 2: Stronger Control Divergence");

    std::vector<float> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = 1.0f + (i % 13) * 0.01f;

    float *d = nullptr;
    CHECK_CUDA(cudaMalloc(&d, n * sizeof(float)));

    int grid = (n + block_size - 1) / block_size;

    CHECK_CUDA(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    float t_branchy = benchmark_kernel([&]
                                       { divergence_heavy_branchy_kernel<<<grid, block_size>>>(d, n); });

    CHECK_CUDA(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    float t_uniform = benchmark_kernel([&]
                                       { divergence_warp_uniform_kernel<<<grid, block_size>>>(d, n); });

    std::cout << "n = " << n << ", block_size = " << block_size << "\n";
    std::cout << "intra-warp divergent : " << t_branchy << " ms\n";
    std::cout << "warp-uniform         : " << t_uniform << " ms\n";
    std::cout << "speedup              : " << (t_branchy / t_uniform) << "x\n";
    std::cout << "Hypothesis           : warp-uniform control flow should be faster.\n";
    std::cout << "Profile next         : branch metrics, eligible warps, issue stall reasons.\n";

    CHECK_CUDA(cudaFree(d));
}

template <int COARSEN>
static void experiment_coarsen_impl(int n, int block_size)
{
    print_experiment_header("STEP 3: Thread Coarsening");

    std::vector<float> h_a(n), h_b(n), h_c(n);
    fill_f32(h_a, 1.0f);
    fill_f32(h_b, 2.0f);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int logical_threads = (n + COARSEN - 1) / COARSEN;
    int grid = (logical_threads + block_size - 1) / block_size;

    float t = benchmark_kernel([&]
                               { vec_add_coarsened_kernel<COARSEN><<<grid, block_size>>>(d_a, d_b, d_c, n); });

    double bytes = 3.0 * n * sizeof(float);
    double bw = bytes / (t * 1e6);

    std::cout << "n = " << n << ", block_size = " << block_size
              << ", coarsen = " << COARSEN << "\n";
    std::cout << "time              : " << t << " ms\n";
    std::cout << "effective BW      : " << bw << " GB/s\n";
    std::cout << "Hypothesis        : modest coarsening may reduce overhead and improve throughput.\n";
    std::cout << "Profile next      : duration, occupancy, register pressure.\n";

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

static void experiment_privatize(int n, int block_size)
{
    print_experiment_header("STEP 4: Privatization / Register Reuse");

    std::vector<float> h_a(n), h_b(n), h_out(n);
    fill_f32(h_a, 1.0f);
    fill_f32(h_b, 2.0f);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (n + block_size - 1) / block_size;

    float t_reload = benchmark_kernel([&]
                                      { axpy_reload_kernel<<<grid, block_size>>>(d_a, d_b, d_out, 2.0f, n); });

    float t_privatized = benchmark_kernel([&]
                                          { axpy_privatized_kernel<<<grid, block_size>>>(d_a, d_b, d_out, 2.0f, n); });

    std::cout << "n = " << n << ", block_size = " << block_size << "\n";
    std::cout << "reload-heavy      : " << t_reload << " ms\n";
    std::cout << "privatized        : " << t_privatized << " ms\n";
    std::cout << "speedup           : " << (t_reload / t_privatized) << "x\n";
    std::cout << "Hypothesis        : register reuse should reduce redundant loads.\n";
    std::cout << "Profile next      : local/global memory traffic, register count, duration.\n";

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}

static void experiment_bound_pointwise(int n, int block_size)
{
    print_experiment_header("STEP 5: Memory-bound Pointwise Kernel");

    std::vector<float> h_x(n), h_y(n);
    fill_f32(h_x);

    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (n + block_size - 1) / block_size;

    float t = benchmark_kernel([&]
                               { relu_kernel<<<grid, block_size>>>(d_x, d_y, n); });

    double bytes = 2.0 * n * sizeof(float);
    double bw = bytes / (t * 1e6);

    std::cout << "n = " << n << ", block_size = " << block_size << "\n";
    std::cout << "time              : " << t << " ms\n";
    std::cout << "effective BW      : " << bw << " GB/s\n";
    std::cout << "arithmetic intensity (est.) = " << estimate_relu_intensity_f32() << "\n";
    std::cout << "Classification     : expected strongly memory-bound.\n";
    std::cout << "Profile next       : DRAM throughput and issue efficiency.\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
}

static void experiment_bound_gemm(int M, int N, int K, int tile)
{
    print_experiment_header("STEP 6: GEMM Compute-vs-Memory Classification");

    std::vector<float> hA(M * K), hB(K * N), hC(M * N);
    fill_f32(hA);
    fill_f32(hB);

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (M + tile - 1) / tile);

    float t = benchmark_kernel([&]
                               { gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K); }, 2, 10);

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (t * 1e6);

    int blocks_per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gemm_naive_kernel,
        tile * tile,
        0));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K
              << ", tile = " << tile << "\n";
    std::cout << "threads/block     : " << tile * tile << "\n";
    std::cout << "max active blocks/SM: " << blocks_per_sm << "\n";
    std::cout << "time              : " << t << " ms\n";
    std::cout << "GFLOP/s           : " << gflops << "\n";
    std::cout << "arithmetic intensity (est.) = " << estimate_gemm_intensity(M, N, K) << "\n";
    std::cout << "Classification     : larger GEMM trends compute-bound.\n";
    std::cout << "Profile next       : SM throughput, tensor/core utilization if applicable, DRAM pressure.\n";

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}

template <int TILE>
static float run_tiled_gemm_case(const float *dA, const float *dB, float *dC,
                                 int M, int N, int K)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    return benchmark_kernel([&]
                            { gemm_tiled_kernel<TILE><<<grid, block>>>(dA, dB, dC, M, N, K); }, 2, 10);
}

static void print_tiled_case(int tile, float t, int M, int N, int K, int blocks_per_sm)
{
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (t * 1e6);

    std::cout << "tile = " << std::setw(2) << tile
              << " | time = " << std::setw(10) << t << " ms"
              << " | GFLOP/s = " << std::setw(10) << gflops
              << " | max active blocks/SM = " << blocks_per_sm
              << "\n";
}

static void experiment_tile_sweep(int M, int N, int K)
{
    print_experiment_header("STEP 7: Tile Sweep for Quantization / Occupancy Tradeoffs");

    std::vector<float> hA(M * K), hB(K * N), hC(M * N);
    fill_f32(hA);
    fill_f32(hB);

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << "\n";
    std::cout << "Sweep tiles over 8, 16, 32 and compare time, GFLOP/s, and active blocks/SM.\n\n";

    {
        int blocks_per_sm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, gemm_tiled_kernel<8>, 8 * 8, 0));
        float t = run_tiled_gemm_case<8>(dA, dB, dC, M, N, K);
        print_tiled_case(8, t, M, N, K, blocks_per_sm);
    }

    {
        int blocks_per_sm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, gemm_tiled_kernel<16>, 16 * 16, 0));
        float t = run_tiled_gemm_case<16>(dA, dB, dC, M, N, K);
        print_tiled_case(16, t, M, N, K, blocks_per_sm);
    }

    {
        int blocks_per_sm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, gemm_tiled_kernel<32>, 32 * 32, 0));
        float t = run_tiled_gemm_case<32>(dA, dB, dC, M, N, K);
        print_tiled_case(32, t, M, N, K, blocks_per_sm);
    }

    std::cout << "\nInterpretation:\n";
    std::cout << "- Higher occupancy does not guarantee better performance.\n";
    std::cout << "- Larger tiles may increase reuse but also raise resource pressure.\n";
    std::cout << "- Non-multiple dimensions can expose quantization/padding effects.\n";
    std::cout << "- Repeat this with K=512 and K=513, or M/N as multiples vs non-multiples of 32.\n";

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}

static void usage()
{
    std::cout
        << "Usage:\n"
        << "  ./cuda_lab info\n"
        << "  ./cuda_lab coalesce <n> <block>\n"
        << "  ./cuda_lab divergence <n> <block>\n"
        << "  ./cuda_lab coarsen <n> <block> <factor:1|2|4|8>\n"
        << "  ./cuda_lab privatize <n> <block>\n"
        << "  ./cuda_lab bound pointwise <n> <block>\n"
        << "  ./cuda_lab bound gemm <M> <N> <K> <tile>\n"
        << "  ./cuda_lab tile_sweep <M> <N> <K>\n";
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        usage();
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "info")
    {
        print_device_info();
        return 0;
    }

    print_device_info();

    if (mode == "coalesce" && argc >= 4)
    {
        int n = std::atoi(argv[2]);
        int block = std::atoi(argv[3]);
        experiment_coalesce(n, block);
    }
    else if (mode == "divergence" && argc >= 4)
    {
        int n = std::atoi(argv[2]);
        int block = std::atoi(argv[3]);
        experiment_divergence(n, block);
    }
    else if (mode == "coarsen" && argc >= 5)
    {
        int n = std::atoi(argv[2]);
        int block = std::atoi(argv[3]);
        int factor = std::atoi(argv[4]);

        switch (factor)
        {
        case 1:
            experiment_coarsen_impl<1>(n, block);
            break;
        case 2:
            experiment_coarsen_impl<2>(n, block);
            break;
        case 4:
            experiment_coarsen_impl<4>(n, block);
            break;
        case 8:
            experiment_coarsen_impl<8>(n, block);
            break;
        default:
            std::cerr << "Unsupported coarsen factor: " << factor << "\n";
            return 1;
        }
    }
    else if (mode == "privatize" && argc >= 4)
    {
        int n = std::atoi(argv[2]);
        int block = std::atoi(argv[3]);
        experiment_privatize(n, block);
    }
    else if (mode == "bound" && argc >= 3)
    {
        std::string sub = argv[2];
        if (sub == "pointwise" && argc >= 5)
        {
            int n = std::atoi(argv[3]);
            int block = std::atoi(argv[4]);
            experiment_bound_pointwise(n, block);
        }
        else if (sub == "gemm" && argc >= 7)
        {
            int M = std::atoi(argv[3]);
            int N = std::atoi(argv[4]);
            int K = std::atoi(argv[5]);
            int tile = std::atoi(argv[6]);

            if (tile <= 0 || tile > 32 || tile * tile > 1024)
            {
                std::cerr << "Invalid tile. Use values like 8, 16, or 32.\n";
                return 1;
            }

            experiment_bound_gemm(M, N, K, tile);
        }
        else
        {
            usage();
        }
    }
    else if (mode == "tile_sweep" && argc >= 5)
    {
        int M = std::atoi(argv[2]);
        int N = std::atoi(argv[3]);
        int K = std::atoi(argv[4]);
        experiment_tile_sweep(M, N, K);
    }
    else
    {
        usage();
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

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
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);

    // Warm-up
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Benchmark with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 10;
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
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
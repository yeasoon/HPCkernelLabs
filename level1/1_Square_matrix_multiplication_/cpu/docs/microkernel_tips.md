# Matmul Microkernel Design Methodology
## A Complete Guide for Any Shape and Any CPU

---

## Table of Contents
1. [Gather CPU Facts](#step-1-gather-cpu-facts)
2. [Derive MR and NR From Register Count](#step-2-derive-mr-and-nr-from-register-count)
3. [Verify With FMA Pipeline Constraint](#step-3-verify-with-the-fma-pipeline-constraint)
4. [Derive KC, MC, NC From Cache Sizes](#step-4-derive-kc-mc-nc-from-cache-sizes)
5. [Design the Pack Layout](#step-5-design-the-pack-layout)
6. [Write the Microkernel](#step-6-write-the-microkernel)
7. [Handle All Shapes](#step-7-handle-all-shapes-edge-tiles)
8. [Complete Decision Flowchart](#complete-decision-flowchart)
9. [Quick Reference Card](#quick-reference-card-per-architecture)
10. [Complete C Implementation](#complete-c-implementation)

---

## Step 1: Gather CPU Facts

Before writing a single line of code, answer these questions:

```bash
# Get everything you need
lscpu | grep -E "Model|MHz|cache|NUMA|Thread|Core|Socket"

# Check SIMD support
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E "avx|fma|sse"
```

Build this table for your CPU:

```
┌─────────────────────────────────────────────────────┐
│ Parameter         │ How to find      │ E5-2620 v4    │
├─────────────────────────────────────────────────────┤
│ SIMD width        │ flags in cpuinfo │ AVX2 = 256bit │
│ Floats per reg    │ width/dtype size │ 8 (float32)   │
│ FMA units         │ uarch docs       │ 2             │
│ FMA latency       │ uarch docs       │ 5 cycles      │
│ FMA throughput    │ uarch docs       │ 0.5 cyc/insn  │
│ L1d size          │ lscpu            │ 32 KB         │
│ L2 size           │ lscpu            │ 256 KB        │
│ L3 size           │ lscpu            │ 20 MB         │
│ L1 bandwidth      │ measure          │ ~200 GB/s     │
│ DRAM bandwidth    │ measure          │ ~45 GB/s      │
│ Vector registers  │ uarch docs       │ 16 (ymm0-15)  │
└─────────────────────────────────────────────────────┘
```

**Reference docs by vendor:**
```
Intel:  software.intel.com/sites/landingpage/IntrinsicsGuide
        agner.org/optimize/instruction_tables.pdf
AMD:    developer.amd.com/resources/developer-guides-manuals
ARM:    developer.arm.com/architectures/instruction-sets/intrinsics
```

---

## Step 2: Derive MR and NR From Register Count

This is the most critical step. The goal is to **fill all vector registers with the C tile**
so C never touches memory during the K loop.

```
Formula:
  total_regs  = vector register file size (16 for AVX2, 32 for AVX-512)
  regs_for_B  = 2  (need ~2 B registers in flight)
  regs_for_A  = 1  (broadcast, reused)
  regs_for_C  = total_regs - regs_for_B - regs_for_A

  C tile shape: MR rows x NR cols
  Each C row needs NR / SIMD_WIDTH registers

  So: MR x (NR / SIMD_WIDTH) = regs_for_C

  Pick NR = multiple of SIMD_WIDTH
  Then: MR = regs_for_C x SIMD_WIDTH / NR
```

**Examples across architectures:**

```
┌──────────────┬──────────┬───────────┬──────┬──────┬────┬────────┐
│ CPU          │ ISA      │ SIMD_WIDTH│ REGS │ MR   │ NR │ C regs │
├──────────────┼──────────┼───────────┼──────┼──────┼────┼────────┤
│ Broadwell    │ AVX2     │ 8 floats  │  16  │  8   │ 16 │   16   │
│ Skylake-X    │ AVX-512  │ 16 floats │  32  │  6   │ 64 │   24   │
│ Zen3 (AMD)   │ AVX2     │ 8 floats  │  16  │  8   │ 16 │   16   │
│ Apple M1/M2  │ NEON     │ 4 floats  │  32  │  8   │ 16 │   32   │
│ ARM Cortex-A │ NEON     │ 4 floats  │  16  │  4   │ 16 │   16   │
│ SSE4.2 only  │ SSE      │ 4 floats  │   8  │  4   │  8 │    8   │
└──────────────┴──────────┴───────────┴──────┴──────┴────┴────────┘
```

**Rule of thumb:**
```
Leave 3-4 registers free for A broadcasts and B loads.
Maximize MR*NR within that constraint.
Prefer wider NR over taller MR for better B reuse.
```

---

## Step 3: Verify With the FMA Pipeline Constraint

Even with the right register tile, you can still stall the FMA pipeline
if you don't have enough independent operations in flight:

```
Required in-flight FMAs = FMA_latency x FMA_units

E5-2620 v4:  5 cycles latency x 2 units = 10 in-flight needed
Skylake:     4 cycles latency x 2 units =  8 in-flight needed
Apple M1:    4 cycles latency x 4 units = 16 in-flight needed

Your C tile provides: MR x (NR/SIMD_WIDTH) independent FMAs per k-step

Check: MR x (NR/SIMD_WIDTH) >= FMA_latency x FMA_units

E5-2620 v4:  8 x (16/8) = 16 >= 10  PASS
4x8 tile:    4 x ( 8/8) =  4 <  10  FAIL - pipeline stalls!
```

---

## Step 4: Derive KC, MC, NC From Cache Sizes

The blocking parameters control which level of cache each matrix lives in:

```
┌──────────────────────┬───────────────────────────────────┐
│ What lives where     │ Constraint                        │
├──────────────────────┼───────────────────────────────────┤
│ C tile  (MR x NR)    │ fits in registers  (step 2)       │
│ A micro (MR x KC)    │ fits in L1 during microkernel     │
│ B micro (KC x NR)    │ fits in L1 during microkernel     │
│ A panel (MC x KC)    │ fits in L2                        │
│ B panel (KC x NC)    │ fits in L3                        │
└──────────────────────┴───────────────────────────────────┘

Formulas:

  KC: (MR + NR) x KC x dtype_bytes < L1d_size
      KC < L1d / ((MR + NR) x dtype_bytes)
      E5-2620: KC < 32768 / ((8+16)x4) = 341  -> use KC=256

  MC: MC x KC x dtype_bytes < L2_size x 0.5
      MC < L2 x 0.5 / (KC x dtype_bytes)
      E5-2620: MC < 131072 / (256x4) = 128    -> use MC=128 or 256

  NC: KC x NC x dtype_bytes < L3_size x 0.5
      NC < L3 x 0.5 / (KC x dtype_bytes)
      E5-2620: NC < 10485760 / (256x4) = 10240 -> use NC=4096
```

---

## Step 5: Design the Pack Layout

Packing transforms the matrix so the microkernel accesses memory stride-1:

```
Pack A into MR-wide column strips:

  Original A (row-major):        Packed A (MR strips, column by column):
  [a00 a01 a02 a03 ...]          [a00 a10 ... a(MR-1)0]  <- k=0 panel
  [a10 a11 a12 a13 ...]    ->    [a01 a11 ... a(MR-1)1]  <- k=1 panel
  [a20 a21 ...]                  ...
                                 MR rows packed contiguously per k step

Pack B into NR-wide row strips:

  Original B (row-major):        Packed B (NR strips, row by row):
  [b00 b01 ... b(NR-1)0 ...]     [b00 b01 ... b(NR-1)0]  <- k=0 panel
  [b10 b11 ...]            ->    [b10 b11 ... b(NR-1)1]  <- k=1 panel
  ...                            NR cols packed contiguously per k step

Result: microkernel inner loop is purely stride-1
  Ap[k*MR + row]  -> sequential reads
  Bp[k*NR + col]  -> sequential reads
```

---

## Step 6: Write the Microkernel

The microkernel structure is always the same — only the intrinsics change per ISA:

```c
// Generic template — substitute intrinsics for your ISA
void microkernel_MRxNR(int K,
                        const float* Ap,   // packed, stride MR
                        const float* Bp,   // packed, stride NR
                        float*       C,
                        int          ldc)
{
    // 1. LOAD C tile into registers
    VEC c[MR][NR/SIMD_WIDTH];
    for (int i = 0; i < MR; i++)
        for (int v = 0; v < NR/SIMD_WIDTH; v++)
            c[i][v] = VLOAD(C + i*ldc + v*SIMD_WIDTH);

    // 2. K LOOP — C stays in registers, never touches memory
    for (int k = 0; k < K; k++) {
        VEC b[NR/SIMD_WIDTH];
        for (int v = 0; v < NR/SIMD_WIDTH; v++)
            b[v] = VLOAD(Bp + v*SIMD_WIDTH);

        for (int i = 0; i < MR; i++) {
            VEC a = VBROADCAST(Ap + i);
            for (int v = 0; v < NR/SIMD_WIDTH; v++)
                c[i][v] = VFMA(a, b[v], c[i][v]);
        }
        Ap += MR;
        Bp += NR;
    }

    // 3. STORE C tile back to memory
    for (int i = 0; i < MR; i++)
        for (int v = 0; v < NR/SIMD_WIDTH; v++)
            VSTORE(C + i*ldc + v*SIMD_WIDTH, c[i][v]);
}
```

**ISA substitution table:**

```
┌───────────────┬──────────────────┬──────────────────────┬──────────────────────────┐
│ ISA           │ VEC type         │ VLOAD / VSTORE        │ VFMA                     │
├───────────────┼──────────────────┼──────────────────────┼──────────────────────────┤
│ AVX2 (Intel)  │ __m256           │ _mm256_loadu_ps       │ _mm256_fmadd_ps          │
│ AVX-512       │ __m512           │ _mm512_loadu_ps       │ _mm512_fmadd_ps          │
│ NEON (ARM)    │ float32x4_t      │ vld1q_f32             │ vfmaq_f32                │
│ SVE (ARM)     │ svfloat32_t      │ svld1_f32             │ svmla_f32_x              │
│ SSE4.2        │ __m128           │ _mm_loadu_ps          │ _mm_add_ps(_mm_mul_ps()) │
│ RVV (RISC-V)  │ vfloat32m1_t     │ vle32_v_f32m1         │ vfmacc_vv_f32m1          │
└───────────────┴──────────────────┴──────────────────────┴──────────────────────────┘
```

**Concrete AVX2 example for E5-2620 v4 (MR=8, NR=16):**

```c
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma")
#include <immintrin.h>

static void microkernel_8x16(int K, const float* Ap, const float* Bp,
                              float* C, int ldc)
{
    // Load 8x16 C tile = 16 ymm registers (fills entire AVX2 register file)
    __m256 c00=_mm256_loadu_ps(C+0*ldc),   c01=_mm256_loadu_ps(C+0*ldc+8);
    __m256 c10=_mm256_loadu_ps(C+1*ldc),   c11=_mm256_loadu_ps(C+1*ldc+8);
    __m256 c20=_mm256_loadu_ps(C+2*ldc),   c21=_mm256_loadu_ps(C+2*ldc+8);
    __m256 c30=_mm256_loadu_ps(C+3*ldc),   c31=_mm256_loadu_ps(C+3*ldc+8);
    __m256 c40=_mm256_loadu_ps(C+4*ldc),   c41=_mm256_loadu_ps(C+4*ldc+8);
    __m256 c50=_mm256_loadu_ps(C+5*ldc),   c51=_mm256_loadu_ps(C+5*ldc+8);
    __m256 c60=_mm256_loadu_ps(C+6*ldc),   c61=_mm256_loadu_ps(C+6*ldc+8);
    __m256 c70=_mm256_loadu_ps(C+7*ldc),   c71=_mm256_loadu_ps(C+7*ldc+8);

    for (int k = 0; k < K; k++, Ap += 8, Bp += 16) {
        __m256 b0=_mm256_loadu_ps(Bp), b1=_mm256_loadu_ps(Bp+8);
        __m256 a;
        a=_mm256_broadcast_ss(Ap+0); c00=_mm256_fmadd_ps(a,b0,c00); c01=_mm256_fmadd_ps(a,b1,c01);
        a=_mm256_broadcast_ss(Ap+1); c10=_mm256_fmadd_ps(a,b0,c10); c11=_mm256_fmadd_ps(a,b1,c11);
        a=_mm256_broadcast_ss(Ap+2); c20=_mm256_fmadd_ps(a,b0,c20); c21=_mm256_fmadd_ps(a,b1,c21);
        a=_mm256_broadcast_ss(Ap+3); c30=_mm256_fmadd_ps(a,b0,c30); c31=_mm256_fmadd_ps(a,b1,c31);
        a=_mm256_broadcast_ss(Ap+4); c40=_mm256_fmadd_ps(a,b0,c40); c41=_mm256_fmadd_ps(a,b1,c41);
        a=_mm256_broadcast_ss(Ap+5); c50=_mm256_fmadd_ps(a,b0,c50); c51=_mm256_fmadd_ps(a,b1,c51);
        a=_mm256_broadcast_ss(Ap+6); c60=_mm256_fmadd_ps(a,b0,c60); c61=_mm256_fmadd_ps(a,b1,c61);
        a=_mm256_broadcast_ss(Ap+7); c70=_mm256_fmadd_ps(a,b0,c70); c71=_mm256_fmadd_ps(a,b1,c71);
    }

    _mm256_storeu_ps(C+0*ldc,c00); _mm256_storeu_ps(C+0*ldc+8,c01);
    _mm256_storeu_ps(C+1*ldc,c10); _mm256_storeu_ps(C+1*ldc+8,c11);
    _mm256_storeu_ps(C+2*ldc,c20); _mm256_storeu_ps(C+2*ldc+8,c21);
    _mm256_storeu_ps(C+3*ldc,c30); _mm256_storeu_ps(C+3*ldc+8,c31);
    _mm256_storeu_ps(C+4*ldc,c40); _mm256_storeu_ps(C+4*ldc+8,c41);
    _mm256_storeu_ps(C+5*ldc,c50); _mm256_storeu_ps(C+5*ldc+8,c51);
    _mm256_storeu_ps(C+6*ldc,c60); _mm256_storeu_ps(C+6*ldc+8,c61);
    _mm256_storeu_ps(C+7*ldc,c70); _mm256_storeu_ps(C+7*ldc+8,c71);
}
```

---

## Step 7: Handle All Shapes (Edge Tiles)

```
Three types of edges:
  1. M % MR != 0  -> last row panel has mr < MR rows
  2. N % NR != 0  -> last col panel has nr < NR cols
  3. K % KC != 0  -> last depth panel has kb < KC depth

Tile map example: M=100, N=200, MR=8, NR=16

         col 0    col 1    ...   col 11   col 12(edge)
       +--------+--------+-----+--------+--------+
row 0  |  8x16  |  8x16  | ... |  8x16  |  8x8   |  <- full rows, edge col nr=8
row 1  |  8x16  |  ...   |     |        |        |
...    |  ...   |        |     |        |        |
row 11 |  8x16  |  8x16  | ... |  8x16  |  8x8   |
row 12 |  4x16  |  4x16  | ... |  4x16  |  4x8   |  <- edge row mr=4, corner
       +--------+--------+-----+--------+--------+
         fast microkernel              edge kernel (temp buffer trick)
```

**Solution — zero-pad during packing + temp buffer for edge:**

```c
// pack_A zero-pads to MR rows
static void pack_A(int M, int K, const float* A, int lda, float* Ap)
{
    for (int i = 0; i < M; i += MR) {
        int ib = (M-i < MR) ? M-i : MR;          // real rows in this panel
        for (int k = 0; k < K; k++) {
            for (int ii = 0;  ii < ib; ii++) Ap[ii] = A[(i+ii)*lda+k];
            for (int ii = ib; ii < MR; ii++) Ap[ii] = 0.0f;  // zero-pad
            Ap += MR;
        }
    }
}

// pack_B zero-pads to NR cols
static void pack_B(int K, int N, const float* B, int ldb, float* Bp)
{
    for (int j = 0; j < N; j += NR) {
        int jb = (N-j < NR) ? N-j : NR;          // real cols in this panel
        for (int k = 0; k < K; k++) {
            for (int jj = 0;  jj < jb; jj++) Bp[jj] = B[k*ldb+(j+jj)];
            for (int jj = jb; jj < NR; jj++) Bp[jj] = 0.0f;  // zero-pad
            Bp += NR;
        }
    }
}

// Edge kernel: temp buffer -> full microkernel -> writeback
static void microkernel_edge(int mr, int nr, int K,
                              const float* Ap, const float* Bp,
                              float* C, int ldc)
{
    float Ctmp[MR*NR];
    memset(Ctmp, 0, sizeof(Ctmp));

    // Copy real C values into full MR x NR temp buffer
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            Ctmp[i*NR+j] = C[i*ldc+j];

    // Run full microkernel on temp (ldc=NR, all fits in registers)
    microkernel_8x16(K, Ap, Bp, Ctmp, NR);

    // Write back only the real mr x nr region
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            C[i*ldc+j] = Ctmp[i*NR+j];
}

// Dispatcher: one call handles full and edge tiles transparently
static inline void dispatch(int mr, int nr, int K,
                             const float* Ap, const float* Bp,
                             float* C, int ldc)
{
    if (mr == MR && nr == NR)
        microkernel_8x16(K, Ap, Bp, C, ldc);        // hot path (~99% of work)
    else
        microkernel_edge(mr, nr, K, Ap, Bp, C, ldc); // cold path (edges only)
}
```

---

## Complete Decision Flowchart

```
START
  |
  v
[1] PROFILE CPU
    +- SIMD_WIDTH  = floats per vector register
    +- TOTAL_REGS  = number of vector registers
    +- FMA_UNITS   = number of FMA execution units
    +- FMA_LATENCY = cycles from input to output
    +- L1/L2/L3   = cache sizes in bytes
  |
  v
[2] COMPUTE REGISTER TILE
    regs_for_C = TOTAL_REGS - 3
    NR = 2 x SIMD_WIDTH          (good starting point)
    MR = regs_for_C x SIMD_WIDTH / NR
  |
  v
[3] CHECK FMA PIPELINE
    MR x (NR/SIMD_WIDTH) >= FMA_LATENCY x FMA_UNITS ?
    NO  -> increase MR or NR -> back to [2]
    YES -> continue
  |
  v
[4] COMPUTE BLOCK SIZES
    KC = floor(L1d / ((MR+NR) x dtype_bytes)), round to power of 2
    MC = floor(L2 x 0.5 / (KC x dtype_bytes))
    NC = floor(L3 x 0.5 / (KC x dtype_bytes))
  |
  v
[5] IMPLEMENT PACKING
    pack_A: MR-wide strips, column-major within strip, zero-pad to MR
    pack_B: NR-wide strips, row-major within strip, zero-pad to NR
  |
  v
[6] WRITE MICROKERNEL
    load C tile (MRxNR regs) -> K loop: broadcast A, load B, FMA -> store C
    ALL C values stay in registers for the entire K loop
  |
  v
[7] WRITE EDGE HANDLER
    temp buffer trick: copy real C -> run full microkernel -> write back
    dispatch: mr==MR && nr==NR -> fast path, else -> edge path
  |
  v
[8] ASSEMBLE 5-LOOP GEMM
    for jc (NC strips):
      pack B panel -> fits in L3
      for kc (KC strips):
        for ic (MC strips):
          pack A panel -> fits in L2
          for ir (MR strips):
            for jr (NR strips):
              dispatch(mr, nr, kb, Ap, Bp, C, ldc)
  |
  v
[9] BENCHMARK + TUNE
    perf stat -> check IPC, L1-miss, LLC-miss
    IPC < 2.0  -> tune KC/MC/NC block sizes
    L1-miss > 20% -> reduce KC
    LLC-miss > 5% -> reduce NC
    IPC > 2.5  -> done
```

---

## Quick Reference Card Per Architecture

```
+-----------------+----------+----+----+-----+-----+-------+--------------------+
| CPU Family      | ISA      | MR | NR |  KC |  MC |    NC | Notes              |
+-----------------+----------+----+----+-----+-----+-------+--------------------+
| Intel Broadwell | AVX2     |  8 | 16 | 256 | 256 |  4096 | 16 ymm regs        |
| Intel Skylake-X | AVX-512  |  6 | 64 | 256 | 128 |  4096 | 32 zmm regs        |
| AMD Zen3        | AVX2     |  8 | 16 | 256 | 256 |  4096 | same as Intel AVX2 |
| AMD Zen4        | AVX-512  |  6 | 64 | 256 | 128 |  8192 | larger L3          |
| Apple M1        | NEON     |  8 | 16 | 512 | 256 |  8192 | 32 q-regs, fast L2 |
| ARM Cortex-A76  | NEON     |  4 | 16 | 256 | 128 |  2048 | 16 q-regs          |
| SSE4.2 only     | SSE      |  4 |  8 | 256 | 128 |  2048 | 8 xmm regs         |
| RISC-V (RVV)    | RVV      |  4 | 4x | 256 | 128 |  2048 | VLEN-dependent     |
+-----------------+----------+----+----+-----+-----+-------+--------------------+
Note: These are starting points. Always benchmark and tune for your specific CPU.
```

---

## Complete C Implementation

```c
// matmul_microkernel_any_shape.c
// Works for ANY M, N, K — handles all edge cases via zero-padding + temp buffer
// Compile: gcc -O3 -march=native -mfma -mavx2 -std=c11 -o matmul matmul_microkernel_any_shape.c -lm

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma")

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// ── Tile sizes (E5-2620 v4 tuned) ──────────────────────────────────────────
#define MR   8
#define NR  16
#define MC  256
#define KC  256
#define NC  4096

// ── Pack A: any M x K, zero-pads to MR boundary ────────────────────────────
static void pack_A(int M, int K, const float* A, int lda, float* Ap)
{
    for (int i = 0; i < M; i += MR) {
        int ib = (M-i < MR) ? M-i : MR;
        for (int k = 0; k < K; k++) {
            for (int ii = 0;  ii < ib; ii++) Ap[ii] = A[(i+ii)*lda+k];
            for (int ii = ib; ii < MR; ii++) Ap[ii] = 0.0f;
            Ap += MR;
        }
    }
}

// ── Pack B: any K x N, zero-pads to NR boundary ────────────────────────────
static void pack_B(int K, int N, const float* B, int ldb, float* Bp)
{
    for (int j = 0; j < N; j += NR) {
        int jb = (N-j < NR) ? N-j : NR;
        for (int k = 0; k < K; k++) {
            for (int jj = 0;  jj < jb; jj++) Bp[jj] = B[k*ldb+(j+jj)];
            for (int jj = jb; jj < NR; jj++) Bp[jj] = 0.0f;
            Bp += NR;
        }
    }
}

// ── Full 8x16 microkernel: C tile lives in registers entire K loop ──────────
static void microkernel_8x16(int K, const float* Ap, const float* Bp,
                              float* C, int ldc)
{
    __m256 c00=_mm256_loadu_ps(C+0*ldc),   c01=_mm256_loadu_ps(C+0*ldc+8);
    __m256 c10=_mm256_loadu_ps(C+1*ldc),   c11=_mm256_loadu_ps(C+1*ldc+8);
    __m256 c20=_mm256_loadu_ps(C+2*ldc),   c21=_mm256_loadu_ps(C+2*ldc+8);
    __m256 c30=_mm256_loadu_ps(C+3*ldc),   c31=_mm256_loadu_ps(C+3*ldc+8);
    __m256 c40=_mm256_loadu_ps(C+4*ldc),   c41=_mm256_loadu_ps(C+4*ldc+8);
    __m256 c50=_mm256_loadu_ps(C+5*ldc),   c51=_mm256_loadu_ps(C+5*ldc+8);
    __m256 c60=_mm256_loadu_ps(C+6*ldc),   c61=_mm256_loadu_ps(C+6*ldc+8);
    __m256 c70=_mm256_loadu_ps(C+7*ldc),   c71=_mm256_loadu_ps(C+7*ldc+8);

    for (int k = 0; k < K; k++, Ap += MR, Bp += NR) {
        __m256 b0=_mm256_loadu_ps(Bp), b1=_mm256_loadu_ps(Bp+8);
        __m256 a;
        a=_mm256_broadcast_ss(Ap+0); c00=_mm256_fmadd_ps(a,b0,c00); c01=_mm256_fmadd_ps(a,b1,c01);
        a=_mm256_broadcast_ss(Ap+1); c10=_mm256_fmadd_ps(a,b0,c10); c11=_mm256_fmadd_ps(a,b1,c11);
        a=_mm256_broadcast_ss(Ap+2); c20=_mm256_fmadd_ps(a,b0,c20); c21=_mm256_fmadd_ps(a,b1,c21);
        a=_mm256_broadcast_ss(Ap+3); c30=_mm256_fmadd_ps(a,b0,c30); c31=_mm256_fmadd_ps(a,b1,c31);
        a=_mm256_broadcast_ss(Ap+4); c40=_mm256_fmadd_ps(a,b0,c40); c41=_mm256_fmadd_ps(a,b1,c41);
        a=_mm256_broadcast_ss(Ap+5); c50=_mm256_fmadd_ps(a,b0,c50); c51=_mm256_fmadd_ps(a,b1,c51);
        a=_mm256_broadcast_ss(Ap+6); c60=_mm256_fmadd_ps(a,b0,c60); c61=_mm256_fmadd_ps(a,b1,c61);
        a=_mm256_broadcast_ss(Ap+7); c70=_mm256_fmadd_ps(a,b0,c70); c71=_mm256_fmadd_ps(a,b1,c71);
    }

    _mm256_storeu_ps(C+0*ldc,c00); _mm256_storeu_ps(C+0*ldc+8,c01);
    _mm256_storeu_ps(C+1*ldc,c10); _mm256_storeu_ps(C+1*ldc+8,c11);
    _mm256_storeu_ps(C+2*ldc,c20); _mm256_storeu_ps(C+2*ldc+8,c21);
    _mm256_storeu_ps(C+3*ldc,c30); _mm256_storeu_ps(C+3*ldc+8,c31);
    _mm256_storeu_ps(C+4*ldc,c40); _mm256_storeu_ps(C+4*ldc+8,c41);
    _mm256_storeu_ps(C+5*ldc,c50); _mm256_storeu_ps(C+5*ldc+8,c51);
    _mm256_storeu_ps(C+6*ldc,c60); _mm256_storeu_ps(C+6*ldc+8,c61);
    _mm256_storeu_ps(C+7*ldc,c70); _mm256_storeu_ps(C+7*ldc+8,c71);
}

// ── Edge kernel: any mr <= MR, nr <= NR via temp buffer trick ───────────────
static void microkernel_edge(int mr, int nr, int K,
                              const float* Ap, const float* Bp,
                              float* C, int ldc)
{
    float Ctmp[MR*NR];
    memset(Ctmp, 0, sizeof(Ctmp));
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            Ctmp[i*NR+j] = C[i*ldc+j];
    microkernel_8x16(K, Ap, Bp, Ctmp, NR);
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            C[i*ldc+j] = Ctmp[i*NR+j];
}

// ── Dispatcher ──────────────────────────────────────────────────────────────
static inline void dispatch(int mr, int nr, int K,
                             const float* Ap, const float* Bp,
                             float* C, int ldc)
{
    if (mr == MR && nr == NR) microkernel_8x16(K, Ap, Bp, C, ldc);
    else                       microkernel_edge(mr, nr, K, Ap, Bp, C, ldc);
}

// ── 5-loop BLIS GEMM: works for any M, N, K ────────────────────────────────
void sgemm(int M, int N, int K,
           const float* A, int lda,
           const float* B, int ldb,
           float*       C, int ldc)
{
    float* Ap = (float*)aligned_alloc(32, MC*KC*sizeof(float));
    float* Bp = (float*)aligned_alloc(32, KC*(NC+NR)*sizeof(float));

    for (int jc = 0; jc < N; jc += NC) {
        int nb = (N-jc < NC) ? N-jc : NC;
        for (int kc = 0; kc < K; kc += KC) {
            int kb = (K-kc < KC) ? K-kc : KC;
            pack_B(kb, nb, B+kc*ldb+jc, ldb, Bp);
            for (int ic = 0; ic < M; ic += MC) {
                int mb = (M-ic < MC) ? M-ic : MC;
                pack_A(mb, kb, A+ic*lda+kc, lda, Ap);
                const float* Ap_ptr = Ap;
                for (int ir = 0; ir < mb; ir += MR) {
                    int mr = (mb-ir < MR) ? mb-ir : MR;
                    const float* Bp_ptr = Bp;
                    for (int jr = 0; jr < nb; jr += NR) {
                        int nr = (nb-jr < NR) ? nb-jr : NR;
                        dispatch(mr, nr, kb, Ap_ptr, Bp_ptr,
                                 C+(ic+ir)*ldc+(jc+jr), ldc);
                        Bp_ptr += kb*NR;
                    }
                    Ap_ptr += kb*MR;
                }
            }
        }
    }
    free(Ap); free(Bp);
}

// ── Reference naive matmul for correctness check ───────────────────────────
static void naive(int M, int N, int K,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float* C, int ldc)
{
    for (int i=0;i<M;i++) for (int k=0;k<K;k++) {
        float a=A[i*lda+k];
        for (int j=0;j<N;j++) C[i*ldc+j]+=a*B[k*ldb+j];
    }
}

static int check(int M, int N, const float* ref, const float* got, int ldc)
{
    float err=0;
    for (int i=0;i<M;i++) for (int j=0;j<N;j++) {
        float d=fabsf(ref[i*ldc+j]-got[i*ldc+j]);
        if(d>err) err=d;
    }
    printf("max_err=%.2e %s\n", err, err<1e-3f?"PASS":"FAIL");
    return err<1e-3f;
}

static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

int main(void)
{
    // Test correctness on tricky non-multiple shapes
    printf("=== Correctness Tests ===\n");
    int shapes[][3] = {
        {4096, 4096, 4096},  // square, multiple of MR/NR
        {100,  200,  300},   // small non-multiples
        {1,    1,    1},     // trivial
        {1,    4096, 4096},  // skinny M (GEMV-like)
        {4096, 1,    4096},  // skinny N
        {4096, 4096, 1},     // skinny K (outer product)
        {127,  255,  511},   // just below powers of 2
        {513,  257,  129},   // just above
    };
    int ns = sizeof(shapes)/sizeof(shapes[0]);

    for (int s=0; s<ns; s++) {
        int M=shapes[s][0], N=shapes[s][1], K=shapes[s][2];
        printf("M=%-5d N=%-5d K=%-5d ", M,N,K);
        float *A=calloc(M*K,4), *B=calloc(K*N,4);
        float *R=calloc(M*N,4), *G=calloc(M*N,4);
        for (int i=0;i<M*K;i++) A[i]=(i%7)*0.1f;
        for (int i=0;i<K*N;i++) B[i]=(i%5)*0.1f;
        naive(M,N,K,A,K,B,N,R,N);
        sgemm(M,N,K,A,K,B,N,G,N);
        check(M,N,R,G,N);
        free(A); free(B); free(R); free(G);
    }

    // Benchmark on N=4096
    printf("\n=== Benchmark N=4096 ===\n");
    int N=4096;
    float *A=(float*)aligned_alloc(32,N*N*4);
    float *B=(float*)aligned_alloc(32,N*N*4);
    float *C=(float*)aligned_alloc(32,N*N*4);
    for (int i=0;i<N*N;i++) { A[i]=(i%7)*0.1f; B[i]=(i%5)*0.1f; C[i]=0; }

    sgemm(N,N,N,A,N,B,N,C,N);   // warmup
    memset(C,0,N*N*4);

    double t=now_sec();
    sgemm(N,N,N,A,N,B,N,C,N);
    t=now_sec()-t;

    printf("Time  : %.4f s\n", t);
    printf("GFLOPS: %.2f\n", 2.0*N*N*N/t/1e9);

    free(A); free(B); free(C);
    return 0;
}
```
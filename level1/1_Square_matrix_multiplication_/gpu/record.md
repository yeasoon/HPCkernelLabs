## ncu need sudo to run, so we need to add sudo before ncu command

```bash
sudo cp ncu-autho.conf  /etc/modprobe.d/
sudo update-initramfs -u
sudo reboot
```
### 1080ti not support ncu, so we need to use nvprof instead

```bash
nvprof ./matrixMul
nvprof --print-gpu-trace ./matrixMul
```
### ncu
```bash
CUDA_VISIBLE_DEVICES=0 sudo /usr/local/cuda-12.1/bin/ncu ./matmul &> ncu_log
CUDA_VISIBLE_DEVICES=0 sudo /usr/local/cuda-12.1/bin/ncu --metrics   sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed  ./matmul

CUDA_VISIBLE_DEVICES=0 sudo /usr/local/cuda-12.1/bin/ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  ./matmul

CUDA_VISIBLE_DEVICES=0 sudo /usr/local/cuda-12.1/bin/ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps  ./matmul
```

## naive
```cpp
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
dim3 threads(16, 16);
dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
```
### ncu log analysis
Let me break down exactly what your profile is telling you:
```
sm__throughput    = 98.61%  ✅ SMs are fully busy
l1tex__throughput = 98.66%  🔥 L1 is completely saturated (BOTTLENECK)
lts__throughput   = 22.56%  ⚠️  L2 barely used
dram__throughput  = 26.70%  ⚠️  DRAM barely used
```
The Diagnosis
```
L1 tex = 98.66% (saturated)
L2     = 22.56% (barely touched)
DRAM   = 26.70% (barely touched)

This means: L1 is doing ALL the work.
Every thread is independently re-fetching the same B[k][col]
from L1 cache — zero data reuse across threads.
```
The core problem: In your naive kernel, each thread in a warp accesses B[k * n + col] — threads in the same warp have different col values, so they each pull a different column of B every iteration of k. That's 16 different memory streams hitting L1 simultaneously, with no cross-thread reuse. L1 thrashes. L2 and DRAM barely matter because L1 is the wall.
```
Warp (16 threads, same row, different cols):
  Thread 0: B[k*4096 + 0]   ← cache line 0
  Thread 1: B[k*4096 + 1]   ← same cache line 0 ✓ (coalesced read)
  ...
  Thread 15: B[k*4096 + 15] ← same cache line 0 ✓

BUT: next iteration k++, different 16 columns stay resident only briefly.
With N=4096 and 256 warps competing for L1, the same B data is evicted
before other thread blocks can reuse it → L1 thrashing at 98.66%
```
## tiled
```cpp
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

```

### ncu log analysis
```
NCU Analysis — Tiled MatMul (Critical Issues Found)
This is a fascinating result. Let me decode every metric:
sm__throughput    = 99.09%  ✅ SMs fully busy
sm__warps_active  = 99.74%  ✅ Excellent occupancy
l1tex__throughput = 99.12%  🔥 STILL saturated (same problem!)
dram__throughput  =  9.72%  📉 Even WORSE than naive (was 26.7%)
lts__throughput   =  8.10%  📉 Even WORSE than naive (was 22.5%)

Bank conflicts (stores) = 134,217,728  💀 MASSIVE store conflicts
Bank conflicts (loads)  = 0            ✅ Loads are fine
Two Separate Problems
```
#### Problem 1: Shared Memory Store Bank Conflicts (💀 Critical)
134M store conflicts is catastrophic. Let's trace exactly why:
```cpp
cpp// Your B tile load:
Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
//  ↑ row=threadIdx.y, col=threadIdx.x

// Bank mapping for 32 banks, float (4 bytes):
// bank_id = (row * TILE + col) % 32
// For Bs[threadIdx.y][threadIdx.x] with TILE=32:
// bank_id = (threadIdx.y * 32 + threadIdx.x) % 32 = threadIdx.x % 32

// 16 threads in x-dim → threadIdx.x = 0..15, each hits unique bank ✅ LOAD OK

// BUT your inner compute loop reads:
sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
//                          ↑↑ Bs[k][threadIdx.x]
// All threads with same threadIdx.x but different threadIdx.y
// access Bs[k=0][0], Bs[k=0][0], Bs[k=0][0]... 
// → 16 threads reading SAME element = broadcast ✅

// The STORE conflict comes from the As load pattern:
As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
// threadIdx.y varies 0..15 → As[0..15][threadIdx.x]
// bank = (threadIdx.y * TILE + threadIdx.x) % 32
//      = (threadIdx.y * 32 + threadIdx.x) % 32
//      = threadIdx.x  (since threadIdx.y*32 % 32 = 0)
// All 16 rows map to SAME 16 banks → OK for 16-wide block

// REAL culprit: TILE=32 but blockDim=16
// Each warp stores to As[threadIdx.y][0..15]
// bank_id = threadIdx.x % 32 → unique banks ✅
// 
// Wait — let's count: 134,217,728 = 128M = 4096²/16 * 8
// This is 2-way conflict on EVERY store across N²/TILE² tiles
```

Let me show the actual conflict pattern more clearly:
```
TILE=32, blockDim.x=16 (only half the tile width!)

Bs store: Bs[threadIdx.y][threadIdx.x]
          col 0..15 only (threads don't cover full 32 columns)

Two warps fill one row of Bs:
  Warp A: Bs[row][0..15]   bank = 0..15
  Warp B: Bs[row][16..31]  bank = 16..31  (serialized with Warp A on same row!)

Actually the conflict is: TILE(32) != blockDim(16)
The tile is BIGGER than the block → multiple passes, same banks revisited!
```

---

#### Problem 2: L1 Still Saturated Despite Tiling
```
l1tex__throughput = 99.12%  (was 98.66% in naive — NO improvement!)
dram__throughput  =  9.72%  (dropped from 26.7%!)

Interpretation: Data IS being reused (DRAM dropped 3×)
but it's all hitting L1 cache, not shared memory.
The shared memory tiling isn't working as intended.
```

**Root cause — TILE vs blockDim mismatch:**
```
TILE = 32  →  32×32 smem tile = 1024 elements
blockDim = 16×16 = 256 threads

Each thread must load 1024/256 = 4 elements into smem
But your kernel only loads 1 element per thread per tile!
→ 3/4 of the smem is NEVER filled
→ The inner loop reads uninitialized smem for k=16..31
→ Falls back to L1/L2 for correctness (or gives wrong answers)
```
## coarse
```cpp
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

```

#### Why This Works: The Math
```
Smem per block:
  As[32][33] = 32×33×4 = 4224 bytes
  Bs[32][33] = 32×33×4 = 4224 bytes
  Total      = 8448 bytes ≈ 8.25 KB

Threads per block = 256 = 8 warps

CC 8.6 occupancy:
  By smem:   100KB / 8.25KB = 12 blocks/SM
  By threads: 1536 / 256    =  6 blocks/SM  ← binding limit
  By warps:   48   / 8      =  6 blocks/SM  ✅ consistent
  → 6 × 8 = 48 warps = 100% warp occupancy ✅

Arithmetic intensity per block:
  FLOPs  = 2 × TILE² × (N/TILE) = 2 × 32² × 128 = 262,144
  Bytes  = 2 × TILE² × 4B (smem load) = 8192 bytes
  AI     = 32 FLOPs/byte  (vs 0.5 naive, 16 broken-tiled)

Bank conflict analysis for As[TILE][TILE+1]:
  Store: As[row][col], bank = (row*(TILE+1) + col) % 32
         = (row*33 + col) % 32
  For col=0: row=0→0, row=1→1, row=2→2... all unique ✅
  For any col: offset by col, still stride-33, all unique ✅
  → Zero bank conflicts on both As and Bs
```
### ncu log analysis
```

Analysis — Coarse Kernel Has a Critical Bug
Good news and bad news:
sm__warps_active  = 99.00%  ✅ Occupancy restored
sm__throughput    = 95.38%  ✅ SMs busy
l1tex__throughput = 95.46%  🔥 STILL saturated (should have dropped!)
bank_conflicts_st = 134.2M  💀 SAME as broken-T32 — padding did nothing
l1tex__global_ld  = 1073M   💀 4× MORE than broken-T32 (536M → 1073M)
lts__read_hit_rate = n/a    💀 L2 STILL not participating
dram__throughput  = 34.89%  📈 Better but wrong reason
The Bank Conflict Is Still There — Here's Exactly Why
cpp// Your As declaration:
__shared__ float As[TILE][TILE + 1];  // As[32][33]

// Your store:
As[r][c] = ...  where r = ty*COARSE+i (0..31), c = tx*COARSE+j (0..31)

// Bank formula: bank = ((row * stride) + col) % 32
//               stride = TILE+1 = 33

// Thread (ty=0,tx=0): writes As[0][0], As[0][1], As[1][0], As[1][1]
// Thread (ty=0,tx=1): writes As[0][2], As[0][3], As[1][2], As[1][3]
// ...
// Thread (ty=0,tx=15): writes As[0][30], As[0][31], As[1][30], As[1][31]

// ALL 256 threads write simultaneously — let's check warp 0 (ty=0, tx=0..15):
// tx=0:  As[0][0] bank=(0*33+0)%32=0,  As[0][1] bank=1
// tx=1:  As[0][2] bank=2,              As[0][3] bank=3
// ...
// tx=15: As[0][30] bank=30,            As[0][31] bank=31
// → Warp 0 row 0: banks 0..31, all unique ✅

// BUT row 1 (i=1 in the inner loop, SAME warp, sequential store):
// tx=0:  As[1][0] bank=(1*33+0)%32=1,  As[1][1] bank=2
// tx=1:  As[1][2] bank=4,              As[1][3] bank=5
// ...FINE, but this is a SECOND STORE INSTRUCTION from same warp

// The real problem: the inner loop emits 4 SEPARATE store instructions
// per thread. They are NOT part of the same warp transaction.
// NCU counts each serialized bank access per instruction, not per warp.
The actual bug — loop structure causes repeated bank hits:
cpp// Your load loop:
for (int i = 0; i < COARSE; i++)        // i = 0, 1
    for (int j = 0; j < COARSE; j++)    // j = 0, 1
    {
        int r = ty * COARSE + i;         // r = 0,0,1,1
        int c = tx * COARSE + j;         // c = 0,1,0,1

        As[r][c] = ...   // 4 separate store instructions!
    }

// Instruction 1: all threads store As[ty*2+0][tx*2+0]
//   warp stores to columns {0,2,4,...,30} — stride 2, hits banks {0,2,4,...,30} ✅
// Instruction 2: all threads store As[ty*2+0][tx*2+1]  
//   warp stores to columns {1,3,5,...,31} — banks {1,3,5,...,31} ✅
// Instruction 3: all threads store As[ty*2+1][tx*2+0]
//   warp stores to columns {0,2,4,...,30} of ROW ty*2+1
//   bank = ((ty*2+1)*33 + tx*2) % 32 = (33 + tx*2) % 32 = (1 + tx*2) % 32
//   tx=0→1, tx=1→3, tx=2→5... tx=15→31 ✅ still unique
// Instruction 4: columns {1,3,...,31} of row ty*2+1 → banks {2,4,...,0} ✅

// So WHY are there conflicts? 
// → Bs store is the culprit, not As!
// Bs[r][c] where stride = TILE+1 = 33... same analysis applies
// Let's check Bs[row][col] with 16 threads in x covering cols 0,2,4...30

// The issue: tx*COARSE stride = 2, with 16 threads = cols 0..30 (even only)
// Two warps (ty=0..7 and ty=8..15) write to SAME columns of Bs!
// Warp (ty<8):  Bs[ty*2][tx*2], bank = (ty*2 * 33 + tx*2) % 32
// Warp (ty>=8): Bs[ty*2][tx*2], bank = (ty*2 * 33 + tx*2) % 32
// For ty=0: bank = tx*2 % 32
// For ty=8: bank = (16*33 + tx*2) % 32 = (528 + tx*2) % 32 = (16+tx*2) % 32
// For ty=0,tx=8:  bank = 16
// For ty=8,tx=0:  bank = 16  ← CONFLICT between different warps' stores!
```

#### And Why L1 Global Load Sectors QUADRUPLED (1073M)
```
l1tex__t_sectors_pipe_lsu_mem_global_op_ld = 1,073,741,824
= 4096² × 16 bytes / 64 bytes_per_sector
= exact count for loading A and B with NO cache reuse

Each block loads TILE×TILE = 1024 floats from A and 1024 from B
= 2 × 1024 × 4 = 8192 bytes = 128 cache lines = 128 sectors per block
128 blocks² × 128 sectors × 2 matrices... 

But 1073M = 2× what a correct tiled kernel would do
→ Your kernel is loading global memory TWICE per tile iteration
→ The smem isn't being used for the compute at all — falling back to L1
This means the __syncthreads() / smem pipeline is broken. The compute reads smem BEFORE it's fully written, so the compiler may be keeping values in L1 instead.

```
## bank conflict fix
```cpp

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
```

### ncu log analysis
```
Great progress — bank conflicts are zero! But we have two clear blockers:
launch__occupancy_limit_registers = 5  ← BINDING LIMIT (warps says 6, regs says 5)
sm__warps_active                  = 82.73%  ← was 99%, dropped because of this
lts__read_hit_rate                = n/a     ← L2 still bypassed
l1tex__global_ld                  = 536M    ← should be 268M, still 2× too high
l1tex__throughput                 = 95.29%  ← still saturated
```
#### Problem 1: Register Pressure is Killing Occupancy
launch__occupancy_limit_registers = 5 blocks/SM  ← binding
launch__occupancy_limit_warps     = 6 blocks/SM
launch__occupancy_limit_shared_mem= 7 blocks/SM

Registers are the bottleneck. Let's calculate:

CC 8.6: 65536 registers per SM
5 blocks × 256 threads × R registers = 65536
→ R = 65536 / (5 × 256) = 51.2 → you're using ~52 registers/thread

For 6 blocks (warp-limited target):
→ 65536 / (6 × 256) = 42.6 → need ≤ 42 registers/thread

Check with: nvcc -Xptxas="-v" → look for "Used X registers"

#### Problem 2: lts__read_hit_rate = n/a — L2 Still Bypassed
This is the most persistent issue. With bank conflicts now zero, the only remaining explanation is:
l1tex__global_ld = 536M (should be 268M for one load of A and B)

268M  = N²/TILE × TILE² × 2 matrices / 32 bytes_per_sector
      = (4096²/32) × 32² × 2 / 32
      = 2,097,152 sectors  ... let me recount

Theoretical minimum global loads for 4096×4096 matmul:
  Load A: 4096×4096 floats = 67M floats = 268M bytes = 4.19M sectors (128 bytes each? no)
  1 sector = 32 bytes = 8 floats
  A has 4096² = 16.7M floats → 2.09M sectors
  B has 2.09M sectors
  Total = 4.19M sectors... but we see 536M

536M / 4.19M = 128× more loads than necessary!
= N/TILE = 4096/32 = 128 ← each element loaded 128 times, zero L2 reuse

This means: every tile iteration reloads from DRAM, no cross-block reuse.
L2 is not caching between blocks. lts__read_hit_rate = n/a because
L2 sees so few requests (all served from L1 smem path or DRAM directly).
The for loop inside the load is causing the compiler to keep the loaded values in L1 registers rather than going through the proper smem → L2 path:
cpp// This loop structure is the culprit:
for (int e = tid; e < TILE*TILE; e += NTHREADS)
{
    int r = e / TILE;  // division = expensive, compiler keeps in registers
    int c = e % TILE;  // modulo = expensive, compiler keeps in registers
    As[r][c] = ...     // compiler may cache-bypass smem and use L1 directly
}


# ncu lab
- stride access will cause bank conflict, so we need to make sure that consecutive threads access consecutive columns in the shared memory tile. This way, we can achieve coalesced memory access and avoid bank conflicts, which will improve the performance of our matrix multiplication kernel.   40%
* Non-coalesced: Lower L1 cache throughput, higher DRAM throughput, slower kernel duration.
* Coalesced: Higher L1 cache throughput, lower DRAM throughput, faster kernel duration.
* Key takeaway: Coalesced memory accesses significantly improve performance, especially for larger inputs.

- branch divergence occurs when threads within the same warp take different execution paths due to conditional statements. This can lead to performance degradation as the GPU has to serialize the execution of different paths. To minimize branch divergence, we can structure our code to ensure that threads within the same warp follow the same execution path as much as possible, such as by using uniform conditions or by restructuring loops and conditionals to reduce divergence. 2%

- Optimizations for Memory-Bound Kernels:

* Fusions: Combining multiple operations into a single kernel to reduce memory accesses.
* Quantization: Reducing data type size to improve arithmetic intensity.
* Compilation: Using compilers like Torch Compiler to optimize memory access patterns and reduce overhead.
* Optimizations for Compute-Bound Kernels:

- Algorithm optimization: Rewriting the algorithm with fewer operations or improved mathematical formulations.
- Key takeaway: Understanding the bottleneck (memory or compute) is crucial for selecting the right optimization strategies.
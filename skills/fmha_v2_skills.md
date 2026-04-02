# SKILL: What the `fmha_v2` `.cu` Kernels Do to Get Best Performance

> **Read this before modifying any kernel in `cpp/kernels/fmha_v2/`**  
> Source lives in `cpp/kernels/fmha_v2/` and is compiled into  
> `cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu/`

This document explains the *mechanisms* the kernels use — not just what to tune, but
**why each design decision exists and what performance property it buys**.

---

## 1. The Fundamental Problem the Kernel Solves

Naive attention is memory-bandwidth-bound because it writes a full `[S, S]` score
matrix to DRAM between the two GEMMs:

```
Naive:   Q·Kᵀ ──► gmem ──► softmax ──► gmem ──► ·V ──► gmem
         ↑ writes S² × 2B every layer, every request
```

For S=4096, H=32, batch=8: that is **~8 GB of ephemeral traffic per layer**.

The kernel eliminates this entirely by keeping the live state in SMEM and registers:

```
fmha_v2: [Q tile · K tile ──► partial softmax ──► · V tile]  (zero gmem for scores)
          └────────────────── all inside one CTA ─────────────┘
```

Every other optimization in the kernel is in service of making *this loop* run as
fast as the GPU's tensor cores can go.

---

## 2. The Inner Loop: What Each Iteration Actually Does

One pass through the inner loop processes one `(Q_tile, K_tile, V_tile)` triple.
Here is the exact sequence with the hardware instructions used:

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Load K tile → SMEM                                 │
│    Ampere:  cp.async  (LDGSTS)    — async, non-blocking     │
│    Hopper:  TMA load via descriptor — hardware-offloaded    │
│                                                             │
│  STEP 2: GEMM-I:  S = Q · Kᵀ  (scales by 1/√d)            │
│    Ampere:  mma.sync.m16n8k16  (HMMA, per-warp)            │
│    Hopper:  wgmma.mma_async    (WGMMA, 128-thread warpgrp) │
│    → accumulates in FP32 registers (when FP32Acc=1)        │
│                                                             │
│  STEP 3: Online softmax update                              │
│    m_new = max(m_old, rowmax(S))       — warp reduce        │
│    α     = exp(m_old - m_new)          — rescale factor     │
│    P     = exp(S - m_new)              — elementwise exp    │
│    ℓ_new = α·ℓ_old + rowsum(P)        — warp reduce        │
│    O_acc = α·O_acc   ← rescale old output accum in-place   │
│                                                             │
│  STEP 4: Load V tile → SMEM  (issued early, overlaps GEMM) │
│                                                             │
│  STEP 5: GEMM-II:  O = P · V                               │
│    same instruction as GEMM-I                               │
│    accumulates into O_acc (FP32 registers)                  │
└─────────────────────────────────────────────────────────────┘
After all tiles: O = O_acc / ℓ_new  (epilogue, once per Q row)
```

**Everything in steps 2–5 is register / SMEM resident. Nothing touches gmem.**

---

## 3. Double-Buffering: Hiding Memory Latency

Global memory latency is ~600 cycles. The kernel hides this with double-buffering
(called "ping-pong" in some CUTLASS contexts).

Two SMEM staging buffers are allocated for K (and V). While the compute pipeline
runs on buffer A, the memory pipeline prefetches into buffer B:

```
Iteration j:
  ┌── compute ──────────────────────────────────────┐
  │  GEMM-I(Q, K_buf_A)                             │
  │  softmax update                                  │
  │  GEMM-II(P, V_buf_A)                            │
  └──────────────────────────────────────────────────┘
  ┌── async prefetch ──────────────────────────────┐
  │  cp.async K[j+1] → K_buf_B                     │
  │  cp.async V[j+1] → V_buf_B                     │
  └────────────────────────────────────────────────┘
Iteration j+1: swap buf_A ↔ buf_B, repeat
```

On **Hopper**, the TMA hardware does this even more efficiently: the producer
warpgroup fires off TMA descriptors, posts a barrier, and the consumer warpgroup
`arrive/wait`s on it. Zero compute cycles are spent on address generation or
predication.

**Cost of NOT doing this**: memory-stall cycles equal to latency × number of tiles,
which for S=4096, d=128 is ~9600 cycles of dead time.

---

## 4. Warp Specialization (Hopper Only)

On SM90, the CTA is split into two warpgroups that **never switch roles**:

```
CTA (256 threads = 2 warpgroups)
 ├── Producer WG (128 threads, warp 0-3):
 │     Only issues TMA loads.
 │     Arrives on mbarrier after each load.
 │     Never touches tensor cores.
 │
 └── Consumer WG (128 threads, warp 4-7):
       Waits on mbarrier.
       Only issues WGMMA instructions.
       Never touches memory pipes.
```

Why this matters: **WGMMA and TMA compete for instruction dispatch slots**.
Interleaving them in the same warp causes pipeline stalls. Separation achieves
near-perfect overlap: while the consumer does GEMM, the producer is already
fetching the next tile.

This is the single biggest performance lever on H100/H200. Kernels with
`WarpSpecialization=0` leave 20-40% peak throughput on the table on Hopper.

**The kernel template parameter `WarpSpecialization` (0 or 1) controls this.**
Always use `WarpSpecialization=1` on SM90 targets.

---

## 5. GEMM-I / GEMM-II Pipeline Overlap

Within a single iteration, GEMM-I and the V-load can overlap:

```
Time ──►
   [issue V TMA load]
   [GEMM-I: Q·Kᵀ      ←←← 40 cycles ←←←]
   [softmax: 8 cycles]
            [wait V barrier: should already be done]
   [GEMM-II: P·V       ←←← 40 cycles ←←←]
   [issue K TMA load for next iter]
```

The V load is issued *before* GEMM-I starts. By the time GEMM-I + softmax
complete (~48 cycles), the V tile is already in SMEM. This hides the full V
load latency (~50-100 cycles of TMA latency) behind useful compute.

The code pattern (simplified):

```cpp
// Issue V load early
tma_load(smem_v_next, tma_v_desc, iter + 1);

// Run GEMM-I — V is loading in background
wgmma(acc_s, smem_q, smem_k);   // Q·Kᵀ

// Softmax on acc_s (while V load finishes)
update_softmax(acc_s, m, l);

// Wait for V (usually already done)
mbar_wait(v_barrier);

// Run GEMM-II
wgmma(acc_o, acc_p, smem_v);   // P·V
```

---

## 6. Online Softmax — Why It's Numerically Correct

Storing `(m, ℓ)` statistics and rescaling avoids two passes:

```
Standard 2-pass:  pass 1 → compute max → store S to gmem
                  pass 2 → compute exp(S - max), sum, divide
                  cost: 2× gmem round-trips for full S matrix

Online (Flash):   single pass over K tiles, no S stored
                  rescale O accumulator each iteration

Rescaling identity (critical):
  O_final = Σⱼ ( exp(Sᵢⱼ - m_final) · Vⱼ ) / ℓ_final
           = Σⱼ αⱼ · P_partial_j · Vⱼ   where αⱼ = exp(m_old_j - m_final)
```

**Register layout**: `m` and `ℓ` are stored as per-row scalars in registers
(one float per Q row per thread). For `bM=64` rows and 128 threads per warpgroup,
each thread owns `64/128 * 2 = 1` row of stats (in the common case).

**FP32 accumulation** for stats (`FP32Accumulation=1`) is non-negotiable for
correct results at long sequence lengths. The exponent subtraction `m_old - m_new`
can be very small in FP16 and produces zero-exp underflow. Always use
`enabled_with_fp32_acc` in production.

---

## 7. SMEM Layout: Swizzling to Eliminate Bank Conflicts

SMEM is organized as 32 banks × 4 bytes. A naive row-major layout of K
(`[64, 128]` in FP16) causes 8-way bank conflicts on the column-stride access
pattern of GEMM.

The kernel applies an XOR swizzle to the address mapping:

```
Naive address:  smem[row][col]
Swizzled:       smem[row][col ^ (row >> 3)]   // XOR bits [3:5] of row into col
```

This spreads consecutive K columns across different banks. Zero conflicts = 2×
effective SMEM bandwidth.

On **Hopper**, TMA descriptors carry a `TMA_SWIZZLE_128B` attribute that the
hardware applies automatically during the async copy. The kernel does not need
explicit XOR logic — it just sets the descriptor correctly.

**If you add a new tile size, re-derive the swizzle pattern.** Wrong swizzle =
silent bank conflicts = 30-50% SMEM throughput loss.

---

## 8. Register File Pressure Management

FP32 O accumulator for `bM=64, d=128`: `64 × 128 × 4B = 32 KB` per CTA.
With 128 threads per warpgroup, each thread holds `32768 / 128 = 256 bytes = 64 float registers`.

NVIDIA GPUs have 255 registers/thread. The kernel is at the edge of this limit.

Consequences:
- Exceed 255 regs → compiler spills to local memory (LMEM) → catastrophic perf loss
- Reduce `bM` to 32 to lower register count at the cost of Q-tile reuse

**How to check**: compile with `--ptxas-options=-v` and read:
```
ptxas: 24576 bytes smem, 192 registers
           ↑                 ↑
       should be < SM limit  should be < 255
```

If registers > 240, reduce `bM` or switch to FP16 accumulation for the O buffer
(at the cost of precision).

---

## 9. What the Template Parameters Do to the Generated Code

Each combination becomes a separate `.cu` instantiation. Understanding what each
controls helps when reading or extending the code.

| Parameter | Value | What the Compiled Kernel Does Differently |
|---|---|---|
| `FlashAttention=0` | vanilla | No tiling; full `S×S` score in SMEM; fast only for S<256 |
| `FlashAttention=1` | tiled FA2 | Inner loop over K/V tiles; O(S) SMEM |
| `WarpSpecialization=0` | off | Single warpgroup does both load and compute (Ampere style) |
| `WarpSpecialization=1` | on | Producer/consumer split; Hopper-only; +20-40% throughput |
| `Tiled=0` | non-tiled | All of K and V fit in SMEM at once (small S) |
| `Tiled=1` | tiled | K/V streamed in blocks; handles arbitrary S |
| `FP32Accumulation=0` | FP16 acc | Faster; may lose precision for long S; avoid in production |
| `FP32Accumulation=1` | FP32 acc | Stable softmax stats; required for S > ~2K |
| `Interleaved=1` | QKV packed | Input is `[B, S, 3, H, D]`; stride arithmetic differs |
| `Interleaved=0` | separate | Input is `[B, S, H, D]` × 3; simpler addressing |
| `AttentionMaskType=1` | causal | Upper triangular masked; GEMM-I skips future K tiles |
| `AttentionMaskType=2` | sliding window | Only last W tokens attend; K/V tile loop starts later |

---

## 10. What the Causal Mask Optimization Does

For causal (autoregressive) attention, future K tiles (j > current Q tile) contain
entirely masked tokens. The kernel skips these with a bounds check:

```cpp
// No work done when all K indices > max valid Q index for this tile
if (k_tile_start > q_tile_end) continue;   // skip entire GEMM-I + GEMM-II
```

For partial tiles (boundary between masked and unmasked), the kernel applies the
mask as a conditional min before softmax:

```cpp
float mask = (k_idx <= q_idx) ? 0.f : -INFINITY;
s += mask;   // fused with GEMM-I epilogue
```

**This halves the FLOPs for S≫D** (the typical case). The effective arithmetic
intensity is ~2× higher than uncausal attention for the same sequence length.

---

## 11. Paged KV Cache: How the Kernel Handles Indirection

With paged KV, the K and V tiles are not contiguous. Each block is a separately
allocated `[block_size, H, D]` chunk.

The kernel receives a **block pointer table** (array of pointers, one per page).
The inner loop:
1. Computes `page_idx = k_tile_offset / block_size`
2. Reads `block_ptr = kv_block_table[page_idx]`
3. Loads from `block_ptr + (k_tile_offset % block_size) * stride`

This introduces one L2/gmem indirection per tile. To minimize cost:
- `block_size` should be ≥ 16 tokens (to amortize the pointer fetch)
- On Hopper, use TMA with per-tile descriptor updates — the kernel reconfigures
  the TMA descriptor's base address each iteration from the page table

**KV block reuse** (`kv_cache_enable_block_reuse`): blocks from prior requests can
be shared. The kernel treats them identically — read-only access to shared pages is
safe because the kernel never writes K/V.

---

## 12. FP8 Path (Ada SM89 / Hopper SM90)

When `use_fp8_context_fmha=enable`:

```
Input:  Q, K in E4M3 (FP8) — range ±448
        Scale tensors: q_scale, k_scale, v_scale (per tensor or per-head)

GEMM-I: E4M3 × E4M3 → FP32 acc
        result S is in FP32 before softmax (critical for stability)

GEMM-II: FP8 × FP32 → FP32 acc
         P (the softmax output) stays FP32
         V is in E4M3

Output: rescaled from FP32 → FP8 or FP16 in epilogue
```

**Why it needs paged context FMHA simultaneously**: FP8 KV cache blocks store
quantized K/V. The kernel dequantizes on load using the scale tensor. Non-paged
mode cannot co-locate the per-block scale with the block data.

---

## 13. Kernel Epilogue: What Happens After the Inner Loop

After all K/V tiles are processed, each thread holds a partial O accumulator and
`(m, ℓ)` statistics. The epilogue:

1. **Finalize O**: `O[i] = O_acc[i] / ℓ[i]` — one division per row
2. **Write to gmem**: coalesced write of the completed `[bM, d]` output tile
3. **Optionally return softmax stats**: if `ReturnSoftmaxStats=1` (needed for
   chunked context), write `(m, log(ℓ))` to a separate output buffer so the
   calling code can merge results across chunks

For **FP16 output**: the FP32 O is rounded to FP16 in the epilogue store
instruction (`stg.global.v4` with FP32→FP16 conversion). The store is vectorized
4-wide (64-bit stores) for peak bandwidth.

---

## 14. Build System: How `.cu` Files Are Generated

`fmha_v2` sources are not compiled directly. The build script generates `.cu`
instantiation files:

```
build_wheel.py
  → iterates over all (arch, dtype, S, D, mask, ...) combinations
  → emits: fmha_flash_attn_fp16_S128_D128_causal_sm80.cu
           (each file is just: #include "kernel.h" + explicit template instantiation)
  → places them in: contextFusedMultiHeadAttention/fmha_v2_cu/
  → each is compiled to a .o with nvcc -arch=sm_80 -maxrregcount=...
```

**To add a new kernel variant**: add the instantiation to the generation script,
not to a `.cu` file directly. The script also sets `-maxrregcount` per arch to
avoid register spills.

---

## 15. Performance Impact Summary

| Technique | Perf Gain | Where |
|---|---|---|
| SMEM tiling (vs. naive) | 5–10× | Memory bandwidth |
| Double-buffering async loads | 1.3–1.5× | Hide load latency |
| Warp specialization (SM90) | 1.2–1.4× | Overlap TMA + WGMMA |
| GEMM-I/II V-load overlap | 1.05–1.15× | Per-iteration latency |
| SMEM swizzle | 1.1–1.3× | Eliminate bank conflicts |
| Causal mask skip | ~1.5–2× FLOPs | Arithmetic reduction |
| FP8 path | 1.5–2× | Reduce memory BW + TC throughput |
| Paged KV with block reuse | up to 2× E2E throughput | KV cache hit rate |

**Key insight**: the first two rows (tiling + double-buffering) are the floor.
Everything else is incremental. If a kernel modification breaks either of these,
expect catastrophic regression regardless of what else it adds.

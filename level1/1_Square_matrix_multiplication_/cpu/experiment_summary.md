# CPU Experiment Summary

This file summarizes the CPU-side square matrix multiplication experiments in this directory and condenses the longer notes in `docs/record.md`, `docs/perf.md`, and `docs/microkernel_tips.md`.

## Goal

Optimize `N = 4096` square matrix multiplication on the CPU by improving memory locality first, then improving SIMD utilization and register reuse.

## Test Platform

- CPU: Intel Xeon E5-2620 v4 @ 2.10 GHz
- ISA features relevant to this work: AVX2 + FMA
- Topology: 2 sockets, 8 cores per socket, 2 threads per core
- Cache hierarchy noted in the experiments:
  - L1d: 512 KiB total
  - L2: 4 MiB total
  - L3: 40 MiB total

## Experiment Progression

| Stage | Main idea | Time | Performance | Main conclusion |
|---|---|---:|---:|---|
| Naive triple loop | Baseline `i-j-k` traversal | 345.968 s | 0.397 GFLOPS | Extremely memory-bound due to poor access pattern for `B` |
| Loop transposition | Change order to `i-k-j` | 24.023 s | 5.721 GFLOPS | Huge gain from better spatial locality and reuse |
| Cache blocking | Tile the matrix multiply, best shown at `BS=1024` | 13.004 s | 10.569 GFLOPS | Strong reduction in LLC misses; cache locality becomes much better |
| AVX2 + unroll4 | Add vectorization and instruction-level parallelism | 11.980 s | 11.473 GFLOPS | SIMD helps, but register reuse and packing are still the next bottlenecks |
| Microkernel design target | Keep `C` in registers and pack `A/B` | target ~10.5 s | target ~14–16 GFLOPS | Main remaining opportunity is a true BLIS-style microkernel |

## 1. Baseline: Naive Implementation

The baseline implementation uses the classic `i-j-k` loop nest. This is the worst case for a row-major `B` matrix because the innermost loop walks down a column of `B`, causing poor cache locality.

### Result

- Time: **345.968 s**
- Performance: **0.397259 GFLOPS**

### Key observations

- IPC was only about **0.10**
- Cache miss rate was about **65%**
- LLC misses were very high
- dTLB misses were also extremely high

### Takeaway

The baseline kernel is dominated by memory stalls rather than arithmetic throughput.

## 2. Loop Transposition

The first major optimization changes the loop order to `i-k-j`. This hoists `A[i, k]` into a scalar and lets the innermost loop traverse rows of `B` and `C` contiguously.

### Result

- Time: **24.0226 s**
- Performance: **5.72123 GFLOPS**

### Why it helped

- `B[k, j]` becomes contiguous in memory inside the inner loop
- `C[i, j]` is updated in a contiguous row
- Reuse of `A[i, k]` improves

### Takeaway

This was the biggest single jump in performance because it fixed the worst memory-access problem before any SIMD work was added.

## 3. Cache Blocking

The next step applies tiling/blocking so smaller submatrices fit better into cache. The experiment notes show a block-size sweep, with the best result highlighted at **BS = 1024**.

### Best reported result

- Time: **13.0037 s**
- Performance: **10.5692 GFLOPS**

### Why it helped

- Much lower LLC miss rate
- Better temporal reuse of tiles from `A`, `B`, and `C`
- Clear evidence that cache hierarchy, not just loop order, was limiting performance

### Takeaway

After fixing traversal order, the next most important gain came from fitting useful working sets into cache.

## 4. AVX2 + Unroll4

After improving locality, the next experiment adds AVX2 vector instructions and loop unrolling. This increases arithmetic throughput and reduces branch overhead in the hot loop.

### Best reported result

- Time: **11.9795 s**
- Performance: **11.4728 GFLOPS**

### What improved

- More work per instruction using 256-bit SIMD registers
- Lower branch-miss rate
- Better instruction throughput than scalar blocked code

### Remaining limitation

The kernel still reloads data that an optimized microkernel would keep in registers. SIMD helped, but the implementation is still not using the full register file efficiently.

## 5. Microkernel Direction

The notes identify the next major optimization step as a **true microkernel** design:

- Keep a small tile of `C` in registers for the whole inner `K` loop
- Pack `A` and `B` so the microkernel reads stride-1 data
- Use all 16 AVX2 YMM registers for the output tile
- Target a register tile such as **MR = 8**, **NR = 16**

### Expected outcome from the notes

- Current best so far: about **~13 GFLOPS** / **~10.5 s** class performance
- Estimated microkernel ceiling: **~14–16 GFLOPS**
- Theoretical AVX2 peak noted in the docs: **16.8 GFLOPS**

### Takeaway

The optimization path is now shifting from “fix memory access” to “maximize register reuse and FMA throughput.”

## Overall Conclusions

1. **Memory access pattern matters more than anything at the beginning.**
   - The naive algorithm was catastrophically memory-bound.
2. **Loop reordering produced the largest performance jump.**
   - It turned an unusable access pattern into a cache-friendly one.
3. **Cache blocking delivered the next major gain.**
   - It dramatically reduced expensive LLC misses.
4. **SIMD improved throughput, but not enough on its own.**
   - AVX2 + unrolling helped only after locality was fixed.
5. **The next step is a real packed microkernel.**
   - That is the clearest remaining path toward OpenBLAS-like efficiency.

## Recommended Next Steps

- Implement and benchmark the packed microkernel in `src/microkernel.cpp`
- Add a short table comparing all versions in one place after each run
- Capture the same `perf stat` counters for every version for easier regression tracking
- Add a small reproducibility section with exact build and run commands

## Related Files

- `docs/record.md`: full experiment log and measured numbers
- `docs/perf.md`: perf workflow and bottleneck analysis guide
- `docs/microkernel_tips.md`: microkernel design methodology
- `src/naive.cpp`: baseline and blocked CPU implementation work
- `src/microkernel.cpp`: current microkernel-oriented implementation

# 🔬 Using `perf` to Analyze CPU Kernel Bottlenecks

---

## 1. Install & Setup
```bash
# Install perf
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)

# Verify
perf --version

# Allow perf without root (optional)
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
echo 0   | sudo tee /proc/sys/kernel/kptr_restrict

# Disable CPU frequency scaling for stable results
sudo cpupower frequency-set -g performance
# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# → performance
```

---

## 2. The 5 Levels of perf Analysis
```
Level 1 — perf stat        → macro bottleneck (IPC, cache, branch)
     ↓
Level 2 — perf record      → which functions are hot
     ↓
Level 3 — perf annotate    → which lines inside function are hot
     ↓
Level 4 — perf mem         → memory access pattern analysis
     ↓
Level 5 — toplev / pmu     → microarchitecture top-down analysis
```

---

## 3. Level 1 — perf stat (Start Here Always)

### 3a. Basic run
```bash
perf stat ./my_kernel

# Output:
# Performance counter stats for './my_kernel':
#
#     14,203,411,204      cycles
#      2,104,882,301      instructions       # 0.15 insn per cycle  ← LOW IPC
#         38,291,044      cache-misses       # 91.2% of all cache refs ← BAD
#         41,982,110      cache-references
#            921,334      branch-misses      # 4.2% of all branches
#
#       3.841194234 seconds time elapsed
```

### 3b. Full bottleneck stat (use this every time)
```bash
perf stat -e \
  cycles,\
  instructions,\
  cache-misses,\
  cache-references,\
  branch-misses,\
  branches,\
  L1-dcache-loads,\
  L1-dcache-load-misses,\
  L1-icache-load-misses,\
  LLC-loads,\
  LLC-load-misses,\
  dTLB-load-misses,\
  iTLB-load-misses,\
  cpu-migrations,\
  context-switches \
  ./my_kernel
```

### 3c. Repeat 5 times for stable median
```bash
perf stat --repeat 5 -e \
  cycles,instructions,cache-misses,branch-misses \
  ./my_kernel

# Output adds variance:
# 14,203,411,204  cycles    ( +-  0.12% )
#  2,104,882,301  instructions ( +-  0.08% )
```

### 3d. Read the output — diagnosis table
```
Metric                      Good        Warn        Bad         Meaning
─────────────────────────────────────────────────────────────────────────
insn per cycle (IPC)        > 2.0       1.0-2.0     < 1.0       CPU utilization
cache-miss rate             < 1%        1-10%       > 10%       Memory bound
LLC-load-miss rate          < 5%        5-20%       > 20%       DRAM bound
branch-miss rate            < 1%        1-5%        > 5%        Branch predict
dTLB-load-misses            < 0.1%      0.1-1%      > 1%        TLB thrashing
context-switches            0           < 10        > 100       OS interference
```

---

## 4. Level 2 — perf record (Find Hot Functions)

### 4a. Basic hotspot recording
```bash
# Record with call graph
perf record -g ./my_kernel

# View report
perf report

# Non-interactive text output
perf report --stdio | head -50
```

### 4b. High-frequency sampling (more precise)
```bash
# Sample every 100us (10000 Hz)
perf record -F 10000 -g ./my_kernel

# Sample specific events instead of time
perf record -e cache-misses -g ./my_kernel
perf record -e LLC-load-misses -g ./my_kernel
perf record -e branch-misses -g ./my_kernel
```

### 4c. Record with DWARF for better stack unwind
```bash
perf record --call-graph dwarf -F 5000 ./my_kernel
perf report --call-graph --stdio
```

### 4d. Sample output
```
# Overhead  Command     Shared Object     Symbol
# ........  ..........  ................  .......................
    42.31%  my_kernel   my_kernel         [.] matrix_multiply
    18.74%  my_kernel   my_kernel         [.] attention_forward
     8.42%  my_kernel   libgomp.so        [.] GOMP_parallel
     4.21%  my_kernel   my_kernel         [.] layer_norm
     3.18%  my_kernel   libc.so           [.] memcpy
     2.94%  my_kernel   my_kernel         [.] softmax_forward

# → matrix_multiply takes 42% of time → optimize this first
```

### 4e. Focus on one function
```bash
# Record only the hot function
perf record -e cycles --dso my_kernel \
    --symbol matrix_multiply ./my_kernel

# Or filter in report
perf report --symbol-filter matrix_multiply
```

---

## 5. Level 3 — perf annotate (Find Hot Lines)

### 5a. Annotate hot function
```bash
# Must compile with debug info
gcc -O2 -g my_kernel.c -o my_kernel

perf record -g ./my_kernel
perf annotate matrix_multiply

# Or non-interactive
perf annotate --stdio matrix_multiply
```

### 5b. Sample annotated output
```asm
  Percent | Source code & Disassembly
─────────────────────────────────────────────────────
          : void matrix_multiply(float* A, float* B,
          :                      float* C, int N) {
          :     for (int i = 0; i < N; i++) {
     0.12 :       mov  eax, DWORD PTR [rbp-0x8]
          :         for (int j = 0; j < N; j++) {
     0.08 :           mov  ecx, DWORD PTR [rbp-0xc]
          :             for (int k = 0; k < N; k++) {
    42.31 :               vmovss xmm0,[rax+rdx*4]     ← HOT: loading B[k][j]
    38.74 :               vfmadd231ss xmm1,xmm0,[rcx] ← HOT: fused multiply-add
     8.12 :               add rdx, 0x1
     4.21 :               cmp rdx, rsi

# → 42% on vmovss loading B[k][j]
# → Column-major access on row-major B → cache miss on every load
# → Fix: transpose B or apply loop tiling
```

### 5c. Annotate with source interleaved
```bash
perf annotate --stdio -l --no-plabel \
    --sym-annotate matrix_multiply \
    -- ./my_kernel 2>&1 | head -80
```

---

## 6. Level 4 — perf mem (Memory Access Analysis)

### 6a. Record memory events
```bash
# Record memory load/store events
perf mem record ./my_kernel

# View memory report
perf mem report

# Non-interactive
perf mem report --stdio | head -40
```

### 6b. Sample memory report
```
# Overhead  Symbol                  Local Weight  Memory access
    42.1%   matrix_multiply          236          L3 miss
    18.7%   attention_forward         89          L2 hit
     8.4%   layer_norm                12          L1 hit
     4.2%   softmax_forward          198          L3 miss

# Local Weight = average memory latency in cycles
# L3 miss = going to DRAM = 100-300 cycles
# L1 hit  = 4 cycles
```

### 6c. Cache line false sharing detection
```bash
# perf c2c — detect false sharing between threads
perf c2c record -g ./my_openmp_kernel
perf c2c report

# Output:
# Shared Cache Line Event Information
# ─────────────────────────────────────────────────
# Total records   :    143234
# Loads           :    112847
# Stores          :     30387
#
# ─── False sharing hotspot ───
# 0xffff8801234: thread_results[0]  ← threads sharing cache line!
# Fix: pad struct to 64 bytes (cache line size)
```

---

## 7. Level 5 — toplev Top-Down Analysis

### 7a. Install and run
```bash
# Install pmu-tools
git clone https://github.com/andikleen/pmu-tools
cd pmu-tools

# Run top-down level 1 (macro categories)
./toplev --level 1 ./my_kernel
```

### 7b. Drill down level by level
```bash
# Level 1 — find dominant category
./toplev --level 1 -v ./my_kernel
# Output:
# FE             Frontend_Bound:   8.2%
# BAD            Bad_Speculation:  9.1%
# BE             Backend_Bound:   71.4%  ← dominant
# RET            Retiring:        11.3%

# Level 2 — drill into Backend_Bound
./toplev --level 2 -v ./my_kernel
# Output:
# BE/Mem         Memory_Bound:    64.1%  ← memory is the issue
# BE/Core        Core_Bound:       7.3%

# Level 3 — drill into Memory_Bound
./toplev --level 3 -v ./my_kernel
# Output:
# BE/Mem/L1      L1_Bound:        12.3%
# BE/Mem/L2      L2_Bound:         8.1%
# BE/Mem/L3      L3_Bound:        15.8%
# BE/Mem/DRAM    DRAM_Bound:      27.9%  ← going to DRAM
# BE/Mem/Store   Store_Bound:      0.0%

# Level 4 — DRAM_Bound drill
./toplev --level 4 -v ./my_kernel
# Output:
# BE/Mem/DRAM/MLP      MLP:          2.1%
# BE/Mem/DRAM/Mem_BW   Memory_BW:   24.3%  ← bandwidth saturated
# BE/Mem/DRAM/Mem_Lat  Memory_Lat:   1.5%
```

### 7c. Get concrete advice from toplev
```bash
./toplev --level 4 --verbose --metric-group +Summary \
    -v ./my_kernel 2>&1 | grep -A 3 "Bottleneck\|Advice\|Suggest"

# Output:
# Bottleneck: Memory_BW 24.3%
# Advice: Apply loop tiling to reduce DRAM traffic.
#         Suggested tile size for L2 (256KB): T = 80
#         Rewrite: gcc -O3 -DTILE=80 -march=native matmul.c
```

---

## 8. Flamegraph — Visual Bottleneck

### 8a. Generate flamegraph
```bash
# Get FlameGraph tools
git clone https://github.com/brendangregg/FlameGraph
export PATH=$PATH:$(pwd)/FlameGraph

# Record
perf record -F 99 -g ./my_kernel

# Convert to flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# Open in browser
firefox flame.svg
# or
google-chrome flame.svg
```

### 8b. Differential flamegraph (before vs after)
```bash
# Record baseline
perf record -F 99 -g ./baseline -o perf_before.data
perf script -i perf_before.data | stackcollapse-perf.pl > before.folded

# Record optimized
perf record -F 99 -g ./optimized -o perf_after.data
perf script -i perf_after.data | stackcollapse-perf.pl > after.folded

# Diff flamegraph (red = regression, blue = improvement)
difffolded.pl before.folded after.folded | flamegraph.pl > diff.svg
firefox diff.svg
```

---

## 9. Per-Core & Threading Analysis

### 9a. Per-CPU breakdown
```bash
# Show per-CPU stats
perf stat -A -e cycles,instructions,cache-misses \
    ./my_openmp_kernel

# Output:
# CPU0   4,201,234,123  cycles     # ← 42% of total
# CPU1   4,198,721,456  cycles     # balanced
# CPU2      102,341,234  cycles    # ← only 1% ! underutilized
# CPU3      103,211,456  cycles    # ← only 1% ! underutilized
# → Threads not spread across all cores
```

### 9b. Thread-level profiling
```bash
# Record per-thread
perf record -g --per-thread ./my_openmp_kernel

# Check thread migration (bad for cache locality)
perf stat -e cpu-migrations ./my_openmp_kernel
# cpu-migrations: 0      → good (threads stay on same core)
# cpu-migrations: 1234   → bad  (OS moving threads around)

# Pin threads to cores
taskset -c 0-7 ./my_openmp_kernel      # use cores 0-7
numactl --cpunodebind=0 ./my_kernel    # use NUMA node 0 only
```

---

## 10. Full Diagnosis Script
```bash
#!/bin/bash
# Usage: ./diagnose.sh ./my_kernel
BINARY=$1

echo "================================================"
echo " STEP 1: Macro bottleneck (perf stat)"
echo "================================================"
perf stat --repeat 3 -e \
  cycles,instructions,\
  cache-misses,cache-references,\
  LLC-loads,LLC-load-misses,\
  branch-misses,branches \
  $BINARY 2>&1 | tail -20

echo ""
echo "================================================"
echo " STEP 2: Hot functions (perf record + report)"
echo "================================================"
perf record -g -F 5000 $BINARY -o /tmp/perf_diag.data 2>/dev/null
perf report -i /tmp/perf_diag.data --stdio --no-children 2>/dev/null \
  | head -25

echo ""
echo "================================================"
echo " STEP 3: Top-down microarch (toplev)"
echo "================================================"
~/pmu-tools/toplev --level 3 -v $BINARY 2>&1 \
  | grep -E "Bound|Retiring|Speculation|Frontend"

echo ""
echo "================================================"
echo " STEP 4: Memory access pattern (perf mem)"
echo "================================================"
perf mem record $BINARY -o /tmp/perf_mem.data 2>/dev/null
perf mem report -i /tmp/perf_mem.data --stdio 2>/dev/null | head -20

echo ""
echo "================================================"
echo " STEP 5: VERDICT"
echo "================================================"
python3 - <<'EOF'
# Parse stats and print verdict
# (replace values with parsed ones from step 1)
ipc          = 0.15   # instructions / cycles
cache_miss   = 91.2   # % cache miss rate
branch_miss  = 4.2    # % branch miss rate
dram_bound   = 27.9   # % toplev DRAM_Bound

print("Bottleneck Analysis:")
if ipc < 1.0:
    print("  ❌ LOW IPC — CPU stalling, not retiring instructions")
if cache_miss > 10:
    print("  ❌ HIGH CACHE MISS — apply tiling or prefetching")
if dram_bound > 20:
    print("  ❌ DRAM BOUND — data doesn't fit in cache hierarchy")
if branch_miss > 5:
    print("  ❌ BRANCH MISPREDICTION — use branchless code")
if ipc > 2 and cache_miss < 1 and branch_miss < 1:
    print("  ✅ Well optimized — near peak hardware utilization")
EOF
```

---

## 11. Bottleneck → Fix Mapping
```
perf Symptom                          Root Cause           Fix
────────────────────────────────────────────────────────────────────────
IPC < 0.5                             Memory stalls        Cache tiling
cache-miss > 20%                      Cache thrashing      Tiling / prefetch
LLC-load-miss > 5%                    DRAM access          Reduce working set
toplev: DRAM_Bound > 20%              Bandwidth saturated  SoA layout / tiling
toplev: L1_Bound > 20%               L1 pressure          Smaller tile / reuse
branch-miss > 5%                      Misprediction        Branchless / sort data
dTLB-load-misses > 1%                TLB thrashing        Hugepages / locality
cpu-migrations > 0                    Thread migration     taskset / numactl
perf c2c: false sharing detected      Cache line conflict  Pad struct to 64 bytes
annotate: single line > 30%          Hot instruction       Unroll / vectorize
FE bound > 30% (toplev)              Icache / decode      Reduce code size
```

---

## 12. Verification Checklist

### Setup
- [ ] `perf --version` works without sudo
- [ ] `perf_event_paranoid` set to -1
- [ ] CPU governor set to `performance`
- [ ] ASLR disabled for reproducibility: `echo 0 > /proc/sys/kernel/randomize_va_space`
- [ ] Binary compiled with `-g` for annotation

### Level 1 — perf stat
- [ ] IPC recorded (target > 2.0)
- [ ] Cache miss rate recorded (target < 1%)
- [ ] Branch miss rate recorded (target < 1%)
- [ ] LLC miss rate recorded (target < 5%)
- [ ] Run 3+ times, variance < 2%

### Level 2 — perf record
- [ ] Top 10 hot functions identified
- [ ] Hottest function takes < 30% (if > 50% it's the clear target)
- [ ] Call graph shows full stack (not just leaf)

### Level 3 — perf annotate
- [ ] Hot lines identified inside hottest function
- [ ] Assembly instruction causing stall identified
- [ ] Root cause mapped (stride access, branch, etc.)

### Level 4 — perf mem
- [ ] Average memory latency per symbol measured
- [ ] L3 miss vs L1 hit ratio documented
- [ ] False sharing checked with `perf c2c`

### Level 5 — toplev
- [ ] Dominant top-down category identified
- [ ] Drilled to level 3 or 4
- [ ] Concrete fix from toplev advice applied

### After Fix
- [ ] Same `perf stat` re-run on optimized binary
- [ ] IPC improved
- [ ] Cache miss rate improved
- [ ] Speedup ratio calculated and documented
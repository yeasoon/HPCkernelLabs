# ncu

## Basic NCU profile
```bash
ncu --set full -o baseline.ncu-rep ./your_kernel

# Or targeted metrics
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__throughput.avg.pct_of_peak_sustained_active,\
  lts__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./your_kernel
```

**Key questions from the roofline:**
- Is your kernel **compute-bound** or **memory-bound**?
- What is your **arithmetic intensity** (FLOPs / bytes)?

---

## Phase 2: NCU Log Analysis — What to Look At

### 2.1 Top-Level Summary Metrics

| NCU Metric | What it Tells You | Red Flag Threshold |
|---|---|---|
| `sm__throughput` | % of SM compute used | < 60% = underutilized |
| `dram__throughput` | % of DRAM BW used | < 50% = memory underutilized |
| `l1tex__throughput` | L1 cache activity | High = good locality |
| `lts__throughput` | L2 cache activity | High DRAM + low L2 = no reuse |
| `gpu__time_duration` | Kernel wall time | Your baseline |

### 2.2 Occupancy Section
```
# Key NCU occupancy metrics
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  launch__occupancy_limit_registers,\
  launch__occupancy_limit_shared_mem,\
  launch__occupancy_limit_warps \
  ./your_kernel

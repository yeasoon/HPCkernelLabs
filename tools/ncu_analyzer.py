"""
NCU (Nsight Compute) Profile Wrapper
=====================================
Step-by-step CUDA GPU kernel performance analysis covering:
  - Cache Pressure
  - Shared Memory Bank Conflicts
  - Register Pressure
  - Instruction Utilization
  - Memory Throughput
  - Warp Efficiency / Occupancy
  - Compute Throughput

Usage:
    python ncu_profile_wrapper.py --binary ./my_cuda_app [args...]
    python ncu_profile_wrapper.py --csv existing_report.csv
    python ncu_profile_wrapper.py --ncu-report existing_report.ncu-rep
"""

import subprocess
import sys
import os
import csv
import json
import argparse
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. NCU METRIC GROUPS
#    Each group maps a concern → list of ncu metric names
# ─────────────────────────────────────────────────────────────────────────────

METRIC_GROUPS = {
    # ── Cache Pressure ────────────────────────────────────────────────────────
    "cache_pressure": [
        "l1tex__t_sector_hit_rate.pct",                     # L1 hit rate
        "lts__t_sector_hit_rate.pct",                       # L2 hit rate
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
        "lts__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "lts__t_requests_srcunit_tex_op_read.sum",
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    ],

    # ── Shared Memory Bank Conflicts ─────────────────────────────────────────
    "bank_conflicts": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    ],

    # ── Register Pressure ─────────────────────────────────────────────────────
    "register_pressure": [
        "launch__registers_per_thread",
        "launch__occupancy_limit_registers",
        "sm__maximum_warps_per_active_cycle_pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_eligible.avg.pct_of_peak_sustained_active",
        "launch__shared_mem_per_block_static",
        "launch__shared_mem_per_block_dynamic",
        "launch__thread_count",
        "launch__block_size",
        "launch__grid_size",
        "sm__warps_active.avg.per_cycle_active",
    ],

    # ── Instruction Utilization ───────────────────────────────────────────────
    "instruction_utilization": [
        "sm__inst_executed.sum",
        "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active",
        "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_op_global_ld.sum",
        "smsp__inst_executed_op_global_st.sum",
        "smsp__inst_executed_op_shared_ld.sum",
        "smsp__inst_executed_op_shared_st.sum",
    ],

    # ── Memory Throughput ─────────────────────────────────────────────────────
    "memory_throughput": [
        "l1tex__t_bytes.sum",
        "lts__t_bytes.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    ],

    # ── Warp Efficiency & Occupancy ───────────────────────────────────────────
    "warp_occupancy": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__maximum_warps_per_active_cycle_pct",
        "sm__warps_eligible.avg.pct_of_peak_sustained_active",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_membar_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct",
        "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    ],

    # ── Compute Throughput ────────────────────────────────────────────────────
    "compute_throughput": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__cycles_elapsed.sum",
        "sm__cycles_active.sum",
        "sm__active_cycles.sum",
    ],
}

ALL_METRICS = list({m for group in METRIC_GROUPS.values() for m in group})


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KernelMetrics:
    name: str
    metrics: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.metrics.get(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(str(self.metrics.get(key, default)).replace(",", ""))
        except (ValueError, TypeError):
            return default


@dataclass
class AnalysisResult:
    section: str
    status: str           # "GOOD", "WARNING", "CRITICAL"
    headline: str
    details: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 3. NCU RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class NcuRunner:
    """
    Launch ncu with the full metric set and return path to CSV output.

    Supports:
      - Explicit ncu binary path  (--ncu-path /usr/local/cuda-12.1/bin/ncu)
      - sudo prefix               (--sudo)
      - CUDA_VISIBLE_DEVICES      (--cuda-device 0)
      - Automatic fallback search when no explicit path is given
    """

    # Candidate names tried when --ncu-path is not specified
    NCU_SEARCH_PATHS = [
        # Explicit CUDA versioned installs (newest first)
        "/usr/local/cuda-12.4/bin/ncu",
        "/usr/local/cuda-12.3/bin/ncu",
        "/usr/local/cuda-12.2/bin/ncu",
        "/usr/local/cuda-12.1/bin/ncu",
        "/usr/local/cuda-12.0/bin/ncu",
        "/usr/local/cuda-11.8/bin/ncu",
        "/usr/local/cuda-11.7/bin/ncu",
        "/usr/local/cuda/bin/ncu",          # symlinked default
        # Legacy CLI name
        "/usr/local/cuda/bin/nv-nsight-cu-cli",
    ]
    NCU_PATH_NAMES = ["ncu", "nv-nsight-cu-cli"]  # tried via PATH lookup

    def __init__(
        self,
        binary: str,
        binary_args: list[str],
        output_csv: str,
        ncu_path: Optional[str] = None,
        use_sudo: bool = False,
        cuda_device: Optional[str] = None,
    ):
        self.binary      = binary
        self.binary_args = binary_args
        self.output_csv  = output_csv
        self.ncu_path    = ncu_path        # explicit override, e.g. /usr/local/cuda-12.1/bin/ncu
        self.use_sudo    = use_sudo        # prepend sudo
        self.cuda_device = cuda_device     # value for CUDA_VISIBLE_DEVICES

    def _resolve_ncu(self) -> str:
        """Return the ncu executable to use, or raise with a helpful message."""

        # 1. User supplied an explicit path — trust it directly.
        if self.ncu_path:
            if not os.path.isfile(self.ncu_path):
                raise RuntimeError(
                    f"Specified --ncu-path '{self.ncu_path}' does not exist."
                )
            if not os.access(self.ncu_path, os.X_OK):
                raise RuntimeError(
                    f"'{self.ncu_path}' is not executable. "
                    f"Try: chmod +x {self.ncu_path}"
                )
            return self.ncu_path

        # 2. Search well-known absolute paths.
        for candidate in self.NCU_SEARCH_PATHS:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                print(f"[NCU] Found ncu at: {candidate}")
                return candidate

        # 3. Fall back to PATH lookup.
        for name in self.NCU_PATH_NAMES:
            result = subprocess.run(
                ["which", name], capture_output=True, text=True
            )
            if result.returncode == 0:
                found = result.stdout.strip()
                print(f"[NCU] Found ncu via PATH: {found}")
                return found

        raise RuntimeError(
            "ncu (Nsight Compute CLI) not found.\n"
            "  • Install CUDA Toolkit >= 11.0, or\n"
            "  • Pass --ncu-path /usr/local/cuda-<ver>/bin/ncu explicitly."
        )

    def _build_env(self) -> dict:
        """Build the environment dict, injecting CUDA_VISIBLE_DEVICES if set."""
        env = os.environ.copy()
        if self.cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
            print(f"[NCU] CUDA_VISIBLE_DEVICES={self.cuda_device}")
        return env

    # ── internal helpers ──────────────────────────────────────────────────────

    def _wrap_sudo(self, cmd: list[str]) -> list[str]:
        """Prepend 'sudo -E' when requested."""
        return (["sudo", "-E"] + cmd) if self.use_sudo else cmd

    def _run_subprocess(
        self, cmd: list[str], env: dict, label: str
    ) -> tuple[int, str, str]:
        """Run *cmd*, stream stderr lines live, return (returncode, stdout, stderr)."""
        print(f"\n[NCU] {label}:\n  {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        # Print stderr lines as they arrive so the user sees ncu progress live.
        import threading

        def drain_stderr():
            for line in proc.stderr:
                stderr_chunks.append(line)
                print(f"  [ncu stderr] {line}", end="")

        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()
        stdout_data = proc.stdout.read()
        proc.stdout.close()
        t.join()
        proc.wait()
        return proc.returncode, stdout_data, "".join(stderr_chunks)

    def _probe_ncu_version(self, ncu: str, env: dict):
        """Print ncu version so it's visible in logs."""
        cmd = self._wrap_sudo([ncu, "--version"])
        rc, out, err = self._run_subprocess(cmd, env, "Probing ncu version")
        print(out.strip() or err.strip())

    def _query_available_metrics(self, ncu: str, env: dict) -> set[str]:
        """
        Ask ncu which metrics actually exist on this device/toolkit.
        Returns a set of valid metric names.
        """
        cmd = self._wrap_sudo([ncu, "--query-metrics", "--csv"])
        rc, out, err = self._run_subprocess(
            cmd, env, "Querying available metrics"
        )
        available: set[str] = set()
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("==") or line.startswith('"Metric'):
                continue
            # CSV columns: "Metric Name","Description" — grab first field
            first = line.split(",")[0].strip().strip('"')
            if first:
                available.add(first)
        print(f"  [ncu] {len(available)} metrics available on this device.")
        return available

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> str:
        ncu = self._resolve_ncu()
        env = self._build_env()

        # ── Step 1: version banner ────────────────────────────────────────────
        self._probe_ncu_version(ncu, env)

        # ── Step 2: filter requested metrics to those the device supports ─────
        print("\n[NCU] Checking which requested metrics are available …")
        available = self._query_available_metrics(ncu, env)

        if available:
            valid   = [m for m in ALL_METRICS if m in available]
            dropped = [m for m in ALL_METRICS if m not in available]
            if dropped:
                print(f"  [ncu] Dropping {len(dropped)} unsupported metric(s):")
                for m in dropped:
                    print(f"        ✗ {m}")
            print(f"  [ncu] Collecting {len(valid)} supported metric(s).")
        else:
            # --query-metrics failed silently (some toolkit builds omit it);
            # fall back to the full list and let ncu report errors itself.
            print("  [ncu] Could not query metrics — using full list.")
            valid = list(ALL_METRICS)

        if not valid:
            raise RuntimeError(
                "No requested metrics are supported by this device/toolkit.\n"
                "Run with --list-metrics to see what was requested, then check\n"
                f"  {ncu} --query-metrics --csv"
            )

        metrics_str = ",".join(valid)

        # ── Step 3: build the profile command ─────────────────────────────────
        #
        # Flag notes:
        #   --csv                   → write CSV rows to stdout (NOT a .ncu-rep)
        #   --target-processes all  → profile child processes too
        #   --kernel-regex-base mangled  → safer default; 'demangled' requires
        #                                  the C++ demangler to be available and
        #                                  can silently match nothing on some
        #                                  toolkit builds
        #
        # We do NOT use --log-file: that writes a binary .ncu-rep file.
        # When sudo is involved, that file would be owned by root.
        # Instead Python (running as the user) captures stdout and writes
        # the CSV file itself.
        ncu_cmd = [
            ncu,
            "--metrics",        metrics_str,
            "--csv",
            "--target-processes", "all",
            "--kernel-regex-base", "mangled",   # reliable across toolkit versions
            self.binary,
        ] + self.binary_args

        cmd = self._wrap_sudo(ncu_cmd)

        # ── Step 4: run and capture ───────────────────────────────────────────
        rc, stdout_data, stderr_data = self._run_subprocess(
            cmd, env, "Profiling kernel(s)"
        )

        # Dump the raw stdout so the user can see what ncu actually produced
        print(f"\n[NCU] Raw stdout ({len(stdout_data)} chars):")
        preview = stdout_data[:2000]
        print(preview)
        if len(stdout_data) > 2000:
            print(f"  … ({len(stdout_data) - 2000} more chars) …")

        if rc != 0:
            raise RuntimeError(
                f"ncu exited with code {rc}.\n"
                f"Check [ncu stderr] lines above for details.\n"
                f"Common causes:\n"
                f"  ERR_NVGPUCTRPERM  → add --sudo, or set:\n"
                f"    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0'"
                f" | sudo tee /etc/modprobe.d/nvidia-profiling.conf\n"
                f"  No kernels found  → check --binary path and that the app "
                f"actually launches CUDA kernels"
            )

        # ── Step 5: strip ncu banner lines and check we got real CSV ──────────
        csv_lines = [
            l for l in stdout_data.splitlines(keepends=True)
            if not l.startswith("==")
        ]

        # Detect if ncu wrote nothing useful (happens when no kernel matched)
        non_empty = [l for l in csv_lines if l.strip()]
        if not non_empty:
            raise RuntimeError(
                "ncu ran successfully but produced no CSV rows.\n"
                "Possible reasons:\n"
                "  • No CUDA kernels were launched by the binary\n"
                "  • --kernel-regex-base filtered out all kernels\n"
                "  • The binary exited before any kernel ran\n"
                "Try running ncu manually to confirm:\n"
                f"  {' '.join(cmd)}"
            )

        # ── Step 6: write CSV as the calling user (no root ownership issue) ───
        with open(self.output_csv, "w", encoding="utf-8") as f:
            f.writelines(csv_lines)

        print(f"\n[NCU] CSV written → {self.output_csv}  "
              f"({len(non_empty)} data lines)")
        return self.output_csv


# ─────────────────────────────────────────────────────────────────────────────
# 4. CSV PARSER
# ─────────────────────────────────────────────────────────────────────────────

class NcuCsvParser:
    """
    Parse a ncu --csv stdout-captured file into KernelMetrics objects.

    ncu CSV layout (one row per metric per kernel invocation):
        "ID","Process ID","Process Name","Host Name","Kernel Name",
        "Kernel Time","Context","Stream","Section Name",
        "Metric Name","Metric Unit","Metric Value"
    The == banner lines are already stripped before the file is written.
    """

    def parse(self, csv_path: str) -> list[KernelMetrics]:
        kernels: dict[str, KernelMetrics] = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            raw = f.read()

        # Belt-and-suspenders: drop any residual == banner lines
        lines = [l for l in raw.splitlines(keepends=True)
                 if not l.startswith("==")]

        reader = csv.DictReader(lines)
        for row in reader:
            # Column names differ slightly across ncu versions — try all known forms
            kernel_name = (
                row.get("Kernel Name") or
                row.get("kernel_name") or
                row.get("Name") or
                row.get("ID", "unknown")
            ).strip()

            metric_name = (
                row.get("Metric Name") or
                row.get("metric_name") or ""
            ).strip()

            metric_value = (
                row.get("Metric Value") or
                row.get("metric_value") or ""
            ).strip()

            if not metric_name:
                continue

            if kernel_name not in kernels:
                kernels[kernel_name] = KernelMetrics(name=kernel_name)

            # Last value wins if the same metric appears multiple times
            # (ncu emits one row per pass; we keep the final pass value)
            kernels[kernel_name].metrics[metric_name] = metric_value

        return list(kernels.values())


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class KernelAnalyzer:
    """
    Analyses a KernelMetrics object step by step across all concern areas.
    """

    THRESHOLDS = {
        "l1_hit_rate_warn":   60.0,
        "l1_hit_rate_good":   85.0,
        "l2_hit_rate_warn":   60.0,
        "l2_hit_rate_good":   85.0,
        "bank_conflict_warn":  0.05,   # fraction: conflicts / total accesses
        "regs_per_thread_warn": 64,
        "regs_per_thread_crit": 128,
        "occupancy_warn":     50.0,
        "occupancy_crit":     25.0,
        "pipe_util_warn":     50.0,
        "dram_bw_warn":       60.0,
        "stall_warn":         15.0,    # % stall per warp active
    }

    # ── 5.1  Cache Pressure ───────────────────────────────────────────────────
    def analyse_cache_pressure(self, km: KernelMetrics) -> AnalysisResult:
        l1_hit = km.get_float("l1tex__t_sector_hit_rate.pct")
        l2_hit = km.get_float("lts__t_sector_hit_rate.pct")

        status = "GOOD"
        details = [
            f"L1 Cache Hit Rate : {l1_hit:.1f}%",
            f"L2 Cache Hit Rate : {l2_hit:.1f}%",
        ]
        recs = []

        if l1_hit < self.THRESHOLDS["l1_hit_rate_warn"]:
            status = "CRITICAL"
            recs += [
                "Increase data locality — reuse loaded data within a warp.",
                "Consider tiling / blocking to fit working set in L1.",
                "Avoid non-sequential (strided / random) global memory access.",
            ]
        elif l1_hit < self.THRESHOLDS["l1_hit_rate_good"]:
            status = "WARNING"
            recs.append("Moderate L1 miss rate — consider loop tiling.")

        if l2_hit < self.THRESHOLDS["l2_hit_rate_warn"]:
            status = max(status, "WARNING", key=["GOOD","WARNING","CRITICAL"].index)
            recs += [
                "High DRAM pressure — data working set exceeds L2 capacity.",
                "Prefetch or software-pipeline global loads.",
                "Compress or quantise data if precision allows.",
            ]

        l1_sectors_per_req = km.get_float(
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
        )
        if l1_sectors_per_req > 2.0:
            details.append(
                f"L1 sectors/request  : {l1_sectors_per_req:.2f}  "
                f"(ideal ≤ 1 for 128-B transactions)"
            )
            recs.append(
                "High sectors-per-request indicates uncoalesced access → align "
                "data to 128-byte boundaries and access consecutively within a warp."
            )

        headline = (
            f"L1={l1_hit:.0f}% L2={l2_hit:.0f}%  "
            f"({'pressure detected' if status != 'GOOD' else 'healthy'})"
        )
        return AnalysisResult("Cache Pressure", status, headline, details, recs)

    # ── 5.2  Bank Conflicts ────────────────────────────────────────────────────
    def analyse_bank_conflicts(self, km: KernelMetrics) -> AnalysisResult:
        ld_conflicts = km.get_float(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
        )
        st_conflicts = km.get_float(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
        )
        ld_wavefronts = km.get_float(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
        ) or 1
        st_wavefronts = km.get_float(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
        ) or 1

        ld_conflict_ratio = ld_conflicts / ld_wavefronts
        st_conflict_ratio = st_conflicts / st_wavefronts

        details = [
            f"Shared Mem Load  Conflicts : {ld_conflicts:.0f}  "
            f"({ld_conflict_ratio*100:.1f}% of wavefronts)",
            f"Shared Mem Store Conflicts : {st_conflicts:.0f}  "
            f"({st_conflict_ratio*100:.1f}% of wavefronts)",
        ]
        recs = []

        worst = max(ld_conflict_ratio, st_conflict_ratio)
        if worst > self.THRESHOLDS["bank_conflict_warn"]:
            status = "CRITICAL" if worst > 0.25 else "WARNING"
            recs += [
                "Add padding to shared memory arrays: "
                "e.g., `__shared__ float tile[32][33]` (one extra column).",
                "Rearrange data layout so consecutive threads access different banks.",
                "Use shuffle instructions (__shfl_*) instead of shared memory "
                "for intra-warp communication where possible.",
            ]
        else:
            status = "GOOD"

        headline = (
            f"Load conflicts={ld_conflicts:.0f}  Store conflicts={st_conflicts:.0f}"
        )
        return AnalysisResult("Bank Conflicts", status, headline, details, recs)

    # ── 5.3  Register Pressure ────────────────────────────────────────────────
    def analyse_register_pressure(self, km: KernelMetrics) -> AnalysisResult:
        regs = km.get_float("launch__registers_per_thread")
        occ_limit_regs = km.get("launch__occupancy_limit_registers", "N/A")
        active_warp_pct = km.get_float(
            "sm__warps_active.avg.pct_of_peak_sustained_active"
        )
        shared_static = km.get_float("launch__shared_mem_per_block_static") / 1024
        shared_dynamic = km.get_float("launch__shared_mem_per_block_dynamic") / 1024
        block_size = km.get_float("launch__block_size") or km.get_float("launch__thread_count")

        details = [
            f"Registers per Thread       : {regs:.0f}",
            f"Occupancy Limited by Regs  : {occ_limit_regs}",
            f"Active Warp %              : {active_warp_pct:.1f}%",
            f"Shared Mem per Block       : {shared_static:.1f} KB static "
            f"+ {shared_dynamic:.1f} KB dynamic",
            f"Block Size                 : {block_size:.0f} threads",
        ]
        recs = []

        if regs >= self.THRESHOLDS["regs_per_thread_crit"]:
            status = "CRITICAL"
            recs += [
                f"Very high register count ({regs:.0f}/thread) severely limits occupancy.",
                "Use `__launch_bounds__(max_threads, min_blocks)` to cap registers.",
                "Compile with `-maxrregcount=<N>` (e.g., 64) — may spill to local mem.",
                "Refactor kernel: split into multiple passes / reduce live variables.",
            ]
        elif regs >= self.THRESHOLDS["regs_per_thread_warn"]:
            status = "WARNING"
            recs += [
                f"Elevated registers ({regs:.0f}/thread).",
                "Consider `__launch_bounds__` or `-maxrregcount`.",
            ]
        else:
            status = "GOOD"

        if active_warp_pct < self.THRESHOLDS["occupancy_crit"]:
            status = "CRITICAL"
            recs.append(
                f"Critically low warp occupancy ({active_warp_pct:.0f}%) — "
                "GPU under-utilised; check block size and resource limits."
            )
        elif active_warp_pct < self.THRESHOLDS["occupancy_warn"]:
            if status == "GOOD":
                status = "WARNING"
            recs.append(
                f"Low warp occupancy ({active_warp_pct:.0f}%) — "
                "try larger block sizes or reduce per-thread resource usage."
            )

        headline = (
            f"Regs/thread={regs:.0f}  Active warps={active_warp_pct:.0f}%  "
            f"SharedMem={shared_static+shared_dynamic:.1f} KB/block"
        )
        return AnalysisResult("Register Pressure", status, headline, details, recs)

    # ── 5.4  Instruction Utilization ──────────────────────────────────────────
    def analyse_instruction_utilization(self, km: KernelMetrics) -> AnalysisResult:
        pipe_map = {
            "ALU"     : "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
            "FMA"     : "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
            "FP16"    : "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active",
            "FP64"    : "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
            "LSU"     : "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
            "TEX"     : "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active",
            "Uniform" : "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active",
        }
        utils = {name: km.get_float(metric) for name, metric in pipe_map.items()}

        gld = km.get_float("smsp__inst_executed_op_global_ld.sum")
        gst = km.get_float("smsp__inst_executed_op_global_st.sum")
        sld = km.get_float("smsp__inst_executed_op_shared_ld.sum")
        sst = km.get_float("smsp__inst_executed_op_shared_st.sum")

        details = [f"{name:8s} pipe util: {pct:.1f}%" for name, pct in utils.items()]
        details += [
            f"Global loads  : {gld:.0f}  stores: {gst:.0f}",
            f"Shared loads  : {sld:.0f}  stores: {sst:.0f}",
        ]

        recs = []
        dominant = max(utils, key=utils.get)
        dominant_pct = utils[dominant]

        if dominant_pct < self.THRESHOLDS["pipe_util_warn"]:
            status = "WARNING"
            recs += [
                f"Peak pipe utilization is only {dominant_pct:.0f}% ({dominant}).",
                "Kernel is likely memory-bound or stall-bound, not compute-bound.",
                "Profile stall reasons (warp_occupancy section) before optimising compute.",
            ]
        else:
            status = "GOOD"

        if utils["FP64"] > 5.0:
            recs.append(
                "Significant FP64 usage detected — use FP32 or FP16 where precision "
                "allows (FP64 is 2–32× slower on consumer/gaming GPUs)."
            )
        if utils["LSU"] > utils.get("FMA", 0) and utils["LSU"] > 30:
            recs.append(
                "LSU (load/store) pipe dominates — kernel is memory-bound. "
                "Prioritise memory access pattern optimisation over arithmetic tuning."
            )

        headline = (
            f"Dominant pipe: {dominant} @ {dominant_pct:.0f}%  "
            f"| FMA={utils['FMA']:.0f}%  LSU={utils['LSU']:.0f}%"
        )
        return AnalysisResult(
            "Instruction Utilization", status, headline, details, recs
        )

    # ── 5.5  Memory Throughput ────────────────────────────────────────────────
    def analyse_memory_throughput(self, km: KernelMetrics) -> AnalysisResult:
        dram_pct = km.get_float(
            "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        )
        l1_pct = km.get_float(
            "l1tex__throughput.avg.pct_of_peak_sustained_active"
        )
        l2_pct = km.get_float(
            "lts__throughput.avg.pct_of_peak_sustained_elapsed"
        )
        dram_read  = km.get_float("dram__bytes_read.sum")  / 1e6
        dram_write = km.get_float("dram__bytes_write.sum") / 1e6

        details = [
            f"DRAM Throughput  : {dram_pct:.1f}% of peak",
            f"L1   Throughput  : {l1_pct:.1f}% of peak",
            f"L2   Throughput  : {l2_pct:.1f}% of peak",
            f"DRAM Bytes Read  : {dram_read:.1f} MB",
            f"DRAM Bytes Write : {dram_write:.1f} MB",
        ]
        recs = []

        if dram_pct > self.THRESHOLDS["dram_bw_warn"]:
            status = "WARNING"
            recs += [
                f"DRAM is at {dram_pct:.0f}% utilisation — you are hitting DRAM bandwidth.",
                "Ensure memory accesses are coalesced (128-byte aligned, sequential).",
                "Use shared memory to stage data and reduce DRAM traffic.",
                "Consider reducing data precision (FP16 / BF16 / INT8) for DRAM savings.",
                "Explore read-only __ldg() cache for broadcast/read-only data.",
            ]
        else:
            status = "GOOD"

        if dram_pct < 20.0 and l1_pct < 20.0 and l2_pct < 20.0:
            recs.append(
                "All memory subsystems show low utilisation — "
                "kernel may be compute-bound or heavily stall-bound."
            )

        headline = (
            f"DRAM={dram_pct:.0f}%  L2={l2_pct:.0f}%  L1={l1_pct:.0f}%  "
            f"({dram_read+dram_write:.0f} MB total DRAM traffic)"
        )
        return AnalysisResult("Memory Throughput", status, headline, details, recs)

    # ── 5.6  Warp Stalls & Occupancy ─────────────────────────────────────────
    def analyse_warp_occupancy(self, km: KernelMetrics) -> AnalysisResult:
        stall_map = {
            "Long Scoreboard (mem dep)" : "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "Short Scoreboard"          : "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "Barrier"                   : "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "Memory Bar"                : "smsp__warp_issue_stalled_membar_per_warp_active.pct",
            "Wait"                      : "smsp__warp_issue_stalled_wait_per_warp_active.pct",
            "Texture Throttle"          : "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
            "Drain"                     : "smsp__warp_issue_stalled_drain_per_warp_active.pct",
            "IMC Miss"                  : "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct",
        }
        stalls = {name: km.get_float(m) for name, m in stall_map.items()}
        occupancy = km.get_float("sm__warps_active.avg.pct_of_peak_sustained_active")

        details = [
            f"Theoretical Occupancy : {occupancy:.1f}%",
        ] + [f"  Stall [{name:28s}]: {pct:.2f}%" for name, pct in stalls.items()]

        recs = []
        dominant_stall = max(stalls, key=stalls.get)
        dominant_pct   = stalls[dominant_stall]

        status = "GOOD"
        if dominant_pct > self.THRESHOLDS["stall_warn"]:
            status = "WARNING" if dominant_pct < 30 else "CRITICAL"
            stall_advice = {
                "Long Scoreboard (mem dep)": (
                    "Global memory latency stalls. Increase ILP: interleave independent "
                    "loads, use prefetching, or pipeline software."
                ),
                "Short Scoreboard": (
                    "Dependency on fast (register/shared) ops. Reorder independent "
                    "instructions to fill the gap."
                ),
                "Barrier": (
                    "__syncthreads() bottleneck. Minimise barriers or restructure "
                    "algorithm to reduce synchronisation points."
                ),
                "Memory Bar":  "Memory fence stalls — reduce fence scope (e.g., __threadfence_block).",
                "Wait":        "Warp-level instruction latency. Increase occupancy to hide latency.",
                "Texture Throttle": "Texture unit saturated — reduce texture fetches or use L1.",
                "Drain":       "Divergent warp serialisation. Reduce branch divergence.",
                "IMC Miss":    "Instruction cache miss — large kernel; consider splitting it.",
            }
            recs.append(
                f"Top stall reason: '{dominant_stall}' at {dominant_pct:.1f}%."
            )
            recs.append(stall_advice.get(dominant_stall, "Investigate stall further."))

        if occupancy < self.THRESHOLDS["occupancy_warn"]:
            if status == "GOOD":
                status = "WARNING"
            recs.append(
                f"Low occupancy ({occupancy:.0f}%). "
                "Increase block size or reduce per-thread resource usage."
            )

        headline = (
            f"Occupancy={occupancy:.0f}%  "
            f"Top stall='{dominant_stall}' ({dominant_pct:.1f}%)"
        )
        return AnalysisResult("Warp Stalls & Occupancy", status, headline, details, recs)

    # ── 5.7  Compute Throughput ───────────────────────────────────────────────
    def analyse_compute_throughput(self, km: KernelMetrics) -> AnalysisResult:
        sm_pct = km.get_float(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        )
        gpu_pct = km.get_float(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
        )
        cycles_elapsed = km.get_float("sm__cycles_elapsed.sum")
        cycles_active  = km.get_float("sm__cycles_active.sum") or \
                         km.get_float("sm__active_cycles.sum")
        active_ratio = (cycles_active / cycles_elapsed * 100) if cycles_elapsed else 0.0

        details = [
            f"SM  Throughput  : {sm_pct:.1f}% of peak",
            f"GPU Memory+Comp : {gpu_pct:.1f}% of peak",
            f"SM Active Ratio : {active_ratio:.1f}%",
        ]
        recs = []

        if sm_pct > 80:
            status = "GOOD"
        elif sm_pct > 50:
            status = "WARNING"
            recs.append(
                f"SM throughput is {sm_pct:.0f}% — room to improve. "
                "Examine dominant bottleneck (memory vs stalls vs occupancy)."
            )
        else:
            status = "CRITICAL"
            recs += [
                f"Very low SM throughput ({sm_pct:.0f}%). Kernel is likely:",
                "  • Memory-bound → optimise access patterns & cache use.",
                "  • Stall-bound  → see Warp Stalls section.",
                "  • Occupancy-limited → see Register Pressure section.",
            ]

        headline = f"SM throughput={sm_pct:.0f}%  active_ratio={active_ratio:.0f}%"
        return AnalysisResult("Compute Throughput", status, headline, details, recs)

    # ── Master analyser ───────────────────────────────────────────────────────
    def analyse_all(self, km: KernelMetrics) -> list[AnalysisResult]:
        return [
            self.analyse_cache_pressure(km),
            self.analyse_bank_conflicts(km),
            self.analyse_register_pressure(km),
            self.analyse_instruction_utilization(km),
            self.analyse_memory_throughput(km),
            self.analyse_warp_occupancy(km),
            self.analyse_compute_throughput(km),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

STATUS_ICON = {"GOOD": "✅", "WARNING": "⚠️ ", "CRITICAL": "🔴"}
STATUS_COLOR = {
    "GOOD":     "\033[92m",   # green
    "WARNING":  "\033[93m",   # yellow
    "CRITICAL": "\033[91m",   # red
    "RESET":    "\033[0m",
}


def _colorize(text: str, status: str) -> str:
    c = STATUS_COLOR.get(status, "")
    return f"{c}{text}{STATUS_COLOR['RESET']}"


def print_report(kernels: list[KernelMetrics], save_json: Optional[str] = None):
    analyzer = KernelAnalyzer()
    all_results: dict[str, list[AnalysisResult]] = {}

    sep = "─" * 80

    print("\n" + "═" * 80)
    print("  NCU KERNEL PERFORMANCE ANALYSIS REPORT")
    print("═" * 80)

    for km in kernels:
        print(f"\n{'━'*80}")
        print(f"  KERNEL: {km.name}")
        print(f"{'━'*80}")

        results = analyzer.analyse_all(km)
        all_results[km.name] = results

        for step, res in enumerate(results, 1):
            icon = STATUS_ICON.get(res.status, "?")
            header = f"  Step {step}: {res.section}"
            print(f"\n{sep}")
            print(_colorize(f"{icon} {header}  [{res.status}]", res.status))
            print(f"  {res.headline}")
            print(sep)

            if res.details:
                print("  Details:")
                for d in res.details:
                    print(f"    {d}")

            if res.recommendations:
                print("  Recommendations:")
                for r in res.recommendations:
                    wrapped = textwrap.fill(
                        r, width=76, initial_indent="    • ",
                        subsequent_indent="      "
                    )
                    print(wrapped)

        # Summary table
        print(f"\n{sep}")
        print("  SUMMARY")
        print(sep)
        for res in results:
            icon = STATUS_ICON.get(res.status, "?")
            line = f"  {icon} {res.section:<30s}  {res.status}"
            print(_colorize(line, res.status))

    # Optional JSON export
    if save_json:
        export = {}
        for kname, results in all_results.items():
            export[kname] = [
                {
                    "section": r.section,
                    "status": r.status,
                    "headline": r.headline,
                    "details": r.details,
                    "recommendations": r.recommendations,
                }
                for r in results
            ]
        with open(save_json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\n[INFO] JSON report saved → {save_json}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEMO MODE  (no GPU required)
# ─────────────────────────────────────────────────────────────────────────────

def _demo_kernel(name: str, profile: str) -> KernelMetrics:
    """
    Returns synthetic KernelMetrics for demonstration when no GPU is present.
    profile: "memory_bound" | "register_heavy" | "balanced"
    """
    presets = {
        "memory_bound": {
            "l1tex__t_sector_hit_rate.pct": "28.5",
            "lts__t_sector_hit_rate.pct": "42.1",
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "4.2",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "12000",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "3000",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "40000",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "10000",
            "launch__registers_per_thread": "48",
            "launch__occupancy_limit_registers": "No",
            "sm__warps_active.avg.pct_of_peak_sustained_active": "72.3",
            "sm__maximum_warps_per_active_cycle_pct": "72.0",
            "launch__shared_mem_per_block_static": "16384",
            "launch__shared_mem_per_block_dynamic": "0",
            "launch__block_size": "256",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": "22.1",
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": "18.4",
            "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active": "0.0",
            "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": "0.5",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "68.9",
            "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active": "5.2",
            "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active": "3.1",
            "smsp__inst_executed_op_global_ld.sum": "500000",
            "smsp__inst_executed_op_global_st.sum": "100000",
            "smsp__inst_executed_op_shared_ld.sum": "40000",
            "smsp__inst_executed_op_shared_st.sum": "10000",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": "87.4",
            "l1tex__throughput.avg.pct_of_peak_sustained_active": "65.1",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": "78.3",
            "dram__bytes_read.sum": "524288000",
            "dram__bytes_write.sum": "104857600",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "38.2",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "4.1",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "2.0",
            "smsp__warp_issue_stalled_membar_per_warp_active.pct": "1.5",
            "smsp__warp_issue_stalled_wait_per_warp_active.pct": "5.0",
            "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct": "0.8",
            "smsp__warp_issue_stalled_drain_per_warp_active.pct": "0.3",
            "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct": "0.2",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "31.0",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "87.4",
            "sm__cycles_elapsed.sum": "5000000",
            "sm__active_cycles.sum": "3200000",
        },
        "register_heavy": {
            "l1tex__t_sector_hit_rate.pct": "80.0",
            "lts__t_sector_hit_rate.pct": "88.0",
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "1.1",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "100",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "50",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "50000",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "20000",
            "launch__registers_per_thread": "120",
            "launch__occupancy_limit_registers": "Yes",
            "sm__warps_active.avg.pct_of_peak_sustained_active": "21.5",
            "sm__maximum_warps_per_active_cycle_pct": "21.0",
            "launch__shared_mem_per_block_static": "8192",
            "launch__shared_mem_per_block_dynamic": "4096",
            "launch__block_size": "512",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": "74.3",
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": "80.1",
            "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active": "0.0",
            "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": "0.0",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "18.5",
            "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active": "1.2",
            "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active": "2.0",
            "smsp__inst_executed_op_global_ld.sum": "100000",
            "smsp__inst_executed_op_global_st.sum": "50000",
            "smsp__inst_executed_op_shared_ld.sum": "200000",
            "smsp__inst_executed_op_shared_st.sum": "80000",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": "18.0",
            "l1tex__throughput.avg.pct_of_peak_sustained_active": "22.0",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": "20.0",
            "dram__bytes_read.sum": "10485760",
            "dram__bytes_write.sum": "5242880",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "5.0",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "18.5",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "8.2",
            "smsp__warp_issue_stalled_membar_per_warp_active.pct": "0.5",
            "smsp__warp_issue_stalled_wait_per_warp_active.pct": "3.0",
            "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct": "0.2",
            "smsp__warp_issue_stalled_drain_per_warp_active.pct": "0.1",
            "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct": "0.3",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "45.0",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "45.0",
            "sm__cycles_elapsed.sum": "8000000",
            "sm__active_cycles.sum": "4200000",
        },
        "balanced": {
            "l1tex__t_sector_hit_rate.pct": "91.0",
            "lts__t_sector_hit_rate.pct": "92.5",
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "1.05",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "0",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "0",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "30000",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "10000",
            "launch__registers_per_thread": "40",
            "launch__occupancy_limit_registers": "No",
            "sm__warps_active.avg.pct_of_peak_sustained_active": "88.0",
            "sm__maximum_warps_per_active_cycle_pct": "88.0",
            "launch__shared_mem_per_block_static": "4096",
            "launch__shared_mem_per_block_dynamic": "2048",
            "launch__block_size": "256",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": "71.0",
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": "82.0",
            "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active": "40.0",
            "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": "0.0",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "32.0",
            "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active": "8.0",
            "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active": "4.0",
            "smsp__inst_executed_op_global_ld.sum": "150000",
            "smsp__inst_executed_op_global_st.sum": "60000",
            "smsp__inst_executed_op_shared_ld.sum": "300000",
            "smsp__inst_executed_op_shared_st.sum": "100000",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": "41.0",
            "l1tex__throughput.avg.pct_of_peak_sustained_active": "55.0",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": "48.0",
            "dram__bytes_read.sum": "104857600",
            "dram__bytes_write.sum": "52428800",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "8.0",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "5.0",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "3.5",
            "smsp__warp_issue_stalled_membar_per_warp_active.pct": "1.0",
            "smsp__warp_issue_stalled_wait_per_warp_active.pct": "2.0",
            "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct": "0.5",
            "smsp__warp_issue_stalled_drain_per_warp_active.pct": "0.2",
            "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct": "0.1",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "82.0",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "82.0",
            "sm__cycles_elapsed.sum": "3000000",
            "sm__active_cycles.sum": "2700000",
        },
    }
    km = KernelMetrics(name=name)
    km.metrics = presets.get(profile, presets["balanced"])
    return km


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NCU Profile Wrapper — step-by-step CUDA kernel analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Typical invocation matching: CUDA_VISIBLE_DEVICES=0 sudo ncu ...
          python ncu_profile_wrapper.py \\
              --ncu-path /usr/local/cuda-12.1/bin/ncu \\
              --cuda-device 0 --sudo \\
              --binary ./matmul 1024

          # Explicit path + sudo, default GPU:
          python ncu_profile_wrapper.py \\
              --ncu-path /usr/local/cuda-12.1/bin/ncu --sudo \\
              --binary ./my_kernel

          # Auto-detect ncu, no sudo, GPU 1:
          python ncu_profile_wrapper.py --cuda-device 1 --binary ./my_kernel

          # Analyse an existing ncu CSV export:
          python ncu_profile_wrapper.py --csv my_profile.csv

          # Run demo (no GPU needed):
          python ncu_profile_wrapper.py --demo

          # Save analysis to JSON:
          python ncu_profile_wrapper.py --demo --save-json analysis.json

          # List all ~50 metrics that are collected:
          python ncu_profile_wrapper.py --list-metrics

        Permission note:
          If ncu reports ERR_NVGPUCTRPERM, use --sudo, or disable the restriction
          permanently (no reboot needed after modprobe reload):
            echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \\
                sudo tee /etc/modprobe.d/nvidia-profiling.conf
            sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm
        """),
    )

    # ── Input source (mutually exclusive) ─────────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument("--binary", metavar="PATH",
                     help="CUDA binary to profile with ncu")
    src.add_argument("--csv",    metavar="PATH",
                     help="Existing ncu --csv output file to analyse")
    src.add_argument("--demo",   action="store_true",
                     help="Run with synthetic demo kernels (no GPU required)")

    p.add_argument("binary_args", nargs=argparse.REMAINDER,
                   help="Arguments forwarded to the profiled binary")

    # ── ncu invocation options ─────────────────────────────────────────────────
    ncu_grp = p.add_argument_group("ncu invocation")
    ncu_grp.add_argument(
        "--ncu-path", metavar="PATH",
        default=None,
        help=(
            "Absolute path to the ncu binary "
            "(default: auto-detect from versioned CUDA dirs then PATH). "
            "Example: /usr/local/cuda-12.1/bin/ncu"
        ),
    )
    ncu_grp.add_argument(
        "--sudo", dest="use_sudo", action="store_true",
        default=False,
        help=(
            "Prefix the ncu command with 'sudo -E'. "
            "Required on most systems for hardware performance counters."
        ),
    )
    ncu_grp.add_argument(
        "--cuda-device", metavar="ID",
        default=None,
        help=(
            "Value to set for CUDA_VISIBLE_DEVICES before launching ncu. "
            "Example: 0  (targets the first GPU). "
            "Omit to use whatever device the binary selects."
        ),
    )

    # ── Output options ─────────────────────────────────────────────────────────
    out_grp = p.add_argument_group("output")
    out_grp.add_argument("--output-csv", default="ncu_output.csv",
                         metavar="PATH",
                         help="Where to write the raw ncu CSV (default: ncu_output.csv)")
    out_grp.add_argument("--save-json",  metavar="PATH",
                         help="Also save the analysis report as JSON to this path")
    out_grp.add_argument("--list-metrics", action="store_true",
                         help="Print all collected metric names grouped by concern and exit")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_metrics:
        print("\nAll collected NCU metrics:")
        for group, metrics in METRIC_GROUPS.items():
            print(f"\n[{group}]")
            for m in metrics:
                print(f"  {m}")
        return

    kernels: list[KernelMetrics] = []

    if args.demo:
        print("\n[DEMO MODE] Using synthetic kernel profiles (no GPU required)\n")
        kernels = [
            _demo_kernel("matmul_shared_v1<<<...>>>",     "memory_bound"),
            _demo_kernel("attention_fwd_kernel<<<...>>>", "register_heavy"),
            _demo_kernel("conv2d_tiled_v3<<<...>>>",      "balanced"),
        ]

    elif args.binary:
        runner = NcuRunner(
            binary=args.binary,
            binary_args=args.binary_args,
            output_csv=args.output_csv,
            ncu_path=args.ncu_path,
            use_sudo=args.use_sudo,
            cuda_device=args.cuda_device,
        )
        csv_path = runner.run()
        kernels = NcuCsvParser().parse(csv_path)

    elif args.csv:
        print(f"[INFO] Parsing existing CSV: {args.csv}")
        kernels = NcuCsvParser().parse(args.csv)

    else:
        parser.print_help()
        sys.exit(1)

    if not kernels:
        print("[WARN] No kernels found in profile output.")
        sys.exit(0)

    print_report(kernels, save_json=args.save_json)


if __name__ == "__main__":
    main()
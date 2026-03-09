#!/usr/bin/env python3
"""
cpu_perf_analyzer.py
--------------------
Run 'perf stat' on a CPU kernel binary, parse the output,
display a rich dashboard, and print actionable optimisation advice.

Usage:
  python3 cpu_perf_analyzer.py ./my_kernel [args...]
  python3 cpu_perf_analyzer.py --paste          # paste raw perf output manually
  python3 cpu_perf_analyzer.py --file perf.txt  # read from saved perf output

Requirements:
  pip install rich
  Linux with perf installed  (sudo apt install linux-tools-common)
"""

import re
import sys
import subprocess
import argparse
import shutil
from dataclasses import dataclass, field
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.rule import Rule
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.align import Align
except ImportError:
    print("Missing dependency. Run:  pip install rich")
    sys.exit(1)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# PERF EVENTS we request
# ─────────────────────────────────────────────────────────────────────────────
PERF_EVENTS = ",".join([
    "cycles",
    "instructions",
    "cache-misses",
    "cache-references",
    "branch-misses",
    "branches",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "L1-icache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
    "dTLB-load-misses",
    "iTLB-load-misses",
    "cpu-migrations",
    "context-switches",
    "task-clock",
])

# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PerfData:
    cycles:               Optional[int]   = None
    instructions:         Optional[int]   = None
    cache_misses:         Optional[int]   = None
    cache_references:     Optional[int]   = None
    branch_misses:        Optional[int]   = None
    branches:             Optional[int]   = None
    l1d_loads:            Optional[int]   = None
    l1d_load_misses:      Optional[int]   = None
    l1i_load_misses:      Optional[int]   = None
    llc_loads:            Optional[int]   = None
    llc_load_misses:      Optional[int]   = None
    dtlb_load_misses:     Optional[int]   = None
    itlb_load_misses:     Optional[int]   = None
    cpu_migrations:       Optional[int]   = None
    context_switches:     Optional[int]   = None
    elapsed_sec:          Optional[float] = None
    user_sec:             Optional[float] = None
    sys_sec:              Optional[float] = None
    raw_text:             str             = ""

    # ── derived metrics ──────────────────────────────────────────────────────
    @property
    def ipc(self) -> Optional[float]:
        if self.instructions and self.cycles:
            return self.instructions / self.cycles
        return None

    @property
    def l1d_miss_rate(self) -> Optional[float]:
        if self.l1d_load_misses and self.l1d_loads and self.l1d_loads > 0:
            return self.l1d_load_misses / self.l1d_loads * 100
        return None

    @property
    def llc_miss_rate(self) -> Optional[float]:
        if self.llc_load_misses and self.llc_loads and self.llc_loads > 0:
            return self.llc_load_misses / self.llc_loads * 100
        return None

    @property
    def branch_miss_rate(self) -> Optional[float]:
        if self.branch_misses and self.branches and self.branches > 0:
            return self.branch_misses / self.branches * 100
        return None

    @property
    def cache_miss_rate(self) -> Optional[float]:
        if self.cache_misses and self.cache_references and self.cache_references > 0:
            return self.cache_misses / self.cache_references * 100
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_perf(cmd: list[str]) -> str:
    """Run perf stat on the given command, return combined stdout+stderr."""
    if not shutil.which("perf"):
        console.print("[bold red]ERROR:[/] 'perf' not found. "
                      "Install with:  sudo apt install linux-tools-common linux-tools-generic")
        sys.exit(1)

    perf_cmd = ["perf", "stat", "-e", PERF_EVENTS, "--"] + cmd
    console.print(f"\n[dim]Running:[/] {' '.join(perf_cmd)}\n")

    with Progress(SpinnerColumn(), TextColumn("[cyan]Profiling kernel..."),
                  transient=True) as prog:
        prog.add_task("run")
        result = subprocess.run(
            perf_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # capture stderr separately
            text=True,
        )

    # perf stat writes its output to stderr; the kernel's own output is stdout
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    return stdout + "\n" + stderr


def read_paste() -> str:
    console.print("\n[bold cyan]Paste your perf stat output below.[/]")
    console.print("[dim]Press Ctrl-D (or Ctrl-Z on Windows) when done.[/]\n")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────
def _int(s: str) -> int:
    return int(s.replace(",", "").replace(".", "").strip())

def _float(s: str) -> float:
    return float(s.replace(",", "").strip())

def parse_perf(text: str) -> PerfData:
    d = PerfData(raw_text=text)

    patterns = {
        "cycles":           r"([\d,]+)\s+cycles",
        "instructions":     r"([\d,]+)\s+instructions",
        "cache_misses":     r"([\d,]+)\s+cache-misses",
        "cache_references": r"([\d,]+)\s+cache-references",
        "branch_misses":    r"([\d,]+)\s+branch-misses",
        "branches":         r"([\d,]+)\s+branches",
        "l1d_loads":        r"([\d,]+)\s+L1-dcache-loads",
        "l1d_load_misses":  r"([\d,]+)\s+L1-dcache-load-misses",
        "l1i_load_misses":  r"([\d,]+)\s+L1-icache-load-misses",
        "llc_loads":        r"([\d,]+)\s+LLC-loads",
        "llc_load_misses":  r"([\d,]+)\s+LLC-load-misses",
        "dtlb_load_misses": r"([\d,]+)\s+dTLB-load-misses",
        "itlb_load_misses": r"([\d,]+)\s+iTLB-load-misses",
        "cpu_migrations":   r"([\d,]+)\s+cpu-migrations",
        "context_switches": r"([\d,]+)\s+context-switches",
    }

    for attr, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            try:
                setattr(d, attr, _int(m.group(1)))
            except ValueError:
                pass

    # elapsed time
    m = re.search(r"([\d.]+)\s+seconds time elapsed", text)
    if m: d.elapsed_sec = float(m.group(1))

    m = re.search(r"([\d.]+)\s+seconds user", text)
    if m: d.user_sec = float(m.group(1))

    m = re.search(r"([\d.]+)\s+seconds sys", text)
    if m: d.sys_sec = float(m.group(1))

    return d


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(n: Optional[int], unit: str = "") -> str:
    if n is None: return "[dim]N/A[/]"
    if n >= 1_000_000_000: return f"{n/1e9:.2f}B{unit}"
    if n >= 1_000_000:     return f"{n/1e6:.2f}M{unit}"
    if n >= 1_000:         return f"{n/1e3:.1f}K{unit}"
    return f"{n}{unit}"

def _pct(v: Optional[float], warn: float, bad: float) -> str:
    if v is None: return "[dim]N/A[/]"
    s = f"{v:.2f}%"
    if v >= bad:  return f"[bold red]{s}[/]"
    if v >= warn: return f"[yellow]{s}[/]"
    return f"[green]{s}[/]"

def _ipc_color(v: Optional[float]) -> str:
    if v is None: return "[dim]N/A[/]"
    s = f"{v:.3f}"
    if v >= 2.5:  return f"[bold green]{s}[/]"
    if v >= 1.5:  return f"[green]{s}[/]"
    if v >= 1.0:  return f"[yellow]{s}[/]"
    return f"[bold red]{s}[/]"


def display_dashboard(d: PerfData):
    console.print()
    console.print(Rule("[bold white] CPU KERNEL PERFORMANCE REPORT [/]", style="bright_cyan"))
    console.print()

    # ── timing panel ────────────────────────────────────────────────────────
    timing = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    timing.add_column("metric", style="dim")
    timing.add_column("value",  style="bold white")
    if d.elapsed_sec: timing.add_row("Wall time",  f"[cyan]{d.elapsed_sec:.3f}s[/]")
    if d.user_sec:    timing.add_row("User time",  f"{d.user_sec:.3f}s")
    if d.sys_sec:     timing.add_row("Sys time",   f"{d.sys_sec:.3f}s")
    if d.context_switches is not None:
        timing.add_row("Ctx switches", _fmt(d.context_switches))
    if d.cpu_migrations is not None:
        timing.add_row("CPU migrations", _fmt(d.cpu_migrations))

    # ── compute panel ────────────────────────────────────────────────────────
    compute = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    compute.add_column("metric", style="dim")
    compute.add_column("value",  style="bold white")
    compute.add_row("Cycles",       _fmt(d.cycles))
    compute.add_row("Instructions", _fmt(d.instructions))
    compute.add_row("IPC",          _ipc_color(d.ipc))
    compute.add_row("Branches",     _fmt(d.branches))
    compute.add_row("Branch misses",
        f"{_fmt(d.branch_misses)} ({_pct(d.branch_miss_rate, 1.0, 5.0)})")

    # ── cache panel ──────────────────────────────────────────────────────────
    cache = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    cache.add_column("metric", style="dim")
    cache.add_column("value",  style="bold white")
    cache.add_row("L1d loads",      _fmt(d.l1d_loads))
    cache.add_row("L1d misses",
        f"{_fmt(d.l1d_load_misses)} ({_pct(d.l1d_miss_rate, 10.0, 20.0)})")
    cache.add_row("L1i misses",     _fmt(d.l1i_load_misses))
    cache.add_row("LLC loads",      _fmt(d.llc_loads))
    cache.add_row("LLC misses",
        f"{_fmt(d.llc_load_misses)} ({_pct(d.llc_miss_rate, 2.0, 5.0)})")
    cache.add_row("Cache miss rate", _pct(d.cache_miss_rate, 0.5, 2.0))

    # ── TLB panel ────────────────────────────────────────────────────────────
    tlb = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    tlb.add_column("metric", style="dim")
    tlb.add_column("value",  style="bold white")
    tlb.add_row("dTLB misses", _fmt(d.dtlb_load_misses))
    tlb.add_row("iTLB misses", _fmt(d.itlb_load_misses))

    console.print(Columns([
        Panel(timing,  title="[bold cyan]⏱  Timing[/]",   border_style="cyan",   width=38),
        Panel(compute, title="[bold yellow]⚡ Compute[/]", border_style="yellow", width=38),
    ]))
    console.print(Columns([
        Panel(cache, title="[bold magenta]🗄  Cache[/]",   border_style="magenta", width=38),
        Panel(tlb,   title="[bold blue]🔍 TLB[/]",        border_style="blue",    width=38),
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK CLASSIFIER + ADVICE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Issue:
    severity: str          # "critical" | "warning" | "info"
    category: str
    headline: str
    detail:   str
    advice:   list[str] = field(default_factory=list)


def classify(d: PerfData) -> list[Issue]:
    issues: list[Issue] = []

    # ── IPC ──────────────────────────────────────────────────────────────────
    if d.ipc is not None:
        if d.ipc < 0.5:
            issues.append(Issue(
                severity="critical", category="Compute",
                headline=f"Extremely low IPC ({d.ipc:.2f}) — severe stalls",
                detail="CPU is idle most of the time. Almost certainly a memory bottleneck "
                       "or a long serialised dependency chain.",
                advice=[
                    "Profile with: perf stat -e cycle_activity.stalls_l1d_miss,stalls_l2_miss,stalls_l3_miss",
                    "Check for pointer-chasing loops (linked lists, tree traversals)",
                    "Look for long dependency chains in the hot loop",
                    "Consider prefetching: __builtin_prefetch(&ptr->next, 0, 1)",
                ]))
        elif d.ipc < 1.0:
            issues.append(Issue(
                severity="warning", category="Compute",
                headline=f"Low IPC ({d.ipc:.2f}) — CPU frequently stalling",
                detail="Broadwell/Skylake peak IPC is ~4. You are using <25% of issue width. "
                       "Memory latency or instruction dependency chains are the likely cause.",
                advice=[
                    "Unroll inner loops to expose more independent instructions",
                    "Use multiple accumulators to break false dependencies",
                    "Add -funroll-loops or manual unroll pragma",
                    "Inspect hot loop in: objdump -d binary | grep -A 50 '<hotfunc>'",
                ]))
        elif d.ipc < 2.0:
            issues.append(Issue(
                severity="info", category="Compute",
                headline=f"Moderate IPC ({d.ipc:.2f}) — room for improvement",
                detail="IPC is reasonable but Broadwell can sustain 3–4 IPC for well-vectorized loops. "
                       "Likely some memory stalls or suboptimal instruction mix.",
                advice=[
                    "Verify AVX2 vectorization: objdump -d binary | grep -c 'ymm'",
                    "Enable auto-vectorization report: -fopt-info-vec-optimized",
                    "Try Profile-Guided Optimization (PGO): -fprofile-generate / -fprofile-use",
                ]))
        else:
            issues.append(Issue(
                severity="info", category="Compute",
                headline=f"Good IPC ({d.ipc:.2f})",
                detail="IPC is healthy. Compute units are well-utilized.",
                advice=["Consider multi-threading to scale to more cores if not already done."]))

    # ── L1d miss rate ─────────────────────────────────────────────────────────
    if d.l1d_miss_rate is not None:
        if d.l1d_miss_rate > 25:
            issues.append(Issue(
                severity="critical", category="L1 Cache",
                headline=f"Critical L1d miss rate ({d.l1d_miss_rate:.1f}%)",
                detail="More than 1 in 4 L1 loads is missing. This is the dominant bottleneck. "
                       "Each miss costs 4–12 cycles going to L2.",
                advice=[
                    "Reduce working set: use smaller block/tile size",
                    "Improve data locality: reorder loops to access memory sequentially",
                    "Pack hot data structures to eliminate padding gaps (use __attribute__((packed)))",
                    "Separate hot and cold fields in structs (SoA instead of AoS)",
                    "Check block size: for matmul, (MR+NR)*KC*4 should be < 32KB (L1d)",
                ]))
        elif d.l1d_miss_rate > 15:
            issues.append(Issue(
                severity="warning", category="L1 Cache",
                headline=f"High L1d miss rate ({d.l1d_miss_rate:.1f}%)",
                detail="Significant L1 pressure. Data is spilling to L2 frequently.",
                advice=[
                    "Reduce tile/block size to fit hot arrays in L1d (32KB)",
                    "Use __builtin_prefetch to hide L2 latency: __builtin_prefetch(ptr+N, 0, 1)",
                    "Ensure hot arrays are cache-line aligned: alignas(64)",
                ]))

    # ── LLC miss rate ─────────────────────────────────────────────────────────
    if d.llc_miss_rate is not None:
        if d.llc_miss_rate > 10:
            issues.append(Issue(
                severity="critical", category="LLC / DRAM",
                headline=f"Critical LLC miss rate ({d.llc_miss_rate:.1f}%) — DRAM bound",
                detail="Data is coming from DRAM. Each miss costs 100–300 cycles. "
                       "This is the worst possible memory bottleneck.",
                advice=[
                    "Reduce working set to fit in L3 (your L3 = ~20MB for E5-2620 v4)",
                    "Use blocking/tiling: NC = floor(L3*0.5 / (KC*4))",
                    "Pre-pack B matrix to avoid non-sequential DRAM access",
                    "Consider NUMA locality: numactl --cpunodebind=0 --membind=0 ./binary",
                    "Measure DRAM bandwidth: stream benchmark",
                ]))
        elif d.llc_miss_rate > 3:
            issues.append(Issue(
                severity="warning", category="LLC / DRAM",
                headline=f"Elevated LLC miss rate ({d.llc_miss_rate:.1f}%)",
                detail="Some data is going to DRAM. Tighten your NC blocking parameter.",
                advice=[
                    "Reduce NC block size so B panel fits in L3",
                    "Profile with: perf stat -e cycle_activity.stalls_l3_miss ./binary",
                ]))

    # ── Branch mispredictions ─────────────────────────────────────────────────
    if d.branch_miss_rate is not None:
        if d.branch_miss_rate > 5:
            issues.append(Issue(
                severity="warning", category="Branch Predictor",
                headline=f"High branch miss rate ({d.branch_miss_rate:.1f}%)",
                detail="Branch mispredictions cause a ~15-cycle pipeline flush each time. "
                       "Common causes: data-dependent branches, sparse if/else in hot loops.",
                advice=[
                    "Replace data-dependent branches with branchless cmov:",
                    "  int result = (a > b) ? a : b;  ->  use std::max or ternary",
                    "Use __builtin_expect for predictable branches: if (__builtin_expect(cond, 1))",
                    "Consider lookup tables instead of switch statements",
                    "Profile hot functions with: perf record -e branch-misses ./binary && perf report",
                ]))

    # ── dTLB misses ───────────────────────────────────────────────────────────
    if d.dtlb_load_misses is not None:
        if d.dtlb_load_misses > 50_000_000:
            issues.append(Issue(
                severity="warning", category="TLB",
                headline=f"High dTLB miss count ({d.dtlb_load_misses/1e6:.1f}M)",
                detail="TLB misses cause page-table walks (~10–100 cycles each). "
                       "Common with large arrays and non-sequential access patterns.",
                advice=[
                    "Use huge pages: mmap(..., MAP_HUGETLB) or /proc/sys/vm/nr_hugepages",
                    "Align large allocations: aligned_alloc(2*1024*1024, size)",
                    "Reduce pointer chasing / improve spatial locality",
                    "Enable transparent huge pages: echo madvise > /sys/kernel/mm/transparent_hugepage/enabled",
                ]))

    # ── Context switches ──────────────────────────────────────────────────────
    if d.context_switches is not None and d.context_switches > 500:
        issues.append(Issue(
            severity="warning", category="Scheduling",
            headline=f"High context switches ({d.context_switches})",
            detail="Frequent context switches cause cache pollution and TLB flushes. "
                   "Kernel is preempting your process or you have I/O waits.",
            advice=[
                "Pin to a CPU core: taskset -c 0 ./binary",
                "Set real-time priority: chrt -f 99 ./binary  (requires root)",
                "Disable CPU frequency scaling: cpupower frequency-set -g performance",
                "Check for unintended I/O in hot path",
            ]))

    # ── CPU migrations ────────────────────────────────────────────────────────
    if d.cpu_migrations is not None and d.cpu_migrations > 2:
        issues.append(Issue(
            severity="warning", category="Scheduling",
            headline=f"CPU migrations detected ({d.cpu_migrations})",
            detail="Process moved between CPU cores during execution. "
                   "This causes cache warmup overhead and can cross NUMA domains.",
            advice=[
                "Pin to specific cores: taskset -c 0-7 ./binary",
                "For NUMA: numactl --cpunodebind=0 --membind=0 ./binary",
            ]))

    # ── No issues ─────────────────────────────────────────────────────────────
    if not issues:
        issues.append(Issue(
            severity="info", category="Overall",
            headline="No major bottlenecks detected",
            detail="All measured metrics are within healthy ranges.",
            advice=["Consider multi-threading if you haven't already.",
                    "Profile with VTune or AMD uProf for deeper microarchitectural analysis."]))

    return issues


def display_advice(issues: list[Issue]):
    console.print()
    console.print(Rule("[bold white] BOTTLENECK ANALYSIS & OPTIMISATION ADVICE [/]",
                       style="bright_cyan"))
    console.print()

    severity_style = {
        "critical": ("bold red",    "🔴 CRITICAL"),
        "warning":  ("yellow",      "🟡 WARNING"),
        "info":     ("bright_blue", "🔵 INFO"),
    }

    for i, issue in enumerate(issues, 1):
        style, label = severity_style[issue.severity]
        title = f"[{style}]{label}[/]  [{style}]{issue.category}[/]"

        body = Text()
        body.append(f"{issue.headline}\n", style=f"bold {style}")
        body.append(f"\n{issue.detail}\n", style="white")

        if issue.advice:
            body.append("\nOptimisation steps:\n", style="bold cyan")
            for step in issue.advice:
                body.append(f"  › {step}\n", style="dim white")

        console.print(Panel(body, title=title, border_style=style, padding=(0, 2)))
        console.print()


# ─────────────────────────────────────────────────────────────────────────────
# ROOFLINE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def display_roofline_summary(d: PerfData):
    console.print(Rule("[bold white] ROOFLINE SUMMARY [/]", style="bright_cyan"))
    console.print()

    rows = []

    # IPC efficiency (Broadwell peak ~4)
    if d.ipc is not None:
        pct = min(d.ipc / 4.0 * 100, 100)
        bar = make_bar(pct)
        rows.append(("IPC efficiency",
                     f"[cyan]{d.ipc:.2f}[/] / 4.0",
                     f"{bar} {pct:.0f}%"))

    # L1d hit rate
    if d.l1d_miss_rate is not None:
        hit = 100 - d.l1d_miss_rate
        bar = make_bar(hit)
        rows.append(("L1d hit rate", f"[cyan]{hit:.1f}%[/]", f"{bar} {hit:.0f}%"))

    # LLC hit rate
    if d.llc_miss_rate is not None:
        hit = 100 - d.llc_miss_rate
        bar = make_bar(hit)
        rows.append(("LLC hit rate", f"[cyan]{hit:.1f}%[/]", f"{bar} {hit:.0f}%"))

    # Branch prediction accuracy
    if d.branch_miss_rate is not None:
        acc = 100 - d.branch_miss_rate
        bar = make_bar(acc)
        rows.append(("Branch accuracy", f"[cyan]{acc:.2f}%[/]", f"{bar} {acc:.0f}%"))

    if rows:
        t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
        t.add_column("Metric",      style="white",      width=22)
        t.add_column("Value",       style="bold white", width=16)
        t.add_column("Visual",      style="white",      width=36)
        for r in rows:
            t.add_row(*r)
        console.print(t)
    console.print()


def make_bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    if pct >= 75:   color = "green"
    elif pct >= 40: color = "yellow"
    else:           color = "red"
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/]"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run perf stat on a kernel, parse results, and give optimisation advice.")
    parser.add_argument("cmd", nargs="*",
                        help="Command to profile, e.g.: ./matmul_cpu")
    parser.add_argument("--paste", action="store_true",
                        help="Paste raw perf output interactively")
    parser.add_argument("--file", metavar="FILE",
                        help="Read perf output from a saved file")
    args = parser.parse_args()

    # ── Get raw perf text ────────────────────────────────────────────────────
    if args.file:
        with open(args.file) as f:
            raw = f.read()
    elif args.paste:
        raw = read_paste()
    elif args.cmd:
        raw = run_perf(args.cmd)
    else:
        parser.print_help()
        console.print("\n[bold yellow]Example usage:[/]")
        console.print("  python3 cpu_perf_analyzer.py ./matmul_cpu")
        console.print("  python3 cpu_perf_analyzer.py --paste")
        console.print("  python3 cpu_perf_analyzer.py --file saved_perf.txt")
        sys.exit(0)

    # ── Parse ────────────────────────────────────────────────────────────────
    d = parse_perf(raw)

    # ── Display ──────────────────────────────────────────────────────────────
    display_dashboard(d)
    display_roofline_summary(d)
    issues = classify(d)
    display_advice(issues)

    console.print(Rule(style="dim"))
    console.print(
        "[dim]Tip: run with deeper events for more insight:[/]\n"
        "[cyan]  perf stat -e cycle_activity.stalls_l1d_miss,"
        "cycle_activity.stalls_l2_miss,"
        "cycle_activity.stalls_l3_miss,"
        "fp_arith_inst_retired.256b_packed_single "
        "./your_kernel[/]\n"
    )


if __name__ == "__main__":
    main()
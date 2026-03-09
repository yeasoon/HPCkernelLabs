#pragma once
// =============================================================================
// benchmark.hpp  —  Zero-dependency C++ kernel benchmarking framework
//
// Usage:
//   #include "benchmark.hpp"
//   #include "your_kernel.h"
//
//   int main() {
//       bench::Suite suite("My Kernel");
//
//       suite.add("matmul_naive", [&]{ your_kernel(A, B, C, N); },
//                 /*flops=*/ 2.0 * N * N * N,
//                 /*bytes=*/ 3.0 * N * N * sizeof(float));
//
//       suite.run();
//       suite.report();
//   }
//
// Compile:
//   g++ -O3 -march=native -mfma -mavx2 -std=c++17 -o bench my_bench.cpp
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <optional>

// ── optional PAPI hardware counter support ───────────────────────────────────
#ifdef USE_PAPI
#include <papi.h>
#endif

namespace bench
{

    // =============================================================================
    // UTILITIES
    // =============================================================================

    // Prevent the compiler from optimising away a result
    template <typename T>
    inline void do_not_optimise(T const &val)
    {
        asm volatile("" : : "r,m"(val) : "memory");
    }
    // Overload for non-const (e.g. output arrays)
    template <typename T>
    inline void do_not_optimise(T &val)
    {
#if defined(__clang__)
        asm volatile("" : "+r,m"(val) : : "memory");
#else
        asm volatile("" : "+m,r"(val) : : "memory");
#endif
    }

    // Memory fence — prevents store-to-load forwarding across iterations
    inline void memory_fence()
    {
        asm volatile("" : : : "memory");
    }

    // High-resolution wall clock
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Seconds = std::chrono::duration<double>;

    inline double now_sec()
    {
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

// CPU cycle counter (x86 only)
#if defined(__x86_64__) || defined(__i386__)
    inline uint64_t rdtsc()
    {
        uint32_t lo, hi;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return (uint64_t)hi << 32 | lo;
    }
    inline uint64_t rdtscp()
    {
        uint32_t lo, hi, aux;
        __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
        return (uint64_t)hi << 32 | lo;
    }
#else
    inline uint64_t rdtsc() { return 0; }
    inline uint64_t rdtscp() { return 0; }
#endif

    // =============================================================================
    // CONFIG
    // =============================================================================
    struct Config
    {
        int warmup_iters = 3;       // iterations before timing starts
        int min_iters = 10;         // minimum timed iterations
        double min_duration = 1.0;  // keep running until this many seconds
        bool show_histogram = true; // print latency histogram
        bool show_raw = false;      // print every iteration time
        bool use_cycles = true;     // report cycles if rdtsc available
        double cpu_ghz = 0.0;       // if 0, auto-detect from timing
        int histogram_bins = 10;
    };

    // =============================================================================
    // RESULT — statistics for one benchmark case
    // =============================================================================
    struct Result
    {
        std::string name;
        std::string group;

        // raw samples (seconds per iteration)
        std::vector<double> samples;

        // optional metadata
        double flops = 0.0; // FLOPs per kernel call (for GFLOPS)
        double bytes = 0.0; // bytes transferred (for bandwidth)

        // ── computed statistics ──────────────────────────────────────────────────
        double mean_sec = 0.0;
        double median_sec = 0.0;
        double min_sec = 0.0;
        double max_sec = 0.0;
        double stddev_sec = 0.0;
        double p95_sec = 0.0;
        double p99_sec = 0.0;

        // derived
        double gflops() const { return (flops > 0 && mean_sec > 0) ? flops / mean_sec / 1e9 : 0.0; }
        double bandwidth() const { return (bytes > 0 && mean_sec > 0) ? bytes / mean_sec / 1e9 : 0.0; }
        double throughput() const { return mean_sec > 0 ? 1.0 / mean_sec : 0.0; }

        void compute_stats()
        {
            if (samples.empty())
                return;
            std::sort(samples.begin(), samples.end());

            min_sec = samples.front();
            max_sec = samples.back();
            mean_sec = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
            median_sec = samples[samples.size() / 2];

            auto pct = [&](double p)
            {
                size_t idx = (size_t)(p / 100.0 * samples.size());
                return samples[std::min(idx, samples.size() - 1)];
            };
            p95_sec = pct(95);
            p99_sec = pct(99);

            double var = 0;
            for (auto s : samples)
                var += (s - mean_sec) * (s - mean_sec);
            stddev_sec = std::sqrt(var / samples.size());
        }
    };

    // =============================================================================
    // CASE — one benchmark entry
    // =============================================================================
    struct Case
    {
        std::string name;
        std::string group;
        std::function<void()> fn;
        double flops = 0.0;
        double bytes = 0.0;
        std::string note;

        // optional: reference function for correctness check
        std::function<bool()> checker;
    };

    // =============================================================================
    // SUITE — collection of benchmarks
    // =============================================================================
    class Suite
    {
    public:
        explicit Suite(std::string title = "Kernel Benchmark", Config cfg = {})
            : title_(std::move(title)), cfg_(cfg) {}

        // ── add a benchmark ──────────────────────────────────────────────────────
        Suite &add(const std::string &name,
                   std::function<void()> fn,
                   double flops = 0.0,
                   double bytes = 0.0,
                   const std::string &group = "",
                   const std::string &note = "")
        {
            cases_.push_back({name, group, std::move(fn), flops, bytes, note});
            return *this;
        }

        // add with correctness checker
        Suite &add_checked(const std::string &name,
                           std::function<void()> fn,
                           std::function<bool()> checker,
                           double flops = 0.0,
                           double bytes = 0.0,
                           const std::string &group = "")
        {
            Case c;
            c.name = name;
            c.group = group;
            c.fn = std::move(fn);
            c.checker = std::move(checker);
            c.flops = flops;
            c.bytes = bytes;
            cases_.push_back(std::move(c));
            return *this;
        }

        Config &config() { return cfg_; }

        // ── run all cases ────────────────────────────────────────────────────────
        void run()
        {
            print_header();

            for (auto &c : cases_)
            {
                Result r = run_one(c);
                results_.push_back(r);
                print_result_line(r);
            }
        }

        // ── print full report ────────────────────────────────────────────────────
        void report() const
        {
            print_separator('=');
            print_center("DETAILED REPORT: " + title_);
            print_separator('=');

            for (auto &r : results_)
            {
                print_result_detail(r);
            }

            print_comparison();
            print_separator('=');
        }

        // ── export to CSV ────────────────────────────────────────────────────────
        void to_csv(const std::string &path) const
        {
            std::ofstream f(path);
            if (!f)
            {
                std::cerr << "Cannot open " << path << "\n";
                return;
            }
            f << "name,group,mean_ms,median_ms,min_ms,max_ms,stddev_ms,"
                 "p95_ms,p99_ms,gflops,bandwidth_gbs,iters\n";
            for (auto &r : results_)
            {
                auto ms = [](double s)
                { return s * 1e3; };
                f << r.name << "," << r.group << ","
                  << ms(r.mean_sec) << "," << ms(r.median_sec) << ","
                  << ms(r.min_sec) << "," << ms(r.max_sec) << ","
                  << ms(r.stddev_sec) << "," << ms(r.p95_sec) << ","
                  << ms(r.p99_sec) << "," << r.gflops() << ","
                  << r.bandwidth() << "," << r.samples.size() << "\n";
            }
            std::cout << "CSV saved: " << path << "\n";
        }

        const std::vector<Result> &results() const { return results_; }

    private:
        // ── run one benchmark case ───────────────────────────────────────────────
        Result run_one(const Case &c)
        {
            Result r;
            r.name = c.name;
            r.group = c.group;
            r.flops = c.flops;
            r.bytes = c.bytes;

            // correctness check first
            if (c.checker)
            {
                bool ok = c.checker();
                if (!ok)
                {
                    std::cout << "  [FAIL] " << c.name << " correctness check failed!\n";
                }
            }

            // warmup
            for (int i = 0; i < cfg_.warmup_iters; i++)
            {
                c.fn();
                memory_fence();
            }

            // timed iterations
            double total = 0.0;
            int iters = 0;

            while (iters < cfg_.min_iters || total < cfg_.min_duration)
            {
                auto t0 = Clock::now();
                c.fn();
                memory_fence();
                auto t1 = Clock::now();

                double elapsed = Seconds(t1 - t0).count();
                r.samples.push_back(elapsed);
                total += elapsed;
                iters++;

                // safety cap
                if (iters >= 10000)
                    break;
            }

            r.compute_stats();
            return r;
        }

        // ── formatting helpers ───────────────────────────────────────────────────
        static std::string fmt_time(double sec)
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(3);
            if (sec < 1e-6)
                ss << sec * 1e9 << " ns";
            else if (sec < 1e-3)
                ss << sec * 1e6 << " us";
            else if (sec < 1.0)
                ss << sec * 1e3 << " ms";
            else
                ss << sec << "  s";
            return ss.str();
        }

        static std::string fmt_num(double v, int prec = 2)
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(prec) << v;
            return ss.str();
        }

        static void print_separator(char c = '-', int w = 88)
        {
            std::cout << std::string(w, c) << "\n";
        }

        static void print_center(const std::string &s, int w = 88)
        {
            int pad = (w - (int)s.size()) / 2;
            std::cout << std::string(std::max(pad, 0), ' ') << s << "\n";
        }

        void print_header() const
        {
            print_separator('=');
            print_center(title_);
            print_separator('=');
            std::cout << std::left
                      << std::setw(28) << "Name"
                      << std::setw(12) << "Mean"
                      << std::setw(12) << "Median"
                      << std::setw(12) << "Min"
                      << std::setw(10) << "Stddev"
                      << std::setw(10) << "GFLOPS"
                      << std::setw(10) << "GB/s"
                      << "Iters\n";
            print_separator('-');
        }

        void print_result_line(const Result &r) const
        {
            std::cout << std::left
                      << std::setw(28) << r.name.substr(0, 27)
                      << std::setw(12) << fmt_time(r.mean_sec)
                      << std::setw(12) << fmt_time(r.median_sec)
                      << std::setw(12) << fmt_time(r.min_sec)
                      << std::setw(10) << fmt_time(r.stddev_sec);
            if (r.flops > 0)
                std::cout << std::setw(10) << fmt_num(r.gflops());
            else
                std::cout << std::setw(10) << "  -";
            if (r.bytes > 0)
                std::cout << std::setw(10) << fmt_num(r.bandwidth());
            else
                std::cout << std::setw(10) << "  -";
            std::cout << r.samples.size() << "\n";
        }

        void print_result_detail(const Result &r) const
        {
            std::cout << "\n";
            print_separator('-');
            std::cout << "  Benchmark : " << r.name << "\n";
            if (!r.group.empty())
                std::cout << "  Group     : " << r.group << "\n";
            print_separator('-');

            auto row = [](const std::string &k, const std::string &v)
            {
                std::cout << "  " << std::left << std::setw(18) << k << v << "\n";
            };

            row("Mean", fmt_time(r.mean_sec));
            row("Median", fmt_time(r.median_sec));
            row("Min", fmt_time(r.min_sec));
            row("Max", fmt_time(r.max_sec));
            row("Std Dev", fmt_time(r.stddev_sec));
            row("P95", fmt_time(r.p95_sec));
            row("P99", fmt_time(r.p99_sec));
            row("Samples", std::to_string(r.samples.size()));

            if (r.flops > 0)
                row("GFLOPS", fmt_num(r.gflops(), 3));
            if (r.bytes > 0)
                row("Bandwidth", fmt_num(r.bandwidth(), 3) + " GB/s");

            // coefficient of variation
            double cv = r.mean_sec > 0 ? r.stddev_sec / r.mean_sec * 100 : 0;
            row("CV%", fmt_num(cv, 1) + "%" + (cv > 5 ? "  ← noisy" : "  ← stable"));

            if (cfg_.show_histogram && r.samples.size() >= 5)
                print_histogram(r);
        }

        void print_histogram(const Result &r) const
        {
            std::cout << "\n  Latency histogram (ms):\n";
            int bins = cfg_.histogram_bins;
            double lo = r.min_sec;
            double hi = r.max_sec;
            double bw = (hi - lo) / bins;
            if (bw <= 0)
                return;

            std::vector<int> counts(bins, 0);
            for (double s : r.samples)
            {
                int b = (int)((s - lo) / bw);
                counts[std::min(b, bins - 1)]++;
            }
            int maxc = *std::max_element(counts.begin(), counts.end());
            int barw = 30;

            for (int i = 0; i < bins; i++)
            {
                double edge = (lo + i * bw) * 1e3;
                int len = maxc > 0 ? counts[i] * barw / maxc : 0;
                std::cout << "  " << std::fixed << std::setw(8) << std::setprecision(3)
                          << edge << " ms |"
                          << std::string(len, '#')
                          << std::string(barw - len, ' ')
                          << "| " << counts[i] << "\n";
            }
        }

        void print_comparison() const
        {
            if (results_.size() < 2)
                return;

            std::cout << "\n";
            print_separator('-');
            print_center("SPEEDUP COMPARISON (vs slowest)");
            print_separator('-');

            // find slowest
            auto slowest = std::max_element(results_.begin(), results_.end(),
                                            [](const Result &a, const Result &b)
                                            { return a.mean_sec < b.mean_sec; });

            std::cout << std::left
                      << std::setw(28) << "Name"
                      << std::setw(14) << "Mean"
                      << std::setw(12) << "Speedup"
                      << "Bar\n";
            print_separator('-');

            for (auto &r : results_)
            {
                double speedup = slowest->mean_sec / r.mean_sec;
                int barlen = (int)(speedup / (slowest->mean_sec / slowest->mean_sec + 0.01) * 20 / (results_.size()));
                barlen = std::min(barlen, 40);
                int filled = (int)(speedup * 4);
                filled = std::min(filled, 40);

                std::cout << std::left
                          << std::setw(28) << r.name.substr(0, 27)
                          << std::setw(14) << fmt_time(r.mean_sec)
                          << std::setw(12) << (fmt_num(speedup, 2) + "x")
                          << std::string(filled, '|') << "\n";
            }
        }

        std::string title_;
        Config cfg_;
        std::vector<Case> cases_;
        std::vector<Result> results_;
    };

// =============================================================================
// CONVENIENCE MACROS
// =============================================================================

// Time a single expression N times and print result
#define BENCH_EXPR(name, expr, N)    \
    do                               \
    {                                \
        bench::Suite _s(name);       \
        _s.add(name, [&] { expr; }); \
        _s.config().min_iters = (N); \
        _s.run();                    \
        _s.report();                 \
    } while (0)

// Quick single-shot timing
#define TIME_IT(label, expr)                                  \
    do                                                        \
    {                                                         \
        auto _t0 = bench::Clock::now();                       \
        expr;                                                 \
        auto _t1 = bench::Clock::now();                       \
        double _ms = bench::Seconds(_t1 - _t0).count() * 1e3; \
        std::cout << "[TIME] " << label << ": "               \
                  << std::fixed << std::setprecision(3)       \
                  << _ms << " ms\n";                          \
    } while (0)

    // =============================================================================
    // HARDWARE INFO  (best-effort, Linux only)
    // =============================================================================
    inline void print_hw_info()
    {
        std::cout << "\n=== Hardware Info ===\n";

#ifdef __linux__
        // CPU model
        if (FILE *f = fopen("/proc/cpuinfo", "r"))
        {
            char line[256];
            while (fgets(line, sizeof(line), f))
            {
                if (strncmp(line, "model name", 10) == 0)
                {
                    std::cout << "CPU     : " << (strchr(line, ':') + 2);
                    break;
                }
            }
            fclose(f);
        }

        // CPU MHz
        if (FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r"))
        {
            long khz = 0;
            (void)fscanf(f, "%ld", &khz);
            fclose(f);
            std::cout << "Freq    : " << khz / 1000 << " MHz (current)\n";
        }

        // Cache sizes
        for (int level = 1; level <= 3; level++)
        {
            std::string p = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(level) + "/size";
            if (FILE *f = fopen(p.c_str(), "r"))
            {
                char buf[32] = {};
                (void)fscanf(f, "%31s", buf);
                fclose(f);
                std::cout << "L" << level << " cache : " << buf << "\n";
            }
        }

        // Core count
        int cores = 0;
        if (FILE *f = fopen("/proc/cpuinfo", "r"))
        {
            char line[256];
            while (fgets(line, sizeof(line), f))
                if (strncmp(line, "processor", 9) == 0)
                    cores++;
            fclose(f);
            std::cout << "Cores   : " << cores << " logical\n";
        }
#endif

        // SIMD support
        std::cout << "SIMD    :";
#ifdef __AVX512F__
        std::cout << " AVX-512";
#endif
#ifdef __AVX2__
        std::cout << " AVX2";
#endif
#ifdef __AVX__
        std::cout << " AVX";
#endif
#ifdef __SSE4_2__
        std::cout << " SSE4.2";
#endif
#ifdef __FMA__
        std::cout << " FMA";
#endif
        std::cout << "\n";
        std::cout << "====================\n\n";
    }

} // namespace bench

// =============================================================================
// OPTIONAL: include this to get a ready-made main() with CLI argument parsing
// =============================================================================
#ifdef BENCH_MAIN
#include <cstdlib>

// Users define this function in their .cpp file
void register_benchmarks(bench::Suite &suite, int argc, char **argv);

int main(int argc, char **argv)
{
    bench::print_hw_info();

    bench::Config cfg;
    // parse simple CLI flags
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--warmup" && i + 1 < argc)
            cfg.warmup_iters = atoi(argv[++i]);
        if (a == "--iters" && i + 1 < argc)
            cfg.min_iters = atoi(argv[++i]);
        if (a == "--duration" && i + 1 < argc)
            cfg.min_duration = atof(argv[++i]);
        if (a == "--no-hist")
            cfg.show_histogram = false;
        if (a == "--raw")
            cfg.show_raw = true;
    }

    bench::Suite suite("Kernel Benchmark", cfg);
    register_benchmarks(suite, argc, argv);
    suite.run();
    suite.report();

    // auto-export CSV if --csv flag present
    for (int i = 1; i < argc - 1; i++)
        if (std::string(argv[i]) == "--csv")
            suite.to_csv(argv[i + 1]);

    return 0;
}
#endif // BENCH_MAIN
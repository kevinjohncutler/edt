/*
 * General Lp Distance Transform (ND)
 *
 * Computes the p-th power of the Lp distance transform for any p >= 1.
 * For p=2, this reduces to the squared Euclidean distance transform.
 *
 * Input: a labels array or a pre-built voxel connectivity graph
 *        (bit-encoded uint8 for 1-4D, uint16 for 5-8D, uint32 for 9-16D, uint64 for 17-32D).
 *
 * Pipeline (edtp / edtp_from_labels_fused):
 *   1. Build a compact connectivity graph (same as L2 version -- metric-independent).
 *   2. Run all EDT passes directly from the graph:
 *      - Pass 0 (innermost axis): forward/backward scan with |d|^p cost.
 *      - Passes 1..N-1: lower envelope with bisection-based intersection finding.
 *        For p=2, uses closed-form parabola intersection (O(1) per intersection).
 *        For general p, uses bisection (O(log n) per intersection).
 *   Parallelized across scanlines.
 *
 * Algorithm reference:
 *   Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions" (2012)
 *   -- Section on generalization to convex cost functions with bisection.
 */

#ifndef EDT_LP_HPP
#define EDT_LP_HPP

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#ifndef _WIN32
  #include <unistd.h>  // gethostname
#endif
#include "threadpool.h"

// MSVC uses __restrict (no trailing underscores); GCC/Clang use __restrict__
#ifdef _MSC_VER
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

namespace lp {

// Maximum dimensionality supported. Limited by the bit-encoded graph type
// (uint64 covers 2*(dims-1)+1 bits, so dims<=32) and by the fixed stack
// arrays in the typed implementation (strides[32], shape64[32], etc.).
// Inputs above this throw std::invalid_argument from the dispatcher.
static constexpr size_t EDT_MAX_DIMS = 32;

// Tuning parameter: more chunks = better load balancing with atomic work-stealing
static size_t ND_CHUNKS_PER_THREAD = 4;

inline void set_tuning(size_t chunks_per_thread) {
    if (chunks_per_thread > 0) ND_CHUNKS_PER_THREAD = chunks_per_thread;
}

// Shared fork-join pool keyed by thread count; created lazily on first use.
// Same thread-safety contract as src/edt.hpp::shared_pool_for: static
// mutex serializes the map operations; references into the static map
// stay valid for the process lifetime; the pool itself is safe for
// concurrent parallel() calls.
inline edt::ForkJoinPool& shared_pool_for(size_t threads) {
    static std::mutex pool_mu;
    static std::unordered_map<size_t, std::unique_ptr<edt::ForkJoinPool>> pools;
    std::lock_guard<std::mutex> lock(pool_mu);
    auto& entry = pools[threads];
    if (!entry) {
        entry = std::make_unique<edt::ForkJoinPool>(threads);
    }
    return *entry;
}

// Per-call-chain "active pool". Top-level entry points (e.g.
// expand_labels_fused) install a pool reference here for the duration
// of one invocation; dispatch_parallel reads it instead of looking up
// shared_pool_for(threads) on every call. This collapses N pool lookups
// (one per axis pass / dispatch) down to one per top-level call, which
// matters on x86 many-core where the mutex+unordered_map cost is
// non-trivial relative to a small dispatch (≈30-100us total per
// dispatch on TR PRO 3995WX).
inline thread_local edt::ForkJoinPool* tls_active_pool = nullptr;

struct ActivePoolGuard {
    edt::ForkJoinPool* prev;
    explicit ActivePoolGuard(edt::ForkJoinPool& pool) : prev(tls_active_pool) {
        tls_active_pool = &pool;
    }
    ~ActivePoolGuard() { tls_active_pool = prev; }
    ActivePoolGuard(const ActivePoolGuard&) = delete;
    ActivePoolGuard& operator=(const ActivePoolGuard&) = delete;
};

// Per-pass thread cap. Limits threads based on work in a single EDT axis pass.
// This is a C++-level inner cap; the caller-supplied `desired` is already
// capped at the Python level, and at the calibration layer by per-host
// kernel-saturation probes (see probe_cache).
//
// The 60000 / 120000 / 400000 voxel break-points come from per-pass wall-time
// sweeps on three hosts: AMD Threadripper PRO 3995WX (128c Zen 3), Intel Core
// i9-9900K (8c/16t Coffee Lake), and Apple M1 Ultra. Below each threshold,
// cache thrash and dispatch overhead exceed the parallelizable work, so adding
// threads stops helping. Conservative on smaller core counts.
inline size_t compute_threads(size_t desired, size_t total_lines, size_t axis_len) {
    if (desired <= 1 || total_lines <= 1) return 1;

    size_t threads = std::min<size_t>(desired, total_lines);

    // Per-pass workload caps (small workloads -- dispatch overhead dominates)
    const size_t total_work = axis_len * total_lines;
    if (total_work <= 60000) {
        threads = std::min<size_t>(threads, 4);   // small pass: diminishing returns above 4T
    } else if (total_work <= 120000) {
        threads = std::min<size_t>(threads, 8);   // medium pass: cap at 8T
    } else if (total_work <= 400000) {
        threads = std::min<size_t>(threads, 12);  // large pass: cap at 12T
    }

    return std::max<size_t>(1, threads);
}

// =============================================================================
// Per-(P_FIXED) kernel saturation probe with persistent cache
// =============================================================================
//
// Empirical "wisdom" probe (FFTW model). Measures the actual axis-pass kernel
// across a sweep of T values, finds the saturation point with statistical
// rigor, and persists the result so subsequent imports are zero-cost.
//
// Why empirical and not analytical?
//   The kernel saturation point is not derivable from system info alone:
//     - Memory bandwidth saturation (memcpy probe) is a *necessary* but not
//       *sufficient* lower bound. Real kernels need extra threads to hide
//       memory-access latency (RAW hazards, irregular strides, coherency
//       traffic) that pure memcpy doesn't have.
//     - Memory channel count (would let us derive T_sat ≈ K x channels) is
//       not reliably readable without root (dmidecode) or per-CPU lookup
//       tables, and the multiplier K varies by kernel anyway.
//   The honest answer is to measure the actual kernel.
//
// Statistical rigor (vs. the prior best-of-5 + 5%-threshold attempt):
//   - 5 warmup + 25 measurement reps per T → median of 25 (low-variance).
//   - Stop sweep when next T's median doesn't beat current best by at least
//     max(IQR_of_best, 3% of best). The IQR-based threshold makes the
//     stopping rule self-calibrating to per-host noise.
//
// Persistent cache (vs. probe-on-every-import):
//   - Result saved to OS-appropriate cache dir, keyed by hostname + hw
//     concurrency + P_FIXED. First import on a host probes (~1-3 s);
//     subsequent imports read the cached integer (~1 ms).
//
// What's deliberately NOT done:
//   - No memory channel detection / lookup tables / per-platform constants.
//   - No fitted multipliers (no "K=6 x T_mem" magic).
//   - No online adaptation (one probe per host; if the workload is far from
//     the probe's representative size, calib_T may be sub-optimal -- that's
//     a known limitation, fixable later by per-size-class wisdom files).

// Forward declaration so the probe can call the public 2D entry point.
template <typename T>
inline void expand_labels_fused(const T* data, uint32_t* labels_out,
                                const size_t* shape, const float* anisotropy,
                                float p, size_t dims, bool black_border, int parallel);

// Thread-local guard: when true, the kernel-saturation cap inside
// expand_labels_fused is bypassed so the probe (which itself calls
// expand_labels_fused) doesn't recurse back into kernel_saturation_T
// while it's still initializing.
inline thread_local bool tls_probe_active = false;

namespace probe_cache {

inline std::string sanitize(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
        r.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    }
    return r;
}

inline std::string get_hostname() {
    char buf[256] = {};
#ifdef _WIN32
    DWORD sz = sizeof(buf);
    if (GetComputerNameA(buf, &sz)) return std::string(buf);
#else
    if (gethostname(buf, sizeof(buf) - 1) == 0) return std::string(buf);
#endif
    return "unknown";
}

inline std::filesystem::path cache_dir() {
    namespace fs = std::filesystem;
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (!home) return {};
    fs::path base;
#ifdef __APPLE__
    base = fs::path(home) / "Library" / "Caches";
#elif defined(_WIN32)
    const char* lad = std::getenv("LOCALAPPDATA");
    base = lad ? fs::path(lad) : (fs::path(home) / "AppData" / "Local");
#else
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    base = (xdg && *xdg) ? fs::path(xdg) : (fs::path(home) / ".cache");
#endif
    return base / "edt_lp";
}

inline std::filesystem::path cache_file(int P_FIXED, int DIM) {
    const std::string key = sanitize(get_hostname()) +
                            "_hw" + std::to_string(std::thread::hardware_concurrency()) +
                            "_p" + std::to_string(P_FIXED) +
                            "_d" + std::to_string(DIM);
    return cache_dir() / ("probe_" + key + ".txt");
}

inline size_t read_T(const std::filesystem::path& f) {
    std::ifstream in(f);
    if (!in) return 0;
    long long t = 0;
    in >> t;
    return (in && t > 0 && t < 4096) ? size_t(t) : 0;
}

inline void write_T(const std::filesystem::path& f, size_t T) {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(f.parent_path());
        const fs::path tmp = f.parent_path() / (f.filename().string() + ".tmp");
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out) return;
            out << T << "\n";
        }
        fs::rename(tmp, f);  // atomic
    } catch (...) {
        // best-effort; cache miss next time is non-fatal
    }
}

}  // namespace probe_cache

// Pick a representative probe shape from (p_class, ndim). Sizes chosen so
// the working set exceeds L3 (forcing real DRAM traffic) for L1 -- which
// is bandwidth-bound -- while keeping probe time bounded for L2/general,
// which are compute-heavier and saturate at smaller sizes.
inline void probe_shape_for(int p_class, int ndim, size_t shape[3], size_t& pdims) {
    if (ndim == 3) {
        pdims = 3;
        const size_t s = (p_class == 1) ? 256 : 128;
        shape[0] = shape[1] = shape[2] = s;
    } else {
        pdims = 2;
        const size_t s = (p_class == 1) ? 4096 : 1024;
        shape[0] = shape[1] = s;
        shape[2] = 1;
    }
}

// Empirical per-(p_class, ndim) saturation probe. Single function with
// runtime parameters -- mirrors the unified compute path that already
// runs ND for any (D, P_FIXED). Result cached on disk; an in-memory
// memo deduplicates repeat lookups within one process.
inline size_t kernel_saturation_T(int p_class, int ndim) {
    namespace fs = std::filesystem;
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw <= 2) return std::max<unsigned>(1, hw);

    // In-memory memo (avoids re-reading the disk cache on every call)
    static std::mutex memo_mu;
    static std::unordered_map<uint32_t, size_t> memo;
    const uint32_t key = (uint32_t(p_class) & 0xFF) << 8 | (uint32_t(ndim) & 0xFF);
    {
        std::lock_guard<std::mutex> lock(memo_mu);
        auto it = memo.find(key);
        if (it != memo.end()) return it->second;
    }

    const bool dbg = std::getenv("EDT_LP_PROBE_DEBUG") != nullptr;
    const fs::path cf = probe_cache::cache_file(p_class, ndim);

    // Disk cache hit
    if (size_t cached = probe_cache::read_T(cf); cached > 0) {
        if (dbg) std::fprintf(stderr, "[edt_lp probe p=%d d=%d] cache hit T_sat=%zu (%s)\n",
                              p_class, ndim, cached, cf.string().c_str());
        const size_t T = std::min<size_t>(cached, hw);
        std::lock_guard<std::mutex> lock(memo_mu);
        memo[key] = T;
        return T;
    }

    // Fresh probe.
    size_t shape[3]; size_t pdims;
    probe_shape_for(p_class, ndim, shape, pdims);
    size_t total = 1;
    for (size_t d = 0; d < pdims; ++d) total *= shape[d];

    std::unique_ptr<uint32_t[]> mask(new uint32_t[total]);
    std::memset(mask.get(), 0, total * sizeof(uint32_t));
    for (size_t i = 0; i < 50; ++i) {
        size_t off = 0, stride = 1;
        for (size_t d = pdims; d-- > 0; ) {
            const size_t coord = ((i * (29 + d * 13) + 100 * (d + 1)) % (shape[d] - 40)) + 20;
            off += coord * stride;
            stride *= shape[d];
        }
        mask[off] = static_cast<uint32_t>(i + 1);
    }
    std::unique_ptr<uint32_t[]> labels_out(new uint32_t[total]);
    std::memset(labels_out.get(), 0, total * sizeof(uint32_t));

    const float anis[3] = {1.0f, 1.0f, 1.0f};
    const float p_val   = (p_class == 1) ? 1.0f : (p_class == 2 ? 2.0f : 3.0f);

    struct ProbeGuard {
        ProbeGuard()  { tls_probe_active = true; }
        ~ProbeGuard() { tls_probe_active = false; }
    } _guard;

    constexpr int WARMUP = 5;
    const     int REPS   = (ndim == 3) ? 15 : 25;  // 3D is slower per call
    auto bench_T = [&](size_t T) -> double {
        for (int i = 0; i < WARMUP; ++i) {
            expand_labels_fused<uint32_t>(mask.get(), labels_out.get(), shape,
                                          anis, p_val, pdims, false, static_cast<int>(T));
        }
        std::vector<double> samples(REPS);
        for (int rep = 0; rep < REPS; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            expand_labels_fused<uint32_t>(mask.get(), labels_out.get(), shape,
                                          anis, p_val, pdims, false, static_cast<int>(T));
            samples[rep] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        }
        std::sort(samples.begin(), samples.end());
        return samples[REPS / 2];  // median
    };

    size_t best_T = 1;
    double best_med = std::numeric_limits<double>::infinity();
    // Stopping rule: ≥5% improvement to count, two consecutive non-
    // improvements to actually stop. Single noisy T at NUMA placement
    // boundaries doesn't terminate the sweep early.
    constexpr double IMPROVE_FRAC = 0.05;
    constexpr int    STOP_AFTER   = 2;
    int strikes = 0;
    for (size_t T : {size_t(1), size_t(2), size_t(4), size_t(8), size_t(12),
                     size_t(16), size_t(24), size_t(32), size_t(48), size_t(64),
                     size_t(96), size_t(128)}) {
        if (T > hw) break;
        const double med = bench_T(T);
        if (dbg) std::fprintf(stderr, "[edt_lp probe p=%d d=%d] T=%zu: %.2f ms\n",
                              p_class, ndim, T, med * 1000.0);
        const bool first = std::isinf(best_med);
        if (first || med < best_med * (1.0 - IMPROVE_FRAC)) {
            best_T = T; best_med = med; strikes = 0;
        } else if (++strikes >= STOP_AFTER) {
            break;
        }
    }

    if (dbg) std::fprintf(stderr, "[edt_lp probe p=%d d=%d] FINAL T_sat=%zu\n",
                          p_class, ndim, best_T);
    probe_cache::write_T(cf, best_T);
    const size_t T = std::max<size_t>(2, best_T);
    {
        std::lock_guard<std::mutex> lock(memo_mu);
        memo[key] = T;
    }
    return T;
}

// Per-thread buffer cache for expand_labels -- avoids repeated allocation /
// page-fault overhead on repeated calls. Stored thread_local so two Python
// threads calling expand_labels concurrently each see their own cache;
// freed when the thread exits. expand_cache().clear() is exposed from
// Python as edt_lp.clear_expand_cache() for expert release. Mirrors
// src/edt.hpp.
struct ExpandBufCache {
    static constexpr int N_SLOTS = 8;
    void* bufs[N_SLOTS] = {};
    size_t sizes[N_SLOTS] = {};

    void* get(int slot, size_t bytes) {
        if (bytes <= sizes[slot]) return bufs[slot];
        std::free(bufs[slot]);
        bufs[slot] = std::malloc(bytes);
        sizes[slot] = bytes;
        return bufs[slot];
    }
    void clear() {
        for (int i = 0; i < N_SLOTS; i++) {
            std::free(bufs[i]);
            bufs[i] = nullptr;
            sizes[i] = 0;
        }
    }
    ~ExpandBufCache() {
        clear();
    }
};

inline ExpandBufCache& expand_cache() {
    thread_local ExpandBufCache cache;
    return cache;
}

inline void clear_expand_cache() {
    expand_cache().clear();
}

// Distribute [0, total) into up to max_chunks chunks across threads.
// Calls work(begin, end) directly when threads==1; otherwise via shared pool.
// Uses atomic work-stealing: each thread claims chunks via fetch_add.
// Blocks until all chunks complete. The pool is the per-call-chain
// active pool when set (see ActivePoolGuard), else looked up by
// thread count.
template <typename F>
inline void dispatch_parallel(size_t threads, size_t total, size_t max_chunks, F work) {
    if (threads <= 1 || total == 0) {
        work(size_t(0), total);
        return;
    }
    const size_t n_chunks = std::min(max_chunks, total);
    const size_t chunk_sz = (total + n_chunks - 1) / n_chunks;
    std::atomic<size_t> next{0};
    edt::ForkJoinPool& pool = (tls_active_pool != nullptr)
        ? *tls_active_pool
        : shared_pool_for(threads);
    pool.parallel([&]() {
        size_t idx;
        while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < n_chunks) {
            const size_t begin = idx * chunk_sz;
            const size_t end = std::min(total, begin + chunk_sz);
            work(begin, end);
        }
    });
}

// Precomputed per-pass iteration layout for an EDT axis pass.
// Gathers all "other" (non-axis) dimensions and their strides, and
// exposes for_each_line() to iterate every scanline in a slice range.
struct AxisPassInfo {
    size_t num_other = 0;   // number of non-axis dims
    size_t other_extents[EDT_MAX_DIMS];   // extents of non-axis dims (in shape order)
    size_t other_strides[EDT_MAX_DIMS];   // strides of non-axis dims
    size_t total_lines = 1; // product of all other extents
    size_t first_extent  = 1;  // extent of first other dim  (parallelized over)
    size_t first_stride  = 0;  // stride of first other dim
    size_t rest_prod  = 1;  // product of other_extents[1..num_other-1]

    AxisPassInfo(const size_t* shape, const size_t* strides,
                 size_t dims, size_t axis) {
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            other_extents[num_other] = shape[d];
            other_strides[num_other] = strides[d];
            total_lines *= shape[d];
            num_other++;
        }
        if (num_other > 0) {
            first_extent = other_extents[0];
            first_stride = other_strides[0];
            for (size_t d = 1; d < num_other; d++)
                rest_prod *= other_extents[d];
        }
    }

    // Call fn(base) for every scanline starting offset whose first-other-dim
    // index falls in [begin, end).  Handles 1D and ND sub-iteration.
    //
    // For the ND branch, coords[1..num_other-1] are guaranteed to return to
    // all-zeros after exactly rest_prod inner iterations, so they are
    // initialized once and not re-initialized per i0 row.
    template <typename F>
    void for_each_line(size_t begin, size_t end, F fn) const {
        if (num_other <= 1) {
            // Simple path: one scanline per first-dim row
            for (size_t i0 = begin; i0 < end; i0++)
                fn(i0 * first_stride);
        } else {
            // ND path: iterate the inner dims with a multi-dim counter.
            // coords reused across i0 rows; invariant: all-zero at start of each row.
            size_t coords[EDT_MAX_DIMS] = {};
            for (size_t i0 = begin; i0 < end; i0++) {
                size_t base = i0 * first_stride;
                for (size_t i = 0; i < rest_prod; i++) {
                    fn(base);
                    for (size_t d = 1; d < num_other; d++) {
                        coords[d]++;
                        base += other_strides[d];
                        if (coords[d] < other_extents[d]) break;
                        base -= coords[d] * other_strides[d];
                        coords[d] = 0;
                    }
                }
            }
        }
    }
};

//=============================================================================
// Lp cost helpers -- templated for compile-time specialization
//
// P_FIXED: 0 = general (runtime p), 2 = L2 specialization, etc.
// When P_FIXED != 0, the compiler eliminates all branches and the p
// parameter is ignored, producing code identical to hand-written L2.
//=============================================================================

// Compute w^p (precomputed per-axis)
inline float wp_from_anisotropy(float anisotropy, float p) {
    if (p == 2.0f) return anisotropy * anisotropy;
    if (p == 1.0f) return anisotropy;
    return std::pow(anisotropy, p);
}

inline float sq(float x) { return x * x; }

// Compute wp * |d|^p
template <int P_FIXED = 0>
inline float lp_cost(float wp, float d, float p) {
    if constexpr (P_FIXED == 2) { return wp * sq(d); }
    else if constexpr (P_FIXED == 1) { return wp * std::abs(d); }
    else {
        const float ad = std::abs(d);
        if (p == 1.0f) return wp * ad;
        if (p == 2.0f) return wp * ad * ad;
        if (p == 3.0f) return wp * ad * ad * ad;
        if (p == 4.0f) { const float a2 = ad * ad; return wp * a2 * a2; }
        return wp * std::pow(ad, p);
    }
}

// Compute wp * d^p for d >= 0 (avoids abs)
template <int P_FIXED = 0>
inline float lp_cost_pos(float wp, float d, float p) {
    if constexpr (P_FIXED == 2) { return wp * d * d; }
    else if constexpr (P_FIXED == 1) { return wp * d; }
    else {
        if (p == 1.0f) return wp * d;
        if (p == 2.0f) return wp * d * d;
        if (p == 3.0f) return wp * d * d * d;
        if (p == 4.0f) { const float d2 = d * d; return wp * d2 * d2; }
        return wp * std::pow(d, p);
    }
}

// Compute |d|^p in double (for intersection bisection)
template <int P_FIXED = 0>
inline double lp_dpow(double ad, double p) {
    if constexpr (P_FIXED == 2) { return ad * ad; }
    else if constexpr (P_FIXED == 1) { return ad; }
    else {
        if (p == 1.0) return ad;
        if (p == 2.0) return ad * ad;
        if (p == 3.0) return ad * ad * ad;
        if (p == 4.0) { double a2 = ad * ad; return a2 * a2; }
        return std::pow(ad, p);
    }
}

//=============================================================================
// Lp lower-envelope intersection
//
// Finds the crossing point of two cost curves:
//   F_a(x) = ff[a] + wp * |x - a|^p
//   F_b(x) = ff[b] + wp * |x - b|^p
//
// p=1: closed-form V-shape intersection
// p=2: closed-form parabola intersection
// general p: bisection on h(x) = F_a(x) - F_b(x)
//=============================================================================

inline float lp_intersect(
    const float* ff, int a, int b,
    float wp, float p, int n
) {
    // p=1: V-shape intersection
    if (p == 1.0f) {
        const double dfa = double(ff[a]);
        const double dfb = double(ff[b]);
        const double wd = double(wp) * double(b - a);
        if (dfa - dfb > wd) return -std::numeric_limits<float>::infinity();
        if (dfb - dfa > wd) return  std::numeric_limits<float>::infinity();
        return float((dfb - dfa) / (2.0 * double(wp)) + double(a + b) * 0.5);
    }
    // p=2: closed-form parabola intersection
    if (p == 2.0f) {
        const double d1 = double(b - a) * double(wp);
        return float((double(ff[b]) - double(ff[a]) + d1 * double(a + b)) / (2.0 * d1));
    }
    // General Lp: bisection
    const double dp = double(p);
    const double dwp = double(wp);
    if (ff[a] == ff[b]) return float(a + b) * 0.5f;
    auto h = [&](double x) -> double {
        return double(ff[a]) - double(ff[b])
             + dwp * (lp_dpow<0>(std::abs(x - a), dp)
                    - lp_dpow<0>(std::abs(x - b), dp));
    };
    double lo = double(a), hi = double(b);
    double h_lo = h(lo), h_hi = h(hi);
    double span = double(b - a) + 1.0;
    while (h_lo > 0 && lo > double(-n)) { lo -= span; h_lo = h(lo); span *= 2.0; }
    span = double(b - a) + 1.0;
    while (h_hi < 0 && hi < double(2 * n)) { hi += span; h_hi = h(hi); span *= 2.0; }
    if (h_lo > 0) return -std::numeric_limits<float>::infinity();
    if (h_hi < 0) return  std::numeric_limits<float>::infinity();
    for (int iter = 0; iter < 50; iter++) {
        double mid = (lo + hi) * 0.5;
        if (h(mid) <= 0.0) lo = mid; else hi = mid;
    }
    return float((lo + hi) * 0.5);
}

//-----------------------------------------------------------------------------
// Pass 0 from Graph
//
// Reads the voxel connectivity graph and computes the Rosenfeld-Pfaltz
// 1D Lp distance (pass 0) directly. Does not write segment labels.
//-----------------------------------------------------------------------------

template <int P_FIXED, typename GRAPH_T>
inline void lp_edt_1d_from_graph_direct(
    const GRAPH_T* graph,
    float* d,
    const int n,
    const int64_t stride,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const float p,
    const bool black_border
) {
    if (n <= 0) return;

    const float wp = wp_from_anisotropy(anisotropy, p);
    int i = 0;

    while (i < n) {
        // Check if this voxel is background (graph == 0)
        if (graph[i * stride] == 0) {
            d[i * stride] = 0.0f;
            i++;
            continue;
        }

        // Foreground: find segment extent using connectivity bits
        const int seg_start = i;
        GRAPH_T edge = graph[i * stride];
        i++;

        // Follow connectivity along axis
        while (i < n && (edge & axis_bit)) {
            edge = graph[i * stride];
            if (edge == 0) break;
            i++;
        }
        const int seg_len = i - seg_start;

        // Compute Lp EDT for this segment.
        // Store Lp distances directly; subsequent passes accumulate in-place.
        const bool left_border = (seg_start > 0) || black_border;
        const bool right_border = (i < n) || black_border;

        // Forward pass: Lp distance from left border
        if (left_border) {
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = lp_cost_pos<P_FIXED>(wp, float(k + 1), p);
            }
        } else {
            const float inf = std::numeric_limits<float>::infinity();
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = inf;
            }
        }

        // Backward pass: take min with Lp distance from right border
        if (right_border) {
            for (int k = seg_len - 1; k >= 0; k--) {
                const float v_right = lp_cost_pos<P_FIXED>(wp, float(seg_len - k), p);
                const int64_t idx = (seg_start + k) * stride;
                if (v_right < d[idx]) {
                    d[idx] = v_right;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Pass 0 from Graph (parallel dispatch)
//-----------------------------------------------------------------------------

template <int P_FIXED, typename GRAPH_T>
inline void edt_pass0_from_graph_lp_parallel(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const float p,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;
    const int n = int(shape[axis]);
    const int64_t axis_stride = strides[axis];
    if (n == 0) return;

    const AxisPassInfo info(shape, strides, dims, axis);
    const size_t threads = compute_threads(parallel, info.total_lines, (size_t)n);

    auto process_range = [&](size_t begin, size_t end) {
        info.for_each_line(begin, end, [&](size_t base) {
            lp_edt_1d_from_graph_direct<P_FIXED, GRAPH_T>(
                graph + base, output + base,
                n, axis_stride, axis_bit, anisotropy, p, black_border
            );
        });
    };

    dispatch_parallel(threads, info.first_extent, threads, process_range);
}

/*
 * Lower Envelope Pass from Graph
 *
 * Reads voxel connectivity graph directly; no separate segment label
 * building step. For p=2 uses closed-form parabola intersection; for
 * general p uses bisection on the cost difference function.
 */

template <int P_FIXED, typename GRAPH_T>
inline void lp_edt_1d_envelope_from_graph_ws(
    const GRAPH_T* graph,
    float* f,
    const int n,
    const int64_t stride,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const float p,
    const bool black_border,
    int* v,
    float* ff,
    float* ranges
) {
    if (n <= 0) return;

    constexpr int SMALL_THRESHOLD = 8;
    const float wp = wp_from_anisotropy(anisotropy, p);

    // Fast path for small segments: O(n^2) brute force
    auto process_small_run = [&](int start, int len, bool left_border, bool right_border) {
        float original[SMALL_THRESHOLD];
        for (int q = 0; q < len; ++q) {
            original[q] = f[(start + q) * stride];
        }
        for (int j = 0; j < len; ++j) {
            float best = original[j];
            if (left_border) {
                const float cap_left = lp_cost_pos<P_FIXED>(wp, float(j + 1), p);
                if (cap_left < best) best = cap_left;
            }
            if (right_border) {
                const float cap_right = lp_cost_pos<P_FIXED>(wp, float(len - j), p);
                if (cap_right < best) best = cap_right;
            }
            for (int q = 0; q < len; ++q) {
                const float candidate = original[q] + lp_cost<P_FIXED>(wp, float(j - q), p);
                if (candidate < best) best = candidate;
            }
            f[(start + j) * stride] = best;
        }
    };

    // Lower envelope for larger segments
    auto process_large_run = [&](int start, int len, bool left_border, bool right_border) {
        // Copy to workspace
        for (int i = 0; i < len; i++) {
            ff[i] = f[(start + i) * stride];
        }

        // Skip INF-valued sources when building the lower envelope.
        // INF sources never win the minimum, and INF - INF = NaN corrupts
        // the intersection formula, leaving all subsequent ranges as NaN
        // and preventing the output pass from ever advancing k.
        int first_src = 0;
        while (first_src < len && std::isinf(ff[first_src])) first_src++;

        int k = 0;
        // If all sources are INF, fall back to v[0]=0 with ff[0]=INF so
        // the output pass correctly produces INF (borders still applied).
        v[0] = (first_src < len) ? first_src : 0;
        ranges[0] = -std::numeric_limits<float>::infinity();
        ranges[1] = std::numeric_limits<float>::infinity();

        auto intersect = [&](int a, int b) -> float {
            if constexpr (P_FIXED == 1 || P_FIXED == 2) {
                return lp_intersect(ff, a, b, wp, float(P_FIXED), len);
            } else {
                return lp_intersect(ff, a, b, wp, p, len);
            }
        };

        float s;
        const int loop_start = (first_src < len) ? first_src + 1 : len;
        for (int i = loop_start; i < len; i++) {
            if (std::isinf(ff[i])) continue;  // INF never wins the minimum

            s = intersect(v[k], i);
            while (k > 0 && s <= ranges[k]) {
                k--;
                s = intersect(v[k], i);
            }

            k++;
            v[k] = i;
            ranges[k] = s;
            ranges[k + 1] = std::numeric_limits<float>::infinity();
        }

        // Output pass: use specialized loops to avoid per-iteration conditionals
        k = 0;
        if (left_border && right_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                const float envelope = lp_cost<P_FIXED>(wp, float(i - v[k]), p) + ff[v[k]];
                const float border = std::fminf(
                    lp_cost_pos<P_FIXED>(wp, float(i + 1), p),
                    lp_cost_pos<P_FIXED>(wp, float(len - i), p)
                );
                f[(start + i) * stride] = std::fminf(border, envelope);
            }
        } else if (left_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = std::fminf(
                    lp_cost_pos<P_FIXED>(wp, float(i + 1), p),
                    lp_cost<P_FIXED>(wp, float(i - v[k]), p) + ff[v[k]]
                );
            }
        } else if (right_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = std::fminf(
                    lp_cost_pos<P_FIXED>(wp, float(len - i), p),
                    lp_cost<P_FIXED>(wp, float(i - v[k]), p) + ff[v[k]]
                );
            }
        } else {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = lp_cost<P_FIXED>(wp, float(i - v[k]), p) + ff[v[k]];
            }
        }
    };

    // Scan graph to find foreground segments (single pass)
    // Key insight: segment boundary when prev didn't connect forward (!(prev & axis_bit))
    // Background has graph=0, so axis_bit check handles both cases

    // Skip leading background
    int i = 0;
    while (i < n && graph[i * stride] == 0) i++;
    if (i >= n) return;

    int seg_start = i;
    GRAPH_T g = graph[i * stride];
    i++;

    while (i < n) {
        const GRAPH_T prev_g = g;
        g = graph[i * stride];

        // Boundary if previous didn't connect forward
        // Note: axis_bit encodes connectivity, so if current is background,
        // previous won't have axis_bit set (labels differ). No need for g==0 check.
        if (!(prev_g & axis_bit)) {
            // Process segment [seg_start, i)
            const int seg_len = i - seg_start;
            const bool left_border = (seg_start > 0) || black_border;
            if (seg_len <= SMALL_THRESHOLD) {
                process_small_run(seg_start, seg_len, left_border, true);
            } else {
                process_large_run(seg_start, seg_len, left_border, true);
            }

            // Skip background, find next segment start
            while (i < n && graph[i * stride] == 0) i++;
            if (i >= n) return;
            seg_start = i;
            g = graph[i * stride];
        }
        i++;
    }

    // Final segment
    {
        const int seg_len = n - seg_start;
        const bool left_border = (seg_start > 0) || black_border;
        if (seg_len <= SMALL_THRESHOLD) {
            process_small_run(seg_start, seg_len, left_border, black_border);
        } else {
            process_large_run(seg_start, seg_len, left_border, black_border);
        }
    }
}

//-----------------------------------------------------------------------------
// Lower Envelope Pass from Graph (parallel dispatch)
//-----------------------------------------------------------------------------

template <int P_FIXED, typename GRAPH_T>
inline void edt_pass_envelope_from_graph_lp_parallel(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const float p,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;
    const int n = int(shape[axis]);
    const int64_t axis_stride = strides[axis];
    if (n == 0) return;

    const AxisPassInfo info(shape, strides, dims, axis);
    const size_t threads = compute_threads(parallel, info.total_lines, (size_t)n);

    auto process_range = [&](size_t begin, size_t end) {
        std::vector<int> v(n);
        std::vector<float> ff(n), ranges(n + 1);

        info.for_each_line(begin, end, [&](size_t base) {
            lp_edt_1d_envelope_from_graph_ws<P_FIXED, GRAPH_T>(
                graph + base, output + base,
                n, axis_stride, axis_bit, anisotropy, p, black_border,
                v.data(), ff.data(), ranges.data()
            );
        });
    };

    dispatch_parallel(threads, info.first_extent, threads, process_range);
}

//-----------------------------------------------------------------------------
// Full EDT from Voxel Graph
//-----------------------------------------------------------------------------

template <int P_FIXED, typename GRAPH_T>
inline void edtp_from_graph_impl(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const float p,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;
    if (dims > EDT_MAX_DIMS) {
        throw std::invalid_argument("EDT (Lp) supports at most 32 dimensions");
    }

    size_t total = 1;
    size_t strides[EDT_MAX_DIMS];
    for (size_t d = dims; d-- > 0;) {
        strides[d] = total;
        total *= shape[d];
    }
    if (total == 0) return;

    // Pass 0: innermost axis (always bit 1 in the graph encoding)
    {
        const size_t axis = dims - 1;
        const GRAPH_T axis_bit = GRAPH_T(1) << 1;
        edt_pass0_from_graph_lp_parallel<P_FIXED, GRAPH_T>(
            graph, output,
            shape, strides, dims, axis, axis_bit,
            anisotropy[axis], p, black_border, parallel
        );
    }

    // Lower envelope passes: axes dims-2 down to 0
    for (size_t axis = dims - 1; axis-- > 0;) {
        const GRAPH_T axis_bit = GRAPH_T(1) << (2 * (dims - 1 - axis) + 1);
        edt_pass_envelope_from_graph_lp_parallel<P_FIXED, GRAPH_T>(
            graph, output,
            shape, strides, dims, axis, axis_bit,
            anisotropy[axis], p, black_border, parallel
        );
    }
}

// Dispatch wrapper: picks compile-time specialization for common p values
template <typename GRAPH_T>
inline void edtp_from_graph(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const float p,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims > EDT_MAX_DIMS) {
        throw std::invalid_argument("edtp_from_graph supports at most 32 dimensions");
    }
    if (p == 2.0f) {
        edtp_from_graph_impl<2, GRAPH_T>(graph, output, shape, anisotropy, p, dims, black_border, parallel);
    } else if (p == 1.0f) {
        edtp_from_graph_impl<1, GRAPH_T>(graph, output, shape, anisotropy, p, dims, black_border, parallel);
    } else {
        edtp_from_graph_impl<0, GRAPH_T>(graph, output, shape, anisotropy, p, dims, black_border, parallel);
    }
}

//-----------------------------------------------------------------------------
// Build connectivity graph from labels (single-pass, unified ND algorithm)
//
// The graph is metric-independent -- identical to the src/ L2 implementation.
// 1D: dedicated linear scan.
// 2D+: unified ND path (chunk-based background skipping on innermost dim).
// Fixed internal arrays support up to 32D.
//-----------------------------------------------------------------------------

template <typename T, typename GRAPH_T = uint8_t>
inline void build_connectivity_graph(
    const T* labels,
    GRAPH_T* graph,
    const size_t* shape,
    const size_t dims,
    const int parallel
) {
    if (dims == 0) return;

    size_t total = 1;
    for (size_t d = 0; d < dims; d++) total *= shape[d];
    if (total == 0) return;

    const int threads = std::max(1, parallel);
    constexpr GRAPH_T fg_bit = 0b00000001;  // Foreground bit (bit 0)

    //-------------------------------------------------------------------------
    // 1D path: simple linear scan
    //-------------------------------------------------------------------------
    if (dims == 1) {
        const size_t n = shape[0];
        constexpr GRAPH_T axis_bit = 0b00000010;  // axis 0 bit for 1D

        auto process_1d = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; i++) {
                const T label = labels[i];
                GRAPH_T g = (label != 0) ? fg_bit : 0;
                if (label != 0 && i + 1 < n && labels[i + 1] == label) {
                    g |= axis_bit;
                }
                graph[i] = g;
            }
        };
        dispatch_parallel((size_t)threads, n, (size_t)threads, process_1d);
        return;
    }

    //-------------------------------------------------------------------------
    // Unified ND path for 2D+ - parallelize over first dimension with
    // chunk-based background skipping on the inner loop
    //-------------------------------------------------------------------------
    int64_t strides[EDT_MAX_DIMS];
    int64_t shape64[EDT_MAX_DIMS];
    GRAPH_T axis_bits[EDT_MAX_DIMS];
    {
        int64_t s = 1;
        for (size_t d = dims; d-- > 0;) {
            strides[d] = s;
            shape64[d] = shape[d];
            s *= shape64[d];
        }
        for (size_t d = 0; d < dims; d++) {
            axis_bits[d] = GRAPH_T(1) << (2 * (dims - 1 - d) + 1);
        }
    }

    const int64_t first_extent = shape64[0];
    const int64_t first_stride = strides[0];
    const int64_t last_extent = shape64[dims - 1];
    const GRAPH_T last_bit = axis_bits[dims - 1];
    const GRAPH_T first_bit = axis_bits[0];

    // Middle dimensions product (dims 1 to dims-2); = 1 for 2D (empty product)
    int64_t mid_product = 1;
    for (size_t d = 1; d + 1 < dims; d++) {
        mid_product *= shape64[d];
    }

    // Number of middle dimensions (dims between first and last); 0 for 2D, 1 for 3D, etc.
    // Safe: dims >= 2 is guaranteed by the dims == 1 early return above.
    const size_t num_mid = dims - 2;

    constexpr int64_t CHUNK = 8;  // chunk size for background-skipping in inner loop

    // Process range of first dimension (outer loop) for 2D+
    auto process_dim0_range = [&](int64_t d0_start, int64_t d0_end) {
        // Thread-local storage for precomputed middle dimension info
        const T* mid_neighbor_row[30];  // Neighbor row pointers for middle dims (max 30 for 32D)
        bool mid_can_check[30];         // Whether we can check each mid neighbor
        GRAPH_T mid_bits[30];           // Bit to set for each mid dimension (constant per call)
        for (size_t mid = 0; mid < num_mid; mid++)
            mid_bits[mid] = axis_bits[mid + 1];

        for (int64_t d0 = d0_start; d0 < d0_end; d0++) {
            const int64_t base0 = d0 * first_stride;
            const bool can_d0 = (d0 + 1 < first_extent);

            // Iterate middle dimensions (dims 1 to dims-2)
            int64_t mid_coords[30] = {0};  // For dims 1..dims-2 (max 30 for 32D)
            int64_t mid_offset = 0;

            for (int64_t mid = 0; mid < mid_product; mid++) {
                const int64_t base = base0 + mid_offset;

                // Precompute row pointers for tight inner loop
                const T* row = labels + base;
                GRAPH_T* rowg = graph + base;
                const T* row_d0_next = can_d0 ? (labels + base + first_stride) : nullptr;

                // Precompute middle dimension neighbor info BEFORE inner loop
                for (size_t mid = 0; mid < num_mid; mid++) {
                    const size_t d = mid + 1;  // Actual dimension index
                    mid_can_check[mid] = (mid_coords[mid] + 1 < shape64[d]);
                    mid_neighbor_row[mid] = mid_can_check[mid] ? (labels + base + strides[d]) : nullptr;
                }

                // Inner loop over last dimension with chunk-based background skipping
                int64_t x = 0;
                const int64_t chunk_end = last_extent - (last_extent % CHUNK);
                for (; x < chunk_end; x += CHUNK) {
                    T any_fg = row[x]   | row[x+1] | row[x+2] | row[x+3] |
                               row[x+4] | row[x+5] | row[x+6] | row[x+7];
                    if (any_fg == 0) {
                        std::memset(rowg + x, 0, CHUNK * sizeof(GRAPH_T));
                    } else {
                        for (int64_t i = 0; i < CHUNK; i++) {
                            const int64_t xi = x + i;
                            const T label = row[xi];
                            GRAPH_T g = (label != 0) ? fg_bit : 0;
                            if (label != 0) {
                                if (xi + 1 < last_extent && row[xi + 1] == label) g |= last_bit;
                                if (can_d0 && row_d0_next[xi] == label) g |= first_bit;
                                for (size_t mid = 0; mid < num_mid; mid++) {
                                    if (mid_can_check[mid] && mid_neighbor_row[mid][xi] == label) g |= mid_bits[mid];
                                }
                            }
                            rowg[xi] = g;
                        }
                    }
                }
                for (; x < last_extent; x++) {
                    const T label = row[x];
                    GRAPH_T g = (label != 0) ? fg_bit : 0;
                    if (label != 0) {
                        if (x + 1 < last_extent && row[x + 1] == label) g |= last_bit;
                        if (can_d0 && row_d0_next[x] == label) g |= first_bit;
                        for (size_t mid = 0; mid < num_mid; mid++) {
                            if (mid_can_check[mid] && mid_neighbor_row[mid][x] == label) g |= mid_bits[mid];
                        }
                    }
                    rowg[x] = g;
                }

                // Increment mid coords; skip on last mid iteration
                // (mid_coords is re-initialized for each d0 row, so
                //  the final increment before that reset is always wasted)
                if (mid + 1 < mid_product) {
                    for (size_t d = dims - 2; d >= 1; d--) {
                        mid_coords[d - 1]++;
                        mid_offset += strides[d];
                        if (mid_coords[d - 1] < shape64[d]) break;
                        mid_offset -= mid_coords[d - 1] * strides[d];
                        mid_coords[d - 1] = 0;
                    }
                }
            }
        }
    };

    dispatch_parallel((size_t)threads, (size_t)first_extent, (size_t)threads,
        [&](size_t begin, size_t end) { process_dim0_range((int64_t)begin, (int64_t)end); });
}

//-----------------------------------------------------------------------------
// Fused labels-to-EDT: Build graph internally, run EDT, free graph
// This is more efficient than separate Python calls because:
// 1. No Python/Cython overhead between build and EDT
// 2. Graph memory is allocated and freed in C++ (faster)
// 3. Thread pool is already warm from graph build
//-----------------------------------------------------------------------------

// Internal: allocate graph of type GRAPH_T, build connectivity, run EDT.
// `total` (precomputed by caller) is passed to avoid recomputing for the allocation.
template <typename T, typename GRAPH_T>
inline void _edtp_fused_typed(
    const T* labels, float* output, const size_t* shape,
    const float* anisotropy, const float p, const size_t dims,
    const bool black_border, const int parallel, const size_t total
) {
    std::unique_ptr<GRAPH_T[]> graph(new GRAPH_T[total]);
    build_connectivity_graph<T, GRAPH_T>(labels, graph.get(), shape, dims, parallel);
    edtp_from_graph<GRAPH_T>(graph.get(), output, shape, anisotropy, p, dims, black_border, parallel);
}

template <typename T>
inline void edtp_from_labels_fused(
    const T* labels,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const float p,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;
    size_t total = 1;
    for (size_t d = 0; d < dims; d++) total *= shape[d];
    if (total == 0) return;

    // Graph type: smallest unsigned integer fitting 2*(dims-1)+1 bits.
    // uint8 <=4D (max bit 7), uint16 <=8D (max bit 15),
    // uint32 <=16D (max bit 31), uint64 <=32D (max bit 63).
    if      (dims <=  4) _edtp_fused_typed<T, uint8_t> (labels, output, shape, anisotropy, p, dims, black_border, parallel, total);
    else if (dims <=  8) _edtp_fused_typed<T, uint16_t>(labels, output, shape, anisotropy, p, dims, black_border, parallel, total);
    else if (dims <= 16) _edtp_fused_typed<T, uint32_t>(labels, output, shape, anisotropy, p, dims, black_border, parallel, total);
    else                 _edtp_fused_typed<T, uint64_t>(labels, output, shape, anisotropy, p, dims, black_border, parallel, total);
}

//=============================================================================
// Expand labels: blocked-transpose pipeline with seed-skipping
//=============================================================================

// Sort all axes by stride ascending (innermost first)
inline void _expand_sort_axes(
    size_t* paxes,
    const size_t* shape,
    const size_t* strides,
    const size_t dims
) {
    for (size_t d = 0; d < dims; ++d) paxes[d] = d;
    for (size_t i = 1; i < dims; ++i) {
        size_t key = paxes[i];
        int j = (int)i - 1;
        while (j >= 0 && (strides[paxes[j]] > strides[key] ||
               (strides[paxes[j]] == strides[key] && shape[paxes[j]] < shape[key]))) {
            paxes[j + 1] = paxes[j];
            --j;
        }
        paxes[j + 1] = key;
    }
}

template <typename T>
inline bool _expand_1d_setup(
    const T* data, const size_t n,
    std::vector<size_t>& seeds, std::vector<double>& mids
) {
    for (size_t i = 0; i < n; ++i)
        if (data[i] != 0) seeds.push_back(i);
    if (seeds.empty()) return false;
    mids.resize(seeds.size() - 1);
    for (size_t i = 0; i < mids.size(); ++i)
        mids[i] = (seeds[i] + seeds[i + 1]) * 0.5;
    return true;
}

//-----------------------------------------------------------------------------
// Unified pass 0 across all p (any-D, any-P_FIXED).
//   P_FIXED == 1: L1 raster sweep (init+forward+backward fused). D may be
//                 int32_t (faster on x86 -- int add 1-cycle vs FP add 3-cycle)
//                 or float (when anisotropy is non-integer).
//   P_FIXED == 2: parabolic envelope, closed-form intersections (one
//                 instantiation of lp_cost<2>). D=float.
//   P_FIXED == 0: parabolic envelope, general bisection-based intersection.
//                 D=float, runtime p used.
//
// Pass 0 specialization: for the FIRST axis, all seeds have dist=0 and the
// envelope reduces to midpoints. For L1 this is identical to the raster sweep
// outcome (faster path). For parabolic, the simplification holds for any p.
//-----------------------------------------------------------------------------

template <typename D, int P_FIXED>
inline void _expand_pass0(
    uint32_t* RESTRICT lbl,
    D* RESTRICT dist,
    const size_t n,
    const size_t num_lines,
    const D anis,
    const float p,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);

    if constexpr (P_FIXED == 1) {
        // L1 raster sweep: fused init + forward + backward. D is int32_t
        // (integer anisotropy) or float. anis is the per-step increment.
        constexpr D HUGE_DIST = std::numeric_limits<D>::max() / 4;
        auto process_chunk = [&](size_t begin, size_t end) {
            for (size_t line = begin; line < end; ++line) {
                uint32_t* ll = lbl + line * n;
                D*        dd = dist + line * n;

                // Forward sweep with branchless update: compute both
                // candidates and select. clang/gcc + MSVC all lower the
                // if/else over locals + unconditional store to csel/cmov +
                // store, avoiding a hard-to-predict branch in the store path.
                D prev_d = black_border ? D(0) : HUGE_DIST;
                uint32_t prev_l = 0;
                for (size_t i = 0; i < n; ++i) {
                    const uint32_t init_l = ll[i];
                    const D init_d = (init_l != 0) ? D(0) : HUGE_DIST;
                    const D cd = prev_d + anis;
                    D out_d; uint32_t out_l;
                    if (cd < init_d) { out_d = cd;     out_l = prev_l; }
                    else             { out_d = init_d; out_l = init_l; }
                    dd[i] = out_d;
                    ll[i] = out_l;
                    prev_d = out_d;
                    prev_l = out_l;
                }
                if (black_border) {
                    if (anis < dd[n - 1]) { dd[n - 1] = anis; ll[n - 1] = 0; }
                }
                for (size_t i = n - 1; i-- > 0; ) {
                    const D cd = dd[i + 1] + anis;
                    if (cd < dd[i]) { dd[i] = cd; ll[i] = ll[i + 1]; }
                }
            }
        };
        dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
    } else {
        // P_FIXED == 2 (closed-form) or 0 (general): parabolic envelope.
        static_assert(std::is_same_v<D, float>,
                      "Parabolic envelope requires D=float");
        const float wp = wp_from_anisotropy((float)anis, p);
        constexpr float HUGE_DIST_F = std::numeric_limits<float>::max() / 4.0f;
        float* dist_f = reinterpret_cast<float*>(dist);

        auto process_chunk = [&](size_t begin, size_t end) {
            std::vector<int> v(n);
            std::vector<uint32_t> lbl_save(n);

            for (size_t line = begin; line < end; ++line) {
                uint32_t* ll = lbl + line * n;
                float*    dd = dist_f + line * n;

                // Collect seed positions and init dist
                int n_seeds = 0;
                bool any_nonseed = false;
                for (size_t j = 0; j < n; ++j) {
                    if (ll[j] != 0) {
                        dd[j] = 0.0f;
                        v[n_seeds++] = (int)j;
                    } else {
                        dd[j] = HUGE_DIST_F;
                        any_nonseed = true;
                    }
                }
                if (!any_nonseed) continue;  // all seeds
                if (n_seeds == 0) {
                    if (black_border) {
                        for (size_t i = 0; i < n; ++i)
                            dd[i] = std::fminf(
                                lp_cost_pos<P_FIXED>(wp, float(i + 1), p),
                                lp_cost_pos<P_FIXED>(wp, float(n - i), p));
                    }
                    continue;
                }

                std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));

                int k = 0;
                if (black_border) {
                    for (size_t i = 0; i < n; ++i) {
                        while (k + 1 < n_seeds &&
                               (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                        const float envelope = lp_cost<P_FIXED>(wp, float((int)i - v[k]), p);
                        const float border = std::fminf(
                            lp_cost_pos<P_FIXED>(wp, float(i + 1), p),
                            lp_cost_pos<P_FIXED>(wp, float(n - i), p));
                        dd[i] = std::fminf(border, envelope);
                        ll[i] = lbl_save[v[k]];
                    }
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        while (k + 1 < n_seeds &&
                               (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                        dd[i] = lp_cost<P_FIXED>(wp, float((int)i - v[k]), p);
                        ll[i] = lbl_save[v[k]];
                    }
                }
            }
        };
        dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
    }
}

template <typename INDEX>
inline void _expand_pass0_feat(
    uint32_t* RESTRICT lbl,
    float* RESTRICT dist,
    INDEX* RESTRICT feat,
    const size_t n,
    const size_t num_lines,
    const float anis,
    const float p,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);
    const float wp = wp_from_anisotropy(anis, p);
    const float HUGE_DIST = std::numeric_limits<float>::max() / 4.0f;

    auto process_chunk = [&](size_t begin, size_t end) {
        std::vector<int> v(n);
        std::vector<uint32_t> lbl_save(n);
        std::vector<INDEX> feat_save(n);

        for (size_t line = begin; line < end; ++line) {
            uint32_t* ll = lbl + line * n;
            float*    dd = dist + line * n;
            INDEX*    ff = feat + line * n;

            int n_seeds = 0;
            bool any_nonseed = false;
            for (size_t j = 0; j < n; ++j) {
                if (ll[j] != 0) {
                    dd[j] = 0.0f;
                    v[n_seeds++] = (int)j;
                } else {
                    dd[j] = HUGE_DIST;
                    any_nonseed = true;
                }
            }
            if (!any_nonseed) continue;
            if (n_seeds == 0) {
                if (black_border) {
                    for (size_t i = 0; i < n; ++i)
                        dd[i] = std::fminf(
                            lp_cost_pos(wp, float(i + 1), p),
                            lp_cost_pos(wp, float(n - i), p));
                }
                continue;
            }

            std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));
            std::memcpy(feat_save.data(), ff, n * sizeof(INDEX));

            int k = 0;
            if (black_border) {
                for (size_t i = 0; i < n; ++i) {
                    while (k + 1 < n_seeds &&
                           (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                    const float envelope = lp_cost(wp, float((int)i - v[k]), p);
                    const float border = std::fminf(
                        lp_cost_pos(wp, float(i + 1), p),
                        lp_cost_pos(wp, float(n - i), p));
                    dd[i] = std::fminf(border, envelope);
                    ll[i] = lbl_save[v[k]];
                    ff[i] = feat_save[v[k]];
                }
            } else {
                for (size_t i = 0; i < n; ++i) {
                    while (k + 1 < n_seeds &&
                           (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                    dd[i] = lp_cost(wp, float((int)i - v[k]), p);
                    ll[i] = lbl_save[v[k]];
                    ff[i] = feat_save[v[k]];
                }
            }
        }
    };
    dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

//-----------------------------------------------------------------------------
// Passes 1+: standard Lp envelope on contiguous (num_lines, n) data.
// All entries participate (accumulated distances are finite).
//-----------------------------------------------------------------------------

// Subsequent-axis pass (renamed from _expand_parabolic): unified across all p.
//   P_FIXED == 1: L1 raster sweep propagate (forward + backward, +anis steps).
//                 No envelope build needed; dist already has valid 1D-L1
//                 distances from the previous axis.
//   P_FIXED == 2 / 0: parabolic envelope, accumulated dist values participate.
template <typename D, int P_FIXED>
inline void _expand_propagate(
    uint32_t* RESTRICT lbl,
    D* RESTRICT dist,
    const size_t n,
    const size_t num_lines,
    const D anis,
    const float p,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);

    if constexpr (P_FIXED == 1) {
        auto process_chunk = [&](size_t begin, size_t end) {
            for (size_t line = begin; line < end; ++line) {
                uint32_t* ll = lbl + line * n;
                D*        dd = dist + line * n;
                if (black_border) {
                    if (anis < dd[0])     { dd[0] = anis;       ll[0] = 0; }
                }
                for (size_t i = 1; i < n; ++i) {
                    const D cd = dd[i - 1] + anis;
                    if (cd < dd[i]) { dd[i] = cd; ll[i] = ll[i - 1]; }
                }
                if (black_border) {
                    if (anis < dd[n - 1]) { dd[n - 1] = anis;   ll[n - 1] = 0; }
                }
                for (size_t i = n - 1; i-- > 0; ) {
                    const D cd = dd[i + 1] + anis;
                    if (cd < dd[i]) { dd[i] = cd; ll[i] = ll[i + 1]; }
                }
            }
        };
        dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
    } else {
        static_assert(std::is_same_v<D, float>,
                      "Parabolic envelope requires D=float");
        const float wp = wp_from_anisotropy((float)anis, p);
        const int nn = (int)n;
        float* dist_f = reinterpret_cast<float*>(dist);

        auto process_chunk = [&](size_t begin, size_t end) {
            std::vector<int>      v(n);
            std::vector<float>    ff(n), ranges(n + 1);
            std::vector<uint32_t> lbl_save(n);

            for (size_t line = begin; line < end; ++line) {
                uint32_t* ll = lbl + line * n;
                float*    dd = dist_f + line * n;

            // Quick check: if all dist==0 (all seeds), labels won't change
            bool any_nonzero = false;
            for (size_t j = 0; j < n; ++j) {
                if (dd[j] != 0.0f) { any_nonzero = true; break; }
            }
            if (!any_nonzero) continue;

            std::memcpy(ff.data(), dd, n * sizeof(float));
            std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));

            // Build lower envelope (F&H, all entries participate)
            int k = 0;
            v[0] = 0;
            ranges[0] = -std::numeric_limits<float>::infinity();
            ranges[1] = std::numeric_limits<float>::infinity();

            auto intersect = [&](int a, int b) -> float {
                return lp_intersect(ff.data(), a, b, wp, p, nn);
            };

            float s;
            for (int i = 1; i < nn; i++) {
                s = intersect(v[k], i);
                while (k > 0 && s <= ranges[k]) {
                    k--;
                    s = intersect(v[k], i);
                }
                k++;
                v[k] = i;
                ranges[k] = s;
                ranges[k + 1] = std::numeric_limits<float>::infinity();
            }

            // Output pass
            k = 0;
            if (black_border) {
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    const float envelope = lp_cost<P_FIXED>(wp, float(i - v[k]), p) + ff[v[k]];
                    const float border = std::fminf(
                        lp_cost_pos<P_FIXED>(wp, float(i + 1), p),
                        lp_cost_pos<P_FIXED>(wp, float(nn - i), p));
                    dd[i] = std::fminf(border, envelope);
                    ll[i] = lbl_save[v[k]];
                }
            } else {
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    dd[i] = lp_cost<P_FIXED>(wp, float(i - v[k]), p) + ff[v[k]];
                    ll[i] = lbl_save[v[k]];
                }
            }
        }
        };
        dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
    }
}

template <typename INDEX>
inline void _expand_parabolic_feat(
    uint32_t* RESTRICT lbl,
    float* RESTRICT dist,
    INDEX* RESTRICT feat,
    const size_t n,
    const size_t num_lines,
    const float anis,
    const float p,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);
    const float wp = wp_from_anisotropy(anis, p);
    const int nn = (int)n;

    auto process_chunk = [&](size_t begin, size_t end) {
        std::vector<int>      v(n);
        std::vector<float>    ff(n), ranges(n + 1);
        std::vector<uint32_t> lbl_save(n);
        std::vector<INDEX>    feat_save(n);

        for (size_t line = begin; line < end; ++line) {
            uint32_t* ll = lbl + line * n;
            float*    dd = dist + line * n;
            INDEX*    ft = feat + line * n;

            bool any_nonzero = false;
            for (size_t j = 0; j < n; ++j) {
                if (dd[j] != 0.0f) { any_nonzero = true; break; }
            }
            if (!any_nonzero) continue;

            std::memcpy(ff.data(), dd, n * sizeof(float));
            std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));
            std::memcpy(feat_save.data(), ft, n * sizeof(INDEX));

            // Build lower envelope (standard F&H, all entries participate)
            int k = 0;
            v[0] = 0;
            ranges[0] = -std::numeric_limits<float>::infinity();
            ranges[1] = std::numeric_limits<float>::infinity();

            auto intersect = [&](int a, int b) -> float {
                return lp_intersect(ff.data(), a, b, wp, p, nn);
            };

            float s;
            for (int i = 1; i < nn; i++) {
                s = intersect(v[k], i);
                while (k > 0 && s <= ranges[k]) {
                    k--;
                    s = intersect(v[k], i);
                }
                k++;
                v[k] = i;
                ranges[k] = s;
                ranges[k + 1] = std::numeric_limits<float>::infinity();
            }

            // Output pass
            k = 0;
            if (black_border) {
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    const float envelope = lp_cost(wp, float(i - v[k]), p) + ff[v[k]];
                    const float border = std::fminf(
                        lp_cost_pos(wp, float(i + 1), p),
                        lp_cost_pos(wp, float(nn - i), p));
                    dd[i] = std::fminf(border, envelope);
                    ll[i] = lbl_save[v[k]];
                    ft[i] = feat_save[v[k]];
                }
            } else {
                // No borders -- parabolic result only
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    dd[i] = lp_cost(wp, float(i - v[k]), p) + ff[v[k]];
                    ll[i] = lbl_save[v[k]];
                    ft[i] = feat_save[v[k]];
                }
            }
        }
    };
    dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

//-----------------------------------------------------------------------------
// Blocked transpose with streaming stores for non-contiguous axis processing.
// Uses non-temporal stores for the strided writes to avoid read-for-ownership
// cache line fetches, which cause 16x bandwidth amplification on x86.
// 3 barriers per axis (transpose → process → transpose back).
//-----------------------------------------------------------------------------

constexpr size_t TRANSPOSE_BLOCK = 64;

// Transpose A planes of (rows x cols) → (cols x rows), one array.
// Read-sequential (inner loop over c) with strided writes using a small
// register-resident tile to amortize write-combining. Block size 64.
template <typename T>
inline void _transpose_planes_nt(
    const T* RESTRICT src, T* RESTRICT dst,
    const size_t A, const size_t rows, const size_t cols,
    const size_t threads
) {
    const size_t ncb = (cols + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t nrb = (rows + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t bpp = nrb * ncb;
    const size_t total = A * bpp;

    dispatch_parallel(threads, total, threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const size_t a   = idx / bpp;
                const size_t blk = idx % bpp;
                const size_t rb  = blk / ncb;
                const size_t cb  = blk % ncb;
                const size_t r0 = rb * TRANSPOSE_BLOCK, r1 = std::min(r0 + TRANSPOSE_BLOCK, rows);
                const size_t c0 = cb * TRANSPOSE_BLOCK, c1 = std::min(c0 + TRANSPOSE_BLOCK, cols);
                const T* sp = src + a * rows * cols;
                T* dp = dst + a * cols * rows;
                for (size_t r = r0; r < r1; ++r)
                    for (size_t c = c0; c < c1; ++c)
                        dp[c * rows + r] = sp[r * cols + c];
            }
        }
    );
}

// Fused transpose of two arrays
template <typename T1, typename T2>
inline void _transpose_planes_2_nt(
    const T1* RESTRICT s1, T1* RESTRICT d1,
    const T2* RESTRICT s2, T2* RESTRICT d2,
    const size_t A, const size_t rows, const size_t cols,
    const size_t threads
) {
    const size_t ncb = (cols + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t nrb = (rows + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t bpp = nrb * ncb;
    const size_t total = A * bpp;

    dispatch_parallel(threads, total, threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const size_t a   = idx / bpp;
                const size_t blk = idx % bpp;
                const size_t rb  = blk / ncb;
                const size_t cb  = blk % ncb;
                const size_t r0 = rb * TRANSPOSE_BLOCK, r1 = std::min(r0 + TRANSPOSE_BLOCK, rows);
                const size_t c0 = cb * TRANSPOSE_BLOCK, c1 = std::min(c0 + TRANSPOSE_BLOCK, cols);
                const size_t plane = a * rows * cols;
                const size_t tplane = a * cols * rows;
                for (size_t r = r0; r < r1; ++r)
                    for (size_t c = c0; c < c1; ++c) {
                        d1[tplane + c * rows + r] = s1[plane + r * cols + c];
                        d2[tplane + c * rows + r] = s2[plane + r * cols + c];
                    }
            }
        }
    );
}

// Fused transpose of three arrays
template <typename T1, typename T2, typename T3>
inline void _transpose_planes_3_nt(
    const T1* RESTRICT s1, T1* RESTRICT d1,
    const T2* RESTRICT s2, T2* RESTRICT d2,
    const T3* RESTRICT s3, T3* RESTRICT d3,
    const size_t A, const size_t rows, const size_t cols,
    const size_t threads
) {
    const size_t ncb = (cols + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t nrb = (rows + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t bpp = nrb * ncb;
    const size_t total = A * bpp;

    dispatch_parallel(threads, total, threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const size_t a   = idx / bpp;
                const size_t blk = idx % bpp;
                const size_t rb  = blk / ncb;
                const size_t cb  = blk % ncb;
                const size_t r0 = rb * TRANSPOSE_BLOCK, r1 = std::min(r0 + TRANSPOSE_BLOCK, rows);
                const size_t c0 = cb * TRANSPOSE_BLOCK, c1 = std::min(c0 + TRANSPOSE_BLOCK, cols);
                const size_t plane = a * rows * cols;
                const size_t tplane = a * cols * rows;
                for (size_t r = r0; r < r1; ++r)
                    for (size_t c = c0; c < c1; ++c) {
                        const size_t si = plane + r * cols + c;
                        const size_t di = tplane + c * rows + r;
                        d1[di] = s1[si];
                        d2[di] = s2[si];
                        d3[di] = s3[si];
                    }
            }
        }
    );
}

//-----------------------------------------------------------------------------
// Strided variants: streaming transpose → contiguous process → streaming transpose back.
//-----------------------------------------------------------------------------

// Strided pass-0 / propagate, unified across p.
//   P_FIXED == 1: direct strided access (no transpose). L1's per-element
//                 work (+anis) is so cheap that the transpose round-trip
//                 dominates -- direct strided wins by 2-5x.
//   P_FIXED == 2 / 0: transpose-then-contiguous. Parabolic envelope's
//                     per-row work (intersection arithmetic, stack
//                     management) is heavy enough to amortize the round-trip.
template <typename D, int P_FIXED>
inline void _expand_pass0_strided(
    uint32_t* RESTRICT lbl,
    D* RESTRICT dist,
    uint32_t* RESTRICT ws_lbl,
    D* RESTRICT ws_dist,
    const size_t B, const size_t C, const size_t A,
    const D anis, const float p,
    const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    if constexpr (P_FIXED == 1) {
        // Direct strided access -- no transpose round-trip.
        constexpr D HUGE_DIST = std::numeric_limits<D>::max() / 4;
        const int64_t stride = (int64_t)C;

        auto process_chunk = [&](size_t l0, size_t l1) {
            for (size_t l = l0; l < l1; ++l) {
                const size_t a = l / C;
                const size_t c = l - a * C;
                const size_t base = a * B * C + c;
                uint32_t* ll = lbl + base;
                D*        dd = dist + base;

                // Pointer-walk forward sweep -- clang/gcc strength-reduce
                // automatically; MSVC's loop optimizer is weaker for templated
                // `D*` accessors so explicit pointer-walk produces tighter code.
                D prev_d = black_border ? D(0) : HUGE_DIST;
                uint32_t prev_l = 0;
                uint32_t* ll_p = ll;
                D*        dd_p = dd;
                for (size_t i = 0; i < B; ++i) {
                    const uint32_t init_l = *ll_p;
                    const D init_d = (init_l != 0) ? D(0) : HUGE_DIST;
                    const D cd = prev_d + anis;
                    D out_d; uint32_t out_l;
                    if (cd < init_d) { out_d = cd;     out_l = prev_l; }
                    else             { out_d = init_d; out_l = init_l; }
                    *dd_p = out_d;
                    *ll_p = out_l;
                    prev_d = out_d;
                    prev_l = out_l;
                    ll_p += stride;
                    dd_p += stride;
                }
                uint32_t* ll_back = ll + (B - 1) * stride;
                D*        dd_back = dd + (B - 1) * stride;
                if (black_border) {
                    if (anis < *dd_back) { *dd_back = anis; *ll_back = 0; }
                }
                uint32_t* ll_next = ll_back;
                D*        dd_next = dd_back;
                for (size_t i = B - 1; i-- > 0; ) {
                    ll_back -= stride; dd_back -= stride;
                    const D cd = *dd_next + anis;
                    if (cd < *dd_back) { *dd_back = cd; *ll_back = *ll_next; }
                    ll_next = ll_back; dd_next = dd_back;
                }
            }
        };
        dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
    } else {
        // Parabolic envelope: transpose-then-contiguous round-trip amortizes.
        // Cast through `void*` so the discarded branch type-checks against
        // float* even when D=int32_t (this branch is never instantiated for
        // the int32 path because `if constexpr` discards it).
        static_assert(std::is_same_v<D, float>,
                      "Parabolic envelope strided requires D=float");
        float* dist_f    = reinterpret_cast<float*>(dist);
        float* ws_dist_f = reinterpret_cast<float*>(ws_dist);
        _transpose_planes_nt(lbl, ws_lbl, A, B, C, threads);
        _expand_pass0<float, P_FIXED>(ws_lbl, ws_dist_f, B, num_lines,
                                      (float)anis, p, black_border, parallel);
        _transpose_planes_2_nt(ws_lbl, lbl, ws_dist_f, dist_f, A, C, B, threads);
    }
}

template <typename D, int P_FIXED>
inline void _expand_propagate_strided(
    uint32_t* RESTRICT lbl,
    D* RESTRICT dist,
    uint32_t* RESTRICT ws_lbl,
    D* RESTRICT ws_dist,
    const size_t B, const size_t C, const size_t A,
    const D anis, const float p,
    const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    if constexpr (P_FIXED == 1) {
        const int64_t stride = (int64_t)C;

        auto process_chunk = [&](size_t l0, size_t l1) {
            for (size_t l = l0; l < l1; ++l) {
                const size_t a = l / C;
                const size_t c = l - a * C;
                const size_t base = a * B * C + c;
                uint32_t* ll = lbl + base;
                D*        dd = dist + base;

                if (black_border) {
                    if (anis < dd[0]) { dd[0] = anis; ll[0] = 0; }
                }
                uint32_t* ll_prev = ll;
                D*        dd_prev = dd;
                uint32_t* ll_p = ll + stride;
                D*        dd_p = dd + stride;
                for (size_t i = 1; i < B; ++i) {
                    const D cd = *dd_prev + anis;
                    if (cd < *dd_p) { *dd_p = cd; *ll_p = *ll_prev; }
                    ll_prev = ll_p; dd_prev = dd_p;
                    ll_p += stride; dd_p += stride;
                }
                uint32_t* ll_back = ll + (B - 1) * stride;
                D*        dd_back = dd + (B - 1) * stride;
                if (black_border) {
                    if (anis < *dd_back) { *dd_back = anis; *ll_back = 0; }
                }
                uint32_t* ll_next = ll_back;
                D*        dd_next = dd_back;
                for (size_t i = B - 1; i-- > 0; ) {
                    ll_back -= stride; dd_back -= stride;
                    const D cd = *dd_next + anis;
                    if (cd < *dd_back) { *dd_back = cd; *ll_back = *ll_next; }
                    ll_next = ll_back; dd_next = dd_back;
                }
            }
        };
        dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
    } else {
        static_assert(std::is_same_v<D, float>,
                      "Parabolic envelope strided requires D=float");
        float* dist_f    = reinterpret_cast<float*>(dist);
        float* ws_dist_f = reinterpret_cast<float*>(ws_dist);
        _transpose_planes_2_nt(lbl, ws_lbl, dist_f, ws_dist_f, A, B, C, threads);
        _expand_propagate<float, P_FIXED>(ws_lbl, ws_dist_f, B, num_lines,
                                          (float)anis, p, black_border, parallel);
        _transpose_planes_2_nt(ws_lbl, lbl, ws_dist_f, dist_f, A, C, B, threads);
    }
}

template <typename INDEX>
inline void _expand_pass0_feat_strided(
    uint32_t* RESTRICT lbl,
    float* RESTRICT dist,
    INDEX* RESTRICT feat,
    uint32_t* RESTRICT ws_lbl,
    float* RESTRICT ws_dist,
    INDEX* RESTRICT ws_feat,
    const size_t B, const size_t C, const size_t A,
    const float anis, const float p,
    const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    _transpose_planes_2_nt(lbl, ws_lbl, feat, ws_feat, A, B, C, threads);
    _expand_pass0_feat(ws_lbl, ws_dist, ws_feat, B, num_lines, anis, p, black_border, parallel);
    _transpose_planes_3_nt(ws_lbl, lbl, ws_dist, dist, ws_feat, feat, A, C, B, threads);
}

template <typename INDEX>
inline void _expand_parabolic_feat_strided(
    uint32_t* RESTRICT lbl,
    float* RESTRICT dist,
    INDEX* RESTRICT feat,
    uint32_t* RESTRICT ws_lbl,
    float* RESTRICT ws_dist,
    INDEX* RESTRICT ws_feat,
    const size_t B, const size_t C, const size_t A,
    const float anis, const float p,
    const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    _transpose_planes_3_nt(lbl, ws_lbl, dist, ws_dist, feat, ws_feat, A, B, C, threads);
    _expand_parabolic_feat(ws_lbl, ws_dist, ws_feat, B, num_lines, anis, p, black_border, parallel);
    _transpose_planes_3_nt(ws_lbl, lbl, ws_dist, dist, ws_feat, feat, A, C, B, threads);
}

//=============================================================================
// Expand labels orchestrators (blocked-transpose pipeline with cached buffers)
//=============================================================================

// labels-only mode
//=============================================================================
// 2D L1 stripe-band pass -- fastest for 2D specifically. Each thread takes
// a contiguous range of columns and processes the full forward+backward
// sweep over those columns row-by-row. Unit-stride reads/writes within
// the band (vs the per-line strided path's large strides) → up to 3x faster.
template <typename D>
inline void _expand_l1_2d_col_stripe(
    uint32_t* RESTRICT lbl,
    D* RESTRICT dist,
    const size_t H, const size_t W,
    const D anis,
    const bool black_border,
    const int parallel
) {
    if (H <= 1 || W == 0) return;
    // 2D L1 col-stripe is only invoked for L1 (no L2 analog), so always P_FIXED=1.
    const size_t threads = compute_threads(parallel, W, H);
    auto process_band = [&](size_t x0, size_t x1) {
        // Forward (rows 1..H-1)
        for (size_t y = 1; y < H; ++y) {
            uint32_t* lr = lbl  + y * W;
            D*        dr = dist + y * W;
            const uint32_t* lt = lbl  + (y - 1) * W;
            const D*        dt = dist + (y - 1) * W;
            for (size_t x = x0; x < x1; ++x) {
                const D cd = dt[x] + anis;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lt[x]; }
            }
        }
        // black_border: virtual seed at row -1 contributes anis to row 0;
        // virtual seed at row H contributes anis to row H-1.
        if (black_border) {
            D*        dr0 = dist;
            uint32_t* lr0 = lbl;
            for (size_t x = x0; x < x1; ++x) {
                if (anis < dr0[x]) { dr0[x] = anis; lr0[x] = 0; }
            }
            D*        drH = dist + (H - 1) * W;
            uint32_t* lrH = lbl + (H - 1) * W;
            for (size_t x = x0; x < x1; ++x) {
                if (anis < drH[x]) { drH[x] = anis; lrH[x] = 0; }
            }
        }
        // Backward (rows H-2..0)
        for (size_t y = H - 1; y-- > 0; ) {
            uint32_t* lr = lbl  + y * W;
            D*        dr = dist + y * W;
            const uint32_t* lb = lbl  + (y + 1) * W;
            const D*        db = dist + (y + 1) * W;
            for (size_t x = x0; x < x1; ++x) {
                const D cd = db[x] + anis;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lb[x]; }
            }
        }
    };
    dispatch_parallel(threads, W, threads * ND_CHUNKS_PER_THREAD, process_band);
}

template <typename T>
inline void expand_labels_fused(
    const T* data,
    uint32_t* labels_out,
    const size_t* shape,
    const float* anisotropy,
    const float p,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // 1D path
    if (dims == 1) {
        const size_t n = shape[0];
        if (n == 0) return;
        std::vector<size_t> seeds;
        std::vector<double> mids;
        if (!_expand_1d_setup(data, n, seeds, mids)) {
            std::fill(labels_out, labels_out + n, uint32_t(0));
            return;
        }
        size_t k = 0;
        for (size_t i = 0; i < n; ++i) {
            while (k < mids.size() && (double)i >= mids[k]) ++k;
            const size_t seed_idx = seeds[std::min(k, seeds.size() - 1)];
            if (black_border) {
                const size_t border_dist = std::min(i + 1, n - i);
                const size_t seed_dist   = (i >= seed_idx) ? (i - seed_idx) : (seed_idx - i);
                if (border_dist <= seed_dist) { labels_out[i] = 0; continue; }
            }
            labels_out[i] = (uint32_t)data[seed_idx];
        }
        return;
    }

    if (dims > EDT_MAX_DIMS) {
        throw std::invalid_argument("expand_labels supports at most 32 dimensions");
    }
    // ND path: blocked-transpose pipeline with cached buffers
    size_t total = 1;
    size_t strides[EDT_MAX_DIMS], paxes[EDT_MAX_DIMS];
    for (size_t d = dims; d-- > 0;) { strides[d] = total; total *= shape[d]; }
    if (total == 0) return;

    _expand_sort_axes(paxes, shape, strides, dims);

    // Pool size: start at user-supplied `parallel`, then apply
    //   (1) small-workload cap (dispatch overhead dominates)
    //   (2) kernel-saturation cap (empirical probe, runtime (p, ndim) lookup)
    // Both target the active pool used by dispatch_parallel.
    //
    // The 60000 / 120000 / 400000 thresholds match compute_threads() above;
    // see the comment there for sweep provenance.
    size_t pool_size = (parallel > 0) ? (size_t)parallel : 1;
    if (!tls_probe_active) {
        if      (total <= 60000)   pool_size = std::min<size_t>(pool_size, 4);
        else if (total <= 120000)  pool_size = std::min<size_t>(pool_size, 8);
        else if (total <= 400000)  pool_size = std::min<size_t>(pool_size, 12);
        if (total > 1000000) {
            const int p_class = (p == 1.0f) ? 1 : (p == 2.0f ? 2 : 0);
            const int ndim    = (dims == 2 || dims == 3) ? (int)dims : 0;
            pool_size = std::min(pool_size, kernel_saturation_T(p_class, ndim));
        }
    }
    if (std::getenv("EDT_LP_PROBE_DEBUG")) {
        std::fprintf(stderr, "[edt_lp dispatch] p=%g dims=%zu total=%zu parallel=%d -> pool_size=%zu\n",
                     p, dims, total, parallel, pool_size);
    }
    edt::ForkJoinPool& active_pool = shared_pool_for(pool_size);
    ActivePoolGuard pool_guard(active_pool);

    // Use labels_out directly as the lbl work buffer -- saves one full pass
    // of memory traffic vs allocating a separate cached lbl + final memcpy.
    // (Slot 0 is no longer needed.)  Slots: 1=dist, 2=ws_lbl, 3=ws_dist.
    auto& cache = expand_cache();
    uint32_t* lbl     = labels_out;
    float*    dist    = (float*)cache.get(1, total * sizeof(float));
    uint32_t* ws_lbl  = (uint32_t*)cache.get(2, total * sizeof(uint32_t));
    float*    ws_dist = (float*)cache.get(3, total * sizeof(float));

    // Initial cast/copy data → labels_out. Always serial: a single
    // thread saturates 25-50 GB/s on consumer DDR4/5; parallel adds
    // dispatch + memory-channel contention for no consistent win, and
    // hurts on AMD Zen4 with oversubscribed SMT (>1.5x slowdown at
    // T=32 / 4096^2 vs serial). cpp_proto does the same.
    constexpr size_t COPY_PARALLEL_THRESHOLD = SIZE_MAX;
    if (total < COPY_PARALLEL_THRESHOLD) {
        if constexpr (std::is_same_v<T, uint32_t>) {
            std::memcpy(labels_out, data, total * sizeof(uint32_t));
        } else {
            for (size_t i = 0; i < total; ++i) lbl[i] = (uint32_t)data[i];
        }
    } else {
        const size_t par_threads = compute_threads(parallel, total, 1);
        dispatch_parallel(par_threads, total, par_threads * ND_CHUNKS_PER_THREAD,
            [&](size_t begin, size_t end) {
                if constexpr (std::is_same_v<T, uint32_t>) {
                    std::memcpy(labels_out + begin, data + begin, (end - begin) * sizeof(uint32_t));
                } else {
                    for (size_t i = begin; i < end; ++i)
                        lbl[i] = (uint32_t)data[i];
                }
            });
    }

    // p=1.0 takes the dedicated raster-sweep path (2-5x faster than the
    // bisection-based parabolic envelope at p=1). For all other p the
    // parabolic-envelope code handles things, with closed-form
    // intersection at p=2 and bisection elsewhere.
    const bool use_l1 = (p == 1.0f);

    // L1 with integer-valued anisotropy → int32 dist buffer. Apple Silicon
    // FP add has 3-cycle latency vs int's 1-cycle; the forward raster sweep
    // is a register-dependency chain on prev_d, so int dist runs ~3x faster
    // through that chain. Common case: anisotropy=1.0 for all axes.
    bool l1_int = use_l1;
    if (l1_int) {
        for (size_t d = 0; d < dims; ++d) {
            const float a = anisotropy[d];
            if (a < 1.0f || a > 1e6f || a != std::floor(a)) { l1_int = false; break; }
        }
    }

    // Helper: a small per-axis dispatcher that picks the unified template
    // instantiation by P_FIXED. The L1 path uses int32 dist when l1_int is
    // true (integer anisotropy) for tighter codegen on the dependency-chain
    // forward sweep; otherwise uses float dist.
    //
    // Templated callable rather than a single big switch so each P_FIXED
    // path is compile-time selected and inlined.
    auto run_axes = [&](auto p_tag) {
        constexpr int P_FIXED = decltype(p_tag)::value;
        using D = std::conditional_t<P_FIXED == 1, int32_t, float>;
        D* dist_typed;
        if constexpr (P_FIXED == 1) {
            dist_typed = (int32_t*)cache.get(1, total * sizeof(int32_t));
        } else {
            dist_typed = dist;
        }

        // 2D L1 fast path: pass0 over rows + column-stripe over outer axis.
        // Only applicable for L1 -- L2 envelope construction can't be done
        // column-band-parallel because each row's envelope depends on the
        // whole row at once.
        if constexpr (P_FIXED == 1) {
            if (dims == 2 && strides[paxes[0]] == 1) {
                const size_t inner_axis = paxes[0];
                const size_t outer_axis = paxes[1];
                const size_t H = shape[outer_axis];
                const size_t W = shape[inner_axis];
                const D inner_anis = (D)anisotropy[inner_axis];
                const D outer_anis = (D)anisotropy[outer_axis];
                _expand_pass0<D, P_FIXED>(lbl, dist_typed, W, H, inner_anis, /*p=*/1.0f, black_border, parallel);
                _expand_l1_2d_col_stripe<D>(lbl, dist_typed, H, W, outer_anis, black_border, parallel);
                return;
            }
        }

        for (size_t pass = 0; pass < dims; ++pass) {
            const size_t axis     = paxes[pass];
            const size_t axis_len = shape[axis];
            const D anis = (D)anisotropy[axis];

            if (strides[axis] == 1) {
                const size_t num_lines = total / axis_len;
                if (pass == 0)
                    _expand_pass0<D, P_FIXED>(lbl, dist_typed, axis_len, num_lines, anis, p, black_border, parallel);
                else
                    _expand_propagate<D, P_FIXED>(lbl, dist_typed, axis_len, num_lines, anis, p, black_border, parallel);
            } else {
                const size_t C = strides[axis];
                const size_t B = axis_len;
                const size_t A = total / (B * C);
                if (pass == 0)
                    _expand_pass0_strided<D, P_FIXED>(lbl, dist_typed,
                        ws_lbl, (D*)ws_dist, B, C, A, anis, p, black_border, parallel);
                else
                    _expand_propagate_strided<D, P_FIXED>(lbl, dist_typed,
                        ws_lbl, (D*)ws_dist, B, C, A, anis, p, black_border, parallel);
            }
        }
    };

    if (l1_int) {
        run_axes(std::integral_constant<int, 1>{});
    } else if (use_l1) {
        // L1 with non-integer anisotropy → still raster sweep, but using
        // float dist to preserve the fractional anis correctly. P_FIXED==1
        // selects raster; D==float keeps fractional-anis arithmetic.
        // (Reachable only when anisotropy has non-integer entries.)
        // Same dispatch as l1_int but with float dist.
        // Inline the equivalent here to avoid making run_axes type-flexible.
        if (dims == 2 && strides[paxes[0]] == 1) {
            const size_t inner_axis = paxes[0];
            const size_t outer_axis = paxes[1];
            const size_t H = shape[outer_axis];
            const size_t W = shape[inner_axis];
            const float inner_anis = anisotropy[inner_axis];
            const float outer_anis = anisotropy[outer_axis];
            _expand_pass0<float, 1>(lbl, dist, W, H, inner_anis, /*p=*/1.0f, black_border, parallel);
            _expand_l1_2d_col_stripe<float>(lbl, dist, H, W, outer_anis, black_border, parallel);
        } else {
            for (size_t pass = 0; pass < dims; ++pass) {
                const size_t axis     = paxes[pass];
                const size_t axis_len = shape[axis];
                const float anis = anisotropy[axis];
                if (strides[axis] == 1) {
                    const size_t num_lines = total / axis_len;
                    if (pass == 0)
                        _expand_pass0<float, 1>(lbl, dist, axis_len, num_lines, anis, 1.0f, black_border, parallel);
                    else
                        _expand_propagate<float, 1>(lbl, dist, axis_len, num_lines, anis, 1.0f, black_border, parallel);
                } else {
                    const size_t C = strides[axis];
                    const size_t B = axis_len;
                    const size_t A = total / (B * C);
                    if (pass == 0)
                        _expand_pass0_strided<float, 1>(lbl, dist, ws_lbl, ws_dist, B, C, A, anis, 1.0f, black_border, parallel);
                    else
                        _expand_propagate_strided<float, 1>(lbl, dist, ws_lbl, ws_dist, B, C, A, anis, 1.0f, black_border, parallel);
                }
            }
        }
    } else if (p == 2.0f) {
        run_axes(std::integral_constant<int, 2>{});
    } else {
        run_axes(std::integral_constant<int, 0>{});
    }
    // lbl IS labels_out -- no final copy needed
}

// labels + feature indices mode
template <typename T, typename INDEX>
inline void expand_labels_features_fused(
    const T* data,
    uint32_t* labels_out,
    INDEX* features_out,
    const size_t* shape,
    const float* anisotropy,
    const float p,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // 1D path
    if (dims == 1) {
        const size_t n = shape[0];
        if (n == 0) return;
        std::vector<size_t> seeds;
        std::vector<double> mids;
        if (!_expand_1d_setup(data, n, seeds, mids)) {
            std::fill(labels_out, labels_out + n, uint32_t(0));
            std::fill(features_out, features_out + n, INDEX(0));
            return;
        }
        size_t k = 0;
        for (size_t i = 0; i < n; ++i) {
            while (k < mids.size() && (double)i >= mids[k]) ++k;
            const size_t seed_idx = seeds[std::min(k, seeds.size() - 1)];
            if (black_border) {
                const size_t border_dist = std::min(i + 1, n - i);
                const size_t seed_dist   = (i >= seed_idx) ? (i - seed_idx) : (seed_idx - i);
                if (border_dist <= seed_dist) {
                    labels_out[i]   = 0;
                    features_out[i] = INDEX(seed_idx);
                    continue;
                }
            }
            labels_out[i]   = (uint32_t)data[seed_idx];
            features_out[i] = INDEX(seed_idx);
        }
        return;
    }

    if (dims > EDT_MAX_DIMS) {
        throw std::invalid_argument("feature_transform supports at most 32 dimensions");
    }
    // ND path: blocked-transpose pipeline with feature tracking
    size_t total = 1;
    size_t strides[EDT_MAX_DIMS], paxes[EDT_MAX_DIMS];
    for (size_t d = dims; d-- > 0;) { strides[d] = total; total *= shape[d]; }
    if (total == 0) return;

    _expand_sort_axes(paxes, shape, strides, dims);

    // Slots: 0=lbl, 1=dist, 2=ws_lbl, 3=ws_dist
    auto& cache = expand_cache();
    uint32_t* lbl     = (uint32_t*)cache.get(0, total * sizeof(uint32_t));
    float*    dist    = (float*)cache.get(1, total * sizeof(float));
    uint32_t* ws_lbl  = (uint32_t*)cache.get(2, total * sizeof(uint32_t));
    float*    ws_dist = (float*)cache.get(3, total * sizeof(float));

    // Feat/ws_feat use separate malloc (template type can't easily cache)
    INDEX* feat    = (INDEX*)std::malloc(total * sizeof(INDEX));
    INDEX* ws_feat = (INDEX*)std::malloc(total * sizeof(INDEX));

    const size_t par_threads = compute_threads(parallel, total, 1);
    dispatch_parallel(par_threads, total, par_threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                lbl[i]  = (uint32_t)data[i];
                feat[i] = (INDEX)i;
            }
        });

    for (size_t pass = 0; pass < dims; ++pass) {
        const size_t axis     = paxes[pass];
        const size_t axis_len = shape[axis];
        const float  anis     = anisotropy[axis];

        if (strides[axis] == 1) {
            const size_t num_lines = total / axis_len;
            if (pass == 0)
                _expand_pass0_feat(lbl, dist, feat, axis_len, num_lines, anis, p, black_border, parallel);
            else
                _expand_parabolic_feat(lbl, dist, feat, axis_len, num_lines, anis, p, black_border, parallel);
        } else {
            const size_t C = strides[axis];
            const size_t B = axis_len;
            const size_t A = total / (B * C);
            if (pass == 0)
                _expand_pass0_feat_strided(lbl, dist, feat, ws_lbl, ws_dist, ws_feat, B, C, A, anis, p, black_border, parallel);
            else
                _expand_parabolic_feat_strided(lbl, dist, feat, ws_lbl, ws_dist, ws_feat, B, C, A, anis, p, black_border, parallel);
        }
    }

    dispatch_parallel(par_threads, total, par_threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            std::memcpy(labels_out + begin, lbl + begin, (end - begin) * sizeof(uint32_t));
            std::memcpy(features_out + begin, feat + begin, (end - begin) * sizeof(INDEX));
        });
    std::free(feat);
    std::free(ws_feat);
}

} // namespace lp

#endif // EDT_LP_HPP

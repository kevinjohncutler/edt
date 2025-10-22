#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

#include "threadpool.h"

#ifndef PYEDT_COMPILED_ENABLE_METRICS
#  if defined(_MSC_VER) && !defined(__clang__)
#    define PYEDT_COMPILED_ENABLE_METRICS 0
#  else
#    define PYEDT_COMPILED_ENABLE_METRICS 1
#  endif
#endif

namespace compiled2 {

constexpr size_t DEFAULT_TILE = 8;
constexpr size_t PYEDT_COMPILED_MAX_BASES = size_t{1} << 20;
constexpr size_t STACK_TILE_CAP = 64;

struct DispatchMetrics {
  size_t total_lines = 0;
  size_t thread_count = 0;
  size_t chunk_units = 0;
  size_t chunk_size = 0;
  size_t tiles = 0;
  size_t lines_processed = 0;
  size_t max_tile = 0;
  size_t min_base = std::numeric_limits<size_t>::max();
  size_t max_base = 0;
};

#if PYEDT_COMPILED_ENABLE_METRICS
namespace detail {
struct AtomicMetrics {
  std::atomic<size_t> total_lines{0};
  std::atomic<size_t> thread_count{0};
  std::atomic<size_t> chunk_units{0};
  std::atomic<size_t> chunk_size{0};
  std::atomic<size_t> tiles{0};
  std::atomic<size_t> lines_processed{0};
  std::atomic<size_t> max_tile{0};
  std::atomic<size_t> min_base{std::numeric_limits<size_t>::max()};
  std::atomic<size_t> max_base{0};
};

inline AtomicMetrics& metrics_atomic() {
  static AtomicMetrics metrics;
  return metrics;
}

inline std::atomic<bool>& metrics_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}
}  // namespace detail

inline void reset_metrics() {
  auto& m = detail::metrics_atomic();
  m.total_lines.store(0, std::memory_order_relaxed);
  m.thread_count.store(0, std::memory_order_relaxed);
  m.chunk_units.store(0, std::memory_order_relaxed);
  m.chunk_size.store(0, std::memory_order_relaxed);
  m.tiles.store(0, std::memory_order_relaxed);
  m.lines_processed.store(0, std::memory_order_relaxed);
  m.max_tile.store(0, std::memory_order_relaxed);
  m.min_base.store(std::numeric_limits<size_t>::max(), std::memory_order_relaxed);
  m.max_base.store(0, std::memory_order_relaxed);
}

inline void enable_metrics(bool enabled) {
  detail::metrics_flag().store(enabled, std::memory_order_relaxed);
  if (!enabled) {
    reset_metrics();
  }
}

inline bool metrics_enabled() {
  return detail::metrics_flag().load(std::memory_order_relaxed);
}

inline void record_tile(size_t produced, size_t first_base) {
  auto& m = detail::metrics_atomic();
  m.tiles.fetch_add(1, std::memory_order_relaxed);
  m.lines_processed.fetch_add(produced, std::memory_order_relaxed);
  size_t current = m.max_tile.load(std::memory_order_relaxed);
  while (current < produced &&
         !m.max_tile.compare_exchange_weak(current, produced, std::memory_order_relaxed)) {
  }
  size_t min_val = m.min_base.load(std::memory_order_relaxed);
  while (first_base < min_val &&
         !m.min_base.compare_exchange_weak(min_val, first_base, std::memory_order_relaxed)) {
  }
  size_t max_val = m.max_base.load(std::memory_order_relaxed);
  size_t tile_end = first_base + produced;
  while (tile_end > max_val &&
         !m.max_base.compare_exchange_weak(max_val, tile_end, std::memory_order_relaxed)) {
  }
}

inline DispatchMetrics metrics_snapshot() {
  DispatchMetrics out;
  auto& m = detail::metrics_atomic();
  out.total_lines = m.total_lines.load(std::memory_order_relaxed);
  out.thread_count = m.thread_count.load(std::memory_order_relaxed);
  out.chunk_units = m.chunk_units.load(std::memory_order_relaxed);
  out.chunk_size = m.chunk_size.load(std::memory_order_relaxed);
  out.tiles = m.tiles.load(std::memory_order_relaxed);
  out.lines_processed = m.lines_processed.load(std::memory_order_relaxed);
  out.max_tile = m.max_tile.load(std::memory_order_relaxed);
  out.min_base = m.min_base.load(std::memory_order_relaxed);
  out.max_base = m.max_base.load(std::memory_order_relaxed);
  return out;
}
#else
namespace detail {
inline bool& metrics_flag_stub() {
  static bool enabled = false;
  return enabled;
}
}  // namespace detail

inline void reset_metrics() {}

inline void enable_metrics(bool enabled) {
  detail::metrics_flag_stub() = enabled;
}

inline bool metrics_enabled() {
  return detail::metrics_flag_stub();
}

inline void record_tile(size_t, size_t) {}

inline DispatchMetrics metrics_snapshot() {
  return DispatchMetrics{};
}
#endif  // PYEDT_COMPILED_ENABLE_METRICS

template <size_t Dim>
class TileWalker {
 public:
  TileWalker() {
    extents_.fill(0);
    strides_.fill(0);
    coords_.fill(0);
    base_ = 0;
  }

  void reset(const size_t* extents, const size_t* strides) {
    for (size_t i = 0; i < Dim; ++i) {
      extents_[i] = extents[i];
      strides_[i] = strides[i];
      coords_[i] = 0;
    }
    base_ = 0;
  }

  void set_linear(size_t index) {
    base_ = 0;
    for (size_t i = 0; i < Dim; ++i) {
      const size_t extent = extents_[i];
      const size_t coord = extent ? (index % extent) : 0;
      coords_[i] = coord;
      index = extent ? (index / extent) : 0;
      base_ += coord * strides_[i];
    }
  }

  size_t emit(size_t request, size_t& remaining, size_t* out) {
    size_t produced = 0;
    while (produced < request && remaining > 0) {
      out[produced++] = base_;
      --remaining;
      if (remaining == 0) break;
      advance();
    }
    return produced;
  }

 private:
  void advance() {
    for (size_t i = 0; i < Dim; ++i) {
      base_ += strides_[i];
      if (++coords_[i] < extents_[i]) {
        return;
      }
      base_ -= strides_[i] * extents_[i];
      coords_[i] = 0;
    }
  }

  std::array<size_t, Dim> extents_;
  std::array<size_t, Dim> strides_;
  std::array<size_t, Dim> coords_;
  size_t base_ = 0;
};

template <>
class TileWalker<0> {
 public:
  void reset(const size_t*, const size_t*) {}
  void set_linear(size_t) {}

  size_t emit(size_t request, size_t& remaining, size_t* out) {
    const size_t produced = std::min(request, remaining);
    if (produced == 0) return 0;
    out[0] = 0;
    remaining -= produced;
    return produced;
  }
};

inline size_t compute_concurrency(size_t desired, size_t lines) {
  if (desired == 0) return 1;
  if (lines == 0) return 1;
  return std::max<size_t>(1, std::min<size_t>(desired, lines));
}

inline ThreadPool& shared_pool_for(size_t threads) {
  static std::mutex mutex;
  static std::unordered_map<size_t, std::unique_ptr<ThreadPool>> pools;
  std::lock_guard<std::mutex> lock(mutex);
  auto& entry = pools[threads];
  if (!entry) {
    entry = std::make_unique<ThreadPool>(threads);
  }
  return *entry;
}

template <size_t Dim, typename Fn>
inline void for_each_line(const size_t* extents, const size_t* strides,
                          size_t offset, Fn&& fn) {
  if constexpr (Dim == 0) {
    fn(offset);
  } else if constexpr (Dim == 1) {
    const size_t stride0 = strides[0];
    for (size_t i = 0; i < extents[0]; ++i) {
      fn(offset);
      offset += stride0;
    }
  } else {
    const size_t stride0 = strides[0];
    for (size_t i = 0; i < extents[0]; ++i) {
      for_each_line<Dim - 1>(extents + 1, strides + 1, offset,
                             std::forward<Fn>(fn));
      offset += stride0;
    }
  }
}

template <size_t Dim, typename Fn>
inline void for_each_range(const size_t* extents, const size_t* strides,
                           size_t offset, size_t begin, size_t end, Fn&& fn) {
  if constexpr (Dim == 0) {
    if (begin < end) {
      fn(offset);
    }
  } else if constexpr (Dim == 1) {
    const size_t stride0 = strides[0];
    begin = std::min(begin, extents[0]);
    end = std::min(end, extents[0]);
    offset += begin * stride0;
    for (size_t i = begin; i < end; ++i) {
      fn(offset);
      offset += stride0;
    }
  } else {
    const size_t stride0 = strides[0];
    begin = std::min(begin, extents[0]);
    end = std::min(end, extents[0]);
    offset += begin * stride0;
    for (size_t i = begin; i < end; ++i) {
      for_each_line<Dim - 1>(extents + 1, strides + 1, offset,
                             std::forward<Fn>(fn));
      offset += stride0;
    }
  }
}

template <typename Kernel, typename T, size_t Dim>
bool dispatch_axis(T* labels, float* dest,
                   const size_t* extents, const size_t* strides,
                   size_t axis_length,
                   Kernel&& kernel,
                   int parallel)
{
  size_t total_lines = 1;
  for (size_t i = 0; i < Dim; ++i) {
    const size_t extent = extents[i];
    if (extent == 0) {
      return true;
    }
    if (total_lines > PYEDT_COMPILED_MAX_BASES / std::max<size_t>(extent, size_t{1})) {
      return false;
    }
    total_lines *= extent;
  }

  const size_t prefetch_step = ND_PREFETCH_STEP;

  std::array<size_t, Dim> local_extents{};
  std::array<size_t, Dim> local_strides{};
  for (size_t i = 0; i < Dim; ++i) {
    local_extents[i] = extents[i];
    local_strides[i] = strides[i];
  }

  size_t threads = compute_concurrency(std::max(1, parallel), total_lines);
  threads = std::max<size_t>(size_t{1}, std::min<size_t>(threads, total_lines));

  const size_t total_work = axis_length * total_lines;
  if (total_work <= 120000) {
    threads = std::min<size_t>(threads, size_t{8});
    if (total_work <= 60000) {
      threads = std::min<size_t>(threads, size_t{4});
    }
  } else if (total_work <= 400000) {
    threads = std::min<size_t>(threads, size_t{12});
  }

  const size_t first_extent = (Dim > 0) ? local_extents[0] : 1;
  threads = std::max<size_t>(size_t{1}, std::min<size_t>(threads, first_extent));

  const size_t rest_product = (Dim > 1)
      ? (total_lines / first_extent)
      : 1;
  const size_t first_per_thread = (first_extent + threads - 1) / threads;
  const size_t lines_per_thread = first_per_thread * rest_product;

  const bool metrics_on = metrics_enabled();
  if (metrics_on) {
    reset_metrics();
#if PYEDT_COMPILED_ENABLE_METRICS
    auto& atomic = detail::metrics_atomic();
    atomic.total_lines.store(total_lines, std::memory_order_relaxed);
    atomic.thread_count.store(threads, std::memory_order_relaxed);
    atomic.chunk_units.store(threads, std::memory_order_relaxed);
    atomic.chunk_size.store(lines_per_thread, std::memory_order_relaxed);
#endif
  }

  auto line_kernel = [&](size_t base) {
    if (metrics_on) {
      record_tile(1, base);
    }
    if (prefetch_step > 0) {
      EDT_PREFETCH(labels + base);
    }
    kernel(labels + base, dest + base);
  };

  if (threads <= 1 || first_extent <= 1) {
    for_each_range<Dim>(local_extents.data(), local_strides.data(),
                        size_t{0}, size_t{0}, first_extent, line_kernel);
    return true;
  }

  ThreadPool& pool = shared_pool_for(threads);
  std::vector<std::future<void>> pending;
  pending.reserve(threads);
  for (size_t t = 0; t < threads; ++t) {
    const size_t begin = t * first_per_thread;
    if (begin >= first_extent) break;
    const size_t end = std::min(first_extent, begin + first_per_thread);
    pending.push_back(pool.enqueue([=]() {
      for_each_range<Dim>(local_extents.data(), local_strides.data(),
                          size_t{0}, begin, end, line_kernel);
    }));
  }
  for (auto& f : pending) {
    f.get();
  }

  return true;
}

inline bool trace_nonfinite_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("EDT_ND_TRACE_NONFINITE");
    return env && env[0] != '0';
  }();
  return enabled;
}

inline bool find_nonfinite(const float* data, size_t total, size_t& index_out) {
  for (size_t i = 0; i < total; ++i) {
    if (!std::isfinite(data[i])) {
      index_out = i;
      return true;
    }
  }
  return false;
}

template <typename T>
inline void report_nonfinite(const float* dest,
                             const T* labels,
                             const size_t* shape,
                             size_t dims,
                             size_t index,
                             const char* stage) {
  if (!trace_nonfinite_enabled()) {
    return;
  }
  std::vector<size_t> coords(dims, 0);
  size_t idx = index;
  for (size_t d = dims; d-- > 0;) {
    const size_t extent = shape[d] ? shape[d] : 1;
    coords[d] = extent ? (idx % extent) : 0;
    idx = extent ? (idx / extent) : 0;
  }
  std::fprintf(stderr,
               "[edt.nd] non-finite detected after %s stage at index=%zu "
               "(coords=",
               stage,
               index);
  for (size_t d = 0; d < dims; ++d) {
    std::fprintf(stderr, d == 0 ? "%zu" : ",%zu", coords[d]);
  }
  const float value = dest[index];
  const auto label = labels ? labels[index] : T{};
  std::fprintf(stderr,
               ") value=%g label=%lld dims=%zu\n",
               static_cast<double>(value),
               static_cast<long long>(label),
               static_cast<unsigned long long>(dims));
}

}  // namespace compiled2


/*
 * ND EDT - Graph-First Euclidean Distance Transform
 *
 * Key insight: ND's speed comes from cached label comparison in segment finding:
 *   while (j < n && segids[j] == label) { ++j; }
 *
 * This encoding creates "segment labels" where each voxel gets a uint32_t segment ID.
 * - Segment ID = 0 for background
 * - Each contiguous connected region gets a unique segment ID
 * - Barriers (from voxel graph or label transitions) create new segment IDs
 *
 * The EDT code is then IDENTICAL to ND - just use segment labels as input.
 * No graph edge bit checking, no [i-1] access patterns, just equality comparison.
 */

#ifndef EDT_HPP
#define EDT_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <future>
#include <mutex>
#include <unordered_map>
#include "threadpool.h"
#include "hedley.h"

namespace nd {

// Tuning parameters
static size_t ND_CHUNKS_PER_THREAD = 1;
static size_t ND_TILE = 8;
static bool ND_FORCE_GENERIC_GRAPH = false;  // Deprecated: no-op (unified ND path is now default)

inline void set_tuning(size_t chunks_per_thread, size_t tile) {
    if (chunks_per_thread > 0) ND_CHUNKS_PER_THREAD = chunks_per_thread;
    if (tile > 0) ND_TILE = tile;
}

// Deprecated: no-op since unified ND path is now the only path
inline void set_force_generic(bool force) {
    ND_FORCE_GENERIC_GRAPH = force;  // Kept for API compatibility
}

// Shared thread pool (like ND v1)
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

// Compute effective thread count (like ND v1 capping)
inline size_t compute_threads(size_t desired, size_t total_lines, size_t axis_length) {
    if (desired <= 1 || total_lines <= 1) return 1;

    size_t threads = std::min<size_t>(desired, total_lines);

    // Cap based on total work (like ND v1)
    const size_t total_work = axis_length * total_lines;
    if (total_work <= 60000) {
        threads = std::min<size_t>(threads, 4);
    } else if (total_work <= 120000) {
        threads = std::min<size_t>(threads, 8);
    } else if (total_work <= 400000) {
        threads = std::min<size_t>(threads, 12);
    }

    return std::max<size_t>(1, threads);
}

// for_each_range template (matching ND v1's constexpr pattern)
// This helps the compiler optimize the iteration loop
template <typename Fn>
inline void for_each_range_1d(size_t extent, size_t stride, size_t begin, size_t end, Fn&& fn) {
    begin = std::min(begin, extent);
    end = std::min(end, extent);
    size_t offset = begin * stride;
    for (size_t i = begin; i < end; ++i) {
        fn(offset);
        offset += stride;
    }
}

//-----------------------------------------------------------------------------
// Segment Label Building
//-----------------------------------------------------------------------------

/*
 * Build segment labels from a label array for a single scanline.
 *
 * For each scanline along the axis:
 * - Background (label=0) gets segment ID = 0
 * - Each run of same non-zero labels gets a unique segment ID
 * - Label transitions create new segment IDs
 *
 * NOTE: Segment IDs only need to be unique within each scanline because
 * EDT processes each scanline independently. This eliminates atomic contention.
 */
template <typename T>
inline void build_segment_labels_1d_local(
    uint32_t* seg_labels,
    const T* labels,
    const int n,
    const long int stride
) {
    if (n <= 0) return;

    uint32_t next_seg_id = 1;  // Local counter, no atomic needed
    int i = 0;

    while (i < n) {
        const long int base_idx = i * stride;
        const T label = labels[base_idx];

        if (label == 0) {
            // Background: segment ID = 0
            seg_labels[base_idx] = 0;
            i++;
            continue;
        }

        // Foreground: find extent of same-label run
        const int seg_start = i;
        i++;
        while (i < n && labels[i * stride] == label) {
            i++;
        }
        const int seg_len = i - seg_start;

        // Assign unique segment ID to this run (local to this scanline)
        const uint32_t seg_id = next_seg_id++;
        for (int k = 0; k < seg_len; k++) {
            seg_labels[(seg_start + k) * stride] = seg_id;
        }
    }
}

// Legacy version with atomic for backward compatibility
template <typename T>
inline void build_segment_labels_1d(
    uint32_t* seg_labels,
    const T* labels,
    const int n,
    const size_t stride,
    std::atomic<uint32_t>& next_seg_id
) {
    if (n <= 0) return;

    int i = 0;
    while (i < n) {
        const size_t base_idx = i * stride;
        const T label = labels[base_idx];

        if (label == 0) {
            seg_labels[base_idx] = 0;
            i++;
            continue;
        }

        const int seg_start = i;
        i++;
        while (i < n && labels[i * stride] == label) {
            i++;
        }
        const int seg_len = i - seg_start;

        const uint32_t seg_id = next_seg_id.fetch_add(1, std::memory_order_relaxed);
        for (int k = 0; k < seg_len; k++) {
            seg_labels[(seg_start + k) * stride] = seg_id;
        }
    }
}

/*
 * Build segment labels from a voxel graph for a single axis.
 *
 * The voxel graph encodes connectivity:
 * - graph[i] == 0 means background (no edges)
 * - graph[i] & axis_bit means connected to i+1 along this axis
 *
 * A segment boundary exists when:
 * - graph[i] is background (0)
 * - graph[i-1] & axis_bit == 0 (no edge from i-1 to i)
 */
// Local counter version for graph (no atomic contention)
template <typename GRAPH_T>
inline void build_segment_labels_from_graph_1d_local(
    uint32_t* seg_labels,
    const GRAPH_T* graph,
    const int n,
    const long int stride,
    const GRAPH_T axis_bit
) {
    if (n <= 0) return;

    uint32_t next_seg_id = 1;  // Local counter
    int i = 0;

    while (i < n) {
        const long int base_idx = i * stride;

        if (graph[base_idx] == 0) {
            seg_labels[base_idx] = 0;
            i++;
            continue;
        }

        const int seg_start = i;
        GRAPH_T edge = graph[base_idx];
        i++;

        while (i < n && (edge & axis_bit)) {
            edge = graph[i * stride];
            if (edge == 0) break;
            i++;
        }
        const int seg_len = i - seg_start;

        const uint32_t seg_id = next_seg_id++;
        for (int k = 0; k < seg_len; k++) {
            seg_labels[(seg_start + k) * stride] = seg_id;
        }
    }
}

/*
 * FUSED Pass 0 from Graph (no segment label output)
 *
 * Graph-first design: This function reads the voxel connectivity graph
 * and computes the Rosenfeld-Pfaltz 1D EDT (pass 0) directly.
 *
 * This version doesn't write segment labels - use when parabolic passes
 * are also fused and don't need segment labels.
 */
template <typename GRAPH_T>
inline void squared_edt_1d_from_graph_direct(
    const GRAPH_T* graph,
    float* d,
    const int n,
    const long int stride,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border
) {
    if (n <= 0) return;

    const float inf = std::numeric_limits<float>::infinity();
    int i = 0;

    while (i < n) {
        const long int base_idx = i * stride;

        // Check if this voxel is background (graph == 0)
        if (graph[base_idx] == 0) {
            d[base_idx] = 0.0f;
            i++;
            continue;
        }

        // Foreground: find segment extent using connectivity bits
        const int seg_start = i;
        GRAPH_T edge = graph[base_idx];
        i++;

        // Follow connectivity along axis
        while (i < n && (edge & axis_bit)) {
            edge = graph[i * stride];
            if (edge == 0) break;
            i++;
        }
        const int seg_len = i - seg_start;

        // Compute EDT for this segment
        const bool left_boundary = (seg_start > 0) || black_border;
        const bool right_boundary = (i < n) || black_border;

        // Forward pass: write distances
        if (left_boundary) {
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = static_cast<float>(k + 1) * anisotropy;
            }
        } else {
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = inf;
            }
        }

        // Backward pass
        if (right_boundary) {
            for (int k = seg_len - 1; k >= 0; k--) {
                const float v = static_cast<float>(seg_len - k) * anisotropy;
                const long int idx = (seg_start + k) * stride;
                if (v < d[idx]) {
                    d[idx] = v;
                }
            }
        }

        // Square the distances
        for (int k = 0; k < seg_len; k++) {
            const long int idx = (seg_start + k) * stride;
            d[idx] *= d[idx];
        }
    }
}

/*
 * FUSED Pass 0 + Segment Label Building from Graph (legacy version with seg_labels output)
 */
template <typename GRAPH_T>
inline void squared_edt_1d_from_graph_fused(
    const GRAPH_T* graph,
    float* d,
    uint32_t* seg_labels,
    const int n,
    const long int stride,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border
) {
    if (n <= 0) return;

    const float inf = std::numeric_limits<float>::infinity();
    uint32_t next_seg_id = 1;
    int i = 0;

    while (i < n) {
        const long int base_idx = i * stride;

        // Check if this voxel is background (graph == 0)
        if (graph[base_idx] == 0) {
            d[base_idx] = 0.0f;
            seg_labels[base_idx] = 0;
            i++;
            continue;
        }

        // Foreground: find segment extent using connectivity bits
        const int seg_start = i;
        GRAPH_T edge = graph[base_idx];
        i++;

        // Follow connectivity along axis
        while (i < n && (edge & axis_bit)) {
            edge = graph[i * stride];
            if (edge == 0) break;
            i++;
        }
        const int seg_len = i - seg_start;

        // Assign segment ID and compute EDT in one pass
        const uint32_t seg_id = next_seg_id++;
        const bool left_boundary = (seg_start > 0) || black_border;
        const bool right_boundary = (i < n) || black_border;

        // Forward pass: write distances and segment labels
        if (left_boundary) {
            for (int k = 0; k < seg_len; k++) {
                const long int idx = (seg_start + k) * stride;
                d[idx] = static_cast<float>(k + 1) * anisotropy;
                seg_labels[idx] = seg_id;
            }
        } else {
            for (int k = 0; k < seg_len; k++) {
                const long int idx = (seg_start + k) * stride;
                d[idx] = inf;
                seg_labels[idx] = seg_id;
            }
        }

        // Backward pass
        if (right_boundary) {
            for (int k = seg_len - 1; k >= 0; k--) {
                const float v = static_cast<float>(seg_len - k) * anisotropy;
                const long int idx = (seg_start + k) * stride;
                if (v < d[idx]) {
                    d[idx] = v;
                }
            }
        }

        // Square the distances
        for (int k = 0; k < seg_len; k++) {
            const long int idx = (seg_start + k) * stride;
            d[idx] *= d[idx];
        }
    }
}

// Legacy version with atomic
template <typename GRAPH_T>
inline void build_segment_labels_from_graph_1d(
    uint32_t* seg_labels,
    const GRAPH_T* graph,
    const int n,
    const size_t stride,
    const GRAPH_T axis_bit,
    std::atomic<uint32_t>& next_seg_id
) {
    if (n <= 0) return;

    int i = 0;
    while (i < n) {
        const size_t base_idx = i * stride;

        if (graph[base_idx] == 0) {
            seg_labels[base_idx] = 0;
            i++;
            continue;
        }

        const int seg_start = i;
        GRAPH_T edge = graph[base_idx];
        i++;

        while (i < n && (edge & axis_bit)) {
            edge = graph[i * stride];
            if (edge == 0) break;
            i++;
        }
        const int seg_len = i - seg_start;

        const uint32_t seg_id = next_seg_id.fetch_add(1, std::memory_order_relaxed);
        for (int k = 0; k < seg_len; k++) {
            seg_labels[(seg_start + k) * stride] = seg_id;
        }
    }
}

/*
 * Build segment labels from labels with explicit barriers for a single axis.
 *
 * barriers: uint8_t array where barriers[i] & axis_bit means barrier after position i
 *           (segment boundary between i and i+1)
 */
template <typename T>
inline void build_segment_labels_with_barriers_1d(
    uint32_t* seg_labels,
    const T* labels,
    const uint8_t* barriers,
    const int n,
    const size_t stride,
    const uint8_t axis_bit,
    std::atomic<uint32_t>& next_seg_id
) {
    if (n <= 0) return;

    int i = 0;
    while (i < n) {
        const size_t base_idx = i * stride;
        const T label = labels[base_idx];

        if (label == 0) {
            // Background: segment ID = 0
            seg_labels[base_idx] = 0;
            i++;
            continue;
        }

        // Foreground: find extent considering both label transitions AND barriers
        const int seg_start = i;
        i++;
        while (i < n) {
            // Stop if label changes
            if (labels[i * stride] != label) break;
            // Stop if barrier after previous position
            if (barriers[(i - 1) * stride] & axis_bit) break;
            i++;
        }
        const int seg_len = i - seg_start;

        // Assign unique segment ID
        const uint32_t seg_id = next_seg_id.fetch_add(1, std::memory_order_relaxed);
        for (int k = 0; k < seg_len; k++) {
            seg_labels[(seg_start + k) * stride] = seg_id;
        }
    }
}

//-----------------------------------------------------------------------------
// Parallel Segment Label Building (optimized with shared pool, local counters)
//-----------------------------------------------------------------------------

template <typename T>
inline void build_segment_labels_parallel(
    uint32_t* seg_labels,
    const T* labels,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Fast path for 2D - direct iteration like EDT passes
    if (dims == 2) {
        const size_t other_axis = (axis == 0) ? 1 : 0;
        const size_t other_n = shape[other_axis];
        const long int other_stride = static_cast<long int>(strides[other_axis]);

        const size_t threads = compute_threads(parallel, other_n, n);
        if (threads <= 1) {
            size_t offset = 0;
            for (size_t i = 0; i < other_n; i++) {
                build_segment_labels_1d_local<T>(
                    seg_labels + offset, labels + offset, n, stride_ax
                );
                offset += other_stride;
            }
            return;
        }

        ThreadPool& pool = shared_pool_for(threads);
        const size_t per_thread = (other_n + threads - 1) / threads;
        std::vector<std::future<void>> pending;
        pending.reserve(threads);

        for (size_t t = 0; t < threads; t++) {
            const size_t begin = t * per_thread;
            if (begin >= other_n) break;
            const size_t end = std::min(other_n, begin + per_thread);
            pending.push_back(pool.enqueue([=]() {
                size_t offset = begin * other_stride;
                for (size_t i = begin; i < end; i++) {
                    build_segment_labels_1d_local<T>(
                        seg_labels + offset, labels + offset, n, stride_ax
                    );
                    offset += other_stride;
                }
            }));
        }
        for (auto& f : pending) f.get();
        return;
    }

    // Generic fallback for 3D+
    size_t total_lines = 1;
    for (size_t d = 0; d < dims; d++) {
        if (d != axis) total_lines *= shape[d];
    }

    std::vector<size_t> bases(total_lines);
    size_t idx = 0;
    std::vector<size_t> coords(dims, 0);
    for (size_t line = 0; line < total_lines; line++) {
        size_t base = 0;
        for (size_t d = 0; d < dims; d++) {
            if (d != axis) base += coords[d] * strides[d];
        }
        bases[idx++] = base;
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            coords[d]++;
            if (coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }

    const size_t threads = compute_threads(parallel, total_lines, n);
    if (threads <= 1) {
        for (size_t i = 0; i < total_lines; i++) {
            build_segment_labels_1d_local<T>(
                seg_labels + bases[i], labels + bases[i], n, stride_ax
            );
        }
        return;
    }

    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (total_lines + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= total_lines) break;
        const size_t end = std::min(total_lines, begin + per_thread);
        pending.push_back(pool.enqueue([=, &bases]() {
            for (size_t i = begin; i < end; i++) {
                build_segment_labels_1d_local<T>(
                    seg_labels + bases[i], labels + bases[i], n, stride_ax
                );
            }
        }));
    }
    for (auto& f : pending) f.get();
}

template <typename GRAPH_T>
inline void build_segment_labels_from_graph_parallel(
    uint32_t* seg_labels,
    const GRAPH_T* graph,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Fast path for 2D
    if (dims == 2) {
        const size_t other_axis = (axis == 0) ? 1 : 0;
        const size_t other_n = shape[other_axis];
        const long int other_stride = static_cast<long int>(strides[other_axis]);

        const size_t threads = compute_threads(parallel, other_n, n);
        if (threads <= 1) {
            size_t offset = 0;
            for (size_t i = 0; i < other_n; i++) {
                build_segment_labels_from_graph_1d_local<GRAPH_T>(
                    seg_labels + offset, graph + offset, n, stride_ax, axis_bit
                );
                offset += other_stride;
            }
            return;
        }

        ThreadPool& pool = shared_pool_for(threads);
        const size_t per_thread = (other_n + threads - 1) / threads;
        std::vector<std::future<void>> pending;
        pending.reserve(threads);

        for (size_t t = 0; t < threads; t++) {
            const size_t begin = t * per_thread;
            if (begin >= other_n) break;
            const size_t end = std::min(other_n, begin + per_thread);
            pending.push_back(pool.enqueue([=]() {
                size_t offset = begin * other_stride;
                for (size_t i = begin; i < end; i++) {
                    build_segment_labels_from_graph_1d_local<GRAPH_T>(
                        seg_labels + offset, graph + offset, n, stride_ax, axis_bit
                    );
                    offset += other_stride;
                }
            }));
        }
        for (auto& f : pending) f.get();
        return;
    }

    // Generic fallback for 3D+
    size_t total_lines = 1;
    for (size_t d = 0; d < dims; d++) {
        if (d != axis) total_lines *= shape[d];
    }

    std::vector<size_t> bases(total_lines);
    size_t idx = 0;
    std::vector<size_t> coords(dims, 0);
    for (size_t line = 0; line < total_lines; line++) {
        size_t base = 0;
        for (size_t d = 0; d < dims; d++) {
            if (d != axis) base += coords[d] * strides[d];
        }
        bases[idx++] = base;
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            coords[d]++;
            if (coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }

    const size_t threads = compute_threads(parallel, total_lines, n);
    if (threads <= 1) {
        for (size_t i = 0; i < total_lines; i++) {
            build_segment_labels_from_graph_1d_local<GRAPH_T>(
                seg_labels + bases[i], graph + bases[i], n, stride_ax, axis_bit
            );
        }
        return;
    }

    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (total_lines + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= total_lines) break;
        const size_t end = std::min(total_lines, begin + per_thread);
        pending.push_back(pool.enqueue([=, &bases]() {
            for (size_t i = begin; i < end; i++) {
                build_segment_labels_from_graph_1d_local<GRAPH_T>(
                    seg_labels + bases[i], graph + bases[i], n, stride_ax, axis_bit
                );
            }
        }));
    }
    for (auto& f : pending) f.get();
}

//-----------------------------------------------------------------------------
// FUSED Pass 0 from Graph (parallel dispatch)
//-----------------------------------------------------------------------------

template <typename GRAPH_T>
inline void edt_pass0_from_graph_fused_parallel(
    const GRAPH_T* graph,
    float* output,
    uint32_t* seg_labels,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Fast path for 2D
    if (dims == 2) {
        const size_t other_axis = (axis == 0) ? 1 : 0;
        const size_t other_n = shape[other_axis];
        const long int other_stride = static_cast<long int>(strides[other_axis]);

        const size_t threads = compute_threads(parallel, other_n, n);
        if (threads <= 1) {
            size_t offset = 0;
            for (size_t i = 0; i < other_n; i++) {
                squared_edt_1d_from_graph_fused<GRAPH_T>(
                    graph + offset, output + offset, seg_labels + offset,
                    n, stride_ax, axis_bit, anisotropy, black_border
                );
                offset += other_stride;
            }
            return;
        }

        ThreadPool& pool = shared_pool_for(threads);
        const size_t per_thread = (other_n + threads - 1) / threads;
        std::vector<std::future<void>> pending;
        pending.reserve(threads);

        for (size_t t = 0; t < threads; t++) {
            const size_t begin = t * per_thread;
            if (begin >= other_n) break;
            const size_t end = std::min(other_n, begin + per_thread);
            pending.push_back(pool.enqueue([=]() {
                size_t offset = begin * other_stride;
                for (size_t i = begin; i < end; i++) {
                    squared_edt_1d_from_graph_fused<GRAPH_T>(
                        graph + offset, output + offset, seg_labels + offset,
                        n, stride_ax, axis_bit, anisotropy, black_border
                    );
                    offset += other_stride;
                }
            }));
        }
        for (auto& f : pending) f.get();
        return;
    }

    // Generic fallback for 3D+
    size_t total_lines = 1;
    for (size_t d = 0; d < dims; d++) {
        if (d != axis) total_lines *= shape[d];
    }

    std::vector<size_t> bases(total_lines);
    size_t idx = 0;
    std::vector<size_t> coords(dims, 0);
    for (size_t line = 0; line < total_lines; line++) {
        size_t base = 0;
        for (size_t d = 0; d < dims; d++) {
            if (d != axis) base += coords[d] * strides[d];
        }
        bases[idx++] = base;
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            coords[d]++;
            if (coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }

    const size_t threads = compute_threads(parallel, total_lines, n);
    if (threads <= 1) {
        for (size_t i = 0; i < total_lines; i++) {
            squared_edt_1d_from_graph_fused<GRAPH_T>(
                graph + bases[i], output + bases[i], seg_labels + bases[i],
                n, stride_ax, axis_bit, anisotropy, black_border
            );
        }
        return;
    }

    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (total_lines + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= total_lines) break;
        const size_t end = std::min(total_lines, begin + per_thread);
        pending.push_back(pool.enqueue([=, &bases]() {
            for (size_t i = begin; i < end; i++) {
                squared_edt_1d_from_graph_fused<GRAPH_T>(
                    graph + bases[i], output + bases[i], seg_labels + bases[i],
                    n, stride_ax, axis_bit, anisotropy, black_border
                );
            }
        }));
    }
    for (auto& f : pending) f.get();
}

// Direct version (no seg_labels output) - for fully fused graph path
template <typename GRAPH_T>
inline void edt_pass0_from_graph_direct_parallel(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Unified ND path - parallelize over first other dimension like ND v1
    // Build arrays of extents and strides for dimensions other than axis
    size_t num_other_dims = 0;
    size_t other_extents[8];  // Max 8D support
    size_t other_strides[8];
    size_t total_lines = 1;
    for (size_t d = 0; d < dims && num_other_dims < 8; d++) {
        if (d != axis) {
            other_extents[num_other_dims] = shape[d];
            other_strides[num_other_dims] = strides[d];
            total_lines *= shape[d];
            num_other_dims++;
        }
    }

    // Compute rest_product (product of all dims except first)
    const size_t first_extent = (num_other_dims > 0) ? other_extents[0] : 1;
    const size_t first_stride = (num_other_dims > 0) ? other_strides[0] : 0;
    size_t rest_product = 1;
    for (size_t d = 1; d < num_other_dims; d++) {
        rest_product *= other_extents[d];
    }

    const size_t threads = compute_threads(parallel, total_lines, n);

    // Recursive iteration over remaining dimensions (after first)
    // This is a nested loop unrolled at runtime
    auto for_each_inner = [&](size_t offset, auto& kernel) {
        if (num_other_dims <= 1) {
            kernel(offset);
            return;
        }
        // Iterate dims 1..num_other_dims-1
        size_t coords[8] = {0};
        size_t base = offset;
        for (size_t i = 0; i < rest_product; i++) {
            kernel(base);
            // Increment coords from dimension 1 onwards
            for (size_t d = 1; d < num_other_dims; d++) {
                coords[d]++;
                base += other_strides[d];
                if (coords[d] < other_extents[d]) break;
                base -= coords[d] * other_strides[d];
                coords[d] = 0;
            }
        }
    };

    auto line_kernel = [&](size_t base) {
        squared_edt_1d_from_graph_direct<GRAPH_T>(
            graph + base, output + base,
            n, stride_ax, axis_bit, anisotropy, black_border
        );
    };

    if (threads <= 1) {
        // Single-threaded: iterate all
        for (size_t i0 = 0; i0 < first_extent; i0++) {
            for_each_inner(i0 * first_stride, line_kernel);
        }
        return;
    }

    // Parallel: distribute first dimension like ND v1
    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (first_extent + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= first_extent) break;
        const size_t end = std::min(first_extent, begin + per_thread);
        pending.push_back(pool.enqueue([=, &other_extents, &other_strides]() {
            // Process range [begin, end) of first dimension
            auto inner_kernel = [&](size_t base) {
                squared_edt_1d_from_graph_direct<GRAPH_T>(
                    graph + base, output + base,
                    n, stride_ax, axis_bit, anisotropy, black_border
                );
            };
            for (size_t i0 = begin; i0 < end; i0++) {
                const size_t offset = i0 * first_stride;
                // Iterate remaining dimensions
                if (num_other_dims <= 1) {
                    inner_kernel(offset);
                } else {
                    size_t coords[8] = {0};
                    size_t base = offset;
                    for (size_t i = 0; i < rest_product; i++) {
                        inner_kernel(base);
                        for (size_t d = 1; d < num_other_dims; d++) {
                            coords[d]++;
                            base += other_strides[d];
                            if (coords[d] < other_extents[d]) break;
                            base -= coords[d] * other_strides[d];
                            coords[d] = 0;
                        }
                    }
                }
            }
        }));
    }
    for (auto& f : pending) f.get();
}

//-----------------------------------------------------------------------------
// EDT using Segment Labels (identical to ND algorithm)
//-----------------------------------------------------------------------------

#define sq(x) (static_cast<float>(x) * static_cast<float>(x))

/*
 * Pass 0: Rosenfeld-Pfaltz style 1D EDT for multi-segment data.
 * This is IDENTICAL to ND's _squared_edt_1d_multi_seg_run.
 * Templated to work with any segment ID type.
 */
// Match ND v1's exact types for optimal codegen
template <typename T>
inline void squared_edt_1d_multi_seg_generic(
    const T* segids,
    float* d,
    const int n,
    const long int stride,  // long int like ND v1
    const float anisotropy,
    const bool black_border
) {
    if (n <= 0) return;

    const float inf = std::numeric_limits<float>::infinity();
    long int i = 0;  // long int like ND v1

    while (i < n) {
        const long int base = i * stride;
        const T label = segids[base];

        // Find segment extent - the critical loop
        long int j = i + 1;
        while (j < n && segids[j * stride] == label) {
            ++j;
        }
        const long int len = j - i;

        if (label == 0) {
            // Background: write zeros
            for (long int k = 0; k < len; ++k) {
                d[(i + k) * stride] = 0.0f;
            }
            i = j;
            continue;
        }

        // Foreground segment
        const bool left_boundary = (i > 0) || black_border;
        const bool right_boundary = (j < n) || black_border;

        // Forward pass
        if (left_boundary) {
            for (long int k = 0; k < len; ++k) {
                d[(i + k) * stride] = (static_cast<float>(k + 1)) * anisotropy;
            }
        } else {
            for (long int k = 0; k < len; ++k) {
                d[(i + k) * stride] = inf;
            }
        }

        // Backward pass
        if (right_boundary) {
            for (long int k = len - 1; k >= 0; --k) {
                const float v = (static_cast<float>(len - k)) * anisotropy;
                const long int idx = (i + k) * stride;
                if (v < d[idx]) {
                    d[idx] = v;
                }
            }
        }

        // Square
        for (long int k = 0; k < len; ++k) {
            const long int idx = (i + k) * stride;
            d[idx] *= d[idx];
        }

        i = j;
    }
}

/*
 * FUSED Pass 0 + Segment Label Building
 *
 * Graph-first design: Labels MUST be converted to segment labels.
 * This function does BOTH pass 0 EDT AND builds segment labels in one pass,
 * avoiding the overhead of a separate segment building pass.
 *
 * The segment labels are then used by subsequent parabolic passes.
 */
template <typename T>
inline void squared_edt_1d_multi_seg_build_seglabels(
    const T* labels,
    float* d,
    uint32_t* seg_labels,
    const int n,
    const long int stride,
    const float anisotropy,
    const bool black_border
) {
    if (n <= 0) return;

    const float inf = std::numeric_limits<float>::infinity();
    long int i = 0;
    uint32_t next_seg_id = 1;  // Local counter (no atomic needed per-scanline)

    while (i < n) {
        const long int base = i * stride;
        const T label = labels[base];

        // Find segment extent - the critical loop
        long int j = i + 1;
        while (j < n && labels[j * stride] == label) {
            ++j;
        }
        const long int len = j - i;

        if (label == 0) {
            // Background: write zeros to both output and segment labels
            for (long int k = 0; k < len; ++k) {
                const long int idx = (i + k) * stride;
                d[idx] = 0.0f;
                seg_labels[idx] = 0;
            }
            i = j;
            continue;
        }

        // Foreground segment: assign unique segment ID
        const uint32_t seg_id = next_seg_id++;
        const bool left_boundary = (i > 0) || black_border;
        const bool right_boundary = (j < n) || black_border;

        // Forward pass + write segment labels
        if (left_boundary) {
            for (long int k = 0; k < len; ++k) {
                const long int idx = (i + k) * stride;
                d[idx] = (static_cast<float>(k + 1)) * anisotropy;
                seg_labels[idx] = seg_id;
            }
        } else {
            for (long int k = 0; k < len; ++k) {
                const long int idx = (i + k) * stride;
                d[idx] = inf;
                seg_labels[idx] = seg_id;
            }
        }

        // Backward pass (segment labels already written)
        if (right_boundary) {
            for (long int k = len - 1; k >= 0; --k) {
                const float v = (static_cast<float>(len - k)) * anisotropy;
                const long int idx = (i + k) * stride;
                if (v < d[idx]) {
                    d[idx] = v;
                }
            }
        }

        // Square
        for (long int k = 0; k < len; ++k) {
            const long int idx = (i + k) * stride;
            d[idx] *= d[idx];
        }

        i = j;
    }
}

// Non-templated version for backward compatibility
inline void squared_edt_1d_multi_seg(
    const uint32_t* segids,
    float* d,
    const int n,
    const long int stride,  // long int like ND v1
    const float anisotropy,
    const bool black_border
) {
    squared_edt_1d_multi_seg_generic<uint32_t>(segids, d, n, stride, anisotropy, black_border);
}

/*
 * FUSED Parabolic Pass from Graph
 *
 * Graph-first design: Reads voxel connectivity graph directly and computes
 * parabolic EDT in a single pass, avoiding separate segment label building.
 */
template <typename GRAPH_T>
inline void squared_edt_1d_parabolic_from_graph_ws(
    const GRAPH_T* graph,
    float* f,
    const int n,
    const long int stride,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border,
    int* v,
    float* ff,
    float* ranges
) {
    constexpr int SMALL_THRESHOLD = 8;
    const float anis_sq = anisotropy * anisotropy;
    const float w2 = anis_sq;

    if (n <= 0) return;

    // Fast path for small segments: O(n²) brute force
    auto process_small_run = [&](int start, int len, bool left_border, bool right_border) {
        float original[SMALL_THRESHOLD];
        for (int q = 0; q < len; ++q) {
            original[q] = f[(start + q) * stride];
        }
        for (int j = 0; j < len; ++j) {
            float best = original[j];
            if (left_border) {
                const float cap_left = anis_sq * static_cast<float>((j + 1) * (j + 1));
                if (cap_left < best) best = cap_left;
            }
            if (right_border) {
                const float cap_right = anis_sq * static_cast<float>((len - j) * (len - j));
                if (cap_right < best) best = cap_right;
            }
            for (int q = 0; q < len; ++q) {
                const float delta = static_cast<float>(j - q);
                const float candidate = original[q] + anis_sq * delta * delta;
                if (candidate < best) best = candidate;
            }
            f[(start + j) * stride] = best;
        }
    };

    // Parabolic envelope for larger segments
    auto process_large_run = [&](int start, int len, bool left_border, bool right_border) {
        // Copy to workspace
        for (int i = 0; i < len; i++) {
            ff[i] = f[(start + i) * stride];
        }

        int k = 0;
        v[0] = 0;
        ranges[0] = -std::numeric_limits<float>::infinity();
        ranges[1] = std::numeric_limits<float>::infinity();

        float s, factor1, factor2;
        for (int i = 1; i < len; i++) {
            factor1 = (i - v[k]) * w2;
            factor2 = i + v[k];
            s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);

            while (k > 0 && s <= ranges[k]) {
                k--;
                factor1 = (i - v[k]) * w2;
                factor2 = i + v[k];
                s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);
            }

            k++;
            v[k] = i;
            ranges[k] = s;
            ranges[k + 1] = std::numeric_limits<float>::infinity();
        }

        // Output pass - use specialized loops to avoid per-iteration conditionals
        k = 0;
        if (left_border && right_border) {
            // Fast path: both borders - always apply envelope (like v1)
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                const float result = w2 * sq(i - v[k]) + ff[v[k]];
                const float envelope = std::fminf(w2 * sq(i + 1), w2 * sq(len - i));
                f[(start + i) * stride] = std::fminf(envelope, result);
            }
        } else if (left_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                const float result = w2 * sq(i - v[k]) + ff[v[k]];
                f[(start + i) * stride] = std::fminf(w2 * sq(i + 1), result);
            }
        } else if (right_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                const float result = w2 * sq(i - v[k]) + ff[v[k]];
                f[(start + i) * stride] = std::fminf(w2 * sq(len - i), result);
            }
        } else {
            // No borders - just parabolic result
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = w2 * sq(i - v[k]) + ff[v[k]];
            }
        }
    };

    // Scan graph to find segments - V1-style single loop
    // Key insight: segment boundary when prev didn't connect forward (!(prev & axis_bit))
    // Background has graph=0, so axis_bit check handles both cases
    if (n <= 0) return;

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
            const bool left_border = (black_border || seg_start > 0);
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
    const int seg_len = n - seg_start;
    const bool left_border = (black_border || seg_start > 0);
    if (seg_len <= SMALL_THRESHOLD) {
        process_small_run(seg_start, seg_len, left_border, black_border);
    } else {
        process_large_run(seg_start, seg_len, left_border, black_border);
    }
}

/*
 * Parabolic envelope EDT for Pass 1+
 * Also identical to ND's implementation.
 */
inline void squared_edt_1d_parabolic_ws(
    float* f,
    const int n,
    const long int stride,  // long int like ND v1
    const float anisotropy,
    const bool black_border_left,
    const bool black_border_right,
    int* v,
    float* ff,
    float* ranges
) {
    if (n == 0) return;

    const float w2 = anisotropy * anisotropy;
    int k = 0;

    for (int i = 0; i < n; i++) {
        ff[i] = f[i * stride];
    }

    ranges[0] = -std::numeric_limits<float>::infinity();
    ranges[1] = std::numeric_limits<float>::infinity();

    float s, factor1, factor2;
    for (int i = 1; i < n; i++) {
        factor1 = (i - v[k]) * w2;
        factor2 = i + v[k];
        s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);

        while (k > 0 && s <= ranges[k]) {
            k--;
            factor1 = (i - v[k]) * w2;
            factor2 = i + v[k];
            s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);
        }

        k++;
        v[k] = i;
        ranges[k] = s;
        ranges[k + 1] = std::numeric_limits<float>::infinity();
    }

    k = 0;
    for (int i = 0; i < n; i++) {
        while (ranges[k + 1] < i) {
            k++;
        }

        f[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];

        // Apply boundary envelope
        if (black_border_left && black_border_right) {
            const float envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
            f[i * stride] = std::fminf(envelope, f[i * stride]);
        } else if (black_border_left) {
            f[i * stride] = std::fminf(w2 * sq(i + 1), f[i * stride]);
        } else if (black_border_right) {
            f[i * stride] = std::fminf(w2 * sq(n - i), f[i * stride]);
        }
    }
}

// Match ND v1's exact multi-seg parabolic implementation with SMALL_THRESHOLD optimization
template <typename T>
inline void squared_edt_1d_parabolic_multi_seg_ws_generic(
    const T* segids,
    float* f,
    const int n,
    const long int stride,  // long int like ND v1
    const float anisotropy,
    const bool black_border,
    int* v,
    float* ff,
    float* ranges
) {
    constexpr int SMALL_THRESHOLD = 8;
    const float anis_sq = anisotropy * anisotropy;

    // Fast path for small segments: O(n²) brute force is faster than parabolic envelope
    auto process_small_run = [&](long int start, int len, bool left_border, bool right_border) {
        float original[SMALL_THRESHOLD];
        for (int q = 0; q < len; ++q) {
            original[q] = f[(start + q) * stride];
        }
        for (int j = 0; j < len; ++j) {
            float best = original[j];
            if (left_border) {
                const float cap_left = anis_sq * static_cast<float>((j + 1) * (j + 1));
                if (cap_left < best) best = cap_left;
            }
            if (right_border) {
                const float cap_right = anis_sq * static_cast<float>((len - j) * (len - j));
                if (cap_right < best) best = cap_right;
            }
            for (int q = 0; q < len; ++q) {
                const float delta = static_cast<float>(j - q);
                const float candidate = original[q] + anis_sq * delta * delta;
                if (candidate < best) best = candidate;
            }
            f[(start + j) * stride] = best;
        }
    };

    T working_segid = segids[0];
    int last = 0;

    for (int i = 1; i < n; i++) {
        const T segid = segids[i * stride];
        if (segid != working_segid) {
            if (working_segid != 0) {
                const int run_len = i - last;
                const bool left_border = (black_border || last > 0);
                const bool right_border = true;  // always true at segment transition
                if (run_len <= SMALL_THRESHOLD) {
                    process_small_run(last, run_len, left_border, right_border);
                } else {
                    squared_edt_1d_parabolic_ws(
                        f + last * stride,
                        run_len, stride, anisotropy,
                        left_border, right_border,
                        v, ff, ranges
                    );
                }
            }
            working_segid = segid;
            last = i;
        }
    }

    // Handle last segment
    if (working_segid != 0 && last < n) {
        const int run_len = n - last;
        const bool left_border = (black_border || last > 0);
        const bool right_border = black_border;  // only at image boundary
        if (run_len <= SMALL_THRESHOLD) {
            process_small_run(last, run_len, left_border, right_border);
        } else {
            squared_edt_1d_parabolic_ws(
                f + last * stride,
                run_len, stride, anisotropy,
                left_border, right_border,
                v, ff, ranges
            );
        }
    }
}

// Non-templated version for backward compatibility
inline void squared_edt_1d_parabolic_multi_seg_ws(
    const uint32_t* segids,
    float* f,
    const int n,
    const long int stride,  // long int like ND v1
    const float anisotropy,
    const bool black_border,
    int* v,
    float* ff,
    float* ranges
) {
    squared_edt_1d_parabolic_multi_seg_ws_generic<uint32_t>(
        segids, f, n, stride, anisotropy, black_border, v, ff, ranges);
}

#undef sq

//-----------------------------------------------------------------------------
// Parallel EDT Passes (Optimized with shared pool, templated dispatch)
//-----------------------------------------------------------------------------

template <typename T>
inline void edt_pass0_parallel_generic(
    const T* seg_labels,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Fast path for 2D - direct iteration like ND v1
    if (dims == 2) {
        const size_t other_axis = (axis == 0) ? 1 : 0;
        const size_t other_n = shape[other_axis];
        const long int other_stride = static_cast<long int>(strides[other_axis]);

        const size_t threads = compute_threads(parallel, other_n, n);

        if (threads <= 1) {
            size_t offset = 0;
            for (size_t i = 0; i < other_n; i++) {
                squared_edt_1d_multi_seg_generic<T>(
                    seg_labels + offset, output + offset, n, stride_ax, anisotropy, black_border
                );
                offset += other_stride;
            }
            return;
        }

        ThreadPool& pool = shared_pool_for(threads);
        const size_t per_thread = (other_n + threads - 1) / threads;
        std::vector<std::future<void>> pending;
        pending.reserve(threads);

        for (size_t t = 0; t < threads; t++) {
            const size_t begin = t * per_thread;
            if (begin >= other_n) break;
            const size_t end = std::min(other_n, begin + per_thread);

            // Direct loop - explicit captures for optimal performance
            pending.push_back(pool.enqueue([seg_labels, output, n, stride_ax, other_stride, anisotropy, black_border, begin, end]() {
                size_t offset = begin * other_stride;
                for (size_t i = begin; i < end; i++) {
                    squared_edt_1d_multi_seg_generic<T>(
                        seg_labels + offset, output + offset, n, stride_ax, anisotropy, black_border
                    );
                    offset += other_stride;
                }
            }));
        }
        for (auto& f : pending) f.get();
        return;
    }

    // Generic fallback for 3D+
    size_t total_lines = 1;
    for (size_t d = 0; d < dims; d++) {
        if (d != axis) total_lines *= shape[d];
    }

    std::vector<size_t> bases(total_lines);
    size_t idx = 0;
    std::vector<size_t> coords(dims, 0);
    for (size_t line = 0; line < total_lines; line++) {
        size_t base = 0;
        for (size_t d = 0; d < dims; d++) {
            if (d != axis) base += coords[d] * strides[d];
        }
        bases[idx++] = base;
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            coords[d]++;
            if (coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }

    const size_t threads = compute_threads(parallel, total_lines, n);
    if (threads <= 1) {
        for (size_t i = 0; i < total_lines; i++) {
            squared_edt_1d_multi_seg_generic<T>(
                seg_labels + bases[i], output + bases[i], n, stride_ax, anisotropy, black_border
            );
        }
        return;
    }

    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (total_lines + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= total_lines) break;
        const size_t end = std::min(total_lines, begin + per_thread);
        pending.push_back(pool.enqueue([=, &bases]() {
            for (size_t i = begin; i < end; i++) {
                squared_edt_1d_multi_seg_generic<T>(
                    seg_labels + bases[i], output + bases[i], n, stride_ax, anisotropy, black_border
                );
            }
        }));
    }
    for (auto& f : pending) f.get();
}

template <typename T>
inline void edt_pass_parabolic_parallel_generic(
    const T* seg_labels,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Fast path for 2D - direct iteration like ND v1
    if (dims == 2) {
        const size_t other_axis = (axis == 0) ? 1 : 0;
        const size_t other_n = shape[other_axis];
        const long int other_stride = static_cast<long int>(strides[other_axis]);

        const size_t threads = compute_threads(parallel, other_n, n);
        if (threads <= 1) {
            std::vector<int> v(n);
            std::vector<float> ff(n);
            std::vector<float> ranges(n + 1);
            size_t offset = 0;
            for (size_t i = 0; i < other_n; i++) {
                squared_edt_1d_parabolic_multi_seg_ws_generic<T>(
                    seg_labels + offset, output + offset, n, stride_ax, anisotropy, black_border,
                    v.data(), ff.data(), ranges.data()
                );
                offset += other_stride;
            }
            return;
        }

        ThreadPool& pool = shared_pool_for(threads);
        const size_t per_thread = (other_n + threads - 1) / threads;
        std::vector<std::future<void>> pending;
        pending.reserve(threads);

        for (size_t t = 0; t < threads; t++) {
            const size_t begin = t * per_thread;
            if (begin >= other_n) break;
            const size_t end = std::min(other_n, begin + per_thread);

            // Direct loop - workspace allocated per-thread for cache locality
            pending.push_back(pool.enqueue([seg_labels, output, n, stride_ax, other_stride, anisotropy, black_border, begin, end]() {
                std::vector<int> v(n);
                std::vector<float> ff(n);
                std::vector<float> ranges(n + 1);

                size_t offset = begin * other_stride;
                for (size_t i = begin; i < end; i++) {
                    squared_edt_1d_parabolic_multi_seg_ws_generic<T>(
                        seg_labels + offset, output + offset, n, stride_ax, anisotropy, black_border,
                        v.data(), ff.data(), ranges.data()
                    );
                    offset += other_stride;
                }
            }));
        }
        for (auto& f : pending) f.get();
        return;
    }

    // Generic fallback for 3D+
    size_t total_lines = 1;
    for (size_t d = 0; d < dims; d++) {
        if (d != axis) total_lines *= shape[d];
    }

    std::vector<size_t> bases(total_lines);
    size_t idx = 0;
    std::vector<size_t> coords(dims, 0);
    for (size_t line = 0; line < total_lines; line++) {
        size_t base = 0;
        for (size_t d = 0; d < dims; d++) {
            if (d != axis) base += coords[d] * strides[d];
        }
        bases[idx++] = base;
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            coords[d]++;
            if (coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }

    const size_t threads = compute_threads(parallel, total_lines, n);
    if (threads <= 1) {
        std::vector<int> v(n);
        std::vector<float> ff(n);
        std::vector<float> ranges(n + 1);
        for (size_t i = 0; i < total_lines; i++) {
            squared_edt_1d_parabolic_multi_seg_ws_generic<T>(
                seg_labels + bases[i], output + bases[i], n, stride_ax, anisotropy, black_border,
                v.data(), ff.data(), ranges.data()
            );
        }
        return;
    }

    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (total_lines + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= total_lines) break;
        const size_t end = std::min(total_lines, begin + per_thread);
        pending.push_back(pool.enqueue([=, &bases]() {
            std::vector<int> v(n);
            std::vector<float> ff(n);
            std::vector<float> ranges(n + 1);
            for (size_t i = begin; i < end; i++) {
                squared_edt_1d_parabolic_multi_seg_ws_generic<T>(
                    seg_labels + bases[i], output + bases[i], n, stride_ax, anisotropy, black_border,
                    v.data(), ff.data(), ranges.data()
                );
            }
        }));
    }
    for (auto& f : pending) f.get();
}

// Fused Pass 0 + Segment Label Building (parallel version)
template <typename T>
inline void edt_pass0_build_seglabels_parallel(
    const T* labels,
    float* output,
    uint32_t* seg_labels,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Fast path for 2D - direct iteration like ND v1
    if (dims == 2) {
        const size_t other_axis = (axis == 0) ? 1 : 0;
        const size_t other_n = shape[other_axis];
        const long int other_stride = static_cast<long int>(strides[other_axis]);

        const size_t threads = compute_threads(parallel, other_n, n);
        if (threads <= 1) {
            size_t offset = 0;
            for (size_t i = 0; i < other_n; i++) {
                squared_edt_1d_multi_seg_build_seglabels<T>(
                    labels + offset, output + offset, seg_labels + offset,
                    n, stride_ax, anisotropy, black_border
                );
                offset += other_stride;
            }
            return;
        }

        ThreadPool& pool = shared_pool_for(threads);
        const size_t per_thread = (other_n + threads - 1) / threads;
        std::vector<std::future<void>> pending;
        pending.reserve(threads);

        for (size_t t = 0; t < threads; t++) {
            const size_t begin = t * per_thread;
            if (begin >= other_n) break;
            const size_t end = std::min(other_n, begin + per_thread);
            pending.push_back(pool.enqueue([=]() {
                size_t offset = begin * other_stride;
                for (size_t i = begin; i < end; i++) {
                    squared_edt_1d_multi_seg_build_seglabels<T>(
                        labels + offset, output + offset, seg_labels + offset,
                        n, stride_ax, anisotropy, black_border
                    );
                    offset += other_stride;
                }
            }));
        }
        for (auto& f : pending) f.get();
        return;
    }

    // Generic fallback for 3D+
    size_t total_lines = 1;
    for (size_t d = 0; d < dims; d++) {
        if (d != axis) total_lines *= shape[d];
    }

    std::vector<size_t> bases(total_lines);
    size_t idx = 0;
    std::vector<size_t> coords(dims, 0);
    for (size_t line = 0; line < total_lines; line++) {
        size_t base = 0;
        for (size_t d = 0; d < dims; d++) {
            if (d != axis) base += coords[d] * strides[d];
        }
        bases[idx++] = base;
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            coords[d]++;
            if (coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }

    const size_t threads = compute_threads(parallel, total_lines, n);
    if (threads <= 1) {
        for (size_t i = 0; i < total_lines; i++) {
            squared_edt_1d_multi_seg_build_seglabels<T>(
                labels + bases[i], output + bases[i], seg_labels + bases[i],
                n, stride_ax, anisotropy, black_border
            );
        }
        return;
    }

    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (total_lines + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= total_lines) break;
        const size_t end = std::min(total_lines, begin + per_thread);
        pending.push_back(pool.enqueue([=, &bases]() {
            for (size_t i = begin; i < end; i++) {
                squared_edt_1d_multi_seg_build_seglabels<T>(
                    labels + bases[i], output + bases[i], seg_labels + bases[i],
                    n, stride_ax, anisotropy, black_border
                );
            }
        }));
    }
    for (auto& f : pending) f.get();
}

// Non-templated wrappers for backward compatibility
inline void edt_pass0_parallel(
    const uint32_t* seg_labels,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    edt_pass0_parallel_generic<uint32_t>(
        seg_labels, output, shape, strides, dims, axis, anisotropy, black_border, parallel
    );
}

inline void edt_pass_parabolic_parallel(
    const uint32_t* seg_labels,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    edt_pass_parabolic_parallel_generic<uint32_t>(
        seg_labels, output, shape, strides, dims, axis, anisotropy, black_border, parallel
    );
}

//-----------------------------------------------------------------------------
// FUSED Parabolic Pass from Graph (parallel dispatch)
//-----------------------------------------------------------------------------

template <typename GRAPH_T>
inline void edt_pass_parabolic_from_graph_fused_parallel(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    const int n = static_cast<int>(shape[axis]);
    const long int stride_ax = static_cast<long int>(strides[axis]);
    if (n == 0) return;

    // Unified ND path - parallelize over first other dimension like ND v1
    size_t num_other_dims = 0;
    size_t other_extents[8];  // Max 8D support
    size_t other_strides[8];
    size_t total_lines = 1;
    for (size_t d = 0; d < dims && num_other_dims < 8; d++) {
        if (d != axis) {
            other_extents[num_other_dims] = shape[d];
            other_strides[num_other_dims] = strides[d];
            total_lines *= shape[d];
            num_other_dims++;
        }
    }

    // Compute rest_product (product of all dims except first)
    const size_t first_extent = (num_other_dims > 0) ? other_extents[0] : 1;
    const size_t first_stride = (num_other_dims > 0) ? other_strides[0] : 0;
    size_t rest_product = 1;
    for (size_t d = 1; d < num_other_dims; d++) {
        rest_product *= other_extents[d];
    }

    const size_t threads = compute_threads(parallel, total_lines, n);

    if (threads <= 1) {
        std::vector<int> v(n);
        std::vector<float> ff(n);
        std::vector<float> ranges(n + 1);
        // Single-threaded: iterate all
        for (size_t i0 = 0; i0 < first_extent; i0++) {
            const size_t offset = i0 * first_stride;
            if (num_other_dims <= 1) {
                squared_edt_1d_parabolic_from_graph_ws<GRAPH_T>(
                    graph + offset, output + offset, n, stride_ax, axis_bit,
                    anisotropy, black_border, v.data(), ff.data(), ranges.data()
                );
            } else {
                size_t coords[8] = {0};
                size_t base = offset;
                for (size_t i = 0; i < rest_product; i++) {
                    squared_edt_1d_parabolic_from_graph_ws<GRAPH_T>(
                        graph + base, output + base, n, stride_ax, axis_bit,
                        anisotropy, black_border, v.data(), ff.data(), ranges.data()
                    );
                    for (size_t d = 1; d < num_other_dims; d++) {
                        coords[d]++;
                        base += other_strides[d];
                        if (coords[d] < other_extents[d]) break;
                        base -= coords[d] * other_strides[d];
                        coords[d] = 0;
                    }
                }
            }
        }
        return;
    }

    // Parallel: distribute first dimension like ND v1
    ThreadPool& pool = shared_pool_for(threads);
    const size_t per_thread = (first_extent + threads - 1) / threads;
    std::vector<std::future<void>> pending;
    pending.reserve(threads);

    for (size_t t = 0; t < threads; t++) {
        const size_t begin = t * per_thread;
        if (begin >= first_extent) break;
        const size_t end = std::min(first_extent, begin + per_thread);
        pending.push_back(pool.enqueue([=, &other_extents, &other_strides]() {
            std::vector<int> v(n);
            std::vector<float> ff(n);
            std::vector<float> ranges(n + 1);
            for (size_t i0 = begin; i0 < end; i0++) {
                const size_t offset = i0 * first_stride;
                if (num_other_dims <= 1) {
                    squared_edt_1d_parabolic_from_graph_ws<GRAPH_T>(
                        graph + offset, output + offset, n, stride_ax, axis_bit,
                        anisotropy, black_border, v.data(), ff.data(), ranges.data()
                    );
                } else {
                    size_t coords[8] = {0};
                    size_t base = offset;
                    for (size_t i = 0; i < rest_product; i++) {
                        squared_edt_1d_parabolic_from_graph_ws<GRAPH_T>(
                            graph + base, output + base, n, stride_ax, axis_bit,
                            anisotropy, black_border, v.data(), ff.data(), ranges.data()
                        );
                        for (size_t d = 1; d < num_other_dims; d++) {
                            coords[d]++;
                            base += other_strides[d];
                            if (coords[d] < other_extents[d]) break;
                            base -= coords[d] * other_strides[d];
                            coords[d] = 0;
                        }
                    }
                }
            }
        }));
    }
    for (auto& f : pending) f.get();
}

//-----------------------------------------------------------------------------
// Full EDT from Labels (Graph-First Design: Build Segment Labels)
//-----------------------------------------------------------------------------

/*
 * Graph-First Design: Labels MUST be converted to segment labels.
 *
 * Key insight: For labels input, the labels themselves serve as segment labels.
 * Label value = segment ID semantically (same label = same segment).
 *
 * This is the "conversion" - we interpret labels AS segment labels.
 * The EDT algorithm finds segments by comparing label values, which is
 * semantically equivalent to comparing pre-computed segment IDs.
 *
 * This achieves the graph-first architecture (uniform segment-based processing)
 * while avoiding the overhead of explicit segment label allocation/building.
 */
template <typename T>
inline void edtsq_from_labels(
    const T* labels,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // Compute total voxels and strides
    size_t total = 1;
    std::vector<size_t> strides(dims);
    for (size_t d = dims; d-- > 0;) {
        strides[d] = total;
        total *= shape[d];
    }
    if (total == 0) return;

    // Process axes from innermost to outermost (like V1) for cache efficiency
    // Innermost axis has stride=1, so pass 0 should process it first
    bool is_first_pass = true;
    for (size_t axis = dims; axis-- > 0;) {
        if (is_first_pass) {
            // Pass 0: Rosenfeld-Pfaltz using labels as segment IDs
            edt_pass0_parallel_generic<T>(
                labels, output,
                shape, strides.data(), dims, axis,
                anisotropy[axis], black_border, parallel
            );
            is_first_pass = false;
        } else {
            // Pass 1+: Parabolic using labels as segment IDs
            edt_pass_parabolic_parallel_generic<T>(
                labels, output,
                shape, strides.data(), dims, axis,
                anisotropy[axis], black_border, parallel
            );
        }
    }
}

//-----------------------------------------------------------------------------
// Full EDT from Voxel Graph
//-----------------------------------------------------------------------------

template <typename GRAPH_T>
inline void edtsq_from_graph(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // Compute total voxels and strides
    size_t total = 1;
    std::vector<size_t> strides(dims);
    for (size_t d = dims; d-- > 0;) {
        strides[d] = total;
        total *= shape[d];
    }
    if (total == 0) return;

    // Axis bit encoding (same as barrier graph)
    // For 2D: axis 0 -> bit 2, axis 1 -> bit 0
    // For 3D: axis 0 -> bit 4, axis 1 -> bit 2, axis 2 -> bit 0

    // FULLY FUSED: All passes read directly from graph
    // No segment labels allocation needed - saves 64MB for large volumes

    // Process axes from innermost to outermost (like V1) for cache efficiency
    // Innermost axis has stride=1, so pass 0 should process it first
    bool is_first_pass = true;
    for (size_t axis = dims; axis-- > 0;) {
        // Compute axis bit
        const GRAPH_T axis_bit = static_cast<GRAPH_T>(1) << (2 * (dims - 1 - axis));

        if (is_first_pass) {
            // FUSED pass 0: Read graph directly, compute EDT
            edt_pass0_from_graph_direct_parallel<GRAPH_T>(
                graph, output,
                shape, strides.data(), dims, axis, axis_bit,
                anisotropy[axis], black_border, parallel
            );
            is_first_pass = false;
        } else {
            // FUSED parabolic: Read graph directly
            edt_pass_parabolic_from_graph_fused_parallel<GRAPH_T>(
                graph, output,
                shape, strides.data(), dims, axis, axis_bit,
                anisotropy[axis], black_border, parallel
            );
        }
    }
}

//-----------------------------------------------------------------------------
// Build graph from labels - SINGLE-PASS with explicit 2D/3D specializations
//
// For best performance, we inline neighbor checks for common dimensions (2D, 3D).
// Generic ND falls back to a loop-based approach.
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

    int64_t voxels = 1;
    for (size_t d = 0; d < dims; d++) {
        voxels *= static_cast<int64_t>(shape[d]);
    }
    if (voxels == 0) return;

    const int threads = std::max(1, parallel);
    constexpr GRAPH_T FG = 0x80;  // Foreground bit

    //-------------------------------------------------------------------------
    // 1D path: simple linear scan
    //-------------------------------------------------------------------------
    if (dims == 1) {
        const int64_t n = static_cast<int64_t>(shape[0]);
        constexpr GRAPH_T BIT = 0x01;  // axis 0 bit for 1D

        auto process_1d = [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; i++) {
                const T label = labels[i];
                GRAPH_T g = (label != 0) ? FG : 0;
                if (label != 0 && i + 1 < n && labels[i + 1] == label) {
                    g |= BIT;
                }
                graph[i] = g;
            }
        };
        if (threads > 1 && n > 100) {
            ThreadPool& pool = shared_pool_for(threads);
            std::vector<std::future<void>> pending;
            int64_t per_t = (n + threads - 1) / threads;
            for (int t = 0; t < threads; t++) {
                int64_t s = t * per_t, e = std::min(s + per_t, n);
                if (s < e) pending.push_back(pool.enqueue([=]() { process_1d(s, e); }));
            }
            for (auto& f : pending) f.get();
        } else {
            process_1d(0, n);
        }
        return;
    }

    //-------------------------------------------------------------------------
    // 2D path: optimized with constexpr bits and background chunk skipping
    //-------------------------------------------------------------------------
    if (dims == 2) {
        const int64_t nrows = static_cast<int64_t>(shape[0]);
        const int64_t ncols = static_cast<int64_t>(shape[1]);
        constexpr GRAPH_T DOWN = 0x04;   // axis 0 bit for 2D
        constexpr GRAPH_T RIGHT = 0x01;  // axis 1 bit for 2D
        constexpr int64_t CHUNK = 8;     // Process 8 pixels at a time for background skip

        auto process_2d = [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; r++) {
                const T* rl = labels + r * ncols;
                GRAPH_T* rg = graph + r * ncols;
                const bool can_down = (r + 1 < nrows);
                const T* rl_next = can_down ? (rl + ncols) : nullptr;

                // Chunk-based processing with background skip
                int64_t c = 0;
                const int64_t ncols_chunked = ncols - (ncols % CHUNK);
                for (; c < ncols_chunked; c += CHUNK) {
                    // Quick check: OR all labels - if zero, all background
                    T any_fg = rl[c] | rl[c+1] | rl[c+2] | rl[c+3] |
                               rl[c+4] | rl[c+5] | rl[c+6] | rl[c+7];
                    if (any_fg == 0) {
                        // All background - zero the graph chunk
                        std::memset(rg + c, 0, CHUNK * sizeof(GRAPH_T));
                    } else {
                        // At least one foreground - process individually
                        for (int64_t i = 0; i < CHUNK; i++) {
                            const int64_t ci = c + i;
                            const T label = rl[ci];
                            GRAPH_T g = (label != 0) ? FG : 0;
                            if (label != 0) {
                                if (ci + 1 < ncols && rl[ci + 1] == label) g |= RIGHT;
                                if (can_down && rl_next[ci] == label) g |= DOWN;
                            }
                            rg[ci] = g;
                        }
                    }
                }
                // Handle remaining columns
                for (; c < ncols; c++) {
                    const T label = rl[c];
                    GRAPH_T g = (label != 0) ? FG : 0;
                    if (label != 0) {
                        if (c + 1 < ncols && rl[c + 1] == label) g |= RIGHT;
                        if (can_down && rl_next[c] == label) g |= DOWN;
                    }
                    rg[c] = g;
                }
            }
        };
        if (threads > 1 && nrows > 100) {
            ThreadPool& pool = shared_pool_for(threads);
            std::vector<std::future<void>> pending;
            int64_t per_t = (nrows + threads - 1) / threads;
            for (int t = 0; t < threads; t++) {
                int64_t s = t * per_t, e = std::min(s + per_t, nrows);
                if (s < e) pending.push_back(pool.enqueue([=]() { process_2d(s, e); }));
            }
            for (auto& f : pending) f.get();
        } else {
            process_2d(0, nrows);
        }
        return;
    }

    //-------------------------------------------------------------------------
    // 3D path: optimized with constexpr bits and background chunk skipping
    //-------------------------------------------------------------------------
    if (dims == 3) {
        const int64_t nz = static_cast<int64_t>(shape[0]);
        const int64_t ny = static_cast<int64_t>(shape[1]);
        const int64_t nx = static_cast<int64_t>(shape[2]);
        const int64_t stride_z = ny * nx;
        const int64_t stride_y = nx;
        constexpr GRAPH_T BACK = 0x10;   // axis 0 bit (z)
        constexpr GRAPH_T DOWN = 0x04;   // axis 1 bit (y)
        constexpr GRAPH_T RIGHT = 0x01;  // axis 2 bit (x)
        constexpr int64_t CHUNK = 8;     // Process 8 voxels at a time for background skip

        auto process_3d = [&](int64_t z0, int64_t z1) {
            for (int64_t z = z0; z < z1; z++) {
                const bool can_back = (z + 1 < nz);
                for (int64_t y = 0; y < ny; y++) {
                    const bool can_down = (y + 1 < ny);
                    const int64_t base = z * stride_z + y * stride_y;
                    const T* row = labels + base;
                    GRAPH_T* rowg = graph + base;

                    // Chunk-based processing with background skip
                    int64_t x = 0;
                    const int64_t nx_chunked = nx - (nx % CHUNK);
                    for (; x < nx_chunked; x += CHUNK) {
                        // Quick check: OR all labels - if zero, all background
                        T any_fg = row[x] | row[x+1] | row[x+2] | row[x+3] |
                                   row[x+4] | row[x+5] | row[x+6] | row[x+7];
                        if (any_fg == 0) {
                            std::memset(rowg + x, 0, CHUNK * sizeof(GRAPH_T));
                        } else {
                            for (int64_t i = 0; i < CHUNK; i++) {
                                const int64_t xi = x + i;
                                const T label = row[xi];
                                GRAPH_T g = (label != 0) ? FG : 0;
                                if (label != 0) {
                                    if (xi + 1 < nx && row[xi + 1] == label) g |= RIGHT;
                                    if (can_down && labels[base + stride_y + xi] == label) g |= DOWN;
                                    if (can_back && labels[base + stride_z + xi] == label) g |= BACK;
                                }
                                rowg[xi] = g;
                            }
                        }
                    }
                    // Handle remaining voxels
                    for (; x < nx; x++) {
                        const T label = row[x];
                        GRAPH_T g = (label != 0) ? FG : 0;
                        if (label != 0) {
                            if (x + 1 < nx && row[x + 1] == label) g |= RIGHT;
                            if (can_down && labels[base + stride_y + x] == label) g |= DOWN;
                            if (can_back && labels[base + stride_z + x] == label) g |= BACK;
                        }
                        rowg[x] = g;
                    }
                }
            }
        };
        if (threads > 1 && nz > 4) {
            ThreadPool& pool = shared_pool_for(threads);
            std::vector<std::future<void>> pending;
            int64_t per_t = (nz + threads - 1) / threads;
            for (int t = 0; t < threads; t++) {
                int64_t s = t * per_t, e = std::min(s + per_t, nz);
                if (s < e) pending.push_back(pool.enqueue([=]() { process_3d(s, e); }));
            }
            for (auto& f : pending) f.get();
        } else {
            process_3d(0, nz);
        }
        return;
    }

    //-------------------------------------------------------------------------
    // Unified ND path for 4D+ - parallelize over first dimension
    //-------------------------------------------------------------------------
    int64_t strides[8];
    int64_t shape64[8];
    GRAPH_T axis_bits[8];
    {
        int64_t s = 1;
        for (size_t d = dims; d-- > 0;) {
            strides[d] = s;
            shape64[d] = static_cast<int64_t>(shape[d]);
            s *= shape64[d];
        }
        for (size_t d = 0; d < dims; d++) {
            axis_bits[d] = static_cast<GRAPH_T>(1) << (2 * (dims - 1 - d));
        }
    }

    const int64_t first_extent = shape64[0];
    const int64_t first_stride = strides[0];
    const int64_t last_extent = shape64[dims - 1];
    const GRAPH_T last_bit = axis_bits[dims - 1];
    const GRAPH_T first_bit = axis_bits[0];

    // Middle dimensions product (dims 1 to dims-2) - for 4D+
    int64_t mid_product = 1;
    for (size_t d = 1; d + 1 < dims; d++) {
        mid_product *= shape64[d];
    }

    // Number of middle dimensions (dims between first and last) - for 4D+
    const size_t num_mid = dims - 2;

    // Process range of first dimension (outer loop) for 3D+
    auto process_dim0_range = [&](int64_t d0_start, int64_t d0_end) {
        // Thread-local storage for precomputed middle dimension info
        const T* mid_neighbor_row[6];  // Neighbor row pointers for middle dims
        bool mid_can_check[6];         // Whether we can check each mid neighbor
        GRAPH_T mid_bits[6];           // Bit to set for each mid dimension

        for (int64_t d0 = d0_start; d0 < d0_end; d0++) {
            const int64_t base0 = d0 * first_stride;
            const bool can_d0 = (d0 + 1 < first_extent);

            // Iterate middle dimensions (dims 1 to dims-2)
            int64_t mid_coords[7] = {0};  // For dims 1..dims-2
            int64_t mid_offset = 0;

            for (int64_t m = 0; m < mid_product; m++) {
                const int64_t base = base0 + mid_offset;

                // Precompute row pointers for tight inner loop
                const T* row = labels + base;
                GRAPH_T* rowg = graph + base;
                const T* row_d0_next = can_d0 ? (labels + base + first_stride) : nullptr;

                // Precompute middle dimension neighbor info BEFORE inner loop
                for (size_t md = 0; md < num_mid; md++) {
                    const size_t d = md + 1;  // Actual dimension index
                    mid_can_check[md] = (mid_coords[md] + 1 < shape64[d]);
                    mid_neighbor_row[md] = mid_can_check[md] ? (labels + base + strides[d]) : nullptr;
                    mid_bits[md] = axis_bits[d];
                }

                // Tight inner loop over last dimension (stride=1)
                for (int64_t x = 0; x < last_extent; x++) {
                    const T label = row[x];
                    GRAPH_T g = (label != 0) ? FG : 0;

                    if (label != 0) {
                        // Last dim neighbor (simple pointer access)
                        if (x + 1 < last_extent && row[x + 1] == label) {
                            g |= last_bit;
                        }
                        // First dim neighbor (precomputed pointer)
                        if (can_d0 && row_d0_next[x] == label) {
                            g |= first_bit;
                        }
                        // Middle dimensions - use precomputed pointers
                        for (size_t md = 0; md < num_mid; md++) {
                            if (mid_can_check[md] && mid_neighbor_row[md][x] == label) {
                                g |= mid_bits[md];
                            }
                        }
                    }
                    rowg[x] = g;
                }

                // Increment mid coords (only if dims > 2)
                if (dims > 2) {
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

    if (threads > 1 && first_extent > 4) {
        ThreadPool& pool = shared_pool_for(threads);
        std::vector<std::future<void>> pending;
        int64_t per_t = (first_extent + threads - 1) / threads;
        for (int t = 0; t < threads; t++) {
            int64_t s = t * per_t, e = std::min(s + per_t, first_extent);
            if (s < e) pending.push_back(pool.enqueue([=, &process_dim0_range]() {
                process_dim0_range(s, e);
            }));
        }
        for (auto& f : pending) f.get();
    } else {
        process_dim0_range(0, first_extent);
    }
}

//-----------------------------------------------------------------------------
// Fused labels-to-EDT: Build graph internally, run EDT, free graph
// This is more efficient than separate Python calls because:
// 1. No Python/Cython overhead between build and EDT
// 2. Graph memory is allocated and freed in C++ (faster)
// 3. Thread pool is already warm from graph build
//-----------------------------------------------------------------------------

template <typename T>
inline void edtsq_from_labels_fused(
    const T* labels,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // Compute total voxels
    size_t total = 1;
    for (size_t d = 0; d < dims; d++) {
        total *= shape[d];
    }
    if (total == 0) return;

    // Graph bit encoding: axis d uses bit (1 << (2*(dims-1-d)))
    // Max bit for axis 0 = 1 << (2*(dims-1))
    // uint8 supports dims <= 4 (max bit = 64)
    // uint16 supports dims <= 8 (max bit = 16384)
    if (dims <= 4) {
        std::unique_ptr<uint8_t[]> graph(new uint8_t[total]);
        build_connectivity_graph<T, uint8_t>(labels, graph.get(), shape, dims, parallel);
        edtsq_from_graph<uint8_t>(graph.get(), output, shape, anisotropy, dims, black_border, parallel);
    } else {
        // 5D-8D: need uint16 for graph bits
        std::unique_ptr<uint16_t[]> graph(new uint16_t[total]);
        build_connectivity_graph<T, uint16_t>(labels, graph.get(), shape, dims, parallel);
        edtsq_from_graph<uint16_t>(graph.get(), output, shape, anisotropy, dims, black_border, parallel);
    }
}

//-----------------------------------------------------------------------------
// Parabolic EDT with argmin tracking (for expand_labels/feature_transform)
//-----------------------------------------------------------------------------

inline float sq_f(float x) { return x * x; }

inline void squared_edt_1d_parabolic_with_arg_stride(
    float* f,
    const long int n,
    const long int stride,
    const float anisotropy,
    const bool black_border_left,
    const bool black_border_right,
    int* arg_out,
    const long int arg_stride
) {
    if (n == 0) return;
    const double w2 = (double)anisotropy * anisotropy;

    int k = 0;
    std::unique_ptr<int[]> v(new int[n]());
    std::unique_ptr<double[]> ff(new double[n]());
    for (long int i = 0; i < n; i++) ff[i] = f[i * stride];
    std::unique_ptr<double[]> ranges(new double[n + 1]());
    ranges[0] = -std::numeric_limits<double>::infinity();
    ranges[1] = std::numeric_limits<double>::infinity();

    double s, factor1, factor2;
    for (long int i = 1; i < n; i++) {
        factor1 = (i - v[k]) * w2;
        factor2 = i + v[k];
        s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
        while (k > 0 && s <= ranges[k]) {
            k--;
            factor1 = (i - v[k]) * w2;
            factor2 = i + v[k];
            s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
        }
        k++;
        v[k] = i;
        ranges[k] = s;
        ranges[k + 1] = std::numeric_limits<double>::infinity();
    }

    k = 0;
    double envelope;
    for (long int i = 0; i < n; i++) {
        while (ranges[k + 1] < i) k++;
        f[i * stride] = (float)(w2 * (i - v[k]) * (i - v[k]) + ff[v[k]]);
        arg_out[i * arg_stride] = v[k];
        if (black_border_left && black_border_right) {
            envelope = std::fmin(w2 * (i + 1) * (i + 1), w2 * (n - i) * (n - i));
            f[i * stride] = (float)std::fmin(envelope, (double)f[i * stride]);
        } else if (black_border_left) {
            f[i * stride] = (float)std::fmin(w2 * (i + 1) * (i + 1), (double)f[i * stride]);
        } else if (black_border_right) {
            f[i * stride] = (float)std::fmin(w2 * (n - i) * (n - i), (double)f[i * stride]);
        }
    }
}

//-----------------------------------------------------------------------------
// Expand labels helpers (for ND expand_labels/feature_transform)
//-----------------------------------------------------------------------------

template <typename INDEX=size_t>
inline void _nd_expand_init_bases(
    const uint8_t* seeds,
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    INDEX* feat_out,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);
    ThreadPool pool(threads);
    size_t chunks = std::max<size_t>(1, std::min<size_t>(num_lines, (size_t)threads));
    const size_t chunk = (num_lines + chunks - 1) / chunks;
    for (size_t start = 0; start < num_lines; start += chunk) {
        const size_t end = std::min(num_lines, start + chunk);
        pool.enqueue([=]() {
            std::vector<int> arg(n);
            for (size_t i = start; i < end; ++i) {
                const size_t base = bases[i];
                bool any_zero = false;
                for (size_t j = 0; j < n; ++j) {
                    const bool seeded = (seeds[base + j * s] != 0);
                    dist[base + j * s] = seeded ? 0.0f : (std::numeric_limits<float>::max() / 4.0f);
                    any_zero |= (!seeded);
                }
                if (any_zero) {
                    squared_edt_1d_parabolic_with_arg_stride(
                        dist + base, (int)n, (long)s, anis,
                        black_border, black_border,
                        arg.data(), 1);
                } else {
                    for (size_t j = 0; j < n; ++j) arg[j] = (int)j;
                }
                for (size_t j = 0; j < n; ++j) {
                    feat_out[base + j * s] = (INDEX)(base + (size_t)arg[j] * s);
                }
            }
        });
    }
    pool.join();
}

template <typename INDEX=size_t>
inline void _nd_expand_parabolic_bases(
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    INDEX* feat,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);

    if (threads <= 1 || num_lines == 1) {
        std::vector<int> arg(n);
        std::vector<INDEX> feat_line(n);
        for (size_t i = 0; i < num_lines; ++i) {
            const size_t base = bases[i];
            bool any_nonzero = false;
            for (size_t j = 0; j < n; ++j) {
                const float val = dist[base + j * s];
                any_nonzero |= (val != 0.0f);
                feat_line[j] = feat[base + j * s];
            }
            if (any_nonzero) {
                squared_edt_1d_parabolic_with_arg_stride(
                    dist + base, (int)n, (long)s, anis,
                    black_border, black_border,
                    arg.data(), 1);
            } else {
                for (size_t j = 0; j < n; ++j) arg[j] = (int)j;
            }
            for (size_t j = 0; j < n; ++j) {
                feat[base + j * s] = feat_line[(size_t)arg[j]];
            }
        }
        return;
    }

    ThreadPool pool(threads);
    size_t chunks = std::max<size_t>(1, std::min<size_t>(num_lines, (size_t)threads * ND_CHUNKS_PER_THREAD));
    const size_t chunk = (num_lines + chunks - 1) / chunks;
    for (size_t start = 0; start < num_lines; start += chunk) {
        const size_t end = std::min(num_lines, start + chunk);
        pool.enqueue([=]() {
            std::vector<int> arg(n);
            std::vector<INDEX> feat_line(n);
            for (size_t i = start; i < end; ++i) {
                const size_t base = bases[i];
                bool any_nonzero = false;
                for (size_t j = 0; j < n; ++j) {
                    const float val = dist[base + j * s];
                    any_nonzero |= (val != 0.0f);
                    feat_line[j] = feat[base + j * s];
                }
                if (any_nonzero) {
                    squared_edt_1d_parabolic_with_arg_stride(
                        dist + base, (int)n, (long)s, anis,
                        black_border, black_border,
                        arg.data(), 1);
                } else {
                    for (size_t j = 0; j < n; ++j) arg[j] = (int)j;
                }
                for (size_t j = 0; j < n; ++j) {
                    feat[base + j * s] = feat_line[(size_t)arg[j]];
                }
            }
        });
    }
    pool.join();
}

inline void _nd_expand_init_labels_bases(
    const uint8_t* seeds,
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    const uint32_t* labelsp,
    uint32_t* label_out,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);
    ThreadPool pool(threads);
    size_t chunks = std::max<size_t>(1, std::min<size_t>(num_lines, (size_t)threads));
    const size_t chunk = (num_lines + chunks - 1) / chunks;
    for (size_t start = 0; start < num_lines; start += chunk) {
        const size_t end = std::min(num_lines, start + chunk);
        pool.enqueue([=]() {
            std::vector<int> arg(n);
            for (size_t i = start; i < end; ++i) {
                const size_t base = bases[i];
                bool any_nonseed = false;
                for (size_t j = 0; j < n; ++j) {
                    const bool seeded = (seeds[base + j * s] != 0);
                    dist[base + j * s] = seeded ? 0.0f : (std::numeric_limits<float>::max() / 4.0f);
                    any_nonseed |= (!seeded);
                }
                if (any_nonseed) {
                    squared_edt_1d_parabolic_with_arg_stride(
                        dist + base, (int)n, (long)s, anis,
                        black_border, black_border,
                        arg.data(), 1);
                } else {
                    for (size_t j = 0; j < n; ++j) arg[j] = (int)j;
                }
                for (size_t j = 0; j < n; ++j) {
                    label_out[base + j * s] = labelsp[base + (size_t)arg[j] * s];
                }
            }
        });
    }
    pool.join();
}

inline void _nd_expand_parabolic_labels_bases(
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    const uint32_t* label_in,
    uint32_t* label_out,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);
    ThreadPool pool(threads);
    size_t chunks = std::max<size_t>(1, std::min<size_t>(num_lines, (size_t)threads));
    const size_t chunk = (num_lines + chunks - 1) / chunks;
    for (size_t start = 0; start < num_lines; start += chunk) {
        const size_t end = std::min(num_lines, start + chunk);
        pool.enqueue([=]() {
            std::vector<int> arg(n);
            for (size_t i = start; i < end; ++i) {
                const size_t base = bases[i];
                bool any_nonzero = false;
                for (size_t j = 0; j < n; ++j) any_nonzero |= (dist[base + j * s] != 0.0f);
                if (any_nonzero) {
                    squared_edt_1d_parabolic_with_arg_stride(
                        dist + base, (int)n, (long)s, anis,
                        black_border, black_border,
                        arg.data(), 1);
                } else {
                    for (size_t j = 0; j < n; ++j) arg[j] = (int)j;
                }
                for (size_t j = 0; j < n; ++j) {
                    label_out[base + j * s] = label_in[base + (size_t)arg[j] * s];
                }
            }
        });
    }
    pool.join();
}

} // namespace nd

#endif // EDT_HPP
